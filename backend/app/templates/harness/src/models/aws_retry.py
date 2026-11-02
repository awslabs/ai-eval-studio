# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared retry decorator for AWS-fronted Bedrock surfaces (runtime + mantle).

Single retry vocabulary across ``BedrockRuntimeModel`` (boto3 Converse) and
``BedrockMantleModel`` (raw HTTPS + SigV4). Same env knobs
(``BEDROCK_RETRY_MAX_ATTEMPTS``, ``BEDROCK_RETRY_INITIAL_S``,
``BEDROCK_RETRY_CAP_S``), same exponential-backoff-plus-25%-jitter math.
Extends the boto3-only classifier with httpx transient errors + HTTP
429/5xx for the mantle path. The ``validation_error`` 400 from mantle is
explicitly non-retryable — it is the capability-probe self-correction
signal consumed by the discovery layer.

Public surface:
- ``_aws_retry(fn, *, classifier)`` — parameterized retry-decorator factory.
- ``_bedrock_retry(fn)`` — alias bound to the boto3 classifier.
- ``_mantle_retry(fn)`` — alias bound to the httpx classifier.
- ``_is_retryable_bedrock_error(exc)`` — boto3 ClientError + network classifier.
- ``_is_retryable_mantle_error(exc)`` — httpx + HTTP-status classifier.

Constants ``_RETRYABLE_BEDROCK_MARKERS``, ``_RETRYABLE_NETWORK_EXCEPTIONS``,
``_RETRYABLE_HTTPX_EXCEPTIONS`` are exported for tests that introspect
the policy directly.

This file is duplicated byte for byte between ``backend/app/services/``
and ``backend/app/templates/harness/src/models/`` so behavior inside the
platform matches behavior inside a scaffolded harness project.
``tests/test_templates_parity.py`` enforces the match, so edit both copies
together.
"""

import logging
import os
import random
import time

import httpx
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)


logger = logging.getLogger(__name__)


# Substring markers (lowercase) for retryable Bedrock client errors.
# Mirrors DeepEval's ``BEDROCK_MESSAGE_MARKERS`` retryable subset
# (``deepeval/models/retry_policy.py``). DeepEval surfaces the AWS error
# code only via the message body (botocore's ClientError doesn't expose
# a typed enum), so we substring-match the same way.
_RETRYABLE_BEDROCK_MARKERS: tuple[str, ...] = (
    "throttlingexception",
    "too many requests",
    "rate exceeded",
    "serviceunavailableexception",
    "service unavailable",
    "internalserverexception",
    "internal server error",
    "modeltimeoutexception",
    "model timeout",
)

# Botocore-level transient network failures. Always retryable — they
# represent a connection issue, not a server-side decision.
_RETRYABLE_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
    ConnectionClosedError,
)

# httpx-level transient network failures for the mantle (raw HTTPS +
# SigV4) path. These are the symmetric set to ``_RETRYABLE_NETWORK_EXCEPTIONS``
# for code that uses the httpx transport instead of botocore.
_RETRYABLE_HTTPX_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

# Value/range error exclusion markers for temperature capability rejection.
# A ValidationException whose message mentions ``temperature`` is a capability
# rejection (drop the parameter, retry once, cache the learned flag) EXCEPT
# when it is an out-of-range / value error. Value errors are caller-config
# bugs that must propagate so the caller sees the misconfiguration instead of
# silent-dropout. This small exclusion set gates the
# broad rule: if the message contains any of these markers, it is a value
# error (return False); otherwise, a temperature-mentioning ValidationException
# is a capability rejection (return True).
_TEMPERATURE_VALUE_ERROR_MARKERS: tuple[str, ...] = (
    "must be between",
    "must be in range",
    "out of range",
    "value must be",
    "valid range",
    "<=",
    ">=",
)


def is_temperature_capability_reject(exc: BaseException) -> bool:
    """Return True if this exception signals a temperature capability rejection.

    Broad, list-free rule: a ValidationException whose lowercased message
    contains the literal token ``temperature`` is a capability rejection
    (the model does not support the parameter) EXCEPT when it is a value/
    range error (numeric bounds language indicating a caller-config bug).

    Capability rejection (return True):
      * "temperature is deprecated for this model" (Opus 4.8 regression)
      * "temperature is not supported"
      * "temperature does not support this model"
      * Any other ValidationException mentioning ``temperature`` that is NOT
        a value error or an echo-only message.

    Value/range error (return False — must propagate as caller bug):
      * "temperature must be between 0 and 1"
      * "temperature: value must be in range [0, 2]"

    Echo-only / unrelated (return False — not a verdict about the parameter):
      * "Input message too long: 'What is the temperature in Paris?'"
        (echoes user content; is a length/shape error, not a param verdict)
      * "messages: required field is missing" (no temperature mention)

    Polarity note: This is an INCLUSION-by-default rule (opposite of the old
    phrase list). Every temperature-mentioning ValidationException is treated
    as a capability rejection unless it contains value-error language or is
    a generic length/shape error that merely quotes user content. The small
    ``_TEMPERATURE_VALUE_ERROR_MARKERS`` exclusion set keeps the rule
    maintainable without enumerating every possible rejection phrasing.

    Non-ClientError exceptions return False (shape guard — only ValidationException
    from boto3 can carry a capability rejection).

    Args:
        exc: The exception to classify.

    Returns:
        True if the exception is a temperature capability rejection (drop
        parameter, retry once, cache the flag). False otherwise (propagate
        the exception as-is).
    """
    # Stage 1: shape guard. Only boto3 ClientError carrying a ValidationException
    # can signal a capability rejection.
    if not isinstance(exc, ClientError):
        return False
    msg_lower = str(exc).lower()
    if "validationexception" not in msg_lower and "validation error" not in msg_lower:
        return False

    # Stage 2: token + exclusions. The message must mention "temperature" AND
    # must NOT be a value/range error or a generic length/shape error that
    # merely echoes user content.
    if "temperature" not in msg_lower:
        return False

    # Exclusion 1: value/range errors (caller-config bugs that must propagate).
    if any(marker in msg_lower for marker in _TEMPERATURE_VALUE_ERROR_MARKERS):
        return False

    # Exclusion 2: generic length/shape errors that quote user content are
    # not verdicts about the temperature parameter. Detect by checking for
    # length/shape language combined with quoting markers.
    if ("too long" in msg_lower or "message length" in msg_lower) and (
        "'" in msg_lower or '"' in msg_lower
    ):
        return False

    # Everything else that mentions temperature in a ValidationException is
    # a capability rejection.
    return True


def _is_retryable_bedrock_error(exc: BaseException) -> bool:
    """Return True if this exception should trigger a Bedrock retry.

    Retryable: ClientError whose message matches a known transient marker
    (throttling / 5xx / model timeout), or a botocore network exception.
    Non-retryable: AccessDeniedException, ValidationException,
    ResourceNotFoundException — surfacing those immediately gives the
    caller a real error instead of a slow retry-then-fail.
    """
    if isinstance(exc, _RETRYABLE_NETWORK_EXCEPTIONS):
        return True
    if isinstance(exc, ClientError):
        msg = str(exc).lower()
        return any(marker in msg for marker in _RETRYABLE_BEDROCK_MARKERS)
    return False


def _is_retryable_mantle_error(exc: BaseException) -> bool:
    """Return True if this exception should trigger a Bedrock Mantle retry.

    Retryable:
      * httpx connect/read/timeout/protocol/pool-timeout errors (transient
        network) — see ``_RETRYABLE_HTTPX_EXCEPTIONS``.
      * ``httpx.HTTPStatusError`` whose status code is 429, 500, 502, 503,
        or 504 (transient server-side conditions on the mantle endpoint).

    Non-retryable (explicit):
      * ``MantleAuthorizationFailure`` — 401 that persisted after a
        successful credential refresh. The credential is valid and freshly
        minted, but the principal is not authorized; backing off cannot help.
      * 400 with ``error.code == "validation_error"`` — this is the
        capability-probe self-correction signal that the discovery layer
        consumes. Retrying would
        mask schema drift / wrong-API mismatches; surface immediately.
      * Any other 4xx — caller bug or auth failure; backing off won't fix.
      * Anything that isn't an httpx network error or HTTPStatusError.
    """
    # Explicit non-retryable: MantleAuthorizationFailure (post-refresh 401).
    # Check by class name to avoid circular import (bedrock_mantle_model
    # imports aws_retry, so aws_retry cannot import bedrock_mantle_model).
    if type(exc).__name__ == "MantleAuthorizationFailure":
        return False

    if isinstance(exc, _RETRYABLE_HTTPX_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 500, 502, 503, 504):
            return True
        if status == 400:
            # Defensive: the body might not be JSON. Any failure to
            # decode the validation_error signal falls through to
            # "non-retryable" rather than masking an unexpected shape.
            try:
                body = exc.response.json()
            except (ValueError, TypeError):
                return False
            try:
                code = body.get("error", {}).get("code")
            except AttributeError:
                return False
            if code == "validation_error":
                return False
        return False
    return False


def _aws_retry(fn, *, classifier=None):
    """Decorator factory: retry an AWS-fronted call on transient failures.

    Mirrors DeepEval's ``@retry_bedrock`` semantics for our custom
    BedrockRuntimeModel and BedrockMantleModel. Default of 4 attempts
    (waits of 1s, 2s and 4s between them, each capped at 8s, plus up to
    25% jitter) is higher than DeepEval's 2 because the
    harness has no upper-layer retry between this and the metric scorer
    — runner-level retry runs only after ``evaluate()`` returns, which
    is too late to repair a null score on the parallel pass.

    Tunable via env (override applied per-call so tests can shorten):
      BEDROCK_RETRY_MAX_ATTEMPTS  int   (default 4, >=1)
      BEDROCK_RETRY_INITIAL_S     float (default 1.0)
      BEDROCK_RETRY_CAP_S         float (default 8.0)

    Args:
        fn: The callable to wrap.
        classifier: A function ``(exc) -> bool`` that returns True if the
            exception is retryable. Defaults to
            ``_is_retryable_bedrock_error`` (boto3 path). Pass
            ``_is_retryable_mantle_error`` for the httpx path.
    """
    if classifier is None:
        classifier = _is_retryable_bedrock_error

    def wrapper(*args, **kwargs):
        max_attempts = max(1, int(os.environ.get("BEDROCK_RETRY_MAX_ATTEMPTS", "4")))
        initial = float(os.environ.get("BEDROCK_RETRY_INITIAL_S", "1.0"))
        cap = float(os.environ.get("BEDROCK_RETRY_CAP_S", "8.0"))

        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — re-raised below if not retryable
                attempt += 1
                if attempt >= max_attempts or not classifier(e):
                    raise
                # Exponential backoff capped at ``cap``, plus 25% jitter
                # so concurrent retriers don't synchronize and stampede
                # the same throttled endpoint.
                wait = min(cap, initial * (2 ** (attempt - 1)))
                wait += random.uniform(0, wait * 0.25)
                logger.warning(
                    "[bedrock] retry %d/%d after %.1fs (%s)",
                    attempt, max_attempts, wait, type(e).__name__,
                )
                time.sleep(wait)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


def _bedrock_retry(fn):
    """Retry decorator bound to the boto3 (ClientError) classifier.

    Convenience alias preserving the existing callsite ergonomics from
    the pre-extraction ``BedrockRuntimeModel.generate_with_system`` decorator.
    """
    return _aws_retry(fn, classifier=_is_retryable_bedrock_error)


def _mantle_retry(fn):
    """Retry decorator bound to the httpx (HTTPStatusError) classifier.

    Used by ``BedrockMantleModel`` for raw HTTPS + SigV4 calls to the
    ``bedrock-mantle.{region}.api.aws`` endpoint family.
    """
    return _aws_retry(fn, classifier=_is_retryable_mantle_error)
