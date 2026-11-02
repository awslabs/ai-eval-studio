# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared AWS retry decorator and classifiers.

Covers the boto3 (ClientError) classifier preserved from the original
``BedrockRuntimeModel`` retry block, the httpx classifier required by the
mantle adapter, and the parameterized ``_aws_retry``
decorator factory that both classifiers consume.

The load-bearing assertion in this file is
``test_mantle_classifier_non_retryable_validation_400`` — the
``validation_error`` 400 from the bedrock-mantle endpoint family is the
capability-probe self-correction signal. Retrying it would mask schema
drift and wrong-API mismatches;
this test pins the contract.

The boto3 classifier path is also covered indirectly by
``test_bedrock_runtime_model_retry.py``, which exercises
``BedrockRuntimeModel.generate_with_system`` end-to-end after the
``_bedrock_retry`` decorator was extracted into ``aws_retry``. That test
file MUST stay green — this file complements it by covering
the extracted module's public surface directly, including the
mantle classifier and the decorator factory parameterization.
"""

from __future__ import annotations

import httpx
import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
)

from app.services.aws_retry import (
    _RETRYABLE_BEDROCK_MARKERS,
    _RETRYABLE_HTTPX_EXCEPTIONS,
    _RETRYABLE_NETWORK_EXCEPTIONS,
    _aws_retry,
    _bedrock_retry,
    _is_retryable_bedrock_error,
    _is_retryable_mantle_error,
    _mantle_retry,
)


# ---------------------------------------------------------------------------
# Boto3 classifier (preserved from original BedrockRuntimeModel retry block)
# ---------------------------------------------------------------------------


def test_bedrock_classifier_retryable_throttling():
    """A ThrottlingException ClientError is retryable — covers the
    primary symptom that caused the extraction to ship in the first
    place (DeepEval's ignore_errors swallow on ThrottlingException
    storms during eval).
    """
    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded for model",
            }
        },
        operation_name="Converse",
    )
    assert _is_retryable_bedrock_error(exc) is True


def test_bedrock_classifier_non_retryable_validation():
    """ValidationException is non-retryable — backing off won't fix the
    request shape; surface immediately so the caller sees the real
    error instead of a slow retry-then-fail.
    """
    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "temperature is not supported for this model",
            }
        },
        operation_name="Converse",
    )
    assert _is_retryable_bedrock_error(exc) is False


def test_bedrock_classifier_retryable_network_exception():
    """A botocore network-level failure is retryable even though it
    isn't a ClientError. Matches one of ``_RETRYABLE_NETWORK_EXCEPTIONS``.
    """
    exc = EndpointConnectionError(
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/"
    )
    assert _is_retryable_bedrock_error(exc) is True
    # Sanity-check the constant is non-empty (tests that import it
    # directly — like the mantle adapter's parity test —
    # would silently drift if this list shrank to nothing).
    assert len(_RETRYABLE_NETWORK_EXCEPTIONS) > 0
    assert len(_RETRYABLE_BEDROCK_MARKERS) > 0


# ---------------------------------------------------------------------------
# Mantle classifier (httpx path)
# ---------------------------------------------------------------------------


def test_mantle_classifier_retryable_httpx_connect_error():
    """An httpx connect-level failure is retryable — symmetric to the
    boto3 ``EndpointConnectionError`` case for the raw HTTPS path.
    """
    assert _is_retryable_mantle_error(httpx.ConnectError("conn refused")) is True
    assert len(_RETRYABLE_HTTPX_EXCEPTIONS) > 0


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_mantle_classifier_retryable_5xx_and_429(status_code):
    """Transient server-side conditions on the mantle endpoint are
    retryable: 429 (throttling) and 5xx (gateway errors). 400/401/403
    are NOT in this set.
    """
    exc = httpx.HTTPStatusError(
        message=str(status_code),
        request=httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/"),
        response=httpx.Response(status_code),
    )
    assert _is_retryable_mantle_error(exc) is True


def test_mantle_classifier_non_retryable_validation_400():
    """A 400 with ``error.code == "validation_error"`` is the
    capability-probe self-correction signal that the discovery layer
    (``bedrock_discovery``) consumes to re-resolve which API a model
    supports. Retrying would mask schema drift and wrong-API mismatches.
    This test is the load-bearing assertion that pins the contract.
    """
    body = {
        "error": {
            "code": "validation_error",
            "message": (
                "The model 'openai.gpt-5.5' does not support "
                "the '/v1/chat/completions' API"
            ),
        }
    }
    exc = httpx.HTTPStatusError(
        message="400",
        request=httpx.Request(
            "POST", "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
        ),
        response=httpx.Response(400, json=body),
    )
    assert _is_retryable_mantle_error(exc) is False


def test_mantle_classifier_non_retryable_400_without_validation_code():
    """A 400 that isn't carrying ``code == "validation_error"`` is also
    non-retryable. The classifier defaults to "do not retry" on any
    400; it doesn't try to be clever about other error codes.
    """
    body = {"error": {"code": "unknown", "message": "x"}}
    exc = httpx.HTTPStatusError(
        message="400",
        request=httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/"),
        response=httpx.Response(400, json=body),
    )
    assert _is_retryable_mantle_error(exc) is False


def test_mantle_classifier_400_with_non_json_body_is_non_retryable():
    """Defensive: a 400 whose body isn't JSON must NOT raise from the
    classifier — it falls through to "non-retryable". A masked decode
    error here would defeat the whole retry policy.
    """
    exc = httpx.HTTPStatusError(
        message="400",
        request=httpx.Request("POST", "https://bedrock-mantle.us-east-1.api.aws/"),
        response=httpx.Response(400, content=b"not-json"),
    )
    assert _is_retryable_mantle_error(exc) is False


def test_mantle_classifier_unknown_exception_returns_false():
    """Unrelated exceptions don't trigger retry. The classifier is
    closed over httpx network errors + ``HTTPStatusError``; everything
    else is non-retryable so backoff doesn't mask caller bugs.
    """
    assert _is_retryable_mantle_error(ValueError("x")) is False


# ---------------------------------------------------------------------------
# _aws_retry decorator factory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Shrink retry timing so tests don't sleep real wall-time.

    The decorator reads the ``BEDROCK_RETRY_*`` env knobs on every
    invocation, so monkeypatching at fixture-scope keeps the values
    pinned regardless of which test is running.
    """
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.0")
    monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.0")


def test_aws_retry_decorator_succeeds_after_one_retry():
    """``_aws_retry`` retries on a retryable error and returns the
    second-call result. Uses the mantle classifier so we exercise the
    httpx path through the decorator (the boto3 path is covered by
    test_bedrock_runtime_model_retry.py end-to-end).
    """
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient")
        return "ok"

    wrapped = _aws_retry(fn, classifier=_is_retryable_mantle_error)
    assert wrapped() == "ok"
    assert calls["n"] == 2


def test_aws_retry_decorator_gives_up_after_max_attempts():
    """After ``BEDROCK_RETRY_MAX_ATTEMPTS`` attempts the decorator
    re-raises the underlying exception. The fixture sets max=3.
    """
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise httpx.ConnectError("always fails")

    wrapped = _aws_retry(fn, classifier=_is_retryable_mantle_error)
    with pytest.raises(httpx.ConnectError):
        wrapped()
    assert calls["n"] == 3


def test_aws_retry_decorator_does_not_retry_non_retryable():
    """A non-retryable exception is re-raised on the first attempt;
    the decorator doesn't sleep and doesn't loop.
    """
    calls = {"n": 0}
    non_retryable = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "bad request",
            }
        },
        operation_name="Converse",
    )

    def fn():
        calls["n"] += 1
        raise non_retryable

    # Default classifier (boto3) — ValidationException is non-retryable.
    wrapped = _aws_retry(fn)
    with pytest.raises(ClientError):
        wrapped()
    assert calls["n"] == 1


def test_aws_retry_default_classifier_is_bedrock():
    """Calling ``_aws_retry`` without a classifier kwarg defaults to
    the boto3 classifier (``_is_retryable_bedrock_error``). The
    ``_bedrock_retry`` alias and the bare ``_aws_retry`` call must
    behave identically on the boto3 path.
    """
    throttle = ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "rate exceeded",
            }
        },
        operation_name="Converse",
    )

    calls = {"a": 0, "b": 0}

    def fn_default():
        calls["a"] += 1
        if calls["a"] == 1:
            raise throttle
        return "ok"

    def fn_alias():
        calls["b"] += 1
        if calls["b"] == 1:
            raise throttle
        return "ok"

    assert _aws_retry(fn_default)() == "ok"
    assert _bedrock_retry(fn_alias)() == "ok"
    assert calls["a"] == calls["b"] == 2


def test_aws_retry_preserves_function_metadata():
    """``wrapper.__name__`` and ``wrapper.__doc__`` mirror the wrapped
    callable so tooling (logs, tracebacks, sphinx) keeps the right
    attribution after extraction.
    """

    def my_specific_name():
        """A doc that should survive wrapping."""
        return 1

    wrapped = _aws_retry(my_specific_name)
    assert wrapped.__name__ == "my_specific_name"
    assert wrapped.__doc__ == "A doc that should survive wrapping."


def test_mantle_retry_alias_uses_mantle_classifier():
    """The ``_mantle_retry`` alias exists for callsite ergonomics
    parity with ``_bedrock_retry``. Verifying it routes httpx errors
    through the mantle classifier (an httpx.ConnectError retries; a
    ValidationException ClientError does NOT).
    """
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient")
        return "ok"

    assert _mantle_retry(fn)() == "ok"
    assert calls["n"] == 2

    # And: a non-httpx error that the mantle classifier doesn't know
    # about is non-retryable.
    boom = ClientError(
        error_response={"Error": {"Code": "X", "Message": "y"}},
        operation_name="Z",
    )
    other = {"n": 0}

    def fn2():
        other["n"] += 1
        raise boom

    with pytest.raises(ClientError):
        _mantle_retry(fn2)()
    assert other["n"] == 1


def test_bedrock_classifier_5xx_message_marker_retryable():
    """ServiceUnavailable / InternalServerError / ModelTimeout markers
    are retryable on the boto3 path. This duplicates a check in
    test_bedrock_runtime_model_retry.py at the classifier level so a future
    refactor that drops a marker is caught at unit-of-classifier
    granularity, not only at the BedrockRuntimeModel call layer.
    """
    for code, msg in [
        ("ServiceUnavailableException", "Service unavailable"),
        ("InternalServerException", "internal server error"),
        ("ModelTimeoutException", "model timeout"),
    ]:
        exc = ClientError(
            error_response={"Error": {"Code": code, "Message": msg}},
            operation_name="Converse",
        )
        assert _is_retryable_bedrock_error(exc) is True, (
            f"{code} should still be retryable"
        )


def test_bedrock_classifier_connect_timeout_retryable():
    """A ``ConnectTimeoutError`` is in ``_RETRYABLE_NETWORK_EXCEPTIONS``
    and must classify as retryable.
    """
    exc = ConnectTimeoutError(endpoint_url="https://x")
    assert _is_retryable_bedrock_error(exc) is True


# ---------------------------------------------------------------------------
# is_temperature_capability_reject (broad list-free rule)
# ---------------------------------------------------------------------------


def test_is_temperature_capability_reject_deprecated():
    """The regression: Opus 4.8 rejects temperature with 'deprecated'.
    This was not in the old phrase list, causing silent 4% failures.
    """
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "temperature is deprecated for this model",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is True


def test_is_temperature_capability_reject_not_supported():
    """Legacy phrasing 'is not supported' must still be caught."""
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Malformed input request: temperature is not supported.",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is True


def test_is_temperature_capability_reject_does_not_support():
    """Legacy phrasing 'does not support' must still be caught."""
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "temperature does not support this model",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is True


def test_is_temperature_capability_reject_must_be_between():
    """Out-of-range error (caller bug) must propagate, not latch the gate."""
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "temperature must be between 0 and 1",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is False


def test_is_temperature_capability_reject_value_must_be_in_range():
    """Another range phrasing — value errors must propagate."""
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "temperature: value must be in range [0, 2]",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is False


def test_is_temperature_capability_reject_non_temperature_validation():
    """A ValidationException that doesn't mention 'temperature' is not a capability reject."""
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "messages: required field is missing",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is False


def test_is_temperature_capability_reject_temperature_in_user_prompt():
    """A ValidationException that echoes 'temperature' from user content
    (not a verdict about the parameter) must propagate, not latch.
    """
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Input message too long: 'What is the temperature in Paris?'",
            }
        },
        operation_name="Converse",
    )
    assert is_temperature_capability_reject(exc) is False


def test_is_temperature_capability_reject_non_client_error():
    """A non-ClientError exception (even if the message mentions temperature)
    is not a capability reject.
    """
    from app.services.aws_retry import is_temperature_capability_reject

    exc = ValueError("temperature")
    assert is_temperature_capability_reject(exc) is False


# ---------------------------------------------------------------------------
# __wrapped__ exposure
# ---------------------------------------------------------------------------
#
# Discovery probes call the underlying generate function ONCE per
# probe attempt — the 4-attempt retry loop here would inflate a 3-API
# probe sweep to 12 wall-clock attempts (and stretch the per-id worst
# case far past the 60s probe budget). Exposing
# ``wrapper.__wrapped__ = fn`` lets the discovery code call the
# unwrapped function via ``fn.__wrapped__(...)`` while every existing
# call site continues to go through the retry loop.
#
# The four behaviors below are the contract the capability probe relies on.


def test_aws_retry_exposes_wrapped_attribute():
    """The wrapper carries a ``__wrapped__`` reference to the original
    callable so the discovery probe can bypass the retry loop in 20s
    flat.
    """

    def my_fn(x):
        return x

    wrapped = _aws_retry(my_fn)
    assert wrapped.__wrapped__ is my_fn


def test_bedrock_retry_alias_exposes_wrapped():
    """Same contract on the boto3-classifier alias — both aliases route
    through ``_aws_retry``, so the exposure must transitively appear on
    every wrapper this module produces.
    """

    def my_fn(x):
        return x

    assert _bedrock_retry(my_fn).__wrapped__ is my_fn


def test_mantle_retry_alias_exposes_wrapped():
    """Same contract on the httpx-classifier alias used by
    ``BedrockMantleModel.generate_with_system``. The capability probe in
    ``bedrock_discovery`` calls
    ``BedrockMantleModel.generate_with_system.__wrapped__`` to bypass
    the 4-attempt retry — the alias MUST carry the attribute.
    """

    def my_fn(x):
        return x

    assert _mantle_retry(my_fn).__wrapped__ is my_fn


def test_aws_retry_wrapped_bypasses_retry_loop(monkeypatch):
    """Calling ``wrapper.__wrapped__(...)`` directly skips the retry
    loop entirely — verifiable by mocking ``time.sleep`` and asserting
    it is never called even when the wrapped function always raises a
    retryable error. The wrapped surface (``wrapper(*args)``) DOES
    retry; the bypass is the bypass.
    """
    sleeps: list[float] = []

    # Capture every call to time.sleep at the module level the
    # decorator imports it from. The decorator wraps the time call
    # ``time.sleep(wait)`` in its loop; intercepting at
    # ``app.services.aws_retry.time.sleep`` is the load-bearing edit.
    import app.services.aws_retry as aws_retry_module

    monkeypatch.setattr(
        aws_retry_module.time, "sleep", lambda s: sleeps.append(s)
    )

    def always_fails():
        raise httpx.ReadTimeout("transient")

    wrapped = _aws_retry(always_fails, classifier=_is_retryable_mantle_error)

    # 1. Bypass: __wrapped__ is the raw callable; one call → one raise,
    #    no sleeps captured.
    with pytest.raises(httpx.ReadTimeout):
        wrapped.__wrapped__()
    assert sleeps == [], (
        "calling __wrapped__ MUST bypass the retry loop entirely; "
        "time.sleep was called %d times" % len(sleeps)
    )

    # 2. Wrapped surface: the same callable through ``wrapped(...)``
    #    DOES retry — sleeps captured (BEDROCK_RETRY_MAX_ATTEMPTS=3 in
    #    the autouse fixture, so 2 sleeps between the 3 attempts).
    with pytest.raises(httpx.ReadTimeout):
        wrapped()
    assert len(sleeps) >= 1, (
        "wrapped surface must still retry — sleeps captured: %r" % sleeps
    )
