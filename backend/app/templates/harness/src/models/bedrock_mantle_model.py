# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepEval-compatible LLM wrapper for AWS Bedrock Mantle (raw HTTPS + SigV4).

Mirror of ``bedrock_runtime_model.py`` for the Bedrock Mantle endpoint family.
Models served there (from providers such as OpenAI, xAI, Anthropic, Mistral
and Qwen) are called by signing SigV4 with service ``bedrock-mantle`` against
the model's region (NOT the coordinator's) and POSTing raw JSON over HTTPS
to ``https://bedrock-mantle.{region}.api.aws/{path}``.

Returns ``(text, latency_ms, usage)`` tuples on
``generate_with_system`` for thread-safety under DeepEval's
``ThreadPoolExecutor(max_workers=5)`` — every per-call value is on the
return, never on ``self``.

Operational guards this adapter carries:

  * Shared module-level ``httpx.Client``.
  * Single-source region in ``_sign_and_send`` — signer's region_name
    and the URL host's region segment derive from the same descriptor
    access.
  * ``cancel_token: threading.Event`` parameter on ``generate`` /
    ``generate_with_system`` / ``_sign_and_send``.
  * Schema-driven parse (delegated to the ``bedrock_mantle_mappers``
    parser for the descriptor's API) raising ``MantleSchemaDrift`` on
    any field-level mismatch.
  * ``verify=certifi.where()`` on the shared client; cloud-mode boot
    smoke check warms the cert chain.
  * ``MANTLE_*`` env knobs propagated via ``_project_subprocess_env``
    factories (see ``test_subprocess_env_mantle.py``).
  * ``x-amzn-requestid`` captured on every response and logged.
  * ``MantleReasoningTruncated`` named error on
    ``status='incomplete'`` + ``reason='max_output_tokens'``.
  * Reasoning-token-aware ``_accrue_usage`` — never double-counts.
  * Response-driven temperature self-correction signal: descriptor's
    ``supports_temperature`` flag drives whether ``temperature`` is
    sent; rejection caching is the discovery layer's responsibility.
  * ``MantleSignatureMismatch`` carries 403 body verbatim.

This file is duplicated byte for byte between ``backend/app/services/``
and ``backend/app/templates/harness/src/models/``;
``tests/test_templates_parity.py`` enforces the match, so edit both copies
together. Imports use only stdlib + boto3 + httpx + certifi + deepeval —
symbols that resolve in BOTH the service venv and the harness scaffold
venv.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Callable

import boto3
import certifi
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from deepeval.models.base_model import DeepEvalBaseLLM

from .aws_retry import _mantle_retry
from .bedrock_mantle_mappers import (
    MantleAPIName,
    MantleAPIRejected,
    MantleContentSafetyBlocked,
    MantleRequestRejected,
    MantleTemperatureUnsupported,
    MantleReasoningTruncated,
    MantleSchemaDrift,
    _build_anthropic_messages_request,
    _build_chat_completions_request,
    _build_responses_request,
    _parse_anthropic_messages_response,
    _parse_chat_completions_response,
    _parse_responses_response,
)
from .model_descriptors import MantleDescriptor


logger = logging.getLogger(__name__)


# Default region for the boot-time TLS warm-up. Cross-region descriptors
# warm lazily on the first eval — bursting the smoke check across every
# enabled region is wasteful when mantle-served models concentrate on
# us-east-1 / us-west-2.
_MANTLE_DEFAULT_REGION_FOR_BOOT_CHECK = "us-east-1"


# Module-level singleton httpx.Client. Shared across all
# BedrockMantleModel instances on the same coordinator process.
# Per-call client construction would TLS-handshake every request,
# exhausting file descriptors at the eval concurrency of 20.
_shared_client: httpx.Client | None = None
_shared_client_lock = threading.Lock()


# Module-level credential provider cache. Keyed by profile_name (None for
# the default credential chain — the Fargate container role).
# Cache the refreshable PROVIDER (result of session.get_credentials()),
# NOT the frozen credentials — botocore's provider auto-refreshes near
# expiry only if you keep the provider and call .get_frozen_credentials()
# per request. Caching the frozen value instead would save the per-call
# lookup but go stale after about an hour, when the Fargate task role's
# credentials rotate.
_shared_mantle_credentials: dict[str | None, Any] = {}
_shared_mantle_credentials_lock = threading.Lock()


def _get_shared_mantle_client() -> httpx.Client:
    """Lazily build the module-level singleton httpx.Client.

    Bounded pool sized via env knobs (``MANTLE_MAX_CONNECTIONS`` /
    ``MANTLE_MAX_KEEPALIVE``). ``verify=certifi.where()`` pins the
    explicit Mozilla CA bundle for ``*.api.aws`` instead of the system
    trust store, so the Amazon Trust Services chain for this host family
    validates the same way in the coordinator image and in every harness
    venv.

    Thread-safe via double-checked locking, mirroring the lazy client
    construction in ``BedrockRuntimeModel.load_model``.

    Returns:
        The shared ``httpx.Client``; constructed on first call,
        re-used thereafter.
    """
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                max_connections = int(
                    os.environ.get("MANTLE_MAX_CONNECTIONS", "20")
                )
                max_keepalive = int(
                    os.environ.get("MANTLE_MAX_KEEPALIVE", "10")
                )
                read_timeout = int(
                    os.environ.get("BEDROCK_READ_TIMEOUT_S", "300")
                )
                _shared_client = httpx.Client(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=float(read_timeout),
                        write=30.0,
                        pool=30.0,
                    ),
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_keepalive,
                    ),
                    verify=certifi.where(),
                    http2=False,
                )
    return _shared_client


def _close_shared_mantle_client() -> None:
    """Close the singleton client and clear the slot.

    Test-only — call between tests to reset module state. Production
    callers don't close the client; it lives for the process lifetime.
    """
    global _shared_client
    with _shared_client_lock:
        if _shared_client is not None:
            _shared_client.close()
            _shared_client = None


def _get_shared_mantle_credentials(profile_name: str | None) -> Any:
    """Lazily build and cache the refreshable credential provider.

    Returns the result of ``boto3.Session(profile_name).get_credentials()``
    (a ``botocore.credentials.Credentials`` provider) keyed by profile_name.
    The provider is cached for the process lifetime and its
    ``get_frozen_credentials()`` method is called per request in
    ``_sign_and_send`` so a rotating Fargate task role's new creds are
    picked up near expiry.

    WHY cache the provider and NOT the frozen credentials: botocore's
    provider auto-refreshes near expiry only if you keep the provider
    alive and call ``.get_frozen_credentials()`` per request. Caching the
    frozen value would trade the per-call Session overhead for a
    stale-creds-after-~1h bug when the Fargate task role rotates.

    Thread-safe via double-checked locking, keyed on profile_name. In dev
    different adapter instances may pass different profiles; in
    cloud/Fargate profile_name is None (the container role chain) so all
    instances share one cache entry.

    Args:
        profile_name: AWS profile name for credential resolution (None
            uses the standard credential chain).

    Returns:
        The refreshable credential provider (``Credentials`` instance).

    Raises:
        RuntimeError: ``get_credentials()`` returns None — no credentials
            in the chain. Keep the message text stable: it surfaces
            through ``_sign_and_send``, and callers and tests look for it
            verbatim.
    """
    global _shared_mantle_credentials
    if profile_name not in _shared_mantle_credentials:
        with _shared_mantle_credentials_lock:
            if profile_name not in _shared_mantle_credentials:
                session = boto3.Session(profile_name=profile_name)
                creds_provider = session.get_credentials()
                if creds_provider is None:
                    raise RuntimeError(
                        "No AWS credentials available for Bedrock Mantle SigV4 "
                        "signing — set AWS_PROFILE or run on a Fargate/EC2 task "
                        "with an attached role."
                    )
                _shared_mantle_credentials[profile_name] = creds_provider
    return _shared_mantle_credentials[profile_name]


def _close_shared_mantle_credentials() -> None:
    """Clear the credential provider cache.

    Test-only — call between tests to reset module state. Production
    never clears the cache; providers live for the process lifetime.
    """
    global _shared_mantle_credentials
    with _shared_mantle_credentials_lock:
        _shared_mantle_credentials.clear()


class MantleSignatureMismatch(Exception):
    """403 SignatureDoesNotMatch — region or credential drift.

    Carries ``region_signed`` (from descriptor), ``region_host`` (from
    URL — equal in the success case but the constructor accepts both
    so a future probe can distinguish them) and the verbatim 403 body
    so the operator sees the AWS error code.
    """

    def __init__(self, region_signed: str, region_host: str, body: str):
        super().__init__(
            f"SigV4 signed for {region_signed} but URL host is in "
            f"{region_host}; body: {body}"
        )
        self.region_signed = region_signed
        self.region_host = region_host
        self.body = body


class MantleAuthorizationFailure(Exception):
    """401 Unauthorized that persisted after credential refresh — non-retryable.

    Distinct from the transient expired-credential case: this exception means
    the credential is valid and freshly minted, but the principal is not
    authorized to access the model. Backing off cannot help; the user must
    grant model access or switch to a different model.

    Carries the verbatim response body so the real reason survives into
    ``additional_metadata.error`` and the infra-failure classifier.
    """

    def __init__(self, body: str):
        super().__init__(
            f"Mantle authorization failure (post-refresh 401): {body}"
        )
        self.body = body


class MantleCancelled(Exception):
    """Raised when ``cancel_token`` is set before/after a network call.

    Sync httpx can't honor a mid-flight cancel; the bracketing checks
    in ``_sign_and_send`` are the surface.
    """


class MantleNotInvocable(Exception):
    """Transient or environmental preflight failure — descriptor is NOT invalidated.

    Wraps connect errors, timeouts, signature mismatches, and other
    non-deterministic failures of a 1-token "Say OK" probe. The
    descriptor stays on disk because the failure mode does not imply
    the descriptor is stale; the user/operator fixes the environment
    (creds, network, throttling) and retries.

    Args:
        model_id: The mantle model id whose probe failed.
        reason: Short identifier — typically the wrapped exception's
            class name (e.g. ``"MantleSignatureMismatch"``,
            ``"ConnectError"``).
        detail: Optional verbatim error string from the wrapped
            exception. Empty string when not available.
    """

    def __init__(self, model_id: str, reason: str, detail: str = ""):
        if detail:
            super().__init__(f"{model_id} ({reason}): {detail}")
        else:
            super().__init__(f"{model_id} ({reason})")
        self.model_id = model_id
        self.reason = reason
        self.detail = detail


class ModelRemovedFromMantle(Exception):
    """Mantle 404 — descriptor is stale; dispatch invalidates and re-resolves.

    Raised when the resolved api/region path returns 404 for the
    descriptor's model id. The model has been removed from the mantle
    catalog (or the descriptor's api routing is wrong post-rotation);
    the dispatch caller deletes the descriptor entry and the next
    ``list_bedrock_models`` re-resolution fires fresh.

    Args:
        model_id: The mantle model id that no longer resolves.
    """

    def __init__(self, model_id: str):
        super().__init__(
            f"{model_id} (model removed from mantle — descriptor invalidated)"
        )
        self.model_id = model_id


class BedrockMantleModel(DeepEvalBaseLLM):
    """Thread-safe Bedrock Mantle adapter with the same return contract as BedrockRuntimeModel.

    ``generate_with_system()`` returns ``(text, latency_ms, usage)`` so
    the same instance is safe under DeepEval's
    ``ThreadPoolExecutor(max_workers=5)``. All per-call values are on
    the return; ``self`` carries only the descriptor + cumulative
    usage counters.

    Args:
        descriptor: ``MantleDescriptor`` carrying ``endpoint``, ``api``,
            ``region`` (single source for SigV4 + URL), and optional
            capability flags.
        model_id: The model identifier (e.g. ``"openai.gpt-5.5"``).
        max_output_tokens: Hard cap on completion length.
        temperature: Sampling temperature; only sent when the
            descriptor's ``supports_temperature`` is True.
        profile_name: AWS profile name for SigV4 credentials (None
            uses the standard credential chain — same path as
            ``BedrockRuntimeModel``).
    """

    # Strategy dispatch — picked once in ``__init__`` from
    # ``descriptor.api``. Callers never see the dict.
    _BUILDERS: dict[MantleAPIName, Callable] = {
        "responses": _build_responses_request,
        "chat_completions": _build_chat_completions_request,
        "anthropic_messages": _build_anthropic_messages_request,
    }
    _PARSERS: dict[MantleAPIName, Callable] = {
        "responses": _parse_responses_response,
        "chat_completions": _parse_chat_completions_response,
        "anthropic_messages": _parse_anthropic_messages_response,
    }
    # Map api → URL path. The single place an API name maps to a path —
    # the adapter never builds the path string elsewhere.
    _PATHS: dict[MantleAPIName, str] = {
        "responses": "/openai/v1/responses",
        "chat_completions": "/v1/chat/completions",
        "anthropic_messages": "/anthropic/v1/messages",
    }

    def __init__(
        self,
        descriptor: MantleDescriptor,
        model_id: str,
        *,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        profile_name: str | None = None,
    ):
        self._descriptor = descriptor
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.profile_name = profile_name
        # Resolve mapper + parser + path ONCE from the descriptor.
        api: MantleAPIName = descriptor["api"]  # type: ignore[assignment]
        self._build_request = self._BUILDERS[api]
        self._parse_response = self._PARSERS[api]
        self._path = self._PATHS[api]
        # Judge-cost accounting — mirrors ``BedrockRuntimeModel`` exactly so
        # ``bedrock_pricing.estimate_cost_usd`` consumes the same
        # ``{inputTokens, outputTokens}`` shape downstream.
        self._usage_lock = threading.Lock()
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0

    def _accrue_usage(self, usage: dict | None) -> None:
        """Add one Mantle response's normalized usage to the running total.

        Defensive: a malformed/absent ``usage`` must never raise — same
        contract as ``BedrockRuntimeModel._accrue_usage``. The mapper's parser
        already converted Responses' ``input_tokens``/``output_tokens``
        to ``inputTokens``/``outputTokens``, so this method reads the
        normalized keys.

        Reasoning tokens accrue as part of ``output_tokens`` because
        the Responses contract emits them inside the same total — do
        NOT add ``output_tokens_details.reasoning_tokens`` separately.

        Args:
            usage: Normalized usage dict from the parser, or None.
        """
        if not usage:
            return
        try:
            input_tokens = int(usage.get("inputTokens", 0) or 0)
            output_tokens = int(usage.get("outputTokens", 0) or 0)
        except (TypeError, ValueError):
            return
        with self._usage_lock:
            self._usage_input_tokens += input_tokens
            self._usage_output_tokens += output_tokens

    @property
    def accumulated_usage(self) -> dict:
        """Return summed usage across every Mantle call so far.

        Returns:
            ``{"inputTokens", "outputTokens"}`` — the shape
            ``bedrock_pricing.estimate_cost_usd`` consumes.
        """
        with self._usage_lock:
            return {
                "inputTokens": self._usage_input_tokens,
                "outputTokens": self._usage_output_tokens,
            }

    def preflight(
        self,
        *,
        timeout_s: float = 20.0,
        cancel_token: threading.Event | None = None,
    ) -> None:
        """Probe the descriptor's api/region with a 1-token "Say OK" call.

        Used by the dispatch layer to catch a stale descriptor BEFORE a
        20-minute eval cycle is wasted on a hung or removed model. The
        probe is one-shot (bypasses ``@_mantle_retry`` via the
        ``__wrapped__`` attribute set in ``aws_retry``) and runs against
        a throwaway adapter so the OWNER instance's
        ``accumulated_usage`` is NOT polluted (mirrors the
        ``bedrock_discovery._probe_capability`` pattern).

        Returns ``None`` on success; raises one of three named
        exceptions on failure so the dispatch caller can classify:

        - ``MantleAPIRejected``: 400 with
          ``error.code == "validation_error"``. Re-raised verbatim;
          dispatch INVALIDATES the descriptor (the api/region pairing
          is wrong; re-resolution must fire).
        - ``ModelRemovedFromMantle``: 404 on the resolved api path.
          Dispatch INVALIDATES the descriptor (the model id is gone).
        - ``MantleNotInvocable``: every other failure — signature
          mismatch (403), connect errors, timeouts, schema drift, etc.
          Dispatch DOES NOT invalidate (transient or environmental —
          descriptor may still be correct).

        Truncation (``MantleReasoningTruncated``) is treated as a
        SUCCESSFUL preflight: it means the api/region routing reached
        the model and got a (cap-limited) response. Reasoning models
        burn the small probe budget on hidden reasoning tokens and
        truncate at any reasonable cap; the production path uses a
        generous user-set cap.

        Args:
            timeout_s: Advisory wall-clock ceiling. The actual
                enforcement lives at the dispatch async call site via
                ``asyncio.wait_for(asyncio.to_thread(model.preflight),
                timeout=20.0)``; the underlying httpx client's
                ``read_timeout`` (from ``BEDROCK_READ_TIMEOUT_S``,
                default 300s) is the in-process upper bound. Documented
                here so callers know the parameter is informational and
                must be enforced externally.
            cancel_token: Optional ``threading.Event`` checked before
                AND after the network call.

        Raises:
            MantleAPIRejected: 400 validation_error → invalidate.
            ModelRemovedFromMantle: 404 → invalidate.
            MantleNotInvocable: Anything else → do not invalidate.
        """
        # Throwaway adapter so the owner's accumulated_usage stays
        # untouched, following the probe-isolation pattern — ``_probe_capability``
        # constructs a dedicated probe instance for the same reason.
        # max_output_tokens=64 — verified live: 1 trips a min-value
        # 400 before the wrong-api signal can surface; 16 is the API
        # min but reasoning models truncate at it; 64 is the smallest
        # cap that lets reasoning models complete the reasoning + emit
        # a token (or truncate cleanly, which preflight treats as OK).
        probe = BedrockMantleModel(
            self._descriptor,
            self.model_id,
            max_output_tokens=64,
            temperature=0.0,
            profile_name=self.profile_name,
        )
        try:
            # ``__wrapped__`` is the undecorated function, set by the
            # ``_aws_retry`` decorator factory in ``aws_retry`` (which
            # ``_mantle_retry`` uses); calling it bypasses the retry loop,
            # so preflight makes exactly one attempt.
            probe.generate_with_system.__wrapped__(
                probe, None, "Say OK.", cancel_token=cancel_token
            )
        except MantleReasoningTruncated:
            # Reasoning model burned the cap on hidden reasoning tokens
            # — but the request reached the model and got a response.
            # That's the entire signal preflight needs (descriptor's
            # api/region/auth all work). Production calls use a generous
            # user-set cap.
            return
        except MantleAPIRejected:
            # Re-raise verbatim so the dispatch caller invalidates the
            # descriptor (api/region routing is wrong).
            raise
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ModelRemovedFromMantle(self.model_id) from e
            # Any other HTTP status — wrap as transient/environmental.
            raise MantleNotInvocable(
                self.model_id, type(e).__name__, str(e)
            ) from e
        except MantleSignatureMismatch as e:
            # 403 — credential or region drift in the environment, NOT
            # a stale descriptor. Wrap so dispatch does not invalidate.
            raise MantleNotInvocable(
                self.model_id, "MantleSignatureMismatch", str(e)
            ) from e
        except Exception as e:  # noqa: BLE001 — last-resort surface
            # Connect errors, timeouts, schema drift, truncation,
            # cancellation — all transient/environmental from the
            # descriptor's perspective. Preserve the original class
            # name so the dispatch error message points at the cause.
            raise MantleNotInvocable(
                self.model_id, type(e).__name__, str(e)
            ) from e

    def load_model(self):
        """Return the module-level shared httpx.Client.

        Provided for DeepEval interface compatibility — unlike
        BedrockRuntimeModel, no per-instance client state is held.
        """
        return _get_shared_mantle_client()

    def get_model_name(self) -> str:
        """Return the Mantle model identifier."""
        return self.model_id

    def _supports_temperature(self) -> bool:
        """Read the descriptor's cached temperature-support flag.

        Defaults True if absent. The temperature self-correction trigger
        (caching False after a 400 mentioning ``temperature``) lives
        in the discovery layer; this adapter just reads what's there.
        """
        return bool(self._descriptor.get("supports_temperature", True))

    def _sign_and_send(
        self,
        body: bytes,
        *,
        cancel_token: threading.Event | None = None,
    ) -> httpx.Response:
        """Sign the body with SigV4 and POST it to the descriptor's region.

        Single-source-region helper. The region is read
        ONCE from ``self._descriptor["region"]``; the signer's
        ``region_name`` and the URL host's region segment are derived
        from the same access — never accept region as a separate kwarg.

        On a 401 response, clears the cached credential provider and
        retries ONCE with a refreshed credential. If the second attempt
        also returns 401, raises ``MantleAuthorizationFailure`` (a
        non-retryable genuine authorization failure, distinct from the
        transient expired-credential case). The retry logic is inline here
        so the ``@_mantle_retry`` attempt budget is never consumed against
        a known-dead credential.

        Args:
            body: Already-serialized JSON request body.
            cancel_token: Optional ``threading.Event`` checked before
                AND after the network call. Sync httpx can't honor
                mid-flight cancel; the bracketing checks are the
                surface.

        Returns:
            The raw ``httpx.Response`` (caller decides how to
            interpret status / body).

        Raises:
            MantleCancelled: ``cancel_token`` is set.
            MantleAuthorizationFailure: 401 persisted after credential
                refresh — not retryable.
            RuntimeError: No AWS credentials in the standard chain
                (originates in ``_get_shared_mantle_credentials`` at
                first-call cache-build time; surfaced through
                ``_sign_and_send``).
        """
        if cancel_token is not None and cancel_token.is_set():
            raise MantleCancelled(
                f"cancelled before request to {self.model_id}"
            )

        region = self._descriptor["region"]
        url = f"https://bedrock-mantle.{region}.api.aws{self._path}"

        # Inline helper to sign and send once
        def _do_request() -> httpx.Response:
            # Resolve the cached refreshable provider once, then re-freeze
            # per request so a rotating Fargate task role's new creds are
            # picked up near expiry.
            creds_provider = _get_shared_mantle_credentials(self.profile_name)
            creds = creds_provider.get_frozen_credentials()
            req = AWSRequest(
                method="POST",
                url=url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            SigV4Auth(creds, "bedrock-mantle", region).add_auth(req)

            client = _get_shared_mantle_client()
            response = client.post(url, content=body, headers=dict(req.headers))

            # Capture x-amzn-requestid on every response — success or
            # error — for CloudWatch correlation.
            request_id = response.headers.get("x-amzn-requestid", "?")
            api_name = self._descriptor["api"]
            logger.info(
                "[mantle] %s %s region=%s api=%s status=%d req_id=%s",
                "POST",
                url,
                region,
                api_name,
                response.status_code,
                request_id,
            )
            return response

        # First attempt
        response = _do_request()

        if cancel_token is not None and cancel_token.is_set():
            raise MantleCancelled(
                f"cancelled after request to {self.model_id}"
            )

        # Credential refresh on 401
        if response.status_code == 401:
            logger.warning(
                "[mantle] 401 Unauthorized from %s — forcing credential refresh and retrying once",
                self.model_id,
            )
            # Clear the cached provider so the next resolution re-runs
            # the credential_process and yields a freshly minted credential.
            _close_shared_mantle_credentials()

            # Second attempt with refreshed credential
            response = _do_request()

            if cancel_token is not None and cancel_token.is_set():
                raise MantleCancelled(
                    f"cancelled after retry to {self.model_id}"
                )

            # If still 401, it's a genuine authorization failure
            if response.status_code == 401:
                raise MantleAuthorizationFailure(response.text)

        return response

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Convert non-2xx responses into named exceptions.

        Surfaces:
          * 403 → ``MantleSignatureMismatch`` carrying the body verbatim
            (never wrap as a generic AccessDenied).
          * 400 with ``error.code == "validation_error"`` →
            ``MantleAPIRejected``. The
            classifier in ``aws_retry._is_retryable_mantle_error``
            also short-circuits this — defense in depth.
          * Any OTHER 400 (fell through the temperature / safety /
            validation_error branches) → ``MantleRequestRejected``
            carrying the provider body message verbatim, so the real
            reason (e.g. a thinking-vs-forced-tool_choice conflict)
            survives into ``additional_metadata.error`` and the
            infra-failure classifier instead of being flattened to a
            bare "400 Bad Request" wrapper. Non-retryable.
          * Any other 4xx/5xx → ``response.raise_for_status()`` so the
            ``@_mantle_retry`` classifier sees an
            ``httpx.HTTPStatusError`` and decides retry-vs-fail (429/5xx
            stay retryable).

        Args:
            response: The httpx.Response from ``_sign_and_send``.
        """
        if response.status_code == 403:
            region = self._descriptor["region"]
            # We can't tell the host's "true" region from a 403 alone —
            # the URL host already encodes it (and the URL is built
            # from ``region``). Pass the same value into both fields;
            # the test asserts they match in the success case.
            raise MantleSignatureMismatch(
                region_signed=region,
                region_host=region,
                body=response.text,
            )
        if response.status_code == 400:
            try:
                body = response.json()
            except Exception:  # noqa: BLE001 — body might not be JSON
                body = {}
            err = body.get("error") if isinstance(body, dict) else None
            code = (
                err.get("code")
                if isinstance(err, dict)
                else None
            )
            err_type = (
                err.get("type")
                if isinstance(err, dict)
                else None
            )
            msg = (
                err.get("message", "(no message)")
                if isinstance(err, dict)
                else "(no message)"
            )
            param = err.get("param") if isinstance(err, dict) else None

            # Temperature self-correction trigger — generalized across all three
            # APIs. Reasoning models reject ``temperature`` with several
            # shapes that all mention the word in the message text.
            # Verified: openai/xai responses use code="unsupported_parameter"
            # + param="temperature"; anthropic uses
            # "temperature is deprecated, use top_p" with no special code.
            # The permissive trigger: any 400 whose lowercase message
            # contains "temperature", OR whose param == "temperature".
            # NEVER hardcode a model list — the live 400 IS the signal.
            msg_lower = msg.lower() if isinstance(msg, str) else ""
            if "temperature" in msg_lower or param == "temperature":
                raise MantleTemperatureUnsupported(
                    f"{self.model_id}: {msg}"
                )

            # Content-safety / policy block — the request reached the
            # model and was declined on policy grounds. Distinct from
            # MantleAPIRejected: the api routing is correct, the
            # descriptor is keepable. Discovery treats this as "reachable
            # → keep". Detected by code/type substring match across
            # provider variations (verified shapes: ``content_filter``,
            # ``content_policy``, ``content_safety``, ``safety_violation``,
            # type ``content_filter_error``).
            code_lower = code.lower() if isinstance(code, str) else ""
            type_lower = err_type.lower() if isinstance(err_type, str) else ""
            _SAFETY_MARKERS = (
                "content_filter",
                "content_policy",
                "content_safety",
                "safety_violation",
            )
            if any(m in code_lower for m in _SAFETY_MARKERS) or any(
                m in type_lower for m in _SAFETY_MARKERS
            ):
                raise MantleContentSafetyBlocked(
                    f"{self.model_id}: {msg}"
                )

            if code == "validation_error":
                raise MantleAPIRejected(f"{self.model_id}: {msg}")

            # Final 400 fall-through. The request reached the endpoint
            # and was rejected for a reason we don't special-case (e.g.
            # the thinking-vs-forced-tool_choice conflict). Surface the
            # provider's own message — fall back to the raw response text
            # when the body had no ``error.message`` (or wasn't the
            # expected JSON shape) so a differently-shaped body still
            # carries its text instead of "(no message)". This is the
            # signal downstream self-diagnosis needs; without it the bare
            # ``raise_for_status()`` below would drop it.
            fall_through_msg = msg if msg != "(no message)" else response.text
            raise MantleRequestRejected(f"{self.model_id}: {fall_through_msg}")
        response.raise_for_status()

    def generate(
        self,
        prompt: str,
        *,
        cancel_token: threading.Event | None = None,
    ) -> str:
        """Generate a response without a system prompt.

        Delegates to ``generate_with_system`` so accrual happens at a
        single point — same shape as ``BedrockRuntimeModel.generate``.

        Args:
            prompt: The user prompt.
            cancel_token: Optional cancellation event.

        Returns:
            The model's text output.
        """
        text, _, _ = self.generate_with_system(
            None, prompt, cancel_token=cancel_token
        )
        return text

    @_mantle_retry
    def generate_with_system(
        self,
        system_prompt: str | None,
        user_prompt: str,
        *,
        cancel_token: threading.Event | None = None,
    ) -> tuple[str, float, dict]:
        """Generate a response with an optional system prompt.

        Builds the request via the descriptor-bound mapper, signs and
        sends it, parses the response via the matching parser, accrues
        usage, and returns ``(text, latency_ms, usage)``.

        Decorated with ``@_mantle_retry`` (httpx classifier from
        ``aws_retry``). ``MantleAPIRejected`` is non-retryable per the
        classifier — surfaces immediately as the discovery layer's
        capability-probe signal.

        Args:
            system_prompt: System-level instructions (None or empty
                → omitted on the wire per the Responses contract).
            user_prompt: User message.
            cancel_token: Optional ``threading.Event`` for cancel.

        Returns:
            ``(text, latency_ms, usage)`` — same shape as
            ``BedrockRuntimeModel.generate_with_system``.
        """
        try:
            return self._invoke_once(
                system_prompt, user_prompt, cancel_token=cancel_token
            )
        except MantleTemperatureUnsupported as exc:
            # Temperature self-correction: the live 400 IS the
            # signal — no hardcoded reasoning-model list. Cache the flag
            # in-memory on the descriptor so subsequent calls in this
            # adapter instance skip ``temperature`` automatically. The
            # discovery layer is responsible for persisting this back to
            # the on-disk descriptor at next harness rebuild; we do NOT
            # write through to disk from here (the adapter doesn't own
            # ``project_dir``, and write-through from the eval hot-path
            # would race with the discovery writer).
            if not self._supports_temperature():
                # Already corrected once — give up. Surface so the
                # discovery layer / operator sees the real failure
                # rather than infinite recursion.
                raise
            logger.info(
                "[mantle] %s rejected temperature — caching "
                "supports_temperature=False on descriptor and retrying once. "
                "Live 400: %s",
                self.model_id,
                exc,
            )
            self._descriptor["supports_temperature"] = False
            return self._invoke_once(
                system_prompt, user_prompt, cancel_token=cancel_token
            )

    def _invoke_once(
        self,
        system_prompt: str | None,
        user_prompt: str,
        *,
        cancel_token: threading.Event | None = None,
    ) -> tuple[str, float, dict]:
        """Single request/response cycle — extracted for self-correction retry reuse.

        Reads the current ``supports_temperature`` flag from the
        descriptor every call so the temperature self-correction's in-memory
        flip takes effect on the second attempt.
        """
        req_dict = self._build_request(
            model_id=self.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.max_output_tokens,
            supports_temperature=self._supports_temperature(),
            temperature=self.temperature,
        )
        body = json.dumps(req_dict).encode("utf-8")

        start = time.perf_counter()
        response = self._sign_and_send(body, cancel_token=cancel_token)
        latency_ms = (time.perf_counter() - start) * 1000

        self._handle_error_response(response)

        payload = response.json()
        text, usage = self._parse_response(payload)

        # Single accrual point — generate() delegates here, so accruing
        # only in generate_with_system() covers both entry points
        # without double-count.
        self._accrue_usage(usage)
        return text, latency_ms, usage

    async def a_generate(self, prompt: str) -> str:
        """Async generation — runs the sync call in a thread.

        Returns only text to satisfy DeepEval's async interface.

        Args:
            prompt: The user prompt.

        Returns:
            The model's text output.
        """
        return await asyncio.to_thread(self.generate, prompt)


async def _verify_mantle_tls() -> None:
    """Boot-time TLS handshake check for the mantle endpoint family.

    Non-blocking by contract — failure logs a WARN and returns
    cleanly. A transient mantle outage at boot must NOT block the
    coordinator from serving runtime evals.

    Cloud mode runs unconditionally; dev mode skips by default but
    can opt in via ``MANTLE_BOOT_SMOKE=1`` (handy for local cert-chain
    debugging).

    A HEAD request on ``/v1/models`` sends no body and expects a 401 (no
    auth) — the test is the cert-chain handshake, not the auth. Any 4xx
    response is success for our purposes.
    """
    is_cloud = os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"
    opt_in = os.environ.get("MANTLE_BOOT_SMOKE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not (is_cloud or opt_in):
        return
    region = _MANTLE_DEFAULT_REGION_FOR_BOOT_CHECK
    url = f"https://bedrock-mantle.{region}.api.aws/v1/models"
    try:
        async with httpx.AsyncClient(
            verify=certifi.where(),
            timeout=5.0,
        ) as client:
            response = await client.head(url)
            logger.info(
                "[mantle] startup TLS check OK status=%s",
                response.status_code,
            )
    except Exception as exc:  # noqa: BLE001 — boot must not fail-closed
        logger.warning(
            "[mantle] startup TLS check failed: %s — continuing", exc
        )
