# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the BedrockMantleModel adapter.

Covers every guard the adapter carries:
  * (text, latency_ms, usage) tuple return shape.
  * SigV4 single-source region introspection — the signed
    ``Authorization`` header's ``Credential=.../{region}/bedrock-mantle/...``
    segment matches the URL host's region segment for a cross-region
    descriptor.
  * Wrong-API 400 → ``MantleAPIRejected`` first-attempt-only — no
    retry on validation_error.
  * Reasoning truncation → ``MantleReasoningTruncated``.
  * Schema drift → ``MantleSchemaDrift``.
  * Module-level singleton ``httpx.Client``.
  * ``MANTLE_MAX_CONNECTIONS`` env knob tunability.
  * ``cancel_token`` is checked before AND after the network call.
  * ``x-amzn-requestid`` is captured in the structured log line, for
    correlation.
  * ``accumulated_usage`` matches BedrockRuntimeModel's shape; no double-count
    across calls.
  * Temperature omitted for descriptors with
    ``supports_temperature=False``.
  * Boot smoke check is fail-open + gated on cloud or
    ``MANTLE_BOOT_SMOKE=1``.

All tests use ``pytest-httpx``'s ``httpx_mock`` fixture so no real
network or AWS service is touched.
"""

import asyncio
import json
import logging
import re
import threading
from pathlib import Path
from unittest.mock import patch

import certifi
import httpx
import pytest

from app.services.bedrock_mantle_mappers import (
    MantleAPIRejected,
    MantleReasoningTruncated,
    MantleRequestRejected,
    MantleSchemaDrift,
    MantleTemperatureUnsupported,
    _build_chat_completions_request,
    _build_responses_request,
    _parse_chat_completions_response,
    _parse_responses_response,
)
from app.services.bedrock_mantle_model import (
    BedrockMantleModel,
    MantleCancelled,
    MantleSignatureMismatch,
    _close_shared_mantle_client,
    _close_shared_mantle_credentials,
    _get_shared_mantle_client,
    _verify_mantle_tls,
)
from app.services.model_descriptors import MantleDescriptor


FIXTURES = Path(__file__).parent / "fixtures" / "mantle_payloads"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def reset_shared_client():
    """Each test gets a fresh shared client to avoid state leak between tests."""
    _close_shared_mantle_client()
    _close_shared_mantle_credentials()
    yield
    _close_shared_mantle_client()
    _close_shared_mantle_credentials()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Shorten retry delays so retry-or-not behavior tests run in <1s."""
    monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.001")
    monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.002")


@pytest.fixture(autouse=True)
def aws_creds(monkeypatch):
    """Provide static AWS creds so SigV4 signing has something to sign with."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTACCESSKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    # Strip session token to keep header parsing predictable.
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    # Ensure the standard boto3 credential chain doesn't try to read
    # config files (sandbox may not have them).
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def descriptor() -> MantleDescriptor:
    return {"endpoint": "mantle", "api": "responses", "region": "us-east-1"}


@pytest.fixture
def cross_region_descriptor() -> MantleDescriptor:
    return {"endpoint": "mantle", "api": "responses", "region": "us-west-2"}


# --------------------------------------------------------------------------- #
# happy path returns (text, latency_ms, usage)                                 #
# --------------------------------------------------------------------------- #


def test_generate_with_system_returns_tuple_from_responses_200(httpx_mock, descriptor):
    """Captured 200 → ('OK', latency_ms, {inputTokens:9, outputTokens:12})."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    text, latency_ms, usage = model.generate_with_system(None, "Say OK")
    assert text == "OK"
    assert usage == {"inputTokens": 9, "outputTokens": 12}
    assert latency_ms > 0


# --------------------------------------------------------------------------- #
# SigV4 single-source region introspection                                    #
# --------------------------------------------------------------------------- #


def test_signer_region_matches_host_region(httpx_mock, monkeypatch, cross_region_descriptor):
    """Descriptor=us-west-2 + AWS_DEFAULT_REGION=us-east-1 → both signer + host carry us-west-2."""
    # Coordinator's default region is DELIBERATELY different from the
    # descriptor's region. The single-source rule says the descriptor
    # wins for both signer and URL host — and the signed Authorization
    # header proves it.
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )

    model = BedrockMantleModel(cross_region_descriptor, "openai.gpt-5.5")
    text, _, _ = model.generate_with_system(None, "Say OK")
    assert text == "OK"

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    auth_header = req.headers.get("Authorization", "")
    # Authorization: AWS4-HMAC-SHA256 Credential=AKID/DATE/REGION/SERVICE/aws4_request, ...
    cred_match = re.search(
        r"Credential=[A-Z0-9]+/\d+/([a-z0-9-]+)/bedrock-mantle/aws4_request",
        auth_header,
    )
    assert cred_match is not None, f"Authorization header missing Credential= field: {auth_header}"
    signed_region = cred_match.group(1)
    # URL host: bedrock-mantle.{region}.api.aws
    host = str(req.url.host)
    host_match = re.match(r"bedrock-mantle\.([a-z0-9-]+)\.api\.aws", host)
    assert host_match is not None, f"unexpected URL host: {host}"
    host_region = host_match.group(1)

    assert signed_region == host_region == "us-west-2", (
        f"single-source region rule violated: signed={signed_region}, "
        f"host={host_region}, expected both 'us-west-2'"
    )


# --------------------------------------------------------------------------- #
# Wrong-API 400 → MantleAPIRejected, no retry                                 #
# --------------------------------------------------------------------------- #


def test_wrong_api_400_raises_mantle_api_rejected(httpx_mock, monkeypatch, descriptor):
    """400 with code='validation_error' raises MantleAPIRejected on first attempt."""
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "4")
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleAPIRejected, match="does not support"):
        model.generate_with_system(None, "Say OK")
    # Critically: only ONE network call — the validation_error short-circuits retry.
    assert len(httpx_mock.get_requests()) == 1


# --------------------------------------------------------------------------- #
# Temperature self-correction on an unsupported-temperature 400                #
# --------------------------------------------------------------------------- #


def test_unsupported_temperature_400_self_corrects_and_retries(
    httpx_mock, descriptor
):
    """Reasoning model rejects 'temperature' → adapter caches flag, retries without it.

    Live shape: GPT-5.5/Grok 4.3/etc. reject
    ``temperature`` with a 400 of code ``unsupported_parameter``,
    param ``temperature``. The temperature self-correction catches it, sets
    ``descriptor['supports_temperature'] = False`` in memory, and
    retries the same call WITHOUT temperature. Second call succeeds
    end-to-end.
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_unsupported_temperature.json"),
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    # Default-true → first request includes temperature
    assert descriptor.get("supports_temperature", True) is True

    text, latency_ms, usage = model.generate_with_system(None, "Say OK")
    assert text == "OK"
    assert usage["inputTokens"] > 0

    # Two requests fired: first with temperature → 400, second without → 200.
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    first_body = json.loads(requests[0].read())
    second_body = json.loads(requests[1].read())
    assert "temperature" in first_body
    assert "temperature" not in second_body, (
        "Second request must omit temperature after self-correction"
    )

    # Flag cached on the descriptor for subsequent calls.
    assert descriptor.get("supports_temperature") is False


def test_unsupported_temperature_message_only_also_self_corrects(
    httpx_mock, descriptor
):
    """Schema variation: code may be missing/different but message names temperature.

    Defensive — if a future provider family returns a slightly different
    400 shape but the message still mentions ``temperature``, treat it
    the same way. A code-only check would be an enumeration of known error
    shapes, and it goes stale as soon as a provider words the error
    differently.
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json={
            "error": {
                "code": "unsupported_parameter",
                "message": "'temperature' parameter is not supported",
            }
        },
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    text, _, _ = model.generate_with_system(None, "Say OK")
    assert text == "OK"
    assert descriptor.get("supports_temperature") is False


def test_already_supports_temperature_false_does_not_retry(
    httpx_mock,
):
    """Already-corrected descriptor: a second 400 surfaces the failure, no infinite loop."""
    descriptor: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": False,
    }
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_unsupported_temperature.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleTemperatureUnsupported):
        model.generate_with_system(None, "Say OK")
    # Only ONE request — no retry-loop after the flag is already False.
    assert len(httpx_mock.get_requests()) == 1


def test_unsupported_temperature_distinct_from_validation_error(
    httpx_mock, descriptor
):
    """A code='validation_error' 400 still raises MantleAPIRejected (not TemperatureUnsupported).

    The two 400 paths must stay distinct: validation_error means
    "wrong API" (discovery falls through to the next api);
    unsupported_parameter+temperature means "right API, drop the flag".
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json=_load("responses_400_wrong_api.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleAPIRejected):
        model.generate_with_system(None, "Say OK")
    # No retry, no flag mutation.
    assert descriptor.get("supports_temperature", True) is True


# --------------------------------------------------------------------------- #
# 400 fall-through — body-carrying MantleRequestRejected                       #
# --------------------------------------------------------------------------- #


def test_fall_through_400_surfaces_body_message(httpx_mock, descriptor):
    """A 400 that is neither temperature / safety / validation_error raises
    ``MantleRequestRejected`` whose ``str()`` carries the provider's body
    message verbatim — not the bare httpx "400 Bad Request" wrapper.

    The motivating live shape: forcing a tool call on a reasoning model
    with thinking enabled returns ``"Thinking may not be enabled when
    tool_choice forces tool use"`` with ``type='invalid_request_error'``
    and no ``temperature`` mention, no safety code, and no
    ``code='validation_error'`` — so it falls through every special-case
    branch. Without the fall-through, that reason is dropped and only a
    generic status string reaches ``additional_metadata.error``.
    """
    conflict_message = (
        "Thinking may not be enabled when tool_choice forces tool use"
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        json={
            "error": {
                "message": conflict_message,
                "type": "invalid_request_error",
            }
        },
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleRequestRejected) as excinfo:
        model.generate_with_system(None, "Say OK")
    # The provider's message text — not "400 Bad Request" — is in str(exc).
    assert conflict_message in str(excinfo.value)
    assert "400 Bad Request" not in str(excinfo.value)
    # Non-retryable: only ONE network call fired.
    assert len(httpx_mock.get_requests()) == 1


def test_fall_through_400_non_json_body_uses_response_text(
    httpx_mock, descriptor
):
    """A 400 whose body is not the expected ``error.message`` JSON shape
    still surfaces its raw text via the ``response.text`` fallback, so a
    differently-shaped body never collapses to "(no message)".
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        status_code=400,
        text="upstream rejected the request shape",
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleRequestRejected, match="upstream rejected"):
        model.generate_with_system(None, "Say OK")


def test_503_still_raises_http_status_error_and_stays_retryable(
    httpx_mock, monkeypatch, descriptor
):
    """A 503 still raises ``httpx.HTTPStatusError`` (via the preserved
    trailing ``raise_for_status()``), so ``_is_retryable_mantle_error``
    keeps retrying it — locking the scope of the 400-only fall-through.

    Capping ``BEDROCK_RETRY_MAX_ATTEMPTS`` at 2 keeps the test fast while
    still proving the 503 path is retried (two responses are queued; the
    final one surfaces the ``HTTPStatusError``).
    """
    monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "2")
    for _ in range(2):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            status_code=503,
            text="service unavailable",
        )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(httpx.HTTPStatusError):
        model.generate_with_system(None, "Say OK")
    # Retried: two network calls fired before the final raise.
    assert len(httpx_mock.get_requests()) == 2


# --------------------------------------------------------------------------- #
# Reasoning truncation                                                        #
# --------------------------------------------------------------------------- #


def test_responses_incomplete_max_output_raises_truncation(httpx_mock, descriptor):
    """Status='incomplete' + reason='max_output_tokens' → MantleReasoningTruncated."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_incomplete_truncation.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleReasoningTruncated, match="max_output_tokens") as excinfo:
        model.generate_with_system(None, "Say OK")
    assert "openai.gpt-5.5" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Schema drift                                                                #
# --------------------------------------------------------------------------- #


def test_schema_drift_raises_when_output_text_missing(httpx_mock, descriptor):
    """Payload with reasoning-only output (no message block) → MantleSchemaDrift."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json={
            "id": "x",
            "status": "completed",
            "output": [{"type": "reasoning", "summary": []}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleSchemaDrift, match="output_text"):
        model.generate_with_system(None, "Say OK")


# --------------------------------------------------------------------------- #
# Module-level singleton httpx.Client                                          #
# --------------------------------------------------------------------------- #


def test_shared_client_reuses_connection(httpx_mock, descriptor):
    """50 sequential calls reuse the same Client object."""
    payload = _load("responses_200_minimal.json")
    for _ in range(50):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=payload,
        )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    first_client = _get_shared_mantle_client()
    for _ in range(50):
        text, _, _ = model.generate_with_system(None, "Say OK")
        assert text == "OK"
    # Singleton property — same object across the call sequence.
    assert _get_shared_mantle_client() is first_client


# --------------------------------------------------------------------------- #
# verify=certifi.where()                                                      #
# --------------------------------------------------------------------------- #


def test_uses_certifi_bundle(monkeypatch):
    """Shared client uses certifi.where() for the *.api.aws CA chain."""
    captured_kwargs: dict = {}
    real_ctor = httpx.Client

    def capture(**kwargs):
        captured_kwargs.update(kwargs)
        return real_ctor(**kwargs)

    monkeypatch.setattr("app.services.bedrock_mantle_model.httpx.Client", capture)
    _close_shared_mantle_client()
    client = _get_shared_mantle_client()
    assert captured_kwargs.get("verify") == certifi.where()
    # Default env knobs.
    limits = captured_kwargs.get("limits")
    assert limits is not None
    assert limits.max_connections == 20
    assert limits.max_keepalive_connections == 10


def test_max_connections_env_knob(monkeypatch):
    """Env-tunability — MANTLE_MAX_CONNECTIONS=5 lands on the new client's limits."""
    monkeypatch.setenv("MANTLE_MAX_CONNECTIONS", "5")
    captured_kwargs: dict = {}
    real_ctor = httpx.Client

    def capture(**kwargs):
        captured_kwargs.update(kwargs)
        return real_ctor(**kwargs)

    monkeypatch.setattr("app.services.bedrock_mantle_model.httpx.Client", capture)
    _close_shared_mantle_client()
    _get_shared_mantle_client()
    limits = captured_kwargs.get("limits")
    assert limits is not None
    assert limits.max_connections == 5


# --------------------------------------------------------------------------- #
# cancel_token                                                                #
# --------------------------------------------------------------------------- #


def test_cancel_token_aborts_before_request(descriptor):
    """Cancel_token set before generate → MantleCancelled, no HTTP call."""
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    event = threading.Event()
    event.set()
    with pytest.raises(MantleCancelled, match="before request"):
        model.generate_with_system(None, "Say OK", cancel_token=event)


def test_cancel_token_aborts_after_request(httpx_mock, descriptor):
    """Cancel_token set during the call → MantleCancelled after the response (post-request bracket)."""
    event = threading.Event()

    def cancel_in_flight(request: httpx.Request) -> httpx.Response:
        # Sync httpx can't honor mid-flight cancel; setting the event
        # here is detected by the post-request check.
        event.set()
        return httpx.Response(200, json=_load("responses_200_minimal.json"))

    httpx_mock.add_callback(cancel_in_flight)
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    with pytest.raises(MantleCancelled, match="after request"):
        model.generate_with_system(None, "Say OK", cancel_token=event)


# --------------------------------------------------------------------------- #
# x-amzn-requestid logged                                                     #
# --------------------------------------------------------------------------- #


def test_x_amzn_requestid_logged(httpx_mock, descriptor):
    """X-amzn-requestid is captured in the structured log line.

    Robust against the ``app.main``-setting-``app``-logger-``propagate=False``
    side effect of co-running tests: explicitly capture records on the
    adapter's logger via a dedicated list-handler instead of relying on
    propagation to caplog's root handler.
    """
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
        headers={"x-amzn-requestid": "test-req-id-XYZ"},
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    captured: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _ListHandler(level=logging.INFO)
    target = logging.getLogger("app.services.bedrock_mantle_model")
    target.addHandler(handler)
    prev_level = target.level
    target.setLevel(logging.INFO)
    try:
        model.generate_with_system(None, "Say OK")
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)
    matching = [r for r in captured if "req_id=test-req-id-XYZ" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "region=us-east-1" in msg
    assert "api=responses" in msg
    assert "status=200" in msg


# --------------------------------------------------------------------------- #
# accumulated_usage shape + no double-count                                   #
# --------------------------------------------------------------------------- #


def test_accumulated_usage_shape_matches_bedrock_runtime_model(httpx_mock, descriptor):
    """accumulated_usage adds inputTokens/outputTokens once per call."""
    payload = _load("responses_200_minimal.json")
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=payload,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=payload,
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    model.generate_with_system(None, "Say OK")
    assert model.accumulated_usage == {"inputTokens": 9, "outputTokens": 12}
    model.generate_with_system(None, "Say OK")
    assert model.accumulated_usage == {"inputTokens": 18, "outputTokens": 24}


# --------------------------------------------------------------------------- #
# temperature omitted when descriptor.supports_temperature=False              #
# --------------------------------------------------------------------------- #


def test_temperature_omitted_for_descriptor_supports_false(httpx_mock):
    """Descriptor with supports_temperature=False → request body has no 'temperature' key."""
    desc: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": False,
    }
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(desc, "openai.gpt-5.5", temperature=0.7)
    model.generate_with_system(None, "Say OK")
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert "temperature" not in body


def test_temperature_included_for_descriptor_default_true(httpx_mock, descriptor):
    """Control — default descriptor (no flag) → temperature included."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5", temperature=0.7)
    model.generate_with_system(None, "Say OK")
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["temperature"] == 0.7


# --------------------------------------------------------------------------- #
# Async parity + DeepEval interface                                           #
# --------------------------------------------------------------------------- #


def test_a_generate_returns_text_via_thread(httpx_mock, descriptor):
    """a_generate runs the sync call in a thread; returns text."""
    httpx_mock.add_response(
        method="POST",
        url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
        json=_load("responses_200_minimal.json"),
    )
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    text = asyncio.run(model.a_generate("Say OK"))
    assert text == "OK"


def test_load_model_returns_shared_client(descriptor):
    """Contract — load_model returns the module-level shared client."""
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    c1 = model.load_model()
    c2 = model.load_model()
    assert c1 is c2 is _get_shared_mantle_client()


def test_get_model_name_returns_model_id(descriptor):
    """get_model_name returns the model_id passed to __init__."""
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    assert model.get_model_name() == "openai.gpt-5.5"


# --------------------------------------------------------------------------- #
# boot smoke check is fail-open                                               #
# --------------------------------------------------------------------------- #


def _attach_list_handler(logger_name: str, level: int) -> tuple[list[logging.LogRecord], logging.Handler, logging.Logger, int]:
    """Attach a list-recording handler to a named logger; return (captured, handler, logger, prev_level)."""
    captured: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = _ListHandler(level=level)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(level)
    return captured, handler, logger, prev_level


async def test_verify_mantle_tls_skips_in_dev_without_opt_in(monkeypatch):
    """Dev mode without MANTLE_BOOT_SMOKE → no log records, returns immediately."""
    monkeypatch.delenv("EVAL_STUDIO_MODE", raising=False)
    monkeypatch.delenv("MANTLE_BOOT_SMOKE", raising=False)
    captured, handler, logger, prev_level = _attach_list_handler(
        "app.services.bedrock_mantle_model", logging.INFO
    )
    try:
        await _verify_mantle_tls()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    relevant = [r for r in captured if "[mantle] startup TLS" in r.getMessage()]
    assert relevant == []


async def test_verify_mantle_tls_logs_warning_on_failure(monkeypatch):
    """Fail-open — handshake error logs WARN and returns; never raises."""
    monkeypatch.setenv("MANTLE_BOOT_SMOKE", "1")

    class FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("simulated cert chain failure")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "app.services.bedrock_mantle_model.httpx.AsyncClient",
        FailingAsyncClient,
    )
    captured, handler, logger, prev_level = _attach_list_handler(
        "app.services.bedrock_mantle_model", logging.WARNING
    )
    try:
        await _verify_mantle_tls()  # must NOT raise
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    warns = [
        r
        for r in captured
        if r.levelno == logging.WARNING
        and "[mantle] startup TLS check failed" in r.getMessage()
    ]
    assert len(warns) == 1


async def test_verify_mantle_tls_logs_info_on_success(monkeypatch, httpx_mock):
    """Happy path — opt-in via MANTLE_BOOT_SMOKE; HEAD response logs INFO."""
    monkeypatch.setenv("MANTLE_BOOT_SMOKE", "1")
    httpx_mock.add_response(
        method="HEAD",
        url="https://bedrock-mantle.us-east-1.api.aws/v1/models",
        status_code=401,  # no auth — that's fine; the check is the handshake
    )
    captured, handler, logger, prev_level = _attach_list_handler(
        "app.services.bedrock_mantle_model", logging.INFO
    )
    try:
        await _verify_mantle_tls()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    matching = [
        r for r in captured if "[mantle] startup TLS check OK" in r.getMessage()
    ]
    assert len(matching) == 1
    assert "status=401" in matching[0].getMessage()


# --------------------------------------------------------------------------- #
# Strategy dispatch picks the right mapper from descriptor.api                #
# --------------------------------------------------------------------------- #


def test_strategy_dispatch_picks_responses_mapper(descriptor):
    """descriptor.api='responses' → builder + parser are the live functions."""
    model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
    assert model._build_request is _build_responses_request
    assert model._parse_response is _parse_responses_response
    assert model._path == "/openai/v1/responses"


def test_strategy_dispatch_picks_chat_completions_mapper_live():
    """Descriptor.api='chat_completions' → live mapper resolved (active).

    The stubs are gone; the live chat_completions mapper builds an
    OpenAI-compat ``messages`` payload and parses ``choices[0].message.content``.
    Wire-format verified live against ``mistral-large-3``.
    """
    desc_cc: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "chat_completions",
        "region": "us-east-1",
    }
    model = BedrockMantleModel(desc_cc, "mistral.mistral-large-3")
    assert model._build_request is _build_chat_completions_request
    assert model._parse_response is _parse_chat_completions_response
    assert model._path == "/v1/chat/completions"
    # Build a request body in isolation (no network) and check shape.
    body = _build_chat_completions_request(
        model_id="mistral.mistral-large-3",
        system_prompt=None,
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.0,
    )
    assert body["model"] == "mistral.mistral-large-3"
    assert body["max_tokens"] == 64
    assert body["messages"] == [{"role": "user", "content": "Say OK"}]
    assert body["temperature"] == 0.0


def test_strategy_dispatch_picks_anthropic_messages_mapper_live():
    """Descriptor.api='anthropic_messages' → live mapper resolved (active).

    Wire-format verified live against ``anthropic.claude-opus-4-8``.
    Required field ``anthropic_version='bedrock-2023-05-31'`` (400
    without it).
    """
    from app.services.bedrock_mantle_mappers import (
        _build_anthropic_messages_request,
        _parse_anthropic_messages_response,
    )

    desc_an: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "anthropic_messages",
        "region": "us-east-1",
    }
    model = BedrockMantleModel(desc_an, "anthropic.claude-opus-4-8")
    assert model._build_request is _build_anthropic_messages_request
    assert model._parse_response is _parse_anthropic_messages_response
    assert model._path == "/anthropic/v1/messages"
    body = _build_anthropic_messages_request(
        model_id="anthropic.claude-opus-4-8",
        system_prompt="be brief",
        user_prompt="Say OK",
        max_output_tokens=64,
        supports_temperature=True,
        temperature=0.0,
    )
    assert body["model"] == "anthropic.claude-opus-4-8"
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 64
    assert body["messages"] == [{"role": "user", "content": "Say OK"}]
    assert body["system"] == "be brief"
    assert body["temperature"] == 0.0


# --------------------------------------------------------------------------- #
# preflight() public method                                                   #
# --------------------------------------------------------------------------- #


from app.services.bedrock_mantle_model import (  # noqa: E402
    MantleNotInvocable,
    ModelRemovedFromMantle,
)


class TestPreflight:
    """Public ``preflight()`` method on ``BedrockMantleModel``.

    Pinned behaviors:
      * Returns ``None`` on a 200 success.
      * 404 → ``ModelRemovedFromMantle`` (dispatch invalidates).
      * 400 validation_error → ``MantleAPIRejected`` (dispatch invalidates).
      * 403 sig mismatch → ``MantleNotInvocable`` (dispatch does NOT
        invalidate).
      * Any other exception (connect error, timeout) → wrapped as
        ``MantleNotInvocable``.
      * Throwaway-adapter pattern: owner instance's
        ``accumulated_usage`` is NOT polluted on success.
      * 1-token request body — ``max_output_tokens`` floor.
    """

    def test_preflight_returns_none_on_success(self, httpx_mock, descriptor):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_load("responses_200_minimal.json"),
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        assert model.preflight() is None
        # Exactly one network call — preflight is one-shot.
        assert len(httpx_mock.get_requests()) == 1

    def test_preflight_raises_model_removed_on_404(
        self, httpx_mock, descriptor
    ):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            status_code=404,
            json={"error": {"message": "model not found"}},
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        with pytest.raises(ModelRemovedFromMantle) as excinfo:
            model.preflight()
        assert excinfo.value.model_id == "openai.gpt-5.5"
        assert "openai.gpt-5.5" in str(excinfo.value)
        assert "descriptor invalidated" in str(excinfo.value)

    def test_preflight_raises_mantle_api_rejected_on_400(
        self, httpx_mock, descriptor
    ):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            status_code=400,
            json={
                "error": {
                    "code": "validation_error",
                    "message": (
                        "does not support the '/openai/v1/responses' API"
                    ),
                }
            },
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        with pytest.raises(MantleAPIRejected, match="does not support"):
            model.preflight()

    def test_preflight_raises_not_invocable_on_signature_mismatch(
        self, httpx_mock, descriptor
    ):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            status_code=403,
            text="<XML>SignatureDoesNotMatch</XML>",
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        with pytest.raises(MantleNotInvocable) as excinfo:
            model.preflight()
        # Wrapped class name surfaced as ``reason`` so dispatch can
        # format a helpful error message.
        assert excinfo.value.reason == "MantleSignatureMismatch"
        assert excinfo.value.model_id == "openai.gpt-5.5"

    def test_preflight_raises_not_invocable_on_connect_error(
        self, httpx_mock, descriptor
    ):
        httpx_mock.add_exception(httpx.ConnectError("DNS failed"))
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        with pytest.raises(MantleNotInvocable) as excinfo:
            model.preflight()
        assert excinfo.value.model_id == "openai.gpt-5.5"
        # Either the class name or the original message is preserved
        # in the wrapped exception's reason/detail.
        msg = str(excinfo.value)
        assert "ConnectError" in msg or "DNS failed" in msg

    def test_preflight_does_not_pollute_accumulated_usage(
        self, httpx_mock, descriptor
    ):
        """Throwaway-adapter pattern — owner's accumulated_usage stays at zero.

        ``_probe_capability`` constructs a separate adapter to
        probe; preflight mirrors that pattern to keep the owner's
        judge-cost accounting clean.
        """
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_load("responses_200_minimal.json"),
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        # Pre-condition: clean slate.
        assert model.accumulated_usage == {
            "inputTokens": 0,
            "outputTokens": 0,
        }
        model.preflight()
        # Post-condition: still clean — the throwaway adapter accrued,
        # not the owner.
        assert model.accumulated_usage == {
            "inputTokens": 0,
            "outputTokens": 0,
        }

    def test_preflight_uses_64_token_max_output(
        self, httpx_mock, descriptor
    ):
        """Probe body sets ``max_output_tokens=64`` — verified live as the
        smallest cap that lets reasoning models complete the reasoning
        budget and emit (or cleanly truncate). 1 trips a min-value 400
        before the wrong-api signal can surface; 16 (the API minimum)
        is consumed entirely by reasoning tokens on gpt-5.5/grok-4.3.
        """
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_load("responses_200_minimal.json"),
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        model.preflight()
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        # ``_build_responses_request`` puts ``max_output_tokens`` at the
        # top level of the JSON body.
        assert body.get("max_output_tokens") == 64

    def test_preflight_treats_truncation_as_success(
        self, httpx_mock, descriptor
    ):
        """Truncation in preflight = success — request reached the model.

        Reasoning models burn the small probe budget on hidden reasoning
        tokens and truncate; that proves the api/region/auth all work,
        which is the entire signal preflight needs. Production calls
        use a generous user-set cap. Without this, GPT-5.5 / Grok 4.3 /
        any reasoning model is excluded from eval as 'not invocable'.
        """
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_load("responses_incomplete_truncation.json"),
        )
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")
        # No exception — preflight returns None on truncation.
        result = model.preflight()
        assert result is None


# --------------------------------------------------------------------------- #
# Credential provider caching                                                  #
# --------------------------------------------------------------------------- #


def test_credential_provider_resolved_once_across_multiple_calls(httpx_mock, descriptor, monkeypatch):
    """boto3.Session / get_credentials called ONCE; get_frozen_credentials per request.

    The provider is resolved once and cached at module level, then
    .get_frozen_credentials() is called per request to re-freeze with live
    creds (so a rotating Fargate task role's new creds are picked up). A
    per-call resolution would build a boto3.Session on every _sign_and_send.
    """
    from botocore.credentials import ReadOnlyCredentials
    from unittest.mock import MagicMock, patch

    # Mock the provider whose get_frozen_credentials() returns valid creds
    mock_provider = MagicMock()
    mock_provider.get_frozen_credentials.return_value = ReadOnlyCredentials(
        access_key="AKIATESTACCESSKEY",
        secret_key="test-secret",
        token=None,
    )

    # Mock the Session to return our mock provider
    mock_session = MagicMock()
    mock_session.get_credentials.return_value = mock_provider

    # Set up httpx mock for 3 successful responses
    for _ in range(3):
        httpx_mock.add_response(
            method="POST",
            url="https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
            json=_load("responses_200_minimal.json"),
        )

    with patch("app.services.bedrock_mantle_model.boto3.Session", return_value=mock_session):
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")

        # Drive 3 calls
        for i in range(3):
            text, _, _ = model.generate_with_system(None, f"Call {i+1}")
            assert text == "OK"

        # The provider is cached, so get_credentials runs once across 3 calls.
        assert mock_session.get_credentials.call_count == 1, \
            f"get_credentials should be called once (cached), was {mock_session.get_credentials.call_count}"

        # get_frozen_credentials should be called per request (3 times)
        assert mock_provider.get_frozen_credentials.call_count == 3, \
            f"get_frozen_credentials should be called per request (3×), was {mock_provider.get_frozen_credentials.call_count}"


def test_credential_provider_none_raises_runtime_error(httpx_mock, descriptor):
    """When get_credentials() returns None, RuntimeError still fires with the exact message.

    With the provider cached, the error fires when the cache is first built
    (the first call), with the same message text.
    """
    from unittest.mock import MagicMock, patch

    # Mock a Session whose get_credentials() returns None
    mock_session = MagicMock()
    mock_session.get_credentials.return_value = None

    with patch("app.services.bedrock_mantle_model.boto3.Session", return_value=mock_session):
        model = BedrockMantleModel(descriptor, "openai.gpt-5.5")

        # The error must surface from _sign_and_send (or the cache builder it calls)
        with pytest.raises(RuntimeError) as excinfo:
            model.generate_with_system(None, "Test")

        # Exact message text must be preserved (optimizer may grep for it)
        err_msg = str(excinfo.value)
        assert "No AWS credentials available for Bedrock Mantle SigV4 signing" in err_msg
        assert "set AWS_PROFILE or run on a Fargate/EC2 task with an attached role" in err_msg
