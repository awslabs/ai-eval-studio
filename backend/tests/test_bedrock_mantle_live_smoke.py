# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live smoke tests against the real Bedrock Mantle endpoint.

Marked ``integration`` — skipped by default. Run with::

    AWS_PROFILE=... pytest -m integration tests/test_bedrock_mantle_live_smoke.py

Purpose: hermetic unit tests replay captured fixtures, which CAN'T catch
"captured fixture diverges from live API" — for example, a `temperature`
rejection that looked like a `validation_error` 400 in the fixture while the
live shape was `unsupported_parameter`, which broke GPT-5.5 invocation.
Running this live smoke before a deploy catches that class of drift.

Tests are thin — one 1-token call to a known reasoning model — and
check only that the adapter completes the round-trip without raising,
returning a non-empty text. Cost per run is ~$0.0001.
"""

from __future__ import annotations

import functools
import os

import pytest

from app.services.bedrock_mantle_model import BedrockMantleModel
from app.services.model_descriptors import MantleDescriptor


pytestmark = pytest.mark.integration


# Snapshot the REAL credential environment at import time, BEFORE any fixture
# runs. ``tests/conftest.py`` installs an autouse fixture that scrubs the AWS
# chain for every test (AWS_ACCESS_KEY_ID="testing", config files -> /dev/null)
# so hermetic tests can never touch real AWS. That is correct for the other
# ~2000 tests and fatal for these four: they are gated on real credentials being
# present in the parent environment, then had them replaced before the body ran,
# so they failed with "No AWS credentials in the standard chain" rather than
# skipping or passing. Restoring the snapshot is what makes an opt-in live test
# possible without weakening the default isolation.
_REAL_AWS_ENV = {
    k: v for k, v in os.environ.items() if k.startswith("AWS_")
}


@pytest.fixture(autouse=True)
def _restore_real_aws_credentials(monkeypatch):
    """Undo conftest's credential scrub for this module only.

    Module-level autouse fixtures run AFTER conftest-level ones of the same
    scope, so this re-applies the real values on top of the dummies. Scoped to
    this file, so the global isolation the rest of the suite depends on is
    untouched.
    """
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        if key in _REAL_AWS_ENV:
            monkeypatch.setenv(key, _REAL_AWS_ENV[key])
        else:
            monkeypatch.delenv(key, raising=False)


def _has_real_creds() -> bool:
    """True when the test runner has live AWS credentials available."""
    if os.environ.get("AWS_PROFILE"):
        return True
    if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get(
        "AWS_ACCESS_KEY_ID", ""
    ).lower().startswith("testing"):
        return True
    return False


@functools.lru_cache(maxsize=1)
def _mantle_reachable() -> bool:
    """True when the caller's identity is actually authorized for bedrock-mantle.

    Credential PRESENCE is not the requirement these tests have. ``bedrock-mantle``
    is a separate authorization surface from ``bedrock`` / ``bedrock-runtime``:
    the CDK grants it to the coordinator and worker roles only
    (``arn:aws:bedrock-mantle:*:*:project/*`` plus a managed policy). An ordinary
    developer identity can list 121 foundation models and invoke Converse
    successfully while mantle still answers ``401 invalid_api_key`` — so gating
    on ``AWS_PROFILE`` being set made these tests FAIL rather than skip for
    anyone without the mantle grant.

    Probes the SigV4-signed model listing, which invokes no model and therefore
    costs nothing. Cached so the probe runs once per session.

    ``_list_mantle_in_region`` deliberately SWALLOWS transport and status errors
    (one bad region must not abort a multi-region gather), so a 401 comes back as
    an empty list rather than an exception. Authorization is therefore inferred
    from a NON-EMPTY catalog: a reachable, authorized region always lists models.

    The probe MUST leave no trace. ``bedrock_discovery`` caches a module-level
    ``httpx.AsyncClient`` bound to whichever event loop first used it, and
    ``asyncio.run`` closes the loop it creates — so probing without resetting
    that global leaves a client attached to a dead loop, and every later call
    fails and is swallowed into an empty catalog. Reset it afterwards.

    Returns:
        True when the listing returns at least one model. False on 401/403, an
        empty catalog, or any transport error — never raises, because an
        unreachable endpoint must skip these tests, not error them.
    """
    try:
        import asyncio

        from app.services import bedrock_discovery

        try:
            return bool(
                asyncio.run(bedrock_discovery._list_mantle_in_region("us-east-1"))
            )
        finally:
            bedrock_discovery._shared_async_client = None
            bedrock_discovery._shared_async_lock = None
    except Exception:  # noqa: BLE001 — any failure means "cannot run live"
        return False


_LIVE = _has_real_creds() and _mantle_reachable()
_LIVE_REASON = (
    "requires live AWS credentials"
    if not _has_real_creds()
    else "identity is not authorized for bedrock-mantle (401/403) — the CDK "
    "grants it to the coordinator/worker roles, not to developer identities"
)


@pytest.mark.skipif(not _LIVE, reason=_LIVE_REASON)
def test_live_reasoning_model_smoke():
    """1-token call to GPT-5.5 succeeds end-to-end with temperature self-correction.

    Validates the headline capability: a reasoning model on mantle
    invokes cleanly. The adapter starts with default
    ``supports_temperature=True``; the live 400 trips the self-correction path; the
    second call (without temperature) succeeds. Without the self-correction fix
    this raises ``MantleTemperatureUnsupported`` on attempt 1.

    Skipped when credentials are not present so CI hermeticity is
    preserved. Run manually before each deploy.
    """
    descriptor: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
        # Default supports_temperature=True so we exercise the
        # self-correction path on the first call.
    }
    model = BedrockMantleModel(
        descriptor, "openai.gpt-5.5", max_output_tokens=64
    )
    text, latency_ms, usage = model.generate_with_system(None, "Say OK.")
    assert isinstance(text, str)
    assert text.strip(), "live model returned empty text"
    assert usage["inputTokens"] > 0
    assert usage["outputTokens"] > 0
    assert latency_ms > 0
    # Self-correction must have flipped the in-memory flag.
    assert descriptor.get("supports_temperature") is False, (
        "GPT-5.5 is a reasoning model; the live 400 must have triggered "
        "the temperature self-correction and cached supports_temperature=False"
    )


@pytest.mark.skipif(not _LIVE, reason=_LIVE_REASON)
def test_live_mantle_listing_returns_non_empty():
    """Live SigV4 GET /v1/models in us-east-1 returns the OpenAI-shaped catalog.

    Validates the wire-format contract that hermetic fixtures replay.
    Live divergence has hit the listing path as well as invocation (the
    ``data`` vs ``models`` key), so the listing path gets its own live
    smoke — fixtures alone are too easy to drift.

    Asserts:
      * response shape is ``{"data": [...], "object": "list"}``
      * at least one entry exists (the account has models authorized)
      * each entry has a string ``id``
    """
    import asyncio

    from app.services.bedrock_discovery import _list_mantle_in_region

    entries = asyncio.run(_list_mantle_in_region("us-east-1"))
    assert entries, "live us-east-1 mantle listing returned 0 entries"
    # Every entry has the expected catalog shape (the converter already
    # extracted the id; an empty list back means the response shape was
    # wrong even though the HTTP call succeeded).
    for entry in entries:
        assert isinstance(entry["model_id"], str) and entry["model_id"]
        assert entry["endpoint"] == "mantle"
        assert entry["region"] == "us-east-1"


@pytest.mark.skipif(not _LIVE, reason=_LIVE_REASON)
def test_live_full_discovery_yields_one_survivor_per_api(tmp_path):
    """End-to-end discover_models surfaces one model per api family.

    The crystal-clear regression gate: after discovery against the live
    account, the catalog must contain at least one ``responses`` model
    (gpt-5.5), one ``chat_completions`` model (mistral.mistral-large-3),
    and one ``anthropic_messages`` model (anthropic.claude-opus-4-8) —
    proving the probe sweep, all three mappers, and the full discovery
    pipeline work end to end. A live-divergence bug in any one mapper
    shows up as ONE missing api family in the catalog, which this single
    test catches.

    Skipped when credentials are not present so CI hermeticity is
    preserved. Run manually before each deploy.
    """
    import asyncio

    from app.services.bedrock_discovery import discover_models

    result = asyncio.run(discover_models("live-smoke-test", tmp_path))

    by_api: dict[str, list[str]] = {
        "responses": [],
        "chat_completions": [],
        "anthropic_messages": [],
    }
    for entry in result.catalog:
        api = entry.get("api")
        if api in by_api:
            by_api[api].append(entry["model_id"])

    assert by_api["responses"], (
        "live discovery yielded zero responses-api models — gpt-5.5 and "
        "similar reasoning models are unreachable. Likely cause: probe "
        "token cap, temperature self-correction, or responses mapper drift."
    )
    assert by_api["chat_completions"], (
        "live discovery yielded zero chat_completions-api models — "
        "mistral.mistral-large-3 and similar are unreachable. Likely "
        "cause: chat_completions mapper drift (max_tokens vs "
        "max_completion_tokens, message shape, etc.)."
    )
    assert by_api["anthropic_messages"], (
        "live discovery yielded zero anthropic_messages-api models — "
        "anthropic.claude-opus-4-8 unreachable. Likely cause: missing "
        "anthropic_version field, system-as-message vs top-level, etc."
    )


@pytest.mark.skipif(not _LIVE, reason=_LIVE_REASON)
def test_live_reasoning_model_skips_round_trip_when_flag_preset():
    """Pre-set supports_temperature=False → first call succeeds (no 400 round trip).

    Validates that the harness path — where the descriptor on disk
    already carries the flag from a previous discovery — invokes the
    model in ONE network call, not two.
    """
    descriptor: MantleDescriptor = {
        "endpoint": "mantle",
        "api": "responses",
        "region": "us-east-1",
        "supports_temperature": False,
    }
    model = BedrockMantleModel(
        descriptor, "openai.gpt-5.5", max_output_tokens=64
    )
    text, _, usage = model.generate_with_system(None, "Say OK.")
    assert text.strip()
    assert usage["inputTokens"] > 0
