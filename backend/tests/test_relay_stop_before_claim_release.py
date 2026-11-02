# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relay must stop the microVM BEFORE releasing the turn claim.

``invoke_agentcore_turn``'s ``finally`` releases the per-(user, project) claim on
EVERY exit. On a transport error or the 300s read timeout the local drain stops
but the microVM keeps running and keeps writing the transcript/session. Releasing
the claim then admits the user's retry as a SECOND concurrent writer against the
still-live first turn — the two-writers-on-one-transcript corruption the claim
exists to prevent.

``httpx.ReadTimeout`` is an ``httpx.HTTPError``, so every turn that outruns
BEDROCK_READ_TIMEOUT_S lands in that handler deterministically. The stop must be
issued before the claim frees — the ordering ``app.api.chat.stop_turn`` already
uses on the cancel path.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import httpx
import pytest
import pytest_httpx

from app.services.agentcore_relay_client import invoke_agentcore_turn

_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
_URL = (
    f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
    f"{quote(_ARN, safe='')}/invocations?qualifier=DEFAULT"
)


class _RelayHarness:
    """Bundle the patches every relay-drain test needs.

    Records published envelopes, claim releases, and StopRuntimeSession calls so
    each test can assert ORDER between them: the stop must come before the claim
    release.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []
        self.project_events: list[dict] = []
        self.events: list[str] = []
        self.stop_calls: list[tuple[str, str, str]] = []

    async def publish(self, owner_id: str, envelope: dict) -> None:
        self.published.append((owner_id, envelope))

    async def publish_project(self, _project_id: str, _owner_id: str, event: dict) -> None:
        self.project_events.append(event)

    async def release(self, *_args, **_kwargs) -> None:
        self.events.append("release_claim")

    async def stop(self, session_id: str, arn: str, token: str) -> None:
        self.events.append("stop_runtime_session")
        self.stop_calls.append((session_id, arn, token))

    @property
    def terminal_envelopes(self) -> list[dict]:
        return [e for _o, e in self.published if e.get("kind") == "result"]


def _harness_patches(harness: _RelayHarness):
    """Context managers wiring a harness into the relay module."""
    mock_hub = MagicMock()
    mock_hub.publish_to_user = harness.publish
    # _publish_terminal_error also emits the {type:"error"} project event that
    # drives the toast — record it so tests can assert the UI is told.
    mock_hub.publish_project = harness.publish_project
    return (
        patch(
            "app.services.agentcore_relay_client.get_ws_hub",
            return_value=mock_hub,
        ),
        patch(
            "app.services.agentcore_relay_client.resolve_resume",
            AsyncMock(return_value=(None, None)),
        ),
        patch("app.services.agentcore_relay_client.record_session", AsyncMock()),
        patch(
            "app.services.agentcore_relay_client.release_turn",
            side_effect=harness.release,
        ),
        patch(
            "app.api.chat._stop_agentcore_runtime_session",
            side_effect=harness.stop,
        ),
        patch.dict(
            os.environ,
            {"AGENTCORE_RUNTIME_ARN": _ARN, "AWS_REGION": "us-east-1"},
        ),
    )


async def _invoke(turn_id: str = "turn-abc") -> None:
    await invoke_agentcore_turn(
        project_id="proj-test",
        message="test message",
        owner_id="user-123",
        runtime_session_id=None,
        turn_id=turn_id,
        access_token="Bearer test-token-123",
    )


@pytest.mark.asyncio
async def test_read_timeout_stops_microvm_before_releasing_claim(httpx_mock):
    """The 300s read timeout must stop the microVM before the claim frees.

    ``httpx.ReadTimeout`` is an ``httpx.HTTPError``, so every turn that outruns
    BEDROCK_READ_TIMEOUT_S lands here. Against the buggy code the finally released
    the claim with the microVM still running, admitting the user's retry as a
    second concurrent writer on the same transcript.
    """
    httpx_mock.add_exception(httpx.ReadTimeout("read timeout"), method="POST", url=_URL)

    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6:
        await _invoke("turn-timeout")

    assert harness.stop_calls, (
        "A read timeout must attempt StopRuntimeSession — otherwise the microVM "
        "keeps writing while the claim frees and the retry races it."
    )
    assert harness.events.index("stop_runtime_session") < harness.events.index(
        "release_claim"
    ), f"Stop must precede claim release. Order: {harness.events}"

    # The stop must target the session id actually sent on the invoke header.
    session_id, arn, token = harness.stop_calls[0]
    assert session_id and len(session_id) >= 33, (
        f"Stop must use the invoke session id (>=33 chars), got {session_id!r}"
    )
    assert arn == _ARN
    assert token == "Bearer test-token-123"

    # The user still gets a terminal error so the UI unblocks.
    terminals = harness.terminal_envelopes
    assert len(terminals) == 1 and terminals[0]["data"]["subtype"] == "error"


@pytest.mark.asyncio
async def test_stop_failure_does_not_block_claim_release(httpx_mock):
    """A failing StopRuntimeSession must not mask the error or leak the claim.

    The stop is best-effort by design: it runs on already-failing paths, so an
    exception there must never prevent the terminal error or the claim release
    (a leaked claim wedges the project for the TTL).
    """
    httpx_mock.add_exception(httpx.ReadTimeout("read timeout"), method="POST", url=_URL)

    harness = _RelayHarness()

    async def _failing_stop(*_args, **_kwargs):
        harness.events.append("stop_attempted")
        raise RuntimeError("StopRuntimeSession API error")

    mock_hub = MagicMock()
    mock_hub.publish_to_user = harness.publish
    # _publish_terminal_error also emits the {type:"error"} project event that
    # drives the toast — record it so tests can assert the UI is told.
    mock_hub.publish_project = harness.publish_project
    with (
        patch(
            "app.services.agentcore_relay_client.get_ws_hub", return_value=mock_hub
        ),
        patch(
            "app.services.agentcore_relay_client.resolve_resume",
            AsyncMock(return_value=(None, None)),
        ),
        patch("app.services.agentcore_relay_client.record_session", AsyncMock()),
        patch(
            "app.services.agentcore_relay_client.release_turn",
            side_effect=harness.release,
        ),
        patch(
            "app.api.chat._stop_agentcore_runtime_session", side_effect=_failing_stop
        ),
        patch.dict(
            os.environ,
            {"AGENTCORE_RUNTIME_ARN": _ARN, "AWS_REGION": "us-east-1"},
        ),
    ):
        await _invoke("turn-stopfail")

    assert "stop_attempted" in harness.events
    assert "release_claim" in harness.events, (
        "The claim MUST still be released when the stop fails — otherwise the "
        "project is wedged until the claim TTL expires."
    )
    terminals = harness.terminal_envelopes
    assert len(terminals) == 1 and terminals[0]["data"]["subtype"] == "error", (
        "The original transport error must still surface to the user."
    )


@pytest.mark.asyncio
async def test_no_stop_attempted_when_token_missing(httpx_mock):
    """No access token → skip the stop rather than issue a guaranteed 401.

    Mirrors the cancel path's discipline: log the gap, don't make a call that
    cannot possibly be authorized on an OAuth runtime.
    """
    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6:
        # No access_token → the early token guard fires before any HTTP call.
        await invoke_agentcore_turn(
            project_id="proj-test",
            message="test message",
            owner_id="user-123",
            runtime_session_id=None,
            turn_id="turn-notoken",
            access_token=None,
        )

    assert harness.stop_calls == [], (
        f"Must not call StopRuntimeSession without a token: {harness.stop_calls}"
    )
    assert "release_claim" in harness.events
    terminals = harness.terminal_envelopes
    assert len(terminals) == 1 and terminals[0]["data"]["subtype"] == "error"
