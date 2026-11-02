# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A clean SSE EOF without a ``result`` frame is not success.

The drain loop fell through to the success path when the stream ended without a
terminal ``result`` frame, so no terminal envelope ever reached the browser: the
spinner hangs forever while the claim frees. There is a confirmed in-repo
producer — ``agent_runner`` logs "Ended WITHOUT ResultMessage" and completes the
turn without emitting a result envelope.

The fix tracks ``saw_result`` and synthesizes the terminal error (plus a
best-effort microVM stop, since the runtime may still be alive) when the drain
ends without one.
"""
from __future__ import annotations

import asyncio
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
async def test_eof_without_result_frame_publishes_terminal_error(httpx_mock):
    """A clean EOF with no ``result`` frame must publish a terminal error.

    Against the buggy drain the stream just ends, the function returns via the
    success path, and the browser gets text deltas but NO terminal frame — the
    spinner hangs while the claim is released.
    """
    # Text deltas, then a clean EOF. No `result` frame — the confirmed
    # agent_runner "Ended WITHOUT ResultMessage" shape.
    canned_sse = (
        b"event: text_delta\n"
        b'data: {"delta": "Hello"}\n'
        b"\n"
        b"event: text_delta\n"
        b'data: {"delta": " world"}\n'
        b"\n"
    )
    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([canned_sse]),
        headers={"Content-Type": "text/event-stream"},
    )

    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6:
        await _invoke()

    terminals = harness.terminal_envelopes
    assert len(terminals) == 1, (
        "A stream that ends without a result frame must still publish exactly one "
        f"terminal envelope so the UI unblocks. Published: {harness.published}"
    )
    assert terminals[0]["data"]["subtype"] == "error"
    assert terminals[0]["data"]["error"], "Terminal error must carry a message"

    # The microVM may still be alive — stop must be attempted before the release.
    assert harness.events.index("stop_runtime_session") < harness.events.index(
        "release_claim"
    ), (
        "StopRuntimeSession must be issued BEFORE the claim is released, or a "
        f"retry races the still-live turn. Order: {harness.events}"
    )
    # A failed turn must NOT be reported as completed — the mirror is
    # success-path only.
    assert not any(
        e.get("resource") == "agent_turn" and e.get("type") == "completed"
        for e in harness.project_events
    ), f"No agent_turn/completed on the error path: {harness.project_events}"


@pytest.mark.asyncio
async def test_result_frame_present_is_success_and_publishes_no_error(httpx_mock):
    """Counterpart: a well-formed stream must NOT gain a synthetic error.

    Keeps the fix from firing on the happy path (which would surface a
    spurious error toast on every successful turn).
    """
    canned_sse = (
        b"event: text_delta\n"
        b'data: {"delta": "Hello"}\n'
        b"\n"
        b"event: result\n"
        b'data: {"subtype": "success", "result": "done", "session_id": "sess-relay-1"}\n'
        b"\n"
    )
    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([canned_sse]),
        headers={"Content-Type": "text/event-stream"},
    )

    harness = _RelayHarness()

    # ordering pin: the transcript append must complete BEFORE the
    # terminal mirror is published — the SPA's truncated-stream recovery
    # refetches history when the mirror arrives, so the mirror is a
    # durable-completeness signal. Capture the mirror count at append time.
    mirrors_at_append_time: list[int] = []

    async def _record_append(*args, **kwargs):
        mirrors_at_append_time.append(len([
            e for e in harness.project_events
            if e.get("resource") == "agent_turn"
        ]))

    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6, patch(
        "app.services.agent_runner._append_turn_to_transcript",
        new_callable=AsyncMock,
        side_effect=_record_append,
    ) as mock_append:
        await _invoke("turn-ok")

    # single-writer: the coordinator appends the finished turn to the
    # durable transcript using the result frame's session_id + final answer.
    # (The microVM skips its own append inside the runtime.)
    mock_append.assert_awaited_once_with(
        "proj-test", "sess-relay-1", final_answer="done", turn_id="turn-ok"
    )
    assert mirrors_at_append_time == [0], (
        "transcript append must run BEFORE the agent_turn terminal mirror "
        f"(mirrors already published at append time: {mirrors_at_append_time})"
    )

    terminals = harness.terminal_envelopes
    assert len(terminals) == 1, f"Expected the runtime's own result frame: {terminals}"
    assert terminals[0]["data"].get("subtype") == "success"
    assert harness.stop_calls == [], (
        f"A successful turn must not be stopped: {harness.stop_calls}"
    )
    assert "release_claim" in harness.events
    # The coordinator mirrors agent_turn/completed onto the project
    # stream. The microVM's own completed publish reaches zero browser
    # connections, so without this mirror relayed turns never deliver the
    # turn-end refresh signal (harness config / run-list panels stay stale
    # until remount).
    completed = [
        e for e in harness.project_events
        if e.get("resource") == "agent_turn" and e.get("type") == "completed"
    ]
    assert len(completed) == 1, (
        f"Expected exactly one agent_turn/completed project event: {harness.project_events}"
    )
    assert completed[0]["turn_id"] == "turn-ok"




@pytest.mark.asyncio
async def test_errored_result_frame_mirrors_agent_turn_error_not_completed(httpx_mock):
    """The terminal mirror follows the result frame's subtype.

    A turn that reached its result frame with subtype=error terminated
    cleanly at the transport level (saw_result is True — no synthetic
    error), but announcing it as ``completed`` on the project stream would
    lie to every subscriber that distinguishes the terminal kinds.
    """
    canned_sse = (
        b"event: result\n"
        b'data: {"subtype": "error", "result": null, "error": "agent blew up"}\n'
        b"\n"
    )
    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([canned_sse]),
        headers={"Content-Type": "text/event-stream"},
    )
    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6:
        await _invoke("turn-err")

    terminal_types = [
        e["type"] for e in harness.project_events
        if e.get("resource") == "agent_turn"
    ]
    assert terminal_types == ["error"], (
        f"Expected exactly one agent_turn/error mirror: {harness.project_events}"
    )
    assert harness.stop_calls == [], (
        "A turn that delivered its own terminal frame must not be force-stopped"
    )


@pytest.mark.asyncio
async def test_frame_idle_limit_tears_down_wedged_session(httpx_mock, monkeypatch):
    """Frame silence on a live connection ends the turn honestly.

    The socket read timeout only fires when no BYTES arrive; SSE keepalives
    defeat it, so a stalled session used to hang the drain indefinitely while
    the user watched a spinner. With the frame-idle guard: microVM stopped
    BEFORE the claim frees, terminal error published, no agent_turn/completed
    lie.
    """
    monkeypatch.setenv("RELAY_FRAME_IDLE_LIMIT_S", "0.2")

    async def _stalled_frames(_byte_iter):
        # One real frame, then frame-silence forever (the live signature:
        # the socket stays open — keepalives — but no frame ever arrives).
        yield ("text_delta", {"delta": "working"})
        await asyncio.sleep(3600)

    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([b": keepalive\n\n"]),
        headers={"Content-Type": "text/event-stream"},
    )
    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6, patch(
        "app.services.agentcore_relay_client._parse_sse_frames",
        _stalled_frames,
    ):
        await _invoke("turn-wedged")

    # Terminal error published so the UI unblocks with the truth.
    terminals = harness.terminal_envelopes
    assert len(terminals) == 1, f"Expected one synthesized terminal: {harness.published}"
    assert terminals[0]["data"]["subtype"] == "error"
    assert "stopped responding" in terminals[0]["data"]["error"]
    # The wedged microVM is stopped BEFORE the claim releases.
    assert "stop_runtime_session" in harness.events
    assert harness.events.index("stop_runtime_session") < harness.events.index(
        "release_claim"
    )
    # No completed mirror — the turn did not complete.
    assert not any(
        e.get("resource") == "agent_turn" and e.get("type") == "completed"
        for e in harness.project_events
    )


@pytest.mark.asyncio
async def test_heartbeats_are_liveness_not_content(httpx_mock, monkeypatch):
    """Heartbeat frames reset the idle guard but never reach the browser.

    A turn quietly working (heartbeats flowing, no content frames) must NOT
    be torn down by the idle limit — a harness build can run for many minutes
    without a content frame and is still real work. And no heartbeat may be
    forwarded, seq'd, or logged.
    """
    monkeypatch.setenv("RELAY_FRAME_IDLE_LIMIT_S", "0.2")

    async def _heartbeat_then_result(_byte_iter):
        # Frame gaps stay under the idle limit only thanks to heartbeats.
        for _ in range(4):
            await asyncio.sleep(0.1)
            yield ("heartbeat", {})
        yield ("result", {"subtype": "success", "result": "done", "session_id": "sess-hb"})

    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([b": keepalive\n\n"]),
        headers={"Content-Type": "text/event-stream"},
    )
    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6, patch(
        "app.services.agentcore_relay_client._parse_sse_frames",
        _heartbeat_then_result,
    ), patch(
        "app.services.agent_runner._append_turn_to_transcript",
        new_callable=AsyncMock,
    ):
        await _invoke("turn-hb")

    # Completed normally: result forwarded, completed mirror published.
    assert len(harness.terminal_envelopes) == 1
    assert harness.terminal_envelopes[0]["data"]["subtype"] == "success"
    # No heartbeat ever forwarded to the turn stream.
    assert not any(
        e.get("kind") == "heartbeat" for _o, e in harness.published
    ), f"heartbeats must never be forwarded: {harness.published}"
    # No synthesized error, no force-stop.
    assert harness.stop_calls == []


@pytest.mark.asyncio
async def test_turn_cap_fires_through_heartbeats(httpx_mock, monkeypatch):
    """Heartbeats cannot extend a turn past the absolute ceiling."""
    monkeypatch.setenv("RELAY_FRAME_IDLE_LIMIT_S", "5")
    monkeypatch.setenv("RELAY_TURN_MAX_S", "0.3")

    async def _heartbeats_forever(_byte_iter):
        while True:
            await asyncio.sleep(0.05)
            yield ("heartbeat", {})

    httpx_mock.add_response(
        method="POST",
        url=_URL,
        status_code=200,
        stream=pytest_httpx.IteratorStream([b": keepalive\n\n"]),
        headers={"Content-Type": "text/event-stream"},
    )
    harness = _RelayHarness()
    p1, p2, p3, p4, p5, p6 = _harness_patches(harness)
    with p1, p2, p3, p4, p5, p6, patch(
        "app.services.agentcore_relay_client._parse_sse_frames",
        _heartbeats_forever,
    ):
        await _invoke("turn-capped")

    terminals = harness.terminal_envelopes
    assert len(terminals) == 1
    assert terminals[0]["data"]["subtype"] == "error"
    assert "maximum allowed duration" in terminals[0]["data"]["error"]
    # MicroVM stopped before the claim frees.
    assert harness.events.index("stop_runtime_session") < harness.events.index(
        "release_claim"
    )
