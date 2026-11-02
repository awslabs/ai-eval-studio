# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-event frame cross-process publish reachability.

The failure this guards: `delete_run` from chat with the Evaluations tab open leaves
the deleted run on screen, because the run list is pull-only REST and the publish
lands in the microVM's `ws_hub`, which has zero browser connections. A test that
exercises only the coordinator path and asserts that a publish was ATTEMPTED passes
even when the publish lands in a hub with zero connections.

These assert the effect at the DESTINATION — a browser-reachable envelope on the
project stream, read from a real `WsConnection` queue on the real `WsHub` — which
is the only assertion that failure cannot satisfy.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import pytest
import pytest_httpx

from app.agentcore_runtime import _sse_frames
from app.platform import process_role
from app.services.agentcore_relay_client import invoke_agentcore_turn
from app.services.agent_turn import get_agent_turn_tracker
from app.services.ws_hub import WsConnection, get_ws_hub

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset process singletons after each test for isolation.

    AgentTurnTracker is a process singleton that conftest.py does NOT reset.
    AgentTurn.__init__ registers a replay provider on the WsHub singleton.
    test_execution_home_guards.py already uses project id "proj-1", so this file
    uses unique project ids (proj-frame-*) to avoid collision.

    This fixture's teardown unregisters all connections and replay providers created
    by these tests and resets the turn tracker.
    """
    yield
    # Teardown
    tracker = get_agent_turn_tracker()
    tracker._reset()
    hub = get_ws_hub()
    # Unregister all connections that might have been created
    for conn_list in list(hub._by_user.values()):
        for conn in list(conn_list):
            hub.unregister(conn)
    # Clear replay providers
    hub._replay_providers.clear()


@pytest.fixture
def _reset_warned():
    """No-op kept for the test that requests it: it only imports ``app.services.jobs``.

    The one-shot warning latch it used to reset no longer exists — unroutable
    publishes are stashed in the outbox instead — so there is nothing to reset.
    """
    import app.services.jobs as jobs_mod
    yield


async def test_microvm_to_coordinator_boundary_reaches_project_stream(tmp_path: Path, httpx_mock):
    """Load-bearing: project event from microVM reaches project:<id> via the drain.

    Three stages: (1) microVM producer appends frame to turn buffer, (2) real
    _sse_frames emits it across SSE boundary, (3) coordinator drain re-targets it
    onto project:<id> where a real WsConnection receives it.

    The PRIMARY assertion is the contents of a real WsConnection.queue after the
    real hub fan-out. Asserting that a forwarding helper was called, or that
    publish_project was called, would measure intent, not browser reachability.
    """
    from app.services.mcp_tools.eval import delete_eval_run as _delete_run_tool
    delete_eval_run = _delete_run_tool.handler

    project_id = "proj-frame-t1"
    owner_id = "owner-t1"
    turn_id = "turn-t1"

    # Resolve delete_eval_run's project_dir lookup
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    output_dir = project_dir / "output"
    output_dir.mkdir()
    result_file = output_dir / "eval_results_20260101_120000.json"
    result_file.write_text(json.dumps({"models": []}))

    # Stage 1: microVM producer (process role flipped to AgentCore)
    with patch.object(process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE):
        # Create a real turn so the tracker resolution is real
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn(project_id, origin="user", owner_id=owner_id)
        # Override the generated turn_id with our known one for the relay
        turn.turn_id = turn_id
        turn.status = "running"

        # Point delete_eval_run at the test project tree
        with patch(
            "app.services.mcp_tools.eval.resolve_project_dir",
            return_value=project_dir,
        ):
            # Call the REAL delete_eval_run handler (the failing scenario)
            await delete_eval_run({
                "project_id": project_id,
                "run_filename": "eval_results_20260101_120000.json",
            })

        # Let the fire-and-forget append tasks run
        for _ in range(5):
            await asyncio.sleep(0)

    # Append a terminal result so _sse_frames terminates
    from app.services.ws_envelope import make_envelope
    result_envelope = make_envelope(
        stream=f"turn:{turn_id}",
        kind="result",
        data={"type": "complete"},
    )
    await turn.append_event(result_envelope)
    await asyncio.sleep(0)

    # Stage 2: the real SSE carrier
    sse_bytes = []
    async for chunk in _sse_frames(turn):
        sse_bytes.append(chunk)
    sse_payload = "".join(sse_bytes)

    # Stage 3: coordinator consumer (role flipped back to coordinator)
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Register a real WsConnection on the real WsHub
        hub = get_ws_hub()
        ws_mock = MagicMock()
        conn = WsConnection(connection_id="conn-t1", user_id=owner_id, websocket=ws_mock)
        hub.register(conn)

        # Relay harness patches (from test_relay_terminal_frame.py)
        _ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        _URL = (
            f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
            f"{quote(_ARN, safe='')}/invocations?qualifier=DEFAULT"
        )

        # Apply harness patches using contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ))
            stack.enter_context(patch("app.services.agentcore_relay_client.record_session", AsyncMock()))
            stack.enter_context(patch("app.services.agentcore_relay_client.release_turn", AsyncMock()))
            stack.enter_context(patch(
                "app.api.chat._stop_agentcore_runtime_session",
                AsyncMock(),
            ))
            stack.enter_context(patch.dict(
                os.environ,
                {"AGENTCORE_RUNTIME_ARN": _ARN, "AWS_REGION": "us-east-1"},
            ))

            # Serve the SSE bytes captured above
            httpx_mock.add_response(
                url=_URL,
                method="POST",
                status_code=200,
                stream=pytest_httpx.IteratorStream([sse_payload.encode()]),
            )

            await invoke_agentcore_turn(
                project_id=project_id,
                message="delete that run",
                owner_id=owner_id,
                runtime_session_id=None,
                turn_id=turn_id,
                access_token="fake-token",
            )

    # PRIMARY ASSERTION: read the real WsConnection queue
    received = []
    try:
        while True:
            envelope = conn.queue.get_nowait()
            received.append(envelope)
    except asyncio.QueueEmpty:
        pass

    # Find the project_event envelopes on stream project:<id>. The relay
    # also mirrors agent_turn/completed on success — filter it out;
    # the event under test here is the framed eval_run one.
    project_events = [
        e for e in received
        if e.get("stream") == f"project:{project_id}"
        and e.get("kind") == "project_event"
        and e["data"].get("resource") != "agent_turn"
    ]

    assert len(project_events) == 1, (
        f"expected exactly 1 non-agent_turn project_event on project:{project_id}, "
        f"got {len(project_events)}; received={received}"
    )

    event_data = project_events[0]["data"]
    assert event_data["resource"] == "eval_run", f"wrong resource: {event_data}"
    assert event_data["type"] == "deleted", f"wrong type: {event_data}"
    # ``run_filename`` is the field name both publishers use (the MCP tool and the
    # HTTP DELETE endpoint), so asserting it here also pins the payload contract the
    # SPA's consumer reads.
    assert event_data["run_filename"] == "eval_results_20260101_120000.json", (
        f"run_filename missing or wrong: {event_data}"
    )


async def test_project_frame_not_on_turn_stream_and_not_durably_appended(tmp_path: Path, httpx_mock):
    """project_event never lands on turn:<id> and never goes to durable log.

    A frame on the turn stream with kind=project_event would be lost, because no
    project-event consumer subscribes to turn streams. The durable append must be
    skipped too — a project frame is not part of the turn transcript.
    """
    from app.services.mcp_tools.eval import delete_eval_run as _delete_run_tool
    delete_eval_run = _delete_run_tool.handler

    project_id = "proj-frame-t2"
    owner_id = "owner-t2"
    turn_id = "turn-t2"

    # Resolve delete_eval_run's project_dir lookup
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    output_dir = project_dir / "output"
    output_dir.mkdir()
    result_file = output_dir / "eval_results_20260102_120000.json"
    result_file.write_text(json.dumps({"models": []}))

    # Stage 1: microVM producer
    with patch.object(process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE):
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn(project_id, origin="user", owner_id=owner_id)
        # Override the generated turn_id with our known one for the relay
        turn.turn_id = turn_id
        turn.status = "running"

        with patch(
            "app.services.mcp_tools.eval.resolve_project_dir",
            return_value=project_dir,
        ):
            await delete_eval_run({
                "project_id": project_id,
                "run_filename": "eval_results_20260102_120000.json",
            })

        for _ in range(5):
            await asyncio.sleep(0)

    # Terminal result
    from app.services.ws_envelope import make_envelope
    result_envelope = make_envelope(
        stream=f"turn:{turn_id}", kind="result", data={"type": "complete"}
    )
    await turn.append_event(result_envelope)
    await asyncio.sleep(0)

    # Stage 2: SSE bytes
    sse_bytes = []
    async for chunk in _sse_frames(turn):
        sse_bytes.append(chunk)
    sse_payload = "".join(sse_bytes)

    # Stage 3: coordinator drain with recording spies on durable append
    allocate_calls = []
    append_calls = []

    async def _record_allocate(turn_id):
        seq = len(allocate_calls) + 1
        allocate_calls.append((turn_id, seq))
        return seq

    async def _record_append(turn_id, seq, envelope, ttl):
        append_calls.append((turn_id, seq, envelope, ttl))

    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        hub = get_ws_hub()
        ws_mock = MagicMock()
        conn = WsConnection(connection_id="conn-t2", user_id=owner_id, websocket=ws_mock)
        hub.register(conn)

        _ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        _URL = (
            f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
            f"{quote(_ARN, safe='')}/invocations?qualifier=DEFAULT"
        )

        # Patch metadata store to record calls
        mock_store = MagicMock()
        mock_store.allocate_turn_seq = _record_allocate
        mock_store.append_turn_event = _record_append

        mock_platform = MagicMock()
        mock_platform.return_value.metadata_store = mock_store

        # Apply harness patches using contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ))
            stack.enter_context(patch("app.services.agentcore_relay_client.record_session", AsyncMock()))
            stack.enter_context(patch("app.services.agentcore_relay_client.release_turn", AsyncMock()))
            stack.enter_context(patch(
                "app.api.chat._stop_agentcore_runtime_session",
                AsyncMock(),
            ))
            stack.enter_context(patch.dict(
                os.environ,
                {"AGENTCORE_RUNTIME_ARN": _ARN, "AWS_REGION": "us-east-1"},
            ))
            stack.enter_context(patch("app.services.agentcore_relay_client.get_platform", mock_platform))

            httpx_mock.add_response(
                url=_URL,
                method="POST",
                status_code=200,
                stream=pytest_httpx.IteratorStream([sse_payload.encode()]),
            )

            await invoke_agentcore_turn(
                project_id=project_id,
                message="delete that run",
                owner_id=owner_id,
                runtime_session_id=None,
                turn_id=turn_id,
                access_token="fake-token",
            )

        # Let durable append tasks complete
        for _ in range(10):
            await asyncio.sleep(0)

    # Collect envelopes from connection
    received = []
    try:
        while True:
            envelope = conn.queue.get_nowait()
            received.append(envelope)
    except asyncio.QueueEmpty:
        pass

    # Assert NO envelope with stream=turn:<id> and kind=project_event
    turn_project_events = [
        e for e in received
        if e.get("stream") == f"turn:{turn_id}"
        and e.get("kind") == "project_event"
    ]
    assert turn_project_events == [], (
        f"project_event must not appear on turn stream; found {turn_project_events}"
    )

    # Assert the durable append was never called with a project_event frame
    project_appends = [
        call for call in append_calls
        if call[2].get("kind") == "project_event"
    ]
    assert project_appends == [], (
        f"project_event must not be durably appended; found {project_appends}"
    )


async def test_coordinator_path_unchanged():
    """On the coordinator, _publish_project_event produces exactly one publish.

    No double publish, no turn-buffer injection, no forwarding attempt.
    """
    import app.services.jobs as jobs_mod


    project = MagicMock()
    project.owner_id = "owner-t3"
    pm = MagicMock()
    pm.get_project.return_value = project

    published: list[tuple] = []

    async def _publish_project(pid, owner, event):
        published.append((pid, owner, event))

    hub = MagicMock()
    hub.publish_project = _publish_project

    # Counting publishes alone cannot distinguish "published once" from "published
    # once AND also injected a frame the relay would publish a second time". Spying on
    # the forwarder closes that gap: on the coordinator it must never be reached, so a
    # regression that forwards from both homes fails here rather than shipping a
    # duplicate render.
    with (
        patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE),
        patch("app.main.get_project_manager", return_value=pm),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch(
            "app.services.project_event_frame.forward_project_event_to_coordinator"
        ) as mock_forward,
    ):
        jobs_mod._publish_project_event(
            "proj-frame-t3", {"resource": "eval_job", "type": "started"}
        )
        # Let async tasks complete
        await asyncio.sleep(0)

    # Exactly one publish on project stream
    assert len(published) == 1, (
        f"coordinator must publish exactly once; got {len(published)}"
    )
    assert published[0][0] == "proj-frame-t3"
    assert published[0][1] == "owner-t3"
    assert published[0][2]["resource"] == "eval_job"
    mock_forward.assert_not_called()


async def test_unroutable_publish_from_microvm_is_stashed(_reset_warned):
    """An unroutable microVM publish is STASHED, not lost.

    This test used to pin a latched warning while the event reached nobody.
    The contract is now the durable outbox: when no live turn can carry the
    event, the publisher stashes it under the owner's partition and the
    coordinator's reconciler drains it — late, never lost. The local publish
    is still attempted, never suppressed.
    """
    import app.services.jobs as jobs_mod

    project = MagicMock()
    project.owner_id = "owner-t4"
    pm = MagicMock()
    pm.get_project.return_value = project

    published: list[tuple] = []

    async def _publish_project(pid, owner, event):
        published.append((pid, owner, event))

    hub = MagicMock()
    hub.publish_project = _publish_project

    stashed: list[tuple] = []

    async def _stash(owner_id, project_id, event):
        stashed.append((owner_id, project_id, event))

    with (
        patch.object(process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE),
        patch("app.main.get_project_manager", return_value=pm),
        patch("app.services.ws_hub.get_ws_hub", return_value=hub),
        patch(
            "app.services.project_event_outbox.stash_unroutable_event",
            side_effect=_stash,
        ),
    ):
        # No turn exists for this project, so forwarding will fail
        jobs_mod._publish_project_event(
            "proj-frame-t4", {"resource": "eval_job", "type": "started"}
        )
        # Let async tasks complete
        await asyncio.sleep(0)

    # The event landed in the outbox with the right identity.
    assert stashed == [
        ("owner-t4", "proj-frame-t4", {"resource": "eval_job", "type": "started"})
    ], f"unroutable event must be stashed; got {stashed}"

    # Local publish still attempted
    assert len(published) == 1, "local publish must not be suppressed"


async def test_mismatched_project_id_rejected():
    """A frame naming a different project is dropped and logged.

    Fail closed: the coordinator publishes nothing on either stream when the frame's
    project_id differs from the relay's own.
    """
    from app.services.project_event_frame import read_project_event_frame

    # Build a frame naming "proj-other"
    data = {
        "project_id": "proj-other",
        "event": {"resource": "eval_job", "type": "started"},
    }

    # Try to decode it expecting "proj-expected"
    with patch("app.services.project_event_frame.logger") as mock_logger:
        result = read_project_event_frame(data, "proj-expected")

    # Must return None
    assert result is None, "mismatched project_id must be rejected"

    # Must log a warning
    warnings = [c for c in mock_logger.warning.call_args_list]
    assert len(warnings) == 1, f"expected warning on mismatch, got {len(warnings)}"
    warning_text = str(warnings[0])
    # The logger call shows truncated ids ([:8] in the code), so check for truncated versions
    assert "proj-oth" in warning_text or "proj-other" in warning_text, (
        f"warning must name mismatched id: {warning_text}"
    )
    assert "proj-exp" in warning_text or "proj-expected" in warning_text, (
        f"warning must name expected id: {warning_text}"
    )


async def test_five_sequential_publishes_in_one_turn_all_reach_project_stream(tmp_path: Path, httpx_mock):
    """Load-bearing: five sequential publishes in ONE turn all reach project:<id> in order.

    It guards the failure where only the first publish of a turn arrives and calls
    2-5 go silent. The PRIMARY
    assertion is the real WsConnection queue — exactly five envelopes with
    stream=project:<id> and kind=project_event, and their run_filename values in the
    SAME ORDER as the five calls. Asserting order as well as count catches a drop,
    a duplicate, AND a reorder.

    Following test_microvm_to_coordinator_boundary_reaches_project_stream's three-stage
    shape exactly: (1) microVM producer appends five frames to turn buffer, (2) real
    _sse_frames emits them across SSE boundary, (3) coordinator drain re-targets them
    onto project:<id> where a real WsConnection receives them.
    """
    from app.services.mcp_tools.eval import delete_eval_run as _delete_run_tool
    delete_eval_run = _delete_run_tool.handler

    project_id = "proj-frame-t6"
    owner_id = "owner-t6"
    turn_id = "turn-t6"

    # Resolve delete_eval_run's project_dir lookup
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    output_dir = project_dir / "output"
    output_dir.mkdir()

    # Create five distinct result files
    filenames = [
        "eval_results_20260101_120000.json",
        "eval_results_20260101_130000.json",
        "eval_results_20260101_140000.json",
        "eval_results_20260101_150000.json",
        "eval_results_20260101_160000.json",
    ]
    for fname in filenames:
        (output_dir / fname).write_text(json.dumps({"models": []}))

    # Stage 1: microVM producer (process role flipped to AgentCore)
    with patch.object(process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE):
        # Create a real turn so the tracker resolution is real
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn(project_id, origin="user", owner_id=owner_id)
        # Override the generated turn_id with our known one for the relay
        turn.turn_id = turn_id
        turn.status = "running"

        # Point delete_eval_run at the test project tree
        with patch(
            "app.services.mcp_tools.eval.resolve_project_dir",
            return_value=project_dir,
        ):
            # Call the REAL delete_eval_run handler five times
            for fname in filenames:
                await delete_eval_run({
                    "project_id": project_id,
                    "run_filename": fname,
                })
                # Yield the loop a few times after each call so fire-and-forget appends land in order
                for _ in range(3):
                    await asyncio.sleep(0)

        # Let any remaining fire-and-forget append tasks run
        for _ in range(5):
            await asyncio.sleep(0)

    # Append a terminal result so _sse_frames terminates
    from app.services.ws_envelope import make_envelope
    result_envelope = make_envelope(
        stream=f"turn:{turn_id}",
        kind="result",
        data={"type": "complete"},
    )
    await turn.append_event(result_envelope)
    await asyncio.sleep(0)

    # Stage 2: the real SSE carrier
    sse_bytes = []
    async for chunk in _sse_frames(turn):
        sse_bytes.append(chunk)
    sse_payload = "".join(sse_bytes)

    # Stage 3: coordinator consumer (role flipped back to coordinator)
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Register a real WsConnection on the real WsHub
        hub = get_ws_hub()
        ws_mock = MagicMock()
        conn = WsConnection(connection_id="conn-t6", user_id=owner_id, websocket=ws_mock)
        hub.register(conn)

        # Relay harness patches (from test_relay_terminal_frame.py)
        _ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:agent-runtime/test"
        _URL = (
            f"https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
            f"{quote(_ARN, safe='')}/invocations?qualifier=DEFAULT"
        )

        # Apply harness patches using contextlib
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(
                "app.services.agentcore_relay_client.resolve_resume",
                AsyncMock(return_value=(None, None)),
            ))
            stack.enter_context(patch("app.services.agentcore_relay_client.record_session", AsyncMock()))
            stack.enter_context(patch("app.services.agentcore_relay_client.release_turn", AsyncMock()))
            stack.enter_context(patch(
                "app.api.chat._stop_agentcore_runtime_session",
                AsyncMock(),
            ))
            stack.enter_context(patch.dict(
                os.environ,
                {"AGENTCORE_RUNTIME_ARN": _ARN, "AWS_REGION": "us-east-1"},
            ))

            # Serve the SSE bytes captured above
            httpx_mock.add_response(
                url=_URL,
                method="POST",
                status_code=200,
                stream=pytest_httpx.IteratorStream([sse_payload.encode()]),
            )

            await invoke_agentcore_turn(
                project_id=project_id,
                message="delete those runs",
                owner_id=owner_id,
                runtime_session_id=None,
                turn_id=turn_id,
                access_token="fake-token",
            )

    # PRIMARY ASSERTION: read the real WsConnection queue
    received = []
    try:
        while True:
            envelope = conn.queue.get_nowait()
            received.append(envelope)
    except asyncio.QueueEmpty:
        pass

    # Find the project_event envelopes on stream project:<id>. The relay
    # also mirrors agent_turn/completed on success — filter it out;
    # the events under test are the five framed eval_run ones.
    project_events = [
        e for e in received
        if e.get("stream") == f"project:{project_id}"
        and e.get("kind") == "project_event"
        and e["data"].get("resource") != "agent_turn"
    ]

    assert len(project_events) == 5, (
        f"expected exactly 5 non-agent_turn project_events on project:{project_id}, "
        f"got {len(project_events)}; received={received}"
    )

    # Assert all five run_filenames are present in the SAME ORDER as the calls
    received_filenames = [e["data"]["run_filename"] for e in project_events]
    assert received_filenames == filenames, (
        f"expected filenames in order {filenames}, got {received_filenames}"
    )
