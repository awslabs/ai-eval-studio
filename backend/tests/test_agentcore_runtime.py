# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentCore runtime serving app.

Verifies the /ping + /invocations SSE contract, that the runtime advertises no
CORS policy, and that the adapter drains the AgentTurn buffer (not a
reimplemented SDK loop).
"""
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def runtime_app():
    """Import and return the agentcore_runtime app."""
    from app.agentcore_runtime import app
    return app


@pytest.fixture
def client(runtime_app):
    """Create a test client for the runtime app."""
    return TestClient(runtime_app)


class TestPing:
    """Test the GET /ping health endpoint."""

    def test_ping_returns_healthy_without_sdk(self, client):
        """GET /ping returns 200 {status:Healthy} and never imports/starts the SDK."""
        with patch("app.agentcore_runtime.start_agent_turn") as mock_start:
            response = client.get("/ping")

        assert response.status_code == 200
        assert response.json() == {"status": "Healthy"}
        # Verify no turn was started
        mock_start.assert_not_called()

    def test_ping_does_not_advance_idle_timer_field(self):
        """GET /ping does not write a per-ping idle-timer field.

        A concrete offline code check: the `/ping` handler must not advance a
        per-ping `time_of_last_update` / `last_update` field, or sessions never
        idle-terminate and exhaust quota. This is a GUARD test: it pins the
        current behavior (the handler answers Healthy or HealthyBusy from
        in-flight work and writes no timestamp) so a future adapter change that
        starts writing a per-ping timestamp is caught offline.

        The robust form is a source-level assertion: the handler source contains
        no `time_of_last_update` write. This is load-bearing because the `/ping`
        HTTP response legitimately has no such field to inspect — the check must
        verify the handler does not WRITE the field that AgentCore's idle timer
        keys on.
        """
        import inspect
        from app import agentcore_runtime

        # Get the ping handler source code
        ping_source = inspect.getsource(agentcore_runtime.ping)

        # Assert the handler contains no time_of_last_update assignment
        assert "time_of_last_update" not in ping_source, (
            "ping handler must not write time_of_last_update per-ping "
            "(advancing it defeats the idle timer)"
        )
        assert "last_update" not in ping_source, (
            "ping handler must not write last_update per-ping "
            "(advancing it defeats the idle timer)"
        )


class TestInvocationsProbe:
    """Test the POST /invocations fast-probe path."""

    def test_probe_body_returns_fast_without_turn(self, client):
        """POST /invocations with empty/trivial probe body returns 200 quickly WITHOUT starting a turn."""
        with patch("app.agentcore_runtime.start_agent_turn") as mock_start:
            # Empty body probe
            response = client.post("/invocations", json={})

        assert response.status_code == 200
        # Verify no turn was started (fast readiness path)
        mock_start.assert_not_called()

    def test_probe_body_with_probe_flag(self, client):
        """POST /invocations with {probe:true} returns fast without turn."""
        with patch("app.agentcore_runtime.start_agent_turn") as mock_start:
            response = client.post("/invocations", json={"probe": True})

        assert response.status_code == 200
        mock_start.assert_not_called()

    def test_metrics_probe_returns_memory_report_without_turn(self, client):
        """POST /invocations {probe:metrics} returns the memory self-report as plain
        JSON and starts NO turn (peak RSS against the 8 GB cap)."""
        with patch("app.agentcore_runtime.start_agent_turn") as mock_start:
            with patch(
                "app.agentcore_runtime._memory_report",
                return_value={
                    "cgroup_peak_bytes": 123,
                    "cgroup_current_bytes": 45,
                    "vmhwm_bytes": 67,
                },
            ):
                response = client.post("/invocations", json={"probe": "metrics"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert set(body) == {"cgroup_peak_bytes", "cgroup_current_bytes", "vmhwm_bytes"}
        assert body["cgroup_peak_bytes"] == 123
        # Must NOT run a coordinator turn — this is a metrics probe, not a payload.
        mock_start.assert_not_called()


def _fake_terminal_turn():
    """Build a fake AgentTurn whose drain exits immediately (terminal status)."""
    fake_turn = Mock()
    fake_turn.turn_id = "turn-123"
    fake_turn.status = "completed"  # terminal so _sse_frames exits at once
    fake_turn.events = []
    fake_turn.events_dropped = 0
    fake_turn.session_id = None  # AgentTurn.session_id is None until SDK result
    fake_turn.wait_for_events = AsyncMock(return_value=False)
    return fake_turn


def _patch_authorized_platform():
    """Patch get_platform so extract_user_id succeeds and require_access allows."""
    mock_auth = Mock()
    mock_auth.extract_user_id = AsyncMock(return_value="user-1")
    mock_auth.require_access = AsyncMock()
    mock_platform = patch("app.agentcore_runtime.get_platform")
    started = mock_platform.start()
    started.return_value.auth = mock_auth
    return mock_platform, mock_auth


class TestInvocationsSSE:
    """Test the POST /invocations SSE streaming path."""

    def test_real_payload_returns_text_event_stream(self, client):
        """POST /invocations with {message, project_id} returns text/event-stream.

        Uses autospec on start_agent_turn so the call is validated against the
        REAL signature — a wrong-arg call would raise TypeError here,
        not silently pass as it did when start_agent_turn was a bare Mock.
        """
        fake_turn = _fake_terminal_turn()
        fake_project = Mock()
        fake_project.project_id = "proj-abc"

        mock_platform_cm, _ = _patch_authorized_platform()
        try:
            with patch(
                "app.agentcore_runtime.start_agent_turn",
                autospec=True,
                return_value=fake_turn,
            ) as mock_start:
                with patch("app.main.get_project_manager") as mock_pm:
                    mock_pm.return_value.get_project.return_value = fake_project
                    with patch("app.agentcore_runtime.build_coordinator_prompt", return_value="prompt"):
                        with patch("app.agentcore_runtime.resolve_project_dir", return_value="/fake/proj"):
                            response = client.post(
                                "/invocations",
                                json={"message": "test", "project_id": "proj-abc"},
                            )
        finally:
            mock_platform_cm.stop()

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        # Guard: the adapter called the turn engine with a signature the
        # real start_agent_turn accepts (autospec validated the bind). Assert
        # the positional/kw contract we depend on.
        mock_start.assert_called_once()
        _args, _kwargs = mock_start.call_args
        bound = (list(_args), _kwargs)
        # project_id + message + coordinator_prompt reach the engine; session_id
        # is passed as the resume handle.
        assert "proj-abc" in bound[0] or bound[1].get("project_id") == "proj-abc"
        assert "test" in bound[0] or bound[1].get("prompt") == "test"
        assert "session_id" in bound[1] or len(bound[0]) >= 4

    def test_missing_auth_is_rejected_401(self, client):
        """No/invalid credentials -> 401, and NO turn is started (fail-closed)."""
        mock_auth = Mock()
        # extract_user_id raises 401 like the cloud auth does for a missing token
        from fastapi import HTTPException
        mock_auth.extract_user_id = AsyncMock(side_effect=HTTPException(status_code=401, detail="Unauthorized"))
        with patch("app.agentcore_runtime.get_platform") as mock_platform:
            mock_platform.return_value.auth = mock_auth
            with patch("app.agentcore_runtime.start_agent_turn", autospec=True) as mock_start:
                response = client.post(
                    "/invocations",
                    json={"message": "test", "project_id": "proj-abc"},
                )

        assert response.status_code == 401
        mock_start.assert_not_called()

    def test_access_denied_is_rejected_403(self, client):
        """Authenticated user without project access -> 403, and NO turn is started."""
        from fastapi import HTTPException
        mock_auth = Mock()
        mock_auth.extract_user_id = AsyncMock(return_value="user-1")
        mock_auth.require_access = AsyncMock(side_effect=HTTPException(status_code=403, detail="Access denied"))
        with patch("app.agentcore_runtime.get_platform") as mock_platform:
            mock_platform.return_value.auth = mock_auth
            with patch("app.agentcore_runtime.start_agent_turn", autospec=True) as mock_start:
                response = client.post(
                    "/invocations",
                    json={"message": "test", "project_id": "proj-other-tenant"},
                )

        assert response.status_code == 403
        mock_start.assert_not_called()

    def test_session_mapping_recorded_with_resolved_session(self, client):
        """When a runtimeSessionId + resumable session resolve, the mapping is
        recorded using the RESOLVED session_id, not turn.session_id.

        turn.session_id is None at turn-start, so keying the mapping off it
        (the original bug) never recorded anything. This asserts the mapping
        is written with the resolved resume handle and the owning project_id.
        """
        fake_turn = _fake_terminal_turn()  # .session_id is None, as in production
        fake_project = Mock()
        fake_project.project_id = "proj-abc"

        mock_platform_cm, _ = _patch_authorized_platform()
        try:
            with patch("app.agentcore_runtime.start_agent_turn", autospec=True, return_value=fake_turn):
                with patch("app.main.get_project_manager") as mock_pm:
                    mock_pm.return_value.get_project.return_value = fake_project
                    with patch("app.agentcore_runtime.build_coordinator_prompt", return_value="prompt"):
                        with patch("app.agentcore_runtime.resolve_project_dir", return_value="/fake/proj"):
                            # resolver returns a concrete resume handle
                            with patch(
                                "app.agentcore_runtime.resume_session_id_for",
                                return_value="resolved-session-xyz",
                            ):
                                with patch("app.agentcore_runtime.record_session_mapping") as mock_record:
                                    response = client.post(
                                        "/invocations",
                                        headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "rt-42"},
                                        json={"message": "test", "project_id": "proj-abc"},
                                    )
        finally:
            mock_platform_cm.stop()

        assert response.status_code == 200
        # Recorded with (runtime_session_id, resolved_session_id, project_id) —
        # NOT turn.session_id (which is None).
        mock_record.assert_called_once_with("rt-42", "resolved-session-xyz", "proj-abc")

    def test_drain_seam_yields_sse_frames_in_order(self):
        """_sse_frames yields one SSE frame per buffered envelope IN ORDER via wait_for_events + .events."""
        from app.agentcore_runtime import _sse_frames

        # Create a fake turn with pre-seeded events
        fake_turn = Mock()
        fake_turn.turn_id = "turn-drain-test"
        fake_turn.status = "completed"  # Terminal status so drain exits
        fake_turn.events_dropped = 0
        fake_turn.events = [
            {"stream": "turn:123", "seq": 0, "kind": "text_delta", "data": {"text": "hello"}},
            {"stream": "turn:123", "seq": 1, "kind": "tool_start", "data": {"tool": "Read"}},
            {"stream": "turn:123", "seq": 2, "kind": "tool_result", "data": {"result": "ok"}},
            {"stream": "turn:123", "seq": 3, "kind": "result", "data": {"final": True}},
        ]

        # Mock wait_for_events to return immediately (events already buffered)
        async def mock_wait_for_events(known_count, timeout):
            # Simulate the buffer already having more events
            return len(fake_turn.events) > known_count

        fake_turn.wait_for_events = mock_wait_for_events

        # Collect frames
        async def collect():
            frames = []
            async for frame in _sse_frames(fake_turn):
                frames.append(frame)
            return frames

        frames = asyncio.run(collect())

        # Verify one SSE frame per envelope, in order
        assert len(frames) >= 4  # At least the 4 seeded events
        # Check SSE format (event: <kind>\ndata: <json>\n\n)
        assert "event: text_delta" in frames[0]
        assert '"text": "hello"' in frames[0]
        assert "event: tool_start" in frames[1]
        assert '"tool": "Read"' in frames[1]
        assert "event: tool_result" in frames[2]
        assert "event: result" in frames[3]


class TestInvocationsRequestContext:
    """The microVM populates request_context so in-process MCP eval tools authorize.

    Regression guard for the relay-on eval-tool breakage: the standalone
    AgentCore serving app does NOT mount app.main's auth middleware, so nothing
    set the per-request ContextVar. The SDK MCP eval tools authorize via
    app.platform.mcp_authz, which reads request_context and returns
    mcp_error("not authenticated") when it is unset — so EVERY eval tool failed
    inside the microVM when the relay flag was on. The handler now sets the
    ContextVar (from the already-resolved user_id + project_id) before
    start_agent_turn, whose create_task copies it into the turn task where the
    tools run.
    """

    def test_request_context_is_set_when_turn_starts(self, client):
        """At start_agent_turn call time, request_context carries the auth'd user+project.

        That is the context the spawned turn task (and its MCP tools) inherits
        via create_task, so this is exactly what mcp_authz would see. Without the
        fix the captured context is None (the "not authenticated" cause).
        """
        from app.platform.types import request_context

        fake_turn = _fake_terminal_turn()
        fake_project = Mock()
        fake_project.project_id = "proj-abc"

        captured = {}

        def _capture_ctx(*_args, **_kwargs):
            # Snapshot the ContextVar AT CALL TIME — this is what create_task
            # copies into run_agent_turn (and every MCP tool it invokes).
            captured["ctx"] = request_context.get()
            return fake_turn

        mock_platform_cm, _ = _patch_authorized_platform()
        try:
            with patch(
                "app.agentcore_runtime.start_agent_turn",
                autospec=True,
                side_effect=_capture_ctx,
            ):
                with patch("app.main.get_project_manager") as mock_pm:
                    mock_pm.return_value.get_project.return_value = fake_project
                    with patch("app.agentcore_runtime.build_coordinator_prompt", return_value="prompt"):
                        with patch("app.agentcore_runtime.resolve_project_dir", return_value="/fake/proj"):
                            response = client.post(
                                "/invocations",
                                json={"message": "test", "project_id": "proj-abc"},
                            )
        finally:
            mock_platform_cm.stop()

        assert response.status_code == 200
        ctx = captured.get("ctx")
        # The bug: ctx is None here (no middleware set it) → mcp_authz returns
        # "not authenticated" for every eval tool. The fix populates it.
        assert ctx is not None, (
            "request_context was not set at turn start — MCP eval tools would see "
            "'not authenticated' inside the microVM"
        )
        assert ctx.user_id == "user-1"
        assert ctx.project_id == "proj-abc"

    def test_request_context_does_not_leak_after_handler_returns(self, client):
        """The handler resets its own context view; the ContextVar does not leak.

        The turn task keeps its own copied snapshot (create_task), but the
        handler task must not leave request_context set after the response.
        """
        from app.platform.types import request_context

        fake_turn = _fake_terminal_turn()
        fake_project = Mock()
        fake_project.project_id = "proj-abc"

        mock_platform_cm, _ = _patch_authorized_platform()
        try:
            with patch(
                "app.agentcore_runtime.start_agent_turn",
                autospec=True,
                return_value=fake_turn,
            ):
                with patch("app.main.get_project_manager") as mock_pm:
                    mock_pm.return_value.get_project.return_value = fake_project
                    with patch("app.agentcore_runtime.build_coordinator_prompt", return_value="prompt"):
                        with patch("app.agentcore_runtime.resolve_project_dir", return_value="/fake/proj"):
                            response = client.post(
                                "/invocations",
                                json={"message": "test", "project_id": "proj-abc"},
                            )
        finally:
            mock_platform_cm.stop()

        assert response.status_code == 200
        # The test process's own context must be clean (the reset ran).
        assert request_context.get() is None


class TestNoCORS:
    """The runtime answers no browser, so it advertises no CORS policy."""

    def test_options_invocations_is_not_routed(self, client):
        """OPTIONS /invocations is unhandled and no allow-origin header is sent.

        The only caller signs its POST with SigV4 server-side, where CORS has no
        effect. An allow-origin header here would widen what a browser may do
        with this response for no caller that needs it.
        """
        response = client.options(
            "/invocations",
            headers={"Origin": "https://example.cloudfront.net"},
        )

        assert response.status_code == 405
        assert "access-control-allow-origin" not in response.headers


class TestMemoryReport:
    """Test the cgroup-first memory self-report reader."""

    def test_cgroup_v2_preferred(self):
        """cgroup v2 memory.peak/current are used when present."""
        from app import agentcore_runtime as m

        def fake_read(path):
            return {
                m._CGROUP_V2_PEAK: 8_000_000_000,
                m._CGROUP_V2_CURRENT: 2_000_000_000,
            }.get(path)

        with patch("app.agentcore_runtime._read_int_file", side_effect=fake_read):
            with patch("app.agentcore_runtime._read_vmhwm_bytes", return_value=500_000_000):
                report = m._memory_report()

        assert report["cgroup_peak_bytes"] == 8_000_000_000
        assert report["cgroup_current_bytes"] == 2_000_000_000
        assert report["vmhwm_bytes"] == 500_000_000

    def test_falls_back_to_cgroup_v1_then_vmhwm(self):
        """When cgroup v2 is absent, fall back to v1; peak degrades to VmHWM if no cgroup."""
        from app import agentcore_runtime as m

        # No cgroup files at all -> peak falls back to VmHWM, current stays None.
        with patch("app.agentcore_runtime._read_int_file", return_value=None):
            with patch("app.agentcore_runtime._read_vmhwm_bytes", return_value=333):
                report = m._memory_report()

        assert report["cgroup_peak_bytes"] == 333  # VmHWM last-resort peak
        assert report["cgroup_current_bytes"] is None
        assert report["vmhwm_bytes"] == 333

    def test_cgroup_v1_peak_used_when_v2_missing(self):
        """cgroup v1 memory.max_usage_in_bytes is used when v2 peak is missing."""
        from app import agentcore_runtime as m

        def fake_read(path):
            # v2 paths return None; v1 peak present.
            return {m._CGROUP_V1_PEAK: 7_500_000_000}.get(path)

        with patch("app.agentcore_runtime._read_int_file", side_effect=fake_read):
            with patch("app.agentcore_runtime._read_vmhwm_bytes", return_value=100):
                report = m._memory_report()

        assert report["cgroup_peak_bytes"] == 7_500_000_000


class TestSseHeartbeat:
    """The drain emits REAL heartbeat frames on quiet turns.

    SSE comments keep the socket alive but the coordinator's parser never
    yields them — its frame-idle watchdog couldn't distinguish a quietly
    working turn from a dead session. The heartbeat frame is only produced
    while this process drains a live turn, so its absence is a true death
    signal.
    """

    @pytest.mark.asyncio
    async def test_quiet_turn_yields_heartbeat_frame(self):
        from app.agentcore_runtime import _sse_frames

        turn = Mock()
        turn.turn_id = "t-heartbeat"
        turn.events = []
        turn.events_dropped = 0
        turn.status = "running"
        calls = {"n": 0}

        async def _wait_for_events(known, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return False  # timeout branch → heartbeat expected
            turn.status = "completed"  # then terminate the drain
            return False

        turn.wait_for_events = _wait_for_events

        frames = [f async for f in _sse_frames(turn)]
        assert any(f.startswith("event: heartbeat\n") for f in frames), (
            f"quiet turn must emit a heartbeat FRAME (not a comment): {frames}"
        )
        assert not any(f.startswith(": keepalive") for f in frames), (
            "the comment keepalive was replaced by the heartbeat frame"
        )
