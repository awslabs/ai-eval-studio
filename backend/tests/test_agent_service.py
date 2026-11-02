# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent turn architecture (dict envelopes).

``ws_envelope.make_envelope`` (dict envelopes) plus ``ws_hub.publish_to_user``
fan-out replaced the older ``format_sse``/``stream_turn_events`` SSE strings.
``turn.events`` is now ``list[dict]`` — every assertion in this
file reads ``e["kind"]`` / ``e["data"]`` instead of substring-matching the
old SSE string.

Test isolation note: ``patch.dict("sys.modules", {"claude_agent_sdk": mock})``
records sys.modules at __enter__ and restores it at __exit__, REMOVING any
keys added during the patch. Numpy's C extension cannot be re-imported
once unloaded ("ImportError: cannot load module more than once per
process"). To avoid contaminating subsequent tests, this module pre-imports
``app.main`` (which transitively imports numpy/pandas via dataset_service)
at module load time so those keys exist in sys.modules BEFORE any test's
patch.dict snapshot, and are therefore preserved on every restore.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import app.main → app.api.eval → app.services.dataset_service →
# pandas → numpy so they're in sys.modules before patch.dict captures it.
import app.main  # noqa: F401
from app.services.agent_runner import run_agent_turn
from app.services.agent_turn import AgentTurn, get_agent_turn_tracker


class TestAgentTurnTracker:
    def setup_method(self):
        get_agent_turn_tracker()._reset()

    def test_create_and_get_turn(self):
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("proj-1")
        assert turn.project_id == "proj-1"
        assert turn.status == "running"
        assert tracker.get_turn(turn.turn_id) is turn

    @pytest.mark.asyncio
    async def test_get_active_turn(self):
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("proj-1")
        assert tracker.get_active_turn("proj-1") is turn
        await turn.complete()
        assert tracker.get_active_turn("proj-1") is None

    def test_new_turn_replaces_old(self):
        tracker = get_agent_turn_tracker()
        old = tracker.create_turn("proj-1")
        new = tracker.create_turn("proj-1")
        assert tracker.get_active_turn("proj-1") is new
        assert old.status == "failed"  # Stale turn marked failed directly


class TestRunAgentTurn:
    @pytest.fixture(autouse=True)
    def _isolate_transcript_write(self):
        """Stub the per-turn transcript append so tests never touch live projects/.

        ``run_agent_turn`` persists each turn to ``<project_dir>/chat/transcript.jsonl``
        via ``_append_turn_to_transcript``. These tests drive it with ``project_id="p1"``,
        which resolves to the real ``projects/p1/`` working tree — so without this guard
        the suite writes a stray transcript into the live projects folder. Individual
        tests already patch ``_persist_session``; this autouse fixture neutralizes the
        sibling transcript side-effect for the whole class (including the error-path test
        that patches nothing), keeping the assertions about turn events/status intact.
        """
        with patch(
            "app.services.agent_runner._append_turn_to_transcript",
            new_callable=AsyncMock,
        ):
            yield

    def _make_result(self, session_id="sess-1"):
        m = MagicMock()
        m.subtype = "success"
        m.result = "Done"
        m.session_id = session_id
        m.total_cost_usd = 0.001
        m.num_turns = 1
        m.is_error = False  # Must be explicit or MagicMock returns truthy
        m.event = None
        m.content = None
        m.tool_use_result = None
        return m

    def _make_text_event(self, text="Hello"):
        m = MagicMock()
        m.event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        m.session_id = None
        m.num_turns = None
        return m

    def _make_tool_event(self, name="describe_data"):
        m = MagicMock()
        m.event = {"type": "content_block_start", "content_block": {"type": "tool_use", "name": name, "id": "t-1"}}
        m.session_id = None
        m.num_turns = None
        return m

    def _mock_sdk(self, messages):
        async def fake_receive():
            for m in messages:
                yield m

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.receive_response = MagicMock(return_value=fake_receive())

        mock_sdk = MagicMock()
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
        mock_sdk.ClaudeAgentOptions = MagicMock()
        return mock_sdk, mock_client

    @pytest.mark.asyncio
    async def test_turn_buffers_text_events(self):
        text_evt = self._make_text_event("Hello world")
        result = self._make_result()
        mock_sdk, _ = self._mock_sdk([text_evt, result])

        turn = AgentTurn(turn_id="t1", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        assert turn.status == "completed"
        text_events = [e for e in turn.events if e["kind"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["data"]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_turn_buffers_tool_events(self):
        tool_evt = self._make_tool_event("describe_data")
        result = self._make_result()
        mock_sdk, _ = self._mock_sdk([tool_evt, result])

        turn = AgentTurn(turn_id="t2", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "analyze", "p1", "test")

        tool_events = [e for e in turn.events if e["kind"] == "tool_start"]
        assert len(tool_events) == 1
        assert tool_events[0]["data"]["name"] == "describe_data"

    @pytest.mark.asyncio
    async def test_turn_persists_session(self):
        result = self._make_result(session_id="sess-saved")
        mock_sdk, _ = self._mock_sdk([result])
        mock_persist = AsyncMock()

        turn = AgentTurn(turn_id="t3", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", mock_persist):
            await run_agent_turn(turn, "hi", "p1", "test")

        mock_persist.assert_awaited_once_with("p1", "sess-saved")
        assert turn.session_id == "sess-saved"

    def _make_text_event_with_parent(self, text: str, parent_tool_use_id: str):
        """Text stream event tagged as subagent-internal (non-None parent)."""
        m = MagicMock()
        m.event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        m.session_id = None
        m.num_turns = None
        m.parent_tool_use_id = parent_tool_use_id
        return m

    @pytest.mark.asyncio
    async def test_subagent_text_is_not_forwarded_to_chat(self):
        """Messages with a non-empty ``parent_tool_use_id`` are subagent-internal;
        they must not surface as ``text_delta`` events in the user-visible stream.
        Only the top-level (parent_tool_use_id == None) coordinator text should.
        """
        top_level = self._make_text_event("Coordinator says hi")
        subagent = self._make_text_event_with_parent("SUBAGENT LEAK", "tool-use-123")
        result = self._make_result()
        mock_sdk, _ = self._mock_sdk([top_level, subagent, result])

        turn = AgentTurn(turn_id="t-sub", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        text_events = [e for e in turn.events if e["kind"] == "text_delta"]
        # Coordinator text passed through; subagent text was filtered.
        texts = [e["data"].get("text", "") for e in text_events]
        assert any("Coordinator says hi" in t for t in texts)
        assert not any("SUBAGENT LEAK" in t for t in texts)

    @pytest.mark.asyncio
    async def test_subagent_tool_result_is_forwarded(self):
        """UserMessage with ``tool_use_result`` carries parent_tool_use_id too,
        but it has no user-visible text — it's the SDK delivering a tool's
        result back into the conversation. The filter must NOT suppress it:
        frontend listens for ``tool_result`` envelopes to trigger mid-turn
        file-tree refreshes after filesystem-mutating tools run.
        """
        tool_result_msg = MagicMock()
        tool_result_msg.event = None  # not a StreamEvent
        tool_result_msg.content = None  # not an AssistantMessage
        tool_result_msg.tool_use_result = {"output": "ok"}
        tool_result_msg.parent_tool_use_id = "tool-use-456"  # from a subagent
        tool_result_msg.session_id = None
        tool_result_msg.num_turns = None

        result = self._make_result()
        mock_sdk, _ = self._mock_sdk([tool_result_msg, result])

        turn = AgentTurn(turn_id="t-subtool", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        tool_result_events = [e for e in turn.events if e["kind"] == "tool_result"]
        assert len(tool_result_events) == 1

    @pytest.mark.asyncio
    async def test_turn_handles_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.query.side_effect = ConnectionError("died")

        mock_sdk = MagicMock()
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
        mock_sdk.ClaudeAgentOptions = MagicMock()

        turn = AgentTurn(turn_id="t4", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            await run_agent_turn(turn, "hi", "p1", "test")

        assert turn.status == "failed"
        # On uncaught exception the runner emits a ``result`` envelope with
        # ``subtype="error"`` (read by the frontend's "result" handler) —
        # see the exception handler in ``run_agent_turn``
        # (app/services/agent_runner.py).
        result_envelopes = [e for e in turn.events if e["kind"] == "result"]
        assert len(result_envelopes) >= 1
        assert result_envelopes[0]["data"]["subtype"] == "error"
        assert "died" in (result_envelopes[0]["data"].get("error") or "")


class TestEvictOldTurns:
    """Tests for AgentTurnTracker._evict_old_turns capacity management."""

    def setup_method(self):
        get_agent_turn_tracker()._reset()

    @pytest.mark.asyncio
    async def test_evicts_oldest_completed_turns_over_capacity(self):
        """Old completed turns are evicted when exceeding max_terminal."""
        tracker = get_agent_turn_tracker()

        # Create 5 turns and complete them
        turn_ids = []
        for i in range(5):
            turn = tracker.create_turn(f"proj-evict-{i}")
            await turn.complete()
            turn_ids.append(turn.turn_id)

        # All should be present before explicit eviction
        assert all(tracker.get_turn(tid) is not None for tid in turn_ids)

        # Evict with max_terminal=2, should remove the 3 oldest
        tracker._evict_old_turns(max_terminal=2)

        evicted = [tid for tid in turn_ids if tracker.get_turn(tid) is None]
        remaining = [tid for tid in turn_ids if tracker.get_turn(tid) is not None]
        assert len(evicted) == 3
        assert len(remaining) == 2

    def test_keeps_running_turns_during_eviction(self):
        """Running turns are never evicted regardless of capacity."""
        tracker = get_agent_turn_tracker()

        # Create a running turn
        running_turn = tracker.create_turn("proj-running")
        assert running_turn.status == "running"

        # Create completed turns
        completed_ids = []
        for i in range(3):
            turn = tracker.create_turn(f"proj-done-{i}")
            turn.status = "completed"
            completed_ids.append(turn.turn_id)

        # Evict with max_terminal=1
        tracker._evict_old_turns(max_terminal=1)

        # Running turn must still be present
        assert tracker.get_turn(running_turn.turn_id) is not None
        assert tracker.get_turn(running_turn.turn_id).status == "running"

    def test_no_eviction_under_capacity(self):
        """No turns evicted when terminal count is within capacity."""
        tracker = get_agent_turn_tracker()

        turn = tracker.create_turn("proj-safe")
        turn.status = "completed"

        tracker._evict_old_turns(max_terminal=50)
        assert tracker.get_turn(turn.turn_id) is not None


class TestEmptySuccessGuard:
    """Turn-reliability guard: an 'empty success' ResultMessage (success
    subtype, no assistant output, no tool activity, empty result, $0 cost)
    means the model was never called; it has been seen on the first resume
    after an AgentCore fleet recycle. One in-place retry; a second empty
    success surfaces as a real error instead of rendering silence."""

    @pytest.fixture(autouse=True)
    def _no_transcript_writes(self):
        with patch(
            "app.services.agent_runner._append_turn_to_transcript",
            new_callable=AsyncMock,
        ):
            yield

    def _make_empty_result(self):
        m = MagicMock()
        m.subtype = "success"
        m.is_error = False
        m.result = ""
        m.session_id = "sess-empty"
        m.total_cost_usd = 0
        m.num_turns = 0
        m.event = None
        m.content = None
        m.tool_use_result = None
        return m

    def _make_good_result(self, session_id="sess-good"):
        m = MagicMock()
        m.subtype = "success"
        m.is_error = False
        m.result = "Done"
        m.session_id = session_id
        m.total_cost_usd = 0.002
        m.num_turns = 1
        m.event = None
        m.content = None
        m.tool_use_result = None
        return m

    def _make_text_event(self, text="Hello"):
        m = MagicMock()
        m.event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        m.session_id = None
        m.num_turns = None
        return m

    def _mock_sdk_attempts(self, *attempts):
        """SDK mock whose receive_response yields a fresh stream per attempt."""
        async def _gen(messages):
            for m in messages:
                yield m

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.receive_response = MagicMock(
            side_effect=[_gen(msgs) for msgs in attempts]
        )

        mock_sdk = MagicMock()
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)
        mock_sdk.ClaudeAgentOptions = MagicMock()
        return mock_sdk, mock_client

    @pytest.mark.asyncio
    async def test_empty_success_retries_once_then_succeeds(self):
        mock_sdk, mock_client = self._mock_sdk_attempts(
            [self._make_empty_result()],
            [self._make_text_event("Second try"), self._make_good_result()],
        )
        mock_persist = AsyncMock()

        turn = AgentTurn(turn_id="t-empty1", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", mock_persist):
            await run_agent_turn(turn, "hi", "p1", "test")

        # Retried: two streams consumed, one retrying status emitted.
        assert mock_client.receive_response.call_count == 2
        retry_events = [
            e for e in turn.events
            if e["kind"] == "status" and e["data"].get("state") == "retrying"
        ]
        assert len(retry_events) == 1

        # The ghost attempt left no trace: session persisted only once,
        # exactly one result event, and it is the good one.
        mock_persist.assert_awaited_once_with("p1", "sess-good")
        result_events = [e for e in turn.events if e["kind"] == "result"]
        assert len(result_events) == 1
        assert result_events[0]["data"]["subtype"] == "success"
        assert turn.status == "completed"

    @pytest.mark.asyncio
    async def test_double_empty_success_surfaces_error(self):
        mock_sdk, mock_client = self._mock_sdk_attempts(
            [self._make_empty_result()],
            [self._make_empty_result()],
        )
        mock_publish_err = AsyncMock()
        mock_persist = AsyncMock()

        turn = AgentTurn(turn_id="t-empty2", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", mock_persist), \
             patch("app.services.agent_runner._publish_cli_error", mock_publish_err):
            await run_agent_turn(turn, "hi", "p1", "test")

        assert mock_client.receive_response.call_count == 2
        # Surfaced as a real error, not silence.
        result_events = [e for e in turn.events if e["kind"] == "result"]
        assert len(result_events) == 1
        assert result_events[0]["data"]["subtype"] == "error"
        assert "empty result" in result_events[0]["data"]["error"]
        mock_publish_err.assert_awaited_once()
        assert mock_publish_err.call_args.kwargs["error_type"] == "empty_result"
        # No session persisted from empty attempts.
        mock_persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_legitimate_zero_output_with_cost_not_retried(self):
        """A turn that billed tokens (cost > 0) is NOT an empty success even
        with no visible output — the model ran; don't second-guess it."""
        billed = self._make_good_result()
        billed.result = ""
        mock_sdk, mock_client = self._mock_sdk_attempts([billed])

        turn = AgentTurn(turn_id="t-billed", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        assert mock_client.receive_response.call_count == 1
        result_events = [e for e in turn.events if e["kind"] == "result"]
        assert len(result_events) == 1
        assert result_events[0]["data"]["subtype"] == "success"

    @pytest.mark.asyncio
    async def test_cli_stderr_callback_registered_and_logs(self):
        mock_sdk, _ = self._mock_sdk_attempts([self._make_good_result()])

        turn = AgentTurn(turn_id="t-stderr", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert callable(kwargs.get("stderr")), "stderr callback must be wired into SDK options"

        # The callback logs each non-empty line at WARNING with the turn prefix.
        with patch("app.services.agent_runner.logger.warning") as warn:
            kwargs["stderr"]("Invalid session ID. Must be a valid UUID.")
            kwargs["stderr"]("   ")  # whitespace-only lines are dropped
        assert warn.call_count == 1
        logged = warn.call_args.args
        assert "[cli-stderr]" in logged[0]
        assert any("Invalid session ID" in str(a) for a in logged)


class TestTranscriptSingleWriter:
    """The transcript has ONE writing process.

    In the AgentCore microVM the turn loop must SKIP its transcript append —
    the coordinator's relay drain owns it. The microVM's rename-based
    tail/record rewrites raced the coordinator's appends (process-local locks
    can't serialize across the boundary) and dropped lines, measured live.
    In-process turns (dev, coordinator-side orchestrator turns) keep the
    append exactly as before.
    """

    @pytest.mark.asyncio
    async def test_microvm_turn_skips_transcript_append(self):
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-mv")
        mock_sdk, _ = harness._mock_sdk([result])
        turn = AgentTurn(turn_id="t-mv", project_id="p1")
        # Patch by STRING target: earlier tests' sys.modules patch.dict blocks
        # evict lazily-imported modules on exit, so a `from app.platform
        # import process_role` here can hand back a STALE module object while
        # run_agent_turn re-imports a fresh one (this file's module docstring
        # documents the trap). The string target makes mock import/patch the
        # same instance the code under test resolves.
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock) as mock_append, \
             patch("app.platform.process_role._PROCESS_ROLE", "agentcore_runtime"):
            await run_agent_turn(turn, "hi", "p1", "test")
        assert turn.status == "completed"
        mock_append.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_in_process_turn_still_appends_transcript(self):
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-co")
        mock_sdk, _ = harness._mock_sdk([result])
        turn = AgentTurn(turn_id="t-co", project_id="p1")
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock) as mock_append:
            await run_agent_turn(turn, "hi", "p1", "test")
        assert turn.status == "completed"
        mock_append.assert_awaited_once_with(
            "p1", "sess-co", final_answer="Done", turn_id="t-co"
        )


class TestForegroundSubagentGuard:
    """The model may not run subagents in the background.

    On the pinned SDK a background Task dies silently when the turn's result
    frame closes stdin (SDK #1103: the subagent reports 'started' and the turn
    ends seconds later). The bundled CLI ALSO ships a remote-flag-gated
    background-by-default mode where an OMITTED flag means async; the same
    binary can switch into it with zero local changes, and then every
    delegation silently breaks. Stripping the
    flag only handles the first hazard; only an EXPLICIT false opts out of
    both. The PreToolUse guard therefore FORCES run_in_background=False on
    every Task call — allow-with-updatedInput, never a deny (the subagent
    itself is wanted, just not in the background).
    """

    @pytest.mark.asyncio
    async def test_background_task_forced_synchronous(self):
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-fg")
        mock_sdk, mock_client = harness._mock_sdk([result])
        turn = AgentTurn(turn_id="t-fg", project_id="p1")
        captured_hooks = {}

        def capture_options(**kwargs):
            captured_hooks.update(kwargs.get("hooks") or {})
            return MagicMock()

        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
        # HookMatcher must preserve its constructor args so the test can
        # reach the real hook function (a bare MagicMock would swallow them).
        from types import SimpleNamespace
        mock_sdk.HookMatcher = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
            )
        )
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        # Find the PreToolUse hook and drive it with a background Task call.
        pre_hooks = captured_hooks.get("PreToolUse")
        assert pre_hooks, f"PreToolUse hooks must be registered: {list(captured_hooks)}"
        hook_fn = pre_hooks[0].hooks[0]
        out = await hook_fn(
            {
                "tool_name": "Task",
                "tool_input": {
                    "subagent_type": "harness-builder",
                    "prompt": "build it",
                    "run_in_background": True,
                },
            },
            None,
            None,
        )
        hso = out.get("hookSpecificOutput") or {}
        assert hso.get("permissionDecision") == "allow"
        updated = hso.get("updatedInput")
        # Explicit false, not absence: under the CLI's background-by-default
        # mode an omitted flag means async — only false opts out everywhere.
        assert updated is not None and updated.get("run_in_background") is False
        assert updated.get("subagent_type") == "harness-builder"

        # A Task WITHOUT the flag gets it forced to false too (an omitted
        # flag means async under the flag-gated CLI default).
        out2 = await hook_fn(
            {
                "tool_name": "Task",
                "tool_input": {"subagent_type": "data-analyzer", "prompt": "x"},
            },
            None,
            None,
        )
        hso2 = out2.get("hookSpecificOutput") or {}
        assert hso2.get("permissionDecision") == "allow"
        updated2 = hso2.get("updatedInput")
        assert updated2 is not None and updated2.get("run_in_background") is False
        assert updated2.get("prompt") == "x"

        # Already-explicit false passes through untouched (no rewrite churn).
        out3 = await hook_fn(
            {
                "tool_name": "Task",
                "tool_input": {
                    "subagent_type": "data-analyzer",
                    "prompt": "y",
                    "run_in_background": False,
                },
            },
            None,
            None,
        )
        assert out3 == {}

    async def _setup_and_capture_hook(self):
        """Shared setup: run a turn, capture the PreToolUse hook, return it."""
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-guard")
        mock_sdk, _ = harness._mock_sdk([result])
        turn = AgentTurn(turn_id="t-guard", project_id="p1")
        captured_hooks = {}

        def capture_options(**kwargs):
            captured_hooks.update(kwargs.get("hooks") or {})
            return MagicMock()

        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
        from types import SimpleNamespace
        mock_sdk.HookMatcher = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
            )
        )
        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock):
            await run_agent_turn(turn, "hi", "p1", "test")

        pre_hooks = captured_hooks.get("PreToolUse")
        assert pre_hooks, f"PreToolUse hooks must be registered: {list(captured_hooks)}"
        return pre_hooks[0].hooks[0]

    @pytest.mark.asyncio
    async def test_agent_named_delegation_forced_synchronous(self):
        """Agent-named delegations get run_in_background forced to False.

        Prevents silent failure when the CLI's background-by-default mode
        turns on and the delegation tool arrives as "Agent" instead of "Task".
        A backgrounded subagent never reports back — the coordinator promises
        findings in the same turn, which can never be fulfilled.
        """
        hook_fn = await self._setup_and_capture_hook()

        # Case 1: Agent with missing flag → forced to False
        out = await hook_fn(
            {
                "tool_name": "Agent",
                "tool_input": {
                    "subagent_type": "data-analyzer",
                    "prompt": "analyze",
                },
            },
            None,
            None,
        )
        hso = out.get("hookSpecificOutput") or {}
        assert hso.get("permissionDecision") == "allow", \
            "Agent delegation with missing flag must be allowed with updatedInput"
        updated = hso.get("updatedInput")
        assert updated is not None, "Agent delegation with missing flag must receive updatedInput"
        assert updated.get("run_in_background") is False, \
            "Agent delegation must have run_in_background forced to False when omitted"
        assert updated.get("subagent_type") == "data-analyzer", \
            "Other input keys must be preserved when forcing run_in_background"

        # Case 2: Agent with explicit True → forced to False
        out2 = await hook_fn(
            {
                "tool_name": "Agent",
                "tool_input": {
                    "subagent_type": "prompt-optimizer",
                    "prompt": "optimize",
                    "run_in_background": True,
                },
            },
            None,
            None,
        )
        hso2 = out2.get("hookSpecificOutput") or {}
        assert hso2.get("permissionDecision") == "allow", \
            "Agent delegation with explicit background=True must be allowed with updatedInput"
        updated2 = hso2.get("updatedInput")
        assert updated2 is not None and updated2.get("run_in_background") is False, \
            "Agent delegation must have run_in_background forced to False even when model requested True"
        assert updated2.get("prompt") == "optimize", \
            "Other keys preserved when overriding explicit background request"

        # Case 3: Agent with explicit False → passthrough (no rewrite churn)
        out3 = await hook_fn(
            {
                "tool_name": "Agent",
                "tool_input": {
                    "subagent_type": "harness-builder",
                    "prompt": "build",
                    "run_in_background": False,
                },
            },
            None,
            None,
        )
        assert out3 == {}, \
            "Agent delegation with explicit False must pass through unchanged (no rewrite churn)"

    @pytest.mark.asyncio
    async def test_non_delegation_tool_input_untouched(self):
        """Non-delegation tools are not rewritten by the forced-sync guard.

        The guard only touches Task and Agent; other tools (Read, WebSearch, etc.)
        must never receive a synthetic run_in_background key.
        """
        hook_fn = await self._setup_and_capture_hook()

        # Drive with WebSearch (a non-delegation tool with no path arg)
        out = await hook_fn(
            {
                "tool_name": "WebSearch",
                "tool_input": {"query": "Bedrock pricing"},
            },
            None,
            None,
        )
        assert out == {}, \
            "Non-delegation tools must not be rewritten — the hook must return empty"

    @pytest.mark.asyncio
    async def test_guard_logs_every_rewrite(self):
        """Every forced-sync rewrite is logged; explicit background requests escalate to WARNING.

        Silent rewrites (missing flag) log at INFO; explicit background requests
        (model requested True) log at WARNING. This ensures operational visibility
        for the guard's actions — when only the explicit case logged, silent
        rewrites left no trace at all.
        """
        hook_fn = await self._setup_and_capture_hook()

        # Patch the logger at module scope rather than using caplog, which reads
        # global logging state that other tests can change.
        import app.services.agent_runner as agent_runner_module
        with patch.object(agent_runner_module, "logger") as mock_logger:
            # Case 1: Silent rewrite (missing flag) → INFO, no WARNING
            await hook_fn(
                {
                    "tool_name": "Agent",
                    "tool_input": {
                        "subagent_type": "data-analyzer",
                        "prompt": "analyze",
                    },
                },
                None,
                None,
            )
            assert mock_logger.info.call_count == 1, \
                "Silent rewrite (missing flag) must log exactly once at INFO"
            assert mock_logger.warning.call_count == 0, \
                "Silent rewrite must NOT log at WARNING"
            # Check that INFO log contains relevant context
            info_call_args = mock_logger.info.call_args
            assert "Agent" in str(info_call_args), \
                "INFO log must mention the tool name (Agent)"
            mock_logger.reset_mock()

            # Case 2: Explicit background request → WARNING
            await hook_fn(
                {
                    "tool_name": "Task",
                    "tool_input": {
                        "subagent_type": "prompt-optimizer",
                        "prompt": "optimize",
                        "run_in_background": True,
                    },
                },
                None,
                None,
            )
            assert mock_logger.warning.call_count == 1, \
                "Explicit background request must log at WARNING"
            # Check that WARNING log contains relevant context
            warning_call_args = mock_logger.warning.call_args
            assert "background" in str(warning_call_args).lower(), \
                "WARNING log must mention that background was requested"


class TestOrphanedSubagentGuard:
    """Orphaned subagent detection via start/stop pairing.

    A synchronous subagent always stops before the turn's result; a backgrounded
    one never does. The turn architecture assumes delegation is synchronous — the
    coordinator promises "I'll report the findings" in the same turn. An
    unmatched SubagentStart at result time means the delegated work never
    happened, and the turn must fail loudly instead of reporting success.
    """

    @pytest.mark.asyncio
    async def test_unmatched_subagent_start_fails_the_turn(self):
        """An unmatched SubagentStart at result time fails the turn.

        Prevents a turn that reports success while the delegated work never ran.
        The user sees an ERROR log, a toast, and turn.status == "failed".
        """
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-orphan")
        turn = AgentTurn(turn_id="t-orphan", project_id="p1")
        captured_hooks = {}

        def capture_options(**kwargs):
            captured_hooks.update(kwargs.get("hooks") or {})
            return MagicMock()

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
        from types import SimpleNamespace
        mock_sdk.HookMatcher = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
            )
        )

        # Custom message generator that calls hooks before yielding result
        async def fake_messages():
            # Yield a text event first to avoid empty-success guard
            text_evt = harness._make_text_event("Analyzing...")
            yield text_evt
            # Call the SubagentStart hook before the result
            if "SubagentStart" in captured_hooks:
                start_hook = captured_hooks["SubagentStart"][0].hooks[0]
                await start_hook({"agent_type": "data-analyzer"}, None, None)
            # No SubagentStop hook call (unmatched)
            yield result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.receive_response = MagicMock(return_value=fake_messages())
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock), \
             patch("app.services.agent_runner._publish_cli_error", new_callable=AsyncMock) as mock_publish:
            await run_agent_turn(turn, "delegate", "p1", "test")

        # Assert the turn failed
        assert turn.status == "failed", \
            f"Turn with unmatched SubagentStart must fail, got {turn.status}"
        assert turn.error is not None, \
            "Turn must have an attached error when failing on orphaned subagent"
        # Assert the error was published with correct type
        mock_publish.assert_awaited_once()
        assert mock_publish.call_args.kwargs["error_type"] == "orphaned_subagent", \
            f"Error type must be 'orphaned_subagent', got {mock_publish.call_args.kwargs.get('error_type')}"

    @pytest.mark.asyncio
    async def test_paired_start_and_stop_completes_normally(self):
        """Paired SubagentStart + SubagentStop completes the turn successfully.

        Ensures the pairing counter does not false-positive on healthy delegations.
        """
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-paired")
        turn = AgentTurn(turn_id="t-paired", project_id="p1")
        captured_hooks = {}

        def capture_options(**kwargs):
            captured_hooks.update(kwargs.get("hooks") or {})
            return MagicMock()

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
        from types import SimpleNamespace
        mock_sdk.HookMatcher = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
            )
        )

        # Custom message generator that calls both start and stop hooks
        async def fake_messages():
            # Yield a text event first to avoid empty-success guard
            text_evt = harness._make_text_event("Building harness...")
            yield text_evt
            # Call SubagentStart hook
            if "SubagentStart" in captured_hooks:
                start_hook = captured_hooks["SubagentStart"][0].hooks[0]
                await start_hook({"agent_type": "harness-builder"}, None, None)
            # Call SubagentStop hook (paired)
            if "SubagentStop" in captured_hooks:
                stop_hook = captured_hooks["SubagentStop"][0].hooks[0]
                await stop_hook({"agent_type": "harness-builder"}, None, None)
            yield result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.receive_response = MagicMock(return_value=fake_messages())
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock), \
             patch("app.services.agent_runner._publish_cli_error", new_callable=AsyncMock) as mock_publish:
            await run_agent_turn(turn, "delegate", "p1", "test")

        # Assert the turn completed successfully
        assert turn.status == "completed", \
            f"Turn with paired SubagentStart/Stop must complete normally, got {turn.status}"
        # Assert no error was published (use assert_not_awaited for AsyncMock)
        try:
            mock_publish.assert_not_awaited()
        except AssertionError:
            # If it was awaited, show details
            call_info = str(mock_publish.call_args) if mock_publish.call_args else "unknown"
            raise AssertionError(f"Paired subagent must NOT trigger _publish_cli_error, but was: {call_info}")

    @pytest.mark.asyncio
    async def test_user_stopped_turn_does_not_trip_the_guard(self):
        """A user-cancelled turn does not trip the orphaned subagent guard.

        Even if a SubagentStart fired, a user cancel must not publish an error or
        fail the turn on the orphan path — the cancellation is intentional.
        """
        harness = TestRunAgentTurn()
        result = harness._make_result(session_id="sess-cancel")
        turn = AgentTurn(turn_id="t-cancel", project_id="p1")
        # Mark turn as user-stopped BEFORE running
        turn._stopped_by_user = True
        captured_hooks = {}

        def capture_options(**kwargs):
            captured_hooks.update(kwargs.get("hooks") or {})
            return MagicMock()

        mock_sdk = MagicMock()
        mock_sdk.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: capture_options(**kw))
        from types import SimpleNamespace
        mock_sdk.HookMatcher = MagicMock(
            side_effect=lambda **kw: SimpleNamespace(
                matcher=kw.get("matcher"), hooks=kw.get("hooks") or []
            )
        )

        # Custom message generator that calls SubagentStart but not Stop
        async def fake_messages():
            # Yield a text event first to avoid empty-success guard
            text_evt = harness._make_text_event("Optimizing...")
            yield text_evt
            # Call SubagentStart hook
            if "SubagentStart" in captured_hooks:
                start_hook = captured_hooks["SubagentStart"][0].hooks[0]
                await start_hook({"agent_type": "prompt-optimizer"}, None, None)
            # No SubagentStop (unmatched), but user cancelled
            yield result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.receive_response = MagicMock(return_value=fake_messages())
        mock_sdk.ClaudeSDKClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}), \
             patch("app.services.agent_runner._persist_session", new_callable=AsyncMock), \
             patch("app.services.agent_runner._append_turn_to_transcript", new_callable=AsyncMock), \
             patch("app.services.agent_runner._publish_cli_error", new_callable=AsyncMock) as mock_publish:
            await run_agent_turn(turn, "delegate", "p1", "test")

        # Assert no error was published (use assert_not_awaited for AsyncMock)
        try:
            mock_publish.assert_not_awaited()
        except AssertionError:
            call_info = str(mock_publish.call_args) if mock_publish.call_args else "unknown"
            raise AssertionError(f"User-cancelled turn must NOT trigger _publish_cli_error, but was: {call_info}")
        # The turn status may vary (cancelled vs completed), but the important
        # invariant is that the orphan guard did not fire
