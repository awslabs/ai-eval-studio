# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Job Manager architecture.

Exercises the turn queue routing and job cancellation semantics
that are not covered by the existing MCP tool-level tests.
"""

import asyncio
import pytest

from .async_wait import until
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.agent_turn import AgentTurn, get_agent_turn_tracker
from app.services.completion_turn import request_completion_turn, _reset as _reset_completions
from app.services.jobs import Job, get_job_tracker


@pytest.fixture(autouse=True)
def clean_state():
    """Reset singletons between tests."""
    get_agent_turn_tracker()._reset()
    get_job_tracker()._reset()
    _reset_completions()
    yield
    get_agent_turn_tracker()._reset()
    get_job_tracker()._reset()
    _reset_completions()


class TestCompletionTurnRoutesQueue:
    """Completion turns route through the turn queue, not direct start."""

    @pytest.mark.asyncio
    async def test_completion_turn_queues_behind_active_user_turn(self):
        """When a user turn is running, the completion turn queues behind it."""
        tracker = get_agent_turn_tracker()
        project_id = "test-project-queue"

        # Simulate a running user turn
        user_turn = tracker.create_turn(project_id, origin="user")
        assert user_turn.status == "running"

        # Create a completed job to trigger a completion turn
        job_tracker = get_job_tracker()
        job = job_tracker.create_job(project_id, "eval", config={
            "prompt_version": "baseline",
            "model_ids": ["test-model"],
        })
        job_tracker.update_state(job.job_id, "completed", result={
            "model_runs": [],
            "output_file": "test.json",
        })

        # Fire a completion turn — should queue, not evict the user turn.
        # request_completion_turn spawns an async task that calls
        # _execute_completion_turn which calls tracker.submit(). The
        # tracker is a singleton and already has the user turn active.
        await request_completion_turn(project_id, job, visibility="visible")

        # Join on what the spawned task PRODUCES — the queued turn — before
        # checking the invariant that the user turn survived. A fixed sleep here
        # asserted the invariant against a queue the task may not have reached
        # yet, so the interesting half could pass without the task running at all.
        await until(
            lambda: tracker.get_queued_turns(project_id),
            message="the completion turn never reached the queue",
        )

        # The user turn should still be active (not evicted)
        active = tracker.get_active_turn(project_id, origin="user")
        assert active is not None
        assert active.turn_id == user_turn.turn_id
        assert active.status == "running"

        # An orchestrator turn should be queued
        queued = tracker.get_queued_turns(project_id)
        assert len(queued) == 1
        assert queued[0].origin == "orchestrator"

    @pytest.mark.asyncio
    async def test_queued_turn_starts_after_active_completes(self):
        """When the active turn completes, the next queued turn starts."""
        tracker = get_agent_turn_tracker()
        project_id = "test-project-advance"

        # Create a running user turn
        user_turn = tracker.create_turn(project_id, origin="user")

        # Queue an orchestrator turn
        build_called = {"count": 0}

        def _mock_build(reserved: str):
            build_called["count"] += 1
            # Must honour the reserved id — the tracker rejects a builder that
            # mints its own, because the caller was already told the reserved one.
            return tracker.create_turn(
                project_id, origin="orchestrator", turn_id=reserved
            )

        tracker.submit(project_id, origin="orchestrator", build_turn=_mock_build)

        # The orchestrator turn should be queued (not started)
        assert build_called["count"] == 0
        queued = tracker.get_queued_turns(project_id)
        assert len(queued) == 1

        # Complete the user turn → triggers queue advancement
        await user_turn.complete()
        tracker.notify_turn_complete(project_id)

        # The queued turn should now have started
        assert build_called["count"] == 1
        queued_after = tracker.get_queued_turns(project_id)
        assert len(queued_after) == 0


class TestTerminalMessagePostsToChat:
    """Abnormal terminal branches must post a chat message, not
    just publish a toast event. ``request_terminal_message`` routes a
    one-line orchestrator turn through the same queue."""

    @pytest.mark.asyncio
    async def test_terminal_message_submits_orchestrator_turn(self):
        from app.services.completion_turn import request_terminal_message

        tracker = get_agent_turn_tracker()
        project_id = "test-project-terminal"

        # No active turn → the submitted orchestrator turn starts via the
        # queue's _build. Patch start_agent_turn (called inside _build) so
        # we don't spin up a real SDK turn; assert it was invoked with the
        # message + orchestrator origin.
        # relay flag defaults False in dev, so no claim is taken (off-path).
        with patch("app.services.agent_runner.start_agent_turn") as mock_start, \
             patch("app.main.get_project_manager") as mock_pm, \
             patch("app.services.coordinator_prompt.build_coordinator_prompt", return_value="sys"), \
             patch("app.services.project_paths.resolve_project_dir", return_value="/tmp/p"):
            project = MagicMock()
            project.session_id = "s"
            project.agent_model = "opus"
            mock_pm.return_value.get_project.return_value = project
            mock_start.return_value = MagicMock(turn_id="t-terminal-123")

            await request_terminal_message(project_id, "Optimization stopped — boom.")

        mock_start.assert_called_once()
        kwargs = mock_start.call_args.kwargs
        assert kwargs["prompt"] == "Optimization stopped — boom."
        assert kwargs["origin"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_terminal_message_is_best_effort(self):
        """A failure building the turn must not raise — the underlying
        terminal state is already published by the caller."""
        from app.services.completion_turn import request_terminal_message

        get_agent_turn_tracker()._reset()
        # No project manager / project → _build raises inside submit; the
        # helper must swallow it.
        with patch("app.main.get_project_manager", side_effect=RuntimeError("boom")):
            # Must not raise.
            await request_terminal_message("nonexistent", "anything")


class TestHarnessBuildCancellation:
    """Cancelling a harness_build job terminates its underlying task."""

    @pytest.mark.asyncio
    async def test_cancel_stops_task(self):
        """cancel_job on a harness_build job cancels the asyncio task."""
        job_tracker = get_job_tracker()
        project_id = "test-project-cancel"

        job = job_tracker.create_job(project_id, "harness_build", config={
            "project_id": project_id,
            "project_dir": "/tmp/fake",
            "scaffold": True,
            "validate": True,
        })

        # Create a fake long-running task and attach it to the job
        async def _long_running():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_long_running(), name="test-harness-task")
        job._task = task

        # Cancel via JobTracker
        job_tracker.cancel_job(job.job_id)

        # Give the event loop a tick to propagate cancellation
        await asyncio.sleep(0)

        assert job.state == "cancelled"
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_without_task_handle_still_updates_state(self):
        """cancel_job works even when no task handle is stored."""
        job_tracker = get_job_tracker()
        project_id = "test-project-cancel-notask"

        job = job_tracker.create_job(project_id, "harness_build", config={})
        assert job._task is None

        job_tracker.cancel_job(job.job_id)

        assert job.state == "cancelled"
        assert job.completed_at is not None
