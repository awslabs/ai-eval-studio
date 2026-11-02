# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval job cancellation.

Covers the cancel_job tracker method, the REST cancel endpoint, the process-group
kill helper, and the agent-facing cancel_eval MCP tool.

The MCP-tool class is the one that matters most here. Cancelling from chat is how
a user actually stops a run, and that tool previously killed only a local pid — so
for a detached cloud worker it stopped nothing while still reporting success, and
the worker kept spending Bedrock on target and judge models. Those cases assert
the compute seam was really invoked, not just that a success string came back.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.jobs import get_job_tracker
from app.services.harness_runner import kill_harness_subprocess


def _new_eval_job(tracker, project_id="proj-1", model_ids=("model-a",), prompt_version="baseline"):
    """Helper for the standard ``eval`` job shape used across this test file."""
    return tracker.create_job(
        project_id,
        job_type="eval",
        config={"model_ids": list(model_ids), "prompt_version": prompt_version},
    )


class TestCancelJob:
    """Tests for ``JobTracker.cancel_job``."""

    def setup_method(self):
        """Reset the tracker before each test."""
        get_job_tracker()._reset()

    def test_cancel_pending_job(self):
        """Cancelling a pending job sets status to cancelled."""
        tracker = get_job_tracker()
        job = _new_eval_job(tracker)

        tracker.cancel_job(job.job_id)

        assert job.status == "cancelled"
        assert job.completed_at is not None
        assert "Cancelled" in job.progress["message"]

    def test_cancel_running_job(self):
        """Cancelling a running job sets status to cancelled."""
        tracker = get_job_tracker()
        job = _new_eval_job(tracker)
        tracker.update_progress(job.job_id, message="Running...")

        assert job.status == "running"
        tracker.cancel_job(job.job_id)
        assert job.status == "cancelled"

    def test_cancelled_job_not_active(self):
        """Cancelled jobs should not be returned by get_active_job."""
        tracker = get_job_tracker()
        job = _new_eval_job(tracker)
        tracker.cancel_job(job.job_id)

        assert tracker.get_active_job("proj-1") is None

    def test_cancelled_job_evicted(self):
        """Cancelled jobs count as terminal for eviction."""
        tracker = get_job_tracker()
        job = _new_eval_job(tracker)
        tracker.cancel_job(job.job_id)

        # Cancelled job should be in the terminal set for eviction
        terminal = [j for j in tracker._jobs.values() if j.status == "cancelled"]
        assert len(terminal) == 1

    def test_pid_stored_on_job(self):
        """PID field exists and defaults to None."""
        tracker = get_job_tracker()
        job = _new_eval_job(tracker)

        assert job.pid is None
        job.pid = 12345
        assert job.pid == 12345


class TestCancelEndpoint:
    """Tests for the POST cancel eval job REST endpoint."""

    def setup_method(self):
        """Reset the tracker before each test."""
        get_job_tracker()._reset()

    @pytest.mark.asyncio
    @patch("app.api.eval_jobs.get_project_dir_or_404")
    async def test_cancel_running_job(self, mock_project_dir, tmp_path):
        """POST cancel should mark job as cancelled and return 200."""
        mock_project_dir.return_value = tmp_path

        tracker = get_job_tracker()
        job = _new_eval_job(tracker)
        tracker.update_progress(job.job_id, message="Running...")

        from app.api.eval_jobs import cancel_eval_job
        result = await cancel_eval_job("proj-1", job.job_id)

        assert result["status"] == "cancelled"
        assert job.status == "cancelled"

    @pytest.mark.asyncio
    @patch("app.api.eval_jobs.get_project_dir_or_404")
    async def test_cancel_already_completed(self, mock_project_dir, tmp_path):
        """POST cancel on completed job should return 409."""
        mock_project_dir.return_value = tmp_path

        tracker = get_job_tracker()
        job = _new_eval_job(tracker)
        tracker.complete_job(job.job_id, [], "results.json")

        from fastapi import HTTPException
        from app.api.eval_jobs import cancel_eval_job
        with pytest.raises(HTTPException) as exc_info:
            await cancel_eval_job("proj-1", job.job_id)
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    @patch("app.api.eval_jobs.get_project_dir_or_404")
    async def test_cancel_nonexistent_job(self, mock_project_dir, tmp_path):
        """POST cancel on nonexistent job should return 404."""
        mock_project_dir.return_value = tmp_path

        from fastapi import HTTPException
        from app.api.eval_jobs import cancel_eval_job
        with pytest.raises(HTTPException) as exc_info:
            await cancel_eval_job("proj-1", "nonexistent-id")
        assert exc_info.value.status_code == 404


class TestKillHarnessSubprocess:
    """Tests for the shared kill_harness_subprocess helper.

    The helper must signal the entire process group (not just the immediate
    PID) so that `uv run`'s grandchild Python actually stops, and must
    escalate to SIGKILL if SIGTERM doesn't take within the grace period.

    ``getpgid`` returns the pid itself below, because that is the only shape the
    helper group-kills: every harness spawn passes ``start_new_session``, which
    makes the child the leader of a group whose id is its own pid. A pid that
    leads no group is signalled alone — see ``test_bounded_subprocess``.
    """

    @pytest.mark.asyncio
    async def test_sends_sigterm_to_process_group(self):
        """SIGTERM must land on the group, not the immediate child."""
        import signal

        with patch("app.services.harness_runner.os.getpgid", return_value=12345), \
             patch("app.services.harness_runner.os.killpg") as mock_killpg, \
             patch("app.services.harness_runner.os.kill") as mock_kill, \
             patch("app.services.harness_runner.asyncio.sleep", new_callable=AsyncMock):
            # Simulate the process already exited after SIGTERM (first os.kill
            # check with signal 0 raises ProcessLookupError).
            mock_kill.side_effect = ProcessLookupError

            await kill_harness_subprocess(12345, grace_seconds=0.0)

        # SIGTERM went to the group, not the immediate PID.
        mock_killpg.assert_any_call(12345, signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_escalates_to_sigkill_when_process_survives_grace(self):
        """If the child is still alive after the grace period, send SIGKILL."""
        import signal

        with patch("app.services.harness_runner.os.getpgid", return_value=12345), \
             patch("app.services.harness_runner.os.killpg") as mock_killpg, \
             patch("app.services.harness_runner.os.kill") as mock_kill, \
             patch("app.services.harness_runner.asyncio.sleep", new_callable=AsyncMock):
            # os.kill(pid, 0) succeeds → child still alive → escalate.
            mock_kill.return_value = None

            await kill_harness_subprocess(12345, grace_seconds=0.0)

        sent_signals = [call.args[1] for call in mock_killpg.call_args_list]
        assert signal.SIGTERM in sent_signals
        assert signal.SIGKILL in sent_signals

    @pytest.mark.asyncio
    async def test_handles_already_dead_process(self):
        """getpgid raising ProcessLookupError means the child already exited."""
        with patch(
            "app.services.harness_runner.os.getpgid",
            side_effect=ProcessLookupError,
        ), patch("app.services.harness_runner.os.killpg") as mock_killpg, \
             patch("app.services.harness_runner.os.kill") as mock_kill:
            await kill_harness_subprocess(12345, grace_seconds=0.0)

        mock_killpg.assert_not_called()
        mock_kill.assert_not_called()


class TestHardCancelProjectEval:
    """``hard_cancel_project_eval`` is the consolidated cancel path used
    by both the optimization /cancel endpoint and the parent-job
    /jobs/{id}/cancel path. It must:

    * Iterate every active job for the project (eval + optimization
      iteration + harness_build).
    * SIGKILL each running eval subprocess by awaiting
      ``kill_harness_subprocess`` (NOT fire-and-forget — the previous
      pattern returned before kill completed).
    * Be idempotent — calling twice is a no-op.
    """

    def setup_method(self):
        get_job_tracker()._reset()

    @pytest.mark.asyncio
    async def test_awaits_kill_for_every_active_eval_pid(self):
        """Each active eval's pid is awaited (not just scheduled)."""
        from app.services.harness_runner import hard_cancel_project_eval

        tracker = get_job_tracker()
        job_a = _new_eval_job(tracker, project_id="proj-multi")
        job_a.pid = 11111
        tracker.update_progress(job_a.job_id, message="running")
        job_b = _new_eval_job(tracker, project_id="proj-multi")
        job_b.pid = 22222
        tracker.update_progress(job_b.job_id, message="running")

        with patch(
            "app.services.harness_runner.kill_harness_subprocess",
            new_callable=AsyncMock,
        ) as mock_kill:
            cancelled = await hard_cancel_project_eval("proj-multi")

        assert cancelled == 2
        # Both pids were awaited (gather → AsyncMock called twice).
        called_pids = sorted(c.args[0] for c in mock_kill.await_args_list)
        assert called_pids == [11111, 22222]

    @pytest.mark.asyncio
    async def test_kills_child_eval_when_parent_optimization_cancelled(self):
        """Parent ``optimization_iteration`` carries no pid; the active
        child ``eval`` does. ``hard_cancel`` must SIGKILL the child."""
        from app.services.harness_runner import hard_cancel_project_eval

        tracker = get_job_tracker()
        parent = tracker.create_job(
            "proj-nested", job_type="optimization_iteration", config={},
        )
        # Parent has no pid (no subprocess of its own).
        assert parent.pid is None
        tracker.update_progress(parent.job_id, message="iter 1")

        child = _new_eval_job(tracker, project_id="proj-nested")
        child.pid = 33333
        tracker.update_progress(child.job_id, message="evaluating")

        with patch(
            "app.services.harness_runner.kill_harness_subprocess",
            new_callable=AsyncMock,
        ) as mock_kill:
            await hard_cancel_project_eval("proj-nested")

        # Parent had no pid so wasn't killed; child's pid was awaited.
        assert mock_kill.await_count == 1
        assert mock_kill.await_args.args[0] == 33333

    @pytest.mark.asyncio
    async def test_idempotent_when_called_twice(self):
        """Second call finds no active jobs (already terminal) → no-op."""
        from app.services.harness_runner import hard_cancel_project_eval

        tracker = get_job_tracker()
        job = _new_eval_job(tracker, project_id="proj-idem")
        job.pid = 55555
        tracker.update_progress(job.job_id, message="running")

        with patch(
            "app.services.harness_runner.kill_harness_subprocess",
            new_callable=AsyncMock,
        ) as mock_kill:
            first = await hard_cancel_project_eval("proj-idem")
            second = await hard_cancel_project_eval("proj-idem")

        assert first == 1
        assert second == 0
        assert mock_kill.await_count == 1

    @pytest.mark.asyncio
    async def test_publishes_done_envelope_for_each_cancelled_eval(self):
        """Each eval cancelled by hard_cancel emits a per-job done envelope.

        The fix for the stuck eval bar: ``hard_cancel_project_eval`` was
        the only path that reliably executed the kill but never told the
        SPA the job ended, so the bar wedged at "model 0 of N" with a
        live Stop button forever. Routing through ``update_state`` makes
        the chokepoint emit the envelope for every cancelled eval.
        """
        from app.services.harness_runner import hard_cancel_project_eval

        tracker = get_job_tracker()
        job_a = _new_eval_job(tracker, project_id="proj-done")
        job_a.pid = 11111
        tracker.update_progress(job_a.job_id, message="running")
        job_b = _new_eval_job(tracker, project_id="proj-done")
        job_b.pid = 22222
        tracker.update_progress(job_b.job_id, message="running")

        with patch(
            "app.services.harness_runner.kill_harness_subprocess",
            new_callable=AsyncMock,
        ), patch("app.services.jobs._publish_per_job_done") as mock_done, patch(
            "app.services.jobs._publish_project_event",
        ):
            await hard_cancel_project_eval("proj-done")

        # Both evals were cancelled → two done envelopes, each with
        # status="cancelled".
        assert mock_done.call_count == 2
        statuses = [c.kwargs["status"] for c in mock_done.call_args_list]
        assert statuses == ["cancelled", "cancelled"]


class TestListEvalJobsFiltersByType:
    """``GET /eval/jobs`` returns only ``eval`` jobs.

    Regression: during an optimization the parent
    ``optimization_iteration`` is also ``status=="running"`` and was
    being returned alongside the child ``eval``. The SPA's
    ``useEvalJobStream.pollJobs`` picks the first running job and
    subscribes to its stream — without this filter it could subscribe
    to the parent stream which doesn't emit per-case progress, so the
    bar froze at "model 0 of N".
    """

    def setup_method(self):
        get_job_tracker()._reset()

    def test_excludes_optimization_iteration_jobs(self):
        from app.api.eval_jobs import _eval_job_to_dict

        tracker = get_job_tracker()
        eval_job = _new_eval_job(tracker, project_id="proj-filter")
        opt_job = tracker.create_job(
            "proj-filter", job_type="optimization_iteration", config={},
        )

        # Mirror the endpoint's behavior: list_jobs returns both, but
        # the response should include only the eval.
        all_jobs = tracker.list_jobs("proj-filter")
        assert {j.job_id for j in all_jobs} == {eval_job.job_id, opt_job.job_id}

        eval_only = [_eval_job_to_dict(j) for j in all_jobs if j.type == "eval"]
        assert [j["job_id"] for j in eval_only] == [eval_job.job_id]
        # ``job_type`` is on the wire so the SPA can defensively filter.
        assert eval_only[0]["job_type"] == "eval"


class TestCancelEvalMcpTool:
    """The agent-facing cancel_eval MCP tool, exercised through its handler.

    These go through ``cancel_eval.handler`` — the decorator-wrapped tool the agent
    actually calls — not the REST endpoint. The tool had no coverage at all before
    this, which is how it shipped unable to stop a detached worker while reporting
    that it had.
    """

    def setup_method(self):
        """Reset the tracker so no job leaks between cases."""
        get_job_tracker()._reset()

    @staticmethod
    def _handler():
        """Return the underlying coroutine of the decorator-wrapped tool."""
        from app.services.mcp_tools.eval import cancel_eval

        return cancel_eval.handler

    @staticmethod
    def _running_job(project_id="proj-mcp", pid=None):
        """Create a running eval job, optionally with a local pid."""
        tracker = get_job_tracker()
        job = tracker.create_job(
            project_id,
            job_type="eval",
            config={"model_ids": ["m"], "prompt_version": "baseline"},
        )
        tracker.update_progress(job.job_id, message="Running...")
        if pid is not None:
            job.pid = pid
        return job

    @staticmethod
    def _patches(*, compute, store, tmp_path, arch_split=True):
        """Common patch stack: compute seam, metadata store, project dir, flag."""
        platform = MagicMock()
        platform.metadata_store = store
        return (
            patch("app.services.mcp_tools.eval.resolve_project_dir", lambda pid: tmp_path),
            patch("app.platform.arch_split.resolve_arch_split", new_callable=AsyncMock,
                  return_value=arch_split),
            patch("app.platform.get_platform", return_value=platform),
            patch("app.platform.job_compute.get_job_compute", return_value=compute),
        )

    @pytest.mark.asyncio
    async def test_fargate_job_is_stopped_through_compute_seam(self, tmp_path):
        """A detached worker must actually be stopped, with its ARN on the handle."""
        task_arn = "arn:aws:ecs:us-east-1:111122223333:task/cluster/abc123"
        job = self._running_job()
        assert job.pid is None  # detached: no local process to signal

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={"compute_handle": task_arn})
        store.get_optimization_state = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4:
            result = await self._handler()(
                {"project_id": "proj-mcp", "job_id": job.job_id}
            )

        assert compute.cancel.await_count == 1, (
            f"the worker was not stopped, yet the tool answered {result!r}"
        )
        handle = compute.cancel.await_args.args[0]
        assert handle.value == task_arn
        assert handle.kind == "fargate_task"
        assert job.status == "cancelled"

    @pytest.mark.asyncio
    async def test_dev_job_kills_subprocess_group(self, tmp_path):
        """A dev/in-process eval still has its subprocess group signalled."""
        job = self._running_job(pid=4242)

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={"compute_handle": None})
        store.get_optimization_state = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patches(
            compute=compute, store=store, tmp_path=tmp_path, arch_split=False
        )
        with p1, p2, p3, p4, patch(
            "app.services.harness_runner.kill_harness_subprocess",
            new_callable=AsyncMock,
        ) as mock_kill:
            await self._handler()({"project_id": "proj-mcp", "job_id": job.job_id})

        mock_kill.assert_awaited_once_with(4242)
        compute.cancel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_durable_only_job_is_found(self, tmp_path):
        """A job present only in the durable store must be found, not 404'd.

        This is the cold-tracker case: the process handling the cancel did not
        dispatch the eval, so its in-memory tracker knows nothing about it.
        """
        task_arn = "arn:aws:ecs:us-east-1:111122223333:task/cluster/cold99"
        job_id = "job-durable-only"
        project_id = "proj-mcp"
        assert get_job_tracker().get_job(job_id) is None  # tracker is cold

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={"compute_handle": task_arn})
        store.get_optimization_state = AsyncMock(return_value=None)
        store.list_jobs = AsyncMock(return_value=[{
            "job_id": job_id,
            "project_id": project_id,
            "owner_id": "owner-1",
            "type": "eval",
            "state": "running",
            "config": {"source": "manual"},
            "created_at": "2026-07-28T00:00:00+00:00",
        }])

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4:
            result = await self._handler()(
                {"project_id": project_id, "job_id": job_id}
            )

        text = result["content"][0]["text"]
        assert "not found" not in text.lower(), text
        assert compute.cancel.await_count == 1

    @pytest.mark.asyncio
    async def test_optimization_stop_is_version_conditional_on_durable_path(self, tmp_path):
        """The stop write carries the version from its own read, and never reads EFS."""
        job = self._running_job()

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={
            "compute_handle": "arn:aws:ecs:us-east-1:111122223333:task/cluster/v1"
        })
        store.get_optimization_state = AsyncMock(return_value={
            "version": 41, "status": "running", "iterations": [],
            "target_metric_scores": {"Correctness": 0.8},
            "max_iterations": 5, "plateau_window": 3,
        })
        store.put_optimization_state = AsyncMock(return_value=42)

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4, patch(
            "app.services.optimization_state.OptimizationState.load"
        ) as mock_load:
            await self._handler()({"project_id": "proj-mcp", "job_id": job.job_id})

        mock_load.assert_not_called()
        assert store.put_optimization_state.await_count == 1
        kwargs = store.put_optimization_state.await_args.kwargs
        assert kwargs.get("expected_version") == 41, (
            "the conditional write must carry the version from its own read"
        )
        written = store.put_optimization_state.await_args.args[2]
        assert written["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_version_race_does_not_fail_the_cancel(self, tmp_path):
        """A lost version race must not turn a successful worker stop into an error."""
        from app.platform.metadata_store import ConcurrentUpdateError

        job = self._running_job()

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={
            "compute_handle": "arn:aws:ecs:us-east-1:111122223333:task/cluster/race"
        })
        store.get_optimization_state = AsyncMock(return_value={
            "version": 7, "status": "running", "iterations": [],
            "target_metric_scores": {"Correctness": 0.8},
            "max_iterations": 5, "plateau_window": 3,
        })
        store.put_optimization_state = AsyncMock(
            side_effect=ConcurrentUpdateError("version moved")
        )

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4:
            result = await self._handler()(
                {"project_id": "proj-mcp", "job_id": job.job_id}
            )

        assert compute.cancel.await_count == 1
        assert result.get("is_error") is not True, result
        assert "cancelled" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_other_project_job_is_refused(self, tmp_path):
        """A job id from another project must not reach any cancel call."""
        job = self._running_job(project_id="proj-owner")

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={"compute_handle": None})
        store.get_optimization_state = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4:
            result = await self._handler()(
                {"project_id": "proj-intruder", "job_id": job.job_id}
            )

        assert result.get("is_error") is True, result
        compute.cancel.assert_not_awaited()
        assert job.status == "running"

    @pytest.mark.asyncio
    async def test_already_terminal_job_reports_without_cancelling(self, tmp_path):
        """A finished job is reported as such, with no cancel attempted."""
        job = self._running_job()
        get_job_tracker().cancel_job(job.job_id)

        compute = MagicMock()
        compute.cancel = AsyncMock()
        # StopTask returning is only ACCEPTANCE; the cancel path confirms
        # lastStatus == STOPPED before treating the worker as gone.
        compute.status = AsyncMock(return_value="STOPPED")
        store = MagicMock()
        store.get_job = AsyncMock(return_value={"compute_handle": None})
        store.get_optimization_state = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patches(compute=compute, store=store, tmp_path=tmp_path)
        with p1, p2, p3, p4:
            result = await self._handler()(
                {"project_id": "proj-mcp", "job_id": job.job_id}
            )

        assert "already finished" in result["content"][0]["text"].lower()
        compute.cancel.assert_not_awaited()


def test_cancel_mechanism_has_no_second_copy():
    """Neither entry point may reach the compute seam itself.

    The mechanism lives in one module. A future edit that re-inlines a compute-seam
    call into the router or the tool fails here rather than silently reintroducing
    a cancel path that can drift out of step with the other.
    """
    import ast
    from pathlib import Path

    import app.api.eval_jobs as eval_jobs_mod
    import app.services.mcp_tools.eval as mcp_eval_mod

    offenders = {}
    for module in (eval_jobs_mod, mcp_eval_mod):
        path = Path(module.__file__)
        tree = ast.parse(path.read_text())
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "get_job_compute"
        ]
        if hits:
            offenders[path.name] = hits

    assert not offenders, (
        "the compute seam must be reached only through app.services.eval_cancel; "
        f"direct references found: {offenders}"
    )
