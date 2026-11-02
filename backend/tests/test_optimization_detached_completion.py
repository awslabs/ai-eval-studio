# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for detached-worker optimization completion (event-log terminal + progress bridge).

On ``arch_split=True`` an optimization iteration's eval runs on a detached
Fargate worker. The coordinator's in-memory ``JobTracker`` never observes that
worker's completion, so ``_wait_for_eval`` must detect terminal off the DURABLE
EVENT LOG (``data.event == "results"``) and read the result via a bounded
consistent-read loop — mirroring the progress republisher.

Matched pair with the worker ``durable_write=False`` fix: once the coordinator
detects the ``results`` event and reads the result, it writes the eval record
TERMINAL (``state="completed"``) authoritatively, so the now-handle-bearing
reconciler cannot false-fail a genuinely-completed eval after the worker exits.

These tests trip the guards behaviorally:

* arch_split=True: ``_wait_for_eval`` terminates off a seeded event-log
  ``results`` signal (the in-memory eval job stays queued), reads the durable
  result via consistent-read, returns iteration_data, AND writes the record
  terminal via ``update_job_state(state="completed")``.
* arch_split=True CEILING: a worker that never emits ``results`` still hits the
  ceiling → parent failed with ``OPTIMIZATION_EVAL_WAIT_EXCEEDED`` + tag-first
  eval reap.
* Double-fire prevention: the optimization-path progress poller runs
  progress-only (``fire_completion=False``) — see
  ``test_optimization_progress_bridge_no_double_fire``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs.optimization_executor import _wait_for_eval
from app.services.resource_error import Codes, ResourceError


@dataclass
class _FakeJob:
    """Minimal Job stand-in for _wait_for_eval poll-loop tests."""

    job_id: str
    project_id: str = "proj-1"
    state: str = "queued"
    error: ResourceError | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.state


@dataclass
class _FakeTracker:
    """In-memory tracker recording calls; never schedules real work."""

    jobs: dict[str, _FakeJob] = field(default_factory=dict)
    cancel_calls: list[str] = field(default_factory=list)
    state_updates: list[tuple[str, str, ResourceError | None]] = field(default_factory=list)

    def get_job(self, job_id: str) -> _FakeJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)

    def update_state(
        self, job_id: str, state: str, *, error: ResourceError | None = None
    ) -> None:
        self.state_updates.append((job_id, state, error))
        if job_id in self.jobs:
            self.jobs[job_id].state = state
            if error is not None:
                self.jobs[job_id].error = error


def _running_state():
    """Return an OptimizationState-like object whose status keeps cancel dormant."""
    s = MagicMock()
    s.status = "running"
    return s


def _handle(state, project_id, project_dir, **kwargs):
    """Build a real OptimizationStateHandle around a given state.

    A REAL handle, not a mock: the executor reads .state/.status off it and the
    write path reads .arch_split/.owner_id/.version, so a MagicMock would silently
    accept a wrong field name and pass.
    """
    from app.services.optimization_state_repository import OptimizationStateHandle

    return OptimizationStateHandle(
        state=state,
        version=kwargs.get("version", 0),
        arch_split=kwargs.get("arch_split", False),
        owner_id=kwargs.get("owner_id"),
        project_id=project_id,
        project_dir=project_dir,
    )


async def _running_handle_read(project_id, project_dir, **kwargs):
    """Repository read stand-in yielding a live ("running") optimization state."""
    return _handle(_running_state(), project_id, project_dir, **kwargs)


async def _empty_handle_read(project_id, project_dir, **kwargs):
    """Repository read stand-in yielding no optimization state."""
    return _handle(None, project_id, project_dir, **kwargs)


@pytest.fixture
def _no_sleep(monkeypatch):
    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "app.services.jobs.optimization_executor.asyncio.sleep", _sleep
    )


@pytest.mark.asyncio
async def test_arch_split_terminates_off_event_log_reads_file_writes_record_terminal(
    monkeypatch, _no_sleep, tmp_path: Path
):
    """arch_split=True: terminal off event log, metrics from the MATERIALIZED FILE, record→completed.

    The FILE is the source of truth: the recorded run_output_file is the
    real filename from the terminal event's data.file (NOT the eval UUID), and
    metrics come from load_materialized_results (NOT the DDB `result` field).
    """
    # In-memory eval job is stuck queued (detached worker — coordinator never
    # sees it flip). Terminal MUST come from the event log, not this status.
    eval_job = _FakeJob(job_id="eval-1", state="queued")
    parent_job = _FakeJob(
        job_id="parent-1",
        state="running",
        config={"arch_split": True, "owner_id": "owner-1"},
    )
    tracker = _FakeTracker(jobs={"eval-1": eval_job, "parent-1": parent_job})

    store = AsyncMock()
    # Cancel-detection reads opt-state: keep it running so cancel is dormant.
    store.get_optimization_state = AsyncMock(return_value={"status": "running", "version": 1})
    # Event log carries the once-per-eval terminal `results` signal with the file.
    store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "scoring_start"}},
        {"seq": 2, "kind": "progress", "data": {"event": "results", "file": "eval_results_v2_20260101.json"}},
    ])
    # The DDB `result` field is deliberately EMPTY — metrics must NOT read it.
    store.get_job = AsyncMock(return_value={"type": "eval", "result": []})
    store.update_job_state = AsyncMock()

    # Patch the REPOSITORY read (which now owns the arch_split branch and the
    # _from_dict call) rather than OptimizationState._from_dict. The store mock is
    # left in place, so this test's get_optimization_state / update assertions
    # still exercise the real durable path.
    monkeypatch.setattr(
        "app.services.optimization_state_repository.read",
        _running_handle_read,
    )
    monkeypatch.setattr(
        "app.services.jobs.iteration_metrics.extract_iteration_metrics",
        lambda results: {
            "passed_cases": results[0].get("passed", 0),
            "total_cases": results[0].get("total", 0),
            "metrics_summary": {},
        },
    )

    # Materialize writes a KNOWN results list to a tmp file; the loop reads THAT.
    materialized = tmp_path / "output" / "eval_results_v2_20260101.json"
    materialized.parent.mkdir(parents=True, exist_ok=True)
    materialized.write_text('[{"model_id": "m1", "passed": 7, "total": 10}]')

    async def fake_materialize(project_id, job_id, run_filename):
        assert run_filename == "eval_results_v2_20260101.json"
        return materialized

    with patch("app.platform.get_platform") as mock_platform, \
         patch(
             "app.services.run_output_snapshot.materialize_detached_run",
             side_effect=fake_materialize,
         ):
        mock_platform.return_value.metadata_store = store

        result = await _wait_for_eval(
            eval_job_id="eval-1",
            project_dir=tmp_path,
            project_id="proj-1",
            parent_job=parent_job,  # type: ignore[arg-type]
            job_tracker=tracker,  # type: ignore[arg-type]
        )

    # Recorded run filename is the real filename (data.file), NOT the eval UUID.
    assert result is not None
    assert result["run_output_file"] == "eval_results_v2_20260101.json"
    # Metrics reflect the MATERIALIZED FILE (7/10), proving the DDB `result`
    # (empty list) was not the source.
    assert result.get("passed_cases") == 7
    assert result.get("total_cases") == 10

    # Coordinator wrote the eval record terminal (matched pair with worker
    # durable_write=False) so the reconciler cannot false-fail it — and carries
    # the real filename as output_file.
    completed_writes = [
        c for c in store.update_job_state.call_args_list
        if c.kwargs.get("state") == "completed"
    ]
    assert len(completed_writes) == 1, store.update_job_state.call_args_list
    call = completed_writes[0]
    assert call.args[0] == "owner-1"  # normalized owner key
    assert call.args[2] == "eval-1"
    assert call.kwargs.get("output_file") == "eval_results_v2_20260101.json"

    # Coordinator ALSO claims completion_fired for the eval: the optimization loop
    # (run_optimization_loop) owns the iteration completion turn, so the child
    # eval's completion is already "handled". Suppresses spurious
    # reconciler/poller completion.
    store.claim_completion_fired.assert_awaited_once_with("owner-1", "proj-1", "eval-1")


@pytest.mark.asyncio
async def test_arch_split_materialization_unavailable_fails_iteration(
    monkeypatch, _no_sleep, tmp_path: Path
):
    """arch_split=True: materialize ceilings out → iteration fails EVAL_RESULTS_UNAVAILABLE + reap."""
    eval_job = _FakeJob(job_id="eval-x", state="queued")
    parent_job = _FakeJob(
        job_id="parent-x",
        state="running",
        config={"arch_split": True, "owner_id": "owner-1"},
    )
    tracker = _FakeTracker(jobs={"eval-x": eval_job, "parent-x": parent_job})

    store = AsyncMock()
    store.get_optimization_state = AsyncMock(return_value={"status": "running", "version": 1})
    store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "results", "file": "eval_results_v1.json"}},
    ])
    store.update_job_state = AsyncMock()

    # Patch the REPOSITORY read (which now owns the arch_split branch and the
    # _from_dict call) rather than OptimizationState._from_dict. The store mock is
    # left in place, so this test's get_optimization_state / update assertions
    # still exercise the real durable path.
    monkeypatch.setattr(
        "app.services.optimization_state_repository.read",
        _running_handle_read,
    )

    from app.services.run_output_snapshot import RunOutputNotAvailable

    async def fake_materialize(*a, **k):
        raise RunOutputNotAvailable("never landed")

    # Tag-first reap must fire for the detached eval.
    compute = MagicMock()
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    compute.__class__ = FargateTaskCompute
    compute.cancel_by_tag = AsyncMock(return_value="arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123")

    with patch("app.platform.get_platform") as mock_platform, \
         patch("app.platform.job_compute.get_job_compute", return_value=compute), \
         patch(
             "app.services.run_output_snapshot.materialize_detached_run",
             side_effect=fake_materialize,
         ):
        mock_platform.return_value.metadata_store = store

        result = await _wait_for_eval(
            eval_job_id="eval-x",
            project_dir=tmp_path,
            project_id="proj-1",
            parent_job=parent_job,  # type: ignore[arg-type]
            job_tracker=tracker,  # type: ignore[arg-type]
        )

    assert result is None
    parent_failed = [
        (jid, st, err) for (jid, st, err) in tracker.state_updates
        if jid == "parent-x" and st == "failed"
    ]
    assert len(parent_failed) == 1
    assert parent_failed[0][2].code == Codes.EVAL_RESULTS_UNAVAILABLE
    # Detached eval reaped tag-first.
    compute.cancel_by_tag.assert_awaited_with("owner-1", "proj-1", "eval-x")
    # Eval record written terminal `failed` (does not linger non-terminal).
    failed_writes = [
        c for c in store.update_job_state.call_args_list
        if c.kwargs.get("state") == "failed"
    ]
    assert len(failed_writes) == 1


@pytest.mark.asyncio
async def test_arch_split_ceiling_reaps_tag_first_and_fails_parent(
    monkeypatch, _no_sleep, tmp_path: Path
):
    """arch_split=True: no results event → ceiling → parent failed + tag-first eval reap."""
    monkeypatch.setenv("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S", "10")

    eval_job = _FakeJob(job_id="eval-2", state="queued")
    parent_job = _FakeJob(
        job_id="parent-2",
        state="running",
        config={"arch_split": True, "owner_id": "owner-1"},
    )
    tracker = _FakeTracker(jobs={"eval-2": eval_job, "parent-2": parent_job})

    store = AsyncMock()
    store.get_optimization_state = AsyncMock(return_value={"status": "running", "version": 1})
    store.events_since = AsyncMock(return_value=[])  # worker never emits results
    store.get_job = AsyncMock(return_value=None)
    store.update_job_state = AsyncMock()

    # Patch the REPOSITORY read (which now owns the arch_split branch and the
    # _from_dict call) rather than OptimizationState._from_dict. The store mock is
    # left in place, so this test's get_optimization_state / update assertions
    # still exercise the real durable path.
    monkeypatch.setattr(
        "app.services.optimization_state_repository.read",
        _running_handle_read,
    )

    # Tag-first reap: assert cancel_by_tag fires for the detached eval.
    compute = MagicMock()
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    compute.__class__ = FargateTaskCompute
    compute.cancel_by_tag = AsyncMock(return_value="arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123")

    with patch("app.platform.get_platform") as mock_platform, \
         patch("app.platform.job_compute.get_job_compute", return_value=compute):
        mock_platform.return_value.metadata_store = store

        result = await _wait_for_eval(
            eval_job_id="eval-2",
            project_dir=tmp_path,
            project_id="proj-1",
            parent_job=parent_job,  # type: ignore[arg-type]
            job_tracker=tracker,  # type: ignore[arg-type]
        )

    assert result is None
    parent_failed = [
        (jid, st, err) for (jid, st, err) in tracker.state_updates
        if jid == "parent-2" and st == "failed"
    ]
    assert len(parent_failed) == 1
    assert parent_failed[0][2].code == Codes.OPTIMIZATION_EVAL_WAIT_EXCEEDED
    # Detached eval reaped tag-first (not just the in-memory no-op cancel).
    compute.cancel_by_tag.assert_awaited_with("owner-1", "proj-1", "eval-2")


@pytest.mark.asyncio
async def test_arch_split_false_unchanged_in_memory_terminal(
    monkeypatch, _no_sleep, tmp_path: Path
):
    """arch_split=False: terminal still read from in-memory job_tracker status."""
    eval_job = _FakeJob(job_id="eval-3", state="completed", config={})
    parent_job = _FakeJob(job_id="parent-3", state="running", config={})
    tracker = _FakeTracker(jobs={"eval-3": eval_job, "parent-3": parent_job})
    eval_job.__dict__["result"] = []
    eval_job.__dict__["output_file"] = "eval_results_v1.json"

    monkeypatch.setattr(
        "app.services.optimization_state_repository.read", _empty_handle_read
    )
    monkeypatch.setattr(
        "app.services.jobs.iteration_metrics.extract_iteration_metrics",
        lambda results: {"passed_cases": 0, "total_cases": 0, "metrics_summary": {}},
    )

    # No store patch — arch_split=False must not touch the durable store.
    result = await _wait_for_eval(
        eval_job_id="eval-3",
        project_dir=tmp_path,
        project_id="proj-1",
        parent_job=parent_job,  # type: ignore[arg-type]
        job_tracker=tracker,  # type: ignore[arg-type]
    )

    assert result is not None
    assert result["run_output_file"] == "eval_results_v1.json"
    assert tracker.cancel_calls == []


@pytest.mark.asyncio
async def test_optimization_progress_bridge_no_double_fire(monkeypatch):
    """dispatch_eval_job starts a progress-ONLY poller (fire_completion=False) on arch_split.

    The optimization loop (run_optimization_loop in
    services/optimization_loop_orchestrator.py) owns iteration completion. Wiring
    a claiming poller into the optimization path would double-fire, because the
    loop's completion turn bypasses claim_completion_fired. So the
    optimization-path bridge must run progress-only.
    """
    from pathlib import Path

    from app.services import optimization_orchestrator
    from app.services.optimization_orchestrator import OptimizationConfig, dispatch_eval_job

    cfg = OptimizationConfig(
        project_id="proj-1",
        owner_id="owner-1",
        model_ids=["m1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="baseline",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=10,
        plateau_window=3,
    )

    fake_job = MagicMock()
    fake_job.job_id = "eval-1"
    tracker = MagicMock()
    tracker.create_job.return_value = fake_job

    compute = MagicMock()
    compute.launch = AsyncMock(return_value="arn:aws:ecs:us-east-1:123:task/x")
    store = AsyncMock()
    # Unclaimed launch gate: echo the claimant back, as the real store does for the
    # first caller. An AsyncMock would invent a MagicMock here, which reads as
    # someone else's claim and makes dispatch decline to launch at all.
    store.claim_launch_gate = AsyncMock(side_effect=lambda *a, claimant, **k: claimant)

    started_pollers: list[dict] = []

    async def fake_maybe_start_poller(project_id, job_id, owner_id, arch_split, fire_completion=True):
        started_pollers.append({
            "job_id": job_id,
            "owner_id": owner_id,
            "arch_split": arch_split,
            "fire_completion": fire_completion,
        })

    with patch("app.services.jobs.get_job_tracker", return_value=tracker), \
         patch.object(Path, "exists", return_value=True), \
         patch("app.platform.job_compute.get_job_compute", return_value=compute), \
         patch("app.platform.get_platform") as mock_platform, \
         patch("app.services.job_progress_republisher.maybe_start_poller", fake_maybe_start_poller):
        mock_platform.return_value.metadata_store = store

        await dispatch_eval_job(
            Path("/tmp/proj"), cfg, "baseline", None, arch_split=True
        )

    assert len(started_pollers) == 1, "a progress bridge must start on arch_split=True"
    bridge = started_pollers[0]
    assert bridge["job_id"] == "eval-1"
    assert bridge["owner_id"] == "owner-1"
    assert bridge["arch_split"] is True
    assert bridge["fire_completion"] is False, (
        "optimization-path poller must be progress-only (the loop owns completion)"
    )
