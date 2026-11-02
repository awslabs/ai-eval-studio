# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reconciled ``GET /projects/{id}/activity`` endpoint.

The endpoint's job is to resolve the disk-vs-memory disagreement that
produced the "two bars disagree / stuck Evaluation running" symptom: it
reads ``OptimizationState`` (disk) and the active eval ``Job`` (in-memory
``JobTracker``) and reconciles them server-side so the SPA reads one
authoritative answer. These tests pin that reconciliation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from app.api.optimization import get_project_activity
from app.services.jobs import get_job_tracker
from app.services.optimization_state import OptimizationState


def _run(coro):
    """Run an async endpoint body in a fresh loop for sync test bodies."""
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def reset_tracker():
    tracker = get_job_tracker()
    tracker._reset()
    yield
    tracker._reset()


@pytest.fixture
def project_dir(tmp_path):
    """A project directory with the patch in place so the endpoint resolves it."""
    d = tmp_path / "proj-activity"
    d.mkdir()
    with patch("app.api.optimization.get_project_dir_or_404", return_value=d):
        yield d


def _make_running_eval(project_id: str, *, source: str, models_total: int = 3):
    """Register an active (running) eval job of the given source."""
    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id, "eval",
        config={"model_ids": ["m"] * models_total, "prompt_version": "v1", "source": source},
    )
    tracker.update_progress(job.job_id, message="running", models_total=models_total)
    return job


class TestActivityReconciliation:
    def test_no_optimization_no_eval_returns_both_null(self, project_dir):
        result = _run(get_project_activity("proj-activity", _user="u"))
        assert result == {"optimization": None, "eval": None}

    def test_running_optimization_with_child_eval_reports_both(self, project_dir):
        state = OptimizationState.create(project_dir, target_metric_scores={"M": 0.9})
        state.status = "running"
        state.save()
        job = _make_running_eval("proj-activity", source="optimization")

        result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["optimization"]["status"] == "running"
        assert result["eval"] is not None
        assert result["eval"]["job_id"] == job.job_id
        assert result["eval"]["status"] == "running"
        assert result["eval"]["models_total"] == 3

    def test_running_eval_reported_regardless_of_optimization_status(self, project_dir):
        """The core fix (inverted from the old behavior): a genuinely
        active eval is reported as running NO MATTER what the on-disk
        optimization status says. The endpoint used to suppress an
        optimization-child eval whenever the loop's disk status wasn't
        "live" — which hid a still-executing eval forever when a loop
        wrote a terminal status (e.g. optimizer_stalled) mid-child-eval.
        Liveness now comes solely from the eval job; the loop lifecycle
        keeps the tracker honest by reaping its child on exit, so a
        surviving active eval is by definition real and must show."""
        state = OptimizationState.create(project_dir, target_metric_scores={"M": 0.9})
        state.status = "optimizer_stalled"
        state.save()
        job = _make_running_eval("proj-activity", source="optimization")

        result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["optimization"]["status"] == "optimizer_stalled"
        # No guessing against disk status — the live eval is reported.
        assert result["eval"] is not None
        assert result["eval"]["job_id"] == job.job_id
        assert result["eval"]["status"] == "running"

    def test_manual_eval_reported_without_any_optimization(self, project_dir):
        """A standalone manual eval (no optimization loop at all) is
        reported as-is."""
        job = _make_running_eval("proj-activity", source="manual")

        result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["optimization"] is None
        assert result["eval"] is not None
        assert result["eval"]["job_id"] == job.job_id

    def test_manual_eval_survives_terminal_optimization_history(self, project_dir):
        """A manual eval run AFTER an optimization converged must still
        show — liveness comes from the eval job, not from the loop's
        terminal status."""
        state = OptimizationState.create(project_dir, target_metric_scores={"M": 0.9})
        state.status = "converged"
        state.save()
        job = _make_running_eval("proj-activity", source="manual")

        result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["optimization"]["status"] == "converged"
        assert result["eval"] is not None
        assert result["eval"]["job_id"] == job.job_id

    def test_completed_eval_not_reported_active(self, project_dir):
        """A finished eval is not active → eval: null."""
        tracker = get_job_tracker()
        job = tracker.create_job(
            "proj-activity", "eval",
            config={"model_ids": ["m"], "prompt_version": "v1", "source": "manual"},
        )
        tracker.complete_job(job.job_id, result=[], output_file="r.json")

        result = _run(get_project_activity("proj-activity", _user="u"))
        assert result["eval"] is None

    def test_stopping_optimization_keeps_child_eval(self, project_dir):
        """``stopping`` (the cancel is in flight) does not hide the child
        eval — the endpoint reports any active eval whatever the loop's
        status."""
        state = OptimizationState.create(project_dir, target_metric_scores={"M": 0.9})
        state.status = "stopping"
        state.save()
        _make_running_eval("proj-activity", source="optimization")

        result = _run(get_project_activity("proj-activity", _user="u"))
        assert result["eval"] is not None


class TestActivityModelsFold:
    """/activity folds the event log into models[] on arch_split=True only."""

    def test_arch_split_true_populates_models_from_event_log(self, project_dir):
        """arch_split=True + seeded event log → eval payload carries a folded models[]."""
        from unittest.mock import AsyncMock

        _make_running_eval("proj-activity", source="manual", models_total=1)

        # Seeded durable event log for the active eval (manual-shaped).
        events = [
            {"stream": "job:x", "seq": 1, "kind": "progress",
             "data": {"event": "generating", "model": "m1", "current": 0, "total": 3}},
            {"stream": "job:x", "seq": 2, "kind": "progress",
             "data": {"event": "scoring_start", "model": "m1", "total": 3}},
            {"stream": "job:x", "seq": 3, "kind": "progress",
             "data": {"event": "scoring_progress", "model": "m1", "current": 2, "total": 3}},
        ]
        store = AsyncMock()
        store.get_optimization_state = AsyncMock(return_value=None)
        store.events_since = AsyncMock(return_value=events)

        with patch("app.api.optimization.resolve_arch_split", new=AsyncMock(return_value=True)), \
             patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store
            result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["eval"] is not None
        models = result["eval"].get("models")
        assert models is not None and len(models) == 1
        m = models[0]
        assert m["model"] == "m1"
        assert m["phase"] == "scoring"
        assert m["cases_total"] == 3
        assert m["generated"] == 3  # scoring_start set generated = cases_total
        assert m["scored_count"] == 2
        # events_since keyed on the active eval job id, full replay (-1).
        store.events_since.assert_awaited_once()
        assert store.events_since.call_args.args[1] == -1

    def test_arch_split_false_omits_models(self, project_dir):
        """arch_split=False (dev / in-process) → models absent (dev byte-behavior unchanged)."""
        _make_running_eval("proj-activity", source="manual", models_total=2)

        # resolve_arch_split returns False in dev by default (no patch needed),
        # but pin it explicitly for determinism.
        from unittest.mock import AsyncMock

        with patch("app.api.optimization.resolve_arch_split", new=AsyncMock(return_value=False)):
            result = _run(get_project_activity("proj-activity", _user="u"))

        assert result["eval"] is not None
        assert "models" not in result["eval"]


def test_activity_appears_for_foreign_process_job(project_dir):
    """The activity view returns a live eval for a job that exists only in the durable store (foreign-process job)."""
    from unittest.mock import AsyncMock, MagicMock
    import time

    project_id = "proj-activity"
    job_id = "job-foreign"
    owner_id = "owner-001"

    # Confirm the tracker is empty for this project (isolation precondition)
    tracker = get_job_tracker()
    active = tracker.get_active_job_by_type(project_id, "eval")
    assert active is None, "Tracker should be empty before the test"

    # Mock the store to return one non-terminal job
    mock_store = MagicMock()
    mock_store.get_optimization_state = AsyncMock(return_value=None)
    mock_store.list_jobs = AsyncMock(
        return_value=[
            {
                "job_id": job_id,
                "project_id": project_id,
                "owner_id": owner_id,
                "type": "eval",
                "state": "running",
                "created_at": time.time(),
                "config": {"source": "manual"},
                "progress": {"models_total": 3, "current": 1, "total": 10},
            }
        ]
    )
    mock_store.events_since = AsyncMock(return_value=[])

    with (
        patch("app.api.optimization.resolve_arch_split", new=AsyncMock(return_value=True)),
        patch("app.platform.get_platform", return_value=MagicMock(metadata_store=mock_store)),
    ):
        result = _run(get_project_activity(project_id, _user=owner_id))

    # Assert the endpoint returned a live eval (sourced from the durable store)
    assert result["eval"] is not None
    assert result["eval"]["job_id"] == job_id
    assert result["eval"]["status"] == "running"


def test_activity_clears_for_terminal_job(project_dir):
    """The activity view stops returning a live eval once the durable record is terminal (terminal-refresh mode)."""
    from unittest.mock import AsyncMock, MagicMock
    import time

    project_id = "proj-activity"
    job_id = "job-terminal"
    owner_id = "owner-001"

    # Seed a stale non-terminal entry in the tracker (simulates a mid-eval restart)
    tracker = get_job_tracker()
    job = tracker.create_job(
        project_id,
        "eval",
        config={"model_ids": ["m1"], "prompt_version": "v1", "source": "manual"},
    )
    # Force the job_id to match the one we'll return from the store
    job.job_id = job_id
    job.state = "running"
    tracker._jobs[job_id] = job

    # Mock the store to return the same job BUT with terminal state
    mock_store = MagicMock()
    mock_store.get_optimization_state = AsyncMock(return_value=None)
    mock_store.list_jobs = AsyncMock(
        return_value=[
            {
                "job_id": job_id,
                "project_id": project_id,
                "owner_id": owner_id,
                "type": "eval",
                "state": "completed",  # Terminal
                "created_at": time.time(),
                "config": {"source": "manual"},
                "progress": {"models_total": 3, "current": 3, "total": 10},
                "completed_at": time.time(),
            }
        ]
    )
    mock_store.events_since = AsyncMock(return_value=[])

    with (
        patch("app.api.optimization.resolve_arch_split", new=AsyncMock(return_value=True)),
        patch("app.platform.get_platform", return_value=MagicMock(metadata_store=mock_store)),
    ):
        result = _run(get_project_activity(project_id, _user=owner_id))

    # Assert the endpoint returned no live eval (terminal refresh cleared the stale entry)
    assert result["eval"] is None
