# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skip-eval (reused prior results) resume path.

When a loop restart/adopt or resume finds the current prompt already
evaluated (the last recorded iteration has the same prompt_version), the
executor skips the eval and returns the PRIOR iteration's numbers. Those
numbers were already announced in chat when the original iteration
recorded them — so the result must carry ``reused_prior_results`` and the
orchestrator must suppress the visible completion turn.

Without the flag, a backend restart mid-loop (a dev reload; on cloud, any
deploy) adopts the orphaned loop, reuses the last iteration's results, and
posts a new "iteration N+1" chat message that duplicates iteration N byte for
byte — same run file, same numbers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs import Job
from app.services.optimization_state import OptimizationState


def _iteration_job(project_dir: Path, prompt_version: str) -> Job:
    return Job(
        job_id="job-iter-2",
        project_id="proj-abc",
        type="optimization_iteration",
        config={
            "project_id": "proj-abc",
            "project_dir": str(project_dir),
            "prompt_version": prompt_version,
            "iteration": 2,
            "arch_split": False,
            "model_ids": ["m1"],
        },
    )


def _seed_state_with_recorded_iteration(project_dir: Path, prompt_version: str) -> None:
    state = OptimizationState.create(
        project_dir=project_dir,
        target_metric_scores={"ExactMatch": 0.9},
        max_iterations=10,
    )
    state.status = "running"
    state.record_iteration(
        prompt_version=prompt_version,
        run_id="eval_results_v6_x.json",
        metrics_summary={"ExactMatch": 0.812},
        total_cases=416,
        passed=338,
        per_model_pass_rates={"m1": 0.812},
    )
    # record_iteration may persist; ensure status stays running for the test.
    state.status = "running"
    state.save()


@pytest.mark.asyncio
async def test_reused_results_are_flagged_and_eval_is_skipped():
    """Re-running an already-evaluated prompt reuses results and flags them."""
    from app.services.jobs.optimization_executor import (
        run_optimization_iteration_job,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _seed_state_with_recorded_iteration(project_dir, "v6")

        dispatch = AsyncMock(return_value="eval-job-x")
        with (
            patch("app.services.jobs.get_job_tracker") as mock_tracker,
            patch(
                "app.services.optimization_orchestrator.dispatch_eval_job",
                new=dispatch,
            ),
            patch("app.services.jobs.optimization_executor._publish_project_event"),
            patch(
                "app.services.jobs.optimization_executor._dispatch_iteration_reports",
                create=True,
            ),
        ):
            mock_tracker.return_value = MagicMock()
            result = await run_optimization_iteration_job(
                _iteration_job(project_dir, "v6")
            )

        assert result.status == "completed"
        # No new eval was dispatched — prior results reused.
        dispatch.assert_not_awaited()
        # THE flag the orchestrator keys the completion-turn suppression on.
        assert result.data is not None
        assert result.data.get("reused_prior_results") is True
        # Sanity: the reused numbers came from the recorded iteration.
        assert result.data.get("prompt_version") == "v6"
        assert result.data.get("passed_cases") == 338


def test_orchestrator_suppresses_completion_turn_for_reused_results():
    """The loop's completion-turn gate skips announce when results are reused.

    The gate is a two-line conditional inside ``run_optimization_loop``
    (too heavy to drive end-to-end here); this pins its exact predicate so
    a refactor that drops the flag check fails loudly.
    """
    import inspect

    from app.services import optimization_loop_orchestrator as orch

    src = inspect.getsource(orch.run_optimization_loop)
    assert "reused_prior_results" in src, (
        "run_optimization_loop must gate request_completion_turn on "
        "reused_prior_results — without it, restart/adopt re-announces "
        "the previous iteration's numbers as a duplicate chat message"
    )
