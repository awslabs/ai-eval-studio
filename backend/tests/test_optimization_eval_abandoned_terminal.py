# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terminal write when an eval is abandoned without results.

When ``_wait_for_eval`` gives up — the worker died, the wait hit its
ceiling, or the eval's durable record went terminal with no ``results``
event — it returns ``None``. The iteration caller then owns persisting a
terminal optimization status with an error attached.

Without that write the optimization state stays ``running`` with
``iterations=0`` forever: no worker to finish it, no error to surface, and
the SPA sits on "Analyzing failures…" with no recovery path. That wedge is
the failure this module guards, so every assertion here is on the
*reloaded* state rather than on a mock call — a passing mock would not have
caught the original bug.

The layer boundary matters as much as the write: ``_wait_for_eval`` detects
abandonment but must not persist it, because the caller is the only place
holding the read version needed for the conditional write. A test that only
checked the end state would pass with the write in the wrong layer, so the
boundary is asserted separately.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.jobs import Job
from app.services.optimization_state import OptimizationState
from app.services.resource_error import Codes


def _iteration_job(project_dir: Path) -> Job:
    """Build an optimization_iteration Job on the EFS (arch_split=False) path."""
    return Job(
        job_id="job-iter-1",
        project_id="proj-abc",
        type="optimization_iteration",
        config={
            "project_id": "proj-abc",
            "project_dir": str(project_dir),
            "prompt_version": "v1",
            "iteration": 1,
            "arch_split": False,
            "model_ids": ["m1"],
        },
    )


def _seed_running_state(project_dir: Path) -> None:
    """Seed a ``running`` optimization state with zero iterations on disk."""
    state = OptimizationState.create(
        project_dir=project_dir,
        target_metric_scores={"ExactMatch": 0.85},
        max_iterations=10,
    )
    state.status = "running"
    state.save()


@pytest.mark.asyncio
async def test_abandoned_eval_persists_terminal_status_and_error():
    """An abandoned eval must leave the RELOADED state terminal with an error.

    Reverting the caller's terminal write leaves status ``running`` and
    ``error is None`` — the wedge — so this fails if that write is removed.
    """
    from app.services.jobs.optimization_executor import (
        run_optimization_iteration_job,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _seed_running_state(project_dir)

        with (
            patch("app.services.jobs.get_job_tracker") as mock_tracker,
            patch(
                "app.services.optimization_orchestrator.dispatch_eval_job",
                new_callable=AsyncMock,
                return_value="eval-job-1",
            ),
            patch(
                "app.services.jobs.optimization_executor._wait_for_eval",
                new_callable=AsyncMock,
                return_value=None,  # eval abandoned
            ),
            patch("app.services.jobs.optimization_executor._publish_project_event"),
        ):
            mock_tracker.return_value = MagicMock()

            result = await run_optimization_iteration_job(_iteration_job(project_dir))

        assert result.status == "failed"

        # The EFFECT: reload from disk, don't trust the return value.
        reloaded = OptimizationState.load(project_dir)
        assert reloaded is not None
        assert reloaded.status != "running", (
            "abandoned eval must not leave the optimization running — that is the wedge"
        )
        assert reloaded.error is not None, (
            "a terminal status with no error gives the user nothing to act on"
        )
        assert reloaded.error.code == Codes.EVAL_DIED_WITHOUT_RESULTS


@pytest.mark.asyncio
async def test_abandoned_eval_does_not_overwrite_user_cancel():
    """A user cancel in flight must survive the abandonment write.

    The caller guards on ``status not in ("cancelled", "stopping")``. Without
    that guard a cancel would be relabelled as an eval failure, so the user's
    own action is lost from the record.
    """
    from app.services.jobs.optimization_executor import (
        run_optimization_iteration_job,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
        )
        state.status = "stopping"  # user clicked Stop while the eval was in flight
        state.save()

        with (
            patch("app.services.jobs.get_job_tracker") as mock_tracker,
            patch(
                "app.services.optimization_orchestrator.dispatch_eval_job",
                new_callable=AsyncMock,
                return_value="eval-job-1",
            ),
            patch(
                "app.services.jobs.optimization_executor._wait_for_eval",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("app.services.jobs.optimization_executor._publish_project_event"),
        ):
            mock_tracker.return_value = MagicMock()
            await run_optimization_iteration_job(_iteration_job(project_dir))

        reloaded = OptimizationState.load(project_dir)
        assert reloaded is not None
        assert reloaded.status == "stopping", (
            "the eval-failure write must not stomp an in-flight user cancel"
        )


def test_wait_for_eval_does_not_own_the_terminal_write():
    """The wait detects abandonment; the CALLER persists it.

    Asserted on source structure because the boundary is the invariant: the
    caller holds the read version the conditional write needs, so a future
    refactor that moves ``_record_terminal_error`` into the wait would break
    the version threading while still ending in a terminal state — which an
    end-state-only assertion would happily accept.
    """
    import inspect

    from app.services.jobs import optimization_executor

    wait_src = inspect.getsource(optimization_executor._wait_for_eval)
    assert "_record_terminal_error" not in wait_src, (
        "_wait_for_eval must surface abandonment, not persist it — "
        "the caller owns the terminal write"
    )

    caller_src = inspect.getsource(
        optimization_executor.run_optimization_iteration_job
    )
    assert "_record_terminal_error" in caller_src, (
        "the iteration caller must persist the terminal error"
    )
