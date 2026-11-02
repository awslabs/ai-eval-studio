# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization loop terminal exit paths persist terminal status to DynamoDB.

These tests exercise the loop's OWN exit paths (cancel branch + terminal-status
tail in `run_optimization_loop`) to prove both persist terminal status
to DynamoDB when arch_split=True. An earlier change made launch/lifecycle
DDB-aware but MISSED the loop's own exit
paths — on cloud, a user cancel published "cancelled" to WS but DDB stayed
"stopping" → refresh read "stopping" forever.

Test A (cancel): the CancelledError branch must write "cancelled" durably.
Test B (terminal tail catch-all): the tail must flip a lingering "running" to
    "cancelled" in DDB, not do an EFS-only save that leaves DDB "running".
Test C (EFS regression guard): arch_split=False keeps the state.json path.
Test D (durable source guard): a source-code assertion that the bare EFS tail
    save does not come back.
"""
from __future__ import annotations

import asyncio
import inspect
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.optimization_state import OptimizationState, OPTIMIZATION_STATE_FILE


@pytest.mark.asyncio
async def test_cancel_branch_persists_terminal_status_to_ddb():
    """Test A: CancelledError branch writes status='cancelled' to DDB on arch_split=True.

    The seeded record is "running"; the cancel branch must leave it "cancelled"
    in the durable store, with a threaded expected_version and no state.json.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    # Stateful in-memory store mirroring the real DDB semantics
    stored: dict = {}

    async def _put(user_id, project_id, data, expected_version=None):
        new_version = (stored.get("version", 0) + 1) if stored else 1
        stored.clear()
        stored.update(data)
        stored["version"] = new_version
        return new_version

    async def _get(user_id, project_id):
        return dict(stored) if stored else None

    mock_store = MagicMock()
    mock_store.put_optimization_state = AsyncMock(side_effect=_put)
    mock_store.get_optimization_state = AsyncMock(side_effect=_get)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    # Seed the store with a "running" record so the loop enters the iteration body
    # where run_optimization_iteration_job will raise CancelledError
    await _put(
        "user-xyz",
        "proj-abc",
        {
            "target_metric_scores": {"ExactMatch": 0.85},
            "max_iterations": 5,
            "plateau_window": 3,
            "plateau_threshold": 0.02,
            "iterations": [],
            "status": "running",
            "max_iter_cap": 5,
            "workload": None,
            "pending_resume_drift": None,
            "winning_model_id": None,
            "current_prompt_version": None,
            "error": None,
        },
        expected_version=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        with (
            patch("app.platform.get_platform", return_value=mock_platform),
            patch("app.services.optimization_loop_orchestrator.get_job_tracker") as mock_tracker,
            patch("app.services.optimization_loop_orchestrator._publish_project_event"),
            patch("app.services.jobs.optimization_executor.run_optimization_iteration_job",
                  new_callable=AsyncMock, side_effect=asyncio.CancelledError),
            patch("app.services.optimization_loop_orchestrator._run_prompt_optimizer", new_callable=AsyncMock),
            patch("app.services.completion_turn.request_completion_turn", new_callable=AsyncMock),
            patch("app.services.completion_turn.request_terminal_message", new_callable=AsyncMock),
            patch("app.services.optimization_orchestrator.detect_latest_prompt", return_value="v1"),
        ):
            # Mock tracker to return None for active eval (so _reap_loop_child_eval no-ops)
            mock_tracker_inst = MagicMock()
            mock_job = MagicMock()
            mock_tracker_inst.create_job.return_value = mock_job
            mock_tracker_inst.get_active_job_by_type.return_value = None
            mock_tracker.return_value = mock_tracker_inst

            # Drive the loop with arch_split=True
            await run_optimization_loop({
                "project_id": "proj-abc",
                "project_dir": str(project_dir),
                "user_id": "user-xyz",
                "arch_split": True,
                "prompt_version": "v1",
                "target_metric_scores": {"ExactMatch": 0.85},
                "max_iterations": 5,
            })

        # The durable record must end "cancelled", not the seeded "running".
        final_data = await _get("user-xyz", "proj-abc")
        assert final_data is not None, "store should still have the seeded record"
        assert final_data["status"] == "cancelled", \
            "cancel branch must have written 'cancelled' to DDB"

        # Assert: the cancel-branch save threaded a non-None expected_version
        # (version threading — an existing record must not be written with
        # expected_version=None, which the store's attribute_not_exists guard
        # would reject as a ConcurrentUpdateError).
        put_calls = mock_store.put_optimization_state.call_args_list
        assert any(
            call.kwargs.get("expected_version") is not None for call in put_calls
        ), "cancel-branch save must thread expected_version"

        # Assert: no state.json exists (proves write went to DDB, not EFS)
        assert not (project_dir / OPTIMIZATION_STATE_FILE).exists(), \
            "arch_split=True must NOT write state.json"


@pytest.mark.asyncio
async def test_terminal_tail_flips_lingering_running_status_to_cancelled():
    """Test B: terminal-status tail flips DDB status='running' to 'cancelled' when loop exits normally.

    A bare OptimizationState.load(project_dir) returns None on arch_split=True and a
    bare .save() writes only EFS, so a tail built that way would leave DDB "running".
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    # Stateful in-memory store
    stored: dict = {}

    async def _put(user_id, project_id, data, expected_version=None):
        new_version = (stored.get("version", 0) + 1) if stored else 1
        stored.clear()
        stored.update(data)
        stored["version"] = new_version
        return new_version

    async def _get(user_id, project_id):
        return dict(stored) if stored else None

    mock_store = MagicMock()
    mock_store.put_optimization_state = AsyncMock(side_effect=_put)
    mock_store.get_optimization_state = AsyncMock(side_effect=_get)
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    # Seed with status="running" (as a normal start would leave it)
    await _put(
        "user-xyz",
        "proj-abc",
        {
            "target_metric_scores": {"ExactMatch": 0.85},
            "max_iterations": 1,
            "plateau_window": 3,
            "plateau_threshold": 0.02,
            "iterations": [],
            "status": "running",
            "max_iter_cap": 1,
            "workload": None,
            "pending_resume_drift": None,
            "winning_model_id": None,
            "current_prompt_version": None,
            "error": None,
        },
        expected_version=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Mock a SUCCESSFUL non-converging iteration
        mock_job_result = MagicMock()
        mock_job_result.status = "completed"
        mock_job_result.result = {"convergence": None}

        with (
            patch("app.platform.get_platform", return_value=mock_platform),
            patch("app.services.optimization_loop_orchestrator.get_job_tracker") as mock_tracker,
            patch("app.services.optimization_loop_orchestrator._publish_project_event"),
            patch("app.services.jobs.optimization_executor.run_optimization_iteration_job",
                  new_callable=AsyncMock, return_value=mock_job_result),
            patch("app.services.optimization_loop_orchestrator._run_prompt_optimizer",
                  new_callable=AsyncMock, return_value=True),
            patch("app.services.completion_turn.request_completion_turn", new_callable=AsyncMock),
            patch("app.services.completion_turn.request_terminal_message", new_callable=AsyncMock),
            patch("app.services.optimization_orchestrator.detect_latest_prompt", return_value="v2"),
        ):
            mock_tracker_inst = MagicMock()
            mock_job = MagicMock()
            mock_tracker_inst.create_job.return_value = mock_job
            mock_tracker_inst.get_active_job_by_type.return_value = None
            mock_tracker.return_value = mock_tracker_inst

            # Drive the loop: 1 iteration, exits normally, status="running" never flipped
            await run_optimization_loop({
                "project_id": "proj-abc",
                "project_dir": str(project_dir),
                "user_id": "user-xyz",
                "arch_split": True,
                "prompt_version": "v1",
                "target_metric_scores": {"ExactMatch": 0.85},
                "max_iterations": 1,
            })

        # Assert: the tail flipped status="running" → "cancelled" in DDB
        final_data = await _get("user-xyz", "proj-abc")
        assert final_data is not None, "tail must read and write store"
        assert final_data["status"] == "cancelled", \
            "tail must flip lingering 'running' to 'cancelled'"

        # Assert: save was called with a non-None expected_version (version threading)
        put_calls = mock_store.put_optimization_state.call_args_list
        assert any(
            call.kwargs.get("expected_version") is not None for call in put_calls
        ), "tail save must thread expected_version"

        # Assert: no state.json exists
        assert not (project_dir / OPTIMIZATION_STATE_FILE).exists(), \
            "arch_split=True must NOT write state.json"


@pytest.mark.asyncio
async def test_arch_split_false_efs_path_unchanged():
    """Test C: arch_split=False keeps the EFS state.json tail behavior.

    No-regression guard for the EFS path: the tail writes state.json and never
    touches the durable store.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    # Mock store to assert it's NOT called on arch_split=False
    mock_store = MagicMock()
    mock_store.put_optimization_state = AsyncMock()
    mock_store.get_optimization_state = AsyncMock()
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Seed a running state.json on disk (EFS path)
        OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=1,
            arch_split=False,
            user_id=None,
            project_id=None,
            expected_version=None,
        )

        # Mock a successful iteration
        mock_job_result = MagicMock()
        mock_job_result.status = "completed"
        mock_job_result.result = {"convergence": None}

        with (
            patch("app.platform.get_platform", return_value=mock_platform),
            patch("app.services.optimization_loop_orchestrator.get_job_tracker") as mock_tracker,
            patch("app.services.optimization_loop_orchestrator._publish_project_event"),
            patch("app.services.jobs.optimization_executor.run_optimization_iteration_job",
                  new_callable=AsyncMock, return_value=mock_job_result),
            patch("app.services.optimization_loop_orchestrator._run_prompt_optimizer",
                  new_callable=AsyncMock, return_value=True),
            patch("app.services.completion_turn.request_completion_turn", new_callable=AsyncMock),
            patch("app.services.completion_turn.request_terminal_message", new_callable=AsyncMock),
            patch("app.services.optimization_orchestrator.detect_latest_prompt", return_value="v2"),
        ):
            mock_tracker_inst = MagicMock()
            mock_job = MagicMock()
            mock_tracker_inst.create_job.return_value = mock_job
            mock_tracker_inst.get_active_job_by_type.return_value = None
            mock_tracker.return_value = mock_tracker_inst

            # Drive the loop with arch_split=False (default)
            await run_optimization_loop({
                "project_id": "proj-abc",
                "project_dir": str(project_dir),
                "user_id": None,
                "arch_split": False,
                "prompt_version": "v1",
                "target_metric_scores": {"ExactMatch": 0.85},
                "max_iterations": 1,
            })

        # Assert: state.json exists and status="cancelled" (EFS tail path worked)
        state_file = project_dir / OPTIMIZATION_STATE_FILE
        assert state_file.exists(), "arch_split=False must write state.json"

        loaded = OptimizationState.load(project_dir)
        assert loaded is not None, "state.json must be loadable"
        assert loaded.status == "cancelled", "EFS tail must flip 'running' to 'cancelled'"

        # Assert: store was NOT called
        mock_store.put_optimization_state.assert_not_called()


def test_terminal_tail_no_bare_save_in_source():
    """Test D: durable source guard — bare tail save 'final_state.save()' is absent.

    The tail reads and writes through ``optimization_state_repository``, which
    picks the home from ``arch_split``; a bare ``final_state.save()`` would write
    EFS only.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    source = inspect.getsource(run_optimization_loop)

    # The bare save pattern (no arguments) must NOT be present
    assert "final_state.save()" not in source, \
        "bare tail save() without args must not exist — prevents future EFS regression"

    # Positive check: the tail save SHOULD carry arch_split= (proof it's flag-aware)
    assert "arch_split=" in source, \
        "tail save must be arch_split-aware (pass arch_split= kwarg)"


@pytest.mark.asyncio
async def test_cancel_branch_flips_stopping_to_cancelled_on_efs_path():
    """The CancelledError branch must flip stopping -> cancelled on BOTH homes.

    Scope of this test: it pins the END STATE, which the terminal tail also
    guarantees. Re-gating the in-branch write on ``arch_split`` therefore does
    NOT fail this test. The in-branch write
    is a uniformity/latency improvement (the durable path transitions immediately;
    the EFS path used to wait for the tail), not a correctness fix.

    What this test does protect is the invariant that matters to the SPA: a
    cancelled loop on the EFS path never ends at "stopping". Previously that
    end-state had no coverage at all on this path.
    """
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    # Store must NOT be touched on the EFS path.
    mock_store = MagicMock()
    mock_store.get_optimization_state = AsyncMock(
        side_effect=AssertionError("EFS path must not read the durable store")
    )
    mock_platform = MagicMock()
    mock_platform.metadata_store = mock_store

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=1,
            arch_split=False,
        )
        # A cancel already landed: the canceller wrote "stopping" and expects the
        # loop to finish the transition on its way out.
        state.status = "stopping"
        state.save()

        with (
            patch("app.platform.get_platform", return_value=mock_platform),
            patch("app.services.optimization_loop_orchestrator.get_job_tracker") as mock_tracker,
            patch("app.services.optimization_loop_orchestrator._publish_project_event"),
            patch(
                "app.services.jobs.optimization_executor.run_optimization_iteration_job",
                new_callable=AsyncMock, side_effect=asyncio.CancelledError,
            ),
            patch("app.services.completion_turn.request_terminal_message", new_callable=AsyncMock),
        ):
            mock_tracker.return_value.create_job = MagicMock(
                return_value=MagicMock(job_id="job-efs-cancel")
            )
            try:
                await run_optimization_loop({
                    "project_id": "proj-efs",
                    "project_dir": str(project_dir),
                    "prompt_version": "baseline",
                    "max_iterations": 1,
                    "arch_split": False,
                    "user_id": None,
                })
            except asyncio.CancelledError:
                pass

        final = OptimizationState.load(project_dir)
        assert final is not None
        assert final.status == "cancelled", (
            "A cancelled loop on the EFS path must leave status 'cancelled', not "
            f"'stopping'. Got {final.status!r}."
        )
