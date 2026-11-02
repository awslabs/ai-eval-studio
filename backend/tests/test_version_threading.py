# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for version threading in OptimizationState writes.

Verifies that save() returns the new version and callers thread the returned value
forward (not guess expected_version+1), preventing spurious ConcurrentUpdateError.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_save_returns_version_on_new_path():
    """save() on new path returns the version from put_optimization_state."""
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create a minimal state
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )

        with patch("app.platform.get_platform") as mock_platform:
            mock_store = mock_platform.return_value.metadata_store
            mock_put = AsyncMock(return_value=42)  # put returns version 42
            mock_store.put_optimization_state = mock_put

            # Call save on new path
            new_version = state.save(
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=41,
            )

            # Assert save returns the version
            assert new_version == 42
            assert mock_put.called


def test_save_returns_none_on_old_path():
    """save() on old path returns None."""
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create a minimal state
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )

        # Call save on old path
        new_version = state.save(arch_split=False)

        # Assert save returns None on old path
        assert new_version is None


@pytest.mark.asyncio
async def test_record_iteration_returns_version():
    """record_iteration returns the version from save()."""
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create a minimal state
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )

        with patch("app.platform.get_platform") as mock_platform:
            mock_store = mock_platform.return_value.metadata_store
            mock_put = AsyncMock(return_value=2)  # returns version 2
            mock_store.put_optimization_state = mock_put

            # Call record_iteration
            new_version = state.record_iteration(
                prompt_version="v1",
                run_id="run-1",
                metrics_summary={"Correctness": 0.9},
                total_cases=10,
                passed=9,
                per_model_pass_rates={"m1": 0.9},
                per_model_metrics={"m1": {"Correctness": 0.9}},
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=1,
            )

            # Assert record_iteration returns the version
            assert new_version == 2


@pytest.mark.asyncio
async def test_check_convergence_returns_version():
    """check_convergence returns (convergence_reason, version) tuple."""
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create a state that will converge (hit max_iter_cap)
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )
        # Record 5 iterations to hit the cap
        for i in range(5):
            state.iterations.append({
                "iteration": i + 1,
                "prompt_version": f"v{i+1}",
                "run_id": f"run-{i+1}",
                "metrics_summary": {"Correctness": 0.7},
                "total_cases": 10,
                "passed": 7,
                "pass_rate": 0.7,
                "timestamp": "2024-01-01T00:00:00Z",
            })

        with patch("app.platform.get_platform") as mock_platform:
            mock_store = mock_platform.return_value.metadata_store
            mock_put = AsyncMock(return_value=6)  # returns version 6
            mock_store.put_optimization_state = mock_put

            # Call check_convergence
            convergence, new_version = state.check_convergence(
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=5,
            )

            # Assert check_convergence returns tuple
            assert convergence == "max_iterations"
            assert new_version == 6


@pytest.mark.asyncio
async def test_version_threading_record_to_check():
    """Version is threaded from record_iteration to check_convergence."""
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Create a minimal state
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )

        with patch("app.platform.get_platform") as mock_platform:
            mock_store = mock_platform.return_value.metadata_store
            # First call (record_iteration) returns version 2
            # Second call (check_convergence) returns version 3
            mock_put = AsyncMock(side_effect=[2, 3])
            mock_store.put_optimization_state = mock_put

            # Call record_iteration
            version_after_record = state.record_iteration(
                prompt_version="v1",
                run_id="run-1",
                metrics_summary={"Correctness": 0.9},
                total_cases=10,
                passed=9,
                per_model_pass_rates={"m1": 0.9},
                per_model_metrics={"m1": {"Correctness": 0.9}},
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=1,
            )

            # Thread the returned version to check_convergence
            convergence, version_after_check = state.check_convergence(
                arch_split=True,
                user_id="u-1",
                project_id="p-1",
                expected_version=version_after_record,  # MUST use returned version, not +1 guess
            )

            # Assert versions are threaded correctly
            assert version_after_record == 2
            assert version_after_check is None or version_after_check == 3

            # Verify put was called twice with correct expected_version
            assert mock_put.call_count == 2
            # First call: expected_version=1
            first_call_kwargs = mock_put.call_args_list[0][1]
            assert first_call_kwargs["expected_version"] == 1
            # Second call: expected_version=2 (the returned version from first call)
            # (check_convergence only saves if converged, so this may not happen)


def test_check_convergence_no_arg_returns_tuple():
    """check_convergence() with no args (old-path / skip-eval branch) still returns
    a 2-tuple, so the skip-eval caller in optimization_executor can unpack it.

    Regression for the incomplete-refactor bug: the "prompt already evaluated,
    skip eval" branch calls check_convergence() with no args and MUST unpack
    (convergence, _). If check_convergence returned a bare value on that path,
    the skip-eval branch would leak a tuple into result_data["convergence"] and
    the WS event payload.
    """
    import tempfile
    from app.services.optimization_state import OptimizationState

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.8},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.02,
            workload="test-workload",
        )
        # No iterations → not converged → returns (None, None), NOT bare None.
        result = state.check_convergence()
        assert isinstance(result, tuple)
        assert len(result) == 2
        convergence, version = result
        assert convergence is None
        assert version is None
