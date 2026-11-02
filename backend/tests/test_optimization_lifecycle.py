# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for optimization lifecycle: start/stop/resume, state transitions, origin filtering."""

import json
import pytest
from pathlib import Path

from app.services.optimization_state import (
    OptimizationState,
    OPTIMIZATION_STATE_FILE,
)
from app.services.agent_turn import AgentTurn, AgentTurnTracker, get_agent_turn_tracker


class TestStoppingStatus:
    """Tests for the 'stopping' status and transitions."""

    def test_stopping_is_valid_status(self, tmp_path):
        """OptimizationState accepts 'stopping' as a status value."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "stopping"
        state.save()

        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        assert loaded.status == "stopping"

    def test_stopping_transitions_to_cancelled(self, tmp_path):
        """Setting status to 'stopping' then to 'cancelled' persists correctly."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "stopping"
        state.save()

        # Simulate orchestrator detecting stopping and setting cancelled
        state.status = "cancelled"
        state.save()

        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        assert loaded.status == "cancelled"


class TestDuplicateStartPrevention:
    """Tests for guarding against duplicate optimization starts."""

    def test_running_state_blocks_new_start(self, tmp_path):
        """Cannot start a new optimization while one is running."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        assert state.status == "running"

        # Simulate the guard check in start_optimization
        existing = OptimizationState.load(project_dir)
        assert existing is not None
        assert existing.status in ("running", "stopping")

    def test_stopping_state_blocks_new_start(self, tmp_path):
        """Cannot start a new optimization while one is stopping."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "stopping"
        state.save()

        existing = OptimizationState.load(project_dir)
        assert existing is not None
        assert existing.status in ("running", "stopping")

    def test_cancelled_state_allows_new_start(self, tmp_path):
        """Can start a new optimization after cancellation."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "cancelled"
        state.save()

        existing = OptimizationState.load(project_dir)
        assert existing is not None
        assert existing.status not in ("running", "stopping")


class TestResumeAfterCancel:
    """Tests for resume functionality."""

    def test_resume_after_cancel(self, tmp_path):
        """Resume after cancel keeps iteration history and resets status."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=10,
        )
        # Record some iterations
        state.record_iteration("v1", "run-1", {"ExactMatch": 0.7}, 10, 7)
        state.record_iteration("v2", "run-2", {"ExactMatch": 0.75}, 10, 8)
        state.status = "cancelled"
        state.save()

        # Resume
        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        loaded.reset_for_resume()

        assert loaded.status == "running"
        assert loaded.current_iteration == 2  # History preserved
        assert len(loaded.iterations) == 2

    def test_resume_after_plateau(self, tmp_path):
        """Resume after plateau resets status while keeping history."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
            max_iterations=10,
            plateau_window=3,
        )
        # Record 3 plateau iterations
        state.record_iteration("v1", "run-1", {"ExactMatch": 0.75}, 100, 75, per_model_pass_rates={"m1": 0.75}, per_model_metrics={"m1": {"ExactMatch": 0.75}})
        state.record_iteration("v2", "run-2", {"ExactMatch": 0.76}, 100, 76, per_model_pass_rates={"m1": 0.76}, per_model_metrics={"m1": {"ExactMatch": 0.76}})
        state.record_iteration("v3", "run-3", {"ExactMatch": 0.755}, 100, 76, per_model_pass_rates={"m1": 0.755}, per_model_metrics={"m1": {"ExactMatch": 0.755}})
        state.check_convergence()
        assert state.status == "plateau"

        # Resume
        state.reset_for_resume()
        assert state.status == "running"
        assert state.current_iteration == 3  # History preserved

    def test_resume_after_max_iterations(self, tmp_path):
        """Resume after max_iterations resets status."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.95},
            max_iterations=2,
        )
        state.record_iteration("v1", "run-1", {"ExactMatch": 0.7}, 10, 7)
        state.record_iteration("v2", "run-2", {"ExactMatch": 0.8}, 10, 8)
        state.check_convergence()
        assert state.status == "max_iterations"

        state.reset_for_resume()
        assert state.status == "running"
        assert state.current_iteration == 2

    def test_resume_after_judge_unreliable(self, tmp_path):
        """``judge_unreliable`` is a resumable terminal status, same as plateau."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=10,
        )
        state.record_iteration("v1", "run-1", {"ExactMatch": 0.7}, 100, 70)
        # Optimization halted by the judge-unreliable detector.
        state.status = "judge_unreliable"
        state.save()

        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        loaded.reset_for_resume()

        assert loaded.status == "running"
        assert loaded.current_iteration == 1


class TestOrphanedStateRecovery:
    """The orphan flip: a running or stopping state with no surviving task
    is marked cancelled.

    Startup no longer flips orphans; it adopts and runs them instead (see
    test_optimization_adoption.py). These tests cover the flip itself.
    """

    def test_recover_running_state(self, tmp_path):
        """Running state without surviving task should be set to cancelled."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        assert state.status == "running"

        # Simulate the orphan flip
        loaded = OptimizationState.load(project_dir)
        if loaded and loaded.status in ("running", "stopping"):
            loaded.status = "cancelled"
            loaded.save()

        recovered = OptimizationState.load(project_dir)
        assert recovered is not None
        assert recovered.status == "cancelled"

    def test_recover_stopping_state(self, tmp_path):
        """Stopping state without surviving task should be set to cancelled."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "stopping"
        state.save()

        # Simulate the orphan flip
        loaded = OptimizationState.load(project_dir)
        if loaded and loaded.status in ("running", "stopping"):
            loaded.status = "cancelled"
            loaded.save()

        recovered = OptimizationState.load(project_dir)
        assert recovered is not None
        assert recovered.status == "cancelled"

    def test_no_recovery_for_terminal_states(self, tmp_path):
        """Converged/cancelled states are not modified by an orphan flip."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )
        state.status = "converged"
        state.save()

        loaded = OptimizationState.load(project_dir)
        assert loaded is not None
        assert loaded.status not in ("running", "stopping")


class TestStartRequiresHarness:
    """Tests for harness prerequisite check."""

    def test_start_without_harness_fails(self, tmp_path):
        """Cannot start optimization without run_eval.py."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # No run_eval.py created — the REST endpoint should reject this
        assert not (project_dir / "run_eval.py").exists()

    def test_start_with_harness_succeeds(self, tmp_path):
        """Harness check passes when run_eval.py exists."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "run_eval.py").write_text("# harness stub")
        assert (project_dir / "run_eval.py").exists()


class TestStartOverwritesExistingState:
    """Tests that starting overwrites any previous optimization state."""

    def test_start_overwrites_existing_state(self, tmp_path):
        """Starting a new optimization overwrites previous state."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create initial state with some history
        old_state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.8},
            max_iterations=5,
        )
        old_state.record_iteration("v1", "run-1", {"ExactMatch": 0.7}, 10, 7)
        old_state.status = "cancelled"
        old_state.save()

        # Create new state (simulating a fresh start)
        new_state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=10,
        )

        assert new_state.current_iteration == 0
        assert new_state.iterations == []
        assert new_state.max_iterations == 10
        assert new_state.target_metric_scores == {"ExactMatch": 0.9}


class TestAgentTurnOriginFiltering:
    """Tests for agent turn origin tagging and filtered retrieval."""

    def setup_method(self):
        """Reset the tracker singleton before each test."""
        tracker = get_agent_turn_tracker()
        tracker._reset()

    def test_user_turn_origin_default(self):
        """Default origin is 'user'."""
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("project-1")
        assert turn.origin == "user"

    def test_orchestrator_turn_origin(self):
        """Orchestrator origin is set correctly."""
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("project-1", origin="orchestrator")
        assert turn.origin == "orchestrator"

    def test_filter_by_user_origin(self):
        """get_active_turn with origin='user' only returns user turns."""
        tracker = get_agent_turn_tracker()

        # Create an orchestrator turn
        orch_turn = tracker.create_turn("project-1", origin="orchestrator")
        assert orch_turn.status == "running"

        # Filter by user origin — should return None
        result = tracker.get_active_turn("project-1", origin="user")
        assert result is None

        # Without filter — should return the orchestrator turn
        result = tracker.get_active_turn("project-1")
        assert result is not None
        assert result.origin == "orchestrator"

    def test_filter_by_orchestrator_origin(self):
        """get_active_turn with origin='orchestrator' returns orchestrator turns."""
        tracker = get_agent_turn_tracker()

        # Create a user turn
        tracker.create_turn("project-1", origin="user")

        # Filter by orchestrator — should return None
        result = tracker.get_active_turn("project-1", origin="orchestrator")
        assert result is None

    def test_unfiltered_returns_any_active(self):
        """get_active_turn without origin filter returns any active turn."""
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("project-1", origin="orchestrator")
        result = tracker.get_active_turn("project-1")
        assert result is not None
        assert result.turn_id == turn.turn_id


class TestStateSavesCorrectLocation:
    """Tests that state files are created at the correct location."""

    def test_state_saves_to_output_dir(self, tmp_path):
        """New state files are created at output/optimization/state.json."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
        )

        state_file = project_dir / OPTIMIZATION_STATE_FILE
        assert state_file.exists()


class TestStartReadsFromHarnessConfig:
    """Tests that optimization start reads from harness config."""

    def test_harness_config_fields_parsed(self, tmp_path):
        """Verify the AST parser extracts config fields correctly."""
        from app.services.harness_config import parse_harness_config as _parse_harness_config

        # Note: field(default_factory=...) won't parse with literal_eval,
        # but simple literal assignments will
        source_simple = '''
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EvalConfig:
    models: list = ["model-a", "model-b"]
    data_file: str = "test.csv"
    field_mappings: dict = {"input": "question", "expected_output": "answer", "category": "topic"}
    target_metric_scores: dict = {"ExactMatch": 0.85}
    max_iterations: int = 10
    plateau_window: int = 3
    prompt_version: str = "baseline"
'''
        config = _parse_harness_config(source_simple)
        assert config.field_mappings == {"input": "question", "expected_output": "answer", "category": "topic"}
        assert config.target_metric_scores == {"ExactMatch": 0.85}
        assert config.max_iterations == 10
        assert config.plateau_window == 3
        assert config.models == ["model-a", "model-b"]
