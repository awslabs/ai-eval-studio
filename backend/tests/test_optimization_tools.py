# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OptimizationState operations behind the optimization MCP tools."""

import pytest
from pathlib import Path
from app.services.optimization_state import OptimizationState


class TestOptimizationTools:
    """Create, record/converge and best-iteration, driven on OptimizationState directly."""

    def test_start_optimization(self, tmp_path):
        """Starting optimization creates a state file."""
        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        from app.services.optimization_state import OPTIMIZATION_STATE_FILE
        assert (tmp_path / OPTIMIZATION_STATE_FILE).exists()
        assert state.status == "running"

    def test_record_and_check(self, tmp_path):
        """Recording an iteration and checking convergence works."""
        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=10,
        )
        state.record_iteration(
            "baseline", "r1", {"ExactMatch": 0.5}, 10, 5,
            per_model_pass_rates={"m1": 0.5},
            per_model_metrics={"m1": {"ExactMatch": 0.5}},
        )
        result, _version = state.check_convergence()
        assert result is None

        state.record_iteration(
            "v1", "r2", {"ExactMatch": 0.95}, 10, 10,
            per_model_pass_rates={"m1": 1.0},
            per_model_metrics={"m1": {"ExactMatch": 0.95}},
        )
        result, _version = state.check_convergence()
        assert result == "target_reached"

    def test_optimization_summary(self, tmp_path):
        """State produces a useful summary for the agent."""
        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=10,
        )
        # Per-model rates included because that is what every writer records,
        # and what best_iteration ranks on — a row without them is unrankable.
        state.record_iteration(
            "baseline", "r1", {"ExactMatch": 0.5}, 10, 5,
            per_model_pass_rates={"m1": 0.5},
        )
        state.record_iteration(
            "v1", "r2", {"ExactMatch": 0.7}, 10, 7,
            per_model_pass_rates={"m1": 0.7},
        )

        best = state.best_iteration()
        assert best is not None
        assert best["prompt_version"] == "v1"
        assert state.current_iteration == 2
