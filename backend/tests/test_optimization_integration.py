# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the optimization loop tool chain.

Tests the full flow: start -> record -> analyze -> converge
without calling actual models (uses pre-built result files).
"""

import json
import pytest
from pathlib import Path

from app.services.optimization_state import OptimizationState
from app.services.failure_analyzer import analyze_failures


@pytest.fixture
def project_dir(tmp_path):
    """Create a project directory with sample data."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "baseline.md").write_text("You are a helpful assistant.")
    return tmp_path


def write_eval_results(project_dir: Path, filename: str, pass_rate: float, prompt_version: str):
    """Write a synthetic eval results file in DeepEval-native format."""
    n_total = 10
    n_passed = int(n_total * pass_rate)
    results = []
    for i in range(n_total):
        success = i < n_passed
        results.append({
            "name": f"test_case_{i}",
            "success": success,
            "input": f"test input {i}",
            "actual_output": f"expected {i}" if success else f"wrong {i}",
            "expected_output": f"expected {i}",
            "additional_metadata": {"id": str(i), "category": "test"},
            "metrics_data": [{
                "name": "ExactMatch",
                "threshold": 1.0,
                "success": success,
                "score": 1.0 if success else 0.0,
                "reason": "Exact match" if success else "Mismatch",
            }],
        })

    model_runs = [{
        "test_run_id": "test-model",
        "timestamp": "2026-01-01T00:00:00",
        "hyperparameters": {"model": "test-model", "prompt_version": prompt_version},
        "results": results,
    }]
    (project_dir / "output" / filename).write_text(json.dumps(model_runs))
    return model_runs


class TestOptimizationIntegration:
    def test_full_optimization_flow(self, project_dir):
        """Start -> record iteration -> analyze failures -> converge."""
        # Start optimization targeting 90% pass rate
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        assert state.status == "running"

        # Iteration 1: baseline at 50%
        write_eval_results(project_dir, "run1.json", 0.5, "baseline")
        state.record_iteration("baseline", "run1", {"ExactMatch": 0.5}, 10, 5, per_model_pass_rates={"m1": 0.5}, per_model_metrics={"m1": {"ExactMatch": 0.5}})
        convergence, _ = state.check_convergence(); assert convergence is None

        # Analyze failures
        result = analyze_failures(project_dir / "output" / "run1.json")
        analysis = result["model_analyses"][0]
        assert analysis["failed_cases"] == 5
        assert analysis["failure_by_metric"]["ExactMatch"]["failed"] == 5

        # Iteration 2: v1 at 70%
        write_eval_results(project_dir, "run2.json", 0.7, "v1")
        state.record_iteration("v1", "run2", {"ExactMatch": 0.7}, 10, 7, per_model_pass_rates={"m1": 0.7}, per_model_metrics={"m1": {"ExactMatch": 0.7}})
        convergence, _ = state.check_convergence(); assert convergence is None

        # Iteration 3: v2 at 95% — should converge (target was 90%)
        write_eval_results(project_dir, "run3.json", 0.95, "v2")
        state.record_iteration("v2", "run3", {"ExactMatch": 0.95}, 10, 10, per_model_pass_rates={"m1": 1.0}, per_model_metrics={"m1": {"ExactMatch": 0.95}})
        convergence, _ = state.check_convergence(); assert convergence == "target_reached"
        assert state.status == "converged"

        # Best iteration is v2
        best = state.best_iteration()
        assert best["prompt_version"] == "v2"

    def test_plateau_detection_flow(self, project_dir):
        """Detect plateau when scores stop improving."""
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 1.0},
            max_iterations=20,
            plateau_window=3,
        )

        # 3 iterations with no meaningful improvement
        for i, rate in enumerate([0.7, 0.71, 0.705]):
            write_eval_results(project_dir, f"run{i}.json", rate, f"v{i}")
            state.record_iteration(f"v{i}", f"run{i}", {"ExactMatch": rate}, 10, int(rate * 10), per_model_pass_rates={"m1": rate}, per_model_metrics={"m1": {"ExactMatch": rate}})

        convergence, _ = state.check_convergence(); assert convergence == "plateau"
        assert state.status == "plateau"

    def test_max_iterations_flow(self, project_dir):
        """Stop at max iterations."""
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 1.0},
            max_iterations=2,
        )

        state.record_iteration("v1", "r1", {"ExactMatch": 0.5}, 10, 5)
        convergence, _ = state.check_convergence(); assert convergence is None
        state.record_iteration("v2", "r2", {"ExactMatch": 0.6}, 10, 6)
        convergence, _ = state.check_convergence(); assert convergence == "max_iterations"


class TestMetricNameNormalization:
    """Suffix drift robustness: DeepEval's version-
    dependent `` [GEval]`` suffix may appear on EITHER side of the convergence
    comparison — the serialized result names OR the user's config keys."""

    def test_converges_when_config_key_carries_suffix(self, project_dir):
        """Config key ``"Correctness [GEval]"`` vs stripped serialized name
        ``"Correctness"`` — the mirror image of the suffixed-result case below."""
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness [GEval]": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        state.record_iteration(
            "v1", "run1", {"Correctness": 0.95}, 10, 10,
            per_model_pass_rates={"m1": 1.0},
            per_model_metrics={"m1": {"Correctness": 0.95}},
        )
        convergence, _ = state.check_convergence()
        assert convergence == "target_reached", (
            "a suffixed config key must match the stripped serialized name"
        )

    def test_converges_when_result_key_carries_suffix(self, project_dir):
        """Bare config key vs suffixed result name (the original defect)."""
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Correctness": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        state.record_iteration(
            "v1", "run1", {"Correctness [GEval]": 0.95}, 10, 10,
            per_model_pass_rates={"m1": 1.0},
            per_model_metrics={"m1": {"Correctness [GEval]": 0.95}},
        )
        convergence, _ = state.check_convergence()
        assert convergence == "target_reached"

    def test_missing_metric_still_withholds_convergence(self, project_dir):
        """Normalization must not soften the missing-metric = 0.0 rule."""
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"Recall": 0.5},
            max_iterations=5,
            plateau_window=3,
        )
        state.record_iteration(
            "v1", "run1", {"Correctness": 0.99}, 10, 10,
            per_model_pass_rates={"m1": 1.0},
            per_model_metrics={"m1": {"Correctness": 0.99}},
        )
        convergence, _ = state.check_convergence()
        assert convergence is None


class TestMetricNameCollisions:
    """find_metric_name_collisions: distinct names collapsing to one base."""

    def test_detects_suffix_only_difference(self):
        from app.services.optimization_state import find_metric_name_collisions
        assert find_metric_name_collisions(
            ["Correctness [strict]", "Correctness [loose]"]
        ) == ["Correctness"]

    def test_no_collision_for_distinct_bases(self):
        from app.services.optimization_state import find_metric_name_collisions
        assert find_metric_name_collisions(["Correctness", "Recall [GEval]"]) == []

    def test_identical_names_are_not_a_collision(self):
        from app.services.optimization_state import find_metric_name_collisions
        # The same string twice (e.g. config name repeated in target keys)
        # is one metric, not a collision.
        assert find_metric_name_collisions(["Correctness", "Correctness"]) == []

    def test_bare_plus_suffixed_is_a_collision(self):
        from app.services.optimization_state import find_metric_name_collisions
        # Within ONE list, "Correctness" and "Correctness [GEval]" are two
        # configured metrics that would merge after normalization.
        assert find_metric_name_collisions(
            ["Correctness", "Correctness [GEval]"]
        ) == ["Correctness"]
