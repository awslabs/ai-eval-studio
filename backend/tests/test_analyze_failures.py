# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for failure analysis module.

Tests the analyze_failures function that processes eval results and extracts
structured failure information by category, metric, and individual case details.
"""

import pytest
import json
from pathlib import Path
from app.services.failure_analyzer import (
    analyze_failures,
    diff_regressions,
    shape_failure_digest,
    _MAX_PASSING_EXEMPLARS,
    _MAX_FAILED_EXEMPLARS,
)


@pytest.fixture
def sample_results_data():
    """Sample eval results in DeepEval-native format (array of model runs)."""
    return [{
        "test_run_id": "test-model",
        "timestamp": "2026-01-01T00:00:00",
        "hyperparameters": {"model": "test-model", "prompt_version": "baseline"},
        "results": [
            {
                "name": "test_case_0",
                "success": True,
                "input": "What is 2+2?",
                "actual_output": "4",
                "expected_output": "4",
                "additional_metadata": {"id": "1", "category": "math"},
                "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": True, "score": 1.0, "reason": "Exact match"}],
            },
            {
                "name": "test_case_1",
                "success": False,
                "input": "What is 3+3?",
                "actual_output": "six",
                "expected_output": "6",
                "additional_metadata": {"id": "2", "category": "math"},
                "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": False, "score": 0.0, "reason": "Expected: 6"}],
            },
            {
                "name": "test_case_2",
                "success": False,
                "input": "Capital of France?",
                "actual_output": "paris",
                "expected_output": "Paris",
                "additional_metadata": {"id": "3", "category": "geography"},
                "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": False, "score": 0.0, "reason": "Expected: Paris"}],
            },
            {
                "name": "test_case_3",
                "success": True,
                "input": "Capital of Germany?",
                "actual_output": "Berlin",
                "expected_output": "Berlin",
                "additional_metadata": {"id": "4", "category": "geography"},
                "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": True, "score": 1.0, "reason": "Exact match"}],
            },
            {
                "name": "test_case_4",
                "success": False,
                "input": "Spell 'hello'",
                "actual_output": "Hello",
                "expected_output": "hello",
                "additional_metadata": {"id": "5", "category": "spelling"},
                "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": False, "score": 0.0, "reason": "Expected: hello"}],
            },
        ],
    }]


@pytest.fixture
def results_file(tmp_path, sample_results_data):
    """Create a temporary results file with sample data."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(sample_results_data, indent=2))
    return results_path


def test_analyze_failures_basic(results_file):
    """Test that basic statistics (total, passed, failed, pass_rate) are computed correctly."""
    result = analyze_failures(results_file)
    analysis = result["model_analyses"][0]

    assert analysis["test_run_id"] == "test-model"
    assert analysis["prompt_version"] == "baseline"
    assert analysis["total_cases"] == 5
    assert analysis["passed_cases"] == 2
    assert analysis["failed_cases"] == 3
    assert analysis["pass_rate"] == 0.4  # 2/5


def test_failure_by_category(results_file):
    """Test that failure breakdown by category is correct."""
    result = analyze_failures(results_file)
    analysis = result["model_analyses"][0]

    assert "failure_by_category" in analysis
    by_category = analysis["failure_by_category"]

    # Math: 2 total, 1 failed (case 2)
    assert by_category["math"]["total"] == 2
    assert by_category["math"]["failed"] == 1
    assert by_category["math"]["pass_rate"] == 0.5

    # Geography: 2 total, 1 failed (case 3)
    assert by_category["geography"]["total"] == 2
    assert by_category["geography"]["failed"] == 1
    assert by_category["geography"]["pass_rate"] == 0.5

    # Spelling: 1 total, 1 failed (case 5)
    assert by_category["spelling"]["total"] == 1
    assert by_category["spelling"]["failed"] == 1
    assert by_category["spelling"]["pass_rate"] == 0.0


def test_failure_by_metric(results_file):
    """Test that failure breakdown by metric is correct."""
    result = analyze_failures(results_file)
    analysis = result["model_analyses"][0]

    assert "failure_by_metric" in analysis
    by_metric = analysis["failure_by_metric"]

    # ExactMatch: 5 total, 3 failed
    assert by_metric["ExactMatch"]["total"] == 5
    assert by_metric["ExactMatch"]["failed"] == 3
    assert by_metric["ExactMatch"]["pass_rate"] == 0.4


def test_failed_case_details_include_io(results_file):
    """Test that failed case details include input, expected, actual, and failing metrics."""
    result = analyze_failures(results_file)
    analysis = result["model_analyses"][0]

    assert "failed_case_details" in analysis
    failed_cases = analysis["failed_case_details"]

    # Should have 3 failed cases
    assert len(failed_cases) == 3

    # Check first failed case (case_id 2)
    case_2 = next(c for c in failed_cases if c["case_id"] == "2")
    assert case_2["input"] == "What is 3+3?"
    assert case_2["expected_output"] == "6"
    assert case_2["actual_output"] == "six"
    assert case_2["category"] == "math"

    # Check failing metrics structure
    assert "failing_metrics" in case_2
    assert "ExactMatch" in case_2["failing_metrics"]
    exact_match = case_2["failing_metrics"]["ExactMatch"]
    assert exact_match["score"] == 0.0
    assert exact_match["reason"] == "Expected: 6"
    assert exact_match["threshold"] == 1.0


class TestPassingExemplars:
    """The digest carries passing-case exemplars as the 'what good looks
    like' template, bounded for size, category-agnostic."""

    def test_passing_cases_surfaced_with_full_metrics(self, results_file):
        """Passing cases appear with input/expected/actual + every metric
        (score + reason) — the template the optimizer extends to failures."""
        analysis = analyze_failures(results_file)["model_analyses"][0]
        passing = analysis["passing_case_details"]
        # Two passing cases in the fixture, both under the bound.
        assert len(passing) == 2
        ids = {c["case_id"] for c in passing}
        assert ids == {"1", "4"}
        case_1 = next(c for c in passing if c["case_id"] == "1")
        assert case_1["input"] == "What is 2+2?"
        assert case_1["actual_output"] == "4"
        # Passing cases show the metric that PASSED (template signal),
        # under the generic ``metrics`` key (not ``failing_metrics``).
        assert "metrics" in case_1
        assert case_1["metrics"]["ExactMatch"]["score"] == 1.0
        assert analysis["passing_exemplars_truncated"] is False

    def test_passing_exemplars_bounded(self, tmp_path):
        """A run with many passing cases is down-sampled to the bound, and
        the truncation flag is set."""
        results = [{
            "test_run_id": "m", "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {"model": "m", "prompt_version": "v1"},
            "results": [
                {
                    "name": f"c{i}", "success": True,
                    "input": f"q{i}", "actual_output": "a", "expected_output": "a",
                    "additional_metadata": {"id": str(i)},
                    "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0, "threshold": 1.0, "reason": "ok"}],
                }
                for i in range(_MAX_PASSING_EXEMPLARS * 3)
            ],
        }]
        path = tmp_path / "big.json"
        path.write_text(json.dumps(results))
        analysis = analyze_failures(path)["model_analyses"][0]
        assert len(analysis["passing_case_details"]) == _MAX_PASSING_EXEMPLARS
        assert analysis["passing_exemplars_truncated"] is True

    def test_passing_exemplars_work_without_category(self, tmp_path):
        """Category is optional — exemplar selection must not require it."""
        results = [{
            "test_run_id": "m", "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {"model": "m", "prompt_version": "v1"},
            "results": [
                {
                    "name": f"c{i}", "success": True,
                    "input": f"q{i}", "actual_output": "a", "expected_output": "a",
                    "additional_metadata": {"id": str(i)},  # no category
                    "metrics_data": [{"name": "ExactMatch", "success": True, "score": 1.0, "threshold": 1.0, "reason": "ok"}],
                }
                for i in range(_MAX_PASSING_EXEMPLARS * 2)
            ],
        }]
        path = tmp_path / "nocat.json"
        path.write_text(json.dumps(results))
        analysis = analyze_failures(path)["model_analyses"][0]
        # Bounded, and the cases are spread (stride sample), not the first N.
        exemplars = analysis["passing_case_details"]
        assert len(exemplars) == _MAX_PASSING_EXEMPLARS
        picked = [c["case_id"] for c in exemplars]
        assert picked[0] == "0"
        assert picked != [str(i) for i in range(_MAX_PASSING_EXEMPLARS)]


class TestRegressionDiff:
    """diff_regressions surfaces pass->fail cases vs the prior run as DATA
    (not a gate). Best-effort; never raises."""

    def _run(self, model_id, cases):
        return [{
            "test_run_id": model_id, "timestamp": "2026-01-01T00:00:00",
            "hyperparameters": {"model": model_id, "prompt_version": "v"},
            "results": cases,
        }]

    def _case(self, cid, success, actual="a"):
        return {
            "name": f"c{cid}", "success": success,
            "input": f"q{cid}", "actual_output": actual, "expected_output": "a",
            "additional_metadata": {"id": cid},
            "metrics_data": [{"name": "ExactMatch", "success": success, "score": 1.0 if success else 0.0, "threshold": 1.0, "reason": "r"}],
        }

    def test_detects_pass_to_fail(self, tmp_path):
        prior = tmp_path / "prior.json"
        current = tmp_path / "current.json"
        prior.write_text(json.dumps(self._run("m", [
            self._case("1", True), self._case("2", True), self._case("3", False),
        ])))
        current.write_text(json.dumps(self._run("m", [
            self._case("1", True),
            self._case("2", False, actual="broke"),  # regressed
            self._case("3", True),                    # fixed (not a regression)
        ])))
        out = diff_regressions(current, prior)
        assert len(out["regressions"]) == 1
        block = out["regressions"][0]
        assert block["model"] == "m"
        assert block["count"] == 1
        regressed = block["cases"][0]
        assert regressed["case_id"] == "2"
        # The prior passing output is attached so the optimizer sees what
        # it looked like when it worked.
        assert regressed["prior_actual_output"] == "a"

    def test_no_regressions_returns_empty(self, tmp_path):
        prior = tmp_path / "p.json"
        current = tmp_path / "c.json"
        prior.write_text(json.dumps(self._run("m", [self._case("1", True)])))
        current.write_text(json.dumps(self._run("m", [self._case("1", True)])))
        assert diff_regressions(current, prior) == {"regressions": []}

    def test_missing_prior_file_is_best_effort(self, tmp_path):
        current = tmp_path / "c.json"
        current.write_text(json.dumps(self._run("m", [self._case("1", False)])))
        # Nonexistent prior path → no raise, empty result.
        assert diff_regressions(current, tmp_path / "nope.json") == {"regressions": []}

    def test_corrupt_prior_file_is_best_effort(self, tmp_path):
        current = tmp_path / "c.json"
        prior = tmp_path / "p.json"
        current.write_text(json.dumps(self._run("m", [self._case("1", False)])))
        prior.write_text("{not valid json")
        assert diff_regressions(current, prior) == {"regressions": []}

    def test_model_set_change_skips_unmatched(self, tmp_path):
        """If the prior run had a different model, that model can't be
        diffed — skip it rather than reporting phantom regressions."""
        prior = tmp_path / "p.json"
        current = tmp_path / "c.json"
        prior.write_text(json.dumps(self._run("old-model", [self._case("1", True)])))
        current.write_text(json.dumps(self._run("new-model", [self._case("1", False)])))
        assert diff_regressions(current, prior) == {"regressions": []}


class TestShapeFailureDigest:
    """Tests for shape_failure_digest (optimizer-boundary bounding)."""

    def test_bounds_failed_cases_under_limit(self, tmp_path):
        """Large number of failures → bounded to _MAX_FAILED_EXEMPLARS."""
        # Build a results file with 150 failed cases, each with ~100K input
        large_input = "x" * 100_000
        results = [{
            "test_run_id": "m1",
            "hyperparameters": {"model": "m1"},
            "results": [
                {
                    "name": f"tc{i}",
                    "success": False,
                    "input": large_input,
                    "expected_output": large_input,
                    "actual_output": "wrong",
                    "additional_metadata": {"id": str(i)},
                    "metrics_data": [
                        {
                            "name": "Correctness",
                            "score": 0.0,
                            "success": False,
                            "reason": "Failed",
                        }
                    ],
                }
                for i in range(150)
            ],
        }]
        path = tmp_path / "large.json"
        path.write_text(json.dumps(results))

        # Run analyze_failures then shape the digest
        analysis = analyze_failures(path)
        shaped = shape_failure_digest(analysis)

        from app.services.mcp_response import MAX_MCP_PAYLOAD_BYTES
        serialized = json.dumps(shaped, default=str)
        assert len(serialized.encode("utf-8", errors="ignore")) < MAX_MCP_PAYLOAD_BYTES

        # Check bounded failed sample
        model_analysis = shaped["model_analyses"][0]
        assert len(model_analysis["failed_case_details"]) <= _MAX_FAILED_EXEMPLARS
        assert model_analysis["failed_details_truncated"] is True
        assert model_analysis["failed_cases"] == 150  # True total preserved

        # Verify failing_metrics key is preserved
        if model_analysis["failed_case_details"]:
            assert "failing_metrics" in model_analysis["failed_case_details"][0]

    def test_raw_analyze_failures_unchanged(self, tmp_path):
        """shape_failure_digest does not mutate the raw analysis."""
        # Build results with large inputs
        large_input = "y" * 100_000
        results = [{
            "test_run_id": "m1",
            "hyperparameters": {"model": "m1"},
            "results": [
                {
                    "name": f"tc{i}",
                    "success": False,
                    "input": large_input,
                    "expected_output": large_input,
                    "actual_output": "wrong",
                    "additional_metadata": {"id": str(i)},
                    "metrics_data": [{"name": "Correctness", "score": 0.0, "success": False}],
                }
                for i in range(150)
            ],
        }]
        path = tmp_path / "raw.json"
        path.write_text(json.dumps(results))

        # Call analyze_failures
        analysis = analyze_failures(path)
        original_failed_count = len(analysis["model_analyses"][0]["failed_case_details"])
        original_input_length = len(analysis["model_analyses"][0]["failed_case_details"][0]["input"])

        # Shape the digest
        shaped = shape_failure_digest(analysis)

        # Assert raw analysis unchanged
        assert len(analysis["model_analyses"][0]["failed_case_details"]) == original_failed_count
        assert len(analysis["model_analyses"][0]["failed_case_details"][0]["input"]) == original_input_length

    def test_small_results_file_not_truncated(self, tmp_path):
        """Small results file → failed_details_truncated is False."""
        results = [{
            "test_run_id": "m1",
            "hyperparameters": {"model": "m1"},
            "results": [
                {
                    "name": "tc1",
                    "success": False,
                    "input": "short",
                    "expected_output": "output",
                    "actual_output": "wrong",
                    "additional_metadata": {"id": "1"},
                    "metrics_data": [{"name": "Correctness", "score": 0.0, "success": False}],
                }
                for _ in range(3)
            ],
        }]
        path = tmp_path / "small.json"
        path.write_text(json.dumps(results))

        analysis = analyze_failures(path)
        shaped = shape_failure_digest(analysis)

        model_analysis = shaped["model_analyses"][0]
        assert model_analysis["failed_details_truncated"] is False
        assert len(model_analysis["failed_case_details"]) == 3
