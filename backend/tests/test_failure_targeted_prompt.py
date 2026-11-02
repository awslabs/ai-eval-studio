# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure-targeted seeding prompt builder.

Feeds mock ``failure_analyzer`` outputs into ``build_failure_seed_prompt``
and ``failures_to_seed_goldens`` and asserts the derived scenario /
instructions / seed shape is correct. Uses the same fixture shape as
``test_analyze_failures.py`` for consistency.
"""

from __future__ import annotations

import pytest
from deepeval.dataset import Golden

from app.services.failure_analyzer import analyze_failures
from app.services.failure_targeted_prompt import (
    build_failure_seed_prompt,
    failures_to_seed_goldens,
)


@pytest.fixture
def sample_results_data():
    """Mirror of the fixture in test_analyze_failures for easy comparison."""
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
def analysis(tmp_path, sample_results_data):
    """Run the real ``analyze_failures`` so we test against its exact shape."""
    import json
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(sample_results_data))
    return analyze_failures(results_path)


# ---------------------------------------------------------------------------
# build_failure_seed_prompt
# ---------------------------------------------------------------------------

class TestBuildFailureSeedPrompt:
    """Tests for :func:`build_failure_seed_prompt`."""

    def test_picks_lowest_pass_rate_category(self, analysis):
        """Spelling has pass_rate=0.0 (worst) — it should drive the subject."""
        result = build_failure_seed_prompt(analysis)

        # Subject should include the worst-performing category.
        assert "spelling" in result["subject"].lower()
        # Instructions should call out spelling specifically.
        assert "spelling" in result["instructions"].lower()
        assert "pass rate 0%" in result["instructions"].lower()

    def test_instructions_respects_max_examples(self, analysis):
        """``max_examples`` caps the number of concrete exemplars embedded."""
        result = build_failure_seed_prompt(analysis, max_examples=2)

        # With 3 failed cases + max_examples=2 we expect exactly 2 "Example N:" blocks.
        assert result["instructions"].count("Example 1:") == 1
        assert result["instructions"].count("Example 2:") == 1
        assert result["instructions"].count("Example 3:") == 0

    def test_instructions_include_input_expected_and_metric(self, analysis):
        """Each example block must include input + expected_output + metric name."""
        result = build_failure_seed_prompt(analysis, max_examples=6)

        # The worst category is spelling — its single failing case must appear first.
        assert "Spell 'hello'" in result["instructions"]
        assert "'hello'" in result["instructions"]  # expected_output
        assert "ExactMatch" in result["instructions"]

    def test_output_format_detects_freetext(self, analysis):
        """Short answers like '6' / 'Paris' / 'hello' → free-text format."""
        result = build_failure_seed_prompt(analysis)

        assert "free-text" in result["output_format"].lower()
        assert "json" not in result["output_format"].lower()

    def test_output_format_detects_json(self):
        """expected_outputs starting with '{' tip the heuristic over to JSON."""
        fake_analysis = {
            "model_analyses": [{
                "test_run_id": "m",
                "failure_by_category": {
                    "api": {"total": 3, "failed": 3, "pass_rate": 0.0},
                },
                "failure_by_metric": {},
                "failed_case_details": [
                    {"input": "q1", "expected_output": '{"ok": true, "id": 1}',
                     "category": "api", "failing_metrics": {"JSONCorrectness": {}}},
                    {"input": "q2", "expected_output": '{"ok": false}',
                     "category": "api", "failing_metrics": {"JSONCorrectness": {}}},
                    {"input": "q3", "expected_output": '{"status": "x"}',
                     "category": "api", "failing_metrics": {"JSONCorrectness": {}}},
                ],
            }],
        }

        result = build_failure_seed_prompt(fake_analysis)
        assert "json" in result["output_format"].lower()

    def test_falls_back_to_metric_when_no_categories(self):
        """If no failed case has a category we should describe failures by metric."""
        fake_analysis = {
            "model_analyses": [{
                "test_run_id": "m",
                "failure_by_category": {},
                "failure_by_metric": {
                    "Relevance": {"total": 2, "failed": 2, "pass_rate": 0.0},
                },
                "failed_case_details": [
                    {"input": "q1", "expected_output": "a1",
                     "category": None, "failing_metrics": {"Relevance": {}}},
                ],
            }],
        }

        result = build_failure_seed_prompt(fake_analysis)
        assert "Relevance" in result["subject"]

    def test_empty_analysis_raises(self):
        """Empty input must raise ``ValueError``."""
        with pytest.raises(ValueError):
            build_failure_seed_prompt({})

    def test_error_analysis_raises(self):
        """Analyzer errors must propagate as ``ValueError``."""
        with pytest.raises(ValueError):
            build_failure_seed_prompt({"error": "bad json"})

    def test_no_model_analyses_raises(self):
        """``model_analyses: []`` should not be silently treated as valid input."""
        with pytest.raises(ValueError):
            build_failure_seed_prompt({"model_analyses": []})


# ---------------------------------------------------------------------------
# failures_to_seed_goldens
# ---------------------------------------------------------------------------

class TestFailuresToSeedGoldens:
    """Tests for :func:`failures_to_seed_goldens`."""

    def test_returns_goldens(self, analysis):
        """Basic happy path: every returned seed is a DeepEval ``Golden``."""
        seeds = failures_to_seed_goldens(analysis, max_seeds=10)
        assert seeds  # Three failing cases in the fixture.
        assert all(isinstance(s, Golden) for s in seeds)

    def test_respects_max_seeds(self, analysis):
        """``max_seeds`` should truncate the returned list."""
        seeds = failures_to_seed_goldens(analysis, max_seeds=2)
        assert len(seeds) == 2

    def test_prioritises_worst_category(self, analysis):
        """The worst-category (spelling) case should come first."""
        seeds = failures_to_seed_goldens(analysis, max_seeds=10)

        first = seeds[0]
        assert (first.additional_metadata or {}).get("category") == "spelling"
        assert first.input == "Spell 'hello'"

    def test_metadata_carries_failing_metrics(self, analysis):
        """``failing_metrics`` names must survive into Golden metadata."""
        seeds = failures_to_seed_goldens(analysis, max_seeds=10)

        first_meta = seeds[0].additional_metadata or {}
        assert first_meta.get("source") == "failure_targeted"
        assert "ExactMatch" in first_meta.get("failing_metrics", [])

    def test_empty_analysis_raises(self):
        """Empty / error inputs raise instead of returning empty lists."""
        with pytest.raises(ValueError):
            failures_to_seed_goldens({"model_analyses": []})
        with pytest.raises(ValueError):
            failures_to_seed_goldens({"error": "bad"})

    def test_no_failed_cases_returns_empty(self):
        """An analysis with zero failures is valid — just yields no seeds."""
        fake_analysis = {
            "model_analyses": [{
                "test_run_id": "m",
                "failure_by_category": {},
                "failure_by_metric": {},
                "failed_case_details": [],
            }],
        }
        assert failures_to_seed_goldens(fake_analysis) == []
