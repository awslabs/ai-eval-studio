# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for get_eval_results summary + get_case_details drilldown.

The old get_results tool returned the raw per-case JSON (~10 MB for a
realistic run) and blew through the SDK's 1 MiB transport cap, killing
the entire turn. The summary shape keeps responses well under 100 KB and
defers per-case drilldown to get_case_details (bounded by count).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.mcp_tools.eval import (
    _summarize_results,
    get_case_details as _get_case_details_tool,
    get_eval_results as _get_eval_results_tool,
)

_summarize = _summarize_results
_get_results = _get_eval_results_tool.handler
_get_cases = _get_case_details_tool.handler


def _make_case(idx: int, *, success: bool, metric_success: bool = True, reason_len: int = 400) -> dict:
    """Build a realistic case record with a long judge rationale."""
    return {
        "name": f"case-{idx}",
        "success": success,
        "input": f"nl query {idx}",
        "actual_output": f"Answer expression {idx}",
        "expected_output": f"Answer expected {idx}",
        "context": None,
        "retrieval_context": None,
        "additional_metadata": {"id": idx},
        "metrics_data": [
            {
                "name": "Correctness",
                "score": 0.95 if metric_success else 0.1,
                "threshold": 1.0,
                "success": metric_success,
                "reason": "x" * reason_len,
            },
            {
                "name": "Hallucination",
                "score": 1.0,
                "threshold": 0.5,
                "success": True,
                "reason": "y" * reason_len,
            },
        ],
    }


def _make_run_payload(num_cases: int = 10, num_models: int = 2, reason_len: int = 400) -> list[dict]:
    """Build a multi-model eval-results payload in the real harness shape."""
    payload = []
    for m in range(num_models):
        payload.append({
            "model_id": f"us.anthropic.claude-v{m}",
            "test_run_id": f"model-{m}",
            "timestamp": "2026-05-04T12:00:00Z",
            "hyperparameters": {"prompt_version": "v1"},
            "results": [
                _make_case(i, success=(i % 3 != 0), reason_len=reason_len)
                for i in range(num_cases)
            ],
        })
    return payload


class TestSummarize:
    """Pure function tests for the summary builder."""

    def test_summary_has_per_model_counts(self):
        data = _make_run_payload(num_cases=6, num_models=2)
        summary = _summarize(data, "eval_results_v1.json")

        assert summary["run_filename"] == "eval_results_v1.json"
        assert summary["prompt_version"] == "v1"
        assert summary["total_cases"] == 12
        assert len(summary["per_model"]) == 2
        for m in summary["per_model"]:
            # 6 cases, i % 3 != 0 → 4 passing: 1,2,4,5
            assert m["passed"] == 4
            assert m["total"] == 6
            assert abs(m["pass_rate"] - 4 / 6) < 1e-9

    def test_summary_includes_per_metric_rates(self):
        data = _make_run_payload(num_cases=6, num_models=1)
        summary = _summarize(data, "eval_results_v1.json")

        per_metric = summary["per_model"][0]["per_metric"]
        # Every case carries Correctness + Hallucination, with success matching
        # the case's overall success flag (Correctness) or always True (Hallucination).
        assert "Correctness" in per_metric
        assert "Hallucination" in per_metric
        assert per_metric["Hallucination"]["rate"] == 1.0

    def test_summary_size_stays_bounded_for_realistic_run(self):
        """The critical regression: 600+ cases with ~500-char rationales
        must produce a summary under 100 KB (raw payload would be ~10 MB).
        """
        data = _make_run_payload(num_cases=208, num_models=3, reason_len=500)
        summary = _summarize(data, "eval_results_v6.json")
        size = len(json.dumps(summary))
        assert size < 100_000, f"Summary should stay <100 KB; got {size} bytes"


class TestGetEvalResults:
    """Integration tests for the MCP tool wrapper."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> dict:
        project_id = "results-test"
        project_dir = tmp_path / project_id
        (project_dir / "output").mkdir(parents=True)

        filename = "eval_results_v1_20260504_120000.json"
        (project_dir / "output" / filename).write_text(
            json.dumps(_make_run_payload(num_cases=6, num_models=2))
        )
        return {"project_id": project_id, "project_dir": project_dir, "filename": filename}

    async def test_returns_summary_not_full_data(self, project, monkeypatch):
        """Success path: returns bounded summary, never raw per-case data."""
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_results({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
        })
        assert "is_error" not in result
        parsed = json.loads(result["content"][0]["text"])

        assert parsed["total_cases"] == 12
        assert "per_model" in parsed
        # Critical: no per-case payloads leaked into the summary
        for model_summary in parsed["per_model"]:
            assert "results" not in model_summary
            assert "cases" not in model_summary

    async def test_rejects_missing_file(self, project, monkeypatch):
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_results({
            "project_id": project["project_id"],
            "run_filename": "nonexistent.json",
        })
        assert result.get("is_error") is True


class TestGetCaseDetails:
    """Integration tests for the drilldown tool."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> dict:
        project_id = "case-test"
        project_dir = tmp_path / project_id
        (project_dir / "output").mkdir(parents=True)

        filename = "eval_results_v1_20260504_120000.json"
        (project_dir / "output" / filename).write_text(
            json.dumps(_make_run_payload(num_cases=10, num_models=2))
        )
        return {"project_id": project_id, "project_dir": project_dir, "filename": filename}

    async def test_returns_requested_indices(self, project, monkeypatch):
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_cases({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
            "case_indices": [0, 3, 7],
        })
        assert "is_error" not in result
        parsed = json.loads(result["content"][0]["text"])

        assert parsed["returned_count"] == 3
        assert [c["_index"] for c in parsed["cases"]] == [0, 3, 7]
        # Full per-case payload including judge rationale
        assert parsed["cases"][0]["metrics_data"][0]["reason"]

    async def test_respects_limit_when_no_indices(self, project, monkeypatch):
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_cases({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
            "limit": 4,
        })
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["returned_count"] == 4

    async def test_enforces_hard_cap(self, project, monkeypatch):
        """More than 50 indices must return an error, not silently truncate."""
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_cases({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
            "case_indices": list(range(60)),
        })
        assert result.get("is_error") is True
        assert "50" in result["content"][0]["text"]

    async def test_model_filter(self, project, monkeypatch):
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_cases({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
            "model_id": "us.anthropic.claude-v1",
            "case_indices": [0],
        })
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["model_id"] == "us.anthropic.claude-v1"

    async def test_unknown_model_errors(self, project, monkeypatch):
        monkeypatch.setattr(
            "app.services.mcp_tools.eval.resolve_project_dir",
            lambda pid: project["project_dir"],
        )

        result = await _get_cases({
            "project_id": project["project_id"],
            "run_filename": project["filename"],
            "model_id": "nonexistent-model",
        })
        assert result.get("is_error") is True
