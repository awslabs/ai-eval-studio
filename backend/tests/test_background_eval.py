# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for background eval job flow via harness subprocess.

Exercises the MCP dispatch_eval tool handler → harness dispatch → job tracker,
with run_eval_via_harness mocked to simulate a successful harness run.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.jobs import get_job_tracker, Job
from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
from app.services.mcp_tools.eval import (
    check_eval_status as _check_eval_status_tool,
)

_dispatch_eval = _dispatch_eval_tool.handler
_check_eval_status = _check_eval_status_tool.handler

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag-qa-10.csv"


@pytest.fixture
def project_setup(tmp_path):
    """Create a project directory with fixture data, baseline prompt, and dummy harness."""
    project_id = "test-project-123"
    project_dir = tmp_path / project_id
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "prompts").mkdir(parents=True)
    (project_dir / "output").mkdir(parents=True)

    dest = project_dir / "data" / "raw" / "rag-qa-10.csv"
    dest.write_text(FIXTURE_PATH.read_text())

    (project_dir / "prompts" / "baseline.md").write_text(
        "Answer the question based on the provided context."
    )

    # Dummy harness so the tool handler accepts the project
    (project_dir / "run_eval.py").write_text("# dummy harness for tests\n")

    get_job_tracker()._reset()
    get_job_tracker()._reset()

    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "tmp_path": tmp_path,
    }


def _extract_job_id(text: str) -> str:
    """Extract job_id from run_eval tool response text."""
    for line in text.split("\n"):
        if line.startswith("job_id:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Could not extract job_id from: {text}")


def _make_fake_harness(tracker, project_dir):
    """Create a mock _run_eval_via_harness that simulates a successful eval."""
    async def fake_harness(job_id, project_dir, model_ids, prompt_version, max_test_cases, judge_model_id, data_file_override=None, source="manual", owner_id=None):
        # Simulate progress
        tracker.update_progress(job_id, current_model=model_ids[0],
            models_completed=0, models_total=len(model_ids),
            current_case=0, cases_total=3, message="generating")

        # Simulate completion with mock results — uses the canonical
        # short-name rule (services.model_id_short.short_model_id) so the
        # fixture stays in sync with what the real harness writes.
        from app.services.model_id_short import short_model_id

        mock_results = []
        for mid in model_ids:
            mock_results.append({
                "source": source,
                "test_run_id": short_model_id(mid),
                "timestamp": "2026-01-01T00:00:00Z",
                "hyperparameters": {
                    "model": mid,
                    "endpoint": "runtime",
                    "region": "us-east-1",
                    "prompt_version": prompt_version,
                },
                "results": [
                    {"name": f"tc{i}", "success": True, "input": "q", "actual_output": "a",
                     "expected_output": "a", "additional_metadata": {}, "metrics_data": []}
                    for i in range(3)
                ],
            })

        # Save results to disk (like the real harness does)
        output_file = f"eval_results_test_{job_id[:8]}.json"
        (project_dir / "output" / output_file).write_text(json.dumps(mock_results, indent=2))

        tracker.complete_job(job_id, mock_results, output_file)

    return fake_harness


class TestBackgroundEval:
    """Tests for the background eval flow via harness subprocess."""

    @pytest.mark.asyncio
    async def test_run_eval_creates_job_and_completes(self, project_setup):
        """Full flow: dispatch_eval → harness dispatch → job completes."""
        project_id = project_setup["project_id"]
        tmp_path = project_setup["tmp_path"]
        tracker = get_job_tracker()

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch("app.services.harness_runner.run_eval_via_harness") as mock_harness, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock, return_value=[],
             ):
            mock_resolve.return_value = tmp_path / project_id
            mock_harness.side_effect = _make_fake_harness(tracker, tmp_path / project_id)

            result = await _dispatch_eval({
                "project_id": project_id,
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
                "max_test_cases": 3,
            })

            assert "is_error" not in result, f"dispatch_eval returned error: {result}"
            # dispatch_eval returns JSON via mcp_json
            content = result["content"][0]
            data = json.loads(content["text"])
            job_id = data["job_id"]

            # Wait for the mock harness to complete
            for _ in range(50):
                await asyncio.sleep(0.1)
                job = tracker.get_job(job_id)
                if job and job.status in ("completed", "failed"):
                    break

        assert job.status == "completed", f"Job status: {job.status}, error: {job.error}"
        assert job.output_file is not None
        assert job.result is not None
        assert len(job.result) == 1

        # Check check_eval_status returns summary (legacy tracker has the result)
        status_result = await _check_eval_status({
            "project_id": project_id,
            "job_id": job_id,
        })
        assert "Eval complete" in status_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_dispatch_eval_requires_harness(self, project_setup):
        """dispatch_eval returns error when no harness exists."""
        project_id = project_setup["project_id"]
        tmp_path = project_setup["tmp_path"]

        # Remove the dummy harness
        (tmp_path / project_id / "run_eval.py").unlink()

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path / project_id

            result = await _dispatch_eval({
                "project_id": project_id,
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["some-model"],
            })

        assert result.get("is_error") is True
        assert "harness" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_dispatch_eval_missing_data_file(self, project_setup):
        """dispatch_eval returns error for missing data file."""
        project_id = project_setup["project_id"]
        tmp_path = project_setup["tmp_path"]

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path / project_id

            result = await _dispatch_eval({
                "project_id": project_id,
                "data_file": "nonexistent.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["some-model"],
            })

        assert result.get("is_error") is True
        assert "not found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_dispatch_eval_missing_prompt(self, project_setup):
        """dispatch_eval returns error for missing prompt version."""
        project_id = project_setup["project_id"]
        tmp_path = project_setup["tmp_path"]

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path / project_id

            result = await _dispatch_eval({
                "project_id": project_id,
                "data_file": "rag-qa-10.csv",
                "prompt_version": "nonexistent",
                "target_model_ids": ["some-model"],
            })

        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_check_eval_status_not_found(self, project_setup):
        """check_eval_status returns error for nonexistent job."""
        result = await _check_eval_status({
            "project_id": "any",
            "job_id": "nonexistent-job-id",
        })
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_dispatch_eval_multi_model(self, project_setup):
        """Harness eval works with multiple models via dispatch_eval."""
        project_id = project_setup["project_id"]
        tmp_path = project_setup["tmp_path"]
        tracker = get_job_tracker()

        with patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve, \
             patch("app.services.harness_runner.run_eval_via_harness") as mock_harness, \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock,
                 return_value=[],
             ):
            mock_resolve.return_value = tmp_path / project_id
            mock_harness.side_effect = _make_fake_harness(tracker, tmp_path / project_id)

            result = await _dispatch_eval({
                "project_id": project_id,
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": [
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "us.anthropic.claude-sonnet-4-6",
                ],
                "max_test_cases": 2,
            })

            assert "is_error" not in result, f"Error: {result}"
            data = json.loads(result["content"][0]["text"])
            job_id = data["job_id"]

            for _ in range(50):
                await asyncio.sleep(0.1)
                job = tracker.get_job(job_id)
                if job and job.status in ("completed", "failed"):
                    break

        assert job.status == "completed", f"Status: {job.status}, error: {job.error}"
        assert len(job.result) == 2  # Two models
