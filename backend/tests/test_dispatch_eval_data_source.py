# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``data_source`` parameter on the ``dispatch_eval`` MCP tool.

Covers three branches:

    * ``data_source`` omitted / ``"raw"`` → no override, harness gets
      called with ``data_file_override=None``.
    * ``data_source="synthetic"`` → resolves to a processed CSV under
      ``data/processed/synthetic_only_*.csv`` which is passed through.
    * ``data_source="synthetic"`` with no approved cases → returns an
      MCP error without dispatching the harness.

The harness runner is mocked — we only care about the arguments passed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.jobs import get_job_tracker
from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool

_run_eval = _dispatch_eval_tool.handler

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag-qa-10.csv"


@pytest.fixture
def project(tmp_path):
    """Build a minimal project on disk with harness + synthetic batch.

    The ``src/config.py`` must be parseable by ``load_harness_config``
    — the ``data_source != "raw"`` branches rely on it to pull
    ``field_mappings``.
    """
    project_id = "ds-test-project"
    project_dir = tmp_path / project_id

    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "data" / "processed").mkdir(parents=True)
    (project_dir / "prompts").mkdir(parents=True)
    (project_dir / "output").mkdir(parents=True)
    (project_dir / "src").mkdir(parents=True)

    (project_dir / "data" / "raw" / "rag-qa-10.csv").write_text(
        FIXTURE_PATH.read_text()
    )
    (project_dir / "prompts" / "baseline.md").write_text("Answer the question.\n")
    (project_dir / "run_eval.py").write_text("# dummy harness for tests\n")
    (project_dir / "src" / "config.py").write_text("""
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    data_file: str = "rag-qa-10.csv"
    field_mappings: dict = field(default_factory=lambda: {
        "input": "input",
        "expected_output": "expected_output",
    })
""")

    get_job_tracker()._reset()

    yield {
        "project_id": project_id,
        "project_dir": project_dir,
        "tmp_path": tmp_path,
    }
    get_job_tracker()._reset()


def _seed_approved_batch(project_dir: Path) -> None:
    """Seed ``data/synthetic/`` with one batch containing one approved case."""
    synth_dir = project_dir / "data" / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    batch_id = "2026-04-22T10-00-00_paraphrase"
    (synth_dir / f"{batch_id}.csv").write_text(
        "input,expected_output\nq_synth_1,a_synth_1\n"
    )
    manifest = {
        "batch_id": batch_id,
        "timestamp": "2026-04-22T10-00-00",
        "strategy": "paraphrase",
        "total_cases": 1,
        "field_mappings": {
            "input": "input",
            "expected_output": "expected_output",
        },
        "csv_file": f"{batch_id}.csv",
        "cases": [{"index": 0, "status": "approved"}],
    }
    (synth_dir / f"{batch_id}.manifest.json").write_text(json.dumps(manifest))


class TestDispatchEvalDataSource:
    """Exercises the ``data_source`` wiring on the dispatch_eval MCP tool."""

    async def test_raw_default_passes_no_override(self, project):
        """Omitted data_source ⇒ harness called with data_file_override=None."""
        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir"
        ) as mock_resolve, patch(
            "app.services.harness_runner.run_eval_via_harness",
            new_callable=AsyncMock,
        ) as mock_harness, patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_resolve.return_value = project["project_dir"]

            result = await _run_eval({
                "project_id": project["project_id"],
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
            })
            assert "is_error" not in result, result

            # Give the spawned task a chance to dispatch
            for _ in range(20):
                await asyncio.sleep(0.01)

            assert mock_harness.await_count == 1
            kwargs = mock_harness.await_args.kwargs
            assert kwargs.get("data_file_override") is None

    async def test_synthetic_resolves_processed_file(self, project):
        """data_source=synthetic writes a processed CSV and forwards its path."""
        _seed_approved_batch(project["project_dir"])

        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir"
        ) as mock_resolve, patch(
            "app.services.harness_runner.run_eval_via_harness",
            new_callable=AsyncMock,
        ) as mock_harness, patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_resolve.return_value = project["project_dir"]

            result = await _run_eval({
                "project_id": project["project_id"],
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
                "data_source": "synthetic",
            })
            assert "is_error" not in result, result

            for _ in range(20):
                await asyncio.sleep(0.01)

            assert mock_harness.await_count == 1
            override = mock_harness.await_args.kwargs.get("data_file_override")
            assert override is not None
            assert "synthetic_only_" in override
            assert Path(override).exists()

    async def test_fences_when_optimization_running(self, project):
        """Active optimization loop ⇒ dispatch_eval returns mcp_error, never dispatches.

        Prevents the "phantom v1 eval" bug where the coordinator double-calls
        dispatch_eval after dispatch_optimization already kicked off iteration 1.
        """
        from app.services.optimization_state import OptimizationState

        OptimizationState.create(
            project_dir=project["project_dir"],
            target_metric_scores={"Correctness": 1.0},
            max_iterations=10,
            plateau_window=3,
        )
        # create() sets status="running" by default.

        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir"
        ) as mock_resolve, patch(
            "app.services.harness_runner.run_eval_via_harness",
            new_callable=AsyncMock,
        ) as mock_harness:
            mock_resolve.return_value = project["project_dir"]

            result = await _run_eval({
                "project_id": project["project_id"],
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
            })

            assert result.get("is_error") is True, result
            text = result["content"][0]["text"]
            assert "optimization loop" in text.lower()
            assert mock_harness.await_count == 0

    async def test_allows_run_when_optimization_converged(self, project):
        """Converged optimization ⇒ dispatch_eval proceeds normally (for post-loop manual evals)."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=project["project_dir"],
            target_metric_scores={"Correctness": 1.0},
            max_iterations=10,
            plateau_window=3,
        )
        state.status = "converged"
        state.save()

        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir"
        ) as mock_resolve, patch(
            "app.services.harness_runner.run_eval_via_harness",
            new_callable=AsyncMock,
        ) as mock_harness, patch(
            "app.services.mcp_tools.dispatch._preflight_models_invocable",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_resolve.return_value = project["project_dir"]

            result = await _run_eval({
                "project_id": project["project_id"],
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
            })
            assert "is_error" not in result, result

            for _ in range(20):
                await asyncio.sleep(0.01)
            assert mock_harness.await_count == 1

    async def test_synthetic_rejects_when_no_approvals(self, project):
        """No approved batch ⇒ tool returns mcp_error and never dispatches."""
        with patch(
            "app.services.mcp_tools.dispatch.resolve_project_dir"
        ) as mock_resolve, patch(
            "app.services.harness_runner.run_eval_via_harness",
            new_callable=AsyncMock,
        ) as mock_harness:
            mock_resolve.return_value = project["project_dir"]

            result = await _run_eval({
                "project_id": project["project_id"],
                "data_file": "rag-qa-10.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["us.anthropic.claude-haiku-4-5-20251001-v1:0"],
                "data_source": "synthetic",
            })
            # Tool should mark the result as an error
            assert result.get("is_error") is True, result
            text = result["content"][0]["text"]
            assert "approved synthetic" in text.lower()

            # No harness dispatch
            assert mock_harness.await_count == 0
