# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synthesis MCP tools (generate_synthetic, list_synthetic_batches,
approve_synthetic_cases).

Mocks DeepEval's ``Synthesizer`` via ``app.services.synthesizer.DeepEvalSynthesizer``
and ``BedrockRuntimeModel`` so no real Bedrock traffic is generated. Focus is on the
tool-layer dispatch + error surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepeval.dataset import Golden

from app.services.mcp_tools.synthesis import (
    approve_synthetic_cases as _approve_tool,
    generate_synthetic as _generate_tool,
    list_synthetic_batches as _list_tool,
)

# SdkMcpTool wraps the handler — extract the raw async function for pytest.
generate_synthetic = _generate_tool.handler
list_synthetic_batches = _list_tool.handler
approve_synthetic_cases = _approve_tool.handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path, with_harness: bool = True, with_data: bool = True) -> Path:
    """Build a minimal project directory with an optional harness + seed data.

    Writes a realistic ``src/config.py`` that exposes ``data_file`` and
    ``field_mappings`` so ``load_harness_config`` parses them correctly.
    """
    project_dir = tmp_path / "test-project-id"
    project_dir.mkdir()
    (project_dir / "config.json").write_text(json.dumps({
        "id": "test-project-id", "name": "Synth Test", "description": "",
        "created_at": "2026-01-01",
    }))

    if with_harness:
        (project_dir / "src").mkdir()
        (project_dir / "src" / "config.py").write_text("""
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    data_file: str = "seed.csv"
    field_mappings: dict = field(default_factory=lambda: {
        "input": "question",
        "expected_output": "answer",
        "category": "topic",
    })
""")

    if with_data:
        raw_dir = project_dir / "data" / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "seed.csv").write_text(
            "question,answer,topic\n"
            "What is 2+2?,4,math\n"
            "Capital of France?,Paris,geography\n"
        )

    return project_dir


@pytest.fixture
def mock_deepeval_synth():
    """Replace DeepEvalSynthesizer with a MagicMock that exposes sane defaults."""
    with patch("app.services.synthesizer.DeepEvalSynthesizer") as mock_cls:
        instance = MagicMock()
        instance.async_mode = False
        instance.max_concurrent = 10
        instance.set_styling_config = False
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_bedrock():
    """Patch BedrockRuntimeModel inside the synthesis tool module so we never hit AWS."""
    with patch("app.services.mcp_tools.synthesis.BedrockRuntimeModel") as m:
        m.return_value = MagicMock(name="BedrockRuntimeModel")
        yield m


# ---------------------------------------------------------------------------
# generate_synthetic
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    """Tests for the ``generate_synthetic`` MCP tool."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_rejects_unknown_strategy(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Unknown strategy names should produce a structured error."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "bogus",
        })

        assert result.get("is_error") is True
        assert "unsupported" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_missing_harness_errors(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Running without a harness config should return a clear error."""
        mock_resolve.return_value = _make_project(tmp_path, with_harness=False)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "paraphrase",
        })

        assert result.get("is_error") is True
        assert "harness" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_paraphrase_missing_data_file(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Paraphrase should error when the harness data file doesn't exist."""
        mock_resolve.return_value = _make_project(tmp_path, with_data=False)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "paraphrase",
        })

        assert result.get("is_error") is True
        assert "seed data file not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_paraphrase_happy_path(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Paraphrase should load seed cases + write CSV + manifest."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        # DeepEval returns a single new golden.
        mock_deepeval_synth.generate_goldens_from_goldens.return_value = [
            Golden(input="What is 3+3?", expected_output="6"),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })

        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["strategy"] == "paraphrase"
        assert payload["total_cases"] == 1
        assert payload["status"] == "pending_review"

        # CSV + manifest on disk.
        synthetic_dir = project_dir / "data" / "synthetic"
        assert any(p.suffix == ".csv" for p in synthetic_dir.iterdir())
        assert any(p.name.endswith(".manifest.json") for p in synthetic_dir.iterdir())

        # DeepEval was called with two Goldens (the seed rows).
        call = mock_deepeval_synth.generate_goldens_from_goldens.call_args
        assert len(call.kwargs["goldens"]) == 2
        assert call.kwargs["max_goldens_per_golden"] == 1

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_from_docs_requires_paths(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """from_docs without ``document_paths`` should error."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "from_docs",
        })

        assert result.get("is_error") is True
        assert "document_paths" in result["content"][0]["text"]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_from_docs_missing_file(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """from_docs should error when a document path doesn't exist."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "from_docs",
            "document_paths": ["missing.pdf"],
        })

        assert result.get("is_error") is True
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_from_docs_happy_path(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """from_docs should forward resolved absolute paths to DeepEval."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        (project_dir / "data" / "raw" / "guide.txt").write_text("example content")
        mock_deepeval_synth.generate_goldens_from_docs.return_value = [
            Golden(input="explain X", expected_output="..."),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "from_docs",
            "document_paths": ["guide.txt"],
        })

        assert "is_error" not in result
        call = mock_deepeval_synth.generate_goldens_from_docs.call_args
        assert call.kwargs["document_paths"][0].endswith("guide.txt")

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_from_scratch_missing_fields(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """from_scratch should error when subject/task/output_format are missing."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "from_scratch",
            "subject": "math",
        })

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "task" in text and "output_format" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_from_scratch_happy_path(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """from_scratch should configure styling + call DeepEval with num_goldens."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        mock_deepeval_synth.generate_goldens_from_scratch.return_value = [
            Golden(input="q1", expected_output="a1"),
            Golden(input="q2", expected_output="a2"),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "from_scratch",
            "subject": "medical Q&A",
            "task": "answer the user's question",
            "output_format": "one sentence",
            "num_cases": 2,
        })

        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["total_cases"] == 2

        styling = mock_deepeval_synth.styling_config
        assert styling.scenario == "medical Q&A"
        mock_deepeval_synth.generate_goldens_from_scratch.assert_called_once_with(num_goldens=2)


# ---------------------------------------------------------------------------
# generate_synthetic — failure_targeted strategy
# ---------------------------------------------------------------------------


def _write_results_file(
    project_dir: Path,
    filename: str = "run_2026-04-22.json",
    *,
    num_failures: int = 3,
    test_run_id: str = "run-42",
) -> Path:
    """Write a fake eval-results JSON under ``output/`` and return its path.

    Shape mirrors the fixture in ``test_analyze_failures.py``. We emit a
    small set of passing cases plus ``num_failures`` failing ones so the
    analyzer produces a non-trivial breakdown.
    """
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One passing case (math) + N failing cases spread across categories.
    results = [{
        "name": "case_pass",
        "success": True,
        "input": "What is 2+2?",
        "actual_output": "4",
        "expected_output": "4",
        "additional_metadata": {"id": "p1", "category": "math"},
        "metrics_data": [{"name": "ExactMatch", "threshold": 1.0, "success": True, "score": 1.0, "reason": "match"}],
    }]
    for i in range(num_failures):
        category = "spelling" if i == 0 else "math"
        results.append({
            "name": f"case_fail_{i}",
            "success": False,
            "input": f"fail input {i}",
            "actual_output": f"wrong {i}",
            "expected_output": f"right {i}",
            "additional_metadata": {"id": f"f{i}", "category": category},
            "metrics_data": [{
                "name": "ExactMatch", "threshold": 1.0,
                "success": False, "score": 0.0, "reason": f"Expected: right {i}",
            }],
        })

    payload = [{
        "test_run_id": test_run_id,
        "timestamp": "2026-04-22T10:00:00",
        "hyperparameters": {"model": test_run_id, "prompt_version": "baseline"},
        "results": results,
    }]

    path = output_dir / filename
    path.write_text(json.dumps(payload))
    return path


class TestGenerateSyntheticFailureTargeted:
    """Tests for the ``failure_targeted`` branch of ``generate_synthetic``."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_happy_path_writes_batch_with_seed_run_id(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """End-to-end: read failures, generate, write manifest with seed_run_id set."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        _write_results_file(project_dir, "run_A.json", num_failures=3, test_run_id="run-42")

        # DeepEval returns a fixed list of goldens the tool should persist.
        mock_deepeval_synth.generate_goldens_from_scratch.return_value = [
            Golden(input="synthetic q1", expected_output="a1"),
            Golden(input="synthetic q2", expected_output="a2"),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "failure_targeted",
            "seed_run_filename": "run_A.json",
            "num_cases": 2,
        })

        assert "is_error" not in result, result
        payload = json.loads(result["content"][0]["text"])
        assert payload["strategy"] == "failure_targeted"
        assert payload["total_cases"] == 2
        assert payload["status"] == "pending_review"

        # Manifest should carry the test_run_id from the picked analysis,
        # plus a seed_source pointing at the file we mined.
        synthetic_dir = project_dir / "data" / "synthetic"
        manifest_paths = list(synthetic_dir.glob("*.manifest.json"))
        assert len(manifest_paths) == 1
        manifest = json.loads(manifest_paths[0].read_text())
        assert manifest["strategy"] == "failure_targeted"
        assert manifest["seed_run_id"] == "run-42"
        assert manifest["seed_source"] == "output/run_A.json"

        # DeepEval was called with num_goldens=2 and styling config derived
        # from the failure analysis (worst category = spelling).
        mock_deepeval_synth.generate_goldens_from_scratch.assert_called_once_with(num_goldens=2)
        styling = mock_deepeval_synth.styling_config
        assert "spelling" in styling.scenario.lower()
        assert "Example 1:" in styling.task  # embedded failing-case exemplar

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_falls_back_to_most_recent_run(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Without ``seed_run_filename`` we should mine the latest *.json."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        import time
        older = _write_results_file(project_dir, "old.json", num_failures=2, test_run_id="old-run")
        # Tweak mtimes so "latest.json" is unambiguously newer.
        old_mtime = older.stat().st_mtime - 10
        import os as _os
        _os.utime(older, (old_mtime, old_mtime))
        time.sleep(0.01)
        _write_results_file(project_dir, "latest.json", num_failures=3, test_run_id="latest-run")

        mock_deepeval_synth.generate_goldens_from_scratch.return_value = [
            Golden(input="q", expected_output="a"),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "failure_targeted",
            "num_cases": 1,
        })

        assert "is_error" not in result, result
        manifest = json.loads(
            next((project_dir / "data" / "synthetic").glob("*.manifest.json")).read_text()
        )
        assert manifest["seed_source"] == "output/latest.json"
        assert manifest["seed_run_id"] == "latest-run"

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_not_enough_failures_errors(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Fewer than MIN_FAILURES_FOR_SEEDING failures → mcp_error."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        _write_results_file(project_dir, "tiny.json", num_failures=1)

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "failure_targeted",
            "seed_run_filename": "tiny.json",
        })

        assert result.get("is_error") is True
        assert "not enough failures" in result["content"][0]["text"].lower()
        # DeepEval must NOT have been invoked — we bailed early.
        mock_deepeval_synth.generate_goldens_from_scratch.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_no_run_files_errors(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """With no eval results at all, the tool should return a clear error."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        # Intentionally DO NOT create output/ or any results files.

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "failure_targeted",
        })

        assert result.get("is_error") is True
        text = result["content"][0]["text"].lower()
        assert "no eval" in text or "not found" in text
        mock_deepeval_synth.generate_goldens_from_scratch.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_explicit_filename_missing_errors(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Explicit ``seed_run_filename`` that doesn't exist → mcp_error."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        (project_dir / "output").mkdir()

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "failure_targeted",
            "seed_run_filename": "nope.json",
        })

        assert result.get("is_error") is True
        assert "not found" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# list_synthetic_batches
# ---------------------------------------------------------------------------

class TestListSyntheticBatches:
    """Tests for the ``list_synthetic_batches`` MCP tool."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_empty_when_no_folder(self, mock_resolve, tmp_path):
        """With no synthetic folder, list should return an empty batches array."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await list_synthetic_batches({"project_id": "test-project-id"})

        assert "is_error" not in result
        assert json.loads(result["content"][0]["text"]) == {"batches": []}

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_lists_manifest_counts(self, mock_resolve, tmp_path):
        """Per-batch pending/approved/rejected counts should be correct."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        synth_dir = project_dir / "data" / "synthetic"
        synth_dir.mkdir()
        (synth_dir / "2026-04-22T10-00-00_paraphrase.manifest.json").write_text(json.dumps({
            "batch_id": "2026-04-22T10-00-00_paraphrase",
            "strategy": "paraphrase",
            "timestamp": "2026-04-22T10-00-00",
            "total_cases": 3,
            "csv_file": "2026-04-22T10-00-00_paraphrase.csv",
            "cases": [
                {"index": 0, "status": "pending"},
                {"index": 1, "status": "approved"},
                {"index": 2, "status": "rejected"},
            ],
        }))
        (synth_dir / "ignored.txt").write_text("noise")

        result = await list_synthetic_batches({"project_id": "test-project-id"})

        payload = json.loads(result["content"][0]["text"])
        assert len(payload["batches"]) == 1
        b = payload["batches"][0]
        assert b["strategy"] == "paraphrase"
        assert b["pending"] == 1
        assert b["approved"] == 1
        assert b["rejected"] == 1
        assert b["total_cases"] == 3


# ---------------------------------------------------------------------------
# approve_synthetic_cases
# ---------------------------------------------------------------------------

class TestApproveSyntheticCases:
    """Tests for the ``approve_synthetic_cases`` MCP tool."""

    def _write_batch(self, project_dir: Path, batch_id: str) -> Path:
        synth_dir = project_dir / "data" / "synthetic"
        synth_dir.mkdir(parents=True)
        manifest_path = synth_dir / f"{batch_id}.manifest.json"
        manifest_path.write_text(json.dumps({
            "batch_id": batch_id,
            "strategy": "paraphrase",
            "timestamp": "2026-04-22T10-00-00",
            "total_cases": 3,
            "csv_file": f"{batch_id}.csv",
            "cases": [
                {"index": 0, "status": "pending"},
                {"index": 1, "status": "pending"},
                {"index": 2, "status": "pending"},
            ],
        }))
        return manifest_path

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_invalid_batch_id(self, mock_resolve, tmp_path):
        """Unknown batch_id should return a structured error."""
        mock_resolve.return_value = _make_project(tmp_path)

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "batch_id": "nope",
            "approve_all": True,
        })

        assert result.get("is_error") is True
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_requires_some_action(self, mock_resolve, tmp_path):
        """Calling without approve/reject/approve_all should error."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        self._write_batch(project_dir, "b1")

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "batch_id": "b1",
        })

        assert result.get("is_error") is True

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_approve_and_reject_by_index(self, mock_resolve, tmp_path):
        """Mixed approve/reject indices should update the manifest + counts."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        manifest_path = self._write_batch(project_dir, "b1")

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "batch_id": "b1",
            "approve_indices": [0, 1],
            "reject_indices": [2],
        })

        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["approved"] == 2
        assert payload["rejected"] == 1
        assert payload["pending"] == 0

        saved = json.loads(manifest_path.read_text())
        statuses = [c["status"] for c in saved["cases"]]
        assert statuses == ["approved", "approved", "rejected"]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_approve_all(self, mock_resolve, tmp_path):
        """``approve_all`` should flip every pending case to approved."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        self._write_batch(project_dir, "b2")

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "batch_id": "b2",
            "approve_all": True,
        })

        payload = json.loads(result["content"][0]["text"])
        assert payload["approved"] == 3
        assert payload["pending"] == 0


class TestGenerationModelRequired:
    """Synthetic generation never falls back to a hardcoded model.

    ``generation_model_id`` used to be optional and silently defaulted to
    the adapter's baked-in model id — synthetic data generated by a model
    nobody picked. Missing id is now an explicit tool error."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_missing_generation_model_id_is_tool_error(
        self, mock_resolve, tmp_path
    ):
        mock_resolve.return_value = _make_project(tmp_path)

        result = await generate_synthetic({
            "project_id": "p-no-model",
            "strategy": "paraphrase",
        })

        assert result.get("is_error") is True
        assert "generation_model_id" in result["content"][0]["text"]


class TestUnscoreableSyntheticCases:
    """A synthetic case with no ground truth must not reach a dataset.

    A generated batch can contain cases whose ``expected_output`` is EMPTY even
    though the field mapping declares one. Any metric that reads
    expected_output fails such a case whatever the model produces, so
    approving it books guaranteed failures and drags every pass rate down for a
    reason unrelated to the model. Human review was the only backstop, and
    ``approve_all`` exists precisely to skip review.
    """

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_generation_reports_cases_with_no_expected_output(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Surfaced at GENERATION time: it changes whether the batch is worth
        reviewing at all."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_deepeval_synth.generate_goldens_from_goldens.return_value = [
            Golden(input="What is 3+3?", expected_output="6"),
            Golden(input="What is 4+4?", expected_output=""),
            Golden(input="What is 5+5?", expected_output=None),
        ]

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "test-gen-model",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })
        payload = json.loads(result["content"][0]["text"])
        assert payload["unscoreable_cases"] == 2
        assert "EMPTY expected output" in payload["warning"]

        manifest = json.loads(
            next(
                p for p in (project_dir / "data" / "synthetic").iterdir()
                if p.name.endswith(".manifest.json")
            ).read_text()
        )
        assert manifest["unscoreable_cases"] == 2
        flagged = [c["index"] for c in manifest["cases"] if c.get("unscoreable")]
        assert flagged == [1, 2]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_approve_all_holds_back_the_unscoreable_ones(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Approve the good ones, leave the rest pending, say how many.

        Refusing the whole call would punish an 18-good/2-bad batch; approving
        everything is the bug.
        """
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_deepeval_synth.generate_goldens_from_goldens.return_value = [
            Golden(input="a", expected_output="A"),
            Golden(input="b", expected_output=""),
        ]
        gen = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "m",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })
        batch_id = json.loads(gen["content"][0]["text"])["batch_id"]

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "batch_id": batch_id,
            "approve_all": True,
        })
        payload = json.loads(result["content"][0]["text"])
        assert payload["approved"] == 1
        assert payload["pending"] == 1
        assert payload["held_back_unscoreable"] == 1
        assert "no expected output" in payload["note"]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_naming_an_unscoreable_index_is_refused(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Explicit beats convenient: the caller named this case, so silently
        skipping it would report an approval that did not happen."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_deepeval_synth.generate_goldens_from_goldens.return_value = [
            Golden(input="a", expected_output="A"),
            Golden(input="b", expected_output=""),
        ]
        gen = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "m",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })
        batch_id = json.loads(gen["content"][0]["text"])["batch_id"]

        result = await approve_synthetic_cases({
            "project_id": "test-project-id",
            "batch_id": batch_id,
            "approve_indices": [1],
        })
        assert result.get("is_error") is True
        assert "[1]" in result["content"][0]["text"]

        # And nothing was written: a refused call must not half-apply.
        manifest = json.loads(
            next(
                p for p in (project_dir / "data" / "synthetic").iterdir()
                if p.name.endswith(".manifest.json")
            ).read_text()
        )
        assert all(c["status"] == "pending" for c in manifest["cases"])

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_synthesis_refuses_a_mapping_without_expected_output(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        """Why the flag can be unconditional on this path.

        The tool already requires both ``input`` and ``expected_output`` in
        ``field_mappings``, so every synthetic batch HAS a ground-truth column
        and an empty value in it is always a defect. Pinned because the guard's
        condition reads ``if expected_col and ...`` — if this requirement were
        ever relaxed, that condition is what stops the flag becoming noise for a
        dataset that legitimately has no expected outputs.
        """
        project_dir = _make_project(tmp_path)
        (project_dir / "src" / "config.py").write_text("""
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    data_file: str = "seed.csv"
    field_mappings: dict = field(default_factory=lambda: {"input": "question"})
""")
        mock_resolve.return_value = project_dir

        result = await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "m",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })
        assert result.get("is_error") is True
        assert "expected_output" in result["content"][0]["text"]


class TestParaphraseSeedsCarryContext:
    """Why the empty expected outputs happened in the first place.

    Read from DeepEval 3.9.9's source: ``generate_goldens_from_goldens``
    collects ``golden.context`` from the seeds and, when that list is EMPTY,
    calls ``generate_goldens_from_scratch(...)`` — which takes no
    ``include_expected_output`` parameter at all. So seeds without context turn
    "paraphrase these cases" into "invent new ones with no ground truth", using
    the seeds only for prompt styling. That is how a batch ends up with empty
    expected outputs, and it is silent.
    """

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.synthesis.resolve_project_dir")
    async def test_every_seed_golden_carries_its_pair_as_context(
        self, mock_resolve, tmp_path, mock_deepeval_synth, mock_bedrock
    ):
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_deepeval_synth.generate_goldens_from_goldens.return_value = [
            Golden(input="x", expected_output="y"),
        ]

        await generate_synthetic({
            "project_id": "test-project-id",
            "generation_model_id": "m",
            "strategy": "paraphrase",
            "max_goldens_per_seed": 1,
        })

        seeds = mock_deepeval_synth.generate_goldens_from_goldens.call_args.kwargs[
            "goldens"
        ]
        assert seeds, "no seeds were passed"
        for seed in seeds:
            assert seed.context, (
                "a seed with no context makes DeepEval discard the seeds and "
                "generate from scratch, which cannot produce expected outputs"
            )
            # Grounded on the seed's OWN pair — that is what makes the generated
            # case a paraphrase of it rather than an unrelated invention.
            assert seed.input in seed.context
            assert seed.expected_output in seed.context
