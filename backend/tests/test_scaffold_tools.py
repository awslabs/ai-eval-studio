# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaffold_harness and validate_harness MCP tools.

Covers template copying, placeholder substitution, file protection during
re-scaffolding, the validate_harness pre-flight checks, and pricing file
generation after successful validation.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.mcp_tools import harness as harness_mod
from app.services.mcp_tools.harness import scaffold_harness as _scaffold_tool, validate_harness as _validate_tool

# SdkMcpTool wraps the handler — extract the actual async function
scaffold_harness = _scaffold_tool.handler
validate_harness = _validate_tool.handler


def _make_project(tmp_path: Path, name: str = "test-project") -> Path:
    """Create a minimal project directory with config.json."""
    project_dir = tmp_path / "test-project-id"
    project_dir.mkdir(exist_ok=True)
    config = {"id": "test-project-id", "name": name, "description": "", "created_at": "2026-01-01"}
    (project_dir / "config.json").write_text(json.dumps(config))
    return project_dir


#: Config-validation payload for step 3, shaped as the real step emits it.
_VALIDATION_JSON = json.dumps({
    "status": "ok",
    "judge_model": "us.anthropic.claude-sonnet-4-6-v1:0",
    "test_cases": 5,
    "models": 1,
    "all_model_ids": ["us.anthropic.claude-sonnet-4-6-v1:0"],
    "target_models": ["us.anthropic.claude-sonnet-4-6-v1:0"],
    "aws_region": "us-east-1",
    "prompt_version": "baseline",
    "data_file": "qa.csv",
}).encode()


def _prepare_validated_project(tmp_path: Path) -> tuple[Path, Path]:
    """Scaffold a project that clears every validate_harness step before step 6.

    Returns ``(project_dir, smoke_output_file)``. The smoke output is pre-written
    with a healthy result; the caller's subprocess router touches it so its mtime
    lands at or after the launch instant, which is what the validator requires.
    """
    project_dir = _make_project(tmp_path)
    for f in ["run_eval.py", "pyproject.toml"]:
        (project_dir / f).write_text("# stub")
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "config.py").write_text("# stub")
    (project_dir / "src" / "evaluation").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "evaluation" / "dataset_loader.py").write_text("# stub")
    (project_dir / "src" / "metrics").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "metrics" / "metrics.py").write_text("# stub")

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_output = [{
        "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
        "results": [{
            "name": "case_0",
            "success": True,
            "input": "x",
            "actual_output": "y",
            "expected_output": "y",
            "additional_metadata": {"latency_ms": 42},
            "metrics_data": [{
                "name": "ExactMatch",
                "score": 1.0,
                "threshold": 1.0,
                "success": True,
                "reason": "matched",
            }],
        }],
    }]
    output_file = output_dir / "eval_results_baseline_stub.json"
    output_file.write_text(json.dumps(smoke_output))
    return project_dir, output_file


def _subprocess_router(output_file: Path, selftest):
    """Build an ``asyncio.create_subprocess_exec`` stand-in for validate_harness.

    Routes by command args rather than call order, so inserting a step does not
    silently re-point an existing branch. ``selftest`` is called with the mock
    process for the metric self-test invocation, letting each test decide how the
    probe behaves (clean JSON, a hang, unreadable output).

    The self-test branch is matched FIRST and on a probe-only marker: the probe's
    source also contains ``EvalConfig()``, so ordering it after the config branch
    would route the probe into the config step's payload.
    """
    async def mock_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = 0
        str_args = [a for a in args if isinstance(a, str)]

        if any("_degenerate_check" in a for a in str_args):
            selftest(proc)
        elif "sync" in str_args:
            proc.communicate.return_value = (b"", b"")
        elif any("EvalConfig()" in a for a in str_args):
            proc.communicate.return_value = (_VALIDATION_JSON, b"")
        elif "run_eval.py" in str_args:
            proc.communicate.return_value = (b"", b"")
            output_file.touch()
        else:
            raise AssertionError(f"Unexpected subprocess invocation: {args!r}")
        return proc

    return mock_subprocess


class TestScaffoldHarness:
    """Tests for the scaffold_harness MCP tool."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_creates_template_files(self, mock_resolve, tmp_path):
        """scaffold_harness should create all template files in the project."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        result = await scaffold_harness({"project_id": "test-project-id"})

        assert "is_error" not in result
        text = result["content"][0]["text"]
        assert "run_eval.py" in text

        # Verify key template files exist
        assert (project_dir / "run_eval.py").exists()
        assert (project_dir / "src" / "models" / "bedrock_runtime_model.py").exists()
        assert (project_dir / "src" / "evaluation" / "runner.py").exists()
        assert (project_dir / "pyproject.toml").exists()
        assert (project_dir / "tests" / "test_smoke.py").exists()
        assert (project_dir / "README.md").exists()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_substitutes_project_name(self, mock_resolve, tmp_path):
        """scaffold_harness should replace {project_name} in .toml and .md files.

        Display names are slugged into PEP 503-compatible package names for the
        pyproject.toml ``[project].name`` field (spaces and punctuation →
        dashes), so ``uv sync`` doesn't reject the scaffolded project.
        """
        project_dir = _make_project(tmp_path, name="My Cool Project")
        mock_resolve.return_value = project_dir

        await scaffold_harness({"project_id": "test-project-id"})

        toml_content = (project_dir / "pyproject.toml").read_text()
        assert "{project_name}" not in toml_content
        assert "My-Cool-Project" in toml_content

        readme_content = (project_dir / "README.md").read_text()
        assert "{project_name}" not in readme_content

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_slugs_special_chars_in_project_name(self, mock_resolve, tmp_path):
        """Display names with slashes, dots, or unicode must yield a valid package name."""
        project_dir = _make_project(tmp_path, name="Customer Support QA v2")
        mock_resolve.return_value = project_dir

        await scaffold_harness({"project_id": "test-project-id"})

        toml_content = (project_dir / "pyproject.toml").read_text()
        assert "Customer-Support-QA-v2" in toml_content
        # Must not leak raw space characters into the name field.
        assert 'name = "Customer Support QA' not in toml_content

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_pins_deepeval_to_backend_version(self, mock_resolve, tmp_path):
        """scaffold_harness must pin the project's DeepEval to the backend's exact version."""
        from importlib.metadata import version as pkg_version
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        await scaffold_harness({"project_id": "test-project-id"})

        toml_content = (project_dir / "pyproject.toml").read_text()
        assert f'"deepeval=={pkg_version("deepeval")}"' in toml_content
        assert "{deepeval_version}" not in toml_content

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_protects_agent_written_files(self, mock_resolve, tmp_path):
        """scaffold_harness should not overwrite config.py or dataset_loader.py."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        # Create agent-written files first
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "config.py").write_text("# agent-written config")
        (project_dir / "src" / "evaluation").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "evaluation" / "dataset_loader.py").write_text("# agent-written loader")
        (project_dir / "src" / "metrics").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "metrics" / "__init__.py").write_text("# agent-written metrics")

        result = await scaffold_harness({"project_id": "test-project-id"})

        # Agent-written files should be preserved
        assert (project_dir / "src" / "config.py").read_text() == "# agent-written config"
        assert (project_dir / "src" / "evaluation" / "dataset_loader.py").read_text() == "# agent-written loader"
        assert (project_dir / "src" / "metrics" / "__init__.py").read_text() == "# agent-written metrics"

        # Template files should still be created
        assert (project_dir / "run_eval.py").exists()

        # Skipped files mentioned in output
        text = result["content"][0]["text"]
        assert "skipped" in text.lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_does_not_mirror_skills_into_project(self, mock_resolve, tmp_path):
        """scaffold_harness must NOT create a per-project ``.claude/skills/``.

        Skills are installed at user-scope (``$HOME/.claude/skills/``) in
        the Dockerfile and auto-discovered by the bundled claude CLI.
        A per-project mirror would be N copies of the same skill files
        across N projects on EFS, with re-scaffold required after every
        skill update — the wrong shape. This test guards against the
        mirror getting re-introduced by accident.
        """
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        await scaffold_harness({"project_id": "test-project-id"})

        skills_dir = project_dir / ".claude" / "skills"
        assert not skills_dir.exists(), (
            f"scaffold_harness created {skills_dir} — skills should be "
            "installed user-scope at $HOME/.claude/skills/ instead, not "
            "mirrored per-project."
        )

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_re_scaffold_updates_templates(self, mock_resolve, tmp_path):
        """Re-scaffolding should update template files but preserve agent files."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        # First scaffold
        await scaffold_harness({"project_id": "test-project-id"})

        # Write agent file
        (project_dir / "src" / "config.py").write_text("# custom config")

        # Re-scaffold
        await scaffold_harness({"project_id": "test-project-id"})

        # Template files re-created, agent file preserved
        assert (project_dir / "run_eval.py").exists()
        assert (project_dir / "src" / "config.py").read_text() == "# custom config"


class TestValidateHarness:
    """Tests for the validate_harness MCP tool."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_reports_missing_files(self, mock_resolve, tmp_path):
        """validate_harness should report which required files are missing."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "run_eval.py" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_reports_partial_missing(self, mock_resolve, tmp_path):
        """validate_harness should list only missing files, not present ones."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        # Create some but not all required files
        (project_dir / "run_eval.py").write_text("#!/usr/bin/env python3")
        (project_dir / "pyproject.toml").write_text("[project]\nname = 'test'")

        result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        # Should report missing config.py and dataset_loader
        assert "src/config.py" in text
        assert "src/evaluation/dataset_loader.py" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_writes_pricing_after_validation(self, mock_resolve, mock_write_pricing, tmp_path):
        """validate_harness should call write_pricing_file after config passes."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        def _clean_selftest(proc):
            proc.communicate.return_value = (
                json.dumps({"status": "ok", "errors": [], "metrics": [{
                    "name": "ExactMatch",
                    "class": "ExactMatchMetric",
                    "judge_backed": False,
                    "required_params": ["input", "actual_output", "expected_output"],
                    "degenerate": {"verdict": "ok", "detail": None},
                    "formatting": {"ran": True, "rejected": [], "accepted": [],
                                   "errors": []},
                }]}).encode(),
                b"",
            )

        router = _subprocess_router(output_file, _clean_selftest)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        # Should succeed
        assert "is_error" not in result
        text = result["content"][0]["text"]
        assert "validation passed" in text.lower() or "ready" in text.lower()
        assert "Metric self-test passed" in text

        # write_pricing_file should have been called with the model IDs
        mock_write_pricing.assert_called_once()
        call_args = mock_write_pricing.call_args
        assert call_args[0][0] == project_dir
        assert "us.anthropic.claude-sonnet-4-6-v1:0" in call_args[0][1]


def _stub_project_for_smoke(tmp_path: Path) -> Path:
    """Create a minimal project that passes the file, sync and config checks.

    Used by the smoke-eval test cases below so they can focus on behavior
    of Step 5 (the smoke eval) without duplicating scaffolding boilerplate.
    """
    project_dir = tmp_path / "test-project-id"
    project_dir.mkdir(exist_ok=True)
    config = {"id": "test-project-id", "name": "smoke", "description": "", "created_at": "2026-01-01"}
    (project_dir / "config.json").write_text(json.dumps(config))

    for f in ["run_eval.py", "pyproject.toml"]:
        (project_dir / f).write_text("# stub")
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "config.py").write_text("# stub")
    (project_dir / "src" / "evaluation").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "evaluation" / "dataset_loader.py").write_text("# stub")
    (project_dir / "src" / "metrics").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "metrics" / "metrics.py").write_text("# stub")

    return project_dir


_SMOKE_VALIDATION_JSON = json.dumps({
    "status": "ok",
    "judge_model": "us.anthropic.claude-sonnet-4-6-v1:0",
    "test_cases": 1,
    "models": 1,
    "all_model_ids": ["us.anthropic.claude-sonnet-4-6-v1:0"],
    "target_models": ["us.anthropic.claude-sonnet-4-6-v1:0"],
    "aws_region": "us-east-1",
    "prompt_version": "baseline",
    "data_file": "qa.csv",
}).encode()


def _make_mock_subprocess_factory(
    *,
    smoke_returncode: int = 0,
    smoke_stderr: bytes = b"",
    output_file: Path | None = None,
):
    """Build a subprocess mock that routes by command args.

    The sync and config steps always succeed; Step 5 (smoke eval) uses the configured
    return code and stderr so tests can exercise each failure branch.
    When ``output_file`` is passed and the smoke subprocess "succeeds",
    the file's mtime is touched to simulate a real harness write — the
    validator's mtime-floor check only recognises files modified at or
    after the subprocess was launched.
    """
    async def mock_subprocess(*args, **kwargs):
        proc = AsyncMock()
        str_args = [a for a in args if isinstance(a, str)]

        if "sync" in str_args:
            proc.returncode = 0
            proc.communicate.return_value = (b"", b"")
        elif any("EvalConfig()" in a for a in str_args):
            proc.returncode = 0
            proc.communicate.return_value = (_SMOKE_VALIDATION_JSON, b"")
        elif any("from src.config import" in a for a in str_args):
            proc.returncode = 0
            proc.communicate.return_value = (b"OK", b"")
        elif "run_eval.py" in str_args:
            proc.returncode = smoke_returncode
            proc.communicate.return_value = (b"", smoke_stderr)
            if smoke_returncode == 0 and output_file is not None and output_file.exists():
                output_file.touch()
        else:
            raise AssertionError(f"Unexpected subprocess invocation: {args!r}")
        return proc

    return mock_subprocess


def _write_smoke_output(project_dir: Path, payload: list) -> Path:
    """Write a pretend harness output file and return its path."""
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_results_baseline_stub.json"
    path.write_text(json.dumps(payload))
    return path


class TestSmokeEval:
    """Tests for Step 5 of validate_harness — the 1-case smoke eval."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_smoke_eval_fails_on_null_metric_scores(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """A run where every metric scored null must fail validation loudly."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = None

        output_file = _write_smoke_output(project_dir, [{
            "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
            "results": [{
                "name": "case_0",
                "success": False,
                "actual_output": "model said something",
                "additional_metadata": {},
                "metrics_data": [{
                    "name": "ExactMatch",
                    "score": None,
                    "success": False,
                    "reason": None,
                    "error": None,
                }],
            }],
        }])

        with patch("asyncio.create_subprocess_exec",
                   side_effect=_make_mock_subprocess_factory(output_file=output_file)), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"].lower()
        assert "null score" in text or "smoke eval" in text
        # Pricing must not be fetched on a failed smoke eval.
        mock_write_pricing.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_smoke_eval_fails_on_empty_results(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """An output with no test cases is a hard failure."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir

        output_file = _write_smoke_output(project_dir, [{
            "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
            "results": [],
        }])

        with patch("asyncio.create_subprocess_exec",
                   side_effect=_make_mock_subprocess_factory(output_file=output_file)), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        assert "empty" in result["content"][0]["text"].lower() or \
               "missing" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_smoke_eval_passes_on_well_formed_output(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """A real score + success flag + reason should validate cleanly."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        output_file = _write_smoke_output(project_dir, [{
            "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
            "results": [{
                "name": "case_0",
                "success": True,
                "actual_output": "a real model response",
                "additional_metadata": {"latency_ms": 42},
                "metrics_data": [{
                    "name": "ExactMatch",
                    "score": 0.87,
                    "success": True,
                    "reason": "close enough",
                }],
            }],
        }])

        with patch("asyncio.create_subprocess_exec",
                   side_effect=_make_mock_subprocess_factory(output_file=output_file)), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result
        text = result["content"][0]["text"].lower()
        assert "smoke eval passed" in text or "0.87" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_smoke_eval_handles_missing_output_file(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """Subprocess succeeds but wrote no output file → fail with a pointer."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir

        # Deliberately do NOT call _write_smoke_output.

        with patch("asyncio.create_subprocess_exec",
                   side_effect=_make_mock_subprocess_factory()), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        assert "output" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_smoke_eval_handles_subprocess_nonzero_exit(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """Subprocess exits 1 → return error with stderr excerpt."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir

        factory = _make_mock_subprocess_factory(
            smoke_returncode=1,
            smoke_stderr=b"Traceback (most recent call last):\nValueError: boom",
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "smoke eval" in text.lower()
        assert "boom" in text or "ValueError" in text


class TestSmokeEvalFullSweep:
    """Pin the full-sweep behavior of validate_harness Step 5: every
    target model must produce a run + non-empty output + scored metrics
    for the gate to pass. The previous shape (smoke against models[0]
    only) hid reasoning-model empty-output bugs until eval time.
    """

    @staticmethod
    def _multi_target_validation_json() -> bytes:
        return json.dumps({
            "status": "ok",
            "judge_model": "us.anthropic.claude-sonnet-4-6-v1:0",
            "test_cases": 1,
            "models": 2,
            "all_model_ids": [
                "us.anthropic.claude-sonnet-4-6-v1:0",
                "openai.gpt-5.4",
            ],
            "target_models": [
                "us.anthropic.claude-sonnet-4-6-v1:0",
                "openai.gpt-5.4",
            ],
            "aws_region": "us-east-1",
            "prompt_version": "baseline",
            "data_file": "qa.csv",
        }).encode()

    @staticmethod
    def _make_factory(*, output_file: Path, validation_json: bytes):
        async def mock_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.returncode = 0
            str_args = [a for a in args if isinstance(a, str)]
            if "sync" in str_args:
                proc.communicate.return_value = (b"", b"")
            elif any("EvalConfig()" in a for a in str_args):
                proc.communicate.return_value = (validation_json, b"")
            elif "run_eval.py" in str_args:
                proc.communicate.return_value = (b"", b"")
                output_file.touch()
            else:
                raise AssertionError(
                    f"Unexpected subprocess invocation: {args!r}"
                )
            return proc

        return mock_subprocess

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_passes_when_every_target_model_runs_cleanly(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """Smoke output with one run per target + scored metrics → gate passes."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        good_run = lambda mid: {
            "model_id": mid,
            "results": [{
                "name": "case_0",
                "success": True,
                "actual_output": "real model response",
                "additional_metadata": {"latency_ms": 42},
                "metrics_data": [{
                    "name": "ExactMatch",
                    "score": 1.0,
                    "success": True,
                    "reason": "matched",
                }],
            }],
        }
        output_file = _write_smoke_output(project_dir, [
            good_run("us.anthropic.claude-sonnet-4-6-v1:0"),
            good_run("openai.gpt-5.4"),
        ])

        factory = self._make_factory(
            output_file=output_file,
            validation_json=self._multi_target_validation_json(),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result, result["content"][0]["text"]
        text = result["content"][0]["text"]
        # Per-model summary line for both targets.
        assert "us.anthropic.claude-sonnet-4-6-v1:0" in text
        assert "openai.gpt-5.4" in text
        # Full-sweep banner (the user-facing signal that the gate covered every model).
        assert "Full-sweep smoke eval passed" in text
        assert "1 case × 2 target models" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_fails_when_a_target_model_is_missing_from_output(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """Smoke output covering only models[0] (the previous shape) →
        gate fails with the specific missing model id surfaced. This is
        the regression that lets a broken target slip through to eval."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = None

        # Only the first target produced a run. The harness either
        # crashed mid-run or skipped the second target.
        output_file = _write_smoke_output(project_dir, [{
            "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
            "results": [{
                "name": "case_0",
                "success": True,
                "actual_output": "ok",
                "additional_metadata": {"latency_ms": 42},
                "metrics_data": [{
                    "name": "ExactMatch",
                    "score": 1.0,
                    "success": True,
                    "reason": "matched",
                }],
            }],
        }])
        factory = self._make_factory(
            output_file=output_file,
            validation_json=self._multi_target_validation_json(),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"]
        assert "openai.gpt-5.4" in text, (
            "missing-model error must name the specific id so the user "
            "knows which target the harness skipped"
        )
        assert "did not produce a run for every target model" in text
        # Pricing must NOT be fetched on a failed smoke.
        mock_write_pricing.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_fails_when_one_target_model_returns_empty_output(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """Output present for both targets but one is empty (the
        reasoning-model regression: GPT-5.4 burning the budget on hidden
        reasoning and emitting empty visible text) → gate fails."""
        project_dir = _stub_project_for_smoke(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = None

        output_file = _write_smoke_output(project_dir, [
            {
                "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
                "results": [{
                    "name": "case_0",
                    "success": True,
                    "actual_output": "real response",
                    "additional_metadata": {"latency_ms": 42},
                    "metrics_data": [{
                        "name": "ExactMatch",
                        "score": 1.0,
                        "success": True,
                        "reason": "matched",
                    }],
                }],
            },
            {
                "model_id": "openai.gpt-5.4",
                "results": [{
                    "name": "case_0",
                    "success": False,
                    # Empty output is the reasoning-model bug shape.
                    "actual_output": "",
                    "additional_metadata": {"latency_ms": 30},
                    "metrics_data": [{
                        "name": "ExactMatch",
                        "score": None,
                        "success": False,
                        "reason": None,
                        "error": None,
                    }],
                }],
            },
        ])
        factory = self._make_factory(
            output_file=output_file,
            validation_json=self._multi_target_validation_json(),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        text = result["content"][0]["text"].lower()
        assert "empty output" in text
        # Pricing must NOT be fetched on a failed smoke.
        mock_write_pricing.assert_not_called()


class TestValidateCodeShape:
    """Pin the shape of Step-3's validate_code subprocess source.

    The full-sweep smoke (Step 5) is the gate that catches
    reasoning-model empty-output regressions and per-model wiring bugs.
    Step 3's job is the imports/config/data check plus emitting the
    target_models list that Step 5 invokes the harness against. These
    assertions guard against silently dropping that emission or
    reintroducing a redundant 'Say OK' probe surface.
    """

    def _extract_validate_code(self) -> str:
        """Return the validate_code triple-quoted string from harness.py."""
        import re
        from pathlib import Path
        # Resolve from __file__ so the test does not depend on the working directory.
        backend_dir = Path(__file__).parent.parent
        src = (backend_dir / "app/services/mcp_tools/harness.py").read_text(encoding="utf-8")
        m = re.search(
            r'validate_code = """(.*?)"""', src, flags=re.DOTALL
        )
        assert m, "validate_code string not found in harness.py"
        return m.group(1)

    def test_validate_code_imports_required_symbols(self):
        """Step-3 validate_code must import every adapter + descriptor
        symbol the harness uses. A missing symbol in the harness template
        must fail the gate at validate time, not at first dispatch."""
        code = self._extract_validate_code()
        assert "from src.config import EvalConfig" in code
        assert "from src.evaluation.dataset_loader import" in code
        assert "from src.metrics import build_metrics" in code
        assert "from src.models.bedrock_runtime_model import BedrockRuntimeModel" in code
        assert "from src.models.bedrock_mantle_model import BedrockMantleModel" in code
        assert "from src.models.bedrock_mantle_mappers import MantleAPIRejected" in code
        assert "from src.models.model_descriptors import read_descriptors" in code

    def test_validate_code_emits_target_models_list(self):
        """Step-3 must emit ``target_models`` so the orchestrator's
        Step-5 full-sweep smoke can invoke the harness against EVERY
        target model, not just models[0]. The previous shape (single
        ``target_model`` field) hid reasoning-model empty-output bugs
        until eval time."""
        code = self._extract_validate_code()
        assert '"target_models": list(config.models)' in code, (
            "Step-3 output must include target_models so Step-5 can run "
            "the full-sweep smoke. Without this, validation passes when "
            "models[0] works but other targets are silently broken."
        )

    def test_validate_code_no_per_model_say_ok_probe(self):
        """The previous Step-4 'Say OK' Converse probe is removed — it
        returned OK for reasoning models that burned their token budget
        on hidden reasoning and emitted empty visible text, then the
        harness silently scored 0 across the board. The full-sweep
        smoke (Step 5) is the gate now; redundant probe surface is
        forbidden."""
        code = self._extract_validate_code()
        assert "Say OK." not in code, (
            "validate_code must not contain a 'Say OK' reachability probe — "
            "the full-sweep smoke is the gate."
        )
        # No ad-hoc ThreadPoolExecutor probe sweep either.
        assert "ThreadPoolExecutor" not in code

    def test_validate_code_parses_as_python(self):
        """The validate_code subprocess source must be valid Python —
        a typo here breaks every validate_harness call at runtime, not
        at import time."""
        import ast
        code = self._extract_validate_code()
        ast.parse(code)  # raises SyntaxError on failure


class TestMetricSelfTestSeverity:
    """What the metric self-test may and may not stop a user for.

    The feature blocks only on a result that positively demonstrates a metric
    defect. Its cost, by contrast, is judge latency: measured on DeepEval 3.9.9,
    one ``measure()`` is several judge calls (AnswerRelevancy 3, Faithfulness 4,
    the contextual metrics 2 each), and the degenerate check scores 3 cases — a
    five-metric RAG lineup runs to 33 serialized judge calls. Failing validation
    on that clock would be exactly the false positive the feature already refuses
    to make when it declines to block on one nondeterministic judge sample.
    """

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_a_probe_timeout_warns_and_validation_still_passes(
        self, mock_resolve, mock_write_pricing, tmp_path, monkeypatch
    ):
        """A slow or throttled judge must not fail a harness that works."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"
        # Real wait_for, real kill path — just a budget this stub cannot meet.
        monkeypatch.setattr(harness_mod, "_SELFTEST_TIMEOUT_S", 0.05)

        def _hangs(proc):
            async def _never(*a, **k):
                await asyncio.sleep(5)
                return (b"", b"")
            proc.communicate = _never

        router = _subprocess_router(output_file, _hangs)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result, "a judge timeout must not fail validation"
        text = result["content"][0]["text"]
        assert "validation passed" in text.lower()
        assert "did not finish" in text and "NOT checked" in text
        # And the user is told it is latency, not a broken metric.
        assert "judge" in text.lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_an_unreadable_probe_result_warns_rather_than_failing(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """A crashed probe is a gap in the PROBE, which says nothing about metrics."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        def _garbage(proc):
            proc.communicate.return_value = (b"", b"Traceback: interpreter died")

        router = _subprocess_router(output_file, _garbage)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result
        text = result["content"][0]["text"]
        assert "no readable result" in text and "NOT checked" in text
        assert "interpreter died" in text

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_metrics_that_cannot_be_built_still_block(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """The load-bearing block: build_metrics raising IS a metric defect.

        Pinned alongside the warn cases so relaxing the timeout cannot be read as
        relaxing the whole check.
        """
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir

        def _cannot_build(proc):
            proc.communicate.return_value = (
                json.dumps({
                    "status": "error",
                    "errors": ["ImportError: no module named 'src.metrics'"],
                }).encode(),
                b"",
            )

        router = _subprocess_router(output_file, _cannot_build)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        assert "could not build" in result["content"][0]["text"]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_a_deterministic_stuck_metric_still_blocks(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """The other load-bearing block: conclusive for a local scorer."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir

        def _stuck(proc):
            proc.communicate.return_value = (
                json.dumps({"status": "ok", "errors": [], "metrics": [{
                    "name": "ExactMatch",
                    "judge_backed": False,
                    "degenerate": {
                        "verdict": "always_fail",
                        "detail": "failed output identical to the expected output",
                    },
                }]}).encode(),
                b"",
            )

        router = _subprocess_router(output_file, _stuck)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"):
            result = await validate_harness({"project_id": "test-project-id"})

        assert result.get("is_error") is True
        assert "cannot" in result["content"][0]["text"]


class TestMetricSelfTestOperatorTrace:
    """The self-test must leave a trace an operator can find in CloudWatch.

    Every neighbouring validation step logs a progress line; this one — the
    newest and most expensive, with judge calls against a 300s budget — logged
    nothing at all, so there was no way to confirm it ran, how long it took, or
    what it concluded, and its warnings never left the tool's rendered output.

    ``patch.object`` on the module logger rather than ``caplog``: caplog depends
    on root-logger propagation, which other tests in this suite reconfigure.
    """

    @staticmethod
    def _clean_selftest(proc):
        proc.communicate.return_value = (
            json.dumps({"status": "ok", "errors": [], "metrics": [{
                "name": "ExactMatch",
                "class": "ExactMatchMetric",
                "judge_backed": False,
                "required_params": ["input", "actual_output", "expected_output"],
                "degenerate": {"verdict": "ok", "detail": None},
                "formatting": {"ran": True, "rejected": [], "accepted": [],
                               "errors": []},
            }]}).encode(),
            b"",
        )

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_logs_a_start_line_and_a_verdict_line_with_counts(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """One line before the subprocess, one after carrying the counts."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        router = _subprocess_router(output_file, self._clean_selftest)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch.object(harness_mod, "logger") as mock_logger:
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result
        info_lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]

        started = [ln for ln in info_lines if "Running metric self-test" in ln]
        assert started, f"no self-test start line logged. Got: {info_lines}"
        assert "300s budget" in started[0]

        finished = [ln for ln in info_lines if "Metric self-test finished" in ln]
        assert finished, f"no self-test verdict line logged. Got: {info_lines}"
        assert "1 metric(s) probed" in finished[0]
        assert "0 blocking" in finished[0]
        assert "0 warning(s)" in finished[0]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_warnings_are_logged_not_only_rendered(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """A warning that only reaches the tool's return value never reaches
        CloudWatch, which is how a degenerate judge metric goes unnoticed."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        def _judge_backed_degenerate(proc):
            proc.communicate.return_value = (
                json.dumps({"status": "ok", "errors": [], "metrics": [{
                    "name": "AnswerRelevancy",
                    "judge_backed": True,
                    "degenerate": {
                        "verdict": "always_pass",
                        "detail": "passed output that is not an answer",
                    },
                }]}).encode(),
                b"",
            )

        router = _subprocess_router(output_file, _judge_backed_degenerate)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch.object(harness_mod, "logger") as mock_logger:
            result = await validate_harness({"project_id": "test-project-id"})

        # Judge-backed degeneracy warns rather than blocks (one sample is not
        # proof), so validation still passes — but it must be greppable.
        assert "is_error" not in result
        warned = [
            c.args[0] % c.args[1:] for c in mock_logger.warning.call_args_list
        ]
        assert any("AnswerRelevancy" in ln for ln in warned), (
            f"degenerate judge metric was not logged as a warning. Got: {warned}"
        )

        info_lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
        finished = [ln for ln in info_lines if "Metric self-test finished" in ln]
        assert finished and "1 warning(s)" in finished[0]

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.harness.write_pricing_file", new_callable=AsyncMock)
    @patch("app.services.mcp_tools.harness.resolve_project_dir")
    async def test_a_timeout_is_marked_in_the_verdict_line(
        self, mock_resolve, mock_write_pricing, tmp_path
    ):
        """"Timed out" and "0 metrics probed" are different facts; the line has
        to say which, or a silent self-test looks like a clean one."""
        project_dir, output_file = _prepare_validated_project(tmp_path)
        mock_resolve.return_value = project_dir
        mock_write_pricing.return_value = project_dir / "src" / "pricing.json"

        def _hangs(proc):
            proc.communicate.side_effect = asyncio.TimeoutError()

        router = _subprocess_router(output_file, _hangs)
        with patch("asyncio.create_subprocess_exec", side_effect=router), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch.object(harness_mod, "logger") as mock_logger:
            result = await validate_harness({"project_id": "test-project-id"})

        assert "is_error" not in result
        info_lines = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
        finished = [ln for ln in info_lines if "Metric self-test finished" in ln]
        assert finished, f"no verdict line on timeout. Got: {info_lines}"
        assert "TIMED OUT" in finished[0]
        assert "0 metric(s) probed" in finished[0]
