# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards that stop the optimization loop from re-running identical prompts
and that catch misconfiguration that would make the loop spin without ever
reaching the target.

Covers:

1. ``save_prompt`` refuses a write when the content (ignoring the
   changelog header) is identical to the newest existing prompt.
2. ``record_optimization_iteration`` refuses to double-record the same
   prompt_version so plateau/window math doesn't see inflated counts.
3. ``dispatch_optimization`` fails fast when ``target_metric_scores`` names
   a metric that isn't configured in EvalConfig.metrics.
4. ``detect_latest_prompt`` ranks ``baseline`` strictly below every
   ``vN``.

The "skip already-graded prompt" and "optimizer exhausted terminal
status" guards used to live here too — those are now exercised inside
the per-iteration job executor (``app.services.jobs.optimization_executor``)
and tested alongside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.mcp_tools.optimization import (
    record_optimization_iteration as _record_tool,
)
from app.services.mcp_tools.dispatch import dispatch_optimization as _dispatch_opt_tool
from app.services.mcp_tools.prompt import save_prompt as _save_prompt_tool

save_prompt = _save_prompt_tool.handler
record_optimization_iteration = _record_tool.handler
dispatch_optimization = _dispatch_opt_tool.handler


def _make_project(tmp_path: Path) -> Path:
    """Build a minimal project tree with prompts/ and config.json."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "prompts").mkdir()
    (project_dir / "config.json").write_text(
        json.dumps({"id": "proj", "name": "t", "description": "", "created_at": "2026-04-30"})
    )
    return project_dir


class TestSavePromptDedup:
    """``save_prompt`` must not persist a no-op write."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_saves_first_prompt(self, mock_resolve, tmp_path):
        """Baseline: saving the first prompt always succeeds."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        result = await save_prompt({
            "project_id": "proj",
            "version": "baseline",
            "content": "You are a helpful assistant.",
        })

        assert "is_error" not in result
        assert (project_dir / "prompts" / "baseline.md").read_text() == "You are a helpful assistant."

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_refuses_identical_content_under_new_version(self, mock_resolve, tmp_path):
        """Saving v2 with the same body as v1 must fail."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        (project_dir / "prompts" / "v1.md").write_text("You are a helpful assistant.")

        result = await save_prompt({
            "project_id": "proj",
            "version": "v2",
            "content": "You are a helpful assistant.",
        })

        assert result.get("is_error") is True
        assert "identical" in result["content"][0]["text"].lower()
        assert not (project_dir / "prompts" / "v2.md").exists()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_refuses_identical_content_despite_changelog_header(self, mock_resolve, tmp_path):
        """A fresh changelog header alone does not make a prompt different."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        (project_dir / "prompts" / "v1.md").write_text(
            "<!-- Version: v1\n     Changes: baseline -->\n\nYou are a helpful assistant."
        )

        result = await save_prompt({
            "project_id": "proj",
            "version": "v2",
            "content": (
                "<!-- Version: v2\n     Changes: attempted tweak but forgot to edit body -->\n\n"
                "You are a helpful assistant."
            ),
        })

        assert result.get("is_error") is True
        assert "identical" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_accepts_meaningful_change(self, mock_resolve, tmp_path):
        """Any non-header change passes."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        (project_dir / "prompts" / "v1.md").write_text("You are a helpful assistant.")

        result = await save_prompt({
            "project_id": "proj",
            "version": "v2",
            "content": "You are a helpful assistant. Always reply in JSON.",
        })

        assert "is_error" not in result
        assert (project_dir / "prompts" / "v2.md").exists()

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_rejects_empty_content(self, mock_resolve, tmp_path):
        """Whitespace-only content is refused before touching disk."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        result = await save_prompt({
            "project_id": "proj",
            "version": "v1",
            "content": "   \n\t\n   ",
        })

        assert result.get("is_error") is True
        assert "empty" in result["content"][0]["text"].lower()


class TestRecordIterationDedup:
    """``record_optimization_iteration`` must not accept duplicate rows."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.optimization.resolve_project_dir")
    async def test_refuses_duplicate_prompt_version(self, mock_resolve, tmp_path):
        from app.services.optimization_state import OptimizationState

        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir
        state = OptimizationState.create(
            project_dir=project_dir,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        state.record_iteration(
            prompt_version="v1",
            run_id="run-1.json",
            metrics_summary={"ExactMatch": 0.6},
            total_cases=10,
            passed=6,
        )

        result = await record_optimization_iteration({
            "project_id": "proj",
            "prompt_version": "v1",
            "run_id": "run-2.json",
        })

        assert result.get("is_error") is True
        assert "already recorded" in result["content"][0]["text"].lower()
        reloaded = OptimizationState.load(project_dir)
        assert reloaded is not None
        assert len(reloaded.iterations) == 1  # duplicate did NOT append


class TestStartOptimizationMetricValidation:
    """``dispatch_optimization`` must reject targets a run can never reach."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.dispatch.resolve_project_dir")
    @patch("app.services.mcp_tools.dispatch.load_harness_config")
    async def test_rejects_raw_unit_target_value(
        self, mock_load, mock_resolve, tmp_path,
    ):
        """A raw-unit target like ``{"Latency": 3000.0}`` (the ms budget
        instead of a 0–1 pass rate) makes convergence impossible — the
        score is capped at 1.0 so ``1.0 >= 3000`` never holds. Reject it
        at dispatch instead of silently running to plateau."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        harness = MagicMock()
        harness.models = ["m"]
        harness.field_mappings = {"input": "q", "expected_output": "a"}
        harness.target_metric_scores = {"Answer Correctness": 1.0, "Latency": 3000.0}
        harness.max_iterations = 5
        harness.plateau_window = 3
        harness.plateau_threshold = None
        harness.judge_model_id = None
        harness.data_file = "d.csv"
        mock_load.return_value = harness

        result = await dispatch_optimization({"project_id": "proj"})

        assert result.get("is_error") is True
        msg = result["content"][0]["text"]
        assert "Latency" in msg
        assert "0–1 pass rate" in msg or "0-1" in msg or "pass rate" in msg

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.dispatch.resolve_project_dir")
    @patch("app.services.mcp_tools.dispatch.load_harness_config")
    async def test_accepts_geval_suffixed_metric_name(
        self, mock_load, mock_resolve, tmp_path,
    ):
        """DeepEval records GEval metrics as ``Correctness [GEval]`` — the
        target may just say ``Correctness``. The fuzzy match should allow it."""
        from app.models.schemas import HarnessConfig

        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        mock_load.return_value = HarnessConfig(
            models=["m"],
            metrics=[{"name": "Correctness [GEval]", "type": "geval"}],
            data_file="d.csv",
            judge_model_id=None,
            field_mappings={"input": "q", "expected_output": "a"},
            target_metric_scores={"Correctness": 0.9},
            max_iterations=5,
            plateau_window=3,
        )

        # Replace asyncio.create_task so we don't actually run the loop,
        # but still close the unused coroutine to suppress a RuntimeWarning.
        def _fake_create_task(coro, name=None):
            coro.close()
            return MagicMock()

        with patch("app.services.mcp_tools.dispatch.asyncio.create_task", side_effect=_fake_create_task), \
             patch("app.services.mcp_tools.dispatch.attach_error_sink"), \
             patch(
                 "app.services.mcp_tools.dispatch._preflight_models_invocable",
                 new_callable=AsyncMock, return_value=[],
             ):
            result = await dispatch_optimization({"project_id": "proj"})

        assert "is_error" not in result


class TestDetectLatestPromptOrdering:
    """``detect_latest_prompt`` must rank baseline below every vN."""

    def test_baseline_below_v1(self, tmp_path):
        from app.services.optimization_orchestrator import detect_latest_prompt

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "baseline.md").write_text("x")
        (prompts / "v1.md").write_text("y")

        assert detect_latest_prompt(tmp_path) == "v1"

    def test_highest_v(self, tmp_path):
        from app.services.optimization_orchestrator import detect_latest_prompt

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "v1.md").write_text("x")
        (prompts / "v2.md").write_text("y")
        (prompts / "v10.md").write_text("z")

        assert detect_latest_prompt(tmp_path) == "v10"

    def test_empty_dir(self, tmp_path):
        from app.services.optimization_orchestrator import detect_latest_prompt

        (tmp_path / "prompts").mkdir()
        assert detect_latest_prompt(tmp_path) == "baseline"

    def test_ignores_unrecognised_filenames(self, tmp_path):
        from app.services.optimization_orchestrator import detect_latest_prompt

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "baseline.md").write_text("x")
        (prompts / "draft.md").write_text("y")
        (prompts / "scratch.md").write_text("z")

        assert detect_latest_prompt(tmp_path) == "baseline"


class TestNextPromptVersion:
    """``next_prompt_version`` derives the next name from the current
    prompt name, not the loop counter — the counter-based formula skipped
    v2 (baseline → v1 → v3 → …)."""

    def test_baseline_to_v1(self):
        from app.services.optimization_orchestrator import next_prompt_version
        assert next_prompt_version("baseline") == "v1"

    def test_v1_to_v2_no_skip(self):
        """The regression guard: v1 → v2, NOT v3."""
        from app.services.optimization_orchestrator import next_prompt_version
        assert next_prompt_version("v1") == "v2"

    def test_increments_correctly(self):
        from app.services.optimization_orchestrator import next_prompt_version
        assert next_prompt_version("v2") == "v3"
        assert next_prompt_version("v9") == "v10"
        assert next_prompt_version("v10") == "v11"

    def test_full_sequence_no_gap(self):
        """Simulate the loop's version walk — must be contiguous."""
        from app.services.optimization_orchestrator import next_prompt_version
        seq = ["baseline"]
        cur = "baseline"
        for _ in range(4):
            cur = next_prompt_version(cur)
            seq.append(cur)
        assert seq == ["baseline", "v1", "v2", "v3", "v4"]

    def test_non_v_name_falls_back_to_v1(self):
        from app.services.optimization_orchestrator import next_prompt_version
        assert next_prompt_version("draft") == "v1"


class TestSavePromptAtomicity:
    """``save_prompt`` must use an atomic write so a crash mid-write
    can't leave partial content in the target file."""

    @pytest.mark.asyncio
    @patch("app.services.mcp_tools.prompt.resolve_project_dir")
    async def test_atomic_swap_on_overwrite(self, mock_resolve, tmp_path):
        """When overwriting an existing prompt, the file is swapped
        atomically — no ``.tmp`` leftovers, no partial content."""
        project_dir = _make_project(tmp_path)
        mock_resolve.return_value = project_dir

        # Seed a file; save a meaningfully-different one on the same name.
        # (Different stems would trip the identical-content guard.)
        (project_dir / "prompts" / "v1.md").write_text("old content")
        result = await save_prompt({
            "project_id": "proj",
            "version": "v1",
            "content": "brand new content with real changes",
        })

        assert "is_error" not in result
        assert (project_dir / "prompts" / "v1.md").read_text() == "brand new content with real changes"
        # No temp files left behind.
        leftovers = list((project_dir / "prompts").glob(".v1.md.*.tmp"))
        assert leftovers == [], f"Atomic write leaked temp files: {leftovers}"


class TestPerModelPlateau:
    """Plateau detection must use per-model rates when available —
    aggregate-only misses real signal when one model climbs while another
    regresses."""

    def test_no_plateau_when_one_model_improves_and_other_regresses(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.99},
            max_iterations=10,
            plateau_window=3,
        )
        # Aggregate stays at 0.60 across all three iterations, but per-model
        # the series moves: A climbs 0.30→0.35, B regresses 0.30→0.25.
        # Aggregate would report plateau; per-model should not.
        for i, (rate_a, rate_b) in enumerate([(0.30, 0.30), (0.32, 0.28), (0.35, 0.25)]):
            state.record_iteration(
                prompt_version=f"v{i+1}",
                run_id=f"run-{i+1}.json",
                metrics_summary={"ExactMatch": (rate_a + rate_b) / 2},
                total_cases=20,
                passed=int((rate_a + rate_b) * 10),
                per_model_pass_rates={"model-a": rate_a, "model-b": rate_b},
            )

        convergence, _ = state.check_convergence(); assert convergence is None  # Not plateaued

    def test_plateau_when_all_models_flat(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.99},
            max_iterations=10,
            plateau_window=3,
        )
        for i in range(3):
            state.record_iteration(
                prompt_version=f"v{i+1}",
                run_id=f"run-{i+1}.json",
                metrics_summary={"ExactMatch": 0.60},
                total_cases=20,
                passed=12,
                per_model_pass_rates={"model-a": 0.605, "model-b": 0.595},
            )

        convergence, _ = state.check_convergence(); assert convergence == "plateau"

    def test_plateau_withheld_for_legacy_rows(self, tmp_path):
        """INVERSION of the old fallback pin. A window with no per-model
        data is not plateau evidence — a run-terminating verdict demands the
        same evidence quality as convergence (aggregate flatness can mask one
        model climbing while another regresses). This test used to assert the
        aggregate fallback FIRED; the fallback is deleted, so the verdict is
        withheld and the run stays bounded by max_iterations."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.99},
            max_iterations=10,
            plateau_window=3,
        )
        for i in range(3):
            state.record_iteration(
                prompt_version=f"v{i+1}",
                run_id=f"run-{i+1}.json",
                metrics_summary={"ExactMatch": 0.60},
                total_cases=20,
                passed=12,
                # Explicitly no per_model_pass_rates.
            )

        convergence, _ = state.check_convergence(); assert convergence is None
        # And the run is NOT terminated: status stays running for the loop
        # to continue toward the max_iterations bound.
        assert state.status == "running"


class TestConfigurablePlateauThreshold:
    """``plateau_threshold`` must be overridable per-project."""

    def test_tight_threshold_detects_small_spread(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.99},
            max_iterations=10,
            plateau_window=3,
            plateau_threshold=0.005,  # 0.5% — anything moving > 0.5 pp is not plateaued
        )
        # Spread of 1.0 pp across the window: default (2%) would say plateau,
        # tight (0.5%) should NOT.
        for i, rate in enumerate([0.60, 0.605, 0.61]):
            state.record_iteration(
                prompt_version=f"v{i+1}",
                run_id=f"run-{i+1}.json",
                metrics_summary={"ExactMatch": rate},
                total_cases=100,
                passed=int(rate * 100),
            )

        convergence, _ = state.check_convergence(); assert convergence is None

    def test_threshold_round_trips_through_save_load(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.9},
            max_iterations=5,
            plateau_window=3,
            plateau_threshold=0.01,
        )
        reloaded = OptimizationState.load(tmp_path)
        assert reloaded is not None
        assert reloaded.plateau_threshold == 0.01


class TestResumeWorkloadDriftDetection:
    """Resume must surface when ``src/config.py`` has drifted from the
    workload captured at start time."""

    def test_fingerprint_captures_ordered_canonical_form(self):
        from app.services.optimization_state import OptimizationState

        fp_a = OptimizationState.workload_fingerprint(
            model_ids=["model-b", "model-a"],
            judge_model_id="judge-x",
            data_file="train.csv",
        )
        fp_b = OptimizationState.workload_fingerprint(
            model_ids=["model-a", "model-b"],
            judge_model_id="judge-x",
            data_file="train.csv",
        )
        assert fp_a == fp_b

    def test_detects_model_swap(self):
        from app.services.optimization_state import OptimizationState

        before = OptimizationState.workload_fingerprint(
            model_ids=["model-a"], judge_model_id="j", data_file="d.csv",
        )
        after = OptimizationState.workload_fingerprint(
            model_ids=["model-b"], judge_model_id="j", data_file="d.csv",
        )
        assert before != after
        assert before["models"] != after["models"]

    def test_pending_drift_attached_to_next_iteration_then_cleared(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.9},
            max_iterations=5,
            plateau_window=3,
        )
        state.pending_resume_drift = {"models": {"before": ["a"], "after": ["b"]}}
        state.save()

        state.record_iteration(
            prompt_version="v1",
            run_id="r.json",
            metrics_summary={"M": 0.5},
            total_cases=10,
            passed=5,
        )

        assert state.iterations[-1]["resume_drift"] == {
            "models": {"before": ["a"], "after": ["b"]},
        }
        # Second iteration recorded without new drift must not inherit it.
        state.record_iteration(
            prompt_version="v2",
            run_id="r2.json",
            metrics_summary={"M": 0.6},
            total_cases=10,
            passed=6,
        )
        assert "resume_drift" not in state.iterations[-1]


class TestModelIdIsSourceOfTruth:
    """``model_id`` (full Bedrock ID) is the join key. ``test_run_id`` is
    display-only. The orchestrator must prefer the full ID when it exists
    and fall back gracefully for legacy outputs."""

    def test_orchestrator_keys_by_full_model_id(self):
        """Per-model pass rates must be keyed by the full Bedrock ID
        when ``model_id`` is present on the run."""
        # Simulate the keying logic from the orchestrator loop.
        run_with_full_id = {
            "model_id": "us.anthropic.claude-sonnet-4-6-v1:0",
            "test_run_id": "claude-sonnet-4-6",
            "hyperparameters": {"model": "us.anthropic.claude-sonnet-4-6-v1:0"},
            "results": [{"success": True}, {"success": False}],
        }
        key = (
            run_with_full_id.get("model_id")
            or run_with_full_id.get("hyperparameters", {}).get("model")
            or run_with_full_id.get("test_run_id")
        )
        assert key == "us.anthropic.claude-sonnet-4-6-v1:0"

    def test_falls_back_to_hyperparameters_model_for_legacy_runs(self):
        """Older outputs predating the model_id field still resolve."""
        legacy_run = {
            "test_run_id": "claude-sonnet-4-6",
            "hyperparameters": {"model": "us.anthropic.claude-sonnet-4-6-v1:0"},
        }
        key = (
            legacy_run.get("model_id")
            or legacy_run.get("hyperparameters", {}).get("model")
            or legacy_run.get("test_run_id")
        )
        assert key == "us.anthropic.claude-sonnet-4-6-v1:0"

    def test_harness_template_runner_writes_model_id(self):
        """Template _serialize_result returns a dict keyed by full model_id."""
        from unittest.mock import MagicMock

        from tests.harness_template_stubs import load_template_runner

        module = load_template_runner("harness_template_runner_modelid")

        evaluator = module.EvalRunner.__new__(module.EvalRunner)
        evaluator.source = "manual"
        eval_result = MagicMock()
        eval_result.test_results = []

        serialized = evaluator._serialize_result(
            model_id="us.anthropic.claude-sonnet-4-6-v1:0",
            eval_result=eval_result,
            hyperparams={},
            total_input_cases=len(eval_result.test_results),
        )

        assert serialized["model_id"] == "us.anthropic.claude-sonnet-4-6-v1:0"
        # Display short name stays intact too.
        assert serialized["test_run_id"] == "claude-sonnet-4-6"


class TestAnyModelConvergence:
    """Target-reached fires as soon as **any** model clears every target.
    The goal is to pick the best model for the job — keep going past the
    strongest model's win is pointless once it's already hit the bar."""

    def test_converges_when_one_model_hits_all_targets_others_weak(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"Correctness": 0.85, "Fluency": 0.80},
            max_iterations=10,
            plateau_window=3,
        )

        # Strong model clears both targets; weak model fails both.
        state.record_iteration(
            prompt_version="v1",
            run_id="r1.json",
            metrics_summary={"Correctness": 0.65, "Fluency": 0.60},  # aggregate is below
            total_cases=40, passed=24,
            per_model_pass_rates={"strong": 0.90, "weak": 0.40},
            per_model_metrics={
                "strong": {"Correctness": 0.92, "Fluency": 0.88},
                "weak":   {"Correctness": 0.40, "Fluency": 0.30},
            },
        )

        convergence, _ = state.check_convergence(); assert convergence == "target_reached"
        assert state.status == "converged"
        assert state.winning_model_id == "strong"

    def test_does_not_converge_when_no_model_clears_every_target(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"Correctness": 0.85, "Fluency": 0.80},
            max_iterations=10,
            plateau_window=5,
        )

        # Model A nails Correctness but fails Fluency; Model B the reverse.
        # Neither clears BOTH, so no convergence.
        state.record_iteration(
            prompt_version="v1",
            run_id="r1.json",
            metrics_summary={"Correctness": 0.87, "Fluency": 0.82},
            total_cases=40, passed=34,
            per_model_pass_rates={"a": 0.70, "b": 0.70},
            per_model_metrics={
                "a": {"Correctness": 0.95, "Fluency": 0.50},
                "b": {"Correctness": 0.60, "Fluency": 0.90},
            },
        )

        convergence, _ = state.check_convergence(); assert convergence is None
        assert state.status == "running"
        assert state.winning_model_id is None

    def test_tolerates_geval_metric_suffix_on_per_model_rates(self, tmp_path):
        """DeepEval stores GEval metrics as ``Correctness [GEval]``; the
        target says ``Correctness``. Fuzzy match should apply per-model too."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"Correctness": 0.85},
            max_iterations=10,
            plateau_window=3,
        )

        state.record_iteration(
            prompt_version="v1",
            run_id="r1.json",
            metrics_summary={"Correctness [GEval]": 0.90},
            total_cases=10, passed=9,
            per_model_pass_rates={"m": 0.90},
            per_model_metrics={"m": {"Correctness [GEval]": 0.90}},
        )

        convergence, _ = state.check_convergence(); assert convergence == "target_reached"
        assert state.winning_model_id == "m"

    def test_winning_model_id_persists_through_save_load(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.5},
            max_iterations=5,
            plateau_window=3,
        )
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"M": 0.9},
            total_cases=10, passed=9,
            per_model_metrics={"winner": {"M": 0.9}},
        )
        state.check_convergence()  # returns tuple now

        reloaded = OptimizationState.load(tmp_path)
        assert reloaded is not None
        assert reloaded.winning_model_id == "winner"


class TestBestIterationMatchesConvergence:
    """best_iteration() must be consistent with the any-model convergence
    rule — picking by aggregate would disagree with which iteration
    actually produced the winner."""

    def test_returns_winning_model_best_iteration_after_convergence(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.85},
            max_iterations=10,
            plateau_window=5,
        )
        # Iter 1: winner scores 0.86 (would clear), other at 0.30. Aggregate 0.58.
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"M": 0.58},
            total_cases=40, passed=23,
            per_model_pass_rates={"winner": 0.86, "other": 0.30},
            per_model_metrics={"winner": {"M": 0.86}, "other": {"M": 0.30}},
        )
        convergence, _ = state.check_convergence(); assert convergence == "target_reached"
        # winning_model_id captured; best_iteration should be this one.
        best = state.best_iteration()
        assert best is not None
        assert best["iteration"] == 1
        assert state.winning_model_id == "winner"

    def test_prefers_winner_iteration_even_when_aggregate_peaks_elsewhere(self, tmp_path):
        """Iter 2 has higher aggregate (both models mid-range) but iter 1 is
        where the winner scored highest. best_iteration must return iter 1."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.85},
            max_iterations=10,
            plateau_window=5,
        )
        # Iter 1: winner nails it (0.90), other weak (0.20). Aggregate 0.55.
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"M": 0.55},
            total_cases=40, passed=22,
            per_model_pass_rates={"winner": 0.90, "other": 0.20},
            per_model_metrics={"winner": {"M": 0.90}, "other": {"M": 0.20}},
        )
        state.check_convergence()  # fires target_reached, sets winning_model_id="winner"

        # Iter 2: winner drops a little (0.82), other improves a lot (0.70).
        # Aggregate jumps to 0.76, but iter 1 is still the winner's peak.
        # (Loop would normally stop after iter 1, but we force-append iter 2
        # to test best_iteration's selection logic directly.)
        state.status = "running"
        state.record_iteration(
            prompt_version="v2", run_id="r2.json",
            metrics_summary={"M": 0.76},
            total_cases=40, passed=30,
            per_model_pass_rates={"winner": 0.82, "other": 0.70},
            per_model_metrics={"winner": {"M": 0.82}, "other": {"M": 0.70}},
        )

        best = state.best_iteration()
        assert best["iteration"] == 1, (
            "best_iteration should track the winning model's peak, not the "
            f"highest aggregate; got iteration {best['iteration']}"
        )

    def test_falls_back_to_best_per_model_rate_without_winner(self, tmp_path):
        """Before convergence, return the iteration with the strongest
        single-model score, not the highest aggregate."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.99},  # unreachable — no convergence
            max_iterations=10,
            plateau_window=5,
        )
        # Iter 1: best single-model score 0.85, aggregate 0.60.
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"M": 0.60},
            total_cases=40, passed=24,
            per_model_pass_rates={"a": 0.85, "b": 0.35},
        )
        # Iter 2: best single-model 0.70, aggregate 0.70.
        state.record_iteration(
            prompt_version="v2", run_id="r2.json",
            metrics_summary={"M": 0.70},
            total_cases=40, passed=28,
            per_model_pass_rates={"a": 0.70, "b": 0.70},
        )

        best = state.best_iteration()
        assert best["iteration"] == 1, (
            "Iter 1 has a higher single-model peak (0.85 vs 0.70) even though "
            "its aggregate is lower — best_iteration must track per-model peaks."
        )

    def test_rows_without_per_model_data_rank_last_rather_than_by_aggregate(
        self, tmp_path,
    ):
        """The aggregate substitution is gone, and this class is why.

        This whole class exists because ranking by aggregate disagrees with the
        any-model convergence rule. The old fallback did exactly that for rows
        lacking per-model data, so the module both forbade and performed the same
        substitution depending on which rows it was handed.
        """
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"M": 0.99},
            max_iterations=10,
            plateau_window=5,
        )
        # A rankable row with a MODEST per-model score...
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"M": 0.60},
            total_cases=10, passed=6,
            per_model_pass_rates={"m": 0.60},
        )
        # ...and an unrankable row with a HIGHER aggregate. Under the old
        # fallback this won on 0.80 vs 0.60; it must now lose to the row that
        # was actually measured per model.
        state.record_iteration(
            prompt_version="v2", run_id="r2.json",
            metrics_summary={"M": 0.80},
            total_cases=10, passed=8,
        )

        best = state.best_iteration()
        assert best is not None
        assert best["iteration"] == 1, (
            "an unmeasured row out-ranked a measured one on a different quantity"
        )


class TestConvergenceWithholdsOnPostDriftIteration:
    """``check_convergence`` must not report target_reached on the single
    post-drift iteration — the workload changed, so that score is against
    a different baseline than the earlier history."""

    def test_target_reached_withheld_on_post_drift_iteration(self, tmp_path):
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
        )

        # Pretend the user resumed with different models.
        state.pending_resume_drift = {
            "models": {"before": ["a"], "after": ["b"]},
        }
        state.save()

        # First iteration after the resume hits the target. Because it's
        # flagged as resume_drift, convergence should NOT be declared.
        state.record_iteration(
            prompt_version="v1",
            run_id="r1.json",
            metrics_summary={"ExactMatch": 0.95},
            total_cases=20,
            passed=19,
        )
        assert state.iterations[-1].get("resume_drift") is not None
        convergence, _ = state.check_convergence(); assert convergence is None
        assert state.status == "running"

    def test_target_reached_on_second_post_drift_iteration(self, tmp_path):
        """One more clean iteration after the drift marker is enough."""
        from app.services.optimization_state import OptimizationState

        state = OptimizationState.create(
            project_dir=tmp_path,
            target_metric_scores={"ExactMatch": 0.85},
            max_iterations=10,
            plateau_window=3,
        )

        state.pending_resume_drift = {"models": {"before": ["a"], "after": ["b"]}}
        state.save()
        # Iteration 1 — post-drift, hits target but withheld.
        state.record_iteration(
            prompt_version="v1", run_id="r1.json",
            metrics_summary={"ExactMatch": 0.95},
            total_cases=20, passed=19,
            per_model_pass_rates={"m1": 0.95},
            per_model_metrics={"m1": {"ExactMatch": 0.95}},
        )
        convergence, _ = state.check_convergence(); assert convergence is None

        # Iteration 2 — no drift marker, still hits target → converge.
        state.record_iteration(
            prompt_version="v2", run_id="r2.json",
            metrics_summary={"ExactMatch": 0.90},
            total_cases=20, passed=18,
            per_model_pass_rates={"m1": 0.9},
            per_model_metrics={"m1": {"ExactMatch": 0.90}},
        )
        convergence, _ = state.check_convergence(); assert convergence == "target_reached"


