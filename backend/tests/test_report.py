# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the narrative-led report builder.

The HTML/iframe renderer is gone; reports are now React components fed
by ``{facts, narrative, metadata}`` JSON. These tests pin the deterministic
facts builders and the report-JSON sidecar I/O — the two pure-Python halves
the API endpoint serves.

The frontend renders the React components; their tests live in
``frontend/src/components/reports/reports.test.tsx``. The narrator agent is
not run here: ``test_report_generation.py`` covers narration with the agent
stubbed out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.model_id_short import short_model_id
from app.services.report import (
    REPORT_SCHEMA_VERSION,
    build_exec_facts,
    build_run_facts,
    read_report_json,
    report_json_path,
    write_report_json,
    _facts_to_jsonable,
    _resolve_exec_winner,
)


# --------------------------------------------------------------------------
# Fixtures / builders for synthetic on-disk eval runs
# --------------------------------------------------------------------------

SONNET = "us.anthropic.claude-sonnet-4-6"
HAIKU = "us.anthropic.claude-haiku-4-5"
NOVA = "us.amazon.nova-2-lite-v1:0"


def _case(idx, *, success, metric_specs, latency_ms=100.0, gen_cost=0.001,
          category=None, in_tokens=100, out_tokens=50,
          cache_read=0, cache_write=0):
    """Build one per-case result dict matching the harness on-disk shape.

    metric_specs: list of (name, score, threshold, success) tuples.
    """
    meta = {
        "id": idx,
        "latency_ms": latency_ms,
        "generation_cost": gen_cost,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
    }
    if category is not None:
        meta["category"] = category
    return {
        "name": f"case-{idx}",
        "success": success,
        "input": f"q{idx}",
        "actual_output": f"a{idx}",
        "expected_output": f"e{idx}",
        "context": None,
        "retrieval_context": None,
        "additional_metadata": meta,
        "metrics_data": [
            {"name": n, "score": s, "threshold": t, "success": ok, "reason": "r", "error": None}
            for (n, s, t, ok) in metric_specs
        ],
    }


def _model_run(model_id, *, cases, prompt_version="v2", source="optimization",
               timestamp="2026-06-05T10:00:00Z", gen_cost_usd=0.02,
               judge_cost_usd=0.01, temperature=0.0, max_tokens=2048,
               data_quality=None):
    return {
        "source": source,
        "model_id": model_id,
        "test_run_id": short_model_id(model_id),
        "timestamp": timestamp,
        "hyperparameters": {
            "model": model_id,
            "endpoint": "runtime",
            "region": "us-east-1",
            "prompt_version": prompt_version,
            "temperature": temperature, "max_tokens": max_tokens,
        },
        "results": cases,
        "generation_cost_usd": gen_cost_usd,
        "judge_cost_usd": judge_cost_usd,
        "data_quality": data_quality or {
            "reliability": "ok", "errored_count": 0, "dropped_count": 0,
            "null_count": 0, "unrecovered_count": 0, "total_cases": len(cases),
        },
    }


def _write_run(project_dir: Path, runs: list, filename: str) -> str:
    (project_dir / "output" / filename).write_text(json.dumps(runs))
    return filename


def _scaffold_project(tmp_path: Path, *, name="My Eval Project",
                      description="Testing the report.",
                      prompts=None) -> Path:
    (tmp_path / "output").mkdir()
    (tmp_path / "prompts").mkdir()
    for stem, body in (prompts or {"baseline": "baseline body",
                                   "v2": "You are an optimized assistant.\n{input}"}).items():
        (tmp_path / "prompts" / f"{stem}.md").write_text(body)
    (tmp_path / "config.json").write_text(json.dumps({
        "id": "proj-123",
        "name": name,
        "description": description,
        "created_at": "2026-06-01T00:00:00Z",
        "agent_model": "opus",
    }))
    return tmp_path


def _write_state(project_dir: Path, *,
                 target_metric_scores=None, iterations=None):
    sd = project_dir / "output" / "optimization"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "state.json").write_text(json.dumps({
        "target_metric_scores": target_metric_scores or {"Correctness": 0.8, "Relevance": 0.8},
        "max_iterations": 10, "plateau_window": 3, "status": "converged",
        "iterations": iterations or [],
    }))


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Two-model run, two metrics, no optimization state — base case."""
    pd = _scaffold_project(tmp_path)
    sonnet = _model_run(SONNET, cases=[
        _case(0, success=True, metric_specs=[("Correctness", 0.9, 0.5, True), ("Relevance", 0.95, 0.5, True)]),
        _case(1, success=True, metric_specs=[("Correctness", 0.85, 0.5, True), ("Relevance", 0.9, 0.5, True)]),
    ], gen_cost_usd=0.02, judge_cost_usd=0.01)
    haiku = _model_run(HAIKU, cases=[
        _case(0, success=True, metric_specs=[("Correctness", 0.8, 0.5, True), ("Relevance", 0.7, 0.5, True)]),
        _case(1, success=False, metric_specs=[("Correctness", 0.2, 0.5, False), ("Relevance", 0.9, 0.5, True)]),
    ], gen_cost_usd=0.008, judge_cost_usd=0.004)
    _write_run(pd, [sonnet, haiku], "eval_results_v2_20260605.json")
    return pd


# --------------------------------------------------------------------------
# build_run_facts
# --------------------------------------------------------------------------

class TestBuildRunFacts:
    def test_metric_columns_are_dynamic(self, project_dir):
        facts = build_run_facts(project_dir)
        assert sorted(facts.metric_names) == ["Correctness", "Relevance"]

    def test_project_metadata_loaded(self, project_dir):
        facts = build_run_facts(project_dir)
        assert facts.project_name == "My Eval Project"
        # The project description is on the facts so the frontend can
        # render it as the header subtitle.
        assert facts.project_description == "Testing the report."
        assert facts.case_count == 2

    def test_run_timestamp_carried_through(self, project_dir):
        # The header date in the UI uses the eval RUN timestamp, never
        # report-generation time. The facts must expose it.
        facts = build_run_facts(project_dir)
        assert facts.run_timestamp == "2026-06-05T10:00:00Z"

    def test_per_model_pass_rates(self, project_dir):
        facts = build_run_facts(project_dir)
        by_id = {m.model_id: m for m in facts.models}
        assert by_id[SONNET].pass_rate == pytest.approx(100.0)
        assert by_id[HAIKU].pass_rate == pytest.approx(50.0)

    def test_per_model_cost_per_1k_gen_only(self, project_dir):
        facts = build_run_facts(project_dir)
        sonnet = next(m for m in facts.models if m.model_id == SONNET)
        # gen 0.02 / 2 cases * 1000 = $10/1k.
        assert sonnet.cost_per_1k_gen == pytest.approx(10.0)
        assert sonnet.total_cost_usd == pytest.approx(0.03)

    def test_run_integrity_clean_run(self, project_dir):
        facts = build_run_facts(project_dir)
        ri = facts.run_integrity
        assert ri.errored == 0
        assert ri.dropped == 0
        assert ri.unrecovered == 0
        assert ri.reliability == "ok"
        assert ri.effective_total == ri.case_count

    def test_run_integrity_surfaces_errored_count(self, tmp_path):
        pd = _scaffold_project(tmp_path)
        run = _model_run(SONNET, cases=[
            _case(i, success=True, metric_specs=[("Acc", 0.9, 0.5, True)])
            for i in range(8)
        ], data_quality={
            "reliability": "degraded", "errored_count": 2,
            "dropped_count": 0, "null_count": 0, "unrecovered_count": 1,
            "total_cases": 10,
        })
        _write_run(pd, [run], "eval_results_v2_x.json")
        facts = build_run_facts(pd)
        assert facts.run_integrity.errored == 2
        assert facts.run_integrity.unrecovered == 1
        assert facts.run_integrity.reliability == "degraded"
        assert facts.run_integrity.effective_total == 7

    def test_token_aggregation(self, project_dir):
        """Uncached run. NOTE: this fixture cannot see the cached-prompt
        defect in either direction — see ``TestCachedTokenAccounting``."""
        facts = build_run_facts(project_dir)
        sonnet = next(m for m in facts.models if m.model_id == SONNET)
        assert sonnet.avg_input_tokens == pytest.approx(100.0)
        assert sonnet.avg_output_tokens == pytest.approx(50.0)
        assert sonnet.cache_read_ratio == 0.0

    def test_failed_cases_one_per_failure(self, project_dir):
        facts = build_run_facts(project_dir)
        assert len(facts.failed_cases) == 1
        fc = facts.failed_cases[0]
        assert fc.metric == "Correctness"
        assert fc.is_latency is False
        assert fc.model_id == HAIKU

    def test_recommendation_tag_no_targets(self, project_dir):
        facts = build_run_facts(project_dir)
        recs = {r.model_id: r.tag for r in facts.recommendations}
        assert recs[SONNET] == "Top pass rate"

    def test_recommended_model_id_is_set_without_targets(self, project_dir):
        # SSOT field: KPI tiles, header, and narrative all key on this id.
        # Should match the highest-pass-rate model when no targets are set.
        facts = build_run_facts(project_dir)
        assert facts.recommended_model_id == SONNET

    def test_hyperparameters_carried_through(self, project_dir):
        facts = build_run_facts(project_dir)
        assert facts.hyperparameters.get("temperature") == 0.0
        assert facts.hyperparameters.get("max_tokens") == 2048

    def test_appendix_has_prompt_content(self, project_dir):
        facts = build_run_facts(project_dir)
        assert "optimized assistant" in facts.prompt_content
        assert facts.prompt_label == "v2"

    def test_category_count_reflects_distinct_categories(self, tmp_path):
        # The facts expose category_count for the category chip.
        pd = _scaffold_project(tmp_path)
        run = _model_run(SONNET, cases=[
            _case(0, success=True, metric_specs=[("Acc", 0.9, 0.5, True)], category="alpha"),
            _case(1, success=True, metric_specs=[("Acc", 0.9, 0.5, True)], category="beta"),
            _case(2, success=True, metric_specs=[("Acc", 0.9, 0.5, True)], category="alpha"),
        ])
        _write_run(pd, [run], "eval_results_v2_x.json")
        facts = build_run_facts(pd)
        assert facts.category_count == 2

    def test_category_count_zero_when_uncategorized(self, project_dir):
        facts = build_run_facts(project_dir)
        assert facts.category_count == 0

    def test_no_runs_raises(self, tmp_path):
        pd = _scaffold_project(tmp_path, prompts={})
        for p in (pd / "prompts").glob("*"):
            p.unlink()
        with pytest.raises((FileNotFoundError, ValueError)):
            build_run_facts(pd)


# --------------------------------------------------------------------------
# build_run_facts — with optimization state
# --------------------------------------------------------------------------

class TestBuildRunFactsWithOptState:
    def test_recommended_clears_targets(self, project_dir):
        _write_state(project_dir, target_metric_scores={"Correctness": 0.8, "Relevance": 0.8})
        facts = build_run_facts(project_dir)
        recs = {r.model_id: r.tag for r in facts.recommendations}
        assert recs[SONNET] == "Recommended"
        assert recs[HAIKU] in ("Below targets", "Below latency target")
        # SSOT field agrees with the verdict tag.
        assert facts.recommended_model_id == SONNET

    def test_recommended_model_id_on_plateau_no_clearer(self, tmp_path):
        # Plateau case: targets set but NO model clears them. Verdict tags
        # all read "Below targets", so the frontend can't find a recommended
        # model via tag lookup. The SSOT field still picks the best-of-the-
        # bunch (highest pass rate) so KPI tiles + narrative agree.
        pd = _scaffold_project(tmp_path)
        # Both models miss a 1.0 target on Correctness — Sonnet at 50%, Haiku at 0%.
        sonnet = _model_run(SONNET, cases=[
            _case(0, success=True, metric_specs=[("Correctness", 0.9, 0.5, True)]),
            _case(1, success=False, metric_specs=[("Correctness", 0.4, 0.5, False)]),
        ])
        haiku = _model_run(HAIKU, cases=[
            _case(0, success=False, metric_specs=[("Correctness", 0.3, 0.5, False)]),
            _case(1, success=False, metric_specs=[("Correctness", 0.2, 0.5, False)]),
        ])
        _write_run(pd, [sonnet, haiku], "eval_results_v2_20260605.json")
        _write_state(pd, target_metric_scores={"Correctness": 1.0})
        facts = build_run_facts(pd)
        # Neither carries the "Recommended" tag — both miss the target.
        recs = {r.model_id: r.tag for r in facts.recommendations}
        assert "Recommended" not in recs.values()
        # But the SSOT field still picks Sonnet (higher pass rate). Without
        # this, the frontend's tag-based lookup would fall through to "first
        # model" — which is the cheapest, not the best.
        assert facts.recommended_model_id == SONNET

    def test_iteration_index_when_run_id_matches(self, tmp_path):
        pd = _scaffold_project(tmp_path)
        run = _model_run(SONNET, cases=[
            _case(0, success=True, metric_specs=[("Acc", 0.9, 0.5, True)]),
        ])
        _write_run(pd, [run], "eval_results_v2_x.json")
        _write_state(pd, iterations=[
            {"iteration": 5, "prompt_version": "v2", "run_id": "eval_results_v2_x.json",
             "metrics_summary": {}, "total_cases": 1, "passed": 1, "pass_rate": 1.0,
             "timestamp": "2026-06-05T10:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0}},
        ])
        facts = build_run_facts(pd, "eval_results_v2_x.json")
        assert facts.prompt_iteration == 5

    def test_prior_step_from_journey(self, tmp_path):
        pd = _scaffold_project(tmp_path, prompts={"v1": "v1 body", "v2": "v2 body"})
        v1_run = _model_run(SONNET, prompt_version="v1", timestamp="2026-06-04T10:00:00Z",
                            cases=[
            _case(0, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
            _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
        ])
        v2_run = _model_run(SONNET, prompt_version="v2", timestamp="2026-06-05T10:00:00Z",
                            cases=[
            _case(0, success=True, metric_specs=[("Acc", 0.9, 0.5, True)]),
            _case(1, success=True, metric_specs=[("Acc", 0.9, 0.5, True)]),
        ])
        _write_run(pd, [v1_run], "eval_results_v1_a.json")
        _write_run(pd, [v2_run], "eval_results_v2_a.json")
        facts = build_run_facts(pd, "eval_results_v2_a.json")
        assert facts.prior_step is not None
        assert facts.prior_step["version"] == "v1"
        assert facts.prior_step["delta"] == pytest.approx(100.0)

    def test_per_model_deltas_vs_prior_version(self, tmp_path):
        # Two versions, two models. Compute per-model and per-metric Δ
        # against the prior version's latest run for each model. None when
        # rendering the baseline (no prior).
        pd = _scaffold_project(tmp_path, prompts={"v1": "v1 body", "v2": "v2 body"})
        # v1: sonnet 50% (1/2 pass on Acc), haiku 0% (0/2 pass).
        v1 = [
            _model_run(SONNET, prompt_version="v1", timestamp="2026-06-04T10:00:00Z", cases=[
                _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
                _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
            ]),
            _model_run(HAIKU, prompt_version="v1", timestamp="2026-06-04T10:00:00Z", cases=[
                _case(0, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
                _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
            ]),
        ]
        # v2: sonnet 100% (+50), haiku 50% (+50).
        v2 = [
            _model_run(SONNET, prompt_version="v2", timestamp="2026-06-05T10:00:00Z", cases=[
                _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
                _case(1, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            ]),
            _model_run(HAIKU, prompt_version="v2", timestamp="2026-06-05T10:00:00Z", cases=[
                _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
                _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
            ]),
        ]
        _write_run(pd, v1, "eval_results_v1_a.json")
        _write_run(pd, v2, "eval_results_v2_a.json")

        # Render v2 → both models get a delta vs v1.
        facts = build_run_facts(pd, "eval_results_v2_a.json")
        by_id = {m.model_id: m for m in facts.models}
        assert by_id[SONNET].pass_rate_delta == pytest.approx(50.0)
        assert by_id[SONNET].metric_rate_deltas["Acc"] == pytest.approx(50.0)
        assert by_id[HAIKU].pass_rate_delta == pytest.approx(50.0)
        assert by_id[HAIKU].metric_rate_deltas["Acc"] == pytest.approx(50.0)

        # Render v1 → no prior version, no deltas.
        baseline_facts = build_run_facts(pd, "eval_results_v1_a.json")
        for m in baseline_facts.models:
            assert m.pass_rate_delta is None
            assert m.metric_rate_deltas == {}


# --------------------------------------------------------------------------
# build_exec_facts
# --------------------------------------------------------------------------

class TestBuildExecFacts:
    def test_journey_walks_run_files_not_state(self, tmp_path):
        pd = _scaffold_project(
            tmp_path,
            prompts={"baseline": "b", "v1": "v1", "v2": "v2", "v3": "v3"},
        )
        for v, ts in [("baseline", "2026-06-01T00:00:00Z"),
                      ("v1", "2026-06-02T00:00:00Z"),
                      ("v2", "2026-06-03T00:00:00Z"),
                      ("v3", "2026-06-04T00:00:00Z")]:
            run = _model_run(SONNET, prompt_version=v, timestamp=ts, cases=[
                _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            ])
            _write_run(pd, [run], f"eval_results_{v}_x.json")
        _write_state(pd, target_metric_scores={"Acc": 0.5}, iterations=[
            {"iteration": 1, "prompt_version": "v3", "run_id": "eval_results_v3_x.json",
             "metrics_summary": {}, "total_cases": 1, "passed": 1, "pass_rate": 1.0,
             "timestamp": "2026-06-04T00:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0}},
        ])
        ef = build_exec_facts(pd)
        versions = [j.version for j in ef.journey]
        assert versions == ["baseline", "v1", "v2", "v3"]

    def test_winner_picked_via_trophy_not_latest(self, tmp_path):
        pd = _scaffold_project(
            tmp_path, prompts={"v8": "winner", "v9": "regression"},
        )
        winner = _model_run(SONNET, prompt_version="v8",
                            timestamp="2026-06-12T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 1.0, True)]),
            _case(1, success=True, metric_specs=[("Acc", 1.0, 1.0, True)]),
        ])
        loser = _model_run(SONNET, prompt_version="v9",
                           timestamp="2026-06-13T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 1.0, True)]),
            _case(1, success=False, metric_specs=[("Acc", 0.0, 1.0, False)]),
        ])
        _write_run(pd, [winner], "eval_results_v8_x.json")
        _write_run(pd, [loser], "eval_results_v9_x.json")
        _write_state(pd, target_metric_scores={"Acc": 1.0}, iterations=[
            {"iteration": 1, "prompt_version": "v8",
             "run_id": "eval_results_v8_x.json",
             "metrics_summary": {"Acc": 1.0}, "total_cases": 2, "passed": 2,
             "pass_rate": 1.0, "timestamp": "2026-06-12T00:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0},
             "per_model_metrics": {SONNET: {"Acc": 1.0}}},
        ])
        ef = build_exec_facts(pd)
        assert ef.winner_version == "v8"
        assert "v8" in ef.winner_run_filename

    def test_winner_no_targets_picks_best_pass_rate_not_latest(self, tmp_path):
        # When the project has NO optimization state / targets, the trophy
        # ranker returns None. The exec must still pick the BEST run, not
        # the newest — picking newest would silently recommend a
        # regression that happens to be the most recent run.
        pd = _scaffold_project(tmp_path, prompts={"v8": "winner", "v9": "regression"})
        winner = _model_run(SONNET, prompt_version="v8",
                            timestamp="2026-06-12T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
        ])
        regression = _model_run(SONNET, prompt_version="v9",
                                timestamp="2026-06-13T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
        ])
        _write_run(pd, [winner], "eval_results_v8_x.json")
        _write_run(pd, [regression], "eval_results_v9_x.json")
        # No state.json written → no targets → trophy picker returns None.
        ef = build_exec_facts(pd)
        assert ef.winner_version == "v8"
        assert "v8" in ef.winner_run_filename

    def test_generated_from_label_no_raw_filename(self, tmp_path):
        # The exec stamp is humanized — it must NOT leak the raw .json
        # filename into the user-visible label.
        pd = _scaffold_project(tmp_path, prompts={"v8": "p"})
        run = _model_run(SONNET, prompt_version="v8",
                         timestamp="2026-06-12T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
        ])
        _write_run(pd, [run], "eval_results_v8_20260612_223238.json")
        _write_state(pd, target_metric_scores={"Acc": 0.5}, iterations=[
            {"iteration": 1, "prompt_version": "v8",
             "run_id": "eval_results_v8_20260612_223238.json",
             "metrics_summary": {"Acc": 1.0}, "total_cases": 1, "passed": 1,
             "pass_rate": 1.0, "timestamp": "2026-06-12T00:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0},
             "per_model_metrics": {SONNET: {"Acc": 1.0}}},
        ])
        ef = build_exec_facts(pd)
        assert ".json" not in ef.generated_from_label
        assert "eval_results" not in ef.generated_from_label
        # And it does carry version + date.
        assert "v8" in ef.generated_from_label
        assert "2026" in ef.generated_from_label

    def test_exec_carries_category_count(self, tmp_path):
        # The exec facts expose category_count, so the chip counts
        # categories rather than prompt versions.
        pd = _scaffold_project(tmp_path, prompts={"v1": "p"})
        run = _model_run(SONNET, prompt_version="v1", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)], category="alpha"),
            _case(1, success=True, metric_specs=[("Acc", 1.0, 0.5, True)], category="beta"),
        ])
        _write_run(pd, [run], "eval_results_v1_x.json")
        _write_state(pd, target_metric_scores={"Acc": 0.5}, iterations=[
            {"iteration": 1, "prompt_version": "v1",
             "run_id": "eval_results_v1_x.json",
             "metrics_summary": {}, "total_cases": 2, "passed": 2,
             "pass_rate": 1.0, "timestamp": "2026-06-05T10:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0},
             "per_model_metrics": {SONNET: {"Acc": 1.0}}},
        ])
        ef = build_exec_facts(pd)
        assert ef.category_count == 2


# --------------------------------------------------------------------------
# _resolve_exec_winner — shared between build_exec_facts and the
# post-eval auto-regen guard. Asymmetry between these two callers is
# what previously left no-targets projects stuck on the lean fallback
# after every eval (GET dropped the cached narrative; the guard's
# trophy-only check returned None and refused to regen).
# --------------------------------------------------------------------------

class TestResolveExecWinner:
    def test_no_targets_returns_best_pass_rate_run(self, tmp_path):
        # Same setup as test_winner_no_targets_picks_best_pass_rate_not_latest
        # but exercising the resolver directly — pinning the symmetric
        # contract the dispatcher relies on (any non-None here = the
        # GET-side cache drop will fire = the regen MUST also fire).
        from app.services.eval_runner import EvalRunner
        from app.services.optimization_state import OptimizationState
        pd = _scaffold_project(tmp_path, prompts={"v8": "winner", "v9": "regression"})
        winner = _model_run(SONNET, prompt_version="v8",
                            timestamp="2026-06-12T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
        ])
        regression = _model_run(SONNET, prompt_version="v9",
                                timestamp="2026-06-13T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
        ])
        _write_run(pd, [winner], "eval_results_v8_x.json")
        _write_run(pd, [regression], "eval_results_v9_x.json")
        # No state.json → no targets. This used to return None, so the
        # auto-regen guard skipped while GET still dropped the narrative.
        winner_file = _resolve_exec_winner(pd, EvalRunner(), OptimizationState.load(pd))
        assert winner_file == "eval_results_v8_x.json"

    def test_no_targets_winner_shifts_when_new_run_is_better(self, tmp_path):
        # The scenario this guards: project has no targets, a new eval
        # lands with a higher pass rate, the resolver returns the new file →
        # guard's prior!=new comparison fires → the exec narrative
        # regenerates instead of getting dropped.
        from app.services.eval_runner import EvalRunner
        from app.services.optimization_state import OptimizationState
        pd = _scaffold_project(tmp_path, prompts={"v1": "first", "v2": "better"})
        first = _model_run(SONNET, prompt_version="v1",
                           timestamp="2026-06-10T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=False, metric_specs=[("Acc", 0.0, 0.5, False)]),
        ])
        _write_run(pd, [first], "eval_results_v1_x.json")
        prior = _resolve_exec_winner(pd, EvalRunner(), OptimizationState.load(pd))
        better = _model_run(SONNET, prompt_version="v2",
                            timestamp="2026-06-11T00:00:00Z", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
            _case(1, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
        ])
        _write_run(pd, [better], "eval_results_v2_x.json")
        new = _resolve_exec_winner(pd, EvalRunner(), OptimizationState.load(pd))
        assert prior == "eval_results_v1_x.json"
        assert new == "eval_results_v2_x.json"
        assert prior != new  # → guard regenerates the narrative.

    def test_no_runs_returns_none(self, tmp_path):
        from app.services.eval_runner import EvalRunner
        from app.services.optimization_state import OptimizationState
        pd = _scaffold_project(tmp_path, prompts={"v1": "p"})
        assert _resolve_exec_winner(pd, EvalRunner(), OptimizationState.load(pd)) is None

    def test_with_targets_defers_to_trophy_ranker(self, tmp_path):
        # Sanity: with targets configured the resolver returns the trophy
        # run (matching dashboard semantics), not the best-pass-rate
        # fallback.
        from app.services.eval_runner import EvalRunner
        from app.services.optimization_state import OptimizationState
        pd = _scaffold_project(tmp_path, prompts={"v1": "p"})
        run = _model_run(SONNET, prompt_version="v1", cases=[
            _case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)]),
        ])
        _write_run(pd, [run], "eval_results_v1_x.json")
        _write_state(pd, target_metric_scores={"Acc": 0.5}, iterations=[
            {"iteration": 1, "prompt_version": "v1",
             "run_id": "eval_results_v1_x.json",
             "metrics_summary": {"Acc": 1.0}, "total_cases": 1, "passed": 1,
             "pass_rate": 1.0, "timestamp": "2026-06-05T10:00:00Z",
             "per_model_pass_rates": {SONNET: 1.0},
             "per_model_metrics": {SONNET: {"Acc": 1.0}}},
        ])
        winner_file = _resolve_exec_winner(pd, EvalRunner(), OptimizationState.load(pd))
        assert winner_file == "eval_results_v1_x.json"


# --------------------------------------------------------------------------
# Sidecar I/O
# --------------------------------------------------------------------------

class TestReportJsonSidecar:
    def test_write_and_read_run_sidecar(self, project_dir):
        facts = build_run_facts(project_dir)
        narrative = {"takeaway": "It worked.", "failure_patterns": []}
        path = write_report_json(
            project_dir, "run", facts, narrative,
            run_filename="eval_results_v2_20260605.json",
        )
        assert path.exists()
        assert path.name == "eval_results_v2_20260605.report.json"
        data = read_report_json(
            project_dir, "run", run_filename="eval_results_v2_20260605.json",
        )
        assert data is not None
        assert data["schema_version"] == REPORT_SCHEMA_VERSION
        assert data["scope"] == "run"
        assert data["narrative"]["takeaway"] == "It worked."
        assert data["facts"]["project_name"] == "My Eval Project"

    def test_executive_sidecar_uses_executive_stem(self, project_dir):
        _write_state(project_dir, target_metric_scores={"Correctness": 0.5, "Relevance": 0.5})
        ef = build_exec_facts(project_dir)
        path = write_report_json(project_dir, "executive", ef, {})
        assert path.name == "executive.report.json"

    def test_read_returns_none_when_missing(self, project_dir):
        assert read_report_json(project_dir, "executive") is None

    def test_read_returns_none_for_unknown_schema_version(self, project_dir):
        target = report_json_path(project_dir, "executive")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 999, "facts": {}, "narrative": {}}))
        assert read_report_json(project_dir, "executive") is None


# --------------------------------------------------------------------------
# Facts-pack JSON serializability (the API endpoint returns this verbatim)
# --------------------------------------------------------------------------

class TestFactsJsonSerializable:
    def test_run_facts_round_trips_through_json(self, project_dir):
        facts = build_run_facts(project_dir)
        d = _facts_to_jsonable(facts)
        # json.dumps must succeed without a custom encoder.
        s = json.dumps(d, default=str)
        round_tripped = json.loads(s)
        assert round_tripped["project_name"] == facts.project_name
        assert round_tripped["case_count"] == facts.case_count
        assert round_tripped["category_count"] == facts.category_count

    def test_exec_facts_round_trips_through_json(self, project_dir):
        _write_state(project_dir, target_metric_scores={"Correctness": 0.5, "Relevance": 0.5})
        ef = build_exec_facts(project_dir)
        d = _facts_to_jsonable(ef)
        s = json.dumps(d, default=str)
        round_tripped = json.loads(s)
        assert round_tripped["winner_version"] == ef.winner_version
        # generated_from_label is the humanized stamp the frontend renders
        # in the exec header — never the raw filename.
        assert round_tripped["generated_from_label"] == ef.generated_from_label
        assert ".json" not in round_tripped["generated_from_label"]


# --------------------------------------------------------------------------
# ModelFacts.pricing_available flag — server-side classification
# --------------------------------------------------------------------------


class TestUnknownCostIsNotZeroCost:
    """Null cost means UNKNOWN, and must never render as $0.00.

    The harness always writes ``generation_cost_usd`` and ``judge_cost_usd``,
    writing null when it had no pricing entry and refused to invent rates. The
    availability flags used to test for KEY PRESENCE — a check dating from runs
    that predated cost tracking — so a run reporting unknown judge spend came
    back available and rendered $0.00: exactly what the flag exists to prevent.
    """

    def test_null_judge_cost_is_reported_unavailable(self, tmp_path):
        from app.services.report import build_run_facts

        project_dir = _scaffold_project(tmp_path)
        run_filename = _write_run(
            project_dir,
            [_model_run(
                HAIKU,
                cases=[_case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)])],
                judge_cost_usd=None,
            )],
            "run-2026-06-20.json",
        )

        model = build_run_facts(project_dir, run_filename).models[0]
        assert model.judge_cost_available is False, (
            "unknown judge spend was reported as available, so the report shows "
            "$0.00 for a number nobody measured"
        )
        assert model.judge_cost_usd == 0.0, "the facts carry 0, the flag hides it"

    def test_real_judge_cost_is_reported_available(self, tmp_path):
        from app.services.report import build_run_facts

        project_dir = _scaffold_project(tmp_path)
        run_filename = _write_run(
            project_dir,
            [_model_run(
                HAIKU,
                cases=[_case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)])],
                judge_cost_usd=0.01,
            )],
            "run-2026-06-21.json",
        )
        model = build_run_facts(project_dir, run_filename).models[0]
        assert model.judge_cost_available is True

    def test_zero_judge_cost_is_available_not_unknown(self, tmp_path):
        """0.0 is a measurement. Only null is unknown."""
        from app.services.report import build_run_facts

        project_dir = _scaffold_project(tmp_path)
        run_filename = _write_run(
            project_dir,
            [_model_run(
                HAIKU,
                cases=[_case(0, success=True, metric_specs=[("Acc", 1.0, 0.5, True)])],
                judge_cost_usd=0.0,
            )],
            "run-2026-06-22.json",
        )
        model = build_run_facts(project_dir, run_filename).models[0]
        assert model.judge_cost_available is True


class TestPricingAvailableFlag:
    """ModelFacts.pricing_available mirrors judge_cost_available — set
    False only when LiteLLM has no entry for the model id (the NA
    sentinel from bedrock_pricing). Renderer hides USD values rather
    than $0.00 in this case (server-side)."""

    def test_pricing_available_true_for_runtime_model(self, tmp_path):
        """Runtime models default to pricing_available=True even when
        lookup_cached_pricing returns None (cache cold) — only the NA
        sentinel signals 'intentionally unavailable'."""
        from app.services.report import build_run_facts

        project_dir = _scaffold_project(tmp_path)
        run_filename = _write_run(
            project_dir,
            [
                _model_run(
                    HAIKU,
                    cases=[_case(0, success=True, metric_specs=[
                        ("Acc", 1.0, 0.5, True),
                    ])],
                ),
            ],
            "run-2026-06-19.json",
        )

        facts = build_run_facts(project_dir, run_filename)
        assert facts.models[0].pricing_available is True

    def test_pricing_available_false_for_na_priced_mantle_model(
        self, tmp_path, monkeypatch,
    ):
        """A mantle-tier model id with no LiteLLM pricing yields
        pricing_available=False so the renderer shows literal n/a in
        cost columns (closes the bug where $0.00 was shown)."""
        from app.services import bedrock_pricing
        from app.services.report import build_run_facts

        mantle_id = "qwen.qwen3-coder"

        def fake_lookup(model_id):
            if model_id == mantle_id:
                return bedrock_pricing.NA
            return None

        # Patch the binding in the consumer module (report.py imports
        # lookup_cached_pricing at module-top, so we patch *that* name).
        monkeypatch.setattr(
            "app.services.report.lookup_cached_pricing",
            fake_lookup,
        )

        project_dir = _scaffold_project(tmp_path)
        run_filename = _write_run(
            project_dir,
            [
                _model_run(
                    mantle_id,
                    cases=[_case(0, success=True, metric_specs=[
                        ("Acc", 1.0, 0.5, True),
                    ])],
                ),
            ],
            "run-2026-06-19.json",
        )

        facts = build_run_facts(project_dir, run_filename)
        assert facts.models[0].pricing_available is False


# --------------------------------------------------------------------------
# Path containment — the report builder's two untrusted path inputs
# --------------------------------------------------------------------------

def _two_tenant_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Build the cloud per-user layout and return (attacker_dir, victim_dir).

    Cloud mode nests projects under ``{root}/users/{owner_sub}/projects/{pid}``
    (``ProjectManager._project_dir``), so four ``..`` segments out of a
    project's ``output/`` reach a sibling tenant. Mirroring that layout is
    what makes these tests exercise the real escape shape rather than a
    generic ``..``.
    """
    attacker = tmp_path / "users" / "attacker-sub" / "projects" / "proj-A"
    victim = tmp_path / "users" / "victim-sub" / "projects" / "proj-V"
    attacker.mkdir(parents=True)
    victim.mkdir(parents=True)
    _scaffold_project(attacker)
    _scaffold_project(victim, prompts={"v7": "VICTIM SECRET SYSTEM PROMPT\nChanges: victim-change-line\n"})
    return attacker, victim


class TestRunFilenameContainment:
    """``build_run_facts`` must not read run files outside ``output/``.

    The run filename arrives in a POST body, so unlike the eval-runs
    routes it is not constrained by single-segment route matching.
    """

    def test_traversal_to_other_tenant_is_rejected(self, tmp_path):
        attacker, victim = _two_tenant_layout(tmp_path)
        _write_run(
            victim,
            [_model_run(SONNET, cases=[_case(0, success=False, metric_specs=[
                ("Acc", 0.1, 0.5, False),
            ])])],
            "eval_results_v7_secret.json",
        )
        payload = "../../../../victim-sub/projects/proj-V/output/eval_results_v7_secret.json"

        with pytest.raises(FileNotFoundError):
            build_run_facts(attacker, payload)

    def test_absolute_path_is_rejected(self, tmp_path):
        """``Path / "/abs"`` discards the base, so absolute input escapes too."""
        attacker, victim = _two_tenant_layout(tmp_path)
        target = _write_run(
            victim,
            [_model_run(SONNET, cases=[_case(0, success=True, metric_specs=[
                ("Acc", 0.9, 0.5, True),
            ])])],
            "eval_results_v7_secret.json",
        )
        absolute = str(victim / "output" / target)

        with pytest.raises(FileNotFoundError):
            build_run_facts(attacker, absolute)

    def test_own_run_file_still_loads(self, tmp_path):
        """The guard must not break the legitimate bare-filename case."""
        attacker, _ = _two_tenant_layout(tmp_path)
        run_filename = _write_run(
            attacker,
            [_model_run(SONNET, cases=[_case(0, success=True, metric_specs=[
                ("Acc", 0.9, 0.5, True),
            ])])],
            "eval_results_v2_20260605.json",
        )
        facts = build_run_facts(attacker, run_filename)
        assert facts.run_filename == run_filename
        assert facts.models[0].model_id == SONNET


class TestPromptVersionContainment:
    """``prompt_version`` comes from run-file content, so it is untrusted.

    A poisoned ``hyperparameters.prompt_version`` must not read a prompt
    outside the project's own ``prompts/`` directory. Asserting on the
    returned facts (not just the path helper) is the point — the leak
    shape was ``facts.prompt_content`` reaching the HTTP response.
    """

    @staticmethod
    def _run_with_prompt_version(version: str) -> list[dict]:
        return [_model_run(
            SONNET,
            prompt_version=version,
            cases=[_case(0, success=True, metric_specs=[("Acc", 0.9, 0.5, True)])],
        )]

    def test_traversal_does_not_leak_other_tenant_prompt(self, tmp_path):
        attacker, _ = _two_tenant_layout(tmp_path)
        run_filename = _write_run(
            attacker,
            self._run_with_prompt_version(
                "../../../../victim-sub/projects/proj-V/prompts/v7"
            ),
            "eval_results_poisoned.json",
        )

        facts = build_run_facts(attacker, run_filename)

        assert "VICTIM SECRET" not in facts.prompt_content
        assert facts.prompt_changes != "victim-change-line"
        # Falls back to the attacker's own newest prompt rather than erroring.
        assert facts.prompt_label in {"baseline", "v2"}

    def test_absolute_path_does_not_leak_other_tenant_prompt(self, tmp_path):
        attacker, victim = _two_tenant_layout(tmp_path)
        run_filename = _write_run(
            attacker,
            self._run_with_prompt_version(str(victim / "prompts" / "v7")),
            "eval_results_poisoned_abs.json",
        )

        facts = build_run_facts(attacker, run_filename)

        assert "VICTIM SECRET" not in facts.prompt_content
        assert facts.prompt_changes != "victim-change-line"

    def test_own_prompt_version_still_loads(self, tmp_path):
        """The guard must not break the legitimate bare-stem case."""
        attacker, _ = _two_tenant_layout(tmp_path)
        (attacker / "prompts" / "v3.md").write_text(
            "OWN PROMPT BODY\nChanges: own-change-line\n"
        )
        run_filename = _write_run(
            attacker,
            self._run_with_prompt_version("v3"),
            "eval_results_v3.json",
        )

        facts = build_run_facts(attacker, run_filename)

        assert facts.prompt_label == "v3"
        assert "OWN PROMPT BODY" in facts.prompt_content
        assert facts.prompt_changes == "own-change-line"


class TestCachedTokenAccounting:
    """Prompt size and cache hit fraction on a CACHED run.

    Why this class exists: ``_case`` has always accepted ``cache_read`` /
    ``cache_write``, and no test ever passed a non-zero value. Every fixture was
    uncached, so ``test_token_aggregation`` asserted 100.0 tokens and a 0.0 ratio
    and passed identically whether or not the code counted cache reads. Two
    defects lived behind that blind spot:

    * ``avg_input_tokens`` summed only ``input_tokens``. Bedrock reports
      ``cacheReadInputTokens`` as a SEPARATE counter, so a cached prompt showed
      near-zero input tokens on the field the caching diagnostic calls its
      "prompt-size proxy".
    * ``cache_read_ratio`` was ``read / (read + write)`` — the read/write mix —
      while its docstring called it the hit ratio, so it read about 0.9 for a
      prompt served almost entirely from cache.

    The numbers below have the shape a cached prompt produces: a tiny uncached
    remainder against a large cache read.
    """

    #: One case's worth: 2 uncached tokens, 2,320 served from cache.
    UNCACHED = 2
    CACHE_READ = 2320

    @pytest.fixture
    def cached_project(self, tmp_path: Path) -> Path:
        pd = _scaffold_project(tmp_path)
        run = _model_run(SONNET, cases=[
            _case(
                i, success=True,
                metric_specs=[("Correctness", 0.9, 0.5, True)],
                in_tokens=self.UNCACHED,
                cache_read=self.CACHE_READ,
                cache_write=0,
            )
            for i in range(4)
        ])
        _write_run(pd, [run], "eval_results_v2_cached.json")
        return pd

    def _sonnet(self, pd: Path):
        return next(
            m for m in build_run_facts(pd).models if m.model_id == SONNET
        )

    def test_prompt_size_includes_cache_reads(self, cached_project):
        """THE DEFECT: this reported 2.0 instead of 2322.0."""
        assert self._sonnet(cached_project).avg_input_tokens == pytest.approx(
            self.UNCACHED + self.CACHE_READ
        )

    def test_prompt_size_is_not_the_uncached_remainder(self, cached_project):
        """Stated separately so the failure message is unambiguous about which
        of the two numbers the code picked."""
        assert self._sonnet(cached_project).avg_input_tokens != pytest.approx(
            self.UNCACHED
        )

    def test_cache_ratio_is_the_cached_fraction_of_the_prompt(self, cached_project):
        expected = self.CACHE_READ / (self.UNCACHED + self.CACHE_READ)
        assert self._sonnet(cached_project).cache_read_ratio == pytest.approx(expected)
        # ~0.999, not the ~0.9 the read/write mix produced.
        assert self._sonnet(cached_project).cache_read_ratio > 0.99

    def test_cache_writes_do_not_dilute_the_hit_fraction(self, tmp_path: Path):
        """The old formula divided by (read + write), so a cache WRITE pushed the
        "hit ratio" down even though writes are part of populating the cache, not
        of missing it. Here a large write must leave the fraction untouched.
        """
        pd = _scaffold_project(tmp_path)
        run = _model_run(SONNET, cases=[
            _case(0, success=True, metric_specs=[("Correctness", 0.9, 0.5, True)],
                  in_tokens=self.UNCACHED, cache_read=self.CACHE_READ,
                  cache_write=5000),
        ])
        _write_run(pd, [run], "eval_results_v2_cw.json")
        ratio = self._sonnet(pd).cache_read_ratio
        expected = self.CACHE_READ / (self.UNCACHED + self.CACHE_READ)
        assert ratio == pytest.approx(expected)
        # Under the old formula this would have been 2320/7320 = 0.317.
        assert ratio > 0.99

    def test_an_uncached_run_still_reports_zero(self, project_dir):
        """The frontend's "caching is OFF" note keys on ``=== 0``
        (``RunReport.tsx`` ``cacheOffNote``), so this value must stay exactly
        0.0 — not 0.0001 — when nothing was cached."""
        sonnet = next(
            m for m in build_run_facts(project_dir).models if m.model_id == SONNET
        )
        assert sonnet.cache_read_ratio == 0.0
        assert sonnet.avg_input_tokens == pytest.approx(100.0)

    def test_the_ratio_survives_serialization_rounding(self, cached_project):
        """``cache_read_ratio`` must not sit in ``_PERCENT_FIELDS``: 1-decimal
        rounding collapses 0.999 to 1.0 and 0.94 to 0.9, erasing the resolution
        this fix provides, because real values cluster above 0.9."""
        from app.services.report import _facts_to_jsonable

        facts = build_run_facts(cached_project)
        model = next(
            m for m in _facts_to_jsonable(facts)["models"]
            if m["model_id"] == SONNET
        )
        assert model["cache_read_ratio"] != 1.0, (
            "the ratio was rounded to 1 decimal, which cannot distinguish a "
            "94%-cached prompt from a 99.9%-cached one"
        )
        assert model["cache_read_ratio"] > 0.99
