# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The metric-name derivation, and the joins that depend on it agreeing.

Three platform-side copies of "strip DeepEval's ``[GEval]`` suffix" existed and
one of them differed: ``report._short_metric_name`` was ``raw.split(" [")[0]``,
which cuts at the FIRST bracket anywhere in the name. So a metric named
``"Recall [strict] check"`` rendered in reports as ``"Recall"`` while the rest of
the system kept the full name — and any per-metric marker keyed on the report's
name would never find its own audit verdict.

These tests pin two things a single derivation is not enough to guarantee:

* the platform copy and the harness template's copy behave identically (the
  template cannot import from here, so they are two implementations by
  necessity);
* the name spaces that have to JOIN actually do — the metric audit's verdict
  keys against the report's metric list — because a silent join miss looks
  exactly like "nothing was wrong".
"""
from __future__ import annotations

import json
from pathlib import Path

from app.services.metric_name import base_metric_name

# Inputs chosen for the cases where the three old copies disagreed, not just the
# happy path: a trailing suffix, no suffix, brackets mid-name, brackets with
# nothing bracket-shaped after them, and empty.
CASES = [
    ("Answer Correctness [GEval]", "Answer Correctness"),
    ("Answer Correctness", "Answer Correctness"),
    ("Recall [strict] check", "Recall [strict] check"),
    ("Exact Match [ExactMatchMetric]", "Exact Match"),
    ("[GEval]", ""),
    ("", ""),
]


class TestBaseMetricName:
    def test_strips_only_a_trailing_bracket_suffix(self):
        for raw, want in CASES:
            assert base_metric_name(raw) == want, raw

    def test_brackets_mid_name_are_content(self):
        """The case the old report copy got wrong, called out on its own.

        ``split(" [")[0]`` returned ``"Recall"`` here, so the report showed a
        name that matched nothing else in the system.
        """
        assert base_metric_name("Recall [strict] check") == "Recall [strict] check"

    def test_matches_the_harness_template_implementation(self):
        """Two implementations, one behavior — the template runs in another venv.

        A drift here is invisible until a project's metric name happens to hit
        the differing case, at which point gating, convergence and the reports
        each disagree about which metric they are talking about.
        """
        from tests.harness_template_stubs import load_template_runner

        template = load_template_runner("harness_template_runner_metricname")
        template_impl = template.EvalRunner._base_metric_name
        for raw, _want in CASES:
            assert template_impl(raw) == base_metric_name(raw), raw


class TestAuditAndReportShareOneNameSpace:
    """A suffixed run file must not split into two name spaces.

    The report renders a metric list; the audit records a verdict per metric.
    The per-metric trust marker joins them by NAME. If the run file carries
    ``"X [GEval]"`` and only one side normalizes, the join misses and the marker
    silently never appears — the failure mode is an absent warning, which is
    indistinguishable from a clean run.
    """

    @staticmethod
    def _suffixed_run() -> list[dict]:
        """One model, 200 cases, both metric names carrying a type suffix."""
        gate, diag = "Answer Correctness [GEval]", "Normalized Match [ExactMatch]"
        cases = []
        for i in range(200):
            cases.append({
                "name": f"case-{i}",
                "success": i >= 60,
                "input": f"q{i}",
                "actual_output": f"a{i}",
                "expected_output": f"e{i}",
                "additional_metadata": {"id": f"c{i}"},
                "metrics_data": [
                    {"name": gate, "score": 1.0 if i >= 60 else 0.0,
                     "threshold": 1.0, "success": i >= 60, "reason": "r",
                     "error": None},
                    # Saturated diagnostic: passes all but one case.
                    {"name": diag, "score": 1.0 if i != 0 else 0.0,
                     "threshold": 1.0, "success": i != 0, "reason": "r",
                     "error": None},
                ],
            })
        return [{
            "source": "manual",
            "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "test_run_id": "haiku-4-5",
            "timestamp": "2026-06-05T10:00:00Z",
            "hyperparameters": {
                "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "endpoint": "runtime", "region": "us-east-1",
                "prompt_version": "v1", "temperature": 0.0, "max_tokens": 2048,
            },
            "results": cases,
            "generation_cost_usd": 0.02,
            "judge_cost_usd": 0.01,
            "data_quality": {
                "reliability": "ok", "errored_count": 0, "dropped_count": 0,
                "null_count": 0, "unrecovered_count": 0, "total_cases": 200,
            },
        }]

    def test_audit_keys_are_base_names(self):
        from app.services.metric_disagreement import analyze_run

        audit = analyze_run(
            self._suffixed_run(),
            gating_metrics={"Answer Correctness"},  # config spelled WITHOUT suffix
        )
        assert audit["status"] == "ok"
        assert set(audit["metrics"]) == {"Answer Correctness", "Normalized Match"}

    def test_a_gate_spelled_two_ways_is_still_a_gate(self):
        """The severity-critical half of the join.

        Config says ``"Answer Correctness"``, the run file says
        ``"Answer Correctness [GEval]"``. If those don't reconcile, the metric that
        decides pass/fail is classified ``diagnostic`` and every finding about
        it is downgraded — the exact ranking error the role-aware severity work
        was for.
        """
        from app.services.metric_disagreement import analyze_run

        audit = analyze_run(
            self._suffixed_run(), gating_metrics={"Answer Correctness"},
        )
        assert audit["metrics"]["Answer Correctness"]["role"] == "gating"
        assert audit["metrics"]["Normalized Match"]["role"] == "diagnostic"

    def test_a_gate_spelled_with_the_suffix_in_CONFIG_is_still_a_gate(self):
        """The other direction, which the test above cannot see.

        Normalizing only the run-file side is enough when the config happens to
        be spelled plainly — so that test stays green even if the gating set is
        left raw. The reverse pairing is the one that needs it: the user's
        ``target_metric_scores`` key carries ``[GEval]`` (the harness config is
        hand-written, and DeepEval's own ``__name__`` is where people copy the
        name from) while the run file records the base name.
        """
        from app.services.metric_disagreement import analyze_run

        runs = self._suffixed_run()
        # Strip the suffixes from the run file so ONLY the config carries them.
        for case in runs[0]["results"]:
            for md in case["metrics_data"]:
                md["name"] = base_metric_name(md["name"])

        audit = analyze_run(
            runs, gating_metrics={"Answer Correctness [GEval]"},
        )
        assert audit["metrics"]["Answer Correctness"]["role"] == "gating", (
            "a gate whose config key carries the type suffix was demoted to "
            "diagnostic, which downgrades every finding about the metric that "
            "decides pass/fail"
        )

    def test_trust_reason_metric_matches_the_reported_metric_list(self, tmp_path: Path):
        """End to end: the name a reason names is a name the report renders."""
        from app.services.metric_disagreement import analyze_run
        from app.services.report import build_run_facts
        from app.services.run_trust import build_run_trust

        project_dir = tmp_path / "proj"
        (project_dir / "output").mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "id": "proj-metric-name", "name": "Metric name join",
            "description": "", "created_at": "2026-06-01T00:00:00Z",
        }))
        runs = self._suffixed_run()
        run_file = "eval_results_v1_20260605_100000.json"
        (project_dir / "output" / run_file).write_text(json.dumps(runs))

        facts = build_run_facts(project_dir, run_file)
        trust = build_run_trust(
            metric_audit=analyze_run(runs, gating_metrics={"Answer Correctness"}),
            case_count=200,
        )

        named = {m for r in trust.reasons for m in r.metrics}
        assert named, "expected at least one metric-attributed reason"
        assert named <= set(facts.metric_names), (
            f"trust reasons name {named - set(facts.metric_names)}, which the "
            f"report never renders ({facts.metric_names}) — the marker would "
            "silently never appear"
        )
