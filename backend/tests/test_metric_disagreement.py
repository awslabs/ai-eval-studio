# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-run metric-disagreement detector.

The properties that matter here are not "does it count correctly" (it does) but
the design rules that were arrived at the hard way, each of which a naive
implementation gets wrong:

1. Severity follows error DIRECTION and metric ROLE, never disagreement count.
   A run can hold hundreds of over-strict cases and a single over-lenient one,
   and the single over-lenient case is the serious finding.
2. Nothing is tested against ZERO disagreement. Zero is not stable — two runs
   with identical metrics can differ by one reverse contradiction.
3. No English is matched. Verdicts come from ``success``/``score`` only; reason
   strings ride along as evidence for a human.
4. A run is never failed. Worst case is a "disputed" verdict on a sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.services.metric_disagreement import (
    DISPUTE_RATIO,
    analyze_run,
    audit_run_file,
    sidecar_path,
)

GATE = "Judge"
DIAG = "StrictMatch"


def _case(idx: int, *verdicts: tuple[str, bool], reason: str = "because") -> dict:
    """One serialized test case with the given per-metric verdicts."""
    return {
        "name": f"test_case_{idx}",
        "success": verdicts[0][1],
        "input": "in",
        "actual_output": "`out`",
        "expected_output": "out",
        "additional_metadata": {"id": idx, "case_index": idx},
        "metrics_data": [
            {"name": n, "score": 1.0 if ok else 0.0, "success": ok,
             "threshold": 0.5, "reason": reason, "error": None}
            for n, ok in verdicts
        ],
    }


def _run(model: str, cases: list[dict]) -> dict:
    return {"model_id": model, "test_run_id": model, "results": cases}


class TestDirectionOutranksMagnitude:
    """The core rule, stated as two runs that invert count vs severity."""

    def test_many_over_strict_is_only_review(self):
        """Diagnostic fails 30 cases the gate passed: wasteful, not corrupting.

        Both metrics must genuinely discriminate here, or the degenerate-metric
        check fires instead and masks what this test is about.
        """
        # gate fails 0-9; diag fails 0-39 ⇒ 30 over-strict, 0 over-lenient.
        cases = [_case(i, (GATE, i >= 10), (DIAG, i >= 40)) for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})

        assert audit["metrics"][GATE]["verdict"] == "ok", "gate must discriminate"
        assert audit["metrics"][DIAG]["verdict"] == "ok", "diag must discriminate"
        pair = audit["pairs"][0]
        assert pair["a_pass_b_fail"] == 30
        assert pair["b_pass_a_fail"] == 0
        assert pair["verdict"] == "review"
        assert pair["severity"] == "info"
        assert audit["worst_verdict"] == "review"
        assert "over-strict" in pair["findings"][0]

    def test_single_over_lenient_is_disputed(self):
        """Diagnostic passes ONE case the gate failed: outranks the 30 above."""
        # Both must discriminate, or the degeneracy cap fires and masks the point.
        # gate fails 0-9 (90% pass); diag passes case 5 (a gate FAIL) + 40.. (61%).
        cases = [_case(i, (GATE, i >= 10), (DIAG, i == 5 or i >= 40))
                 for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})

        pair = audit["pairs"][0]
        assert pair["b_pass_a_fail"] == 1, "one over-lenient case"
        assert pair["a_pass_b_fail"] > 1, "and many over-strict ones"
        assert pair["verdict"] == "disputed"
        assert audit["worst_verdict"] == "disputed"
        assert any("over-lenient" in f for f in pair["findings"])

    def test_over_lenient_has_no_ratio_floor(self):
        """1-in-1000 must still surface — a floor would discard the real case."""
        # gate fails 0-99 (90% pass); diag passes case 50 (a gate FAIL) + 500.. (50%).
        # Both discriminate, so the single over-lenient case is the only signal.
        cases = [_case(i, (GATE, i >= 100), (DIAG, i == 50 or i >= 500))
                 for i in range(1000)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})

        pair = audit["pairs"][0]
        assert pair["b_pass_a_fail"] == 1
        assert pair["b_pass_a_fail_ratio"] < DISPUTE_RATIO, "far below the floor"
        assert pair["verdict"] == "disputed", "and reported anyway"


class TestGatingRoleEscalates:
    def test_gating_metric_failing_what_another_passes_is_an_error(self):
        # diag must discriminate too (fails 90-99), else the degeneracy cap fires.
        cases = [_case(i, (GATE, i >= 30), (DIAG, i < 90)) for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        pair = audit["pairs"][0]
        assert pair["severity"] == "error"
        assert pair["verdict"] == "disputed"
        assert any("convergence" in f for f in pair["findings"])

    def test_two_diagnostics_disagreeing_is_informational(self):
        """No gate declared ⇒ no escalation, and no false claim about gating."""
        cases = [_case(i, ("A", True), ("B", i >= 30)) for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics=set())
        pair = audit["pairs"][0]
        assert pair["severity"] == "info"
        assert "neither gates case success" in pair["findings"][0]

    def test_gating_pair_never_claims_neither_gates(self):
        """Regression: the informational branch used to fire for gating pairs.

        The gate fails a few cases so it discriminates; a gate at a flat 100%
        would be saturated and escalated by ``_escalate_degenerate_gates``, which
        is a different mechanism than the one under test.
        """
        cases = [_case(i, (GATE, i >= 5), (DIAG, i >= 30)) for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        joined = " ".join(audit["pairs"][0]["findings"])
        assert "neither gates case success" not in joined


class TestDegenerateAndErroredMetrics:
    def test_always_failing_metric_is_non_discriminating(self):
        cases = [_case(i, (GATE, i >= 50), (DIAG, False)) for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert audit["metrics"][DIAG]["verdict"] == "non_discriminating"

    def test_null_scored_metric_is_not_counted_as_disagreement(self):
        """A metric that ERRORED must not be blamed on its partner."""
        cases = []
        for i in range(50):
            c = _case(i, (GATE, i >= 5), (DIAG, False))
            c["metrics_data"][1]["score"] = None  # judge failed to score
            cases.append(c)
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert DIAG not in audit["metrics"], "errored metric contributes nothing"
        assert audit["pairs"] == []

    def test_small_run_is_insufficient_data(self):
        cases = [_case(i, (GATE, True), (DIAG, False)) for i in range(5)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert audit["pairs"][0]["verdict"] == "insufficient_data"
        assert audit["worst_verdict"] == "ok", "never escalate on thin data"


class TestRobustness:
    def test_garbage_artifact_is_not_analyzable_not_an_exception(self):
        for junk in (None, {}, [], "nope", [1, 2, 3], [{"no": "results"}]):
            audit = analyze_run(junk, gating_metrics={GATE})
            assert audit["status"] in ("not_analyzable", "ok")
            assert audit["worst_verdict"] == "ok"

    def test_evidence_carries_reason_without_being_matched(self):
        # The gate discriminates (fails 0-4) so only the evidence path is live.
        cases = [_case(i, (GATE, i >= 5), (DIAG, i >= 30), reason="strings differ")
                 for i in range(100)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        ev = audit["pairs"][0]["evidence"]
        assert ev and ev[0]["failed_reason"] == "strings differ"
        assert ev[0]["failed"] == DIAG and ev[0]["passed"] == GATE

    def test_disagreements_are_counted_per_model_not_pooled_by_index(self):
        """Two models share case indices; verdicts must not collide.

        The gate fails 3 of model b's cases so its pooled pass rate is 95% and it
        is not treated as saturated — that would be a second live mechanism.
        """
        a = [_case(i, (GATE, True), (DIAG, False)) for i in range(30)]
        b = [_case(i, (GATE, i >= 3), (DIAG, True)) for i in range(30)]
        audit = analyze_run([_run("m1", a), _run("m2", b)], gating_metrics={GATE})
        pair = audit["pairs"][0]
        assert pair["cases_compared"] == 60
        assert pair["a_pass_b_fail"] == 30


class TestSidecar:
    def test_writes_sidecar_beside_run_file(self, tmp_path: Path):
        run = tmp_path / "eval_results_v1_20260903.json"
        cases = [_case(i, (GATE, i >= 5), (DIAG, i >= 40)) for i in range(100)]
        run.write_text(json.dumps([_run("m1", cases)]))

        audit = audit_run_file(run, {GATE})

        out = sidecar_path(run)
        assert out.name == "eval_results_v1_20260903.metric-audit.json"
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk["run_file"] == run.name
        assert on_disk["worst_verdict"] == audit["worst_verdict"]
        assert on_disk["gating_metrics"] == [GATE]

    def test_unreadable_run_file_returns_none_and_writes_nothing(self, tmp_path: Path):
        run = tmp_path / "eval_results_bad.json"
        run.write_text("{not json")
        assert audit_run_file(run, {GATE}) is None
        assert not sidecar_path(run).exists()


class TestFindingsAreAdditive:
    """Both directions are frequently true at once; neither may silence the other.

    Regression: the branches were each guarded on ``verdict == "ok"``, so a single
    over-lenient case set "disputed" first and a large over-strict finding — the
    defect the detector was built for — emitted no text at all.
    """

    def _both_directions(self) -> list[dict]:
        # gate fails 0-9 and case 500; diag fails 0-39 but PASSES case 500.
        return [
            _case(i, (GATE, i >= 10 and i != 500), (DIAG, i >= 40 or i == 500))
            for i in range(600)
        ]

    def test_both_findings_are_reported(self):
        audit = analyze_run(
            [_run("m1", self._both_directions())], gating_metrics={GATE}
        )
        pair = audit["pairs"][0]
        assert pair["a_pass_b_fail"] > 20, "over-strict direction present"
        assert pair["b_pass_a_fail"] == 1, "over-lenient direction present"
        joined = " ".join(pair["findings"])
        assert "over-lenient" in joined
        assert "over-strict" in joined, (
            "the dominant over-strict finding was silenced by the rarer "
            "over-lenient one"
        )
        assert len(pair["findings"]) >= 2

    def test_severity_is_not_demoted_by_a_later_info_finding(self):
        audit = analyze_run(
            [_run("m1", self._both_directions())], gating_metrics={GATE}
        )
        pair = audit["pairs"][0]
        assert pair["severity"] == "warning", "info finding must not demote"
        assert pair["verdict"] == "disputed"

    def test_evidence_covers_the_rare_direction(self):
        """The verdict is about the rare direction, so evidence must show it."""
        audit = analyze_run(
            [_run("m1", self._both_directions())], gating_metrics={GATE}
        )
        ev = audit["pairs"][0]["evidence"]
        passers = {e["passed"] for e in ev}
        assert GATE in passers, "over-strict examples missing"
        assert DIAG in passers, (
            "the single over-lenient case never reached the evidence sample — a "
            "flat head-slice of a many-to-one list is all over-strict"
        )


class TestUninformativeComparatorCannotEscalate:
    """A non-discriminating metric is not evidence against another metric.

    Without the cap the audit reports DISPUTED/error because the gating metric
    "failed N% of cases that 'Answer Relevancy' passed" — while the same audit has
    already labelled 'Answer Relevancy' non_discriminating at a near-100% pass rate.
    A metric that passes nearly everything makes that comparison close to
    tautological, and the alarm names the DISCRIMINATING metric as the suspect.

    Rules written against a non-gating metric STRICTER than the gate — where
    disagreement does carry information — silently assume that shape. Permissive
    judges are stock metrics, so this case is common.
    """

    LENIENT = "Relevancy"

    def _gate_vs_lenient(self, n: int = 600):
        # Lenient passes all but 2 (>= NON_DISCRIMINATING_RATIO ⇒ degenerate);
        # the gate fails 25% ⇒ a large "gate failed what lenient passed" ratio.
        return [
            _case(i, (GATE, i % 4 != 0), (self.LENIENT, i >= 2))
            for i in range(n)
        ]

    def test_lenient_comparator_is_flagged_non_discriminating(self):
        audit = analyze_run(
            [_run("m1", self._gate_vs_lenient())], gating_metrics={GATE}
        )
        assert audit["metrics"][self.LENIENT]["verdict"] == "non_discriminating"

    def test_pair_severity_is_capped_not_escalated(self):
        audit = analyze_run(
            [_run("m1", self._gate_vs_lenient())], gating_metrics={GATE}
        )
        pair = audit["pairs"][0]
        assert pair["verdict"] == "review", (
            "escalating on an uninformative comparator is a false alarm"
        )
        assert pair["severity"] == "info"
        assert any(f.startswith("SEVERITY CAPPED") for f in pair["findings"]), (
            "the cap must explain itself, not silently downgrade"
        )

    def test_capped_pair_keeps_its_descriptive_findings(self):
        """The numbers still describe the data — only the verdict is capped."""
        audit = analyze_run(
            [_run("m1", self._gate_vs_lenient())], gating_metrics={GATE}
        )
        findings = audit["pairs"][0]["findings"]
        assert len(findings) >= 2, "descriptive findings must survive the cap"

    def test_cap_does_not_apply_when_both_metrics_discriminate(self):
        """The true positive must be preserved: no cap, still disputed."""
        # Both discriminate; DIAG passes 1 case the gate fails ⇒ over-lenient.
        cases = [
            _case(i, (GATE, i >= 10 and i != 500), (DIAG, i >= 40 or i == 500))
            for i in range(600)
        ]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        pair = audit["pairs"][0]
        assert pair["verdict"] == "disputed"
        assert not any(f.startswith("SEVERITY CAPPED") for f in pair["findings"])


class TestSaturatedGateOutranksSaturatedDiagnostic:
    """Degeneracy severity must read the metric's ROLE, not just its pass rate.

    The two runs below were constructed to be mirror images: in A the GATING
    metric is saturated and the diagnostic discriminates; in B the reverse. The
    detector used to report them IDENTICALLY — both ``non_discriminating`` at the
    metric level with the pair capped to review/info — and they are not
    equivalent situations:

    - A: case success is decided by a metric that passes 99.5% of cases, so
      against a 0.90 target the loop converges immediately and every pass rate in
      the report is fiction.
    - B: one report column is useless. Nothing else is affected.

    Worse, ``non_discriminating`` is rank 2 in ``_RANK`` while ``disputed`` is
    rank 3, so a broken GATE reported LESS severe than a pair needing
    adjudication. The case is reachable: a gate is commonly a GEval at threshold
    0.5, and a stock judge metric in the same run can saturate to 100% — judges
    saturate.
    """

    def _gate_saturated(self, n: int = 200) -> list[dict]:
        """GATE passes 199/200 (99.5%); DIAG passes 140/200 (70%)."""
        return [_case(i, (GATE, i != 0), (DIAG, i >= 60)) for i in range(n)]

    def _diagnostic_saturated(self, n: int = 200) -> list[dict]:
        """GATE passes 140/200 (70%); DIAG passes 199/200 (99.5%)."""
        return [_case(i, (GATE, i >= 60), (DIAG, i != 0)) for i in range(n)]

    def test_saturated_gate_is_disputed_at_the_metric_level(self):
        audit = analyze_run(
            [_run("m1", self._gate_saturated())], gating_metrics={GATE}
        )
        entry = audit["metrics"][GATE]
        assert entry["non_discriminating"] is True, "still measured as degenerate"
        assert entry["verdict"] == "disputed", (
            "a saturated GATE ranked below a disputed pair, so the most serious "
            "thing the detector can find reported as the milder of the two"
        )
        assert entry["severity"] == "error"
        assert audit["worst_verdict"] == "disputed"

    def test_saturated_diagnostic_is_not_escalated(self):
        audit = analyze_run(
            [_run("m1", self._diagnostic_saturated())], gating_metrics={GATE}
        )
        assert audit["metrics"][DIAG]["verdict"] == "non_discriminating"
        assert audit["metrics"][DIAG]["severity"] == "warning"
        assert audit["metrics"][GATE]["verdict"] == "ok", "the gate discriminates"
        assert audit["worst_verdict"] == "non_discriminating"

    def test_the_two_runs_are_no_longer_reported_identically(self):
        """The regression stated as one assertion, since that was the whole bug."""
        a = analyze_run([_run("m1", self._gate_saturated())],
                        gating_metrics={GATE})
        b = analyze_run([_run("m1", self._diagnostic_saturated())],
                        gating_metrics={GATE})
        assert a["worst_verdict"] != b["worst_verdict"], (
            "a broken gate and a useless display column produced the same verdict"
        )

    def test_finding_names_the_consequence_for_pass_rates_and_convergence(self):
        """"Fix or drop it" is not actionable for the metric that gates."""
        audit = analyze_run(
            [_run("m1", self._gate_saturated())], gating_metrics={GATE}
        )
        text = " ".join(audit["metrics"][GATE]["findings"])
        assert "OVERSTATED" in text
        assert "converg" in text
        assert "culling" in text, (
            "models are culled on these pass rates, so say so"
        )

    def test_disagreeing_witness_is_named(self):
        audit = analyze_run(
            [_run("m1", self._gate_saturated())], gating_metrics={GATE}
        )
        text = " ".join(audit["metrics"][GATE]["findings"])
        assert DIAG in text and "29.5%" in text, (
            "the corroborating metric and its disagreement rate are the evidence"
        )

    def test_stuck_failing_gate_also_escalates(self):
        """The other direction ruins the run too: nothing can ever pass.

        Less corrupting than stuck-passing (no false success) but it drives every
        pass rate to ~0, so the loop cannot converge and any model culled on those
        numbers was culled on the metric rather than on its output.
        """
        cases = [_case(i, (GATE, i == 0), (DIAG, i >= 60)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        entry = audit["metrics"][GATE]
        assert entry["verdict"] == "disputed"
        assert entry["severity"] == "error"
        assert "cannot converge" in " ".join(entry["findings"])

    def test_capped_pair_does_not_tell_you_to_drop_the_gate(self):
        """The pair cap still fires; only its remedy changes for a gate."""
        cases = [_case(i, (GATE, i == 0), (DIAG, i >= 60)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        cap = next(f for f in audit["pairs"][0]["findings"]
                   if f.startswith("SEVERITY CAPPED"))
        assert "cannot simply be dropped" in cap
        assert "Fix or drop" not in cap, (
            "the gate decides case success; dropping it is not an option"
        )


class TestSaturatedGateEscalationRequiresCorroboration:
    """Without a discriminating witness there is no evidence, so no alarm.

    A gate passing 99% of a genuinely solved task is the GOAL of this product, not
    a defect. Escalating on the pass rate alone would report every converged run
    as fiction, and an audit that cries wolf on success is one users stop reading
    — the same false-positive failure that produced the pair-level cap.
    """

    def test_agreeing_metrics_do_not_escalate_a_high_pass_rate(self):
        # Gate 99.5%, diagnostic 99.0%: both degenerate, and they AGREE.
        cases = [_case(i, (GATE, i != 0), (DIAG, i > 1)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert audit["metrics"][GATE]["verdict"] == "non_discriminating", (
            "a solved task must not be reported as a broken gate"
        )
        assert audit["worst_verdict"] == "non_discriminating"

    def test_a_degenerate_partner_cannot_corroborate(self):
        """Symmetry with the pair cap: an uninformative metric is not evidence.

        Here the partner disagrees on 99.5% of the gate's passes — a huge ratio —
        but only because it fails everything. Counting it would make any
        stuck-failing diagnostic condemn its own gate.
        """
        cases = [_case(i, (GATE, i != 0), (DIAG, False)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert audit["pairs"][0]["a_pass_b_fail_ratio"] > DISPUTE_RATIO
        assert audit["metrics"][GATE]["verdict"] == "non_discriminating"
        assert audit["metrics"][GATE]["severity"] == "warning"

    def test_lone_saturated_gate_is_not_escalated(self):
        """One metric means no cross-metric evidence exists in either direction."""
        cases = [_case(i, (GATE, i != 0)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
        assert audit["pairs"] == []
        assert audit["metrics"][GATE]["verdict"] == "non_discriminating"

    def test_thin_pair_cannot_corroborate(self):
        """A witness that scored only a handful of cases is noise, not evidence.

        Reachable because ``_verdict_of`` drops a metric PER CASE when it did not
        score: a judge that errored on most cases leaves a fully-scored gate
        paired against a 10-case witness. Here that witness disagrees on 90% of
        the overlap, which would escalate the gate on 9 cases of noise, while the
        gate's own 200 cases are enough to call it saturated.
        """
        cases = []
        for i in range(200):
            c = _case(i, (GATE, i != 0), (DIAG, False))
            if i >= 10:
                c["metrics_data"][1]["score"] = None  # the witness errored here
            cases.append(c)
        audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})

        assert audit["metrics"][GATE]["non_discriminating"] is True, "premise"
        assert audit["metrics"][DIAG]["cases_scored"] == 10, "premise: thin witness"
        pair = audit["pairs"][0]
        assert pair["verdict"] == "insufficient_data"
        assert pair["a_pass_b_fail_ratio"] > DISPUTE_RATIO, (
            "premise: the thin ratio is large enough to escalate if counted"
        )
        assert audit["metrics"][GATE]["verdict"] == "non_discriminating", (
            "9 cases of overlap cannot condemn a metric scored on 200"
        )

    def test_no_gating_set_declared_means_no_escalation(self):
        """With no declared gate the module does not know what decides success."""
        cases = [_case(i, (GATE, i != 0), (DIAG, i >= 60)) for i in range(200)]
        audit = analyze_run([_run("m1", cases)], gating_metrics=set())
        assert audit["metrics"][GATE]["verdict"] == "non_discriminating"


def test_disputed_log_names_the_metric_when_no_pair_is_disputed(tmp_path):
    """The escalation would otherwise log an EMPTY "metrics DISPUTED" line.

    A saturated gate is disputed at the METRIC level while its pairs are held at
    "review" by the cap, and the log built its message only from disputed PAIRS.
    The log is where an operator reads the audit, so an empty findings list there
    hides the escalation.
    """
    import json as _json
    from unittest.mock import patch

    from app.services import metric_disagreement as md

    cases = [_case(i, (GATE, i != 0), (DIAG, i >= 60)) for i in range(200)]
    run = tmp_path / "eval_results_y.json"
    run.write_text(_json.dumps([_run("m1", cases)]))

    with patch.object(md, "logger") as mock_logger:
        audit = md.audit_run_file(run, {GATE})

    assert audit["worst_verdict"] == "disputed"
    assert not any(p["verdict"] == "disputed" for p in audit["pairs"]), (
        "premise: the pair is capped, so only the metric is disputed"
    )
    warned = " ".join(str(c.args) for c in mock_logger.warning.call_args_list)
    assert "DISPUTED" in warned
    assert GATE in warned.split("DISPUTED", 1)[1], (
        "the DISPUTED line printed an empty findings list"
    )


def test_over_strict_finding_offers_both_readings():
    """It must not assert the stricter metric costs nothing in accuracy.

    If the stricter metric is right, the GATE is passing wrong answers and the
    reported pass rate is overstated. Which reading holds is a domain judgement, so
    the finding states both instead of picking one.
    """
    cases = [_case(i, (GATE, i >= 10), (DIAG, i >= 40)) for i in range(100)]
    audit = analyze_run([_run("m1", cases)], gating_metrics={GATE})
    text = " ".join(audit["pairs"][0]["findings"])
    assert "OVERSTATED" in text, "must warn the pass rate may be overstated"
    assert "Costs nothing in accuracy" not in text


def test_non_discriminating_metric_warns_even_when_a_pair_outranks_it(tmp_path):
    """"Delete this useless metric" must not be buried under a pair verdict.

    A metric that passes ~99-100% of cases across models the gate separates widely
    would otherwise be reported only via the over-lenient framing, because the log
    printed the worst verdict's evidence and "disputed" outranks
    "non_discriminating". The degeneracy is the more actionable of the two.

    Asserts on the module logger rather than ``caplog``: caplog is order-dependent
    in this suite (it can be EMPTY in a full run while the test passes alone), so
    patching the logger is the only deterministic form.
    """
    import json as _json
    from unittest.mock import patch

    from app.services import metric_disagreement as md

    # Gate discriminates; LENIENT passes all but 2 ⇒ degenerate. Both verdicts present.
    cases = [_case(i, (GATE, i % 4 != 0), ("Lenient", i >= 2)) for i in range(600)]
    run = tmp_path / "eval_results_x.json"
    run.write_text(_json.dumps([_run("m1", cases)]))

    with patch.object(md, "logger") as mock_logger:
        audit = md.audit_run_file(run, {GATE})

    assert audit["metrics"]["Lenient"]["verdict"] == "non_discriminating"
    warned = " ".join(
        str(c.args) for c in mock_logger.warning.call_args_list
    )
    assert "NON-DISCRIMINATING" in warned, (
        "the degeneracy must warn on its own, not only when it is the worst verdict"
    )
    assert "Lenient" in warned, "must name the offending metric"
