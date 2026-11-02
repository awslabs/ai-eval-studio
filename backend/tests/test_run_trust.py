# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single run-trust verdict every surface reads.

The properties that matter are not "does it copy fields" but the ones that were
got wrong on the way here:

1. Severity follows the audit's DIRECTION classification, never the largest
   ratio on the page. The first version of the translation printed "disagree on
   21% of cases" for a pair whose severity came from a single over-lenient case —
   reintroducing the magnitude-over-direction error the audit exists to prevent.
2. Nothing is re-derived from the audit's prose. The verdict reads structured
   ``signals``; if it parsed ``findings`` text a reworded finding would silently
   change what the user sees.
3. A missing audit is "nothing known", never "nothing wrong". Runs that finished
   before the audit existed must not render an all-clear.
4. A capped signal cannot escalate. The audit downgrades signals when a
   degenerate metric explains the disagreement, and this must respect that rather
   than re-reading the raw ratios.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.metric_disagreement import analyze_run
from app.services.run_trust import SEVERITY_RANK, build_run_trust

GATE = "Judge"
DIAG = "StrictMatch"


@dataclass
class Integrity:
    """Stand-in for ``RunIntegrityFacts`` (duck-typed by ``build_run_trust``)."""

    case_count: int = 100
    errored: int = 0
    dropped: int = 0
    null_count: int = 0
    unrecovered: int = 0
    effective_total: int = 100
    reliability: str = "ok"


def _case(idx: int, *verdicts: tuple[str, bool]) -> dict:
    return {
        "name": f"c{idx}",
        "success": verdicts[0][1],
        "input": "in",
        "actual_output": "out",
        "expected_output": "out",
        "additional_metadata": {"case_index": idx},
        "metrics_data": [
            {"name": n, "score": 1.0 if ok else 0.0, "success": ok,
             "threshold": 0.5, "reason": "r", "error": None}
            for n, ok in verdicts
        ],
    }


def _audit(cases: list[dict], gating=frozenset({GATE})) -> dict:
    return analyze_run([{"model_id": "m1", "results": cases}], gating_metrics=gating)


def _ids(trust) -> list[str]:
    return [r.id for r in trust.reasons]


class TestCleanState:
    def test_no_inputs_is_an_explicit_all_clear(self):
        trust = build_run_trust()
        assert trust.severity == "ok"
        assert trust.reasons == []

    def test_clean_integrity_and_clean_audit_stays_ok(self):
        cases = [_case(i, (GATE, i >= 10), (DIAG, i >= 12)) for i in range(100)]
        trust = build_run_trust(
            run_integrity=Integrity(), metric_audit=_audit(cases), case_count=100,
        )
        assert trust.severity == "ok", _ids(trust)

    def test_missing_audit_is_not_an_all_clear_about_metrics(self):
        """A run whose audit did not run must not claim its metrics are fine."""
        trust = build_run_trust(run_integrity=Integrity(), metric_audit=None)
        assert trust.severity == "ok"
        assert trust.reasons == [], "no metric reason may be invented from absence"
        assert trust.audit_available is False, (
            "severity 'ok' with no reasons is indistinguishable from a clean run "
            "unless the unchecked state is recorded explicitly"
        )


class TestAuditAvailability:
    """"Checked and clean" must be distinguishable from "not checked".

    Recorded as a fact about the run rather than left to each reader to infer
    from an absent verdict, because the obvious inference — no news is good news
    — is the wrong one, and it is the same ambiguity the audit itself sat in
    while nothing read its sidecar.
    """

    def test_true_when_the_audit_ran(self):
        cases = [_case(i, (GATE, i >= 10), (DIAG, i >= 12)) for i in range(100)]
        assert build_run_trust(metric_audit=_audit(cases)).audit_available is True

    def test_false_when_there_is_no_audit(self):
        assert build_run_trust(metric_audit=None).audit_available is False

    def test_false_when_the_audit_gave_up(self):
        """``not_analyzable`` means it ran but concluded nothing — still unchecked."""
        cases = [_case(i, (GATE, True)) for i in range(3)]
        audit = analyze_run("garbage", gating_metrics={GATE})  # type: ignore[arg-type]
        assert audit["status"] == "not_analyzable", "premise"
        assert build_run_trust(metric_audit=audit).audit_available is False
        assert build_run_trust(metric_audit=_audit(cases)).audit_available is True, (
            "a thin-but-real audit still counts as having run"
        )


class TestIntegrityReasons:
    def test_dropped_cases_report_the_honest_denominator(self):
        trust = build_run_trust(
            run_integrity=Integrity(errored=3, dropped=1, effective_total=96),
            case_count=100,
        )
        assert _ids(trust) == ["cases_not_scored"]
        assert trust.severity == "warning"
        assert "96 of 100" in trust.reasons[0].headline

    def test_bad_judge_reliability_is_an_error(self):
        trust = build_run_trust(run_integrity=Integrity(reliability="bad"))
        assert trust.severity == "error"
        assert "judge_unreliable" in _ids(trust)

    def test_degraded_judge_reliability_is_a_warning(self):
        trust = build_run_trust(run_integrity=Integrity(reliability="degraded"))
        assert trust.severity == "warning"
        assert "judge_degraded" in _ids(trust)


class TestDirectionOutranksMagnitude:
    """The property the first translation broke."""

    def _both_directions(self) -> list[dict]:
        # Gate fails 0-9 and case 500; DIAG fails 0-39 but PASSES case 500.
        # => over-strict on ~5% of cases, over-lenient on exactly 1.
        return [
            _case(i, (GATE, i >= 10 and i != 500), (DIAG, i >= 40 or i == 500))
            for i in range(600)
        ]

    def test_the_single_over_lenient_case_outranks_the_bulk_over_strict(self):
        trust = build_run_trust(metric_audit=_audit(self._both_directions()))
        assert {"gate_rejects_accepted", "over_strict"} <= set(_ids(trust))
        worst = trust.reasons[0]
        assert worst.id == "gate_rejects_accepted", (
            "the bulk over-strict finding outranked the single case that made the "
            "pair disputed — magnitude beat direction"
        )
        assert trust.severity == "warning"

    def test_the_rare_direction_is_described_by_COUNT_not_percentage(self):
        """"1 case" is actionable; "0%" reads as nothing at all."""
        trust = build_run_trust(metric_audit=_audit(self._both_directions()))
        headline = trust.reasons[0].headline
        assert "1 case" in headline
        assert "0%" not in headline

    def test_both_directions_are_reported_not_just_the_worst(self):
        trust = build_run_trust(metric_audit=_audit(self._both_directions()))
        assert len(trust.reasons) >= 2, (
            "the over-strict direction is the dominant one on this run and must "
            "not be silenced by the rarer, more severe one"
        )

    def test_reasons_are_ordered_worst_first(self):
        """Insertion order must NOT already be worst-first, or this proves nothing.

        Integrity reasons are collected before audit reasons, so a run with a
        WARNING from integrity and an ERROR from a saturated gate is inserted
        mildest-first — which is the only arrangement that catches a missing sort.
        """
        saturated = [_case(i, (GATE, i != 0), (DIAG, i >= 60)) for i in range(200)]
        trust = build_run_trust(
            run_integrity=Integrity(errored=2, effective_total=98),
            metric_audit=_audit(saturated),
        )
        severities = [r.severity for r in trust.reasons]
        assert "warning" in severities and "error" in severities, "premise"
        assert severities.index("error") < severities.index("warning"), (
            "an integrity warning was listed above a broken gate — a renderer "
            "that truncates would drop the reason that matters"
        )
        ranks = [SEVERITY_RANK[s] for s in severities]
        assert ranks == sorted(ranks, reverse=True)

    def test_a_systematic_disagreement_carries_the_audits_error_severity(self):
        """Severity comes from the signal, not from a fixed value per kind.

        The gate rejects 25% of what the diagnostic accepts — above the audit's
        floor, so the audit rates it ``error`` rather than ``warning``. Reading
        that through is what keeps one scale end to end.
        """
        # Gate fails every 4th case; DIAG passes all but 2 of the gate's failures
        # ... but DIAG must discriminate, so it also fails a distinct block.
        cases = [_case(i, (GATE, i % 4 != 0), (DIAG, i < 400))
                 for i in range(600)]
        audit = _audit(cases)
        assert audit["pairs"][0]["severity"] == "error", "premise: audit says error"
        trust = build_run_trust(metric_audit=audit)
        assert trust.severity == "error"
        assert trust.reasons[0].id == "gate_rejects_accepted"

    def test_the_over_strict_direction_stays_informational(self):
        """It must not be promoted to a warning just for being large."""
        trust = build_run_trust(metric_audit=_audit(self._both_directions()))
        over_strict = next(r for r in trust.reasons if r.id == "over_strict")
        assert over_strict.severity == "info"


class TestSaturatedGate:
    def test_saturated_gate_is_the_error_case_and_names_the_stakes(self):
        # Gate passes 199/200; DIAG discriminates and disagrees => escalated.
        cases = [_case(i, (GATE, i != 0), (DIAG, i >= 60)) for i in range(200)]
        trust = build_run_trust(metric_audit=_audit(cases))
        gate_reason = next(r for r in trust.reasons if r.id == "gate_saturated")
        assert gate_reason.severity == "error"
        assert trust.severity == "error"
        assert gate_reason.metrics == [GATE], "a renderer must be able to mark the column"
        assert "unproven" in gate_reason.impact

    def test_saturated_diagnostic_is_only_a_warning(self):
        cases = [_case(i, (GATE, i >= 60), (DIAG, i != 0)) for i in range(200)]
        trust = build_run_trust(metric_audit=_audit(cases))
        assert "metric_non_discriminating" in _ids(trust)
        assert "gate_saturated" not in _ids(trust)
        assert trust.severity == "warning"

    def test_only_a_GATING_metric_can_raise_the_gate_saturated_reason(self):
        """Guards the claim "decides pass/fail", which would be false otherwise.

        Hand-built because the audit only ever marks a GATE disputed, so this
        state is unreachable through ``analyze_run`` today. The guard is defensive
        against that invariant changing, and an untested guard is a guard that
        quietly stops working.
        """
        audit = {
            "status": "ok",
            "gating_metrics": [GATE],
            "metrics": {
                DIAG: {"role": "diagnostic", "verdict": "disputed",
                       "severity": "error", "pass_ratio": 0.995,
                       "non_discriminating": True, "findings": []},
            },
            "pairs": [],
        }
        trust = build_run_trust(metric_audit=audit)
        assert "gate_saturated" not in _ids(trust), (
            "a diagnostic was described as deciding pass/fail"
        )

    def test_a_useless_metric_reason_names_that_metric(self):
        cases = [_case(i, (GATE, i >= 60), (DIAG, i != 0)) for i in range(200)]
        trust = build_run_trust(metric_audit=_audit(cases))
        reason = next(r for r in trust.reasons
                      if r.id == "metric_non_discriminating")
        assert reason.metrics == [DIAG]


class TestCappedSignalsCannotEscalate:
    """The audit's false-alarm cap must survive translation.

    A degenerate metric's disagreements are explained by its own degeneracy, so
    the audit downgrades those signals. Re-reading the raw ratios here would
    re-raise exactly the alarm the cap suppresses.
    """

    def _gate_vs_lenient(self) -> list[dict]:
        # Lenient passes all but 2 => degenerate; the gate fails 25%.
        return [_case(i, (GATE, i % 4 != 0), ("Relevancy", i >= 2))
                for i in range(600)]

    def test_capped_pair_does_not_produce_an_error(self):
        audit = _audit(self._gate_vs_lenient())
        assert audit["pairs"][0]["verdict"] == "review", "premise: capped"
        trust = build_run_trust(metric_audit=audit)
        assert all(r.severity != "error" for r in trust.reasons), _ids(trust)

    def test_the_degenerate_metric_is_still_reported(self):
        trust = build_run_trust(metric_audit=_audit(self._gate_vs_lenient()))
        assert "metric_non_discriminating" in _ids(trust)


class TestRobustness:
    def test_garbage_audit_yields_no_reasons_and_no_exception(self):
        for junk in (None, {}, [], "nope", 7, {"status": "not_analyzable"},
                     {"status": "ok", "metrics": "bad", "pairs": "bad"}):
            trust = build_run_trust(metric_audit=junk)  # type: ignore[arg-type]
            assert trust.severity == "ok"
            assert trust.reasons == []

    def test_an_unanalyzable_audit_is_ignored_even_when_it_carries_content(self):
        """The status field is the gate, not the emptiness of the payload.

        An audit that gave up mid-way can still hold partial metrics — reporting
        those to a user as findings would put numbers on screen that the audit
        itself declined to stand behind.
        """
        audit = {
            "status": "not_analyzable",
            "gating_metrics": [GATE],
            "metrics": {
                GATE: {"role": "gating", "verdict": "non_discriminating",
                       "severity": "warning", "pass_ratio": 0.999,
                       "non_discriminating": True, "findings": []},
            },
            "pairs": [{"metric_a": GATE, "metric_b": DIAG, "verdict": "disputed",
                       "severity": "error", "signals": [
                           {"kind": "gate_rejects_accepted", "severity": "error",
                            "gate": GATE, "other": DIAG, "ratio": 0.4,
                            "count": 99}]}],
        }
        trust = build_run_trust(metric_audit=audit)
        assert trust.reasons == []
        assert trust.severity == "ok"

    def test_missing_integrity_attributes_do_not_raise(self):
        class Sparse:
            pass

        trust = build_run_trust(run_integrity=Sparse(), case_count=10)
        assert trust.severity == "ok"

    def test_unknown_signal_kind_is_skipped_not_rendered_blank(self):
        """A future audit signal must not produce an empty, meaningless card."""
        audit = {
            "status": "ok", "metrics": {},
            "pairs": [{"signals": [{"kind": "brand_new", "severity": "error"}]}],
        }
        trust = build_run_trust(metric_audit=audit)
        assert trust.reasons == []
        assert trust.severity == "ok"

    def test_insufficient_data_pairs_contribute_nothing(self):
        cases = [_case(i, (GATE, True), (DIAG, False)) for i in range(5)]
        trust = build_run_trust(metric_audit=_audit(cases))
        assert trust.reasons == []
