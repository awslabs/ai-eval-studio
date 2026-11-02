# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One verdict on whether a run's numbers can be trusted.

WHY THIS EXISTS. Seven separate mechanisms in this codebase answer some part of
"can I believe these pass rates", and before this module each had invented its own
output channel: the harness's ``data_quality`` block was a card at the BOTTOM of
the run report (below the latency tables — a caveat about the numbers rendered
after all the numbers), cross-model fault attribution was three tiles, judge
unreliability halted the loop, the gating-unconfigured warning was a stderr line,
and the metric audit wrote a sidecar file that nothing read. Two of the seven were
invisible in the product, and the same integrity card was duplicated in both
report renderers.

The result was that every new check had to invent a placement, and adding one more
badge somewhere was never going to fix it. This module is the missing middle: the
detectors stay separate — they measure genuinely different things and collapsing
them would be wrong — but they all report through ONE severity scale and ONE list
of reasons, which every surface renders.

WHAT MUST STAY SINGLE, AND WHAT MUST NOT:

- **The verdict is single.** One severity, one reason list, one vocabulary. The
  report cards, the run-list marker and the agent's tool response all read this,
  so a reason cannot appear on one surface and not another, and cannot be worded
  two ways.
- **The detectors are NOT single, deliberately.** Dropped-case accounting counts
  what never got scored; fault attribution compares models against ground truth
  on cases everything failed; the metric audit compares metrics against each
  other across all cases. Measured on six real runs, fault attribution's
  "likely scoring errors" bucket fired ZERO times while the metric audit found
  221 over-strict cases in one of those same runs — because the bucket only looks
  at all-models-failed cases and requires byte equality with the expected output.
  They are different instruments aimed at one question, not copies.

SEVERITY VOCABULARY is borrowed from ``metric_disagreement`` rather than reinvented
(``info`` / ``warning`` / ``error``), so a reason keeps its severity end to end.

NEVER BLOCKS. Like the audit it draws from, this is commentary on results that
already exist. A missing or malformed input yields fewer reasons, never an
exception and never a failed run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Ordering for "worst wins". Matches ``metric_disagreement._SEVERITY_RANK``.
SEVERITY_RANK = {"ok": 0, "info": 1, "warning": 2, "error": 3}


@dataclass
class TrustReason:
    """One reason to doubt a run, in the shape every surface renders.

    ``headline`` says what is wrong in the user's terms. ``impact`` says what it
    means for the numbers on the page, which is the part a user actually needs
    and the part a bare log line never gave them.

    ``metrics`` names every metric the reason implicates, so a renderer marks
    those columns without re-deriving anything from the text. It is a LIST
    because two diagnostics contradicting each other implicate both and neither
    can be singled out — attributing that reason to one of them would tell the
    user the other column is sound. Empty for run-wide reasons (dropped cases,
    judge reliability), which belong to no single metric.
    """

    id: str
    severity: str
    headline: str
    impact: str
    metrics: list[str] = field(default_factory=list)


@dataclass
class RunTrustFacts:
    """A run's trust verdict: the worst severity plus every reason found.

    ``audit_available`` records whether the metric audit actually ran for this
    run, and exists so "checked and clean" is never confused with "not checked".
    That distinction is a FACT ABOUT THE RUN, not a missing field: an absent
    verdict would have to be interpreted by each reader, and the obvious
    interpretation — no news is good news — is the wrong one. A run whose audit
    failed to write still reports its integrity reasons; it just cannot claim its
    metrics agree.
    """

    severity: str = "ok"
    reasons: list[TrustReason] = field(default_factory=list)
    audit_available: bool = False

    def add(self, reason: TrustReason) -> None:
        """Append a reason and raise the overall severity, never lowering it."""
        self.reasons.append(reason)
        if SEVERITY_RANK.get(reason.severity, 0) > SEVERITY_RANK.get(self.severity, 0):
            self.severity = reason.severity


def _integrity_reasons(trust: RunTrustFacts, integrity: Any, case_count: int) -> None:
    """Cases that never got scored, so the denominators are not the dataset.

    Kept as its own reason rather than folded into the metric reasons: a run can
    have perfect metrics over a fraction of the data, and the user needs to know
    the fraction before comparing anything.
    """
    if integrity is None:
        return
    errored = getattr(integrity, "errored", 0) or 0
    dropped = getattr(integrity, "dropped", 0) or 0
    unrecovered = getattr(integrity, "unrecovered", 0) or 0
    missing = errored + dropped + unrecovered
    if missing > 0:
        total = getattr(integrity, "case_count", 0) or case_count
        effective = getattr(integrity, "effective_total", 0) or max(0, total - missing)
        # Keep the per-cause breakdown the report already showed: "3 errored"
        # and "3 dropped" point at different problems (the judge failing versus
        # the harness discarding a case), so collapsing them to one number
        # would lose the only clue about which.
        breakdown = ", ".join(
            f"{n} {label}" for n, label in (
                (errored, "errored"), (dropped, "dropped"),
                (unrecovered, "unrecovered"),
            ) if n
        )
        trust.add(TrustReason(
            id="cases_not_scored",
            severity="warning",
            headline=f"Pass rates cover {effective} of {total} cases",
            impact=(
                f"{missing} case(s) never produced a usable score ({breakdown}), "
                "so these percentages are over a smaller set than your dataset. "
                "Comparisons between models are still fair; comparisons against a "
                "target score are not."
            ),
        ))

    # The harness's own reliability label, which it derives from judge null and
    # error ratios after its retry and reconciliation passes.
    reliability = getattr(integrity, "reliability", "ok")
    if reliability == "bad":
        trust.add(TrustReason(
            id="judge_unreliable",
            severity="error",
            headline="The judge failed on too many cases to trust the scores",
            impact=(
                "Scoring errors survived retry and reconciliation. Treat every "
                "number on this page as provisional until the judge is fixed."
            ),
        ))
    elif reliability == "degraded":
        trust.add(TrustReason(
            id="judge_degraded",
            severity="warning",
            headline="The judge needed retries to score this run",
            impact=(
                "Scores were recovered rather than clean on the first pass, so "
                "small differences between models may be noise."
            ),
        ))


def _audit_reasons(trust: RunTrustFacts, audit: dict[str, Any] | None) -> None:
    """Translate the metric audit into user-facing reasons.

    Reads the audit's structured verdicts only — never its finding text, which is
    written for an operator reading a log. The audit already decided severity by
    metric ROLE and error DIRECTION; this re-words, it does not re-rank.
    """
    if not isinstance(audit, dict) or audit.get("status") != "ok":
        return

    metrics = audit.get("metrics")
    if isinstance(metrics, dict):
        for name, entry in metrics.items():
            if not isinstance(entry, dict):
                continue
            verdict, gating = entry.get("verdict"), entry.get("role") == "gating"
            pass_ratio = entry.get("pass_ratio")
            if not isinstance(pass_ratio, (int, float)):
                continue
            same = max(pass_ratio, 1 - pass_ratio)
            if verdict == "disputed" and gating:
                # The serious one: case success is decided by a metric that
                # answers the same way regardless of what the model produced.
                trust.add(TrustReason(
                    id="gate_saturated",
                    severity="error",
                    headline=(
                        f"{name} decides pass/fail but scores "
                        f"{same:.0%} of cases the same"
                    ),
                    impact=(
                        "Every pass rate on this page comes from that metric, and "
                        "another metric that does separate your models disagrees "
                        "with it. Treat the percentages — and any decision to "
                        "stop optimizing or drop a model — as unproven."
                    ),
                    metrics=[name],
                ))
            elif verdict == "non_discriminating":
                trust.add(TrustReason(
                    id="metric_non_discriminating",
                    severity="warning",
                    headline=f"{name} scores {same:.0%} of cases the same",
                    impact=(
                        "It cannot separate a good model from a bad one, so its "
                        "column tells you nothing when comparing models."
                    ),
                    metrics=[name],
                ))

    for pair in audit.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        for signal in pair.get("signals") or []:
            if isinstance(signal, dict):
                _signal_reason(trust, signal)


def _signal_reason(trust: RunTrustFacts, signal: dict[str, Any]) -> None:
    """Word one disagreement signal for a user.

    Reads the signal's ``kind`` and ``severity`` rather than picking the largest
    ratio on the pair. That distinction is the whole point: a pair is routinely
    over-strict on 21% of cases AND over-lenient on one, and the single case is
    what makes it serious. Choosing the bigger number to describe reproduces the
    magnitude-over-direction error the audit was built to avoid — which is exactly
    what the first version of this function did.
    """
    kind = signal.get("kind")
    severity = signal.get("severity")
    if severity not in SEVERITY_RANK or severity == "ok":
        return
    gate, other = signal.get("gate"), signal.get("other")
    ratio = signal.get("ratio") or 0.0
    count = signal.get("count") or 0

    if kind == "gate_rejects_accepted":
        cases = f"{count} case(s)" if count < 20 else f"{ratio:.0%} of cases"
        trust.add(TrustReason(
            id="gate_rejects_accepted", severity=severity, metrics=[gate],
            headline=f"{gate} failed {cases} that {other} passed",
            impact=(
                f"{gate} decides pass/fail, so those count as failures while "
                f"{other} says they are fine. Either {gate} is wrong — in which "
                "case your pass rates are understated and a model may have been "
                f"ranked low for the metric's reasons — or {other} is too "
                "lenient, which costs nothing today but would record a wrong "
                "result if it were ever used to skip a more expensive metric."
            ),
        ))
    elif kind == "over_strict":
        trust.add(TrustReason(
            id="over_strict", severity=severity, metrics=[other],
            headline=f"{other} failed {ratio:.0%} of cases that {gate} passed",
            impact=(
                f"Two readings, and only you can tell which: either {other} is "
                f"too strict and its column understates quality, or {gate} is "
                "passing wrong answers and the headline pass rate is overstated. "
                "Read a few of these cases."
            ),
        ))
    elif kind == "both_diagnostic":
        trust.add(TrustReason(
            id="metrics_contradict", severity=severity,
            metrics=[n for n in (gate, other) if n],
            headline=f"{gate} and {other} disagree on {ratio:.0%} of cases",
            impact=(
                "Neither decides pass/fail, so your headline numbers are "
                "unaffected — but one of these two columns is wrong."
            ),
        ))


def format_for_agent(trust: RunTrustFacts | None) -> str:
    """Render the verdict for the coordinator's post-eval message.

    WHY THE AGENT GETS THIS AT ALL. The coordinator is what decides whether to
    cull a model, rewrite the prompt, or declare convergence, and it decided all
    of that from pass rates alone — with no way to know a metric was unreliable,
    because this verdict only ever reached a log line and a sidecar. A badge tells
    the user after the fact; telling the coordinator stops the bad decision.

    The text is deliberately DIRECTIVE, not just descriptive. A list of findings
    invites a summary that mentions them politely and proceeds anyway; the whole
    point is that an ``error`` verdict should change what the agent does next.

    Returns:
        Lines to append to the completion message, or ``""`` when there is
        nothing to report — the caller appends nothing rather than a reassuring
        "no problems found", which would be a claim about a check that may not
        have run.
    """
    if trust is None or trust.severity == "ok" or not trust.reasons:
        return ""

    n = len(trust.reasons)
    lines = [
        f"\nMETRIC TRUST CHECK: {trust.severity.upper()} "
        f"({n} finding{'' if n == 1 else 's'}). These are about the METRICS, not "
        "the models — read them before drawing conclusions from the rates above.",
    ]
    for reason in trust.reasons:
        lines.append(f"  [{reason.severity}] {reason.headline}. {reason.impact}")

    if trust.severity == "error":
        lines.append(
            "  ACTION: do NOT cull a model, declare convergence, or tell the "
            "user which model is best on these numbers. Say plainly that the "
            "metric problem above has to be resolved first, and offer to "
            "investigate it."
        )
    else:
        lines.append(
            "  ACTION: state the affected metric's limitation in your summary "
            "instead of reporting its column as fact, and base any "
            "model comparison on the metrics that do discriminate."
        )
    return "\n".join(lines)


def build_run_trust(
    *,
    run_integrity: Any = None,
    metric_audit: dict[str, Any] | None = None,
    case_count: int = 0,
) -> RunTrustFacts:
    """Collect every reason to doubt a run into one verdict.

    Args:
        run_integrity: The run's :class:`RunIntegrityFacts`, or ``None``.
        metric_audit: A parsed metric-audit sidecar, or ``None`` when the run
            predates the audit or it could not be written.
        case_count: Dataset size, used only as a fallback denominator.

    Returns:
        A :class:`RunTrustFacts` whose ``severity`` is ``"ok"`` and whose
        ``reasons`` are empty when nothing is wrong — the clean state a renderer
        can show as an explicit all-clear rather than as absence.
    """
    trust = RunTrustFacts(
        audit_available=(
            isinstance(metric_audit, dict) and metric_audit.get("status") == "ok"
        ),
    )
    try:
        _integrity_reasons(trust, run_integrity, case_count)
        _audit_reasons(trust, metric_audit)
    except (AttributeError, TypeError, ValueError) as e:  # never break a report
        logger.warning("[run-trust] could not build verdict: %s", e)
    # Worst first, so a renderer that truncates keeps the reason that matters.
    trust.reasons.sort(key=lambda r: -SEVERITY_RANK.get(r.severity, 0))
    return trust
