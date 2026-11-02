# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run detector for metrics that disagree with each other.

WHY THIS EXISTS. ``validate_harness`` proves a harness RUNS; nothing proved its
metrics measure anything sensible. A metric can be perfectly well-formed — a real
float score, a real bool ``success``, a real ``reason`` string — and still return
the wrong verdict on most cases. Harness validation cannot tell the difference,
because at the shape level "correctly failed a bad answer" and "failed a good
answer for a dumb reason" are identical. For example, a normalized exact-match
metric can fail a large share of the cases an LLM judge passes, purely because
the model wraps its output in backticks the normalizer does not strip.

WHAT THE SIGNAL IS. When two metrics score the same case and reach opposite
verdicts, one of them is wrong about that case. That disagreement is already
sitting in every run file, is task-agnostic, and catches artifacts nobody thought
to enumerate — quote style, trailing semicolons, casing, not just code fences.

DELIBERATE NON-GOALS, each of which was a wrong turn we are not taking:

- **No text matching.** We do NOT parse a judge's ``reason`` for words like
  "equivalent". Verdicts are derived only from structured fields
  (``success``/``score``). Reason strings are carried into the output as evidence
  for a human, never branched on. Matching English is how a detector silently
  stops working when a judge rephrases.
- **No artifact knowledge.** Nothing here knows about backticks, query dialects, or exact
  match. A detector keyed on one project's artifact would generalise to
  nothing; the platform generates metrics per project and will see deterministic,
  judge-based, and hybrid scorers over text, JSON, numbers and ranked lists.
- **Never auto-fail a run.** A deliberately stricter metric and a broken one are
  structurally identical; only task knowledge separates them. We flag for human
  adjudication rather than rejecting results.
- **Never test for zero disagreement in either direction.** Zero is not stable:
  two runs of the same configuration can differ by a single contradicting case,
  so a gate keyed on ``== 0`` flips state between them. Everything here
  thresholds on RATIOS.

WHO READS THIS. The sidecar this module writes is consumed by
``run_trust.build_run_trust``, which turns it into the one verdict every surface
renders: the reports' Data quality section and the trust line above the numbers,
and the post-eval coordinator message — the last of which matters most, because
the coordinator decides whether to cull a model or declare convergence. The
sidecar is therefore not mere commentary; it is on the path of a decision, which
is why the executors AWAIT the audit rather than firing it.

SEVERITY FOLLOWS ROLE AND DIRECTION, NOT MAGNITUDE. This is the core rule, and
it is counter-intuitive enough to state plainly: the over-strict direction can
outnumber the over-lenient one by two orders of magnitude, and the rare
over-lenient finding is still the serious one. Over-strict costs money (the
expensive metric runs anyway; the verdict stays correct). Over-lenient
corrupts a verdict. Any ranking that sorts by disagreement COUNT
buries the finding that matters under the one that does not.

The rule applies to SINGLE-metric findings too, which is easy to miss because a
metric's degeneracy is measured without reference to its role. A saturated
diagnostic wastes a report column; a saturated *gate* means every pass rate in
the run came from a metric that answered the same way regardless of the output.
See :func:`_escalate_degenerate_gates`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.metric_name import base_metric_name

logger = logging.getLogger(__name__)

#: Version 2 added per-metric ``severity`` and the structural
#: ``non_discriminating`` flag, and reports a saturated GATING metric as
#: ``disputed`` instead of ``non_discriminating``. It documents the sidecar's
#: shape; ``run_trust`` does not branch on it.
SCHEMA_VERSION = 2

#: Sidecar suffix. The audit is written BESIDE the run file rather than into it:
#: run artifacts can already come within a few KB of the Agent-SDK's 1 MiB
#: (1,048,576-byte) per-message ceiling, so inflating them would push large runs
#: over it.
SIDECAR_SUFFIX = ".metric-audit.json"

#: A metric is called non-discriminating when it returns the same verdict on at
#: least this fraction of cases. Such a metric carries no information regardless
#: of whether it agrees with anything — a metric stuck at always-fail looks
#: identical to a task nobody can do.
NON_DISCRIMINATING_RATIO = 0.99

#: Minimum disagreement ratio before a pair is reported at all. Below this the
#: pair is recorded with verdict "ok" so the UI still sees it was checked.
DISPUTE_RATIO = 0.05

#: Cases needed before ratios mean anything. Under this the audit still runs but
#: every verdict is downgraded to "insufficient_data".
MIN_CASES = 20


def _verdict_of(metric: dict[str, Any]) -> bool | None:
    """Return a metric entry's pass/fail, or ``None`` when it did not score.

    Reads ``success`` only when it is a real bool. A null score or a non-bool
    ``success`` means the metric errored on this case; that is a different
    defect (harness validation already covers it) and must not be counted as a
    disagreement, which would blame a working metric for its partner's error.
    """
    success = metric.get("success")
    if not isinstance(success, bool):
        return None
    if not isinstance(metric.get("score"), (int, float)):
        return None
    return success


def _direction_evidence(
    examples: dict[tuple[tuple[str, str], str], list[dict[str, Any]]],
    a: str,
    b: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Concrete disagreements for a human, sampled PER DIRECTION.

    Sampling the pair's disagreements as one flat list loses the rare direction:
    when hundreds of over-strict cases sit beside a single over-lenient one, any
    head-slice of the combined list is all over-strict and the single case that
    drove the warning never appears. Taking up to ``limit``
    from each direction guarantees the reader sees an example of whatever the
    verdict is actually about.

    Evidence only — nothing in this module branches on these values.
    """
    key = (min(a, b), max(a, b))
    out: list[dict[str, Any]] = []
    for passer in (a, b):
        out.extend(examples.get((key, passer), [])[:limit])
    return out


def analyze_run(
    runs: Any,
    gating_metrics: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Build a metric-disagreement audit for one run artifact.

    Args:
        runs: Parsed run artifact — a list of per-model run dicts, each with a
            ``results`` list of serialized test cases carrying ``metrics_data``.
        gating_metrics: Metric names that decide case success (the harness's
            ``target_metric_scores`` keys). A disagreement involving one of
            these can corrupt pass rates, convergence and every optimization
            decision, so it outranks a disagreement between two diagnostics.
            Base-normalized on the way in: config keys and serialized run-file
            names carry DeepEval's ``[GEval]`` suffix independently, and a
            gate whose two spellings fail to match is silently demoted to
            "diagnostic" — which downgrades exactly the finding that matters
            most.

    Returns:
        Audit dict (see :data:`SCHEMA_VERSION`). Always returns a dict; an
        unusable artifact yields ``status: "not_analyzable"`` rather than
        raising, because this must never be able to fail an eval.
    """
    gating = frozenset(base_metric_name(n) for n in (gating_metrics or ()))
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gating_metrics": sorted(gating),
        "status": "ok",
        "metrics": {},
        "pairs": [],
        "worst_verdict": "ok",
    }

    if not isinstance(runs, list) or not runs:
        audit["status"] = "not_analyzable"
        return audit

    # verdicts[metric_name][(model_id, case_index)] = True/False
    verdicts: dict[str, dict[tuple[str, int], bool]] = {}
    examples: dict[tuple[tuple[str, str], str], list[dict[str, Any]]] = {}

    for run in runs:
        if not isinstance(run, dict):
            continue
        model_id = str(run.get("model_id") or run.get("test_run_id") or "?")
        results = run.get("results")
        if not isinstance(results, list):
            continue
        for idx, case in enumerate(results):
            if not isinstance(case, dict):
                continue
            metrics_data = case.get("metrics_data")
            if not isinstance(metrics_data, list):
                continue
            per_case: dict[str, bool] = {}
            for metric in metrics_data:
                if not isinstance(metric, dict):
                    continue
                raw_name = metric.get("name")
                if not isinstance(raw_name, str) or not raw_name:
                    continue
                # Base-normalized so this audit's metric keys are the SAME
                # name space the reports and the gating config use. Without
                # it a run file carrying DeepEval's ``[GEval]`` suffix
                # produces verdicts under a name no other surface can join
                # against, and every marker built on them silently vanishes.
                name = base_metric_name(raw_name)
                verdict = _verdict_of(metric)
                if verdict is None:
                    continue
                per_case[name] = verdict
                verdicts.setdefault(name, {})[(model_id, idx)] = verdict
            # Record concrete evidence for each disagreeing ordered pair.
            for a, a_pass in per_case.items():
                for b, b_pass in per_case.items():
                    if a >= b or a_pass == b_pass:
                        continue
                    passer, failer = (a, b) if a_pass else (b, a)
                    # Keyed by direction (which metric PASSED), so a rare
                    # direction cannot be crowded out of the evidence sample.
                    key = ((min(a, b), max(a, b)), passer)
                    examples.setdefault(key, []).append({
                        "model_id": model_id,
                        "case_index": idx,
                        "passed": passer,
                        "failed": failer,
                        "actual_output": case.get("actual_output"),
                        "expected_output": case.get("expected_output"),
                        "failed_reason": next(
                            (m.get("reason") for m in metrics_data
                             if isinstance(m, dict) and m.get("name") == failer),
                            None,
                        ),
                    })

    if not verdicts:
        audit["status"] = "not_analyzable"
        return audit

    # ---- per-metric summary (discrimination is role-independent) ----
    for name, by_case in verdicts.items():
        n = len(by_case)
        passes = sum(1 for v in by_case.values() if v)
        pass_ratio = passes / n if n else 0.0
        role = "gating" if name in gating else "diagnostic"
        entry: dict[str, Any] = {
            "role": role,
            "cases_scored": n,
            "passed": passes,
            "pass_ratio": round(pass_ratio, 4),
            "verdict": "ok",
            "severity": "info",
            # Recorded as its own flag, not inferred from ``verdict``: the verdict
            # is escalated for a saturated GATE, and readers still need to know
            # WHY it was escalated without matching on finding text.
            "non_discriminating": False,
            "findings": [],
        }
        if n < MIN_CASES:
            entry["verdict"] = "insufficient_data"
        elif pass_ratio >= NON_DISCRIMINATING_RATIO or pass_ratio <= (
            1 - NON_DISCRIMINATING_RATIO
        ):
            entry["verdict"] = "non_discriminating"
            entry["severity"] = "warning"
            entry["non_discriminating"] = True
            entry["findings"].append(
                f"returns the same verdict on {max(pass_ratio, 1 - pass_ratio):.1%} "
                "of cases — it carries almost no information, whether or not it "
                "agrees with other metrics"
            )
        audit["metrics"][name] = entry

    # Metrics whose own verdict says they carry almost no information. Their
    # disagreements are explained by that degeneracy, so they must not be used as
    # evidence AGAINST another metric — see ``_classify_pair``.
    uninformative = frozenset(
        n for n, e in audit["metrics"].items() if e["non_discriminating"]
    )

    # ---- pairwise disagreement ----
    names = sorted(verdicts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = verdicts[a].keys() & verdicts[b].keys()
            n = len(shared)
            if not n:
                continue
            a_pass_b_fail = sum(
                1 for k in shared if verdicts[a][k] and not verdicts[b][k]
            )
            b_pass_a_fail = sum(
                1 for k in shared if verdicts[b][k] and not verdicts[a][k]
            )
            pair = {
                "metric_a": a,
                "metric_b": b,
                "cases_compared": n,
                "a_pass_b_fail": a_pass_b_fail,
                "b_pass_a_fail": b_pass_a_fail,
                "a_pass_b_fail_ratio": round(a_pass_b_fail / n, 4),
                "b_pass_a_fail_ratio": round(b_pass_a_fail / n, 4),
                "verdict": "ok",
                "severity": "info",
                "findings": [],
                # Machine-readable twin of ``findings``: one entry per finding,
                # carrying the direction that produced it. Consumers read these;
                # ``findings`` is prose for a human reading a log.
                "signals": [],
                "evidence": _direction_evidence(examples, a, b),
            }
            if n < MIN_CASES:
                pair["verdict"] = "insufficient_data"
            else:
                _classify_pair(pair, a, b, gating, uninformative)
            audit["pairs"].append(pair)

    # Role-aware pass over degenerate metrics. Must run AFTER the pairs, because
    # it needs their disagreement ratios as corroboration.
    _escalate_degenerate_gates(audit, gating, uninformative)

    audit["worst_verdict"] = _worst(audit)
    return audit


def _escalate_degenerate_gates(
    audit: dict[str, Any],
    gating: frozenset[str],
    uninformative: frozenset[str],
) -> None:
    """Raise a saturated GATING metric from ``non_discriminating`` to disputed.

    Degeneracy DETECTION is a property of one metric in isolation, so the
    per-metric pass is deliberately role-independent. Degeneracy SEVERITY is not,
    and treating it as though it were contradicted this module's own rule that
    severity follows ROLE and DIRECTION. A saturated diagnostic makes one report
    column useless. A saturated *gating* metric means the number that every pass
    rate, every model comparison and every convergence check is built on came
    from a metric that answered the same way no matter what the model produced —
    yet both landed on ``non_discriminating``, which ranks BELOW a disputed pair.

    Escalation requires CORROBORATION: some metric that does discriminate must
    disagree with the gate, in the direction the gate is saturated toward.
    Without that condition every genuinely converged run would be reported as
    fiction — a gate passing 99% of a solved task is the goal, not a defect — and
    an audit that cries wolf on success is one users learn to ignore. When no
    other metric discriminates (or there is no other metric at all), this module
    has no evidence either way and the honest ceiling stays
    ``non_discriminating``.

    Applies in BOTH directions of saturation, because both destroy the run:
    stuck-passing overstates every pass rate and can converge on nothing;
    stuck-failing drives every pass rate to zero, so the loop cannot converge and
    a model culled on those numbers was culled on the metric, not its output.
    """
    for name, entry in audit["metrics"].items():
        if not entry["non_discriminating"] or name not in gating:
            continue
        saturated_pass = entry["pass_ratio"] >= NON_DISCRIMINATING_RATIO

        corroborators: list[tuple[str, float]] = []
        for pair in audit["pairs"]:
            if pair["verdict"] == "insufficient_data":
                continue
            a, b = pair["metric_a"], pair["metric_b"]
            if name not in (a, b):
                continue
            other = b if name == a else a
            if other in uninformative:
                continue  # one uninformative metric cannot corroborate another
            # Orientation is the easy thing to invert here: "a_pass_b_fail" is
            # metric_a passing while metric_b fails. For a stuck-PASSING gate we
            # want the partner FAILING what the gate passed; for a stuck-FAILING
            # gate, the partner PASSING what the gate failed. Both are "the gate
            # is on its saturated side and the partner is not".
            gate_is_a = name == a
            if saturated_pass:
                ratio = pair["a_pass_b_fail_ratio" if gate_is_a
                              else "b_pass_a_fail_ratio"]
            else:
                ratio = pair["b_pass_a_fail_ratio" if gate_is_a
                              else "a_pass_b_fail_ratio"]
            if ratio >= DISPUTE_RATIO:
                corroborators.append((other, ratio))

        if not corroborators:
            continue

        entry["verdict"] = "disputed"
        entry["severity"] = "error"
        witnesses = "; ".join(
            f"{o!r} {'fails' if saturated_pass else 'passes'} {r:.1%} of them"
            for o, r in sorted(corroborators)
        )
        if saturated_pass:
            entry["findings"].append(
                f"GATING metric {name!r} passes {entry['pass_ratio']:.1%} of cases, "
                f"and metrics that DO discriminate disagree: {witnesses}. Case "
                f"success is decided by {name!r}, so if it is saturated rather "
                "than the task being solved, every reported pass rate is "
                "OVERSTATED and any convergence against a target score was "
                "reached on a metric that cannot tell a good answer from a bad "
                "one. Read a sample of the disagreeing cases before trusting "
                "these numbers or culling any model on them."
            )
        else:
            entry["findings"].append(
                f"GATING metric {name!r} fails "
                f"{1 - entry['pass_ratio']:.1%} of cases, and metrics that DO "
                f"discriminate disagree: {witnesses}. Case success is decided by "
                f"{name!r}, so near-zero pass rates may be an artifact of the "
                "metric rather than of model quality: the loop cannot converge, "
                "and every model looks equally bad regardless of what it "
                "produced. Fix this before comparing or culling models."
            )


def _classify_pair(
    pair: dict[str, Any],
    a: str,
    b: str,
    gating: frozenset[str],
    uninformative: frozenset[str] = frozenset(),
) -> None:
    """Assign a pair's verdict + severity from ROLE and DIRECTION, not count.

    Both directions are thresholded on a ratio; neither is tested against zero
    (see the module docstring). Each direction is judged on what it would cost
    if that metric were trusted:

    - A GATING metric failing what another metric passes can corrupt pass rates
      and every downstream optimization decision — the highest severity here.
    - A NON-GATING metric that PASSES what a gating metric FAILS is the
      over-lenient direction. It is only a display problem today, but it is the
      shape that silently records a wrong verdict the moment such a metric is
      used to short-circuit an expensive one, so it is ranked ABOVE the
      over-strict direction even when far rarer.
    - A NON-GATING metric that merely fails more than a gating metric is
      over-strict: wasteful and misleading in the UI, not verdict-corrupting.
    """
    a_gating, b_gating = a in gating, b in gating
    ra, rb = pair["a_pass_b_fail_ratio"], pair["b_pass_a_fail_ratio"]

    def note(
        sev: str,
        verdict: str,
        msg: str,
        *,
        kind: str,
        gate: str | None = None,
        other: str | None = None,
        ratio: float = 0.0,
        count: int = 0,
    ) -> None:
        """Record a finding; escalate severity/verdict but never lower them.

        **Findings are additive evidence, not exclusive verdicts.** Both
        directions of a disagreement are frequently true at once — many
        over-strict cases AND a single over-lenient one on the same pair — and
        guarding each branch on ``verdict == "ok"`` would let whichever branch
        ran first silence the rest: one over-lenient case would suppress the
        large over-strict finding this detector exists to report.

        Ranking still lives in the severity/verdict fields, which only ever move
        up, so an ``info`` finding appended after a ``warning`` cannot demote the
        pair.

        Every finding also records a machine-readable ``signal``. ``findings`` is
        prose for an operator reading a log; a caller that needs to know WHICH
        direction drove the verdict must not have to parse it. The first consumer
        did exactly that and got it wrong — it reported the larger ratio on a pair
        whose severity came from the rarer direction, reintroducing the
        magnitude-over-direction error this module exists to prevent.
        """
        pair["findings"].append(msg)
        pair["signals"].append({
            "kind": kind,
            "severity": sev,
            "gate": gate,
            "other": other,
            "ratio": round(ratio, 4),
            "count": count,
        })
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(pair["severity"], 0):
            pair["severity"] = sev
        if _RANK.get(verdict, 0) > _RANK.get(pair["verdict"], 0):
            pair["verdict"] = verdict

    # ---- The gate REJECTED what another metric ACCEPTED ----
    # One direction, ONE finding, two readings — the same treatment the
    # over-strict direction below already gets. This used to be two separate
    # branches ("the gate is wrong" at error, "the other metric is over-lenient"
    # at warning) that both fired on the identical set of cases, so one
    # disagreement was reported as if it were two problems. It is one
    # disagreement whose blame is undecidable from structure alone.
    #
    # Severity still follows DIRECTION, not count: any single case where the gate
    # rejects what another metric accepts is reported (no floor — one case is
    # enough), and crossing the ratio floor raises it to error because
    # a systematic split is more likely to be the gate than an edge case.
    for gate, other, ratio, count in (
        (a, b, rb, pair["b_pass_a_fail"]),
        (b, a, ra, pair["a_pass_b_fail"]),
    ):
        if gate not in gating or not count:
            continue
        # When BOTH metrics gate, neither is "the lenient one", so only a
        # systematic split is meaningful; keep the floor in that case.
        if other in gating and ratio < DISPUTE_RATIO:
            continue
        systematic = ratio >= DISPUTE_RATIO
        note("error" if systematic else "warning", "disputed",
             f"gating metric {gate!r} FAILED {count} case(s) ({ratio:.1%}) that "
             f"{other!r} PASSED. TWO readings: either {gate!r} is wrong on them, "
             "in which case pass rates and convergence are wrong, or "
             f"{other!r} is over-lenient — harmless while it only displays, "
             "unsafe the moment it is used to skip a more expensive metric, "
             "because nothing would catch the wrong verdict.",
             kind="gate_rejects_accepted", gate=gate, other=other, ratio=ratio,
             count=count)

    # Plain over-strict diagnostic. NOT guarded on the current verdict — it is
    # often the dominant finding by count, and a single over-lenient case must
    # not silence it. Ratio orientation matters and
    # is easy to invert: "other failed what gate passed" is a_pass_b_fail_ratio
    # when the gate is a, and b_pass_a_fail_ratio when the gate is b.
    for gate, other, ratio, count in (
        (a, b, ra, pair["a_pass_b_fail"]),
        (b, a, rb, pair["b_pass_a_fail"]),
    ):
        if gate in gating and other not in gating and ratio >= DISPUTE_RATIO:
            note("info", "review",
                 f"{other!r} failed {ratio:.1%} of cases that gating metric "
                 f"{gate!r} passed. TWO readings, and only domain knowledge "
                 f"separates them: either {other!r} is over-strict (its column "
                 "misleads and any expensive call it was meant to save is spent "
                 f"anyway), or {gate!r} is passing wrong answers that {other!r} "
                 "correctly rejects — in which case the reported pass rate is "
                 "OVERSTATED. Read a sample of these cases before concluding.",
                 kind="over_strict", gate=gate, other=other, ratio=ratio,
                 count=count)

    # Two non-gating metrics disagreeing: still worth a human look. Guarded on
    # role — without the guard this fires for gating pairs too and asserts
    # "neither gates case success", which would be a false statement in the
    # output rather than merely a missed finding.
    if a not in gating and b not in gating and max(ra, rb) >= DISPUTE_RATIO:
        note("info", "review",
             f"{a!r} and {b!r} disagree on {max(ra, rb):.1%} of cases; neither "
             "gates case success, so this is informational",
             kind="both_diagnostic", other=a, gate=b, ratio=max(ra, rb),
             count=max(pair["a_pass_b_fail"], pair["b_pass_a_fail"]))

    # ---- CAP: an uninformative metric is not evidence against anything ----
    # A metric that passes nearly everything (the audit has already labelled it
    # non_discriminating) makes "the gate failed cases this passed" close to
    # tautological, so the disagreement is explained by its own degeneracy, not
    # by the gate being wrong. Escalating on it is a false alarm that points the
    # finger at the DISCRIMINATING metric rather than the degenerate one.
    #
    # The severity rules above assume the non-gating metric is the STRICTER one,
    # in which case disagreement carries real information. Permissive judges like
    # Answer Relevancy are stock DeepEval metrics, so the opposite shape is the
    # common case.
    #
    # Findings are kept — they still describe the data — but the verdict cannot
    # exceed "review"/info on this evidence.
    #
    # The cap is direction- and role-blind ON PURPOSE: the tautology argument
    # holds whichever metric is degenerate, so a pair is never the right place to
    # raise the alarm. What the ROLE changes is the remedy — "drop it" is not
    # available for the metric that decides case success, and a saturated gate is
    # escalated at the METRIC level by ``_escalate_degenerate_gates`` instead.
    degenerate = sorted(m for m in (a, b) if m in uninformative)
    if degenerate and _RANK.get(pair["verdict"], 0) > _RANK["review"]:
        pair["verdict"] = "review"
        pair["severity"] = "info"
        # The signals are capped alongside the pair. A consumer that ranked by
        # signal severity while the pair said "review" would re-raise exactly the
        # false alarm this cap exists to suppress.
        for signal in pair["signals"]:
            signal["severity"] = "info"
            signal["capped"] = True
        degenerate_gates = [m for m in degenerate if m in gating]
        if degenerate_gates:
            remedy = (
                "It also DECIDES case success, so it cannot simply be dropped — "
                "see its own metric-level finding, which is where this is raised."
            )
        else:
            remedy = (
                "Fix or drop the non-discriminating metric first, then re-read "
                "this pair."
            )
        pair["findings"].append(
            "SEVERITY CAPPED: "
            + ", ".join(repr(m) for m in degenerate)
            + " is non-discriminating (it returns the same verdict on nearly every "
            "case), so its disagreements are explained by that degeneracy rather "
            "than by the other metric being wrong. " + remedy
        )


_RANK = {"ok": 0, "insufficient_data": 0, "review": 1, "non_discriminating": 2,
         "disputed": 3}

#: Severity ordering, so a later ``info`` finding cannot demote a ``warning``.
_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def _worst(audit: dict[str, Any]) -> str:
    """Highest verdict across metrics and pairs, for a one-glance UI badge."""
    worst = "ok"
    for entry in list(audit["metrics"].values()) + list(audit["pairs"]):
        v = entry.get("verdict", "ok")
        if _RANK.get(v, 0) > _RANK.get(worst, 0):
            worst = v
    return worst


def sidecar_path(run_file: Path) -> Path:
    """Audit sidecar path for a run file (``<stem>.metric-audit.json``)."""
    return run_file.with_suffix("").with_name(run_file.stem + SIDECAR_SUFFIX)


def audit_run_file(
    run_file: Path, gating_metrics: frozenset[str] | set[str] | None = None
) -> dict[str, Any] | None:
    """Analyze a run file and write its audit sidecar.

    Best-effort by contract: every failure path logs and returns ``None``. A
    metric audit must never be able to fail an eval that already produced
    results — the numbers are the deliverable, this is commentary on them.

    Args:
        run_file: Path to an ``eval_results_*.json`` artifact.
        gating_metrics: Metric names that decide case success.

    Returns:
        The audit dict, or ``None`` if the run file could not be analyzed.
    """
    try:
        runs = json.loads(run_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("[metric-audit] cannot read %s: %s", run_file.name, e)
        return None

    try:
        audit = analyze_run(runs, gating_metrics)
        audit["run_file"] = run_file.name
        out = sidecar_path(run_file)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        tmp.replace(out)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("[metric-audit] cannot write audit for %s: %s", run_file.name, e)
        return None

    # A non-discriminating metric is at least as ACTIONABLE as a disputed pair —
    # "this metric cannot tell your best model from your worst; fix or drop it" is a
    # concrete instruction, where a disputed pair only asks for adjudication. At
    # INFO it gets buried, and the degeneracy framing is invisible whenever a pair
    # verdict outranks it — a metric passing nearly every case on models the gate
    # separates widely would be reported only as "over-lenient". Both warn, and
    # the degeneracy is named explicitly.
    degenerate = [
        n for n, e in audit["metrics"].items() if e["non_discriminating"]
    ]
    if degenerate:
        logger.warning(
            "[metric-audit] %s: NON-DISCRIMINATING metric(s) %s — returns the same "
            "verdict on ~every case, so it cannot separate a good model from a bad "
            "one. Fix or drop it (or, if it gates case success, fix it — the run's "
            "pass rates rest on it); until then its disagreements are not evidence "
            "about any other metric.",
            run_file.name, ", ".join(sorted(degenerate)),
        )
    if audit["worst_verdict"] == "disputed":
        # Metric findings first, and included at all because a saturated GATING
        # metric is disputed at the METRIC level while its pairs stay capped at
        # "review" — reading only pair findings here printed an empty message.
        logger.warning(
            "[metric-audit] %s: metrics DISPUTED — %s",
            run_file.name,
            "; ".join(
                [f for e in audit["metrics"].values() if e["verdict"] == "disputed"
                 for f in e["findings"]]
                + [f for p in audit["pairs"] if p["verdict"] == "disputed"
                   for f in p["findings"]]
            )[:600],
        )
    elif not degenerate:
        logger.info(
            "[metric-audit] %s: %s (%d metric(s), %d pair(s))",
            run_file.name, audit["worst_verdict"],
            len(audit["metrics"]), len(audit["pairs"]),
        )
    return audit


def gating_metrics_for(project_dir: Path) -> frozenset[str]:
    """Gating metric names for a project, from the harness's target scores.

    Returns an empty set when the harness cannot be read or sets no targets —
    the audit then reports pairwise disagreement without role escalation, which
    is still useful and is honest about not knowing which metric decides.
    """
    try:
        from app.services.harness_config import load_harness_config
        harness = load_harness_config(project_dir)
    except Exception as e:  # noqa: BLE001 — never let this fail a run
        logger.warning("[metric-audit] cannot load harness config: %s", e)
        return frozenset()
    targets = getattr(harness, "target_metric_scores", None) if harness else None
    return frozenset(targets) if isinstance(targets, dict) else frozenset()
