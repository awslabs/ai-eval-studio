# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure analysis service for eval results.

This module provides structured analysis of eval runs, breaking down
results by category, metric, and individual case details — BOTH failing
and passing. The optimizer uses the passing cases as a "what good looks
like" template (the single biggest gap vs. the manual tuning process,
which converged fast precisely because the human read the whole result,
not just the failures) and the failing cases as the work list.

Expects DeepEval-native format: a JSON array of model runs, each containing
a "results" array with test results using DeepEval field names (success,
metrics_data, additional_metadata).

Size discipline: the MCP transport caps a tool response at
``mcp_response.MAX_MCP_PAYLOAD_BYTES`` (900 KB). ``analyze_failures``
emits full untruncated per-case detail (so the synthesis/goldens path gets
complete data); the optimizer tool boundary (``mcp_tools/prompt.py``) calls
``shape_failure_digest`` to bound + truncate + backstop under the cap before
sending to the agent. The optimizer Reads the raw run file for full detail.
"""

import copy
import json
from pathlib import Path
from typing import Any


#: How many passing-case exemplars the digest carries. The optimizer
#: Reads the raw run file for full ground truth (its cwd is the project
#: dir and it has the Read tool); these exemplars are just a high-signal
#: "what good looks like" sample so the digest alone shows the winning
#: shape. Kept small so the digest stays well under the MCP payload cap
#: regardless of how high the pass rate climbs toward convergence.
_MAX_PASSING_EXEMPLARS = 8

#: Maximum failed-case sample size for the optimizer-boundary digest.
#: Kept small (like passing exemplars) to prevent pathological runs with
#: ~150 failures of ~100K-char inputs from exceeding the MCP cap. The
#: optimizer Reads the raw run file for full case detail; the digest
#: sample only needs to surface the dominant failure pattern.
_MAX_FAILED_EXEMPLARS = 12


def _select_passing_exemplars(
    passing: list[dict[str, Any]], limit: int = _MAX_PASSING_EXEMPLARS,
) -> list[dict[str, Any]]:
    """Pick a small, diverse sample of passing cases for the digest.

    Diversity is best-effort and category-OPTIONAL: ``category`` comes
    from ``additional_metadata`` and is frequently absent. When categories
    exist we round-robin across them so the sample spans case types;
    when they don't (or there are few), we fall back to evenly-spaced
    sampling across the list so the exemplars aren't all clustered at the
    front. Returns ``passing`` unchanged when it's already within ``limit``.

    Args:
        passing: Projected passing-case dicts (from :func:`_case_detail`).
        limit: Maximum exemplars to return.

    Returns:
        Up to ``limit`` passing-case dicts.
    """
    if len(passing) <= limit:
        return passing

    # Group by category (None is its own bucket). If every case shares one
    # bucket — the common no-category case — this degrades to a single
    # group and we fall through to stride sampling within it.
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for case in passing:
        buckets.setdefault(case.get("category"), []).append(case)

    if len(buckets) > 1:
        # Round-robin across categories until we hit the limit.
        selected: list[dict[str, Any]] = []
        ordered = [list(v) for v in buckets.values()]
        idx = 0
        while len(selected) < limit and any(ordered):
            group = ordered[idx % len(ordered)]
            if group:
                selected.append(group.pop(0))
            idx += 1
        return selected[:limit]

    # Single bucket — evenly-spaced (stride) sample so we don't just take
    # the first ``limit`` (which can cluster on one part of the dataset).
    stride = len(passing) / limit
    return [passing[int(i * stride)] for i in range(limit)]


def _case_detail(r: dict[str, Any], *, failing_only: bool) -> dict[str, Any]:
    """Project one DeepEval result row to the optimizer-facing case shape.

    Args:
        r: A single test-result dict from a model run's ``results`` array.
        failing_only: When True, emits ``failing_metrics`` carrying only the
            metrics that FAILED (the work list for a failing case). When
            False, emits ``metrics`` carrying every metric with its score +
            judge reason (so a passing case shows WHY it passed — the
            template signal). The key name differs so a consumer can never
            mistake a failing case's partial metric block for a full one.

    Returns:
        Dict with case_id, input, expected_output, actual_output, category,
        and a metric dict keyed by metric name under ``failing_metrics``
        (failing_only=True) or ``metrics`` (failing_only=False).
    """
    metrics: dict[str, Any] = {}
    for md in r.get("metrics_data", []) or []:
        if failing_only and md.get("success", False):
            continue
        name = md.get("name")
        if not name:
            continue
        metrics[name] = {
            "score": md.get("score"),
            "reason": md.get("reason"),
            "threshold": md.get("threshold"),
        }
    meta = r.get("additional_metadata") or {}
    return {
        "case_id": meta.get("id"),
        "input": r.get("input"),
        "expected_output": r.get("expected_output"),
        "actual_output": r.get("actual_output"),
        "category": meta.get("category"),
        "failing_metrics" if failing_only else "metrics": metrics,
    }


def analyze_failures(results_path: Path) -> dict[str, Any]:
    """Analyze an eval results file (DeepEval-native format), pass AND fail.

    The file is a JSON array of model runs. Analyzes each model separately.

    Args:
        results_path: Path to eval results JSON file.

    Returns:
        Dict with per-model analysis under "model_analyses" key. Each
        analysis contains total/passed/failed counts, pass_rate, failure
        breakdowns by category and metric, ``failed_case_details`` (the
        work list — full input/expected/actual plus ``failing_metrics``
        with judge reasons, read by ``failure_targeted_prompt``), and
        ``passing_case_details`` (the template — the same fields with
        ``metrics`` carrying every metric's score + reason).
    """
    with results_path.open() as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {"error": "Invalid format: expected array of model runs"}

    analyses = []
    for model_run in data:
        test_run_id = model_run.get("test_run_id", "unknown")
        results = model_run.get("results", [])
        prompt_version = model_run.get("hyperparameters", {}).get("prompt_version", "")

        total = len(results)
        passed = sum(1 for r in results if r.get("success", False))
        failed = total - passed

        # By category
        category_stats: dict[str, dict[str, int]] = {}
        for r in results:
            category = (r.get("additional_metadata") or {}).get("category")
            if not category:
                continue
            if category not in category_stats:
                category_stats[category] = {"total": 0, "failed": 0}
            category_stats[category]["total"] += 1
            if not r.get("success", False):
                category_stats[category]["failed"] += 1

        failure_by_category = {}
        for cat, stats in category_stats.items():
            t = stats["total"]
            f = stats["failed"]
            failure_by_category[cat] = {
                "total": t, "failed": f,
                "pass_rate": (t - f) / t if t > 0 else 0.0,
            }

        # By metric
        metric_stats: dict[str, dict[str, int]] = {}
        for r in results:
            for md in r.get("metrics_data", []):
                name = md.get("name", "unknown")
                if name not in metric_stats:
                    metric_stats[name] = {"total": 0, "failed": 0}
                metric_stats[name]["total"] += 1
                if not md.get("success", False):
                    metric_stats[name]["failed"] += 1

        failure_by_metric = {}
        for name, stats in metric_stats.items():
            t = stats["total"]
            f = stats["failed"]
            failure_by_metric[name] = {
                "total": t, "failed": f,
                "pass_rate": (t - f) / t if t > 0 else 0.0,
            }

        # Per-case detail, split into the work list (failures) and the
        # template (passes). Passing cases are
        # collected in full first, then DOWN-SAMPLED to a bounded set of
        # exemplars for the digest — the optimizer Reads the raw run file
        # for the complete picture, so the digest only needs enough
        # passing cases to show the winning shape, and stays small as the
        # pass rate climbs toward convergence.
        failed_details = []
        all_passing = []
        for r in results:
            if r.get("success", False):
                all_passing.append(_case_detail(r, failing_only=False))
            else:
                failed_details.append(_case_detail(r, failing_only=True))

        passing_exemplars = _select_passing_exemplars(all_passing)

        analyses.append({
            "test_run_id": test_run_id,
            "prompt_version": prompt_version,
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "failure_by_category": failure_by_category,
            "failure_by_metric": failure_by_metric,
            "failed_case_details": failed_details,
            # Bounded sample (≤ _MAX_PASSING_EXEMPLARS); not the full set.
            "passing_case_details": passing_exemplars,
            "passing_exemplars_truncated": len(all_passing) > len(passing_exemplars),
        })

    return {"model_analyses": analyses}


def _case_key(r: dict[str, Any]) -> Any:
    """Stable identity for a result row, for cross-run case matching.

    Prefers the explicit case id from ``additional_metadata``; falls back
    to the input text when no id is stamped (older harness scaffolds).
    Two runs of the same dataset present the same cases, so this lets us
    line up case N in the prior run with case N in the current run.
    """
    meta = r.get("additional_metadata") or {}
    cid = meta.get("id")
    if cid is not None:
        return ("id", cid)
    return ("input", r.get("input"))


def diff_regressions(
    current_path: Path, prior_path: Path,
) -> dict[str, Any]:
    """Find cases that went pass→fail from the prior run to the current run.

    Regression visibility is DATA the optimizer can act on — it is NOT a
    loop gate. Nothing here blocks a regressing prompt version; the
    optimizer is simply shown which cases its last change broke so it can
    choose to fix or accept the trade. (A hard reject gate is a possible
    follow-up if testing shows regressions still slip through.)

    Cases are matched across runs by ``_case_key`` (stamped id, else input
    text). Models are matched by their hyperparameters ``model`` id so a
    multi-model run diffs each model against its own prior result.

    Args:
        current_path: Path to the just-completed run's results file.
        prior_path: Path to the immediately-preceding run's results file.

    Returns:
        Dict ``{"regressions": [{model, count, cases: [...]}, ...]}`` where
        each case carries the projected detail plus the prior run's
        passing ``actual_output`` so the optimizer can see what the output
        looked like when it still passed. Returns ``{"regressions": []}``
        when nothing regressed or either file is unreadable (best-effort:
        a missing/parse-failed prior run means "no regression data", never
        an error that derails optimization).
    """
    try:
        with current_path.open() as f:
            current_data = json.load(f)
        with prior_path.open() as f:
            prior_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"regressions": []}

    if not isinstance(current_data, list) or not isinstance(prior_data, list):
        return {"regressions": []}

    def _model_id(model_run: dict) -> str:
        return model_run.get("hyperparameters", {}).get("model", "")

    # Index prior runs by model id → {case_key: result_row}.
    prior_by_model: dict[str, dict[Any, dict]] = {}
    for model_run in prior_data:
        prior_by_model[_model_id(model_run)] = {
            _case_key(r): r for r in model_run.get("results", [])
        }

    regressions = []
    for model_run in current_data:
        model_id = _model_id(model_run)
        prior_cases = prior_by_model.get(model_id)
        if not prior_cases:
            # No matching prior model run (e.g. model set changed) — can't
            # attribute regressions for this model; skip it.
            continue

        regressed_cases = []
        for r in model_run.get("results", []):
            if r.get("success", False):
                continue  # currently passing — not a regression
            prior = prior_cases.get(_case_key(r))
            if prior is None or not prior.get("success", False):
                continue  # wasn't passing before — not a regression
            # pass (prior) → fail (current): a regression.
            detail = _case_detail(r, failing_only=True)
            detail["prior_actual_output"] = prior.get("actual_output")
            regressed_cases.append(detail)

        if regressed_cases:
            regressions.append({
                "model": model_id,
                "count": len(regressed_cases),
                "cases": regressed_cases,
            })

    return {"regressions": regressions}


def _truncate_text_field(value: Any, snippet_bytes: int = 2000) -> str:
    """Truncate a large text field to a snippet with a marker.

    Args:
        value: The field value to truncate (typically input/expected/actual).
        snippet_bytes: Maximum bytes to keep.

    Returns:
        Truncated string with a sentinel marker appended when truncated.
    """
    text = str(value) if value is not None else ""
    if len(text.encode("utf-8", errors="ignore")) <= snippet_bytes:
        return text
    # Truncate to snippet and append truncation marker
    truncated = text[:snippet_bytes]
    remaining = len(text) - len(truncated)
    return f"{truncated}…[truncated {remaining} chars]"


def shape_failure_digest(analysis: dict[str, Any]) -> dict[str, Any]:
    """Bound the failure digest under the MCP payload cap for optimizer tool boundary.

    This function takes the FULL analyze_failures output and returns a
    DEEP-COPIED, bounded version safe for the MCP transport
    (``MAX_MCP_PAYLOAD_BYTES``). It MUST NOT mutate its input — the
    synthesis path (``failures_to_seed_goldens``) reuses the raw dict and
    needs complete ``input``/``expected_output`` to build real DeepEval
    Golden objects.

    Bounding strategy:
    1. Limit ``failed_case_details`` to ``_MAX_FAILED_EXEMPLARS`` using
       the same category-spread/stride approach as ``_select_passing_exemplars``.
    2. Truncate large text fields (``input``, ``expected_output``,
       ``actual_output``) in both failed and passing case details.
    3. Apply analogous truncation to regression block ``cases``.
    4. Backstop: if still over cap after field truncation, drop more
       failed/passing exemplars until under cap or empty.

    Invariants:
    - ``failed_cases`` count (true total) is never recomputed from sample
    - ``failing_metrics`` survives bounding intact (``failure_targeted_prompt``
      reads it per case)
    - ``failed_details_truncated`` flag set when sample < total
    - Keys and nesting match the unbounded output, so a consumer reads both
      the same way

    Args:
        analysis: The full analyze_failures output (possibly with ``regression`` key).

    Returns:
        A deep-copied, bounded dict safe for ``mcp_json``.
    """
    from app.services.mcp_response import MAX_MCP_PAYLOAD_BYTES

    # Deep copy so we never mutate the raw analysis
    shaped = copy.deepcopy(analysis)

    # Bound and truncate per-model analyses
    for model_analysis in shaped.get("model_analyses", []):
        failed_details = model_analysis.get("failed_case_details", [])
        true_failed_count = len(failed_details)

        # Bound failed sample using the same category-spread logic as passing
        if len(failed_details) > _MAX_FAILED_EXEMPLARS:
            bounded_failed = _select_passing_exemplars(failed_details, _MAX_FAILED_EXEMPLARS)
            model_analysis["failed_case_details"] = bounded_failed
            model_analysis["failed_details_truncated"] = True
        else:
            model_analysis["failed_details_truncated"] = False

        # Preserve true total (never recompute from sample)
        model_analysis["failed_cases"] = true_failed_count

        # Truncate large text fields in failed cases
        for case in model_analysis.get("failed_case_details", []):
            case["input"] = _truncate_text_field(case.get("input"))
            case["expected_output"] = _truncate_text_field(case.get("expected_output"))
            case["actual_output"] = _truncate_text_field(case.get("actual_output"))

        # Truncate large text fields in passing exemplars
        for case in model_analysis.get("passing_case_details", []):
            case["input"] = _truncate_text_field(case.get("input"))
            case["expected_output"] = _truncate_text_field(case.get("expected_output"))
            case["actual_output"] = _truncate_text_field(case.get("actual_output"))

    # Truncate regression block cases if present
    if "regression" in shaped:
        for reg in shaped["regression"].get("regressions", []):
            for case in reg.get("cases", []):
                case["input"] = _truncate_text_field(case.get("input"))
                case["expected_output"] = _truncate_text_field(case.get("expected_output"))
                case["actual_output"] = _truncate_text_field(case.get("actual_output"))
                if "prior_actual_output" in case:
                    case["prior_actual_output"] = _truncate_text_field(case["prior_actual_output"])

    # Backstop: if still over cap, progressively drop more exemplars
    serialized = json.dumps(shaped, default=str)
    if len(serialized.encode("utf-8", errors="ignore")) > MAX_MCP_PAYLOAD_BYTES:
        # Try dropping more failed exemplars first
        for model_analysis in shaped.get("model_analyses", []):
            while len(model_analysis.get("failed_case_details", [])) > 0:
                model_analysis["failed_case_details"].pop()
                model_analysis["failed_details_truncated"] = True
                serialized = json.dumps(shaped, default=str)
                if len(serialized.encode("utf-8", errors="ignore")) <= MAX_MCP_PAYLOAD_BYTES:
                    return shaped

            # Still over? Drop passing exemplars too
            while len(model_analysis.get("passing_case_details", [])) > 0:
                model_analysis["passing_case_details"].pop()
                model_analysis["passing_exemplars_truncated"] = True
                serialized = json.dumps(shaped, default=str)
                if len(serialized.encode("utf-8", errors="ignore")) <= MAX_MCP_PAYLOAD_BYTES:
                    return shaped

    return shaped
