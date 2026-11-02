# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval report facts: facts builders + report-JSON store.

Three stages, each independently unit-testable:

1. **Facts builders** (this module): walk a project's run files and produce
   :class:`RunFacts` / :class:`ExecFacts` — every number computed
   deterministically from the run files. No LLM, no agent.
2. **Narrator agent** (``app/agents/report_narrator.py``): an LLM that is
   handed the facts-pack and writes prose ONLY (recommendation, narrative,
   failure clusters). Output goes through the ``save_report`` MCP tool, which
   recomputes facts server-side so the agent can never poison numbers.
3. **Rendering** happens in the SPA, not in this module: ``api/reports.py``
   serves ``{facts, narrative}`` and the React components in
   ``frontend/src/components/reports/`` lay it out. When the narrative is
   missing (agent never ran or failed), they render a lean computed-only
   fallback from the facts alone — tables, KPI tiles, a computed verdict.

The on-disk report JSON at ``output/reports/<stem>.report.json`` is a
first-class user-readable artifact: facts + narrative + metadata, with
documented field names.
"""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.models.schemas import ProjectConfig
from app.services.metric_name import base_metric_name
from app.services.bedrock_pricing import NA, lookup_cached_pricing
from app.services.eval_runner import EvalRunner
from app.services.failure_analyzer import _case_key
from app.services.jobs.iteration_metrics import extract_iteration_metrics
from app.services.optimization_state import OptimizationState
from app.services.project_paths import safe_basename_path
from app.services.run_trust import RunTrustFacts, build_run_trust

logger = logging.getLogger(__name__)


# Sidecar/report-JSON schema version. Bump when the on-disk shape changes
# in a way the renderer can't reverse. The renderer rejects unknown
# versions and falls through to lean fallback.
REPORT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Facts-pack data shape (the deterministic, computed half of a report)
# --------------------------------------------------------------------------

@dataclass
class ModelFacts:
    """Per-model aggregates that drive every model-comparison table.

    Attributes:
        model_id: Full Bedrock model id.
        display_name: Friendly short name (e.g. ``nova-2-lite``).
        total: Cases this model ran.
        passed: Cases where every metric passed.
        pass_rate: ``passed/total`` as a percentage (0–100).
        metric_rates: Per-metric pass rate (0–100), keyed by short metric name.
        avg_latency_ms: Mean per-case latency, or None when unmeasured.
        latency_percentiles: ``{median, p95, p99, max}`` (int ms) +
            ``over_budget_count`` (int|None) + ``total`` (int). Inner values
            are None when the model has no per-case ``latency_ms``.
        generation_cost_usd: Target-model spend.
        judge_cost_usd: LLM-as-judge spend.
        total_cost_usd: Sum.
        cost_per_1k_gen: GENERATION cost per 1,000 queries — the
            production deploy figure. Excludes judge/eval spend.
        judge_cost_available: False when the run reports judge spend as unknown
            (null); the renderer hides judge cost rather than showing $0.00.
        pricing_available: False when LiteLLM has no pricing entry for the
            model id (mantle-tier ids in the allowlist that LiteLLM doesn't
            yet price). The renderer hides USD values rather than $0.00 in
            this case (mirrors ``judge_cost_available``).
        avg_input_tokens / avg_output_tokens / cache_read_ratio:
            Token usage averages and the prompt-cache hit ratio (0–1).
            Used for the diagnostics section ("caching off" flag).

            ``avg_input_tokens`` is the WHOLE prompt — uncached input plus
            cache reads — because Bedrock reports those as two separate
            counters and ``inputTokens`` alone is the uncached remainder.
            ``cache_read_ratio`` is that prompt's cached fraction
            (``cache_read / (input + cache_read)``), so the two read together:
            size, and how much of it was free.
    """

    model_id: str
    display_name: str
    total: int
    passed: int
    pass_rate: float
    metric_rates: dict[str, float]
    avg_latency_ms: float | None
    latency_percentiles: dict[str, int | None]
    generation_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float
    cost_per_1k_gen: float
    judge_cost_available: bool
    avg_input_tokens: float
    avg_output_tokens: float
    cache_read_ratio: float
    pricing_available: bool = True
    # Per-model deltas vs the prior prompt version's latest run for THIS
    # same model. None when no prior run for the model exists (first
    # appearance, or rendering the baseline). Same shape conventions as
    # ``pass_rate``/``metric_rates``: percentage points.
    pass_rate_delta: float | None = None
    metric_rate_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class FailedCaseFacts:
    """One (case, failed metric) row carried in the facts-pack.

    The narrator may sample these to cluster patterns; the renderer looks up
    a pattern's ``representative_case_id`` against this list to fetch the
    example to display.
    """

    case_id: Any
    category: str
    model_id: str
    model_display: str
    metric: str
    is_latency: bool
    input: str
    expected: str
    actual: str
    reason: str


@dataclass
class JourneyVersionFacts:
    """One prompt version on the journey (one row per ``vN`` on disk)."""

    version: str
    best_pass_rate: float
    delta: float | None  # None for the first version
    per_model_pass_rates: dict[str, float]  # display_name -> rate %
    cost_per_1k_gen: float | None  # winner-of-version's gen cost
    avg_input_tokens: float | None  # prompt-size proxy
    what_changed: str | None  # parsed from the prompt header


@dataclass
class RunIntegrityFacts:
    """Honest-denominator check, lifted from the harness's per-model
    ``data_quality`` dicts. When any count > 0, the report must say so.

    Counts are summed across all models in the run (the harness writes one
    block per model_run). ``case_count`` is the dataset's intended size;
    ``effective_total`` = case_count − errored − dropped − unrecovered.
    """

    case_count: int
    errored: int
    dropped: int
    null_count: int
    unrecovered: int
    effective_total: int
    reliability: str  # "ok" | "degraded" | "bad" — harness's own label


@dataclass
class RecommendationFacts:
    """Per-model verdict tag the renderer shows in the comparison table.

    The narrator writes the prose; this struct captures the deterministic
    tag (``Recommended`` / ``Strong, costly`` / ``Viable`` / ``Below targets``)
    so the lean fallback also has a label.
    """

    model_id: str
    tag: str


@dataclass
class FailureBucketFacts:
    """Failure-attribution bucket counts (cases EVERY model failed).

    Three buckets, computed by :func:`_classify_failed_by_all`:
    ``judge_false_negatives`` (output==expected, all failed → judge bug),
    ``data_review`` (models agree, ≠expected → data bug),
    ``model_gaps`` (models disagree → genuine capability gap).
    """

    judge_false_negatives: int
    data_review: int
    model_gaps: int


@dataclass
class RunFacts:
    """Everything a per-run report needs, pre-computed from the run file.

    The narrator writes around these; the renderer lays them out. JSON-
    serialized verbatim into the report sidecar.
    """

    schema_version: int
    project_name: str
    project_description: str
    run_filename: str
    run_timestamp: str
    run_source: str  # "manual" | "optimization"
    prompt_label: str  # e.g. "v8"
    prompt_iteration: int | None  # 1-based iteration index when known
    prompt_content: str  # the full prompt text for the appendix
    prompt_changes: str | None  # parsed Changes: header, single string
    case_count: int
    category_count: int  # distinct categories present on the cases (0 when uncategorized)
    metric_names: list[str]  # short names, sorted
    metric_thresholds: dict[str, float]
    metric_criteria: dict[str, str]  # raw GEval criteria — narrator-only
    latency_budget_ms: int | None
    has_latency_metric: bool
    hyperparameters: dict[str, Any]  # temperature, max_tokens, etc.
    models: list[ModelFacts]
    recommendations: list[RecommendationFacts]
    # Single source of truth for "which model is recommended" across the
    # report — KPI tiles, header, and narrative all read THIS field. The
    # ``recommendations[].tag`` semantics ("Recommended" / "Top pass rate")
    # depend on whether targets are set AND cleared, so they go null on a
    # plateau even when there's still a clear best-of-the-bunch model. This
    # field stays populated regardless and is what the renderer should key on.
    recommended_model_id: str | None
    failed_cases: list[FailedCaseFacts]
    failure_buckets: FailureBucketFacts
    run_integrity: RunIntegrityFacts
    # Every reason to doubt this run's numbers, with one severity scale. Built
    # from ``run_integrity`` plus the metric-audit sidecar. The renderers, the
    # run list and the agent all read THIS rather than each interpreting the
    # underlying checks, so a caveat cannot be worded two ways or shown on one
    # surface and not another. ``severity == "ok"`` with no reasons is the
    # explicit all-clear, not an absence.
    run_trust: RunTrustFacts
    prior_step: dict[str, Any] | None  # {version, pass_rate, delta} or None
    is_winning_run: bool


@dataclass
class ExecFacts:
    """Everything an executive report needs, pre-computed from run files.

    The exec is built from the project's full run history (one row per
    prompt version, latest run per version). The winner is picked via
    ``_pick_trophy_run``; that run's per-run facts are inlined as
    ``winner_run`` so the renderer can drive the model comparison and
    diagnostics from a single source.
    """

    schema_version: int
    project_name: str
    project_description: str
    case_count: int
    category_count: int
    metric_names: list[str]
    metric_thresholds: dict[str, float]
    metric_criteria: dict[str, str]
    latency_budget_ms: int | None
    journey: list[JourneyVersionFacts]
    winner_version: str
    winner_run_filename: str
    winner_run: RunFacts
    # Humanized "v8 run · June 12, 2026" stamp (frontend-friendly; never the
    # raw .json filename — that's an implementation detail users don't want).
    generated_from_label: str
    generated_at: str  # ISO timestamp set by the builder


# --------------------------------------------------------------------------
# Helpers — config, run discovery, generic
# --------------------------------------------------------------------------

def _load_project_config(project_dir: Path) -> ProjectConfig:
    """Read ``config.json`` into a :class:`ProjectConfig`."""
    config_path = project_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {project_dir}")
    return ProjectConfig(**json.loads(config_path.read_text()))


def _model_id_of(model_run: dict) -> str:
    """Resolve a per-model run dict's model id, with the usual fallbacks."""
    return (
        model_run.get("model_id")
        or model_run.get("hyperparameters", {}).get("model")
        or model_run.get("test_run_id")
        or "unknown"
    )


def _percentile(sorted_values: list[float], q: float) -> int:
    """Return the q-th percentile of ``sorted_values`` (asc), rounded to int."""
    if not sorted_values:
        return 0
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return int(round(sorted_values[idx]))


# --------------------------------------------------------------------------
# Trophy ranking — scoring lives in app.services.trophy (THE single
# implementation, shared with the wire that feeds the dashboard). This module
# only picks winners from those scores.
# --------------------------------------------------------------------------

def _pick_trophy_run(
    project_dir: Path,
    runner: EvalRunner,
    state: "OptimizationState | None",
) -> str | None:
    """Pick the trophy-winning run filename (same scores the dashboard ranks).

    Targets come from HARNESS CONFIG (the criteria owner) — not from the
    OptimizationState snapshot this used to read. The panel already
    ranked against config; ranking the report against the dispatch-time
    snapshot meant a user who edited targets after dispatch got a report
    crowning one run and a panel crowning another.
    """
    from app.services.trophy import (
        resolve_trophy_targets,
        trophy_score_loop,
        trophy_score_run,
        trophy_sort_key,
    )

    targets = resolve_trophy_targets(project_dir)
    if not targets:
        return None

    try:
        all_runs = runner.list_runs(project_dir)
    except Exception:
        return None
    runs_by_file = {r["_file"]: r for r in all_runs}

    candidates: list[tuple[str, str, tuple[int, float, float]]] = []

    for run in all_runs:
        candidates.append((
            run["_file"],
            run.get("timestamp") or "",
            trophy_sort_key(trophy_score_run(run, targets)),
        ))
    if state is not None:
        for it in state.iterations:
            run_id = it.get("run_id")
            if not run_id or run_id not in runs_by_file:
                continue
            ts = it.get("timestamp") or runs_by_file[run_id].get("timestamp") or ""
            loop_score = trophy_sort_key(trophy_score_loop(it, targets))
            existing = next(
                (i for i, c in enumerate(candidates) if c[0] == run_id),
                None,
            )
            if existing is None:
                candidates.append((run_id, ts, loop_score))
            elif loop_score >= candidates[existing][2]:
                candidates[existing] = (run_id, ts, loop_score)

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[2], c[1]))
    return candidates[-1][0]


def _resolve_exec_winner(
    project_dir: Path,
    runner: EvalRunner,
    state: "OptimizationState | None",
) -> str | None:
    """Pick the executive-scope winning run filename.

    With targets configured this defers to :func:`_pick_trophy_run`. With no
    targets it falls back to "best per-model pass rate, latest timestamp as
    tiebreak" — same logic ``build_exec_facts`` uses to choose what to
    summarize. Returns ``None`` only when the project has no eval runs at
    all.

    Centralizing this is what keeps the GET-side cache-drop check (``did the
    winner change?``) symmetric with the post-eval auto-regen guard. Before
    this helper existed, the guard called ``_pick_trophy_run`` directly and
    returned early on no-targets — so the cache drop fired but the regen
    didn't, leaving every no-targets project stuck on the lean fallback
    after each new eval until someone hit Regenerate manually.
    """
    trophy = _pick_trophy_run(project_dir, runner, state)
    if trophy is not None:
        return trophy
    try:
        runs = runner.list_runs(project_dir)
    except Exception:
        return None
    if not runs:
        return None
    def _best_rate_for(run: dict) -> float:
        best = 0.0
        for mr in run.get("model_runs") or []:
            tot = mr.get("total") or 0
            passed = mr.get("passed") or 0
            rate = (passed / tot) if tot else 0.0
            if rate > best:
                best = rate
        return best
    ranked = sorted(
        runs,
        key=lambda r: (_best_rate_for(r), r.get("timestamp") or ""),
    )
    return ranked[-1]["_file"]


# --------------------------------------------------------------------------
# Run-file readers
# --------------------------------------------------------------------------

def _select_run(
    runner: EvalRunner, project_dir: Path, run_filename: str | None
) -> tuple[str, list[dict]]:
    """Resolve which run file to report on and load it.

    ``run_filename`` is untrusted — the report endpoints take it from the
    request body, so it is not constrained by route matching the way the
    single-segment path params on ``api/eval_runs.py`` are. It goes
    through ``safe_basename_path`` (the same helper the eval-runs routes
    and the ``save_report`` MCP tool use) because run files are always a
    bare filename in ``output/``: that rejects separators, ``..``, hidden
    names, and embedded NULs, and catches symlinks leaving the tree.
    Validating here rather than at the two call sites also covers the
    prior-version recursion below and any future caller.
    """
    if run_filename is None:
        runs = runner.list_runs(project_dir)
        if not runs:
            raise ValueError(f"Project {project_dir} has no eval runs")
        run_filename = runs[0]["_file"]

    results_path = safe_basename_path(project_dir / "output", run_filename)
    if results_path is None or not results_path.exists():
        raise FileNotFoundError(f"Run not found: {run_filename}")

    data = runner.load_results(results_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Run {run_filename} is empty or malformed")
    return run_filename, data


def _parse_metric_definitions(project_dir: Path) -> dict[str, str]:
    """Extract per-metric definitions from the project's metrics module.

    Resolves simple module-level ``_NAME = "..."`` aliases so call sites
    that pass a constant by reference yield their criteria text. Empty
    when the file is absent or unparseable.
    """
    import ast

    path = project_dir / "src" / "metrics" / "metrics.py"
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return {}

    aliases: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, str):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                aliases[tgt.id] = value

    def _resolve(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name) and node.id in aliases:
            return aliases[node.id]
        try:
            value = ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, str) else None

    definitions: dict[str, str] = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        name = _resolve(kw.get("name"))
        criteria = _resolve(kw.get("criteria"))
        if name and criteria:
            definitions[name] = re.sub(r"\s+", " ", criteria).strip()
    return definitions


def _parse_latency_budget_ms_from_criteria(criteria: str) -> int | None:
    """Pull the millisecond budget out of a Latency criteria string."""
    if not criteria:
        return None
    match = re.search(r"(\d{2,5})\s*ms\b", criteria)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _detect_latency_metric(results: list[dict]) -> bool:
    """True iff any case carries a metrics_data entry named ``Latency``."""
    for model_run in results:
        for case in model_run.get("results", []):
            for md in case.get("metrics_data", []) or []:
                if base_metric_name(md.get("name", "")) == "Latency":
                    return True
    return False


def _detect_latency_budget_ms_from_runs(results: list[dict]) -> int | None:
    """Pull the ms budget out of any Latency metric's ``reason`` text.

    Custom Latency metrics (``BaseMetric`` subclass, not ``GEval``) don't
    surface their budget through AST parsing — but every per-case ``reason``
    emits the budget (e.g. ``"Latency 591ms is within budget (3000ms).")``.
    """
    for model_run in results:
        for case in model_run.get("results", []):
            for md in case.get("metrics_data", []) or []:
                if base_metric_name(md.get("name", "")) != "Latency":
                    continue
                reason = md.get("reason") or ""
                match = re.search(r"\((\d{2,5})\s*ms\)", reason)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
    return None


# --------------------------------------------------------------------------
# Per-model facts builder
# --------------------------------------------------------------------------

def _build_model_facts(results: list[dict], has_latency_metric: bool) -> list[ModelFacts]:
    """Produce one :class:`ModelFacts` per model in the run, fully computed."""
    agg = extract_iteration_metrics(results)
    per_model_rates = agg["per_model_pass_rates"]
    per_model_metrics = agg["per_model_metrics"]

    runner = EvalRunner()
    out: list[ModelFacts] = []
    for model_run in results:
        cases = model_run.get("results", [])
        model_id = _model_id_of(model_run)
        total = len(cases)
        # Two ways pricing can be unavailable, and the renderer shows literal
        # "n/a" rather than $0.00 for both: a mantle-allowlist id with no LiteLLM
        # entry (the ``NA`` sentinel), or a run whose runner reported cost as
        # unknown (null ``generation_cost_usd`` — no pricing.json entry, and it
        # refused to invent rates). The key is always written, so there is no
        # third "key absent" case to treat as available.
        pricing_available = (
            lookup_cached_pricing(model_id) is not NA
            and model_run.get("generation_cost_usd") is not None
        )

        latencies: list[float] = []
        over_budget = 0
        in_tokens = 0
        out_tokens = 0
        cache_read = 0

        for case in cases:
            meta = case.get("additional_metadata") or {}
            lat = meta.get("latency_ms")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
            in_t = meta.get("input_tokens") or 0
            out_t = meta.get("output_tokens") or 0
            cr = meta.get("cache_read_input_tokens") or 0
            in_tokens += in_t
            out_tokens += out_t
            cache_read += cr
            if has_latency_metric:
                for md in case.get("metrics_data", []) or []:
                    if base_metric_name(md.get("name", "")) == "Latency":
                        if md.get("success") is False:
                            over_budget += 1
                        break

        avg_latency = sum(latencies) / len(latencies) if latencies else None
        if latencies:
            sorted_lat = sorted(latencies)
            pcts: dict[str, int | None] = {
                "median": int(round(sorted_lat[len(sorted_lat) // 2])),
                "p95": _percentile(sorted_lat, 0.95),
                "p99": _percentile(sorted_lat, 0.99),
                "max": int(round(sorted_lat[-1])),
            }
        else:
            pcts = {"median": None, "p95": None, "p99": None, "max": None}
        pcts["over_budget_count"] = over_budget if has_latency_metric else None
        pcts["total"] = total

        # Availability is a REAL NUMBER, not a present key. The harness always
        # writes ``judge_cost_usd`` and writes null when the judge spend is
        # unknown, so testing for the key was always true and a run with unknown
        # judge cost was reported as available — rendering $0.00, the exact thing
        # this flag exists to prevent. The key-absence test dated from runs that
        # predated judge-cost tracking; there are none.
        judge_available = model_run.get("judge_cost_usd") is not None
        # Same for generation cost: the key is always written, so the per-case
        # re-summing fallback for "runs predating the top-level total" was dead.
        # Null means the runner had no pricing entry and refused to invent rates;
        # that is 0.0 here and is surfaced by ``pricing_available``.
        gen_cost = model_run.get("generation_cost_usd") or 0.0
        judge_cost = model_run.get("judge_cost_usd", 0.0) or 0.0
        total_cost = gen_cost + judge_cost
        cost_per_1k_gen = (gen_cost / total * 1000.0) if total else 0.0

        # Prompt size must INCLUDE cache reads. Bedrock's ``inputTokens``
        # counts only the uncached remainder — ``cacheReadInputTokens`` is
        # reported alongside it, not inside it — so summing ``input_tokens``
        # alone reported 2.0 tokens for a ~4,300-token cached prompt (measured:
        # input_tokens=2, cache_read_input_tokens=2320 per case). This field is
        # the caching diagnostic's "prompt-size proxy", so it was wrong exactly
        # where it is used. Cost is unaffected and always was: the harness
        # prices input, cache_read and cache_write separately.
        prompt_tokens = in_tokens + cache_read
        avg_in = (prompt_tokens / total) if total else 0.0
        avg_out = (out_tokens / total) if total else 0.0
        # Hit fraction against the WHOLE prompt, which is what the name says.
        # Previously this was cache_read / (cache_read + cache_write) — the
        # read/write MIX, which reads 0.904 on a prompt that was 99.9% served
        # from cache. It could distinguish "caching off" from "caching on" and
        # nothing in between.
        cache_ratio = (cache_read / prompt_tokens) if prompt_tokens else 0.0

        metric_rates = {
            base_metric_name(name): rate * 100.0
            for name, rate in per_model_metrics.get(model_id, {}).items()
        }

        out.append(ModelFacts(
            model_id=model_id,
            display_name=runner.short_model_name(model_id),
            total=total,
            passed=round(per_model_rates.get(model_id, 0.0) * total),
            pass_rate=per_model_rates.get(model_id, 0.0) * 100.0,
            metric_rates=metric_rates,
            avg_latency_ms=avg_latency,
            latency_percentiles=pcts,
            generation_cost_usd=gen_cost,
            judge_cost_usd=judge_cost,
            total_cost_usd=total_cost,
            cost_per_1k_gen=cost_per_1k_gen,
            judge_cost_available=judge_available,
            avg_input_tokens=avg_in,
            avg_output_tokens=avg_out,
            cache_read_ratio=cache_ratio,
            pricing_available=pricing_available,
        ))
    return out


def _metric_names_and_thresholds(results: list[dict]) -> tuple[list[str], dict[str, float]]:
    """Collect the run's metric names + thresholds, dynamically."""
    thresholds: dict[str, float] = {}
    for model_run in results:
        for case in model_run.get("results", []):
            for md in case.get("metrics_data", []) or []:
                name = base_metric_name(md.get("name", ""))
                if name and name not in thresholds:
                    t = md.get("threshold")
                    if isinstance(t, (int, float)):
                        thresholds[name] = float(t)
    return sorted(thresholds.keys()), thresholds


def _build_failed_cases(results: list[dict]) -> list[FailedCaseFacts]:
    """Walk every model run; emit one row per (case, failed metric) pair."""
    runner = EvalRunner()
    rows: list[FailedCaseFacts] = []
    for model_run in results:
        model_id = _model_id_of(model_run)
        display = runner.short_model_name(model_id)
        for case in model_run.get("results", []):
            metrics = case.get("metrics_data", []) or []
            failed_metrics = [md for md in metrics if md.get("success") is False]
            if not failed_metrics:
                continue
            meta = case.get("additional_metadata") or {}
            case_id = meta.get("id")
            if case_id is None:
                key = _case_key(case)
                case_id = key[1] if isinstance(key, tuple) and len(key) > 1 else key
            for md in failed_metrics:
                metric_name = base_metric_name(md.get("name", ""))
                rows.append(FailedCaseFacts(
                    case_id=case_id,
                    category=str(meta.get("category") or ""),
                    model_id=model_id,
                    model_display=display,
                    metric=metric_name,
                    is_latency=(metric_name == "Latency"),
                    input=str(case.get("input", "") or ""),
                    expected=str(case.get("expected_output", "") or ""),
                    actual=str(case.get("actual_output", "") or ""),
                    reason=str(md.get("reason", "") or ""),
                ))
    return rows


def _count_categories(results: list[dict]) -> int:
    """Count distinct category labels present on any case across all models.

    Uncategorized runs return 0; the chip strip then omits the "categories"
    chip rather than rendering a misleading "0 categories".
    """
    seen: set[str] = set()
    for model_run in results:
        for case in model_run.get("results", []):
            cat = (case.get("additional_metadata") or {}).get("category")
            if cat:
                seen.add(str(cat))
    return len(seen)


def _build_run_integrity(results: list[dict]) -> RunIntegrityFacts:
    """Sum the harness's per-model ``data_quality`` blocks into one struct.

    The harness's ``reliability`` label is judge-only: it reports generation
    failures separately. A generation-only failure (above-tolerance
    generation-failed ratio but clean judge metrics)
    must still report as non-"ok" so the report honestly reflects data quality.
    This function folds both domains into the final label.
    """
    from app.services.metric_error_analyzer import detect_generation_unreliable

    case_count = 0
    errored = dropped = nulls = unrecovered = 0
    reliabilities: list[str] = []
    for model_run in results:
        dq = model_run.get("data_quality") or {}
        # The harness writes ``total_cases`` as the dataset size; use the max
        # across models since they should all match. Fall back to len(results)
        # when missing.
        tc = dq.get("total_cases")
        if isinstance(tc, int) and tc > case_count:
            case_count = tc
        errored += int(dq.get("errored_count") or 0)
        dropped += int(dq.get("dropped_count") or 0)
        nulls += int(dq.get("null_count") or 0)
        unrecovered += int(dq.get("unrecovered_count") or 0)
        rel = dq.get("reliability")
        if isinstance(rel, str):
            reliabilities.append(rel)
    if case_count == 0:
        # Per-model run length as the fallback.
        for model_run in results:
            n = len(model_run.get("results", []))
            if n > case_count:
                case_count = n

    # Worst label wins: bad > degraded > ok.
    # Start with the judge-domain reliability verdict.
    rank = {"ok": 0, "degraded": 1, "bad": 2}
    label = "ok"
    for rel in reliabilities:
        if rank.get(rel, 0) > rank.get(label, 0):
            label = rel

    # Fold generation domain in: if any model is above the generation-failure
    # tolerance, rank that as "degraded" too. Without this, a generation-only
    # failure would report as "ok" after the domain split.
    gen_error = detect_generation_unreliable(results)
    if gen_error is not None:
        label = "degraded" if rank.get(label, 0) < rank["degraded"] else label

    effective = max(0, case_count - errored - dropped - unrecovered)
    return RunIntegrityFacts(
        case_count=case_count,
        errored=errored,
        dropped=dropped,
        null_count=nulls,
        unrecovered=unrecovered,
        effective_total=effective,
        reliability=label,
    )


@dataclass
class _DQBuckets:
    """Internal — three failure-attribution buckets, with full case detail.

    The renderer's lean fallback never reads the cases (just counts);
    keeping detail here for the narrator's input. We only persist counts to
    the report JSON via :class:`FailureBucketFacts`.
    """

    judge_false_negatives: list[FailedCaseFacts] = field(default_factory=list)
    data_review: list[FailedCaseFacts] = field(default_factory=list)
    model_gaps: list[FailedCaseFacts] = field(default_factory=list)


def _classify_failed_by_all(results: list[dict]) -> _DQBuckets:
    """Bucket cases that EVERY model failed, by output agreement."""
    runner = EvalRunner()
    by_case: dict[Any, list[tuple[str, dict]]] = {}
    model_ids_seen: list[str] = []
    for model_run in results:
        model_id = _model_id_of(model_run)
        if model_id not in model_ids_seen:
            model_ids_seen.append(model_id)
        display = runner.short_model_name(model_id)
        for r in model_run.get("results", []):
            by_case.setdefault(_case_key(r), []).append((display, r))

    out = _DQBuckets()
    n_models = len(model_ids_seen)
    if n_models < 2:
        return out

    def _norm(text: Any) -> str:
        return (text or "").strip()

    def _failed(row: dict) -> bool:
        return not row.get("success", False)

    def _judge_reason(rows: list[tuple[str, dict]]) -> str:
        for _name, row in rows:
            for md in row.get("metrics_data", []) or []:
                if not md.get("success", True) and md.get("reason"):
                    return str(md["reason"])
        return ""

    for case_key, rows in by_case.items():
        if len(rows) < n_models:
            continue
        if not all(_failed(row) for _name, row in rows):
            continue

        sample = rows[0][1]
        expected = _norm(sample.get("expected_output"))
        if not expected:
            continue
        outputs = {name: _norm(row.get("actual_output")) for name, row in rows}
        meta = sample.get("additional_metadata") or {}
        case_id = meta.get("id")
        if case_id is None:
            case_id = case_key[1] if isinstance(case_key, tuple) else case_key
        # Surface the FIRST model's row as the representative for the bucket.
        first_display, first_row = rows[0]
        first_model_id = _model_id_of({"model_id": "", "results": [first_row]})
        # The classifier doesn't have the model_id directly; reconstruct
        # from rows order against model_ids_seen.
        first_model_id = model_ids_seen[0]

        bucket_row = FailedCaseFacts(
            case_id=case_id,
            category=str(meta.get("category") or ""),
            model_id=first_model_id,
            model_display=first_display,
            metric="Correctness" if not any(
                base_metric_name(md.get("name", "")) == "Latency"
                for md in (first_row.get("metrics_data") or [])
                if md.get("success") is False
            ) else "Latency",
            is_latency=False,
            input=str(sample.get("input", "") or ""),
            expected=str(sample.get("expected_output", "") or ""),
            actual=str(first_row.get("actual_output", "") or ""),
            reason=_judge_reason(rows),
        )

        distinct_outputs = set(outputs.values())
        if len(distinct_outputs) > 1:
            out.model_gaps.append(bucket_row)
        elif next(iter(distinct_outputs)) == expected:
            out.judge_false_negatives.append(bucket_row)
        else:
            out.data_review.append(bucket_row)

    return out


def _load_prompt(project_dir: Path, prompt_version: str) -> tuple[str, str]:
    """Read the prompt content for the appendix.

    ``prompt_version`` comes from run-file *content*
    (``hyperparameters.prompt_version``), so it is untrusted whenever the
    run file itself is. A prompt version is always a bare stem naming one
    file in ``prompts/`` — ``safe_basename_path`` enforces that, matching
    the containment check ``mcp_tools/dispatch.py`` applies to the same
    field. An unsafe value falls through to the newest-prompt fallback
    below, so a poisoned run file degrades to the default rather than
    reading outside the project.
    """
    prompts_dir = project_dir / "prompts"
    if prompt_version:
        candidate = safe_basename_path(prompts_dir, f"{prompt_version}.md")
        if candidate is not None and candidate.exists():
            return prompt_version, candidate.read_text()

    if prompts_dir.exists():
        files = [p for p in prompts_dir.iterdir() if p.is_file() and p.suffix == ".md"]
        if files:
            def _key(p: Path) -> int:
                stem = p.stem
                if stem == "baseline":
                    return -1
                if stem.startswith("v") and stem[1:].isdigit():
                    return int(stem[1:])
                return -2
            best = max(files, key=_key)
            return best.stem, best.read_text()
    return "", ""


_CHANGES_RE = re.compile(r"Changes:\s*(.+?)(?:\n\s*(?:Targeting|Strategy|Why):|-->|$)", re.S)


def _parse_prompt_changes(project_dir: Path, version: str) -> str | None:
    """Extract the ``Changes:`` text from a version's prompt header comment.

    Preserves newlines (each line is a separate change) and collapses
    horizontal whitespace; the renderer's per-line splitter relies on this.

    ``version`` is untrusted for the same reason as in ``_load_prompt``
    — it reaches here both from a run file's ``hyperparameters`` and from
    the ``list_runs`` glob — so it goes through the same containment check.
    """
    if not version:
        return None
    path = safe_basename_path(project_dir / "prompts", f"{version}.md")
    if path is None or not path.exists():
        return None
    head = path.read_text()[:2000]
    match = _CHANGES_RE.search(head)
    if not match:
        return None
    raw = match.group(1)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return cleaned or None


# --------------------------------------------------------------------------
# Recommendation tagging — descriptive verdict label per model
# --------------------------------------------------------------------------

def _verdict_tag(
    m: ModelFacts,
    targets: dict[str, float],
    is_recommended: bool,
) -> str:
    """Return the verdict-column label for a model.

    Labels: ``Recommended`` (green) / ``Strong, costly`` / ``Viable`` /
    ``Below targets`` / ``Below latency target``. No ship/no-ship gate;
    just descriptive tags.
    """
    if not targets:
        return "Top pass rate" if is_recommended else ""
    cleared = _model_clears_targets_facts(m, targets)
    if is_recommended and cleared:
        return "Recommended"
    if cleared:
        # A non-recommended cleared model is the redundancy / fallback engine.
        return "Strong, costly"
    # Below targets — distinguish latency-only miss for honesty.
    short = _worst_shortfall_facts(m, targets)
    if short and short[0] == "Latency":
        return "Below latency target"
    if m.pass_rate >= 95.0:
        return "Viable"
    return "Below targets"


def _model_clears_targets_facts(m: ModelFacts, targets: dict[str, float]) -> bool:
    for name, target in targets.items():
        rate = m.metric_rates.get(name)
        if rate is None or rate < target * 100.0:
            return False
    return True


def _worst_shortfall_facts(
    m: ModelFacts, targets: dict[str, float]
) -> tuple[str, float, float, int, int] | None:
    """``(metric, rate%, target%, missed, total)`` for the worst miss."""
    worst: tuple[str, float, float, int, int] | None = None
    worst_gap = 0.0
    total = m.total
    for name, target in targets.items():
        target_pct = target * 100.0
        rate = m.metric_rates.get(name)
        rate_pct = rate if rate is not None else 0.0
        gap = target_pct - rate_pct
        if gap > 0 and gap > worst_gap:
            worst_gap = gap
            passed_for_metric = round(rate_pct / 100.0 * total) if total else 0
            missed = max(total - passed_for_metric, 0)
            worst = (name, rate_pct, target_pct, missed, total)
    return worst


def _pick_recommended_model(
    models: list[ModelFacts], targets: dict[str, float]
) -> ModelFacts | None:
    """Pick the recommended model: cheapest among cleared, else best pass rate."""
    if not models:
        return None
    if targets:
        cleared = [m for m in models if _model_clears_targets_facts(m, targets)]
        if cleared:
            # Cheapest by cost_per_1k_gen, ties broken by latency.
            return min(
                cleared,
                key=lambda m: (
                    m.cost_per_1k_gen,
                    m.avg_latency_ms if m.avg_latency_ms is not None else float("inf"),
                ),
            )
    return max(
        models,
        key=lambda m: (
            m.pass_rate,
            -m.cost_per_1k_gen,
            -(m.avg_latency_ms or 0),
        ),
    )


# --------------------------------------------------------------------------
# RunFacts builder (per-run report)
# --------------------------------------------------------------------------

def _iteration_for_run(
    state: OptimizationState | None, run_filename: str
) -> int | None:
    """Look up the 1-based iteration index whose ``run_id`` matches the run."""
    if state is None:
        return None
    for it in state.iterations:
        if it.get("run_id") == run_filename:
            value = it.get("iteration")
            if isinstance(value, int):
                return value
    return None


def _latest_run_for_version(
    project_dir: Path, runner: EvalRunner, version: str,
) -> str | None:
    """Return the filename of the latest-by-timestamp run for a given prompt version."""
    try:
        all_runs = runner.list_runs(project_dir)
    except Exception:
        return None
    candidates = [r for r in all_runs if r.get("prompt_version") == version]
    if not candidates:
        return None
    latest = max(candidates, key=lambda r: r.get("timestamp") or "")
    return latest.get("_file")


def _journey_from_run_files(
    project_dir: Path, runner: EvalRunner
) -> list[JourneyVersionFacts]:
    """Reconstruct the journey from disk: one entry per prompt version.

    For each ``vN`` (and ``baseline``) with at least one run on disk, take
    the latest run for that version and compute the best-model pass rate.
    Δ is vs the immediately-prior version in version order.

    State.json's iterations are NOT consulted — they can hold a single
    iteration while 10+ versions exist on disk. Run files are the journey's
    source of truth.
    """
    try:
        all_runs = runner.list_runs(project_dir)
    except Exception:
        return []

    # Group by version. ``list_runs`` already exposes ``prompt_version``
    # on the run summary; the model_runs entries don't carry it.
    by_version: dict[str, list[dict]] = {}
    for r in all_runs:
        model_runs = r.get("model_runs") or []
        if not model_runs:
            continue
        version = r.get("prompt_version") or "unknown"
        by_version.setdefault(version, []).append(r)

    def _vsort(version: str) -> int:
        if version == "baseline":
            return -1
        if version.startswith("v") and version[1:].isdigit():
            return int(version[1:])
        return 9999

    versions_ordered = sorted(by_version.keys(), key=_vsort)
    out: list[JourneyVersionFacts] = []
    prev_best: float | None = None
    for v in versions_ordered:
        runs_for_v = by_version[v]
        # Newest by timestamp.
        latest = max(runs_for_v, key=lambda r: r.get("timestamp") or "")
        # Compute best-model pass rate from the run summary's model_runs.
        # ``list_runs`` exposes per-model rows as
        # ``{test_run_id, model, total, passed, per_metric}``.
        per_model: dict[str, float] = {}
        for mr in latest.get("model_runs") or []:
            mid = mr.get("model") or mr.get("model_id") or "unknown"
            display = runner.short_model_name(mid)
            tot = mr.get("total") or 0
            passed = mr.get("passed") or 0
            rate = (passed / tot * 100.0) if tot else 0.0
            per_model[display] = rate
        best = max(per_model.values()) if per_model else 0.0
        delta = None if prev_best is None else best - prev_best

        # Best model's gen cost/1k as the journey cost signal — needs the
        # full run-file read since list_runs() summary drops cost. Skip for
        # now (None is acceptable; the renderer doesn't require it).
        cost_per_1k = None
        avg_in = None

        out.append(JourneyVersionFacts(
            version=v,
            best_pass_rate=best,
            delta=delta,
            per_model_pass_rates=per_model,
            cost_per_1k_gen=cost_per_1k,
            avg_input_tokens=avg_in,
            what_changed=_parse_prompt_changes(project_dir, v),
        ))
        prev_best = best
    return out


def _load_metric_audit(project_dir: Path, run_filename: str) -> dict[str, Any] | None:
    """Read a run's metric-audit sidecar, or ``None`` when there isn't a usable one.

    Absent for any run that finished before the audit existed, and for any run
    whose audit could not be written, so callers must treat ``None`` as "nothing
    known" rather than "nothing wrong".
    """
    from app.services.metric_disagreement import sidecar_path
    path = sidecar_path(project_dir / "output" / run_filename)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("[run-trust] unreadable audit %s: %s", path.name, e)
        return None
    return audit if isinstance(audit, dict) else None


def load_run_trust(project_dir: Path, run_filename: str) -> RunTrustFacts:
    """Build a finished run's trust verdict straight from its artifacts on disk.

    A cheaper door into the same verdict :func:`build_run_facts` puts on
    ``RunFacts``, for callers that need only this and not a whole facts pack —
    the post-eval coordinator turn being the one that matters, since it is what
    decides whether to cull a model or declare convergence.

    Deliberately shares ``_build_run_integrity`` and ``_load_metric_audit`` with
    the report path rather than re-deriving anything: the agent and the report
    must never be able to disagree about whether a run is trustworthy.

    Args:
        project_dir: Project root.
        run_filename: Bare ``eval_results_*.json`` name inside ``output/``.

    Returns:
        The verdict. An unreadable run file yields a ``severity == "ok"`` verdict
        with no reasons — the same "nothing known" the renderers treat as
        unchecked, never as proof of health.
    """
    run_path = project_dir / "output" / run_filename
    try:
        results = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("[run-trust] cannot read %s: %s", run_filename, e)
        return RunTrustFacts()
    if not isinstance(results, list):
        return RunTrustFacts()

    integrity = _build_run_integrity(results)
    case_count = max(
        (len(r.get("results", [])) for r in results if isinstance(r, dict)),
        default=integrity.case_count,
    )
    return build_run_trust(
        run_integrity=integrity,
        metric_audit=_load_metric_audit(project_dir, run_filename),
        case_count=case_count,
    )


def build_run_facts(project_dir: Path, run_filename: str | None = None) -> RunFacts:
    """Build the deterministic facts-pack for one eval run.

    The narrator agent reads this pack and writes prose around it. The
    renderer can also render it directly (lean fallback) when no narrator
    output is available.

    Args:
        project_dir: Project root.
        run_filename: Explicit run, or None for the latest.

    Returns:
        Fully populated :class:`RunFacts`.
    """
    project_dir = Path(project_dir)
    config = _load_project_config(project_dir)

    runner = EvalRunner()
    resolved_filename, results = _select_run(runner, project_dir, run_filename)

    has_latency_metric = _detect_latency_metric(results)
    models = _build_model_facts(results, has_latency_metric)
    metric_names, thresholds = _metric_names_and_thresholds(results)
    metric_criteria = _parse_metric_definitions(project_dir)

    latency_budget_ms: int | None = None
    if has_latency_metric:
        latency_budget_ms = (
            _parse_latency_budget_ms_from_criteria(metric_criteria.get("Latency", ""))
            or _detect_latency_budget_ms_from_runs(results)
        )

    failed_cases = _build_failed_cases(results)
    dq_buckets = _classify_failed_by_all(results)
    failure_buckets = FailureBucketFacts(
        judge_false_negatives=len(dq_buckets.judge_false_negatives),
        data_review=len(dq_buckets.data_review),
        model_gaps=len(dq_buckets.model_gaps),
    )
    run_integrity = _build_run_integrity(results)

    state = OptimizationState.load(project_dir)
    targets = dict(state.target_metric_scores) if state else {}

    recommended = _pick_recommended_model(models, targets)
    rec_id = recommended.model_id if recommended else None
    recommendations = [
        RecommendationFacts(
            model_id=m.model_id,
            tag=_verdict_tag(m, targets, is_recommended=(m.model_id == rec_id)),
        )
        for m in models
    ]

    case_count = max((m.total for m in models), default=run_integrity.case_count)

    # Built here, not in the renderers: both reports and the agent must read one
    # verdict, and the exec report gets it for free by inlining ``winner_run``.
    run_trust = build_run_trust(
        run_integrity=run_integrity,
        metric_audit=_load_metric_audit(project_dir, resolved_filename),
        case_count=case_count,
    )

    prompt_version = (results[0].get("hyperparameters") or {}).get("prompt_version", "")
    prompt_label, prompt_content = _load_prompt(project_dir, prompt_version)
    prompt_changes = _parse_prompt_changes(project_dir, prompt_version)

    # Δ vs prior version: walk run files (ignore state.json — see
    # _journey_from_run_files for why).
    prior_step: dict[str, Any] | None = None
    prior_version_label: str | None = None
    journey = _journey_from_run_files(project_dir, runner)
    for i, step in enumerate(journey):
        if step.version == prompt_version and i > 0:
            prior = journey[i - 1]
            prior_step = {
                "version": prior.version,
                "pass_rate": prior.best_pass_rate,
                "delta": step.delta,
            }
            prior_version_label = prior.version
            break

    # Per-model deltas: locate the prior version's latest run on disk,
    # build its model facts, and compute (this − prior) per model and per
    # metric. None when no prior run exists for the model (e.g. baseline
    # render, or a model added partway through the journey).
    if prior_version_label:
        try:
            prior_run_filename = _latest_run_for_version(
                project_dir, runner, prior_version_label,
            )
        except Exception:
            prior_run_filename = None
        if prior_run_filename:
            try:
                _, prior_results = _select_run(runner, project_dir, prior_run_filename)
                prior_models = _build_model_facts(prior_results, has_latency_metric)
                prior_by_id = {pm.model_id: pm for pm in prior_models}
                for m in models:
                    pm = prior_by_id.get(m.model_id)
                    if pm is None:
                        continue
                    m.pass_rate_delta = round(m.pass_rate - pm.pass_rate, 1)
                    m.metric_rate_deltas = {
                        name: round(m.metric_rates[name] - pm.metric_rates.get(name, 0.0), 1)
                        for name in m.metric_rates
                        if name in pm.metric_rates
                    }
            except Exception:
                # Best-effort — never block the report on prior-run read.
                logger.exception(
                    "Failed to compute per-model deltas vs %s for %s",
                    prior_version_label, resolved_filename,
                )

    iteration = _iteration_for_run(state, resolved_filename)

    # Trophy match → is_winning_run.
    trophy = _pick_trophy_run(project_dir, runner, state)
    is_winning_run = (trophy == resolved_filename)

    hyperparameters: dict[str, Any] = {}
    if results:
        hp = results[0].get("hyperparameters") or {}
        for k in ("temperature", "max_tokens", "top_p"):
            if k in hp:
                hyperparameters[k] = hp[k]

    return RunFacts(
        schema_version=REPORT_SCHEMA_VERSION,
        project_name=config.name,
        project_description=config.description or "",
        run_filename=resolved_filename,
        run_timestamp=results[0].get("timestamp", ""),
        run_source=results[0].get("source", "manual"),
        prompt_label=prompt_label,
        prompt_iteration=iteration,
        prompt_content=prompt_content,
        prompt_changes=prompt_changes,
        case_count=case_count,
        category_count=_count_categories(results),
        metric_names=metric_names,
        metric_thresholds=thresholds,
        metric_criteria=metric_criteria,
        latency_budget_ms=latency_budget_ms,
        has_latency_metric=has_latency_metric,
        hyperparameters=hyperparameters,
        models=models,
        recommendations=recommendations,
        recommended_model_id=rec_id,
        failed_cases=failed_cases,
        failure_buckets=failure_buckets,
        run_integrity=run_integrity,
        run_trust=run_trust,
        prior_step=prior_step,
        is_winning_run=is_winning_run,
    )


def build_exec_facts(project_dir: Path) -> ExecFacts:
    """Build the deterministic facts-pack for the project's executive report.

    Walks the project's full run history, picks the trophy winner, and
    inlines that run's per-run facts. The journey is reconstructed from run
    files (one row per prompt version), NOT from ``state.json``.

    Args:
        project_dir: Project root.

    Returns:
        Fully populated :class:`ExecFacts`.
    """
    from datetime import datetime, timezone

    project_dir = Path(project_dir)
    config = _load_project_config(project_dir)

    runner = EvalRunner()
    state = OptimizationState.load(project_dir)
    trophy_filename = _resolve_exec_winner(project_dir, runner, state)
    if trophy_filename is None:
        raise ValueError(f"Project {project_dir} has no eval runs")

    winner_run = build_run_facts(project_dir, trophy_filename)

    journey = _journey_from_run_files(project_dir, runner)

    # Humanized stamp: "v8 run · June 12, 2026". Never the raw filename —
    # users don't want "eval_results_v8_20260612_223238.json" in their face.
    label_parts: list[str] = []
    if winner_run.prompt_label:
        label_parts.append(f"{winner_run.prompt_label} run")
    if winner_run.run_timestamp:
        from datetime import datetime as _dt
        try:
            ts = _dt.fromisoformat(winner_run.run_timestamp.replace("Z", "+00:00"))
            label_parts.append(f"{ts.strftime('%B')} {ts.day}, {ts.year}")
        except (ValueError, TypeError):
            pass
    generated_from_label = " · ".join(label_parts) or "winning run"

    return ExecFacts(
        schema_version=REPORT_SCHEMA_VERSION,
        project_name=config.name,
        project_description=config.description or "",
        case_count=winner_run.case_count,
        category_count=winner_run.category_count,
        metric_names=winner_run.metric_names,
        metric_thresholds=winner_run.metric_thresholds,
        metric_criteria=winner_run.metric_criteria,
        latency_budget_ms=winner_run.latency_budget_ms,
        journey=journey,
        winner_version=winner_run.prompt_label,
        winner_run_filename=trophy_filename,
        winner_run=winner_run,
        generated_from_label=generated_from_label,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Report-JSON sidecar I/O
# --------------------------------------------------------------------------

def _reports_dir(project_dir: Path) -> Path:
    """Project's reports directory; created if missing."""
    out = project_dir / "output" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_json_path(project_dir: Path, scope: str, run_filename: str | None = None) -> Path:
    """Resolve the on-disk path for a scope's report JSON.

    Args:
        project_dir: Project root.
        scope: ``"run"`` or ``"executive"``.
        run_filename: Required for ``"run"`` scope; ignored otherwise.
    """
    if scope == "executive":
        return _reports_dir(project_dir) / "executive.report.json"
    if scope == "run":
        if not run_filename:
            raise ValueError("run scope requires run_filename")
        stem = Path(run_filename).stem
        return _reports_dir(project_dir) / f"{stem}.report.json"
    raise ValueError(f"Unknown scope: {scope}")


def _facts_to_jsonable(facts: RunFacts | ExecFacts) -> dict[str, Any]:
    """Serialize a facts dataclass to a JSON-safe dict.

    Nested dataclasses are converted via ``asdict``. Percentages and
    deltas are rounded to 1 decimal at serialization time so the
    narrator agent (which quotes provided numbers verbatim) can't leak
    raw IEEE-754 floats like ``93.33333333333333%`` into prose. Cost
    values use 4 decimals (sub-cent precision matters for $/1k figures).
    """
    return _round_for_serialization(asdict(facts))


#: Percentage-POINT fields (0–100). Rounded to 1 decimal.
#:
#: ``cache_read_ratio`` is deliberately NOT here despite the name. It is a 0–1
#: fraction, and 1-decimal rounding collapses 0.999 to 1.0 and 0.94 to 0.9 —
#: which would erase the resolution the field exists to provide, since real
#: values cluster above 0.9. It falls through to 4 decimals.
_PERCENT_FIELDS = {
    "pass_rate", "best_pass_rate", "delta",
}
_PERCENT_DICT_FIELDS = {
    "metric_rates", "per_model_pass_rates",
}
_COST_FIELDS = {
    "generation_cost_usd", "judge_cost_usd", "total_cost_usd",
    "cost_per_1k_gen", "cost_per_1k", "avg_input_tokens",
    "avg_output_tokens",
}


def _round_for_serialization(node: Any) -> Any:
    """Walk a JSON-safe tree, rounding numeric fields the narrator quotes."""
    if isinstance(node, dict):
        rounded: dict[str, Any] = {}
        for k, v in node.items():
            if isinstance(v, float):
                if k in _PERCENT_FIELDS:
                    rounded[k] = round(v, 1)
                elif k in _COST_FIELDS:
                    rounded[k] = round(v, 4)
                else:
                    rounded[k] = round(v, 4)
            elif k in _PERCENT_DICT_FIELDS and isinstance(v, dict):
                rounded[k] = {
                    mk: round(mv, 1) if isinstance(mv, float) else mv
                    for mk, mv in v.items()
                }
            else:
                rounded[k] = _round_for_serialization(v)
        return rounded
    if isinstance(node, list):
        return [_round_for_serialization(x) for x in node]
    return node


def write_report_json(
    project_dir: Path,
    scope: str,
    facts: RunFacts | ExecFacts,
    narrative: dict[str, Any] | None,
    run_filename: str | None = None,
) -> Path:
    """Write the report JSON sidecar atomically.

    Schema:
    ```
    {
      "schema_version": int,
      "scope": "run" | "executive",
      "generated_at": ISO timestamp,
      "facts": {...},          ← deterministic, server-computed
      "narrative": {...} | null,  ← agent prose, optional
    }
    ```
    """
    from datetime import datetime, timezone
    import os
    import tempfile

    target = report_json_path(project_dir, scope, run_filename)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facts": _facts_to_jsonable(facts),
        "narrative": narrative,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        mode="w",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        # The name is read here, but the file is replaced only after this block,
        # where the context manager has already closed and flushed it.
        # nosemgrep: tempfile-without-flush
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)
    return target


def read_report_json(
    project_dir: Path, scope: str, run_filename: str | None = None
) -> dict[str, Any] | None:
    """Read a report JSON sidecar, or None when missing/unparseable."""
    path = report_json_path(project_dir, scope, run_filename)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != REPORT_SCHEMA_VERSION:
        return None
    return data


# --------------------------------------------------------------------------
# Format helpers
# --------------------------------------------------------------------------

def _esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation."""
    return html.escape(str(value), quote=True)


def _fmt_usd(value: float) -> str:
    """USD with two decimals (``$2.32``)."""
    return f"${value:,.2f}"


def _fmt_date(iso: str) -> str:
    """Humanize an ISO-8601 timestamp."""
    from datetime import datetime
    text = (iso or "").strip()
    if not text:
        return ""
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except (ValueError, TypeError):
        return text


def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


def _fmt_pct_precise(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_ms(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value)):,} ms"


def _fmt_ms_int_compact(value: int | None) -> str:
    """Latency cell: ``821`` or ``1,834`` (no unit; column header carries it)."""
    if value is None:
        return "n/a"
    return f"{value:,}"

