# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-targeted seeding prompt builder for synthetic data generation.

This module turns the structured output of ``failure_analyzer.analyze_failures``
into two artefacts the synthesizer needs when running the ``failure_targeted``
strategy:

    * ``build_failure_seed_prompt`` — derives the four free-text fields
      DeepEval's styling config expects (``subject`` / ``task`` /
      ``output_format`` / ``instructions``). The first three shape generation;
      the last is appended to the ``task`` by the MCP tool so the generator
      sees concrete failing-case examples to imitate/avoid.
    * ``failures_to_seed_goldens`` — picks the top-N worst failing cases from
      the analysis and materialises them as DeepEval ``Golden`` objects.
      Priority is given to the category (or metric) with the lowest pass rate
      so the resulting seeds stress-test the model's weakest area.

The module is deliberately I/O-free and deterministic — the caller owns the
filesystem side. Raises ``ValueError`` on malformed / empty analyses so the
MCP tool layer can translate the failure into an ``mcp_error`` cleanly.
"""

from __future__ import annotations

from typing import Any

from deepeval.dataset import Golden


def _pick_worst_model_analysis(analysis: dict, run_index: int | None = None) -> dict:
    """Return one model analysis, raising ``ValueError`` on degenerate input.

    Args:
        analysis: Output of ``analyze_failures``. Must have
            ``model_analyses`` populated and must not carry an ``error``.
        run_index: Optional 0-based index into ``model_analyses``. If
            ``None`` we default to the first (primary) model analysis.

    Returns:
        The chosen per-model analysis dict.

    Raises:
        ValueError: If the analysis is empty, errored, or the index is
            out of range.
    """
    if not analysis or "error" in analysis:
        raise ValueError(
            f"Cannot build failure-targeted prompt from errored analysis: "
            f"{analysis.get('error') if analysis else 'empty input'}"
        )
    runs = analysis.get("model_analyses") or []
    if not runs:
        raise ValueError("Cannot build failure-targeted prompt: no model_analyses present")

    if run_index is None:
        return runs[0]
    if run_index < 0 or run_index >= len(runs):
        raise ValueError(
            f"run_index {run_index} out of range for model_analyses (len={len(runs)})"
        )
    return runs[run_index]


def _worst_category(model_analysis: dict) -> tuple[str | None, float]:
    """Return the category with the lowest pass rate (ties broken by size).

    Categories with zero totals are skipped. Returns ``(None, 1.0)`` when no
    category info is available.
    """
    by_cat: dict[str, dict[str, Any]] = model_analysis.get("failure_by_category") or {}
    candidates = [
        (cat, stats) for cat, stats in by_cat.items() if stats.get("total", 0) > 0
    ]
    if not candidates:
        return None, 1.0
    # Primary sort: lowest pass_rate. Secondary: largest "total" so a widely-
    # failing category wins over a single-case outlier with pass_rate=0.0.
    candidates.sort(key=lambda kv: (kv[1].get("pass_rate", 1.0), -kv[1].get("total", 0)))
    cat, stats = candidates[0]
    return cat, float(stats.get("pass_rate", 1.0))


def _worst_metric(model_analysis: dict) -> tuple[str | None, float]:
    """Return the metric with the lowest pass rate."""
    by_metric: dict[str, dict[str, Any]] = model_analysis.get("failure_by_metric") or {}
    candidates = [
        (name, stats) for name, stats in by_metric.items() if stats.get("total", 0) > 0
    ]
    if not candidates:
        return None, 1.0
    candidates.sort(key=lambda kv: (kv[1].get("pass_rate", 1.0), -kv[1].get("total", 0)))
    name, stats = candidates[0]
    return name, float(stats.get("pass_rate", 1.0))


def _infer_output_format(failed_cases: list[dict]) -> str:
    """Heuristic output-format label derived from expected_output shapes.

    If more than half of the expected outputs start with ``{``, we treat the
    target as JSON. Otherwise we describe it as short free-text. Kept
    deliberately simple — the generator only needs a rough hint.
    """
    if not failed_cases:
        return "short free-text answer"
    total = 0
    json_like = 0
    for c in failed_cases:
        expected = (c.get("expected_output") or "").strip()
        if not expected:
            continue
        total += 1
        if expected.startswith("{") or expected.startswith("["):
            json_like += 1
    if total > 0 and json_like / total > 0.5:
        return "JSON object matching the structure of the example expected outputs"
    return "short free-text answer, same style/length as the examples"


def _sorted_failed_cases(model_analysis: dict) -> list[dict]:
    """Return failed cases ordered worst-category-first.

    Cases whose category has the lowest pass rate come first, followed by
    cases in the next-worst category, and so on. Within a category we keep
    the original order — ``failure_analyzer`` already emits cases in the
    order DeepEval produced them.
    """
    by_cat: dict[str, dict[str, Any]] = model_analysis.get("failure_by_category") or {}
    cases: list[dict] = list(model_analysis.get("failed_case_details") or [])
    if not cases:
        return []

    # Rank categories by pass_rate ascending; unknown-category cases sink
    # to the end so they don't jump ahead of concretely-worst ones.
    def _category_rank(case: dict) -> tuple[float, int, int]:
        cat = case.get("category")
        stats = by_cat.get(cat) if cat else None
        if not stats:
            return (1.0, 0, cases.index(case))
        return (
            float(stats.get("pass_rate", 1.0)),
            -int(stats.get("total", 0)),
            cases.index(case),
        )

    return sorted(cases, key=_category_rank)


def build_failure_seed_prompt(analysis: dict, max_examples: int = 6) -> dict:
    """Turn a ``failure_analyzer`` result into a seeding prompt bundle.

    Produces the subject/task/output_format strings that DeepEval's
    ``StylingConfig`` understands, plus an ``instructions`` block that lists
    up to ``max_examples`` concrete failing-case exemplars. The MCP tool
    layer is expected to fold ``instructions`` into the ``task`` free-text
    before calling ``generate_from_scratch``.

    Args:
        analysis: Output of ``analyze_failures``. Must contain at least one
            model analysis with ``failed_case_details``.
        max_examples: Upper bound on the number of failing-case examples
            embedded in ``instructions``. Defaults to 6.

    Returns:
        Dict with keys ``subject``, ``task``, ``output_format``,
        ``instructions``.

    Raises:
        ValueError: If the analysis is empty or contains an error.
    """
    model_analysis = _pick_worst_model_analysis(analysis)
    worst_cat, cat_pass_rate = _worst_category(model_analysis)
    worst_metric_name, metric_pass_rate = _worst_metric(model_analysis)

    # Prefer describing the subject by the worst-performing category when we
    # have one — agent and user both think in terms of "fix the spelling
    # category", not "fix ExactMatch". Fall back to the metric if categories
    # are missing (which happens when upstream data doesn't tag cases).
    if worst_cat is not None:
        # Collect up to two additional struggling categories so the scenario
        # string reflects reality without becoming a novel.
        by_cat = model_analysis.get("failure_by_category") or {}
        struggling = sorted(
            (c for c, s in by_cat.items() if s.get("pass_rate", 1.0) < 1.0 and s.get("total", 0) > 0),
            key=lambda c: (by_cat[c]["pass_rate"], -by_cat[c].get("total", 0)),
        )[:3]
        subject = f"Cases in categories: {', '.join(struggling) if struggling else worst_cat}"
    elif worst_metric_name is not None:
        subject = f"Cases that fail the {worst_metric_name} metric"
    else:
        subject = "Cases that fail the current eval"

    sorted_cases = _sorted_failed_cases(model_analysis)
    examples = sorted_cases[:max_examples]
    output_format = _infer_output_format(sorted_cases)

    # Generic task description — we don't know the user's domain precisely
    # so keep it shape-focused. The embedded examples carry the specifics.
    task = (
        "Produce test cases the target model should handle. Mirror the "
        "style and difficulty of the failing examples below so the generated "
        "cases stress-test the same weaknesses."
    )

    instruction_lines: list[str] = []
    if worst_cat is not None:
        instruction_lines.append(
            f"The weakest category is '{worst_cat}' with pass rate "
            f"{cat_pass_rate:.0%}. Focus new cases on scenarios like these:"
        )
    elif worst_metric_name is not None:
        instruction_lines.append(
            f"The weakest metric is '{worst_metric_name}' with pass rate "
            f"{metric_pass_rate:.0%}. Focus new cases on scenarios like these:"
        )
    else:
        instruction_lines.append(
            "The target model failed the cases below. Focus new cases on "
            "scenarios like these:"
        )

    for idx, case in enumerate(examples, start=1):
        failing_metrics = case.get("failing_metrics") or {}
        metric_names = ", ".join(failing_metrics.keys()) or "unknown metric"
        instruction_lines.append(
            f"Example {idx}:\n"
            f"  input: {case.get('input')!r}\n"
            f"  expected_output: {case.get('expected_output')!r}\n"
            f"  failed metric(s): {metric_names}"
        )

    instructions = "\n\n".join(instruction_lines)

    return {
        "subject": subject,
        "task": task,
        "output_format": output_format,
        "instructions": instructions,
    }


def failures_to_seed_goldens(analysis: dict, max_seeds: int = 10) -> list[Golden]:
    """Convert the top failing cases into DeepEval ``Golden`` objects.

    Cases are prioritised by worst-category-first ordering (see
    ``_sorted_failed_cases``) and truncated to ``max_seeds``. The returned
    Goldens carry the original ``input`` and ``expected_output`` plus an
    ``additional_metadata`` dict that includes the ``category`` label and a
    list of ``failing_metrics`` names so downstream consumers can trace
    provenance.

    Args:
        analysis: Output of ``analyze_failures``.
        max_seeds: Upper bound on returned Goldens. Defaults to 10.

    Returns:
        A list of ``Golden`` objects, possibly empty if no failing cases
        exist. May be shorter than ``max_seeds``.

    Raises:
        ValueError: If the analysis is empty or contains an error.
    """
    model_analysis = _pick_worst_model_analysis(analysis)
    sorted_cases = _sorted_failed_cases(model_analysis)

    seeds: list[Golden] = []
    for case in sorted_cases[:max_seeds]:
        failing_metrics = list((case.get("failing_metrics") or {}).keys())
        metadata: dict[str, Any] = {
            "source": "failure_targeted",
            "failing_metrics": failing_metrics,
        }
        if case.get("category") is not None:
            metadata["category"] = case["category"]
        if case.get("case_id") is not None:
            metadata["seed_case_id"] = case["case_id"]
        seeds.append(
            Golden(
                input=case.get("input") or "",
                expected_output=case.get("expected_output") or "",
                additional_metadata=metadata,
            )
        )
    return seeds
