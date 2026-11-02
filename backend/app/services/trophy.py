# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trophy scoring — THE single implementation.

One place computes "how good is this candidate against the configured
targets." Before this module, the scoring existed twice: here in Python
(``report.py``, feeding the exec winner and ``is_winning_run``) and a
hand-synced TypeScript copy in ``OptimizationRunDetail.tsx`` (feeding the
panel's trophy) whose header admitted the coupling: "the dashboard relies on
this exact scoring logic so reports never disagree." Two copies with no
shared test is how the report and the panel drift apart silently.

The backend computes scores and serves them on the wire (run
summaries and iteration rows each carry ``trophy_score``); the frontend only
ORDERS what it is given (``pickTrophyIndex``). The golden vector at
``backend/tests/fixtures/trophy_golden.json`` is executed by BOTH suites —
pytest asserts these functions produce the expected tuples, vitest asserts
the frontend comparator picks the expected winner from them.

Targets come from HARNESS CONFIG, the criteria owner — not from
``OptimizationState``'s dispatch-time snapshot. The panel already read
config; the report read the snapshot, so a user who edited targets after
dispatch got a report ranked by one bar and a panel ranked by another.

Score semantics (ordering is lexicographic on the tuple, later-wins on tie):

* ``at_target`` — how many evaluated models met EVERY configured target.
* ``best_rate`` — highest single-model pass rate (0-1).
* ``avg_rate``  — mean pass rate across evaluated models (0-1).

Rows and runs WITHOUT per-metric data score ``at_target=0``. They used to be
compared against ``min(targets)`` — the loosest configured bar — so that a row
recorded before per-metric data existed was not silently unrankable. That traded
one silent failure for a worse one: clearing the single easiest target is not
"met EVERY configured target", so such a candidate could out-rank a model that
genuinely cleared all of them but one. Pre-release there is no such data, and a
candidate that cannot be scored should lose, not be scored generously.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class TrophyScore(TypedDict):
    """Wire shape for a candidate's trophy score."""

    at_target: int
    best_rate: float
    avg_rate: float


def resolve_trophy_targets(project_dir: Path) -> dict[str, float]:
    """Targets for trophy ranking, from the criteria owner: harness config.

    Empty dict when config is absent, unreadable or has no targets — callers
    treat that as "no ranking possible" (no trophy, exec winner falls back to
    best rate), never as a fabricated default. Listing eval runs and building
    the executive report both call this, so a broken config.py must not fail
    either; the agent is told about the file in chat and can repair it.
    """
    from app.services.harness_config import load_harness_config

    try:
        harness = load_harness_config(project_dir) if project_dir.exists() else None
    except (SyntaxError, ValueError) as exc:
        logger.warning("[trophy] src/config.py unreadable, no targets: %s", exc)
        return {}
    if harness is None:
        return {}
    return dict(harness.target_metric_scores or {})


def model_meets_targets(
    metric_rates: dict[str, float], targets: dict[str, float]
) -> bool:
    """True when ``metric_rates`` clears every target (suffix-tolerant).

    Suffix tolerance matches DeepEval's decorated metric names: a target
    named ``Answer Correctness`` matches a rate keyed ``Answer Correctness
    [GEval]``.
    """
    for name, target in targets.items():
        match = next(
            (k for k in metric_rates if k == name or k.startswith(name + " [")),
            None,
        )
        if match is None or metric_rates[match] < target:
            return False
    return True


def trophy_score_loop(
    iteration: dict[str, Any], targets: dict[str, float]
) -> TrophyScore:
    """Score an optimization iteration row (``OptimizationState.iterations``)."""
    per_model_rates = iteration.get("per_model_pass_rates") or {}
    per_model_metrics = iteration.get("per_model_metrics") or {}
    model_ids = list(per_model_rates.keys())
    if not model_ids:
        # No per-model data: unrankable on ``at_target``, which is the primary
        # key. The aggregate still populates the tie-breakers so the row is
        # ordered rather than dropped, but it cannot claim to have met targets
        # it was never measured against.
        agg = float(iteration.get("pass_rate") or 0.0)
        return TrophyScore(at_target=0, best_rate=agg, avg_rate=agg)
    at_target = 0
    best_rate = 0.0
    total_rate = 0.0
    for mid in model_ids:
        rate = float(per_model_rates.get(mid) or 0.0)
        if rate > best_rate:
            best_rate = rate
        total_rate += rate
        metric_rates = per_model_metrics.get(mid) or {}
        if metric_rates and model_meets_targets(metric_rates, targets):
            at_target += 1
    avg_rate = total_rate / len(model_ids) if model_ids else 0.0
    return TrophyScore(at_target=at_target, best_rate=best_rate, avg_rate=avg_rate)


def trophy_score_run(
    run_summary: dict[str, Any], targets: dict[str, float]
) -> TrophyScore:
    """Score a run summary (``EvalRunner.list_runs`` shape)."""
    model_runs = run_summary.get("model_runs") or []
    if not model_runs:
        return TrophyScore(at_target=0, best_rate=0.0, avg_rate=0.0)
    at_target = 0
    best_rate = 0.0
    total_rate = 0.0
    for mr in model_runs:
        total = mr.get("total") or 0
        passed = mr.get("passed") or 0
        rate = passed / total if total else 0.0
        if rate > best_rate:
            best_rate = rate
        total_rate += rate
        per_metric = mr.get("per_metric") or {}
        # No per-metric data means the per-target check cannot run, so this
        # model does not count toward ``at_target``. Substituting the loosest
        # target let it claim compliance it was never measured for.
        if per_metric and model_meets_targets(per_metric, targets):
            at_target += 1
    avg_rate = total_rate / len(model_runs)
    return TrophyScore(at_target=at_target, best_rate=best_rate, avg_rate=avg_rate)


def trophy_sort_key(score: TrophyScore) -> tuple[int, float, float]:
    """Lexicographic ordering tuple; callers break ties later-wins."""
    return (score["at_target"], score["best_rate"], score["avg_rate"])
