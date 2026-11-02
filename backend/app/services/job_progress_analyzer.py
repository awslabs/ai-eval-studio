# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fold the durable event log into per-model live-progress counters.

The coordinator's in-memory ``job.progress`` is frozen at the queued seed for
a DETACHED worker (the worker mutates its own tracker, not the coordinator's),
so ``/activity`` carries no live per-case counts. The ONLY durable source of
live per-case progress is the event log (``events_since``), which the progress
republisher already reads to fan out over ``/ws``.

This module provides a PURE fold of that event log into per-model counters,
mirroring the frontend ``handleEnvelope`` switch on ``data["event"]`` in
``useEvalJobStream.ts``. The API layer exposes the fold additively on
``/activity`` so a mid-scoring remount re-seeds ``Generate X/N`` from the
durable log instead of rebuilding only from the (already-passed) live stream.

Role justification (naming-conventions.md): ``_analyzer`` — derives structured
findings from data with no mutations.
"""
from __future__ import annotations

from typing import Any


def fold_event_log_progress(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold a job's durable event log into per-model progress counters.

    Mirrors the frontend ``handleEnvelope`` fold: each event is the envelope
    ``{stream, seq, kind, data}`` where ``data`` is the harness event. Branches
    on ``data["event"]``:

    * ``generating`` — tracks a per-model set of ``current`` case indices;
      ``generated`` = set size; ``phase="generating"``; ``cases_total = total``.
    * ``scoring_start`` — ``phase="scoring"``, ``generated = cases_total``,
      ``scored=False``, ``scored_count=0``.
    * ``scoring_progress`` — ``phase="scoring"``, ``scored_count = current``,
      ``cases_total = total or prior``.
    * ``scoring_complete`` — ``phase="scoring"``, ``scored=True``,
      ``scored_count = cases_total``, ``passed = passed``.
    * ``complete`` — ``phase="complete"``, ``scored=True``,
      ``passed = passed or prior``, ``generated = cases_total``.

    ``generated`` is emitted as a plain scalar (the set size at fold time). The
    frontend seed overwrites it and never backfills the live Set — safe because
    a mid-scoring remount is already past the generate phase.

    Args:
        events: The list returned by ``store.events_since(job_id, -1)`` (may be
            empty or contain malformed items — both handled defensively).

    Returns:
        A list of per-model dicts with snake_case wire fields:
        ``{model, phase, generated, cases_total, scored, scored_count, passed}``.
        Empty when ``events`` is empty/absent.
    """
    # Preserve first-seen model order for a stable wire shape.
    order: list[str] = []
    state: dict[str, dict[str, Any]] = {}
    # Per-model set of generated case indices (deduped, out-of-order tolerant).
    generated_sets: dict[str, set[int]] = {}

    def _entry(model: str) -> dict[str, Any]:
        if model not in state:
            order.append(model)
            state[model] = {
                "model": model,
                "phase": "generating",
                "generated": 0,
                "cases_total": 0,
                "scored": False,
                "scored_count": 0,
                "passed": 0,
            }
            generated_sets[model] = set()
        return state[model]

    for event in events or []:
        if not isinstance(event, dict):
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        event_type = data.get("event")
        model = data.get("model")
        if not model:
            # Whole-eval events (e.g. terminal `results`) carry no model —
            # they don't participate in the per-model fold.
            continue

        entry = _entry(model)

        if event_type == "generating":
            current = data.get("current")
            if current is not None:
                generated_sets[model].add(current)
            entry["phase"] = "generating"
            entry["generated"] = len(generated_sets[model])
            total = data.get("total")
            if total is not None:
                entry["cases_total"] = total
        elif event_type == "scoring_start":
            entry["phase"] = "scoring"
            entry["generated"] = entry["cases_total"]
            entry["scored"] = False
            entry["scored_count"] = 0
        elif event_type == "scoring_progress":
            entry["phase"] = "scoring"
            current = data.get("current")
            if current is not None:
                entry["scored_count"] = current
            total = data.get("total")
            if total:
                entry["cases_total"] = total
        elif event_type == "scoring_complete":
            entry["phase"] = "scoring"
            entry["scored"] = True
            entry["scored_count"] = entry["cases_total"]
            passed = data.get("passed")
            if passed is not None:
                entry["passed"] = passed
        elif event_type == "complete":
            entry["phase"] = "complete"
            entry["scored"] = True
            passed = data.get("passed")
            if passed is not None:
                entry["passed"] = passed
            entry["generated"] = entry["cases_total"]

    return [state[m] for m in order]


__all__ = ["fold_event_log_progress"]
