# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fold_event_log_progress — the durable event-log progress aggregator.

The aggregator folds a job's durable event log into per-model counters,
mirroring the frontend handleEnvelope fold. It is source-agnostic: the same
harness event shapes drive both a MANUAL eval and an OPTIMIZATION-child eval,
so both must fold identically (proven explicitly). generate uses a set→size
(dedup + out-of-order tolerant). Empty log → [].
"""
from __future__ import annotations

from app.services.job_progress_analyzer import fold_event_log_progress


def _env(seq: int, data: dict) -> dict:
    """Build a durable envelope {stream, seq, kind, data}."""
    return {"stream": "job:x", "seq": seq, "kind": "progress", "data": data}


def _manual_shaped_log(model: str, n: int) -> list[dict]:
    """A manual-eval-shaped log: generating→scoring_start→progress→complete."""
    events = []
    seq = 1
    for i in range(n):
        events.append(_env(seq, {"event": "generating", "model": model, "current": i, "total": n}))
        seq += 1
    events.append(_env(seq, {"event": "scoring_start", "model": model, "total": n}))
    seq += 1
    events.append(_env(seq, {"event": "scoring_progress", "model": model, "current": n // 2, "total": n}))
    seq += 1
    events.append(_env(seq, {"event": "scoring_complete", "model": model, "passed": n - 1}))
    return events


def test_empty_log_returns_empty_list():
    assert fold_event_log_progress([]) == []
    assert fold_event_log_progress(None) == []  # type: ignore[arg-type]


def test_folds_manual_shaped_log_per_model():
    """A manual eval's log folds into correct per-model counters."""
    log = _manual_shaped_log("m1", 10)
    result = fold_event_log_progress(log)

    assert len(result) == 1
    m = result[0]
    assert m["model"] == "m1"
    assert m["phase"] == "scoring"
    assert m["cases_total"] == 10
    assert m["generated"] == 10  # scoring_start set generated = cases_total
    assert m["scored"] is True
    assert m["scored_count"] == 10  # scoring_complete sets scored_count = cases_total
    assert m["passed"] == 9


def test_folds_optimization_child_shaped_log_per_model():
    """An optimization-child eval's log (same event shape) folds identically.

    The aggregator is source-agnostic — proving both explicitly matters
    because the bug hits both the optimization and manual progress views.
    """
    # Two models, mid-scoring (no scoring_complete yet).
    log = []
    seq = 1
    for model in ("opt-m1", "opt-m2"):
        for i in range(8):
            log.append(_env(seq, {"event": "generating", "model": model, "current": i, "total": 8}))
            seq += 1
        log.append(_env(seq, {"event": "scoring_start", "model": model, "total": 8}))
        seq += 1
        log.append(_env(seq, {"event": "scoring_progress", "model": model, "current": 3, "total": 8}))
        seq += 1

    result = fold_event_log_progress(log)
    assert len(result) == 2
    by_model = {m["model"]: m for m in result}
    for name in ("opt-m1", "opt-m2"):
        m = by_model[name]
        assert m["phase"] == "scoring"
        assert m["cases_total"] == 8
        assert m["generated"] == 8  # scoring_start
        assert m["scored"] is False
        assert m["scored_count"] == 3


def test_generate_counter_is_set_size_not_event_count():
    """Duplicate/out-of-order `current` values → generated = distinct set size."""
    log = [
        _env(1, {"event": "generating", "model": "m1", "current": 0, "total": 5}),
        _env(2, {"event": "generating", "model": "m1", "current": 2, "total": 5}),
        _env(3, {"event": "generating", "model": "m1", "current": 0, "total": 5}),  # dup
        _env(4, {"event": "generating", "model": "m1", "current": 1, "total": 5}),
        _env(5, {"event": "generating", "model": "m1", "current": 2, "total": 5}),  # dup
    ]
    result = fold_event_log_progress(log)
    assert len(result) == 1
    m = result[0]
    # 3 distinct indices {0, 1, 2}, not 5 events.
    assert m["generated"] == 3
    assert m["phase"] == "generating"
    assert m["cases_total"] == 5


def test_complete_event_sets_complete_phase():
    """A `complete` event finalizes the model: phase=complete, generated=cases_total."""
    log = [
        _env(1, {"event": "generating", "model": "m1", "current": 0, "total": 3}),
        _env(2, {"event": "scoring_start", "model": "m1", "total": 3}),
        _env(3, {"event": "scoring_complete", "model": "m1", "passed": 2}),
        _env(4, {"event": "complete", "model": "m1", "passed": 2}),
    ]
    result = fold_event_log_progress(log)
    m = result[0]
    assert m["phase"] == "complete"
    assert m["scored"] is True
    assert m["passed"] == 2
    assert m["generated"] == 3


def test_malformed_events_ignored_defensively():
    """Non-dict events, missing data, and model-less events are skipped."""
    log = [
        "not a dict",  # type: ignore[list-item]
        {"seq": 1},  # no data
        _env(2, {"event": "results", "file": "eval_results_v1.json"}),  # whole-eval, no model
        _env(3, {"event": "generating", "model": "m1", "current": 0, "total": 2}),
    ]
    result = fold_event_log_progress(log)
    assert len(result) == 1
    assert result[0]["model"] == "m1"
    assert result[0]["generated"] == 1
