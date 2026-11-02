# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judge-unreliable detector inside the optimization executor.

Covers the second tier of run-quality guards: the >=80% dominance case is
already handled by ``detect_metric_error_dominance``; this detector flips
``judge_unreliable`` when any model run reports ``data_quality.reliability
== "degraded"`` (>5% null after retry, set by the runner template).
"""

from __future__ import annotations

from app.services.jobs.optimization_executor import _detect_judge_unreliable
from app.services.resource_error import Codes


def _model_run(
    reliability: str | None,
    null_count: int = 0,
    total: int = 0,
    *,
    dropped_count: int = 0,
    unrecovered_count: int = 0,
    total_cases: int = 0,
) -> dict:
    """Build a minimal model-run payload — only the data_quality block matters."""
    run: dict = {"model_id": "m1", "results": []}
    if reliability is not None:
        run["data_quality"] = {
            "null_count": null_count,
            "total": total,
            "null_ratio": (null_count / total) if total else 0.0,
            "reliability": reliability,
            "dropped_count": dropped_count,
            "unrecovered_count": unrecovered_count,
            "total_cases": total_cases,
        }
    return run


class TestDetectJudgeUnreliable:
    def test_returns_none_for_empty_result(self):
        assert _detect_judge_unreliable([]) is None

    def test_returns_none_when_no_data_quality_field(self):
        # Older runs / non-template projects don't stamp data_quality.
        # Absence means we have nothing to flag — fall through to normal
        # iteration recording.
        assert _detect_judge_unreliable([{"model_id": "m1", "results": []}]) is None

    def test_returns_none_when_reliability_ok(self):
        runs = [_model_run("ok", null_count=2, total=200)]
        assert _detect_judge_unreliable(runs) is None

    def test_flags_single_degraded_model(self):
        # 30/200 = 15% null > 5% threshold → degraded by runner; detector
        # surfaces it.
        runs = [_model_run("degraded", null_count=30, total=200)]
        err = _detect_judge_unreliable(runs)
        assert err is not None
        assert err.code == Codes.EVAL_JUDGE_UNRELIABLE
        # Subset assertion — the detector also carries reconciliation
        # counters (``dropped_count``, ``unrecovered_count``,
        # ``total_cases``) that default to 0 here. Assert the nulls
        # we care about are surfaced.
        details = err.details["degraded_models"][0]
        assert details["model_id"] == "m1"
        assert details["null_count"] == 30
        assert details["total"] == 200
        assert details["null_ratio"] == 0.15
        assert "30/200" in err.message
        assert "throttled" in (err.hint or "").lower()

    def test_flags_when_any_model_degraded(self):
        # Mixed bag: one ok, one degraded. Detector returns the degraded one.
        runs = [
            _model_run("ok", null_count=1, total=200),
            {**_model_run("degraded", null_count=20, total=200), "model_id": "m2"},
        ]
        err = _detect_judge_unreliable(runs)
        assert err is not None
        assert len(err.details["degraded_models"]) == 1
        assert err.details["degraded_models"][0]["model_id"] == "m2"

    def test_unrecovered_count_takes_priority_in_message(self):
        """When unrecovered_count > 0 the banner leads with the silent-drop
        message, not the null-ratio message — drops are scarier than nulls."""
        runs = [_model_run(
            "degraded",
            null_count=0,  # nulls below threshold
            total=200,
            dropped_count=1,
            unrecovered_count=1,
            total_cases=210,
        )]
        err = _detect_judge_unreliable(runs)
        assert err is not None
        assert "silently dropped" in err.message
        assert "1/210" in err.message

    def test_unrecovered_alone_is_enough_to_halt(self):
        """Even when null_ratio is 0%, any unrecovered case must halt."""
        runs = [_model_run(
            "degraded",  # runner flipped this for us
            null_count=0,
            total=1000,
            dropped_count=2,
            unrecovered_count=2,
            total_cases=210,
        )]
        err = _detect_judge_unreliable(runs)
        assert err is not None
        assert err.details["degraded_models"][0]["unrecovered_count"] == 2
        assert err.details["degraded_models"][0]["total_cases"] == 210
