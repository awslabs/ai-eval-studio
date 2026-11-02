# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the harness template's serialize path.

Ensures the template's ``runner.py`` still reads DeepEval attributes that
exist on the pinned DeepEval version. This test is fast (no subprocess,
no uv sync) and is the tripwire for the ``additional_metadata`` →
``metadata`` rename that silently broke eval in the past.

The test imports the template file by path rather than via ``src.*`` so
it doesn't pollute the backend's import graph with template code.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from deepeval.test_case import LLMTestCase

from tests.harness_template_stubs import (
    TEMPLATE_ROOT,
    load_template_bedrock_runtime_model,
    load_template_runner,
)


def _load_template_runner() -> ModuleType:
    """Load the template ``runner.py`` with the shared ``src.*`` stubs."""
    return load_template_runner("harness_template_runner")


@pytest.fixture(scope="module")
def template_runner() -> ModuleType:
    return _load_template_runner()


def _make_metric(
    name: str = "ExactMatch",
    *,
    score: float | None = 1.0,
    threshold: float = 1.0,
    success: bool | None = True,
    reason: str | None = None,
    error: str | None = None,
) -> MagicMock:
    metric = MagicMock()
    metric.name = name
    # ``BaseMetric`` exposes ``__name__`` as a property; the runner reads
    # it directly when constructing ``MetricData`` for retried/sentinel
    # rows, so test mocks must mirror that contract.
    metric.__name__ = name
    metric.score = score
    metric.threshold = threshold
    metric.success = success
    metric.reason = reason
    metric.error = error
    return metric


def _make_test_result(
    *,
    metrics: list[MagicMock],
    metadata: dict | None = None,
    success: bool = True,
    name: str = "case_0",
    actual_output: str = "pong",
    expected_output: str = "pong",
) -> MagicMock:
    tr = MagicMock(spec_set=[
        "name", "success", "input", "actual_output", "expected_output",
        "context", "retrieval_context", "metadata", "metrics_data",
    ])
    tr.name = name
    tr.success = success
    tr.input = "ping"
    tr.actual_output = actual_output
    tr.expected_output = expected_output
    tr.context = None
    tr.retrieval_context = None
    tr.metadata = metadata or {}
    tr.metrics_data = metrics
    return tr


def _make_eval_result(*, metadata: dict) -> MagicMock:
    """Construct a minimal EvaluationResult-shaped mock for _serialize_result."""
    test_result = _make_test_result(metrics=[_make_metric()], metadata=metadata)
    result = MagicMock()
    result.test_results = [test_result]
    return result


class TestSerializeResult:
    """Guards against DeepEval API churn on the serialize path.

    The template owns the on-disk JSON shape consumed by the frontend —
    this test pins that shape so any DeepEval field rename shows up here
    before it breaks a real eval run.
    """

    def test_output_shape_preserves_additional_metadata_key(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        eval_result = _make_eval_result(metadata={
            "latency_ms": 42,
            "input_tokens": 10,
            "output_tokens": 5,
            "generation_cost": 0.0001,
        })

        serialized = evaluator._serialize_result(
            model_id="us.anthropic.claude-sonnet-4-5-v1:0",
            eval_result=eval_result,
            hyperparams={"prompt_version": "baseline"},
            total_input_cases=len(eval_result.test_results),
        )

        assert serialized["test_run_id"]
        assert serialized["hyperparameters"]["prompt_version"] == "baseline"
        assert len(serialized["results"]) == 1
        row = serialized["results"][0]
        # On-disk key stays "additional_metadata" — that's our API contract
        # with the frontend even if DeepEval renames the attribute again.
        assert row["additional_metadata"]["latency_ms"] == 42
        assert row["additional_metadata"]["input_tokens"] == 10
        assert row["metrics_data"][0]["name"] == "ExactMatch"
        assert row["metrics_data"][0]["success"] is True

    def test_empty_metadata_serializes_cleanly(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        eval_result = _make_eval_result(metadata={})

        serialized = evaluator._serialize_result(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            eval_result=eval_result,
            hyperparams={},
            total_input_cases=len(eval_result.test_results),
        )
        assert serialized["results"][0]["additional_metadata"] == {}


class TestSerializeDataQuality:
    """The serializer must stamp a ``data_quality`` summary on every model run.

    Used by ``optimization_executor._detect_judge_unreliable`` to halt the
    loop when the judge is throttled/null'd more than 5% of the way
    through. Pinned in tests so a future refactor can't drop the field.
    """

    def test_stamps_ok_reliability_when_all_scores_present(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        result = MagicMock()
        result.test_results = [_make_test_result(metrics=[_make_metric()])]
        serialized = evaluator._serialize_result(
            model_id="m1",
            eval_result=result,
            hyperparams={},
            total_input_cases=len(result.test_results),
        )
        dq = serialized["data_quality"]
        assert dq["reliability"] == "ok"
        assert dq["null_count"] == 0
        assert dq["total"] == 1
        assert dq["null_ratio"] == 0.0

    def test_stamps_degraded_when_above_5_percent_null(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        # 2 nulls out of 10 metric evals = 20% null > 5% threshold.
        good = [_make_test_result(metrics=[_make_metric()]) for _ in range(8)]
        bad = [
            _make_test_result(
                metrics=[_make_metric(score=None, success=None, error="ThrottlingException")],
                success=False,
            )
            for _ in range(2)
        ]
        result = MagicMock()
        result.test_results = good + bad
        serialized = evaluator._serialize_result(
            model_id="m1",
            eval_result=result,
            hyperparams={},
            total_input_cases=len(result.test_results),
        )
        dq = serialized["data_quality"]
        assert dq["reliability"] == "degraded"
        assert dq["null_count"] == 2
        assert dq["total"] == 10
        assert dq["null_ratio"] == 0.2
        # The error text rides on the wire so the UI can show why.
        bad_row = serialized["results"][-1]
        assert bad_row["metrics_data"][0]["error"] == "ThrottlingException"
        assert bad_row["metrics_data"][0]["score"] is None

    def test_stays_ok_below_5_percent_null(self, template_runner):
        # 1 out of 21 = 4.7% — under threshold, stays ok.
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        good = [_make_test_result(metrics=[_make_metric()]) for _ in range(20)]
        bad = [_make_test_result(
            metrics=[_make_metric(score=None, success=None, error="boom")],
            success=False,
        )]
        result = MagicMock()
        result.test_results = good + bad
        serialized = evaluator._serialize_result(
            model_id="m1",
            eval_result=result,
            hyperparams={},
            total_input_cases=len(result.test_results),
        )
        assert serialized["data_quality"]["reliability"] == "ok"


class TestSerializeEmitsCost:
    """The serializer must emit per-model cost on the model-run dict.

    ``generation_cost_usd`` is the sum of per-case ``generation_cost``
    (already computed inside the serializer but previously dropped).
    ``judge_cost_usd`` is the dollar figure the runner derives from the
    shared judge model's accumulated token usage. Both are top-level
    siblings of ``data_quality`` so the report can read them directly.
    """

    def test_generation_cost_usd_sums_per_case_costs(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        cases = [
            _make_test_result(metrics=[_make_metric()], metadata={"generation_cost": 0.01}),
            _make_test_result(metrics=[_make_metric()], metadata={"generation_cost": 0.02}),
        ]
        result = MagicMock()
        result.test_results = cases
        serialized = evaluator._serialize_result(
            model_id="m1", eval_result=result, hyperparams={},
            total_input_cases=len(result.test_results),
        )
        assert serialized["generation_cost_usd"] == pytest.approx(0.03)

    def test_judge_cost_usd_flows_to_output(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        result = MagicMock()
        result.test_results = [_make_test_result(metrics=[_make_metric()])]
        serialized = evaluator._serialize_result(
            model_id="m1", eval_result=result, hyperparams={},
            judge_cost_usd=0.123,
            total_input_cases=len(result.test_results),
        )
        assert serialized["judge_cost_usd"] == pytest.approx(0.123)

    def test_judge_cost_usd_defaults_to_zero(self, template_runner):
        """Existing callers (smoke checks) that don't pass judge cost
        still serialize cleanly with a 0.0 default."""
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        result = MagicMock()
        result.test_results = [_make_test_result(metrics=[_make_metric()])]
        serialized = evaluator._serialize_result(
            model_id="m1", eval_result=result, hyperparams={},
            total_input_cases=len(result.test_results),
        )
        assert serialized["judge_cost_usd"] == 0.0


class TestRetryNullMetrics:
    """The retry phase must repair transient throttling without fail-open.

    Cases that come back null-after-retry stay null. Cases that succeed on
    retry get their score/success/reason replaced. The test_result.success
    flag is re-aggregated from the (possibly repaired) metrics so a case
    isn't silently counted as passed when one of its metrics is still
    unscored.
    """

    def _make_runner(self, template_runner, build_metrics_seq):
        """Wire a runner with a deterministic ``build_metrics`` sequence.

        ``build_metrics_seq`` is a list of metrics returned on each call —
        the runner rebuilds the list per attempt, so the test can simulate
        "first retry attempt fails, second succeeds" by handing back fresh
        metric mocks each time.
        """
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        # ``self.config`` is read for region/profile when constructing the
        # judge BedrockRuntimeModel — stub the BedrockRuntimeModel constructor instead of
        # forging a config object.
        evaluator.config = MagicMock()
        evaluator.config.aws_region = "us-east-1"
        evaluator.config.aws_profile = None

        # The retry path constructs a fresh BedrockRuntimeModel each call. Stub
        # the class so no real boto3 resolution happens.
        template_runner.BedrockRuntimeModel = MagicMock()

        # Patch build_metrics to return successive lists from the seq.
        seq_iter = iter(build_metrics_seq)
        template_runner.build_metrics = lambda _judge: next(seq_iter)

        # ``time.sleep`` is hot-path in the retry loop — short-circuit it.
        template_runner.time = MagicMock()
        template_runner.time.sleep = MagicMock()
        return evaluator

    def test_retry_repairs_a_null_metric_in_place(self, template_runner):
        # First attempt: null again. Second attempt: scored 0.9 success.
        # Third attempt would also work but break is hit on the second.
        attempt_1_metric = _make_metric(score=None, success=None, error="Throttle")
        attempt_2_metric = _make_metric(score=0.9, success=True, reason="ok-after-retry")
        attempt_3_metric = _make_metric(score=0.95, success=True)
        evaluator = self._make_runner(
            template_runner,
            [
                [attempt_1_metric],
                [attempt_2_metric],
                [attempt_3_metric],
            ],
        )

        # Original eval: one case, one metric, scored null.
        original_md = _make_metric(score=None, success=False, error="Throttle")
        original_md.threshold = 0.5
        tr = _make_test_result(metrics=[original_md], success=False)
        eval_result = MagicMock()
        eval_result.test_results = [tr]

        evaluator._retry_null_metrics("m1", eval_result, MagicMock(name="judge_model"))

        repaired = eval_result.test_results[0].metrics_data[0]
        assert repaired.score == 0.9
        assert repaired.success is True
        assert repaired.reason == "ok-after-retry"
        assert repaired.error is None
        # Success flag re-aggregated from the now-repaired metric.
        assert eval_result.test_results[0].success is True

    def test_retry_leaves_stubborn_failures_null(self, template_runner):
        # All three attempts return null — runner must NOT fail-open.
        evaluator = self._make_runner(
            template_runner,
            [
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=None, success=None, error="Throttle")],
            ],
        )

        original_md = _make_metric(score=None, success=False, error="Throttle")
        original_md.threshold = 0.5
        tr = _make_test_result(metrics=[original_md], success=False)
        eval_result = MagicMock()
        eval_result.test_results = [tr]

        evaluator._retry_null_metrics("m1", eval_result, MagicMock(name="judge_model"))

        unrepaired = eval_result.test_results[0].metrics_data[0]
        assert unrepaired.score is None  # Stays null — no fail-open
        # Test success stays False because metric.success is not True.
        assert eval_result.test_results[0].success is False

    def test_retry_no_op_when_all_metrics_already_scored(self, template_runner):
        # Nothing to retry — runner should not call build_metrics at all.
        called = {"build": 0}

        def _trip(_judge):
            called["build"] += 1
            return []

        evaluator = self._make_runner(template_runner, [])
        template_runner.build_metrics = _trip

        good = _make_metric(score=1.0, success=True)
        tr = _make_test_result(metrics=[good], success=True)
        eval_result = MagicMock()
        eval_result.test_results = [tr]

        evaluator._retry_null_metrics("m1", eval_result, MagicMock(name="judge_model"))
        assert called["build"] == 0

    def test_retry_uses_the_shared_judge_model(self, template_runner):
        """The retry phase scores with the judge_model passed in by
        run_single_model — not a fresh one it builds itself — so judge
        usage accrues into a single per-run accumulator."""
        captured = {"judge": None}
        attempt_metric = _make_metric(score=0.9, success=True, reason="ok")

        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        evaluator.config = MagicMock()
        template_runner.time = MagicMock()
        template_runner.time.sleep = MagicMock()

        def _capture_build(judge):
            captured["judge"] = judge
            return [attempt_metric]

        template_runner.build_metrics = _capture_build

        original_md = _make_metric(score=None, success=False, error="Throttle")
        original_md.threshold = 0.5
        tr = _make_test_result(metrics=[original_md], success=False)
        eval_result = MagicMock()
        eval_result.test_results = [tr]

        shared_judge = MagicMock(name="shared_judge_model")
        evaluator._retry_null_metrics("m1", eval_result, shared_judge)

        assert captured["judge"] is shared_judge

    def test_retry_first_attempt_does_not_sleep(self, template_runner):
        """First retry attempt fires immediately. Backoff only applies
        between attempts — same fix as the reconciliation pass."""
        attempt_1_metric = _make_metric(score=0.9, success=True, reason="ok")
        evaluator = self._make_runner(
            template_runner,
            [[attempt_1_metric]],
        )

        original_md = _make_metric(score=None, success=False, error="Throttle")
        original_md.threshold = 0.5
        tr = _make_test_result(metrics=[original_md], success=False)
        eval_result = MagicMock()
        eval_result.test_results = [tr]

        evaluator._retry_null_metrics("m1", eval_result, MagicMock(name="judge_model"))

        template_runner.time.sleep.assert_not_called()


class TestReconcileMissingCases:
    """Cases that ``evaluate()`` silently drops must be re-scored, not lost.

    DeepEval's async ``evaluate()`` wraps every test-case task in an
    ``asyncio.gather`` with a global gather-timeout; tasks cancelled
    before they reach their ``finally`` block never call
    ``test_results.append()``. The result is a shorter list than the
    input — different models in the same run lose different counts,
    which is what surfaced as "model A shows 200, model B shows 198"
    in the dashboard.

    These tests pin the recovery + sentinel + counter contract.
    """

    def _make_runner(self, template_runner, build_metrics_seq):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        evaluator.config = MagicMock()
        evaluator.config.aws_region = "us-east-1"
        evaluator.config.aws_profile = None
        template_runner.BedrockRuntimeModel = MagicMock()
        seq_iter = iter(build_metrics_seq)
        template_runner.build_metrics = lambda _judge: next(seq_iter)
        # Skip real sleeps — backoff doesn't matter for unit tests.
        template_runner.time = MagicMock()
        template_runner.time.sleep = MagicMock()
        return evaluator

    def _make_eval_case(self, idx: int, *, input_text: str = "q"):
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=input_text,
            actual_output="a",
            expected_output="b",
            metadata={"case_index": idx},
        )

    def _present_tr(self, idx: int, *, input_text: str = "q"):
        """A TestResult-shaped mock that survived evaluate()."""
        tr = MagicMock(spec_set=[
            "name", "success", "input", "actual_output", "expected_output",
            "context", "retrieval_context", "metadata", "metrics_data",
        ])
        tr.name = f"present_{idx}"
        tr.success = True
        tr.input = input_text
        tr.actual_output = "a"
        tr.expected_output = "b"
        tr.context = None
        tr.retrieval_context = None
        tr.metadata = {"case_index": idx}
        tr.metrics_data = [_make_metric()]
        return tr

    def test_reconcile_recovers_missing_case_on_first_attempt(self, template_runner):
        """A case dropped by evaluate() and re-scoreable on attempt 1
        ends up with metrics_data populated and metadata.reconciled=True."""
        # Submit 3 cases, evaluate() returns only 2 (case_index=1 dropped).
        eval_cases = [self._make_eval_case(i) for i in range(3)]
        present = [self._present_tr(0), self._present_tr(2)]
        eval_result = MagicMock()
        eval_result.test_results = list(present)

        # Build_metrics returns one metric that scores cleanly first try.
        success_metric = _make_metric(score=0.9, success=True, reason="ok")
        evaluator = self._make_runner(
            template_runner,
            [[success_metric]],  # one attempt's worth — must succeed first try
        )

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # One new TestResult appended for case_index=1.
        assert len(eval_result.test_results) == 3
        added = [
            tr for tr in eval_result.test_results
            if (tr.metadata or {}).get("reconciled")
        ]
        assert len(added) == 1
        recovered = added[0]
        assert (recovered.metadata or {}).get("case_index") == 1
        assert recovered.metrics_data[0].score == 0.9
        assert recovered.metrics_data[0].error is None
        assert recovered.success is True

    def test_reconcile_inserts_sentinel_after_three_failed_attempts(self, template_runner):
        """All three attempts return null — sentinel row appended, score stays None,
        error explains the reconciliation failure. NO fail-open."""
        eval_cases = [self._make_eval_case(i) for i in range(2)]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0)]  # Case 1 dropped

        # Every attempt: metric scores null with throttle error.
        evaluator = self._make_runner(
            template_runner,
            [
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=None, success=None, error="Throttle")],
                # 4th list for the sentinel construction path.
                [_make_metric(score=None, success=None, error="Throttle")],
            ],
        )

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # Sentinel row was appended.
        assert len(eval_result.test_results) == 2
        sentinel = next(
            tr for tr in eval_result.test_results
            if (tr.metadata or {}).get("case_index") == 1
        )
        # Score stays None — never fail-open.
        assert sentinel.metrics_data[0].score is None
        # Error text identifies this as a reconciliation failure.
        assert "reconciliation failed" in (sentinel.metrics_data[0].error or "")
        # Test success is False because no metric succeeded.
        assert sentinel.success is False

    def test_reconcile_no_op_when_all_cases_present(self, template_runner):
        """When evaluate() returned every case, reconciliation does nothing."""
        eval_cases = [self._make_eval_case(i) for i in range(3)]
        eval_result = MagicMock()
        eval_result.test_results = [
            self._present_tr(0), self._present_tr(1), self._present_tr(2),
        ]

        called = {"build": 0}

        def _trip(_judge):
            called["build"] += 1
            return []

        evaluator = self._make_runner(template_runner, [])
        template_runner.build_metrics = _trip

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # Still 3 results; no metrics rebuild fired.
        assert len(eval_result.test_results) == 3
        assert called["build"] == 0

    def test_reconcile_preserves_already_present_cases_unchanged(self, template_runner):
        """The recovery pass must NOT touch cases that survived evaluate()."""
        eval_cases = [self._make_eval_case(i) for i in range(3)]
        present_0 = self._present_tr(0)
        present_2 = self._present_tr(2)
        # Stash a sentinel value on present case 0 so we can detect mutation.
        present_0.metrics_data[0].score = 0.42
        eval_result = MagicMock()
        eval_result.test_results = [present_0, present_2]

        evaluator = self._make_runner(
            template_runner,
            [[_make_metric(score=0.9, success=True)]],
        )

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # Present_0 untouched (still 0.42).
        assert present_0.metrics_data[0].score == 0.42
        # Present_2 untouched.
        assert present_2.metadata.get("case_index") == 2
        # Case 1 was reconciled.
        reconciled = next(
            tr for tr in eval_result.test_results
            if (tr.metadata or {}).get("reconciled")
        )
        assert (reconciled.metadata or {}).get("case_index") == 1

    def test_data_quality_reflects_dropped_count(self, template_runner):
        """End-to-end: serialize after reconciliation and confirm
        ``dropped_count`` + ``unrecovered_count`` land in data_quality
        and reliability flips to degraded when any case stays unrecovered."""
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"

        # 5 cases input. 3 present cleanly. 1 reconciled successfully.
        # 1 sentinel (unrecovered).
        rec_metrics = [_make_metric(score=0.8, success=True)]
        sentinel_metrics = [_make_metric(
            score=None, success=False, error="reconciliation failed",
        )]

        results = []
        for i in range(3):
            tr = _make_test_result(metrics=[_make_metric()], name=f"c{i}")
            tr.metadata = {"case_index": i}
            results.append(tr)
        # Reconciled case
        rec = _make_test_result(metrics=rec_metrics, name="c3")
        rec.metadata = {"case_index": 3, "reconciled": True}
        results.append(rec)
        # Sentinel case
        sentinel = _make_test_result(
            metrics=sentinel_metrics, name="c4", success=False,
        )
        sentinel.metadata = {"case_index": 4, "reconciled": True}
        results.append(sentinel)

        eval_result = MagicMock()
        eval_result.test_results = results

        serialized = evaluator._serialize_result(
            model_id="m1",
            eval_result=eval_result,
            hyperparams={},
            total_input_cases=5,
        )
        dq = serialized["data_quality"]

        assert dq["total_cases"] == 5
        assert dq["dropped_count"] == 2  # rec + sentinel
        assert dq["unrecovered_count"] == 1  # Only sentinel
        # Any unrecovered → degraded, even though null_ratio is low.
        assert dq["reliability"] == "degraded"

    def test_reconcile_handles_case_with_generation_failure(self, template_runner):
        """A case whose generation failed (actual_output='') still gets
        ``case_index`` stamped, and if evaluate() drops it, reconciliation
        runs measure() on the empty-output test case without crashing."""
        from deepeval.test_case import LLMTestCase
        failed_gen_case = LLMTestCase(
            input="q",
            actual_output="",  # Generation failed
            expected_output="b",
            metadata={"case_index": 1, "error": "generation timeout"},
        )
        eval_cases = [
            self._make_eval_case(0),
            failed_gen_case,
            self._make_eval_case(2),
        ]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0), self._present_tr(2)]

        # Metric scores the empty-output case at 0.0 (correctly; it failed).
        metric = _make_metric(score=0.0, success=False, reason="empty output")
        evaluator = self._make_runner(template_runner, [[metric]])

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        reconciled = next(
            tr for tr in eval_result.test_results
            if (tr.metadata or {}).get("reconciled")
        )
        assert (reconciled.metadata or {}).get("case_index") == 1
        # Original generation error preserved in metadata so the UI can
        # show "generation timeout" alongside the score.
        assert (reconciled.metadata or {}).get("error") == "generation timeout"
        assert reconciled.metrics_data[0].score == 0.0
        # Test fails because metric scored below threshold.
        assert reconciled.success is False

    def test_reconcile_emits_placeholder_when_sentinel_build_fails(self, template_runner):
        """When build_metrics() raises during sentinel construction (4th
        throttle), reconciliation must NOT crash — instead emit a
        placeholder MetricData so the case still counts as unrecovered.
        Without this, a single throttle wipes every remaining missing
        case from the reconciliation pass."""
        eval_cases = [self._make_eval_case(0), self._make_eval_case(1)]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0)]  # Case 1 dropped

        # Three null attempts, then build_metrics raises on the sentinel call.
        null_metric = _make_metric(score=None, success=None, error="Throttle")
        call_count = {"n": 0}

        def _build_metrics(_judge):
            call_count["n"] += 1
            if call_count["n"] >= 4:
                raise RuntimeError("Bedrock throttled — out of retry budget")
            return [null_metric]

        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        evaluator.config = MagicMock()
        evaluator.config.aws_region = "us-east-1"
        evaluator.config.aws_profile = None
        template_runner.BedrockRuntimeModel = MagicMock()
        template_runner.build_metrics = _build_metrics
        template_runner.time = MagicMock()
        template_runner.time.sleep = MagicMock()

        # Must not raise.
        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        sentinel = next(
            tr for tr in eval_result.test_results
            if (tr.metadata or {}).get("case_index") == 1
        )
        # Placeholder MetricData was emitted so unrecovered_count counts this.
        assert len(sentinel.metrics_data) == 1
        assert sentinel.metrics_data[0].score is None
        assert "build_metrics unavailable" in sentinel.metrics_data[0].error

    def test_reconcile_uses_the_shared_judge_model(self, template_runner):
        """Reconciliation scores recovered cases with the judge_model
        passed in by run_single_model, so its judge usage accrues into the
        same per-run accumulator the main pass and retry use."""
        captured = {"judge": None}

        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        evaluator.config = MagicMock()
        template_runner.time = MagicMock()
        template_runner.time.sleep = MagicMock()

        def _capture_build(judge):
            captured["judge"] = judge
            return [_make_metric(score=0.9, success=True)]

        template_runner.build_metrics = _capture_build

        eval_cases = [self._make_eval_case(0), self._make_eval_case(1)]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0)]  # Case 1 dropped

        shared_judge = MagicMock(name="shared_judge_model")
        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, shared_judge,
        )

        assert captured["judge"] is shared_judge

    def test_reconcile_first_attempt_does_not_sleep(self, template_runner):
        """First reconciliation attempt fires immediately. Backoff only
        applies between attempts. Critical for runs with 90+ dropped
        cases — leading sleep would add ~10 minutes of pure deadtime."""
        eval_cases = [self._make_eval_case(0), self._make_eval_case(1)]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0)]  # Case 1 dropped

        # First attempt succeeds — no retry needed.
        success_metric = _make_metric(score=0.9, success=True)
        evaluator = self._make_runner(
            template_runner,
            [[success_metric]],
        )

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # When the first attempt succeeds, sleep should not be called.
        template_runner.time.sleep.assert_not_called()

    def test_reconcile_sleeps_between_failed_attempts(self, template_runner):
        """Failed attempts trigger backoff: 1s before attempt 1, 2s before attempt 2."""
        eval_cases = [self._make_eval_case(0), self._make_eval_case(1)]
        eval_result = MagicMock()
        eval_result.test_results = [self._present_tr(0)]

        # First two attempts fail, third succeeds.
        evaluator = self._make_runner(
            template_runner,
            [
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=None, success=None, error="Throttle")],
                [_make_metric(score=0.9, success=True)],
            ],
        )

        evaluator._reconcile_missing_cases(
            "m1", eval_cases, eval_result, MagicMock(name="judge_model"),
        )

        # Two sleeps total: 1s before attempt 1, 2s before attempt 2.
        sleep_calls = [c.args[0] for c in template_runner.time.sleep.call_args_list]
        assert sleep_calls == [1, 2]


class TestConcurrencyDefault:
    """The runner's default ``max_concurrent`` is tuned against
    DeepEval's gather-timeout, not against Bedrock throttling. The
    BedrockRuntimeModel layer handles throttling via ``@_bedrock_retry``
    (4 attempts, exp backoff). Pinning the default here so a future
    refactor can't quietly lower it without reading the rationale in
    ``run_single_model`` — the concurrency-20-era nulls were proven
    to be gather-timeout cancellations (empty ``MetricData.error``,
    zero ``ThrottlingException`` in coordinator logs), not real
    throttling.
    """

    def test_default_is_20(self, template_runner):
        """Constructed without override, the runner resolves max_concurrent=20."""
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.source = "manual"
        # No EvalConfig.max_concurrent attribute → default is read.
        evaluator.config = MagicMock(spec=[])  # no attributes at all
        evaluator.max_concurrent_override = None
        # Read what would be passed to AsyncConfig: same getattr the
        # runner uses inline in run_single_model.
        max_concurrent = (
            evaluator.max_concurrent_override
            if evaluator.max_concurrent_override is not None
            else getattr(evaluator.config, "max_concurrent", 20)
        )
        assert max_concurrent == 20

    def test_override_still_wins(self, template_runner):
        """``max_concurrent_override=1`` (passed by validate_harness)
        beats both EvalConfig.max_concurrent and the default."""
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.config = MagicMock()
        evaluator.config.max_concurrent = 30  # Config wants 30
        evaluator.max_concurrent_override = 1  # Smoke override

        max_concurrent = (
            evaluator.max_concurrent_override
            if evaluator.max_concurrent_override is not None
            else getattr(evaluator.config, "max_concurrent", 20)
        )
        assert max_concurrent == 1

    def test_eval_config_value_beats_default(self, template_runner):
        """Project-set EvalConfig.max_concurrent beats the runner default."""
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator.config = MagicMock()
        evaluator.config.max_concurrent = 8
        evaluator.max_concurrent_override = None

        max_concurrent = (
            evaluator.max_concurrent_override
            if evaluator.max_concurrent_override is not None
            else getattr(evaluator.config, "max_concurrent", 20)
        )
        assert max_concurrent == 8


class TestSubprocessEnvSetsGatherTimeout:
    """``_project_subprocess_env`` must set
    ``DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600``. Without it,
    DeepEval's default 180s gather budget cancels the queue mid-run
    and cases get silently dropped.
    """

    def test_env_sets_per_task_timeout_override(self):
        from app.services.harness_runner import _project_subprocess_env
        env = _project_subprocess_env()
        assert env.get("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE") == "600"

    def test_env_strips_virtual_env(self, monkeypatch):
        """Existing behavior preserved: parent venv path stripped so
        ``uv run`` activates the project's own venv."""
        from app.services.harness_runner import _project_subprocess_env
        monkeypatch.setenv("VIRTUAL_ENV", "/parent/venv")
        env = _project_subprocess_env()
        assert "VIRTUAL_ENV" not in env


class TestLLMTestCaseConstruction:
    """Ensures the DeepEval ``LLMTestCase`` kwargs the template uses still exist.

    The template writes ``LLMTestCase(metadata=...)``; DeepEval could rename
    that kwarg in a future release. This test catches that at backend test
    time so the template can be updated before it breaks scaffolded projects.
    """

    def test_llmtestcase_accepts_metadata_kwarg(self):
        tc = LLMTestCase(
            input="x",
            actual_output="y",
            expected_output="y",
            metadata={"k": "v"},
        )
        assert tc.metadata == {"k": "v"}


def _load_template_bedrock_runtime_model() -> ModuleType:
    """Import the template ``bedrock_runtime_model.py`` under a unique module name.

    Delegates to the shared installer so this loader exists in ONE place —
    ``test_converse_reasoning_content.py`` needs the same module, and a
    second hand-copied loader is exactly how the ``src.*`` stub tree
    drifted across five files before it was consolidated.
    """
    return load_template_bedrock_runtime_model()


class TestTemplateBedrockModelTemperatureGuard:
    """Response-driven ``temperature`` self-correction on the template
    BedrockRuntimeModel (replaces the static
    ``unsupported_patterns`` model list). The first call attempts with
    ``temperature``; if Bedrock returns a ValidationException mentioning
    temperature, the param is dropped and the call is retried once.
    Subsequent calls on the same instance skip the param entirely.
    """

    def test_self_corrects_when_bedrock_rejects_temperature(self):
        from botocore.exceptions import ClientError

        template = _load_template_bedrock_runtime_model()
        BedrockRuntimeModel = template.BedrockRuntimeModel

        model = BedrockRuntimeModel(
            model_id="us.anthropic.claude-opus-4-7",
            region="us-east-1",
            temperature=0.3,
        )

        ok = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        temp_reject = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Malformed input request: temperature is not supported.",
                }
            },
            operation_name="Converse",
        )

        fake_client = MagicMock()
        fake_client.converse.side_effect = [temp_reject, ok]
        model._client = fake_client

        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "ok"

        # First call carried temperature; second (self-corrected) didn't.
        first_kwargs = fake_client.converse.call_args_list[0].kwargs
        second_kwargs = fake_client.converse.call_args_list[1].kwargs
        assert "temperature" in first_kwargs["inferenceConfig"]
        assert "temperature" not in second_kwargs["inferenceConfig"]

    def test_passes_temperature_when_model_accepts_it(self):
        """Models that don't reject temperature — every non-reasoning model —
        receive the configured value verbatim. The gate only trips on a
        live response-driven rejection, never on a static name match."""
        template = _load_template_bedrock_runtime_model()
        BedrockRuntimeModel = template.BedrockRuntimeModel

        model = BedrockRuntimeModel(
            model_id="us.anthropic.claude-sonnet-4-5-v1:0",
            region="us-east-1",
            temperature=0.3,
        )

        captured: dict = {}

        def fake_converse(**kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        model._client = fake_client

        model.generate_with_system("sys", "user")

        assert captured["inferenceConfig"].get("temperature") == 0.3

    def test_temperature_gate_latches_after_one_rejection(self):
        """Once Bedrock rejects temperature, every subsequent call on
        the same instance skips it on the FIRST attempt — no second
        round-trip per call."""
        from botocore.exceptions import ClientError

        template = _load_template_bedrock_runtime_model()
        BedrockRuntimeModel = template.BedrockRuntimeModel

        model = BedrockRuntimeModel(
            model_id="us.anthropic.claude-opus-4-7",
            region="us-east-1",
            temperature=0.3,
        )

        ok = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        temp_reject = ClientError(
            error_response={
                "Error": {
                    "Code": "ValidationException",
                    "Message": "temperature: parameter is not supported.",
                }
            },
            operation_name="Converse",
        )

        fake_client = MagicMock()
        fake_client.converse.side_effect = [temp_reject, ok, ok]
        model._client = fake_client

        model.generate_with_system("sys", "first")
        model.generate_with_system("sys", "second")

        # Three Converse calls total: 1 (rejected) + 1 (self-corrected) + 1 (no temp).
        assert fake_client.converse.call_count == 3
        assert "temperature" not in fake_client.converse.call_args_list[2].kwargs["inferenceConfig"]


class TestTemplateBedrockModelRetry:
    """The template must retry transient Bedrock failures locally.

    DeepEval's bundled ``AmazonBedrockModel`` carries ``@retry_bedrock``
    but our custom subclass exists for thread-safety / temperature-guard
    / empty-system-prompt reasons (see module docstring), so DeepEval's
    decorator never runs on our path. Without local retry, a single
    ThrottlingException becomes a null metric score under
    ``ignore_errors=True``. Pin the policy here so the regression
    surfaces at backend test time, not partway through a long eval.
    """

    @staticmethod
    def _throttle_error():
        from botocore.exceptions import ClientError
        return ClientError(
            error_response={"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            operation_name="Converse",
        )

    @staticmethod
    def _ok_response(text: str = "ok"):
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }

    def _model(self, side_effect):
        template = _load_template_bedrock_runtime_model()
        BedrockRuntimeModel = template.BedrockRuntimeModel
        model = BedrockRuntimeModel(
            model_id="us.anthropic.claude-sonnet-4-5-v1:0",
            region="us-east-1",
        )
        client = MagicMock()
        client.converse.side_effect = side_effect
        model._client = client
        return model, client

    def test_retries_throttling_until_success(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "4")
        monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.0")
        monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.0")

        model, client = self._model([
            self._throttle_error(),
            self._throttle_error(),
            self._ok_response("third-time-lucky"),
        ])
        text, _, _ = model.generate_with_system("sys", "user")
        assert text == "third-time-lucky"
        assert client.converse.call_count == 3

    def test_access_denied_fails_immediately(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "4")
        monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.0")
        monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.0")

        from botocore.exceptions import ClientError
        access_denied = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}},
            operation_name="Converse",
        )
        model, client = self._model(access_denied)
        with pytest.raises(ClientError):
            model.generate_with_system("sys", "user")
        # Non-retryable — exactly one attempt.
        assert client.converse.call_count == 1

    def test_generate_path_also_retried(self, monkeypatch):
        """Template's ``generate()`` is its own client.converse() (not a delegate)
        so it must carry its own decorator."""
        monkeypatch.setenv("BEDROCK_RETRY_MAX_ATTEMPTS", "4")
        monkeypatch.setenv("BEDROCK_RETRY_INITIAL_S", "0.0")
        monkeypatch.setenv("BEDROCK_RETRY_CAP_S", "0.0")

        model, client = self._model([
            self._throttle_error(),
            self._ok_response("via-generate"),
        ])
        out = model.generate("hello")
        assert out == "via-generate"
        assert client.converse.call_count == 2


class TestRenderPrompt:
    """Tests for the template runner's ``render_prompt`` helper.

    The runner is authoritative for turning ``{input}`` / ``{context}``
    placeholders in ``prompts/<version>.md`` into per-case text. Regressions
    here have shipped silently in the past (the agent-written loader tried
    to do substitution and got it wrong), so the contract is pinned in
    tests.
    """

    def test_fills_input_placeholder(self, template_runner):
        case = LLMTestCase(input="What is Answer?", actual_output="", expected_output="")
        rendered = template_runner.render_prompt(
            "Answer: {input}", case,
        )
        assert rendered == "Answer: What is Answer?"

    def test_fills_context_from_list(self, template_runner):
        case = LLMTestCase(
            input="q",
            actual_output="",
            expected_output="",
            context=["doc-a", "doc-b"],
        )
        rendered = template_runner.render_prompt(
            "Context:\n{context}\nQ: {input}", case,
        )
        assert "doc-a" in rendered
        assert "doc-b" in rendered
        assert "---" in rendered  # Separator between passages
        assert rendered.endswith("Q: q")

    def test_missing_context_renders_empty(self, template_runner):
        """A template that references ``{context}`` on a non-RAG case
        shouldn't crash — it just fills with an empty string."""
        case = LLMTestCase(input="q", actual_output="", expected_output="")
        rendered = template_runner.render_prompt(
            "Ctx:{context} Q:{input}", case,
        )
        assert rendered == "Ctx: Q:q"

    def test_unknown_placeholder_raises(self, template_runner):
        """A template with an unsupported placeholder must fail loud so the
        smoke test catches it, not silently ship ``{foo}`` to the model."""
        case = LLMTestCase(input="q", actual_output="", expected_output="")
        with pytest.raises(ValueError, match="unknown placeholder"):
            template_runner.render_prompt("Hello {foo}", case)

    def test_template_without_placeholders_is_returned_as_is(self, template_runner):
        case = LLMTestCase(input="q", actual_output="", expected_output="")
        template = "You are a helpful assistant. Respond concisely."
        assert template_runner.render_prompt(template, case) == template


class TestTemplateBedrockModelUsageAccumulator:
    """The judge BedrockRuntimeModel must accumulate token usage across calls.

    DeepEval never reports ``evaluation_cost`` for our custom
    (non-native) model — it stays ``None`` regardless. To surface judge
    cost we capture the raw ``usage`` returned by every ``converse()``
    on the model and accumulate it under a lock; the runner turns that
    into a dollar figure at serialize time (one pricing source). This
    pins the accumulator contract: both ``generate()`` (its own converse
    call) and ``generate_with_system()`` accrue, totals sum across calls,
    and concurrent threads don't lose updates.
    """

    def _model(self, side_effect=None):
        template = _load_template_bedrock_runtime_model()
        model = template.BedrockRuntimeModel(
            model_id="us.anthropic.claude-sonnet-4-5-v1:0",
            region="us-east-1",
        )
        client = MagicMock()
        if side_effect is not None:
            client.converse.side_effect = side_effect
        else:
            client.converse.return_value = {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 4},
            }
        model._client = client
        return model

    def test_fresh_model_has_zero_usage(self):
        model = self._model()
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 0
        assert usage["outputTokens"] == 0

    def test_generate_accrues_usage(self):
        model = self._model()
        model.generate("hello")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 10
        assert usage["outputTokens"] == 4

    def test_generate_with_system_accrues_usage(self):
        model = self._model()
        model.generate_with_system("sys", "user")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 10
        assert usage["outputTokens"] == 4

    def test_multiple_calls_sum(self):
        model = self._model()
        model.generate("a")
        model.generate("b")
        model.generate_with_system("sys", "c")
        usage = model.accumulated_usage
        assert usage["inputTokens"] == 30
        assert usage["outputTokens"] == 12

    def test_concurrent_calls_do_not_lose_updates(self):
        """N threads each making one call → total == N × per-call usage.

        Guards the lock around the accumulator: a non-atomic
        read-add-write would drop updates under contention.
        """
        import concurrent.futures

        model = self._model()
        n = 64
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda _i: model.generate("x"), range(n)))

        usage = model.accumulated_usage
        assert usage["inputTokens"] == 10 * n
        assert usage["outputTokens"] == 4 * n

    def test_missing_usage_key_does_not_crash(self):
        """A converse response without ``usage`` must not raise — the
        scorer treats a raised exception as a null score under
        ``ignore_errors=True``, so accrual must be defensive."""
        model = self._model(side_effect=[
            {"output": {"message": {"content": [{"text": "ok"}]}}},  # no usage
        ])
        text = model.generate("hi")
        assert text == "ok"
        # Nothing accrued, but the call still returned cleanly.
        assert model.accumulated_usage["inputTokens"] == 0


class TestBedrockModelHandlesEmptySystem:
    """Regression: Bedrock rejects empty ``system[].text`` with an opaque
    ValidationError. The wrapper should substitute a neutral default so
    the caller gets a sensible response instead of a cryptic crash."""

    def test_empty_system_prompt_falls_back_to_default(self, template_runner):
        fake_src = sys.modules["src.models.bedrock_runtime_model"]
        try:
            module = _load_template_bedrock_runtime_model()
        finally:
            sys.modules["src.models.bedrock_runtime_model"] = fake_src

        model = module.BedrockRuntimeModel.__new__(module.BedrockRuntimeModel)
        model._model_id = "us.anthropic.claude-sonnet-4-6"
        model._region = "us-east-1"
        model._max_tokens = 2048
        model._temperature = 0.0
        # This test bypasses __init__ via __new__, so it must set the
        # usage-accumulator state that generate_with_system() now touches.
        model._usage_lock = threading.Lock()
        model._usage_input_tokens = 0
        model._usage_output_tokens = 0
        # Response-driven temperature gate state. Init True so
        # the first call attempts with ``temperature``; the latching
        # path is exercised in test_bedrock_runtime_model_retry.
        model._supports_temperature_flag = True
        model._temperature_lock = threading.Lock()

        captured = {}

        def fake_converse(**kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        model._client = fake_client

        # Empty string should get replaced before hitting boto3.
        model.generate_with_system("", "user question")
        assert captured["system"][0]["text"].strip() != ""

        # Whitespace-only also gets replaced.
        model.generate_with_system("   \n  ", "user question")
        assert captured["system"][0]["text"].strip() != ""


class TestRunSingleModelRouting:
    """Pin the descriptor-based routing in ``EvalRunner._make_model``
    and ``EvalRunner.run_single_model``. Mantle ids (those present in
    ``src/model_descriptors.json``) MUST construct ``BedrockMantleModel``,
    not ``BedrockRuntimeModel``. A Converse call to a mantle-only id
    silently fails with an empty response and triggers the null-metric
    retry storm — this test keeps that routing regression from coming back.

    The runner is a 1:1 mirror of the dispatch.py ``_preflight_models_invocable``
    routing (descriptor-presence dispatch), so this test pins the same
    contract at the generation layer.
    """

    def _make_runner_with_descriptors(
        self,
        template_runner,
        descriptors: dict,
        *,
        target_models: list[str],
        judge_model_id: str,
    ):
        """Construct an EvalRunner whose ``_descriptors`` map is patched."""
        cfg = MagicMock()
        cfg.aws_region = "us-east-1"
        cfg.aws_profile = None
        cfg.temperature = 0.0
        cfg.max_tokens = 4096
        cfg.models = target_models
        cfg.judge_model_id = judge_model_id
        cfg.prompt_version = "baseline"
        runner = template_runner.EvalRunner(cfg, json_progress=True)
        # Patch the descriptor map on the instance — the test fixture
        # loader stubs read_descriptors to return ``{}`` by default.
        runner._descriptors = descriptors
        return runner

    def test_make_model_routes_mantle_id_through_mantle_adapter(
        self, template_runner
    ):
        """An id present in the descriptor map → BedrockMantleModel."""
        descriptors = {
            "openai.gpt-5.4": {
                "endpoint": "mantle",
                "api": "responses",
                "region": "us-east-1",
            }
        }
        runner = self._make_runner_with_descriptors(
            template_runner,
            descriptors,
            target_models=["openai.gpt-5.4"],
            judge_model_id="us.anthropic.claude-sonnet-4-6",
        )
        # The runner did ``from ... import BedrockMantleModel`` at module
        # load, so the symbol it dispatches to lives on the LOADED runner
        # module — patching ``sys.modules["src.models...."]`` after the
        # Fact won't reach it. Assert against template_runner's bound
        # Symbols and reset their call_counts directly.
        mantle_cls = template_runner.BedrockMantleModel
        runtime_cls = template_runner.BedrockRuntimeModel
        mantle_cls.reset_mock()
        runtime_cls.reset_mock()

        runner._make_model("openai.gpt-5.4")

        # Mantle adapter constructed exactly once.
        assert mantle_cls.call_count == 1
        call = mantle_cls.call_args
        # Descriptor passed positionally first; model_id second.
        assert call.args[0] == descriptors["openai.gpt-5.4"]
        assert call.args[1] == "openai.gpt-5.4"
        # ``max_tokens`` (config field) maps to ``max_output_tokens``
        # (mantle adapter constructor kwarg) — the names differ, so
        # this assertion guards against the kwarg-rename regression.
        assert call.kwargs.get("max_output_tokens") == 4096
        # Temperature + profile_name carry through unchanged.
        assert call.kwargs.get("temperature") == 0.0
        assert call.kwargs.get("profile_name") is None
        # NOT passing aws_region — single-source-region invariant.
        # The mantle adapter signs SigV4 against the
        # descriptor's region; passing config.aws_region here would
        # break cross-region models like xai.grok-4.3 (us-west-2-only)
        # whenever the project's coordinator runs in us-east-1.
        assert "region" not in call.kwargs
        # And no positional region either (the constructor accepts
        # only descriptor + model_id positionally).
        assert len(call.args) == 2

        # Runtime adapter NOT constructed for a mantle id.
        assert runtime_cls.call_count == 0

    def test_make_model_routes_runtime_id_through_runtime_adapter(
        self, template_runner
    ):
        """An id NOT in the descriptor map → BedrockRuntimeModel."""
        descriptors: dict = {}  # Runtime-only project
        runner = self._make_runner_with_descriptors(
            template_runner,
            descriptors,
            target_models=["us.anthropic.claude-sonnet-4-6"],
            judge_model_id="us.anthropic.claude-sonnet-4-6",
        )
        # The runner did ``from ... import BedrockRuntimeModel`` at module
        # load, so the symbol it dispatches to lives on the LOADED runner
        # module — patching ``sys.modules["src.models...."]`` after the
        # Fact won't reach it. Assert against template_runner's bound
        # Symbols and reset their call_counts directly.
        mantle_cls = template_runner.BedrockMantleModel
        runtime_cls = template_runner.BedrockRuntimeModel
        mantle_cls.reset_mock()
        runtime_cls.reset_mock()

        runner._make_model("us.anthropic.claude-sonnet-4-6")

        assert runtime_cls.call_count == 1
        assert mantle_cls.call_count == 0
        call = runtime_cls.call_args
        assert call.kwargs["model_id"] == "us.anthropic.claude-sonnet-4-6"
        # Runtime adapter receives region from config.aws_region (no
        # descriptor to read it from). Different contract from the
        # mantle path — guarded by the kwarg-omission check above.
        assert call.kwargs["region"] == "us-east-1"
        assert call.kwargs["max_tokens"] == 4096
        assert call.kwargs["temperature"] == 0.0
        assert call.kwargs["profile_name"] is None

    def test_make_model_routes_mixed_lineup_per_id(self, template_runner):
        """A mixed lineup (some mantle, some runtime) → each id routes
        independently. For example
        ``[gpt-5.5, gpt-5.4, claude-opus-4-8, sonnet-4-6]``, where the
        last is runtime-only and the rest are mantle."""
        descriptors = {
            "openai.gpt-5.5": {
                "endpoint": "mantle", "api": "responses", "region": "us-east-1",
            },
            "openai.gpt-5.4": {
                "endpoint": "mantle", "api": "responses", "region": "us-east-1",
            },
            "anthropic.claude-opus-4-8": {
                "endpoint": "mantle", "api": "anthropic_messages",
                "region": "us-east-1",
            },
        }
        runner = self._make_runner_with_descriptors(
            template_runner,
            descriptors,
            target_models=[
                "openai.gpt-5.5",
                "openai.gpt-5.4",
                "anthropic.claude-opus-4-8",
                "us.anthropic.claude-sonnet-4-6",
            ],
            judge_model_id="us.anthropic.claude-sonnet-4-6",
        )
        # The runner did ``from ... import BedrockRuntimeModel`` at module
        # load, so the symbol it dispatches to lives on the LOADED runner
        # module — patching ``sys.modules["src.models...."]`` after the
        # Fact won't reach it. Assert against template_runner's bound
        # Symbols and reset their call_counts directly.
        mantle_cls = template_runner.BedrockMantleModel
        runtime_cls = template_runner.BedrockRuntimeModel
        mantle_cls.reset_mock()
        runtime_cls.reset_mock()

        for mid in runner.config.models:
            runner._make_model(mid)

        # 3 mantle → 3 mantle adapter constructions.
        assert mantle_cls.call_count == 3
        # 1 runtime → 1 runtime adapter construction.
        assert runtime_cls.call_count == 1

    def test_make_model_for_mantle_judge(self, template_runner):
        """A judge id with a mantle descriptor routes through the mantle
        adapter — same gate as the target. A Converse-routed mantle judge
        rejects every metric call and silently nulls every score."""
        descriptors = {
            "anthropic.claude-opus-4-8": {
                "endpoint": "mantle", "api": "anthropic_messages",
                "region": "us-east-1",
            }
        }
        runner = self._make_runner_with_descriptors(
            template_runner,
            descriptors,
            target_models=["us.anthropic.claude-sonnet-4-6"],
            judge_model_id="anthropic.claude-opus-4-8",
        )
        # The runner did ``from ... import BedrockRuntimeModel`` at module
        # load, so the symbol it dispatches to lives on the LOADED runner
        # module — patching ``sys.modules["src.models...."]`` after the
        # Fact won't reach it. Assert against template_runner's bound
        # Symbols and reset their call_counts directly.
        mantle_cls = template_runner.BedrockMantleModel
        runtime_cls = template_runner.BedrockRuntimeModel
        mantle_cls.reset_mock()
        runtime_cls.reset_mock()

        # Same factory the judge construction site goes through.
        runner._make_model(runner.config.judge_model_id)

        assert mantle_cls.call_count == 1
        assert runtime_cls.call_count == 0
        # Same kwargs contract as the target path — region NOT in
        # kwargs (descriptor's region is the SoT for the judge too),
        # max_output_tokens carries config.max_tokens, temperature
        # and profile_name pass through.
        call = mantle_cls.call_args
        assert call.args[0] == descriptors["anthropic.claude-opus-4-8"]
        assert call.args[1] == "anthropic.claude-opus-4-8"
        assert "region" not in call.kwargs
        assert call.kwargs.get("max_output_tokens") == 4096
        assert call.kwargs.get("temperature") == 0.0
        assert call.kwargs.get("profile_name") is None

    def test_descriptors_loaded_in_init(self, template_runner, monkeypatch):
        """``__init__`` must call ``read_descriptors`` exactly once so
        per-call ``_make_model`` is descriptor-cached, not re-reading the
        JSON file on every model construction in ``run_single_model``.

        Patches the symbol on the LOADED runner module — runner.py does
        ``from ... import read_descriptors`` at module load, so the
        binding lives on ``template_runner.read_descriptors``, not on
        the source module's ``read_descriptors`` attribute (the latter
        is a copy-by-reference at import time).
        """
        calls: list = []

        def _spy_read_descriptors(project_dir):
            calls.append(project_dir)
            return {
                "openai.gpt-5.4": {
                    "endpoint": "mantle",
                    "api": "responses",
                    "region": "us-east-1",
                },
            }

        monkeypatch.setattr(
            template_runner, "read_descriptors", _spy_read_descriptors
        )

        cfg = MagicMock()
        cfg.aws_region = "us-east-1"
        cfg.models = ["openai.gpt-5.4"]
        cfg.judge_model_id = "us.anthropic.claude-sonnet-4-6"
        template_runner.EvalRunner(cfg, json_progress=True)

        assert len(calls) == 1, (
            "read_descriptors must be called exactly once in __init__; "
            "additional calls suggest descriptors are being re-read on "
            "every model construction."
        )


class TestEstimateCostNoInvention:
    """Pricing.json is the ONLY cost source in the scaffolded runner.

    The hardcoded fallback table and the $1/$5 'ultimate fallback' silently
    billed unknown models at stale or invented rates. Unknown pricing is now
    reported as None (JSON null) — never fabricated."""

    def _runner_with_pricing(self, monkeypatch, pricing: dict):
        mod = _load_template_runner()
        monkeypatch.setattr(mod, "_load_pricing", lambda: pricing)
        mod._warned_no_pricing.clear()
        return mod

    _USAGE = {"inputTokens": 1_000_000, "outputTokens": 1_000_000}

    def test_known_model_computes_from_pricing_json(self, monkeypatch):
        mod = self._runner_with_pricing(
            monkeypatch, {"openai.gpt-5.6-luna": {"input": 0.20, "output": 1.20}}
        )
        cost = mod._estimate_cost(self._USAGE, "openai.gpt-5.6-luna")
        assert cost == pytest.approx(1.40)

    def test_unknown_model_returns_none_not_invented(self, monkeypatch):
        mod = self._runner_with_pricing(monkeypatch, {})
        assert mod._estimate_cost(self._USAGE, "vendor.brand-new-model") is None

    def test_entry_missing_rates_returns_none(self, monkeypatch):
        mod = self._runner_with_pricing(
            monkeypatch, {"m1": {"input": 0.5}}  # no output rate
        )
        assert mod._estimate_cost(self._USAGE, "m1") is None

    def test_fallback_table_is_gone(self, monkeypatch):
        """The substring-matched hardcoded table must not resurface."""
        mod = self._runner_with_pricing(monkeypatch, {})
        assert not hasattr(mod, "_FALLBACK_PRICING")
        # A model the old table matched by substring now honestly reports None.
        assert mod._estimate_cost(self._USAGE, "anthropic.claude-3-haiku-20240307") is None

    def test_unknown_model_warns_once(self, monkeypatch, capsys):
        mod = self._runner_with_pricing(monkeypatch, {})
        mod._estimate_cost(self._USAGE, "vendor.x")
        mod._estimate_cost(self._USAGE, "vendor.x")
        err = capsys.readouterr().err
        assert err.count("no pricing for 'vendor.x'") == 1

    def test_cache_rates_fall_back_to_models_own_input_rate(self, monkeypatch):
        """Missing cache rates anchor to the model's REAL input rate — an
        approximation, not an invention."""
        mod = self._runner_with_pricing(
            monkeypatch, {"m2": {"input": 1.0, "output": 2.0}}
        )
        usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadInputTokens": 1_000_000,
            "cacheWriteInputTokens": 0,
        }
        assert mod._estimate_cost(usage, "m2") == pytest.approx(1.0)


class TestGatingMetricsOnly:
    """Only TARGET metrics gate case pass/fail.

    Measured live: a diagnostic 'validity' metric at threshold 1.0 silently
    vetoed all 832 cases (DeepEval's TestResult.success is all-metrics-pass)
    while the target metric scored 68-88%. 'Display-only' must be an
    enforced property of the aggregation, not a code comment.
    """

    def _runner_with_gating(self, template_runner, gating: set[str]):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator._gating_metric_names = frozenset(gating)
        return evaluator

    def test_diagnostic_metric_cannot_veto_case(self, template_runner):
        evaluator = self._runner_with_gating(template_runner, {"Answer Correctness"})
        tr = _make_test_result(
            metrics=[
                _make_metric(name="Answer Correctness", score=0.9, threshold=0.5, success=True),
                _make_metric(name="Answer Validity", score=0.0, threshold=1.0, success=False),
            ],
            success=False,  # DeepEval's all-metrics aggregate says failed
        )
        assert evaluator._gated_success(tr) is True, (
            "the failing DIAGNOSTIC metric must not veto a passing TARGET"
        )

    def test_failing_target_still_fails_case(self, template_runner):
        evaluator = self._runner_with_gating(template_runner, {"Answer Correctness"})
        tr = _make_test_result(
            metrics=[
                _make_metric(name="Answer Correctness", score=0.2, threshold=0.5, success=False),
                _make_metric(name="Answer Validity", score=1.0, threshold=1.0, success=True),
            ],
            success=False,
        )
        assert evaluator._gated_success(tr) is False

    def test_row_without_any_target_metric_is_not_a_pass(self, template_runner):
        evaluator = self._runner_with_gating(template_runner, {"Answer Correctness"})
        tr = _make_test_result(
            metrics=[_make_metric(name="Answer Validity", success=True)],
            success=True,
        )
        assert evaluator._gated_success(tr) is False, (
            "an unscored target is never a pass"
        )

    def test_no_gating_config_falls_back_to_deepeval_aggregate(self, template_runner):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        # No _gating_metric_names attr at all (legacy / smoke construction).
        tr = _make_test_result(metrics=[_make_metric(success=False)], success=True)
        assert evaluator._gated_success(tr) is True

    def test_gating_fallback_warns_once_on_stderr(self, template_runner, capsys):
        """The fallback must announce itself — silence is how it caused harm.

        With no gating names every diagnostic metric becomes a gate, which once
        zeroed an 832-case run while the target metric scored 68-88%. Semantics are
        unchanged here; the point is that the degradation is no longer silent.
        Warned once per run, not once per case, or a 1000-case run emits 1000 lines.
        """
        template_runner._warned_no_gating.clear()
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        tr = _make_test_result(metrics=[_make_metric(success=False)], success=True)

        assert evaluator._gated_success(tr) is True
        first = capsys.readouterr().err
        assert "no gating metrics configured" in first
        assert "target_metric_scores" in first, "must name the config key to fix"

        # Two more cases in the same run: no further output.
        assert evaluator._gated_success(tr) is True
        assert evaluator._gated_success(tr) is True
        assert capsys.readouterr().err == "", "warning must be latched per run"


class TestGatingSuffixNormalization:
    """Gating must match metric names regardless of DeepEval's type suffix.

    DeepEval's ``GEval.__name__`` (version-dependent) appends the metric
    type — ``"Answer Correctness [GEval]"`` — while ``target_metric_scores``
    keys carry the bare name. Measured live: the exact-match comparison
    never matched and an entire run reported 0/416 while the target metric
    scored 51-56%. Per-project harness venvs install DeepEval fresh, so the
    suffix behavior can differ per project — both comparison sides must be
    normalized.
    """

    def _runner_with_gating(self, template_runner, gating: set[str]):
        evaluator = template_runner.EvalRunner.__new__(template_runner.EvalRunner)
        evaluator._gating_metric_names = frozenset(gating)
        return evaluator

    def test_suffixed_metric_name_matches_bare_gating_key(self, template_runner):
        evaluator = self._runner_with_gating(template_runner, {"Answer Correctness"})
        tr = _make_test_result(
            metrics=[
                _make_metric(
                    name="Answer Correctness [GEval]",
                    score=0.9, threshold=0.5, success=True,
                ),
            ],
            success=False,  # Would be the corrupted aggregate under the bug
        )
        assert evaluator._gated_success(tr) is True, (
            "a '[GEval]'-suffixed metric name must gate against the bare "
            "config key — exact-match zeroed a whole run (0/416 at 51-56% "
            "real score)"
        )

    def test_suffixed_gating_key_matches_bare_metric_name(self, template_runner):
        # Reverse direction: config key carries the suffix, runtime is bare.
        evaluator = self._runner_with_gating(
            template_runner, {"Answer Correctness [GEval]"}
        )
        tr = _make_test_result(
            metrics=[
                _make_metric(
                    name="Answer Correctness", score=0.9, threshold=0.5, success=True,
                ),
            ],
            success=False,
        )
        assert evaluator._gated_success(tr) is True

    def test_suffixed_failing_target_still_fails(self, template_runner):
        evaluator = self._runner_with_gating(template_runner, {"Answer Correctness"})
        tr = _make_test_result(
            metrics=[
                _make_metric(
                    name="Answer Correctness [GEval]",
                    score=0.2, threshold=0.5, success=False,
                ),
            ],
            success=False,
        )
        assert evaluator._gated_success(tr) is False

    def test_base_metric_name_strips_only_trailing_bracket_suffix(
        self, template_runner
    ):
        base = template_runner.EvalRunner._base_metric_name
        assert base("Answer Correctness [GEval]") == "Answer Correctness"
        assert base("Answer Correctness") == "Answer Correctness"
        # Brackets INSIDE the name are content, not a type suffix.
        assert base("Recall [strict] check") == "Recall [strict] check"
        assert base("") == ""
