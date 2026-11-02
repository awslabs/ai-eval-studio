# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metric-error dominance detector."""

from app.services.metric_error_analyzer import (
    detect_generation_failure,
    detect_infra_failure,
    detect_metric_error_dominance,
)
from app.services.resource_error import Codes


def _case(success: bool, error: str | None = None) -> dict:
    return {
        "name": "tc",
        "success": success,
        "actual_output": "generated output",  # Non-empty so generation check passes
        "metrics_data": [
            {"name": "Correctness", "score": 0.0, "success": success, "error": error},
        ],
    }


def _model_run(cases: list[dict]) -> dict:
    return {"test_run_id": "m", "results": cases}


def _gen_case(actual_output: str, error: str | None = None) -> dict:
    """Helper to build a test case for generation-failure testing.

    Mimics the shape of a real failing run where generation produced
    empty output but metrics still scored it (as a legitimate 0.0).
    """
    return {
        "name": "tc",
        "success": False,
        "actual_output": actual_output,
        "additional_metadata": {"error": error} if error else {},
        "metrics_data": [
            {"name": "Correctness", "score": 0.0, "success": False, "error": None},
        ],
    }


class TestDominanceDetection:
    def test_returns_none_for_empty_result(self):
        assert detect_metric_error_dominance([]) is None

    def test_returns_none_when_no_errors(self):
        result = [_model_run([_case(True), _case(False), _case(True)])]
        assert detect_metric_error_dominance(result) is None

    def test_returns_none_below_threshold(self):
        # 2 errored out of 10 = 20% — well below 80% threshold.
        cases = [_case(False, "boom") for _ in range(2)] + [_case(True) for _ in range(8)]
        assert detect_metric_error_dominance([_model_run(cases)]) is None

    def test_flags_100_percent_errored(self):
        cases = [_case(False, "ExpiredTokenException: session expired") for _ in range(20)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert err.code == Codes.EVAL_METRIC_ERRORS_DOMINANT
        assert err.details["errored_scores"] == 20
        assert err.details["total_scores"] == 20
        assert err.details["error_ratio"] == 1.0
        assert err.details["unique_errors"] == 1

    def test_flags_at_threshold(self):
        # 80% errored (exactly at threshold) should trip.
        cases = [_case(False, "AccessDeniedException") for _ in range(8)] + [_case(True) for _ in range(2)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert err.details["error_ratio"] == 0.8

    def test_hint_maps_aws_token_expiry(self):
        cases = [_case(False, "ExpiredTokenException: The security token included in the request is expired")
                 for _ in range(5)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert "aws sso login" in (err.hint or "").lower()

    def test_hint_maps_access_denied(self):
        cases = [_case(False, "AccessDeniedException: user is not authorized to perform bedrock:InvokeModel")
                 for _ in range(5)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert "bedrock" in (err.hint or "").lower()
        assert "access" in (err.hint or "").lower()

    def test_hint_maps_throttling(self):
        cases = [_case(False, "ThrottlingException: rate exceeded") for _ in range(5)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert "throttl" in (err.hint or "").lower()

    def test_hint_generic_fallback_for_unknown_error(self):
        cases = [_case(False, "RuntimeError: surprise") for _ in range(5)]
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert err.hint  # Generic fallback present
        assert "aws" not in (err.hint or "").lower()

    def test_counts_across_multiple_models(self):
        model_a = _model_run([_case(False, "boom") for _ in range(10)])
        model_b = _model_run([_case(False, "boom") for _ in range(10)])
        err = detect_metric_error_dominance([model_a, model_b])
        assert err is not None
        assert err.details["total_scores"] == 20

    def test_distinct_error_messages_count(self):
        cases = ([_case(False, "boom A") for _ in range(10)] +
                 [_case(False, "boom B") for _ in range(10)])
        err = detect_metric_error_dominance([_model_run(cases)])
        assert err is not None
        assert err.details["unique_errors"] == 2


class TestGenerationFailureDetection:
    def test_flags_generation_dominant_with_creds_error(self):
        """Generation-failed with session-expired error → EVAL_GENERATION_ERRORS_DOMINANT + creds hint."""
        cases = [
            _gen_case(
                "[]",
                "Your session has expired or credentials have changed. Please reauthenticate using 'aws login'.",
            )
            for _ in range(20)
        ]
        err = detect_generation_failure([_model_run(cases)])
        assert err is not None
        assert err.code == Codes.EVAL_GENERATION_ERRORS_DOMINANT
        assert err.details["generation_failed_cases"] == 20
        assert err.details["total_cases"] == 20
        assert err.details["error_ratio"] == 1.0
        assert err.hint is not None
        hint_lower = err.hint.lower()
        assert "aws" in hint_lower and "login" in hint_lower

    def test_returns_none_below_threshold(self):
        """2 generation-failed + 8 healthy → None (below 80% threshold)."""
        cases = [_gen_case("[]", "error A") for _ in range(2)] + [
            _gen_case("non-empty output") for _ in range(8)
        ]
        assert detect_generation_failure([_model_run(cases)]) is None

    def test_returns_none_for_genuine_low_score(self):
        """All non-empty output, no errors, metrics scored → None."""
        cases = [_gen_case("some output") for _ in range(10)]
        assert detect_generation_failure([_model_run(cases)]) is None

    def test_detects_empty_output_without_error_text(self):
        """All cases have empty output but no additional_metadata.error → still flagged."""
        cases = [_gen_case("[]") for _ in range(20)]
        err = detect_generation_failure([_model_run(cases)])
        assert err is not None
        assert err.code == Codes.EVAL_GENERATION_ERRORS_DOMINANT
        assert err.details["unique_errors"] == 0
        assert err.details["top_error_preview"] == ""


class TestInfraFailureDetection:
    def test_generation_dominant_returns_generation_error(self):
        """Generation failures checked first → returns EVAL_GENERATION_ERRORS_DOMINANT."""
        cases = [
            _gen_case(
                "[]",
                "Your session has expired or credentials have changed. Please reauthenticate using 'aws login'.",
            )
            for _ in range(20)
        ]
        err = detect_infra_failure([_model_run(cases)])
        assert err is not None
        assert err.code == Codes.EVAL_GENERATION_ERRORS_DOMINANT

    def test_metric_error_dominant_returns_metric_error(self):
        """Metric errors dominate → returns EVAL_METRIC_ERRORS_DOMINANT."""
        # Use _case (not _gen_case) to produce cases with non-empty actual_output
        # and metric errors — this ensures generation check passes through and
        # detect_metric_error_dominance triggers
        cases = [_case(False, "ThrottlingException") for _ in range(20)]
        err = detect_infra_failure([_model_run(cases)])
        assert err is not None
        assert err.code == Codes.EVAL_METRIC_ERRORS_DOMINANT

    def test_returns_none_for_healthy_run(self):
        """No generation or metric failures → None."""
        cases = [_gen_case("valid output") for _ in range(10)]
        assert detect_infra_failure([_model_run(cases)]) is None
