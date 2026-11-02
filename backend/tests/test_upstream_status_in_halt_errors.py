# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for upstream HTTP status extraction and classification in halt errors.

Behavioral coverage: status/endpoint/model named, number-not-prose classification,
conditional credentials clause, graceful no-status fallback, discipline guards.
"""

from app.services.metric_error_analyzer import (
    detect_generation_unreliable,
    _extract_dominant_upstream_status,
)
from app.services.jobs.optimization_executor import _detect_judge_unreliable
from app.services.resource_error import Codes


def _model_run_with_generation_failures(
    model_id: str,
    endpoint: str,
    region: str,
    failed_count: int,
    total_cases: int,
    status_code: int,
) -> dict:
    """Build a model run with generation failures carrying upstream status."""
    results = []
    # Build failed cases with status in error text
    for i in range(failed_count):
        results.append({
            "name": f"case_{i}",
            "success": False,
            "actual_output": "",
            "additional_metadata": {
                "error": f"Client error '{status_code} Service Unavailable' for url 'https://bedrock-mantle.{region}.api.aws/v1/...'",
            },
            "metrics_data": [],
        })
    # Build successful cases
    for i in range(total_cases - failed_count):
        results.append({
            "name": f"case_{failed_count + i}",
            "success": True,
            "actual_output": "output",
            "additional_metadata": {},
            "metrics_data": [{"name": "Correctness", "score": 1.0, "success": True}],
        })

    return {
        "model_id": model_id,
        "hyperparameters": {"endpoint": endpoint, "region": region},
        "results": results,
        "data_quality": {
            "generation_failed_count": failed_count,
            "total_cases": total_cases,
            "generation_failed_ratio": round(failed_count / total_cases, 4) if total_cases else 0.0,
            "null_count": 0,
            "total": total_cases * 2,  # metrics per case
            "reliability": "ok",
        },
    }


class TestDominantUpstreamStatusExtraction:
    """Test the _extract_dominant_upstream_status helper."""

    def test_extracts_503_from_mantle_error(self):
        """503 in Client error text classifies as 503 with repo label."""
        error_texts = [
            "Client error '503 Service Unavailable' for url 'https://bedrock-mantle.us-east-1.api.aws/v1/...'",
        ] * 10
        result = _extract_dominant_upstream_status(error_texts)
        assert result is not None
        assert result["status"] == 503
        assert result["label"] == "Service Unavailable"
        assert result["count"] == 10

    def test_extracts_500_from_runtime_error(self):
        """500 in Server error text classifies as 500."""
        error_texts = [
            "Server error '500 Internal Server Error' for url 'https://bedrock-runtime.us-east-1.amazonaws.com/...'",
        ] * 5
        result = _extract_dominant_upstream_status(error_texts)
        assert result is not None
        assert result["status"] == 500
        assert result["count"] == 5

    def test_classifies_by_number_not_prose(self):
        """503 with altered reason phrase still classifies as 503 with our label."""
        error_texts = [
            "Client error '503 Potato' for url 'https://bedrock-mantle.us-east-1.api.aws/v1/...'",
        ] * 10
        result = _extract_dominant_upstream_status(error_texts)
        assert result is not None
        assert result["status"] == 503
        assert result["label"] == "Service Unavailable"  # Our label, not "Potato"

    def test_phrase_without_number_returns_none(self):
        """Reason phrase without numeric status does not classify."""
        error_texts = [
            "Service Unavailable - something went wrong",
        ] * 10
        result = _extract_dominant_upstream_status(error_texts)
        # Should return None since no parseable status
        assert result is None

    def test_incidental_digits_do_not_match(self):
        """Bare 3-digit number in non-status context does not classify."""
        error_texts = [
            "token count 503 exceeded maximum",
            "case index 401 failed validation",
        ]
        result = _extract_dominant_upstream_status(error_texts)
        assert result is None

    def test_dominant_status_wins(self):
        """Most common status wins when multiple present."""
        error_texts = (
            ["Client error '503 Service Unavailable' ..."] * 96 +
            ["Client error '500 Internal Server Error' ..."] * 2
        )
        result = _extract_dominant_upstream_status(error_texts)
        assert result is not None
        assert result["status"] == 503
        assert result["count"] == 96
        assert result["observed"] == {503: 96, 500: 2}

    def test_returns_none_for_empty_list(self):
        """Empty error list returns None."""
        assert _extract_dominant_upstream_status([]) is None

    def test_returns_none_for_non_http_exceptions(self):
        """Non-HTTP exception text returns None."""
        error_texts = ["ValueError: invalid data", "TypeError: wrong type"]
        result = _extract_dominant_upstream_status(error_texts)
        assert result is None


class TestGenerationUnreliableWithStatus:
    """Test detect_generation_unreliable with upstream status extraction."""

    def test_names_status_endpoint_model_on_mantle_503(self):
        """Above-tolerance mantle 503 names status + label + endpoint + model."""
        results = [
            _model_run_with_generation_failures(
                model_id="openai.gpt-4o",
                endpoint="mantle",
                region="us-east-1",
                failed_count=40,
                total_cases=208,
                status_code=503,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        assert error.code == Codes.EVAL_GENERATION_UNRELIABLE

        # Message must name model, count/total, status+label, endpoint
        msg = error.message
        assert "openai.gpt-4o" in msg
        assert "40" in msg and "208" in msg
        assert "503" in msg
        assert "Service Unavailable" in msg or "service unavailable" in msg.lower()
        assert "mantle" in msg.lower()

        # Details must have structured fields
        assert error.details["models"][0]["status"] == 503
        assert error.details["models"][0]["label"] == "Service Unavailable"
        assert error.details["models"][0]["endpoint"] == "mantle"

    def test_credentials_clause_absent_when_401_observed(self):
        """401 in observed statuses prevents 'not credentials' claim."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="mantle",
                region="us-east-1",
                failed_count=10,
                total_cases=50,
                status_code=401,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        msg_lower = error.message.lower()
        assert "not credentials" not in msg_lower
        assert "401" in error.message

    def test_credentials_clause_present_when_no_auth_status(self):
        """503 with no 401/403 allows 'capacity not credentials' claim."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="runtime",
                region="us-west-2",
                failed_count=10,
                total_cases=50,
                status_code=503,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        msg_lower = error.message.lower()
        # Should claim it's capacity, not credentials
        assert "capacity" in msg_lower or "throttl" in msg_lower
        # Exact phrasing may vary but should not be auth-focused
        assert "401" not in error.message and "403" not in error.message

    def test_no_status_fallback_graceful(self):
        """No parseable status still halts but with generic message."""
        # Build payload with non-HTTP exceptions
        results = [{
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 10,
                "total_cases": 50,
                "generation_failed_ratio": 0.2,
            },
        }]
        error = detect_generation_unreliable(results)
        assert error is not None
        # Should not assert a status
        assert "status" not in error.details.get("models", [{}])[0]
        # Hint should be generic
        assert error.hint is not None

    def test_verdict_unchanged_for_report_fold(self):
        """Sub-tolerance generation failure returns None, which the run-integrity
        fold in report.py (``_build_run_integrity``) relies on."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="runtime",
                region="us-east-1",
                failed_count=1,
                total_cases=208,
                status_code=503,
            )
        ]
        # 1/208 = 0.48% < 2% tolerance
        error = detect_generation_unreliable(results)
        assert error is None  # Exactly None for report.py fold

    def test_message_discipline_no_judge(self):
        """Generation halt message contains no 'judge' reference."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="mantle",
                region="us-east-1",
                failed_count=10,
                total_cases=50,
                status_code=503,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        assert "judge" not in error.message.lower()

    def test_message_discipline_no_null_fragment(self):
        """Generation halt message contains no null-count fragment."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="runtime",
                region="us-east-1",
                failed_count=40,
                total_cases=208,
                status_code=503,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        assert "null" not in error.message.lower()
        # No 0/N pattern
        import re
        assert not re.search(r"(?:^|[^\d])0\s*/\s*\d+", error.message)


class TestJudgeUnreliableWithStatus:
    """Test _detect_judge_unreliable with upstream status extraction."""

    def test_names_dominant_status_503(self):
        """Judge degraded with 503 errors names the status."""
        results = [{
            "model_id": "test-model",
            "data_quality": {"reliability": "degraded", "null_count": 30, "total": 200},
            "results": [
                {
                    "metrics_data": [
                        {"name": "Correctness", "score": None, "error": "Client error '503 Service Unavailable' ..."}
                    ]
                } for _ in range(30)
            ] + [
                {
                    "metrics_data": [
                        {"name": "Correctness", "score": 0.8, "error": None}
                    ]
                } for _ in range(170)
            ],
        }]
        error = _detect_judge_unreliable(results)
        assert error is not None
        assert error.code == Codes.EVAL_JUDGE_UNRELIABLE

        # Message must name status
        msg = error.message
        assert "503" in msg
        assert "Service Unavailable" in msg or "service unavailable" in msg.lower()

        # Existing null summary preserved
        assert "30/200" in msg or "30" in msg

    def test_throttled_hint_reachable(self):
        """503 status maps to the throttling hint, as
        ``test_judge_unreliable_detector.py::test_flags_single_degraded_model`` also pins."""
        results = [{
            "model_id": "test-model",
            "data_quality": {"reliability": "degraded", "null_count": 30, "total": 200},
            "results": [
                {
                    "metrics_data": [
                        {"name": "Correctness", "score": None, "error": "Client error '503 Service Unavailable' ..."}
                    ]
                } for _ in range(30)
            ],
        }]
        error = _detect_judge_unreliable(results)
        assert error is not None
        assert "throttl" in (error.hint or "").lower()

    def test_no_judge_endpoint_or_model_asserted(self):
        """Judge halt still fires when the run carries no judge endpoint or model.

        The run data names only the target model and endpoint, so there is no
        judge identifier for the message to report. The message wording itself
        is not asserted here.
        """
        results = [{
            "model_id": "target-model-id",
            "hyperparameters": {"endpoint": "runtime"},
            "data_quality": {"reliability": "degraded", "null_count": 30, "total": 200},
            "results": [
                {
                    "metrics_data": [
                        {"name": "Correctness", "score": None, "error": "Client error '503 ..."}
                    ]
                } for _ in range(30)
            ],
        }]
        error = _detect_judge_unreliable(results)
        assert error is not None


class TestStatusValuesUnchanged:
    """Verify halt status strings remain unchanged for resume allowlist."""

    def test_generation_unreliable_status_unchanged(self):
        """detect_generation_unreliable returns code EVAL_GENERATION_UNRELIABLE."""
        results = [
            _model_run_with_generation_failures(
                model_id="test-model",
                endpoint="runtime",
                region="us-east-1",
                failed_count=10,
                total_cases=50,
                status_code=503,
            )
        ]
        error = detect_generation_unreliable(results)
        assert error is not None
        assert error.code == Codes.EVAL_GENERATION_UNRELIABLE

    def test_judge_unreliable_status_unchanged(self):
        """_detect_judge_unreliable returns code EVAL_JUDGE_UNRELIABLE."""
        results = [{
            "model_id": "test-model",
            "data_quality": {"reliability": "degraded", "null_count": 30, "total": 200},
        }]
        error = _detect_judge_unreliable(results)
        assert error is not None
        assert error.code == Codes.EVAL_JUDGE_UNRELIABLE
