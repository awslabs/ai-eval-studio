# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for infra-failure detection wiring in eval and optimization executors.

The executors call detect_infra_failure to catch both generation-phase
and metric-phase infrastructure failures. These tests verify the wiring
at the detector import/call level — asserting the correct function is
imported and invoked with the expected result types.
"""

from app.services.metric_error_analyzer import detect_infra_failure
from app.services.resource_error import Codes


def test_eval_executor_imports_detect_infra_failure():
    """eval_executor imports detect_infra_failure (not detect_metric_error_dominance)."""
    from app.services.jobs import eval_executor

    # Assert the module imports detect_infra_failure
    assert hasattr(eval_executor, "detect_infra_failure")
    # The imported symbol should be the actual function
    assert eval_executor.detect_infra_failure is detect_infra_failure


def test_optimization_executor_imports_detect_infra_failure():
    """optimization_executor imports detect_infra_failure (not detect_metric_error_dominance)."""
    from app.services.jobs import optimization_executor

    # Assert the module imports detect_infra_failure
    assert hasattr(optimization_executor, "detect_infra_failure")
    # The imported symbol should be the actual function
    assert optimization_executor.detect_infra_failure is detect_infra_failure


def test_detect_infra_failure_returns_generation_error_first():
    """Combined detector returns generation error before metric error."""
    # Build a result with both generation failures AND metric errors
    # (pathological case but tests priority)
    result = [
        {
            "test_run_id": "m1",
            "results": [
                {
                    "name": f"tc{i}",
                    "success": False,
                    "actual_output": "[]",
                    "additional_metadata": {
                        "error": "ExpiredTokenException: session expired"
                    },
                    "metrics_data": [
                        {
                            "name": "Correctness",
                            "score": 0.0,
                            "success": False,
                            "error": "ThrottlingException: rate exceeded",
                        }
                    ],
                }
                for i in range(20)
            ],
        }
    ]

    err = detect_infra_failure(result)
    assert err is not None
    # Generation failure is checked first → should return EVAL_GENERATION_ERRORS_DOMINANT
    assert err.code == Codes.EVAL_GENERATION_ERRORS_DOMINANT


def test_detect_infra_failure_falls_back_to_metric_error():
    """Combined detector falls back to metric error when no generation failures."""
    # Build a result with only metric errors (non-empty output)
    result = [
        {
            "test_run_id": "m1",
            "results": [
                {
                    "name": f"tc{i}",
                    "success": False,
                    "actual_output": "some generated output",
                    "metrics_data": [
                        {
                            "name": "Correctness",
                            "score": 0.0,
                            "success": False,
                            "error": "ThrottlingException: rate exceeded",
                        }
                    ],
                }
                for i in range(20)
            ],
        }
    ]

    err = detect_infra_failure(result)
    assert err is not None
    # No generation failures → should fall back to EVAL_METRIC_ERRORS_DOMINANT
    assert err.code == Codes.EVAL_METRIC_ERRORS_DOMINANT
