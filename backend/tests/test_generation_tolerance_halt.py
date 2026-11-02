# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generation-tolerance halt tiers.

Coverage: detector behavior (``detect_generation_unreliable``,
``_detect_judge_unreliable``, ``detect_infra_failure``) and the caveat builder
(``build_generation_caveats``).

These tests call the detectors directly rather than driving
``run_optimization_iteration_job`` and asserting on reloaded, persisted status:
the full executor flow requires extensive mocking (job tracker, eval job,
project manager, platform arch_split, metadata store) that would make the tests
brittle without adding coverage beyond what the detector tests provide.
"""

from app.services.resource_error import Codes


def test_sub_tolerance_generation_failure_does_not_halt():
    """Assert sub-tolerance generation failure proceeds without halt.

    Tests detector behavior: calls detect_generation_unreliable directly.
    """
    # Payload with sub-tolerance generation failure (1/208 = 0.48%)
    results_payload = [
        {
            "model_id": "test-model",
            "data_quality": {
                "reliability": "ok",
                "null_count": 0,
                "total": 414,
                "generation_failed_count": 1,
                "total_cases": 208,
                "generation_failed_ratio": round(1 / 208, 4),
            },
        }
    ]

    from app.services.metric_error_analyzer import detect_generation_unreliable
    error = detect_generation_unreliable(results_payload)
    assert error is None


def test_above_tolerance_generation_failure_halts():
    """Assert above-tolerance generation failure halts with generation_unreliable.

    Tests detector behavior: calls detect_generation_unreliable directly.
    """
    # Payload with above-tolerance generation failure (40/208 = 19%)
    results_payload = [
        {
            "model_id": "test-model",
            "data_quality": {
                "reliability": "ok",
                "generation_failed_count": 40,
                "total_cases": 208,
                "generation_failed_ratio": round(40 / 208, 4),
            },
        }
    ]

    from app.services.metric_error_analyzer import detect_generation_unreliable
    error = detect_generation_unreliable(results_payload)
    assert error is not None
    assert error.code == Codes.EVAL_GENERATION_UNRELIABLE

    # Message discipline: no "judge" and no null-count fragments
    assert "judge" not in error.message.lower()
    assert "null" not in error.message.lower()
    assert "0/414" not in error.message

    # Regex check for zero-null ratio - anchored to catch literal "0/X" pattern
    # (not mid-number like "40/208"). Must have non-digit or string start before the 0.
    import re
    assert not re.search(r"(?:^|[^\d])0\s*/\s*\d+", error.message)


def test_judge_degraded_run_still_halts_as_judge_unreliable():
    """Assert judge-degraded run still halts as judge_unreliable.

    Regression guard: the generation tolerance must not weaken the judge tier.
    Tests detector behavior: calls _detect_judge_unreliable directly.
    """
    from app.services.jobs.optimization_executor import _detect_judge_unreliable

    results_payload = [
        {
            "model_id": "test-model",
            "data_quality": {
                "reliability": "degraded",
                "null_count": 30,
                "total": 200,
                "null_ratio": 0.15,
                "generation_failed_count": 0,
            },
        }
    ]

    error = _detect_judge_unreliable(results_payload)
    assert error is not None
    assert error.code == Codes.EVAL_JUDGE_UNRELIABLE
    assert "30/200" in error.message or "30" in error.message


def test_dominant_generation_failure_is_a_hard_fail():
    """Assert >=80% generation-failed yields GENERATION_ERRORS_DOMINANT (hard fail).

    Tier ordering: the hard-fail tier wins over the tolerance tier.
    Tests detector behavior: calls detect_infra_failure directly.
    """
    from app.services.metric_error_analyzer import detect_infra_failure

    # Payload where >=80% cases are generation-failed
    results_payload = [
        {
            "model_id": "test-model",
            "results": [
                {"actual_output": "", "additional_metadata": {"error": "boom"}}
                for _ in range(85)
            ] + [
                {"actual_output": "ok", "additional_metadata": {}}
                for _ in range(15)
            ],
        }
    ]

    error = detect_infra_failure(results_payload)
    assert error is not None
    assert error.code == Codes.EVAL_GENERATION_ERRORS_DOMINANT


def test_sub_tolerance_failure_produces_caveat():
    """Assert a sub-tolerance failure produces a caveat line.

    Tests caveat builder behavior: calls build_generation_caveats directly.
    """
    from app.services.metric_error_analyzer import build_generation_caveats

    results_payload = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 1,
                "total_cases": 208,
                "generation_failed_ratio": round(1 / 208, 4),
            },
        }
    ]

    caveats = build_generation_caveats(results_payload)
    assert isinstance(caveats, list)
    assert len(caveats) > 0
    assert "1/208" in caveats[0]
    assert "within tolerance" in caveats[0]


def test_zero_generation_failures_produce_no_caveat():
    """Assert zero generation failures yields no caveat.

    Tests caveat builder behavior: calls build_generation_caveats directly.
    """
    from app.services.metric_error_analyzer import build_generation_caveats

    results_payload = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 0,
                "total_cases": 208,
            },
        }
    ]

    caveats = build_generation_caveats(results_payload)
    assert caveats == []
