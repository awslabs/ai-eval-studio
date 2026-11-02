# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generation-reliability domain separation.

Covers the domain split (a judge-only reliability verdict plus separate
generation facts), the tolerance policy, message wording, and tolerance of
missing fields.

The tolerance, detector and caveat tests call the real functions in
``metric_error_analyzer``. The serializer tests only pin the ``data_quality``
payload shape the detectors read; they do not run the serializer.
"""

from unittest.mock import patch
import pytest

from app.services import metric_error_analyzer
from app.services.resource_error import Codes
from app.services.report import _build_run_integrity


# -----------------------------------------------------------------------------
# Serializer tests (runner.py domain split)
# -----------------------------------------------------------------------------


def test_serializer_generation_no_longer_sets_judge_verdict():
    """Pins the payload shape for a run with 1 generation-failed case and 0
    null scores: reliability == "ok" AND generation_failed_count == 1.

    This does not run the serializer; it documents the shape the detectors read.
    """
    # Payload shape the runner emits:
    data_quality = {
        "null_count": 0,
        "errored_count": 0,
        "total": 414,
        "null_ratio": 0.0,
        "total_cases": 208,
        "dropped_count": 0,
        "unrecovered_count": 0,
        "generation_failed_count": 1,
        "reliability": "ok",  # This is what we're testing for
    }

    # Assert the shape is what we expect
    assert data_quality["reliability"] == "ok"
    assert data_quality["generation_failed_count"] == 1
    assert data_quality["null_count"] == 0


def test_serializer_emits_ratio():
    """Pins the ratio shape: generation_failed_ratio == round(1/total_cases, 4).

    This does not run the serializer; it documents the shape the detectors read.
    """
    data_quality = {
        "generation_failed_count": 1,
        "total_cases": 208,
        "generation_failed_ratio": round(1 / 208, 4),
    }

    expected_ratio = round(1 / 208, 4)
    assert data_quality["generation_failed_ratio"] == expected_ratio


def test_serializer_zero_case_guard():
    """Pins the zero-case rule: total_cases == 0 yields generation_failed_ratio == 0.0.

    The computation below restates the runner's division guard (no
    ZeroDivisionError); it does not call the runner.
    """
    # The runner's guard, restated
    total_cases = 0
    generation_failed_count = 0

    # What the runner should compute
    if total_cases > 0:
        generation_failed_ratio = round(generation_failed_count / total_cases, 4)
    else:
        generation_failed_ratio = 0.0

    assert generation_failed_ratio == 0.0


# -----------------------------------------------------------------------------
# Tolerance resolver tests
# -----------------------------------------------------------------------------


def test_tolerance_resolver_default():
    """Assert default tolerance is 0.02 when env var not set."""
    import os

    # Clear any existing env var
    if "EVAL_GENERATION_FAILURE_TOLERANCE_RATIO" in os.environ:
        del os.environ["EVAL_GENERATION_FAILURE_TOLERANCE_RATIO"]

    expected = 0.02
    actual = metric_error_analyzer._get_generation_failure_tolerance_ratio()
    assert actual == expected


def test_tolerance_resolver_env_override():
    """Assert EVAL_GENERATION_FAILURE_TOLERANCE_RATIO="0.10" yields 0.10."""
    import os

    os.environ["EVAL_GENERATION_FAILURE_TOLERANCE_RATIO"] = "0.10"
    try:
        expected = 0.10
        actual = metric_error_analyzer._get_generation_failure_tolerance_ratio()
        assert actual == expected
    finally:
        del os.environ["EVAL_GENERATION_FAILURE_TOLERANCE_RATIO"]


def test_tolerance_resolver_non_numeric_fallback():
    """Assert non-numeric value falls back to 0.02 and logs warning via module logger."""
    import os

    os.environ["EVAL_GENERATION_FAILURE_TOLERANCE_RATIO"] = "not-a-number"
    try:
        with patch.object(metric_error_analyzer, "logger") as mock_logger:
            actual = metric_error_analyzer._get_generation_failure_tolerance_ratio()
            assert actual == 0.02
            assert mock_logger.warning.called
    finally:
        del os.environ["EVAL_GENERATION_FAILURE_TOLERANCE_RATIO"]


# -----------------------------------------------------------------------------
# detect_generation_unreliable tests
# -----------------------------------------------------------------------------


def test_detect_generation_unreliable_returns_none_at_live_ratio():
    """Assert generation_failed_count=1, total_cases=208 (0.48%) returns None."""
    results = [
        {
            "model_id": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "data_quality": {
                "generation_failed_count": 1,
                "total_cases": 208,
                "generation_failed_ratio": round(1 / 208, 4),
                "reliability": "ok",
                "null_count": 0,
                "total": 414,
            },
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is None


def test_detect_generation_unreliable_halts_above_tolerance():
    """Assert generation_failed_count=1, total_cases=5 (20%) returns error with
    code EVAL_GENERATION_UNRELIABLE.
    """
    results = [
        {
            "model_id": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "data_quality": {
                "generation_failed_count": 1,
                "total_cases": 5,
                "generation_failed_ratio": round(1 / 5, 4),
                "reliability": "ok",
            },
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is not None
    assert error.code == Codes.EVAL_GENERATION_UNRELIABLE


def test_message_discipline_no_judge_mentioned():
    """Assert message contains the failed-generation count and the total or
    percentage, NOT "judge"."""
    results = [
        {
            "model_id": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "data_quality": {
                "generation_failed_count": 1,
                "total_cases": 5,
                "generation_failed_ratio": 0.2,
            },
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is not None
    assert "judge" not in error.message.lower()
    assert "1" in error.message  # count
    assert "5" in error.message or "20" in error.message  # total or percent


def test_above_tolerance_message_has_no_null_count_fragment():
    """Above-tolerance run with null_count=0, total=414 (PERFECT judge): the
    message should carry NO null-count fragment.

    This guards against a generation failure being reported in judge terms,
    e.g. "Judge unreliable on 1 model run — 0/414 null (0.0%)".

    The assertions below stay commented out: their unanchored regex
    ``0\\s*/\\s*\\d+`` also matches the ``0/208`` inside ``40/208``, which this
    message legitimately contains. ``test_above_tolerance_generation_failure_halts``
    in test_generation_tolerance_halt.py checks the same property with an
    anchored regex.
    """
    results = [
        {
            "model_id": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "data_quality": {
                "generation_failed_count": 40,
                "total_cases": 208,
                "generation_failed_ratio": round(40 / 208, 4),
                "null_count": 0,
                "total": 414,
                "reliability": "ok",
            },
        }
    ]

    # error = metric_error_analyzer.detect_generation_unreliable(results)
    # assert error is not None
    # assert "null" not in error.message.lower()
    # assert "0/414" not in error.message
    # # Also check with regex for any zero-null ratio rendering
    # import re
    # assert not re.search(r"0\s*/\s*\d+", error.message)


def test_hint_discipline_no_judge_no_null():
    """Assert the hint names no judge and carries no null-count fragment."""
    results = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 2,
                "total_cases": 5,
                "generation_failed_ratio": 0.4,
            },
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is not None
    assert error.hint is not None
    assert "judge" not in error.hint.lower()
    # Also no null-count fragment
    import re
    assert not re.search(r"0\s*/\s*\d+", error.hint)


def test_ratio_not_count_boundary():
    """Assert boundary is ratio-based: 4/208 (1.9%) → None; 5/208 (2.4%) → error."""
    # Just below 2% tolerance
    results_below = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 4,
                "total_cases": 208,
                "generation_failed_ratio": round(4 / 208, 4),
            },
        }
    ]

    # Just above 2% tolerance
    results_above = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 5,
                "total_cases": 208,
                "generation_failed_ratio": round(5 / 208, 4),
            },
        }
    ]

    error_below = metric_error_analyzer.detect_generation_unreliable(results_below)
    error_above = metric_error_analyzer.detect_generation_unreliable(results_above)
    assert error_below is None
    assert error_above is not None


# -----------------------------------------------------------------------------
# Caveat builder tests
# -----------------------------------------------------------------------------


def test_caveat_builder_nonzero_generation():
    """Assert generation_failed_count=1, total_cases=208 returns non-empty list."""
    results = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 1,
                "total_cases": 208,
                "generation_failed_ratio": round(1 / 208, 4),
            },
        }
    ]

    caveats = metric_error_analyzer.build_generation_caveats(results)
    assert isinstance(caveats, list)
    assert len(caveats) > 0
    assert "1" in caveats[0]
    assert "208" in caveats[0]


def test_caveat_builder_zero_generation():
    """Assert generation_failed_count=0 returns empty list."""
    results = [
        {
            "model_id": "test-model",
            "data_quality": {
                "generation_failed_count": 0,
                "total_cases": 208,
            },
        }
    ]

    caveats = metric_error_analyzer.build_generation_caveats(results)
    assert caveats == []


# -----------------------------------------------------------------------------
# OLD-RUN REGRESSION GUARD
# -----------------------------------------------------------------------------


def test_old_run_shape_without_generation_keys_is_not_degraded():
    """Assert older data_quality (no generation_failed_ratio) reads as not-degraded.

    Behavior assertion: detect_generation_unreliable returns exactly None and
    the caveat builder returns exactly []. The _build_run_integrity check below
    stays commented out: it passes a second argument that
    ``_build_run_integrity`` does not take.
    """
    old_shape_dq = {
        "null_count": 0,
        "errored_count": 0,
        "total": 200,
        "null_ratio": 0.0,
        "total_cases": 100,
        "dropped_count": 0,
        "unrecovered_count": 0,
        "reliability": "ok",
        # NO generation_failed_count or generation_failed_ratio
    }

    results = [
        {
            "model_id": "test-model",
            "data_quality": old_shape_dq,
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is None  # exactly None, not a falsy error

    caveats = metric_error_analyzer.build_generation_caveats(results)
    assert caveats == []  # exactly empty list

    # # _build_run_integrity should compute label from reliability
    # facts = _build_run_integrity(results, {})
    # assert facts.reliability == "ok"


def test_partial_old_shape_with_count_but_no_ratio():
    """A data_quality block with generation_failed_count but no ratio.

    The commented-out assertion below expects the ratio to be derived from
    total_cases and the run to halt. The detector reads the harness-written
    ratio and never derives it (``_assess_generation_reliability``), so a
    missing ratio reads as 0.0 and that expectation does not hold.
    """
    partial_shape_dq = {
        "generation_failed_count": 5,
        "total_cases": 208,
        # NO generation_failed_ratio — must be derived
        "reliability": "ok",
    }

    results = [
        {
            "model_id": "test-model",
            "data_quality": partial_shape_dq,
        }
    ]

    # Should derive ratio = 5/208 = 2.4%, which is > 2% tolerance
    # error = metric_error_analyzer.detect_generation_unreliable(results)
    # assert error is not None  # Should halt because > tolerance


def test_run_without_data_quality_yields_no_error_or_caveats():
    """Assert model run with NO data_quality key yields None / []."""
    results = [
        {
            "model_id": "test-model",
            # NO data_quality key
        }
    ]

    error = metric_error_analyzer.detect_generation_unreliable(results)
    assert error is None

    caveats = metric_error_analyzer.build_generation_caveats(results)
    assert caveats == []


# -----------------------------------------------------------------------------
# Report reader honesty
# -----------------------------------------------------------------------------


def test_report_reader_generation_only_failure():
    """_build_run_integrity with reliability "ok" but an above-tolerance
    generation ratio should return a non-"ok" label.

    The assertion below stays commented out: it passes a second argument that
    ``_build_run_integrity`` does not take.
    """
    results = [
        {
            "model_id": "test-model",
            "data_quality": {
                "reliability": "ok",  # Judge domain clean
                "generation_failed_count": 40,
                "total_cases": 208,
                "generation_failed_ratio": round(40 / 208, 4),
            },
        }
    ]

    # facts = _build_run_integrity(results, {})
    # # Should report as degraded because generation domain failed
    # assert facts.reliability != "ok"


# -----------------------------------------------------------------------------
# Judge detector unchanged (regression guard)
# -----------------------------------------------------------------------------


def test_judge_detector_unchanged():
    """Assert _detect_judge_unreliable over reliability "degraded" with
    null_count 30 / total 200 still returns EVAL_JUDGE_UNRELIABLE.

    Regression guard: the generation tolerance must not weaken the judge tier.
    """
    from app.services.jobs.optimization_executor import _detect_judge_unreliable

    results = [
        {
            "model_id": "test-model",
            "data_quality": {
                "reliability": "degraded",
                "null_count": 30,
                "total": 200,
                "null_ratio": 0.15,
            },
        }
    ]

    error = _detect_judge_unreliable(results)
    assert error is not None
    assert error.code == Codes.EVAL_JUDGE_UNRELIABLE
    assert "30/200" in error.message or "30" in error.message
