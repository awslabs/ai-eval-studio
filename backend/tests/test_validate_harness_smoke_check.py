# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``_check_smoke_output`` — the validation-time checker that
decides whether a 1-case smoke eval's output file is acceptable.

Regression guard for the bug that shipped a broken harness where
`actual_output` was empty across every case: DeepEval scored 0 across
the board, which the previous checker accepted as "valid metric data."
The updated checker must flag empty output as its own problem class.
"""

from __future__ import annotations

from app.services.mcp_tools.harness import _check_smoke_output


def _case(actual_output: str, metrics: list[dict] | None = None) -> dict:
    """Build a minimally-valid case dict for the smoke-output checker."""
    return {
        "name": "case_0",
        "success": True,
        "input": "q",
        "actual_output": actual_output,
        "expected_output": "a",
        "additional_metadata": {},
        "metrics_data": metrics if metrics is not None else [
            {"name": "ExactMatch", "score": 1.0, "threshold": 0.5, "success": True},
        ],
    }


def _run(cases: list[dict]) -> list[dict]:
    return [{"test_run_id": "model-0", "results": cases}]


def test_passes_when_output_is_present_and_metrics_scored():
    problems = _check_smoke_output(_run([_case("a real model answer")]))
    assert problems == []


def test_flags_empty_actual_output_as_its_own_problem():
    """Empty output means generation failed silently. The checker must
    catch this before it blames the metrics."""
    problems = _check_smoke_output(_run([_case("")]))
    assert any("empty output" in p.lower() for p in problems)


def test_flags_whitespace_only_actual_output():
    problems = _check_smoke_output(_run([_case("   \n  ")]))
    assert any("empty output" in p.lower() for p in problems)


def test_empty_output_skips_metric_checks():
    """When output is empty, metric scores are meaningless — the checker
    shouldn't also complain about metric scores (which would hide the
    real root cause under a pile of noise)."""
    problems = _check_smoke_output(_run([
        _case("", metrics=[
            {"name": "ExactMatch", "score": None, "threshold": 0.5, "success": False},
        ]),
    ]))
    # Exactly one problem: the empty output. Metric-score complaints are
    # suppressed because the metric score being 0/None is a downstream
    # symptom of empty output.
    assert len(problems) == 1
    assert "empty output" in problems[0].lower()


def test_non_string_actual_output_is_flagged():
    """If the runner serializes actual_output as None or a number, that's
    still "generation didn't produce text" — treat it the same."""
    problems = _check_smoke_output(_run([_case(None)]))  # type: ignore[arg-type]
    assert any("empty output" in p.lower() for p in problems)


def _truncated_case() -> dict:
    """Build a case dict mirroring the runner's MantleReasoningTruncated
    branch — empty actual_output AND ``MANTLE_REASONING_TRUNCATED: True``
    in additional_metadata. Reproduces the on-disk shape downstream
    consumers see."""
    return {
        "name": "case_0",
        "success": False,
        "input": "q",
        "actual_output": "",
        "expected_output": "a",
        "additional_metadata": {
            "case_index": 0,
            "MANTLE_REASONING_TRUNCATED": True,
            "error": (
                "openai.gpt-5.4: Responses returned status='incomplete' "
                "reason='max_output_tokens'"
            ),
        },
        "metrics_data": [
            {"name": "ExactMatch", "score": None, "threshold": 0.5, "success": False},
        ],
    }


def test_reasoning_truncated_gets_specific_failure_message():
    """When a case carries ``MANTLE_REASONING_TRUNCATED: True``, the
    smoke checker must emit a truncation-specific failure that tells
    the user to raise ``max_output_tokens`` — NOT the generic
    "check prompt rendering" message that misdirects the fix.

    Distinguishes the reasoning-model truncation case from genuinely
    empty output (wrong protocol routing, broken prompt template,
    Bedrock validation rejection) which need different fixes.
    """
    runs = [{"test_run_id": "model-0", "model_id": "openai.gpt-5.4",
             "results": [_truncated_case()]}]
    problems = _check_smoke_output(runs)
    assert problems, "truncated case must surface as a problem"
    text = " ".join(problems).lower()
    # Truncation-specific signal — the user must know which knob to turn.
    assert "reasoning truncated" in text
    assert "max_output_tokens" in text
    # Must NOT misdirect to prompt rendering or Bedrock wrapper handling.
    assert "prompt rendering" not in text
    assert "system-prompt handling" not in text


def test_generic_empty_output_still_uses_generic_message():
    """A case with empty output but NO truncation flag stays on the
    generic empty-output message (the existing failure mode covers
    Converse-on-mantle protocol mismatch, broken prompt rendering, etc.).
    Regression guard against accidentally collapsing both branches into
    the truncation-specific text."""
    runs = [{"test_run_id": "model-0", "model_id": "us.anthropic.claude-sonnet-4-6",
             "results": [_case("")]}]
    problems = _check_smoke_output(runs)
    text = " ".join(problems).lower()
    assert "empty output" in text
    # NOT the truncation-specific text.
    assert "reasoning truncated" not in text
