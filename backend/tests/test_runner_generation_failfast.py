# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generation-error visibility + empty-output fail-fast in runner.py.

  * Generation exceptions under json_progress=True emit a
    [generation-error] stderr line (the channel harness_runner._drain_stderr
    forwards to the coordinator log) + a [generation] N/total summary.
  * Empty-output / errored cases are NOT sent to evaluate() /
    _retry_null_metrics, ARE counted into data_quality.generation_failed_count,
    leave the judge-only ``reliability`` field untouched, and still render in
    result["results"].

Loads the harness template's ``runner.py`` via the synthetic-package loader
pattern from ``test_runner_mantle_truncated.py`` so the import graph stays
clean (no real ``src.*`` package needed).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from deepeval.test_case import LLMTestCase

from tests.harness_template_stubs import TEMPLATE_ROOT, load_template_runner


def _load_template_runner() -> ModuleType:
    """Load the template ``runner.py`` with the shared ``src.*`` stubs."""
    return load_template_runner("harness_template_runner_failfast")


@pytest.fixture(scope="module")
def template_runner() -> ModuleType:
    return _load_template_runner()


def _make_runner(template_runner: ModuleType, json_progress: bool = False):
    """Construct an EvalRunner with the minimum config needed for
    ``generate_outputs`` / ``run_single_model`` to run.
    """
    cfg = MagicMock()
    cfg.metrics_json = "[]"
    cfg.results_dir = "/tmp"
    cfg.max_concurrent = 5
    runner = template_runner.EvalRunner(cfg, json_progress=json_progress)
    return runner


def _case(idx: int, **metadata) -> LLMTestCase:
    """Minimal LLMTestCase for the runner's generation loop."""
    return LLMTestCase(
        input=f"q-{idx}",
        actual_output="",
        expected_output=f"a-{idx}",
        metadata=metadata or None,
    )


class _FakeRaisingModel:
    """Model whose ``generate_with_system`` raises a configured exception."""

    def __init__(self, exc: Exception, model_name: str = "openai.gpt-5.5"):
        self._exc = exc
        self._name = model_name

    def get_model_name(self) -> str:
        return self._name

    def generate_with_system(self, system_prompt, user_prompt):
        raise self._exc


# --------------------------------------------------------------------------- #
# Generation-error stderr emission under json_progress=True                   #
# --------------------------------------------------------------------------- #


def test_generation_error_emits_stderr_under_json_progress(template_runner, capsys):
    """A generation exception under json_progress=True emits [generation-error] to stderr.

    The stderr line is unconditional, so harness_runner._drain_stderr forwards
    it to the coordinator log even when stdout carries JSON progress.
    """
    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(RuntimeError("boom"))

    # Drive generate_outputs with one case
    result = runner.generate_outputs(model, "You are a helpful assistant.", [_case(0)])

    # Case should be in result with empty actual_output + error metadata
    assert len(result) == 1
    assert result[0].actual_output == ""
    assert result[0].metadata["error"] == "boom"

    # Capture stderr
    captured = capsys.readouterr()
    # stderr carries [generation-error] + case index + model name + exception info
    assert "[generation-error]" in captured.err
    assert "boom" in captured.err


def test_generation_summary_emits_stderr_when_failures_exist(template_runner, capsys):
    """After the generation loop, a [generation] N/total stderr line is emitted."""
    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(RuntimeError("boom"))

    # Drive with one failing case
    runner.generate_outputs(model, "You are a helpful assistant.", [_case(0)])

    captured = capsys.readouterr()
    # stderr carries the [generation] 1/1 summary
    assert "[generation]" in captured.err
    assert "1/1" in captured.err or "1 / 1" in captured.err


# --------------------------------------------------------------------------- #
# empty-output fail-fast (not sent to evaluate / generation_failed_count)      #
# --------------------------------------------------------------------------- #


def test_empty_output_case_not_sent_to_evaluate(template_runner):
    """Empty-output / errored cases are excluded from evaluate() test_cases.

    Only non-empty, non-errored cases are passed to the judge.
    """
    from unittest.mock import MagicMock, patch
    from deepeval.evaluate.types import EvaluationResult, TestResult

    runner = _make_runner(template_runner, json_progress=False)

    # Build two pre-generated cases: one good, one empty
    good_case = LLMTestCase(
        input="q-0",
        actual_output="ok",
        expected_output="a-0",
        metadata={"case_index": 0},
    )
    empty_case = LLMTestCase(
        input="q-1",
        actual_output="",
        expected_output="a-1",
        metadata={"case_index": 1, "error": "boom"},
    )

    # Patch generate_outputs to return the pre-built cases
    with patch.object(runner, "generate_outputs", return_value=[good_case, empty_case]):
        # Patch _make_model to return a trivial fake
        fake_model = MagicMock()
        fake_model.get_model_name.return_value = "test.model"
        with patch.object(runner, "_make_model", return_value=fake_model):
            # Patch evaluate to record what test_cases it receives
            mock_evaluate = MagicMock()
            # Build a minimal EvaluationResult with one TestResult for the good case
            test_result_good = TestResult(
                name="test_case_0",
                success=True,
                metrics_data=[],
                conversational=False,
                multimodal=False,
                input="q-0",
                actual_output="ok",
                expected_output="a-0",
                context=None,
                retrieval_context=None,
                metadata={"case_index": 0},
            )
            eval_result = EvaluationResult(
                test_results=[test_result_good],
                conversational=False,
                multimodal=False,
                confident_link="",
                test_run_id="test-run",
            )
            mock_evaluate.return_value = eval_result

            with patch.object(template_runner, "evaluate", mock_evaluate):
                # Run the full flow
                result = runner.run_single_model(
                    model_id="test.model",
                    system_prompt="You are a helpful assistant.",
                    base_cases=[good_case, empty_case],  # Doesn't matter, generate_outputs is patched
                )

                # evaluate() received only the good case.
                assert mock_evaluate.called
                call_kwargs = mock_evaluate.call_args.kwargs
                assert "test_cases" in call_kwargs
                test_cases_passed = call_kwargs["test_cases"]
                assert len(test_cases_passed) == 1, \
                    f"Only evaluable (non-empty) cases should be sent to evaluate, got {len(test_cases_passed)}"
                assert test_cases_passed[0].actual_output == "ok"

                # Domain split: generation_failed_count exists and is tracked.
                assert "generation_failed_count" in result["data_quality"]
                assert result["data_quality"]["generation_failed_count"] == 1

                # Domain split: generation_failed_ratio exists (0.5 = 1/2 cases).
                assert "generation_failed_ratio" in result["data_quality"]
                expected_ratio = round(1 / 2, 4)  # 1 failed of 2 total
                assert result["data_quality"]["generation_failed_ratio"] == expected_ratio

                # Domain split: reliability is JUDGE-ONLY now, so a generation-failed
                # case with no judge failures yields reliability="ok". The backend
                # owns the tolerance decision for the generation domain separately.
                assert result["data_quality"]["reliability"] == "ok"

                # The empty case still appears in result["results"] with its error
                assert len(result["results"]) == 2  # Both cases rendered
                # Find the empty case in results (case_index=1)
                empty_result = next(
                    (r for r in result["results"] if r["additional_metadata"].get("case_index") == 1),
                    None,
                )
                assert empty_result is not None
                assert empty_result["actual_output"] == ""
                assert "error" in empty_result["additional_metadata"]
                assert empty_result["additional_metadata"]["error"] == "boom"
