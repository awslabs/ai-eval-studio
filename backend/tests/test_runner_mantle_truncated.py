# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MantleReasoningTruncated typed catch in the harness runner.

When the harness generation loop catches a ``MantleReasoningTruncated``, it
must surface the typed signal in case metadata as
``MANTLE_REASONING_TRUNCATED: True`` BEFORE the generic ``except Exception``
swallows it as a plain error string. validate_harness's smoke-output check
(``_check_smoke_output``) reads the flag and tells the user to raise
``max_output_tokens`` instead of reporting a generic empty output.

Loads the harness template's ``runner.py`` via the synthetic-package
loader pattern from ``test_harness_template_smoke.py`` so the import
graph stays clean (no real ``src.*`` package needed).
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
    """Load the template ``runner.py`` with the shared ``src.*`` stubs.

    The shared installer supplies the REAL ``MantleReasoningTruncated`` —
    required here, because the runner's typed ``except`` matches by class
    identity against the exception these tests raise from the fake model.
    """
    return load_template_runner("harness_template_runner_truncated")


@pytest.fixture(scope="module")
def template_runner() -> ModuleType:
    return _load_template_runner()


def _make_runner(template_runner: ModuleType, json_progress: bool = False):
    """Construct an EvalRunner with the minimum config needed for
    ``generate_outputs`` to run.
    """
    cfg = MagicMock()
    cfg.metrics_json = "[]"
    cfg.results_dir = "/tmp"
    cfg.max_concurrent = 5
    runner = template_runner.EvalRunner(cfg, json_progress=json_progress)
    return runner


class _FakeRaisingModel:
    """Model whose ``generate_with_system`` raises a configured exception.

    Mirrors the BedrockRuntimeModel surface ``EvalRunner.generate_outputs``
    consumes: ``get_model_name()`` and ``generate_with_system(system,
    user)`` returning ``(text, latency_ms, usage)``.
    """

    def __init__(self, exc: Exception, model_name: str = "openai.gpt-5.5"):
        self._exc = exc
        self._name = model_name

    def get_model_name(self) -> str:
        return self._name

    def generate_with_system(self, system_prompt, user_prompt):
        raise self._exc


def _case(idx: int, **metadata) -> LLMTestCase:
    """Minimal LLMTestCase for the runner's generation loop."""
    return LLMTestCase(
        input=f"q-{idx}",
        actual_output="",
        expected_output=f"a-{idx}",
        metadata=metadata or None,
    )


def test_typed_catch_sets_metadata_flag(template_runner):
    """``MantleReasoningTruncated`` → metadata['MANTLE_REASONING_TRUNCATED']=True."""
    from app.services.bedrock_mantle_mappers import MantleReasoningTruncated

    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(
        MantleReasoningTruncated(
            "openai.gpt-5.5: status=incomplete reason=max_output_tokens"
        )
    )
    cases = [_case(0), _case(1)]
    out = runner.generate_outputs(model, "system prompt", cases)

    assert len(out) == 2
    for tc in out:
        assert tc.metadata is not None
        assert tc.metadata.get("MANTLE_REASONING_TRUNCATED") is True
        assert "max_output_tokens" in tc.metadata.get("error", "")


def test_generic_exception_does_not_set_truncated_flag(template_runner):
    """Generic ``Exception`` paths must NOT set the truncated flag.

    Regression gate: the typed catch is BEFORE the generic; a
    non-truncation error must still take the generic branch and
    leave ``MANTLE_REASONING_TRUNCATED`` unset.
    """
    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(RuntimeError("network down"))
    cases = [_case(0)]
    out = runner.generate_outputs(model, "system prompt", cases)

    assert len(out) == 1
    tc = out[0]
    assert tc.metadata is not None
    # Error captured in metadata, but the flag is NOT set.
    assert tc.metadata.get("error") == "network down"
    assert tc.metadata.get("MANTLE_REASONING_TRUNCATED") is None


def test_typed_catch_preserves_case_index(template_runner):
    """``case_index`` is the stable identity for reconciliation."""
    from app.services.bedrock_mantle_mappers import MantleReasoningTruncated

    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(MantleReasoningTruncated("trunc"))
    cases = [_case(0), _case(1), _case(2)]
    out = runner.generate_outputs(model, "system prompt", cases)

    indices = sorted(tc.metadata["case_index"] for tc in out)
    assert indices == [0, 1, 2]


def test_typed_catch_preserves_input_metadata(template_runner):
    """Original metadata fields flow through alongside the truncation flag."""
    from app.services.bedrock_mantle_mappers import MantleReasoningTruncated

    runner = _make_runner(template_runner, json_progress=True)
    model = _FakeRaisingModel(MantleReasoningTruncated("trunc"))
    cases = [_case(0, foo="bar", category="alpha")]
    out = runner.generate_outputs(model, "system prompt", cases)

    assert len(out) == 1
    md = out[0].metadata
    assert md is not None
    assert md.get("foo") == "bar"
    assert md.get("category") == "alpha"
    assert md.get("MANTLE_REASONING_TRUNCATED") is True
