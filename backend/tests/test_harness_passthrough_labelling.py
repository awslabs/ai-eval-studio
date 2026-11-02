# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess console output must never be logged as if it were ours.

DeepEval prints its own pass-rate summary to stderr, and the harness passthrough
logged it under the same ``[harness <id>]`` prefix the app uses for its own
lines. Two numbers computed differently — DeepEval counts every metric result,
the app counts cases against the gating metric — appearing equally
authoritative, with nothing in the log saying which one is the app's answer.

Structural rather than behavioural, and deliberately so: the passthrough is a
closure inside ``run_eval_via_harness``, which spawns a real subprocess, so
reaching it in a unit test would mean running an eval. This checks the property
that was missing — every passthrough line carries a provenance tag — and would
catch a third passthrough added without one. It does NOT prove the tag reaches a
log file; that was verified by reading a real run's output.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.services import harness_runner

_SOURCE = Path(inspect.getfile(harness_runner)).read_text()

#: The prefix the app's own harness-related log lines use. A passthrough line
#: shares it, which is exactly why it needs something more.
_JOB_PREFIX = "[harness %s]"

_TAGS = (harness_runner._STDERR_TAG, harness_runner._STDOUT_TAG)


def _passthrough_log_calls() -> list[ast.Call]:
    """Every ``logger.info`` call in the module whose format uses the job prefix."""
    tree = ast.parse(_SOURCE)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "info"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "logger"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if _JOB_PREFIX in first.value:
                calls.append(node)
    return calls


def test_the_module_still_has_passthrough_log_calls_to_check():
    """Guards the guard: if the passthrough moves, this file must follow it."""
    assert _passthrough_log_calls(), (
        f"no logger.info call in harness_runner formats with {_JOB_PREFIX!r} any "
        "more. The passthrough moved or was renamed — point this test at it "
        "rather than deleting it."
    )


@pytest.mark.parametrize("tag", _TAGS)
def test_each_tag_is_distinguishable_from_an_app_line(tag):
    """A tag that looked like an app prefix would defeat the whole point."""
    assert tag.strip() and tag not in _JOB_PREFIX


def test_every_passthrough_line_is_tagged_with_its_stream():
    """The defect: an untagged passthrough reads as an app measurement."""
    untagged = [
        ast.unparse(call)
        for call in _passthrough_log_calls()
        if not any(
            isinstance(arg, ast.Name) and arg.id in ("_STDERR_TAG", "_STDOUT_TAG")
            for arg in call.args
        )
    ]
    assert not untagged, (
        "these harness log calls forward subprocess output without saying so, so "
        "a pass rate printed by DeepEval is indistinguishable from the app's "
        f"own: {untagged}"
    )


def test_the_tags_name_the_stream_rather_than_guessing_the_library():
    """Attributing a stderr line to DeepEval would be a confident wrong guess.

    The same stream carries genuine harness tracebacks and uv output. The tag
    says where the line came from, which is knowable, instead of who wrote it,
    which is not.
    """
    for tag in _TAGS:
        assert "deepeval" not in tag.lower(), (
            f"{tag!r} attributes the line to a library, but harness tracebacks "
            "and uv output share this stream"
        )
    assert "stderr" in harness_runner._STDERR_TAG
    assert "stdout" in harness_runner._STDOUT_TAG
