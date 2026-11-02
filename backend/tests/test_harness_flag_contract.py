# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flag-contract test: the harness command builder and the template
argparse must agree.

The "unrecognized arguments: --data-file" crash shipped because
``harness_runner.build_harness_command`` could emit a flag the template's
``run_eval.py`` argparse never defined, and the validation smoke eval
(which used to build its own command independently) didn't exercise it.

This test closes that gap structurally: every flag the shared builder can
emit MUST be a defined option in the template's argparse. If someone adds
a flag to the builder without adding it to the template, this test fails in
the backend suite — before any scaffolded harness can hit that crash.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.services.harness_runner import build_harness_command

TEMPLATE_RUN_EVAL = (
    Path(__file__).resolve().parent.parent
    / "app" / "templates" / "harness" / "run_eval.py"
)


def _template_defined_flags() -> set[str]:
    """Parse the template run_eval.py and return every defined --flag.

    Static AST parse (no import, no subprocess) — finds every
    ``parser.add_argument("--flag", ...)`` first positional string arg.
    """
    tree = ast.parse(TEMPLATE_RUN_EVAL.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
        ):
            flags.add(node.args[0].value)
    return flags


def _emitted_flags(cmd: list[str]) -> set[str]:
    """Extract the --flags present in a built command list."""
    return {tok for tok in cmd if tok.startswith("--")}


def test_builder_flags_are_all_defined_in_template():
    """Every flag the builder can emit (with ALL options set) must be
    defined in the template argparse — the core regression guard."""
    defined = _template_defined_flags()
    # Build with every optional flag populated so the builder emits its
    # full surface.
    cmd = build_harness_command(
        "uv",
        model_ids=["m1", "m2"],
        prompt_version="v1",
        json_progress=True,
        source="optimization",
        max_test_cases=5,
        judge_model_id="judge",
        data_file_override="/abs/held_out.csv",
        max_concurrent=3,
    )
    emitted = _emitted_flags(cmd)
    missing = emitted - defined
    assert not missing, (
        f"build_harness_command emits flags the template run_eval.py "
        f"argparse does not define: {sorted(missing)}. Add them to the "
        f"template's parser.add_argument calls, or stop emitting them."
    )


def test_data_file_flag_specifically_defined():
    """Pin the exact flag that caused the production crash."""
    assert "--data-file" in _template_defined_flags()


def test_builder_emits_data_file_when_override_set():
    """The builder must actually emit --data-file when given an override
    (otherwise the smoke eval can't exercise it)."""
    cmd = build_harness_command(
        "uv", model_ids=["m"], prompt_version="baseline",
        data_file_override="/abs/val.csv",
    )
    assert "--data-file" in cmd
    assert "/abs/val.csv" in cmd


def test_builder_omits_optional_flags_when_unset():
    """Minimal invocation emits only the always-on flags — no stray
    --data-file / --judge-model / --max-examples / --max-concurrent."""
    cmd = build_harness_command("uv", model_ids=["m"], prompt_version="baseline")
    emitted = _emitted_flags(cmd)
    assert emitted == {"--models", "--prompt-version", "--json-progress", "--source"}
