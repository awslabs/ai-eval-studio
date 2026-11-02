# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the Python lint gate from the test suite.

This project has no CI workflow and no git hook, so the test suite is the real
enforcement point — a lint config that only runs when someone remembers to type
the command is not a gate. Running it here means an undefined name fails the same
command everyone already runs before committing.

The rule set lives in the repo-root ``ruff.toml``. F821 (undefined name) is the
load-bearing member: a function that uses a name it never bound (a logger, say)
fails only when that line runs, and nothing else checks for it offline.

Fails loudly when the linter is missing rather than skipping. A gate that skips
itself reports green while protecting nothing, which is worse than having no gate
at all.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LINT_TREES = ("backend", "infra")


def _ruff_executable() -> str:
    """Locate the ruff executable, preferring one on PATH.

    Falls back to the interpreter currently running pytest, since ``ruff`` is a
    backend dev dependency and therefore installed beside that interpreter. This
    used to be a hardcoded absolute container path, so on a
    host whose venv lives anywhere else — a macOS checkout using
    ``backend/.venv`` — the gate resolved to a bare ``"ruff"``, found nothing on
    PATH, and failed while a perfectly good ruff sat in the venv running the
    test. ``sys.executable`` covers both layouts and hardcodes neither.

    Returns:
        Path or bare name of the ruff executable to invoke.
    """
    found = shutil.which("ruff")
    if found:
        return found
    venv_ruff = Path(sys.executable).parent / "ruff"
    if venv_ruff.exists():
        return str(venv_ruff)
    return "ruff"


def test_lint_gate_is_clean():
    """``ruff check`` over backend + infra must pass with the repo-root config."""
    ruff = _ruff_executable()
    assert shutil.which(ruff) or Path(ruff).exists(), (
        f"ruff not found (looked for {ruff!r}). Install the backend dev "
        "dependencies with `uv sync --group dev` from backend/. This test FAILS "
        "rather than skips on purpose: a gate that skips itself protects nothing."
    )

    result = subprocess.run(
        [ruff, "check", *LINT_TREES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "lint gate failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_undefined_name_rule_is_selected():
    """The undefined-name rule family must stay selected in ruff.toml.

    Pinned separately from the clean-tree check because dropping ``F82`` would
    leave the gate passing while silently un-guarding the exact defect class the
    gate was installed for.
    """
    config = (REPO_ROOT / "ruff.toml").read_text()
    assert "F82" in config, (
        "ruff.toml must keep F82 selected — F821 (undefined name) is the rule "
        "this gate exists for."
    )
