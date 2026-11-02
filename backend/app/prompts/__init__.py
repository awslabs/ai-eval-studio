# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt markdown files for the AI Eval Studio agent.

This package contains .md prompt files for the coordinator and subagents.
The build_coordinator_prompt() function lives in app.services.coordinator_prompt.

This package owns the single authoritative derivation of the packaged prompts directory.
Callers import PROMPTS_DIR instead of recomputing it from __file__ + parent-chain arithmetic.

Rationale: A hand-rolled parent chain derives the prompts directory relative to the caller's
own file location. If a caller is moved up or down the tree (e.g., refactoring a function from
app/services/mcp_tools/dispatch.py to app/services/optimization_loop_orchestrator.py), the
same arithmetic silently breaks because it now lands on a different directory. An anchor that
IS the package's own directory travels with the package, so the resolution cannot drift.
"""

from pathlib import Path

# Single anchor for the packaged prompts directory. This derivation is zero parent hops,
# because the package directory IS the resource directory (backend/app/prompts/).
PROMPTS_DIR = Path(__file__).resolve().parent

# Static resources this package must ship. Adding a .md file that the runtime reads
# means adding it here.
_REQUIRED_PROMPT_FILES = (
    "_rules.md",
    "coordinator.md",
    "data-analyzer.md",
    "deepeval-metric-reference.md",
    "harness-builder.md",
    "prompt-optimizer.md",
    "report-narrator.md",
)


def verify_prompt_resources() -> None:
    """Verify all packaged prompt resources exist.

    Enumerates the expected prompt markdown files through the same PROMPTS_DIR anchor
    the runtime reads from. If any are missing, raises a single RuntimeError naming
    the resolved directory and all missing files.

    This check deliberately resolves through the same anchor the runtime uses. Deriving
    the expected path independently would let the check pass while the runtime resolution
    is wrong, which is the exact failure mode this function exists to prevent.

    Raises:
        RuntimeError: One or more prompt resources are missing from PROMPTS_DIR.
    """
    missing = [name for name in _REQUIRED_PROMPT_FILES if not (PROMPTS_DIR / name).exists()]
    if missing:
        raise RuntimeError(
            f"Missing prompt resources in {PROMPTS_DIR}: {sorted(missing)}. "
            f"This is a build defect — prompt markdown files are packaged static resources."
        )


__all__ = ["PROMPTS_DIR", "verify_prompt_resources"]
