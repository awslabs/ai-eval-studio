# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every SDK spawn site must raise the 1 MiB stdout-frame ceiling.

The failure this exists for: the report narrator died in cloud with ``JSON
message exceeded maximum buffer size of 1048576 bytes``, raised by the SDK's
own stdout reader
(``claude_agent_sdk/_internal/transport/subprocess_cli.py``). The default frame
ceiling is 1 MiB and this backend legitimately puts more than that on the wire:
MCP tool results are admitted up to ``MAX_MCP_PAYLOAD_BYTES`` (900,000) and the
CLI echoes both tool results and user prompts back as stdout frames.

Two properties are checked, because getting one right is not enough:

1. The ceiling is derived from the tool guard, so raising that guard cannot
   silently re-open the same hole.
2. EVERY spawn site applies it. This class of bug is "the guard exists and one
   call site doesn't use it" — exactly how the narrator's failed-case sampler
   also came to no-op on the executive path.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.services.mcp_response import MAX_MCP_PAYLOAD_BYTES
from app.services.sdk_subprocess_env import SDK_MAX_STDOUT_FRAME_BYTES

#: The SDK's own default, read from the installed package rather than hardcoded
#: so an SDK bump that changes it makes this test speak up instead of asserting
#: a stale number.
def _sdk_default_buffer_size() -> int:
    from claude_agent_sdk._internal.transport import subprocess_cli

    return subprocess_cli._DEFAULT_MAX_BUFFER_SIZE


#: Spawn sites that must pass the ceiling. Mirrors the list in
#: ``sdk_subprocess_env``'s docstring; a new spawn site belongs in both.
_SPAWN_SITES = (
    "app/services/report_generation.py",
    "app/services/optimization_loop_orchestrator.py",
    "app/services/agent_runner.py",
)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_the_ceiling_clears_the_sdk_default() -> None:
    """A ceiling at or below the default would fix nothing."""
    assert SDK_MAX_STDOUT_FRAME_BYTES > _sdk_default_buffer_size()


def test_the_ceiling_clears_our_own_tool_payload_guard() -> None:
    """The tool guard is one of the two producers of oversized frames.

    A 900,000-byte tool result echoed back as a JSON frame left only ~145 KB of
    headroom under the SDK default before any string escaping.
    """
    assert SDK_MAX_STDOUT_FRAME_BYTES > MAX_MCP_PAYLOAD_BYTES


def test_the_ceiling_is_derived_from_the_tool_guard_not_a_literal() -> None:
    """Drift-proofing: raising the tool guard must raise the ceiling with it.

    Checks the relationship rather than the number — if someone changes the tool
    guard to something the ceiling no longer covers even at worst-case escaping,
    this fails.
    """
    from app.services import sdk_subprocess_env as mod

    worst_case = mod._WORST_CASE_JSON_ESCAPE * MAX_MCP_PAYLOAD_BYTES
    assert SDK_MAX_STDOUT_FRAME_BYTES >= worst_case, (
        "the ceiling no longer covers a max-size tool payload at worst-case "
        f"JSON escaping ({worst_case} bytes)"
    )


@pytest.mark.parametrize("relpath", _SPAWN_SITES)
def test_every_spawn_site_sets_max_buffer_size(relpath: str) -> None:
    """The property that was missing: applied everywhere, not just once.

    Source-level rather than behavioural because two of the three sites build
    options inside a function that spawns a real CLI subprocess. It checks that
    the symbol is wired at the site, which is the thing that was absent.
    """
    source = (_BACKEND_ROOT / relpath).read_text()
    assert "max_buffer_size" in source, (
        f"{relpath} spawns an SDK client without setting max_buffer_size, so it "
        "keeps the 1 MiB default that killed the report narrator"
    )
    assert "SDK_MAX_STDOUT_FRAME_BYTES" in source, (
        f"{relpath} sets max_buffer_size to something other than the shared "
        "SDK_MAX_STDOUT_FRAME_BYTES — one ceiling, one definition"
    )


def test_the_spawn_site_list_still_matches_the_code() -> None:
    """Guards the guard: a fourth spawn site must not slip past this file.

    Finds every module that constructs SDK client options and asserts it is
    accounted for. Without this, adding a spawn site would leave the parametrize
    list quietly incomplete — the same shape as the bug being fixed.
    """
    found: set[str] = set()
    for path in (_BACKEND_ROOT / "app").rglob("*.py"):
        text = path.read_text()
        if "ClaudeAgentOptions(" not in text:
            continue
        tree = ast.parse(text)
        constructs = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ClaudeAgentOptions"
            for node in ast.walk(tree)
        )
        if constructs:
            found.add(path.relative_to(_BACKEND_ROOT).as_posix())

    assert found == set(_SPAWN_SITES), (
        "the set of SDK spawn sites changed.\n"
        f"  in code but not covered here: {sorted(found - set(_SPAWN_SITES))}\n"
        f"  listed here but not in code:  {sorted(set(_SPAWN_SITES) - found)}\n"
        "Add it to _SPAWN_SITES (and to sdk_subprocess_env's docstring)."
    )


def test_the_sdk_still_reads_the_option_we_are_setting() -> None:
    """An SDK bump that renames or drops the option must not go unnoticed.

    Setting an option the SDK ignores looks identical to a working fix, right up
    until the same error reappears in cloud.
    """
    from claude_agent_sdk.types import ClaudeAgentOptions

    assert "max_buffer_size" in inspect.signature(ClaudeAgentOptions).parameters, (
        "claude_agent_sdk no longer accepts max_buffer_size — the ceiling this "
        "backend sets is being silently ignored"
    )
