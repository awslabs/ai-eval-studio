# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK subprocess configuration shared by all Claude Agent SDK spawn sites.

The Claude Agent SDK merges ``ClaudeAgentOptions.env`` over ``os.environ``
when it execs the bundled CLI subprocess. This module provides the canonical
env dict that every SDK spawn site in this backend must apply — the chat
turn runner (``agent_runner.py``), the prompt-optimizer session
(``optimization_loop_orchestrator.py``), and the report-narrator session
(``report_generation.py``). A new spawn site has one obvious thing to call.

It also owns :data:`SDK_MAX_STDOUT_FRAME_BYTES`, which every spawn site must
pass as ``max_buffer_size`` for the same reason: it is transport configuration
that is wrong to get right in only some places.

The pins exist because the bundled CLI ships remote feature gates that can
flip mid-day with zero local changes: identical turns have alternated between
sync and async delegation with no code or config edits. The two env vars below
make the CLI's behavior deterministic.

BREADTH + EXPIRY WARNING: ``DISABLE_GROWTHBOOK`` pins EVERY remote gate in
the CLI to its shipped default, not just the delegation gate — vendor
kill-switches for future CLI bugs are disabled in this subprocess. And when
the claude-agent-sdk pin (``backend/pyproject.toml``) is bumped to a CLI
where backgrounding IS the shipped default, BOTH layers (env pin + PreToolUse
rewrite) go inert at once: this env var stops mattering AND gate-on mode
removes ``run_in_background`` from the Task schema, so the forced-false
PreToolUse rewrite has nothing to rewrite. Re-verify synchronous subagent
dispatch on EVERY SDK pin bump.
"""

from app.services.mcp_response import MAX_MCP_PAYLOAD_BYTES

#: Worst-case growth when arbitrary text is escaped into a JSON string: every
#: byte becoming ``\uXXXX``. Real judge prose measures 1.003x, but the ceiling
#: has to hold for the worst content we could ever carry, not the average.
_WORST_CASE_JSON_ESCAPE = 6

#: ``max_buffer_size`` for every SDK client, bounding ONE NDJSON frame read off
#: the CLI's stdout.
#:
#: The SDK defaults to 1 MiB (``_DEFAULT_MAX_BUFFER_SIZE`` in
#: ``claude_agent_sdk/_internal/transport/subprocess_cli.py``) and raises
#: ``JSON message exceeded maximum buffer size of 1048576 bytes`` from its
#: stdout reader when a single frame is larger, which ends the session — the
#: report narrator's large facts pack is enough to hit it.
#:
#: 1 MiB is simply too close to what this backend legitimately puts on that
#: wire. Two independent producers of large frames:
#:
#: * MCP tool results. Our own guard admits payloads up to
#:   ``MAX_MCP_PAYLOAD_BYTES`` (900,000), and the CLI echoes each result back as
#:   a ``user`` frame — leaving only ~145 KB of headroom under the SDK default
#:   before any escaping.
#: * The prompt we send. The CLI echoes user messages too
#:   (``message_parser.py`` handles an inbound ``case "user"``), so a large
#:   facts pack comes straight back at the reader.
#:
#: Derived from the tool guard rather than written as a literal, so raising that
#: guard cannot silently re-create the same failure. Floored at 8 MiB to cover
#: the prompt-echo producer, which is not bounded by the tool guard at all.
#:
#: Costs nothing when unused: the SDK's framer grows to the size of the frame it
#: actually receives, so this raises a ceiling rather than reserving memory.
SDK_MAX_STDOUT_FRAME_BYTES = max(
    8 * 1024 * 1024,
    _WORST_CASE_JSON_ESCAPE * MAX_MCP_PAYLOAD_BYTES,
)

# Pins applied to every SDK subprocess this backend spawns.
_SDK_SUBPROCESS_ENV_PINS: dict[str, str] = {
    # Feature-targeted kill switch for the CLI's background-by-default
    # delegation mode. When the remote gate is on, the CLI REMOVES
    # run_in_background from the Task schema and backgrounds unconditionally,
    # so the forced-false PreToolUse rewrite cannot opt out. This env var
    # makes the CLI honor run_in_background=False.
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    # Pins all remote feature gates in the bundled CLI to their shipped
    # defaults (not just the backgrounding gate). Makes CLI behavior
    # deterministic and deploy-free. Side effect: vendor kill-switches for
    # future CLI bugs are disabled in this subprocess (see module docstring).
    "DISABLE_GROWTHBOOK": "1",
}


def sdk_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment dict for a Claude Agent SDK subprocess.

    The SDK merges ``ClaudeAgentOptions.env`` over ``os.environ`` when it
    execs the bundled CLI. This helper applies the canonical pins LAST so a
    caller's ``base`` can never override them — the pins are load-bearing for
    correct subagent behavior (synchronous delegation, deterministic gate
    evaluation).

    Args:
        base: Optional caller-specific environment variables to include. The
            pins are applied AFTER these, so any conflicting keys in ``base``
            are overridden.

    Returns:
        A dict suitable for ``ClaudeAgentOptions(env=...)``, with the pins
        merged last.
    """
    result = dict(base) if base else {}
    result.update(_SDK_SUBPROCESS_ENV_PINS)
    return result
