# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for logging Claude Agent SDK message content.

The interactive agent runner (chat) iterates SDK messages and uses these
helpers to extract short, human-readable one-line previews so the backend
console/log files show *what* Claude is doing, not just metadata like
block counts and cost.

Previews built here are for payloads the platform itself produced. User
message bodies, tool arguments and tool results are NOT logged: one shared
log group holds every tenant's turns, so a preview of those is a preview of
somebody's data. Log their size and outcome instead.
"""

from typing import Any


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse whitespace/newlines and truncate to ``limit`` chars."""
    collapsed = " ".join(text.split())
    return collapsed[:limit] if len(collapsed) > limit else collapsed


def scrub_control_chars(text: str) -> str:
    """Replace non-printable characters with a visible ``\\xNN`` escape.

    Subprocess output reaches the log verbatim, and a subprocess is free to
    write terminal control sequences: DeepEval's progress indicators emit ANSI
    escapes, and anything that can write to stderr can emit a carriage return
    or a cursor-movement sequence that rewrites a reader's terminal or forges
    an additional log line. Escaping keeps one produced line to one logged
    line.
    """
    return "".join(
        ch if ch.isprintable() or ch == " " else f"\\x{ord(ch):02x}" for ch in text
    )


def subprocess_log_line(text: str, limit: int = 500) -> str:
    """Return one bounded, control-character-free line of subprocess output.

    Both halves matter: the escape keeps the line honest, and the cap keeps a
    single runaway line from dominating the shared log group.
    """
    return scrub_control_chars(_one_line(text, limit))


def assistant_text_preview(content: list) -> str:
    """Return a ≤200-char preview of concatenated text blocks in an AssistantMessage."""
    parts = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return _one_line(" ".join(parts))


def tool_result_preview(tool_use_result: Any) -> str:
    """Return a ≤200-char preview of a tool_use_result payload.

    Handles string payloads, lists of content blocks (dict or object style
    with .text/.content attributes), and falls back to ``str()`` for
    anything else. Always returns a non-empty placeholder so log lines
    never have a dangling field.
    """
    if isinstance(tool_use_result, str):
        text = tool_use_result
    elif isinstance(tool_use_result, list):
        parts: list[str] = []
        for block in tool_use_result:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        text = " ".join(p for p in parts if p)
    else:
        text = str(tool_use_result)
    return _one_line(text) or "<empty>"
