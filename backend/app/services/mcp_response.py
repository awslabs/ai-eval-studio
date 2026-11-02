# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool response construction helpers.

The Claude Agent SDK expects MCP tool handlers to return a dict with a
``content`` list (each item ``{"type": "text", "text": ...}``) and an
optional ``is_error`` flag. This module centralizes the three shapes so
the MCP tool modules under ``services/mcp_tools/`` can stop repeating the
literal.
"""
from __future__ import annotations

import json
from typing import Any

# Leave headroom below the SDK's 1,048,576 byte per-message cap. A payload
# larger than this cap crashes the agent_runner's message reader with
# "Failed to decode JSON: JSON message exceeded maximum buffer size" — a
# fatal, turn-ending error the agent can't observe or retry. Converting
# oversize responses into a normal ``is_error`` keeps the agent alive and
# lets it pick a different tool (e.g. a summary variant).
MAX_MCP_PAYLOAD_BYTES = 900_000


def mcp_success(text: str) -> dict[str, Any]:
    """Return a successful MCP tool response with a single text content item.

    Args:
        text: The text content to include in the response.

    Returns:
        Dict with ``content`` list containing a single text item.
    """
    return {"content": [{"type": "text", "text": text}]}


def mcp_error(message: str) -> dict[str, Any]:
    """Return an error MCP tool response.

    Sets ``is_error: True`` so Claude observes the failure and can react
    instead of silently succeeding.

    Args:
        message: The error message to include in the response.

    Returns:
        Dict with ``content`` list and ``is_error: True``.
    """
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def mcp_json(data: Any) -> dict[str, Any]:
    """Return a successful MCP response containing pretty-printed JSON.

    Uses ``default=str`` so non-JSON-native values (Path, datetime, UUID,
    Decimal) serialize cleanly without the caller having to coerce them.

    Guards against payloads that would crash the SDK transport: anything
    above ``MAX_MCP_PAYLOAD_BYTES`` is converted into an ``is_error``
    response so the agent can react gracefully instead of the whole turn
    dying with a JSON-buffer overflow.

    Args:
        data: Any JSON-serializable value (or values with a str fallback).

    Returns:
        Dict with ``content`` list containing indented JSON text, or an
        error response when the serialized size exceeds the transport cap.
    """
    serialized = json.dumps(data, indent=2, default=str)
    if len(serialized) > MAX_MCP_PAYLOAD_BYTES:
        return mcp_error(
            f"Tool response exceeds {MAX_MCP_PAYLOAD_BYTES:,} byte limit "
            f"(actual: {len(serialized):,} bytes). "
            "Use a summary tool, pass a filter/limit, or request a subset."
        )
    return mcp_success(serialized)
