# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MCP response construction helpers."""
from pathlib import Path

from app.services.mcp_response import mcp_error, mcp_json, mcp_success


def test_mcp_success_wraps_text():
    assert mcp_success("hello") == {
        "content": [{"type": "text", "text": "hello"}],
    }


def test_mcp_error_sets_is_error_true():
    result = mcp_error("something failed")
    assert result == {
        "content": [{"type": "text", "text": "something failed"}],
        "is_error": True,
    }


def test_mcp_json_serializes_pretty():
    result = mcp_json({"a": 1, "b": [1, 2]})
    assert result["content"][0]["type"] == "text"
    # Indented JSON output
    assert '"a": 1' in result["content"][0]["text"]
    assert "\n" in result["content"][0]["text"]


def test_mcp_json_handles_non_jsonable_via_default_str():
    result = mcp_json({"path": Path("/tmp/x")})
    assert "/tmp/x" in result["content"][0]["text"]


def test_mcp_json_returns_error_for_oversize_payload():
    """Payloads above the SDK transport cap must convert to an is_error
    response instead of crashing the agent_runner's message reader.
    """
    # Construct a ~2MB string — comfortably above the 900KB guard.
    big = {"blob": "x" * 2_000_000}
    result = mcp_json(big)
    assert result.get("is_error") is True
    text = result["content"][0]["text"]
    assert "exceeds" in text.lower()
    assert "900,000" in text or "900000" in text


def test_mcp_json_allows_small_payload():
    """Payloads under the cap still return a success response."""
    small = {"ok": "a" * 1000}
    result = mcp_json(small)
    assert "is_error" not in result
    assert '"ok"' in result["content"][0]["text"]
