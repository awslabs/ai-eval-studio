# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared SDK session message → ChatMessage mapper.

Verifies the unified mapping logic shared by both the read path
(``get_chat_history``) and the durable-transcript write path
(``agent_runner``).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.services.chat_message_parser import (
    ORCHESTRATOR_MSG_PREFIX,
    find_last_prompt_uuid,
    is_cli_task_envelope,
    is_synthetic_message,
    map_session_messages,
    session_has_reply_after_prompt,
)


def _msg(
    role: str,
    content: str | list[dict[str, Any]],
    uuid: str = "test-uuid",
) -> SimpleNamespace:
    """Build a minimal SDK SessionMessage stub."""
    return SimpleNamespace(
        type=role,
        message={"content": content},
        uuid=uuid,
    )


def test_map_session_messages_coalesces_consecutive_assistant_records() -> None:
    """Multiple assistant records with text blocks join into one ChatMessage with ordered parts."""
    raw = [
        _msg("assistant", [{"type": "text", "text": "First part"}], "uuid-1"),
        _msg("assistant", [{"type": "text", "text": "Second part"}], "uuid-2"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["parts"] == [
        {"type": "text", "text": "First part"},
        {"type": "text", "text": "Second part"},
    ]
    assert result[0]["id"] == "uuid-1"  # anchored to the first record


def test_map_session_messages_tool_calls_concatenate_in_order() -> None:
    """Multiple tool_use blocks from separate records become ordered tool parts."""
    raw = [
        _msg("assistant", [{"type": "tool_use", "name": "tool1", "input": {"x": 1}, "id": "t1"}], "uuid-1"),
        _msg("assistant", [{"type": "tool_use", "name": "tool2", "input": {"y": 2}, "id": "t2"}], "uuid-2"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    assert result[0]["parts"] == [
        {"type": "tool", "name": "tool1", "input": {"x": 1}, "result_preview": None, "status": "running", "truncated": False},
        {"type": "tool", "name": "tool2", "input": {"y": 2}, "result_preview": None, "status": "running", "truncated": False},
    ]


def test_map_session_messages_plain_user_message_flushes_assistant_turn() -> None:
    """A plain string user message is a turn boundary — flushes the assistant."""
    raw = [
        _msg("assistant", [{"type": "text", "text": "Assistant reply"}], "uuid-a"),
        _msg("user", "User message", "uuid-u"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "user"
    assert result[1]["parts"] == [{"type": "text", "text": "User message"}]


def test_map_session_messages_synthetic_user_message_dropped() -> None:
    """User messages with ORCHESTRATOR_MSG_PREFIX are dropped (synthetic)."""
    raw = [
        _msg("assistant", [{"type": "text", "text": "First"}], "uuid-1"),
        _msg("user", f"{ORCHESTRATOR_MSG_PREFIX}synthetic", "uuid-synth"),
        _msg("assistant", [{"type": "text", "text": "Second"}], "uuid-2"),
    ]
    result = map_session_messages(raw)
    # Synthetic message is dropped; two separate assistant turns because it flushed
    assert len(result) == 2
    assert result[0]["parts"] == [{"type": "text", "text": "First"}]
    assert result[1]["parts"] == [{"type": "text", "text": "Second"}]


def test_map_session_messages_cli_task_envelope_user_message_dropped() -> None:
    """A type:"user" entry whose content is a <task-notification> is dropped.

    The CLI sometimes materializes a dequeued background-task notification as a
    real user entry; it must never render as a chat bubble.
    """
    envelope = (
        "<task-notification>\n<task-id>abc</task-id>\n"
        "<status>completed</status>\n</task-notification>"
    )
    raw = [
        _msg("assistant", [{"type": "text", "text": "First"}], "uuid-1"),
        _msg("user", envelope, "uuid-env"),
        _msg("assistant", [{"type": "text", "text": "Second"}], "uuid-2"),
    ]
    result = map_session_messages(raw)
    # Envelope dropped; two separate assistant turns because it flushed between them.
    assert len(result) == 2
    assert all(m["role"] == "assistant" for m in result)
    assert result[0]["parts"] == [{"type": "text", "text": "First"}]
    assert result[1]["parts"] == [{"type": "text", "text": "Second"}]
    # No user bubble carrying the envelope text survived.
    assert not any(
        part.get("text", "").lstrip().startswith("<task-notification>")
        for m in result
        for part in m["parts"]
    )


def test_map_session_messages_queue_operation_entry_produces_none() -> None:
    """A type:"queue-operation" entry is skipped by the role-type contract."""
    raw = [
        _msg("queue-operation", "<task-notification>x</task-notification>", "uuid-q"),
    ]
    result = map_session_messages(raw)
    assert result == []


def test_is_cli_task_envelope_detects_wrapper_tags() -> None:
    """is_cli_task_envelope matches the SDK task-wrapper tags, ignoring leading ws."""
    assert is_cli_task_envelope("<task-notification>hi</task-notification>") is True
    assert is_cli_task_envelope("   \n<task-progress>50%</task-progress>") is True
    assert is_cli_task_envelope("<task-started>go</task-started>") is True
    # Genuine conversation is never matched.
    assert is_cli_task_envelope("Please run the task-notification script") is False
    assert is_cli_task_envelope("normal message") is False
    assert is_cli_task_envelope("") is False


def test_map_session_messages_user_list_content_attaches_tool_result() -> None:
    """User messages with list content (SDK tool_result) match to tool parts by tool_use_id."""
    raw = [
        _msg("assistant", [{"type": "tool_use", "name": "search", "input": {"q": "test"}, "id": "tool-1"}], "uuid-1"),
        _msg("user", [{"type": "tool_result", "tool_use_id": "tool-1", "content": "result text"}], "uuid-tool"),
        _msg("assistant", [{"type": "text", "text": "Got it"}], "uuid-2"),
    ]
    result = map_session_messages(raw)
    # One coalesced assistant turn with tool part + result_preview attached
    assert len(result) == 1
    assert result[0]["parts"] == [
        {
            "type": "tool",
            "name": "search",
            "input": {"q": "test"},
            "result_preview": "result text",
            "status": "ok",
            "truncated": False,
        },
        {"type": "text", "text": "Got it"},
    ]


def test_map_session_messages_subagent_rewrite() -> None:
    """Task/Agent tool_use with subagent_type input rewrites to subagent:<type>."""
    raw = [
        _msg("assistant", [
            {"type": "tool_use", "name": "Task", "input": {"subagent_type": "data-analyzer"}, "id": "t1"}
        ], "uuid-1"),
    ]
    result = map_session_messages(raw)
    assert result[0]["parts"] == [
        {
            "type": "tool",
            "name": "subagent:data-analyzer",
            "input": {"subagent_type": "data-analyzer"},
            "result_preview": None,
            "status": "running",
            "truncated": False,
        },
    ]


def test_is_synthetic_message_detects_prefix() -> None:
    """is_synthetic_message returns True only for ORCHESTRATOR_MSG_PREFIX."""
    assert is_synthetic_message(f"{ORCHESTRATOR_MSG_PREFIX}synthetic") is True
    assert is_synthetic_message("normal message") is False
    assert is_synthetic_message("") is False


# ---- Raw dict entry path tests ----


def test_map_session_messages_accepts_raw_dict_entries() -> None:
    """Dict entries map identically to equivalent SessionMessage objects.

    Proves the dict path and object path go through ONE mapper.
    """
    # Dict entries — user dict + assistant dict with list-of-blocks message
    dict_entries = [
        {"type": "user", "message": {"content": "hi"}, "uuid": "u1"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "name": "search", "input": {"q": "docs"}, "id": "s1"},
                ]
            },
            "uuid": "a1",
        },
    ]
    # Equivalent object entries
    obj_entries = [
        _msg("user", "hi", "u1"),
        _msg("assistant", [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "search", "input": {"q": "docs"}, "id": "s1"},
        ], "a1"),
    ]

    dict_result = map_session_messages(dict_entries)
    obj_result = map_session_messages(obj_entries)

    # Same output: one user bubble + one coalesced assistant turn with parts
    assert len(dict_result) == 2
    assert dict_result == obj_result
    assert dict_result[0] == {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""}
    assert dict_result[1]["role"] == "assistant"
    assert dict_result[1]["parts"] == [
        {"type": "text", "text": "Hello"},
        {"type": "tool", "name": "search", "input": {"q": "docs"}, "result_preview": None, "status": "running", "truncated": False},
    ]


def test_map_session_messages_filters_synthetic_from_dict_entries() -> None:
    """Synthetic filter works on the dict path.

    Raw dict user entry with ORCHESTRATOR_MSG_PREFIX is dropped.
    """
    raw = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "First"}]}, "uuid": "a1"},
        {"type": "user", "message": {"content": f"{ORCHESTRATOR_MSG_PREFIX}synthetic"}, "uuid": "synth"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Second"}]}, "uuid": "a2"},
    ]
    result = map_session_messages(raw)

    # Synthetic message dropped; two separate assistant turns
    assert len(result) == 2
    assert result[0]["parts"] == [{"type": "text", "text": "First"}]
    assert result[1]["parts"] == [{"type": "text", "text": "Second"}]


# ---- Parts model tests ----


def test_map_session_messages_ordered_parts_text_thinking_tool() -> None:
    """Turn with text + thinking + tool maps to ORDERED parts in source order."""
    raw = [
        _msg("assistant", [
            {"type": "text", "text": "Let me think"},
            {"type": "thinking", "thinking": "My reasoning here"},
            {"type": "tool_use", "name": "search", "input": {"q": "test"}, "id": "t1"},
        ], "uuid-1"),
        _msg("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "found it"}], "uuid-tool"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    assert result[0]["parts"] == [
        {"type": "text", "text": "Let me think"},
        {"type": "thinking", "text": "My reasoning here", "truncated": False},
        {"type": "tool", "name": "search", "input": {"q": "test"}, "result_preview": "found it", "status": "ok", "truncated": False},
    ]


def test_map_session_messages_thinking_truncates() -> None:
    """Over-budget thinking block yields truncated:true and bounded text."""
    long_thinking = "x" * 5000  # over _THINKING_PREVIEW_MAX_CHARS (4000)
    raw = [
        _msg("assistant", [{"type": "thinking", "thinking": long_thinking}], "uuid-1"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    thinking_part = result[0]["parts"][0]
    assert thinking_part["type"] == "thinking"
    assert thinking_part["truncated"] is True
    assert len(thinking_part["text"]) == 4000
    assert thinking_part["text"] == "x" * 4000


def test_map_session_messages_tool_result_truncates() -> None:
    """Over-budget tool_result yields truncated:true and bounded result_preview."""
    long_result = "y" * 3000  # over _TOOL_RESULT_PREVIEW_MAX_CHARS (2000)
    raw = [
        _msg("assistant", [{"type": "tool_use", "name": "fetch", "input": {}, "id": "t1"}], "uuid-1"),
        _msg("user", [{"type": "tool_result", "tool_use_id": "t1", "content": long_result}], "uuid-tool"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    tool_part = result[0]["parts"][0]
    assert tool_part["type"] == "tool"
    assert tool_part["truncated"] is True
    assert len(tool_part["result_preview"]) == 2000


def test_map_session_messages_tool_result_is_error_flag() -> None:
    """tool_result with is_error:true sets status to 'error'."""
    raw = [
        _msg("assistant", [{"type": "tool_use", "name": "fetch", "input": {}, "id": "t1"}], "uuid-1"),
        _msg("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "failed", "is_error": True}], "uuid-tool"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    tool_part = result[0]["parts"][0]
    assert tool_part["status"] == "error"


def test_map_session_messages_unmatched_tool_result_ignored() -> None:
    """tool_result with no matching tool_use_id is ignored (defensive)."""
    raw = [
        _msg("assistant", [{"type": "tool_use", "name": "search", "input": {}, "id": "t1"}], "uuid-1"),
        _msg("user", [{"type": "tool_result", "tool_use_id": "unmatched", "content": "orphan"}], "uuid-tool"),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    # Only the tool part; no crash, no stray part
    tool_part = result[0]["parts"][0]
    assert tool_part["type"] == "tool"
    assert tool_part["result_preview"] is None  # never matched


def test_map_session_messages_tool_use_result_text_ending() -> None:
    """A session ending tool_use -> tool_result -> assistant text maps to ONE assistant message.

    Guards the exact shape the windowed reader mangled (it dropped the trailing text block).
    The full-lineage raw reader preserves this ordering, so this test anchors the expected
    behavior for the per-turn append path.
    """
    # Build raw dict entries (mirror test_map_session_messages_accepts_raw_dict_entries pattern)
    raw = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "search", "input": {"q": "docs"}, "id": "search-1"}
                ]
            },
            "uuid": "a1",
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "search-1", "content": "Found 3 results"}
                ]
            },
            "uuid": "u-result",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Based on the search, here's your answer."}
                ]
            },
            "uuid": "a2",
        },
    ]

    result = map_session_messages(raw)

    # Should coalesce into ONE assistant message
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["id"] == "a1"  # anchored to first assistant record

    # Verify parts are in order: tool (with matched result) + text
    parts = result[0]["parts"]
    assert len(parts) == 2

    # First part: tool with matched result
    assert parts[0]["type"] == "tool"
    assert parts[0]["name"] == "search"
    assert parts[0]["input"] == {"q": "docs"}
    assert parts[0]["result_preview"] == "Found 3 results"
    assert parts[0]["status"] == "ok"
    assert parts[0]["truncated"] is False

    # Second part: trailing text (the actual answer)
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "Based on the search, here's your answer."


# ---- created_at population tests ----


def test_map_session_messages_populates_created_at_from_timestamp() -> None:
    """User + assistant messages populate created_at from raw entry timestamp field."""
    raw = [
        {
            "type": "user",
            "message": {"content": "Hello"},
            "uuid": "u1",
            "timestamp": "2026-07-10T12:34:56.789Z",
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hi there"}]},
            "uuid": "a1",
            "timestamp": "2026-07-10T12:35:01.234Z",
        },
    ]
    result = map_session_messages(raw)

    # User message should carry its timestamp
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["created_at"] == "2026-07-10T12:34:56.789Z"

    # Assistant message should carry its timestamp (the first entry's timestamp for coalesced turns)
    assert result[1]["role"] == "assistant"
    assert result[1]["created_at"] == "2026-07-10T12:35:01.234Z"


def test_map_session_messages_assistant_turn_uses_first_entry_timestamp() -> None:
    """Coalesced assistant turn uses the timestamp from the FIRST assistant entry."""
    raw = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "First"}]},
            "uuid": "a1",
            "timestamp": "2026-07-10T10:00:00.000Z",
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Second"}]},
            "uuid": "a2",
            "timestamp": "2026-07-10T10:00:05.000Z",  # 5 seconds later
        },
    ]
    result = map_session_messages(raw)

    # Single coalesced assistant message uses the FIRST entry's timestamp
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["created_at"] == "2026-07-10T10:00:00.000Z"  # First, not second


def test_map_session_messages_created_at_fallback_when_missing_timestamp() -> None:
    """Messages without timestamp field fall back to empty string (SessionMessage object path)."""
    raw = [
        _msg("user", "hi", "u1"),  # SessionMessage object — no timestamp attr
        _msg("assistant", [{"type": "text", "text": "hello"}], "a1"),
    ]
    result = map_session_messages(raw)

    # Both messages fall back to "" when timestamp is absent
    assert len(result) == 2
    assert result[0]["created_at"] == ""
    assert result[1]["created_at"] == ""


def test_map_session_messages_synthetic_assistant_entry_dropped() -> None:
    """CLI-synthesized assistant bookkeeping never renders.

    The CLI writes placeholder assistant records (model="<synthetic>", e.g.
    "No response requested." after an aborted turn) into the session. The
    discriminator is the STRUCTURAL model field — never the text — so any
    wording in any language is dropped, and a real reply that happens to say
    the same words survives (it carries a real model id).
    """
    raw = [
        SimpleNamespace(
            type="assistant",
            message={
                "model": "<synthetic>",
                "content": [{"type": "text", "text": "No response requested."}],
            },
            uuid="uuid-synth",
        ),
        SimpleNamespace(
            type="assistant",
            message={
                "model": "us.anthropic.claude-opus-4-6",
                "content": [{"type": "text", "text": "Real reply"}],
            },
            uuid="uuid-real",
        ),
    ]
    result = map_session_messages(raw)
    assert len(result) == 1
    assert result[0]["parts"] == [{"type": "text", "text": "Real reply"}]


# ---- Session introspection helpers (prompt boundary, reply-after predicate) ----


def test_last_prompt_uuid_returns_newest_string_content_user_entry() -> None:
    """The newest user entry with string content is the turn boundary.

    If the boundary is wrong, the append path will mistake a tool-result entry
    for a prompt and schedule a backfill on every turn with tool use, burning
    its poll budget for no reason.
    """
    entries = [
        {"type": "user", "message": {"content": "First prompt"}, "uuid": "p1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reply"}]}, "uuid": "a1"},
        {"type": "user", "message": {"content": "Second prompt"}, "uuid": "p2"},
    ]
    assert find_last_prompt_uuid(entries) == "p2"


def test_last_prompt_uuid_skips_tool_result_user_entries() -> None:
    """A list-content tool_result user entry after the prompt must not become the boundary.

    If tool-result entries are treated as prompts, the append path will glue
    the current turn's answer onto the previous turn's message (the tool result
    looks like it separates turns, but it keeps the same assistant turn open).
    """
    entries = [
        {"type": "user", "message": {"content": "Prompt"}, "uuid": "p1"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "tu-1"}]}, "uuid": "a1"},
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok"}]},
            "uuid": "tr1",
        },
    ]
    # The prompt is p1, not the tool-result entry tr1.
    assert find_last_prompt_uuid(entries) == "p1"


def test_last_prompt_uuid_treats_hidden_orchestrator_prompt_as_a_prompt() -> None:
    """A hidden orchestrator prompt is a prompt for boundary purposes.

    If hidden prompts are skipped, completion turns (eval/optimization reports)
    will never schedule their backfill and the report will never land in the
    durable transcript.
    """
    entries = [
        {"type": "user", "message": {"content": "First prompt"}, "uuid": "p1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reply"}]}, "uuid": "a1"},
        {"type": "user", "message": {"content": f"{ORCHESTRATOR_MSG_PREFIX}Done"}, "uuid": "p2-hidden"},
    ]
    assert find_last_prompt_uuid(entries) == "p2-hidden"


def test_last_prompt_uuid_none_when_no_prompt_present() -> None:
    """When no prompt exists in the session, return None (unanchored).

    The append path will then use the permissive fallback in the reply
    predicate, so the session behaves as it does today.
    """
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "orphan"}]}, "uuid": "a1"},
    ]
    assert find_last_prompt_uuid(entries) is None


def test_reply_after_prompt_distinguishes_before_from_after() -> None:
    """The predicate separates a reply before the prompt from one after it.

    If the boundary check is wrong, the append path will reconcile against a
    previous turn's message (appending this turn's answer to it) or schedule
    a backfill when the reply is already present.
    """
    # Session: prompt, then assistant reply.
    entries_after = [
        {"type": "user", "message": {"content": "Prompt"}, "uuid": "p1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reply"}]}, "uuid": "a1"},
    ]
    assert session_has_reply_after_prompt(entries_after, "p1") is True

    # Session: assistant reply, then prompt (the reply is BEFORE the boundary).
    entries_before = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "old reply"}]}, "uuid": "a1"},
        {"type": "user", "message": {"content": "New prompt"}, "uuid": "p2"},
    ]
    assert session_has_reply_after_prompt(entries_before, "p2") is False


def test_reply_after_prompt_ignores_synthetic_bookkeeping_assistant() -> None:
    """The only post-prompt assistant entry carries model:'<synthetic>' → no reply.

    If synthetic entries count as replies, an aborted turn's placeholder record
    will convince the append path that the reply has flushed, and the backfill
    will never be scheduled — the real reply is permanently missing from the
    durable transcript.
    """
    entries = [
        {"type": "user", "message": {"content": "Prompt"}, "uuid": "p1"},
        {
            "type": "assistant",
            "message": {"model": "<synthetic>", "content": [{"type": "text", "text": "No response requested."}]},
            "uuid": "a1",
        },
    ]
    assert session_has_reply_after_prompt(entries, "p1") is False


def test_reply_after_prompt_falls_back_to_newest_prompt_for_unknown_uuid() -> None:
    """Anchor uuid absent from the entries → falls back to newest prompt.

    If the fallback is wrong, a compacted/rewritten session will report
    permanent false and burn the backfill's poll budget, leaving the reply
    missing until a later turn backfills it.
    """
    entries = [
        {"type": "user", "message": {"content": "Prompt"}, "uuid": "p1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reply"}]}, "uuid": "a1"},
    ]
    # Anchor uuid "p-unknown" is not in the entries, so fallback to p1.
    # The reply a1 is after p1, so the predicate reports true.
    assert session_has_reply_after_prompt(entries, "p-unknown") is True
