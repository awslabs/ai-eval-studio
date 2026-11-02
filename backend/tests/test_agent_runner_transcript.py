# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trailing-answer reconciliation in the transcript append path.

Regression cover for the flush-ordering race: the SDK can deliver
``ResultMessage`` before it flushes the turn's final text block to the session
file, so re-reading the session yields an assistant turn that ends on a tool
with no trailing answer. ``_reconcile_trailing_answer`` backfills the answer
from ``ResultMessage.result`` so the LAST turn of a conversation is never
persisted truncated (the tail-rewrite repair only runs on a subsequent turn).
"""
from __future__ import annotations

from app.services.agent_runner import _reconcile_trailing_answer
from app.services.chat_message_parser import ORCHESTRATOR_MSG_PREFIX


def _assistant(parts: list[dict]) -> dict:
    return {"id": "turn-1", "role": "assistant", "parts": parts, "created_at": ""}


def test_backfills_answer_when_turn_ends_on_tool() -> None:
    """Race case: turn ends on a tool with no trailing text → answer appended."""
    mapped = [
        _assistant([
            {"type": "thinking", "text": "let me check", "truncated": False},
            {"type": "text", "text": "Looking into it:"},
            {"type": "tool", "name": "Bash", "input": {}, "result_preview": "ok", "status": "ok", "truncated": False},
        ])
    ]
    _reconcile_trailing_answer(mapped, "Here is the final answer.")
    parts = mapped[-1]["parts"]
    assert parts[-1] == {"type": "text", "text": "Here is the final answer."}
    # The lead-in text and the tool are untouched.
    assert [p["type"] for p in parts] == ["thinking", "text", "tool", "text"]


def test_no_duplicate_when_answer_already_present() -> None:
    """Flushed case: turn already ends with a text part → no append (no duplication)."""
    parts = [
        {"type": "tool", "name": "Bash", "input": {}, "result_preview": "ok", "status": "ok", "truncated": False},
        {"type": "text", "text": "Here is the final answer."},
    ]
    mapped = [_assistant(list(parts))]
    _reconcile_trailing_answer(mapped, "Here is the final answer.")
    assert mapped[-1]["parts"] == parts


def test_noop_when_final_answer_empty_or_none() -> None:
    """A genuinely tool-ending turn with no SDK result text is left untouched."""
    parts = [{"type": "tool", "name": "Bash", "input": {}, "result_preview": "ok", "status": "ok", "truncated": False}]
    for empty in (None, "", "   "):
        mapped = [_assistant(list(parts))]
        _reconcile_trailing_answer(mapped, empty)
        assert mapped[-1]["parts"] == parts


def test_noop_when_tail_is_not_assistant() -> None:
    """Defensive: a user-tailed mapping is never mutated."""
    mapped = [{"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""}]
    _reconcile_trailing_answer(mapped, "answer")
    assert mapped[-1]["parts"] == [{"type": "text", "text": "hi"}]


def test_noop_on_empty_mapping() -> None:
    """Defensive: empty mapping is a no-op (no IndexError)."""
    mapped: list[dict] = []
    _reconcile_trailing_answer(mapped, "answer")
    assert mapped == []


def test_only_reconciles_tail_turn() -> None:
    """Only the just-finished (tail) turn is reconciled; earlier turns untouched."""
    earlier = _assistant([
        {"type": "tool", "name": "Bash", "input": {}, "result_preview": "ok", "status": "ok", "truncated": False},
    ])
    tail = _assistant([
        {"type": "tool", "name": "Grep", "input": {}, "result_preview": "ok", "status": "ok", "truncated": False},
    ])
    mapped = [earlier, tail]
    _reconcile_trailing_answer(mapped, "final answer")
    # Earlier turn keeps its single tool part (body-immutable on append anyway).
    assert [p["type"] for p in mapped[0]["parts"]] == ["tool"]
    # Tail turn gets the backfilled answer.
    assert mapped[1]["parts"][-1] == {"type": "text", "text": "final answer"}


# --- Whole-message flush race: detached backfill lands the unflushed tail ---


class _FakeSessionReads:
    """Session re-reads that return no assistant tail until ``flush_after`` calls.

    Models the WHOLE-MESSAGE variant of the flush-ordering race: the SDK
    delivered ResultMessage but hasn't flushed the final assistant entry to
    the session file yet, so early re-reads map a tail that is the turn's
    USER message. Without the detached backfill the report is silently never
    appended until a later turn (the live "iteration report invisible until
    the next turn / resume" defect).
    """

    def __init__(self, flush_after: int) -> None:
        self.flush_after = flush_after
        self.calls = 0

    def entries(self) -> list[dict]:
        self.calls += 1
        out = [
            {"type": "user", "uuid": "u-1", "message": {"role": "user", "content": "iteration 7 complete"}},
        ]
        if self.calls > self.flush_after:
            out.append({
                "type": "assistant", "uuid": "a-1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "v6 came in at 81.2%"}]},
            })
        return out


def _fake_map(entries):
    """Minimal mapper: one dict per entry with role + a text part."""
    mapped = []
    for e in entries:
        content = e["message"]["content"]
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = [{"type": "text", "text": content[0]["text"]}]
        mapped.append({
            "id": e["uuid"], "role": e["message"]["role"],
            "parts": parts, "created_at": "",
        })
    return mapped


def test_backfill_retries_until_assistant_tail_flushed() -> None:
    """The detached backfill re-reads the session until the assistant lands."""
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner

    fake = _FakeSessionReads(flush_after=2)
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=lambda pid: []), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
        ))

    # Re-read repeated until the assistant entry flushed (2 misses + 1 hit).
    assert fake.calls == 3
    # Exactly ONE append, ending on the assistant report.
    assert len(appended) == 1
    assert appended[0][-1]["role"] == "assistant"
    assert "81.2%" in appended[0][-1]["parts"][-1]["text"]


def test_backfill_gives_up_after_budget_without_appending() -> None:
    """A tail that never flushes exhausts the budget with zero appends."""
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner

    fake = _FakeSessionReads(flush_after=999)  # never flushes
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=lambda pid: []), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
        ))

    assert fake.calls == 10  # full budget consumed
    assert appended == []  # next-turn append remains the fallback


def test_backfill_bails_when_newer_turn_provisional_is_tail() -> None:
    """ORDERING GUARD: a newer turn's provisional user bubble aborts the backfill.

    The next turn writes a provisional user bubble the moment it starts, which
    can land inside the backfill's poll window. Appending turn N's assistant
    message after it would permanently misorder the durable transcript
    (append_messages bodies are immutable). The backfill must bail and let the
    next turn's own full-session append restore assistant order.
    """
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner
    from app.services.chat_transcript import PROVISIONAL_USER_ID_PREFIX

    fake = _FakeSessionReads(flush_after=0)  # assistant tail available immediately
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    # Durable transcript tail: provisional bubble for a DIFFERENT (newer) turn.
    def _fake_transcript(pid):
        return [
            {"id": "u-1", "role": "user", "parts": [], "created_at": ""},
            {"id": f"{PROVISIONAL_USER_ID_PREFIX}turn-NEXT", "role": "user",
             "parts": [{"type": "text", "text": "next question"}], "created_at": ""},
        ]

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=_fake_transcript), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
            turn_id="turn-THIS",
        ))

    assert appended == [], (
        "backfill must NOT append behind a newer turn's provisional bubble"
    )


def test_backfill_proceeds_when_tail_is_own_turn_provisional() -> None:
    """The guard only fires for a DIFFERENT turn's provisional bubble —
    our own turn's provisional user bubble as the tail is the NORMAL state
    (the assistant message is exactly what the backfill is appending)."""
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner
    from app.services.chat_transcript import PROVISIONAL_USER_ID_PREFIX

    fake = _FakeSessionReads(flush_after=0)
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    def _fake_transcript(pid):
        return [
            {"id": f"{PROVISIONAL_USER_ID_PREFIX}turn-THIS", "role": "user",
             "parts": [{"type": "text", "text": "iteration 7 complete"}], "created_at": ""},
        ]

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=_fake_transcript), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
            turn_id="turn-THIS",
        ))

    assert len(appended) == 1
    assert appended[0][-1]["role"] == "assistant"


def test_append_schedules_backfill_only_on_missing_assistant_tail(tmp_path) -> None:
    """_append_turn_to_transcript: inline append stays; backfill is gated.

    - Tail already assistant → NO backfill task (flush won the race).
    - Tail not assistant + final answer → backfill task scheduled.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services import agent_runner

    scheduled: list[str] = []

    def _fake_attach(task, **kwargs):
        scheduled.append(kwargs.get("resource", ""))
        # Cancel immediately — we only assert the scheduling decision.
        task.cancel()

    async def _drive(entries_role: str) -> None:
        async def _fake_load(session_id, *, cwd, session_store):
            return [{
                "type": entries_role, "uuid": "x-1",
                "message": {"role": entries_role, "content": "hello"},
            }]

        async def _fake_append(project_id, msgs):
            return None

        with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
             patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
             patch("app.services.chat_transcript.append_messages", new=_fake_append), \
             patch.object(agent_runner, "attach_error_sink", new=_fake_attach), \
             patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
             patch("app.platform.get_platform", return_value=MagicMock(session_store=None)):
            await agent_runner._append_turn_to_transcript(
                "proj-x", "sess-1", final_answer="answer text",
            )

    asyncio.run(_drive("assistant"))
    assert scheduled == []  # flushed in time — no backfill

    asyncio.run(_drive("user"))
    assert scheduled == ["transcript_backfill"]  # race lost — backfill scheduled


# --- Hidden-prompt whole-message race: per-turn backfill gate + readiness ---


def _build_hidden_race_entries(previous_reply_ending: str = "text") -> list[dict]:
    """Build raw entries for hidden-prompt whole-message race tests.

    Returns: [previous prompt, previous reply (text or tool ending), hidden prompt].
    The current turn's reply is NOT present (whole-message race).

    Args:
        previous_reply_ending: "text" (reply ends with a text part) or
            "tool" (reply ends with a tool part, no trailing text).
    """
    entries = [
        {"type": "user", "message": {"content": "Run the eval"}, "uuid": "p1"},
    ]
    if previous_reply_ending == "text":
        entries.append({
            "type": "assistant",
            "message": {"model": "us.anthropic.claude-opus-4-6", "content": [{"type": "text", "text": "Running eval..."}]},
            "uuid": "a1",
        })
    elif previous_reply_ending == "tool":
        entries.extend([
            {
                "type": "assistant",
                "message": {"model": "us.anthropic.claude-opus-4-6", "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                "uuid": "a1",
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok", "is_error": False}]},
                "uuid": "tr1",
            },
        ])
    # This turn's hidden prompt (completion turn).
    entries.append({
        "type": "user",
        "message": {"content": f"{ORCHESTRATOR_MSG_PREFIX}Done — the baseline eval passed 10/10."},
        "uuid": "p2-hidden",
    })
    return entries


def test_hidden_prompt_race_schedules_backfill(tmp_path) -> None:
    """Hidden-prompt whole-message race: backfill is scheduled, inline append is a no-op.

    Without the per-turn identity check, the eval/optimization completion report
    is never written to the durable transcript — it appears live then vanishes
    after a refresh.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services import agent_runner

    scheduled: list[str] = []
    appended: list[list[dict]] = []

    def _fake_attach(task, **kwargs):
        scheduled.append(kwargs.get("resource", ""))
        task.cancel()

    async def _fake_load(session_id, *, cwd, session_store):
        return _build_hidden_race_entries(previous_reply_ending="text")

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch.object(agent_runner, "attach_error_sink", new=_fake_attach), \
         patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
         patch("app.platform.get_platform", return_value=MagicMock(session_store=None)):
        asyncio.run(agent_runner._append_turn_to_transcript(
            "proj-x", "sess-1", final_answer="Done — the baseline eval passed 10/10 (100%).",
        ))

    # Backfill MUST be scheduled for this shape.
    assert scheduled == ["transcript_backfill"], (
        f"Expected backfill scheduled, got {scheduled}"
    )
    # The inline append carried NO content from the current turn (it mapped
    # only the previous turn, so the report string never appeared).
    assert len(appended) == 1
    payload_text = " ".join(
        part.get("text", "")
        for msg in appended[0]
        for part in msg.get("parts", [])
    )
    assert "100%" not in payload_text, (
        "The inline append must be a no-op for the hidden-prompt race — "
        "the report string should not appear in the appended payload."
    )


def test_hidden_prompt_race_never_glues_answer_onto_previous_reply(tmp_path) -> None:
    """Hidden-prompt race with tool-ending previous reply: reconcile must not run.

    Without the per-turn guard on reconcile, the current turn's answer is
    appended to the previous turn's message, permanently, in the durable
    transcript. The previous turn's message would then carry this turn's answer.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services import agent_runner

    scheduled: list[str] = []
    appended: list[list[dict]] = []

    def _fake_attach(task, **kwargs):
        scheduled.append(kwargs.get("resource", ""))
        task.cancel()

    async def _fake_load(session_id, *, cwd, session_store):
        return _build_hidden_race_entries(previous_reply_ending="tool")

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch.object(agent_runner, "attach_error_sink", new=_fake_attach), \
         patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
         patch("app.platform.get_platform", return_value=MagicMock(session_store=None)):
        asyncio.run(agent_runner._append_turn_to_transcript(
            "proj-x", "sess-1", final_answer="Done — the baseline eval passed 10/10 (100%).",
        ))

    # The appended payload's tail message parts are exactly one tool part — no text part added.
    assert len(appended) == 1
    tail_msg = appended[0][-1]
    tail_parts = tail_msg.get("parts", [])
    assert [p["type"] for p in tail_parts] == ["tool"], (
        "Expected exactly one tool part; the previous turn's message must not "
        "be mutated to carry this turn's answer."
    )
    # The report string must not appear anywhere in the appended payload.
    payload_text = " ".join(
        part.get("text", "")
        for msg in appended[0]
        for part in msg.get("parts", [])
    )
    assert "100%" not in payload_text, (
        "The previous turn's message would carry this turn's answer, permanently."
    )


def test_backfill_waits_for_this_turns_reply_not_a_stale_one() -> None:
    """The backfill readiness check is anchored to this turn's prompt.

    Without the anchor, the backfill finishes on attempt 1 against a stale
    mapping (the previous turn's reply) and appends a payload that does not
    contain the current turn's report.
    """
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner

    call_count = 0
    appended: list[list[dict]] = []

    async def _fake_load_session(session_id, *, cwd, session_store):
        nonlocal call_count
        call_count += 1
        entries = _build_hidden_race_entries(previous_reply_ending="text")
        if call_count >= 3:
            # On the 3rd call, append this turn's reply.
            entries.append({
                "type": "assistant",
                "message": {"model": "us.anthropic.claude-opus-4-6", "content": [
                    {"type": "text", "text": "Done — the baseline eval passed 10/10 (100%)."},
                ]},
                "uuid": "a2",
            })
        return entries

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load_session), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", return_value=[]), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None,
            "Done — the baseline eval passed 10/10 (100%).",
            turn_id="turn-THIS",
            prompt_uuid="p2-hidden",
        ))

    # The session was read 3 times (2 stale reads + 1 ready).
    assert call_count == 3, f"Expected 3 session reads, got {call_count}"
    # Exactly ONE append, and that payload's tail message carries the report text.
    assert len(appended) == 1
    tail_msg = appended[0][-1]
    tail_text = " ".join(part.get("text", "") for part in tail_msg.get("parts", []))
    assert "100%" in tail_text, (
        "The backfill must append the current turn's reply, not a stale one."
    )


def test_partial_flush_reconciles_inline_and_schedules_nothing(tmp_path) -> None:
    """Partial flush: this turn's assistant present with trailing text missing → reconcile inline, no backfill.

    This is the pin: the behavior this preserves. Without the guard, the
    fix would turn the partial-flush repair off and the trailing answer would
    be missing from the durable transcript.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services import agent_runner

    scheduled: list[str] = []
    appended: list[list[dict]] = []

    def _fake_attach(task, **kwargs):
        scheduled.append(kwargs.get("resource", ""))
        task.cancel()

    async def _fake_load(session_id, *, cwd, session_store):
        # Visible prompt, this turn's assistant with a tool step, tool result, no trailing text.
        return [
            {"type": "user", "message": {"content": "Run the eval"}, "uuid": "p1"},
            {
                "type": "assistant",
                "message": {"model": "us.anthropic.claude-opus-4-6", "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                "uuid": "a1",
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok", "is_error": False}]},
                "uuid": "tr1",
            },
        ]

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch.object(agent_runner, "attach_error_sink", new=_fake_attach), \
         patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
         patch("app.platform.get_platform", return_value=MagicMock(session_store=None)):
        asyncio.run(agent_runner._append_turn_to_transcript(
            "proj-x", "sess-1", final_answer="Done — the baseline eval passed 10/10 (100%).",
        ))

    # NO backfill scheduled (the reply is already present, just missing its trailing text).
    assert scheduled == [], f"Expected no backfill, got {scheduled}"
    # The appended tail message parts are [tool, text] with the answer as the text part.
    assert len(appended) == 1
    tail_msg = appended[0][-1]
    tail_parts = tail_msg.get("parts", [])
    assert [p["type"] for p in tail_parts] == ["tool", "text"], (
        "Expected [tool, text] — the answer reconciled inline as the trailing text part."
    )
    tail_text = tail_parts[-1].get("text", "")
    assert "100%" in tail_text


def test_pure_tool_turn_reconciles_nothing_and_schedules_nothing(tmp_path) -> None:
    """Pure-tool turn: empty final answer → no mutation, no backfill.

    This is the pin for empty-result turns. The behavior must be unchanged.
    """
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services import agent_runner

    scheduled: list[str] = []
    appended: list[list[dict]] = []

    def _fake_attach(task, **kwargs):
        scheduled.append(kwargs.get("resource", ""))
        task.cancel()

    async def _fake_load(session_id, *, cwd, session_store):
        # Same shape as partial flush, but with empty final_answer.
        return [
            {"type": "user", "message": {"content": "Run the eval"}, "uuid": "p1"},
            {
                "type": "assistant",
                "message": {"model": "us.anthropic.claude-opus-4-6", "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "ls"}},
                ]},
                "uuid": "a1",
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "ok", "is_error": False}]},
                "uuid": "tr1",
            },
        ]

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch.object(agent_runner, "attach_error_sink", new=_fake_attach), \
         patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
         patch("app.platform.get_platform", return_value=MagicMock(session_store=None)):
        asyncio.run(agent_runner._append_turn_to_transcript(
            "proj-x", "sess-1", final_answer="",  # empty
        ))

    # NO backfill (empty answer → gate does not fire).
    assert scheduled == []
    # The appended tail message parts are exactly one tool part.
    assert len(appended) == 1
    tail_msg = appended[0][-1]
    tail_parts = tail_msg.get("parts", [])
    assert [p["type"] for p in tail_parts] == ["tool"]


def test_backfill_bails_when_a_newer_turn_COMPLETED_inside_the_poll_window() -> None:
    """The case the provisional-shape guard cannot see.

    The shape check looks for a DIFFERENT turn's provisional bubble as the tail.
    A newer turn that starts AND FINISHES inside the 5s poll window defeats it:
    its provisional bubble is upgraded to a real user message and its reply
    appended, so the tail carries no provisional prefix at all.

    That matters because ``session_has_reply_after_prompt`` is then satisfied by
    the NEWER turn's reply — it asks "is there a real assistant entry after our
    prompt", which a newer turn's reply also answers. With our own reply never
    flushing and that reply ending on a tool call with no trailing text,
    ``_reconcile_trailing_answer`` would glue OUR answer onto the newer turn's
    message, permanently (transcript bodies are immutable).

    Detected by comparing the tail id against the one snapshotted at scheduling
    time rather than by looking for one shape.
    """
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner

    fake = _FakeSessionReads(flush_after=0)  # a reply is visible immediately
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    # First read: our own turn's append is the tail (the normal snapshot state).
    # Subsequent reads: a newer turn has fully COMPLETED — real user message plus
    # its own assistant reply. No provisional prefix anywhere.
    reads = {"n": 0}

    def _fake_transcript(pid):
        reads["n"] += 1
        base = [{"id": "u-1", "role": "user", "parts": [], "created_at": ""},
                {"id": "a-1", "role": "assistant", "parts": [], "created_at": ""}]
        if reads["n"] == 1:
            return base
        return base + [
            {"id": "u-2", "role": "user", "parts": [], "created_at": ""},
            {"id": "a-2", "role": "assistant", "parts": [], "created_at": ""},
        ]

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=_fake_transcript), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
            turn_id="turn-THIS",
        ))

    assert appended == [], (
        "backfill appended after a newer turn had already completed — our "
        "answer would be glued onto that turn's message, permanently"
    )


def test_backfill_proceeds_when_the_tail_is_unchanged() -> None:
    """The guard must not bail on a quiet transcript.

    Pins the other direction of the id comparison: with no other writer, the
    snapshot and every subsequent read agree, and the backfill does its job.
    Without this, a guard that always bailed would look correct to the test
    above while silently disabling the whole mechanism.
    """
    import asyncio
    from unittest.mock import patch

    from app.services import agent_runner

    fake = _FakeSessionReads(flush_after=0)
    appended: list[list[dict]] = []

    async def _fake_load(session_id, *, cwd, session_store):
        return fake.entries()

    async def _fake_append(project_id, msgs):
        appended.append(msgs)

    async def _no_sleep(_):
        return None

    def _fake_transcript(pid):
        return [{"id": "u-1", "role": "user", "parts": [], "created_at": ""}]

    with patch("app.platform.chat_history.load_raw_session_entries", new=_fake_load), \
         patch("app.services.chat_message_parser.map_session_messages", new=_fake_map), \
         patch("app.services.chat_transcript.append_messages", new=_fake_append), \
         patch("app.services.chat_transcript.load_transcript", new=_fake_transcript), \
         patch.object(agent_runner.asyncio, "sleep", new=_no_sleep):
        asyncio.run(agent_runner._backfill_transcript_when_flushed(
            "proj-x", "sess-1", "/tmp/proj-x", None, "v6 came in at 81.2%",
            turn_id="turn-THIS",
        ))

    assert appended, "the backfill must still run when nothing else wrote"
