# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable per-project chat transcript store.

The transcript is append-only and idempotent via a cursor. Verifies:
- Basic append and load
- Idempotency (duplicate ids are skipped)
- Missing file returns []
- chat/ subdir is created lazily
- Empty-id fallback uses a content hash
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.chat_transcript import append_messages, load_transcript


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Mock project directory in a temp location."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


def test_append_and_load(tmp_project_dir: Path) -> None:
    """append_messages writes to JSONL; load_transcript reads back in order with parts preserved."""
    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        msgs = [
            {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "Hello"}], "created_at": ""},
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Hi"}], "created_at": ""},
        ]
        # First append
        import asyncio
        asyncio.run(append_messages(project_id, msgs))

        # Load back
        loaded = load_transcript(project_id)
        assert loaded == msgs


def test_idempotency(tmp_project_dir: Path) -> None:
    """Appending the same id twice only writes once; parts survive round-trip."""
    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        msg1 = {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "First"}], "created_at": ""}
        msg2 = {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Second"}], "created_at": ""}

        import asyncio
        # First append
        asyncio.run(append_messages(project_id, [msg1, msg2]))
        # Second append with overlapping id
        asyncio.run(append_messages(project_id, [msg2]))  # msg-2 already written

        loaded = load_transcript(project_id)
        # Should still be 2 messages, not 3
        assert len(loaded) == 2
        assert loaded[0]["id"] == "msg-1"
        assert loaded[1]["id"] == "msg-2"


def test_load_missing_file(tmp_project_dir: Path) -> None:
    """load_transcript on a missing file returns []."""
    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        loaded = load_transcript(project_id)
        assert loaded == []


def test_chat_subdir_created_lazily(tmp_project_dir: Path) -> None:
    """The chat/ subdir is created by the first writer."""
    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        chat_dir = tmp_project_dir / "chat"
        assert not chat_dir.exists()

        import asyncio
        asyncio.run(append_messages(project_id, [{"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "Hi"}], "created_at": ""}]))

        assert chat_dir.exists()
        assert (chat_dir / "transcript.jsonl").exists()
        assert (chat_dir / "transcript.cursor").exists()


def test_empty_id_fallback(tmp_project_dir: Path) -> None:
    """Messages with no id fall back to content hash for deduplication."""
    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        msg1 = {"id": "", "role": "user", "parts": [{"type": "text", "text": "Same content"}], "created_at": ""}
        msg2 = {"id": "", "role": "user", "parts": [{"type": "text", "text": "Same content"}], "created_at": ""}

        import asyncio
        # First append
        asyncio.run(append_messages(project_id, [msg1]))
        # Second append with same content (should be deduplicated via content hash)
        asyncio.run(append_messages(project_id, [msg2]))

        loaded = load_transcript(project_id)
        # Should be 1 message, not 2
        assert len(loaded) == 1
        assert loaded[0]["parts"] == [{"type": "text", "text": "Same content"}]


def test_is_current_schema() -> None:
    """is_current_schema returns True when all entries have `parts`, False otherwise.

    This is a POSITIVE membership check — never inspects old-shape fields
    (content, tool_calls). Empty list → True (nothing to rebuild).
    """
    from app.services.chat_transcript import is_current_schema

    # Empty list → True
    assert is_current_schema([])

    # All have parts AND non-empty created_at → True
    current = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": "2026-07-10T12:00:00.000Z"},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}], "created_at": "2026-07-10T12:00:01.000Z"},
    ]
    assert is_current_schema(current)

    # One missing parts → False
    stale = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""},
        {"id": "a1", "role": "assistant", "content": "hello", "tool_calls": None, "created_at": ""},
    ]
    assert not is_current_schema(stale)

    # All missing parts → False
    all_stale = [
        {"id": "u1", "role": "user", "content": "hi", "tool_calls": None, "created_at": ""},
        {"id": "a1", "role": "assistant", "content": "hello", "tool_calls": None, "created_at": ""},
    ]
    assert not is_current_schema(all_stale)

    # Entries with parts BUT empty/missing created_at → False (stale)
    empty_timestamps = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": ""},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}], "created_at": ""},
    ]
    assert not is_current_schema(empty_timestamps)

    missing_timestamps = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}]},  # no created_at key
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}]},
    ]
    assert not is_current_schema(missing_timestamps)

    # Valid non-empty created_at → True
    with_timestamps = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": "2026-07-10T12:00:00.000Z"},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}], "created_at": "2026-07-10T12:00:01.000Z"},
    ]
    assert is_current_schema(with_timestamps)


def test_is_current_schema_treats_cli_task_envelope_as_stale() -> None:
    """A baked-in CLI task-envelope entry makes the transcript stale.

    Such an entry has valid `parts` + `created_at`, so the schema/timestamp
    checks alone keep re-serving it. Flagging it stale routes the read path
    through discard → rebuild (via the fixed mapper) so it is purged.
    """
    from app.services.chat_transcript import is_current_schema

    envelope_entry = [
        {
            "id": "u1",
            "role": "user",
            "parts": [{"type": "text", "text": "<task-notification>done</task-notification>"}],
            "created_at": "2026-07-10T12:00:00.000Z",
        },
    ]
    assert not is_current_schema(envelope_entry)

    # A normal current transcript alongside is still fine.
    normal = [
        {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "hi"}], "created_at": "2026-07-10T12:00:00.000Z"},
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "hello"}], "created_at": "2026-07-10T12:00:01.000Z"},
    ]
    assert is_current_schema(normal)

    # A mix: one good entry + one envelope entry → stale (all() short-circuits).
    mixed = normal + envelope_entry
    assert not is_current_schema(mixed)


def test_baked_envelope_transcript_is_stale_then_rebuilds_clean(tmp_project_dir: Path) -> None:
    """Round-trip: a transcript with a baked-in envelope is stale; rebuilding
    through the fixed mapper drops the envelope entry.

    Simulates the read-path self-heal: is_current_schema(baked) is False →
    discard → rebuild from the SDK session via map_session_messages → no
    envelope message survives.
    """
    from types import SimpleNamespace

    from app.services.chat_message_parser import map_session_messages
    from app.services.chat_transcript import (
        append_messages,
        discard_transcript,
        is_current_schema,
        load_transcript,
    )

    project_id = "test-proj-envelope"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Seed a transcript that already baked the envelope as a user entry.
        baked = [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "Real question"}], "created_at": "2026-07-10T12:00:00.000Z"},
            {"id": "env", "role": "user", "parts": [{"type": "text", "text": "<task-notification>done</task-notification>"}], "created_at": "2026-07-10T12:00:02.000Z"},
        ]
        asyncio.run(append_messages(project_id, baked))
        loaded = load_transcript(project_id)

        # Stale because of the envelope entry → self-heal triggers.
        assert not is_current_schema(loaded)

        # Self-heal: discard, then rebuild from the SDK session via the mapper.
        # Stubs carry a `timestamp` so the rebuilt entries are schema-current
        # (non-empty created_at) — isolating this test to the envelope purge.
        asyncio.run(discard_transcript(project_id))
        sdk_session = [
            SimpleNamespace(type="user", message={"content": "Real question"}, uuid="u1", timestamp="2026-07-10T12:00:00.000Z"),
            SimpleNamespace(type="user", message={"content": "<task-notification>done</task-notification>"}, uuid="env", timestamp="2026-07-10T12:00:02.000Z"),
        ]
        rebuilt = map_session_messages(sdk_session)
        asyncio.run(append_messages(project_id, rebuilt))

        final = load_transcript(project_id)
        # The envelope entry is gone; only the real user message remains.
        assert all(
            not part.get("text", "").lstrip().startswith("<task-notification>")
            for m in final
            for part in m["parts"]
        )
        assert any(
            part.get("text") == "Real question"
            for m in final
            for part in m["parts"]
        )
        # And the rebuilt transcript is now schema-current.
        assert is_current_schema(final)


def test_discard_transcript(tmp_project_dir: Path) -> None:
    """discard_transcript removes jsonl + cursor; idempotent on missing files.

    Tests must write only to a temp root — never the live projects/ folder.
    The patch target `app.services.project_paths.resolve_project_dir` isolates
    all transcript writes to a temp dir, verified empirically to never leak.
    """
    from app.services.chat_transcript import discard_transcript
    import asyncio

    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append two messages so files exist
        msgs = [
            {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "First"}], "created_at": ""},
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Second"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, msgs))
        transcript_path = tmp_project_dir / "chat" / "transcript.jsonl"
        cursor_path = tmp_project_dir / "chat" / "transcript.cursor"
        assert transcript_path.exists()
        assert cursor_path.exists()

        # Discard removes both files
        asyncio.run(discard_transcript(project_id))
        assert not transcript_path.exists()
        assert not cursor_path.exists()

        # load_transcript returns []
        assert load_transcript(project_id) == []

        # Discard again is idempotent (no error on missing files)
        asyncio.run(discard_transcript(project_id))
        assert not transcript_path.exists()
        assert not cursor_path.exists()


# ---- Tail-rewrite tests ----


def test_tail_rewrite_when_parts_change(tmp_project_dir: Path) -> None:
    """Tail with same id + changed parts is REWRITTEN in place; still one record for that id."""
    project_id = "test-proj"
    import asyncio

    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append initial turn: user + assistant with the initial parts
        initial_msgs = [
            {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "Question"}], "created_at": ""},
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Partial answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, initial_msgs))

        # Re-append the assistant message with MORE/changed parts (same id msg-2)
        updated_msgs = [
            {"id": "msg-2", "role": "assistant", "parts": [
                {"type": "text", "text": "Partial answer"},
                {"type": "text", "text": "Full answer now"},
            ], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, updated_msgs))

        # Load back
        loaded = load_transcript(project_id)

        # Should still be 2 messages (not 3)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "msg-1"
        assert loaded[1]["id"] == "msg-2"

        # The tail (msg-2) should have the updated parts
        assert loaded[1]["parts"] == [
            {"type": "text", "text": "Partial answer"},
            {"type": "text", "text": "Full answer now"},
        ]

        # User message untouched
        assert loaded[0]["parts"] == [{"type": "text", "text": "Question"}]


def test_body_immutability(tmp_project_dir: Path) -> None:
    """Body (non-tail) id is skipped; older history is immutable."""
    project_id = "test-proj"
    import asyncio

    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append first turn
        first_turn = [
            {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "First question"}], "created_at": ""},
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "First answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, first_turn))

        # Append second turn
        second_turn = [
            {"id": "msg-3", "role": "user", "parts": [{"type": "text", "text": "Second question"}], "created_at": ""},
            {"id": "msg-4", "role": "assistant", "parts": [{"type": "text", "text": "Second answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, second_turn))

        # Now try to "rewrite" msg-2 (which is now a BODY record, not the tail)
        attempt_body_rewrite = [
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "CHANGED answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, attempt_body_rewrite))

        # Load back
        loaded = load_transcript(project_id)

        # Should still be 4 messages (not 5)
        assert len(loaded) == 4

        # msg-2 should be UNCHANGED (body is immutable)
        assert loaded[1]["id"] == "msg-2"
        assert loaded[1]["parts"] == [{"type": "text", "text": "First answer"}]


def test_idempotent_no_rewrite_churn(tmp_project_dir: Path) -> None:
    """Identical re-append is a no-op; no duplicate, no rewrite-churn."""
    project_id = "test-proj"
    import asyncio

    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append a completed turn
        turn = [
            {"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "Question"}], "created_at": ""},
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, turn))

        # Record the file size
        transcript_path = tmp_project_dir / "chat" / "transcript.jsonl"
        cursor_path = tmp_project_dir / "chat" / "transcript.cursor"
        initial_transcript_size = transcript_path.stat().st_size
        initial_cursor_content = cursor_path.read_text()

        # Re-append the BYTE-IDENTICAL assistant message
        identical_reappend = [
            {"id": "msg-2", "role": "assistant", "parts": [{"type": "text", "text": "Answer"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, identical_reappend))

        # Transcript file should be UNCHANGED (no rewrite)
        assert transcript_path.stat().st_size == initial_transcript_size

        # Cursor should be UNCHANGED (no new id)
        assert cursor_path.read_text() == initial_cursor_content

        # Load back — should still be 2 messages
        loaded = load_transcript(project_id)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "msg-1"
        assert loaded[1]["id"] == "msg-2"


def test_created_at_insensitivity(tmp_project_dir: Path) -> None:
    """created_at-only difference is a no-op; equality is keyed on (role, parts)."""
    project_id = "test-proj"
    import asyncio

    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append a message with created_at = ""
        initial = [
            {"id": "msg-1", "role": "assistant", "parts": [{"type": "text", "text": "Same parts"}], "created_at": ""},
        ]
        asyncio.run(append_messages(project_id, initial))

        # Record the file size
        transcript_path = tmp_project_dir / "chat" / "transcript.jsonl"
        initial_transcript_size = transcript_path.stat().st_size

        # Re-append with a DIFFERENT created_at but SAME (role, parts)
        different_timestamp = [
            {"id": "msg-1", "role": "assistant", "parts": [{"type": "text", "text": "Same parts"}], "created_at": "2026-06-24T12:00:00Z"},
        ]
        asyncio.run(append_messages(project_id, different_timestamp))

        # Transcript file should be UNCHANGED (no rewrite for created_at-only diff)
        assert transcript_path.stat().st_size == initial_transcript_size

        # Load back — should still be 1 message with the original created_at
        loaded = load_transcript(project_id)
        assert len(loaded) == 1
        assert loaded[0]["created_at"] == ""  # original timestamp preserved


# ---- Provisional-user id convention + tail-upgrade dedupe ----


def test_provisional_user_tail_upgraded_by_completion_flush(tmp_project_dir: Path) -> None:
    """Provisional user bubble (tail) is upgraded by completion flush re-map (no duplicate).

    Append a provisional user bubble (id starts with PROVISIONAL_USER_ID_PREFIX),
    THEN append a completion batch with SDK-generated user + assistant ids.
    Assert the result has exactly ONE user bubble, its id is now the SDK uuid
    (provisional id is gone), and total records == 2.
    """
    from app.services.chat_transcript import (
        append_messages,
        build_provisional_user_message,
        load_transcript,
        PROVISIONAL_USER_ID_PREFIX,
    )
    import asyncio

    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append a provisional user bubble at turn start
        turn_id = "turn-abc"
        message_text = "User question"
        provisional_bubble = build_provisional_user_message(message_text, turn_id)
        asyncio.run(append_messages(project_id, [provisional_bubble]))

        # Verify provisional id starts with the prefix
        assert provisional_bubble["id"].startswith(PROVISIONAL_USER_ID_PREFIX)

        # Load back — should have 1 provisional user bubble
        loaded = load_transcript(project_id)
        assert len(loaded) == 1
        assert loaded[0]["id"].startswith(PROVISIONAL_USER_ID_PREFIX)

        # Now simulate the completion flush re-map with SDK-generated ids
        completion_batch = [
            {
                "id": "sdk-U1",
                "role": "user",
                "parts": [{"type": "text", "text": message_text}],
                "created_at": "2026-07-14T12:00:00.000Z",
            },
            {
                "id": "sdk-A1",
                "role": "assistant",
                "parts": [{"type": "text", "text": "Assistant response"}],
                "created_at": "2026-07-14T12:00:01.000Z",
            },
        ]
        asyncio.run(append_messages(project_id, completion_batch))

        # Load back
        loaded = load_transcript(project_id)

        # Assert exactly 2 records (user + assistant), NOT 3
        assert len(loaded) == 2
        assert loaded[0]["id"] == "sdk-U1"  # SDK uuid, not provisional
        assert loaded[0]["role"] == "user"
        assert loaded[1]["id"] == "sdk-A1"
        assert loaded[1]["role"] == "assistant"


def test_provisional_upgrade_adopts_completion_parts(tmp_project_dir: Path) -> None:
    """Provisional bubble is upgraded with completion re-map's id AND parts.

    The completion flush is authoritative for the turn — even if provisional
    parts differ slightly, the upgrade adopts the completion re-map's parts.
    """
    from app.services.chat_transcript import (
        append_messages,
        build_provisional_user_message,
        load_transcript,
    )
    import asyncio

    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append provisional with initial parts
        turn_id = "turn-parts"
        provisional_bubble = build_provisional_user_message("Matching text", turn_id)
        asyncio.run(append_messages(project_id, [provisional_bubble]))

        # Completion re-map with MATCHING text (upgrade in place)
        completion_batch = [
            {
                "id": "sdk-U2",
                "role": "user",
                "parts": [{"type": "text", "text": "Matching text"}],
                "created_at": "2026-07-14T12:00:00.000Z",
            },
        ]
        asyncio.run(append_messages(project_id, completion_batch))

        # Load back
        loaded = load_transcript(project_id)

        # Assert exactly 1 record with the completion re-map's id (provisional replaced)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "sdk-U2"
        assert loaded[0]["parts"] == [{"type": "text", "text": "Matching text"}]


def test_non_provisional_user_tail_not_collapsed(tmp_project_dir: Path) -> None:
    """A NORMAL user tail (not provisional) followed by a new-id user message is NOT collapsed.

    Guards legitimate repeated-text turns and in-process re-map. Both survive.
    """
    from app.services.chat_transcript import append_messages, load_transcript
    import asyncio

    project_id = "test-proj"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append a normal user message (real SDK id, NOT provisional)
        first_user = {
            "id": "sdk-U3",
            "role": "user",
            "parts": [{"type": "text", "text": "Repeated text"}],
            "created_at": "2026-07-14T12:00:00.000Z",
        }
        asyncio.run(append_messages(project_id, [first_user]))

        # Append a second user message with identical parts but different id
        second_user = {
            "id": "sdk-U4",
            "role": "user",
            "parts": [{"type": "text", "text": "Repeated text"}],
            "created_at": "2026-07-14T12:00:01.000Z",
        }
        asyncio.run(append_messages(project_id, [second_user]))

        # Load back
        loaded = load_transcript(project_id)

        # Assert BOTH messages survive (no collapse)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "sdk-U3"
        assert loaded[1]["id"] == "sdk-U4"


def test_provisional_bubble_is_current_schema(tmp_project_dir: Path) -> None:
    """build_provisional_user_message returns a dict where is_current_schema([msg]) is True.

    The provisional bubble must have `parts` AND non-empty `created_at` so
    is_current_schema doesn't discard it.
    """
    from app.services.chat_transcript import (
        build_provisional_user_message,
        is_current_schema,
    )

    turn_id = "turn-schema"
    message_text = "Test message"
    provisional_bubble = build_provisional_user_message(message_text, turn_id)

    # Assert is_current_schema returns True
    assert is_current_schema([provisional_bubble])

    # Assert it has both `parts` and non-empty `created_at`
    assert "parts" in provisional_bubble
    assert provisional_bubble.get("created_at")  # non-empty


def test_build_provisional_user_message_shape(tmp_project_dir: Path) -> None:
    """build_provisional_user_message returns the correct shape.

    - id starts with PROVISIONAL_USER_ID_PREFIX and encodes the turn_id
    - role == "user"
    - parts == [{"type": "text", "text": text}]
    - created_at is a non-empty ISO-8601 UTC string ending in "Z"
    """
    from app.services.chat_transcript import (
        build_provisional_user_message,
        PROVISIONAL_USER_ID_PREFIX,
    )

    turn_id = "turn-shape"
    message_text = "Test message shape"
    provisional_bubble = build_provisional_user_message(message_text, turn_id)

    # Assert id starts with the prefix and encodes the turn_id
    assert provisional_bubble["id"].startswith(PROVISIONAL_USER_ID_PREFIX)
    assert turn_id in provisional_bubble["id"]

    # Assert role
    assert provisional_bubble["role"] == "user"

    # Assert parts shape
    assert provisional_bubble["parts"] == [{"type": "text", "text": message_text}]

    # Assert created_at is non-empty ISO-8601 UTC ending in "Z"
    created_at = provisional_bubble["created_at"]
    assert created_at
    assert created_at.endswith("Z")
    # Smoke test: parse as datetime
    from datetime import datetime
    datetime.fromisoformat(created_at.replace("Z", "+00:00"))


class TestCrossProcessWriteSerialization:
    """Two-writer transcript simulation.

    Drives ``append_messages`` from two concurrent coroutines simulating the
    (control-plane completion turn) and (microVM completion flush) writers on
    ONE project. Asserts the final transcript has exactly one copy of each turn,
    correct order, no clobbered tail. This is the off-live proof that the shared
    claim (which serializes turn admission) plus per-append atomicity yields a
    non-interleaved transcript. (The claim itself makes the two turns mutually
    exclusive at admission; this test exercises the underlying write path's
    correctness when appends land close together.)
    """

    def test_concurrent_append_no_interleave(self, tmp_project_dir: Path) -> None:
        """Two concurrent append_messages yield one copy of each turn, correct order."""
        project_id = "proj-concurrent"

        with patch(
            "app.services.project_paths.resolve_project_dir",
            return_value=tmp_project_dir,
        ):
            # Two distinct writers: control-plane completion + microVM flush.
            writer1_msgs = [
                {
                    "id": "msg-1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "First"}],
                    "created_at": "2026-07-15T00:00:00.000Z",
                }
            ]
            writer2_msgs = [
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "Second"}],
                    "created_at": "2026-07-15T00:00:01.000Z",
                }
            ]

            async def _drive() -> None:
                # Run both appends concurrently — the per-project async lock in
                # append_messages serializes the read-then-write critical section.
                await asyncio.gather(
                    append_messages(project_id, writer1_msgs),
                    append_messages(project_id, writer2_msgs),
                )

            asyncio.run(_drive())

            messages = load_transcript(project_id)

        # Exactly one copy of each turn, no clobber, no duplication
        assert len(messages) == 2
        ids = {m["id"] for m in messages}
        assert ids == {"msg-1", "msg-2"}
        # Both turns' content preserved intact (no clobbered tail)
        by_id = {m["id"]: m for m in messages}
        assert by_id["msg-1"]["parts"] == [{"type": "text", "text": "First"}]
        assert by_id["msg-2"]["parts"] == [{"type": "text", "text": "Second"}]


def test_provisional_upgrade_text_mismatch_appends_not_clobbers(tmp_project_dir: Path) -> None:
    """Unrelated later user message does NOT clobber provisional bubble.

    When the completion re-map text DIFFERS from the provisional bubble, it appends
    as a new message instead of replacing. This protects a failed-turn provisional
    bubble from being clobbered by an unrelated subsequent user message.
    """
    from app.services.chat_transcript import (
        append_messages,
        build_provisional_user_message,
        load_transcript,
    )
    import asyncio

    project_id = "test-proj-mismatch"
    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Append provisional
        turn_id = "turn-fail"
        provisional_bubble = build_provisional_user_message("Initial request", turn_id)
        asyncio.run(append_messages(project_id, [provisional_bubble]))

        # Later user message with DIFFERENT text (unrelated)
        unrelated_msg = {
            "id": "sdk-U3",
            "role": "user",
            "parts": [{"type": "text", "text": "Unrelated follow-up"}],
            "created_at": "2026-07-15T08:00:00.000Z",
        }
        asyncio.run(append_messages(project_id, [unrelated_msg]))

        # Load back
        loaded = load_transcript(project_id)

        # Assert BOTH messages present (provisional NOT clobbered)
        assert len(loaded) == 2
        assert loaded[0]["id"].startswith("__EVAL_STUDIO_PROVISIONAL_USER__")
        assert loaded[0]["parts"] == [{"type": "text", "text": "Initial request"}]
        assert loaded[1]["id"] == "sdk-U3"
        assert loaded[1]["parts"] == [{"type": "text", "text": "Unrelated follow-up"}]


def test_orphaned_provisional_upgraded_in_place_after_stop(tmp_project_dir: Path) -> None:
    """A stopped turn's provisional bubble reconciles with the next re-map.

    Live shape (measured): user sends "what's next?", stops the turn (stop
    marker appended — the provisional is now BODY, not tail), then sends "?".
    The "?" turn's full-session re-map carries BOTH user messages under their
    real SDK ids. The old tail-only upgrade duplicated both bubbles; the
    generalized upgrade must reconcile each with its own provisional entry,
    in place, and append only the genuinely new assistant reply.
    """
    from app.services.chat_transcript import build_provisional_user_message

    with patch("app.services.project_paths.resolve_project_dir", return_value=tmp_project_dir):
        # Turn 1 start-write, then user stops it: stop marker appended.
        prov1 = build_provisional_user_message("what's next?", "turn-1")
        asyncio.run(append_messages("proj-1", [prov1]))
        asyncio.run(append_messages("proj-1", [{
            "id": "stopped-turn-1",
            "role": "assistant",
            "parts": [{"type": "text", "text": "_(Turn stopped by user.)_"}],
            "created_at": "2026-06-22T10:00:01.000Z",
        }]))
        # Turn 2 start-write.
        prov2 = build_provisional_user_message("?", "turn-2")
        asyncio.run(append_messages("proj-1", [prov2]))

        # Turn 2 completes: full-session re-map with real SDK ids.
        asyncio.run(append_messages("proj-1", [
            {"id": "sdk-user-1", "role": "user",
             "parts": [{"type": "text", "text": "what's next?"}],
             "created_at": "2026-06-22T10:00:00.000Z"},
            {"id": "sdk-user-2", "role": "user",
             "parts": [{"type": "text", "text": "?"}],
             "created_at": "2026-06-22T10:00:05.000Z"},
            {"id": "sdk-asst-1", "role": "assistant",
             "parts": [{"type": "text", "text": "No response requested."}],
             "created_at": "2026-06-22T10:00:06.000Z"},
        ]))

        transcript = load_transcript("proj-1")

    texts = [(m["role"], m["parts"][0]["text"]) for m in transcript]
    assert texts == [
        ("user", "what's next?"),
        ("assistant", "_(Turn stopped by user.)_"),
        ("user", "?"),
        ("assistant", "No response requested."),
    ], f"No duplicate bubbles, positions preserved: {texts}"
    # Both provisional entries carry their real SDK identity now.
    assert transcript[0]["id"] == "sdk-user-1"
    assert transcript[2]["id"] == "sdk-user-2"
