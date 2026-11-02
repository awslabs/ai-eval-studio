# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the platform chat-history reader.

Both modes are exercised:

* **Store mode** — a fake :class:`SessionStore` returns canned entries
  and message lists. We verify the helper filters out UUIDs flagged
  ``isCompactSummary: True`` and forwards everything else.
* **Dev mode (no store)** — falls back to the SDK's on-disk JSONL.
  Patched so the test doesn't depend on the user's local
  ``~/.claude/projects/`` state.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform.chat_history import (
    _compact_summary_uuids_from_entries,
    load_session_messages,
)


def _msg(uuid: str) -> SimpleNamespace:
    """Stand-in for an SDK ``SessionMessage`` — only ``.uuid`` is read."""
    return SimpleNamespace(uuid=uuid)


def test_compact_summary_uuids_picks_flagged_entries() -> None:
    entries = [
        {"uuid": "a", "isCompactSummary": False},
        {"uuid": "b", "isCompactSummary": True},
        {"uuid": "c"},                      # missing flag
        {"isCompactSummary": True},         # missing uuid → ignored
        {"uuid": "d", "isCompactSummary": True},
        "not a dict",                       # ignored
    ]
    assert _compact_summary_uuids_from_entries(entries) == {"b", "d"}


@pytest.mark.asyncio
async def test_store_mode_filters_compact_summaries() -> None:
    """When a SessionStore is provided, summaries from raw entries get filtered."""
    store = MagicMock()
    store.load = AsyncMock(return_value=[
        {"uuid": "summary-1", "isCompactSummary": True},
        {"uuid": "real-1"},
    ])

    with patch(
        "app.platform.chat_history.get_session_messages_from_store",
        new=AsyncMock(return_value=[_msg("summary-1"), _msg("real-1")]),
    ) as fetch:
        messages = await load_session_messages(
            "sess-123",
            cwd="/mnt/projects/users/u/projects/p",
            session_store=store,
        )

    assert [m.uuid for m in messages] == ["real-1"]
    fetch.assert_awaited_once()
    store.load.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_mode_no_summaries_returns_all() -> None:
    """No flagged entries → message list passes through unchanged."""
    store = MagicMock()
    store.load = AsyncMock(return_value=[{"uuid": "real-1"}, {"uuid": "real-2"}])
    with patch(
        "app.platform.chat_history.get_session_messages_from_store",
        new=AsyncMock(return_value=[_msg("real-1"), _msg("real-2")]),
    ):
        messages = await load_session_messages(
            "sess-x",
            cwd="/mnt/projects/users/u/projects/p",
            session_store=store,
        )
    assert [m.uuid for m in messages] == ["real-1", "real-2"]


@pytest.mark.asyncio
async def test_store_mode_propagates_transport_errors() -> None:
    """Transport failures must surface — silently degrading to ``[]``
    would present a wrong key prefix or an AccessDenied as an empty
    history. The history endpoint should 500 on real errors so they're
    observable."""
    store = MagicMock()
    store.load = AsyncMock(side_effect=RuntimeError("S3 unavailable"))
    with patch(
        "app.platform.chat_history.get_session_messages_from_store",
        new=AsyncMock(side_effect=RuntimeError("S3 unavailable")),
    ):
        with pytest.raises(RuntimeError, match="S3 unavailable"):
            await load_session_messages(
                "sess-y",
                cwd="/mnt/projects/users/u/projects/p",
                session_store=store,
            )


@pytest.mark.asyncio
async def test_dev_mode_falls_back_to_sdk_default(tmp_path) -> None:
    """``session_store=None`` routes through the SDK's on-disk default."""
    with patch(
        "app.platform.chat_history.get_session_messages",
        return_value=[_msg("u1"), _msg("u2")],
    ) as sdk_call, patch(
        "app.platform.chat_history._compact_summary_uuids_from_disk",
        return_value={"u2"},
    ):
        messages = await load_session_messages(
            "sess-z",
            cwd="/ignored-in-dev-mode",
            session_store=None,
        )

    assert [m.uuid for m in messages] == ["u1"]
    sdk_call.assert_called_once_with(
        session_id="sess-z", directory="/ignored-in-dev-mode"
    )


# ---- load_raw_session_entries tests ----


@pytest.mark.asyncio
async def test_load_raw_session_entries_store_mode_filters_compact_summaries() -> None:
    """Store mode returns the full entry list with only isCompactSummary entries removed."""
    from app.platform.chat_history import load_raw_session_entries

    store = MagicMock()
    store.load = AsyncMock(return_value=[
        {"uuid": "pre-1", "type": "user"},
        {"uuid": "summary-1", "isCompactSummary": True},
        {"uuid": "post-1", "type": "assistant"},
        {"uuid": "summary-2", "isCompactSummary": True},
        {"uuid": "post-2", "type": "user"},
    ])

    entries = await load_raw_session_entries(
        "sess-123",
        cwd="/mnt/projects/users/u/projects/p",
        session_store=store,
    )

    # Compact summaries filtered, everything else preserved
    assert len(entries) == 3
    assert [e["uuid"] for e in entries] == ["pre-1", "post-1", "post-2"]
    store.load.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_raw_session_entries_dev_mode_reads_jsonl(tmp_path) -> None:
    """Dev mode reads raw JSONL with isCompactSummary filtered, malformed lines skipped."""
    from app.platform.chat_history import load_raw_session_entries
    import json

    # Setup fake SDK on-disk JSONL
    projects_dir = tmp_path / ".claude" / "projects"
    projects_dir.mkdir(parents=True)
    session_dir = projects_dir / "proj-abc"
    session_dir.mkdir()
    jsonl_path = session_dir / "sess-456.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"uuid": "entry-1", "type": "user"}) + "\n")
        f.write("\n")  # blank line
        f.write(json.dumps({"uuid": "summary-1", "isCompactSummary": True}) + "\n")
        f.write("malformed{json\n")  # torn line
        f.write(json.dumps({"uuid": "entry-2", "type": "assistant"}) + "\n")

    with patch("app.platform.chat_history.Path.home", return_value=tmp_path):
        entries = await load_raw_session_entries(
            "sess-456",
            cwd="/ignored",
            session_store=None,
        )

    # Summaries filtered, torn line skipped, rest preserved
    assert len(entries) == 2
    assert [e["uuid"] for e in entries] == ["entry-1", "entry-2"]


@pytest.mark.asyncio
async def test_load_raw_session_entries_missing_session_returns_empty() -> None:
    """Missing session returns [] (not an exception)."""
    from app.platform.chat_history import load_raw_session_entries

    # Store mode — load returns None
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    entries = await load_raw_session_entries(
        "missing-sess",
        cwd="/mnt/projects/users/u/projects/p",
        session_store=store,
    )
    assert entries == []

    # Dev mode — no JSONL match
    with patch("app.platform.chat_history.Path.home") as mock_home:
        mock_home.return_value.exists.return_value = False
        entries = await load_raw_session_entries(
            "missing-sess-2",
            cwd="/ignored",
            session_store=None,
        )
    assert entries == []
