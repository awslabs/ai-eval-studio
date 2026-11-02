# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-history reader bridging the SDK and the platform SessionStore.

The Claude Agent SDK auto-compacts long sessions, leaving "summary" entries
in the transcript that should not appear as user messages in the UI. The
``SessionMessage`` type returned by ``get_session_messages`` doesn't expose
the ``isCompactSummary`` flag, so we read raw store entries (where the
flag is preserved) and filter the rendered messages by their UUIDs.

Two modes:

* **Cloud / explicit store** — the platform's :attr:`Platform.session_store`
  is the source of truth. ``SessionStore.load`` returns the raw JSONL
  entries as dicts; we scan them for ``isCompactSummary == True`` and
  collect the corresponding UUIDs. ``get_session_messages_from_store``
  renders the message list against the same store.
* **Dev / SDK default** — when ``platform.session_store`` is ``None``,
  the SDK's built-in on-disk store at ``~/.claude/projects/`` is in
  effect. We use ``get_session_messages`` for rendering and the on-disk
  JSONL scan in :func:`_compact_summary_uuids_from_disk` for filtering.

Two read paths:

* **Windowed accessor** (:func:`load_session_messages`) — returns only the
  active post-compaction window, suitable for the agent's live context.
* **Raw full-history accessor** (:func:`load_raw_session_entries`) — returns
  the FULL pre-compaction entry list as raw dicts, suitable for backfilling
  the durable transcript. Backfill needs the entire history; the windowed
  accessor returns only survivors after the SDK auto-compacted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    SessionKey,
    SessionStore,
    SessionStoreEntry,
    get_session_messages,
    get_session_messages_from_store,
    project_key_for_directory,
)

logger = logging.getLogger(__name__)


def _compact_summary_uuids_from_entries(
    entries: list[SessionStoreEntry],
) -> set[str]:
    """Pick the UUIDs flagged ``isCompactSummary`` from raw store entries.

    Args:
        entries: As returned by :meth:`SessionStore.load`.

    Returns:
        Set of UUIDs to filter out of the rendered message list.
    """
    out: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("isCompactSummary") is True:
            uuid = entry.get("uuid")
            if isinstance(uuid, str):
                out.add(uuid)
    return out


def _compact_summary_uuids_from_disk(session_id: str) -> set[str]:
    """Dev-mode path: scan the SDK's local JSONL for ``isCompactSummary``.

    Taken only when no platform :class:`SessionStore` is configured, i.e.
    dev mode, where the SDK's built-in on-disk store under
    ``~/.claude/projects/`` holds the transcript. Cloud mode reads the
    platform store instead and never takes this path.

    Returns an empty set when the JSONL can't be located.
    """
    import json

    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return set()

    target = f"{session_id}.jsonl"
    matches = list(projects_dir.glob(f"*/{target}"))
    if not matches:
        return set()
    if len(matches) > 1:
        logger.warning(
            "[chat-history] Multiple JSONL files matched session_id=%s — using first: %s",
            session_id, matches[0],
        )

    out: set[str] = set()
    try:
        with matches[0].open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isCompactSummary") is True:
                    uuid = entry.get("uuid")
                    if isinstance(uuid, str):
                        out.add(uuid)
    except OSError:
        pass
    return out


async def load_session_messages(
    session_id: str,
    *,
    cwd: str,
    session_store: SessionStore | None,
) -> list[Any]:
    """Return the visible :class:`SessionMessage` list for a session.

    Visible = compact-summary entries are filtered out. The function
    routes through :class:`SessionStore` when one is configured (cloud
    mode), and falls back to the SDK's on-disk default otherwise (dev
    mode).

    Args:
        session_id: SDK session id from ``project.session_id``.
        cwd: Project working directory the agent ran in. The SDK
            derives every session-store key from
            ``project_key_for_directory(cwd)``; passing the wrong cwd
            (or letting the SDK fall back to ``os.getcwd()``, which in
            the FastAPI server is ``/app``) lists an empty prefix and
            silently returns ``[]``. Both the explicit ``store.load``
            we use to scan compact-summary uuids and the SDK's
            ``get_session_messages_from_store`` must agree on this value.
            Ignored in dev mode, where the SDK manages its own location.
        session_store: ``platform.session_store``. ``None`` means
            "use the SDK's default on-disk store".

    Returns:
        SessionMessages with compact summaries removed.

    Raises:
        Whatever the underlying SDK / SessionStore raises. Real errors
        — AccessDenied on S3, network blackholes, malformed transcript
        files — are surfaced so the caller can 500. A *missing* session
        is not an exception in either backend: the SDK store returns
        an empty list, and ``SessionStore.load`` returns ``None`` which
        we collapse to an empty list before filtering. Don't add a
        broad ``except Exception`` here — silently degrading to ``[]``
        would hide those real errors behind an empty chat.
    """
    if session_store is not None:
        messages = await get_session_messages_from_store(
            session_store=session_store,
            session_id=session_id,
            directory=cwd,
        )
        key: SessionKey = {
            "project_key": project_key_for_directory(cwd),
            "session_id": session_id,
            "subpath": "",
        }
        entries = await session_store.load(key) or []
        compact_uuids = _compact_summary_uuids_from_entries(entries)
    else:
        # SDK default on-disk store. ``get_session_messages`` is
        # synchronous against that path. Pass ``directory=cwd`` for
        # the same reason as the store path: agent_runner writes the
        # transcript with ``cwd=project_dir``, so reads must derive
        # the same project_key or land on the wrong JSONL file.
        messages = get_session_messages(session_id=session_id, directory=cwd)
        compact_uuids = _compact_summary_uuids_from_disk(session_id)

    if not compact_uuids:
        return list(messages)
    return [m for m in messages if getattr(m, "uuid", None) not in compact_uuids]


async def load_raw_session_entries(
    session_id: str,
    *,
    cwd: str,
    session_store: SessionStore | None,
) -> list[dict]:
    """Return raw session entries for backfill, filtering only ``isCompactSummary``.

    WHY this exists separately from :func:`load_session_messages`: Backfill
    needs the FULL pre-compaction history to seed the durable transcript.
    The windowed accessor returns only the active post-compaction window
    (what the agent sees), not the full lineage. Projects that auto-compacted
    before the durable-transcript feature landed would recover only the tail
    if backfill used the windowed path.

    Returns raw entry dicts (NOT SessionMessage objects) with ONLY the
    ``isCompactSummary`` entries removed — suitable for passing to
    :func:`map_session_messages`, which accepts both dict and object inputs.
    Synthetic messages (those starting with ``ORCHESTRATOR_MSG_PREFIX``) are
    NOT filtered here — the mapper already handles that.

    Args:
        session_id: SDK session id from ``project.session_id``.
        cwd: Project working directory. Key derivation is load-bearing —
            passing the wrong cwd yields an empty result. Must match the
            cwd used by ``agent_runner`` when writing the session.
        session_store: ``platform.session_store``. ``None`` means "use the
            SDK's default on-disk store".

    Returns:
        List of raw entry dicts with compact-summary entries removed.
        Returns ``[]`` if the session doesn't exist (no exception).

    Raises:
        Real errors (S3 AccessDenied, network blackhole) are surfaced.
        A *missing* session is NOT an exception — both backends return ``[]``.
    """
    if session_store is not None:
        # Cloud / store path — load raw entries and filter isCompactSummary
        key: SessionKey = {
            "project_key": project_key_for_directory(cwd),
            "session_id": session_id,
            "subpath": "",
        }
        entries = await session_store.load(key) or []
        # Filter out compact-summary entries
        return [
            e for e in entries
            if isinstance(e, dict) and e.get("isCompactSummary") is not True
        ]
    else:
        # Dev path — read raw JSONL from SDK's on-disk store
        import json

        projects_dir = Path.home() / ".claude" / "projects"
        if not projects_dir.exists():
            return []

        target = f"{session_id}.jsonl"
        matches = list(projects_dir.glob(f"*/{target}"))
        if not matches:
            return []
        if len(matches) > 1:
            logger.warning(
                "[chat-history] Multiple JSONL files matched session_id=%s — using first: %s",
                session_id, matches[0],
            )

        result: list[dict] = []
        try:
            with matches[0].open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip torn/partial lines
                        continue
                    # Filter out compact-summary entries
                    if isinstance(entry, dict) and entry.get("isCompactSummary") is not True:
                        result.append(entry)
        except OSError:
            # File I/O error → treat as missing session
            pass
        return result


__all__ = ["load_session_messages", "load_raw_session_entries"]
