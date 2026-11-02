# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, append-only, per-project chat transcript.

This is the user-visible source of truth that survives SDK session auto-compaction.
The SDK compacts older turns out of its working context to keep context windows
manageable; the UI must not lose them. The transcript accumulates every turn ever
sent, independent of SDK compaction.

**Storage shape:** Per-project JSONL at `<project_dir>/chat/transcript.jsonl`, with
a cursor sidecar at `<project_dir>/chat/transcript.cursor` (newline-delimited set
of already-written message ids). The cursor makes appends idempotent so a re-run
or restart never duplicates messages.

**Invariant — tail-mutable, body-immutable:** The TAIL record (the most-recently-written
message) is re-mappable: if the same ID appears with changed ``(role, parts)``, the
tail is rewritten in place. Older body records are immutable — a re-append with a body
ID is an idempotent no-op. WHY: Closes the flush-ordering window where the ``ResultMessage``
is delivered before the SDK flushes the final text entry to the session file. Under the
old append-only rule, that partial tail would lock forever. The tail-rewrite allows the
next turn to deliver the full assistant turn (tool + trailing text), updating the incomplete
prior write without duplicating the turn.

**Naming rationale:** Bare-noun identity module `chat_transcript` (it IS the durable
transcript entity, one instance per project — same shape rationale as optimization_state.py).
It always resolves to a project-local/EFS path via resolve_project_dir, so it is not a
swappable-backend `_store`. Per docs/naming-conventions.md bare-noun rule.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.services.chat_message_parser import is_cli_task_envelope

__all__ = [
    "append_messages",
    "load_transcript",
    "is_current_schema",
    "discard_transcript",
    "PROVISIONAL_USER_ID_PREFIX",
    "build_provisional_user_message",
    "build_turn_marker_message",
    "tag_turn_on_trailing_reply",
]

logger = logging.getLogger(__name__)

# Provisional user id sentinel prefix for relay start-of-turn writes.
# A deterministic token that a real SDK uuid can never collide with. The relay
# client writes a provisional user bubble at turn START (before the AgentCore
# HTTPS POST fires) so a mid-turn page refresh renders the conversation. When
# the microVM completion flush re-maps the turn, it carries the real SDK-generated
# uuid, and the tail-upgrade branch in append_messages replaces the provisional
# bubble with the authoritative one (no duplicate).
PROVISIONAL_USER_ID_PREFIX = "__EVAL_STUDIO_PROVISIONAL_USER__"


def _utc_now_iso() -> str:
    """Current UTC time in the transcript's timestamp format (``…Z``, ms precision).

    A non-empty ``created_at`` is LOAD-BEARING: :func:`is_current_schema` requires
    it, so a record written without one makes the whole transcript look
    stale-schema and get discarded on the next history load.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def build_turn_marker_message(turn_id: str, text: str, *, kind: str) -> dict:
    """Build an assistant-role marker standing in for a reply that never came.

    Some turns end with no assistant message at all: the user stopped them, or
    the CLI failed before any content streamed. The transcript then ends on the
    bare user bubble, and the user sees their own prompt with no reply and no
    explanation — the live WebSocket toast that said what happened does not
    survive a refresh, because the durable transcript stores only user/assistant
    messages.

    The marker fixes that by BEING an assistant message, which also means it
    needs no new part type and no frontend rendering path: it renders as
    ordinary assistant text. Two consequences worth knowing:

    * it changes the transcript's tail role, so ``GET /chat/history``'s
      interruption heuristic (last visible message is a user message) correctly
      stops offering "Resume" for a turn that is not resumable-by-retry. That
      was the original reason the stop marker exists.
    * the id is deterministic per ``(kind, turn_id)``, so a replayed or repeated
      write is deduped by :func:`append_messages` (case 3) rather than appending
      a second copy — and a changed message for the same turn rewrites the tail
      in place (case 2).

    Args:
        turn_id: The turn this marker explains.
        text: Plain human sentence, no markdown — this function applies the
            italic convention so every marker reads the same.
        kind: Short slug identifying the marker class (``"stopped"``,
            ``"failed"``). Forms the id prefix, so changing it for an existing
            marker class would orphan the old records rather than dedupe them.

    Returns:
        A current-schema ChatMessage dict.
    """
    return {
        "id": f"{kind}-{turn_id}",
        "role": "assistant",
        # Also carried as a real field, not just encoded in the id: the SPA's
        # reconcile guard matches on it (see tag_turn_on_trailing_reply), and a
        # turn that ends in a marker instead of a reply must still satisfy that
        # guard — otherwise the guard never converges for stopped/failed turns.
        "turn_id": turn_id,
        "parts": [{"type": "text", "text": f"_({text})_"}],
        "created_at": _utc_now_iso(),
    }


def tag_turn_on_trailing_reply(mapped: list[dict], turn_id: str | None) -> None:
    """Stamp ``turn_id`` on the turn's own trailing assistant message.

    Gives the SPA an identity link between the reply it streamed and the reply
    that reached the transcript. It has none otherwise — the client mints its
    own ``crypto.randomUUID()`` for the finalized bubble — so the only
    alternative is comparing message COUNTS, and every count proxy breaks
    somewhere:

    * a marker written by another path grows the count without being this
      turn's reply, so the guard passes and the bubble still vanishes;
    * a turn that legitimately produces no reply never grows the count, so the
      guard never converges;
    * with two turns in flight, a count cannot say WHICH reply arrived.

    Only ``mapped[-1]`` is stamped, and only when it is an assistant message —
    the same tail :func:`_reconcile_trailing_answer` targets. The CALLER must
    have already established that this turn's reply is present
    (``session_has_reply_after_prompt``): against a mapping that contains
    nothing of this turn the tail belongs to a PREVIOUS turn, and the
    transcript's body-immutability would make a wrong stamp permanent.

    Additive and optional by design. Messages written before this field existed
    simply lack it, which the consuming guard treats as "not this turn" and so
    degrades to keeping the streamed state — the safe direction. At 0.1 there is
    no compatibility to preserve and "absent means pre-field" is sufficient.

    Args:
        mapped: ChatMessage dicts from ``map_session_messages`` (mutated in place).
        turn_id: The turn that produced the trailing reply. Falsy is a no-op, so
            a caller without a turn id cannot stamp an empty marker that a
            guard might then match against.
    """
    if not turn_id or not mapped:
        return
    last = mapped[-1]
    if last.get("role") != "assistant":
        return
    last["turn_id"] = turn_id


def _extract_text_from_parts(parts: list) -> str:
    """Extract concatenated text from a message's parts list.

    Args:
        parts: List of part dicts (current schema: [{"type": "text", "text": "..."}]).

    Returns:
        Concatenated text from all text-type parts.
    """
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            if text:
                texts.append(text)
    return "".join(texts)


def build_provisional_user_message(text: str, turn_id: str) -> dict:
    """Build a provisional user message for relay start-of-turn writes.

    Returns a CURRENT-SCHEMA ChatMessage dict with:
    - id: f"{PROVISIONAL_USER_ID_PREFIX}{turn_id}" (deterministic, non-colliding)
    - role: "user"
    - parts: [{"type": "text", "text": text}]
    - created_at: current UTC time as ISO-8601 string ending in "Z"

    The non-empty created_at is LOAD-BEARING: is_current_schema requires it,
    so a mid-turn refresh doesn't discard the transcript and backfill from an
    incomplete session.

    Args:
        text: User message text.
        turn_id: Turn identifier for the provisional id.

    Returns:
        ChatMessage dict (current schema, non-empty created_at).
    """
    return {
        "id": f"{PROVISIONAL_USER_ID_PREFIX}{turn_id}",
        "role": "user",
        "parts": [{"type": "text", "text": text}],
        "created_at": _utc_now_iso(),
    }


async def append_messages(project_id: str, msgs: list[dict]) -> None:
    """Append messages to the durable transcript (idempotent, lock-guarded, tail-rewrite).

    Implements the tail-mutable / body-immutable invariant: the most-recently-written
    message can be rewritten if its ``(role, parts)`` changes, but older messages are
    immutable. WHY: Closes the flush-ordering window where ``ResultMessage`` is delivered
    before the SDK flushes the final text entry to the session file. Under the old
    append-only rule, that partial tail would lock forever. The tail-rewrite allows the
    next turn to deliver the full assistant turn (tool + trailing text), updating the
    incomplete prior write without duplicating the turn.

    For each message:
      1. New id (not in cursor) → append to JSONL and add to cursor
      2. id == TAIL id AND (role, parts) differs → REWRITE tail in place
      3. id == TAIL id AND (role, parts) identical → idempotent no-op
      4. id matches a NON-tail (body) id → skip (older history is immutable)

    Equality is keyed on ``(role, parts)`` only; ``created_at`` differences are ignored
    so a re-map with a new timestamp but identical content is a no-op, not a rewrite.

    The read-then-write critical section is serialized via the per-project asyncio
    lock from ``app.services.project_locks``. This ensures concurrent appends (e.g.,
    two optimization turns landing simultaneously) never race and corrupt the JSONL
    or cursor.

    Args:
        project_id: Project UUID.
        msgs: List of ChatMessage dicts (from map_session_messages).
    """
    from app.services.project_locks import get_project_lock
    from app.services.project_paths import resolve_project_dir

    project_dir = resolve_project_dir(project_id)
    chat_dir = project_dir / "chat"
    transcript_path = chat_dir / "transcript.jsonl"
    cursor_path = chat_dir / "transcript.cursor"

    async with await get_project_lock(project_id):
        # Lazily create chat/ dir on first write
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cursor (set of already-written ids)
        existing_ids: set[str] = set()
        if cursor_path.exists():
            try:
                existing_ids = set(cursor_path.read_text(encoding="utf-8").strip().splitlines())
            except Exception as exc:
                logger.warning("Failed to read cursor for %s: %s", project_id[:12], exc)

        # Read the current tail record (last non-blank line) to enable tail-rewrite
        tail_record: dict | None = None
        if transcript_path.exists():
            try:
                with transcript_path.open("r", encoding="utf-8") as rf:
                    lines = rf.readlines()
                    # Walk backwards to find the last valid JSON line (torn-line tolerant)
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            tail_record = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            # Skip malformed lines (torn write), keep looking
                            continue
            except Exception as exc:
                logger.warning("Failed to read tail for %s: %s", project_id[:12], exc)

        # Collect ALL un-upgraded provisional user entries (file order).
        # A STOPPED turn never delivers the completion flush that
        # upgrades its provisional start-write bubble, so that bubble goes
        # stale in the BODY (e.g. with the stop marker appended after it).
        # The next turn's full-session re-map then carries the same user
        # message under its real SDK id — a tail-only upgrade cannot
        # reconcile it and appends a DUPLICATE user bubble after a Stop.
        # Upgrade-by-text-match is therefore allowed against ANY provisional
        # entry, not just the tail: provisional bubbles are placeholders
        # awaiting their SDK identity, never immutable history.
        provisional_user_entries: list[dict] = []
        if transcript_path.exists():
            try:
                with transcript_path.open("r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rec_id = rec.get("id") or ""
                        if rec.get("role") == "user" and rec_id.startswith(PROVISIONAL_USER_ID_PREFIX):
                            provisional_user_entries.append(rec)
            except Exception as exc:
                logger.warning(
                    "Failed to scan provisional entries for %s: %s", project_id[:12], exc
                )

        # Compute the tail id for comparison
        tail_id: str | None = None
        if tail_record:
            tail_id = tail_record.get("id")
            if not tail_id:
                # Tail has no id → compute its content hash
                content_for_hash = json.dumps(
                    {"role": tail_record.get("role"), "content": tail_record.get("content"), "tool_calls": tail_record.get("tool_calls")},
                    sort_keys=True,
                )
                tail_id = hashlib.sha256(content_for_hash.encode("utf-8")).hexdigest()[:16]

        # Process each incoming message
        new_ids: list[str] = []
        for msg in msgs:
            # Compute dedupe key: msg["id"] if truthy, else content hash
            msg_id = msg.get("id")
            if not msg_id:
                # Fallback: hash role+content+tool_calls for deduplication
                content_for_hash = json.dumps(
                    {"role": msg.get("role"), "content": msg.get("content"), "tool_calls": msg.get("tool_calls")},
                    sort_keys=True,
                )
                msg_id = hashlib.sha256(content_for_hash.encode("utf-8")).hexdigest()[:16]

            # Branch on tail vs body vs new
            # Provisional-tail upgrade for the relay start-write.
            # When the incoming message has role "user" AND is a NEW id (not in cursor)
            # AND the current tail is a user message whose id starts with the provisional
            # prefix → REWRITE the tail in place with the incoming message (the microVM's
            # completion flush carrying the real SDK uuid). This lets the completion re-map
            # REPLACE the provisional start-write bubble rather than duplicate it.
            #
            # The tail upgrade compares id/text before upgrading.
            # Match on role + provisional-tail + text equality. If an UNRELATED later
            # user message arrives, it must NOT clobber the provisional entry — append
            # as new instead. This protects against a later unrelated user message
            # overwriting the wrong provisional bubble.
            #
            # Text extraction: compare the structured `parts` text, never English
            # patterns. Both provisional and SDK messages have
            # parts=[{"type": "text", "text": ...}] in the current schema.
            if (
                msg_id not in existing_ids
                and msg.get("role") == "user"
                and provisional_user_entries
            ):
                msg_text = _extract_text_from_parts(msg.get("parts", []))

                # Match the OLDEST un-upgraded provisional with equal
                # text. Session re-maps arrive oldest-first, so first-match
                # pairs each SDK user entry with its own start-write bubble
                # even when several provisionals are pending. Text equality
                # keeps the protection: an unrelated user message never
                # clobbers someone else's provisional bubble.
                matched: dict | None = None
                for cand in provisional_user_entries:
                    if _extract_text_from_parts(cand.get("parts", [])) == msg_text:
                        matched = cand
                        break

                if matched is not None:
                    # Upgrade the provisional entry IN PLACE (position kept)
                    _rewrite_record_by_id(
                        transcript_path, matched["id"], msg, project_id
                    )
                    provisional_user_entries.remove(matched)
                    # Add the incoming id to the cursor (the provisional id is already there)
                    new_ids.append(msg_id)
                    existing_ids.add(msg_id)
                    # If the upgraded entry was the tail, track the new identity
                    if tail_id == matched["id"]:
                        tail_record = msg
                        tail_id = msg_id
                else:
                    # No matching provisional — this is an UNRELATED user message.
                    # Append as new (do NOT clobber any provisional bubble).
                    # failed-turn outcome: the provisional bubble remains as the
                    # user's message; the failure surfaces via the error envelope.
                    with transcript_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    new_ids.append(msg_id)
                    existing_ids.add(msg_id)
                    # Update tail (the new message is now the tail)
                    tail_record = msg
                    tail_id = msg_id

            elif msg_id not in existing_ids:
                # Case 1: New id → append
                with transcript_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                new_ids.append(msg_id)
                existing_ids.add(msg_id)
                # Update tail_record and tail_id for subsequent messages in this batch
                tail_record = msg
                tail_id = msg_id

            elif msg_id == tail_id:
                # Case 2 or 3: id matches the tail
                # Compare (role, parts) only — not created_at
                msg_role_parts = (msg.get("role"), json.dumps(msg.get("parts", []), sort_keys=True))
                tail_role_parts = (tail_record.get("role"), json.dumps(tail_record.get("parts", []), sort_keys=True))

                if msg_role_parts != tail_role_parts:
                    # Case 2: tail content changed → rewrite
                    _rewrite_tail(transcript_path, msg, project_id)
                    # Update tail_record for subsequent messages in this batch
                    tail_record = msg
                    tail_id = msg_id
                # else: Case 3: identical (role, parts) → no-op
            # else: Case 4: id matches a non-tail body id → skip (immutable older history)

        # Persist updated cursor (only if we appended new ids)
        if new_ids:
            with cursor_path.open("a", encoding="utf-8") as cf:
                cf.write("\n".join(new_ids) + "\n")


def _rewrite_record_by_id(
    transcript_path: Path, target_id: str, new_record: dict, project_id: str
) -> None:
    """Rewrite the record with ``target_id`` in place atomically.

    Generalizes ``_rewrite_tail`` to any line: used to upgrade an orphaned
    provisional user bubble (a stopped turn never flushed its completion, so
    the bubble is no longer the tail) with its real SDK identity. Position is
    preserved — the message keeps its place in the conversation. Same atomic
    temp-file + rename discipline as ``_rewrite_tail``; on failure the old
    record stays (the caller has already cursored the new id, so the retry
    path is the next turn's re-map, same as tail-rewrite failures).

    Args:
        transcript_path: Path to the transcript JSONL.
        target_id: The id of the record to replace.
        new_record: The replacement record dict.
        project_id: Project UUID for logging.
    """
    import tempfile

    try:
        with transcript_path.open("r", encoding="utf-8") as rf:
            lines = rf.readlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if rec.get("id") == target_id:
                lines[i] = json.dumps(new_record, ensure_ascii=False) + "\n"
                break

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=transcript_path.parent,
            delete=False,
        ) as tmp:
            tmp.writelines(lines)
            # The name is read here, but the file is replaced only after this
            # block, where the context manager has already closed and flushed it.
            # nosemgrep: tempfile-without-flush
            tmp_path = Path(tmp.name)

        tmp_path.replace(transcript_path)

    except Exception as exc:
        logger.warning(
            "Failed to rewrite record %s for %s: %s",
            target_id[:24], project_id[:12], exc,
        )


def _rewrite_tail(transcript_path: Path, new_tail: dict, project_id: str) -> None:
    """Rewrite the tail record in place atomically (read all lines, replace tail, atomic rename).

    WHY atomic rename: A crash mid-rewrite must never leave a torn/blanked transcript.
    Write to a temp file in the same dir, then atomically replace via os.replace() so
    a crash produces either the old tail or the new tail, never a partial file.

    Args:
        transcript_path: Path to the transcript JSONL.
        new_tail: The updated tail record dict.
        project_id: Project UUID for logging.
    """
    import tempfile

    try:
        # Read all existing lines
        with transcript_path.open("r", encoding="utf-8") as rf:
            lines = rf.readlines()

        # Replace the last non-blank line with the new tail
        # Walk backwards to find it (torn-line tolerant)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                lines[i] = json.dumps(new_tail, ensure_ascii=False) + "\n"
                break

        # Write to a temp file in the same dir (atomic rename requires same filesystem)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=transcript_path.parent,
            delete=False,
        ) as tmp:
            tmp.writelines(lines)
            # The name is read here, but the file is replaced only after this
            # block, where the context manager has already closed and flushed it.
            # nosemgrep: tempfile-without-flush
            tmp_path = Path(tmp.name)

        # Atomic rename
        tmp_path.replace(transcript_path)

    except Exception as exc:
        logger.warning("Failed to rewrite tail for %s: %s", project_id[:12], exc)
        # Best-effort: if rewrite fails, leave the old tail in place


def load_transcript(project_id: str) -> list[dict]:
    """Read the durable transcript from disk (tolerates missing file and torn lines).

    Returns every message in order. Missing file → []. Tolerates a malformed trailing
    line (catch json.JSONDecodeError per-line and skip) so a torn write never blanks
    the whole history — the worst case is "last turn missing", never "history blanked".

    Args:
        project_id: Project UUID.

    Returns:
        List of ChatMessage dicts.
    """
    from app.services.project_paths import resolve_project_dir

    project_dir = resolve_project_dir(project_id)
    transcript_path = project_dir / "chat" / "transcript.jsonl"

    if not transcript_path.exists():
        return []

    result: list[dict] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in transcript for %s: %s",
                    line_num,
                    project_id[:12],
                    exc,
                )
    return result


def is_current_schema(entries: list[dict]) -> bool:
    """Check whether the transcript uses the current `parts` schema AND has valid timestamps.

    The transcript is a DERIVED CACHE from the SDK session. When the schema
    evolves (e.g., the move from a flat `{content, tool_calls}` shape to an
    ordered `parts` list, or the addition of non-empty `created_at`), old
    transcripts become stale. Rather than parse the old shape, we discard and
    rebuild from the SDK session (cache invalidation).

    A transcript is "current" when every entry:
    1. Carries the `parts` key (parts model migration)
    2. Has a non-empty `created_at` field (timestamp backfill)
    3. Is NOT a baked-in CLI task envelope (a `<task-notification>`-style
       background-task wrapper the CLI sometimes materialized as a user entry
       before the mapper filter landed). Such an entry has valid `parts` +
       `created_at`, so the schema checks alone would keep re-serving it —
       treating it as stale routes the read path through
       ``discard_transcript`` → rebuild via the fixed mapper, which drops it.

    This is a POSITIVE membership check — it never inspects old-shape fields
    (`content`, `tool_calls`). Empty list → True (nothing to rebuild; the
    empty-transcript backfill owns that case).

    Args:
        entries: List of ChatMessage dicts from load_transcript.

    Returns:
        True if every entry has `parts` + non-empty `created_at` AND is not a
        CLI task envelope; False otherwise.
    """
    return all(
        "parts" in entry
        and entry.get("created_at")
        and not is_cli_task_envelope(_extract_text_from_parts(entry.get("parts", [])))
        for entry in entries
    )


async def discard_transcript(project_id: str) -> None:
    """Remove the durable transcript files so a rebuild can recreate them.

    Discards the stale-schema transcript by removing both `transcript.jsonl` and
    `transcript.cursor`. The removal is serialized via the same per-project lock
    as `append_messages` so we never half-delete mid-append. Missing files are
    a no-op (idempotent). The SDK session is never touched — it is the
    source-of-truth that the rebuild reads.

    This is cache-invalidation, not a data loss path: the SDK session holds the
    real history; the transcript is a derived UI cache. A stale transcript gets
    discarded on read, then the existing empty-transcript backfill branch rebuilds
    it in the current format from the SDK session (via the same resolve_project_dir
    + platform.session_store + map_session_messages wiring it already uses).

    Args:
        project_id: Project UUID whose transcript to discard.
    """
    from app.services.project_locks import get_project_lock
    from app.services.project_paths import resolve_project_dir

    project_dir = resolve_project_dir(project_id)
    chat_dir = project_dir / "chat"
    transcript_path = chat_dir / "transcript.jsonl"
    cursor_path = chat_dir / "transcript.cursor"

    async with await get_project_lock(project_id):
        # Idempotent: missing_ok so we never raise on absent files
        transcript_path.unlink(missing_ok=True)
        cursor_path.unlink(missing_ok=True)
