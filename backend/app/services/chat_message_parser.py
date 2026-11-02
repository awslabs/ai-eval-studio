# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK SessionMessage → ChatMessage mapper with ordered parts model.

Single source of truth for converting SDK SessionMessages to frontend
ChatMessage dicts. Shared by:
- **Read path**: ``get_chat_history`` (serves the durable transcript on reload).
- **Write path**: ``agent_runner`` (appends each turn to the transcript).

Produces an ordered ``parts`` list that captures text/thinking/tool blocks
in source order, with tool results matched by tool_use_id and large fields
truncated with explicit markers. This guarantees the read path and write
path stay in lockstep — whatever the user saw in the live stream is exactly
what reappears on page refresh, including thinking content and tool results.

**Truncation rationale**: Tool results can run 100K+ chars (raw JSON), and
thinking blocks can be arbitrarily long. The durable transcript must carry
bounded previews so it cannot overflow the 900 KB MCP payload guard / 1 MiB
SDK transport cap (the raw content stays in the SDK session, reachable on
demand). We truncate to 2 KB for tool results and 4 KB for thinking, with
explicit ``truncated: true`` markers so the frontend can indicate when a
field has been truncated.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "map_session_messages",
    "is_synthetic_message",
    "is_lifecycle_message",
    "is_cli_task_envelope",
    "strip_lifecycle_markers",
    "ORCHESTRATOR_MSG_PREFIX",
    "LIFECYCLE_MSG_PREFIX",
]

# Preview budget for a tool_result in the durable transcript.
# Tool results (raw JSON from MCP tools) can exceed 100K chars; we truncate
# to this limit so the transcript stays under the 900 KB MCP payload guard /
# 1 MiB SDK transport cap. The raw result is reachable in the SDK session.
_TOOL_RESULT_PREVIEW_MAX_CHARS = 2000

# Preview budget for thinking blocks in the durable transcript.
# Thinking can be arbitrarily long; truncating to a few KB prevents the
# transcript from exceeding the transport caps while still surfacing enough
# context for the UI to show what Claude was reasoning about.
_THINKING_PREVIEW_MAX_CHARS = 4000

# Machine-readable prefix prepended to every orchestrator-origin message
# before it enters the SDK session. On history reload, messages with this
# prefix are filtered out so they never appear as user bubbles. Uses a
# deterministic token that survives JSON serialization and is impossible
# for a user to accidentally type.
ORCHESTRATOR_MSG_PREFIX = "__EVAL_STUDIO_SYS_7f3a9b__:"

# Machine-readable prefix for lifecycle-triggered orchestrator messages
# (optimization start/resume/cancel). Prepended AFTER ORCHESTRATOR_MSG_PREFIX
# so the message has both markers. On history reload, these render as
# attributed system messages (role:"system") instead of being dropped.
# The prefix order is load-bearing: is_synthetic_message checks startswith
# ORCHESTRATOR_MSG_PREFIX first; reversing the order breaks that check and
# renders the message as a user bubble with raw markers visible.
LIFECYCLE_MSG_PREFIX = "__EVAL_STUDIO_LIFECYCLE__:"


def is_synthetic_message(content: str) -> bool:
    """Detect orchestrator-origin synthetic messages in session history.

    Args:
        content: Raw message content string from the session.

    Returns:
        True if this message should be hidden from the user.
    """
    return content.startswith(ORCHESTRATOR_MSG_PREFIX)


# SDK/CLI system-turn envelopes injected for background shell tasks. These are
# the SDK's canonical wrapper tags (a fixed structured format, NOT free text),
# never genuine conversation. The CLI normally persists them as
# ``type:"queue-operation"`` entries (dropped by the role filter), but it
# sometimes materializes a dequeued notification as a real ``type:"user"``
# entry — which then renders as a user bubble and bakes into the durable
# transcript. Matching the leading tag (structural, not English) filters those.
_CLI_TASK_ENVELOPE_TAGS = ("<task-notification>", "<task-progress>", "<task-started>")


def is_cli_task_envelope(content: str) -> bool:
    """Detect an SDK/CLI background-task envelope masquerading as a user message.

    Args:
        content: Raw message content string from the session.

    Returns:
        True if the content is a CLI task-envelope wrapper (must not render as
        a chat bubble).
    """
    return content.lstrip().startswith(_CLI_TASK_ENVELOPE_TAGS)


def is_lifecycle_message(content: str) -> bool:
    """Detect a lifecycle-triggered orchestrator message.

    CRITICAL: This tests the REMAINDER after stripping ORCHESTRATOR_MSG_PREFIX,
    NOT the raw content. The prefix order is ORCHESTRATOR + LIFECYCLE; reversing
    it breaks is_synthetic_message (which checks startswith ORCHESTRATOR_MSG_PREFIX)
    and causes the message to render as a user bubble with raw markers visible.

    Args:
        content: Raw message content string from the session (already known to
            be synthetic via is_synthetic_message).

    Returns:
        True if this is a lifecycle message (start/resume/cancel) that should
        render as role:"system".
    """
    # Strip the orchestrator prefix first, then test for lifecycle marker.
    # This documents the order dependency at the detection site.
    remainder = content.removeprefix(ORCHESTRATOR_MSG_PREFIX)
    return remainder.startswith(LIFECYCLE_MSG_PREFIX)


def strip_lifecycle_markers(content: str) -> str:
    """Strip both ORCHESTRATOR_MSG_PREFIX and LIFECYCLE_MSG_PREFIX from content.

    Emitted system-part text must carry neither prefix
    (``test_chat_api.py::TestChatHistoryTranscript::test_synthetic_messages_excluded`` asserts
    "__EVAL_STUDIO_SYS_" never appears in emitted text). This helper
    removes both markers to produce clean display text.

    Args:
        content: Raw message content from the session.

    Returns:
        Display text with both prefixes removed.
    """
    clean = content.removeprefix(ORCHESTRATOR_MSG_PREFIX)
    clean = clean.removeprefix(LIFECYCLE_MSG_PREFIX)
    return clean


def _block_get(block: Any, key: str, default: Any = "") -> Any:
    """Read a field from a content block (dict or object)."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _entry_get(entry: Any, key: str, default: Any = None) -> Any:
    """Read a field from a session entry (dict or SDK SessionMessage object)."""
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def map_session_messages(raw_messages: list) -> list[dict]:
    """Convert SDK SessionMessages to ChatMessage dicts with ordered parts.

    The SDK persists each content block as a separate SessionMessage in the
    transcript (one record per text block, one per tool_use, etc.). From the
    user's perspective those all belong to a single agent turn, so we
    coalesce consecutive assistant records into one ChatMessage.

    **Ordered parts model**: Instead of joining text with blank lines and
    flattening tool_calls, we produce an ordered ``parts`` list capturing
    text/thinking/tool blocks IN SOURCE ORDER. Tool results (delivered as
    list-content user messages) are matched to their tool parts by
    ``tool_use_id`` and attached as ``result_preview`` + ``status`` fields.
    Large fields (tool results, thinking) are truncated with explicit
    ``truncated: true`` markers.

    Filters out SDK-internal synthetic user messages (orchestrator-origin
    messages prefixed with ORCHESTRATOR_MSG_PREFIX).

    Accepts BOTH SDK SessionMessage objects (from ``get_session_messages``)
    AND raw store/JSONL dict entries (from ``load_raw_session_entries``).
    The dual input mode is handled by field accessors that work on both
    shapes — dict keys when the entry is a dict, object attributes when it
    is a SessionMessage.

    Args:
        raw_messages: List of SessionMessage objects OR raw dict entries.

    Returns:
        List of ChatMessage dicts with id, role, parts, created_at. The
        ``parts`` field is an ordered list of
        ``{"type":"text"|"thinking"|"tool", ...}`` dicts.
    """
    result: list[dict] = []
    # Accumulator for the assistant turn currently being built.
    assistant_id = ""
    assistant_parts: list[dict] = []
    assistant_created_at = ""
    # Map tool_use_id → part index so we can match tool_result blocks to
    # their tool parts out-of-order.
    tool_use_id_map: dict[str, int] = {}

    def flush_assistant() -> None:
        """Emit the accumulated assistant turn (if any) and reset the buffer."""
        nonlocal assistant_id, assistant_parts, assistant_created_at, tool_use_id_map
        if assistant_parts:
            result.append({
                "id": assistant_id,
                "role": "assistant",
                "parts": assistant_parts,
                "created_at": assistant_created_at or "",
            })
        assistant_id = ""
        assistant_parts = []
        assistant_created_at = ""
        tool_use_id_map = {}

    for msg in raw_messages:
        role = _entry_get(msg, "type")  # "user" or "assistant"
        # Type contract (defensive, no behavior change today): only "user" and
        # "assistant" entries carry conversation. Anything else (e.g. the SDK's
        # "queue-operation" entries) is skipped so it can never render.
        if role not in ("user", "assistant"):
            continue
        raw_message = _entry_get(msg, "message") or {}
        raw_content = raw_message.get("content", "") if isinstance(raw_message, dict) else ""
        msg_uuid = _entry_get(msg, "uuid") or ""

        # Drop CLI-synthesized assistant bookkeeping entries. The CLI writes
        # placeholder assistant records into the session for turns that
        # produced no model response (e.g. "No response requested." after an
        # aborted turn). The discriminator is STRUCTURAL, not textual: these
        # entries carry model="<synthetic>" (a real reply carries a real
        # model id). They are session bookkeeping, not conversation — with
        # the transcript rebuilt from the full session (single-writer),
        # rendering them puts stray bubbles in the chat (for example after
        # a stop-then-continue).
        if (
            role == "assistant"
            and isinstance(raw_message, dict)
            and raw_message.get("model") == "<synthetic>"
        ):
            continue

        if role == "user":
            if isinstance(raw_content, str) and raw_content.strip():
                # Plain string content: check for CLI task envelopes first
                # (unconditional drop), then check for synthetic messages.
                # Lifecycle synthetic messages emit as role:"system"; all other
                # synthetic messages are dropped.
                if is_cli_task_envelope(raw_content):
                    flush_assistant()
                    continue
                if is_synthetic_message(raw_content):
                    if is_lifecycle_message(raw_content):
                        flush_assistant()
                        # Emit as system message with exactly ONE text part.
                        # The frontend runtime THROWS if a system message has
                        # zero parts or multiple parts — this is a runtime
                        # invariant, not a type-check the compiler can enforce.
                        clean_text = strip_lifecycle_markers(raw_content)
                        result.append({
                            "id": msg_uuid,
                            "role": "system",
                            "parts": [{"type": "text", "text": clean_text}],
                            "created_at": _entry_get(msg, "timestamp") or "",
                        })
                        continue
                    # Non-lifecycle synthetic (iterations, phantoms) — drop.
                    flush_assistant()
                    continue
                # Real user message → boundary: flush assistant turn first.
                flush_assistant()
                result.append({
                    "id": msg_uuid,
                    "role": "user",
                    "parts": [{"type": "text", "text": raw_content}],
                    "created_at": _entry_get(msg, "timestamp") or "",
                })
            elif isinstance(raw_content, list):
                # List content = SDK internal tool_result blocks. Match each
                # to its tool part by tool_use_id and attach result_preview +
                # status. This keeps the assistant turn open (no flush).
                for block in raw_content:
                    btype = _block_get(block, "type")
                    if btype == "tool_result":
                        tool_use_id = _block_get(block, "tool_use_id")
                        is_error_flag = _block_get(block, "is_error", False)
                        status = "error" if is_error_flag else "ok"
                        # Extract the result content (can be string or list).
                        result_raw = _block_get(block, "content", "")
                        if isinstance(result_raw, list):
                            # List of blocks; concatenate text from each.
                            parts_text = []
                            for rb in result_raw:
                                rb_text = _block_get(rb, "text", "")
                                if rb_text:
                                    parts_text.append(str(rb_text))
                            result_text = " ".join(parts_text)
                        else:
                            result_text = str(result_raw) if result_raw else ""
                        # Truncate to the preview budget.
                        truncated = len(result_text) > _TOOL_RESULT_PREVIEW_MAX_CHARS
                        result_preview = result_text[:_TOOL_RESULT_PREVIEW_MAX_CHARS] if truncated else result_text
                        # Collapse whitespace to one line.
                        result_preview = " ".join(result_preview.split())
                        # Match to the tool part by tool_use_id.
                        if tool_use_id in tool_use_id_map:
                            idx = tool_use_id_map[tool_use_id]
                            assistant_parts[idx]["result_preview"] = result_preview
                            assistant_parts[idx]["status"] = status
                            assistant_parts[idx]["truncated"] = truncated
                        # Else: unmatched tool_result (defensive — ignore).

        elif role == "assistant":
            if not assistant_id:
                # First SDK record of a new assistant turn — anchor the id + timestamp.
                assistant_id = msg_uuid
                assistant_created_at = _entry_get(msg, "timestamp") or ""
            if isinstance(raw_content, list):
                for block in raw_content:
                    btype = _block_get(block, "type")
                    if btype == "text":
                        text = _block_get(block, "text")
                        if text:
                            assistant_parts.append({"type": "text", "text": str(text)})
                    elif btype == "thinking":
                        # The SDK thinking block carries its text under the
                        # ``thinking`` field, with ``text`` as a fallback.
                        thinking_text = _block_get(block, "thinking") or _block_get(block, "text", "")
                        if thinking_text:
                            thinking_str = str(thinking_text)
                            truncated = len(thinking_str) > _THINKING_PREVIEW_MAX_CHARS
                            preview = thinking_str[:_THINKING_PREVIEW_MAX_CHARS] if truncated else thinking_str
                            assistant_parts.append({
                                "type": "thinking",
                                "text": preview,
                                "truncated": truncated,
                            })
                    elif btype == "tool_use":
                        tool_name = _block_get(block, "name")
                        tool_input = _block_get(block, "input", {}) or {}
                        tool_use_id = _block_get(block, "id", "")
                        # Mirror the live-stream rewrite in agent_runner.py:
                        # the SDK's built-in Task tool carries subagent
                        # delegation — surface the subagent identity.
                        if tool_name in ("Task", "Agent") and isinstance(tool_input, dict):
                            subagent_type = tool_input.get("subagent_type")
                            if isinstance(subagent_type, str) and subagent_type:
                                tool_name = f"subagent:{subagent_type}"
                        part_idx = len(assistant_parts)
                        assistant_parts.append({
                            "type": "tool",
                            "name": tool_name,
                            "input": tool_input,
                            "result_preview": None,
                            "status": "running",
                            "truncated": False,
                        })
                        # Register this part so we can match the tool_result later.
                        if tool_use_id:
                            tool_use_id_map[tool_use_id] = part_idx

    # Flush any assistant turn still in the buffer (e.g. the final reply).
    flush_assistant()
    return result


def find_last_prompt_uuid(entries: list) -> str | None:
    """Find the uuid of the newest prompt entry in the session.

    A prompt is a type:"user" entry whose message.content is a non-empty string
    (not a list). This discriminator is the same one the mapper uses to separate
    real prompts (turn boundaries) from tool-result user entries (which carry
    list content and keep the assistant turn open).

    Hidden orchestrator prompts (content starting with ORCHESTRATOR_MSG_PREFIX)
    ARE prompts for boundary purposes — the marker changes rendering, never turn
    identity. A hidden prompt starts a turn.

    Args:
        entries: Raw session entries (SDK SessionMessage objects or store dicts).

    Returns:
        The uuid of the newest prompt entry, or None when no prompt exists or
        when the prompt entry carries no uuid (unanchored).
    """
    last_prompt_uuid = None
    for entry in entries:
        entry_type = _entry_get(entry, "type")
        if entry_type != "user":
            continue
        raw_message = _entry_get(entry, "message")
        if not isinstance(raw_message, dict):
            raw_message = {"content": getattr(raw_message, "content", None)}
        raw_content = raw_message.get("content")
        # Prompt discriminator: string content that is non-empty after strip.
        # List content = tool results (keep assistant turn open, not a boundary).
        if isinstance(raw_content, str) and raw_content.strip():
            last_prompt_uuid = _entry_get(entry, "uuid")
    return last_prompt_uuid


def session_has_reply_after_prompt(entries: list, prompt_uuid: str | None) -> bool:
    """Check if the session holds a real assistant entry newer than the anchored prompt.

    This is the load-bearing question for the transcript append path: has THIS
    turn's reply flushed to the session yet? The mapped tail's role cannot answer
    it because the mapper drops hidden prompts, so the newest mapped message can
    belong to the PREVIOUS turn.

    The predicate separates three race shapes:
    - Reply present and complete (trailing answer text flushed)
    - Reply present but its trailing text block not yet flushed (partial flush)
    - Reply entirely absent (whole-message race)

    The third shape looks identical to "someone else's reply is the newest thing
    in the session" unless the prompt boundary is consulted, which is precisely
    why the caller cannot use the mapped tail's role.

    Args:
        entries: Raw session entries (SDK SessionMessage objects or store dicts).
        prompt_uuid: The uuid of this turn's prompt entry. When truthy AND present
            in the entries, used as the boundary. When absent or None, falls back
            to the newest prompt entry. When no prompt exists at all, the predicate
            is deliberately permissive (returns True if any real assistant entry
            exists) so a prompt-less session shape keeps its current behavior.

    Returns:
        True when any entry AFTER the boundary is a real assistant entry
        (type:"assistant", not model:"<synthetic>"). The <synthetic> exclusion
        mirrors the mapper's structural drop: an aborted turn's placeholder record
        cannot be mistaken for a reply.
    """
    # Resolve the boundary: find the index of the anchored prompt (or fallback).
    boundary_idx = -1
    if prompt_uuid:
        # Try to find the anchored prompt uuid.
        for i, entry in enumerate(entries):
            if _entry_get(entry, "uuid") == prompt_uuid:
                boundary_idx = i
                break
    if boundary_idx == -1:
        # Anchor uuid not found or not provided — fall back to newest prompt.
        # This degrades to a best-effort answer instead of reporting permanent
        # false and burning a poll budget when the session was compacted/rewritten.
        for i, entry in enumerate(entries):
            entry_type = _entry_get(entry, "type")
            if entry_type != "user":
                continue
            raw_message = _entry_get(entry, "message")
            if not isinstance(raw_message, dict):
                raw_message = {"content": getattr(raw_message, "content", None)}
            raw_content = raw_message.get("content")
            if isinstance(raw_content, str) and raw_content.strip():
                boundary_idx = i

    # Check for any real assistant entry after the boundary.
    for i in range(boundary_idx + 1, len(entries)):
        entry = entries[i]
        entry_type = _entry_get(entry, "type")
        if entry_type != "assistant":
            continue
        # Exclude synthetic bookkeeping entries (mirrors mapper's structural drop).
        raw_message = _entry_get(entry, "message")
        if isinstance(raw_message, dict) and raw_message.get("model") == "<synthetic>":
            continue
        # Real assistant entry found after the boundary.
        return True

    return False
