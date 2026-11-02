# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore relay client: server-side turn invocation + SSE drain + republish.

This module implements the relay (used when ``resolve_agentcore_relay`` is True, i.e.
in cloud mode): the control plane
invokes the deployed AgentCore runtime server-side via raw HTTPS with bearer token
authentication, drains the streaming turn response, and republishes each turn event
onto the existing /ws on stream turn:<id> via ws_hub. The browser keeps its current
/ws chat UI while the agent turn runs in an isolated AgentCore microVM instead of
in-process.

AWS requires raw HTTPS POST with Authorization: Bearer <jwt> for OAuth-integrated
runtimes — the AWS SDK InvokeAgentRuntime call (SigV4) is explicitly disallowed
for such runtimes.

**The relay replaces the in-process turn:** When the relay is on, the in-process
SDK turn does NOT also run — the relay path replaces it, not adds to it.

**Failure handling:** Every failure exit publishes a terminal error/result envelope
on stream turn:<id> AND a {type:"error"} project event so the UI never hangs "in
progress". The failure envelope reuses the existing error channel (subtype:"error"
result envelope + ResourceError -> toast), not a parallel error channel.

**Token lifetime:** The invoke carries the caller's access token, and the control
plane holds no refresh token, so nothing renews it. An auth rejection (HTTP 401/403)
surfaces as a DISTINGUISHABLE "Session expired — please sign in and retry" terminal
error. The drain is bounded by a frame-idle limit (RELAY_FRAME_IDLE_LIMIT_S) and an
absolute turn cap (RELAY_TURN_MAX_S); the socket read timeout (BEDROCK_READ_TIMEOUT_S)
only fires when no bytes arrive at all. Long work runs as Fargate eval jobs.

**Durable turn log:** Each drained frame is durably appended to the turn-event log
via allocate_turn_seq + append_turn_event (background, non-fatal). The durable
payload carries the DDB-allocated seq, NOT make_envelope's in-memory seq: one seq
domain per stream, because two domains skip or duplicate frames on replay. The
terminal frame (success result OR the error/result) is also appended durably, so a
mid-turn reconnect can replay the terminal state. This is the in-flight source the
/ws turn replay reads (``_replay_turn_stream_from_durable_log``).

**Project-event re-target:** The drain also recognizes project_event frames emitted
by the microVM (via forward_project_event_to_coordinator injecting into the turn
buffer). These are re-targeted onto project:<id> where the browser's useBackendEvents
consumer listens, allocating no durable turn seq and appending nothing to the turn
log. The re-target is necessary because the SSE wire carries only kind and data — the
destination stream does not survive the boundary.

**Session resume:** The relay resolves the resume runtime_session_id +
app_session_id from the DURABLE session registry (agentcore_session_registry.py,
per-(user, project) grain) so the invoked turn resumes the right conversation from
the S3 transcript across StopRuntimeSession / microVM recycle. The in-memory map
(agentcore_session.py) is no longer used on this relay/control-plane path.

**Turn claim release:** The (user, project) turn claim is acquired in
POST /chat BEFORE this task is spawned; this task RELEASES the claim in a finally on
EVERY exit (success, non-2xx, transport error, unexpected exception, early returns)
so a crashed or errored relay turn never permanently wedges the project.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator
from urllib.parse import quote

import certifi
import httpx

from app.platform import get_platform
from app.services.agentcore_session_registry import (
    record_session,
    release_turn,
    resolve_resume,
)
from app.services.project_event_frame import (
    PROJECT_EVENT_FRAME_KIND,
    read_project_event_frame,
)
from app.services.resource_error import ResourceError
from app.services.ws_hub import get_ws_hub
from app.services.ws_envelope import make_envelope, make_envelope_with_seq

logger = logging.getLogger(__name__)

# Module-level set to retain durable-append background tasks (not GC'd mid-flight)
_durable_append_tasks: set = set()

#: Milliseconds above which one durable turn-event append is called out
#: individually.
#:
#: This is the FIRST stated budget for this write — the relay's durable append
#: was never timed, so "is the durable write fast enough" had no answer in the
#: logs and could not be checked in cloud at all. No prior spec sets a number
#: here; searched and there is none, so do not go looking for one. 300ms is
#: chosen as a threshold a single-item DynamoDB write should comfortably beat, so
#: crossing it means something is wrong rather than merely busy. It is a
#: reporting threshold only and throttles nothing — retune it freely against
#: what the summary line actually reports.
#:
#: Why it matters: the append is fire-and-forget in a background task, so a slow
#: or failing write is invisible live and only shows up later as a replayed turn
#: missing events. Timing it is what makes that failure mode observable before a
#: user hits it.
_DURABLE_APPEND_SLOW_MS = 300.0


async def _publish_terminal_error(
    turn_id: str,
    project_id: str,
    owner_id: str,
    error_message: str,
    error_code: str = "relay.invocation_failed",
) -> None:
    """Publish terminal error envelope + project event on relay failure.

    Every failure exit in invoke_agentcore_turn calls this to emit:
    1. A terminal error/result envelope on stream turn:<id> (same shape as
       agent_runner.py in-process error envelope)
    2. A {type:"error"} project event so useBackendEvents toasts
    3. Durably append the terminal frame so reconnect sees the terminal state

    This is the existing error channel — reusing the in-process error shape,
    not inventing a parallel channel.

    Args:
        turn_id: Turn identifier for stream turn:<id>.
        project_id: Project UUID for project event.
        owner_id: User ID for tenant-scoped publish.
        error_message: Human-readable error message for the toast.
        error_code: Stable error code (defaults to relay.invocation_failed).
    """
    hub = get_ws_hub()
    store = get_platform().metadata_store

    # Build the terminal error envelope (same shape as run_agent_turn's in-process
    # error ``result`` envelope)
    error_data = {
        "subtype": "error",
        "result": None,
        "error": error_message,
        "session_id": None,
        "cost_usd": None,
        "num_turns": None,
    }
    # ONE SEQ DOMAIN: allocate the durable seq FIRST and use it for both the
    # live publish and the durable append below. This is the terminal frame, so a
    # client-side drop here is the difference between the UI settling and hanging
    # forever — precisely the frame that must not land in a second seq domain.
    #
    # Allocation failure degrades to the in-memory counter rather than dropping the
    # terminal frame: a mismatched seq the client may still accept beats no terminal
    # envelope at all.
    try:
        terminal_seq = await store.allocate_turn_seq(turn_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[relay] Could not allocate a durable seq for turn %s's terminal frame "
            "(%s) — falling back to the in-memory counter so the UI still unblocks",
            turn_id[:8], exc,
        )
        terminal_seq = None

    if terminal_seq is None:
        terminal_envelope = make_envelope(
            stream=f"turn:{turn_id}", kind="result", data=error_data,
        )
    else:
        terminal_envelope = make_envelope_with_seq(
            stream=f"turn:{turn_id}", kind="result", data=error_data, seq=terminal_seq,
        )

    # Publish on turn:<id> (tenant-scoped)
    await hub.publish_to_user(owner_id, terminal_envelope)

    # Durably append the terminal error frame (non-fatal on failure). Uses the
    # seq allocated above and the SAME envelope that went out live — no second
    # allocation, so the wire and the durable log cannot disagree.
    try:
        if terminal_seq is None:
            raise RuntimeError("no durable seq allocated for the terminal frame")
        ttl = int(time.time()) + 604800
        await store.append_turn_event(
            turn_id, terminal_seq, terminal_envelope, ttl=ttl
        )
    except Exception as exc:
        logger.warning(
            "[relay] Failed to durably append terminal error for turn %s: %s",
            turn_id[:8],
            exc,
        )

    # Build + publish the project error event (same shape as run_agent_turn's)
    turn_error = ResourceError(
        code=error_code,
        message=error_message,
        hint=None,
        details={"origin": "relay"},
    )
    await hub.publish_project(
        project_id,
        owner_id,
        {
            "resource": "agent_turn",
            "type": "error",
            "turn_id": turn_id,
            **turn_error.to_dict(),
        },
    )


class RelayIdleTimeout(Exception):
    """Raised when no REAL SSE frame arrives within the idle limit.

    Distinct from the socket read timeout: keepalive bytes keep the socket
    "alive" while a stalled session produces no frames. With the runtime's
    heartbeat frames counting as liveness, this now fires only when
    the runtime itself stopped serving the turn — a true death signal.
    """


class RelayTurnCapExceeded(Exception):
    """Raised when a turn outruns the absolute duration ceiling.

    Heartbeats prove the runtime is alive, so a wedged CLI subprocess that
    never finishes would otherwise hold the claim forever. The cap is the
    backstop bound: generous enough for legitimately long agent work (a
    harness build can run for most of an hour), tight enough that "forever"
    is impossible.
    """


async def _frames_with_idle_guard(
    frames: AsyncIterator[tuple[str, dict[str, Any]]],
    idle_limit_s: float,
    max_total_s: float,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield frames, bounding frame-silence AND total duration.

    Frame-silence bound: ``asyncio.wait_for`` on ``__anext__`` between
    MEANINGFUL frames (the parser never yields keepalive comments; runtime
    heartbeats count — that's their purpose). Total bound: a monotonic
    deadline checked per wait, so heartbeats can't extend a turn forever.
    """
    it = frames.__aiter__()
    deadline = asyncio.get_running_loop().time() + max_total_s
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RelayTurnCapExceeded(
                f"turn exceeded the {max_total_s:.0f}s duration ceiling"
            )
        try:
            item = await asyncio.wait_for(
                it.__anext__(), timeout=min(idle_limit_s, remaining)
            )
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            if deadline - asyncio.get_running_loop().time() <= 0:
                raise RelayTurnCapExceeded(
                    f"turn exceeded the {max_total_s:.0f}s duration ceiling"
                ) from exc
            raise RelayIdleTimeout(
                f"no SSE frame for {idle_limit_s:.0f}s"
            ) from exc
        yield item


async def _parse_sse_frames(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse SSE frames from an async byte iterator (e.g. httpx response.aiter_bytes()).

    Mirrors the frame-boundary logic from agentcore_runtime.py _sse_frames
    (event:/data:/blank-line framing, keepalive comments ignored, terminate on
    the 'result' kind). The control plane drains the AgentCore response just as
    the AgentCore serving adapter drains the AgentTurn buffer.

    Args:
        byte_iter: An async byte iterator (e.g., httpx response.aiter_bytes()).

    Yields:
        (kind, data) tuples, where data is the parsed JSON payload.
    """
    # Buffer for accumulating lines until we hit a blank line (frame boundary)
    current_event_kind = None
    current_data_lines: list[str] = []

    def _emit_frame() -> tuple[str, dict[str, Any]] | None:
        """Emit the accumulated frame, if complete."""
        nonlocal current_event_kind, current_data_lines
        if current_event_kind and current_data_lines:
            # Join multi-line data and parse JSON
            data_json = "".join(current_data_lines)
            try:
                data = json.loads(data_json)
            except json.JSONDecodeError:
                logger.warning("Failed to parse SSE data JSON: %s", data_json[:100])
                data = {}
            result = (current_event_kind, data)
            # Reset for next frame
            current_event_kind = None
            current_data_lines = []
            return result
        return None

    def _process_line(stripped: str) -> tuple[str, dict[str, Any]] | None:
        """Apply one SSE line to the frame accumulator; return a frame on boundary."""
        nonlocal current_event_kind, current_data_lines
        if not stripped:
            # Blank line — frame boundary
            return _emit_frame()
        if stripped.startswith(":"):
            # Keepalive comment — ignore
            return None
        if stripped.startswith("event:"):
            current_event_kind = stripped[len("event:") :].strip()
            return None
        if stripped.startswith("data:"):
            current_data_lines.append(stripped[len("data:") :].strip())
            return None
        return None

    # read the byte stream INCREMENTALLY and parse frames as bytes arrive,
    # yielding each frame as soon as its blank-line boundary is seen — so
    # publish_to_user fires per delta instead of after the whole turn. A line
    # split across a chunk boundary is held in `pending` until its trailing
    # newline arrives.
    pending = ""  # partial (unterminated) line carried across chunk boundaries

    async for chunk in byte_iter:
        if not chunk:
            break  # EOF
        pending += chunk.decode("utf-8")

        # Extract only COMPLETE lines (terminated by \n); keep the trailing
        # partial line in `pending` for the next chunk.
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            stripped = line.rstrip("\r")
            frame = _process_line(stripped)
            if frame:
                kind, data = frame
                yield kind, data
                if kind == "result":
                    return

    # Flush any trailing partial line (unterminated final line) + trailing frame.
    if pending:
        stripped = pending.rstrip("\r")
        frame = _process_line(stripped)
        if frame:
            kind, data = frame
            yield kind, data
            if kind == "result":
                return

    # Emit any trailing frame accumulated but not yet closed by a blank line.
    frame = _emit_frame()
    if frame:
        kind, data = frame
        yield kind, data


async def _stop_microvm_before_claim_release(
    turn_id: str,
    invoke_session_id: str | None,
    access_token: str | None,
    reason: str,
) -> bool:
    """Stop the microVM for a turn whose drain is being abandoned mid-flight.

    Why the order matters: the ``finally`` in ``invoke_agentcore_turn``
    releases the per-(user, project) claim on every exit. On a transport error or
    the 300s read timeout the local drain stops but the microVM keeps running and
    keeps writing the transcript/session. Releasing the claim then admits the
    user's retry as a SECOND concurrent writer against the still-live first turn —
    exactly the corruption the claim exists to prevent. So the stop must be issued
    BEFORE the claim frees, mirroring the ordering already used by the cancel path
    in ``app.api.chat.stop_turn``.

    Best-effort by design: this runs on paths that are already failing, and a
    failure here must never mask the original error or skip the claim release.

    Args:
        turn_id: Turn identifier (for logging).
        invoke_session_id: The AgentCore runtime session id actually sent on the
            invoke header. None before the header is built (nothing to stop).
        access_token: Caller's raw Authorization header value — StopRuntimeSession
            on an OAuth runtime needs it, so a missing token means we cannot stop.
        reason: Short description of the abandoning path, for the log line.

    Returns:
        ``True`` when a stop was issued, ``False`` on every skip/failure path.
    """
    agent_runtime_arn = os.environ.get("AGENTCORE_RUNTIME_ARN")
    if not invoke_session_id:
        logger.warning(
            "[relay] %s for turn %s with no runtime session id — cannot stop the "
            "microVM; it may run until its own read-timeout while the claim frees",
            reason, turn_id[:8],
        )
        return False
    if not agent_runtime_arn:
        logger.warning(
            "[relay] %s for turn %s but AGENTCORE_RUNTIME_ARN is unset — cannot "
            "stop the microVM before releasing the claim",
            reason, turn_id[:8],
        )
        return False
    if not access_token:
        logger.warning(
            "[relay] %s for turn %s with no access token — StopRuntimeSession "
            "would fail auth; skipping rather than issuing a guaranteed 401",
            reason, turn_id[:8],
        )
        return False

    try:
        # Lazy import: app.api.chat imports this module lazily at its spawn site,
        # so a module-scope import here would close the cycle.
        from app.api.chat import _stop_agentcore_runtime_session

        await _stop_agentcore_runtime_session(
            invoke_session_id, agent_runtime_arn, access_token,
        )
        logger.info(
            "[relay] %s for turn %s — StopRuntimeSession issued before claim release",
            reason, turn_id[:8],
        )
        return True
    except Exception as exc:  # noqa: BLE001 — must never mask the original error
        logger.warning(
            "[relay] %s for turn %s — StopRuntimeSession failed: %s. The microVM "
            "may still be writing when the claim frees.",
            reason, turn_id[:8], exc,
        )
        return False


async def invoke_agentcore_turn(
    project_id: str,
    message: str,
    owner_id: str,
    runtime_session_id: str | None,
    turn_id: str,
    access_token: str | None,
    app_session_fallback: str | None = None,
) -> None:
    """Invoke AgentCore runtime and relay turn events to /ws.

    Server-side invocation of the deployed AgentCore runtime via raw HTTPS POST
    with bearer token authentication. AWS explicitly disallows the SDK
    InvokeAgentRuntime call (SigV4) for OAuth-integrated runtimes; raw HTTPS
    with Authorization: Bearer <jwt> is required.

    The control plane drains the streaming response and republishes each turn
    event on stream turn:<id> via ws_hub.publish_to_user (tenant-scoped).

    **Start-of-turn persistence:** Before the HTTPS POST fires,
    persists project.session_id and writes the user message to the durable
    transcript (both best-effort, non-fatal). So a mid-turn page refresh renders
    the conversation (not a blank splash). When the microVM completion flush
    re-appends the turn, the provisional user bubble is upgraded in place (no
    duplicate).

    Args:
        project_id: Project UUID.
        message: User message to send to the agent.
        owner_id: User ID for tenant-scoped publish. This IS the caller's Cognito
            sub, so it doubles as the (user, project) session claim's user_id.
        runtime_session_id: AgentCore session ID (optional, for resume).
            Resolved against the DURABLE registry.
        turn_id: Turn identifier for the stream turn:<id>.
        access_token: The caller's raw incoming Authorization header value (may or
            may not include the "Bearer " prefix; this is the Cognito ACCESS token,
            never re-minted, never the id token).
        app_session_fallback: Caller-supplied app session id, used only when
            neither the durable registry nor the project config has one.

    **Error handling:** Logs and returns (does not raise) on transient AgentCore
    errors (throttling, timeouts, non-2xx responses, httpx transport errors).
    Every such exit publishes a terminal error envelope to the user instead
    (see ``_publish_terminal_error``) — this function does not crash the caller.

    **Claim release:** The (user, project) turn claim was
    acquired in POST /chat before this task was spawned. This function releases it
    in a finally on EVERY exit — the ARN-missing early return, the token-missing
    early return, the non-2xx return, an httpx transport error, an unexpected
    exception, AND the success path — so a crashed or errored relay turn never
    permanently wedges the project. Release is best-effort (never raises).
    """
    # Wrap the ENTIRE body in try/finally so the (user, project) claim is released
    # on every exit (early returns, non-2xx, transport error, exception, success).
    # owner_id is the Cognito sub — it doubles as the claim's user_id.
    #
    # Bound before the try so the error handlers can reach it: the mid-flight stop
    # needs the session id actually sent on the invoke header, and the
    # handlers run before that local would otherwise be assigned on failure paths.
    invoke_session_id: str | None = None
    try:
        # Bind the metadata store ONCE at the top of the function. It is used by
        # the turn-metadata write (owner-guard source), the terminal-error path,
        # and the drain-loop durable append. Assigning it only inside the drain
        # loop made ``store`` a function-local that was unbound at the earlier
        # put_turn_metadata call — silently swallowed by its try/except, which
        # meant turn metadata never persisted and durable replay always resolved
        # no owner.
        store = get_platform().metadata_store

        # Durable-append latency for THIS turn, summarised on the relay-complete
        # line. A plain dict rather than locals because the appends happen inside
        # a nested closure; mutating a dict needs no ``nonlocal`` chain.
        _append_stats: dict[str, float] = {
            "count": 0, "total_ms": 0.0, "max_ms": 0.0, "slow": 0, "failed": 0,
        }

        # Read env vars for the AgentCore runtime
        agent_runtime_arn = os.environ.get("AGENTCORE_RUNTIME_ARN")
        region = os.environ.get("AWS_REGION", "us-east-1")

        if not agent_runtime_arn:
            logger.error("[relay] AGENTCORE_RUNTIME_ARN not set — cannot invoke relay")
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                "AgentCore runtime ARN not configured — relay invocation failed",
                "relay.runtime_arn_missing",
            )
            return

        # Token guard: missing access_token is non-fatal (same discipline as ARN guard)
        if not access_token:
            logger.error("[relay] no access token — cannot invoke JWT-authorized runtime")
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                "Session expired — please sign in and retry",
                "relay.token_missing",
            )
            return

        # Resolve resume ids from the DURABLE registry, superseding the
        # in-memory resolver on this control-plane path. Returns
        # (runtime_session_id_to_use, app_session_id_to_resume): the durable item's
        # stored ids when present, else the project's persisted session_id fallback.
        runtime_session_id, session_id = await resolve_resume(
            owner_id, project_id, runtime_session_id
        )
        # The caller's ``session_id`` lands here and ONLY here — as the app
        # session of last resort, which is the role ``ChatRequest.session_id``
        # documents ("fallback if the backend failed to persist the session to
        # the project config"). It must never reach ``resolve_resume`` as the
        # runtime session id: that would let a caller choose which microVM
        # session the turn joins. Applied after resolution so a persisted or
        # durably-recorded session always wins over client-supplied state.
        if not session_id and app_session_fallback:
            logger.info(
                "[relay] No stored app session for project %s; using the "
                "caller-supplied fallback %s",
                project_id[:8],
                app_session_fallback[:12],
            )
            session_id = app_session_fallback
        if session_id:
            logger.debug(
                "[relay] Resolved durable resume -> app session %s for project %s",
                session_id[:12],
                project_id[:8],
            )

        # AgentCore's X-Amzn-Bedrock-AgentCore-Runtime-Session-Id has a
        # documented minimum length (verified: 33 chars). On a project's first turn
        # runtime_session_id is None; sending "" gets rejected. Generate a fresh
        # conforming id instead ("session-" + 32-char hex = 40 chars, >= 33).
        invoke_session_id = runtime_session_id or ("session-" + uuid.uuid4().hex)

        # Relay cancellation needs the runtime session id: the relay-task registry
        # entry is created at the spawn site with runtime_session_id=None, and the
        # runtime knows this turn by invoke_session_id (the header value above).
        # Write that id back into the entry so stop_turn's StopRuntimeSession has a
        # real session id to stop. Lazy import to avoid an import cycle with
        # app.api.chat (which imports this module lazily). Defensive: if the entry
        # is already gone (a cancel that raced in before this write-back — the
        # tiny pre-invoke window that stays a logged skip), do nothing. dict
        # item-set is atomic under the single event loop.
        try:
            from app.api.chat import _relay_tasks
            entry = _relay_tasks.get(turn_id)
            if entry is not None:
                entry["runtime_session_id"] = invoke_session_id
        except Exception as exc:  # noqa: BLE001 — write-back must never crash the turn
            logger.warning(
                "[relay] Could not write back invoke_session_id for turn %s: %s "
                "(cancel would skip StopRuntimeSession)",
                turn_id[:8], exc,
            )

        # START-OF-TURN PERSISTENCE: Persist session_id and write
        # the provisional user bubble to the durable transcript BEFORE the HTTPS
        # POST fires, so a mid-turn page refresh renders the conversation (not a
        # blank splash). The microVM reads the runtime-session HEADER via
        # resume_session_id_for and IGNORES the payload session_id; it falls back
        # to project.session_id. So persisting project.session_id at start makes
        # the microVM adopt the same id → no transcript fork.
        #
        # Best-effort: a failure here must NOT crash the turn or leak the claim.
        # Mirror _persist_session's except discipline (log at warning, continue).
        # The existing finally-release still runs on every exit.
        try:
            # Step 1: Resolve the app session id to persist.
            # Mint only when resolve_resume returned no app session id (fresh turn 1).
            # CANONICAL DASHED FORM: the CLI validates ``--session-id`` as a
            # strict UUID and exits 1 on the dashless ``uuid4().hex`` form
            # ("Invalid session ID. Must be a valid UUID."); ``resume`` tolerates
            # dashless, create-with-id does not.
            app_session_id = session_id or str(uuid.uuid4())

            # Step 2: Persist it via agent_runner._persist_session (ONE writer).
            # Lazy import to avoid import cycles.
            from app.services.agent_runner import _persist_session

            await _persist_session(project_id, app_session_id)

            # Step 3: Write the provisional user bubble to the durable transcript.
            # The provisional bubble is current-schema (non-empty created_at) so a
            # mid-turn refresh renders it instead of discarding the transcript.
            # When the microVM completion flush later re-appends the turn, the
            # upgrade branch replaces the provisional bubble with the real SDK-id'd
            # one — no duplicate.
            from app.services.chat_transcript import (
                append_messages,
                build_provisional_user_message,
            )

            provisional_bubble = build_provisional_user_message(message, turn_id)
            await append_messages(project_id, [provisional_bubble])

            # Keep the local session_id in sync with the persisted value
            # so the microVM's resume fallback, the payload, and the completion
            # record_session all agree on ONE id.
            session_id = app_session_id
        except Exception as exc:
            # Start-persist failure is non-fatal: log and continue. The existing
            # finally-release still runs so the claim is never leaked.
            logger.warning(
                "[relay] Failed to persist session/transcript at start for turn %s: %s (non-fatal)",
                turn_id[:8],
                exc,
            )

        # Build payload (after start-persist so it uses the synced session_id)
        payload = {
            "message": message,
            "project_id": project_id,
        }
        if session_id:
            payload["session_id"] = session_id

        # Store turn metadata for owner resolution (turn: replay owner-guard).
        # This enables _replay_turn_stream_from_durable_log to resolve turn_id →
        # (user_id, project_id) → owner_id without an owner-less turn-by-id lookup.
        # Non-fatal on failure (best-effort, same as start-persist above).
        try:
            ttl = int(time.time()) + 604800  # 7 days, same as turn events
            await store.put_turn_metadata(turn_id, user_id=owner_id, project_id=project_id, ttl=ttl)
        except Exception as exc:
            logger.warning(
                "[relay] Failed to store turn metadata for turn %s: %s (non-fatal)",
                turn_id[:8],
                exc,
            )

        # Normalize bearer header: RFC 6750 §2.1 — "Bearer" scheme is case-insensitive
        # (same rule as CognitoBearerJwtAuth.extract_user_id)
        first_token = access_token.split(None, 1)[0] if access_token else ""
        if first_token.lower() == "bearer":
            auth_header = access_token
        else:
            auth_header = f"Bearer {access_token}"

        # Raw HTTPS POST to /runtimes/{quote(arn)}/invocations?qualifier=DEFAULT
        # on bedrock-agentcore.{region}.amazonaws.com.
        #
        # AWS requires raw HTTPS with Authorization: Bearer <jwt> for OAuth-integrated
        # runtimes. The AWS SDK InvokeAgentRuntime call (SigV4) is explicitly
        # disallowed for such runtimes (AWS docs: "If you're integrating your agent
        # with OAuth, you can't use the AWS SDK to call InvokeAgentRuntime. Instead,
        # make a HTTPS request to InvokeAgentRuntime.").
        #
        # URL: /runtimes/{agentRuntimeArn}/invocations?qualifier={qualifier}
        #   agentRuntimeArn must be URL-encoded (urllib.parse.quote with safe='')
        #   qualifier: endpoint name pointing to a version; DEFAULT is the default
        #   accountId: omitted when passing a full ARN (only required for agent ID)
        #
        # Headers:
        #   Authorization: Bearer <access token> (caller's Cognito token)
        #   Content-Type: application/json
        #   Accept: text/event-stream (streaming SSE response)
        #   X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: {invoke_session_id} (min 33)
        url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{quote(agent_runtime_arn, safe='')}/invocations?qualifier=DEFAULT"

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": invoke_session_id,
        }

        read_timeout = float(os.environ.get("BEDROCK_READ_TIMEOUT_S", "300"))
        # FRAME-level liveness bound. The socket read timeout above only
        # fires when NO BYTES arrive — SSE keepalive comments defeat it, so a
        # live connection to a dead or stalled session would leave the drain
        # sitting in aiter_bytes indefinitely. The idle limit bounds the gap
        # between REAL frames
        # (keepalives don't count — the parser never yields them). Generous by
        # design: long tool executions (validate_harness) are silent between
        # tool_start and its result, so this is a safety net, not a latency
        # tool. A turn that produces nothing for this long is wedged.
        idle_limit = float(os.environ.get("RELAY_FRAME_IDLE_LIMIT_S", "900"))
        # Absolute duration ceiling. Heartbeats prove liveness, so a
        # wedged-but-alive CLI would otherwise hold the claim forever.
        # Default 2h: a legitimately long turn (a harness build can run for most
        # of an hour) fits with slack; "forever" does not. Users can always Stop
        # sooner.
        turn_cap = float(os.environ.get("RELAY_TURN_MAX_S", "7200"))

        # a clean SSE EOF is NOT proof of a completed turn. The runtime can
        # end the stream without a terminal `result` frame — there is a confirmed
        # in-repo producer: agent_runner logs "Ended WITHOUT ResultMessage" and
        # completes the turn, emitting no result envelope. Falling through to the
        # success path there leaves the UI spinning forever with no terminal
        # envelope while the claim is released. Track it and synthesize a terminal
        # error when the drain ends without one.
        saw_result = False
        result_subtype: str | None = None
        result_session_id: str | None = None
        result_final_answer: str | None = None

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=read_timeout,
                    write=30.0,
                    pool=30.0,
                ),
                verify=certifi.where(),
            ) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    content=json.dumps(payload).encode(),
                ) as response:
                    # Non-2xx response
                    if response.status_code >= 300:
                        body = await response.aread()
                        body_text = body.decode("utf-8", errors="replace")[:200]
                        logger.warning(
                            "[relay] InvokeAgentRuntime returned %d for turn %s: %s",
                            response.status_code,
                            turn_id[:8],
                            body_text,
                        )
                        # Distinguish auth-expiry (401/403) from generic errors
                        if response.status_code in (401, 403):
                            error_message = "Session expired — please sign in and retry"
                            error_code = "relay.token_expired"
                        else:
                            error_message = f"AgentCore invocation failed (HTTP {response.status_code})"
                            error_code = "relay.invocation_failed"

                        await _publish_terminal_error(
                            turn_id,
                            project_id,
                            owner_id,
                            error_message,
                            error_code,
                        )
                        return

                    # Drain the streaming response and republish (durable append added).
                    # ``store`` is already bound at the top of the function.
                    hub = get_ws_hub()
                    loop = asyncio.get_running_loop()

                    async for kind, data in _frames_with_idle_guard(
                        _parse_sse_frames(response.aiter_bytes()), idle_limit, turn_cap
                    ):
                        # Re-target project_event frames onto project:<id> where the
                        # browser's useBackendEvents consumer listens. The SSE wire
                        # carries only kind and data — the destination stream does not
                        # survive the boundary. Republishing on the turn stream instead
                        # would put the event where no project-event consumer subscribes,
                        # producing a change that logs success and updates nothing.
                        #
                        # The coordinator mints the destination envelope (via
                        # publish_project → make_envelope) rather than forwarding one
                        # built upstream, so the project stream keeps a single seq
                        # source: this process's counter. A pre-minted envelope from
                        # the microVM would introduce a second counter, and the client
                        # keeps one cursor per stream — the collision would be
                        # swallowed as a duplicate or cost a visible domain reset.
                        #
                        # This frame deliberately consumes no durable turn seq and is
                        # not appended to the turn log, because it is not part of the
                        # turn transcript. A hole or a replayed non-turn frame are
                        # both worse than nothing.
                        # heartbeats are liveness, not content. The idle
                        # guard has already counted this frame (that's its
                        # whole purpose); it must not reach the browser, the
                        # seq allocator, or the durable turn log.
                        if kind == "heartbeat":
                            continue
                        if kind == PROJECT_EVENT_FRAME_KIND:
                            event = read_project_event_frame(data, project_id)
                            if event is not None:
                                await hub.publish_project(project_id, owner_id, event)
                            # Continue past seq allocation, turn publish, and durable append
                            continue

                        if kind == "result":
                            saw_result = True
                            # Capture the terminal subtype for the project-event
                            # mirror below — a turn that ended with an error or
                            # cancellation must not be reported as completed.
                            # session_id + result feed the coordinator-side
                            # transcript append (single-writer).
                            if isinstance(data, dict):
                                result_subtype = data.get("subtype")
                                result_session_id = data.get("session_id")
                                result_final_answer = (
                                    data.get("result") if result_subtype == "success" else None
                                )
                        # Allocate a DDB seq synchronously (in emission order)
                        seq = await store.allocate_turn_seq(turn_id)

                        # ONE SEQ DOMAIN: the live publish carries the SAME
                        # durable seq allocated above, not a separate in-memory one.
                        # With two domains the client's single per-stream cursor
                        # either dropped the live frame as a duplicate or dropped the
                        # later durable replay as stale — silently, terminal frames
                        # included.
                        envelope = make_envelope_with_seq(
                            stream=f"turn:{turn_id}", kind=kind, data=data, seq=seq
                        )
                        await hub.publish_to_user(owner_id, envelope)

                        # Background durable append (non-fatal on failure)
                        async def _durable_append(seq: int, durable_event: dict):
                            started = time.perf_counter()
                            try:
                                # The SAME envelope that went out live — one domain
                                # means the durable record and the wire agree
                                # byte-for-byte on stream/seq/kind/data.
                                # TTL: turn events expire after 7 days.
                                ttl = int(time.time()) + 604800
                                await store.append_turn_event(turn_id, seq, durable_event, ttl=ttl)
                                elapsed_ms = (time.perf_counter() - started) * 1000.0
                                _append_stats["count"] += 1
                                _append_stats["total_ms"] += elapsed_ms
                                _append_stats["max_ms"] = max(
                                    _append_stats["max_ms"], elapsed_ms
                                )
                                if elapsed_ms > _DURABLE_APPEND_SLOW_MS:
                                    _append_stats["slow"] += 1
                                    logger.warning(
                                        "[relay] Slow durable append for turn %s seq %d: "
                                        "%.0fms (budget %.0fms)",
                                        turn_id[:8], seq, elapsed_ms,
                                        _DURABLE_APPEND_SLOW_MS,
                                    )
                            except Exception as exc:
                                _append_stats["failed"] += 1
                                # Durable append failure is non-fatal — log and continue
                                logger.warning(
                                    "[relay] Durable turn event append failed for turn %s seq %d: %s",
                                    turn_id[:8],
                                    seq,
                                    exc,
                                )

                        # Create background task and retain it (not GC'd mid-flight)
                        task = loop.create_task(_durable_append(seq, envelope))
                        _durable_append_tasks.add(task)
                        task.add_done_callback(_durable_append_tasks.discard)

                        logger.debug(
                            "[relay] Published kind=%s seq=%d to owner=%s on turn:%s",
                            kind,
                            seq,
                            owner_id[:8],
                            turn_id[:8],
                        )
        except RelayTurnCapExceeded as exc:
            # alive but endless — same teardown discipline, distinct
            # message so the user knows the turn was working, not dead.
            logger.warning(
                "[relay] Turn duration cap hit for turn %s (%s) — tearing down",
                turn_id[:8],
                exc,
            )
            await _stop_microvm_before_claim_release(
                turn_id, invoke_session_id, access_token,
                reason=f"turn duration cap ({exc})",
            )
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                "The turn ran longer than the maximum allowed duration and was "
                "ended. Progress made so far is saved — retry to continue.",
                "relay.turn_cap_exceeded",
            )
            return
        except RelayIdleTimeout as exc:
            # the connection is alive but the session stopped producing
            # frames — a wedged turn (a stalled subagent, for example). Same
            # teardown discipline as a transport error: stop the microVM
            # BEFORE the claim frees, then tell the user the truth.
            logger.warning(
                "[relay] Frame-idle limit hit for turn %s (%s) — tearing down wedged session",
                turn_id[:8],
                exc,
            )
            await _stop_microvm_before_claim_release(
                turn_id, invoke_session_id, access_token,
                reason=f"frame-idle limit ({exc})",
            )
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                "The agent stopped responding mid-turn, so the turn was ended. "
                "Your project state is intact — please retry the request.",
                "relay.frame_idle_timeout",
            )
            return
        except httpx.HTTPError as exc:
            # Covers the socket read timeout (httpx.ReadTimeout is an HTTPError),
            # which fires only when no bytes arrive for BEDROCK_READ_TIMEOUT_S.
            logger.warning(
                "[relay] HTTP error invoking AgentCore for turn %s: %s",
                turn_id[:8],
                exc,
            )
            # stop the microVM BEFORE the finally releases the claim, or the
            # user's retry races the still-running first turn.
            await _stop_microvm_before_claim_release(
                turn_id, invoke_session_id, access_token,
                reason=f"transport error/timeout ({type(exc).__name__})",
            )
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                f"AgentCore connection failed: {exc}",
                "relay.http_error",
            )
            return
        except Exception as exc:
            logger.exception(
                "[relay] Unexpected error invoking AgentCore for turn %s",
                turn_id[:8],
            )
            # same ordering as the transport-error path above.
            await _stop_microvm_before_claim_release(
                turn_id, invoke_session_id, access_token,
                reason=f"unexpected relay error ({type(exc).__name__})",
            )
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                f"Unexpected relay error: {exc}",
                "relay.unexpected_error",
            )
            return

        # the drain ended. A clean EOF without a terminal `result` frame is
        # NOT success — no terminal envelope reached the browser, so the UI spins
        # forever while the claim frees below. Publish the terminal error and
        # stop the microVM (it may still be alive and mid-write) before releasing.
        if not saw_result:
            logger.warning(
                "[relay] Turn %s stream ended WITHOUT a terminal result frame — "
                "synthesizing a terminal error so the UI does not hang",
                turn_id[:8],
            )
            await _stop_microvm_before_claim_release(
                turn_id, invoke_session_id, access_token,
                reason="stream ended without a result frame",
            )
            await _publish_terminal_error(
                turn_id,
                project_id,
                owner_id,
                "The agent turn ended without returning a result. Please retry.",
                "relay.no_result_frame",
            )
            return

        # single-writer: the COORDINATOR appends the finished turn to the
        # durable transcript. The microVM used to do this at its own turn end,
        # making the transcript a two-process write target: the coordinator
        # (provisional start-writes, stop markers) and the microVM (turn-end
        # re-map) each serialize behind their OWN process-local lock, and the
        # re-map's atomic whole-file rename could silently drop lines the
        # other process appended after its read (a stopped turn's bubble, or
        # the next user message). agent_runner skips its append inside the
        # AgentCore runtime; this call is the one
        # writer for relay turns, in the same process as every other
        # transcript write, behind the same per-project lock. Reads the raw
        # SDK session from the session store — the identical machinery the
        # GET /chat/history backfill already runs coordinator-side.
        #
        # ORDER MATTERS: this append runs BEFORE the terminal mirror
        # below. The SPA's truncated-stream recovery refetches the transcript
        # when the mirror arrives for a still-active turn — the mirror is
        # therefore a durable-completeness signal, and publishing it before
        # the append would hand the recovery a transcript missing the very
        # turn it's recovering — the mirror would clear the spinner while the
        # bubble stayed mid-sentence until a manual refresh.
        if result_session_id:
            try:
                from app.services.agent_runner import _append_turn_to_transcript

                await _append_turn_to_transcript(
                    project_id, result_session_id,
                    final_answer=result_final_answer,
                    turn_id=turn_id,
                )
            except Exception:
                # Best-effort, same discipline as the microVM-side call: a
                # transcript failure must never fail the relay. The read-path
                # backfill self-heals on the next history load.
                logger.warning(
                    "[relay] Coordinator-side transcript append failed for turn %s",
                    turn_id[:8],
                    exc_info=True,
                )
        # Durable failure marker for relay turns. agent_runner's own marker is
        # skipped inside the AgentCore runtime (single-writer: the transcript is
        # a one-process write target), so the coordinator writes it here —
        # immediately after the transcript append above, which is what puts the
        # explanation AFTER any partial reply rather than before it.
        #
        # Only ``error`` qualifies. A ``cancelled`` subtype already has its own
        # "stopped" marker from the cancel endpoint, and a second contradicting
        # marker is worse than none. The wording is deliberately generic: unlike
        # the in-process path, the CLI's own error text lives inside the microVM
        # and reaches the coordinator as a published event, not as a value in
        # this loop — so this says what is known and nothing more.
        if result_subtype == "error":
            try:
                from app.services.chat_transcript import (
                    append_messages,
                    build_turn_marker_message,
                )

                await append_messages(project_id, [
                    build_turn_marker_message(
                        turn_id,
                        "Turn failed. See the error notification for details, "
                        "then retry.",
                        kind="failed",
                    )
                ])
            except Exception:  # noqa: BLE001 — best-effort, never fail the relay
                logger.warning(
                    "[relay] Failed to persist the turn-failure marker for turn %s",
                    turn_id[:8],
                    exc_info=True,
                )

        if not result_session_id:
            logger.warning(
                "[relay] Result frame for turn %s carried no session_id — "
                "transcript append skipped (read-path backfill will recover)",
                turn_id[:8],
            )

        # Terminal mirror: republish agent_runner's ``agent_turn/<terminal>``
        # project event from the coordinator side. The microVM's own terminal
        # publishes fan out to zero browser connections (its ws_hub holds
        # none), so relayed turns never delivered the turn-end signals — SPA
        # panels that refetch on agent_turn/completed (harness config, run
        # lists) stayed stale until remount, and the chat surface's
        # stranded-turn safety net (which clears the spinner on ANY terminal
        # project event) was dead for cloud chat. Published AFTER the
        # transcript append above so the mirror doubles as a durable-
        # completeness signal for the SPA's truncated-stream recovery. The
        # type mirrors the result frame's subtype — an errored or cancelled
        # turn must not be announced as completed. Best-effort like the error
        # mirror above.
        terminal_type = {
            "error": "error",
            "cancelled": "cancelled",
        }.get(result_subtype or "success", "completed")
        try:
            await hub.publish_project(
                project_id,
                owner_id,
                {
                    "resource": "agent_turn",
                    "type": terminal_type,
                    "turn_id": turn_id,
                },
            )
        except Exception:
            logger.warning(
                "[relay] Failed to publish agent_turn/%s for turn %s (non-fatal)",
                terminal_type,
                turn_id[:8],
            )

        # Durably record the resume mapping for next-turn warm affinity +
        # transcript resume. Recording the mapping does NOT forge a
        # claim (put_agentcore_session leaves status untouched); the claim release
        # below flips running->idle.
        try:
            await record_session(
                owner_id,
                project_id,
                runtime_session_id=invoke_session_id,
                app_session_id=session_id,
            )
        except Exception:
            # Recording the mapping is best-effort — a failure here must not crash
            # the relay task or skip the claim release (finally still runs).
            logger.warning(
                "[relay] Failed to record durable session for turn %s (non-fatal)",
                turn_id[:8],
            )

        # Durable-write summary on the same line an operator already greps for.
        # Emitted even when count==0, because "no durable appends happened" is
        # itself the interesting case for a turn that produced events.
        _n = int(_append_stats["count"])
        logger.info(
            "[relay] AgentCore turn %s relay complete — durable appends=%d "
            "mean=%.0fms max=%.0fms slow=%d failed=%d (budget %.0fms)",
            turn_id[:8], _n,
            (_append_stats["total_ms"] / _n) if _n else 0.0,
            _append_stats["max_ms"],
            int(_append_stats["slow"]),
            int(_append_stats["failed"]),
            _DURABLE_APPEND_SLOW_MS,
        )
    finally:
        # Release the (user, project) claim on EVERY exit so the project is never
        # permanently wedged. Best-effort: release_turn never raises.
        #
        # Fenced on this turn: reaching here does not prove the claim is still
        # ours. A turn whose lease expired, or one whose stop was issued while it
        # kept running, arrives here after the next turn has already claimed the
        # project — and an unfenced release would clear that live claim.
        await release_turn(owner_id, project_id, turn_id=turn_id)
