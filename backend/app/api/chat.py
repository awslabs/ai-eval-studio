# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat API endpoints.

Agent turns run as background tasks, decoupled from HTTP connections.
POST /api/chat spawns a turn and returns a turn_id. Live envelopes for
the turn flow over the project owner's WebSocket connection (see
``app.api.ws``). The chat history endpoint serves the durable per-project
transcript.
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.schemas import ChatRequest
from app.platform import get_platform
from app.platform.api_authz import require_project_read
from app.platform.types import UserId, current_user_id
from app.services.chat_message_parser import map_session_messages
from app.services.coordinator_prompt import build_coordinator_prompt
from app.services.agent_runner import start_agent_turn
from app.services.agent_turn import get_agent_turn_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

def _schedule_provisional_bubble(project_id: str, message: str, turn_id: str) -> None:
    """Fire-and-forget persist of the user's message at turn start.

    In-process counterpart of the relay path's provisional write: makes the
    user's message survive a mid-turn refresh instead of appearing only when
    the turn completes. Scheduled as a task (callers include sync _build
    closures); failure is non-fatal — the completion flush persists the real
    message anyway, and ``append_messages`` upgrades/dedupes on id.
    """
    import asyncio

    async def _write() -> None:
        try:
            from app.services.chat_transcript import (
                append_messages,
                build_provisional_user_message,
            )
            await append_messages(
                project_id, [build_provisional_user_message(message, turn_id)]
            )
        except Exception as exc:  # noqa: BLE001 — best-effort seed
            logger.warning(
                "[chat] provisional bubble write failed for turn %s: %s",
                turn_id[:8], exc,
            )

    try:
        asyncio.get_running_loop().create_task(
            _write(), name=f"provisional-bubble-{turn_id[:8]}"
        )
    except RuntimeError:
        # No running loop (shouldn't happen under FastAPI) — skip the seed.
        logger.warning(
            "[chat] no event loop for provisional bubble (turn %s)", turn_id[:8]
        )


# turn_id-keyed relay-task structure (replaces bare set for addressability)
# Each entry: {turn_id: {"task": Task, "runtime_session_id": str, "project_id": str, "owner_id": str}}
_relay_tasks: dict[str, dict] = {}


async def _stop_agentcore_runtime_session(
    runtime_session_id: str,
    agent_runtime_arn: str,
    access_token: str,
) -> None:
    """Stop an AgentCore runtime session (relay cancellation).

    Calls StopRuntimeSession via unsigned botocore + OAuth bearer to halt the
    microVM compute. Without this, the microVM keeps writing the transcript
    after cancel — a post-cancel writer racing the next turn.

    Best-effort: logs and returns (does not raise) on failure.

    Args:
        runtime_session_id: AgentCore session ID to stop.
        agent_runtime_arn: AgentCore runtime ARN.
        access_token: Cognito access token (raw Authorization header value).
    """
    import os

    import boto3
    import botocore.config

    try:
        # OAuth bearer approach (unsigned botocore + before-send header injection)
        # for AgentCore runtimes with customJWTAuthorizer.
        # Resolve the region from the environment (Platform has NO .config
        # attribute — get_platform().config raised AttributeError, silently
        # swallowed by the except below, so StopRuntimeSession never ran and
        # Stop was a live no-op). Mirror invoke_agentcore_turn's resolution.
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client(
            "bedrock-agentcore",
            region_name=region,
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )

        # Strip 'Bearer ' prefix if present
        token = access_token
        if token and token.lower().startswith("bearer "):
            token = token[7:].strip()

        # Inject Authorization header
        def inject_bearer_token(request, **kwargs):
            request.headers["Authorization"] = f"Bearer {token}"

        client.meta.events.register("before-send", inject_bearer_token)

        # Call StopRuntimeSession
        client.stop_runtime_session(
            agentRuntimeArn=agent_runtime_arn,
            runtimeSessionId=runtime_session_id,
        )

        logger.info(
            "[relay-cancel] StopRuntimeSession called for session %s",
            runtime_session_id[:8],
        )

    except Exception as e:
        # Best-effort: log and continue (claim release must still happen)
        logger.warning(
            "[relay-cancel] StopRuntimeSession failed for session %s: %s",
            runtime_session_id[:8] if runtime_session_id else "unknown",
            e,
        )


@router.get("/projects/{project_id}/chat/history")
async def get_chat_history(
    project_id: str,
    _user: "UserId" = Depends(require_project_read),
):
    """Load chat history from the durable per-project transcript.

    Serves the durable transcript (SDK session is used only for one-time backfill
    when transcript.jsonl is absent but session_id exists). The SDK auto-compacts
    older turns; the transcript accumulates everything, so the UI never loses history.

    Returns ``{"messages": [...], "interrupted": bool}`` so the frontend can
    detect when a page refresh happened mid-turn and show a resume banner.
    A session is considered interrupted when the last visible message is from
    the user (assistant never replied).

    Args:
        project_id: UUID of the project whose chat history to retrieve.

    Returns:
        dict: ``messages`` list and ``interrupted`` flag.
    """
    from app.main import get_project_manager
    from app.services.chat_transcript import append_messages, load_transcript, is_current_schema, discard_transcript

    pm = get_project_manager()
    project = pm.get_project(project_id)
    if not project:
        return {"messages": [], "interrupted": False}

    # Serve from the durable transcript. Do NOT gate this on session_id: a
    # fresh project has no session until its FIRST turn completes, but its
    # transcript already holds the provisional user bubble written at turn
    # start — returning early on a missing session_id would throw that away,
    # and a refresh during the first turn would render an empty chat (the
    # relay path persists the session id at turn START, so only the in-process
    # path is exposed). session_id is only required by the backfill branch
    # below.
    transcript = load_transcript(project_id)

    # STALE-SCHEMA CHECK: If the transcript is non-empty but lacks the current
    # `parts` schema, discard it (cache invalidation) and reroute into the
    # backfill branch below so it rebuilds from the SDK session in the current
    # format. No old-format parsing — the transcript is a derived cache, the
    # SDK session is the source of truth.
    if transcript and not is_current_schema(transcript):
        await discard_transcript(project_id)
        transcript = []  # Reroute to backfill

    # BACKFILL: If no transcript.jsonl yet AND session_id exists, seed it once
    # from the SDK session's FULL raw pre-compaction history (best-effort).
    # The raw-entry source bypasses the windowed accessor to capture the
    # entire lineage — projects that auto-compacted before the durable-transcript
    # feature landed now recover their full history instead of only the tail.
    # This lazy runtime seed is NOT a migration script — it is on-demand.
    if not transcript and project.session_id:
        try:
            from app.platform.chat_history import load_raw_session_entries
            from app.services.project_paths import resolve_project_dir

            platform = get_platform()
            # Use the same cwd + session_store as agent_runner (load-bearing)
            cwd = str(resolve_project_dir(project_id))
            entries = await load_raw_session_entries(
                project.session_id,
                cwd=cwd,
                session_store=platform.session_store,
            )
            mapped = map_session_messages(entries)
            if mapped:
                await append_messages(project_id, mapped)
                transcript = load_transcript(project_id)  # Re-load to prove round-trip
        except Exception as exc:
            # Backfill failure is not fatal — the transcript will accumulate from here
            logger.warning("Backfill failed for %s: %s", project_id[:12], exc)

    mapped = transcript

    # Detect a genuinely resumable interruption: the last visible message
    # is a real user message that the agent never answered (you typed
    # something, then refreshed before any reply streamed). That is the
    # only case where "Continue where you left off" is unambiguously what
    # the user wants.
    #
    # We deliberately do NOT treat "last assistant turn ended mid-tool-
    # call" as resumable. That shape is indistinguishable from a turn the
    # user *cancelled* on purpose: cancelling an optimization interrupts
    # the orchestrator turn mid-execution and filters out its synthetic
    # message, leaving exactly that trailing shape. Offering "Resume"
    # there would keep proposing to continue work the user had just
    # stopped. Intent — "cancelled" vs
    # "cut short" — isn't recoverable from the persisted transcript shape
    # alone, so we resume only the unambiguous case rather than guess.
    #
    # Running is not interrupted: a provisional user tail during a LIVE
    # in-flight turn (running durable claim / active turn log) reports
    # "running" (interrupted=False), not "interrupted". The relay start-write
    # leaves a provisional user bubble as the tail for the entire in-flight
    # turn, so a mid-turn refresh must not show interrupted=True. Only show
    # interrupted=True when there is NO live turn.
    interrupted = bool(mapped) and mapped[-1]["role"] == "user"

    if interrupted:
        # Check if there's a live in-flight turn (in-process or relay)
        # In-process: check AgentTurnTracker
        try:
            from app.services.agent_turn import get_agent_turn_tracker
            tracker = get_agent_turn_tracker()
            active_turn = tracker.get_active_turn(project_id)
            if active_turn is not None:
                interrupted = False  # Running in-process turn
            elif _user:
                # Relay: check durable session registry.
                # NOTE: get_platform is imported at module scope (top of file).
                # A function-local ``import`` here would make get_platform a
                # local for the whole function, shadowing the module-level name
                # and raising UnboundLocalError on the earlier backfill path.
                store = get_platform().metadata_store
                session = await store.get_agentcore_session(_user, project_id)
                if session and session.get("status") == "running" and session.get("current_turn_id"):
                    interrupted = False  # Running relay turn
        except Exception:
            # If the session check fails (e.g., in tests with mocked infrastructure),
            # fall back to the transcript-only interrupted derivation
            pass

    return {"messages": mapped, "interrupted": interrupted}


@router.get("/projects/{project_id}/chat/active-turn")
async def get_active_turn(
    project_id: str,
    user_id: "UserId" = Depends(require_project_read),
):
    """Return the in-flight agent turn for a project, if any.

    Used by the SPA on chat-page mount to learn the ``turn_id`` for a
    turn that was already running before the page existed. The SPA
    then sets that as its ``activeTurnId`` and the WS client's
    subscribe-with-replay path delivers every buffered envelope on
    ``turn:<turn_id>`` (user message, tool calls, partial assistant
    deltas) followed by live frames — single channel, no race window.

    Checks BOTH AgentTurnTracker (in-process turns) AND the durable
    session registry (relay turns). Relay turns bypass the tracker, so the
    durable session's status + current_turn_id is the source of truth for
    relay.

    Returns:
        ``{"turn_id": str, "status": "running"}`` when a turn is
        active, or ``null`` when idle. Buffered envelopes are NOT in
        the response — they ride the WS replay path.
    """
    # First, check in-process tracker (for non-relay turns)
    tracker = get_agent_turn_tracker()
    active = tracker.get_active_turn(project_id)
    if active is not None:
        return {
            "turn_id": active.turn_id,
            "status": active.status,
        }

    # Check durable session registry for relay turns
    # (relay turns bypass AgentTurnTracker; see the relay branch of ``chat``)
    from app.platform import get_platform
    store = get_platform().metadata_store
    session = await store.get_agentcore_session(user_id, project_id)
    if session and session.get("status") == "running":
        turn_id = session.get("current_turn_id")
        if turn_id:
            return {
                "turn_id": turn_id,
                "status": "running",
            }

    return None


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """Start an agent turn as a background task and return the turn_id.

    The agent runs independently of this HTTP request. The frontend
    receives live envelopes for the turn over the project owner's
    WebSocket connection (``/ws``); the returned ``turn_id`` lets the
    SPA filter envelopes by ``stream="turn:<turn_id>"``.

    If an orchestrator turn is currently running, the user's turn is queued
    and starts when the orchestrator turn finishes. The response includes
    ``"status": "queued"`` so the frontend can show a queued indicator.

    **Relay gate:** When ``resolve_agentcore_relay`` returns True (cloud
    mode), the turn runs in AgentCore via server-side invoke (relay path)
    instead of in-process. The relay path REPLACES the in-process path (not
    additive) — when it is on, the in-process SDK turn does NOT also run.

    Args:
        req: Chat request containing user message and project context.
        request: FastAPI Request object (auto-injected) for reading headers.

    Returns:
        JSON with turn_id and status: "running", "queued" (in-process
        path, behind another turn), or "busy" (relay path, another turn holds
        the claim; the message is not queued).
    """
    import asyncio
    from uuid import uuid4

    from app.main import get_project_manager
    from app.platform.arch_split import resolve_agentcore_relay
    from app.services.agentcore_relay_client import invoke_agentcore_turn
    from app.services.agentcore_session_registry import claim_turn, release_turn
    from app.services.project_paths import resolve_project_dir

    # Authorize: project_id comes from the request body (not the path), so
    # we can't use the require_project_write Depends helper here. Manual
    # call against the same Auth Protocol keeps the contract identical.
    user_id = current_user_id()
    await get_platform().auth.require_access(user_id, req.project_id, "write")

    # Resolve the relay flag ONCE at entry (resolve-once rule)
    relay_enabled = await resolve_agentcore_relay(req.project_id)

    pm = get_project_manager()
    project = pm.get_project(req.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # The relay path REPLACES the in-process one: when relay_enabled is on,
    # the in-process SDK turn does NOT also run.
    # This if/else ensures only ONE turn fires per message.
    if relay_enabled:
        # Relay path: invoke AgentCore server-side and relay events to /ws.
        # The turn runs in an isolated AgentCore microVM instead of in-process.
        # Dispatch as a BACKGROUND task so POST /chat returns immediately
        # (same mechanism as the in-process branch) — the turn's lifetime is
        # decoupled from the HTTP request and the browser socket.
        # Relay turns deliberately bypass AgentTurnTracker registration
        # (ephemeral, reconnect catches up via chat/history from S3 transcript).
        turn_id = str(uuid4())
        # ``req.session_id`` is an APP session id, never an AgentCore
        # runtimeSessionId. The schema documents it as a fallback for the
        # project's persisted session, and the SPA takes its value from the
        # ``result`` envelope, which carries the SDK's ``message.session_id``.
        # Forwarding it as the runtimeSessionId let a caller name the microVM
        # session its turn joined — including one belonging to another user,
        # since AgentCore does not enforce session-to-user mapping and treats
        # the id as opaque. It went unnoticed because an app session id is a
        # 36-char UUID, which clears the 33-character minimum AgentCore sets for
        # that header (see ``invoke_agentcore_turn``).
        #
        # ``resolve_resume`` is now the only source of a runtime session id. It
        # reads the (user, project)-keyed registry, so the id can only ever be
        # this caller's, and None here takes the same fresh-session path a
        # project's first turn already takes.
        #
        # Nothing resumable is lost. The registry item lives 2h
        # (AGENTCORE_SESSION_TTL_SECONDS) while an AgentCore session is reaped
        # after 1h idle (idle_runtime_session_timeout), so in the only window
        # where the caller's value reached AgentCore — registry expired — the
        # session it named was already gone. Conversation continuity comes from
        # the app session id, which is what this value is passed through as.
        runtime_session_id = None

        # Claim the (user, project) turn on the durable
        # registry BEFORE spawning the relay task. The DDB conditional-write claim
        # is the LOAD-BEARING one-turn-at-a-time serializer — AgentCore provides no
        # native turn serialization. On a held claim (a turn is already running for
        # this SAME (user, project)), return status=queued WITHOUT invoking
        # AgentCore, so we never fire a parallel invoke. This matches the per-project
        # "one turn at a time" the in-process AgentTurnTracker path enforces; the
        # grain is per-(user, project), so project A does not block project B for the
        # same user. The claim is released on every terminal relay exit (see
        # invoke_agentcore_turn's finally).
        # Pass turn_id so the session records current_turn_id, which is how the
        # active-turn and chat-history endpoints find this relay turn.
        won = await claim_turn(user_id, req.project_id, turn_id=turn_id)
        if not won:
            # Return an HONEST busy rejection, not a misleading "queued"
            # status: the relay has no queue, so the message is DROPPED and not
            # retried. Matching the in-process tracker.submit's ordering and
            # persistence would need a real queue.
            logger.warning(
                "[chat] Relay claim busy for (user=%s, project=%s) — rejecting message (NOT queued)",
                user_id[:8],
                req.project_id[:8],
            )
            return {"turn_id": turn_id, "status": "busy"}

        # Forward the caller's Authorization header verbatim to the relay.
        # The relay client normalizes the Bearer prefix and handles the None case.
        auth_header = request.headers.get("Authorization")

        # wrap claim→create_task window — if create_task fails after a
        # successful claim, release the claim so we don't leak a 'running' claim.
        try:
            # retain the task so it isn't GC'd mid-flight.
            # The relay task owns the claim release (finally in invoke_agentcore_turn),
            # so user_id + project_id are threaded through as owner_id + project_id.
            task = asyncio.create_task(
                invoke_agentcore_turn(
                    project_id=req.project_id,
                    message=req.message,
                    owner_id=user_id,
                    runtime_session_id=runtime_session_id,
                    turn_id=turn_id,
                    access_token=auth_header,
                    app_session_fallback=req.session_id,
                ),
                name=f"relay-turn-{turn_id[:8]}",
            )
            # register relay turn in turn_id-keyed structure for addressability.
            # Store the caller's Authorization header so a later stop_turn can
            # authorize StopRuntimeSession — without it the microVM keeps running
            # and writing the transcript after cancel. Best-effort: the
            # token may be expired by cancel time; the stop path handles that.
            _relay_tasks[turn_id] = {
                "task": task,
                "runtime_session_id": runtime_session_id,
                "project_id": req.project_id,
                "owner_id": user_id,
                "access_token": auth_header,
            }
            task.add_done_callback(lambda _: _relay_tasks.pop(turn_id, None))
        except Exception:
            # Scheduling failed — release the claim before re-raising. Fenced on
            # this turn: nothing ran yet, so the claim should still be ours, and
            # if it somehow is not, it belongs to a turn that must keep it.
            await release_turn(user_id, req.project_id, turn_id=turn_id)
            raise

        return {"turn_id": turn_id, "status": "running"}

    # In-process path: run the turn in this process rather than the relay.
    tracker = get_agent_turn_tracker()
    active = tracker.get_active_turn(req.project_id)

    # Resolve the project's AUTHORITATIVE optimization state here, in async scope,
    # and pass it into build_coordinator_prompt (which is sync and must stay sync —
    # it is called from sync _build closures on this and three other paths).
    # Reading it inside that sync function could only be the EFS read, which is the
    # wrong home on the durable path and made the agent report stale optimization
    # status to the user.
    from app.services import optimization_state_repository as opt_repo

    _turn_project_dir = resolve_project_dir(req.project_id)
    turn_opt_state = await opt_repo.read_for_display(
        req.project_id, _turn_project_dir
    )

    if active is not None:
        # A turn is already running (user or orchestrator) — queue this one.
        # User turns get priority 0 so they preempt queued orchestrator turns.
        session_id = project.session_id or req.session_id

        def _build(reserved_turn_id: str):
            from app.main import get_project_manager as _pm
            _project = _pm().get_project(req.project_id)
            _session = _project.session_id or req.session_id if _project else session_id
            if _project is None:
                _prompt = ""
            else:
                _project_dir = resolve_project_dir(req.project_id)
                _prompt = build_coordinator_prompt(
                    _project, _project_dir, turn_opt_state
                )
            # Runs under the id already returned to the browser. Minting a new one
            # here left the client subscribed to a stream that never carried an
            # event, with Stop returning 404 and the real turn's terminal event
            # rejected by the client's id guard — a spinner that never cleared.
            _turn = start_agent_turn(
                project_id=req.project_id,
                prompt=req.message,
                coordinator_prompt=_prompt,
                session_id=_session,
                model=_project.agent_model if _project else None,
                turn_id=reserved_turn_id,
            )
            # Persist the provisional user bubble the moment the turn STARTS
            # (not at submit — the prior turn's completion flush must land
            # first or the transcript misorders). See the immediate-start
            # branch below for why this write exists on the in-process path.
            _schedule_provisional_bubble(req.project_id, req.message, _turn.turn_id)
            return _turn

        turn_id = tracker.submit(
            project_id=req.project_id,
            origin="user",
            build_turn=_build,
        )

        return {"turn_id": turn_id, "status": "queued"}

    # No turn active — start immediately
    session_id = project.session_id or req.session_id
    project_dir = resolve_project_dir(req.project_id)
    coordinator_prompt = build_coordinator_prompt(project, project_dir, turn_opt_state)

    turn = start_agent_turn(
        project_id=req.project_id,
        prompt=req.message,
        coordinator_prompt=coordinator_prompt,
        session_id=session_id,
        model=project.agent_model,
    )

    # Persist the provisional user bubble immediately, as the relay (cloud)
    # path does. Writing the transcript only at turn COMPLETION would make a
    # mid-turn refresh on a project whose transcript hadn't been seeded yet
    # render an empty chat, with the user's message missing until the turn
    # finished. The
    # completed turn's append upgrades the provisional bubble to the real
    # SDK-id'd message via the tail-upgrade branch — no duplicate.
    _schedule_provisional_bubble(req.project_id, req.message, turn.turn_id)

    return {"turn_id": turn.turn_id, "status": "running"}


@router.post("/chat/stop/{turn_id}")
async def stop_turn(turn_id: str):
    """Stop a running agent turn by interrupting the SDK client.

    Args:
        turn_id: Agent turn UUID to stop.

    Returns:
        JSON confirmation or 404 if turn not found.
    """
    from app.services.agentcore_session_registry import release_turn as release_claim
    from app.services.ws_envelope import make_envelope
    from app.services.ws_hub import get_ws_hub

    # Relay cancellation: check whether this is a relay turn first (addressable
    # by turn_id)
    if turn_id in _relay_tasks:
        relay_entry = _relay_tasks[turn_id]
        task = relay_entry["task"]
        runtime_session_id = relay_entry["runtime_session_id"]
        project_id = relay_entry["project_id"]
        owner_id = relay_entry["owner_id"]
        access_token = relay_entry.get("access_token")

        # Entry log: cancel was routed to the relay branch for a tracked turn.
        # Every skip below logs its reason, so a live cancel that fails to stop
        # the microVM is self-diagnosing instead of a silent no-op.
        logger.info(
            "[relay-cancel] stop_turn entered for relay turn %s (project=%s, "
            "has_runtime_session=%s, has_token=%s)",
            turn_id[:8], project_id[:8],
            bool(runtime_session_id), bool(access_token),
        )

        # Authorize: stopping a relay turn is a write operation
        user_id = current_user_id()
        await get_platform().auth.require_access(user_id, project_id, "write")

        # ORDERING (claim-release-after-stop): issue the microVM stop BEFORE
        # cancelling the drain task. The drain task's finally releases the
        # per-(user,project) DDB claim; if we cancel it first, the claim frees
        # while the microVM is still running, admitting a concurrent turn (and
        # with it several turns editing the same files at once). So:
        # stop the microVM first, THEN cancel the drain (which releases the claim).
        import os
        agent_runtime_arn = os.environ.get("AGENTCORE_RUNTIME_ARN")
        stop_issued = False
        if not runtime_session_id:
            logger.warning(
                "[relay-cancel] turn %s has no runtime_session_id — cannot call "
                "StopRuntimeSession (first-turn resume handle was None). The microVM "
                "may run until its own read-timeout.",
                turn_id[:8],
            )
        elif not agent_runtime_arn:
            logger.warning(
                "[relay-cancel] AGENTCORE_RUNTIME_ARN not set — cannot call "
                "StopRuntimeSession for turn %s. The microVM may run until its own "
                "read-timeout.",
                turn_id[:8],
            )
        elif not access_token:
            # No stored token (or it was never captured) — StopRuntimeSession
            # would fail auth. Do NOT issue a call guaranteed to 401; record
            # that the microVM was not stopped so the gap is visible.
            #
            # The rule below matches on the words "access token" in the message;
            # the only interpolated value is a truncated turn id.
            # nosemgrep: python-logger-credential-disclosure
            logger.warning(
                "[relay-cancel] No stored access token for turn %s — skipping "
                "StopRuntimeSession; the microVM may keep running until its own "
                "read-timeout. Local drain cancelled regardless.",
                turn_id[:8],
            )
        else:
            # Use the caller's Authorization header captured at spawn so the
            # OAuth-runtime StopRuntimeSession is actually authorized.
            try:
                await _stop_agentcore_runtime_session(
                    runtime_session_id,
                    agent_runtime_arn,
                    access_token,
                )
                stop_issued = True
            except Exception as e:
                # Best-effort: log and continue if StopRuntimeSession fails
                logger.warning(
                    "[relay-cancel] StopRuntimeSession call failed for turn %s: %s",
                    turn_id[:8], e,
                )

        logger.info(
            "[relay-cancel] microVM stop %s for turn %s; now cancelling drain task "
            "(releases the claim)",
            "issued" if stop_issued else "NOT issued", turn_id[:8],
        )

        # Cancel the drain task LAST — its finally releases the DDB claim. Doing
        # this after the stop means the claim only frees once the microVM stop
        # has been issued, so a concurrent turn can't be admitted mid-cancel.
        if not task.done():
            task.cancel()

        # Emit a cancelled ``result`` envelope so the client settles the turn
        hub = get_ws_hub()
        envelope = make_envelope(
            stream=f"turn:{turn_id}",
            kind="result",
            data={
                "subtype": "cancelled",
                "result": None,
                "error": "Turn cancelled by user",
            },
        )
        await hub.publish_to_user(owner_id, envelope)

        # Durably append the cancelled envelope so reconnect sees it
        store = get_platform().metadata_store
        try:
            seq = await store.allocate_turn_seq(turn_id)
            durable_event = {
                "stream": f"turn:{turn_id}",
                "seq": seq,
                "kind": "result",
                "data": {
                    "subtype": "cancelled",
                    "result": None,
                    "error": "Turn cancelled by user",
                },
            }
            # TTL is an absolute epoch expiry, not a raw duration — DynamoDB
            # reaps the item once wall-clock passes it.
            await store.append_turn_event(turn_id, seq, durable_event, ttl=int(time.time()) + 86400)
        except Exception as e:
            logger.warning("[relay-cancel] Failed to append cancelled event: %s", e)

        # Append an honest stop marker to the durable transcript. The killed
        # microVM never persists its partial reply, so without this the
        # transcript tail is the bare user message — exactly the shape the
        # GET /chat/history interruption heuristic treats as "cut short",
        # which would make every deliberate Stop show a phantom "Session
        # interrupted — Resume" banner after refresh. The marker records what
        # actually happened AND changes the tail role so the heuristic
        # correctly stays quiet. Id is
        # deterministic per turn — append_messages dedupes on replay.
        try:
            from app.services.chat_transcript import (
                append_messages,
                build_turn_marker_message,
            )

            await append_messages(project_id, [
                build_turn_marker_message(
                    turn_id, "Turn stopped by user.", kind="stopped"
                )
            ])
        except Exception as e:  # noqa: BLE001 — marker is best-effort
            logger.warning(
                "[relay-cancel] Failed to append stop marker for turn %s: %s",
                turn_id[:8], e,
            )

        # Cleanup the relay task entry
        _relay_tasks.pop(turn_id, None)

        return {"status": "cancelled"}

    # In-process turn handling
    tracker = get_agent_turn_tracker()
    turn = tracker.get_turn(turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found")

    # Authorize: stopping a turn is a write operation against the project.
    user_id = current_user_id()
    await get_platform().auth.require_access(user_id, turn.project_id, "write")

    if turn.status != "running":
        return {"status": turn.status, "message": "Turn already finished"}

    # Mark the turn as failed immediately so the frontend doesn't see it
    # as active on reconnect. The SDK subprocess may still be running
    # (tool execution continues until it finishes), but the turn is done
    # from the user's perspective.
    turn.status = "failed"
    # Flag this as user cancellation — the agent_runner's exception
    # handler checks this to publish a "cancelled" event instead of
    # "error", so the frontend won't show a red toast.
    turn._stopped_by_user = True

    # Notify the turn queue so the next queued turn can start immediately
    # rather than waiting for the underlying task to exit via the finally block.
    tracker.notify_turn_complete(turn.project_id)

    # Best-effort stop of the SDK client — stop_task kills the running tool,
    # interrupt signals the agent loop. Neither is guaranteed to work immediately
    # if a tool subprocess is blocked.
    client = turn._client
    if client:
        try:
            await client.stop_task()
        except Exception:
            try:
                await client.interrupt()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Stop/interrupt failed: %s", e)

    return {"status": "stopped"}


