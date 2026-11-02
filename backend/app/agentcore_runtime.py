# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore serving ASGI app: /ping + /invocations SSE.

Standalone FastAPI app (separate from app.main:app) that answers AgentCore's
serving contract. The ARM64 image selects this via a distinct
container command.

Routes:
- GET /ping: Fast health check returning {"status":"Healthy"} with no SDK work
- POST /invocations: Accepts {message, project_id}, runs ONE coordinator turn
  via the EXISTING run_agent_turn / start_agent_turn path, and streams the
  AgentTurn event buffer as SSE (text/event-stream)

There is deliberately no CORS handling here. Every caller is
``agentcore_relay_client``, which reaches the runtime with a SigV4-signed
server-side POST to ``bedrock-agentcore.<region>.amazonaws.com``; CORS is a
browser-enforced mechanism and has no effect on that path. The browser once
called ``/invocations`` directly, and that path was removed because its cancel
could only abort the browser's reader while the microVM kept running.

The adapter REUSES the turn engine (start_agent_turn/run_agent_turn) and DRAINS
the AgentTurn buffer (wait_for_events/.events) — it does NOT reimplement the
SDK loop, so every execution home runs the same turn engine.

Two behaviours here are verified only against a deployed runtime, never by
in-repo tests, because they depend on how AgentCore itself forwards traffic:
- the exact SSE event:/data: framing, and how AgentCore forwards
  text/event-stream through to the relay client
- runtimeSessionId delivery, which reads the header AgentCore injects and
  falls back to the payload
"""
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.platform import get_platform
from app.platform.types import RequestContextValue, request_context
from app.services.agent_runner import start_agent_turn
from app.services.background_work import has_background_work_in_flight
from app.services.agentcore_session import record_session_mapping, resume_session_id_for
from app.services.coordinator_prompt import build_coordinator_prompt
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)

# cgroup / proc paths for the memory self-report. Module-level so tests can
# monkeypatch them without touching the reader logic.
_CGROUP_V2_PEAK = "/sys/fs/cgroup/memory.peak"
_CGROUP_V2_CURRENT = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_PEAK = "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"
_CGROUP_V1_CURRENT = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_PROC_STATUS = "/proc/self/status"


def _read_int_file(path: str) -> int | None:
    """Read a single integer from a sysfs/procfs file, or None if unavailable.

    Args:
        path: Absolute path to a file whose entire content is one integer.

    Returns:
        The parsed integer, or None if the file is missing/unreadable/non-numeric.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_vmhwm_bytes() -> int | None:
    """Read peak resident set size (VmHWM) of THIS process from /proc/self/status.

    This is the last-resort fallback: it only covers the Python process, NOT the
    spawned ``claude`` CLI subprocess, so it undercounts a real turn. Prefer the
    cgroup high-watermark, which spans the whole container.

    Returns:
        VmHWM in bytes, or None if unavailable.
    """
    try:
        with open(_PROC_STATUS, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    # Format: "VmHWM:\t   12345 kB"
                    kb = int(line.split()[1])
                    return kb * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _memory_report() -> dict[str, int | None]:
    """Report peak + current memory for the whole container (cgroup-first).

    AgentCore caps the runtime at 2 vCPU / 8 GB, and a turn spawns the ``claude``
    CLI as a subprocess, so a per-process RSS reading undercounts. Read the
    CGROUP high-watermark, which counts Python + the CLI + any grandchildren:
    cgroup v2 ``memory.peak`` (monotonic peak since container start), falling
    back to cgroup v1 ``memory.max_usage_in_bytes``, then ``/proc/self/status``
    VmHWM as a last resort. ``vmhwm_bytes`` is always the per-process figure so
    the operator can see the gap between the process and the whole cgroup.

    Returns:
        Dict with cgroup_peak_bytes, cgroup_current_bytes, vmhwm_bytes; any
        field is None when that source is unavailable on the host.
    """
    cgroup_peak = _read_int_file(_CGROUP_V2_PEAK)
    cgroup_current = _read_int_file(_CGROUP_V2_CURRENT)
    if cgroup_peak is None:
        # cgroup v1 fallback
        cgroup_peak = _read_int_file(_CGROUP_V1_PEAK)
    if cgroup_current is None:
        cgroup_current = _read_int_file(_CGROUP_V1_CURRENT)
    vmhwm = _read_vmhwm_bytes()
    # Last-resort peak: if neither cgroup source is present, surface VmHWM as the
    # peak so the report is never entirely empty on a non-cgroup host.
    if cgroup_peak is None:
        cgroup_peak = vmhwm
    return {
        "cgroup_peak_bytes": cgroup_peak,
        "cgroup_current_bytes": cgroup_current,
        "vmhwm_bytes": vmhwm,
    }


# Standalone app (not app.main:app)
#
# No CORS middleware: the sole caller signs its POST with SigV4 server-side, so
# there is no browser preflight to answer and an allow-origin header would only
# widen what a browser is permitted to do with a response it cannot currently
# read at all. See the module docstring for the path that was removed.
app = FastAPI(title="AgentCore Runtime")


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check for AgentCore's readiness probe, reporting busy vs idle.

    Answers immediately without importing or starting the SDK, so the readiness
    probe never trips a restart loop, and without writing any per-ping timestamp.

    The status is load-bearing for session lifetime, not just health. AgentCore
    counts a ``Healthy`` answer against the idle timer and terminates the session
    once the configured idle window elapses; ``HealthyBusy`` keeps it alive. This
    runtime spawns background work that outlives the turn that created it — the
    optimization loop above all — so answering ``Healthy`` unconditionally got
    that work killed mid-flight, leaving jobs stranded non-terminal.

    What the busy answer buys: while a recognized background task is still
    running, the session is not counted against the idle timeout.

    What it does NOT buy: it does not extend the runtime's configured
    ``max_lifetime`` ceiling (4 hours on this deploy). Work that outruns that
    ceiling is still terminated, so a long loop needs an execution home that is
    not tied to a session at all.

    Returns:
        Dict with a ``status`` of ``"HealthyBusy"`` while recognized background
        work is in flight, otherwise ``"Healthy"``.
    """
    if has_background_work_in_flight():
        return {"status": "HealthyBusy"}
    return {"status": "Healthy"}


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    """Handle AgentCore invocation: run one coordinator turn and stream SSE.

    Implements the AgentCore serving contract:
    - Trivial/empty probe body -> fast 200 (readiness contract)
    - Real payload {message, project_id} -> one turn via start_agent_turn,
      streaming the AgentTurn buffer as text/event-stream

    The runtimeSessionId is read from the header AgentCore injects
    (X-Amzn-Bedrock-AgentCore-Runtime-Session-Id) with a payload fallback.

    Args:
        request: FastAPI Request object.

    Returns:
        StreamingResponse with text/event-stream for real payloads,
        or Response(status_code=200) for probe bodies.
    """
    # Parse body
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Metrics-probe path: {"probe":"metrics"} -> plain-JSON memory self-report
    # WITHOUT starting a turn. AgentCore forwards client traffic to /invocations
    # only (a new GET /metrics route likely wouldn't be reachable), so the RSS
    # self-report piggybacks on the probe branch. Kept OFF the SSE/envelope path
    # so handleEnvelope / the SPA are unaffected. The operator calls this right
    # after a large turn to capture peak RSS against the 8 GB cap.
    if body.get("probe") == "metrics":
        return Response(
            content=json.dumps(_memory_report()),
            status_code=200,
            media_type="application/json",
        )

    # Fast-probe path: empty body or explicit probe flag -> 200 without turn
    # (readiness contract — slow default causes restart loop)
    if not body or body.get("probe") is True:
        return Response(status_code=200)

    # Real payload path
    message = body.get("message", "")
    project_id = body.get("project_id")

    if not project_id or not message:
        return Response(
            content=json.dumps({"error": "message and project_id required"}),
            status_code=400,
            media_type="application/json",
        )

    # Read runtimeSessionId from header (AgentCore injects it) with payload fallback
    runtime_session_id = request.headers.get(
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id",
        body.get("runtimeSessionId"),
    )

    # Enforce tenancy — FAIL CLOSED.
    #
    # This standalone app does NOT mount app.main's auth middleware, so
    # current_user_id()/request.state are never populated here. Extract the
    # user directly from the request the same way app.platform.auth's
    # middleware does (Authorization: Bearer <Cognito JWT>), then authorize
    # against the project — mirroring chat.py's manual require_access call
    # (project_id comes from the body, so a path-based Depends can't be used).
    #
    # In production the AgentCore JWT authorizer validates the token
    # at the runtime edge; this in-container check is defense-in-depth and the
    # tenant->project authorization the edge authorizer does NOT perform.
    auth = get_platform().auth
    try:
        user_id = await auth.extract_user_id(request)
    except HTTPException as exc:
        return Response(
            content=json.dumps({"error": exc.detail}),
            status_code=exc.status_code,
            media_type="application/json",
        )
    except Exception:
        logger.exception("Failed to extract user from /invocations request")
        return Response(
            content=json.dumps({"error": "Unauthorized"}),
            status_code=401,
            media_type="application/json",
        )

    try:
        await auth.require_access(user_id, project_id, "write")
    except HTTPException as exc:
        logger.warning("Access denied for user %s project %s", user_id, project_id)
        return Response(
            content=json.dumps({"error": exc.detail}),
            status_code=exc.status_code,
            media_type="application/json",
        )
    except Exception as e:
        logger.warning("Access denied for user %s project %s: %s", user_id, project_id, e)
        return Response(
            content=json.dumps({"error": "Access denied"}),
            status_code=403,
            media_type="application/json",
        )

    # Resolve session_id for resume
    session_id = None
    if runtime_session_id:
        session_id = resume_session_id_for(runtime_session_id, project_id)

    # Populate request_context BEFORE the prompt build and durable refresh
    # (tenancy in the microVM). This standalone app does not mount
    # app.main's auth middleware, so nothing sets the per-request ContextVar here.
    # The in-process SDK MCP eval tools authorize via app.platform.mcp_authz,
    # which reads request_context and returns mcp_error("not authenticated") when
    # it is unset — so without this, EVERY eval tool fails inside the microVM
    # when the relay flag is on (the documented "require_access runs on every MCP
    # tool wrapper" invariant depends on the context being present during
    # tool execution). We already resolved user_id +
    # project_id above; publish them.
    #
    # The context must also be populated for the durable optimization read
    # below: it keys on current_request().user_id and falls back to the dev-user
    # id on RuntimeError, so with the context unset it would read the wrong
    # tenant's partition and the prompt would silently carry no optimization
    # status. Nothing between this set and the prompt build depends on the
    # context being unset.
    #
    # start_agent_turn spawns run_agent_turn via asyncio.create_task, which copies
    # the current context — so the turn task (and every MCP tool it invokes)
    # inherits this snapshot even though the handler resets its own view below.
    # Mirrors app.platform.auth.auth_middleware's set/reset discipline.
    ctx = RequestContextValue(
        user_id=user_id,
        request_id=str(uuid.uuid4()),
        project_id=project_id,
    )
    ctx_token = request_context.set(ctx)

    # Build prompt, with a durable refresh immediately before so the turn header
    # reflects durable job state rather than this microVM's cold tracker.
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        if project is None:
            request_context.reset(ctx_token)
            return Response(
                content=json.dumps({"error": "Project not found"}),
                status_code=404,
                media_type="application/json",
            )
        project_dir = resolve_project_dir(project_id)

        # Refresh durable job state AND durable optimization state before building
        # the prompt so the turn header reports accurate state. This is scoped to
        # the microVM entry point and deliberately NOT added to the coordinator-side
        # call sites (chat.py, completion_turn.py): in the coordinator process, the
        # in-memory Job is the live executing object and optimization writes happen
        # in-process, so cache-as-truth is correct by design. The microVM is the
        # only process whose tracker and optimization view are cold or stale replicas.
        from app.platform.arch_split import resolve_arch_split
        from app.services.jobs import get_job_tracker
        from app.services.optimization_state import OptimizationState

        # Resolve arch_split once and share it between job refresh and opt read
        arch_split = False
        opt_state = None
        try:
            arch_split = await resolve_arch_split(project_id)
        except Exception as e:
            logger.warning(
                "[agentcore_runtime] resolve_arch_split failed for project=%s: %s",
                project_id[:12], e
            )

        # Refresh jobs from durable store
        try:
            await get_job_tracker().rebuild_jobs_from_store(
                project_id, arch_split, refresh_terminal=True
            )
        except Exception as e:
            # Store errors degrade to whatever the in-memory tracker already
            # holds rather than raising into the prompt build. A store hiccup
            # must not fail the turn.
            logger.warning(
                "[agentcore_runtime] rebuild failed for project=%s: %s",
                project_id[:12], e
            )

        # Resolve durable optimization state on the durable path
        if arch_split:
            from app.platform.types import current_request
            try:
                # Resolve user_id (context already set above)
                try:
                    resolved_user_id = current_request().user_id
                except RuntimeError:
                    from app.platform.auth import get_local_dev_user_id
                    resolved_user_id = get_local_dev_user_id()

                store = get_platform().metadata_store
                opt_data = await store.get_optimization_state(resolved_user_id, project_id)
                opt_state = OptimizationState._from_dict(project_dir, opt_data) if opt_data else None
            except Exception as e:
                # Store failure degrades to None: the prompt then carries no
                # optimization status rather than a stale one.
                logger.warning(
                    "[agentcore_runtime] durable optimization read failed for project=%s: %s",
                    project_id[:12], e
                )
                opt_state = None

        coordinator_prompt = build_coordinator_prompt(project, project_dir, opt_state)
    except Exception as e:
        logger.exception("Failed to build coordinator prompt for project %s", project_id)
        request_context.reset(ctx_token)
        return Response(
            content=json.dumps({"error": f"Failed to prepare turn: {e}"}),
            status_code=500,
            media_type="application/json",
        )

    # Start ONE turn via the EXISTING turn engine (do NOT reimplement
    # run_agent_turn). Signature: start_agent_turn(project_id, prompt,
    #   coordinator_prompt, session_id=None, model=None, origin="user",
    #   turn_id=None). The resolved session_id is the resume handle — passing it
    # makes run_agent_turn read the S3 transcript.
    try:
        turn = start_agent_turn(
            project_id,
            message,
            coordinator_prompt,
            session_id=session_id,
        )
    except Exception as e:
        logger.exception("Failed to start agent turn for project %s", project_id)
        request_context.reset(ctx_token)
        return Response(
            content=json.dumps({"error": f"Failed to start turn: {e}"}),
            status_code=500,
            media_type="application/json",
        )
    finally:
        # Reset the handler's own context view. The turn task already captured
        # its own copy at create_task time, so this does not disturb it — it just
        # prevents this ContextVar from leaking into unrelated work on this
        # handler's task after the response is returned.
        request_context.reset(ctx_token)

    # Record the (project, runtimeSessionId) -> app session_id mapping so a
    # later invoke with the same runtimeSessionId resumes the same conversation.
    # Use the RESOLVED resume session_id, not turn.session_id: AgentTurn.session_id
    # is None until the SDK returns a result, so reading it here would never
    # record anything. When there is no resume handle yet (fresh
    # session), there is nothing durable to map — run_agent_turn persists the
    # project's session_id on completion, and the next invoke resolves it via
    # the persisted-session fallback in resume_session_id_for.
    if runtime_session_id and session_id:
        record_session_mapping(runtime_session_id, session_id, project_id)

    # Stream the turn as SSE
    return StreamingResponse(
        _sse_frames(turn),
        media_type="text/event-stream",
    )


async def _sse_frames(turn: Any) -> AsyncGenerator[str, None]:
    """Drain AgentTurn.events buffer and yield SSE frames.

    Mirrors the WS pump seam (_pump_outbound in app/api/ws.py): walks
    AgentTurn.events using wait_for_events, maps each envelope to an SSE
    frame, emits a heartbeat frame on timeout, and terminates after the
    'result' kind or terminal turn status.

    SSE frame format:
        event: <kind>
        data: <json.dumps(envelope.data)>
        <blank line>

    Args:
        turn: The AgentTurn instance (from start_agent_turn).

    Yields:
        SSE frame strings.
    """
    known_count = 0
    keepalive_timeout = 15.0  # seconds

    def _pending_frames() -> tuple[list[str], bool]:
        """Slice unsent envelopes from the buffer into SSE frames.

        Returns (frames, saw_result). Mirrors the WS pump's ordinal bookkeeping:
        events_dropped tracks buffer eviction, so the live slice starts at
        known_count - events_dropped. Advances the closed-over known_count.
        """
        nonlocal known_count
        buffer_start = turn.events_dropped
        start_idx = max(0, known_count - buffer_start)
        frames: list[str] = []
        saw_result = False
        for envelope in turn.events[start_idx:]:
            kind = envelope.get("kind", "unknown")
            data = envelope.get("data", {})
            frames.append(f"event: {kind}\ndata: {json.dumps(data)}\n\n")
            known_count += 1
            if kind == "result":
                saw_result = True
                break
        return frames, saw_result

    while True:
        # Wait for new events or timeout
        has_new = await turn.wait_for_events(known_count, timeout=keepalive_timeout)

        if has_new:
            frames, saw_result = _pending_frames()
            for frame in frames:
                yield frame
            # Terminate after the 'result' kind
            if saw_result:
                logger.debug("SSE drain complete (result kind) for turn %s", turn.turn_id[:8])
                return
        else:
            # Timeout — emit a REAL heartbeat frame, not just an SSE comment.
            # Comments keep the TCP socket alive but the coordinator's
            # frame parser never yields them, so its frame-idle watchdog
            # can't tell "runtime alive, turn quietly working" from
            # "runtime dead behind a connection AgentCore keeps open" — the
            # exact ambiguity that let a stalled-looking-but-alive turn and a
            # genuinely dead session read identically. The heartbeat is
            # emitted only while THIS process is draining a live turn, so
            # its absence is a true death signal. The coordinator counts it
            # as liveness and never forwards it (no seq, no turn log).
            yield "event: heartbeat\ndata: {}\n\n"

        # Check terminal status. Do a FINAL non-blocking drain first so trailing
        # frames aren't dropped if the terminal transition raced the last
        # wait_for_events — e.g. the 'result' envelope landed in the
        # buffer just as status flipped to completed and wait_for_events had
        # already returned on the timeout branch.
        if turn.status in ("completed", "failed"):
            frames, _ = _pending_frames()
            for frame in frames:
                yield frame
            logger.debug("SSE drain complete (status=%s) for turn %s", turn.status, turn.turn_id[:8])
            return
