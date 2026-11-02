# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent turn runner with WebSocket-streamed envelopes.

Executes agent turns as background asyncio.Tasks, decoupled from HTTP
connections. Events are written to an in-memory buffer on the
:class:`AgentTurn` AND fanned out via :class:`WsHub` so any active
WebSocket connection for the turn's owner receives each envelope live.
The buffer is the source of truth for replay-on-reconnect.

Flow:
  POST /api/chat  →  spawns background task  →  returns turn_id
  WebSocket /ws    →  receives envelopes via ws_hub fan-out
"""
import asyncio
import json
import logging
import os
from dataclasses import replace

from botocore.config import Config

# ``HookEventMessage`` is imported at module scope (rather than inside
# ``run_agent_turn`` like the heavier SDK types) because the turn loop
# uses ``isinstance`` against it to discriminate lifecycle events on
# the message stream. Tests patch ``sys.modules["claude_agent_sdk"]``
# with a MagicMock to avoid spawning the real CLI, which would leave
# function-scoped imports of this class resolving to a MagicMock (not
# a real type) and break ``isinstance``. Binding it once at module
# load keeps the type stable regardless of per-test patches.
from claude_agent_sdk import HookEventMessage

from app.agents import SUBAGENTS
from app.services.agent_turn import AgentTurn, get_agent_turn_tracker
from app.services.mcp_server_cache import get_eval_mcp_server
from app.services.resource_error import Codes, ResourceError
from app.services.sdk_logging import (
    assistant_text_preview,
    subprocess_log_line,
    tool_result_preview,
)
from app.services.sdk_subprocess_env import (
    SDK_MAX_STDOUT_FRAME_BYTES,
    sdk_subprocess_env,
)
from app.platform.types import RequestContextValue, request_context
from app.services.task_error_sink import attach_error_sink
from app.services.ws_envelope import make_envelope
from app.services.ws_hub import get_ws_hub

logger = logging.getLogger(__name__)


# Tools available to the coordinator agent (subagents define their own)
COORDINATOR_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "WebSearch", "WebFetch",
]

# Built-ins hard-removed from the model's context via ``disallowed_tools``.
# AskUserQuestion is a Claude Code built-in (available by default — ``tools``
# is unset, and ``allowed_tools`` is an auto-approve list, not an
# availability gate). The SDK can only deliver it via a ``can_use_tool``
# callback we don't set, so a call renders as an unanswerable ✓ chip and
# pauses the turn in-memory (lost on refresh). Disallowing it forces the
# coordinator to ask for decisions in plain chat (reinforced in
# coordinator.md), which reloads with history and survives refresh/crash.
# Subagents inherit this restriction.
DISALLOWED_TOOLS = ["AskUserQuestion"]

# AWS Knowledge MCP — a remote HTTP MCP (no local process spawn). Gives
# the agent live access to AWS's structured docs, including Bedrock
# model cards (use-case, context window, reasoning support, modalities,
# launch date), so model recommendations come from current facts rather
# than the model's stale training memory.
AWS_KNOWLEDGE_MCP_SERVER = "aws_knowledge"
# Bare domain, NO ``/mcp`` suffix. Both URLs pass the initialize
# handshake (``get_mcp_status`` reports "connected" for each), but the
# ``/mcp`` path fails on the actual tool-call POST with
# ``"Http operation is not supported for gateway protocol type MCP"`` —
# reproduced against the real CLI streamable-HTTP transport, not curl
# (a one-shot POST succeeds on both paths, which is why curl is
# misleading; the failure is in the CLI's session/SSE POST flow). The
# bare domain returns real model-card data on the tool call. This is the
# URL every official awslabs client config uses.
AWS_KNOWLEDGE_MCP_URL = "https://knowledge-mcp.global.api.aws"

# Scope tightly: only the two read/search documentation tools, not the
# server's ``recommend`` / ``list_regions`` / ``get_regional_availability``
# / ``retrieve_skill`` tools. The server's own tool names are already
# prefixed ``aws___``; the SDK prefixes again with ``mcp__<server>__``,
# yielding the double-underscore names below. Listed explicitly (not a
# ``mcp__aws_knowledge__*`` wildcard) so the agent can't roam the rest of
# the server's surface.
AWS_KNOWLEDGE_DOC_TOOLS = [
    f"mcp__{AWS_KNOWLEDGE_MCP_SERVER}__aws___search_documentation",
    f"mcp__{AWS_KNOWLEDGE_MCP_SERVER}__aws___read_documentation",
]


def _aws_knowledge_mcp_config() -> dict:
    """Return the remote HTTP MCP server config for the AWS Knowledge MCP.

    Shape verified against the installed ``claude_agent_sdk`` (0.2.85)
    ``McpHttpServerConfig`` TypedDict: ``{type: "http", url, headers?}``.
    No headers required — the AWS Knowledge MCP is unauthenticated.
    """
    return {"type": "http", "url": AWS_KNOWLEDGE_MCP_URL}


# Retained fire-and-forget publish tasks so they aren't garbage-collected
# mid-flight. Same pattern (and same reason) as ``jobs._ws_publish_tasks``:
# asyncio holds only a weak reference to a bare ``create_task`` result.
_turn_publish_tasks: set[asyncio.Task] = set()


def schedule_project_publish(
    project_id: str, owner_id: str, event: dict, *, turn_id: str
) -> None:
    """Publish a project event from the turn path without blocking the turn.

    ``publish_project`` allocates a durable project seq, which is
    store I/O. Awaiting that anywhere on a turn's path was measured to stall the
    turn before its terminal transition, leaving the client waiting forever for a
    ``result`` envelope that never arrives. Every project publish on this path
    therefore goes through here.

    This is a class-level rule, not a patch for one call site: a project event is
    a notification about a turn, so it must never gate the turn it describes. Two
    sites hit the stall in practice (turn-complete and the CLI-error fallback) and
    any future one would too.

    The durable counter remains the single seq issuer, so cross-producer
    monotonicity is unchanged; only the moment of allocation moves off this path.
    Concurrent scheduled publishes may still be *delivered* slightly out of
    order; that window is covered by the replay buffer's cannot-serve signal,
    deliberately not by a wider seq scheme.

    Failures are logged, never raised — a notification must not fail a turn.

    Args:
        project_id: Project whose stream carries the event.
        owner_id: Project owner — the fan-out target.
        event: The project-bus payload (``{"resource": ..., "type": ..., ...}``).
        turn_id: Turn this publish belongs to; used for log correlation only.
    """
    async def _publish() -> None:
        try:
            await get_ws_hub().publish_project(project_id, owner_id, event)
        except Exception as exc:  # noqa: BLE001 — a notification must not fail a turn
            logger.warning(
                "[turn %s] Project publish failed (resource=%s type=%s): %s",
                turn_id[:8],
                event.get("resource"),
                event.get("type"),
                exc,
            )

    try:
        task = asyncio.get_running_loop().create_task(_publish())
    except RuntimeError:
        # No running loop (sync test context). Turn state is already persisted;
        # the SPA refetches on its next load.
        return
    _turn_publish_tasks.add(task)
    task.add_done_callback(_turn_publish_tasks.discard)


# Tools whose successful completion means the project filesystem changed —
# Data/Code/Prompts/Summary tabs all need to refetch. Kept as a closed set
# here (instead of a generic "is_mutation" flag on every tool) because the
# blast radius of a false positive is a UI thrash mid-turn; read-only tools
# (list_upload, describe_data, …) MUST NOT appear here.
_FS_MUTATING_TOOLS: set[str] = {
    "mcp__eval__move_file",
    "mcp__eval__finalize_upload",
    "mcp__eval__scaffold_harness",
    "mcp__eval__validate_harness",
    "mcp__eval__save_prompt",
    "mcp__eval__generate_synthetic",
    "mcp__eval__approve_synthetic_cases",
    "mcp__eval__refresh_pricing",
    "Write",
    "Edit",
}


import os.path as _ospath


# Tools whose resolved target path the jail inspects. ``Bash`` is
# deliberately excluded — a static command-line scan can't reliably
# contain shell escapes (``~``, ``..``, ``$VAR``, ``$(...)``, glued
# redirects ``>>/etc/x``, brace expansion), so a path-scanning jail
# over Bash would assert a guarantee it does not deliver. The path-arg tools
# below are structurally checkable: exactly one user-supplied path
# per call, no shell layer to fight.
_JAILED_TOOLS: frozenset[str] = frozenset({"Write", "Edit", "NotebookEdit"})


# The CLI's delegation tool can arrive under either name (the second comes
# with the CLI's background-by-default mode). ONE constant exists so the
# forced-sync rewrite and the subagent:<type> display rename always cover the
# same name set.
_DELEGATION_TOOLS: frozenset[str] = frozenset({"Task", "Agent"})


# User-facing message when a subagent starts but never stops. The turn
# architecture assumes delegation is synchronous — the coordinator promises
# "I'll report the findings" in the same turn. A backgrounded subagent never
# reports, so the promised work is silently dropped.
_ORPHANED_SUBAGENT_MESSAGE = (
    "The agent delegated work to a subagent that never reported back. "
    "The delegated step did not run. Please send the request again."
)


def _pre_tool_use_jail_decision(
    tool_name: str,
    tool_input: dict | None,
    project_dir: str | os.PathLike,
    tool_cwd: str | None = None,
) -> dict | None:
    """Return a deny payload if a tool's resolved target leaves project_dir.

    Pure function so the test suite can pin the jail logic without
    standing up a full agent turn. Returns ``None`` when the tool is
    out-of-scope for the jail or when every candidate path resolves
    inside ``project_dir``; otherwise returns the
    ``hookSpecificOutput`` payload the SDK PreToolUse hook expects.

    Args:
        tool_name: SDK tool name (Write / Edit / NotebookEdit).
        tool_input: The tool call's argument bag.
        project_dir: Project working directory; the jail's containment
            target. ``realpath``-ed before the prefix check so symlinks
            don't bypass.
        tool_cwd: The cwd the SDK reports for this tool call (from
            ``HookInput["cwd"]``). Used as the base for resolving
            relative paths supplied by the agent. When None, falls
            back to ``project_dir`` — that matches production where
            ``ClaudeAgentOptions.cwd = project_dir`` so relative paths
            ARE inside the tree by construction. The fallback exists
            so ``_pre_tool_use_jail_decision`` can be tested in
            isolation without faking a full HookInput.
    """
    if tool_name not in _JAILED_TOOLS:
        return None
    if not isinstance(tool_input, dict):
        return None

    candidate_paths: list[str] = []
    if tool_name in ("Write", "Edit"):
        fp = tool_input.get("file_path")
        if isinstance(fp, str) and fp:
            candidate_paths.append(fp)
    elif tool_name == "NotebookEdit":
        fp = tool_input.get("notebook_path")
        if isinstance(fp, str) and fp:
            candidate_paths.append(fp)

    if not candidate_paths:
        return None

    project_root = _ospath.realpath(str(project_dir))
    base_for_relative = _ospath.realpath(str(tool_cwd)) if tool_cwd else project_root

    for path_str in candidate_paths:
        # Anchor relative paths against the agent's reported cwd
        # (HookInput["cwd"]), NOT the FastAPI process cwd. The hook
        # runs in-process; the agent CLI subprocess has its own cwd
        # set by ClaudeAgentOptions.cwd. ``os.path.realpath`` would
        # otherwise resolve against the host process cwd and deny
        # every legitimate relative-path Write the agent issues.
        if _ospath.isabs(path_str):
            target = _ospath.realpath(path_str)
        else:
            target = _ospath.realpath(_ospath.join(base_for_relative, path_str))
        if target == project_root:
            continue
        if not target.startswith(project_root + "/"):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Refusing {tool_name}: target {path_str!r} resolves outside "
                        f"the project working directory ({project_root}). "
                        "Tools may only read/write inside project_dir."
                    ),
                }
            }
    return None


def _forced_sync_delegation_input(
    tool_name: str, tool_input: dict | None
) -> dict | None:
    """Return the forced-sync input when a delegation needs rewriting.

    Pure function so the test suite can pin the forced-sync decision
    without standing up a full agent turn. Delegation tools (Task and
    Agent) MUST run synchronously — the turn architecture assumes the
    coordinator's promise "I'll report the findings" happens in the
    SAME turn. A backgrounded subagent never reports, and the promised
    work is silently dropped.

    The CLI ships a remote-flag-gated background-by-default mode where
    an OMITTED ``run_in_background`` means async. Only an explicit
    ``False`` opts out under both the old default (sync) and the new
    one (async). This helper forces ``run_in_background=False`` when
    it is omitted or set to ``True``, and passes through when it is
    already ``False`` (no rewrite churn).

    Args:
        tool_name: SDK tool name.
        tool_input: The tool call's argument bag.

    Returns:
        A new dict with ``run_in_background`` forced to ``False`` when
        the tool is a delegation tool and the flag is not already
        ``False``; otherwise ``None`` (no rewrite needed).
    """
    if tool_name not in _DELEGATION_TOOLS:
        return None
    if not isinstance(tool_input, dict):
        return None
    if tool_input.get("run_in_background") is False:
        return None
    return {**tool_input, "run_in_background": False}


def _resolve_owner_id(project_id: str) -> str | None:
    """Look up the owner_id for ``project_id`` for ws_hub fan-out.

    Every realtime envelope must be addressed to a specific user. The
    coordinator turn's owner is the project owner — whoever created
    the project (cloud) or ``LOCAL_DEV_USER_ID`` (dev). Returns
    ``None`` when the project can't be loaded, in which case the
    publisher falls back to dropping the envelope (no active WS
    consumer can receive it without a user_id, and the SPA refetches
    via REST on its next page load).

    Args:
        project_id: Project UUID.

    Returns:
        The project's owner user_id, or ``None`` if unavailable.
    """
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        if project is None:
            return None
        return project.owner_id
    except Exception:
        # Defensive: never let a lookup failure crash a turn. The
        # coordinator can finish its work; the user just won't see
        # live updates until they reload (REST is unaffected).
        logger.debug("[agent-runner] owner_id resolution failed for %s",
                     project_id, exc_info=True)
        return None


async def _publish_cli_error(
    *,
    project_id: str,
    owner_id: str | None,
    turn: AgentTurn,
    error_type: str,
    text_preview: str | None,
    api_error_status: int | None = None,
) -> None:
    """Publish a ResourceError for a CLI-reported turn failure.

    Fires whenever the SDK signals the turn failed via either:
    - ``AssistantMessage.error`` — the CLI's structured error type
      (``authentication_failed``, ``rate_limit``, ``server_error``, …).
    - ``ResultMessage.is_error`` — final-message failure flag for cases
      where the CLI aborted before any assistant content streamed.

    Routes through the WebSocket hub so the existing
    ``useBackendEvents`` → ``toast.error`` pipeline surfaces it to the
    user. Without this the error lives only inside a chat bubble,
    which is easy to miss while the user is looking at another tab.

    Args:
        project_id: Project UUID the turn belongs to.
        owner_id: Project owner user_id (already resolved by caller).
        turn: The AgentTurn carrying this error (used for turn_id).
        error_type: The SDK-reported error kind (literal like
            ``"authentication_failed"`` or fallback like ``"error"``).
        text_preview: First chunk of the assistant text if available —
            ``"API Error: Your session has expired. Please reauthenticate."``
            and friends. Used as the toast description so the user sees
            the actionable guidance the CLI gave.
        api_error_status: HTTP status from ``ResultMessage.api_error_status``
            (SDK 0.1.76+). Populated for 429 / 5xx failures; distinguishes
            throttling (429) from transport errors (500/529) when deciding
            whether a retry is worth it. Passed through to the toast hint
            so operators see the status code without trawling the log.
    """
    hint_for = {
        "authentication_failed": "Re-authenticate the AWS credentials Bedrock is using and retry.",
        "rate_limit": "Wait a moment and retry — Bedrock is rate-limiting this session.",
        "server_error": "Bedrock reported a server error. Retry shortly.",
    }
    hint = hint_for.get(error_type)
    # Only interpret api_error_status when it's actually an int — the SDK
    # passes it through from the HTTP layer as int | None, but test mocks
    # hand us MagicMocks which would blow up on ``>=``.
    status_code = api_error_status if isinstance(api_error_status, int) else None
    if status_code == 429 and hint is None:
        hint = "HTTP 429 — the API is rate-limiting this session. Wait a moment and retry."
    elif status_code is not None and status_code >= 500 and hint is None:
        hint = f"HTTP {status_code} — upstream API error. Retry shortly."
    # User-initiated stop: the CLI error is an expected side effect of
    # the interrupt (e.g. error_during_execution). Don't show a red toast.
    if getattr(turn, "_stopped_by_user", False):
        return
    details: dict = {"error_type": error_type, "turn_id": turn.turn_id}
    if status_code is not None:
        details["api_error_status"] = status_code
    error = ResourceError(
        code=Codes.AGENT_CLI_ERROR,
        message=text_preview or f"Agent CLI reported an error: {error_type}",
        hint=hint,
        details=details,
    )
    # Record the same story on the turn so the exit chokepoint can persist a
    # durable marker. The toast below is LIVE ONLY — it rides the WebSocket and
    # is gone after a refresh, which would leave the user's prompt sitting there
    # with no reply and no explanation. Stashed rather than written here because
    # this fires mid-stream, and a marker written now would land BEFORE any
    # partial reply the turn still flushes; ``finally`` runs after the
    # transcript append.
    turn._failure_text = " ".join(
        part for part in (error.message, hint) if part
    ).strip()
    if owner_id:
        # Scheduled, not awaited — see ``schedule_project_publish``. An error
        # notification must not stall the very turn it is reporting on.
        schedule_project_publish(
            project_id,
            owner_id,
            {
                "resource": "agent_turn",
                "type": "error",
                "turn_id": turn.turn_id,
                **error.to_dict(),
            },
            turn_id=turn.turn_id,
        )


def _log_scoped_session_store_diagnostic(
    *,
    owner_id: str,
    project_id: str,
    project_dir,
    bucket: str,
    scoped_session,
) -> None:
    """Log the scoped session store's tag and prefix evidence for one turn.

    Emits one INFO line from inside the turn capturing BOTH the tag evidence
    (uid, proj, assumed-role ARN) and the exact S3 bucket + list prefix, so an
    operator reading the AgentCore log group can tell apart the two ways a
    scoped transcript read fails: (1) the principal tags are absent at
    authorization time, or (2) the tags land but the SDK-derived list prefix
    does not match the ABAC grant template.

    This is a best-effort, secret-safe diagnostic. An STS failure or any
    exception is caught and logged; the diagnostic never aborts a turn.

    Args:
        owner_id: User ID (Cognito sub) sent as the uid session tag.
        project_id: Project UUID sent as the proj session tag.
        project_dir: The project directory path the SDK runs in (cwd). The
            SDK canonicalizes this via project_key_for_directory to derive
            the S3 list prefix.
        bucket: The sessions bucket name.
        scoped_session: The boto3.Session minted with scoped STS credentials,
            or None if minting failed (fallback to task-role).

    Logs (one INFO line):
        - path: "scoped" or "unminted" (fallback)
        - uid: The session tag value for the user
        - proj: The session tag value for the project
        - bucket: The sessions bucket name
        - project_key: The SDK-derived list prefix (canonicalized from cwd)
        - expected: The ABAC-template resolution
            `-mnt-projects-users-{uid}-projects-{proj}`
        - prefix_matches: Boolean — True if project_key == expected
        - assumed_role_arn: The assumed-role ARN (proof tags landed), or
            "unavailable" on STS failure
    """
    try:
        # Lazy import: project_key_for_directory lives in the SDK, which we
        # only import inside run_agent_turn to avoid early CLI subprocess
        # spawn. Match that discipline here.
        from claude_agent_sdk import project_key_for_directory

        # Compute the SDK-derived list prefix
        project_key = project_key_for_directory(str(project_dir))

        # Compute the expected ABAC-template prefix
        expected = f"-mnt-projects-users-{owner_id}-projects-{project_id}"

        # Discriminator: does the derived prefix match the ABAC template?
        prefix_matches = (project_key == expected)

        if scoped_session is not None:
            # Scoped path: attempt to get the assumed-role ARN as proof the
            # tagged assume succeeded. Best-effort: if get_caller_identity
            # fails, record ARN as unavailable and still log.
            assumed_role_arn = "unavailable"
            try:
                sts_client = scoped_session.client("sts")
                identity = sts_client.get_caller_identity()
                assumed_role_arn = identity.get("Arn", "unavailable")
            except Exception:
                # STS call failed — record as unavailable, do not raise
                pass

            logger.info(
                "Scoped session store diagnostic: path=scoped uid=%s proj=%s "
                "bucket=%s project_key=%s expected=%s prefix_matches=%s "
                "assumed_role_arn=%s",
                owner_id,
                project_id,
                bucket,
                project_key,
                expected,
                prefix_matches,
                assumed_role_arn,
            )
        else:
            # Fallback path: minting unavailable (task-role credentials)
            logger.info(
                "Scoped session store diagnostic: path=unminted uid=%s proj=%s "
                "bucket=%s project_key=%s expected=%s prefix_matches=%s",
                owner_id,
                project_id,
                bucket,
                project_key,
                expected,
                prefix_matches,
            )
    except Exception as exc:
        # Best-effort: diagnostic failure must never abort the turn
        logger.warning(
            "Scoped session store diagnostic failed: %s", exc, exc_info=True
        )


async def run_agent_turn(
    turn: AgentTurn,
    prompt: str,
    project_id: str,
    coordinator_prompt: str,
    session_id: str | None = None,
    model: str | None = None,
) -> None:
    """Run an agent turn as a background task, writing events to the turn buffer.

    This is the core agent loop — it creates a ClaudeSDKClient, sends the query,
    and iterates over response messages, converting each to an event and
    appending it to the turn's event buffer. Readers (the WebSocket hub)
    consume from the buffer independently.

    Args:
        turn: The AgentTurn to write events into.
        prompt: User message text.
        project_id: Project UUID for session persistence.
        coordinator_prompt: Coordinator agent's system instructions.
        session_id: Optional session ID to resume conversation.
        model: Optional Bedrock model ID override.
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookContext,
        HookInput,
        HookJSONOutput,
        HookMatcher,
    )

    # Resolve project directory for sandbox scoping. ``resolve_project_dir``
    # returns the user-prefixed EFS path in cloud mode and the flat local
    # path in dev mode.
    from app.services.project_paths import resolve_project_dir
    project_dir = resolve_project_dir(project_id)

    # Resolve owner_id for WebSocket fan-out. Every envelope this turn
    # publishes is addressed to ``owner_id`` — the project's owner sees
    # the live stream regardless of which tab they're looking at.
    owner_id = _resolve_owner_id(project_id)

    # Identity attribution for DETACHED turns. ``authorize_tool_call`` gates
    # every MCP tool on the request_context contextvar. Turns rooted in an
    # HTTP request inherit it through the task chain; turns spawned from
    # detached background chains (completion turns fired by the
    # job-progress republisher / reconciler / startup-adopted loops) carry
    # NONE — so every project-scoped tool call inside them would return "not
    # authenticated", and a turn reporting on a finished baseline could not
    # read the results it was reporting on.
    # The coordinator spawns these turns ON BEHALF OF the project's owner,
    # so attribute them to that owner. NOT a bypass: authorize_tool_call
    # still runs require_access(owner, project, action) on every call —
    # cross-project access still denies.
    #
    # An inherited context keeps its user_id, but still has to be told which
    # project this turn is for. The HTTP middleware sets user_id and
    # request_id and leaves project_id None, because it runs before routing
    # and has no project in hand; a turn rooted in an HTTP request therefore
    # inherits a context that is PRESENT but not bound to a project. That is
    # not merely incomplete: authorize_tool_call pins every project-scoped
    # tool call to ``ctx.project_id``, so an unbound context denies all of
    # them. Testing presence alone would leave every interactive chat turn
    # unable to call a single project tool. Only project_id and turn_id are
    # filled in — reattributing to the owner here would launder a live user's
    # turn as a background action and destroy the audit distinction the
    # marker below exists to draw.
    #
    # The relay path sets project_id itself, so the rebind is a no-op there.
    #
    # AUDIT MARKER: the ``detached-turn-`` request_id prefix is the explicit
    # forensic marker that this context is owner-ATTRIBUTED, not
    # owner-initiated — audit log lines carrying it mean "the platform acted
    # on the owner's behalf for a background turn", never "the owner made
    # this request". Grep for it when reviewing authz logs.
    #
    # TASK-SCOPE INVARIANT: this set is confined to the turn's own asyncio
    # task — run_agent_turn must ONLY ever be entered via create_task. An
    # inline await from a long-lived worker task would leak the owner
    # identity into that task's context across turns.
    ctx = request_context.get()
    if ctx is None and owner_id:
        request_context.set(RequestContextValue(
            user_id=owner_id,
            request_id=f"detached-turn-{turn.turn_id[:8]}",
            project_id=project_id,
            turn_id=turn.turn_id,
        ))
        logger.info(
            "[turn %s] Detached turn attributed to project owner for MCP "
            "authz (owner=%s)",
            turn.turn_id[:8], owner_id[:8],
        )
    elif ctx is not None and ctx.project_id != project_id:
        request_context.set(replace(
            ctx, project_id=project_id, turn_id=turn.turn_id,
        ))
        logger.debug(
            "[turn %s] Bound inherited context to this turn's project "
            "(user=%s was=%s)",
            turn.turn_id[:8], ctx.user_id[:8], str(ctx.project_id)[:12],
        )

    turn_stream = f"turn:{turn.turn_id}"
    ws_hub = get_ws_hub()

    # Cloud-mode tenancy controls, and what each one does NOT cover:
    #   * the app-layer ``require_access`` chokepoint gates every API route
    #     and every MCP tool call, but not the agent's built-in tools, which
    #     reach the filesystem directly (only Write/Edit/NotebookEdit are
    #     path-jailed; Bash, Read, Glob and Grep are not);
    #   * the per-user EFS layout
    #     (``/mnt/projects/users/{owner_sub}/projects/{pid}/``) is a naming
    #     convention, not a permission boundary: every tree sits under one
    #     EFS access point with a single POSIX identity;
    #   * the per-request STS-scoped credentials below narrow this turn's
    #     transcript reads and writes to one S3 prefix, with the tenant tag
    #     chosen by the minter.
    # A per-UID variant is not available at this layer: the SDK exposes no
    # way to set the subprocess UID (verified against
    # ``claude_agent_sdk.types``) and the bundled ``claude`` CLI rejects
    # unknown kwargs. The README's "Security scope" section describes these
    # gaps and what closing them would take.
    is_cloud = os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"

    opts: dict = {
        "allowed_tools": (
            list(COORDINATOR_TOOLS)
            + ["mcp__eval__*"]
            + AWS_KNOWLEDGE_DOC_TOOLS
        ),
        # Hard-disable AskUserQuestion (see DISALLOWED_TOOLS). Removes it
        # from the model's context (→ ``--disallowedTools`` CLI flag)
        # without touching any other built-in, MCP, or subagent tool.
        "disallowed_tools": list(DISALLOWED_TOOLS),
        "permission_mode": "bypassPermissions",
        "cwd": str(project_dir),
        # Disabled: empirically confirmed incompatible with Fargate.
        # Direct evidence captured via ECS Exec on a running task:
        # ``bwrap --unshare-user``
        # returns "No permissions to create new namespace"; CapEff
        # has no CAP_SYS_ADMIN; ``unprivileged_userns_clone`` sysctl
        # isn't surfaced. Setting True is a silent no-op — the
        # bundled CLI tolerates the failed bwrap probe and runs
        # unsandboxed, so we'd lose ~6s to the failed probe per turn
        # and gain zero kernel fence, so no in-container bubblewrap
        # sandbox is attempted.
        "sandbox": {"enabled": False},
        "system_prompt": coordinator_prompt,
        "max_turns": 100,
        "include_partial_messages": True,
        # ``stderr`` is installed below, once ``turn`` exists to tag the lines.
        # Surface hook lifecycle events on the message stream so the
        # turn loop can observe PreToolUse / PostToolUse / SubagentStart /
        # Stop in the same channel as assistant content. They are logged for
        # observability; the HookMatcher callbacks remain the authoritative
        # source for side effects.
        "include_hook_events": True,
        # Which filesystem settings scopes the bundled CLI loads. This is
        # the load-bearing isolation control:
        #   * dev → ``["project"]`` only. Crucially this EXCLUDES the
        #     developer's own ``~/.claude/settings.json``, which may set env
        #     such as ``CLAUDE_CODE_USE_MANTLE=1`` — left in, the CLI can start
        #     a login flow that never completes, and ``initialize`` times out.
        #   * cloud → ``["user", "project"]``. The image's ``~/.claude``
        #     is platform-controlled and clean (no Mantle env), so
        #     loading it is safe — and keeping the user scope means a
        #     future platform skill can be installed at user-scope and
        #     picked up without per-project mirroring (the absence of
        #     such mirroring is guarded by
        #     ``test_does_not_mirror_skills_into_project``).
        # If this were unset, the SDK defaults it to load ALL scopes,
        # which brings the developer's settings (and that hang) back in dev.
        "setting_sources": ["user", "project"] if is_cloud else ["project"],
        # In-process eval MCP + the remote AWS Knowledge MCP. The latter
        # is an HTTP MCP (no subprocess spawn — see AWS_KNOWLEDGE_MCP_*
        # above); its doc tools are scoped via AWS_KNOWLEDGE_DOC_TOOLS in
        # allowed_tools so the agent can look up Bedrock model cards at
        # recommendation time without roaming the rest of the server.
        "mcp_servers": {
            "eval": get_eval_mcp_server(),
            AWS_KNOWLEDGE_MCP_SERVER: _aws_knowledge_mcp_config(),
        },
        "agents": SUBAGENTS,
        # Eager flush mirrors transcript frames to S3 as they arrive
        # instead of batching to turn end. Required for our mid-turn
        # page-refresh story: the chat-history endpoint reads the
        # SessionStore, so without eager flush a refresh during a
        # running turn returns ``[]`` (the user message and partial
        # assistant frames live only in the SDK's in-memory transcript
        # mirror until the turn completes).
        "session_store_flush": "eager",
    }
    # ``session_id`` handling is deferred until after the session store is
    # wired (below): whether it means "resume" or "adopt and create" depends
    # on whether the session OBJECT exists in the store, and the scoped store
    # is only built further down. See _apply_session_identity.
    if model:
        opts["model"] = model

    # Plug the platform's SessionStore into the SDK so cloud mode persists
    # transcripts to S3 and dev mode keeps the SDK's built-in on-disk
    # behavior. The platform returns ``None`` in dev (SDK default).
    #
    # Cloud mode: mint per-turn STS-scoped credentials and
    # build a project-scoped ``S3SessionStore`` whose underlying S3
    # client is narrowed to the SDK's cwd-derived key shape
    # (``project_key_for_directory(cwd)``; see ``sts_token_vending``
    # for the policy). SDK transcript writes then go through scoped
    # creds — a bug in require_access can't leak transcripts because
    # the AWS API itself rejects cross-tenant calls.
    from app.platform import get_platform
    _platform = get_platform()
    _session_store = _platform.session_store
    if _session_store is not None and is_cloud:
        from app.main import get_project_manager
        from app.platform.aws.s3_session_store import (
            LOAD_CONCURRENCY,
            S3SessionStore,
        )
        from app.services.aws_credential_cache import get_aws_credential_cache

        _project = get_project_manager().get_project(project_id)
        _owner_id = _project.owner_id if _project else None
        if _owner_id:
            # Mint scoped creds once at turn start. The credential
            # cache holds the boto3.Session for 5 min, so the SDK's
            # multiple transport callbacks during this turn share one
            # ``sts:AssumeRole`` call. ``project_dir`` is the same
            # cwd the SDK runs the agent in; the inline IAM policy
            # uses it to derive the SDK-canonical S3 prefix.
            _scoped_session = await get_aws_credential_cache().get_session(
                _owner_id, project_id, str(project_dir)
            )
            if _scoped_session is not None:
                _scoped_s3 = _scoped_session.client(
                    "s3", config=Config(max_pool_connections=LOAD_CONCURRENCY)
                )
                opts["session_store"] = S3SessionStore(
                    s3_client=_scoped_s3,
                    bucket=_session_store.bucket,
                )
                # Best-effort diagnostic: log scoped session evidence
                try:
                    _log_scoped_session_store_diagnostic(
                        owner_id=_owner_id,
                        project_id=project_id,
                        project_dir=project_dir,
                        bucket=_session_store.bucket,
                        scoped_session=_scoped_session,
                    )
                except Exception:
                    # Diagnostic failure must never abort the turn
                    pass
            else:
                # STS minting unavailable (dev TokenVending returned
                # None, or transient STS failure). Fall back to the
                # task-role-credentialed store; app-layer require_access
                # is the only fence on that path.
                opts["session_store"] = _session_store
                # Best-effort diagnostic: log unminted fallback
                try:
                    _log_scoped_session_store_diagnostic(
                        owner_id=_owner_id,
                        project_id=project_id,
                        project_dir=project_dir,
                        bucket=_session_store.bucket,
                        scoped_session=None,
                    )
                except Exception:
                    # Diagnostic failure must never abort the turn
                    pass
        else:
            opts["session_store"] = _session_store
    elif _session_store is not None:
        opts["session_store"] = _session_store

    # Session identity: resume vs adopt-and-create, derived from the STORE.
    #
    # The relay's start-of-turn persistence MINTS a fresh session id for a
    # brand-new project and persists it before the POST, so a mid-turn refresh
    # renders the conversation, and the microVM must ADOPT that id. Passing
    # every persisted id as ``resume`` breaks that case: the CLI hard-fails
    # resuming a session that was never created ("No conversation found with
    # session ID", exit 1), so a new project's first cloud turn would die and
    # every later turn would resume the same never-created id. Existing
    # projects are unaffected — their ids point at real sessions.
    #
    # The branch is derived from the one fact that decides it — does the
    # session object exist? — checked against the SAME store the SDK will
    # use, with the SAME key shape ``chat_history`` reads with. No flag to
    # drift, and the wedged-project case self-heals: the ghost id becomes
    # the ADOPTED id of the first session actually created under it.
    if session_id:
        await _apply_session_identity(opts, session_id, project_dir)

    # First turn on a brand-new project can beat the project directory's
    # visibility on this EFS mount (cached negative lookup). Wait briefly
    # rather than letting the SDK's cwd check fail a user-visible turn.
    # Cloud-only: in dev the coordinator and runner share one local
    # filesystem, so no cross-mount visibility race exists (and the wait's
    # sleeps would just slow every dev/test turn on a missing dir).
    if is_cloud:
        await _await_project_dir(project_dir)

    async def emit(event_type: str, data: dict) -> None:
        """Append an envelope to the turn buffer and fan out via ws_hub.

        Two writes per emit by design:
        - Buffer (``turn.append_event``) is the source of truth used by
          replay-on-reconnect when a WS client drops mid-turn.
        - ``ws_hub.publish_to_user`` pushes the envelope to every active
          WS connection for this turn's owner so the SPA renders it
          live.
        """
        envelope = make_envelope(turn_stream, event_type, data)
        await turn.append_event(envelope)
        if owner_id:
            await ws_hub.publish_to_user(owner_id, envelope)

    # Subagent start/stop pairing state. Counts outstanding starts to detect
    # backgrounded subagents that never report. The count is name-free: both
    # hooks read `agent_type` from input_data, and pairing on that value would
    # false-positive every healthy delegation the moment the CLI labels start
    # and stop differently. Counting starts and decrementing on ANY stop is
    # immune to that drift.
    _subagent_pairing_count = {"outstanding": 0}
    _subagent_types_started: list[str] = []

    # Subagent lifecycle hooks — emit a status event so the frontend shows
    # "Analyzing data…" / "Building harness…" / "Optimizing prompt…"
    async def _subagent_start_hook(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        agent_type = input_data.get("agent_type", "unknown")
        _subagent_pairing_count["outstanding"] += 1
        if agent_type not in _subagent_types_started:
            _subagent_types_started.append(agent_type)
        logger.info("[turn %s] Subagent start: %s", turn.turn_id[:8], agent_type)
        await emit("status", {"state": "tool_running", "tool_name": f"subagent:{agent_type}"})
        return {}

    async def _subagent_stop_hook(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        agent_type = input_data.get("agent_type", "unknown")
        _subagent_pairing_count["outstanding"] = max(0, _subagent_pairing_count["outstanding"] - 1)
        logger.info("[turn %s] Subagent stop: %s", turn.turn_id[:8], agent_type)
        await emit("status", {"state": "streaming"})
        return {}

    async def _project_dir_pre_tool_use_hook(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Reject Write/Edit/NotebookEdit whose resolved target leaves project_dir.

        Scope: ``Write``, ``Edit``, ``NotebookEdit`` only. Bash is
        intentionally NOT jailed — a static command-line scan cannot
        reliably contain shell escapes (``~``, ``..``, ``$VAR``,
        ``$(...)``, glued redirects, brace expansion), so scanning it
        would assert a guarantee it does not deliver.

        Defense in depth on top of the SDK's ``cwd`` setting for the
        path-arg tools we DO jail: exactly one user-supplied path per
        call, no shell layer to fight, so the check is structural.

        Logic lives in the pure ``_pre_tool_use_jail_decision`` helper
        so the test suite can pin it without standing up a full
        agent turn.
        """
        if not isinstance(input_data, dict):
            return {}
        tool_name = input_data.get("tool_name") or ""
        tool_input = input_data.get("tool_input") or {}
        # Subagents run in the FOREGROUND, always. The turn architecture
        # assumes delegation is synchronous — the coordinator promises "I'll
        # report the findings" inside the SAME turn. An async subagent's output
        # lands in a file nobody reads; the turn has already reported success.
        #
        # The CLI's delegation tool can arrive under either name (Task or
        # Agent), and the CLI's background-by-default mode can turn on with no
        # local change: under it the same binary returns "Async agent launched
        # successfully", because an OMITTED run_in_background means async and
        # only an EXPLICIT false opts out. The rewrite therefore covers both
        # names and forces run_in_background=False when the flag is absent or
        # True. The env pin (CLAUDE_CODE_DISABLE_BACKGROUND_TASKS) and
        # this rewrite are both subject to vendor schema changes; every SDK bump
        # must re-verify synchronous delegation.
        forced = _forced_sync_delegation_input(tool_name, tool_input)
        if forced is not None:
            subagent_type = tool_input.get("subagent_type") if isinstance(tool_input, dict) else None
            requested_background = bool(tool_input.get("run_in_background")) if isinstance(tool_input, dict) else False
            if requested_background:
                logger.warning(
                    "[turn %s] PreToolUse: model requested background %s "
                    "(subagent_type=%s) — forcing synchronous",
                    turn.turn_id[:8],
                    tool_name,
                    subagent_type,
                )
            else:
                logger.info(
                    "[turn %s] PreToolUse: forcing %s synchronous (subagent_type=%s)",
                    turn.turn_id[:8],
                    tool_name,
                    subagent_type,
                )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": forced,
                }
            }
        # SDK reports the agent's actual cwd per-invocation. The hook
        # runs in the FastAPI process whose own cwd is unrelated; we
        # MUST anchor relative-path resolution to the SDK's value or
        # legitimate relative writes get denied.
        tool_cwd = input_data.get("cwd") if isinstance(input_data.get("cwd"), str) else None
        decision = _pre_tool_use_jail_decision(
            tool_name, tool_input, project_dir, tool_cwd=tool_cwd,
        )
        if decision is None:
            return {}
        logger.warning(
            "[turn %s] PreToolUse DENY tool=%s reason=%s",
            turn.turn_id[:8],
            tool_name,
            decision["hookSpecificOutput"]["permissionDecisionReason"],
        )
        return decision

    async def _fs_mutation_post_tool_use_hook(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Publish ``filesystem/changed`` to the project's owner WS after any
        filesystem-mutating tool completes.

        Replaces the previous post-tool-result string-match against
        ``_FS_MUTATING_TOOLS`` inside the UserMessage branch of the
        turn loop. Running from a PostToolUse hook means the tool
        name comes from the SDK directly (no stale-state risk from
        ``current_tool_name`` being cleared by intervening messages)
        and the dispatch happens in one place instead of threaded
        through the stream loop.
        """
        tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
        if tool_name and tool_name in _FS_MUTATING_TOOLS:
            # Through the execution-home-aware chokepoint, NOT a direct
            # ws_hub publish. In the AgentCore microVM a direct publish
            # fans out to zero browser connections and the event is LOST —
            # the SPA's config/file refetch triggers (useHarnessConfig,
            # CodeTab, PromptsTab all key on filesystem/changed) never
            # fire, and the panel goes stale until a remount. The
            # chokepoint frames the event into the live turn's SSE stream
            # so the coordinator re-targets it onto the project stream
            # (outbox fallback if no live turn); outside the microVM it
            # publishes locally exactly as before.
            from app.services.jobs import _publish_project_event
            _publish_project_event(project_id, {
                "resource": "filesystem",
                "type": "changed",
                "tool": tool_name,
            })
        return {}

    # Defense-in-depth for the unbounded part-count ceiling: raise the SDK
    # resume SessionStore.load() deadline from the default 60s to 180s.
    # This is NOT a substitute for the parallel read — it just
    # gives the parallelized load more time to complete on truly massive
    # sessions before the SDK times out.
    opts["load_timeout_ms"] = 180_000

    # SDK subprocess env pins (CLAUDE_CODE_DISABLE_BACKGROUND_TASKS,
    # DISABLE_GROWTHBOOK). The rationale for both pins and the operational
    # re-verify-on-SDK-bump warning are in sdk_subprocess_env.py.
    opts["env"] = sdk_subprocess_env(opts.get("env"))

    # Bounds ONE stdout frame read off the CLI. The SDK's 1 MiB default sits
    # under what a legitimate MCP tool result can serialize to; see
    # SDK_MAX_STDOUT_FRAME_BYTES.
    opts["max_buffer_size"] = SDK_MAX_STDOUT_FRAME_BYTES

    opts["hooks"] = {
        "SubagentStart": [HookMatcher(matcher=None, hooks=[_subagent_start_hook])],
        "SubagentStop": [HookMatcher(matcher=None, hooks=[_subagent_stop_hook])],
        # PreToolUse fires before each Write/Edit/NotebookEdit dispatch
        # and blocks (deny + reason surfaced to the agent) if the
        # resolved target is outside ``project_dir``. Bash is NOT
        # jailed — see _JAILED_TOOLS rationale at module scope.
        "PreToolUse": [HookMatcher(matcher=None, hooks=[_project_dir_pre_tool_use_hook])],
        # PostToolUse with ``matcher=None`` fires for every tool call the
        # coordinator (and its subagents) make. The hook filters inline
        # by tool name against the _FS_MUTATING_TOOLS allowlist. Requires
        # claude-agent-sdk >= 0.1.74 for the hook to run in-process via
        # the message stream.
        "PostToolUse": [HookMatcher(matcher=None, hooks=[_fs_mutation_post_tool_use_hook])],
    }

    # CLI stderr → our logs. The SDK pipes the subprocess's stderr only
    # when a callback is registered; without it the CLI's actual complaint
    # ("Invalid session ID. Must be a valid UUID.") is thrown away and every
    # failure logs as an opaque "exit code 1". Line-buffered, bounded, WARNING
    # level.
    #
    # WARNING rather than DEBUG is deliberate: these lines are the CLI's own
    # complaint about a failure we would otherwise diagnose blind, so they have
    # to survive the default level. The stable ``[cli-stderr]`` prefix keeps
    # them filterable in CloudWatch Logs Insights.
    def _log_cli_stderr(line: str) -> None:
        text = subprocess_log_line(line)
        if text:
            logger.warning("[turn %s][cli-stderr] %s", turn.turn_id[:8], text)

    opts["stderr"] = _log_cli_stderr

    await emit("status", {"state": "connecting"})

    try:
        logger.info("[turn %s] Connecting (session=%s)", turn.turn_id[:8], session_id)

        # Time the SDK initialize handshake (entering the context
        # manager runs the control-protocol initialize round-trip;
        # awaiting it here measures end-to-end). Used to catch
        # sandbox-related slowdowns on Fargate without waiting for
        # the 60s SDK timeout to fire — we want forensics during the
        # hang, not after. ``HOME`` is captured in the warning line
        # so a $HOME misconfiguration shows up without ECS Exec.
        # Turn-reliability guard: one in-place retry when the CLI returns
        # an 'empty success' (success subtype, no assistant output, no tool
        # activity, $0 cost — the model was never called). It can happen on
        # the first resume after an AgentCore fleet recycle, and nothing
        # downstream would otherwise treat it as a failure.
        empty_success_retry = False
        # Orphaned subagent types (survives the retry loop). A synchronous
        # subagent always stops before the turn's result; a backgrounded one
        # never does. Pairing state is per-attempt, but this list accumulates
        # across both attempts so the turn-level failure can report which
        # subagents were orphaned.
        _orphaned_subagent_types: list[str] = []
        for _turn_attempt in (0, 1):
            import time as _time
            _init_t0 = _time.monotonic()
            async with ClaudeSDKClient(options=ClaudeAgentOptions(**opts)) as client:
                _init_ms = (_time.monotonic() - _init_t0) * 1000
                if _init_ms > 5000:
                    import os as _os
                    logger.warning(
                        "[agent-init] slow init %.1fs sandbox=%s home=%s uid=%d",
                        _init_ms / 1000.0,
                        opts.get("sandbox"),
                        _os.environ.get("HOME"),
                        _os.geteuid(),
                    )
                else:
                    logger.info(
                        "[agent-init] turn=%s init=%.0fms sandbox=%s",
                        turn.turn_id[:8], _init_ms, opts.get("sandbox"),
                    )
                # Store client reference so the turn can be interrupted externally
                turn._client = client

                await emit("status", {"state": "streaming"})
                # Size, not content: this is the user's message, and one shared
                # log group holds every tenant's turns.
                logger.info(
                    "[turn %s] Sending user message (%d chars)",
                    turn.turn_id[:8], len(prompt),
                )

                await client.query(prompt)

                has_streamed_text = False
                saw_tool_activity = False
                current_tool_name = ""
                got_result = False
                # One-shot guard so the AssistantMessage-side and ResultMessage-side
                # CLI-error signals don't each publish a toast for the same turn.
                cli_error_published = False
                # Accumulate input_json_delta partials per block index so we can
                # log the full tool input at content_block_stop.
                tool_input_buf: dict[int, str] = {}
                # Reset pairing state for this attempt. The empty-success retry
                # must not inherit the ghost attempt's counters.
                _subagent_pairing_count["outstanding"] = 0
                _subagent_types_started.clear()

                # CRITICAL: Use asyncio.wait() not asyncio.wait_for() to avoid
                # corrupting the async iterator.
                async def _next_msg(aiter):
                    return await aiter.__anext__()

                response_iter = client.receive_response().__aiter__()
                while True:
                    task = asyncio.create_task(_next_msg(response_iter))
                    # Re-emit the current tool's running status every 15s while
                    # waiting for the next SDK message — the SPA uses this as a
                    # "still alive" signal for in-progress tool indicators.
                    # Network-level keepalive lives on the ws_hub keepalive task
                    # (every 30s) so we no longer need a transport-level pulse
                    # here.
                    while True:
                        done, _ = await asyncio.wait({task}, timeout=15.0)
                        if done:
                            break
                        if current_tool_name:
                            await emit("status", {
                                "state": "tool_running",
                                "tool_name": current_tool_name,
                            })

                    try:
                        message = task.result()
                    except StopAsyncIteration:
                        break

                    # Hide subagent-internal narration from the user-visible stream.
                    # Subagent assistant text + streaming deltas carry a non-null
                    # ``parent_tool_use_id``; without filtering they'd bubble into
                    # chat as "You are working on project ..." wall-of-text. The
                    # SubagentStart / SubagentStop hooks already surface a UI
                    # status chip for the delegation itself.
                    #
                    # Important: UserMessage carrying a tool_use_result ALSO has
                    # parent_tool_use_id set (it's the SDK delivering the tool's
                    # result back into the conversation), but it has no
                    # user-visible text — we emit a tool_result event, not
                    # assistant content. Filtering those breaks mid-turn
                    # refreshes (frontend dispatches PROJECT_FS_CHANGED_EVENT on
                    # tool_result). So scope the filter to AssistantMessage and
                    # StreamEvent only.
                    parent_id = getattr(message, "parent_tool_use_id", None)
                    if isinstance(parent_id, str) and parent_id:
                        is_assistant = (
                            hasattr(message, "content")
                            and isinstance(getattr(message, "content", None), list)
                        )
                        is_stream_event = (
                            hasattr(message, "event")
                            and isinstance(getattr(message, "event", None), dict)
                        )
                        if is_assistant or is_stream_event:
                            continue

                    # StreamEvent — has .event dict
                    if hasattr(message, "event") and isinstance(getattr(message, "event", None), dict):
                        event = message.event
                        event_type = event.get("type", "")

                        if event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                has_streamed_text = True
                                await emit("text_delta", {"text": delta["text"]})
                            elif delta.get("type") == "thinking_delta":
                                # Thinking content streams live. The SDK thinking
                                # delta carries its text under the ``thinking``
                                # field (not ``text``).
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    await emit("thinking_delta", {"text": thinking_text})
                            elif delta.get("type") == "input_json_delta":
                                partial = delta.get("partial_json", "")
                                idx = event.get("index")
                                if isinstance(idx, int):
                                    tool_input_buf[idx] = tool_input_buf.get(idx, "") + partial
                                await emit("tool_input", {"partial_json": partial})

                        elif event_type == "content_block_start":
                            block = event.get("content_block", {})
                            block_type = block.get("type", "")
                            if block_type == "tool_use":
                                name = block.get("name", "")
                                current_tool_name = name
                                saw_tool_activity = True
                                # No log here: the operational "Tool: <name> <input>"
                                # INFO line is emitted once at content_block_stop
                                # below, when the input has finished streaming —
                                # one readable line per call instead of a separate
                                # start line + input line + per-chunk messages.
                                await emit("tool_start", {"name": name, "id": block.get("id", "")})
                                await emit("status", {"state": "tool_running", "tool_name": name})
                            elif block_type == "thinking":
                                # Start of a thinking (reasoning) block. Emit a
                                # start envelope so the assembler can open a new
                                # reasoning part cleanly. Signature blocks /
                                # redacted-thinking carry no displayable text —
                                # the text is delivered via thinking_delta above.
                                await emit("thinking_start", {})
                            elif block_type == "text":
                                # Per-block plumbing — DEBUG, not part of the story.
                                logger.debug("[turn %s] Text block start", turn.turn_id[:8])

                        elif event_type == "content_block_stop":
                            idx = event.get("index")
                            # The SDK's delegation tool can arrive under either name
                            # (Task or Agent). The subagent identity lives in the
                            # ``subagent_type`` arg, which only arrives via
                            # input_json_delta partials and is complete at this point.
                            # Rewrite the tool name to ``subagent:<type>`` so the
                            # frontend can show a meaningful label.
                            rename_to: str | None = None
                            if current_tool_name and isinstance(idx, int):
                                input_json = tool_input_buf.pop(idx, "")
                                is_subagent_tool = current_tool_name in _DELEGATION_TOOLS
                                if is_subagent_tool and input_json:
                                    try:
                                        parsed_input = json.loads(input_json)
                                    except json.JSONDecodeError:
                                        parsed_input = None
                                    if isinstance(parsed_input, dict):
                                        subagent_type = parsed_input.get("subagent_type")
                                        if isinstance(subagent_type, str) and subagent_type:
                                            rename_to = f"subagent:{subagent_type}"
                                # One consolidated operational line per tool call:
                                # name + argument size, logged once here (input has
                                # finished streaming) instead of the old separate
                                # Tool-start + Tool-input lines. Skip Task/Agent —
                                # those are subagent delegations, and the Subagent
                                # start/stop lines already mark them; logging a
                                # "Tool:" line too would double up.
                                #
                                # The arguments themselves are NOT logged. They carry
                                # file contents for Write/Edit and whole command lines
                                # for Bash, into a log group shared by every tenant.
                                if not is_subagent_tool:
                                    logger.info(
                                        "[turn %s] Tool: %s (args %d chars)",
                                        turn.turn_id[:8],
                                        current_tool_name,
                                        len(input_json or ""),
                                    )
                            if current_tool_name:
                                end_payload: dict = {}
                                if rename_to:
                                    end_payload["name"] = rename_to
                                await emit("tool_end", end_payload)

                    # AssistantMessage — complete response (after tool use)
                    elif hasattr(message, "content") and isinstance(getattr(message, "content", None), list):
                        block_types = [getattr(b, "type", "?") for b in message.content] if message.content else []
                        text_preview = assistant_text_preview(message.content)
                        # SDK attaches an ``error`` attribute on the AssistantMessage
                        # when the CLI itself failed (expired session, transport
                        # error, etc.). Log those at ERROR so they don't blend in
                        # with successful-turn INFO lines, and publish a one-shot
                        # ResourceError so the frontend toasts it — otherwise the
                        # error message sits inside a chat bubble that's easy to
                        # miss.
                        cli_error = getattr(message, "error", None)
                        is_error_msg = cli_error is not None
                        # Success path → DEBUG: the per-message "blocks=... text_len=...
                        # streamed=..." plumbing and the assistant text preview are
                        # developer trace, not the operational story (the turn's
                        # Result line carries the outcome). Error path STAYS at ERROR
                        # so a CLI failure (expired session, transport error) never
                        # blends into the success noise — and it still drives the
                        # one-shot ResourceError toast below.
                        log_assistant = logger.error if is_error_msg else logger.debug
                        log_assistant("[turn %s] AssistantMessage: blocks=%s text_len=%d streamed=%s",
                            turn.turn_id[:8], block_types, len(text_preview), has_streamed_text)
                        if text_preview:
                            log_assistant("[turn %s] Assistant: %s", turn.turn_id[:8], text_preview)
                        if is_error_msg and not cli_error_published:
                            await _publish_cli_error(
                                project_id=project_id,
                                owner_id=owner_id,
                                turn=turn,
                                error_type=str(cli_error),
                                text_preview=text_preview,
                            )
                            cli_error_published = True
                        if not has_streamed_text:
                            for block in message.content:
                                if hasattr(block, "text"):
                                    await emit("text_delta", {"text": block.text})
                        has_streamed_text = False

                    # UserMessage with tool results — log and emit the
                    # ``tool_result`` event with result_preview + status. The
                    # filesystem-changed notification moved to
                    # ``_fs_mutation_post_tool_use_hook`` above (PostToolUse
                    # hook), so this branch no longer string-matches tool names —
                    # the hook owns that decision.
                    elif hasattr(message, "tool_use_result") and message.tool_use_result:
                        current_tool_name = ""
                        saw_tool_activity = True
                        # Build a bounded preview for the live assembler. Reuse
                        # the one-line collapse from tool_result_preview, then
                        # truncate to the transcript preview budget (2 KB) so the
                        # envelope stays bounded.
                        preview_text = tool_result_preview(message.tool_use_result)
                        truncated = len(preview_text) > 2000
                        result_preview = preview_text[:2000] if truncated else preview_text
                        # Determine status from the is_error flag (if present).
                        is_error_flag = getattr(message.tool_use_result, "is_error", False) if not isinstance(message.tool_use_result, str) else False
                        status = "error" if is_error_flag else "ok"
                        # Outcome and size, not content: a tool result is file
                        # contents, query output or dataset rows.
                        logger.info(
                            "[turn %s] Tool result: %s (%d chars%s)",
                            turn.turn_id[:8], status, len(preview_text),
                            ", truncated" if truncated else "",
                        )
                        await emit("tool_result", {
                            "status": status,
                            "result_preview": result_preview,
                            "truncated": truncated,
                        })

                    # HookEventMessage — lifecycle events surfaced on the
                    # message stream (``include_hook_events=True``). Logged at
                    # DEBUG for observability; the HookMatcher callbacks
                    # (SubagentStart/Stop, PostToolUse) remain the
                    # authoritative sources for the side-effects. Use
                    # isinstance against the imported SDK class so subclasses
                    # resolve correctly and we discriminate the same way the
                    # other branches (content list / tool_use_result) do.
                    elif isinstance(message, HookEventMessage):
                        logger.debug(
                            "[turn %s] Hook: %s subtype=%s",
                            turn.turn_id[:8],
                            getattr(message, "hook_event_name", None),
                            getattr(message, "subtype", None),
                        )

                    # ResultMessage — final turn result
                    elif hasattr(message, "session_id") and hasattr(message, "num_turns"):
                        got_result = True
                        # ``is_error`` is the SDK's canonical signal that the turn
                        # failed (transport errors, expired session, max-turns
                        # overrun, etc.); surface those at ERROR so they don't
                        # blend in with successful-turn INFO lines.
                        result_is_error = getattr(message, "is_error", False)
                        log_fn = logger.error if result_is_error else logger.info
                        log_fn("[turn %s] Result: %s cost=$%s",
                            turn.turn_id[:8], message.subtype, message.total_cost_usd)

                        # Empty-success guard: success subtype with no
                        # assistant output, no tool activity, an empty result
                        # string AND $0 cost means the model was never called —
                        # a failure the CLI mislabels as success. Retry once;
                        # a second one becomes a real error. Detection sits
                        # BEFORE the session-persist / transcript / result-emit
                        # side effects so the ghost attempt leaves no trace.
                        if (
                            message.subtype == "success"
                            and not result_is_error
                            and not has_streamed_text
                            and not saw_tool_activity
                            and not (message.result or "").strip()
                            and not message.total_cost_usd
                        ):
                            logger.error(
                                "[turn %s] Empty-success result (attempt %d): no "
                                "assistant output, no tool activity, $0 cost — the "
                                "model was never called. raw=%r",
                                turn.turn_id[:8], _turn_attempt + 1, message,
                            )
                            if _turn_attempt == 0:
                                empty_success_retry = True
                                break
                            # Second empty success in a row: surface as a real
                            # turn failure instead of rendering silence.
                            if not cli_error_published:
                                await _publish_cli_error(
                                    project_id=project_id,
                                    owner_id=owner_id,
                                    turn=turn,
                                    error_type="empty_result",
                                    text_preview=(
                                        "The agent returned an empty result twice — "
                                        "the model was never called. Send the message "
                                        "again; if it persists, check the backend logs."
                                    ),
                                )
                                cli_error_published = True
                            await emit("result", {
                                "subtype": "error",
                                "result": None,
                                "error": (
                                    "Agent returned an empty result twice (no output, "
                                    "no tools, $0 cost). The turn did not run."
                                ),
                                "session_id": message.session_id,
                                "cost_usd": 0,
                                "num_turns": message.num_turns,
                            })
                            break
                        # Fallback publish: some CLI failures arrive without a
                        # preceding AssistantMessage.error (e.g. the turn is
                        # aborted before any assistant content streams). The
                        # ``cli_error_published`` guard keeps us from toasting
                        # twice when both signals fire.
                        if result_is_error and not cli_error_published:
                            await _publish_cli_error(
                                project_id=project_id,
                                owner_id=owner_id,
                                turn=turn,
                                error_type=str(getattr(message, "subtype", "") or "error"),
                                text_preview=None,
                                api_error_status=getattr(message, "api_error_status", None),
                            )
                            cli_error_published = True
                        if message.session_id:
                            turn.session_id = message.session_id
                            # Persist session for all turns — orchestrator turns share
                            # the project session so the coordinator has full context
                            # (iteration history, domain knowledge) when reviewing
                            # job results and delegating to subagents.
                            await _persist_session(project_id, message.session_id)
                            # Pass the SDK's canonical final answer so the transcript
                            # append can backfill the trailing answer if this
                            # ResultMessage was delivered before the SDK flushed the
                            # final text block to the session file (flush-ordering race).
                            #
                            # single-writer: inside the AgentCore runtime the
                            # append is SKIPPED — the coordinator's relay drain runs
                            # it when the result frame arrives. The transcript is a
                            # one-process write target: this process's rename-based
                            # tail/record rewrites would race the coordinator's
                            # appends (process-local locks can't serialize across
                            # the boundary) and drop lines.
                            from app.platform.process_role import is_agentcore_runtime
                            if not is_agentcore_runtime():
                                final_answer = message.result if message.subtype == "success" else None
                                await _append_turn_to_transcript(
                                    project_id, message.session_id,
                                    final_answer=final_answer,
                                    turn_id=turn.turn_id,
                                )
                        # Orphaned subagent guard: a synchronous subagent always
                        # stops before the turn's result; a backgrounded one never
                        # does. Unmatched starts at result time mean the delegated
                        # work never happened. This pairing check holds when the CLI
                        # changes the tool name, the flag, or the schema again.
                        if _subagent_pairing_count["outstanding"] > 0 and not getattr(turn, "_stopped_by_user", False):
                            _orphaned_subagent_types.extend(_subagent_types_started)
                            logger.error(
                                "[turn %s] Orphaned subagent detected: %d outstanding start(s), types=%r — "
                                "delegation ran in the background, promised work never reported",
                                turn.turn_id[:8],
                                _subagent_pairing_count["outstanding"],
                                _subagent_types_started,
                            )
                            if not cli_error_published:
                                await _publish_cli_error(
                                    project_id=project_id,
                                    owner_id=owner_id,
                                    turn=turn,
                                    error_type="orphaned_subagent",
                                    text_preview=_ORPHANED_SUBAGENT_MESSAGE,
                                )
                                cli_error_published = True
                        await emit("result", {
                            "subtype": message.subtype,
                            "result": message.result if message.subtype == "success" else None,
                            "session_id": message.session_id,
                            "cost_usd": message.total_cost_usd,
                            "num_turns": message.num_turns,
                        })

            if not empty_success_retry:
                break
            empty_success_retry = False
            logger.warning(
                "[turn %s] Empty-success result — retrying the turn once",
                turn.turn_id[:8],
            )
            await emit("status", {"state": "retrying"})

        if not got_result:
            logger.warning("[turn %s] Ended WITHOUT ResultMessage", turn.turn_id[:8])
        # Fail the turn when orphaned subagents were detected, otherwise complete normally.
        if _orphaned_subagent_types:
            await turn.fail(ResourceError(
                code=Codes.AGENT_CLI_ERROR,
                message=_ORPHANED_SUBAGENT_MESSAGE,
                hint="The subagent delegation ran in the background and never reported. Send the request again.",
                details={
                    "turn_id": turn.turn_id,
                    "orphaned_subagents": _orphaned_subagent_types,
                },
            ))
            logger.error(
                "[turn %s] Failed: orphaned subagents %r",
                turn.turn_id[:8], _orphaned_subagent_types
            )
        else:
            await turn.complete()
            logger.info("[turn %s] Completed (%d events)", turn.turn_id[:8], len(turn.events))

        # Turn-complete notification on the project's WS stream. Tabs
        # that cached something cheap (prompts list, harness config,
        # etc.) listen for this as their "refresh on agent activity"
        # trigger instead of subscribing to a generic window event.
        #
        # Scheduled, NOT awaited. ``publish_project`` allocates a durable
        # project seq, so awaiting it here puts store I/O on the
        # turn's critical path — measured to stall the turn before its terminal
        # transition, leaving the client blocked on a ``result`` envelope that
        # never arrives. A cache-invalidation notice must never gate a turn
        # completing. Monotonicity is unaffected: the single durable counter is
        # still the only issuer, it is simply read off this path.
        if owner_id:
            schedule_project_publish(
                project_id,
                owner_id,
                {
                    "resource": "agent_turn",
                    "type": "completed",
                    "turn_id": turn.turn_id,
                },
                turn_id=turn.turn_id,
            )

    except Exception as exc:
        # If the user explicitly stopped this turn, the exception is
        # expected (CancelledError or similar from the interrupt). Don't
        # publish an error event — the frontend shouldn't show a red toast
        # for user-initiated cancellation.
        if getattr(turn, "_stopped_by_user", False):
            logger.info("[turn %s] Stopped by user", turn.turn_id[:8])
            await emit("result", {
                "subtype": "cancelled",
                "result": None,
                "error": None,
                "session_id": None,
                "cost_usd": None,
                "num_turns": None,
            })
            # turn.status is already "failed" (set by stop_turn); don't
            # attach an error. Publish a cancelled event so panels can refresh.
            if owner_id:
                await ws_hub.publish_project(project_id, owner_id, {
                    "resource": "agent_turn",
                    "type": "cancelled",
                    "turn_id": turn.turn_id,
                })
        else:
            logger.error("[turn %s] Error: %s", turn.turn_id[:8], exc, exc_info=True)
            # SDK 0.1.77+ surfaces the CLI's actual error text directly in
            # the exception. Using ``str(exc)`` as the toast message avoids
            # the redundant "Agent turn failed:" prefix — the exception
            # string already contains the actionable info ("Reached maximum
            # number of turns", "Command failed: 429 Too Many Requests", …).
            turn_error = ResourceError(
                code=Codes.AGENT_SDK_STREAM_ERROR,
                message=str(exc) or f"Agent turn failed ({type(exc).__name__})",
                hint="Check the backend log for the traceback; see Bedrock credentials if the error mentions tokens.",
                details={"exception": type(exc).__name__, "origin": turn.origin},
            )
            # Emit the error through the WS stream (existing contract —
            # frontend reads ``error`` from the ``result`` event) AND attach
            # it to the turn for out-of-band consumers + publish on the
            # project's WS stream so listeners on other tabs see it.
            await emit("result", {
                "subtype": "error",
                "result": None,
                "error": str(exc),
                "session_id": None,
                "cost_usd": None,
                "num_turns": None,
            })
            await turn.fail(turn_error)
            if owner_id:
                await ws_hub.publish_project(project_id, owner_id, {
                    "resource": "agent_turn",
                    "type": "error",
                    "turn_id": turn.turn_id,
                    **turn_error.to_dict(),
                })
    finally:
        turn._client = None
        # Durable failure marker. The live ResourceError toast does not survive a
        # refresh, so without this a failed turn leaves the user's prompt with no
        # reply and no explanation. Written HERE because this is the single
        # chokepoint every turn exits through — the four ``_publish_cli_error``
        # sites straddle the transcript append (two before, one after, one on a
        # path that never reaches it), so any earlier site would cover some
        # failures and silently miss others.
        #
        # SINGLE-WRITER: skipped inside the AgentCore runtime for the same reason
        # ``_append_turn_to_transcript`` is — the transcript is a one-process
        # write target, and a second process's appends can be silently dropped
        # by the other's atomic rename. For relay turns the coordinator's drain
        # writes this marker instead.
        await _persist_turn_failure_marker(project_id, turn)
        # Guarantee a terminal status before notifying the queue. This is
        # the single chokepoint every turn exits through, so enforcing it
        # here closes the wedge at the root rather than at each cancel
        # call site. A turn can reach ``finally`` still marked "running"
        # when it's torn down by ``asyncio.CancelledError`` — which is a
        # ``BaseException`` in py3.11, so it bypasses the ``except
        # Exception`` above and neither ``complete()`` nor ``fail()``
        # runs. (An external ``client.interrupt()`` issued without first
        # setting a status produces exactly this.)
        # Left unset, the turn stays "running" forever; ``get_active_turn``
        # keeps returning it and ``_try_process_queue`` refuses to start
        # any new turn, wedging the project's chat queue until a restart.
        if turn.status == "running":
            turn.status = "failed"
            logger.info(
                "[turn %s] Reached finally still 'running' (torn down "
                "without a terminal transition — typically an external "
                "interrupt); marking failed so the queue unblocks.",
                turn.turn_id[:8],
            )
        # Release the one-writer-per-transcript session claim (the DDB claim)
        # if this turn holds it. Completion/orchestrator turns acquire the claim
        # when relay is enabled (serializes against relay user turns
        # cross-process). Release on ALL exits (success, error, cancel) so a
        # crashed turn never wedges the project. Best-effort: logs and continues
        # on store errors (matches relay client's finally-release discipline).
        if turn._holds_ddb_claim:
            try:
                from app.services.agentcore_session_registry import release_turn
                owner_id = turn.owner_id or _resolve_owner_id(project_id)
                # Fenced on this turn — a release that outlives its own claim
                # would otherwise clear whichever turn holds it now.
                await release_turn(owner_id, project_id, turn_id=turn.turn_id)
                logger.debug(
                    "[turn %s] Released DDB claim for (user=%s, project=%s)",
                    turn.turn_id[:8],
                    owner_id[:8],
                    project_id[:12],
                )
            except Exception as e:
                logger.warning(
                    "[turn %s] DDB claim release failed (best-effort): %s",
                    turn.turn_id[:8],
                    e,
                )
        # Unconditionally notify the turn queue so the next queued turn
        # starts regardless of how this turn ended (success, error, cancel).
        get_agent_turn_tracker().notify_turn_complete(project_id)


def start_agent_turn(
    project_id: str,
    prompt: str,
    coordinator_prompt: str,
    session_id: str | None = None,
    model: str | None = None,
    origin: str = "user",
    turn_id: str | None = None,
) -> AgentTurn:
    """Create an agent turn and spawn the background task.

    Orchestrator-origin turns have their prompt prefixed with a
    machine-readable marker so they can be filtered from chat history
    on page reload (they should never render as user bubbles).

    Args:
        project_id: Project UUID.
        prompt: User message text.
        coordinator_prompt: Coordinator agent's system instructions with project context.
        session_id: Optional session ID to resume.
        model: Optional model override.
        origin: Who initiated this turn — "user" or "orchestrator".
        turn_id: Reuse an id the caller already committed to. See
            ``AgentTurnTracker.create_turn``.

    Returns:
        The new AgentTurn (background task is already running).
    """
    from app.services.chat_message_parser import ORCHESTRATOR_MSG_PREFIX

    tracker = get_agent_turn_tracker()
    # ``create_turn`` resolves ``owner_id`` from project metadata so
    # the per-turn replay provider is bound to the right tenant; raises
    # if the project has no resolvable owner.
    turn = tracker.create_turn(project_id, origin=origin, turn_id=turn_id)

    # Tag orchestrator messages so chat history can filter them out
    actual_prompt = prompt
    if origin == "orchestrator":
        actual_prompt = ORCHESTRATOR_MSG_PREFIX + prompt

    attach_error_sink(
        asyncio.create_task(
            run_agent_turn(turn, actual_prompt, project_id, coordinator_prompt, session_id, model),
            name=f"agent-turn-{turn.turn_id[:8]}",
        ),
        project_id=project_id,
        resource="agent_turn",
    )

    # Notify frontend so it can connect to orchestrator turns live.
    # User turns don't need this — the frontend connects immediately
    # after POST /chat returns the turn_id.
    if origin == "orchestrator":
        owner_id = turn.owner_id
        if owner_id:
            # ``ws_hub.publish_project`` is async; this function is sync
            # (called from sync REST handlers and background scheduling
            # paths). Schedule the publish on the running loop so the
            # caller doesn't have to await it.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(get_ws_hub().publish_project(project_id, owner_id, {
                    "resource": "agent_turn",
                    "type": "started",
                    "turn_id": turn.turn_id,
                    "origin": origin,
                }))
            except RuntimeError:
                # No running loop (sync test context) — drop the
                # notification; the orchestrator-turn UI is best-effort.
                pass

    return turn


def _canonical_session_id(session_id: str) -> str:
    """Return ``session_id`` in the canonical dashed UUID form.

    The CLI validates session ids as strict UUIDs: ``--session-id`` with the
    dashless ``uuid4().hex`` form exits 1 ("Invalid session ID. Must be a
    valid UUID."). Dashless ids can only be relay-minted ghosts — the CLI
    reports dashed ids, so real sessions were always persisted dashed.
    Anything that doesn't parse as a UUID is returned unchanged (fail-open;
    the CLI is the authority on rejecting it).
    """
    try:
        import uuid as _uuid

        return str(_uuid.UUID(session_id))
    except (ValueError, AttributeError, TypeError):
        return session_id


async def _await_project_dir(project_dir, *, attempts: int = 4, delay_s: float = 1.5) -> None:
    """Wait briefly for ``project_dir`` to become visible on this mount.

    First cloud turn on a brand-new project can race EFS visibility: the
    coordinator created the directory, but the microVM's NFS client may hold
    a cached negative lookup, so the SDK's cwd check fails with ``Working
    directory does not exist`` while a retry seconds later succeeds. Listing
    the PARENT forces a fresh readdir, which
    revalidates the dentry. Bounded and silent when the dir is already there
    (the overwhelmingly common case); logs each retry when it isn't. Gives
    up quietly after the last attempt — the SDK's own cwd check then raises
    the same clear error as before.
    """
    for attempt in range(attempts):
        def _visible() -> bool:
            try:
                # Parent readdir first: busts a cached negative dentry for
                # the child on NFS in a way a bare stat may not.
                os.listdir(str(project_dir.parent))
            except OSError:
                pass
            return project_dir.is_dir()

        if await asyncio.to_thread(_visible):
            return
        logger.warning(
            "[project-dir] %s not visible yet (attempt %d/%d) — waiting %.1fs",
            project_dir, attempt + 1, attempts, delay_s,
        )
        await asyncio.sleep(delay_s)


async def _apply_session_identity(
    opts: dict, session_id: str, project_dir
) -> None:
    """Set ``resume`` or ``session_id`` on the SDK opts, from store evidence.

    ``resume`` when the session object exists in the store the SDK will read;
    ``session_id`` (create-with-id, ``--session-id``) when it does not — the
    id was minted by start-of-turn persistence for a session that has never
    run, and the SDK must ADOPT it rather than resume it. Key shape matches
    ``chat_history`` (``project_key_for_directory(cwd)`` + session id), and
    the existence probe is a prefix LIST via the store's own client — never a
    full transcript load.

    Fail-open to ``resume``: if the probe itself errors (S3 blip), an existing
    session still resumes, and a truly absent one fails the way a plain resume
    would, no worse.

    The id is normalized to the canonical dashed UUID form first. Dashless
    32-hex ids exist only as relay-minted ghosts (the CLI reports dashed ids
    in ResultMessage, so every REAL session's persisted id is dashed), and
    the CLI rejects dashless ``--session-id`` outright ("Invalid session ID.
    Must be a valid UUID."). Normalizing at the decision point also repairs a
    dashless id that is already persisted.
    """
    session_id = _canonical_session_id(session_id)
    store = opts.get("session_store")
    if store is None:
        # Dev/SDK-default on-disk store: ids are only persisted after a
        # successful turn there (the relay's start-of-turn persistence is
        # cloud-only), so resume semantics are already correct.
        opts["resume"] = session_id
        return
    try:
        from claude_agent_sdk import project_key_for_directory

        prefix = f"{project_key_for_directory(str(project_dir))}/{session_id}/"

        def _exists() -> bool:
            resp = store.s3.list_objects_v2(
                Bucket=store.bucket, Prefix=prefix, MaxKeys=1
            )
            return bool(resp.get("Contents"))

        exists = await asyncio.to_thread(_exists)
    except Exception as exc:  # noqa: BLE001 — probe failure must not change behavior
        logger.warning(
            "[session-identity] existence probe failed (%s) — defaulting to "
            "resume for session %s", exc, session_id[:12],
        )
        exists = True

    if exists:
        opts["resume"] = session_id
    else:
        logger.info(
            "[session-identity] session %s has no stored object — adopting "
            "as NEW session (create-with-id) instead of resuming a ghost",
            session_id[:12],
        )
        opts["session_id"] = session_id


async def _persist_session(project_id: str, session_id: str) -> None:
    """Persist session_id to the project's config.json.

    Called after each successful turn so the next message can resume
    the conversation via resume=session_id.

    Args:
        project_id: Project UUID.
        session_id: Session ID from the ResultMessage.
    """
    from app.main import get_project_manager
    try:
        pm = get_project_manager()
        project = pm.get_project(project_id)
        if project:
            project.session_id = session_id
            pm.save_config(project)
            logger.info("Session %s saved for project %s", session_id[:12], project_id[:12])
    except Exception as exc:
        logger.warning("Failed to persist session: %s", exc)


def turn_failure_marker_text(turn: AgentTurn) -> str | None:
    """The sentence a failed turn should leave behind, or ``None`` if it shouldn't.

    Three cases produce nothing:

    * a turn that completed — there is a real reply to read;
    * a turn the USER stopped — the cancel path writes its own "stopped" marker,
      and a second "failed" marker would contradict it;
    * a turn that failed with no recorded reason at all, where a bare "this
      failed" bubble adds a claim without adding information. The status is
      already visible in the UI; inventing prose for it is not an improvement.

    Prefers the CLI's own wording (``_publish_cli_error`` stashes it) over the
    turn's ``ResourceError``, because the CLI text is the actionable one — "Your
    session has expired. Please reauthenticate." beats a generic code.

    Args:
        turn: The turn being torn down.

    Returns:
        A plain sentence for :func:`chat_transcript.build_turn_marker_message`,
        or ``None`` when no marker should be written.
    """
    if turn.status == "completed":
        return None
    if getattr(turn, "_stopped_by_user", False):
        return None
    stashed = getattr(turn, "_failure_text", None)
    if isinstance(stashed, str) and stashed.strip():
        return f"Turn failed: {stashed.strip()}"
    error = getattr(turn, "error", None)
    message = getattr(error, "message", None) if error is not None else None
    if isinstance(message, str) and message.strip():
        hint = getattr(error, "hint", None)
        detail = " ".join(
            part for part in (message.strip(), hint if isinstance(hint, str) else None)
            if part
        )
        return f"Turn failed: {detail}"
    return None


async def _persist_turn_failure_marker(project_id: str, turn: AgentTurn) -> None:
    """Append a durable failure marker for a turn that ended without a reply.

    Best-effort by design: a transcript write must NEVER change how a turn ends
    (same discipline as ``_persist_session`` and ``_append_turn_to_transcript``),
    and this runs inside the turn's ``finally``, where an exception would mask
    whatever actually killed the turn.

    Args:
        project_id: Project UUID whose transcript to append to.
        turn: The turn being torn down.
    """
    try:
        text = turn_failure_marker_text(turn)
        if not text:
            return
        from app.platform.process_role import is_agentcore_runtime

        if is_agentcore_runtime():
            # See the call site: the coordinator is the single writer for relay.
            return
        from app.services.chat_transcript import (
            append_messages,
            build_turn_marker_message,
        )

        await append_messages(project_id, [
            build_turn_marker_message(turn.turn_id, text, kind="failed")
        ])
    except Exception:  # noqa: BLE001 — best-effort, and we are inside finally
        logger.warning(
            "[turn %s] Failed to persist the turn-failure marker",
            turn.turn_id[:8],
            exc_info=True,
        )


async def _append_turn_to_transcript(
    project_id: str,
    session_id: str,
    *,
    final_answer: str | None = None,
    turn_id: str | None = None,
) -> None:
    """Append the just-finished turn to the durable transcript (best-effort).

    Reads the FULL raw SDK session (not the windowed accessor) so the assistant turn
    keeps its trailing answer text and message IDs remain stable across turns. Mirrors
    the proven read-path backfill in get_chat_history.

    WHY raw session reader: The windowed ``load_session_messages`` truncates the assistant
    turn's trailing text block and emits unstable IDs as windowing shifts boundaries
    across turns. The raw reader ``load_raw_session_entries`` captures the full lineage,
    ensuring the durable transcript matches what the SDK actually recorded.

    The transcript cursor makes the append idempotent — older turns already written
    are skipped, only new ones land.

    Turn-identity predicate: checks whether THIS turn's reply has reached the session
    yet (structural comparison of entry order and uuids). The mapped tail's role is
    insufficient because hidden prompts (orchestrator/completion turns) are filtered
    by the mapper, so the newest mapped message can belong to a previous turn.
    Reconcile runs only when the reply is present, so the current turn's answer is
    never appended to another turn's message. Backfill is scheduled only when the
    reply is absent, fixing the hidden-prompt whole-message race.

    This is best-effort: a transcript-write failure must NEVER kill the turn
    (same discipline as ``_persist_session``'s except clause).

    Args:
        project_id: Project UUID.
        session_id: Session ID from the ResultMessage.
        final_answer: The SDK's canonical final answer text (``ResultMessage.result``),
            used to backfill the trailing answer when the session re-read missed it
            due to the flush-ordering race. See ``_reconcile_trailing_answer``.
        turn_id: Turn ID for the ordering guard in backfill (prevents appending
            behind a newer turn's provisional bubble).
    """
    try:
        from app.platform import get_platform
        from app.platform.chat_history import load_raw_session_entries
        from app.services.chat_message_parser import (
            find_last_prompt_uuid,
            map_session_messages,
            session_has_reply_after_prompt,
        )
        from app.services.chat_transcript import (
            append_messages,
            tag_turn_on_trailing_reply,
        )
        from app.services.project_paths import resolve_project_dir

        # Use the same cwd + session_store as get_chat_history (load-bearing)
        cwd = str(resolve_project_dir(project_id))
        session_store = get_platform().session_store

        entries = await load_raw_session_entries(
            session_id, cwd=cwd, session_store=session_store
        )
        mapped = map_session_messages(entries)

        # Resolve this turn's prompt uuid and check if the reply has flushed.
        # The mapped tail's role cannot answer this question because the mapper
        # drops hidden prompts, so the newest mapped message can belong to the
        # PREVIOUS turn.
        prompt_uuid = find_last_prompt_uuid(entries)
        reply_flushed = session_has_reply_after_prompt(entries, prompt_uuid)

        # Reconcile the trailing answer only when this turn's reply is present.
        # Against a mapping that contains nothing of this turn, appending the
        # answer would attach it to another turn's message, and the transcript's
        # body-immutability makes that permanent.
        if reply_flushed:
            _reconcile_trailing_answer(mapped, final_answer)
            # Same gate, same tail: only stamp the turn id once this turn's
            # reply is actually the trailing message. Both write paths reach
            # here — the in-process one and the coordinator-side relay — so
            # tagging at this single chokepoint keeps the field from depending
            # on which process wrote the reply.
            tag_turn_on_trailing_reply(mapped, turn_id)
        await append_messages(project_id, mapped)

        # Flush-ordering race, WHOLE-MESSAGE case: the SDK can deliver
        # ResultMessage before flushing the turn's final assistant entry to
        # the session file. The question is: has THIS turn's reply reached the
        # session yet? The mapped tail's role cannot answer it because the
        # mapper drops hidden prompts (orchestrator/completion turns), so the
        # newest mapped message can belong to the PREVIOUS turn.
        #
        # The predicate separates three shapes:
        # - Reply present and complete (trailing answer text flushed)
        # - Reply present but its trailing text block not yet flushed (partial)
        # - Reply entirely absent (whole-message race)
        #
        # Reconcile is gated on the same predicate: against a mapping that
        # contains nothing of this turn, appending the answer would attach it
        # to another turn's message, and the transcript's body-immutability
        # makes that permanent.
        #
        # The backfill is detached rather than inline: the turn's critical
        # path must not wait on timers (the ``result`` envelope is emitted
        # after this call), and ``append_messages`` is idempotent so the
        # backfill can safely re-run the whole mapping.
        #
        # Residual: an unflushed PROMPT entry leaves the predicate unable to
        # see this turn at all. The prompt entry is written at turn start, so
        # that window is orders of magnitude narrower than the reply-flush
        # window. The next turn's append remains the fallback for that case.
        if not reply_flushed and final_answer and final_answer.strip():
            attach_error_sink(
                asyncio.create_task(
                    _backfill_transcript_when_flushed(
                        project_id, session_id, cwd, session_store, final_answer,
                        turn_id=turn_id,
                        prompt_uuid=prompt_uuid,
                    ),
                    name=f"transcript-backfill-{project_id[:8]}",
                ),
                project_id=project_id,
                resource="transcript_backfill",
            )
    except Exception as exc:
        logger.warning("Failed to append turn to transcript for %s: %s", project_id[:12], exc)


async def _backfill_transcript_when_flushed(
    project_id: str,
    session_id: str,
    cwd: str,
    session_store,
    final_answer: str,
    *,
    turn_id: str | None = None,
    prompt_uuid: str | None = None,
) -> None:
    """Detached bounded backfill for the whole-message flush race (best-effort).

    Re-reads the session until THIS turn's reply appears, then appends. Runs
    OFF the turn's critical path so the ``result`` envelope is never delayed
    and environments without a persistent loop (TestClient) never hang on the
    sleeps. Bounded: gives up after ~5s and leaves the existing next-turn
    backfill as the fallback.

    Readiness is anchored on this turn's prompt uuid. The previous behavior
    finished on attempt 1 against a stale mapping for hidden-prompt turns
    (the mapped tail was the previous turn's reply, not this turn's), which
    is why the anchor is passed in rather than re-derived as "the newest
    prompt".

    ORDERING GUARD, two complementary signals. Another turn writing to the
    transcript around this task makes our append a misordering, and
    ``append_messages`` bodies are immutable, so it is permanent.

    1. TAIL IS A DIFFERENT TURN'S PROVISIONAL BUBBLE. Catches a newer turn that
       had already started when we were scheduled — there is nothing to compare
       against for that one, only the shape gives it away.

    2. TAIL ID CHANGED SINCE WE WERE SCHEDULED. Catches a newer turn that starts
       AND FINISHES inside the 5s window, which signal 1 cannot see: that turn's
       provisional bubble is upgraded to a real user message and its reply
       appended, so the tail no longer carries the provisional prefix and the
       shape check waves us through. Then
       ``session_has_reply_after_prompt`` is satisfied by THAT turn's reply — it
       asks "is there a real assistant entry after our prompt", which a newer
       turn's reply also answers — and if that reply ends on a tool call with no
       trailing text, ``_reconcile_trailing_answer`` glues OUR answer onto the
       newer turn's message. Comparing ids is stable under the tail-REWRITE path,
       since a rewrite keeps the record's id.

    Neither subsumes the other: signal 1 sees a pre-existing writer, signal 2
    sees an arriving one. The trade on signal 2 is deliberate — any unrelated
    transcript write in the window now costs us this backfill, which is the cheap
    direction, because a skipped backfill is repaired by the next turn's
    full-session append while a misordered body is not repairable at all.
    """
    from app.platform.chat_history import load_raw_session_entries
    from app.services.chat_message_parser import (
        map_session_messages,
        session_has_reply_after_prompt,
    )
    from app.services.chat_transcript import (
        PROVISIONAL_USER_ID_PREFIX,
        append_messages,
        load_transcript,
        tag_turn_on_trailing_reply,
    )

    def _tail_id() -> str:
        transcript = load_transcript(project_id)
        return str(transcript[-1].get("id", "")) if transcript else ""

    def _newer_turn_provisional_is_tail(tail_id: str) -> bool:
        return (
            tail_id.startswith(PROVISIONAL_USER_ID_PREFIX)
            and tail_id != f"{PROVISIONAL_USER_ID_PREFIX}{turn_id}"
        )

    # Snapshot at scheduling time: everything our own turn was going to write
    # has already been written by the caller's ``append_messages``.
    tail_at_start = _tail_id()

    try:
        for attempt in range(10):
            await asyncio.sleep(0.5)
            tail_now = _tail_id()
            if _newer_turn_provisional_is_tail(tail_now) or tail_now != tail_at_start:
                logger.info(
                    "Transcript backfill for %s: another turn owns the "
                    "transcript tail (at schedule=%s, now=%s) — bailing, next "
                    "turn's append repairs",
                    project_id[:12],
                    tail_at_start[:32] or "<empty>",
                    tail_now[:32] or "<empty>",
                )
                return
            entries = await load_raw_session_entries(
                session_id, cwd=cwd, session_store=session_store
            )
            # Readiness: has THIS turn's reply flushed to the session?
            if session_has_reply_after_prompt(entries, prompt_uuid):
                mapped = map_session_messages(entries)
                _reconcile_trailing_answer(mapped, final_answer)
                tag_turn_on_trailing_reply(mapped, turn_id)
                await append_messages(project_id, mapped)
                logger.info(
                    "Transcript backfill for %s: this turn's reply landed on "
                    "attempt %d",
                    project_id[:12], attempt + 1,
                )
                return
        logger.warning(
            "Transcript backfill for %s: this turn's reply never flushed within "
            "budget — next turn's append will repair",
            project_id[:12],
        )
    except Exception as exc:
        logger.warning(
            "Transcript backfill failed for %s: %s", project_id[:12], exc
        )


def _reconcile_trailing_answer(mapped: list[dict], final_answer: str | None) -> None:
    """Backfill the final answer text lost to the session flush-ordering race.

    The SDK can deliver ``ResultMessage`` before it flushes the turn's final text
    block to the session file, so re-reading the session (``map_session_messages``)
    yields an assistant turn that ends on a tool with no trailing answer. The durable
    transcript would then persist that truncated turn — and because the tail-rewrite
    repair in ``chat_transcript.append_messages`` only runs on a SUBSEQUENT turn, the
    LAST turn of a conversation stays truncated until the user sends another message
    (the answer vanishes on reload).

    ``ResultMessage.result`` is the SDK's canonical final answer for the turn and is
    in hand at the append call site. When the just-finished turn (``mapped[-1]``) has
    no trailing text part — text after its last thinking/tool step — but we have a
    non-empty final answer, append it as the trailing text. This fires ONLY in the
    race case: if the session flushed in time the mapped turn already ends with the
    answer and we leave it untouched, so there is no duplication. Only the tail
    (current) turn is reconciled; earlier turns are body-immutable on append and were
    already repaired by their own following turn.

    Mutates the last assistant message in ``mapped`` in place.

    Args:
        mapped: ChatMessage dicts from ``map_session_messages`` (mutated in place).
        final_answer: ``ResultMessage.result`` for the just-finished turn, or None.
    """
    if not final_answer or not final_answer.strip():
        return
    if not mapped:
        return
    last = mapped[-1]
    if last.get("role") != "assistant":
        return
    parts = last.get("parts") or []
    # Index of the last reasoning/tool step; any text after it is the trailing answer.
    last_step_idx = -1
    for i, part in enumerate(parts):
        if part.get("type") in ("thinking", "tool"):
            last_step_idx = i
    has_trailing_text = any(
        part.get("type") == "text" for part in parts[last_step_idx + 1 :]
    )
    if has_trailing_text:
        return
    parts.append({"type": "text", "text": final_answer})
    last["parts"] = parts
