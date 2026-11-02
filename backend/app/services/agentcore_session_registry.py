# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore session registry helper — resolve-once claim/release + durable resume.

Thin wrapper over the MetadataStore session-registry primitives, keeping
the chat.py and agentcore_relay_client.py thin and honoring the resolve-once-and-thread
discipline (arch_split.py pattern).

Provides four resolve-once helpers for the relay path:
  - claim_turn(user_id, project_id) — first-wins claim (returns True/False)
  - release_turn(user_id, project_id) — idempotent release (best-effort, never raises)
  - resolve_resume(user_id, project_id, runtime_session_id) — durable resume-id resolution
  - record_session(user_id, project_id, *, runtime_session_id, app_session_id) — persist mapping

The claim grain is per-(user, project), not per-user. Each project gets its
own AgentCore microVM + turn claim — a user can have multiple projects claimed concurrently.
This matches the existing per-project turn serialization in AgentTurnTracker._by_project.

The durable registry is the PRIMARY one-turn-at-a-time serializer: AgentCore
provides no native turn serialization.
"""
import logging
import time

from app.platform import get_platform

logger = logging.getLogger(__name__)

# Stale-claim backstop TTL (in seconds). Acts as a backstop if the release-on-all-exits
# discipline fails (crashed relay task / unhandled Exception). The explicit release is
# the PRIMARY mechanism; this TTL is belt-and-suspenders only.
#
# Chosen conservatively at 2 hours to accommodate long-running agent turns without
# blocking legitimate next-turn invokes. Shorter than the eval-lock's 24h (evals run
# far longer than agent turns).
AGENTCORE_SESSION_TTL_SECONDS = 2 * 3600  # 2 hours


async def claim_turn(user_id: str, project_id: str, *, turn_id: str | None = None) -> bool:
    """Atomically claim the (user, project) session for one turn (first-wins).

    Returns True iff this caller won the claim. A second concurrent claim on the
    SAME (user, project) while the first is held returns False. A claim on a
    DIFFERENT project for the SAME user returns True (per-(user,project) grain,
    not per-user).

    The claim is a conditional-write status idle→running on the durable session
    item (MetadataStore.claim_agentcore_session_turn). This is the PRIMARY
    one-turn-at-a-time serializer — AgentCore provides no
    native turn serialization.

    Bumps claim_version (monotonic fence) + updated_at on win. TTL acts as a
    backstop if the turn crashes without releasing.

    Args:
        user_id: User's Cognito sub.
        project_id: Project UUID.
        turn_id: Turn UUID being claimed (optional, for in-flight turn discovery).

    Returns:
        True if this caller won the claim, False if already held.
    """
    ttl = int(time.time()) + AGENTCORE_SESSION_TTL_SECONDS
    store = get_platform().metadata_store
    won = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl, current_turn_id=turn_id)

    if won:
        logger.debug(
            "[registry] Claimed turn for (user=%s, project=%s, turn=%s)",
            user_id[:8],
            project_id[:8],
            turn_id[:8] if turn_id else "None",
        )
    else:
        logger.debug(
            "[registry] Turn already held for (user=%s, project=%s) — returning False",
            user_id[:8],
            project_id[:8],
        )

    return won


async def release_turn(
    user_id: str, project_id: str, *, turn_id: str | None = None
) -> None:
    """Release the (user, project) session turn claim (idempotent, best-effort).

    Flips status running→idle. Idempotent: swallows ConditionalCheckFailed when
    not running / item gone. Never deletes the item — the mapping persists for
    next-turn warm affinity + resume.

    This is designed to be called in a finally block on the relay task's exit
    (success / failure / cancel) to ensure the claim is always released. Best-effort:
    logs and returns on store errors (never raises), matching the relay's
    non-raising discipline.

    Pass ``turn_id``. Because this runs from a ``finally``, a slow or already-
    superseded turn can reach it long after its lease expired and another turn
    claimed the project — and an unfenced release would then clear the new turn's
    live claim, which is the opposite of what a claim is for.

    Args:
        user_id: User's Cognito sub.
        project_id: Project UUID.
        turn_id: The releasing turn. Only releases if it still holds the claim.
    """
    try:
        store = get_platform().metadata_store
        await store.release_agentcore_session_turn(
            user_id, project_id, expected_turn_id=turn_id
        )
        logger.debug(
            "[registry] Released turn for (user=%s, project=%s)",
            user_id[:8],
            project_id[:8],
        )
    except Exception as exc:
        # Best-effort: store errors on release are non-fatal. The TTL backstop
        # will clear the claim eventually. Log and continue so the relay task
        # completes normally.
        logger.warning(
            "[registry] Failed to release turn for (user=%s, project=%s): %s",
            user_id[:8],
            project_id[:8],
            exc,
        )


async def resolve_resume(
    user_id: str,
    project_id: str,
    runtime_session_id: str | None,
) -> tuple[str | None, str | None]:
    """Resolve durable resume ids for AgentCore turn invocation.

    Returns (runtime_session_id_to_use, app_session_id_to_resume). Reads the
    durable session item; if present AND its app_session_id is non-empty, returns
    its stored ids. If the item exists but app_session_id is empty (the claim
    placeholder), falls back to the project's persisted session_id (the
    resume proof — a turn resumes from the S3 transcript). If absent, also falls
    back to the project's persisted session_id and the caller-supplied
    runtime_session_id.

    This handles the case: claim_agentcore_session_turn creates the durable
    item with resume ids UNSET, so resolve_resume must not short-circuit the
    persisted-session fallback just because the item exists — it falls through when
    the stored app_session_id is empty.

    This supersedes the in-memory agentcore_session.resume_session_id_for on the
    RELAY path (control plane) — ports its persisted-session fallback logic to
    read the DURABLE registry first. The in-memory map stays in the repo
    (still used by agentcore_runtime.py, the serving adapter inside the microVM).

    Args:
        user_id: User's Cognito sub.
        project_id: Project UUID.
        runtime_session_id: Caller-supplied runtime session id (may be None).

    Returns:
        (runtime_session_id_to_use, app_session_id_to_resume). Both may be None
        for a fresh session.
    """
    store = get_platform().metadata_store
    session = await store.get_agentcore_session(user_id, project_id)

    if session is not None:
        # Durable mapping exists — return its stored ids ONLY if app_session_id is non-empty
        stored_runtime = session.get("runtime_session_id")
        stored_app = session.get("app_session_id")
        if stored_app:
            # Non-empty app_session_id — use the stored ids
            logger.debug(
                "[registry] Resolved durable session for (user=%s, project=%s): runtime=%s, app=%s",
                user_id[:8],
                project_id[:8],
                stored_runtime[:12] if stored_runtime else "None",
                stored_app[:12] if stored_app else "None",
            )
            return (stored_runtime, stored_app)
        # Item exists but app_session_id is empty (claim placeholder) — fall through to persisted-session fallback
        # Preserve stored_runtime if present (warm affinity), otherwise use caller-supplied
        logger.debug(
            "[registry] Durable session exists but app_session_id empty for (user=%s, project=%s), falling through to persisted session",
            user_id[:8],
            project_id[:8],
        )
        runtime_session_id = stored_runtime or runtime_session_id

    # No durable mapping or app_session_id empty — fallback to project's persisted session_id
    # (resume proof: returning the persisted session_id means the
    # turn will read the S3 transcript via resume=...)
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        if project is not None and project.session_id:
            logger.debug(
                "[registry] No durable app_session_id, falling back to persisted session %s for project %s",
                project.session_id[:12],
                project_id[:8],
            )
            return (runtime_session_id, project.session_id)
    except Exception as exc:
        logger.warning(
            "[registry] Failed to resolve persisted session_id for project %s: %s",
            project_id[:8],
            exc,
        )

    # No durable mapping and no persisted session — fresh session
    return (runtime_session_id, None)


async def record_session(
    user_id: str,
    project_id: str,
    *,
    runtime_session_id: str | None,
    app_session_id: str | None,
) -> None:
    """Durably persist the AgentCore session identity/resume mapping.

    Upserts the runtime_session_id and app_session_id fields on the (user, project)
    session item without touching status (recording the mapping mid-idle does NOT
    forge a claim).

    Called after successful agent turns to persist the mapping for next-turn warm
    affinity + transcript resume.

    Args:
        user_id: User's Cognito sub.
        project_id: Project UUID.
        runtime_session_id: AgentCore runtimeSessionId (UUID) or None.
        app_session_id: S3SessionStore session_id (UUID) or None.
    """
    ttl = int(time.time()) + AGENTCORE_SESSION_TTL_SECONDS
    store = get_platform().metadata_store
    await store.put_agentcore_session(
        user_id,
        project_id,
        runtime_session_id=runtime_session_id,
        app_session_id=app_session_id,
        ttl=ttl,
    )
    logger.debug(
        "[registry] Recorded session mapping for (user=%s, project=%s): runtime=%s, app=%s",
        user_id[:8],
        project_id[:8],
        runtime_session_id[:12] if runtime_session_id else "None",
        app_session_id[:12] if app_session_id else "None",
    )
