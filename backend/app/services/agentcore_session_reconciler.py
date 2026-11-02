# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore session reconciler — startup orphan-claim cleanup.

This module provides a one-shot startup reconciler that clears orphaned
AgentCore session claims (status="running") after a coordinator restart.

**The orphaned-claim invariant:** After a process restart, every in-process
relay task is dead (the asyncio event loop is gone). Therefore, any persisted
session claim with status="running" is definitionally orphaned — there is no
live relay task to eventually release it. The reconciler unconditionally clears
these claims at startup via version-fenced `clear_stale_agentcore_session_claim`.

**Version fence:** The clear is version-fenced so a claim
released+reclaimed between the list and the clear is NOT cleared (protecting
a legitimately-active turn from double-invoke). However, the restart-implies-orphaned
invariant means the version fence will never trip on a real startup run — it's
a belt-and-suspenders defense for testing and future concurrent scenarios.

**Two sweeps, two different safety arguments:**

- :func:`reconcile_orphaned_agentcore_claims` — ONE-SHOT at startup. Clears every
  running claim unconditionally, justified by the restart-implies-orphaned
  invariant above.
- :func:`reconcile_expired_agentcore_claims` — PERIODIC, on the eval_reconciler
  cadence. The startup invariant does NOT hold mid-run (a running claim usually
  belongs to a live turn), so this one clears only claims whose ``ttl`` has
  already passed. It exists because DynamoDB TTL deletion is best-effort and lags
  expiry by up to days (AWS-documented), so an expired-but-undeleted claim would
  otherwise wedge its (user, project) pair for that entire window on a
  long-lived coordinator — startup-only recovery never sees it.

**Never blocks startup:** All exceptions are swallowed and logged. A reconcile
failure never prevents the coordinator from serving traffic.

**Dev vs cloud:** In dev mode the in-memory dict is empty on startup (dev orphans
cannot survive a restart), but `reconcile_orphaned_agentcore_claims` runs
identically for Protocol parity and testability.
"""

import logging
import time
from typing import TypedDict

from app.platform import get_platform

logger = logging.getLogger(__name__)


class ReconcileResult(TypedDict):
    """Result of a one-shot reconcile pass."""

    claims_cleared: int


async def reconcile_orphaned_agentcore_claims() -> ReconcileResult:
    """Clear orphaned AgentCore session claims at startup.

    After a process restart, every AgentCore relay task is dead (the asyncio
    event loop is gone), so any persisted claim with status="running" is
    definitionally orphaned. This pass lists running sessions across all users
    and version-fences clears them.

    **Why unconditional clear is safe:** The reconciler runs ONLY at startup,
    so there is no live in-flight turn to protect — every running claim at
    startup time is an orphan. The version fence prevents clearing a claim
    released+reclaimed mid-reconcile (concurrent test scenario), but on a real
    startup the fence will never trip because no relay task exists to reclaim.

    **Error handling:** Swallows all exceptions and returns partial count so
    a reconcile failure never blocks startup.

    Returns:
        Count of claims cleared (claims_cleared).
    """
    result: ReconcileResult = {"claims_cleared": 0}
    store = get_platform().metadata_store

    try:
        # List all running sessions across users (a GSI1 Query on the sparse
        # running-claims index in DDB)
        running_sessions = await store.list_running_agentcore_sessions()

        for session in running_sessions:
            user_id = session["user_id"]
            project_id = session["project_id"]
            claim_version = session.get("claim_version")

            if claim_version is None:
                logger.warning(
                    "[reconciler] Running session (user=%s, project=%s) has no claim_version — skipping",
                    user_id[:8],
                    project_id[:8],
                )
                continue

            # Version-fenced clear: if the claim was released+reclaimed between
            # the list and now, the clear will fail (fence tripped). On a real
            # startup this never happens (no live relay task), but the fence
            # protects test and future concurrent scenarios.
            logger.info(
                "[reconciler] Clearing orphaned AgentCore claim (user=%s, project=%s, version=%s)",
                user_id[:8],
                project_id[:8],
                claim_version,
            )
            cleared = await store.clear_stale_agentcore_session_claim(
                user_id, project_id, expected_version=claim_version
            )
            if cleared:
                result["claims_cleared"] += 1
            else:
                # Fence tripped (version changed) — log but don't treat as error
                logger.info(
                    "[reconciler] Claim version mismatch for (user=%s, project=%s) — skipped clear",
                    user_id[:8],
                    project_id[:8],
                )

    except Exception as exc:  # noqa: BLE001
        # Never let an exception block startup
        logger.exception("[reconciler] AgentCore claim reconcile failed: %s", exc)

    # Log summary
    logger.info(
        "[reconciler] AgentCore claim reconcile complete: claims_cleared=%d",
        result["claims_cleared"],
    )

    return result


async def reconcile_expired_agentcore_claims() -> ReconcileResult:
    """Clear AgentCore session claims whose TTL has already passed (periodic).

    Runs on the eval_reconciler cadence, NOT at startup. The startup sweep's
    "a restart kills every relay task, so every running claim is an orphan"
    invariant does not hold mid-run — a running claim normally belongs to a live
    turn. So this sweep clears ONLY claims whose ``ttl`` is in the past.

    Be careful what that buys. An expired ttl does NOT prove the holder is gone.
    Nothing renews a claim mid-turn: the ttl is set once, at claim time, and the
    only other writers are the release and the end-of-turn mapping write. The
    lease is 2h, the relay's own turn cap is also 2h but measured from a strictly
    later instant, and the microVM's max_lifetime is 4h — so a long turn can
    outlive its claim and be swept while still running. The release is fenced on
    the turn id, which stops the swept turn from later clearing its successor's
    claim, but it does not stop the two from overlapping.

    **Why this is needed at all:** the TTL was treated as the backstop that bounds
    a leaked claim, but DynamoDB TTL deletion is a best-effort background process
    that AWS documents as "typically" deleting expired items within a few DAYS.
    An expired-but-undeleted claim therefore blocks its (user, project) pair for
    that whole window, and a startup-only reconciler never sees it on a
    long-lived coordinator. Paired with the store-level ``#ttl < :now`` claim
    condition, this makes claim recovery active rather than dependent on AWS's
    deletion schedule.

    **Version fence:** each clear is version-fenced, so a claim that was
    released and legitimately re-claimed between the list and the clear is left
    alone.

    **Error handling:** swallows all exceptions and returns a partial count so a
    reconcile failure never kills the periodic loop.

    Returns:
        Count of expired claims cleared (claims_cleared).
    """
    result: ReconcileResult = {"claims_cleared": 0}
    store = get_platform().metadata_store

    try:
        running_sessions = await store.list_running_agentcore_sessions()
        now = int(time.time())

        for session in running_sessions:
            user_id = session["user_id"]
            project_id = session["project_id"]
            claim_version = session.get("claim_version")
            claim_ttl = session.get("ttl")

            # Only expired claims are provably orphaned mid-run. A claim with no
            # ttl at all is left to the startup sweep rather than guessed at.
            if claim_ttl is None or int(claim_ttl) >= now:
                continue

            if claim_version is None:
                logger.warning(
                    "[reconciler] Expired claim (user=%s, project=%s) has no "
                    "claim_version — cannot fence the clear, skipping",
                    user_id[:8], project_id[:8],
                )
                continue

            logger.warning(
                "[reconciler] Clearing EXPIRED AgentCore claim (user=%s, "
                "project=%s, version=%s, expired %ds ago) — DynamoDB TTL deletion "
                "lags expiry, so the pair would stay wedged until the item is "
                "physically deleted",
                user_id[:8], project_id[:8], claim_version, now - int(claim_ttl),
            )
            cleared = await store.clear_stale_agentcore_session_claim(
                user_id, project_id, expected_version=claim_version
            )
            if cleared:
                result["claims_cleared"] += 1
            else:
                logger.info(
                    "[reconciler] Claim version changed for (user=%s, project=%s) "
                    "— re-claimed legitimately, skipping clear",
                    user_id[:8], project_id[:8],
                )

    except Exception as exc:  # noqa: BLE001
        # Never let an exception kill the periodic reconciler loop
        logger.exception(
            "[reconciler] Expired AgentCore claim reconcile failed: %s", exc
        )

    if result["claims_cleared"]:
        logger.info(
            "[reconciler] Expired AgentCore claim sweep cleared %d claim(s)",
            result["claims_cleared"],
        )

    return result
