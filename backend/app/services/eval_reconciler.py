# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval reconciler — triple-duty ECS↔DDB pass.

This module provides the in-process periodic reconciler that does three
operations in a single ECS `list_tasks` ↔ durable-job comparison:

1. **Stop orphaned workers**: ECS tasks with no matching non-terminal job.
2. **Clear stale locks + mark stuck jobs failed**: Non-terminal jobs with
   no visible worker after a grace period (stale-phantom reclaim). A job whose
   cancel is stopping its worker is left to the cancel, and one whose cancel
   gave up is marked cancelled rather than failed.
3. **Kill-switch aggressive StopTask-all**: When the cost kill switch is active,
   stop ALL running workers (emergency brake).

**ECS-race safety:** A job created less than the
``_RECONCILE_GRACE_SECONDS`` grace period ago whose worker is not yet
visible in `list_tasks` is NEVER marked failed and its lock is NEVER
cleared. This protects still-starting workers from being killed by the
reconciler (ECS `list_tasks` has eventual consistency — a worker in
PROVISIONING/PENDING may not appear for 5–15s).

**Stale-phantom reclaim:** The lock clear is fenced on BOTH the lock version
and the holding job id, so the reconciler only ever clears a lock the job it is
reconciling still holds. The version alone covers just the read→delete window;
a lock already reclaimed by a newer job presents its own current version, so a
version-only fence deleted a live job's lock and let a third eval start.

**Loop resilience:** The periodic task swallows non-cancel exceptions and
continues — no transient error can kill the loop.

**Dev mode:** In dev mode (non-Fargate compute), `reconcile_once` runs only the
duties that involve no detached worker (expired-claim sweep, event-outbox drain,
orphaned-loop adoption) and returns zeroed counts, because dev has no detached
workers to reconcile.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from app.platform import get_platform
from app.platform.arch_split import is_cost_kill_switch_active
from app.platform.aws.fargate_task_compute import FargateTaskCompute
from app.platform.job_compute import JobComputeHandle, get_job_compute
from app.services.eval_cancel import CANCEL_IN_FLIGHT_S

# Grace period before acting on "job with compute_handle but no visible worker"
# (allows for ECS list_tasks eventual consistency + Fargate launch latency).
# 60s covers both launch (~20–40s) and
# ECS visibility lag (~5–15s).
_RECONCILE_GRACE_SECONDS = 60

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

# Module-level task ref for the periodic loop
_reconciler_task: asyncio.Task | None = None

logger = logging.getLogger(__name__)


class ReconcileResult(TypedDict):
    """Result of a reconcile pass.

    `pass_failed`: True when the pass raised an exception (the blanket
    `except` at the bottom of `reconcile_once`). Disambiguates a clean
    zero-reap pass from a pass that aborted mid-execution, so operators
    can attribute `stuck_jobs_failed=0` correctly.
    """

    orphans_stopped: int
    stale_locks_cleared: int
    stuck_jobs_failed: int
    pass_failed: bool


def _reaped_state(record: dict[str, Any], now: int) -> str | None:
    """The terminal state for a job whose worker is gone, or None to leave it.

    A worker that vanished failed, unless a cancel stopped it. A cancel stops
    the worker before it marks the job cancelled, so a job whose cancel was
    requested recently is the cancel's to finish: closing it here would release
    the project's eval lock while the containers may still be up. An older
    request belongs to a cancel that gave up or died with its process, and the
    user still asked for the eval to stop.
    """
    requested_at = record.get("cancel_requested_at")
    if not requested_at:
        return "failed"
    try:
        requested = int(datetime.fromisoformat(requested_at).timestamp())
    except (TypeError, ValueError):
        return "cancelled"
    if now - requested <= CANCEL_IN_FLIGHT_S:
        return None
    return "cancelled"


async def reconcile_once() -> ReconcileResult:
    """Run one triple-duty reconcile pass (orphan-stop, stale-lock-clear, stuck-job-fail).

    This is the ECS sprawl-detection + cleanup pass the reconciler loop calls
    periodically. It does three operations in a single `list_running_workers()` ↔
    `list_non_terminal_jobs()` comparison:

    1. **Orphan workers**: tasks with no matching non-terminal job → StopTask.
    2. **Stale locks**: non-terminal jobs past grace with no visible worker →
       version-fenced lock clear + mark failed (or cancelled; see
       ``_reaped_state``).
    3. **Kill-switch aggressive**: when cost kill switch active, StopTask ALL
       workers (not just orphans).

    **Dev mode degrade**: If compute is not FargateTaskCompute (dev/local),
    return a zeroed result once the worker-independent duties (expired-claim
    sweep, event-outbox drain, orphaned-loop adoption) have run — dev has no
    detached workers.

    **ECS-race safety**: Jobs younger than ``_RECONCILE_GRACE_SECONDS`` are
    skipped even if their worker is not visible (protects still-starting workers).

    **Error handling**: Swallows all exceptions and returns partial counts so
    the periodic loop never dies.

    Returns:
        Counts of actions taken (orphans_stopped, stale_locks_cleared, stuck_jobs_failed).
    """
    result: ReconcileResult = {
        "orphans_stopped": 0,
        "stale_locks_cleared": 0,
        "stuck_jobs_failed": 0,
        "pass_failed": False,
    }

    # Expired AgentCore session claims. Deliberately BEFORE the dev-mode
    # early return below: this duty has nothing to do with detached workers, so
    # gating it on the compute backend would both couple unrelated concerns and
    # make it unreachable in dev (where the store Protocol behaves identically and
    # is where the behavior is tested).
    #
    # The AgentCore claim reconciler ran at startup only, but a claim that leaks on
    # a long-lived coordinator is bounded only by DynamoDB TTL deletion — which AWS
    # documents as best-effort, "typically within a few days". Sweeping expired
    # claims on this cadence makes recovery active instead of dependent on AWS's
    # deletion schedule. Own try/except so it can never disturb the eval-side
    # counters or kill the periodic loop.
    try:
        from app.services.agentcore_session_reconciler import (
            reconcile_expired_agentcore_claims,
        )

        await reconcile_expired_agentcore_claims()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[reconciler] Expired-claim sweep failed: %s", exc)

    # microVM event-outbox drain (closes the microVM transport gap).
    # Deliberately BEFORE the dev-mode early return (harmless in dev — the
    # outbox is simply empty) and with its own try/except: a failed drain must
    # not disturb the other duties. Unroutable microVM events (published with
    # no live turn to carry them) become LATE instead of LOST: stashed durably
    # by the publisher, delivered here through the real ws_hub.
    try:
        from app.services.project_event_outbox import drain_project_event_outboxes

        drained = await drain_project_event_outboxes()
        if drained:
            logger.info("[reconciler] Drained %d outboxed project event(s)", drained)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[reconciler] Event-outbox drain failed: %s", exc)

    # Orphaned-loop adoption (dispatch-to-owner seam). Deliberately BEFORE
    # the dev-mode early return: adoption is about asyncio tasks and durable
    # records, not detached Fargate workers, and dev restarts strand loops the
    # same way coordinator deploys do. Own try/except — a failed adoption pass
    # must not disturb the eval-side duties below.
    try:
        from app.services.optimization_loop_orchestrator import (
            adopt_orphan_optimization_loops,
        )

        adopted = await adopt_orphan_optimization_loops()
        if adopted:
            logger.info("[reconciler] Adopted %d orphaned optimization loop(s)", adopted)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[reconciler] Loop-adoption sweep failed: %s", exc)

    # Dev-mode degrade: only FargateTaskCompute has workers to reconcile
    compute = get_job_compute()
    if not isinstance(compute, FargateTaskCompute):
        logger.debug("[reconciler] Dev mode — no detached workers, skipping reconcile")
        return result

    store = get_platform().metadata_store

    try:

        # Kill-switch aggressive mode: stop ALL workers immediately (emergency brake)
        kill_switch_active = is_cost_kill_switch_active()
        if kill_switch_active:
            logger.warning("[reconciler] Cost kill switch ACTIVE — stopping ALL workers")
            all_tasks = await compute.list_running_workers()
            for task in all_tasks:
                task_arn = task["taskArn"]
                try:
                    await compute.cancel(JobComputeHandle(kind="fargate_task", value=task_arn))
                    result["orphans_stopped"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[reconciler] Failed to stop task %s: %s", task_arn[:40], exc
                    )
            # Kill-switch mode stops all workers; no further reconcile needed
            return result

        # Normal mode: triple-duty pass
        # 1. List running workers from ECS
        worker_tasks = await compute.list_running_workers()

        # Build job_id → task_arn map from tags
        job_to_task: dict[str, str] = {}
        orphan_candidate_arns: list[str] = []

        for task in worker_tasks:
            task_arn = task["taskArn"]
            tags = {t["key"]: t["value"] for t in task.get("tags", [])}
            job_id = tags.get("Job")

            if job_id:
                job_to_task[job_id] = task_arn
            else:
                # Task with no Job tag is an orphan candidate
                orphan_candidate_arns.append(task_arn)

        # 2. List non-terminal jobs from DDB
        jobs = await store.list_non_terminal_jobs()
        now = int(time.time())

        # Build set of job_ids with non-terminal records for orphan detection
        active_job_ids = {job["job_id"] for job in jobs}

        # 3. Correlate jobs → workers (stuck-job + stale-lock detection)
        for job in jobs:
            job_id = job["job_id"]
            compute_handle = job.get("compute_handle")

            # Handle-HAVING branch: a worker was assigned. If it's still visible
            # in ECS the job is healthy — skip. If not, fall through to the
            # shared past-grace reap below.
            #
            # Handle-LESS branch: previously blanket-`continue`d, which is why
            # handle-less non-terminal EVAL jobs leaked forever against the caps
            # (no TTL on job items). A handle-less EVAL past grace is a true
            # zombie — an eval that died before its handle write, or a record
            # whose handle a worker's full-item rewrite wiped (workers now
            # register with ``create_job(durable_write=False)``). Reap it. The
            # age/grace guard is the sole protection for this branch (there is
            # no worker to check visibility against): a job within grace is
            # the legitimate brief queued→dispatched window and is left
            # untouched.
            #
            # CRITICAL: the reap is scoped to type=="eval" ONLY. Other job types
            # are handle-less BY DESIGN and legitimately run past the 60s grace —
            # the optimization_iteration parent launches no worker (its child
            # eval does) and runs for minutes; a harness_build job is in-process.
            # Reaping those on the grace timer would mark the parent failed +
            # clear the eval lock mid-run, false-failing every live optimization.
            # Only evals are EXPECTED to carry a compute_handle, so only a
            # handle-less eval is a zombie.
            if compute_handle:
                if job_id in job_to_task:
                    # Worker found — healthy, skip
                    continue
                # Worker NOT visible — fall through to the past-grace reap
                # (applies to any type whose worker vanished).
            elif job.get("type") != "eval":
                # Handle-less by design (optimization_iteration parent,
                # harness_build) — never a grace-timer zombie. Skip.
                continue

            # Grace check (ECS-race safety for the handle-having branch; and the
            # queued→dispatched-window guard for the handle-less branch).
            created_at_iso = job.get("created_at")
            if not created_at_iso:
                logger.warning(
                    "[reconciler] Job %s has no created_at — skipping", job_id[:8]
                )
                continue

            try:
                created_at = datetime.fromisoformat(created_at_iso)
                job_age = now - int(created_at.timestamp())
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "[reconciler] Job %s has invalid created_at (%s): %s",
                    job_id[:8],
                    created_at_iso,
                    exc,
                )
                continue

            if job_age <= _RECONCILE_GRACE_SECONDS:
                # Within grace period — skip (worker may be PROVISIONING/PENDING,
                # or a handle-less job may be mid queued→dispatched).
                logger.debug(
                    "[reconciler] Job %s age=%ds within grace — skipping",
                    job_id[:8],
                    job_age,
                )
                continue

            # Past grace — worker stuck/dead (handle-having) or a handle-less
            # zombie leaking against the caps. Both take the same reap.
            owner_id = job.get("owner_id") or job.get("user_id")
            project_id = job["project_id"]

            # Decide on a strong read of the record, not on the listing. The
            # listing comes from an eventually consistent index, and the reap
            # cannot be taken back: acting on a stale row overwrites a terminal
            # state that has landed since, or misses a cancel request written
            # just before the worker left ECS's running list.
            current = await store.get_job(owner_id, project_id, job_id, consistent=True)
            if current is None or current.get("state") in _TERMINAL_STATES:
                continue
            reaped_state = _reaped_state(current, now)
            if reaped_state is None:
                logger.info(
                    "[reconciler] Job %s (proj=%s) is being cancelled — leaving it "
                    "to the cancel",
                    job_id[:8],
                    project_id[:8],
                )
                continue

            logger.warning(
                "[reconciler] Job %s (proj=%s) past grace=%ds, %s — marking %s",
                job_id[:8],
                project_id[:8],
                job_age,
                "worker not found" if compute_handle else "handle-less zombie",
                reaped_state,
            )

            # (a) Clear the stale eval lock, fenced on version AND holder.
            #
            # The holder check is what makes this safe. A lock released by this
            # stale job and then RECLAIMED by a newer one presents its own
            # current version, so a version-only fence matched and deleted a
            # live job's lock — after which a third eval could start against the
            # same project. Only clear a lock this job still holds.
            lock = await store.get_eval_lock(owner_id, project_id)
            if lock:
                lock_version = lock.get("lock_version")
                lock_holder = lock.get("lock_holder_job_id")
                if lock_holder != job_id:
                    logger.info(
                        "[reconciler] Lock for proj=%s is held by %s, not the stale "
                        "job %s — leaving it alone",
                        project_id[:8],
                        str(lock_holder)[:8] if lock_holder else "nobody",
                        job_id[:8],
                    )
                elif lock_version is not None:
                    cleared = await store.clear_stale_eval_lock(
                        owner_id,
                        project_id,
                        expected_version=lock_version,
                        expected_holder_job_id=job_id,
                    )
                    if cleared:
                        result["stale_locks_cleared"] += 1
                        logger.info(
                            "[reconciler] Cleared stale lock for proj=%s (version=%s)",
                            project_id[:8],
                            lock_version,
                        )
                    else:
                        logger.info(
                            "[reconciler] Lock for proj=%s changed hands or version "
                            "since the read — skipped clear",
                            project_id[:8],
                        )

            # Belt-and-suspenders release keyed on job_id
            await store.release_eval_lock(owner_id, project_id, job_id)

            # (b) Mark the job terminal
            # The reason is logged above; error field is not persisted to durable store
            await store.update_job_state(
                owner_id,
                project_id,
                job_id,
                state=reaped_state,
            )
            if reaped_state == "failed":
                result["stuck_jobs_failed"] += 1

        # 4. Orphan workers: tasks with Job tag but no matching non-terminal job
        for job_id, task_arn in job_to_task.items():
            if job_id not in active_job_ids:
                logger.info(
                    "[reconciler] Orphan worker task=%s (job=%s) — stopping",
                    task_arn[:40],
                    job_id[:8],
                )
                try:
                    await compute.cancel(
                        JobComputeHandle(kind="fargate_task", value=task_arn)
                    )
                    result["orphans_stopped"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[reconciler] Failed to stop orphan task %s: %s",
                        task_arn[:40],
                        exc,
                    )

        # Also stop tasks with no Job tag (malformed orphans)
        for task_arn in orphan_candidate_arns:
            logger.info(
                "[reconciler] Orphan worker task=%s (no Job tag) — stopping",
                task_arn[:40],
            )
            try:
                await compute.cancel(
                    JobComputeHandle(kind="fargate_task", value=task_arn)
                )
                result["orphans_stopped"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[reconciler] Failed to stop orphan task %s: %s", task_arn[:40], exc
                )

    except Exception as exc:  # noqa: BLE001
        # Never let an exception kill the triple-duty pass
        logger.exception("[reconciler] Triple-duty pass failed: %s", exc)
        result["pass_failed"] = True

    # Adopt unpolled jobs (coordinator-side poller ownership).
    # The coordinator owns poller lifecycle (progress republishing + completion turns)
    # because it holds the browser WebSocket connections. Jobs created in the AgentCore
    # microVM have no poller — this sweep starts one for each Fargate-handled non-terminal
    # job. Per-job dedupe makes repeated sweeps safe — calling this every 60s
    # reconcile pass does NOT stack pollers. This block runs even if the triple-duty pass
    # threw above, and a failure here does not prevent the completion-recovery block below.
    adopted_count = 0
    try:
        from app.services.job_progress_republisher import adopt_unpolled_jobs
        adopted_count = await adopt_unpolled_jobs()
    except Exception as exc:  # noqa: BLE001
        # Log the error but do not raise — adoption failure should not kill the pass
        logger.warning("[reconciler] Adopt sweep failed: %s", exc)

    if adopted_count > 0:
        logger.info("[reconciler] Adopted %d unpolled jobs", adopted_count)

    # Independent completion-recovery block
    # This recovery runs even if list_running_workers() throws above.
    try:
        from app.services.completion_turn import request_completion_turn
        from app.services.jobs import Job

        # Query recent completed/failed jobs (24h window aligned to event-log TTL)
        recent_jobs = await store.list_recent_completed_jobs(within_seconds=24 * 3600)

        for job_record in recent_jobs:
            # Only re-fire if completion_fired is falsy (orphaned turn)
            if job_record.get("completion_fired"):
                continue

            job_id = job_record["job_id"]
            project_id = job_record["project_id"]
            owner_id = job_record.get("owner_id") or job_record.get("user_id")

            if not owner_id:
                logger.warning(
                    "[reconciler] Job %s has no owner_id — skipping completion recovery",
                    job_id[:8],
                )
                continue

            try:
                job_type = job_record.get("type")
                if job_type == "eval":
                    # EVAL readiness is keyed on the durable EVENT LOG terminal
                    # signal + the MATERIALIZED FILE — NOT the DDB `result`
                    # (a full result payload exceeds DynamoDB's 400 KB item
                    # limit, so that write fails and would leave the completion
                    # unfired). The reconciler
                    # sources the run filename from the event log (which always
                    # lands durably) and materializes the file itself.
                    from app.services.job_progress_republisher import (
                        _TERMINAL_HARNESS_EVENTS,
                    )
                    from app.services.run_output_snapshot import (
                        RunOutputNotAvailable,
                        load_materialized_results,
                        materialize_detached_run,
                    )

                    events = await store.events_since(job_id, -1)
                    run_filename = None
                    for event in events or []:
                        data = event.get("data") or {}
                        if data.get("event") in _TERMINAL_HARNESS_EVENTS:
                            run_filename = data.get("file")
                    if not run_filename:
                        # No terminal event yet — genuinely not ready. Skip.
                        logger.debug(
                            "[reconciler] No terminal event for job=%s — skipping fire",
                            job_id[:8],
                        )
                        continue

                    try:
                        efs_path = await materialize_detached_run(
                            project_id, job_id, run_filename
                        )
                        file_result = load_materialized_results(efs_path)
                    except RunOutputNotAvailable as exc:
                        # Upload never landed — skip this pass, the next reconcile
                        # retries. Do not fire on an absent file, do not crash.
                        logger.warning(
                            "[reconciler] run output unavailable for job=%s (%s) — "
                            "skipping this pass",
                            job_id[:8], exc,
                        )
                        continue

                    won = await store.claim_completion_fired(owner_id, project_id, job_id)
                    if won:
                        logger.info(
                            "[reconciler] WON completion claim for orphaned job=%s — firing turn",
                            job_id[:8],
                        )
                        job = Job(
                            job_id=job_record["job_id"],
                            project_id=job_record["project_id"],
                            type=job_record["type"],
                            state=job_record["state"],
                            config=job_record.get("config", {}),
                            created_at=job_record.get("created_at"),
                            progress=job_record.get("progress"),
                            result=file_result,  # from the materialized FILE
                            output_file=run_filename,  # real filename (data.file)
                        )
                        # Gate: loop-owned jobs do not fire their own completion turn
                        # (the loop reports iteration completion). This is the reconciler's
                        # implementation of the rule ``jobs.is_loop_owned`` centralises,
                        # also applied by ``job_progress_republisher.adopt_unpolled_jobs``
                        # and ``optimization_loop_orchestrator._reap_loop_child_eval``.
                        from app.services.jobs import is_loop_owned
                        if not is_loop_owned(job):
                            await request_completion_turn(project_id, job)
                        else:
                            logger.debug(
                                "[reconciler] Skipped completion turn for loop-owned job=%s",
                                job_id[:8],
                            )

                        # Report symmetry: an orphan-recovered run gets its report
                        # like the live path does. Fire-and-forget.
                        try:
                            from app.services.jobs.eval_executor import (
                                _dispatch_run_report,
                            )
                            from app.services.project_paths import resolve_project_dir

                            project_dir = resolve_project_dir(project_id)
                            asyncio.create_task(_dispatch_run_report(
                                project_dir, project_id, run_filename, owner_id,
                            ))
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "[reconciler] Failed to dispatch report for job=%s: %s",
                                job_id[:8], e,
                            )
                    else:
                        logger.debug(
                            "[reconciler] LOST completion claim for job=%s — already fired",
                            job_id[:8],
                        )
                else:
                    # Non-eval job types: result-presence is sufficient.
                    # Bounded consistent re-read for cross-process readiness.
                    max_retries = 10
                    retry_delay = 0.25  # 250ms
                    result_ready = False
                    for attempt in range(max_retries):
                        fresh_record = await store.get_job(owner_id, project_id, job_id, consistent=True)
                        if fresh_record:
                            job_record = fresh_record
                            result_ready = True
                            break
                        if attempt < max_retries - 1:
                            logger.debug(
                                "[reconciler] Job record not found for job=%s (attempt %d/%d)",
                                job_id[:8], attempt + 1, max_retries
                            )
                            await asyncio.sleep(retry_delay)

                    if not result_ready:
                        logger.warning(
                            "[reconciler] Result not ready after retries for job=%s — skipping fire",
                            job_id[:8]
                        )
                        continue

                    won = await store.claim_completion_fired(owner_id, project_id, job_id)
                    if won:
                        logger.info(
                            "[reconciler] WON completion claim for orphaned job=%s — firing turn",
                            job_id[:8],
                        )
                        job = Job(
                            job_id=job_record["job_id"],
                            project_id=job_record["project_id"],
                            type=job_record["type"],
                            state=job_record["state"],
                            config=job_record.get("config", {}),
                            created_at=job_record.get("created_at"),
                            progress=job_record.get("progress"),
                            result=job_record.get("result"),
                            output_file=job_record.get("output_file"),
                        )
                        # Gate: loop-owned jobs do not fire their own completion turn
                        # (the loop reports iteration completion). Non-eval branch of the
                        # same rule applied above in the eval-specific recovery.
                        from app.services.jobs import is_loop_owned
                        if not is_loop_owned(job):
                            await request_completion_turn(project_id, job)
                        else:
                            logger.debug(
                                "[reconciler] Skipped completion turn for loop-owned job=%s",
                                job_id[:8],
                            )
                    else:
                        logger.debug(
                            "[reconciler] LOST completion claim for job=%s — already fired",
                            job_id[:8],
                        )
            except Exception as e:  # noqa: BLE001
                # Log claim/fire errors but do not crash the reconciler
                logger.warning(
                    "[reconciler] Failed to claim/fire completion for job=%s: %s",
                    job_id[:8],
                    e,
                )

    except Exception as exc:  # noqa: BLE001
        # Never let completion-recovery exceptions kill the periodic loop
        logger.exception("[reconciler] Completion-recovery failed: %s", exc)

    # Log summary of the full reconcile pass
    logger.info(
        "[reconciler] Pass complete: orphans_stopped=%d stale_locks_cleared=%d stuck_jobs_failed=%d pass_failed=%s",
        result["orphans_stopped"],
        result["stale_locks_cleared"],
        result["stuck_jobs_failed"],
        result["pass_failed"],
    )

    return result


async def _loop(interval_seconds: int) -> None:
    """Internal loop task that calls reconcile_once periodically."""
    logger.info("[reconciler] Starting periodic loop (interval=%ds)", interval_seconds)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await reconcile_once()
        except asyncio.CancelledError:
            # Clean shutdown
            logger.info("[reconciler] Loop cancelled (shutdown)")
            raise
        except Exception as exc:  # noqa: BLE001
            # Swallow non-cancel exceptions and keep looping
            logger.exception("[reconciler] Loop iteration failed: %s", exc)


def start_reconciler_loop(interval_seconds: int = 60) -> None:
    """Start the in-process periodic reconciler loop (idempotent).

    The reconciler runs as an asyncio background task that calls
    `reconcile_once()` every `interval_seconds`. It swallows non-cancel
    exceptions so transient errors don't kill the loop.

    The loop is idempotent — calling start again is a no-op if the task
    is already running. Call `stop_reconciler_loop()` to cancel the task.

    Wiring: `main.py` lifespan makes the startup call (after
    `ws_hub.start_keepalive()`), in cloud mode only — dev has no detached
    workers to reconcile.

    Args:
        interval_seconds: Seconds between reconcile passes (default 60s).
    """
    global _reconciler_task

    if _reconciler_task is not None and not _reconciler_task.done():
        # Already running — idempotent no-op
        logger.debug("[reconciler] Loop already running — skipping start")
        return

    _reconciler_task = asyncio.create_task(
        _loop(interval_seconds), name="reconciler-loop"
    )
    logger.info("[reconciler] Loop started (interval=%ds)", interval_seconds)


def stop_reconciler_loop() -> None:
    """Stop the reconciler loop (idempotent).

    Cancels the background task if it's running. Safe to call multiple times.
    Wiring: `main.py` lifespan cleanup makes the shutdown call.
    """
    global _reconciler_task

    if _reconciler_task is None or _reconciler_task.done():
        # Not running — idempotent no-op
        logger.debug("[reconciler] Loop not running — skipping stop")
        return

    _reconciler_task.cancel()
    logger.info("[reconciler] Loop stopped")
    _reconciler_task = None
