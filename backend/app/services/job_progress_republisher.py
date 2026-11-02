# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Poll the durable event log and republish progress onto /ws.

This module implements the durable progress bridge: a detached Fargate worker's
per-case progress, written to the DDB event log, is polled via
`events_since` and republished onto the owner's `/ws job:<id>` stream. Reconnect
replays from the DURABLE log, not process memory, so the per-case bars render
the same as they do for an in-process run, including multi-tab.

Role justification (naming-conventions.md): `_republisher` is a custom role —
this is neither a pure `_runner` (no end-to-end execution orchestration) nor a
`_service` (narrower than multi-operation fallback). It's a poll-republish bridge
with a single specific job: drain the durable log and fan out via ws_hub. The
descriptive custom name `job_progress_republisher` makes the role immediately
clear without overloading a generic suffix.

Purpose: carry separate-process progress over /ws without keeping the replay
buffer in ws_hub memory; replay a mid-run reconnect from since_seq/DDB with
identical bars across tabs; and publish tenant-scoped on the bridge path.

CRITICAL: The republished envelope's `seq` MUST be the DDB seq, NOT a fresh
`make_envelope` in-memory counter. One coherent seq drives live delivery,
`since_seq` replay, and client dedup.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.platform import get_platform
from app.platform.arch_split import resolve_arch_split
from app.services.completion_turn import request_completion_turn
from app.services.jobs import Job
from app.services.ws_hub import get_ws_hub

if TYPE_CHECKING:
    from app.platform.types import JobId, ProjectId

logger = logging.getLogger(__name__)

# retain poller background tasks so they aren't GC'd mid-flight
_poller_tasks: set = set()

# Per-job dedupe registry: maps job_id to the live poller task. When the coordinator
# adopt sweep runs every 60s reconcile pass, without per-job dedupe each pass
# would stack another poller for the same job and double-publish the same progress
# envelopes. This dict ensures that calling maybe_start_poller twice for one job_id
# leaves exactly one live poller. When a poller task finishes or is cancelled, its
# job_id is removed from the dedupe registry so a later legitimate re-adoption can
# re-start it.
_active_pollers: dict[str, asyncio.Task] = {}

# Terminal harness events: the harness emits {"event": "results"}
# exactly ONCE from runner.py save_results when the whole eval finishes. Per-model
# {"event": "complete"} fires N times (once per model) and is NOT the whole-eval
# terminal signal. The poller detects terminal by inspecting the durable event-log
# signal (data.event in this set), not job_record.state (which the detached worker
# does not reliably mirror).
_TERMINAL_HARNESS_EVENTS: frozenset[str] = frozenset({"results"})


async def poll_and_republish_progress(
    job_id: JobId,
    owner_id: str,
    project_id: ProjectId,
    poll_interval: float = 3.0,
    *,
    fire_completion: bool = True,
) -> None:
    """Poll the durable event log and republish progress on /ws job:<id>.

    Infinite loop: reads `events_since(job_id, last_seq)` each cycle and
    republishes each new event on stream `job:<id>`. The republished envelope's
    `seq` is the DDB seq — NOT a fresh `make_envelope` counter.

    When a terminal signal is detected in the durable event log (data.event in
    _TERMINAL_HARNESS_EVENTS), the poller (when ``fire_completion`` is True)
    claims the right to fire the completion turn via `claim_completion_fired`.
    Only the winner fires `request_completion_turn`; losers fire nothing (the
    exactly-once). The terminal TRIGGER is the event-log signal (all events land
    durably), not job_record.state (which the detached worker does not reliably
    mirror). The poller still fetches job_record for the Job rebuild payload.

    An eval that ends WITHOUT results — failed or cancelled — emits no such
    signal, so the poller also reads the job record each cycle and stops when it
    shows either state. A failed manual eval fires its completion turn through
    the same claim; a cancel fires nothing.

    Args:
        job_id: Job identifier.
        owner_id: Job owner's user_id (for tenant-scoped publish).
        project_id: Project identifier (for completion turn and flag resolution).
        poll_interval: Seconds between poll cycles (default 3s).
        fire_completion: When True (default), the poller OWNS completion — on the
            terminal signal it claims + fires the completion turn (manual eval
            path + startup reattach). When False, the poller republishes progress
            and still DETECTS the terminal signal so it can break cleanly, but
            SKIPS the claim + fire entirely. The optimization path passes False
            because the loop owns iteration completion at its per-iteration site
            (``optimization_loop_orchestrator.run_optimization_loop``), which
            BYPASSES claim_completion_fired — a claiming poller there would
            double-fire.

    Raises:
        asyncio.CancelledError: When the poller is cancelled (normal termination).
    """
    # Defense-in-depth gate: The wrapper gate (maybe_start_poller) protects known
    # call sites, but this gate protects the coroutine itself so a direct invocation
    # (e.g., asyncio.create_task(poll_and_republish_progress(...))) cannot claim
    # completion from the wrong process. Poller ownership belongs to the coordinator
    # (the process holding browser WebSocket connections), not to the microVM
    # (which runs the same coordinator turn code but has zero connections).
    from app.platform.process_role import is_agentcore_runtime
    if is_agentcore_runtime():
        logger.warning(
            "[progress-poller] Skipping coroutine for job=%s — poller ownership "
            "belongs to the coordinator, not the AgentCore runtime",
            job_id[:8]
        )
        return

    store = get_platform().metadata_store
    hub = get_ws_hub()
    last_seq = 0  # Start from the beginning of the log
    terminal_observed = False  # Track if we've observed terminal status
    # Real results filename carried on the terminal `results` event's data.file
    # (harness stdout). Used on the manual won-claim path to materialize the
    # output/ file from S3 and dispatch the report with the real filename.
    terminal_run_filename: str | None = None

    logger.info(
        "[progress-poller] Starting for job=%s owner=%s project=%s interval=%.1fs",
        job_id[:8], owner_id, project_id[:8], poll_interval
    )

    try:
        while True:
            await asyncio.sleep(poll_interval)

            # Read new events from the durable log
            try:
                events = await store.events_since(job_id, last_seq)
            except Exception as e:
                # Log the error but do not die — the durable log is the source of truth
                logger.warning(
                    "[progress-poller] Failed to read events for job=%s: %s",
                    job_id[:8], e
                )
                continue

            # Track if we see a terminal event in this cycle
            terminal_seen = False

            # Republish each new event
            for event in events:
                # Wrap per-event construction + publish in the inner try so
                # ONE malformed event can't raise a KeyError that escapes the loop
                # and kills the poller permanently (terminal→completion claim would
                # then never fire). Validate with .get() and skip malformed events.
                try:
                    seq = event.get("seq")
                    kind = event.get("kind")
                    data = event.get("data")
                    if seq is None or kind is None or data is None:
                        logger.warning(
                            "[progress-poller] Skipping malformed event for job=%s "
                            "(seq=%r kind=%r has_data=%s) — missing required field",
                            job_id[:8], seq, kind, data is not None
                        )
                        # Advance last_seq past the malformed event when we can so
                        # the next cycle doesn't re-read it (only if seq is usable).
                        if isinstance(seq, int):
                            last_seq = max(last_seq, seq)
                        continue

                    # Build the envelope MANUALLY carrying the DDB seq.
                    # Do NOT use make_envelope — it would allocate a fresh in-memory seq.
                    envelope = {
                        "stream": f"job:{job_id}",
                        "seq": seq,  # DDB seq, not _next_seq
                        "kind": kind,
                        "data": data,
                    }

                    # Publish ONLY to the job owner's connections (tenant-scoped)
                    await hub.publish_to_user(owner_id, envelope)

                    # Advance last_seq so the next cycle doesn't re-publish this event
                    last_seq = max(last_seq, seq)

                    # Detect the terminal signal in the durable event log.
                    # The harness emits {"event": "results"} exactly once from save_results
                    # when the whole eval finishes. This is the RELIABLE terminal signal
                    # (all events land durably), not job_record.state (which the detached
                    # worker does not reliably mirror). Branch on the structured field.
                    if not terminal_seen and data.get("event") in _TERMINAL_HARNESS_EVENTS:
                        terminal_seen = True
                        # Retain the real results filename from the terminal
                        # event — the manual won-claim path materializes the
                        # output/ file from S3 and dispatches the report with it.
                        terminal_run_filename = data.get("file")
                except Exception as e:
                    # Any per-event error (publish failure, unexpected shape) is
                    # logged and skipped — never exit the loop for one bad event.
                    logger.warning(
                        "[progress-poller] Failed to republish event for job=%s: %s",
                        job_id[:8], e
                    )
                    # Best-effort advance so we don't spin on the same bad event.
                    _seq = event.get("seq")
                    if isinstance(_seq, int):
                        last_seq = max(last_seq, _seq)
                    continue

            # After republishing events, check if we saw a terminal signal.
            # The terminal TRIGGER is a durable-event-log signal
            # (data.event in _TERMINAL_HARNESS_EVENTS), not job_record.state
            # (which the detached worker does not reliably mirror). The poller
            # still fetches job_record for the Job rebuild payload (result, progress),
            # but the TERMINAL TRIGGER is the event-log signal.
            if terminal_seen and not terminal_observed:
                terminal_observed = True
                logger.info(
                    "[progress-poller] Terminal signal detected in event log for job=%s",
                    job_id[:8]
                )

                # Progress-only mode (optimization path): the loop owns
                # completion in
                # ``optimization_loop_orchestrator.run_optimization_loop``
                # (which bypasses claim_completion_fired), so a claiming
                # poller here would double-fire. Republish already happened
                # above; stop the loop cleanly without claiming/firing.
                if not fire_completion:
                    logger.info(
                        "[progress-poller] Progress-only mode — stopping poll loop "
                        "for terminal job=%s (completion owned by the loop)",
                        job_id[:8]
                    )
                    break

                try:
                    # ORDER IS LOAD-BEARING: materialize EVERYTHING the fire needs
                    # first, and claim only once we are certain we can fire.
                    #
                    # `claim_completion_fired` is a one-way, exactly-once gate, and
                    # the reconciler's completion-recovery block skips any job whose
                    # `completion_fired` is set. So a claim taken and then abandoned
                    # (upload hadn't landed, record not readable) permanently loses
                    # the completion turn — no safety net covers a claimed job.
                    # Claiming last makes the reconciler a real backstop: an
                    # unclaimed job is re-materialized and fired on a later pass.
                    # This mirrors eval_reconciler's own ordering for eval jobs.

                    # FILE IS SOURCE OF TRUTH: materialize the worker-uploaded
                    # output/ file from S3 to EFS and read the per-model run list
                    # from the FILE — a full result payload exceeds DynamoDB's
                    # 400 KB item limit, so the DDB `result` cannot be relied on.
                    # Reading the file is what gives a detached manual run its
                    # completion turn and a viewable report. Never crash the
                    # poll loop.
                    from app.services.run_output_snapshot import (
                        RunOutputNotAvailable,
                        load_materialized_results,
                        materialize_detached_run,
                    )
                    try:
                        efs_path = await materialize_detached_run(
                            project_id, job_id, terminal_run_filename
                        )
                        file_result = load_materialized_results(efs_path)
                    except RunOutputNotAvailable as exc:
                        logger.warning(
                            "[progress-poller] run output unavailable for job=%s (%s) "
                            "— leaving the completion UNCLAIMED so reconciler "
                            "orphan-recovery can fire it",
                            job_id[:8], exc,
                        )
                        break

                    # Read the durable record for the Job rebuild payload
                    # (type/state/config/created_at/progress), but SOURCE the
                    # result + output_file from the materialized FILE.
                    job_record = await store.get_job(
                        owner_id, project_id, job_id, consistent=True
                    )
                    if not job_record:
                        logger.warning(
                            "[progress-poller] Terminal signal + materialized file but "
                            "job_record is None for job=%s — leaving the completion "
                            "UNCLAIMED for reconciler recovery",
                            job_id[:8]
                        )
                        break

                    # Everything needed to fire is in hand — NOW take the
                    # exactly-once claim. Only the winner fires.
                    won = await store.claim_completion_fired(owner_id, project_id, job_id)
                    if not won:
                        logger.info(
                            "[progress-poller] LOST completion claim for job=%s — firing nothing",
                            job_id[:8]
                        )
                        break

                    logger.info(
                        "[progress-poller] WON completion claim for job=%s — firing",
                        job_id[:8]
                    )

                    job = Job(
                        job_id=job_record["job_id"],
                        project_id=job_record["project_id"],
                        type=job_record["type"],
                        state=job_record["state"],
                        config=job_record.get("config", {}),
                        created_at=job_record.get("created_at"),
                        progress=job_record.get("progress"),
                        result=file_result,  # from the materialized FILE, not DDB
                        output_file=terminal_run_filename,  # real filename (data.file)
                    )
                    await request_completion_turn(project_id, job)

                    # Report dispatch for detached MANUAL runs: the in-process
                    # path fires these from eval_executor after terminal, so the
                    # detached path fires them here. Dispatch the per-run
                    # report + winner-gated exec sidecar with the REAL filename.
                    # Fire-and-forget (errors swallowed inside the dispatchers).
                    try:
                        from app.services.jobs.eval_executor import (
                            _dispatch_exec_if_winner_changed,
                            _dispatch_run_report,
                        )
                        from app.services.project_paths import resolve_project_dir

                        project_dir = resolve_project_dir(project_id)
                        asyncio.create_task(_dispatch_run_report(
                            project_dir, project_id, terminal_run_filename, owner_id,
                        ))
                        asyncio.create_task(_dispatch_exec_if_winner_changed(
                            project_dir, project_id, owner_id,
                        ))
                    except Exception as e:
                        logger.warning(
                            "[progress-poller] Failed to dispatch report for job=%s: %s",
                            job_id[:8], e,
                        )

                    # Stop the poll loop after observing terminal (break)
                    logger.info(
                        "[progress-poller] Stopping poll loop for terminal job=%s",
                        job_id[:8]
                    )
                    break

                except Exception as e:
                    # Log terminal-check errors but do not crash the loop
                    logger.warning(
                        "[progress-poller] Failed to check terminal status for job=%s: %s",
                        job_id[:8], e
                    )

            # A failed or cancelled eval never emits ``results``, so without this
            # check its poller ran until the coordinator restarted, and a failed
            # manual eval never reached chat. The job record is where both ends
            # land: the worker's ``fail_job`` (drained before the worker exits),
            # the reconciler's reap of a worker that vanished, and the cancel path.
            # A cancel needs no turn from here — the UI's cancel posts its own
            # line, and the agent's cancel is the agent's own reply.
            if not terminal_seen:
                ended = await _read_ended_without_results(
                    store, owner_id, project_id, job_id
                )
                if ended is not None:
                    if fire_completion and ended.get("state") == "failed":
                        try:
                            await _fire_failure_completion(
                                store, owner_id, project_id, job_id, ended
                            )
                        except Exception as e:  # noqa: BLE001 — the claim keeps a retry exactly-once
                            logger.warning(
                                "[progress-poller] Failure completion for job=%s "
                                "failed (%s) — retrying next cycle",
                                job_id[:8], e,
                            )
                            continue
                    logger.info(
                        "[progress-poller] job=%s ended %s without results — "
                        "stopping poll loop",
                        job_id[:8], ended.get("state"),
                    )
                    break

    except asyncio.CancelledError:
        logger.info(
            "[progress-poller] Cancelled for job=%s (last_seq=%d)",
            job_id[:8], last_seq
        )
        raise  # Re-raise so the caller knows we stopped
    except Exception as e:
        logger.exception(
            "[progress-poller] Unexpected error for job=%s: %s",
            job_id[:8], e
        )
        # Log and continue — never die


async def _read_ended_without_results(
    store, owner_id: str, project_id: str, job_id: str
) -> dict | None:
    """The durable job record, if the job ended failed or cancelled; else None.

    A failed read is one lost observation, not a verdict — the next cycle reads
    again — so it returns None rather than raising.
    """
    try:
        record = await store.get_job(owner_id, project_id, job_id, consistent=True)
    except Exception as e:  # noqa: BLE001 — retried next cycle
        logger.warning(
            "[progress-poller] Could not read the job record for job=%s: %s",
            job_id[:8], e,
        )
        return None
    if record and record.get("state") in ("failed", "cancelled"):
        return record
    return None


async def _fire_failure_completion(
    store, owner_id: str, project_id: str, job_id: str, record: dict
) -> None:
    """Tell the coordinator a manual eval failed, exactly once across processes.

    Takes the same ``claim_completion_fired`` gate the results path takes, so a
    failure is announced once however many pollers observe it. The Job is rebuilt
    from the record, which carries no error — the reason reaches the message
    through the progress text ``fail_job`` writes.
    """
    if not await store.claim_completion_fired(owner_id, project_id, job_id):
        logger.info(
            "[progress-poller] LOST completion claim for failed job=%s — firing nothing",
            job_id[:8],
        )
        return
    logger.info(
        "[progress-poller] WON completion claim for failed job=%s — firing",
        job_id[:8],
    )
    job = Job(
        job_id=record["job_id"],
        project_id=record["project_id"],
        type=record["type"],
        state="failed",
        config=record.get("config", {}),
        created_at=record.get("created_at"),
        progress=record.get("progress") or {},
        output_file=record.get("output_file"),
    )
    await request_completion_turn(project_id, job)


async def maybe_start_poller(
    project_id: ProjectId,
    job_id: JobId,
    owner_id: str,
    arch_split: bool,
    *,
    fire_completion: bool = True,
) -> bool:
    """Start the progress bridge poller if the arch_split flag is enabled.

    Args:
        project_id: Project identifier (for completion turn).
        job_id: Job identifier.
        owner_id: Job owner's user_id (for tenant-scoped publish).
        arch_split: Pre-resolved arch_split flag (caller resolves once).
        fire_completion: Threaded into the poller. Default True preserves the
            manual eval path + startup reattach behavior byte-for-byte. The
            optimization path passes False (progress-only) so the poller never
            fires completion — the loop owns iteration completion.

    Returns:
        ``True`` when this call actually created a poller task; ``False`` on every
        skip path (wrong process, old path, or a poller already live for this job).
        The periodic adopt sweep reports how many jobs it newly took ownership of,
        so it needs to distinguish "started" from "already covered" — without a
        return value the sweep's count would climb every pass and imply repeated
        adoption that never happened.
    """
    # First gate: Poller ownership belongs to the coordinator (the process holding
    # browser WebSocket connections), not the AgentCore runtime (which runs the same
    # coordinator turn code but has zero connections). This gate protects the known
    # call sites (dispatch.py, main.py startup reattach, eval_reconciler adopt sweep).
    from app.platform.process_role import is_agentcore_runtime
    if is_agentcore_runtime():
        logger.warning(
            "[progress-poller] Skipping wrapper for job=%s — poller ownership "
            "belongs to the coordinator, not the AgentCore runtime",
            job_id[:8]
        )
        return False

    # Accept the pre-resolved boolean instead of re-resolving here
    # (resolve-once-and-thread, consistent with dispatch.py)
    if not arch_split:
        logger.debug(
            "[progress-poller] Skipping for job=%s (arch_split=False)",
            job_id[:8]
        )
        return False

    # Per-job dedupe: the coordinator-side adopt sweep runs every 60s
    # reconcile pass. Without dedupe, each pass would stack another poller for the
    # same job and double-publish the same progress envelopes. Check if a poller
    # is already live for this job_id; if so, skip creating a second one.
    existing_task = _active_pollers.get(job_id)
    if existing_task is not None and not existing_task.done():
        logger.info(
            "[progress-poller] Already polling job=%s — skipping duplicate poller",
            job_id[:8]
        )
        return False

    # Start the poller as a background task
    # retain the task so it isn't GC'd mid-flight
    task = asyncio.create_task(
        poll_and_republish_progress(
            job_id, owner_id, project_id, fire_completion=fire_completion
        ),
        name=f"progress-poller-{job_id[:8]}"
    )
    _poller_tasks.add(task)
    _active_pollers[job_id] = task

    # Register a done callback that cleans up BOTH registries, but only removes
    # from _active_pollers if the finishing task is still the one mapped to that
    # job_id (so a legitimate re-adoption after cancellation is not clobbered by
    # a late callback from the prior task).
    def _cleanup(finished_task: asyncio.Task) -> None:
        _poller_tasks.discard(finished_task)
        if _active_pollers.get(job_id) is finished_task:
            _active_pollers.pop(job_id, None)

    task.add_done_callback(_cleanup)
    logger.info(
        "[progress-poller] Started task for job=%s project=%s owner=%s",
        job_id[:8], project_id[:8], owner_id
    )

    # Do NOT await the task — it runs in the background until cancelled
    return True


async def adopt_unpolled_jobs() -> int:
    """Adopt non-terminal Fargate-handled jobs that are not already being polled (coordinator-side sweep).

    This is the reusable coordinator-side sweep that establishes poller ownership
    for jobs created in another process (the AgentCore microVM). The microVM runs
    the same coordinator turn code, so an agent-dispatched eval writes its job to
    the durable store and then returns. That job has no in-memory presence on the
    coordinator and no poller. This sweep lists non-terminal jobs, filters for
    Fargate-handled ones, and starts a poller for each. Per-job dedupe makes
    repeated calls safe — calling this sweep every 60s reconcile pass does NOT
    stack pollers.

    Derives the ``fire_completion`` flag from the durable record's config source
    field: optimization-sourced jobs are adopted progress-only (False), because
    the optimization loop owns iteration completion at its per-iteration site
    (``optimization_loop_orchestrator.run_optimization_loop``), which bypasses
    claim_completion_fired. Manual-sourced jobs are
    adopted completion-owning (True). The previous startup-only routine passed
    the completion-owning default for every job, which would have let a reattached
    poller claim a completion the optimization loop owns — this sweep closes that.

    Returns:
        Count of jobs whose poller this sweep NEWLY started. Jobs already being
        polled are not counted, so a steady-state pass reports zero rather than
        re-reporting the same job every interval.
    """
    # ``_is_fargate_handle`` is imported function-locally because it lives in an
    # API router module: a module-scope import here would point a service at a
    # router, inverting the dependency direction and risking an import cycle at
    # app construction. ``get_platform`` is deliberately NOT re-imported locally —
    # the module-scope binding is used, so this module has exactly ONE patch point
    # for the platform. A shadowing local import would silently bypass a test that
    # patches the module attribute.
    from app.api.eval_jobs import _is_fargate_handle

    store = get_platform().metadata_store
    adopted = 0

    try:
        jobs = await store.list_non_terminal_jobs()
    except Exception as exc:  # noqa: BLE001
        # Log the error but do not raise — one sweep failure should not block
        # the next sweep or kill the reconciler pass
        logger.warning("[adopt-sweep] Failed to list non-terminal jobs: %s", exc)
        return adopted

    for job in jobs:
        compute_handle = job.get("compute_handle")

        # Skip unless the compute handle is an ECS task ARN (Fargate-handled).
        # PID-shaped or missing handles are dev in-process jobs, not adoptable.
        #
        # Both launch sites persist ``compute_handle=str(handle)``, and
        # ``JobComputeHandle.__str__`` returns the bare ``value`` — so the stored
        # shape is a FLAT ARN STRING, never the dataclass or a serialized dict.
        # Reuse the single canonical predicate rather than re-deriving the
        # discrimination here: a second, subtly different notion of "is this a
        # Fargate handle" is how this sweep would silently adopt nothing.
        if not _is_fargate_handle(compute_handle):
            continue

        project_id = job.get("project_id")
        job_id = job.get("job_id")
        owner_id = job.get("owner_id") or job.get("user_id")

        if not (project_id and job_id and owner_id):
            # Incomplete record — log and skip, do not block the rest of the sweep
            logger.warning(
                "[adopt-sweep] Skipping incomplete job record: job=%s project=%s owner=%s",
                job_id[:8] if job_id else "?",
                project_id[:8] if project_id else "?",
                owner_id[:8] if owner_id else "?",
            )
            continue

        # Derive fire_completion from the durable record's config source field.
        # Optimization-sourced jobs are adopted progress-only (fire_completion=False)
        # because the optimization loop owns iteration completion. Manual-sourced
        # jobs are adopted completion-owning (fire_completion=True).
        config = job.get("config") or {}
        source = config.get("source")
        fire_completion = source != "optimization"

        try:
            # arch_split=True is the pre-resolved new-path boolean: reaching here
            # means the job already carries a Fargate handle, so it is definitionally
            # on the durable path. Per-job dedupe makes repeated calls safe.
            #
            # Count only what this sweep NEWLY took ownership of. Counting every
            # iteration would report the same job as freshly adopted on every 60s
            # pass, turning a steady-state log line into a phantom re-adoption loop
            # and hiding whether adoption ever actually happened.
            if await maybe_start_poller(
                project_id, job_id, owner_id, arch_split=True, fire_completion=fire_completion
            ):
                adopted += 1
        except Exception as exc:  # noqa: BLE001
            # Log per-job exceptions but do not let one malformed record abort
            # the sweep.
            logger.warning(
                "[adopt-sweep] Failed to adopt job=%s: %s", job_id[:8], exc
            )

    return adopted
