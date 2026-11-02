# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization loop REST endpoints: get state, start, resume, and cancel."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import OptimizationStartRequest
from app.platform.api_authz import require_project_read, require_project_write
from app.platform.arch_split import resolve_arch_split
from app.platform.types import UserId
from app.services.dataset_service import DatasetService
from app.services.jobs import get_job_tracker
from app.services.harness_config import load_harness_config
from app.services.optimization_state import OptimizationState
from app.services.task_error_sink import attach_error_sink

from app.services.project_manager import get_project_dir_or_404

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["eval"])


@router.post("/{project_id}/optimization/cancel")
async def cancel_optimization(
    project_id: str,
    _user: UserId = Depends(require_project_write),
):
    """Cancel the running optimization loop and any active eval job.

    Sets optimization state to cancelled and kills any running eval
    subprocess. The orchestrator's poll loop detects the cancellation
    and exits cleanly.

    Args:
        project_id: Unique identifier of the project.

    Returns:
        Confirmation dict.
    """
    project_dir = get_project_dir_or_404(project_id)

    # Resolve arch_split at the top (this endpoint previously did not resolve at all)
    arch_split = await resolve_arch_split(project_id)

    # This cancel is FUNCTIONAL not cosmetic — on arch_split=True the loop polls
    # DDB for the cancel signal (dispatch.py + optimization_executor._wait_for_eval),
    # so an EFS write here would never stop the loop.
    # Routed through the repository, which keys durable reads on the project
    # OWNER, the partition the loop writes to, never on the requester.
    from app.services import optimization_state_repository as opt_repo

    handle = await opt_repo.read(project_id, project_dir, arch_split=arch_split)
    opt_state = handle.state

    if not handle.is_live():
        raise HTTPException(status_code=409, detail="No active optimization to cancel")

    # Capture iteration BEFORE mutating status
    cancelled_at_iteration = opt_state.current_iteration

    # Set to "stopping" first — if the orchestrator is alive it will
    # observe this and exit cleanly, flipping to "cancelled". If no
    # orchestrator task exists we flip to ``cancelled`` ourselves
    # below (otherwise the SPA would stick at "stopping" forever).
    opt_state.status = "stopping"
    # write() returns a handle carrying the NEW version, so the orphan direct-flip
    # below chains off it by reassignment instead of tracking a separate local
    # (never guess +1).
    handle = opt_repo.write(handle)
    logger.info("[eval-api] optimization.cancel project=%s", project_id[:8])

    # Publish a terminal envelope so the SPA's OptimizationStateProvider
    # refetches via useProjectEvent on the same channel as
    # convergence/plateau (see ChatRuntimeProvider's mid-turn-replay
    # plumbing — the project bus uses the same shape). Without this
    # the bar lags until the next backstop poll after the Stop click. The
    # orchestrator loop's existing CancelledError branch publishes
    # again when it actually exits so the bar flips from "stopping"
    # to the final "cancelled" status without waiting for the poll.
    from app.services.jobs import _publish_project_event
    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "terminal",
        "reason": "cancelled",
        "message": "Optimization cancelled.",
    })

    # Cancel the orchestrator asyncio task
    from app.services.optimization_loop_orchestrator import _loop_task_name
    orchestrator_task_name = _loop_task_name(project_id)
    orchestrator_found = False
    for task in asyncio.all_tasks():
        if task.get_name() == orchestrator_task_name:
            task.cancel()
            orchestrator_found = True
            break
    logger.info("[eval-api] orchestrator-task.cancel task=%s found=%s",
                orchestrator_task_name, orchestrator_found)

    # Cancel every active job for this project AND await every harness
    # subprocess kill. ``hard_cancel_project_eval`` is the single
    # canonical path: it iterates active jobs, marks them cancelled,
    # and *awaits* SIGTERM→SIGKILL escalation on each pid before
    # returning. Without this the kill is fire-and-forget via
    # ``JobTracker.cancel_job``'s ``loop.create_task(...)`` — the
    # response returns before the subprocess actually dies.
    from app.services.harness_runner import hard_cancel_project_eval
    cancelled = await hard_cancel_project_eval(project_id)
    logger.info("[eval-api] hard-cancel project=%s cancelled=%d",
                project_id[:8], cancelled)

    # If no orchestrator task exists, the optimization was already
    # orphaned (coordinator restart, OOM, or the orchestrator died
    # mid-run). Nothing will ever flip ``stopping`` → ``cancelled`` for
    # us; do it here so the bar doesn't stick at "stopping" forever
    # waiting on a dead task. Re-publish so the SPA picks up the final
    # status without a poll.
    if not orchestrator_found:
        opt_state.status = "cancelled"
        # Chains the version returned by the stopping-write above; consecutive
        # conditional writes must use the store-returned version or the second
        # write races.
        handle = opt_repo.write(handle)
        # The dead (or never-adopted) loop can never run its terminal tail, so
        # the eval lock the dispatch claimed for it would leak forever — the
        # reconciler's stale-lock duty is keyed off jobs, and a wedged dispatch
        # has none. The canceller completes the whole transition the loop
        # cannot. release_eval_lock is holder-fenced, so this is a no-op if a
        # manual eval holds the lock instead.
        if handle.arch_split and handle.owner_id:
            from app.platform import get_platform
            await get_platform().metadata_store.release_eval_lock(
                handle.owner_id, project_id, f"optimization-{project_id}"
            )
            logger.info(
                "[eval-api] optimization.cancel released orphaned loop lock "
                "project=%s", project_id[:8],
            )
        _publish_project_event(project_id, {
            "resource": "optimization",
            "type": "terminal",
            "reason": "cancelled",
            "message": "Optimization cancelled (orchestrator not running).",
        })
        logger.info(
            "[eval-api] optimization.cancel project=%s direct-flip=cancelled "
            "(orchestrator was dead)", project_id[:8],
        )

    # Drain pending completion-turn messages so stale results don't
    # fire a coordinator turn after cancel
    from app.services.completion_turn import drain_pending_completions
    drained_pending = drain_pending_completions(project_id)
    if drained_pending:
        logger.info("[eval-api] completion-turn.drained pending=%d", drained_pending)

    # Drain queued orchestrator turns that haven't started yet
    from app.services.agent_turn import get_agent_turn_tracker
    turn_tracker = get_agent_turn_tracker()
    drained_queued = turn_tracker.drain_queued_turns(project_id, origin="orchestrator")
    if drained_queued:
        logger.info("[eval-api] turn-queue.drained orchestrator=%d", drained_queued)

    # Cancel any running orchestrator agent turn
    active_turn = turn_tracker.get_active_turn(project_id, origin="orchestrator")
    if active_turn and active_turn._client:
        # Mark terminal BEFORE interrupting — symmetric with the user
        # Stop path (``chat.py`` stop_turn) and with the elif branch
        # below. ``interrupt()`` tears the turn down via CancelledError,
        # which bypasses run_agent_turn's ``except Exception``; without
        # setting status here the turn would only flip terminal when its
        # task eventually reaches ``finally`` (which forces it),
        # and until then ``get_active_turn`` keeps reporting it running
        # and the chat queue stays wedged. Setting it up front unblocks
        # the queue the instant cancel is requested.
        active_turn.status = "failed"
        turn_tracker.notify_turn_complete(project_id)
        try:
            await active_turn._client.interrupt()
            logger.info("[eval-api] agent-turn.interrupted turn=%s", active_turn.turn_id[:8])
        except Exception as e:
            logger.warning("[eval-api] agent-turn.interrupt-failed turn=%s error=%s", active_turn.turn_id[:8], e)
    elif active_turn:
        # Client not yet attached (turn is still connecting). Cancel the
        # underlying asyncio task directly so the turn dies cleanly.
        task_name = f"agent-turn-{active_turn.turn_id[:8]}"
        cancelled = False
        for task in asyncio.all_tasks():
            if task.get_name() == task_name:
                task.cancel()
                cancelled = True
                break
        active_turn.status = "failed"
        turn_tracker.notify_turn_complete(project_id)
        logger.info(
            "[eval-api] agent-turn.task-cancelled turn=%s found=%s",
            active_turn.turn_id[:8], cancelled,
        )

    # Notify the agent AFTER drains so the message survives them
    from app.services.completion_turn import request_terminal_message
    await request_terminal_message(
        project_id,
        f"Optimization cancelled from the UI at iteration {cancelled_at_iteration}.",
        lifecycle=True,
    )

    return {"status": "stopping"}


def _load_harness_config_or_400(project_dir, action: str):
    """Load src/config.py, answering 400 rather than 500 if it is unreadable.

    The file is written by the agent, so a syntax error or a wrongly shaped
    value is a state the user fixes by asking in chat, not a server fault.

    Args:
        project_dir: The project's directory.
        action: ``"start"`` or ``"resume"``, for the log line.

    Returns:
        The parsed HarnessConfig, or None when config.py does not exist.

    Raises:
        HTTPException: 400 naming the problem when config.py cannot be read.
    """
    try:
        return load_harness_config(project_dir)
    except (SyntaxError, ValueError) as exc:
        logger.warning(
            "[eval-api] optimization.%s.rejected reason=unreadable-config detail=%s",
            action, exc,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"src/config.py could not be read: {exc}. Ask in chat to fix it."
            ),
        ) from exc


def _optimization_state_to_dict(state, *, project_dir=None) -> dict:
    """Serialize an :class:`OptimizationState` to the SPA's wire shape.

    Shared by ``GET /optimization`` and ``GET /activity`` so the two
    endpoints can never drift on the optimization payload — a single
    serializer is the whole point of the activity consolidation.

    When ``project_dir`` is provided, each iteration row is enriched with a
    ``trophy_score`` computed by the single backend implementation
    (``app.services.trophy``) against harness-config targets — the SPA orders
    these tuples instead of re-scoring.
    """
    iterations = state.iterations
    if project_dir is not None:
        from app.services.trophy import resolve_trophy_targets, trophy_score_loop

        targets = resolve_trophy_targets(project_dir)
        iterations = [
            {**it, "trophy_score": trophy_score_loop(it, targets)}
            for it in state.iterations
        ]
    return {
        "status": state.status,
        "current_iteration": state.current_iteration,
        # Sources-of-truth: each fact under its own name. This dict used
        # to ship max_iter_cap AS "max_iterations" while the actual
        # max_iterations field rode under "iterations_per_run" — three facts
        # sharing one name across the stack, guaranteed to diverge after a
        # max_iterations resume (reset_for_resume: cap = current + budget).
        # No compat alias: one deployment, consumers renamed with it.
        "iteration_cap": state.max_iter_cap,
        "iterations_per_run": state.max_iterations,
        "target_metric_scores": state.target_metric_scores,
        "iterations": iterations,
        "best_iteration": state.best_iteration(),
        "plateau_window": state.plateau_window,
        "plateau_threshold": state.plateau_threshold,
        "winning_model_id": state.winning_model_id,
        "current_prompt_version": state.current_prompt_version,
        # This hand-built dict is a third copy of the display field set: a
        # field added to OptimizationState for display must be added here
        # too, or this endpoint silently drops it. The contract test in
        # test_optimization_api.py locks phase through the full read path.
        "phase": state.phase,
        "error": state.error.to_dict() if state.error else None,
    }


@router.get("/{project_id}/activity")
async def get_project_activity(
    project_id: str,
    _user: UserId = Depends(require_project_read),
):
    """Return the single "what is running" view for a project.

    Replaces the SPA's two independent polls (optimization-state on disk
    + eval-jobs in memory) and the client-side stitching of the two:

    * ``optimization`` — the loop-level view from ``OptimizationState``
      (disk, authoritative, survives restart). ``null`` when no
      optimization has ever run for the project.
    * ``eval`` — the subprocess-level view from the active eval ``Job``
      in the in-memory ``JobTracker``. ``null`` when no eval is active.

    Liveness comes from one place: the eval job's own state. The endpoint
    used to second-guess an active eval against the on-disk optimization
    status (suppress an optimization-child eval whenever the loop wasn't
    in a "live" status). That guess existed only to cover a state that
    shouldn't be possible — the eval-job tracker and the loop disagreeing
    — and it was wrong precisely when a loop wrote a terminal status while
    a child eval was still executing, hiding a genuinely-running eval
    forever. The invariant is now enforced at the source instead: the
    loop reaps its in-flight child eval on every exit path
    (``_reap_loop_child_eval`` in ``optimization_loop_orchestrator.py`` + the
    cancel REST endpoint's ``hard_cancel_project_eval``), so the tracker never
    holds an active eval that no live loop owns. With that guaranteed, the
    eval is reported straight from the active job — no cross-checking.

    Per-case progress (``models``) is included for the live eval so the
    SPA's detail rows render without a second call; the high-frequency
    per-case animation still streams over the ``job:<id>`` WebSocket.
    """
    project_dir = get_project_dir_or_404(project_id)
    from app.services import optimization_state_repository as opt_repo

    # Resolve arch_split ONCE here and thread it: this endpoint uses it for three
    # separate decisions (the state read, the job rebuild, and the models fold), and
    # re-resolving per decision is what the resolve-once discipline forbids — a
    # mid-request flip would split one response across both architectures.
    #
    # Resolved here (not inside the repository) and threaded in, matching the other
    # three endpoints in this module: the flag governs decisions BEYOND the state
    # read, so this function must own it explicitly rather than recover it from a
    # handle as a side effect.
    arch_split = await resolve_arch_split(project_id)
    handle = await opt_repo.read(project_id, project_dir, arch_split=arch_split)
    state = handle.state
    optimization = (
        _optimization_state_to_dict(state, project_dir=project_dir)
        if state is not None
        else None
    )

    tracker = get_job_tracker()

    # On arch_split=True, rebuild jobs from the durable store before reading the
    # active eval. The job may have been created in another process (the AgentCore
    # microVM), so this tracker entry is a cold replica rather than a live executing
    # object, and cache-as-truth (the premise of rebuild_jobs_from_store's guidance
    # for the coordinator) does not apply to it. Without the rebuild, a foreign-process
    # job is invisible and the browser never learns the job id to subscribe with.
    # Without terminal refresh, that adopted job stays non-terminal in this process's
    # cache and the bar runs forever. The rebuild-on-read sites in the jobs API are
    # the precedent for this call shape. Provenance is the axis, not process identity:
    # the mode is for any reader whose cached entry is a replica it did not execute
    # (agent-facing tools in the microVM, and now the coordinator's activity view
    # for a job dispatched elsewhere), while cache-as-truth continues to apply to
    # jobs the reading process launched itself.
    if arch_split:
        await tracker.rebuild_jobs_from_store(
            project_id, arch_split=True, refresh_terminal=True
        )

    active_eval = tracker.get_active_job_by_type(project_id, "eval")

    eval_payload: dict | None = None
    if active_eval is not None:
        progress = active_eval.progress or {}
        eval_payload = {
            "job_id": active_eval.job_id,
            "status": active_eval.status,
            "models_total": progress.get("models_total", 0),
            "progress": progress,
        }
        # On arch_split=True the eval runs in a DETACHED worker, so the
        # coordinator's in-memory job.progress is frozen at the queued seed —
        # the only durable source of live per-case progress is the event log.
        # Fold it into per-model counters so a mid-scoring remount re-seeds
        # Generate X/N from the log. On arch_split=False (dev / in-process) the
        # WS stream already carries live progress within the one process and
        # events_since returns [] on that path, so models stays absent there.
        if arch_split:
            from app.platform import get_platform
            from app.services.job_progress_analyzer import fold_event_log_progress

            # Bound here rather than at the top: the state read moved into the
            # repository, so this is now the only site in the function that needs
            # the raw store.
            store = get_platform().metadata_store
            events = await store.events_since(active_eval.job_id, -1)
            eval_payload["models"] = fold_event_log_progress(events)

    return {"optimization": optimization, "eval": eval_payload}


async def _admit_optimization_launch(
    project_id: str, *, arch_split: bool, kind: str
) -> tuple[str | None, str | None]:
    """Run the shared admission checks, then claim the project's eval lock.

    Shared by the start and resume routes so they cannot drift from each other or
    from the MCP dispatch path. All three now go through the same checks; these
    two used to go through none of them.

    Args:
        project_id: Project being launched into.
        arch_split: Durable path in use. Dev has no detached workers and no
            durable lock, so admission is a no-op there.
        kind: "start" or "resume", for the log line.

    Returns:
        ``(lock_owner_id, lock_holder_id)`` to thread into the loop's config, or
        ``(None, None)`` when no lock was taken (dev).

    Raises:
        HTTPException: 429 when a cap or the kill switch refuses the launch, 409
            when another eval already holds the project's lock.
    """
    if not arch_split:
        return None, None

    from app.services.agent_turn import _resolve_owner_id_for_project
    from app.services.job_admission import (
        check_launch_admission,
        claim_project_eval_lock,
        optimization_lock_holder_id,
    )

    # The OWNER's caps and the OWNER's lock partition — not the requester's —
    # the same keys the agent's dispatch uses, so both share one budget and lock.
    owner_id = _resolve_owner_id_for_project(project_id)

    refusal = await check_launch_admission(
        owner_id=owner_id, project_id=project_id, kind="optimization"
    )
    if refusal is not None:
        logger.warning(
            "[eval-api] optimization.%s.rejected reason=%s project=%s",
            kind, refusal.code, project_id[:8],
        )
        raise HTTPException(status_code=429, detail=refusal.message)

    holder_id = optimization_lock_holder_id(project_id)
    if not await claim_project_eval_lock(
        owner_id=owner_id, project_id=project_id, holder_id=holder_id
    ):
        logger.warning(
            "[eval-api] optimization.%s.rejected reason=eval_already_running project=%s",
            kind, project_id[:8],
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "An eval is already running for this project. Wait for it to "
                "finish or cancel it before starting an optimization loop."
            ),
        )
    return owner_id, holder_id


async def _release_optimization_lock(
    owner_id: str | None, project_id: str, holder_id: str | None
) -> None:
    """Release a lock claimed by :func:`_admit_optimization_launch`.

    For the window between claiming and handing ownership to the loop: if the
    loop never starts, nothing else will ever release it.
    """
    if not owner_id or not holder_id:
        return
    from app.services.job_admission import release_project_eval_lock

    await release_project_eval_lock(
        owner_id=owner_id, project_id=project_id, holder_id=holder_id
    )


async def _abandon_optimization_launch(
    project_id: str,
    project_dir,
    *,
    arch_split: bool,
    user_id: str,
    lock_owner_id: str | None,
    lock_holder_id: str | None,
) -> None:
    """Undo a launch that was admitted but never handed to a loop.

    Two things need undoing, and both only matter in this window: the state is
    already persisted as ``running`` and the eval lock is already held, while the
    loop that would normally settle each one does not exist. Leaving them makes
    the project look permanently busy — the UI shows a run that cannot finish and
    the lock blocks the next eval until its TTL.

    Best-effort and ordered state-first: the caller is already raising, so the
    useful outcome is that as much as possible is undone, and a failure to settle
    the state must not skip the lock release.
    """
    from app.services import optimization_state_repository as opt_repo

    try:
        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.is_live():
            handle.state.status = "cancelled"
            opt_repo.write(handle)
    except Exception:  # noqa: BLE001 - must not skip the release below
        logger.exception(
            "[eval-api] could not settle abandoned optimization state for %s",
            project_id[:8],
        )

    await _release_optimization_lock(lock_owner_id, project_id, lock_holder_id)


@router.post("/{project_id}/optimization/start")
async def start_optimization_rest(
    project_id: str,
    body: OptimizationStartRequest | None = None,
    user: UserId = Depends(require_project_write),
):
    """Start a new optimization loop from UI.

    Reads all parameters from the harness config (src/config.py). Overwrites
    any existing optimization state. Guards: harness must exist, no running
    optimization, required config fields present.

    Args:
        project_id: Unique identifier of the project.
        body: Optional request body with ``data_source`` selector. Defaults
            to ``"raw"`` (harness-configured dataset / existing train
            split) when omitted or null.

    Returns:
        Dict with started status.
    """
    project_dir = get_project_dir_or_404(project_id)

    logger.info("[eval-api] optimization.start project=%s", project_id[:8])

    # Resolve arch_split BEFORE the existing-check and before create
    arch_split = await resolve_arch_split(project_id)

    # Guard: harness must exist
    if not (project_dir / "run_eval.py").exists():
        logger.warning("[eval-api] optimization.start.rejected reason=no-harness")
        raise HTTPException(status_code=400, detail="No eval harness found. Complete Phase 3 first.")

    # Guard: no running optimization. The repository owns the home choice AND the
    # owner partition, so this read finds a running loop however it was started.
    from app.services import optimization_state_repository as opt_repo

    existing_handle = await opt_repo.read(project_id, project_dir, arch_split=arch_split)
    existing = existing_handle.state

    if existing_handle.is_live():
        logger.warning("[eval-api] optimization.start.rejected reason=already-%s", existing.status)
        raise HTTPException(status_code=409, detail="Optimization is already running.")

    # Guard: no active eval. A live eval — most often a prior loop's
    # iteration still finishing — means a second loop would race it for
    # the harness subprocess and leave the tracker holding an active eval
    # no live loop owns (the exact drift the /activity endpoint now refuses
    # to paper over). The on-disk optimization status alone can't catch
    # this: a stale/racy terminal status (e.g. optimizer_stalled written
    # while a child eval was still executing) would defeat it. The eval
    # job's own liveness is authoritative.
    # Rehydrate first. The tracker is process-local, so an eval dispatched from
    # the AgentCore runtime or surviving a coordinator restart is invisible to it
    # — which is exactly why /activity rehydrates before answering. Without this
    # the guard below silently passed for detached work.
    tracker = get_job_tracker()
    if arch_split:
        try:
            await tracker.rebuild_jobs_from_store(
                project_id, arch_split=True, refresh_terminal=True
            )
        except Exception:  # noqa: BLE001 - fall through to the durable lock below
            logger.warning(
                "[eval-api] optimization.start could not rehydrate jobs for %s; "
                "relying on the durable eval lock",
                project_id[:8],
                exc_info=True,
            )
    existing_eval = tracker.get_active_job_by_type(project_id, "eval")
    if existing_eval is not None:
        logger.warning("[eval-api] optimization.start.rejected reason=active-eval job=%s",
                       existing_eval.job_id[:8])
        raise HTTPException(
            status_code=409,
            detail=(
                f"An eval is still finishing (job {existing_eval.job_id[:8]}). "
                "Cancel it or wait before starting an optimization loop."
            ),
        )

    # Read harness config
    harness = _load_harness_config_or_400(project_dir, "start")
    if harness is None:
        logger.warning("[eval-api] optimization.start.rejected reason=no-config")
        raise HTTPException(status_code=400, detail="No harness config found at src/config.py.")
    logger.info("[eval-api] harness-config models=%s targets=%s max_iter=%s",
                harness.models, harness.target_metric_scores, harness.max_iterations)

    # Validate required fields
    if not harness.models:
        raise HTTPException(status_code=400, detail="No models configured in harness config.")
    fm = harness.field_mappings or {}
    if "input" not in fm or "expected_output" not in fm:
        raise HTTPException(status_code=400, detail="field_mappings must include 'input' and 'expected_output' keys in harness config.")
    # Optimization parameters come from the harness config the user settled
    # in chat — a missing field is a 400 naming it, never a silently
    # invented default (this endpoint used to fabricate ExactMatch@0.85 /
    # 10 iterations / plateau 3, so a half-configured project optimized
    # against targets nobody chose).
    #
    # Metrics are not in this list: they live as code in
    # src/metrics/metrics.py, and the harness's build_metrics fails loudly if
    # they are missing.
    missing = [
        name
        for name, value in (
            ("target_metric_scores", harness.target_metric_scores),
            ("max_iterations", harness.max_iterations),
            ("plateau_window", harness.plateau_window),
        )
        if not value
    ]
    if missing:
        logger.warning(
            "[eval-api] optimization.start.rejected reason=missing-fields fields=%s",
            missing,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Harness config (src/config.py) is missing required optimization "
                f"fields: {', '.join(missing)}. Ask in chat to set them."
            ),
        )
    target_scores = harness.target_metric_scores

    # Collision guard: distinct metric names that collapse to the same base
    # after suffix normalization ("Correctness [strict]" vs "Correctness
    # [loose]") would merge silently in gating/aggregation — reject loudly.
    from app.services.optimization_state import find_metric_name_collisions

    collisions = find_metric_name_collisions(sorted(target_scores))
    if collisions:
        logger.warning(
            "[eval-api] optimization.start.rejected reason=metric-name-collision "
            "collisions=%s", collisions,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Metric names collide after stripping the trailing '[...]' "
                f"suffix: {collisions}. Rename the metrics so their base "
                "names are distinct."
            ),
        )

    # Resolve the data source BEFORE creating optimization state so a bad
    # selection (e.g. "synthetic" with zero approved cases) doesn't leave
    # a half-initialised state file behind.
    data_source = (body.data_source if body else "raw")
    data_file_override: str | None = None
    if data_source != "raw":
        try:
            data_file_override = DatasetService().resolve_data_file(
                project_dir=project_dir,
                data_source=data_source,
                base_data_file=harness.data_file or "",
                field_mappings=harness.field_mappings or {},
            )
            logger.info(
                "[eval-api] optimization.start data_source=%s override=%s",
                data_source, data_file_override,
            )
        except ValueError as exc:
            logger.warning(
                "[eval-api] optimization.start.rejected reason=data-source error=%s", exc,
            )
            raise HTTPException(status_code=400, detail=str(exc))

    # Before creating state on the new path, read the existing record's version.
    # Because the existing-check above rejects running/stopping, any surviving
    # DDB record at create time is TERMINAL and safe to overwrite. Read-version-
    # before-write: on arch_split=True, expected_version=None (genuine first create)
    # or int (overwrite terminal record). The store's condition is
    # attribute_not_exists(version) OR version = :expected — passing None uses the
    # attribute_not_exists guard; passing an int uses the version-match guard.
    # Without threading expected_version, a bare None crashes on the 2nd start
    # because a record already exists.
    # The version rides on ``existing_handle`` from the guard read above — this used
    # to re-read the store just to recover it, which was both an extra round trip
    # and a second chance to key the read on the wrong partition.
    # Admission: cost kill switch + per-user and account job caps, and then the
    # project's durable eval lock. This route enforced NONE of them — the caps
    # and the switch lived only in the MCP dispatch module, so the launch path
    # the UI actually uses had no cost ceiling and no cross-process mutual
    # exclusion, only the process-local tracker check above.
    #
    # BEFORE the state write, not after. ``opt_repo.create`` persists
    # status="running", so admitting afterwards meant a refused launch returned
    # 429 or 409 with a durable record claiming the project was optimizing and no
    # loop anywhere behind it — the UI would show a run that could never finish.
    # Everything from here to the spawn releases the lock on failure, because
    # nothing else will: ownership only transfers to the loop once it is running.
    lock_owner_id, lock_holder_id = await _admit_optimization_launch(
        project_id, arch_split=arch_split, kind="start"
    )

    try:
        state = opt_repo.create(
            existing_handle,
            target_metric_scores=target_scores,
            max_iterations=harness.max_iterations,
            plateau_window=harness.plateau_window,
            plateau_threshold=harness.plateau_threshold,
            workload=OptimizationState.workload_fingerprint_from_harness(
                harness, data_file_override
            ),
        ).state
    except Exception as exc:
        await _release_optimization_lock(lock_owner_id, project_id, lock_holder_id)
        # ConcurrentUpdateError surfaces cleanly (HTTP 409) rather than 500
        from app.platform.metadata_store import ConcurrentUpdateError
        if isinstance(exc, ConcurrentUpdateError):
            logger.warning("[eval-api] optimization.start.concurrent-update project=%s", project_id[:8])
            raise HTTPException(
                status_code=409,
                detail="Optimization state was modified concurrently. Please retry."
            )
        raise

    # Detect starting prompt version and spawn the job-based loop.
    #
    # One try covering everything from here to the spawn: until the loop is
    # actually running, nothing else will release the lock or settle the state
    # this route already wrote. A raise anywhere in between would otherwise leave
    # the project locked until the lock's TTL and marked running forever.
    from app.services.optimization_orchestrator import detect_latest_prompt
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    try:
        prompt_version = detect_latest_prompt(project_dir)

        # The loop's data file is the synthetic/merged override when the user
        # picked one, else the harness-configured dataset. No train.csv
        # auto-split — that machinery was removed (never wired end-to-end).

        iteration_config = {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "arch_split": arch_split,
            "user_id": user,
            "model_ids": harness.models,
            "data_file": harness.data_file or "",
            "field_mappings": harness.field_mappings or {},
            "judge_model_id": harness.judge_model_id,
            "target_metric_scores": target_scores,
            "max_iterations": harness.max_iterations,
            "plateau_window": harness.plateau_window,
            "max_test_cases": None,
            "data_file_override": data_file_override,
            "prompt_version": prompt_version,
            # Hand lock ownership to the loop, whose finalizer releases it. Without
            # these keys the loop's release is a no-op and a REST-started run held no
            # durable lock at any point.
            "eval_lock_owner_id": lock_owner_id,
            "eval_lock_holder_id": lock_holder_id,
        }

        attach_error_sink(
            asyncio.create_task(
                run_optimization_loop(iteration_config),
                name=f"optimization-{project_id[:8]}",
            ),
            project_id=project_id,
            resource="optimization",
        )
    except Exception:
        await _abandon_optimization_launch(
            project_id,
            project_dir,
            arch_split=arch_split,
            user_id=user,
            lock_owner_id=lock_owner_id,
            lock_holder_id=lock_holder_id,
        )
        raise

    # Publish so the SPA flips out of any prior terminal state
    # (converged / cancelled / failed) immediately, instead of
    # waiting for the backstop poll. Pairs with the cancel/resume publishes.
    from app.services.jobs import _publish_project_event
    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "running",
        "reason": "started",
        "message": "Optimization started.",
    })

    # Notify the agent so it has the loop's origin in context
    from app.services.completion_turn import request_terminal_message
    target_scores_str = ", ".join(f"{m}={s}" for m, s in target_scores.items())
    data_source_note = f" with data override '{data_file_override}'" if data_file_override else ""
    await request_terminal_message(
        project_id,
        f"Optimization started from the UI: prompt version '{prompt_version}', "
        f"target scores {target_scores_str}, max {harness.max_iterations} iterations, "
        f"{len(harness.models)} model(s){data_source_note}.",
        lifecycle=True,
    )

    logger.info("[eval-api] optimization.started project=%s prompt=%s models=%d targets=%s",
                project_id[:8], prompt_version, len(harness.models), target_scores)
    return {"status": "started"}


@router.post("/{project_id}/optimization/resume")
async def resume_optimization(
    project_id: str,
    body: OptimizationStartRequest | None = None,
    user: UserId = Depends(require_project_write),
):
    """Resume a stopped optimization loop.

    Resumable statuses: cancelled, plateau, max_iterations,
    optimizer_exhausted, optimizer_stalled, judge_unreliable and
    generation_unreliable. Reads current harness config for model/metric
    settings (may have changed),
    detects the latest prompt version, and spawns a new orchestrator task.
    Iteration counter continues from where it left off.

    Args:
        project_id: Unique identifier of the project.
        body: Optional request body with ``data_source`` selector. Same
            semantics as ``/start`` — defaults to ``"raw"`` when omitted.

    Returns:
        Dict with resumed status and current iteration count.
    """
    project_dir = get_project_dir_or_404(project_id)

    logger.info("[eval-api] optimization.resume project=%s", project_id[:8])

    # Resolve arch_split BEFORE the state read
    arch_split = await resolve_arch_split(project_id)

    # Guard: state must exist and be resumable
    from app.services import optimization_state_repository as opt_repo

    handle = await opt_repo.read(project_id, project_dir, arch_split=arch_split)
    opt_state = handle.state

    if not opt_state:
        logger.warning("[eval-api] optimization.resume.rejected reason=no-state")
        raise HTTPException(status_code=404, detail="No optimization state found.")
    logger.info("[eval-api] optimization.resume project=%s status=%s iter=%d cap=%d",
                project_id[:8], opt_state.status, opt_state.current_iteration, opt_state.max_iter_cap)
    if opt_state.status not in (
        "cancelled",
        "plateau",
        "max_iterations",
        "optimizer_exhausted",
        "optimizer_stalled",
        "judge_unreliable",
        "generation_unreliable",
    ):
        logger.warning("[eval-api] optimization.resume.rejected reason=status-%s", opt_state.status)
        raise HTTPException(
            status_code=409,
            detail=f"Cannot resume optimization with status '{opt_state.status}'.",
        )

    # Guard: no active eval (same rationale as /start). Resuming over a
    # still-finishing eval would spawn a second loop racing the live one
    # for the harness subprocess. Trust the eval job's liveness, not the
    # on-disk optimization status.
    # Rehydrate first — same reason as /start: the tracker cannot see an eval
    # dispatched from another process.
    tracker = get_job_tracker()
    if arch_split:
        try:
            await tracker.rebuild_jobs_from_store(
                project_id, arch_split=True, refresh_terminal=True
            )
        except Exception:  # noqa: BLE001 - fall through to the durable lock
            logger.warning(
                "[eval-api] optimization.resume could not rehydrate jobs for %s; "
                "relying on the durable eval lock",
                project_id[:8],
                exc_info=True,
            )
    existing_eval = tracker.get_active_job_by_type(project_id, "eval")
    if existing_eval is not None:
        logger.warning("[eval-api] optimization.resume.rejected reason=active-eval job=%s",
                       existing_eval.job_id[:8])
        raise HTTPException(
            status_code=409,
            detail=(
                f"An eval is still finishing (job {existing_eval.job_id[:8]}). "
                "Cancel it or wait before resuming the optimization loop."
            ),
        )

    # Validate harness config BEFORE mutating state — if validation fails,
    # we don't want to leave state stuck at "running" with no orchestrator
    harness = _load_harness_config_or_400(project_dir, "resume")
    if harness is None:
        logger.warning("[eval-api] optimization.resume.rejected reason=no-config")
        raise HTTPException(status_code=400, detail="No harness config found at src/config.py.")

    if not harness.models:
        logger.warning("[eval-api] optimization.resume.rejected reason=no-models")
        raise HTTPException(status_code=400, detail="No models configured in harness config.")

    # Resolve data source BEFORE mutating state (same guard as /start).
    data_source = (body.data_source if body else "raw")
    data_file_override: str | None = None
    if data_source != "raw":
        try:
            data_file_override = DatasetService().resolve_data_file(
                project_dir=project_dir,
                data_source=data_source,
                base_data_file=harness.data_file or "",
                field_mappings=harness.field_mappings or {},
            )
            logger.info(
                "[eval-api] optimization.resume data_source=%s override=%s",
                data_source, data_file_override,
            )
        except ValueError as exc:
            logger.warning(
                "[eval-api] optimization.resume.rejected reason=data-source error=%s", exc,
            )
            raise HTTPException(status_code=400, detail=str(exc))

    # Workload drift detection. If the user edited src/config.py between
    # cancel and resume (swapped models or judge, pointed at a different
    # data file), the resumed iterations are no longer
    # comparable with the prior ones. Log loudly and stash a marker on
    # the next iteration row so the dashboard can flag it.
    current_workload = OptimizationState.workload_fingerprint_from_harness(
        harness, data_file_override
    )
    drift: dict[str, dict] = {}
    if opt_state.workload is not None and opt_state.workload != current_workload:
        for key in current_workload:
            before = opt_state.workload.get(key)
            after = current_workload[key]
            if before != after:
                drift[key] = {"before": before, "after": after}
        logger.warning(
            "[eval-api] optimization.resume.workload_drift project=%s drift=%s",
            project_id[:8], drift,
        )

    # Admission before the reset, for the same reason as /start: reset_for_resume
    # sets status="running" and the write below persists it, so admitting
    # afterwards meant a refused resume returned 429 or 409 over a durable record
    # claiming the project was optimizing with no loop behind it.
    lock_owner_id, lock_holder_id = await _admit_optimization_launch(
        project_id, arch_split=arch_split, kind="resume"
    )

    # All guards passed — NOW reset state for resume (in-memory only)
    opt_state.reset_for_resume()

    # Update the workload fingerprint to the current config and attach a
    # resume-drift marker to the next iteration's history so the UI can
    # render a "workload changed here" annotation.
    if drift:
        opt_state.workload = current_workload
        opt_state.pending_resume_drift = drift

    # Resume MUST persist the reset before spawning the loop; the loop re-reads
    # state and would see the stale terminal status otherwise. The repository routes
    # to the authoritative home and threads the version read above as the
    # conditional-write guard — an UNCONDITIONAL persist that folds in the drift
    # fields when they exist.
    # One try from the persist through to the spawn: the lock is already held and
    # the state is about to say running, and only a live loop settles either.
    from app.services.optimization_orchestrator import detect_latest_prompt
    from app.services.optimization_loop_orchestrator import run_optimization_loop

    try:
        try:
            opt_repo.write(handle)
        except Exception as exc:
            # ConcurrentUpdateError surfaces cleanly (HTTP 409) rather than 500
            from app.platform.metadata_store import ConcurrentUpdateError
            if isinstance(exc, ConcurrentUpdateError):
                logger.warning("[eval-api] optimization.resume.concurrent-update project=%s", project_id[:8])
                raise HTTPException(
                    status_code=409,
                    detail="Optimization state was modified concurrently. Please retry."
                ) from exc
            raise

        prompt_version = detect_latest_prompt(project_dir)

        # The loop's data file is the synthetic/merged override when the user
        # picked one, else the harness-configured dataset. No train.csv
        # auto-split (removed — never wired end-to-end).

        iteration_config = {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "arch_split": arch_split,
            "user_id": user,
            "model_ids": harness.models,
            "data_file": harness.data_file or "",
            "field_mappings": harness.field_mappings or {},
            "judge_model_id": harness.judge_model_id,
            "target_metric_scores": opt_state.target_metric_scores,
            "max_iterations": opt_state.max_iterations,
            "plateau_window": opt_state.plateau_window,
            "max_test_cases": None,
            "data_file_override": data_file_override,
            "prompt_version": prompt_version,
            # Loop owns the lock from here; its finalizer releases it.
            "eval_lock_owner_id": lock_owner_id,
            "eval_lock_holder_id": lock_holder_id,
        }

        attach_error_sink(
            asyncio.create_task(
                run_optimization_loop(iteration_config),
                name=f"optimization-{project_id[:8]}",
            ),
            project_id=project_id,
            resource="optimization",
        )
    except Exception:
        await _abandon_optimization_launch(
            project_id,
            project_dir,
            arch_split=arch_split,
            user_id=user,
            lock_owner_id=lock_owner_id,
            lock_holder_id=lock_holder_id,
        )
        raise

    # Publish so the SPA flips out of the prior terminal state
    # (plateau / max_iterations / optimizer_stalled) immediately.
    from app.services.jobs import _publish_project_event
    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "running",
        "reason": "resumed",
        "message": "Optimization resumed.",
    })

    # Notify the agent so it has the resume context
    from app.services.completion_turn import request_terminal_message
    target_scores_str = ", ".join(f"{m}={s}" for m, s in opt_state.target_metric_scores.items())
    drift_note = ""
    if drift:
        drift_keys = ", ".join(drift.keys())
        drift_note = f" Note: workload changed since last run — {drift_keys} modified."
    await request_terminal_message(
        project_id,
        f"Optimization resumed from the UI at iteration {opt_state.current_iteration}: "
        f"prompt version '{prompt_version}', target scores {target_scores_str}, "
        f"max {opt_state.max_iterations} iterations.{drift_note}",
        lifecycle=True,
    )

    logger.info("[eval-api] optimization.resumed project=%s iter=%d prompt=%s models=%d cap=%d",
                project_id[:8], opt_state.current_iteration, prompt_version, len(harness.models), opt_state.max_iter_cap)
    return {"status": "resumed", "iteration": opt_state.current_iteration}
