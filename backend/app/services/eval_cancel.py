# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single eval-cancel mechanism shared by the REST route and the agent tool.

Cancelling an eval means one of two very different things depending on where the
work runs: a detached Fargate worker is stopped through the compute seam using the
task ARN persisted on the durable job record, while a dev/in-process eval is a
local subprocess group that must be signalled. Getting that branch wrong is not a
cosmetic bug — a cancel that takes the wrong path stops nothing while still
reporting success, and the worker keeps spending Bedrock on target and judge
models until it finishes on its own.

That decision previously lived only inside the REST router, so the agent-facing
tool — the surface a user actually talks to — had its own pid-only copy that could
never stop a detached worker. A second copy is exactly how that recurs, so the
mechanism lives here and both entry points call it. Protocol shaping stays with
the callers: this returns an outcome, and never raises ``HTTPException`` or builds
an MCP envelope.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from app.platform import get_platform
from app.services import launch_gate
from app.services.jobs import JobNotFoundError, get_job_tracker

logger = logging.getLogger(__name__)

# What happened, for the caller to map onto its own protocol (HTTP status or MCP
# envelope). Keeping this a plain outcome rather than an exception means neither
# caller has to catch the other's error type.
#: Result of a cancel attempt.
#:
#: ``stop_failed`` is deliberately NOT terminal: the worker could not be
#: confirmed stopped, so the job stays running and the caller should retry. A
#: cancel that reports success while compute keeps billing is the failure this
#: outcome exists to prevent.
CancelOutcome = Literal[
    "cancelled", "not_found", "wrong_project", "already_terminal", "stop_failed"
]

#: How long to wait for ECS to report a stopped task, after StopTask is accepted.
#:
#: A worker that ignores SIGTERM gets a 30s ``stopTimeout`` before SIGKILL, so the
#: window has to cover that with margin — StopTask returning only means the
#: request was accepted, and the task keeps running through STOPPING. It only
#: covers it because confirmation waits for the containers to stop, not for the
#: whole task: see ``_CONTAINERS_STOPPED_STATES``.
_STOP_CONFIRM_ATTEMPTS = 20
_STOP_CONFIRM_DELAY_S = 2.0

#: ECS task states in which every container has exited — nothing of ours runs.
#:
#: DEPROVISIONING comes after the containers stop and before STOPPED. It is the
#: network teardown, the task's interface being detached and deleted, which takes
#: tens of seconds on Fargate. Waiting for STOPPED put that teardown inside the
#: window above: a worker SIGKILLed at 30s reached STOPPED at about 66s, so every
#: such cancel ran the window out, reported ``stop_failed``, and left the job for
#: the reconciler to close as failed.
_CONTAINERS_STOPPED_STATES = frozenset({"DEPROVISIONING", "STOPPED"})

#: How many times to look for a worker by its ECS ``Job`` tag before concluding
#: there is none.
#:
#: ``list_tasks`` is eventually consistent, and a task in PROVISIONING/PENDING can
#: take 5-15s to appear. One empty result is not proof: the durable handle is
#: written only after RunTask returns, so a cancel in that window sees no handle
#: AND may see no tagged task, and would otherwise mark the job cancelled while
#: the worker was still starting.
_TAG_LOOKUP_ATTEMPTS = 4
_TAG_LOOKUP_DELAY_S = 5.0

#: How long to keep looking for a worker once the launch gate says one is coming.
#:
#: A longer budget than the lookup above, and for a different reason: this is
#: positive knowledge rather than a guess. A dispatch has claimed the gate, so
#: RunTask is happening, and the wait has to cover the snapshot upload plus the
#: 5-15s before a new task becomes visible — the same launch window the eval
#: reconciler allows 60s for.
_PENDING_LAUNCH_ATTEMPTS = 12
_PENDING_LAUNCH_DELAY_S = 5.0

#: How long after a cancel is requested it may still be stopping the worker.
#:
#: The longest a cancel waits is a StopTask that fails on the handle, every tag
#: lookup, the whole launch chase, and then a stop confirmation. This is the sum of
#: those waits, doubled for the API calls between them. The eval reconciler leaves
#: a job carrying a cancel request this recent to the cancel, and closes an older
#: one as cancelled: that cancel gave up, or the process running it died.
CANCEL_IN_FLIGHT_S = 2 * (
    _TAG_LOOKUP_ATTEMPTS * _TAG_LOOKUP_DELAY_S
    + _PENDING_LAUNCH_ATTEMPTS * _PENDING_LAUNCH_DELAY_S
    + _STOP_CONFIRM_ATTEMPTS * _STOP_CONFIRM_DELAY_S
)


async def cancel_eval_job(
    project_id: str,
    job_id: str,
    *,
    user_id: str,
    project_dir: Path,
    arch_split: bool,
) -> tuple[CancelOutcome, str | None]:
    """Cancel an eval job, stopping whichever execution home is actually running it.

    Args:
        project_id: Project the job must belong to.
        job_id: Job to cancel.
        user_id: Caller's user id. The durable read that yields the cancel handle
            is scoped to it, so a job id from another tenant cannot resolve a
            handle and reach that tenant's worker.
        project_dir: Project working tree, for the optimization-state read/write.
        arch_split: Pre-resolved durable-path flag. Passed in rather than resolved
            here because it is resolved once per operation at the entry point and
            threaded down — re-resolving mid-operation reopens the dual-writer
            window that discipline exists to close.

    Returns:
        ``(outcome, status)``. ``status`` carries the job's terminal status when
        the outcome is ``already_terminal``, and is ``None`` otherwise.
    """
    # Imported here, not at module scope: these live in an API router module, so a
    # module-scope import would point a service at a router and invert the
    # dependency direction. Same justification the durable progress bridge already
    # carries for the same two helpers — and reusing them rather than re-deriving
    # the ARN test matters, because a second, subtly different notion of "is this a
    # Fargate handle" is precisely how a cancel silently reaches nothing.
    from app.api.eval_jobs import _is_fargate_handle, _resolve_compute_handle

    tracker = get_job_tracker()

    # Rebuild from the durable store BEFORE reading the tracker. The process doing
    # the cancelling may hold a cold or stale replica — an agent turn runs in a
    # separate execution home from the one that dispatched the eval — and without
    # this a live job simply reads as missing. ``refresh_terminal`` also means the
    # durable record wins when memory and the store disagree about terminal state,
    # which is the correct answer for a cancel decision.
    try:
        await tracker.rebuild_jobs_from_store(
            project_id, arch_split, refresh_terminal=True
        )
    except Exception as exc:  # noqa: BLE001 — a store hiccup must not fail the cancel
        logger.warning(
            "[eval-cancel] durable rebuild failed for project=%s: %s",
            project_id[:12], exc,
        )

    try:
        job = tracker.get_job_for_project(project_id, job_id)
    except JobNotFoundError:
        job = tracker.get_job(job_id)
        if job is None:
            return "not_found", None
        return "wrong_project", None

    if job.status not in ("pending", "running"):
        return "already_terminal", job.status

    compute_handle = await _resolve_compute_handle(user_id, project_id, job_id)

    # Detached when there is a Fargate handle, and ALSO when the durable path is
    # in use with no handle yet: the handle is written after RunTask returns, so a
    # cancel arriving in that window sees a handle-null record for a worker that
    # is running. The condition is on the compute, never on the flag alone — a
    # Fargate handle means detached compute whatever the flag currently says, and
    # treating it as local would kill a pid this process never owned.
    if _is_fargate_handle(compute_handle) or arch_split:
        # Stop the compute BEFORE recording the terminal state.
        #
        # The old order marked the job cancelled first, published that to the
        # SPA, released the project's eval lock — and only then attempted
        # StopTask, with no handling for its failure. So a failed stop left the
        # UI saying "cancelled" while the worker kept invoking models and
        # billing, the lock free for a second eval to start against the same
        # project, and a retry short-circuiting on "already terminal" without
        # ever trying to stop it again.
        #
        # That ordering leaves the job non-terminal while its worker stops, so
        # the record has to say why first. See ``_record_cancel_request``.
        await _record_cancel_request(user_id, project_id, job_id)
        stopped = await _stop_detached_worker(
            compute_handle,
            owner_id=user_id,
            project_id=project_id,
            job_id=job_id,
            is_fargate_handle=_is_fargate_handle,
        )
        if not stopped:
            # Leave the job non-terminal on purpose. It stays visible as running,
            # the caller can retry, and the reconciler's orphan sweep is the
            # backstop — none of which is possible once it reads terminal.
            logger.error(
                "[eval-cancel] could not stop worker for job=%s — leaving the job "
                "non-terminal so a retry and the reconciler can still act",
                job_id[:8],
            )
            return "stop_failed", job.status

        await _adopt_worker_progress(job, user_id, project_id, job_id)
        tracker.cancel_job(job_id)
        logger.info("[eval-cancel] job.cancelled job=%s (worker stopped)", job_id[:8])
    else:
        # Dev/in-process. Here the terminal write comes FIRST and that ordering is
        # load-bearing: the harness checks its own job's status after the
        # subprocess dies to decide whether to discard the results it produced, so
        # marking cancelled after the kill would race that read. The compute is a
        # local child process rather than billable detached compute, so the
        # exposure the cloud path has does not apply.
        tracker.cancel_job(job_id)
        logger.info("[eval-cancel] job.cancelled job=%s", job_id[:8])

        # Signal the subprocess group (SIGTERM → grace → SIGKILL, so `uv run` and
        # its grandchild python both stop) and cancel the driver task. Both
        # dispatcher name shapes are matched so this covers a manually dispatched
        # eval and the optimization orchestrator's child eval alike.
        from app.services.harness_runner import kill_harness_subprocess

        if job.pid:
            logger.info("[eval-cancel] job.sigterm pid=%d", job.pid)
            await kill_harness_subprocess(job.pid)

        candidate_names = {f"job-eval-{job_id[:8]}", f"eval-{job_id[:8]}"}
        for task in asyncio.all_tasks():
            if task.get_name() in candidate_names:
                task.cancel()
                logger.info("[eval-cancel] cancelled task %s", task.get_name())
                break

    await _stop_optimization_loop(
        project_id, project_dir, user_id=user_id, arch_split=arch_split
    )
    return "cancelled", None


async def _record_cancel_request(user_id: str, project_id: str, job_id: str) -> None:
    """Mark the job's durable record as being cancelled, before the stop starts.

    While the worker stops, the job is deliberately non-terminal, and ECS has
    already taken the task off its running list, because StopTask changes the
    task's desired status at once. To the eval reconciler that is exactly a
    worker that crashed, and it closed such a job as failed. A failed manual
    eval gets its own chat turn, so the user was told of a failure before the
    cancel's own line arrived. The reconciler checks for this mark instead.

    Best-effort: a failed write is logged and the cancel goes ahead, because
    stopping a worker that is still invoking models matters more than the label.
    """
    try:
        await get_platform().metadata_store.mark_cancel_requested(
            user_id, project_id, job_id
        )
    except Exception:  # noqa: BLE001 - never block the stop on the mark
        logger.warning(
            "[eval-cancel] could not record the cancel request for job=%s; a "
            "reconciler pass during the stop may close the job as failed",
            job_id[:8], exc_info=True,
        )


async def _adopt_worker_progress(job, user_id: str, project_id: str, job_id: str) -> None:
    """Take the stopped worker's last recorded progress as the job's own.

    A detached eval runs in its worker, not in this process, so the progress
    held here is whatever it was when this process first saw the job, and it
    does not move after that. The worker writes its progress to the durable
    record as each model finishes, so once it has stopped, the record says how
    far it got. The cancelled state written next carries that, and so does the
    chat line announcing a cancel from the UI.

    A failed read keeps the progress already held: it is stale, but the cancel
    itself has succeeded and must still be recorded.
    """
    try:
        record = await get_platform().metadata_store.get_job(
            user_id, project_id, job_id, consistent=True
        )
    except Exception:  # noqa: BLE001 - the cancel has succeeded either way
        logger.warning(
            "[eval-cancel] could not read the stopped worker's progress for job=%s",
            job_id[:8], exc_info=True,
        )
        return
    progress = (record or {}).get("progress")
    if progress:
        job.progress = dict(progress)


async def _stop_detached_worker(
    compute_handle: str | None,
    *,
    owner_id: str,
    project_id: str,
    job_id: str,
    is_fargate_handle,
) -> bool:
    """Stop a detached eval worker. Returns True only if it is confirmed stopped.

    Tries the durable task ARN first, then the ECS ``Job`` tag. The tag path is
    not a fallback for tidiness: the durable ``compute_handle`` is written AFTER
    ``RunTask`` returns, so a cancel arriving in that window — or after a
    coordinator died in it — sees a handle-null record while a tagged worker is
    very much running. Treating that as "nothing to stop" and falling through to
    the in-process branch killed a pid the coordinator never owned and left the
    worker alone.

    Args:
        compute_handle: Durable handle value, possibly None.
        owner_id: Owner tag, cross-checked before stopping a tagged task.
        project_id: Project tag, cross-checked likewise.
        job_id: The job whose worker should stop.
        is_fargate_handle: Predicate shared with the eval-jobs router, so this
            cannot develop a second, subtly different idea of a Fargate handle.

    Returns:
        True when a worker was stopped, or when there is provably none to stop.
    """
    from app.platform.job_compute import get_job_compute

    compute = get_job_compute()

    if is_fargate_handle(compute_handle):
        logger.info("[eval-cancel] job.stop-task handle=%s", compute_handle)
        try:
            return await _stop_and_confirm(compute, compute_handle, job_id=job_id)
        except Exception:  # noqa: BLE001 - fall through to the tag path
            logger.warning(
                "[eval-cancel] StopTask failed for handle=%s; trying the Job tag",
                compute_handle, exc_info=True,
            )

    # No usable handle, or StopTask on it failed. Retry the tag lookup rather
    # than trusting a single miss: ECS list_tasks is eventually consistent, so a
    # worker launched moments ago may not appear yet — and the handle is written
    # only after RunTask returns, which is exactly when a cancel is most likely to
    # find no handle. Concluding "nothing to stop" from one empty list would mark
    # the job cancelled while the worker was still starting up.
    last_error: Exception | None = None
    for attempt in range(1, _TAG_LOOKUP_ATTEMPTS + 1):
        try:
            task_arn = await compute.cancel_by_tag(owner_id, project_id, job_id)
        except Exception as e:  # noqa: BLE001 - retried, then reported
            last_error = e
            logger.warning(
                "[eval-cancel] tag-based stop attempt %d failed for job=%s: %s",
                attempt, job_id[:8], e,
            )
        else:
            if task_arn:
                # Same contract as the direct-ARN path: StopTask was accepted, so
                # now confirm it. Returning success here on acceptance alone gave
                # the two addressing modes different guarantees for one word, and
                # this is the mode a cancel in the launch window actually takes.
                logger.info(
                    "[eval-cancel] job.stop-task-by-tag job=%s (attempt %d)",
                    job_id[:8], attempt,
                )
                return await _confirm_stopped(compute, task_arn, job_id=job_id)
        if attempt < _TAG_LOOKUP_ATTEMPTS:
            await asyncio.sleep(_TAG_LOOKUP_DELAY_S)

    if last_error is not None:
        logger.error(
            "[eval-cancel] every tag-based stop attempt failed for job=%s",
            job_id[:8], exc_info=last_error,
        )
        return False

    # Repeated lookups found no task carrying this job's tag. That is a true
    # observation, and on its own it still does not mean there is nothing to stop:
    # the task may be in the FUTURE rather than merely invisible. The job became
    # cancellable before the model preflight, and the snapshot upload and RunTask
    # come after it, so this whole window can elapse with the dispatch yet to
    # launch anything. Retrying a lookup cannot distinguish the two — only the
    # launch gate can, by taking the decision away from the dispatch.
    if await launch_gate.claim_for_cancel(owner_id, project_id, job_id):
        logger.info(
            "[eval-cancel] no running worker for job=%s and no launch can start "
            "one now — nothing to stop",
            job_id[:8],
        )
        return True

    # A dispatch holds the gate, so a worker exists or is seconds from existing.
    return await _stop_a_launching_worker(
        compute, owner_id=owner_id, project_id=project_id, job_id=job_id
    )


async def _stop_and_confirm(compute, task_arn: str, *, job_id: str) -> bool:
    """StopTask a task, then report whether its containers actually stopped.

    For callers holding an ARN that nothing has stopped yet. ``cancel_by_tag``
    issues its own StopTask on the task it matches, so that path uses
    :func:`_confirm_stopped` instead — calling this on its ARN would stop the same
    task twice.

    Args:
        compute: The job-compute adapter.
        task_arn: Task to stop.
        job_id: Job the task belongs to, for logging.

    Returns:
        True only when ECS reported the task's containers stopped.
    """
    from app.platform.job_compute import JobComputeHandle

    await compute.cancel(JobComputeHandle(kind="fargate_task", value=task_arn))
    return await _confirm_stopped(compute, task_arn, job_id=job_id)


async def _confirm_stopped(compute, task_arn: str, *, job_id: str) -> bool:
    """Report whether an already-stopped task's containers actually stopped.

    One place, because every addressing mode needs the identical guarantee and
    having it written per-mode is how they came to differ: StopTask returning means
    ECS accepted the request, not that the task stopped. ECS sends SIGTERM and
    allows a 30s ``stopTimeout`` before SIGKILL, so the task sits in STOPPING and
    keeps invoking models meanwhile, and treating acceptance as termination
    released the project's eval lock while the worker was still billing.

    Args:
        compute: The job-compute adapter.
        task_arn: Task that StopTask has already been issued against.
        job_id: Job the task belongs to, for logging.

    Returns:
        True only when ECS reported the task's containers stopped.
    """
    from app.platform.job_compute import JobComputeHandle

    handle = JobComputeHandle(kind="fargate_task", value=task_arn)
    if await _await_task_stopped(compute, handle, job_id=job_id):
        return True
    logger.error(
        "[eval-cancel] worker for job=%s did not stop within %.0fs of StopTask",
        job_id[:8], _STOP_CONFIRM_ATTEMPTS * _STOP_CONFIRM_DELAY_S,
    )
    return False


async def _stop_a_launching_worker(
    compute, *, owner_id: str, project_id: str, job_id: str
) -> bool:
    """Chase a worker whose launch is in flight, and stop it once it appears.

    Reached only when the launch gate reports a dispatch claimed it. That is
    positive knowledge rather than an inference: RunTask is happening, so a lookup
    miss here means "not yet" and never "none", and the loop is entitled to wait.

    Both addresses are retried each pass — the durable ``compute_handle``, which
    the dispatch writes as soon as RunTask returns, and the ECS ``Job`` tag, which
    takes another 5-15s to appear.

    Args:
        compute: The job-compute adapter.
        owner_id: Owner, for the durable read and the tag cross-check.
        project_id: Project, likewise.
        job_id: The job whose worker should stop.

    Returns:
        True once the worker is confirmed stopped; False if the window elapsed
        without ever finding it, which leaves the job non-terminal for the retry
        and the reconciler.
    """
    from app.api.eval_jobs import _is_fargate_handle, _resolve_compute_handle

    for attempt in range(1, _PENDING_LAUNCH_ATTEMPTS + 1):
        try:
            handle_value = await _resolve_compute_handle(owner_id, project_id, job_id)
        except Exception as e:  # noqa: BLE001 - try the tag instead
            handle_value = None
            logger.warning(
                "[eval-cancel] could not re-read the handle for job=%s "
                "(attempt %d): %s",
                job_id[:8], attempt, e,
            )

        if _is_fargate_handle(handle_value):
            logger.info(
                "[eval-cancel] the launching worker for job=%s published a handle "
                "(attempt %d)",
                job_id[:8], attempt,
            )
            try:
                return await _stop_and_confirm(compute, handle_value, job_id=job_id)
            except Exception:  # noqa: BLE001 - fall through to the tag
                logger.warning(
                    "[eval-cancel] StopTask failed for the newly published handle "
                    "on job=%s; trying the Job tag",
                    job_id[:8], exc_info=True,
                )

        try:
            task_arn = await compute.cancel_by_tag(owner_id, project_id, job_id)
        except Exception as e:  # noqa: BLE001 - retried
            task_arn = None
            logger.warning(
                "[eval-cancel] tag lookup for the launching worker on job=%s "
                "failed (attempt %d): %s",
                job_id[:8], attempt, e,
            )
        if task_arn:
            # cancel_by_tag already issued StopTask against this ARN.
            return await _confirm_stopped(compute, task_arn, job_id=job_id)

        if attempt < _PENDING_LAUNCH_ATTEMPTS:
            await asyncio.sleep(_PENDING_LAUNCH_DELAY_S)

    logger.error(
        "[eval-cancel] a launch holds the gate for job=%s but no worker could be "
        "found within %.0fs — leaving the job non-terminal so a retry and the "
        "reconciler can still act",
        job_id[:8], _PENDING_LAUNCH_ATTEMPTS * _PENDING_LAUNCH_DELAY_S,
    )
    return False


async def _await_task_stopped(compute, handle, *, job_id: str) -> bool:
    """Poll a task until ECS reports its containers stopped. True only if they did.

    Confirmation requires a task ECS returned whose ``lastStatus`` is in
    ``_CONTAINERS_STOPPED_STATES``. Two things that are NOT that:

    * A response carrying no task. ``status`` reports ``UNKNOWN`` for it, and this
      keeps polling. Immediately after StopTask an empty result is ordinary
      eventual consistency, so reading absence as termination confirmed stops that
      had not happened.
    * A failed DescribeTasks. One lost observation is not a verdict either — a
      throttle must not be able to decide the outcome of a cancel — so it costs an
      attempt and the loop continues.

    Exhausting the window without ever seeing a stopped state returns False,
    which leaves the job non-terminal for the retry and the reconciler.
    """
    state: str | None = None
    for attempt in range(1, _STOP_CONFIRM_ATTEMPTS + 1):
        try:
            state = await compute.status(handle)
        except Exception as e:  # noqa: BLE001 - one lost observation, not a verdict
            state = None
            logger.warning(
                "[eval-cancel] could not read worker status for job=%s "
                "(attempt %d): %s",
                job_id[:8], attempt, e,
            )
        if state in _CONTAINERS_STOPPED_STATES:
            logger.info(
                "[eval-cancel] worker for job=%s confirmed stopped (state=%s, "
                "attempt %d)",
                job_id[:8], state, attempt,
            )
            return True
        if attempt < _STOP_CONFIRM_ATTEMPTS:
            await asyncio.sleep(_STOP_CONFIRM_DELAY_S)
    logger.error(
        "[eval-cancel] worker for job=%s never reported its containers stopped; "
        "last observed state=%s",
        job_id[:8], state,
    )
    return False


async def _stop_optimization_loop(
    project_id: str,
    project_dir: Path,
    *,
    user_id: str,
    arch_split: bool,
) -> None:
    """Ask a running optimization loop to stop, because its eval was cancelled.

    Writes the *stopping* state rather than a terminal one: the loop's own
    cancellation branch publishes the real terminal on exit, so writing terminal
    here would race that write.

    On the durable path the state is read from the metadata store and written back
    with the version from that same read, because the store's write is conditional
    on it. A lost version race is logged and swallowed — the worker has already
    been stopped by the time this runs, and downgrading a successful cancel to a
    reported failure over a state-write race would be a worse answer than a
    slightly stale bar.

    Args:
        project_id: Project whose loop should stop.
        project_dir: Project working tree.
        user_id: Owner for the durable read/write.
        arch_split: Pre-resolved durable-path flag.
    """
    from app.platform.metadata_store import ConcurrentUpdateError
    from app.services import optimization_state_repository as opt_repo

    handle = await opt_repo.read(
        project_id, project_dir, arch_split=arch_split, owner_id=user_id
    )
    # Deliberately ``!= "running"`` and NOT ``is_live()``: a run already ``stopping``
    # needs no second stop write, so widening this to include it would burn a
    # conditional write and could lose a version race for no gain.
    if handle.state is None or handle.state.status != "running":
        return

    handle.state.status = "stopping"
    try:
        opt_repo.write(handle)
    except ConcurrentUpdateError as exc:
        # A lost race here means another writer already moved the run on (the
        # worker stopped itself). The cancel itself still succeeded.
        logger.warning(
            "[eval-cancel] optimization stop write lost a version race "
            "project=%s: %s (worker already stopped; not failing the cancel)",
            project_id[:8], exc,
        )
        return

    logger.info(
        "[eval-cancel] optimization.stopping project=%s reason=eval-job-cancelled",
        project_id[:8],
    )

    # Flip the SPA's optimization bar immediately instead of making it wait on the
    # next poll. The loop publishes again when it actually exits.
    from app.services.jobs import _publish_project_event

    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "terminal",
        "reason": "cancelled",
        "message": "Optimization cancelled.",
    })
