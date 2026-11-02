# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess orchestration for standalone harness evaluations.

The harness project (``projects/<id>/run_eval.py``) is executed as a
separate process so it can exercise its own virtualenv. This module owns
the subprocess lifecycle: spawn, stream stderr + stdout concurrently (the
stderr drain is mandatory — DeepEval floods the 64KB pipe buffer and
deadlocks otherwise), decode progress events, persist results, and surface
structured progress to the JobTracker.
"""

import asyncio
import json
import logging
import os
import signal
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.services.active_eval_state import clear_active_eval, record_active_eval
from app.services.jobs import get_job_tracker
from app.services.project_paths import safe_path
from app.services.resource_error import Codes, ResourceError
from app.services.sdk_logging import subprocess_log_line


def _project_subprocess_env() -> dict[str, str]:
    """Return a clean env dict for the harness subprocess.

    Strips ``VIRTUAL_ENV`` (parent venv path) so ``uv run`` activates the
    project's own ``.venv`` without warnings.

    Sets ``DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600``. DeepEval's
    async ``evaluate()`` wraps every test case task in
    ``asyncio.wait_for(timeout=get_gather_timeout_seconds())`` where
    ``get_gather_timeout_seconds = DEEPEVAL_PER_TASK_TIMEOUT_SECONDS
    + DEEPEVAL_TASK_GATHER_BUFFER_SECONDS`` (180s + ~10-60s default).
    When the gather deadline fires, every still-queued task is
    cancelled and DeepEval silently drops those cases from
    ``test_results`` — which is how models end up with different case
    totals, and why the harness runs a slow sequential reconciliation pass.
    Per-call retry safety lives at the harness's BedrockRuntimeModel layer
    (``@_bedrock_retry`` with botocore's read timeout from
    ``BEDROCK_READ_TIMEOUT_S``, 300s by default, × 4 attempts), so a
    generous gather budget can't hang indefinitely — it just stops DeepEval
    from prematurely cancelling the queue.

    Set on the runner side (not via ``EvalConfig``) so existing
    project scaffolds pick it up without re-scaffold.

    ``MANTLE_*`` env knobs (``MANTLE_MAX_CONNECTIONS``,
    ``MANTLE_MAX_KEEPALIVE``, ``MANTLE_BOOT_SMOKE``, etc.) are
    automatically forwarded via ``os.environ.copy()`` —
    Do NOT add explicit ``MANTLE_*`` env-set
    lines here; that creates a third surface to keep in sync.
    Asserted by
    ``tests/test_subprocess_env_mantle.py::test_production_factory_propagates_mantle_env``.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    # Also strip UV_PROJECT_ENVIRONMENT: when the coordinator's own env pins uv
    # to the backend venv (as it does in dev, where the backend venv is
    # relocated), a child ``uv run``/``uv sync`` in the *harness project* would
    # resolve — and re-sync — the coordinator's venv instead of the project's
    # own, silently mutating the backend environment out from under a running
    # process. The harness project must resolve its own environment.
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"
    return env

logger = logging.getLogger(__name__)

#: Tags marking a log line as VERBATIM subprocess console output rather than an
#: app measurement.
#:
#: Why this exists: DeepEval prints its own pass-rate summary to stderr, and the
#: passthrough logged it with the same ``[harness <id>]`` prefix the app uses for
#: its own lines. Two different numbers, computed differently — DeepEval counts
#: every metric result, the app counts cases against the gating metric — read as
#: equally authoritative. An operator comparing them has no way to tell which is
#: the app's answer.
#:
#: Tagged by STREAM rather than by attributing lines to DeepEval, because the
#: same stderr also carries genuine harness tracebacks and uv output. Labelling
#: all of it ``[deepeval]`` would be a confident, wrong attribution, and the only
#: way to do better would be matching on the library's English output — which
#: breaks the moment DeepEval rewords a line.
_STDERR_TAG = "[subprocess stderr]"
_STDOUT_TAG = "[subprocess stdout]"

#: How a harness run ended, returned so a caller can branch on the outcome.
#:
#: ``run_eval_via_harness`` records the outcome on the job record and does not
#: raise for an ordinary failure, so "did it raise?" cannot distinguish a failed
#: eval from a successful one. The detached worker needs that distinction to pick
#: its process exit code, and previously did not have it: every non-raising
#: failure looked like success and the container exited 0.
HarnessOutcome = Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True)
class HarnessRun:
    """Outcome of a harness run, plus the results when completion is deferred.

    Attributes:
        outcome: How the run ended.
        results: Per-model run dicts, set ONLY when ``defer_completion`` was
            requested and the run completed. The caller owns recording
            completion in that case, so it needs the payload.
        output_file: Filename the harness wrote, for the same reason.
    """

    outcome: HarnessOutcome
    results: Any | None = None
    output_file: str | None = None

#: Bytes of subprocess stderr retained in memory, at each end of the stream.
#:
#: The failure report needs the head (what went wrong first) and the tail (where
#: it ended), never the middle. Retaining both ends bounds memory at twice this
#: value no matter how much a runaway harness writes; the previous code kept
#: every byte for the lifetime of the job.
_STDERR_RETAINED_BYTES = 8192

#: Ceiling for ``uv sync``. Generous on purpose: a cold uv cache downloading and
#: building the harness dependency set legitimately takes minutes, and a ceiling
#: that rejects a first run is worse than none. It exists to bound a WEDGED sync,
#: not to make a slow one fail.
#:
#: Public rather than underscore-private because ``mcp_tools.harness`` runs the
#: same command during validation and must use the same number — two ceilings for
#: one operation is how they drift.
UV_SYNC_TIMEOUT_S = 600

#: Ceiling for the import-and-construct validation probe. It imports the harness
#: and builds its metrics, so it is short by nature; past this it is stuck.
VALIDATION_PROBE_TIMEOUT_S = 180


def build_harness_command(
    uv_path: str,
    *,
    model_ids: list[str],
    prompt_version: str,
    json_progress: bool = True,
    source: str = "manual",
    max_test_cases: int | None = None,
    judge_model_id: str | None = None,
    data_file_override: str | None = None,
    max_concurrent: int | None = None,
) -> list[str]:
    """Build the ``uv run python run_eval.py`` argv for a harness invocation.

    The SINGLE source of truth for the harness command line, shared by the
    real eval run (``run_eval_via_harness``) and the validation smoke eval
    (``validate_harness``). They previously each built their own arg list,
    which let them drift — the smoke eval omitted ``--data-file``, so the
    "unrecognized arguments: --data-file" crash slipped through validation
    and only fired at real-run dispatch. Routing both through this builder
    means any flag the real run can emit, the smoke eval emits too (for the
    same inputs), so a missing-flag regression fails at validation time.

    Each flag here MUST be defined in the template's ``run_eval.py``
    argparse — ``tests/test_harness_flag_contract.py`` asserts exactly that.

    Args:
        uv_path: Absolute path to the ``uv`` binary.
        model_ids: Target model IDs (joined comma-separated for ``--models``).
        prompt_version: Prompt version stem (``--prompt-version``).
        json_progress: Emit ``--json-progress`` (always True in platform use).
        source: ``--source`` value (``"manual"``, ``"optimization"``, or
            ``"validation"``). ``"validation"`` is the 1-case smoke run
            from ``validate_harness`` and is filtered out of ``list_runs``
            so smokes never pollute the dashboard.
        max_test_cases: ``--max-examples`` when > 0 (smoke passes 1).
        judge_model_id: ``--judge-model`` when set.
        data_file_override: ``--data-file`` when set (bare or absolute).
        max_concurrent: ``--max-concurrent`` when set (smoke passes 1).

    Returns:
        The full argv list for ``asyncio.create_subprocess_exec``.
    """
    cmd = [uv_path, "run", "python", "-u", "run_eval.py"]
    cmd.extend(["--models", ",".join(model_ids)])
    cmd.extend(["--prompt-version", prompt_version])
    if max_test_cases and max_test_cases > 0:
        cmd.extend(["--max-examples", str(max_test_cases)])
    if judge_model_id:
        cmd.extend(["--judge-model", judge_model_id])
    if data_file_override:
        cmd.extend(["--data-file", data_file_override])
    if max_concurrent and max_concurrent > 0:
        cmd.extend(["--max-concurrent", str(max_concurrent)])
    if json_progress:
        cmd.append("--json-progress")
    cmd.extend(["--source", source])
    return cmd


async def kill_harness_subprocess(pid: int, *, grace_seconds: float = 3.0) -> None:
    """Kill a harness subprocess group, escalating SIGTERM → SIGKILL.

    The harness is spawned with ``start_new_session=True`` so its PID is the
    process-group leader; we need to signal the **group** to reach both the
    ``uv run`` parent and its Python grandchild. SIGTERM on the immediate
    child alone lets DeepEval finish its current batch and produce orphan
    result files — escalation to SIGKILL after a short grace ensures the
    subprocess actually stops.

    Safe to call with a stale PID: all lookups tolerate ``ProcessLookupError``.

    Args:
        pid: Process ID of the harness subprocess (group leader).
        grace_seconds: Seconds to wait after SIGTERM before sending SIGKILL.
            3s is long enough for DeepEval to finish an in-flight Bedrock
            call; short enough that the user sees the cancel take effect.
    """
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except OSError as e:
        logger.warning("[kill] getpgid(%d) failed: %s", pid, e)
        pgid = None

    # Signal the group only when ``pid`` LEADS it. A process spawned without
    # ``start_new_session`` inherits this process's group, so ``killpg`` on its
    # pgid would signal the coordinator itself along with everything else in
    # that group. Every spawn here sets the flag; this keeps a future one that
    # forgets from turning a subprocess kill into killing the service.
    if pgid is not None and pgid != pid:
        logger.warning(
            "[kill] pid=%d does not lead its process group (pgid=%d); "
            "signalling the process alone",
            pid, pgid,
        )
        pgid = None

    def _send(sig: int) -> bool:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return True
            except ProcessLookupError:
                return False
            except OSError as err:
                logger.warning("[kill] killpg(%d, %d) failed: %s", pgid, sig, err)
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError as err:
            logger.warning("[kill] kill(%d, %d) failed: %s", pid, sig, err)
            return False

    if not _send(signal.SIGTERM):
        return
    logger.info("[kill] SIGTERM group=%s pid=%d", str(pgid), pid)

    await asyncio.sleep(grace_seconds)

    try:
        os.kill(pid, 0)
        still_alive = True
    except ProcessLookupError:
        still_alive = False

    if still_alive:
        if _send(signal.SIGKILL):
            logger.info("[kill] Escalated to SIGKILL group=%s pid=%d", str(pgid), pid)


class SubprocessTimeout(Exception):
    """A bounded helper subprocess outlived its timeout and was killed.

    Carries the label and timeout so a caller can phrase its own user-facing
    error without re-deriving either.
    """

    def __init__(self, label: str, timeout_s: float) -> None:
        super().__init__(f"{label} exceeded {timeout_s:g}s")
        self.label = label
        self.timeout_s = timeout_s


async def run_bounded_subprocess(
    argv: list[str],
    *,
    timeout_s: float,
    cwd: str,
    env: dict[str, str],
    label: str,
) -> tuple[int | None, bytes, bytes]:
    """Run ``argv`` to completion under a timeout, killing its GROUP on expiry.

    Every helper subprocess here — ``uv sync``, the validation probe, the smoke
    eval, the metric self-test — goes through this rather than calling
    ``communicate()`` directly, because three things have to happen together and
    each was missing somewhere:

    **A timeout.** ``uv sync`` and the validation probe awaited ``communicate()``
    with no bound at all, so a wedged process held the turn open indefinitely.

    **A GROUP kill.** ``uv run python …`` makes the interpreter a grandchild, and
    that grandchild is the process calling Bedrock. Killing the immediate child
    leaves it running and billing. ``start_new_session=True`` makes the PID a
    group leader so :func:`kill_harness_subprocess` can signal the whole group,
    escalating SIGTERM to SIGKILL.

    **A reap.** ``kill`` only asks. Two sites killed and returned without ever
    awaiting the process, leaving a zombie holding its pipes, so the event loop
    kept a transport for a process nobody would collect.

    The group is killed on CANCELLATION as well as on timeout. A cancelled turn
    or job stops awaiting this call, and a group left running then is the same
    spend the timeout exists to stop, reached through the other exit.

    Args:
        argv: Command and arguments. No shell is involved.
        timeout_s: Wall-clock ceiling. Generous rather than tight — the cost of
            a ceiling set too low is a legitimate run that cannot start.
        cwd: Working directory for the child.
        env: Environment for the child.
        label: Human-readable name used in the timeout message and logs.

    Returns:
        ``(returncode, stdout, stderr)``.

    Raises:
        SubprocessTimeout: The process outlived ``timeout_s``. Its group has been
            killed and reaped by the time this is raised.
        asyncio.CancelledError: The caller was cancelled. The group has been
            sent SIGKILL before this propagates.
    """
    # Audited. ``argv`` is a LIST and no shell is involved, so a metacharacter in
    # any element arrives as a literal argument. The callers pass one of: a fixed
    # ``uv sync``; ``python -c`` with a program that is a string literal in this
    # repo or a file read from it (the validation and metric self-test probes);
    # or ``build_harness_command``'s output, where each value is its own element
    # after the flag it belongs to. The project's own harness code runs by
    # design — executing it is what an eval is.
    # nosemgrep: dangerous-asyncio-create-exec-audit
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Load-bearing: without its own session the PID is not a group leader and
        # the group kill below degrades to killing only this process.
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.CancelledError:
        # SIGKILL at once, not the SIGTERM-then-grace of the timeout path. Nobody
        # will read this output, and the grace period is an await: a second
        # cancel landing inside it would abandon the kill half-done. The child
        # watcher still reaps the process, so no zombie is left behind.
        logger.info(
            "[subprocess] %s cancelled — killing group pid=%d", label, proc.pid,
        )
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as err:
            logger.warning(
                "[subprocess] killpg(%d) on cancel failed: %s", proc.pid, err,
            )
        raise
    except asyncio.TimeoutError:
        logger.warning(
            "[subprocess] %s exceeded %gs — killing group pid=%d",
            label, timeout_s, proc.pid,
        )
        await kill_harness_subprocess(proc.pid)
        # Collect the exit status. Bounded because a process wedged in
        # uninterruptible IO will not die even on SIGKILL, and hanging here would
        # turn a subprocess timeout into a stuck turn — the thing being fixed.
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(
                "[subprocess] %s survived SIGKILL; leaving it unreaped pid=%d",
                label, proc.pid,
            )
        raise SubprocessTimeout(label, timeout_s) from None
    return proc.returncode, stdout, stderr


async def hard_cancel_project_eval(project_id: str) -> int:
    """Synchronously cancel and SIGKILL every active eval for a project.

    Replaces the pre-existing fire-and-forget pattern — ``JobTracker.
    cancel_job`` schedules ``kill_harness_subprocess`` via
    ``loop.create_task(...)`` and returns immediately, so the SIGTERM
    →3s→SIGKILL escalation may not run to completion before the API
    response returns. Under load (or if the calling task tears down)
    the kill silently never completes and the subprocess keeps
    burning Bedrock credits.

    This helper *awaits* every kill so the API endpoint doesn't return
    until the subprocess is dead. It also handles the parent-cancels-
    child case for ``optimization_iteration`` jobs: those don't carry
    a pid themselves, but the endpoint that targets a parent must
    propagate the kill to the running child eval.

    Idempotent — calling twice is a no-op (already-terminal jobs are
    skipped). Best-effort on already-dead PIDs (``ProcessLookupError``
    is swallowed inside ``kill_harness_subprocess``).

    Args:
        project_id: Project whose active jobs should be cancelled.

    Returns:
        Number of jobs that were active and got cancelled.
    """
    tracker = get_job_tracker()
    pids_to_kill: list[int] = []
    fargate_jobs: list[tuple[str, str]] = []  # (job_id, owner_id) for durable handle lookup
    cancelled_count = 0

    # Snapshot the active jobs first so the loop iterates over an
    # immutable list (state mutations below would otherwise race).
    active_jobs = list(tracker.get_active_jobs(project_id))

    for active in active_jobs:
        # We DON'T call ``tracker.cancel_job`` here because that path
        # also schedules a fire-and-forget ``kill_harness_subprocess``
        # via ``loop.create_task(...)``. We want exactly one awaited
        # kill per pid, not two killpg storms (one fire-and-forget,
        # one awaited). Instead we drive the cancel state directly
        # and aggregate pids for the synchronous gather below.
        try:
            tracker.update_state(active.job_id, "cancelled")
        except KeyError:
            continue
        active.progress["message"] = "Cancelled by user"
        if active._task is not None and not active._task.done():
            active._task.cancel()
        cancelled_count += 1
        if active.type == "eval":
            # arch_split (new-path) eval jobs run on a Fargate worker; the
            # task ARN is the durable cancel handle (survives control-plane
            # restart). Old-path/dev jobs use the process-group PID kill.
            if active.config.get("arch_split"):
                owner_id = active.config.get("owner_id") or active.config.get("user_id")
                if owner_id:
                    fargate_jobs.append((active.job_id, owner_id))
            elif active.pid:
                pids_to_kill.append(active.pid)

    # StopTask every Fargate job. Tag-first is PRIMARY: the durable
    # ``compute_handle`` can be null on an eval record (a worker's full-item
    # rewrite used to wipe it — workers now register with
    # ``create_job(durable_write=False)`` — and records written that way stay
    # handle-null), so cancelling by ECS Owner/Project/Job tags
    # reaches the worker regardless of the record. The durable
    # ``compute_handle`` StopTask is kept as a secondary fast path when the
    # tag lookup finds no live task but the record does carry a handle.
    for job_id, owner_id in fargate_jobs:
        try:
            from app.platform import get_platform
            from app.platform.aws.fargate_task_compute import FargateTaskCompute
            from app.platform.job_compute import JobComputeHandle, get_job_compute

            compute = get_job_compute()
            # Only FargateTaskCompute has ECS tags/tasks to cancel by (dev's
            # LocalSubprocessCompute has neither) — narrow like the reconciler
            # rather than widening the JobCompute Protocol.
            if not isinstance(compute, FargateTaskCompute):
                continue

            # The ARN of the task StopTask was issued against, or None. This path
            # deliberately does not wait for STOPPED: it is the kill switch /
            # shutdown sweep, which stops everything it can find as fast as it can
            # and is not reporting an outcome to a caller. The per-job cancel,
            # which does report one, confirms termination instead.
            task_arn = await compute.cancel_by_tag(owner_id, project_id, job_id)
            if task_arn:
                logger.info(
                    "[hard-cancel] stop-task-by-tag job=%s arn=%s",
                    job_id[:8], task_arn,
                )
                continue

            # Secondary fast path: no live task found by tag, but the durable
            # record may carry a handle (e.g. a task that just transitioned).
            store = get_platform().metadata_store
            record = await store.get_job(owner_id, project_id, job_id)
            compute_handle = record.get("compute_handle") if record else None
            if compute_handle and compute_handle.startswith("arn:aws:ecs:"):
                handle = JobComputeHandle(kind="fargate_task", value=compute_handle)
                await compute.cancel(handle)
                logger.info(
                    "[hard-cancel] stop-task-by-handle job=%s handle=%s",
                    job_id[:8], compute_handle,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort cancel
            logger.warning(
                "[hard-cancel] stop-task failed job=%s: %s", job_id[:8], exc,
            )

    # Await every PID kill in parallel. ``kill_harness_subprocess`` already
    # handles ProcessLookupError internally, so a child that exited on
    # its own between get_active_jobs and now is a no-op.
    if pids_to_kill:
        results = await asyncio.gather(
            *(kill_harness_subprocess(pid) for pid in pids_to_kill),
            return_exceptions=True,
        )
        for pid, result in zip(pids_to_kill, results):
            if isinstance(result, Exception):
                logger.warning(
                    "[hard-cancel] kill failed pid=%d: %s", pid, result,
                )

    return cancelled_count


async def run_eval_via_harness(
    job_id: str,
    project_dir: Path,
    model_ids: list[str],
    prompt_version: str,
    max_test_cases: int | None,
    judge_model_id: str | None,
    data_file_override: str | None = None,
    source: str = "manual",
    *,
    skip_sync: bool = False,
    defer_completion: bool = False,
) -> HarnessRun:
    """Run an evaluation via the project's standalone harness subprocess.

    Executes ``uv sync`` to ensure dependencies (unless ``skip_sync=True``),
    then invokes the harness ``run_eval.py`` as an async subprocess. Parses
    JSON progress lines from stdout to update the JobTracker in real time.
    On completion, loads results from the output file. Non-JSON stdout lines
    are silently skipped (harness may print regular output alongside JSON
    progress lines).

    Args:
        job_id: Tracker job ID to update with progress.
        project_dir: Path to the project directory containing run_eval.py.
        model_ids: Bedrock model IDs to evaluate.
        prompt_version: Prompt version name (e.g. "baseline", "v1").
        max_test_cases: Optional limit on test cases (0 or None means all).
        judge_model_id: Optional Bedrock model ID for LLM-as-judge metrics.
        data_file_override: Optional data file name to override harness config.
        source: Evaluation source ("manual", "optimization", "validation").
        skip_sync: When True, skip the ``uv sync`` step (deps baked into
            worker image). Default False preserves the dev path
            byte-for-byte. The harness still uses ``uv run`` to invoke
            ``run_eval.py`` inside the baked venv; ``skip_sync`` skips only
            the dependency-resolution step, not ``uv run``.
        defer_completion: When True, do NOT record completion — return the
            results and let the caller do it. Failures and cancellations are
            still recorded here either way.

            The detached worker needs this. ``complete_job`` is the terminal
            chokepoint: it schedules the durable ``completed`` write, publishes
            the per-job ``done`` envelope and the project event, and releases the
            project's eval lock — all immediately. The worker only uploads its
            results to S3 afterwards, and its EFS mount is read-only so ``/tmp``
            holds the only copy. Recording completion here therefore announced
            success before the results were durable, and a later ``fail_job``
            could not take it back: the fan-out is gated on the job not already
            being terminal, so the SPA kept "completed" while the durable record
            went to "failed". Deferring keeps it to one terminal transition, after
            the output is safe.
    """
    tracker = get_job_tracker()
    uv_path = shutil.which("uv")
    if not uv_path:
        tracker.fail_job(job_id, ResourceError(
            code=Codes.HARNESS_UV_SYNC_FAILED,
            message="uv not found on PATH.",
            hint="Install uv from https://astral.sh/uv to run harness projects.",
        ))
        return HarnessRun("failed")

    try:
        # Step 1: Run `uv sync` to ensure project dependencies are installed
        # (skip when skip_sync=True — deps baked into worker image)
        if skip_sync:
            logger.info("[harness] Skipping uv sync (deps baked into image)")
        else:
            tracker.update_progress(job_id, message="Installing dependencies (uv sync)...")
            logger.info("[harness] Running uv sync in %s", project_dir)

            try:
                sync_rc, sync_stdout, sync_stderr = await run_bounded_subprocess(
                    [uv_path, "sync"],
                    timeout_s=UV_SYNC_TIMEOUT_S,
                    cwd=str(project_dir),
                    env=_project_subprocess_env(),
                    label="uv sync",
                )
            except SubprocessTimeout as timeout:
                tracker.fail_job(job_id, ResourceError(
                    code=Codes.HARNESS_UV_SYNC_FAILED,
                    message=str(timeout) + ".",
                    hint=(
                        "Dependency resolution is wedged or the package index is "
                        "unreachable. The process group was killed."
                    ),
                ))
                return

            if sync_rc != 0:
                error_msg = sync_stderr.decode("utf-8", errors="replace").strip()
                tracker.fail_job(job_id, ResourceError(
                    code=Codes.HARNESS_UV_SYNC_FAILED,
                    message=f"uv sync failed (exit {sync_rc}).",
                    hint="Check the harness project's pyproject.toml and dependencies.",
                    details={
                        "exit_code": sync_rc,
                        "stderr": error_msg[:2000],
                    },
                ))
                logger.error("[harness] uv sync failed: %s", error_msg)
                return HarnessRun("failed")

            logger.info("[harness] uv sync complete")

        # Step 2: Build the harness command
        # Built via the shared builder so the real run and the smoke eval
        # can't diverge on flags. ``-u`` (unbuffered) is baked into the
        # builder's argv so JSON progress lines arrive immediately.
        cmd = build_harness_command(
            uv_path,
            model_ids=model_ids,
            prompt_version=prompt_version,
            json_progress=True,
            source=source,
            max_test_cases=max_test_cases,
            judge_model_id=judge_model_id,
            data_file_override=data_file_override,
        )

        tracker.update_progress(
            job_id,
            message=f"Starting evaluation ({len(model_ids)} models)",
            models_total=len(model_ids),
            models_completed=0,
        )
        logger.info("[harness] Executing: %s", " ".join(cmd))

        # Step 3: Run the harness subprocess and stream stdout.
        # start_new_session=True puts the child in its own process group so
        # cancel_eval can kill the whole group (uv + grandchild python) via
        # os.killpg — SIGTERM to the immediate child alone doesn't propagate
        # to `uv run`'s grandchild, which lets DeepEval run to completion
        # and produce orphan result files.
        run_kwargs: dict = {
            "cwd": str(project_dir),
            "env": _project_subprocess_env(),
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "start_new_session": True,
        }

        # Audited. ``cmd`` is an argv LIST and no shell is involved, so a
        # metacharacter in any element is not interpreted — it arrives as a
        # literal argument to run_eval.py's argparse. build_harness_command does
        # not sanitise its inputs and does not need to: each value is emitted as
        # its own argv element after the flag it belongs to, so the worst a
        # flag-shaped value can do is make argparse reject it.
        # nosemgrep: dangerous-asyncio-create-exec-audit
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            **run_kwargs,
        )

        # Attach PID before any further await so a cancel_eval firing
        # during spawn has a handle. Then re-check cancellation — if the
        # user cancelled while we were spawning, kill the subprocess now
        # rather than letting it run to completion (would otherwise keep
        # burning Bedrock credits until the read loop observes the kill).
        job = tracker.get_job(job_id)
        if job and proc.pid:
            job.pid = proc.pid
            # Persist the PID to disk so the startup recovery hook in
            # ``main.py`` can SIGKILL this child if the coordinator
            # restarts mid-eval. JobTracker is in-memory and the pid
            # handle is lost on restart; the on-disk record is the only
            # way recovery can match the surviving subprocess. Recorded
            # for EVERY eval — manual and optimization alike — so a
            # manual eval's subprocess is no longer an unkillable orphan
            # across a restart (previously the PID was stored inside
            # OptimizationState and only for source=="optimization").
            # Best-effort — failure here MUST NOT block eval startup.
            record_active_eval(project_dir, proc.pid, job_id)
        if job and job.status == "cancelled":
            logger.info(
                "[harness] job=%s cancelled during spawn — killing subprocess",
                job_id[:8],
            )
            if proc.pid:
                await kill_harness_subprocess(proc.pid)

        output_file = None
        # Head and tail only — see _STDERR_RETAINED_BYTES.
        stderr_head = bytearray()
        stderr_tail = bytearray()

        def _retain_stderr(chunk: bytes) -> None:
            """Keep the first and last _STDERR_RETAINED_BYTES of stderr."""
            if len(stderr_head) < _STDERR_RETAINED_BYTES:
                stderr_head.extend(chunk[: _STDERR_RETAINED_BYTES - len(stderr_head)])
            stderr_tail.extend(chunk)
            if len(stderr_tail) > _STDERR_RETAINED_BYTES:
                del stderr_tail[: len(stderr_tail) - _STDERR_RETAINED_BYTES]

        async def _drain_stderr():
            """Read stderr in background to prevent pipe buffer deadlock.

            DeepEval writes heavily to stderr (progress indicators, warnings,
            telemetry). If we don't drain it concurrently with stdout, the
            64KB pipe buffer fills and the subprocess blocks. Drained bytes
            are kept for the final error message AND forwarded line-by-line
            through the logger so operators can watch subprocess activity in
            the console in real time.

            Tagged :data:`_STDERR_TAG` so it is never mistaken for the app's own
            measurements — see that constant.

            Every forwarded line goes through :func:`subprocess_log_line`: this
            stream carries agent-authored harness output into a log group shared
            by all tenants, so it is capped per line and stripped of terminal
            control sequences (DeepEval's progress indicators emit ANSI escapes).
            """
            assert proc.stderr is not None
            job_tag = job_id[:8]
            line_buf = ""
            while True:
                chunk = await proc.stderr.read(8192)
                if not chunk:
                    break
                _retain_stderr(chunk)
                line_buf += chunk.decode("utf-8", errors="replace")
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    line = subprocess_log_line(line)
                    if line:
                        logger.info("[harness %s] %s %s", job_tag, _STDERR_TAG, line)
            # Flush any trailing non-newline-terminated text
            tail = subprocess_log_line(line_buf)
            if tail:
                logger.info("[harness %s] %s %s", job_tag, _STDERR_TAG, tail)

        # Drain stderr concurrently with stdout to avoid deadlock
        stderr_task = asyncio.create_task(_drain_stderr())

        try:
            # Read stdout line by line, parsing JSON progress events
            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Try to parse as JSON progress event
                try:
                    event = json.loads(line)
                    handle_harness_progress(tracker, job_id, event, len(model_ids))
                    # Append to the job's event buffer for real-time streaming
                    job = tracker.get_job(job_id)
                    if job:
                        await job.append_event(event)
                    # Track the results file path if reported
                    if event.get("event") == "results" and event.get("file"):
                        output_file = event["file"]
                except (json.JSONDecodeError, ValueError):
                    # Not JSON — regular stdout from the harness. Surface it so
                    # stray prints / tracebacks written to stdout aren't invisible.
                    logger.info(
                        "[harness %s] %s %s", job_id[:8], _STDOUT_TAG, line[:500]
                    )
        finally:
            # Ensure the stderr drain finishes and the subprocess is reaped
            # even if we bail out of the stdout loop via exception or cancel.
            # Without this, a kill_harness_subprocess() that races with
            # ProcessLookupError could leave a zombie that only GC collects.
            try:
                await stderr_task
            except Exception:
                logger.exception("[harness] stderr drain raised during cleanup")
            if proc.returncode is None:
                try:
                    await proc.wait()
                except Exception:
                    logger.exception("[harness] proc.wait() raised during cleanup")

        stderr_head_text = bytes(stderr_head).decode("utf-8", errors="replace").strip()
        stderr_tail_text = bytes(stderr_tail).decode("utf-8", errors="replace").strip()

        # If job was cancelled while running, discard any produced result
        # file and bail out. Prevents orphan results.json entries from
        # landing in list_runs / the orchestrator's metric history when
        # a cancel races a late completion.
        job = tracker.get_job(job_id)
        if job and job.status == "cancelled":
            logger.info("[harness] job=%s was cancelled — discarding any results", job_id[:8])
            if output_file:
                for candidate in (project_dir / output_file, project_dir / "output" / output_file):
                    try:
                        if candidate.exists():
                            candidate.unlink()
                            logger.info("[harness] job=%s deleted orphan output %s", job_id[:8], candidate)
                    except OSError as e:
                        logger.warning("[harness] job=%s failed to delete orphan %s: %s", job_id[:8], candidate, e)
            return HarnessRun("cancelled")

        if proc.returncode != 0:
            tracker.fail_job(job_id, ResourceError(
                code=Codes.HARNESS_SUBPROCESS_FAILED,
                message=f"Harness exited with code {proc.returncode}.",
                hint="See the per-job log file for the full traceback.",
                details={
                    "exit_code": proc.returncode,
                    "stderr_tail": stderr_tail_text[-2000:],
                },
            ))
            logger.error(
                "[harness] job=%s failed (exit %d): %s",
                job_id[:8], proc.returncode, stderr_head_text[:500],
            )
            return HarnessRun("failed")

        # Step 4: Load results from the output file
        results_data = _load_harness_results(project_dir, output_file)
        if results_data is not None:
            resolved_output = output_file or "results.json"
            if defer_completion:
                # Hand the payload back WITHOUT terminalizing. The caller records
                # completion once its own durability requirement is met — see the
                # ``defer_completion`` note in this function's docstring.
                logger.info(
                    "[harness] job=%s produced results, deferring completion to "
                    "the caller (output=%s)",
                    job_id[:8], output_file,
                )
                return HarnessRun("completed", results_data, resolved_output)
            tracker.complete_job(job_id, results_data, resolved_output)
            logger.info("[harness] job=%s complete, output=%s", job_id[:8], output_file)
            return HarnessRun("completed", results_data, resolved_output)

        # Exit 0 with no readable results is a FAILURE, not an empty success.
        # _load_harness_results also returns None for a truncated or malformed
        # results file, so recording completion here published an eval that had
        # produced nothing and could never be viewed.
        tracker.fail_job(job_id, ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message="Harness exited successfully but produced no readable results.",
            hint=(
                "The results file is missing, empty or malformed. Check the "
                "per-job log for what the harness wrote."
            ),
            details={
                "expected_output": output_file or "results.json",
                "stderr_tail": stderr_tail_text[-2000:],
            },
        ))
        logger.error(
            "[harness] job=%s exited 0 but no readable results file was found (expected %s)",
            job_id[:8], output_file or "results.json",
        )
        return HarnessRun("failed")

    except Exception as e:
        tracker.fail_job(job_id, ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message=f"Harness execution error: {e}",
            details={"exception": type(e).__name__},
        ))
        logger.error("[harness] job=%s FAILED: %s", job_id[:8], e, exc_info=True)
        return HarnessRun("failed")
    finally:
        # Symmetric counterpart to the record_active_eval call on spawn.
        # Clear the persisted record whichever way we exit (complete /
        # fail / cancel / unhandled exception) so the startup recovery
        # hook never sees a stale pid for a process that already
        # terminated. Covers all sources — manual and optimization.
        clear_active_eval(project_dir)


def handle_harness_progress(
    tracker, job_id: str, event: dict, models_total: int
) -> None:
    """Update the JobTracker from a harness JSON progress event.

    Translates the harness progress protocol events (generating, scoring_start,
    scoring_complete, complete) into tracker progress updates.

    Args:
        tracker: The JobTracker instance.
        job_id: Job ID to update.
        event: Parsed JSON event dict from the harness stdout.
        models_total: Total number of models being evaluated.
    """
    event_type = event.get("event", "")
    model = event.get("model", "")
    current = event.get("current", 0)
    total = event.get("total", 0)

    # Current model number = completed + 1 (the one we're working on now)
    job = tracker.get_job(job_id)
    completed_so_far = job.progress.get("models_completed", 0) if job else 0

    if event_type == "generating":
        tracker.update_progress(
            job_id,
            current_model=model,
            models_total=models_total,
            models_completed=completed_so_far,
            current_case=current,
            cases_total=total,
            message=f"Generating: {model} ({current}/{total})",
        )
    elif event_type == "scoring_start":
        tracker.update_progress(
            job_id,
            current_model=model,
            models_completed=completed_so_far,
            current_case=0,
            cases_total=total,
            message=f"Scoring: {model} ({total} cases)",
        )
    elif event_type == "scoring_progress":
        tracker.update_progress(
            job_id,
            current_model=model,
            models_total=models_total,
            models_completed=completed_so_far,
            current_case=current,
            cases_total=total,
            message=f"Scoring: {model} ({current}/{total})",
        )
    elif event_type == "scoring_complete":
        passed = event.get("passed", 0)
        tracker.update_progress(
            job_id,
            current_model=model,
            current_case=total,
            cases_total=total,
            models_completed=completed_so_far,
            message=f"Scored: {model} ({passed}/{total} passed)",
        )
    elif event_type == "complete":
        passed = event.get("passed", 0)
        # Increment models_completed counter
        job = tracker.get_job(job_id)
        models_done = (job.progress.get("models_completed", 0) + 1) if job else 1
        tracker.update_progress(
            job_id,
            current_model=model,
            current_case=total,
            cases_total=total,
            models_completed=models_done,
            message=f"Complete: {model} ({passed}/{total} passed)",
        )


def _load_harness_results(
    project_dir: Path, output_file: str | None
) -> list[dict] | None:
    """Load results JSON from the harness output directory.

    Tries the specific output file first (if reported via progress protocol),
    then falls back to the default ``output/results.json``. Returns the parsed
    JSON data or None if no results file is found.

    Args:
        project_dir: Path to the project directory.
        output_file: Filename reported by the harness, or None.

    Returns:
        Parsed results list, or None if not found.
    """
    candidates = []
    if output_file:
        # Validate paths stay within project dir (output_file comes from harness stdout)
        safe_project = safe_path(project_dir, output_file)
        if safe_project:
            candidates.append(safe_project)
        safe_output = safe_path(project_dir / "output", output_file)
        if safe_output:
            candidates.append(safe_output)
    # Default fallback
    candidates.append(project_dir / "output" / "results.json")

    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Normalize to list if needed
                if isinstance(data, dict):
                    return [data]
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[harness] Failed to load results from %s: %s", path, e)
                continue

    return None
