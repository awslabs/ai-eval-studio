# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization iteration executor.

Each optimization iteration is a single Job. The executor:
1. Dispatches an eval job and waits for completion.
2. Records the iteration results and checks convergence.
3. Returns the result — the calling loop
   (``optimization_loop_orchestrator.run_optimization_loop``) fires a short
   summary completion turn and then runs the prompt-optimizer itself.

The orchestrator loop (start → eval → record → optimize → repeat) is
split into per-iteration methods so each iteration is one Job with a
clear lifecycle. The completion-turn service handles the coordinator's
summary turn and coalescing of back-to-back results.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.services.jobs import Job, JobResult, _publish_project_event, get_job_tracker
from app.services.metric_error_analyzer import (
    build_generation_caveats,
    detect_generation_unreliable,
    detect_infra_failure,
)
from app.services.resource_error import Codes, ResourceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.optimization_state_repository import OptimizationStateHandle

logger = logging.getLogger(__name__)

# Hard ceiling for ``_wait_for_eval``. Defense in depth — there is no
# single subprocess timeout in the harness (DeepEval uses per-task gather
# budgets, BedrockRuntimeModel applies per-call read-timeout × retries),
# so a stuck job could otherwise pin a ``while True`` poll loop forever.
# 4 hours is generous enough that a real long-running eval never trips
# it; operators can dial it tight in test/CI via the env var.
_DEFAULT_WAIT_FOR_EVAL_MAX_S = 4 * 60 * 60  # 14400

# Terminal signal on the durable event log. The harness emits {"event":
# "results"} exactly once from save_results when the whole eval finishes.
# Mirrors job_progress_republisher._TERMINAL_HARNESS_EVENTS — on the detached
# path the in-memory JobTracker never observes the worker's completion, so the
# durable event log is the trustworthy terminal signal ("event log > job state").
_TERMINAL_HARNESS_EVENTS: frozenset[str] = frozenset({"results"})


async def _reap_detached_eval(
    owner_id: str | None, project_id: str, eval_job_id: str
) -> None:
    """Best-effort tag-first StopTask of a detached eval worker.

    The in-memory ``job_tracker.cancel_job`` is a no-op for a detached worker
    (the coordinator only holds a phantom queued entry). Route the actual reap
    through ``cancel_by_tag`` so the ECS worker is stopped by its Job tag
    regardless of whether the durable record carries a compute_handle. Never
    raises into the cancel/ceiling path.
    """
    if not owner_id:
        return
    try:
        from app.platform.aws.fargate_task_compute import FargateTaskCompute
        from app.platform.job_compute import get_job_compute

        compute = get_job_compute()
        # Only FargateTaskCompute has ECS tasks to reap (dev has none).
        if isinstance(compute, FargateTaskCompute):
            await compute.cancel_by_tag(owner_id, project_id, eval_job_id)
    except Exception as exc:  # noqa: BLE001 — best-effort reap
        logger.warning(
            "[opt-executor] detached eval reap failed eval=%s: %s",
            eval_job_id[:8], exc,
        )


def _wait_for_eval_max_s() -> int:
    """Resolve the hard ceiling, with env-var override.

    Returns the value of ``OPTIMIZATION_WAIT_FOR_EVAL_MAX_S`` when set
    to a positive integer; otherwise the 4-hour default. Invalid values
    fall back to the default rather than raising mid-loop.
    """
    raw = os.environ.get("OPTIMIZATION_WAIT_FOR_EVAL_MAX_S")
    if raw is None:
        return _DEFAULT_WAIT_FOR_EVAL_MAX_S
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else _DEFAULT_WAIT_FOR_EVAL_MAX_S
    except ValueError:
        logger.warning(
            "[opt-executor] Ignoring invalid OPTIMIZATION_WAIT_FOR_EVAL_MAX_S=%r",
            raw,
        )
        return _DEFAULT_WAIT_FOR_EVAL_MAX_S


async def _dispatch_iteration_reports(
    project_dir: Path,
    project_id: str,
    run_filename: str,
    owner_id: str | None,
) -> None:
    """Fire per-run + winner-gated exec narrations for one iteration.

    Async because the metric audit is AWAITED rather than fired — see the call
    below. The two narration dispatches stay fire-and-forget; they take an agent
    turn each and nothing downstream reads their output synchronously.

    Mirrors what the standalone-eval path already does in the eval
    executor — optimization iterations bypass that path (they go
    straight through ``run_eval_via_harness``), so without this hook
    every per-run report and the live exec narrative would stay
    barebones for the duration of an optimization run.

    Every dispatcher swallows its own errors, so this never fails the
    iteration. The exec dispatch is winner-gated, so it only narrates
    when an iteration actually advances the trophy — a few calls per
    optimization, not one per iteration. Coalescing inside
    ``generate_exec_report`` makes it safe alongside the
    convergence-time exec dispatch below.
    """
    if not run_filename:
        return
    # Lazy import — both modules live in the jobs package and there's
    # no real cycle (eval_executor doesn't import optimization_executor),
    # but lazy matches the house style and keeps the cold-start surface
    # of this module small.
    from app.services.jobs.eval_executor import (
        _dispatch_run_report,
        _dispatch_exec_if_winner_changed,
        _dispatch_metric_audit,
    )
    asyncio.create_task(_dispatch_run_report(
        project_dir, project_id, Path(run_filename).name, owner_id,
    ))
    asyncio.create_task(_dispatch_exec_if_winner_changed(
        project_dir, project_id, owner_id,
    ))
    # Same metric audit the standalone path runs. Optimization iterations are
    # the half that matters most: a mis-scoring metric here does not just
    # mislead a reader, it feeds every rewrite decision the loop makes.
    #
    # AWAITED, not fired. The post-eval coordinator message reads this verdict,
    # and racing it means the coordinator decides whether to converge or cull a
    # model with no idea a metric is unreliable. It takes a few milliseconds
    # even on a large run file, and it swallows its own exceptions, so awaiting
    # is free and cannot fail the iteration.
    # Guarded HERE as well as inside the dispatcher. While this was fired with
    # create_task, a raising audit could only log; awaiting it puts it on the
    # iteration's own path, so the no-propagation guarantee has to be local
    # rather than borrowed from a callee that might later be refactored.
    try:
        await _dispatch_metric_audit(
            project_dir, project_id, Path(run_filename).name,
        )
    except Exception as e:  # noqa: BLE001 — commentary must not fail an iteration
        logger.warning(
            "[opt-executor] metric audit failed for %s: %s", run_filename, e,
        )


def _record_terminal_error(
    handle: "OptimizationStateHandle",
    error: ResourceError,
    *,
    new_status: str = "cancelled",
) -> None:
    """Record a terminal error on optimization state and announce it.

    Takes a handle rather than (state, project_id, arch_split, user_id,
    expected_version): those five arguments were the write context, and every call
    site had to assemble them correctly and identically. The handle carries them, so
    the only decisions left here are the error and the terminal status.

    Args:
        handle: Handle from the repository, with state loaded.
        error: The terminal error to record.
        new_status: Terminal status to set (default: "cancelled").
    """
    from app.services import optimization_state_repository as opt_repo

    handle.state.error = error
    handle.state.status = new_status
    opt_repo.write(handle)
    _publish_project_event(handle.project_id, {
        "resource": "optimization",
        "type": "error",
        **error.to_dict(),
    })


async def run_optimization_iteration_job(job: Job) -> JobResult:
    """Execute a single optimization iteration.

    Expected config keys:
        project_id: Project UUID.
        project_dir: Absolute path to project directory.
        model_ids: List of Bedrock model IDs.
        prompt_version: Current prompt version to evaluate.
        judge_model_id: Optional judge model ID.
        max_test_cases: Optional case limit.
        data_file_override: Optional data file path override.
        iteration: Iteration number (1-based).
        arch_split: Flag resolved once at dispatch entry.
        user_id: Required on new path for put_optimization_state.

    Args:
        job: The Job instance (must be type "optimization_iteration").

    Returns:
        JobResult with iteration outcome data.
    """
    from app.services.jobs import get_job_tracker
    from app.services.optimization_orchestrator import dispatch_eval_job, OptimizationConfig

    tracker = get_job_tracker()
    config = job.config
    project_id = config["project_id"]
    project_dir = Path(config["project_dir"])
    prompt_version = config["prompt_version"]
    iteration = config["iteration"]
    arch_split = config.get("arch_split", False)
    user_id = config.get("user_id")

    try:
        tracker.update_progress(job.job_id, {
            "message": f"Iteration {iteration}: evaluating prompt {prompt_version}",
            "iteration": iteration,
            "prompt_version": prompt_version,
            "phase": "eval",
        })

        # Load optimization state. On the new path (arch_split=True), read
        # from DynamoDB via get_optimization_state to obtain the current
        # version for the conditional-write funnel.
        from app.services import optimization_state_repository as opt_repo

        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.state is None:
            error = ResourceError(
                code=Codes.OPTIMIZATION_PROJECT_NOT_FOUND,
                message="Optimization state not found.",
            )
            tracker.update_state(job.job_id, "failed", error=error)
            return JobResult(status="failed", error=error)
        opt_state = handle.state

        # Check for cancellation
        if opt_state.status not in ("running",):
            tracker.update_state(job.job_id, "cancelled")
            return JobResult(status="failed", data={"reason": "cancelled"})

        # Check if this prompt was already evaluated (resume path).
        # If so, skip both eval dispatch AND recording — the iteration
        # is already in the history. Jump straight to the result so the
        # loop can run the prompt-optimizer on it.
        prior = opt_state.iterations[-1] if opt_state.iterations else None
        if prior and prior.get("prompt_version") == prompt_version:
            logger.info(
                "[opt-executor] Prompt %s already evaluated in iteration %d — "
                "skipping eval and record, reusing prior results",
                prompt_version, prior.get("iteration", "?"),
            )
            iteration_data = _extract_iteration_data_from_prior(prior)
            # Skip recording — already in history. Check convergence from
            # what was already recorded. check_convergence returns
            # (convergence, new_version); this skip-eval branch does no write,
            # so the version is discarded.
            convergence, _ = opt_state.check_convergence()
            result_data = {
                "iteration": iteration,
                "prompt_version": prompt_version,
                "passed_cases": iteration_data.get("passed_cases", 0),
                "total_cases": iteration_data.get("total_cases", 0),
                "metrics_summary": iteration_data.get("metrics_summary", {}),
                "per_model_pass_rates": iteration_data.get("per_model_pass_rates", {}),
                "run_output_file": iteration_data.get("run_output_file", ""),
                "convergence": convergence,
                "status": opt_state.status,
                # These results were already announced when the ORIGINAL
                # iteration recorded them; the orchestrator uses this flag to
                # suppress the visible completion turn so a restart/adopt or
                # resume doesn't re-report the same run to the user as a new
                # iteration (same run file, same numbers).
                "reused_prior_results": True,
            }
            tracker.update_state(job.job_id, "completed", result=result_data)
            _publish_project_event(project_id, {
                "resource": "job",
                "type": "completed",
                "job_id": job.job_id,
                "job_type": "optimization_iteration",
                "iteration": iteration,
                "convergence": convergence,
            })
            await _dispatch_iteration_reports(
                project_dir, project_id,
                result_data.get("run_output_file") or "",
                config.get("owner_id") or config.get("user_id"),
            )
            return JobResult(status="completed", data=result_data)
        else:
            # Dispatch eval and wait
            opt_config = OptimizationConfig(
                project_id=project_id,
                owner_id=config.get("owner_id") or config.get("user_id", ""),
                model_ids=config["model_ids"],
                data_file=config.get("data_file", ""),
                field_mappings=config.get("field_mappings", {}),
                prompt_version=prompt_version,
                judge_model_id=config.get("judge_model_id"),
                target_metric_scores=config.get("target_metric_scores", {}),
                max_iterations=config.get("max_iterations", 10),
                plateau_window=config.get("plateau_window", 3),
                max_test_cases=config.get("max_test_cases"),
                data_file_override=config.get("data_file_override"),
            )

            eval_job_id = await dispatch_eval_job(
                project_dir, opt_config, prompt_version,
                config.get("data_file_override"),
                arch_split=arch_split,  # Thread resolved flag from loop entry
            )

            # Wait for eval completion
            iteration_data = await _wait_for_eval(
                eval_job_id, project_dir, project_id, job, tracker
            )

            if iteration_data is None:
                # Eval abandoned (died, ceiling, cancelled) — persist terminal
                # optimization status with error. The wait surfaces the abandonment
                # (returns None); the caller owns the terminal write (layer boundary).
                error = ResourceError(
                    code=Codes.EVAL_DIED_WITHOUT_RESULTS,
                    message=f"Eval {eval_job_id} reached terminal state without producing results — failing iteration.",
                    details={"eval_job_id": eval_job_id},
                )
                # Re-read: the eval took time, so the version observed at
                # iteration entry is stale for a conditional write.
                handle = await opt_repo.read(
                    project_id, project_dir,
                    arch_split=arch_split, owner_id=user_id,
                )
                if handle.state is None:
                    return JobResult(status="failed", error=error)

                if handle.state.status not in ("cancelled", "stopping"):
                    # Don't overwrite user-cancel with eval-failure
                    _record_terminal_error(handle, error, new_status="cancelled")
                return JobResult(status="failed", error=error)

        # Record iteration and check convergence
        tracker.update_progress(job.job_id, {
            "message": f"Iteration {iteration}: recording results",
            "phase": "recording",
        })

        # Re-read: state may have changed during the eval (a user cancel lands
        # here), and the write below needs the current version.
        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.state is None or handle.state.status not in ("running",):
            tracker.update_state(job.job_id, "cancelled")
            return JobResult(status="failed", data={"reason": "cancelled"})
        opt_state = handle.state

        # Check metric error dominance / judge unreliability / generation unreliability.
        # Each resumable domain (judge, generation) gets its own status because the
        # remedies differ: judge-unreliable needs quota or judge-model change,
        # generation-unreliable needs transient-retry or generation-concurrency reduction.
        # The >=80% dominance cases (metric/generation errors dominant) stay as
        # ``cancelled`` (the default — caller treats them as hard fails).
        if iteration_data.get("metric_error"):
            error = iteration_data["metric_error"]
            # Map error code to terminal status
            code_to_status = {
                Codes.EVAL_JUDGE_UNRELIABLE: "judge_unreliable",
                Codes.EVAL_GENERATION_UNRELIABLE: "generation_unreliable",
            }
            terminal_status = code_to_status.get(error.code, "cancelled")

            _record_terminal_error(handle, error, new_status=terminal_status)
            tracker.update_state(job.job_id, "failed", error=error)
            return JobResult(status="failed", error=error)

        # Record the iteration, then check convergence. Both WRITE, so both
        # participate in the version chain — each returns a handle carrying the
        # version its write produced, and the next call consumes it by reassignment.
        # Hand-threading these two versions through separate locals is what used to
        # lose conditional writes when a branch forgot one.
        handle = opt_repo.record_iteration(
            handle,
            prompt_version=prompt_version,
            run_id=iteration_data.get("run_output_file", ""),
            metrics_summary=iteration_data.get("metrics_summary", {}),
            total_cases=iteration_data.get("total_cases", 0),
            passed=iteration_data.get("passed_cases", 0),
            per_model_pass_rates=iteration_data.get("per_model_pass_rates", {}),
            per_model_metrics=iteration_data.get("per_model_metrics", {}),
            caveats=iteration_data.get("caveats"),
        )
        convergence, handle = opt_repo.check_convergence(handle)

        # Build result data for the completion turn
        result_data = {
            "iteration": iteration,
            "prompt_version": prompt_version,
            "passed_cases": iteration_data.get("passed_cases", 0),
            "total_cases": iteration_data.get("total_cases", 0),
            "metrics_summary": iteration_data.get("metrics_summary", {}),
            "per_model_pass_rates": iteration_data.get("per_model_pass_rates", {}),
            "run_output_file": iteration_data.get("run_output_file", ""),
            "convergence": convergence,
            "status": opt_state.status,
        }

        # Include caveats in result_data so the completion-turn synthetic
        # message can surface them in chat.
        if "caveats" in iteration_data:
            result_data["caveats"] = iteration_data["caveats"]

        if convergence:
            logger.info(
                "[opt-executor] Converged: reason=%s iteration=%d",
                convergence, iteration,
            )
            result_data["converged"] = True
            result_data["convergence_reason"] = convergence
            tracker.update_state(job.job_id, "completed", result=result_data)
        else:
            tracker.update_state(job.job_id, "completed", result=result_data)

        _publish_project_event(project_id, {
            "resource": "job",
            "type": "completed",
            "job_id": job.job_id,
            "job_type": "optimization_iteration",
            "iteration": iteration,
            "convergence": convergence,
        })

        # Per-iteration narrations: per-run sidecar always, exec on
        # winner-change only (the dispatch is winner-gated). Optimization
        # iterations call ``run_eval_via_harness`` directly via
        # ``dispatch_eval_job``, bypassing the standalone-eval path's
        # report dispatchers — without this hook, every per-run report
        # and the live exec narrative would stay barebones until
        # convergence.
        await _dispatch_iteration_reports(
            project_dir, project_id,
            result_data.get("run_output_file") or "",
            config.get("owner_id") or config.get("user_id"),
        )

        # On convergence, generate the executive report unconditionally
        # (background, never blocks the iteration result). The
        # per-iteration winner-gated dispatch above usually beats us to
        # it, but if the final iteration didn't change the winner, this
        # is what guarantees a fresh exec narrative reflecting the full
        # converged journey.
        if convergence:
            owner_id = config.get("owner_id") or config.get("user_id")
            asyncio.create_task(_dispatch_exec_report(
                project_dir, project_id, owner_id,
            ))

        return JobResult(status="completed", data=result_data)

    except asyncio.CancelledError:
        tracker.cancel_job(job.job_id)
        raise
    except Exception as e:
        logger.error(
            "[opt-executor] job=%s iteration=%s error: %s",
            job.job_id[:8], config.get("iteration"), e, exc_info=True,
        )
        error = ResourceError(
            code=Codes.SYSTEM_TASK_CRASHED,
            message=f"Optimization iteration failed: {e}",
            details={"exception": type(e).__name__, "iteration": config.get("iteration")},
        )
        tracker.update_state(job.job_id, "failed", error=error)

        # Also record on optimization state. A fresh read: this is the crash
        # handler, so any version held above may be stale or unset.
        from app.services import optimization_state_repository as opt_repo

        crash_handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        # Only record if nothing else already recorded a terminal error — the
        # FIRST cause is the useful one; overwriting it with a downstream symptom
        # loses the diagnosis.
        if crash_handle.state is not None and crash_handle.state.error is None:
            _record_terminal_error(crash_handle, error)

        return JobResult(status="failed", error=error)


def _extract_iteration_data(results: list[dict], run_output_file: str) -> dict[str, Any]:
    """Extract iteration data from eval results using the shared helper.

    Args:
        results: List of per-model run dicts from the eval.
        run_output_file: Filename of the saved results.

    Returns:
        Dict with passed_cases, total_cases, metrics_summary, etc.
    """
    from app.services.jobs.iteration_metrics import extract_iteration_metrics
    data = extract_iteration_metrics(results)
    data["run_output_file"] = run_output_file
    return data


async def _wait_for_eval(
    eval_job_id: str,
    project_dir: Path,
    project_id: str,
    parent_job: Job,
    job_tracker,
) -> dict[str, Any] | None:
    """Poll for eval job completion, checking for cancellation.

    On the detached path (arch_split=True) the eval runs on a Fargate worker
    the coordinator's in-memory ``JobTracker`` NEVER observes flipping to
    terminal. So terminal is derived from the DURABLE EVENT LOG (an event whose
    ``data.event == "results"``, the once-per-eval save_results signal),
    mirroring the progress republisher. The real run filename is taken from that
    terminal event's ``data.file``, and the eval RESULT is read from the
    MATERIALIZED FILE (``materialize_detached_run`` downloads the worker's output
    from S3 to the coordinator's EFS, then ``load_materialized_results`` reads
    it) — NOT the durable job record's ``result`` field: a full result payload
    exceeds DynamoDB's 400 KB item limit. Once terminal + result are in hand the
    coordinator writes the eval record TERMINAL authoritatively — the worker
    registers with ``create_job(durable_write=False)``, so it never overwrites
    the coordinator's full record — and the reconciler, which sees the record's
    compute_handle, cannot false-fail a completed eval after the worker exits.

    On arch_split=False the path is byte-behaviorally unchanged: terminal comes
    from the in-memory ``job_tracker`` status and the result from
    ``eval_job.result`` / ``eval_job.output_file``.

    Args:
        eval_job_id: ID of the dispatched eval job.
        project_dir: Path to project directory.
        project_id: Project UUID.
        parent_job: The parent optimization iteration job.
        job_tracker: The JobTracker instance.

    Returns:
        Dict with iteration data, or None if failed/cancelled.
    """
    # This poll is the in-eval cancel detector. On arch_split=True a user cancel
    # writes DDB (REST cancel endpoint + the loop's DDB cancel-detection), so an
    # EFS read here would be blind (split-brain) and the in-flight iteration eval
    # would not be reaped. Resolve the normalized owner key (owner_id or user_id,
    # consistent) — optimization-path evals carry owner_id, not
    # user_id.
    arch_split = parent_job.config.get("arch_split", False)
    user_id = parent_job.config.get("user_id")
    owner_key = parent_job.config.get("owner_id") or user_id

    poll_count = 0
    poll_interval_s = 10
    max_wait_s = _wait_for_eval_max_s()

    # Durable-path result + output_file, populated when the arch_split=True
    # branch detects the terminal `results` event and consistent-reads the
    # record. On arch_split=False these stay None and the in-memory eval_job
    # supplies the values below.
    durable_result: list[dict] | None = None
    durable_output_file: str | None = None
    since_seq = 0  # event-log cursor for the detached-path terminal detection

    while True:
        await asyncio.sleep(poll_interval_s)
        poll_count += 1

        # Check for external cancellation
        # On arch_split=True, read the cancel signal from DDB; else keep EFS load.
        from app.services import optimization_state_repository as opt_repo

        poll_handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        disk_state = poll_handle.state

        if disk_state and disk_state.status in ("stopping", "cancelled"):
            # Reap the eval. On the detached path the in-memory cancel_job is a
            # no-op — route the actual StopTask through cancel_by_tag.
            if arch_split:
                await _reap_detached_eval(owner_key, project_id, eval_job_id)
            job_tracker.cancel_job(eval_job_id)
            try:
                job_tracker.update_state(eval_job_id, "cancelled")
            except KeyError:
                pass
            if disk_state.status == "stopping":
                disk_state.status = "cancelled"
                # The handle carries the version read on THIS poll tick, so the
                # conditional write is guarded by a version we actually observed.
                opt_repo.write(poll_handle)
            job_tracker.update_state(parent_job.job_id, "cancelled")
            return None

        if arch_split:
            # Detached path: derive terminal from TWO sources —
            # the DURABLE EVENT LOG (success signal) AND the DURABLE JOB RECORD
            # (reconciler-reaped / failed signal). The in-memory eval job never
            # flips for a detached worker, so branching on job_tracker status
            # here would spin to the ceiling forever. Both sources are needed:
            # the event log only carries the harness's own success signal, so a
            # worker that dies before emitting it (credential expiry, task kill,
            # reconciler reap) would otherwise be polled to the ceiling.
            from app.platform import get_platform
            store = get_platform().metadata_store
            try:
                events = await store.events_since(eval_job_id, since_seq)
            except Exception as exc:  # noqa: BLE001 — never die on a read error
                logger.warning(
                    "[opt-executor] events_since failed eval=%s: %s",
                    eval_job_id[:8], exc,
                )
                events = []

            # Check durable eval record for terminal state (second source).
            # Deliberately does NOT log here: the record goes terminal on the
            # SUCCESS path too, so logging at read time emitted a "died without
            # results" warning for healthy evals. The log belongs on the branch
            # that actually acts on it (below, once `terminal_seen` is known).
            durable_eval_terminal = False
            durable_eval_state: str | None = None
            try:
                durable_eval_record = await store.get_job(owner_key, project_id, eval_job_id)
                if durable_eval_record and durable_eval_record.get("state") in ("completed", "failed", "cancelled"):
                    durable_eval_terminal = True
                    durable_eval_state = durable_eval_record.get("state")
            except Exception as exc:  # noqa: BLE001 — never die on a read error
                logger.warning(
                    "[opt-executor] get_job failed eval=%s: %s",
                    eval_job_id[:8], exc,
                )

            terminal_seen = False
            for event in events:
                seq = event.get("seq")
                if isinstance(seq, int):
                    since_seq = max(since_seq, seq)
                data = event.get("data") or {}
                if data.get("event") in _TERMINAL_HARNESS_EVENTS:
                    terminal_seen = True
                    # The terminal `results` event always carries the real
                    # filename in data.file. Capture it here — no job-UUID
                    # fallback is needed, and it sidesteps the consistent-read
                    # race entirely (the record's output_file may still be
                    # lagging when we read it).
                    durable_output_file = data.get("file")

            # Durable-record terminal is a second release condition, but ONLY
            # when the success signal is genuinely absent. A successful eval
            # sets both signals seconds apart (worker emits `results`, then the
            # record is mirrored `state="completed"`), so any 10s tick landing
            # after both observes BOTH in one window. Checking the record first
            # unconditionally force-failed those successful evals — StopTask on
            # a healthy worker plus a failed iteration. `terminal_seen` wins.
            if durable_eval_terminal and not terminal_seen:
                # Eval died without emitting results — reap and fail the iteration
                logger.warning(
                    "[opt-executor] Eval %s terminal via durable record (state=%s) "
                    "with no results event — failing the iteration",
                    eval_job_id[:8], durable_eval_state,
                )
                await _reap_detached_eval(owner_key, project_id, eval_job_id)
                job_tracker.cancel_job(eval_job_id)
                try:
                    job_tracker.update_state(eval_job_id, "failed")
                except KeyError:
                    pass
                # Return None to signal iteration failure. The CALLER owns the
                # terminal write: it holds the read version the conditional
                # write needs, which this layer does not.
                return None

            if terminal_seen:
                # FILE IS SOURCE OF TRUTH: materialize the worker-uploaded
                # output/ file from S3 to EFS, then read the per-model run list
                # from the FILE via the normalizing companion — NOT the DDB
                # `result` (a full result payload exceeds DynamoDB's 400 KB
                # item limit).
                # On ceiling-out (upload never landed), fail the iteration with a
                # clear code + tag-first worker reap — never hang, never fall
                # through to metric detection on empty results.
                from app.services.run_output_snapshot import (
                    RunOutputNotAvailable,
                    load_materialized_results,
                    materialize_detached_run,
                )
                try:
                    efs_path = await materialize_detached_run(
                        project_id, eval_job_id, durable_output_file
                    )
                    durable_result = load_materialized_results(efs_path)
                except RunOutputNotAvailable as exc:
                    logger.error(
                        "[opt-executor] run output unavailable eval=%s: %s",
                        eval_job_id[:8], exc,
                    )
                    error = ResourceError(
                        code=Codes.EVAL_RESULTS_UNAVAILABLE,
                        message=(
                            f"Eval {eval_job_id} completed but its results file "
                            f"never materialized from S3 — failing the iteration."
                        ),
                        details={"eval_job_id": eval_job_id},
                    )
                    await _reap_detached_eval(owner_key, project_id, eval_job_id)
                    try:
                        await store.update_job_state(
                            owner_key, project_id, eval_job_id, state="failed"
                        )
                    except Exception as exc2:  # noqa: BLE001
                        logger.warning(
                            "[opt-executor] failed-record write failed eval=%s: %s",
                            eval_job_id[:8], exc2,
                        )
                    job_tracker.cancel_job(eval_job_id)
                    try:
                        job_tracker.update_state(eval_job_id, "failed", error=error)
                    except KeyError:
                        pass
                    job_tracker.update_state(parent_job.job_id, "failed", error=error)
                    return None

                # Coordinator authoritative terminal write (the worker registers
                # with durable_write=False, so it never re-writes the full
                # record): advance the record past queued so the reconciler
                # cannot false-fail a completed eval once the worker exits past
                # grace. Pass output_file so the durable record carries the real
                # filename (helps the reconciler).
                # Log-but-don't-swallow at WARNING — a failure here reintroduces
                # the false-fail risk.
                try:
                    await store.update_job_state(
                        owner_key, project_id, eval_job_id,
                        state="completed", output_file=durable_output_file,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[opt-executor] coordinator terminal write failed eval=%s: %s",
                        eval_job_id[:8], exc,
                    )

                # Claim the eval's completion turn on its behalf: the optimization
                # loop owns the iteration completion turn (a SEPARATE turn keyed on
                # the parent job_id), so the child eval's own completion is already
                # "handled" here. Marking it claimed suppresses ALL eval-keyed
                # firers — the reconciler's completion-recovery and any reattached
                # poller — from firing a spurious second turn for this eval ~60s
                # later (the poller on this path is progress-only and never claims,
                # so without this the eval record would look orphaned-completed).
                # This does NOT touch the loop's iteration turn (different job_id).
                # Best-effort: a failed claim only risks a redundant recovery turn,
                # not correctness — log at WARNING, don't fail the iteration.
                try:
                    await store.claim_completion_fired(
                        owner_key, project_id, eval_job_id
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[opt-executor] completion claim failed eval=%s: %s",
                        eval_job_id[:8], exc,
                    )
                break
        else:
            eval_job = job_tracker.get_job(eval_job_id)
            if eval_job is None:
                error = ResourceError(
                    code=Codes.OPTIMIZATION_DISPATCH_FAILED,
                    message="Eval job vanished from the tracker mid-run.",
                    details={"eval_job_id": eval_job_id},
                )
                job_tracker.update_state(parent_job.job_id, "failed", error=error)
                return None

            if eval_job.status in ("completed", "failed", "cancelled"):
                break

        # Hard ceiling — defense in depth so a wedged eval can't pin the
        # loop forever (applies to BOTH paths: a dead detached worker that
        # never emits `results` must not pin the loop for 4h). Cancel the
        # eval, mark the parent failed, and bail with a clear error code.
        elapsed_s = poll_count * poll_interval_s
        if elapsed_s >= max_wait_s:
            error = ResourceError(
                code=Codes.OPTIMIZATION_EVAL_WAIT_EXCEEDED,
                message=(
                    f"Eval job did not reach a terminal state within "
                    f"{max_wait_s}s — failing the iteration."
                ),
                details={
                    "eval_job_id": eval_job_id,
                    "elapsed_s": elapsed_s,
                    "max_wait_s": max_wait_s,
                },
            )
            logger.error(
                "[opt-executor] wait_for_eval ceiling reached eval=%s elapsed=%ds",
                eval_job_id[:8], elapsed_s,
            )
            # Reap the detached worker tag-first; the in-memory cancel is a
            # no-op for it. Then mark the eval record terminal `failed` so it
            # doesn't linger non-terminal against the caps.
            if arch_split:
                await _reap_detached_eval(owner_key, project_id, eval_job_id)
                try:
                    from app.platform import get_platform
                    store = get_platform().metadata_store
                    await store.update_job_state(
                        owner_key, project_id, eval_job_id, state="failed"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[opt-executor] ceiling terminal write failed eval=%s: %s",
                        eval_job_id[:8], exc,
                    )
            job_tracker.cancel_job(eval_job_id)
            try:
                job_tracker.update_state(eval_job_id, "failed", error=error)
            except KeyError:
                pass
            job_tracker.update_state(parent_job.job_id, "failed", error=error)
            return None

        if poll_count % 6 == 0:
            logger.info(
                "[opt-executor] Waiting for eval job=%s polls=%d elapsed=%ds",
                eval_job_id[:8], poll_count, elapsed_s,
            )

    # Unify the terminal result. On arch_split=True use the durable result +
    # output_file read above; on arch_split=False use the in-memory eval_job
    # (harness_runner wrote it via complete_job / fail_job). Do NOT reshape the
    # list — every reader (incl. the eval-completion-turn synthetic message)
    # expects the list form.
    if arch_split:
        # durable_result now comes from the materialized FILE, and
        # durable_output_file is the real filename from the terminal event's
        # data.file — no UUID fallback (the terminal event always carries it).
        results = durable_result or []
        output_file = durable_output_file
    else:
        if eval_job.status == "failed":
            error = eval_job.error or ResourceError(
                code=Codes.OPTIMIZATION_DISPATCH_FAILED,
                message="Eval job failed.",
            )
            job_tracker.update_state(parent_job.job_id, "failed", error=error)
            return None

        if eval_job.status == "cancelled":
            job_tracker.update_state(parent_job.job_id, "cancelled")
            return None

        results = eval_job.result or []
        output_file = eval_job.output_file or eval_job_id

    metric_error = detect_infra_failure(results)
    if metric_error:
        return {"metric_error": metric_error}

    # Generation reliability check. Sits between the >=80% dominance tier
    # (detect_infra_failure) and the judge tier (_detect_judge_unreliable)
    # because the dominance check must keep winning (hard fail), and the
    # judge tier must only see runs whose generation domain is acceptable.
    # Checks the above-tolerance band (>2% by default, <80%); sub-tolerance
    # failures are recorded as caveats below.
    gen_error = detect_generation_unreliable(results)
    if gen_error:
        return {"metric_error": gen_error}

    # Judge unreliability check. Distinct from the >=80% dominance guard
    # above: this catches the 5-80% degraded band where iteration-to-
    # iteration noise from null scores starts polluting the optimization
    # signal but isn't a hard infra failure. Halts the loop so the user
    # can investigate before more iterations chase phantom regressions.
    judge_error = _detect_judge_unreliable(results)
    if judge_error:
        return {"metric_error": judge_error}

    iteration_data = _extract_iteration_data(results, output_file)

    # Attach sub-tolerance generation caveats so tolerated failures survive
    # into history and explain why a pass rate moved.
    caveats = build_generation_caveats(results)
    if caveats:
        iteration_data["caveats"] = caveats

    return iteration_data


def _extract_iteration_data_from_prior(prior: dict) -> dict[str, Any]:
    """Extract iteration data from a previously recorded iteration.

    Args:
        prior: Dict from OptimizationState.iterations[-1].

    Returns:
        Dict matching the shape expected by the iteration recorder.
    """
    return {
        "passed_cases": prior.get("passed", 0),
        "total_cases": prior.get("total_cases", 0),
        "metrics_summary": prior.get("metrics_summary", {}),
        "per_model_pass_rates": prior.get("per_model_pass_rates", {}),
        "per_model_metrics": prior.get("per_model_metrics", {}),
        "run_output_file": prior.get("run_id", ""),
    }


def _detect_judge_unreliable(
    results: list[dict[str, Any]],
) -> ResourceError | None:
    """Return a ``judge_unreliable`` :class:`ResourceError` when any model
    run reports ``data_quality.reliability == "degraded"``, else None.

    ``reliability`` is JUDGE-ONLY. The runner
    template flips it to ``"degraded"`` when EITHER:

    * >5% of metric evaluations stayed null after the sequential retry
      phase (judge throttling / flakiness), or
    * any case was silently dropped by ``evaluate()`` and couldn't be
      recovered by the reconciliation pass — surfaced as
      ``unrecovered_count > 0``. Even one unrecovered case is enough
      to halt because the optimizer would chase noise on the
      remaining cases.

    Generation failures (target-model calls) are not in scope here —
    they route through ``detect_generation_unreliable`` instead. This
    detector covers only the judge domain (LLM-as-a-judge scoring).

    Distinct from :func:`detect_metric_error_dominance` which catches
    the >=80% hard-fail case (likely token expiry / model access). This
    fires in the 5-80% degraded band plus the any-unrecovered case.
    """
    degraded_models: list[dict[str, Any]] = []
    for model_run in results or []:
        dq = model_run.get("data_quality") or {}
        if dq.get("reliability") == "degraded":
            degraded_models.append({
                "model_id": model_run.get("model_id", "?"),
                "null_count": dq.get("null_count", 0),
                "total": dq.get("total", 0),
                "null_ratio": dq.get("null_ratio", 0.0),
                "dropped_count": dq.get("dropped_count", 0),
                "unrecovered_count": dq.get("unrecovered_count", 0),
                "total_cases": dq.get("total_cases", 0),
            })

    if not degraded_models:
        return None

    # Extract upstream status from judge errors (metrics_data[*].error)
    from app.services.metric_error_analyzer import _extract_dominant_upstream_status, _iter_metric_records

    metric_records = _iter_metric_records(results)
    error_texts = [str(m["error"])[:400] for m in metric_records if m.get("error") is not None]
    status_info = _extract_dominant_upstream_status(error_texts)

    # Build a human-readable summary the SPA banner can show as-is.
    # When ``unrecovered_count`` is non-zero, lead with that — silent
    # drops are scarier than null scores and the user should see them
    # first.
    parts: list[str] = []
    for m in degraded_models:
        if m["unrecovered_count"] > 0:
            parts.append(
                f"{m['model_id']}: {m['unrecovered_count']}/{m['total_cases']} "
                f"cases silently dropped"
            )
        else:
            parts.append(
                f"{m['model_id']}: {m['null_count']}/{m['total']} "
                f"null ({m['null_ratio'] * 100:.1f}%)"
            )
    summary = "; ".join(parts)

    # Append status info to message when available
    message = f"Judge unreliable on {len(degraded_models)} model run(s) — {summary}"
    if status_info:
        message += f" — dominant upstream status {status_info['status']} {status_info['label']}"
    message += "."

    # Keep "throttled" in the hint for 503 (pinned by
    # test_judge_unreliable_detector.py::TestDetectJudgeUnreliable::test_flags_single_degraded_model)
    if status_info and status_info["status"] in (429, 503):
        hint = (
            "Bedrock throttled the judge during scoring even after "
            "sequential retry. Raise the judge model's per-region "
            "throughput quota or pick a less-throttled judge before "
            "resuming."
        )
    elif status_info:
        # Non-throttling status
        hint = (
            f"Judge scoring failed with upstream status {status_info['status']} "
            f"{status_info['label']}. Check judge model configuration and retry."
        )
    else:
        # No parseable status
        hint = (
            "Bedrock throttled the judge during scoring even after "
            "sequential retry. Raise the judge model's per-region "
            "throughput quota or pick a less-throttled judge before "
            "resuming."
        )

    return ResourceError(
        code=Codes.EVAL_JUDGE_UNRELIABLE,
        message=message,
        hint=hint,
        details={"degraded_models": degraded_models},
    )


async def _dispatch_exec_report(
    project_dir: Path, project_id: str, owner_id: str | None,
) -> None:
    """Fire-and-forget executive report generation. Logs and swallows errors."""
    try:
        from app.services.report_generation import generate_exec_report
        await generate_exec_report(project_dir, project_id, owner_id)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s exec generation failed: %s",
            project_id[:8], e,
        )
