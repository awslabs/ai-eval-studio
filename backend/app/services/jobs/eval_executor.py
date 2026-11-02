# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval job executor — runs evaluation via the project's harness subprocess.

Wraps ``run_eval_via_harness`` in the JobExecutor protocol so the
JobTracker can dispatch eval runs uniformly. Progress and completion
flow through the JobTracker; the completion-turn service fires when the
job reaches a terminal state.

The harness subprocess needs a real ``cwd`` (``uv sync`` against
pyproject.toml, ``run_eval.py``'s data-file lookups). The project
directory is always on disk — local in dev, the EFS mount in cloud —
and is resolved at dispatch time into ``config["project_dir"]``.
"""

import asyncio
import logging
from pathlib import Path

from app.services.jobs import Job, JobResult, get_job_tracker, _publish_project_event
from app.services.metric_error_analyzer import detect_infra_failure
from app.services.resource_error import Codes, ResourceError

logger = logging.getLogger(__name__)


async def run_eval_job(job: Job) -> JobResult:
    """Execute an eval job via the harness subprocess.

    Dispatches ``run_eval_via_harness`` and monitors progress via the
    existing harness-runner callbacks (which update the job's progress
    fields). On completion, publishes to the project event bus.

    Expected config keys:
        project_dir: Absolute path to the project directory.
        model_ids: List of Bedrock model IDs to evaluate.
        prompt_version: Prompt file stem (e.g. "baseline", "v1").
        metrics: List of metric configuration dicts.
        max_test_cases: Optional int limit.
        judge_model_id: Optional Bedrock model ID for LLM-as-judge.
        data_file_override: Optional absolute path override.

    Args:
        job: The Job instance (must be type "eval").

    Returns:
        JobResult with status and eval results data.
    """
    from app.services.harness_runner import run_eval_via_harness

    tracker = get_job_tracker()
    config = job.config

    try:
        tracker.update_progress(
            job.job_id,
            message=f"Starting eval: {len(config['model_ids'])} model(s)",
            models_total=len(config["model_ids"]),
            models_completed=0,
        )

        # The project directory is resolved at dispatch time and always
        # exists on disk — local in dev, the EFS mount in cloud.
        project_dir = Path(config["project_dir"])

        # ``run_eval_via_harness`` writes progress + completion + failure
        # directly to the JobTracker. On return, ``job`` reflects the
        # terminal state — we only need to layer on the
        # metric-error-dominance check and publish the unified
        # ``resource=job`` event.
        await run_eval_via_harness(
            job_id=job.job_id,
            project_dir=project_dir,
            model_ids=config["model_ids"],
            prompt_version=config["prompt_version"],
            max_test_cases=config.get("max_test_cases"),
            judge_model_id=config.get("judge_model_id"),
            data_file_override=config.get("data_file_override"),
            source=config.get("source", "manual"),
        )

        # Re-read the terminal state harness_runner committed.
        finished = tracker.get_job(job.job_id)

        if finished and finished.state == "completed":
            result_data = {
                "model_runs": finished.result or [],
                "output_file": finished.output_file,
            }

            # Infra failures (generation or metric errors) are detected even
            # when the harness exited cleanly. Promote them to a failed
            # terminal state so the dashboards don't claim a successful run.
            metric_error = detect_infra_failure(finished.result or [])
            if metric_error:
                tracker.update_state(
                    job.job_id, "failed", error=metric_error
                )
                _publish_project_event(job.project_id, {
                    "resource": "job",
                    "type": "failed",
                    "job_id": job.job_id,
                    "job_type": "eval",
                    **metric_error.to_dict(),
                })
                return JobResult(status="failed", error=metric_error)

            _publish_project_event(job.project_id, {
                "resource": "job",
                "type": "completed",
                "job_id": job.job_id,
                "job_type": "eval",
                "output_file": finished.output_file,
            })

            # Fire the report-narrator in the background (chat-decoupled).
            # The narrative sidecar lands when the agent finishes; the
            # renderer's lean fallback covers anyone who downloads the
            # report before the narrator returns. Any failure (creds,
            # network, etc.) is logged inside ``generate_run_report`` —
            # never propagated, never blocks the eval-completion path.
            output_file = finished.output_file
            if output_file:
                run_filename = Path(output_file).name
                asyncio.create_task(_dispatch_run_report(
                    project_dir, job.project_id, run_filename,
                    config.get("owner_id") or config.get("user_id"),
                ))
                # Metric-disagreement audit. AWAITED, not fired: the sidecar it
                # writes is read by the post-eval coordinator message and by the
                # reports, so racing it means the coordinator silently decides on
                # pass rates with no idea a metric is unreliable. It takes a few
                # milliseconds even on a multi-megabyte run file, so backgrounding
                # it bought nothing and cost determinism. It swallows every
                # exception internally, so awaiting cannot fail the run.
                # Guarded here too, not only inside the dispatcher: awaiting puts
                # the audit on the eval's own completion path, so a raising audit
                # must not be able to turn a finished eval into a failed job.
                try:
                    await _dispatch_metric_audit(
                        project_dir, job.project_id, run_filename,
                    )
                except Exception as e:  # noqa: BLE001 — never fail a finished eval
                    logger.warning(
                        "[eval-executor] metric audit failed for %s: %s",
                        run_filename, e,
                    )
                # On a winning run, also kick the exec report so the
                # recommendation tracks the latest trophy. Best-effort —
                # decided server-side by comparing the freshly-built
                # exec facts vs the prior winner stamp.
                asyncio.create_task(_dispatch_exec_if_winner_changed(
                    project_dir, job.project_id,
                    config.get("owner_id") or config.get("user_id"),
                ))

            return JobResult(status="completed", data=result_data)

        elif finished and finished.state == "failed":
            error = finished.error or ResourceError(
                code=Codes.HARNESS_SUBPROCESS_FAILED,
                message="Eval harness failed without a specific error.",
            )
            _publish_project_event(job.project_id, {
                "resource": "job",
                "type": "failed",
                "job_id": job.job_id,
                "job_type": "eval",
                **error.to_dict(),
            })
            return JobResult(status="failed", error=error)

        else:
            # Cancelled, evicted, or disappeared.
            error = ResourceError(
                code=Codes.HARNESS_SUBPROCESS_FAILED,
                message="Eval job ended in an unexpected state.",
            )
            tracker.update_state(job.job_id, "failed", error=error)
            return JobResult(status="failed", error=error)

    except asyncio.CancelledError:
        # State already set to "cancelled" by JobTracker.cancel_job
        # (the external caller that triggered this cancellation).
        # Just re-raise — don't call cancel_job again.
        raise
    except Exception as e:
        logger.error("[eval-executor] job=%s error: %s", job.job_id[:8], e, exc_info=True)
        error = ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message=f"Eval execution failed: {e}",
            details={"exception": type(e).__name__},
        )
        tracker.update_state(job.job_id, "failed", error=error)
        _publish_project_event(job.project_id, {
            "resource": "job",
            "type": "failed",
            "job_id": job.job_id,
            "job_type": "eval",
            **error.to_dict(),
        })
        return JobResult(status="failed", error=error)


async def _dispatch_run_report(
    project_dir: Path, project_id: str, run_filename: str, owner_id: str | None,
) -> None:
    """Fire-and-forget per-run report generation. Logs and swallows errors."""
    try:
        from app.services.report_generation import generate_run_report
        await generate_run_report(project_dir, project_id, run_filename, owner_id)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s run=%s background generation failed: %s",
            project_id[:8], run_filename, e,
        )


async def _dispatch_metric_audit(
    project_dir: Path, project_id: str, run_filename: str,
) -> None:
    """Metric-disagreement audit for a finished run.

    Writes a ``<run>.metric-audit.json`` sidecar, which ``run_trust`` reads to
    build the verdict shown in both reports and sent to the post-eval coordinator
    turn. NOT fire-and-forget for that reason — see the awaited call site.
    Deliberately dispatched from
    BOTH completion paths — standalone evals here and optimization iterations in
    ``optimization_executor`` — because optimization iterations bypass
    ``run_eval_job`` entirely, so wiring only one path silently covers half the
    runs (and the optimization half is where a bad metric compounds).

    Logs and swallows everything: an audit must never fail a run that already
    produced results.
    """
    try:
        from app.services.metric_disagreement import audit_run_file, gating_metrics_for
        run_file = project_dir / "output" / run_filename
        if not run_file.exists():
            logger.warning(
                "[metric-audit] project=%s run=%s missing on disk — skipped",
                project_id[:8], run_filename,
            )
            return
        audit_run_file(run_file, gating_metrics_for(project_dir))
    except Exception as e:  # noqa: BLE001 — commentary must not break the run
        logger.warning(
            "[metric-audit] project=%s run=%s audit failed: %s",
            project_id[:8], run_filename, e,
        )


async def _dispatch_exec_if_winner_changed(
    project_dir: Path, project_id: str, owner_id: str | None,
) -> None:
    """Regenerate the exec report when the winning run has changed.

    Cheap check: compare the new ``_resolve_exec_winner`` outcome against
    the winner stamped on any existing ``executive.report.json``. When they
    differ, regenerate. A run that's merely WORSE than the winner won't
    change the winner, so this stays quiet — but the generated report
    carries a "generated from run X · date" stamp so a reader of the
    latest-but-worse run isn't confused.

    The resolver matches the GET-side check in ``api/reports.py``: with
    targets it picks the trophy, without targets it falls back to best
    pass rate. Sharing the resolver keeps the cache-drop and the
    auto-regen symmetric, so a no-targets project doesn't get stuck on the
    lean fallback after each new eval.
    """
    try:
        from app.services.eval_runner import EvalRunner
        from app.services.optimization_state import OptimizationState
        from app.services.report import _resolve_exec_winner
        from app.services.report_generation import (
            generate_exec_report, previous_winner_filename,
        )

        state = OptimizationState.load(project_dir)
        new_winner = _resolve_exec_winner(project_dir, EvalRunner(), state)
        if new_winner is None:
            return  # Project has no runs yet.
        prior_winner = previous_winner_filename(project_dir)
        if prior_winner == new_winner:
            return  # No change.
        await generate_exec_report(project_dir, project_id, owner_id)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s exec winner-change check failed: %s",
            project_id[:8], e,
        )
