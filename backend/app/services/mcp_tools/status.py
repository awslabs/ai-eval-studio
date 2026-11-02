# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool for project status snapshot: ``get_project_status``.

Returns a comprehensive snapshot of the project's current state including
harness config, optimization state, available prompts, and the latest
eval run summary. The coordinator prompt instructs the agent to call this
before answering any question about the project so narration doesn't
drift from the actual state (the optimization loop runs autonomously in
the background).
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.services.harness_config import load_harness_config
from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


@tool(
    "get_project_status",
    "Get the current project status: harness config, optimization state, latest eval results, "
    "and available prompt versions. ALWAYS call this before answering questions about the "
    "project's current state — do NOT rely on conversation history.",
    {"project_id": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_project_status(args: dict[str, Any]) -> dict[str, Any]:
    """Return a comprehensive snapshot of the project's current state.

    Includes harness config (models, field mappings, optimization params),
    optimization state (status, iterations, convergence), available prompts, and
    latest eval run summary. This gives the agent full context regardless of
    conversation history.

    On the durable path (arch_split=True), rebuilds jobs from the store before
    reading the tracker so the agent sees durable jobs even when the in-memory
    tracker is empty (cross-process/microVM case), and the durable store wins on
    terminal state (refresh_terminal=True). The optimization state is also read
    from the durable store on this path. On the old path (arch_split=False),
    rebuild is a no-op and optimization reads from EFS.

    Args:
        args: Dict with project_id.

    Returns:
        Dict with project status snapshot.
    """
    logger.info("[get_project_status] project=%s", args.get("project_id", "?")[:12])
    try:
        project_dir = resolve_project_dir(args["project_id"])
        status: dict[str, Any] = {}

        # Harness config
        harness = load_harness_config(project_dir)
        if harness is not None:
            status["harness_config"] = {
                "models": harness.models,
                "data_file": harness.data_file,
                "judge_model_id": harness.judge_model_id,
                "prompt_version": harness.prompt_version,
                "field_mappings": harness.field_mappings,
                "target_metric_scores": harness.target_metric_scores,
                "max_iterations": harness.max_iterations,
                "plateau_window": harness.plateau_window,
            }
        else:
            status["harness_config"] = None

        # Resolve arch_split ONCE and share between optimization read and job rebuild
        from app.platform.arch_split import resolve_arch_split
        try:
            arch_split = await resolve_arch_split(args["project_id"])
        except Exception as e:
            # Resolver failure degrades to old-path behavior
            logger.warning(
                "[get_project_status] resolve_arch_split failed for project=%s: %s",
                args["project_id"][:12], e
            )
            arch_split = False

        # Optimization state.
        #
        # read_for_display degrades a store failure to None ("unknown"), NOT to an
        # EFS read. This site used to fall back to EFS, which on the durable path is
        # the WRONG HOME: the agent would read plausible-but-stale state and report
        # it to the user as fact. "I can't tell" is the honest answer, and it is the
        # one the caller can act on.
        #
        # The read is also keyed on the project OWNER now rather than the requester
        # resolved from current_request() with a dev fallback — in the microVM there
        # is no request context, so that fallback read an empty partition.
        from app.services import optimization_state_repository as opt_repo

        opt_state = await opt_repo.read_for_display(
            args["project_id"], project_dir, arch_split=arch_split
        )

        if opt_state:
            status["optimization"] = {
                "status": opt_state.status,
                "current_iteration": opt_state.current_iteration,
                "max_iter_cap": opt_state.max_iter_cap,
                "max_iterations_per_run": opt_state.max_iterations,
                "target_metric_scores": opt_state.target_metric_scores,
                "best_iteration": opt_state.best_iteration(),
                "latest_iteration": opt_state.iterations[-1] if opt_state.iterations else None,
                "iteration_count": len(opt_state.iterations),
            }
        else:
            status["optimization"] = None

        # Available prompts
        prompts_dir = project_dir / "prompts"
        if prompts_dir.exists():
            prompts = sorted([f.stem for f in prompts_dir.iterdir() if f.suffix == ".md"])
            status["prompts"] = prompts
        else:
            status["prompts"] = []

        # Latest eval run summary
        from app.services.eval_runner import EvalRunner
        runner = EvalRunner()
        runs = runner.list_runs(project_dir)
        if runs:
            latest = runs[0]  # Already sorted newest first
            status["latest_run"] = {
                "file": latest.get("_file"),
                "prompt_version": latest.get("prompt_version"),
                "timestamp": latest.get("timestamp"),
                "models": [m.get("test_run_id") for m in latest.get("model_runs", [])],
                "total_cases": sum(m.get("total", 0) for m in latest.get("model_runs", [])),
            }
            status["total_runs"] = len(runs)
        else:
            status["latest_run"] = None
            status["total_runs"] = 0

        # Harness exists?
        status["harness_ready"] = (project_dir / "run_eval.py").exists()

        # Active jobs from the unified JobTracker
        # Rebuild from durable store if on the arch_split path (microVM recovery)
        # Reuse the already-resolved arch_split from above
        from app.services.jobs import get_job_tracker

        job_tracker = get_job_tracker()

        # Rebuild jobs from durable store before reading (arch_split already resolved)
        try:
            await job_tracker.rebuild_jobs_from_store(args["project_id"], arch_split, refresh_terminal=True)
        except Exception as e:
            # Store errors degrade to old-path behavior (arch_split=False fallback)
            # rather than raising into the agent loop
            logger.warning(
                "[get_project_status] rebuild failed for project=%s: %s",
                args["project_id"][:12], e
            )

        active_jobs = job_tracker.get_active_jobs(args["project_id"])
        if active_jobs:
            status["active_jobs"] = [
                {
                    "job_id": j.job_id,
                    "type": j.type,
                    "state": j.state,
                    "pause_reason": j.pause_reason,
                    "created_at": j.created_at,
                }
                for j in active_jobs
            ]
        else:
            status["active_jobs"] = []

        return mcp_json(status)
    except Exception as e:
        logger.error("[get_project_status] Error: %s", e, exc_info=True)
        return mcp_error(f"Error: {e}")
