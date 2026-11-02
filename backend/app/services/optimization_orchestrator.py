# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization orchestrator utilities.

Provides ``OptimizationConfig``, ``dispatch_eval_job``,
``detect_latest_prompt`` and ``next_prompt_version`` — shared by the
optimization loop (``optimization_loop_orchestrator``), the per-iteration
executor, the ``dispatch_optimization`` MCP tool, and the REST start/resume
endpoints.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.jobs import get_job_tracker
from app.services.optimization_state import OptimizationState
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """All configuration needed to run the optimization loop autonomously.

    Built by the optimization iteration executor from the loop's iteration
    config.

    Attributes:
        project_id: Project UUID.
        owner_id: Cognito sub of the project owner (for launch tenant-scoped mint).
        model_ids: Bedrock model IDs to evaluate.
        data_file: Filename in data/raw/.
        field_mappings: Unified field mapping dict from harness config.
        prompt_version: Starting prompt version name.
        judge_model_id: Optional Bedrock model ID for LLM-as-judge.
        target_metric_scores: Target pass rates per metric.
        max_iterations: Maximum optimization iterations.
        plateau_window: Iterations to check for plateau.
        max_test_cases: Optional limit on test cases per eval.
        data_file_override: Optional absolute path (or harness-relative
            name) that takes precedence over the harness's configured data
            file. Set by the start/resume endpoints when the user picks a
            synthetic or merged data source.
    """
    project_id: str
    owner_id: str
    model_ids: list[str]
    data_file: str
    field_mappings: dict[str, str]
    prompt_version: str
    judge_model_id: str | None
    target_metric_scores: dict[str, float]
    max_iterations: int
    plateau_window: int
    max_test_cases: int | None = None
    data_file_override: str | None = None


async def dispatch_eval_job(
    project_dir: Path,
    config: OptimizationConfig,
    prompt_version: str,
    data_file_override: str | None = None,
    source: str = "optimization",
    arch_split: bool = False,
) -> str:
    """Create an eval job and dispatch via the harness subprocess or JobCompute seam.

    Registers the eval job in the JobTracker (for harness_runner progress
    updates, get_project_status visibility, and the concurrent-eval guard).

    Requires ``run_eval.py`` to exist in the project directory; the
    coordinator must scaffold the harness before any optimization run.

    Args:
        project_dir: Path to the project directory.
        config: Optimization config with model/metric/data settings.
        prompt_version: Which prompt version to evaluate.
        data_file_override: Optional data file path override (e.g. a
            resolved synthetic or merged dataset).
        source: Run-attribution tag written into the eval results file
            ("optimization" for loop-driven dispatch, "manual" for direct
            user-triggered runs). Read by the SPA to bucket runs into the
            optimization timeline vs the "other runs" toggle.
        arch_split: Resolved flag threaded from the loop entry. The
            optimization loop resolves once and threads it into iteration_config
            so a mid-run flag toggle cannot split a single run. Never re-resolve
            inside this function.

    Returns:
        The job_id for tracking.

    Raises:
        FileNotFoundError: If the eval harness doesn't exist.
    """
    if not (project_dir / "run_eval.py").exists():
        raise FileNotFoundError(
            f"No eval harness found at {project_dir / 'run_eval.py'}. "
            "The optimization orchestrator requires a harness (Phase 3)."
        )

    from app.platform import get_platform
    from app.services.jobs import get_job_tracker

    # arch_split is threaded from the loop entry — never re-resolve it here

    # Register the eval job once in the JobTracker.
    job_tracker = get_job_tracker()
    job = job_tracker.create_job(
        project_id=config.project_id,
        job_type="eval",
        config={
            "owner_id": config.owner_id,
            "arch_split": arch_split,
            "project_dir": str(project_dir),
            "model_ids": config.model_ids,
            "prompt_version": prompt_version,
            "source": "optimization",
        },
    )

    # Both eval launch paths funnel through this one seam;
    # do not reintroduce a direct create_task on the new path.
    if arch_split:
        # New path: launch via JobCompute (Fargate worker on cloud, in-process on dev)
        from app.platform.job_compute import get_job_compute
        from app.services import launch_gate

        # Settle the launch/cancel race immediately before starting the worker.
        # The job has been visible since create_job above, and the durable
        # compute_handle is not written until after RunTask returns, so a cancel
        # arriving between the two saw no handle and no tagged task and used to read
        # that as "nothing to stop" — then this code launched a worker for a job
        # already marked cancelled.
        decision = await launch_gate.claim_for_launch(
            config.owner_id, config.project_id, job.job_id
        )
        if decision is launch_gate.LaunchDecision.UNAVAILABLE:
            # Launching is unsafe with cancellation state unknown, and no cancel
            # owns this job, so it must not be left looking live. Raising reaches
            # the iteration's error handling, which fails the job.
            raise RuntimeError(
                f"could not confirm whether eval {job.job_id[:8]} was still "
                "wanted before starting its worker"
            )
        if decision is launch_gate.LaunchDecision.CANCELLED:
            # The cancel that won the gate owns the terminal state. Report the job
            # id so the caller can observe the outcome rather than wait on a worker
            # that will never start.
            logger.info(
                "[opt-orchestrator] eval=%s was cancelled before its worker "
                "started; not launching",
                job.job_id[:8],
            )
            return job.job_id

        compute = get_job_compute()
        try:
            handle = await compute.launch(
                job_id=job.job_id,
                owner_id=config.owner_id,
                project_id=config.project_id,
            )
        except BaseException:
            # No worker exists — hand the gate back so a later cancel is not left
            # waiting on a launch that never happened.
            await launch_gate.abandon_launch(
                config.owner_id, config.project_id, job.job_id
            )
            raise

        # Persist the compute_handle on the durable job record for cancel/status
        store = get_platform().metadata_store
        await store.update_job_state(
            config.owner_id,
            config.project_id,
            job.job_id,
            compute_handle=str(handle),
        )

        # Start a PROGRESS-ONLY durable bridge so per-case progress republishes
        # onto the owner's /ws job:<id> stream (live bar during optimization
        # iterations — parity with the manual eval path). fire_completion=False:
        # the optimization loop owns iteration completion
        # (``optimization_loop_orchestrator.run_optimization_loop``, which
        # bypasses claim_completion_fired), so a claiming poller here
        # would double-fire. Lazy import mirrors the manual path (dispatch.py).
        from app.services.job_progress_republisher import maybe_start_poller

        await maybe_start_poller(
            config.project_id,
            job.job_id,
            config.owner_id,
            arch_split=True,
            fire_completion=False,
        )
    else:
        # Old path: spawn in-process via asyncio.create_task (preserved byte-for-byte)
        from app.services.harness_runner import run_eval_via_harness
        from app.services.task_error_sink import attach_error_sink

        task = asyncio.create_task(
            run_eval_via_harness(
                job_id=job.job_id,
                project_dir=project_dir,
                model_ids=config.model_ids,
                prompt_version=prompt_version,
                max_test_cases=config.max_test_cases,
                judge_model_id=config.judge_model_id,
                data_file_override=data_file_override,
                source=source,
            ),
            name=f"eval-{job.job_id[:8]}",
        )
        job._task = task
        attach_error_sink(task, project_id=config.project_id, resource="eval_job")

    return job.job_id


def detect_latest_prompt(project_dir: Path) -> str:
    """Find the newest prompt version in the project's prompts directory.

    Ranks ``baseline`` below any ``vN`` (baseline is the starting point).
    Among ``vN`` files, picks the highest N. Files with unrecognised names
    (e.g. ``draft.md``, ``scratch.md``) are ignored — only the canonical
    baseline/vN scheme counts toward optimizer bookkeeping.

    Args:
        project_dir: Path to the project directory.

    Returns:
        Prompt version name (without .md extension), or ``baseline`` when
        no matching files exist.
    """
    prompts_dir = project_dir / "prompts"
    if not prompts_dir.exists():
        return "baseline"

    # (rank, tiebreak, name) — rank 0 for baseline, (1, n) for v{n}.
    # The extra ``1`` on v* entries keeps them strictly above baseline
    # even when v0.md somehow exists (baseline stays the rank-zero slot).
    versions: list[tuple[int, int, str]] = []
    for f in prompts_dir.iterdir():
        if not (f.is_file() and f.suffix == ".md"):
            continue
        name = f.stem
        if name == "baseline":
            versions.append((0, 0, name))
        elif name.startswith("v") and name[1:].isdigit():
            versions.append((1, int(name[1:]), name))

    if not versions:
        return "baseline"

    versions.sort()
    return versions[-1][2]


def next_prompt_version(current: str) -> str:
    """Compute the next prompt version name from the current one.

    ``baseline`` (or any non-``vN`` name) → ``v1``; ``vN`` → ``v{N+1}``.

    Derived from the current prompt NAME, never from a loop counter. The
    loop's iteration counter is 1-based, so the old ``v{iteration + 1}``
    formula skipped v2 (baseline → v1 → v3 → …); name-derivation also
    survives resumes/stalls where the counter and the on-disk version
    diverge. The optimizer follows the returned name literally when it
    saves the new prompt, so this is the single origin of the version
    name.

    Args:
        current: The current prompt version (e.g. ``"baseline"``, ``"v3"``).

    Returns:
        The next version name (e.g. ``"v1"``, ``"v4"``).
    """
    if current.startswith("v") and current[1:].isdigit():
        return f"v{int(current[1:]) + 1}"
    return "v1"
