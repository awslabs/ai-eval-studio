# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion turn service — fires coordinator turns from job results.

When a Job finishes (completed, failed, awaiting_user, or reports an
intermediate milestone), this service constructs a synthetic user message
from the job result and submits a coordinator turn via the turn queue.

Key behaviors:
- Coalescing: if a completion turn is already queued (not running) for
  the same project, new results merge into its synthetic message rather
  than producing a second near-identical turn.
- Visibility: controlled by project config (visible or silent).
- The coordinator turn receives the result as its user message, reviews
  it, and can delegate to subagents with full project context.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Literal

from app.services.jobs import Job, get_job_tracker, is_loop_owned

logger = logging.getLogger(__name__)

# Pending completion messages keyed by project_id. When a completion
# turn is queued-but-not-running, new results merge here.
_pending_completions: dict[str, list[dict[str, Any]]] = {}
_pending_locks: dict[str, asyncio.Lock] = {}

# Acquire-or-wait bound for the one-writer-per-transcript session claim on
# the completion/orchestrator path. User turns run minutes, so the ceiling is
# generous by default; each value is env-tunable.
#
# The completion turn takes the SAME per-(user,project) DDB claim a relay user
# turn takes — that shared claim across both processes IS the cross-process
# serializer (prevents an eval-completion turn and a relayed user turn from
# writing the session+transcript concurrently). When relay is off, no claim is
# taken.
_COMPLETION_CLAIM_WAIT_CEILING_S = float(
    os.environ.get("COMPLETION_CLAIM_WAIT_CEILING_S", "600.0")
)
_COMPLETION_CLAIM_BACKOFF_INITIAL_S = float(
    os.environ.get("COMPLETION_CLAIM_BACKOFF_INITIAL_S", "0.5")
)
_COMPLETION_CLAIM_BACKOFF_CAP_S = float(
    os.environ.get("COMPLETION_CLAIM_BACKOFF_CAP_S", "5.0")
)


# Retain scheduled claim-release tasks so they are not GC'd mid-flight
# (the same retention pattern as ``jobs._durable_append_tasks`` and the
# chat router's ``_relay_tasks``).
_claim_release_tasks: set[asyncio.Task] = set()


def _schedule_claim_release(
    owner_id: str, project_id: str, *, turn_id: str | None = None
) -> None:
    """Schedule a best-effort async claim release from a sync context.

    Used by the sync ``_build`` closure: if the turn fails to build, the
    tracker swallows the exception and ``run_agent_turn``'s finally-release
    never fires, so the claim must be released here. ``_build`` runs inside the
    event loop but is synchronous, so we spawn a task rather than await.

    Args:
        owner_id: The project owner's Cognito sub (the claim user_id).
        project_id: Project UUID.
        turn_id: The turn whose claim this is, so the release only touches its
            own claim and never a successor's.
    """
    from app.services.agentcore_session_registry import release_turn

    async def _release() -> None:
        try:
            await release_turn(owner_id, project_id, turn_id=turn_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[completion-turn] scheduled claim release failed "
                "(best-effort): %s", e,
            )

    try:
        task = asyncio.create_task(
            _release(), name=f"claim-release-{project_id[:8]}"
        )
        _claim_release_tasks.add(task)
        task.add_done_callback(_claim_release_tasks.discard)
    except RuntimeError:
        # No running loop (shouldn't happen inside _build, but be safe) —
        # nothing more to do; the TTL + startup reconciler are the backstop.
        logger.warning(
            "[completion-turn] could not schedule claim release for project=%s "
            "(no running loop) — TTL/reconciler will clear it", project_id[:12],
        )


def _resolve_project_owner(project_id: str) -> str:
    """Resolve the project owner's Cognito sub for the session claim.

    Returns the SAME value ``AgentTurn.owner_id`` resolves to (via
    ``_resolve_owner_id_for_project``) so the claim acquired here on
    ``(owner, project)`` matches the release in ``run_agent_turn``'s finally.

    Args:
        project_id: Project UUID.

    Returns:
        The project owner's user_id (dev fallback when no metadata).
    """
    from app.services.agent_turn import _resolve_owner_id_for_project

    return _resolve_owner_id_for_project(project_id)


async def _acquire_completion_claim_or_wait(
    owner_id: str, project_id: str, *, turn_id: str | None = None
) -> bool:
    """Acquire the shared (user, project) session claim, waiting if held.

    The completion/orchestrator turn takes the SAME durable DDB
    claim a relay user turn takes so the two never write session+transcript
    concurrently. Acquire-or-WAIT with a generous, env-tunable ceiling and
    exponential backoff (a relay user turn may run minutes). On timeout the
    caller DEFERS (does not run the turn, does not drop the result) — the eval
    result stays persisted on the Job/durable record for the next turn or the
    reconciler (defer-not-drop; never a silent drop).

    Args:
        owner_id: The project owner's Cognito sub (the claim user_id — the
            SAME value ``AgentTurn.owner_id`` resolves to, so the release in
            ``run_agent_turn`` uses the matching (owner, project)).
        project_id: Project UUID.
        turn_id: Turn the claim is taken for. Recorded on the claim so the
            eventual release can be fenced to this turn rather than releasing
            whatever happens to be running.

    Returns:
        ``True`` iff the claim was won within the ceiling, ``False`` on timeout.
    """
    from app.services.agentcore_session_registry import claim_turn

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _COMPLETION_CLAIM_WAIT_CEILING_S
    backoff = _COMPLETION_CLAIM_BACKOFF_INITIAL_S

    while True:
        won = await claim_turn(owner_id, project_id, turn_id=turn_id)
        if won:
            return True
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        # Sleep the backoff, but never past the deadline
        await asyncio.sleep(min(backoff, remaining))
        backoff = min(backoff * 2, _COMPLETION_CLAIM_BACKOFF_CAP_S)


def _get_lock(project_id: str) -> asyncio.Lock:
    """Get or create a per-project lock for coalescing."""
    if project_id not in _pending_locks:
        _pending_locks[project_id] = asyncio.Lock()
    return _pending_locks[project_id]


async def _build_synthetic_message(job: Job) -> str:
    """Build a synthetic user message from a job's result.

    The message reads like a status report the coordinator can act on.
    Each job type produces a distinct format tuned for the coordinator's
    Phase 6 instructions.

    Per-job-type ``job.result`` schema (load-bearing — reading the
    wrong shape used to crash the completion turn with AttributeError):

    * ``optimization_iteration`` → ``dict`` produced by
      ``optimization_executor`` with passed/total/convergence/etc.
    * ``eval`` → ``list[dict]`` of per-model run dicts written by
      ``harness_runner.complete_job``. The result file path lives on
      ``job.output_file`` (a separate dataclass field), NOT inside
      ``job.result``.
    * ``harness_build`` → ``dict`` with validate sub-payload.

    Args:
        job: The completed/failed/awaiting_user Job.

    Returns:
        Human-readable synthetic message.
    """
    config = job.config

    if job.type == "optimization_iteration":
        result = job.result or {}
        iteration = result.get("iteration") or config.get("iteration", "?")
        prompt_version = result.get("prompt_version") or config.get("prompt_version", "?")
        passed = result.get("passed_cases", 0)
        total = result.get("total_cases", 0)
        rate = f"{passed/total*100:.1f}%" if total else "N/A"
        metrics = result.get("metrics_summary", {})
        run_file = result.get("run_output_file", "")
        convergence = result.get("convergence")
        per_model = result.get("per_model_pass_rates", {})

        parts = [
            f"Optimization iteration {iteration} complete.",
            f"Prompt: {prompt_version}",
            f"Pass rate: {passed}/{total} ({rate})",
        ]
        if per_model:
            model_lines = [f"  {m}: {r*100:.1f}%" for m, r in per_model.items()]
            parts.append("Per-model pass rates:\n" + "\n".join(model_lines))
        if metrics:
            metric_lines = [f"  {m}: {v*100:.1f}%" for m, v in metrics.items()]
            parts.append("Per-metric pass rates:\n" + "\n".join(metric_lines))
        if run_file:
            parts.append(f"Run file: {run_file}")

        # Include best-so-far context so the coordinator doesn't falsely
        # call incremental improvements "new best" against local deltas.
        try:
            from app.services.project_paths import resolve_project_dir
            project_dir = resolve_project_dir(job.project_id)

            # arch_split comes from the JOB CONFIG, not a fresh resolve: the job was
            # dispatched under a specific flag value and must be described under that
            # same one, even if the project flag changed since.
            from app.services import optimization_state_repository as opt_repo

            opt_state = await opt_repo.read_for_display(
                job.project_id,
                project_dir,
                arch_split=config.get("arch_split", False),
            )

            if opt_state:
                best = opt_state.best_iteration()
                if best:
                    best_rate = (
                        best["passed"] / best["total_cases"]
                        if best.get("total_cases")
                        else 0
                    )
                    is_new_best = (best.get("prompt_version") == prompt_version)
                    parts.append(
                        f"\nBest so far: {best.get('prompt_version')} at "
                        f"{best_rate*100:.1f}% "
                        f"({'this iteration is the new best' if is_new_best else 'still the best'})."
                    )
        except Exception:
            # Best-so-far is context that sharpens the summary, not part of it.
            # The completion turn must still be delivered if optimization state
            # is missing or unreadable, so this failure is logged and dropped
            # rather than raised.
            logger.debug(
                "[completion] No best-so-far context for job %s",
                job.job_id[:8],
                exc_info=True,
            )

        # Before the convergence instruction, not after: a saturated gating
        # metric means "converged" may be an artifact of the metric, and the
        # coordinator must weigh that against the instruction rather than read
        # it as a footnote to a decision already stated.
        if run_file:
            trust_text = _trust_text(job.project_id, run_file)
            if trust_text:
                parts.append(trust_text)

        if convergence:
            parts.append(f"\nConvergence: {convergence}. Optimization complete.")
            parts.append("Review the results and provide a summary to the user.")
        else:
            parts.append("\nNot converged. Analyze failures and write an improved prompt.")

        return "\n".join(parts)

    elif job.type == "eval":
        # Eval result is a list[dict] of per-model runs — ``output_file``
        # lives on the Job dataclass, not inside the result. Reading it
        # as a dict via ``.get(...)`` used to AttributeError the
        # completion-turn task on every successful eval.
        output_file = job.output_file or ""
        model_runs = job.result if isinstance(job.result, list) else []
        total = sum(len(mr.get("results", [])) for mr in model_runs)
        passed = sum(
            sum(1 for r in mr.get("results", []) if r.get("success"))
            for mr in model_runs
        )
        prompt_version = config.get("prompt_version", "?")

        # A failed eval must not read as a finished one: without this branch it
        # reaches the coordinator as "complete, pass rate 0/0", and the
        # coordinator relays that as a result.
        if job.state == "failed":
            return "\n".join([
                f"Eval on prompt '{prompt_version}' failed: {_failure_reason(job)}",
                "Tell the user it failed and why in one or two sentences, and "
                "suggest what to do next.",
            ])

        # Detect "result not ready" for an eval (cross-process race).
        # A genuinely-completed eval ALWAYS scored >=1 case. total==0 on a
        # completed job means the result field isn't loaded yet (eventual
        # consistency window), NOT a real zero. Degrade to a neutral seed
        # that doesn't fabricate a pass rate — never report 0/0 (N/A) / "no
        # cases scored" / "file may be malformed" on a not-ready result.
        if total == 0 and job.state == "completed":
            # Result not ready yet — degrade to neutral seed
            parts = [
                f"Eval on prompt '{prompt_version}' complete.",
                "Results are finalizing; review the results file.",
            ]
            if output_file:
                parts.insert(1, f"Results file: {output_file}")
            return "\n".join(parts)

        # Normal path: result is ready, report the real rate
        rate = f"{passed/total*100:.1f}%" if total else "N/A"
        parts = [
            f"Eval on prompt '{prompt_version}' complete.",
            f"Pass rate: {passed}/{total} ({rate})",
        ]
        if output_file:
            parts.append(f"Results file: {output_file}")
            trust_text = _trust_text(job.project_id, output_file)
            if trust_text:
                parts.append(trust_text)
        parts.append("Review the results.")
        return "\n".join(parts)

    elif job.type == "harness_build":
        result = job.result or {}
        validate_result = result.get("validate", {})
        has_error = job.state == "failed"
        if has_error:
            error_msg = job.error.message if job.error else "Unknown error"
            return f"Harness build failed: {error_msg}\nReview and fix the issue."
        return "Harness build complete. Validation passed. Review."

    return f"Job {job.job_id[:8]} ({job.type}) finished with state: {job.state}."


def _failure_reason(job: Job) -> str:
    """The most specific account of why a job failed that the job carries.

    ``job.error`` when the failure was recorded in this process. A job rebuilt
    from the durable record has no error — the record does not persist it — but
    ``fail_job`` writes the reason into the progress message, and that is
    persisted. A job the reconciler failed because its worker vanished has
    neither.
    """
    if job.error is not None and job.error.message:
        return job.error.message
    message = (job.progress or {}).get("message")
    if isinstance(message, str) and message.startswith("Failed: "):
        return message.removeprefix("Failed: ")
    return "its worker stopped without reporting a result"


def _trust_text(project_id: str, run_file: str) -> str:
    """Metric-trust lines for a finished run, or ``""`` if there are none.

    Best-effort by contract, like every other enrichment on this message: a run
    that produced results must still reach the coordinator even if the trust
    verdict cannot be assembled. Returning ``""`` on failure means the message
    is silent about trust rather than claiming the run is clean.

    Both callers pass a path or bare filename from the job result, so the
    basename is taken here rather than at each call site.
    """
    try:
        from pathlib import Path

        from app.services.project_paths import resolve_project_dir
        from app.services.report import load_run_trust
        from app.services.run_trust import format_for_agent

        return format_for_agent(
            load_run_trust(resolve_project_dir(project_id), Path(run_file).name)
        )
    except Exception as e:  # noqa: BLE001 — never block a completion turn
        logger.warning(
            "[completion-turn] project=%s run=%s trust verdict unavailable: %s",
            project_id[:8], run_file, e,
        )
        return ""


def _is_degenerate_result(job: Job) -> bool:
    """Check if a completed job has zero scored cases (a degenerate result).

    A degenerate result is a **completed** job whose result carries zero scored
    cases. This typically occurs when an optimization iteration's child eval runs
    but produces no scored cases (the loop already reports the iteration, so the
    child eval's completion adds nothing).

    Definition: completed job + zero scored cases derived per job type from what
    _build_synthetic_message already reads. Failed/awaiting_user jobs are NOT
    degenerate — a failure with zero cases is exactly when the coordinator most
    needs to hear about it.

    **Design decision on the manual eval not-ready case:**
    The neutral-seed branch in ``_build_synthetic_message`` exists for a
    cross-process race where a genuinely-completed eval's result is not yet
    loaded. For a manual eval, the neutral seed is the user's ONLY signal that
    their eval finished — suppressing it would leave a single-eval user with
    silence, which is worse than a vague message. So this predicate returns
    False for that branch (total==0 AND completed AND eval type) — preserve the
    neutral seed for manual evals. For optimization_iteration jobs, total==0 AND
    completed IS degenerate (the loop already reported the iteration).

    Args:
        job: The Job to check.

    Returns:
        True if the job is completed with zero scored cases and should be suppressed.
    """
    if job.state != "completed":
        return False

    if job.type == "optimization_iteration":
        result = job.result or {}
        total_cases = result.get("total_cases", 0)
        return total_cases == 0

    elif job.type == "eval":
        # For eval jobs, total==0 on a completed job means the result is not
        # ready yet, not a real zero, and the neutral seed covers it. Do NOT
        # suppress that — it's the user's only signal for a manual eval. Return
        # False here so the neutral-seed message still fires.
        return False

    # Other job types (harness_build, etc.) never degenerate
    return False


def _build_coalesced_message(messages: list[dict[str, Any]]) -> str:
    """Build a single synthetic message from multiple coalesced results.

    Args:
        messages: List of dicts with "job" and "synthetic_text" keys.

    Returns:
        Combined message with clear separators.
    """
    if len(messages) == 1:
        return messages[0]["synthetic_text"]

    parts = []
    for i, msg in enumerate(messages, 1):
        parts.append(f"--- Result {i} ---")
        parts.append(msg["synthetic_text"])
    return "\n\n".join(parts)


async def request_completion_turn(
    project_id: str,
    job: Job,
    visibility: Literal["visible", "silent"] = "visible",
) -> None:
    """Request a coordinator completion turn for a finished job.

    The optimization loop awaits each completion turn before starting
    the next iteration, so coalescing doesn't trigger in today's flow.
    Kept as defensive scaffolding for future parallel-dispatch scenarios
    (multi-project batch, concurrent optimizations).

    Suppresses degenerate results (completed job with zero scored cases) at
    the publisher — the coordinator never receives a turn for them. A failed
    job with zero cases is NOT suppressed (the coordinator must hear about
    failures even when no cases were scored).

    Args:
        project_id: Project UUID.
        job: The Job whose result triggers the turn.
        visibility: Whether the turn's reply shows in chat or only transcript.
    """
    # Suppress degenerate zero-case completions at the publisher (not downstream).
    # Logged so operators can distinguish "suppressed a phantom" from "broke".
    if _is_degenerate_result(job):
        logger.info(
            "[completion-turn] Suppressed degenerate zero-case completion for "
            "job=%s (type=%s, state=%s) on project=%s",
            job.job_id[:8], job.type, job.state, project_id[:12],
        )
        return

    synthetic_text = await _build_synthetic_message(job)
    entry = {
        "job_id": job.job_id,
        "job_type": job.type,
        "synthetic_text": synthetic_text,
        "visibility": visibility,
    }

    lock = _get_lock(project_id)
    async with lock:
        # Check if there's already a queued completion for this project
        if project_id in _pending_completions:
            # Coalesce: merge into the existing queued message
            _pending_completions[project_id].append(entry)
            logger.info(
                "[completion-turn] Coalesced result for project=%s "
                "(now %d pending messages)",
                project_id[:12],
                len(_pending_completions[project_id]),
            )
            return

        # No existing queue — create one and dispatch
        _pending_completions[project_id] = [entry]

    # Fire the completion turn (outside the lock so it doesn't block coalescing)
    asyncio.create_task(
        _execute_completion_turn(project_id, visibility),
        name=f"completion-turn-{project_id[:8]}",
    )


async def _execute_completion_turn(
    project_id: str,
    visibility: Literal["visible", "silent"],
) -> None:
    """Execute the queued completion turn for a project.

    Drains the pending-completions queue, builds the coalesced message,
    and submits a coordinator turn via the turn queue. The turn queue
    ensures this doesn't evict a running user turn.

    Args:
        project_id: Project UUID.
        visibility: Whether the turn's reply shows in chat.
    """
    from app.services.agent_turn import get_agent_turn_tracker
    from app.platform.arch_split import resolve_agentcore_relay

    # Brief yield to let other coroutines run — coalescing happens when
    # request_completion_turn fires while we still have pending entries
    # (the check in request_completion_turn handles that). This yield is
    # purely cooperative, not a timing-based coalescing window.
    await asyncio.sleep(0)

    lock = _get_lock(project_id)
    async with lock:
        messages = _pending_completions.pop(project_id, [])
        if not messages:
            return

    synthetic_message = _build_coalesced_message(messages)

    logger.info(
        "[completion-turn] Submitting to queue for project=%s (%d coalesced results, %s)",
        project_id[:12], len(messages), visibility,
    )

    # Resolve the relay flag ONCE. When relay is on, the completion turn must
    # take the SAME per-(user,project) session claim a relay user turn takes
    # so the two never write concurrently. When off, no claim is taken.
    relay_enabled = await resolve_agentcore_relay(project_id)
    holds_claim = False
    owner_id: str | None = None
    # Reserve the turn id BEFORE the claim, so the claim records which turn holds
    # it and the release can be fenced on that. Claiming without an id made the
    # claim unattributable: a release could only assert "something is running",
    # which is not a fence at all once a later turn owns the same item.
    reserved_turn_id = str(uuid.uuid4())
    if relay_enabled:
        owner_id = _resolve_project_owner(project_id)
        # Acquire-or-WAIT (generous bound). On timeout: DEFER — do not run the
        # turn and do not drop the result (it stays on the Job/durable record
        # for the next turn/reconciler). Never a silent drop.
        holds_claim = await _acquire_completion_claim_or_wait(
            owner_id, project_id, turn_id=reserved_turn_id
        )
        if not holds_claim:
            logger.warning(
                "[completion-turn] Deferred: could not acquire session claim for "
                "project=%s within %.0fs (a relay user turn holds it). The eval "
                "result stays persisted for the next turn/reconciler — NOT dropped.",
                project_id[:12],
                _COMPLETION_CLAIM_WAIT_CEILING_S,
            )
            return

    # Resolve the AUTHORITATIVE optimization state in async scope and close over
    # it. build_coordinator_prompt is sync (called from the sync _build closure
    # below), so a read inside it could only be the EFS read — the wrong home on
    # the durable path.
    from app.services import optimization_state_repository as opt_repo
    from app.services.project_paths import resolve_project_dir as _resolve_dir

    turn_opt_state = await opt_repo.read_for_display(project_id, _resolve_dir(project_id))

    def _build(reserved: str):
        """Build and start the coordinator turn when the queue slot opens."""
        from app.services.coordinator_prompt import build_coordinator_prompt
        from app.services.agent_runner import start_agent_turn
        from app.services.project_paths import resolve_project_dir
        from app.main import get_project_manager

        try:
            pm = get_project_manager()
            project = pm.get_project(project_id)
            if not project:
                logger.warning(
                    "[completion-turn] Project %s not found at build time, skipping",
                    project_id[:12],
                )
                raise RuntimeError(f"Project {project_id} not found")

            # Pass the resolved project dir so cloud mode finds the
            # user-prefixed EFS path.
            project_dir = resolve_project_dir(project_id)
            coordinator_prompt = build_coordinator_prompt(
                project, project_dir, turn_opt_state
            )

            turn = start_agent_turn(
                project_id=project_id,
                prompt=synthetic_message,
                coordinator_prompt=coordinator_prompt,
                session_id=project.session_id,
                model=project.agent_model,
                origin="orchestrator",
                # Same id the session claim was taken under, so the fenced
                # release in run_agent_turn recognises its own claim.
                turn_id=reserved,
            )
        except Exception:
            # The queue swallows _build failures (_try_process_queue),
            # so if the turn never gets built the finally-release in
            # run_agent_turn never fires. Release the claim here (schedule an
            # async release — _build is sync but runs inside the event loop)
            # so a build failure doesn't wedge the project for the TTL.
            if holds_claim and owner_id is not None:
                _schedule_claim_release(owner_id, project_id, turn_id=reserved)
            raise

        # Mark the turn as claim-holder so run_agent_turn's finally
        # releases the claim on ALL exits (success/failure/cancel). This is how
        # the claim wraps the turn LIFETIME through the tracker — the release
        # happens when the tracker-run turn reaches terminal.
        if holds_claim:
            turn._holds_ddb_claim = True

        logger.info(
            "[completion-turn] Started turn=%s for project=%s",
            turn.turn_id[:8], project_id[:12],
        )
        return turn

    # If this queued turn is DROPPED before _build runs (cancel via
    # drain_queued_turns), neither run_agent_turn's finally nor the _build-failure
    # release fires — so the claim would leak until the 2h TTL backstop, wedging
    # the project's relay serialization. Register an on_drop hook that releases
    # the claim on that path. Only when relay actually took the claim; in-process
    # completions pass no hook.
    on_drop = None
    if holds_claim and owner_id is not None:
        _drop_owner = owner_id
        on_drop = lambda: _schedule_claim_release(
            _drop_owner, project_id, turn_id=reserved_turn_id
        )

    # Submit through the turn queue — orchestrator priority (10) so
    # user turns (priority 0) preempt in the queue.
    tracker = get_agent_turn_tracker()
    tracker.submit(
        project_id=project_id,
        origin="orchestrator",
        build_turn=_build,
        on_drop=on_drop,
        turn_id=reserved_turn_id,
    )


async def _submit_orchestrator_message(
    project_id: str, message: str, *, lifecycle: bool = False
) -> None:
    """Submit a plain orchestrator coordinator turn with ``message``.

    The shared core behind both the job-results completion turn and the
    terminal-status notifications: builds the coordinator prompt and
    starts an ``origin="orchestrator"`` turn through the turn queue (so a
    running user turn preempts it). Unlike :func:`request_completion_turn`
    this takes a ready-made message string rather than synthesizing one
    from a Job's results — used for "the loop ended this way" one-liners
    that have no job-result payload.

    Like the job-results completion turn, this takes the SAME
    per-(user,project) session claim a relay user turn takes when relay is
    enabled — acquire-or-wait, defer-not-drop on timeout, release via the
    turn's finally. When relay is off, no claim is taken.

    Args:
        project_id: Project UUID.
        message: The synthetic user message handed to the coordinator.
        lifecycle: If True, prepend LIFECYCLE_MSG_PREFIX to the message body
            so it renders as role:"system" on reload. start_agent_turn will
            prepend ORCHESTRATOR_MSG_PREFIX in front of it (correct order).
    """
    from app.services.agent_turn import get_agent_turn_tracker
    from app.platform.arch_split import resolve_agentcore_relay

    # Resolve the relay flag ONCE and acquire-or-wait when on.
    relay_enabled = await resolve_agentcore_relay(project_id)
    holds_claim = False
    owner_id: str | None = None
    # Reserved before the claim so the claim names its turn — see
    # _execute_completion_turn.
    reserved_turn_id = str(uuid.uuid4())
    if relay_enabled:
        owner_id = _resolve_project_owner(project_id)
        holds_claim = await _acquire_completion_claim_or_wait(
            owner_id, project_id, turn_id=reserved_turn_id
        )
        if not holds_claim:
            logger.warning(
                "[completion-turn] Deferred terminal message: could not acquire "
                "session claim for project=%s within %.0fs (a relay user turn "
                "holds it).",
                project_id[:12],
                _COMPLETION_CLAIM_WAIT_CEILING_S,
            )
            return

    # Resolve the AUTHORITATIVE optimization state in async scope and close over
    # it. build_coordinator_prompt is sync (called from the sync _build closure
    # below), so a read inside it could only be the EFS read — the wrong home on
    # the durable path.
    from app.services import optimization_state_repository as opt_repo
    from app.services.project_paths import resolve_project_dir as _resolve_dir

    turn_opt_state = await opt_repo.read_for_display(project_id, _resolve_dir(project_id))

    # Prepend lifecycle marker to the message BODY if lifecycle=True.
    # start_agent_turn will prepend ORCHESTRATOR_MSG_PREFIX in front of it
    # (correct order: ORCHESTRATOR + LIFECYCLE).
    actual_message = message
    if lifecycle:
        from app.services.chat_message_parser import LIFECYCLE_MSG_PREFIX
        actual_message = LIFECYCLE_MSG_PREFIX + message

    def _build(reserved: str):
        from app.services.coordinator_prompt import build_coordinator_prompt
        from app.services.agent_runner import start_agent_turn
        from app.services.project_paths import resolve_project_dir
        from app.main import get_project_manager

        try:
            pm = get_project_manager()
            project = pm.get_project(project_id)
            if not project:
                logger.warning(
                    "[completion-turn] Project %s not found at build time, skipping",
                    project_id[:12],
                )
                raise RuntimeError(f"Project {project_id} not found")

            project_dir = resolve_project_dir(project_id)
            coordinator_prompt = build_coordinator_prompt(
                project, project_dir, turn_opt_state
            )
            turn = start_agent_turn(
                project_id=project_id,
                prompt=actual_message,
                coordinator_prompt=coordinator_prompt,
                session_id=project.session_id,
                model=project.agent_model,
                origin="orchestrator",
                turn_id=reserved,
            )
        except Exception:
            # Release the claim if the turn never gets built (the
            # queue swallows _build failures — see _execute_completion_turn).
            if holds_claim and owner_id is not None:
                _schedule_claim_release(owner_id, project_id, turn_id=reserved)
            raise

        if holds_claim:
            turn._holds_ddb_claim = True
        logger.info(
            "[completion-turn] Started terminal-message turn=%s for project=%s",
            turn.turn_id[:8], project_id[:12],
        )
        return turn

    # If this queued turn is DROPPED before _build runs (cancel via
    # drain_queued_turns), the claim is released nowhere — not by run_agent_turn's
    # finally (the turn never ran), not by the _build-failure path above (_build
    # never ran). It would leak until the claim's TTL item is actually deleted,
    # which the DynamoDB TTL contract bounds only loosely (deletion lags expiry by
    # up to days), wedging the pair's relay serialization for that whole window.
    # Register the same on_drop release the job-results completion turn uses. Only
    # when relay actually took the claim; in-process completions pass no hook.
    on_drop = None
    if holds_claim and owner_id is not None:
        _drop_owner = owner_id
        on_drop = lambda: _schedule_claim_release(
            _drop_owner, project_id, turn_id=reserved_turn_id
        )

    get_agent_turn_tracker().submit(
        project_id=project_id,
        origin="orchestrator",
        build_turn=_build,
        on_drop=on_drop,
        turn_id=reserved_turn_id,
    )


async def request_terminal_message(
    project_id: str, message: str, *, lifecycle: bool = False
) -> None:
    """Post a one-line coordinator chat message for a loop lifecycle transition.

    When the optimization loop ends without a results-bearing completion
    turn — iteration failure, optimizer stall, crash — chat would
    otherwise go silent (only a toast fires via the project-event bus).
    Chat is a separate surface from the status bar/toast; this wires it
    for the failure terminals the way the success path is wired via
    :func:`request_completion_turn`.

    Also used to record UI-initiated lifecycle transitions (start / resume /
    cancel) so the coordinator has the loop's origin in context and cannot
    attribute a loop to a phantom auto-start. When ``lifecycle=True``, the
    message is prepended with LIFECYCLE_MSG_PREFIX so it renders as an
    attributed system message (role:"system") on reload instead of being
    dropped.

    The coordinator receives ``message`` as a synthetic user turn and
    relays/expands it to the user. Best-effort and fire-and-forget — a
    failure to post the chat line must never mask the underlying transition
    (the status + project event are already published by the caller).

    Args:
        project_id: Project UUID.
        message: One-line, user-facing description of the lifecycle transition.
        lifecycle: If True, prepend LIFECYCLE_MSG_PREFIX so the message renders
            as role:"system" on reload (used for start/resume/cancel).
    """
    try:
        await _submit_orchestrator_message(project_id, message, lifecycle=lifecycle)
    except Exception:
        logger.exception(
            "[completion-turn] failed to post terminal message for project=%s",
            project_id[:12],
        )


#: The background tasks posting a UI-cancel line, retained so they are not
#: garbage-collected mid-flight.
_ui_cancel_notice_tasks: set[asyncio.Task] = set()


def announce_eval_cancelled_from_ui(project_id: str, job: Job) -> None:
    """Record in chat that the user stopped a manual eval from the UI.

    The optimization loop's UI cancel posts a lifecycle line, so the
    conversation — and the coordinator, which replies to it — knows the user
    stopped it. A manual eval stopped from the UI gets the same line.

    Scheduled rather than awaited: posting can wait on the chat session's claim
    while a user turn runs, and the cancel's response must not wait with it.

    Not for an eval the agent cancelled — its own reply says so, and the only
    caller is the UI's cancel route — nor for one owned by an optimization loop,
    which reports its own end.

    Args:
        project_id: Project UUID.
        job: The eval that was cancelled.
    """
    if is_loop_owned(job):
        return
    task = asyncio.create_task(
        request_terminal_message(project_id, _describe_ui_cancel(job), lifecycle=True),
        name=f"completion-turn-cancel-{job.job_id[:8]}",
    )
    _ui_cancel_notice_tasks.add(task)
    task.add_done_callback(_ui_cancel_notice_tasks.discard)


def _describe_ui_cancel(job: Job) -> str:
    """One line recording a UI cancel, naming the prompt and how far it got."""
    prompt_version = job.config.get("prompt_version")
    subject = f"Eval of prompt '{prompt_version}'" if prompt_version else "Eval"
    done = (job.progress or {}).get("models_completed")
    total = (job.progress or {}).get("models_total")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        return f"{subject} cancelled from the UI with {done} of {total} models finished."
    return f"{subject} cancelled from the UI."


def drain_pending_completions(project_id: str) -> int:
    """Remove any queued completion messages for a project.

    Used by cancel endpoints to prevent stale completion turns (from
    jobs that were cancelled before their completion turn fired) from
    running after cancel.

    Args:
        project_id: Project UUID.

    Returns:
        Number of pending messages removed.
    """
    messages = _pending_completions.pop(project_id, [])
    return len(messages)


def _reset() -> None:
    """Clear all pending completions. For testing only."""
    _pending_completions.clear()
    _pending_locks.clear()
