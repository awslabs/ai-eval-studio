# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization loop driver — sequences per-iteration jobs to convergence.

Extracted from ``mcp_tools/dispatch.py``, which had grown to 1,975 lines holding
five MCP tools, this loop, a prompt-optimizer runner and a child-eval reaper. The
loop belongs next to :mod:`app.services.jobs.optimization_executor`, which runs the
iterations it sequences — reading one without the other was the main reason the
loop's state handling drifted from the executor's.

**State handling.** Every state read and write goes through
:mod:`app.services.optimization_state_repository` rather than a hand-rolled
read-mutate-save block. Such blocks are the densest concentration of the
dual-home bug class: each re-implements the arch_split branch, the owner
partition key and the version threading, and each is an independent chance to
get one of the three wrong — a version hand-carried through a local that is
reassigned at many points is exactly the shape that loses a conditional write
when one branch forgets.

**Responsibilities:**

* Drive iterations to a terminal condition (convergence, plateau, stall,
  max-iterations, cancel, or crash).
* Own the run's eval lock: claimed by ``dispatch_optimization`` before its
  invocability preflight and released by this module's terminal tail, so the lock
  cannot outlive the run.
* Publish the terminal project event and post the terminal chat line.

**Not responsibilities:** dispatching the eval (``optimization_orchestrator``),
running an iteration (``jobs.optimization_executor``), or deciding convergence
(``OptimizationState.check_convergence``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.prompts import PROMPTS_DIR
from app.services.jobs import _publish_project_event, get_job_tracker

logger = logging.getLogger(__name__)


def _reap_loop_child_eval(project_id: str) -> None:
    """Cancel any optimization-child eval still active when the loop exits.

    The optimization loop dispatches each iteration's eval as a separate
    asyncio task (``run_eval_via_harness``) tracked under its own ``eval``
    Job. When the loop task itself is cancelled mid-wait (process shutdown,
    a resume race) — i.e. NOT via the cancel REST endpoint, which already
    reaps through ``hard_cancel_project_eval`` — that child eval keeps
    running and stays ``active`` in the tracker. With ``/activity`` now
    trusting the eval job as the single source of truth, an orphaned active
    eval would be reported "running" forever. Reaping it here keeps the
    invariant the activity endpoint depends on: the tracker never holds an
    active eval that no live loop owns.

    Best-effort and idempotent: a child eval that already reached a
    terminal state (the common path on stall/plateau/convergence) is not
    active, so this is a no-op there.
    """
    tracker = get_job_tracker()
    child_eval = tracker.get_active_job_by_type(project_id, "eval")
    if child_eval is None:
        return
    if child_eval.config.get("source") != "optimization":
        # A standalone manual eval is not ours to cancel.
        return
    try:
        tracker.cancel_job(child_eval.job_id)
    except KeyError:
        pass
    logger.info(
        "[opt-loop] Reaped in-flight child eval job=%s on loop exit for project=%s",
        child_eval.job_id[:8], project_id[:12],
    )


async def run_optimization_loop(iteration_config: dict) -> None:
    """Drive the optimization loop using per-iteration Jobs.

    Each iteration:
    1. Creates a Job(type="optimization_iteration") via JobTracker.
    2. Runs the optimization executor (eval + record + convergence check).
    3. Fires a short, visible completion turn in which the coordinator
       summarizes the iteration for the user (no delegation).
    4. Runs the prompt-optimizer directly (``_run_prompt_optimizer``), which
       saves the next prompt version.
    5. Detects the new prompt, loops.

    Args:
        iteration_config: Template config for each iteration job.
    """
    from app.services import optimization_state_repository as opt_repo
    from app.services.completion_turn import (
        request_completion_turn,
        request_terminal_message,
    )
    from app.services.jobs.optimization_executor import run_optimization_iteration_job
    from app.services.optimization_orchestrator import detect_latest_prompt
    from app.services.resource_error import Codes, ResourceError

    project_id = iteration_config["project_id"]
    project_dir = Path(iteration_config["project_dir"])
    prompt_version = iteration_config["prompt_version"]
    max_iterations = iteration_config.get("max_iterations", 10)
    arch_split = iteration_config.get("arch_split", False)
    user_id = iteration_config.get("user_id")

    # EXECUTION-HOME GUARD. This loop must run on the COORDINATOR.
    #
    # A loop task created inside the AgentCore microVM would run there, where:
    # its progress publishes reach a hub with zero browser connections (nobody
    # sees them), its child evals are invisible to the coordinator's tracker,
    # reconciler and caps, and the whole loop dies when the microVM hits its
    # idle or max-lifetime ceiling, stranding state non-terminal. That is the
    # "second execution home" class of defect.
    #
    # Failing loud here rather than degrading: a loop that silently runs in the
    # wrong home looks like it is working. A dispatch from the microVM does not
    # spawn the loop at all — ``dispatch_optimization`` writes the running
    # record and returns, and the coordinator adopts it
    # (:func:`adopt_orphan_optimization_loops`, the dispatch-to-owner seam).
    # This guard turns any other route into the wrong home into an immediate,
    # attributable failure.
    from app.platform.process_role import is_agentcore_runtime

    if is_agentcore_runtime():
        logger.error(
            "[opt-loop] REFUSING to run the optimization loop in the AgentCore "
            "runtime for project=%s. The loop must run on the coordinator: here its "
            "publishes reach no browser, its child evals are invisible to the "
            "tracker/reconciler/caps, and it dies with the microVM session.",
            project_id[:12],
        )
        _publish_project_event(project_id, {
            "resource": "optimization",
            "type": "error",
            "code": "optimization.wrong_execution_home",
            "message": (
                "Optimization could not start: it was dispatched in an execution "
                "home that cannot own it. Retry the request."
            ),
        })
        return

    logger.info(
        "[opt-loop] Starting job-based optimization: project=%s prompt=%s arch_split=%s",
        project_id[:12], prompt_version, arch_split,
    )

    try:
        # ``_run_slot`` is the RUN-relative budget counter (max_iterations
        # iterations per run). The user-visible iteration NUMBER is absolute,
        # derived from the record's iteration history below — a resumed loop
        # continues at N+1, it does not restart at 1 (the logs used to say
        # "Iteration 1" while the bar correctly said "Iteration 4").
        for _run_slot in range(1, max_iterations + 1):
            # Check for external cancellation
            handle = await opt_repo.read(
                project_id, project_dir, arch_split=arch_split, owner_id=user_id
            )
            opt_state = handle.state
            if opt_state is None:
                logger.info("[opt-loop] State=None, exiting")
                break

            if opt_state.status not in ("running",):
                if opt_state.status == "stopping":
                    # A cancel landed between iterations: finish the transition
                    # the canceller started rather than leaving it at "stopping".
                    opt_state.status = "cancelled"
                    handle = opt_repo.write(handle)
                logger.info("[opt-loop] State=%s, exiting", opt_state.status)
                break

            # Absolute iteration number — the record's history is the truth.
            iteration = len(opt_state.iterations) + 1

            # Update current_prompt_version + phase on state so the frontend
            # shows the correct prompt and an honest phase during evaluation.
            # Piggybacks on the write this pass already makes — no extra I/O.
            opt_state.current_prompt_version = prompt_version
            opt_state.phase = "evaluating"
            handle = opt_repo.write(handle)

            # Create a Job for this iteration
            tracker = get_job_tracker()
            job = tracker.create_job(
                project_id=project_id,
                job_type="optimization_iteration",
                config={
                    **iteration_config,
                    "iteration": iteration,
                    "prompt_version": prompt_version,
                },
            )

            logger.info(
                "[opt-loop] Iteration %d: job=%s prompt=%s",
                iteration, job.job_id[:8], prompt_version,
            )

            # Run the iteration (eval + record + convergence check)
            result = await run_optimization_iteration_job(job)

            if result.status == "failed":
                logger.info("[opt-loop] Iteration %d failed, stopping", iteration)
                # Set state to a real terminal status and publish so the
                # SPA flips out of "running" immediately. Without this,
                # state stayed "running" until the loop's tail-block
                # rewrote it to "cancelled" — wrong shape (this isn't a
                # cancel) and not published either, so the SPA would
                # only see the change on its next backstop poll.
                # Re-read: the iteration took time, so the version observed at
                # the top of this pass is stale for a conditional write.
                handle = await opt_repo.read(
                    project_id, project_dir,
                    arch_split=arch_split, owner_id=user_id,
                )
                opt_state = handle.state
                if handle.is_live():
                    opt_state.status = "failed"
                    handle = opt_repo.write(handle)
                # Surface the specific reason when the executor recorded one
                # (metric-error dominance, judge-unreliable, dispatch failure
                # — see optimization_executor); else a generic line.
                reason_detail = (
                    opt_state.error.message
                    if opt_state and opt_state.error
                    else "the eval harness failed"
                )
                _publish_project_event(project_id, {
                    "resource": "optimization",
                    "type": "terminal",
                    "reason": "failed",
                    "message": "Optimization stopped — iteration failed.",
                    "iteration": iteration,
                })
                # Chat is a separate surface from the toast/bar — post a
                # one-line message so chat doesn't go silent after
                # "Optimization launched".
                await request_terminal_message(
                    project_id,
                    f"Optimization stopped — {reason_detail}.",
                )
                break

            convergence = (job.result or {}).get("convergence")

            # Fire a SHORT completion turn — summary only, no delegation.
            # Chat unblocks immediately after this (~10s).
            #
            # EXCEPT for reused results: the resume/adopt skip-eval branch
            # returns the PRIOR iteration's numbers (no new eval ran), and
            # those were already announced when the original iteration
            # recorded them. Re-announcing produced duplicate iteration
            # messages in chat after any mid-loop backend restart.
            if not (job.result or {}).get("reused_prior_results"):
                await request_completion_turn(project_id, job, visibility="visible")

            if convergence:
                logger.info("[opt-loop] Converged: %s after iteration %d", convergence, iteration)
                # Publish a terminal event so the frontend shows a toast.
                # Without this only optimizer_stalled surfaced a toast and
                # plateau/max_iterations/target_reached ended silently.
                _terminal_messages = {
                    "target_reached": "Optimization converged — target metrics reached.",
                    "plateau": "Optimization stopped — pass rate plateaued.",
                    "max_iterations": "Optimization stopped — max iterations reached.",
                }
                _publish_project_event(project_id, {
                    "resource": "optimization",
                    "type": "terminal",
                    "reason": convergence,
                    "message": _terminal_messages.get(
                        convergence, f"Optimization ended: {convergence}"
                    ),
                    "iteration": iteration,
                })
                break

            # Phase transition: the eval is done, the optimizer turn is the
            # long between-evals stretch the status bar used to paper over
            # with an invented "Analyzing failures". Fresh read — the
            # executor advanced the record's version during the iteration,
            # so the handle from the top of this pass is stale for a
            # conditional write. Best-effort: a version race here loses a
            # LABEL, never state, so it must not kill the loop.
            try:
                handle = await opt_repo.read(
                    project_id, project_dir,
                    arch_split=arch_split, owner_id=user_id,
                )
                if handle.is_live() and handle.state.status == "running":
                    handle.state.phase = "optimizing_prompt"
                    handle = opt_repo.write(handle)
            except Exception:  # noqa: BLE001 — a phase label must never stop the loop
                logger.warning(
                    "[opt-loop] Could not write optimizing_prompt phase "
                    "(non-fatal)", exc_info=True,
                )

            # Run prompt-optimizer directly (not via coordinator delegation).
            # This keeps chat interactive while the optimizer analyzes
            # failures and writes the next prompt version.
            optimizer_success = await _run_prompt_optimizer(
                project_id=project_id,
                project_dir=project_dir,
                iteration=iteration,
                prompt_version=prompt_version,
                job_result=job.result or {},
                owner_id=user_id,
            )

            if not optimizer_success:
                logger.warning("[opt-loop] Optimizer did not produce a new prompt")
                handle = await opt_repo.read(
                    project_id, project_dir,
                    arch_split=arch_split, owner_id=user_id,
                )
                opt_state = handle.state
                # Only "running" — a run already stopping/cancelled must not be
                # relabelled as a stall (that would mask a user cancel).
                if opt_state is not None and opt_state.status == "running":
                    opt_state.status = "optimizer_stalled"
                    handle = opt_repo.write(handle)
                    # Publish as "terminal" (not "error") — stalled is an
                    # expected end state, not a failure requiring a red toast.
                    _publish_project_event(project_id, {
                        "resource": "optimization",
                        "type": "terminal",
                        "reason": "optimizer_stalled",
                        "message": "The optimizer couldn't produce a new prompt. Try resuming or starting a new loop.",
                        "iteration": iteration,
                        "prompt_version": prompt_version,
                    })
                    # Chat surface — don't leave chat silent on a stall.
                    await request_terminal_message(
                        project_id,
                        "Optimization stopped — the optimizer couldn't produce a "
                        "new prompt. You can resume or start a new loop.",
                    )
                break

            # Detect the new prompt version (optimizer already saved it)
            new_prompt = detect_latest_prompt(project_dir)
            logger.info("[opt-loop] Prompt advanced: %s → %s", prompt_version, new_prompt)
            prompt_version = new_prompt

    except asyncio.CancelledError:
        logger.info("[opt-loop] Cancelled")
        # Reap the iteration's in-flight child eval before anything else:
        # a raw task-cancel (shutdown / resume race) bypasses the cancel
        # REST endpoint's ``hard_cancel_project_eval``, so without this the
        # child eval would survive as an orphaned active job that
        # ``/activity`` now faithfully reports "running" forever.
        _reap_loop_child_eval(project_id)

        # Persist the terminal status to the control-plane store on the cloud path.
        # Without this, DDB stays "stopping" (as the cancel REST endpoint wrote it)
        # while WS publishes "cancelled" → a browser refresh reads "stopping" forever
        # from /activity (which reads DDB). The version is threaded to satisfy the
        # store's conditional-write guard (expected_version ensures we don't stomp
        # concurrent writes). Mirrors the crash branch's read/mutate/version-thread
        # pattern below.
        # Un-gated from ``arch_split`` for uniformity, NOT as a bug fix: the
        # terminal tail below runs on every exit path and would flip a lingering
        # "stopping" on either home anyway, so the observable end state was already
        # correct. What the gate cost was a publish-then-correct sequence on the EFS
        # path (this branch skipped, tail fixes it later) where the durable path got
        # it right immediately. Both homes now transition here, and the tail stays as
        # the catch-all for exits that never reach this branch at all.
        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.is_live():
            handle.state.status = "cancelled"
            handle = opt_repo.write(handle)

        # Also publish on actual loop exit so the SPA flips from
        # "stopping" → terminal once the orchestrator is gone, not
        # on its next backstop poll. The cancel API publishes first
        # (snappy response to the click); this one closes the loop.
        #
        # Deliberately NO chat message here (unlike the other terminal
        # branches). Cancel is user-initiated: the user clicked
        # Stop, so a "stopped" chat line would be redundant noise, same
        # rationale as the toast-dedup for user cancels. The bar flip is
        # the feedback.
        _publish_project_event(project_id, {
            "resource": "optimization",
            "type": "terminal",
            "reason": "cancelled",
            "message": "Optimization cancelled.",
        })
    except Exception as e:
        logger.error("[opt-loop] Unexpected error: %s", e, exc_info=True)
        # Same invariant as the cancel branch: don't let an iteration's
        # child eval outlive the crashed loop as an orphaned active job.
        _reap_loop_child_eval(project_id)
        # Reporting the crash must not itself crash. The store read below and the
        # versioned write further down can both fail — and when the cause of the
        # crash IS the store, they reliably do. Raising from in here replaced the
        # real diagnosis with a teardown error and pushed the failure out of the
        # loop task entirely.
        try:
            await _report_loop_crash(
                e,
                project_id=project_id,
                project_dir=project_dir,
                arch_split=arch_split,
                user_id=user_id,
            )
        except Exception:  # noqa: BLE001 — the original error is the useful one
            logger.exception(
                "[opt-loop] Could not record the crash for project=%s",
                project_id[:12],
            )
    finally:
        # Everything below has to run on EVERY exit path, so it lives in a
        # ``finally`` rather than trailing the try. It used to sit after the
        # except blocks with a comment claiming it "runs after EVERY exit path
        # … because it sits outside the try/except above" — which is the
        # opposite of what that position means. A raise inside either handler,
        # or in the state repair below, skipped straight past it and leaked the
        # run's eval lock until its 24h TTL, blocking every later eval on the
        # project. Both handlers do a store read and a versioned write, and the
        # write documents that it raises on a concurrent update.
        final_status = "unknown"

        # Ensure terminal status — catch-all safety net for ANY exit path that
        # left the status non-terminal. A FRESH read on purpose: the branch
        # above may already have written "cancelled", in which case this
        # observes it and no-ops, so the two writes never fight over the same
        # version. Best-effort, and explicitly so: the lock release below is the
        # mandatory part, and a failed repair must not prevent it.
        try:
            final_handle = await opt_repo.read(
                project_id, project_dir, arch_split=arch_split, owner_id=user_id
            )
            if final_handle.is_live():
                final_handle.state.status = "cancelled"
                opt_repo.write(final_handle)
            final_status = final_handle.status or "unknown"
        except Exception:  # noqa: BLE001 — must not skip the release below
            logger.exception(
                "[opt-loop] Could not settle terminal status for project=%s",
                project_id[:12],
            )

        # Release the run's eval lock. dispatch_optimization claims the lock
        # before its invocability preflight so that window cannot admit a
        # concurrent manual eval, and hands ownership to this loop task.
        # Holder-conditional and idempotent: releasing a lock this run no longer
        # holds (e.g. the reconciler already cleared it) is a no-op.
        lock_owner_id = iteration_config.get("eval_lock_owner_id")
        lock_holder_id = iteration_config.get("eval_lock_holder_id")
        if lock_owner_id and lock_holder_id:
            try:
                from app.platform import get_platform
                await get_platform().metadata_store.release_eval_lock(
                    lock_owner_id, project_id, lock_holder_id
                )
                logger.info(
                    "[opt-loop] Released run eval lock for project=%s", project_id[:12],
                )
            except Exception:  # noqa: BLE001 — never mask the loop's terminal state
                logger.exception(
                    "[opt-loop] Failed to release run eval lock for project=%s — the "
                    "reconciler's stale-lock sweep is the backstop",
                    project_id[:12],
                )

        logger.info("[opt-loop] Finished: status=%s", final_status)


async def _report_loop_crash(
    error: Exception,
    *,
    project_id: str,
    project_dir,
    arch_split: bool,
    user_id: str | None,
) -> None:
    """Record an unexpected loop failure on the state, the bus and in chat.

    Extracted from the crash handler so its caller can contain a failure here:
    when the store is what broke, this is exactly the code that breaks again.

    Args:
        error: The original failure. Its message is what the user sees.
        project_id: Project whose loop crashed.
        project_dir: Project directory, for the state read.
        arch_split: Whether the durable path is in use.
        user_id: Owner, for the durable read.
    """
    # Imported here for the same reason the loop does it: this module is pulled
    # in during startup and these pull the job/completion graph behind them.
    from app.services import optimization_state_repository as opt_repo
    from app.services.completion_turn import request_terminal_message
    from app.services.resource_error import Codes, ResourceError

    handle = await opt_repo.read(
        project_id, project_dir, arch_split=arch_split, owner_id=user_id
    )
    opt_state = handle.state
    # ``error is None`` matters: the FIRST recorded cause is the useful one.
    # Overwriting it with a downstream crash symptom loses the diagnosis.
    if opt_state is None or opt_state.error is not None or opt_state.status != "running":
        return

    opt_state.error = ResourceError(
        code=Codes.SYSTEM_TASK_CRASHED,
        message=f"Optimization loop crashed: {error}",
        details={"exception": type(error).__name__},
    )
    opt_state.status = "cancelled"
    opt_repo.write(handle)
    # ``error`` envelope feeds the toast layer
    # (useBackendEvents.ts handles type=="error" for toast).
    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "error",
        **opt_state.error.to_dict(),
    })
    # ``terminal`` envelope feeds OptimizationStateProvider so the bar refetches
    # without waiting for its backstop poll. Two subscribers, two envelopes, same
    # source event.
    _publish_project_event(project_id, {
        "resource": "optimization",
        "type": "terminal",
        "reason": "crashed",
        "message": f"Optimization loop crashed: {error}",
    })
    # Chat surface — a crash must say something in chat, not just flash a toast.
    await request_terminal_message(
        project_id,
        f"Optimization stopped — the loop hit an unexpected error: {error}.",
    )


async def _run_prompt_optimizer(
    project_id: str,
    project_dir,
    iteration: int,
    prompt_version: str,
    job_result: dict,
    owner_id: str | None = None,
) -> bool:
    """Run the prompt-optimizer subagent directly from the optimization loop.

    Invokes the prompt-optimizer via the Claude Agent SDK as a subagent
    call, providing iteration results and project context. The subagent
    analyzes failures and calls save_prompt to write the next version.

    This runs outside the coordinator turn, so chat stays interactive
    while the optimizer works — which also means no request context exists
    here. Three of the optimizer's four tools are MCP tools
    (``analyze_failures``, ``get_eval_results``, ``save_prompt``) and
    ``mcp_authz`` fails CLOSED without a context, so all three would be
    refused and the optimizer could not write a prompt at all. Only ``Read``
    is a built-in and therefore ungated. Hence the owner attribution below.

    Args:
        project_id: Project UUID.
        project_dir: Path to the project directory.
        iteration: Current iteration number.
        prompt_version: Current prompt version that was evaluated.
        job_result: Iteration result data (metrics, pass rates, run file).
        owner_id: Project owner, used to attribute this detached run for MCP
            authz. ``None`` leaves the context unset, which means every MCP
            tool the optimizer holds will be refused.

    Returns:
        True if a new prompt was saved, False if the optimizer stalled.
    """
    from pathlib import Path
    from app.services.optimization_orchestrator import detect_latest_prompt

    project_dir = Path(project_dir) if not isinstance(project_dir, Path) else project_dir

    # Build the optimizer prompt with iteration context
    passed = job_result.get("passed_cases", 0)
    total = job_result.get("total_cases", 0)
    rate = f"{passed}/{total} ({passed/total*100:.1f}%)" if total else "0/0"
    metrics = job_result.get("metrics_summary", {})
    run_file = job_result.get("run_output_file", "")
    per_model = job_result.get("per_model_pass_rates", {})

    model_lines = "\n".join(f"  {m}: {r*100:.1f}%" for m, r in per_model.items())
    metric_lines = "\n".join(f"  {m}: {v*100:.1f}%" for m, v in metrics.items())

    # Next version name from the current prompt, not the loop counter
    # (see next_prompt_version — the counter-based formula skipped v2).
    from app.services.optimization_orchestrator import next_prompt_version
    next_version = next_prompt_version(prompt_version)

    optimizer_prompt = (
        f"Analyze iteration {iteration + 1} and write an improved prompt "
        f"({next_version}) for this project.\n\n"
        f"project_id: {project_id}\n"
        f"Use EXACTLY this project_id for every tool call (analyze_failures, "
        f"save_prompt, etc.). Never invent, guess, or derive a project_id "
        f"from the task domain — copy the value above verbatim.\n\n"
        f"Current prompt: {prompt_version}\n"
        f"Pass rate: {rate}\n"
        f"Per-model pass rates:\n{model_lines}\n"
        f"Per-metric pass rates:\n{metric_lines}\n"
        f"Run file: {run_file}\n\n"
        f"To understand this run, FIRST call analyze_failures with the run "
        f"file — it returns the failure breakdown by category/metric, "
        f"passing-case exemplars (the template for what works), and a "
        f"regression diff showing any cases your last change broke "
        f"(pass→fail). For full ground truth on specific cases, Read the "
        f"raw run file directly. Then make the smallest change that "
        f"addresses the dominant failure pattern while preserving what's "
        f"passing, and call save_prompt with the improved prompt."
    )

    logger.info(
        "[opt-loop] Running prompt-optimizer for iteration %d (prompt=%s, rate=%s)",
        iteration, prompt_version, rate,
    )

    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
        from app.services.mcp_server_cache import get_eval_mcp_server
        from app.services.project_manager import _is_cloud_mode
        from app.services.sdk_subprocess_env import (
            SDK_MAX_STDOUT_FRAME_BYTES,
            sdk_subprocess_env,
        )
        from app.services.sdk_tool_hooks import (
            TOOL_OUTCOME_HOOK_EVENTS,
            ToolCallOutcome,
            make_tool_outcome_hook,
        )

        # Inherit the project's agent model (set via chat model selector)
        from app.main import get_project_manager
        pm = get_project_manager()
        project = pm.get_project(project_id)
        model = project.agent_model if project else "opus"

        # Read the prompt-optimizer system prompt
        optimizer_system = (PROMPTS_DIR / "prompt-optimizer.md").read_text()

        # Track save_prompt's outcome. Diagnostic only — the verdict below is
        # the filesystem. See ``sdk_tool_hooks`` for why reading success out of
        # the payload's shape did not work.
        save_outcome = ToolCallOutcome()
        _save_hook = make_tool_outcome_hook(save_outcome)

        opts = ClaudeAgentOptions(
            # ``allowed_tools`` is an AUTO-APPROVE list, NOT a gate. Listing only
            # Read + the eval MCP tools here does not remove anything: combined
            # with ``permission_mode="bypassPermissions"`` below, Write/Edit/Bash
            # would stay fully usable, and the optimizer could write
            # ``prompts/<version>.md`` directly, bypassing every guard in
            # ``save_prompt`` (dedup, degenerate-state, empty-content,
            # version-name).
            allowed_tools=["Read", "mcp__eval__*"],
            # ``disallowed_tools`` is the REMOVAL list — this is the gate.
            #
            # Chosen over ``tools=[...]`` (an allowlist gate that maps to
            # ``--tools a,b,c``) deliberately: an allowlist would have to name the
            # MCP tools too, and if the CLI treats ``--tools`` as excluding
            # anything unlisted, a wrong entry silently strips the optimizer's own
            # MCP tools and the loop stalls. Removal touches only the paths we mean
            # to close and leaves MCP resolution untouched.
            #
            # With no direct write path, a ``save_prompt`` refusal STALLS the
            # iteration ("Optimizer did not produce a new prompt") instead of
            # being silently routed around. That
            # is the guards working — a dedup refusal means the optimizer produced
            # the same prompt again, and burning a full judge-cost iteration on a
            # duplicate is worse than stopping.
            #
            # This depends on ``save_prompt`` clearing authz from this detached
            # run (the context attribution below): with that tool refused and
            # the direct write path removed, the optimizer would have no way to
            # save a prompt at all.
            disallowed_tools=["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"],
            permission_mode="bypassPermissions",
            cwd=str(project_dir),
            # Disabled for the same reasons as ``agent_runner.py``: bwrap
            # is incompatible with Fargate (no CAP_SYS_ADMIN, unprivileged
            # user namespaces are off) and redundant given Firecracker's
            # microVM boundary. Setting this to True hangs the bundled
            # CLI's ``initialize()`` on the bwrap preflight. Tenant isolation
            # is enforced at the app layer instead.
            sandbox={"enabled": False},
            # Mirror transcript frames to S3 as they arrive rather than
            # at turn end — same reason as ``agent_runner.py``: a
            # mid-turn page refresh must surface in-flight messages.
            session_store_flush="eager",
            system_prompt=optimizer_system,
            max_turns=50,
            effort="high",
            mcp_servers={"eval": get_eval_mcp_server()},
            # Same filesystem-settings isolation as the coordinator turn
            # (see ``agent_runner.py`` for the full rationale): dev loads
            # project scope only, because a developer's ``~/.claude``
            # settings may set ``CLAUDE_CODE_USE_MANTLE=1``, which stops
            # the bundled CLI's ``initialize`` from completing. Cloud
            # also loads user scope (the runtime image's ``~/.claude``
            # is platform-controlled and clean). If unset the SDK
            # default loads all scopes and the dev optimizer run hangs.
            setting_sources=["user", "project"] if _is_cloud_mode() else ["project"],
            # SDK subprocess env pins: see sdk_subprocess_env.py for rationale.
            env=sdk_subprocess_env(),
            # Bounds ONE stdout frame. This optimizer reads run files through
            # MCP tools, which is the other producer of near-1-MiB frames; see
            # SDK_MAX_STDOUT_FRAME_BYTES.
            max_buffer_size=SDK_MAX_STDOUT_FRAME_BYTES,
            hooks={
                event: [
                    HookMatcher(
                        matcher="mcp__eval__save_prompt",
                        hooks=[_save_hook],
                    ),
                ]
                for event in TOOL_OUTCOME_HOOK_EVENTS
            },
        )
        if model:
            opts.model = model

        # Plug platform-managed SessionStore so cloud mode persists the
        # optimizer's session to S3. Dev mode keeps the SDK default.
        from app.platform import get_platform
        _ss = get_platform().session_store
        if _ss is not None:
            opts.session_store = _ss

        # Give this optimizer run a context its MCP tools can clear authz with.
        # Attribution to the owner happens only when no context exists; a run
        # started from the API inherits the caller's identity and keeps it, but
        # still needs binding to this project, because the HTTP middleware
        # leaves project_id unset and the authz pin compares against it. Must
        # precede ClaudeSDKClient construction — the client's transport tasks
        # inherit the ContextVar as it stands when they are created.
        from app.platform.types import attribute_detached_context, request_context
        ctx_token = attribute_detached_context(
            owner_id=owner_id,
            project_id=project_id,
            request_id=f"detached-optimizer-{project_id[:8]}",
        )
        try:
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(optimizer_prompt)
                async for message in client.receive_response():
                    pass
        finally:
            if ctx_token is not None:
                request_context.reset(ctx_token)

        # Outcome is the filesystem: a new prompt version exists or it doesn't.
        # The hook state is diagnostic, not the verdict — and now that Write and
        # Edit are removed from this session, ``save_prompt`` is the only way a
        # version file can appear. A new version with a hook that never fired
        # therefore means the HOOK is broken, not that the agent wrote around the
        # tool; log the two apart so that reads as the defect it would be.
        new_version = detect_latest_prompt(project_dir)
        if new_version != prompt_version:
            if save_outcome.succeeded:
                logger.info(
                    "[opt-loop] Optimizer saved new prompt: %s → %s",
                    prompt_version, new_version,
                )
            else:
                logger.warning(
                    "[opt-loop] New prompt %s → %s appeared but the save_prompt "
                    "hooks did not report success — the hook wiring is suspect; "
                    "the prompt itself passed the tool's guards. Hooks saw: %s",
                    prompt_version, new_version, save_outcome.describe(),
                )
            return True

        logger.warning(
            "[opt-loop] Optimizer finished without producing a new prompt "
            "(iteration %d, current=%s, attempted=%s, succeeded=%s). Hooks saw: %s",
            iteration, prompt_version,
            save_outcome.attempted, save_outcome.succeeded, save_outcome.describe(),
        )
        return False

    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("[opt-loop] Prompt optimizer timed out (iteration %d)", iteration)
        return False
    except Exception as e:
        logger.error("[opt-loop] Prompt optimizer failed: %s", e, exc_info=True)
        return False



# ---------------------------------------------------------------------------
# Dispatch-to-owner seam: coordinator adoption of orphaned loops
# ---------------------------------------------------------------------------

def _loop_task_name(project_id: str) -> str:
    """Task name for a project's loop — the registry both spawn paths and the
    cancel endpoint scan. One definition so adoption, dispatch, and cancel can
    never disagree on what "a loop task exists" means."""
    return f"optimization-{project_id[:8]}"


def _local_loop_task_exists(project_id: str) -> bool:
    """Whether this process is already driving a loop for the project."""
    name = _loop_task_name(project_id)
    return any(t.get_name() == name for t in asyncio.all_tasks())


async def _iter_live_optimization_records() -> list[dict[str, Any]]:
    """Enumerate live OPT records: walk project dirs, point-read each record.

    Why not a store listing: the OPT item carries no GSI keys, so "list every
    live record" would need a table-wide Scan — a permission the coordinator
    role deliberately does NOT have (the IAM stack grants item-level CRUD and
    queries, no Scan, so a Scan fails with AccessDeniedException). The
    project working trees on EFS are the enumeration index; the record itself
    is still read from its owner, the store, one point-read per project.
    Owner comes from the cloud path layout (``users/{owner}/projects/{pid}``);
    flat dev-layout dirs resolve to the local dev user.
    """
    from app.main import _iter_project_dirs  # lazy: app.main imports many services
    from app.platform import get_platform
    from app.platform.auth import get_local_dev_user_id

    store = get_platform().metadata_store
    records: list[dict[str, Any]] = []
    for project_dir in _iter_project_dirs():
        project_id = project_dir.name
        parts = project_dir.parts
        try:
            owner_id = parts[parts.index("users") + 1]
        except ValueError:
            owner_id = get_local_dev_user_id()
        try:
            state = await store.get_optimization_state(owner_id, project_id)
        except Exception:  # noqa: BLE001 — one unreadable record must not kill the sweep
            logger.exception(
                "[opt-adopt] Could not read opt-state for project=%s", project_id[:12]
            )
            continue
        if not state or state.get("status") not in ("running", "stopping"):
            continue
        records.append({
            "user_id": owner_id, "project_id": project_id, "state": state,
        })
    return records


async def adopt_orphan_optimization_loops() -> int:
    """Spawn a loop for every live optimization record no process is driving.

    THE dispatch-to-owner seam. A ``running``/``stopping`` OPT record is a
    durable declaration that a loop task should exist; the record itself is
    the queue. Two producers write such records without being able to own the
    loop:

    * ``dispatch_optimization`` executing in the AgentCore microVM — it writes
      state + claims the eval lock, then deliberately does NOT spawn (the
      microVM dies with the turn; see the execution-home guard in
      :func:`run_optimization_loop`).
    * A coordinator that crashed or was replaced mid-loop — its asyncio task
      died with the process, stranding the record live.

    This sweep runs ONLY on the coordinator (the process that owns job
    lifecycle side effects) at startup and on the reconciler cadence, and
    reconstructs the loop config from the facts' owners rather than from a
    stashed copy: targets/budget/plateau from the record, models/metrics/judge
    from harness config, prompt from the record or the prompts dir, lock
    identity deterministically. ``max_test_cases`` is per-dispatch ephemera
    and does not survive adoption — an adopted loop evaluates the harness's
    configured dataset.

    Deliberately DEFERRED (not adopted) while the project still has a live
    optimization-child job: after a coordinator restart the old child eval may
    still be running in its detached worker, and spawning a new loop now would
    double-run that iteration. The child reaching a terminal state unblocks
    adoption on the next sweep.

    Returns:
        Number of loops adopted this pass.
    """
    from app.platform import get_platform
    from app.platform.process_role import is_agentcore_runtime

    if is_agentcore_runtime():
        return 0

    store = get_platform().metadata_store
    records = await _iter_live_optimization_records()
    if not records:
        return 0

    # One listing for all candidates — the child-job defer check below reads it.
    try:
        non_terminal_jobs = await store.list_non_terminal_jobs()
    except Exception:  # noqa: BLE001 — defer adoption rather than guess
        logger.exception("[opt-adopt] Could not list non-terminal jobs; deferring pass")
        return 0

    adopted = 0
    for rec in records:
        project_id = rec["project_id"]
        user_id = rec["user_id"]
        state = rec["state"]

        if _local_loop_task_exists(project_id):
            continue  # a loop is already driving this record (normal case)

        live_children = [
            j for j in non_terminal_jobs
            if j.get("project_id") == project_id
            and (
                j.get("type") == "optimization_iteration"
                or (j.get("config") or {}).get("source") == "optimization"
            )
        ]
        if live_children:
            logger.info(
                "[opt-adopt] Deferring adoption for project=%s — %d live "
                "optimization-child job(s) still running (adopting now would "
                "double-run the iteration). Next sweep retries.",
                project_id[:12], len(live_children),
            )
            continue

        try:
            adopted += await _adopt_one(project_id, user_id, state)
        except Exception:  # noqa: BLE001 — one bad record must not block the rest
            logger.exception(
                "[opt-adopt] Adoption failed for project=%s", project_id[:12]
            )
    return adopted


async def _adopt_one(project_id: str, user_id: str, state: dict[str, Any]) -> int:
    """Reconstruct config from the owners and spawn the loop for one record.

    Returns 1 on spawn, 0 when the record was flipped to failed instead
    (unadoptable: harness config gone). On the unadoptable path the eval lock
    is released too — the loop that would have released it will never run.
    """
    from app.platform import get_platform
    from app.platform.arch_split import resolve_arch_split
    from app.services import optimization_state_repository as opt_repo
    from app.services.harness_config import load_harness_config
    from app.services.optimization_orchestrator import detect_latest_prompt
    from app.services.project_paths import resolve_project_dir
    from app.services.resource_error import Codes, ResourceError
    from app.services.task_error_sink import attach_error_sink

    project_dir = resolve_project_dir(project_id)
    lock_holder_id = f"optimization-{project_id}"
    try:
        harness = load_harness_config(project_dir) if project_dir.exists() else None
    except (SyntaxError, ValueError) as exc:
        # An unreadable config.py cannot drive a loop either; take the same
        # unadoptable path as a missing one instead of failing the adoption.
        logger.warning(
            "[opt-adopt] src/config.py unreadable for project=%s: %s",
            project_id[:8], exc,
        )
        harness = None

    if harness is None or not harness.models:
        # Unadoptable: the record demands a loop the project can no longer
        # support (config deleted, unreadable or edited since dispatch).
        # Metrics are not checked: they live as code in
        # src/metrics/metrics.py, which the harness's build_metrics owns.
        # Flip it to failed (through the repository, version-fenced)
        # and release the lock the dispatch claimed, otherwise the record
        # re-logs every sweep and the lock leaks forever.
        logger.error(
            "[opt-adopt] Record for project=%s is unadoptable (harness config "
            "missing or empty) — flipping to failed and releasing its lock.",
            project_id[:12],
        )
        arch_split = await resolve_arch_split(project_id)
        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.state is not None and handle.is_live():
            handle.state.status = "failed"
            handle.state.error = ResourceError(
                code=Codes.OPTIMIZATION_DISPATCH_FAILED,
                message=(
                    "Optimization could not be adopted by the coordinator: "
                    "harness config is missing or has no models."
                ),
            )
            opt_repo.write(handle)
        await get_platform().metadata_store.release_eval_lock(
            user_id, project_id, lock_holder_id
        )
        _publish_project_event(project_id, {
            "resource": "optimization",
            "type": "error",
            "code": "optimization.unadoptable",
            "message": "Optimization could not start: harness config is missing.",
        })
        return 0

    arch_split = await resolve_arch_split(project_id)
    prompt_version = state.get("current_prompt_version") or detect_latest_prompt(project_dir)

    iteration_config = {
        "project_id": project_id,
        "user_id": user_id,
        "arch_split": arch_split,
        "project_dir": str(project_dir),
        "model_ids": harness.models,
        "data_file": harness.data_file or "",
        "field_mappings": harness.field_mappings or {},
        "judge_model_id": harness.judge_model_id,
        "target_metric_scores": state.get("target_metric_scores") or {},
        # Strict reads: repo.create has always persisted both fields on the
        # durable record, so a default here is dead code at best and an
        # invented parameter at worst. A malformed record fails loudly.
        "max_iterations": state["max_iterations"],
        "plateau_window": state["plateau_window"],
        "max_test_cases": None,  # per-dispatch ephemera; not durable
        "data_file_override": None,
        "prompt_version": prompt_version,
        "eval_lock_owner_id": user_id,
        "eval_lock_holder_id": lock_holder_id,
    }

    attach_error_sink(
        asyncio.create_task(
            run_optimization_loop(iteration_config),
            name=_loop_task_name(project_id),
        ),
        project_id=project_id,
        resource="optimization",
    )
    logger.info(
        "[opt-adopt] Adopted orphaned optimization loop: project=%s prompt=%s "
        "status=%s iterations_so_far=%d",
        project_id[:12], prompt_version, state.get("status"),
        len(state.get("iterations") or []),
    )
    return 1
