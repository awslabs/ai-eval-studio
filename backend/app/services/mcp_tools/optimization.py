# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool for optimization lifecycle: ``record_optimization_iteration``.

Optimization dispatch lives in ``dispatch.py``'s ``dispatch_optimization``
tool; this module exposes the per-iteration record/convergence helper
the coordinator calls from inside the loop.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import tool

from app.services.mcp_response import mcp_error, mcp_json
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


@tool(
    "record_optimization_iteration",
    "Record a completed optimization iteration and check convergence. "
    "Call this after each eval run in the optimization loop. Scores are read "
    "from the named run file — you do not supply them. Returns whether to "
    "continue, stop (converged/plateau/max), and the best iteration so far.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "prompt_version": {"type": "string", "description": "Prompt version used in this iteration"},
            "run_id": {"type": "string", "description": "Eval run ID"},
        },
        "required": ["project_id", "prompt_version", "run_id"],
    },
)
async def record_optimization_iteration(args: dict[str, Any]) -> dict[str, Any]:
    """Record an iteration and check if the optimization loop should stop.

    Updates OptimizationState with iteration results and checks convergence
    conditions (target reached, max iterations, plateau). Returns actionable
    guidance to the agent on whether to continue or stop.

    **Durable-state path:** On the new path (arch_split=True), this
    tool performs a funnel-discipline-aware write: it reads the current state
    + version via get_optimization_state, applies the record_iteration mutation,
    then writes via put_optimization_state WITH expected_version. This satisfies
    the single-writer discipline (one code path + version seatbelt). On
    ConcurrentUpdateError from the conditional write, surface via mcp_error so
    the agent can retry/reconcile rather than crashing the loop.

    Args:
        args: Dict containing project_id, prompt_version, and run_id. Scores
              are read from the run file, never taken from the caller.

    Returns:
        Dict with iteration status, convergence check, and best iteration info
    """
    try:
        from app.platform.metadata_store import ConcurrentUpdateError
        from app.platform.types import current_request
        from app.platform.auth import get_local_dev_user_id

        project_id = args["project_id"]
        project_dir = resolve_project_dir(project_id)

        # Resolve arch_split from the MCP call context. The coordinator's
        # dispatch_optimization turn already resolved it once and threaded it
        # into the iteration jobs, but this MCP tool is agent-invoked and runs
        # SYNCHRONOUSLY in the agent turn (not inside the executor), so it
        # re-resolves here — correct, because the tool call is its own
        # operation entry point, not mid-operation.
        from app.platform.arch_split import resolve_arch_split
        arch_split = await resolve_arch_split(project_id)

        try:
            user_id = current_request().user_id
        except RuntimeError:
            user_id = get_local_dev_user_id()

        from app.services import optimization_state_repository as opt_repo

        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split, owner_id=user_id
        )
        if handle.state is None:
            return mcp_error("No optimization in progress. Call dispatch_optimization first.")
        state = handle.state

        # Refuse to double-record the same prompt_version. Duplicate rows
        # skew plateau/window math (same pass rate counted twice looks
        # like a stable plateau) and advance ``current_iteration`` past
        # the real iteration count.
        if state.iterations and state.iterations[-1].get("prompt_version") == args["prompt_version"]:
            prior = state.iterations[-1]
            return mcp_error(
                f"Iteration for prompt_version={args['prompt_version']!r} is already recorded "
                f"(iteration {prior.get('iteration')}, pass_rate={prior.get('pass_rate'):.2%}). "
                "Skip the re-record and proceed with the next optimization step."
            )

        # Sources-of-truth: the run FILE is the results owner. The row is
        # DERIVED from the named run file via the same
        # extract_iteration_metrics the iteration executor and the report
        # builder use (one implementation, so the callers can't drift). The
        # tool takes no scores as arguments: an agent-typed pass rate is a
        # claim, not a measurement, and rows written from claims carried no
        # per-model data — which is what kept aggregate fallbacks alive in
        # convergence/plateau and made "converged" mean two different things
        # depending on the writer.
        from app.services.eval_runner import EvalRunner
        from app.services.jobs.iteration_metrics import extract_iteration_metrics

        results_path = project_dir / "output" / args["run_id"]
        if not results_path.exists():
            return mcp_error(
                f"Run not found: {args['run_id']}. An iteration can only be "
                "recorded against an existing run file — the file is the "
                "results record, not the numbers in this call."
            )
        results = EvalRunner().load_results(results_path)
        derived = extract_iteration_metrics(results)

        # Mutate state (record_iteration appends to iterations list and calls save)
        # On the new path, record_iteration will call save with arch_split params
        # Both writes chain versions through the handle — record_iteration returns
        # a version-advanced handle that check_convergence then consumes.
        handle = opt_repo.record_iteration(
            handle,
            prompt_version=args["prompt_version"],
            run_id=args["run_id"],
            metrics_summary=derived["metrics_summary"],
            total_cases=derived["total_cases"],
            passed=derived["passed_cases"],
            per_model_pass_rates=derived["per_model_pass_rates"],
            per_model_metrics=derived["per_model_metrics"],
        )
        convergence, handle = opt_repo.check_convergence(handle)
        best = state.best_iteration()

        result = {
            "iteration": state.current_iteration,
            "convergence": convergence,
            "status": state.status,
            "best_iteration": best,
            "current_pass_rate": (
                derived["passed_cases"] / derived["total_cases"]
                if derived["total_cases"] > 0
                else 0
            ),
        }

        if convergence and best:
            best_rate = best.get('pass_rate', best['passed'] / best['total_cases'] if best.get('total_cases') else 0)
            result["message"] = f"Optimization stopped: {convergence}. Best prompt: {best['prompt_version']} ({best_rate:.1%})"
        elif convergence:
            # ``best_iteration`` returns None when no row carries per-model data.
            # Unreachable while convergence itself demands that data — but the
            # coupling is implicit, and the old code would have raised
            # AttributeError here rather than saying anything useful.
            result["message"] = (
                f"Optimization stopped: {convergence}. No iteration carries "
                "per-model results, so no best prompt can be named."
            )
        else:
            result["message"] = f"Iteration {state.current_iteration}/{state.max_iterations}. Continue optimizing."

        return mcp_json(result)

    except ConcurrentUpdateError as e:
        # Stale expected_version: surface via mcp_error so the agent reconciles
        # rather than crashing the loop (loud seatbelt)
        return mcp_error(
            f"OptimizationState version conflict: {e}. Another writer updated "
            "the state concurrently. The agent should retry the record with "
            "fresh state."
        )
    except Exception as e:
        return mcp_error(f"Error: {e}")
