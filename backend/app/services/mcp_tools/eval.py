# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for evaluation lifecycle: check_eval_status, list_eval_runs,
get_eval_results, get_case_details, and cancel_eval.

Eval dispatch lives in ``dispatch.py``'s ``dispatch_eval`` tool; this
module exposes the read-side and cancellation helpers the coordinator
needs while a job is running.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.platform.types import current_request
from app.services.jobs import _publish_project_event, get_job_tracker
from app.services.eval_runner import EvalRunner, _derived_dir
from app.services.mcp_response import mcp_error, mcp_json, mcp_success
from app.services.project_paths import resolve_project_dir, safe_basename_path, safe_path

logger = logging.getLogger(__name__)


def _build_results_summary(model_runs: list[dict]) -> str:
    """Build a human-readable summary of eval results for the agent.

    Generates per-model pass counts and per-metric pass rates from the
    completed model_runs data.

    Args:
        model_runs: List of per-model run result dicts from EvalRunner.

    Returns:
        Multi-line summary string with pass rates and metric breakdowns.
    """
    lines = []
    for mr in model_runs:
        results = mr.get("results", [])
        passed = sum(1 for r in results if r.get("success", False))
        total = len(results)
        lines.append(f"  {mr['test_run_id']}: {passed}/{total} passed")
        # Per-metric pass rates
        metric_names: set[str] = set()
        for r in results:
            for md in r.get("metrics_data", []):
                metric_names.add(md["name"])
        for mname in sorted(metric_names):
            mpassed = sum(
                1 for r in results
                for md in r.get("metrics_data", [])
                if md["name"] == mname and md.get("success", False)
            )
            rate = mpassed / total if total > 0 else 0
            lines.append(f"    {mname}: {rate*100:.1f}%")
    return "\n".join(lines)


@tool(
    "check_eval_status",
    "Check status of a running eval job. Returns progress (model/case counts) while running, "
    "or full results summary when complete.",
    {"project_id": str, "job_id": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def check_eval_status(args: dict[str, Any]) -> dict[str, Any]:
    """Poll a background eval job for progress or final results.

    Returns structured progress info while the job is running, a full results
    summary when complete, or an error message if the job failed.

    On the durable path (arch_split=True), rebuilds jobs from the store before
    reading the tracker so the agent sees durable jobs even when the in-memory
    tracker is empty (cross-process/microVM case), and the durable store wins on
    terminal state (refresh_terminal=True). On the old path (arch_split=False),
    rebuild is a no-op.

    Args:
        args: Dict with ``project_id`` and ``job_id``.

    Returns:
        Dict with text content describing current job state.
    """
    from app.platform.arch_split import resolve_arch_split

    tracker = get_job_tracker()

    # Resolve arch_split once and rebuild jobs from durable store before reading
    try:
        arch_split = await resolve_arch_split(args["project_id"])
        await tracker.rebuild_jobs_from_store(args["project_id"], arch_split, refresh_terminal=True)
    except Exception as e:
        # Store errors degrade to old-path behavior rather than raising into agent loop
        logger.warning(
            "[check_eval_status] rebuild failed for project=%s: %s",
            args["project_id"][:12], e
        )

    job = tracker.get_job(args["job_id"])
    if not job:
        return mcp_error(f"Job not found: {args['job_id']}")

    if job.status in ("pending", "running"):
        # Simple lookup — no server-side throttle needed since the system prompt
        # tells the agent not to poll in a loop. The dashboard shows live progress.
        progress = job.progress
        return mcp_success(
            f"Status: {job.status}\n"
            f"Models: {progress.get('models_completed', 0)}/{progress.get('models_total', 0)}\n"
            f"Current: {progress.get('message', 'Waiting...')}"
        )
    elif job.status == "completed":
        # Build a full results summary from the stored model runs
        summary = _build_results_summary(job.result) if job.result else "No results data."
        return mcp_success(f"Eval complete. Saved to: {job.output_file}\n\n{summary}")
    elif job.status == "failed":
        err = job.error
        detail = err.message if err else "unknown error"
        if err and err.hint:
            detail = f"{detail} {err.hint}"
        return mcp_error(f"Eval failed: {detail}")
    elif job.status == "cancelled":
        return mcp_success("Eval job was cancelled by the user.")

    # Shouldn't happen, but defensive fallback
    return mcp_error(f"Unknown job status: {job.status}")


@tool(
    "list_runs",
    "List all completed evaluation runs for a project. Returns run IDs, timestamps, and summary metrics.",
    {"project_id": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_eval_runs(args: dict[str, Any]) -> dict[str, Any]:
    """List all completed evaluation runs for a project.

    Reads run summaries from the project's output directory, sorted newest first.

    Args:
        args: Dict with ``project_id``.

    Returns:
        Dict with text listing runs and their pass/fail counts.
    """
    logger.info("[list_eval_runs] project=%s", args.get("project_id", "?")[:12])
    try:
        project_dir = resolve_project_dir(args["project_id"])
        runner = EvalRunner()
        runs = runner.list_runs(project_dir)
        logger.info("[list_eval_runs] Found %d runs", len(runs))

        if not runs:
            return mcp_success("No evaluation runs found for this project.")

        summaries = []
        for run in runs:
            model_parts = []
            for mr in run.get("model_runs", []):
                model_parts.append(f"{mr['test_run_id']} ({mr['passed']}/{mr['total']})")
            summaries.append(
                f"{run.get('_file', '')} ({run.get('timestamp', '')}): "
                f"prompt={run.get('prompt_version', '')}, "
                f"models: {', '.join(model_parts)}"
            )
        return mcp_success("\n".join(summaries))

    except Exception as e:
        return mcp_error(f"Error: {e}")


@tool(
    "get_results",
    "Summarize a specific eval run: per-model pass rates, per-metric scores, pass/fail counts, "
    "and iteration metadata. Does NOT return per-case details — use `analyze_failures` for "
    "failure drilldown or `get_case_details(case_indices=[...])` for specific cases.",
    {"project_id": str, "run_filename": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_eval_results(args: dict[str, Any]) -> dict[str, Any]:
    """Load a lightweight summary of a specific eval run.

    Returns per-model pass rates and per-metric scores without any per-case
    payload. Full per-case data would routinely exceed the SDK transport's
    1 MiB cap (a run file carrying judge rationales is typically several MB).

    Args:
        args: Dict with ``project_id`` and ``run_filename``.

    Returns:
        Dict with per-model summaries, per-metric rates, and metadata.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        output_dir = project_dir / "output"
        results_path = safe_path(output_dir, args["run_filename"])
        if results_path is None:
            return mcp_error(f"Invalid filename: {args['run_filename']}")
        if not results_path.exists():
            return mcp_error(f"Run not found: {args['run_filename']}")

        runner = EvalRunner()
        data = runner.load_results(results_path)
        summary = _summarize_results(data, args["run_filename"])
        return mcp_json(summary)

    except Exception as e:
        return mcp_error(f"Error: {e}")


def _summarize_results(data: Any, run_filename: str) -> dict[str, Any]:
    """Build a bounded-size summary from a raw eval-results JSON payload.

    The raw payload is a list of per-model run dicts; each contains a
    ``results`` array with one entry per test case, each entry carrying
    the full judge rationale. We collapse to per-model and per-metric
    pass rates so the total size stays well under the SDK cap.

    Args:
        data: Parsed JSON from the results file (list of model-run dicts).
        run_filename: The run's filename, echoed back for context.

    Returns:
        Summary dict with run_filename, prompt_version, timestamp, total_cases,
        and a per-model breakdown of pass counts and per-metric scores.
    """
    if not isinstance(data, list):
        return {
            "run_filename": run_filename,
            "error": "Unexpected results shape (expected list of model runs).",
        }

    first = data[0] if data else {}
    prompt_version = first.get("hyperparameters", {}).get("prompt_version", "")
    timestamp = first.get("timestamp", "")

    per_model: list[dict[str, Any]] = []
    for model_run in data:
        results = model_run.get("results", [])
        total = len(results)
        passed = sum(1 for r in results if r.get("success"))

        # Aggregate per-metric pass rates using the same metric_name set
        # the UI/coordinator expects (GEval, Hallucination, etc.).
        metric_stats: dict[str, dict[str, int]] = {}
        for r in results:
            for md in r.get("metrics_data", []) or []:
                name = md.get("name", "")
                if not name:
                    continue
                stats = metric_stats.setdefault(name, {"passed": 0, "total": 0})
                stats["total"] += 1
                if md.get("success"):
                    stats["passed"] += 1

        per_metric = {
            name: {
                "passed": s["passed"],
                "total": s["total"],
                "rate": s["passed"] / s["total"] if s["total"] else 0.0,
            }
            for name, s in sorted(metric_stats.items())
        }

        per_model.append({
            "model_id": model_run.get("model_id"),
            "test_run_id": model_run.get("test_run_id"),
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "per_metric": per_metric,
        })

    # Overall aggregate for at-a-glance reporting.
    total_cases = sum(m["total"] for m in per_model)
    total_passed = sum(m["passed"] for m in per_model)

    return {
        "run_filename": run_filename,
        "prompt_version": prompt_version,
        "timestamp": timestamp,
        "total_cases": total_cases,
        "total_passed": total_passed,
        "overall_pass_rate": total_passed / total_cases if total_cases else 0.0,
        "per_model": per_model,
        "hint": (
            "Use analyze_failures for a failure-pattern drilldown, or "
            "get_case_details(case_indices=[...]) for specific cases."
        ),
    }


@tool(
    "get_case_details",
    "Load full per-case details (input, actual_output, expected_output, metric scores, judge rationales) "
    "for a bounded set of cases from a specific eval run. "
    "Use this after `get_results` or `analyze_failures` when you need to inspect specific rows.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "run_filename": {"type": "string"},
            "model_id": {
                "type": "string",
                "description": "Optional model ID to filter by (omit for the first model run in the file).",
            },
            "case_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Specific case indices (0-based) to return. If omitted, "
                    "returns the first `limit` cases. Max 50 per call."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max cases to return when case_indices is omitted. Defaults to 20, hard-capped at 50.",
            },
        },
        "required": ["project_id", "run_filename"],
    },
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_case_details(args: dict[str, Any]) -> dict[str, Any]:
    """Load detailed per-case data for a subset of an eval run's cases.

    Bounded by count (``limit``, hard cap 50) so the response never exceeds
    the SDK transport's 1 MiB cap. The ``mcp_json`` size guard catches any
    overflow as a fallback.

    Args:
        args: Dict with project_id, run_filename, optional model_id,
            case_indices, and limit.

    Returns:
        Dict with selected case records for the chosen model.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        output_dir = project_dir / "output"
        results_path = safe_path(output_dir, args["run_filename"])
        if results_path is None:
            return mcp_error(f"Invalid filename: {args['run_filename']}")
        if not results_path.exists():
            return mcp_error(f"Run not found: {args['run_filename']}")

        runner = EvalRunner()
        data = runner.load_results(results_path)
        if not isinstance(data, list) or not data:
            return mcp_error("Results file is empty or malformed.")

        model_id_filter = args.get("model_id")
        if model_id_filter:
            model_run = next(
                (m for m in data if m.get("model_id") == model_id_filter),
                None,
            )
            if model_run is None:
                available = [m.get("model_id") for m in data]
                return mcp_error(
                    f"Model {model_id_filter!r} not found in run. "
                    f"Available: {available}"
                )
        else:
            model_run = data[0]

        results = model_run.get("results", []) or []

        # Resolve which indices to return — hard cap at 50 either way so a
        # single call can't bust the size guard even with verbose rationales.
        MAX_LIMIT = 50
        indices = args.get("case_indices")
        if indices is None:
            limit = min(int(args.get("limit") or 20), MAX_LIMIT)
            chosen_indices = list(range(min(limit, len(results))))
        else:
            if len(indices) > MAX_LIMIT:
                return mcp_error(
                    f"Too many case_indices ({len(indices)}); limit is {MAX_LIMIT} "
                    "per call. Split into multiple calls."
                )
            chosen_indices = [i for i in indices if 0 <= i < len(results)]

        cases = [dict(results[i], _index=i) for i in chosen_indices]

        return mcp_json({
            "run_filename": args["run_filename"],
            "model_id": model_run.get("model_id"),
            "total_cases_in_run": len(results),
            "returned_count": len(cases),
            "cases": cases,
        })

    except Exception as e:
        return mcp_error(f"Error: {e}")


@tool(
    "delete_run",
    "Delete one eval-results JSON file from a project's output/ directory. "
    "Use only for a run the user has asked you to remove — a run file is the "
    "results record and nothing else deletes one. Resolves the filename "
    "safely under ``output/`` (no traversal).",
    {"project_id": str, "run_filename": str},
)
async def delete_eval_run(args: dict[str, Any]) -> dict[str, Any]:
    """Delete a single ``output/eval_results_*.json`` file.

    Mirror of the HTTP ``DELETE /api/projects/{id}/eval/runs/{file}``
    endpoint so the agent can do the same cleanup in response to a user
    request without bouncing the request through the SPA.

    Publishes ``eval_run/deleted`` on success. The run list is pull-only
    REST, so without the event the Evaluations tab keeps rendering a run
    whose file is gone until unrelated navigation happens to refetch.

    On the relay path this tool executes in the AgentCore microVM and the
    event is carried to the coordinator over the turn's SSE stream, which
    publishes it on the project stream where the Evaluations tab subscribes,
    so the tab updates live. The publish is only lost when no live turn can
    carry it (logged by ``_publish_project_event``).

    Args:
        args: Dict with ``project_id`` and ``run_filename`` (basename
            only — path traversal is rejected).

    Returns:
        ``mcp_success`` on delete; ``mcp_error`` on validation failure
        or filesystem error so the agent loop survives.
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        output_dir = project_dir / "output"
        # Same validator as the HTTP DELETE endpoint
        # (api/eval_runs.py::_resolve_run_file_or_404). Rejects path
        # separators, hidden files, embedded NULs, and symlinks
        # pointing outside output/.
        results_path = safe_basename_path(output_dir, args["run_filename"])
        if results_path is None:
            return mcp_error(f"Invalid filename: {args['run_filename']}")
        if not results_path.is_file():
            return mcp_error(f"Run not found: {args['run_filename']}")
        results_path.unlink()
        # Sidecars are a derived cache of an immutable run — deleting the run
        # invalidates them. Best-effort cleanup avoids orphan _derived/<stem>/ trees.
        shutil.rmtree(_derived_dir(results_path), ignore_errors=True)
        logger.info(
            "[delete_run] project=%s file=%s",
            args["project_id"][:8], args["run_filename"],
        )
    except OSError as e:
        # Log full path server-side; return generic message to the
        # agent so the absolute EFS path doesn't end up in the
        # session transcript / S3 storage.
        logger.warning("[delete_run] OSError on unlink: %s", e)
        return mcp_error("Could not delete run file")
    except Exception as e:  # noqa: BLE001
        logger.warning("[delete_run] unexpected error: %s", e)
        return mcp_error("Internal error deleting run file")

    # The run list is a pull-only REST resource, so without this the Evaluations
    # tab keeps rendering a run whose file is gone until some unrelated
    # navigation happens to refetch. Published after the try block so it can
    # never be mistaken for part of the delete's own failure surface.
    _publish_project_event(args["project_id"], {
        "resource": "eval_run",
        "type": "deleted",
        "run_filename": results_path.name,
    })
    return mcp_success(f"Deleted run file: {args['run_filename']}")


@tool(
    "cancel_eval",
    "Cancel a running evaluation job. Kills the harness subprocess and marks the "
    "job as cancelled. If an optimization loop is active, it stops that too.",
    {"project_id": str, "job_id": str},
)
async def cancel_eval(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel a running eval job, stopping whichever execution home is running it.

    A detached cloud eval is stopped through the compute seam using the task ARN on
    its durable record; a dev/in-process eval has its subprocess group signalled.
    The mechanism is shared verbatim with the REST cancel endpoint so this tool
    cannot drift into reporting a cancel it did not perform. A running optimization
    loop is asked to stop as well.

    Args:
        args: Dict with ``project_id`` and ``job_id``.

    Returns:
        Dict with confirmation or error.
    """
    try:
        from app.platform.arch_split import resolve_arch_split
        from app.services.eval_cancel import cancel_eval_job

        project_id = args["project_id"]
        job_id = args["job_id"]
        project_dir = resolve_project_dir(project_id)
        arch_split = await resolve_arch_split(project_id)

        try:
            user_id = current_request().user_id
        except RuntimeError:
            from app.platform.auth import get_local_dev_user_id
            user_id = get_local_dev_user_id()

        outcome, status = await cancel_eval_job(
            project_id,
            job_id,
            user_id=user_id,
            project_dir=project_dir,
            arch_split=arch_split,
        )

        if outcome == "not_found":
            return mcp_error(f"Job not found: {job_id}")
        if outcome == "wrong_project":
            return mcp_error("Job does not belong to this project.")
        if outcome == "already_terminal":
            return mcp_success(f"Job already finished with status: {status}")
        if outcome == "stop_failed":
            # An error, not a success: the eval is still running and still
            # costing money. Saying "cancelled" here would tell the coordinator
            # the project is free when it is not.
            return mcp_error(
                f"Could not stop the worker for job {job_id[:8]}. The eval is "
                "still running — retry the cancel."
            )

        return mcp_success(f"Eval job {job_id[:8]} cancelled.")

    except Exception as e:
        logger.error("[cancel_eval] FAILED: %s", e, exc_info=True)
        return mcp_error(f"Error: {e}")
