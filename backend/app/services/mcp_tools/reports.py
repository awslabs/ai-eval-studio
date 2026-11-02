# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool for narrative-led report generation: ``save_report``.

The narrator agent reads a server-prepared facts-pack and writes only the
prose fields (recommendation, journey narrative, failure pattern clusters,
etc.). It delivers those fields by calling this tool.

Critical contract: the agent NEVER supplies numbers. The tool recomputes the
facts-pack server-side from the project + run identifier, merges it with the
agent's narrative, and writes the report JSON sidecar. This is what makes it
impossible for the agent to poison the deterministic numbers.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import tool

from app.services.mcp_response import mcp_error, mcp_success
from app.services.project_locks import get_project_lock
from app.services.project_paths import resolve_project_dir, safe_path
from app.services.report_snapshot import upload_report_to_s3
from app.services.report import (
    build_exec_facts,
    build_run_facts,
    write_report_json,
)

logger = logging.getLogger(__name__)


# Schema describes ALL narrative fields the agent may emit. The tool
# accepts whatever subset is supplied; missing fields are stored as None and
# the renderer's per-section logic handles their absence.
_NARRATIVE_FIELDS_SCHEMA = {
    # Per-run fields
    "takeaway": str,
    "failure_patterns": list,  # list of {label, representative_case_id, count, insight}
    "what_changed_worked": str,
    # Executive fields
    "intro": str,
    "recommendation": str,
    "journey_narrative": str,
    "data_quality_insight": str,
    "outlook": str,
    # Both
    "metric_descriptions": dict,  # {metric_name: one-line description}
}


@tool(
    "save_report",
    "Persist a narrative report. Pass the prose fields the report needs "
    "(takeaway, failure_patterns, recommendation, etc.); the tool recomputes "
    "the deterministic facts server-side and writes the merged report JSON. "
    "The agent must NEVER supply numbers — pass them only as references "
    "inside the prose. failure_patterns groups ALL failures into ≤5 patterns "
    "per the report-narrator contract.",
    {
        "project_id": str,
        "report_type": str,  # "run" | "executive"
        "run_filename": str,  # required for "run", ignored for "executive"
        "narrative": dict,  # narrative fields keyed per the schema
    },
)
async def save_report(args: dict[str, Any]) -> dict[str, Any]:
    """Save a report sidecar after re-computing facts server-side.

    Args:
        args: ``project_id``, ``report_type`` ("run" | "executive"),
            ``run_filename`` (required for "run"), and ``narrative`` (a dict
            of prose fields the agent produced).

    Returns:
        ``mcp_success`` with the on-disk path, or ``mcp_error`` on validation
        failure.
    """
    try:
        project_id = args.get("project_id")
        if not project_id:
            return mcp_error("project_id is required")
        report_type = args.get("report_type")
        if report_type not in ("run", "executive"):
            return mcp_error(
                f"report_type must be 'run' or 'executive', got: {report_type!r}"
            )

        narrative_in = args.get("narrative") or {}
        if not isinstance(narrative_in, dict):
            return mcp_error("narrative must be a JSON object")

        # Filter narrative to known fields only — defensive, so a stray key
        # from the agent doesn't pollute the sidecar.
        narrative: dict[str, Any] = {}
        for key, py_type in _NARRATIVE_FIELDS_SCHEMA.items():
            if key in narrative_in:
                value = narrative_in[key]
                if value is None:
                    narrative[key] = None
                elif isinstance(value, py_type):
                    narrative[key] = value
                else:
                    logger.warning(
                        "[save_report] dropping field %s (wrong type: got %s, want %s)",
                        key, type(value).__name__, py_type.__name__,
                    )

        project_dir = resolve_project_dir(project_id)
        if report_type == "run":
            run_filename = args.get("run_filename")
            if not run_filename:
                return mcp_error("run_filename is required for report_type='run'")
            safe = safe_path(project_dir / "output", run_filename)
            if safe is None or not safe.exists():
                return mcp_error(f"Run file not found: {run_filename}")

            project_lock = await get_project_lock(project_id)
            async with project_lock:
                facts = build_run_facts(project_dir, run_filename)
                target = write_report_json(
                    project_dir, "run", facts, narrative, run_filename=run_filename,
                )
            logger.info(
                "[save_report] project=%s scope=run run=%s wrote=%s",
                project_id[:8], run_filename, target.name,
            )
            # Durable copy OUTSIDE the lock: it is a network round-trip and the
            # EFS write is already committed, so holding the project lock for it
            # would block prompt/state writes for no benefit.
            await upload_report_to_s3(project_id, target)
            return mcp_success(f"Report saved: {target.relative_to(project_dir)}")

        # Executive report.
        # When the orchestrator (``generate_exec_report``) holds the
        # serialized exec-narration lock, it stashes the facts the
        # narrator was reasoning about. Use those instead of rebuilding
        # so the sidecar's stamped facts and the agent's prose come from
        # the same build — otherwise a winner change landing mid-narration
        # would produce prose-A about an old winner with stamp-B for the
        # new one. Outside the orchestrator path the stash is empty and
        # we rebuild as before.
        from app.services.report_generation import get_exec_facts_snapshot

        project_lock = await get_project_lock(project_id)
        async with project_lock:
            stashed = get_exec_facts_snapshot(project_id)
            facts = stashed if stashed is not None else build_exec_facts(project_dir)
            target = write_report_json(project_dir, "executive", facts, narrative)
        logger.info(
            "[save_report] project=%s scope=executive wrote=%s (snapshot=%s)",
            project_id[:8], target.name, stashed is not None,
        )
        await upload_report_to_s3(project_id, target)
        return mcp_success(f"Report saved: {target.relative_to(project_dir)}")

    except FileNotFoundError as e:
        return mcp_error(f"Not found: {e}")
    except ValueError as e:
        return mcp_error(f"Invalid input: {e}")
    except Exception as e:  # pragma: no cover - defensive
        logger.exception(
            "[save_report] project=%s ERROR: %s",
            (args.get("project_id") or "?")[:8], e,
        )
        return mcp_error(f"Error: {e}")
