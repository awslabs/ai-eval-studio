# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness build job executor — scaffold + validate as a background job.

Wraps the scaffold_harness and validate_harness operations in the
JobExecutor protocol for dispatch via the Job Manager. The raw MCP
tools remain available for direct coordinator use; this executor is
for when the build is dispatched as a background job.
"""

import logging
from pathlib import Path
from typing import Any

from app.services.jobs import Job, JobResult, _publish_project_event, get_job_tracker
from app.services.resource_error import Codes, ResourceError

logger = logging.getLogger(__name__)


async def run_harness_build_job(job: Job) -> JobResult:
    """Execute a harness build job (scaffold + validate).

    Expected config keys:
        project_id: Project UUID.
        project_dir: Absolute path to the project directory.
        scaffold: Whether to run scaffolding (default True).
        validate: Whether to run validation after scaffold (default True).

    Args:
        job: The Job instance (must be type "harness_build").

    Returns:
        JobResult with validation outcome.
    """
    from app.services.mcp_tools.harness import scaffold_harness, validate_harness

    tracker = get_job_tracker()
    config = job.config
    project_id = config["project_id"]

    try:
        tracker.update_progress(job.job_id, {"message": "Building harness..."})

        results: dict[str, Any] = {}

        if config.get("scaffold", True):
            tracker.update_progress(job.job_id, {"message": "Scaffolding harness files..."})
            scaffold_result = await scaffold_harness.handler({"project_id": project_id})
            results["scaffold"] = scaffold_result

        if config.get("validate", True):
            tracker.update_progress(job.job_id, {"message": "Validating harness..."})
            validate_result = await validate_harness.handler({"project_id": project_id})
            results["validate"] = validate_result

            # ``mcp_error`` puts ``is_error`` at the TOP LEVEL of the response,
            # alongside ``content`` — not on the individual content items (see
            # ``mcp_response.py``). Reading it off each item found it nowhere,
            # so this gate never fired and a failed validation was reported as
            # a COMPLETED harness build, leaving the user to run evals against
            # a harness that had already been rejected.
            content = validate_result.get("content", [])
            if validate_result.get("is_error"):
                error_text = next(
                    (item.get("text", "") for item in content if isinstance(item, dict)),
                    "Validation failed",
                )
                error = ResourceError(
                    code=Codes.HARNESS_SUBPROCESS_FAILED,
                    message=f"Harness validation failed: {error_text[:200]}",
                )
                tracker.update_state(job.job_id, "failed", result=results, error=error)
                _publish_project_event(job.project_id, {
                    "resource": "job",
                    "type": "failed",
                    "job_id": job.job_id,
                    "job_type": "harness_build",
                    **error.to_dict(),
                })
                return JobResult(status="failed", data=results, error=error)

        tracker.update_state(job.job_id, "completed", result=results)
        _publish_project_event(job.project_id, {
            "resource": "job",
            "type": "completed",
            "job_id": job.job_id,
            "job_type": "harness_build",
        })
        return JobResult(status="completed", data=results)

    except Exception as e:
        logger.error("[harness-executor] job=%s error: %s", job.job_id[:8], e, exc_info=True)
        error = ResourceError(
            code=Codes.HARNESS_SUBPROCESS_FAILED,
            message=f"Harness build failed: {e}",
            details={"exception": type(e).__name__},
        )
        tracker.update_state(job.job_id, "failed", error=error)
        _publish_project_event(job.project_id, {
            "resource": "job",
            "type": "failed",
            "job_id": job.job_id,
            "job_type": "harness_build",
            **error.to_dict(),
        })
        return JobResult(status="failed", error=error)
