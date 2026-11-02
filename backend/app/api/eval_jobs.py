# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval job lifecycle endpoints: list, get, and cancel eval jobs.

Live progress for an eval job flows over the project owner's WebSocket
connection (see ``app.api.ws``); this router only owns the REST CRUD
surface.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId
from app.services.jobs import JobNotFoundError, get_job_tracker

from app.services.project_manager import get_project_dir_or_404

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["eval"])



def _eval_job_to_dict(job) -> dict:
    """Serialize a Job to dict for the REST API.

    Manually builds the dict to avoid asdict()'s deepcopy, which fails
    on the asyncio.Condition field (uvloop objects can't be pickled).
    Excludes internal fields: result, events, _condition.
    """
    return {
        "job_id": job.job_id,
        "project_id": job.project_id,
        # ``job_type`` is stamped in the response so the SPA's
        # eval-progress poller can defensively filter to the eval type
        # (the backend filter at ``list_eval_jobs`` is the correctness
        # fix; this mirrors it on the wire as defense in depth).
        "job_type": job.type,
        "status": job.status,
        "progress": job.progress,
        "output_file": job.output_file,
        "error": job.error.to_dict() if job.error else None,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "pid": job.pid,
    }


def _is_fargate_handle(compute_handle: str | None) -> bool:
    """Return True when the compute_handle is a Fargate task ARN.

    A Fargate job's durable cancel handle is its ECS task ARN
    (``arn:aws:ecs:...:task/...``). Dev/local jobs either carry no
    compute_handle (in-process path) or a PID-shaped value; those keep
    the process-group kill path. Discriminating on the ARN prefix keeps
    the decision purely structural — no mode flag re-read at cancel time.
    """
    return bool(compute_handle) and compute_handle.startswith("arn:aws:ecs:")


async def _resolve_compute_handle(
    user_id: UserId, project_id: str, job_id: str
) -> str | None:
    """Read the durable job record's compute_handle from the metadata store.

    Reading from the durable record (not the in-memory Job) is what makes
    cancel survive a control-plane restart: after a restart the in-memory
    JobTracker is empty, but the task ARN persisted on the DynamoDB job
    record is still there for StopTask. Returns None when the store has no
    record or no handle (dev/local job).
    """
    from app.platform import get_platform

    store = get_platform().metadata_store
    record = await store.get_job(user_id, project_id, job_id)
    if record is None:
        return None
    return record.get("compute_handle")


@router.get("/{project_id}/eval/jobs/{job_id}")
async def get_eval_job(
    project_id: str,
    job_id: str,
    _user: UserId = Depends(require_project_read),
):
    """Get job status and progress for a specific eval job.

    Args:
        project_id: Unique identifier of the project.
        job_id: Unique identifier of the eval job.

    Returns:
        Job status dict excluding the large result payload.

    Raises:
        HTTPException 404: If the project or job is not found.
    """
    get_project_dir_or_404(project_id)
    tracker = get_job_tracker()
    try:
        job = tracker.get_job_for_project(project_id, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _eval_job_to_dict(job)


@router.post("/{project_id}/eval/jobs/{job_id}/cancel")
async def cancel_eval_job(
    project_id: str,
    job_id: str,
    _user: UserId = Depends(require_project_write),
):
    """Cancel a running eval job.

    Stops whichever execution home is running the eval — a detached Fargate worker
    through the compute seam, or a local subprocess group — marks the job cancelled,
    and asks a running optimization loop to stop. The mechanism itself lives in
    ``app.services.eval_cancel`` so the agent-facing tool runs the identical path;
    this function only maps the outcome onto HTTP.

    Args:
        project_id: Unique identifier of the project.
        job_id: Unique identifier of the eval job to cancel.

    Returns:
        Confirmation dict with status.

    Raises:
        HTTPException 404: If the project or job is not found.
        HTTPException 409: If the job is already in a terminal state.
    """
    from app.platform.arch_split import resolve_arch_split
    from app.services.eval_cancel import cancel_eval_job as cancel_eval_job_impl

    project_dir = get_project_dir_or_404(project_id)
    arch_split = await resolve_arch_split(project_id)

    outcome, status = await cancel_eval_job_impl(
        project_id,
        job_id,
        user_id=_user,
        project_dir=project_dir,
        arch_split=arch_split,
    )

    if outcome == "not_found":
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if outcome == "wrong_project":
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if outcome == "already_terminal":
        raise HTTPException(status_code=409, detail=f"Job already finished: {status}")
    if outcome == "stop_failed":
        # 503, not 200. The job is deliberately still running: reporting success
        # would show "cancelled" in the UI while the worker keeps invoking models.
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not stop the eval worker. The eval is still running — "
                "retry the cancel."
            ),
        )

    # Chat is a separate surface from the job bar and the toast, and a cancel
    # made here was made in the UI — so record it there, as the optimization
    # loop's UI cancel does. The agent's own cancel tool does not come through
    # this route, which is what keeps a cancel it made from being announced twice.
    from app.services.completion_turn import announce_eval_cancelled_from_ui
    from app.services.jobs import get_job_tracker

    job = get_job_tracker().get_job(job_id)
    if job is not None:
        announce_eval_cancelled_from_ui(project_id, job)

    return {"status": "cancelled", "job_id": job_id}


