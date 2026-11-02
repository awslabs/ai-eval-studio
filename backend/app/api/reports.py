# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report endpoints: serve the report JSON, trigger narrator regeneration.

The old HTML-rendering endpoints are gone. The frontend now renders both
reports as React components fed by the report JSON envelope
(``schema_version``, ``scope``, ``facts``, ``narrative``, ``generated_at``,
``stale`` — see ``_report_envelope``); typography and theme inherit from
the dashboard for free.

Two fetch endpoints serve the report JSON (per-run + executive); they are
POST so the run filename can ride in the body. Two regenerate endpoints
(write-gated) start the report-narrator agent and return 202 without waiting
for it; the agent's ``save_report`` MCP tool writes the sidecar that the fetch
endpoints return, and a ``report`` project event reports the outcome.

PDF export is client-side via ``@react-pdf/renderer`` — no backend
involvement.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId
from app.services.project_manager import get_project_dir_or_404
from app.services.report import (
    _facts_to_jsonable,
    build_exec_facts,
    build_run_facts,
    read_report_json,
)
from app.services.task_error_sink import attach_error_sink

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["reports"])

#: Regenerations started by the endpoints below. The event loop holds only a
#: weak reference to a task, so this set keeps each one alive until it ends.
_regenerate_tasks: set[asyncio.Task] = set()


def _start_regeneration(
    coro: Coroutine[Any, Any, bool], *, project_id: str, scope: str,
) -> None:
    """Run a narration in the background so the request can return at once.

    A narration takes 30-50 seconds, longer than CloudFront's 30-second
    origin read timeout, so a request that waited for it reached the browser
    as a 504 even though the report was written.

    Args:
        coro: The ``generate_*_report`` call to run.
        project_id: Project the narration is for.
        scope: ``"run"`` or ``"executive"``, used in the task name.
    """
    task = asyncio.create_task(
        coro, name=f"report-regenerate-{scope}-{project_id[:8]}",
    )
    _regenerate_tasks.add(task)
    task.add_done_callback(_regenerate_tasks.discard)
    attach_error_sink(task, project_id=project_id, resource="report")


class RunReportRequest(BaseModel):
    """Body for a per-run report fetch.

    Attributes:
        run: Optional run filename; the latest run is used when omitted.
    """

    run: str | None = None


class RegenerateRequest(BaseModel):
    """Body for the per-run regenerate endpoint."""

    run: str | None = None


def _report_envelope(
    scope: str,
    facts_dict: dict,
    narrative: dict | None,
    generated_at: str | None,
    *,
    stale: bool = False,
) -> dict:
    """Standard JSON shape: ``{schema_version, scope, facts, narrative, ...}``.

    ``stale`` (executive only) flags a kept-but-outdated narrative — the
    cached prose is for an earlier run of the same winning version, so
    the wording may not reflect the latest numbers. Stat cards are
    always recomputed fresh; only prose can drift.
    """
    from app.services.report import REPORT_SCHEMA_VERSION
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": scope,
        "facts": facts_dict,
        "narrative": narrative,
        "generated_at": generated_at,
        "stale": stale,
    }


@router.post("/{project_id}/report")
async def get_run_report(
    project_id: str,
    body: RunReportRequest,
    _user: UserId = Depends(require_project_read),
) -> dict:
    """Fetch the per-run report JSON.

    POST instead of GET so the frontend can pass the run filename in the
    body; the URL stays opinion-free. The body is small (one optional
    string), so this isn't a real cost.

    Returns the report envelope (see ``_report_envelope``). When the
    narrator hasn't run yet (or failed), ``narrative`` is null and the
    frontend renders the lean fallback.

    Args:
        project_id: Project UUID.
        body: Run selection (optional; latest by default).
        _user: Authorized user id (read gate).

    Raises:
        HTTPException: 404 when no runs exist or the named run is missing.
    """
    project_dir = get_project_dir_or_404(project_id)
    try:
        facts = build_run_facts(project_dir, body.run)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    facts_dict = _facts_to_jsonable(facts)
    sidecar = read_report_json(
        project_dir, "run", run_filename=facts.run_filename,
    )
    narrative = sidecar.get("narrative") if sidecar else None
    generated_at = sidecar.get("generated_at") if sidecar else None
    return _report_envelope("run", facts_dict, narrative, generated_at)


@router.post("/{project_id}/report/executive")
async def get_executive_report(
    project_id: str,
    _user: UserId = Depends(require_project_read),
) -> dict:
    """Fetch the executive report JSON for the project.

    Always summarizes the trophy-winning run + the journey across versions.
    """
    project_dir = get_project_dir_or_404(project_id)
    try:
        facts = build_exec_facts(project_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    facts_dict = _facts_to_jsonable(facts)
    sidecar = read_report_json(project_dir, "executive")
    narrative = sidecar.get("narrative") if sidecar else None
    generated_at = sidecar.get("generated_at") if sidecar else None
    # Drop the cached narrative only when the winning *version* changes —
    # not when a new run of the same winning version lands. The prose is
    # about the recommended prompt version; a fresh run of the same
    # version doesn't invalidate the wording, and dropping on the run
    # *file* caused a barebones flicker on every optimization iteration
    # while the same version kept winning. Stat-card numbers are always
    # recomputed fresh, so any same-version staleness is limited to
    # prose wording until the next auto-regen. A genuine version change
    # still drops so we never show a stale recommendation.
    # ``_dispatch_exec_if_winner_changed`` keeps regenerating the
    # narrative per winning run; this only changes when the GET blanks
    # it.
    cached_version = (sidecar or {}).get("facts", {}).get("winner_version")
    if cached_version and cached_version != facts.winner_version:
        narrative = None
        generated_at = None
    # Stale-flag: same winning version, but a newer run of that version
    # has landed since the cached narrative was written. The prose is
    # about the recommendation (the version) so it survives, but its
    # specific numbers may be a run behind. Only meaningful when we
    # actually KEEP the narrative — a None narrative is honest, not stale.
    cached_run = (sidecar or {}).get("facts", {}).get("winner_run_filename")
    stale = (
        narrative is not None
        and cached_run is not None
        and cached_run != facts.winner_run_filename
    )
    return _report_envelope(
        "executive", facts_dict, narrative, generated_at, stale=stale,
    )


@router.post("/{project_id}/report/regenerate", status_code=202)
async def post_report_regenerate(
    project_id: str,
    body: RegenerateRequest,
    user_id: UserId = Depends(require_project_write),
) -> dict:
    """Start regenerating the per-run narrative sidecar (manual trigger).

    Returns as soon as the narrator is started. When it ends, a
    ``report``/``updated`` project event lets an open Reports tab refetch
    the new prose, or ``report``/``failed`` says no new prose was written
    (the lean fallback still renders).

    Args:
        project_id: Project UUID.
        body: Run selection (``run`` is required).
        user_id: Authorized user id (write gate).

    Returns:
        ``{"ok": True}``.
    """
    project_dir = get_project_dir_or_404(project_id)
    if not body.run:
        raise HTTPException(status_code=400, detail="run is required")
    from app.services.report_generation import generate_run_report
    _start_regeneration(
        generate_run_report(project_dir, project_id, body.run, user_id),
        project_id=project_id, scope="run",
    )
    return {"ok": True}


@router.post("/{project_id}/report/executive/regenerate", status_code=202)
async def post_executive_report_regenerate(
    project_id: str,
    user_id: UserId = Depends(require_project_write),
) -> dict:
    """Start regenerating the executive narrative sidecar (manual trigger).

    Same contract as :func:`post_report_regenerate`, with ``scope``
    ``"executive"`` on the outcome event.
    """
    project_dir = get_project_dir_or_404(project_id)
    from app.services.report_generation import generate_exec_report
    _start_regeneration(
        generate_exec_report(project_dir, project_id, user_id),
        project_id=project_id, scope="executive",
    )
    return {"ok": True}
