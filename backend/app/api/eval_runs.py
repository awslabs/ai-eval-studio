# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval run history and dataset endpoints: describe dataset, list/get runs, list/get prompts."""

import logging
import shutil

from fastapi import APIRouter, Depends, HTTPException

from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId
from app.services.eval_runner import EvalRunner, RunFileMissingError, _derived_dir
from app.services.project_paths import safe_basename_path

from app.services.project_manager import get_project_dir_or_404

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["eval"])

_eval_runner: EvalRunner | None = None


def _get_eval_runner() -> EvalRunner:
    """Get or create the singleton EvalRunner instance.

    Returns:
        EvalRunner: The shared runner instance.
    """
    global _eval_runner
    if _eval_runner is None:
        _eval_runner = EvalRunner()
    return _eval_runner


@router.get("/{project_id}/prompts")
async def list_prompts(
    project_id: str,
    _user: UserId = Depends(require_project_read),
):
    """List all prompt version files in the project's prompts directory.

    Args:
        project_id: Unique identifier of the project.

    Returns:
        Sorted list of prompt filenames. Empty list if no prompts directory
        or no files exist.
    """
    project_dir = get_project_dir_or_404(project_id)
    prompts_dir = project_dir / "prompts"
    if not prompts_dir.exists():
        return []
    # Sort by version number descending (v14, v13, ..., v1, baseline)
    files = [f.name for f in prompts_dir.iterdir() if f.is_file()]
    def _version_key(name: str) -> int:
        stem = name.removesuffix(".md")
        if stem == "baseline":
            return -1
        if stem.startswith("v") and stem[1:].isdigit():
            return int(stem[1:])
        return -2
    return sorted(files, key=_version_key, reverse=True)


@router.get("/{project_id}/eval/runs")
async def list_eval_runs(
    project_id: str,
    _user: UserId = Depends(require_project_read),
):
    """List all evaluation runs for a project, newest first.

    Returns DeepEval-native format summaries.
    """
    logger.debug("[eval-api] eval-runs.list project=%s", project_id[:8])
    project_dir = get_project_dir_or_404(project_id)
    runner = _get_eval_runner()
    runs = runner.list_runs(project_dir)
    # The backend is THE trophy-scoring implementation; the SPA only
    # orders these tuples (pickTrophyIndex). Targets from harness config,
    # the criteria owner — same source the panel displays as Convergence.
    from app.services.trophy import resolve_trophy_targets, trophy_score_run

    targets = resolve_trophy_targets(project_dir)
    for run in runs:
        run["trophy_score"] = trophy_score_run(run, targets)
    logger.debug("[eval-api] eval-runs.list project=%s count=%d", project_id[:8], len(runs))
    return runs


@router.get("/{project_id}/eval/runs/{run_filename}")
async def get_eval_run(
    project_id: str,
    run_filename: str,
    _user: UserId = Depends(require_project_read),
):
    """Get the full results of a specific eval run (DeepEval-native format)."""
    logger.debug("[eval-api] eval-run.get project=%s file=%s", project_id[:8], run_filename)
    project_dir = get_project_dir_or_404(project_id)
    results_path = _resolve_run_file_or_404(project_dir, run_filename)

    runner = _get_eval_runner()
    try:
        data = runner.load_results(results_path)
    except RunFileMissingError as exc:
        raise _run_gone_404(run_filename, exc) from exc
    total_results = sum(len(mr.get("results", [])) for mr in data) if isinstance(data, list) else 0
    logger.debug("[eval-api] eval-run.loaded file=%s results=%d", run_filename, total_results)
    return data


@router.get("/{project_id}/eval/runs/{run_filename}/compact")
async def get_eval_run_compact(
    project_id: str,
    run_filename: str,
    _user: UserId = Depends(require_project_read),
):
    """Get compact results of a specific eval run for low-bandwidth transfer.

    Returns the same model-run structure as get_eval_run but with slim
    per-case projections: full text fields (input, actual_output,
    expected_output, context, retrieval_context) replaced with previews
    and hashes; per-metric reasons/thresholds/errors dropped;
    latency/token fields promoted from additional_metadata to top level.
    """
    logger.debug("[eval-api] eval-run.compact project=%s file=%s", project_id[:8], run_filename)
    project_dir = get_project_dir_or_404(project_id)
    results_path = _resolve_run_file_or_404(project_dir, run_filename)

    runner = _get_eval_runner()
    try:
        data = runner.load_compact(results_path)
    except RunFileMissingError as exc:
        raise _run_gone_404(run_filename, exc) from exc
    total_results = sum(len(mr.get("results", [])) for mr in data) if isinstance(data, list) else 0
    logger.debug("[eval-api] eval-run.compact-loaded file=%s results=%d", run_filename, total_results)
    return data


@router.get("/{project_id}/eval/runs/{run_filename}/cases/{case_id}")
async def get_eval_case_detail(
    project_id: str,
    run_filename: str,
    case_id: str,
    _user: UserId = Depends(require_project_read),
):
    """Get full per-case detail for a single test case across all models.

    Returns input/expected_output (shared) and per-model actual_output,
    token/latency stats, and metric reasons. Used for on-demand lazy
    loading when expanding a specific case row in the Compare panel.
    """
    logger.debug(
        "[eval-api] eval-case-detail.get project=%s file=%s case=%s",
        project_id[:8], run_filename, case_id[:16],
    )
    project_dir = get_project_dir_or_404(project_id)
    results_path = _resolve_run_file_or_404(project_dir, run_filename)

    runner = _get_eval_runner()
    try:
        detail = runner.load_case_detail(results_path, case_id)
    except RunFileMissingError as exc:
        raise _run_gone_404(run_filename, exc) from exc
    if not detail["models"]:
        logger.warning(
            "[eval-api] eval-case-detail.not-found file=%s case=%s",
            run_filename, case_id[:16],
        )
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    logger.debug(
        "[eval-api] eval-case-detail.loaded file=%s case=%s models=%d",
        run_filename, case_id[:16], len(detail["models"]),
    )
    return detail


def _run_gone_404(run_filename: str, exc: RunFileMissingError) -> HTTPException:
    """Build the 404 for a run file that vanished between resolve and read.

    ``_resolve_run_file_or_404`` checks existence, but the read happens after
    that check returns, so a concurrent delete lands in between and the reader
    raises. Without this the race escaped as an unhandled ``FileNotFoundError``
    and a 500, which described a server fault where there was none.

    Emits the same ``Run not found: <file>`` detail as the pre-check's 404 so a
    client cannot tell the race apart from an ordinary missing run — they mean
    the same thing to the caller. The resolved absolute path stays server-side:
    echoing it would leak EFS paths and the per-tenant owner_sub.

    Args:
        run_filename: Client-supplied run filename, for the response detail.
        exc: The raised domain error, logged for the server-side trail.

    Returns:
        The ``HTTPException`` to raise at the call site.
    """
    logger.info(
        "[eval-api] eval-run.deleted-during-read file=%s err=%s", run_filename, exc,
    )
    return HTTPException(status_code=404, detail=f"Run not found: {run_filename}")


def _resolve_run_file_or_404(project_dir, run_filename: str):
    """Resolve ``output/{run_filename}`` defensively and 404 if missing.

    Single source of truth: routes through
    ``services.project_paths.safe_basename_path``, which rejects path
    separators, hidden files, embedded NULs, and (via the underlying
    ``safe_path`` containment check) symlinks pointing outside the
    project's ``output/`` directory. The MCP ``delete_run`` tool uses
    the same helper, so HTTP and MCP cannot drift.

    A run is identified by its FILENAME, full stop. This used to fall back to
    treating an unresolvable name as a bare iteration UUID and correlating it
    against optimization state to guess a file — support for iterations recorded
    before run files were materialized. Pre-release, there is no such data to
    support, and the guess had teeth: it picked the lexicographically newest file
    matching the prompt version, so a request for one iteration's results could
    be answered with a DIFFERENT run of the same version.
    """
    output_dir = project_dir / "output"
    results_path = safe_basename_path(output_dir, run_filename)
    if results_path is None:
        raise HTTPException(status_code=400, detail="Invalid run filename")
    if results_path.is_file():
        return results_path

    logger.warning("[eval-api] eval-run.not-found file=%s", results_path)
    raise HTTPException(status_code=404, detail=f"Run not found: {run_filename}")


@router.delete("/{project_id}/eval/runs/{run_filename}", status_code=204)
async def delete_eval_run(
    project_id: str,
    run_filename: str,
    _user: UserId = Depends(require_project_write),
):
    """Delete one ``output/eval_results_*.json`` file from the project tree.

    Run files are otherwise write-once and accumulate for the life of the
    project: nothing else in the system removes one, so pruning is the
    only way a user drops a run they don't want in their history (or in
    the trophy ranking, which scores every run the API returns).

    Authz: ``require_project_write`` — only project owners (or those
    with explicit write access) can prune their run history.
    """
    logger.info("[eval-api] eval-run.delete project=%s file=%s", project_id[:8], run_filename)
    project_dir = get_project_dir_or_404(project_id)
    results_path = _resolve_run_file_or_404(project_dir, run_filename)
    try:
        results_path.unlink()
        # Sidecars are a derived cache of an immutable run — deleting the run
        # invalidates them. Best-effort cleanup avoids orphan _derived/<stem>/ trees.
        shutil.rmtree(_derived_dir(results_path), ignore_errors=True)
    except OSError as exc:
        # Log full exception (including resolved path) server-side; do
        # NOT echo the OSError into the HTTP response, which would leak
        # absolute EFS paths and the per-tenant owner_sub to the client.
        logger.warning(
            "[eval-api] eval-run.delete-failed file=%s err=%s",
            run_filename, exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Could not delete run file",
        ) from exc

    # Same event the MCP ``delete_run`` tool publishes — the run list is a
    # pull-only resource, so any other open tab (or panel) would otherwise keep
    # rendering the deleted run until something else happened to refetch.
    # Published after the try block so this can never be mistaken for part of
    # the delete's own failure surface.
    from app.services.jobs import _publish_project_event
    _publish_project_event(project_id, {
        "resource": "eval_run",
        "type": "deleted",
        "run_filename": results_path.name,
    })


@router.get("/{project_id}/prompts/{version}")
async def get_prompt_content(
    project_id: str,
    version: str,
    _user: UserId = Depends(require_project_read),
):
    """Get the text content of a specific prompt version.

    Args:
        project_id: Unique identifier of the project.
        version: Prompt filename (e.g., "baseline.md", "v1.md").

    Returns:
        Dict with version name, content text, word count, and char count.
    """
    logger.debug("[eval-api] prompt.get project=%s version=%s", project_id[:8], version)
    project_dir = get_project_dir_or_404(project_id)
    prompt_path = project_dir / "prompts" / version
    if not prompt_path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt not found: {version}")
    content = prompt_path.read_text()
    return {
        "version": version,
        "content": content,
        "word_count": len(content.split()),
        "char_count": len(content),
    }
