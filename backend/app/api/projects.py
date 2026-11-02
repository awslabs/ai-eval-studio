# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST API endpoints for evaluation project management.

Provides CRUD operations for projects:
- List the current user's projects
- Create new project
- Get project details
- Update / delete project

Authorization: list/create/get/update/delete check
``platform.auth.require_access`` for the current user against the
project record. Dev mode: every project is owned by ``LOCAL_DEV_USER_ID``
so the check is effectively transparent. Cloud mode: only the owner may
read or write a project.
"""
import asyncio
import logging
import os
import re

from fastapi import APIRouter, Depends, HTTPException, Response

from app.models.schemas import (
    ProjectConfig,
    ProjectCreate,
    ProjectSummary,
    ProjectUpdate,
)
from app.platform.api_authz import require_project_read, require_project_write
from app.platform.types import UserId, current_user_id

logger = logging.getLogger(__name__)

# UUID v4 pattern for validating project IDs
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _validate_project_id(project_id: str) -> None:
    """Validate that project_id is a proper UUID format.

    Args:
        project_id: The project ID string to validate.

    Raises:
        HTTPException: 400 if the ID is not a valid UUID.
    """
    if not _UUID_RE.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _is_cloud_mode() -> bool:
    return os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"


def _get_pm():
    """Retrieve the ProjectManager singleton instance.

    Lazy import to avoid circular dependency issues during module initialization.

    Returns:
        ProjectManager: The singleton project manager instance.
    """
    from app.main import get_project_manager
    return get_project_manager()


@router.get("", response_model=list[ProjectSummary])
async def list_projects():
    """List projects owned by the current user.

    Filters by ``owner_id`` so cloud mode is correctly user-scoped. Records
    without an ``owner_id`` are skipped (unreachable via authz).

    Returns:
        list[ProjectSummary]: The requester's projects.
    """
    return _get_pm().list_projects(owner_id=current_user_id())


@router.post("", response_model=ProjectConfig, status_code=201)
async def create_project(req: ProjectCreate):
    """Create a new evaluation project owned by the current user.

    Stamps ``owner_id`` from the auth-middleware-populated request
    context so each project records ownership at creation time. Cloud
    mode also mirrors the record to DynamoDB so ``Auth.require_access``
    can find the project by id without knowing the owner partition.

    Args:
        req: Project creation request containing name and description.

    Returns:
        ProjectConfig: Complete configuration of the newly created project.
    """
    user_id = current_user_id()
    config = _get_pm().create_project(req, owner_id=user_id)

    # Cloud mode: persist project record + index entry to DDB so the
    # cloud auth chokepoint (find_project_by_id) works. Dev mode's FilesystemMetadataStore reads
    # config.json directly so put_project there is harmless duplication;
    # we still call it for consistency with cloud behavior. Failure is
    # surfaced as 500 rather than swallowed — a half-created project
    # (on EFS but not in DDB) would be invisible to subsequent calls.
    from app.platform import get_platform
    await get_platform().metadata_store.put_project(config.model_dump())

    return config


async def _strict_uuid(project_id: str) -> str:
    """Reject non-UUID project_ids on /api/projects/{id} CRUD routes.

    The platform-wide ``require_project_read``/``require_project_write``
    dependencies use a permissive validator (UUID *or* test fixture
    strings), but the user-facing project CRUD endpoints want strict
    UUID rejection. This dependency runs first by virtue of being
    declared before the auth dependency.
    """
    _validate_project_id(project_id)
    return project_id


@router.get("/{project_id}", response_model=ProjectConfig)
async def get_project(
    project_id: str,
    _checked: str = Depends(_strict_uuid),
    _user: UserId = Depends(require_project_read),
):
    """Retrieve full configuration for a specific project.

    Args:
        project_id: Unique identifier of the project to retrieve.

    Returns:
        ProjectConfig: Complete project configuration and metadata.

    Raises:
        HTTPException: 404 if project not found, 403 if not authorized.
    """
    config = _get_pm().get_project(project_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return config


@router.patch("/{project_id}", response_model=ProjectConfig)
async def update_project(
    project_id: str,
    updates: ProjectUpdate,
    _checked: str = Depends(_strict_uuid),
    _user: UserId = Depends(require_project_write),
):
    """Apply a partial update to a project's mutable configuration.

    ``agent_model``, ``name`` and ``description`` are exposed. Fields left unset
    on the payload are ignored, so a caller patches one field without having to
    restate the others.

    ``description`` is checked against ``None`` rather than truthiness: an empty
    string is how the client clears a description, and a truthiness test would
    silently make that a no-op.

    Args:
        project_id: Unique identifier of the project to update.
        updates: Partial patch payload (see :class:`ProjectUpdate`).

    Returns:
        ProjectConfig: The project after applying the update.

    Raises:
        HTTPException: 404 if the project does not exist, 403 if not authorized.
    """
    _validate_project_id(project_id)
    pm = _get_pm()
    config = pm.get_project(project_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Project not found")
    from app.platform import get_platform
    store = get_platform().metadata_store
    if _is_cloud_mode():
        # The config's owner comes from the directory it was found in. Refuse
        # before writing anything if that is not the owner the index — and so
        # the authorization check — resolved this id to: the copy on disk is
        # then not this project's, and saving it would write somewhere else.
        record = await store.find_project_by_id(project_id)
        if record is not None and record.get("owner_id") != config.owner_id:
            logger.error(
                "[update_project] project %s resolves to a config outside its "
                "owner's tree; refusing to save",
                project_id,
            )
            raise HTTPException(status_code=409, detail="Project ownership conflict")
    settings = {
        field: value
        for field, value in (
            ("agent_model", updates.agent_model),
            ("name", updates.name),
            ("description", updates.description),
        )
        if value is not None
    }
    for field, value in settings.items():
        setattr(config, field, value)
    pm.save_config(config)
    # Mirror the edited settings, and only those, to the store, so nothing an
    # agent wrote into config.json reaches the record that authorizes access.
    try:
        if not await store.update_project_settings(config.owner_id, project_id, settings):
            logger.warning("[update_project] no store record for %s", project_id)
    except Exception:  # noqa: BLE001 - filesystem write succeeded; log + continue
        import logging
        logging.getLogger(__name__).warning(
            "[update_project] DDB sync failed for %s",
            project_id,
            exc_info=True,
        )
    return config


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    _checked: str = Depends(_strict_uuid),
    user_id: UserId = Depends(require_project_write),
):
    """Delete a project and all its contents — reliably and atomically-enough.

    Deletion spans three independent stores (S3 chat transcript, EFS tree,
    DynamoDB record+index). No sequential order across them is safe against
    an interruption (crash / OOM / deploy mid-delete), so we don't rely on
    ordering. Instead:

    1. **Commit the intent durably FIRST** — set ``deleting: true`` in
       config.json. This one fast write is all the request waits on. Once
       set, the project is committed to deletion: ``get_project`` /
       ``list_projects`` treat it as already gone (no zombie ever shows in
       the UI), so the response can return 204 immediately.
    2. **Return 204, finish cleanup in the BACKGROUND** — store removal
       (halt work → transcript → DDB → multi-GB ``rmtree``, tree LAST since
       it holds the marker) runs in a background task. Holding the response
       for the full rmtree (tens of seconds on EFS) previously made the
       client / load balancer fire a second DELETE mid-delete, which 404'd
       because the DDB auth pointer was already gone — the source of the
       spurious "already removed" on normal deletes.
    3. **Guaranteed completion** — if the background task or the whole
       process dies before cleanup finishes, the ``deleting`` marker remains
       and the startup reconcile sweep (:func:`_recover_pending_deletes`)
       finishes it. A retry DELETE also re-drives it. Idempotent throughout.

    Args:
        project_id: Unique identifier of the project to delete.

    Returns:
        Response: Empty 204 — the project is committed to deletion and
        hidden; store cleanup completes in the background.

    Raises:
        HTTPException: 404 if project doesn't exist, 403 if not authorized.
    """
    _validate_project_id(project_id)
    pm = _get_pm()

    # Resolve the project, INCLUDING one already mid-deletion (a retry of an
    # interrupted delete). 404 only when it's truly gone.
    config = pm.load_config_including_deleting(project_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Commit the deletion intent durably. This one fast write is all the
    # request waits on — from here the project is invisible to read paths
    # and guaranteed to be finished, either by the background task below or,
    # if the process dies first, by the startup reconcile sweep.
    if not config.deleting:
        config.deleting = True
        await asyncio.to_thread(pm.save_config, config)

    # Run the slow store cleanup (halt work → transcript → DDB → multi-GB
    # ``rmtree``) in the BACKGROUND and return 204 immediately. Holding the
    # response for the full rmtree (tens of seconds on EFS) is what caused
    # the client / load balancer to fire a second DELETE mid-delete, which
    # 404'd because the DDB auth pointer was already gone. With an instant
    # 204 nothing retries, so that window closes.
    _schedule_delete_cleanup(
        project_id, owner_id=config.owner_id, session_id=config.session_id
    )
    return Response(status_code=204)


# In-flight background delete-cleanup tasks, keyed by project_id. Holding a
# reference prevents the task from being garbage-collected mid-run, and the
# key lets a concurrent/retry DELETE skip scheduling a duplicate cleanup
# (two parallel ``rmtree`` on the same dir). Cleared on completion.
_delete_cleanup_tasks: dict[str, "asyncio.Task[None]"] = {}


def _schedule_delete_cleanup(
    project_id: str, *, owner_id: str | None, session_id: str | None
) -> None:
    """Schedule background store cleanup for an already-marked project.

    Idempotent per project: if a cleanup task is already in flight for this
    id (a double-fire, or a retry while the first is still running), this is
    a no-op — the existing task finishes the job. After a process restart
    the registry is empty, so a retry (or the startup sweep) re-drives it.
    """
    existing = _delete_cleanup_tasks.get(project_id)
    if existing is not None and not existing.done():
        return

    task = asyncio.create_task(
        _run_delete_cleanup(
            project_id, owner_id=owner_id, session_id=session_id
        )
    )
    _delete_cleanup_tasks[project_id] = task
    task.add_done_callback(
        lambda _t, pid=project_id: _delete_cleanup_tasks.pop(pid, None)
    )


async def _run_delete_cleanup(
    project_id: str, *, owner_id: str | None, session_id: str | None
) -> None:
    """Halt in-flight work, then remove every store. Background entry point.

    On failure the ``deleting`` marker stays set (we never clear it here),
    so the next DELETE retry or the startup reconcile sweep completes it.
    The user already got a 204 — from their view the project is gone, and
    it will be.
    """
    try:
        await _halt_project_work(project_id, owner_id=owner_id)
        await _complete_project_deletion(
            project_id, owner_id=owner_id, session_id=session_id
        )
    except Exception:  # noqa: BLE001 - marker kept; retry/sweep finishes it
        logger.warning(
            "[delete_project] background cleanup failed for %s; marker kept "
            "for retry/startup-sweep", project_id, exc_info=True,
        )


async def _complete_project_deletion(
    project_id: str,
    *,
    owner_id: str | None,
    session_id: str | None,
) -> None:
    """Remove every store for a project already flagged ``deleting``.

    Idempotent and order-safe: each step tolerates "already gone". Shared by
    the delete route and the startup reconcile sweep so they can't drift.
    Raises if any store fails — the caller keeps the ``deleting`` marker so
    the delete is retried (next request) or completed (next startup sweep).

    Order: transcript → report copies → DDB record → EFS tree. The tree holds
    the durable ``deleting`` marker, so it goes LAST: until it's removed, an
    interrupted run is always recoverable from the marker. All blocking work
    runs on a thread so the event loop keeps answering health checks.

    Args:
        project_id: Project being deleted.
        owner_id: Project owner (for the DDB delete PK). May be ``None`` in
            dev mode, where there's no separate metadata record.
        session_id: SDK session id whose S3 transcript to cascade-delete, or
            ``None`` when no agent turn ever ran.
    """
    from app.platform import get_platform

    pm = _get_pm()

    # Transcript (S3, cloud only; no-op in dev where session_store is None).
    if session_id:
        await _delete_session_transcript(project_id, session_id, owner_id=owner_id)

    # Durable report copies (S3, cloud only). Keyed by project id rather than
    # by session, so this runs whether or not an agent turn ever happened.
    from app.services.report_snapshot import delete_project_reports_from_s3

    await delete_project_reports_from_s3(project_id)

    # DDB record + project-index (cloud auth pointer). Cloud only, and the mode
    # test is the load-bearing part: this was guarded on ``owner_id`` alone, but
    # dev projects carry an owner too, so it ran in dev — where the metadata
    # store is tree-backed and its "record" IS ``config.json``. Deleting it here
    # removed the project's ``deleting`` marker before the tree step below, so a
    # retry mid-cleanup could no longer resolve the project and 404'd. Dev needs
    # no separate record: the tree delete is the record delete.
    if _is_cloud_mode() and owner_id:
        await get_platform().metadata_store.delete_project(owner_id, project_id)

    # EFS tree LAST — holds the ``deleting`` marker. ``shutil.rmtree`` can
    # walk tens of thousands of files; offload so the loop stays responsive.
    await asyncio.to_thread(pm.delete_project, project_id)


async def _halt_project_work(project_id: str, *, owner_id: str | None) -> None:
    """Cancel any running optimization loop and eval subprocess.

    Best-effort: marks optimization state cancelled, cancels the
    orchestrator asyncio task, and awaits SIGTERM→SIGKILL on every active
    eval subprocess (``hard_cancel_project_eval``). Logged-but-swallowed so
    a cancel failure doesn't block the delete.

    Args:
        project_id: Project whose background work should be stopped.
        owner_id: Project owner, required in cloud mode to resolve the tree.
    """
    try:
        # Flip optimization state so the orchestrator loop won't redispatch.
        #
        # Routed through the repository: this is a read-modify-WRITE and it used to
        # be an ungated EFS read + bare EFS save. On the durable path the loop polls
        # DynamoDB for its status, so an EFS-only write left DDB saying "running"
        # and the loop kept dispatching evals THROUGH the delete.
        from app.services import optimization_state_repository as opt_repo
        from app.services.project_manager import project_dir_for_owner

        # Owner-aware for the same reason as the transcript delete: the project
        # is already flagged ``deleting``, so a lookup-based resolve would hand
        # back the flat path and read optimization state from a directory that
        # does not exist — leaving the loop's status untouched in cloud mode.
        project_dir = project_dir_for_owner(project_id, owner_id)
        handle = await opt_repo.read(project_id, project_dir)
        if handle.is_live():
            handle.state.status = "cancelled"
            opt_repo.write(handle)

        # Cancel the orchestrator task if it's alive.
        orchestrator_task_name = f"optimization-{project_id[:8]}"
        for task in asyncio.all_tasks():
            if task.get_name() == orchestrator_task_name:
                task.cancel()
                break

        # Await the kill of every active eval subprocess for this project.
        from app.services.harness_runner import hard_cancel_project_eval
        await hard_cancel_project_eval(project_id)
    except Exception:  # noqa: BLE001 - best-effort; delete proceeds regardless
        logger.warning(
            "[delete_project] halting in-flight work failed for %s",
            project_id,
            exc_info=True,
        )


async def _delete_session_transcript(
    project_id: str, session_id: str, *, owner_id: str | None
) -> None:
    """Delete the project's chat transcript from the sessions bucket.

    Cloud only: the SDK's session store is S3-backed and keyed by the
    agent cwd (the project dir). Dev mode's ``session_store`` is ``None``
    and transcripts live under the project tree, so this is a no-op there.

    The owner id has to be passed in. By the time cleanup runs the project is
    flagged ``deleting``, which makes ordinary resolution report it as gone and
    fall back to the flat layout — so deriving the directory here produced
    ``{projects_dir}/{pid}`` instead of ``users/{owner}/projects/{pid}``. That is
    a different SDK project key, and deleting a prefix that matches nothing is a
    silent success in S3, so every cloud project's transcript survived its own
    deletion. Nothing reaped them afterwards either: the sessions bucket has no
    expiration precisely because this cascade was expected to cover it.

    Raises on failure so the caller can report partial deletion and keep
    the DDB record for retry rather than orphaning the transcript objects.
    ``S3SessionStore.delete`` already runs its list+delete on a thread, so
    this doesn't block the event loop. The cascade removes every part file
    under the ``{project_key}/{session_id}/`` prefix.

    Args:
        project_id: Project whose transcript should be removed.
        session_id: SDK session id from the project config.
        owner_id: Project owner, required in cloud mode to build the real key.
    """
    from claude_agent_sdk import project_key_for_directory

    from app.platform import get_platform
    from app.services.project_manager import project_dir_for_owner

    session_store = get_platform().session_store
    if session_store is None:
        return  # dev mode — transcript lives under the tree
    # Raises when cloud mode has no owner_id. Failing keeps the deleting marker
    # so the sweep retries; guessing would delete some other prefix, or none.
    project_dir = project_dir_for_owner(project_id, owner_id)
    await session_store.delete(
        {
            "project_key": project_key_for_directory(str(project_dir)),
            "session_id": session_id,
            "subpath": None,
        }
    )
