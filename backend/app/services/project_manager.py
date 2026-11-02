# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project lifecycle management and filesystem operations.

This module handles all CRUD operations for evaluation projects, including:
- Creating projects with standardized directory structures
- Loading and saving project configurations
- Listing and deleting projects
- Computing enriched project metadata for list views

Also exposes :func:`get_project_dir_or_404` — the canonical helper for
resolving ``{project_id}`` URL params to an absolute project directory
in REST handlers. Combines the existence check with the traversal check
so routers get both guarantees with one call.

Layout:
    Cloud mode  →  ``{projects_dir}/users/{owner_sub}/projects/{pid}/``
    Dev mode    →  ``{projects_dir}/{pid}/`` (flat)

The cloud-mode prefix gives every project a per-owner path. It is a
layout, not a kernel fence: the EFS access point squashes every NFS op to
UID/GID 1000, no process drops privileges, and authorization is enforced
by the app-layer ``require_access`` check.
"""
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from app.models.schemas import ProjectConfig, ProjectCreate, ProjectSummary
from app.services.eval_run_files import eval_run_files
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


def _is_cloud_mode() -> bool:
    """Return True when the platform runs against AWS-backed adapters."""
    return os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"

# Standard directory structure created for each evaluation project.
# Organized by purpose: documentation, source code, and outputs.
# ``data/*`` subfolders (``uploads/``, ``raw/``, ``synthetic/``,
# ``processed/``) are created lazily by the first writer so fresh
# projects don't carry empty placeholder directories around.
PROJECT_SUBDIRS = [
    "docs",               # Document uploads (see uploads.ALLOWED_DEST_SUBDIRS)
    "prompts",            # Prompt templates and variations
    "src/metrics",        # Scaffolded metric implementations
    "src/evaluation",     # Scaffolded evaluation orchestration
    "src/models",         # Scaffolded model adapters
    "tests",              # Scaffolded harness smoke tests
    # Not pre-created: "output/reports" is made lazily by its only writer
    # (report._reports_dir mkdirs on first report), the same way the runner's
    # save_results creates output/ on the first run, matching the data/*
    # lazy-creation rule.
]


class ProjectManager:
    """Manages evaluation project lifecycle and filesystem operations.

    Handles creation, retrieval, listing, and deletion of projects. Each project
    is stored as a directory containing a config.json file and standardized subdirectories.
    """

    def __init__(self, projects_dir: Path):
        """Initialize the project manager.

        Args:
            projects_dir: Root directory where all project folders are stored.
                In cloud mode this is ``/mnt/projects`` (the EFS mount root);
                projects nest under ``users/{owner_sub}/projects/{pid}/``.
                In dev mode it's a local directory holding flat
                ``{pid}/`` subdirectories.
        """
        self.projects_dir = projects_dir

    def _project_dir(self, project_id: str, owner_id: str | None) -> Path:
        """Resolve a project directory under the per-mode layout.

        Cloud mode requires ``owner_id``; dev mode ignores it (flat layout).
        Used by every method that previously did ``self.projects_dir / pid``.

        Args:
            project_id: Project UUID.
            owner_id: Owner's Cognito ``sub`` (cloud mode); ``None`` is
                accepted in dev mode.

        Returns:
            Absolute path to ``{projects_dir}/[users/{owner_sub}/projects/]{pid}/``.

        Raises:
            ValueError: cloud mode + ``owner_id`` is None.
        """
        if _is_cloud_mode():
            if not owner_id:
                raise ValueError(
                    "owner_id is required in cloud mode to resolve a project directory"
                )
            return self.projects_dir / "users" / owner_id / "projects" / project_id
        return self.projects_dir / project_id

    def create_project(
        self,
        req: ProjectCreate,
        *,
        owner_id: str | None = None,
    ) -> ProjectConfig:
        """Create a new evaluation project with standard directory structure.

        Generates a unique project ID, creates all subdirectories, initializes
        the configuration file, and returns the complete project config.

        Args:
            req: Project creation request with name and description.
            owner_id: Cognito ``sub`` of the user who owns the new project.
                Routes pass the value derived from ``current_user_id()``
                so the on-disk config records ownership at creation time.
                ``None`` defers to the dev-mode default (``LOCAL_DEV_USER_ID``)
                so older callers / tests keep working unchanged.

        Returns:
            ProjectConfig: Complete configuration of the newly created project.
        """
        if owner_id is None:
            from app.platform.auth import get_local_dev_user_id
            owner_id = get_local_dev_user_id()

        # Generate unique identifier for the project
        project_id = str(uuid.uuid4())
        project_dir = self._project_dir(project_id, owner_id)

        # Create standardized directory structure
        for subdir in PROJECT_SUBDIRS:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize project configuration with creation metadata.
        config = ProjectConfig(
            id=project_id,
            name=req.name,
            description=req.description,
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_id=owner_id,
        )
        self.save_config(config)

        # No per-user chown: the EFS access point squashes all NFS ops
        # to UID/GID 1000, so per-owner UIDs aren't visible to the
        # storage layer. Tenant isolation lives in
        # ``app.platform.auth.require_access`` + the per-user path
        # layout under ``users/{owner_sub}/projects/{pid}/``.

        return config

    def list_projects(
        self,
        *,
        owner_id: str | None = None,
    ) -> list[ProjectSummary]:
        """List projects, optionally filtered by ``owner_id``.

        Scans the projects directory for config.json files and computes metadata
        from the filesystem: last activity timestamp, latest eval pass rate,
        model count, and lifecycle status.

        Args:
            owner_id: When provided, only projects whose ``config.owner_id``
                matches are returned. Records without an ``owner_id`` are
                skipped (unreachable via authz). ``None`` (default) returns
                every project on disk — used by background sweeps like
                the startup orphan-eval recovery.

        Returns:
            list[ProjectSummary]: Enriched project summaries, most recently
                active first (``last_activity`` desc, ``created_at`` as the
                fallback for never-active projects). This ordering is the
                DECLARED contract — the SPA renders it as received.
        """
        projects = []
        if not self.projects_dir.exists():
            return projects

        # Cloud mode: layout is ``users/{sub}/projects/{pid}/config.json``.
        # Dev mode: layout is ``{pid}/config.json``.
        # When ``owner_id`` is provided, we can scan a single user's tree;
        # when ``None`` (dev sweeps), we walk every user.
        if _is_cloud_mode():
            users_root = self.projects_dir / "users"
            if not users_root.exists():
                return projects
            if owner_id is not None:
                config_glob = list(
                    sorted((users_root / owner_id / "projects").glob("*/config.json"))
                )
            else:
                config_glob = list(
                    sorted(users_root.glob("*/projects/*/config.json"))
                )
        else:
            config_glob = list(sorted(self.projects_dir.glob("*/config.json")))

        for config_path in config_glob:
            data = self._read_config_file(config_path)
            if data is None:
                # Skip projects with invalid/corrupted config files
                continue

            # A project flagged for deletion is committed to deletion —
            # treat it as already gone so it never shows as a zombie while
            # the delete (or the startup reconcile sweep) finishes it.
            if data.get("deleting"):
                continue

            project_owner = data.get("owner_id")
            if not project_owner:
                # Records without owner_id are unreachable via authz — skip.
                logger.warning(
                    "[project-manager] skipping config without owner_id: %s",
                    config_path,
                )
                continue

            if owner_id is not None and project_owner != owner_id:
                continue

            try:
                project_dir = config_path.parent
                enriched = self._compute_project_metadata(project_dir, data)
                projects.append(enriched)
            except Exception:
                continue

        # Declared ordering: most recently ACTIVE first (last_activity,
        # falling back to created_at for projects that have never produced
        # a file). Before this, the list carried the glob's directory-name
        # order — UUID hex, i.e. effectively random from the user's seat.
        # Backend owns the ordering (same single-authority rule as
        # eval_runner.list_runs / lib/run-list.ts): the SPA renders
        # in the order received and must not re-sort.
        projects.sort(
            key=lambda p: p.last_activity or p.created_at or "",
            reverse=True,
        )
        return projects

    def _compute_project_metadata(
        self, project_dir: Path, config_data: dict
    ) -> ProjectSummary:
        """Compute enriched metadata for a single project from its filesystem.

        Inspects data/raw, prompts/, output/, and src/config.py to determine
        project status and statistics. Uses try/except throughout so corrupt
        or missing files never break the listing.

        Args:
            project_dir: Root directory of the project.
            config_data: Parsed contents of config.json.

        Returns:
            ProjectSummary: Enriched summary with metadata fields populated.
        """
        # Last activity: use config.json mtime for speed
        last_activity = None
        try:
            config_path = project_dir / "config.json"
            mtime = config_path.stat().st_mtime
            last_activity = datetime.fromtimestamp(
                mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            pass

        # Parse latest eval result for pass rate and model count
        latest_pass_rate = None
        model_count = 0
        try:
            # Runs only — ``output/`` also holds the metric-audit sidecar and the
            # active-eval marker. The sidecar sorts AFTER its own run file
            # (``.metric-audit.json`` > ``.json``), so globbing every JSON here
            # made the newest sidecar the "latest run": it has no ``model_runs``,
            # which zeroed the model count and dropped the pass rate entirely.
            result_files = eval_run_files(project_dir / "output")
            if result_files:
                latest = json.loads(result_files[-1].read_text())
                # Result files are JSON arrays of model runs
                model_runs = latest if isinstance(latest, list) else latest.get("model_runs", [])
                model_count = len(model_runs)

                # Compute pass rate from all model runs combined
                total_cases = 0
                passed_cases = 0
                for run in model_runs:
                    if not isinstance(run, dict):
                        continue
                    results = run.get("results", [])
                    total_cases += len(results)
                    passed_cases += sum(
                        1 for r in results if r.get("success", False)
                    )
                if total_cases > 0:
                    latest_pass_rate = passed_cases / total_cases

                # Update last_activity if result file is newer
                result_mtime = result_files[-1].stat().st_mtime
                result_ts = datetime.fromtimestamp(
                    result_mtime, tz=timezone.utc
                ).isoformat()
                if last_activity is None or result_ts > last_activity:
                    last_activity = result_ts
        except (OSError, json.JSONDecodeError, KeyError):
            pass

        # Eval liveness comes from the job tracker, the same source
        # ``GET /{project_id}/activity`` reports from, so the list and the
        # project page cannot disagree about whether a project is busy.
        #
        # Deliberately NOT the on-disk active-eval marker: in cloud mode the
        # harness runs in its own task against a snapshot under
        # ``/tmp/workspace/{job_id}``, so that marker is written to ephemeral
        # task storage and never appears in the project directory this process
        # can see. The tracker is correct in both modes, and it is rehydrated
        # from the durable store on restart.
        #
        # Imported inside the method because ``app.services.jobs`` reaches
        # ``app.main`` for the project manager — the same cycle the tracker
        # itself avoids with function-level imports.
        from app.services.jobs import get_job_tracker

        return ProjectSummary(
            id=config_data["id"],
            name=config_data["name"],
            description=config_data.get("description", ""),
            created_at=config_data["created_at"],
            last_activity=last_activity,
            model_count=model_count,
            latest_pass_rate=latest_pass_rate,
            is_running=get_job_tracker().get_active_job(config_data["id"]) is not None,
        )

    def get_project(self, project_id: str) -> ProjectConfig | None:
        """Retrieve full configuration for a specific project.

        Cloud mode: scans every user's prefix to find the project. The
        filesystem scan is bounded by the number of users; an owner-keyed
        lookup would need a metadata-store read instead.

        Args:
            project_id: Unique identifier of the project to retrieve.

        Returns:
            ProjectConfig: Complete project configuration, or None if not
            found, the config is invalid, or ``owner_id`` is missing. A
            record without ``owner_id`` has no reachable owner and is
            treated as absent rather than being adopted by the caller.
        """
        config = self._load_config(project_id)
        # A project flagged for deletion is committed to deletion — treat it
        # as already gone for every normal caller (chat, upload, file ops,
        # path resolution). The delete + reconcile paths use
        # ``load_config_including_deleting`` to still resolve it.
        if config is not None and config.deleting:
            return None
        return config

    def load_config_including_deleting(
        self, project_id: str
    ) -> ProjectConfig | None:
        """Like :meth:`get_project` but also returns projects mid-deletion.

        Used only by the delete handler and the startup reconcile sweep,
        which must resolve a project flagged ``deleting`` to finish removing
        its stores. Every other caller uses :meth:`get_project`, which hides
        ``deleting`` projects.
        """
        return self._load_config(project_id)

    def _load_config(self, project_id: str) -> ProjectConfig | None:
        """Read + parse a project's config.json, or ``None`` if absent/invalid."""
        config_path = self._find_config_path(project_id)
        if config_path is None or not config_path.exists():
            return None
        data = self._read_config_file(config_path)
        if data is None:
            # Return None for corrupted configurations
            return None
        try:
            return ProjectConfig(**data)
        except (KeyError, TypeError, ValueError):
            return None

    def _read_config_file(self, config_path: Path) -> dict | None:
        """Parse a config.json, with its identity taken from where it was found.

        ``config.json`` sits in the project tree, which the project's agent can
        write, so its ``id`` and ``owner_id`` are not evidence of anything. Every
        save derives its destination from these two fields (``save_config``
        builds the path from them, ``put_project`` the record and index keys),
        so trusting them let an edited file redirect a privileged write into
        another user's project. The directory the file was found in is the
        identity: ``{pid}/`` in dev mode, ``users/{owner}/projects/{pid}/`` in
        cloud mode, where the owner comes from the path too. A file that names
        anything else is logged and corrected here, and the next save rewrites
        it with the corrected values.

        Args:
            config_path: A ``config.json`` under the per-mode layout.

        Returns:
            The parsed config with ``id`` (and in cloud mode ``owner_id``) set
            from the path, or ``None`` if the file is unreadable or not a JSON
            object.
        """
        try:
            data = json.loads(config_path.read_text())
        except (ValueError, OSError):
            # ValueError covers JSONDecodeError and a non-UTF-8 file.
            return None
        if not isinstance(data, dict):
            return None
        identity = {"id": config_path.parent.name}
        if _is_cloud_mode():
            identity["owner_id"] = config_path.parent.parent.parent.name
        mismatched = sorted(k for k, v in identity.items() if data.get(k) != v)
        if mismatched:
            logger.warning(
                "[project-manager] config.json names a different %s than its "
                "location; using the location: %s",
                " and ".join(mismatched),
                config_path,
            )
        data.update(identity)
        return data

    def _find_config_path(self, project_id: str) -> Path | None:
        """Locate the config.json for ``project_id`` under the per-mode layout.

        Cloud mode walks ``users/*/projects/{pid}/config.json``; dev mode
        looks at ``{projects_dir}/{pid}/config.json`` directly.
        """
        if _is_cloud_mode():
            users_root = self.projects_dir / "users"
            if not users_root.exists():
                return None
            for user_dir in users_root.iterdir():
                if not user_dir.is_dir():
                    continue
                cand = user_dir / "projects" / project_id / "config.json"
                if cand.exists():
                    return cand
            return None
        return self.projects_dir / project_id / "config.json"

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its contents from the filesystem.

        Removes the entire project directory, and removes ``config.json`` LAST.

        That ordering is load-bearing, not tidiness. ``config.json`` carries the
        ``deleting`` marker, and every path that has to recognise a delete already
        in progress — the authorization dependency, the delete route's retry
        branch, the startup reconcile sweep — resolves the project by reading it.
        A single ``rmtree`` over the whole directory gives no ordering guarantee
        and in practice unlinked the marker early in its walk, so a retry landing
        while multi-gigabyte subtrees were still going found no project and got a
        404. That is the "already removed" response on a perfectly normal delete
        that returning 204 early was supposed to eliminate; the fast return
        shrank the window instead of closing it.

        With the marker removed last, the project stays resolvable for the whole
        cleanup. A retry arriving after the marker is gone still 404s, which is
        correct by then — nothing is left to delete.

        Args:
            project_id: Unique identifier of the project to delete.

        Returns:
            bool: True if project was deleted, False if project didn't exist.
        """
        config_path = self._find_config_path(project_id)
        if config_path is None or not config_path.exists():
            return False
        project_dir = config_path.parent

        for entry in project_dir.iterdir():
            if entry == config_path:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        # The marker, then the now-empty directory.
        config_path.unlink()
        project_dir.rmdir()
        return True

    def save_config(self, config: ProjectConfig) -> None:
        """Persist project configuration to disk.

        Writes the configuration as formatted JSON to the project's config.json file.
        In cloud mode the path is owner-prefixed; ``config.owner_id`` must
        be populated. In dev mode the layout is flat.

        No chown or chmod: the EFS access point squashes every write to
        UID/GID 1000, so file ownership carries no per-user permission (see
        the module docstring).

        Args:
            config: Project configuration to save.
        """
        config_path = self._project_dir(config.id, config.owner_id) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config.model_dump(), indent=2))


def get_project_dir_or_404(project_id: str) -> Path:
    """Resolve a project directory for REST handlers or raise HTTP 404.

    Combines the existence check (via :class:`ProjectManager`) with the
    traversal check (via :func:`resolve_project_dir`) so every handler
    scoped to ``/api/projects/{project_id}/...`` gets both guarantees
    with one call.

    Cloud mode: the path is owner-prefixed,
    ``{projects_dir}/users/{owner_sub}/projects/{pid}/``. Resolution
    routes through :meth:`ProjectManager._project_dir` after the
    existence check so handlers don't need to know the layout.

    Imported lazily via ``from app.main import get_project_manager``
    inside the function body because ``main`` imports this module during
    FastAPI startup — pulling ``main`` at module load time would create
    a circular import.

    Args:
        project_id: Project UUID from the URL path.

    Returns:
        Resolved absolute Path to the project directory.

    Raises:
        HTTPException: 404 when the project does not exist.
        ProjectTraversalError: ``project_id`` would escape the projects
            base directory. Subclass of ``ValueError``; typically surfaces
            as 500 since malformed project_ids shouldn't reach handlers.
    """
    from app.main import get_project_manager
    pm = get_project_manager()
    project = pm.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    project_dir = pm._project_dir(project_id, project.owner_id).resolve()
    if not project_dir.is_relative_to(pm.projects_dir.resolve()):
        from app.services.project_paths import ProjectTraversalError
        raise ProjectTraversalError(f"Invalid project_id: {project_id}")
    return project_dir


def project_dir_for_owner(project_id: str, owner_id: str | None) -> Path:
    """Resolve a project directory from an owner the caller already holds.

    For callers that cannot look the project up, because the lookup would
    deliberately fail. :func:`resolve_project_dir` asks ``get_project``, and
    ``get_project`` returns ``None`` for a project flagged ``deleting`` — by
    design, so ordinary read paths treat it as gone. It then falls back to the
    flat layout, which in cloud mode is ``{projects_dir}/{pid}`` rather than
    ``users/{owner}/projects/{pid}``: a different directory, and therefore a
    different SDK project key. Deletion cleanup needs the real one.

    Args:
        project_id: Project UUID.
        owner_id: Owner's Cognito ``sub``. Required in cloud mode.

    Returns:
        Resolved absolute Path to the project directory.

    Raises:
        ValueError: cloud mode and ``owner_id`` is None — the path cannot be
            derived, and guessing would target another prefix.
        ProjectTraversalError: ``project_id`` would escape the base directory.
    """
    from app.main import get_project_manager
    pm = get_project_manager()
    project_dir = pm._project_dir(project_id, owner_id).resolve()
    if not project_dir.is_relative_to(pm.projects_dir.resolve()):
        from app.services.project_paths import ProjectTraversalError
        raise ProjectTraversalError(f"Invalid project_id: {project_id}")
    return project_dir
