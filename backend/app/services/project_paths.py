# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized project path resolution and safe-path checking.

The MCP tool modules under ``services/mcp_tools/`` and the REST handlers
under ``api/`` both call ``resolve_project_dir()`` here so resolution
stays in one place. The existence check is intentionally NOT part of
resolution — MCP tools sometimes create directories on first use; REST
handlers add their own existence check via ``ProjectManager.get_project()``.

Cloud mode lays out projects per-user under
``{projects_dir}/users/{owner_sub}/projects/{pid}/``. This module
delegates layout decisions to :class:`ProjectManager` so REST handlers
and MCP tools that already call ``resolve_project_dir(pid)`` get the
right path automatically.
"""
from __future__ import annotations

from pathlib import Path


class ProjectTraversalError(ValueError):
    """Raised when a project_id would escape the projects base directory.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    continue to work.
    """


def resolve_project_dir(project_id: str, base_dir: Path | None = None) -> Path:
    """Resolve a project directory and validate no path traversal.

    Does NOT verify that the directory exists — callers that need that
    guarantee should call ``ProjectManager.get_project()`` first (REST
    handlers) or check ``.exists()`` explicitly.

    Cloud mode: looks up the project's ``owner_id`` via
    :class:`ProjectManager.get_project` and returns
    ``{base_dir}/users/{owner}/projects/{pid}``. Dev mode keeps the flat
    ``{base_dir}/{pid}`` layout.

    ``base_dir`` defaults to ``get_project_manager().projects_dir`` for
    runtime use. Tests pass ``base_dir`` explicitly so they don't need to
    boot the ProjectManager singleton; tests run in dev mode so the flat
    layout is used.

    Args:
        project_id: Project UUID string (or any path component to resolve).
        base_dir: Base directory to resolve under. Defaults to the
            ProjectManager's projects directory.

    Returns:
        Resolved absolute Path for the project directory.

    Raises:
        ProjectTraversalError: If project_id would escape the base directory.
    """
    if base_dir is None:
        # Lazy import to avoid import cycles (app.main imports many services).
        from app.main import get_project_manager
        pm = get_project_manager()
        base_dir = pm.projects_dir
        # Cloud-mode user-aware layout: ask the project manager directly.
        # _project_dir owns the cloud/dev layout decision so this module
        # stays a thin path-safety wrapper.
        from app.services.project_manager import _is_cloud_mode
        if _is_cloud_mode():
            project = pm.get_project(project_id)
            if project is None:
                # Project doesn't exist yet (e.g. about to be created).
                # Fall through to flat resolution under base_dir; callers
                # that need an existing project handle 404 themselves.
                resolved = (base_dir / project_id).resolve()
            else:
                resolved = pm._project_dir(project_id, project.owner_id).resolve()
            if not resolved.is_relative_to(base_dir.resolve()):
                raise ProjectTraversalError(f"Invalid project_id: {project_id}")
            return resolved
    resolved = (base_dir / project_id).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise ProjectTraversalError(f"Invalid project_id: {project_id}")
    return resolved


def safe_path(base_dir: Path, untrusted_name: str) -> Path | None:
    """Resolve ``untrusted_name`` under ``base_dir`` or return ``None`` on escape.

    Callers decide how to handle a rejected path (most return an MCP error);
    it does not raise, so call sites stay terse.

    Args:
        base_dir: The directory the resolved path must stay within.
        untrusted_name: Untrusted filename or relative path component.

    Returns:
        Resolved Path if safe, None if it would escape base_dir.
    """
    resolved = (base_dir / untrusted_name).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        return None
    return resolved


def safe_basename_path(base_dir: Path, untrusted_name: str) -> Path | None:
    """Stricter ``safe_path`` for endpoints that take a single basename.

    Use when the input represents one filename in a directory — never a
    nested path. Rejects empty input, path separators, hidden files,
    embedded NULs, and any name that doesn't survive
    ``PurePosixPath(name).name == name`` (which catches every
    multi-component shape). Then runs the standard ``safe_path``
    containment check, so symlinks pointing outside ``base_dir`` are
    rejected too.

    Single source of truth for the eval-runs DELETE endpoint and the
    matching ``delete_run`` MCP tool — they previously had divergent
    inline validators.

    Args:
        base_dir: Directory the resolved path must stay within.
        untrusted_name: Untrusted bare filename.

    Returns:
        Resolved ``Path`` when safe, ``None`` on any rejection.
    """
    from pathlib import PurePosixPath

    if (
        not isinstance(untrusted_name, str)
        or not untrusted_name
        or "/" in untrusted_name
        or "\\" in untrusted_name
        or "\x00" in untrusted_name
        or untrusted_name.startswith(".")
        or PurePosixPath(untrusted_name).name != untrusted_name
    ):
        return None
    return safe_path(base_dir, untrusted_name)
