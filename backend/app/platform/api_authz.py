# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency helpers for project-scoped authorization.

Routes that operate on a specific project should declare a dependency
on :func:`require_project_read` or :func:`require_project_write` so the
authorization check is enforced uniformly. The dependency:

1. Reads ``user_id`` from the per-request :class:`RequestContextValue`
   set by the auth middleware.
2. Calls ``platform.auth.require_access(user_id, project_id, action)``.
3. Returns ``user_id`` so handlers can use it for downstream calls
   (e.g. resolving the project directory for the authorized user).

The dependency raises ``HTTPException(404)`` for missing projects and
``HTTPException(403)`` for unauthorized access — handlers don't need
to repeat the check.
"""
from __future__ import annotations

import re

from fastapi import Depends, HTTPException, Request

from . import get_platform
from .types import UserId, current_user_id

# Permissive validator for project_ids embedded in URL paths. Real
# projects always use UUIDs, but tests scaffold projects with shorter
# string ids; both must work. The check here rejects values that
# would obviously be path injection attempts (``/`` ``..`` whitespace)
# and caps the length so a malformed payload can't probe DB / log
# subsystems with a megabyte-sized "project_id". 64 chars accommodates
# UUIDs (36) with comfortable margin.
# Stricter UUID validation lives in ``api/projects.py`` for the
# user-facing CRUD endpoints.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_project_id_format(project_id: str) -> None:
    if not project_id or not _PROJECT_ID_RE.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")


async def _require(project_id: str, action: str, request: Request) -> UserId:
    """Shared body of :func:`require_project_read`/``_write``."""
    _validate_project_id_format(project_id)
    try:
        user_id = current_user_id()
    except RuntimeError as exc:
        # No request context — auth middleware didn't run. This is a
        # programmer bug, not an unauthenticated user, so 500 is right.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    platform = get_platform()
    await platform.auth.require_access(user_id, project_id, action)  # type: ignore[arg-type]
    return user_id


async def require_project_read(
    project_id: str, request: Request
) -> UserId:
    """Verify the requester may read ``project_id``; return ``user_id``."""
    return await _require(project_id, "read", request)


async def require_project_write(
    project_id: str, request: Request
) -> UserId:
    """Verify the requester may write ``project_id``; return ``user_id``."""
    return await _require(project_id, "write", request)


__all__ = ["require_project_read", "require_project_write"]
