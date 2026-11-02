# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the dev-mode platform adapters.

Verifies the Protocol contracts against the filesystem-backed
implementations that dev mode runs on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.platform import Platform, bootstrap, set_platform
from app.platform.auth import DEFAULT_LOCAL_DEV_USER_ID
from app.platform.dev_local import (
    FilesystemMetadataStore,
    LocalAuth,
    NoOpTokenVending,
    build_dev_platform,
)
from app.platform.metadata_store import ConcurrentUpdateError


@pytest.fixture
def projects_dir(tmp_path: Path) -> Path:
    """Throwaway projects root for each test."""
    return tmp_path / "projects"


@pytest.fixture
def metadata_store(projects_dir: Path) -> FilesystemMetadataStore:
    return FilesystemMetadataStore(projects_dir=projects_dir)


# --- Metadata store --------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_store_project_round_trip(
    metadata_store: FilesystemMetadataStore,
) -> None:
    """put_project / get_project / delete_project lifecycle."""
    user = DEFAULT_LOCAL_DEV_USER_ID
    proj = {
        "id": "p1",
        "name": "test",
        "description": "",
        "created_at": "2026-05-15T00:00:00Z",
        "owner_id": user,
    }
    await metadata_store.put_project(proj)
    fetched = await metadata_store.get_project(user, "p1")
    assert fetched is not None
    assert fetched["name"] == "test"
    listed = await metadata_store.list_user_projects(user)
    assert any(p["id"] == "p1" for p in listed)
    assert await metadata_store.delete_project(user, "p1")
    assert await metadata_store.get_project(user, "p1") is None


@pytest.mark.asyncio
async def test_metadata_store_optimization_state_optimistic_lock(
    metadata_store: FilesystemMetadataStore,
) -> None:
    """Conditional write rejects mismatched expected_version."""
    user, project = "u1", "p2"
    # Seed a config so the project dir exists.
    await metadata_store.put_project({
        "id": project,
        "name": "x",
        "description": "",
        "created_at": "2026-05-15T00:00:00Z",
        "owner_id": user,
    })
    v1 = await metadata_store.put_optimization_state(
        user, project, {"status": "running"}, expected_version=0
    )
    assert v1 == 1
    # Stale write must fail.
    with pytest.raises(ConcurrentUpdateError):
        await metadata_store.put_optimization_state(
            user, project, {"status": "converged"}, expected_version=0
        )
    # Correct version succeeds.
    v2 = await metadata_store.put_optimization_state(
        user, project, {"status": "converged"}, expected_version=1
    )
    assert v2 == 2


# --- Auth ------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_auth_owner_passes(metadata_store: FilesystemMetadataStore) -> None:
    auth = LocalAuth(metadata_store)
    user = DEFAULT_LOCAL_DEV_USER_ID
    await metadata_store.put_project({
        "id": "p3",
        "name": "x",
        "description": "",
        "created_at": "2026-05-15T00:00:00Z",
        "owner_id": user,
    })
    # Owner: read or write — both fine, no exception.
    await auth.require_access(user, "p3", "read")
    await auth.require_access(user, "p3", "write")


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["read", "write"])
async def test_local_auth_denies_everyone_but_the_owner(
    metadata_store: FilesystemMetadataStore, action: str,
) -> None:
    """Only the owner gets in, even if the record still lists ``viewer_ids``.

    Records written while sharing existed can carry that field. It must grant
    nothing now that the only access rule is ownership.
    """
    auth = LocalAuth(metadata_store)
    await metadata_store.put_project({
        "id": "p4",
        "name": "x",
        "description": "",
        "created_at": "2026-05-15T00:00:00Z",
        "owner_id": "owner",
        "viewer_ids": ["other-user"],
    })
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await auth.require_access("other-user", "p4", action)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_local_auth_unknown_user_404(metadata_store: FilesystemMetadataStore) -> None:
    """Missing project returns 404 (don't leak existence to non-owners)."""
    auth = LocalAuth(metadata_store)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await auth.require_access("anyone", "missing-project", "read")
    assert exc.value.status_code == 404


# --- Bootstrap -------------------------------------------------------


def test_bootstrap_returns_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bootstrap()`` returns a Platform with its fields populated."""
    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
    set_platform(None)  # force fresh build
    p = bootstrap()
    assert isinstance(p, Platform)
    assert p.metadata_store is not None
    # session_store is intentionally None in dev (SDK default)
    assert p.token_vending is not None
    assert p.auth is not None


def test_bootstrap_unknown_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_STUDIO_MODE", "weird")
    set_platform(None)
    with pytest.raises(ValueError, match="EVAL_STUDIO_MODE"):
        bootstrap()
