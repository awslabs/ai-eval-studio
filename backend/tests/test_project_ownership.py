# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify ProjectConfig.owner_id is stamped at creation and filters listings.

Cloud mode relies on ``owner_id`` for every authorization
decision, so this is the canonical regression test for the field.
Three things must hold:

1. POST ``/api/projects`` writes ``current_user_id()`` to the on-disk
   config so it survives restarts and reflects the requester.
2. GET ``/api/projects`` filters by owner — User A never sees User B's
   projects.
3. On-disk records with no ``owner_id`` field are unreachable: the runtime
   never back-fills an owner, so no user — not even ``LOCAL_DEV_USER_ID`` —
   sees them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.platform import set_platform
from app.platform.dev_local import build_dev_platform


@pytest.fixture
def app(tmp_path, monkeypatch):
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    set_platform(None)
    application = create_app()
    yield application
    main_module._project_manager = None
    main_module.PROJECTS_DIR = original
    set_platform(None)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def with_user(monkeypatch):
    """Return a callable that sets ``LOCAL_DEV_USER_ID`` for the next request.

    Also rebuilds the platform singleton, so each user switch runs against
    a freshly built dev platform. ``LocalAuth.extract_user_id`` reads the
    env var on each call, so the new id takes effect either way.
    """
    def _set(uid: str) -> None:
        monkeypatch.setenv("LOCAL_DEV_USER_ID", uid)
        set_platform(None)
        set_platform(build_dev_platform())
    return _set


async def test_create_project_stamps_owner_id_on_disk(client, with_user, app):
    """POST writes owner_id to config.json so it survives restarts."""
    with_user("user-alice")
    resp = await client.post("/api/projects", json={"name": "Alice's eval"})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["owner_id"] == "user-alice"
    assert "viewer_ids" not in body

    # Re-read straight from disk — the field must be persisted, not just
    # echoed by the response model.
    import app.main as main_module
    config_path = main_module.PROJECTS_DIR / body["id"] / "config.json"
    saved = json.loads(config_path.read_text())
    assert saved["owner_id"] == "user-alice"


async def test_list_projects_filters_by_owner(client, with_user, app):
    """User A creates; User B lists; User B sees nothing."""
    with_user("user-alice")
    create_resp = await client.post("/api/projects", json={"name": "Alice's"})
    assert create_resp.status_code == 201

    with_user("user-bob")
    list_resp = await client.get("/api/projects")
    assert list_resp.status_code == 200
    # Bob can't see Alice's project.
    bob_projects = list_resp.json()
    assert all(p["name"] != "Alice's" for p in bob_projects), bob_projects

    with_user("user-alice")
    list_resp = await client.get("/api/projects")
    assert list_resp.status_code == 200
    alice_projects = list_resp.json()
    assert any(p["name"] == "Alice's" for p in alice_projects)


async def test_record_without_owner_id_is_skipped(
    client, with_user, app, tmp_path,
):
    """A config.json without owner_id is unreachable via authz.

    The runtime never back-fills a missing ``owner_id``: guessing an owner
    would hand one user's project to another, so such a record stays
    unreachable instead.
    """
    import app.main as main_module
    legacy_id = "legacy-0000-0000-0000-000000000000"
    legacy_dir = main_module.PROJECTS_DIR / legacy_id
    legacy_dir.mkdir()
    (legacy_dir / "config.json").write_text(json.dumps({
        "id": legacy_id,
        "name": "Legacy project",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        # No owner_id — pre-multi-user record on disk.
    }))

    with_user("local-dev")
    list_resp = await client.get("/api/projects")
    assert list_resp.status_code == 200
    names = [p["name"] for p in list_resp.json()]
    assert "Legacy project" not in names

    with_user("someone-else")
    list_resp = await client.get("/api/projects")
    names = [p["name"] for p in list_resp.json()]
    assert "Legacy project" not in names
