# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


async def _drain_delete_cleanup(project_id: str) -> None:
    """Await the background store-cleanup task a DELETE scheduled, if any.

    The delete route returns 204 immediately and finishes store removal
    (transcript → DDB → rmtree) in a background task. Tests await that task
    here so they can assert the end state deterministically.
    """
    from app.api.projects import _delete_cleanup_tasks

    task = _delete_cleanup_tasks.get(project_id)
    if task is not None:
        await task


@pytest.fixture
def app(tmp_path):
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()

    main_module._project_manager = None
    application = create_app()
    yield application

    main_module._project_manager = None
    main_module.PROJECTS_DIR = original


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestProjectsAPI:
    async def test_list_projects_empty(self, client):
        resp = await client.get("/api/projects")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_create_project(self, client):
        resp = await client.post("/api/projects", json={"name": "My Eval", "description": "Testing"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Eval"
        assert "id" in data

    async def test_create_project_no_name(self, client):
        resp = await client.post("/api/projects", json={"name": ""})
        assert resp.status_code == 422

    async def test_get_project(self, client):
        create_resp = await client.post("/api/projects", json={"name": "Test"})
        project_id = create_resp.json()["id"]
        resp = await client.get(f"/api/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test"

    async def test_get_project_invalid_id(self, client):
        """Non-UUID project ID returns 400."""
        resp = await client.get("/api/projects/nonexistent")
        assert resp.status_code == 400

    async def test_get_project_not_found(self, client):
        """Valid UUID but nonexistent project returns 404."""
        resp = await client.get("/api/projects/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    async def test_delete_project(self, client):
        create_resp = await client.post("/api/projects", json={"name": "To Delete"})
        project_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/projects/{project_id}")
        assert resp.status_code == 204
        await _drain_delete_cleanup(project_id)
        resp = await client.get(f"/api/projects/{project_id}")
        assert resp.status_code == 404

    async def test_delete_project_removes_tree_on_disk(self, client):
        """Delete physically removes the project directory and its contents."""
        import app.main as main_module

        create_resp = await client.post("/api/projects", json={"name": "Tree Gone"})
        project_id = create_resp.json()["id"]
        project_dir = main_module.PROJECTS_DIR / project_id
        # Seed a nested file so we prove the whole tree is removed, not just config.
        (project_dir / "src").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "config.py").write_text("X = 1\n")
        assert project_dir.exists()

        resp = await client.delete(f"/api/projects/{project_id}")
        assert resp.status_code == 204
        await _drain_delete_cleanup(project_id)
        assert not project_dir.exists()

    async def test_delete_project_with_session_id(self, client):
        """Delete succeeds when the project has a session_id.

        Exercises the transcript-delete branch — a no-op in dev mode
        (session_store is None) but it must not raise.
        """
        import json

        import app.main as main_module

        create_resp = await client.post("/api/projects", json={"name": "Has Session"})
        project_id = create_resp.json()["id"]
        # Stamp a session_id into config.json so the delete path reads it.
        config_path = main_module.PROJECTS_DIR / project_id / "config.json"
        config = json.loads(config_path.read_text())
        config["session_id"] = "sess-abc-123"
        config_path.write_text(json.dumps(config))

        resp = await client.delete(f"/api/projects/{project_id}")
        assert resp.status_code == 204
        await _drain_delete_cleanup(project_id)
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_delete_returns_immediately_and_hides_before_cleanup(
        self, client
    ):
        """204 + hidden the instant the marker is written — BEFORE the slow
        store cleanup runs.

        This is the fix for the held-response double-fire: the request must
        not wait for the (potentially tens-of-seconds) rmtree. We assert the
        project is already hidden from list/get while the background cleanup
        is still pending, then drain it and confirm the tree is gone.
        """
        import app.main as main_module
        from app.api.projects import _delete_cleanup_tasks

        create_resp = await client.post("/api/projects", json={"name": "Fast"})
        project_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/projects/{project_id}")
        assert resp.status_code == 204
        # Cleanup is still scheduled (we haven't drained it), yet the project
        # is ALREADY invisible — the marker, not the slow cleanup, drives
        # visibility.
        assert (await client.get(f"/api/projects/{project_id}")).status_code == 404
        assert all(
            p["id"] != project_id
            for p in (await client.get("/api/projects")).json()
        )

        await _drain_delete_cleanup(project_id)
        assert project_id not in _delete_cleanup_tasks  # task cleared itself
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_delete_double_fire_does_not_404(self, client):
        """Two DELETEs for the same project both return 204 — never 404.

        The failure: while the first delete's slow cleanup runs, a second DELETE
        could 404 (DDB auth pointer already gone) → a scary 'already removed'
        toast on a normal delete. The marker is set first, so a re-fire
        converges to 204.
        """
        import app.main as main_module

        create_resp = await client.post("/api/projects", json={"name": "Double"})
        project_id = create_resp.json()["id"]

        first = await client.delete(f"/api/projects/{project_id}")
        # Second DELETE before draining — mirrors a re-click / LB retry while
        # the first cleanup is still in flight.
        second = await client.delete(f"/api/projects/{project_id}")
        assert first.status_code == 204
        assert second.status_code == 204

        await _drain_delete_cleanup(project_id)
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_delete_keeps_marker_when_a_store_fails(self, client):
        """A failing store leaves the ``deleting`` marker set so a retry /
        the startup sweep finishes it. The user already got 204 and the
        project is hidden — never a visible zombie, never an orphan.
        """
        import dataclasses
        import json

        import app.main as main_module
        from app.platform import get_platform, set_platform

        create_resp = await client.post("/api/projects", json={"name": "Wedged"})
        project_id = create_resp.json()["id"]
        config_path = main_module.PROJECTS_DIR / project_id / "config.json"
        config = json.loads(config_path.read_text())
        config["session_id"] = "sess-xyz"
        config_path.write_text(json.dumps(config))

        # Session-store delete raises so background cleanup can't finish.
        class _FailingStore:
            bucket = "sessions"

            async def delete(self, key):
                raise RuntimeError("S3 down")

        original = get_platform()
        set_platform(dataclasses.replace(original, session_store=_FailingStore()))
        try:
            resp = await client.delete(f"/api/projects/{project_id}")
            # 204 immediately + hidden, even though cleanup will fail.
            assert resp.status_code == 204
            assert (await client.get(f"/api/projects/{project_id}")).status_code == 404
            await _drain_delete_cleanup(project_id)  # fails internally, swallowed
        finally:
            set_platform(original)

        # The tree (with the marker) survives so it's recoverable.
        assert (main_module.PROJECTS_DIR / project_id).exists()
        marker = json.loads(config_path.read_text())
        assert marker.get("deleting") is True

        # Retry with a healthy platform now completes the delete.
        retry = await client.delete(f"/api/projects/{project_id}")
        assert retry.status_code == 204
        await _drain_delete_cleanup(project_id)
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_startup_sweep_completes_interrupted_delete(self, client):
        """A project left flagged ``deleting`` by a crash is finished on the
        next startup by the reconcile sweep — data guaranteed gone.
        """
        import json

        import app.main as main_module

        create_resp = await client.post("/api/projects", json={"name": "Crashed"})
        project_id = create_resp.json()["id"]
        # Simulate a crash AFTER the intent marker but BEFORE store removal.
        config_path = main_module.PROJECTS_DIR / project_id / "config.json"
        config = json.loads(config_path.read_text())
        config["deleting"] = True
        config_path.write_text(json.dumps(config))

        # Already invisible to the UI while flagged.
        assert all(
            p["id"] != project_id
            for p in (await client.get("/api/projects")).json()
        )

        # The startup sweep finishes it.
        await main_module._recover_pending_deletes()
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_delete_project_invalid_id(self, client):
        """Non-UUID project ID returns 400."""
        resp = await client.delete("/api/projects/nonexistent")
        assert resp.status_code == 400

    async def test_delete_project_not_found(self, client):
        """Valid UUID, gone from BOTH tree and DDB → 404."""
        resp = await client.delete("/api/projects/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestProjectRename:
    """``PATCH`` accepts ``name`` and ``description``.

    Before this, the create dialog was the only moment either could be set and
    there was no route to change them afterwards — so a project kept whatever
    label it was given before it existed.
    """

    async def _create(self, client, **kwargs):
        """Create a project and return its id."""
        payload = {"name": "Original"} | kwargs
        resp = await client.post("/api/projects", json=payload)
        assert resp.status_code == 201
        return resp.json()["id"]

    async def test_name_can_be_changed(self, client):
        project_id = await self._create(client)

        resp = await client.patch(f"/api/projects/{project_id}", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"

        # Persisted, not just echoed back.
        assert (await client.get(f"/api/projects/{project_id}")).json()["name"] == "Renamed"

    async def test_description_can_be_changed(self, client):
        project_id = await self._create(client)

        resp = await client.patch(
            f"/api/projects/{project_id}", json={"description": "What this evaluates"}
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "What this evaluates"

    async def test_an_empty_description_clears_it(self, client):
        """Empty string is a value, not an omission.

        A truthiness check in the handler would silently make clearing a no-op.
        """
        project_id = await self._create(client, description="To be removed")

        resp = await client.patch(f"/api/projects/{project_id}", json={"description": ""})
        assert resp.status_code == 200
        assert resp.json()["description"] == ""

    async def test_an_empty_name_is_rejected(self, client):
        """Bounds match creation, so a name that can't be created can't be set."""
        project_id = await self._create(client)

        resp = await client.patch(f"/api/projects/{project_id}", json={"name": ""})
        assert resp.status_code == 422
        assert (await client.get(f"/api/projects/{project_id}")).json()["name"] == "Original"

    async def test_an_overlong_name_is_rejected(self, client):
        project_id = await self._create(client)

        resp = await client.patch(f"/api/projects/{project_id}", json={"name": "x" * 201})
        assert resp.status_code == 422

    async def test_omitted_fields_are_left_alone(self, client):
        """A one-field patch must not blank the fields it doesn't mention."""
        project_id = await self._create(client, description="Keep me")

        resp = await client.patch(f"/api/projects/{project_id}", json={"name": "New Name"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "New Name"
        assert body["description"] == "Keep me"

    async def test_renaming_does_not_move_the_project(self, client):
        """Every path and durable key derives from the id, never the name.

        If a rename ever relocated storage, the project would 404 right after
        being renamed — so fetching by the original id has to keep working.
        """
        project_id = await self._create(client)
        await client.patch(f"/api/projects/{project_id}", json={"name": "Totally Different"})

        resp = await client.get(f"/api/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id

    async def test_patching_a_missing_project_is_404(self, client):
        resp = await client.patch(
            "/api/projects/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", json={"name": "x"}
        )
        assert resp.status_code == 404


class TestDeleteMarkerOutlivesCleanup:
    """The ``deleting`` marker must stay resolvable for the whole cleanup.

    ``config.json`` holds the marker, and every path that has to recognise an
    in-progress delete resolves the project by reading it — the authorization
    dependency, this route's retry branch, and the startup reconcile sweep. If
    anything removes it before the tree step, a retry arriving mid-cleanup gets
    404 and the user sees "already removed" on a normal delete.

    That is what happened: the dev metadata store is tree-backed, so its
    "record" IS ``config.json``, and the record delete was guarded on
    ``owner_id`` alone — which dev projects have. It ran, unlinking the marker
    before the tree removal it was documented to run after.

    These tests hold the cleanup open instead of racing it, so they fail
    deterministically rather than depending on thread scheduling.
    """

    async def test_a_retry_mid_cleanup_returns_204(self, client, monkeypatch):
        import threading

        import app.api.projects as projects_module

        release = threading.Event()
        entered = threading.Event()
        pm = projects_module._get_pm()
        real_delete = pm.delete_project

        def blocked_delete(project_id):
            """Park the tree removal in its worker thread.

            A ``threading`` event, not an asyncio one: this runs via
            ``asyncio.to_thread``, so the event loop stays free to answer the
            retry — which is the situation being tested.
            """
            entered.set()
            assert release.wait(timeout=5), "test never released the tree step"
            return real_delete(project_id)

        monkeypatch.setattr(pm, "delete_project", blocked_delete)

        create_resp = await client.post("/api/projects", json={"name": "Retry Me"})
        project_id = create_resp.json()["id"]

        first = await client.delete(f"/api/projects/{project_id}")
        assert first.status_code == 204
        # Wait for the cleanup to reach the tree step without blocking the loop.
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "cleanup never reached the tree step"

        # Cleanup is parked at the tree step; the project must still resolve.
        second = await client.delete(f"/api/projects/{project_id}")
        assert second.status_code == 204, (
            "a retry while cleanup is in flight must not 404 — the marker was "
            "removed before the tree step"
        )

        release.set()
        await _drain_delete_cleanup(project_id)

    async def test_the_marker_survives_until_the_tree_is_empty(self, client, monkeypatch):
        """``delete_project`` unlinks ``config.json`` after the subtrees.

        Asserted on the filesystem at each ``rmtree`` rather than on call order,
        so it holds however the removal is expressed.
        """
        import shutil

        import app.main as main_module

        create_resp = await client.post("/api/projects", json={"name": "Ordered"})
        project_id = create_resp.json()["id"]
        config_path = main_module.PROJECTS_DIR / project_id / "config.json"

        marker_present_at_each_rmtree = []
        real_rmtree = shutil.rmtree

        def spy(path, *a, **k):
            marker_present_at_each_rmtree.append(config_path.exists())
            return real_rmtree(path, *a, **k)

        monkeypatch.setattr(shutil, "rmtree", spy)

        await client.delete(f"/api/projects/{project_id}")
        await _drain_delete_cleanup(project_id)

        assert marker_present_at_each_rmtree, "no subtree removal observed"
        assert all(marker_present_at_each_rmtree), (
            "config.json was already gone during subtree removal, so a retry "
            "arriving then could not resolve the project"
        )
        assert not (main_module.PROJECTS_DIR / project_id).exists()

    async def test_dev_does_not_delete_the_metadata_record_separately(
        self, client, monkeypatch
    ):
        """In dev the tree delete IS the record delete; a second one is the bug."""
        from app.platform import get_platform

        calls = []
        store = get_platform().metadata_store

        async def spy(user_id, project_id):
            calls.append(project_id)
            return True

        monkeypatch.setattr(store, "delete_project", spy)

        create_resp = await client.post("/api/projects", json={"name": "No Double Delete"})
        project_id = create_resp.json()["id"]
        await client.delete(f"/api/projects/{project_id}")
        await _drain_delete_cleanup(project_id)

        assert calls == []
        assert not (
            __import__("app.main", fromlist=["x"]).PROJECTS_DIR / project_id
        ).exists()
