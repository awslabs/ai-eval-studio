# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The REST launch routes must pass the same admission gate as MCP dispatch.

The cost kill switch and the per-user / account job caps lived only in
``mcp_tools.dispatch``, so they applied to launches the coordinator made and to
nothing else. The REST start and resume routes — which is what the UI calls —
enforced none of them, and a REST-started run also held no durable eval lock at
any point, because the lock keys were never threaded into the loop's config.
"""
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app

pytestmark = pytest.mark.asyncio


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


@pytest.fixture(autouse=True)
def _no_agent_turn_on_success():
    """Stub the "optimization started" chat notification.

    On the admitted path the route tells the coordinator the loop began, which
    submits a real turn and spawns the SDK CLI. Left alone it reaches for the
    network and the test pays a multi-minute timeout at teardown for something no
    test here asserts.
    """
    with patch(
        "app.services.completion_turn.request_terminal_message", new=AsyncMock()
    ):
        yield


async def _project_with_harness(client) -> str:
    """Create a project with a harness the start route accepts.

    Reuses the optimization-API scaffolder so this file cannot drift from what
    ``load_harness_config`` actually parses.
    """
    import app.main as main_module

    from .test_optimization_api import _scaffold_project

    resp = await client.post("/api/projects", json={"name": "Opt"})
    project_id = resp.json()["id"]
    _scaffold_project(Path(main_module.PROJECTS_DIR), project_id, False)
    return project_id


def _store_admitting_everything() -> MagicMock:
    """A store double that admits every launch and REMEMBERS what it was told.

    The optimization state has to be readable back. A double whose
    ``get_optimization_state`` always answers None cannot show whether the route's
    compensation repaired the state, because the compensation reads before it
    writes and finds nothing to repair — so a test against that double proves the
    lock was released and silently nothing else.
    """
    store = MagicMock()

    persisted: dict[str, Any] = {"state": None, "version": 0}

    async def _get_optimization_state(_user_id, _project_id):
        if persisted["state"] is None:
            return None
        return {**persisted["state"], "version": persisted["version"]}

    async def _put_optimization_state(
        _user_id, _project_id, state, *, expected_version=None
    ):
        persisted["state"] = dict(state)
        persisted["version"] += 1
        return persisted["version"]

    store.get_optimization_state = AsyncMock(side_effect=_get_optimization_state)
    store.put_optimization_state = AsyncMock(side_effect=_put_optimization_state)
    store.persisted = persisted
    store.count_active_jobs_for_owner = AsyncMock(return_value=0)
    store.list_non_terminal_jobs = AsyncMock(return_value=[])
    store.claim_eval_lock = AsyncMock(return_value=True)
    store.release_eval_lock = AsyncMock(return_value=None)
    return store


async def test_rest_start_is_refused_when_the_kill_switch_is_active(client, monkeypatch):
    """Cost control must stop the UI's launch path, not just the agent's."""
    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ), patch(
        "app.platform.arch_split.is_cost_kill_switch_active", return_value=True
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 429, resp.text
    assert "cost_kill_switch_active" in resp.text
    store.claim_eval_lock.assert_not_awaited()


async def test_rest_start_is_refused_at_the_per_user_cap(client):
    """The per-user job cap applies to the route the UI calls."""
    from app.services.job_admission import PER_USER_MAX_JOBS

    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()
    store.count_active_jobs_for_owner = AsyncMock(return_value=PER_USER_MAX_JOBS)

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 429, resp.text
    assert "user_job_limit" in resp.text


async def test_rest_start_is_refused_at_the_account_cap(client):
    """The account backstop applies too."""
    from app.services.job_admission import ACCOUNT_MAX_JOBS

    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()
    store.list_non_terminal_jobs = AsyncMock(
        return_value=[{"job_id": f"j{i}"} for i in range(ACCOUNT_MAX_JOBS)]
    )

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 429, resp.text
    assert "account_job_limit" in resp.text


async def test_rest_start_is_refused_when_another_eval_holds_the_lock(client):
    """A live eval elsewhere in the deployment must block the loop.

    The route's own guard reads a process-local tracker, so it cannot see an eval
    dispatched from the AgentCore runtime or one that survived a restart. The
    durable lock is what actually enforces one-eval-per-project.
    """
    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()
    store.claim_eval_lock = AsyncMock(return_value=False)

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 409, resp.text


async def test_a_refused_start_writes_no_running_state(client):
    """A rejected launch must leave nothing behind.

    Admission runs BEFORE the state write for this reason: opt_repo.create
    persists status="running", so admitting afterwards returned 429 over a durable
    record claiming the project was optimizing with no loop behind it — a run the
    UI would show as live and that could never finish.
    """
    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()
    store.count_active_jobs_for_owner = AsyncMock(return_value=99)

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 429, resp.text
    store.put_optimization_state.assert_not_awaited()


async def test_a_failed_spawn_releases_the_lock_and_settles_the_state(
    client, monkeypatch
):
    """Between admission and a live loop, this route owns the compensation.

    Nothing else can clean up: the lock's owner is a loop that never started, and
    the state already says running.
    """
    from app.services import optimization_loop_orchestrator as loop_mod

    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()

    def _explode(_config):
        raise RuntimeError("cannot spawn")

    monkeypatch.setattr(loop_mod, "run_optimization_loop", _explode)

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        with pytest.raises(RuntimeError):
            await client.post(f"/api/projects/{project_id}/optimization/start")

    store.release_eval_lock.assert_awaited_once()
    # Releasing the lock is only half of it. The state was already persisted as
    # running, and the loop that would normally settle it does not exist — so
    # leaving it there shows the UI a run that is permanently in flight and can
    # never finish. Assert the compensation actually repaired it.
    assert store.persisted["state"] is not None, "the running state was persisted"
    assert store.persisted["state"]["status"] == "cancelled", (
        "an abandoned launch must leave the state terminal, not running"
    )


async def test_rest_start_hands_the_lock_to_the_loop(client, monkeypatch):
    """The loop's finalizer can only release a lock it was told about."""
    from app.services import optimization_loop_orchestrator as loop_mod

    project_id = await _project_with_harness(client)
    store = _store_admitting_everything()
    captured: dict = {}

    async def _fake_loop(iteration_config):
        captured["config"] = iteration_config

    monkeypatch.setattr(loop_mod, "run_optimization_loop", _fake_loop)

    with patch(
        "app.api.optimization.resolve_arch_split", new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "app.platform.get_platform", return_value=MagicMock(metadata_store=store)
    ):
        resp = await client.post(f"/api/projects/{project_id}/optimization/start")

    assert resp.status_code == 200, resp.text
    store.claim_eval_lock.assert_awaited_once()

    from .async_wait import until

    await until(lambda: "config" in captured)
    config = captured["config"]
    assert config["eval_lock_holder_id"] == f"optimization-{project_id}", (
        "the loop must receive the holder id the route claimed under, or its "
        "release is a no-op and the lock outlives the run"
    )
    assert config["eval_lock_owner_id"]
