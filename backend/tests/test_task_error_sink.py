# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``attach_error_sink`` — the background-task crash backstop.

The sink used to publish on the (now-deleted) ``ProjectEventBus``; it now
publishes on ``ws_hub.publish_project``.
``_resolve_owner_id`` looks up the project's ``owner_id`` via the
``ProjectManager`` — tests seed a real project on disk so the lookup
finds an owner and the WS hub can route the envelope.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.main as main_module
from app.services.resource_error import Codes
from app.services.task_error_sink import attach_error_sink
from app.services.ws_envelope import reset_counters
from app.services.ws_hub import WsConnection, get_ws_hub


@pytest.fixture
def project_with_owner(tmp_path: Path, monkeypatch):
    """Create a real project on disk so ``_resolve_owner_id`` finds it.

    The sink calls ``get_project_manager().get_project(project_id)`` to
    resolve ``owner_id`` for the WS routing key; that function reads
    ``config.json`` from the on-disk project layout.
    """
    monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")
    monkeypatch.setenv("LOCAL_DEV_USER_ID", "sink-owner")
    original = main_module.PROJECTS_DIR
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None

    pm = main_module.get_project_manager()
    from app.models.schemas import ProjectCreate

    project = pm.create_project(
        ProjectCreate(name="sink-test", description=""),
        owner_id="sink-owner",
    )
    yield project.id

    main_module._project_manager = None
    main_module.PROJECTS_DIR = original


@pytest.fixture(autouse=True)
def _reset_ws_state():
    reset_counters()
    get_ws_hub()._reset()
    yield
    reset_counters()
    get_ws_hub()._reset()


def _register_test_conn(user_id: str) -> WsConnection:
    """Add a fake :class:`WsConnection` (no real WebSocket) to the hub."""
    conn = WsConnection(
        connection_id=f"sink-{user_id}",
        user_id=user_id,
        websocket=None,
    )
    get_ws_hub().register(conn)
    return conn


class TestAttachErrorSink:
    @pytest.mark.asyncio
    async def test_crash_publishes_system_error(self, project_with_owner):
        project_id = project_with_owner
        conn = _register_test_conn("sink-owner")

        async def crasher():
            raise RuntimeError("boom")

        attach_error_sink(
            asyncio.create_task(crasher(), name="test-crasher"),
            project_id=project_id,
            resource="optimization",
        )
        envelope = await asyncio.wait_for(conn.queue.get(), timeout=1.0)

        # Envelope shape: project_event wrapping the error payload.
        assert envelope["kind"] == "project_event"
        assert envelope["stream"] == f"project:{project_id}"
        data = envelope["data"]
        assert data["resource"] == "optimization"
        assert data["type"] == "error"
        assert data["code"] == Codes.SYSTEM_TASK_CRASHED
        assert "boom" in data["message"]
        assert data["details"]["task_name"] == "test-crasher"
        assert data["details"]["exception"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_successful_task_publishes_nothing(self, project_with_owner):
        project_id = project_with_owner
        conn = _register_test_conn("sink-owner")

        async def ok():
            return "done"

        attach_error_sink(
            asyncio.create_task(ok()),
            project_id=project_id,
            resource="optimization",
        )
        # Give the done-callback a tick.
        await asyncio.sleep(0.05)
        # No envelope should have arrived.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.queue.get(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_cancelled_task_publishes_nothing(self, project_with_owner):
        """Cancellation is a normal exit — don't fire a spurious crash event."""
        project_id = project_with_owner
        conn = _register_test_conn("sink-owner")

        async def slow():
            await asyncio.sleep(10)

        task = attach_error_sink(
            asyncio.create_task(slow()),
            project_id=project_id,
            resource="agent_turn",
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # No envelope.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(conn.queue.get(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_resource_label_surfaces_on_event(self, project_with_owner):
        project_id = project_with_owner
        conn = _register_test_conn("sink-owner")

        async def crasher():
            raise ValueError("oops")

        attach_error_sink(
            asyncio.create_task(crasher()),
            project_id=project_id,
            resource="eval_job",
        )
        envelope = await asyncio.wait_for(conn.queue.get(), timeout=1.0)
        assert envelope["data"]["resource"] == "eval_job"


# ---------------------------------------------------------------------------
# Fire-and-forget WS task retention test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_publish_tasks_retained():
    """Fire-and-forget WS publish tasks are retained in module set."""
    import app.services.jobs as jobs_module
    from app.services.jobs import _publish_project_event, _publish_per_job_done
    from app.services.jobs import Job
    from unittest.mock import MagicMock

    # Clear the retention set before test
    jobs_module._ws_publish_tasks.clear()
    assert len(jobs_module._ws_publish_tasks) == 0, "Retention set should start empty"

    # Create a mock job for testing
    job = MagicMock(spec=Job)
    job.job_id = "test-job"
    job.project_id = "test-project"
    job.output_file = None
    job.error = None

    # Mock get_project_manager to return a project with owner_id
    from app import main as main_module
    original_pm = main_module._project_manager

    try:
        mock_pm = MagicMock()
        mock_project = MagicMock()
        mock_project.owner_id = "test-owner"
        mock_pm.get_project.return_value = mock_project
        main_module._project_manager = mock_pm

        # Call fire-and-forget helpers
        _publish_project_event("test-project", {"resource": "eval_job", "type": "started"})

        # CRITICAL: Verify the task was added to the retention set
        # (it should be present immediately after create_task)
        assert len(jobs_module._ws_publish_tasks) > 0, \
            "Fire-and-forget WS task should be retained in _ws_publish_tasks set"

        # Wait for the tasks to finish; each discards itself from the set via
        # its done callback. Poll against a deadline rather than sleeping a
        # fixed interval: on a loaded machine a fixed 0.1s can elapse before
        # the publish task has even run, which failed a correct retention set.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while jobs_module._ws_publish_tasks and loop.time() < deadline:
            await asyncio.sleep(0.01)

        # After completion, the set should be empty again (discard callback fired)
        assert len(jobs_module._ws_publish_tasks) == 0, \
            "Completed tasks should be discarded from retention set"

    finally:
        main_module._project_manager = original_pm
