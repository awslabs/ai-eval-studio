# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for microVM job visibility: agent-facing READ tools + durable running write.

This is the LOAD-BEARING coverage for job visibility inside the microVM: an
AgentCore microVM has a FRESH empty JobTracker (empty _jobs dict) but sees
durable jobs because the agent-facing READ tools rebuild from DynamoDB
before reading the tracker.

Simulates the microVM by using a fresh JobTracker (the autouse
_reset_singletons fixture in conftest.py resets JobTracker._instance before
each test, so get_job_tracker() returns an empty tracker) alongside a
populated moto DynamoDB store.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from moto import mock_aws

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.services.jobs import get_job_tracker
from app.services.mcp_tools.status import get_project_status as _get_project_status_tool
from app.services.mcp_tools.eval import check_eval_status as _check_eval_status_tool
from app.services.mcp_tools.dispatch import (
    request_user_input as _request_user_input_tool,
    resume_job as _resume_job_tool,
)

# Extract handlers from SDK tool wrappers
_get_project_status = _get_project_status_tool.handler
_check_eval_status = _check_eval_status_tool.handler
_request_user_input = _request_user_input_tool.handler
_resume_job = _resume_job_tool.handler

logger = logging.getLogger(__name__)


def _create_metadata_table(ddb_client, table_name: str):
    """Create the metadata DynamoDB table for moto tests."""
    ddb_client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "project_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "project_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )


@pytest.fixture
def moto_store():
    """Provide a moto-backed DynamoDB store for microVM simulation tests."""
    with mock_aws():
        import boto3
        ddb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-metadata-table"
        _create_metadata_table(ddb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=ddb_client, table_name=table_name)
        yield store


@pytest.mark.asyncio
async def test_get_project_status_sees_durable_running_job_with_empty_tracker(moto_store, tmp_path):
    """get_project_status sees durable job with empty tracker (arch_split path)."""
    project_id = "test-proj-001"
    job_id = "eval-job-001"

    # Seed a durable running eval job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {"message": "Scoring 5/10"},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {"arch_split": True, "user_id": "test_owner"},
    })

    # Verify tracker is empty (autouse fixture reset)
    tracker = get_job_tracker()
    assert tracker.get_active_jobs(project_id) == []

    # Create minimal project dir so load_harness_config doesn't error
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Patch platform + owner resolver + resolve_arch_split
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)), \
         patch("app.services.mcp_tools.status.resolve_project_dir", return_value=project_dir):
        mock_plat.return_value.metadata_store = moto_store

        # Call get_project_status — should rebuild and see the job
        result = await _get_project_status({"project_id": project_id})

    # Parse JSON response
    assert result.get("is_error") is not True
    import json
    data = json.loads(result["content"][0]["text"])
    assert "active_jobs" in data
    assert len(data["active_jobs"]) == 1
    assert data["active_jobs"][0]["job_id"] == job_id
    assert data["active_jobs"][0]["state"] == "running"


@pytest.mark.asyncio
async def test_get_project_status_no_rebuild_on_old_path(moto_store, tmp_path):
    """get_project_status old path (arch_split=False) is unchanged — no store read."""
    project_id = "test-proj-002"
    job_id = "eval-job-002"

    # Seed a durable job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_active_jobs(project_id) == []

    # Create minimal project dir
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Patch resolve_arch_split -> False
    with patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=False)), \
         patch("app.services.mcp_tools.status.resolve_project_dir", return_value=project_dir):
        result = await _get_project_status({"project_id": project_id})

    # Old path: active_jobs should still be empty (no rebuild)
    assert result.get("is_error") is not True
    import json
    data = json.loads(result["content"][0]["text"])
    assert data["active_jobs"] == []


@pytest.mark.asyncio
async def test_check_eval_status_finds_durable_job_empty_tracker(moto_store):
    """check_eval_status finds durable job with empty tracker (arch_split path)."""
    project_id = "test-proj-003"
    job_id = "eval-job-003"

    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {"models_completed": 1, "models_total": 3, "message": "Scoring model 2"},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {"arch_split": True, "user_id": "test_owner"},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_job(job_id) is None

    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)):
        mock_plat.return_value.metadata_store = moto_store

        result = await _check_eval_status({"project_id": project_id, "job_id": job_id})

    # Should NOT be "Job not found" error
    assert result.get("is_error") is not True
    text = result["content"][0]["text"]
    assert "Status: running" in text
    assert "Models: 1/3" in text


@pytest.mark.asyncio
async def test_check_eval_status_old_path_not_found(moto_store):
    """check_eval_status old path (arch_split=False) returns 'Job not found'."""
    project_id = "test-proj-004"
    job_id = "eval-job-004"

    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_job(job_id) is None

    # Old path: no rebuild
    with patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=False)):
        result = await _check_eval_status({"project_id": project_id, "job_id": job_id})

    # Old path with empty tracker -> "Job not found"
    assert result.get("is_error") is True
    text = result["content"][0]["text"]
    assert "Job not found" in text


@pytest.mark.asyncio
async def test_request_user_input_finds_durable_active_job(moto_store):
    """request_user_input finds durable active job, transitions to awaiting_user."""
    project_id = "test-proj-005"
    job_id = "eval-job-005"

    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {"arch_split": True, "user_id": "test_owner"},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_job(job_id) is None

    # Patch fan-out so update_state doesn't error on WS publish
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)), \
         patch("app.services.jobs._publish_project_event"):
        mock_plat.return_value.metadata_store = moto_store

        result = await _request_user_input({
            "project_id": project_id,
            "job_id": job_id,
            "question": "Which model should I use?"
        })

    # Should NOT error; job now in tracker with state=awaiting_user
    assert result.get("is_error") is not True
    job = tracker.get_job(job_id)
    assert job is not None
    assert job.state == "awaiting_user"
    assert job.pause_reason == "Which model should I use?"


@pytest.mark.asyncio
async def test_resume_job_finds_durable_awaiting_job(moto_store):
    """resume_job finds durable awaiting_user job, transitions to running."""
    project_id = "test-proj-006"
    job_id = "eval-job-006"

    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "awaiting_user",
        "progress": {},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {"arch_split": True, "user_id": "test_owner"},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_job(job_id) is None

    # Patch turn-submit path so resume_job doesn't spin a real turn
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)), \
         patch("app.services.agent_turn.get_agent_turn_tracker") as mock_turn_tracker:
        mock_plat.return_value.metadata_store = moto_store
        mock_turn_tracker.return_value.submit = MagicMock()

        result = await _resume_job({
            "project_id": project_id,
            "job_id": job_id,
            "user_input": "Use gpt-4"
        })

    # Should NOT error; job now running
    assert result.get("is_error") is not True
    job = tracker.get_job(job_id)
    assert job is not None
    assert job.state == "running"
    assert job.config["user_input"] == "Use gpt-4"


@pytest.mark.asyncio
async def test_update_progress_writes_running_durably_moto_roundtrip(moto_store):
    """update_progress writes state=running durably on queued->running transition."""
    project_id = "test-proj-007"
    job_id = "eval-job-007"

    # Coordinator put_job at state=queued
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "queued",
        "progress": {},
        "created_at": "2026-07-22T10:00:00Z",
        "config": {"arch_split": True, "user_id": "test_owner"},
    })

    # Register the SAME job in the tracker (worker pattern)
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={"arch_split": True, "user_id": "test_owner"},
        state="queued",
        progress={},
        created_at="2026-07-22T10:00:00Z",
    )

    # Patch platform
    with patch("app.platform.get_platform") as mock_plat:
        mock_plat.return_value.metadata_store = moto_store

        # Call update_progress once -> should flip to running + write through
        tracker.update_progress(job_id, message="Scoring 1/10")

        # Drain pending writes
        await tracker.drain_pending_writes()

    # Read item back from moto DDB
    stored = await moto_store.get_job("test_owner", project_id, job_id)
    assert stored is not None
    assert stored["state"] == "running"
    assert stored["progress"]["message"] == "Scoring 1/10"


@pytest.mark.asyncio
async def test_update_progress_running_write_once():
    """update_progress schedules state=running write ONCE (not per progress update)."""
    job_id = "eval-job-008"
    project_id = "test-proj-008"

    # Create a queued job in the tracker
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={"arch_split": True, "owner_id": "test_owner"},
        state="queued",
        progress={},
        created_at="2026-07-22T10:00:00Z",
    )

    # Mock store
    mock_store = MagicMock()
    mock_store.update_job_state = AsyncMock()

    with patch("app.platform.get_platform") as mock_plat:
        mock_plat.return_value.metadata_store = mock_store

        # First update_progress -> should schedule state=running write
        # (runs in the real async event loop, not mocked)
        tracker.update_progress(job_id, message="Progress 1")

        # Drain pending writes
        import asyncio
        await tracker.drain_pending_writes()

        # Verify called once with state="running"
        assert mock_store.update_job_state.call_count == 1
        call_args = mock_store.update_job_state.call_args
        assert call_args[1]["state"] == "running"

        # Second update_progress -> should NOT schedule another state write
        tracker.update_progress(job_id, message="Progress 2")
        await tracker.drain_pending_writes()

        # Still just 1 call (no second state write)
        assert mock_store.update_job_state.call_count == 1


@pytest.mark.asyncio
async def test_update_progress_no_durable_write_old_path():
    """update_progress schedules NO durable write on arch_split=False path."""
    job_id = "eval-job-009"
    project_id = "test-proj-009"

    # Old-path job (arch_split=False)
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={"arch_split": False},
        state="queued",
        progress={},
        created_at="2026-07-22T10:00:00Z",
    )

    # Mock store
    mock_store = MagicMock()
    mock_store.update_job_state = AsyncMock()

    with patch("app.platform.get_platform") as mock_plat:
        mock_plat.return_value.metadata_store = mock_store

        # update_progress
        tracker.update_progress(job_id, message="Old path")

        # Drain
        import asyncio
        await asyncio.sleep(0.01)

        # No store write on old path
        mock_store.update_job_state.assert_not_called()


# ========== Durable terminal state wins (refresh_terminal mode) ==========


@pytest.mark.asyncio
async def test_refresh_terminal_cached_running_store_cancelled(moto_store):
    """refresh_terminal=True: cached running + store cancelled -> cached becomes cancelled."""
    project_id = "test-proj-term-001"
    job_id = "eval-job-term-001"

    # Seed durable cancelled job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "cancelled",
        "completed_at": "2026-07-27T10:00:00Z",
        "progress": {"message": "Cancelled by user"},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Pre-populate tracker with stale running job (with result and error)
    from app.services.jobs import Job
    from app.services.resource_error import ResourceError
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="running",
        progress={"message": "Scoring 5/10"},
        result={"pass_rate": 0.8},
        error=ResourceError(code="TEST-01", message="Some error"),
        created_at="2026-07-27T09:00:00Z",
    )

    # Rebuild with refresh_terminal=True
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"):
        mock_plat.return_value.metadata_store = moto_store
        await tracker.rebuild_jobs_from_store(project_id, arch_split=True, refresh_terminal=True)

    # Verify: state updated to cancelled, completed_at populated
    job = tracker.get_job(job_id)
    assert job is not None
    assert job.state == "cancelled"
    assert job.completed_at == "2026-07-27T10:00:00Z"
    assert job.progress["message"] == "Cancelled by user"
    # Result and error preserved (durable row carries neither)
    assert job.result == {"pass_rate": 0.8}
    assert job.error is not None
    assert job.error.code == "TEST-01"


@pytest.mark.asyncio
async def test_refresh_terminal_cached_running_store_running_unchanged():
    """refresh_terminal=True: cached running + store running -> cached unchanged."""
    project_id = "test-proj-term-002"
    job_id = "eval-job-term-002"

    # Mock store with running job
    mock_store = MagicMock()
    mock_store.list_jobs = AsyncMock(return_value=[{
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {"message": "Store progress"},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    }])

    # Pre-populate tracker with running job
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="running",
        progress={"message": "Cached progress"},
        created_at="2026-07-27T09:00:00Z",
    )

    # Rebuild with refresh_terminal=True
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"):
        mock_plat.return_value.metadata_store = mock_store
        await tracker.rebuild_jobs_from_store(project_id, arch_split=True, refresh_terminal=True)

    # Verify: cached state UNCHANGED (store is non-terminal)
    job = tracker.get_job(job_id)
    assert job.state == "running"
    assert job.progress["message"] == "Cached progress"  # Not overwritten


@pytest.mark.asyncio
async def test_refresh_terminal_cached_completed_store_running_unchanged():
    """refresh_terminal=True: cached completed + store running -> no downgrade."""
    project_id = "test-proj-term-003"
    job_id = "eval-job-term-003"

    # Mock store with running job (lagging store)
    mock_store = MagicMock()
    mock_store.list_jobs = AsyncMock(return_value=[{
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    }])

    # Pre-populate tracker with completed job
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="completed",
        completed_at="2026-07-27T10:00:00Z",
        progress={},
        created_at="2026-07-27T09:00:00Z",
    )

    # Rebuild with refresh_terminal=True
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"):
        mock_plat.return_value.metadata_store = mock_store
        await tracker.rebuild_jobs_from_store(project_id, arch_split=True, refresh_terminal=True)

    # Verify: state stays completed (terminal is sticky)
    job = tracker.get_job(job_id)
    assert job.state == "completed"


@pytest.mark.asyncio
async def test_check_eval_status_with_refresh_terminal_reports_cancelled(moto_store):
    """check_eval_status with refresh_terminal=True reports durably-cancelled job."""
    project_id = "test-proj-term-004"
    job_id = "eval-job-term-004"

    # Seed durable cancelled job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "cancelled",
        "completed_at": "2026-07-27T10:00:00Z",
        "progress": {"message": "Cancelled"},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Pre-populate tracker with stale running job
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="running",
        progress={},
        created_at="2026-07-27T09:00:00Z",
    )

    # Call check_eval_status (it should rebuild with refresh_terminal=True)
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)):
        mock_plat.return_value.metadata_store = moto_store
        result = await _check_eval_status({"project_id": project_id, "job_id": job_id})

    # Verify response reports cancelled
    assert result.get("is_error") is not True
    text = result["content"][0]["text"]
    assert "cancelled" in text


@pytest.mark.asyncio
async def test_get_project_status_with_refresh_terminal_filters_out_terminal(moto_store, tmp_path):
    """get_project_status with refresh_terminal=True filters out refreshed terminal jobs."""
    project_id = "test-proj-term-005"
    job_id = "eval-job-term-005"

    # Seed durable cancelled job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "cancelled",
        "completed_at": "2026-07-27T10:00:00Z",
        "progress": {},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Pre-populate tracker with stale running job
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="running",
        progress={},
        created_at="2026-07-27T09:00:00Z",
    )

    # Create minimal project dir
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Call get_project_status (should rebuild with refresh_terminal=True)
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)), \
         patch("app.services.mcp_tools.status.resolve_project_dir", return_value=project_dir):
        mock_plat.return_value.metadata_store = moto_store
        result = await _get_project_status({"project_id": project_id})

    # Verify: active_jobs is empty (cancelled job filtered out)
    assert result.get("is_error") is not True
    import json
    data = json.loads(result["content"][0]["text"])
    assert data["active_jobs"] == []


@pytest.mark.asyncio
async def test_rebuild_additive_still_works_empty_cache(moto_store):
    """rebuild_jobs_from_store (default) still adds missing jobs to empty cache."""
    project_id = "test-proj-term-006"
    job_id = "eval-job-term-006"

    # Seed durable running job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Empty tracker
    tracker = get_job_tracker()
    assert tracker.get_job(job_id) is None

    # Rebuild without refresh_terminal (default)
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"):
        mock_plat.return_value.metadata_store = moto_store
        await tracker.rebuild_jobs_from_store(project_id, arch_split=True)

    # Verify job appears in cache (additive behavior)
    job = tracker.get_job(job_id)
    assert job is not None
    assert job.state == "running"


# ========== microVM turn header reflects durable job state ==========


@pytest.mark.asyncio
async def test_agentcore_runtime_refresh_stale_job_to_cancelled(moto_store):
    """agentcore_runtime: refresh refreshes stale cached running job to cancelled."""
    project_id = "test-proj-hdr-001"
    job_id = "eval-job-hdr-001"

    # Seed durable cancelled job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "cancelled",
        "completed_at": "2026-07-27T10:00:00Z",
        "progress": {},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Pre-populate tracker with stale running job
    from app.services.jobs import Job
    tracker = get_job_tracker()
    tracker._jobs[job_id] = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        config={},
        state="running",
        progress={},
        created_at="2026-07-27T09:00:00Z",
    )

    # Simulate the microVM refresh path with populated context
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)):
        mock_plat.return_value.metadata_store = moto_store

        # This is what agentcore_runtime.py does before build_coordinator_prompt
        from app.platform.arch_split import resolve_arch_split
        arch_split = await resolve_arch_split(project_id)
        await tracker.rebuild_jobs_from_store(project_id, arch_split, refresh_terminal=True)

    # Verify: job was refreshed to cancelled, no longer active
    job = tracker.get_job(job_id)
    assert job.state == "cancelled"
    assert tracker.get_active_jobs(project_id) == []  # Terminal job filtered out


@pytest.mark.asyncio
async def test_agentcore_runtime_refresh_cold_cache_fills_gap(moto_store):
    """agentcore_runtime: refresh fills cold cache with durable running job."""
    project_id = "test-proj-hdr-002"
    job_id = "eval-job-hdr-002"

    # Seed durable running job
    await moto_store.put_job({
        "owner_id": "test_owner",
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "running",
        "progress": {"message": "Scoring 5/10"},
        "created_at": "2026-07-27T09:00:00Z",
        "config": {},
    })

    # Empty tracker (cold cache)
    tracker = get_job_tracker()
    assert tracker.get_active_jobs(project_id) == []

    # Simulate the microVM refresh path
    with patch("app.platform.get_platform") as mock_plat, \
         patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="test_owner"), \
         patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=True)):
        mock_plat.return_value.metadata_store = moto_store

        from app.platform.arch_split import resolve_arch_split
        arch_split = await resolve_arch_split(project_id)
        await tracker.rebuild_jobs_from_store(project_id, arch_split, refresh_terminal=True)

    # Verify: job now in tracker and active
    active_jobs = tracker.get_active_jobs(project_id)
    assert len(active_jobs) == 1
    assert active_jobs[0].job_id == job_id
    assert active_jobs[0].state == "running"


# ========== Durable optimization state wins for agent-facing readers ==========


@pytest.mark.asyncio
async def test_dispatch_eval_optimization_fence_durable_cancelled_passes(moto_store, tmp_path):
    """dispatch_eval: durable cancelled opt state -> fence passes, fails at harness guard."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    project_id = "test-proj-opt-001"
    owner_id = "owner-opt-1"

    # Seed durable cancelled optimization state under owner-opt-1
    await moto_store.put_optimization_state(
        owner_id,
        project_id,
        {
            "status": "cancelled",
            "current_iteration": 5,
            "max_iter_cap": 10,
            "max_iterations": 10,
            "plateau_window": 3,
            "target_metric_scores": {},
            "iterations": [],
        },
    )

    # Create project dir WITHOUT run_eval.py so harness guard fires
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Call real dispatch_eval handler
    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value=owner_id),
        patch("app.platform.arch_split.resolve_arch_split", return_value=True),
        patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.platform.get_platform") as mock_platform,
    ):
        # Wire real moto store (do NOT patch OptimizationState.load - testing durable path)
        mock_plat_inst = MagicMock()
        mock_plat_inst.metadata_store = moto_store
        # Mock store methods dispatch uses before optimization fence
        mock_plat_inst.metadata_store.count_active_jobs_for_owner = AsyncMock(return_value=0)
        mock_plat_inst.metadata_store.list_non_terminal_jobs = AsyncMock(return_value=[])
        mock_platform.return_value = mock_plat_inst

        # Setup mocks
        mock_resolve.return_value = project_dir
        mock_tracker_inst = MagicMock()
        mock_tracker.return_value = mock_tracker_inst
        mock_req.return_value = MagicMock(user_id=owner_id)

        result = await dispatch_eval({
            "project_id": project_id,
            "data_file": "test.csv",
            "prompt_version": "baseline",
            "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
        })

        result_text = result["content"][0]["text"]

    # Verify: NOT the optimization-fence refusal, IS the harness-missing error
    assert "Optimization loop is running" not in result_text, \
        f"Fence should NOT refuse cancelled loop, got: {result_text}"
    assert "No eval harness found" in result_text or "run_eval.py missing" in result_text, \
        f"Should fail at harness guard, got: {result_text}"


@pytest.mark.asyncio
async def test_dispatch_eval_optimization_fence_durable_running_refuses(moto_store, tmp_path):
    """dispatch_eval: durable running opt state -> fence refuses."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    project_id = "test-proj-opt-002"
    owner_id = "owner-opt-2"

    # Seed durable running optimization state under owner-opt-2
    await moto_store.put_optimization_state(
        owner_id,
        project_id,
        {
            "status": "running",
            "current_iteration": 5,
            "max_iter_cap": 10,
            "max_iterations": 10,
            "plateau_window": 3,
            "target_metric_scores": {},
            "iterations": [],
        },
    )

    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Call real dispatch_eval handler
    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value=owner_id),
        patch("app.platform.arch_split.resolve_arch_split", return_value=True),
        patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.platform.get_platform") as mock_platform,
    ):
        # Wire real moto store
        mock_plat_inst = MagicMock()
        mock_plat_inst.metadata_store = moto_store
        # Mock store methods dispatch uses before optimization fence
        mock_plat_inst.metadata_store.count_active_jobs_for_owner = AsyncMock(return_value=0)
        mock_plat_inst.metadata_store.list_non_terminal_jobs = AsyncMock(return_value=[])
        mock_platform.return_value = mock_plat_inst

        # Setup mocks
        mock_resolve.return_value = project_dir
        mock_tracker_inst = MagicMock()
        mock_tracker.return_value = mock_tracker_inst
        mock_req.return_value = MagicMock(user_id=owner_id)

        result = await dispatch_eval({
            "project_id": project_id,
            "data_file": "test.csv",
            "prompt_version": "baseline",
            "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
        })

        result_text = result["content"][0]["text"]

    # Verify: IS the optimization-fence refusal
    assert result.get("is_error") is True, f"Should be error, got: {result}"
    assert "Optimization loop is running" in result_text, \
        f"Should refuse with optimization fence message, got: {result_text}"


@pytest.mark.asyncio
async def test_dispatch_eval_optimization_fence_old_path_efs_still_fences(tmp_path):
    """dispatch_eval: arch_split=False + stale EFS running -> fence still refuses (regression guard)."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    project_id = "test-proj-opt-003"
    owner_id = "owner-opt-3"

    # Create project dir with stale EFS optimization state (running)
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    # Create run_eval.py + data file + prompt so those guards pass
    (project_dir / "run_eval.py").write_text("# harness")
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "raw" / "test.csv").write_text("input,output\na,b")
    (project_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (project_dir / "prompts" / "baseline.md").write_text("test prompt")
    (project_dir / "output" / "optimization").mkdir(parents=True, exist_ok=True)

    # Write stale EFS running state (correct filename: state.json)
    import json
    (project_dir / "output" / "optimization" / "state.json").write_text(json.dumps({
        "status": "running",
        "current_iteration": 3,
        "max_iterations": 10,
        "max_iter_cap": 10,
        "plateau_window": 3,
        "target_metric_scores": {},
        "iterations": [],
    }))

    # Call real dispatch_eval with arch_split=False (old path)
    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.agent_turn._resolve_owner_id_for_project", return_value=owner_id),
        patch("app.platform.arch_split.resolve_arch_split", return_value=False),  # Old path
        patch("app.platform.types.current_request") as mock_req,
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),  # Skip model validation
    ):
        # Setup mocks
        mock_resolve.return_value = project_dir
        mock_tracker_inst = MagicMock()
        # Make get_active_job_by_type return None (no existing eval job)
        mock_tracker_inst.get_active_job_by_type.return_value = None
        mock_tracker.return_value = mock_tracker_inst
        mock_req.return_value = MagicMock(user_id=owner_id)

        result = await dispatch_eval({
            "project_id": project_id,
            "data_file": "test.csv",
            "prompt_version": "baseline",
            "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
        })

        result_text = result["content"][0]["text"]

    # Verify: old path still fences on EFS running state
    assert result.get("is_error") is True, f"Should be error, got: {result}"
    assert "Optimization loop is running" in result_text, \
        f"Old path should still fence on EFS running state, got: {result_text}"
