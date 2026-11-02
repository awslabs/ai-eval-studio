# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for JobTracker write-through + rebuild-on-read.

Verifies that with the arch_split flag on:
- create_job persists via put_job
- update_state persists via update_job_state (including complete_job/fail_job)
- rebuild_jobs_from_store repopulates the cache from the store on a miss

And with the flag off:
- zero store calls fire (the tracker stays in-memory only)

Uses Stubber-based DDB adapter tests, following test_aws_adapters.py pattern.
"""

import asyncio
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from botocore.stub import ANY, Stubber

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.services.jobs import Job, JobTracker


@pytest.fixture
def stubbed_store():
    """DynamoDB metadata store with Stubber."""
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    store = DynamoDBMetadataStore(ddb_client=ddb, table_name="test-table")
    stubber.activate()
    yield store, stubber
    stubber.deactivate()


@pytest.fixture
def tracker():
    """Fresh JobTracker instance for each test."""
    return JobTracker()


# --- create_job write-through tests ---


@pytest.mark.asyncio
async def test_create_job_persists_on_new_path(stubbed_store, tracker):
    """On new path (flag=True), create_job calls put_job."""
    store, stubber = stubbed_store

    # Stub the put_job call (put_item in DDB) - use ANY for flexibility
    stubber.add_response(
        "put_item",
        {},
        expected_params={
            "TableName": "test-table",
            "Item": ANY,  # Simplified - just verify the call happens
        },
    )

    # Patch _resolve_owner_id_for_project to return a known owner
    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Create a job with flag=True in config
            job = tracker.create_job(
                "proj_1",
                "eval",
                config={"arch_split": True, "user_id": "test_owner"},
                job_id="job_1",
            )

            # Wait for the background write-through task to complete
            await asyncio.sleep(0.2)

    assert job.job_id == "job_1"
    stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_create_job_no_store_call_on_old_path(tracker):
    """On old path (flag=False), create_job does NOT call put_job."""
    mock_store = AsyncMock()

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Create a job with flag=False (default)
            job = tracker.create_job(
                "proj_1",
                "eval",
                config={},  # No arch_split or False
                job_id="job_1",
            )

            # Wait to ensure no background task fires
            await asyncio.sleep(0.2)

    assert job.job_id == "job_1"
    # Assert put_job was NOT called
    mock_store.put_job.assert_not_called()


# --- update_state write-through tests ---


@pytest.mark.asyncio
async def test_update_state_persists_on_new_path(stubbed_store, tracker):
    """On new path, update_state calls update_job_state."""
    store, stubber = stubbed_store

    # Stub the update_job_state call (update_item in DDB)
    # Note: update_job_state also sets ExpressionAttributeNames
    stubber.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": "test-table",
            "Key": ANY,
            "UpdateExpression": ANY,
            "ExpressionAttributeValues": ANY,
            "ExpressionAttributeNames": ANY,
        },
    )

    # Create a job in the tracker (in-memory)
    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        job = tracker.create_job(
            "proj_1",
            "eval",
            config={"arch_split": True, "user_id": "test_owner"},
            job_id="job_1",
        )

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store

        # Update the job state
        tracker.update_state(job.job_id, "running")

        # Wait for the background write-through task
        await asyncio.sleep(0.2)

    stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_complete_job_persists_via_update_state(stubbed_store, tracker):
    """complete_job routes through update_state → update_job_state on new path."""
    store, stubber = stubbed_store

    # Stub the update_job_state call
    stubber.add_response(
        "update_item",
        {},
        expected_params={
            "TableName": "test-table",
            "Key": ANY,
            "UpdateExpression": ANY,
            "ExpressionAttributeValues": ANY,
            "ExpressionAttributeNames": ANY,
        },
    )

    # Create and complete a job
    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        job = tracker.create_job(
            "proj_1",
            "eval",
            config={"arch_split": True, "user_id": "test_owner"},
            job_id="job_1",
        )

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store

        tracker.complete_job(job.job_id, output_file="output.json", result={"status": "success"})
        await asyncio.sleep(0.2)

    stubber.assert_no_pending_responses()


# --- get_job/list_jobs rebuild-on-read tests ---


@pytest.mark.asyncio
async def test_rebuild_jobs_from_store_repopulates_cache(stubbed_store, tracker):
    """On new path with cache miss, rebuild_jobs_from_store repopulates from store.list_jobs."""
    store, stubber = stubbed_store

    # Stub the list_jobs query (returns one job not in cache)
    stubber.add_response(
        "query",
        {
            "Items": [
                {
                    "PK": {"S": "USER#test_owner"},
                    "SK": {"S": "PROJECT#proj_1#JOB#job_rebuild"},
                    "owner_id": {"S": "test_owner"},
                    "project_id": {"S": "proj_1"},
                    "job_id": {"S": "job_rebuild"},
                    "type": {"S": "eval"},
                    "state": {"S": "completed"},
                    "progress": {"M": {"message": {"S": "Done"}}},
                    "created_at": {"S": "2026-07-01T00:00:00Z"},
                    "completed_at": {"S": "2026-07-01T00:10:00Z"},
                    "result": {"M": {"status": {"S": "success"}}},
                    "config": {"M": {}},
                }
            ]
        },
        expected_params={
            "TableName": "test-table",
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
            "ExpressionAttributeValues": ANY,
            "ScanIndexForward": False,
        },
    )

    # Patch _resolve_owner_id_for_project to return a known owner
    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Cache is empty before rebuild
            assert tracker.get_job("job_rebuild") is None

            # Call rebuild_jobs_from_store (new path)
            await tracker.rebuild_jobs_from_store("proj_1", arch_split=True)

            # Now the job should be in the cache
            job = tracker.get_job("job_rebuild")
            assert job is not None
            assert job.job_id == "job_rebuild"
            assert job.project_id == "proj_1"
            assert job.state == "completed"
            assert job.type == "eval"

    stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_rebuild_preserves_live_jobs_in_cache(stubbed_store, tracker):
    """Rebuild does NOT clobber jobs already in the cache (cache-as-truth for live jobs)."""
    store, stubber = stubbed_store

    # Create a live job in the cache with flag=False (no write-through)
    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        live_job = tracker.create_job(
            "proj_1",
            "eval",
            config={"arch_split": False},
            job_id="job_live",
        )
        live_job.state = "running"

    # Stub the list_jobs query (returns the SAME job_id but with different state from store)
    stubber.add_response(
        "query",
        {
            "Items": [
                {
                    "PK": {"S": "USER#test_owner"},
                    "SK": {"S": "PROJECT#proj_1#JOB#job_live"},
                    "owner_id": {"S": "test_owner"},
                    "project_id": {"S": "proj_1"},
                    "job_id": {"S": "job_live"},
                    "type": {"S": "eval"},
                    "state": {"S": "completed"},  # Store says "completed"
                    "progress": {"M": {"message": {"S": "Done"}}},
                    "created_at": {"S": "2026-07-01T00:00:00Z"},
                    "config": {"M": {}},
                }
            ]
        },
        expected_params={
            "TableName": "test-table",
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
            "ExpressionAttributeValues": ANY,
            "ScanIndexForward": False,
        },
    )

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Rebuild from store
            await tracker.rebuild_jobs_from_store("proj_1", arch_split=True)

            # Cache should preserve the live job's state (cache-as-truth)
            job = tracker.get_job("job_live")
            assert job is not None
            assert job.state == "running"  # NOT "completed" from store

    stubber.assert_no_pending_responses()


# --- scope edge test (no reattach/resume) ---


@pytest.mark.asyncio
async def test_rebuild_returns_metadata_only_no_reattach(stubbed_store, tracker):
    """Scope edge: rebuild-on-read returns job metadata ONLY.

    Does NOT:
    - Relaunch an eval subprocess
    - Reattach to a worker
    - Resume any in-flight work
    - Register replay providers for rebuilt jobs

    Rebuild is a metadata-recovery path only: it makes the job record
    queryable and leaves execution alone.
    """
    store, stubber = stubbed_store

    # Stub the list_jobs query (returns an eval job with subprocess pid)
    stubber.add_response(
        "query",
        {
            "Items": [
                {
                    "PK": {"S": "USER#test_owner"},
                    "SK": {"S": "PROJECT#proj_1#JOB#job_eval"},
                    "owner_id": {"S": "test_owner"},
                    "project_id": {"S": "proj_1"},
                    "job_id": {"S": "job_eval"},
                    "type": {"S": "eval"},
                    "state": {"S": "running"},
                    "progress": {"M": {"message": {"S": "Scoring 50/100"}}},
                    "created_at": {"S": "2026-07-01T00:00:00Z"},
                    "config": {"M": {}},
                }
            ]
        },
        expected_params={
            "TableName": "test-table",
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
            "ExpressionAttributeValues": ANY,
            "ScanIndexForward": False,
        },
    )

    # Mock subprocess to assert it's NOT called during rebuild
    mock_subprocess = MagicMock()

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            with patch("subprocess.Popen", mock_subprocess):
                with patch("asyncio.create_subprocess_exec", mock_subprocess):
                    # Rebuild from store
                    await tracker.rebuild_jobs_from_store("proj_1", arch_split=True)

                    # Assert the job is now in cache (metadata reconstructed)
                    job = tracker.get_job("job_eval")
                    assert job is not None
                    assert job.job_id == "job_eval"
                    assert job.state == "running"
                    assert job.type == "eval"

                    # scope edge: NO subprocess relaunch
                    mock_subprocess.assert_not_called()

                    # NO replay provider registered, NO task, NO events for rebuilt jobs
                    # (replay providers are registered at create_job time only)
                    # We reconstructed the Job dataclass metadata ONLY — no _task,
                    # no events buffer setup, proving no worker was relaunched.
                    assert job._task is None
                    assert job.events == []

    stubber.assert_no_pending_responses()


# --- "jobs by owner" query test ---


@pytest.mark.asyncio
async def test_list_jobs_by_owner_query(stubbed_store):
    """'jobs by owner' is answerable via list_jobs_by_owner."""
    store, stubber = stubbed_store

    # Stub the list_jobs_by_owner query (base-table owner query)
    stubber.add_response(
        "query",
        {
            "Items": [
                {
                    "PK": {"S": "USER#owner_1"},
                    "SK": {"S": "PROJECT#proj_a#JOB#job_1"},
                    "owner_id": {"S": "owner_1"},
                    "project_id": {"S": "proj_a"},
                    "job_id": {"S": "job_1"},
                    "type": {"S": "eval"},
                    "state": {"S": "running"},
                },
                {
                    "PK": {"S": "USER#owner_1"},
                    "SK": {"S": "PROJECT#proj_b#JOB#job_2"},
                    "owner_id": {"S": "owner_1"},
                    "project_id": {"S": "proj_b"},
                    "job_id": {"S": "job_2"},
                    "type": {"S": "eval"},
                    "state": {"S": "completed"},
                },
            ]
        },
        expected_params={
            "TableName": "test-table",
            "KeyConditionExpression": ANY,
            "ExpressionAttributeValues": ANY,
            "ScanIndexForward": ANY,
        },
    )

    jobs = await store.list_jobs_by_owner("owner_1")

    assert len(jobs) == 2
    assert jobs[0]["job_id"] == "job_1"
    assert jobs[1]["job_id"] == "job_2"
    stubber.assert_no_pending_responses()


# --- terminal fan-out fires exactly once ---


@pytest.mark.asyncio
async def test_terminal_fanout_fires_once(tracker):
    """Terminal fan-out fires exactly once, even with redundant transitions."""
    event_count = 0

    def count_events(*args, **kwargs):
        nonlocal event_count
        event_count += 1

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        with patch("app.services.jobs._publish_per_job_done", side_effect=count_events):
            job = tracker.create_job("proj_1", "eval", job_id="job_1")

            # First transition to terminal
            tracker.update_state(job.job_id, "completed")

            # Redundant transition (should NOT emit again)
            tracker.update_state(job.job_id, "completed")

    assert event_count == 1  # Fired exactly once


# --- drain_pending_writes test ---


@pytest.mark.asyncio
async def test_drain_pending_writes_awaits_all_tasks(tracker):
    """drain_pending_writes awaits both _pending_writes and _durable_append_tasks."""
    from app.services import jobs

    # Create controllable tasks that we can verify were awaited
    completed_flags = {"state": False, "event": False}

    async def state_write():
        await asyncio.sleep(0.01)
        completed_flags["state"] = True

    async def event_write():
        await asyncio.sleep(0.01)
        completed_flags["event"] = True

    # Add tasks to both sets
    state_task = asyncio.create_task(state_write())
    event_task = asyncio.create_task(event_write())

    tracker._pending_writes.add(state_task)
    jobs._durable_append_tasks.add(event_task)

    # Drain should await both to completion
    await tracker.drain_pending_writes()

    # Verify both tasks completed (not cancelled)
    assert state_task.done()
    assert event_task.done()
    assert completed_flags["state"]
    assert completed_flags["event"]

    # Clean up: clear the global sets to avoid test pollution
    tracker._pending_writes.clear()
    jobs._durable_append_tasks.clear()


# --- Result blob NOT written durably (pointers only) ---


@pytest.mark.asyncio
async def test_durable_write_omits_result_blob(tracker):
    """Test A: durable update_job_state call omits result blob (mock store).

    On the durable path (arch_split=True), when update_state is called with
    a result payload, the background store.update_job_state call MUST NOT
    include the result blob — only state, progress, and output_file pointers.
    The in-memory Job.result MUST still be set from the passed result arg.
    """
    mock_store = AsyncMock()

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="test_owner",
    ):
        # Create a job with arch_split=True
        job = tracker.create_job(
            "proj_1",
            "eval",
            config={"arch_split": True, "user_id": "test_owner"},
            job_id="job_1",
        )
        # Set output_file before terminal transition
        job.output_file = "eval_results_v1_20260718_100000.json"

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store

        # Patch fan-out to silence it (borrowed from test_config_key_normalization.py)
        with patch("app.services.jobs._publish_per_job_done"):
            with patch("app.services.jobs._publish_project_event"):
                # Call update_state with a large result payload
                large_result = {"model_runs": [{"model": f"model_{i}", "data": "x" * 1000} for i in range(100)]}
                tracker.update_state(job.job_id, "completed", result=large_result)

                # Wait for the background write-through task
                await asyncio.sleep(0.2)

    # Assert the mock store's update_job_state was called once
    mock_store.update_job_state.assert_called_once()

    # Get the call kwargs
    call_kwargs = mock_store.update_job_state.call_args.kwargs

    # Assert state, progress, and output_file ARE passed
    assert call_kwargs["state"] == "completed"
    assert "progress" in call_kwargs
    assert call_kwargs["output_file"] == "eval_results_v1_20260718_100000.json"

    # Assert result is EITHER absent or None (NOT the large payload)
    result_val = call_kwargs.get("result")
    assert result_val is None, f"Expected result=None, got {result_val}"

    # Assert in-memory Job.result IS set (preserved)
    assert tracker.get_job("job_1").result == large_result


@pytest.mark.asyncio
async def test_durable_write_omits_result_blob_moto_roundtrip():
    """Test B: moto DDB round-trip proves result NOT stored, pointers ARE stored.

    Round-trip via moto DynamoDB: put_job a job, update_state to completed
    with a result payload, then read the item back from the store. The stored
    item MUST NOT have a "result" attribute but MUST carry state, progress,
    and output_file.
    """
    from moto import mock_aws
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    # Helper to create the test table (from test_worker_durable_write.py)
    def _create_metadata_table(ddb):
        ddb.create_table(
            TableName="test-table",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name="test-table")

        tracker = JobTracker()

        with patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value="test_owner",
        ):
            # Create a job with arch_split=True
            job = tracker.create_job(
                "proj_1",
                "eval",
                config={"arch_split": True, "user_id": "test_owner"},
                job_id="job_moto",
            )
            job.output_file = "eval_results_v2_20260718_110000.json"

        # Put the job durably (coordinator entry point)
        await store.put_job({
            "job_id": "job_moto",
            "project_id": "proj_1",
            "owner_id": "test_owner",
            "type": "eval",
            "state": "queued",
            "progress": job.progress,
            "config": job.config,
            "created_at": job.created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        })

        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Patch fan-out to silence it
            with patch("app.services.jobs._publish_per_job_done"):
                with patch("app.services.jobs._publish_project_event"):
                    # Update to completed with result payload
                    result_payload = {"status": "success", "data": [1, 2, 3]}
                    tracker.update_state("job_moto", "completed", result=result_payload)

                    # Drain pending writes so the background update completes
                    await tracker.drain_pending_writes()

        # Read the item back from DDB
        stored_job = await store.get_job("test_owner", "proj_1", "job_moto")

        # Assert stored item has NO "result" attribute
        assert "result" not in stored_job, f"Expected no 'result' in stored job, got {stored_job.get('result')}"

        # Assert state, progress, and output_file ARE stored
        assert stored_job["state"] == "completed"
        assert "progress" in stored_job
        assert stored_job["output_file"] == "eval_results_v2_20260718_110000.json"

        # Assert in-memory Job.result IS still set
        assert tracker.get_job("job_moto").result == result_payload
