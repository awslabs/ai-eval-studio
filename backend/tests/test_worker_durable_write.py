# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobTracker.create_job(durable_write=False) — worker record-clobber fix.

The detached Fargate worker calls ``create_job`` for the SAME job_id the
coordinator already registered. Before this fix, the worker's create_job
scheduled a full-item ``put_job`` write-through (the app's only full-item
put_item writer) which ran tens of seconds after the coordinator's
``compute_handle`` write and OVERWROTE the durable record — wiping the
handle, re-freezing state to ``queued``, and replacing config with the
worker's sparse shape.

``durable_write=False`` makes the worker register the in-memory Job (+ replay
provider) WITHOUT the put_job write-through, so the coordinator-owned durable
record survives. These tests trip that guard behaviorally:

* ``create_job(durable_write=False)`` fires NO put_job (mocked store.put_job
  is never called), yet the in-memory Job + replay provider are still
  registered.
* A coordinator-written ``compute_handle`` on the durable record survives a
  subsequent worker-shaped ``create_job(durable_write=False)`` for the same
  job_id (via moto DDB round-trip).
* ``create_job(durable_write=True)`` (the default) still fires put_job — the
  coordinator's create-time write-through is byte-behaviorally unchanged.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.services.jobs import JobTracker


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema (mirrors tests/test_event_log.py)."""
    ddb_client.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def tracker():
    """Fresh JobTracker with a clean in-memory job map for each test."""
    t = JobTracker()
    t._reset()
    return t


@pytest.mark.asyncio
async def test_create_job_durable_write_false_fires_no_put_job(tracker):
    """durable_write=False: no put_job, but in-memory Job + replay provider registered."""
    mock_store = AsyncMock()

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="owner-1",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store
            with patch("app.services.ws_hub.get_ws_hub") as mock_hub:
                register = MagicMock()
                mock_hub.return_value.register_replay_provider = register

                job = tracker.create_job(
                    "proj-1",
                    "eval",
                    config={"arch_split": True, "owner_id": "owner-1"},
                    job_id="job-worker-1",
                    durable_write=False,
                )

                # Give any (erroneously scheduled) background write a chance to run.
                await asyncio.sleep(0.2)

    # No durable put_job fired — the worker must not clobber the record.
    mock_store.put_job.assert_not_called()

    # In-memory Job is still registered so append_event / progress work.
    assert tracker.get_job("job-worker-1") is job

    # Replay provider is still registered so a reconnecting SPA gets progress.
    register.assert_called_once()
    assert register.call_args.args[0] == "job:job-worker-1"


@pytest.mark.asyncio
async def test_create_job_durable_write_true_still_fires_put_job(tracker):
    """durable_write=True (default): coordinator's create-time put_job is preserved."""
    mock_store = AsyncMock()

    with patch(
        "app.services.agent_turn._resolve_owner_id_for_project",
        return_value="owner-1",
    ):
        with patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store
            with patch("app.services.ws_hub.get_ws_hub") as mock_hub:
                mock_hub.return_value.register_replay_provider = MagicMock()

                tracker.create_job(
                    "proj-1",
                    "eval",
                    config={"arch_split": True, "user_id": "owner-1"},
                    job_id="job-coord-1",
                    # durable_write defaults to True
                )

                await asyncio.sleep(0.2)

    mock_store.put_job.assert_called_once()


@pytest.mark.asyncio
async def test_coordinator_compute_handle_survives_worker_create_job(tracker):
    """A coordinator-written compute_handle survives the worker's create_job(durable_write=False).

    This reproduces the live clobber via a moto DDB round-trip:
    1. Coordinator put_job + update_job_state(compute_handle=...) writes the record.
    2. Worker create_job(durable_write=False) for the SAME job_id.
    3. The durable record STILL carries the coordinator's compute_handle.
    """
    from moto import mock_aws

    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb, "test-table")
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name="test-table")

        owner_id = "owner-1"
        project_id = "proj-1"
        job_id = "job-shared-1"

        # 1. Coordinator writes the durable record + compute_handle.
        await store.put_job(
            {
                "owner_id": owner_id,
                "project_id": project_id,
                "job_id": job_id,
                "type": "eval",
                "state": "queued",
                "progress": {"message": "Queued"},
                "created_at": "2026-07-16T12:00:00Z",
                "config": {"arch_split": True, "owner_id": owner_id},
            }
        )
        await store.update_job_state(
            owner_id, project_id, job_id,
            compute_handle="arn:aws:ecs:us-east-1:123:task/abc",
        )

        # 2. Worker registers the SAME job_id with durable_write=False.
        with patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value=owner_id,
        ):
            with patch("app.platform.get_platform") as mock_platform:
                mock_platform.return_value.metadata_store = store
                with patch("app.services.ws_hub.get_ws_hub") as mock_hub:
                    mock_hub.return_value.register_replay_provider = MagicMock()

                    tracker.create_job(
                        project_id,
                        "eval",
                        config={"arch_split": True, "owner_id": owner_id},
                        job_id=job_id,
                        durable_write=False,
                    )
                    await asyncio.sleep(0.2)

        # 3. The durable record STILL carries the coordinator's compute_handle.
        record = await store.get_job(owner_id, project_id, job_id, consistent=True)
        assert record is not None
        assert record.get("compute_handle") == "arn:aws:ecs:us-east-1:123:task/abc", (
            "worker create_job(durable_write=False) must NOT clobber the "
            "coordinator's compute_handle"
        )
