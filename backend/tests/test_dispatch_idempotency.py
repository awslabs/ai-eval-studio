# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for dispatch idempotency + reconciler/cap read helpers.

Tests the idempotency-key claim (collapses duplicates within TTL), per-owner
active job count (for caps), and cross-user non-terminal job scan (for
reconciler).
"""

import pytest
import time
from moto import mock_aws
import boto3
from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import FilesystemMetadataStore


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema (inline helper from integration tests)."""
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


@pytest.mark.asyncio
async def test_claim_idempotency_key_first_wins_cloud():
    """First claim returns (True, None), second returns (False, original job_id) (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        idempotency_key = "eval-click-123"
        job_id_1 = "job-1"
        job_id_2 = "job-2"
        ttl_seconds = 300

        # First claim succeeds
        claimed_1, existing_job_1 = await store.claim_idempotency_key(
            owner_id, idempotency_key, job_id_1, ttl_seconds
        )
        assert claimed_1 is True, "First claim should succeed"
        assert existing_job_1 is None, "First claim should have no existing job"

        # Duplicate claim fails, returns original job_id
        claimed_2, existing_job_2 = await store.claim_idempotency_key(
            owner_id, idempotency_key, job_id_2, ttl_seconds
        )
        assert claimed_2 is False, "Duplicate claim should fail"
        assert existing_job_2 == job_id_1, "Duplicate claim should return first job's id"


@pytest.mark.asyncio
async def test_two_different_keys_both_succeed_cloud():
    """Two different keys both succeed (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        key_1 = "eval-click-123"
        key_2 = "eval-click-456"
        job_id_1 = "job-1"
        job_id_2 = "job-2"
        ttl_seconds = 300

        # Both claims succeed (different keys)
        claimed_1, _ = await store.claim_idempotency_key(
            owner_id, key_1, job_id_1, ttl_seconds
        )
        claimed_2, _ = await store.claim_idempotency_key(
            owner_id, key_2, job_id_2, ttl_seconds
        )

        assert claimed_1 is True, "First key should succeed"
        assert claimed_2 is True, "Second key should succeed"


@pytest.mark.asyncio
async def test_count_active_jobs_for_owner_cloud():
    """count_active_jobs_for_owner returns non-terminal count (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"

        # Seed jobs: 2 running, 1 queued, 1 completed
        await store.put_job({
            "job_id": "job-1",
            "project_id": project_id,
            "owner_id": owner_id,
            "state": "running",
            "created_at": "2026-01-01T00:00:00Z",
        })
        await store.put_job({
            "job_id": "job-2",
            "project_id": project_id,
            "owner_id": owner_id,
            "state": "running",
            "created_at": "2026-01-01T00:01:00Z",
        })
        await store.put_job({
            "job_id": "job-3",
            "project_id": project_id,
            "owner_id": owner_id,
            "state": "queued",
            "created_at": "2026-01-01T00:02:00Z",
        })
        await store.put_job({
            "job_id": "job-4",
            "project_id": project_id,
            "owner_id": owner_id,
            "state": "completed",
            "created_at": "2026-01-01T00:03:00Z",
        })

        # Count active (non-terminal) jobs
        active_count = await store.count_active_jobs_for_owner(owner_id)
        assert active_count == 3, "Should count 2 running + 1 queued (not completed)"


@pytest.mark.asyncio
async def test_list_non_terminal_jobs_cloud():
    """list_non_terminal_jobs returns running+queued across owners (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Seed jobs for two different owners
        await store.put_job({
            "job_id": "job-1",
            "project_id": "project-1",
            "owner_id": "owner-1",
            "state": "running",
            "created_at": "2026-01-01T00:00:00Z",
        })
        await store.put_job({
            "job_id": "job-2",
            "project_id": "project-2",
            "owner_id": "owner-2",
            "state": "queued",
            "created_at": "2026-01-01T00:01:00Z",
        })
        await store.put_job({
            "job_id": "job-3",
            "project_id": "project-1",
            "owner_id": "owner-1",
            "state": "completed",
            "created_at": "2026-01-01T00:02:00Z",
        })

        # List non-terminal (running + queued) across all owners
        non_terminal = await store.list_non_terminal_jobs()
        job_ids = {j["job_id"] for j in non_terminal}

        assert len(non_terminal) == 2, "Should return 2 non-terminal jobs"
        assert "job-1" in job_ids, "Should include running job"
        assert "job-2" in job_ids, "Should include queued job"
        assert "job-3" not in job_ids, "Should NOT include completed job"


# --- Dev adapter parity tests ---


@pytest.mark.asyncio
async def test_claim_idempotency_key_dev(tmp_path):
    """Dev adapter: first claim wins, duplicate returns original job_id."""
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    owner_id = "owner-1"
    key = "eval-click-123"
    job_id_1 = "job-1"
    job_id_2 = "job-2"
    ttl_seconds = 300

    # First claim succeeds
    claimed_1, existing_1 = await store.claim_idempotency_key(
        owner_id, key, job_id_1, ttl_seconds
    )
    assert claimed_1 is True, "First claim should succeed (dev)"
    assert existing_1 is None, "First claim should have no existing job (dev)"

    # Duplicate claim fails, returns original job_id
    claimed_2, existing_2 = await store.claim_idempotency_key(
        owner_id, key, job_id_2, ttl_seconds
    )
    assert claimed_2 is False, "Duplicate claim should fail (dev)"
    assert existing_2 == job_id_1, "Duplicate claim should return first job's id (dev)"


@pytest.mark.asyncio
async def test_count_active_and_list_non_terminal_dev(tmp_path):
    """Dev adapter: count active + list non-terminal parity."""
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    owner_id = "owner-1"
    project_id = "project-1"

    # Seed jobs
    await store.put_job({
        "job_id": "job-1",
        "project_id": project_id,
        "owner_id": owner_id,
        "state": "running",
        "created_at": "2026-01-01T00:00:00Z",
    })
    await store.put_job({
        "job_id": "job-2",
        "project_id": project_id,
        "owner_id": owner_id,
        "state": "queued",
        "created_at": "2026-01-01T00:01:00Z",
    })
    await store.put_job({
        "job_id": "job-3",
        "project_id": project_id,
        "owner_id": owner_id,
        "state": "completed",
        "created_at": "2026-01-01T00:02:00Z",
    })

    # Count active
    active_count = await store.count_active_jobs_for_owner(owner_id)
    assert active_count == 2, "Should count running + queued (dev)"

    # List non-terminal
    non_terminal = await store.list_non_terminal_jobs()
    job_ids = {j["job_id"] for j in non_terminal}
    assert len(non_terminal) == 2, "Should return 2 non-terminal jobs (dev)"
    assert "job-1" in job_ids
    assert "job-2" in job_ids
    assert "job-3" not in job_ids
