# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for eval concurrency lock.

Tests the claim/release/version-fenced-clear lifecycle across DynamoDB and dev
implementations. The eval lock enforces one-eval-per-project via conditional
writes, releases on every terminal state, and supports reconciler-based stale-
lock clearing with version fencing to prevent race conditions.
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
async def test_claim_eval_lock_first_wins_cloud():
    """First claim succeeds, second claim fails (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id_1 = "job-1"
        job_id_2 = "job-2"

        # First claim succeeds
        claimed_1 = await store.claim_eval_lock(owner_id, project_id, job_id_1)
        assert claimed_1 is True, "First claim should succeed"

        # Second claim on same project fails
        claimed_2 = await store.claim_eval_lock(owner_id, project_id, job_id_2)
        assert claimed_2 is False, "Second concurrent claim should fail"


@pytest.mark.asyncio
async def test_release_then_reclaim_succeeds_cloud():
    """Release → reclaim succeeds (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id_1 = "job-1"
        job_id_2 = "job-2"

        # Claim + release
        await store.claim_eval_lock(owner_id, project_id, job_id_1)
        await store.release_eval_lock(owner_id, project_id, job_id_1)

        # Reclaim succeeds
        claimed = await store.claim_eval_lock(owner_id, project_id, job_id_2)
        assert claimed is True, "Reclaim after release should succeed"


@pytest.mark.asyncio
async def test_release_by_non_holder_is_noop_cloud():
    """Release by non-holder is a no-op (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id_holder = "job-1"
        job_id_other = "job-2"

        # Claim by job-1
        await store.claim_eval_lock(owner_id, project_id, job_id_holder)

        # Release by job-2 (non-holder) is a no-op
        await store.release_eval_lock(owner_id, project_id, job_id_other)

        # Lock still held — next claim fails
        claimed = await store.claim_eval_lock(owner_id, project_id, job_id_other)
        assert claimed is False, "Lock should still be held after non-holder release"


@pytest.mark.asyncio
async def test_get_eval_lock_returns_version_cloud():
    """get_eval_lock returns lock_version for reconciler (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id = "job-1"

        # No lock initially
        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is None, "No lock should exist initially"

        # Claim lock
        await store.claim_eval_lock(owner_id, project_id, job_id)

        # Get lock returns version=1
        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is not None, "Lock should exist after claim"
        assert lock["lock_version"] == 1, "Initial lock version should be 1"
        assert lock["lock_holder_job_id"] == job_id, "Holder should match"


@pytest.mark.asyncio
async def test_clear_stale_eval_lock_with_matching_version_cloud():
    """clear_stale_eval_lock with matching version deletes (cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id = "job-1"

        # Claim lock
        await store.claim_eval_lock(owner_id, project_id, job_id)
        lock = await store.get_eval_lock(owner_id, project_id)
        read_version = lock["lock_version"]

        # Clear with matching version AND holder succeeds
        cleared = await store.clear_stale_eval_lock(
            owner_id, project_id, read_version, job_id
        )
        assert cleared is True, "Clear with matching version and holder should succeed"

        # Lock no longer exists
        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is None, "Lock should be gone after clear"


@pytest.mark.asyncio
async def test_clear_stale_eval_lock_with_mismatched_version_fails_cloud():
    """clear_stale_eval_lock with mismatched version returns False (version fence, cloud path)."""
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"
        job_id_1 = "job-1"
        job_id_2 = "job-2"

        # Claim lock
        await store.claim_eval_lock(owner_id, project_id, job_id_1)
        lock = await store.get_eval_lock(owner_id, project_id)
        stale_version = lock["lock_version"]

        # Release + reclaim (version changes)
        await store.release_eval_lock(owner_id, project_id, job_id_1)
        await store.claim_eval_lock(owner_id, project_id, job_id_2)

        # Clear with stale version fails (version fence)
        cleared = await store.clear_stale_eval_lock(
            owner_id, project_id, stale_version, job_id_1
        )
        assert cleared is False, "Clear with stale version should fail (version fence)"

        # Lock still held by job-2
        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is not None, "Lock should still exist after failed clear"
        assert lock["lock_holder_job_id"] == job_id_2, "Lock should still be held by job-2"


@pytest.mark.asyncio
async def test_clear_stale_eval_lock_with_current_version_but_wrong_holder_fails_cloud():
    """The holder fence catches what the version fence cannot.

    A lock reclaimed by another job carries that job's CURRENT version, so a
    reconciler reading the lock fresh sees a matching version and a
    version-only conditional delete succeeds — removing a live job's lock.
    """
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(
            ddb_client=dynamodb_client, table_name=table_name
        )

        owner_id = "owner-1"
        project_id = "project-1"

        # job-1 held the lock, released it; job-2 holds it now.
        await store.claim_eval_lock(owner_id, project_id, "job-1")
        await store.release_eval_lock(owner_id, project_id, "job-1")
        await store.claim_eval_lock(owner_id, project_id, "job-2")

        # Read the CURRENT version, as the reconciler does.
        lock = await store.get_eval_lock(owner_id, project_id)
        current_version = lock["lock_version"]
        assert lock["lock_holder_job_id"] == "job-2"

        # Reconciling stale job-1 must not remove job-2's lock.
        cleared = await store.clear_stale_eval_lock(
            owner_id, project_id, current_version, "job-1"
        )
        assert cleared is False, "the holder fence must reject a foreign holder"

        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is not None, "job-2's live lock must survive"
        assert lock["lock_holder_job_id"] == "job-2"


# --- Dev adapter parity tests ---


@pytest.mark.asyncio
async def test_claim_release_dev():
    """Dev adapter: claim → release → reclaim parity."""
    store = FilesystemMetadataStore()

    owner_id = "owner-1"
    project_id = "project-1"
    job_id_1 = "job-1"
    job_id_2 = "job-2"

    # First claim succeeds
    claimed_1 = await store.claim_eval_lock(owner_id, project_id, job_id_1)
    assert claimed_1 is True, "First claim should succeed (dev)"

    # Second claim fails
    claimed_2 = await store.claim_eval_lock(owner_id, project_id, job_id_2)
    assert claimed_2 is False, "Second claim should fail (dev)"

    # Release + reclaim
    await store.release_eval_lock(owner_id, project_id, job_id_1)
    claimed_3 = await store.claim_eval_lock(owner_id, project_id, job_id_2)
    assert claimed_3 is True, "Reclaim after release should succeed (dev)"


@pytest.mark.asyncio
async def test_clear_stale_eval_lock_dev():
    """Dev adapter: clear with version + holder check parity."""
    store = FilesystemMetadataStore()

    owner_id = "owner-1"
    project_id = "project-1"
    job_id = "job-1"

    # Claim lock
    await store.claim_eval_lock(owner_id, project_id, job_id)
    lock = await store.get_eval_lock(owner_id, project_id)
    read_version = lock["lock_version"]

    # Clear with matching version and holder succeeds
    cleared = await store.clear_stale_eval_lock(
        owner_id, project_id, read_version, job_id
    )
    assert cleared is True, "Clear with matching version and holder should succeed (dev)"

    # Lock gone
    lock = await store.get_eval_lock(owner_id, project_id)
    assert lock is None, "Lock should be gone after clear (dev)"


@pytest.mark.asyncio
async def test_clear_stale_eval_lock_wrong_holder_dev():
    """Dev parity for the holder fence: a foreign holder's lock survives."""
    store = FilesystemMetadataStore()

    owner_id = "owner-1"
    project_id = "project-1"

    await store.claim_eval_lock(owner_id, project_id, "job-1")
    await store.release_eval_lock(owner_id, project_id, "job-1")
    await store.claim_eval_lock(owner_id, project_id, "job-2")

    lock = await store.get_eval_lock(owner_id, project_id)
    cleared = await store.clear_stale_eval_lock(
        owner_id, project_id, lock["lock_version"], "job-1"
    )
    assert cleared is False, "the holder fence must reject a foreign holder (dev)"
    assert (await store.get_eval_lock(owner_id, project_id)) is not None


# ---------------------------------------------------------------------------
# Atomic lock_version counter fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_version_monotonic_across_release_reclaim():
    """lock_version is strictly increasing across claim→release→claim.

    This test validates that the version counter increments monotonically
    even across release/reclaim cycles, which is essential for the reconciler's
    version-fenced clear to work correctly.
    """
    with mock_aws():
        # Setup
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "project-1"

        # First claim → should get version 1
        job_id_1 = "job-1"
        claimed_1 = await store.claim_eval_lock(owner_id, project_id, job_id_1)
        assert claimed_1 is True, "First claim should succeed"

        lock_1 = await store.get_eval_lock(owner_id, project_id)
        version_1 = lock_1["lock_version"]
        assert version_1 == 1, f"First claim should get version 1, got: {version_1}"

        # Release
        await store.release_eval_lock(owner_id, project_id, job_id_1)

        # Second claim → should get version 2 (strictly > version 1)
        job_id_2 = "job-2"
        claimed_2 = await store.claim_eval_lock(owner_id, project_id, job_id_2)
        assert claimed_2 is True, "Second claim should succeed"

        lock_2 = await store.get_eval_lock(owner_id, project_id)
        version_2 = lock_2["lock_version"]

        # CRITICAL: Version must be strictly increasing (2, not 1 again)
        assert version_2 > version_1, \
            f"Second claim version ({version_2}) must be > first version ({version_1}), " \
            f"but got same/lower (version reuse bug)"
        assert version_2 == 2, f"Second claim should get version 2, got: {version_2}"

        # Third claim (after another release) → version 3
        await store.release_eval_lock(owner_id, project_id, job_id_2)

        job_id_3 = "job-3"
        claimed_3 = await store.claim_eval_lock(owner_id, project_id, job_id_3)
        assert claimed_3 is True, "Third claim should succeed"

        lock_3 = await store.get_eval_lock(owner_id, project_id)
        version_3 = lock_3["lock_version"]

        assert version_3 > version_2, \
            f"Third claim version ({version_3}) must be > second version ({version_2})"
        assert version_3 == 3, f"Third claim should get version 3, got: {version_3}"
