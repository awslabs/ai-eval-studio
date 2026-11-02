# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cross-process project sequence number allocation.

Tests both cloud (DynamoDB with moto) and dev (filesystem) adapters for
the project seq allocator. The allocator
is the single authority for `project:<id>` stream sequence numbers,
replacing the per-process in-memory counter, under which two producers in
one coordinator lifetime could both issue seq=0.

Mutation proofs:
- Mutation 1 (cloud concurrency): replace atomic ADD with read-then-write
- Mutation 2 (cloud namespace): change PK to `self._user_pk(...)` with `SK=f"PROJECT#{pid}#SEQ"`
- Mutation 3 (dev cross-instance): make the counter an instance attribute
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from moto import mock_aws

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import FilesystemMetadataStore


# --- Cloud adapter (DynamoDB with moto) ---


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
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )


@pytest.fixture
def moto_store() -> DynamoDBMetadataStore:
    """Cloud metadata store backed by moto-mocked DynamoDB.

    Reuses the test_microvm_job_visibility.py fixture pattern.
    """
    with mock_aws():
        import boto3

        ddb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-metadata-table"
        _create_metadata_table(ddb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=ddb_client, table_name=table_name)
        yield store


@pytest.mark.asyncio
async def test_cloud_sequential_calls_strictly_increasing(moto_store):
    """Two sequential allocate_project_seq calls return strictly increasing ints.

    First call returns 1 (the counter starts at 1, not 0 — matches turn/job seq).
    """
    pid = "test-proj-seq-01"
    seq1 = await moto_store.allocate_project_seq(pid)
    seq2 = await moto_store.allocate_project_seq(pid)

    assert seq1 == 1, "First allocation must return 1"
    assert seq2 == 2, "Second allocation must return 2"
    assert seq2 > seq1, "Sequence numbers must be strictly increasing"


@pytest.mark.asyncio
async def test_cloud_concurrent_calls_no_repeats(moto_store):
    """20 concurrent allocations return 20 distinct values (anti-collision).

    This is the property per-process counters violate: two producers
    issuing the same seq on one project stream.

    Mutation proof 1: replace the atomic ADD with read-then-write. The test
    will go red because concurrent writes collide and produce duplicate seqs.
    """
    pid = "test-proj-concurrent"

    # Spawn 20 concurrent allocations
    tasks = [moto_store.allocate_project_seq(pid) for _ in range(20)]
    seqs = await asyncio.gather(*tasks)

    # All must be distinct
    assert len(set(seqs)) == 20, f"Expected 20 distinct seqs, got {len(set(seqs))}: {sorted(seqs)}"
    # All must be in range [1, 20]
    assert min(seqs) == 1
    assert max(seqs) == 20


@pytest.mark.asyncio
async def test_cloud_independent_counters_per_project(moto_store):
    """Two different project_ids have independent counters.

    Allocating for project A does not advance project B's counter.
    """
    pid_a = "test-proj-a"
    pid_b = "test-proj-b"

    seq_a1 = await moto_store.allocate_project_seq(pid_a)
    seq_b1 = await moto_store.allocate_project_seq(pid_b)
    seq_a2 = await moto_store.allocate_project_seq(pid_a)

    assert seq_a1 == 1
    assert seq_b1 == 1, "Project B starts at 1, independent of project A"
    assert seq_a2 == 2


@pytest.mark.asyncio
async def test_cloud_counter_invisible_to_list_user_projects(moto_store):
    """Counter item does not appear in list_user_projects.

    The counter lives under PROJECT_SEQ#{project_id}, not USER#{uid}, so
    the two `begins_with(SK, "PROJECT#")` queries on the user partition
    are provably unaffected.

    Mutation proof 2: change the PK to `self._user_pk(owner_id)` with
    `SK=f"PROJECT#{pid}#SEQ"`. The test will go red because the counter
    item now appears in the user partition and either breaks the query
    or requires a filter that is not present.
    """
    owner_id = "test-user-001"
    pid = "test-proj-namespace"

    # Create a real project record
    await moto_store.put_project({
        "id": pid,
        "owner_id": owner_id,
        "name": "Test Project",
        "created_at": "2026-08-07T00:00:00Z",
    })

    # Allocate a seq for the project
    await moto_store.allocate_project_seq(pid)

    # List the user's projects
    projects = await moto_store.list_user_projects(owner_id)

    # Should see exactly one project (the real one, not the counter item)
    assert len(projects) == 1
    assert projects[0]["id"] == pid


# --- Dev adapter (filesystem) ---


@pytest.mark.asyncio
async def test_dev_cross_instance_persistence(tmp_path: Path):
    """Counter allocated through one instance continues in a second instance.

    This is the cross-process property the whole fix rests on: two publishing
    processes (coordinator + microVM worker) drawing from the same counter
    rather than separate per-process counters.

    Mutation proof 3: make the counter an instance attribute instead of a
    persisted file. The test will go red because the second instance starts
    at 1, not 3.
    """
    pid = "test-proj-dev-persist"

    # First instance: allocate two seqs
    store1 = FilesystemMetadataStore(projects_dir=tmp_path)
    seq1 = await store1.allocate_project_seq(pid)
    seq2 = await store1.allocate_project_seq(pid)

    assert seq1 == 1
    assert seq2 == 2

    # Second instance (separate object, same projects_dir) — stands in for
    # a second publishing process
    store2 = FilesystemMetadataStore(projects_dir=tmp_path)
    seq3 = await store2.allocate_project_seq(pid)

    assert seq3 == 3, "Second instance must continue the counter, not restart at 1"


@pytest.mark.asyncio
async def test_dev_sequential_calls(tmp_path: Path):
    """Dev adapter returns strictly increasing ints, first call returns 1."""
    store = FilesystemMetadataStore(projects_dir=tmp_path)
    pid = "test-proj-dev-seq"

    seq1 = await store.allocate_project_seq(pid)
    seq2 = await store.allocate_project_seq(pid)

    assert seq1 == 1
    assert seq2 == 2
    assert seq2 > seq1
