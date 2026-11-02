# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for durable-state round-trips against emulated DynamoDB.

These tests cover two behaviors that are impractical to exercise by hand
(restart-recover, optimization-state store round-trip): ``scripts/start.sh``
runs dev mode (no DynamoDB), there is no DynamoDB-Local wiring, and cloud mode
requires Cognito. Instead of asserting mocked *calls* (as the unit
suite does), these tests drive the REAL ``DynamoDBMetadataStore`` adapter against
a moto-emulated DynamoDB table — a genuine write → read → reconstruct round-trip.

The pattern is: stand up the table with moto, exercise the production code,
assert the durable behavior.

Marked ``@pytest.mark.integration`` so it skips by default and only runs under
``pytest -m integration``.

Covers:
1. A job written to DynamoDB is rebuilt into a FRESH JobTracker
   (simulating a control-plane restart with an empty in-memory cache) with
   matching state, progress, and metadata; the by-owner query answers too.
2. OptimizationState written via the single-writer conditional-write
   funnel reads back byte-for-byte with an incremented version, and a stale
   ``expected_version`` raises ``ConcurrentUpdateError`` (the loud seatbelt).
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def _import_moto():
    """Return moto.mock_aws, or None if moto isn't installed."""
    try:
        moto = importlib.import_module("moto")
    except ImportError:
        return None
    return getattr(moto, "mock_aws", None)


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema the DynamoDBMetadataStore expects.

    PK/SK base keys + GSI1 (GSI1PK/GSI1SK) for the by-state ops query.
    Matches the schema documented at the top of dynamodb_metadata_store.py.
    """
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_survives_restart_via_ddb_rebuild():
    """A job persisted to DynamoDB is rebuilt into a fresh JobTracker.

    Simulates a control-plane restart: write the job to the durable store, then
    build a brand-new JobTracker (empty in-memory cache, as a restarted process
    would have) and call rebuild_jobs_from_store. Assert the job comes back with
    matching state/progress/metadata, and that the sync read paths and the
    by-owner query answer from the rebuilt cache.
    """
    mock_aws = _import_moto()
    if mock_aws is None:
        pytest.skip(
            "moto is not installed. Add `moto>=5.0.0` to backend dev "
            "dependencies and rerun with `pytest -m integration`."
        )

    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.services.jobs import JobTracker

    owner_id = "owner-abc"
    project_id = "proj-123"
    job_id = "job-xyz"
    table_name = "test-metadata"

    with mock_aws():
        import boto3

        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb, table_name=table_name)

        # --- Write the job to DynamoDB as the write-through path would ---
        job_record = {
            "owner_id": owner_id,
            "project_id": project_id,
            "job_id": job_id,
            "type": "eval",
            "state": "running",
            "progress": {"message": "Scoring 3/5", "models_done": 1, "models_total": 2},
            "created_at": "2026-07-02T12:00:00Z",
            "config": {"model_ids": ["m1", "m2"], "arch_split": True},
        }
        await store.put_job(job_record)

        # Sanity: the item is really in DynamoDB (direct adapter read).
        fetched = await store.get_job(owner_id, project_id, job_id)
        assert fetched is not None
        assert fetched["state"] == "running"

        # --- Simulate a control-plane restart: FRESH tracker, empty cache ---
        fresh_tracker = JobTracker()
        assert fresh_tracker.get_job(job_id) is None  # nothing in memory yet

        # Point the platform singleton's metadata_store at our moto store, and
        # make the server-side owner resolver return our owner (rebuild uses both).
        fake_platform = type("P", (), {"metadata_store": store})()
        with patch("app.platform.get_platform", return_value=fake_platform), patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value=owner_id,
        ):
            await fresh_tracker.rebuild_jobs_from_store(project_id, arch_split=True)

        # --- Assert the rebuild recovered the job with matching metadata ---
        rebuilt = fresh_tracker.get_job(job_id)
        assert rebuilt is not None, "job must be rebuilt from DynamoDB after restart"
        assert rebuilt.job_id == job_id
        assert rebuilt.project_id == project_id
        assert rebuilt.type == "eval"
        assert rebuilt.state == "running"
        assert rebuilt.created_at == "2026-07-02T12:00:00Z"
        assert rebuilt.progress["message"] == "Scoring 3/5"
        assert rebuilt.progress["models_total"] == 2
        assert rebuilt.config["model_ids"] == ["m1", "m2"]

        # The sync list_jobs read path serves the rebuilt job for the project.
        listed = fresh_tracker.list_jobs(project_id)
        assert [j.job_id for j in listed] == [job_id]

        # The "jobs by owner" query answers from DynamoDB across projects.
        by_owner = await store.list_jobs_by_owner(owner_id)
        assert any(j["job_id"] == job_id for j in by_owner)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rebuild_does_not_clobber_live_jobs():
    """Cache-as-truth: a job already live in the cache is NOT overwritten
    by a stale store copy during rebuild (store fills gaps only)."""
    mock_aws = _import_moto()
    if mock_aws is None:
        pytest.skip("moto not installed; run with `pytest -m integration`.")

    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.services.jobs import JobTracker

    owner_id = "owner-abc"
    project_id = "proj-123"
    table_name = "test-metadata"

    with mock_aws():
        import boto3

        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb, table_name=table_name)

        # Store has a STALE copy (state=queued).
        await store.put_job(
            {
                "owner_id": owner_id,
                "project_id": project_id,
                "job_id": "job-live",
                "type": "eval",
                "state": "queued",
                "progress": {"message": "Queued"},
                "created_at": "2026-07-02T12:00:00Z",
                "config": {"arch_split": True},
            }
        )

        # A live tracker already holds the SAME job in a newer state.
        tracker = JobTracker()
        tracker.create_job(
            project_id=project_id,
            job_type="eval",
            config={"arch_split": False},  # off → no store write from create
            job_id="job-live",
        )
        tracker.update_state("job-live", "running")
        assert tracker.get_job("job-live").state == "running"

        fake_platform = type("P", (), {"metadata_store": store})()
        with patch("app.platform.get_platform", return_value=fake_platform), patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value=owner_id,
        ):
            await tracker.rebuild_jobs_from_store(project_id, arch_split=True)

        # Live cache wins — the stale queued copy must NOT clobber running.
        assert tracker.get_job("job-live").state == "running"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_optimization_state_roundtrip_and_conditional_write():
    """OptimizationState written via the conditional-write funnel reads
    back with matching data + incremented version; a stale expected_version raises
    ConcurrentUpdateError (the loud single-writer seatbelt).
    """
    mock_aws = _import_moto()
    if mock_aws is None:
        pytest.skip("moto not installed; run with `pytest -m integration`.")

    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.platform.metadata_store import ConcurrentUpdateError

    user_id = "owner-abc"
    project_id = "proj-123"
    table_name = "test-metadata"

    with mock_aws():
        import boto3

        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb, table_name=table_name)

        # First write (no prior record): expected_version=None → attribute_not_exists guard.
        state_v1 = {
            "status": "running",
            "current_iteration": 1,
            "iterations": [
                {"iteration": 1, "prompt_version": "v1", "passed": 8, "total_cases": 10}
            ],
            "winning_model_id": None,
        }
        v1 = await store.put_optimization_state(user_id, project_id, state_v1)
        assert v1 == 1, "first write starts at version 1"

        # Read back: data round-trips and carries version 1.
        read1 = await store.get_optimization_state(user_id, project_id)
        assert read1 is not None
        assert read1["status"] == "running"
        assert read1["current_iteration"] == 1
        assert read1["iterations"][0]["prompt_version"] == "v1"
        assert read1["iterations"][0]["passed"] == 8
        assert read1["version"] == 1

        # Second write threads the RETURNED version (v1), not a guess.
        state_v2 = {**state_v1, "current_iteration": 2}
        v2 = await store.put_optimization_state(
            user_id, project_id, state_v2, expected_version=v1
        )
        assert v2 == 2

        read2 = await store.get_optimization_state(user_id, project_id)
        assert read2["current_iteration"] == 2
        assert read2["version"] == 2

        # Stale expected_version (still 1 after it's already 2) → loud failure.
        with pytest.raises(ConcurrentUpdateError):
            await store.put_optimization_state(
                user_id, project_id, {**state_v2, "current_iteration": 99},
                expected_version=1,
            )

        # The failed conditional write did NOT mutate the stored state.
        read3 = await store.get_optimization_state(user_id, project_id)
        assert read3["current_iteration"] == 2
        assert read3["version"] == 2
