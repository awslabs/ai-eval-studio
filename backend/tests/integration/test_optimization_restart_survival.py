# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: optimization survives a control-plane restart.

The optimization loop dispatches each iteration's eval through the SAME
``JobCompute`` seam as the standalone path (proven by unit tests in
``test_optimization_dispatch.py``). Because the job record — including the
durable ``compute_handle`` (the Fargate task ARN) — lives in DynamoDB, an
optimization run is NOT lost when the stateless control plane rolls: a fresh
process with an empty in-memory ``JobTracker`` can rebuild the active-jobs
view from the durable store and the run's compute handle is still present for
cancel/status.

This is the durable half of the proof (browser-close is inherently covered —
the loop has no browser dependency). We drive the REAL
``DynamoDBMetadataStore`` against a moto-emulated table, write an eval job
carrying a ``compute_handle`` (as the new-path optimization dispatch would),
then rebuild a fresh ``JobTracker`` and assert the job + handle survive.

Mirrors ``test_durable_state_roundtrip.py``'s moto harness. Marked
``@pytest.mark.integration`` so it skips by default.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema the DynamoDBMetadataStore expects."""
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
async def test_optimization_iteration_survives_control_plane_restart():
    """An in-flight optimization iteration + its compute_handle survive a restart.

    Write an optimization-iteration eval job to DynamoDB carrying a Fargate
    task-ARN compute_handle (as the new-path optimization dispatch persists
    it), then simulate a control-plane restart by rebuilding a FRESH
    JobTracker from the durable store and assert the job comes back active
    with its compute_handle intact — so cancel/status can still target the
    running worker after the roll.
    """
    # moto is a locked dev dependency (moto>=5.0.0), so importorskip only skips
    # on a genuinely missing install — it will NOT paper over an import that
    # breaks mid-session (e.g. a sibling test mutating the environment). That
    # failure must surface, not silently drop this proof.
    mock_aws = pytest.importorskip("moto").mock_aws

    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
    from app.services.jobs import JobTracker

    owner_id = "owner-opt-abc"
    project_id = "proj-opt-123"
    job_id = "eval-iter-xyz"
    task_arn = "arn:aws:ecs:us-east-1:123456789012:task/eval-cluster/iter-xyz"
    table_name = "test-metadata"

    with mock_aws():
        import boto3

        ddb = boto3.client("dynamodb", region_name="us-east-1")
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb, table_name=table_name)

        # --- New-path optimization dispatch writes the iteration eval job ---
        job_record = {
            "owner_id": owner_id,
            "project_id": project_id,
            "job_id": job_id,
            "type": "eval",
            "state": "running",
            "progress": {"message": "Scoring 5/10"},
            "created_at": "2026-07-03T09:00:00Z",
            "config": {
                "arch_split": True,
                "owner_id": owner_id,
                "source": "optimization",
                "model_ids": ["m1"],
            },
        }
        await store.put_job(job_record)
        # ...and persists the compute_handle exactly as dispatch_eval_job does
        await store.update_job_state(
            owner_id, project_id, job_id, compute_handle=task_arn
        )

        # Sanity: the durable record carries the handle
        fetched = await store.get_job(owner_id, project_id, job_id)
        assert fetched is not None
        assert fetched["compute_handle"] == task_arn
        assert fetched["state"] == "running"

        # --- Simulate a control-plane restart: fresh tracker, empty cache ---
        fresh_tracker = JobTracker()
        assert fresh_tracker.get_job(job_id) is None

        fake_platform = type("P", (), {"metadata_store": store})()
        with patch("app.platform.get_platform", return_value=fake_platform), patch(
            "app.services.agent_turn._resolve_owner_id_for_project",
            return_value=owner_id,
        ):
            await fresh_tracker.rebuild_jobs_from_store(project_id, arch_split=True)

        # --- The optimization run is NOT lost: job rebuilt + active ---
        rebuilt = fresh_tracker.get_job(job_id)
        assert rebuilt is not None, "iteration job must survive the restart"
        assert rebuilt.state == "running"
        assert rebuilt.config["source"] == "optimization"
        assert rebuilt.config["arch_split"] is True

        # The active-jobs view rebuilds so /activity keeps reporting the run
        active = list(fresh_tracker.get_active_jobs(project_id))
        assert any(j.job_id == job_id for j in active), "run dropped from active view after restart"

        # The durable compute_handle is still there for cancel/status (StopTask)
        after_restart = await store.get_job(owner_id, project_id, job_id)
        assert after_restart["compute_handle"] == task_arn
