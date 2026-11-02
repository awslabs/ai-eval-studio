# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for list_jobs_by_owner: the "jobs by owner" query.

This module tests both the DynamoDB cloud adapter (Stubber-based) and the dev
filesystem adapter to enforce dev/cloud parity.
"""

import pytest
from botocore.stub import Stubber
from pathlib import Path

from app.platform.aws.dynamodb_metadata_store import (
    DynamoDBMetadataStore,
    _to_dynamo,
    _item_to_dict,
)
from app.platform.dev_local import FilesystemMetadataStore


# ---------------------------------------------------------------------------
# DynamoDB cloud adapter tests (Stubber-based)
# ---------------------------------------------------------------------------


class TestJobsByOwnerDDB:
    """Test list_jobs_by_owner against the DynamoDB adapter using Stubber."""

    @pytest.fixture
    def store(self):
        """Create a DynamoDBMetadataStore with a stubbed client."""
        import boto3

        client = boto3.client("dynamodb", region_name="us-east-1")
        store = DynamoDBMetadataStore(ddb_client=client, table_name="test-table")
        return store

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_base_table_query(self, store):
        """list_jobs_by_owner queries base table PK=USER#{owner}, no IndexName."""
        owner_id = "owner-123"

        with Stubber(store.client) as stubber:
            stubber.add_response(
                "query",
                {
                    "Items": [
                        {
                            "PK": _to_dynamo(f"USER#{owner_id}"),
                            "SK": _to_dynamo("PROJECT#proj-A#JOB#job-1"),
                            "job_id": _to_dynamo("job-1"),
                            "project_id": _to_dynamo("proj-A"),
                            "state": _to_dynamo("completed"),
                        },
                        {
                            "PK": _to_dynamo(f"USER#{owner_id}"),
                            "SK": _to_dynamo("PROJECT#proj-B#JOB#job-2"),
                            "job_id": _to_dynamo("job-2"),
                            "project_id": _to_dynamo("proj-B"),
                            "state": _to_dynamo("running"),
                        },
                    ]
                },
                {
                    "TableName": "test-table",
                    "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
                    "ExpressionAttributeValues": {
                        ":pk": _to_dynamo(f"USER#{owner_id}"),
                        ":sk": _to_dynamo("PROJECT#"),
                    },
                    "ScanIndexForward": False,
                },
            )

            jobs = await store.list_jobs_by_owner(owner_id)
            assert len(jobs) == 2
            assert jobs[0]["job_id"] == "job-1"
            assert jobs[1]["job_id"] == "job-2"

            stubber.assert_no_pending_responses()

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_filters_non_job_items(self, store):
        """list_jobs_by_owner excludes TURN/OPT/COST items sharing the owner partition."""
        owner_id = "owner-456"

        with Stubber(store.client) as stubber:
            stubber.add_response(
                "query",
                {
                    "Items": [
                        {
                            "PK": _to_dynamo(f"USER#{owner_id}"),
                            "SK": _to_dynamo("PROJECT#proj-A#JOB#job-1"),
                            "job_id": _to_dynamo("job-1"),
                        },
                        {
                            "PK": _to_dynamo(f"USER#{owner_id}"),
                            "SK": _to_dynamo("PROJECT#proj-A#TURN#turn-1"),
                            "turn_id": _to_dynamo("turn-1"),
                        },
                        {
                            "PK": _to_dynamo(f"USER#{owner_id}"),
                            "SK": _to_dynamo("PROJECT#proj-A#OPT"),
                            "version": _to_dynamo(1),
                        },
                    ]
                },
                {
                    "TableName": "test-table",
                    "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
                    "ExpressionAttributeValues": {
                        ":pk": _to_dynamo(f"USER#{owner_id}"),
                        ":sk": _to_dynamo("PROJECT#"),
                    },
                    "ScanIndexForward": False,
                },
            )

            jobs = await store.list_jobs_by_owner(owner_id)
            # Only the JOB item
            assert len(jobs) == 1
            assert jobs[0]["job_id"] == "job-1"

            stubber.assert_no_pending_responses()

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_returns_empty_when_none(self, store):
        """list_jobs_by_owner returns [] when no jobs exist."""
        owner_id = "owner-empty"

        with Stubber(store.client) as stubber:
            stubber.add_response(
                "query",
                {"Items": []},
                {
                    "TableName": "test-table",
                    "KeyConditionExpression": "PK = :pk AND begins_with(SK, :sk)",
                    "ExpressionAttributeValues": {
                        ":pk": _to_dynamo(f"USER#{owner_id}"),
                        ":sk": _to_dynamo("PROJECT#"),
                    },
                    "ScanIndexForward": False,
                },
            )

            jobs = await store.list_jobs_by_owner(owner_id)
            assert jobs == []

            stubber.assert_no_pending_responses()


# ---------------------------------------------------------------------------
# Dev filesystem adapter tests (parity)
# ---------------------------------------------------------------------------


class TestJobsByOwnerDev:
    """Test list_jobs_by_owner against the dev filesystem adapter for parity."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a FilesystemMetadataStore pointing at a temp directory."""
        store = FilesystemMetadataStore(projects_dir=Path(tmp_path))
        return store

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_cross_project(self, store):
        """Dev list_jobs_by_owner returns jobs from all of an owner's projects."""
        owner_id = "owner-dev-1"
        proj_A = "proj-A"
        proj_B = "proj-B"

        # Put jobs in two different projects for the same owner
        await store.put_job(
            {
                "job_id": "job-A1",
                "project_id": proj_A,
                "owner_id": owner_id,
                "state": "completed",
            }
        )
        await store.put_job(
            {
                "job_id": "job-B1",
                "project_id": proj_B,
                "owner_id": owner_id,
                "state": "running",
            }
        )

        # List all jobs for this owner
        jobs = await store.list_jobs_by_owner(owner_id)
        assert len(jobs) == 2

        job_ids = {j["job_id"] for j in jobs}
        assert job_ids == {"job-A1", "job-B1"}

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_newest_first(self, store):
        """Dev list_jobs_by_owner returns newest first."""
        owner_id = "owner-dev-2"
        project_id = "proj-C"

        # Put three jobs with different created_at timestamps
        await store.put_job(
            {
                "job_id": "job-1",
                "project_id": project_id,
                "owner_id": owner_id,
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
        await store.put_job(
            {
                "job_id": "job-2",
                "project_id": project_id,
                "owner_id": owner_id,
                "created_at": "2026-01-03T00:00:00Z",
            }
        )
        await store.put_job(
            {
                "job_id": "job-3",
                "project_id": project_id,
                "owner_id": owner_id,
                "created_at": "2026-01-02T00:00:00Z",
            }
        )

        jobs = await store.list_jobs_by_owner(owner_id)
        assert len(jobs) == 3

        # Newest first
        assert jobs[0]["job_id"] == "job-2"
        assert jobs[1]["job_id"] == "job-3"
        assert jobs[2]["job_id"] == "job-1"

    @pytest.mark.asyncio
    async def test_list_jobs_by_owner_returns_empty_when_none(self, store):
        """Dev list_jobs_by_owner returns [] when no jobs exist."""
        owner_id = "owner-dev-empty"

        jobs = await store.list_jobs_by_owner(owner_id)
        assert jobs == []
