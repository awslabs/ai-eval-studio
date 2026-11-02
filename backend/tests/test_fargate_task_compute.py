# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FargateTaskCompute adapter — ECS RunTask, StopTask, DescribeTasks."""
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from moto import mock_aws

from app.platform.aws.fargate_task_compute import FargateTaskCompute
from app.platform.job_compute import JobCompute, JobComputeHandle

#: Stand-in credentials for the mocked token vendor. Deliberately not AWS's
#: documentation example pair: that pair is shaped like a live key, so secret
#: scanners report it, and a reviewer then has to confirm by hand that a test
#: fixture is not a leak. These match no credential pattern.
#:
#: ``FAKE_SECRET_ACCESS_KEY`` is a module constant rather than a literal because
#: one test asserts the secret never reaches a log record — repeating the string
#: there let the two drift.
FAKE_ACCESS_KEY_ID = "AKIAFAKEFARGATE"
FAKE_SECRET_ACCESS_KEY = "fake-secret-access-key-not-a-credential"
FAKE_SESSION_TOKEN = "fake-session-token"


@pytest.fixture
def mock_ecs_client():
    """Mock ECS client for testing."""
    client = MagicMock()
    client.run_task = MagicMock(return_value={
        "tasks": [{
            "taskArn": "arn:aws:ecs:us-east-1:123456789012:task/cluster-name/abc123def456",
            "lastStatus": "RUNNING"
        }],
        "failures": []
    })
    client.stop_task = MagicMock(return_value={})
    client.describe_tasks = MagicMock(return_value={
        "tasks": [{
            "taskArn": "arn:aws:ecs:us-east-1:123456789012:task/cluster-name/abc123def456",
            "lastStatus": "RUNNING"
        }]
    })
    return client


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    client = MagicMock()
    client.put_object = MagicMock(return_value={})
    return client


@pytest.fixture
def mock_token_vending():
    """Mock token vending for testing."""
    from app.platform.token_vending import ScopedCredentials

    vending = AsyncMock()
    vending.mint = AsyncMock(return_value=ScopedCredentials(
        access_key_id=FAKE_ACCESS_KEY_ID,
        secret_access_key=FAKE_SECRET_ACCESS_KEY,
        session_token=FAKE_SESSION_TOKEN,
        expiration="2026-07-03T10:00:00Z"
    ))
    return vending


@pytest.fixture
def fargate_compute(mock_ecs_client, mock_s3_client, mock_token_vending):
    """FargateTaskCompute instance with mocked dependencies."""
    return FargateTaskCompute(
        ecs_client=mock_ecs_client,
        cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster",
        task_definition_arn="arn:aws:ecs:us-east-1:123456789012:task-definition/worker:1",
        subnet_ids=["subnet-abc123", "subnet-def456"],
        security_group_id="sg-xyz789",
        token_vending=mock_token_vending,
        s3_client=mock_s3_client,
        snapshot_bucket="eval-snapshots",
        projects_dir="/mnt/projects"
    )


class TestFargateTaskComputeSatisfiesProtocol:
    """Verify FargateTaskCompute satisfies the JobCompute Protocol."""

    def test_satisfies_protocol(self, fargate_compute):
        """FargateTaskCompute satisfies JobCompute Protocol."""
        assert isinstance(fargate_compute, JobCompute)


class TestLaunch:
    """Tests for FargateTaskCompute.launch()."""

    @pytest.mark.asyncio
    async def test_launch_returns_full_task_arn(self, fargate_compute, mock_ecs_client):
        """launch() returns a handle wrapping the full task ARN."""
        handle = await fargate_compute.launch(
            job_id="eval-001",
            owner_id="user-123",
            project_id="proj-456"
        )

        assert isinstance(handle, JobComputeHandle)
        assert handle.kind == "fargate_task"
        assert handle.value == "arn:aws:ecs:us-east-1:123456789012:task/cluster-name/abc123def456"
        assert str(handle) == handle.value

    @pytest.mark.asyncio
    async def test_launch_passes_task_definition_family_through_unchanged(
        self, mock_ecs_client, mock_s3_client, mock_token_vending
    ):
        """A bare task-definition FAMILY reaches RunTask verbatim.

        The deploy supplies the family rather than a revisioned ARN, because that
        value is exported cross-stack and a per-deploy revision cannot be updated
        once another stack imports it. ECS resolves a bare family to its latest
        ACTIVE revision, so nothing here may normalize, suffix, or otherwise
        rewrite it on the way to the API call.
        """
        compute = FargateTaskCompute(
            ecs_client=mock_ecs_client,
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster",
            task_definition_arn="ai-eval-studio-worker",
            subnet_ids=["subnet-abc123"],
            security_group_id="sg-xyz789",
            token_vending=mock_token_vending,
            s3_client=mock_s3_client,
            snapshot_bucket="eval-snapshots",
            projects_dir="/mnt/projects",
        )

        await compute.launch(
            job_id="eval-family", owner_id="user-123", project_id="proj-456"
        )

        kwargs = mock_ecs_client.run_task.call_args.kwargs
        assert kwargs["taskDefinition"] == "ai-eval-studio-worker", (
            "the family must reach RunTask verbatim so ECS can resolve the latest "
            f"ACTIVE revision; got {kwargs['taskDefinition']!r}"
        )

    @pytest.mark.asyncio
    async def test_launch_does_not_inject_static_credentials(
        self, fargate_compute, mock_ecs_client, mock_token_vending
    ):
        """launch() NO LONGER injects static minted creds; worker self-mints on boot.

        The worker calls write_worker_credentials_config on boot to set up a
        credential_process chain that self-refreshes for the life of any eval
        run. No static credential triple is injected via RunTask env, so the
        environment keys are job/owner/project only.
        """
        await fargate_compute.launch(
            job_id="eval-001",
            owner_id="user-123",
            project_id="proj-456"
        )

        # Verify mint was NOT called (worker mints on its own boot)
        mock_token_vending.mint.assert_not_called()

        # Verify RunTask was called with env vars containing NO static creds
        mock_ecs_client.run_task.assert_called_once()
        call_kwargs = mock_ecs_client.run_task.call_args.kwargs

        env_vars = call_kwargs["overrides"]["containerOverrides"][0]["environment"]
        env_dict = {e["name"]: e["value"] for e in env_vars}

        # Static credential keys are NOT present
        assert "AWS_ACCESS_KEY_ID" not in env_dict
        assert "AWS_SECRET_ACCESS_KEY" not in env_dict
        assert "AWS_SESSION_TOKEN" not in env_dict

        # Job identifiers ARE present
        assert env_dict["JOB_ID"] == "eval-001"
        assert env_dict["OWNER_ID"] == "user-123"
        assert env_dict["PROJECT_ID"] == "proj-456"

    @pytest.mark.asyncio
    async def test_launch_sets_enable_execute_command_false(
        self, fargate_compute, mock_ecs_client
    ):
        """launch() hardcodes enableExecuteCommand=False (security hardening)."""
        await fargate_compute.launch(
            job_id="eval-001",
            owner_id="user-123",
            project_id="proj-456"
        )

        call_kwargs = mock_ecs_client.run_task.call_args.kwargs
        assert call_kwargs["enableExecuteCommand"] is False

    @pytest.mark.asyncio
    async def test_no_creds_logged(self, fargate_compute, caplog):
        """launch() never logs minted credentials."""
        with caplog.at_level(logging.DEBUG):
            await fargate_compute.launch(
                job_id="eval-001",
                owner_id="user-123",
                project_id="proj-456"
            )

        # Secret access key should never appear in logs
        for record in caplog.records:
            assert FAKE_SECRET_ACCESS_KEY not in record.message


class TestCancel:
    """Tests for FargateTaskCompute.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_calls_stop_task_with_full_arn(
        self, fargate_compute, mock_ecs_client
    ):
        """cancel() calls ecs.stop_task with the full task ARN."""
        handle = JobComputeHandle(
            kind="fargate_task",
            value="arn:aws:ecs:us-east-1:123456789012:task/cluster-name/abc123def456"
        )

        await fargate_compute.cancel(handle)

        mock_ecs_client.stop_task.assert_called_once()
        call_kwargs = mock_ecs_client.stop_task.call_args.kwargs
        assert call_kwargs["cluster"] == "arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster"
        assert call_kwargs["task"] == handle.value
        assert call_kwargs["reason"] == "User canceled evaluation"


class TestStatus:
    """Tests for FargateTaskCompute.status()."""

    @pytest.mark.asyncio
    async def test_status_maps_last_status(self, fargate_compute, mock_ecs_client):
        """status() maps ECS lastStatus to state string."""
        handle = JobComputeHandle(
            kind="fargate_task",
            value="arn:aws:ecs:us-east-1:123456789012:task/cluster-name/abc123def456"
        )

        state = await fargate_compute.status(handle)

        assert state == "RUNNING"
        mock_ecs_client.describe_tasks.assert_called_once()
        call_kwargs = mock_ecs_client.describe_tasks.call_args.kwargs
        assert call_kwargs["cluster"] == "arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster"
        assert handle.value in call_kwargs["tasks"]

    @pytest.mark.asyncio
    async def test_status_reports_a_missing_task_as_unknown_not_stopped(
        self, fargate_compute, mock_ecs_client
    ):
        """An empty response is the absence of evidence, not evidence of a stop.

        This used to return STOPPED, and a cancel consumed that as confirmation.
        DescribeTasks comes back empty for two very different situations —
        eventual consistency right after StopTask, and a task ECS reaped long
        after it finished — and in the first one the worker is still running and
        still billing. Reporting it as stopped released the project's eval lock
        underneath a live worker.
        """
        mock_ecs_client.describe_tasks.return_value = {
            "tasks": [],
            "failures": [{"reason": "MISSING"}],
        }

        handle = JobComputeHandle(
            kind="fargate_task",
            value="arn:aws:ecs:us-east-1:123456789012:task/cluster-name/missing"
        )

        state = await fargate_compute.status(handle)
        assert state == "UNKNOWN"


class TestListRunningWorkers:
    """Tests for FargateTaskCompute.list_running_workers() (sprawl detection)."""

    @pytest.mark.asyncio
    async def test_list_running_workers_returns_tagged_tasks(
        self, fargate_compute, mock_ecs_client
    ):
        """list_running_workers() returns task dicts with taskArn, lastStatus, and tags."""
        # Mock paginator
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"taskArns": [
                "arn:aws:ecs:us-east-1:123456789012:task/cluster/task1",
                "arn:aws:ecs:us-east-1:123456789012:task/cluster/task2"
            ]}
        ]
        mock_ecs_client.get_paginator.return_value = paginator

        # Mock describe_tasks with tags
        mock_ecs_client.describe_tasks.return_value = {
            "tasks": [
                {
                    "taskArn": "arn:aws:ecs:us-east-1:123456789012:task/cluster/task1",
                    "lastStatus": "RUNNING",
                    "tags": [
                        {"key": "Owner", "value": "user-123"},
                        {"key": "Project", "value": "proj-456"},
                        {"key": "Job", "value": "job-001"}
                    ]
                },
                {
                    "taskArn": "arn:aws:ecs:us-east-1:123456789012:task/cluster/task2",
                    "lastStatus": "PENDING",
                    "tags": [
                        {"key": "Owner", "value": "user-789"},
                        {"key": "Project", "value": "proj-abc"},
                        {"key": "Job", "value": "job-002"}
                    ]
                }
            ]
        }

        tasks = await fargate_compute.list_running_workers()

        # Assert returns both task dicts
        assert len(tasks) == 2
        assert tasks[0]["taskArn"] == "arn:aws:ecs:us-east-1:123456789012:task/cluster/task1"
        assert tasks[0]["lastStatus"] == "RUNNING"
        assert len(tasks[0]["tags"]) == 3
        assert tasks[1]["taskArn"] == "arn:aws:ecs:us-east-1:123456789012:task/cluster/task2"
        assert tasks[1]["lastStatus"] == "PENDING"

        # Assert list_tasks called with correct family and desiredStatus
        mock_ecs_client.get_paginator.assert_called_once_with("list_tasks")
        paginate_kwargs = paginator.paginate.call_args.kwargs
        assert paginate_kwargs["cluster"] == "arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster"
        assert paginate_kwargs["family"] == "ai-eval-studio-worker"
        assert paginate_kwargs["desiredStatus"] == "RUNNING"

        # Assert describe_tasks called with include=["TAGS"]
        mock_ecs_client.describe_tasks.assert_called_once()
        describe_kwargs = mock_ecs_client.describe_tasks.call_args.kwargs
        assert describe_kwargs["cluster"] == "arn:aws:ecs:us-east-1:123456789012:cluster/eval-cluster"
        assert set(describe_kwargs["tasks"]) == {
            "arn:aws:ecs:us-east-1:123456789012:task/cluster/task1",
            "arn:aws:ecs:us-east-1:123456789012:task/cluster/task2"
        }
        assert describe_kwargs["include"] == ["TAGS"]

    @pytest.mark.asyncio
    async def test_list_running_workers_empty_result_no_describe_call(
        self, fargate_compute, mock_ecs_client
    ):
        """list_running_workers() returns [] when no task ARNs, does NOT call describe_tasks."""
        # Mock paginator returning empty list
        paginator = MagicMock()
        paginator.paginate.return_value = [{"taskArns": []}]
        mock_ecs_client.get_paginator.return_value = paginator

        tasks = await fargate_compute.list_running_workers()

        assert tasks == []
        mock_ecs_client.get_paginator.assert_called_once_with("list_tasks")
        # describe_tasks NOT called when no task ARNs
        mock_ecs_client.describe_tasks.assert_not_called()
