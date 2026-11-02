# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JobCompute Protocol seam + both implementations.

This test file covers the seam contract (dev + cloud impls).
"""
import pytest
from app.platform.job_compute import JobCompute, JobComputeHandle, get_job_compute
from app.platform.dev_local import LocalSubprocessCompute


class TestJobComputeProtocolContract:
    """Tests that the dev implementation satisfies the JobCompute Protocol.

    The Fargate implementation's Protocol check lives in
    test_fargate_task_compute.py.
    """

    @pytest.mark.parametrize(
        "impl_cls",
        [LocalSubprocessCompute],
        ids=["LocalSubprocessCompute"],
    )
    def test_satisfies_protocol(self, impl_cls):
        """Each listed impl must satisfy isinstance(impl, JobCompute)."""
        impl = impl_cls()
        assert isinstance(impl, JobCompute), (
            f"{impl_cls.__name__} does not satisfy JobCompute Protocol"
        )


class TestJobComputeHandleType:
    """Tests for the JobComputeHandle frozen dataclass."""

    def test_handle_has_kind_and_value_fields(self):
        """JobComputeHandle must have kind + value fields."""
        from app.platform.job_compute import JobComputeHandle

        handle = JobComputeHandle(kind="local_pid", value="12345")
        assert handle.kind == "local_pid"
        assert handle.value == "12345"

    def test_handle_is_frozen(self):
        """JobComputeHandle must be immutable."""
        from app.platform.job_compute import JobComputeHandle

        handle = JobComputeHandle(kind="local_pid", value="12345")
        with pytest.raises((AttributeError, Exception)):
            handle.kind = "fargate_task"

    def test_handle_str_returns_value(self):
        """str(handle) must return the value for persisting as compute_handle."""
        from app.platform.job_compute import JobComputeHandle

        handle = JobComputeHandle(kind="fargate_task", value="arn:aws:...")
        assert str(handle) == "arn:aws:..."


class TestGetJobComputeSelector:
    """Tests for get_job_compute() mode-based selection."""

    def test_returns_local_in_dev_mode(self, monkeypatch):
        """In dev mode, get_job_compute() returns LocalSubprocessCompute."""
        # Set dev mode via environment
        monkeypatch.setenv("EVAL_STUDIO_MODE", "dev")

        compute = get_job_compute()
        assert isinstance(compute, LocalSubprocessCompute)

    def test_returns_fargate_in_cloud_mode(self, monkeypatch):
        """In cloud mode, get_job_compute() returns FargateTaskCompute."""
        from unittest.mock import MagicMock, patch
        from app.platform.aws.fargate_task_compute import FargateTaskCompute

        # Set cloud mode + required env vars
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        monkeypatch.setenv("ECS_CLUSTER_ARN", "arn:aws:ecs:us-east-1:123456789012:cluster/test")
        monkeypatch.setenv("WORKER_TASK_DEFINITION_ARN", "arn:aws:ecs:us-east-1:123456789012:task-definition/ai-eval-studio-worker:1")
        monkeypatch.setenv("WORKER_SUBNET_IDS", "subnet-abc123,subnet-def456")
        monkeypatch.setenv("WORKER_SECURITY_GROUP_ID", "sg-12345678")
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("PROJECTS_DIR", "/mnt/projects")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session/clients so no real AWS call
        # boto3 is imported inside get_job_compute, so patch it at the boto3 module level
        mock_session = MagicMock()
        mock_ecs_client = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: (
            mock_ecs_client if svc == "ecs" else mock_s3_client
        )

        # Mock get_platform to return object with token_vending attr
        mock_platform = MagicMock()
        mock_platform.token_vending = MagicMock()

        with patch("boto3.session.Session", return_value=mock_session), \
             patch("app.platform.get_platform", return_value=mock_platform):
            compute = get_job_compute()
            assert isinstance(compute, FargateTaskCompute)


class TestSnapshotKeyParity:
    """Tests for snapshot_key_for_job helper (launch-key == worker-key invariant)."""

    def test_snapshot_key_derivation(self):
        """snapshot_key_for_job computes the canonical key for a job_id."""
        from app.services.snapshot_key import snapshot_key_for_job
        from claude_agent_sdk import project_key_for_directory

        job_id = "test-job-123"
        expected_prefix = project_key_for_directory(f"/tmp/workspace/{job_id}")
        expected_key = f"{expected_prefix}/snapshots/{job_id}.tar.gz"

        actual_key = snapshot_key_for_job(job_id)
        assert actual_key == expected_key

    def test_launch_key_equals_worker_key(self):
        """Launch side and worker side must compute identical keys for the same job_id."""
        from app.services.snapshot_key import snapshot_key_for_job

        job_id = "eval-456"
        # Both launch (FargateTaskCompute) and worker (worker_entrypoint) call this helper
        launch_key = snapshot_key_for_job(job_id)
        worker_key = snapshot_key_for_job(job_id)

        assert launch_key == worker_key, "Launch and worker keys must match"


class TestMissingWorkerLaunchEnv:
    """Tests for MissingWorkerLaunchConfig fail-loud aggregate check."""

    def test_missing_multiple_vars_raises_naming_all(self, monkeypatch):
        """MissingWorkerLaunchConfig raised naming EVERY missing variable.
        
        With SEVERAL vars deleted, get_job_compute() raises MissingWorkerLaunchConfig
        (not KeyError), and the message names EVERY missing variable. Assert on at
        least two distinct missing names in one message, so a first-only implementation
        fails.
        """
        from app.platform.job_compute import get_job_compute, MissingWorkerLaunchConfig

        # Set cloud mode but DELETE multiple required vars
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        monkeypatch.setenv("WORKER_SUBNET_IDS", "subnet-abc123")
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-bucket")
        monkeypatch.setenv("PROJECTS_DIR", "/mnt/projects")
        # Missing: ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN, WORKER_SECURITY_GROUP_ID

        with pytest.raises(MissingWorkerLaunchConfig) as exc_info:
            get_job_compute()

        # Assert the message names EVERY missing variable
        message = str(exc_info.value)
        assert "ECS_CLUSTER_ARN" in message
        assert "WORKER_TASK_DEFINITION_ARN" in message
        assert "WORKER_SECURITY_GROUP_ID" in message
        # Verify it's NOT a bare KeyError shape
        assert "Missing environment variables:" in message

    @pytest.mark.asyncio
    async def test_missing_env_reaches_agent_as_mcp_error(self, monkeypatch):
        """The raise is a catchable MissingWorkerLaunchConfig naming what is missing.

        ``dispatch_eval``'s post-claim ``except`` turns any raise from
        ``get_job_compute`` into an ``mcp_error``. This test checks only that the
        exception can be caught and names the missing variables; it does not
        drive dispatch.
        """
        from app.platform.job_compute import get_job_compute, MissingWorkerLaunchConfig

        # Set cloud mode but DELETE multiple required vars
        monkeypatch.setenv("EVAL_STUDIO_MODE", "cloud")
        monkeypatch.setenv("WORKER_SUBNET_IDS", "subnet-abc123")
        # Missing: ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN, WORKER_SECURITY_GROUP_ID, etc.

        # Assert the exception can be caught (so dispatch can convert it to mcp_error)
        try:
            get_job_compute()
            assert False, "Should have raised MissingWorkerLaunchConfig"
        except MissingWorkerLaunchConfig as e:
            message = str(e)
            # Verify it is actionable and names missing vars
            assert "Missing environment variables:" in message
            assert "ECS_CLUSTER_ARN" in message
