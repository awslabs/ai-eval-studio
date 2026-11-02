# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker entrypoint — bootstrap, snapshot pull, harness execution."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform.job_compute import JobComputeHandle


def _run(outcome: str):
    """A harness result with no payload — for the failed/cancelled paths."""
    from app.services.harness_runner import HarnessRun

    return HarnessRun(outcome)


def _completed_run(results=None, output_file="eval_results_baseline.json"):
    """A completed harness result carrying its payload back to the caller.

    The worker asks the harness to DEFER completion, so the results and filename
    come back in the return value and the worker records completion itself after
    the upload. A test that returns a bare "completed" string is asserting a
    contract that no longer exists.
    """
    from app.services.harness_runner import HarnessRun

    return HarnessRun("completed", results if results is not None else [], output_file)


@pytest.fixture(autouse=True)
def _temp_workspace_root(tmp_path, monkeypatch):
    """Point the worker's scratch root at a temp dir for every test here.

    The real root, ``/tmp/workspace/{job_id}``, is correct in the Fargate
    container, but a test that writes a fixed absolute path fails with EPERM
    wherever bare ``/tmp`` is not writable, and passes by accident wherever
    the directory already exists.
    """
    from app.services import worker_entrypoint

    monkeypatch.setattr(worker_entrypoint, "WORKSPACE_ROOT", tmp_path / "workspace")


@pytest.fixture(autouse=True)
def _stub_output_upload():
    """Make the results upload succeed by default for tests not about uploading.

    A completed run whose results never reach S3 is now a worker FAILURE — the
    worker's EFS mount is read-only, so an unuploaded result is a lost result.
    Without this stub, every test that mocks the harness to "completed" without
    creating an output directory would exit 1 for a reason unrelated to what it
    is asserting. The two tests that do assert upload behaviour patch the same
    target themselves, and a decorator patch nests inside this one, so theirs
    still wins.
    """
    with patch(
        "app.services.run_output_snapshot.upload_run_output_to_s3",
        new=AsyncMock(return_value=["eval_results_baseline.json"]),
    ):
        yield


@pytest.fixture
def mock_platform():
    """Mock platform with metadata_store only (worker builds its own S3 client).

    Worker does NOT read platform.s3_client — it constructs its own from boto3
    using minted env creds. This fixture uses a spec-constrained object so
    accessing a non-existent attribute like platform.s3_client raises AttributeError.

    The get_job return shape matches create_job's job_dict: top-level keys are
    owner_id/project_id/job_id/type/state/progress/created_at, with harness params
    nested under a 'config' key. Encoding them at the top level (as this fixture
    did before the fix) is exactly what let it ship green — the suite exercised
    a shape that the real DynamoDB get_job never returns.
    """
    from types import SimpleNamespace

    # Metadata store
    metadata_store = AsyncMock()
    metadata_store.get_job = AsyncMock(return_value={
        "job_id": "eval-001",
        "owner_id": "user-123",
        "project_id": "proj-456",
        "config": {
            "model_ids": ["anthropic.claude-3-5-sonnet-20241022-v2:0"],
            "prompt_version": "baseline",
            "max_test_cases": None,
            "judge_model_id": None,
            "data_file_override": None,
            "source": "manual",
        }
    })

    # Use a simple object with only metadata_store — no s3_client
    platform = SimpleNamespace(metadata_store=metadata_store)
    return platform


class TestWorkerSkipsUvSync:
    """Verify worker runs harness with skip_sync=True."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_skips_uv_sync(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker calls run_eval_via_harness with skip_sync=True."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        # Mock platform init
        mock_init_platform.return_value = mock_platform

        # Mock job tracker
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker

        # Mock download snapshot
        mock_download_snapshot.return_value = None

        # Mock run_eval
        mock_run_eval.return_value = _completed_run()

        # Run worker
        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        # Verify run_eval was called with skip_sync=True
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args.kwargs
        assert call_kwargs["skip_sync"] is True

        assert exit_code == 0


class TestWorkerReadsJobSpecFromStore:
    """Verify worker reads job spec from DDB, not from launch env."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_reads_job_spec_from_store(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker reads full job spec from metadata_store.get_job."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.return_value = _completed_run()

        # Run worker
        await run_worker("eval-001", "user-123", "proj-456")

        # Verify get_job was called
        mock_platform.metadata_store.get_job.assert_called_once_with(
            "user-123", "proj-456", "eval-001"
        )

    @pytest.mark.asyncio
    @patch("app.platform.get_platform")
    async def test_worker_returns_1_if_job_missing(self, mock_init_platform, mock_platform):
        """Worker returns exit code 1 if job record is missing."""
        from app.services.worker_entrypoint import run_worker

        mock_platform.metadata_store.get_job = AsyncMock(return_value=None)
        mock_init_platform.return_value = mock_platform

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1


class TestWorkerUnpacksSnapshotSafely:
    """Verify worker unpacks snapshot safely, rejecting traversal."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_unpacks_snapshot_safely(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker unpacks snapshot via traversal-safe download_and_unpack_snapshot."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.return_value = _completed_run()

        await run_worker("eval-001", "user-123", "proj-456")

        # Verify download_and_unpack_snapshot was called
        mock_download_snapshot.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.platform.get_platform")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    async def test_worker_fails_cleanly_on_malicious_snapshot(
        self, mock_download_snapshot, mock_init_platform, mock_platform
    ):
        """Worker fails cleanly if snapshot contains path traversal."""
        from app.services.working_tree_snapshot import SnapshotTraversalError
        from app.services.worker_entrypoint import run_worker

        mock_init_platform.return_value = mock_platform
        mock_download_snapshot.side_effect = SnapshotTraversalError(
            "Escape attempt detected"
        )

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1


class TestWorkerAwaitsTerminalEventBeforeExit:
    """Verify worker awaits terminal event flush before exit."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_awaits_terminal_event_before_exit(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker awaits drain_pending_writes before returning (no exit-before-flush)."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Make drain_pending_writes an AsyncMock so we can assert it was awaited
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.return_value = _completed_run()

        await run_worker("eval-001", "user-123", "proj-456")

        # Verify drain_pending_writes was awaited
        job_tracker.drain_pending_writes.assert_awaited_once()


class TestWorkerDoesNotReadPlatformS3Client:
    """Verify worker does not read platform.s3_client (it builds its own)."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_does_not_read_platform_s3_client(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker builds its own S3 client from boto3; does not read platform.s3_client."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.return_value = _completed_run()

        # Verify platform.s3_client does not exist
        with pytest.raises(AttributeError):
            _ = mock_platform.s3_client

        # Run worker — should succeed despite missing platform.s3_client
        exit_code = await run_worker("eval-001", "user-123", "proj-456")
        assert exit_code == 0


class TestWorkerRequiresSnapshotsBucketEnv:
    """Verify worker fails loudly if S3_SNAPSHOTS_BUCKET env is unset."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.platform.get_platform")
    async def test_worker_requires_snapshots_bucket_env(
        self, mock_init_platform, mock_boto_session, mock_platform, monkeypatch
    ):
        """Worker raises KeyError if S3_SNAPSHOTS_BUCKET is unset (no hardcoded fallback)."""
        from app.services.worker_entrypoint import run_worker

        # Unset S3_SNAPSHOTS_BUCKET
        monkeypatch.delenv("S3_SNAPSHOTS_BUCKET", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform

        # Worker should return 1 (failure) when S3_SNAPSHOTS_BUCKET is missing
        exit_code = await run_worker("eval-001", "user-123", "proj-456")
        assert exit_code == 1


class TestWorkerPassesProjectIdToCreateJob:
    """Verify worker passes project_id to tracker.create_job (signature enforcement)."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_passes_project_id_to_create_job(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker calls tracker.create_job with project_id first positional.

        This test uses autospec=True on the JobTracker mock so create_job
        enforces the real signature. Calling create_job without project_id
        as the first positional will raise TypeError.
        """
        from unittest.mock import create_autospec

        from app.services.jobs import JobTracker
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        # Mock platform init
        mock_init_platform.return_value = mock_platform

        # Mock job tracker with autospec to enforce real create_job signature
        job_tracker = create_autospec(JobTracker, instance=True)
        mock_get_job_tracker.return_value = job_tracker

        # Mock download snapshot
        mock_download_snapshot.return_value = None

        # Mock run_eval
        mock_run_eval.return_value = _completed_run()

        # Run worker
        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        # Verify create_job was called with project_id as first positional
        job_tracker.create_job.assert_called_once()
        call_args = job_tracker.create_job.call_args

        # Check that project_id was passed (either as first positional or kwarg)
        # The real signature is: create_job(project_id, job_type=None, *, config=None, job_id=None)
        if call_args.args:
            # If called with positional args, first should be project_id
            assert call_args.args[0] == "proj-456", f"First positional arg should be project_id='proj-456', got {call_args.args[0]}"
        else:
            # If all kwargs, project_id should be present
            assert "project_id" in call_args.kwargs, "project_id must be passed to create_job"
            assert call_args.kwargs["project_id"] == "proj-456", f"project_id should be 'proj-456', got {call_args.kwargs['project_id']}"

        assert exit_code == 0


class TestWorkerReadsHarnessParamsFromConfig:
    """Contract tests: run_worker sources harness params from job['config']."""

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_reads_harness_params_from_config(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Worker sources model_ids/prompt_version/etc. from job['config'], not top level."""
        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        # Mock platform init
        mock_init_platform.return_value = mock_platform

        # Mock job tracker
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker

        # Mock download snapshot
        mock_download_snapshot.return_value = None

        # Mock run_eval
        mock_run_eval.return_value = _completed_run()

        # Run worker
        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        # Verify run_eval_via_harness received values from the nested config
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args.kwargs

        # The fixture has these values in config — verify they made it through
        assert call_kwargs["model_ids"] == ["anthropic.claude-3-5-sonnet-20241022-v2:0"]
        assert call_kwargs["prompt_version"] == "baseline"
        assert "metrics" not in call_kwargs
        assert call_kwargs["judge_model_id"] is None
        assert call_kwargs["data_file_override"] is None
        assert call_kwargs["source"] == "manual"

        assert exit_code == 0

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_casts_decimal_max_test_cases(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_boto_session,
        monkeypatch,
    ):
        """Worker casts config['max_test_cases'] from Decimal to int."""
        from decimal import Decimal
        from types import SimpleNamespace

        from app.services.worker_entrypoint import run_worker

        # Set required env vars
        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        # Mock boto3 session
        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        # Build a custom mock_platform with Decimal in config
        metadata_store = AsyncMock()
        metadata_store.get_job = AsyncMock(return_value={
            "job_id": "eval-001",
            "owner_id": "user-123",
            "project_id": "proj-456",
            "config": {
                "model_ids": ["anthropic.claude-3-5-sonnet-20241022-v2:0"],
                "prompt_version": "baseline",
                "max_test_cases": Decimal("5"),  # DDB _from_dynamo returns Decimal
                "judge_model_id": None,
                "data_file_override": None,
                "source": "manual",
            }
        })
        custom_platform = SimpleNamespace(metadata_store=metadata_store)
        mock_init_platform.return_value = custom_platform

        # Mock job tracker
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        # Awaitable: the worker awaits the drain, and a failed drain now means
        # the terminal state never landed, so the worker reports failure. A bare
        # MagicMock raises TypeError on await, which used to be swallowed.
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker

        # Mock download snapshot
        mock_download_snapshot.return_value = None

        # Mock run_eval
        mock_run_eval.return_value = _completed_run()

        # Run worker
        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        # Verify run_eval_via_harness received an int, not Decimal
        mock_run_eval.assert_called_once()
        call_kwargs = mock_run_eval.call_args.kwargs
        assert call_kwargs["max_test_cases"] == 5
        assert isinstance(call_kwargs["max_test_cases"], int), "cast must yield int, not Decimal"

        assert exit_code == 0


@pytest.mark.asyncio
@patch("boto3.session.Session")
@patch("app.services.jobs.get_job_tracker")
@patch("app.services.harness_runner.run_eval_via_harness")
@patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
@patch("app.platform.get_platform")
async def test_worker_threads_user_id_to_create_job_config(
    mock_init_platform,
    mock_download_snapshot,
    mock_run_eval,
    mock_get_job_tracker,
    mock_boto_session,
    monkeypatch,
):
    """The worker's create_job config includes user_id from cfg.

    update_state's durable write-through guards the terminal USER#-record write
    with `if user_id:`. Without threading user_id here, the worker's terminal
    state write is silently skipped, the durable job stays `queued`, and
    rebuild_jobs_from_store reconstructs it as an active job that blocks the
    dispatch fence on the next eval.
    """
    from types import SimpleNamespace
    from app.services.worker_entrypoint import run_worker

    # Set required env vars
    monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    # Mock boto3 session
    mock_session = MagicMock()
    mock_s3_client = MagicMock()
    mock_session.client.return_value = mock_s3_client
    mock_boto_session.return_value = mock_session

    # Mock platform with job spec that includes user_id in config
    metadata_store = AsyncMock()
    metadata_store.get_job = AsyncMock(return_value={
        "job_id": "eval-001",
        "owner_id": "user-123",
        "project_id": "proj-456",
        "config": {
            "user_id": "user-123",  # This is what dispatch stores
            "model_ids": ["anthropic.claude-3-5-sonnet-20241022-v2:0"],
            "prompt_version": "baseline",
            "max_test_cases": None,
            "judge_model_id": None,
            "data_file_override": None,
            "source": "manual",
        }
    })
    platform = SimpleNamespace(metadata_store=metadata_store)
    mock_init_platform.return_value = platform

    # Mock job tracker with autospec so we enforce the real create_job signature
    from app.services.jobs import JobTracker
    job_tracker = MagicMock(spec=JobTracker)
    mock_get_job_tracker.return_value = job_tracker

    # Mock download snapshot
    mock_download_snapshot.return_value = None

    # Mock run_eval
    mock_run_eval.return_value = _completed_run()

    # Run worker
    exit_code = await run_worker("eval-001", "user-123", "proj-456")

    # Assert: create_job's config carries user_id alongside arch_split,
    # owner_id and project_id.
    job_tracker.create_job.assert_called_once()
    call_kwargs = job_tracker.create_job.call_args.kwargs
    assert "config" in call_kwargs, "create_job must receive config kwarg"
    config = call_kwargs["config"]
    assert config.get("user_id") == "user-123", (
        "config must include user_id from cfg so update_state's durable "
        "write-through mirrors terminal state to the USER# record"
    )

    assert exit_code == 0


class TestWorkerUploadsOutputOnCompletedPathOnly:
    """Reverse-sync: worker uploads output/ to S3 on exit-0 only.

    The completed detached eval's results must travel worker→S3→coordinator→EFS
    (worker EFS mount is read-only). The upload fires only on the harness
    completed path (exit 0). Cancelled runs already delete the /tmp output;
    failed runs must not sync partial garbage — so a harness exception (exit 1)
    must NOT trigger the upload.
    """

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_uploads_output_on_success(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_upload_output,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Harness exit 0 → upload_run_output_to_s3 is called with the output dir."""
        from app.services.worker_entrypoint import run_worker

        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.return_value = ["eval_results_baseline.json"]

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 0
        mock_upload_output.assert_awaited_once()
        call_args = mock_upload_output.call_args.args
        # (output_dir, s3_client, bucket, job_id)
        from app.services import worker_entrypoint
        assert call_args[0] == worker_entrypoint.WORKSPACE_ROOT / "eval-001" / "output"
        assert call_args[1] is mock_s3_client
        assert call_args[2] == "test-snapshots-bucket"
        assert call_args[3] == "eval-001"

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_worker_does_not_upload_on_harness_failure(
        self,
        mock_init_platform,
        mock_download_snapshot,
        mock_run_eval,
        mock_get_job_tracker,
        mock_upload_output,
        mock_boto_session,
        mock_platform,
        monkeypatch,
    ):
        """Harness raises (exit 1) → upload_run_output_to_s3 is NOT called."""
        from app.services.worker_entrypoint import run_worker

        monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        mock_session = MagicMock()
        mock_s3_client = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session

        mock_init_platform.return_value = mock_platform
        job_tracker = MagicMock()
        job_tracker.create_job = MagicMock()
        job_tracker.drain_pending_writes = AsyncMock()
        mock_get_job_tracker.return_value = job_tracker
        mock_download_snapshot.return_value = None
        mock_run_eval.side_effect = RuntimeError("harness boom")

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1
        mock_upload_output.assert_not_called()


def _worker_mocks(mock_boto_session, mock_init_platform, mock_get_job_tracker,
                  mock_download_snapshot, mock_platform, monkeypatch):
    """Wire the standard worker mocks and return the job-tracker double."""
    monkeypatch.setenv("S3_SNAPSHOTS_BUCKET", "test-snapshots-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    mock_session = MagicMock()
    mock_session.client.return_value = MagicMock()
    mock_boto_session.return_value = mock_session
    mock_init_platform.return_value = mock_platform

    job_tracker = MagicMock()
    job_tracker.create_job = MagicMock()
    job_tracker.fail_job = MagicMock()
    job_tracker.drain_pending_writes = AsyncMock()
    mock_get_job_tracker.return_value = job_tracker
    mock_download_snapshot.return_value = None
    return job_tracker


class TestWorkerExitCodeFollowsHarnessOutcome:
    """The exit code tracks the harness's OWN verdict, not "did it raise".

    ``run_eval_via_harness`` records an ordinary eval failure on the job record
    and returns normally. Branching on exceptions therefore reported every such
    failure as a success: ECS logged exitCode 0 for a failed eval, and the
    success-only upload branch ran anyway.
    """

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_failed_outcome_without_raising_exits_nonzero(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """A "failed" return with no exception must exit 1 and skip the upload."""
        from app.services.worker_entrypoint import run_worker

        _worker_mocks(mock_boto_session, mock_init_platform, mock_get_job_tracker,
                      mock_download_snapshot, mock_platform, monkeypatch)
        mock_run_eval.return_value = _run("failed")

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1
        mock_upload_output.assert_not_called()

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_cancelled_outcome_exits_zero_without_uploading(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """A cancel was honoured, so the task did its job — but nothing to upload."""
        from app.services.worker_entrypoint import run_worker

        _worker_mocks(mock_boto_session, mock_init_platform, mock_get_job_tracker,
                      mock_download_snapshot, mock_platform, monkeypatch)
        mock_run_eval.return_value = _run("cancelled")

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 0
        mock_upload_output.assert_not_called()


class TestCompletedRunRequiresDurableOutput:
    """A completed job must never be published with unreachable results.

    The worker mounts EFS read-only, so ``/tmp`` holds the only copy of the
    results until they reach S3 — and ``/tmp`` dies with the task. The upload
    therefore has to succeed BEFORE the terminal state is committed, and a
    failed upload has to become a failure rather than a warning.
    """

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_upload_failure_fails_the_job_instead_of_exiting_zero(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """Every upload attempt raising → fail_job recorded and exit 1."""
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.side_effect = RuntimeError("s3 down")
        # Keep the test fast — the retry backoff is not what is under test.
        monkeypatch.setattr(
            "app.services.worker_entrypoint._OUTPUT_UPLOAD_BACKOFF_S", 0
        )

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1, "a completed run whose results were lost is a failure"
        job_tracker.fail_job.assert_called_once()

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_upload_retries_then_succeeds(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """A transient upload error is retried while the bytes still exist."""
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.side_effect = [
            RuntimeError("throttled"),
            ["eval_results_baseline.json"],
        ]
        monkeypatch.setattr(
            "app.services.worker_entrypoint._OUTPUT_UPLOAD_BACKOFF_S", 0
        )

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 0
        assert mock_upload_output.await_count == 2
        job_tracker.fail_job.assert_not_called()

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_no_results_to_upload_fails_without_retrying(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """An empty upload list is an inconsistency, not a transient fault."""
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.return_value = []

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1
        job_tracker.fail_job.assert_called_once()
        assert mock_upload_output.await_count == 1, "retrying cannot conjure the file"

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_output_is_uploaded_before_terminal_state_is_drained(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """Ordering is the defect: the drain commits terminal state, so it goes last."""
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()

        order: list[str] = []

        async def _upload(*_a, **_k):
            order.append("upload")
            return ["eval_results_baseline.json"]

        async def _drain(*_a, **_k):
            order.append("drain")

        mock_upload_output.side_effect = _upload
        job_tracker.complete_job = MagicMock(
            side_effect=lambda *_a, **_k: order.append("complete_job")
        )
        job_tracker.drain_pending_writes = AsyncMock(side_effect=_drain)

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 0
        # complete_job is the part that matters, not just the drain: it schedules
        # the durable "completed" write, publishes the terminal envelope and the
        # project event, and releases the project's eval lock — all immediately,
        # and none of it retractable, because the fan-out is gated on the job not
        # already being terminal. It has to come after the upload.
        assert order == ["upload", "complete_job", "drain"], (
            "results must be durable in S3 before completion is recorded"
        )

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_harness_is_asked_to_defer_completion(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """The worker, not the harness, owns the terminal transition.

        If the harness terminalizes, it does so before this function can upload,
        and there is no way to take that back — so the request to defer is itself
        part of the contract.
        """
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run(
            results=[{"model": "m1"}], output_file="eval_results_v2.json"
        )
        mock_upload_output.return_value = ["eval_results_v2.json"]

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 0
        assert mock_run_eval.call_args.kwargs["defer_completion"] is True
        # And the worker records completion with the payload the harness returned,
        # so nothing is lost by deferring.
        job_tracker.complete_job.assert_called_once()
        args = job_tracker.complete_job.call_args.args
        assert args[0] == "eval-001"
        assert args[1] == [{"model": "m1"}]
        assert args[2] == "eval_results_v2.json"

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_a_lost_upload_never_records_completion(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """The job must never read completed when its results are gone.

        This is the whole point of deferring. Previously the harness announced
        completion first, and the later fail_job could not retract it: the
        terminal fan-out is skipped once the job is already terminal, so the SPA
        kept "completed" while the durable record went to "failed".
        """
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.side_effect = RuntimeError("s3 down")
        monkeypatch.setattr(
            "app.services.worker_entrypoint._OUTPUT_UPLOAD_BACKOFF_S", 0
        )

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1
        job_tracker.complete_job.assert_not_called()
        job_tracker.fail_job.assert_called_once()

    @pytest.mark.asyncio
    @patch("boto3.session.Session")
    @patch("app.services.run_output_snapshot.upload_run_output_to_s3")
    @patch("app.services.jobs.get_job_tracker")
    @patch("app.services.harness_runner.run_eval_via_harness")
    @patch("app.services.working_tree_snapshot.download_and_unpack_snapshot")
    @patch("app.platform.get_platform")
    async def test_failed_drain_is_not_reported_as_success(
        self, mock_init_platform, mock_download_snapshot, mock_run_eval,
        mock_get_job_tracker, mock_upload_output, mock_boto_session,
        mock_platform, monkeypatch,
    ):
        """If the terminal state never reached the store, do not exit 0."""
        from app.services.worker_entrypoint import run_worker

        job_tracker = _worker_mocks(
            mock_boto_session, mock_init_platform, mock_get_job_tracker,
            mock_download_snapshot, mock_platform, monkeypatch,
        )
        mock_run_eval.return_value = _completed_run()
        mock_upload_output.return_value = ["eval_results_baseline.json"]
        job_tracker.drain_pending_writes = AsyncMock(
            side_effect=RuntimeError("ddb unavailable")
        )

        exit_code = await run_worker("eval-001", "user-123", "proj-456")

        assert exit_code == 1
