# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval_reconciler triple-duty pass.

The reconciler does three operations in one ECS↔DDB comparison:
1. Stop orphaned workers (task exists but no matching job).
2. Clear stale eval locks and mark stuck jobs failed (job exists but worker not found after grace).
3. Aggressive StopTask-all when kill switch active.

Critical properties (first-class):
- ECS-race safety: jobs younger than grace period are never touched.
- Version-fenced lock clear: a lock released+reclaimed mid-reconcile is not deleted.
- Loop resilience: exceptions inside reconcile_once do not kill the periodic loop.
"""
import asyncio
import inspect
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from .async_wait import until

from app.platform.job_compute import JobComputeHandle
from app.services.eval_reconciler import (
    ReconcileResult,
    reconcile_once,
    start_reconciler_loop,
    stop_reconciler_loop,
)


@pytest.fixture
def mock_compute():
    """Mock JobCompute with controllable list_running_workers + cancel."""
    compute = MagicMock()
    compute.list_running_workers = AsyncMock(return_value=[])
    compute.cancel = AsyncMock()
    return compute


@pytest.fixture
def mock_store():
    """Mock MetadataStore with controllable list_non_terminal_jobs + lock methods.

    The update_job_state mock uses create_autospec to catch unsupported-kwarg drift
    (the reconciler once passed an ``error=`` kwarg the Protocol does not have).
    """
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    store = MagicMock()
    store.list_non_terminal_jobs = AsyncMock(return_value=[])
    store.list_recent_completed_jobs = AsyncMock(return_value=[])
    store.get_eval_lock = AsyncMock(return_value=None)
    store.clear_stale_eval_lock = AsyncMock(return_value=True)
    store.release_eval_lock = AsyncMock()
    store.claim_completion_fired = AsyncMock(return_value=False)

    # Use inspect to validate signature, AsyncMock to track calls
    sig = inspect.signature(DynamoDBMetadataStore.update_job_state)

    def validate_signature(*args, **kwargs):
        # Bind to signature (raises TypeError on unsupported kwargs)
        # Skip self parameter by binding args[1:] as positional
        sig.bind(None, *args, **kwargs)
        return None

    store.update_job_state = AsyncMock(side_effect=validate_signature)

    # The reap re-reads each job strongly before acting on it. With nothing
    # written since the listing, that read returns the listed record.
    async def _get_listed_job(user_id, project_id, job_id, *, consistent=False):
        listed = store.list_non_terminal_jobs.return_value
        return next((dict(j) for j in listed if j["job_id"] == job_id), None)

    store.get_job = AsyncMock(side_effect=_get_listed_job)
    return store


@pytest.fixture
def fargate_compute(mock_compute):
    """Return a compute mock that satisfies isinstance(FargateTaskCompute)."""
    from app.platform.aws.fargate_task_compute import FargateTaskCompute

    # Make the mock pass isinstance checks
    mock_compute.__class__ = FargateTaskCompute
    return mock_compute


@pytest.mark.asyncio
async def test_orphan_worker_cleanup(fargate_compute, mock_store):
    """Orphan worker (task exists but no matching non-terminal job) is stopped."""
    # Arrange: a running task tagged with Job=job-orphan, no matching job in DDB
    fargate_compute.list_running_workers.return_value = [
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/orphan-task",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Owner", "value": "user-1"},
                {"key": "Project", "value": "proj-1"},
                {"key": "Job", "value": "job-orphan"},
            ],
        }
    ]
    mock_store.list_non_terminal_jobs.return_value = []

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act
            result = await reconcile_once()

    # Assert: orphan stopped
    assert result["orphans_stopped"] == 1
    assert result["stale_locks_cleared"] == 0
    assert result["stuck_jobs_failed"] == 0
    fargate_compute.cancel.assert_called_once()
    call_args = fargate_compute.cancel.call_args[0][0]
    assert call_args.kind == "fargate_task"
    assert call_args.value == "arn:aws:ecs:us-east-1:123:task/orphan-task"


@pytest.mark.asyncio
async def test_stuck_job_cleanup(fargate_compute, mock_store, caplog):
    """Non-terminal job with compute_handle, created >grace, no visible worker → marked failed."""
    # Arrange: a job older than grace with a compute_handle, no matching ECS task
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = []
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-stuck",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:stuck"},
            "created_at": old_created_at,
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act
            result = await reconcile_once()

    # Assert: job marked failed
    assert result["stuck_jobs_failed"] == 1
    mock_store.update_job_state.assert_called_once()
    call_args = mock_store.update_job_state.call_args
    assert call_args[0][0] == "user-1"  # user_id
    assert call_args[0][1] == "proj-1"  # project_id
    assert call_args[0][2] == "job-stuck"  # job_id
    assert call_args[1]["state"] == "failed"
    # No error kwarg: the Protocol has none.
    assert "error" not in call_args[1]


@pytest.mark.asyncio
async def test_a_job_that_ended_after_the_listing_is_not_overwritten(
    fargate_compute, mock_store
):
    """The listing is an eventually consistent index; the record decides.

    A cancel confirmed between the listing and the reap has already written
    "cancelled" and released the lock. Reaping from the stale row wrote "failed"
    over it, and a failed manual eval gets a chat turn of its own.
    """
    now = int(time.time())
    listed = {
        "owner_id": "user-1",
        "project_id": "proj-1",
        "job_id": "job-ended",
        "state": "running",
        "compute_handle": "arn:aws:ecs:us-east-1:123:task/ended",
        "created_at": datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat(),
    }
    mock_store.list_non_terminal_jobs.return_value = [listed]
    mock_store.get_job = AsyncMock(return_value={**listed, "state": "cancelled"})

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store
            result = await reconcile_once()

    mock_store.update_job_state.assert_not_called()
    mock_store.release_eval_lock.assert_not_called()
    assert result["stuck_jobs_failed"] == 0
    assert result["pass_failed"] is False


@pytest.mark.asyncio
async def test_stale_lock_reclaim(fargate_compute, mock_store):
    """Stuck job past grace → lock cleared → subsequent claim succeeds (first-class)."""
    # Arrange: a stuck job with a held lock
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = []
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-stale",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:stale"},
            "created_at": old_created_at,
        }
    ]
    # Lock exists for this job
    mock_store.get_eval_lock.return_value = {
        "lock_holder_job_id": "job-stale",
        "lock_version": 42,
        "lock_acquired_at": old_created_at,
    }
    mock_store.clear_stale_eval_lock.return_value = True  # Clear succeeds

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act
            result = await reconcile_once()

    # Assert: lock cleared
    assert result["stale_locks_cleared"] == 1
    mock_store.get_eval_lock.assert_called_once_with("user-1", "proj-1")
    mock_store.clear_stale_eval_lock.assert_called_once_with(
        "user-1",
        "proj-1",
        expected_version=42,
        expected_holder_job_id="job-stale",
    )
    mock_store.release_eval_lock.assert_called_once_with("user-1", "proj-1", "job-stale")


@pytest.mark.asyncio
async def test_reconciler_leaves_a_lock_held_by_a_different_job_alone(
    fargate_compute, mock_store
):
    """A lock reclaimed by a NEWER job must survive the stale job's reconcile.

    The version fence cannot catch this: the newer holder's lock carries its own
    current version, so a version-only conditional delete matched and removed a
    live job's lock — after which a third eval could start on the same project.
    """
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = []
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-stale",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:stale"},
            "created_at": old_created_at,
        }
    ]
    # The lock now belongs to a different, live job.
    mock_store.get_eval_lock.return_value = {
        "lock_holder_job_id": "job-newer-and-alive",
        "lock_version": 43,
        "lock_acquired_at": old_created_at,
    }

    with patch(
        "app.services.eval_reconciler.get_job_compute", return_value=fargate_compute
    ), patch("app.services.eval_reconciler.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        result = await reconcile_once()

    mock_store.clear_stale_eval_lock.assert_not_called()
    assert result["stale_locks_cleared"] == 0
    # The stale job is still failed — only the lock is spared.
    assert result["stuck_jobs_failed"] == 1


@pytest.mark.asyncio
async def test_reconciler_does_not_kill_starting_worker(fargate_compute, mock_store):
    """Job created <grace with no visible worker → NOT touched (ECS-race safety, first-class)."""
    # Arrange: a job created 5s ago (within grace)
    now = int(time.time())
    recent_created_at = datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = []
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-starting",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:starting"},
            "created_at": recent_created_at,
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act (first: within grace)
            result = await reconcile_once()

    # Assert: job NOT touched
    assert result["stuck_jobs_failed"] == 0
    mock_store.update_job_state.assert_not_called()

    # Now simulate grace expired (>60s) by patching the job's created_at
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()
    mock_store.list_non_terminal_jobs.return_value[0]["created_at"] = old_created_at

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act (then: past grace)
            result = await reconcile_once()

    # Assert: NOW marked failed
    assert result["stuck_jobs_failed"] == 1
    mock_store.update_job_state.assert_called_once()


@pytest.mark.asyncio
async def test_version_fenced_clear_skips_reclaimed_lock(fargate_compute, mock_store):
    """Lock version changed between read and clear → clear_stale_eval_lock returns False, lock left intact."""
    # Arrange: job past grace, lock present
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = []
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-stuck",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:stuck"},
            "created_at": old_created_at,
        }
    ]
    mock_store.get_eval_lock.return_value = {
        "lock_holder_job_id": "job-stuck",
        "lock_version": 42,
        "lock_acquired_at": old_created_at,
    }
    # Version fence tripped — clear returns False
    mock_store.clear_stale_eval_lock.return_value = False

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act
            result = await reconcile_once()

    # Assert: lock NOT cleared (stale_locks_cleared == 0)
    assert result["stale_locks_cleared"] == 0
    # Job still marked failed (stuck-job cleanup is independent)
    assert result["stuck_jobs_failed"] == 1


@pytest.mark.asyncio
async def test_kill_switch_stops_all_workers(fargate_compute, mock_store):
    """Kill switch active → ALL workers stopped, not just orphans."""
    # Arrange: two workers, one with a healthy job, one orphan
    now = int(time.time())
    recent_created_at = datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat()

    fargate_compute.list_running_workers.return_value = [
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/healthy-task",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Job", "value": "job-healthy"},
            ],
        },
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/orphan-task",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Job", "value": "job-orphan"},
            ],
        },
    ]
    # job-healthy exists, job-orphan does not
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "owner_id": "user-1",
            "user_id": "user-1",
            "project_id": "proj-1",
            "job_id": "job-healthy",
            "state": "running",
            "compute_handle": {"kind": "fargate_task", "value": "arn:healthy"},
            "created_at": recent_created_at,
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store
            # Activate kill switch
            with patch("app.services.eval_reconciler.is_cost_kill_switch_active", return_value=True):
                # Act
                result = await reconcile_once()

    # Assert: both workers stopped
    assert result["orphans_stopped"] == 2
    assert fargate_compute.cancel.call_count == 2


@pytest.mark.asyncio
async def test_reconcile_once_dev_noop():
    """In dev mode (non-Fargate compute) → zeroed result, no ECS calls."""
    # Arrange: a non-Fargate compute (LocalSubprocessCompute mock)
    from app.platform.dev_local import LocalSubprocessCompute

    dev_compute = MagicMock(spec=LocalSubprocessCompute)
    dev_compute.list_running_workers = AsyncMock()

    with patch("app.services.eval_reconciler.get_job_compute", return_value=dev_compute):
        # Act
        result = await reconcile_once()

    # Assert: zeroed result, list_running_workers NOT called
    assert result["orphans_stopped"] == 0
    assert result["stale_locks_cleared"] == 0
    assert result["stuck_jobs_failed"] == 0
    dev_compute.list_running_workers.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_loop_idempotent():
    """start_reconciler_loop twice creates only one task."""
    # Start the loop
    start_reconciler_loop(interval_seconds=999)

    # Start again — should be no-op
    start_reconciler_loop(interval_seconds=999)

    # Stop
    stop_reconciler_loop()

    # Allow cancellation to complete
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_reconciler_loop_survives_exception():
    """Exception in reconcile_once does NOT kill the loop."""
    call_count = 0

    async def mock_reconcile_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Test exception")
        return ReconcileResult(orphans_stopped=0, stale_locks_cleared=0, stuck_jobs_failed=0)

    with patch("app.services.eval_reconciler.reconcile_once", side_effect=mock_reconcile_once):
        # Start loop with short interval
        start_reconciler_loop(interval_seconds=0.01)

        # Wait for the second tick rather than for a duration that happens to
        # cover two. The budget was 0.05s for two 0.01s ticks, so anything that
        # delayed the loop's first tick — a loaded machine, a slow task start —
        # consumed it and failed a working loop.
        await until(
            lambda: call_count >= 2,
            message="the loop did not tick twice, so it did not survive the "
                    "first tick's exception",
        )

        # Stop
        stop_reconciler_loop()

    # The loop ran again after the first tick raised.
    assert call_count >= 2


# ========================================================================
# Orphaned completion turn recovery tests
# ========================================================================


@pytest.mark.asyncio
async def test_list_recent_completed_jobs_protocol_method_exists():
    """list_recent_completed_jobs method exists on MetadataStore Protocol."""
    from app.platform.metadata_store import MetadataStore
    import inspect

    # Check that the Protocol defines the method
    assert hasattr(MetadataStore, 'list_recent_completed_jobs'), \
        "MetadataStore Protocol must define list_recent_completed_jobs method"

    # Check signature has within_seconds parameter
    sig = inspect.signature(MetadataStore.list_recent_completed_jobs)
    params = list(sig.parameters.keys())
    assert 'within_seconds' in params, \
        "list_recent_completed_jobs must accept within_seconds parameter"


@pytest.mark.asyncio
async def test_reconciler_fires_orphaned_completion_turn(tmp_path):
    """Reconciler fires completion for an orphaned eval keyed on event-log + FILE.

    Readiness no longer depends on the DDB `result` (empty here). The
    reconciler reads run_filename from the event log's terminal `results` event,
    materializes the FILE, then claims + fires from the file-sourced result.
    """
    from app.services.jobs import Job
    from app.platform.aws.fargate_task_compute import FargateTaskCompute

    # Arrange: a recent completed job with completion_fired=false
    now = int(time.time())
    recent_ts = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()

    fargate_compute = MagicMock()
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    # DDB result is EMPTY — readiness must come from the event log + file.
    job_record = {
        "owner_id": "user-1",
        "user_id": "user-1",
        "project_id": "proj-1",
        "job_id": "job-orphan-completed",
        "type": "eval",
        "state": "completed",
        "created_at": recent_ts,
        "completion_fired": False,
        "result": [],
        "config": {},
        "progress": {},
    }
    mock_store.list_recent_completed_jobs = AsyncMock(return_value=[job_record])
    # Event log carries the terminal `results` event with the real filename.
    run_filename = "eval_results_baseline_20260101.json"
    mock_store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "results", "file": run_filename}},
    ])
    mock_store.claim_completion_fired = AsyncMock(return_value=True)  # Win the claim

    file_result = [{"model_id": "test-model", "results": [{"success": True}]}]
    materialized = tmp_path / run_filename
    import json as _json
    materialized.write_text(_json.dumps(file_result))

    async def fake_materialize(proj, jid, fname):
        assert fname == run_filename
        return materialized

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            with patch("app.services.completion_turn.request_completion_turn") as mock_request_turn, \
                 patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
                 patch("app.services.jobs.eval_executor._dispatch_run_report", new=AsyncMock()), \
                 patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path):
                mock_platform.return_value.metadata_store = mock_store

                # Act
                result = await reconcile_once()

    # Assert: claim_completion_fired called, request_completion_turn fired with rebuilt Job
    mock_store.claim_completion_fired.assert_called_once_with("user-1", "proj-1", "job-orphan-completed")
    mock_request_turn.assert_called_once()
    call_args = mock_request_turn.call_args
    project_id_arg = call_args[0][0]
    job_arg = call_args[0][1]
    assert project_id_arg == "proj-1"
    assert isinstance(job_arg, Job)
    assert job_arg.job_id == "job-orphan-completed"
    assert job_arg.state == "completed"
    # Result sourced from the materialized FILE, not the (empty) DDB result.
    assert job_arg.result == file_result
    assert job_arg.output_file == run_filename


@pytest.mark.asyncio
async def test_reconciler_skips_already_fired_completion():
    """Reconciler skips jobs with completion_fired=true (second pass)."""
    fargate_compute = MagicMock()
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    mock_store.list_recent_completed_jobs = AsyncMock(
        return_value=[
            {
                "owner_id": "user-1",
                "project_id": "proj-1",
                "job_id": "job-already-fired",
                "type": "eval",
                "state": "completed",
                "completion_fired": True,  # Already fired
                "result": {},
            }
        ]
    )
    mock_store.claim_completion_fired = AsyncMock(return_value=False)  # Lose the claim

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            with patch("app.services.completion_turn.request_completion_turn") as mock_request_turn:
                mock_platform.return_value.metadata_store = mock_store

                # Act
                await reconcile_once()

    # Assert: request_completion_turn NOT called (lost claim)
    mock_request_turn.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_orphan_recovery_skips_when_output_unavailable(tmp_path):
    """Orphaned eval, terminal event present but materialize ceilings → no fire, no crash."""
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    from app.services.run_output_snapshot import RunOutputNotAvailable

    now = int(time.time())
    recent_ts = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()

    fargate_compute = MagicMock()
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    job_record = {
        "owner_id": "user-1",
        "project_id": "proj-1",
        "job_id": "job-orphan",
        "type": "eval",
        "state": "completed",
        "created_at": recent_ts,
        "completion_fired": False,
        "result": [],
        "config": {},
        "progress": {},
    }
    mock_store.list_recent_completed_jobs = AsyncMock(return_value=[job_record])
    mock_store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "results", "file": "eval_results_v1.json"}},
    ])
    mock_store.claim_completion_fired = AsyncMock(return_value=True)

    async def fake_materialize(*a, **k):
        raise RunOutputNotAvailable("never landed")

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            with patch("app.services.completion_turn.request_completion_turn") as mock_request_turn, \
                 patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize):
                mock_platform.return_value.metadata_store = mock_store

                result = await reconcile_once()

    # No fire, no claim consumed prematurely, pass completes cleanly.
    mock_request_turn.assert_not_called()
    mock_store.claim_completion_fired.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_orphan_recovery_skips_when_no_terminal_event(tmp_path):
    """Orphaned eval with NO terminal event in the log → not ready, skipped."""
    from app.platform.aws.fargate_task_compute import FargateTaskCompute

    now = int(time.time())
    recent_ts = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()

    fargate_compute = MagicMock()
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    job_record = {
        "owner_id": "user-1",
        "project_id": "proj-1",
        "job_id": "job-orphan",
        "type": "eval",
        "state": "completed",
        "created_at": recent_ts,
        "completion_fired": False,
        "result": [],
        "config": {},
        "progress": {},
    }
    mock_store.list_recent_completed_jobs = AsyncMock(return_value=[job_record])
    # No terminal `results` event in the log yet.
    mock_store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "scoring_progress", "current": 5}},
    ])
    mock_store.claim_completion_fired = AsyncMock(return_value=True)

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            with patch("app.services.completion_turn.request_completion_turn") as mock_request_turn:
                mock_platform.return_value.metadata_store = mock_store

                await reconcile_once()

    mock_request_turn.assert_not_called()
    mock_store.claim_completion_fired.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_excludes_cancelled_from_completion_recovery():
    """Reconciler never fires completion turn for cancelled jobs (no turn owed)."""
    fargate_compute = MagicMock()
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(return_value=[])

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    # list_recent_completed_jobs should exclude cancelled by design
    mock_store.list_recent_completed_jobs = AsyncMock(return_value=[])
    mock_store.claim_completion_fired = AsyncMock()

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = mock_store

            # Act
            await reconcile_once()

    # Assert: claim_completion_fired never called (no jobs returned)
    mock_store.claim_completion_fired.assert_not_called()


@pytest.mark.asyncio
async def test_reconciler_completion_recovery_survives_list_running_workers_failure(tmp_path):
    """Completion recovery still runs when list_running_workers() throws."""
    # Arrange: list_running_workers throws, but list_recent_completed_jobs works
    now = int(time.time())
    recent_ts = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()

    fargate_compute = MagicMock()
    from app.platform.aws.fargate_task_compute import FargateTaskCompute
    fargate_compute.__class__ = FargateTaskCompute
    fargate_compute.list_running_workers = AsyncMock(side_effect=Exception("ECS API error"))

    mock_store = MagicMock()
    mock_store.list_non_terminal_jobs = AsyncMock(return_value=[])
    # DDB result EMPTY — readiness from the event log + materialized file.
    job_record = {
        "owner_id": "user-1",
        "project_id": "proj-1",
        "job_id": "job-orphan",
        "type": "eval",
        "state": "completed",
        "created_at": recent_ts,
        "completion_fired": False,
        "result": [],
        "config": {},
        "progress": {},
    }
    run_filename = "eval_results_v1.json"
    mock_store.list_recent_completed_jobs = AsyncMock(return_value=[job_record])
    mock_store.events_since = AsyncMock(return_value=[
        {"seq": 1, "kind": "progress", "data": {"event": "results", "file": run_filename}},
    ])
    mock_store.claim_completion_fired = AsyncMock(return_value=True)

    materialized = tmp_path / run_filename
    import json as _json
    materialized.write_text(_json.dumps([{"model_id": "test-model", "results": [{"success": True}]}]))

    async def fake_materialize(proj, jid, fname):
        return materialized

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute):
        with patch("app.services.eval_reconciler.get_platform") as mock_platform:
            with patch("app.services.completion_turn.request_completion_turn") as mock_request_turn, \
                 patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
                 patch("app.services.jobs.eval_executor._dispatch_run_report", new=AsyncMock()), \
                 patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path):
                mock_platform.return_value.metadata_store = mock_store

                # Act
                result = await reconcile_once()

    # Assert: completion turn still fired despite list_running_workers exception
    mock_request_turn.assert_called_once()
    # The orphan/stuck duties should have failed (zeroed), but completion-recovery succeeded
    assert result["orphans_stopped"] == 0
    assert result["stuck_jobs_failed"] == 0
