# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reconciler's age-based reap of handle-less non-terminal jobs.

Non-terminal jobs with NO ``compute_handle`` were skipped by the reconciler
(``if not compute_handle: continue``) — so they leaked forever against the
per-user/account job caps (no TTL on job items). Dominant sources: the
in-process ``optimization_iteration`` parent (never gets a handle), evals that
died before the handle write, and records whose handle was wiped when the
worker's full-item ``put_job`` still overwrote the coordinator's record (the
worker now registers with ``durable_write=False``).

The reap marks a handle-less non-terminal ``eval`` job ``failed`` once it is
older than the grace window (clearing its stale eval lock), while a handle-less
eval WITHIN grace is left untouched (protects the legitimate brief
queued→dispatched window). The handle-HAVING stuck-job path is unchanged.

The reap is scoped to ``type == "eval"`` only. Other job types are handle-less
BY DESIGN and legitimately run past the 60s grace: the ``optimization_iteration``
parent launches no worker (its child eval does) and runs for minutes, and a
``harness_build`` job is in-process. Reaping those on the grace timer would
false-fail a live optimization mid-run — so only evals (which are EXPECTED to
carry a compute_handle now that the worker no longer overwrites the
coordinator's record) qualify as handle-less zombies.

These tests trip the guard behaviorally (AsyncMock store, mirroring
tests/test_reconciler.py).
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.eval_reconciler import reconcile_once


@pytest.fixture
def mock_store():
    """Mock MetadataStore with the reconciler's read/write surface."""
    store = MagicMock()
    store.list_non_terminal_jobs = AsyncMock(return_value=[])
    store.list_recent_completed_jobs = AsyncMock(return_value=[])
    store.get_eval_lock = AsyncMock(return_value=None)
    store.clear_stale_eval_lock = AsyncMock(return_value=True)
    store.release_eval_lock = AsyncMock()
    store.claim_completion_fired = AsyncMock(return_value=False)
    store.update_job_state = AsyncMock()

    # The reap re-reads each job strongly before acting on it. With nothing
    # written since the listing, that read returns the listed record.
    async def _get_listed_job(user_id, project_id, job_id, *, consistent=False):
        listed = store.list_non_terminal_jobs.return_value
        return next((dict(j) for j in listed if j["job_id"] == job_id), None)

    store.get_job = AsyncMock(side_effect=_get_listed_job)
    return store


@pytest.fixture
def fargate_compute():
    """Compute mock that satisfies isinstance(FargateTaskCompute)."""
    from app.platform.aws.fargate_task_compute import FargateTaskCompute

    compute = MagicMock()
    compute.list_running_workers = AsyncMock(return_value=[])
    compute.cancel = AsyncMock()
    compute.__class__ = FargateTaskCompute
    return compute


@pytest.mark.asyncio
async def test_handleless_job_past_grace_marked_failed(fargate_compute, mock_store):
    """A handle-less non-terminal job older than grace is marked failed + lock cleared."""
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 120, tz=timezone.utc).isoformat()
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "job_id": "job-zombie",
            "project_id": "proj-1",
            "owner_id": "owner-1",
            "type": "eval",
            "created_at": old_created_at,
            # No compute_handle key — an eval that died before the handle write
            # or a record whose handle an old worker-side put_job wiped.
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute), \
         patch("app.services.eval_reconciler.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        result = await reconcile_once()

    assert result["stuck_jobs_failed"] == 1
    failed_writes = [
        c for c in mock_store.update_job_state.call_args_list
        if c.kwargs.get("state") == "failed"
    ]
    assert len(failed_writes) == 1
    assert failed_writes[0].args[0] == "owner-1"
    assert failed_writes[0].args[2] == "job-zombie"
    # Belt-and-suspenders lock release fired for the reaped job.
    mock_store.release_eval_lock.assert_awaited_with("owner-1", "proj-1", "job-zombie")


@pytest.mark.asyncio
async def test_handleless_job_within_grace_skipped(fargate_compute, mock_store):
    """A handle-less non-terminal job younger than grace is NOT touched."""
    now = int(time.time())
    fresh_created_at = datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat()
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "job_id": "job-fresh",
            "project_id": "proj-1",
            "owner_id": "owner-1",
            "type": "eval",
            "created_at": fresh_created_at,
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute), \
         patch("app.services.eval_reconciler.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        result = await reconcile_once()

    assert result["stuck_jobs_failed"] == 0
    mock_store.update_job_state.assert_not_called()


@pytest.mark.asyncio
async def test_handleless_optimization_iteration_parent_never_reaped(
    fargate_compute, mock_store
):
    """A handle-less optimization_iteration parent past grace is NOT reaped.

    Interaction regression: the optimization loop creates an
    ``optimization_iteration`` parent that launches no worker (its child eval
    does), so it is handle-less BY DESIGN and stays non-terminal for the whole
    iteration (minutes ≫ the 60s grace). The blanket handle-less reap would
    mark it failed + clear the eval lock ~120s in, breaking every live
    optimization. The reap must be scoped to ``type == "eval"`` only.
    """
    now = int(time.time())
    old_created_at = datetime.fromtimestamp(now - 600, tz=timezone.utc).isoformat()
    mock_store.list_non_terminal_jobs.return_value = [
        {
            "job_id": "opt-parent",
            "project_id": "proj-1",
            "owner_id": "owner-1",
            "type": "optimization_iteration",
            "created_at": old_created_at,
            # Handle-less by design — no worker is launched for the parent.
        }
    ]

    with patch("app.services.eval_reconciler.get_job_compute", return_value=fargate_compute), \
         patch("app.services.eval_reconciler.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        result = await reconcile_once()

    # The running optimization must be left completely untouched.
    assert result["stuck_jobs_failed"] == 0
    mock_store.update_job_state.assert_not_called()
    mock_store.release_eval_lock.assert_not_called()
    mock_store.clear_stale_eval_lock.assert_not_called()
