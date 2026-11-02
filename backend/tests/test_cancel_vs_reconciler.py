# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reconciler pass during a cancel must not close the eval as failed.

A detached cancel stops the worker before it marks the job cancelled, so a failed
stop never reads "cancelled" while the worker is still billing. For that span the
job is non-terminal, and ECS has already taken the task off its running list:
StopTask sets the task's desired status to STOPPED at once, and the reconciler
lists only tasks whose desired status is RUNNING. A pass landing in that span read
a non-terminal job with no worker, which is how a crashed worker reads, and marked
it failed. A failed manual eval gets a chat turn of its own, so the user was told
the eval had failed before the cancel's own line arrived.

These run the real cancel and the real reconciler against one real dev store; only
ECS and the job tracker are doubles.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import set_platform
from app.platform.aws.fargate_task_compute import FargateTaskCompute
from app.platform.dev_local import FilesystemMetadataStore
from app.services.eval_cancel import CANCEL_IN_FLIGHT_S, cancel_eval_job
from app.services.eval_reconciler import reconcile_once

from .async_wait import until

pytestmark = pytest.mark.asyncio

TASK_ARN = "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123"
OWNER, PROJECT, JOB = "user-1", "proj-1", "job-1234abcd"


@pytest.fixture(autouse=True)
def _only_the_reap():
    """Keep the reconciler's other duties out of these tests.

    They need more of the platform than the store, and poller adoption would
    start a real poller for the eval under test that outlives the test.
    """
    with patch(
        "app.services.agentcore_session_reconciler.reconcile_expired_agentcore_claims",
        new=AsyncMock(),
    ), patch(
        "app.services.project_event_outbox.drain_project_event_outboxes",
        new=AsyncMock(return_value=0),
    ), patch(
        "app.services.optimization_loop_orchestrator.adopt_orphan_optimization_loops",
        new=AsyncMock(return_value=0),
    ), patch(
        "app.services.job_progress_republisher.adopt_unpolled_jobs",
        new=AsyncMock(return_value=0),
    ):
        yield


def _ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


async def _store_with_a_running_eval(tmp_path: Path, **fields) -> FilesystemMetadataStore:
    """A dev store holding one manual eval that has been running for five minutes."""
    store = FilesystemMetadataStore(projects_dir=tmp_path)
    await store.put_job({
        "owner_id": OWNER,
        "project_id": PROJECT,
        "job_id": JOB,
        "type": "eval",
        "state": "running",
        "compute_handle": TASK_ARN,
        "created_at": _ago(300),
        "config": {"source": "manual", "prompt_version": "v2"},
        "progress": {"models_completed": 1, "models_total": 4},
        **fields,
    })
    set_platform(SimpleNamespace(metadata_store=store))
    return store


def _worker_off_the_running_list() -> tuple[MagicMock, asyncio.Event]:
    """ECS for a task that has been sent StopTask, or has died.

    It is absent from the running list, and its containers report stopped only
    once the returned event is set, which keeps a cancel mid-stop for as long as
    a test needs.
    """
    containers_stopped = asyncio.Event()

    async def status(_handle):
        await containers_stopped.wait()
        return "DEPROVISIONING"

    compute = MagicMock()
    compute.__class__ = FargateTaskCompute
    compute.list_running_workers = AsyncMock(return_value=[])
    compute.cancel = AsyncMock()
    compute.status = AsyncMock(side_effect=status)
    return compute, containers_stopped


def _tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.rebuild_jobs_from_store = AsyncMock()
    tracker.get_job_for_project.return_value = MagicMock(status="running", pid=None)
    return tracker


async def test_a_pass_during_the_stop_leaves_the_eval_to_the_cancel(tmp_path):
    store = await _store_with_a_running_eval(tmp_path)
    compute, containers_stopped = _worker_off_the_running_list()
    tracker = _tracker()

    with patch(
        "app.services.eval_cancel.get_job_tracker", return_value=tracker
    ), patch(
        "app.platform.job_compute.get_job_compute", return_value=compute
    ), patch(
        "app.services.eval_reconciler.get_job_compute", return_value=compute
    ), patch(
        "app.services.eval_cancel._stop_optimization_loop", new=AsyncMock()
    ):
        cancel = asyncio.create_task(cancel_eval_job(
            PROJECT, JOB, user_id=OWNER, project_dir=tmp_path, arch_split=True,
        ))
        # StopTask is in, and the cancel is waiting for the containers to stop.
        await until(lambda: compute.status.await_count == 1, message="the cancel never began its stop")

        await reconcile_once()
        during_the_stop = await store.get_job(OWNER, PROJECT, JOB)

        containers_stopped.set()
        outcome, _ = await cancel

    assert during_the_stop["state"] == "running", (
        "the reconciler closed an eval that was being cancelled"
    )
    assert outcome == "cancelled"
    tracker.cancel_job.assert_called_once_with(JOB)


async def test_a_worker_that_died_on_its_own_is_still_failed(tmp_path):
    # The same eval and the same ECS picture, with no cancel: a crash.
    store = await _store_with_a_running_eval(tmp_path)
    compute, _ = _worker_off_the_running_list()

    with patch("app.services.eval_reconciler.get_job_compute", return_value=compute):
        result = await reconcile_once()

    assert (await store.get_job(OWNER, PROJECT, JOB))["state"] == "failed"
    assert result["stuck_jobs_failed"] == 1


async def test_a_cancel_that_gave_up_is_closed_as_cancelled(tmp_path):
    # A cancel requested long ago that never recorded an outcome: it reported
    # that the worker would not stop, or the process running it died.
    store = await _store_with_a_running_eval(
        tmp_path, cancel_requested_at=_ago(CANCEL_IN_FLIGHT_S + 60),
    )
    compute, _ = _worker_off_the_running_list()

    with patch("app.services.eval_reconciler.get_job_compute", return_value=compute):
        result = await reconcile_once()

    assert (await store.get_job(OWNER, PROJECT, JOB))["state"] == "cancelled"
    assert result["stuck_jobs_failed"] == 0
