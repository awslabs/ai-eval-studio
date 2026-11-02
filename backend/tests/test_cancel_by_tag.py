# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tag-first cancel of detached Fargate eval workers.

``compute_handle`` can be null on an optimization eval record (records
written while the worker still overwrote it carry none), so cancel cannot rely
on the durable record carrying the handle. ECS task tags (Owner/Project/Job)
are set on ``run_task`` and are independent of the record — so cancel
discovers the worker task by its ``Job`` tag and ``stop_task``s it.

These tests trip the guard behaviorally:

* ``FargateTaskCompute.cancel_by_tag`` stops the task whose ``Job`` tag matches
  (cross-checking Owner/Project) and returns that task's ARN; it returns None
  when no task matches.
* ``hard_cancel_project_eval`` calls ``cancel_by_tag`` as the PRIMARY mechanism
  for an arch_split eval whose durable ``compute_handle`` is null — so
  ``stop_task`` fires even with a null handle.
* The dev PID-kill path is unchanged (no ECS call).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform.aws.fargate_task_compute import FargateTaskCompute


def _make_compute(list_workers_return):
    """Build a FargateTaskCompute with a mocked ECS client + list_running_workers."""
    compute = FargateTaskCompute(
        ecs_client=MagicMock(),
        cluster_arn="arn:aws:ecs:us-east-1:123:cluster/c",
        task_definition_arn="arn:aws:ecs:us-east-1:123:task-definition/w:1",
        subnet_ids=["subnet-1"],
        security_group_id="sg-1",
        token_vending=MagicMock(),
        s3_client=MagicMock(),
        snapshot_bucket="snap-bucket",
        projects_dir="/mnt/projects",
    )
    compute.list_running_workers = AsyncMock(return_value=list_workers_return)
    compute.cancel = AsyncMock()
    return compute


@pytest.mark.asyncio
async def test_cancel_by_tag_stops_matching_task():
    """cancel_by_tag stops the matching task and returns ITS ARN.

    The ARN, not a bool: StopTask returning only means ECS accepted the request,
    so the caller needs the ARN to go on and confirm the task reached STOPPED.
    """
    compute = _make_compute([
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/other",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Owner", "value": "owner-1"},
                {"key": "Project", "value": "proj-1"},
                {"key": "Job", "value": "job-OTHER"},
            ],
        },
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/target",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Owner", "value": "owner-1"},
                {"key": "Project", "value": "proj-1"},
                {"key": "Job", "value": "job-TARGET"},
            ],
        },
    ])

    result = await compute.cancel_by_tag("owner-1", "proj-1", "job-TARGET")

    assert result == "arn:aws:ecs:us-east-1:123:task/target"
    compute.cancel.assert_awaited_once()
    handle = compute.cancel.call_args.args[0]
    assert handle.value == "arn:aws:ecs:us-east-1:123:task/target"
    assert handle.kind == "fargate_task"


@pytest.mark.asyncio
async def test_cancel_by_tag_no_match_returns_none():
    """cancel_by_tag returns None and stops nothing when no Job tag matches."""
    compute = _make_compute([
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/other",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Owner", "value": "owner-1"},
                {"key": "Project", "value": "proj-1"},
                {"key": "Job", "value": "job-OTHER"},
            ],
        },
    ])

    result = await compute.cancel_by_tag("owner-1", "proj-1", "job-MISSING")

    assert result is None
    compute.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_by_tag_cross_checks_owner_project():
    """cancel_by_tag does NOT stop a task whose Job tag matches but Owner differs."""
    compute = _make_compute([
        {
            "taskArn": "arn:aws:ecs:us-east-1:123:task/wrong-tenant",
            "lastStatus": "RUNNING",
            "tags": [
                {"key": "Owner", "value": "OTHER-OWNER"},
                {"key": "Project", "value": "proj-1"},
                {"key": "Job", "value": "job-TARGET"},
            ],
        },
    ])

    result = await compute.cancel_by_tag("owner-1", "proj-1", "job-TARGET")

    assert result is None
    compute.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_hard_cancel_uses_tag_first_when_handle_null():
    """hard_cancel_project_eval calls cancel_by_tag for an arch_split eval with null handle."""
    from app.services.harness_runner import hard_cancel_project_eval

    # One active arch_split eval job, no pid (detached worker).
    eval_job = MagicMock()
    eval_job.job_id = "job-detached"
    eval_job.type = "eval"
    eval_job.pid = None
    eval_job.config = {"arch_split": True, "owner_id": "owner-1"}
    eval_job._task = None

    tracker = MagicMock()
    tracker.get_active_jobs.return_value = [eval_job]
    tracker.update_state = MagicMock()

    # Durable record has a NULL compute_handle (the leaked-record shape).
    store = AsyncMock()
    store.get_job = AsyncMock(return_value={"compute_handle": None})

    compute = MagicMock(spec=FargateTaskCompute)
    compute.cancel_by_tag = AsyncMock(return_value="arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123")
    compute.cancel = AsyncMock()

    with patch("app.services.harness_runner.get_job_tracker", return_value=tracker), \
         patch("app.platform.get_platform") as mock_platform, \
         patch("app.platform.job_compute.get_job_compute", return_value=compute):
        mock_platform.return_value.metadata_store = store

        count = await hard_cancel_project_eval("proj-1")

    assert count == 1
    # Tag-first cancel fired even though compute_handle is null.
    compute.cancel_by_tag.assert_awaited_once_with("owner-1", "proj-1", "job-detached")
    # Secondary compute_handle StopTask NOT used (handle null + tag succeeded).
    compute.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_hard_cancel_dev_pid_path_unchanged():
    """A non-arch_split eval with a pid takes the PID-kill path, no ECS call."""
    from app.services import harness_runner

    eval_job = MagicMock()
    eval_job.job_id = "job-dev"
    eval_job.type = "eval"
    eval_job.pid = 4242
    eval_job.config = {}  # dev/in-process: no arch_split
    eval_job._task = None

    tracker = MagicMock()
    tracker.get_active_jobs.return_value = [eval_job]
    tracker.update_state = MagicMock()

    kill_mock = AsyncMock()

    with patch("app.services.harness_runner.get_job_tracker", return_value=tracker), \
         patch("app.services.harness_runner.kill_harness_subprocess", kill_mock), \
         patch("app.platform.job_compute.get_job_compute") as get_compute:
        count = await harness_runner.hard_cancel_project_eval("proj-1")

    assert count == 1
    kill_mock.assert_awaited_once_with(4242)
    # No compute seam touched on the dev path.
    get_compute.assert_not_called()
