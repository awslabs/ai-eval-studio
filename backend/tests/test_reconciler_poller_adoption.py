# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral proof of adopt-on-reconcile, per-job dedupe, and source-derived fire_completion.

This test module covers the coordinator-side adopt sweep (adopt_unpolled_jobs)
wired into the reconciler's periodic pass, ensuring that jobs created in the
AgentCore microVM are adopted by the coordinator's poller ownership.

Test coverage:
- A reconcile pass over a non-terminal job carrying an ECS task ARN handle,
  which the coordinator is not already polling, starts one poller for it.
- Two consecutive reconcile passes over the same job leave exactly one live poller (dedupe).
- A job whose durable config marks it optimization-sourced is adopted progress-only (fire_completion=False).
- A manual-sourced job is adopted completion-owning (fire_completion=True).
- A job whose handle is absent or PID-shaped is not adopted.
- A failure reading the job list, or a failure adopting one job, does not prevent the
  remaining jobs from being adopted and does not raise out of the reconcile pass.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import process_role
from app.services import job_progress_republisher


@pytest.fixture(autouse=True)
def reset_poller_state():
    """Clear the poller task registry and dedupe dict before and after each test."""
    job_progress_republisher._poller_tasks.clear()
    job_progress_republisher._active_pollers.clear()
    yield
    job_progress_republisher._poller_tasks.clear()
    job_progress_republisher._active_pollers.clear()


@pytest.mark.asyncio
async def test_reconcile_adopts_unpolled_fargate_job():
    """A reconcile pass over a non-terminal job carrying an ECS task ARN handle starts one poller for it."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with one non-terminal eval job carrying a Fargate handle
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-001",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/abc123",
                    "config": {"source": "manual"},
                }
            ]
        )

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ):
            # Call the adopt sweep
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert exactly one poller was started
            assert adopted == 1
            assert len(job_progress_republisher._poller_tasks) == 1
            assert "job-001" in job_progress_republisher._active_pollers

            # Clean up
            for task in list(job_progress_republisher._poller_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_two_consecutive_sweeps_leave_one_poller():
    """Two consecutive reconcile passes over the same job leave exactly one live poller (dedupe)."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with one non-terminal eval job
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-001",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/abc123",
                    "config": {"source": "manual"},
                }
            ]
        )

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ):
            # First sweep
            adopted1 = await job_progress_republisher.adopt_unpolled_jobs()
            # Second sweep
            adopted2 = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: first sweep adopted 1, second sweep adopted 0 (dedupe worked)
            assert adopted1 == 1
            assert adopted2 == 0
            assert len(job_progress_republisher._poller_tasks) == 1
            assert "job-001" in job_progress_republisher._active_pollers

            # Clean up
            for task in list(job_progress_republisher._poller_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_optimization_sourced_job_adopted_progress_only():
    """A job whose durable config marks it optimization-sourced is adopted progress-only (fire_completion=False)."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with one optimization-sourced eval job
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-opt",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/abc123",
                    "config": {"source": "optimization"},
                }
            ]
        )

        # Track the fire_completion argument passed to maybe_start_poller
        fire_completion_arg = None

        async def track_maybe_start_poller(
            project_id, job_id, owner_id, arch_split, *, fire_completion=True
        ):
            nonlocal fire_completion_arg
            fire_completion_arg = fire_completion
            # Create a real task so dedupe works
            task = asyncio.create_task(asyncio.sleep(10), name=f"progress-poller-{job_id[:8]}")
            job_progress_republisher._poller_tasks.add(task)
            job_progress_republisher._active_pollers[job_id] = task
            task.add_done_callback(lambda t: job_progress_republisher._poller_tasks.discard(t))
            # Mirror the real contract: True means a poller was newly started.
            return True

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ), patch(
            "app.services.job_progress_republisher.maybe_start_poller",
            side_effect=track_maybe_start_poller,
        ):
            # Call the adopt sweep
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: job was adopted with fire_completion=False
            assert adopted == 1
            assert fire_completion_arg is False

            # Clean up
            for task in list(job_progress_republisher._poller_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_manual_sourced_job_adopted_completion_owning():
    """A manual-sourced job is adopted completion-owning (fire_completion=True)."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with one manual-sourced eval job
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-manual",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/abc123",
                    "config": {"source": "manual"},
                }
            ]
        )

        # Track the fire_completion argument passed to maybe_start_poller
        fire_completion_arg = None

        async def track_maybe_start_poller(
            project_id, job_id, owner_id, arch_split, *, fire_completion=True
        ):
            nonlocal fire_completion_arg
            fire_completion_arg = fire_completion
            # Create a real task so dedupe works
            task = asyncio.create_task(asyncio.sleep(10), name=f"progress-poller-{job_id[:8]}")
            job_progress_republisher._poller_tasks.add(task)
            job_progress_republisher._active_pollers[job_id] = task
            task.add_done_callback(lambda t: job_progress_republisher._poller_tasks.discard(t))
            # Mirror the real contract: True means a poller was newly started.
            return True

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ), patch(
            "app.services.job_progress_republisher.maybe_start_poller",
            side_effect=track_maybe_start_poller,
        ):
            # Call the adopt sweep
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: job was adopted with fire_completion=True
            assert adopted == 1
            assert fire_completion_arg is True

            # Clean up
            for task in list(job_progress_republisher._poller_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.asyncio
async def test_skip_pid_shaped_and_absent_handle():
    """A job whose handle is absent or PID-shaped is not adopted."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with two jobs: one PID-shaped, one absent handle
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-pid",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "12345",
                    "config": {"source": "manual"},
                },
                {
                    "job_id": "job-nohandle",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    # No compute_handle
                    "config": {"source": "manual"},
                },
            ]
        )

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ):
            # Call the adopt sweep
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: no jobs were adopted (both skipped)
            assert adopted == 0
            assert len(job_progress_republisher._poller_tasks) == 0


@pytest.mark.asyncio
async def test_failure_reading_job_list_does_not_raise():
    """A failure reading the job list does not raise out of the adopt sweep."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store that throws on list_non_terminal_jobs
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            side_effect=Exception("store unavailable")
        )

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ):
            # Call the adopt sweep — should not raise
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: zero jobs adopted, no exception raised
            assert adopted == 0


@pytest.mark.asyncio
async def test_failure_adopting_one_job_does_not_block_others():
    """A failure adopting one job does not prevent the remaining jobs from being adopted."""
    # Set up coordinator role
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Mock store with two jobs
        mock_store = MagicMock()
        mock_store.list_non_terminal_jobs = AsyncMock(
            return_value=[
                {
                    "job_id": "job-bad",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/bad",
                    "config": {"source": "manual"},
                },
                {
                    "job_id": "job-good",
                    "project_id": "proj-001",
                    "owner_id": "owner-001",
                    "type": "eval",
                    "state": "running",
                    "compute_handle": "arn:aws:ecs:us-east-1:123:task/cluster/good",
                    "config": {"source": "manual"},
                },
            ]
        )

        # Mock maybe_start_poller to fail on the first job, succeed on the second
        call_count = [0]

        async def mock_maybe_start_poller(
            project_id, job_id, owner_id, arch_split, *, fire_completion=True
        ):
            call_count[0] += 1
            if job_id == "job-bad":
                raise Exception("poller start failed")
            # Succeed for job-good
            task = asyncio.create_task(asyncio.sleep(10), name=f"progress-poller-{job_id[:8]}")
            job_progress_republisher._poller_tasks.add(task)
            job_progress_republisher._active_pollers[job_id] = task
            task.add_done_callback(lambda t: job_progress_republisher._poller_tasks.discard(t))
            # Mirror the real contract: True means a poller was newly started.
            return True

        with patch(
            "app.services.job_progress_republisher.get_platform",
            return_value=MagicMock(metadata_store=mock_store),
        ), patch(
            "app.services.job_progress_republisher.maybe_start_poller",
            side_effect=mock_maybe_start_poller,
        ):
            # Call the adopt sweep
            adopted = await job_progress_republisher.adopt_unpolled_jobs()

            # Assert: one job adopted (job-good), one skipped (job-bad failed)
            assert adopted == 1
            assert "job-good" in job_progress_republisher._active_pollers
            assert "job-bad" not in job_progress_republisher._active_pollers

            # Clean up
            for task in list(job_progress_republisher._poller_tasks):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
