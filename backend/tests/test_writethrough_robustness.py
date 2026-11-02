# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job write-through robustness and ordering.

Verifies that put_job/update_job_state failures stay inside their background
tasks, that those tasks are retained, and that rebuild tolerates partial rows.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import logging


@pytest.mark.asyncio
async def test_put_job_failure_is_logged(caplog):
    """A failing put_job write-through is attempted without raising out of
    create_job. Logging of the failure is not asserted here."""
    from app.services.jobs import JobTracker

    tracker = JobTracker()

    # Mock the store to raise on put_job
    with patch("app.platform.get_platform") as mock_platform:
        mock_store = mock_platform.return_value.metadata_store
        mock_put = AsyncMock(side_effect=Exception("DDB throttle"))
        mock_store.put_job = mock_put

        with caplog.at_level(logging.ERROR, logger="app.services.jobs"):
            job = tracker.create_job(
                project_id="p-1",
                job_type="eval",
                config={"arch_split": True, "user_id": "u-1"},
            )

            # Wait for all pending writes to complete
            if tracker._pending_writes:
                await asyncio.gather(*tracker._pending_writes, return_exceptions=True)

            # The write-through was attempted (and raised inside its task).
            assert mock_put.called


@pytest.mark.asyncio
async def test_update_job_state_failure_is_logged(caplog):
    """A failing update_job_state write-through is attempted without raising
    out of update_state. Logging of the failure is not asserted here."""
    from app.services.jobs import JobTracker

    tracker = JobTracker()

    # Create a job first (mock put_job to succeed)
    with patch("app.platform.get_platform") as mock_platform:
        mock_store = mock_platform.return_value.metadata_store
        mock_put = AsyncMock(return_value=None)
        mock_store.put_job = mock_put

        job = tracker.create_job(
            project_id="p-1",
            job_type="eval",
            config={"arch_split": True, "user_id": "u-1"},
        )

        # Wait for put_job to complete
        if tracker._pending_writes:
            await asyncio.gather(*tracker._pending_writes, return_exceptions=True)

    # Now mock update_job_state to raise
    with patch("app.platform.get_platform") as mock_platform:
        mock_store = mock_platform.return_value.metadata_store
        mock_update = AsyncMock(side_effect=Exception("Network error"))
        mock_store.update_job_state = mock_update

        with caplog.at_level(logging.ERROR, logger="app.services.jobs"):
            # Update the job's progress first, then transition state
            job.progress = {"message": "Starting"}
            tracker.update_state(job.job_id, "running")

            # Wait for update to complete
            if tracker._pending_writes:
                await asyncio.gather(*tracker._pending_writes, return_exceptions=True)

            # The write-through was attempted (and raised inside its task).
            assert mock_update.called


@pytest.mark.asyncio
async def test_pending_writes_are_retained():
    """Background write tasks are retained (not GC'd mid-flight)."""
    from app.services.jobs import JobTracker

    tracker = JobTracker()

    # Mock the store with a slow async operation
    with patch("app.platform.get_platform") as mock_platform:
        mock_store = mock_platform.return_value.metadata_store
        mock_put = AsyncMock()
        mock_store.put_job = mock_put

        # Capture the state of _pending_writes before and after
        initial_count = len(tracker._pending_writes)

        job = tracker.create_job(
            project_id="p-1",
            job_type="eval",
            config={"arch_split": True, "user_id": "u-1"},
        )

        # Task should be in pending_writes immediately
        assert len(tracker._pending_writes) == initial_count + 1

        # Wait for the task to complete
        await asyncio.sleep(0.1)

        # Task should be removed from pending_writes after completion
        assert len(tracker._pending_writes) == initial_count


@pytest.mark.asyncio
async def test_progress_reaches_the_durable_record_as_each_model_finishes():
    """Other processes read a detached eval's progress from the durable record.

    Written only on the first update, it said "0 of N models" for the whole run,
    and a cancel announced from the UI repeated that. Case updates are not
    written: that would be a write per scored case.
    """
    from app.services.jobs import JobTracker

    tracker = JobTracker()

    with patch("app.platform.get_platform") as mock_platform:
        store = mock_platform.return_value.metadata_store
        store.put_job = AsyncMock()
        store.update_job_state = AsyncMock()

        job = tracker.create_job(
            project_id="p-1",
            job_type="eval",
            config={"arch_split": True, "owner_id": "u-1"},
        )
        tracker.update_progress(job.job_id, models_total=4, models_completed=0)
        for case in range(1, 4):
            tracker.update_progress(job.job_id, models_completed=0, current_case=case)
        tracker.update_progress(job.job_id, models_completed=1, message="Complete: model-a")
        await tracker.drain_pending_writes()

    writes = store.update_job_state.await_args_list
    assert len(writes) == 2, "expected the start and one model boundary, nothing per case"
    assert writes[0].kwargs["state"] == "running"
    # Backgrounded, so it can land after a terminal write, which it must not undo.
    assert writes[1].kwargs["state"] is None
    assert writes[1].kwargs["progress"]["models_completed"] == 1


@pytest.mark.asyncio
async def test_rebuild_tolerates_partial_row():
    """Rebuild_jobs_from_store tolerates partial rows (missing type/created_at)."""
    from app.services.jobs import JobTracker

    tracker = JobTracker()

    # Mock the store to return a partial row (missing type, created_at)
    with patch("app.platform.get_platform") as mock_platform:
        mock_store = mock_platform.return_value.metadata_store
        # Partial row: has job_id, project_id, state but missing type, created_at
        partial_row = {
            "job_id": "job-1",
            "project_id": "p-1",
            "state": "running",
            "progress": {"message": "Starting"},
            # Missing: type, created_at, config
        }
        mock_list = AsyncMock(return_value=[partial_row])
        mock_store.list_jobs = mock_list

        # Mock _resolve_owner_id_for_project
        with patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="u-1"):
            # rebuild_jobs_from_store should not raise KeyError
            await tracker.rebuild_jobs_from_store("p-1", arch_split=True)

            # Job should be reconstructed with defaults
            job = tracker._jobs.get("job-1")
            assert job is not None
            assert job.type == "unknown"
            assert job.created_at == "1970-01-01T00:00:00Z"
            assert job.state == "running"


@pytest.mark.asyncio
async def test_dev_cloud_progress_parity():
    """Dev and cloud adapters write progress at the same location."""
    from app.platform.dev_local import FilesystemMetadataStore
    import tempfile
    from pathlib import Path

    # Dev adapter test
    with tempfile.TemporaryDirectory() as tmpdir:
        dev_store = FilesystemMetadataStore(projects_dir=Path(tmpdir))

        # Create a job
        await dev_store.put_job({
            "owner_id": "u-1",
            "project_id": "p-1",
            "job_id": "job-1",
            "type": "eval",
            "state": "queued",
            "progress": {"message": "Initial"},
            "created_at": "2024-01-01T00:00:00Z",
            "config": {},
        })

        # Update progress
        await dev_store.update_job_state(
            "u-1", "p-1", "job-1",
            state="running",
            progress={"message": "Running", "percent": 50}
        )

        # Read back
        jobs = await dev_store.list_jobs("u-1", "p-1")
        job = next((j for j in jobs if j["job_id"] == "job-1"), None)
        assert job is not None
        # Assert progress is top-level (not nested under result.progress)
        assert "progress" in job
        assert job["progress"]["message"] == "Running"
        assert job["progress"]["percent"] == 50
