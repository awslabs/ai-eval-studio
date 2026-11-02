# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral proof that the microVM starts no poller and issues no completion claim.

This test module covers the process-role gates in job_progress_republisher.py
(both the wrapper gate in maybe_start_poller and the defense-in-depth gate in
poll_and_republish_progress) plus the source-structure guard that no module other
than the republisher module itself contains a call to the poller coroutine.

The gates prevent the AgentCore runtime from owning progress republishing or
completion turns, which would be blind because the microVM has zero browser
WebSocket connections. Poller ownership belongs to the coordinator.

The source-structure guard catches future regressions where a new call site
adds a direct or indirect reference to the poller coroutine (direct call,
create_task with the coroutine as argument, or deferred-callable form).

Test coverage:
- With the process marked as the AgentCore runtime, maybe_start_poller creates
  no poller task (wrapper gate).
- With the same marker, poll_and_republish_progress returns promptly and never
  claims completion, even with a terminal results event in the log (coroutine gate).
- With the coordinator marker, maybe_start_poller creates exactly one poller task
  (positive control for the wrapper gate).
- With the coordinator marker, poll_and_republish_progress publishes progress and
  claims completion on terminal (positive control for the coroutine gate).
- Calling maybe_start_poller twice for the same job_id leaves exactly one live
  poller task (dedupe).
- Calling maybe_start_poller for two different job_ids leaves two live pollers.
- When a poller task finishes or is cancelled, its job_id is removed from the
  dedupe registry so a later adoption can legitimately re-start it.
- No module under backend/app/ other than the republisher module itself contains
  a reference to the poller coroutine (source-structure guard).
"""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


@pytest.fixture
def mock_store():
    """Mock metadata store with events_since and claim_completion_fired."""
    store = AsyncMock()
    store.events_since = AsyncMock(return_value=[])
    store.claim_completion_fired = AsyncMock(return_value=False)
    # The job record of an eval that is still running. The poller reads it each
    # cycle to notice an eval that ended without results, and again on the
    # results path to rebuild the Job it fires the completion turn with.
    store.get_job = AsyncMock(return_value={
        "job_id": "job-001",
        "project_id": "proj-001",
        "type": "eval",
        "state": "running",
        "config": {},
        "progress": {},
    })
    return store


@pytest.fixture
def mock_hub():
    """Mock WebSocket hub with publish_to_user."""
    hub = AsyncMock()
    hub.publish_to_user = AsyncMock()
    return hub


@pytest.mark.asyncio
async def test_microvm_wrapper_gate_creates_no_poller_task():
    """With the microVM marker, maybe_start_poller creates no poller task."""
    # Patch the resolved role to AgentCore runtime
    with patch.object(
        process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
    ):
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

    # Assert no poller task was created
    assert len(job_progress_republisher._poller_tasks) == 0
    assert len(job_progress_republisher._active_pollers) == 0


@pytest.mark.asyncio
async def test_microvm_coroutine_gate_returns_promptly_no_claim(mock_store, mock_hub):
    """With the microVM marker, the poller coroutine returns promptly and never claims completion."""
    # Seed the durable log with a progress event and a terminal results event
    mock_store.events_since = AsyncMock(
        return_value=[
            {
                "seq": 1,
                "kind": "progress",
                "data": {"event": "progress", "current": 1, "total": 10},
            },
            {
                "seq": 2,
                "kind": "progress",
                "data": {"event": "results", "file": "run_001.json"},
            },
        ]
    )

    # Patch the resolved role to AgentCore runtime, and patch get_platform/get_ws_hub
    with patch.object(
        process_role, "_PROCESS_ROLE", process_role.AGENTCORE_RUNTIME_ROLE
    ), patch(
        "app.services.job_progress_republisher.get_platform", return_value=AsyncMock(metadata_store=mock_store)
    ), patch(
        "app.services.job_progress_republisher.get_ws_hub", return_value=mock_hub
    ):
        # Await the coroutine with a short poll interval and a timeout
        # The coroutine should return immediately due to the gate
        try:
            await asyncio.wait_for(
                job_progress_republisher.poll_and_republish_progress(
                    job_id="job-001",
                    owner_id="owner-001",
                    project_id="proj-001",
                    poll_interval=0.1,
                    fire_completion=True,
                ),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("Coroutine did not return promptly — gate did not fire")

    # Assert zero publishes and that the claim mock was never awaited
    assert mock_hub.publish_to_user.await_count == 0
    assert mock_store.claim_completion_fired.await_count == 0


@pytest.mark.asyncio
async def test_coordinator_wrapper_gate_creates_one_poller_task():
    """With the coordinator marker, maybe_start_poller creates exactly one poller task."""
    # Patch the resolved role to coordinator (default)
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

    # Assert exactly one poller task was created
    assert len(job_progress_republisher._poller_tasks) == 1
    assert len(job_progress_republisher._active_pollers) == 1
    assert "job-001" in job_progress_republisher._active_pollers

    # Clean up
    for task in list(job_progress_republisher._poller_tasks):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_coordinator_coroutine_gate_publishes_and_claims(mock_store, mock_hub):
    """With the coordinator marker, the poller publishes progress and claims completion on terminal.

    Subject under test is the PROCESS-ROLE gate: a coordinator-role process must
    run the poller through to the completion path (the microVM-role sibling test
    asserts the inverse). ``claim_completion_fired`` is the observable proxy for
    "reached the completion path".

    Materialization is mocked because the claim is now taken only AFTER the run
    output is materialized (claiming first permanently lost the completion turn
    when the S3 upload had not landed — the claim is one-way and the reconciler
    skips claimed jobs). In a dev/local test process the real
    ``materialize_detached_run`` refuses outright (cloud/detached-only), so
    without this mock the poller correctly stops before the claim and the test
    would be asserting the old, unsafe ordering rather than the role gate.
    """
    # Seed the durable log with a progress event and a terminal results event
    # The poller should publish the progress envelope and attempt to claim completion
    mock_store.events_since = AsyncMock(
        return_value=[
            {
                "seq": 1,
                "kind": "progress",
                "data": {"event": "progress", "current": 1, "total": 10},
            },
        ]
    )
    mock_store.claim_completion_fired = AsyncMock(return_value=False)

    async def _materialized(*_args, **_kwargs):
        return "/tmp/run_001.json"

    # Patch the resolved role to coordinator and provide the store/hub
    with patch.object(
        process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE
    ), patch(
        "app.services.job_progress_republisher.get_platform",
        return_value=AsyncMock(metadata_store=mock_store),
    ), patch(
        "app.services.job_progress_republisher.get_ws_hub", return_value=mock_hub
    ), patch(
        "app.services.run_output_snapshot.materialize_detached_run",
        side_effect=_materialized,
    ), patch(
        "app.services.run_output_snapshot.load_materialized_results",
        return_value=[{"model": "m", "cases": []}],
    ):
        # Start the poller in the background
        task = asyncio.create_task(
            job_progress_republisher.poll_and_republish_progress(
                job_id="job-001",
                owner_id="owner-001",
                project_id="proj-001",
                poll_interval=0.1,
                fire_completion=True,
            )
        )

        # Let it run one cycle to publish the progress event
        await asyncio.sleep(0.2)

        # Now add a terminal event
        mock_store.events_since = AsyncMock(
            return_value=[
                {
                    "seq": 2,
                    "kind": "progress",
                    "data": {"event": "results", "file": "run_001.json"},
                }
            ]
        )

        # Let it run another cycle to detect terminal
        await asyncio.sleep(0.2)

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Assert at least one publish and at least one claim attempt
    assert mock_hub.publish_to_user.await_count >= 1
    # The claim is attempted when terminal is detected
    assert mock_store.claim_completion_fired.await_count >= 1


@pytest.mark.asyncio
async def test_same_job_twice_dedupe():
    """Calling maybe_start_poller twice for the same job_id leaves exactly one live poller."""
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # First call
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

        # Second call for the same job_id
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

    # Assert exactly one live poller task (dedupe worked)
    assert len(job_progress_republisher._poller_tasks) == 1
    assert len(job_progress_republisher._active_pollers) == 1

    # Clean up
    for task in list(job_progress_republisher._poller_tasks):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_two_different_jobs_two_pollers():
    """Calling maybe_start_poller for two different job_ids leaves two live pollers."""
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # First job
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

        # Second job
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-002",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

    # Assert two live poller tasks
    assert len(job_progress_republisher._poller_tasks) == 2
    assert len(job_progress_republisher._active_pollers) == 2
    assert "job-001" in job_progress_republisher._active_pollers
    assert "job-002" in job_progress_republisher._active_pollers

    # Clean up
    for task in list(job_progress_republisher._poller_tasks):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_poller_done_callback_removes_from_dedupe():
    """When a poller task finishes or is cancelled, its job_id is removed from the dedupe registry."""
    with patch.object(process_role, "_PROCESS_ROLE", process_role.COORDINATOR_ROLE):
        # Start a poller
        await job_progress_republisher.maybe_start_poller(
            project_id="proj-001",
            job_id="job-001",
            owner_id="owner-001",
            arch_split=True,
            fire_completion=True,
        )

    # Assert it's registered
    assert "job-001" in job_progress_republisher._active_pollers

    # Cancel the task
    for task in list(job_progress_republisher._poller_tasks):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Give the done callback a moment to run
    await asyncio.sleep(0.01)

    # Assert the job_id was removed from the dedupe registry
    assert "job-001" not in job_progress_republisher._active_pollers


def test_source_structure_guard_no_external_poller_refs():
    """No module under backend/app/ other than the republisher module itself contains a reference to the poller coroutine."""
    backend_dir = Path(__file__).parent.parent / "app"
    republisher_path = backend_dir / "services" / "job_progress_republisher.py"
    offending_files = set()

    poller_coroutine_name = "poll_and_republish_progress"

    for py_file in backend_dir.rglob("*.py"):
        # Skip the republisher module itself
        if py_file.resolve() == republisher_path.resolve():
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception:
            # Skip files that can't be parsed (shouldn't happen)
            continue

        # Walk every Name node whose id matches the coroutine name, regardless of
        # its syntactic position. This catches:
        # - Direct calls: poll_and_republish_progress(...)
        # - create_task arguments: asyncio.create_task(poll_and_republish_progress(...))
        # - Deferred callable: create_task(poll_and_republish_progress)
        # - Any other reference
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == poller_coroutine_name:
                # Exclude import statements (the module's own re-export surface)
                parent = getattr(node, "parent", None)
                if not isinstance(parent, (ast.Import, ast.ImportFrom)):
                    offending_files.add(str(py_file.relative_to(backend_dir)))

    # Assert no offending files
    if offending_files:
        pytest.fail(
            f"Source-structure guard failed: the following files contain references "
            f"to {poller_coroutine_name}: {sorted(offending_files)}"
        )
