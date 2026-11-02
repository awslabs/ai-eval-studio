# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test job_progress_republisher.py — the durable progress bridge.

This test suite proves that detached worker
progress written to the durable DDB event log is polled via events_since and
republished onto the owner's /ws job:<id> stream, with tenant isolation preserved.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.platform import get_platform
from app.services.job_progress_republisher import (
    maybe_start_poller,
    poll_and_republish_progress,
)


@pytest.mark.asyncio
async def test_poll_and_republish_progress_ddb_seq() -> None:
    """Prove the republished envelope carries the DDB seq (behavioral proof)."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    # Seed the durable log with 3 progress events (simulate the writer)
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    for seq in range(1, 4):
        durable_event = {
            "stream": f"job:{job_id}",
            "seq": seq,  # DDB seq
            "kind": "progress",
            "data": {"event": "progress", "case": seq, "total": 3},
        }
        await store.append_job_event(job_id, seq, durable_event, ttl=ttl)

    # Spy on ws_hub.publish_to_user
    published_envelopes = []
    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        async def capture_publish(user_id, envelope):
            published_envelopes.append((user_id, envelope))
        mock_hub.return_value.publish_to_user = AsyncMock(side_effect=capture_publish)

        # Run ONE poll cycle (manually, not via maybe_start_poller)
        # We don't start the infinite loop — just call the core logic once
        # by simulating what the loop does
        from app.services.job_progress_republisher import poll_and_republish_progress

        # Start the poller task
        task = asyncio.create_task(
            poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
            name=f"test-poller-{job_id[:8]}"
        )

        # Let it run one cycle
        await asyncio.sleep(0.2)

        # Cancel the poller
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Assert: 3 envelopes were published
    assert len(published_envelopes) == 3, (
        f"Expected 3 published envelopes, got {len(published_envelopes)}"
    )

    # Assert: each envelope's seq equals the DDB seq (behavioral proof)
    for idx, (user_id, envelope) in enumerate(published_envelopes, start=1):
        assert user_id == owner_id, "Envelope must be published to the owner"
        assert envelope["seq"] == idx, (
            f"Envelope {idx} has seq={envelope['seq']}, expected {idx}. "
            f"seq coherence violation: republished envelope seq must equal DDB seq."
        )
        assert envelope["stream"] == f"job:{job_id}"
        assert envelope["kind"] == "progress"


@pytest.mark.asyncio
async def test_poll_advances_last_seq_no_republish() -> None:
    """Prove the poller does NOT re-publish already-sent events across cycles."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    # Seed the durable log with 2 events
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    for seq in range(1, 3):
        durable_event = {
            "stream": f"job:{job_id}",
            "seq": seq,
            "kind": "progress",
            "data": {"event": "progress", "case": seq},
        }
        await store.append_job_event(job_id, seq, durable_event, ttl=ttl)

    published_envelopes = []
    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        async def capture_publish(user_id, envelope):
            published_envelopes.append((user_id, envelope))
        mock_hub.return_value.publish_to_user = AsyncMock(side_effect=capture_publish)

        # Start the poller
        task = asyncio.create_task(
            poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
            name=f"test-poller-{job_id[:8]}"
        )

        # Let it run TWO cycles
        await asyncio.sleep(0.3)

        # Cancel the poller
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Assert: exactly 2 envelopes were published (not 4, which would indicate re-publish)
    assert len(published_envelopes) == 2, (
        f"Expected 2 envelopes (no re-publish), got {len(published_envelopes)}"
    )


@pytest.mark.asyncio
async def test_poller_skips_malformed_event_keeps_running() -> None:
    """A malformed event is skipped; the poller keeps republishing later good events.

    Seeds seq=1 (good), seq=2 (malformed — missing "data"), seq=3, seq=4 (good).
    Proves the poller does NOT die on the KeyError at seq=2 and still republishes
    seq=3 and seq=4. Against the buggy code (direct event["data"] access outside
    the inner try), the KeyError escapes the while loop and the poller dies, so
    seq=3 and seq=4 are never republished.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    import time
    ttl = int(time.time()) + (7 * 24 * 3600)

    # seq=1 good
    await store.append_job_event(job_id, 1, {
        "stream": f"job:{job_id}", "seq": 1, "kind": "progress",
        "data": {"event": "progress", "case": 1},
    }, ttl=ttl)
    # seq=2 MALFORMED — missing "data" key
    await store.append_job_event(job_id, 2, {
        "stream": f"job:{job_id}", "seq": 2, "kind": "progress",
        # intentionally NO "data"
    }, ttl=ttl)
    # seq=3, seq=4 good
    for seq in (3, 4):
        await store.append_job_event(job_id, seq, {
            "stream": f"job:{job_id}", "seq": seq, "kind": "progress",
            "data": {"event": "progress", "case": seq},
        }, ttl=ttl)

    published_seqs = []
    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        async def capture_publish(user_id, envelope):
            published_seqs.append(envelope["seq"])
        mock_hub.return_value.publish_to_user = AsyncMock(side_effect=capture_publish)

        task = asyncio.create_task(
            poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
            name=f"test-poller-{job_id[:8]}"
        )
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # The malformed seq=2 must be skipped, but seq=1, 3, 4 must ALL be republished.
    # Against the buggy code, the poller dies at seq=2 and 3/4 never publish.
    assert 1 in published_seqs, "seq=1 (before malformed) should be republished"
    assert 3 in published_seqs, "seq=3 (after malformed) should be republished — poller must survive"
    assert 4 in published_seqs, "seq=4 (after malformed) should be republished — poller must survive"
    assert 2 not in published_seqs, "seq=2 (malformed) should be SKIPPED, not published"


@pytest.mark.asyncio
async def test_publish_targets_owner_id_only() -> None:
    """Prove every publish targets owner_id only (tenant isolation)."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    # Seed one event
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    durable_event = {
        "stream": f"job:{job_id}",
        "seq": 1,
        "kind": "progress",
        "data": {"event": "progress"},
    }
    await store.append_job_event(job_id, 1, durable_event, ttl=ttl)

    published_user_ids = []
    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        async def capture_user_id(user_id, envelope):
            published_user_ids.append(user_id)
        mock_hub.return_value.publish_to_user = AsyncMock(side_effect=capture_user_id)

        # Start the poller
        task = asyncio.create_task(
            poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
            name=f"test-poller-{job_id[:8]}"
        )

        # Let it run one cycle
        await asyncio.sleep(0.2)

        # Cancel
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Assert: all publishes targeted owner_id
    assert len(published_user_ids) == 1, "Expected 1 publish"
    assert published_user_ids[0] == owner_id, (
        f"Published to {published_user_ids[0]}, expected {owner_id}. "
        f"tenant-scoping violation: must publish only to the job owner."
    )


@pytest.mark.asyncio
async def test_maybe_start_poller_flag_gate() -> None:
    """Prove maybe_start_poller does NOT start when arch_split is False."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    owner_id = "test-owner"
    project_id = "test-proj"

    # pass the pre-resolved boolean directly (no more internal resolve)
    # Call maybe_start_poller with arch_split=False
    await maybe_start_poller(project_id, job_id, owner_id, arch_split=False)

    # Wait a bit
    await asyncio.sleep(0.1)

    # Assert: no poller task was created (we can check by looking at active tasks)
    poller_tasks = [t for t in asyncio.all_tasks() if "progress-poller" in t.get_name()]
    assert len(poller_tasks) == 0, "No poller task should start when arch_split=False"


@pytest.mark.asyncio
async def test_maybe_start_poller_starts_on_new_path() -> None:
    """Prove maybe_start_poller starts the poller when resolve_arch_split is True."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    owner_id = "test-owner"
    project_id = "test-proj"

    # pass the pre-resolved boolean directly (no more internal resolve)
    with patch("app.services.job_progress_republisher.get_ws_hub"):
        # Call maybe_start_poller with arch_split=True
        await maybe_start_poller(project_id, job_id, owner_id, arch_split=True)

        # Wait for the task to be created
        await asyncio.sleep(0.1)

        # Find the poller task by name
        poller_tasks = [
            t for t in asyncio.all_tasks()
            if t.get_name().startswith("progress-poller-")
        ]

        # Assert: one poller task exists
        assert len(poller_tasks) == 1, (
            f"Expected 1 poller task, found {len(poller_tasks)}"
        )

        # Clean up
        for task in poller_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_reconnect_replay_from_ddb_no_duplicate_no_skip() -> None:
    """Prove reconnect replays exactly the events with seq>k, in order."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    owner_id = "test-owner"

    # Register job and project with real singletons (no mocking needed)
    # Cross-file test-isolation: the _reset_singletons autouse fixture in
    # conftest.py clears JobTracker and Platform before each test, so this
    # test gets a fresh tracker even after polluter tests that patched
    # get_platform(). Using real singletons instead of patches keeps the test
    # independent of where app/api/ws.py imports get_platform from.
    from app.services.jobs import Job, get_job_tracker
    from app.main import get_project_manager
    from app.models.schemas import ProjectCreate

    # Create real project on disk
    pm = get_project_manager()
    project_config = pm.create_project(
        ProjectCreate(name="Test Project for Replay"),
        owner_id=owner_id,
    )
    project_id = project_config.id

    store = get_platform().metadata_store

    # Seed the durable log with 5 progress events (simulate a running eval)
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    for seq in range(1, 6):
        durable_event = {
            "stream": f"job:{job_id}",
            "seq": seq,
            "kind": "progress",
            "data": {"event": "progress", "case": seq, "total": 5},
        }
        await store.append_job_event(job_id, seq, durable_event, ttl=ttl)

    # Simulate a reconnect subscribe with since_seq=2 (client saw 1, 2; wants 3, 4, 5)
    from app.services.ws_hub import WsConnection
    from app.api.ws import _replay_job_stream_from_durable_log
    from unittest.mock import MagicMock

    # Mock connection
    mock_websocket = MagicMock()
    conn = WsConnection(
        connection_id=f"conn-{uuid.uuid4().hex[:8]}",
        user_id=owner_id,
        websocket=mock_websocket,
    )

    # Create and register job
    tracker = get_job_tracker()
    test_job = Job(
        job_id=job_id,
        project_id=project_id,
        type="eval",
        state="running",
        config={},
    )
    tracker._jobs[job_id] = test_job

    # Replay from since_seq=2
    count, last_seq = await _replay_job_stream_from_durable_log(
        conn, f"job:{job_id}", since_seq=2
    )

    # Assert: 3 envelopes were replayed (seq=3, 4, 5)
    assert count == 3, f"Expected 3 replayed envelopes, got {count}"
    assert last_seq == 5, f"Expected last_seq 5, got {last_seq}"

    # Assert: envelopes in the queue are exactly seq=3, 4, 5 (no duplicate, no skip)
    replayed_envelopes = []
    while not conn.queue.empty():
        replayed_envelopes.append(conn.queue.get_nowait())

    assert len(replayed_envelopes) == 3, (
        f"Expected 3 envelopes in queue, got {len(replayed_envelopes)}"
    )

    # Assert: seqs are exactly [3, 4, 5]
    seqs = [env["seq"] for env in replayed_envelopes]
    assert seqs == [3, 4, 5], (
        f"Expected seqs [3, 4, 5] (no duplicate, no skip), got {seqs}. "
        f"replay violation."
    )


@pytest.mark.asyncio
async def test_reconnect_cross_tenant_denial() -> None:
    """Prove cross-tenant reconnect (different user_id) replays zero events."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    owner_id = "test-owner"
    project_id = "test-proj"
    other_user_id = "other-user"
    store = get_platform().metadata_store

    # Seed the durable log with 2 events
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    for seq in range(1, 3):
        durable_event = {
            "stream": f"job:{job_id}",
            "seq": seq,
            "kind": "progress",
            "data": {"event": "progress", "case": seq},
        }
        await store.append_job_event(job_id, seq, durable_event, ttl=ttl)

    # Simulate a reconnect from a DIFFERENT user
    from app.services.ws_hub import WsConnection
    from app.api.ws import _replay_job_stream_from_durable_log
    from unittest.mock import MagicMock

    mock_websocket = MagicMock()
    conn = WsConnection(
        connection_id=f"conn-{uuid.uuid4().hex[:8]}",
        user_id=other_user_id,  # NOT the owner
        websocket=mock_websocket,
    )

    # Mock job lookup via get_job_tracker (the correct path)
    # and project lookup via get_project_manager
    from app.services.jobs import Job
    with patch("app.services.jobs.get_job_tracker") as mock_tracker:
        with patch("app.main.get_project_manager") as mock_pm:
            # Create a real Job instance owned by owner_id
            test_job = Job(
                job_id=job_id,
                project_id=project_id,
                type="eval",
                state="running",
                config={},
            )
            mock_tracker.return_value.get_job.return_value = test_job

            mock_project = type("Project", (), {"owner_id": owner_id})()
            mock_pm.return_value.get_project.return_value = mock_project

            # Attempt replay
            count, _ = await _replay_job_stream_from_durable_log(
                conn, f"job:{job_id}", since_seq=0
            )

    # Assert: zero envelopes replayed (cross-tenant denial)
    assert count == 0, (
        f"Expected 0 replayed envelopes (cross-tenant denial), got {count}"
    )
    assert conn.queue.empty(), (
        "Queue should be empty after cross-tenant denial"
    )


@pytest.mark.asyncio
async def test_poller_progress_only_does_not_fire_completion():
    """fire_completion=False: poller republishes progress + detects terminal but fires NOTHING.

    On the optimization path the loop owns iteration completion (the
    optimization orchestrator starts this poller with fire_completion=False),
    so the progress bridge runs progress-only. It must still republish envelopes
    and still stop cleanly on the terminal `results` signal, but it must NOT
    claim or fire the completion turn (a claiming poller would double-fire).
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    import time
    ttl = int(time.time()) + (7 * 24 * 3600)

    # Seed progress + a terminal `results` event.
    await store.append_job_event(job_id, 1, {
        "stream": f"job:{job_id}", "seq": 1, "kind": "progress",
        "data": {"event": "progress", "case": 1, "total": 2},
    }, ttl=ttl)
    await store.append_job_event(job_id, 2, {
        "stream": f"job:{job_id}", "seq": 2, "kind": "progress",
        "data": {"event": "results", "file": "eval_results_v1.json"},
    }, ttl=ttl)

    published = []
    claim_calls = 0
    turn_calls = 0

    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        async def capture(user_id, envelope):
            published.append(envelope)
        mock_hub.return_value.publish_to_user = AsyncMock(side_effect=capture)

        with patch("app.services.job_progress_republisher.request_completion_turn") as mock_turn:
            async def track_turn(*a, **k):
                nonlocal turn_calls
                turn_calls += 1
            mock_turn.side_effect = track_turn

            async def mock_claim(*a, **k):
                nonlocal claim_calls
                claim_calls += 1
                return True

            with patch.object(store, "claim_completion_fired", side_effect=mock_claim):
                task = asyncio.create_task(
                    poll_and_republish_progress(
                        job_id, owner_id, project_id, poll_interval=0.1,
                        fire_completion=False,
                    ),
                    name=f"test-poller-{job_id[:8]}",
                )
                await asyncio.sleep(0.3)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # Progress still republished (both events).
    assert len(published) == 2
    # But NO completion claim/fire — the loop owns completion.
    assert claim_calls == 0, "progress-only poller must not claim completion"
    assert turn_calls == 0, "progress-only poller must not fire the completion turn"


@pytest.mark.asyncio
async def test_poller_fires_completion_on_terminal_event_not_state():
    """The poller fires completion off the event-log terminal signal.

    The worker persists all terminal EVENTS durably but never persists terminal STATE.
    The poller already reads every event each cycle — when it observes the harness's
    whole-eval terminal signal (data.event="results" from runner.py save_results)
    in the log, it must fire the completion turn (claim + rebuild Job + request_completion_turn),
    WITHOUT depending on job_record.state being terminal.

    This test seeds a terminal event in the durable log while leaving job_record.state
    NON-terminal (e.g. "queued") to prove completion fires off the EVENT LOG, not off state.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    import time
    ttl = int(time.time()) + (7 * 24 * 3600)

    # Seed progress events + terminal event in the durable log
    for seq in range(1, 3):
        await store.append_job_event(job_id, seq, {
            "stream": f"job:{job_id}",
            "seq": seq,
            "kind": "progress",
            "data": {"event": "progress", "case": seq, "total": 3}
        }, ttl=ttl)

    # Terminal event (data.event="results" = whole-eval terminal signal from runner.py)
    await store.append_job_event(job_id, 3, {
        "stream": f"job:{job_id}",
        "seq": 3,
        "kind": "progress",
        "data": {"event": "results", "file": "eval_results_baseline_20260101_000000.json"}
    }, ttl=ttl)

    # Create a durable job record with NON-terminal state (proves completion triggers
    # from the event log, not from job_record.state). The result must be a list for an eval.
    await store.put_job({
        "user_id": owner_id,
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": "queued",  # NON-terminal: completion must fire off the event log alone
        "progress": {},
        "created_at": "2026-07-10T00:00:00Z",
        "result": [{"model_id": "test-model", "results": [{"success": True} for _ in range(30)]}],
    })

    # Track completion claim + fire
    claim_fired = False
    turn_requested = False

    # The manual won-claim path materializes the output/ file from S3.
    # Patch it to return a known path + file-sourced result so the fire proceeds.
    import tempfile
    tmp_results = Path(tempfile.mkdtemp()) / "eval_results_baseline_20260101_000000.json"
    tmp_results.write_text('[{"model_id": "test-model", "results": [{"success": true}]}]')

    async def fake_materialize(proj, jid, run_filename):
        assert run_filename == "eval_results_baseline_20260101_000000.json"
        return tmp_results

    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        mock_hub.return_value.publish_to_user = AsyncMock()

        with patch("app.services.job_progress_republisher.request_completion_turn") as mock_turn, \
             patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
             patch("app.services.jobs.eval_executor._dispatch_run_report", new=AsyncMock()), \
             patch("app.services.jobs.eval_executor._dispatch_exec_if_winner_changed", new=AsyncMock()):
            async def track_turn(*args, **kwargs):
                nonlocal turn_requested
                turn_requested = True
            mock_turn.side_effect = track_turn

            # Mock claim_completion_fired to return True (winner)
            async def mock_claim(*args, **kwargs):
                nonlocal claim_fired
                claim_fired = True
                return True

            with patch.object(store, "claim_completion_fired", side_effect=mock_claim):
                # Start the poller
                task = asyncio.create_task(
                    poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
                    name=f"test-poller-{job_id[:8]}"
                )

                # Let it run one or two cycles to detect terminal
                await asyncio.sleep(0.25)

                # Cancel
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # Completion fires off the event-log terminal signal, from the file, while
    # job_record.state is still non-terminal ("queued").
    assert claim_fired, (
        "claim_completion_fired should be called when terminal event 'results' is in the log, "
        "even when job_record.state is non-terminal"
    )
    assert turn_requested, (
        "request_completion_turn should be called after winning the claim, "
        "even when job_record.state is non-terminal"
    )


# ========================================================================
# manual republisher materializes the file + dispatches report
# ========================================================================


async def _seed_terminal_log(store, job_id: str, run_filename: str) -> None:
    """Seed a durable event log ending in a terminal `results` event."""
    import time
    ttl = int(time.time()) + (7 * 24 * 3600)
    await store.append_job_event(job_id, 1, {
        "stream": f"job:{job_id}", "seq": 1, "kind": "progress",
        "data": {"event": "progress", "case": 1, "total": 2},
    }, ttl=ttl)
    await store.append_job_event(job_id, 2, {
        "stream": f"job:{job_id}", "seq": 2, "kind": "progress",
        "data": {"event": "results", "file": run_filename},
    }, ttl=ttl)


@pytest.mark.asyncio
async def test_manual_terminal_materializes_file_and_dispatches_report(tmp_path):
    """Won claim → materialize with real filename, fire from FILE result, dispatch report."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    run_filename = "eval_results_baseline_20260102.json"
    store = get_platform().metadata_store

    await _seed_terminal_log(store, job_id, run_filename)
    # Durable record: EMPTY result (proves fire sources the file, not DDB).
    await store.put_job({
        "user_id": owner_id, "project_id": project_id, "job_id": job_id,
        "type": "eval", "state": "queued", "progress": {},
        "created_at": "2026-07-10T00:00:00Z", "result": [],
    })

    file_result = [{"model_id": "m1", "results": [{"success": True}]}]
    materialized = tmp_path / run_filename
    import json as _json
    materialized.write_text(_json.dumps(file_result))

    materialize_calls = []

    async def fake_materialize(proj, jid, fname):
        materialize_calls.append((proj, jid, fname))
        return materialized

    fired = {}
    report_calls = []

    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        mock_hub.return_value.publish_to_user = AsyncMock()
        with patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
             patch("app.services.job_progress_republisher.request_completion_turn") as mock_turn, \
             patch("app.services.jobs.eval_executor._dispatch_run_report") as mock_report, \
             patch("app.services.jobs.eval_executor._dispatch_exec_if_winner_changed", new=AsyncMock()), \
             patch("app.services.project_paths.resolve_project_dir", return_value=tmp_path), \
             patch.object(store, "claim_completion_fired", new=AsyncMock(return_value=True)):

            async def track_turn(proj, job):
                fired["project_id"] = proj
                fired["result"] = job.result
                fired["output_file"] = job.output_file
            mock_turn.side_effect = track_turn

            async def track_report(*args, **kwargs):
                report_calls.append(args)
            mock_report.side_effect = track_report

            task = asyncio.create_task(
                poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
                name=f"test-poller-{job_id[:8]}",
            )
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # materialize awaited with the real filename.
    assert materialize_calls == [(project_id, job_id, run_filename)]
    # Completion turn fired from the FILE result + real output_file.
    assert fired.get("result") == file_result
    assert fired.get("output_file") == run_filename
    # Report dispatched with the real filename (not the job id).
    assert len(report_calls) == 1
    assert report_calls[0][2] == run_filename


@pytest.mark.asyncio
async def test_manual_terminal_materialization_unavailable_skips_fire(tmp_path):
    """RunOutputNotAvailable → no fire, no report, loop does not crash."""
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    await _seed_terminal_log(store, job_id, "eval_results_v1.json")
    await store.put_job({
        "user_id": owner_id, "project_id": project_id, "job_id": job_id,
        "type": "eval", "state": "queued", "progress": {},
        "created_at": "2026-07-10T00:00:00Z", "result": [],
    })

    from app.services.run_output_snapshot import RunOutputNotAvailable

    async def fake_materialize(*a, **k):
        raise RunOutputNotAvailable("never landed")

    turn_calls = 0
    report_calls = 0

    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        mock_hub.return_value.publish_to_user = AsyncMock()
        with patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
             patch("app.services.job_progress_republisher.request_completion_turn") as mock_turn, \
             patch("app.services.jobs.eval_executor._dispatch_run_report") as mock_report, \
             patch.object(store, "claim_completion_fired", new=AsyncMock(return_value=True)):

            async def track_turn(*a, **k):
                nonlocal turn_calls
                turn_calls += 1
            mock_turn.side_effect = track_turn

            async def track_report(*a, **k):
                nonlocal report_calls
                report_calls += 1
            mock_report.side_effect = track_report

            task = asyncio.create_task(
                poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
                name=f"test-poller-{job_id[:8]}",
            )
            await asyncio.sleep(0.3)
            done_before_cancel = task.done()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # No fire, no report — the reconciler is the safety net.
    assert turn_calls == 0
    assert report_calls == 0
    # The loop broke cleanly on the terminal signal (a crash would have raised
    # from ``await task`` above).
    assert done_before_cancel, "the poller must stop on its own after the terminal signal"


@pytest.mark.asyncio
async def test_manual_terminal_lost_claim_does_not_fire(tmp_path):
    """Lost claim → no fire, no report (exactly-once preserved).

    Note on what this test deliberately does NOT assert: materialization now
    happens BEFORE the claim, so a poller that loses the race does perform one
    wasted S3 download. That order is required for correctness — the claim is
    one-way and the reconciler skips claimed jobs, so claiming before the fire
    is guaranteed permanently loses the completion turn whenever the output
    isn't materializable yet. One redundant download on a rare cross-process
    race is the correct price for a completion turn that can always be
    recovered. The load-bearing invariant is exactly-once FIRING, asserted below.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store

    await _seed_terminal_log(store, job_id, "eval_results_v1.json")

    turn_calls = 0
    report_calls = 0

    async def fake_materialize(*a, **k):
        return "/tmp/eval_results_v1.json"

    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub:
        mock_hub.return_value.publish_to_user = AsyncMock()
        with patch("app.services.run_output_snapshot.materialize_detached_run", side_effect=fake_materialize), \
             patch("app.services.run_output_snapshot.load_materialized_results", return_value=[{"model": "m", "cases": []}]), \
             patch("app.services.job_progress_republisher.request_completion_turn") as mock_turn, \
             patch("app.services.jobs.eval_executor._dispatch_run_report") as mock_report, \
             patch.object(store, "claim_completion_fired", new=AsyncMock(return_value=False)):

            async def track_turn(*a, **k):
                nonlocal turn_calls
                turn_calls += 1
            mock_turn.side_effect = track_turn

            async def track_report(*a, **k):
                nonlocal report_calls
                report_calls += 1
            mock_report.side_effect = track_report

            task = asyncio.create_task(
                poll_and_republish_progress(job_id, owner_id, project_id, poll_interval=0.1),
                name=f"test-poller-{job_id[:8]}",
            )
            await asyncio.sleep(0.3)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert turn_calls == 0, "lost claim must not fire the completion turn"
    assert report_calls == 0, "lost claim must not dispatch the report"


# ========================================================================
# an eval that ends without results — failed or cancelled
# ========================================================================


async def _run_poller_against_record(state: str, *, fire_completion: bool = True):
    """Seed a job record in ``state`` with no ``results`` event and run the poller.

    Returns ``(poller_task, completion_turn_mock, claim_mock)``. The task is
    left for the caller to inspect: a poller that noticed the end has returned on
    its own, one that did not is still running and is cancelled here.
    """
    job_id = f"test-job-{uuid.uuid4().hex[:8]}"
    project_id = "test-project"
    owner_id = "test-owner"
    store = get_platform().metadata_store
    await store.put_job({
        "user_id": owner_id,
        "project_id": project_id,
        "job_id": job_id,
        "type": "eval",
        "state": state,
        "config": {"prompt_version": "v2", "source": "manual"},
        # What ``fail_job`` writes, and what the durable record persists.
        "progress": {"message": "Failed: Eval completed but its results could not be saved."},
        "created_at": "2026-09-24T00:00:00Z",
    })

    claim = AsyncMock(return_value=True)
    with patch("app.services.job_progress_republisher.get_ws_hub") as mock_hub, \
         patch("app.services.job_progress_republisher.request_completion_turn",
               new=AsyncMock()) as mock_turn, \
         patch.object(store, "claim_completion_fired", new=claim):
        mock_hub.return_value.publish_to_user = AsyncMock()
        task = asyncio.create_task(
            poll_and_republish_progress(
                job_id, owner_id, project_id, poll_interval=0.05,
                fire_completion=fire_completion,
            ),
            name=f"test-poller-{job_id[:8]}",
        )
        await asyncio.sleep(0.3)
        ended_on_its_own = task.done()
        if not ended_on_its_own:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    return ended_on_its_own, mock_turn, claim


@pytest.mark.asyncio
async def test_a_failed_manual_eval_reaches_chat_and_ends_the_poll():
    """A failed eval never emits ``results``; its record's state is the signal.

    Without this the poller polled a failed eval until the coordinator
    restarted, and the failure never reached chat — only the toast.
    """
    ended, mock_turn, claim = await _run_poller_against_record("failed")

    assert ended, "the poller must stop once the job has failed"
    claim.assert_awaited_once()
    mock_turn.assert_awaited_once()
    job = mock_turn.await_args.args[1]
    assert job.state == "failed"
    assert job.progress["message"].startswith("Failed: ")


@pytest.mark.asyncio
async def test_a_cancelled_eval_ends_the_poll_without_a_turn():
    """A cancel is announced by whoever cancelled — the UI route or the agent."""
    ended, mock_turn, claim = await _run_poller_against_record("cancelled")

    assert ended, "the poller must stop once the job is cancelled"
    claim.assert_not_awaited()
    mock_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_loop_owned_failed_eval_ends_the_poll_without_a_turn():
    """The optimization loop reports its own iteration failures."""
    ended, mock_turn, claim = await _run_poller_against_record(
        "failed", fire_completion=False
    )

    assert ended
    claim.assert_not_awaited()
    mock_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_running_eval_keeps_the_poll_going():
    """Only an ended job stops the poll — a running one is still being watched."""
    ended, mock_turn, _claim = await _run_poller_against_record("running")

    assert not ended
    mock_turn.assert_not_awaited()
