# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test envelope replay marking and TTL enforcement.

Verifies that replayed envelopes are distinguishable from live ones, and that
buffered envelopes older than the TTL are not replayed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import ws_envelope
from app.services import project_event_tracker
from app.services import ws_hub

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(autouse=True)
def _reset_buffers():
    """Clear project event buffers before each test."""
    project_event_tracker._reset()
    yield
    project_event_tracker._reset()


@pytest.fixture
def hub() -> ws_hub.WsHub:
    """Fresh hub with no subscribers."""
    # Reset the singleton instance
    ws_hub.WsHub._instance = None
    return ws_hub.WsHub()


@pytest.fixture
def mock_connection(hub):
    """Create a mock WsConnection for testing replay."""
    def _make_conn(user_id: str, connection_id: str = "conn-test"):
        conn = ws_hub.WsConnection(
            connection_id=connection_id,
            user_id=user_id,
            websocket=None,  # Not needed for replay tests
        )
        hub.register(conn)
        return conn
    return _make_conn


# Test 1: mark_replayed returns a new dict carrying the marker,
# and its input argument is unchanged (copy semantics)
def test_mark_replayed_copy_semantics() -> None:
    """mark_replayed returns a copy with the marker; input is unchanged."""
    ws_envelope.reset_counters()
    # Project streams no longer draw from the in-memory counter,
    # so supply the seq explicitly. This test is about copy semantics, not
    # allocation, so any fixed seq serves.
    original = ws_envelope.make_envelope_with_seq(
        stream="project:test", kind="project_event", data={"type": "test"}, seq=0
    )
    original_id = id(original)

    marked = ws_envelope.mark_replayed(original)

    # Marked envelope has the marker
    assert marked["replayed"] is True
    # Original envelope is unchanged
    assert "replayed" not in original
    # They are different objects
    assert id(marked) != original_id
    # But share the same data reference (shallow copy)
    assert marked["stream"] == original["stream"]
    assert marked["seq"] == original["seq"]
    assert marked["kind"] == original["kind"]
    assert marked["data"] is original["data"]


# Test 2: An envelope delivered to a connection via replay_to_connection
# on a project: stream carries the marker
@pytest.mark.asyncio
async def test_project_stream_replay_marks_envelopes(
    hub: ws_hub.WsHub, mock_connection
) -> None:
    """Replayed project envelopes carry the marker."""
    project_id = "proj-test"
    owner_id = "user-owner"
    event = {"resource": "test", "type": "test_event"}

    # Publish an event (this will buffer it)
    await hub.publish_project(project_id, owner_id, event)

    # Create a connection and replay
    conn = mock_connection(owner_id)
    replayed = await hub.replay_to_connection(
        conn, f"project:{project_id}", since_seq=-1
    )

    # Should have replayed one envelope
    assert replayed == 1
    # Check that the envelope in the queue carries the marker
    envelope = await conn.queue.get()
    assert envelope["replayed"] is True
    assert envelope["kind"] == "project_event"


# Test 3: An envelope delivered by a live publish_project carries no marker key at all
@pytest.mark.asyncio
async def test_live_publish_no_marker(
    hub: ws_hub.WsHub, mock_connection
) -> None:
    """Live published envelopes have no replayed key."""
    project_id = "proj-live"
    owner_id = "user-owner"
    event = {"resource": "test", "type": "live_event"}

    # Subscribe first (so we receive the live publish)
    conn = mock_connection(owner_id)

    # Publish a live event
    await hub.publish_project(project_id, owner_id, event)

    # Check the envelope from the live publish
    envelope = await conn.queue.get()
    # The marker key must be absent, not merely falsy
    assert "replayed" not in envelope
    assert envelope["kind"] == "project_event"


# Test 4: Two successive replays of the same buffered envelope each deliver
# a marked copy while the buffer's own entry stays unmarked
@pytest.mark.asyncio
async def test_replay_preserves_buffer_entry(
    hub: ws_hub.WsHub, mock_connection
) -> None:
    """Multiple replays of the same envelope keep the buffer entry clean."""
    project_id = "proj-multi"
    owner_id = "user-owner"
    event = {"resource": "test", "type": "test_event"}

    # Publish one event (buffers it)
    await hub.publish_project(project_id, owner_id, event)

    # Get the provider to check the buffer
    provider = project_event_tracker.get_provider(project_id)
    buffered = provider.events
    assert len(buffered) == 1
    assert "replayed" not in buffered[0]

    # Replay to two different connections
    conn1 = mock_connection(owner_id, connection_id="conn-1")
    await hub.replay_to_connection(conn1, f"project:{project_id}", since_seq=-1)

    conn2 = mock_connection(owner_id, connection_id="conn-2")
    await hub.replay_to_connection(conn2, f"project:{project_id}", since_seq=-1)

    # Both connections received a marked envelope
    env1 = await conn1.queue.get()
    env2 = await conn2.queue.get()
    assert env1["replayed"] is True
    assert env2["replayed"] is True

    # Buffer entry is still unmarked
    buffered_after = provider.events
    assert len(buffered_after) == 1
    assert "replayed" not in buffered_after[0]


# Test 5: The durable job-stream replay path routes its envelopes through mark_replayed
@pytest.mark.asyncio
async def test_durable_job_replay_marks_envelopes(hub: ws_hub.WsHub, mock_connection) -> None:
    """Job-stream durable replay marks envelopes."""
    from app.api import ws as ws_module

    # Mock the store and job tracker
    mock_store = MagicMock()
    mock_event = {
        "stream": "job:test-job",
        "seq": 1,
        "kind": "progress",
        "data": {"percent": 50},
    }
    mock_store.events_since = AsyncMock(return_value=[mock_event])

    # Mock the job tracker and project manager to satisfy the owner check
    mock_job = MagicMock()
    mock_job.project_id = "proj-test"
    mock_project = MagicMock()
    mock_project.owner_id = "user-owner"

    conn = mock_connection("user-owner")

    with patch.object(ws_module, "mark_replayed", wraps=ws_envelope.mark_replayed) as spy, \
         patch("app.services.jobs.get_job_tracker") as mock_tracker_fn, \
         patch("app.main.get_project_manager") as mock_pm_fn, \
         patch("app.platform.get_platform") as mock_platform_fn:

        mock_tracker = MagicMock()
        mock_tracker.get_job.return_value = mock_job
        mock_tracker_fn.return_value = mock_tracker

        mock_pm = MagicMock()
        mock_pm.get_project.return_value = mock_project
        mock_pm_fn.return_value = mock_pm

        mock_platform = MagicMock()
        mock_platform.metadata_store = mock_store
        mock_platform_fn.return_value = mock_platform

        # Import and call the replay function
        from app.api.ws import _replay_job_stream_from_durable_log

        await _replay_job_stream_from_durable_log(
            conn=conn,
            stream="job:test-job",
            since_seq=0,
        )

        # Verify mark_replayed was called
        assert spy.called
        assert spy.call_count >= 1


# Test 6: The durable turn-stream replay path does the same
@pytest.mark.asyncio
async def test_durable_turn_replay_marks_envelopes(hub: ws_hub.WsHub, mock_connection) -> None:
    """Turn-stream durable replay marks envelopes."""
    from app.api import ws as ws_module

    # Mock the store
    mock_store = MagicMock()
    mock_event = {
        "stream": "turn:test-turn",
        "seq": 1,
        "kind": "text_delta",
        "data": {"delta": "hello"},
    }
    mock_store.turn_events_since = AsyncMock(return_value=[mock_event])
    mock_store.get_turn_metadata = AsyncMock(return_value={"project_id": "proj-test"})

    # Mock the project manager to satisfy the owner check
    mock_project = MagicMock()
    mock_project.owner_id = "user-owner"

    conn = mock_connection("user-owner")

    with patch.object(ws_module, "mark_replayed", wraps=ws_envelope.mark_replayed) as spy, \
         patch("app.main.get_project_manager") as mock_pm_fn, \
         patch("app.platform.get_platform") as mock_platform_fn:

        mock_pm = MagicMock()
        mock_pm.get_project.return_value = mock_project
        mock_pm_fn.return_value = mock_pm

        mock_platform = MagicMock()
        mock_platform.metadata_store = mock_store
        mock_platform_fn.return_value = mock_platform

        # Import and call the replay function
        from app.api.ws import _replay_turn_stream_from_durable_log

        await _replay_turn_stream_from_durable_log(
            conn=conn,
            stream="turn:test-turn",
            since_seq=0,
        )

        # Verify mark_replayed was called
        assert spy.called
        assert spy.call_count >= 1


# Test 7: An envelope recorded longer ago than the TTL is not replayed
def test_expired_envelope_not_replayed() -> None:
    """Envelopes older than the TTL are excluded from replay."""
    initial_time = 100.0
    ttl = 600.0

    # Get a buffer and patch the clock
    provider = project_event_tracker.get_provider("proj-expire")

    # Patch the clock to return a fixed time and record an envelope
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=initial_time):
        envelope = ws_envelope.make_envelope_with_seq(
            stream="project:proj-expire",
            kind="project_event",
            data={"resource": "test", "type": "old_event"},
            seq=0,
        )
        provider.append(envelope)

    # Advance time past the TTL
    expired_time = initial_time + ttl + 1.0
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=expired_time):
        # Read events - should be empty
        events = provider.events
        assert len(events) == 0


# Test 8: An envelope recorded within the TTL is replayed
def test_fresh_envelope_replayed() -> None:
    """Envelopes within the TTL are included in replay."""
    initial_time = 100.0
    ttl = 600.0

    # Get a buffer
    provider = project_event_tracker.get_provider("proj-fresh")

    # Patch the clock to return a fixed time and record an envelope
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=initial_time):
        envelope = ws_envelope.make_envelope_with_seq(
            stream="project:proj-fresh",
            kind="project_event",
            data={"resource": "test", "type": "fresh_event"},
            seq=0,
        )
        provider.append(envelope)

    # Advance time but stay within the TTL
    fresh_time = initial_time + ttl - 10.0
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=fresh_time):
        # Read events - should have the envelope
        events = provider.events
        assert len(events) == 1
        assert events[0]["data"]["type"] == "fresh_event"


# Test 9: TTL is enforced at replay-read time with no intervening publish
def test_ttl_enforced_at_read_time() -> None:
    """TTL filtering works even when no append occurs after expiry."""
    initial_time = 100.0
    ttl = 600.0

    # Get a buffer
    provider = project_event_tracker.get_provider("proj-read-ttl")

    # Record one envelope
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=initial_time):
        envelope = ws_envelope.make_envelope_with_seq(
            stream="project:proj-read-ttl",
            kind="project_event",
            data={"resource": "test", "type": "event"},
            seq=0,
        )
        provider.append(envelope)
        # Verify it's present immediately
        assert len(provider.events) == 1

    # Advance time past the TTL WITHOUT any new append
    expired_time = initial_time + ttl + 1.0
    with patch.object(project_event_tracker, "_get_monotonic_time", return_value=expired_time):
        # Read events - should be empty because TTL is enforced on read
        events = provider.events
        assert len(events) == 0
        # This test distinguishes read-time from append-only enforcement
