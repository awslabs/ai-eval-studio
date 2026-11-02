# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for project-stream replay provider.

Verifies that project-scoped events published with no subscriber are delivered
on the next subscribe, tenant-guarded, with bounded memory.
"""

import pytest
from app.services import ws_hub
from app.services.ws_envelope import make_envelope
from app.services import project_event_tracker


@pytest.fixture(autouse=True)
def _reset_buffers():
    """Clear project event buffers before each test."""
    project_event_tracker._reset()
    yield
    project_event_tracker._reset()


@pytest.fixture
def hub():
    """Create a fresh WsHub instance for each test."""
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


@pytest.mark.asyncio
async def test_publish_with_no_subscriber_then_subscribe_replays(hub, mock_connection):
    """Publish with NO subscriber, then subscribe with since_seq=-1 → envelope arrives.

    Without a replay provider, the envelope would be lost.
    """
    project_id = "proj-abc"
    owner_id = "user-owner"
    event = {"resource": "optimization", "type": "started"}

    # Publish with NO subscriber (no connection registered yet)
    await hub.publish_project(project_id, owner_id, event)

    # Now connect and subscribe with since_seq=-1 (replay everything)
    conn = mock_connection(owner_id)
    replayed = await hub.replay_to_connection(
        conn, f"project:{project_id}", since_seq=-1
    )

    # The envelope must have been delivered via replay
    assert replayed == 1, "Expected 1 envelope replayed"
    # Verify the envelope is in the queue
    envelope = await conn.queue.get()
    assert envelope["stream"] == f"project:{project_id}"
    assert envelope["kind"] == "project_event"
    assert envelope["data"] == event


@pytest.mark.asyncio
async def test_since_seq_filters_duplicates(hub, mock_connection):
    """since_seq=N replays only seq > N (no duplicates).

    The durable project allocator is 1-based, so three publishes
    are seq 1, 2, 3 — not 0, 1, 2. Asking from the FIRST issued seq is the
    servable case this test is about; asking from 0 would be below the buffer's
    oldest retained seq and correctly draws a cannot-serve signal instead
    (covered by test_replay_cannot_serve.py).
    """
    project_id = "proj-def"
    owner_id = "user-owner"

    # Publish three events. Their seqs come from the DURABLE allocator, which
    # persists across tests in the shared dev store — so they are not 0,1,2 and
    # cannot be hardcoded. Derive the cursor from what was actually issued.
    for i in range(3):
        event = {"resource": "eval", "type": "progress", "iteration": i}
        await hub.publish_project(project_id, owner_id, event)

    from app.services import project_event_tracker as tracker
    issued = [e["seq"] for e in tracker.get_provider(project_id).events]
    assert len(issued) == 3, f"expected 3 buffered envelopes, got {issued}"
    first, second, third = issued

    # Subscribe from the FIRST issued seq → should get the second and third.
    # Asking from below the buffer's oldest retained seq is a different case and
    # correctly draws a cannot-serve signal (see test_replay_cannot_serve.py).
    conn = mock_connection(owner_id)
    replayed = await hub.replay_to_connection(
        conn, f"project:{project_id}", since_seq=first
    )

    assert replayed == 2, f"Expected 2 envelopes after seq {first}, got {replayed}"
    # Verify the seqs
    env1 = await conn.queue.get()
    env2 = await conn.queue.get()
    assert env1["seq"] == second
    assert env2["seq"] == third


@pytest.mark.asyncio
async def test_tenant_guard_rejects_cross_user_replay(hub, mock_connection):
    """A connection whose user_id differs from the project owner replays zero envelopes."""
    project_id = "proj-ghi"
    owner_id = "user-owner"
    attacker_id = "user-attacker"

    # Owner publishes an event
    event = {"resource": "secret", "type": "data"}
    await hub.publish_project(project_id, owner_id, event)

    # Attacker connects and tries to subscribe
    attacker_conn = mock_connection(attacker_id, connection_id="conn-attacker")
    replayed = await hub.replay_to_connection(
        attacker_conn, f"project:{project_id}", since_seq=-1
    )

    # Must replay zero envelopes (tenant guard)
    assert replayed == 0, "Cross-tenant replay must return 0"
    # Queue must be empty
    assert attacker_conn.queue.empty()


@pytest.mark.asyncio
async def test_overflow_sheds_oldest_and_increments_dropped(hub, mock_connection):
    """Overflow sheds oldest and increments events_dropped."""
    project_id = "proj-overflow"
    owner_id = "user-owner"

    # Exceed the per-project cap (100 envelopes)
    cap = project_event_tracker._MAX_ENVELOPES_PER_PROJECT
    for i in range(cap + 10):
        event = {"iteration": i}
        await hub.publish_project(project_id, owner_id, event)

    # Check the buffer state
    buffer = project_event_tracker._registry.get(project_id)
    assert buffer is not None
    assert len(buffer.events) == cap, f"Buffer must hold exactly {cap} envelopes"
    assert buffer.events_dropped == 10, "Expected 10 envelopes dropped"

    # The oldest envelopes (0-9) must be gone; first event should be iteration 10
    first_envelope = buffer.events[0]
    assert first_envelope["data"]["iteration"] == 10, "First envelope should be iteration 10"


@pytest.mark.asyncio
async def test_lru_eviction_past_project_ceiling(hub, mock_connection):
    """LRU eviction past the project ceiling drops the coldest buffer."""
    ceiling = project_event_tracker._MAX_PROJECT_BUFFERS
    owner_id = "user-owner"

    # Create buffers for ceiling + 1 projects
    for i in range(ceiling + 1):
        project_id = f"proj-{i:04d}"
        event = {"project": i}
        await hub.publish_project(project_id, owner_id, event)

    # The oldest buffer (proj-0000) must have been evicted
    buffer_0 = project_event_tracker._registry.get("proj-0000")
    assert buffer_0 is None, "Oldest buffer must be evicted"

    # The newest buffer (proj-<ceiling>) must exist
    buffer_newest = project_event_tracker._registry.get(f"proj-{ceiling:04d}")
    assert buffer_newest is not None, "Newest buffer must exist"

    # Total buffers must be exactly at the ceiling
    assert len(project_event_tracker._registry._buffers) == ceiling


@pytest.mark.asyncio
async def test_lru_eviction_also_unregisters_replay_provider(hub, mock_connection):
    """Eviction must drop the hub registration too, or the ceiling bounds nothing.

    The registry dropping its own reference is NOT sufficient: ``publish_project``
    registers each buffer in the hub's ``_replay_providers``, which holds a STRONG
    reference. Without the unregister on eviction, every "evicted" buffer stays
    alive with its envelopes and ``_MAX_PROJECT_BUFFERS`` bounds nothing — an
    unbounded leak on a long-lived coordinator.

    This asserts the HUB's registration count, not just the registry's. The
    original LRU test checked only the registry, which is precisely why the leak
    survived it: the registry shrank correctly while the hub grew without limit.
    """
    ceiling = project_event_tracker._MAX_PROJECT_BUFFERS
    owner_id = "user-owner"
    overshoot = 3

    for i in range(ceiling + overshoot):
        await hub.publish_project(f"proj-{i:04d}", owner_id, {"project": i})

    project_registrations = [
        s for s in hub._replay_providers if s.startswith("project:")
    ]
    assert len(project_registrations) == ceiling, (
        f"hub holds {len(project_registrations)} project registrations for "
        f"{ceiling} live buffers — each extra entry pins an evicted buffer's "
        f"envelopes in memory, so the ceiling bounds nothing"
    )

    # The evicted projects must have no registration left behind.
    for i in range(overshoot):
        assert f"project:proj-{i:04d}" not in hub._replay_providers

    # And replaying an evicted stream yields nothing rather than stale envelopes.
    conn = mock_connection(owner_id)
    replayed = await hub.replay_to_connection(
        conn, "project:proj-0000", since_seq=-1
    )
    assert replayed == 0


@pytest.mark.asyncio
async def test_mutation_proof_publish_with_no_subscriber_fails_without_fix(hub, mock_connection):
    """Mutation proof: a publish with no subscriber must still replay.

    It fails when the buffer is not wired in and passes when it is.

    Mutation: Comment out `record_envelope(project_id, envelope)` in ws_hub.py.
    Expected: This test fails (replayed == 0).
    """
    project_id = "proj-mutation"
    owner_id = "user-owner"
    event = {"type": "test"}

    # Publish with no subscriber
    await hub.publish_project(project_id, owner_id, event)

    # Subscribe and replay
    conn = mock_connection(owner_id)
    replayed = await hub.replay_to_connection(
        conn, f"project:{project_id}", since_seq=-1
    )

    # Buffer wired in: replayed == 1
    # record_envelope commented out: replayed == 0
    assert replayed == 1, (
        "Mutation proof failed: publish with no subscriber must replay. "
        "If this fails, the buffer is not wired correctly."
    )
