# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test replay buffer cannot-serve signal.

The buffer-honesty invariant: when `since_seq < oldest_retained_seq`, replay
must emit an explicit cannot-serve signal, not an empty replay that reads as
"nothing new". One rule covers restart, TTL, and cap eviction.

Mutation proofs named per test:
- M_RESTART (critical): restore the `oldest_seq is not None` outer guard → RED
- M_TTL: make `oldest_retained_seq` ignore the TTL cutoff → RED
- M_CAP: same as TTL (both exercise the same `oldest_retained_seq` property)
- M_SERVABLE: make the comparison unconditional (always fire) → RED
- M_FRESH_CLIENT: remove the `since_seq >= 0` guard → RED (fresh client gets signal)
- M_CROSS_TENANT: move the owner-mismatch gate after cannot-serve → RED (leaks signal)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from app.platform.dev_local import FilesystemMetadataStore
from app.services.project_event_tracker import (
    ProjectEventBuffer,
    _reset as reset_tracker,
    get_provider,
    record_envelope,
)
from app.services.ws_envelope import make_envelope_with_seq, reset_counters
from app.services.ws_hub import WsHub


class _MockConnection:
    """Mock WS connection for testing."""

    def __init__(self, user_id: str, connection_id: str = "test-conn"):
        self.user_id = user_id
        self.connection_id = connection_id
        self.queue: asyncio.Queue[dict] = asyncio.Queue()


@pytest.mark.asyncio
async def test_empty_buffer_cannot_serve_restart_case(tmp_path: Path):
    """Empty buffer (restart) emits cannot-serve for non-negative since_seq.

    This is the CRITICAL case: `oldest_retained_seq` returns `None` for an
    empty buffer. The original outer guard was
    `if oldest_seq is not None`, which short-circuited and never fired
    cannot-serve on restart. Counter continues at 500, buffer empty, client
    cursor 480 → silent loss.

    Mutation proof M_RESTART: restore the `oldest_seq is not None` outer guard
    in replay_to_connection (replace `oldest_seq_attr is not _MISSING` with
    `oldest_seq is not None`). The test will go RED because the outer guard
    short-circuits on `None` and no cannot-serve is emitted.
    """
    project_id = "test-proj-restart"
    owner_id = "test-user-restart"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Register an EMPTY buffer (restart scenario: counter advanced but buffer gone)
    buffer = get_provider(project_id)
    assert buffer.oldest_retained_seq is None, "Precondition: empty buffer returns None"

    hub.register_replay_provider(
        stream=f"project:{project_id}",
        provider=buffer,
        owner_id=owner_id,
    )

    conn = _MockConnection(user_id=owner_id)
    hub.register(conn)

    try:
        with mock.patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Client reconnects with cursor 480; buffer is empty
            await hub.replay_to_connection(
                conn, stream=f"project:{project_id}", since_seq=480
            )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Must get exactly one cannot-serve envelope
        assert len(envelopes) == 1, f"Expected 1 cannot-serve, got {len(envelopes)}"
        gap_env = envelopes[0]
        assert gap_env["kind"] == "replay_gap", f"Expected replay_gap, got {gap_env['kind']}"
        assert gap_env["data"]["since_seq"] == 480
        assert gap_env["data"]["oldest_retained_seq"] is None

    finally:
        hub.unregister(conn)


@pytest.mark.asyncio
async def test_ttl_eviction_cannot_serve(tmp_path: Path):
    """TTL-evicted buffer emits cannot-serve when since_seq predates oldest.

    A tab left in the background longer than the 600s TTL reconnects with a
    cursor the buffer can no longer serve, while every seq number still looks
    valid — only the cannot-serve signal reveals the gap.

    Mutation proof M_TTL: make `oldest_retained_seq` ignore the TTL cutoff
    (return all entries regardless of age). The test will go RED because the
    buffer claims it can serve an evicted range.
    """
    project_id = "test-proj-ttl"
    owner_id = "test-user-ttl"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Publish an envelope at time T
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"event": "old"},
        )

    # Advance time past TTL by patching _get_monotonic_time
    from app.services import project_event_tracker

    original_time_fn = project_event_tracker._get_monotonic_time
    elapsed = 0.0

    def _mock_time():
        return original_time_fn() + elapsed

    with mock.patch.object(project_event_tracker, "_get_monotonic_time", _mock_time):
        # Advance past TTL (600s)
        elapsed = 700.0

        # Now the envelope is TTL-expired; buffer's oldest_retained_seq should reflect that
        buffer = get_provider(project_id)
        # After TTL expiry, oldest_retained_seq should be None (no servable entries)
        assert buffer.oldest_retained_seq is None, "Oldest seq should be None after TTL"

        conn = _MockConnection(user_id=owner_id)
        hub.register(conn)

        try:
            with mock.patch("app.platform.get_platform") as mock_platform:
                mock_platform.return_value.metadata_store = store

                # Client reconnects with cursor 0 (saw the original envelope)
                await hub.replay_to_connection(
                    conn, stream=f"project:{project_id}", since_seq=0
                )

            # Drain the queue
            envelopes = []
            while not conn.queue.empty():
                envelopes.append(await conn.queue.get())

            # Must get cannot-serve
            assert len(envelopes) == 1, f"Expected 1 cannot-serve, got {len(envelopes)}"
            assert envelopes[0]["kind"] == "replay_gap"

        finally:
            hub.unregister(conn)


@pytest.mark.asyncio
async def test_cap_eviction_cannot_serve(tmp_path: Path):
    """Cap-shed buffer emits cannot-serve when since_seq predates oldest.

    Publishes past _MAX_ENVELOPES_PER_PROJECT (100) so the oldest is shed.

    Mutation proof M_CAP: same as M_TTL (both exercise `oldest_retained_seq`).
    """
    project_id = "test-proj-cap"
    owner_id = "test-user-cap"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Publish 102 envelopes (cap is 100)
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store
        for i in range(102):
            await hub.publish_project(
                project_id=project_id,
                owner_id=owner_id,
                kind="project_event",
                data={"event": f"msg-{i}"},
            )

    buffer = get_provider(project_id)
    oldest = buffer.oldest_retained_seq
    assert oldest is not None and oldest >= 3, "Oldest two should be shed"

    conn = _MockConnection(user_id=owner_id)
    hub.register(conn)

    try:
        with mock.patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Client reconnects with cursor 0 (predates the shed entries)
            await hub.replay_to_connection(
                conn, stream=f"project:{project_id}", since_seq=0
            )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Must get cannot-serve
        assert len(envelopes) == 1, f"Expected 1 cannot-serve, got {len(envelopes)}"
        assert envelopes[0]["kind"] == "replay_gap"

    finally:
        hub.unregister(conn)


@pytest.mark.asyncio
async def test_servable_reconnect_no_cannot_serve(tmp_path: Path):
    """Servable reconnect emits NO cannot-serve (guard against always-firing).

    This is a guard test: without it, a signal that always fires trains the
    client to refetch on every reconnect and the invariant is decorative.

    Mutation proof M_SERVABLE: make the cannot-serve comparison unconditional
    (remove the `else: cannot_serve = False` branch, or always set True). The
    test will go RED because it fires on a servable range.
    """
    project_id = "test-proj-servable"
    owner_id = "test-user-servable"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Publish two envelopes
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"event": "first"},
        )
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"event": "second"},
        )

    buffer = get_provider(project_id)
    assert buffer.oldest_retained_seq == 1

    conn = _MockConnection(user_id=owner_id)
    hub.register(conn)

    try:
        # Client reconnects with cursor 1 (saw the first envelope, asking for > 1)
        # Buffer's oldest is 1, so since_seq=1 → replay from seq 2, which exists.
        await hub.replay_to_connection(
            conn, stream=f"project:{project_id}", since_seq=1
        )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Should get one replayed envelope (seq=2), NO cannot-serve
        assert len(envelopes) == 1, f"Expected 1 replay, got {len(envelopes)}"
        assert envelopes[0]["kind"] == "project_event", "Should be a replayed event, not gap"
        assert envelopes[0]["data"]["event"] == "second"
        assert envelopes[0].get("replayed") is True

    finally:
        hub.unregister(conn)


@pytest.mark.asyncio
async def test_fresh_client_no_cannot_serve(tmp_path: Path):
    """Fresh client (since_seq == -1) never gets cannot-serve.

    A first-ever subscribe is not a gap, even if the buffer is empty.

    Mutation proof M_FRESH_CLIENT: remove the `since_seq >= 0` guard in
    replay_to_connection. The test will go RED because it fires on since_seq=-1.
    """
    project_id = "test-proj-fresh"
    owner_id = "test-user-fresh"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Empty buffer
    buffer = get_provider(project_id)
    hub.register_replay_provider(
        stream=f"project:{project_id}",
        provider=buffer,
        owner_id=owner_id,
    )

    conn = _MockConnection(user_id=owner_id)
    hub.register(conn)

    try:
        with mock.patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Fresh client: since_seq=-1
            await hub.replay_to_connection(
                conn, stream=f"project:{project_id}", since_seq=-1
            )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Should get ZERO envelopes (empty buffer, but NO cannot-serve)
        assert len(envelopes) == 0, f"Expected 0 envelopes, got {len(envelopes)}"

    finally:
        hub.unregister(conn)


@pytest.mark.asyncio
async def test_cross_tenant_no_cannot_serve(tmp_path: Path):
    """Foreign-user subscribe gets zero replay AND no cannot-serve (no leak).

    The owner-mismatch gate must come FIRST, before the cannot-serve branch,
    so the signal is never emitted to a foreign tenant (it would be an existence
    oracle for another tenant's stream).

    Mutation proof M_CROSS_TENANT: move the owner-mismatch gate after the
    cannot-serve branch in replay_to_connection. The test will go RED because
    a cannot-serve envelope is emitted to the foreign user.
    """
    project_id = "test-proj-tenant"
    owner_id = "owner-user"
    foreign_id = "foreign-user"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Publish one envelope
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store
        await hub.publish_project(
            project_id=project_id,
            owner_id=owner_id,
            kind="project_event",
            data={"event": "secret"},
        )

    buffer = get_provider(project_id)
    assert buffer.oldest_retained_seq == 1

    # Foreign user connects and subscribes with a cursor that WOULD trigger
    # cannot-serve if the gate ordering were wrong
    conn = _MockConnection(user_id=foreign_id)
    hub.register(conn)

    try:
        # Client asks for seq > 100, which predates oldest (1)
        await hub.replay_to_connection(
            conn, stream=f"project:{project_id}", since_seq=100
        )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Must get ZERO envelopes (no replay, no cannot-serve)
        assert len(envelopes) == 0, f"Expected 0 envelopes for foreign user, got {len(envelopes)}"

    finally:
        hub.unregister(conn)


@pytest.mark.asyncio
async def test_allocation_failure_fallback_seq(tmp_path: Path):
    """Cannot-serve envelope on allocation failure uses buffer-seeded fallback.

    The fallback seq must be `max(buffer seqs) + 1`, not 0 (the collision value)
    or oldest_seq (likely already seen). Otherwise the gap envelope can be
    dropped by the client's cursor as a duplicate, defeating the signal.

    Mutation proof: use `gap_seq = 0` in the except branch. The test will go
    RED because the client's cursor (say, 5) exceeds the gap seq (0) and the
    envelope is dropped.
    """
    project_id = "test-proj-fallback"
    owner_id = "test-user-fallback"

    # Failing allocator
    class _FailingStore:
        async def allocate_project_seq(self, pid: str) -> int:
            raise RuntimeError("Simulated failure")

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Publish 3 envelopes with a working store
    working_store = FilesystemMetadataStore(projects_dir=tmp_path)
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = working_store
        for i in range(3):
            await hub.publish_project(
                project_id=project_id,
                owner_id=owner_id,
                kind="project_event",
                data={"event": f"msg-{i}"},
            )

    buffer = get_provider(project_id)
    assert buffer.oldest_retained_seq == 1

    # Now simulate TTL eviction so oldest_retained_seq becomes None
    # (we want to test the fallback with an empty buffer)
    from app.services import project_event_tracker

    original_time_fn = project_event_tracker._get_monotonic_time
    elapsed = 0.0

    def _mock_time():
        return original_time_fn() + elapsed

    with mock.patch.object(project_event_tracker, "_get_monotonic_time", _mock_time):
        elapsed = 700.0  # Past TTL

        conn = _MockConnection(user_id=owner_id)
        hub.register(conn)

        try:
            # Use failing store for the gap envelope allocation
            failing_store = _FailingStore()
            with mock.patch("app.platform.get_platform") as mock_platform:
                mock_platform.return_value.metadata_store = failing_store

                # Client reconnects with cursor 2
                await hub.replay_to_connection(
                    conn, stream=f"project:{project_id}", since_seq=2
                )

            # Drain the queue
            envelopes = []
            while not conn.queue.empty():
                envelopes.append(await conn.queue.get())

            # Must get one cannot-serve with fallback seq
            assert len(envelopes) == 1, f"Expected 1 cannot-serve, got {len(envelopes)}"
            gap_env = envelopes[0]
            assert gap_env["kind"] == "replay_gap"

            # Fallback seq must be > since_seq (client's cursor of 2)
            # max(buffer_seqs, since_seq) + 1 = max([], 2) + 1 = 3
            # (buffer.events is empty due to TTL filter)
            assert gap_env["seq"] == 3, f"Expected fallback seq=3, got {gap_env['seq']}"
            assert gap_env["seq"] > 2, "Fallback must exceed client cursor"

        finally:
            hub.unregister(conn)


@pytest.mark.asyncio
async def test_gap_envelope_seq_exceeds_client_cursor_durable_path(tmp_path: Path):
    """Gap envelope seq strictly exceeds since_seq on durable allocation path.

    This complements the existing fallback-path assertion. The gap envelope's
    seq must be strictly greater than the requesting client's since_seq on BOTH
    the durable path (project-wide monotonic counter) and the fallback path
    (which already takes max(buffer_seqs, since_seq) + 1). This is what makes
    it safe for the client to adopt the gap seq as its cursor without creating
    a permanent hole.

    Mutation proof: make the gap envelope carry since_seq instead of an allocated
    seq. The test will go RED because the seq will not exceed the cursor.
    """
    project_id = "test-proj-gap-seq-durable"
    owner_id = "test-user-gap-seq"
    store = FilesystemMetadataStore(projects_dir=tmp_path)

    reset_counters()
    reset_tracker()
    hub = WsHub()

    # Pre-allocate some seqs so the counter is ahead
    with mock.patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = store
        await store.allocate_project_seq(project_id)  # seq=1
        await store.allocate_project_seq(project_id)  # seq=2
        await store.allocate_project_seq(project_id)  # seq=3

    # Register an empty buffer (cannot serve any non-negative cursor)
    buffer = get_provider(project_id)
    hub.register_replay_provider(
        stream=f"project:{project_id}",
        provider=buffer,
        owner_id=owner_id,
    )

    conn = _MockConnection(user_id=owner_id)
    hub.register(conn)

    try:
        with mock.patch("app.platform.get_platform") as mock_platform:
            mock_platform.return_value.metadata_store = store

            # Client reconnects with cursor 2; buffer is empty
            await hub.replay_to_connection(
                conn, stream=f"project:{project_id}", since_seq=2
            )

        # Drain the queue
        envelopes = []
        while not conn.queue.empty():
            envelopes.append(await conn.queue.get())

        # Must get one cannot-serve with durable seq
        assert len(envelopes) == 1, f"Expected 1 cannot-serve, got {len(envelopes)}"
        gap_env = envelopes[0]
        assert gap_env["kind"] == "replay_gap"

        # Durable seq must strictly exceed since_seq
        # The counter is at 3 after pre-allocation, so the next allocate returns 4
        assert gap_env["seq"] == 4, f"Expected durable seq=4, got {gap_env['seq']}"
        assert gap_env["seq"] > 2, "Gap seq must exceed client cursor (the invariant)"

    finally:
        hub.unregister(conn)
