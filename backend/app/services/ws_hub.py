# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process WebSocket connection registry with per-user fan-out.

The hub is a process-wide singleton; coordinator code calls
``publish_to_user(user_id, envelope)`` and the hub fans out the
envelope to every WebSocket connection currently held by that user.
The SPA filters by envelope ``stream`` client-side.

Why a single hub keyed by user_id (not by stream id):
- Coordinator producers always know which user owns a turn, job, or
  project — every code path in this app already enforces ownership at
  authz time.
- Browsers open one WebSocket per tab; routing per-user means the
  browser receives every realtime event for its user, including events
  for projects opened in other tabs. Cheaper than a
  one-WS-per-resource-stream design.

The hub keeps per-connection ``asyncio.Queue`` buffers so the WS pump
loop in ``app.api.ws`` can ``await queue.get()`` without contention.
Publishers never await the network — they just enqueue.

**Replay-on-reconnect.** Clients can send a single inbound
``{"action":"subscribe","stream":"<id>","since_seq":N}`` message to ask
the hub to replay every buffered envelope on ``stream`` whose ``seq``
is greater than ``N``. The hub looks up a :class:`StreamReplayProvider`
(any object exposing ``events`` + ``events_dropped`` — both
:class:`AgentTurn` and :class:`Job`` already do), snapshots
``events[since_seq+1:]``, and enqueues the snapshot to the connection.
Generic across stream types so chat-turn refresh and job-progress
refresh share the same primitive.

Live envelopes for a stream being replayed are parked and flushed after
the snapshot — see :meth:`WsConnection.begin_replay`. They are not merely
deduplicated afterwards: the client discards anything at or below its
cursor, so a live envelope arriving FIRST advances that cursor past the
snapshot and the replayed frames are lost, not reordered.

**Replayed envelopes are marked** via ``mark_replayed`` so consumers
can distinguish a buffered envelope from a live one without asking the
server. Momentary consumers — those treating an event as "this is
happening right now" — must check the marker and ignore replayed
envelopes, because arming a spinner or opening a subscription for a
turn that finished minutes ago is actively wrong.

**Replay authorization.** Replay would otherwise be a cross-tenant leak:
``stream`` is client-supplied, so a tenant who learned another tenant's
``turn:<uuid>`` / ``job:<uuid>`` (logs, support handoff, leaked
screenshot) could subscribe and dump that tenant's transcript. Every
provider is registered with the resource's ``owner_id``; replay rejects
when the requesting connection's ``user_id`` doesn't match.

This matches the existing live-publish posture: ``AgentTurn`` and
``Job`` both fan out via ``publish_to_user(owner_id, ...)`` exclusively
(see ``Job.append_event`` and the orchestrator publish path in
``agent_runner.start_agent_turn``). Only the owner may read a project,
so live publishes and replay both gate on the owner alone.

A connection is in the user bucket, and so eligible for live publishes,
from the moment the WS opens — a round trip before the subscribe handler
runs. Envelopes landing in that window are parked by the replay barrier
and flushed after the snapshot. Any resulting duplicate is harmless: the
SPA dedupes via its ``lastSeqByStream`` map in ``ws-client.ts``. Order is
what that map cannot recover from, which is why the barrier exists.

Keepalive: a single background task (started from the FastAPI lifespan)
emits a ``keepalive`` envelope on every active connection every 30s.
This keeps the WS connection alive past the ALB's default 60s idle
timeout — relying on app-level keepalive instead of bumping the LB.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.services.ws_envelope import make_envelope, make_envelope_with_seq, mark_replayed, mark_degraded

logger = logging.getLogger(__name__)


# Outbound queue capacity per WS connection. Generous because envelopes
# are small (text deltas, statuses) and the consumer (the WS pump) drains
# them in the same event loop. If a consumer is wedged for so long that
# 1024 backs up, the ``put_nowait`` raises ``QueueFull`` and we drop the
# envelope — logged at WARNING — rather than blocking the publisher.
_QUEUE_MAXSIZE = 1024

# Caps on envelopes parked while a stream is not yet ready for live delivery
# (see ``WsConnection.offer_is_parked``). Separate from the outbound queue
# because they bound a different thing: the window between a connection joining
# the fanout bucket and its first replay finishing, normally one network round
# trip.
#
# The TOTAL cap is the load-bearing one. A connection only ever subscribes to the
# streams its UI is showing, so anything published to another of that user's
# projects parks and is never flushed — harmless per stream, unbounded across
# them. Capping the total keeps one connection's holding cost fixed no matter how
# many streams its owner has in flight.
_HELD_MAXSIZE = 1024
_HELD_TOTAL_MAXSIZE = 2048


# Application-level keepalive cadence (seconds). 30s is comfortably
# under the ALB's default 60s idle timeout and avoids a chatty NIC for
# idle dashboards. Tunable via ``set_keepalive_interval`` in tests.
_KEEPALIVE_INTERVAL_SEC = 30.0


# Per-project lock registry for publish_project. Prevents
# buffer-order / seq-order drift when concurrent publish calls await the
# durable allocator. A synchronous in-memory mint would keep ordering for free;
# the await between the durable mint and the record is what makes the lock
# mandatory. Uses the same lazy double-checked create pattern as
# ws_envelope._counters. Do NOT reuse services/project_locks.py — that lock's
# docstring scopes it to file critical sections and forbids broadening.
_project_locks: dict[str, asyncio.Lock] = {}
_project_locks_lock = asyncio.Lock()


@runtime_checkable
class StreamReplayProvider(Protocol):
    """Source of buffered envelopes for replay-on-reconnect.

    Implementations: :class:`app.services.agent_turn.AgentTurn` (chat
    turns) and :class:`app.services.jobs.Job` (background eval /
    optimization jobs). Both store envelopes in an ``events`` list with
    a parallel ``events_dropped`` counter that records how many
    oldest entries the buffer has shed.

    The contract is intentionally minimal — anything with these two
    attributes can register against the hub for free, no inheritance
    or wiring beyond a single ``register_replay_provider`` call at
    construction time.
    """

    events: list[dict[str, Any]]
    """Ordered envelope dicts the resource has emitted, oldest first."""

    events_dropped: int
    """Count of oldest entries dropped to keep memory bounded."""


@dataclass(frozen=True)
class _ReplayEntry:
    """Stored replay provider plus the resource owner's user_id.

    The hub binds ``owner_id`` at registration time so
    :meth:`WsHub.replay_to_connection` can reject cross-tenant
    subscribes — ``stream`` is client-supplied, so without this gate any
    authenticated user who learned another tenant's ``turn:<uuid>`` /
    ``job:<uuid>`` could pull that tenant's buffered transcript.
    """

    provider: StreamReplayProvider
    owner_id: str


class WsConnection:
    """One WebSocket connection's bookkeeping.

    Holds the per-connection outbound queue plus the user_id so the hub
    can fan out by user. The actual ``WebSocket`` object lives on this
    class so the keepalive task and the hub never need to import
    FastAPI's WS type directly.

    Attributes:
        connection_id: Process-unique id for log correlation.
        user_id: The authenticated user this connection belongs to.
        websocket: The underlying ``fastapi.WebSocket``. Stored as
            ``Any`` to avoid a hard dependency on FastAPI in this
            module (keeps unit-test fixtures simpler).
        queue: Outbound envelope queue. The route's pump task awaits
            ``queue.get`` and forwards via ``websocket.send_json``.
        held: Live envelopes parked per stream while that stream replays.
            See :meth:`begin_replay`.
    """

    __slots__ = ("connection_id", "user_id", "websocket", "queue", "held")

    def __init__(
        self,
        connection_id: str,
        user_id: str,
        websocket: Any,
    ) -> None:
        self.connection_id = connection_id
        self.user_id = user_id
        self.websocket = websocket
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        # stream -> parked live envelopes. A key present means "replaying".
        self.held: dict[str, list[dict[str, Any]]] = {}

    def offer_is_parked(self, stream: str) -> bool:
        """Whether ``stream`` is holding rather than delivering right now.

        True only while a replay for that stream is in flight. Deliberately NOT
        "until the client has subscribed", which would be the wider guarantee:
        subscription is OPTIONAL in this protocol. A client can open the socket
        and consume live frames without ever subscribing — ``test_chat_ws_e2e``
        does exactly that, and so does any consumer that only wants what happens
        from now on. Holding until a subscribe arrives makes those frames arrive
        never.

        The residual window is therefore real and is documented on
        :meth:`begin_replay`: frames published between registration and the
        server processing the client's first ``subscribe`` still go direct.
        Closing it belongs on the client, which owns the cursor — it can buffer a
        forward-gap frame while its own replay is outstanding instead of
        advancing past it. Doing it here cannot work without breaking delivery for
        every non-subscribing consumer.
        """
        return self.held.get(stream) is not None

    def begin_replay(self, stream: str) -> None:
        """Park live envelopes for ``stream`` until :meth:`end_replay`.

        Covers the replay's own duration, which is the larger part of the race:
        the handler awaits the durable store several times per subscribe, and live
        envelopes published across those awaits were enqueued AHEAD of the
        snapshot.

        That is loss, not reordering. The client advances its per-stream cursor on
        the first frame it accepts and discards anything at or below it, so one
        early live frame turns every replayed frame underneath into a "duplicate"
        it drops and never requests again.

        What this does NOT cover: a frame published after the connection joined
        the fanout bucket but before the server began this replay. Nothing on the
        server knows a subscribe is coming, and holding by default breaks clients
        that never subscribe (see :meth:`offer_is_parked`). That remainder is a
        client-side cursor concern.
        """
        self.held.setdefault(stream, [])

    def end_replay(self, stream: str) -> int:
        """Mark ``stream`` ready and flush the parked envelopes in seq order.

        Everything parked is delivered, including anything the replay may also
        have covered. Overlap is safe and under-delivery is not: the client drops
        a frame whose sequence is at or below its cursor, which is the documented
        dedupe this hub already relies on. Order is what was broken.

        Returns:
            How many parked envelopes were delivered.
        """
        # Pop BEFORE flushing, so anything published while we drain goes straight
        # to the queue behind what we are about to enqueue rather than into a
        # bucket nobody will look at again.
        parked = self.held.pop(stream, [])
        if not parked:
            return 0

        def _seq(env: dict[str, Any]) -> int:
            value = env.get("seq")
            return value if isinstance(value, int) else -1

        delivered = 0
        for envelope in sorted(parked, key=_seq):
            try:
                self.queue.put_nowait(envelope)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(
                    "[ws-hub] queue full flushing held envelopes conn=%s "
                    "stream=%s; dropped kind=%s seq=%s",
                    self.connection_id, stream,
                    envelope.get("kind"), envelope.get("seq"),
                )
                break
        return delivered

    def offer(self, envelope: dict[str, Any]) -> None:
        """Enqueue an envelope, or park it if its stream is not ready yet.

        Raises:
            asyncio.QueueFull: The outbound queue is full and the envelope was
                not parked. Callers log and drop, as before.
        """
        stream = envelope.get("stream")
        if isinstance(stream, str) and self.offer_is_parked(stream):
            parked = self.held.setdefault(stream, [])
            total = sum(len(v) for v in self.held.values())
            if len(parked) < _HELD_MAXSIZE and total < _HELD_TOTAL_MAXSIZE:
                parked.append(envelope)
                return
            # Holding capacity exhausted. Two bad options; this is the less bad
            # one. Dropping loses the frame outright — including a terminal frame,
            # which would hang the UI. Enqueuing restores the old racy ordering,
            # which the client can at least partly recover from via its gap
            # detection. Reaching either cap means frames are piling up for a
            # stream the client has not subscribed to, which is a different
            # problem from the one this barrier solves.
            logger.warning(
                "[ws-hub] held capacity reached conn=%s stream=%s "
                "(stream=%d, total=%d); publishing live before replay "
                "(ordering may be wrong)",
                self.connection_id, stream, len(parked), total,
            )
        self.queue.put_nowait(envelope)


class WsHub:
    """Singleton registry of active WebSocket connections.

    Connections are keyed by user_id; each user can hold N concurrent
    connections (multiple tabs / devices). Publishers call
    :meth:`publish_to_user` and the hub forwards the envelope to each
    of that user's queues.

    Thread safety: all mutation happens on the asyncio event loop
    (FastAPI's WS lifecycle hooks + our keepalive task), so the dict
    pointer swaps are safe under CPython's GIL without explicit locks.
    Same model as ``AgentTurnTracker`` and the old ``ProjectEventBus``.
    """

    _instance: "WsHub | None" = None
    _by_user: dict[str, set[WsConnection]]
    _replay_providers: dict[str, _ReplayEntry]
    _keepalive_task: asyncio.Task | None
    _keepalive_interval_sec: float

    def __new__(cls) -> "WsHub":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._by_user = {}
            instance._replay_providers = {}
            instance._keepalive_task = None
            instance._keepalive_interval_sec = _KEEPALIVE_INTERVAL_SEC
            cls._instance = instance
        return cls._instance

    def register(self, conn: WsConnection) -> None:
        """Add a freshly-accepted connection to the registry.

        Called by the WS route after ``websocket.accept()`` succeeds.
        Idempotent — registering the same connection twice is a
        programming error but won't corrupt state.

        Args:
            conn: The :class:`WsConnection` to register.
        """
        bucket = self._by_user.setdefault(conn.user_id, set())
        bucket.add(conn)
        logger.debug(
            "[ws-hub] register user=%s conn=%s total=%d",
            conn.user_id, conn.connection_id, len(bucket),
        )

    def unregister(self, conn: WsConnection) -> None:
        """Remove a connection from the registry.

        Called by the WS route's ``finally`` block on disconnect (clean
        close, network drop, or token rotation). Drops empty user
        buckets to avoid leaking memory across user churn.

        Args:
            conn: The :class:`WsConnection` to remove.
        """
        bucket = self._by_user.get(conn.user_id)
        if bucket is None:
            return
        bucket.discard(conn)
        if not bucket:
            self._by_user.pop(conn.user_id, None)
        logger.debug(
            "[ws-hub] unregister user=%s conn=%s remaining=%d",
            conn.user_id, conn.connection_id, len(bucket),
        )

    async def publish_to_user(
        self,
        user_id: str,
        envelope: dict[str, Any],
    ) -> None:
        """Fan out an envelope to every connection owned by ``user_id``.

        Non-blocking on the network: the envelope is enqueued to each
        connection's outbound queue, and the per-connection pump task
        is responsible for the actual ``websocket.send_json``. If a
        queue is full the envelope is dropped (logged at WARNING) so a
        wedged consumer can't stall the publisher.

        Args:
            user_id: The owner of the connections to fan out to.
            envelope: A canonically-shaped envelope (use
                :func:`make_envelope` to construct one).
        """
        bucket = self._by_user.get(user_id)
        if not bucket:
            # No active connections for this user — drop silently. The
            # SPA will refetch via REST on its next page load (the
            # browser was never connected to receive the event).
            return
        # Iterate over a copy so a concurrent unregister doesn't mutate
        # the set under us.
        for conn in list(bucket):
            try:
                # ``offer`` rather than a bare put: it parks the envelope when
                # this connection is mid-replay on that stream, so live frames
                # cannot land ahead of the replay snapshot.
                conn.offer(envelope)
            except asyncio.QueueFull:
                logger.warning(
                    "[ws-hub] queue full for conn=%s user=%s; dropped envelope kind=%s stream=%s",
                    conn.connection_id, user_id,
                    envelope.get("kind"), envelope.get("stream"),
                )

    async def publish_project(
        self,
        project_id: str,
        owner_id: str,
        event: dict[str, Any] | None = None,
        *,
        kind: str = "project_event",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Publish a project-scoped event to ``owner_id``'s connections.

        Convenience wrapper for the common ``ProjectEventBus``
        replacement pattern: take a free-form event dict (the same
        shape the old bus consumed), wrap it as a ``project_event``
        envelope on stream ``project:<id>``, and fan out to the
        project's owner.

        Also records the envelope into the project's replay buffer and
        registers the buffer as the replay provider for the stream (on
        first publish) so any future subscribe-with-replay delivers
        buffered envelopes.

        **The durable allocator is the single seq authority.**
        Uses the metadata store's ``allocate_project_seq`` under a
        per-project lock so buffer order == seq order even when concurrent
        publishes await the store. On allocation failure, publishes anyway
        with a buffer-seeded fallback seq (never drops the event) and logs
        WARNING.

        Args:
            project_id: The project the event belongs to.
            owner_id: The user_id owning the project (must be known
                at the call site — every existing publisher already
                has it via ``project.owner_id``).
            event: Free-form event payload (the old bus shape:
                ``{"resource": "...", "type": "...", ...}``). Mutually
                exclusive with ``data``.
            kind: Envelope kind (default ``"project_event"``).
            data: Envelope data payload. If provided, ``event`` must be None.
        """
        from app.platform import get_platform
        from app.services.project_event_tracker import record_envelope, get_provider

        # Normalize event vs data
        if event is not None and data is not None:
            raise ValueError("Cannot provide both event and data")
        payload = data if data is not None else event
        if payload is None:
            raise ValueError("Must provide either event or data")

        # Acquire per-project lock (lazy create, double-checked)
        lock = _project_locks.get(project_id)
        if lock is None:
            async with _project_locks_lock:
                lock = _project_locks.get(project_id)
                if lock is None:
                    lock = asyncio.Lock()
                    _project_locks[project_id] = lock

        # Inside the lock: allocate, mint, record, register, trace, fan-out.
        # The lock preserves buffer-order == seq-order. A synchronous mint would
        # keep ordering for free; the await between the durable mint and the
        # record is what makes the lock mandatory. Deliberately
        # NOT sorting the snapshot in replay_to_connection — that would defend
        # the same hazard and could keep the ordering test green with the lock
        # removed (a second mechanism masking a regression in the first). The
        # lock is the single owner of ordering.
        async with lock:
            stream = f"project:{project_id}"
            seq_source = "durable"

            try:
                seq = await get_platform().metadata_store.allocate_project_seq(project_id)
            except Exception as exc:
                # Allocation failure: publish anyway with buffer-seeded fallback seq.
                # Never drop — the eval-completion event is exactly what the user loses.
                # The fallback seq is no longer load-bearing for correctness (the marker
                # is what prevents the drop), but it provides best-effort local ordering.
                # Seeding from the buffer keeps the fallback ahead of what this process
                # already delivered rather than restarting at 0.
                provider = get_provider(project_id)
                existing_seqs = [e["seq"] for e in provider.events]
                fallback_seq = (max(existing_seqs) if existing_seqs else -1) + 1
                seq = fallback_seq
                seq_source = "fallback"
                logger.warning(
                    "[project-seq-fallback] Allocation failed for project=%s, "
                    "publishing with fallback seq=%d (degraded, not correct by design). "
                    "Error: %s",
                    project_id[:12],
                    seq,
                    exc,
                )

            envelope = make_envelope_with_seq(
                stream=stream,
                kind=kind,
                data=payload,
                seq=seq,
            )

            # Mark envelope as degraded if allocation failed. Both the buffered copy
            # and the live copy need the marker, so we apply it to the envelope before
            # recording or fan-out. The marker is what makes the degraded publish safe:
            # the client bypasses its cursor entirely for degraded envelopes.
            if seq_source == "fallback":
                envelope = mark_degraded(envelope)

            # Record into the buffer BEFORE fanning out so the buffer is
            # consistent even if a subscriber calls replay between these two steps.
            record_envelope(project_id, envelope)

            # Register the provider (idempotent — replaces if already registered).
            provider = get_provider(project_id)
            self.register_replay_provider(stream, provider, owner_id=owner_id)

            # Fan out to live connections.
            await self.publish_to_user(owner_id, envelope)

    def register_replay_provider(
        self,
        stream: str,
        provider: StreamReplayProvider,
        *,
        owner_id: str,
    ) -> None:
        """Register a buffered-envelope source for ``stream``.

        Called by resource constructors (``AgentTurn.__init__``,
        ``JobTracker.create_job``) so any future client subscribe-with-replay
        on this stream can rewind through the resource's event buffer.
        Idempotent — re-registering replaces the previous provider,
        which is what we want when an earlier turn for the same
        project is replaced.

        Args:
            stream: The stream id, e.g. ``"turn:abc"``, ``"job:def"``.
            provider: Anything matching :class:`StreamReplayProvider`.
            owner_id: The user_id who owns the underlying resource.
                :meth:`replay_to_connection` rejects when the
                requesting connection's ``user_id`` doesn't match this
                value — without that check, ``stream`` is a
                client-supplied lookup key that would dump any tenant's
                buffered envelopes to any authenticated caller.
        """
        self._replay_providers[stream] = _ReplayEntry(
            provider=provider, owner_id=owner_id,
        )

    def unregister_replay_provider(self, stream: str) -> None:
        """Drop the replay provider for ``stream``. Called on resource teardown."""
        self._replay_providers.pop(stream, None)

    async def replay_to_connection(
        self,
        conn: WsConnection,
        stream: str,
        since_seq: int,
    ) -> int:
        """Enqueue every buffered envelope on ``stream`` past ``since_seq``.

        Replayed envelopes are marked server-side so consumers can distinguish
        a buffered envelope from a live one without asking the server. Momentary
        consumers — those treating an event as "this is happening right now" —
        must check the marker and ignore replayed envelopes, because arming a
        spinner or opening a subscription for a turn that finished minutes ago
        is actively wrong.

        Returns the count of envelopes enqueued so callers can log /
        observe replay activity. Returns 0 if no provider is registered
        (idle stream, or the resource finalized and was unregistered).

        The SPA dedupes against its own ``lastSeqByStream`` map, so a
        small overlap window where this snapshot races with a live
        publisher fan-out is harmless — duplicates are dropped on
        receipt rather than at the source.

        Args:
            conn: The :class:`WsConnection` to enqueue into.
            stream: The stream id to replay.
            since_seq: The highest envelope seq the client has already
                seen on this stream. ``-1`` means "I've seen nothing,
                replay everything in the buffer".
        """
        entry = self._replay_providers.get(stream)
        if entry is None:
            return 0
        # Cross-tenant guard: the caller's WS connection is bound to a
        # specific user_id at accept-time; the resource was registered
        # with its owner_id at construction time. Any mismatch is either
        # a bug (registration site missed a tenant) or an exploit
        # (caller learned another tenant's stream id). In both cases
        # the safe response is to return zero replayed envelopes — the
        # idle-stream path already returns 0, so the leakage signal
        # ("does this stream exist?") is not increased.
        if entry.owner_id != conn.user_id:
            logger.warning(
                "[ws-hub] replay rejected — owner mismatch conn=%s user=%s stream=%s owner=%s",
                conn.connection_id, conn.user_id, stream, entry.owner_id,
            )
            return 0

        # Buffer-honesty check. When the buffer exposes
        # `oldest_retained_seq` (project streams), verify it can serve the
        # requested range. If `since_seq >= 0` (not a fresh client) and the buffer
        # cannot serve from `since_seq + 1`, emit a cannot-serve envelope instead
        # of an empty replay (which reads as "nothing new").
        #
        # Use a sentinel to distinguish "attribute doesn't exist" (non-project stream)
        # from "attribute exists but is None" (empty buffer after restart). Do NOT
        # use hasattr + getattr — that evaluates the property twice, and it has TTL
        # side effects.
        _MISSING = object()
        oldest_seq_attr = getattr(entry.provider, "oldest_retained_seq", _MISSING)

        if oldest_seq_attr is not _MISSING and since_seq >= 0:
            # Provider exposes oldest_retained_seq (a project stream); check servability.
            # oldest_seq_attr can be None (empty buffer) or int (oldest seq).
            oldest_seq = oldest_seq_attr

            if oldest_seq is None:
                # Empty buffer (restart case): cannot serve any non-negative cursor.
                # This branch must fire cannot-serve: an outer guard that
                # short-circuited on an empty buffer would hand a reconnecting
                # client an empty replay after a restart, which reads as
                # "nothing new".
                cannot_serve = True
            elif since_seq + 1 < oldest_seq:
                # Client cursor predates the oldest retained entry (TTL or cap eviction).
                cannot_serve = True
            else:
                # Buffer can serve from since_seq + 1.
                cannot_serve = False

            if cannot_serve:
                # Mint a cannot-serve envelope through the same durable path
                # as any other project envelope (so it carries a real seq in
                # the one seq domain). The gap envelope's seq is strictly greater
                # than the requesting client's since_seq on both the durable path
                # (a project-wide monotonic counter) and the fallback path (which
                # takes max(buffer_seqs, since_seq) + 1). This is what makes it safe
                # for the client to adopt the gap seq as its cursor without creating
                # a permanent hole.
                #
                # Do NOT record the gap envelope into the shared project buffer: that
                # buffer is the record of what was fanned out to every connection, and
                # injecting a cannot-serve envelope there would deliver an unrequested
                # refetch trigger to other connections on their next replay. A DIFFERENT
                # connection later replaying across the unrecorded seq sees one forward
                # hole and answers it with exactly one silent refetch — bounded structurally,
                # because the cursor advances onto the frame that carried the gap.
                from app.platform import get_platform

                project_id = stream.split(":", 1)[1]
                try:
                    gap_seq = await get_platform().metadata_store.allocate_project_seq(project_id)
                except Exception:
                    # Allocation failed; use buffer-seeded fallback (same as publish_project
                    # rule). Seed from the buffer's existing seqs OR the client's cursor
                    # so the gap envelope stays ahead of anything this process already delivered
                    # OR anything the client has already seen. Do NOT use 0 (the measured
                    # collision value) or oldest_seq (likely already seen by client).
                    buffer_seqs = [e["seq"] for e in entry.provider.events]
                    # Use the max of: buffer seqs, client cursor, or -1 (if both empty)
                    max_seen = max(
                        (max(buffer_seqs) if buffer_seqs else -1),
                        since_seq,
                    )
                    gap_seq = max_seen + 1

                gap_envelope = make_envelope_with_seq(
                    stream=stream,
                    kind="replay_gap",
                    data={"since_seq": since_seq, "oldest_retained_seq": oldest_seq},
                    seq=gap_seq,
                )
                try:
                    conn.queue.put_nowait(gap_envelope)
                except asyncio.QueueFull:
                    logger.warning(
                        "[ws-hub] replay gap queue full conn=%s stream=%s",
                        conn.connection_id, stream,
                    )
                return 1  # One cannot-serve envelope enqueued

        # Snapshot the buffer in one synchronous step so a concurrent
        # ``append_event`` call on the resource doesn't mutate the list
        # while we iterate.
        snapshot = [
            env for env in list(entry.provider.events)
            if isinstance(env.get("seq"), int) and env["seq"] > since_seq
        ]
        # Mark each envelope as replayed before enqueuing. Mark returns a copy,
        # so the buffer entry stays clean for the next replay.
        #
        # Count what was actually ENQUEUED, not what was selected. Returning
        # ``len(snapshot)`` after a truncating break reported a complete replay
        # to every caller and to the log, so nothing downstream could tell that
        # frames had been dropped — the one fact recovery logic needs.
        enqueued = 0
        truncated_at: int | None = None
        for envelope in snapshot:
            marked = mark_replayed(envelope)
            try:
                conn.queue.put_nowait(marked)
            except asyncio.QueueFull:
                truncated_at = envelope.get("seq")
                logger.warning(
                    "[ws-hub] replay queue full conn=%s stream=%s; truncated at "
                    "seq=%s after %d of %d frame(s)",
                    conn.connection_id, stream, truncated_at, enqueued, len(snapshot),
                )
                break
            enqueued += 1
        if snapshot:
            logger.debug(
                "[ws-hub] replay conn=%s stream=%s since_seq=%d enqueued=%d of %d",
                conn.connection_id, stream, since_seq, enqueued, len(snapshot),
            )
        return enqueued

    def start_keepalive(self) -> None:
        """Start the background keepalive task.

        Idempotent — safe to call from FastAPI's lifespan startup hook
        even if a previous run already started one. Called once per
        process at startup.
        """
        if self._keepalive_task is not None and not self._keepalive_task.done():
            return
        loop = asyncio.get_event_loop()
        self._keepalive_task = loop.create_task(
            self._keepalive_loop(),
            name="ws-hub-keepalive",
        )
        logger.info("[ws-hub] keepalive task started (interval=%.1fs)",
                    self._keepalive_interval_sec)

    def stop_keepalive(self) -> None:
        """Cancel the keepalive task. Called from lifespan shutdown."""
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = None

    async def _keepalive_loop(self) -> None:
        """Emit a ``keepalive`` envelope on every active connection.

        Runs forever until cancelled. The envelope uses
        ``stream="user:<user_id>"`` so the SPA can dispatch keepalives
        without touching any per-resource subscriber.
        """
        while True:
            try:
                await asyncio.sleep(self._keepalive_interval_sec)
            except asyncio.CancelledError:
                return
            try:
                # Snapshot the bucket map so concurrent register/unregister
                # during the iteration doesn't ConcurrentModification us.
                snapshot = [
                    (user_id, list(bucket))
                    for user_id, bucket in self._by_user.items()
                ]
                for user_id, conns in snapshot:
                    envelope = make_envelope(
                        stream=f"user:{user_id}",
                        kind="keepalive",
                        data={},
                    )
                    for conn in conns:
                        try:
                            conn.queue.put_nowait(envelope)
                        except asyncio.QueueFull:
                            # A wedged consumer with no headroom for a
                            # keepalive will be force-closed by the WS
                            # pump on its next send attempt.
                            logger.debug(
                                "[ws-hub] keepalive drop conn=%s queue full",
                                conn.connection_id,
                            )
            except Exception:
                # Never let the keepalive task die on a transient error;
                # log and keep running.
                logger.exception("[ws-hub] keepalive iteration failed")

    def set_keepalive_interval(self, interval_sec: float) -> None:
        """Set the keepalive cadence. Tests use a short interval."""
        self._keepalive_interval_sec = interval_sec

    def _reset(self) -> None:
        """Drop all state. For testing only."""
        self._by_user.clear()
        self._replay_providers.clear()
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = None
        self._keepalive_interval_sec = _KEEPALIVE_INTERVAL_SEC


def get_ws_hub() -> WsHub:
    """Return the process-wide :class:`WsHub` singleton."""
    return WsHub()


__all__ = [
    "StreamReplayProvider",
    "WsConnection",
    "WsHub",
    "get_ws_hub",
]
