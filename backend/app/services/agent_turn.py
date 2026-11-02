# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent turn tracker for decoupling agent execution from HTTP connections.

The agent turn runs as a background asyncio.Task writing envelope dicts
to an in-memory buffer. The buffer is the source of truth; the
``ws_hub`` fan-out is best-effort. WS clients reconnecting after a
network blip can replay from the buffer using the per-stream ``seq``.

Architecture:
  POST /api/chat  →  creates AgentTurn + background task  →  returns turn_id
  WebSocket /ws    →  receives envelopes via ws_hub fan-out

The turn queue serializes turns per project: one turn runs at a time.
User-origin turns have higher priority than orchestrator turns. Queued
turns build their configuration lazily (at slot-open time) so they
capture the latest project state.
"""

import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from app.services.resource_error import ResourceError

logger = logging.getLogger(__name__)

# Per-turn event buffer cap. Partial-message streaming produces many
# small envelopes (one per token chunk); 20k covers a long multi-minute
# turn with ~500 tool uses and keeps worst-case memory under ~10 MB per
# turn. When exceeded the oldest envelopes are dropped; ``events_dropped``
# records the shed count for replay logic to skip past evicted seqs.
_MAX_EVENT_BUFFER = 20_000


def _resolve_owner_id_for_project(project_id: str) -> str:
    """Look up the project's owner_id; fall back to the local-dev user.

    Used by :class:`AgentTurnTracker` and :class:`JobTracker` to bind a
    new turn / job's replay provider to its tenant. Replay would
    otherwise be cross-tenant-leakable: ``stream`` is client-supplied,
    so without an owner binding any authenticated user could subscribe
    to another tenant's ``turn:`` / ``job:`` stream and pull buffered
    envelopes.

    The dev-mode fallback to ``get_local_dev_user_id()`` matches what
    other dispatch sites already do (e.g. ``mcp_tools/dispatch.py``)
    when no request context is available — single-user dev never has
    cross-tenant concerns and tests scaffold raw project_ids without
    project metadata.

    Args:
        project_id: Project UUID.

    Returns:
        The project owner's user_id, or the dev fallback.
    """
    try:
        from app.main import get_project_manager
        project = get_project_manager().get_project(project_id)
        if project is not None and project.owner_id:
            return project.owner_id
    except Exception:
        pass
    from app.platform.auth import get_local_dev_user_id
    return get_local_dev_user_id()


class AgentTurn:
    """A single agent turn with its buffered envelope events.

    Uses asyncio.Condition for notification so readers never miss events.
    (asyncio.Event with set()+clear() has a race where the notification
    is lost if the reader hasn't called wait() yet.)

    Attributes:
        turn_id: Unique identifier for this turn.
        project_id: Project this turn belongs to.
        status: Current turn state.
        events: Ordered list of WS envelope dicts (the full buffer).
            Stored as dicts and serialized to JSON only at the WS pump
            boundary in ``app.api.ws``.
        created_at: ISO 8601 timestamp.
        session_id: Session ID from the SDK result message.
        error: Structured :class:`ResourceError` when the turn failed,
            else None. Set by ``fail()`` and inspectable by any code
            that reads the turn after the WS stream ended.
    """

    def __init__(
        self,
        turn_id: str,
        project_id: str,
        origin: str = "user",
        *,
        owner_id: str | None = None,
    ):
        self.turn_id = turn_id
        self.project_id = project_id
        self.owner_id = owner_id or _resolve_owner_id_for_project(project_id)
        self.origin: Literal["user", "orchestrator"] = origin
        self.status: Literal["running", "completed", "failed"] = "running"
        self.events: list[dict[str, Any]] = []
        # Number of oldest events dropped to bound memory. Replay logic
        # uses this so a reconnecting client knows where the buffer
        # actually starts.
        self.events_dropped: int = 0
        self._condition = asyncio.Condition()
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.session_id: str | None = None
        self.error: ResourceError | None = None
        self._client = None  # ClaudeSDKClient reference for interrupt
        # Set to True by stop_turn endpoint when user explicitly stops.
        # The agent_runner exception handler checks this to distinguish
        # user cancellation from genuine SDK errors (no error toast).
        self._stopped_by_user: bool = False
        # Tracks whether this turn holds the one-writer-per-transcript session
        # claim (the DDB claim). When True, run_agent_turn's finally releases
        # the claim on exit.
        # Completion/orchestrator turns set this when relay is enabled to
        # serialize against relay user turns (cross-process mutual exclusion).
        self._holds_ddb_claim: bool = False
        # Register as a replay provider so a SPA reconnecting mid-turn
        # can ask the WS hub for buffered envelopes via the
        # ``{"action":"subscribe","since_seq":N}`` message. ``owner_id``
        # is captured here so the hub can reject cross-tenant replay
        # subscribes (``stream`` is client-supplied). Cleanup happens
        # in ``AgentTurnTracker._evict_old_turns``.
        from app.services.ws_hub import get_ws_hub
        get_ws_hub().register_replay_provider(
            f"turn:{turn_id}", self, owner_id=self.owner_id,
        )

    async def append_event(self, event: dict[str, Any]) -> None:
        """Append an envelope dict to the buffer and notify all waiting readers.

        Drops oldest envelopes past ``_MAX_EVENT_BUFFER`` to keep memory
        bounded on long turns; ``events_dropped`` records the shed count.
        """
        self.events.append(event)
        if len(self.events) > _MAX_EVENT_BUFFER:
            overflow = len(self.events) - _MAX_EVENT_BUFFER
            del self.events[:overflow]
            self.events_dropped += overflow
            if self.events_dropped == overflow or self.events_dropped % 2000 == 0:
                logger.warning(
                    "[turn %s] event buffer overflow; dropped %d so far",
                    self.turn_id[:8],
                    self.events_dropped,
                )
        async with self._condition:
            self._condition.notify_all()

    async def complete(self) -> None:
        """Mark turn as completed and wake all waiting readers."""
        self.status = "completed"
        async with self._condition:
            self._condition.notify_all()

    async def fail(self, error: ResourceError | None = None) -> None:
        """Mark turn as failed and wake all waiting readers.

        Args:
            error: Optional :class:`ResourceError` describing the failure.
                When supplied, attached to the turn so out-of-band
                consumers (not reading the event stream) can still see it.
        """
        self.status = "failed"
        if error is not None:
            self.error = error
        async with self._condition:
            self._condition.notify_all()

    async def wait_for_events(self, known_count: int, timeout: float = 15.0) -> bool:
        """Wait until new events are available or timeout expires.

        ``known_count`` is an absolute event ordinal. Compared against
        ``events_dropped + len(events)`` so buffer eviction doesn't
        strand the waiter on indices that were shed.

        Args:
            known_count: Absolute count of events the caller has seen.
            timeout: Maximum seconds to wait.

        Returns:
            True if new events available, False on timeout.
        """
        async with self._condition:
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(
                        lambda: self.events_dropped + len(self.events) > known_count
                    ),
                    timeout=timeout,
                )
                return True
            except asyncio.TimeoutError:
                return False


@dataclass
class QueuedTurn:
    """A turn waiting in the per-project queue.

    Attributes:
        turn_id: Reserved turn ID, handed to ``build_turn`` when the slot opens
            so the turn that eventually runs IS the one the caller was told
            about. It used to be merely "pre-assigned": ``build_turn`` took no
            arguments and could not receive it, so the builder minted a second,
            unrelated id and the reserved one named nothing.
        project_id: Project this turn belongs to.
        origin: Who initiated the turn.
        priority: Lower = higher priority. User turns get 0, orchestrator gets 10.
        build_turn: Callable that constructs and starts the turn when the slot
            opens, using the reserved turn id it is given.
        created_at: Timestamp for ordering same-priority turns (FIFO).
    """

    turn_id: str
    project_id: str
    origin: Literal["user", "orchestrator"]
    priority: int
    build_turn: Callable[[str], "AgentTurn"]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Optional cleanup hook invoked if this turn is DROPPED from the queue
    # before build_turn ever runs (e.g. drain_queued_turns during cancel).
    # The completion/claim path uses it to release the DDB session claim it
    # acquired before submit — otherwise a cancelled-while-queued turn leaks
    # the claim until the TTL backstop. In-process turns pass None (no-op).
    on_drop: Callable[[], None] | None = None

    def __lt__(self, other: "QueuedTurn") -> bool:
        """Sort by priority first, then by creation time (FIFO within priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class AgentTurnTracker:
    """Singleton in-memory store for active agent turns with turn queue.

    Tracks one turn per project (new turn replaces old). The turn queue
    serializes execution: one turn runs at a time per project. When the
    active turn completes, the next queued turn starts.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._turns: dict[str, AgentTurn] = {}
            cls._instance._by_project: dict[str, str] = {}
            cls._instance._queues: dict[str, list[QueuedTurn]] = {}
            cls._instance._processing: set[str] = set()
        return cls._instance

    def create_turn(
        self,
        project_id: str,
        origin: str = "user",
        *,
        owner_id: str | None = None,
        turn_id: str | None = None,
    ) -> AgentTurn:
        """Create a new turn for a project, replacing any previous active turn.

        Args:
            project_id: Project this turn belongs to.
            origin: Who initiated this turn — "user" or "orchestrator".
            owner_id: User_id who owns the project. Bound to the turn
                so the WS hub can reject cross-tenant replay subscribes.
                When ``None`` the tracker resolves it from project
                metadata so callers don't need to plumb it through.
            turn_id: Use this id instead of minting one. For callers that had to
                commit to an id BEFORE the turn existed — a queued request that
                already answered the browser, or a session claim taken before the
                turn is built. Minting a second id there left the caller holding
                an identifier that named nothing.

        Returns:
            The new AgentTurn instance.
        """
        if owner_id is None:
            owner_id = _resolve_owner_id_for_project(project_id)
        if turn_id is None:
            turn_id = str(uuid.uuid4())
        turn = AgentTurn(
            turn_id=turn_id,
            project_id=project_id,
            owner_id=owner_id,
            origin=origin,
        )

        # Clean up previous turn for this project
        old_id = self._by_project.get(project_id)
        if old_id and old_id in self._turns:
            old = self._turns[old_id]
            if old.status == "running":
                old.status = "failed"  # Mark stale turn directly (avoid async)
            from app.services.ws_hub import get_ws_hub
            get_ws_hub().unregister_replay_provider(f"turn:{old_id}")
            del self._turns[old_id]

        self._turns[turn_id] = turn
        self._by_project[project_id] = turn_id

        # Evict old completed/failed turns to prevent unbounded growth
        self._evict_old_turns(max_terminal=50)

        return turn

    def get_turn(self, turn_id: str) -> AgentTurn | None:
        """Get a turn by ID."""
        return self._turns.get(turn_id)

    def get_active_turn(self, project_id: str, origin: str | None = None) -> AgentTurn | None:
        """Get the active (running) turn for a project, optionally filtered by origin.

        Args:
            project_id: Project to check.
            origin: If provided, only return turns with this origin ("user" or "orchestrator").

        Returns:
            The active AgentTurn, or None.
        """
        turn_id = self._by_project.get(project_id)
        if turn_id:
            turn = self._turns.get(turn_id)
            if turn and turn.status == "running":
                if origin is not None and turn.origin != origin:
                    return None
                return turn
        return None

    def submit(
        self,
        project_id: str,
        origin: Literal["user", "orchestrator"],
        build_turn: Callable[[str], "AgentTurn"],
        priority: int | None = None,
        on_drop: Callable[[], None] | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Submit a turn to the per-project queue.

        If no turn is currently running for this project, the turn starts
        immediately. Otherwise it's queued and starts when the current
        turn completes.

        Args:
            project_id: Project UUID.
            origin: Who initiated this turn.
            build_turn: Callable that creates and starts the AgentTurn, given the
                reserved turn id (called when the slot opens, not at submit time).
            priority: Explicit priority (lower = higher). Defaults to
                0 for user, 10 for orchestrator.
            turn_id: Reserve this id instead of minting one, for a caller that
                already committed to it — e.g. one that took a session claim
                under that id before submitting.
            on_drop: Optional cleanup hook invoked if the turn is dropped
                from the queue before ``build_turn`` runs (e.g. by
                ``drain_queued_turns`` during cancel). Used by the
                completion/claim path to release the DDB session claim it
                acquired before submit. In-process callers pass None.

        Returns:
            Pre-assigned turn_id for the queued turn.
        """
        if priority is None:
            priority = 0 if origin == "user" else 10

        if turn_id is None:
            turn_id = str(uuid.uuid4())
        queued = QueuedTurn(
            turn_id=turn_id,
            project_id=project_id,
            origin=origin,
            priority=priority,
            build_turn=build_turn,
            on_drop=on_drop,
        )

        if project_id not in self._queues:
            self._queues[project_id] = []

        # Insert in priority order
        queue = self._queues[project_id]
        queue.append(queued)
        queue.sort()

        # Try to process if no turn is active
        self._try_process_queue(project_id)

        return turn_id

    def notify_turn_complete(self, project_id: str) -> None:
        """Called when a turn finishes — process the next queued turn.

        Args:
            project_id: Project whose turn just completed.
        """
        self._processing.discard(project_id)
        self._try_process_queue(project_id)

    def get_queued_turns(self, project_id: str) -> list[QueuedTurn]:
        """Get all queued (not yet running) turns for a project.

        Args:
            project_id: Project UUID.

        Returns:
            List of QueuedTurn instances in priority order.
        """
        return list(self._queues.get(project_id, []))

    def drain_queued_turns(
        self,
        project_id: str,
        origin: Literal["user", "orchestrator"] | None = None,
    ) -> int:
        """Remove queued (not yet running) turns for a project.

        Used by cancel endpoints to prevent stale queued turns (e.g.
        completion turns from a cancelled optimization) from running.

        Args:
            project_id: Project UUID.
            origin: If given, only drain turns of this origin. None drains all.

        Returns:
            Number of turns removed.
        """
        queue = self._queues.get(project_id, [])
        if not queue:
            return 0
        if origin is None:
            dropped = queue
            self._queues[project_id] = []
        else:
            kept = [q for q in queue if q.origin != origin]
            dropped = [q for q in queue if q.origin == origin]
            self._queues[project_id] = kept
        # Fire each dropped turn's cleanup hook so a queued-but-never-built turn
        # (e.g. a relay completion turn holding a DDB claim) doesn't leak its
        # resource. Best-effort: a hook error must not block draining the rest.
        for q in dropped:
            if q.on_drop is not None:
                try:
                    q.on_drop()
                except Exception:
                    logger.exception(
                        "[turn-queue] on_drop hook failed for dropped turn=%s project=%s",
                        q.turn_id[:8], project_id[:12],
                    )
        return len(dropped)

    def _try_process_queue(self, project_id: str) -> None:
        """Start the next queued turn if no turn is currently active.

        Called both by submit() and notify_turn_complete().
        """
        if project_id in self._processing:
            return

        active = self.get_active_turn(project_id)
        if active is not None:
            return

        queue = self._queues.get(project_id, [])
        if not queue:
            return

        # Pop the highest-priority item
        next_queued = queue.pop(0)
        self._processing.add(project_id)

        try:
            turn = next_queued.build_turn(next_queued.turn_id)
            if turn is None:
                raise RuntimeError("build_turn returned None")
            if turn.project_id != project_id:
                raise RuntimeError(
                    f"build_turn returned a turn for {turn.project_id}, "
                    f"expected {project_id}"
                )
            # The reserved id is what the caller was already told, so a builder
            # that mints its own leaves that caller holding an id that names
            # nothing: the browser subscribes to a stream with no events, its
            # Stop returns not found, and the real turn's terminal event is
            # rejected by the id guard, so the UI never stops showing "running".
            if turn.turn_id != next_queued.turn_id:
                raise RuntimeError(
                    f"build_turn returned turn {turn.turn_id}, expected the "
                    f"reserved id {next_queued.turn_id}"
                )
            logger.info(
                "[turn-queue] Started queued turn=%s origin=%s for project=%s",
                turn.turn_id[:8], next_queued.origin, project_id[:12],
            )
        except Exception as e:
            logger.error(
                "[turn-queue] Failed to build turn for project=%s: %s",
                project_id[:12], e, exc_info=True,
            )
            self._processing.discard(project_id)
            # Try the next one
            self._try_process_queue(project_id)

    def _evict_old_turns(self, max_terminal: int = 50) -> None:
        """Remove oldest completed/failed turns when capacity is exceeded."""
        terminal = [
            (tid, t) for tid, t in self._turns.items()
            if t.status in ("completed", "failed")
        ]
        if len(terminal) <= max_terminal:
            return
        terminal.sort(key=lambda x: x[1].created_at)
        to_evict = terminal[:len(terminal) - max_terminal]
        from app.services.ws_hub import get_ws_hub
        hub = get_ws_hub()
        for tid, t in to_evict:
            hub.unregister_replay_provider(f"turn:{tid}")
            del self._turns[tid]
            # Clean _by_project if it points to the evicted turn
            if self._by_project.get(t.project_id) == tid:
                del self._by_project[t.project_id]

    def _reset(self) -> None:
        """Reset all state. For testing only."""
        self._turns.clear()
        self._by_project.clear()
        self._queues.clear()
        self._processing.clear()


def get_agent_turn_tracker() -> AgentTurnTracker:
    """Get the singleton AgentTurnTracker instance."""
    return AgentTurnTracker()
