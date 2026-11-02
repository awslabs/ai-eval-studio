# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory registry of bounded per-project envelope buffers, with eviction.

Satisfies :class:`app.services.ws_hub.StreamReplayProvider` so project-scoped
events published with no subscriber are delivered on the next subscribe.

Named ``_tracker`` per ``docs/naming-conventions.md``: this file owns an
in-memory registry of live/recent objects **with eviction** — the role's exact
test. (``_buffer`` is not in the closed role vocabulary; the buffers are the
registry's contents, not its role.) Sibling precedent: ``JobTracker``.

**Memory discipline**: Two backstops prevent unbounded growth:
1. Per-project cap (_MAX_ENVELOPES_PER_PROJECT = 100) — project events are
   small status dicts (~400B), not token streams, so 100 envelopes (~40 KB per
   project) far exceeds what a reconnecting SPA needs. Compare AgentTurn's
   buffer, which is sized for token-by-token streaming and is the wrong
   reference point.
2. LRU eviction past a ceiling (_MAX_PROJECT_BUFFERS = 200) — the coordinator
   is a single Fargate task; 200 live projects (~8 MB of buffers) is well past
   observed usage (typical < 10 concurrent projects).

Both are memory backstops, not correctness knobs. Exceeding either sheds the
OLDEST data — failure mode is "an unusually stale reconnect misses old events",
never a wrong or duplicated event.

**Usage**: Call :func:`record_envelope` from ``WsHub.publish_project`` after
minting the envelope. The hub also calls :func:`get_provider` to register the
provider for replay-on-reconnect.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# Per-project envelope cap. Project events are small status dicts (job state
# transitions, optimization progress), typically ~400 bytes. 100 envelopes is
# ~40 KB per project, far more history than a reconnecting SPA needs. A
# token-by-token turn accumulates 50K+ chars; comparing to AgentTurn's buffer
# (sized for streaming) would be an order-of-magnitude overshoot.
_MAX_ENVELOPES_PER_PROJECT = 100

# Global project-buffer ceiling. The coordinator is a single Fargate task; 200
# projects with live buffers (~8 MB total across all projects) is well past
# observed usage (typical < 10 concurrent projects on one coordinator). When
# exceeded, the least-recently-used project buffer is evicted.
_MAX_PROJECT_BUFFERS = 200

# Replay TTL (seconds). Envelopes older than this are not replayed. 600 seconds
# (10 minutes) is comfortably longer than any reconnect gap (keepalive is 30s,
# ALB idle timeout 60s) and longer than optimization iteration spacing, so a
# legitimately-late state event still replays. Far shorter than "tab open
# overnight". This is the second guard (after the momentary/state classification),
# not the primary fix, so the value doesn't need to be surgical.
_REPLAY_TTL_SECONDS = 600.0


def _get_monotonic_time() -> float:
    """Return the current monotonic clock reading (seconds).

    Exposed as a module-level function so tests can patch it without mocking
    the entire time module. Monotonic is the right clock here: only durations
    within a single process are ever computed, the buffer dies with the
    coordinator, and monotonic is immune to NTP steps and clock adjustment.
    Wall-clock comparability across processes is irrelevant.
    """
    return time.monotonic()


class ProjectEventBuffer:
    """Bounded envelope buffer for one project, satisfying StreamReplayProvider.

    Stores project-scoped envelopes in append order with shed-oldest discipline
    when the per-project cap is exceeded. Envelopes older than the TTL are
    excluded from replay (read-time filter) to bound memory during long idle
    periods and prevent stale envelopes from landing on a refresh.

    Thread-safe under CPython's GIL (all mutation happens on the asyncio event
    loop, same model as AgentTurn).

    Attributes:
        project_id: The project this buffer belongs to.
        events: Ordered envelope dicts younger than the TTL, oldest first.
            This is a property that filters ``_entries`` on read, so staleness
            is unobservable regardless of when the last append happened.
        events_dropped: Count of oldest entries dropped to keep memory bounded
            (cap-shed only; TTL expiry is tracked separately).
        events_expired: Count of entries excluded by TTL (read-time filter).
    """

    __slots__ = ("project_id", "_entries", "events_dropped", "events_expired")

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        # Store (recorded_at, envelope) pairs so we can filter by age on read
        self._entries: list[tuple[float, dict[str, Any]]] = []
        self.events_dropped: int = 0
        self.events_expired: int = 0

    def _ttl_cutoff(self) -> float:
        """Return the TTL cutoff timestamp (monotonic seconds).

        Factored into one helper so the `events` property and
        `oldest_retained_seq` can never disagree — a drift between them
        would silently reintroduce the bug this fix prevents.
        """
        return _get_monotonic_time() - _REPLAY_TTL_SECONDS

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return envelopes younger than the TTL, oldest first.

        Read-time filter makes staleness unobservable: an envelope that expired
        while the buffer sat idle is still excluded. This is the load-bearing
        half; append-time pruning is only a memory measure.
        """
        cutoff = self._ttl_cutoff()
        # Filter and extract just the envelopes (drop timestamps)
        fresh = []
        expired_count = 0
        for recorded_at, envelope in self._entries:
            if recorded_at >= cutoff:
                fresh.append(envelope)
            else:
                expired_count += 1
        # Update expired counter (for observability; doesn't affect correctness)
        if expired_count > 0:
            self.events_expired += expired_count
        return fresh

    @property
    def oldest_retained_seq(self) -> int | None:
        """Return the seq of the oldest entry surviving the TTL filter, or None.

        This is the buffer's honest lower bound. When `since_seq < oldest_retained_seq`,
        the buffer cannot serve the requested range and replay must emit a
        cannot-serve signal.

        Reuses the same TTL cutoff as the `events` property so the two can never
        disagree — a drift would silently reintroduce the silent-loss bug.

        Returns:
            The seq of the oldest servable envelope, or None when the buffer
            holds no servable entries (empty buffer, or all entries expired).
        """
        cutoff = self._ttl_cutoff()
        for recorded_at, envelope in self._entries:
            if recorded_at >= cutoff:
                return envelope["seq"]
        return None

    def append(self, envelope: dict[str, Any]) -> None:
        """Append an envelope to the buffer, shedding oldest if at cap.

        Also prunes expired entries before appending (memory measure only —
        read-time filter is what makes staleness unobservable).

        Args:
            envelope: A canonically-shaped envelope from ``make_envelope``.
        """
        # Prune expired entries before append (memory bound during active sessions)
        now = _get_monotonic_time()
        cutoff = self._ttl_cutoff()
        # Keep only entries within TTL
        fresh_entries = [(ts, env) for ts, env in self._entries if ts >= cutoff]
        expired_count = len(self._entries) - len(fresh_entries)
        if expired_count > 0:
            self._entries = fresh_entries
            self.events_expired += expired_count

        # Append the new envelope with current timestamp
        self._entries.append((now, envelope))

        # Cap-shed if needed (count AFTER pruning, so an expired entry doesn't
        # trigger a cap-shed)
        if len(self._entries) > _MAX_ENVELOPES_PER_PROJECT:
            self._entries.pop(0)
            self.events_dropped += 1


class _BufferRegistry:
    """Global registry of per-project buffers with LRU eviction.

    Keyed by project_id; each project gets a lazily-created buffer. When the
    registry exceeds _MAX_PROJECT_BUFFERS, the least-recently-used project
    buffer is evicted so memory cannot grow without limit.

    Thread-safe under CPython's GIL (all mutation on the asyncio event loop).
    """

    def __init__(self) -> None:
        # OrderedDict tracks insertion/access order for LRU eviction.
        self._buffers: OrderedDict[str, ProjectEventBuffer] = OrderedDict()

    def get_or_create(self, project_id: str) -> ProjectEventBuffer:
        """Get the buffer for ``project_id``, creating it if needed.

        Lazily creates buffers on first publish (not pre-allocated per
        project). Updates LRU order so recently-used projects stay in the
        registry.

        Args:
            project_id: Project UUID.

        Returns:
            The project's buffer (existing or newly-created).
        """
        if project_id in self._buffers:
            # Move to end (mark as recently-used).
            self._buffers.move_to_end(project_id)
            return self._buffers[project_id]

        # Create new buffer
        buffer = ProjectEventBuffer(project_id)
        self._buffers[project_id] = buffer

        # LRU eviction: shed the oldest project buffer if we exceed the ceiling.
        if len(self._buffers) > _MAX_PROJECT_BUFFERS:
            evicted_id, evicted_buffer = self._buffers.popitem(last=False)
            # Dropping our own reference is NOT enough to bound memory: the WS
            # hub's ``_replay_providers`` entry holds a strong reference to this
            # buffer (registered in ``publish_project``), so without the
            # unregister below the evicted buffer stays alive with its envelopes
            # and ``_MAX_PROJECT_BUFFERS`` bounds nothing — the leak the ceiling
            # exists to prevent. Verified: 6 projects with a ceiling of 3 left 3
            # buffers here but 6 live registrations, each pinning its buffer.
            #
            # Imported here rather than at module scope: ws_hub imports this
            # module inside ``publish_project``, so a module-scope import back
            # into ws_hub would be a cycle.
            from app.services.ws_hub import get_ws_hub

            get_ws_hub().unregister_replay_provider(f"project:{evicted_id}")
            logger.info(
                "[project-buffer] LRU eviction: dropped buffer for project=%s "
                "(%d envelopes, %d dropped) and unregistered its replay provider",
                evicted_id[:12],
                len(evicted_buffer.events),
                evicted_buffer.events_dropped,
            )

        return buffer

    def get(self, project_id: str) -> ProjectEventBuffer | None:
        """Get the buffer for ``project_id`` without creating one.

        Used by tests to inspect buffer state. Does NOT update LRU order.

        Args:
            project_id: Project UUID.

        Returns:
            The buffer if it exists, None otherwise.
        """
        return self._buffers.get(project_id)

    def clear(self) -> None:
        """Clear all buffers. For testing only."""
        self._buffers.clear()


# Global registry singleton.
_registry = _BufferRegistry()


def record_envelope(project_id: str, envelope: dict[str, Any]) -> None:
    """Record an envelope into the project's buffer.

    Called by ``WsHub.publish_project`` after minting the envelope. Lazily
    creates the buffer on first publish.

    Args:
        project_id: Project UUID.
        envelope: A canonically-shaped envelope from ``make_envelope``.
    """
    buffer = _registry.get_or_create(project_id)
    buffer.append(envelope)


def get_provider(project_id: str) -> ProjectEventBuffer:
    """Get or create the replay provider for ``project_id``.

    Called by ``WsHub.publish_project`` to register the provider for
    replay-on-reconnect (via ``register_replay_provider``). The provider
    satisfies :class:`StreamReplayProvider` — it exposes ``events`` and
    ``events_dropped`` attributes.

    Args:
        project_id: Project UUID.

    Returns:
        The project's buffer (satisfies StreamReplayProvider).
    """
    return _registry.get_or_create(project_id)


def _reset() -> None:
    """Clear all buffers. For testing only."""
    _registry.clear()
