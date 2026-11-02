# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket envelope shape + per-stream sequence numbering.

Every realtime message the backend pushes to a browser is wrapped in
the same envelope::

    {
        "stream": "turn:<turn_id>" | "job:<job_id>" |
                  "project:<project_id>" | "user:<user_id>",
        "seq":    <monotonic int per stream>,
        "kind":   "text_delta" | "tool_start" | "tool_input" | "tool_end" |
                  "tool_result" | "status" | "result" | "progress" |
                  "done" | "project_event" | "error" | "keepalive" |
                  "replay_gap",
        "data":   { ... kind-specific payload ... },
        "degraded": true  # omitted when healthy (absence means healthy)
    }

The SPA filters by ``stream`` and dispatches by ``kind``. The
monotonic ``seq`` per stream lets the SPA detect dropped messages on
reconnect and ask ``ws_hub`` to replay every buffered envelope whose
``seq`` exceeds the last one it saw.

This module owns the envelope construction; it stays separate from
``ws_hub`` so test fixtures can build envelopes without spinning up
the registry.

**ONE SEQ DOMAIN PER STREAM.** A stream's seq values must all come from the
same source. This matters because the backend has two counters: the in-memory one
below (per process, starts at 0, dies with the process) and the durable
``allocate_*_seq`` counter in the metadata store (starts at 1, persists). A stream
that has a DURABLE leg — turn events and job events both do — must use the durable
seq for BOTH the live publish and the durable append. Mixing them produces two
mechanical client failures, because ``ws-client`` keeps one cursor per stream:

* the live publish's seq collides with an already-seen durable seq, so the client
  drops the frame as a duplicate; or
* the live publish's seq jumps far ahead, so a later durable replay is dropped as
  stale.

Either way the frame is silently discarded, terminal frames included — the spinner
hangs with the backend having done everything right.

**The rule also applies when a stream has no durable EVENT LOG but its replay buffer
is consulted by MORE THAN ONE PUBLISHING PROCESS.** Two producers (the coordinator
hub and a microVM worker) have issued the same seq (both 0) on one ``project:``
stream within a single coordinator lifetime — a collision, not a restart reset. The
client cursor sat at 0 from the first producer, asked for ``> 0`` on reconnect, and
the second producer's buffered envelope (also 0) was filtered out as already-seen.
``project:`` therefore draws from the durable allocator despite having no durable
event log, because its replay buffer (ProjectEventBuffer) and the client cursor
compared against it are both consulted by more than one publisher. ``user:`` does
not need this, because ``ws-client.ts`` discards keepalive envelopes before it
consults the cursor.

**Degraded publishes (allocation failures).** When no durable seq is reachable — the
store is down, the IAM policy is misconfigured, or the counter's key structure changed
— a non-colliding seq is NOT derivable. A fresh process has an empty buffer and no
shared state to derive one from. The honest contract: the publisher marks the envelope
``degraded``, and the client treats its cursor as unusable for that envelope. The
envelope is still recorded into the buffer and still fanned out — the event is never
dropped. The marker key is OMITTED on healthy envelopes (absence means healthy), and
the client bypasses the cursor decision entirely for degraded envelopes, delivering
the payload without comparing seqs. Fixing the underlying IAM denial or store outage
does NOT retire this path: any future allocation failure re-arms it. This ensures
eval-completion events (and all other critical project publishes) are impossible to
drop even when the infrastructure is degraded.

Use :func:`make_envelope_with_seq` on any stream with a durable leg OR multiple
publishers, and plain :func:`make_envelope` only for single-publisher ephemeral
streams (``user:`` keepalive).
"""
from __future__ import annotations

from itertools import count
from threading import Lock
from typing import Any


# Stream scope prefix for project-scoped streams. Used by make_envelope's
# rejection guard — project streams must use the durable allocator, not
# the in-memory counter (measured: two producers, both seq=0).
_PROJECT_SCOPE_PREFIX = "project:"


# Per-stream sequence counters. ``itertools.count`` provides O(1) +1
# semantics; the lock guards the dict insert path so two concurrent
# publishers for a brand-new stream don't both create separate counters.
# Subsequent ``next(_counters[key])`` calls are CPython-atomic under the
# GIL — no lock needed on the hot path.
_counters: dict[str, "count[int]"] = {}
_counters_lock = Lock()


def _next_seq(stream: str) -> int:
    """Return the next monotonic sequence number for ``stream``.

    Counters are created lazily on first use and persist for the lifetime
    of the process. A streaming consumer that reconnects after a long
    idle period MAY see a higher initial seq than it expects — that's
    fine; the contract is "monotonic within a stream", not "starts at 0
    on every connect".

    Args:
        stream: The stream id (``turn:<id>``, ``job:<id>``, etc.).

    Returns:
        The next sequence number for the stream (starts at 0).
    """
    counter = _counters.get(stream)
    if counter is None:
        with _counters_lock:
            # Double-checked: another thread may have created the counter
            # between the dict.get and the lock acquisition.
            counter = _counters.get(stream)
            if counter is None:
                counter = count()
                _counters[stream] = counter
    return next(counter)


def make_envelope(stream: str, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Build a WebSocket envelope with the next sequence number for ``stream``.

    Raises ``ValueError`` when ``stream`` begins with the project scope prefix,
    because project streams must use the durable allocator to avoid the measured
    collision (two producers, both seq=0, one coordinator lifetime).
    Use ``allocate_project_seq`` from the metadata store, then
    ``make_envelope_with_seq``.

    Args:
        stream: The stream id, formatted as ``"<scope>:<id>"`` —
            e.g. ``"turn:abc123"``, ``"job:def456"``, ``"user:dev"``.
            **NOT** ``"project:proj-1"`` — that raises.
        kind: One of the closed kind vocabulary (``text_delta``,
            ``tool_start``, ``tool_input``, ``tool_end``, ``tool_result``,
            ``status``, ``result``, ``progress``, ``done``,
            ``project_event``, ``error``, ``keepalive``). The hub doesn't
            enforce this — callers pass strings — but the SPA's dispatch
            table is the closed set.
        data: Kind-specific payload. Must be JSON-serializable; the WS
            route encodes envelopes via ``websocket.send_json``.

    Returns:
        A new dict — never reused or mutated after return.

    Raises:
        ValueError: When ``stream`` begins with ``"project:"``.
    """
    if stream.startswith(_PROJECT_SCOPE_PREFIX):
        raise ValueError(
            f"Project streams must use the durable allocator, not the in-memory counter. "
            f"Call get_platform().metadata_store.allocate_project_seq(project_id), "
            f"then make_envelope_with_seq(stream, kind, data, seq)."
        )

    return {
        "stream": stream,
        "seq": _next_seq(stream),
        "kind": kind,
        "data": data,
    }


def make_envelope_with_seq(
    stream: str, kind: str, data: dict[str, Any], seq: int
) -> dict[str, Any]:
    """Build an envelope carrying a CALLER-SUPPLIED seq.

    For streams with a durable leg: the caller allocates the durable seq once, then
    uses it for both the live publish and the durable append, so the stream has a
    single seq domain. Passing the same envelope (or the same seq) to both legs is
    what makes the client's per-stream cursor coherent across a reconnect.

    Deliberately a separate function rather than an optional ``seq=`` parameter on
    :func:`make_envelope`: a default would let a caller on a durable stream forget
    and silently fall back to the in-memory counter, which is exactly the bug this
    exists to prevent. Two names make the choice explicit at every call site.

    Args:
        stream: Stream id (``turn:<id>``, ``job:<id>``).
        kind: Envelope kind (see :func:`make_envelope`).
        data: Kind-specific payload.
        seq: The durable sequence number, already allocated via the store's
            ``allocate_turn_seq`` / ``allocate_job_seq``.

    Returns:
        A new dict — never reused or mutated after return.
    """
    return {
        "stream": stream,
        "seq": seq,
        "kind": kind,
        "data": data,
    }


def mark_replayed(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``envelope`` carrying a replay marker.

    The marker is set server-side at the hub so consumers can distinguish a buffered
    envelope from a live one without asking the server. Momentary consumers — those
    treating an event as "this is happening right now" — must check the marker and
    ignore replayed envelopes, because arming a spinner or opening a subscription for
    a turn that finished minutes ago is actively wrong.

    The marker key is OMITTED on live envelopes (the live JSON contract is unchanged),
    so absence means live. Consumers must check for key absence, not merely falsy
    values.

    This function returns a shallow copy and never mutates its argument: a buffer
    entry is read by every subsequent replay, and a live envelope object is shared
    across a user's connections, so in-place marking would corrupt both.

    Args:
        envelope: A canonically-shaped envelope from :func:`make_envelope`.

    Returns:
        A new dict with one extra top-level key indicating the envelope came from
        a buffer. The ``data`` payload is shared (shallow copy), so a mutating
        consumer would still corrupt the original — but no consumer mutates envelopes
        today, and the contract is "envelopes flow one way" (hub → client).
    """
    return {**envelope, "replayed": True}


def mark_degraded(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``envelope`` carrying a degraded marker.

    The marker is set server-side at the hub when a durable seq allocation fails
    (store down, IAM misconfigured, key structure changed). The envelope still
    carries its real payload and kind — it is never dropped — but the marker tells
    the client that the seq is not load-bearing for correctness: the client must
    bypass its cursor decision entirely for degraded envelopes and deliver them
    without comparing seqs.

    Why a marker on the envelope rather than a distinct kind: a distinct kind would
    fail every existing consumer that switches on ``kind`` — ``useBackendEvents``
    in the frontend requires ``project_event`` and would drop the payload outright.
    The requirement is that the eval-completion event (or any other critical publish)
    be DELIVERED, not replaced by a notice that something was lost. The ``replayed``
    marker is the existing precedent for exactly this shape.

    The marker key is OMITTED on healthy envelopes (absence means healthy), so
    consumers must check for key absence, not merely falsy values.

    This function returns a shallow copy and never mutates its argument: both the
    buffered copy and the live copy need the marker, and an in-place mutation would
    corrupt subsequent replays.

    Args:
        envelope: A canonically-shaped envelope from :func:`make_envelope_with_seq`.

    Returns:
        A new dict with one extra top-level key indicating the envelope came from
        a degraded publish (durable seq allocation failed). The ``data`` payload
        is shared (shallow copy).
    """
    return {**envelope, "degraded": True}


def reset_counters() -> None:
    """Reset all sequence counters. For testing only."""
    with _counters_lock:
        _counters.clear()


__all__ = ["make_envelope", "make_envelope_with_seq", "mark_replayed", "mark_degraded", "reset_counters"]
