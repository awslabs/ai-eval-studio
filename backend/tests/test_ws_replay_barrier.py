# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live envelope must not reach the client ahead of the replay snapshot.

A connection joins the user's fanout bucket at accept time, a full round trip
before the client's ``subscribe`` arrives, and the replay then awaits the durable
store several times. Live envelopes published in that window were enqueued ahead
of the snapshot.

That is not a harmless reordering. The client advances its per-stream cursor on
the first frame it accepts and discards anything at or below it, so once a later
live sequence lands first, every replayed frame underneath it is dropped and
never requested again — chat text goes missing with the backend having done
everything else right.
"""
import asyncio

import pytest

from app.services.ws_hub import WsConnection

pytestmark = pytest.mark.asyncio

STREAM = "turn:abc"


def _conn() -> WsConnection:
    return WsConnection(connection_id="c1", user_id="userA", websocket=object())


def _envelope(seq: int, kind: str = "text_delta") -> dict:
    return {"stream": STREAM, "kind": kind, "seq": seq, "data": {}}


def _drain(conn: WsConnection) -> list[int]:
    seqs: list[int] = []
    while not conn.queue.empty():
        seqs.append(conn.queue.get_nowait()["seq"])
    return seqs


async def test_live_envelopes_are_parked_during_replay():
    """While a stream replays, a live frame goes to the holding buffer."""
    conn = _conn()
    conn.begin_replay(STREAM)

    conn.offer(_envelope(105))

    assert _drain(conn) == [], "a live frame must not be queued mid-replay"
    assert conn.held[STREAM][0]["seq"] == 105


async def test_replayed_frames_precede_the_parked_live_frame():
    """The ordering the client needs: snapshot first, then the live tail."""
    conn = _conn()
    conn.begin_replay(STREAM)

    # A live frame arrives first, as it did in the racing window.
    conn.offer(_envelope(105))
    # The replay then enqueues the snapshot it was fetching.
    for seq in (101, 102, 103, 104):
        conn.queue.put_nowait(_envelope(seq))

    conn.end_replay(STREAM)

    assert _drain(conn) == [101, 102, 103, 104, 105], (
        "the live frame must land after the replayed range, not before it"
    )


async def test_parked_frames_are_flushed_in_sequence_order():
    """Several live frames can pile up; they must not invert."""
    conn = _conn()
    conn.begin_replay(STREAM)

    for seq in (107, 105, 106):
        conn.offer(_envelope(seq))
    conn.end_replay(STREAM)

    assert _drain(conn) == [105, 106, 107]


async def test_other_streams_are_unaffected_by_a_replay():
    """The barrier is per stream — a project event must not be delayed."""
    conn = _conn()
    conn.begin_replay(STREAM)

    conn.offer({"stream": "project:p1", "kind": "status", "seq": 7, "data": {}})

    assert _drain(conn) == [7], "an unrelated stream must flow straight through"


async def test_publishing_outside_a_replay_is_delivered_immediately():
    """No replay in flight, no parking.

    Load-bearing, not merely a default. Subscription is OPTIONAL in this protocol:
    a client can open the socket and consume live frames without ever subscribing,
    which ``test_chat_ws_e2e`` relies on. Holding until a subscribe arrives makes
    those frames arrive never — a stricter barrier was tried and hung that test.
    """
    conn = _conn()
    conn.offer(_envelope(200))
    assert _drain(conn) == [200]
    assert conn.held == {}


async def test_total_held_capacity_is_bounded_across_streams():
    """Concurrent replays on many streams must not grow holding without limit."""
    from app.services.ws_hub import _HELD_TOTAL_MAXSIZE

    conn = _conn()
    for i in range(16):
        conn.begin_replay(f"project:p{i}")
    for i in range(_HELD_TOTAL_MAXSIZE):
        conn.offer(
            {"stream": f"project:p{i % 16}", "kind": "status", "seq": i, "data": {}}
        )

    assert conn.queue.empty()
    # Past the total cap, publishing goes direct rather than growing further.
    conn.offer({"stream": "project:p0", "kind": "status", "seq": 9999, "data": {}})
    assert conn.queue.qsize() == 1


async def test_end_replay_is_safe_without_a_begin():
    """Defensive: a stream that never opened a barrier flushes nothing."""
    conn = _conn()
    assert conn.end_replay("turn:never") == 0


async def test_barrier_is_released_even_when_replay_raises():
    """A failed replay must not leave the stream parked forever.

    The route wraps the replay in try/finally for exactly this: a stuck barrier
    would silently stop delivering that stream for the life of the connection.
    """
    from app.api import ws as ws_mod

    conn = _conn()

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("durable store down")

    original = ws_mod._replay_stream
    ws_mod._replay_stream = _boom
    try:
        with pytest.raises(RuntimeError):
            await ws_mod._handle_inbound(
                conn, '{"action":"subscribe","stream":"turn:abc","since_seq":-1}'
            )
    finally:
        ws_mod._replay_stream = original

    assert STREAM not in conn.held, "the barrier must be released on failure"
    conn.offer(_envelope(300))
    assert _drain(conn) == [300], "delivery must resume after a failed replay"


async def test_held_buffer_overflow_falls_back_to_direct_publish():
    """Bounded: a runaway publisher must not grow the holding buffer forever.

    Falling back to the old racy ordering is deliberate — it is worse than
    correct order but better than dropping the frame outright.
    """
    from app.services.ws_hub import _QUEUE_MAXSIZE

    conn = _conn()
    conn.begin_replay(STREAM)

    for seq in range(_QUEUE_MAXSIZE):
        conn.offer(_envelope(seq))
    assert conn.queue.empty(), "everything up to the cap should be parked"

    conn.offer(_envelope(_QUEUE_MAXSIZE))
    assert conn.queue.qsize() == 1, "past the cap, publish directly"


async def test_terminal_frame_held_during_replay_is_still_delivered():
    """The result frame is what clears the spinner — it cannot be lost here."""
    conn = _conn()
    conn.begin_replay(STREAM)
    conn.offer(_envelope(99, kind="result"))
    conn.end_replay(STREAM)

    drained = []
    while not conn.queue.empty():
        drained.append(conn.queue.get_nowait())
    assert [e["kind"] for e in drained] == ["result"]


async def test_hub_publish_respects_the_barrier():
    """The fanout path must go through offer(), not a bare queue put."""
    from app.services.ws_hub import get_ws_hub

    hub = get_ws_hub()
    conn = _conn()
    hub.register(conn)
    try:
        conn.begin_replay(STREAM)
        await hub.publish_to_user("userA", _envelope(150))
        assert conn.queue.empty(), "publish_to_user must park during replay"
        conn.end_replay(STREAM)
        assert _drain(conn) == [150]
    finally:
        hub.unregister(conn)


async def test_concurrent_publish_during_a_slow_replay_keeps_order():
    """End to end through the hub, with the replay actually awaiting."""
    from app.services.ws_hub import get_ws_hub

    hub = get_ws_hub()
    conn = _conn()
    hub.register(conn)
    try:
        conn.begin_replay(STREAM)

        async def _slow_replay() -> None:
            # Stands in for the durable-store round trips the real replay makes.
            await asyncio.sleep(0)
            for seq in (10, 11):
                conn.queue.put_nowait(_envelope(seq))

        async def _live() -> None:
            await hub.publish_to_user("userA", _envelope(12))

        await asyncio.gather(_live(), _slow_replay())
        conn.end_replay(STREAM)

        assert _drain(conn) == [10, 11, 12]
    finally:
        hub.unregister(conn)
