# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket endpoint for realtime chat / job / project events.

Realtime delivery uses a single ``/ws`` endpoint. The browser opens
one WebSocket per tab; the backend pushes envelopes for every event
the user is allowed to see. The SPA filters by ``stream`` client-side.

Auth:
- Browsers cannot set request headers on ``new WebSocket(url)``, so the token
  travels in the FIRST FRAME: the client sends
  ``{"action": "auth", "token": "<jwt>"}`` immediately on open.
- The route accepts the socket, reads that one frame, and calls
  ``Platform.auth.extract_user_id_from_token``. Nothing is registered in the hub
  and no envelope is delivered until it validates, so an unauthenticated socket
  observes nothing. Failure paths ``await ws.close(code=4401)`` (4401 = "auth
  required"; application-defined codes start at 4000 per RFC 6455 §7.4.2).
- NOT the query string, which is where it used to ride. A URL is logged by
  everything that touches it — uvicorn's access log records the full request line
  — so that put a live access token into log storage for the retention period.

Per-connection lifecycle:
1. Accept, read the auth frame, validate → register connection in ``ws_hub``.
2. Spawn an outbound pump task that awaits ``queue.get`` and forwards
   each envelope via ``websocket.send_json``.
3. Read inbound messages in a loop. The SPA may send
   ``{"action":"subscribe","stream":"<id>","since_seq":N}`` to request
   a buffered-envelope replay (chat-turn / eval-job mid-flight
   refresh recovery). The hub looks up the stream's
   ``StreamReplayProvider``, snapshots ``events[since_seq+1:]``, and
   enqueues them on this connection. Any other inbound payload is
   ignored. Receiving inbound also lets us observe
   ``WebSocketDisconnect`` for cleanup.
4. On disconnect (clean close, network drop, token rotation), cancel
   the pump task and unregister the connection from ``ws_hub``.

Reconnection on token rotation: the SPA closes the WS with code 1000
and immediately opens a new one with the fresh token. There's no
mid-connection re-auth handshake — close + reconnect is simpler to
reason about and the per-stream ``seq`` counter on the hub handles any
gap.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.ws_hub import WsConnection, get_ws_hub
from app.services.ws_envelope import mark_replayed

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])


# WebSocket close codes used in this route. Application-defined codes
# live in [4000, 4999] per RFC 6455 §7.4.2.
_CLOSE_AUTH_REQUIRED = 4401  # No / invalid token in the first frame.

#: Seconds an accepted socket may stay anonymous before it is closed.
#:
#: The client sends its auth frame immediately on ``open``, so this only has to
#: cover a slow network, not a slow user. It is short because authenticating after
#: accept rather than before it lets an anonymous caller hold a connection slot for
#: this long. The other thing an anonymous socket can do in that window is send a
#: frame, which the server buffers whole and this route parses as JSON before it
#: knows who sent it. Its size is bounded by uvicorn's ``--ws-max-size`` where the
#: server is launched, not here — by the time this route sees a frame it is
#: already in memory.
_AUTH_FRAME_TIMEOUT_S = 10.0

#: Stream name for connection-level frames that belong to no project stream.
#:
#: Reserved rather than reusing a real stream id: the client routes envelopes by
#: ``stream`` to whoever subscribed to it, and a control frame delivered into a
#: chat or job stream would reach a consumer that has no case for it.
_CONTROL_STREAM = "_control"


async def _receive_text_frame(websocket: WebSocket) -> str | None:
    """Read one frame and return its text, or None when it is not a text frame.

    ``WebSocket.receive_text`` indexes ``message["text"]``, and a binary frame
    carries ``bytes`` instead, so that call raised ``KeyError`` out of the route:
    an ERROR-level traceback any caller could produce with one frame, before
    authenticating. The protocol here is text-only, so a binary frame is handed
    back as None and each caller treats it like any other unusable frame.

    Args:
        websocket: An accepted socket.

    Returns:
        The frame's text, or None for a frame that carries no text.

    Raises:
        WebSocketDisconnect: The client closed the connection.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000), message.get("reason"))
    text = message.get("text")
    return text if isinstance(text, str) else None


async def _authenticate_first_frame(websocket: WebSocket, auth) -> str | None:  # noqa: ANN001
    """Read and validate the client's auth frame, closing the socket on failure.

    Expects ``{"action": "auth", "token": "<jwt>"}``. Anything else — a missing
    frame, a malformed payload, a different action, an invalid token — closes with
    4401 and returns None, so the caller's only job is to return.

    Reading exactly one frame rather than skipping ahead to the first auth frame
    is deliberate: tolerating other actions first would mean deciding what an
    unauthenticated ``subscribe`` does, and the answer has to be "nothing", which
    is simpler to guarantee by refusing to read past the handshake.

    Args:
        websocket: An already-accepted socket.
        auth: The platform's ``Auth`` implementation.

    Returns:
        The authenticated user id, or None when the socket has been closed.
    """
    try:
        raw = await asyncio.wait_for(
            _receive_text_frame(websocket), timeout=_AUTH_FRAME_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.info("[ws] reject — no auth frame within %gs", _AUTH_FRAME_TIMEOUT_S)
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None
    except WebSocketDisconnect:
        # Client hung up before authenticating. Nothing to close or clean up.
        return None

    if raw is None:
        logger.info("[ws] reject — first frame was not a text frame")
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        logger.info("[ws] reject — first frame was not JSON")
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None

    if not isinstance(payload, dict) or payload.get("action") != "auth":
        logger.info("[ws] reject — first frame was not an auth action")
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None

    # Only the frame's SHAPE is checked here. Whether a token is acceptable —
    # including whether an empty one is — belongs to the platform's Auth, the one
    # place that knows the deployment mode. Cloud rejects an empty token in
    # ``_validate_token``; dev ignores the value, and the browser deliberately
    # sends ``""`` there (``getAccessToken`` has no login to draw on). Refusing
    # empty strings at this layer rejected every dev socket, so turns finished
    # on the server while the UI waited for frames that never came.
    token = payload.get("token")
    if not isinstance(token, str):
        logger.info("[ws] reject — auth frame token is not a string")
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None

    user_id = await auth.extract_user_id_from_token(token)
    if user_id is None:
        logger.info("[ws] reject — invalid token")
        await websocket.close(code=_CLOSE_AUTH_REQUIRED)
        return None

    # Acknowledge, so "authenticated" is observable rather than inferred from the
    # absence of a close. Without it a client cannot tell a still-validating
    # socket from a failed one until the 4401 arrives, and a test cannot know when
    # registration has happened without polling for it.
    #
    # Shaped as an ordinary envelope on a reserved stream so nothing downstream
    # needs a special case: the client fans envelopes out by ``stream`` and no
    # component subscribes to this one, so it is dropped after serving its
    # purpose.
    await websocket.send_json(
        {"stream": _CONTROL_STREAM, "seq": 0, "kind": "auth_ok", "data": {}}
    )
    return user_id


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept a WebSocket, authenticate its first frame, and pump envelopes.

    Validation runs against the platform's :class:`Auth` Protocol on the token in
    the client's first frame. The socket is accepted before that, but nothing is
    registered in the hub and no envelope is queued until the token validates, so
    an unauthenticated connection is inert rather than merely unprivileged.

    Args:
        websocket: The FastAPI WebSocket object (auto-injected).
    """
    # 1. Accept, then authenticate from the FIRST FRAME.
    #
    # The token used to ride in the query string, because a browser cannot set
    # request headers on ``new WebSocket(url)``. The cost of that was a URL
    # carrying a live access token through every layer that logs a URL — uvicorn's
    # access log writes the full request line, and those logs are retained for a
    # month — so a token was recoverable from CloudWatch long after the socket
    # closed. Nothing about the query string is private.
    #
    # So the socket is accepted first and the client sends
    # ``{"action": "auth", "token": "<jwt>"}`` as its first frame. Nothing is
    # registered in the hub and no envelope is delivered until that frame
    # validates, so an unauthenticated socket observes nothing.
    #
    # The tradeoff, stated because it was chosen rather than defaulted into:
    # rejecting before ``accept`` returned HTTP 403 without ever completing the
    # upgrade, which is tidier. Authenticating after accept means an anonymous
    # caller can hold a socket for the deadline below. A one-time ticket endpoint
    # would keep the early rejection, but it needs a ticket store that only works
    # while one coordinator serves every connection, and this service already
    # carries one piece of coordinator-affinity state it should not.
    await websocket.accept()

    from app.platform import get_platform
    auth = get_platform().auth
    user_id = await _authenticate_first_frame(websocket, auth)
    if user_id is None:
        return

    conn = WsConnection(
        connection_id=str(uuid.uuid4()),
        user_id=user_id,
        websocket=websocket,
    )
    hub = get_ws_hub()
    hub.register(conn)
    logger.info("[ws] accept user=%s conn=%s", user_id, conn.connection_id[:8])

    # The connection is registered in this process's hub only; nothing writes
    # it to the store's durable connection registry. Delivery is in-process:
    # the poller and the relay both call publish_to_user inside the
    # coordinator process holding this socket.
    #
    # That is not the same as "there is only one coordinator". A rolling deploy
    # deliberately overlaps two tasks (compute_stack sets minHealthyPercent=100
    # and maxHealthyPercent=200), and the load balancer has no stickiness, so
    # during a deploy a publish can happen in the task that does NOT hold the
    # browser's socket — where it finds no connection and is dropped. The browser
    # recovers on its next subscribe, because the durable event log serves the
    # replay cross-process; live push does not recover. The store has a
    # connection registry (put_connection / list_connections_for_user) that
    # this route does not use.

    pump_task = asyncio.create_task(
        _pump_outbound(conn),
        name=f"ws-pump-{conn.connection_id[:8]}",
    )

    try:
        # Read inbound messages. Two protocols:
        #   1. ``subscribe`` — replay buffered envelopes on a stream past
        #      ``since_seq``. Single-shot; acknowledgment is the replayed
        #      envelopes themselves OR a ``replay_gap`` envelope when the buffer
        #      cannot serve the requested range.
        #   2. anything else — discarded, binary frames included. Reading the
        #      socket also lets us observe ``WebSocketDisconnect`` for cleanup.
        while True:
            try:
                raw = await _receive_text_frame(websocket)
            except WebSocketDisconnect as exc:
                # The close code and reason say WHY the socket went away: the
                # client's token rotation closes 1000 with reason
                # ``token-rotation``, a page refresh or navigation 1001, a dropped
                # network 1006. Without them a reconnect in this log cannot be
                # attributed. The reason is client-supplied text, so it is logged
                # with %r, which escapes anything that could forge a log line.
                logger.info(
                    "[ws] disconnect user=%s conn=%s code=%s reason=%r",
                    user_id, conn.connection_id[:8], exc.code, exc.reason,
                )
                return
            if raw is None:
                continue
            await _handle_inbound(conn, raw)
    finally:
        # Tear down: cancel the pump task and remove the conn from the
        # hub. Both are idempotent — running them in any order is safe.
        if not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
        hub.unregister(conn)


async def _replay_job_stream_from_durable_log(
    conn: WsConnection,
    stream: str,
    since_seq: int,
) -> tuple[int, int]:
    """Replay a job:<id> stream from the durable DDB event log.

    Args:
        conn: The WsConnection to enqueue into.
        stream: The stream id (must be "job:<id>").
        since_seq: The highest seq the client has already seen.

    Returns:
        ``(count, last_seq)`` — envelopes enqueued and the highest seq
        enqueued (``since_seq`` unchanged when nothing landed), so the
        caller can seq-merge the in-memory tail on top without ever
        re-sending a seq the durable leg already delivered.
    """
    # Extract the job_id from the stream
    if not stream.startswith("job:"):
        return 0, since_seq
    job_id = stream[len("job:"):]

    # Cross-tenant guard: resolve the job owner from the durable job record
    # and reject replay when conn.user_id != owner (mirror replay_to_connection's
    # owner-mismatch → zero-replay behavior).
    #
    # NOTE: The `job:<id>` subscribe frame carries ONLY stream + since_seq — NO
    # project_id. So we resolve owner via the in-process JobTracker. A job known
    # only to the durable store would need an owner-less job-by-id query, which
    # the store deliberately does not offer — do NOT invent one here.
    try:
        from app.services.jobs import get_job_tracker
        from app.main import get_project_manager

        job = get_job_tracker().get_job(job_id)
        if job is None:
            # Job not found in memory — may have been cleaned up or is from
            # another tenant, or this is a cold-start where the job only exists
            # durably. With a single coordinator process, the in-memory
            # tracker is authoritative.
            logger.debug(
                "[ws] durable replay — job not found in tracker conn=%s job_id=%s",
                conn.connection_id[:8], job_id[:8]
            )
            return 0, since_seq

        # Resolve owner_id from the project
        pm = get_project_manager()
        project = pm.get_project(job.project_id)
        owner_id = project.owner_id if project else None
        if owner_id is None or owner_id != conn.user_id:
            logger.warning(
                "[ws] durable replay rejected — owner mismatch conn=%s user=%s job=%s",
                conn.connection_id[:8], conn.user_id, job_id[:8]
            )
            return 0, since_seq
    except Exception as e:
        logger.warning(
            "[ws] durable replay — owner lookup failed conn=%s job=%s: %s",
            conn.connection_id[:8], job_id[:8], e
        )
        return 0, since_seq

    # Read events from the durable log
    from app.platform import get_platform
    store = get_platform().metadata_store
    try:
        events = await store.events_since(job_id, since_seq)
    except Exception as e:
        logger.warning(
            "[ws] durable replay — events_since failed conn=%s job=%s: %s",
            conn.connection_id[:8], job_id[:8], e
        )
        return 0, since_seq

    # Enqueue each event on the connection, marking as replayed.
    # The events returned by events_since already carry the DDB seq.
    # Mark them so consumers can distinguish durable replays from live events.
    count = 0
    last_seq = since_seq
    for event in events:
        marked = mark_replayed(event)
        try:
            conn.queue.put_nowait(marked)
            count += 1
            event_seq = event.get("seq")
            if isinstance(event_seq, int):
                last_seq = max(last_seq, event_seq)
        except asyncio.QueueFull:
            logger.warning(
                "[ws] durable replay queue full conn=%s stream=%s; truncated at seq=%s",
                conn.connection_id[:8], stream, event.get("seq"),
            )
            break

    if count > 0:
        logger.debug(
            "[ws] durable replay conn=%s stream=%s since_seq=%d count=%d",
            conn.connection_id[:8], stream, since_seq, count
        )

    return count, last_seq


async def _replay_turn_stream_from_durable_log(
    conn: WsConnection,
    stream: str,
    since_seq: int,
) -> tuple[int, int]:
    """Replay a turn:<id> stream from the durable DDB event log.

    Args:
        conn: The WsConnection to enqueue into.
        stream: The stream id (must be "turn:<id>").
        since_seq: The highest seq the client has already seen.

    Returns:
        ``(count, last_seq)`` — envelopes enqueued and the highest seq
        enqueued (``since_seq`` unchanged when nothing landed), so the
        caller can seq-merge the in-memory tail on top without ever
        re-sending a seq the durable leg already delivered.
    """
    # Extract the turn_id from the stream
    if not stream.startswith("turn:"):
        return 0, since_seq
    turn_id = stream[len("turn:"):]

    # Cross-tenant guard: resolve the turn owner from the durable turn metadata
    # and reject replay when conn.user_id != owner.
    #
    # SECURITY NOTE: The `turn:<id>` subscribe frame carries ONLY stream +
    # since_seq — NO project_id. We resolve owner via the durable turn metadata
    # (PK=TURN#{turn_id}, SK=META) stored at turn start by the relay client. This
    # is NOT an "owner-less turn-by-id lookup" — we explicitly store and retrieve
    # the project association. A missing metadata item returns 0 (no replay for
    # unknown turns).
    try:
        from app.platform import get_platform
        from app.main import get_project_manager

        store = get_platform().metadata_store
        metadata = await store.get_turn_metadata(turn_id)

        if metadata is None:
            # Turn metadata not found — may be a turn that predates metadata
            # persistence, one from another tenant, or a turn that never
            # started. Metadata is required here.
            logger.debug(
                "[ws] durable replay — turn metadata not found conn=%s turn_id=%s",
                conn.connection_id[:8], turn_id[:8]
            )
            return 0, since_seq

        # Resolve owner_id from the project
        project_id = metadata.get("project_id")
        if not project_id:
            logger.warning(
                "[ws] durable replay — turn metadata missing project_id conn=%s turn=%s",
                conn.connection_id[:8], turn_id[:8]
            )
            return 0, since_seq

        pm = get_project_manager()
        project = pm.get_project(project_id)
        owner_id = project.owner_id if project else None

        if owner_id is None or owner_id != conn.user_id:
            logger.warning(
                "[ws] durable replay rejected — owner mismatch conn=%s user=%s turn=%s",
                conn.connection_id[:8], conn.user_id, turn_id[:8]
            )
            return 0, since_seq
    except Exception as e:
        logger.warning(
            "[ws] durable replay — owner lookup failed conn=%s turn=%s: %s",
            conn.connection_id[:8], turn_id[:8], e
        )
        return 0, since_seq

    # Read events from the durable log
    try:
        events = await store.turn_events_since(turn_id, since_seq)
    except Exception as e:
        logger.warning(
            "[ws] durable replay — turn_events_since failed conn=%s turn=%s: %s",
            conn.connection_id[:8], turn_id[:8], e
        )
        return 0, since_seq

    # Enqueue each event on the connection, marking as replayed.
    # The events returned by turn_events_since already carry the DDB seq.
    # Mark them so consumers can distinguish durable replays from live events.
    count = 0
    last_seq = since_seq
    for event in events:
        marked = mark_replayed(event)
        try:
            conn.queue.put_nowait(marked)
            count += 1
            event_seq = event.get("seq")
            if isinstance(event_seq, int):
                last_seq = max(last_seq, event_seq)
        except asyncio.QueueFull:
            logger.warning(
                "[ws] durable replay queue full conn=%s stream=%s; truncated at seq=%s",
                conn.connection_id[:8], stream, event.get("seq"),
            )
            break

    if count > 0:
        logger.debug(
            "[ws] durable replay conn=%s stream=%s since_seq=%d count=%d",
            conn.connection_id[:8], stream, since_seq, count
        )

    return count, last_seq


async def _handle_inbound(conn: WsConnection, raw: str) -> None:
    """Dispatch an inbound text frame from the client.

    Currently the only recognized command is ``subscribe`` which asks
    the hub to replay buffered envelopes on a given stream past
    ``since_seq``. Malformed payloads or unknown actions are dropped
    silently — we never want a misbehaving client to be able to crash
    the route.

    Args:
        conn: The connection that sent the frame.
        raw: The raw text payload (we expect JSON).
    """
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return
    if not isinstance(msg, dict):
        return
    if msg.get("action") != "subscribe":
        return
    stream = msg.get("stream")
    since_seq = msg.get("since_seq", -1)
    if not isinstance(stream, str) or not isinstance(since_seq, int):
        return

    # Park live envelopes for this stream until the replay finishes, then flush
    # them in sequence order. Without the barrier a live frame published during
    # the replay's store round trips was enqueued AHEAD of the snapshot, the
    # client advanced its cursor to that later sequence, and every replayed frame
    # underneath it was discarded — losing chat text with no way to ask again.
    conn.begin_replay(stream)
    try:
        await _replay_stream(conn, stream, since_seq)
    finally:
        flushed = conn.end_replay(stream)
        if flushed:
            logger.debug(
                "[ws] flushed %d live envelope(s) held during replay conn=%s "
                "stream=%s",
                flushed, conn.connection_id[:8], stream,
            )


async def _replay_stream(conn: WsConnection, stream: str, since_seq: int) -> None:
    """Serve a subscribe's replay for one stream.

    Split out of :func:`_handle_inbound` so the replay barrier there wraps every
    branch, including the early returns.
    """
    # DDB-backed replay for job:<id> streams on the durable path.
    # When the requested stream is a job stream, source the replay from the DURABLE
    # DDB log via events_since, not the in-memory ws_hub provider. This enables
    # mid-run reconnect with no-duplicate / no-skip bars.
    #
    # Fallback: when the durable log serves ZERO events, fall through to the
    # in-memory replay provider (the Job.events buffer registered at job
    # creation). The durable log is only WRITTEN on the arch_split path
    # (Job.append_event gates seq allocation on arch_split), so on dev /
    # in-process the log is always empty and durable-only replay silently
    # loses every pre-subscribe event — a refresh mid-run rebuilt the bars
    # from nothing ("95/0 passed"). The count-based fallback (rather than an
    # arch_split gate) also covers a transient durable-read failure on the
    # cloud path: the in-memory buffer is a same-process superset there.
    if stream.startswith("job:"):
        durable_last_seq = since_seq
        try:
            _, durable_last_seq = await _replay_job_stream_from_durable_log(
                conn, stream, since_seq
            )
        except Exception:
            logger.exception(
                "[ws] durable replay failed conn=%s stream=%s since_seq=%d",
                conn.connection_id[:8], stream, since_seq,
            )
        # SEQ-MERGE: ALWAYS run the in-memory replay for the tail PAST what
        # the durable leg delivered — not just on zero durable count. A
        # PARTIALLY populated durable log (writer lagging, transient mid-run
        # write failure on the cloud path) used to suppress the in-memory
        # fallback entirely, permanently truncating the replayed stream.
        # Starting the in-memory replay at the durable leg's last seq means
        # no seq is ever re-sent (same seq domain per stream), so the
        # client's backwards-jump domain-reset heuristic can never misread
        # the merge as a new domain.
        try:
            await get_ws_hub().replay_to_connection(conn, stream, durable_last_seq)
        except Exception:
            logger.exception(
                "[ws] in-memory replay failed conn=%s stream=%s since_seq=%d",
                conn.connection_id[:8], stream, durable_last_seq,
            )
        return

    # DDB-backed replay for turn:<id> streams.
    # When the requested stream is a relay turn stream, source the replay from the
    # DURABLE turn event log via turn_events_since, not the in-memory ws_hub provider.
    # This enables mid-turn reconnect with live delta replay + cross-task correctness
    # (the replay reads DDB, so any coordinator task can serve it).
    if stream.startswith("turn:"):
        # Same zero-count fallback as the job: branch above, same reason:
        # the durable turn log is only written on the relay/arch_split path,
        # so on dev it is always empty and durable-only replay would lose the
        # whole in-flight turn on refresh — an empty assistant bubble and no
        # tool pills until the turn completes, after which the transcript
        # append restores the text but never the tools.
        durable_last_seq = since_seq
        try:
            _, durable_last_seq = await _replay_turn_stream_from_durable_log(
                conn, stream, since_seq
            )
        except Exception:
            logger.exception(
                "[ws] durable turn replay failed conn=%s stream=%s since_seq=%d",
                conn.connection_id[:8], stream, since_seq,
            )
        # SEQ-MERGE (see job: branch above): always replay the in-memory
        # tail past the durable leg's last seq, so a partial durable log
        # can't truncate the turn and no seq is ever double-sent.
        try:
            await get_ws_hub().replay_to_connection(conn, stream, durable_last_seq)
        except Exception:
            logger.exception(
                "[ws] in-memory turn replay failed conn=%s stream=%s "
                "since_seq=%d",
                conn.connection_id[:8], stream, durable_last_seq,
            )
        return

    # In-memory replay for every other stream
    try:
        await get_ws_hub().replay_to_connection(conn, stream, since_seq)
    except Exception:
        # Replay must never knock the connection out — the live publisher
        # path still works regardless of replay success.
        logger.exception(
            "[ws] replay failed conn=%s stream=%s since_seq=%d",
            conn.connection_id[:8], stream, since_seq,
        )


async def _pump_outbound(conn: WsConnection) -> None:
    """Forward envelopes from the connection's queue to the WebSocket.

    Runs as a background task per connection. Awaits ``queue.get`` and
    forwards each envelope via ``websocket.send_json``. Exits cleanly
    on cancellation (route's ``finally`` block) or when the underlying
    socket fails — failure mode is left to the surrounding receive
    loop, which observes ``WebSocketDisconnect`` next iteration.

    Args:
        conn: The :class:`WsConnection` to pump.
    """
    websocket = conn.websocket
    try:
        while True:
            envelope = await conn.queue.get()
            try:
                await websocket.send_json(envelope)
            except Exception as exc:
                # Connection is dead from the network's perspective.
                # The receive loop in ``websocket_endpoint`` will see
                # ``WebSocketDisconnect`` next iteration and tear us
                # down. Just log and exit cleanly here.
                logger.debug(
                    "[ws-pump] send failed conn=%s: %s",
                    conn.connection_id[:8], exc,
                )
                return
    except asyncio.CancelledError:
        # Task cancellation is the normal teardown path — re-raise so
        # the route's ``await pump_task`` completes.
        raise


__all__ = ["router"]
