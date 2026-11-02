// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Singleton WebSocket client for the realtime envelope channel.
 *
 * One ``WebSocket`` per browser tab connects to the backend's
 * ``@app.websocket("/ws")`` route. Subscribers register a per-stream
 * listener; this module fans inbound envelopes out by ``stream`` so
 * each consumer receives only what it asked for.
 *
 * Protocol (matches backend ``ws_envelope`` and ``ws_hub``):
 *  - URL: ``<scheme>://<host>/ws``, where the scheme is ``wss`` in cloud
 *    and plain ``ws`` in dev — picked from ``window.location.protocol``,
 *    so an https page never downgrades. The URL carries no credential.
 *  - Auth: the FIRST frame written is ``{"action":"auth","token":<jwt>}``.
 *    The server answers ``auth_ok`` on the reserved ``_control`` stream,
 *    or closes with 4401. Dev mode ignores the token value.
 *  - Inbound envelope shape: ``{stream, seq, kind, data}``.
 *  - ``kind: "keepalive"`` envelopes are silently consumed — they exist
 *    only to keep the ALB / CloudFront connection past the 60s idle
 *    cutoff.
 *  - Outbound: the auth frame, then ``subscribe`` requests (below).
 *
 * Lifecycle:
 *  - **Lazy connect**: nothing happens until the first ``subscribe()``.
 *  - **Reconnect on drop**: exponential backoff from 250ms to 30s,
 *    reset when the server acknowledges the credential (``auth_ok``) —
 *    not on open, because the server accepts a socket before it
 *    authenticates it. Clean closes (code 1000) we initiated reconnect
 *    immediately (no backoff).
 *  - **Reconnect on token rotation**: ``auth.ts`` fires a
 *    ``tokenChanged`` event after silent renew; we close 1000 and
 *    reconnect with the new token.
 *  - **Per-stream ``since_seq`` replay**: we track the last
 *    seq seen per stream and send a single
 *    ``{"action":"subscribe","stream":"<id>","since_seq":N}`` text
 *    frame to the backend whenever a fresh subscription registers
 *    (or the WS reconnects with active subscriptions). The hub
 *    replays buffered envelopes past ``since_seq`` from any
 *    registered :class:`StreamReplayProvider` (chat turns + eval
 *    jobs both implement it). Duplicates that race past the snapshot
 *    are dropped by the dispatch path's ``seq <= lastSeq`` guard.
 */

import { authEvents, getAccessToken } from "./auth";
import { decideCursor } from "./ws-cursor";

/**
 * Realtime message envelope, identical in shape to the backend's
 * ``ws_envelope.make_envelope`` output.
 *
 * Replayed envelopes (delivered from a buffer on reconnect) carry an optional
 * ``replayed`` field set by the server. The field is omitted on live envelopes,
 * so absence means live. Momentary consumers — those treating an event as "this
 * is happening right now" — must check the marker and ignore replayed envelopes.
 *
 * Degraded envelopes (published when durable seq allocation fails) carry an
 * optional ``degraded`` field set by the server. The client bypasses its cursor
 * decision entirely for these — they are delivered without seq comparison.
 */
export interface Envelope {
  /** Stream id, e.g. ``turn:<id>``, ``job:<id>``, ``project:<id>``. */
  stream: string;
  /** Monotonic per-stream sequence (starts at 0 per process). */
  seq: number;
  /** Kind discriminator (e.g. ``text_delta``, ``progress``, ``project_event``). */
  kind: string;
  /** Kind-specific payload. */
  data: unknown;
  /** Set to true by the server on replayed envelopes; omitted on live ones. */
  replayed?: boolean;
  /** Set to true by the server when durable seq allocation failed; omitted when healthy. */
  degraded?: boolean;
  /**
   * Forward gap detected by the CLIENT during dispatch: seq exceeded lastSeq + 1.
   * Attached at dispatch time, not a wire field. Consumer should trigger a refetch.
   */
  gap?: { expectedNext: number; receivedSeq: number };
}

/** Listener invoked for every envelope on a subscribed stream. */
export type EnvelopeListener = (env: Envelope) => void;

/** Initial reconnect backoff after an abnormal close. */
const INITIAL_BACKOFF_MS = 250;
/** Hard cap on reconnect backoff (after repeated failures). */
const MAX_BACKOFF_MS = 30_000;
/**
 * Stream the server uses for connection-level frames (backend ``ws.py``
 * ``_CONTROL_STREAM``). No component subscribes to it; ``onmessage`` handles it.
 */
const CONTROL_STREAM = "_control";

/** Per-stream subscriber registry. */
const subscribers: Map<string, Set<EnvelopeListener>> = new Map();
/** Last seq delivered per stream — sent as ``since_seq`` when (re)subscribing. */
const lastSeqByStream: Map<string, number> = new Map();

/** Active WebSocket, or ``null`` between connections. */
let socket: WebSocket | null = null;
/** Pending reconnect timer, cleared on connect / disconnect. */
let reconnectTimerId: ReturnType<typeof setTimeout> | null = null;
/** Current backoff delay; reset to ``INITIAL_BACKOFF_MS`` on the server's ``auth_ok``. */
let currentBackoffMs = INITIAL_BACKOFF_MS;
/** Flag tracking whether ``disconnect()`` was called explicitly. */
let explicitlyDisconnected = false;
/** Whether the previous close was clean (initiated by us, e.g. token rotation). */
let lastCloseWasClean = false;
/** Listener bound on ``authEvents``; we keep the reference for cleanup. */
let tokenChangedListener: EventListener | null = null;

/**
 * Build the WebSocket URL from the current page location: the secure
 * ``wss`` scheme for an ``https:`` page, plain ``ws`` only when the page
 * itself is served over ``http:``.
 *
 * Carries NO credential. The token used to be a query parameter here, which put
 * it into every log that records a URL — uvicorn writes the full request line to
 * its access log, and that log is retained. It now travels in the first frame
 * instead; see ``sendAuthFrame``.
 *
 * @returns Full WebSocket URL.
 */
function buildWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

/**
 * Send the credential as the connection's first frame.
 *
 * The backend accepts the socket, then reads exactly one frame and expects this
 * shape. Until it validates, the connection is registered nowhere and receives
 * nothing, so this has to be the first thing written — before any ``subscribe``.
 *
 * An empty token is still sent rather than skipped. Dev mode ignores the value,
 * and in cloud mode the server closes with 4401, which is the same outcome as
 * sending nothing except that it happens immediately instead of after the
 * server's handshake deadline.
 *
 * @param ws - The freshly opened socket.
 * @param token - JWT, or the empty string when one could not be obtained.
 */
function sendAuthFrame(ws: WebSocket, token: string): void {
  ws.send(JSON.stringify({ action: "auth", token }));
}

/**
 * Open the WebSocket connection.
 *
 * Defensive against ``getAccessToken`` throwing — if it does, we still
 * connect with an empty token. Dev mode ignores the token entirely;
 * in cloud mode an empty token will be rejected (4401 close), and the
 * normal reconnect-with-backoff path takes over so the user gets a
 * fresh attempt once auth.ts fires ``tokenChanged``.
 *
 * The entry guards are checked TWICE, before and after the token await, and
 * both checks are load-bearing. ``socket`` is assigned only after the await,
 * so the entry check reads ``null`` for every caller that arrives while the
 * token is in flight: two components subscribing in the same tick each opened
 * a connection, and all but the last leaked — still open, still dispatching
 * into the same listeners, but unreferenced, so ``disconnect()`` could not
 * close it and it survived sign-out holding a revoked token. The same window
 * let a ``disconnect()`` during the await be overtaken by the connect it was
 * trying to cancel. React StrictMode's double mount makes the first case
 * reproduce on every dev load, but it is an amplifier, not the cause — any two
 * subscribers in one tick race identically in the built SPA.
 */
async function openSocket(): Promise<void> {
  if (socket && socket.readyState !== WebSocket.CLOSED) return;
  if (explicitlyDisconnected) return;

  let token = "";
  try {
    const result = getAccessToken();
    // ``auth.ts`` exports ``getAccessToken`` as ``async`` (returns
    // ``Promise<string>``), but the type declaration here is ``string``-
    // friendly so the stub case still works. Awaiting a non-promise
    // is a no-op, so this handles both shapes.
    token = (await result) ?? "";
  } catch {
    // Defensive: if auth.ts throws (e.g. /api/config not yet loaded),
    // fall back to no token. Dev mode ignores it; cloud mode will
    // 4401-close and we'll backoff-retry once auth is ready.
    token = "";
  }

  // Re-check both guards: awaiting the token yielded, so a ``disconnect()``
  // (sign-out) or another connect could have landed in between. Opening here
  // would reconnect with a revoked token, or leak a second socket.
  if (explicitlyDisconnected) return;
  if (socket && socket.readyState !== WebSocket.CLOSED) return;

  const ws = new WebSocket(buildWsUrl());
  socket = ws;

  ws.onopen = () => {
    // The backoff is NOT reset here. The server accepts the socket before it
    // authenticates it, so a socket about to be closed 4401 opens first — and
    // resetting on open kept a rejected token reconnecting every 250ms
    // indefinitely. The reset waits for ``auth_ok`` in ``onmessage``.
    lastCloseWasClean = false;

    // FIRST, before any subscribe: the server reads exactly one frame and
    // expects the credential in it. A subscribe sent ahead of this would be
    // read as the auth frame, fail to parse as one, and close the socket.
    sendAuthFrame(ws, token);

    // Ask the hub to replay any buffered envelopes for streams we
    // already have subscribers on — covers the page-refresh-mid-turn
    // and the network-blip cases. ``since_seq`` is the highest seq
    // we've seen for the stream, defaulting to -1 so a fresh client
    // gets the full buffer.
    for (const stream of subscribers.keys()) {
      requestReplay(ws, stream);
    }
  };

  ws.onmessage = (event: MessageEvent<string>) => {
    let envelope: Envelope;
    try {
      envelope = JSON.parse(event.data) as Envelope;
    } catch {
      // Malformed payload — drop silently. We control the publisher;
      // this only fires during local schema drift.
      return;
    }
    if (envelope.stream === CONTROL_STREAM) {
      // ``auth_ok`` means the server validated the credential, which makes
      // this — not ``onopen`` — the successful connect that resets the
      // backoff, so a later drop starts at 250ms again.
      if (envelope.kind === "auth_ok") currentBackoffMs = INITIAL_BACKOFF_MS;
      return;
    }
    dispatch(envelope);
  };

  ws.onerror = () => {
    // Browsers throw a generic ``Event`` here without details; the
    // useful info is in the subsequent ``onclose``. No-op so the
    // ``onclose`` path does the actual recovery.
  };

  ws.onclose = (event: CloseEvent) => {
    if (socket === ws) socket = null;

    // 1000 = normal close (we initiated, e.g. token rotation or
    // ``disconnect()``); 4401 = backend rejected our token.
    const wasClean = event.code === 1000 || lastCloseWasClean;


    if (explicitlyDisconnected) {
      // ``disconnect()`` was called — don't reconnect.
      return;
    }

    if (wasClean) {
      // Reconnect immediately (no backoff) — token-rotation path.
      lastCloseWasClean = false;
      scheduleReconnect(0);
    } else {
      scheduleReconnect(currentBackoffMs);
      currentBackoffMs = Math.min(currentBackoffMs * 2, MAX_BACKOFF_MS);
    }
  };
}

/**
 * Schedule the next reconnect attempt, replacing any pending timer.
 *
 * @param delayMs - Milliseconds to wait before opening a new socket.
 */
function scheduleReconnect(delayMs: number): void {
  if (reconnectTimerId !== null) {
    clearTimeout(reconnectTimerId);
  }
  reconnectTimerId = setTimeout(() => {
    reconnectTimerId = null;
    void openSocket();
  }, delayMs);
}

/**
 * Fan an envelope out to all listeners subscribed to its stream.
 *
 * Keepalive envelopes are silently consumed — they keep the connection
 * alive past ALB's 60s idle timeout but carry no semantic content.
 *
 * @param envelope - Inbound message from the WebSocket.
 */
function dispatch(envelope: Envelope): void {
  if (envelope.kind === "keepalive") return;

  // Degraded publish: server-set marker indicating durable seq allocation failed.
  // The envelope's seq can sort below the cursor and comparing them is what drops
  // the frame. The cursor is unusable for this envelope by construction, so we
  // route it BEFORE the cursor decision (same shape as replay_gap), fan out to
  // listeners normally so the payload is consumed, and return without touching
  // lastSeqByStream. The load-bearing difference from replay_gap: the envelope
  // itself is fanned out, so the eval-completion event (or any other critical
  // publish) is delivered to consumers, not replaced by a notice.
  if (envelope.degraded) {
    const listeners = subscribers.get(envelope.stream);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(envelope);
        } catch (err) {
          console.error("[ws-client] degraded envelope subscriber threw:", err);
        }
      }
    }
    return;
  }

  // replay_gap is metadata about servability, not a stream
  // event. Route it BEFORE the cursor decision (mirroring keepalive's early
  // return). After notifying listeners, advance lastSeqByStream to the gap
  // envelope's own seq. This is not fudging the cursor: the client DID receive
  // this envelope, everything below it was already known-unservable and is being
  // recovered by the refetch, and the server guarantees the gap seq exceeds the
  // since_seq the client asked from. Without this, the durable seq the hub burns
  // is a permanent client-side hole that fires a second, redundant gap signal on
  // the very next frame.
  if (envelope.kind === "replay_gap") {
    // Notify listeners so useBackendEvents can trigger a refetch
    const listeners = subscribers.get(envelope.stream);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(envelope);
        } catch (err) {
          console.error("[ws-client] replay_gap subscriber threw:", err);
        }
      }
    }
    // Advance cursor to the gap envelope's seq so the next frame is not read
    // as a second gap.
    lastSeqByStream.set(envelope.stream, envelope.seq);
    return;
  }

  // Cursor policy lives in ``ws-cursor`` as a pure function so it can be tested
  // without a socket. Three behaviors it encodes, all previously wrong here:
  //
  //  1. The cursor advances ONLY for an envelope we actually deliver. The old
  //     order advanced it before the zero-listener check, so an envelope arriving
  //     while its consumer was unmounted was marked seen and dropped — and then
  //     excluded from the reconnect replay, which is the root of the "state not
  //     rehydrating on in-app navigation" family.
  //  2. A large backwards seq jump is a DOMAIN change (the backend has an
  //     in-memory counter starting at 0 and a durable one starting at 1), not a
  //     duplicate. Read as a duplicate it silently killed every later frame on
  //     the stream, terminal frames included.
  //  3. A forward hole on the live socket (seq > lastSeq + 1) is detected and
  //     reported as a gap. The envelope is still delivered because it is real data;
  //     the gap is an ADDITIONAL signal that frames were missed, never a replacement.
  const listeners = subscribers.get(envelope.stream);
  const hasListeners = !!listeners && listeners.size > 0;
  const decision = decideCursor(
    lastSeqByStream.get(envelope.stream),
    envelope.seq,
    hasListeners,
  );
  if (!decision.deliver) return;
  if (decision.reason === "domain-reset") {
    console.warn(
      `[ws-client] seq domain reset on ${envelope.stream}: ` +
        `${lastSeqByStream.get(envelope.stream)} -> ${envelope.seq}. ` +
        "Accepting and following the new domain.",
    );
  }

  // If decideCursor reported a forward gap, attach it to the envelope for
  // consumers. Do NOT mutate the inbound object (it is shared across a user's
  // connections) — deliver a shallow copy carrying the client-derived gap field.
  let deliveredEnvelope: Envelope = envelope;
  if (decision.deliver && "gap" in decision && decision.gap) {
    deliveredEnvelope = { ...envelope, gap: decision.gap };
  }

  lastSeqByStream.set(envelope.stream, decision.nextSeq);
  if (!listeners) return;
  for (const listener of listeners) {
    try {
      listener(deliveredEnvelope);
    } catch (err) {
      // A misbehaving subscriber must not knock out other subscribers
      // for the same stream — log and continue.
      console.error("[ws-client] subscriber threw:", err);
    }
  }
}

/**
 * Send a ``subscribe`` request asking the hub to replay buffered
 * envelopes on ``stream`` past our last-seen seq.
 *
 * Called from ``onopen`` (for every active stream) and from
 * ``subscribe()`` (for the new stream when the socket is already
 * open). The two cases are disjoint — ``onopen`` runs once per
 * connect, fresh subscriptions on an already-open socket only
 * trigger this from inside ``subscribe()``.
 *
 * @param ws - Target socket (must be in ``OPEN`` state).
 * @param stream - Stream id to request replay for.
 */
function requestReplay(ws: WebSocket, stream: string): void {
  const sinceSeq = lastSeqByStream.get(stream) ?? -1;


  try {
    ws.send(JSON.stringify({ action: "subscribe", stream, since_seq: sinceSeq }));
  } catch {
    // Socket closed between readyState check and send — onclose will
    // fire and the next reconnect's onopen will retry the subscribe.
  }
}

/**
 * Attach the ``tokenChanged`` listener on first connect. Idempotent.
 *
 * On token rotation we close the socket cleanly (code 1000) so the
 * ``onclose`` path schedules an immediate reconnect that picks up the
 * new token via ``getAccessToken()``.
 */
function ensureTokenChangedListener(): void {
  if (tokenChangedListener !== null) return;
  tokenChangedListener = () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      lastCloseWasClean = true;
      socket.close(1000, "token-rotation");
      // ``onclose`` handles the reconnect with backoff=0.
    }
  };
  authEvents.addEventListener("tokenChanged", tokenChangedListener);
}

/** Public API surface for the singleton WebSocket client. */
export const wsClient = {
  /**
   * Forget the replay cursor for one stream so the NEXT subscription
   * requests the full buffer (``since_seq: -1``).
   *
   * Exists for the remount-during-live-turn case: this module outlives SPA
   * navigation, so a component that unmounts
   * (losing all its accumulated turn state) and remounts would resubscribe
   * with the OLD cursor — the hub replays nothing earlier, and the turn's
   * streamed content is simply gone until the turn ends. A hard refresh
   * "fixed" it only because it reset this module. Call this before
   * resubscribing a consumer that has provably lost its state.
   */
  resetStreamCursor(stream: string): void {
    lastSeqByStream.delete(stream);
  },

  /**
   * Subscribe a listener for envelopes whose ``stream`` matches exactly.
   *
   * The first subscribe call triggers a lazy connect. Multiple
   * subscriptions to the same stream are allowed; each listener is
   * called with every envelope on that stream.
   *
   * @param stream - Stream id, e.g. ``turn:<id>``, ``job:<id>``,
   *                 ``project:<id>``.
   * @param listener - Function invoked for each matching envelope.
   * @returns Unsubscribe function. Calling it removes only this
   *          listener — other subscribers for the same stream stay.
   */
  subscribe(stream: string, listener: EnvelopeListener): () => void {
    let bucket = subscribers.get(stream);
    if (!bucket) {
      bucket = new Set<EnvelopeListener>();
      subscribers.set(stream, bucket);
    }
    bucket.add(listener);

    // Lazy connect: open the socket on the first subscriber across
    // all streams. Re-subscribes after a tab idle don't reopen if
    // we're already connected.
    explicitlyDisconnected = false;
    ensureTokenChangedListener();
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      void openSocket();
    } else if (socket.readyState === WebSocket.OPEN) {
      // Socket already open and we just registered a new stream —
      // ask for buffered replay on this stream specifically. (The
      // ``onopen`` path handles all-at-once replay on a fresh
      // connect; this branch handles "subscribe to a stream after
      // the WS is up".)
      requestReplay(socket, stream);
    }

    return () => {
      const set = subscribers.get(stream);
      if (!set) return;
      set.delete(listener);
      if (set.size === 0) subscribers.delete(stream);
      // We intentionally do NOT close the socket when the last
      // subscriber unmounts — most disconnect/reconnect noise would
      // come from short-lived component unmounts (e.g. navigating
      // between project tabs). The socket is cheap to keep open
      // thanks to the keepalive envelope; closing it is the explicit
      // ``disconnect()`` operation.
    };
  },

  /**
   * Force-disconnect the socket and stop reconnecting.
   *
   * Idempotent — calling twice is safe. Used on sign-out (so we don't
   * keep a stale connection open with a revoked token).
   */
  disconnect(): void {
    explicitlyDisconnected = true;
    if (reconnectTimerId !== null) {
      clearTimeout(reconnectTimerId);
      reconnectTimerId = null;
    }
    if (socket) {
      try {
        socket.close(1000, "client-disconnect");
      } catch {
        /* already closed — ignore */
      }
      socket = null;
    }
    if (tokenChangedListener !== null) {
      authEvents.removeEventListener("tokenChanged", tokenChangedListener);
      tokenChangedListener = null;
    }
    subscribers.clear();
    lastSeqByStream.clear();
    currentBackoffMs = INITIAL_BACKOFF_MS;
  },

  /**
   * Current ``readyState`` of the underlying socket, or
   * ``WebSocket.CLOSED`` if no socket exists. Exposed for tests /
   * dev-tools introspection only.
   */
  getReadyState(): number {
    return socket ? socket.readyState : WebSocket.CLOSED;
  },
};
