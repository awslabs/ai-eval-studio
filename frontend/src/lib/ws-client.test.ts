// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Tests for the WebSocket client's dispatch path.
 *
 * `ws-cursor.test.ts` covers the pure cursor policy. These tests cover the
 * WIRING in `ws-client` that the policy alone cannot prove: whether an envelope
 * actually reaches its stream listeners, and whether the module-private cursor
 * map is left in the right state afterwards. A lost eval-completion event turns
 * on exactly those two behaviors — a policy that decides correctly is worthless
 * if `dispatch` drops the frame before consulting it.
 *
 * `dispatch` is module-private, so these tests drive it the way the browser
 * does: through a fake `WebSocket` whose `onmessage` the test fires. That also
 * makes the tests prove the wiring end-to-end rather than a re-implementation
 * of it.
 */
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

import { wsClient, type Envelope } from "@/lib/ws-client";

/** The fake socket instance the client most recently constructed. */
let activeSocket: FakeWebSocket | null = null;
/** Every socket the client constructed, in order — one per real connection. */
let constructedSockets: FakeWebSocket[] = [];

/**
 * Minimal `WebSocket` stand-in.
 *
 * Reports OPEN and invokes `onopen` synchronously on construction so
 * `subscribe()` takes its connected path, and records outbound frames so a test
 * can assert the `since_seq` the client asks for.
 */
class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];

  constructor(public url: string) {
    activeSocket = this;
    constructedSockets.push(this);
    // Fire onopen on a microtask so the client finishes assigning handlers.
    void Promise.resolve().then(() => this.onopen?.());
  }

  send(payload: string): void {
    this.sent.push(payload);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

/** Deliver one envelope through the real socket path into `dispatch`. */
function deliver(envelope: Envelope): void {
  if (!activeSocket?.onmessage) {
    throw new Error("no open fake socket — subscribeAndOpen() did not complete");
  }
  activeSocket.onmessage({ data: JSON.stringify(envelope) } as MessageEvent<string>);
}

/** The current fake socket, asserted non-null for assertions on sent frames. */
function currentSocket(): FakeWebSocket {
  if (!activeSocket) throw new Error("no fake socket constructed");
  return activeSocket;
}

/** Build a `project_event` envelope on the given stream. */
function projectEvent(stream: string, seq: number, extra: Partial<Envelope> = {}): Envelope {
  return { stream, seq, kind: "project_event", data: { resource: "runs" }, ...extra };
}

const STREAM = "project:11111111";

describe("ws-client dispatch", () => {
  beforeEach(() => {
    activeSocket = null;
    constructedSockets = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    // disconnect() clears the subscriber registry AND the per-stream cursor map,
    // so each test starts from a clean cursor without exporting a test-only hook.
    wsClient.disconnect();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /**
   * Subscribe, then flush the microtask that fires `onopen`, so the socket is
   * OPEN and `dispatch` is reachable.
   */
  async function subscribeAndOpen(listener: (env: Envelope) => void): Promise<() => void> {
    const unsub = wsClient.subscribe(STREAM, listener);
    // openSocket() is async (it awaits the access token) and the fake fires
    // onopen on a microtask, so poll until the handlers are actually attached
    // rather than guessing a fixed number of microtask turns.
    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }
    if (!activeSocket?.onmessage) throw new Error("fake socket never opened");
    return unsub;
  }

  it("delivers a degraded envelope whose seq sorts BELOW the cursor", async () => {
    // The lost-completion case: the client's cursor is 1, and the worker's
    // eval-completion publish arrives with fallback seq=0 because no durable seq
    // was allocatable. Compared against the cursor it is a duplicate and is
    // dropped forever. The degraded marker is what must prevent that.
    const received: Envelope[] = [];
    await subscribeAndOpen((env) => received.push(env));

    deliver(projectEvent(STREAM, 1));
    expect(received.map((e) => e.seq)).toEqual([1]);

    deliver(projectEvent(STREAM, 0, { degraded: true }));

    expect(received).toHaveLength(2);
    expect(received[1].seq).toBe(0);
    expect(received[1].degraded).toBe(true);
    // The payload must survive: the requirement is that the event is DELIVERED,
    // not replaced by a notice that something was lost.
    expect(received[1].data).toEqual({ resource: "runs" });
  });

  it("does not let a degraded envelope move the cursor", async () => {
    // A degraded seq is not in the durable domain, so adopting it as the cursor
    // would corrupt every later comparison on the stream.
    const received: Envelope[] = [];
    await subscribeAndOpen((env) => received.push(env));

    deliver(projectEvent(STREAM, 5));
    deliver(projectEvent(STREAM, 0, { degraded: true }));
    // seq=6 is contiguous after 5. If the degraded envelope had moved the cursor
    // to 0, this would be read as a forward hole and report a spurious gap.
    deliver(projectEvent(STREAM, 6));

    expect(received.map((e) => e.seq)).toEqual([5, 0, 6]);
    expect(received[2].gap).toBeUndefined();
  });

  it("reports a forward gap once, then stays silent on contiguous frames", async () => {
    // The live-socket hole: seq 2 and 3 never arrive, and a client that only
    // compares against its cursor accepts 4 after 1 without noticing. Exactly
    // one gap must be reported, and the next contiguous frame must not report
    // another — otherwise every subsequent frame triggers a refetch.
    const received: Envelope[] = [];
    await subscribeAndOpen((env) => received.push(env));

    deliver(projectEvent(STREAM, 1));
    deliver(projectEvent(STREAM, 4));
    deliver(projectEvent(STREAM, 5));

    expect(received.map((e) => e.seq)).toEqual([1, 4, 5]);
    expect(received[1].gap).toEqual({ expectedNext: 2, receivedSeq: 4 });
    expect(received[0].gap).toBeUndefined();
    expect(received[2].gap).toBeUndefined();
  });

  it("advances the cursor onto a replay_gap envelope's own seq", async () => {
    // The hub allocates a durable seq for its cannot-serve envelope but never
    // records it in the shared buffer, so that seq is a permanent hole. If the
    // client did not adopt it, the very next frame would read as a forward gap
    // and fire a second, redundant refetch.
    const received: Envelope[] = [];
    await subscribeAndOpen((env) => received.push(env));

    deliver(projectEvent(STREAM, 1));
    deliver({
      stream: STREAM,
      seq: 7,
      kind: "replay_gap",
      data: { since_seq: 1, oldest_retained_seq: 4 },
    });
    // seq=8 is contiguous after the gap envelope's 7.
    deliver(projectEvent(STREAM, 8));

    expect(received.map((e) => e.kind)).toEqual([
      "project_event",
      "replay_gap",
      "project_event",
    ]);
    expect(received[2].gap).toBeUndefined();
  });

  it("keeps a degraded envelope out of the replay cursor it asks the server for", async () => {
    // The cursor is what the client sends as since_seq on reconnect. A degraded
    // seq must never become that value, or the server would replay from a
    // meaningless point in the durable domain.
    await subscribeAndOpen(() => {});

    deliver(projectEvent(STREAM, 9));
    deliver(projectEvent(STREAM, 0, { degraded: true }));

    currentSocket().sent = [];
    // Re-subscribing on an already-open socket re-requests replay, exposing the
    // cursor the client currently holds.
    wsClient.subscribe(STREAM, () => {});

    const request = currentSocket()
      .sent.map((f) => JSON.parse(f))
      .find((f) => f.action === "subscribe");
    expect(request).toBeDefined();
    expect(request.since_seq).toBe(9);
  });
});

describe("resetStreamCursor", () => {
  beforeEach(() => {
    activeSocket = null;
    constructedSockets = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    wsClient.disconnect();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("makes the next subscription replay from -1 instead of the stale cursor", async () => {
    /* The remount-during-live-turn bug: this module
     * survives SPA navigation, so a remounted consumer that lost all its
     * state resubscribed with the OLD cursor and the hub replayed nothing —
     * empty chat thread until the turn finalized. After resetStreamCursor,
     * the resubscription must ask for the full buffer (since_seq: -1). */
    const unsub = wsClient.subscribe(STREAM, () => {});
    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }
    deliver(projectEvent(STREAM, 5));
    unsub();

    // WITHOUT reset, a resubscription would send since_seq=5. Reset first:
    wsClient.resetStreamCursor(STREAM);
    wsClient.subscribe(STREAM, () => {});
    await Promise.resolve();

    const requests = currentSocket()
      .sent.map((f) => JSON.parse(f))
      .filter((f) => f.action === "subscribe" && f.stream === STREAM);
    expect(requests.length).toBeGreaterThanOrEqual(2);
    expect(requests[requests.length - 1].since_seq).toBe(-1);
  });

  it("leaves other streams' cursors untouched", async () => {
    const otherStream = "turn:other-turn";
    wsClient.subscribe(STREAM, () => {});
    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }
    const unsubOther = wsClient.subscribe(otherStream, () => {});
    deliver(projectEvent(STREAM, 3));
    deliver(projectEvent(otherStream, 7));
    unsubOther();

    wsClient.resetStreamCursor(STREAM);
    wsClient.subscribe(otherStream, () => {});
    await Promise.resolve();

    const requests = currentSocket()
      .sent.map((f) => JSON.parse(f))
      .filter((f) => f.action === "subscribe" && f.stream === otherStream);
    expect(requests[requests.length - 1].since_seq).toBe(7);
  });
});

describe("single-connection guarantee", () => {
  beforeEach(() => {
    activeSocket = null;
    constructedSockets = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    wsClient.disconnect();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /**
   * The module contract is "one WebSocket per browser tab". ``openSocket`` is
   * async because it awaits the access token, and it assigns the module-level
   * ``socket`` only AFTER that await — so its own ``if (socket)`` guard is still
   * reading ``null`` for every caller that arrives while the token is in
   * flight. Two components subscribing in the same tick therefore each opened a
   * connection, and the first was leaked: still open, still dispatching, no
   * longer referenced by ``socket``, so ``disconnect()`` could not close it.
   *
   * The symptom: every client holds TWO simultaneous connections and every
   * project event is delivered twice. React StrictMode's double
   * mount makes it reproduce every time in dev, but it is NOT the cause — any
   * two subscribers in one tick race identically in the built SPA, which is
   * the normal case when a page mounts several event-consuming components.
   */
  it("opens ONE socket when two subscribers arrive in the same tick", async () => {
    wsClient.subscribe("project:p1", () => {});
    wsClient.subscribe("turn:t1", () => {});

    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }

    expect(constructedSockets).toHaveLength(1);
  });

  it("leaves no socket open that disconnect() cannot close", async () => {
    // The real harm of the leak. ``disconnect()`` closes ``socket``, which only
    // ever pointed at the LAST connection opened, so a raced first socket stayed
    // open past sign-out — holding a connection with a token the user just
    // revoked, and still dispatching into the module's listeners.
    //
    // Asserted on close state rather than on duplicate delivery on purpose: the
    // cursor policy silently de-duplicates two frames carrying the same seq, so
    // a delivery-count assertion passes even while the socket is leaking.
    wsClient.subscribe("project:p1", () => {});
    wsClient.subscribe("turn:t1", () => {});

    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }

    wsClient.disconnect();

    const stillOpen = constructedSockets.filter(
      (s) => s.readyState !== FakeWebSocket.CLOSED,
    );
    expect(stillOpen).toHaveLength(0);
  });

  it("does not open a socket when disconnect() lands during the token await", async () => {
    // Sign-out while a connect is in flight. ``explicitlyDisconnected`` is
    // checked on entry, which is BEFORE the await, so without the post-await
    // re-check the connect it was cancelling still opened — reconnecting with
    // the token the user just revoked.
    wsClient.subscribe("project:p1", () => {});
    wsClient.disconnect();

    for (let i = 0; i < 50; i++) await Promise.resolve();

    const stillOpen = constructedSockets.filter(
      (s) => s.readyState !== FakeWebSocket.CLOSED,
    );
    expect(stillOpen).toHaveLength(0);
  });

  it("still reconnects after a close (the guards must not latch)", async () => {
    // A guard that never clears would make the client permanently unable to
    // reconnect — worse than the duplicate it fixes.
    wsClient.subscribe("project:p1", () => {});
    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }
    expect(constructedSockets).toHaveLength(1);

    const first = currentSocket();
    first.readyState = FakeWebSocket.CLOSED;
    first.onclose?.({ code: 1006, reason: "abnormal" } as CloseEvent);

    // Reconnect is scheduled on a timer; drive it.
    await vi.waitFor(() => expect(constructedSockets.length).toBe(2), {
      timeout: 2000,
    });
  });
});

describe("the credential never reaches the URL", () => {
  /**
   * Why: the token used to be a `token=` query parameter, so every layer that
   * records a URL recorded a live access token — uvicorn writes
   * the full request line to its access log, and that log is retained.
   *
   * Both halves have to hold. The server ignoring the parameter is asserted in
   * `backend/tests/test_ws_endpoint.py`; a server that ignores it still leaks it
   * if the client keeps sending one, which is what these assert.
   */
  beforeEach(() => {
    activeSocket = null;
    constructedSockets = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    wsClient.disconnect();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function openSocket(): Promise<FakeWebSocket> {
    wsClient.subscribe("stream-under-test", () => {});
    for (let i = 0; i < 50 && !activeSocket?.onmessage; i++) {
      await Promise.resolve();
    }
    if (!activeSocket) throw new Error("fake socket never opened");
    return activeSocket;
  }

  it("connects to a URL with no query string at all", async () => {
    const ws = await openSocket();
    expect(ws.url).not.toContain("token");
    expect(ws.url).not.toContain("?");
    expect(ws.url).toMatch(/\/ws$/);
  });

  it("sends the credential as the first frame, ahead of any subscribe", async () => {
    // Ordering is load-bearing: the server reads exactly ONE frame and expects
    // the credential in it, so a subscribe sent first would be read as the auth
    // frame, fail to parse as one, and close the socket.
    const ws = await openSocket();

    expect(ws.sent.length).toBeGreaterThan(0);
    const first = JSON.parse(ws.sent[0]) as { action?: string; token?: unknown };
    expect(first.action).toBe("auth");
    expect(first).toHaveProperty("token");

    const laterActions = ws.sent.slice(1).map((raw) => JSON.parse(raw).action);
    expect(laterActions).not.toContain("auth");
  });
});

describe("reconnect backoff", () => {
  /**
   * The server accepts a socket BEFORE it authenticates it, so `onopen` fires
   * even for a connection that is about to be closed 4401. Resetting the backoff
   * on open therefore kept a refused credential — an expired session whose
   * silent renew failed — reconnecting every 250ms for as long as the tab stayed
   * open. The reset belongs to the server's `auth_ok` acknowledgement.
   */
  beforeEach(() => {
    activeSocket = null;
    constructedSockets = [];
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    wsClient.disconnect();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /** Let the async connect finish: the token await, then the fake's `onopen`. */
  async function settle(): Promise<void> {
    for (let i = 0; i < 50; i++) await Promise.resolve();
  }

  /** Close the current socket the way the server refuses a credential. */
  function refuse(): void {
    const ws = currentSocket();
    ws.readyState = FakeWebSocket.CLOSED;
    ws.onclose?.({ code: 4401, reason: "auth required" } as CloseEvent);
  }

  it("keeps growing across repeated 4401 refusals", async () => {
    wsClient.subscribe("project:p1", () => {});
    await settle();
    expect(constructedSockets).toHaveLength(1);

    refuse(); // first refusal: the retry comes after 250ms
    await vi.advanceTimersByTimeAsync(250);
    await settle();
    expect(constructedSockets).toHaveLength(2);

    // The second socket OPENED before it was refused. That must not have reset
    // the delay, so the next retry waits 500ms, not 250ms.
    refuse();
    await vi.advanceTimersByTimeAsync(250);
    await settle();
    expect(constructedSockets).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(250);
    await settle();
    expect(constructedSockets).toHaveLength(3);
  });

  it("starts over once the server acknowledges the credential", async () => {
    wsClient.subscribe("project:p1", () => {});
    await settle();
    refuse(); // the next failure would now wait 500ms
    await vi.advanceTimersByTimeAsync(250);
    await settle();
    expect(constructedSockets).toHaveLength(2);

    deliver({ stream: "_control", seq: 0, kind: "auth_ok", data: {} });

    // A drop AFTER a successful handshake is a fresh failure: 250ms again.
    const ws = currentSocket();
    ws.readyState = FakeWebSocket.CLOSED;
    ws.onclose?.({ code: 1006, reason: "abnormal" } as CloseEvent);
    await vi.advanceTimersByTimeAsync(250);
    await settle();
    expect(constructedSockets).toHaveLength(3);
  });
});
