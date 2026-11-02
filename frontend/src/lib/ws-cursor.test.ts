// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the WebSocket per-stream cursor policy.
 *
 * The behaviors under test are live-path defects of the "silent" kind — no
 * error, no log, just a UI that stops updating while the backend does
 * everything right.
 */
import { describe, expect, it } from "vitest";

import { SEQ_RESET_THRESHOLD, decideCursor } from "@/lib/ws-cursor";

describe("normal delivery", () => {
  it("delivers the first envelope on a stream and sets the cursor", () => {
    expect(decideCursor(undefined, 5, true)).toEqual({
      deliver: true,
      nextSeq: 5,
      reason: "advance",
    });
  });

  it("delivers strictly increasing seqs", () => {
    expect(decideCursor(4, 5, true)).toEqual({
      deliver: true,
      nextSeq: 5,
      reason: "advance",
    });
  });

  it("delivers a forward GAP without stalling", () => {
    // Gaps are normal: the hub replays past since_seq and the client may simply
    // never see intermediate frames. Waiting for them would hang the stream.
    expect(decideCursor(4, 40, true).deliver).toBe(true);
  });
});

describe("duplicate suppression", () => {
  it("drops an exact duplicate", () => {
    // Replay-on-reconnect races the live fan-out, so the same envelope arrives
    // twice. Rendering it twice duplicates text deltas in chat.
    expect(decideCursor(7, 7, true)).toEqual({
      deliver: false,
      reason: "duplicate",
    });
  });

  it("drops a small backwards reorder", () => {
    expect(decideCursor(7, 5, true)).toEqual({
      deliver: false,
      reason: "duplicate",
    });
  });

  it("does not move the cursor when dropping", () => {
    // A dropped envelope must leave the cursor alone or the next replay request
    // would ask from the wrong point.
    const d = decideCursor(7, 5, true);
    expect(d.deliver).toBe(false);
    expect(d).not.toHaveProperty("nextSeq");
  });
});

describe("zero-listener handling (the in-app-navigation bug)", () => {
  it("does NOT advance the cursor when nothing is subscribed", () => {
    // This is the root mechanism of "state not rehydrating on in-app
    // navigation". The old code advanced the cursor and THEN bailed on zero
    // listeners, so an envelope arriving while the consumer was unmounted was
    // marked seen and discarded — and the reconnect replay, which asks only for
    // seqs past the cursor, could never return it. Permanently lost.
    const d = decideCursor(3, 4, false);
    expect(d).toEqual({ deliver: false, reason: "no-listeners" });
    expect(d).not.toHaveProperty("nextSeq");
  });

  it("the same envelope IS delivered once a listener exists (replayable)", () => {
    // Corollary of the above: because the cursor never moved, a remounted
    // consumer receives the envelope it missed.
    expect(decideCursor(3, 4, false).deliver).toBe(false);
    expect(decideCursor(3, 4, true)).toEqual({
      deliver: true,
      nextSeq: 4,
      reason: "advance",
    });
  });

  it("zero listeners takes precedence over the duplicate check", () => {
    // Ordering matters: a duplicate arriving with no listeners must still report
    // no-listeners, so the caller never treats it as cursor-advancing.
    expect(decideCursor(9, 9, false).reason).toBe("no-listeners");
  });
});

describe("seq domain reset (the two-counter bug)", () => {
  it("accepts a large backwards jump as a new domain", () => {
    // The backend has two seq domains: in-memory make_envelope starts at 0 per
    // process, durable allocate_*_seq starts at 1 and persists. A deploy — or a
    // publisher switching legs mid-stream — emits a seq far below the cursor.
    // Read as a duplicate, EVERY later frame on the stream is dropped forever,
    // terminal frames included: the spinner hangs with the backend correct.
    const d = decideCursor(500, 1, true);
    expect(d).toEqual({ deliver: true, nextSeq: 1, reason: "domain-reset" });
  });

  it("resets the cursor DOWN so the new domain can advance", () => {
    // Keeping the old high cursor would drop the entire new domain until it
    // climbed past 500 — which for a fresh in-memory counter is never.
    const d = decideCursor(500, 1, true);
    expect(d.deliver && d.nextSeq).toBe(1);
  });

  it("treats exactly the threshold as a reset", () => {
    const d = decideCursor(SEQ_RESET_THRESHOLD, 0, true);
    expect(d.reason).toBe("domain-reset");
  });

  it("treats one below the threshold as a duplicate", () => {
    // The boundary is asserted from both sides so a future tweak to the constant
    // cannot silently turn ordinary reorders into resets (which would duplicate
    // renders on every replay race).
    const d = decideCursor(SEQ_RESET_THRESHOLD, 1, true);
    expect(d.reason).toBe("duplicate");
  });

  it("a fresh in-memory domain after a durable one recovers within one frame", () => {
    // End-to-end shape of the deploy scenario: durable seqs, then a restart
    // publishing from 0. Frame 0 resets, and subsequent frames advance normally
    // instead of being swallowed.
    let cursor: number | undefined = 842;
    const durableThenRestart = [0, 1, 2, 3];
    const delivered: number[] = [];
    for (const seq of durableThenRestart) {
      const d = decideCursor(cursor, seq, true);
      if (d.deliver) {
        delivered.push(seq);
        cursor = d.nextSeq;
      }
    }
    expect(delivered).toEqual([0, 1, 2, 3]);
  });
});

describe("sequence-level behavior", () => {
  it("a realistic replay-then-live stream delivers each envelope exactly once", () => {
    // Replay returns 1..3, then live delivers 2..5 (overlapping). Each seq must
    // reach the consumer exactly once.
    let cursor: number | undefined;
    const delivered: number[] = [];
    for (const seq of [1, 2, 3, 2, 3, 4, 5]) {
      const d = decideCursor(cursor, seq, true);
      if (d.deliver) {
        delivered.push(seq);
        cursor = d.nextSeq;
      }
    }
    expect(delivered).toEqual([1, 2, 3, 4, 5]);
  });

  it("envelopes missed while unmounted stay eligible after remount", () => {
    // Unmounted for seqs 2 and 3; remounted at 4. The cursor must still be 1, so
    // a replay request asks from 1 and 2..3 are recoverable — rather than the
    // cursor having silently jumped to 3.
    let cursor: number | undefined;
    let d = decideCursor(cursor, 1, true);
    cursor = d.deliver ? d.nextSeq : cursor;

    for (const seq of [2, 3]) {
      const skipped = decideCursor(cursor, seq, false);
      expect(skipped.deliver).toBe(false);
      // cursor deliberately not updated
    }
    expect(cursor).toBe(1);

    d = decideCursor(cursor, 2, true);
    expect(d.deliver).toBe(true);
  });
});

describe("forward gap detection (live-socket hole)", () => {
  it("reports a gap when delivered seq exceeds cursor + 1", () => {
    // The live-socket hole: the worker publishes seq=2 and seq=3 and neither
    // reaches the browser, so with the cursor at 1 the next envelope is seq=4.
    // Delivering it and advancing without a gap signal would never notice that
    // 2 and 3 were missing.
    const d = decideCursor(1, 4, true);
    // Narrow on the `deliver` discriminant before reading delivery-only fields:
    // the non-delivering variant carries no nextSeq. Asserting inside the guard
    // (rather than after a bare `expect(d.deliver).toBe(true)`) is what makes the
    // access type-safe, since a runtime expect does not narrow the static type.
    expect(d.deliver).toBe(true);
    if (!d.deliver) throw new Error("expected a delivering decision");
    expect(d.nextSeq).toBe(4);
    expect(d).toHaveProperty("gap");
    expect(d.gap).toEqual({ expectedNext: 2, receivedSeq: 4 });
  });

  it("reports NO gap for the first envelope on a stream", () => {
    // Nothing to compare against — the stream just started.
    const d = decideCursor(undefined, 10, true);
    expect(d.deliver).toBe(true);
    expect(d).not.toHaveProperty("gap");
  });

  it("reports NO gap for a contiguous advance", () => {
    const d = decideCursor(5, 6, true);
    expect(d.deliver).toBe(true);
    expect(d).not.toHaveProperty("gap");
  });

  it("reports NO gap for a duplicate", () => {
    const d = decideCursor(8, 8, true);
    expect(d.deliver).toBe(false);
    expect(d).not.toHaveProperty("gap");
  });

  it("reports NO gap for a domain reset", () => {
    // A large backwards jump is a domain change, not a hole — the frames
    // in between are from the old domain and deliberately not recovered.
    const d = decideCursor(500, 1, true);
    expect(d.deliver).toBe(true);
    expect(d.reason).toBe("domain-reset");
    expect(d).not.toHaveProperty("gap");
  });

  it("reports NO gap when there are no listeners (zero-listener drop)", () => {
    const d = decideCursor(1, 4, false);
    expect(d.deliver).toBe(false);
    expect(d).not.toHaveProperty("gap");
  });

  it("a gap is reported exactly once per hole (cursor advances onto the gap frame)", () => {
    // This is the structural bound: the cursor advances to the seq that carried
    // the gap, so the next contiguous frame reports no gap.
    let cursor: number | undefined = 1;
    const sequence = [4, 5]; // seq=2 and seq=3 were missed

    const results = sequence.map((seq) => {
      const d = decideCursor(cursor, seq, true);
      if (d.deliver) cursor = d.nextSeq;
      return d;
    });

    // First frame (4) reports the gap
    expect(results[0].deliver).toBe(true);
    expect(results[0]).toHaveProperty("gap");

    // Second frame (5) is contiguous after 4, so no gap
    expect(results[1].deliver).toBe(true);
    expect(results[1]).not.toHaveProperty("gap");
  });
});
