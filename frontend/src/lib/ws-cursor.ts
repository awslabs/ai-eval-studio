// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure per-stream cursor policy for the WebSocket client.
 *
 * Extracted from `ws-client`'s module-private `dispatch` so the dedup/replay
 * decision is testable without a socket, and so the two defects below are stated
 * as policy rather than buried in a fan-out loop.
 *
 * **Defect 1 — the cursor advanced before the listener check.** The old order was:
 * drop-if-seen, advance cursor, then bail when the stream has zero listeners. So
 * an envelope arriving while its consumer was unmounted (in-app navigation away
 * from a tab) was marked SEEN and then thrown away. On remount the client asked
 * the hub to replay only past that cursor, so the envelope could never be
 * recovered — the "state not rehydrating on in-app navigation" family. Nothing
 * about that is specific to one screen, which is why it kept resurfacing.
 * Fix: only advance the cursor for an envelope we actually deliver.
 *
 * **Defect 2 — a seq regression was indistinguishable from a duplicate.** The
 * backend has two seq domains: the in-memory counter starts at 0 per process, the
 * durable DynamoDB counter starts at 1 and persists. A deploy or a publisher
 * switching legs can therefore emit a seq far BELOW the cursor. Treated as a
 * duplicate, every subsequent frame on that stream is dropped forever — including
 * terminal frames, so the UI hangs with the backend having done everything right.
 * Fix: a regression larger than {@link SEQ_RESET_THRESHOLD} is read as a domain
 * change, and the cursor resets to follow the new domain. A small regression stays
 * a duplicate, which is what replay-vs-live races actually produce.
 *
 * The threshold is a heuristic and deliberately asymmetric: wrongly resetting
 * costs one duplicated render, wrongly holding the cursor costs a permanently
 * dead stream. The server-side half of this — one seq domain per stream, so
 * regressions stop happening at all — is the real fix; this is the client's
 * defense in depth.
 *
 * **Defect 3 — a forward hole on the live socket went unnoticed.** Envelopes can
 * go missing on a socket that never drops: say the worker publishes seq=2 and
 * seq=3 and neither reaches the browser, so with the cursor at 1 the next
 * envelope is seq=4. The old code accepted seq=4 straight after seq=1 and
 * advanced, never noticing 2 and 3 were missing. The `replay_gap` envelope only
 * fires when the client REQUESTS an unservable range on reconnect, so on a
 * socket that never dropped nothing fires. Fix: when `lastSeq` is defined and
 * `seq > lastSeq + 1`, report the forward gap alongside delivering the
 * envelope. The envelope is still
 * delivered because it is real data; the gap is an ADDITIONAL signal that frames
 * were missed, never a replacement. The honesty rule: a forward jump means
 * envelopes were missed, and the client's answer is a refetch — the same answer
 * already accepted for the reconnect path, applied to delivery.
 */

/**
 * Backwards jump treated as a new seq domain rather than a duplicate.
 *
 * Sized well above any plausible replay-vs-live reorder (those are a handful of
 * frames) and well below the gap a domain switch produces (a live turn's durable
 * seq is routinely in the hundreds while a fresh in-memory counter restarts near
 * 0).
 */
export const SEQ_RESET_THRESHOLD = 32;

/** What the caller should do with an inbound envelope. */
export type CursorAction =
  /** Deliver it and advance the cursor. */
  | {
      deliver: true;
      nextSeq: number;
      reason: "advance" | "domain-reset";
      /** Forward gap detected: envelope still delivered, but frames were missed. */
      gap?: { expectedNext: number; receivedSeq: number };
    }
  /** Drop it; the cursor does not move. */
  | { deliver: false; reason: "duplicate" | "no-listeners" };

/**
 * Decide whether to deliver an envelope and whether to move the stream cursor.
 *
 * @param lastSeq - Highest seq delivered on this stream, or undefined if none.
 * @param seq - The inbound envelope's seq.
 * @param hasListeners - Whether any consumer is currently subscribed.
 * @returns The action to take.
 */
export function decideCursor(
  lastSeq: number | undefined,
  seq: number,
  hasListeners: boolean,
): CursorAction {
  // No listeners: drop WITHOUT advancing. Advancing here is what made an
  // envelope arriving during unmount permanently unreplayable.
  if (!hasListeners) return { deliver: false, reason: "no-listeners" };

  if (lastSeq === undefined) return { deliver: true, nextSeq: seq, reason: "advance" };

  if (seq > lastSeq) {
    // Forward jump: deliver and advance. If the jump is non-contiguous (seq >
    // lastSeq + 1), also report the gap so the caller can trigger a refetch.
    // The envelope is still delivered because it is real data; the gap is an
    // ADDITIONAL signal, never a replacement.
    if (seq > lastSeq + 1) {
      return {
        deliver: true,
        nextSeq: seq,
        reason: "advance",
        gap: { expectedNext: lastSeq + 1, receivedSeq: seq },
      };
    }
    return { deliver: true, nextSeq: seq, reason: "advance" };
  }

  // seq <= lastSeq. Either a duplicate (replay racing live) or a domain change.
  if (lastSeq - seq >= SEQ_RESET_THRESHOLD) {
    return { deliver: true, nextSeq: seq, reason: "domain-reset" };
  }

  return { deliver: false, reason: "duplicate" };
}
