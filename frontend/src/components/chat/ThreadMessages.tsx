/**
 * The thread's message list, remounted whenever a thread update would crash it.
 *
 * assistant-ui keys its message list, and each message's parts, by POSITION,
 * not id. A slot's mounted part components survive a thread update and re-read
 * whatever message now sits at their index. Measured against the real library
 * with this app's message components, three update shapes take the page down:
 *
 * - a slot keeps its role but its message has fewer parts — the part component
 *   past the new end reads out of bounds
 *   (`tapClientLookup: Index N out of bounds`);
 * - a slot keeps its role but holds a DIFFERENT message with a different part
 *   count — a part component re-reads a part of another type
 *   (`MessagePartText can only be used inside text or reasoning message parts`).
 *
 * - a slot's message grows, but a part of a different type lands at an index
 *   that is already mounted — the same `MessagePartText` error.
 *
 * All three arise whenever the thread is replaced wholesale from the stored
 * transcript, which can legitimately hold fewer or differently split parts than
 * the live view streamed. The update itself cannot be made safe by keeping the
 * two in step, so the list is remounted instead: fresh components mount against
 * the new messages and nothing stale is left to re-read.
 *
 * The decision reads assistant-ui's OWN thread state, not the provider's React
 * state. The runtime applies a new message list after the render that hands it
 * over, so a key bumped at that render would remount against the OLD messages
 * and leave the fresh list to receive the shrink a moment later.
 */
import { ThreadPrimitive, useAuiState } from "@assistant-ui/react";
import { useState } from "react";

import { AssistantMessage, SystemMessage, UserMessage } from "./ChatMessage";

/** What the remount decision needs from one rendered message. */
interface RenderedSlot {
  readonly id: string;
  readonly role: string;
  readonly content: readonly { readonly type: string }[];
}

/**
 * Whether replacing the rendered thread `before` with `after` must remount the
 * message list rather than update it in place.
 *
 * A slot is safe to update in place when every part index already mounted
 * keeps a part of the same type: the new message's part types must start with
 * the old message's. That one condition excludes all three crash shapes —
 * fewer parts, and a different type at a mounted index whether or not the
 * message changed. Only slots both lists share can crash: a slot whose role
 * changes swaps its message component, which remounts on its own, and slots
 * past the end of the shorter list simply unmount.
 *
 * Ordinary streaming appends parts and finalizing stores the same parts, so
 * both update in place and the open/closed state of the thread's tool groups
 * survives every turn.
 *
 * @param before - The messages assistant-ui rendered last.
 * @param after - The messages it is about to render.
 * @returns True when any shared slot would hit one of the crash shapes above.
 */
export function threadUpdateNeedsRemount(
  before: readonly RenderedSlot[],
  after: readonly RenderedSlot[],
): boolean {
  const shared = Math.min(before.length, after.length);
  for (let i = 0; i < shared; i++) {
    const was = before[i];
    const now = after[i];
    if (was.role !== now.role) continue;
    if (now.content.length < was.content.length) return true;
    for (let j = 0; j < was.content.length; j++) {
      if (now.content[j].type !== was.content[j].type) return true;
    }
  }
  return false;
}

/**
 * `ThreadPrimitive.Messages` with this app's message components, keyed so that
 * an update {@link threadUpdateNeedsRemount} flags mounts a fresh list.
 *
 * Uses React's "adjust state while rendering" pattern: the generation bump is
 * applied before any child renders, so the stale list never re-renders against
 * the new messages.
 */
export function ThreadMessages() {
  const messages = useAuiState((s) => s.thread.messages);
  const [rendered, setRendered] = useState(messages);
  const [generation, setGeneration] = useState(0);

  if (messages !== rendered) {
    setRendered(messages);
    if (threadUpdateNeedsRemount(rendered, messages)) {
      setGeneration((g) => g + 1);
    }
  }

  return (
    <ThreadPrimitive.Messages
      key={generation}
      components={{ UserMessage, AssistantMessage, SystemMessage }}
    />
  );
}
