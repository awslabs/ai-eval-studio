// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Thread replacements against the real assistant-ui.
 *
 * `ThreadMessages` claims to survive the thread updates that crash
 * assistant-ui's index-keyed message list. That is a claim about the library,
 * so it is tested against the library: each case renders a thread with this
 * app's real message components and `convertMessage`, replaces it, and records
 * what happened. Every crash shape runs twice — once through the bare
 * `ThreadPrimitive.Messages`, which must still crash (if an assistant-ui upgrade
 * stops it crashing, the remount may no longer be needed), and once through
 * `ThreadMessages`, which must render the new messages.
 */
import { Component, type ReactNode } from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AssistantRuntimeProvider,
  ThreadPrimitive,
  useExternalStoreRuntime,
} from "@assistant-ui/react";

import type { ChatMessage } from "@/lib/types";
import { convertMessage } from "./ChatRuntimeProvider";
import { AssistantMessage, SystemMessage, UserMessage } from "./ChatMessage";
import { ThreadMessages, threadUpdateNeedsRemount } from "./ThreadMessages";

/** Records the error that took the thread down, if any. */
class CrashRecorder extends Component<
  { onCrash: (message: string) => void; children: ReactNode },
  { crashed: boolean }
> {
  state = { crashed: false };

  static getDerivedStateFromError(): { crashed: boolean } {
    return { crashed: true };
  }

  componentDidCatch(error: Error): void {
    this.props.onCrash(error.message);
  }

  render(): ReactNode {
    return this.state.crashed ? null : this.props.children;
  }
}

type MessageList = "bare" | "remounting";

/** A thread rendered the way `ChatThread` renders it, with either message list. */
function Thread({ messages, list }: { messages: ChatMessage[]; list: MessageList }) {
  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage,
    isRunning: false,
    onNew: async () => {},
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root>
        <ThreadPrimitive.Viewport>
          {list === "bare" ? (
            <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage, SystemMessage }} />
          ) : (
            <ThreadMessages />
          )}
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

/**
 * Let the runtime apply an update. It does so after the render that hands it
 * the new messages, so without this every case reads "no crash" — including
 * the ones that do crash.
 */
function settle(): Promise<void> {
  return act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

/** Render each thread in turn and return the crash message, if any. */
async function replaceThread(list: MessageList, ...threads: ChatMessage[][]): Promise<string | null> {
  let crash: string | null = null;
  const onCrash = (message: string): void => {
    crash = message;
  };
  const view = (messages: ChatMessage[]) => (
    <CrashRecorder onCrash={onCrash}>
      <Thread messages={messages} list={list} />
    </CrashRecorder>
  );
  const { rerender } = render(view(threads[0]));
  await settle();
  for (const next of threads.slice(1)) {
    rerender(view(next));
    await settle();
  }
  return crash;
}

const CREATED = "2026-09-22T00:00:00Z";

function user(id: string): ChatMessage {
  return { id, role: "user", parts: [{ type: "text", text: "hi" }], created_at: CREATED };
}

function system(id: string): ChatMessage {
  return { id, role: "system", parts: [{ type: "text", text: "Iteration complete" }], created_at: CREATED };
}

/**
 * An assistant reply: `tools` DISTINCT tool rows, so none coalesce, then an
 * answer that names the reply so a test can see which one rendered.
 */
function reply(id: string, tools: number): ChatMessage {
  return {
    id,
    role: "assistant",
    parts: [
      ...Array.from({ length: tools }, (_, i) => ({
        type: "tool" as const,
        name: `tool_${i}`,
        input: { _detail: `detail ${i}` },
        result_preview: "ok",
        status: "ok" as const,
        truncated: false,
      })),
      { type: "text" as const, text: `answer from ${id}` },
    ],
    created_at: CREATED,
  };
}

/**
 * `msg` grown the way streaming grows a reply: `tools` more distinct tool rows
 * appended, then a closing answer. The closing text keeps the reply ending in
 * prose, as a real one does — text left between tool rows folds into the
 * collapsed work log and is not on screen.
 */
function appended(msg: ChatMessage, tools: number): ChatMessage {
  return {
    ...msg,
    parts: [
      ...msg.parts,
      ...Array.from({ length: tools }, (_, i) => ({
        type: "tool" as const,
        name: `late_tool_${i}`,
        input: { _detail: `late detail ${i}` },
        result_preview: "ok",
        status: "ok" as const,
        truncated: false,
      })),
      { type: "text" as const, text: `more from ${msg.id}` },
    ],
  };
}

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
  // The viewport auto-scrolls; jsdom has no scrollTo.
  Element.prototype.scrollTo = () => {};
  // React logs every error a boundary catches; the bare-list cases crash on
  // purpose. Crashes are asserted through CrashRecorder, not the console.
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** The update shapes that crash the bare list, with the error each produces. */
const CRASHING_UPDATES: {
  name: string;
  threads: ChatMessage[][];
  error: RegExp;
  shows: string;
}[] = [
  {
    // The occurrence from a live run: an orchestrator finalize adopted a
    // transcript whose reply at position 3 had 7 parts where 14 were mounted.
    name: "a slot keeps its role but its new message has fewer parts",
    threads: [
      [user("u1"), reply("r1", 2), system("s1"), reply("streamed-locally", 13)],
      [user("u1"), reply("r1", 2), system("s1"), reply("from-transcript", 6)],
    ],
    error: /^tapClientLookup: Index \d+ out of bounds/,
    shows: "answer from from-transcript",
  },
  {
    name: "the same message loses parts in place",
    threads: [
      [user("u1"), reply("r1", 6)],
      [user("u1"), reply("r1", 2)],
    ],
    error: /^tapClientLookup: Index \d+ out of bounds/,
    shows: "answer from r1",
  },
  {
    name: "a slot gets a different message with more parts",
    threads: [
      [user("u1"), reply("streamed-locally", 2)],
      [user("u1"), reply("from-transcript", 5)],
    ],
    error: /MessagePartText can only be used inside text or reasoning message parts/,
    shows: "answer from from-transcript",
  },
  {
    // Two orchestrator replies with no user message between them, and the
    // transcript drops the first: the second shifts into its slot.
    name: "an earlier message drops out and a later reply shifts into its slot",
    threads: [
      [user("u1"), reply("r1", 1), reply("r2", 4)],
      [user("u1"), reply("r2", 4)],
    ],
    error: /MessagePartText can only be used inside text or reasoning message parts/,
    shows: "answer from r2",
  },
  {
    // The answer was mounted at index 2; the stored reply has tool rows there.
    name: "the same message grows but a tool part lands where text was mounted",
    threads: [
      [user("u1"), reply("r1", 2)],
      [user("u1"), reply("r1", 4)],
    ],
    error: /MessagePartText can only be used inside text or reasoning message parts/,
    shows: "answer from r1",
  },
];

describe("updates that crash the bare message list", () => {
  for (const update of CRASHING_UPDATES) {
    it(`still crash it: ${update.name}`, async () => {
      expect(await replaceThread("bare", ...update.threads)).toMatch(update.error);
    });

    it(`render through ThreadMessages: ${update.name}`, async () => {
      expect(await replaceThread("remounting", ...update.threads)).toBeNull();
      expect(screen.getByText(update.shows)).toBeTruthy();
    });
  }

  it("two adopts in a row render through ThreadMessages", async () => {
    // Two orchestrator turns finalizing close together each adopt the transcript.
    const crash = await replaceThread(
      "remounting",
      [user("u1"), reply("streamed-a", 9)],
      [user("u1"), reply("stored-a", 3), reply("streamed-b", 8)],
      [user("u1"), reply("stored-a", 3), reply("stored-b", 2)],
    );
    expect(crash).toBeNull();
    expect(screen.getByText("answer from stored-b")).toBeTruthy();
  });
});

describe("updates the bare message list survives", () => {
  it("survive both lists, and ThreadMessages renders the last one", async () => {
    const u = user("u1");
    const r = reply("r1", 3);
    const threads = [
      [u, r, user("u2"), reply("r2", 5)],
      [u, r],
      [u, r, system("s1"), reply("r3", 2)],
      [u, r, system("s1"), appended(reply("r3", 2), 2)],
      [u, r, system("s1"), appended(reply("stored-r3", 2), 2)],
    ];
    expect(await replaceThread("bare", ...threads)).toBeNull();
    cleanup();
    expect(await replaceThread("remounting", ...threads)).toBeNull();
    expect(screen.getByText("more from stored-r3")).toBeTruthy();
  });
});

describe("threadUpdateNeedsRemount", () => {
  const slot = (id: string, role: string, ...types: string[]) => ({
    id,
    role,
    content: types.map((type) => ({ type })),
  });

  it("flags a same-role slot that loses parts", () => {
    expect(
      threadUpdateNeedsRemount(
        [slot("a", "assistant", "tool-call", "tool-call", "text")],
        [slot("a", "assistant", "tool-call", "text")],
      ),
    ).toBe(true);
  });

  it("flags a different part type at a mounted index, even as the message grows", () => {
    expect(
      threadUpdateNeedsRemount(
        [slot("a", "assistant", "tool-call", "text")],
        [slot("a", "assistant", "tool-call", "tool-call", "text")],
      ),
    ).toBe(true);
  });

  it("leaves streaming growth in place", () => {
    expect(
      threadUpdateNeedsRemount(
        [slot("streaming", "assistant", "text")],
        [slot("streaming", "assistant", "text", "tool-call")],
      ),
    ).toBe(false);
  });

  it("leaves a reply finalized under its stored id with the same parts in place", () => {
    expect(
      threadUpdateNeedsRemount(
        [slot("streaming", "assistant", "tool-call", "text")],
        [slot("stored", "assistant", "tool-call", "text")],
      ),
    ).toBe(false);
  });

  it("ignores role changes and slots past the shorter list", () => {
    expect(
      threadUpdateNeedsRemount(
        [slot("a", "assistant", "text", "text"), slot("b", "assistant", "text")],
        [slot("s", "system", "text")],
      ),
    ).toBe(false);
  });
});
