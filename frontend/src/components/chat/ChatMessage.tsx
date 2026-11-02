// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat message components for user and assistant messages.
 *
 * User messages render as plain text; assistant messages render with a
 * flat work log (when work exists) holding all interleaved thinking +
 * tool steps + intermediate narration in true order. While the turn runs,
 * everything (including the forming answer) streams inside the open log so
 * nothing moves mid-stream. When the turn completes, the trailing answer
 * lifts out as clean prose below a collapsed "Thinking Complete · N tool
 * calls" strip. Pure-text replies (no thinking, no tools) render prose with
 * NO work log.
 */

import {
  MessagePrimitive,
  ActionBarPrimitive,
  useMessagePartText,
  useMessagePartReasoning,
  useAuiState,
} from "@assistant-ui/react";
import { Copy, Check, Lightbulb, Info } from "lucide-react";
import { useState, useCallback, createContext, useContext } from "react";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { TOOL_UI_REGISTRY, DefaultToolUI, type ToolUIProps } from "./tool-uis";
import { WorkLog } from "./tool-uis/WorkLog";

/**
 * WorkContext — true when a text part is in-log narration (render muted),
 * false (default) when it is trailing answer (render MarkdownText).
 * The group-work case wraps children in WorkContext.Provider value={true};
 * trailing answer leaves render at root outside the provider (default false).
 */
const WorkContext = createContext(false);

/**
 * Plain text renderer for user messages.
 * Uses useMessagePartText() to read text from assistant-ui's message part context.
 */
function PlainText() {
  const { text } = useMessagePartText();
  return <p className="whitespace-pre-wrap">{text}</p>;
}

/** User message displayed as a right-aligned bubble. */
export function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[85%] rounded-lg px-4 py-3 text-sm bg-primary text-primary-foreground">
        <MessagePrimitive.Content components={{ Text: PlainText }} />
      </div>
    </MessagePrimitive.Root>
  );
}

/**
 * System message displayed as a subtle, full-width event line.
 * Lifecycle-triggered orchestrator messages (optimization start/resume/cancel)
 * render here — distinct from chat bubbles, visually separated.
 */
export function SystemMessage() {
  return (
    <MessagePrimitive.Root className="flex items-center justify-center py-3">
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border bg-muted/50 text-xs text-muted-foreground">
        <Info className="w-3.5 h-3.5 flex-shrink-0" />
        <MessagePrimitive.Content components={{ Text: PlainText }} />
      </div>
    </MessagePrimitive.Root>
  );
}

/**
 * Dispatches a single tool call to its specialized UI component.
 * Looks up the tool name in the registry; falls back to DefaultToolUI.
 */
function ToolCallDispatcher({ toolName, args, result, isError }: ToolUIProps) {
  const Component = TOOL_UI_REGISTRY[toolName] || DefaultToolUI;
  return <Component toolName={toolName} args={args} result={result} isError={isError} />;
}

/**
 * Compute the index of the last work step (reasoning or tool-call).
 * Returns -1 if no steps exist (pure-text reply).
 *
 * WHY: The group node's `part.status` mirrors the last contained part and
 * would cause the container to collapse BEFORE the answer starts streaming.
 * We split by index instead — everything at or before lastStepIdx is inside
 * the work log, everything after is the trailing answer (rendered as prose
 * leaves outside the group).
 */
function computeLastStepIdx(parts: readonly { type: string }[]): number {
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].type === "reasoning" || parts[i].type === "tool-call") {
      return i;
    }
  }
  return -1;
}

/**
 * Text part renderer. MUST be defined at module scope (not inside the
 * GroupedParts render callback) so its component identity is stable across
 * streaming flushes. A component defined inside render is a new type every
 * frame, so React unmounts + remounts the node on each flush — which made the
 * streaming text flash / disappear / reappear. Hoisted, the fiber persists and
 * the text simply appends.
 *
 * While running, every text part renders as prose (uniform — no per-token style
 * flip). Once the turn completes, confirmed in-log narration renders muted.
 */
function WorkAwareText() {
  const inLog = useContext(WorkContext);
  const running = useAuiState((s) => s.message.status?.type === "running");
  const { text } = useMessagePartText();
  if (inLog && !running) {
    // Confirmed in-log narration: px-1.5 to share the body edge.
    return (
      <p className="px-1.5 text-xs text-muted-foreground whitespace-pre-wrap">
        {text}
      </p>
    );
  }
  // Answer (or any text while running): prose, inset px-1.5 so its left edge
  // matches the work-log strip above.
  return (
    <div className="px-1.5">
      <MarkdownText />
    </div>
  );
}

/**
 * Reasoning ("Thinking") block renderer. Hoisted to module scope for the same
 * stable-identity reason as {@link WorkAwareText} — otherwise the streaming
 * thinking text remounts every flush and flickers. A hairline top rule sets
 * each block off from the tool rows above it; the `first:` variant drops the
 * rule when a thinking block opens the log so it sits flush under the header.
 */
function ReasoningBlock() {
  const { text } = useMessagePartReasoning();
  return (
    <div className="px-1.5 py-1 text-xs mt-1 pt-2 border-t border-border/30 first:mt-0 first:pt-1 first:border-t-0">
      <div className="flex items-center gap-1.5 text-muted-foreground font-medium">
        <Lightbulb className="size-3 shrink-0 text-muted-foreground/60" />
        <span>Thinking</span>
      </div>
      <div className="text-muted-foreground whitespace-pre-wrap mt-1">
        {text}
      </div>
    </div>
  );
}

/** Assistant message displayed as a left-aligned bubble with markdown,
 *  flat work log (when work exists), and a hover copy action. */
export function AssistantMessage() {
  const isRunning = useAuiState((s) => s.message.status?.type === "running");

  // groupBy decides which parts live inside the work log vs render as the
  // answer below it.
  //
  // WHILE RUNNING we deliberately do NOT split: every part stays in the open
  // work log, in true order. The forming answer streams there too. This is the
  // fix for the mid-stream flashing — the backend tags narration and the final
  // answer as the SAME text kind, so any running split is a guess that flips on
  // every token (a text chunk renders as the big answer, then a tool arrives and
  // it gets yanked into the log as narration, then back out…). Keeping everything
  // in the log while running means nothing moves, so nothing flashes.
  //
  // ON COMPLETE we split once: the trailing text run lifts OUT as the answer
  // (prose below), the rest collapses into the "Thinking Complete · N" strip.
  const groupBy = useCallback(
    (_part: { type: string }, index: number, parts: readonly { type: string }[]) => {
      const lastStepIdx = computeLastStepIdx(parts);

      // No reasoning/tool work at all → pure-text reply → no log (every part is
      // the answer), whether running or done.
      if (lastStepIdx === -1) return null;

      // Running + work present: keep EVERYTHING in the open log (no split, no
      // movement, no flash). The trailing answer streams inside the log.
      if (isRunning) return ["group-work"] as const;

      // Completed: parts after the last step are the answer — leaves OUTSIDE
      // the (now collapsed) log.
      if (index > lastStepIdx) return null;

      // Completed work parts (reasoning, narration, tool-calls) → one flat group.
      return ["group-work"] as const;
    },
    [isRunning]
  );

  return (
    <MessagePrimitive.Root className="flex justify-start">
      <div className="group relative max-w-[95%] min-w-0 overflow-hidden rounded-lg px-4 py-3 text-sm bg-muted border border-border/50">
        <MessagePrimitive.GroupedParts groupBy={groupBy}>
          {({ part, children }) => {
            switch (part.type) {
              case "group-work": {
                // Flat work log: wrap children in WorkContext.Provider so
                // in-log text parts know they're inside the log (they render as
                // muted narration once the turn completes; as prose while running).
                return (
                  <WorkContext.Provider value={true}>
                    <WorkLog>{children}</WorkLog>
                  </WorkContext.Provider>
                );
              }
              case "text":
                // Stable module-level component (see WorkAwareText) so the
                // streaming text node persists across flushes instead of
                // remounting — the fix for the mid-stream flashing.
                return <WorkAwareText />;
              case "reasoning":
                // Stable module-level component (see ReasoningBlock).
                return <ReasoningBlock />;
              case "tool-call":
                // Flat tool row: prefer the ui registered via makeAssistantToolUI
                // if any; otherwise dispatch through our in-repo registry.
                if ("toolUI" in part && part.toolUI) return part.toolUI;
                return (
                  <ToolCallDispatcher
                    toolName={part.toolName}
                    args={part.args as Record<string, unknown>}
                    result={part.result}
                    isError={part.isError}
                  />
                );
              default:
                return null;
            }
          }}
        </MessagePrimitive.GroupedParts>
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
}

/** Hover-reveal copy button for assistant messages. */
function AssistantActionBar() {
  const [copied, setCopied] = useState(false);

  return (
    <ActionBarPrimitive.Root className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
      <ActionBarPrimitive.Copy
        onClick={() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }}
        className="p-1 rounded text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
        title="Copy message"
      >
        {copied ? (
          <Check className="h-3.5 w-3.5 text-success" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </ActionBarPrimitive.Copy>
    </ActionBarPrimitive.Root>
  );
}
