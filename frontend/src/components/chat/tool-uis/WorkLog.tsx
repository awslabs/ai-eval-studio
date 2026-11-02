// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Flat work log container for assistant messages.
 *
 * Wraps all interleaved work (thinking + tool calls + intermediate narration)
 * in ONE quiet collapsible region. Replaces the nested ProcessGroup container
 * with a flat, scannable column.
 *
 * While the turn runs the bubble shows ONLY the streaming work — no header, no
 * chevron — because the pinned status bar (ChatThread's AgentStatusBar) is the
 * single live indicator and an in-bubble header would just duplicate it. When
 * the turn completes the work folds into one calm header strip ("Thinking
 * Complete · N tool calls") with the answer rendered below it.
 */

import { useEffect, useState } from "react";
import { Sparkles, ChevronDown } from "lucide-react";
import { cn } from "@/lib/class-utils";
import { useAuiState } from "@assistant-ui/react";

/**
 * Props for the WorkLog container.
 */
interface WorkLogProps {
  /** The interleaved work steps (thinking + tools + intermediate text). */
  children: React.ReactNode;
}

/**
 * Props for {@link WorkLogHeaderContent}.
 */
interface WorkLogHeaderContentProps {
  /** Whether the turn contains any reasoning parts. */
  hasThinking: boolean;
  /** Number of tool-call parts in the turn. */
  toolCount: number;
}

/**
 * Header content (icon + label) for a COMPLETED work log. Names what KIND of
 * work happened — thinking is a mode, tool calls are discrete actions (counted):
 * - thinking only → "Thinking Complete"
 * - tools only    → "N tool calls"
 * - both          → "Thinking Complete · N tool calls"
 *
 * Only ever rendered once the turn is done; while running the pinned status bar
 * is the single live indicator and the bubble shows no header.
 */
function WorkLogHeaderContent({
  hasThinking,
  toolCount,
}: WorkLogHeaderContentProps) {
  const segments: string[] = [];
  if (hasThinking) segments.push("Thinking Complete");
  if (toolCount > 0) {
    segments.push(`${toolCount} tool call${toolCount === 1 ? "" : "s"}`);
  }
  // The work log only renders when there is ≥1 reasoning or tool part, so at
  // least one segment is always present; "Done" is an unreachable guard.
  const label = segments.join(" · ") || "Done";

  return (
    <>
      <Sparkles className="size-3 shrink-0 text-muted-foreground/60" />
      <span>{label}</span>
    </>
  );
}

/**
 * Renders one flat work log container holding all assistant work.
 *
 * Collapse behavior:
 * - While running: open, body-only (no header), live steps visible.
 * - On running → complete transition: settle collapsed to the header strip.
 *   Collapsing unmounts the body (children), which keeps long multi-turn
 *   histories from exhausting browser memory.
 * - After completion: the user can re-open / re-collapse; that choice sticks.
 *
 * The expanded state is keyed to the running phase. The effect re-syncs it ONLY
 * on the running↔complete transition (deps: [isRunning]), so a click made WHILE
 * streaming does not persist (the turn still settles collapsed when it
 * completes), while a click made AFTER completion sticks (isRunning no longer
 * changes, so the effect won't fire again). This separates "peek while it runs"
 * from "I want this finished turn open".
 *
 * Collapse timing is driven by message-level running status (read via
 * useAuiState), NOT by the group node's part.status which would collapse before
 * the answer starts streaming (creating an empty-bubble gap).
 */
export function WorkLog({ children }: WorkLogProps) {
  // Read message-level status and parts from assistant-ui's state.
  // WorkLog renders inside MessagePrimitive.Root so s.message resolves.
  const parts = useAuiState((s) => s.message.parts);
  const isRunning = useAuiState((s) => s.message.status?.type === "running");

  // Thinking is a mode (does it think?); tool calls are discrete actions (how
  // many?). The completed-state header names each kind distinctly. The tool
  // count sums the per-row repeat counts (_count) set by the data-layer
  // coalescing, so "N tool calls" reflects the REAL number of calls, not the
  // number of collapsed rows.
  const hasThinking = parts.some((p) => p.type === "reasoning");
  const toolCount = parts.reduce((n, p) => {
    if (p.type !== "tool-call") return n;
    const c = (p.args as Record<string, unknown> | undefined)?._count;
    return n + (typeof c === "number" ? c : 1);
  }, 0);

  // Collapse on completion REGARDLESS of trailing-answer state. A collapsed
  // turn unmounts its children (only the header strip stays in the DOM), which
  // is essential for memory: a long autonomous run produces dozens of turns,
  // many ending on a tool with no trailing answer, and keeping those expanded
  // mounts every thinking block + tool row at once — that balloons the DOM
  // until the browser tab OOM-crashes ("Aw, Snap"). The user can click any
  // finished turn to expand it. The effect re-syncs only on the
  // running↔complete transition, so a manual toggle after completion sticks.
  const [expanded, setExpanded] = useState(isRunning);
  useEffect(() => {
    setExpanded(isRunning);
  }, [isRunning]);
  const onToggle = () => setExpanded((prev) => !prev);

  // While running: body only. The pinned status bar carries the live label, so
  // the bubble stays a quiet stream of work with no duplicate header.
  if (isRunning) {
    return (
      <div className="mt-1 mb-3 rounded-md bg-muted/30">
        <div className="pb-1.5 pt-0.5">{children}</div>
      </div>
    );
  }

  // Completed: one calm collapsible header strip; body shows only when expanded.
  return (
    <div className="mt-1 mb-3 rounded-md bg-muted/30">
      {/* Header: label flush-left (same edge as body rows + the answer below),
          chevron pushed to the right. No left gutter, so nothing is indented
          past the answer prose. */}
      <button
        type="button"
        aria-expanded={expanded}
        onClick={onToggle}
        className="flex w-full items-center gap-1.5 px-1.5 py-1.5 text-left text-xs text-muted-foreground rounded-md transition-colors outline-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring"
      >
        <WorkLogHeaderContent hasThinking={hasThinking} toolCount={toolCount} />
        <ChevronDown
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground/60 transition-transform",
            !expanded && "-rotate-90"
          )}
        />
      </button>
      {/* No horizontal padding here: the body rows (ToolUIWrapper, the reasoning
          block) carry their own px-1.5, so the header's px-1.5 and the rows land
          on the same left edge. */}
      {expanded && <div className="pb-1.5 pt-0.5">{children}</div>}
    </div>
  );
}
