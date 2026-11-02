// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat thread component with message viewport, welcome screen, status indicator, and composer.
 * Uses assistant-ui primitives for message rendering, auto-scrolling, and scroll-to-bottom.
 */

import { AuiIf, ThreadPrimitive, useThreadRuntime } from "@assistant-ui/react";
import { FlaskConical, Loader2, AlertCircle, AlertTriangle, ChevronsDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ChatComposer } from "./ChatComposer";
import { ThreadMessages } from "./ThreadMessages";
import { Button } from "@/components/ui/button";
import {
  useAgentStatus,
  useHistoryState,
  useSessionInterrupted,
} from "./ChatRuntimeProvider";
import { useSelectionAutoscroll } from "@/hooks/useSelectionAutoscroll";

/** Suggested questions shown in the empty/welcome state. */
const SUGGESTED_QUESTIONS = [
  "Analyze my uploaded data",
  "Analyze data, recommend metrics, and create the eval harness",
  "Set up everything and start optimization automatically",
];

/**
 * Chat thread with auto-scrolling viewport, welcome state, and message composer.
 *
 * Layout contract:
 *   ThreadPrimitive.Root (flex column, fills available space)
 *     └─ ThreadPrimitive.Viewport (flex-1, scrolls)
 *         ├─ Welcome / Messages
 *         └─ ThreadPrimitive.ViewportFooter (sticky bottom-0)
 *             ├─ "New messages" button (absolute, floats above footer)
 *             ├─ HistoryFailedBanner
 *             ├─ SessionInterruptedBanner
 *             ├─ AgentStatusBar
 *             └─ ChatComposer
 *
 * Everything pinned to the bottom of the chat panel lives inside
 * ViewportFooter so the library measures its height and factors it into
 * auto-scroll math. Viewport's built-in autoScroll pauses when the user
 * scrolls up and resumes at the bottom — we don't maintain our own state.
 */
export function ChatThread() {
  const viewportRef = useRef<HTMLDivElement>(null);
  useSelectionAutoscroll(viewportRef);

  return (
    <ThreadPrimitive.Root className="flex flex-col flex-1 min-h-0 relative">
      <ThreadPrimitive.Viewport ref={viewportRef} className="flex flex-col flex-1 overflow-y-auto">
        {/* Message area padding lives on an inner wrapper so the sticky
         * footer below can span the full viewport width without margin
         * hacks. */}
        <div className="p-4 space-y-4 flex-1">
          <AuiIf condition={(s) => s.thread.isEmpty}>
            <EmptyState />
          </AuiIf>

          <ThreadMessages />
        </div>

        {/* Pinned-at-bottom region. bg-background covers messages that
         * scroll under the sticky footer. Its height is registered with
         * the viewport so "at bottom" accounts for the composer chrome. */}
        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 bg-background">
          <NewMessagesButton />
          <HistoryFailedBanner />
          <SessionInterruptedBanner />
          <AgentStatusBar />
          <ChatComposer />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}

/**
 * "New messages" pill rendered via ThreadPrimitive.ScrollToBottom.
 * The primitive renders nothing (disabled) when the viewport is already
 * at the bottom, so we don't need our own visibility tracking.
 */
function NewMessagesButton() {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <button
        type="button"
        className="
          absolute -top-10 left-1/2 -translate-x-1/2 z-10
          flex items-center gap-1.5 px-3 py-1.5 rounded-full
          bg-card border border-border shadow-lg
          text-xs text-muted-foreground hover:text-foreground hover:bg-accent
          transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring
          disabled:opacity-0 disabled:pointer-events-none
        "
      >
        <ChevronsDown className="size-3" />
        New messages
      </button>
    </ThreadPrimitive.ScrollToBottom>
  );
}

/** Displays agent status with detail about what it's doing + elapsed time. */
function AgentStatusBar() {
  const { state, detail } = useAgentStatus();
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number>(0);

  // Track elapsed time while agent is active
  useEffect(() => {
    if (state === "idle") {
      setElapsed(0);
      return;
    }
    startRef.current = startRef.current || Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => {
      clearInterval(id);
      // The only way into this cleanup with state !== "idle" is an unmount;
      // reset so a later mount starts from zero.
      startRef.current = 0;
    };
  }, [state]);

  if (state === "idle") return null;

  let label: string;
  if (state === "connecting") label = "Connecting to agent…";
  else if (state === "tool_running") label = detail || "Running tool…";
  else label = detail || "Thinking…";

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  const timeStr = elapsed >= 5 ? (mins > 0 ? `${mins}m ${secs}s` : `${secs}s`) : "";

  return (
    <div className="flex items-center gap-2 px-4 py-2 text-xs text-muted-foreground bg-muted/50 border-t border-border/40">
      <Loader2 className="size-3.5 animate-spin text-primary" />
      <span className="truncate font-medium flex-1">{label}</span>
      {timeStr && <span className="text-muted-foreground/50 font-mono text-2xs tabular-nums">{timeStr}</span>}
    </div>
  );
}

/**
 * Banner shown when chat-history hydration failed after the auto-retry.
 * Provides a manual retry button. Renders nothing while loading or once
 * the history is successfully loaded.
 */
function HistoryFailedBanner() {
  const { state, retry } = useHistoryState();
  if (state !== "failed") return null;
  return (
    <div className="flex items-center gap-2 px-4 py-2 text-xs bg-destructive/10 border-t border-destructive/30">
      <AlertTriangle className="size-3.5 text-destructive shrink-0" />
      <span className="text-destructive flex-1">
        Couldn't load chat history. The thread you see may be incomplete.
      </span>
      <button
        onClick={retry}
        className="text-xs font-medium px-2 py-0.5 rounded-md bg-destructive/20 text-destructive hover:bg-destructive/30 transition-colors shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        Retry
      </button>
    </div>
  );
}

/** Banner shown when the session was interrupted (e.g., page refresh mid-turn). */
function SessionInterruptedBanner() {
  const interrupted = useSessionInterrupted();
  const threadRuntime = useThreadRuntime();

  if (!interrupted) return null;

  return (
    <div className="flex items-center gap-2 px-4 py-2 text-xs bg-warning/10 border-t border-warning/30">
      <AlertCircle className="size-3.5 text-warning shrink-0" />
      <span className="text-warning flex-1">
        Session interrupted — the agent's previous turn was cut short.
      </span>
      <button
        onClick={() =>
          threadRuntime.append({
            role: "user",
            content: [{ type: "text", text: "Continue where you left off." }],
          })
        }
        className="text-xs font-medium px-2 py-0.5 rounded-md bg-warning/20 text-warning hover:bg-warning/30 transition-colors shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        Resume
      </button>
    </div>
  );
}

/**
 * Renders inside the empty thread region. Shows a loading indicator
 * while history is hydrating so a refresh on a populated chat doesn't
 * flash the welcome screen for the round-trip duration. Once the
 * load resolves we either render the welcome screen (fresh project)
 * or nothing (history loaded with messages — ``Messages`` above
 * already renders them and ``isEmpty`` flips false).
 */
function EmptyState() {
  const { state } = useHistoryState();
  if (state === "loading") {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-6 text-muted-foreground">
        <Loader2 className="size-5 animate-spin text-primary mb-2" />
        <p className="text-sm">Loading chat history…</p>
      </div>
    );
  }
  return <WelcomeScreen />;
}

/** Welcome screen with branding and clickable suggested questions. */
function WelcomeScreen() {
  const threadRuntime = useThreadRuntime();

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <FlaskConical className="size-10 text-primary mb-3" />
      <p className="text-lg font-medium">AI Eval Studio</p>
      <p className="text-sm text-muted-foreground mt-1 mb-6">
        Describe your evaluation goal to get started, or try a suggestion below.
      </p>
      <div className="flex flex-wrap gap-2 justify-center max-w-md">
        {SUGGESTED_QUESTIONS.map((question) => (
          <Button
            key={question}
            variant="secondary"
            size="sm"
            className="rounded-full whitespace-normal h-auto py-1.5 text-xs"
            onClick={() =>
              threadRuntime.append({
                role: "user",
                content: [{ type: "text", text: question }],
              })
            }
          >
            {question}
          </Button>
        ))}
      </div>
    </div>
  );
}
