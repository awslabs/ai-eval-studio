// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Resizable chat sidebar panel with drag-to-resize and full-screen support.
 * Pushes main content left when open as sidebar, or covers main content in full-screen.
 */

import { useCallback, useRef, useState } from "react";
import { X, MessageSquare, Maximize, Minimize } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useChat } from "@/lib/chat-context";
import { ChatRuntimeProvider } from "./ChatRuntimeProvider";
import { ChatThread } from "./ChatThread";

const MIN_PANEL_WIDTH = 360;
const DEFAULT_PANEL_WIDTH = 580;
const MAX_PANEL_WIDTH_VW = 70;

interface ChatPanelProps {
  /** Project UUID to associate chat messages with (optional — no project context on non-project pages) */
  projectId?: string;
}

/** Resizable chat panel with drag handle, full-screen toggle, and close button. */
export function ChatPanel({ projectId }: ChatPanelProps) {
  const { isOpen, isFullScreen, close, toggleFullScreen } = useChat();
  const [panelWidth, setPanelWidth] = useState(DEFAULT_PANEL_WIDTH);
  const panelWidthRef = useRef(panelWidth);
  const isDraggingRef = useRef(false);

  panelWidthRef.current = panelWidth;

  /** Start drag-to-resize on mouse down on the left edge handle. */
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    const startX = e.clientX;
    const startWidth = panelWidthRef.current;
    const maxWidth = window.innerWidth * (MAX_PANEL_WIDTH_VW / 100);

    const onMouseMove = (moveEvent: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const delta = startX - moveEvent.clientX;
      const newWidth = Math.min(maxWidth, Math.max(MIN_PANEL_WIDTH, startWidth + delta));
      setPanelWidth(newWidth);
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  if (!isOpen) return null;

  // Full-screen: fixed position covering main content area (below topbar, right of sidebar)
  // Sidebar: relative position with configurable width
  const panelClassName = isFullScreen
    ? "fixed top-12 left-48 right-0 bottom-0 z-50 flex flex-col bg-background border-l"
    : "relative border-l flex flex-col bg-background shrink-0";

  return (
    <div
      className={panelClassName}
      style={isFullScreen ? undefined : { width: panelWidth }}
    >
      {/* Drag handle — only in sidebar mode */}
      {!isFullScreen && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/20 active:bg-primary/30 z-10"
          onMouseDown={handleMouseDown}
        />
      )}

      {/* Panel header */}
      <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
        <div className="flex items-center gap-2">
          <MessageSquare className="size-4" />
          <span className="text-sm font-medium">Chat</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleFullScreen}
            className="size-7"
            title={isFullScreen ? "Exit full screen" : "Full screen"}
          >
            {isFullScreen ? <Minimize className="size-3.5" /> : <Maximize className="size-3.5" />}
          </Button>
          <Button variant="ghost" size="icon" onClick={close} className="size-7" title="Close chat">
            <X className="size-4" />
          </Button>
        </div>
      </div>

      {/* Chat requires a project context to function */}
      {projectId ? (
        <ChatRuntimeProvider projectId={projectId}>
          <ChatThread />
        </ChatRuntimeProvider>
      ) : (
        <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground p-4">
          Select a project to start chatting.
        </div>
      )}
    </div>
  );
}
