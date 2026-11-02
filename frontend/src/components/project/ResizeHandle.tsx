// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Vertical drag handle for resizing a sidebar.
 * Reports horizontal drag deltas to the parent via ``onResize``.
 */

import { useCallback, useRef } from "react";

interface ResizeHandleProps {
  /** Called on every mousemove with the horizontal delta since last event. */
  onResize: (delta: number) => void;
}

/**
 * Thin vertical strip that captures mouse drag and emits the horizontal
 * delta since the previous mousemove, for sidebar width adjustments. Uses
 * document-level mousemove/mouseup listeners so dragging continues outside
 * the strip.
 */
export function ResizeHandle({ onResize }: ResizeHandleProps) {
  const dragging = useRef(false);
  const lastX = useRef(0);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    lastX.current = e.clientX;

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientX - lastX.current;
      lastX.current = ev.clientX;
      onResize(delta);
    };

    const onMouseUp = () => {
      dragging.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [onResize]);

  /** Keyboard resize: arrow keys nudge the boundary in 16px steps so the
   *  handle is operable without a mouse. */
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onResize(-16);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onResize(16);
      }
    },
    [onResize],
  );

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      tabIndex={0}
      className="w-1 shrink-0 cursor-col-resize bg-border/50 transition-colors outline-none hover:bg-primary/30 active:bg-primary/50 focus-visible:bg-primary/50"
      onMouseDown={onMouseDown}
      onKeyDown={onKeyDown}
    />
  );
}
