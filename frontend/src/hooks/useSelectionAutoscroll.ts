// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Auto-scroll hook for text-selection drag in a scrollable viewport.
 * When the user drags a text selection to the top or bottom edge of the viewport,
 * this hook scrolls the viewport in that direction to extend the selection beyond
 * the visible area — standard chat-app drag-select behavior.
 */

import { useEffect, type RefObject } from "react";

/**
 * Distance from the top/bottom edge of the viewport within which autoscroll engages.
 * Closer proximity produces faster scroll (up to MAX_SCROLL_STEP_PX per frame).
 */
const EDGE_THRESHOLD_PX = 48;

/**
 * Maximum scroll distance per animation frame (cap on scrollTop delta).
 * Prevents deep-into-the-edge drags from teleporting the viewport; enforces smooth motion.
 */
const MAX_SCROLL_STEP_PX = 18;

/**
 * Auto-scroll the viewport while a text-selection drag approaches an edge.
 *
 * @param viewportRef Ref to the scrollable viewport element (the overflow-y-auto container).
 *
 * Behavior:
 *   - Engages ONLY during an active PRIMARY-button text-selection drag that started inside
 *     the viewport AND the selection is currently non-collapsed (re-checked each frame).
 *   - Scrolls up when the pointer is within EDGE_THRESHOLD_PX of the top edge, down when
 *     near the bottom edge.
 *   - Scroll speed scales linearly with edge proximity, capped at MAX_SCROLL_STEP_PX per frame.
 *   - Continues scrolling while the cursor is held still at an edge (driven by an rAF loop
 *     reading the last-known pointer Y, not by mousemove events — mousemove stops firing
 *     when the cursor doesn't move).
 *   - Stops when the mouse is released, the pointer moves away from the edge, or the selection
 *     collapses. Otherwise inert — never touches scrollTop when inactive, so assistant-ui's
 *     streaming autoScroll is unaffected.
 */
export function useSelectionAutoscroll(viewportRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    // Mutable drag state held in refs so the rAF loop reads live values without re-render / stale closures
    // (same idiom as useWsStream.ts — avoids subscription churn on state updates).
    const dragState = {
      active: false,
      lastPointerY: 0,
      rafId: null as number | null,
    };

    /**
     * rAF tick: scroll the viewport if the activation gate passes.
     *
     * WHY an rAF loop rather than scrolling directly in the mousemove handler:
     * mousemove stops firing when the cursor is held still. A cursor parked at an edge
     * must keep scrolling — the loop reads the last-known pointer Y and scrolls each frame
     * regardless of whether the mouse has moved since the last frame.
     */
    function tick(): void {
      // Re-arm the loop while the drag is active (idle loop applies zero delta if pointer
      // is outside the edge zone or selection is collapsed; simpler than teardown/re-arm logic).
      if (dragState.active) {
        dragState.rafId = requestAnimationFrame(tick);
      }

      // Activation gate — autoscroll only when ALL are true:
      //   1. Primary-button drag started inside the viewport (dragState.active set by mousedown).
      //   2. Button still held (cleared by mouseup).
      //   3. Non-collapsed text selection exists (re-checked each frame since the selection may
      //      start collapsed and grow during the drag).
      const sel = document.getSelection();
      if (!dragState.active || !sel || sel.isCollapsed) return;

      // Read the scroll node from the ref each frame (may be null on first render).
      const viewport = viewportRef.current;
      if (!viewport) return;

      const rect = viewport.getBoundingClientRect();
      const pointerY = dragState.lastPointerY;

      // Edge detection: distance from top/bottom edge.
      const distFromTop = pointerY - rect.top;
      const distFromBottom = rect.bottom - pointerY;

      let scrollDelta = 0;

      // Near the top edge → scroll up (negative delta).
      if (distFromTop >= 0 && distFromTop < EDGE_THRESHOLD_PX) {
        // Closer to the edge = larger step; scale linearly from zero at threshold to max at edge.
        const ratio = 1 - distFromTop / EDGE_THRESHOLD_PX;
        scrollDelta = -Math.min(MAX_SCROLL_STEP_PX * ratio, MAX_SCROLL_STEP_PX);
      }
      // Near the bottom edge → scroll down (positive delta).
      else if (distFromBottom >= 0 && distFromBottom < EDGE_THRESHOLD_PX) {
        const ratio = 1 - distFromBottom / EDGE_THRESHOLD_PX;
        scrollDelta = Math.min(MAX_SCROLL_STEP_PX * ratio, MAX_SCROLL_STEP_PX);
      }

      // Apply the scroll. The hook never touches scrollTop unless the activation gate passes,
      // so assistant-ui's streaming autoScroll is unaffected when no user drag is active.
      if (scrollDelta !== 0) {
        viewport.scrollTop += scrollDelta;
      }
    }

    /**
     * Primary-button mousedown INSIDE the viewport starts a drag.
     *
     * WHY gate on contains(e.target): we want to track only drags that originated inside
     * the viewport, but we attach the listener to `window` for a single add/remove point
     * (the mousemove and mouseup listeners also live on window so the drag works even when
     * the cursor leaves the viewport bounds).
     */
    function handleMouseDown(e: MouseEvent): void {
      if (e.button === 0 && viewportRef.current?.contains(e.target as Node)) {
        dragState.active = true;
        dragState.lastPointerY = e.clientY;
        // Start the rAF loop. If it's already armed from a previous drag, cancel the stale id.
        if (dragState.rafId !== null) cancelAnimationFrame(dragState.rafId);
        dragState.rafId = requestAnimationFrame(tick);
      }
    }

    /**
     * Track the latest pointer Y from mousemove so the rAF loop scrolls based on live position.
     * Listener on `window` so the drag continues when the cursor wanders outside the viewport.
     */
    function handleMouseMove(e: MouseEvent): void {
      dragState.lastPointerY = e.clientY;
    }

    /**
     * Mouse release stops the drag fully — tear down the active flag, clear the last-pointer ref,
     * and cancel any pending rAF (the loop stops re-arming itself once active is false, but
     * we cancel the in-flight frame for cleanliness).
     */
    function handleMouseUp(): void {
      dragState.active = false;
      dragState.lastPointerY = 0;
      if (dragState.rafId !== null) {
        cancelAnimationFrame(dragState.rafId);
        dragState.rafId = null;
      }
    }

    // Wire all listeners on `window` for a single add/remove point.
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    // Cleanup: remove all listeners and cancel any pending rAF so no leaks on unmount.
    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      if (dragState.rafId !== null) {
        cancelAnimationFrame(dragState.rafId);
      }
    };
  }, [viewportRef]);
}
