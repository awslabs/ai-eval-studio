// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generic fullscreen overlay primitive with imperative maximize control.
 *
 * Provides a reusable fullscreen panel pattern: when maximized, renders
 * a fixed overlay covering the whole screen with header chrome (title +
 * close button), Esc-to-close, and body-scroll-lock. When not maximized,
 * renders children directly with no wrapper.
 *
 * Exposes an imperative handle via forwardRef so parents can trigger
 * fullscreen from outside React's render tree (e.g., DOM-attached event
 * handlers like Plotly modebar custom buttons).
 */

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useImperativeHandle,
  useState,
  type ReactNode,
} from "react";
import { Maximize, Minimize } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * True while the panel content is rendered in the fullscreen overlay.
 * Content reads this (via {@link useMaximized}) to hide its own maximize
 * affordance in fullscreen — the overlay header already owns the close
 * control. Defaults to ``false`` so content outside a panel is unaffected.
 */
const MaximizedContext = createContext(false);

/** Hook for panel content to know whether it's rendered fullscreen. */
export function useMaximized(): boolean {
  return useContext(MaximizedContext);
}

/** Imperative handle for controlling panel maximize state from parent refs. */
export interface MaximizablePanelHandle {
  /** Open the panel in fullscreen mode */
  maximize: () => void;
  /** Close fullscreen mode */
  minimize: () => void;
  /** Current maximized state (read-only) */
  readonly maximized: boolean;
}

interface MaximizablePanelProps {
  /**
   * Title shown in the fullscreen header bar.
   * If omitted, no title is displayed (only the close button).
   */
  title?: string;
  /**
   * Additional CSS classes for the maximized body wrapper.
   * Use this to control fullscreen content layout (e.g., flex flex-col h-full).
   */
  className?: string;
  /**
   * Content to render. When not maximized, rendered directly with no wrapper.
   * When maximized, rendered in the overlay body (unless maximizedContent is provided).
   */
  children: ReactNode;
  /**
   * Optional separate content to render in the fullscreen overlay body.
   * If provided, children is rendered only in the resting state and
   * maximizedContent is rendered in the fullscreen body. If omitted,
   * children is rendered in both states.
   */
  maximizedContent?: ReactNode;
}

/**
 * Fullscreen overlay panel with imperative maximize control.
 *
 * Use via ref to call maximize()/minimize() from outside the component tree.
 * When maximized, renders a fixed overlay with header chrome. When not
 * maximized, renders children directly.
 *
 * @param title - Optional title for the fullscreen header
 * @param className - Additional classes for the maximized body wrapper
 * @param children - Content to render (resting state, or both states if no maximizedContent)
 * @param maximizedContent - Optional separate content for fullscreen body
 */
export const MaximizablePanel = forwardRef<MaximizablePanelHandle, MaximizablePanelProps>(
  ({ title, className = "", children, maximizedContent }, ref) => {
    const [maximized, setMaximized] = useState(false);

    const maximize = useCallback(() => setMaximized(true), []);
    const minimize = useCallback(() => setMaximized(false), []);

    /** Expose imperative handle to parent via ref */
    useImperativeHandle(ref, () => ({ maximize, minimize, maximized }), [maximize, minimize, maximized]);

    /** Esc to close fullscreen */
    useEffect(() => {
      if (!maximized) return;
      const handler = (e: KeyboardEvent) => {
        if (e.key === "Escape") minimize();
      };
      window.addEventListener("keydown", handler);
      return () => window.removeEventListener("keydown", handler);
    }, [maximized, minimize]);

    /** Prevent body scroll when maximized */
    useEffect(() => {
      if (maximized) {
        document.body.style.overflow = "hidden";
      } else {
        document.body.style.overflow = "";
      }
      return () => {
        document.body.style.overflow = "";
      };
    }, [maximized]);

    if (maximized) {
      return (
        <MaximizedContext.Provider value={true}>
          <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col">
            <div className="flex items-center justify-between px-6 py-3 border-b">
              {title && <h3 className="text-sm font-medium">{title}</h3>}
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={minimize}
                className="ml-auto"
                title="Close fullscreen"
                aria-label="Close fullscreen view"
              >
                <Minimize className="size-3.5" />
              </Button>
            </div>
            <div className={`flex-1 ${className}`}>
              {maximizedContent ?? children}
            </div>
          </div>
        </MaximizedContext.Provider>
      );
    }

    return <MaximizedContext.Provider value={false}>{children}</MaximizedContext.Provider>;
  }
);

MaximizablePanel.displayName = "MaximizablePanel";

/** Props for {@link PanelMaximizeButton}. */
export interface PanelMaximizeButtonProps {
  /** Imperative panel handle to maximize on click. */
  panelRef: React.RefObject<MaximizablePanelHandle | null>;
  /** Accessible label for the toggle (e.g. "Open results table fullscreen"). */
  label: string;
}

/**
 * Ghost icon button that maximizes its panel. Renders nothing while the panel
 * is already maximized (the overlay header owns the close control), so the
 * same content subtree can be reused in both resting and fullscreen states
 * without showing two competing controls.
 *
 * MUST be rendered inside a {@link MaximizablePanel} so {@link useMaximized}
 * reads the correct state.
 *
 * @param panelRef - handle to the enclosing MaximizablePanel
 * @param label - accessible label for the fullscreen toggle
 */
export function PanelMaximizeButton({ panelRef, label }: PanelMaximizeButtonProps) {
  const maximized = useMaximized();
  if (maximized) return null;
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={() => panelRef.current?.maximize()}
      className="ml-auto"
      title="Fullscreen"
      aria-label={label}
    >
      <Maximize className="size-3.5" />
    </Button>
  );
}
