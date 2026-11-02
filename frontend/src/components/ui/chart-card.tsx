// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chart card wrapper with fullscreen/maximize support.
 *
 * Wraps a Plotly chart (or any content) in a Card with fullscreen capability.
 * The maximize action is triggered via a custom Plotly modebar button (defined
 * in plot.tsx) that finds the nearest `.chart-card-container` ancestor and
 * calls its `__chartCardMaximize` method.
 *
 * Press Esc or click the close button to exit fullscreen.
 */

import {
  Component,
  createContext,
  useContext,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import { Card } from "@/components/ui/card";
import { MaximizablePanel, type MaximizablePanelHandle } from "@/components/ui/maximizable-panel";

/**
 * True while the chart is rendered fullscreen. Chart children (e.g. ``Plot``)
 * read this to drop their fixed layout ``height`` so Plotly autosizes to the
 * fullscreen flex container instead of re-pinning the fixed height on resize.
 * Defaults to ``false`` so charts outside a maximized card are unaffected.
 */
const ChartMaximizedContext = createContext(false);

/** Hook for chart children to know whether they're rendered fullscreen. */
export function useChartMaximized(): boolean {
  return useContext(ChartMaximizedContext);
}

/**
 * Error boundary that catches Plotly render errors and shows a fallback
 * instead of crashing the entire Results/Analysis tab.
 */
class ChartErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: string | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center py-8 text-sm text-muted-foreground">
          Chart rendering error: {this.state.error}
        </div>
      );
    }
    return this.props.children;
  }
}

interface ChartCardProps {
  /** Chart content to render */
  children: ReactNode;
  /** Optional title shown in the fullscreen header */
  title?: string;
  /** Additional className for the card */
  className?: string;
}

/**
 * Card wrapper that supports fullscreen via a Plotly modebar button.
 * Exposes a `__chartCardMaximize` function on the DOM element so the
 * custom Plotly modebar button can trigger fullscreen from inside the chart.
 *
 * @param children - Chart content to render
 * @param title - Optional title shown in the fullscreen header bar
 * @param className - Additional CSS classes for the card
 */
export function ChartCard({ children, title, className = "" }: ChartCardProps) {
  const panelRef = useRef<MaximizablePanelHandle>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  /** Expose maximize function on the DOM element so Plotly custom button can find it.
   * Must re-run when the panel's maximized state changes because the Card DOM node
   * is remounted on each transition out of fullscreen. */
  useEffect(() => {
    if (cardRef.current && panelRef.current) {
      (cardRef.current as any).__chartCardMaximize = () => panelRef.current?.maximize();
    }
  }, [panelRef.current?.maximized]);

  return (
    <MaximizablePanel
      ref={panelRef}
      title={title}
      className="flex-1 p-6 chart-card-fullscreen"
      maximizedContent={
        <ChartMaximizedContext.Provider value={true}>
          {children}
        </ChartMaximizedContext.Provider>
      }
    >
      <Card ref={cardRef} className={`p-4 chart-card-container min-w-0 overflow-hidden ${className}`}>
        <ChartErrorBoundary>{children}</ChartErrorBoundary>
      </Card>
    </MaximizablePanel>
  );
}
