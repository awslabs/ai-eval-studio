// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Segmented control — a compact single-select pill group.
 *
 * Renders a muted track with one raised "pill" marking the active option,
 * mirroring the active-tab treatment of {@link TabsTrigger}. Use for small
 * mutually-exclusive view switches (e.g. report type, comparison filters)
 * where full Tabs would be too heavy. Fully keyboard-operable and themed.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/class-utils";

/** A single segmented-control option. */
export interface SegmentedOption<T extends string> {
  /** Value reported to ``onValueChange`` when selected. */
  value: T;
  /** Visible label (text and/or icon). */
  label: ReactNode;
}

/** Props for {@link SegmentedControl}. */
interface SegmentedControlProps<T extends string> {
  /** Currently-selected value. */
  value: T;
  /** Called with the new value when an option is activated. */
  onValueChange: (value: T) => void;
  /** Options rendered left-to-right. */
  options: SegmentedOption<T>[];
  /** Optional extra classes on the track. */
  className?: string;
  /** Accessible label for the group. */
  "aria-label"?: string;
}

/**
 * A muted-track segmented control with a raised active pill.
 *
 * @typeParam T - The string union of option values.
 */
export function SegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  className,
  "aria-label": ariaLabel,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        // Matches the Tabs primitive: inset "well" (dark track + border) so the
        // lighter active pill stands out in dark mode.
        "inline-flex h-9 items-center gap-0.5 rounded-lg bg-background border border-border p-1 text-muted-foreground",
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onValueChange(opt.value)}
            className={cn(
              "inline-flex h-full items-center justify-center gap-1.5 rounded-md px-3 text-sm font-medium whitespace-nowrap transition-all outline-none",
              "focus-visible:ring-3 focus-visible:ring-ring/50",
              "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
              active
                ? "bg-accent text-accent-foreground border border-border shadow-sm"
                : "hover:bg-accent/50 hover:text-foreground",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
