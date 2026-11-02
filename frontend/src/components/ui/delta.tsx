// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline delta indicator: arrow + signed value + unit suffix.
 *
 * Used by both the eval compare tab (originally lived in
 * ``SummaryTable.tsx``) and the report's model standings table to keep
 * the visual language identical across the app.
 *
 * - Hidden under |0.1| so a flat metric doesn't render "−0.0".
 * - ``invert`` flips the good/bad mapping: pass-rate higher = good
 *   (default), latency-ms higher = bad (set ``invert``).
 * - Styling is intentionally subtle (``text-2xs``, 70% opacity) so the
 *   primary number stays the focal point.
 */

interface DeltaProps {
  value: number | null | undefined;
  /** Unit suffix appended after the number, e.g. ``"%"`` or ``"ms"``. */
  suffix?: string;
  /** True when smaller is better (latency, cost). Default: bigger is better. */
  invert?: boolean;
  /** Optional extra classes (rare). */
  className?: string;
}

export function Delta({ value, suffix = "", invert = false, className = "" }: DeltaProps) {
  if (value === null || value === undefined || Math.abs(value) < 0.1) return null;
  const isGood = invert ? value < 0 : value > 0;
  const color = isGood ? "text-success/70" : "text-error/70";
  const arrow = value > 0 ? "▲" : "▼";
  return (
    <span className={`text-2xs ml-1 ${color} ${className}`}>
      {arrow}{value > 0 ? "+" : ""}{value.toFixed(1)}{suffix}
    </span>
  );
}
