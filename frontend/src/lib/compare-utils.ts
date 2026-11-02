// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Utility helpers shared across the compare/ package.
 *
 * Extracted from ComparePanel.tsx so they can be imported by any compare
 * subcomponent without creating circular dependencies.
 */

import type { EvalRunSummary, CompactModelRun } from "@/lib/types";
import { shortModel } from "@/lib/format-utils";

/** Format a run label from its summary for selector display.
 *
 * Model names go through ``shortModel`` against the canonical full id
 * (``ModelRunSummary.model``). */
export function runLabel(r: EvalRunSummary): string {
  const date = new Date(r.timestamp);
  const dateStr = `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  const models = r.model_runs.map((m) => shortModel(m.model)).join(", ");
  const totalCases = r.model_runs.reduce((s, m) => s + m.total, 0);
  return `${r.prompt_version} — ${models} (${totalCases} cases, ${dateStr})`;
}

/** Short label for a run: prompt version + date to distinguish same-prompt runs. */
export function shortRunLabel(r: EvalRunSummary): string {
  const pv = r.prompt_version || r._file.replace(/\.json$/, "");
  const date = new Date(r.timestamp);
  const dateStr = `${date.toLocaleDateString([], { month: "short", day: "numeric" })} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  return `${pv} (${dateStr})`;
}

/**
 * Filter compact model runs by active metrics, mirroring applyMetricFilter semantics.
 *
 * When activeMetrics is null (all metrics active), returns unchanged. Otherwise,
 * keeps only the metrics in the allow-set, recomputes case-level success from
 * survivors, and drops cases with zero surviving metrics (same behavior as the
 * dashboard's applyMetricFilter on full ModelRun[]).
 *
 * @param runs - Array of compact model runs to filter
 * @param activeMetrics - Set of metric names to keep, or null for all
 * @returns Filtered compact runs with narrowed metrics
 */
export function applyCompactMetricFilter(
  runs: CompactModelRun[],
  activeMetrics: string[] | null,
): CompactModelRun[] {
  if (activeMetrics === null) {
    return runs;
  }
  const allowSet = new Set(activeMetrics);
  return runs.map((mr) => ({
    ...mr,
    results: mr.results
      .map((r) => {
        const narrowed = r.metrics.filter((m) => allowSet.has(m.name));
        if (narrowed.length === 0) return null; // drop case
        const success = narrowed.every((m) => m.success);
        return { ...r, metrics: narrowed, success };
      })
      .filter((r): r is NonNullable<typeof r> => r !== null),
  }));
}
