// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hook for deriving aggregate stats from ModelRun[] — used by ResultsTable
 * for its per-model column headers.
 */

import { useMemo } from "react";
import type { ModelRun } from "@/lib/types";

/** Aggregate statistics derived from a single ModelRun. */
export interface ModelRunStats {
  /** Total number of test cases in the run. */
  totalCases: number;
  /** Fraction of cases that passed (0.0–1.0). */
  passRate: number;
  /** Average response latency in milliseconds, or null if no latency data. */
  avgLatencyMs: number | null;
  /** Sum of input tokens across all cases. */
  totalInputTokens: number;
  /** Sum of output tokens across all cases. */
  totalOutputTokens: number;
  /** Sum of generation cost across all cases. */
  totalCost: number;
}

/**
 * Derive per-run aggregate stats from an array of ModelRun objects.
 *
 * Returns a Map keyed by the canonical full model id
 * (``ModelRun.hyperparameters.model``) so callers can look up stats
 * for any model. Display labels are produced by ``shortModel`` at
 * render time; this hook does not produce display strings.
 *
 * @param runs - Array of ModelRun objects from the selected eval run
 * @returns Map of model id → ModelRunStats
 */
export function useModelRunStats(runs: ModelRun[]): Map<string, ModelRunStats> {
  return useMemo(() => {
    const m = new Map<string, ModelRunStats>();
    for (const run of runs) {
      const cases = run.results;
      const passed = cases.filter((c) => c.success).length;

      const latencies = cases
        .map((c) => c.additional_metadata?.latency_ms as number)
        .filter((v) => v != null && v > 0);
      const avgLatencyMs = latencies.length
        ? latencies.reduce((a, b) => a + b, 0) / latencies.length
        : null;

      const totalInputTokens = cases.reduce(
        (s, c) => s + ((c.additional_metadata?.input_tokens as number) ?? 0),
        0,
      );
      const totalOutputTokens = cases.reduce(
        (s, c) => s + ((c.additional_metadata?.output_tokens as number) ?? 0),
        0,
      );
      const totalCost = cases.reduce(
        (s, c) => s + ((c.additional_metadata?.generation_cost as number) ?? 0),
        0,
      );

      m.set(run.hyperparameters.model, {
        totalCases: cases.length,
        passRate: cases.length > 0 ? passed / cases.length : 0,
        avgLatencyMs,
        totalInputTokens,
        totalOutputTokens,
        totalCost,
      });
    }
    return m;
  }, [runs]);
}
