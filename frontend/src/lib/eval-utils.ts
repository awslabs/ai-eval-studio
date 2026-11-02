// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Utility functions for eval data transformation.
 * Pivots DeepEval ModelRun[] into a grid with one row per test case
 * and per-model columns.
 */

import type { ModelRun, MultiModelCase, TestResult, MetricData } from "./types";

/**
 * Narrow each case's metrics_data + recompute success against the
 * active metric filter.
 *
 * This is the single point where the metric dimension of the global
 * filter bar is applied to data — every chart, table, and tab below
 * the filter bar reads the narrowed result, so the headline
 * pass-rate, status filter, Compare tab regression buckets, and
 * everything else stay consistent.
 *
 * Behavior:
 *   - ``activeMetrics === null`` → return runs unchanged (no filter active).
 *   - Otherwise: drop metrics not in ``activeMetrics`` from each
 *     case's ``metrics_data``; recompute ``success`` from what's
 *     left; **drop cases with zero surviving metrics**. The drop is
 *     load-bearing — without it, ``[].every() === true`` would mark
 *     a metric-less case as passed and inflate the headline pass
 *     rate the moment a user filters to a metric the case doesn't
 *     report.
 *
 * @param runs - The full set of model runs to narrow.
 * @param activeMetrics - Metric names to keep (null = no filter).
 * @returns A new array with new case objects; inputs are not mutated.
 */
export function applyMetricFilter(
  runs: ModelRun[],
  activeMetrics: string[] | null,
): ModelRun[] {
  if (activeMetrics === null) return runs;
  const allow = new Set(activeMetrics);
  return runs.map((mr) => ({
    ...mr,
    results: mr.results.flatMap((r) => {
      const narrowed = r.metrics_data.filter((md) => allow.has(md.name));
      if (narrowed.length === 0) return [];
      const success = narrowed.every((md) => md.success);
      return [{ ...r, metrics_data: narrowed, success }];
    }),
  }));
}

/**
 * Pivot ModelRun[] into MultiModelCase[] for grid display.
 * Groups results by case_id across all models so each row shows
 * every model's output for the same test case side by side.
 */
export function pivotToGrid(modelRuns: ModelRun[]): MultiModelCase[] {
  const caseMap = new Map<string, MultiModelCase>();

  for (const run of modelRuns) {
    for (let i = 0; i < run.results.length; i++) {
      const r = run.results[i];
      const meta = r.additional_metadata ?? {};
      const caseId = String(meta.id ?? r.name ?? `case_${i}`);

      if (!caseMap.has(caseId)) {
        caseMap.set(caseId, {
          case_id: caseId,
          category: (meta.category as string) ?? null,
          input: r.input,
          expected_output: r.expected_output,
          context: r.context ?? null,
          retrieval_context: r.retrieval_context ?? null,
          models: {},
        });
      }

      // Per-case map is keyed by canonical full model id; consumers
      // (ResultsTable et al.) shorten via ``shortModel`` at render.
      caseMap.get(caseId)!.models[run.hyperparameters.model] = {
        success: r.success,
        actual_output: r.actual_output,
        metrics_data: r.metrics_data,
        latency_ms: (meta.latency_ms as number) ?? null,
        input_tokens: (meta.input_tokens as number) ?? null,
        output_tokens: (meta.output_tokens as number) ?? null,
        additional_metadata: r.additional_metadata,
      };
    }
  }

  return Array.from(caseMap.values());
}

/**
 * Compute per-model summary stats from ModelRun data.
 */
export function computeModelStats(run: ModelRun) {
  const results = run.results;
  const total = results.length;
  const passed = results.filter((r) => r.success).length;
  const passRate = total > 0 ? (passed / total) * 100 : 0;

  const latencies = results
    .map((r) => (r.additional_metadata?.latency_ms as number) ?? null)
    .filter((v): v is number => v != null);
  const avgLatency = latencies.length > 0
    ? latencies.reduce((s, v) => s + v, 0) / latencies.length
    : null;

  const metricNames = new Set<string>();
  for (const r of results) {
    for (const md of r.metrics_data) metricNames.add(md.name);
  }
  const metricRates: Record<string, number> = {};
  for (const name of metricNames) {
    const mp = results.filter((r) =>
      r.metrics_data.some((md) => md.name === name && md.success)
    ).length;
    metricRates[name] = total > 0 ? (mp / total) * 100 : 0;
  }

  // Token and cost totals
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalEvalCost = 0;
  let totalCacheRead = 0;
  let totalCacheWrite = 0;
  let totalGenCost = 0;
  for (const r of results) {
    const meta = r.additional_metadata ?? {};
    totalInputTokens += (meta.input_tokens as number) ?? 0;
    totalOutputTokens += (meta.output_tokens as number) ?? 0;
    totalCacheRead += (meta.cache_read_input_tokens as number) ?? 0;
    totalCacheWrite += (meta.cache_write_input_tokens as number) ?? 0;
    totalGenCost += (meta.generation_cost as number) ?? 0;
    for (const md of r.metrics_data) {
      totalEvalCost += md.evaluation_cost ?? 0;
    }
  }

  return {
    total, passed, passRate, avgLatency, metricRates,
    metricNames: Array.from(metricNames),
    totalInputTokens, totalOutputTokens, totalEvalCost,
    totalCacheRead, totalCacheWrite, totalGenCost,
  };
}

/** Metadata keys with special rendering — excluded from dynamic filter discovery. */
export const SYSTEM_METADATA_KEYS = new Set([
  "latency_ms", "input_tokens", "output_tokens", "total_tokens",
  "cache_read_input_tokens", "cache_write_input_tokens",
  "generation_cost", "id", "category", "error",
]);

/**
 * Scan all cases to discover metadata fields suitable for filter dropdowns.
 * Returns keys that have 2-20 distinct non-empty values (as strings), sorted.
 */
export function discoverFilterableMetadata(
  cases: MultiModelCase[],
): Record<string, string[]> {
  const valueSets = new Map<string, Set<string>>();

  for (const c of cases) {
    for (const m of Object.values(c.models)) {
      for (const [key, val] of Object.entries(m.additional_metadata ?? {})) {
        if (SYSTEM_METADATA_KEYS.has(key) || val == null || val === "") continue;
        const strVal = String(val);
        if (!valueSets.has(key)) valueSets.set(key, new Set());
        valueSets.get(key)!.add(strVal);
      }
    }
  }

  const result: Record<string, string[]> = {};
  for (const [key, vals] of valueSets) {
    if (vals.size >= 2 && vals.size <= 20) {
      result[key] = Array.from(vals).sort();
    }
  }
  return result;
}

/* Per-case field extractors for chart components. */

/**
 * Extract normalized category string from test result metadata.
 */
export function getCategory(tr: TestResult): string | null {
  const cat = tr.additional_metadata?.category;
  return typeof cat === "string" ? cat : null;
}

/**
 * Extract latency_ms from test result metadata.
 */
export function getLatency(tr: TestResult): number | null {
  const ms = tr.additional_metadata?.latency_ms;
  return typeof ms === "number" ? ms : null;
}

/**
 * Extract token usage breakdown from test result metadata.
 */
export function getTokenUsage(tr: TestResult):
  { input_tokens: number; output_tokens: number; total_tokens: number } | null {
  const meta = tr.additional_metadata;
  if (typeof meta?.input_tokens !== "number") return null;
  const input = meta.input_tokens;
  const output = typeof meta.output_tokens === "number" ? meta.output_tokens : 0;
  const total = typeof meta.total_tokens === "number" ? meta.total_tokens : input + output;
  return { input_tokens: input, output_tokens: output, total_tokens: total };
}

/**
 * Find a metric by name in the test result's metrics_data array.
 */
export function getMetric(tr: TestResult, name: string): MetricData | null {
  return tr.metrics_data.find((m) => m.name === name) ?? null;
}

