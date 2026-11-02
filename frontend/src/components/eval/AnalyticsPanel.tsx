// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Analytics panel with interactive Plotly charts.
 *
 * Analysis tab feature set:
 * All charts support zoom, pan, hover details, fullscreen, and PNG download.
 *
 * Charts:
 * 1. Category × Metric heatmap
 * 2. Failure breakdown by metric (grouped bar)
 * 3. Failures by category
 * 4. Token usage by model (grouped bar)
 * 5. Latency vs output tokens scatter
 * 6. Generation cost summary
 * 7. Score correlation scatter (when 2+ metrics)
 * 8. Score distribution violin
 * 9. Latency statistics table
 * 10. Latency distribution histogram with threshold
 * 11. Latency by pass/fail box plot
 * 12. Latency by category box plot
 */

import { useMemo } from "react";
import { ChartCard } from "@/components/ui/chart-card";
import type { ModelRun, TestResult } from "@/lib/types";
import { getCategory } from "@/lib/eval-utils";
import { CategoryMetricHeatmap } from "./charts/CategoryMetricHeatmap";
import { FailureByMetricChart } from "./charts/FailureByMetricChart";
import { FailuresByCategoryChart } from "./charts/FailuresByCategoryChart";
import { TokenUsageChart } from "./charts/TokenUsageChart";
import { GenerationCostChart } from "./charts/GenerationCostChart";
import { LatencyVsTokensScatter } from "./charts/LatencyVsTokensScatter";
import { ScoreCorrelationScatter } from "./charts/ScoreCorrelationScatter";
import { ScoreViolinChart } from "./charts/ScoreViolinChart";
import { LatencyStatsTable } from "./charts/LatencyStatsTable";
import { LatencyHistogram } from "./charts/LatencyHistogram";
import { LatencyByPassFail } from "./charts/LatencyByPassFail";
import { LatencyByCategoryBox } from "./charts/LatencyByCategoryBox";

export { SummaryChartsRow } from "./charts/SummaryChartsRow";

interface AnalyticsPanelProps {
  /** Multi-model runs to render charts for. The metric filter has
   *  already been applied upstream (``applyMetricFilter`` in
   *  ``EvalDashboard``), so ``metrics_data`` contains only the
   *  metrics the user wants to see. Charts derive visible metric
   *  names from the data, not from a separate filter prop. */
  modelRuns: ModelRun[];
}

/**
 * Renders all analytics charts with interactive Plotly visualizations.
 * Passes TestResult[] directly to chart components.
 */
export function AnalyticsPanel({ modelRuns }: AnalyticsPanelProps) {
  // Keyed by canonical full model id (``hyperparameters.model``);
  // child charts shorten via ``shortModel`` at render time.
  const modelCasesMap = useMemo(() => {
    const map: Record<string, TestResult[]> = {};
    for (const mr of modelRuns) map[mr.hyperparameters.model] = mr.results;
    return map;
  }, [modelRuns]);

  const modelIds = Object.keys(modelCasesMap);
  const allCases = useMemo(() => Object.values(modelCasesMap).flat(), [modelCasesMap]);

  const metricNames = useMemo(() => {
    const s = new Set<string>();
    for (const c of allCases) for (const m of c.metrics_data) s.add(m.name);
    return Array.from(s);
  }, [allCases]);

  const categories = useMemo(() => {
    const s = new Set<string>();
    for (const c of allCases) s.add(getCategory(c) ?? "uncategorized");
    return Array.from(s).sort();
  }, [allCases]);

  return (
    <div className="space-y-6">
      {/* Category × Metric heatmap */}
      {categories.length > 1 && metricNames.length > 0 && (
        <ChartCard>
          <CategoryMetricHeatmap cases={allCases} categories={categories} metricNames={metricNames} />
        </ChartCard>
      )}

      {/* Failure breakdown + Failures by category */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard>
          <FailureByMetricChart cases={allCases} metricNames={metricNames} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
        <ChartCard>
          <FailuresByCategoryChart cases={allCases} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
      </div>

      {/* Token usage + Cost */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard>
          <TokenUsageChart cases={allCases} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
        <ChartCard>
          <GenerationCostChart modelRuns={modelRuns} />
        </ChartCard>
      </div>

      {/* Scatter plots */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard>
          <LatencyVsTokensScatter cases={allCases} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
        {metricNames.length >= 2 && (
          <ChartCard>
            <ScoreCorrelationScatter cases={allCases} m1={metricNames[0]} m2={metricNames[1]} />
          </ChartCard>
        )}
      </div>

      {/* Score distribution violin */}
      {metricNames.length > 0 && (
        <ChartCard>
          <ScoreViolinChart cases={allCases} metricNames={metricNames} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
      )}

      {/* Latency section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Latency Statistics">
          <LatencyStatsTable cases={allCases} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
        <ChartCard>
          <LatencyHistogram cases={allCases} modelIds={modelIds} modelCasesMap={modelCasesMap} />
        </ChartCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard>
          <LatencyByPassFail cases={allCases} />
        </ChartCard>
        {categories.length > 1 && (
          <ChartCard>
            <LatencyByCategoryBox cases={allCases} />
          </ChartCard>
        )}
      </div>
    </div>
  );
}
