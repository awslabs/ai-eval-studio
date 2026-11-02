// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SummaryChartsRow component.
 *
 * Three multi-model summary charts displayed above the analytics tabs:
 *   1. Pass rate by model (grouped bar)
 *   2. Avg score by model (grouped bar per metric)
 *   3. Latency distribution by model (box plot)
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import { ChartCard } from "@/components/ui/chart-card";
import { shortModel } from "@/lib/format-utils";
import type { ModelRun } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";

/** Display label for a model run — full id passed through ``shortModel``. */
function modelLabel(run: ModelRun): string {
  return shortModel(run.hyperparameters.model);
}

interface SummaryChartsProps {
  /** ModelRun[] with the metric filter already applied upstream. The
   *  pass-rate, latency, and avg-score traces all derive from this
   *  same data — so the filter narrows them all together. */
  modelRuns: ModelRun[];
  /** When true, the chart row stretches to fill its parent's height and each
   *  chart autosizes to fill (Overview's fill-remaining-space layout). When
   *  false (default), charts keep their fixed ~220px height. */
  fill?: boolean;
}

/**
 * Three multi-model summary charts shown above tabs.
 * Chart 1: Pass rate by model (grouped bar, one bar per model)
 * Chart 2: Avg score by model (grouped bars per metric)
 * Chart 3: Latency by model (box plot, one box per model)
 *
 * Render-stability discipline: the data and layout for each chart are
 * memoized so re-renders triggered by an active loop upstream don't
 * churn the props identity, and react-plotly.js skips the re-plot. The
 * user's zoom does not depend on this: the ``Plot`` wrapper's
 * ``uirevision`` keeps it through a re-plot.
 */
export function SummaryChartsRow({ modelRuns, fill = false }: SummaryChartsProps) {
  const { colors, dataViz, layout, hover } = useChartTheme();

  const modelStats = useMemo(() =>
    modelRuns.map((run) => {
      const total = run.results.length;
      const passed = run.results.filter((r) => r.success).length;
      const latencies = run.results
        .map((r) => (r.additional_metadata?.latency_ms as number) ?? null)
        .filter((v): v is number => v != null);
      const totalTokens = run.results.reduce((s, r) => {
        const meta = r.additional_metadata ?? {};
        return s + ((meta.input_tokens as number) ?? 0) + ((meta.output_tokens as number) ?? 0);
      }, 0);
      const totalGenCost = run.results.reduce((s, r) => s + ((r.additional_metadata?.generation_cost as number) ?? 0), 0);
      return {
        // Canonical full id — guaranteed unique per ModelRun in the
        // file. Used as the dict key for metricScoresByModel below
        // so two distinct full ids that collapse to the same short
        // display label can't clobber each other.
        modelId: run.hyperparameters.model,
        // Display label.
        id: modelLabel(run),
        total,
        passed,
        passRate: total > 0 ? (passed / total) * 100 : 0,
        latencies,
        totalTokens,
        totalGenCost,
      };
    }),
  [modelRuns]);

  // Metric names derived from the (already-narrowed) data — no
  // separate filter prop needed.
  const allMetricNames = useMemo(() => {
    const names = new Set<string>();
    for (const run of modelRuns) for (const r of run.results) for (const md of r.metrics_data) names.add(md.name);
    return Array.from(names);
  }, [modelRuns]);

  // Keyed by canonical full model id so two distinct full ids that
  // collapse to the same short display label can't overwrite each
  // other's metric scores. Display labels go through ``modelLabel``
  // at render time only.
  const metricScoresByModel = useMemo(() => {
    const result: Record<string, Record<string, { scores: number[]; threshold: number }>> = {};
    for (const run of modelRuns) {
      const mid = run.hyperparameters.model;
      result[mid] = {};
      for (const name of allMetricNames) result[mid][name] = { scores: [], threshold: 0.5 };
      for (const r of run.results) {
        for (const md of r.metrics_data) {
          if (!result[mid][md.name]) continue;
          if (md.score != null) result[mid][md.name].scores.push(md.score);
          result[mid][md.name].threshold = md.threshold;
        }
      }
    }
    return result;
  }, [modelRuns, allMetricNames]);

  // ---- Chart 1: Pass Rate ----
  const passRateData = useMemo(() => [{
    type: "bar" as const,
    x: modelStats.map((m) => m.id),
    y: modelStats.map((m) => m.passRate),
    text: modelStats.map((m) => `${m.passed}/${m.total}`),
    textposition: "auto" as const,
    marker: {
      color: modelStats.map((m) =>
        m.passRate >= 70 ? dataViz.good : m.passRate >= 40 ? dataViz.neutral : dataViz.bad
      ),
    },
    hovertemplate: "%{x}: %{y:.1f}% (%{text})<extra></extra>",
    ...hover,
  }], [modelStats, dataViz, hover]);

  const passRateLayout = useMemo(() => ({
    ...layout,
    title: { ...layout.title, text: "Pass Rate by Model", font: { size: 13 } },
    height: 220,
    margin: { t: 35, b: 40, l: 40, r: 10 },
    // ``type: "category"`` is structural defense against
    // numeric-looking labels auto-coercing the axis to numeric
    // (which would silently drop non-numeric categories).
    xaxis: { ...layout.xaxis, type: "category" as const },
    yaxis: { ...layout.yaxis, range: [0, 100] as [number, number], ticksuffix: "%" },
    showlegend: false,
  }), [layout]);

  // ---- Chart 2: Avg Score ----
  const avgScoreData = useMemo(() => allMetricNames.map((name, i) => ({
    type: "bar" as const,
    name,
    x: modelRuns.map((run) => modelLabel(run)),
    y: modelRuns.map((run) => {
      const entry = metricScoresByModel[run.hyperparameters.model]?.[name];
      if (!entry || entry.scores.length === 0) return 0;
      return entry.scores.reduce((a, b) => a + b, 0) / entry.scores.length;
    }),
    customdata: modelRuns.map((run) => metricScoresByModel[run.hyperparameters.model]?.[name]?.scores.length ?? 0),
    marker: { color: colors[i % colors.length] },
    hovertemplate: "%{fullData.name}: %{y:.3f} avg (n=%{customdata})<extra></extra>",
    ...hover,
  })), [allMetricNames, modelRuns, metricScoresByModel, colors, hover]);

  const avgScoreLayout = useMemo(() => ({
    ...layout,
    title: { ...layout.title, text: "Avg Score by Model", font: { size: 13 } },
    height: 220,
    margin: { t: 35, b: 40, l: 40, r: 10 },
    barmode: "group" as const,
    xaxis: { ...layout.xaxis, type: "category" as const },
    yaxis: { ...layout.yaxis, range: [0, 1] as [number, number] },
    showlegend: allMetricNames.length > 1,
  }), [layout, allMetricNames.length]);

  // ---- Chart 3: Latency ----
  const latencyData = useMemo(() => modelStats.map((m, i) => ({
    type: "box" as const,
    y: m.latencies,
    // Plain model name — endpoint/region routing context lives
    // in the descriptor metadata, not the legend.
    name: m.id,
    marker: { color: colors[i % colors.length] },
    boxpoints: "outliers" as const,
    ...hover,
  })), [modelStats, colors, hover]);

  const latencyLayout = useMemo(() => ({
    ...layout,
    title: { ...layout.title, text: "Latency Distribution", font: { size: 13 } },
    height: 220,
    margin: { t: 35, b: 40, l: 50, r: 10 },
    xaxis: { ...layout.xaxis, type: "category" as const },
    yaxis: { ...layout.yaxis, title: { text: "ms" } },
    showlegend: modelStats.length > 1,
  }), [layout, modelStats.length]);

  // In fill mode each chart fills its (stretched) card; otherwise keep the
  // fixed 220px height. The card itself stretches via h-full so the Plot's
  // autosize (fill) has a tall container to grow into.
  const cardClass = fill ? "p-3 h-full" : "p-3";
  const plotStyle = fill ? { width: "100%", height: "100%" } : { width: "100%", height: 220 };

  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-4 min-w-0 ${fill ? "h-full" : ""}`}>
      {/* Chart 1: Pass Rate by Model (bar) */}
      <ChartCard title="Pass Rate by Model" className={cardClass}>
        <Plot
          data={passRateData}
          layout={passRateLayout}
          config={PLOT_CONFIG}
          style={plotStyle}
          fill={fill}
        />
      </ChartCard>

      {/* Chart 2: Avg Score by Model (grouped bars, one trace per metric) */}
      <ChartCard title="Avg Score by Model" className={cardClass}>
        <Plot
          data={avgScoreData}
          layout={avgScoreLayout}
          config={PLOT_CONFIG}
          style={plotStyle}
          fill={fill}
        />
      </ChartCard>

      {/* Chart 3: Latency Distribution (box plot by model) */}
      <ChartCard title="Latency Distribution" className={cardClass}>
        <Plot
          data={latencyData}
          layout={latencyLayout}
          config={PLOT_CONFIG}
          style={plotStyle}
          fill={fill}
        />
      </ChartCard>
    </div>
  );
}
