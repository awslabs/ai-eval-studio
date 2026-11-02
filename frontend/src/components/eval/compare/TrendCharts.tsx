// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Trend charts for the compare panel.
 *
 * Renders three Plotly line charts:
 *  1. Pass Rate Trend — one line per model
 *  2. Latency Trend  — one line per model (only when latency data present)
 *  3. Metric Trends  — one line per metric (averaged across models),
 *     hover shows per-model breakdown
 *
 * Renders nothing unless 2+ ordered files are present and there is trend data.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import { ChartCard } from "@/components/ui/chart-card";
import { useChartTheme } from "@/hooks/useChartTheme";
import { fmtScaledPct1, shortModel } from "@/lib/format-utils";
import { shortRunLabel } from "@/lib/compare-utils";
import { escapePlotlyMarkup } from "@/lib/plotly-escape";
import type { EvalRunSummary, CompactModelRun } from "@/lib/types";

export interface TrendChartsProps {
  orderedFiles: string[];
  /** Loaded compact runs with the metric filter already applied upstream
   *  (see ``ComparePanel``). Per-metric trend lines derive from the
   *  data itself. */
  compactRuns: Record<string, CompactModelRun[]>;
  runSummaryByFile: Record<string, EvalRunSummary>;
  allModels: string[];
}

/**
 * Trend charts across the selected runs: pass rate and latency (one line per
 * model) and metric trends (one line per metric). Renders nothing with fewer
 * than 2 runs or no trend data.
 */
export function TrendCharts({ orderedFiles, compactRuns, runSummaryByFile, allModels }: TrendChartsProps) {
  const { colors, layout, hover } = useChartTheme();

  /* Build trend data per model: pass rate, avg latency, per-metric rates */
  const { trendData, metricTrendData, allMetricNames, hasLatencyData } = useMemo(() => {
    const rows: { model: string; runLabel: string; passRate: number; avgLatency: number | null }[] = [];
    const metricRows: { metric: string; model: string; runLabel: string; rate: number }[] = [];
    const metricSet = new Set<string>();

    for (const file of orderedFiles) {
      const summary = runSummaryByFile[file];
      const label = summary ? shortRunLabel(summary) : file;
      const modelRuns = compactRuns[file] ?? [];

      for (const mr of modelRuns) {
        if (!allModels.includes(mr.hyperparameters.model)) continue;
        const results = mr.results;
        const passed = results.filter((r) => r.success).length;
        const passRate = results.length > 0 ? (passed / results.length) * 100 : 0;
        const lats = results.map((r) => r.latency_ms).filter((v) => v > 0);
        const avgLat = lats.length > 0 ? lats.reduce((s, v) => s + v, 0) / lats.length : null;
        rows.push({ model: mr.hyperparameters.model, runLabel: label, passRate, avgLatency: avgLat });

        /* Per-metric per-model pass rates */
        const perMetric: Record<string, { pass: number; total: number }> = {};
        for (const r of results) {
          for (const m of r.metrics) {
            metricSet.add(m.name);
            if (!perMetric[m.name]) perMetric[m.name] = { pass: 0, total: 0 };
            perMetric[m.name].total += 1;
            if (m.success) perMetric[m.name].pass += 1;
          }
        }
        for (const [mn, counts] of Object.entries(perMetric)) {
          if (counts.total > 0) {
            metricRows.push({ metric: mn, model: mr.hyperparameters.model, runLabel: label, rate: (counts.pass / counts.total) * 100 });
          }
        }
      }
    }

    return {
      trendData: rows,
      metricTrendData: metricRows,
      allMetricNames: Array.from(metricSet).sort(),
      hasLatencyData: rows.some((d) => d.avgLatency != null),
    };
  }, [orderedFiles, compactRuns, runSummaryByFile, allModels]);

  /* Build averaged metric trend lines with per-model hover breakdown */
  const metricTraces = useMemo(() => {
    return allMetricNames.map((metric, i) => {
      const runLabels = orderedFiles.map((file) => {
        const summary = runSummaryByFile[file];
        return summary ? shortRunLabel(summary) : file;
      });

      const avgRates: number[] = [];
      // One hover TEMPLATE per point, not ``customdata``. This hover is composed
      // markup — ``<br>`` and ``<b>`` around project text — and the Plot wrapper
      // escapes every customdata string, which printed those tags instead of
      // rendering them. Templates are left alone by the wrapper, so each project
      // value is escaped here, where it is interpolated.
      const hoverTemplates: string[] = [];
      const safeMetric = escapePlotlyMarkup(metric);

      for (const label of runLabels) {
        const safeLabel = escapePlotlyMarkup(label);
        const perModel = metricTrendData.filter((d) => d.metric === metric && d.runLabel === label);
        if (perModel.length === 0) {
          avgRates.push(0);
          hoverTemplates.push(`${safeLabel}<br><b>${safeMetric}: no data</b><extra></extra>`);
          continue;
        }
        const avg = perModel.reduce((s, d) => s + d.rate, 0) / perModel.length;
        avgRates.push(avg);

        const breakdown = perModel
          .map((d) => `  ${escapePlotlyMarkup(shortModel(d.model))}: ${fmtScaledPct1(d.rate)}%`)
          .join("<br>");
        hoverTemplates.push(
          `${safeLabel}<br><b>${safeMetric}: ${fmtScaledPct1(avg)}%</b> (avg)<br>${breakdown}<extra></extra>`,
        );
      }

      return {
        type: "scatter" as const,
        mode: "lines+markers" as const,
        name: metric,
        x: runLabels,
        y: avgRates,
        line: { color: colors[i % colors.length], width: 2 },
        marker: { size: 6, color: colors[i % colors.length] },
        hovertemplate: hoverTemplates,
        ...hover,
      };
    });
  }, [allMetricNames, orderedFiles, runSummaryByFile, metricTrendData, colors, hover]);

  /* Memoize each chart's data + layout so they keep a STABLE identity across
   * renders. react-plotly.js re-plots whenever data/layout identity changes;
   * Compare self-re-renders periodically, so inline (fresh-every-render)
   * data/layout would re-plot each time — flashing the legend hint and
   * resetting the controlled legend. Deps are real content (trendData, models,
   * theme colors/hover/layout — themselves now stable via useChartTheme). */
  const passRateData = useMemo(
    () =>
      allModels.map((model, i) => {
        const modelData = trendData.filter((d) => d.model === model);
        return {
          type: "scatter" as const,
          mode: "lines+markers" as const,
          name: shortModel(model),
          x: modelData.map((d) => d.runLabel),
          y: modelData.map((d) => d.passRate),
          line: { color: colors[i % colors.length], width: 2 },
          marker: { size: 6, color: colors[i % colors.length] },
          hovertemplate: "%{x}: %{y:.1f}%<br>%{fullData.name}<extra></extra>",
          ...hover,
        };
      }),
    [allModels, trendData, colors, hover],
  );

  const latencyData = useMemo(
    () =>
      allModels.map((model, i) => {
        const modelData = trendData.filter((d) => d.model === model);
        return {
          type: "scatter" as const,
          mode: "lines+markers" as const,
          name: shortModel(model),
          x: modelData.map((d) => d.runLabel),
          y: modelData.map((d) => d.avgLatency),
          line: { color: colors[i % colors.length], width: 2 },
          marker: { size: 6, color: colors[i % colors.length] },
          hovertemplate: "%{x}: %{y:.0f}ms<br>%{fullData.name}<extra></extra>",
          ...hover,
        };
      }),
    [allModels, trendData, colors, hover],
  );

  const passRateLayout = useMemo(
    () => ({
      ...layout,
      height: 300,
      title: { text: "Pass Rate Trend", font: { size: 13 }, x: 0, xanchor: "left" as const },
      yaxis: { ...layout.yaxis, range: [0, 100] as [number, number], ticksuffix: "%" },
    }),
    [layout],
  );

  const latencyLayout = useMemo(
    () => ({
      ...layout,
      height: 300,
      title: { text: "Latency Trend", font: { size: 13 }, x: 0, xanchor: "left" as const },
      yaxis: { ...layout.yaxis, ticksuffix: "ms" },
    }),
    [layout],
  );

  const metricLayout = useMemo(
    () => ({
      ...layout,
      height: 300,
      title: { text: "Metric Trends", font: { size: 13 }, x: 0, xanchor: "left" as const },
      yaxis: { ...layout.yaxis, range: [0, 100] as [number, number], ticksuffix: "%" },
    }),
    [layout],
  );

  if (orderedFiles.length < 2 || trendData.length === 0) return null;

  return (
    <div className="space-y-4">
      {/* Pass Rate + Latency side by side */}
      <div className={`grid gap-4 ${hasLatencyData ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}>
        <ChartCard title="Pass Rate Trend" className="p-4">
          <Plot
            data={passRateData}
            layout={passRateLayout}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
          />
        </ChartCard>
        {hasLatencyData && (
          <ChartCard title="Latency Trend" className="p-4">
            <Plot
              data={latencyData}
              layout={latencyLayout}
              config={PLOT_CONFIG}
              style={{ width: "100%" }}
            />
          </ChartCard>
        )}
      </div>

      {/* Per-metric trend (averaged across models, hover shows per-model breakdown) */}
      {allMetricNames.length > 0 && (
        <ChartCard title="Metric Trends" className="p-4">
          <Plot
            data={metricTraces}
            layout={metricLayout}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
          />
        </ChartCard>
      )}
    </div>
  );
}
