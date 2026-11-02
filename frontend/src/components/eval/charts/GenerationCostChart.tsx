// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Cost-by-model bar chart.
 *
 * Renders grouped bars per model: target-model generation cost next to
 * judge (LLM-as-a-judge) cost. Both sums are read from the run-level
 * totals the harness writes (``generation_cost_usd`` / ``judge_cost_usd``)
 * — the same numbers the report shows, so a chart and a report of the
 * same run can never disagree. Either is ``null`` when that spend is
 * unknown, which reads as no bar rather than a $0 bar.
 *
 * Shows an empty state message when no cost data is recorded.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import { shortModel } from "@/lib/format-utils";
import type { ModelRun } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";

/** Resolve a stable display label off the real model id (verbatim
 *  through shortModel). ``hyperparameters.model`` is written by the
 *  harness on every run, so there is nothing to fall back to. Same
 *  pattern as ``SummaryChartsRow.modelLabel`` — keeps both bar charts
 *  in lockstep. */
function modelLabel(run: ModelRun): string {
  return shortModel(run.hyperparameters.model);
}

interface GenerationCostChartProps {
  modelRuns: ModelRun[];
}

/** Format a USD value for the cost chart.
 *
 * 2 decimals for the headline range ($0.12, $1.87), bumped to 4 when
 * the value is below a cent so cheap models don't all collapse to
 * "$0.00" in the labels and tooltip. Zero is the obvious "$0.00".
 */
function fmtUsdSmart(v: number): string {
  if (v === 0) return "$0.00";
  if (Math.abs(v) < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

/** Grouped bar chart of generation + judge cost per model. */
export function GenerationCostChart({ modelRuns }: GenerationCostChartProps) {
  const { colors, layout, hover } = useChartTheme();
  const data = useMemo(() =>
    modelRuns.map((mr) => {
      // Unknown spend (``null``) plots as nothing. Re-summing the
      // per-case costs here would silently substitute the priced subset
      // for the total the harness refused to state.
      const generation = mr.generation_cost_usd ?? 0;
      const judge = mr.judge_cost_usd ?? 0;
      return { id: modelLabel(mr), generation, judge, total: generation + judge };
    }),
  [modelRuns]);

  if (data.every((d) => d.total === 0)) {
    return <div className="text-sm text-muted-foreground py-8 text-center">No cost data</div>;
  }

  // Pre-format the per-category and total dollars per model — Plotly's
  // ``%{y:.Nf}`` printf is single-precision, but cost ranges often span
  // $0.001…$10 in the same chart. ``fmtUsdSmart`` picks 2dp / 4dp by
  // magnitude. Tooltip-only — bar labels are intentionally omitted to
  // stay consistent with the rest of the chart row.
  const generationCustom = data.map((d) => [fmtUsdSmart(d.generation), fmtUsdSmart(d.total)]);
  const judgeCustom = data.map((d) => [fmtUsdSmart(d.judge), fmtUsdSmart(d.total)]);

  // Compute max single bar value for y-axis range (grouped bars, not stacked)
  const maxBar = Math.max(...data.flatMap((d) => [d.generation, d.judge]), 0);

  // Grouped (side-by-side) bars per model so the user sees both spend
  // categories at once. Same shape as "Avg Score by Model".
  return (
    <Plot
      data={[
        {
          type: "bar",
          name: "Generation",
          x: data.map((d) => d.id),
          y: data.map((d) => d.generation),
          customdata: generationCustom,
          marker: { color: colors[0] },
          hovertemplate:
            "%{x} · Generation: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
          ...hover,
        },
        {
          type: "bar",
          name: "Judge",
          x: data.map((d) => d.id),
          y: data.map((d) => d.judge),
          customdata: judgeCustom,
          marker: { color: colors[1 % colors.length] },
          hovertemplate:
            "%{x} · Judge: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
          ...hover,
        },
      ]}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Cost by Model", font: { size: 13 } },
        height: 300,
        barmode: "group",
        // Categorical x — see SummaryChartsRow for the full reasoning.
        xaxis: { ...layout.xaxis, type: "category" },
        yaxis: { ...layout.yaxis, title: { text: "USD" }, tickprefix: "$", range: [0, maxBar * 1.1] },
        showlegend: true,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%", height: 300 }}
    />
  );
}
