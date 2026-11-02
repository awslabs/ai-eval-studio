// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TokenUsageChart component.
 *
 * Renders a grouped (side-by-side) bar chart of input vs output token
 * usage per model — same shape as "Avg Score by Model". Stacked bars
 * were unworkable because input/output ratios are routinely 100×–1000×
 * in real evals, so the smaller series rendered as an invisibly thin
 * band on a linear y-axis.
 *
 * Shows nothing (empty state message) when no token data is available.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getTokenUsage } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

interface TokenUsageChartProps {
  cases: TestResult[];
  modelIds: string[];
  modelCasesMap: Record<string, TestResult[]>;
}

/** Grouped bar chart of input/output token usage per model. */
export function TokenUsageChart({
  cases, modelIds, modelCasesMap,
}: TokenUsageChartProps) {
  const { semantic, layout, hover } = useChartTheme();
  const hasData = cases.some((c) => getTokenUsage(c) != null);

  // Memoize trace data + chart layout so a parent re-render doesn't churn
  // their identity — react-plotly.js re-plots whenever data/layout props
  // change reference, wiping internal state like legend toggles. Same
  // discipline the rest of the chart row follows.
  const data = useMemo(() => {
    const sumTokens = (cs: TestResult[]) => {
      let inp = 0, out = 0;
      for (const c of cs) {
        const tu = getTokenUsage(c);
        inp += tu?.input_tokens ?? 0;
        out += tu?.output_tokens ?? 0;
      }
      return { input: inp, output: out };
    };
    const xLabels = modelIds.map(shortModel);
    return [
      {
        type: "bar" as const,
        name: "Input",
        x: xLabels,
        y: modelIds.map((mid) => sumTokens(modelCasesMap[mid] ?? []).input),
        marker: { color: semantic.warning },
        hovertemplate: "%{x}<br>Input: %{y:,.0f}<extra></extra>",
        ...hover,
      },
      {
        type: "bar" as const,
        name: "Output",
        x: xLabels,
        y: modelIds.map((mid) => sumTokens(modelCasesMap[mid] ?? []).output),
        marker: { color: semantic.success },
        hovertemplate: "%{x}<br>Output: %{y:,.0f}<extra></extra>",
        ...hover,
      },
    ];
  }, [modelIds, modelCasesMap, semantic, hover]);

  const chartLayout = useMemo(() => ({
    ...layout,
    title: { ...layout.title, text: "Token Usage by Model", font: { size: 13 } },
    height: 300,
    barmode: "group" as const,
    // Categorical x — same defence-in-depth as the other bar
    // charts in case future labels look numeric.
    xaxis: { ...layout.xaxis, type: "category" as const },
    yaxis: { ...layout.yaxis, title: { text: "Tokens" } },
  }), [layout]);

  if (!hasData) {
    return <div className="text-sm text-muted-foreground py-8 text-center">No token data</div>;
  }

  return (
    <Plot
      data={data}
      layout={chartLayout}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
