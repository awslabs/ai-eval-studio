// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LatencyVsTokensScatter component.
 *
 * Renders a scatter plot of latency (ms) vs output token count per model.
 * Each model gets a distinct color trace so models can be compared visually.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getLatency, getTokenUsage } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

/**
 * Hover template. The model label comes from ``%{fullData.name}`` rather than
 * being interpolated into the string: the ``Plot`` wrapper escapes a trace's
 * ``name`` but leaves templates alone, so a label pasted in here would reach
 * Plotly as markup.
 */
const HT_LATENCY_TOKENS =
  "%{fullData.name}<br>Tokens: %{x}<br>Latency: %{y:.0f}ms<extra></extra>";

interface LatencyVsTokensScatterProps {
  cases: TestResult[];
  modelIds: string[];
  modelCasesMap: Record<string, TestResult[]>;
}

/** Scatter plot of latency vs output token count, with one trace per model. */
export function LatencyVsTokensScatter({
  modelIds, modelCasesMap,
}: LatencyVsTokensScatterProps) {
  const { colors, layout, hover } = useChartTheme();
  const traces = useMemo(() => {
    return modelIds.map((mid, i) => {
      const mc = modelCasesMap[mid] ?? [];
      const x: number[] = [];
      const y: number[] = [];
      for (const c of mc) {
        const ms = getLatency(c);
        const tu = getTokenUsage(c);
        if (ms == null || !tu?.output_tokens) continue;
        x.push(tu.output_tokens);
        y.push(ms);
      }
      const label = shortModel(mid);
      return {
        type: "scatter" as const,
        mode: "markers" as const,
        name: label,
        x, y,
        marker: { color: colors[i % colors.length], opacity: 0.7, size: 7 },
        hovertemplate: HT_LATENCY_TOKENS,
        ...hover,
      };
    });
  }, [modelIds, modelCasesMap, colors, hover]);

  return (
    <Plot
      data={traces}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Latency vs Output Tokens", font: { size: 13 } },
        height: 300,
        xaxis: { ...layout.xaxis, title: { text: "Output Tokens" } },
        yaxis: { ...layout.yaxis, title: { text: "Latency (ms)" } },
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
