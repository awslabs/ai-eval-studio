// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LatencyHistogram component.
 *
 * Renders overlaid histograms of latency values, one per model when multiple
 * models are present. Shows a P95 reference line computed from all cases.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getLatency } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

interface LatencyHistogramProps {
  cases: TestResult[];
  modelIds?: string[];
  modelCasesMap?: Record<string, TestResult[]>;
}

/** Histogram of latency distribution with a dashed P95 threshold line. */
export function LatencyHistogram({ cases, modelIds, modelCasesMap }: LatencyHistogramProps) {
  const { colors, semantic, layout, hover } = useChartTheme();

  const allLatencies = useMemo(
    () => cases.map((c) => getLatency(c)).filter((v): v is number => v != null),
    [cases],
  );

  const isMultiModel = modelIds != null && modelIds.length > 1 && modelCasesMap != null;

  if (allLatencies.length === 0) return <div className="text-sm text-muted-foreground py-8 text-center">No latency data</div>;

  const sorted = [...allLatencies].sort((a, b) => a - b);
  const p95 = sorted[Math.min(Math.floor(sorted.length * 0.95), sorted.length - 1)];

  const traces = isMultiModel
    ? modelIds!.map((mid, i) => ({
        type: "histogram" as const,
        x: (modelCasesMap![mid] ?? []).map((c) => getLatency(c)).filter((v): v is number => v != null),
        nbinsx: 20,
        name: shortModel(mid),
        opacity: 0.8,
        marker: { color: colors[i % colors.length] },
        hovertemplate: "%{x}<br>Count: %{y}<br>%{fullData.name}<extra></extra>",
        ...hover,
      }))
    : [{
        type: "histogram" as const,
        x: allLatencies,
        nbinsx: 20,
        marker: { color: colors[0], opacity: 0.7 },
        hovertemplate: "%{x}<br>Count: %{y}<extra></extra>",
        ...hover,
      }];

  return (
    <Plot
      data={traces}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Latency Distribution", font: { size: 13 } },
        height: 280,
        barmode: isMultiModel ? "group" : undefined,
        xaxis: { ...layout.xaxis, title: { text: "Latency (ms)" } },
        yaxis: { ...layout.yaxis, title: { text: "Count" } },
        showlegend: isMultiModel,
        shapes: [{
          type: "line",
          x0: p95, x1: p95,
          y0: 0, y1: 1,
          yref: "paper",
          line: { color: semantic.error, width: 2, dash: "dash" },
        }],
        annotations: [{
          x: p95,
          y: 1,
          yref: "paper",
          text: `P95: ${p95.toFixed(0)}ms`,
          showarrow: false,
          font: { color: semantic.error, size: 10 },
          xanchor: "left",
          xshift: 5,
        }],
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
