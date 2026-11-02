// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * FailureByMetricChart component.
 *
 * Renders a bar chart showing failure counts grouped by metric name.
 * When multiple models are present, renders grouped bars (one per model).
 * When a single model is present, renders a horizontal bar chart.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getMetric } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

/** Hovertemplates — <extra></extra> suppresses the colored trace-name secondary label. */
const HT_BAR_H = "%{y}: %{x}<br>%{fullData.name}<extra></extra>";

interface FailureByMetricChartProps {
  cases: TestResult[];
  metricNames: string[];
  modelIds: string[];
  modelCasesMap: Record<string, TestResult[]>;
}

/** Bar chart of failure counts per metric, grouped by model when multiple models are present. */
export function FailureByMetricChart({
  cases, metricNames, modelIds, modelCasesMap,
}: FailureByMetricChartProps) {
  const { colors, semantic, layout, hover } = useChartTheme();
  const isMultiModel = modelIds.length > 1;

  const singleModelData = useMemo(() => {
    if (isMultiModel) return [];
    const counts: Record<string, number> = {};
    for (const c of cases)
      for (const m of c.metrics_data)
        if (!m.success) counts[m.name] = (counts[m.name] ?? 0) + 1;
    return metricNames.map((n) => ({ name: n, failures: counts[n] ?? 0 })).filter((d) => d.failures > 0);
  }, [cases, metricNames, isMultiModel]);

  if (isMultiModel) {
    /* One trace per model, grouped bars */
    const traces = modelIds.map((mid, i) => ({
      type: "bar" as const,
      name: shortModel(mid),
      x: metricNames,
      y: metricNames.map((name) => {
        const modelCases = modelCasesMap[mid] ?? [];
        return modelCases.filter((c) => {
          const m = getMetric(c, name);
          return m != null && !m.success;
        }).length;
      }),
      marker: { color: colors[i % colors.length] },
      hovertemplate: "%{x}: %{y} failures<br>%{fullData.name}<extra></extra>",
      ...hover,
    }));

    return (
      <Plot
        data={traces}
        layout={{
          ...layout,
          title: { ...layout.title, text: "Failure Count by Metric", font: { size: 13 } },
          height: 300,
          barmode: "group",
          xaxis: { ...layout.xaxis },
          yaxis: { ...layout.yaxis, title: { text: "Failures" } },
        }}
        config={PLOT_CONFIG}
        style={{ width: "100%" }}
      />
    );
  }

  return (
    <Plot
      data={[{
        type: "bar",
        y: singleModelData.map((d) => d.name),
        x: singleModelData.map((d) => d.failures),
        orientation: "h",
        marker: { color: semantic.error, opacity: 0.8 },
        hovertemplate: HT_BAR_H,
        ...hover,
      }]}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Failure Count by Metric", font: { size: 13 } },
        height: Math.max(200, singleModelData.length * 40 + 60),
        xaxis: { ...layout.xaxis, title: { text: "Failures" } },
        showlegend: false,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
