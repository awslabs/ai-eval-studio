// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CategoryMetricHeatmap chart component.
 *
 * Renders a Plotly heatmap of pass rate per category × metric combination.
 * Each cell shows the percentage of cases that passed for that category/metric pair.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getCategory, getMetric } from "@/lib/eval-utils";
import { fmtScaledPctInt, formatCategory } from "@/lib/format-utils";

interface CategoryMetricHeatmapProps {
  cases: TestResult[];
  categories: string[];
  metricNames: string[];
}

/** Heatmap of pass rate across category × metric combinations. */
export function CategoryMetricHeatmap({
  cases, categories, metricNames,
}: CategoryMetricHeatmapProps) {
  const { dataViz, layout } = useChartTheme();
  const { z, text } = useMemo(() => {
    const matrix: number[][] = [];
    const labels: string[][] = [];
    for (const cat of categories) {
      const row: number[] = [];
      const textRow: string[] = [];
      for (const metric of metricNames) {
        let passed = 0;
        let total = 0;
        for (const c of cases) {
          if ((getCategory(c) ?? "uncategorized") !== cat) continue;
          const m = getMetric(c, metric);
          if (m) { total++; if (m.success) passed++; }
        }
        const pct = total > 0 ? (passed / total) * 100 : 0;
        row.push(pct);
        textRow.push(total > 0 ? `${fmtScaledPctInt(pct)}%` : "—");
      }
      matrix.push(row);
      labels.push(textRow);
    }
    return { z: matrix, text: labels };
  }, [cases, categories, metricNames]);

  // ``categories`` stays raw for grouping (matches ``getCategory(c)``);
  // ``displayCategories`` is only used for axis labels and hover.
  const displayCategories = categories.map(formatCategory);

  return (
    <Plot
      data={[{
        type: "heatmap",
        z,
        x: metricNames,
        y: displayCategories,
        text,
        texttemplate: "%{text}",
        colorscale: [[0, dataViz.scaleLow], [0.5, dataViz.scaleMid], [1, dataViz.scaleHigh]],
        zmin: 0,
        zmax: 100,
        colorbar: { title: { text: "Pass %" }, ticksuffix: "%" },
        hovertemplate: "%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
      }]}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Pass Rate: Category × Metric", font: { size: 13 } },
        height: Math.max(300, displayCategories.length * 30 + 100),
        margin: { ...layout.margin, t: 60, l: Math.min(200, Math.max(80, Math.max(...displayCategories.map((c) => c.length)) * 7)) },
        xaxis: { ...layout.xaxis, side: "top" as const },
        yaxis: { ...layout.yaxis, autorange: "reversed" as const },
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
