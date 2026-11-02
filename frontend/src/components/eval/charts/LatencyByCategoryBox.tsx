// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LatencyByCategoryBox component.
 *
 * Renders a grouped box plot of latency distributions per test case category.
 * Each category gets its own box with outlier points shown.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getCategory, getLatency } from "@/lib/eval-utils";

interface LatencyByCategoryBoxProps {
  cases: TestResult[];
}

/** Box plot of latency grouped by test case category. */
export function LatencyByCategoryBox({ cases }: LatencyByCategoryBoxProps) {
  const { colors, layout, hover } = useChartTheme();
  const groups = useMemo(() => {
    const g: Record<string, number[]> = {};
    for (const c of cases) {
      const ms = getLatency(c);
      if (ms == null) continue;
      const cat = getCategory(c) ?? "uncategorized";
      if (!g[cat]) g[cat] = [];
      g[cat].push(ms);
    }
    return g;
  }, [cases]);

  const cats = Object.keys(groups).sort();

  return (
    <Plot
      data={cats.map((cat, i) => ({
        type: "box" as const,
        y: groups[cat],
        name: cat,
        marker: { color: colors[i % colors.length] },
        boxpoints: "outliers",
        ...hover,
      }))}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Latency by Category", font: { size: 13 } },
        height: 350,
        yaxis: { ...layout.yaxis, title: { text: "Latency (ms)" } },
        showlegend: false,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
