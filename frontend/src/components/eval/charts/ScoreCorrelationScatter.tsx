// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ScoreCorrelationScatter component.
 *
 * Renders a scatter plot comparing scores of two metrics against each other.
 * Cases are colored by overall pass/fail status to reveal score correlation patterns.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getMetric } from "@/lib/eval-utils";

/** Hovertemplate for scatter points — <extra></extra> suppresses trace-name secondary label. */
const HT_SCATTER = "x: %{x}<br>y: %{y}<br>%{fullData.name}<extra></extra>";

interface ScoreCorrelationScatterProps {
  cases: TestResult[];
  /** Name of the metric plotted on the x-axis. */
  m1: string;
  /** Name of the metric plotted on the y-axis. */
  m2: string;
}

/** Scatter plot of m1 score vs m2 score, with pass/fail coloring. */
export function ScoreCorrelationScatter({ cases, m1, m2 }: ScoreCorrelationScatterProps) {
  const { dataViz, layout, hover } = useChartTheme();
  const { passed, failed } = useMemo(() => {
    const p: { x: number[]; y: number[] } = { x: [], y: [] };
    const f: { x: number[]; y: number[] } = { x: [], y: [] };
    for (const c of cases) {
      const md1 = getMetric(c, m1);
      const md2 = getMetric(c, m2);
      if (!md1 || !md2) continue;
      const target = c.success ? p : f;
      target.x.push(md1.score ?? 0);
      target.y.push(md2.score ?? 0);
    }
    return { passed: p, failed: f };
  }, [cases, m1, m2]);

  return (
    <Plot
      data={[
        { type: "scatter", mode: "markers", name: "Passed", x: passed.x, y: passed.y, marker: { color: dataViz.good, opacity: 0.6, size: 6 }, hovertemplate: HT_SCATTER, ...hover },
        { type: "scatter", mode: "markers", name: "Failed", x: failed.x, y: failed.y, marker: { color: dataViz.bad, opacity: 0.6, size: 6 }, hovertemplate: HT_SCATTER, ...hover },
      ]}
      layout={{
        ...layout,
        title: { ...layout.title, text: `${m1} vs ${m2}`, font: { size: 13 } },
        height: 300,
        xaxis: { ...layout.xaxis, title: { text: m1 }, range: [0, 1] },
        yaxis: { ...layout.yaxis, title: { text: m2 }, range: [0, 1] },
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
