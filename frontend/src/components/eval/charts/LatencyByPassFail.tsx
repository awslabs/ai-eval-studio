// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LatencyByPassFail component.
 *
 * Renders a side-by-side box plot comparing latency distributions between
 * passing and failing eval cases.
 */

import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getLatency } from "@/lib/eval-utils";

interface LatencyByPassFailProps {
  cases: TestResult[];
}

/** Box plot of latency split by pass/fail outcome. */
export function LatencyByPassFail({ cases }: LatencyByPassFailProps) {
  const { dataViz, layout, hover } = useChartTheme();
  const passed: number[] = [];
  const failed: number[] = [];
  for (const c of cases) {
    const ms = getLatency(c);
    if (ms == null) continue;
    (c.success ? passed : failed).push(ms);
  }

  return (
    <Plot
      data={[
        { type: "box", y: passed, name: "Passed", marker: { color: dataViz.good }, boxpoints: "outliers", ...hover },
        { type: "box", y: failed, name: "Failed", marker: { color: dataViz.bad }, boxpoints: "outliers", ...hover },
      ]}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Latency by Pass/Fail", font: { size: 13 } },
        height: 280,
        yaxis: { ...layout.yaxis, title: { text: "Latency (ms)" } },
        showlegend: false,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
