// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * FailuresByCategoryChart component.
 *
 * Renders a bar chart of failure counts grouped by test case category.
 * When multiple models are present, renders grouped bars (one per model).
 * Single-model mode shows failure percentage as bar text.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getCategory } from "@/lib/eval-utils";
import { fmtPctInt, formatCategory, shortModel } from "@/lib/format-utils";

interface FailuresByCategoryChartProps {
  cases: TestResult[];
  modelIds: string[];
  modelCasesMap: Record<string, TestResult[]>;
}

/** Bar chart of failure counts per category, grouped by model when multiple models are present. */
export function FailuresByCategoryChart({
  cases, modelIds, modelCasesMap,
}: FailuresByCategoryChartProps) {
  const { colors, semantic, layout, hover } = useChartTheme();
  const categories = useMemo(() => {
    const cats = new Set<string>();
    for (const c of cases) cats.add(getCategory(c) ?? "uncategorized");
    return Array.from(cats).sort();
  }, [cases]);

  const isMultiModel = modelIds.length > 1;

  // Display labels go through ``formatCategory``; grouping/filtering
  // still uses the raw ``categories`` array.
  const xLabels = categories.map(formatCategory);

  if (isMultiModel) {
    const traces = modelIds.map((mid, i) => ({
      type: "bar" as const,
      name: shortModel(mid),
      x: xLabels,
      y: categories.map((cat) => {
        const mc = modelCasesMap[mid] ?? [];
        return mc.filter((c) => (getCategory(c) ?? "uncategorized") === cat && !c.success).length;
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
          title: { ...layout.title, text: "Failures by Category", font: { size: 13 } },
          height: 300,
          barmode: "group",
          yaxis: { ...layout.yaxis, title: { text: "Failures" } },
        }}
        config={PLOT_CONFIG}
        style={{ width: "100%" }}
      />
    );
  }

  // Single model — simple bar
  const data = categories.map((cat) => {
    const catCases = cases.filter((c) => (getCategory(c) ?? "uncategorized") === cat);
    return { cat, failures: catCases.filter((c) => !c.success).length, total: catCases.length };
  }).filter((d) => d.failures > 0);

  return (
    <Plot
      data={[{
        type: "bar",
        x: data.map((d) => formatCategory(d.cat)),
        y: data.map((d) => d.failures),
        marker: { color: semantic.warning, opacity: 0.8 },
        text: data.map((d) => `${fmtPctInt(d.failures / d.total)}%`),
        textposition: "auto",
        hovertemplate: "%{x}: %{y} failures<extra></extra>",
        ...hover,
      }]}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Failures by Category", font: { size: 13 } },
        height: 300,
        yaxis: { ...layout.yaxis, title: { text: "Failures" } },
        showlegend: false,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
