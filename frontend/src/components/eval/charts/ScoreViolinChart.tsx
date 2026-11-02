// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ScoreViolinChart component.
 *
 * Renders a grouped violin plot of score distributions per metric.
 * When multiple models are present, shows one violin per model within
 * each metric group so distribution differences are visible.
 */

import { useMemo } from "react";
import Plot, { PLOT_CONFIG } from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";
import { useChartTheme } from "@/hooks/useChartTheme";
import { getMetric } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

interface ScoreViolinChartProps {
  cases: TestResult[];
  metricNames: string[];
  modelIds?: string[];
  modelCasesMap?: Record<string, TestResult[]>;
}

/** Grouped violin plot of score distributions per metric, optionally split by model. */
export function ScoreViolinChart({ cases, metricNames, modelIds, modelCasesMap }: ScoreViolinChartProps) {
  const { colors, layout, hover } = useChartTheme();
  const isMultiModel = modelIds != null && modelIds.length > 1 && modelCasesMap != null;

  const traces = useMemo(() => {
    if (isMultiModel) {
      return modelIds!.flatMap((mid, mi) =>
        metricNames.map((name) => ({
          type: "violin" as const,
          y: (modelCasesMap![mid] ?? []).map((c) => getMetric(c, name)?.score).filter((v): v is number => v != null),
          x: (modelCasesMap![mid] ?? []).map((c) => getMetric(c, name)?.score != null ? name : null).filter(Boolean),
          name: shortModel(mid),
          legendgroup: mid,
          scalegroup: name,
          showlegend: metricNames.indexOf(name) === 0,
          box: { visible: true },
          meanline: { visible: true },
          marker: { color: colors[mi % colors.length] },
          ...hover,
        })),
      );
    }
    return metricNames.map((name, i) => ({
      type: "violin" as const,
      y: cases.map((c) => getMetric(c, name)?.score).filter((v): v is number => v != null),
      name,
      box: { visible: true },
      meanline: { visible: true },
      marker: { color: colors[i % colors.length] },
      ...hover,
    }));
  }, [isMultiModel, modelIds, modelCasesMap, cases, metricNames, colors, hover]);

  return (
    <Plot
      data={traces}
      layout={{
        ...layout,
        title: { ...layout.title, text: "Score Distribution by Metric", font: { size: 13 } },
        height: 350,
        yaxis: { ...layout.yaxis, title: { text: "Score" }, range: [0, 1] },
        violinmode: "group",
        showlegend: isMultiModel,
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
