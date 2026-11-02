// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plotly layout defaults shared across analytics/compare charts.
 *
 * ComparePanel and AnalyticsPanel previously each defined a local
 * getBaseLayout / getHoverStyle. This module is the single canonical
 * source — consume via the useChartTheme hook.
 */
import type { Layout } from "plotly.js";

import {
  getChartColors,
  getDataVizColors,
  getSemanticColors,
} from "./color-utils";

/**
 * Base Plotly layout tuned for the app's theme (no series-specific fields).
 *
 * @param isDark - Whether dark mode is active
 * @returns Partial Plotly layout with transparent background, themed grid, and font colors
 */
export function getBaseLayout(isDark: boolean): Partial<Layout> {
  const grid = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
  const zeroLine = isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: isDark ? "#9ca3b0" : "#8a8a8a", size: 11 },
    margin: { t: 35, b: 40, l: 50, r: 20 },
    xaxis: { gridcolor: grid, zerolinecolor: zeroLine, automargin: true },
    yaxis: { gridcolor: grid, zerolinecolor: zeroLine, automargin: true },
    legend: { font: { size: 10 } },
    hoverlabel: {
      bgcolor: isDark ? "#2d2d3d" : "#ffffff",
      font: { color: isDark ? "#e0e0e8" : "#1a1a1a", size: 12 },
      bordercolor: isDark ? "#404058" : "#d0d0d0",
      namelength: 0,
    },
    title: { x: 0, xanchor: "left" as const },
  };
}

/**
 * Per-trace hoverlabel override — theme-aware background with matching text.
 *
 * Use this as a spread on individual Plotly trace objects to override the
 * default hover style per-trace. The hoverlabel in getBaseLayout applies at
 * the layout level; this applies at the trace level for finer control.
 *
 * @param isDark - Whether dark mode is active
 * @returns Object with hoverlabel property ready to spread onto a trace
 */
export function getHoverStyle(isDark: boolean) {
  return {
    hoverlabel: {
      bgcolor: isDark ? "#2d2d3d" : "#ffffff",
      font: { color: isDark ? "#e0e0e8" : "#1a1a1a", size: 12 },
      bordercolor: isDark ? "#404058" : "#d0d0d0",
      namelength: 0,
    },
  };
}

export { getChartColors, getDataVizColors, getSemanticColors };
