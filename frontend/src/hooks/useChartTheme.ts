// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Thin hook returning everything a Plotly chart component needs to match
 * the current theme. Replaces the 3-4-line boilerplate at the top of every
 * chart component.
 */
import { useMemo } from "react";
import { useTheme } from "@/lib/theme-provider";
import {
  getBaseLayout,
  getChartColors,
  getDataVizColors,
  getHoverStyle,
  getSemanticColors,
} from "@/lib/chart-layout";

/**
 * Returns theme-aware values for Plotly chart components.
 *
 * The result is memoized on ``isDark`` so the returned color/layout/hover
 * objects keep a STABLE identity across renders while the theme is unchanged.
 * Without this the getters allocate fresh objects every render, giving every
 * chart's data/layout new identity → react-plotly.js re-plots on each
 * incidental re-render (which flashes the "double-click to hide" hint and
 * visually resets the controlled legend). Stable references let downstream
 * useMemo(data/layout) actually hold.
 *
 * @returns Object containing isDark flag, color palettes, base layout, and hover style
 */
export function useChartTheme() {
  const { mode } = useTheme();
  const isDark = mode === "dark";
  return useMemo(
    () => ({
      isDark,
      colors: getChartColors(isDark),
      dataViz: getDataVizColors(isDark),
      semantic: getSemanticColors(isDark),
      layout: getBaseLayout(isDark),
      hover: getHoverStyle(isDark),
    }),
    [isDark],
  );
}
