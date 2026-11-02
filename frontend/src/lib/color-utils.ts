// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hardcoded hex color palettes for Plotly charts.
 *
 * Plotly.js does NOT support oklch colors. Instead of trying to convert
 * CSS variables at runtime (which is unreliable), we define hex palettes
 * directly. Light/dark variants are selected by passing isDark explicitly.
 *
 * CSS variables + oklch remain the source of truth for Tailwind utility
 * classes (bg-success, text-error, etc.). This module is ONLY for Plotly.
 */

/** Chart categorical palette — 5 distinct hues for multi-series charts. */
const CHART_LIGHT = ["#89b4fc", "#7dcc96", "#e0be60", "#e8919c", "#b49af4"];
const CHART_DARK = ["#8ab8f8", "#6ed08a", "#e8c45c", "#f09098", "#b89cf5"];

/** Semantic UI colors — for status indicators rendered in Plotly (not data meaning). */
const SEMANTIC_LIGHT = { success: "#6bc490", warning: "#d8b858", error: "#e09090", info: "#89b4fc" };
const SEMANTIC_DARK = { success: "#5cc88a", warning: "#e0be55", error: "#e88080", info: "#7aadf5" };

/**
 * Data visualization colors — pass/fail meaning.
 * Light mode uses softer tones; dark mode uses brighter for contrast.
 */
const DATA_VIZ_LIGHT = {
  good: "#7dcc96",     /* pastel green */
  fair: "#a8c86a",     /* pastel lime */
  neutral: "#e0c870",  /* pastel gold */
  warn: "#e8a878",     /* pastel orange */
  bad: "#e89090",      /* pastel red */
  scaleLow: "#e89090",
  scaleMid: "#e0c870",
  scaleHigh: "#7dcc96",
};

const DATA_VIZ_DARK = {
  good: "#5cc88a",     /* soft green */
  fair: "#98c050",     /* soft lime */
  neutral: "#e0be55",  /* soft gold */
  warn: "#e89860",     /* soft orange */
  bad: "#e88080",      /* soft red */
  scaleLow: "#e88080",
  scaleMid: "#e0be55",
  scaleHigh: "#5cc88a",
};

/**
 * Get the chart categorical palette (5 colors) for the given theme mode.
 * Use for multi-series line/bar/scatter charts where color = identity, not meaning.
 *
 * @param isDark - Whether dark mode is active
 * @returns Array of 5 hex color strings
 */
export function getChartColors(isDark: boolean): string[] {
  return isDark ? CHART_DARK : CHART_LIGHT;
}

/**
 * Get semantic UI colors as hex for Plotly (status indicators, threshold lines).
 * For pass/fail DATA meaning, use getDataVizColors() instead.
 *
 * @param isDark - Whether dark mode is active
 * @returns Object with success, warning, error, info as hex strings
 */
export function getSemanticColors(isDark: boolean) {
  return isDark ? SEMANTIC_DARK : SEMANTIC_LIGHT;
}

/**
 * Get data visualization colors — pass/fail/heatmap.
 * Adapts to light/dark mode for optimal contrast.
 *
 * @param isDark - Whether dark mode is active
 * @returns Object with good, fair, neutral, warn, bad, and scale colors as hex
 */
export function getDataVizColors(isDark: boolean) {
  return isDark ? DATA_VIZ_DARK : DATA_VIZ_LIGHT;
}
