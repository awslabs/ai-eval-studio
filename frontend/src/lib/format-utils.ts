// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Small pure display helpers shared across eval components.
 *
 * These one-liners were each redefined locally in 2–4 places (ComparePanel,
 * CaseDetail, FilterBar, FileOpToolUI). Collecting them here gives one
 * canonical implementation per concern and removes subtle drift (e.g. one
 * copy of `truncate` appended "..." while another used "…").
 */

/** Convert snake_case / kebab-case / space-separated text to Title Case. */
export function formatLabel(key: string): string {
  return key
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

/** Format a number with locale-aware thousands separators. */
export function fmtNum(n: number): string {
  return n.toLocaleString();
}

/**
 * Format a 0..1 rate as an integer percentage. Never rounds up to 100%
 * unless the rate is literally 1.0 — ``toFixed(0)`` rounds 99.5% to
 * "100%" which falsely implies zero failures. Truncating avoids that:
 * 1 failure in 208 (99.52%) correctly shows as 99%.
 */
export function fmtPctInt(rate: number): string {
  if (rate >= 1.0) return "100";
  return Math.floor(rate * 100).toString();
}

/**
 * Same semantics as ``fmtPctInt`` but keeps one decimal of precision.
 * Use in compare panels / trend charts where the extra digit helps
 * distinguish near-perfect scores.
 */
export function fmtPct1(rate: number): string {
  if (rate >= 1.0) return "100.0";
  // Truncate to one decimal (e.g. 99.52 → "99.5", 99.99 → "99.9").
  return (Math.floor(rate * 1000) / 10).toFixed(1);
}

/**
 * ``fmtPctInt`` for values already on the 0..100 scale. Truncates to an
 * integer, reserving "100" for literal 100.0. Use when the caller has
 * already multiplied by 100 (e.g. pre-scaled metric rates from
 * eval-utils).
 */
export function fmtScaledPctInt(pct: number): string {
  if (pct >= 100) return "100";
  return Math.floor(pct).toString();
}

/**
 * ``fmtPct1`` for values already on the 0..100 scale. One decimal of
 * precision, reserving "100.0" for literal 100.0.
 */
export function fmtScaledPct1(pct: number): string {
  if (pct >= 100) return "100.0";
  return (Math.floor(pct * 10) / 10).toFixed(1);
}

/**
 * Structural short-name derivation for Bedrock model ids.
 *
 * Mirror of ``backend/app/services/model_id_short.py::short_model_id``
 * — the rule has zero hardcoded provider/region/model lists. The prior
 * regex (``^(us\\.)?(anthropic|openai|bedrock|cohere|meta)\\.``) silently
 * dropped every other vendor — xAI, Google Gemma, Mistral, MiniMax,
 * Kimi, GLM, Qwen, DeepSeek — and could not handle dotted versions
 * (``xai.grok-4.3``, ``openai.gpt-5.4``) at all.
 *
 * Algorithm:
 *   1. Strip leading dot-segments until the first segment containing
 *      a digit. That segment is where the model name begins.
 *   2. Strip trailing ``-v\\d+(:\\d+)?`` (Bedrock version tag).
 *   3. Strip trailing ``-\\d{8,}`` (date stamp).
 *
 * Verbatim remainder — ``claude-opus-4-8`` stays ``claude-opus-4-8``;
 * ``grok-4.3`` keeps its dot.
 */
export function shortModel(id: string): string {
  if (!id) return id;
  const parts = id.split(".");
  let short = id;
  if (parts.length > 1) {
    const firstWithDigit = parts.findIndex((seg) => /\d/.test(seg));
    short = firstWithDigit >= 0 ? parts.slice(firstWithDigit).join(".") : parts[parts.length - 1];
  }
  short = short.replace(/-v\d+(:\d+)?$/, "");
  short = short.replace(/-\d{8,}$/, "");
  return short;
}

/**
 * Format a raw category value for display.
 *
 * Categories can arrive as URL-encoded source-file paths
 * (``documents/2024-01-02%20Security%20Policy.pdf``) which
 * render as illegible escape sequences in chart labels. This helper:
 *
 *   1. ``decodeURIComponent`` so ``%20`` etc. become real characters
 *   2. strips a leading ``documents/`` path prefix
 *   3. strips a trailing ``.pdf`` extension
 *   4. collapses runs of whitespace to a single space
 *
 * Used at render time ONLY. Grouping/lookup must keep using the raw
 * value (see ``getCategory`` in ``eval-utils``) so case-to-category
 * grouping stays exact even when the format changes.
 */
export function formatCategory(raw: string): string {
  let s: string;
  try {
    s = decodeURIComponent(raw);
  } catch {
    // Malformed escape sequence — fall back to the raw value rather
    // than blowing up the chart. The display will still be ugly, but
    // it won't crash.
    s = raw;
  }
  s = s.replace(/^documents\//, "");
  s = s.replace(/\.pdf$/i, "");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

/** Truncate a string to ``n`` characters, appending an ellipsis if cut. */
export function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n) + "…";
}

/**
 * Return the correctly inflected noun form (singular or plural) based on count.
 *
 * Used by eval components to agree iteration-count numbers with their nouns.
 * Three call sites across two files (ActiveJobBar status text, OptimizationRunDetail
 * criteria badges). Lives here rather than inline: shared display helper at 2–4×
 * duplication is the category this module was created for.
 *
 * Callers keep their own number formatting and interpolation; this function
 * returns the NOUN FORM ONLY.
 *
 * @param count - The number to check
 * @param singular - Singular form (e.g., "iteration")
 * @param plural - Optional explicit plural form (default: singular + "s")
 * @returns The correctly inflected noun
 */
export function pluralize(count: number, singular: string, plural?: string): string {
  return count === 1 ? singular : (plural ?? `${singular}s`);
}
