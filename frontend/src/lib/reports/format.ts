// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Report formatting helpers — kept here so every report component uses the
 * same number/text shapes.
 *
 * PERCENTAGES TRUNCATE, THEY DO NOT ROUND, and they delegate to the canonical
 * helpers in ``format-utils`` rather than restating the rule. These used to
 * round while the eval dashboard truncated, so the same run could read 89% in
 * the Evals tab and 90% in its report. Truncation is also the
 * correctness-relevant choice for this product:
 * rounding displays "90%" for a run that scored 89.6%, next to a cell this same
 * table colours RED for missing its 0.90 target, so the number would contradict
 * the marking beside it. ``format-utils`` already documented that reasoning for
 * the dashboard; the reports simply now obey it too.
 */

import { fmtScaledPct1, fmtScaledPctInt } from "@/lib/format-utils";

/** Whole-number percent, truncated: ``72%``. Reserves ``100%`` for a true 100. */
export function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${fmtScaledPctInt(value)}%`;
}

/** One-decimal percent, truncated: ``99.5%`` — for near-miss disambiguation. */
export function fmtPctPrecise(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value === 100 || value === 0) return `${fmtScaledPctInt(value)}%`;
  return `${fmtScaledPct1(value)}%`;
}

/** Auto-precision percent: whole when it lands on it, else 1-decimal. */
export function fmtPctSmart(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return value === 100 || value === 0 ? fmtPct(value) : fmtPctPrecise(value);
}

/** Latency cell: ``821 ms``. */
export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${Math.round(value).toLocaleString()} ms`;
}

/** Latency cell, no unit (column header carries it): ``821``. */
export function fmtMsCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Math.round(value).toLocaleString();
}

/** USD with two decimals: ``$2.32``. */
export function fmtUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** Humanize an ISO-8601 timestamp to ``June 6, 2026``. */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString(undefined, {
      month: "long", day: "numeric", year: "numeric",
    });
  } catch {
    return iso;
  }
}

/** Truncate at sentence boundary near ``limit`` chars. */
export function shorten(text: string, limit: number): string {
  if (!text || text.length <= limit) return text;
  const truncated = text.slice(0, limit);
  const lastPeriod = truncated.lastIndexOf(".");
  if (lastPeriod > limit * 0.5) return truncated.slice(0, lastPeriod + 1);
  return truncated.trimEnd() + "…";
}

/** Δ-cell: ``+14`` / ``−6`` / ``—``. */
export function fmtDelta(delta: number | null | undefined): string {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return "—";
  if (Math.abs(delta) < 0.5) return "—";
  const sign = delta > 0 ? "+" : "−";
  return `${sign}${Math.abs(delta).toFixed(1)}`;
}

/** Slug for filenames: lowercase, hyphenated. */
export function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "report";
}

/**
 * Canonical one-line definitions for DeepEval's built-in metrics.
 *
 * Built-in metrics are constructed without a ``criteria=`` kwarg, so
 * ``_parse_metric_definitions`` (backend) returns nothing for them. When
 * the narrator hasn't run yet (or failed, or is still running on the
 * background thread) the "What was tested" section would otherwise show
 * just the bare threshold ("score ≥ 0.7") with no definition. Keys match
 * the default ``__name__`` values in deepeval's source — see
 * ``deepeval/metrics/<name>/<name>.py``.
 *
 * This is the LAST fallback in the chain — narrator-written description
 * wins, then parsed GEval criteria, then this. Latency is intentionally
 * absent: its line is constructed from the budget directly.
 */
export const BUILTIN_METRIC_DESCRIPTIONS: Record<string, string> = {
  "Answer Relevancy": "how directly the response answers the input question",
  "Faithfulness": "whether the response stays grounded in the retrieved context",
  "Contextual Precision": "whether the most relevant retrieved chunks rank highest",
  "Contextual Recall": "whether retrieval surfaces the information needed for the answer",
  "Contextual Relevancy": "whether retrieved context actually pertains to the question",
  "Hallucination": "whether the response contradicts or invents details vs the source (lower is better)",
  "Bias": "whether the response carries unwarranted bias (lower is better)",
  "Toxicity": "whether the response contains toxic language (lower is better)",
  "Summarization": "whether the summary is faithful to and covers the source",
  "Json Correctness": "whether the response matches the required JSON schema",
  "Prompt Alignment": "whether the response follows explicit instructions in the prompt",
};

/**
 * Render the "What was tested" line for one metric.
 *
 * Description fallback chain (best to worst):
 *   1. Narrator-written one-liner (``narrative.metric_descriptions[name]``)
 *   2. Parsed GEval ``criteria=`` kwarg from the project's metrics module
 *   3. Built-in definition for known DeepEval metrics
 *   4. Threshold-only ("score ≥ 0.7") — never blank
 *
 * Latency is special-cased: the line is built from the budget instead.
 */
export function metricLine(
  name: string,
  threshold: number | undefined,
  criteria: string | undefined,
  budgetMs: number | null,
  description?: string | null,
): string {
  if (name === "Latency" && budgetMs) {
    return description
      ? `${description} (≤ ${budgetMs} ms).`
      : `per-request, passes at ≤ ${budgetMs} ms.`;
  }
  const thrText =
    threshold !== undefined && Math.abs(threshold - 1.0) < 1e-9
      ? "passes at score = 1.0"
      : threshold !== undefined
        ? `score ≥ ${threshold}`
        : "passes";
  // Narrator-written descriptions are pre-shortened one-liners. Raw GEval
  // criteria render IN FULL — a report is a document, and truncating the
  // definition of what was tested mid-sentence loses exactly the part the
  // reader needs (the judging basis). Layout wraps; no data is cut.
  const desc =
    description || criteria || BUILTIN_METRIC_DESCRIPTIONS[name] || "";
  const descText = desc ? ` — ${desc}` : "";
  const line = `${thrText}${descText}`;
  // Narrator descriptions and raw criteria usually end with their own
  // terminal punctuation; appending unconditionally produced "syntax.."
  // which reads like a truncation artifact.
  return /[.!?]$/.test(line) ? line : `${line}.`;
}
