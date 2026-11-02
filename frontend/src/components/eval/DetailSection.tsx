// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared expandable section describing one model's (or run's) result for a
 * single test case. Used by CaseDetail (MultiModelCaseDetail) and the Compare
 * expanded-row detail (ExpandedDetail).
 *
 * Also exports CaseStatus and StatusBadge which were formerly private to
 * ExpandedDetail.tsx and are needed by ChangesGrid.tsx.
 */

import { useMemo } from "react";
import { StatItem } from "@/components/ui/stat-item";
import { fmtPctInt } from "@/lib/format-utils";

/* ================================================================
 * CaseStatus + StatusBadge
 * ================================================================ */

/** Status of a case compared between baseline and a candidate run. */
export type CaseStatus = "regression" | "fixed" | "changed" | "same";

/** Coloured status badge for regression / fixed / changed / same cases. */
export function StatusBadge({
  status,
  size = "normal",
}: {
  status: CaseStatus;
  size?: "normal" | "small";
}) {
  const sizeClass =
    size === "small" ? "text-2xs px-1.5 py-0" : "text-2xs px-2 py-0.5";
  switch (status) {
    case "regression":
      return (
        <span
          className={`inline-block font-bold uppercase tracking-wide rounded ${sizeClass} bg-error/20 text-error`}
        >
          regression
        </span>
      );
    case "fixed":
      return (
        <span
          className={`inline-block font-bold uppercase tracking-wide rounded ${sizeClass} bg-success/20 text-success`}
        >
          fixed
        </span>
      );
    case "changed":
      return (
        <span
          className={`inline-block font-bold uppercase tracking-wide rounded ${sizeClass} bg-warning/20 text-warning`}
        >
          changed
        </span>
      );
    case "same":
      return (
        <span
          className={`inline-block font-bold uppercase tracking-wide rounded ${sizeClass} bg-muted text-muted-foreground`}
        >
          same
        </span>
      );
  }
}

/* ================================================================
 * DetailSection
 * ================================================================ */

/** A single metric entry normalised for use in DetailSection. */
export interface DetailMetric {
  name: string;
  /** Fractional score (0–1), or null when unavailable. */
  score: number | null;
  passed: boolean;
  reason: string | null;
}

/** Props for the shared per-model / per-run detail section. */
export interface DetailSectionProps {
  /** Label shown in the section header (model id, run label, …). */
  label: string;
  /** Whether this run passed overall. */
  passed: boolean;
  /** Flat list of pre-computed stat key/value pairs to show in the stats row. */
  stats: Array<{ label: string; value: string }>;
  /** Actual output text to display. */
  output: string;
  /**
   * When provided the output area renders a character/word-level diff against
   * this expected string. When absent the output is shown as a plain pre.
   */
  diffExpected?: string;
  /** Metrics to display in the pills + failure-reasons section. */
  metrics: DetailMetric[];
  /**
   * Optional compare-view status badge (regression / fixed / changed / same).
   * When null or "same" no extra badge is rendered.
   */
  status?: CaseStatus | null;
}

/**
 * Renders a bordered card containing a run's header, stats, output, and
 * metric pills. Shared by MultiModelCaseDetail and ExpandedDetail.
 *
 * Metric pills come from the ``metrics`` array as-is — the metric filter
 * has already narrowed the data upstream in EvalDashboard / ComparePanel,
 * so this component doesn't take a separate filter prop.
 */
export function DetailSection({
  label,
  passed,
  stats,
  output,
  diffExpected,
  metrics,
  status,
}: DetailSectionProps) {
  const diffSegments = useMemo(() => {
    if (diffExpected === undefined) return null;
    return computeDiff(diffExpected, output);
  }, [diffExpected, output]);

  const isIdentical =
    diffExpected !== undefined && diffExpected.trim() === output.trim();

  const failedWithReasons = metrics.filter((m) => !m.passed && m.reason);

  return (
    <div className="border rounded-lg bg-card shadow-sm p-4 space-y-3 min-w-0 overflow-hidden">
      {/* Header: label + optional status badge + pass/fail */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-semibold text-sm text-muted-foreground">
          {label}
        </span>
        {status && status !== "same" && (
          <StatusBadge status={status} size="small" />
        )}
        <span
          className={`text-2xs font-bold px-2 py-0 rounded ${
            passed ? "bg-success/15 text-success" : "bg-error/15 text-error"
          }`}
        >
          {passed ? "PASS" : "FAIL"}
        </span>
      </div>

      {/* Stats row */}
      {stats.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-muted-foreground">
          {stats.map(({ label: statLabel, value }) => (
            <StatItem key={statLabel} inline label={statLabel} value={value} />
          ))}
        </div>
      )}

      {/* Output — diff-highlighted when diffExpected is supplied */}
      <div>
        {diffExpected !== undefined ? (
          <>
            <div className="flex items-center gap-2 mb-1">
              <p className="text-2xs font-medium text-muted-foreground">
                Output
              </p>
              {isIdentical ? (
                <span className="text-2xs text-success">Identical</span>
              ) : (
                <span className="text-2xs text-warning">
                  Differences highlighted
                </span>
              )}
            </div>
            {isIdentical || !diffSegments ? (
              <pre className="text-xs whitespace-pre-wrap font-mono bg-background border rounded-md p-3 h-40 overflow-y-auto resize-y leading-relaxed break-all text-muted-foreground">
                {output}
              </pre>
            ) : (
              <div className="text-xs whitespace-pre-wrap font-mono bg-background border rounded-md p-3 h-40 overflow-y-auto resize-y leading-relaxed break-all text-muted-foreground">
                {diffSegments.map((seg, i) => {
                  if (seg.type === "equal")
                    return <span key={i}>{seg.text}</span>;
                  if (seg.type === "added") {
                    return (
                      <span
                        key={i}
                        className="bg-success/15 text-success rounded-sm px-0.5"
                      >
                        {seg.text}
                      </span>
                    );
                  }
                  return (
                    <span
                      key={i}
                      className="bg-error/15 text-error line-through rounded-sm px-0.5"
                    >
                      {seg.text}
                    </span>
                  );
                })}
              </div>
            )}
          </>
        ) : (
          <>
            <p className="text-2xs font-medium text-muted-foreground mb-1">
              Output
            </p>
            <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-background border rounded-md p-3 h-40 overflow-y-auto resize-y leading-relaxed text-muted-foreground">
              {output}
            </pre>
          </>
        )}
      </div>

      {/* Metrics: pills + failure-reason pre boxes */}
      {metrics.length > 0 && (
        <div className="space-y-2">
          <p className="text-2xs font-medium text-muted-foreground">
            Metrics
          </p>
          <div className="flex flex-wrap gap-1.5">
            {metrics.map((m) => (
              <span
                key={m.name}
                className="inline-flex items-center gap-1 text-2xs font-medium px-2 py-1 rounded bg-muted text-muted-foreground"
              >
                <span className={m.passed ? "text-success" : "text-error"}>
                  {m.passed ? "\u2713" : "\u2717"}
                </span>
                {m.name}{" "}
                {m.score != null ? fmtPctInt(m.score) + "%" : "\u2014"}
              </span>
            ))}
          </div>
          {failedWithReasons.map((m) => (
            <pre
              key={m.name}
              className="text-xs font-mono whitespace-pre-wrap break-words bg-background border rounded-md p-2 h-20 overflow-y-auto resize-y text-muted-foreground"
            >
              <span className="font-semibold text-muted-foreground">
                {m.name}:
              </span>{" "}
              {m.reason}
            </pre>
          ))}
        </div>
      )}
    </div>
  );
}

/* ================================================================
 * Diff engine (moved from CaseDetail.tsx — shared by DetailSection)
 * ================================================================ */

export interface DiffSegment {
  type: "equal" | "added" | "deleted";
  text: string;
}

/**
 * Computes a character- or word-level diff between expected and actual strings.
 * Falls back to word-level diff for strings longer than 500 characters to
 * avoid O(n*m) blowup.
 *
 * Exported so MultiModelCaseDetail can reuse it without duplicating the
 * diff engine.
 *
 * @param expected - The reference / expected text
 * @param actual - The actual output text
 * @returns Array of diff segments with type annotations
 */
export function computeDiff(expected: string, actual: string): DiffSegment[] {
  if (expected.trim() === actual.trim()) {
    return [{ type: "equal", text: actual }];
  }
  if (expected.length > 500 || actual.length > 500) {
    return wordDiff(expected, actual);
  }
  return charDiff(expected, actual);
}

/**
 * Character-level diff using a longest-common-subsequence approach.
 *
 * @param expected - Reference string
 * @param actual - Candidate string
 */
function charDiff(expected: string, actual: string): DiffSegment[] {
  const segments: DiffSegment[] = [];
  const lcs = longestCommonSubseq(expected, actual);

  let ei = 0;
  let ai = 0;
  let li = 0;

  while (li < lcs.length) {
    const { ei: le, ai: la } = lcs[li];
    if (ei < le) segments.push({ type: "deleted", text: expected.slice(ei, le) });
    if (ai < la) segments.push({ type: "added", text: actual.slice(ai, la) });
    segments.push({ type: "equal", text: expected[le] });
    ei = le + 1;
    ai = la + 1;
    li++;
  }

  if (ei < expected.length)
    segments.push({ type: "deleted", text: expected.slice(ei) });
  if (ai < actual.length)
    segments.push({ type: "added", text: actual.slice(ai) });

  return mergeSegments(segments);
}

/** LCS pair: indices into expected (ei) and actual (ai). */
interface LCSPair {
  ei: number;
  ai: number;
}

/**
 * Computes longest-common-subsequence index pairs for two strings.
 * Standard DP, O(n*m) space and time.
 */
function longestCommonSubseq(a: string, b: string): LCSPair[] {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0)
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const pairs: LCSPair[] = [];
  let i = m;
  let j = n;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      pairs.push({ ei: i - 1, ai: j - 1 });
      i--;
      j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  pairs.reverse();
  return pairs;
}

/**
 * Word-level diff for longer strings.
 * Splits on whitespace boundaries and applies LCS at word granularity.
 */
function wordDiff(expected: string, actual: string): DiffSegment[] {
  const expWords = expected.split(/(\s+)/);
  const actWords = actual.split(/(\s+)/);
  const segments: DiffSegment[] = [];

  const lcs = longestCommonSubseqGeneric(expWords, actWords);
  let ei = 0;
  let ai = 0;

  for (const { ei: le, ai: la } of lcs) {
    if (ei < le)
      segments.push({ type: "deleted", text: expWords.slice(ei, le).join("") });
    if (ai < la)
      segments.push({ type: "added", text: actWords.slice(ai, la).join("") });
    segments.push({ type: "equal", text: expWords[le] });
    ei = le + 1;
    ai = la + 1;
  }

  if (ei < expWords.length)
    segments.push({ type: "deleted", text: expWords.slice(ei).join("") });
  if (ai < actWords.length)
    segments.push({ type: "added", text: actWords.slice(ai).join("") });

  return mergeSegments(segments);
}

/**
 * Generic LCS for string arrays (word-level).
 */
function longestCommonSubseqGeneric(a: string[], b: string[]): LCSPair[] {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0)
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const pairs: LCSPair[] = [];
  let i = m;
  let j = n;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      pairs.push({ ei: i - 1, ai: j - 1 });
      i--;
      j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  pairs.reverse();
  return pairs;
}

/**
 * Merges adjacent diff segments of the same type to reduce DOM nodes.
 */
function mergeSegments(segments: DiffSegment[]): DiffSegment[] {
  if (segments.length === 0) return segments;
  const merged: DiffSegment[] = [segments[0]];
  for (let i = 1; i < segments.length; i++) {
    const prev = merged[merged.length - 1];
    if (segments[i].type === prev.type) {
      prev.text += segments[i].text;
    } else {
      merged.push(segments[i]);
    }
  }
  return merged;
}
