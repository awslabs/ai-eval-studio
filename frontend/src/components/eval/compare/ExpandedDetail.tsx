// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Expanded detail pane for a changes grid row.
 *
 * Shows input, expected output, and per-run sections containing the model's
 * actual output, token stats, latency, and per-metric scores with failure
 * reasons. Used as the expandable bottom section of a ChangeRowComponent.
 *
 * Re-exports CaseStatus and StatusBadge from DetailSection so ChangesGrid
 * can continue to import them from this module without change.
 */

import { useEffect, useMemo, useState } from "react";
import { DetailSection, StatusBadge } from "@/components/eval/DetailSection";
import type { CaseStatus } from "@/components/eval/DetailSection";
import { api } from "@/lib/api";
import { fmtNum } from "@/lib/format-utils";
import { shortRunLabel } from "@/lib/compare-utils";
import type { EvalRunSummary, CaseDetail } from "@/lib/types";

export { StatusBadge };
export type { CaseStatus };

/* ================================================================
 * Types (shared with ChangesGrid)
 * ================================================================ */

/** Per-candidate result for a single case in the changes grid (compact shape). */
export interface CandidateResult {
  passed: boolean;
  outputPreview: string;
  outputHash: string;
  status: CaseStatus;
  scores: Record<string, number>;
  passes: Record<string, boolean>;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  cacheRead: number;
  cacheWrite: number;
}

/** A row in the changes grid (compact shape). */
export interface ChangeRow {
  model: string;
  caseId: string;
  category: string;
  inputPreview: string;
  blPassed: boolean;
  blOutputPreview: string;
  blOutputHash: string;
  blScores: Record<string, number>;
  blPasses: Record<string, boolean>;
  blLatency: number;
  blInputTokens: number;
  blOutputTokens: number;
  blCacheRead: number;
  blCacheWrite: number;
  overallStatus: CaseStatus;
  candidates: Record<string, CandidateResult | null>;
}

export interface ExpandedDetailProps {
  projectId: string;
  baselineFile: string;
  row: ChangeRow;
  candidateFiles: string[];
  runSummaryByFile: Record<string, EvalRunSummary>;
  blShort: string;
}

/* ================================================================
 * Helpers
 * ================================================================ */

/**
 * Builds a stats array for a single run's token/latency figures.
 *
 * @param latencyMs - Latency in milliseconds
 * @param inputTokens - Input token count
 * @param outputTokens - Output token count
 * @param cacheRead - Cache-read token count
 * @param cacheWrite - Cache-write token count
 */
function buildRunStats(
  latencyMs: number,
  inputTokens: number,
  outputTokens: number,
  cacheRead: number,
  cacheWrite: number,
): Array<{ label: string; value: string }> {
  return [
    { label: "Latency", value: latencyMs > 0 ? `${fmtNum(latencyMs)}ms` : "---" },
    { label: "In", value: fmtNum(inputTokens) },
    { label: "Out", value: fmtNum(outputTokens) },
    { label: "Total", value: fmtNum(inputTokens + outputTokens) },
    { label: "Cache Read", value: fmtNum(cacheRead) },
    { label: "Cache Write", value: fmtNum(cacheWrite) },
  ];
}

/* ================================================================
 * Main export
 * ================================================================ */

/** Module-level cache for case details: key = `${file}:${caseId}`.
 *  A value of null means "looked up, not found in that run" so we never retry. */
const caseDetailCache = new Map<string, CaseDetail | null>();

/**
 * Read whatever case-detail entries are already cached for the given files.
 *
 * @param files - Files to look up (baseline + candidates).
 * @param caseId - The case identifier.
 * @returns The partial details map plus whether EVERY file was already cached.
 *   When all are cached the caller can resolve synchronously — no fetch, no
 *   spinner. A cached "not found" is stored as null and still counts as present.
 */
function readCachedDetails(
  files: string[],
  caseId: string,
): { map: Record<string, CaseDetail | null>; allCached: boolean } {
  const map: Record<string, CaseDetail | null> = {};
  let allCached = true;
  for (const file of files) {
    const cacheKey = `${file}:${caseId}`;
    if (caseDetailCache.has(cacheKey)) {
      map[file] = caseDetailCache.get(cacheKey) ?? null;
    } else {
      allCached = false;
    }
  }
  return { map, allCached };
}

/** Expanded detail view with lazy case-detail loading. */
export function ExpandedDetail({
  projectId,
  baselineFile,
  row,
  candidateFiles,
  runSummaryByFile,
  blShort,
}: ExpandedDetailProps) {
  const allFiles = useMemo(
    () => [baselineFile, ...candidateFiles],
    [baselineFile, candidateFiles],
  );

  /* Seed state synchronously from the module-level cache. When the row was
   * already viewed (common: re-expand, or a re-render from a parent state
   * change), every file is cached and we render immediately — no spinner,
   * no setLoading churn, so the user's resized <pre> is never torn down. */
  const initial = readCachedDetails(allFiles, row.caseId);
  const [details, setDetails] = useState<Record<string, CaseDetail | null>>(initial.map);
  const [loading, setLoading] = useState(!initial.allCached);
  const [error, setError] = useState<string | null>(null);

  /* Fetch only the case details not already cached. Runs when row/files change.
   * The spinner is gated on "no data yet" below, so a background refetch keeps
   * the previously-rendered detail (and the resized field) mounted. */
  useEffect(() => {
    const { map: cachedMap, allCached } = readCachedDetails(allFiles, row.caseId);
    // Always reflect what's cached right now.
    setDetails(cachedMap);
    setError(null);

    if (allCached) {
      // Everything is in cache — resolve synchronously, no fetch, no spinner.
      setLoading(false);
      return;
    }

    // A fetch is in flight (at least one file not cached) → show the corner
    // spinner. The pane still renders fully from compact preview data, so this
    // is just a feedback hint, not a layout-occupying loading state.
    setLoading(true);

    let cancelled = false;
    Promise.all(
      allFiles.map(async (file) => {
        const cacheKey = `${file}:${row.caseId}`;
        if (caseDetailCache.has(cacheKey)) {
          return [file, caseDetailCache.get(cacheKey) ?? null] as const;
        }
        try {
          const detail = await api.getEvalCaseDetail(projectId, file, row.caseId);
          caseDetailCache.set(cacheKey, detail);
          return [file, detail] as const;
        } catch {
          // Case not found in this run — cache null so we don't retry.
          caseDetailCache.set(cacheKey, null);
          return [file, null] as const;
        }
      }),
    )
      .then((results) => {
        if (cancelled) return;
        const detailsMap: Record<string, CaseDetail | null> = {};
        for (const [file, detail] of results) {
          detailsMap[file] = detail;
        }
        setDetails(detailsMap);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [projectId, row.caseId, allFiles]);

  const blDetail = details[baselineFile];

  if (error) {
    return (
      <div className="bg-muted/20 border-t p-8">
        <p className="text-sm text-error">Failed to load case detail: {error}</p>
      </div>
    );
  }

  /* Render the full pane immediately from the row's compact preview data
   * (scores, pass/fail, latency, tokens, output preview — all already in
   * memory). Full text + failure reasons swap in when the fetch lands, into
   * fixed-height boxes, so the layout never shifts — the pane opens like a
   * normal pane with no jump. While the fetch is in flight, a thin pulsing bar
   * sits at the very top of the pane as the loading hint: it spans the full
   * width and pulses uniformly, so it needs no centering and looks correct at
   * any horizontal scroll position, inline or fullscreen. */
  return (
    <div className="bg-muted/20 border-t relative">
      {loading && (
        <>
          {/* Dim + blur the pane while the full detail loads. Covers the whole
           *  pane (incl. off-screen columns); no centered content, so nothing
           *  to mis-position at scroll or in fullscreen. */}
          <div className="absolute inset-0 z-20 backdrop-blur-[2px] bg-background/40" />
          {/* Indeterminate progress: a short segment slides left→right on a
           *  loop. Inline keyframes follow the existing ActiveJobBar pattern;
           *  uses the theme primary token (no hardcoded colors). */}
          <div className="absolute inset-x-0 top-0 z-30 h-0.5 overflow-hidden bg-primary/10">
            <style>{`
              @keyframes expanded-detail-progress {
                0% { left: -40%; }
                100% { left: 100%; }
              }
            `}</style>
            <div
              className="absolute top-0 h-full w-2/5 bg-primary/80"
              style={{ animation: "expanded-detail-progress 2.6s ease-in-out infinite" }}
            />
          </div>
        </>
      )}
      {/* Input + Expected from baseline detail (identical for every run). Each
       *  box is width-capped so it stays readable instead of stretching across
       *  the full multi-column table width. */}
      <div className="px-5 pt-4 pb-2 flex flex-wrap gap-4">
        <div className="flex-1 min-w-[280px] max-w-2xl">
          <p className="text-2xs font-medium text-muted-foreground mb-1">Input</p>
          <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-background border rounded-md p-3 h-20 overflow-y-auto resize-y leading-relaxed text-muted-foreground">
            {blDetail?.input ?? row.inputPreview}
          </pre>
        </div>
        <div className="flex-1 min-w-[280px] max-w-2xl">
          <p className="text-2xs font-medium text-muted-foreground mb-1">Expected Output</p>
          <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-background border rounded-md p-3 h-20 overflow-y-auto resize-y leading-relaxed text-muted-foreground">
            {blDetail?.expected_output ?? "---"}
          </pre>
        </div>
      </div>

      {/* Per-run detail sections. Columns size relative to the available width
       *  (minmax(0,1fr)) so they divide the table-width colSpan cell evenly and
       *  can NEVER exceed it — no spill past the table's right edge. With many
       *  runs each column lands ~the same width as the summary table's run
       *  columns and scrolls with the table (one scrollbar). max-w cap keeps a
       *  column from stretching unreadably wide when there are only a few runs. */}
      <div className="px-5 pb-4">
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${1 + candidateFiles.length}, minmax(0, 1fr))` }}
        >
          {/* Baseline detail. Stats + metric scores come from the row's compact
           *  data (already in memory) so the section renders fully on open;
           *  ``output`` shows the preview until the full text fetch lands, then
           *  swaps in place (fixed-height box → no layout shift). */}
          {(() => {
            const modelData = blDetail?.models[row.model];
            const metrics = modelData
              ? modelData.metrics.map((m) => ({
                  name: m.name,
                  score: m.score ?? 0,
                  passed: m.success,
                  reason: m.reason,
                }))
              : Object.keys(row.blScores).map((name) => ({
                  name,
                  score: row.blScores[name] ?? 0,
                  passed: row.blPasses[name] ?? false,
                  reason: null,
                }));
            return (
              <DetailSection
                key="baseline"
                label={`Baseline (${blShort})`}
                passed={row.blPassed}
                stats={buildRunStats(
                  modelData?.latency_ms ?? row.blLatency,
                  modelData?.input_tokens ?? row.blInputTokens,
                  modelData?.output_tokens ?? row.blOutputTokens,
                  modelData?.cache_read_input_tokens ?? row.blCacheRead,
                  modelData?.cache_write_input_tokens ?? row.blCacheWrite,
                )}
                output={modelData?.actual_output ?? row.blOutputPreview}
                metrics={metrics}
                status={null}
              />
            );
          })()}

          {/* Candidate detail sections */}
          {candidateFiles.map((file) => {
            const cand = row.candidates[file];
            const summary = runSummaryByFile[file];
            const label = summary ? shortRunLabel(summary) : file;
            if (!cand) {
              return (
                <div key={file} className="min-w-0">
                  <p className="text-xs font-semibold text-muted-foreground mb-2">{label}</p>
                  <span className="text-xs text-muted-foreground">Not in this run</span>
                </div>
              );
            }

            const candDetail = details[file];
            const modelData = candDetail?.models[row.model];
            /* Render fully from the row's compact candidate data on open; the
             * full output text swaps in when the fetch lands (fixed-height box,
             * no layout shift). */
            const metrics = modelData
              ? modelData.metrics.map((m) => ({
                  name: m.name,
                  score: m.score ?? 0,
                  passed: m.success,
                  reason: m.reason,
                }))
              : Object.keys(cand.scores).map((name) => ({
                  name,
                  score: cand.scores[name] ?? 0,
                  passed: cand.passes[name] ?? false,
                  reason: null,
                }));

            return (
              <DetailSection
                key={file}
                label={label}
                passed={cand.passed}
                stats={buildRunStats(
                  modelData?.latency_ms ?? cand.latencyMs,
                  modelData?.input_tokens ?? cand.inputTokens,
                  modelData?.output_tokens ?? cand.outputTokens,
                  modelData?.cache_read_input_tokens ?? cand.cacheRead,
                  modelData?.cache_write_input_tokens ?? cand.cacheWrite,
                )}
                output={modelData?.actual_output ?? cand.outputPreview}
                metrics={metrics}
                status={cand.status}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
