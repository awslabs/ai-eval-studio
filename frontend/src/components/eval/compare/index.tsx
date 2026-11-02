// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compare panel coordinator — re-exports ComparePanel as the public surface
 * of the compare/ package.
 *
 * State, data fetching, run selectors (baseline + candidates), and layout are
 * all owned here. The three visual sections are delegated to:
 *   - SummaryTable  — per-run summary table with deltas
 *   - TrendCharts   — pass-rate / latency / per-metric trend charts
 *   - ChangesGrid   — per-case diff with filter tabs and expandable rows
 */

import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { MultiSelect } from "@/components/ui/multi-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { fmtPctInt, shortModel } from "@/lib/format-utils";
import { shortRunLabel, applyCompactMetricFilter } from "@/lib/compare-utils";
import type { EvalRunSummary, CompactModelRun } from "@/lib/types";
import { oldestRun } from "@/lib/run-list";
import { SummaryTable } from "./SummaryTable";
import { TrendCharts } from "./TrendCharts";
import { ChangesGrid } from "./ChangesGrid";

/* ================================================================
 * Types
 * ================================================================ */

interface ComparePanelProps {
  projectId: string;
  runs: EvalRunSummary[];
  /** Visible model IDs from global filter (empty = all). */
  visibleModels: string[];
  /** Active metric names from global filter (null = all). */
  activeMetrics: string[] | null;
}

/* ================================================================
 * Main Component
 * ================================================================ */

/**
 * Multi-run comparison panel.
 * Baseline selector (single) + Candidates filter (multi-select, default all).
 * Global model filter from FilterBar is passed via visibleModels.
 */
export function ComparePanel({ projectId, runs, visibleModels, activeMetrics }: ComparePanelProps) {
  const [baselineFile, setBaselineFile] = useState<string | null>(null);
  const [candidateFiles, setCandidateFiles] = useState<string[]>([]);
  const [loadedCompactRuns, setLoadedCompactRuns] = useState<Record<string, CompactModelRun[]>>({});
  const [loading, setLoading] = useState(false);

  /** Auto-select oldest as baseline on first load. */
  useEffect(() => {
    if (runs.length >= 2 && !baselineFile) {
      setBaselineFile(oldestRun(runs)!._file);
    }
  }, [runs, baselineFile]);

  /** Candidate options = all runs except baseline (newest first, matching eval runs dropdown). */
  const candidateOptions = useMemo(() =>
    runs.filter((r) => r._file !== baselineFile),
  [runs, baselineFile]);

  /** Effective candidates: empty = all. Sorted oldest-first for chronological L→R display. */
  const effectiveCandidateFiles = useMemo(() => {
    const files = candidateFiles.length === 0
      ? candidateOptions.map((r) => r._file)
      : candidateFiles;
    // Sort by timestamp ascending (oldest first) so columns read baseline → v1 → v2 → ...
    return [...files].sort((a, b) => {
      const ta = runs.find((r) => r._file === a)?.timestamp ?? "";
      const tb = runs.find((r) => r._file === b)?.timestamp ?? "";
      return ta.localeCompare(tb);
    });
  }, [candidateFiles, candidateOptions, runs]);

  /** All files needed: baseline + effective candidates. */
  const filesToLoad = useMemo(() => {
    const files = new Set<string>();
    if (baselineFile) files.add(baselineFile);
    for (const f of effectiveCandidateFiles) files.add(f);
    return Array.from(files);
  }, [baselineFile, effectiveCandidateFiles]);

  /** Ordered: baseline first, then candidates (oldest first, as sorted above). */
  const orderedFiles = useMemo(() => {
    if (!baselineFile) return [];
    return [baselineFile, ...effectiveCandidateFiles.filter((f) => f !== baselineFile)];
  }, [baselineFile, effectiveCandidateFiles]);

  /** Candidate files with the baseline removed, memoized to a STABLE reference.
   *  Passed to both SummaryTable and ChangesGrid. An inline `.filter(...)` here
   *  would be a brand-new array on every render, which flows into ExpandedDetail's
   *  fetch effect deps and unmounts the panel (losing the user's resized field
   *  height) on unrelated re-renders such as cursor moves. */
  const candidatesNoBaseline = useMemo(
    () => effectiveCandidateFiles.filter((f) => f !== baselineFile),
    [effectiveCandidateFiles, baselineFile],
  );

  /* Fetch all runs (compact) */
  useEffect(() => {
    if (filesToLoad.length < 2) return;
    const missing = filesToLoad.filter((f) => !loadedCompactRuns[f]);
    if (missing.length === 0) return;
    setLoading(true);
    Promise.all(
      missing.map((f) =>
        api.getEvalRunCompact(projectId, f).then((compact) => [f, compact] as const),
      ),
    )
      .then((results) => {
        setLoadedCompactRuns((prev) => {
          const next = { ...prev };
          for (const [f, d] of results) next[f] = d;
          return next;
        });
      })
      .catch((err) => {
        // Failed runs block the compare view from rendering. Surface the
        // failure to the user via the toaster rather than dying silently
        // in the console.
        toast.error("Could not load run data for comparison.", {
          description: err instanceof Error ? err.message : String(err),
          duration: Infinity,
        });
      })
      .finally(() => setLoading(false));
  }, [projectId, filesToLoad, loadedCompactRuns]);

  /** Whether we have all required data loaded */
  const allLoaded = baselineFile
    && loadedCompactRuns[baselineFile]
    && effectiveCandidateFiles.length > 0
    && effectiveCandidateFiles.every((f) => loadedCompactRuns[f]);

  /** All unique model ids across all loaded runs. Keyed by the
   *  canonical full Bedrock id (``hyperparameters.model``); display
   *  goes through ``shortModel``. */
  const allModels = useMemo(() => {
    const s = new Set<string>();
    for (const file of filesToLoad) {
      const fileRuns = loadedCompactRuns[file];
      if (fileRuns) {
        for (const mr of fileRuns) s.add(mr.hyperparameters.model);
      }
    }
    return Array.from(s).sort();
  }, [filesToLoad, loadedCompactRuns]);

  /** Effective models from global filter (empty = all). */
  const effectiveModels = visibleModels.length > 0 ? visibleModels.filter((m) => allModels.includes(m)) : allModels;

  /** Run summary lookup by file */
  const runSummaryByFile = useMemo(() => {
    const map: Record<string, EvalRunSummary> = {};
    for (const r of runs) map[r._file] = r;
    return map;
  }, [runs]);

  /** Loaded compact runs with the metric filter applied. Uses compact-shaped
   *  filter to preserve semantics (keep only allowed metrics, recompute success,
   *  drop cases with zero surviving metrics). */
  const filteredCompactRuns = useMemo(() => {
    const out: Record<string, CompactModelRun[]> = {};
    for (const [file, runs] of Object.entries(loadedCompactRuns)) {
      out[file] = applyCompactMetricFilter(runs, activeMetrics);
    }
    return out;
  }, [loadedCompactRuns, activeMetrics]);

  if (runs.length < 2) {
    return <p className="text-sm text-muted-foreground py-8">Need at least 2 runs to compare.</p>;
  }

  return (
    <div className="space-y-5">
      {/* Run selectors */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="text-2xs text-muted-foreground">Baseline:</span>
          <Select value={baselineFile ?? ""} onValueChange={setBaselineFile}>
            <SelectTrigger className="h-8 text-xs text-muted-foreground border-input w-auto min-w-[180px]">
              <SelectValue placeholder="Select baseline">
                {(() => { const r = runs.find((x) => x._file === baselineFile); return r ? shortRunLabel(r) : "Select baseline"; })()}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {runs.map((r) => (
                <SelectItem key={r._file} value={r._file} className="text-xs">{shortRunLabel(r)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-2xs text-muted-foreground">Candidates:</span>
          <MultiSelect
            options={candidateOptions.map((r) => r._file)}
            selected={candidateFiles}
            onChange={setCandidateFiles}
            placeholder={`All (${candidateOptions.length})`}
            renderLabel={(file) => { const r = runs.find((x) => x._file === file); return r ? shortRunLabel(r) : file; }}
          />
        </div>
      </div>

      {/* Run info pills */}
      {allLoaded && (
        <div className="flex flex-wrap gap-2">
          {orderedFiles.map((file, idx) => {
            const summary = runSummaryByFile[file];
            if (!summary) return null;
            const isBaseline = idx === 0;
            const totalCases = summary.model_runs.reduce((s, m) => s + m.total, 0);
            const totalPassed = summary.model_runs.reduce((s, m) => s + m.passed, 0);
            const rate = totalCases > 0 ? fmtPctInt(totalPassed / totalCases) : "0";
            return (
              <span key={file} className="inline-flex items-center gap-1.5 bg-muted border border-border/50 rounded-md px-2.5 py-1 text-2xs text-foreground/80">
                <span className="font-semibold text-muted-foreground">
                  {isBaseline ? "baseline" : "candidate"}
                </span>
                <span className="font-medium">{shortRunLabel(summary)}</span>
                <span className="text-muted-foreground">
                  {summary.model_runs.map((m) => shortModel(m.model)).join(", ")}
                </span>
                <span className="font-semibold tabular-nums text-muted-foreground">
                  {rate}%
                </span>
              </span>
            );
          })}
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-muted-foreground py-4">
          <Loader2 className="size-4 animate-spin" /><span>Loading runs...</span>
        </div>
      )}

      {allLoaded && !loading && baselineFile && (
        <>
          <SummaryTable
            baselineFile={baselineFile}
            candidateFiles={candidatesNoBaseline}
            compactRuns={filteredCompactRuns}
            runSummaryByFile={runSummaryByFile}
            allModels={effectiveModels}
          />
          <TrendCharts
            orderedFiles={orderedFiles}
            compactRuns={filteredCompactRuns}
            runSummaryByFile={runSummaryByFile}
            allModels={effectiveModels}
          />
          <ChangesGrid
            projectId={projectId}
            baselineFile={baselineFile}
            candidateFiles={candidatesNoBaseline}
            compactRuns={filteredCompactRuns}
            runSummaryByFile={runSummaryByFile}
            allModels={effectiveModels}
          />
        </>
      )}
    </div>
  );
}
