// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Summary table for the compare panel.
 *
 * Rows = models, columns grouped by run. Baseline values shown plain;
 * candidate values show colored delta indicators.
 * Only shows latency column when latency data is present.
 */

import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { fmtScaledPct1, fmtScaledPctInt, shortModel } from "@/lib/format-utils";
import { shortRunLabel } from "@/lib/compare-utils";
import type { EvalRunSummary, CompactModelRun } from "@/lib/types";
import { Delta } from "@/components/ui/delta";

/* ================================================================
 * Types
 * ================================================================ */

/** Per-model summary row for the summary table. */
interface ModelSummaryRow {
  modelId: string;
  /** Keyed by run file: prompt version, pass rate, per-metric rates, avg latency */
  perRun: Record<string, {
    promptVersion: string;
    passRate: number;
    metricRates: Record<string, number>;
    avgLatency: number | null;
  }>;
}

export interface SummaryTableProps {
  baselineFile: string;
  candidateFiles: string[];
  /** Loaded compact runs with the metric filter already applied upstream
   *  (see ``ComparePanel``). Per-metric columns derive from the
   *  data, so the filter narrows the table without a separate prop. */
  compactRuns: Record<string, CompactModelRun[]>;
  runSummaryByFile: Record<string, EvalRunSummary>;
  allModels: string[];
}

/* ================================================================
 * Sub-components
 * ================================================================ */

/** Render sub-header cells for one run group: Pass% | metric rates... | Latency */
function SubHeaders({ allMetrics, hasLatency }: { allMetrics: string[]; hasLatency: boolean }) {
  return (
    <>
      <TableHead className="text-xs text-muted-foreground border-l border-border/30 w-20">Prompt</TableHead>
      <TableHead className="text-xs text-right text-muted-foreground w-20">Pass%</TableHead>
      {allMetrics.map((m) => (
        <TableHead key={m} className="text-xs text-right text-muted-foreground w-20">{m}</TableHead>
      ))}
      {hasLatency && <TableHead className="text-xs text-right text-muted-foreground w-24">Avg Latency</TableHead>}
    </>
  );
}

/** Render data cells for one run within the summary table. */
function RunCells({ stats, blStats, allMetrics, hasLatency }: {
  stats: ModelSummaryRow["perRun"][string];
  blStats: ModelSummaryRow["perRun"][string] | null;
  allMetrics: string[];
  hasLatency: boolean;
}) {
  return (
    <>
      <TableCell className="text-xs text-muted-foreground border-l border-border/30">
        {stats.promptVersion}
      </TableCell>
      <TableCell className="text-xs text-right text-muted-foreground">
        <span className="font-semibold tabular-nums">{fmtScaledPct1(stats.passRate)}%</span>
        {blStats && <Delta value={stats.passRate - blStats.passRate} suffix="%" />}
      </TableCell>
      {allMetrics.map((m) => {
        const val = stats.metricRates[m];
        const blVal = blStats?.metricRates[m];
        return (
          <TableCell key={m} className="text-xs text-right text-muted-foreground">
            <span className="tabular-nums">{val != null ? `${fmtScaledPctInt(val)}%` : "---"}</span>
            {val != null && blVal != null && <Delta value={val - blVal} suffix="%" />}
          </TableCell>
        );
      })}
      {hasLatency && (
        <TableCell className="text-xs text-right text-muted-foreground">
          <span className="tabular-nums">{stats.avgLatency != null ? `${stats.avgLatency.toFixed(0)}ms` : "---"}</span>
          {blStats?.avgLatency != null && stats.avgLatency != null && (
            <Delta value={stats.avgLatency - blStats.avgLatency} suffix="ms" invert />
          )}
        </TableCell>
      )}
    </>
  );
}

/* ================================================================
 * Main export
 * ================================================================ */

/**
 * Summary table with rows = models, columns grouped by run.
 * Baseline values shown plain, candidate values show colored deltas.
 * Only shows latency column if latency data is present.
 */
export function SummaryTable({ baselineFile, candidateFiles, compactRuns, runSummaryByFile, allModels }: SummaryTableProps) {
  /** Compute per-model per-run stats */
  const { rows, allMetrics, hasLatency } = useMemo(() => {
    const allFiles = [baselineFile, ...candidateFiles];
    const metricSet = new Set<string>();
    let anyLatency = false;

    /* Build stats per (model, file) */
    const statsMap: Record<string, Record<string, { promptVersion: string; passRate: number; metricRates: Record<string, number>; avgLatency: number | null }>> = {};

    for (const file of allFiles) {
      const modelRuns = compactRuns[file] ?? [];
      const summary = runSummaryByFile[file];
      for (const mr of modelRuns) {
        const mid = mr.hyperparameters.model;
        if (!allModels.includes(mid)) continue;
        if (!statsMap[mid]) statsMap[mid] = {};

        const results = mr.results;
        const passed = results.filter((r) => r.success).length;
        const passRate = results.length > 0 ? (passed / results.length) * 100 : 0;

        /* Per-metric pass rates */
        const metricNames = [...new Set(results.flatMap((r) => r.metrics.map((m) => m.name)))];
        const metricRates: Record<string, number> = {};
        for (const mn of metricNames) {
          metricSet.add(mn);
          const mp = results.filter((r) => r.metrics.some((m) => m.name === mn && m.success)).length;
          metricRates[mn] = results.length > 0 ? (mp / results.length) * 100 : 0;
        }

        /* Average latency */
        const lats = results.map((r) => r.latency_ms).filter((v) => v > 0);
        const avgLat = lats.length > 0 ? lats.reduce((s, v) => s + v, 0) / lats.length : null;
        if (avgLat != null) anyLatency = true;

        // Use ``||`` (not ``??``) so an empty-string prompt_version on
        // either side falls through to the other source. ``??`` only
        // coalesces null/undefined and would render a blank cell when
        // ``summary.prompt_version === ""``.
        const promptVersion = summary?.prompt_version || mr.hyperparameters.prompt_version || "";
        statsMap[mid][file] = { promptVersion, passRate, metricRates, avgLatency: avgLat };
      }
    }

    const sortedMetrics = Array.from(metricSet).sort();
    const tableRows: ModelSummaryRow[] = allModels
      .filter((mid) => statsMap[mid])
      .map((mid) => ({ modelId: mid, perRun: statsMap[mid] }));

    return { rows: tableRows, allMetrics: sortedMetrics, hasLatency: anyLatency };
  }, [baselineFile, candidateFiles, compactRuns, runSummaryByFile, allModels]);

  const allFiles = [baselineFile, ...candidateFiles];

  return (
    <Card className="p-4 flex flex-col min-h-0">
      <h3 className="text-xs font-medium text-muted-foreground mb-3 shrink-0">Summary</h3>
      <div className="overflow-auto relative flex-1 min-h-0">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="text-xs w-32">Model</TableHead>
            {allFiles.map((file) => {
              const summary = runSummaryByFile[file];
              const label = summary ? shortRunLabel(summary) : file;
              const isBaseline = file === baselineFile;
              return (
                <TableHead key={`hdr-grp-${file}`} colSpan={2 + allMetrics.length + (hasLatency ? 1 : 0)} className="text-xs text-center text-muted-foreground border-l border-border/30">
                  {label}{isBaseline ? " (baseline)" : ""}
                </TableHead>
              );
            })}
          </TableRow>
          <TableRow className="hover:bg-transparent">
            <TableHead className="text-xs" />
            {allFiles.map((file) => (
              <SubHeaders key={`sub-${file}`} allMetrics={allMetrics} hasLatency={hasLatency} />
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const blStats = row.perRun[baselineFile];
            return (
              <TableRow key={row.modelId} className={rows.length > 1 ? "" : "bg-muted/20"}>
                <TableCell className="text-xs font-medium text-muted-foreground">{shortModel(row.modelId)}</TableCell>
                {allFiles.map((file) => {
                  const stats = row.perRun[file];
                  const isBaseline = file === baselineFile;
                  if (!stats) {
                    const emptyCols = 2 + allMetrics.length + (hasLatency ? 1 : 0);
                    return Array.from({ length: emptyCols }, (_, i) => (
                      <TableCell key={`empty-${file}-${i}`} className="text-xs text-center text-muted-foreground border-l border-border/30 first:border-l-border/30">---</TableCell>
                    ));
                  }
                  return (
                    <RunCells
                      key={`cells-${file}`}
                      stats={stats}
                      blStats={isBaseline ? null : (blStats ?? null)}
                      allMetrics={allMetrics}
                      hasLatency={hasLatency}
                    />
                  );
                })}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      </div>
    </Card>
  );
}
