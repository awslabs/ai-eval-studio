// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LatencyStatsTable component.
 *
 * Renders a table of latency percentile statistics (avg, median, P75, P95, P99,
 * min, max, count). When multiple models are present, shows one column per model.
 */

import { useMemo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { TestResult } from "@/lib/types";
import { getLatency } from "@/lib/eval-utils";
import { shortModel } from "@/lib/format-utils";

interface LatencyStatsTableProps {
  cases: TestResult[];
  modelIds?: string[];
  modelCasesMap?: Record<string, TestResult[]>;
}

interface StatsRow {
  count: number;
  avg: number;
  median: number;
  p75: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
}

function computeStats(cases: TestResult[]): StatsRow | null {
  const lats = cases.map((c) => getLatency(c)).filter((v): v is number => v != null && v > 0).sort((a, b) => a - b);
  if (lats.length === 0) return null;
  const n = lats.length;
  const pct = (p: number) => lats[Math.min(Math.floor(n * p), n - 1)];
  return { count: n, avg: lats.reduce((s, v) => s + v, 0) / n, median: pct(0.5), p75: pct(0.75), p95: pct(0.95), p99: pct(0.99), min: lats[0], max: lats[n - 1] };
}

const STAT_ROWS: { label: string; key: keyof StatsRow }[] = [
  { label: "Avg", key: "avg" },
  { label: "Median", key: "median" },
  { label: "P75", key: "p75" },
  { label: "P95", key: "p95" },
  { label: "P99", key: "p99" },
  { label: "Min", key: "min" },
  { label: "Max", key: "max" },
  { label: "Count", key: "count" },
];

/** Table of latency percentile statistics, optionally broken down by model. */
export function LatencyStatsTable({ cases, modelIds, modelCasesMap }: LatencyStatsTableProps) {
  const isMultiModel = modelIds != null && modelIds.length > 1 && modelCasesMap != null;

  const perModelStats = useMemo(() => {
    if (!isMultiModel) return null;
    const map: Record<string, StatsRow | null> = {};
    for (const mid of modelIds!) {
      map[mid] = computeStats(modelCasesMap![mid] ?? []);
    }
    return map;
  }, [isMultiModel, modelIds, modelCasesMap]);

  const allStats = useMemo(() => computeStats(cases), [cases]);

  if (!allStats) return <div className="text-sm text-muted-foreground py-8 text-center">No latency data</div>;

  const fmt = (key: keyof StatsRow, val: number) => key === "count" ? String(val) : `${val.toFixed(0)}ms`;

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-xs text-muted-foreground">Stat</TableHead>
            {isMultiModel ? (
              modelIds!.map((mid) => (
                <TableHead key={mid} className="text-xs text-muted-foreground text-right">{shortModel(mid)}</TableHead>
              ))
            ) : (
              <TableHead className="text-xs text-muted-foreground text-right">Value</TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {STAT_ROWS.map(({ label, key }) => (
            <TableRow key={label}>
              <TableCell className="text-xs text-muted-foreground py-1.5">{label}</TableCell>
              {isMultiModel ? (
                modelIds!.map((mid) => {
                  const s = perModelStats![mid];
                  return (
                    <TableCell key={mid} className="text-xs font-semibold tabular-nums text-muted-foreground text-right py-1.5">
                      {s ? fmt(key, s[key]) : "—"}
                    </TableCell>
                  );
                })
              ) : (
                <TableCell className="text-xs font-semibold tabular-nums text-muted-foreground text-right py-1.5">
                  {fmt(key, allStats[key])}
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}
