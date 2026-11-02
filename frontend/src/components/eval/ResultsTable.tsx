// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Multi-model results table for eval run test cases.
 * Shows one row per test case with per-model columns side by side.
 * Features: column sorting + resizing (via @tanstack/react-table), per-model
 * stats in headers, truncated output preview, colored score badges,
 * and click-to-expand detail.
 *
 * Sorting, resizing, and pagination are library-driven (TanStack v8); the
 * row body and expandable detail are bespoke, layered on the row model.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import {
  type ColumnDef,
  type SortingState,
  type PaginationState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { TablePagination } from "@/components/ui/table-pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { MaximizablePanel, PanelMaximizeButton, type MaximizablePanelHandle } from "@/components/ui/maximizable-panel";
import { MultiModelCaseDetail } from "./CaseDetail";
import type { MultiModelCase, ModelRun } from "@/lib/types";
import { computeModelStats } from "@/lib/eval-utils";
import { fmtPctInt, fmtScaledPctInt, shortModel } from "@/lib/format-utils";
import { useModelRunStats } from "@/hooks/useModelRunStats";

interface ResultsTableProps {
  /** Array of multi-model test cases to display */
  cases: MultiModelCase[];
  /** Model run metadata for computing per-model stats */
  modelRuns: ModelRun[];
  /** Which model IDs are currently visible (empty = all) */
  visibleModels: string[];
  /** Ordered list of metric names to render as inline chips. Already
   *  reflects the metric filter (computed upstream from the narrowed
   *  data), so this component doesn't take a separate filter prop. */
  metricNames: string[];
}

/**
 * Renders an interactive, sortable, resizable multi-model results table.
 *
 * Each row shows: row number, category, input, expected output, then one cell
 * per visible model containing pass/fail badge, truncated output, and inline
 * metric indicators. Sorting/resizing/pagination come from TanStack Table;
 * click-to-expand detail rows are layered on top.
 *
 * @param cases - Multi-model test cases (one row per case, models nested).
 *   These have already been narrowed to the active-metrics subset by
 *   ``applyMetricFilter`` upstream, so ``metrics_data`` on each case
 *   only contains the metrics the user wants to see.
 * @param modelRuns - Raw model run data for computing header stats
 * @param visibleModels - Which models to show columns for (empty = all)
 * @param metricNames - Metric names to render as inline chips. Already
 *   reflects the metric filter; passed in so the column ordering is
 *   stable across cases (deriving per-case would re-shuffle).
 */
export function ResultsTable({
  cases,
  modelRuns,
  visibleModels,
  metricNames,
}: ResultsTableProps) {
  /** All canonical model IDs from the runs (full Bedrock id; display
   *  goes through ``shortModel``). */
  const allModelIds = modelRuns.map((r) => r.hyperparameters.model);

  /** Effective models to display — filtered or all */
  const effectiveModels = visibleModels.length > 0 ? visibleModels : allModelIds;

  /** Visible metric names — already narrowed upstream. */
  const visibleMetrics = metricNames;

  /** Pre-computed aggregate stats per model — used for column header pass rate / latency / tokens. */
  const modelRunStats = useModelRunStats(modelRuns);

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 });
  /** Imperative handle for fullscreen control */
  const panelRef = useRef<MaximizablePanelHandle>(null);

  // Reset pagination when the underlying run changes — otherwise switching
  // runs can leave the user on an empty page N of the prior (larger) dataset.
  // Key on the first run's timestamp, which changes per eval run.
  const resetKey = modelRuns[0]?.timestamp ?? null;
  useEffect(() => {
    setPagination((p) => ({ ...p, pageIndex: 0 }));
  }, [resetKey]);

  /** Toggle expanded detail for a row (keyed by original case index). */
  function toggleExpand(index: number) {
    setExpandedId((prev) => (prev === String(index) ? null : String(index)));
  }

  /**
   * Column definitions. The fixed left columns sort by their string/index
   * value; each model column sorts by that model's pass/fail (success → 1).
   * Header content for model columns is the rich stat block; cells are
   * rendered by {@link MultiModelCaseRow} (not via flexRender) so the
   * bespoke row layout is preserved — under ``table-layout: fixed`` the
   * header widths govern the whole column, so body cells follow.
   */
  const columns = useMemo<ColumnDef<MultiModelCase>[]>(() => {
    const base: ColumnDef<MultiModelCase>[] = [
      { id: "index", accessorFn: (_r, i) => i, header: "#", size: 56, minSize: 44 },
      { id: "category", accessorFn: (r) => r.category ?? "", header: "Category", size: 120, minSize: 80 },
      { id: "input", accessorFn: (r) => r.input, header: "Input", size: 240, minSize: 120 },
      { id: "expected", accessorFn: (r) => r.expected_output, header: "Expected", size: 180, minSize: 120 },
    ];
    const modelCols: ColumnDef<MultiModelCase>[] = effectiveModels.map((modelId) => {
      /** Primary stats from hook — pass rate, latency, token totals. */
      const primary = modelRunStats.get(modelId);
      /** Detailed stats from util — metric rates, cache, cost. */
      const run = modelRuns.find((r) => r.hyperparameters.model === modelId);
      const detail = run ? computeModelStats(run) : null;
      const passed = primary ? Math.round(primary.passRate * primary.totalCases) : 0;
      return {
        id: `model__${modelId}`,
        accessorFn: (r) => (r.models[modelId]?.success ? 1 : 0),
        size: 300,
        minSize: 160,
        header: () => (
          <div className="flex flex-col items-start gap-1 py-1.5">
            <span className="font-semibold text-xs">{shortModel(modelId)}</span>
            {primary && (
              <>
                <span className={`text-2xs font-bold tabular-nums px-1.5 rounded ${rateBadgeClasses(primary.passRate * 100)}`}>
                  {fmtPctInt(primary.passRate)}% ({passed}/{primary.totalCases})
                </span>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-2xs tabular-nums">
                  {primary.avgLatencyMs != null && (
                    <span><span className="text-muted-foreground/60">Latency:</span> <span className="font-medium text-muted-foreground">{primary.avgLatencyMs.toFixed(0)}ms</span></span>
                  )}
                  {primary.totalInputTokens > 0 && (
                    <span><span className="text-muted-foreground/60">In:</span> <span className="font-medium text-muted-foreground">{primary.totalInputTokens.toLocaleString()}</span></span>
                  )}
                  {primary.totalOutputTokens > 0 && (
                    <span><span className="text-muted-foreground/60">Out:</span> <span className="font-medium text-muted-foreground">{primary.totalOutputTokens.toLocaleString()}</span></span>
                  )}
                  {(primary.totalInputTokens > 0 || primary.totalOutputTokens > 0) && (
                    <span><span className="text-muted-foreground/60">Total:</span> <span className="font-medium text-muted-foreground">{(primary.totalInputTokens + primary.totalOutputTokens).toLocaleString()}</span></span>
                  )}
                  {detail && <span><span className="text-muted-foreground/60">Cache Read:</span> <span className="font-medium text-muted-foreground">{detail.totalCacheRead.toLocaleString()}</span></span>}
                  {detail && <span><span className="text-muted-foreground/60">Cache Write:</span> <span className="font-medium text-muted-foreground">{detail.totalCacheWrite.toLocaleString()}</span></span>}
                  {primary.totalCost > 0 && (
                    <span><span className="text-muted-foreground/60">Cost:</span> <span className="font-medium text-muted-foreground">${primary.totalCost.toFixed(4)}</span></span>
                  )}
                </div>
                {detail && (
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(detail.metricRates).map(([name, rate]) => (
                      <span key={name} className={`text-2xs tabular-nums px-1 rounded ${rateBadgeClasses(rate)}`}>
                        {name.length > 15 ? name.slice(0, 15) + "…" : name} {fmtScaledPctInt(rate)}%
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ),
      };
    });
    return [...base, ...modelCols];
  }, [effectiveModels, modelRunStats, modelRuns]);

  const table = useReactTable<MultiModelCase>({
    data: cases,
    columns,
    state: { sorting, pagination },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    columnResizeMode: "onChange",
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  /** Column count for full-width rows (empty state + expanded detail). */
  const colSpanAll = table.getVisibleLeafColumns().length;

  const pageRows = table.getRowModel().rows;

  return (
    <MaximizablePanel ref={panelRef} title="Results" className="min-h-0 flex flex-col p-6">
      <Card className="p-4 flex flex-col flex-1 min-h-0">
        <div className="flex items-center mb-2 shrink-0">
          <PanelMaximizeButton panelRef={panelRef} label="Open results table fullscreen" />
        </div>
        <div className="overflow-auto flex-1 min-h-0 relative">
          <Table containerClassName="overflow-visible" style={{ tableLayout: "fixed", width: table.getTotalSize() }}>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="hover:bg-transparent">
                  {headerGroup.headers.map((header) => {
                    const sortDir = header.column.getIsSorted();
                    return (
                      <TableHead
                        key={header.id}
                        style={{ width: header.getSize(), position: "relative" }}
                        className="text-xs text-muted-foreground align-top select-none"
                      >
                        {/* Click the label to sort; the resize handle is a sibling
                         *  (outside this div) so dragging it never triggers a sort. */}
                        <div
                          onClick={header.column.getToggleSortingHandler()}
                          className="flex items-start gap-1 cursor-pointer hover:text-foreground transition-colors"
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {sortDir === "asc" ? (
                            <ArrowUp className="size-3 mt-0.5 shrink-0" />
                          ) : sortDir === "desc" ? (
                            <ArrowDown className="size-3 mt-0.5 shrink-0" />
                          ) : (
                            <ArrowUpDown className="size-3 mt-0.5 shrink-0 text-muted-foreground/50" />
                          )}
                        </div>
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          className={`absolute right-0 top-0 h-full w-1 cursor-col-resize touch-none select-none ${
                            header.column.getIsResizing() ? "bg-primary" : "bg-border opacity-0 hover:opacity-100"
                          }`}
                        />
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {pageRows.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={colSpanAll}
                    className="text-center text-muted-foreground py-8"
                  >
                    No cases match the current filters.
                  </TableCell>
                </TableRow>
              )}
              {pageRows.map((row) => {
                const c = row.original;
                const originalIndex = row.index;
                const rowId = String(originalIndex);
                const isExpanded = expandedId === rowId;
                const overallStatus = computeOverallStatus(c, effectiveModels);
                return (
                  <MultiModelCaseRow
                    key={rowId}
                    caseData={c}
                    displayIndex={originalIndex}
                    effectiveModels={effectiveModels}
                    visibleMetrics={visibleMetrics}
                    overallStatus={overallStatus}
                    isExpanded={isExpanded}
                    totalColumns={colSpanAll}
                    onClick={() => toggleExpand(originalIndex)}
                  />
                );
              })}
            </TableBody>
          </Table>
        </div>
        <div className="shrink-0">
          <TablePagination
            total={cases.length}
            page={pagination.pageIndex}
            pageSize={pagination.pageSize}
            onPageChange={(p) => table.setPageIndex(p)}
            onPageSizeChange={(s) => table.setPageSize(s)}
          />
        </div>
      </Card>
    </MaximizablePanel>
  );
}

/* -------------------------------------------------------------------
 * Overall status computation
 * ----------------------------------------------------------------- */

/**
 * Compute overall pass/fail/mixed status for a case across visible models.
 * - All pass -> "pass"
 * - All fail -> "fail"
 * - Mixed -> "mixed"
 * - No data -> "unknown"
 */
function computeOverallStatus(
  c: MultiModelCase,
  effectiveModels: string[],
): "pass" | "fail" | "mixed" | "unknown" {
  const validResults = effectiveModels
    .map((m) => c.models[m]?.success)
    .filter((v) => v != null);
  if (validResults.length === 0) return "unknown";
  if (validResults.every(Boolean)) return "pass";
  if (validResults.every((v) => !v)) return "fail";
  return "mixed";
}

/* -------------------------------------------------------------------
 * Multi-model case row + expandable detail
 * ----------------------------------------------------------------- */

interface MultiModelCaseRowProps {
  caseData: MultiModelCase;
  displayIndex: number;
  effectiveModels: string[];
  visibleMetrics: string[];
  overallStatus: "pass" | "fail" | "mixed" | "unknown";
  isExpanded: boolean;
  totalColumns: number;
  onClick: () => void;
}

/**
 * Renders a single multi-model case row with optional expanded detail panel.
 *
 * Left columns: #, Category, Input, Expected.
 * Then one cell per visible model showing PASS/FAIL badge, truncated output,
 * and inline metric indicators.
 */
function MultiModelCaseRow({
  caseData: c,
  displayIndex,
  effectiveModels,
  visibleMetrics,
  overallStatus,
  isExpanded,
  totalColumns,
  onClick,
}: MultiModelCaseRowProps) {
  /** Row background: subtle color based on overall status + focus ring */
  const rowBg = isExpanded
    ? "bg-muted/40 border-b-0"
    : overallStatus === "pass"
      ? "hover:bg-success/[0.04] bg-success/[0.02]"
      : overallStatus === "fail"
        ? "hover:bg-error/[0.04] bg-error/[0.02]"
        : overallStatus === "mixed"
          ? "hover:bg-warning/[0.04] bg-warning/[0.02]"
          : "";


  return (
    <>
      <TableRow
        className={`cursor-pointer transition-colors ${rowBg}`}
        onClick={onClick}
      >
        {/* # */}
        <TableCell className="font-mono text-xs text-muted-foreground tabular-nums">
          {displayIndex + 1}
        </TableCell>
        {/* Category */}
        <TableCell className="text-xs text-muted-foreground truncate">
          {c.category ?? "—"}
        </TableCell>
        {/* Input — truncates to the column's current width; widen to see more,
         *  full text on hover and in the expanded detail row. */}
        <TableCell
          className="text-xs truncate text-muted-foreground"
          title={c.input}
        >
          {c.input}
        </TableCell>
        {/* Expected output — same width-governed truncation. */}
        <TableCell
          className="text-xs font-mono truncate text-muted-foreground"
          title={c.expected_output}
        >
          {c.expected_output}
        </TableCell>
        {/* Per-model cells */}
        {effectiveModels.map((modelId) => {
          const modelData = c.models[modelId];
          if (!modelData) {
            return (
              <TableCell key={modelId} className="text-xs text-muted-foreground text-center">
                {"—"}
              </TableCell>
            );
          }
          return (
            <TableCell key={modelId} className="align-top" title={modelData.actual_output}>
              <div className="flex items-center gap-1.5 mb-1">
                <Badge
                  className={`text-2xs font-bold px-1.5 py-0 ${
                    modelData.success
                      ? "bg-success/15 text-success"
                      : "bg-error/15 text-error"
                  }`}
                >
                  {modelData.success ? "PASS" : "FAIL"}
                </Badge>
              </div>
              {/* Output preview: capped to 2 lines (vertical), wraps to the
               *  column's current width. Full text on hover + in the expanded row. */}
              <div className="text-xs font-mono text-muted-foreground break-words line-clamp-2">
                {modelData.actual_output}
              </div>
              <div className="flex flex-wrap gap-1 mt-1">
                {visibleMetrics.map((name) => {
                  const md = modelData.metrics_data.find((m) => m.name === name);
                  if (!md) return null;
                  return (
                    <span
                      key={name}
                      className="inline-flex items-center gap-0.5 text-2xs font-medium px-1 py-px rounded bg-muted text-muted-foreground"
                    >
                      <span className={md.success ? "text-success" : "text-error"}>{md.success ? "✓" : "✗"}</span>
                      {" "}{name.length > 12 ? name.slice(0, 12) + "…" : name} {md.score != null ? md.score.toFixed(1) : "—"}
                    </span>
                  );
                })}
              </div>
            </TableCell>
          );
        })}
      </TableRow>

      {/* Expanded detail panel — per-model sections with expandable reasons */}
      {isExpanded && (
        <TableRow className="hover:bg-transparent">
          <TableCell colSpan={totalColumns} className="p-0">
            <MultiModelCaseDetail caseData={c} visibleModels={effectiveModels} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

/** Returns Tailwind badge classes based on pass rate using semantic colors. */
function rateBadgeClasses(pct: number): string {
  if (pct >= 70) return "bg-success/10 text-success";
  if (pct >= 40) return "bg-warning/10 text-warning";
  return "bg-error/10 text-error";
}
