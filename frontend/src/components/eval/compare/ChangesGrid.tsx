// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Changes grid for the compare panel.
 *
 * Per-case diff view with filter tabs (all changes / regressions / fixes /
 * changed / same), status badges, and pagination. Each row is expandable to
 * reveal the full ExpandedDetail pane.
 *
 * Column sizing, sorting (Status / Model / Case only), and pagination are
 * driven by @tanstack/react-table v8. The filter tabs, the rich diff cells
 * (RunCell), and the expandable detail rows remain bespoke, layered on top of
 * the table's row model.
 */

import { useEffect, useMemo, useState, useCallback, useRef } from "react";
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
import { MaximizablePanel, PanelMaximizeButton, type MaximizablePanelHandle } from "@/components/ui/maximizable-panel";
import { Card } from "@/components/ui/card";
import { TablePagination } from "@/components/ui/table-pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { shortModel } from "@/lib/format-utils";
import { shortRunLabel } from "@/lib/compare-utils";
import { StatusBadge, ExpandedDetail } from "./ExpandedDetail";
import type { CandidateResult, CaseStatus, ChangeRow } from "./ExpandedDetail";
import type { EvalRunSummary, CompactModelRun } from "@/lib/types";

/* ================================================================
 * Types
 * ================================================================ */

/** Filter tab option for the changes grid. */
type ChangeFilter = "all_changes" | "regressions" | "fixes" | "changed" | "same";

/** Severity rank for sorting the Status column (regression first when desc). */
const STATUS_RANK: Record<CaseStatus, number> = { regression: 3, fixed: 2, changed: 1, same: 0 };

export interface ChangesGridProps {
  projectId: string;
  baselineFile: string;
  candidateFiles: string[];
  /** Loaded compact runs with the metric filter already applied upstream
   *  (see ``ComparePanel``). Per-metric scores are pre-narrowed; this
   *  component just renders what it gets. */
  compactRuns: Record<string, CompactModelRun[]>;
  runSummaryByFile: Record<string, EvalRunSummary>;
  allModels: string[];
}

/* ================================================================
 * RunCell
 * ================================================================ */

/** Render a cell for one run in the changes grid (PASS/FAIL pill + output preview + metric indicators). */
function RunCell({ passed, outputPreview, scores, passes, status }: {
  passed: boolean;
  outputPreview: string;
  scores: Record<string, number>;
  passes: Record<string, boolean>;
  status: CaseStatus | null;
}) {
  return (
    <TableCell className="text-xs border-l border-border/30">
      <div className="flex items-center gap-1.5">
        <span className={`inline-block text-2xs font-bold px-1.5 py-0 rounded ${passed ? "bg-success/15 text-success" : "bg-error/15 text-error"}`}>
          {passed ? "PASS" : "FAIL"}
        </span>
        {status && status !== "same" && (
          <StatusBadge status={status} size="small" />
        )}
      </div>
      {/* Output preview truncates to the column's current width — widen the
          column to reveal more; full text on expand. */}
      <span className="block mt-1 font-mono text-2xs text-muted-foreground truncate" title={outputPreview}>
        {outputPreview}
      </span>
      {/* Metric indicators — scores are already pre-narrowed by the
          metric filter applied upstream in ``ComparePanel``. */}
      <div className="flex flex-wrap gap-1 mt-1">
        {Object.entries(scores).map(([name, score]) => {
          const mp = passes[name] ?? false;
          return (
            <span
              key={name}
              className="inline-flex items-center gap-0.5 text-2xs font-medium px-1 py-px rounded bg-muted text-muted-foreground"
            >
              <span className={mp ? "text-success" : "text-error"}>{mp ? "✓" : "✗"}</span>
              {" "}{name.length > 12 ? name.slice(0, 12) + "…" : name} {score.toFixed(1)}
            </span>
          );
        })}
      </div>
    </TableCell>
  );
}

/* ================================================================
 * Helper for compact metrics
 * ================================================================ */

/** Extract scores/passes from CompactMetric[] (compact shape, no reasons). */
function extractCompactMetrics(metrics: Array<{ name: string; success: boolean; score: number | null }>): {
  scores: Record<string, number>;
  passes: Record<string, boolean>;
} {
  const scores: Record<string, number> = {};
  const passes: Record<string, boolean> = {};
  for (const m of metrics) {
    scores[m.name] = m.score ?? 0;
    passes[m.name] = m.success;
  }
  return { scores, passes };
}

/* ================================================================
 * ChangeRowComponent
 * ================================================================ */

interface ChangeRowProps {
  projectId: string;
  baselineFile: string;
  row: ChangeRow;
  candidateFiles: string[];
  runSummaryByFile: Record<string, EvalRunSummary>;
  blShort: string;
  isExpanded: boolean;
  onToggle: () => void;
  totalCols: number;
}

/** A single row in the changes grid with expandable detail. */
function ChangeRowComponent({ projectId, baselineFile, row, candidateFiles, runSummaryByFile, blShort, isExpanded, onToggle, totalCols }: ChangeRowProps) {
  return (
    <>
      {/* Summary row */}
      <TableRow className={`cursor-pointer ${row.overallStatus === "regression" ? "bg-error/[0.02] hover:bg-error/[0.04]" : row.overallStatus === "fixed" ? "bg-success/[0.02] hover:bg-success/[0.04]" : row.overallStatus === "changed" ? "bg-warning/[0.02] hover:bg-warning/[0.04]" : "hover:bg-accent/30"}`} onClick={onToggle}>
        <TableCell className="text-xs">
          <StatusBadge status={row.overallStatus} />
        </TableCell>
        <TableCell className="text-xs text-muted-foreground truncate">
          {shortModel(row.model)}
        </TableCell>
        <TableCell className="text-xs">
          <div className="truncate">
            <span className="text-muted-foreground">#{row.caseId}</span>
            {row.category && <span className="text-muted-foreground ml-1.5">{row.category}</span>}
          </div>
          {/* Input preview truncates to the Case column's current width. */}
          <span className="block truncate text-muted-foreground" title={row.inputPreview}>{row.inputPreview}</span>
        </TableCell>

        {/* Baseline cell */}
        <RunCell
          passed={row.blPassed}
          outputPreview={row.blOutputPreview}
          scores={row.blScores}
          passes={row.blPasses}
          status={null}
        />

        {/* Candidate cells */}
        {candidateFiles.map((file) => {
          const cand = row.candidates[file];
          if (!cand) {
            return (
              <TableCell key={file} className="text-xs text-muted-foreground border-l border-border/30">
                ---
              </TableCell>
            );
          }
          return (
            <RunCell
              key={file}
              passed={cand.passed}
              outputPreview={cand.outputPreview}
              scores={cand.scores}
              passes={cand.passes}
              status={cand.status}
            />
          );
        })}
      </TableRow>

      {/* Expanded detail */}
      {isExpanded && (
        <TableRow className="hover:bg-transparent">
          <TableCell colSpan={totalCols} className="p-0">
            <ExpandedDetail
              projectId={projectId}
              baselineFile={baselineFile}
              row={row}
              candidateFiles={candidateFiles}
              runSummaryByFile={runSummaryByFile}
              blShort={blShort}
            />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

/* ================================================================
 * Main export
 * ================================================================ */

/**
 * Per-case changes grid showing regressions, fixes, output changes, and unchanged cases.
 * Supports multiple candidates. Each row is expandable to show full detail.
 */
export function ChangesGrid({ projectId, baselineFile, candidateFiles, compactRuns, runSummaryByFile, allModels }: ChangesGridProps) {
  const [changeFilter, setChangeFilter] = useState<ChangeFilter>("all_changes");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 });
  /** Imperative handle for fullscreen control */
  const panelRef = useRef<MaximizablePanelHandle>(null);
  /** Scroll container, measured so columns fill the pane width (no empty
   *  column on the right when the base column widths sum narrower than it). */
  const scrollRef = useRef<HTMLDivElement>(null);
  const [paneWidth, setPaneWidth] = useState(0);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let rafId = 0;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w <= 0) return;
      // Coalesce into a frame and ignore sub-pixel noise so the observer ↔
      // column-width relayout can't feed back into itself (visible jitter).
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        setPaneWidth((prev) => (Math.abs(w - prev) > 1 ? w : prev));
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  const toggleRow = useCallback((id: string) => {
    setExpandedRow((prev) => (prev === id ? null : id));
  }, []);

  /** Build change rows from compact data */
  const changeRows = useMemo(() => {
    const rows: ChangeRow[] = [];
    const blModelRuns = compactRuns[baselineFile] ?? [];

    for (const mid of allModels) {
      const blMr = blModelRuns.find((mr) => mr.hyperparameters.model === mid);
      if (!blMr) continue;

      for (const blResult of blMr.results) {
        const caseId = blResult.case_id;
        const blPassed = blResult.success;
        const blOutputHash = blResult.output_hash;
        const blMetrics = extractCompactMetrics(blResult.metrics);

        const candidates: Record<string, CandidateResult | null> = {};
        let overallStatus: CaseStatus = "same";

        for (const candFile of candidateFiles) {
          const candModelRuns = compactRuns[candFile] ?? [];
          const candMr = candModelRuns.find((mr) => mr.hyperparameters.model === mid);
          if (!candMr) {
            candidates[candFile] = null;
            continue;
          }

          /* Match case by case_id */
          const candResult = candMr.results.find((r) => r.case_id === caseId);

          if (!candResult) {
            candidates[candFile] = null;
            continue;
          }

          const cPassed = candResult.success;
          const cOutputHash = candResult.output_hash;
          const cMetrics = extractCompactMetrics(candResult.metrics);

          let status: CaseStatus = "same";
          if (blPassed && !cPassed) status = "regression";
          else if (!blPassed && cPassed) status = "fixed";
          else if (blOutputHash !== cOutputHash) status = "changed";

          /* Escalate overall status: regression > fixed > changed > same */
          if (status === "regression") overallStatus = "regression";
          else if (status === "fixed" && overallStatus !== "regression") overallStatus = "fixed";
          else if (status === "changed" && overallStatus === "same") overallStatus = "changed";

          candidates[candFile] = {
            passed: cPassed,
            outputPreview: candResult.output_preview,
            outputHash: cOutputHash,
            status,
            scores: cMetrics.scores,
            passes: cMetrics.passes,
            latencyMs: candResult.latency_ms,
            inputTokens: candResult.input_tokens,
            outputTokens: candResult.output_tokens,
            cacheRead: candResult.cache_read_input_tokens,
            cacheWrite: candResult.cache_write_input_tokens,
          };
        }

        rows.push({
          model: mid,
          caseId,
          category: blResult.category ?? "",
          inputPreview: blResult.input_preview,
          blPassed,
          blOutputPreview: blResult.output_preview,
          blOutputHash,
          blScores: blMetrics.scores,
          blPasses: blMetrics.passes,
          blLatency: blResult.latency_ms,
          blInputTokens: blResult.input_tokens,
          blOutputTokens: blResult.output_tokens,
          blCacheRead: blResult.cache_read_input_tokens,
          blCacheWrite: blResult.cache_write_input_tokens,
          overallStatus,
          candidates,
        });
      }
    }

    return rows;
  }, [baselineFile, candidateFiles, compactRuns, allModels]);

  /** Counts by status */
  const counts = useMemo(() => {
    const c = { regression: 0, fixed: 0, changed: 0, same: 0 };
    for (const r of changeRows) c[r.overallStatus]++;
    return c;
  }, [changeRows]);

  /** Filtered rows based on active filter tab */
  const filteredRows = useMemo(() => {
    switch (changeFilter) {
      case "all_changes": return changeRows.filter((r) => r.overallStatus !== "same");
      case "regressions": return changeRows.filter((r) => r.overallStatus === "regression");
      case "fixes": return changeRows.filter((r) => r.overallStatus === "fixed");
      case "changed": return changeRows.filter((r) => r.overallStatus === "changed");
      case "same": return changeRows.filter((r) => r.overallStatus === "same");
    }
  }, [changeRows, changeFilter]);

  const blShort = runSummaryByFile[baselineFile] ? shortRunLabel(runSummaryByFile[baselineFile]) : baselineFile;

  /**
   * Columns. Status / Model / Case are sortable via their accessor; the
   * Baseline and Candidate columns hold rich diff cells (RunCell) so they are
   * not sortable. The body rows are rendered by {@link ChangeRowComponent}
   * (not via flexRender) — under ``table-layout: fixed`` the header widths
   * govern the whole column, so the bespoke body cells follow.
   */
  const columns = useMemo<ColumnDef<ChangeRow>[]>(() => {
    // Narrow fixed columns keep their base width; the flexible text columns
    // (Case, Baseline, each Candidate) share any leftover pane width so the
    // table fills the pane with no empty column on the right. When the base
    // widths already exceed the pane, no scaling is applied and it scrolls.
    const FIXED = 96 + 110; // status + model
    const flexBase = 240 + 260 + 260 * candidateFiles.length; // case + baseline + candidates
    const leftover = Math.max(0, paneWidth - FIXED - flexBase);
    const grow = (base: number) => Math.round(base + leftover * (base / flexBase));

    const cols: ColumnDef<ChangeRow>[] = [
      { id: "status", accessorFn: (r) => STATUS_RANK[r.overallStatus], header: "Status", size: 96, minSize: 72 },
      { id: "model", accessorFn: (r) => shortModel(r.model), header: "Model", size: 110, minSize: 80 },
      { id: "case", accessorFn: (r) => r.caseId, header: "Case", size: grow(240), minSize: 140 },
      { id: "baseline", header: `Baseline (${blShort})`, enableSorting: false, size: grow(260), minSize: 160 },
    ];
    for (const file of candidateFiles) {
      const summary = runSummaryByFile[file];
      const label = summary ? shortRunLabel(summary) : file;
      cols.push({ id: `cand__${file}`, header: label, enableSorting: false, size: grow(260), minSize: 160 });
    }
    return cols;
  }, [candidateFiles, runSummaryByFile, blShort, paneWidth]);

  const table = useReactTable<ChangeRow>({
    data: filteredRows,
    columns,
    state: { sorting, pagination },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    columnResizeMode: "onChange",
    // Include the data-array index so the id is unique even if two baseline
    // results share a caseId within one model (the prior React key carried
    // idx for the same reason). Computed from the source array, so it stays
    // bound to the right row across sort + pagination.
    getRowId: (row, index) => `${row.model}-${row.caseId}-${index}`,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  // Reset pagination + collapse any open row when the inputs to the row list
  // change — new baseline, new candidate set, or a filter-tab switch. Without
  // this, clicking "Fixes" while on page 3 of "All Changes" leaves you on a
  // likely-empty page.
  const resetKey = `${baselineFile}|${candidateFiles.join(",")}|${changeFilter}`;
  useEffect(() => {
    setPagination((p) => ({ ...p, pageIndex: 0 }));
    setExpandedRow(null);
  }, [resetKey]);

  /** Column count for full-width rows (empty state + expanded detail). */
  const totalCols = table.getVisibleLeafColumns().length;

  const filterTabs: { key: ChangeFilter; label: string; count: number }[] = [
    { key: "all_changes", label: "All Changes", count: counts.regression + counts.fixed + counts.changed },
    { key: "regressions", label: "Regressions", count: counts.regression },
    { key: "fixes", label: "Fixes", count: counts.fixed },
    { key: "changed", label: "Changed", count: counts.changed },
    { key: "same", label: "Same", count: counts.same },
  ];

  // The grid is rendered in two layout modes: content-height in the normal
  // (stacked) compare pane so a short list leaves no gap below, and fill-height
  // in the fullscreen overlay. ``fill`` selects between them. Passed as the
  // panel's resting ``children`` (false) and ``maximizedContent`` (true).
  const renderGrid = (fill: boolean) => (
    <Card className={fill ? "p-4 flex flex-col h-full" : "p-4 flex flex-col"}>
      {/* Summary counts */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <h3 className="text-xs font-medium text-muted-foreground">Changes</h3>
        <span className="text-xs text-foreground/80">
          {counts.regression > 0 && <><span className="font-semibold text-error">{counts.regression}</span> regressions</>}
          {counts.regression > 0 && (counts.fixed > 0 || counts.changed > 0 || counts.same > 0) && <span className="text-muted-foreground"> · </span>}
          {counts.fixed > 0 && <><span className="font-semibold text-success">{counts.fixed}</span> fixes</>}
          {counts.fixed > 0 && (counts.changed > 0 || counts.same > 0) && <span className="text-muted-foreground"> · </span>}
          {counts.changed > 0 && <><span className="font-semibold text-warning">{counts.changed}</span> output changes</>}
          {counts.changed > 0 && counts.same > 0 && <span className="text-muted-foreground"> · </span>}
          {counts.same > 0 && <><span className="font-semibold text-muted-foreground">{counts.same}</span> unchanged</>}
        </span>
        <PanelMaximizeButton panelRef={panelRef} label="Open changes table fullscreen" />
      </div>

      {/* Filter tabs */}
      <div className="inline-flex gap-0.5 bg-background border border-border rounded-lg p-1 mb-3 flex-wrap" role="tablist">
        {filterTabs.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={changeFilter === tab.key}
            className={`h-7 text-xs px-3 rounded-md font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring ${changeFilter === tab.key ? "bg-accent text-accent-foreground border border-border shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
            onClick={() => { setChangeFilter(tab.key); setExpandedRow(null); }}
          >
            {tab.label} ({tab.count})
          </button>
        ))}
      </div>

      {/* Changes table — always rendered to prevent layout shift. In normal
       *  view it caps its height (so a short list doesn't leave a gap below in
       *  the stacked compare pane); in fullscreen it fills the screen. */}
      <div
        ref={scrollRef}
        className={`overflow-auto relative ${fill ? "flex-1 min-h-0" : ""}`}
      >
        {filteredRows.length === 0 ? (
          <p className="text-sm text-muted-foreground py-8 text-center">
            {changeFilter === "all_changes"
              ? "No differences between the selected runs."
              : "No cases match this filter."}
          </p>
        ) : (
          <Table containerClassName="overflow-visible" style={{ tableLayout: "fixed", width: table.getTotalSize() }}>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="hover:bg-transparent">
                  {headerGroup.headers.map((header) => {
                    const canSort = header.column.getCanSort();
                    const sortDir = header.column.getIsSorted();
                    // Baseline + candidate columns carry a left divider, matching
                    // the RunCell body cells.
                    const hasDivider = header.column.id === "baseline" || header.column.id.startsWith("cand__");
                    return (
                      <TableHead
                        key={header.id}
                        style={{ width: header.getSize(), position: "relative" }}
                        className={`text-xs text-muted-foreground select-none ${hasDivider ? "border-l border-border/30" : ""}`}
                      >
                        <div
                          onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                          className={`flex items-center gap-1 ${canSort ? "cursor-pointer hover:text-foreground transition-colors" : ""}`}
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {canSort && (
                            sortDir === "asc" ? (
                              <ArrowUp className="size-3 shrink-0" />
                            ) : sortDir === "desc" ? (
                              <ArrowDown className="size-3 shrink-0" />
                            ) : (
                              <ArrowUpDown className="size-3 shrink-0 text-muted-foreground/50" />
                            )
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
              {table.getRowModel().rows.map((tanstackRow) => (
                <ChangeRowComponent
                  key={tanstackRow.id}
                  projectId={projectId}
                  baselineFile={baselineFile}
                  row={tanstackRow.original}
                  candidateFiles={candidateFiles}
                  runSummaryByFile={runSummaryByFile}
                  blShort={blShort}
                  isExpanded={expandedRow === tanstackRow.id}
                  onToggle={() => toggleRow(tanstackRow.id)}
                  totalCols={totalCols}
                />
              ))}
            </TableBody>
          </Table>
        )}
      </div>
      {filteredRows.length > 0 && (
        <div className="shrink-0">
          <TablePagination
            total={filteredRows.length}
            page={pagination.pageIndex}
            pageSize={pagination.pageSize}
            onPageChange={(p) => table.setPageIndex(p)}
            onPageSizeChange={(s) => table.setPageSize(s)}
          />
        </div>
      )}
    </Card>
  );

  return (
    <MaximizablePanel
      ref={panelRef}
      title="Changes"
      className="min-h-0 flex flex-col p-6"
      maximizedContent={renderGrid(true)}
    >
      {renderGrid(false)}
    </MaximizablePanel>
  );
}
