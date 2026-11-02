// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * File content preview component for the Data tab.
 * Renders files with format-appropriate viewers: a table for parsed CSV,
 * and a read-only Monaco editor for everything else (JSON, code,
 * markdown, plain text).
 *
 * The CSV branch (``CsvPreview``) is built on ``@tanstack/react-table`` (v8)
 * so column resizing, sorting, and pagination come from the library rather
 * than hand-rolled state. The visual shell (shadcn Table primitives, badge
 * header) is preserved.
 *
 * FULLSCREEN LIVES AT THE CONTAINER, not in a viewer. It used to sit inside
 * ``CsvPreview``, so exactly one of the Data tab's file types could be
 * maximized: opening a JSON or markdown file made the control vanish, which
 * reads as a broken button rather than a per-format capability. The panel and
 * its one maximize control now wrap the branch, so every viewer inherits them
 * and a new viewer gets fullscreen without opting in.
 */

import { useEffect, useMemo, useRef, useState } from "react";
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
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import Editor from "@monaco-editor/react";
import { Badge } from "@/components/ui/badge";
import { defineAppTheme, getMonacoTheme } from "@/lib/monaco-theme";
import { monacoLanguageFor } from "@/lib/monaco-language";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import { TablePagination } from "@/components/ui/table-pagination";
import { useTheme } from "@/lib/theme-provider";
import { MaximizablePanel, PanelMaximizeButton, type MaximizablePanelHandle } from "@/components/ui/maximizable-panel";
import type { FileContentResponse } from "@/lib/types";

/** A parsed tabular row: column name → cell value. */
type CsvRow = Record<string, unknown>;

/** Props for the FilePreview component. */
interface FilePreviewProps {
  /** File content response from the backend */
  file: FileContentResponse;
}

/**
 * Renders a file preview based on its detected language, inside one
 * fullscreen-capable container.
 *
 * - CSV/tabular: scrollable, sortable, resizable table with pagination
 * - Everything else: Monaco read-only editor with syntax highlighting
 *
 * Owns the {@link MaximizablePanel} and the single maximize control for both
 * branches. The viewers below render bodies only — they neither know nor care
 * that they can be maximized, which is what keeps the two branches from
 * drifting apart on this.
 *
 * @param file - File content response with content, language, and optional parsed rows
 */
export function FilePreview({ file }: FilePreviewProps) {
  const panelRef = useRef<MaximizablePanelHandle>(null);

  /** Whether the backend parsed this file into rows we can table. */
  const isTabular = file.language === "csv" && !!file.rows && !!file.columns;

  /** Basename, for the fullscreen header and the control's accessible label. */
  const fileName = file.path.split("/").pop() || "File preview";

  return (
    <MaximizablePanel ref={panelRef} title={fileName} className="min-h-0 flex flex-col p-6">
      <div className="flex flex-col h-full min-h-0">
        <div className="flex items-center gap-2 mb-2 shrink-0">
          {isTabular && <TabularBadges file={file} />}
          {/* ml-auto lives on the button, so it right-aligns whether or not the
              branch contributed any badges. */}
          <PanelMaximizeButton panelRef={panelRef} label={`Open ${fileName} fullscreen`} />
        </div>
        {isTabular ? <CsvPreview file={file} /> : <CodePreview file={file} />}
      </div>
    </MaximizablePanel>
  );
}

/**
 * Row/column counts and the truncation warning for a tabular file.
 *
 * Split out of {@link CsvPreview} because the container owns the header row now
 * (that is where the shared maximize control sits), while the counts are
 * meaningful only for the tabular branch.
 *
 * @param file - file content response with parsed ``rows`` + ``columns``
 */
function TabularBadges({ file }: { file: FileContentResponse }) {
  const rowCount = (file.rows ?? []).length;
  const columnCount = (file.columns ?? []).length;
  const totalRows = file.total_rows ?? rowCount;
  const isTruncated = totalRows > rowCount;

  return (
    <>
      <Badge variant="secondary" className="text-2xs">{columnCount} columns</Badge>
      <Badge variant="secondary" className="text-2xs">
        {isTruncated
          ? `${rowCount.toLocaleString()} of ${totalRows.toLocaleString()} rows`
          : `${rowCount.toLocaleString()} rows`}
      </Badge>
      {isTruncated && (
        <span
          className="text-2xs text-warning"
          title={`This file has ${totalRows.toLocaleString()} rows. The preview is capped at ${rowCount.toLocaleString()} — use the source file for full access.`}
        >
          preview truncated
        </span>
      )}
    </>
  );
}

/**
 * Read-only Monaco editor for any file the backend did not parse into rows.
 *
 * ``automaticLayout`` is required, not cosmetic: Monaco measures its container
 * once on mount and otherwise never again, so it keeps the resting pane's
 * dimensions after the container is maximized and renders into a fraction of
 * the overlay with the rest blank.
 *
 * @param file - file content response carrying raw ``content`` + ``language``
 */
function CodePreview({ file }: { file: FileContentResponse }) {
  const { mode } = useTheme();
  const isDark = mode === "dark";

  return (
    <div className="flex-1 min-h-0 border rounded-md overflow-hidden bg-card">
      <Editor
        height="100%"
        language={monacoLanguageFor(file.language)}
        value={file.content}
        theme={getMonacoTheme(isDark)}
        options={{
          readOnly: true,
          minimap: { enabled: false },
          fontSize: 12,
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          wordWrap: "on",
          folding: true,
          padding: { top: 8 },
          automaticLayout: true,
        }}
        beforeMount={defineAppTheme}
      />
    </div>
  );
}

/**
 * Paginated, resizable, sortable table view for parsed CSV / tabular data.
 *
 * Built on ``@tanstack/react-table`` v8: the table instance owns sorting,
 * pagination, and column sizing state. Columns are resizable by dragging the
 * right edge of a header and sortable by clicking the header label. The shared
 * {@link TablePagination} bar is driven by the table instance's pagination
 * state (the component itself is unchanged, so the eval/compare tables that
 * also use it are unaffected).
 *
 * Renders a body only. The enclosing {@link FilePreview} owns the fullscreen
 * panel, the header row, and the counts — this component previously owned all
 * three, which is why no other file type could be maximized.
 *
 * @param file - file content response with parsed ``rows`` + ``columns``
 */
function CsvPreview({ file }: { file: FileContentResponse }) {
  const rows = (file.rows ?? []) as CsvRow[];
  // Memoized because ``?? []`` produces a NEW array identity on every render
  // when ``columns`` is absent, which defeats the column-definition useMemo
  // below (it would rebuild every render, remounting the header cells).
  const columnNames = useMemo(() => file.columns ?? [], [file.columns]);

  const scrollRef = useRef<HTMLDivElement>(null);

  const [sorting, setSorting] = useState<SortingState>([]);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 });

  // Width of the scroll container, tracked so columns can fill the pane on
  // load (and on file change / resize) instead of defaulting to 150px each
  // and leaving empty space when there are few columns.
  const [paneWidth, setPaneWidth] = useState(0);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w > 0) setPaneWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Reset to the first page when the user opens a different file so they're
  // not stranded on a page index the new (possibly shorter) data lacks.
  useEffect(() => {
    setPagination((p) => ({ ...p, pageIndex: 0 }));
    setSorting([]);
  }, [file.path]);

  /**
   * Even per-column width that fills the pane on load. Floored at a sensible
   * minimum so many columns still scroll rather than collapsing to slivers.
   * Recomputed when the pane width or column count changes; user drags then
   * override individual columns via the table's columnSizing state.
   */
  const defaultColWidth = useMemo(() => {
    if (paneWidth <= 0 || columnNames.length === 0) return 150;
    return Math.max(120, Math.floor(paneWidth / columnNames.length));
  }, [paneWidth, columnNames.length]);

  /** One column def per CSV column; cells stringify the raw value. */
  const columns = useMemo<ColumnDef<CsvRow>[]>(
    () =>
      columnNames.map((name) => ({
        id: name,
        accessorFn: (row) => row[name],
        header: name,
        cell: (ctx) => String(ctx.getValue() ?? ""),
        size: defaultColWidth,
      })),
    [columnNames, defaultColWidth],
  );

  const table = useReactTable<CsvRow>({
    data: rows,
    columns,
    state: { sorting, pagination },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    columnResizeMode: "onChange",
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div ref={scrollRef} className="overflow-auto flex-1 min-h-0 border rounded-md">
        <Table containerClassName="overflow-visible" style={{ tableLayout: "fixed", width: table.getTotalSize() }}>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sortDir = header.column.getIsSorted();
                  return (
                    <TableHead
                      key={header.id}
                      style={{ width: header.getSize(), position: "relative" }}
                      className="text-2xs font-medium text-muted-foreground select-none"
                    >
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className="flex items-center gap-1 whitespace-nowrap outline-none hover:text-foreground transition-colors"
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {sortDir === "asc" ? (
                          <ArrowUp className="size-3" />
                        ) : sortDir === "desc" ? (
                          <ArrowDown className="size-3" />
                        ) : (
                          <ArrowUpDown className="size-3 opacity-40" />
                        )}
                      </button>
                      {/* Resize handle — drag the right edge of the header. */}
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
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    style={{ width: cell.column.getSize() }}
                    className="text-xs text-muted-foreground whitespace-nowrap max-w-[400px] truncate"
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <TablePagination
        total={table.getFilteredRowModel().rows.length}
        page={pagination.pageIndex}
        pageSize={pagination.pageSize}
        onPageChange={(p) => table.setPageIndex(p)}
        onPageSizeChange={(s) => table.setPageSize(s)}
      />
    </div>
  );
}
