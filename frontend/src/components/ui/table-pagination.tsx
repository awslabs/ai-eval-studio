// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Reusable table pagination controls.
 * Shows page info, page size selector, and navigation buttons.
 * Presentational only — page/pageSize state is owned by the caller
 * (the eval/file tables drive it from TanStack Table's pagination model).
 */

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/** Available page size options. */
const PAGE_SIZE_OPTIONS = [25, 50, 100, 250];

interface TablePaginationProps {
  /** Total number of items (unsliced). */
  total: number;
  /** Current page index (0-based). */
  page: number;
  /** Items per page. */
  pageSize: number;
  /** Called when page changes. */
  onPageChange: (page: number) => void;
  /** Called when page size changes (resets to page 0). */
  onPageSizeChange: (size: number) => void;
}

/**
 * Pagination bar with page size selector, page info, and navigation.
 * Renders below a table. The page-size selector and range summary always
 * show; the navigation buttons appear only when there is more than one page.
 */
export function TablePagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: TablePaginationProps) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const canPrev = page > 0;
  const canNext = page < totalPages - 1;
  const start = total === 0 ? 0 : page * pageSize + 1;
  const end = Math.min((page + 1) * pageSize, total);
  // The page-size selector and "X of Y" summary always render so users can
  // change the page size on small/filtered datasets. Only the prev/next
  // navigation is conditional on there being more than one page.
  const hasMultiplePages = totalPages > 1;

  return (
    <div className="flex items-center justify-between px-2 py-2 text-xs text-muted-foreground border-t">
      {/* Left: rows per page selector */}
      <div className="flex items-center gap-2">
        <span>Rows per page</span>
        <Select value={String(pageSize)} onValueChange={(v) => onPageSizeChange(Number(v))}>
          <SelectTrigger className="h-7 w-[70px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PAGE_SIZE_OPTIONS.map((size) => (
              <SelectItem key={size} value={String(size)} className="text-xs">{size}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Center: showing X-Y of Z */}
      <span className="tabular-nums">
        {start}–{end} of {total}
      </span>

      {/* Right: navigation buttons — only when there's more than one page */}
      {hasMultiplePages && (
        <div className="flex items-center gap-0.5">
          <Button variant="ghost" size="icon" className="size-7" onClick={() => onPageChange(0)} disabled={!canPrev} title="First page">
            <ChevronsLeft className="size-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => onPageChange(page - 1)} disabled={!canPrev} title="Previous page">
            <ChevronLeft className="size-3.5" />
          </Button>
          <span className="px-2 tabular-nums">
            {page + 1} / {totalPages}
          </span>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => onPageChange(page + 1)} disabled={!canNext} title="Next page">
            <ChevronRight className="size-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => onPageChange(totalPages - 1)} disabled={!canNext} title="Last page">
            <ChevronsRight className="size-3.5" />
          </Button>
        </div>
      )}
    </div>
  );
}
