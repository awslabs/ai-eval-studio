// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Synthetic-data review section of the Data tab.
 *
 * Two-pane layout: the left pane lists every synthetic batch for the
 * project with per-status counts; the right pane renders the selected
 * batch's cases via ``SyntheticCaseDetail``. The selected batch id is
 * mirrored to ``?batch=...`` in the URL so deep links from chat tool
 * cards land on the right batch.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Sparkles, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/class-utils";
import {
  useSyntheticBatch,
  useSyntheticBatches,
} from "@/hooks/useSyntheticBatches";
import type { SyntheticBatchSummary, SyntheticCase } from "@/lib/types";
import { SyntheticCaseDetail } from "./SyntheticCaseDetail";

/** Props for the SyntheticDataPanel component. */
interface SyntheticDataPanelProps {
  /** Project UUID to scope the synthesis listing to. */
  projectId: string;
}

/**
 * Strategy-specific badge tint.
 *
 * Tailwind can't derive classes from runtime strings, so we keep an
 * explicit map. Colors are purely informational — users can approve/reject
 * regardless of strategy.
 */
const STRATEGY_STYLES: Record<string, string> = {
  paraphrase: "bg-info/15 text-info border-info/30",
  from_docs: "bg-success/15 text-success border-success/30",
  from_scratch: "bg-primary/15 text-primary border-primary/30",
  failure_targeted: "bg-warning/15 text-warning border-warning/30",
};

/** Fallback tint when a batch reports a strategy we don't have a style for. */
const DEFAULT_STRATEGY_STYLE = "bg-muted text-muted-foreground border-border";

/**
 * Convert a filesystem-safe timestamp (``2026-04-22T14-05-31``) into a
 * locale string. Returns the raw timestamp if parsing fails.
 */
function formatTimestamp(raw: string): string {
  if (!raw) return "";
  // Turn 2026-04-22T14-05-31 back into 2026-04-22T14:05:31 for Date to parse.
  const iso = raw.replace(
    /T(\d{2})-(\d{2})-(\d{2})/,
    "T$1:$2:$3",
  );
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return raw;
  return date.toLocaleString();
}

/**
 * Top-level panel for reviewing synthetic batches.
 *
 * @param projectId - Project UUID for fetching batch data
 */
export function SyntheticDataPanel({ projectId }: SyntheticDataPanelProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    data: batches,
    loading: listLoading,
    refresh: refreshList,
  } = useSyntheticBatches(projectId);

  // URL-driven selection; initialise from the current ``?batch=`` param.
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(
    () => searchParams.get("batch"),
  );

  /**
   * Keep local selection in sync when the URL changes externally
   * (e.g. the chat tool-UI deep links into a specific batch).
   */
  useEffect(() => {
    const urlBatch = searchParams.get("batch");
    if (urlBatch !== selectedBatchId) {
      setSelectedBatchId(urlBatch);
    }
    // Only react to URL changes here; local setter below syncs the other way.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const {
    data: detail,
    loading: detailLoading,
    refresh: refreshDetail,
  } = useSyntheticBatch(projectId, selectedBatchId);

  /**
   * Local optimistic overlay on top of the fetched cases. Keyed by
   * batch id so switching batches always resets cleanly.
   */
  const [statusOverrides, setStatusOverrides] = useState<
    Record<string, Record<number, string>>
  >({});

  // Clear overrides whenever the authoritative detail payload changes —
  // the server response is truth after a refresh.
  useEffect(() => {
    if (detail) {
      setStatusOverrides((prev) => ({ ...prev, [detail.batch_id]: {} }));
    }
  }, [detail?.batch_id, detail?.rows?.length, detail]);

  /** Update the URL selection and the local state in one go. */
  const selectBatch = useCallback(
    (batchId: string | null) => {
      setSelectedBatchId(batchId);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (batchId) {
            next.set("batch", batchId);
          } else {
            next.delete("batch");
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  /** Optimistic status flip from the case card. */
  const handleStatusChange = useCallback(
    (index: number, status: string) => {
      if (!detail) return;
      setStatusOverrides((prev) => ({
        ...prev,
        [detail.batch_id]: {
          ...(prev[detail.batch_id] ?? {}),
          [index]: status,
        },
      }));
      // Refresh the summaries so counts on the left pane catch up; cheap.
      refreshList();
    },
    [detail, refreshList],
  );

  /** Approve every currently-pending case in the selected batch. */
  const [approveAllPending, setApproveAllPending] = useState(false);
  const [approveAllError, setApproveAllError] = useState<string | null>(null);

  const approveAll = useCallback(async () => {
    if (!detail) return;
    setApproveAllError(null);
    setApproveAllPending(true);
    try {
      await api.approveSyntheticCases(projectId, detail.batch_id, {
        approve_all: true,
      });
      await refreshDetail();
      await refreshList();
    } catch (e) {
      setApproveAllError(e instanceof Error ? e.message : "Update failed");
    } finally {
      setApproveAllPending(false);
    }
  }, [detail, projectId, refreshDetail, refreshList]);

  /** Resolve the cases to render, applying optimistic overrides. */
  const overlaidRows: SyntheticCase[] = useMemo(() => {
    if (!detail) return [];
    const overrides = statusOverrides[detail.batch_id] ?? {};
    if (Object.keys(overrides).length === 0) return detail.rows;
    return detail.rows.map((row) =>
      row.index in overrides
        ? { ...row, status: overrides[row.index] }
        : row,
    );
  }, [detail, statusOverrides]);

  /** Live counts reflecting optimistic overrides for the action bar. */
  const overlaidCounts = useMemo(() => {
    const base = { pending: 0, approved: 0, rejected: 0 };
    for (const row of overlaidRows) {
      if (row.status === "approved") base.approved += 1;
      else if (row.status === "rejected") base.rejected += 1;
      else base.pending += 1;
    }
    return base;
  }, [overlaidRows]);

  return (
    <div className="flex gap-4 flex-1 min-h-0" style={{ height: "clamp(400px, 72dvh, 1000px)" }}>
      {/* ── Left pane: batch list ── */}
      <div className="w-[320px] shrink-0 flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-2 shrink-0">
          <h2 className="text-sm font-medium">Batches</h2>
          {listLoading && (
            <Loader2 className="size-3 animate-spin text-muted-foreground" />
          )}
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto space-y-1.5 pr-1">
          {(batches ?? []).length === 0 && !listLoading && (
            <p className="text-xs text-muted-foreground px-1">
              No synthetic batches yet.
            </p>
          )}
          {(batches ?? []).map((batch) => (
            <BatchRow
              key={batch.batch_id}
              batch={batch}
              isSelected={batch.batch_id === selectedBatchId}
              onSelect={() => selectBatch(batch.batch_id)}
            />
          ))}
        </div>
      </div>

      {/* ── Right pane: cases ── */}
      <div className="flex-1 min-w-0 flex flex-col min-h-0">
        {!selectedBatchId ? (
          <EmptyRightPane hasBatches={(batches ?? []).length > 0} />
        ) : detailLoading && !detail ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        ) : detail ? (
          <>
            {/* Action bar */}
            <div className="flex items-center justify-between gap-3 pb-3 shrink-0 border-b border-border">
              <div className="flex items-center gap-3 text-xs">
                <span className="font-mono text-muted-foreground">
                  {detail.batch_id}
                </span>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-2xs border",
                    STRATEGY_STYLES[detail.strategy] ?? DEFAULT_STRATEGY_STYLE,
                  )}
                >
                  {detail.strategy}
                </Badge>
                <span className="text-muted-foreground tabular-nums">
                  {overlaidCounts.pending} pending ·{" "}
                  <span className="text-success">
                    {overlaidCounts.approved} approved
                  </span>{" "}
                  ·{" "}
                  <span className="text-destructive">
                    {overlaidCounts.rejected} rejected
                  </span>
                </span>
              </div>
              <Button
                size="sm"
                variant="default"
                disabled={approveAllPending || overlaidCounts.pending === 0}
                onClick={approveAll}
              >
                {approveAllPending && (
                  <Loader2 className="size-3 animate-spin" />
                )}
                Approve all pending
              </Button>
            </div>

            {approveAllError && (
              <p className="text-xs text-destructive pt-2 shrink-0">
                {approveAllError}
              </p>
            )}

            {/* Cases list */}
            <div className="flex-1 min-h-0 overflow-y-auto space-y-2 pt-3 pr-1">
              {overlaidRows.map((caseItem) => (
                <SyntheticCaseDetail
                  key={caseItem.index}
                  projectId={projectId}
                  batchId={detail.batch_id}
                  caseItem={caseItem}
                  fieldMappings={detail.field_mappings}
                  onStatusChange={handleStatusChange}
                />
              ))}
              {overlaidRows.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  This batch has no cases.
                </p>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground py-4">
            Unable to load batch details.
          </p>
        )}
      </div>
    </div>
  );
}

/** Empty-state rendered in the right pane when nothing is selected. */
function EmptyRightPane({ hasBatches }: { hasBatches: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 text-center px-6 py-10">
      <div className="size-12 rounded-full bg-muted flex items-center justify-center mb-3">
        <Sparkles className="size-5 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground max-w-sm">
        {hasBatches
          ? "Select a batch to review."
          : "Ask the agent to generate synthetic data. Approved cases can be merged into your next eval run."}
      </p>
    </div>
  );
}

/** Single-row entry in the left-pane batch list. */
function BatchRow({
  batch,
  isSelected,
  onSelect,
}: {
  batch: SyntheticBatchSummary;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={isSelected || undefined}
      className={cn(
        "w-full text-left rounded-md border border-border px-2.5 py-2 transition-colors outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring",
        isSelected
          ? "bg-accent border-accent-foreground/20"
          : "hover:bg-accent/40",
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <Badge
          variant="outline"
          className={cn(
            "text-2xs border",
            STRATEGY_STYLES[batch.strategy] ?? DEFAULT_STRATEGY_STYLE,
          )}
        >
          {batch.strategy}
        </Badge>
        <span className="text-2xs text-muted-foreground tabular-nums">
          {batch.total_cases}
        </span>
      </div>
      <p className="text-2xs text-foreground mb-1">
        {formatTimestamp(batch.timestamp)}
      </p>
      <div className="flex items-center gap-2 text-2xs tabular-nums">
        <span className="text-muted-foreground">{batch.pending} pending</span>
        <span className="text-success">{batch.approved} approved</span>
        <span className="text-destructive">{batch.rejected} rejected</span>
      </div>
      {batch.seed_source && (
        <p className="text-2xs text-muted-foreground/70 font-mono mt-1 truncate">
          {batch.seed_source}
        </p>
      )}
    </button>
  );
}
