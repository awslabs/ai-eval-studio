// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Reports tab — fetches the report JSON, renders the React component, and
 * exports it via the browser (Print / Save as PDF) or as a self-contained
 * HTML file. Both exports reuse the on-screen report DOM (single source of
 * truth) rather than a separate document tree — see lib/reports/exportHtml.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Loader2, FileBarChart, RefreshCw, AlertTriangle, Printer, FileCode2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { shortModel } from "@/lib/format-utils";
import { useEvalRuns } from "@/hooks/useEvalRuns";
import { useCursorGapHandler, useProjectEvent } from "@/hooks/useBackendEvents";
import { ExecReport } from "@/components/reports/ExecReport";
import { RunReport } from "@/components/reports/RunReport";
import { slugify } from "@/lib/reports/format";
import { downloadReportHtml, printReport, type ReportExportInput } from "@/lib/reports/exportHtml";
import type { ExecReportEnvelope, RunReportEnvelope } from "@/lib/reports/types";
import { newestRun } from "@/lib/run-list";

interface ReportsTabProps {
  projectId: string;
}

/** A regenerate the backend has accepted but not yet reported an outcome for. */
interface PendingRegenerate {
  scope: "run" | "executive";
  /** Run filename for a per-run regenerate; ``null`` for the executive one. */
  run: string | null;
}

/** Shown when the narrator ends without writing a new summary. */
const REGENERATE_FAILED_MESSAGE =
  "The report could not be regenerated. The version shown is unchanged.";

/** Format a run summary into the selector's display label. */
function runLabel(run: {
  prompt_version: string;
  timestamp: string;
  source: "optimization" | "manual";
  model_runs: { model: string; total: number }[];
}): string {
  const modelNames = run.model_runs.map((m) => shortModel(m.model)).join(", ");
  const caseCount = run.model_runs[0]?.total ?? 0;
  const sourceTag = run.source === "optimization" ? " · optimization loop" : " · manual run";
  const date = new Date(run.timestamp);
  const dateStr = `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  return `${run.prompt_version} — ${modelNames} (${caseCount} cases, ${dateStr}${sourceTag})`;
}

export function ReportsTab({ projectId }: ReportsTabProps) {
  const { runs, loading, error } = useEvalRuns(projectId);
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedFile = searchParams.get("rrun");
  const reportType = searchParams.get("rtype") === "run" ? "run" : "executive";

  const [runReport, setRunReport] = useState<RunReportEnvelope | null>(null);
  const [execReport, setExecReport] = useState<ExecReportEnvelope | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  // The regenerate endpoints return 202 before the narrator finishes, so a
  // regenerate stays pending until a ``report`` event reports its outcome.
  // Tracked with its scope and run so the spinner / "Updating…" badge only
  // appears on the report being regenerated, not on whichever report the
  // user switches to mid-regen. ``null`` = nothing regenerating.
  const [regenerating, setRegenerating] = useState<PendingRegenerate | null>(null);
  const isRegenerating = regenerating !== null
    && regenerating.scope === reportType
    && (reportType === "executive" || regenerating.run === selectedFile);
  // Bumped by ``report/updated`` events from the WS bus. Including it in
  // the fetch effect deps gets the active tab to refetch the moment a
  // narration finishes — no polling, and a coalesced/background regen
  // surfaces on its own.
  const [refreshNonce, setRefreshNonce] = useState(0);

  // Wraps ONLY the rendered report components (not the spinner/badge
  // overlays) so its innerHTML is a clean snapshot for the HTML/print export.
  const reportRef = useRef<HTMLDivElement>(null);

  useProjectEvent({ resource: "report" }, (event) => {
    const isOutcome = event.type === "updated" || event.type === "failed";
    if (event.type === "updated") {
      const showsThisReport = event.scope === "executive"
        ? reportType === "executive"
        : event.scope === "run"
          && reportType === "run"
          && !!event.run_filename
          && event.run_filename === selectedFile;
      if (showsThisReport) setRefreshNonce((n) => n + 1);
    }
    if (!regenerating) return;
    // ``error`` is the background task crashing outright; its toast says why.
    const endsPending = event.type === "error" || (
      isOutcome
      && event.scope === regenerating.scope
      && (regenerating.scope === "executive" || event.run_filename === regenerating.run)
    );
    if (!endsPending) return;
    setRegenerating(null);
    if (event.type === "failed") setReportError(REGENERATE_FAILED_MESSAGE);
  }, "state");

  // A hole in the event stream (a coordinator restart, for one) can swallow
  // the outcome, and a restart also ends the narration, so stop waiting and
  // show what is on disk. A narration that is still running refreshes the
  // report through its ``report/updated`` event when it lands.
  useCursorGapHandler(projectId, () => {
    if (!regenerating) return;
    setRegenerating(null);
    setRefreshNonce((n) => n + 1);
  });

  const setReportType = useCallback((type: "run" | "executive") => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("rtype", type);
      return next;
    }, { replace: true });
  }, [setSearchParams]);

  // Default the selection to the newest run once runs load.
  useEffect(() => {
    if (!selectedFile && runs && runs.length > 0) {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("rrun", newestRun(runs)!._file);
        return next;
      }, { replace: true });
    }
  }, [runs, selectedFile, setSearchParams]);

  const setSelectedFile = useCallback((file: string | null) => {
    if (!file) return;
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("rrun", file);
      return next;
    }, { replace: true });
  }, [setSearchParams]);

  // Fetch the report JSON whenever the report type or selected run changes,
  // or a ``report/updated`` event says a new narrative is on disk.
  useEffect(() => {
    let cancelled = false;
    if (reportType === "run" && !selectedFile) {
      setRunReport(null);
      return;
    }
    // Drop the previous report before loading the next one. Keeping it meant the
    // selector said run B while the panel still rendered run A — and because the
    // spinner and the export button are both gated on "a report exists", Print
    // and Download HTML would export A's content under B's name. A failed fetch
    // showed an error banner over the old report for the same reason.
    if (reportType === "executive") {
      setExecReport(null);
    } else {
      setRunReport(null);
    }
    setReportLoading(true);
    setReportError(null);
    const fetcher = reportType === "executive"
      ? api.getExecutiveReport(projectId).then((data) => {
          if (!cancelled) setExecReport(data);
        })
      : api.getRunReport(projectId, selectedFile!).then((data) => {
          if (!cancelled) setRunReport(data);
        });
    fetcher
      .catch((e: Error) => { if (!cancelled) setReportError(e.message); })
      .finally(() => { if (!cancelled) setReportLoading(false); });
    return () => { cancelled = true; };
  }, [projectId, reportType, selectedFile, refreshNonce]);

  const handleRegenerate = useCallback(async () => {
    if (reportType === "run" && !selectedFile) return;
    setReportError(null);
    // Set before the request so an outcome event that beats the 202 still
    // finds the regenerate pending.
    setRegenerating({ scope: reportType, run: reportType === "run" ? selectedFile : null });
    try {
      if (reportType === "executive") {
        await api.regenerateExecutiveReport(projectId);
      } else if (selectedFile) {
        await api.regenerateReport(projectId, selectedFile);
      }
    } catch (e) {
      // The request was refused, so no outcome event will follow.
      setReportError((e as Error).message);
      setRegenerating(null);
    }
  }, [projectId, reportType, selectedFile]);

  // Build the export payload from whatever report is currently rendered
  // on screen (WYSIWYG). Returns null if nothing is rendered yet.
  const buildExportInput = useCallback((): { input: ReportExportInput; baseName: string } | null => {
    const node = reportRef.current;
    if (!node || !node.innerHTML.trim()) return null;
    const date = new Date().toISOString().slice(0, 10);
    if (reportType === "executive" && execReport) {
      const slug = slugify(execReport.facts.project_name);
      return {
        input: { node, title: `${execReport.facts.project_name} — Executive report` },
        baseName: `${slug}-exec-${date}`,
      };
    }
    if (reportType === "run" && runReport) {
      const slug = slugify(runReport.facts.project_name);
      const ver = runReport.facts.prompt_label || "run";
      return {
        input: { node, title: `${runReport.facts.project_name} — ${ver} run report` },
        baseName: `${slug}-run-${ver}-${date}`,
      };
    }
    return null;
  }, [reportType, execReport, runReport]);

  const handleExportHtml = useCallback(() => {
    const r = buildExportInput();
    if (!r) return;
    downloadReportHtml(r.input, `${r.baseName}.html`);
  }, [buildExportInput]);

  const handlePrint = useCallback(() => {
    const r = buildExportInput();
    if (!r) return;
    printReport(r.input);
  }, [buildExportInput]);

  // The HTML/print exports snapshot the on-screen render, so they only need a
  // report present.
  const exportDisabled = useMemo(() => {
    if (regenerating !== null) return true;
    return reportType === "executive" ? !execReport : !runReport;
  }, [regenerating, reportType, execReport, runReport]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground text-sm py-4">
        <Loader2 className="size-4 animate-spin" /> Loading runs…
      </div>
    );
  }

  if (error) {
    return <p className="text-destructive text-sm py-4">{error.message}</p>;
  }

  if (!runs || runs.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 text-muted-foreground text-sm py-12">
        <FileBarChart className="size-8 opacity-40" />
        <p>Run an evaluation to generate a report.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 flex-1 min-h-0">
      <SegmentedControl
        aria-label="Report type"
        value={reportType}
        onValueChange={setReportType}
        options={[
          { value: "executive", label: "Executive Report" },
          { value: "run", label: "Per-run Report" },
        ]}
      />

      <div className="flex items-end gap-3">
        {reportType === "run" && (
          <div className="space-y-1 flex-1">
            <label className="text-xs font-medium text-muted-foreground">Eval Run</label>
            <Select value={selectedFile ?? ""} onValueChange={setSelectedFile}>
              <SelectTrigger className="w-full h-9 text-xs">
                <SelectValue placeholder="Select a run">
                  {(() => {
                    const r = runs.find((r) => r._file === selectedFile);
                    return r ? runLabel(r) : "Select a run";
                  })()}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {runs.map((r) => (
                  <SelectItem key={r._file} value={r._file}>
                    {runLabel(r)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {reportType === "executive" && (
          <p className="flex-1 text-xs text-muted-foreground self-center">
            Summarizes the trophy-winning run + the journey across versions.
          </p>
        )}
        <Button
          variant="outline"
          onClick={handleRegenerate}
          // One regenerate at a time: a second would replace the pending one,
          // and the first one's outcome would then end nothing.
          disabled={regenerating !== null || (reportType === "run" && !selectedFile)}
          className="h-9"
          title={
            reportType === "executive"
              ? "Regenerate the project executive report"
              : "Regenerate the report for the selected run only"
          }
        >
          {isRegenerating ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <RefreshCw className="size-4" />
          )}
          {reportType === "executive" ? "Regenerate executive report" : "Regenerate this run"}
        </Button>
        <Button
          onClick={handlePrint}
          disabled={exportDisabled}
          className="h-9"
          title="Print or Save as PDF via the browser (reliable pagination, white report style)"
        >
          <Printer className="size-4" />
          Print
        </Button>
        <Button
          variant="outline"
          onClick={handleExportHtml}
          disabled={exportDisabled}
          className="h-9"
          title="Download a self-contained HTML file of this report"
        >
          <FileCode2 className="size-4" />
          Download HTML
        </Button>
      </div>

      {reportError && <p className="text-destructive text-sm">{reportError}</p>}

      {reportType === "executive" && execReport?.stale && (
        // Passive notice — the narrator runs in the background and is
        // best-effort, so we never block the page on it. Stat cards
        // above are still fresh; only the prose is a run behind.
        <div className="flex items-center gap-2 px-3 py-2 rounded-md bg-warning/10 border border-warning/30 text-xs text-warning">
          <AlertTriangle className="size-3.5 shrink-0" />
          <span>
            Summary reflects an earlier run. Regenerate to refresh the
            prose; numbers above are already current.
          </span>
        </div>
      )}

      <div className="flex-1 min-h-[480px] border rounded-lg bg-card relative overflow-y-auto p-6">
        {/* Full-panel spinner only on initial load (no data yet). When the
            user already sees a report and they click Regenerate, the
            existing render stays visible — the button spinner is enough. */}
        {reportLoading && !execReport && !runReport && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        )}
        {isRegenerating && (execReport || runReport) && (
          <div className="absolute top-3 right-3 z-10 px-2 py-1 rounded-md bg-secondary text-secondary-foreground text-xs flex items-center gap-1.5">
            <Loader2 className="size-3 animate-spin" />
            Updating…
          </div>
        )}
        <div ref={reportRef}>
          {reportType === "executive" && execReport && (
            <ExecReport facts={execReport.facts} narrative={execReport.narrative} />
          )}
          {reportType === "run" && runReport && (
            <RunReport facts={runReport.facts} narrative={runReport.narrative} />
          )}
        </div>
      </div>
    </div>
  );
}
