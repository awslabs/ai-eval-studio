// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified active job status bar with optimization lifecycle controls.
 *
 * Replaces both OptimizationProgress and EvalJobProgress with a single
 * persistent bar. Shows Start/Stop/Resume/Start New buttons, optimization
 * phase status, and click-to-expand per-model eval progress.
 *
 * Visibility rules:
 *  - Hidden during the first harness probe (avoids a flash of "Start"
 *    on slow networks).
 *  - Shown whenever the optimization state is non-null, even when the
 *    harness probe raced the scaffold and found no harness yet — the
 *    running/paused branches don't depend on harness config and we don't
 *    want a multi-minute invisible window during iteration 1.
 *  - Shown whenever a standalone eval is running, even when the harness
 *    probe found nothing — the eval branch does not depend on harness
 *    config and hiding it reports a live run as silence.
 *  - Hidden when there's no harness AND no active optimization AND no
 *    standalone eval (the Start branch needs harness config for the
 *    data-source picker).
 */

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { pluralize } from "@/lib/format-utils";
import { isResumableStatus } from "@/lib/optimization-status";
import type { HarnessConfig, DataSource } from "@/lib/types";
import {
  useEvalJobStream,
  type ModelProgress,
} from "@/hooks/useEvalJobStream";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import { useSyntheticBatches } from "@/hooks/useSyntheticBatches";
import { useEvalRuns } from "@/hooks/useEvalRuns";
import { survivingTerminalIteration } from "@/lib/optimization-evidence";
import { useOptimizationState } from "@/components/eval/OptimizationStateProvider";

/** Props for the ActiveJobBar component. */
interface ActiveJobBarProps {
  /** Project UUID */
  projectId: string;
}

/**
 * Per-model progress row for the expandable detail section.
 * Reuses the ModelProgress type from the useEvalJobStream hook.
 */
function ModelProgressRow({ entry, isActive }: { entry: ModelProgress; isActive: boolean }) {
  const genPercent = entry.casesTotal > 0
    ? Math.round((entry.generated / entry.casesTotal) * 100)
    : 0;

  const isComplete = entry.phase === "complete";
  const isScoring = entry.phase === "scoring" && !entry.scored;

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 text-xs">
        {isComplete ? (
          <CheckCircle2 className="size-3 text-success shrink-0" />
        ) : isActive ? (
          <span className="size-2 rounded-full bg-info animate-pulse shrink-0" />
        ) : (
          <span className="size-2 rounded-full bg-muted-foreground/30 shrink-0" />
        )}
        <span className={`font-medium truncate ${isActive ? "text-foreground" : "text-muted-foreground"}`}>
          {entry.model}
        </span>
        {isComplete && (
          <span className="text-success ml-auto shrink-0 text-2xs">
            {entry.passed}/{entry.casesTotal} passed
          </span>
        )}
      </div>
      {/* Generation bar — show absolute count ahead of the percent so
       *  case drops are visible live (e.g. "190/210" makes a 20-case
       *  drop obvious where 90% wouldn't). */}
      <div className="flex items-center gap-2">
        <span className="text-2xs text-muted-foreground w-12 shrink-0">Generate</span>
        <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full bg-info rounded-full transition-[width] duration-500 ease-out"
            style={{ width: `${genPercent}%` }}
          />
        </div>
        <span className="text-2xs text-muted-foreground w-20 text-right shrink-0 tabular-nums">
          {entry.casesTotal > 0
            ? `${entry.generated}/${entry.casesTotal} · ${genPercent}%`
            : `${genPercent}%`}
        </span>
      </div>
      {/* Scoring bar — driven by per-case ``scoring_progress`` events
       *  emitted from the harness's DeepEval poll. Falls back to the
       *  animated indeterminate stripe when the harness hasn't sent
       *  any ``scoring_progress`` yet (e.g., a project scaffolded
       *  before that event existed). */}
      {(isScoring || entry.scored) && (
        <div className="flex items-center gap-2">
          <span className="text-2xs text-muted-foreground w-12 shrink-0">Score</span>
          <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
            {entry.scored ? (
              <div className="h-full bg-success rounded-full" style={{ width: "100%" }} />
            ) : entry.scoredCount > 0 && entry.casesTotal > 0 ? (
              <div
                className="h-full bg-info rounded-full transition-[width] duration-500 ease-out"
                style={{ width: `${Math.round((entry.scoredCount / entry.casesTotal) * 100)}%` }}
              />
            ) : (
              <div
                className="h-full w-full rounded-full"
                style={{
                  backgroundImage:
                    "repeating-linear-gradient(90deg, color-mix(in oklch, var(--success) 20%, transparent) 0%, var(--warning) 25%, var(--success) 50%, color-mix(in oklch, var(--warning) 20%, transparent) 75%, color-mix(in oklch, var(--success) 20%, transparent) 100%)",
                  backgroundSize: "200% 100%",
                  animation: "scoring-flow 2.5s linear infinite",
                }}
              />
            )}
          </div>
          <span className="text-2xs text-muted-foreground w-20 text-right shrink-0 tabular-nums">
            {entry.scored
              ? `${entry.casesTotal}/${entry.casesTotal} · 100%`
              : entry.scoredCount > 0 && entry.casesTotal > 0
                ? `${entry.scoredCount}/${entry.casesTotal} · ${Math.round((entry.scoredCount / entry.casesTotal) * 100)}%`
                : entry.casesTotal > 0
                  ? `0/${entry.casesTotal} · …`
                  : "…"}
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Unified active job status bar.
 *
 * Permanent fixture below the project description. Transitions between
 * states rather than appearing/disappearing. Hidden during the first
 * harness probe, and when there is no harness config (Phase 3 not
 * complete) and no optimization state or standalone eval to report — see
 * the visibility rules in the file header.
 */
export function ActiveJobBar({ projectId }: ActiveJobBarProps) {
  /* Shared optimization state — OptimizationStateProvider owns the poll
   * and the optimistic-transition override. Reading from the provider
   * (instead of a component-local useState+poll) is what keeps the
   * status bar and the eval-dashboard iteration cards in agreement. */
  const { state: optState, activeEval, refetch: refetchOptState, setOptimistic } =
    useOptimizationState();

  /* A convergence claim is a statement about a specific run's results. When
   * that run's file is deleted the claim is no longer supportable, so the bar
   * reports the surviving truth — how many iterations ran — and drops the
   * verdict. Unlike the results panel, this bar cannot omit its line entirely:
   * an empty status string falls through to idle copy ("Ready to optimize"),
   * which would misreport a loop that did finish. */
  const { runs: evalRunsForEvidence } = useEvalRuns(projectId);
  const terminalEvidence = survivingTerminalIteration(
    optState ?? null,
    evalRunsForEvidence ?? [],
  );

  const [harnessConfig, setHarnessConfig] = useState<HarnessConfig | null>(null);
  const [harnessChecked, setHarnessChecked] = useState(false);
  /* Error from the most recent harness probe. "No harness yet" arrives as a
   * successful null response, so anything captured here is a real problem
   * (network, auth, server) and is surfaced as a chip. Without it, a
   * transient 5xx would silently hide the bar for the rest of the session. */
  const [harnessError, setHarnessError] = useState<Error | null>(null);
  const [expanded, setExpanded] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [showExtendConfirm, setShowExtendConfirm] = useState(false);

  /**
   * Data-source picker shown before (re)starting optimization when the
   * project has approved synthetic cases available. ``intent`` tracks
   * which call we should make on confirm — resume preserves iteration
   * history, start begins a fresh run.
   */
  const [dataSourceDialog, setDataSourceDialog] = useState<
    { intent: "start" | "resume"; selected: DataSource } | null
  >(null);

  // Fetch synthetic batches so we can decide whether to show the dialog and
  // display useful count hints. The hook is cheap (manifest reads only).
  const { data: syntheticBatches } = useSyntheticBatches(projectId);

  // Total approved synthetic cases across every batch. Drives the dialog
  // gate — zero approvals means we short-circuit to raw (preserves UX).
  const approvedSyntheticTotal = (syntheticBatches ?? []).reduce(
    (sum, b) => sum + (b.approved || 0),
    0,
  );

  // Eval job stream — per-case WS progress for the active eval the
  // provider detected via /activity. Detection is centralized there; this
  // hook only animates the per-model detail rows.
  const { progress: evalProgress, handleCancel: cancelEvalJob, cancelling: evalCancelling } =
    useEvalJobStream(projectId, activeEval);

  // Check harness config availability. The bar hides itself only when
  // there's no harness AND no active optimization to show progress for;
  // see the visibility comment in the file header. Bus events and the
  // provider's state changes below trigger the re-checks.
  const checkHarness = useCallback(() => {
    api.getHarnessConfig(projectId)
      .then((config) => {
        setHarnessConfig(config);
        setHarnessError(null);
        setHarnessChecked(true);
      })
      .catch((err: unknown) => {
        setHarnessConfig(null);
        // "No harness yet" is a 200-null now (a normal state, not an
        // error), so everything landing here — network blip, 5xx, auth —
        // is captured for the error chip instead of silently hiding.
        console.error("[ActiveJobBar] getHarnessConfig failed", err);
        setHarnessError(err instanceof Error ? err : new Error(String(err)));
        setHarnessChecked(true);
      });
  }, [projectId]);

  useEffect(() => {
    setHarnessChecked(false);
    checkHarness();
  }, [projectId, checkHarness]);

  /* The harness usually becomes ready in chat: the agent scaffolds it and
   * writes the baseline prompt, and the backend sends no event of its own
   * when `ready` flips. Neither starts an optimization or an eval, so the
   * effect below never fires and the Start button would wait for a page
   * refresh. Re-check on the same signals useHarnessConfig uses: the end of
   * an agent turn, a file change and a finished job. */
  useProjectEvent({ resource: "agent_turn", type: "completed" }, checkHarness, "state");
  useProjectEvent({ resource: "filesystem", type: "changed" }, checkHarness, "state");
  useProjectEvent({ resource: "job", type: "completed" }, checkHarness, "state");

  /* Re-check harness whenever the optimization state OR a standalone eval
   * shows up — the initial mount often races the harness scaffold (the agent
   * writes config.py + run_eval.py while the SPA's first /api/harness probe
   * returns null). Re-fetch any time we don't have config but the
   * provider is reporting *some* state (current_iteration may be 0 during the
   * first iteration's eval phase, so check optState !== null instead of the
   * falsy-zero-trap ``optState?.current_iteration``). The same reasoning
   * applies to standalone evals: an appearing eval proves a harness exists on
   * disk right now, so a probe that answered "none" before the scaffold is
   * stale. Without this, `ready` and `data_file` stay wrong until a manual
   * refresh and the bar offers no Start button after the eval ends. The
   * object deps are deliberate: the `/activity` poll hands back a fresh
   * object every tick, so this retries while config is missing and stops
   * the instant it lands. */
  useEffect(() => {
    if (!harnessConfig && (optState !== null || activeEval !== null)) {
      checkHarness();
    }
  }, [harnessConfig, optState, activeEval, checkHarness]);

  // Optimistic stop: immediately show "stopping" locally via the provider.
  const handleStop = useCallback(() => {
    if (actionLoading) return;
    setActionLoading(true);
    setOptimistic({ status: "stopping" });
    api.cancelOptimization(projectId)
      .catch((err: unknown) => {
        // Surface the refusal AND revert — next poll will correct.
        toast.error(err instanceof Error ? err.message : "Failed to stop optimization");
        refetchOptState();
      })
      .finally(() => setActionLoading(false));
  }, [actionLoading, projectId, refetchOptState, setOptimistic]);

  /**
   * Kick off a start call with a resolved data source.
   * Split out so the dialog confirm button can reuse it without going
   * through the gating logic that decides whether to show the dialog.
   */
  const doStart = useCallback((dataSource: DataSource) => {
    setActionLoading(true);
    setOptimistic({ status: "starting" });
    const body = dataSource !== "raw" ? { data_source: dataSource } : undefined;
    api.startOptimization(projectId, body)
      .then(() => refetchOptState())
      .catch((err: unknown) => {
        /* A refused start MUST be visible. The start endpoint answers a
         * refusal with a 400 whose detail names what to fix in chat (for
         * example "missing required optimization fields: …"); swallowing it
         * leaves the user clicking a button that does nothing. */
        toast.error(err instanceof Error ? err.message : "Failed to start optimization");
        refetchOptState();
      })
      .finally(() => setActionLoading(false));
  }, [projectId, refetchOptState, setOptimistic]);

  const handleStart = useCallback(() => {
    if (actionLoading) return;
    // Short-circuit the dialog when no approved synthetic cases exist —
    // preserves the pre-synth UX so users without synthetic data are unaffected.
    if (approvedSyntheticTotal <= 0) {
      doStart("raw");
      return;
    }
    setDataSourceDialog({ intent: "start", selected: "raw" });
  }, [actionLoading, approvedSyntheticTotal, doStart]);

  /** Execute the actual resume API call with the given data source. */
  const doResume = useCallback((dataSource: DataSource = "raw") => {
    setShowExtendConfirm(false);
    setActionLoading(true);
    setOptimistic({ status: "starting" });
    const body = dataSource !== "raw" ? { data_source: dataSource } : undefined;
    api.resumeOptimization(projectId, body)
      .then(() => refetchOptState())
      .catch((err: unknown) => {
        toast.error(err instanceof Error ? err.message : "Failed to resume optimization");
        refetchOptState();
      })
      .finally(() => setActionLoading(false));
  }, [projectId, refetchOptState, setOptimistic]);

  const handleResume = useCallback(() => {
    if (actionLoading) return;

    // Show confirm dialog when extending past max iterations
    if (optState?.status === "max_iterations") {
      setShowExtendConfirm(true);
      return;
    }

    // Show data source picker if approved synthetic cases exist.
    if (approvedSyntheticTotal > 0) {
      setDataSourceDialog({ intent: "resume", selected: "raw" });
      return;
    }

    doResume("raw");
  }, [actionLoading, optState?.status, approvedSyntheticTotal, doResume]);

  /** Confirm handler for the data-source picker dialog. */
  const handleDataSourceConfirm = useCallback(() => {
    if (!dataSourceDialog) return;
    const { intent, selected } = dataSourceDialog;
    setDataSourceDialog(null);
    if (intent === "start") {
      doStart(selected);
    } else {
      doResume(selected);
    }
  }, [dataSourceDialog, doStart, doResume]);

  // Determine the display state — derived first so the visibility
  // check can consult ``hasActiveOpt``.
  const optStatus = optState?.status ?? null;
  const isStarting = optStatus === "starting";
  const isOptRunning = optStatus === "running";
  const isStopping = optStatus === "stopping";
  /* Compile-checked classification: the old `===` chain here silently
   * treated any status it didn't enumerate as non-resumable — the
   * lib/optimization-status Record forces every status in the union to be
   * classified explicitly. */
  const isResumable = isResumableStatus(optStatus);
  const isConverged = optStatus === "converged";
  /* A crashed loop is NOT resumable — the backend's resume endpoint rejects
   * status "failed" (api/optimization.py's allowlist), so it must not offer
   * Resume. It previously matched no branch at all and fell through to the empty
   * default, which rendered the bar as "Ready to optimize" while the run had in
   * fact died. */
  const isFailed = optStatus === "failed";
  /* Any non-null status means we have something meaningful to show in
   * the bar — running progress, paused state, converged summary, or a
   * starting spinner. The harness fetch can race the optimization
   * state on first load; without this split, a 404 on the initial
   * harness probe hides the bar for the entire eval phase of
   * iteration 1 (often 5–10 minutes). */
  const hasActiveOpt = optStatus !== null;

  /* Standalone eval liveness signal for visibility. The reconciled `/activity`
   * slice is authoritative and active by construction — its presence alone is
   * enough and is the EARLIER signal (progress fold seeds from it one commit
   * later). Requiring the folded progress as well would reopen the same blind
   * window, just narrower. This is intentionally NOT the precedence boolean
   * further down (`hasActiveEval`), which yields to a running optimization on
   * purpose — "who owns the status line" and "is there anything to show at
   * all" are different questions. Note: the eval label may land on the
   * following commit once the fold seeds; we don't chase that frame.
   * IMPORTANT: Excludes `awaiting_user` status — that state has no matching
   * status-text branch and must not surface the bar. */
  const hasActiveEvalJob =
    (activeEval?.status === "running" || activeEval?.status === "pending") ||
    (evalProgress != null &&
      (evalProgress.status === "running" || evalProgress.status === "pending"));

  // Wait one tick to avoid a flash of "Start" between mount and first
  // harness response.
  if (!harnessChecked) return null;
  // Hide only when there's nothing to show in either branch — no
  // harness for the Start button AND no optimization state AND no
  // standalone eval to report.
  if (!harnessConfig && !hasActiveOpt && !hasActiveEvalJob) return null;

  // Is there an active eval job (not part of a running optimization)?
  const hasActiveEval = evalProgress != null &&
    (evalProgress.status === "running" || evalProgress.status === "pending") &&
    !isOptRunning && !isStopping && !isStarting;

  // Determine status text (left side)
  let statusIcon: React.ReactNode;
  let statusText: string;

  if (isStarting) {
    statusIcon = <Loader2 className="size-3.5 animate-spin text-info" />;
    statusText = "Starting optimization...";
  } else if (isStopping) {
    statusIcon = <Loader2 className="size-3.5 animate-spin text-muted-foreground" />;
    statusText = "Stopping optimization...";
  } else if (isOptRunning) {
    const currentPrompt = optState?.current_prompt_version;
    const promptLabel = currentPrompt ? `, prompt ${currentPrompt}` : "";
    const iteration = `Iteration ${(optState?.current_iteration ?? 0) + 1} of ${optState?.iteration_cap ?? "?"}${promptLabel}`;
    if (evalProgress && (evalProgress.status === "running" || evalProgress.status === "pending")) {
      const activeModel = evalProgress.models.find((m) => m.phase !== "complete");
      const modelsCompleted = evalProgress.models.filter((m) => m.phase === "complete").length;
      const modelInfo = evalProgress.modelsTotal > 0
        ? ` — Evaluation running, model ${modelsCompleted + (activeModel ? 1 : 0)} of ${evalProgress.modelsTotal}`
        : " — Evaluation running";
      statusText = `Optimization: ${iteration}${modelInfo}`;
    } else if (optState?.phase === "optimizing_prompt") {
      /* The one phase the bar can now claim honestly: the orchestrator
       * wrote it to the durable record when the eval finished and the
       * prompt-optimizer turn began. (The old bar ASSERTED this label as
       * a blind fallback — including on iteration-1 spin-up and on dead
       * loops. Now it's earned: one writer, one reader.) */
      statusText = `Optimization: ${iteration} — Analyzing failures & writing next prompt...`;
    } else if (optState?.phase === "evaluating") {
      /* Eval phase but no job-progress signal yet — the eval is spinning
       * up (job created, models not reporting). */
      statusText = `Optimization: ${iteration} — Starting evaluation...`;
    } else {
      /* Loop is live but hasn't entered a phase yet, and there is no
       * eval-progress signal: say what we know, no more. */
      statusText = `Optimization: ${iteration} — working...`;
    }
    statusIcon = <Loader2 className="size-3.5 animate-spin text-info" />;
  } else if (hasActiveEval) {
    // Active eval takes priority over paused/converged optimization
    statusIcon = <Loader2 className="size-3.5 animate-spin text-info" />;
    const activeModel = evalProgress?.models.find((m) => m.phase !== "complete");
    const modelsCompleted = evalProgress?.models.filter((m) => m.phase === "complete").length ?? 0;
    const total = evalProgress?.modelsTotal ?? 0;
    const modelInfo = total > 0
      ? `, model ${modelsCompleted + (activeModel ? 1 : 0)} of ${total}`
      : "";
    // Show generation progress while generating, scoring progress
    // while scoring. Falls back to a phase-only label when no per-case
    // counter is available yet.
    let caseInfo = "";
    if (activeModel && activeModel.casesTotal > 0) {
      if (activeModel.phase === "scoring" && !activeModel.scored) {
        caseInfo = activeModel.scoredCount > 0
          ? ` (scoring ${activeModel.scoredCount}/${activeModel.casesTotal})`
          : ` (scoring ${activeModel.casesTotal} cases)`;
      } else {
        caseInfo = ` (${activeModel.generated}/${activeModel.casesTotal} cases)`;
      }
    }
    statusText = `Evaluation running${modelInfo}${caseInfo}`;
  } else if (isResumable) {
    statusIcon = <Pause className="size-3.5 text-muted-foreground" />;
    /* "plateau" is a verdict measured across the plateau window, which always
     * includes the terminal iteration — so like convergence it is withheld once
     * that run's file is gone. The remaining suffixes describe how the loop
     * stopped (exhausted, stalled, a failing judge), not a measurement taken
     * from a run, so a deleted artifact does not invalidate them. */
    const reason =
      optStatus === "plateau" ? (terminalEvidence ? " (plateau)" : "") :
      optStatus === "max_iterations" ? " (max iterations)" :
      optStatus === "optimizer_exhausted" ? " (optimizer exhausted — edit the prompt to resume)" :
      optStatus === "optimizer_stalled" ? " (optimizer stalled — review the run log and resume)" :
      optStatus === "judge_unreliable" ? " (judge unreliable — raise quota or pick another judge before resuming)" :
      optStatus === "generation_unreliable" ? " (target-model generation calls failed — retry or reduce generation concurrency before resuming)" :
      "";
    statusText = `Optimization paused at iteration ${optState?.current_iteration ?? 0}/${optState?.iteration_cap ?? "?"}${reason}`;
  } else if (isConverged) {
    statusIcon = <CheckCircle2 className="size-3.5 text-success" />;
    const iterCount = optState?.current_iteration ?? 0;
    statusText = terminalEvidence
      ? `Optimization converged at iteration ${iterCount}`
      : `Optimization ran ${iterCount} ${pluralize(iterCount, "iteration")}`;
  } else if (isFailed) {
    statusIcon = <AlertCircle className="size-3.5 text-destructive" />;
    statusText = `Optimization failed at iteration ${optState?.current_iteration ?? 0}${
      optState?.error?.message ? ` — ${optState.error.message}` : ""
    }`;
  } else {
    // Nothing running, no history or no special state
    statusIcon = null;
    statusText = "";
  }

  // Determine action button (right side)
  let actionButton: React.ReactNode = null;

  if (isStarting) {
    actionButton = (
      <Button variant="ghost" size="sm" disabled className="shrink-0 h-7 px-2.5 text-xs">
        <Loader2 className="size-3 mr-1 animate-spin" />
        Starting...
      </Button>
    );
  } else if (isStopping) {
    actionButton = (
      <Button variant="ghost" size="sm" disabled className="shrink-0 h-7 px-2.5 text-xs">
        <Loader2 className="size-3 mr-1 animate-spin" />
        Stopping...
      </Button>
    );
  } else if (isOptRunning) {
    actionButton = (
      <Button
        variant="ghost"
        size="sm"
        className="shrink-0 h-7 px-2.5 text-xs text-muted-foreground hover:text-destructive"
        onClick={handleStop}
        disabled={actionLoading}
      >
        <Square className="size-3 mr-1" />
        Stop
      </Button>
    );
  } else if (hasActiveEval) {
    actionButton = (
      <Button
        variant="ghost"
        size="sm"
        className="shrink-0 h-7 px-2.5 text-xs text-muted-foreground hover:text-destructive"
        onClick={cancelEvalJob}
        disabled={evalCancelling}
      >
        <Square className="size-3 mr-1" />
        {evalCancelling ? "Stopping..." : "Stop"}
      </Button>
    );
  } else if (isResumable) {
    actionButton = (
      <div className="flex items-center gap-1 shrink-0">
        <Button
          size="sm"
          className="h-7 px-2.5 text-xs"
          onClick={handleResume}
          disabled={actionLoading}
        >
          <RotateCcw className="size-3 mr-1" />
          Resume
        </Button>
        {/* Start-New requires a RUNNABLE harness (ready, not merely
         * parsed — see the Start Optimization branch below). When the
         * harness fetch is racing or transiently failing, hide it rather
         * than show a button that 404s. */}
        {harnessConfig?.ready && (
          <Button
            variant="secondary"
            size="sm"
            className="h-7 px-2.5 text-xs"
            onClick={() => {
              if (window.confirm(
                "Start a new optimization loop? This resets iteration history and begins fresh from the current prompt."
              )) {
                handleStart();
              }
            }}
            disabled={actionLoading}
          >
            <Play className="size-3 mr-1" />
            Start New
          </Button>
        )}
      </div>
    );
  } else if ((isConverged || isFailed) && harnessConfig?.ready) {
    // A failed run offers Start New, not Resume: the backend's resume endpoint
    // refuses status "failed", so a Resume button here would 409.
    actionButton = (
      <Button
        size="sm"
        className="shrink-0 h-7 px-2.5 text-xs"
        onClick={handleStart}
        disabled={actionLoading}
      >
        <Play className="size-3 mr-1" />
        Start New
      </Button>
    );
  } else if (harnessConfig?.ready) {
    // Default: show Start Optimization. Gate on READY, not config-exists:
    // config.py lands with the scaffold, minutes before the metric and
    // baseline prompt do — a button keyed on parse success lit up mid-build
    // and offered an optimization that would fail immediately (readiness is
    // measured live).
    actionButton = (
      <Button
        size="sm"
        className="shrink-0 h-7 px-2.5 text-xs"
        onClick={handleStart}
        disabled={actionLoading}
      >
        <Play className="size-3 mr-1" />
        Start Optimization
      </Button>
    );
  }

  // Has eval progress detail to show?
  const hasEvalDetail = evalProgress != null &&
    evalProgress.models.length > 0 &&
    (isOptRunning || hasActiveEval);

  const activeModel = evalProgress?.models.find((m) => m.phase !== "complete");

  return (
    <div className="rounded-lg border bg-card">
      {/* Scoring animation keyframes */}
      <style>{`
        @keyframes scoring-flow {
          0% { background-position: 0% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>

      {/* Main bar row */}
      <div className="flex items-center gap-3 px-3 h-12">
        {statusIcon}
        <span className="flex-1 min-w-0 text-sm truncate">
          {statusText || <span className="text-muted-foreground">Ready to optimize</span>}
        </span>

        {/* Surface harness-fetch failures (network/auth/5xx). "No harness
         *  yet" is a successful null response, not an error, so it never
         *  renders this chip. */}
        {harnessError && (
          <span
            className="shrink-0 rounded-full border border-destructive/50 bg-destructive/10 px-2 py-0.5 text-2xs text-destructive"
            title={harnessError.message}
          >
            Harness config unavailable
          </span>
        )}

        {/* Expand toggle for eval detail */}
        {hasEvalDetail && (
          <Button
            variant="ghost"
            size="sm"
            className="shrink-0 h-7 px-2 text-xs text-muted-foreground hover:text-foreground gap-1"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Hide details" : "Details"}
            {expanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
          </Button>
        )}

        {actionButton}
      </div>

      {/* Expandable per-model progress detail */}
      {expanded && hasEvalDetail && evalProgress && (
        <div className="border-t px-3 py-2 space-y-2">
          {evalProgress.models.map((entry) => (
            <ModelProgressRow
              key={entry.model}
              entry={entry}
              isActive={entry === activeModel}
            />
          ))}
        </div>
      )}

      {/* Extend iterations confirmation dialog */}
      <Dialog open={showExtendConfirm} onOpenChange={setShowExtendConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Extend Optimization</DialogTitle>
            <DialogDescription>
              The optimization has reached its iteration limit.
              Resume with {optState?.iterations_per_run ?? 10} more iterations?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowExtendConfirm(false)}>
              Cancel
            </Button>
            <Button onClick={() => doResume("raw")}>
              Resume
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Data-source picker — only rendered when approved synthetic cases exist. */}
      <Dialog
        open={dataSourceDialog != null}
        onOpenChange={(open) => { if (!open) setDataSourceDialog(null); }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Choose data source</DialogTitle>
            <DialogDescription>
              Which dataset should the optimization evaluate against?
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-2 py-2">
            {(["raw", "synthetic", "merged"] as const).map((value) => {
              const label =
                value === "raw"
                  ? "Raw"
                  : value === "synthetic"
                  ? "Synthetic only"
                  : "Merged (raw + synthetic)";
              const hint =
                value === "raw"
                  ? harnessConfig?.data_file
                    ? `Uses ${harnessConfig.data_file}`
                    : "Uses the harness-configured dataset"
                  : value === "synthetic"
                  ? `${approvedSyntheticTotal} approved synthetic case${approvedSyntheticTotal === 1 ? "" : "s"}`
                  : `Raw dataset + ${approvedSyntheticTotal} approved synthetic`;
              const isSelected = dataSourceDialog?.selected === value;
              return (
                <label
                  key={value}
                  className={`flex items-start gap-3 rounded-md border p-3 cursor-pointer transition-colors ${
                    isSelected ? "border-primary bg-accent" : "hover:bg-muted/50"
                  }`}
                >
                  <input
                    type="radio"
                    name="data-source"
                    className="mt-1"
                    checked={isSelected}
                    onChange={() => setDataSourceDialog(
                      (prev) => prev ? { ...prev, selected: value } : prev,
                    )}
                  />
                  <div className="flex flex-col gap-0.5">
                    <span className="text-sm font-medium">{label}</span>
                    <span className="text-xs text-muted-foreground">{hint}</span>
                  </div>
                </label>
              );
            })}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDataSourceDialog(null)}>
              Cancel
            </Button>
            <Button onClick={handleDataSourceConfirm}>
              {dataSourceDialog?.intent === "resume" ? "Resume" : "Start"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
