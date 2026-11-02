// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Optimization detail panel showing iteration history, metrics vs targets,
 * and convergence criteria. Lives inside EvalDashboard (Overview tab).
 * No action buttons — all controls live in ActiveJobBar.
 *
 * Card semantics:
 *   - Each card's big number is the BEST per-model pass rate on that
 *     iteration (the "any model wins" convergence rule, made visible).
 *   - The trophy goes to the card with the highest server-computed
 *     TrophyScore, compared in order: at_target (models meeting every
 *     configured target), then best_rate (highest single-model pass
 *     rate), then avg_rate (mean pass rate across models); on a full tie
 *     the later card wins. Both loop iterations and manual runs are
 *     eligible; the rule doesn't discriminate.
 *
 * Clicking a card routes the eval-runs view to that iteration's run file
 * via the ``?run=`` URL param — same mechanism the run dropdown uses, so
 * the two stay in sync.
 */

import { useState } from "react";
import { Trophy, User, History } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { DEFAULT_PLATEAU_THRESHOLD } from "@/lib/backend-defaults";
import { ResourceErrorBanner } from "@/components/ui/resource-error";
import { fmtPctInt, pluralize, shortModel } from "@/lib/format-utils";
import {
  hasAnySurvivingIterationRun,
  resolveIterationRunFile,
  STATUS_NEEDS_ITERATION_EVIDENCE,
  survivingTerminalIteration,
} from "@/lib/optimization-evidence";
import type { EvalRunSummary, HarnessConfig, OptimizationIteration, OptimizationState, TrophyScore } from "@/lib/types";

/** Highest per-model pass rate on this iteration + the winning model_id.
 *
 * Returns 0 with a null modelId when the row has no per-model data, matching
 * `bestModelOnManualRun`. It used to substitute the row's AGGREGATE rate, which
 * put a cross-model average in a field labelled "best model" — and left the two
 * sibling helpers disagreeing about the same question. */
function bestModelOnIteration(
  it: OptimizationIteration,
): { rate: number; modelId: string | null } {
  const pm = it.per_model_pass_rates;
  if (pm && Object.keys(pm).length > 0) {
    let bestModel: string | null = null;
    let bestRate = -Infinity;
    for (const [modelId, rate] of Object.entries(pm)) {
      if (rate > bestRate) {
        bestRate = rate;
        bestModel = modelId;
      }
    }
    return { rate: bestRate, modelId: bestModel };
  }
  return { rate: 0, modelId: null };
}

/** Convergence criteria sourced from harness config (the single home for
 * these values). These determine when the optimization loop declares
 * convergence, plateau, or max-iterations-reached. The panel reads them
 * from harness config exclusively — NOT from the optimization state, which
 * historically kept a copy but created a dual-home pattern. */
export type ConvergenceCriteria = {
  /** Target pass rates per metric (e.g. {"ExactMatch": 0.85}) */
  targets: Record<string, number>;
  /** Pass-rate spread threshold for plateau detection (0.0–1.0) */
  plateauThreshold: number;
  /** Number of iterations to check for plateau detection. Nullable: the
   * backend REQUIRES this in the harness config (no default), so
   * an unset value means "not configured" — the badge is omitted, never
   * fabricated. */
  plateauWindow: number | null;
  /** Maximum optimization iterations allowed (nullable) */
  maxIterations: number | null;
};

/** Extract convergence criteria from harness config. Returns null when
 * config is entirely absent (no src/config.py). When config exists but a
 * field is null, applies the backend default only for the plateau
 * threshold (0.02). plateau_window and max_iterations pass through as
 * null: the backend requires both and never defaults them, so defaulting
 * here would fabricate a criterion the project never configured (the 10
 * that appears at loop start is the loop's choice, not a config value). */
export function convergenceCriteriaFromHarness(
  config: HarnessConfig | null,
): ConvergenceCriteria | null {
  if (config === null) return null;
  return {
    targets: config.target_metric_scores ?? {},
    // The threshold default mirrors the backend's genuine default through
    // the single mirror module — see backend-defaults.ts for the owner.
    plateauThreshold: config.plateau_threshold ?? DEFAULT_PLATEAU_THRESHOLD,
    // plateau_window and max_iterations pass through nullable: the
    // backend REQUIRES both in the harness config (it rejects a start
    // without them, never defaults them), so an unset value here means
    // "not configured" and the badge is omitted. The `?? 3` this replaces
    // mirrored a backend default that no longer exists — it would have
    // rendered a criterion the backend would refuse to run with.
    plateauWindow: config.plateau_window,
    maxIterations: config.max_iterations,
  };
}

/** Trophy selection over SERVER-computed scores (the backend is the
 * single scoring implementation; this is ordering only — lexicographic on
 * (at_target, best_rate, avg_rate), later wins on a full tie). The golden
 * vector at backend/tests/fixtures/trophy_golden.json pins this comparator
 * and the backend scorer against the same expectations. */
export function pickTrophyIndex(scores: TrophyScore[]): number {
  let bestIdx = -1;
  let best: TrophyScore | null = null;
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    if (
      !best ||
      s.at_target > best.at_target ||
      (s.at_target === best.at_target && s.best_rate > best.best_rate) ||
      (s.at_target === best.at_target &&
        s.best_rate === best.best_rate &&
        s.avg_rate > best.avg_rate) ||
      // Final tiebreak: later iteration wins.
      (s.at_target === best.at_target &&
        s.best_rate === best.best_rate &&
        s.avg_rate === best.avg_rate)
    ) {
      best = s;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Props for the OptimizationRunDetail component. */
interface OptimizationRunDetailProps {
  /** Current optimization state from the backend. Null when the project has
   * eval runs but no optimization loop record (for example, a project that
   * has only run manual evals). The panel renders run cards, badges, and
   * trophy ranking with state=null; only loop-progress lines (Iteration N
   * of M, Latest vs target) are suppressed. */
  state: OptimizationState | null;
  /** Convergence criteria sourced from harness config (targets, plateau
   * threshold/window, max iterations). Null when harness config is entirely
   * absent (no src/config.py). The panel reads these from config exclusively;
   * the optimization state's historical copy is no longer read. */
  criteria: ConvergenceCriteria | null;
  /** Full unfiltered eval run list for the project. This list is the sole
   * existence record for cards — an iteration whose run file is absent from
   * this list produces no card. The optimization state supplies metadata only
   * (iteration number, prompt version, timestamp, per-model rates, trophy
   * score). The panel partitions the list into loop cards (iterations that
   * resolve to a run file) and manual cards (runs not claimed by any
   * iteration) internally, so the "Show other runs" toggle continues to gate
   * only unclaimed runs. */
  runs?: EvalRunSummary[];
}

/** One card in the iteration strip. Loop iterations and manual runs share
 * this shape so the strip can render a merged timeline uniformly. The
 * ``trophy`` score drives trophy selection per the rule documented on
 * ``TrophyScore``. ``rate``/``modelId`` are for the card's display-only
 * peak-model line and are unrelated to trophy selection. */
type IterationCard = {
  kind: "loop";
  runId: string | undefined;
  promptVersion: string;
  rate: number;
  modelId: string | null;
  timestamp: string | null;
  iterationKey: string | number;
  trophy: TrophyScore;
} | {
  kind: "manual";
  /** "manual" = standalone eval typed by the user; "optimization" =
   *  orphan from a previous optimization loop that was reset on restart.
   *  Both render under "Show other runs" but with different icons. */
  source: "manual" | "optimization";
  runId: string;
  promptVersion: string;
  rate: number;
  modelId: string | null;
  timestamp: string;
  iterationKey: string;
  trophy: TrophyScore;
};

/** Best per-model rate on a manual run — mirrors bestModelOnIteration
 * but consumes EvalRunSummary which flattens per-model data differently. */
function bestModelOnManualRun(run: EvalRunSummary): { rate: number; modelId: string | null } {
  let bestModel: string | null = null;
  let bestRate = -Infinity;
  for (const mr of run.model_runs) {
    const rate = mr.total > 0 ? mr.passed / mr.total : 0;
    if (rate > bestRate) {
      bestRate = rate;
      bestModel = mr.model;
    }
  }
  if (bestModel === null) return { rate: 0, modelId: null };
  return { rate: bestRate, modelId: bestModel };
}

/** Status labels map exported for test matrix derivation. Keyed on the
 * OptimizationState["status"] union so tsc catches a missing map entry. */
export const OPT_STATUS_LABELS: Record<OptimizationState["status"], string> = {
  starting: "Starting",
  running: "Optimizing",
  stopping: "Stopping",
  converged: "Target Reached",
  plateau: "Plateau Detected",
  max_iterations: "Max Iterations",
  optimizer_exhausted: "Optimizer Exhausted",
  optimizer_stalled: "Optimizer Stalled",
  judge_unreliable: "Judge Unreliable",
  generation_unreliable: "Generation Unreliable",
  failed: "Failed",
  cancelled: "Cancelled",
};

/**
 * Renders optimization loop detail inside the EvalDashboard.
 * Shows iteration cards with pass rates, target metric progress,
 * and convergence criteria with highlighting for the termination cause.
 */
export function OptimizationRunDetail({ state, criteria, runs = [] }: OptimizationRunDetailProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRun = searchParams.get("run");
  // Default the toggle based on whether any iteration has surviving run evidence.
  // Checked when no iterations have surviving runs (an unchecked box would hide the
  // project's only run), unchecked when loop evidence exists (keep the optimizer
  // narrative clean). An optimization record whose run evidence has been deleted
  // cannot support a loop narrative — the panel already withholds the verdict for
  // that reason — so it must not also hide the project's only run behind an
  // unchecked box.
  //
  // This is a mount-time decision. Safe to read `runs` here: the dashboard gate
  // requires `runs.length > 0` before mounting this panel, and `runs` is replaced
  // whole per fetch (never appended), so a mounted panel always has a complete
  // snapshot.
  const [showManual, setShowManual] = useState(!hasAnySurvivingIterationRun(state, runs));

  /* Typed as Record<OptimizationState["status"], …> deliberately: with the old
   * Record<string, string> a status the backend writes but the map omits looked
   * up to undefined and rendered as an unstyled, unlabelled badge. judge_unreliable,
   * generation_unreliable and failed were all missing. Keying on the union makes
   * the next added status a build error here instead of a blank badge in the UI. */
  const statusColors: Record<OptimizationState["status"], string> = {
    starting: "bg-info/10 text-info",
    running: "bg-info/10 text-info",
    stopping: "bg-warning/10 text-warning",
    converged: "bg-success/10 text-success",
    plateau: "bg-warning/10 text-warning",
    max_iterations: "bg-muted text-muted-foreground",
    optimizer_exhausted: "bg-warning/10 text-warning",
    optimizer_stalled: "bg-warning/10 text-warning",
    judge_unreliable: "bg-destructive/10 text-destructive",
    generation_unreliable: "bg-destructive/10 text-destructive",
    failed: "bg-destructive/10 text-destructive",
    cancelled: "bg-muted text-muted-foreground",
  };

  /* Terminal-iteration evidence gate.
   *
   * The run-file list is the sole existence record. A verdict computed from a
   * run's eval results — the status badge, the criteria-row highlight, and the
   * latest-vs-target metrics — is only displayable while that run's file
   * survives. When it is deleted, showing the last known numbers would assert a
   * measurement whose artifact no longer exists and cannot be clicked through
   * to, so those displays are withheld together rather than individually.
   *
   * The terminal iteration is the right anchor: the backend's convergence check
   * reads the last iteration for the target verdict, and the plateau window
   * always contains it, so every terminal verdict depends on that iteration's
   * results.
   *
   * Iteration counts are deliberately NOT gated — deleting an output file does
   * not un-run an iteration, and a count invites no artifact inspection. */
  const terminalEvidence = survivingTerminalIteration(state, runs);
  const verdictNeedsEvidence = state?.status
    ? STATUS_NEEDS_ITERATION_EVIDENCE[state.status]
    : false;
  const showVerdict = !verdictNeedsEvidence || terminalEvidence !== null;

  /** Render a 0–1 target as a percent, clamped to [0, 100]. Targets are
   *  always 0–1 pass rates (the backend rejects raw-unit targets at
   *  dispatch); the clamp is a defensive guard so an out-of-range value
   *  renders "100%" instead of "300000%". */
  const formatTargetPct = (target: number): string =>
    `${Math.round(Math.min(Math.max(target, 0), 1) * 100)}%`;

  /** Build the merged card list. Loop iterations appear only when their run
   * file is present in the runs list — the run file is the sole existence
   * record. Manual runs (unclaimed by any iteration) join when showManual is
   * on, interleaved chronologically so deltas between adjacent cards remain
   * meaningful. */
  const loopCards: IterationCard[] = [];
  const claimedFiles = new Set<string>();

  for (const it of state?.iterations ?? []) {
    const resolvedFile = resolveIterationRunFile(it, runs);
    if (!resolvedFile) continue; // No surviving run file, no card.

    claimedFiles.add(resolvedFile);
    const best = bestModelOnIteration(it);
    loopCards.push({
      kind: "loop",
      runId: resolvedFile,
      promptVersion: it.prompt_version,
      rate: best.rate,
      modelId: best.modelId,
      timestamp: it.timestamp ?? null,
      iterationKey: it.iteration,
      trophy: it.trophy_score,
    });
  }

  // Manual cards are runs not claimed by any iteration.
  const unclaimedRuns = runs.filter((r) => !claimedFiles.has(r._file));
  const manualCount = unclaimedRuns.length;

  const manualCards: IterationCard[] = showManual
    ? unclaimedRuns.map((r) => {
        const best = bestModelOnManualRun(r);
        return {
          kind: "manual",
          source: r.source === "optimization" ? "optimization" : "manual",
          runId: r._file,
          promptVersion: r.prompt_version,
          rate: best.rate,
          modelId: best.modelId,
          timestamp: r.timestamp,
          iterationKey: `manual-${r._file}`,
          trophy: r.trophy_score,
        };
      })
    : [];

  // Merge + sort by timestamp (fall back to insertion order for iterations
  // that don't carry a timestamp). Loop cards without a timestamp anchor
  // at their original position via a stable fallback key.
  const cards: IterationCard[] = [...loopCards, ...manualCards].sort((a, b) => {
    if (!a.timestamp && !b.timestamp) return 0;
    if (!a.timestamp) return -1;
    if (!b.timestamp) return 1;
    return a.timestamp.localeCompare(b.timestamp);
  });

  // Trophy selection: highest TrophyScore — at_target, then best_rate, then
  // avg_rate, with the later card winning a full tie. See TrophyScore doc for
  // the why.
  const trophyIdx = pickTrophyIndex(cards.map((c) => c.trophy));
  const trophyCard = trophyIdx >= 0 ? cards[trophyIdx] : null;

  /** Click handler: route the eval-runs view to this iteration's run file.
   * Uses the same ``?run=`` URL param the run dropdown writes to. */
  const selectRun = (runId: string | undefined) => {
    // Iteration.run_id holds the output filename when available.
    if (!runId) return;
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("run", runId);
      return next;
    }, { replace: true });
  };

  return (
    <Card className="p-4 space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-muted-foreground">Optimization</span>
          {state && showVerdict && (
            <Badge className={`text-2xs px-2 py-0 ${statusColors[state.status] ?? ""}`}>
              {OPT_STATUS_LABELS[state.status] ?? state.status}
            </Badge>
          )}
          {/* Manual-runs toggle — only shown when at least one manual run
              exists on the project. Default checked when no iteration has
              surviving run evidence, unchecked when loop evidence exists. */}
          {manualCount > 0 && (
            <label className="flex items-center gap-1 text-2xs text-muted-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showManual}
                onChange={(e) => setShowManual(e.target.checked)}
                className="size-3 accent-primary cursor-pointer"
              />
              Show other runs ({manualCount})
            </label>
          )}
        </div>
        {state && (
          <span className="text-xs text-muted-foreground tabular-nums">
            Iteration {state.current_iteration} of {state.iteration_cap}
          </span>
        )}
      </div>

      {/* Terminal-error banner — shown when the orchestrator stopped
          due to a failure (mass metric errors, dispatch failure, etc.). */}
      <ResourceErrorBanner error={state?.error ?? null} />


      {/* Iteration cards — horizontal scroll if many.
          Each card shows the best per-model rate on that iteration. The
          trophy goes to the card with the highest TrophyScore (models-at-
          target → bestRate → avgRate → later wins; see ``pickTrophyIndex``
          and the ``TrophyScore`` doc). Cards are buttons that route the
          eval-runs view to this iteration's run file. Other runs (when the
          toggle is on) carry a small icon in the top-right corner: a person
          for a manual run, a history mark for a run left by a previous
          optimization loop. */}
      {cards.length > 0 && (
        <div className="flex gap-2 overflow-x-auto py-1 px-0.5">
          {cards.map((card, i) => {
            const isTrophy = i === trophyIdx;
            const isSelected = selectedRun != null && card.runId === selectedRun;
            const pct = fmtPctInt(card.rate);
            const prevRate = i > 0 ? cards[i - 1].rate : null;
            const delta = prevRate != null ? (card.rate - prevRate) * 100 : null;
            const hoverTitle = card.modelId
              ? `${shortModel(card.modelId)} — open run`
              : "open run";
            const selectionRing = isSelected
              ? "ring-2 ring-inset ring-primary/70"
              : "";
            return (
              <button
                type="button"
                key={card.iterationKey}
                onClick={() => selectRun(card.runId)}
                title={hoverTitle}
                aria-pressed={isSelected}
                className={`relative shrink-0 rounded-md border px-3 py-2 text-center min-w-[80px] cursor-pointer transition-colors hover:bg-muted/60 focus:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring ${
                  isTrophy
                    ? "border-success/40 bg-success/5"
                    : "border-border bg-muted/30"
                } ${selectionRing}`}
              >
                {card.kind === "manual" && card.source === "manual" && (
                  <User
                    className="absolute top-1.5 right-1.5 size-2.5 text-muted-foreground/70"
                    aria-label="manual run"
                  />
                )}
                {card.kind === "manual" && card.source === "optimization" && (
                  <History
                    className="absolute top-1.5 right-1.5 size-2.5 text-muted-foreground/70"
                    aria-label="run from previous optimization loop"
                  />
                )}
                <div className="text-2xs text-muted-foreground mb-1">
                  {card.promptVersion}
                </div>
                <div className="flex items-center justify-center gap-1">
                  {isTrophy && <Trophy className="size-3 text-success" />}
                  <span className={`text-sm font-semibold tabular-nums ${isTrophy ? "text-success" : ""}`}>
                    {pct}%
                  </span>
                </div>
                {delta != null && (
                  <div className={`text-2xs tabular-nums ${delta > 0 ? "text-success" : delta < 0 ? "text-error" : "text-muted-foreground"}`}>
                    {delta > 0 ? "+" : ""}{delta.toFixed(1)}%
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Best model across the run. Always shown when the strip has data;
          tracks the trophy card's winning model, not the convergence-
          declared winner (the two agree once convergence fires). */}
      {trophyCard && trophyCard.modelId && (
        <div className="text-2xs text-muted-foreground">
          <span className="mr-1">Best model:</span>
          <span className="text-foreground font-medium">
            {shortModel(trophyCard.modelId)}
          </span>
          <span className="mx-1">at</span>
          <span className="text-foreground font-medium tabular-nums">
            {fmtPctInt(trophyCard.rate)}%
          </span>
          <span className="mx-1">on</span>
          <span className="text-foreground font-medium">
            {trophyCard.promptVersion}
          </span>
        </div>
      )}

      {/* Latest iteration metrics vs targets — loop progress, only shown
          when a loop record exists. Targets come from criteria, not from
          state. */}
      {state && terminalEvidence && (
        <div className="flex flex-wrap items-center gap-2 text-2xs">
          <span className="text-2xs text-muted-foreground">Latest vs target:</span>
          {Object.entries(criteria?.targets ?? {}).map(([name, target]) => {
            const latest = terminalEvidence;
            // Match metric by name — handle DeepEval's "[GEval]" suffix
            const metricKey = latest ? Object.keys(latest.metrics_summary).find(
              (k) => k === name || k.startsWith(name + " [")
            ) : undefined;
            const current = metricKey ? latest.metrics_summary[metricKey] : 0;
            const met = current >= target;
            return (
              <span
                key={name}
                className={`px-2 py-0.5 rounded ${
                  met ? "bg-success/10 text-success" : "bg-muted text-muted-foreground"
                }`}
              >
                {name}: {fmtPctInt(current)}% / {fmtPctInt(target)}%
              </span>
            );
          })}
        </div>
      )}

      {/* Convergence criteria — sourced from harness config exclusively.
          Omit the entire row when criteria is null (no harness config). */}
      {criteria && (
        <div className="flex flex-wrap items-center gap-1.5 text-2xs">
          <span className="text-2xs text-muted-foreground mr-1">Convergence:</span>
          {/* Target metric criteria */}
          {Object.entries(criteria.targets).map(([name, target]) => (
            <Badge
              key={name}
              variant="outline"
              className={`text-2xs font-normal ${
                state?.status === "converged" && showVerdict
                  ? "border-success/40 bg-success/10 text-success"
                  : ""
              }`}
            >
              {name} &ge; {formatTargetPct(target)}
            </Badge>
          ))}
          {/* Plateau criterion — omit when window unconfigured (same
            * null-omission rule as max iterations; the backend requires the
            * window and never defaults it, so rendering a number here would
            * fabricate a criterion) */}
          {criteria.plateauWindow !== null && (
            <Badge
              variant="outline"
              className={`text-2xs font-normal ${
                state?.status === "plateau" && showVerdict
                  ? "border-warning/40 bg-warning/10 text-warning"
                  : ""
              }`}
            >
              Plateau &lt; {Math.round(criteria.plateauThreshold * 100)}% over {criteria.plateauWindow} {pluralize(criteria.plateauWindow, "iteration")}
            </Badge>
          )}
          {/* Max iterations criterion — omit when null (not configured) */}
          {criteria.maxIterations !== null && (
            <Badge
              variant="outline"
              className={`text-2xs font-normal ${
                state?.status === "max_iterations"
                  ? "border-warning/40 bg-warning/10 text-warning"
                  : ""
              }`}
            >
              Max {criteria.maxIterations} {pluralize(criteria.maxIterations, "iteration")}
            </Badge>
          )}
        </div>
      )}
    </Card>
  );
}
