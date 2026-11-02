// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure reducer for the eval-job progress state machine.
 *
 * Extracted from `useEvalJobStream` so the fold can be tested without a React
 * tree, a WebSocket, or a timer. The logic was already a fold — every branch
 * computed `next` from `prev` plus one envelope — but it was interleaved with
 * `setProgress` calls and two `useRef` mutations, so nothing could exercise it.
 * That is why the "stale bar" family of bugs (bar frozen at "model 0 of N",
 * counts regressing on remount, a cancelled job leaving a live Stop button)
 * kept recurring: the state machine had no test surface at all.
 *
 * Two pieces of mutable state the hook kept in refs are modelled explicitly here
 * so the reducer stays pure:
 *
 * - `completedCases` — per-model set of completed case indices. The harness runs
 *   cases across parallel workers, so `generating` events arrive out of order and
 *   a naive counter regresses. Counting distinct indices is what makes the
 *   generated count monotonic.
 * - `modelsTotal` — captured once when a job is promoted, not carried on each
 *   event.
 *
 * Both live in {@link EvalProgressMachine} alongside the rendered state, so a
 * caller threads one value and the reducer never reaches outside its arguments.
 */

import type { ActivityModelProgress, EvalJob } from "@/lib/types";

/** Default number of parallel workers in the harness template. */
export const DEFAULT_WORKERS = 5;

/** Per-model progress tracking entry. */
export interface ModelProgress {
  /** Model identifier (e.g. "anthropic.claude-3-haiku") */
  model: string;
  /** Current evaluation phase */
  phase: "generating" | "scoring" | "complete";
  /** Number of cases whose generation has completed */
  generated: number;
  /** Total number of cases to generate */
  casesTotal: number;
  /** Whether scoring has finished */
  scored: boolean;
  /** Number of cases scored so far during the scoring phase. */
  scoredCount: number;
  /** Number of cases that passed scoring */
  passed: number;
  /** Number of parallel workers (from event or default) */
  workers: number;
}

/** Top-level progress state derived from WebSocket envelopes. */
export interface EvalProgressState {
  /** Job status from REST API */
  status: EvalJob["status"];
  /** High-level phase */
  phase: "pending" | "generating" | "scoring" | "complete" | "failed" | "cancelled";
  /** Total models being evaluated */
  modelsTotal: number;
  /** Per-model progress entries */
  models: ModelProgress[];
  /** Error message if failed */
  error: string | null;
}

/**
 * Reducer state: the rendered progress plus the bookkeeping the hook kept in
 * refs. Held together so the fold is a total function of its inputs.
 */
export interface EvalProgressMachine {
  /** Rendered progress, or null before any event/seed. */
  progress: EvalProgressState | null;
  /**
   * Per-model completed case indices. Keyed by model name; the VALUE is a set of
   * case indices, not a count, because parallel workers report out of order.
   */
  completedCases: Map<string, Set<number>>;
  /** Total models for the active job, captured at promotion. */
  modelsTotal: number;
}

/** Minimal envelope shape the reducer needs (kind + data). */
export interface ProgressEnvelope {
  kind: string;
  data?: unknown;
}

/**
 * Side effects the fold requests but does not perform.
 *
 * The hook must fire `onComplete` exactly once per job and notify
 * `onStatusChange` on terminal transitions. Returning them as data keeps the
 * reducer pure and makes "fires exactly once" assertable — the duplicate
 * completion-turn class of bug is invisible when the callback is buried in the
 * fold.
 */
export interface EvalProgressEffects {
  /** Terminal status observed on a `done` envelope, if any. */
  statusChanged?: EvalJob["status"];
  /** True when this envelope completed the job for the first time. */
  completed?: boolean;
}

/** Result of folding one envelope. */
export interface EvalProgressResult {
  state: EvalProgressMachine;
  effects: EvalProgressEffects;
}

/** A machine with no job in flight. */
export function initialMachine(modelsTotal = 0): EvalProgressMachine {
  return { progress: null, completedCases: new Map(), modelsTotal };
}

/**
 * Map a backend {@link ActivityModelProgress} (snake_case wire shape) to a
 * client {@link ModelProgress}.
 *
 * Seeds `generated` from the scalar but deliberately does NOT populate the
 * per-model completed-case set (the wire shape carries a count, not case
 * indices). A later live `generating` event for the model therefore rebuilds
 * `generated` from that empty set and the count drops. That is accepted
 * because a mid-scoring remount is already past the generate phase, where no
 * further `generating` events are expected.
 */
export function mapActivityModel(m: ActivityModelProgress): ModelProgress {
  return {
    model: m.model,
    phase: m.phase,
    generated: m.generated,
    casesTotal: m.cases_total,
    scored: m.scored,
    scoredCount: m.scored_count,
    passed: m.passed,
    workers: DEFAULT_WORKERS,
  };
}

/** Apply `updater` to the named model entry, creating it if absent. */
function updateModel(
  models: ModelProgress[],
  modelName: string,
  updater: (entry: ModelProgress) => ModelProgress,
): ModelProgress[] {
  const idx = models.findIndex((m) => m.model === modelName);
  if (idx >= 0) {
    const updated = [...models];
    updated[idx] = updater(updated[idx]);
    return updated;
  }
  const fresh: ModelProgress = {
    model: modelName,
    phase: "generating",
    generated: 0,
    casesTotal: 0,
    scored: false,
    scoredCount: 0,
    passed: 0,
    workers: DEFAULT_WORKERS,
  };
  return [...models, updater(fresh)];
}

/** Clone the completed-case map so the reducer never mutates its input. */
function cloneCompleted(
  completed: Map<string, Set<number>>,
): Map<string, Set<number>> {
  const next = new Map<string, Set<number>>();
  for (const [model, cases] of completed) next.set(model, new Set(cases));
  return next;
}

/**
 * Fold one envelope into the progress machine.
 *
 * @param state - Current machine state.
 * @param env - Envelope from the `job:<id>` stream.
 * @param alreadyCompletedJobId - Job id already reported complete, so the
 *   `completed` effect is requested at most once per job (the hook's
 *   `completedRef`).
 * @param jobId - The active job id, compared against `alreadyCompletedJobId`.
 * @returns The next state plus any effects the caller should perform.
 */
export function reduceEvalProgress(
  state: EvalProgressMachine,
  env: ProgressEnvelope,
  alreadyCompletedJobId: string | null = null,
  jobId: string | null = null,
): EvalProgressResult {
  const { modelsTotal } = state;
  const prev = state.progress;

  if (env.kind === "progress") {
    const data = (env.data ?? {}) as Record<string, unknown>;
    const eventType = data.event as string;
    const modelName = (data.model as string) ?? "";

    if (eventType === "generating") {
      const completedCases = cloneCompleted(state.completedCases);
      let set = completedCases.get(modelName);
      if (!set) {
        set = new Set<number>();
        completedCases.set(modelName, set);
      }
      if (data.current != null) set.add(data.current as number);
      const generated = set.size;
      const total = (data.total as number) ?? 0;
      const workers = (data.workers as number) ?? DEFAULT_WORKERS;
      const models = updateModel(prev?.models ?? [], modelName, (entry) => ({
        ...entry,
        phase: "generating",
        generated,
        casesTotal: total,
        workers,
      }));
      return {
        state: {
          ...state,
          completedCases,
          progress: {
            status: "running",
            phase: "generating",
            modelsTotal,
            models,
            error: null,
          },
        },
        effects: {},
      };
    }

    if (eventType === "scoring_start") {
      const models = updateModel(prev?.models ?? [], modelName, (entry) => ({
        ...entry,
        phase: "scoring",
        generated: entry.casesTotal,
        scored: false,
        scoredCount: 0,
      }));
      return {
        state: {
          ...state,
          progress: {
            status: "running",
            phase: "scoring",
            modelsTotal,
            models,
            error: null,
          },
        },
        effects: {},
      };
    }

    if (eventType === "scoring_progress") {
      const current = (data.current as number) ?? 0;
      const total = (data.total as number) ?? 0;
      const models = updateModel(prev?.models ?? [], modelName, (entry) => ({
        ...entry,
        phase: "scoring",
        scoredCount: current,
        casesTotal: total || entry.casesTotal,
      }));
      return {
        state: {
          ...state,
          progress: {
            status: "running",
            phase: "scoring",
            modelsTotal,
            models,
            error: null,
          },
        },
        effects: {},
      };
    }

    if (eventType === "scoring_complete") {
      const models = updateModel(prev?.models ?? [], modelName, (entry) => ({
        ...entry,
        phase: "scoring",
        scored: true,
        scoredCount: entry.casesTotal,
        passed: (data.passed as number) ?? 0,
      }));
      return {
        state: {
          ...state,
          progress: {
            status: "running",
            phase: "scoring",
            modelsTotal,
            models,
            error: null,
          },
        },
        effects: {},
      };
    }

    if (eventType === "complete") {
      // Drop this model's dedup set — the job may still be running other
      // models, and holding completed sets for finished models leaks memory
      // across a long multi-model run.
      const completedCases = cloneCompleted(state.completedCases);
      completedCases.delete(modelName);
      const models = updateModel(prev?.models ?? [], modelName, (entry) => ({
        ...entry,
        phase: "complete",
        scored: true,
        passed: (data.passed as number) ?? entry.passed,
        generated: entry.casesTotal,
      }));
      return {
        state: {
          ...state,
          completedCases,
          progress: {
            status: "running",
            phase: "complete",
            modelsTotal,
            models,
            error: null,
          },
        },
        effects: {},
      };
    }

    if (eventType === "error") {
      return {
        state: {
          ...state,
          progress: {
            status: "failed",
            phase: "failed",
            modelsTotal,
            models: prev?.models ?? [],
            error: (data.error as string) ?? "Unknown error",
          },
        },
        effects: {},
      };
    }

    // Unknown progress event — no state change. Deliberately not an error: the
    // harness may emit events this client predates, and dropping the whole
    // envelope must never wedge the bar.
    return { state, effects: {} };
  }

  if (env.kind === "done") {
    const data = (env.data ?? {}) as Record<string, unknown>;
    const finalStatus = data.status as EvalJob["status"];
    const effects: EvalProgressEffects = { statusChanged: finalStatus };

    if (finalStatus === "completed") {
      // `completed` is requested at most once per job id: the hook fires a
      // completion callback here, and firing twice double-reports a finished run.
      if (jobId && jobId !== alreadyCompletedJobId) effects.completed = true;
      return {
        state: {
          ...state,
          progress: prev
            ? { ...prev, status: "completed", phase: "complete" }
            : null,
        },
        effects,
      };
    }
    if (finalStatus === "failed") {
      return {
        state: {
          ...state,
          progress: prev ? { ...prev, status: "failed", phase: "failed" } : null,
        },
        effects,
      };
    }
    if (finalStatus === "cancelled") {
      return {
        state: {
          ...state,
          progress: prev
            ? { ...prev, status: "cancelled", phase: "cancelled" }
            : null,
        },
        effects,
      };
    }
    return { state, effects };
  }

  return { state, effects: {} };
}
