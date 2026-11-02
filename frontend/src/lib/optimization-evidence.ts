// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Run-file existence and terminal-evidence derivation for optimization displays.
 *
 * The run-file list is the sole existence record. Optimization state supplies
 * metadata only (iteration number, prompt version, per-model rates). This
 * module is the single home for the iteration-to-run-file join and the
 * terminal-evidence question — no consumer hand-rolls a second check.
 *
 * Terminal evidence: the last iteration's run file is the anchor because the
 * backend's convergence check reads `iterations[-1]` for the target verdict,
 * and the plateau window always contains it (by construction). Every terminal
 * verdict therefore depends on that iteration's eval results.
 */

import type { EvalRunSummary, OptimizationIteration, OptimizationState } from "./types";

/**
 * Resolve an optimization iteration to its run file, if present.
 *
 * Returns the `_file` basename of the run backing this iteration, or null
 * when no matching file exists. An iteration's `run_id` IS its run filename:
 *
 * 1. Empty run_id → null (iteration never had a run)
 * 2. Exact `_file` match → that file
 * 3. Anything else → null
 *
 * An iteration with no surviving run file produces no card: the run-file
 * list is the one existence record.
 *
 * There used to be a third step for iterations recorded before run files were
 * materialized: correlate a bare identifier against the newest loop-sourced run
 * at the same prompt version. It needed a source guard so an iteration could not
 * steal a user's manual run, and a `.json` guard so a genuinely missing file did
 * not get some other run's results attributed to it — two guards protecting a
 * guess that only ever existed to read pre-release data. Deleted with it.
 *
 * @param iteration - Optimization iteration record from state
 * @param runs - Full unfiltered run list (already sorted newest-first)
 * @returns The matching run file basename, or null if absent
 */
export function resolveIterationRunFile(
  iteration: OptimizationIteration,
  runs: EvalRunSummary[],
): string | null {
  const runId = iteration.run_id;
  // Empty or falsy run_id means the iteration never had a run file.
  if (!runId) return null;

  // Exact match: run_id is a file basename that exists in the run list.
  const exactMatch = runs.find((r) => r._file === runId);
  return exactMatch ? exactMatch._file : null;
}

/**
 * Status values that require the terminal iteration's eval evidence to survive.
 * Keyed on the OptimizationState.status union so adding a status is a build
 * error rather than a silent default.
 */
export const STATUS_NEEDS_ITERATION_EVIDENCE: Record<
  NonNullable<OptimizationState["status"]>,
  boolean
> = {
  // Verdicts computed from iteration metrics — require evidence
  converged: true,
  plateau: true,

  // All others do not require evidence:
  // - Counts (max_iterations, cancelled) are loop metadata
  // - User actions (cancelled) don't assert measurements
  // - Error events (failed, optimizer_exhausted, optimizer_stalled,
  //   judge_unreliable, generation_unreliable) describe loop failures,
  //   not iteration results
  // - Live states (running, starting, stopping) have no terminal verdict
  max_iterations: false,
  cancelled: false,
  failed: false,
  optimizer_exhausted: false,
  optimizer_stalled: false,
  judge_unreliable: false,
  generation_unreliable: false,
  running: false,
  starting: false,
  stopping: false,
};

/**
 * Resolve the terminal iteration's evidence: returns the last iteration when
 * its run file is present, otherwise null.
 *
 * Used to gate displays that assert something measured from the terminal
 * iteration's eval results. When the run file is gone, the badge, metrics,
 * and verdict-highlight are withheld. Iteration counts and non-evidence
 * statuses remain visible (they are loop metadata, not measurements).
 *
 * @param state - Optimization state (nullable — returns null for no state)
 * @param runs - Full unfiltered run list
 * @returns The terminal iteration if its run file survives, else null
 */
export function survivingTerminalIteration(
  state: OptimizationState | null,
  runs: EvalRunSummary[],
): OptimizationIteration | null {
  if (state === null || state.iterations.length === 0) return null;

  const terminal = state.iterations[state.iterations.length - 1];
  return resolveIterationRunFile(terminal, runs) !== null ? terminal : null;
}

/**
 * Check whether ANY iteration in the optimization state has a surviving run file.
 *
 * Returns false for null state or empty iteration list; otherwise true when
 * at least one iteration resolves to a run file via `resolveIterationRunFile`.
 *
 * This answers a different question from `survivingTerminalIteration`: that
 * function anchors on the LAST iteration because a terminal verdict is a
 * property of the terminal run. This function asks whether the loop has ANY
 * surviving artifact, which decides whether the panel still has loop content
 * to show. A middle iteration surviving while the terminal one is deleted must
 * return true here and null there — the two are not interchangeable.
 *
 * Used for the other-runs checkbox default: an optimization record whose run
 * evidence has been deleted cannot support a loop narrative (the panel already
 * withholds the verdict for that reason), so it must not also hide the
 * project's only run behind an unchecked box.
 *
 * @param state - Optimization state (nullable — returns false for no state)
 * @param runs - Full unfiltered run list
 * @returns True if any iteration has a surviving run file, else false
 */
export function hasAnySurvivingIterationRun(
  state: OptimizationState | null,
  runs: EvalRunSummary[],
): boolean {
  if (state === null || state.iterations.length === 0) return false;

  for (const iteration of state.iterations) {
    if (resolveIterationRunFile(iteration, runs) !== null) {
      return true;
    }
  }

  return false;
}
