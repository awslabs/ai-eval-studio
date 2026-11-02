// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure-function tests for terminal-evidence derivation.
 *
 * The run-file list is the sole existence record. This module is the single
 * home for the iteration-to-run-file join and the terminal-evidence question.
 */

import { describe, it, expect } from "vitest";
import {
  hasAnySurvivingIterationRun,
  resolveIterationRunFile,
  survivingTerminalIteration,
  STATUS_NEEDS_ITERATION_EVIDENCE,
} from "./optimization-evidence";
import type { EvalRunSummary, OptimizationIteration, OptimizationState } from "./types";

// Fixture: optimization iteration
const makeIteration = (overrides: Partial<OptimizationIteration> = {}): OptimizationIteration => ({
  iteration: 1,
  prompt_version: "v1",
  run_id: "eval_results_v1_20260807_120000.json",
  timestamp: "2026-08-07T12:00:00Z",
  metrics_summary: { exactmatch: 0.85 },
  per_model_metrics: {},
  per_model_pass_rates: {},
  total_cases: 10,
  passed: 8,
  pass_rate: 0.8,
  trophy_score: { at_target: 1, best_rate: 0.8, avg_rate: 0.8 },
  ...overrides,
});

// Fixture: eval run summary
const makeRun = (overrides: Partial<EvalRunSummary> = {}): EvalRunSummary => ({
  _file: "eval_results_v1_20260807_120000.json",
  source: "optimization",
  timestamp: "2026-08-07T12:00:00Z",
  prompt_version: "v1",
  model_runs: [],
  trophy_score: { at_target: 0, best_rate: 0, avg_rate: 0 },
  ...overrides,
});

/** Fixture: optimization state. Supplies every required field so a test can
 * override only what it is exercising. */
const makeState = (overrides: Partial<OptimizationState> = {}): OptimizationState => ({
  status: "running",
  current_iteration: 0,
  iteration_cap: 10,
  iterations_per_run: 1,
  target_metric_scores: {},
  iterations: [],
  plateau_window: 3,
  plateau_threshold: 0.02,
  best_iteration: null,
  current_prompt_version: "v1",
  winning_model_id: null,
  error: null,
  ...overrides,
});

describe("optimization-evidence", () => {
  describe("survivingTerminalIteration", () => {
    it("returns the last iteration when a run whose _file equals its run_id is present", () => {
      const iteration = makeIteration({
        iteration: 2,
        run_id: "eval_results_v2_20260807_130000.json",
      });
      const state = makeState({
        status: "converged",
        iteration_cap: 10,
        plateau_window: 3,
        plateau_threshold: 0.02,
        current_iteration: 2,
        iterations: [makeIteration({ iteration: 1 }), iteration],
        error: null,
        target_metric_scores: {},
      });
      const runs: EvalRunSummary[] = [
        makeRun({ _file: "eval_results_v2_20260807_130000.json" }),
      ];

      const result = survivingTerminalIteration(state, runs);

      expect(result).not.toBeNull();
      expect(result?.iteration).toBe(2);
    });

    it("returns null when the last iteration's run is absent from the list", () => {
      const state = makeState({
        status: "converged",
        iteration_cap: 10,
        plateau_window: 3,
        plateau_threshold: 0.02,
        current_iteration: 1,
        iterations: [makeIteration({ run_id: "eval_results_v1_20260807_120000.json" })],
        error: null,
        target_metric_scores: {},
      });
      const runs: EvalRunSummary[] = [
        makeRun({ _file: "eval_results_baseline_20260807_110000.json", prompt_version: "baseline" }),
      ];

      const result = survivingTerminalIteration(state, runs);

      expect(result).toBeNull();
    });

    it("returns null for state === null and for a state with an empty iteration list", () => {
      const runs: EvalRunSummary[] = [makeRun()];

      expect(survivingTerminalIteration(null, runs)).toBeNull();

      const emptyState = makeState({
        status: "running",
        iteration_cap: 10,
        plateau_window: 3,
        plateau_threshold: 0.02,
        current_iteration: 0,
        iterations: [],
        error: null,
        target_metric_scores: {},
      });
      expect(survivingTerminalIteration(emptyState, runs)).toBeNull();
    });

    it("resolves the LAST iteration, not the first — given two iterations where only the FIRST has a surviving run file, returns null", () => {
      const state = makeState({
        status: "converged",
        iteration_cap: 10,
        plateau_window: 3,
        plateau_threshold: 0.02,
        current_iteration: 2,
        iterations: [
          makeIteration({ iteration: 1, run_id: "eval_results_v1_20260807_120000.json" }),
          makeIteration({ iteration: 2, run_id: "eval_results_v2_20260807_130000.json", prompt_version: "v2" }),
        ],
        error: null,
        target_metric_scores: {},
      });
      // Only the first iteration's run file is present
      const runs: EvalRunSummary[] = [
        makeRun({ _file: "eval_results_v1_20260807_120000.json" }),
      ];

      const result = survivingTerminalIteration(state, runs);

      // Should return null because the LAST iteration has no surviving run
      expect(result).toBeNull();
    });

    it("a bare-identifier run_id resolves to nothing, even when a run at the same prompt version exists", () => {
      // The resolver used to correlate a bare identifier against the newest
      // loop-sourced run at the same prompt version, to read iterations recorded
      // before run files were materialized. A run_id IS a filename now, so an
      // identifier that is not one has no run — and crucially, the terminal
      // verdict must NOT be backed by a run it merely guessed at.
      const state = makeState({
        status: "converged",
        iteration_cap: 10,
        plateau_window: 3,
        plateau_threshold: 0.02,
        current_iteration: 1,
        iterations: [makeIteration({
          iteration: 1,
          run_id: "bare-uuid-123",  // not a filename
          prompt_version: "v1",
        })],
        error: null,
        target_metric_scores: {},
      });
      const loopRun = makeRun({
        _file: "eval_results_v1_20260807_120000.json",
        source: "optimization",
        prompt_version: "v1",
      });
      expect(survivingTerminalIteration(state, [loopRun])).toBeNull();
    });

    it("the evidence-requirement map is keyed on the status union and marks exactly the two verdicts computed from iteration metrics as requiring evidence", () => {
      // Assert via the map's own keys so a future status forces a decision
      const keys = Object.keys(STATUS_NEEDS_ITERATION_EVIDENCE);
      expect(keys.length).toBeGreaterThan(0);

      // The two verdicts computed from metrics
      expect(STATUS_NEEDS_ITERATION_EVIDENCE.converged).toBe(true);
      expect(STATUS_NEEDS_ITERATION_EVIDENCE.plateau).toBe(true);

      // Everything else should NOT require evidence
      const needsEvidence = keys.filter(
        (k) => STATUS_NEEDS_ITERATION_EVIDENCE[k as keyof typeof STATUS_NEEDS_ITERATION_EVIDENCE],
      );
      expect(needsEvidence).toEqual(["converged", "plateau"]);
    });
  });

  describe("resolveIterationRunFile (imported by OptimizationRunDetail tests)", () => {
    it("returns null for empty run_id", () => {
      const iteration = makeIteration({ run_id: "" });
      const runs = [makeRun()];
      expect(resolveIterationRunFile(iteration, runs)).toBeNull();
    });

    it("returns exact match when _file equals run_id", () => {
      const iteration = makeIteration({ run_id: "eval_results_v1_20260807_120000.json" });
      const run = makeRun({ _file: "eval_results_v1_20260807_120000.json" });
      expect(resolveIterationRunFile(iteration, [run])).toBe(run._file);
    });

    it("returns null for a run_id that is not a filename", () => {
      // No prompt-version correlation: picking the newest matching run answered
      // a request for ONE iteration's results with a different run of the same
      // version, and needed a source guard to avoid stealing a manual run.
      const iteration = makeIteration({ run_id: "bare-uuid", prompt_version: "v1" });
      const runs = [
        makeRun({
          _file: "eval_results_v1_20260807_120000.json",
          source: "optimization", prompt_version: "v1",
        }),
        makeRun({
          _file: "eval_results_v1_20260807_110000.json",
          source: "optimization", prompt_version: "v1",
        }),
      ];
      expect(resolveIterationRunFile(iteration, runs)).toBeNull();
    });

    it("never claims a run whose filename it was not given", () => {
      // Previously guarded by a source check inside the correlation branch. With
      // the correlation gone the guarantee is structural: matching is by exact
      // filename, so a manual run at the same prompt version is unreachable.
      const iteration = makeIteration({
        run_id: "bare-uuid",
        prompt_version: "v1",
      });
      const manualRun = makeRun({
        _file: "eval_results_v1_20260807_120000.json",
        source: "manual",
        prompt_version: "v1",
      });
      expect(resolveIterationRunFile(iteration, [manualRun])).toBeNull();
    });
  });
  describe("hasAnySurvivingIterationRun", () => {
    /* This predicate answers "does ANY iteration still have a run file", which
     * is the question the eval dashboard's other-runs toggle needs. It is NOT
     * the same question as survivingTerminalIteration, which anchors on the
     * last iteration alone — see the divergence test at the end of this block. */

    it("returns false for null state (no loop record at all)", () => {
      // Collapses the old `state === null` branch: no record means no evidence,
      // so callers need only this one predicate rather than a second test.
      expect(hasAnySurvivingIterationRun(null, [makeRun()])).toBe(false);
    });

    it("returns false when the state has no iterations", () => {
      expect(hasAnySurvivingIterationRun(makeState({ iterations: [] }), [makeRun()])).toBe(false);
    });

    it("returns true when an iteration's run file is present", () => {
      const iteration = makeIteration({ run_id: "eval_results_v1_20260807_120000.json" });
      const state = makeState({ iterations: [iteration] });
      expect(hasAnySurvivingIterationRun(state, [makeRun()])).toBe(true);
    });

    it("returns false when every iteration's run file has been deleted", () => {
      // The stale-OPT-record case: a loop record survives but its evidence is
      // gone, so it cannot support a loop narrative.
      const state = makeState({
        iterations: [makeIteration({ iteration: 1 }), makeIteration({ iteration: 2 })],
      });
      expect(hasAnySurvivingIterationRun(state, [])).toBe(false);
    });

    it("returns true when only a NON-terminal iteration's run file survives", () => {
      /* The divergence that makes this a distinct predicate. The terminal
       * iteration's file is gone, so survivingTerminalIteration reports no
       * evidence — but a middle iteration's file is still on disk, so the
       * project does have loop evidence and its cards will render. Asserting
       * both predicates on ONE input pins them apart: a future reader cannot
       * substitute the terminal check here without turning this red. */
      const surviving = makeIteration({
        iteration: 1,
        run_id: "eval_results_v1_20260807_120000.json",
      });
      const deletedTerminal = makeIteration({
        iteration: 2,
        prompt_version: "v2",
        run_id: "eval_results_v2_deleted.json",
      });
      const state = makeState({ iterations: [surviving, deletedTerminal] });
      const runs = [makeRun()];

      expect(hasAnySurvivingIterationRun(state, runs)).toBe(true);
      expect(survivingTerminalIteration(state, runs)).toBeNull();
    });

    it("ignores iterations with an empty run_id (never had a run)", () => {
      // Mirrors resolveIterationRunFile's guard: an empty run_id must not be
      // correlated by prompt_version, so it is not evidence.
      const state = makeState({
        iterations: [makeIteration({ run_id: "", prompt_version: "v1" })],
      });
      expect(hasAnySurvivingIterationRun(state, [makeRun()])).toBe(false);
    });
  });
});
