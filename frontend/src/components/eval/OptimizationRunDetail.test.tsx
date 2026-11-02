// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Render matrix for OptimizationRunDetail over run-shape × loop-record × status.
 *
 * The panel's render gate is "runs exist", not "OPT record exists": a project can
 * have runs on disk and no loop record, and keying on the record would hide that
 * project's entire run history. The panel is null-safe (state can be null). This
 * file pins the null-safe behavior and the sources-from-criteria contract.
 *
 * DIVERGENCE FROM ActiveJobBar.test.tsx's ASSERTION CONVENTION (intentional):
 * That file asserts the weak property (something meaningful renders) to avoid
 * pinning decorative copy. This file does the OPPOSITE — it pins load-bearing
 * strings because the failure mode is not "wrong wording" but "absent feature".
 * Specifically:
 *   - PIN: "Iteration 3 of 4" and "Latest vs target" (presence AND absence).
 *   - PIN: "ExactMatch ≥ 91%" and absence of the OPT fixture's "55" (criteria provenance).
 *   - PIN: a card's prompt_version string (cards actually rendered).
 *   - DO NOT PIN: header word "Optimization", status label prose, "Best model:" phrasing.
 *   - PROHIBITED: /Optimization/i and similar — the panel header contains that word
 *     on every render, so such an assertion cannot fail.
 */
import { render, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { EvalRunSummary, HarnessConfig, OptimizationIteration, OptimizationState } from "@/lib/types";
import {
  OptimizationRunDetail,
  OPT_STATUS_LABELS,
  convergenceCriteriaFromHarness,
} from "./OptimizationRunDetail";
import { resolveIterationRunFile } from "@/lib/optimization-evidence";

// Derive status list from the exported map so a 13th union member enters the matrix automatically.
const ALL_STATUSES = Object.keys(OPT_STATUS_LABELS) as OptimizationState["status"][];

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Fixtures — divergent values make criteria-source assertions falsifiable.
// ---------------------------------------------------------------------------

/** Harness config fixture with targets 91%, max 12, plateau window 5, threshold 7%. */
const HARNESS_CONFIG: HarnessConfig = {
  models: ["test-model"],
  data_file: "cases.csv",
  ready: true, // backend-computed runnability
  judge_model_id: null,
  prompt_version: null,
  field_mappings: null,
  target_metric_scores: { ExactMatch: 0.91 },
  max_iterations: 12,
  plateau_window: 5,
  plateau_threshold: 0.07,
};

/** Optimization state fixture with targets 55%, max 4, current iteration 3.
 * Values deliberately differ from HARNESS_CONFIG so asserting "91" appears
 * and "55" does not proves criteria come from config, not state. */
const OPT_ITERATION: OptimizationIteration = {
  iteration: 1,
  run_id: "run-baseline.json",
  prompt_version: "baseline",
  timestamp: "2024-01-01T00:00:00Z",
  total_cases: 10,
  passed: 8,
  pass_rate: 0.75,
  per_model_pass_rates: { "test-model": 0.75 },
  per_model_metrics: { "test-model": { ExactMatch: 0.75 } },
  metrics_summary: { ExactMatch: 0.75 },
  trophy_score: { at_target: 1, best_rate: 0.75, avg_rate: 0.75 },
};

function makeOptState(
  status: OptimizationState["status"],
  iterations: OptimizationIteration[] = [OPT_ITERATION],
): OptimizationState {
  return {
    status,
    current_iteration: 3,
    iteration_cap: 4,
    iterations_per_run: 2,
    plateau_threshold: 0.02,
    plateau_window: 3,
    target_metric_scores: { ExactMatch: 0.55 },
    iterations,
    best_iteration: null,
    current_prompt_version: "baseline",
    winning_model_id: null,
    error: null,
  };
}

/** Loop run fixture matching OPT_ITERATION's run_id. Numeric fields are
 * benign (8/10 passed) so they don't leak into assertions that expect
 * iteration-sourced numbers only. */
const LOOP_RUN: EvalRunSummary = {
  _file: "run-baseline.json",
  prompt_version: "baseline",
  timestamp: "2024-01-01T00:00:00Z",
  source: "optimization",
  model_runs: [
    {
      test_run_id: "loop-run-1",
      model: "test-model",
      passed: 8,
      total: 10,
      per_metric: { ExactMatch: 0.8 },
      endpoint: "runtime",
      region: "us-east-1",
    },
  ],
  trophy_score: { at_target: 1, best_rate: 0.8, avg_rate: 0.8 },
};

const MANUAL_RUN: EvalRunSummary = {
  _file: "manual-run-v2.json",
  prompt_version: "v2",
  timestamp: "2024-01-02T00:00:00Z",
  source: "manual",
  model_runs: [
    {
      test_run_id: "run-1",
      model: "test-model",
      passed: 8,
      total: 10,
      per_metric: { ExactMatch: 0.8 },
      endpoint: "runtime",
      region: "us-east-1",
    },
  ],
  trophy_score: { at_target: 1, best_rate: 0.8, avg_rate: 0.8 },
};

// ---------------------------------------------------------------------------
// Breadth matrix: every combination renders without throwing.
// ---------------------------------------------------------------------------

describe("OptimizationRunDetail — breadth matrix", () => {
  const runShapes = [
    { name: "no runs", runs: [] },
    { name: "manual runs only", runs: [MANUAL_RUN] },
    { name: "loop runs only", runs: [LOOP_RUN] },
    { name: "loop + manual", runs: [LOOP_RUN, MANUAL_RUN] },
  ];

  const loopRecordAxis = [
    { name: "no loop record", state: null },
    ...ALL_STATUSES.map((status) => ({
      name: `status=${status}`,
      state: makeOptState(status),
    })),
  ];

  it.each(
    runShapes.flatMap((rs) =>
      loopRecordAxis.map((lr) => ({
        runShape: rs.name,
        loopRecord: lr.name,
        state: lr.state,
        runs: rs.runs,
      })),
    ),
  )(
    "renders: $runShape × $loopRecord",
    ({ state, runs }) => {
      const { container } = render(
        <MemoryRouter>
          <OptimizationRunDetail
            state={state}
            criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
            runs={runs}
          />
        </MemoryRouter>,
      );
      expect(container.textContent).toBeTruthy();
    },
  );
});

// ---------------------------------------------------------------------------
// Load-bearing cases: each pins a specific contract.
// ---------------------------------------------------------------------------

describe("OptimizationRunDetail — load-bearing cases", () => {
  it("no loop record, manual runs present → cards visible on first paint", () => {
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={null}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[MANUAL_RUN]}
        />
      </MemoryRouter>,
    );
    // Assert the manual run's prompt_version appears with no click / no interaction.
    // This is the reported bug: with state=null every card is manual, so an unchecked
    // default would hide all content.
    expect(container.textContent).toContain("v2");
  });

  it("no loop record → loop-progress lines absent, not zero-filled", () => {
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={null}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Capital-I "Iteration" occurs only in the progress line; badges use lowercase.
    expect(text).not.toContain("Iteration");
    // "Latest vs target" is the other loop-progress line.
    expect(text).not.toContain("Latest vs target");
    // The specific failure this forbids: fabricating "Iteration 0 of 0".
  });

  it("loop record present → both loop-progress literals appear", () => {
    const state = makeOptState("running");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[LOOP_RUN]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Must contain "Iteration 3 of 4" from the fixture (not hardcoded numbers).
    expect(text).toContain("Iteration 3 of 4");
    expect(text).toContain("Latest vs target");
  });

  it("criteria come from harness config, loop record present", () => {
    const state = makeOptState("converged");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[LOOP_RUN]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Assert harness config's 91% target appears (JSX &ge; renders as ≥).
    expect(text).toContain("ExactMatch ≥ 91%");
    // Assert the max-iterations criterion shows 12 (from config, not state's 4).
    expect(text).toContain("Max 12 iterations");
    // Assert the plateau badge shows 7%/5-iteration pair (from config).
    expect(text).toContain("Plateau < 7% over 5 iterations");
    // Assert the OPT fixture's 55 does NOT appear anywhere.
    expect(text).not.toContain("55");
  });

  it("criteria come from harness config, no loop record", () => {
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={null}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Criteria row does not depend on a loop existing.
    expect(text).toContain("ExactMatch ≥ 91%");
  });

  it("criteria === null → Convergence row omitted, no crash", () => {
    const state = makeOptState("running");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={null}
          runs={[LOOP_RUN]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The "Convergence:" label is absent.
    expect(text).not.toContain("Convergence:");
    // Cards still render (loop record present).
    expect(text).toContain("baseline");
    // This also exercises the empty-targets trophy path (Math.min() → Infinity).
    // The assertion is that it renders rather than throws.
  });

  it("max_iterations: null → Max badge omitted, other criteria present", () => {
    const configNoMax: HarnessConfig = {
      ...HARNESS_CONFIG,
      max_iterations: null,
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={null}
          criteria={convergenceCriteriaFromHarness(configNoMax)}
          runs={[]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Target and plateau badges still render.
    expect(text).toContain("ExactMatch ≥ 91%");
    expect(text).toContain("Plateau");
    // Max badge is omitted.
    expect(text).not.toContain("Max");
    // Do not accept a fabricated "10" (the loop's start-time default, not a config value).
    expect(text).not.toContain("10");
  });

  it("checkbox default is unchecked when loop record exists", () => {
    const state = makeOptState("running");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[LOOP_RUN, MANUAL_RUN]}
        />
      </MemoryRouter>,
    );
    const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(checkbox?.checked).toBe(false);
    // Manual run's prompt_version is NOT in the text on first paint.
    expect(container.textContent).not.toContain("v2");
  });

  // --- Existence gating on run-file list ---

  it("existence gate, positive: both iterations' run files present → both cards render", () => {
    // Two iterations with distinct prompt versions; both run files in the list.
    const iter1: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "run-v1.json",
      prompt_version: "v1",
    };
    const iter2: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 2,
      run_id: "run-v2.json",
      prompt_version: "v2",
    };
    const state = makeOptState("running", [iter1, iter2]);
    const run1: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v1.json",
      prompt_version: "v1",
    };
    const run2: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v2.json",
      prompt_version: "v2",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[run1, run2]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // Both prompt versions must appear when both run files exist.
    expect(text).toContain("v1");
    expect(text).toContain("v2");
  });

  it("existence gate, negative: only the first iteration's run file present → surviving card renders, absent does not", () => {
    // Two iterations, but only the first one's run file is in the list.
    // Use non-substring prompt versions so partial matches cannot pass by accident.
    const iter1: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "run-alpha.json",
      prompt_version: "alpha",
    };
    const iter2: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 2,
      run_id: "run-omega.json",
      prompt_version: "omega",
    };
    const state = makeOptState("running", [iter1, iter2]);
    const run1: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-alpha.json",
      prompt_version: "alpha",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[run1]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The surviving card's prompt version must appear.
    expect(text).toContain("alpha");
    // The deleted iteration's prompt version must NOT appear.
    expect(text).not.toContain("omega");
  });

  it("toggle independence: loop record exists, toggle unchecked by default, loop card renders on first paint", () => {
    // State non-null so the default is unchecked. The loop run file is present.
    const state = makeOptState("running");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[LOOP_RUN]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The loop card must render immediately, before any user interaction.
    expect(text).toContain("baseline");
  });

  it("toggle count: one loop run file and one unclaimed run → label count is 1, not 2", () => {
    // One iteration claiming LOOP_RUN, one manual run not claimed by any iteration.
    const state = makeOptState("running");
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[LOOP_RUN, MANUAL_RUN]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The label must show "1", not "2", because only MANUAL_RUN is unclaimed.
    expect(text).toContain("Show other runs (1)");
  });

  it("empty run_id: iteration with empty run_id → no card even when a run file matches prompt_version", () => {
    // Iteration with run_id="" and prompt_version "empty-pv", plus a run file
    // with the same prompt_version. The card must NOT appear because run_id is falsy.
    const iterEmpty: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "",
      prompt_version: "empty-pv",
    };
    const state = makeOptState("running", [iterEmpty]);
    const runMatchingPV: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-empty-pv.json",
      prompt_version: "empty-pv",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[runMatchingPV]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";

    /* Assert the claim itself rather than the absence of the prompt-version
     * string. The run IS expected on screen: no iteration claims it, so it is
     * unclaimed, and an unclaimed run is exactly what the "other runs" toggle
     * reveals. A text-absence assertion therefore fails for a reason that has
     * nothing to do with this test's subject — which is whether an ITERATION
     * card claimed the run.
     *
     * Two independent witnesses that none did:
     *
     * 1. The unclaimed count reads 1. ``claimedFiles`` drives that number, so
     *    had the iteration resolved to this file it would read 0.
     * 2. The one card on screen carries the previous-loop-run chip, which the
     *    component renders only for unclaimed cards. Iteration cards never do. */
    expect(text).toContain("Show other runs (1)");
    expect(
      container.querySelector('[aria-label="run from previous optimization loop"]'),
    ).not.toBeNull();
  });

  it("a bare run_id yields no card, even with a run at the same prompt version", () => {
    // An iteration's run_id IS its run filename. A bare identifier used to be
    // correlated to the newest loop run at the same prompt version, so a card
    // could be rendered from a run that was merely a plausible match.
    const iterBare: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "abc-123-uuid",
      prompt_version: "vbare",
    };
    const state = makeOptState("running", [iterBare]);
    const sameVersionRun: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "eval_results_vbare_20260101_000000.json",
      prompt_version: "vbare",
    };

    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[sameVersionRun]}
        />
      </MemoryRouter>,
    );
    // The run stays UNCLAIMED — it shows up in the "other runs" set rather than
    // as this iteration's card. Asserted this way rather than on the absence of
    // the version string, which also appears in the state's best-model summary
    // and made the assertion pass for the wrong reason.
    expect(container.textContent).toContain("Show other runs (1)");
    expect(
      container.querySelector('a[href*="eval_results_vbare_20260101_000000.json"]'),
    ).toBeNull();
  });

  it("matches by exact filename, so a manual run at the same version is unreachable", () => {
    // This used to need a source guard INSIDE a prompt-version correlation: the
    // lexicographically greatest file wins, and a user's manual run can sit at
    // the same version as a loop iteration. With correlation gone the guarantee
    // is structural rather than guarded.
    const iterBare: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "abc-123-uuid",
      prompt_version: "vshared",
    };
    const manualRun: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "eval_results_vshared_20991231_235959.json",
      prompt_version: "vshared",
      source: "manual",
    };
    const loopRun: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "eval_results_vshared_20260101_000000.json",
      prompt_version: "vshared",
      source: "optimization",
    };

    expect(resolveIterationRunFile(iterBare, [manualRun, loopRun])).toBeNull();

    // And an iteration whose run_id IS a filename resolves to exactly that file.
    const iterNamed: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: loopRun._file,
      prompt_version: "vshared",
    };
    expect(resolveIterationRunFile(iterNamed, [manualRun, loopRun])).toBe(loopRun._file);
  });

  it("trophy output unchanged: all run files present, unequal scores → trophy lands on the same card", () => {
    // Two iterations with SERVER-computed trophy scores (the wire is the
    // only scoring source). The higher score sits on the EARLIER card (v1),
    // so this discriminates score-ordering from the later-wins tiebreak —
    // with scores absent (both UNSCORED) later-wins would crown v2 and this
    // test would pass vacuously for the old assertion.
    const iter1: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "run-v1.json",
      prompt_version: "v1",
      pass_rate: 0.9,
      per_model_pass_rates: { "test-model": 0.9 },
      trophy_score: { at_target: 1, best_rate: 0.9, avg_rate: 0.9 },
    };
    const iter2: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 2,
      run_id: "run-v2.json",
      prompt_version: "v2",
      pass_rate: 0.6,
      per_model_pass_rates: { "test-model": 0.6 },
      trophy_score: { at_target: 0, best_rate: 0.6, avg_rate: 0.6 },
    };
    const state = makeOptState("running", [iter1, iter2]);
    const run1: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v1.json",
      prompt_version: "v1",
    };
    const run2: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v2.json",
      prompt_version: "v2",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[run1, run2]}
        />
      </MemoryRouter>,
    );
    // The trophy must land on the card with the higher SERVER score (v1),
    // NOT the later card — that is the discrimination: an unscored render
    // (both cards UNSCORED) falls through to later-wins and crowns v2.
    const cards = container.querySelectorAll('button[aria-pressed]');
    let trophyCardText = "";
    for (const card of cards) {
      if (card.className.includes("border-success")) {
        trophyCardText = card.textContent ?? "";
        break;
      }
    }
    expect(trophyCardText).toContain("v1");
    expect(trophyCardText).toContain("90%");
  });

  it("terminal run deleted: converged verdict is withheld, criteria row and surviving card remain", () => {
    // Two iterations; the TERMINAL one (v2) has no run file. Convergence is a
    // property of that run, so with its evidence gone the verdict goes too —
    // badge and the row's green highlight together, never one without the other.
    const iter1: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "run-v1.json",
      prompt_version: "v1",
    };
    const iter2: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 2,
      run_id: "run-v2.json",
      prompt_version: "v2",
    };
    const state = makeOptState("converged", [iter1, iter2]);
    const run1: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v1.json",
      prompt_version: "v1",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[run1]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The verdict is withheld — its evidence was deleted.
    expect(text).not.toContain("Target Reached");
    // Latest-vs-target is withheld for the same reason.
    expect(text).not.toContain("Latest vs target");
    // The criteria row still renders: it comes from harness config, not from
    // any run, so a delete cannot invalidate it.
    expect(text).toContain("ExactMatch \u2265 91%");
    // ...but without the converged highlight, which would re-assert the
    // withheld verdict by other means.
    expect(container.querySelector(".bg-success\\/10")).toBeNull();
    // The surviving card still renders.
    expect(text).toContain("v1");
  });

  it("one run file missing → Iteration N of M still reflects state.current_iteration", () => {
    // Two iterations, one run file missing. The "Iteration N of M" line must
    // still reflect the state's current_iteration count, not the card count.
    const iter1: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 1,
      run_id: "run-v1.json",
      prompt_version: "v1",
    };
    const iter2: OptimizationIteration = {
      ...OPT_ITERATION,
      iteration: 2,
      run_id: "run-v2.json",
      prompt_version: "v2",
    };
    const state = makeOptState("running", [iter1, iter2]);
    const run1: EvalRunSummary = {
      ...LOOP_RUN,
      _file: "run-v1.json",
      prompt_version: "v1",
    };
    const { container } = render(
      <MemoryRouter>
        <OptimizationRunDetail
          state={state}
          criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
          runs={[run1]}
        />
      </MemoryRouter>,
    );
    const text = container.textContent ?? "";
    // The "Iteration N of M" line must show the state's current_iteration (3),
    // not the card count (1).
    expect(text).toContain("Iteration 3 of 4");
  });
  /* The other-runs default is derived from surviving run evidence, not from the
   * mere existence of an OPT record. A loop record whose run files are gone
   * cannot support a loop narrative — the panel already withholds its verdict
   * for that reason — so it must not also hide the project's only run behind an
   * unchecked box. These three cases are the contract. */
  describe("other-runs default derives from surviving run evidence", () => {
    it("stale OPT record (no surviving iteration runs) → box CHECKED, the surviving run is visible", () => {
      // Loop record points at a deleted file; only a manual baseline survives.
      const staleIteration: OptimizationIteration = {
        ...OPT_ITERATION,
        iteration: 1,
        run_id: "eval_results_v1_deleted.json",
        prompt_version: "v1",
      };
      const baselineRun: EvalRunSummary = {
        ...MANUAL_RUN,
        _file: "eval_results_baseline.json",
        prompt_version: "baseline-only",
      };
      const { container } = render(
        <MemoryRouter>
          <OptimizationRunDetail
            state={makeOptState("max_iterations", [staleIteration])}
            criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
            runs={[baselineRun]}
          />
        </MemoryRouter>,
      );
      const box = container.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
      expect(box?.checked).toBe(true);
      // The point of defaulting checked: the run is actually on screen.
      expect(container.textContent ?? "").toContain("baseline-only");
    });

    it("live loop with a surviving iteration run → box UNCHECKED, loop narrative stays clean", () => {
      const liveIteration: OptimizationIteration = {
        ...OPT_ITERATION,
        iteration: 1,
        run_id: LOOP_RUN._file,
        prompt_version: LOOP_RUN.prompt_version,
      };
      const { container } = render(
        <MemoryRouter>
          <OptimizationRunDetail
            state={makeOptState("running", [liveIteration])}
            criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
            runs={[LOOP_RUN, MANUAL_RUN]}
          />
        </MemoryRouter>,
      );
      const box = container.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
      expect(box?.checked).toBe(false);
      // The unclaimed manual run is hidden while unchecked.
      expect(container.textContent ?? "").not.toContain(MANUAL_RUN.prompt_version);
    });

    it("no OPT record at all → box CHECKED (no loop record means no loop evidence)", () => {
      const { container } = render(
        <MemoryRouter>
          <OptimizationRunDetail
            state={null}
            criteria={convergenceCriteriaFromHarness(HARNESS_CONFIG)}
            runs={[MANUAL_RUN]}
          />
        </MemoryRouter>,
      );
      const box = container.querySelector('input[type="checkbox"]') as HTMLInputElement | null;
      expect(box?.checked).toBe(true);
      expect(container.textContent ?? "").toContain(MANUAL_RUN.prompt_version);
    });
  });
});
