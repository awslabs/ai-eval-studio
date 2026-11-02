// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Render matrix for ActiveJobBar over optStatus × evalStatus.
 *
 * The bar is the surface where "the backend wrote a status the UI cannot render"
 * shows up, and it has drifted three times. The concrete failure was `"failed"`
 * missing from the status union: a crashed optimization matched no branch, fell
 * through to the empty default, and rendered as "Ready to optimize" — the user was
 * told everything was fine while the run was dead.
 *
 * `tsc` now catches a MISSING MAP ENTRY (the status maps are keyed on the union),
 * but it cannot catch a missing render BRANCH — `optStatus === "failed"` simply
 * being absent from an if/else chain is well-typed. This matrix is what closes
 * that gap: every status the backend can write must produce a non-empty bar.
 *
 * Deliberately asserts the WEAK property (something meaningful renders) rather
 * than exact copy. Pinning strings would make every wording tweak a test failure
 * and teach people to update tests reflexively; the invariant that actually
 * matters is "no backend status renders as silence or as a false-positive
 * 'Ready'".
 */
import { render, cleanup, waitFor, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EvalJob, HarnessConfig, OptimizationState } from "@/lib/types";

// ---------------------------------------------------------------------------
// Mocks. The bar pulls from three hooks and the REST client; none is the subject
// of this test, so each is stubbed to a controllable value.
// ---------------------------------------------------------------------------

const mockOptState = vi.fn();
const mockEvalStream = vi.fn();
/** Controls the run-file list backing the terminal-evidence gate. */
const mockEvalRuns = vi.fn(() => ({ runs: [] as unknown[] }));

vi.mock("@/components/eval/OptimizationStateProvider", () => ({
  useOptimizationState: () => mockOptState(),
}));

vi.mock("@/hooks/useEvalJobStream", () => ({
  useEvalJobStream: () => mockEvalStream(),
}));

vi.mock("@/hooks/useSyntheticBatches", () => ({
  useSyntheticBatches: () => ({ data: [] }),
}));

vi.mock("@/hooks/useEvalRuns", () => ({
  useEvalRuns: () => mockEvalRuns(),
}));

/** Bus handlers the bar subscribed, keyed `resource/type`, so a test can fire one. */
const busHandlers = new Map<string, () => void>();

vi.mock("@/hooks/useBackendEvents", () => ({
  useProjectEvent: (match: { resource: string; type?: string }, handler: () => void) => {
    busHandlers.set(`${match.resource}/${match.type}`, handler);
  },
}));

vi.mock("@/lib/api", () => ({
  api: {
    // Resolve a harness so the bar's "no harness AND no opt state" hide path is
    // not what we accidentally end up testing.
    getHarnessConfig: vi.fn().mockResolvedValue({
      models: ["m1"],
      data_file: "cases.csv",
      field_mappings: { input: "q", expected_output: "a" },
      // action buttons gate on backend-computed readiness.
      ready: true,
    }),
    cancelOptimization: vi.fn(),
    startOptimization: vi.fn(),
    resumeOptimization: vi.fn(),
    cancelEvalJob: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message: string, status = 500) {
      super(message);
      this.status = status;
    }
  },
}));

// Imported after the mocks so the module graph picks them up.
const { ActiveJobBar } = await import("@/components/eval/ActiveJobBar");
const { api } = await import("@/lib/api");

/** Every status the backend can write for an optimization run. */
const OPT_STATUSES: OptimizationState["status"][] = [
  "starting",
  "running",
  "stopping",
  "converged",
  "plateau",
  "max_iterations",
  "optimizer_exhausted",
  "optimizer_stalled",
  "judge_unreliable",
  "generation_unreliable",
  "failed",
  "cancelled",
];

/** Every status the backend can write for an eval job. */
const EVAL_STATUSES: EvalJob["status"][] = [
  "pending",
  "running",
  "completed",
  "failed",
  "cancelled",
  "awaiting_user",
];

function optState(status: OptimizationState["status"]): Partial<OptimizationState> {
  return {
    status,
    current_iteration: 2,
    max_iterations: 5,
    iterations_per_run: 5,
    target_metric_scores: { ExactMatch: 0.9 },
    iterations: [],
    best_iteration: null,
    error:
      status === "failed"
        ? { code: "optimization.crashed", message: "Loop hit an unexpected error" }
        : null,
  } as Partial<OptimizationState>;
}

beforeEach(() => {
  mockOptState.mockReturnValue({
    state: null,
    activeEval: null,
    isLoaded: true,
    refetch: vi.fn(),
    setOptimistic: vi.fn(),
  });
  mockEvalStream.mockReturnValue({
    progress: null,
    jobId: null,
    cancelling: false,
    handleCancel: vi.fn(),
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

/** Visible text of the rendered bar, collapsed to a single string. */
async function renderBar(): Promise<string> {
  const { container } = render(<ActiveJobBar projectId="proj-1" />);
  // The harness fetch gates the first paint (`if (!harnessChecked) return null`),
  // so wait for the bar to mount before reading it. Waits for AT LEAST ONE
  // button rather than a unique one: the resumable states legitimately render
  // two (Resume + Start New), and findByRole throws on multiple matches.
  await waitFor(() => {
    expect(container.querySelector("button")).not.toBeNull();
  });
  return container.textContent ?? "";
}

describe("optimization status matrix", () => {
  it.each(OPT_STATUSES)("renders a meaningful bar for optStatus=%s", async (status) => {
    mockOptState.mockReturnValue({
      state: optState(status),
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });

    const text = await renderBar();

    expect(text.trim().length).toBeGreaterThan(0);

    // The load-bearing assertion.
    //
    // The empty fall-through branch renders the literal string "Ready to
    // optimize" plus a "Start Optimization" button — which is what a crashed
    // run showed while `failed` had no branch of its own: a dead run telling
    // the user everything is fine.
    //
    // Two phrasings that look right but do NOT catch it:
    //   - /Optimization/i — the idle BUTTON contains "Optimization".
    //   - strip-known-button-labels then assert non-empty — the component
    //     inlines a <style> block for the scoring animation, and @keyframes text
    //     leaks into textContent, so the remainder is never empty.
    //
    // So assert against the idle copy directly. Every status in this matrix
    // describes a run that EXISTED, so none of them may render as "ready".
    expect(
      text,
      `optStatus=${status} rendered the IDLE bar. That is the empty fall-through ` +
        `branch: a run in state "${status}" is reported to the user as ready to ` +
        `start, which is how a crashed optimization looked healthy.`,
    ).not.toMatch(/Ready to optimize/i);
  });

  it("a crashed run is NOT rendered as ready-to-start", async () => {
    // The exact reported bug, pinned on its own so a regression names itself
    // instead of hiding inside the parametrized case.
    mockOptState.mockReturnValue({
      state: optState("failed"),
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });

    const text = await renderBar();

    expect(text).toMatch(/fail/i);
    // It must offer Start New (a failed run is not resumable — the backend's
    // resume endpoint rejects status "failed"), never Resume.
    expect(text).toContain("Start New");
    expect(text).not.toContain("Resume");
  });

  it("a resumable pause offers Resume", async () => {
    mockOptState.mockReturnValue({
      state: optState("plateau"),
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    const text = await renderBar();
    expect(text).toContain("Resume");
    expect(text).toMatch(/paused/i);
  });

  it("judge_unreliable explains itself rather than showing a bare pause", async () => {
    // These two statuses were missing from the detail renderer's maps entirely,
    // so they rendered as unstyled, unlabelled badges.
    mockOptState.mockReturnValue({
      state: optState("judge_unreliable"),
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    const text = await renderBar();
    expect(text).toMatch(/judge/i);
  });
});

describe("eval status matrix", () => {
  it.each(EVAL_STATUSES)("does not crash for evalStatus=%s", async (status) => {
    // awaiting_user is the member that was missing from EvalJob["status"]. A
    // status the union does not know about reaches the renderer anyway at
    // runtime — types are erased — so the bar must tolerate every one.
    mockEvalStream.mockReturnValue({
      progress: {
        status,
        phase: status === "running" ? "generating" : "pending",
        modelsTotal: 2,
        models: [
          {
            model: "m1",
            phase: "generating" as const,
            generated: 3,
            casesTotal: 10,
            scored: false,
            scoredCount: 0,
            passed: 0,
            workers: 5,
          },
        ],
        error: null,
      },
      jobId: "job-1",
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const text = await renderBar();
    expect(text.trim().length).toBeGreaterThan(0);
  });

  it("a running eval shows progress and a Stop affordance", async () => {
    mockEvalStream.mockReturnValue({
      progress: {
        status: "running" as const,
        phase: "generating" as const,
        modelsTotal: 2,
        models: [
          {
            model: "m1",
            phase: "generating" as const,
            generated: 3,
            casesTotal: 10,
            scored: false,
            scoredCount: 0,
            passed: 0,
            workers: 5,
          },
        ],
        error: null,
      },
      jobId: "job-1",
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const text = await renderBar();
    expect(text).toMatch(/Evaluation running/i);
    expect(text).toContain("Stop");
  });

  it("a cancelled eval does not leave a live Stop button", async () => {
    // The sticky-Stop-button bug: a terminal eval that still rendered as running
    // with an active Stop control.
    mockEvalStream.mockReturnValue({
      progress: {
        status: "cancelled" as const,
        phase: "cancelled" as const,
        modelsTotal: 2,
        models: [],
        error: null,
      },
      jobId: "job-1",
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const text = await renderBar();
    expect(text).not.toMatch(/Evaluation running/i);
  });
});

describe("combined optStatus × evalStatus", () => {
  // The full cross-product is the point: the bar derives ONE status line from two
  // independent sources, and the precedence between them is where the
  // "optimization running + eval terminal" and "optimization paused + eval
  // running" combinations went wrong before.
  const combos = OPT_STATUSES.flatMap((o) => EVAL_STATUSES.map((e) => [o, e] as const));

  it.each(combos)("renders for optStatus=%s evalStatus=%s", async (o, e) => {
    mockOptState.mockReturnValue({
      state: optState(o),
      activeEval: { job_id: "job-1", status: e, models_total: 1, models: [] },
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    mockEvalStream.mockReturnValue({
      progress: {
        status: e,
        phase: e === "running" ? "generating" : "pending",
        modelsTotal: 1,
        models: [],
        error: null,
      },
      jobId: "job-1",
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const text = await renderBar();
    expect(text.trim().length).toBeGreaterThan(0);
  });
});

describe("converged verdict requires surviving run evidence", () => {
  /** Converged state whose terminal iteration points at a specific run file. */
  const convergedWithTerminalRun = () => ({
    state: {
      ...optState("converged"),
      iterations: [
        {
          iteration: 2,
          run_id: "eval_results_v2.json",
          prompt_version: "v2",
          timestamp: "2026-08-07T12:00:00Z",
          total_cases: 10,
          passed: 9,
          pass_rate: 0.9,
          per_model_pass_rates: { m1: 0.9 },
          per_model_metrics: { m1: { ExactMatch: 0.9 } },
          metrics_summary: { ExactMatch: 0.9 },
        },
      ],
    },
    activeEval: null,
    isLoaded: true,
    refetch: vi.fn(),
    setOptimistic: vi.fn(),
  });

  beforeEach(() => {
    mockOptState.mockReturnValue(convergedWithTerminalRun());
    mockEvalStream.mockReturnValue({
      progress: { status: "idle", phase: "pending", modelsTotal: 0, models: [], error: null },
      jobId: null,
      cancelling: false,
      handleCancel: vi.fn(),
    });
  });

  it("claims convergence while the terminal run file is present", async () => {
    mockEvalRuns.mockReturnValue({
      runs: [
        {
          _file: "eval_results_v2.json",
          prompt_version: "v2",
          timestamp: "2026-08-07T12:00:00Z",
          source: "optimization",
          model_runs: [],
        },
      ],
    });
    const text = await renderBar();
    expect(text).toContain("converged");
  });

  it("states the iteration count instead of the verdict once that run is deleted", async () => {
    mockEvalRuns.mockReturnValue({ runs: [] });
    const text = await renderBar();
    // The verdict is withheld — its evidence is gone.
    expect(text).not.toContain("converged");
    // But the bar must NOT fall through to idle copy: the loop did run.
    expect(text).not.toContain("Ready to optimize");
    expect(text).toContain("ran 2 iterations");
  });

  it("says \"1 iteration\", not \"1 iterations\", when exactly one ran", () => {
    /* The count and its noun must agree. This is the singular boundary of the
     * same status line the test above pins at 2 — the only count where the
     * default plural is wrong. */
    const singleIteration = convergedWithTerminalRun();
    singleIteration.state.current_iteration = 1;
    singleIteration.state.iterations = [singleIteration.state.iterations[0]];
    mockOptState.mockReturnValue(singleIteration);
    mockEvalRuns.mockReturnValue({ runs: [] });

    return renderBar().then((text) => {
      expect(text).toContain("ran 1 iteration");
      expect(text).not.toContain("ran 1 iterations");
    });
  });
});

describe("plateau verdict requires surviving run evidence", () => {
  /** Plateau state whose terminal iteration points at a specific run file. */
  const plateauWithTerminalRun = () => ({
    state: {
      ...optState("plateau"),
      iterations: [
        {
          iteration: 2,
          run_id: "eval_results_v2.json",
          prompt_version: "v2",
          timestamp: "2026-08-07T12:00:00Z",
          total_cases: 10,
          passed: 7,
          pass_rate: 0.7,
          per_model_pass_rates: { m1: 0.7 },
          per_model_metrics: { m1: { ExactMatch: 0.7 } },
          metrics_summary: { ExactMatch: 0.7 },
        },
      ],
    },
    activeEval: null,
    isLoaded: true,
    refetch: vi.fn(),
    setOptimistic: vi.fn(),
  });

  beforeEach(() => {
    mockOptState.mockReturnValue(plateauWithTerminalRun());
    mockEvalStream.mockReturnValue({
      progress: { status: "idle", phase: "pending", modelsTotal: 0, models: [], error: null },
      jobId: null,
      cancelling: false,
      handleCancel: vi.fn(),
    });
  });

  it("names the plateau while the terminal run file is present", async () => {
    mockEvalRuns.mockReturnValue({
      runs: [
        {
          _file: "eval_results_v2.json",
          prompt_version: "v2",
          timestamp: "2026-08-07T12:00:00Z",
          source: "optimization",
          model_runs: [],
        },
      ],
    });
    const text = await renderBar();
    expect(text).toContain("(plateau)");
  });

  it("drops the plateau claim once that run is deleted, keeping the paused line", async () => {
    mockEvalRuns.mockReturnValue({ runs: [] });
    const text = await renderBar();
    // The measured verdict is withheld...
    expect(text).not.toContain("(plateau)");
    // ...but the loop still paused, and that is still true.
    expect(text).toContain("paused at iteration");
    expect(text).not.toContain("Ready to optimize");
  });

  it("keeps non-measurement stop reasons when the terminal run is deleted", async () => {
    // "optimizer exhausted" describes how the loop stopped, not something
    // measured from a run — a deleted artifact must not suppress it.
    mockOptState.mockReturnValue({
      ...plateauWithTerminalRun(),
      state: { ...optState("optimizer_exhausted"), iterations: [] },
    });
    mockEvalRuns.mockReturnValue({ runs: [] });
    const text = await renderBar();
    expect(text).toContain("optimizer exhausted");
  });
});

/**
 * Standalone eval on a project whose harness arrived after mount.
 *
 * The bar's visibility decision is made above the code that knows an eval is
 * running, and its harness probe has exactly one trigger. Without the
 * standalone-eval case in both, a fresh project opened before the agent
 * scaffolds will hide a running eval for its entire duration.
 */
describe("standalone eval on a project whose harness arrived after mount", () => {
  // Active eval slice fixture — passed untyped through mockOptState
  const activeEvalSlice = {
    job_id: "job-9",
    status: "running",
    models_total: 1,
    models: [],
  };

  // Running progress fixture with one model row
  const runningProgress = {
    status: "running" as const,
    phase: "generating" as const,
    modelsTotal: 1,
    models: [
      {
        model: "m1",
        phase: "generating" as const,
        generated: 3,
        casesTotal: 10,
        scored: false,
        scoredCount: 0,
        passed: 0,
        workers: 5,
      },
    ],
    error: null,
  };

  beforeEach(() => {
    // "No harness yet" is a 200-null, not an error — this is the exact
    // state of a project opened before the agent scaffolds, and it is the
    // one state the rest of this file never exercises.
    vi.mocked(api.getHarnessConfig).mockResolvedValue(null);
  });

  afterEach(() => {
    // The global afterEach clears call records but keeps implementations,
    // so an un-restored null would silently re-point every later test at
    // the hide path. The cast is deliberate: the bar reads only `ready` and
    // `data_file` while the real type has 16 required fields.
    vi.mocked(api.getHarnessConfig).mockResolvedValue({
      ready: true,
      data_file: "cases.csv",
    } as unknown as HarnessConfig);
  });

  /**
   * Render and wait for the harness probe to settle.
   *
   * The component hides on `!harnessChecked` too, so without a settled probe
   * a null render is indistinguishable from "still deciding". Note that the
   * visibility test and the hide pin share this helper, so an insufficient
   * flush would fail the visibility test rather than let the hide pin pass
   * vacuously.
   */
  async function renderSettled() {
    const result = render(<ActiveJobBar projectId="proj-1" />);
    await waitFor(() =>
      expect(vi.mocked(api.getHarnessConfig)).toHaveBeenCalled()
    );
    await act(async () => {});
    return result;
  }

  it("renders eval progress when the mount-time harness probe found nothing", async () => {
    mockOptState.mockReturnValue({
      state: null,
      activeEval: activeEvalSlice,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    mockEvalStream.mockReturnValue({
      progress: runningProgress,
      jobId: "job-9",
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const { container } = await renderSettled();
    const text = container.textContent ?? "";

    // Without the widened guard, a multi-minute eval reports as an empty
    // page until the user happens to refresh.
    expect(text, "bar should render eval-running label").toMatch(/Evaluation running/i);
    expect(text, "bar should show per-model generate counter").toContain("3/10");
    expect(text, "bar should show Stop affordance").toContain("Stop");
  });

  it("re-probes the harness when a standalone eval appears", async () => {
    // Start with no eval
    mockOptState.mockReturnValue({
      state: null,
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    mockEvalStream.mockReturnValue({
      progress: null,
      jobId: null,
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const { rerender } = await renderSettled();
    expect(vi.mocked(api.getHarnessConfig)).toHaveBeenCalledTimes(1);

    // Now an eval appears
    mockOptState.mockReturnValue({
      state: null,
      activeEval: activeEvalSlice,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });

    rerender(<ActiveJobBar projectId="proj-1" />);

    // Exactly two: the appearing eval triggers a re-probe AND it does not
    // turn into a fetch per render.
    await waitFor(() =>
      expect(vi.mocked(api.getHarnessConfig)).toHaveBeenCalledTimes(2)
    );
  });

  it("renders nothing when there is no harness and nothing is running", async () => {
    mockOptState.mockReturnValue({
      state: null,
      activeEval: null,
      isLoaded: true,
      refetch: vi.fn(),
      setOptimistic: vi.fn(),
    });
    mockEvalStream.mockReturnValue({
      progress: null,
      jobId: null,
      cancelling: false,
      handleCancel: vi.fn(),
    });

    const { container } = await renderSettled();

    // The Start branch needs harness config for the data-source picker,
    // so an empty project must stay silent — widening the guard must not
    // light up a bar with nothing to say.
    expect(container.firstChild).toBeNull();
  });
});

describe("harness built in chat after mount", () => {
  afterEach(() => {
    // Restore the ready harness the rest of the file relies on.
    vi.mocked(api.getHarnessConfig).mockResolvedValue({
      ready: true,
      data_file: "cases.csv",
    } as unknown as HarnessConfig);
  });

  /** End an agent turn the way the WS bus would deliver it. */
  async function endTurn() {
    await act(async () => {
      busHandlers.get("agent_turn/completed")?.();
    });
  }

  it("offers Start Optimization once a turn ends with the harness ready", async () => {
    // Nothing running, no harness yet: a fresh project before the chat builds it.
    vi.mocked(api.getHarnessConfig).mockResolvedValue(null);
    const { container } = render(<ActiveJobBar projectId="proj-1" />);
    await waitFor(() => expect(vi.mocked(api.getHarnessConfig)).toHaveBeenCalledTimes(1));
    await act(async () => {});
    expect(container.firstChild).toBeNull();

    // The agent scaffolded the harness but has not written the baseline
    // prompt yet, so the button must stay hidden.
    vi.mocked(api.getHarnessConfig).mockResolvedValue({
      ready: false,
      data_file: "cases.csv",
    } as unknown as HarnessConfig);
    await endTurn();
    expect(vi.mocked(api.getHarnessConfig)).toHaveBeenCalledTimes(2);
    expect(container.textContent ?? "").not.toContain("Start Optimization");

    // A later turn writes the prompt; the backend now reports ready.
    vi.mocked(api.getHarnessConfig).mockResolvedValue({
      ready: true,
      data_file: "cases.csv",
    } as unknown as HarnessConfig);
    await endTurn();
    await waitFor(() => expect(container.textContent ?? "").toContain("Start Optimization"));
  });
});
