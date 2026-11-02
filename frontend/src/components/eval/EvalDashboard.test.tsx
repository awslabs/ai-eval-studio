// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * EvalDashboard error-lifetime and render-gate tests.
 *
 * Verifies that run-detail fetch errors have a bounded lifetime keyed to the
 * selection that produced them, and that list vs detail errors have distinct
 * blocking behavior.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";
import { EvalDashboard } from "./EvalDashboard";
import type { EvalRunSummary, ModelRun } from "@/lib/types";

afterEach(() => {
  cleanup();
});

// Mock heavy deps
vi.mock("@monaco-editor/react", () => ({ default: () => null }));
vi.mock("./AnalyticsPanel", () => ({
  AnalyticsPanel: () => null,
  SummaryChartsRow: () => null,
}));
vi.mock("./ResultsTable", () => ({ ResultsTable: () => null }));
vi.mock("./FilterBar", () => ({ FilterBar: () => null }));
vi.mock("./compare", () => ({ ComparePanel: () => null }));
vi.mock("./OptimizationRunDetail", () => ({
  OptimizationRunDetail: () => null,
  convergenceCriteriaFromHarness: () => null,
}));

// Mock API
const mockListEvalRuns = vi.fn<(projectId: string) => Promise<EvalRunSummary[]>>();
const mockGetEvalRun = vi.fn<(projectId: string, file: string) => Promise<ModelRun[]>>();

vi.mock("@/lib/api", () => ({
  api: {
    listEvalRuns: (projectId: string) => mockListEvalRuns(projectId),
    getEvalRun: (projectId: string, file: string) => mockGetEvalRun(projectId, file),
  },
}));

// Mock optimization state provider
const mockOptState = { state: null, isLoaded: true };
vi.mock("./OptimizationStateProvider", () => ({
  useOptimizationState: () => mockOptState,
}));

// Mock harness config
vi.mock("@/hooks/useHarnessConfig", () => ({
  useHarnessConfig: () => ({ config: null }),
}));

// Mock theme provider
vi.mock("@/lib/theme-provider", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

// Mock backend events — capture handlers so tests can fire them
const mockProjectEventHandlers = new Map<string, (event: any) => void>();
/** Captures the replay-gap callback so a test can fire the signal. */
let mockReplayGapCallback: (() => void) | null = null;

vi.mock("@/hooks/useBackendEvents", () => ({
  useProjectEvent: (filter: { resource: string; type: string }, handler: (event: any) => void) => {
    const key = `${filter.resource}/${filter.type}`;
    mockProjectEventHandlers.set(key, handler);
  },
  useCursorGapHandler: (_projectId: string | null, onGap: () => void) => {
    mockReplayGapCallback = onGap;
  },
}));

// Stub UI components to passthrough for simpler assertions
vi.mock("@/components/ui/select", () => ({
  Select: ({ children }: any) => (
    <div data-testid="select">
      {children}
    </div>
  ),
  SelectTrigger: ({ children }: any) => <div>{children}</div>,
  SelectValue: ({ children }: any) => <div>{children}</div>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children }: any) => (
    <div data-testid="select-item">
      {children}
    </div>
  ),
}));

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: any) => <div>{children}</div>,
  TabsList: ({ children }: any) => <div>{children}</div>,
  TabsTrigger: ({ children }: any) => <div>{children}</div>,
  TabsContent: ({ children }: any) => <div>{children}</div>,
}));

// Helper to create a minimal run summary
function makeRun(file: string, version = "baseline"): EvalRunSummary {
  return {
    _file: file,
    prompt_version: version,
    timestamp: "2026-08-07T00:00:00Z",
    source: "manual",
    model_runs: [
      {
        model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
        test_run_id: "test-run",
        total: 10,
        passed: 8,
        per_metric: {},
        endpoint: "runtime",
        region: "us-east-1",
      },
    ],
    trophy_score: { at_target: 0, best_rate: 0.8, avg_rate: 0.8 },
  };
}

// Helper to create minimal model run data
function makeModelRun(_file: string): ModelRun {
  return {
    test_run_id: "test-run",
    hyperparameters: {
      model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
      prompt_version: "baseline",
      endpoint: "runtime",
      region: "us-east-1",
      temperature: 0.7,
      max_tokens: 1000,
    },
    timestamp: "2026-08-07T00:00:00Z",
    results: [],
    generation_cost_usd: 0.02,
    judge_cost_usd: 0.01,
  };
}

// Deferred promise helper for deterministic ordering
function createDeferred<T>() {
  let resolve: (value: T) => void;
  let reject: (reason: any) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve: resolve!, reject: reject! };
}

describe("EvalDashboard error lifetime and blocking behavior", () => {
  beforeEach(() => {
    mockListEvalRuns.mockReset();
    mockGetEvalRun.mockReset();
    mockProjectEventHandlers.clear();
    mockReplayGapCallback = null;
  });

  it("clear-on-selection-change: stale detail error is cleared when selection moves", async () => {
    /* Isolates the EFFECT-START clear from the success-continuation clear.
     *
     * Both clears would remove this error, so a test that lets the new
     * selection's fetch resolve cannot tell which one did it — removing the
     * effect-start clear would still pass. To pin the effect-start clear on
     * its own, the new selection's detail fetch is held PENDING forever:
     * nothing can resolve to trigger the success clear, so the error can only
     * disappear if the effect cleared it on entry. */
    const deletedRun = makeRun("deleted.json", "v2");
    const survivingRun = makeRun("eval_results_baseline_20260807.json", "baseline");
    // Never settles — keeps the success-continuation clear out of reach.
    const pendingForever = createDeferred<ModelRun[]>();

    mockListEvalRuns.mockResolvedValue([deletedRun, survivingRun]);
    mockGetEvalRun.mockImplementation((_, file) => {
      if (file === "deleted.json") {
        return Promise.reject(new Error("Run not found: deleted.json"));
      }
      return pendingForever.promise;
    });

    render(
      <MemoryRouter initialEntries={["/project/test?run=deleted.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // Let the deleted run's error SETTLE into state and render.
    await waitFor(() => {
      expect(screen.getByText(/Run not found: deleted.json/i)).toBeInTheDocument();
    });

    // Delete event → fetchRuns → auto-selects the surviving run → new effect run.
    const deleteHandler = mockProjectEventHandlers.get("eval_run/deleted");
    mockListEvalRuns.mockResolvedValue([survivingRun]);
    if (deleteHandler) {
      deleteHandler({ run_filename: "deleted.json" });
    }

    // The new selection's fetch is now in flight and will never settle.
    await waitFor(() => {
      expect(mockGetEvalRun).toHaveBeenCalledWith("test", survivingRun._file);
    });

    // The stale error is gone, and only the effect-start clear could have done it.
    await waitFor(() => {
      expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    });
    expect(screen.getByText("Eval Run")).toBeInTheDocument();
  });

  it("stale-response guard: rejection settled after new selection does not write error", async () => {
    // Use deferred promises to control ordering
    const deletedDetailDeferred = createDeferred<ModelRun[]>();
    const survivingRun = makeRun("eval_results_baseline_20260807.json", "baseline");
    const listDeferred = createDeferred<EvalRunSummary[]>();

    /* The surviving run's detail fetch is held PENDING so it cannot resolve and
     * clear the error itself. Without that, a late rejection would be masked by
     * the success clear and removing the cancellation guard would still pass —
     * the guard would be unproven. Pending isolates it: the only thing that can
     * keep the stale rejection off screen is `cancelled`. */
    const survivingPending = createDeferred<ModelRun[]>();
    mockListEvalRuns.mockReturnValue(listDeferred.promise);
    mockGetEvalRun.mockImplementation((_, file) => {
      if (file === "deleted.json") {
        return deletedDetailDeferred.promise;
      }
      return survivingPending.promise;
    });

    render(
      <MemoryRouter initialEntries={["/project/test?run=deleted.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // Initial detail fetch starts for deleted.json
    await waitFor(() => expect(mockGetEvalRun).toHaveBeenCalledWith("test", "deleted.json"));

    // Resolve list → auto-select surviving run → new detail fetch
    listDeferred.resolve([survivingRun]);
    await waitFor(() => expect(mockGetEvalRun).toHaveBeenCalledWith("test", survivingRun._file));

    // NOW reject the stale deleted.json fetch
    deletedDetailDeferred.reject(new Error("Run not found: deleted.json"));

    // After all settled, no error text and selector present
    await waitFor(() => {
      expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    });
    expect(screen.getByText("Eval Run")).toBeInTheDocument();
  });

  it("persistence: detail error for current selection survives silent refresh and re-render", async () => {
    // The unreadable file IS the first run, so auto-select keeps it current
    const unreadableRun = makeRun("unreadable.json", "v1");
    mockListEvalRuns.mockResolvedValue([unreadableRun]);
    mockGetEvalRun.mockRejectedValue(new Error("Run not found: unreadable.json"));

    const { rerender } = render(
      <MemoryRouter initialEntries={["/project/test"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // Wait for initial error to appear
    await waitFor(() => {
      expect(screen.getByText(/Run not found: unreadable.json/i)).toBeInTheDocument();
    });

    // Drive a silent list refresh via captured WS handler
    mockListEvalRuns.mockResolvedValue([unreadableRun]);
    const evalJobHandler = mockProjectEventHandlers.get("eval_job/completed");
    if (evalJobHandler) evalJobHandler({ output_file: "some_other_run.json" });

    // Wait a tick, then force unrelated re-render
    await new Promise((r) => setTimeout(r, 10));
    rerender(
      <MemoryRouter initialEntries={["/project/test"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // Error text STILL present
    expect(screen.getByText(/Run not found: unreadable.json/i)).toBeInTheDocument();
  });

  describe("List vs detail error asymmetry (pinned by adjacent tests)", () => {
    it("detail failure is non-blocking: selector survives detail error", async () => {
      const run1 = makeRun("run1.json", "v1");
      const run2 = makeRun("run2.json", "v2");
      mockListEvalRuns.mockResolvedValue([run1, run2]);
      mockGetEvalRun.mockRejectedValue(new Error("Run not found: run1.json"));

      render(
        <MemoryRouter initialEntries={["/project/test?run=run1.json"]}>
          <EvalDashboard projectId="test" />
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Run not found: run1.json/i)).toBeInTheDocument();
      });

      // Selector label is present — user can pick another run
      expect(screen.getByText("Eval Run")).toBeInTheDocument();
    });

    it("list failure is still blocking: body suppressed when the list fetch fails", async () => {
      /* runsError is set only by the mount fetch now. The delete branch that
       * used to route a refetch through `fetchRuns` is gone — every post-mount
       * refetch is silent (errors swallowed, the list self-heals on the next
       * event/focus/gap), and selection movement is owned by the runs-keyed
       * reconciliation effect. A populated-list runsError is therefore
       * unreachable, so this pins the mount-failure asymmetry with the
       * non-blocking detail-failure case above:
       * a failed list load blocks the body (nothing to select), and the
       * EMPTY STATE must not render in its place — the project may well
       * have runs we simply could not list. */
      mockListEvalRuns.mockRejectedValue(new Error("Failed to fetch run list"));
      mockGetEvalRun.mockResolvedValue([makeModelRun("run1.json")]);

      render(
        <MemoryRouter initialEntries={["/project/test"]}>
          <EvalDashboard projectId="test" />
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Failed to fetch run list/i)).toBeInTheDocument();
      });

      // Blocking: no selector, and no misleading empty state either.
      expect(screen.queryByText("Eval Run")).not.toBeInTheDocument();
      expect(screen.queryByText(/No eval runs yet/i)).not.toBeInTheDocument();
    });
  });

  it("zero runs plus unreadable selection: detail error renders, empty state does not", async () => {
    mockListEvalRuns.mockResolvedValue([]);
    mockGetEvalRun.mockRejectedValue(new Error("Run not found: bogus.json"));

    render(
      <MemoryRouter initialEntries={["/project/test?run=bogus.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Run not found: bogus.json/i)).toBeInTheDocument();
    });

    // Empty state text does NOT appear
    expect(screen.queryByText(/No eval runs yet/i)).not.toBeInTheDocument();
  });

  it("clear-on-successful-fetch: subsequent success for same selection clears error", async () => {
    const run = makeRun("flaky.json", "v1");
    let callCount = 0;

    mockListEvalRuns.mockResolvedValue([run]);
    mockGetEvalRun.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.reject(new Error("Run not found: flaky.json"));
      }
      return Promise.resolve([makeModelRun("flaky.json")]);
    });

    const { unmount } = render(
      <MemoryRouter initialEntries={["/project/test?run=flaky.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // First fetch fails
    await waitFor(() => {
      expect(screen.getByText(/Run not found: flaky.json/i)).toBeInTheDocument();
    });

    // Unmount and remount (fresh render) at same URL → refetch same selection → succeeds
    unmount();
    render(
      <MemoryRouter initialEntries={["/project/test?run=flaky.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // Error is cleared
    await waitFor(() => {
      expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    });
  });
  it("replay-gap wiring: an unservable replay refetches the run list", async () => {
    /* The server-side cannot-serve signal is inert unless a consumer acts on it.
     * This asserts the WIRING, which was the gap: the hook existed and was
     * exported but nothing called it, so the backend would emit replay_gap and
     * no client would ever refetch.
     *
     * Asserts the EFFECT (a second list fetch happens) rather than that the hook
     * was invoked, so removing the useCursorGapHandler call site turns it red. */
    const run = makeRun("baseline.json", "baseline");
    mockListEvalRuns.mockResolvedValue([run]);
    mockGetEvalRun.mockResolvedValue([makeModelRun("baseline.json")]);

    render(
      <MemoryRouter initialEntries={["/project/test"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(mockListEvalRuns).toHaveBeenCalledTimes(1);
    });

    // The component must have registered a replay-gap consumer.
    expect(mockReplayGapCallback).not.toBeNull();

    // Fire the signal the server sends when it cannot serve the replay range.
    mockReplayGapCallback!();

    // The honest response is a refetch, not a cursor adjustment.
    await waitFor(() => {
      expect(mockListEvalRuns).toHaveBeenCalledTimes(2);
    });
  });
  it("empty-set boundary: deleting the LAST run clears the detail error and shows the empty state", async () => {
    /* Deleting the last run moves the selection to nothing, which neither
     * detail-error clearing rule covers on its own: clear-on-selection-change
     * runs only for a non-null selection, and clear-on-successful-fetch never
     * applies because the LIST fetch succeeds (count=0) while the DETAIL
     * fetch 404s.
     *
     * Distinct from the case above, which pins that zero runs plus a hand-typed
     * bogus ?run= KEEPS the error — there the selection is still set. Only the
     * transition to a null selection clears. */
    const lastRun = makeRun("last.json", "v1");
    mockListEvalRuns.mockResolvedValue([lastRun]);
    mockGetEvalRun.mockRejectedValue(new Error("Run not found: last.json"));

    render(
      <MemoryRouter initialEntries={["/project/test?run=last.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    // The detail error is on screen while that run is still selected.
    await waitFor(() => {
      expect(screen.getByText(/Run not found: last.json/i)).toBeInTheDocument();
    });

    // Delete the selected run — the list is now empty; the reconciliation
    // effect observes the disappearance and clears the selection.
    mockListEvalRuns.mockResolvedValue([]);
    const deleteHandler = mockProjectEventHandlers.get("eval_run/deleted");
    if (deleteHandler) {
      deleteHandler({ run_filename: "last.json" });
    }

    // The error must not outlive the selection it described.
    await waitFor(() => {
      expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    });
    // And the empty state must render (it is gated on !runDetailError).
    expect(screen.getByText(/No eval runs yet/i)).toBeInTheDocument();
  });

  it("concurrent delete-event and cursor-gap refetches converge to the empty state", async () => {
    /* Deleting the LAST run can start the delete-event refetch and a
     * cursor-gap refetch almost together, both returning an empty list. With
     * selection movement owned by the delete handler alone, the cursor-gap
     * path stayed silent and the stale detail error survived. Reconciliation
     * is keyed on the runs list itself, so the outcome must not depend on how
     * many refetch paths race or which resolves last. */
    const lastRun = makeRun("last.json", "v1");
    mockListEvalRuns.mockResolvedValue([lastRun]);
    mockGetEvalRun.mockRejectedValue(new Error("Run not found: last.json"));

    render(
      <MemoryRouter initialEntries={["/project/test?run=last.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Run not found: last.json/i)).toBeInTheDocument();
    });

    // Both refetch paths observe the deletion concurrently.
    mockListEvalRuns.mockResolvedValue([]);
    mockProjectEventHandlers.get("eval_run/deleted")?.({ run_filename: "last.json" });
    mockReplayGapCallback?.();

    await waitFor(() => {
      expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    });
    expect(screen.getByText(/No eval runs yet/i)).toBeInTheDocument();
  });

  it("any refetch path converges the selection: cursor-gap alone observes the deletion", async () => {
    /* No eval_run/deleted event fires here at all. If selection movement
     * lived in the delete handler, this would go red. It discriminates the
     * invariant from a call-site patch. */
    const runA = makeRun("a.json", "v1");
    const runB = makeRun("b.json", "v2");
    mockListEvalRuns.mockResolvedValue([runA, runB]);
    mockGetEvalRun.mockResolvedValue([makeModelRun("a.json")]);

    render(
      <MemoryRouter initialEntries={["/project/test?run=a.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    await waitFor(() => expect(mockGetEvalRun).toHaveBeenCalledWith("test", "a.json"));

    // a.json is deleted out-of-band; the only signal is a cursor gap.
    mockListEvalRuns.mockResolvedValue([runB]);
    mockReplayGapCallback?.();

    // Selection converges to the newest survivor; no stale error appears.
    await waitFor(() => expect(mockGetEvalRun).toHaveBeenCalledWith("test", "b.json"));
    expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
    expect(screen.getByText("Eval Run")).toBeInTheDocument();
  });

  it("deleting a non-selected run leaves the selection alone", async () => {
    const runA = makeRun("a.json", "v1");
    const runB = makeRun("b.json", "v2");
    mockListEvalRuns.mockResolvedValue([runA, runB]);
    mockGetEvalRun.mockResolvedValue([makeModelRun("a.json")]);

    render(
      <MemoryRouter initialEntries={["/project/test?run=a.json"]}>
        <EvalDashboard projectId="test" />
      </MemoryRouter>
    );

    await waitFor(() => expect(mockGetEvalRun).toHaveBeenCalledWith("test", "a.json"));
    const callsBefore = mockGetEvalRun.mock.calls.length;

    mockListEvalRuns.mockResolvedValue([runA]);
    mockProjectEventHandlers.get("eval_run/deleted")?.({ run_filename: "b.json" });

    // List refreshed, selection untouched — no new detail fetch, no error.
    await waitFor(() => expect(mockListEvalRuns.mock.calls.length).toBeGreaterThanOrEqual(2));
    expect(mockGetEvalRun.mock.calls.length).toBe(callsBefore);
    expect(screen.queryByText(/Run not found/i)).not.toBeInTheDocument();
  });
});
