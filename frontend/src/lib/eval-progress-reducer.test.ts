// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the eval-progress state machine.
 *
 * This fold drives the eval bar, and the "stale bar" failures are its known
 * shapes: the bar frozen at "model 0 of N", generated counts regressing
 * mid-run, a cancelled job leaving a live Stop button. The fold is a pure
 * function, so each shape is tested here directly rather than through the
 * hook that owns the subscription.
 *
 * These tests target the specific failure shapes, not line coverage.
 */
import { describe, expect, it } from "vitest";

import {
  DEFAULT_WORKERS,
  initialMachine,
  mapActivityModel,
  reduceEvalProgress,
} from "@/lib/eval-progress-reducer";
import type { EvalProgressMachine } from "@/lib/eval-progress-reducer";

/** Fold a list of envelopes, returning the final state and all effects. */
function fold(
  envelopes: Array<{ kind: string; data?: unknown }>,
  start: EvalProgressMachine = initialMachine(2),
  jobId = "job-1",
) {
  let state = start;
  const effects: Array<ReturnType<typeof reduceEvalProgress>["effects"]> = [];
  let completedJobId: string | null = null;
  for (const env of envelopes) {
    const r = reduceEvalProgress(state, env, completedJobId, jobId);
    state = r.state;
    effects.push(r.effects);
    if (r.effects.completed) completedJobId = jobId;
  }
  return { state, effects };
}

const gen = (model: string, current: number, total: number) => ({
  kind: "progress",
  data: { event: "generating", model, current, total },
});

describe("generating", () => {
  it("counts DISTINCT case indices, so out-of-order workers cannot regress it", () => {
    // The harness runs cases across parallel workers, so `current` arrives out
    // of order. A naive "generated = current" would report 3 then 1 then 2 and
    // the bar would visibly count backwards.
    const { state } = fold([
      gen("m1", 3, 10),
      gen("m1", 1, 10),
      gen("m1", 2, 10),
    ]);
    const m1 = state.progress!.models.find((m) => m.model === "m1")!;
    expect(m1.generated).toBe(3);
    expect(m1.casesTotal).toBe(10);
  });

  it("ignores duplicate indices (at-least-once delivery must not inflate)", () => {
    // Replay-on-reconnect and the durable log's at-least-once semantics both
    // redeliver events. A counter would double-count; a set cannot.
    const { state } = fold([gen("m1", 1, 10), gen("m1", 1, 10), gen("m1", 1, 10)]);
    expect(state.progress!.models[0].generated).toBe(1);
  });

  it("tracks models independently", () => {
    const { state } = fold([gen("m1", 1, 10), gen("m2", 1, 10), gen("m2", 2, 10)]);
    const byName = Object.fromEntries(
      state.progress!.models.map((m) => [m.model, m.generated]),
    );
    expect(byName).toEqual({ m1: 1, m2: 2 });
  });

  it("defaults workers when the event omits it", () => {
    const { state } = fold([
      { kind: "progress", data: { event: "generating", model: "m1", current: 1, total: 5 } },
    ]);
    expect(state.progress!.models[0].workers).toBe(DEFAULT_WORKERS);
  });
});

describe("phase transitions", () => {
  it("scoring_start pins generated to casesTotal", () => {
    // Generation is finished by definition once scoring starts, so a partial
    // generated count must not linger — that showed as "Generate 7/10" frozen
    // next to a scoring spinner.
    const { state } = fold([gen("m1", 1, 10), { kind: "progress", data: { event: "scoring_start", model: "m1" } }]);
    const m1 = state.progress!.models[0];
    expect(m1.phase).toBe("scoring");
    expect(m1.generated).toBe(10);
    expect(m1.scored).toBe(false);
    expect(m1.scoredCount).toBe(0);
  });

  it("scoring_progress updates scoredCount and can supply casesTotal", () => {
    const { state } = fold([
      { kind: "progress", data: { event: "scoring_progress", model: "m1", current: 4, total: 12 } },
    ]);
    expect(state.progress!.models[0].scoredCount).toBe(4);
    expect(state.progress!.models[0].casesTotal).toBe(12);
  });

  it("scoring_progress with total=0 keeps the known casesTotal", () => {
    // A zero total would otherwise wipe the denominator and render "4/0".
    const { state } = fold([
      gen("m1", 1, 10),
      { kind: "progress", data: { event: "scoring_progress", model: "m1", current: 4, total: 0 } },
    ]);
    expect(state.progress!.models[0].casesTotal).toBe(10);
  });

  it("scoring_complete marks scored and fills scoredCount", () => {
    const { state } = fold([
      gen("m1", 1, 10),
      { kind: "progress", data: { event: "scoring_complete", model: "m1", passed: 7 } },
    ]);
    const m1 = state.progress!.models[0];
    expect(m1.scored).toBe(true);
    expect(m1.scoredCount).toBe(10);
    expect(m1.passed).toBe(7);
  });

  it("complete drops the model's dedup set but keeps its rendered row", () => {
    // The set is per-model bookkeeping; dropping it bounds memory on a long
    // multi-model run. The row must survive — it's what the UI shows.
    const { state } = fold([
      gen("m1", 1, 10),
      { kind: "progress", data: { event: "complete", model: "m1", passed: 9 } },
    ]);
    expect(state.completedCases.has("m1")).toBe(false);
    const m1 = state.progress!.models[0];
    expect(m1.phase).toBe("complete");
    expect(m1.generated).toBe(10);
    expect(m1.passed).toBe(9);
  });

  it("error sets failed status but preserves the model rows", () => {
    // The rows are the diagnostic — wiping them on failure leaves the user with
    // an error and no indication of how far the run got.
    const { state } = fold([
      gen("m1", 3, 10),
      { kind: "progress", data: { event: "error", error: "Bedrock throttled" } },
    ]);
    expect(state.progress!.status).toBe("failed");
    expect(state.progress!.phase).toBe("failed");
    expect(state.progress!.error).toBe("Bedrock throttled");
    expect(state.progress!.models).toHaveLength(1);
  });
});

describe("terminal (done) envelopes", () => {
  it("requests the completed effect exactly ONCE per job", () => {
    // Firing twice double-reports a finished run to the dashboard. The fold sees
    // duplicate `done` envelopes routinely (replay + live fan-out).
    const { effects } = fold([
      gen("m1", 1, 10),
      { kind: "done", data: { status: "completed" } },
      { kind: "done", data: { status: "completed" } },
      { kind: "done", data: { status: "completed" } },
    ]);
    expect(effects.filter((e) => e.completed).length).toBe(1);
  });

  it("reports every terminal status change, including repeats", () => {
    // statusChanged is idempotent for the consumer, so it is NOT deduped —
    // only `completed` is once-per-job.
    const { effects } = fold([
      { kind: "done", data: { status: "completed" } },
      { kind: "done", data: { status: "completed" } },
    ]);
    expect(effects.map((e) => e.statusChanged)).toEqual(["completed", "completed"]);
  });

  it("cancelled reaches a terminal phase so the Stop button clears", () => {
    // A cancelled job that stays phase:"generating" is exactly the reported bug:
    // "Evaluation running, model X of N" with a live Stop button, forever.
    const { state } = fold([
      gen("m1", 3, 10),
      { kind: "done", data: { status: "cancelled" } },
    ]);
    expect(state.progress!.status).toBe("cancelled");
    expect(state.progress!.phase).toBe("cancelled");
  });

  it("failed reaches a terminal phase", () => {
    const { state } = fold([gen("m1", 3, 10), { kind: "done", data: { status: "failed" } }]);
    expect(state.progress!.status).toBe("failed");
    expect(state.progress!.phase).toBe("failed");
  });

  it("a done envelope with no prior progress does not fabricate state", () => {
    // Subscribing after a job already finished delivers `done` with no preceding
    // progress. Inventing an empty progress object would render a bar for a job
    // the user never saw start.
    const { state, effects } = fold([{ kind: "done", data: { status: "completed" } }]);
    expect(state.progress).toBeNull();
    expect(effects[0].statusChanged).toBe("completed");
  });
});

describe("robustness", () => {
  it("an unknown progress event is a no-op, not a wedge", () => {
    // The harness may emit events this client predates. Dropping the envelope
    // must not disturb accumulated state.
    const before = fold([gen("m1", 1, 10)]).state;
    const after = reduceEvalProgress(before, {
      kind: "progress",
      data: { event: "some_future_event", model: "m1" },
    }).state;
    expect(after.progress).toEqual(before.progress);
  });

  it("an unknown envelope kind is a no-op", () => {
    const before = fold([gen("m1", 1, 10)]).state;
    const after = reduceEvalProgress(before, { kind: "keepalive" }).state;
    expect(after).toBe(before);
  });

  it("a missing data payload does not throw", () => {
    expect(() => reduceEvalProgress(initialMachine(1), { kind: "progress" })).not.toThrow();
    expect(() => reduceEvalProgress(initialMachine(1), { kind: "done" })).not.toThrow();
  });

  it("does not mutate the input state (fold purity)", () => {
    // Purity is what makes this testable at all; a shared mutable set would
    // reintroduce cross-render bleed.
    const start = fold([gen("m1", 1, 10)]).state;
    const snapshotSize = start.completedCases.get("m1")!.size;
    reduceEvalProgress(start, gen("m1", 2, 10));
    expect(start.completedCases.get("m1")!.size).toBe(snapshotSize);
  });

  it("carries modelsTotal from the machine, not from events", () => {
    const { state } = fold([gen("m1", 1, 10)], initialMachine(7));
    expect(state.progress!.modelsTotal).toBe(7);
  });
});

describe("mapActivityModel (remount seeding)", () => {
  it("maps the snake_case wire shape to the client shape", () => {
    expect(
      mapActivityModel({
        model: "m1",
        phase: "scoring",
        generated: 10,
        cases_total: 10,
        scored: false,
        scored_count: 3,
        passed: 0,
      }),
    ).toEqual({
      model: "m1",
      phase: "scoring",
      generated: 10,
      casesTotal: 10,
      scored: false,
      scoredCount: 3,
      passed: 0,
      workers: DEFAULT_WORKERS,
    });
  });

  it("a live generating event after seeding rebuilds the count from the empty dedup set", () => {
    // The seeded row deliberately leaves the dedup set EMPTY. This pins the
    // documented consequence: a straggler `generating` event after a mid-scoring
    // remount rebuilds `generated` from that set and reports 1, down from the
    // seeded 10. That is accepted because seeding happens only past the
    // generate phase, so this asserts the shape the code actually produces.
    const seeded = mapActivityModel({
      model: "m1",
      phase: "scoring",
      generated: 10,
      cases_total: 10,
      scored: false,
      scored_count: 3,
      passed: 0,
    });
    const machine: EvalProgressMachine = {
      ...initialMachine(1),
      progress: {
        status: "running",
        phase: "pending",
        modelsTotal: 1,
        models: [seeded],
        error: null,
      },
    };
    const { state } = fold([gen("m1", 1, 10)], machine);
    // Documents real behavior: the live event rebuilds `generated` from the
    // (empty) set. If a future change makes seeding populate the set, this
    // expectation should become 10 — and this test is where that shows up.
    expect(state.progress!.models[0].generated).toBe(1);
  });
});
