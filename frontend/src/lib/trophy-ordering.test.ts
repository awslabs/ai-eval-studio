// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment node
 *
 * Cross-language golden vector. The backend's pytest suite asserts
 * app/services/trophy.py produces `expected_score` for every candidate in
 * backend/tests/fixtures/trophy_golden.json; THIS test asserts the
 * frontend's ordering comparator (pickTrophyIndex — the only trophy logic
 * left client-side) picks `expected_winner_index` from those same tuples.
 * A scoring-semantics change edits the fixture and forces both suites to
 * move together; editing one implementation alone goes red on one side.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { pickTrophyIndex } from "@/components/eval/OptimizationRunDetail";
import type { TrophyScore } from "@/lib/types";

const golden = JSON.parse(
  readFileSync(
    resolve(__dirname, "../../../backend/tests/fixtures/trophy_golden.json"),
    "utf-8",
  ),
) as {
  expected_winner_index: number;
  candidates: { name: string; expected_score: TrophyScore }[];
};

describe("trophy golden vector (shared with backend pytest)", () => {
  it("orders the backend's expected scores to the expected winner", () => {
    const scores = golden.candidates.map((c) => c.expected_score);
    expect(pickTrophyIndex(scores)).toBe(golden.expected_winner_index);
  });

  it("later wins on a full tie (the documented tiebreak)", () => {
    const tie: TrophyScore = { at_target: 1, best_rate: 0.9, avg_rate: 0.9 };
    expect(pickTrophyIndex([tie, { ...tie }, { ...tie }])).toBe(2);
  });

  it("empty input yields no trophy", () => {
    expect(pickTrophyIndex([])).toBe(-1);
  });
});
