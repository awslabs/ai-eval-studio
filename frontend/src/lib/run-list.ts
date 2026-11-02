// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * THE run-list ordering contract, in one place.
 *
 * `GET /eval/runs` returns runs sorted **newest first** — the backend owner
 * is `eval_runner.list_runs` (sorts by run-file timestamp, descending) and
 * its docstring declares it. Callers used to encode that assumption as bare
 * index arithmetic (`runs[0]`, `runs[runs.length - 1]`) scattered across
 * five components, where a future ordering change would silently select
 * wrong runs instead of breaking one named function.
 *
 * These accessors are deliberately trivial: the point is that the
 * assumption has exactly one home and a name, not that the logic is
 * complex. Do not re-sort here — the backend owns the ordering; sorting
 * again would create a second authority that could disagree.
 */

import type { EvalRunSummary } from "@/lib/types";

/** Newest run in the list, or null when empty. */
export function newestRun(runs: EvalRunSummary[]): EvalRunSummary | null {
  return runs.length > 0 ? runs[0] : null;
}

/** Oldest run in the list (e.g. the default compare baseline), or null. */
export function oldestRun(runs: EvalRunSummary[]): EvalRunSummary | null {
  return runs.length > 0 ? runs[runs.length - 1] : null;
}
