// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The ONE compile-checked classification of optimization statuses.
 *
 * Keyed on the `OptimizationState["status"]` union so adding a status to
 * the type fails the build HERE, forcing an explicit classification —
 * instead of silently falling through a string Set (the old
 * TERMINAL_STATUSES) or an `===` chain (the old isResumable), both of
 * which treated an unknown status as "not mine" without anyone deciding
 * that.
 *
 * Display maps (labels, colors) stay with their renderers
 * (`OPT_STATUS_LABELS`, `statusColors` — already Record-typed); this
 * module owns only the behavioral classification.
 */

import type { OptimizationState } from "@/lib/types";

interface StatusClass {
  /** The run is over from the backend's perspective. */
  terminal: boolean;
  /** The resume endpoint accepts this status (mirror of the backend
   * allowlist in api/optimization.py — "failed" is deliberately NOT
   * resumable there, and "converged" reached its goal). */
  resumable: boolean;
}

const STATUS_CLASS: Record<OptimizationState["status"], StatusClass> = {
  starting: { terminal: false, resumable: false },
  running: { terminal: false, resumable: false },
  stopping: { terminal: false, resumable: false },
  converged: { terminal: true, resumable: false },
  plateau: { terminal: true, resumable: true },
  max_iterations: { terminal: true, resumable: true },
  optimizer_exhausted: { terminal: true, resumable: true },
  optimizer_stalled: { terminal: true, resumable: true },
  judge_unreliable: { terminal: true, resumable: true },
  generation_unreliable: { terminal: true, resumable: true },
  failed: { terminal: true, resumable: false },
  cancelled: { terminal: true, resumable: true },
};

/** True when the backend considers the run over. Null/unknown → false. */
export function isTerminalStatus(status: string | null | undefined): boolean {
  if (!status) return false;
  return STATUS_CLASS[status as OptimizationState["status"]]?.terminal ?? false;
}

/** True when the resume endpoint would accept this status. */
export function isResumableStatus(status: string | null | undefined): boolean {
  if (!status) return false;
  return STATUS_CLASS[status as OptimizationState["status"]]?.resumable ?? false;
}
