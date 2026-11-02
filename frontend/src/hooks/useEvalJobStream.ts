// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hook for streaming eval job progress over the unified WebSocket.
 *
 * Encapsulates the WebSocket subscription on the ``job:<id>`` stream of the
 * active job the provider reports, envelope parsing with Set-based dedup for
 * out-of-order case completion from parallel workers, and per-model progress
 * state management. Detecting the active job is the provider's ``/activity``
 * poll, not this hook.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { useJobStream } from "@/hooks/useWsStream";
import { api } from "@/lib/api";
import type { Envelope } from "@/lib/ws-client";
import {
  initialMachine,
  mapActivityModel,
  reduceEvalProgress,
} from "@/lib/eval-progress-reducer";
import type {
  EvalProgressMachine,
  EvalProgressState,
  ModelProgress,
} from "@/lib/eval-progress-reducer";
import type { EvalJob, ProjectActivityEval } from "@/lib/types";

// Re-exported so existing importers (ActiveJobBar and friends) keep working
// after the state machine moved into its own testable module.
export type { EvalProgressState, ModelProgress };

/** Return type from the useEvalJobStream hook. */
export interface UseEvalJobStreamResult {
  /** Current progress state, or null if no active job */
  progress: EvalProgressState | null;
  /** Active job ID, or null */
  jobId: string | null;
  /** Whether a cancel request is in flight */
  cancelling: boolean;
  /** Cancel the current eval job */
  handleCancel: () => void;
}

/**
 * Stream eval job progress over the per-job WebSocket.
 *
 * Active-job DETECTION is no longer this hook's job — it's driven by the
 * single ``/activity`` poll in ``OptimizationStateProvider``, passed in
 * as ``activeEval``. This hook owns only the ``job:<id>`` WebSocket
 * subscription and the per-case progress state machine (Set-based case
 * counting for accurate out-of-order completion across parallel
 * workers). Collapsing the old per-hook ``listEvalJobs`` poll into the
 * provider's ``/activity`` poll is what removed the second source of
 * truth: the bar's liveness now comes from the same reconciled answer
 * the optimization bar reads, so the two can't disagree.
 *
 * @param projectId - Project UUID (for the cancel call).
 * @param activeEval - The reconciled active eval from the provider, or
 *   ``null`` when none is running. The hook subscribes to its
 *   ``job_id`` and clears its progress when this flips to ``null``.
 * @param onComplete - Optional callback when a job completes.
 * @param onStatusChange - Optional callback when job status changes.
 */
export function useEvalJobStream(
  projectId: string,
  activeEval: ProjectActivityEval | null,
  onComplete?: () => void,
  onStatusChange?: (status: EvalJob["status"] | null) => void,
): UseEvalJobStreamResult {
  const [progress, setProgress] = useState<EvalProgressState | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onStatusChangeRef = useRef(onStatusChange);
  onStatusChangeRef.current = onStatusChange;

  const completedRef = useRef<string | null>(null);
  const jobIdRef = useRef<string | null>(null);
  jobIdRef.current = jobId;

  /**
   * The whole progress state machine, held in a ref.
   *
   * Carries the rendered progress AND the per-model completed-case sets +
   * models_total that used to be two separate refs. Everything the fold needs
   * now lives in one value, so the transition logic is a pure function in
   * ``eval-progress-reducer`` and is unit-tested there; this hook only owns
   * subscription, the React mirror, and the effect callbacks.
   */
  const machineRef = useRef<EvalProgressMachine>(initialMachine());

  /**
   * Dispatch an envelope from the ``job:<id>`` stream into the progress state
   * machine.
   *
   * Thin wrapper: the transition itself is ``reduceEvalProgress``, a pure
   * function tested directly in ``eval-progress-reducer.test.ts``. This wrapper
   * threads the ref, mirrors the result into React state, and performs the
   * effects the reducer REQUESTS (rather than performs) — which is what makes
   * "the completion callback fires exactly once per job" an assertable property
   * instead of a side effect buried in the fold.
   */
  const handleEnvelope = useCallback((env: Envelope) => {
    const { state, effects } = reduceEvalProgress(
      machineRef.current,
      env,
      completedRef.current,
      jobIdRef.current,
    );
    machineRef.current = state;
    setProgress(state.progress);

    if (effects.completed && jobIdRef.current) {
      completedRef.current = jobIdRef.current;
      /* ``eval_job.complete_job`` publishes ``{resource:"eval_job",
       * type:"completed"}`` on the project event stream, so dashboards refetch
       * via ``useProjectEvent``. No window dispatch here. */
      onCompleteRef.current?.();
    }
    if (effects.statusChanged !== undefined) {
      onStatusChangeRef.current?.(effects.statusChanged);
    }
  }, []);

  /* Subscribe to the active job's stream. ``useJobStream`` no-ops when
   * ``jobId`` is null and re-subscribes when it changes — the dedup
   * state is also reset on flip by the active-eval effect below. */
  useJobStream(jobId, handleEnvelope);

  /** Cancel the currently active eval job.
   *
   * A refused cancel is shown, as the optimization Stop shows one. The backend
   * answers 503 when it could not confirm the worker stopped and leaves the eval
   * running on purpose; swallowing that turned "Stopping..." back into "Stop"
   * with no word of why, while the eval carried on. */
  const handleCancel = useCallback(() => {
    if (!jobId || cancelling) return;
    setCancelling(true);
    api.cancelEvalJob(projectId, jobId)
      .catch((err: unknown) => {
        toast.error(err instanceof Error ? err.message : "Failed to stop the eval");
      })
      .finally(() => setCancelling(false));
  }, [jobId, cancelling, projectId]);

  /* React to the provider's reconciled active-eval slice. This replaces
   * the old per-hook ``listEvalJobs`` poll: detection is now centralized
   * in the ``/activity`` poll, and the eval slice it returns is already
   * reconciled (a stale optimization-child eval whose loop ended is
   * reported as ``null``). So when ``activeEval`` is null we authoritatively
   * clear the bar — the same "poll can't be dropped" guarantee that
   * unstuck it before, now sourced from one reconciled endpoint. */
  const activeEvalJobId = activeEval?.job_id ?? null;
  useEffect(() => {
    if (activeEvalJobId && activeEvalJobId !== jobIdRef.current) {
      /* New active job — reset per-model dedup state and capture the
       * model count so envelope handlers populate ``modelsTotal``
       * consistently. */
      machineRef.current = initialMachine(activeEval?.models_total ?? 0);
      setJobId(activeEvalJobId);
      /* Seed the per-model rows from the backend event-log fold so a
       * mid-scoring remount repopulates ``Generate X/N`` instead of rebuilding
       * only from the live WS stream (whose seq cursor survives remount and so
       * replays only NEW events, missing the already-passed generate events).
       * The per-model Set is intentionally left empty (see mapActivityModel).
       * Live WS deltas continue to flow through ``updateModel`` (upsert by
       * name), so a seeded model merges cleanly with subsequent events. */
      const seededModels = (activeEval?.models ?? []).map(mapActivityModel);
      const seeded: EvalProgressState = {
        status: activeEval?.status ?? "running",
        phase: "pending",
        modelsTotal: activeEval?.models_total ?? 0,
        models: seededModels,
        error: null,
      };
      machineRef.current = { ...machineRef.current, progress: seeded };
      setProgress(seeded);
      onStatusChangeRef.current?.(activeEval?.status ?? "running");
    } else if (!activeEvalJobId && jobIdRef.current) {
      /* The reconciled poll says no eval is running but we still hold a
       * job locally — the run ended. Clear the bar here rather than
       * waiting on a per-job ``done`` envelope that may never arrive
       * (dropped on coordinator restart / killed subprocess). This is
       * what stops the "Evaluation running, model X of N" bar (with a
       * live Stop button) from sticking after the backend stopped. */
      setProgress(null);
      setJobId(null);
      machineRef.current = initialMachine();
      onStatusChangeRef.current?.(null);
    }
    // ``activeEval?.models`` is read inside (for the seed) but deliberately NOT
    // a dependency: the effect body is guarded on the job ID CHANGING, so it only
    // ever runs at promotion, and the /activity poll returns a fresh ``models``
    // array every 3s. Depending on it would re-run this effect on every poll
    // tick, and re-seeding mid-run would reset the machine and wipe live WS
    // progress — the "bar resets to 0/N every few seconds" failure. Listed via
    // the eslint directive rather than silently, so the omission is a stated
    // decision and not an oversight.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeEvalJobId, activeEval?.models_total, activeEval?.status]);

  /* Reset all local state when the project changes. */
  useEffect(() => {
    setProgress(null);
    setJobId(null);
    completedRef.current = null;
    machineRef.current = initialMachine();
  }, [projectId]);

  return { progress, jobId, cancelling, handleCancel };
}
