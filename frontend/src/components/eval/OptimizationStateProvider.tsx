// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Single source of truth for optimization state across the project page.
 *
 * Problem being solved: ``ActiveJobBar`` and ``EvalDashboard`` previously
 * each owned their own ``optState`` and fetched it independently on
 * different triggers (poll vs window-focus event). The two copies drifted
 * constantly — one would show "running" while the other showed a stale
 * "optimizer_stalled" error banner from the previous run. Consolidating
 * the fetch into one provider means both components read the same value,
 * by construction.
 *
 * The provider owns:
 * - Backstop polling of ``/api/projects/{id}/activity`` at
 *   ``POLL_INTERVAL_MS`` (see that constant for the cadence and why it is
 *   a backstop rather than the primary refresh path).
 * - Optimistic transition state (``starting``/``stopping``) with
 *   write-skip guards so the poll can't undo an optimistic update until
 *   the backend actually catches up.
 * - An imperative ``refetch`` for callers that just issued a mutation
 *   (start/stop/resume) and want the UI to reflect it immediately.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "@/lib/api";
import { isTerminalStatus } from "@/lib/optimization-status";
import type { OptimizationState, ProjectActivityEval } from "@/lib/types";
import { useProjectEvent } from "@/hooks/useBackendEvents";

/** Backstop poll for ``/activity``. Event-driven refresh
 *  (``useProjectEvent`` subscriptions below) handles the common case
 *  sub-second; this poll is the safety net for WS gaps (laptop sleep,
 *  network blip) and publisher paths that mutate state without firing
 *  an event. The cadence drives both the optimization bar and the
 *  eval-bar liveness, so it stays snappy. */
const POLL_INTERVAL_MS = 3_000;

/* Terminal-status classification lives in lib/optimization-status.ts:
 * one compile-checked Record keyed on the status union, so a status added
 * to the type fails the build there instead of silently missing this
 * file's old string Set. */

interface OptimizationStateContextValue {
  /** Latest known optimization state, or ``null`` if none / not loaded. */
  state: OptimizationState | null;
  /** The reconciled active eval for this project, or ``null`` if none.
   *  Comes from the same ``/activity`` poll as ``state`` — the backend
   *  has already suppressed a stale optimization-child eval whose loop
   *  ended, so this is authoritative for "is the eval bar up". */
  activeEval: ProjectActivityEval | null;
  /** Fires once the initial fetch has landed (success or null). */
  isLoaded: boolean;
  /** Force an immediate refetch — call after issuing start/stop/resume. */
  refetch: () => void;
  /** Apply an optimistic transition (e.g. ``"starting"``). The next
   *  poll / refetch will overwrite once the backend confirms. */
  setOptimistic: (partial: Partial<OptimizationState> | null) => void;
}

const OptimizationStateContext =
  createContext<OptimizationStateContextValue | null>(null);

interface ProviderProps {
  projectId: string;
  children: ReactNode;
}

/**
 * Wrap any subtree that reads or mutates optimization state with this
 * provider. One instance per ``ProjectPage`` — it keys internally on the
 * project id and resets polling on project changes.
 */
export function OptimizationStateProvider({ projectId, children }: ProviderProps) {
  const [state, setState] = useState<OptimizationState | null>(null);
  const [activeEval, setActiveEval] = useState<ProjectActivityEval | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  /* Transient optimistic override keeps the UI ahead of the poll after
   * a start/stop click. We merge this into the returned ``state`` so the
   * caller sees the optimistic fields, but the raw server state lives in
   * ``state`` so the next poll can correct any optimistic lies. */
  const optimisticRef = useRef<Partial<OptimizationState> | null>(null);

  const fetchOnce = useCallback(async () => {
    try {
      const activity = await api.getProjectActivity(projectId);
      const fresh = activity.optimization;
      /* The eval slice is already reconciled by the backend (a stale
       * optimization-child eval is reported null), so we adopt it
       * verbatim — no client-side stitching, which is the whole point
       * of the consolidation. */
      setActiveEval(activity.eval);
      setState((prev) => {
        /* Respect optimistic transitions: if we locally declared
         * ``starting`` but the backend still reports the prior terminal
         * status, hold the optimistic value so the button doesn't flicker
         * back to "Resume" for a beat. Same for ``stopping``. Once the
         * backend actually moves to ``running`` or a terminal status
         * matching the intent, we drop the override. */
        const optimistic = optimisticRef.current;
        /* Always clear an optimistic override when the backend reports
         * a terminal status — the run is done either way and holding
         * "starting" past convergence is what made the bar stick on
         * "running" while the chat already announced it. */
        if (fresh && isTerminalStatus(fresh.status)) {
          optimisticRef.current = null;
        } else if (optimistic?.status === "starting") {
          if (fresh?.status === "running" || fresh?.status === "starting") {
            optimisticRef.current = null;
          } else {
            return prev ? { ...prev, ...optimistic, ...fresh, status: "starting" } : fresh;
          }
        } else if (optimistic?.status === "stopping") {
          if (fresh?.status === "cancelled" || fresh?.status === "stopping") {
            optimisticRef.current = null;
          } else {
            return prev ? { ...prev, ...optimistic, ...fresh, status: "stopping" } : fresh;
          }
        }
        return fresh;
      });
    } catch {
      /* Swallow network errors — the next tick retries. We do NOT clear
       * the cached state on a single failed poll because transient
       * hiccups would then look like the run vanished. */
    } finally {
      setIsLoaded(true);
    }
  }, [projectId]);

  /* Poll on mount and every POLL_INTERVAL_MS thereafter. Reset cleanly
   * whenever the project changes — otherwise a leftover interval from
   * the previous project would keep writing stale data into the
   * provider for this one. */
  useEffect(() => {
    optimisticRef.current = null;
    setState(null);
    setActiveEval(null);
    setIsLoaded(false);
    fetchOnce();
    const id = setInterval(fetchOnce, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [fetchOnce, projectId]);

  const refetch = useCallback(() => {
    fetchOnce();
  }, [fetchOnce]);

  /* Event-driven refresh on top of the 3s poll. Backend publishes
   * ``optimization/terminal`` on convergence and ``job/completed`` on
   * each iteration boundary; subscribing means status flips within ~1s
   * of the chat announcing convergence instead of waiting a full poll
   * interval. ``eval_job`` events are subscribed because this provider
   * now also owns the eval slice — ``started`` raises the eval bar the
   * instant dispatch begins (no startup blackout while the model
   * preflight runs), and the terminal events clear it promptly rather
   * than on the next poll tick. The
   * 3s poll stays as the correctness fallback when an envelope is
   * dropped on reconnect (which is exactly the failure mode that left
   * the eval bar stuck before — the poll, which can't be dropped, is
   * the authoritative clear). */
  useProjectEvent({ resource: "optimization", type: "terminal" }, refetch, "state");
  useProjectEvent({ resource: "optimization", type: "running" }, refetch, "state");
  useProjectEvent({ resource: "job", type: "completed" }, refetch, "state");
  useProjectEvent({ resource: "eval_job", type: "started" }, refetch, "state");
  useProjectEvent({ resource: "eval_job", type: "completed" }, refetch, "state");
  useProjectEvent({ resource: "eval_job", type: "cancelled" }, refetch, "state");
  useProjectEvent({ resource: "eval_job", type: "error" }, refetch, "state");

  const setOptimistic = useCallback(
    (partial: Partial<OptimizationState> | null) => {
      optimisticRef.current = partial;
      setState((prev) => {
        if (!partial) return prev;
        if (!prev) {
          /* No base state yet — the user clicked Start on a project that
           * has never optimized before. The placeholder carries ONLY
           * structural emptiness: it used to fabricate config values
           * (iteration_cap 10, plateau 3/0.02) that could render a wrong
           * "Iteration 1 of 10" for a project whose real cap differs.
           * Every consumer null-guards (`optState?.x ?? "?"`), so absent
           * fields render honestly as unknown for the seconds until the
           * first poll returns the real record. The cast is deliberate:
           * this is a knowingly-partial placeholder, not a real record. */
          return {
            status: "starting",
            current_iteration: 0,
            target_metric_scores: {},
            iterations: [],
            best_iteration: null,
            ...partial,
          } as unknown as OptimizationState;
        }
        return { ...prev, ...partial };
      });
    },
    [],
  );

  return (
    <OptimizationStateContext.Provider
      value={{ state, activeEval, isLoaded, refetch, setOptimistic }}
    >
      {children}
    </OptimizationStateContext.Provider>
  );
}

/**
 * Read optimization state from the surrounding provider.
 * Throws if no provider is mounted — prevents silent component-level
 * duplicate fetches that would reintroduce the drift bug.
 */
export function useOptimizationState(): OptimizationStateContextValue {
  const ctx = useContext(OptimizationStateContext);
  if (!ctx) {
    throw new Error(
      "useOptimizationState must be used within an OptimizationStateProvider",
    );
  }
  return ctx;
}
