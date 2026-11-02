// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hook for fetching and caching eval run summaries for a project.
 * Re-fetches when projectId changes, when refresh() is called, and on
 * job-completed events from the WS bus so consumers see new runs land
 * live without a page reload.
 */

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import type { EvalRunSummary } from "@/lib/types";

/** Return value from the useEvalRuns hook. */
export interface UseEvalRunsResult {
  /** Fetched run summaries, or null while the first fetch is in-flight. */
  runs: EvalRunSummary[] | null;
  /** Error from the most recent fetch attempt, or null. */
  error: Error | null;
  /** True while a fetch is in-flight. */
  loading: boolean;
  /** Trigger a manual re-fetch. */
  refresh: () => Promise<void>;
}

/**
 * Fetch eval run summaries for a project, with manual refresh support.
 *
 * @param projectId - Project UUID to list runs for
 * @returns Runs list, loading flag, error state, and refresh function
 */
export function useEvalRuns(projectId: string): UseEvalRunsResult {
  const [runs, setRuns] = useState<EvalRunSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listEvalRuns(projectId);
      setRuns(data);
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Backend publishes ``{resource: "job", type: "completed"}`` when an
  // eval or optimization-iteration completes. Subscribing here keeps
  // every consumer (ReportsTab, SummaryTab, future) auto-current —
  // single bus, single hook. The default-selection guard in consumers
  // (``if (!selectedFile && runs)``) stops this from yanking the user's
  // current selection when a fresh run lands.
  useProjectEvent({ resource: "job", type: "completed" }, () => {
    refresh();
  }, "state");

  // A run deleted out-of-band (the agent's ``delete_run`` tool, or another
  // tab) is not a job completion, so it needs its own subscription — without
  // it every consumer keeps listing a run whose file is already gone.
  useProjectEvent({ resource: "eval_run", type: "deleted" }, () => {
    refresh();
  }, "state");

  return { runs, error, loading, refresh };
}
