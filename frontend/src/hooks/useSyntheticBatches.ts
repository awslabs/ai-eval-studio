// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hooks for fetching synthetic batch data.
 *
 * Exports ``useSyntheticBatches`` (list) and ``useSyntheticBatch`` (single).
 * Both re-fetch on projectId/batchId changes, expose a manual ``refresh()``
 * so approve/reject flows can reconcile after a mutation, and subscribe
 * to ``agent_turn/completed`` + ``job/completed`` so a synthetic batch
 * landing on disk surfaces in the UI without a reload.
 */

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import type { SyntheticBatchDetail, SyntheticBatchSummary } from "@/lib/types";

/** Return value from ``useSyntheticBatches``. */
export interface UseSyntheticBatchesResult {
  /** Fetched batch summaries, or ``null`` while the first fetch is in-flight. */
  data: SyntheticBatchSummary[] | null;
  /** Error from the most recent fetch attempt, or ``null``. */
  error: Error | null;
  /** True while a fetch is in-flight. */
  loading: boolean;
  /** Trigger a manual re-fetch. */
  refresh: () => Promise<void>;
}

/** Return value from ``useSyntheticBatch``. */
export interface UseSyntheticBatchResult {
  /** The batch detail, or ``null`` when no batchId is selected or on error. */
  data: SyntheticBatchDetail | null;
  /** Error from the most recent fetch attempt, or ``null``. */
  error: Error | null;
  /** True while a fetch is in-flight. */
  loading: boolean;
  /** Trigger a manual re-fetch. No-op when ``batchId`` is ``null``. */
  refresh: () => Promise<void>;
}

/**
 * Fetch the list of synthetic batches for a project.
 *
 * @param projectId - Project UUID
 * @returns List data, loading flag, error state, and refresh function
 */
export function useSyntheticBatches(
  projectId: string,
): UseSyntheticBatchesResult {
  const [data, setData] = useState<SyntheticBatchSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const batches = await api.listSyntheticBatches(projectId);
      setData(batches);
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Centralized freshness — synthetic generation runs as an agent turn,
  // so agent_turn/completed is the load-bearing signal. job/completed is
  // kept for symmetry with the rest of the hook family.
  useProjectEvent({ resource: "agent_turn", type: "completed" }, () => { refresh(); }, "state");
  useProjectEvent({ resource: "job", type: "completed" }, () => { refresh(); }, "state");

  return { data, error, loading, refresh };
}

/**
 * Fetch one synthetic batch by id.
 *
 * Short-circuits to a no-op when ``batchId`` is ``null`` — the caller
 * typically tracks the selected batch as ``string | null``, and we'd
 * rather not kick off a request until something is picked.
 *
 * @param projectId - Project UUID
 * @param batchId - Batch identifier, or ``null`` to skip fetching
 * @returns Detail data, loading flag, error state, and refresh function
 */
export function useSyntheticBatch(
  projectId: string,
  batchId: string | null,
): UseSyntheticBatchResult {
  const [data, setData] = useState<SyntheticBatchDetail | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!batchId) {
      setData(null);
      setError(null);
      return;
    }
    setLoading(true);
    try {
      const detail = await api.getSyntheticBatch(projectId, batchId);
      setData(detail);
      setError(null);
    } catch (e) {
      setError(e as Error);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [projectId, batchId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Same freshness wiring as the list hook. ``refresh`` short-circuits
  // when batchId is null, so a fresh event with no selected batch is
  // a cheap no-op rather than a wasted fetch.
  useProjectEvent({ resource: "agent_turn", type: "completed" }, () => { refresh(); }, "state");
  useProjectEvent({ resource: "job", type: "completed" }, () => { refresh(); }, "state");

  return { data, error, loading, refresh };
}
