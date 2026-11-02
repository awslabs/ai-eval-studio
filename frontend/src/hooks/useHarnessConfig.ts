// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hook for fetching and caching the harness configuration for a project.
 * Re-fetches when projectId changes, when refresh() is called, and on
 * job/agent_turn/filesystem events from the WS bus so consumers see harness
 * changes (model swaps, set_models, scaffold) land live.
 */

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import type { HarnessConfig } from "@/lib/types";

/** Return value from the useHarnessConfig hook. */
export interface UseHarnessConfigResult {
  /** Parsed harness config, or null while loading or if not yet configured. */
  config: HarnessConfig | null;
  /** Error from the most recent fetch attempt, or null. */
  error: Error | null;
  /** True while a fetch is in-flight. */
  loading: boolean;
  /** Trigger a manual re-fetch. */
  refresh: () => Promise<void>;
}

/**
 * Fetch the harness configuration for a project, with manual refresh support.
 *
 * @param projectId - Project UUID to fetch harness config for
 * @returns Config, loading flag, error state, and refresh function
 */
export function useHarnessConfig(projectId: string): UseHarnessConfigResult {
  const [config, setConfig] = useState<HarnessConfig | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getHarnessConfig(projectId);
      setConfig(data);
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Centralized freshness — every consumer (SummaryTab, future) inherits
  // WS-driven refresh from one bus + one hook. The agent edits config.py
  // mid-turn (set_models / scaffold_harness), so agent_turn/completed is
  // the dominant signal; filesystem/changed catches direct edits; the
  // job/completed subscription mirrors the pattern used by useEvalRuns.
  useProjectEvent({ resource: "filesystem", type: "changed" }, () => { refresh(); }, "state");
  useProjectEvent({ resource: "agent_turn", type: "completed" }, () => { refresh(); }, "state");
  useProjectEvent({ resource: "job", type: "completed" }, () => { refresh(); }, "state");

  return { config, error, loading, refresh };
}
