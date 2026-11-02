// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Prompts tab wrapper for the project page.
 * Fetches prompt file list and renders the existing PromptsPanel component.
 * Refetches when the agent reports a filesystem change or an eval job
 * completes — both can create new prompt versions.
 */

import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { PromptsPanel } from "@/components/eval/PromptsPanel";
import { useProjectEvent } from "@/hooks/useBackendEvents";

/** Props for the PromptsTab component. */
interface PromptsTabProps {
  /** Project UUID for fetching prompt files */
  projectId: string;
}

/**
 * Wrapper that fetches prompt files and renders PromptsPanel.
 * Re-fetches on project bus notifications that imply new prompt files.
 *
 * @param projectId - Project UUID for the prompts API call
 */
export function PromptsTab({ projectId }: PromptsTabProps) {
  const [promptFiles, setPromptFiles] = useState<string[]>([]);
  /* Distinguishes "still fetching" from "fetched, project has no
   * prompts" so a refresh doesn't flash PromptsPanel's "No prompt
   * files yet" empty state for the round-trip duration. Background
   * refreshes (filesystem-changed events) skip the loader to avoid
   * flickering the panel out and back in. */
  const [initialLoading, setInitialLoading] = useState(true);

  /** Fetch the list of prompt files. ``showLoading`` should be true on
   *  the first fetch only — subsequent event-driven refreshes are
   *  silent so the panel doesn't blink. */
  const fetchPrompts = useCallback((showLoading: boolean) => {
    if (showLoading) setInitialLoading(true);
    api.listPrompts(projectId)
      .then(setPromptFiles)
      .catch(() => {})
      .finally(() => {
        if (showLoading) setInitialLoading(false);
      });
  }, [projectId]);

  useEffect(() => {
    fetchPrompts(true);
  }, [fetchPrompts]);

  /* New prompt files land when save_prompt runs (filesystem changed)
   * or when the optimizer loop writes a new version mid-iteration
   * (agent turn completed / eval job completed both signal "look
   * again"). Subscribing to all three keeps the list fresh without
   * a timer. Pass ``false`` so background refreshes don't flash the
   * loader. */
  const refreshSilently = useCallback(() => fetchPrompts(false), [fetchPrompts]);
  useProjectEvent({ resource: "filesystem", type: "changed" }, refreshSilently, "state");
  useProjectEvent({ resource: "agent_turn", type: "completed" }, refreshSilently, "state");
  useProjectEvent({ resource: "eval_job", type: "completed" }, refreshSilently, "state");
  useProjectEvent({ resource: "job", type: "completed" }, refreshSilently, "state");

  if (initialLoading) {
    return (
      <div className="flex items-center justify-center py-8 text-muted-foreground">
        <Loader2 className="size-4 animate-spin mr-2" />
        <span className="text-sm">Loading prompts…</span>
      </div>
    );
  }

  return <PromptsPanel projectId={projectId} promptFiles={promptFiles} className="flex-1 min-h-0 flex flex-col" />;
}
