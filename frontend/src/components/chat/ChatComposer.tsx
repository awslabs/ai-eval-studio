// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat message input composer with send/cancel controls.
 * Uses assistant-ui primitives for automatic state management.
 */

import { useEffect, useState } from "react";
import { ComposerPrimitive, useThread } from "@assistant-ui/react";
import { SendHorizontal, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ModelSelector } from "@/components/assistant-ui/model-selector";
import { api } from "@/lib/api";
import { useChatProjectId } from "./ChatRuntimeProvider";

/** Claude tier aliases offered in the UI. Haiku is available only by
 * editing ``schemas.py`` — the Bedrock-appropriate pair for both chat
 * orchestration and subagent code-gen is Sonnet + Opus. */
const MODELS = [
  { id: "sonnet", name: "Sonnet", description: "Balanced quality and cost" },
  { id: "opus", name: "Opus", description: "Highest quality, slower" },
];

/** Message composer with expandable textarea and context-aware send/stop button. */
export function ChatComposer() {
  const isRunning = useThread((t) => t.isRunning);
  const projectId = useChatProjectId();
  // Opus matches the backend's default for a new project.
  const [agentModel, setAgentModel] = useState<"sonnet" | "opus">("opus");

  /* Load the current project's saved agent_model on mount. If the stored
   * value isn't one of the tier aliases (config.json can hold an explicit
   * Bedrock ID), we leave the picker at the Opus default — the user can
   * switch explicitly which writes the alias back. */
  useEffect(() => {
    if (!projectId) return;
    let cancelled = false;
    api.getProject(projectId)
      .then((config) => {
        if (cancelled) return;
        if (config.agent_model === "sonnet" || config.agent_model === "opus") {
          setAgentModel(config.agent_model);
        }
      })
      .catch(() => { /* ignore — default Opus stays */ });
    return () => { cancelled = true; };
  }, [projectId]);

  /** Persist the new choice to config.json. The backend reads it on the
   * next turn via project.agent_model. */
  const handleModelChange = (next: string) => {
    if (next !== "sonnet" && next !== "opus") return;
    // Optimistically apply, capturing the prior value so we can roll back if
    // the persist fails (leaving the UI on a model the backend didn't accept).
    const previous = agentModel;
    setAgentModel(next);
    if (!projectId) return;
    api.updateProject(projectId, { agent_model: next })
      .catch(() => setAgentModel(previous));
  };

  return (
    <div className="border-t p-3 shrink-0">
      <ComposerPrimitive.Root className="rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring">
        <ComposerPrimitive.Input
          placeholder="Describe your evaluation goal..."
          rows={4}
          className="w-full resize-none bg-transparent px-3 pt-3 pb-1 text-sm placeholder:text-muted-foreground focus-visible:outline-none disabled:opacity-50 min-h-[5.5rem]"
        />
        <div className="flex items-center justify-between px-2 pb-2 gap-2">
          <ModelSelector
            models={MODELS}
            value={agentModel}
            onValueChange={handleModelChange}
            variant="ghost"
            size="sm"
          />
          {isRunning ? (
            <ComposerPrimitive.Cancel asChild>
              <Button
                size="icon"
                variant="destructive"
                className="h-7 w-7 shrink-0"
                title="Stop generation"
              >
                <Square className="h-3.5 w-3.5" />
              </Button>
            </ComposerPrimitive.Cancel>
          ) : (
            <ComposerPrimitive.Send asChild>
              <Button size="icon" className="h-7 w-7 shrink-0" title="Send message">
                <SendHorizontal className="h-3.5 w-3.5" />
              </Button>
            </ComposerPrimitive.Send>
          )}
        </div>
      </ComposerPrimitive.Root>
    </div>
  );
}
