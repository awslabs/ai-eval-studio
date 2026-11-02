// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for ``mcp__eval__generate_synthetic``.
 *
 * Shows the generation strategy (and num_cases when supplied) inline
 * while the tool is running. When the tool finishes, surfaces a "Review
 * batch" button that deep links into the Data tab with the returned
 * ``batch_id`` (best-effort extraction from the agent tool result).
 */

import { useMemo } from "react";
import { Sparkles, ArrowRight } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { Button } from "@/components/ui/button";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Extract the batch_id from a raw tool result string.
 *
 * Tries a JSON parse first (the synthesizer tool returns an mcp_json
 * payload) and falls back to a loose ``batch_id`` token scan so a
 * plain-text confirmation still linkifies.
 */
function extractBatchId(result: unknown): string | null {
  if (typeof result !== "string" || !result) return null;

  // Fast path: full JSON payload.
  try {
    const parsed = JSON.parse(result);
    if (parsed && typeof parsed === "object" && "batch_id" in parsed) {
      const bid = (parsed as { batch_id?: unknown }).batch_id;
      if (typeof bid === "string" && bid) return bid;
    }
  } catch {
    // Fall through to the regex path.
  }

  // Loose match — works whether the agent wrapped the payload in prose
  // or a markdown fence.
  const match = result.match(/"batch_id"\s*:\s*"([^"]+)"|batch_id[:=]\s*([^\s"',}]+)/);
  if (match) {
    return match[1] || match[2] || null;
  }
  return null;
}

/**
 * Build the detail string shown next to the tool label.
 * Prefers an explicit ``_detail`` override, then strategy + num_cases.
 */
function buildDetail(args?: Record<string, unknown>): string | undefined {
  if (args?._detail && typeof args._detail === "string") return args._detail;
  const strategy = typeof args?.strategy === "string" ? args.strategy : undefined;
  const numCases = typeof args?.num_cases === "number" ? args.num_cases : undefined;
  if (strategy && numCases) return `${strategy} · ${numCases} cases`;
  return strategy;
}

/**
 * Renders the generate_synthetic tool call with a deep-link button.
 */
export function GenerateSyntheticToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const label = TOOL_LABELS[toolName] || "Generating synthetic data...";
  const detail = buildDetail(args);
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  // Only attempt id extraction once per result change.
  const batchId = useMemo(() => (isDone ? extractBatchId(result) : null), [isDone, result]);

  const openBatch = () => {
    const next = new URLSearchParams(location.search);
    next.set("tab", "data");
    if (batchId) next.set("batch", batchId);
    else next.delete("batch");
    navigate({ pathname: location.pathname, search: `?${next.toString()}` });
  };

  return (
    <>
      <ToolUIWrapper
        icon={Sparkles}
        label={label}
        detail={detail}
        count={count}
        isDone={isDone}
        isError={isError}
      />
      {/* The deep-link to the generated batch stays visible below the row
          (not hidden behind a disclosure) so the user can act on it directly. */}
      {isDone && (
        <div className="flex items-center gap-2 px-1.5 pb-1">
          <Button size="xs" variant="outline" onClick={openBatch}>
            Review batch
            <ArrowRight className="size-3" />
          </Button>
          {batchId && (
            <span className="text-muted-foreground/60 font-mono text-2xs">
              {batchId}
            </span>
          )}
        </div>
      )}
    </>
  );
}
