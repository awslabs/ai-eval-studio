// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for mcp__eval__analyze_failures.
 * Shows failure analysis status as a compact one-line row.
 */

import { AlertTriangle } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Renders the analyze_failures tool call as a compact row.
 */
export function AnalyzeFailuresToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || "Analyzing failures...";
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={AlertTriangle}
      label={label}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
