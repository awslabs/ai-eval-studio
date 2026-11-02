// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for mcp__eval__run_eval.
 * Shows evaluation run status as a compact one-line row with model name detail.
 */

import { FlaskConical } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Renders the run_eval tool call as a compact row with model name detail.
 */
export function RunEvalToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || "Running evaluation...";
  const detail =
    (args?._detail as string) ||
    (args?.models as string) ||
    undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={FlaskConical}
      label={label}
      detail={detail}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
