// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for mcp__eval__describe_data.
 * Shows dataset analysis status as a compact one-line row with filename detail.
 */

import { Database } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Renders the describe_data tool call as a compact row with filename detail.
 */
export function DescribeDataToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || "Analyzing dataset...";
  const detail =
    (args?._detail as string) ||
    (args?.filename as string) ||
    undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={Database}
      label={label}
      detail={detail}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
