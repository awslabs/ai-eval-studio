// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for mcp__eval__save_prompt.
 * Shows prompt save status with version detail.
 */

import { Save } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Renders the save_prompt tool call with version detail.
 */
export function SavePromptToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || "Saving prompt...";
  const detail =
    (args?._detail as string) ||
    (args?.version as string) ||
    undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={Save}
      label={label}
      detail={detail}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
