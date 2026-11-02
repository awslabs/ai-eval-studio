// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Default fallback tool UI for any unrecognized tool.
 * Shows a wrench icon and tool label (from registry or cleaned name) as a
 * compact one-line row.
 */

import { Wrench } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS, cleanToolName } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Renders any unrecognized tool as a compact row.
 */
export function DefaultToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || cleanToolName(toolName);
  const detail = args?._detail as string | undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={Wrench}
      label={label}
      detail={detail}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
