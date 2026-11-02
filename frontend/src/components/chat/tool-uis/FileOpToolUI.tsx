// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared tool UI for file operation tools: Read, Write, Edit, Glob, Grep.
 * Shows appropriate icon per tool and file path detail as a compact one-line row.
 */

import { FileText, FilePen, Search } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS, cleanToolName } from "@/lib/tool-labels";
import { formatPath } from "@/lib/format-path";
import type { ToolUIProps } from "./index";

/**
 * Select the appropriate icon based on the file operation tool name.
 */
function getFileOpIcon(toolName: string): React.ComponentType<{ className?: string }> {
  switch (toolName) {
    case "Write":
    case "Edit":
      return FilePen;
    case "Glob":
    case "Grep":
      return Search;
    default:
      return FileText;
  }
}

/**
 * Renders file operation tool calls as a compact row with file path detail.
 */
export function FileOpToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || cleanToolName(toolName);
  const rawPath =
    (args?._detail as string) ||
    (args?.file_path as string) ||
    (args?.path as string) ||
    undefined;
  // Collapse noisy absolute paths; full path survives on hover via `title`.
  const detail = rawPath ? formatPath(rawPath) : undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={getFileOpIcon(toolName)}
      label={label}
      detail={detail}
      detailTitle={rawPath}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
