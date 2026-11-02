// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI for the Bash tool.
 * Shows terminal icon and command detail as a compact one-line row.
 */

import { Terminal } from "lucide-react";
import { ToolUIWrapper } from "./ToolUIWrapper";
import { TOOL_LABELS } from "@/lib/tool-labels";
import type { ToolUIProps } from "./index";

/**
 * Pick a short, human-readable command summary from a raw shell command.
 *
 * The first program-like token is usually the most informative signal at a
 * glance (``uv``, ``pytest``, ``python``). For long pipelines or scripts the
 * full command lives in the title tooltip, so the inline detail can stay
 * terse. Returns undefined when we can't extract anything meaningful.
 */
function summarizeCommand(cmd: string): string | undefined {
  const trimmed = cmd.trim();
  if (!trimmed) return undefined;
  // Strip a leading "$ " prompt if the backend already provided one.
  const body = trimmed.startsWith("$ ") ? trimmed.slice(2) : trimmed;
  // First whitespace-delimited token is typically the executable.
  const first = body.split(/\s+/, 1)[0];
  return first || undefined;
}

/**
 * Renders Bash tool calls as a compact row with command detail.
 */
export function BashToolUI({ toolName, args, result, isError }: ToolUIProps) {
  const label = TOOL_LABELS[toolName] || "Running command...";
  const rawCommand =
    (args?._detail as string) ||
    (typeof args?.command === "string" ? (args.command as string) : undefined);
  const detail = rawCommand ? summarizeCommand(rawCommand) : undefined;
  const count = (args?._count as number) || undefined;
  const isDone = result !== undefined;

  return (
    <ToolUIWrapper
      icon={Terminal}
      label={label}
      detail={detail}
      detailTitle={rawCommand}
      count={count}
      isDone={isDone}
      isError={isError}
    />
  );
}
