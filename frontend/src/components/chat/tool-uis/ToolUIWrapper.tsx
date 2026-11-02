// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared row for all tool-specific UI components.
 * Renders a compact one-line row with icon, label, detail, count, and status.
 * The row is intentionally non-expandable: a tool call shows what it did at a
 * glance (icon + label + detail) and nothing more, so a run of tool calls stays
 * a scannable list rather than a nest of disclosures.
 */

import { Loader2, Check, X } from "lucide-react";

interface ToolUIWrapperProps {
  /** Icon to show (lucide component) */
  icon: React.ComponentType<{ className?: string }>;
  /** One-line label (e.g., "Running evaluation...") */
  label: string;
  /** Optional detail text (e.g., file path, model name) */
  detail?: string;
  /** Full-text tooltip for the detail — useful when ``detail`` is a
   *  shortened / formatted version of a longer string (e.g. an absolute
   *  file path collapsed to project-relative). Hover reveals the original. */
  detailTitle?: string;
  /** Repeat count for collapsed tool calls */
  count?: number;
  /** Whether the tool has finished (result is defined) */
  isDone: boolean;
  /** Whether tool errored */
  isError?: boolean;
}

/**
 * Compact, non-expandable tool call row: icon, label, detail, count, status.
 */
export function ToolUIWrapper({
  icon: Icon,
  label,
  detail,
  detailTitle,
  count,
  isDone,
  isError,
}: ToolUIWrapperProps) {
  return (
    <div className="py-0.5">
      <div className="flex items-center gap-1.5 px-1.5 py-1 text-xs text-muted-foreground">
        {/* Tool icon */}
        <Icon className="size-3 shrink-0 text-muted-foreground/60" />

        {/* Label */}
        <span>{label}</span>

        {/* Detail text — title attribute reveals full original when the
            displayed string has been shortened (e.g. project-relative path). */}
        {detail && (
          <span
            className="text-muted-foreground/60 truncate max-w-[300px] font-mono text-2xs"
            title={detailTitle || detail}
          >
            {detail}
          </span>
        )}

        {/* Repeat count */}
        {count && count > 1 && (
          <span className="text-muted-foreground/50 font-mono text-2xs">
            x{count}
          </span>
        )}

        {/* Status indicator */}
        {isError ? (
          <X className="size-3 text-error shrink-0" />
        ) : isDone ? (
          <Check className="size-3 text-success shrink-0" />
        ) : (
          <Loader2 className="size-3 animate-spin shrink-0" />
        )}
      </div>
    </div>
  );
}
