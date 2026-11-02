// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tool UI registry and shared types for tool-specific chat components.
 * Maps tool names to their specialized UI components, with a default fallback.
 */

import type React from "react";
import { RunEvalToolUI } from "./RunEvalToolUI";
import { DescribeDataToolUI } from "./DescribeDataToolUI";
import { SavePromptToolUI } from "./SavePromptToolUI";
import { AnalyzeFailuresToolUI } from "./AnalyzeFailuresToolUI";
import { GenerateSyntheticToolUI } from "./GenerateSyntheticToolUI";
import { FileOpToolUI } from "./FileOpToolUI";
import { BashToolUI } from "./BashToolUI";
import { DefaultToolUI as DefaultToolUIComponent } from "./DefaultToolUI";

/** Props passed to every tool UI component. */
export interface ToolUIProps {
  /** Tool identifier (e.g., "mcp__eval__run_eval", "Read", "Bash") */
  toolName: string;
  /** Tool input arguments (may include _detail and _count from collapsing) */
  args?: Record<string, unknown>;
  /** Tool result (undefined while running, string when done) */
  result?: unknown;
  /**
   * Whether the tool reported an error. Sourced from the structured
   * ``isError`` flag on the assistant-ui tool-call part — never inferred by
   * matching words in the result text.
   */
  isError?: boolean;
}

/** Registry mapping tool names to their specialized UI components. */
export const TOOL_UI_REGISTRY: Record<string, React.ComponentType<ToolUIProps>> = {
  "mcp__eval__run_eval": RunEvalToolUI,
  "mcp__eval__describe_data": DescribeDataToolUI,
  "mcp__eval__save_prompt": SavePromptToolUI,
  "mcp__eval__analyze_failures": AnalyzeFailuresToolUI,
  "mcp__eval__generate_synthetic": GenerateSyntheticToolUI,
  Read: FileOpToolUI,
  Write: FileOpToolUI,
  Edit: FileOpToolUI,
  Glob: FileOpToolUI,
  Grep: FileOpToolUI,
  Bash: BashToolUI,
};

/** Default fallback component for unrecognized tools. */
export const DefaultToolUI = DefaultToolUIComponent;
