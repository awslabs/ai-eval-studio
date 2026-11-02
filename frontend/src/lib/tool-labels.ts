// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Human-readable labels for tool names shown in the chat UI.
 * Single source of truth — imported by ChatRuntimeProvider (streaming status)
 * and the chat tool UIs in components/chat/tool-uis (inline tool call pills).
 */

/** Map tool names to human-readable labels for display. */
export const TOOL_LABELS: Record<string, string> = {
  "mcp__eval__describe_data": "Analyzing dataset…",
  "mcp__eval__dispatch_eval": "Dispatching evaluation…",
  "mcp__eval__check_eval_status": "Checking eval progress…",
  "mcp__eval__cancel_eval": "Cancelling eval…",
  "mcp__eval__list_runs": "Listing runs…",
  "mcp__eval__get_results": "Loading results…",
  "mcp__eval__get_case_details": "Loading case details…",
  "mcp__eval__save_prompt": "Saving prompt…",
  "mcp__eval__analyze_failures": "Analyzing failures…",
  "mcp__eval__dispatch_optimization": "Starting optimization loop…",
  "mcp__eval__record_optimization_iteration": "Recording iteration…",
  "mcp__eval__resume_job": "Resuming job…",
  "mcp__eval__list_bedrock_models": "Looking up models…",
  "mcp__eval__generate_synthetic": "Generating synthetic data…",
  "mcp__eval__list_synthetic_batches": "Listing synthetic batches…",
  "mcp__eval__approve_synthetic_cases": "Recording approvals…",
  "mcp__eval__scaffold_harness": "Scaffolding harness…",
  "mcp__eval__validate_harness": "Validating harness…",
  "mcp__eval__dispatch_harness_build": "Building harness…",
  "mcp__eval__get_project_status": "Reading project status…",
  "mcp__eval__list_upload": "Listing uploads…",
  "mcp__eval__finalize_upload": "Finalizing upload…",
  "mcp__eval__move_file": "Moving file…",
  "mcp__eval__refresh_pricing": "Refreshing pricing…",
  "mcp__eval__request_user_input": "Awaiting your input…",
  "subagent:data-analyzer": "Analyzing data…",
  "subagent:harness-builder": "Building harness…",
  "subagent:prompt-optimizer": "Optimizing prompt…",
  // Generic labels for the SDK's built-in delegation tool before the
  // subagent_type arg has finished streaming. Once the name is rewritten
  // to "subagent:<type>" at tool_end, the specific label above takes over.
  Task: "Delegating to subagent…",
  Agent: "Delegating to subagent…",
  Read: "Reading file…",
  Write: "Writing file…",
  Edit: "Editing file…",
  Bash: "Running command…",
  Glob: "Searching files…",
  Grep: "Searching content…",
  WebSearch: "Searching the web…",
  WebFetch: "Fetching web page…",
  ToolSearch: "Searching tools…",
};

/**
 * Tools whose raw args should never be shown as inline detail.
 * These tools have internal/machine-readable arguments (selectors, queries)
 * that don't make sense in the UI.
 */
export const TOOL_DETAIL_BLOCKLIST = new Set([
  "ToolSearch",
]);

/**
 * Clean a raw tool name into a human-readable label as a fallback
 * when ``TOOL_LABELS`` lacks a curated entry. New ``@tool(...)``
 * registrations always lag the label map; without this every status-
 * bar / tool-bubble call site would render raw ``mcp__eval__*`` names
 * for unmapped tools.
 *
 * Strips ``mcp__eval__``, replaces underscores with spaces, and
 * title-cases each word. Subagent delegations (``subagent:<type>``)
 * become ``Running <type>…`` so new subagents work out of the box.
 *
 * Always pair this with ``TOOL_LABELS`` lookup —
 * ``TOOL_LABELS[name] || cleanToolName(name)`` — so curated entries
 * with verb-form labels ("Saving prompt…") still win when present.
 */
export function cleanToolName(name: string): string {
  if (name.startsWith("subagent:")) {
    const agentType = name.slice("subagent:".length);
    return agentType ? `Running ${agentType}…` : "Running subagent…";
  }
  return name
    .replace(/^mcp__eval__/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
