// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API client for communicating with the AI Eval Studio backend.
 * All endpoints are relative to the backend's base URL.
 */

import { getAccessToken } from "./auth";
import { saveBlob } from "./download-utils";
import type { RunReportEnvelope, ExecReportEnvelope } from "./reports/types";
import type {
  ProjectConfig,
  ProjectSummary,
  UploadResponse,
  EvalRunSummary,
  ModelRun,
  CompactModelRun,
  CaseDetail,
  ProjectActivity,
  EvalJob,
  ChatMessage,
  FileNode,
  FileContentResponse,
  HarnessConfig,
  SyntheticBatchSummary,
  SyntheticBatchDetail,
  SyntheticBatchStatusResponse,
  DataSource,
} from "./types";

/**
 * Authenticated `fetch` wrapper. In cloud mode, attaches an
 * `Authorization: Bearer <jwt>` header; in dev mode (`getAccessToken`
 * returns `""`) the header is omitted entirely so the backend's
 * `LocalAuth` path runs.
 *
 * Every request to the backend MUST go through this helper — naked
 * `fetch` calls will lack the auth header and 401 in cloud mode.
 *
 * HTTP caching is disabled for every request: a stored older response
 * would make a deleted eval run reappear after SPA navigation, as the
 * browser serves the cached body before the server sees the request.
 * Enforced over caller-supplied options because every endpoint behind
 * this client returns dynamic per-project state — none is cacheable.
 *
 * @param path - API endpoint path (relative to backend).
 * @param init - Fetch options. Any caller-provided `Authorization`
 *               header is preserved (rare; mostly for tests).
 * @returns The raw `Response` object — caller decides how to parse it.
 */
export async function apiFetch(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const token = await getAccessToken();
  // Build headers: caller-provided headers win, then we add Authorization
  // only if we have a token. Empty token = dev mode = no header.
  const headers = new Headers(init?.headers);
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  // Disable HTTP caching: spread init first, then enforce cache mode.
  // Positioned after the spread so caller-supplied cache options are
  // overridden — every API endpoint is dynamic per-project state.
  return fetch(path, { ...init, headers, cache: "no-store" });
}

/**
 * HTTP error carrying the response status so callers can distinguish
 * "not found" (legitimate empty state) from network/auth/server failures.
 */
export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Percent-encode each segment of a relative path, keeping the separators.
 *
 * The backend route takes a multi-segment path, so the slashes have to survive —
 * but the segments themselves must not. Interpolating a filename raw meant `#`
 * was read as a fragment and `?` as a query string, truncating the request to the
 * parent directory, while a literal `%` was passed through and then decoded
 * server-side, so `a%41b.csv` asked for `aAb.csv`. Spaces and non-ASCII happened
 * to work because the URL serializer encodes them, which is why this looked fine.
 *
 * Filenames reach disk close to verbatim (upload keeps the basename, and the
 * agent can create any name), so none of these characters are hypothetical.
 */
export function encodePathSegments(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

/**
 * Generic HTTP request wrapper with JSON parsing and error handling.
 * Routes through `apiFetch` so every JSON request carries the bearer
 * token in cloud mode.
 *
 * @template T - Expected response type
 * @param path - API endpoint path (relative to backend)
 * @param options - Fetch API options (method, body, headers, etc.)
 * @returns Parsed JSON response
 * @throws ApiError carrying the status and the server's error detail if the
 *   request fails
 */
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await apiFetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, error.detail || res.statusText);
  }
  // 204 No Content returns no body
  if (res.status === 204) return undefined as T;
  return res.json();
}

/**
 * Centralized API client for all backend operations.
 * Each method corresponds to a backend FastAPI endpoint.
 */
export const api = {
  /**
   * Fetch all projects for the current user.
   * @returns Array of project summaries
   */
  listProjects: () => request<ProjectSummary[]>("/api/projects"),

  /**
   * Create a new evaluation project.
   * @param name - Project name
   * @param description - Optional project description
   * @returns Full project configuration
   */
  createProject: (name: string, description = "") =>
    request<ProjectConfig>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),

  /**
   * Fetch complete configuration for a specific project.
   * @param id - Project UUID
   * @returns Full project configuration
   */
  getProject: (id: string) => request<ProjectConfig>(`/api/projects/${id}`),

  /**
   * Delete a project and all associated data.
   * @param id - Project UUID
   * @returns Fetch response (no body expected)
   */
  deleteProject: (id: string) =>
    request<void>(`/api/projects/${id}`, { method: "DELETE" }),

  /**
   * Update mutable project fields: ``agent_model``, ``name``, ``description``.
   *
   * Omit a field to leave it untouched. Passing `description: ""` clears the
   * description — an empty string is a value here, not an omission.
   *
   * @param id - Project UUID
   * @param updates - Partial config (only set fields are applied)
   * @returns The project's full config after the update
   */
  updateProject: (
    id: string,
    updates: {
      agent_model?: "sonnet" | "opus";
      name?: string;
      description?: string;
    },
  ) =>
    request<ProjectConfig>(`/api/projects/${id}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    }),

  /**
   * Upload one or more files (including zips) into the project's
   * ``data/uploads/`` directory. The AI agent sorts them into
   * ``data/raw/`` or ``docs/`` afterwards.
   *
   * @param projectId - Project UUID
   * @param files - One or more File objects to upload in a single request
   * @returns List of project-relative paths written and total bytes
   * @throws Error if upload fails
   */
  uploadFiles: async (
    projectId: string,
    files: File[],
  ): Promise<UploadResponse> => {
    const formData = new FormData();
    for (const file of files) formData.append("files", file);

    // Use `apiFetch` so the bearer token rides along; do NOT set
    // Content-Type — the browser must compute the multipart boundary.
    const res = await apiFetch(`/api/projects/${projectId}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || res.statusText);
    }
    return res.json();
  },

  /**
   * List all eval runs for a project, newest first.
   *
   * @param projectId - Project UUID
   * @returns Array of eval run summaries sorted by timestamp descending
   */
  listEvalRuns: (projectId: string) =>
    request<EvalRunSummary[]>(`/api/projects/${projectId}/eval/runs`),

  /**
   * Get full results for a specific eval run.
   *
   * @param projectId - Project UUID
   * @param runFilename - The _file value from EvalRunSummary
   * @returns Array of model runs with DeepEval TestResult objects
   */
  getEvalRun: (projectId: string, runFilename: string) =>
    request<ModelRun[]>(
      `/api/projects/${projectId}/eval/runs/${runFilename}`,
    ),

  /**
   * Get compact eval run results for low-bandwidth transfer.
   *
   * Returns the same model-run structure as getEvalRun but with slim
   * per-case projections: full text replaced with previews/hashes,
   * metric reasons/thresholds/errors dropped, latency/tokens promoted.
   *
   * @param projectId - Project UUID
   * @param runFilename - The _file value from EvalRunSummary
   * @returns Array of compact model runs
   */
  getEvalRunCompact: (projectId: string, runFilename: string) =>
    request<CompactModelRun[]>(
      `/api/projects/${projectId}/eval/runs/${runFilename}/compact`,
    ),

  /**
   * Get full per-case detail for a single test case across all models.
   *
   * Returns input/expected_output (shared) and per-model actual_output,
   * token/latency stats, and metric reasons. Used for on-demand lazy
   * loading when expanding a specific case row in the Compare panel.
   *
   * @param projectId - Project UUID
   * @param runFilename - The _file value from EvalRunSummary
   * @param caseId - Test case identifier (additional_metadata.id or name)
   * @returns Case detail with input/expected and per-model outputs/metrics
   */
  getEvalCaseDetail: (projectId: string, runFilename: string, caseId: string) =>
    request<CaseDetail>(
      `/api/projects/${projectId}/eval/runs/${runFilename}/cases/${encodeURIComponent(caseId)}`,
    ),

  /**
   * List prompt versions for a project.
   *
   * @param projectId - Project UUID
   * @returns Array of prompt filenames (e.g., ["baseline.md", "v1.md"])
   */
  listPrompts: (projectId: string) =>
    request<string[]>(`/api/projects/${projectId}/prompts`),

  /**
   * Get the single reconciled "what is running" view for a project —
   * optimization loop state + active eval, resolved server-side so the
   * two can't disagree. This is the SPA's one source of truth for
   * project activity.
   * @param projectId - Project UUID
   */
  getProjectActivity: (projectId: string) =>
    request<ProjectActivity>(`/api/projects/${projectId}/activity`),

  /**
   * Get the text content of a specific prompt version.
   * @param projectId - Project UUID
   * @param version - Prompt filename (e.g., "baseline.md")
   */
  getPromptContent: (projectId: string, version: string) =>
    request<{ version: string; content: string; word_count: number; char_count: number }>(
      `/api/projects/${projectId}/prompts/${encodeURIComponent(version)}`,
    ),

  /**
   * Get a specific eval job by ID.
   * @param projectId - Project UUID
   * @param jobId - Eval job UUID
   * @returns Eval job state
   */
  getEvalJob: (projectId: string, jobId: string) =>
    request<EvalJob>(`/api/projects/${projectId}/eval/jobs/${jobId}`),

  /**
   * Cancel a running eval job.
   * @param projectId - Project UUID
   * @param jobId - Eval job UUID to cancel
   * @returns Confirmation with cancelled status
   */
  cancelEvalJob: (projectId: string, jobId: string) =>
    request<{ status: string; job_id: string }>(
      `/api/projects/${projectId}/eval/jobs/${jobId}/cancel`,
      { method: "POST" },
    ),

  /**
   * Cancel the running optimization loop and any active eval job.
   * @param projectId - Project UUID
   * @returns Confirmation with cancelled status
   */
  cancelOptimization: (projectId: string) =>
    request<{ status: string }>(
      `/api/projects/${projectId}/optimization/cancel`,
      { method: "POST" },
    ),

  /**
   * Start a new optimization loop from UI (reads all params from harness config).
   *
   * @param projectId - Project UUID
   * @param body - Optional body with a ``data_source`` selector. Omitted
   *   body defaults to ``"raw"`` server-side.
   */
  startOptimization: (
    projectId: string,
    body?: { data_source?: DataSource },
  ) =>
    request<{ status: string }>(
      `/api/projects/${projectId}/optimization/start`,
      {
        method: "POST",
        body: body ? JSON.stringify(body) : undefined,
      },
    ),

  /**
   * Resume a paused optimization loop (cancelled, plateau, or max_iterations).
   *
   * @param projectId - Project UUID
   * @param body - Optional body with a ``data_source`` selector. Omitted
   *   body defaults to ``"raw"`` server-side.
   */
  resumeOptimization: (
    projectId: string,
    body?: { data_source?: DataSource },
  ) =>
    request<{ status: string; iteration: number }>(
      `/api/projects/${projectId}/optimization/resume`,
      {
        method: "POST",
        body: body ? JSON.stringify(body) : undefined,
      },
    ),

  /**
   * Get chat message history for a project.
   * @param projectId - Project UUID
   * @returns Chat history with messages and interrupted flag
   */
  getChatHistory: (projectId: string) =>
    request<{ messages: ChatMessage[]; interrupted: boolean }>(`/api/projects/${projectId}/chat/history`),

  /**
   * Get the in-flight agent turn for a project, if any. Returns
   * ``null`` when no turn is running. Buffered envelopes ride the WS
   * subscribe-with-replay path, not this endpoint — the SPA only
   * needs the ``turn_id`` to start subscribing.
   * @param projectId - Project UUID
   */
  getActiveTurn: (projectId: string) =>
    request<{
      turn_id: string;
      status: "running" | "completed" | "failed";
    } | null>(`/api/projects/${projectId}/chat/active-turn`),

  /**
   * Start an agent turn for a project. The backend spawns a background
   * task and returns the ``turn_id`` so the SPA can subscribe to its
   * WebSocket envelope stream. ``session_id`` is optional — when set,
   * the SDK resumes the named session; when omitted, the project's
   * persisted session_id (or a fresh one) is used server-side.
   *
   * @param projectId - Project UUID
   * @param message - User message text
   * @param sessionId - Optional SDK session id to resume
   * @returns ``{ turn_id, status }`` where status is "running", "queued", or "busy"
   */
  startChat: (projectId: string, message: string, sessionId?: string) =>
    request<{ turn_id: string; status: string }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        project_id: projectId,
        ...(sessionId ? { session_id: sessionId } : {}),
      }),
    }),

  /**
   * Stop a running agent turn. Best-effort: the backend interrupts the
   * SDK client subprocess; an in-flight tool call may still complete.
   *
   * @param turnId - Agent turn UUID returned by ``startChat``
   * @returns ``{ status }`` ("stopped" or the turn's terminal status)
   */
  stopChatTurn: (turnId: string) =>
    request<{ status: string }>(`/api/chat/stop/${turnId}`, {
      method: "POST",
    }),

  /**
   * List files in a project directory tree.
   * @param projectId - Project UUID
   * @returns File tree structure
   */
  listFiles: (projectId: string) =>
    request<{ tree: FileNode[] }>(`/api/projects/${projectId}/files`),

  /**
   * Get file content by path.
   * @param projectId - Project UUID
   * @param path - Relative file path within the project
   * @returns File content with metadata and optional parsed rows
   */
  getFileContent: (projectId: string, path: string) =>
    request<FileContentResponse>(
      `/api/projects/${projectId}/files/${encodePathSegments(path)}`,
    ),

  /**
   * Download the full runnable harness package as a zip and trigger a
   * browser save. The zip contains everything needed to run the eval
   * locally (``uv sync && uv run python run_eval.py``).
   * @param projectId - Project UUID
   * @throws ApiError if the download request fails
   */
  downloadHarness: async (projectId: string): Promise<void> => {
    const res = await apiFetch(`/api/projects/${projectId}/harness/download`);
    if (!res.ok) {
      throw new ApiError(res.status, `Download failed (${res.status})`);
    }
    // Derive the filename from Content-Disposition when present, else a
    // sensible default keyed by project id.
    const disposition = res.headers.get("Content-Disposition") ?? "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match?.[1] ?? `harness-${projectId}.zip`;

    const blob = await res.blob();
    // Route through the shared save helper so the harness zip offers the same
    // "choose where to save" picker as every other download (anchor fallback
    // for non-Chromium).
    await saveBlob(blob, filename, "application/zip");
  },

  /**
   * Fetch the per-run report JSON (facts + narrative).
   *
   * The frontend renders this with a React component; theme inherits
   * from the dashboard, no theme_vars to ship.
   *
   * @param projectId - Project UUID
   * @param runFilename - Optional run file; latest run when omitted
   * @returns ``{schema_version, scope, facts, narrative, generated_at}``
   * @throws ApiError 404 when the project / run is missing.
   */
  getRunReport: async (
    projectId: string,
    runFilename?: string,
  ): Promise<RunReportEnvelope> => {
    const res = await apiFetch(`/api/projects/${projectId}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: runFilename }),
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Report failed (${res.status})`);
    }
    return (await res.json()) as RunReportEnvelope;
  },

  /**
   * Fetch the executive-summary report JSON.
   *
   * Always summarizes the trophy-winning run + the journey across versions.
   * Frontend renders with React (theme inherits from the dashboard).
   *
   * @param projectId - Project UUID
   * @returns ``{schema_version, scope, facts, narrative, generated_at}``
   * @throws ApiError 404 when the project has no runs.
   */
  getExecutiveReport: async (
    projectId: string,
  ): Promise<ExecReportEnvelope> => {
    const res = await apiFetch(`/api/projects/${projectId}/report/executive`, {
      method: "POST",
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Report failed (${res.status})`);
    }
    return (await res.json()) as ExecReportEnvelope;
  },

  /**
   * Start regenerating the per-run report's narrative sidecar.
   *
   * The backend starts the report-narrator agent and answers 202 without
   * waiting for it, because a narration outlasts the CDN's origin timeout.
   * The outcome arrives as a ``report`` project event: ``updated`` once the
   * new sidecar is on disk, ``failed`` when none was written (the lean
   * fallback still renders).
   *
   * @param projectId - Project UUID
   * @param run - Run filename
   */
  regenerateReport: async (projectId: string, run: string): Promise<void> => {
    const res = await apiFetch(`/api/projects/${projectId}/report/regenerate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run }),
    });
    if (!res.ok) {
      throw new ApiError(res.status, `Regenerate failed (${res.status})`);
    }
  },

  /**
   * Start regenerating the executive report's narrative sidecar.
   * @see regenerateReport
   * @param projectId - Project UUID
   */
  regenerateExecutiveReport: async (projectId: string): Promise<void> => {
    const res = await apiFetch(
      `/api/projects/${projectId}/report/executive/regenerate`,
      { method: "POST" },
    );
    if (!res.ok) {
      throw new ApiError(res.status, `Regenerate failed (${res.status})`);
    }
  },

  /**
   * Get parsed harness configuration for a project.
   * @param projectId - Project UUID
   * @returns Harness config with models, metrics, data file, judge model,
   *   prompt version — or null when the project has no harness yet (a
   *   normal state for new projects, served as 200 so browsers don't log
   *   a network error on every page load).
   */
  getHarnessConfig: (projectId: string) =>
    request<HarnessConfig | null>(`/api/projects/${projectId}/harness/config`),

  /**
   * List every synthetic data batch for a project, newest first.
   *
   * @param projectId - Project UUID
   * @returns Array of batch summaries sorted by timestamp descending
   */
  listSyntheticBatches: (projectId: string) =>
    request<SyntheticBatchSummary[]>(
      `/api/projects/${projectId}/synthetic/batches`,
    ),

  /**
   * Get the full manifest + enriched per-case rows for one batch.
   *
   * @param projectId - Project UUID
   * @param batchId - Timestamped batch identifier
   * @returns Batch detail with rows keyed by CSV column
   */
  getSyntheticBatch: (projectId: string, batchId: string) =>
    request<SyntheticBatchDetail>(
      `/api/projects/${projectId}/synthetic/batches/${batchId}`,
    ),

  /**
   * Approve synthetic cases, either explicit indices or every pending case.
   *
   * @param projectId - Project UUID
   * @param batchId - Timestamped batch identifier
   * @param body - Either ``indices`` (non-empty) or ``approve_all: true``
   * @returns Updated counts plus the raw cases array
   */
  approveSyntheticCases: (
    projectId: string,
    batchId: string,
    body: { indices?: number[]; approve_all?: boolean },
  ) =>
    request<SyntheticBatchStatusResponse>(
      `/api/projects/${projectId}/synthetic/batches/${batchId}/approve`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  /**
   * Reject synthetic cases by explicit indices.
   *
   * @param projectId - Project UUID
   * @param batchId - Timestamped batch identifier
   * @param body - Non-empty ``indices`` list
   * @returns Updated counts plus the raw cases array
   */
  rejectSyntheticCases: (
    projectId: string,
    batchId: string,
    body: { indices: number[] },
  ) =>
    request<SyntheticBatchStatusResponse>(
      `/api/projects/${projectId}/synthetic/batches/${batchId}/reject`,
      { method: "POST", body: JSON.stringify(body) },
    ),
};
