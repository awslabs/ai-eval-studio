// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Type definitions for AI Eval Studio API data models.
 * These interfaces match the backend FastAPI response schemas.
 */

/**
 * Data-source selector passed to the optimization/eval start endpoints.
 *
 * - ``raw``: Use the harness-configured dataset (or its train split) as-is.
 * - ``synthetic``: Use only approved synthetic cases.
 * - ``merged``: Use raw + approved synthetic cases concatenated.
 */
export type DataSource = "raw" | "synthetic" | "merged";

/**
 * Lightweight project information for list views.
 * Enriched fields are computed server-side from the project filesystem.
 */
export interface ProjectSummary {
  /** Unique project identifier (UUID) */
  id: string;
  /** User-defined project name */
  name: string;
  /** Optional project description */
  description: string;
  /** ISO 8601 timestamp of project creation */
  created_at: string;
  /** Most recent file modification time (ISO 8601) in the project */
  last_activity?: string | null;
  /** Number of unique model IDs across eval runs */
  model_count?: number;
  /** Pass rate from the most recent eval run (0.0-1.0) */
  latest_pass_rate?: number | null;
  /** Whether an eval is running for this project right now. Sourced from the
   *  job tracker, the same place `GET /api/projects/{id}/activity` reports
   *  from, so the list and the project page agree. */
  is_running?: boolean;
}

/**
 * Complete project configuration including metadata.
 * Target/judge model selection is in the harness config, not here.
 */
export interface ProjectConfig {
  /** Unique project identifier (UUID) */
  id: string;
  /** User-defined project name */
  name: string;
  /** Optional project description */
  description: string;
  /** ISO 8601 timestamp of project creation */
  created_at: string;
  /** Bedrock model ID for the chat agent */
  agent_model: string;
  /** Current active session ID if any */
  session_id: string | null;
}

/**
 * Discriminated union for ordered content parts in a chat message.
 * These map to assistant-ui's native part types:
 * - text → assistant-ui TextMessagePart
 * - thinking → assistant-ui ReasoningMessagePart
 * - tool → assistant-ui ToolCallMessagePart
 */
export type ChatMessagePart =
  | { type: "text"; text: string }
  | { type: "thinking"; text: string; truncated: boolean }
  | {
      type: "tool";
      name: string;
      input: Record<string, unknown> | null;
      result_preview: string | null;
      status: "ok" | "error" | "running";
      truncated: boolean;
    };

/**
 * A single message in a chat conversation with ordered parts.
 * The ``parts`` field is the source of truth for message content; it
 * captures text/thinking/tool blocks IN SOURCE ORDER as delivered by
 * the backend mapper (reload path) and assembled by the streaming path.
 * Both paths produce byte-identical parts so the live view and the
 * reloaded view are guaranteed identical.
 */
export interface ChatMessage {
  /** Unique message identifier (UUID or "streaming" for in-progress) */
  id: string;
  /** Message author role */
  role: "user" | "assistant" | "system";
  /** Ordered content parts (text/thinking/tool) */
  parts: ChatMessagePart[];
  /** ISO 8601 timestamp of message creation */
  created_at: string;
  /**
   * Turn that produced this message, set by the backend on a turn's trailing
   * assistant reply and on its stopped/failed marker.
   *
   * The reconcile guard needs it because locally-streamed messages carry a
   * client-minted id, so there is otherwise no way to ask "did THIS turn's
   * reply reach the transcript?" — only to compare counts, which cannot
   * distinguish a marker from a reply or say which of two in-flight turns
   * landed. Optional: messages written before the field existed lack it, and
   * absent must read as "not this turn".
   */
  turn_id?: string;
}

/**
 * Agent connection/streaming status for UI feedback.
 */
export interface AgentStatusInfo {
  state: "connecting" | "streaming" | "tool_running" | "idle";
  /** Optional detail like tool name */
  detail?: string;
}

/**
 * Response from the multi-file upload endpoint.
 * The frontend refetches the file tree after a successful upload and
 * doesn't consume these fields directly today; they're surfaced only
 * for logging / toast copy.
 */
export interface UploadResponse {
  /** Project-relative paths of every file written to disk */
  files: string[];
  /** Total bytes written (archive + extracted contents) */
  total_bytes: number;
}

/* ================================================================
 * DeepEval-native types — these match the backend response format
 * ================================================================ */

/** DeepEval MetricData — result for a single metric on a single test case. */
export interface MetricData {
  name: string;
  threshold: number;
  success: boolean;
  score: number | null;
  reason: string | null;
  evaluation_cost?: number | null;
  evaluation_model?: string | null;
  strict_mode?: boolean;
  error?: string | null;
}

/** DeepEval TestResult — result for a single test case. */
export interface TestResult {
  name: string;
  success: boolean;
  input: string;
  actual_output: string;
  expected_output: string;
  context: string[] | null;
  retrieval_context: string[] | null;
  additional_metadata: Record<string, unknown> | null;
  metrics_data: MetricData[];
}

/** Hyperparameters block written by the harness/runner per model run.
 *
 * ``model`` / ``endpoint`` / ``region`` are the canonical routing
 * identity — used as keys/labels everywhere. */
export interface RunHyperparameters {
  /** Full Bedrock model id, verbatim (e.g. ``anthropic.claude-opus-4-8``,
   *  ``xai.grok-4.3``). The canonical key for grouping/joining. */
  model: string;
  /** Endpoint family the run executed against. */
  endpoint: "runtime" | "mantle";
  /** AWS region the call was signed for. */
  region: string;
  /** Prompt version stem (no extension). */
  prompt_version: string;
  /** Sampling temperature configured for the run (may have been
   *  dropped at runtime if the model rejected it — see
   *  bedrock_runtime_model.py response-driven self-correction). */
  temperature: number;
  /** Max-tokens cap configured for generation. */
  max_tokens: number;
}

/** A single model's eval run with DeepEval results and metadata wrapper. */
export interface ModelRun {
  test_run_id: string;
  timestamp: string;
  hyperparameters: RunHyperparameters;
  results: TestResult[];
  /** Target-model spend for this run, summed by the harness from the
   *  per-case costs. ``null`` when the model has no pricing entry — the
   *  known subset is NOT presented as the whole spend. */
  generation_cost_usd: number | null;
  /** Judge (LLM-as-a-judge) spend for this run. ``null`` when no judge
   *  ran; ``0`` when a judge ran and cost nothing measurable. */
  judge_cost_usd: number | null;
}

/** Compact metric result — excludes reason/threshold/error for low-bandwidth transfer. */
export interface CompactMetric {
  name: string;
  success: boolean;
  score: number | null;
}

/** Compact test result — slim per-case projection for grid display without full text. */
export interface CompactTestResult {
  name: string;
  case_id: string;
  category: string | null;
  success: boolean;
  metrics: CompactMetric[];
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_write_input_tokens: number;
  output_hash: string;
  output_preview: string;
  input_preview: string;
}

/** Compact model run — same metadata as ModelRun but with slim results. */
export interface CompactModelRun {
  test_run_id: string;
  timestamp: string;
  hyperparameters: RunHyperparameters;
  results: CompactTestResult[];
}

/** Full per-case detail for a single test case across all models (lazy-loaded). */
export interface CaseDetail {
  input: string;
  expected_output: string;
  models: Record<string, {
    actual_output: string;
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens: number;
    cache_write_input_tokens: number;
    metrics: Array<{
      name: string;
      success: boolean;
      score: number | null;
      reason: string | null;
    }>;
  }>;
}

/** A single test case pivoted across all models for grid display. */
export interface MultiModelCase {
  /** Test case identifier (from additional_metadata.id or result name) */
  case_id: string;
  /** Category from additional_metadata */
  category: string | null;
  /** Input text (shared across models — same test case) */
  input: string;
  /** Expected output (shared across models) */
  expected_output: string;
  /** Ground truth context passages (shared across models, for RAG evals) */
  context: string[] | null;
  /** Retrieved context passages (shared across models, for RAG evals) */
  retrieval_context: string[] | null;
  /** Per-model results keyed by canonical full model id
   *  (``ModelRun.hyperparameters.model``). */
  models: Record<string, {
    success: boolean;
    actual_output: string;
    metrics_data: MetricData[];
    latency_ms: number | null;
    input_tokens: number | null;
    output_tokens: number | null;
    additional_metadata: Record<string, unknown> | null;
  }>;
}

/** Summary of a model within a run (from list endpoint). */
export interface ModelRunSummary {
  test_run_id: string;
  model: string;
  total: number;
  passed: number;
  /** Per-metric pass rate for this model, keyed by DeepEval metric name
   *  (may include the ``[GEval]`` suffix). May be an empty object when no
   *  per-metric breakdown is available. */
  per_metric: Record<string, number>;
  /** Routing metadata sourced from the descriptor at write time
   *  (template runner). Never parsed back out of the model id.
   *  Required — the list endpoint always supplies both, defaulting a run
   *  file that lacks them to endpoint "runtime" and an empty region. */
  endpoint: "runtime" | "mantle";
  region: string;
}

/** Trophy score tuple, computed by the backend's single scoring
 * implementation (app/services/trophy.py). Ordering is lexicographic:
 * at_target, then best_rate, then avg_rate; later candidate wins full ties.
 * Pinned cross-language by backend/tests/fixtures/trophy_golden.json. */
export interface TrophyScore {
  /** Models that met EVERY configured target on this candidate. */
  at_target: number;
  /** Highest single-model pass rate (0-1). */
  best_rate: number;
  /** Mean pass rate across evaluated models (0-1). */
  avg_rate: number;
}

/**
 * Summary of an eval run file (from list endpoint).
 * Returned by GET /api/projects/{id}/eval/runs.
 */
export interface EvalRunSummary {
  /** Internal filename for fetching full results */
  _file: string;
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Prompt version used (e.g., "baseline", "v1") */
  prompt_version: string;
  /** Permanent source attribution written by the harness at run time.
   *  "optimization" = run came from an optimization loop iteration.
   *  "manual" = standalone eval. ``"validation"`` smoke runs are
   *  filtered out server-side (``EvalRunner.list_runs``) so the SPA
   *  never sees them. */
  source: "optimization" | "manual";
  /** Per-model summaries */
  model_runs: ModelRunSummary[];
  /** Server-computed trophy score (app/services/trophy.py — THE single
   *  scoring implementation, targets from harness config). The SPA only
   *  orders these tuples; it never re-scores. Always present: the API
   *  scores every run it returns, and an absent score read as zero would
   *  silently hand the trophy to whichever card happens to sort last. */
  trophy_score: TrophyScore;
}

/**
 * Structured error attached to a long-running resource.
 *
 * Emitted by the backend whenever an optimization run, eval job, or
 * agent turn enters a terminal error state. The `code` is a stable
 * namespaced identifier (e.g. `bedrock.token_expired`) the UI can
 * branch on for specific treatments; the `message` + `hint` are
 * always safe to render verbatim.
 */
export interface ResourceError {
  /** Stable namespaced code, e.g. `bedrock.token_expired`. */
  code: string;
  /** One-line human-readable description of the failure. */
  message: string;
  /** Optional remediation text, e.g. "Run `aws sso login`." */
  hint: string | null;
  /** ISO 8601 UTC timestamp when the error was recorded. */
  at: string;
  /** Optional structured payload (exit code, first exception, etc.). */
  details: Record<string, unknown> | null;
}

/**
 * Optimization loop state from the backend.
 */
export interface OptimizationState {
  /**
   * Loop status. Must stay in step with every status the backend writes —
   * `"failed"` was missing here, so a crashed run fell through every renderer's
   * status check and displayed as "Ready to optimize" (see
   * `mcp_tools/dispatch.py`, which sets `status = "failed"` on loop crash).
   *
   * Consumers should type their status→display maps as
   * `Record<OptimizationState["status"], T>` so adding a member here fails the
   * build at every renderer that has not handled it, rather than silently
   * rendering nothing.
   */
  status:
    | "starting"
    | "running"
    | "stopping"
    | "converged"
    | "plateau"
    | "max_iterations"
    | "optimizer_exhausted"
    | "optimizer_stalled"
    | "judge_unreliable"
    | "generation_unreliable"
    | "failed"
    | "cancelled";
  current_iteration: number;
  /** Absolute iteration ceiling for the run (grows on max_iterations
   * resume: cap = current + budget). Progress lines render against THIS.
   * Formerly shipped as "max_iterations", colliding with the harness
   * config's per-run budget of the same name — two different numbers
   * rendered under one label in the same panel. */
  iteration_cap: number;
  /* (trophy_score rides on each iteration row — see OptimizationIteration) */
  /** Per-run iteration budget (used in the resume confirmation dialog) */
  iterations_per_run: number;
  target_metric_scores: Record<string, number>;
  iterations: OptimizationIteration[];
  best_iteration: OptimizationIteration | null;
  /** Number of iterations to check for plateau detection */
  plateau_window: number;
  /** Plateau threshold (e.g. 0.02 = 2%) */
  plateau_threshold: number;
  /** The prompt version currently being evaluated. Updated at the start
   *  of each iteration — source of truth for the status bar. */
  current_prompt_version: string | null;

  /** Loop phase written by the orchestrator: "evaluating" while an
   *  iteration's eval job runs, "optimizing_prompt" while the optimizer
   *  analyzes failures and writes the next prompt version. Absent/null on
   *  records predating the field — the bar falls back to a generic label
   *  rather than inventing a phase. */
  phase?: string | null;
  /** Full Bedrock model_id of the first model to clear every target,
   *  populated when status === "converged". Display this as the
   *  recommended winner in the UI. */
  winning_model_id: string | null;
  /** Structured error set when the loop hit a terminal failure
   *  (mass metric errors, dispatch failure, optimizer timeout, etc.).
   *  `null` when the run ended cleanly or is still live. Cleared on
   *  resume. */
  error: ResourceError | null;
}

/**
 * A single iteration in the optimization loop.
 */
export interface OptimizationIteration {
  iteration: number;
  prompt_version: string;
  run_id: string;
  metrics_summary: Record<string, number>;
  total_cases: number;
  passed: number;
  pass_rate: number;
  timestamp: string;
  /** Per-model per-metric pass rates for this iteration. Keys are full
   *  Bedrock model IDs; values map metric name → pass rate. Absent on
   *  iterations recorded without per-model data. */
  per_model_metrics?: Record<string, Record<string, number>>;
  /** Per-model aggregate pass rates, keyed by full Bedrock model ID. */
  per_model_pass_rates?: Record<string, number>;
  /** Server-computed trophy score (see EvalRunSummary.trophy_score). */
  trophy_score: TrophyScore;
  /** Non-fatal run-quality notes recorded on the iteration (e.g., a
   *  tolerated transient generation failure). Absent when nothing to
   *  report. Nothing renders these today — they're persisted for future
   *  visibility. */
  caveats?: string[];
}

/* ================================================================
 * Eval job types — background evaluation job tracking
 * ================================================================ */

/**
 * Progress details for a running eval job.
 * Updated in real-time as models and test cases are processed.
 */
export interface EvalJobProgress {
  /** Currently evaluating model identifier */
  current_model: string;
  /** Number of models that have completed evaluation */
  models_completed: number;
  /** Total number of models to evaluate */
  models_total: number;
  /** Current test case index within the current model */
  current_case: number;
  /** Total test cases for the current model */
  cases_total: number;
  /** Human-readable status message */
  message: string;
}

/**
 * Background evaluation job state.
 * Returned by the job status REST endpoints.
 */
export interface EvalJob {
  /** Unique job identifier (UUID) */
  job_id: string;
  /** Project this job belongs to */
  project_id: string;
  /** Job type — ``"eval"``, ``"optimization_iteration"``, or
   *  ``"harness_build"``. The eval-progress hook filters to
   *  ``"eval"`` only; the others appear via different UIs. */
  job_type: "eval" | "optimization_iteration" | "harness_build";
  /**
   * Current job status. `"awaiting_user"` is a real backend state
   * (`JobState` in `services/jobs/__init__.py`, surfaced with `pause_reason`
   * when a job needs user input) and was missing here.
   */
  status:
    | "pending"
    | "running"
    | "completed"
    | "failed"
    | "cancelled"
    | "awaiting_user";
  /** Real-time progress details */
  progress: EvalJobProgress;
  /** Output results filename (set on completion) */
  output_file: string | null;
  /** Structured error (set on failure). `null` while running / completed OK. */
  error: ResourceError | null;
  /** ISO 8601 timestamp of job creation */
  created_at: string;
  /** ISO 8601 timestamp of job completion (null if still running) */
  completed_at: string | null;
}

/**
 * Per-model progress row folded from the durable event log by the backend
 * (``fold_event_log_progress``). Snake_case wire fields mirror the harness
 * event shape. Present on {@link ProjectActivityEval.models} only on the cloud
 * detached (``arch_split=True``) path — it re-seeds the per-model bars on
 * remount so ``Generate X/N`` survives navigation (the ws-client seq cursor
 * outlives React remounts, so the live stream alone would replay only NEW
 * events and miss the already-passed generate events).
 */
export interface ActivityModelProgress {
  /** Model identifier. */
  model: string;
  /** Current evaluation phase. */
  phase: "generating" | "scoring" | "complete";
  /** Number of cases whose generation has completed (scalar at fold time). */
  generated: number;
  /** Total number of cases. */
  cases_total: number;
  /** Whether scoring has finished. */
  scored: boolean;
  /** Number of cases scored so far. */
  scored_count: number;
  /** Number of cases that passed scoring. */
  passed: number;
}

/**
 * The active eval slice of {@link ProjectActivity} — the backend's
 * reconciled "is an eval running" answer. Present only when an eval is
 * genuinely active (a stale optimization-child eval whose loop has
 * ended on disk is reported as ``null`` by the backend, which is what
 * keeps the bar from sticking on "Evaluation running").
 */
export interface ProjectActivityEval {
  /** Active eval job id (for cancel + WS subscription). */
  job_id: string;
  /** Job status — an active value (pending, running, or awaiting_user). */
  status: EvalJob["status"];
  /** Total models being evaluated, for the "model X of N" headline. */
  models_total: number;
  /** Full progress dict (mirrors EvalJob.progress) for the bar. */
  progress: EvalJobProgress;
  /**
   * Per-model progress rows folded from the durable event log (cloud detached
   * path only). Absent/empty on dev / in-process. Used to seed the per-model
   * bars on mount so a mid-scoring remount repopulates ``Generate X/N``.
   */
  models?: ActivityModelProgress[];
}

/**
 * Single reconciled "what is running for this project" view, served by
 * ``GET /projects/{id}/activity``. Replaces the SPA's two independent
 * polls (optimization-on-disk + eval-jobs-in-memory) and the
 * client-side stitching that let them disagree. The backend resolves
 * the disk-vs-memory conflict before responding, so the SPA renders one
 * answer.
 */
export interface ProjectActivity {
  /** Loop-level optimization state, or ``null`` if none has run. */
  optimization: OptimizationState | null;
  /** Subprocess-level active eval, or ``null`` if none is running. */
  eval: ProjectActivityEval | null;
}

/* ================================================================
 * File browser types — project file tree and content endpoints
 * ================================================================ */

/** A node in the project file tree (from GET /files). */
export interface FileNode {
  /** File or directory name */
  name: string;
  /** Relative path from project root */
  path: string;
  /** Whether this node is a file or directory */
  type: "file" | "dir";
  /** File size in bytes (null for directories) */
  size: number | null;
  /** Child nodes (null for files) */
  children: FileNode[] | null;
}

/** File content response (from GET /files/{path}). */
export interface FileContentResponse {
  /** Relative path from project root */
  path: string;
  /** Text content of the file */
  content: string;
  /** File size in bytes */
  size: number;
  /** Detected language for syntax highlighting */
  language: string;
  /** Parsed rows for tabular data (CSV/Excel) */
  rows: Record<string, unknown>[] | null;
  /** Column names for tabular data */
  columns: string[] | null;
  /** Total row count in the source CSV before any truncation. When
   *  ``rows.length < total_rows``, the backend capped the response and
   *  the UI should surface that to the user. */
  total_rows: number | null;
}

/* ================================================================
 * Synthetic data types — batches of agent-generated test cases
 * awaiting user review.
 * ================================================================ */

/**
 * Summary of a synthetic batch shown in the left-pane list.
 * Matches backend ``SyntheticBatchSummary``.
 */
export interface SyntheticBatchSummary {
  /** Timestamped batch identifier (matches manifest filename stem). */
  batch_id: string;
  /** Generation strategy (paraphrase, from_docs, from_scratch, failure_targeted). */
  strategy: string;
  /** Filesystem-safe timestamp the batch was written at. */
  timestamp: string;
  /** Free-text label describing where the seed data came from. */
  seed_source: string | null;
  /** Optional eval run id the batch was seeded from (failure-targeted only). */
  seed_run_id: string | null;
  /** Total number of cases in the batch. */
  total_cases: number;
  /** Count of cases currently awaiting review. */
  pending: number;
  /** Count of cases currently approved. */
  approved: number;
  /** Count of cases currently rejected. */
  rejected: number;
  /** Companion CSV filename under data/synthetic/. */
  csv_file: string;
}

/**
 * One synthetic case enriched with its CSV row values.
 * Matches backend ``SyntheticCase``.
 */
export interface SyntheticCase {
  /** 0-based row index in the batch CSV. */
  index: number;
  /** Current review state ("pending" | "approved" | "rejected"). */
  status: string;
  /** CSV column name -> string value for this row. */
  values: Record<string, string>;
}

/**
 * Full batch manifest plus enriched per-case rows.
 * Matches backend ``SyntheticBatchDetail``.
 */
export interface SyntheticBatchDetail {
  batch_id: string;
  timestamp: string;
  strategy: string;
  seed_source: string | null;
  seed_run_id: string | null;
  total_cases: number;
  /** Harness field mapping snapshot at generation time (role -> column). */
  field_mappings: Record<string, string>;
  csv_file: string;
  /** Enriched per-case rows (index + status + CSV values). */
  rows: SyntheticCase[];
}

/**
 * Response from approve/reject endpoints.
 * Matches backend ``SyntheticBatchStatusResponse``.
 */
export interface SyntheticBatchStatusResponse {
  batch_id: string;
  pending: number;
  approved: number;
  rejected: number;
  /** Raw cases array ({"index", "status"}) for optimistic reconcile. */
  cases: { index: number; status: string }[];
}

/** Parsed harness config (from GET /harness/config). */
export interface HarnessConfig {
  /** Target model IDs to evaluate */
  models: string[];
  /** Path to the evaluation data file */
  data_file: string | null;
  /** Judge model ID for LLM-as-a-judge metrics */
  judge_model_id: string | null;
  /** Current prompt version name */
  prompt_version: string | null;
  /** Unified field mapping dict — Tier 1 keys map to LLMTestCase params, Tier 2 to metadata */
  field_mappings: Record<string, string> | null;
  /** Target pass rates per metric for optimization (e.g. {"ExactMatch": 0.85}) */
  target_metric_scores: Record<string, number> | null;
  /** Maximum optimization iterations allowed */
  max_iterations: number | null;
  /** Number of iterations to check for plateau detection */
  plateau_window: number | null;
  /** Pass-rate spread threshold for plateau detection (0.0–1.0, default 0.02).
   * When the spread between the highest and lowest pass rates over the
   * plateau_window falls below this threshold, the loop is flat and may
   * plateau. Backend default is 0.02 (2 percentage points). */
  plateau_threshold: number | null;
  /** The harness is RUNNABLE (models + metrics wired, dataset selected,
   * referenced prompt file present on disk) — not merely scaffolded.
   * Backend-computed; action buttons must gate on this, never on the config
   * merely parsing. */
  ready: boolean;
}
