// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Type definitions for the report JSON the backend serves.
 *
 * The backend (``app/services/report.py``) is the source of truth for
 * shape; these mirror its dataclasses one-for-one. Numbers are computed
 * server-side (deterministic); the optional ``narrative`` block carries
 * the report-narrator agent's prose. When ``narrative`` is null the UI
 * renders the lean fallback (facts only).
 */

/** Latency percentiles, ms (or null when no per-case latency was recorded). */
export interface LatencyPercentiles {
  median: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
  /** Cases the Latency metric flagged failing (null when no Latency metric). */
  over_budget_count: number | null;
  /** Total cases this model ran (denominator for over_budget). */
  total: number;
}

/** Per-model aggregates that drive every model-comparison table. */
export interface ModelFacts {
  model_id: string;
  display_name: string;
  total: number;
  passed: number;
  pass_rate: number;
  metric_rates: Record<string, number>;
  avg_latency_ms: number | null;
  latency_percentiles: LatencyPercentiles;
  generation_cost_usd: number;
  judge_cost_usd: number;
  total_cost_usd: number;
  cost_per_1k_gen: number;
  judge_cost_available: boolean;
  /** False when LiteLLM has no pricing entry for the model id (mantle-tier ids in the allowlist that LiteLLM doesn't yet price). Renderer hides USD values rather than $0.00. */
  pricing_available: boolean;
  avg_input_tokens: number;
  avg_output_tokens: number;
  cache_read_ratio: number;
  /** Pass-rate delta vs the prior version's run for THIS model (pp). */
  pass_rate_delta: number | null;
  /** Per-metric pass-rate deltas vs the prior version's run (pp). */
  metric_rate_deltas: Record<string, number>;
}

/** One (case, failed metric) row carried in the facts-pack. */
export interface FailedCaseFacts {
  case_id: string | number;
  category: string;
  model_id: string;
  model_display: string;
  metric: string;
  is_latency: boolean;
  input: string;
  expected: string;
  actual: string;
  reason: string;
}

/** One prompt version on the journey (one row per ``vN`` on disk). */
export interface JourneyVersionFacts {
  version: string;
  best_pass_rate: number;
  delta: number | null;
  per_model_pass_rates: Record<string, number>;
  cost_per_1k_gen: number | null;
  avg_input_tokens: number | null;
  what_changed: string | null;
}

/** Honest-denominator check from the harness's ``data_quality`` block. */
export interface RunIntegrityFacts {
  case_count: number;
  errored: number;
  dropped: number;
  null_count: number;
  unrecovered: number;
  effective_total: number;
  reliability: string;
}

/** Per-model verdict tag the renderer shows in the comparison table. */
export interface RecommendationFacts {
  model_id: string;
  tag: string;
}

/** Failure-attribution bucket counts (cases EVERY model failed). */
export interface FailureBucketFacts {
  judge_false_negatives: number;
  data_review: number;
  model_gaps: number;
}

/** Per-run report facts. */
export interface RunFacts {
  schema_version: number;
  project_name: string;
  project_description: string;
  run_filename: string;
  run_timestamp: string;
  run_source: string;
  prompt_label: string;
  prompt_iteration: number | null;
  prompt_content: string;
  prompt_changes: string | null;
  case_count: number;
  category_count: number;
  metric_names: string[];
  metric_thresholds: Record<string, number>;
  metric_criteria: Record<string, string>;
  latency_budget_ms: number | null;
  has_latency_metric: boolean;
  hyperparameters: Record<string, unknown>;
  models: ModelFacts[];
  recommendations: RecommendationFacts[];
  /**
   * Single source of truth for "which model is recommended". KPI tiles,
   * the recommendation header, and the narrative all key on THIS field
   * — the verdict-tag system on ``recommendations`` only labels
   * cleared-vs-not against targets, so on a plateau (no clearer) those
   * tags can't identify a recommended model. Stays populated even then.
   */
  recommended_model_id: string | null;
  failed_cases: FailedCaseFacts[];
  failure_buckets: FailureBucketFacts;
  run_integrity: RunIntegrityFacts;
  /**
   * Every reason to doubt this run's numbers, on one severity scale. Built
   * server-side from run integrity, judge reliability and the metric audit, so
   * the reports, the run list and the agent all read the same verdict instead of
   * each interpreting the underlying checks.
   *
   * NOT optional: facts are rebuilt fresh on every request, so this is always
   * present. Whether the metric audit ran is carried by ``audit_available``
   * rather than by the field being absent — a reader forced to interpret a
   * missing verdict takes the obvious and wrong reading, that no news is good
   * news.
   */
  run_trust: RunTrustFacts;
  prior_step: { version: string; pass_rate: number; delta: number } | null;
  is_winning_run: boolean;
}

/** Severity of a single trust reason, worst last. */
export type TrustSeverity = "info" | "warning" | "error";

/** One reason to doubt a run, worded for a reader rather than a log. */
export interface TrustReason {
  id: string;
  severity: TrustSeverity;
  /** What is wrong, in the user's terms. */
  headline: string;
  /** What it means for the numbers on the page. */
  impact: string;
  /**
   * Every metric this reason implicates, so a renderer can mark those columns
   * without re-deriving anything from the text. A LIST because two diagnostics
   * contradicting each other implicate both and neither can be singled out —
   * naming one would tell the reader the other column is sound. Empty for
   * run-wide reasons (dropped cases, judge reliability).
   */
  metrics: string[];
}

/** A run's trust verdict: worst severity plus every reason found. */
export interface RunTrustFacts {
  severity: "ok" | TrustSeverity;
  reasons: TrustReason[];
  /**
   * Whether the metric audit ran for this run. A fact about the run, not an
   * absent field: "checked and clean" and "not checked" must not render the
   * same, and a reader forced to infer it from absence takes the wrong reading.
   */
  audit_available: boolean;
}

/** Executive report facts. */
export interface ExecFacts {
  schema_version: number;
  project_name: string;
  project_description: string;
  case_count: number;
  category_count: number;
  metric_names: string[];
  metric_thresholds: Record<string, number>;
  metric_criteria: Record<string, string>;
  latency_budget_ms: number | null;
  journey: JourneyVersionFacts[];
  winner_version: string;
  winner_run_filename: string;
  winner_run: RunFacts;
  /** Humanized "v8 run · June 12, 2026" stamp. Never the raw filename. */
  generated_from_label: string;
  generated_at: string;
}

/** Per-run narrator output. All fields optional — agent may emit a subset. */
export interface RunNarrative {
  /** One-sentence framing of what this run tested (use case + scale). */
  context?: string | null;
  takeaway?: string | null;
  failure_patterns?: Array<{
    label: string;
    representative_case_id: string | number;
    count: number;
    insight: string;
  }> | null;
  what_changed_worked?: string | null;
  metric_descriptions?: Record<string, string> | null;
}

/** Executive narrator output. */
export interface ExecNarrative {
  intro?: string | null;
  recommendation?: string | null;
  journey_narrative?: string | null;
  data_quality_insight?: string | null;
  outlook?: string | null;
  metric_descriptions?: Record<string, string> | null;
}

/** Wire-format envelope returned by the API endpoints. */
export interface RunReportEnvelope {
  schema_version: number;
  scope: "run";
  facts: RunFacts;
  narrative: RunNarrative | null;
  generated_at: string | null;
}

export interface ExecReportEnvelope {
  schema_version: number;
  scope: "executive";
  facts: ExecFacts;
  narrative: ExecNarrative | null;
  generated_at: string | null;
  /**
   * True when the cached narrative was written about an EARLIER run of
   * the same winning version. Stat cards always show fresh numbers, but
   * the prose may not reflect the latest run. Only meaningful when
   * ``narrative`` is non-null. Always false on per-run reports — those
   * are pinned to one run by construction.
   */
  stale?: boolean;
}
