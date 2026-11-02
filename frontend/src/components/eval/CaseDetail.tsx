// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Expandable detail panel for a single test case across multiple models.
 * Shows input/expected with per-model output diff, stats, and metrics.
 */

import { useMemo } from "react";
import type { MultiModelCase } from "@/lib/types";
import { SYSTEM_METADATA_KEYS } from "@/lib/eval-utils";
import { formatLabel } from "@/lib/format-utils";
import { DetailSection } from "@/components/eval/DetailSection";

/** Props for the multi-model case detail panel. */
interface MultiModelCaseDetailProps {
  /** Pivoted case data with per-model results. The metric filter has
   *  already narrowed the per-case ``metrics_data`` upstream
   *  (EvalDashboard ``applyMetricFilter``), so this component just
   *  renders what it gets. */
  caseData: MultiModelCase;
  /** Model IDs that should be visible (controlled by parent filters) */
  visibleModels: string[];
}

/**
 * Renders the expanded detail view for a test case across multiple models.
 *
 * Layout:
 * - Top: Shared input + expected output (2 columns)
 * - Body: Per-model detail sections with output diff, stats, and expandable metrics
 *
 * @param caseData - Pivoted case data with per-model results
 * @param visibleModels - Model IDs to display (controlled by parent filters)
 */
export function MultiModelCaseDetail({ caseData: c, visibleModels }: MultiModelCaseDetailProps) {
  return (
    <div className="bg-muted/20 border-t p-5 space-y-5">
      {/* Shared: Input + Expected */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <TextBlock label="Input" text={c.input} />
        <TextBlock label="Expected Output" text={c.expected_output} />
      </div>

      {/* Context fields (RAG evals) */}
      {(c.context || c.retrieval_context) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {c.context && <TextBlock label="Context" text={c.context.join("\n\n---\n\n")} />}
          {c.retrieval_context && <TextBlock label="Retrieval Context" text={c.retrieval_context.join("\n\n---\n\n")} />}
        </div>
      )}

      {/* Per-model detail sections — equal-width grid columns */}
      <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${visibleModels.length}, minmax(280px, 1fr))` }}>
        {visibleModels.map((modelId) => {
          const m = c.models[modelId];
          if (!m) return null;
          return (
            <ModelDetailSection
              key={modelId}
              modelId={modelId}
              data={m}
              expected={c.expected_output}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ===================================================================
 * Internal sub-components for multi-model view
 * =================================================================== */

/**
 * A single model's detail section within a multi-model case detail.
 * Delegates rendering to the shared DetailSection component.
 *
 * @param modelId - Model identifier for the header
 * @param data - The model's results for this test case
 * @param expected - Expected output for diff computation
 */
function ModelDetailSection({ modelId, data, expected }: {
  modelId: string;
  data: MultiModelCase["models"][string];
  expected: string;
}) {
  const stats = useMemo(() => {
    const result: Array<{ label: string; value: string }> = [];
    if (data.latency_ms != null) result.push({ label: "Latency", value: `${data.latency_ms.toFixed(0)}ms` });
    if (data.input_tokens != null) result.push({ label: "In", value: data.input_tokens.toLocaleString() });
    if (data.output_tokens != null) result.push({ label: "Out", value: data.output_tokens.toLocaleString() });
    if (data.input_tokens != null && data.output_tokens != null) {
      result.push({ label: "Total", value: (data.input_tokens + data.output_tokens).toLocaleString() });
    }
    if (data.additional_metadata?.cache_read_input_tokens != null) {
      result.push({ label: "Cache Read", value: Number(data.additional_metadata.cache_read_input_tokens).toLocaleString() });
    }
    if (data.additional_metadata?.cache_write_input_tokens != null) {
      result.push({ label: "Cache Write", value: Number(data.additional_metadata.cache_write_input_tokens).toLocaleString() });
    }
    if (data.additional_metadata?.generation_cost != null) {
      result.push({ label: "Cost", value: `$${Number(data.additional_metadata.generation_cost).toFixed(6)}` });
    }
    /* Dynamic metadata fields (Tier 2 — user dataset columns) */
    Object.entries(data.additional_metadata ?? {})
      .filter(([key]) => !SYSTEM_METADATA_KEYS.has(key))
      .filter(([, value]) => value != null && value !== "")
      .forEach(([key, value]) => result.push({ label: formatLabel(key), value: String(value) }));
    return result;
  }, [data]);

  const metrics = useMemo(
    () => data.metrics_data.map((md) => ({
      name: md.name,
      score: md.score,
      passed: md.success,
      reason: md.reason,
    })),
    [data.metrics_data],
  );

  return (
    <DetailSection
      label={modelId}
      passed={data.success}
      stats={stats}
      output={data.actual_output}
      diffExpected={expected}
      metrics={metrics}
    />
  );
}

/* ===================================================================
 * Shared helper components
 * =================================================================== */

/**
 * Small labeled text block with a pre-formatted content area.
 *
 * @param label - Heading label above the text block
 * @param text - Text content to render
 */
function TextBlock({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <p className="text-2xs font-medium text-muted-foreground mb-1">
        {label}
      </p>
      <pre className="text-xs whitespace-pre-wrap font-mono bg-background border rounded-md p-3 h-20 overflow-y-auto resize-y leading-relaxed text-muted-foreground">{text}</pre>
    </div>
  );
}
