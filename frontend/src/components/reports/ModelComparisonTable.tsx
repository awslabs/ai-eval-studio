// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-model standings table, shared by the executive and per-run reports.
 *
 * Lived inside ``ExecReport.tsx`` and was imported out of it by
 * ``RunReport.tsx`` — so the run report's main table was a detail of the other
 * report's file, and editing the executive report could break it. It belongs to
 * neither; it lives here.
 */

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Delta } from "@/components/ui/delta";
import type { ModelFacts, TrustReason } from "@/lib/reports/types";
import { MetricTrustIcon } from "@/components/reports/DataQuality";
import { fmtMsCompact, fmtPctSmart, fmtUsd } from "@/lib/reports/format";
import type { ReactElement } from "react";

interface ComparisonTableProps {
  models: ModelFacts[];
  metricNames: string[];
  targets: Record<string, number>;
  recById: Map<string, string>;
  recommendedId: string | undefined;
  /** Exec form: includes Verdict column. Per-run: omits it. */
  withVerdict: boolean;
  /**
   * Per-run flag: when true, render small Δ-vs-prior lines under each
   * pass-rate / metric cell, sourced from ``m.pass_rate_delta`` and
   * ``m.metric_rate_deltas``. Exec hides these (its journey already
   * covers per-version movement).
   */
  showDeltas?: boolean;
  /**
   * Worst trust reason per metric, from ``metricTrustIndex``. Required, not
   * optional: the whole point is that a disputed metric must not render with the
   * same authority as a sound one, and a default-empty map would silently
   * restore exactly that.
   */
  metricTrust: Map<string, TrustReason>;
}

export function ModelComparisonTable({
  models, metricNames, targets, recById, recommendedId, withVerdict, showDeltas,
  metricTrust,
}: ComparisonTableProps): ReactElement {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Model</TableHead>
          <TableHead className="text-right">Pass rate</TableHead>
          {metricNames.map((m) => {
            const reason = metricTrust.get(m);
            return (
              <TableHead key={m} className="text-right">
                {m}
                {reason && <MetricTrustIcon reason={reason} />}
              </TableHead>
            );
          })}
          <TableHead className="text-right">Median</TableHead>
          {withVerdict && <TableHead className="text-right">p95</TableHead>}
          <TableHead className="text-right">Cost / 1k (gen)</TableHead>
          {withVerdict && <TableHead>Verdict</TableHead>}
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((m) => {
          const isWinner = m.model_id === recommendedId;
          return (
            <TableRow
              key={m.model_id}
              className={isWinner ? "bg-success/5 font-medium" : ""}
            >
              <TableCell>{m.display_name}</TableCell>
              {/* Pass rate is a composite (all metrics at once) with no
                  target of its own — coloring it against a hardcoded 100%
                  painted a 97% top scorer red while the same number showed
                  green in its metric column. Render it neutral; per-metric
                  cells carry the pass/fail coloring against real targets. */}
              <PctCell
                value={m.pass_rate}
                thresholdPct={null}
                delta={showDeltas ? m.pass_rate_delta : undefined}
              />
              {metricNames.map((name) => (
                <PctCell
                  key={name}
                  value={m.metric_rates[name]}
                  thresholdPct={targets[name] !== undefined ? targets[name] * 100 : null}
                  delta={showDeltas ? m.metric_rate_deltas?.[name] : undefined}
                />
              ))}
              <TableCell className="text-right tabular-nums">
                {fmtMsCompact(m.latency_percentiles.median)}
              </TableCell>
              {withVerdict && (
                <TableCell className="text-right tabular-nums">
                  {fmtMsCompact(m.latency_percentiles.p95)}
                </TableCell>
              )}
              <TableCell className="text-right tabular-nums">
                {m.pricing_available ? fmtUsd(m.cost_per_1k_gen) : "n/a"}
              </TableCell>
              {withVerdict && <TableCell>{verdictBadge(recById.get(m.model_id))}</TableCell>}
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

function PctCell({
  value, thresholdPct, delta,
}: {
  value: number | null | undefined;
  thresholdPct: number | null;
  delta?: number | null;
}): ReactElement {
  if (value === null || value === undefined) {
    return <TableCell className="text-right tabular-nums">n/a</TableCell>;
  }
  let cls = "text-right tabular-nums whitespace-nowrap";
  if (thresholdPct !== null) {
    cls += value >= thresholdPct - 1e-9 ? " text-success" : " text-destructive";
  }
  return (
    <TableCell className={cls}>
      {fmtPctSmart(value)}
      {delta !== undefined && <Delta value={delta} suffix="%" />}
    </TableCell>
  );
}

function verdictBadge(tag: string | undefined): ReactElement | null {
  if (!tag) return null;
  if (tag === "Recommended" || tag === "Top pass rate") {
    return <Badge variant="success">{tag}</Badge>;
  }
  return <Badge variant="secondary">{tag}</Badge>;
}
