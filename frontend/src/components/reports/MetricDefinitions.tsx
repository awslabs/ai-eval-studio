// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * "What was tested" — the metric list, shared by both reports.
 *
 * The two reports carried byte-identical copies of this block, so adding
 * anything to it meant editing it twice and hoping the copies stayed together.
 * They already had not: the failure-bucket labels beside them differed by a
 * hyphen. One definition, imported by both.
 *
 * A metric the trust verdict names is marked HERE, next to its definition,
 * rather than only in the Data quality section further down. A reader who has
 * just been told what a metric measures is the reader who needs to know the
 * metric is in question — and unlike the column-header icon, this placement has
 * room for the reason in words, so it survives a printed report.
 */

import type { ReactElement } from "react";

import { Section } from "@/components/reports/primitives";
import { SeverityLine, metricTrustIndex } from "@/components/reports/DataQuality";
import { metricLine } from "@/lib/reports/format";
import type { RunTrustFacts, TrustReason } from "@/lib/reports/types";

interface MetricDefinitionsProps {
  metricNames: string[];
  thresholds: Record<string, number>;
  criteria: Record<string, string>;
  latencyBudgetMs: number | null;
  /**
   * Narrator one-liners per metric; raw criteria are the fallback. ``null`` when
   * the narrator produced no prose for this report at all, which is a normal
   * outcome (its session can fail while the computed report still renders).
   */
  descriptions: Record<string, string> | null | undefined;
  /**
   * The run's verdict. Required, not optional: an absent verdict would render
   * every metric as sound, which is the reading this whole section exists to
   * prevent.
   */
  trust: RunTrustFacts;
}

/** Metric list with a trust mark on any metric the verdict names. */
export function MetricDefinitions({
  metricNames, thresholds, criteria, latencyBudgetMs, descriptions, trust,
}: MetricDefinitionsProps): ReactElement | null {
  if (metricNames.length === 0) return null;
  const marked = metricTrustIndex(trust);

  return (
    <Section title="What was tested">
      <ul className="flex flex-col gap-1.5 text-sm">
        {metricNames.map((name) => {
          const reason: TrustReason | undefined = marked.get(name);
          return (
            <li key={name} className="flex flex-col gap-1">
              <div className="flex flex-col sm:flex-row sm:gap-2">
                <span className="font-semibold sm:min-w-[140px]">{name}</span>
                <span className="text-muted-foreground">
                  {metricLine(
                    name,
                    thresholds[name],
                    criteria[name],
                    latencyBudgetMs,
                    descriptions?.[name],
                  )}
                </span>
              </div>
              {reason && (
                <SeverityLine severity={reason.severity} className="sm:ml-[148px]">
                  {reason.headline}. {reason.impact}
                </SeverityLine>
              )}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
