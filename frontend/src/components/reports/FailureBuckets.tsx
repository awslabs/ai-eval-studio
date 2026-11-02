// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * "What's blocking" — cases every model failed, attributed by likely cause.
 *
 * Both reports rendered these three buckets from the same backend field with
 * their own component, and the two copies had already drifted: the labels
 * differed ("Data review needed" against "Data-review needed"), the hints
 * differed ("matched the expected answer" against "produced the expected
 * answer"), and the executive report showed all three tiles including empty ones
 * while the run report hid the zeros. One component, one wording.
 *
 * Zero-count buckets are hidden — the run report's behaviour. A greyed "0
 * Model gaps" tile spends a third of the row telling the reader nothing, and the
 * section already states the total in its subtitle.
 *
 * This answers WHOSE FAULT a shared failure is, which is a different question
 * from "can I trust these numbers" — that one lives in ``DataQuality``. The two
 * were tangled in one section in the executive report, which is why the metric
 * audit had nowhere obvious to go.
 */

import type { ReactElement } from "react";

import { Card } from "@/components/ui/card";
import type { FailureBucketFacts } from "@/lib/reports/types";
import { Section } from "@/components/reports/primitives";

interface BucketSpec {
  label: string;
  count: number;
  tone: string;
  hint: string;
}

/** Ordered bucket definitions — the single copy of this wording. */
function specs(b: FailureBucketFacts): BucketSpec[] {
  return [
    {
      label: "Likely scoring errors",
      count: b.judge_false_negatives ?? 0,
      tone: "text-warning",
      hint: "All models produced the expected answer, but the metric flagged them — judge or rubric is over-strict.",
    },
    {
      label: "Data review needed",
      count: b.data_review ?? 0,
      tone: "text-warning",
      hint: "All models agreed on a different answer than expected — the test data may be wrong.",
    },
    {
      label: "Model gaps",
      count: b.model_gaps ?? 0,
      tone: "text-destructive",
      hint: "Models disagreed; no candidate solved these — a real capability gap.",
    },
  ];
}

/**
 * Render the shared-failure attribution section, or nothing on a clean run.
 *
 * @param buckets - Per-cause counts from the run facts.
 * @param insight - Optional narrator prose describing the counts.
 */
export function FailureBucketsSection({
  buckets, insight,
}: {
  buckets: FailureBucketFacts | null | undefined;
  insight?: string | null;
}): ReactElement | null {
  if (!buckets) return null;
  const items = specs(buckets).filter((s) => s.count > 0);
  const total = items.reduce((sum, s) => sum + s.count, 0);
  if (total === 0) return null;

  return (
    <Section
      title="What's blocking"
      subtitle={`${total} case${total === 1 ? "" : "s"} no model passed, broken down by likely cause.`}
    >
      {insight && (
        <Card className="p-3 bg-muted/40 border-l-4 border-l-primary text-sm leading-relaxed">
          {insight}
        </Card>
      )}
      <div className="flex flex-wrap gap-3">
        {items.map((it) => (
          <Card key={it.label} className="p-3 gap-1 flex-1 min-w-[200px]">
            <div className={`text-2xl font-bold tabular-nums ${it.tone}`}>{it.count}</div>
            <div className="text-sm font-semibold">{it.label}</div>
            <div className="text-xs text-muted-foreground">{it.hint}</div>
          </Card>
        ))}
      </div>
    </Section>
  );
}
