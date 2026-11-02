// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-run report — React component, fed by the report JSON.
 *
 * Sections (omit cleanly when data is absent):
 *  1. Header (chips + winner badge if winning run)
 *  2. Takeaway banner — computed result FIRST, narrator prose elaborating
 *  3. Trust line (one line, only when something needs checking)
 *  4. Headline tiles (pass rate, cost, latency) for the leading model
 *  5. What was tested
 *  6. Model standings (computed; Δ-vs-prior column)
 *  7. What changed this run (parsed prompt header + narrative insight)
 *  8. What's blocking — shared failure attribution
 *  9. What's still failing — narrative-clustered ≤5 cards, or lean fallback
 *     (≤5 raw failures + "+N more" trailer)
 * 10. Diagnostics (latency + eval cost)
 * 11. Data quality — the trust verdict in full, shared with the exec report
 * 12. Appendix (How we measured FIRST, then Prompt)
 *
 * The order matches the executive report's: result, then what qualifies it, then
 * the numbers, then how we got here, then what broke.
 */

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import type {
  FailedCaseFacts,
  ModelFacts,
  RunFacts,
  RunNarrative,
} from "@/lib/reports/types";
import {
  fmtDate,
  fmtMs,
  fmtMsCompact,
  fmtPct,
  fmtUsd,
  metricLine,
} from "@/lib/reports/format";
import { Chip, Kpi, Section } from "@/components/reports/primitives";
import { ModelComparisonTable } from "@/components/reports/ModelComparisonTable";
import { FailureBucketsSection } from "@/components/reports/FailureBuckets";
import {
  DataQualitySection,
  TrustLine,
  metricTrustIndex,
} from "@/components/reports/DataQuality";
import { MetricDefinitions } from "@/components/reports/MetricDefinitions";

interface RunReportProps {
  facts: RunFacts;
  narrative: RunNarrative | null;
}

export function RunReport({ facts, narrative }: RunReportProps) {
  // One index, read by the metric definitions and the standings columns, so a
  // metric cannot be marked in one place and unmarked in the other.
  const metricTrust = metricTrustIndex(facts.run_trust);
  const recById = new Map((facts.recommendations ?? []).map((r) => [r.model_id, r.tag]));
  // Backend SSOT — populated regardless of whether targets cleared, so
  // the highlighted row matches what the executive summary recommends.
  const recommendedId = facts.recommended_model_id ?? undefined;

  return (
    <div className="flex flex-col gap-6 w-full">
      <header className="border-b pb-5">
        <h1 className="text-xl font-semibold tracking-tight">
          {facts.project_name} — Run report{facts.prompt_label && ` (${facts.prompt_label})`}
        </h1>
        {(narrative?.context || facts.project_description) && (
          <p className="text-sm text-muted-foreground mt-1">
            {narrative?.context || facts.project_description}
          </p>
        )}
        <div className="flex flex-wrap gap-2 mt-3">
          {facts.prompt_label && <Chip>Prompt: <b>{facts.prompt_label}</b></Chip>}
          {facts.prompt_iteration !== null && facts.prompt_iteration !== undefined && (
            <Chip>Iteration <b>{facts.prompt_iteration}</b></Chip>
          )}
          {facts.run_source === "optimization" && <Chip>Optimization run</Chip>}
          {facts.run_source === "manual" && <Chip>Manual run</Chip>}
          {facts.run_timestamp && <Chip>{fmtDate(facts.run_timestamp)}</Chip>}
          {facts.is_winning_run && (
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-success/15 text-success text-xs font-semibold">
              ✓ best run so far
            </span>
          )}
        </div>
      </header>

      <TakeawayBanner facts={facts} narrative={narrative} />

      {/* Qualifies the claim directly above it, and is ONE line however many
          problems exist — a variable-height block here would push the result
          down, and real runs produce up to four reasons. Detail lives in the
          Data quality section further down. */}
      <TrustLine trust={facts.run_trust} />

      <RunKpis facts={facts} />

      <MetricDefinitions
        metricNames={facts.metric_names}
        thresholds={facts.metric_thresholds}
        criteria={facts.metric_criteria}
        latencyBudgetMs={facts.latency_budget_ms}
        descriptions={narrative?.metric_descriptions}
        trust={facts.run_trust}
      />

      <Section
        title="Model standings"
        subtitle={facts.prior_step ? `This run, all models. Δ shown vs ${facts.prior_step.version}.` : "This run, all models."}
      >
        <ModelComparisonTable
          models={facts.models}
          metricNames={facts.metric_names}
          targets={facts.metric_thresholds}
          recById={recById}
          recommendedId={recommendedId}
          withVerdict={false}
          showDeltas={!!facts.prior_step}
          metricTrust={metricTrust}
        />
      </Section>

      {/* Directly after the standings, because it explains the number that just
          moved. It used to sit below both failure sections, which buried the most
          interesting thing about an optimization run — and left this report
          telling its story in the opposite order to the executive one. */}
      {(facts.prompt_changes || narrative?.what_changed_worked) && (
        <Section title="What changed this run">
          {narrative?.what_changed_worked && (
            <p className="text-sm leading-relaxed">{narrative.what_changed_worked}</p>
          )}
          {facts.prompt_changes && (
            <div className="flex flex-col gap-2">
              {splitChangesIntoCards(facts.prompt_changes).map(([head, body], i) => (
                <Card key={i} className="border-l-4 border-l-primary p-3 gap-1">
                  <div className="text-sm font-semibold">{head}</div>
                  <div className="text-xs text-muted-foreground">{body}</div>
                </Card>
              ))}
            </div>
          )}
        </Section>
      )}

      <FailureBucketsSection buckets={facts.failure_buckets} />
      <FailuresSection facts={facts} narrative={narrative} />

      <Diagnostics facts={facts} />

      {/* Replaces an unlabelled card that rendered here only when cases were
          dropped. Same position, so nothing above it shifts — but now labelled,
          always present, and shared with the executive report. */}
      <DataQualitySection trust={facts.run_trust} caseCount={facts.case_count} />

      <RunAppendix facts={facts} narrative={narrative} />
    </div>
  );
}

/* ----------------------- Sections ----------------------- */

/**
 * Headline numbers for the leading model, mirroring the executive report's three
 * tiles. The run report previously made the reader extract these from the table,
 * or scroll to Diagnostics for cost and latency.
 *
 * The cost detail is deliberately plain rather than the executive report's
 * "up to ~Nx cheaper" comparison: that comparison is about choosing between
 * models across the whole optimization, which is not what a single run answers.
 */
function RunKpis({ facts }: { facts: RunFacts }): React.ReactElement | null {
  const lead = leadModel(facts);
  if (!lead) return null;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      <Kpi value={fmtPct(lead.pass_rate)} label="Pass rate" detail={lead.display_name} />
      <Kpi
        value={lead.pricing_available ? fmtUsd(lead.cost_per_1k_gen) : "n/a"}
        label="Cost / 1k queries"
        detail={lead.pricing_available ? "Generation cost only" : "No pricing data"}
      />
      <Kpi
        value={fmtMs(lead.latency_percentiles.median)}
        label="Median response"
        detail="Per request, end-to-end"
      />
    </div>
  );
}

/**
 * Resolve the run's leading model: the backend's recommendation when it set one,
 * else the highest pass rate.
 */
function leadModel(facts: RunFacts): ModelFacts | undefined {
  if (!facts.models.length) return undefined;
  const recommended = facts.models.find(
    (m) => m.model_id === facts.recommended_model_id,
  );
  return recommended ?? [...facts.models].sort((a, b) => b.pass_rate - a.pass_rate)[0];
}

function TakeawayBanner({ facts, narrative }: { facts: RunFacts; narrative: RunNarrative | null }) {
  const prior = facts.prior_step;
  let bannerCls = "border-l-primary bg-primary/5";
  if (prior && prior.delta >= 0.5) bannerCls = "border-l-success bg-success/5";
  else if (prior && prior.delta <= -0.5) bannerCls = "border-l-destructive bg-destructive/5";

  // The RESULT is stated from the facts, always — the narrator's prose then
  // elaborates beneath it. This used to be the other way round: the narrator's
  // takeaway was the headline and the computed line only appeared if the
  // narrator produced nothing, so a vague or missing narration left the run
  // report with no reliable statement of what happened. The executive report's
  // Recommendation card already works the right way round; this matches it.
  const lead = leadModel(facts);
  const headline = lead
    ? `${lead.display_name} reached ${fmtPct(lead.pass_rate)} pass rate.`
    : "No models in this run.";
  const elaboration = narrative?.takeaway?.trim();

  let detail: React.ReactNode = "First run on this prompt — no prior version to compare.";
  if (prior && Math.abs(prior.delta) >= 0.5) {
    // ``prior.pass_rate`` is the PRIOR version's best rate; ``prior.delta`` is
    // ``current − prior``. So the current rate = prior + delta.
    const fromRate = Math.min(100, Math.max(0, prior.pass_rate));
    const toRate = Math.min(100, Math.max(0, prior.pass_rate + prior.delta));
    const sign = prior.delta >= 0 ? "+" : "−";
    detail = (
      <>
        Best-model pass rate moved <b>{fmtPct(fromRate)} → {fmtPct(toRate)} ({sign}{Math.abs(prior.delta).toFixed(1)})</b> vs {prior.version}.
      </>
    );
  } else if (prior) {
    detail = `No change vs ${prior.version}.`;
  }

  return (
    <Card className={`border-l-4 ${bannerCls} p-4 gap-1`}>
      <div className="text-base font-semibold">{headline}</div>
      <div className="text-sm">{detail}</div>
      {elaboration && (
        <p className="text-sm leading-relaxed text-muted-foreground">{elaboration}</p>
      )}
    </Card>
  );
}

function FailuresSection({ facts, narrative }: { facts: RunFacts; narrative: RunNarrative | null }) {
  if (!facts.failed_cases.length) return null;
  const patterns = (narrative?.failure_patterns ?? []).slice(0, 5);
  const caseById = new Map(facts.failed_cases.map((fc) => [String(fc.case_id), fc]));

  if (patterns.length > 0) {
    return (
      <Section
        title="What's still failing"
        subtitle={`${facts.failed_cases.length} failure${facts.failed_cases.length === 1 ? "" : "s"} clustered into ${patterns.length} pattern${patterns.length === 1 ? "" : "s"}. Open the list at the bottom for every individual failure.`}
      >
        <div className="flex flex-col gap-2">
          {patterns.map((p, i) => {
            const example = caseById.get(String(p.representative_case_id)) ?? facts.failed_cases[0];
            const isLat = !!example?.is_latency;
            return (
              <Card key={i} className={`border-l-4 ${isLat ? "border-l-warning" : "border-l-destructive"} p-3 gap-2`}>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded ${isLat ? "bg-warning/10 text-warning" : "bg-destructive/10 text-destructive"}`}>
                    {p.label}
                  </span>
                  <span className="text-xs text-muted-foreground">{p.count} cases</span>
                </div>
                {example && (
                  <ExamplePreview fc={example} extraLabel={`Example (case #${example.case_id}, ${example.model_display} · ${example.metric})`} />
                )}
                {p.insight && <p className="text-xs text-muted-foreground">{p.insight}</p>}
              </Card>
            );
          })}
        </div>
        <FullFailureList cases={facts.failed_cases} />
      </Section>
    );
  }

  // Lean fallback (no narrator clusters): show ≤5 worst-first cards as a
  // quick preview and the full expandable list for every failure.
  const shown = facts.failed_cases.slice(0, 5);
  return (
    <Section title="What's still failing" subtitle={`${facts.failed_cases.length} failure${facts.failed_cases.length === 1 ? "" : "s"} this run.`}>
      <div className="flex flex-col gap-2">
        {shown.map((fc, i) => (
          <FailureCard key={i} fc={fc} />
        ))}
      </div>
      <FullFailureList cases={facts.failed_cases} />
    </Section>
  );
}

function FailureCard({ fc }: { fc: FailedCaseFacts }) {
  return (
    <Card className={`border-l-4 ${fc.is_latency ? "border-l-warning" : "border-l-destructive"} p-3 gap-2`}>
      <div className="flex items-center gap-2 text-xs flex-wrap">
        <span className="font-mono text-muted-foreground">case #{fc.case_id}</span>
        <span className="font-bold">{fc.model_display}</span>
        {fc.category && <span className="px-1.5 py-0.5 rounded bg-muted text-foreground">{fc.category}</span>}
        <span className={`px-1.5 py-0.5 rounded font-semibold ${fc.is_latency ? "bg-warning/10 text-warning" : "bg-destructive/10 text-destructive"}`}>
          {fc.metric}
        </span>
      </div>
      <div className="text-sm"><span className="text-muted-foreground font-medium">Input:</span> {fc.input}</div>
      {(fc.expected || fc.actual) && (
        <div className="bg-muted/40 rounded px-2 py-1.5 font-mono text-xs">
          <div><span className="text-muted-foreground">Expected:</span> <span className="text-success break-words">{fc.expected}</span></div>
          <div><span className="text-muted-foreground">Actual:</span> <span className="text-destructive break-words">{fc.actual}</span></div>
        </div>
      )}
      {fc.reason && (
        <div className="text-sm text-foreground/90 leading-relaxed">
          <span className="text-muted-foreground font-medium">Why it failed:</span> {fc.reason}
        </div>
      )}
    </Card>
  );
}

function FullFailureList({ cases }: { cases: FailedCaseFacts[] }) {
  if (cases.length === 0) return null;
  return (
    <details className="bg-card border rounded-md mt-2 group">
      <summary className="px-4 py-2.5 cursor-pointer text-sm font-semibold text-muted-foreground hover:text-foreground flex items-center gap-2">
        <span className="group-open:hidden">▶</span>
        <span className="hidden group-open:inline">▼</span>
        All {cases.length} failure{cases.length === 1 ? "" : "s"} (click to expand)
      </summary>
      <div className="p-3 border-t flex flex-col gap-2">
        {cases.map((fc, i) => (
          <FailureCard key={i} fc={fc} />
        ))}
      </div>
    </details>
  );
}

function ExamplePreview({ fc, extraLabel }: { fc: FailedCaseFacts; extraLabel: string }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-sm">
        <b className="text-muted-foreground font-medium">{extraLabel}:</b> {fc.input}
      </div>
      {(fc.expected || fc.actual) && (
        <div className="bg-muted/40 rounded px-2 py-1.5 font-mono text-xs">
          <div><span className="text-muted-foreground">Expected:</span> <span className="text-success break-words">{fc.expected}</span></div>
          <div><span className="text-muted-foreground">Actual:</span> <span className="text-destructive break-words">{fc.actual}</span></div>
        </div>
      )}
      {fc.reason && (
        <div className="text-sm text-foreground/90 leading-relaxed">
          <span className="text-muted-foreground font-medium">Why it failed:</span> {fc.reason}
        </div>
      )}
    </div>
  );
}

function Diagnostics({ facts }: { facts: RunFacts }) {
  const models = facts.models;
  if (!models.length) return null;
  const hasLat = facts.has_latency_metric;
  const budget = facts.latency_budget_ms;
  const overBudgetHeader = hasLat ? (budget ? `Over ${budget} ms` : "Over budget") : null;
  const showJudge = models.some((m) => m.judge_cost_available);

  return (
    <Section title="Diagnostics">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Median</TableHead>
              <TableHead className="text-right">p95</TableHead>
              <TableHead className="text-right">p99</TableHead>
              <TableHead className="text-right">Max</TableHead>
              {overBudgetHeader && <TableHead className="text-right">{overBudgetHeader}</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {models.map((m) => {
              const p = m.latency_percentiles;
              const maxOver = budget && p.max !== null && p.max > budget;
              return (
                <TableRow key={m.model_id}>
                  <TableCell>{m.display_name}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtMsCompact(p.median)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtMsCompact(p.p95)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmtMsCompact(p.p99)}</TableCell>
                  <TableCell className={`text-right tabular-nums ${maxOver ? "text-destructive" : ""}`}>
                    {fmtMsCompact(p.max)}
                  </TableCell>
                  {overBudgetHeader && (
                    <TableCell className={`text-right tabular-nums ${(p.over_budget_count ?? 0) > 0 ? "text-destructive" : "text-success"}`}>
                      {p.over_budget_count === null ? "n/a" : `${p.over_budget_count} / ${p.total}`}
                    </TableCell>
                  )}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Generation</TableHead>
              {showJudge && <TableHead className="text-right">Judge</TableHead>}
              <TableHead className="text-right">Total</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {models.map((m) => (
              <TableRow key={m.model_id}>
                <TableCell>{m.display_name}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {m.pricing_available ? fmtUsd(m.generation_cost_usd) : "n/a"}
                </TableCell>
                {showJudge && (
                  <TableCell className="text-right tabular-nums">
                    {m.judge_cost_available ? fmtUsd(m.judge_cost_usd) : "n/a"}
                  </TableCell>
                )}
                <TableCell className="text-right tabular-nums">
                  {m.pricing_available ? fmtUsd(m.total_cost_usd) : "n/a"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {cacheOffNote(models) && (
        <p className="text-xs text-muted-foreground">{cacheOffNote(models)}</p>
      )}
    </Section>
  );
}

function cacheOffNote(models: ModelFacts[]): string {
  const allOff = models.every((m) => m.cache_read_ratio === 0);
  const meaningful = models.some((m) => m.avg_input_tokens > 100);
  return allOff && meaningful
    ? "Prompt caching is OFF (no cache hits) — a cost lever if this scales."
    : "";
}

function RunAppendix({ facts, narrative }: { facts: RunFacts; narrative: RunNarrative | null }) {
  const hyper = facts.hyperparameters ?? {};
  const budget = facts.latency_budget_ms;
  return (
    <section className="flex flex-col gap-2">
      <details className="bg-card border rounded-md">
        <summary className="px-4 py-2.5 cursor-pointer text-sm font-semibold text-muted-foreground hover:text-foreground">
          How we measured
        </summary>
        <div className="p-4 pt-3 border-t text-sm">
          <ol className="list-decimal pl-5 space-y-1.5 text-muted-foreground">
            <li>
              Each model generated output for {facts.case_count} cases using prompt {facts.prompt_label}
              {hyperParamsText(hyper) && ` (${hyperParamsText(hyper)})`}.
            </li>
            {facts.metric_names.map((name) => (
              <li key={name}>
                <b className="text-foreground">{name}</b>: {metricLine(
                  name,
                  facts.metric_thresholds[name],
                  facts.metric_criteria[name],
                  budget,
                  narrative?.metric_descriptions?.[name],
                )}
              </li>
            ))}
            <li>A case passes only when every metric meets its threshold.</li>
          </ol>
        </div>
      </details>
      {facts.prompt_content && (
        <details className="bg-card border rounded-md">
          <summary className="px-4 py-2.5 cursor-pointer text-sm font-semibold text-muted-foreground hover:text-foreground">
            Prompt{facts.prompt_label && ` (${facts.prompt_label})`}
          </summary>
          <div className="p-4 pt-3 border-t">
            <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-muted/40 p-3 rounded">
              {facts.prompt_content}
            </pre>
          </div>
        </details>
      )}
    </section>
  );
}

/* ----------------------- helpers ----------------------- */

function splitChangesIntoCards(text: string): Array<[string, string]> {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 1) {
    const line = lines[0];
    if (line.includes(":")) {
      const idx = line.indexOf(":");
      return [[line.slice(0, idx).trim(), line.slice(idx + 1).trim() || line]];
    }
    return [[line.length > 80 ? line.slice(0, 77) + "…" : line, line]];
  }
  return lines.slice(0, 5).map((line): [string, string] => {
    if (line.includes(":")) {
      const idx = line.indexOf(":");
      return [line.slice(0, idx).trim(), line.slice(idx + 1).trim() || line];
    }
    return [line.length > 80 ? line.slice(0, 77) + "…" : line, line];
  });
}

function hyperParamsText(hyper: Record<string, unknown>): string {
  const bits: string[] = [];
  if (hyper.temperature !== undefined && hyper.temperature !== null) bits.push(`temperature ${hyper.temperature}`);
  if (hyper.max_tokens !== undefined && hyper.max_tokens !== null) bits.push(`max ${hyper.max_tokens} tokens`);
  return bits.join(", ");
}

