// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Executive report — React component, fed by the report JSON.
 *
 * Theme/typography inherit from the dashboard automatically (this is just
 * a tree of shadcn primitives). When ``narrative`` is null the lean
 * fallback renders facts only — tables, KPI tiles, computed verdict line.
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
import type { ExecFacts, ExecNarrative, JourneyVersionFacts, ModelFacts } from "@/lib/reports/types";
import {
  fmtDate,
  fmtDelta,
  fmtMs,
  fmtPct,
  fmtUsd,
  metricLine,
} from "@/lib/reports/format";
import { Chip, ChipNumeric, Kpi, Section } from "@/components/reports/primitives";
import { FailureBucketsSection } from "@/components/reports/FailureBuckets";
import {
  DataQualitySection,
  TRUST_TONE,
  metricTrustIndex,
  worstTrustReason,
} from "@/components/reports/DataQuality";
import { MetricDefinitions } from "@/components/reports/MetricDefinitions";
import { ModelComparisonTable } from "@/components/reports/ModelComparisonTable";

interface ExecReportProps {
  facts: ExecFacts;
  narrative: ExecNarrative | null;
}

export function ExecReport({ facts, narrative }: ExecReportProps) {
  const winner = facts.winner_run;
  const winnerModels = winner?.models ?? [];
  const recById = new Map(
    (winner?.recommendations ?? []).map((r) => [r.model_id, r.tag]),
  );
  // Single source of truth — backend ``recommended_model_id`` is set even
  // on plateau runs where no model cleared its targets. The verdict-tag
  // map (``recById``) drives the comparison-table column only.
  const recommendedId = winner?.recommended_model_id ?? undefined;
  const recommendedModel: ModelFacts | undefined =
    winnerModels.find((m) => m.model_id === recommendedId);
  // One verdict, read here and rendered in three places: the Summary clause, the
  // Recommendation card's tone and caveat, and the Data quality section.
  const trust = winner?.run_trust;
  const worstReason = worstTrustReason(trust);
  // Same index for the winner run's metric definitions and its comparison
  // columns — the exec report's numbers ARE that run's numbers.
  const metricTrust = metricTrustIndex(trust);

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Header */}
      <header className="border-b pb-5">
        <h1 className="text-2xl font-semibold tracking-tight">{facts.project_name}</h1>
        {facts.project_description && (
          <p className="text-sm text-muted-foreground mt-1">{facts.project_description}</p>
        )}
        <div className="flex flex-wrap gap-2 mt-3">
          <ChipNumeric value={facts.case_count} label="test cases" />
          {facts.category_count > 0 && (
            <ChipNumeric value={facts.category_count} label="categories" />
          )}
          {winnerModels.length > 0 && (
            <ChipNumeric value={winnerModels.length} label="models" />
          )}
          {facts.winner_version && (
            <Chip>Best prompt: <b>{facts.winner_version}</b></Chip>
          )}
          {winner?.run_timestamp && <Chip>{fmtDate(winner.run_timestamp)}</Chip>}
        </div>
        {(facts.winner_version || winner?.run_timestamp) && (
          <p className="text-xs text-muted-foreground mt-2">
            Generated from{" "}
            {[
              facts.winner_version && `${facts.winner_version} run`,
              winner?.run_timestamp && fmtDate(winner.run_timestamp),
            ].filter(Boolean).join(" · ")}
          </p>
        )}
      </header>

      {/* Summary — goal + scale + candidate models */}
      <Section title="Summary">
        {narrative?.intro ? (
          <p className="text-base leading-relaxed">{narrative.intro}</p>
        ) : (
          <p className="text-base leading-relaxed">
            {facts.project_description || `Evaluation across ${facts.journey.length} prompt versions.`}
          </p>
        )}
        <p className="text-sm text-muted-foreground">
          Evaluated{" "}
          {winnerModels.length > 0 && (
            <>
              <span className="text-foreground font-medium">
                {winnerModels.map((m) => m.display_name).join(", ")}
              </span>{" "}
            </>
          )}
          on {facts.case_count.toLocaleString()} cases
          {facts.category_count > 0 && ` (${facts.category_count} categories)`}
          {facts.metric_names.length > 0 && (
            <>
              , scored on{" "}
              <span className="text-foreground font-medium">
                {facts.metric_names.join(" + ")}
              </span>
            </>
          )}
          .
        </p>
        {!narrative?.intro && facts.journey.length >= 2 && (
          <p className="text-sm text-muted-foreground">
            {leadFallback(facts, recommendedModel)}
          </p>
        )}
        {/* The gist has to include how much confidence to place in it. Written
            by the renderer from the verdict, not by the narrator, so it cannot
            go missing or be reworded on a rerun. */}
        {worstReason && (
          <p className="text-sm text-muted-foreground">
            {worstReason.headline} — see Data quality.
          </p>
        )}
      </Section>

      {/* KPI tiles — recommended model summary */}
      {recommendedModel && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Kpi
            value={fmtPct(recommendedModel.pass_rate)}
            label="Pass rate"
            detail={recommendedModel.display_name}
          />
          <Kpi
            value={recommendedModel.pricing_available ? fmtUsd(recommendedModel.cost_per_1k_gen) : "n/a"}
            label="Cost / 1k queries"
            detail={costComparisonDetail(recommendedModel, winnerModels)}
          />
          <Kpi
            value={fmtMs(recommendedModel.latency_percentiles.median)}
            label="Median response"
            detail="Per request, end-to-end"
          />
        </div>
      )}

      {/* Recommendation — carries its OWN confidence. It is derived from the
          pass rates, so when something undermines those the card cannot stay
          confident green with the caveat buried five sections below. It still
          names a model either way: withholding the recommendation leaves the
          reader with nothing to act on. */}
      <Section title="Recommendation">
        <Card
          className={`border-l-4 p-4 gap-2 ${
            worstReason ? TRUST_TONE[worstReason.severity] : "border-l-success bg-success/5"
          }`}
        >
          {recommendedModel && (
            <div className="text-base font-semibold">
              {recommendedModel.display_name}, with the {facts.winner_version} prompt.
            </div>
          )}
          <p className="text-sm">
            {narrative?.recommendation ?? recommendationFallback(facts, recommendedModel, winnerModels)}
          </p>
          {worstReason && (
            <p className="text-sm">
              <b>Before acting on this:</b> {worstReason.headline}. {worstReason.impact}
            </p>
          )}
        </Card>
      </Section>

      <MetricDefinitions
        metricNames={facts.metric_names}
        thresholds={facts.metric_thresholds}
        criteria={facts.metric_criteria}
        latencyBudgetMs={facts.latency_budget_ms}
        descriptions={narrative?.metric_descriptions}
        trust={trust}
      />

      {/* Model comparison */}
      <Section
        title="Model comparison"
        subtitle={`Final standings on the ${facts.winner_version} prompt.`}
      >
        <ModelComparisonTable
          models={winnerModels}
          metricNames={facts.metric_names}
          targets={facts.metric_thresholds}
          recById={recById}
          recommendedId={recommendedId}
          metricTrust={metricTrust}
          withVerdict
        />
      </Section>

      {/* Journey — omit when <2 versions */}
      {(facts.journey?.length ?? 0) >= 2 && (
        <Section
          title="How the prompt got here"
          subtitle="Best-model pass rate per version, reconstructed from run history."
        >
          <JourneyChart journey={facts.journey} winnerVersion={facts.winner_version} />
          {narrative?.journey_narrative && (
            <p className="text-sm leading-relaxed">{narrative.journey_narrative}</p>
          )}
          <JourneyTable journey={facts.journey} winnerVersion={facts.winner_version} />
        </Section>
      )}

      {/* Shared-failure attribution and "can I trust this" used to be one
          section, which is why the metric audit had nowhere obvious to go. They
          answer different questions and are now separate — the same two sections,
          in the same order, as the run report. */}
      <FailureBucketsSection
        buckets={winner?.failure_buckets}
        insight={narrative?.data_quality_insight}
      />

      <DataQualitySection trust={trust} caseCount={facts.case_count} />

      {/* Outlook (narrative-only) */}
      {narrative?.outlook && (
        <Section title="Outlook">
          <p className="text-sm leading-relaxed">{narrative.outlook}</p>
        </Section>
      )}

      {/* Appendix */}
      <ExecAppendix facts={facts} narrative={narrative} />
    </div>
  );
}

/* ----------------------- Tables / charts ----------------------- */

function JourneyChart({ journey, winnerVersion }: { journey: JourneyVersionFacts[]; winnerVersion: string }) {
  const peak = Math.max(...journey.map((j) => j.best_pass_rate), 100);
  return (
    <Card className="p-4">
      <div className="flex items-end gap-2 h-40 px-2">
        {journey.map((j) => {
          const isWinner = j.version === winnerVersion;
          const height = Math.max(6, (j.best_pass_rate / peak) * 100);
          return (
            <div key={j.version} className="flex-1 flex flex-col items-center justify-end h-full">
              <div className="text-xs font-semibold tabular-nums mb-1">{fmtPct(j.best_pass_rate)}</div>
              <div
                className={`w-full max-w-[46px] rounded-t-md ${isWinner ? "bg-success" : "bg-primary"}`}
                style={{ height: `${height}%` }}
              />
              <div className="text-xs text-muted-foreground mt-1.5">
                {j.version}{isWinner && " ✓"}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function JourneyTable({ journey, winnerVersion }: { journey: JourneyVersionFacts[]; winnerVersion: string }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Version</TableHead>
          <TableHead>What changed</TableHead>
          <TableHead className="text-right">Best pass</TableHead>
          <TableHead className="text-right">Δ</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {journey.map((j) => {
          const isWinner = j.version === winnerVersion;
          return (
            <TableRow key={j.version} className={isWinner ? "bg-success/5 font-medium" : ""}>
              <TableCell>
                {j.version}{isWinner && <span className="text-success"> ✓ winner</span>}
              </TableCell>
              {/* Full change note, wrapping — the table cell default is
                  whitespace-nowrap, which is what forced the old 120-char
                  truncation. The note is the report's only record of WHY a
                  version moved the score. */}
              <TableCell className="text-sm whitespace-normal break-words min-w-[240px]">
                {j.what_changed ?? "—"}
              </TableCell>
              <TableCell className="text-right tabular-nums">{fmtPct(j.best_pass_rate)}</TableCell>
              <TableCell className="text-right">{fmtDelta(j.delta)}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

/* ----------------------- Failures + Appendix ----------------------- */

function ExecAppendix({ facts, narrative }: { facts: ExecFacts; narrative: ExecNarrative | null }) {
  const winner = facts.winner_run;
  const hyper = winner?.hyperparameters ?? {};
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
              Each model generated output for all {facts.case_count} cases using the {facts.winner_version} prompt
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
      {winner?.prompt_content && (
        <details className="bg-card border rounded-md">
          <summary className="px-4 py-2.5 cursor-pointer text-sm font-semibold text-muted-foreground hover:text-foreground">
            Final prompt{facts.winner_version && ` (${facts.winner_version})`}
          </summary>
          <div className="p-4 pt-3 border-t">
            <pre className="text-xs font-mono whitespace-pre-wrap break-words bg-muted/40 p-3 rounded">
              {winner.prompt_content}
            </pre>
          </div>
        </details>
      )}
    </section>
  );
}

/* ----------------------- Fallback prose ----------------------- */

function leadFallback(facts: ExecFacts, recommended: ModelFacts | undefined): string {
  if (!recommended) return "";
  if (facts.journey.length >= 2) {
    const first = facts.journey[0].best_pass_rate;
    const last = Math.max(...facts.journey.map((j) => j.best_pass_rate));
    return `Across ${facts.journey.length} prompt versions, the best model went from ${fmtPct(first)} to ${fmtPct(last)}. The recommended setup is ${recommended.display_name} on ${facts.winner_version}.`;
  }
  return `Recommended: ${recommended.display_name} at ${fmtPct(recommended.pass_rate)} pass rate on ${facts.winner_version}.`;
}

function recommendationFallback(
  facts: ExecFacts, rec: ModelFacts | undefined, _models: ModelFacts[],
): string {
  if (!rec) return "No models in this run.";
  const targets = facts.metric_thresholds;
  const cleared = Object.keys(targets).every(
    (name) => (rec.metric_rates[name] ?? 0) >= targets[name] * 100 - 1e-9,
  );
  if (cleared) {
    return `It scores ${fmtPct(rec.pass_rate)} on every metric and is the cheapest model that meets the bar.`;
  }
  return `Best available at ${fmtPct(rec.pass_rate)} — no model fully cleared the targets.`;
}

function costComparisonDetail(rec: ModelFacts, models: ModelFacts[]): string {
  // when the recommended model is NA-priced, no peer comparison
  // is meaningful — the KPI shows "n/a" and this detail line explains why.
  if (!rec.pricing_available) return "Cost comparison unavailable (no pricing data)";
  if (!rec.cost_per_1k_gen || models.length < 2) return "Generation cost only";
  // Single clause — "Up to ~Nx cheaper than pricier models". The earlier
  // "~2× cheaper, ~3× cheaper" join read as a meaningless pair without
  // referents, since the user has no way to know which peer goes with
  // which ratio.
  const ratios = models
    .filter((m) => m !== rec && m.pricing_available && m.cost_per_1k_gen > 0)
    .map((m) => m.cost_per_1k_gen / rec.cost_per_1k_gen)
    .filter((r) => r > 1.5);
  if (ratios.length === 0) return "Generation cost only";
  const peak = Math.round(Math.max(...ratios));
  return `Up to ~${peak}× cheaper than pricier models`;
}

function hyperParamsText(hyper: Record<string, unknown>): string {
  const bits: string[] = [];
  if (hyper.temperature !== undefined && hyper.temperature !== null) bits.push(`temperature ${hyper.temperature}`);
  if (hyper.max_tokens !== undefined && hyper.max_tokens !== null) bits.push(`max ${hyper.max_tokens} tokens`);
  return bits.join(", ");
}

