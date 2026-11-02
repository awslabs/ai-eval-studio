// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * "Data quality" — can these numbers be trusted, and why not.
 *
 * One component for both reports, reading the single ``run_trust`` verdict the
 * backend builds. Before this, the same information arrived by three different
 * routes: a labelled section in the executive report, an unlabelled card wedged
 * between Diagnostics and the appendix in the run report, and — for the metric
 * audit — a file nothing read at all.
 *
 * Two pieces, and the split matters:
 *
 * - ``TrustLine`` is ONE line, always one line, however many problems exist. It
 *   sits high on the page. A variable-height block there would push the
 *   recommendation down, and a run in the real data produced four reasons.
 * - ``DataQualitySection`` carries the detail, and stays at the position the
 *   old card already occupied, so nothing above it shifts.
 *
 * An explicit all-clear is rendered when nothing is wrong. Absence of a warning
 * cannot be told apart from "never checked", which is precisely the state the
 * metric audit was in.
 */

import type { ReactElement, ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, OctagonAlert } from "lucide-react";

import { Card } from "@/components/ui/card";
import type { RunTrustFacts, TrustReason, TrustSeverity } from "@/lib/reports/types";
import { Section } from "@/components/reports/primitives";

/**
 * Left-border + tint per severity, matching the card style already in use.
 *
 * Exported because the executive report's Recommendation card is tinted from the
 * same verdict — a recommendation derived from questionable numbers should not be
 * rendered in confident green while the caveat sits five sections below.
 */
export const TRUST_TONE: Record<TrustSeverity, string> = {
  error: "border-l-error bg-error/5",
  warning: "border-l-warning bg-warning/5",
  info: "border-l-info bg-info/5",
};

const TONE = TRUST_TONE;

/**
 * The single most serious reason, or ``null`` when there is nothing to report.
 *
 * ``run_trust.reasons`` is already ordered worst-first by the backend, so this
 * takes the head rather than re-ranking — re-ranking in a renderer is how the
 * severity scale would drift away from the one the audit assigned.
 */
export function worstTrustReason(trust: RunTrustFacts): TrustReason | null {
  if (!hasFindings(trust)) return null;
  return trust.reasons[0] ?? null;
}

/**
 * Worst reason per metric, for marking a metric where its numbers are shown.
 *
 * The section below tells a reader that a metric is in question; it does not
 * stop them reading that metric's column as fact three sections earlier. This
 * index is what lets the metric definition and the standings column carry the
 * same verdict.
 *
 * Takes the FIRST reason naming each metric because ``reasons`` arrives ordered
 * worst-first from the backend — the same reason the section shows at the top.
 * A metric can appear in several reasons (saturated AND disputed by a
 * comparator); marking it with anything but the worst would understate it.
 */
export function metricTrustIndex(trust: RunTrustFacts): Map<string, TrustReason> {
  const index = new Map<string, TrustReason>();
  for (const reason of trust.reasons ?? []) {
    for (const metric of reason.metrics ?? []) {
      if (!index.has(metric)) index.set(metric, reason);
    }
  }
  return index;
}

/**
 * One severity-toned line of text with its icon.
 *
 * Used wherever a finding has to appear IN PLACE — beside a metric's definition
 * — rather than in the Data quality section. Kept here with the icon and tone
 * maps so a second severity vocabulary cannot grow elsewhere in the reports.
 */
export function SeverityLine({
  severity, className = "", children,
}: {
  severity: TrustSeverity;
  className?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <p className={`flex items-start gap-1.5 text-xs ${ICON_CLASS[severity]} ${className}`}>
      <SeverityIcon severity={severity} />
      <span className="text-muted-foreground">{children}</span>
    </p>
  );
}

/**
 * Severity icon for one metric, carrying the reason as its hover text.
 *
 * For places where only an icon fits — a table column header. The full wording
 * stays with the metric's definition and in the Data quality section, so a
 * printed report (where nothing hovers) still explains the mark; here the icon's
 * job is to stop a reader taking the column at face value.
 */
export function MetricTrustIcon({
  reason,
}: {
  reason: TrustReason;
}): ReactElement {
  return (
    <span title={`${reason.headline}. ${reason.impact}`} className="ml-1 inline-flex align-middle">
      <SeverityIcon severity={reason.severity} />
    </span>
  );
}

const ICON_CLASS: Record<TrustSeverity, string> = {
  error: "text-error",
  warning: "text-warning",
  info: "text-info",
};

function SeverityIcon({ severity }: { severity: TrustSeverity }): ReactElement {
  const cls = `size-4 shrink-0 ${ICON_CLASS[severity]}`;
  if (severity === "error") return <OctagonAlert className={cls} aria-hidden />;
  if (severity === "warning") return <AlertTriangle className={cls} aria-hidden />;
  return <Info className={cls} aria-hidden />;
}

/** True when the verdict says something a reader needs to act on. */
function hasFindings(trust: RunTrustFacts): boolean {
  return trust.severity !== "ok" && (trust.reasons?.length ?? 0) > 0;
}

/**
 * One fixed-height line summarising the verdict, for the top of a report.
 *
 * Renders nothing when the run is clean or unchecked — the all-clear belongs in
 * the section, not at the top of the page, where it would be one more thing
 * between the reader and the result.
 */
export function TrustLine({
  trust,
}: {
  trust: RunTrustFacts;
}): ReactElement | null {
  if (!hasFindings(trust)) return null;
  const n = trust.reasons.length;
  const worst = trust.reasons[0];
  return (
    <Card
      className={`flex-row items-center gap-2 border-l-4 px-3 py-2 text-sm ${TONE[trust.severity as TrustSeverity]}`}
    >
      <SeverityIcon severity={trust.severity as TrustSeverity} />
      <span className="min-w-0 truncate">
        <b>
          {n === 1
            ? "1 thing to check before trusting these numbers"
            : `${n} things to check before trusting these numbers`}
        </b>
        {" — "}
        {worst.headline}. See Data quality.
      </span>
    </Card>
  );
}

/**
 * The trust detail section: one card per reason, or an explicit all-clear.
 *
 * @param trust - The run's verdict. Required: ``audit_available`` distinguishes
 *   "checked and clean" from "never checked", so an absent verdict can never be
 *   read as a clean one.
 * @param caseCount - Dataset size, used only in the all-clear wording.
 */
export function DataQualitySection({
  trust, caseCount,
}: {
  trust: RunTrustFacts;
  caseCount: number;
}): ReactElement {
  const findings = hasFindings(trust);

  // Clean AND checked is the only state that earns an all-clear.
  if (!findings && trust.audit_available) {
    return (
      <Section title="Data quality">
        <Card className="flex-row items-start gap-2 border-l-4 border-l-success bg-success/5 p-3 text-sm leading-relaxed">
          <CheckCircle2 className="size-4 shrink-0 text-success" aria-hidden />
          <span>
            All {caseCount.toLocaleString()} cases scored cleanly, and no metric
            contradicted another. The numbers above can be read as they stand.
          </span>
        </Card>
      </Section>
    );
  }

  return (
    <Section
      title="Data quality"
      subtitle={findings
        ? "What would make the numbers above wrong, worst first."
        : undefined}
    >
      {trust.reasons.map((r, i) => (
        <Card
          key={`${r.id}-${i}`}
          className={`flex-row items-start gap-2 border-l-4 p-3 text-sm leading-relaxed ${TONE[r.severity]}`}
        >
          <SeverityIcon severity={r.severity} />
          <span>
            <b>{r.headline}.</b> {r.impact}
          </span>
        </Card>
      ))}
      {/* Stated alongside any other reasons, not instead of them: a run can have
          dropped cases recorded while the metric audit still failed to write. */}
      {!trust.audit_available && (
        <Card className="flex-row items-start gap-2 border-l-4 border-l-muted bg-muted/40 p-3 text-sm leading-relaxed">
          <Info className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          <span>
            The metric cross-check did not run for this run, so nothing here
            confirms the metrics agree with each other.
          </span>
        </Card>
      )}
    </Section>
  );
}
