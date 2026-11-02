// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout primitives shared by the executive and per-run reports.
 *
 * These were defined separately in both report files. ``Section`` was
 * byte-identical in each and ``Chip`` differed only by an optional
 * ``className``, so the two copies were free to drift apart while looking
 * interchangeable — and the failure-bucket labels beside them already had
 * ("Data review needed" in one report, "Data-review needed" in the other).
 * One definition each, imported by both.
 */

import type { ReactElement, ReactNode } from "react";

import { Card } from "@/components/ui/card";

/** Titled report section with an optional one-line subtitle. */
export function Section({
  title, subtitle, children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
      {children}
    </section>
  );
}

/** Small pill used in report headers. */
export function Chip({
  children, className = "",
}: {
  children: ReactNode;
  className?: string;
}): ReactElement {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-1 rounded-md bg-secondary text-secondary-foreground text-xs ${className}`}
    >
      {children}
    </span>
  );
}

/** Header chip of the form "<b>208</b> test cases". */
export function ChipNumeric({
  value, label,
}: {
  value: number;
  label: string;
}): ReactElement {
  return <Chip><b>{value}</b>&nbsp;{label}</Chip>;
}

/** Big headline number with a label and optional detail line. */
export function Kpi({
  value, label, detail,
}: {
  value: string;
  label: string;
  detail?: string;
}): ReactElement {
  return (
    <Card className="p-5 gap-1 items-center text-center">
      <div className="text-3xl font-bold tracking-tight tabular-nums">{value}</div>
      <div className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">
        {label}
      </div>
      {detail && <div className="text-xs text-muted-foreground">{detail}</div>}
    </Card>
  );
}
