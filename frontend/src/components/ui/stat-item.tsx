// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Small label / value pair for compact stat display — used in summary
 * headers, case-detail metrics blocks, and compare expanded-row metrics.
 */
import type { ReactNode } from "react";

import { cn } from "@/lib/class-utils";

interface StatItemProps {
  label: string;
  value: ReactNode;
  className?: string;
  /** Muted label + emphasized value layout (horizontal). */
  inline?: boolean;
}

/** Renders a compact label/value pair in either inline or stacked layout. */
export function StatItem({ label, value, className, inline = false }: StatItemProps) {
  if (inline) {
    return (
      <span className={cn(className)}>
        <span className="text-muted-foreground/60">{label}: </span>
        <span className="font-medium tabular-nums">{value}</span>
      </span>
    );
  }
  return (
    <div className={cn("flex flex-col gap-0.5", className)}>
      <span className="text-2xs uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="text-sm font-medium tabular-nums">{value}</span>
    </div>
  );
}
