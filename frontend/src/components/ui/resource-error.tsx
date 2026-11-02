// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Banner that surfaces a backend ResourceError inline on a resource panel.
 *
 * Renders message + optional hint with destructive-themed accents and a
 * timestamp for context. Designed to be placed at the top of the panel
 * whose resource just failed (optimization panel, eval job panel, etc.).
 * Clean/empty when `error` is null.
 */

import { AlertTriangle, X } from "lucide-react";

import type { ResourceError } from "@/lib/types";
import { cn } from "@/lib/class-utils";
import { Button } from "@/components/ui/button";

interface ResourceErrorBannerProps {
  error: ResourceError | null;
  /** Optional dismiss handler. Omit for non-dismissable banners. */
  onDismiss?: () => void;
  /** Optional action slot rendered on the right (e.g., a Resume button). */
  action?: React.ReactNode;
  className?: string;
}

/**
 * Format an ISO 8601 timestamp as a short HH:MM:SS local-time string.
 * Falls back to the raw string on parse failures so malformed
 * timestamps never crash the banner.
 */
function formatTimestamp(at: string): string {
  try {
    const d = new Date(at);
    if (Number.isNaN(d.getTime())) return at;
    return d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return at;
  }
}

export function ResourceErrorBanner({
  error,
  onDismiss,
  action,
  className,
}: ResourceErrorBannerProps) {
  if (!error) return null;

  return (
    <div
      role="alert"
      aria-live="polite"
      className={cn(
        "flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2.5 text-sm",
        className,
      )}
    >
      <AlertTriangle
        className="mt-0.5 h-4 w-4 shrink-0 text-destructive"
        aria-hidden="true"
      />
      <div className="flex-1 min-w-0 space-y-1">
        <div className="flex items-baseline gap-2">
          <span className="font-medium text-destructive">{error.message}</span>
          <span className="text-xs text-muted-foreground tabular-nums">
            {formatTimestamp(error.at)}
          </span>
        </div>
        {error.hint && (
          <p className="text-xs text-muted-foreground">{error.hint}</p>
        )}
        <p className="text-2xs font-mono text-muted-foreground/70">
          {error.code}
        </p>
      </div>
      {action && <div className="shrink-0">{action}</div>}
      {onDismiss && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onDismiss}
          className="shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          aria-label="Dismiss error"
        >
          <X className="size-3.5" />
        </Button>
      )}
    </div>
  );
}
