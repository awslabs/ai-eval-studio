// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-case review card for a synthetic batch entry.
 *
 * Shows the case's CSV row values laid out by ``field_mappings``
 * (input first, expected_output second, remaining columns after) plus
 * Approve / Reject controls. Updates status optimistically; reverts and
 * surfaces an inline error message if the POST fails.
 */

import { useState } from "react";
import { Check, X, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/class-utils";
import type { SyntheticCase } from "@/lib/types";

/** Props for a single synthetic case card. */
interface SyntheticCaseDetailProps {
  /** Project UUID (for the API call). */
  projectId: string;
  /** Batch identifier the case belongs to. */
  batchId: string;
  /** The case to render. */
  caseItem: SyntheticCase;
  /** Field mapping snapshot from the manifest (role -> CSV column). */
  fieldMappings: Record<string, string>;
  /**
   * Called with the new status before the request is sent (optimistic
   * update), and again with the previous status if the request fails, so
   * the parent can keep its local counts / row state in step.
   */
  onStatusChange: (index: number, status: string) => void;
}

/** Ordered list of field mapping roles for display. */
const PRIMARY_ROLES = ["input", "expected_output"] as const;

/**
 * Build the ordered (label, column, value) tuples this card renders.
 *
 * The primary roles go first (input, then expected_output) so the user
 * always sees the question/answer pair on top. Remaining columns in the
 * row are appended in the order they appear in the CSV.
 */
function orderedFields(
  values: Record<string, string>,
  mappings: Record<string, string>,
): { label: string; column: string; value: string }[] {
  const fields: { label: string; column: string; value: string }[] = [];
  const emittedColumns = new Set<string>();

  for (const role of PRIMARY_ROLES) {
    const column = mappings[role];
    if (column && column in values) {
      fields.push({
        label: humanizeRole(role),
        column,
        value: values[column] ?? "",
      });
      emittedColumns.add(column);
    }
  }

  // Any role in mappings that wasn't a primary role (category, id, etc.).
  for (const [role, column] of Object.entries(mappings)) {
    if (PRIMARY_ROLES.includes(role as typeof PRIMARY_ROLES[number])) continue;
    if (emittedColumns.has(column)) continue;
    if (!(column in values)) continue;
    fields.push({
      label: humanizeRole(role),
      column,
      value: values[column] ?? "",
    });
    emittedColumns.add(column);
  }

  // Trailing CSV columns that weren't mapped to any role.
  for (const column of Object.keys(values)) {
    if (emittedColumns.has(column)) continue;
    fields.push({ label: column, column, value: values[column] ?? "" });
    emittedColumns.add(column);
  }

  return fields;
}

/** Convert a role key like ``expected_output`` to ``Expected output``. */
function humanizeRole(role: string): string {
  const spaced = role.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Status-specific badge styles. */
function statusBadge(status: string) {
  if (status === "approved") {
    return (
      <Badge
        variant="outline"
        className="border-success/30 text-success bg-success/10"
      >
        <Check className="size-3" />
        Approved
      </Badge>
    );
  }
  if (status === "rejected") {
    return (
      <Badge
        variant="outline"
        className="border-destructive/30 text-destructive bg-destructive/10"
      >
        <X className="size-3" />
        Rejected
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="text-muted-foreground">
      Pending
    </Badge>
  );
}

/**
 * Render one synthetic case with Approve / Reject controls.
 *
 * @param projectId - Project UUID for the API call
 * @param batchId - Batch identifier this case belongs to
 * @param caseItem - The case (index, status, values)
 * @param fieldMappings - Manifest field mappings used to order values
 * @param onStatusChange - Parent-side reconcile callback
 */
export function SyntheticCaseDetail({
  projectId,
  batchId,
  caseItem,
  fieldMappings,
  onStatusChange,
}: SyntheticCaseDetailProps) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fields = orderedFields(caseItem.values, fieldMappings);

  /**
   * Apply the new status optimistically through ``onStatusChange``, then
   * fire the mutation; roll back through the same callback if it fails.
   */
  const flip = async (target: "approved" | "rejected") => {
    const previous = caseItem.status;
    setError(null);
    setPending(true);
    onStatusChange(caseItem.index, target);
    try {
      if (target === "approved") {
        await api.approveSyntheticCases(projectId, batchId, {
          indices: [caseItem.index],
        });
      } else {
        await api.rejectSyntheticCases(projectId, batchId, {
          indices: [caseItem.index],
        });
      }
    } catch (e) {
      // Roll back the optimistic update and surface the error inline.
      onStatusChange(caseItem.index, previous);
      setError(e instanceof Error ? e.message : "Update failed");
    } finally {
      setPending(false);
    }
  };

  const isApproved = caseItem.status === "approved";
  const isRejected = caseItem.status === "rejected";

  return (
    <Card
      size="sm"
      className={cn(
        "px-4",
        isApproved && "ring-success/40",
        isRejected && "opacity-70",
      )}
    >
      {/* Header row: index + status badge + controls */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-2xs text-muted-foreground font-mono">
            #{caseItem.index + 1}
          </span>
          {statusBadge(caseItem.status)}
        </div>
        <div className="flex items-center gap-1.5">
          {pending && (
            <Loader2 className="size-3 animate-spin text-muted-foreground" />
          )}
          <Button
            size="xs"
            variant={isApproved ? "default" : "outline"}
            disabled={pending}
            onClick={() => flip("approved")}
          >
            <Check className="size-3" />
            Approve
          </Button>
          <Button
            size="xs"
            variant={isRejected ? "destructive" : "outline"}
            disabled={pending}
            onClick={() => flip("rejected")}
          >
            <X className="size-3" />
            Reject
          </Button>
        </div>
      </div>

      {/* Field values grid */}
      <div className="space-y-2">
        {fields.map((field) => (
          <div key={field.column} className="space-y-0.5">
            <div className="flex items-baseline gap-2 text-2xs uppercase tracking-wider text-muted-foreground">
              <span>{field.label}</span>
              <span className="font-mono text-muted-foreground/60 normal-case tracking-normal">
                {field.column}
              </span>
            </div>
            <p className="text-sm whitespace-pre-wrap break-words">
              {field.value || (
                <span className="text-muted-foreground/60 italic">empty</span>
              )}
            </p>
          </div>
        ))}
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}
    </Card>
  );
}
