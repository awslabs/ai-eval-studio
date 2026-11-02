// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Click-to-edit text with an edit button beside it.
 *
 * Used for the project name and description in the page header. Both were
 * display-only before, and the only way to set either was the create dialog —
 * so a project could never be re-labelled after the moment you knew least
 * about it.
 *
 * On a failed save the component stays in edit mode holding the typed value.
 * Dropping back to display would discard what the user wrote and show the old
 * text, which reads as if the edit had been accepted.
 */

import { useEffect, useRef, useState } from "react";
import { Pencil, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/class-utils";

interface EditableTextProps {
  /** Current persisted value. */
  value: string;
  /** Persist a new value. Rejecting keeps the editor open with the input intact. */
  onSave: (next: string) => Promise<void>;
  /** Field name, used for the button's accessible label. */
  label: string;
  /** Shown in place of an empty value, as the affordance to add one. */
  placeholder?: string;
  /** When false, a blank value is refused rather than saved. */
  allowEmpty?: boolean;
  /** Classes for the rendered text, so a caller controls heading vs body size. */
  className?: string;
}

/**
 * Render `value` with an edit affordance, swapping to an input while editing.
 *
 * @param props - See {@link EditableTextProps}
 * @returns The field in display or edit mode
 */
export function EditableText({
  value,
  onSave,
  label,
  placeholder,
  allowEmpty = true,
  className,
}: EditableTextProps): React.ReactElement {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [isSaving, setIsSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  /* Re-seed the draft whenever the persisted value changes while we are NOT
   * editing. Without the guard, a background refetch landing mid-edit would
   * overwrite what the user is typing. */
  useEffect(() => {
    if (!isEditing) setDraft(value);
  }, [value, isEditing]);

  useEffect(() => {
    if (isEditing) inputRef.current?.focus();
  }, [isEditing]);

  /** Persist the draft, or refuse it when blank and blanks aren't allowed. */
  async function commit(): Promise<void> {
    const next = draft.trim();
    if (next === value.trim()) {
      setIsEditing(false);
      return;
    }
    if (!next && !allowEmpty) {
      toast.error(`${label} cannot be empty`);
      return;
    }
    setIsSaving(true);
    try {
      await onSave(next);
      setIsEditing(false);
    } catch {
      // Stay open so the typed value survives; the api layer surfaces the cause.
      toast.error(`Could not save ${label.toLowerCase()}`);
    } finally {
      setIsSaving(false);
    }
  }

  if (isEditing) {
    return (
      <div className="flex items-center gap-2">
        <Input
          ref={inputRef}
          value={draft}
          disabled={isSaving}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void commit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              setDraft(value);
              setIsEditing(false);
            }
          }}
          /* Commit on blur so clicking away keeps the edit, matching the Enter
           * path. Escape is the discard, and it fires before blur. */
          onBlur={() => void commit()}
          aria-label={label}
          className={cn("h-8", className)}
        />
        {isSaving && (
          <Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" />
        )}
      </div>
    );
  }

  return (
    <div className="group/edit flex items-center gap-2 min-w-0">
      <span
        className={cn(
          "min-w-0 truncate",
          !value && "text-muted-foreground/60 italic",
          className,
        )}
      >
        {value || placeholder}
      </span>
      <button
        type="button"
        onClick={() => setIsEditing(true)}
        className="shrink-0 rounded-md p-1 text-muted-foreground/40 outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring group-hover/edit:text-muted-foreground"
        title={`Edit ${label.toLowerCase()}`}
        aria-label={`Edit ${label.toLowerCase()}`}
      >
        <Pencil className="size-3.5" />
      </button>
    </div>
  );
}
