// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared affordances for clickable surfaces (rows and cards).
 *
 * Centralizes the hover + focus-visible + selected treatment and keyboard
 * activation so list rows and clickable cards stop re-implementing (and
 * forgetting) them. Two shapes:
 *  - {@link SelectableRow}: a full-width ``<button>`` for simple rows.
 *  - {@link clickableCardClass} + {@link onActivateKeyDown}: for cards that
 *    contain nested interactive elements and therefore can't be a ``<button>``.
 */

import type { KeyboardEvent, ReactNode } from "react";
import { cn } from "@/lib/class-utils";

/** Shared hover/focus/selected classes for a clickable list row. */
export const selectableRowClass =
  "w-full text-left rounded-md transition-colors outline-none cursor-pointer " +
  "hover:bg-accent/50 focus-visible:bg-accent focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring " +
  "aria-[current=true]:bg-accent aria-[current=true]:text-accent-foreground";

/**
 * Classes for a clickable Card (one that contains nested buttons, so it
 * must stay a ``<div>``/``role=button`` rather than a real ``<button>``).
 * Pair with {@link onActivateKeyDown}, ``role="button"``, and ``tabIndex={0}``.
 */
export const clickableCardClass =
  "cursor-pointer outline-none transition-shadow transition-colors " +
  "hover:border-ring/40 hover:shadow-md " +
  "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background";

/**
 * Keyboard handler that activates a non-button clickable surface on
 * Enter/Space (matching native button behavior).
 *
 * @param handler - Called when the surface is activated.
 * @returns An ``onKeyDown`` handler.
 */
export function onActivateKeyDown(handler: () => void) {
  return (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handler();
    }
  };
}

/** Props for {@link SelectableRow}. */
interface SelectableRowProps {
  /** Selected/active state (sets ``aria-current``). */
  selected?: boolean;
  /** Activation handler. */
  onSelect: () => void;
  /** Row content. */
  children: ReactNode;
  /** Extra classes (layout/padding supplied by the caller). */
  className?: string;
  /** Accessible label when content isn't self-describing. */
  "aria-label"?: string;
}

/**
 * A full-width, keyboard-operable selectable list row.
 *
 * Renders a ``<button>`` so Enter/Space, focus, and screen-reader semantics
 * come for free. Use for file-tree entries, batch lists, etc. Do NOT use when
 * the row contains its own buttons — use {@link clickableCardClass} instead.
 */
export function SelectableRow({
  selected,
  onSelect,
  children,
  className,
  "aria-label": ariaLabel,
}: SelectableRowProps) {
  return (
    <button
      type="button"
      aria-current={selected || undefined}
      aria-label={ariaLabel}
      onClick={onSelect}
      className={cn(selectableRowClass, className)}
    >
      {children}
    </button>
  );
}
