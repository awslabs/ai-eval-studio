// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Multi-select filter dropdown using Popover + Checkbox.
 * Convention: `selected = []` means all selected (no filter active).
 * When items are unchecked, `selected` contains the checked items.
 * Shows a ListFilter icon and highlighted border when filtering.
 */

import { useState } from "react";
import { ChevronDown, ListFilter } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Checkbox } from "@/components/ui/checkbox";

interface MultiSelectProps {
  /** All available options */
  options: string[];
  /** Currently selected options (empty = all selected) */
  selected: string[];
  /** Callback when selection changes. Empty array means "all". */
  onChange: (selected: string[]) => void;
  /** Placeholder text when all selected */
  placeholder?: string;
  /** Width class */
  className?: string;
  /** Custom label renderer for options */
  renderLabel?: (option: string) => string;
}

/**
 * Dropdown multi-select with checkboxes and exclusion-style display.
 * Empty `selected` array = all selected (no filter active).
 * Shows "-N excluded" count and highlighted border when filtering.
 *
 * "Select All" toggles: when all checked → visually unchecks all (internal
 * state only, onChange still emits [] = all). Clicking any individual item
 * from that state starts a fresh selection.
 */
export function MultiSelect({
  options,
  selected,
  onChange,
  placeholder = "All",
  className = "w-[180px]",
  renderLabel,
}: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  /** Internal flag: user clicked "deselect all" — visually empty but functionally all. */
  const [visuallyEmpty, setVisuallyEmpty] = useState(false);
  const getLabel = (option: string) =>
    renderLabel ? renderLabel(option) : option;

  /* Derive filter state */
  const allSelected = selected.length === 0;
  const isFiltering = !allSelected && selected.length < options.length;
  const excludedCount = allSelected ? 0 : options.length - selected.length;

  /** Toggle a single option, normalizing to [] when all are checked. */
  const toggle = (option: string) => {
    if (visuallyEmpty) {
      /* Was visually empty → select just this one */
      setVisuallyEmpty(false);
      onChange([option]);
      return;
    }
    setVisuallyEmpty(false);
    if (allSelected) {
      /* Was "all" -> deselect this one (keep all others) */
      onChange(options.filter((o) => o !== option));
    } else if (selected.includes(option)) {
      const next = selected.filter((o) => o !== option);
      onChange(next.length === 0 || next.length === options.length ? [] : next);
    } else {
      const next = [...selected, option];
      onChange(next.length === options.length ? [] : next);
    }
  };

  /** Toggle All: all → visually empty; partial → all; visually empty → all. */
  const toggleAll = () => {
    if (allSelected && !visuallyEmpty) {
      setVisuallyEmpty(true);
      // onChange stays [] — no filter change, just visual
    } else {
      setVisuallyEmpty(false);
      onChange([]);
    }
  };

  /* Trigger label */
  const label = allSelected
    ? placeholder
    : isFiltering
      ? (selected.length === 1 ? getLabel(selected[0]) : `${placeholder} (-${excludedCount})`)
      : placeholder;

  /* Visual checkbox state: when visuallyEmpty, nothing is checked */
  const isChecked = (option: string) => {
    if (visuallyEmpty) return false;
    return allSelected || selected.includes(option);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className={`inline-flex items-center gap-2 rounded-md border bg-background px-3 h-8 text-xs hover:bg-accent/50 whitespace-nowrap outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 disabled:pointer-events-none ${
          isFiltering ? "border-primary/40 text-foreground/80" : "border-input text-muted-foreground"
        } ${className}`}
      >
        {isFiltering && <ListFilter className="h-3.5 w-3.5 shrink-0 opacity-60" />}
        <span className="truncate flex-1 text-left">{label}</span>
        <ChevronDown className="h-3.5 w-3.5 opacity-50 shrink-0 ml-auto" />
      </PopoverTrigger>
      <PopoverContent align="start" className="w-auto min-w-[160px] p-1.5">
        <div className="max-h-[280px] overflow-y-auto space-y-0.5">
          {/* Select All row */}
          <label className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs cursor-pointer hover:bg-accent has-focus-visible:bg-accent font-medium border-b border-border mb-0.5 pb-1.5">
            <Checkbox
              checked={allSelected && !visuallyEmpty}
              indeterminate={isFiltering}
              onCheckedChange={toggleAll}
            />
            <span>Select All</span>
          </label>

          {/* Individual options */}
          {options.map((option) => (
            <label
              key={option}
              className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs cursor-pointer hover:bg-accent has-focus-visible:bg-accent"
            >
              <Checkbox
                checked={isChecked(option)}
                onCheckedChange={() => toggle(option)}
              />
              <span className="truncate">{getLabel(option)}</span>
            </label>
          ))}

          {options.length === 0 && (
            <p className="px-2 py-1.5 text-xs text-muted-foreground">
              No options
            </p>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
