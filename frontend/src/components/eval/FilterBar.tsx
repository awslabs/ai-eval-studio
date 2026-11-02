// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Collapsible filter bar for eval results.
 *
 * Uses proper multi-select dropdowns for categories and metrics (scales
 * to 50+ items), collapsible section, live case count, and CSV/JSON export.
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, Download, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/ui/multi-select";
import { formatLabel as formatMetadataLabel, formatCategory, shortModel } from "@/lib/format-utils";
import { getCategory, getLatency } from "@/lib/eval-utils";
import { saveBlob } from "@/lib/download-utils";
import type { MultiModelCase, TestResult } from "@/lib/types";

/**
 * Shape of the active filter state for eval results.
 */
export interface EvalFilters {
  /** Selected categories (empty = all). */
  categories: string[];
  /** Pass/fail status filter. Empty = all (no filter). */
  status: string[];
  /** Free-text search applied to input, expected, and actual output. */
  search: string;
  /** Active metric names — only these count toward pass/fail. Null = all active. */
  activeMetrics: string[] | null;
  /** Selected canonical model ids (``ModelRun.hyperparameters.model``)
   *  to display. Empty = all. */
  visibleModels: string[];
  /** Dynamic metadata filters — key is metadata field name, value is selected values. */
  metadataFilters: Record<string, string[]>;
}

interface FilterBarProps {
  allCategories: string[];
  allModels: string[];
  filters: EvalFilters;
  onChange: (filters: EvalFilters) => void;
  totalCases: number;
  filteredCount: number;
  filteredCases: TestResult[];
  metricNames: string[];
  /** Multi-model grid cases for export (replaces filteredCases for multi-model) */
  gridCases?: MultiModelCase[];
  /** Auto-discovered metadata fields with <=20 distinct values, suitable for dropdowns. */
  filterableMetadata?: Record<string, string[]>;
}

/**
 * Collapsible filter bar with multi-select dropdowns for categories and metrics.
 */
export function FilterBar({
  allCategories,
  allModels,
  filters,
  onChange,
  totalCases,
  filteredCount,
  filteredCases,
  metricNames,
  gridCases,
  filterableMetadata,
}: FilterBarProps) {
  const [expanded, setExpanded] = useState(false);
  const isFiltered = filteredCount !== totalCases;
  const hasMetadataFilters = Object.values(filters.metadataFilters ?? {}).some((v) => v.length > 0);
  const hasActiveFilters = filters.categories.length > 0 || filters.status.length > 0 || filters.search || filters.activeMetrics !== null || filters.visibleModels.length > 0 || hasMetadataFilters;

  return (
    <div className="border rounded-lg bg-card shadow-sm">
      {/* Header row — always visible */}
      <div className="flex items-center gap-3 px-3 py-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 rounded text-xs font-medium text-muted-foreground hover:text-foreground transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {expanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
          Filters
        </button>

        {hasActiveFilters && !expanded && (
          <span className="text-2xs text-primary font-medium">Active</span>
        )}

        <span className="text-xs text-muted-foreground tabular-nums ml-auto">
          {isFiltered ? (
            <>Showing <span className="font-semibold text-foreground">{filteredCount}</span> of {totalCases} cases</>
          ) : (
            <><span className="font-semibold text-foreground">{totalCases}</span> cases</>
          )}
        </span>

        {hasActiveFilters && (
          <Button variant="ghost" size="sm" className="h-6 px-2 text-2xs text-muted-foreground" onClick={() => onChange({ categories: [], status: [], search: "", activeMetrics: null, visibleModels: [], metadataFilters: {} })} title="Clear all filters">
            <X className="size-3 mr-0.5" />Clear
          </Button>
        )}

        <div className="flex gap-1">
          <Button variant="outline" size="sm" className="h-6 px-2 text-2xs" onClick={() => {
            if (gridCases && gridCases.length > 0) {
              exportMultiModelCSV(gridCases, allModels, metricNames);
            } else {
              exportCSV(filteredCases, metricNames);
            }
          }} title="Export CSV">
            <Download className="size-3 mr-0.5" />CSV
          </Button>
          <Button variant="outline" size="sm" className="h-6 px-2 text-2xs" onClick={() => {
            if (gridCases && gridCases.length > 0) {
              exportMultiModelJSON(gridCases);
            } else {
              exportJSON(filteredCases);
            }
          }} title="Export JSON">
            <Download className="size-3 mr-0.5" />JSON
          </Button>
        </div>
      </div>

      {/* Expandable filter controls */}
      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t">
          <div className="flex flex-wrap items-center gap-3">
            {/* Models */}
            {allModels.length > 1 && (
              <div className="flex items-center gap-1.5">
                <span className="text-2xs text-muted-foreground">Models:</span>
                <MultiSelect
                  options={allModels}
                  selected={filters.visibleModels}
                  onChange={(visibleModels) => onChange({ ...filters, visibleModels })}
                  placeholder="All models"
                  renderLabel={shortModel}
                />
              </div>
            )}

            {/* Categories */}
            {allCategories.length > 0 && (
              <div className="flex items-center gap-1.5">
                <span className="text-2xs text-muted-foreground">Category:</span>
                <MultiSelect
                  options={allCategories}
                  selected={filters.categories}
                  onChange={(categories) => onChange({ ...filters, categories })}
                  placeholder="All categories"
                  renderLabel={formatCategory}
                />
              </div>
            )}

            {/* Dynamic metadata filters */}
            {filterableMetadata && Object.entries(filterableMetadata).map(([key, values]) => (
              <div key={key} className="flex items-center gap-1.5">
                <span className="text-2xs text-muted-foreground">{formatMetadataLabel(key)}:</span>
                <MultiSelect
                  options={values}
                  selected={filters.metadataFilters?.[key] ?? []}
                  onChange={(selected) => onChange({
                    ...filters,
                    metadataFilters: { ...filters.metadataFilters, [key]: selected },
                  })}
                  placeholder={`All`}
                />
              </div>
            ))}

            {/* Status */}
            <div className="flex items-center gap-1.5">
              <span className="text-2xs text-muted-foreground">Status:</span>
              <MultiSelect
                options={STATUS_OPTIONS}
                selected={filters.status}
                onChange={(status) => onChange({ ...filters, status })}
                placeholder="All"
              />
            </div>

            {/* Metrics */}
            {metricNames.length > 1 && (
              <div className="flex items-center gap-1.5">
                <span className="text-2xs text-muted-foreground">Metrics:</span>
                <MultiSelect
                  options={metricNames}
                  selected={filters.activeMetrics ?? []}
                  onChange={(selected) => onChange({ ...filters, activeMetrics: selected.length === 0 ? null : selected })}
                  placeholder="All metrics"
                />
              </div>
            )}

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3 text-muted-foreground" />
              <Input
                className="h-8 text-xs w-[200px] pl-7"
                placeholder="Search all text..."
                value={filters.search}
                onChange={(e) => onChange({ ...filters, search: e.target.value })}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ================================================================
 * Status dropdown — pass/fail/different buckets.
 *
 * "Different" only has meaning in multi-model runs (some models passed
 * while others failed on the same case). The option stays in the list
 * because removing it per-run would complicate the MultiSelect contract;
 * on single-model runs selecting it simply narrows to zero rows, which
 * tells the user the filter doesn't apply.
 * ================================================================ */

const STATUS_OPTIONS = ["Passed", "Failed", "Different"];

/* ================================================================ */

/** Export multi-model grid cases to CSV with per-model columns.
 *
 * Column headers use the canonical short name (``shortModel``) so a
 * dotted-version model id like ``us.anthropic.claude-opus-4-8`` becomes
 * ``output__claude-opus-4-8`` — readable in Excel/BI tools and stable
 * across consumers that previously joined on the short id. The cell
 * lookup keeps using the canonical full id (the keys of
 * ``MultiModelCase.models``) — only the rendered header changes. */
function exportMultiModelCSV(cases: MultiModelCase[], modelIds: string[], metricNames: string[]) {
  const shortIds = modelIds.map(shortModel);
  const header = [
    "case_id", "category", "input", "expected_output",
    ...shortIds.flatMap((s) => [`output__${s}`, `pass__${s}`]),
    ...metricNames.flatMap((n) => shortIds.map((s) => `${n}__${s}`)),
  ];
  const rows = cases.map((c) => [
    // case_id and category come from the user's dataset columns, so they need
    // escaping as much as the text fields do. Passing them through raw meant a
    // comma or quote in either one broke the row's column structure rather than
    // just its own cell.
    csvEscape(c.case_id), csvEscape(c.category ?? ""),
    csvEscape(c.input), csvEscape(c.expected_output),
    ...modelIds.flatMap((m) => {
      const d = c.models[m];
      return [csvEscape(d?.actual_output ?? ""), d?.success ? "PASS" : "FAIL"];
    }),
    ...metricNames.flatMap((n) => modelIds.map((m) => {
      const md = c.models[m]?.metrics_data?.find((x) => x.name === n);
      return md?.score != null ? (md.score * 100).toFixed(1) : "";
    })),
  ]);
  saveBlob([header.join(","), ...rows.map((r) => r.join(","))].join("\n"), "eval-results.csv", "text/csv");
}

/** Export multi-model grid cases to JSON. */
function exportMultiModelJSON(cases: MultiModelCase[]) {
  saveBlob(JSON.stringify(cases, null, 2), "eval-results.json", "application/json");
}

function exportCSV(cases: TestResult[], metricNames: string[]) {
  const header = ["case_id", "category", "passed", "input", "expected_output", "actual_output", "latency_ms", ...metricNames.map((n) => `metric:${n}`)];
  const rows = cases.map((c, idx) => {
    const caseId = String(c.additional_metadata?.id ?? c.name ?? `case_${idx}`);
    const latencyMs = getLatency(c);
    return [
      // Both derive from user dataset columns — escape them like any other text.
      csvEscape(caseId), csvEscape(getCategory(c) ?? ""), c.success ? "true" : "false",
      csvEscape(c.input), csvEscape(c.expected_output), csvEscape(c.actual_output),
      latencyMs != null ? latencyMs.toFixed(0) : "",
      ...metricNames.map((n) => { const m = c.metrics_data.find((md) => md.name === n); return m?.score != null ? (m.score * 100).toFixed(1) : ""; }),
    ];
  });
  saveBlob([header.join(","), ...rows.map((r) => r.join(","))].join("\n"), "eval-results.csv", "text/csv");
}

function exportJSON(cases: TestResult[]) {
  saveBlob(JSON.stringify(cases, null, 2), "eval-results.json", "application/json");
}

/** Quote and escape one CSV field, neutralising spreadsheet formulas.
 *
 * Two jobs, both needed. RFC 4180 quoting keeps a value that contains a comma,
 * quote, newline, carriage return or tab inside its own cell — without it, a
 * comma in any field silently shifts every later column of that row. Tab was
 * missing from the old test, and a tab inside an unquoted field splits the cell
 * in Excel.
 *
 * The formula prefix is separate. A cell starting with `=`, `+`, `-` or `@` is
 * evaluated as a formula when the file is opened in a spreadsheet, which turns
 * exported model output into something that runs on the reader's machine. A
 * leading apostrophe makes the spreadsheet treat it as text. It is visible in
 * the cell, which is the tradeoff: this export is for reading, and
 * ``exportJSON`` is the byte-exact option. */
export function csvEscape(value: string): string {
  const text = value ?? "";
  const formulaSafe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[,"\n\r\t]/.test(formulaSafe)
    ? `"${formulaSafe.replace(/"/g, '""')}"`
    : formulaSafe;
}
