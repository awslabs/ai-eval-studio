// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Eval results dashboard rendered as the "Evaluations" tab content.
 * Fetches eval run data (DeepEval-native ModelRun[]) and displays:
 * - Multi-model summary charts (pass rate, average score, latency by model)
 * - Multi-model results grid with per-model columns and expandable details
 * - Analytics charts for all models in the selected run
 * - Run comparison, prompts viewer, raw JSON
 *
 * Active-job status (optimization + eval progress) lives in the
 * ``ActiveJobBar`` rendered by ``ProjectPage`` above all tabs, not
 * inside this component.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Download, Loader2, LayoutDashboard, Table2, LineChart, GitCompare, FileJson } from "lucide-react";
import Editor from "@monaco-editor/react";
import { defineAppTheme, getMonacoTheme } from "@/lib/monaco-theme";
import { useTheme } from "@/lib/theme-provider";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import type {
  EvalRunSummary,
  ModelRun,
} from "@/lib/types";
import { applyMetricFilter, pivotToGrid, discoverFilterableMetadata } from "@/lib/eval-utils";
import { saveBlob } from "@/lib/download-utils";
import { shortModel } from "@/lib/format-utils";
import { ResultsTable } from "./ResultsTable";
import { FilterBar, type EvalFilters } from "./FilterBar";
import { AnalyticsPanel, SummaryChartsRow } from "./AnalyticsPanel";
import { ComparePanel } from "./compare";
import { OptimizationRunDetail, convergenceCriteriaFromHarness } from "./OptimizationRunDetail";
import { useOptimizationState } from "./OptimizationStateProvider";
import { useProjectEvent, useCursorGapHandler } from "@/hooks/useBackendEvents";
import { useHarnessConfig } from "@/hooks/useHarnessConfig";
import { newestRun } from "@/lib/run-list";

/** Props for the EvalDashboard component. */
interface EvalDashboardProps {
  projectId: string;
}

export function EvalDashboard({ projectId }: EvalDashboardProps) {
  const [searchParams, setSearchParams] = useSearchParams();

  const [runs, setRuns] = useState<EvalRunSummary[]>([]);
  const [selectedFile, setSelectedFileState] = useState<string | null>(searchParams.get("run"));
  const [runData, setRunData] = useState<ModelRun[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  /** Run-list fetch error. Suppresses the dashboard body (with no list there is
   *  nothing to select). This blocking behavior is deliberate. */
  const [runsError, setRunsError] = useState<string | null>(null);
  /** Selected run's detail fetch error. Renders its own line and does NOT
   *  suppress the selector, so a persistent "Run not found" is recoverable by
   *  picking another run. */
  const [runDetailError, setRunDetailError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(searchParams.get("dtab") ?? "overview");

  /* Optimization state comes from the provider so it can't drift from the
   * status bar above. Previously this component kept its own copy fetched
   * on window focus, which would routinely disagree with the status bar's
   * polled copy. */
  const { state: optState, isLoaded } = useOptimizationState();

  /* Harness config supplies convergence criteria (targets, plateau threshold,
   * plateau window, max iterations). Sourced at the dashboard level so the
   * panel stays presentational (no hook calls, no projectId prop). */
  const { config: harnessConfig } = useHarnessConfig(projectId);
  const [filters, setFilters] = useState<EvalFilters>({
    categories: [],
    status: [],
    search: "",
    activeMetrics: null,
    visibleModels: [],
    metadataFilters: {},
  });

  const setSelectedFile = useCallback((file: string | null) => {
    setSelectedFileState(file);
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (file) next.set("run", file); else next.delete("run");
      return next;
    }, { replace: true });
  }, [setSearchParams]);

  /* ---- Derived multi-model data ---- */

  const allModelIds = useMemo(() => (runData ?? []).map((r) => r.hyperparameters.model), [runData]);

  /** Every metric name observed in the run, regardless of filter — used to
   *  populate the FilterBar's Metrics dropdown. */
  const allMetricNames = useMemo(() => {
    if (!runData) return [];
    const names = new Set<string>();
    for (const run of runData) for (const r of run.results) for (const md of r.metrics_data) names.add(md.name);
    return Array.from(names);
  }, [runData]);

  /** Run data with the active-metrics filter applied at the source.
   *  Everything below the filter bar derives from this — pass-rate,
   *  charts, grid, sub-tabs — so the metric filter narrows the whole
   *  page consistently. ``filteredRunData`` and ``filteredGridCases``
   *  layer the remaining (category / status / search / model /
   *  metadata) filters on top. The Raw Data tab is the one exception
   *  and reads ``runData`` directly. */
  const metricNarrowedRunData = useMemo(
    () => (runData ? applyMetricFilter(runData, filters.activeMetrics) : null),
    [runData, filters.activeMetrics],
  );

  /** Metric names that survived the active-metrics filter — used for
   *  rendering metric columns/chips downstream. ``allMetricNames``
   *  stays full so the FilterBar dropdown lists every option. */
  const visibleMetricNames = useMemo(() => {
    if (!filters.activeMetrics) return allMetricNames;
    return allMetricNames.filter((n) => filters.activeMetrics!.includes(n));
  }, [allMetricNames, filters.activeMetrics]);

  const gridCases = useMemo(
    () => (metricNarrowedRunData ? pivotToGrid(metricNarrowedRunData) : []),
    [metricNarrowedRunData],
  );

  const categories = useMemo(() => {
    const cats = new Set<string>();
    for (const c of gridCases) if (c.category) cats.add(c.category);
    return Array.from(cats).sort();
  }, [gridCases]);

  const filterableMetadata = useMemo(() => discoverFilterableMetadata(gridCases), [gridCases]);

  const filteredGridCases = useMemo(() => {
    if (gridCases.length === 0) return [];
    return gridCases.filter((c) => {
      if (filters.categories.length > 0 && !filters.categories.includes(c.category ?? "")) return false;
      if (filters.search) {
        const q = filters.search.toLowerCase();
        const matchesBase = c.input.toLowerCase().includes(q) || c.expected_output.toLowerCase().includes(q);
        const matchesModel = Object.values(c.models).some((m) => m.actual_output.toLowerCase().includes(q));
        if (!matchesBase && !matchesModel) return false;
      }
      if (filters.status.length > 0) {
        const visModels = filters.visibleModels.length > 0 ? filters.visibleModels : allModelIds;
        const results = visModels.map((m) => c.models[m]?.success).filter((v) => v != null);
        const isPassed = results.every(Boolean);
        const isFailed = results.every((v) => !v);
        const isDifferent = results.length >= 2 && results.some(Boolean) && !results.every(Boolean);
        const matchesAny = filters.status.some((s) => {
          if (s === "Passed") return isPassed;
          if (s === "Failed") return isFailed;
          if (s === "Different") return isDifferent;
          return false;
        });
        if (!matchesAny) return false;
      }
      /* Dynamic metadata filters */
      for (const [key, values] of Object.entries(filters.metadataFilters ?? {})) {
        if (values.length === 0) continue;
        const anyModelMatches = Object.values(c.models).some(
          (m) => values.includes(String(m.additional_metadata?.[key] ?? ""))
        );
        if (!anyModelMatches) return false;
      }
      return true;
    });
  }, [gridCases, filters, allModelIds]);

  /** ModelRun[] with the metric filter already applied + the remaining
   *  case-level filters layered on top. Used by SummaryChartsRow and
   *  AnalyticsPanel so charts respect every filter dimension consistently.
   *  ``r.success`` here has been recomputed against the active metrics
   *  (see ``applyMetricFilter``) so the headline pass-rate and status
   *  filter line up with the metric subset the user picked. */
  const filteredRunData = useMemo(() => {
    if (!metricNarrowedRunData) return null;
    const visModels = filters.visibleModels.length > 0 ? filters.visibleModels : allModelIds;
    return metricNarrowedRunData
      .filter((mr) => visModels.includes(mr.hyperparameters.model))
      .map((mr) => {
        const filtered = mr.results.filter((r) => {
          /* Category filter */
          if (filters.categories.length > 0) {
            const cat = (r.additional_metadata?.category as string) ?? "";
            if (!filters.categories.includes(cat)) return false;
          }
          /* Search filter */
          if (filters.search) {
            const q = filters.search.toLowerCase();
            if (!r.input.toLowerCase().includes(q) && !r.expected_output.toLowerCase().includes(q) && !r.actual_output.toLowerCase().includes(q)) return false;
          }
          /* Status filter — reads the recomputed ``r.success`` so
           * "Passed"/"Failed" reflects the active metric subset. */
          if (filters.status.length > 0) {
            const matchesAny = filters.status.some((s) => {
              if (s === "Passed") return r.success;
              if (s === "Failed") return !r.success;
              return false;
            });
            if (!matchesAny) return false;
          }
          /* Dynamic metadata filters */
          for (const [key, values] of Object.entries(filters.metadataFilters ?? {})) {
            if (values.length === 0) continue;
            if (!values.includes(String(r.additional_metadata?.[key] ?? ""))) return false;
          }
          return true;
        });
        return { ...mr, results: filtered };
      })
      .filter((mr) => mr.results.length > 0);
  }, [metricNarrowedRunData, filters, allModelIds]);

  /* ---- Data fetching ---- */

  /** Ref to setSelectedFile to avoid re-creating fetchRuns on every render. */
  const setSelectedFileRef = useRef(setSelectedFile);
  setSelectedFileRef.current = setSelectedFile;

  /** Current selection, so fetchRuns can respect it without depending on it. */
  const selectedFileRef = useRef(selectedFile);
  selectedFileRef.current = selectedFile;

  /** Initial fetch — shows loading state and auto-selects the newest run.
   *
   * Auto-select only when nothing is selected yet. ``selectedFile`` is seeded
   * from the ``?run=`` query param, so overwriting it unconditionally discarded
   * whatever run a shared link pointed at — and because selecting also rewrites
   * that param, the URL sync could not restore it afterwards. The Reports tab
   * has always had this guard on its own ``?rrun=`` param. */
  const fetchRuns = useCallback(() => {
    setIsLoading(true);
    setRunsError(null);
    api.listEvalRuns(projectId)
      .then((data) => {
        setRuns(data);
        // Keep an existing selection only if that run is still there; a link to
        // a deleted run would otherwise leave the tab stuck on nothing.
        const current = selectedFileRef.current;
        if (current && data.some((r) => r._file === current)) return;
        const newest = newestRun(data);
        if (newest) setSelectedFileRef.current(newest._file);
      })
      .catch((e) => setRunsError(e.message))
      .finally(() => setIsLoading(false));
  }, [projectId]);

  /** Background refresh — updates run list quietly without resetting selection or showing spinner. */
  const refreshRunsSilently = useCallback(() => {
    api.listEvalRuns(projectId)
      .then((data) => setRuns(data))
      .catch(() => {
        /* Silent path — the next refresh (focus, ws event, backstop) retries. */
      });
  }, [projectId]);

  useEffect(() => { fetchRuns(); }, [fetchRuns]);

  /* Refresh on window focus (user might have new out-of-band runs)
   * and on eval-job completion events from the project bus. The
   * optimization state itself is NOT refreshed here — the provider
   * polls it centrally. */
  useEffect(() => {
    const onFocus = () => refreshRunsSilently();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refreshRunsSilently]);

  useProjectEvent(
    { resource: "eval_job", type: "completed" },
    () => refreshRunsSilently(),
    "state"
  );
  useProjectEvent(
    { resource: "job", type: "completed" },
    () => refreshRunsSilently(),
    "state"
  );

  /* A run deleted out-of-band (the agent's ``delete_run`` tool, or another
   * tab) leaves this list holding a file that no longer exists. This handler
   * only refreshes the list — moving the selection off a deleted run is owned
   * by the reconciliation effect below, keyed on the runs list itself.
   *
   * It used to also clear the selection and take the full fetch path when the
   * deleted run was the selected one. That made selection movement depend on
   * WHICH refetch path observed the deletion: this handler moved it, but the
   * concurrent cursor-gap refetch (or focus, or the iteration backstop) did
   * not — and whichever resolved last decided whether the UI kept a stale
   * "Run not found". One owner, every path converges. */
  useProjectEvent(
    { resource: "eval_run", type: "deleted" },
    () => refreshRunsSilently(),
    "state"
  );

  /* When the optimization loop advances to a new iteration the eval run
   * list gains a new entry. The eval-job event above usually covers it;
   * this is a backstop driven off the provider's state so we never miss
   * the visual update even if an event got dropped on reconnect. */
  const iterationCount = optState?.iterations?.length ?? 0;
  useEffect(() => {
    if (iterationCount > 0) refreshRunsSilently();
  }, [iterationCount, refreshRunsSilently]);

  /* Cursor gap signal: any of three conditions indicating frames were missed or
   * unservable. Covers unservable replay (restart, TTL, cap eviction), degraded
   * publish (allocation failure), and live forward hole (seq > cursor + 1). An
   * empty replay or a missed frame would leave the list silently stale, which
   * is the failure this signal exists to prevent. The honest response is to
   * refetch the list outright; deliberately NOT to adjust the cursor, which
   * would reintroduce a second numbering authority. */
  useCursorGapHandler(projectId, () => refreshRunsSilently());

  // Sync selectedFile from the URL so external writes (e.g. the
  // optimization iteration cards calling setSearchParams) trigger a
  // refetch. Without this, ?run= updates change the address bar but the
  // panel keeps rendering the stale run.
  const urlRun = searchParams.get("run");
  useEffect(() => {
    /* Only adopt a ``?run=`` that names a run we actually have. Without the
     * membership check this effect resurrects a just-deleted selection: once
     * the selection is cleared, this effect re-sets it from the still-stale
     * URL param, the restarted detail fetch 404s and re-arms the error — so
     * clearing on a null selection alone would not hold at the empty-set
     * boundary.
     *
     * Skipping while ``runs`` is empty is safe: a real ``?run=`` is adopted as
     * soon as the list arrives, since ``runs`` is in the deps. */
    if (urlRun && urlRun !== selectedFile && runs.some((r) => r._file === urlRun)) {
      setSelectedFileState(urlRun);
    }
  }, [urlRun, selectedFile, runs]);

  /* Selection↔runs reconciliation — the single owner of "the selection must
   * name a member of the run list".
   *
   * Keyed on the runs list itself so EVERY refetch path converges the
   * selection: the delete event, cursor-gap, focus, the iteration backstop,
   * and any path added later. Owning this per call site breaks when refetch
   * paths race: if the delete handler moves the selection but a concurrent
   * cursor-gap refetch does not, the stale selection 404s into a permanent
   * "Run not found".
   *
   * The prev-membership guard is load-bearing: only a selection this session
   * actually OBSERVED in the list is reconciled away when it disappears —
   * that transition is a deletion. A hand-typed bogus ``?run=`` was never a
   * member, so it stays selected and its "Run not found" persists,
   * deliberately: that error describes a real, current, unreadable
   * selection, and silently displacing it would hide that the URL is bad. */
  const prevRunFilesRef = useRef<Set<string> | null>(null);
  useEffect(() => {
    const files = new Set(runs.map((r) => r._file));
    const prev = prevRunFilesRef.current;
    prevRunFilesRef.current = files;
    if (!selectedFile || files.has(selectedFile)) return;
    if (prev?.has(selectedFile)) {
      setSelectedFile(newestRun(runs)?._file ?? null);
    }
  }, [runs, selectedFile, setSelectedFile]);

  useEffect(() => {
    /* The empty-set boundary. Deleting the LAST run leaves no selectable run,
     * and neither detail-error clearing rule covers that case on its own:
     * clear-on-selection-change (below) runs only for a non-null selection,
     * and clear-on-successful-fetch never applies because the LIST fetch
     * succeeds (count=0) while it is the DETAIL fetch that 404s — separate
     * identities, correctly so. Without this branch the error would outlive
     * the selection it described and no later refetch could clear it.
     *
     * Keyed on the SELECTION, not on an empty list: zero runs plus a
     * hand-typed bad ``?run=`` must keep showing "Run not found" (the selection
     * is real and unreadable), so an empty-list key would wrongly clear it. */
    if (!selectedFile) {
      setRunDetailError(null);
      setRunData(null);
      return;
    }

    // Clear detail error and data when selection changes.
    setRunDetailError(null);
    setRunData(null);

    // Cancellation flag so in-flight fetches for superseded selections
    // cannot write either data or error. A fetch for a deleted run can
    // reject after the selection has already moved on; without this guard
    // that late rejection re-arms an error describing a run the user no
    // longer has selected.
    let cancelled = false;

    api.getEvalRun(projectId, selectedFile)
      .then((data) => {
        if (cancelled) return;
        setRunData(data);
        // Clear error on successful fetch (redundant with effect-start clear
        // for selection changes, but holds for future refetch paths that don't
        // change the selection — e.g. retry button, WS-driven detail refresh).
        setRunDetailError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setRunDetailError(e.message);
      });

    return () => {
      cancelled = true;
    };
  }, [projectId, selectedFile]);

  /* ---- Render ---- */

  // The detail fetch is genuinely pending only when there is no data AND no
  // detail error. Derived from existing states so Results/Analysis/Raw tabs
  // don't claim "Loading..." when a detail fetch failed and the error is on
  // screen above.
  const detailFetchPending = !runData && !runDetailError;

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-4">
      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center gap-2 text-muted-foreground py-8">
          <Loader2 className="size-4 animate-spin" />
          <span>Loading eval runs...</span>
        </div>
      )}

      {/* Run-list error (blocking — suppresses dashboard body) */}
      {!isLoading && runsError && (
        <p className="text-error text-sm py-4">{runsError}</p>
      )}

      {/* Run-detail error (non-blocking — selector survives so user can pick
          another run). List error is blocking, detail error is not — this
          asymmetry is deliberate: with no list there is nothing to select, but
          one failed detail load out of several should not destroy the recovery
          path (clicking another run). */}
      {!isLoading && !runsError && runDetailError && (
        <p className="text-error text-sm py-4">{runDetailError}</p>
      )}

      {/* Empty state */}
      {!isLoading && !runsError && !runDetailError && runs.length === 0 && (
        <p className="text-muted-foreground text-sm py-4">No eval runs yet. Use the chat to run an evaluation.</p>
      )}

      {/* Run selector — only when runs exist. Detail error no longer suppresses
          this block (the selector must survive so another run can be picked). */}
      {!isLoading && !runsError && runs.length > 0 && (
        <>
        <div className="space-y-1 shrink-0">
        <label className="text-xs font-medium text-muted-foreground">Eval Run</label>
        <Select value={selectedFile ?? ""} onValueChange={setSelectedFile}>
          <SelectTrigger className="w-full h-9 text-xs">
            <SelectValue placeholder="Select a run">
              {(() => {
                const r = runs.find((r) => r._file === selectedFile);
                if (!r) return "Select a run";
                const modelNames = r.model_runs.map((m) => shortModel(m.model)).join(", ");
                const caseCount = r.model_runs[0]?.total ?? 0;
                return `${r.prompt_version} — ${modelNames} (${caseCount} cases, ${new Date(r.timestamp).toLocaleDateString()} ${new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })})`;
              })()}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {runs.map((r) => {
              const modelNames = r.model_runs.map((m) => shortModel(m.model)).join(", ");
              const caseCount = r.model_runs[0]?.total ?? 0;
              // Read source from the run file itself — permanent attribution
              // written by the harness at run time. Survives state resets.
              const sourceTag = r.source === "optimization"
                ? " · optimization loop"
                : " · manual run";
              const dateStr = `${new Date(r.timestamp).toLocaleDateString()} ${new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
              return (
                <SelectItem key={r._file} value={r._file}>
                  {r.prompt_version} — {modelNames} ({caseCount} cases, {dateStr}{sourceTag})
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
        </div>

      {/* Metadata pills */}
      {runData && runData.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <MetaPill label="Date" value={new Date(runData[0].timestamp).toLocaleString()} />
          <MetaPill label="Models" value={String(runData.length)} />
          <MetaPill label="Tests" value={String(gridCases.length)} />
          <MetaPill label="Prompt" value={String(runData[0].hyperparameters.prompt_version ?? "")} />
        </div>
      )}

      {/* Filters — above tabs, applies to all subtabs */}
      {runData && (
        <div className="shrink-0">
          <FilterBar
            allCategories={categories}
            allModels={allModelIds}
            filters={filters}
            onChange={setFilters}
            totalCases={gridCases.length}
            filteredCount={filteredGridCases.length}
            filteredCases={[]}
            gridCases={filteredGridCases}
            metricNames={allMetricNames}
            filterableMetadata={filterableMetadata}
          />
        </div>
      )}

      <Tabs
        className="flex-1 min-h-0"
        value={activeTab}
        onValueChange={(tab) => {
          setActiveTab(tab);
          setSearchParams((prev) => { const next = new URLSearchParams(prev); next.set("dtab", tab); return next; }, { replace: true });
        }}
      >
        <TabsList className="shrink-0">
          <TabsTrigger value="overview"><LayoutDashboard className="size-3.5" />Overview</TabsTrigger>
          <TabsTrigger value="results"><Table2 className="size-3.5" />Results</TabsTrigger>
          <TabsTrigger value="analysis"><LineChart className="size-3.5" />Analysis</TabsTrigger>
          <TabsTrigger value="compare"><GitCompare className="size-3.5" />Compare</TabsTrigger>
          <TabsTrigger value="raw"><FileJson className="size-3.5" />Raw Data</TabsTrigger>
        </TabsList>

        {/* Overview tab — optimization run detail + summary charts. Flex
            column: the optimization card is natural height and the charts
            row is a fixed 300px (both shrink-0); if the viewport is too
            short, overflow-auto scrolls instead. */}
        <TabsContent value="overview" className="overflow-auto flex flex-col gap-4">
          {/* Optimization panel — show whenever this project has runs and
              optimization state has settled. The gate is "runs exist", not
              "OPT record exists" — a project can have runs on disk and no
              loop record, and the panel renders with state=null. isLoaded
              guarantees state has settled so the other-runs checkbox default
              (checked unless some iteration's run file survives) is computed
              at mount time on stable data. The panel receives the full
              unfiltered run list and partitions it internally — card
              existence is decided by the run file, so the panel needs every
              run to resolve which iterations survive. */}
          {runs.length > 0 && isLoaded && (
            <div className="shrink-0">
              <OptimizationRunDetail
                state={optState}
                criteria={convergenceCriteriaFromHarness(harnessConfig)}
                runs={runs}
              />
            </div>
          )}

          {/* Summary charts — read filteredRunData (every filter dimension
              already applied: model, category, status, search, metric,
              metadata). Fixed 300px tall whether or not the optimization card
              is present, so the charts stay a consistent, scannable size and
              the rest of the tab scrolls below them. */}
          {filteredRunData && filteredRunData.length > 0 && (
            <div className="flex flex-col shrink-0" style={{ height: 300 }}>
              <SummaryChartsRow modelRuns={filteredRunData} fill />
            </div>
          )}
        </TabsContent>

        {/* Results tab — multi-model grid */}
        <TabsContent value="results" className="flex-1 min-h-0 flex flex-col">
          {runData ? (
            <ResultsTable
              cases={filteredGridCases}
              // Pass the UNFILTERED run data so the table keeps every
              // model's column regardless of active status/category
              // filters. Previously, filtering out every row a model
              // had would drop its column and reshape the whole table.
              modelRuns={runData}
              visibleModels={filters.visibleModels}
              metricNames={visibleMetricNames}
            />
          ) : detailFetchPending ? (
            <div className="flex items-center gap-2 text-muted-foreground py-4">
              <Loader2 className="size-4 animate-spin" />
              <span>Loading results...</span>
            </div>
          ) : null}
        </TabsContent>

        {/* Analysis tab — multi-model charts */}
        <TabsContent value="analysis" className="overflow-auto">
          {runData ? (
            <AnalyticsPanel modelRuns={filteredRunData ?? []} />
          ) : detailFetchPending ? (
            <div className="flex items-center gap-2 text-muted-foreground py-4">
              <Loader2 className="size-4 animate-spin" />
              <span>Loading analytics...</span>
            </div>
          ) : null}
        </TabsContent>

        <TabsContent value="compare" keepMounted className="overflow-auto">
          <ComparePanel projectId={projectId} runs={runs} visibleModels={filters.visibleModels} activeMetrics={filters.activeMetrics} />
        </TabsContent>

        <TabsContent value="raw" className="flex-1 min-h-0 flex flex-col">
          {runData ? (
            <RawDataView data={runData} />
          ) : detailFetchPending ? (
            <p className="text-sm text-muted-foreground py-4">Select a run to view raw data.</p>
          ) : null}
        </TabsContent>
      </Tabs>
      </>
      )}
    </div>
  );
}

/** Raw data viewer with Monaco JSON editor and export. */
function RawDataView({ data }: { data: ModelRun[] }) {
  const totalCases = data.reduce((sum, mr) => sum + mr.results.length, 0);
  const { mode } = useTheme();
  const isDark = mode === "dark";
  const jsonStr = useMemo(() => JSON.stringify(data, null, 2), [data]);

  const downloadJSON = () => saveBlob(jsonStr, "eval-results.json", "application/json");


  return (
    <div className="flex flex-col flex-1 min-h-0 gap-3">
      <div className="flex items-center gap-3 shrink-0">
        <Button variant="outline" size="sm" className="h-7 text-xs" onClick={downloadJSON}><Download className="size-3 mr-1" />Export JSON</Button>
        <span className="text-xs text-muted-foreground">{totalCases} cases &middot; {data.length} model{data.length !== 1 ? "s" : ""}</span>
      </div>
      <div className="border rounded-lg overflow-hidden bg-card flex-1 min-h-0">
        <Editor
          height="100%"
          language="json"
          value={jsonStr}
          theme={getMonacoTheme(isDark)}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            fontSize: 12,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            wordWrap: "on",
            folding: true,
            foldingStrategy: "indentation",
            padding: { top: 8 },
          }}
          beforeMount={defineAppTheme}
        />
      </div>
    </div>
  );
}

/** Compact metadata pill. */
function MetaPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 bg-muted border border-border/50 rounded-md px-2.5 py-1 text-2xs text-foreground/80">
      <span className="font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </span>
  );
}
