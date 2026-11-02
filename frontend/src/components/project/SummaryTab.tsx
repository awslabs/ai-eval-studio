// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Summary tab for the project page.
 * Compact overview: inline stats, harness config, and eval highlights
 * (latest + best run per model). Avoids redundancy with the page header.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Database, Upload, Trophy, Clock, ArrowRight } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { fmtPct1, shortModel } from "@/lib/format-utils";
import { useHarnessConfig } from "@/hooks/useHarnessConfig";
import { useEvalRuns } from "@/hooks/useEvalRuns";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import type { ProjectConfig, EvalRunSummary } from "@/lib/types";
import { newestRun } from "@/lib/run-list";

/** Props for the SummaryTab component. */
interface SummaryTabProps {
  /** Project UUID for API calls */
  projectId: string;
  /** Full project configuration from the parent */
  project: ProjectConfig;
}

/** Highest single-model pass rate in a run.
 *
 * Best of the bunch, not the average: the average gets dragged down by
 * weak comparison models — see the coordinator prompt's note that the
 * aggregate can be lower than any single model. This is a pass-rate
 * ranking only. The Reports page summarizes the trophy-winning run and
 * recommends the cheapest model that clears every target, falling back
 * to the best pass rate (``_pick_recommended_model``), so it can name a
 * different run or model than the "Best" badge here.
 */
function bestSingleModelRate(run: EvalRunSummary): number {
  if (run.model_runs.length === 0) return 0;
  return Math.max(
    ...run.model_runs.map((m) => (m.total > 0 ? m.passed / m.total : 0)),
  );
}

/** Find the best run per model across all eval runs.
 *
 * Keyed by the canonical full model id (``ModelRunSummary.model``);
 * display labels go through ``shortModel`` at render time. */
function findBestPerModel(runs: EvalRunSummary[]): Map<string, { rate: number; passed: number; total: number; prompt: string; timestamp: string }> {
  const best = new Map<string, { rate: number; passed: number; total: number; prompt: string; timestamp: string }>();
  for (const run of runs) {
    for (const m of run.model_runs) {
      const rate = m.total > 0 ? m.passed / m.total : 0;
      const prev = best.get(m.model);
      if (!prev || rate > prev.rate) {
        best.set(m.model, { rate, passed: m.passed, total: m.total, prompt: run.prompt_version, timestamp: run.timestamp });
      }
    }
  }
  return best;
}

/** Semantic color for a pass rate. */
function rateColor(rate: number): string {
  return rate >= 0.8 ? "text-success" : rate >= 0.5 ? "text-warning" : "text-error";
}

/**
 * Project overview tab with compact stats, harness config, and eval highlights.
 */
export function SummaryTab({ projectId, project }: SummaryTabProps) {
  const [, setSearchParams] = useSearchParams();
  const [promptFiles, setPromptFiles] = useState<string[]>([]);
  const [dataFileCount, setDataFileCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  /** Eval runs fetched via hook — the hook subscribes to job/completed
   *  itself, so consumers don't manage refresh wiring. */
  const { runs: evalRunsRaw } = useEvalRuns(projectId);
  // Memoized for identity stability: ``?? []`` yields a fresh array each
  // render while runs are loading, which made both derived useMemos below
  // recompute on every render.
  const evalRuns = useMemo(() => evalRunsRaw ?? [], [evalRunsRaw]);

  /** Harness config fetched via hook — same self-refresh discipline. */
  const { config: harnessConfig } = useHarnessConfig(projectId);

  /** Fetch remaining summary data (prompts, file tree). */
  const fetchData = useCallback((showLoading: boolean) => {
    if (showLoading) setIsLoading(true);
    Promise.all([
      api.listPrompts(projectId).catch(() => [] as string[]),
      api.listFiles(projectId).catch(() => ({ tree: [] })),
    ]).then(([prompts, files]) => {
      setPromptFiles(prompts);
      const dataDir = files.tree.find((n) => n.name === "data");
      const rawDir = dataDir?.children?.find((n) => n.name === "raw");
      setDataFileCount(rawDir?.children?.filter((n) => n.type === "file").length ?? 0);
      if (showLoading) setIsLoading(false);
    });
  }, [projectId]);

  useEffect(() => { fetchData(true); }, [fetchData]);

  /* Eval runs + harness config refresh themselves via their hooks now —
   * every consumer inherits WS-driven freshness from one bus. The only
   * surface left here is the prompt list + file-tree count, which is
   * fetched directly. ``filesystem/changed`` is the right signal for
   * those (uploads, prompt saves), so wire just that one event. */
  useProjectEvent({ resource: "filesystem", type: "changed" }, () => fetchData(false), "state");

  const latestRun = newestRun(evalRuns);
  const caseCount = latestRun?.model_runs[0]?.total ?? 0;
  const bestPerModel = useMemo(() => findBestPerModel(evalRuns), [evalRuns]);
  const bestRun = useMemo(() => {
    if (evalRuns.length === 0) return null;
    // Best-of-the-bunch by single-model pass rate (see bestSingleModelRate
    // for how this can differ from the Reports page's pick).
    return evalRuns.reduce((best, run) =>
      bestSingleModelRate(run) > bestSingleModelRate(best) ? run : best
    );
  }, [evalRuns]);
  /** Whether best and latest are the same run file. */
  const bestIsLatest = bestRun?._file === latestRun?._file;

  /** Navigate to a specific tab. */
  const goToTab = (tab: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("tab", tab);
      return next;
    }, { replace: true });
  };

  // Welcome state when no data has been uploaded yet
  if (!isLoading && dataFileCount === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4 text-center space-y-6">
        <div className="size-16 rounded-full bg-muted flex items-center justify-center">
          <Database className="size-7 text-muted-foreground" />
        </div>
        <div className="space-y-2 max-w-md">
          <h2 className="text-lg font-semibold">Welcome to {project.name}</h2>
          <p className="text-sm text-muted-foreground">
            To get started, upload your evaluation dataset and then describe your goals in the chat.
            The assistant will guide you through data analysis, metric selection, harness creation, and prompt optimization.
          </p>
        </div>
        <Button variant="default" size="sm" onClick={() => goToTab("data")}>
          <Upload className="size-3.5 mr-1.5" />
          Upload Data
        </Button>
        <div className="text-xs text-muted-foreground max-w-sm space-y-1">
          <p className="font-medium">How it works:</p>
          <ol className="list-decimal list-inside space-y-0.5 text-left">
            <li>Upload a CSV, JSON, or Excel file with your test cases</li>
            <li>Chat with the assistant to configure models and metrics</li>
            <li>The system creates an eval harness and baseline prompt</li>
            <li>Run optimization to automatically improve your prompt</li>
          </ol>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* ── Inline stats bar ── */}
      <div className="flex items-center gap-6 text-xs text-muted-foreground">
        <span>
          <span className="text-foreground font-semibold tabular-nums">{isLoading ? "..." : dataFileCount}</span> data file{dataFileCount !== 1 ? "s" : ""}
        </span>
        <span className="text-border">|</span>
        <span>
          <span className="text-foreground font-semibold tabular-nums">{isLoading ? "..." : promptFiles.length}</span> prompt version{promptFiles.length !== 1 ? "s" : ""}
        </span>
        <span className="text-border">|</span>
        <span>
          <span className="text-foreground font-semibold tabular-nums">{isLoading ? "..." : evalRuns.length}</span> eval run{evalRuns.length !== 1 ? "s" : ""}
        </span>
        {caseCount > 0 && (
          <>
            <span className="text-border">|</span>
            <span>
              <span className="text-foreground font-semibold tabular-nums">{caseCount}</span> test cases
            </span>
          </>
        )}
      </div>

      {/* ── Two-column layout: Config + Eval Highlights ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Left: Harness Config */}
        {!isLoading && harnessConfig && (
          <Card className="p-4 space-y-3">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Configuration</h3>

            {/* Models — display via ``shortModel`` to match the rest of
                the app; the raw id is still the React key, so duplicates
                across providers can't collide. */}
            {harnessConfig.models.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {harnessConfig.models.map((model) => (
                  <Badge key={model} variant="secondary" className="text-2xs font-mono">
                    {shortModel(model)}
                  </Badge>
                ))}
              </div>
            )}

            {/* Key-value grid for config fields */}
            <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
              {harnessConfig.data_file && (
                <>
                  <span className="text-muted-foreground">Data</span>
                  <span className="font-mono truncate">{harnessConfig.data_file}</span>
                </>
              )}
              {harnessConfig.judge_model_id && (
                <>
                  <span className="text-muted-foreground">Judge</span>
                  <span className="font-mono truncate">{shortModel(harnessConfig.judge_model_id)}</span>
                </>
              )}
              {harnessConfig.prompt_version && (
                <>
                  <span className="text-muted-foreground">Prompt</span>
                  <span>{harnessConfig.prompt_version}</span>
                </>
              )}
            </div>

            {/* Field mappings */}
            {harnessConfig.field_mappings && (
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1.5">Fields</p>
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                  {Object.entries(harnessConfig.field_mappings).map(([role, column]) => (
                    <span key={role}>
                      <span className="text-muted-foreground">{role.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase())}: </span>
                      <span className="font-mono">{column}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Optimization targets */}
            {harnessConfig.target_metric_scores && (
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1.5">Targets</p>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(harnessConfig.target_metric_scores).map(([metric, target]) => (
                    <Badge key={metric} variant="outline" className="text-2xs">
                      {metric} &ge; {Math.round(target * 100)}%
                    </Badge>
                  ))}
                  {harnessConfig.max_iterations != null && (
                    <Badge variant="outline" className="text-2xs text-muted-foreground">
                      {harnessConfig.max_iterations} iter max
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </Card>
        )}

        {/* Right: Eval highlights */}
        {!isLoading && evalRuns.length > 0 && (
          <div className="space-y-4">
            {/* Best per model — the hero card */}
            {bestPerModel.size > 0 && (
              <Card className="p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Trophy className="size-3.5 text-warning" />
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Best Results</h3>
                </div>
                <div className="space-y-2.5">
                  {Array.from(bestPerModel.entries()).map(([modelId, b]) => (
                    <div key={modelId} className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-xs text-muted-foreground truncate">{shortModel(modelId)}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className={`text-sm font-semibold tabular-nums ${rateColor(b.rate)}`}>
                          {fmtPct1(b.rate)}%
                        </span>
                        <span className="text-2xs text-muted-foreground tabular-nums">
                          {b.passed}/{b.total}
                        </span>
                        <Badge variant="outline" className="text-2xs text-muted-foreground">
                          {b.prompt}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Latest run */}
            {latestRun && (
              <Card className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Clock className="size-3.5 text-muted-foreground" />
                    <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Latest Run</h3>
                    {bestIsLatest && (
                      <Badge variant="outline" className="text-2xs text-success border-success/30">Best</Badge>
                    )}
                  </div>
                  <span className="text-2xs text-muted-foreground">
                    {new Date(latestRun.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <Badge variant="secondary" className="text-2xs">{latestRun.prompt_version}</Badge>
                  <span className="text-muted-foreground">{caseCount} cases</span>
                </div>
                <div className="space-y-1.5">
                  {latestRun.model_runs.map((m) => {
                    const rate = m.total > 0 ? m.passed / m.total : 0;
                    return (
                      <div key={m.model} className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground truncate">{shortModel(m.model)}</span>
                        <div className="flex items-center gap-3 shrink-0">
                          <span className={`font-medium tabular-nums ${rateColor(rate)}`}>
                            {fmtPct1(rate)}%
                          </span>
                          <span className="text-muted-foreground/60 tabular-nums">
                            {m.passed}/{m.total}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <button
                  onClick={() => goToTab("evaluations")}
                  className="flex items-center gap-1 rounded text-2xs text-muted-foreground hover:text-foreground transition-colors pt-1 outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  View all runs <ArrowRight className="size-3" />
                </button>
              </Card>
            )}
          </div>
        )}
      </div>

      {/* Empty state */}
      {!isLoading && evalRuns.length === 0 && !harnessConfig && (
        <Card className="p-8 text-center">
          <p className="text-sm text-muted-foreground">
            No evaluations yet. Use the chat to configure your eval harness and get started.
          </p>
        </Card>
      )}
    </div>
  );
}
