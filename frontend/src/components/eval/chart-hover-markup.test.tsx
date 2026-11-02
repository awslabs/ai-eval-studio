// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Hover markup in the charts that compose it around project text.
 *
 * The `Plot` wrapper escapes caller-supplied trace VALUES and leaves author
 * templates alone. That split is only safe if each chart puts text on the right
 * side of it, which the wrapper's own tests cannot see — they never render a
 * chart. Two charts got it wrong in opposite directions: one put its intentional
 * `<br>`/`<b>` into `customdata`, so the wrapper escaped the tags and the
 * tooltip printed them; the other interpolated a model label into a template,
 * which the wrapper never touches.
 *
 * `Plot` is mocked to capture what each chart hands it, and the real
 * `escapeTraceMarkup` is then applied to reproduce what Plotly receives.
 */
import { cleanup, render } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { escapeTraceMarkup } from "@/lib/plotly-escape";
import type { CompactModelRun } from "@/lib/types";

const { plotted } = vi.hoisted(() => ({ plotted: [] as unknown[][] }));

vi.mock("@/components/ui/plot", () => ({
  default: ({ data }: { data: unknown[] }) => {
    plotted.push(data);
    return null;
  },
  PLOT_CONFIG: {},
}));
vi.mock("@/lib/theme-provider", () => ({ useTheme: () => ({ mode: "light" }) }));
vi.mock("@/components/ui/chart-card", () => ({
  ChartCard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

import { LatencyVsTokensScatter } from "./charts/LatencyVsTokensScatter";
import { TrendCharts } from "./compare/TrendCharts";

type Trace = Record<string, unknown>;

/** The first captured trace with this name, as the chart built it. */
function traceNamed(name: string): Trace {
  const trace = plotted.flat().find((t) => (t as Trace).name === name);
  if (!trace) throw new Error(`no trace named ${name} was plotted`);
  return trace as Trace;
}

beforeEach(() => {
  plotted.length = 0;
});

afterEach(() => {
  cleanup();
});

describe("the compare panel's metric trend hover", () => {
  it("renders its markup and escapes the project text inside it", () => {
    const metric = "<i>acc</i>";
    const runFile = "<b>run-a</b>.json";
    const run: CompactModelRun = {
      test_run_id: "r",
      timestamp: "2026-01-01T00:00:00Z",
      hyperparameters: { model: "m1" } as CompactModelRun["hyperparameters"],
      results: [
        { success: true, latency_ms: 10, metrics: [{ name: metric, success: true }] },
      ] as unknown as CompactModelRun["results"],
    };

    render(
      <TrendCharts
        orderedFiles={[runFile, "run-b.json"]}
        compactRuns={{ [runFile]: [run], "run-b.json": [run] }}
        runSummaryByFile={{}}
        allModels={["m1"]}
      />,
    );

    // What Plotly receives: the chart's trace after the wrapper's escaping.
    const sent = escapeTraceMarkup(traceNamed(metric)) as {
      hovertemplate: string[];
      customdata?: unknown;
    };
    expect(sent.customdata).toBeUndefined();

    const [firstPoint] = sent.hovertemplate;
    // The intentional markup survives the wrapper, so the tooltip still breaks
    // lines and bolds the average rather than printing the tags...
    expect(firstPoint).toContain("<br>");
    expect(firstPoint).toContain("<b>");
    // ...and the project text between those tags reaches Plotly escaped.
    expect(firstPoint).toContain("&lt;b&gt;run-a&lt;/b&gt;.json");
    expect(firstPoint).toContain("&lt;i&gt;acc&lt;/i&gt;");
    expect(firstPoint).not.toContain("<i>acc</i>");
  });
});

describe("the latency-vs-tokens scatter hover", () => {
  it("takes the model label from the escaped trace name, not the template", () => {
    const model = "<b>model</b>";

    render(<LatencyVsTokensScatter cases={[]} modelIds={[model]} modelCasesMap={{}} />);

    const sent = escapeTraceMarkup(traceNamed(model)) as {
      hovertemplate: string;
      name: string;
    };
    // The template stays a constant — the label is not pasted into it.
    expect(sent.hovertemplate).not.toContain(model);
    expect(sent.hovertemplate).toContain("%{fullData.name}");
    // The label reaches the hover through the name, which the wrapper escapes.
    expect(sent.name).toBe("&lt;b&gt;model&lt;/b&gt;");
  });
});
