// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * What the user sees after interacting with a chart, driven through the real
 * plotly.js rather than a mock.
 *
 * Both behaviours under test happen inside Plotly, so a mocked Plotly cannot
 * see either one:
 *
 * - A box zoom is stored by Plotly writing the new range into the axis object it
 *   was handed. A chart that builds its axes inline hands it a fresh axis on the
 *   next render, and without `uirevision` the chart snaps back to its full view.
 *   The dashboard re-renders on every optimization-state poll that returns a
 *   record, every few seconds, so such a zoom lasts only until the next poll.
 * - Plotly no longer renders an axis title given as a bare string; the title
 *   silently falls back to a placeholder that is never drawn.
 *
 * jsdom has no canvas, so its unimplemented `getContext` is replaced before
 * Plotly loads, which is when Plotly first calls it; the charts here draw in SVG.
 */
import { act, cleanup, render, waitFor } from "@testing-library/react";
import Plotly from "plotly.js-dist-min";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Plot from "@/components/ui/plot";
import type { TestResult } from "@/lib/types";

vi.hoisted(() => {
  HTMLCanvasElement.prototype.getContext = () => null;
});

vi.mock("@/lib/theme-provider", () => ({
  useTheme: () => ({ themeId: "test", mode: "light", resolveVar: () => "#000" }),
}));

import { LatencyHistogram } from "./charts/LatencyHistogram";

/** The graph-div internals these tests read: what Plotly resolved, and its event hook. */
interface GraphDiv extends HTMLElement {
  _fullLayout: {
    _guiEditing: boolean;
    xaxis: { range: [number, number] };
    yaxis: { range: [number, number] };
  };
  /** The traces Plotly was last handed. */
  data: Array<{ visible?: unknown }>;
  once(event: string, handler: () => void): void;
}

/** Wait for react-plotly.js to finish the first plot, and return its graph div. */
function plotted(container: HTMLElement): Promise<GraphDiv> {
  return waitFor(() => {
    const gd = container.querySelector<GraphDiv>(".js-plotly-plot");
    if (!gd?._fullLayout?.xaxis?.range) throw new Error("chart not plotted yet");
    return gd;
  });
}

/**
 * Zoom the way a box drag does.
 *
 * Plotly ends the drag with a GUI relayout of both axis ranges. It does not
 * export that entry point, so this repeats its body — flag the edit as the
 * user's, relayout, clear the flag. The flag is what lets `uirevision` tell a
 * user's zoom from a range the chart itself set.
 */
async function boxZoom(gd: GraphDiv, x: [number, number], y: [number, number]): Promise<void> {
  // Plotly takes attribute-path keys here; its types only model nested objects.
  const ranges = {
    "xaxis.range[0]": x[0],
    "xaxis.range[1]": x[1],
    "yaxis.range[0]": y[0],
    "yaxis.range[1]": y[1],
  } as Partial<Plotly.Layout>;
  gd._fullLayout._guiEditing = true;
  const done = Plotly.relayout(gd, ranges);
  gd._fullLayout._guiEditing = false;
  await done;
}

/** Run `rerender` and resolve once react-plotly.js has re-plotted the chart. */
function replot(gd: GraphDiv, rerender: () => void): Promise<void> {
  const replotted = new Promise<void>((resolve) => gd.once("plotly_react", () => resolve()));
  rerender();
  return replotted;
}

/**
 * Click a legend entry the way a user does, and resolve once the chart re-plots.
 *
 * Plotly listens for the press and release on the entry's own toggle rect. The
 * wrapper hides the trace itself, so the click re-plots through react-plotly.js.
 */
async function clickLegendEntry(gd: GraphDiv, index: number): Promise<void> {
  const toggle = gd.querySelectorAll(".legendtoggle")[index];
  if (!toggle) throw new Error(`no legend entry ${index}`);
  const replotted = new Promise<void>((resolve) => gd.once("plotly_react", () => resolve()));
  await act(async () => {
    toggle.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, buttons: 1 }));
    toggle.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
  });
  await replotted;
}

/** Cases carrying only what the latency histogram reads. */
function latencyCases(): TestResult[] {
  return Array.from({ length: 40 }, (_, i) => ({
    name: `case-${i}`,
    success: true,
    input: "",
    actual_output: "",
    expected_output: "",
    context: null,
    retrieval_context: null,
    additional_metadata: { latency_ms: 200 + ((i * 37) % 900) },
    metrics_data: [],
  }));
}

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe(): void {}
      disconnect(): void {}
    },
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the user's zoom survives a re-plot", () => {
  const data: Plotly.Data[] = [{ type: "scatter", x: [0, 10], y: [0, 10] }];
  /** A fresh layout with fresh axes on every call — what an inline-built chart renders. */
  const inlineLayout = (extra: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> => ({
    xaxis: { title: { text: "x" } },
    yaxis: { title: { text: "y" } },
    height: 200,
    ...extra,
  });

  it("keeps a box zoom when a re-render hands Plotly a fresh layout", async () => {
    const { container, rerender } = render(<Plot data={data} layout={inlineLayout()} />);
    const gd = await plotted(container);

    await boxZoom(gd, [2, 4], [3, 5]);
    await replot(gd, () => rerender(<Plot data={data} layout={inlineLayout()} />));

    expect(gd._fullLayout.xaxis.range).toEqual([2, 4]);
    expect(gd._fullLayout.yaxis.range).toEqual([3, 5]);
  });

  it("keeps a box zoom through a legend click and the re-render after it", async () => {
    // A legend click re-plots with new data and an unchanged layout, the object
    // Plotly wrote the zoom into. Re-handed it, Plotly took the zoom for the
    // chart's own range and dropped its record of the user's edit, so the next
    // re-render, the dashboard's poll, snapped the chart back to its full view.
    const traces: Plotly.Data[] = [
      { type: "scatter", name: "model-a", x: [0, 10], y: [0, 10] },
      { type: "scatter", name: "model-b", x: [0, 10], y: [10, 0] },
    ];
    const withLegend = (): Partial<Plotly.Layout> => inlineLayout({ showlegend: true });
    const { container, rerender } = render(<Plot data={traces} layout={withLegend()} />);
    const gd = await plotted(container);

    await boxZoom(gd, [2, 4], [3, 5]);
    await clickLegendEntry(gd, 0);
    await replot(gd, () => rerender(<Plot data={traces} layout={withLegend()} />));

    expect(gd.data[0].visible).toBe("legendonly");
    expect(gd._fullLayout.xaxis.range).toEqual([2, 4]);
    expect(gd._fullLayout.yaxis.range).toEqual([3, 5]);
  });

  it("still lets a chart reset the zoom by changing its own uirevision", async () => {
    // The wrapper's default must not override a chart's own ``uirevision``,
    // or no chart could ever reset its zoom deliberately (on new data, say).
    const { container, rerender } = render(
      <Plot data={data} layout={inlineLayout({ uirevision: "run-a" })} />,
    );
    const gd = await plotted(container);
    const fullView = [...gd._fullLayout.xaxis.range];

    await boxZoom(gd, [2, 4], [3, 5]);
    await replot(gd, () =>
      rerender(<Plot data={data} layout={inlineLayout({ uirevision: "run-b" })} />),
    );

    expect(gd._fullLayout.xaxis.range).toEqual(fullView);
  });
});

describe("LatencyHistogram", () => {
  it("keeps a box zoom through a parent re-render", async () => {
    const cases = latencyCases();
    const { container, rerender } = render(<LatencyHistogram cases={cases} />);
    const gd = await plotted(container);

    await boxZoom(gd, [400, 600], [0, 2]);
    // A new but equal case list, as a refetch produces: it changes the data's
    // identity, so the chart re-plots however its layout is built.
    await replot(gd, () => rerender(<LatencyHistogram cases={[...cases]} />));

    expect(gd._fullLayout.xaxis.range).toEqual([400, 600]);
    expect(gd._fullLayout.yaxis.range).toEqual([0, 2]);
  });

  it("keeps a box zoom through a legend click, with one histogram per model", async () => {
    const cases = latencyCases();
    const byModel = { "model-a": cases.slice(0, 20), "model-b": cases.slice(20) };
    const histogram = (): React.ReactElement => (
      <LatencyHistogram cases={cases} modelIds={["model-a", "model-b"]} modelCasesMap={byModel} />
    );
    const { container, rerender } = render(histogram());
    const gd = await plotted(container);

    await boxZoom(gd, [400, 600], [0, 2]);
    await clickLegendEntry(gd, 1);
    await replot(gd, () => rerender(histogram()));

    expect(gd.data[1].visible).toBe("legendonly");
    expect(gd._fullLayout.xaxis.range).toEqual([400, 600]);
    expect(gd._fullLayout.yaxis.range).toEqual([0, 2]);
  });

  it("draws its axis titles", async () => {
    const { container } = render(<LatencyHistogram cases={latencyCases()} />);
    const gd = await plotted(container);

    expect(gd.querySelector(".g-xtitle")?.textContent).toBe("Latency (ms)");
    expect(gd.querySelector(".g-ytitle")?.textContent).toBe("Count");
  });
});
