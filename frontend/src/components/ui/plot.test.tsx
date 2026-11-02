// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Tests for the `Plot` wrapper's controlled legend state.
 *
 * `plotly.js-dist-min` needs browser APIs jsdom does not have, so it is mocked
 * along with the react-plotly factory. The factory returns a stand-in that
 * records the props the wrapper hands Plotly — which is the wrapper's whole
 * contract, and what these assert on.
 */
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type RenderedProps = {
  data: Array<Record<string, unknown>>;
  layout: Record<string, unknown>;
  onLegendClick: (event: { data: unknown; curveNumber: number }) => boolean;
};

const { rendered } = vi.hoisted(() => ({ rendered: [] as RenderedProps[] }));

vi.mock("plotly.js-dist-min", () => ({ default: { Plots: { resize: () => {} } } }));
vi.mock("react-plotly.js/factory", () => ({
  default: () => (props: RenderedProps) => {
    rendered.push(props);
    return null;
  },
}));
vi.mock("@/lib/theme-provider", () => ({
  useTheme: () => ({ themeId: "test", mode: "light", resolveVar: () => "#000" }),
}));
vi.mock("@/components/ui/chart-card", () => ({ useChartMaximized: () => false }));

import Plot from "./plot";

/** The props of the most recent render. */
function lastRender(): RenderedProps {
  const props = rendered[rendered.length - 1];
  if (!props) throw new Error("Plot never rendered");
  return props;
}

beforeEach(() => {
  rendered.length = 0;
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

describe("what the wrapper hands Plotly", () => {
  it("escapes layout text as well as trace values", () => {
    // A chart title can name the metrics it plots, and metric names are
    // project-controlled; the layout never passes through trace escaping.
    render(
      <Plot
        data={[{ name: "<i>m</i>", y: [1] }]}
        layout={{ title: { text: "<b>acc</b> vs rel" }, height: 300 }}
      />,
    );

    expect(lastRender().data[0].name).toBe("&lt;i&gt;m&lt;/i&gt;");
    expect((lastRender().layout.title as { text: string }).text).toBe(
      "&lt;b&gt;acc&lt;/b&gt; vs rel",
    );
  });
});

describe("controlled legend visibility", () => {
  it("keeps a hidden trace hidden across a data change when its name needs escaping", () => {
    // ``hidden`` is keyed by the ESCAPED name, because a legend click reports the
    // escaped trace Plotly was given. The cleanup that forgets names no longer
    // plotted has to compare in that same spelling; comparing raw names forgot
    // any name holding ``&``, ``<`` or ``>`` on the next data change, and the
    // trace the user had just hidden came back.
    const { rerender } = render(<Plot data={[{ name: "a & b", y: [1] }]} layout={{}} />);

    act(() => {
      lastRender().onLegendClick({ data: lastRender().data, curveNumber: 0 });
    });
    expect(lastRender().data[0].visible).toBe("legendonly");

    // Same trace, fresh array identity — what a parent re-render with new data does.
    rerender(<Plot data={[{ name: "a & b", y: [2] }]} layout={{}} />);

    expect(lastRender().data[0].name).toBe("a &amp; b");
    expect(lastRender().data[0].visible).toBe("legendonly");
  });
});
