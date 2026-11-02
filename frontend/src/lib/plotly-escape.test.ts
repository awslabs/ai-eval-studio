// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the markup escaping applied to Plotly trace values.
 *
 * Plotly interprets a subset of HTML in axis labels, hover text and legend
 * entries, and substitutes data values into a `hovertemplate` without escaping
 * them. The values there are project-controlled — run labels, model ids, metric
 * names, categories — so markup in one of them renders as markup. Runtime
 * testing showed `<b>` and `<span style>` take effect while script tags, image
 * event handlers and `javascript:` URLs are neutralised by Plotly, which makes
 * this content spoofing rather than script execution.
 *
 * The escaping lives in the `Plot` wrapper so it covers every chart including the
 * next one. These tests pin the two halves of that decision: caller-supplied
 * VALUES are escaped, author-written TEMPLATES are not.
 */
import { describe, expect, it } from "vitest";

import { escapeLayoutMarkup, escapeTraceMarkup } from "./plotly-escape";

describe("escaping caller-supplied trace values", () => {
  it("escapes markup in a category label", () => {
    const trace = escapeTraceMarkup({
      x: ["ordinary", "<b>bold</b>", '<span style="color:red">red</span>'],
      y: [1, 2, 3],
    }) as { x: string[]; y: number[] };

    expect(trace.x[0]).toBe("ordinary");
    expect(trace.x[1]).toBe("&lt;b&gt;bold&lt;/b&gt;");
    expect(trace.x[2]).toContain("&lt;span");
    // Numbers are untouched, so a numeric axis still plots as numbers.
    expect(trace.y).toEqual([1, 2, 3]);
  });

  it("escapes a trace name, which is the legend entry", () => {
    const trace = escapeTraceMarkup({ name: "<b>model</b>" }) as { name: string };
    expect(trace.name).toBe("&lt;b&gt;model&lt;/b&gt;");
  });

  it("escapes hover text and nested customdata", () => {
    const trace = escapeTraceMarkup({
      hovertext: ["<i>x</i>"],
      customdata: [["<b>a</b>", 2], ["b", 3]],
    }) as { hovertext: string[]; customdata: unknown[][] };

    expect(trace.hovertext[0]).toBe("&lt;i&gt;x&lt;/i&gt;");
    expect(trace.customdata[0][0]).toBe("&lt;b&gt;a&lt;/b&gt;");
    expect(trace.customdata[0][1]).toBe(2);
  });

  it("escapes an ampersand before the angle brackets", () => {
    // Ordering matters: escaping `<` first and `&` second would double-encode
    // the entity into `&amp;lt;` and print the text instead of rendering it.
    const trace = escapeTraceMarkup({ text: ["a & b < c"] }) as { text: string[] };
    expect(trace.text[0]).toBe("a &amp; b &lt; c");
  });
});

describe("leaving author-written templates alone", () => {
  it("does not touch hovertemplate, whose <br> is intentional", () => {
    const template = "%{y}<br>%{x}: %{z:.1f}%<extra></extra>";
    const trace = escapeTraceMarkup({ hovertemplate: template }) as {
      hovertemplate: string;
    };
    expect(trace.hovertemplate).toBe(template);
  });

  it("returns the original object when nothing needed escaping", () => {
    // Identity is asserted because a fresh object on every render makes
    // react-plotly.js re-plot, which is a real performance regression on a
    // dashboard of fifteen charts.
    const trace = { x: [1, 2], y: [3, 4], hovertemplate: "%{y}<br>" };
    expect(escapeTraceMarkup(trace)).toBe(trace);
  });

  it("passes through values that are not objects", () => {
    expect(escapeTraceMarkup(null)).toBeNull();
    expect(escapeTraceMarkup("string")).toBe("string");
  });
});

describe("escaping layout text", () => {
  // Trace escaping never reaches the layout, and a layout is not all literals:
  // the score-correlation chart titles itself with the two metric names it
  // plots, and metric names are project-controlled.
  it("escapes the chart title, axis titles and tick labels", () => {
    const layout = escapeLayoutMarkup({
      title: { text: "<b>acc</b> vs rel", font: { size: 13 } },
      xaxis: { title: { text: "<i>acc</i>" }, ticktext: ["<b>a</b>", "b"], range: [0, 1] },
      yaxis2: { title: { text: "rel" } },
    }) as {
      title: { text: string; font: { size: number } };
      xaxis: { title: { text: string }; ticktext: string[]; range: number[] };
      yaxis2: { title: { text: string } };
    };

    expect(layout.title.text).toBe("&lt;b&gt;acc&lt;/b&gt; vs rel");
    expect(layout.title.font).toEqual({ size: 13 });
    expect(layout.xaxis.title.text).toBe("&lt;i&gt;acc&lt;/i&gt;");
    expect(layout.xaxis.ticktext).toEqual(["&lt;b&gt;a&lt;/b&gt;", "b"]);
    expect(layout.xaxis.range).toEqual([0, 1]);
    expect(layout.yaxis2.title.text).toBe("rel");
  });

  it("escapes the legend title and annotation text", () => {
    const layout = escapeLayoutMarkup({
      legend: { title: { text: "<b>models</b>" }, font: { size: 10 } },
      annotations: [{ text: "<span>note</span>", x: 1 }, { text: "P95: 120ms" }],
    }) as {
      legend: { title: { text: string }; font: { size: number } };
      annotations: Array<{ text: string; x?: number }>;
    };

    expect(layout.legend.title.text).toBe("&lt;b&gt;models&lt;/b&gt;");
    expect(layout.legend.font).toEqual({ size: 10 });
    expect(layout.annotations[0]).toEqual({ text: "&lt;span&gt;note&lt;/span&gt;", x: 1 });
    expect(layout.annotations[1].text).toBe("P95: 120ms");
  });

  it("returns the original layout when nothing needed escaping", () => {
    // Same reason as traces: a fresh layout object every render makes
    // react-plotly.js re-plot the chart.
    const layout = {
      title: { text: "Latency Distribution" },
      xaxis: { gridcolor: "#ccc" },
      annotations: [{ text: "P95: 120ms" }],
      height: 280,
    };
    expect(escapeLayoutMarkup(layout)).toBe(layout);
    expect(escapeLayoutMarkup(undefined)).toBeUndefined();
  });
});
