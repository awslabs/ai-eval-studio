// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Plotly wrapper for Vite/ESM compatibility and theme awareness.
 *
 * Uses the react-plotly.js factory function with plotly.js-dist-min.
 * Exports a useThemeLayout hook for dark/light-aware chart styling,
 * a standard PLOT_CONFIG with a custom fullscreen modebar button that
 * integrates with ChartCard's maximize functionality.
 */

// @ts-expect-error — ESM interop: factory module exports { default: fn }
import factoryModule from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "@/lib/theme-provider";
import { useChartMaximized } from "@/components/ui/chart-card";
import { escapeLayoutMarkup, escapeTraceMarkup } from "@/lib/plotly-escape";

/** Resolve the factory function from the ESM module wrapper. */
const createPlotlyComponent =
  typeof factoryModule === "function" ? factoryModule : factoryModule.default;

/** Raw Plotly component from factory. */
const RawPlot = createPlotlyComponent(Plotly);

/**
 * The wrapper's props: react-plotly.js's own, with the figure typed.
 *
 * The factory import is untyped, so without these two fields every chart's
 * figure would be ``any``. Typed, the compiler rejects a figure Plotly would
 * silently ignore — an axis ``title`` given as a bare string, for one, which
 * Plotly no longer renders.
 */
type PlotProps = Omit<React.ComponentProps<typeof RawPlot>, "data" | "layout"> & {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
  fill?: boolean;
};

/**
 * Plotly's switch for keeping what the user did to a chart — a box zoom, a
 * pan, the drag mode picked in the modebar — when the chart re-plots.
 *
 * Plotly records a zoom by writing the new range into the axis object it was
 * handed. A chart that builds its axes inline (``xaxis: { ...layout.xaxis,
 * title }``) hands it a fresh axis on every render, one that carries no range,
 * so without this any parent re-render would snap the chart back to its full
 * view. With a ``uirevision`` that stays the same, Plotly keeps each
 * interaction whose attribute the new layout did not change, as long as every
 * re-plot gets a layout Plotly has not written into (see ``figure``).
 *
 * A chart that wants its zoom to reset when its data changes can pass its own
 * ``uirevision``, keyed on that data; it takes precedence over this default.
 */
const KEEP_USER_INTERACTION = "keep-user-interaction";

/**
 * Plotly React component with true responsive resize.
 * Uses ResizeObserver on a wrapper div (not window resize) so charts
 * respond to layout changes like sidebar toggle, not just window resize.
 */
function Plot({ style, layout, data, onLegendClick, fill, ...rest }: PlotProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<any>(null);
  const maximized = useChartMaximized();
  // ``fill`` opts a non-fullscreen chart into the same autosize behavior used
  // in fullscreen: drop the fixed layout height and let the chart grow to its
  // container (e.g. a flex-stretched Overview chart row).
  const expand = maximized || !!fill;

  // Controlled legend visibility — Plotly's internal toggle (``visible:
  // "legendonly"``) lives on the graph div and gets thrown away whenever
  // react-plotly.js re-plots (parent rerender, tab switch, theme change).
  // Track which trace names the user has hidden in React state so the
  // intent survives re-plots, then project that state back onto every
  // trace's ``visible`` prop before handing data to Plotly.
  const [hidden, setHidden] = useState<Record<string, boolean>>({});

  // Escape before the legend projection below, not after: ``hidden`` is keyed by
  // the trace name, and Plotly reports the name it was GIVEN when a legend entry
  // is clicked. Escaping afterwards would key the state on one spelling and look
  // it up under another, which silently breaks legend toggling.
  const safeData = useMemo(() => {
    if (!Array.isArray(data)) return data;
    return (data as unknown[]).map(escapeTraceMarkup);
  }, [data]);

  // Drop names from ``hidden`` that aren't in the current data anymore —
  // otherwise a rename or removed trace would leave a tombstone that
  // re-hides a future trace if its name happens to match.
  //
  // Reads the ESCAPED names, the spelling ``hidden`` is keyed by. Reading the
  // raw ones dropped the entry for any name containing ``&``, ``<`` or ``>`` on
  // every data change, un-hiding a trace the user had just hidden.
  useEffect(() => {
    const present = new Set<string>();
    if (Array.isArray(safeData)) {
      for (const trace of safeData as Array<{ name?: unknown }>) {
        if (typeof trace?.name === "string") present.add(trace.name);
      }
    }
    setHidden((prev) => {
      let changed = false;
      const next: Record<string, boolean> = {};
      for (const key of Object.keys(prev)) {
        if (present.has(key)) next[key] = prev[key];
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [safeData]);

  const handleLegendClick = useCallback(
    (event: Readonly<Plotly.LegendClickEvent>): boolean => {
      const trace = (event.data as Array<{ name?: unknown }>)[event.curveNumber];
      const name = typeof trace?.name === "string" ? trace.name : null;
      if (name) {
        setHidden((prev) => ({ ...prev, [name]: !prev[name] }));
      }
      // Caller can extend the behavior; their return value still wins.
      const userResult = onLegendClick?.(event);
      // Returning ``false`` tells Plotly not to apply its built-in toggle
      // so the controlled ``visible`` prop stays the source of truth.
      return userResult ?? false;
    },
    [onLegendClick],
  );

  const visibleData = useMemo(() => {
    if (!Array.isArray(safeData)) return safeData;
    if (Object.keys(hidden).length === 0) return safeData;
    return (safeData as Array<{ name?: unknown; visible?: unknown }>).map((trace) => {
      const name = typeof trace?.name === "string" ? trace.name : null;
      if (!name || !hidden[name]) return trace;
      return { ...trace, visible: "legendonly" };
    });
  }, [safeData, hidden]);

  const onInitialized = useCallback((_: unknown, graphDiv: any) => {
    plotRef.current = graphDiv;
  }, []);

  // ResizeObserver on wrapper div — triggers Plotly.Plots.resize
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const observer = new ResizeObserver(() => {
      // ``querySelector`` types this as Element; Plots.resize wants Plotly's
      // Root, which is the graph div it attached ``.js-plotly-plot`` to. The
      // class is only present on a div Plotly itself initialised.
      const gd = wrapper.querySelector<HTMLElement>(".js-plotly-plot");
      if (gd) Plotly.Plots.resize(gd);
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  // When maximized, the layout MUST NOT carry a fixed ``height``: an explicit
  // height beats ``autosize`` in Plotly, so ``Plots.resize`` (fired by the
  // ResizeObserver after the fullscreen transition / a DPR change on
  // multi-monitor) would re-pin the chart to its small fixed height. Dropping
  // ``height`` lets ``autosize`` fill the wrapper, which the fullscreen CSS
  // (``.chart-card-fullscreen``) stretches to 100%.
  //
  // A caller whose layout and data keep their identity hands react-plotly.js
  // nothing new, and no re-plot happens. A caller that builds its layout inline
  // still re-plots on every render; ``KEEP_USER_INTERACTION`` is what keeps the
  // user's zoom through it.
  //
  // Every re-plot gets its own deep copy of the layout, because Plotly writes the
  // user's zoom into the layout object it is handed, and on the next re-plot it
  // reads whatever that object holds as the chart's own choice. Handed the same
  // object again, it adopts the zoom as the chart's range and forgets it was the
  // user's, so the next layout the chart builds snaps back to the full view. A
  // legend click did exactly that: it re-plots with new data and an unchanged
  // layout, and the poll re-render after it reset the zoom. A re-plot comes from
  // new data or a new layout, so the copy is made with the figure it is
  // plotted with.
  //
  // Layout text is escaped like trace values are: a title can name the metrics
  // a chart plots, and those names are project-controlled.
  const safeLayout = useMemo(() => escapeLayoutMarkup(layout), [layout]) as typeof layout;
  const figure = useMemo(() => {
    const layoutCopy = structuredClone(safeLayout);
    if (expand) delete layoutCopy.height;
    return {
      data: visibleData,
      layout: { uirevision: KEEP_USER_INTERACTION, ...layoutCopy, autosize: true },
    };
  }, [visibleData, safeLayout, expand]);
  const wrapperHeight = expand ? "100%" : (safeLayout.height ?? 220);

  return (
    <div ref={wrapperRef} className="w-full" style={{ height: wrapperHeight, ...style }}>
      <RawPlot
        {...rest}
        data={figure.data}
        layout={figure.layout}
        style={{ width: "100%", height: "100%" }}
        onInitialized={onInitialized}
        onLegendClick={handleLegendClick}
      />
    </div>
  );
}

export default Plot;

/**
 * Returns a Plotly layout object that adapts to the active theme.
 *
 * Plotly needs concrete color strings (it can't consume Tailwind tokens), so
 * we resolve the active theme's CSS variables at render. This tracks ALL
 * themes (not just light/dark): font/grid/hover colors follow the same
 * ``--muted-foreground`` / ``--border`` / ``--popover`` vars the rest of the
 * UI uses. Recomputes when the theme id or mode changes (the provider applies
 * the new vars before this memo re-runs).
 */
export function useThemeLayout(): Partial<Plotly.Layout> {
  const { themeId, mode, resolveVar } = useTheme();

  return useMemo(
    () => {
      const fontColor = resolveVar("muted-foreground");
      const border = resolveVar("border");
      // Translucent gridlines from the theme's border color; color-mix keeps
      // the subtle look without a separate token. zerolines a touch stronger.
      const grid = `color-mix(in oklch, ${border} 60%, transparent)`;
      const zero = `color-mix(in oklch, ${border} 90%, transparent)`;
      return {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: fontColor, size: 11 },
        xaxis: { gridcolor: grid, zerolinecolor: zero },
        yaxis: { gridcolor: grid, zerolinecolor: zero },
        legend: { font: { size: 10 } },
        hoverlabel: {
          bgcolor: resolveVar("popover"),
          font: { color: resolveVar("popover-foreground"), size: 12 },
          bordercolor: border,
        },
      };
    },
    // themeId + mode drive the resolved CSS vars; resolveVar is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [themeId, mode],
  );
}

/**
 * Custom modebar button that triggers ChartCard fullscreen.
 * Walks up the DOM from the Plotly graph div to find the nearest
 * `.chart-card-container` element and calls its exposed maximize function.
 */
const fullscreenButton: Partial<Plotly.ModeBarButton> = {
  name: "fullscreen",
  title: "Fullscreen",
  icon: {
    width: 875,
    height: 1e3,
    /* Bold expand icon with thicker 90-unit strokes for visual prominence.
       Smaller viewbox (875) makes it render slightly larger in the modebar. */
    path: "M0 812.5v-275h90v185h185v90H0zm875 0H600v-90h185v-185h90v275zM0 187.5V462.5h90V277.5h185v-90H0zm875 0v275h-90V277.5H600v-90h275z",
  },
  click: (gd: any) => {
    const container = gd.closest?.(".chart-card-container");
    if (container?.__chartCardMaximize) {
      container.__chartCardMaximize();
    }
  },
};

/** Standard Plotly config for all charts — zoom, pan, hover toolbar, PNG download, fullscreen. */
export const PLOT_CONFIG: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: "hover",
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  modeBarButtonsToAdd: [fullscreenButton as any],
  displaylogo: false,
  toImageButtonOptions: { format: "png", height: 600, width: 1000, scale: 2 },
};
