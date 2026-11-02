// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Escape project-controlled text before it reaches a Plotly trace or layout.
 *
 * Separate from the `Plot` wrapper because it depends on nothing: importing the
 * wrapper pulls in `plotly.js-dist-min`, which needs browser globals and so
 * cannot be unit-tested in a node environment.
 */

/**
 * Trace fields whose string VALUES Plotly renders as markup.
 *
 * Plotly interprets a subset of HTML in axis labels, hover text and legend
 * entries, and it substitutes data values into a ``hovertemplate`` without
 * escaping them. Everything listed here carries project-controlled text — run
 * labels, model ids, metric names, categories — so a value containing markup is
 * rendered as markup. Runtime testing showed ``<b>`` and ``<span style>`` take
 * effect; script tags, image event handlers and ``javascript:`` URLs are
 * neutralised by Plotly, so this is a content-spoofing surface rather than a
 * script-execution one.
 *
 * ``hovertemplate`` and ``texttemplate`` are deliberately NOT here. Those are
 * written by chart authors and carry intentional ``<br>``; escaping them would
 * print the tags instead of breaking the line. The distinction that matters is
 * author-written template versus caller-supplied value — so a chart that
 * interpolates project text INTO a template owns escaping that text, with
 * ``escapePlotlyMarkup`` below. Markup placed in one of these VALUE fields is
 * escaped like any other value and prints as text.
 */
const MARKUP_BEARING_TRACE_FIELDS = [
  "x",
  "y",
  "text",
  "hovertext",
  "name",
  "customdata",
] as const;

/**
 * Replace the characters Plotly would read as markup with entities.
 *
 * Exported for the one case the `Plot` wrapper cannot handle for a chart: a
 * template the chart COMPOSES, where project text sits between intentional tags
 * (`<br>`, `<b>`). The wrapper leaves templates alone, so a chart building one
 * escapes each project value with this before interpolating it.
 */
export function escapePlotlyMarkup(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Map an array, returning the SAME array when no item changed.
 *
 * Identity matters here. A version that always returned `value.map(...)`
 * produced a fresh array every render, which makes react-plotly.js re-plot every
 * chart on every parent render — measurable on a dashboard of fifteen charts,
 * and invisible in any test that only checks the escaped output.
 */
function mapSame(value: unknown, fn: (item: unknown) => unknown): unknown {
  if (!Array.isArray(value)) return value;
  let changed = false;
  const next = value.map((item) => {
    const mapped = fn(item);
    if (mapped !== item) changed = true;
    return mapped;
  });
  return changed ? next : value;
}

/**
 * Escape strings anywhere inside a value, leaving non-strings untouched.
 *
 * Returns the IDENTICAL value when nothing needed escaping, arrays included
 * (see `mapSame`).
 */
function escapeDeep(value: unknown): unknown {
  if (typeof value === "string") {
    const escaped = escapePlotlyMarkup(value);
    return escaped === value ? value : escaped;
  }
  return mapSame(value, escapeDeep);
}

/**
 * Apply `fn` to `obj[key]` copy-on-write, returning `obj` itself when the key is
 * absent or `fn` changed nothing — so an untouched layout keeps its identity.
 */
function patchKey(obj: unknown, key: string, fn: (value: unknown) => unknown): unknown {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return obj;
  const source = obj as Record<string, unknown>;
  if (!(key in source)) return obj;
  const next = fn(source[key]);
  return next === source[key] ? obj : { ...source, [key]: next };
}

/** Escape a title object's `text`, the only title form Plotly still renders. */
function escapeTitle(title: unknown): unknown {
  return patchKey(title, "text", escapeDeep);
}

/** Plotly's cartesian axis keys: `xaxis`, `yaxis`, `xaxis2`, … */
const AXIS_KEY = /^[xy]axis\d*$/;

/**
 * Return a trace with markup escaped in every field that renders as markup.
 *
 * Applied centrally rather than at each chart because there are fifteen chart
 * components and the fix has to hold for the sixteenth. Numbers, dates and
 * anything else non-string pass through, so a numeric axis is unaffected.
 */
export function escapeTraceMarkup(trace: unknown): unknown {
  if (!trace || typeof trace !== "object") return trace;
  const source = trace as Record<string, unknown>;
  let copy: Record<string, unknown> | null = null;
  for (const field of MARKUP_BEARING_TRACE_FIELDS) {
    if (!(field in source)) continue;
    const escaped = escapeDeep(source[field]);
    if (escaped === source[field]) continue;
    copy ??= { ...source };
    copy[field] = escaped;
  }
  return copy ?? trace;
}

/**
 * Return a layout with markup escaped in the text Plotly renders from it.
 *
 * Trace escaping does not reach the layout, and layout text is not always a
 * literal: a chart title can name the metrics it plots, and metric names are
 * project-controlled. Covered here: the chart title, each cartesian axis's title
 * and `ticktext`, the legend title, and annotation text. Titles are read from
 * `title.text` only — Plotly no longer accepts a bare string title, so one given
 * that way is not rendered and has nothing to escape.
 *
 * These fields are escaped WHOLE, so intentional markup in them would print as
 * text; none of them carries any today. Identity is kept when nothing changed,
 * for the same re-plot reason as traces.
 */
export function escapeLayoutMarkup(layout: unknown): unknown {
  if (!layout || typeof layout !== "object") return layout;
  let next = patchKey(layout, "title", escapeTitle);
  next = patchKey(next, "legend", (legend) => patchKey(legend, "title", escapeTitle));
  next = patchKey(next, "annotations", (list) =>
    mapSame(list, (note) =>
      patchKey(patchKey(note, "text", escapeDeep), "hovertext", escapeDeep),
    ),
  );
  for (const key of Object.keys(layout as Record<string, unknown>)) {
    if (!AXIS_KEY.test(key)) continue;
    next = patchKey(next, key, (axis) =>
      patchKey(patchKey(axis, "title", escapeTitle), "ticktext", escapeDeep),
    );
  }
  return next;
}

