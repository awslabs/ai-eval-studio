// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Self-contained HTML / print export for eval reports.
 *
 * Builds ONE standalone HTML document from the already-rendered report DOM
 * (ExecReport / RunReport) and offers two outputs from the same document:
 *   - downloadReportHtml() — saves it as a single .html file
 *   - printReport()        — prints it via an offscreen iframe (Save as PDF)
 *
 * WHY this approach (rather than a separate PDF component tree):
 *   - The on-screen report is already real, semantic HTML (shadcn <Table> →
 *     <table>/<thead>/<tr>, CSS-bar charts, no canvas). We reuse it verbatim —
 *     single source of truth, no second layout to keep in sync.
 *   - The document carries NO `.dark` class, so the app's `:root` (light) theme
 *     tokens apply automatically → a clean white "artifact" look with zero
 *     theme-rewriting. (Dark mode is `:is(.dark *)`-scoped; nothing matches.)
 *   - Pagination is delegated to the browser's paged-media engine (repeating
 *     <thead>, break-inside:avoid), which is far more reliable than @react-pdf
 *     for variable-length content.
 *
 * The captured CSS is whatever the live app shipped (Tailwind v4 utilities +
 * theme tokens), so the export tracks the app's styling without a second
 * stylesheet to maintain.
 */

import { saveBlob } from "@/lib/download-utils";

/** Input for an export: the report markup + a human title used for the doc + filename. */
export interface ReportExportInput {
  /**
   * The rendered report subtree to export. The standalone document has no
   * `.dark` ancestor, so its class names resolve to the light palette.
   */
  node: HTMLElement;
  /** Document <title>. */
  title: string;
}

/**
 * Serialize every readable (same-origin) stylesheet in the live document into
 * one CSS string. Cross-origin sheets throw on `.cssRules` access and are
 * skipped. Includes constructable/adopted sheets when present.
 */
function collectDocumentCss(): string {
  const chunks: string[] = [];

  const dumpRules = (rules: CSSRuleList | undefined) => {
    if (!rules) return;
    for (const rule of Array.from(rules)) chunks.push(rule.cssText);
  };

  for (const sheet of Array.from(document.styleSheets)) {
    try {
      dumpRules(sheet.cssRules);
    } catch {
      // Cross-origin / unreadable sheet — skip it.
    }
  }

  // Constructable stylesheets are not part of document.styleSheets.
  const adopted = (document as Document & { adoptedStyleSheets?: CSSStyleSheet[] }).adoptedStyleSheets;
  if (adopted) {
    for (const sheet of adopted) {
      try {
        dumpRules(sheet.cssRules);
      } catch {
        // ignore
      }
    }
  }

  return chunks.join("\n");
}

/**
 * Report-artifact overrides layered AFTER the captured app CSS:
 *   - white page, centered document column;
 *   - paged-media rules so the browser paginates cleanly instead of slicing.
 *
 * Font is intentionally left to the captured CSS (Geist when it resolves, plain
 * sans-serif otherwise) — we do not fight to match typography.
 */
const REPORT_PRINT_CSS = `
  @page { size: letter; margin: 14mm; }
  html, body { background: #ffffff; }
  body { margin: 0; }
  .report-export-root {
    max-width: 880px;
    margin: 0 auto;
    padding: 28px;
    color: var(--foreground, #1a1d24);
  }

  /* Every <details> is force-opened for the export (see prepareBodyHtml) so no
     content is hidden. Drop the disclosure triangle + interactive cursor so the
     summaries read as plain section headers. RunReport keeps its own ▼ marker. */
  details > summary { list-style: none; cursor: default; }
  details > summary::-webkit-details-marker { display: none; }

  /* Pagination — keep atomic blocks whole; the browser still breaks anything
     taller than a page (graceful, unlike a hard clip). Both the modern and
     legacy properties for Chromium + Safari coverage. */
  [data-slot="card"],
  tr {
    break-inside: avoid;
    page-break-inside: avoid;
  }
  /* Repeat table headers on every page a table spans. */
  thead { display: table-header-group; }
  /* Don't strand a heading at the foot of a page. */
  h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
  /* The final-prompt block can be very long: let it flow across pages. */
  pre { break-inside: auto; page-break-inside: auto; white-space: pre-wrap; }

  /* Table fitting is PRINT-ONLY. On screen (the .html file) the report keeps
     the app's native table behavior (overflow-x:auto, nowrap) — touching it
     globally made wide tables bleed out of the document column. */
  @media print {
    .report-export-root { max-width: none; margin: 0; padding: 0; }
    /* The shadcn wrapper is overflow-x:auto → a dead, unscrollable scrollbar
       that clips off-page columns in paged output. Lay it out visibly, shrink
       the type, tighten padding, and let cells wrap so the table fits the page
       (the on-screen default is whitespace-nowrap). Over-long single tokens
       break only as a last resort. */
    [data-slot="table-container"] { overflow: visible !important; }
    [data-slot="table"] { font-size: 8.5pt; }
    [data-slot="table-head"],
    [data-slot="table-cell"] {
      white-space: normal !important;
      overflow-wrap: break-word;
      padding: 4px 6px;
    }
  }
`;

/** Minimal HTML-escape for the <title> (body markup is already-rendered DOM). */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Clone the report subtree and force every `<details>` open so collapsed
 * appendix content (methodology, final prompt, failure groups) is fully
 * present in the static export instead of frozen shut and unclickable in a PDF.
 * Cloning keeps the live on-screen report untouched.
 */
function prepareBodyHtml(node: HTMLElement): string {
  const clone = node.cloneNode(true) as HTMLElement;
  clone.querySelectorAll("details").forEach((d) => d.setAttribute("open", ""));
  return clone.innerHTML;
}

/** Assemble the full standalone HTML document string. */
function buildReportDocument({ node, title }: ReportExportInput): string {
  const appCss = collectDocumentCss();
  const bodyHtml = prepareBodyHtml(node);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${escapeHtml(title)}</title>
<style>${appCss}</style>
<style>${REPORT_PRINT_CSS}</style>
</head>
<body>
<div class="report-export-root">${bodyHtml}</div>
</body>
</html>`;
}

/** Save the report as a single self-contained .html file. */
export function downloadReportHtml(input: ReportExportInput, filename: string): void {
  const html = buildReportDocument(input);
  // Shared save helper → "choose where to save" picker on Chromium, anchor
  // fallback elsewhere. Fire-and-forget; the helper handles cancel + fallback.
  void saveBlob(html, filename, "text/html;charset=utf-8");
}

/**
 * Print the report through an offscreen iframe (the user picks "Save as PDF"
 * in the browser dialog). Printing an isolated document — rather than the live
 * page — means no app chrome leaks in and the dark theme never applies.
 */
export function printReport(input: ReportExportInput): void {
  const html = buildReportDocument(input);

  const iframe = document.createElement("iframe");
  iframe.setAttribute("aria-hidden", "true");
  // Offscreen but real Letter-ish size so layout + pagination actually compute.
  iframe.style.cssText =
    "position:fixed;left:-9999px;top:0;width:816px;height:1056px;border:0;";

  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    // Defer so the print job has fully detached from the frame.
    setTimeout(() => iframe.remove(), 1000);
  };

  iframe.onload = () => {
    const win = iframe.contentWindow;
    if (!win) {
      iframe.remove();
      return;
    }
    win.addEventListener("afterprint", cleanup, { once: true });
    // One frame so the inlined <style> is applied before the snapshot.
    requestAnimationFrame(() => {
      win.focus();
      win.print();
      // Fallback: some browsers never fire afterprint (or the user cancels).
      setTimeout(cleanup, 60000);
    });
  };

  // srcdoc avoids the about:blank load race that document.write() introduces.
  iframe.srcdoc = html;
  document.body.appendChild(iframe);
}
