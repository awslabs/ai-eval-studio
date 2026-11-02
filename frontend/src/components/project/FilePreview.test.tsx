// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Fullscreen must be a property of the Data tab's preview container, not of one
 * viewer.
 *
 * The defect: `MaximizablePanel` lived inside `CsvPreview`, so the maximize
 * control existed for parsed CSVs and silently vanished for JSON, markdown and
 * every other file type. Nothing looked wrong in either branch on its own —
 * the control was simply absent from one of them, which reads as a broken
 * button rather than a per-format capability.
 *
 * These tests assert the effect (a control exists and opens an overlay) for
 * BOTH branches, so putting the panel back inside one viewer fails.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { FilePreview } from "./FilePreview";
import type { FileContentResponse } from "@/lib/types";

/** Monaco options from the most recent render, for the layout assertion. */
let lastEditorOptions: Record<string, unknown> | undefined;

vi.mock("@monaco-editor/react", () => ({
  default: (props: { options?: Record<string, unknown> }) => {
    lastEditorOptions = props.options;
    return <div data-testid="monaco" />;
  },
}));

vi.mock("@/lib/theme-provider", () => ({
  useTheme: () => ({ mode: "light" }),
}));

// jsdom ships no ResizeObserver, and CsvPreview uses one to size columns to the
// pane. Stubbed rather than asserted on — column sizing is not what these tests
// are about, and without it the tabular branch cannot mount at all.
class ResizeObserverStub implements ResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

afterEach(() => {
  cleanup();
  lastEditorOptions = undefined;
});

/** A parsed CSV, the branch that always had fullscreen. */
const csvFile: FileContentResponse = {
  path: "data/cases.csv",
  content: "a,b\n1,2\n",
  size: 8,
  language: "csv",
  columns: ["a", "b"],
  rows: [{ a: 1, b: 2 }],
  total_rows: 1,
};

/** A JSON file, the branch that had none. */
const jsonFile: FileContentResponse = {
  path: "src/config.json",
  content: '{\n  "a": 1\n}\n',
  size: 12,
  language: "json",
  columns: null,
  rows: null,
  total_rows: null,
};

/** The container's maximize control, by its accessible label. */
function maximizeControl(fileName: string) {
  return screen.queryByRole("button", { name: `Open ${fileName} fullscreen` });
}

describe("FilePreview fullscreen is container-level", () => {
  it("offers fullscreen for a non-tabular file", async () => {
    // THE REGRESSION: before the panel moved to the container, this branch had
    // no maximize control at all.
    render(<FilePreview file={jsonFile} />);
    expect(maximizeControl("config.json")).toBeInTheDocument();
  });

  it("still offers fullscreen for a tabular file", async () => {
    render(<FilePreview file={csvFile} />);
    expect(maximizeControl("cases.csv")).toBeInTheDocument();
  });

  it.each([
    ["non-tabular", jsonFile, "config.json"],
    ["tabular", csvFile, "cases.csv"],
  ])("opens the overlay for a %s file, titled with the filename", async (_label, file, fileName) => {
    render(<FilePreview file={file} />);
    fireEvent.click(maximizeControl(fileName)!);

    // The overlay owns the close control, and the resting maximize control
    // hides itself — so this asserts a real state change, not just a click.
    expect(
      screen.getByRole("button", { name: "Close fullscreen view" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: fileName })).toBeInTheDocument();
    expect(maximizeControl(fileName)).not.toBeInTheDocument();
  });

  it("renders exactly one maximize control, not one per layer", async () => {
    // A container-level panel plus a leftover viewer-level one would give the
    // user two competing buttons.
    render(<FilePreview file={csvFile} />);
    expect(screen.getAllByRole("button", { name: /fullscreen$/ })).toHaveLength(1);
  });
});

describe("FilePreview viewer bodies", () => {
  it("gives Monaco automaticLayout so it resizes into the overlay", () => {
    // Monaco measures its container once and never again on its own, so
    // without this it keeps the resting pane's size and renders into a
    // fraction of the fullscreen overlay.
    render(<FilePreview file={jsonFile} />);
    expect(lastEditorOptions?.automaticLayout).toBe(true);
  });

  it("shows tabular counts only for the tabular branch", () => {
    render(<FilePreview file={csvFile} />);
    expect(screen.getByText("2 columns")).toBeInTheDocument();
    expect(screen.getByText("1 rows")).toBeInTheDocument();

    cleanup();
    render(<FilePreview file={jsonFile} />);
    expect(screen.queryByText(/columns$/)).not.toBeInTheDocument();
  });

  it("warns when the preview is truncated", () => {
    render(
      <FilePreview
        file={{ ...csvFile, rows: [{ a: 1, b: 2 }], total_rows: 5000 }}
      />,
    );
    expect(screen.getByText("preview truncated")).toBeInTheDocument();
    expect(screen.getByText("1 of 5,000 rows")).toBeInTheDocument();
  });
});
