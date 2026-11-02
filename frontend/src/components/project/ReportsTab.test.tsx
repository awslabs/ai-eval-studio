// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * The regenerate endpoints answer 202 before the narrator finishes, because a
 * narration outlasts the CDN's 30-second origin timeout and a request that
 * waited for it reached the browser as a 504. These tests pin the tab's side of
 * that contract: a regenerate stays pending until a ``report`` event reports
 * its outcome, and every way that outcome can arrive (or fail to) ends it.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";

import type { BackendEvent } from "@/hooks/useBackendEvents";

/** The tab's ``report`` bus handler and gap handler, captured so a test can fire them. */
let reportHandler: ((event: BackendEvent) => void) | null = null;
let gapHandler: (() => void) | null = null;

vi.mock("@/hooks/useBackendEvents", () => ({
  useProjectEvent: (match: { resource: string }, handler: (event: BackendEvent) => void) => {
    if (match.resource === "report") reportHandler = handler;
  },
  useCursorGapHandler: (_projectId: string | null, onGap: () => void) => {
    gapHandler = onGap;
  },
}));

vi.mock("@/hooks/useEvalRuns", () => ({
  useEvalRuns: () => ({
    runs: [
      {
        _file: "run-a.json",
        prompt_version: "v2",
        timestamp: "2026-10-01T10:00:00Z",
        source: "manual",
        model_runs: [{ model: "m1", total: 3 }],
      },
      {
        _file: "run-b.json",
        prompt_version: "v1",
        timestamp: "2026-09-30T10:00:00Z",
        source: "manual",
        model_runs: [{ model: "m1", total: 3 }],
      },
    ],
    loading: false,
    error: null,
  }),
}));

vi.mock("@/components/reports/ExecReport", () => ({
  ExecReport: ({ narrative }: { narrative: { intro?: string } | null }) => (
    <div>exec: {narrative?.intro ?? "none"}</div>
  ),
}));

vi.mock("@/components/reports/RunReport", () => ({
  RunReport: ({ narrative }: { narrative: { context?: string } | null }) => (
    <div>run: {narrative?.context ?? "none"}</div>
  ),
}));

vi.mock("@/lib/api", () => ({
  api: {
    getExecutiveReport: vi.fn(),
    getRunReport: vi.fn(),
    regenerateExecutiveReport: vi.fn(),
    regenerateReport: vi.fn(),
  },
}));

const { ReportsTab } = await import("./ReportsTab");
const { api } = await import("@/lib/api");

function envelope(scope: "run" | "executive", text: string) {
  return {
    schema_version: 1,
    scope,
    facts: { project_name: "Demo", prompt_label: "v2" },
    narrative: scope === "executive" ? { intro: text } : { context: text },
    generated_at: "2026-10-01T10:00:00Z",
    stale: false,
  };
}

function renderTab(search = "?rrun=run-a.json") {
  return render(
    <MemoryRouter initialEntries={[`/p/proj-1${search}`]}>
      <ReportsTab projectId="proj-1" />
    </MemoryRouter>,
  );
}

/** Fire a ``report`` bus event into the tab. */
function fire(event: Partial<BackendEvent>) {
  act(() => {
    reportHandler!({ resource: "report", ...event } as BackendEvent);
  });
}

const regenerateButton = () => screen.getByRole("button", { name: /regenerate/i });

beforeEach(() => {
  reportHandler = null;
  gapHandler = null;
  vi.mocked(api.getExecutiveReport).mockReset().mockResolvedValue(
    envelope("executive", "old summary") as never,
  );
  vi.mocked(api.getRunReport).mockReset().mockResolvedValue(
    envelope("run", "old run summary") as never,
  );
  vi.mocked(api.regenerateExecutiveReport).mockReset().mockResolvedValue(undefined);
  vi.mocked(api.regenerateReport).mockReset().mockResolvedValue(undefined);
});

afterEach(cleanup);

describe("ReportsTab regenerate", () => {
  it("stays pending after the 202 and refetches when the report is updated", async () => {
    renderTab();
    await screen.findByText("exec: old summary");

    fireEvent.click(regenerateButton());
    await waitFor(() => expect(api.regenerateExecutiveReport).toHaveBeenCalledWith("proj-1"));

    // The request has resolved, but the narrator is still running: the old
    // report stays on screen under the badge, with no refetch yet.
    expect(await screen.findByText("Updating…")).toBeInTheDocument();
    expect(screen.getByText("exec: old summary")).toBeInTheDocument();
    expect(regenerateButton()).toBeDisabled();
    expect(api.getExecutiveReport).toHaveBeenCalledTimes(1);

    vi.mocked(api.getExecutiveReport).mockResolvedValue(
      envelope("executive", "new summary") as never,
    );
    fire({ type: "updated", scope: "executive" });

    expect(await screen.findByText("exec: new summary")).toBeInTheDocument();
    expect(screen.queryByText("Updating…")).not.toBeInTheDocument();
    expect(regenerateButton()).toBeEnabled();
  });

  it("ends the pending regenerate with a message when the narrator fails", async () => {
    renderTab();
    await screen.findByText("exec: old summary");
    fireEvent.click(regenerateButton());
    await screen.findByText("Updating…");

    fire({ type: "failed", scope: "executive" });

    expect(await screen.findByText(/could not be regenerated/i)).toBeInTheDocument();
    expect(screen.queryByText("Updating…")).not.toBeInTheDocument();
    // Nothing new was written, so there is nothing to refetch.
    expect(api.getExecutiveReport).toHaveBeenCalledTimes(1);
    expect(screen.getByText("exec: old summary")).toBeInTheDocument();
  });

  it("ends a per-run regenerate only on that run's outcome", async () => {
    renderTab("?rtype=run&rrun=run-a.json");
    await screen.findByText("run: old run summary");
    fireEvent.click(regenerateButton());
    await waitFor(() => expect(api.regenerateReport).toHaveBeenCalledWith("proj-1", "run-a.json"));
    await screen.findByText("Updating…");

    // An automatic narration of another run, and the executive report, end nothing.
    fire({ type: "updated", scope: "run", run_filename: "run-b.json" });
    fire({ type: "failed", scope: "executive" });
    expect(screen.getByText("Updating…")).toBeInTheDocument();
    expect(api.getRunReport).toHaveBeenCalledTimes(1);

    fire({ type: "updated", scope: "run", run_filename: "run-a.json" });
    await waitFor(() => expect(screen.queryByText("Updating…")).not.toBeInTheDocument());
    await waitFor(() => expect(api.getRunReport).toHaveBeenCalledTimes(2));
  });

  it("ends the pending regenerate when the background task crashes", async () => {
    renderTab();
    await screen.findByText("exec: old summary");
    fireEvent.click(regenerateButton());
    await screen.findByText("Updating…");

    fire({ type: "error" });

    await waitFor(() => expect(screen.queryByText("Updating…")).not.toBeInTheDocument());
  });

  it("shows the error and stops waiting when the request is refused", async () => {
    vi.mocked(api.regenerateExecutiveReport).mockRejectedValue(
      new Error("Regenerate failed (403)"),
    );
    renderTab();
    await screen.findByText("exec: old summary");
    fireEvent.click(regenerateButton());

    expect(await screen.findByText("Regenerate failed (403)")).toBeInTheDocument();
    expect(screen.queryByText("Updating…")).not.toBeInTheDocument();
    expect(regenerateButton()).toBeEnabled();
  });

  it("stops waiting and refetches after a gap in the event stream", async () => {
    renderTab();
    await screen.findByText("exec: old summary");
    fireEvent.click(regenerateButton());
    await screen.findByText("Updating…");

    // A coordinator restart both ends the narration and can drop its event.
    act(() => gapHandler!());

    await waitFor(() => expect(screen.queryByText("Updating…")).not.toBeInTheDocument());
    await waitFor(() => expect(api.getExecutiveReport).toHaveBeenCalledTimes(2));
  });
});
