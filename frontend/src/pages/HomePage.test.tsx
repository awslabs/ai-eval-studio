// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * What the project card shows, and what it no longer shows.
 *
 * Two regressions are pinned here. The metadata badges vanished whenever a run
 * had a metric-audit sidecar, because the summary's `model_count` came back 0 —
 * that fix is in the backend, so these tests assert the card renders the badges
 * it is GIVEN, which is the half that has to hold for the fix to be visible.
 *
 * And the card used to print "No description" on every project, since the field
 * could only be set at creation and was therefore always empty. The description
 * line must now be absent rather than filled with a placeholder — the place to
 * add one is the project header.
 */

import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";
import { HomePage } from "./HomePage";
import type { ProjectSummary } from "@/lib/types";

const listProjects = vi.fn();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: { listProjects: () => listProjects() },
  };
});

/** A summary with every enriched field populated; override per test. */
function summary(over: Partial<ProjectSummary> = {}): ProjectSummary {
  return {
    // Hex letters in the last group on purpose: an all-digit 12-character final
    // group would look like an AWS account id to secret scanners.
    id: "11111111-2222-3333-4444-5f5f5f5f5f5f",
    name: "Support Triage v2",
    description: "",
    created_at: "2026-09-08T21:17:00Z",
    last_activity: "2026-09-08T21:17:00Z",
    model_count: 2,
    latest_pass_rate: 0.96,
    is_running: false,
    ...over,
  };
}

/** Render the page and wait for the project list to settle. */
async function renderList(projects: ProjectSummary[]): Promise<void> {
  listProjects.mockResolvedValue(projects);
  render(
    <MemoryRouter>
      <HomePage />
    </MemoryRouter>,
  );
  await waitFor(() => expect(screen.getByText(projects[0].name)).toBeInTheDocument());
}

beforeEach(() => listProjects.mockReset());
afterEach(cleanup);

describe("ProjectCard", () => {
  it("shows the model count and pass rate a summary reports", async () => {
    await renderList([summary()]);

    expect(screen.getByText("2 models")).toBeInTheDocument();
    expect(screen.getByText("96% pass")).toBeInTheDocument();
  });

  it("singularises a one-model run", async () => {
    await renderList([summary({ model_count: 1 })]);

    expect(screen.getByText("1 model")).toBeInTheDocument();
  });

  it("omits the badges for a project that has never been evaluated", async () => {
    await renderList([summary({ model_count: 0, latest_pass_rate: null })]);

    expect(screen.queryByText(/models?$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/% pass/)).not.toBeInTheDocument();
  });

  it("shows a Running badge while an eval is in flight", async () => {
    await renderList([summary({ is_running: true })]);

    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("shows no Running badge when idle", async () => {
    await renderList([summary({ is_running: false })]);

    expect(screen.queryByText("Running")).not.toBeInTheDocument();
  });

  it("omits the description line entirely when there is no description", async () => {
    await renderList([summary({ description: "" })]);

    // The old placeholder occupied a line on every card while saying nothing.
    expect(screen.queryByText("No description")).not.toBeInTheDocument();
  });

  it("shows the description when the project has one", async () => {
    await renderList([summary({ description: "Retry after the judge fix" })]);

    expect(screen.getByText("Retry after the judge fix")).toBeInTheDocument();
  });

  it("renders no status dot", async () => {
    /* The dot encoded a four-state lifecycle ladder in two near-identical greys
     * plus blue and green, with no legend — and its "evaluated" state was the
     * same condition as the badges beside it. */
    await renderList([summary()]);

    expect(document.querySelectorAll("span.rounded-full.size-2")).toHaveLength(0);
  });

  it("still offers the delete action", async () => {
    await renderList([summary({ name: "Deletable" })]);

    expect(
      screen.getByRole("button", { name: "Delete Deletable" }),
    ).toBeInTheDocument();
  });
});

describe("live refresh", () => {
  /* There is no WebSocket for this view — the stream is subscribed per project
   * and the list has no project context — so a running eval only becomes
   * "finished" on a refetch. These assert the poll runs exactly when it should.
   *
   * Fake timers are installed BEFORE mounting: the interval is created during
   * the mount effect, and swapping the clock afterwards leaves a real interval
   * that no amount of advancing controls. */

  afterEach(() => vi.useRealTimers());

  /** Mount the page under fake timers with the mount fetch already settled. */
  async function mount(projects: ProjectSummary[]): Promise<void> {
    vi.useFakeTimers();
    listProjects.mockResolvedValue(projects);
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
  }

  /** Advance the fake clock, flushing the fetches each tick schedules. */
  async function advance(ms: number): Promise<void> {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  it("polls while a project is running", async () => {
    await mount([summary({ is_running: true })]);
    const afterMount = listProjects.mock.calls.length;

    await advance(30_000);

    expect(listProjects.mock.calls.length).toBeGreaterThan(afterMount);
  });

  it("does not poll when every project is idle", async () => {
    await mount([summary({ is_running: false })]);
    const afterMount = listProjects.mock.calls.length;

    await advance(120_000);

    expect(listProjects.mock.calls.length).toBe(afterMount);
  });

  it("stops polling once the running project finishes", async () => {
    await mount([summary({ is_running: true })]);

    // The next tick reports it finished, which should retire the interval.
    listProjects.mockResolvedValue([summary({ is_running: false })]);
    await advance(30_000);
    expect(screen.queryByText("Running")).not.toBeInTheDocument();

    const afterFinish = listProjects.mock.calls.length;
    await advance(120_000);
    expect(listProjects.mock.calls.length).toBe(afterFinish);
  });

  it("does not poll while the tab is hidden", async () => {
    const spy = vi
      .spyOn(document, "visibilityState", "get")
      .mockReturnValue("hidden");
    await mount([summary({ is_running: true })]);
    const afterMount = listProjects.mock.calls.length;

    await advance(120_000);

    // The focus/visibilitychange listener covers the return-to-tab case, so
    // skipping hidden ticks loses nothing.
    expect(listProjects.mock.calls.length).toBe(afterMount);
    spy.mockRestore();
  });
});
