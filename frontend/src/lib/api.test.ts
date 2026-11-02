// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Transport contract tests for the API client.
 *
 * Verifies that HTTP caching is disabled for every API request so the
 * browser cannot serve a stale cached response after SPA navigation —
 * which made a deleted eval run reappear as an unwanted card.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { MockInstance } from "vitest";
import { api } from "./api";
import * as auth from "./auth";

describe("API transport — HTTP cache bypass", () => {
  // Declared without an explicit annotation: a generic
  // `ReturnType<typeof vi.spyOn>` widens the mock's signature and fails to
  // type-check against the specific overloads of `fetch`.
  let fetchSpy: MockInstance;
  let getAccessTokenSpy: MockInstance;

  beforeEach(() => {
    // Stub fetch to return a successful JSON response
    fetchSpy = vi.spyOn(globalThis, "fetch" as never).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: "test" }),
    } as Response);

    // Stub getAccessToken to return empty string (dev mode — no token)
    getAccessTokenSpy = vi.spyOn(auth, "getAccessToken" as never).mockResolvedValue("" as never);
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    getAccessTokenSpy.mockRestore();
  });

  it("issues run-list requests with HTTP caching disabled", async () => {
    await api.listEvalRuns("test-project");

    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/projects/test-project/eval/runs",
      expect.objectContaining({
        cache: "no-store",
      }),
    );
  });

  it("issues unrelated API requests with HTTP caching disabled (chokepoint-wide guarantee)", async () => {
    await api.getProjectActivity("test-project");

    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/projects/test-project/activity",
      expect.objectContaining({
        cache: "no-store",
      }),
    );
  });

  it("enforces the non-storing mode even when caller supplies cache: default", async () => {
    // Call apiFetch directly to test the enforcement layer
    const { apiFetch } = await import("./api");
    await apiFetch("/api/test", { cache: "default" });

    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/test",
      expect.objectContaining({
        cache: "no-store",
      }),
    );
  });
});
