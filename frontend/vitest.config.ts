// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Vitest configuration.
 *
 * Separate from `vite.config.ts` on purpose: that config carries a dev-server
 * proxy, a Monaco `optimizeDeps` workaround and a Tailwind plugin, none of which
 * a unit-test run needs. Sharing it would make every test run pay for the app
 * build graph and would couple test behavior to dev-server settings.
 *
 * `environment: "jsdom"` is set per-file rather than globally — the first tests
 * are pure reducers and a ws-client fold that need no DOM, and jsdom setup is the
 * dominant cost in a suite this size. Component tests opt in with a
 * `@vitest-environment jsdom` docblock.
 */
import path from "path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    // Node by default; DOM-dependent files declare `@vitest-environment jsdom`.
    environment: "node",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    // Fail on an unhandled rejection rather than printing it and passing — the
    // backend suite's equivalent leniency is how swallowed async failures hid.
    dangerouslyIgnoreUnhandledErrors: false,
    coverage: {
      provider: "v8",
      // Reported, not enforced. A threshold on a suite this young would either be
      // set so low it means nothing or would block honest work; the value here is
      // seeing which of the bug-history hot spots are actually covered.
      reporter: ["text", "html"],
      include: ["src/lib/**", "src/hooks/**"],
    },
  },
});
