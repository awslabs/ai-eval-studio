// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import path from "path";
import { readFileSync } from "fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Surface the package version to the app (shown on the About page) without
// hardcoding it in two places. Read at config time from package.json.
const pkg = JSON.parse(
  readFileSync(path.resolve(__dirname, "package.json"), "utf-8"),
) as { version: string };

export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Vite 8's rolldown dev pre-bundler tree-shakes side-effect-only imports
  // as "unused" — Monaco's .contribution modules register basic-languages via
  // side effects, so they never run in dev (prod vite build is unaffected).
  // Pin them here so the dev pre-bundler anchors them and registration runs.
  optimizeDeps: {
    include: [
      "monaco-editor/esm/vs/editor/editor.api",
      "monaco-editor/esm/vs/basic-languages/python/python.contribution",
      "monaco-editor/esm/vs/basic-languages/markdown/markdown.contribution",
      "monaco-editor/esm/vs/basic-languages/yaml/yaml.contribution",
      "monaco-editor/esm/vs/basic-languages/ini/ini.contribution",
      "monaco-editor/esm/vs/basic-languages/javascript/javascript.contribution",
      "monaco-editor/esm/vs/basic-languages/typescript/typescript.contribution",
      "monaco-editor/esm/vs/basic-languages/html/html.contribution",
      "monaco-editor/esm/vs/basic-languages/css/css.contribution",
      "monaco-editor/esm/vs/basic-languages/shell/shell.contribution",
      "monaco-editor/esm/vs/language/json/monaco.contribution",
    ],
  },
  server: {
    port: 3100,
    proxy: {
      "/api": {
        target: "http://localhost:8100",
        changeOrigin: true,
        // SSE requires: no buffering, long timeout for tool execution
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            // Prevent proxy from buffering SSE chunks
            proxyRes.headers["cache-control"] = "no-cache";
            proxyRes.headers["x-accel-buffering"] = "no";
          });
        },
        // 10 minute timeout for long-running agent turns with tool calls
        timeout: 600000,
      },
      // WebSocket proxy for the realtime envelope channel (/ws).
      // ``ws: true`` makes Vite forward the HTTP Upgrade handshake to
      // the backend; without it the browser's ``ws://localhost:3100/ws``
      // never reaches uvicorn (8100), so live chat/job streaming silently
      // fails in local dev and only renders after a refresh (which reloads
      // history via REST). Cloud is unaffected — it routes /ws through
      // CloudFront, not this dev proxy.
      "/ws": {
        target: "ws://localhost:8100",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
