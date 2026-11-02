// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import "@/lib/monaco-setup"; // wire Monaco to the bundled npm package, not the CDN
import App from "./App";
import { AuthGate } from "@/components/AuthGate";

/** Suppress known Monaco DiffEditor disposal race condition (cosmetic, not functional). */
window.addEventListener("error", (e) => {
  if (e.message?.includes("TextModel got disposed before DiffEditorWidget")) {
    e.preventDefault();
  }
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AuthGate>
      <App />
    </AuthGate>
  </StrictMode>,
);
