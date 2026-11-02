// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared Monaco editor theme definitions.
 *
 * Uses simple transparent/inherit approach: Monaco inherits from vs-dark/vs
 * with transparent backgrounds so the parent container's CSS theme shows through.
 * This works with ANY theme without needing oklch→hex conversion.
 */

import type { Monaco } from "@monaco-editor/react";

/**
 * Define Monaco themes. Call in `beforeMount` of any Editor/DiffEditor.
 * Uses transparent background so the editor blends with whatever CSS theme
 * is applied to the page.
 */
export function defineAppTheme(monaco: Monaco): void {
  monaco.editor.defineTheme("app-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#00000000",
      "editorGutter.background": "#00000000",
      "diffEditor.insertedTextBackground": "#22c55e18",
      "diffEditor.removedTextBackground": "#ef444418",
    },
  });

  monaco.editor.defineTheme("app-light", {
    base: "vs",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#00000000",
      "editorGutter.background": "#00000000",
    },
  });
}

/**
 * Get the Monaco theme name for the current app mode.
 */
export function getMonacoTheme(isDark: boolean): string {
  return isDark ? "app-dark" : "app-light";
}
