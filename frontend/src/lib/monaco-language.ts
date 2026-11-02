// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Map backend language strings (from ``FileContentResponse.language``) to
 * Monaco editor language identifiers. Single source of truth for the
 * mapping — both the file viewer (``FilePreview``) and the code editor
 * (``CodeTab``) consume this so the two stay in sync as we add languages.
 */

/** Backend language → Monaco language id. */
const BACKEND_TO_MONACO: Record<string, string> = {
  python: "python",
  json: "json",
  jsonl: "json",
  markdown: "markdown",
  yaml: "yaml",
  toml: "ini",
  csv: "plaintext",
  text: "plaintext",
  javascript: "javascript",
  typescript: "typescript",
  html: "html",
  css: "css",
  shell: "shell",
};

/**
 * Resolve a Monaco language id for a backend-provided language tag.
 *
 * @param lang - Language string as produced by the backend file API.
 * @returns The matching Monaco language id; ``"plaintext"`` for unknowns.
 */
export function monacoLanguageFor(lang: string): string {
  return BACKEND_TO_MONACO[lang] ?? "plaintext";
}
