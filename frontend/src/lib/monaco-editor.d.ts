// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Type shim for Monaco's slim ``editor.api`` entry under its full
 * ``esm/vs/...`` path, re-exporting the public package's types.
 *
 * ``editor.main`` pulls in the full basic-language registry (~85
 * languages); the slim ``editor.api`` entry is a sibling file with the
 * editor namespace only — no language imports.
 *
 * ``monaco-setup.ts`` does not import the path declared here. It imports
 * the slim entry as ``monaco-editor/editor/editor.api``, which the
 * package's ``exports`` map rewrites to ``./esm/vs/editor/editor.api.js``;
 * TypeScript's bundler resolution then finds the ``.d.ts`` beside it, so
 * nothing in ``src`` depends on this declaration.
 */
declare module "monaco-editor/esm/vs/editor/editor.api" {
  export * from "monaco-editor";
}

/* Internal Monaco modules ``monaco-setup.test.ts`` uses to see what is
 * registered and to catch the errors an editor reports, typed only as far as
 * the test uses them. Monaco ships no declarations for them. */
declare module "monaco-editor/platform/instantiation/common/extensions" {
  export function getSingletonServiceDescriptors(): [{ toString(): string }, unknown][];
}
declare module "monaco-editor/editor/browser/editorExtensions" {
  export const EditorExtensionsRegistry: {
    getEditorContributions(): { id: string }[];
  };
}
declare module "monaco-editor/base/common/errors" {
  export const errorHandler: { onUnexpectedError(e: unknown): void };
}
declare module "monaco-editor/languages/features/json/jsonMode";
