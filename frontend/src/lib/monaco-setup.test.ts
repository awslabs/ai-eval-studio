// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * Monaco must have every editor service registered before the first editor
 * exists.
 *
 * The standalone editor fixes its set of services once, when the first editor
 * is created. A contribution registered after that point is still attached to
 * every later editor, but the services it depends on are not in the fixed set,
 * so each later editor throws "depends on UNKNOWN service" on mount and loses
 * hover, the context menu and quick fixes.
 *
 * The slim `editor.api` entry registers no contributions. The JSON language
 * service registers all of them, from a module it loads lazily when the first
 * JSON model appears, which is after the first editor in practice. So the test
 * replays that order: an editor, then the lazy module, then a second editor.
 */
import { describe, expect, it } from "vitest";

/** The jsdom gaps the editor touches on create. Rendering itself is not tested. */
function stubBrowserApis(): void {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  };
  window.matchMedia = () =>
    ({
      matches: false,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
    }) as unknown as MediaQueryList;
  document.queryCommandSupported = () => false;
}

describe("monaco-setup", () => {
  // Importing the whole editor feature set takes a few seconds to transform.
  const timeout = 60_000;

  it("leaves no editor service to register after the first editor", { timeout }, async () => {
    stubBrowserApis();
    const { errorHandler } = await import("monaco-editor/base/common/errors");
    const errors: string[] = [];
    errorHandler.onUnexpectedError = (e: unknown) => {
      errors.push(e instanceof Error ? e.message : String(e));
    };

    await import("./monaco-setup");
    const monaco = await import("monaco-editor/editor/editor.api");
    const { getSingletonServiceDescriptors } = await import(
      "monaco-editor/platform/instantiation/common/extensions"
    );
    const { EditorExtensionsRegistry } = await import(
      "monaco-editor/editor/browser/editorExtensions"
    );
    const services = () =>
      new Set(getSingletonServiceDescriptors().map(([id]) => id.toString()));

    const first = document.createElement("div");
    document.body.appendChild(first);
    monaco.editor.create(first, { value: "x = 1", language: "python" });
    const servicesAtFirstEditor = services();

    // What the first JSON model triggers through `languages.onLanguage`.
    await import("monaco-editor/languages/features/json/jsonMode");

    const late = [...services()].filter((id) => !servicesAtFirstEditor.has(id));
    expect(late, "services registered after the first editor").toEqual([]);

    // Most contributions instantiate on idle; asking for each one by id
    // instantiates it now, which is where a missing service throws.
    const second = document.createElement("div");
    document.body.appendChild(second);
    const editor = monaco.editor.create(second, { value: "", language: "plaintext" });
    const contributions = EditorExtensionsRegistry.getEditorContributions();
    expect(contributions.length).toBeGreaterThan(0);
    for (const { id } of contributions) editor.getContribution(id);

    expect(errors.filter((m) => m.includes("depends on UNKNOWN service"))).toEqual([]);
  });
});
