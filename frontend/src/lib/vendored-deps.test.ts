// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Version floors for dependencies that VENDOR a copy of a vulnerable library.
 *
 * Two packages here ship an inlined copy of another library rather than
 * importing it from `node_modules`, which makes the usual evidence misleading in
 * a specific way: `npm ls` and `npm audit` read the dependency graph, and an
 * `overrides` entry moves the graph, but the browser executes the inlined copy.
 * Both were reported "patched" on graph evidence while the shipped bytes still
 * carried the vulnerable version.
 *
 * Grepping the built bundle is not the answer either — minification strips the
 * license banners that carry the version, so a bundle grep for a version string
 * passes whether or not the upgrade landed. The version is only legible in the
 * installed package, which is what these tests read.
 *
 * A failure here means the floor was crossed by a downgrade or a lockfile
 * regression, and the fix is to raise the package version, never to relax the
 * floor.
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require_ = createRequire(import.meta.url);

/** Parse a dotted version into comparable numeric parts. */
function parseVersion(raw: string): number[] {
  return raw.split(".").map((part) => Number.parseInt(part, 10));
}

/** True when `actual` is the same as or newer than `floor`. */
function atLeast(actual: string, floor: string): boolean {
  const a = parseVersion(actual);
  const f = parseVersion(floor);
  for (let i = 0; i < Math.max(a.length, f.length); i += 1) {
    const av = a[i] ?? 0;
    const fv = f[i] ?? 0;
    if (av !== fv) return av > fv;
  }
  return true;
}

describe("monaco-editor's inlined DOMPurify", () => {
  /**
   * monaco imports DOMPurify from its own tree at
   * `esm/vs/base/browser/dompurify/dompurify.js` rather than from the
   * `node_modules/dompurify` an override can reach, so patching DOMPurify here
   * means upgrading monaco.
   *
   * The floor is what monaco currently ships, NOT a fully-patched DOMPurify.
   * Four advisories still affect <=3.4.12 and monaco 0.56.0 — the newest
   * release — pins 3.4.8, so no monaco upgrade clears them today. Monaco's own
   * usage does not reach any of the four: it never calls `setConfig`,
   * `clearConfig`, `IN_PLACE` or `CUSTOM_ELEMENT_HANDLING` (it calls `sanitize`
   * with `RETURN_DOM_FRAGMENT`/`RETURN_TRUSTED_TYPE` and clears hooks with
   * `removeAllHooks`). Raise this floor to 3.4.13 when a monaco release carries
   * it.
   */
  it("is at or above the version monaco currently pins", () => {
    const vendored = require_.resolve(
      "monaco-editor/base/browser/dompurify/dompurify.js",
    );
    const banner = readFileSync(vendored, "utf8").slice(0, 400);
    const match = banner.match(/DOMPurify (\d+\.\d+\.\d+)/);

    expect(
      match,
      `no DOMPurify version banner in ${vendored} — monaco may have changed how ` +
        "it vendors DOMPurify, so this floor is no longer being checked",
    ).not.toBeNull();
    expect(atLeast(match![1], "3.4.8"), `vendored DOMPurify is ${match![1]}`).toBe(
      true,
    );
  });
});

describe("plotly.js-dist-min's inlined MapLibre", () => {
  /**
   * `plotly.js-dist-min` is a single pre-built file with MapLibre GL inlined.
   * The separate `node_modules/maplibre-gl` is not the code the browser runs, so
   * bumping it leaves the bundle unchanged. 6.4.1 is the fixed release for the
   * sanitizer-bypass advisory against 4.7.1.
   */
  it("is at or above the fixed release", () => {
    const dist = require_.resolve("plotly.js-dist-min");
    const source = readFileSync(dist, "utf8");
    const versions = [
      ...source.matchAll(/maplibre[^\n]{0,80}?(\d+\.\d+\.\d+)/g),
    ].map((m) => m[1]);

    expect(
      versions.length,
      `no MapLibre version string found in ${dist} — plotly may have stopped ` +
        "inlining MapLibre, which would make this floor untested rather than met",
    ).toBeGreaterThan(0);

    for (const version of new Set(versions)) {
      expect(atLeast(version, "6.4.1"), `inlined MapLibre is ${version}`).toBe(
        true,
      );
    }
  });
});
