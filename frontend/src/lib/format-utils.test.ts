// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for display formatting helpers.
 *
 * Covers ``pluralize``, which exists so a rendered count and its noun agree.
 * The UI previously read "ran 1 iterations" — a count-noun disagreement that
 * three separate call sites reproduced independently.
 */

import { describe, it, expect } from "vitest";

import { pluralize } from "./format-utils";

describe("pluralize", () => {
  it("returns the singular form for exactly 1", () => {
    expect(pluralize(1, "iteration")).toBe("iteration");
  });

  it("returns the plural form for 0 and for counts above 1", () => {
    // Zero takes the plural in English ("0 iterations"), which is why the
    // predicate is `count === 1` rather than `count <= 1`.
    expect(pluralize(0, "iteration")).toBe("iterations");
    expect(pluralize(2, "iteration")).toBe("iterations");
    expect(pluralize(12, "iteration")).toBe("iterations");
  });

  it("honours an explicit plural for nouns that do not take a bare -s", () => {
    expect(pluralize(1, "entry", "entries")).toBe("entry");
    expect(pluralize(3, "entry", "entries")).toBe("entries");
  });
});
