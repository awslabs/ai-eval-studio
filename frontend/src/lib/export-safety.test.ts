// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CSV field escaping and URL path encoding — both fed by user-controlled data.
 *
 * Dataset values reach the CSV export, and dataset-derived filenames reach the
 * file-preview URL, so neither can assume well-behaved input.
 */
import { describe, it, expect } from "vitest";

import { csvEscape } from "@/components/eval/FilterBar";
import { encodePathSegments } from "@/lib/api";

describe("csvEscape", () => {
  it("leaves an ordinary value untouched", () => {
    expect(csvEscape("hello")).toBe("hello");
  });

  it("quotes a value containing a comma so it stays in one cell", () => {
    // Unquoted, this shifted every later column of the row.
    expect(csvEscape("Paris, France")).toBe('"Paris, France"');
  });

  it("doubles embedded quotes", () => {
    expect(csvEscape('say "hi"')).toBe('"say ""hi"""');
  });

  it("quotes newlines and carriage returns", () => {
    expect(csvEscape("a\nb")).toBe('"a\nb"');
    expect(csvEscape("a\r\nb")).toBe('"a\r\nb"');
  });

  it("quotes a tab, which splits a cell in Excel", () => {
    // Tab was absent from the original pattern.
    expect(csvEscape("a\tb")).toContain('"');
  });

  // The apostrophe goes on the FIELD VALUE. When that value also needs RFC 4180
  // quoting the quote wraps it, so the serialized form is `"'=..."` — the cell a
  // spreadsheet reads still begins with the apostrophe, which is what disarms it.
  it.each(["=", "+", "-", "@"])(
    "neutralises a cell starting with %s so a spreadsheet does not evaluate it",
    (lead) => {
      const bare = csvEscape(`${lead}SUM(A1)`);
      expect(bare).toBe(`'${lead}SUM(A1)`);

      const needsQuoting = csvEscape(`${lead}HYPERLINK("http://x","click")`);
      expect(needsQuoting.startsWith(`"'${lead}`)).toBe(true);
    },
  );

  it("neutralises the classic formula-injection payload", () => {
    expect(csvEscape("=cmd|'/c calc'!A0")).toBe("'=cmd|'/c calc'!A0");
  });

  it("does not treat a mid-value operator as a formula", () => {
    expect(csvEscape("2+2=4")).toBe("2+2=4");
  });

  it("handles an empty value", () => {
    expect(csvEscape("")).toBe("");
  });
});

describe("encodePathSegments", () => {
  it("keeps separators and encodes each segment", () => {
    expect(encodePathSegments("data/raw/seed.csv")).toBe("data/raw/seed.csv");
  });

  it("encodes a hash, which would otherwise be read as a fragment", () => {
    // Raw, the request truncated to the parent directory and never sent the name.
    expect(encodePathSegments("data/a#b.csv")).toBe("data/a%23b.csv");
  });

  it("encodes a question mark, which would otherwise start a query string", () => {
    expect(encodePathSegments("data/a?b.csv")).toBe("data/a%3Fb.csv");
  });

  it("encodes a literal percent so the server does not decode it twice", () => {
    // Passed through raw, `a%41b.csv` was unquoted server-side into `aAb.csv`.
    expect(encodePathSegments("data/a%41b.csv")).toBe("data/a%2541b.csv");
  });

  it("encodes spaces", () => {
    expect(encodePathSegments("data/my file.csv")).toBe("data/my%20file.csv");
  });

  it("does not escape the separators themselves", () => {
    expect(encodePathSegments("a/b/c")).not.toContain("%2F");
  });
});
