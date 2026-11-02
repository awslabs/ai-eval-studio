// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @vitest-environment jsdom
 *
 * A 404 on the file preview means the tree is older than the filesystem, not
 * that the file is gone.
 *
 * While an agent sorts uploads, every ``move_file`` relocates a file and the
 * backend announces it on the project stream. A client that misses one of those
 * frames keeps rendering the old paths, and then every click on one returns 404
 * and the pane reads "Unable to load file preview" until the user navigates
 * away and back. These tests assert the recovery EFFECT — the tree is refetched
 * and the same selection is retried — so removing either half fails.
 *
 * The negative case matters as much: a non-404 is a real failure and must be
 * reported, not retried behind the user's back.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { ApiError } from "@/lib/api";
import type { FileContentResponse } from "@/lib/types";

/** Stubbed api surface; ``ApiError`` stays REAL so ``instanceof`` holds. */
const listFiles = vi.fn();
const getFileContent = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: { listFiles: (...a: unknown[]) => listFiles(...a),
           getFileContent: (...a: unknown[]) => getFileContent(...a) },
  };
});

vi.mock("@/hooks/useBackendEvents", () => ({ useProjectEvent: () => undefined }));
vi.mock("@/hooks/useSyntheticBatches", () => ({ useSyntheticBatches: () => ({ data: [] }) }));
vi.mock("@/components/synthesis/SyntheticDataPanel", () => ({
  SyntheticDataPanel: () => <div />,
}));
vi.mock("./FilePreview", () => ({
  FilePreview: ({ file }: { file: FileContentResponse }) => (
    <div data-testid="preview">{file.content}</div>
  ),
}));

/** Minimal FileTree: one button per known path, so a click drives onSelect. */
vi.mock("./FileTree", () => ({
  FileTree: ({ onSelect }: { onSelect: (p: string) => void }) => (
    <button onClick={() => onSelect("data/raw/moved.csv")}>pick</button>
  ),
}));

const TREE = [{ name: "data", type: "dir" as const, path: "data", children: [] }];

const CONTENT: FileContentResponse = {
  content: "a,b\n1,2\n",
  size: 8,
  language: "csv",
} as FileContentResponse;

let DataTab: typeof import("./DataTab").DataTab;

beforeEach(async () => {
  listFiles.mockReset();
  getFileContent.mockReset();
  listFiles.mockResolvedValue({ tree: TREE });
  ({ DataTab } = await import("./DataTab"));
});

afterEach(() => cleanup());

describe("DataTab preview recovery", () => {
  it("refetches the tree and retries the selection once after a 404", async () => {
    getFileContent
      .mockRejectedValueOnce(new ApiError(404, "File not found"))
      .mockResolvedValueOnce(CONTENT);

    render(<DataTab projectId="p1" />);
    await waitFor(() => expect(listFiles).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByText("pick"));

    await waitFor(() => expect(screen.getByTestId("preview")).toBeInTheDocument());
    expect(getFileContent).toHaveBeenCalledTimes(2);
    expect(getFileContent).toHaveBeenNthCalledWith(2, "p1", "data/raw/moved.csv");
    expect(listFiles).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("Unable to load file preview.")).not.toBeInTheDocument();
  });

  it("reports the failure when the retry also 404s", async () => {
    getFileContent.mockRejectedValue(new ApiError(404, "File not found"));

    render(<DataTab projectId="p1" />);
    await waitFor(() => expect(listFiles).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByText("pick"));

    await waitFor(() =>
      expect(screen.getByText("Unable to load file preview.")).toBeInTheDocument(),
    );
    /* Exactly one retry — not a loop. */
    expect(getFileContent).toHaveBeenCalledTimes(2);
  });

  it("does not retry a non-404, and does not refetch the tree", async () => {
    getFileContent.mockRejectedValue(new ApiError(500, "boom"));

    render(<DataTab projectId="p1" />);
    await waitFor(() => expect(listFiles).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByText("pick"));

    await waitFor(() =>
      expect(screen.getByText("Unable to load file preview.")).toBeInTheDocument(),
    );
    expect(getFileContent).toHaveBeenCalledTimes(1);
    expect(listFiles).toHaveBeenCalledTimes(1);
  });
});
