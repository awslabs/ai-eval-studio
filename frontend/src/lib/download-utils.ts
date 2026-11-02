// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared client-side "save file" helper.
 *
 * Every downloadable artifact in the app (eval CSV/JSON exports, the harness
 * zip, the self-contained report HTML) routes through {@link saveBlob} so they
 * all offer the same "choose where to save" experience: the File System Access
 * API picker on Chromium, with a silent anchor-download fallback elsewhere
 * (Firefox/Safari). Previously this logic was duplicated per call site
 * (FilterBar, EvalDashboard) and missing entirely on others (harness, reports).
 */

/**
 * Save arbitrary content to a user-chosen location.
 *
 * On browsers that support the File System Access API (`window.showSaveFilePicker`,
 * i.e. Chromium), shows a native "Save as" dialog seeded with `filename`. The
 * user cancelling the dialog (`AbortError`) is a no-op, not an error. On every
 * other browser, falls back to a standard anchor download (the file lands in the
 * default Downloads folder), preserving the prior behavior.
 *
 * @param data - The content to save. A string is wrapped in a Blob using
 *   `mimeType`; a Blob is written as-is (its own type wins for the picker's
 *   accept filter when `mimeType` is omitted).
 * @param filename - Suggested filename, including extension (e.g. `report.html`).
 * @param mimeType - MIME type. Required when `data` is a string; optional for a
 *   Blob (defaults to the Blob's own `type`, then `application/octet-stream`).
 */
export async function saveBlob(
  data: Blob | string,
  filename: string,
  mimeType?: string,
): Promise<void> {
  const blob =
    typeof data === "string"
      ? new Blob([data], { type: mimeType ?? "application/octet-stream" })
      : data;
  const effectiveMime = mimeType ?? (blob.type || "application/octet-stream");

  // File System Access API (Chromium): real "Save as" dialog.
  if ("showSaveFilePicker" in window) {
    try {
      const ext = filename.split(".").pop() ?? "";
      const handle = await (
        window as unknown as {
          showSaveFilePicker: (opts: unknown) => Promise<{
            createWritable: () => Promise<{
              write: (b: Blob) => Promise<void>;
              close: () => Promise<void>;
            }>;
          }>;
        }
      ).showSaveFilePicker({
        suggestedName: filename,
        types: ext
          ? [{ description: ext.toUpperCase(), accept: { [effectiveMime]: [`.${ext}`] } }]
          : undefined,
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return;
    } catch (e) {
      // User dismissed the picker — treat as a no-op, don't fall through to a
      // silent anchor download (that would defeat the cancel).
      if ((e as { name?: string })?.name === "AbortError") return;
      // Any other failure (e.g. picker unavailable in this context): fall back.
    }
  }

  // Fallback: anchor download to the default Downloads folder.
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
