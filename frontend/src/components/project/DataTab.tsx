// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Data tab for the project page.
 * Full-width file upload zone with drag-and-drop support, a project
 * file browser for data/ and docs/ directories, and an inline
 * synthetic data review section (shown only when batches exist).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { api, ApiError } from "@/lib/api";
import { FileTree } from "./FileTree";
import { FilePreview } from "./FilePreview";
import { ResizeHandle } from "./ResizeHandle";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import { SyntheticDataPanel } from "@/components/synthesis/SyntheticDataPanel";
import { useSyntheticBatches } from "@/hooks/useSyntheticBatches";
import type { FileNode, FileContentResponse } from "@/lib/types";

/** Min/max/default sidebar width in pixels for the file tree. */
const SIDEBAR_MIN = 160;
const SIDEBAR_MAX = 400;
const SIDEBAR_DEFAULT = 240;

/** Props for the DataTab component. */
interface DataTabProps {
  /** Project UUID for upload API calls */
  projectId: string;
}

/**
 * Data management tab with drag-and-drop upload, a project file browser,
 * and inline synthetic data review.
 *
 * @param projectId - Project UUID for the upload API call
 */
export function DataTab({ projectId }: DataTabProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* File browser state */
  const [dataNodes, setDataNodes] = useState<FileNode[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<FileContentResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);

  /* Synthetic batches — only render the section when there's data */
  const { data: syntheticBatches } = useSyntheticBatches(projectId);
  const hasSyntheticBatches = (syntheticBatches ?? []).length > 0;

  /** Refetch file tree (reused on mount, after upload, after agent turn). */
  const refreshTree = useCallback(() => {
    api.listFiles(projectId)
      .then(({ tree }) => {
        const filtered = tree.filter(
          (node) => node.type === "dir" && (node.name === "data" || node.name === "docs"),
        );
        setDataNodes(filtered);
      })
      .catch(() => setDataNodes([]));
  }, [projectId]);

  /** Refresh tree on mount. */
  useEffect(() => {
    refreshTree();
  }, [refreshTree]);

  /* Refresh the tree whenever the backend reports a filesystem change
   * (an FS-mutating tool ran) or when an agent turn completes (catches
   * any mutation the whitelist missed). The project bus is already
   * scoped to this project by its WebSocket stream subscription, so no
   * further projectId filtering is needed. */
  useProjectEvent({ resource: "filesystem", type: "changed" }, refreshTree, "state");
  useProjectEvent({ resource: "agent_turn", type: "completed" }, refreshTree, "state");

  /** Load file content when a file is selected in the browser.
   *
   * Guarded against a stale response: clicking a second file before the first
   * request returns left whichever resolved LAST in the preview, so a slow
   * earlier file could overwrite the newer selection and the pane would show
   * content belonging to a path the tree no longer highlights.
   *
   * A 404 is treated as a stale tree rather than a missing file, and costs one
   * resync instead of an error. While an agent is sorting uploads, every
   * ``move_file`` changes where a file lives; the backend announces each one on
   * the project stream, but a client that misses a single frame keeps rendering
   * paths that have already moved, and every click on one 404s. Refetching the
   * tree and retrying once recovers both the preview and the tree the click
   * came from. Only 404 retries — any other status is a real failure and is
   * reported immediately rather than being retried behind the user's back. */
  useEffect(() => {
    if (!selectedPath) {
      setPreviewFile(null);
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);

    const load = async (): Promise<void> => {
      try {
        const content = await api.getFileContent(projectId, selectedPath);
        if (!cancelled) setPreviewFile(content);
        return;
      } catch (err) {
        if (cancelled) return;
        if (!(err instanceof ApiError) || err.status !== 404) {
          setPreviewFile(null);
          return;
        }
      }
      /* Stale-tree path: resync, then give the same selection one more try. */
      refreshTree();
      try {
        const content = await api.getFileContent(projectId, selectedPath);
        if (!cancelled) setPreviewFile(content);
      } catch {
        if (!cancelled) setPreviewFile(null);
      }
    };

    void load().finally(() => {
      if (!cancelled) setPreviewLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [projectId, selectedPath, refreshTree]);

  /** Upload one or more files; the new contents show up in the tree below. */
  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setError(null);
      setIsUploading(true);
      try {
        await api.uploadFiles(projectId, files);
        refreshTree();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setIsUploading(false);
      }
    },
    [projectId, refreshTree],
  );

  /** Handle files selected via the file input. */
  const onFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (files.length > 0) handleUpload(files);
      e.target.value = "";
    },
    [handleUpload],
  );

  /** Handle files dropped onto the drop zone. */
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) handleUpload(files);
    },
    [handleUpload],
  );

  /** Mark drag-over state for visual feedback. */
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  /** Clear drag-over state when cursor leaves drop zone. */
  const onDragLeave = useCallback(() => setIsDragOver(false), []);

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-6">
      {/* Upload section */}
      <div className="shrink-0">
        <h2 className="text-sm font-medium mb-1">Upload Files</h2>
        <p className="text-xs text-muted-foreground mb-3">
          Drop any mix of data, docs, or zip bundles. The AI agent will sort them into the right place.
        </p>
      </div>

      {/* Drop zone */}
      <Card
        className={`border-2 border-dashed p-4 text-center cursor-pointer transition-colors ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-border/60 hover:border-border hover:bg-accent/30"
        }`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".csv,.json,.jsonl,.xlsx,.xls,.zip,.md,.pdf,.txt"
          className="hidden"
          onChange={onFileSelect}
        />
        {isUploading ? (
          <div className="flex items-center justify-center gap-2 text-muted-foreground">
            <Loader2 className="size-5 animate-spin" />
            <span className="text-sm">Uploading...</span>
          </div>
        ) : (
          <>
            <Upload className="size-6 mx-auto text-muted-foreground/60 mb-2" />
            <p className="text-sm text-muted-foreground">
              Drop files here or click to browse (CSV, JSON, Excel, Markdown, PDF, or .zip)
            </p>
          </>
        )}
      </Card>

      {/* Error display */}
      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      {/* Project files browser */}
      {dataNodes.length > 0 && (
        <div className="flex flex-col flex-1 min-h-0">
          <h3 className="text-sm font-medium mb-2 shrink-0">Project Files</h3>
          <div className="flex flex-1 min-h-0 border rounded-md overflow-hidden">
            {/* Resizable file tree sidebar */}
            <div
              className="shrink-0 overflow-y-auto bg-muted/30 py-2"
              style={{ width: sidebarWidth }}
            >
              <FileTree
                nodes={dataNodes}
                selectedPath={selectedPath}
                onSelect={setSelectedPath}
                expandedPaths={["data/uploads"]}
              />
            </div>
            <ResizeHandle
              onResize={(delta) =>
                setSidebarWidth((w) => Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, w + delta)))
              }
            />
            <div className="flex-1 min-w-0 min-h-0 flex flex-col p-3">
              {previewLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="size-4 animate-spin text-muted-foreground" />
                </div>
              ) : previewFile ? (
                <FilePreview file={previewFile} />
              ) : selectedPath ? (
                <p className="text-sm text-muted-foreground py-4">
                  Unable to load file preview.
                </p>
              ) : (
                <p className="text-sm text-muted-foreground py-4">
                  Select a file to preview its contents.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Synthetic data review — only shown when batches exist */}
      {hasSyntheticBatches && (
        <div>
          <h3 className="text-sm font-medium mb-3">Synthetic Data</h3>
          <SyntheticDataPanel projectId={projectId} />
        </div>
      )}
    </div>
  );
}
