// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Code tab with Monaco editor and resizable file tree sidebar.
 * Allows browsing and viewing project source files (read-only), and
 * downloading the whole runnable harness as a zip. Harness changes go
 * through chat — direct source editing is disabled (the agent edits with
 * knowledge of the field-mapping / metric / data-loader contracts and
 * re-runs validation).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { Loader2, Download, ChevronRight } from "lucide-react";
import { defineAppTheme, getMonacoTheme } from "@/lib/monaco-theme";
import { monacoLanguageFor } from "@/lib/monaco-language";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useTheme } from "@/lib/theme-provider";
import { useProjectEvent } from "@/hooks/useBackendEvents";
import { FileTree } from "./FileTree";
import { ResizeHandle } from "./ResizeHandle";
import type { FileNode, FileContentResponse } from "@/lib/types";

/** Props for the CodeTab component. */
interface CodeTabProps {
  /** Project UUID for file API calls */
  projectId: string;
}

/** Min/max sidebar width in pixels. */
const SIDEBAR_MIN = 160;
const SIDEBAR_MAX = 400;
const SIDEBAR_DEFAULT = 260;

/**
 * Read-only code viewer tab with a resizable file tree sidebar and Monaco
 * editor. Fetches the project file tree on mount, loads file content on
 * selection. Direct source editing is intentionally disabled (the harness
 * source has agent-maintained contracts; all edits go through chat); the
 * "Download harness" toolbar action is the user-side write path.
 *
 * @param projectId - Project UUID for API calls
 */
export function CodeTab({ projectId }: CodeTabProps) {
  const [tree, setTree] = useState<FileNode[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [file, setFile] = useState<FileContentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [treeLoading, setTreeLoading] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);

  /** Detect dark mode for Monaco theme selection. */
  const { mode } = useTheme();
  const isDark = mode === "dark";

  /** Refetch the file tree — reused on mount and on FS-change/turn-complete events. */
  const refreshTree = useCallback(() => {
    setTreeLoading(true);
    api.listFiles(projectId)
      .then(({ tree: nodes }) => setTree(nodes))
      .catch(() => setTree([]))
      .finally(() => setTreeLoading(false));
  }, [projectId]);

  /** Fetch the file tree on mount. */
  useEffect(() => {
    refreshTree();
  }, [refreshTree]);

  /* Refresh tree when the backend reports a filesystem mutation or an
   * agent turn completes. The bus is already project-scoped via the
   * project WebSocket stream that ``useBackendEvents`` subscribes to. */
  useProjectEvent({ resource: "filesystem", type: "changed" }, refreshTree, "state");
  useProjectEvent({ resource: "agent_turn", type: "completed" }, refreshTree, "state");

  /** Load file content when a file is selected.
   *
   * ``latestRequest`` discards a stale response. Two quick clicks resolved in
   * arrival order, so a slow earlier file could overwrite the newer selection —
   * and the first response's ``finally`` also cleared the spinner while the
   * second request was still in flight. */
  const latestRequest = useRef(0);
  const handleSelect = useCallback(
    async (path: string) => {
      const requestId = ++latestRequest.current;
      setSelectedPath(path);
      setIsLoading(true);
      try {
        const content = await api.getFileContent(projectId, path);
        if (requestId !== latestRequest.current) return;
        setFile(content);
      } catch {
        if (requestId !== latestRequest.current) return;
        toast.error("Failed to load file", { duration: Infinity });
        setFile(null);
      } finally {
        if (requestId === latestRequest.current) setIsLoading(false);
      }
    },
    [projectId],
  );

  /** Download the whole runnable harness package as a zip. */
  const handleDownload = useCallback(async () => {
    setIsDownloading(true);
    try {
      await api.downloadHarness(projectId);
    } catch {
      toast.error("Failed to download harness", { duration: Infinity });
    } finally {
      setIsDownloading(false);
    }
  }, [projectId]);


  /** Build breadcrumb segments from the file path. */
  const breadcrumbs = file?.path.split("/") ?? [];

  return (
    <div className="flex border rounded-md overflow-hidden flex-1 min-h-0">
      {/* Resizable file tree sidebar */}
      <div
        className="shrink-0 border-r overflow-y-auto bg-muted/30 py-2"
        style={{ width: sidebarWidth }}
      >
        {treeLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        ) : tree.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-8">
            No files yet
          </p>
        ) : (
          <FileTree nodes={tree} selectedPath={selectedPath} onSelect={handleSelect} />
        )}
      </div>

      {/* Drag handle */}
      <ResizeHandle
        onResize={(delta) => {
          setSidebarWidth((w) => Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, w + delta)));
        }}
      />

      {/* Main editor area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Toolbar — breadcrumb (when a file is open) + download + read-only hint */}
        <div className="flex items-center justify-between border-b px-3 py-1.5 bg-muted/20 shrink-0 gap-3">
          {/* Breadcrumb */}
          <div className="flex items-center gap-1 text-xs text-muted-foreground overflow-hidden">
            {breadcrumbs.map((segment, i) => (
              <span key={i} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="size-3 shrink-0" />}
                <span className={i === breadcrumbs.length - 1 ? "text-foreground font-medium" : ""}>
                  {segment}
                </span>
              </span>
            ))}
          </div>

          {/* Read-only hint + download harness */}
          <div className="flex items-center gap-3 shrink-0">
            <span className="text-2xs text-muted-foreground/70 hidden sm:inline">
              To change the harness, ask in chat.
            </span>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs gap-1.5"
              onClick={handleDownload}
              disabled={isDownloading}
              title="Download the full runnable harness as a zip"
            >
              {isDownloading ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              Download harness
            </Button>
          </div>
        </div>

        {/* Editor or placeholder — flex-1 fills remaining height, Monaco scrolls internally */}
        <div className="flex-1 min-h-0 bg-card">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="size-5 animate-spin text-muted-foreground" />
            </div>
          ) : file ? (
            <Editor
              height="100%"
              language={monacoLanguageFor(file.language)}
              value={file.content}
              theme={getMonacoTheme(isDark)}
              options={{
                readOnly: true,
                minimap: { enabled: false },
                fontSize: 13,
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                wordWrap: "on",
                padding: { top: 8 },
              }}
              beforeMount={defineAppTheme}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
              Select a file to view
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

