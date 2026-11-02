// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Prompts tab for viewing and comparing prompt versions.
 *
 * Two modes:
 * - View: Single prompt in Monaco editor (read-only, markdown highlighting)
 * - Compare: Monaco DiffEditor with side-by-side or inline diff view
 *
 * Uses the app's Monaco themes (app-dark / app-light, see lib/monaco-theme)
 * so the editor blends with the page.
 */

import { useCallback, useEffect, useState } from "react";
import { FileText, GitCompare, Columns2, Rows2 } from "lucide-react";
import Editor, { DiffEditor } from "@monaco-editor/react";
import { defineAppTheme, getMonacoTheme } from "@/lib/monaco-theme";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTheme } from "@/lib/theme-provider";
import { api } from "@/lib/api";
import { cn } from "@/lib/class-utils";

/** Props for the PromptsPanel component. */
interface PromptsPanelProps {
  /** Project UUID */
  projectId: string;
  /** Available prompt filenames */
  promptFiles: string[];
  /** Optional className for flex layout passthrough */
  className?: string;
}


/**
 * Prompts tab with view/compare modes.
 * View mode shows a single prompt in Monaco.
 * Compare mode shows Monaco DiffEditor between two versions.
 *
 * @param projectId - Project UUID for fetching prompt content
 * @param promptFiles - List of prompt filenames from the API
 */
export function PromptsPanel({ projectId, promptFiles, className }: PromptsPanelProps) {
  const { mode: themeMode } = useTheme();
  const isDark = themeMode === "dark";

  const [mode, setMode] = useState<"view" | "compare">("view");
  const [selected, setSelected] = useState("");
  const [leftVersion, setLeftVersion] = useState("");
  const [rightVersion, setRightVersion] = useState("");

  const [content, setContent] = useState<string>("");
  const [leftContent, setLeftContent] = useState<string>("");
  const [rightContent, setRightContent] = useState<string>("");
  const [stats, setStats] = useState<{ word_count: number; char_count: number } | null>(null);

  /** Set defaults when promptFiles arrive or change. */
  useEffect(() => {
    if (promptFiles.length === 0) return;
    // promptFiles sorted latest-first by backend (v14, v13, ..., baseline)
    setSelected(promptFiles[0]);
    setRightVersion(promptFiles[0]);
    setLeftVersion(promptFiles.length >= 2 ? promptFiles[1] : promptFiles[0]);
  }, [promptFiles]);

  /** Fetch single prompt content. */
  const fetchPrompt = useCallback(
    (version: string) => {
      if (!version) return;
      api.getPromptContent(projectId, version).then((data) => {
        setContent(data.content);
        setStats({ word_count: data.word_count, char_count: data.char_count });
      }).catch(() => {});
    },
    [projectId],
  );

  /** Fetch both prompts for comparison. */
  const fetchCompare = useCallback(() => {
    if (!leftVersion || !rightVersion) return;
    Promise.all([
      api.getPromptContent(projectId, leftVersion),
      api.getPromptContent(projectId, rightVersion),
    ]).then(([left, right]) => {
      setLeftContent(left.content);
      setRightContent(right.content);
    }).catch(() => {});
  }, [projectId, leftVersion, rightVersion]);

  useEffect(() => {
    if (mode === "view" && selected) fetchPrompt(selected);
  }, [mode, selected, fetchPrompt]);

  useEffect(() => {
    if (mode === "compare") fetchCompare();
  }, [mode, leftVersion, rightVersion, fetchCompare]);

  if (promptFiles.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-8">
        No prompt files yet. Use the chat to create a baseline prompt.
      </p>
    );
  }

  const monacoTheme = getMonacoTheme(isDark);

  return (
    <div className={cn("gap-3 flex flex-col", className)}>
      {/* Mode toggle + selectors */}
      <div className="flex items-center gap-3 shrink-0">
        <Tabs value={mode} onValueChange={(v) => setMode(v as "view" | "compare")}>
          <TabsList>
            <TabsTrigger value="view" disabled={false}>
              <FileText className="size-3 mr-1.5" />
              View
            </TabsTrigger>
            <TabsTrigger value="compare" disabled={promptFiles.length < 2}>
              <GitCompare className="size-3 mr-1.5" />
              Compare
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {mode === "view" ? (
          <Select value={selected} onValueChange={(v) => setSelected(v ?? "")}>
            <SelectTrigger className="w-[200px] h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {promptFiles.map((f) => (
                <SelectItem key={f} value={f} className="text-xs">{f}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <>
            <Select value={leftVersion} onValueChange={(v) => setLeftVersion(v ?? "")}>
              <SelectTrigger className="w-[160px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {promptFiles.map((f) => (
                  <SelectItem key={f} value={f} className="text-xs">{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className="text-xs text-muted-foreground">vs</span>
            <Select value={rightVersion} onValueChange={(v) => setRightVersion(v ?? "")}>
              <SelectTrigger className="w-[160px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {promptFiles.map((f) => (
                  <SelectItem key={f} value={f} className="text-xs">{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </>
        )}

        {mode === "view" && stats && (
          <span className="text-2xs text-muted-foreground tabular-nums ml-auto">
            ~{stats.word_count} words &middot; {stats.char_count.toLocaleString()} chars
          </span>
        )}
      </div>

      {/* View mode — Monaco editor */}
      {mode === "view" && content && (
        <div className="border rounded-md overflow-hidden bg-card flex-1 min-h-[300px]">
          <Editor
            height="100%"
            language="markdown"
            value={content}
            theme={monacoTheme}
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
        </div>
      )}

      {/* Compare mode — Monaco DiffEditor.
       * Kept mounted but hidden when not in compare mode to avoid the
       * "TextModel got disposed before DiffEditorWidget model got reset"
       * error that occurs when DiffEditor unmounts mid-lifecycle. */}
      {mode === "compare" && leftVersion === rightVersion && (
        <p className="text-sm text-muted-foreground py-4">
          Select two different prompt versions to compare.
        </p>
      )}
      <div className={mode === "compare" && leftVersion !== rightVersion ? "flex-1 min-h-0 flex flex-col" : "hidden"}>
        <PromptDiffEditor
          leftName={leftVersion}
          rightName={rightVersion}
          leftText={leftContent}
          rightText={rightContent}
          theme={monacoTheme}
        />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------
 * Monaco DiffEditor wrapper
 * ----------------------------------------------------------------- */

/**
 * Monaco-based diff viewer for comparing two prompt versions.
 * Supports side-by-side and inline diff modes with matching app theme.
 */
function PromptDiffEditor({
  leftName,
  rightName,
  leftText,
  rightText,
  theme,
}: {
  leftName: string;
  rightName: string;
  leftText: string;
  rightText: string;
  theme: string;
}) {
  const [inline, setInline] = useState(false);

  return (
    <div className="gap-2 flex flex-col flex-1 min-h-0">
      <div className="flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3 text-xs">
          <span className="font-medium text-muted-foreground">{leftName}</span>
          <span className="text-muted-foreground/50">vs</span>
          <span className="font-medium text-muted-foreground">{rightName}</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-2xs text-muted-foreground"
          onClick={() => setInline(!inline)}
          title={inline ? "Side-by-side view" : "Inline view"}
        >
          {inline ? <Columns2 className="size-3 mr-1" /> : <Rows2 className="size-3 mr-1" />}
          {inline ? "Side-by-side" : "Inline"}
        </Button>
      </div>
      <div className="border rounded-md overflow-hidden flex-1 min-h-[300px]">
        <DiffEditor
          height="100%"
          language="markdown"
          original={leftText}
          modified={rightText}
          theme={theme}
          options={{
            readOnly: true,
            renderSideBySide: !inline,
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            wordWrap: "on",
            renderOverviewRuler: false,
            originalEditable: false,
            padding: { top: 8 },
          }}
          beforeMount={defineAppTheme}
        />
      </div>
    </div>
  );
}
