// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main tab container for the project page.
 * Provides six top-level tabs: Summary, Data, Code, Prompts, Evaluations, Reports.
 * Persists active tab selection via URL search parameter (?tab=).
 */

import { useSearchParams } from "react-router-dom";
import {
  LayoutDashboard,
  Database,
  Code2,
  FileText,
  BarChart3,
  FileBarChart,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ProjectConfig } from "@/lib/types";
import { SummaryTab } from "./SummaryTab";
import { DataTab } from "./DataTab";
import { CodeTab } from "./CodeTab";
import { PromptsTab } from "./PromptsTab";
import { ReportsTab } from "./ReportsTab";
import { EvalDashboard } from "@/components/eval/EvalDashboard";

/** Props for the ProjectTabs component. */
interface ProjectTabsProps {
  /** Project UUID for passing to child tabs */
  projectId: string;
  /** Full project configuration for the SummaryTab */
  project: ProjectConfig;
}

/** Valid tab identifiers for URL persistence. */
const VALID_TABS = ["summary", "data", "code", "prompts", "evaluations", "reports"] as const;

/** Default tab shown when no valid tab is specified in the URL. */
const DEFAULT_TAB = "summary";

/**
 * Top-level project tab container with URL-persisted tab selection.
 * Each tab renders its respective content component.
 *
 * @param projectId - Project UUID passed to all child tabs
 * @param project - Full project config passed to SummaryTab
 */
export function ProjectTabs({ projectId, project }: ProjectTabsProps) {
  const [searchParams, setSearchParams] = useSearchParams();

  const rawTab = searchParams.get("tab");
  const activeTab = VALID_TABS.includes(rawTab as typeof VALID_TABS[number])
    ? (rawTab as string)
    : DEFAULT_TAB;

  /** Update the URL search param when the user switches tabs. */
  const handleTabChange = (tab: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set("tab", tab);
      /* Clear dashboard-specific params when leaving the Evaluations tab */
      if (tab !== "evaluations") {
        next.delete("dtab");
        next.delete("run");
      }
      /* Clear the selected batch when leaving the Data tab */
      if (tab !== "data") {
        next.delete("batch");
      }
      /* Clear the report's run selection + type when leaving Reports */
      if (tab !== "reports") {
        next.delete("rrun");
        next.delete("rtype");
      }
      return next;
    }, { replace: true });
  };

  return (
    <Tabs value={activeTab} onValueChange={handleTabChange} className="flex flex-col min-h-0 flex-1">
      <TabsList className="shrink-0">
        <TabsTrigger value="summary">
          <LayoutDashboard className="size-3.5" />
          Summary
        </TabsTrigger>
        <TabsTrigger value="data">
          <Database className="size-3.5" />
          Data
        </TabsTrigger>
        <TabsTrigger value="code">
          <Code2 className="size-3.5" />
          Code
        </TabsTrigger>
        <TabsTrigger value="prompts">
          <FileText className="size-3.5" />
          Prompts
        </TabsTrigger>
        <TabsTrigger value="evaluations">
          <BarChart3 className="size-3.5" />
          Evaluations
        </TabsTrigger>
        <TabsTrigger value="reports">
          <FileBarChart className="size-3.5" />
          Reports
        </TabsTrigger>
      </TabsList>

      <TabsContent value="summary">
        <SummaryTab projectId={projectId} project={project} />
      </TabsContent>

      <TabsContent value="data" className="flex-1 min-h-0 flex flex-col">
        <DataTab projectId={projectId} />
      </TabsContent>

      <TabsContent value="code" className="flex-1 min-h-0 flex flex-col">
        <CodeTab projectId={projectId} />
      </TabsContent>

      <TabsContent value="prompts" className="flex-1 min-h-0 flex flex-col">
        <PromptsTab projectId={projectId} />
      </TabsContent>

      <TabsContent value="evaluations" className="flex-1 min-h-0 flex flex-col">
        <EvalDashboard projectId={projectId} />
      </TabsContent>

      <TabsContent value="reports" className="flex-1 min-h-0 flex flex-col">
        <ReportsTab projectId={projectId} />
      </TabsContent>
    </Tabs>
  );
}
