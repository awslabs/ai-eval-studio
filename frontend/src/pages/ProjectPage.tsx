// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Individual project page with tab-based layout.
 * Renders a header with project title/description, the ActiveJobBar
 * (optimization + eval controls), and the ProjectTabs container.
 */

import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { EditableText } from "@/components/project/EditableText";
import { ProjectTabs } from "@/components/project/ProjectTabs";
import { ActiveJobBar } from "@/components/eval/ActiveJobBar";
import { OptimizationStateProvider } from "@/components/eval/OptimizationStateProvider";
import type { ProjectConfig } from "@/lib/types";

/**
 * Project page displaying project header, progress banners, and tabbed content.
 * Fetches project config on mount, redirects to home if not found.
 */
export function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [project, setProject] = useState<ProjectConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    // Reset per-project state so a slow load never renders the PREVIOUS
    // project's header, and a stale response cannot overwrite a newer one.
    let cancelled = false;
    setProject(null);
    setLoadError(null);
    setIsLoading(true);
    api
      .getProject(projectId)
      .then((config) => {
        if (!cancelled) setProject(config);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        // Only a real 404 means "this project is gone", and only that should
        // bounce to Home. Redirecting on every failure told the user their
        // project did not exist when the truth was an expired session, a server
        // error or no network — and threw away the URL they would retry from.
        if (err instanceof ApiError && err.status === 404) {
          navigate("/");
          return;
        }
        setLoadError(
          err instanceof ApiError
            ? `Could not load this project (${err.status}). ${err.message}`
            : "Could not load this project. Check your connection and retry.",
        );
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, navigate]);

  /**
   * Patch one or more project fields and adopt the server's response.
   *
   * The PATCH returns the full config, so the header re-renders from what was
   * actually persisted rather than from the optimistic local value — a rejected
   * or coerced edit can't leave the UI asserting something the backend refused.
   *
   * Targets ``project.id``, the record on screen, rather than the id in the URL.
   * The two are the same in normal use; they diverge for as long as a route
   * change is in flight, and an edit committed then would have been applied to
   * the newly selected project while showing the old one's text.
   *
   * @param updates - Fields to change; omitted fields are left alone
   */
  async function saveField(updates: {
    name?: string;
    description?: string;
  }): Promise<void> {
    if (!project) return;
    setProject(await api.updateProject(project.id, updates));
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 p-8 text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        <span>Loading project...</span>
      </div>
    );
  }
  if (loadError) {
    return (
      <div className="p-8 flex flex-col items-start gap-3">
        <p className="text-sm text-destructive">{loadError}</p>
        <button
          type="button"
          onClick={() => navigate(0)}
          className="text-sm underline text-muted-foreground hover:text-foreground"
        >
          Retry
        </button>
      </div>
    );
  }
  if (!project) return null;

  return (
    <OptimizationStateProvider projectId={project.id}>
      <div className="p-8 flex flex-col gap-4 h-full">
        {/* Header — both fields editable in place. This is the only place either
            can be changed; the project list deliberately has no edit affordance,
            so renaming happens in the project you are looking at. */}
        <div>
          <EditableText
            value={project.name}
            label="Name"
            allowEmpty={false}
            onSave={(next) => saveField({ name: next })}
            className="text-2xl font-bold tracking-tight"
          />
          <div className="mt-1">
            <EditableText
              value={project.description}
              label="Description"
              placeholder="Add a description"
              onSave={(next) => saveField({ description: next })}
              className="text-muted-foreground text-sm"
            />
          </div>
        </div>

        {/* Active job bar — optimization controls + eval progress */}
        <ActiveJobBar projectId={project.id} />

        {/* Tabbed content */}
        <ProjectTabs projectId={project.id} project={project} />
      </div>
    </OptimizationStateProvider>
  );
}
