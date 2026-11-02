// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Home page displaying the list of all evaluation projects.
 * Shows enriched project cards with metadata badges and a proper
 * create-project dialog (replacing the old browser prompt).
 */

import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, FlaskConical, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { clickableCardClass, onActivateKeyDown } from "@/components/ui/interactive";
import { cn } from "@/lib/class-utils";
import { api, ApiError } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";

/**
 * Format an ISO timestamp as a human-readable relative time string.
 * Returns strings like "just now", "5m ago", "3d ago", etc.
 *
 * @param iso - ISO 8601 timestamp string
 * @returns Human-readable relative time
 */
function timeAgo(iso: string): string {
  const now = Date.now();
  const then = new Date(iso).getTime();
  const seconds = Math.floor((now - then) / 1000);

  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

/**
 * Home page component with enriched project cards and create dialog.
 * Fetches projects on mount and displays them as clickable cards with
 * metadata badges (models, pass rate, status).
 */
export function HomePage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  // Project pending deletion (drives the confirmation dialog), or null.
  const [pendingDelete, setPendingDelete] = useState<ProjectSummary | null>(null);

  /** Fetch the project list from the backend, replacing the local copy. */
  const refreshProjects = useCallback(() => {
    return api
      .listProjects()
      .then(setProjects)
      .catch(() => {
        /* Transient list-fetch failures keep the current view; the next
         * focus/visibility refresh retries. Don't blank the list. */
      });
  }, []);

  // Initial load on mount.
  useEffect(() => {
    refreshProjects().finally(() => setIsLoading(false));
  }, [refreshProjects]);

  /* Re-fetch when the tab regains focus or becomes visible so a project
   * deleted in another tab / a prior session / an interrupted attempt
   * disappears on its own — no stale card that 404s on click, no manual
   * hard-refresh. */
  useEffect(() => {
    const refetch = () => {
      if (document.visibilityState === "visible") refreshProjects();
    };
    window.addEventListener("focus", refetch);
    document.addEventListener("visibilitychange", refetch);
    return () => {
      window.removeEventListener("focus", refetch);
      document.removeEventListener("visibilitychange", refetch);
    };
  }, [refreshProjects]);

  /* Poll while at least one project is running, and not otherwise.
   *
   * Polling is the exception here — everything else in the app learns about
   * change over the WebSocket. It does not work for this view: the stream is
   * subscribed per project (`project:<id>`), and this page is a list with no
   * project context, so there is no stream carrying "project 3 finished".
   *
   * Gated on a running project so an idle list costs nothing, and on tab
   * visibility so a backgrounded tab stops. The focus listener above already
   * covers the return-to-tab case, so nothing is missed while hidden. */
  const anyRunning = projects.some((p) => p.is_running);
  useEffect(() => {
    if (!anyRunning) return;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") refreshProjects();
    }, 30_000);
    return () => clearInterval(id);
  }, [anyRunning, refreshProjects]);

  /**
   * Create a new project from the dialog form values.
   * Navigates to the new project page on success.
   */
  const [createError, setCreateError] = useState<string | null>(null);

  const handleCreate = async () => {
    const trimmedName = newName.trim();
    if (!trimmedName) return;
    setIsCreating(true);
    setCreateError(null);
    try {
      const project = await api.createProject(trimmedName, newDescription.trim());
      navigate(`/projects/${project.id}`);
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : "Failed to create project");
    } finally {
      setIsCreating(false);
    }
  };

  /** Open the create dialog and reset form fields. */
  const openDialog = () => {
    setNewName("");
    setNewDescription("");
    setDialogOpen(true);
  };

  /**
   * Permanently delete a project after the confirmation dialog confirms.
   * Removes it from the list on success and closes the dialog. A 404 is
   * treated as success — the project is already gone (deleted in another
   * tab / a prior session / an earlier interrupted attempt), which is the
   * desired end state. Real errors propagate to the dialog's error path.
   */
  const handleDeleteConfirmed = async (project: ProjectSummary) => {
    try {
      await api.deleteProject(project.id);
    } catch (e) {
      // A 404 here still means the delete succeeded from the user's point
      // of view — they asked to delete it and it's gone. (It can surface
      // if the project was already removed in another tab/session.) Treat
      // it as success with the normal "Deleted" wording — "already removed"
      // wrongly implies someone else deleted it. Real errors propagate to
      // the dialog's error path.
      if (!(e instanceof ApiError && e.status === 404)) {
        throw e;
      }
    }
    setProjects((prev) => prev.filter((p) => p.id !== project.id));
    setPendingDelete(null);
    toast.success(`Deleted "${project.name}"`);
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header with title and create button */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Create and manage your evaluation projects
          </p>
        </div>
        <Button onClick={openDialog}>
          <Plus className="size-4 mr-2" />
          New Project
        </Button>
      </div>

      {/* Project list with loading and empty states */}
      {isLoading ? (
        <p className="text-muted-foreground">Loading...</p>
      ) : projects.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <FlaskConical className="size-10 text-muted-foreground/40 mb-4" />
          <p className="text-muted-foreground mb-4">
            No projects yet. Create one to get started.
          </p>
          <Button onClick={openDialog}>
            <Plus className="size-4 mr-2" />
            Create your first project
          </Button>
        </div>
      ) : (
        <div className="grid gap-3">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onClick={() => navigate(`/projects/${project.id}`)}
              onRequestDelete={() => setPendingDelete(project)}
            />
          ))}
        </div>
      )}

      {/* Create project dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>New Project</DialogTitle>
            <DialogDescription>
              Create a new evaluation project. You can configure models and
              metrics later.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label htmlFor="project-name" className="text-sm font-medium">
                Name <span className="text-error">*</span>
              </label>
              <Input
                id="project-name"
                placeholder="e.g., RAG Pipeline Evaluation"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && newName.trim()) handleCreate();
                }}
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <label
                htmlFor="project-description"
                className="text-sm font-medium"
              >
                Description
              </label>
              <Input
                id="project-description"
                placeholder="Optional description..."
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
              />
            </div>
          </div>
          {createError && (
            <p className="text-sm text-destructive">{createError}</p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDialogOpen(false)}
              disabled={isCreating}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!newName.trim() || isCreating}
            >
              {isCreating ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation dialog (type-the-name to confirm) */}
      <DeleteProjectDialog
        project={pendingDelete}
        onCancel={() => setPendingDelete(null)}
        onConfirm={handleDeleteConfirmed}
      />
    </div>
  );
}

/**
 * Permanent-delete confirmation dialog. Requires the user to type the
 * exact project name before the destructive action enables — a hard guard
 * against accidental deletion of an unrecoverable resource.
 *
 * @param project - The project pending deletion, or null when closed.
 * @param onCancel - Close the dialog without deleting.
 * @param onConfirm - Perform the delete; may reject (surfaced as an error).
 */
function DeleteProjectDialog({
  project,
  onCancel,
  onConfirm,
}: {
  project: ProjectSummary | null;
  onCancel: () => void;
  onConfirm: (project: ProjectSummary) => Promise<void>;
}) {
  const [confirmText, setConfirmText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset transient state whenever the target project changes.
  useEffect(() => {
    setConfirmText("");
    setError(null);
    setIsDeleting(false);
  }, [project]);

  const nameMatches = project != null && confirmText === project.name;

  const handleConfirm = async () => {
    if (!project || !nameMatches) return;
    setIsDeleting(true);
    setError(null);
    try {
      await onConfirm(project);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete project");
      setIsDeleting(false);
    }
  };

  return (
    <Dialog
      open={project != null}
      // While a delete is in flight, swallow every dismissal (Esc,
      // click-outside, X — Base UI routes all three through here) so the
      // user can't close the dialog and lose the in-progress / failure
      // feedback. Only an explicit Cancel (disabled mid-delete) or a
      // completed delete closes it.
      onOpenChange={(open) => {
        if (!open && !isDeleting) onCancel();
      }}
    >
      <DialogContent
        className="sm:max-w-md"
        showCloseButton={!isDeleting}
      >
        <DialogHeader>
          <DialogTitle>Delete project</DialogTitle>
          <DialogDescription>
            This permanently deletes{" "}
            <span className="font-medium text-foreground">{project?.name}</span>{" "}
            and everything in it — data, prompts, eval runs, the harness, and
            chat history. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2 py-2">
          <label htmlFor="confirm-name" className="text-sm text-muted-foreground">
            Type{" "}
            <span className="font-medium text-foreground">{project?.name}</span>{" "}
            to confirm:
          </label>
          <Input
            id="confirm-name"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && nameMatches) handleConfirm();
            }}
            disabled={isDeleting}
            autoFocus
            autoComplete="off"
          />
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            disabled={!nameMatches || isDeleting}
          >
            {isDeleting ? (
              <>
                <Loader2 className="size-4 mr-2 animate-spin" />
                Deleting...
              </>
            ) : (
              "Delete permanently"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Project card: name, optional description, and one metadata line.
 *
 * The metadata sits on its own line under the name rather than at the far right
 * edge of a full-width card, where the eye had to cross the whole viewport to
 * connect a pass rate to the project it belonged to.
 *
 * Description renders only when there is one. The old "No description"
 * placeholder occupied a line on every card while stating nothing — the place to
 * ADD a description is the project header, so that is where the placeholder
 * lives now.
 *
 * @param project - Project summary with enriched metadata
 * @param onClick - Click handler for navigation
 */
function ProjectCard({
  project,
  onClick,
  onRequestDelete,
}: {
  project: ProjectSummary;
  onClick: () => void;
  onRequestDelete: () => void;
}) {
  const passRate = project.latest_pass_rate;
  const modelCount = project.model_count ?? 0;
  const activityTime = project.last_activity ?? project.created_at;

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={onActivateKeyDown(onClick)}
      className={cn("group hover:bg-accent/50", clickableCardClass)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <CardTitle className="text-base font-semibold">
              {project.name}
            </CardTitle>
            {project.description && (
              <CardDescription className="mt-1 line-clamp-2">
                {project.description}
              </CardDescription>
            )}

            {/* Metadata line: what is happening, then what was measured. */}
            <div className="flex items-center gap-2 flex-wrap mt-2">
              {project.is_running && (
                <Badge variant="info" className="text-2xs font-medium">
                  <Loader2 className="size-3 animate-spin" />
                  Running
                </Badge>
              )}
              {modelCount > 0 && (
                <Badge variant="secondary" className="text-2xs font-normal">
                  {modelCount} {modelCount === 1 ? "model" : "models"}
                </Badge>
              )}
              {passRate != null && <PassRateBadge rate={passRate} />}
              <span className="text-2xs text-muted-foreground/70 tabular-nums">
                {timeAgo(activityTime)}
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={(e) => {
              // Don't navigate into the project when deleting it.
              e.stopPropagation();
              onRequestDelete();
            }}
            className="shrink-0 p-1.5 rounded-md text-muted-foreground/50 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 hover:text-destructive hover:bg-destructive/10 transition-all outline-none focus-visible:ring-2 focus-visible:ring-ring"
            title="Delete project"
            aria-label={`Delete ${project.name}`}
          >
            <Trash2 className="size-4" />
          </button>
        </div>
      </CardHeader>
    </Card>
  );
}

/**
 * Color-coded pass rate badge.
 * Green for >= 70%, yellow/warning for >= 40%, red/error for < 40%.
 *
 * @param rate - Pass rate as a decimal (0.0-1.0)
 */
function PassRateBadge({ rate }: { rate: number }) {
  const pct = Math.round(rate * 100);
  const variant = rate >= 0.7 ? "success" : rate >= 0.4 ? "warning" : "error";

  return (
    <Badge variant={variant} className="text-2xs font-medium">
      {pct}% pass
    </Badge>
  );
}
