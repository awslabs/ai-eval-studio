# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""System prompt assembly for the coordinator agent.

Combines the coordinator prompt, shared rules, and dynamic project context
(files, optimization state, active jobs, harness config) into a single
system prompt string.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.models.schemas import ProjectConfig
from app.prompts import PROMPTS_DIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.optimization_state import OptimizationState
from app.services.eval_run_files import eval_run_files
from app.services.harness_config import load_harness_config

logger = logging.getLogger(__name__)


def _read_prompt(filename: str) -> str:
    """Read a prompt markdown file from the prompts directory."""
    return (PROMPTS_DIR / filename).read_text()


def build_coordinator_prompt(
    project: ProjectConfig,
    project_dir: Path,
    opt_state: "OptimizationState | None",
) -> str:
    """Build the coordinator's system prompt with project context injected.

    Combines coordinator.md + _rules.md + dynamic project context including
    project ID, uploaded files, prompt versions, optimization state, and
    harness configuration.

    Args:
        project: The project configuration.
        project_dir: The resolved project directory, as returned by
            ``resolve_project_dir``. Cloud-mode-safe: it already carries the
            owner-prefixed ``users/{sub}/projects/{pid}/`` path. Callers pass
            the resolved dir, never the projects root.
        opt_state: The project's authoritative OptimizationState, or ``None``
                   when the project has none. REQUIRED — there is no default,
                   because an omitted argument previously fell back to a sync EFS
                   read, which is the wrong home on the durable path and made the
                   agent report stale optimization status to the user. Callers
                   read it via ``optimization_state_repository`` in their async
                   scope and pass it in.

    Returns:
        str: Coordinator prompt with project context.
    """
    sections = [
        _read_prompt("coordinator.md"),
        _read_prompt("_rules.md"),
        # DeepEval metric reference EMBEDDED. coordinator.md used to
        # instruct the agent to Read this file by repo-relative path — a path
        # that exists in a dev checkout but nowhere the agent actually runs:
        # in the AgentCore microVM the cwd is the project working tree and
        # the packaged copy lives under the app install root, so the lookup
        # failed on every metric proposal and the agent fell back to
        # training memory of DeepEval's API (the exact drift the reference
        # exists to prevent). It is a 5KB
        # packaged resource verified at startup — embedding costs a few
        # cached tokens and removes the filesystem dependence in every
        # execution home.
        "# DeepEval metric reference\n\n" + _read_prompt("deepeval-metric-reference.md"),
        _build_project_context(project, project_dir, opt_state),
    ]
    return "\n\n".join(sections)


def _build_project_context(
    project: ProjectConfig,
    project_dir: Path,
    opt_state: "OptimizationState | None",
) -> str:
    """Build the dynamic project context section.

    Includes project ID, name, uploaded data files, prompt versions,
    eval run count, optimization state, active eval jobs, and harness
    config summary.

    Args:
        project: The project configuration.
        project_dir: Path to the project directory.
        opt_state: The project's authoritative OptimizationState, or ``None``
                   when the project has none. REQUIRED (see
                   :func:`build_coordinator_prompt`).

    Returns:
        str: Formatted project context block.
    """
    # List uploaded data files
    data_dir = project_dir / "data" / "raw"
    data_files = []
    if data_dir.exists():
        data_files = sorted(f.name for f in data_dir.iterdir() if f.is_file())

    # List prompt versions
    prompts_dir = project_dir / "prompts"
    prompt_files = []
    if prompts_dir.exists():
        prompt_files = sorted(f.name for f in prompts_dir.iterdir() if f.is_file())

    # List eval runs. Runs only — listing every JSON in ``output/`` also handed
    # the agent the metric-audit sidecars and the active-eval marker, which read
    # as extra runs it could load results from.
    eval_runs = [p.name for p in eval_run_files(project_dir / "output")]

    # Project ID and name look superficially similar (both short
    # strings) and the agent can confuse them — filling
    # ``project_id="my-demo-project"`` (a name-derived slug) into
    # MCP tool args, which mcp_authz rightly 404s. The backticks
    # around the UUID + the explicit "use this exact value" /
    # "never pass to tools" annotations are a structural fix:
    # backtick-fenced opaque tokens read as machine identifiers in a
    # way the model rarely confuses with prose.
    context_lines = [
        "## Current Project Context",
        f"- **Project ID** (use this exact value for every `project_id` tool argument): `{project.id}`",
        f"- **Project Display Name** (label only; never pass to tools): {project.name}",
    ]

    if project.description:
        context_lines.append(f"- **Description:** {project.description}")

    if data_files:
        context_lines.append(f"- **Uploaded Data Files:** {', '.join(data_files)}")
    else:
        context_lines.append("- **Uploaded Data Files:** none yet")

    if prompt_files:
        context_lines.append(
            f"- **Prompt Versions:** {', '.join(prompt_files)}"
        )

    if eval_runs:
        context_lines.append(
            f"- **Eval Runs:** {len(eval_runs)} completed"
        )

    # Synthetic batches — summarise only if the folder exists so the
    # context stays clean for projects that haven't touched synthesis.
    synthetic_dir = project_dir / "data" / "synthetic"
    if synthetic_dir.exists():
        batch_count = 0
        pending_total = 0
        approved_total = 0
        for manifest_path in synthetic_dir.glob("*.manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            batch_count += 1
            for c in manifest.get("cases") or []:
                status = c.get("status", "pending")
                if status == "pending":
                    pending_total += 1
                elif status == "approved":
                    approved_total += 1
        if batch_count:
            context_lines.append(
                f"- **Synthetic Batches:** {batch_count} "
                f"({pending_total} pending, {approved_total} approved)"
            )

    # Optimization state is PASSED IN, never read here.
    #
    # This function is sync while its callers run in async scope, so reading
    # state here could only ever be the sync EFS read — which on the durable path
    # is the WRONG HOME. The agent would then report a stale or absent
    # optimization status to the user.
    #
    # There is deliberately no default: an omitted argument used to fall back to
    # EFS silently. Callers resolve the authoritative state through
    # optimization_state_repository in their async scope and pass it in.
    # ``None`` means "no optimization state", which is a real answer and
    # distinct from "not supplied".
    if opt_state:
        context_lines.append(f"- **Optimization Status:** {opt_state.status}")
        context_lines.append(f"- **Iteration:** {opt_state.current_iteration}/{opt_state.max_iter_cap}")
        context_lines.append(f"- **Targets:** {json.dumps(opt_state.target_metric_scores)}")
        if opt_state.iterations:
            latest = opt_state.iterations[-1]
            latest_rate = latest['passed'] / latest['total_cases'] if latest.get('total_cases') else 0
            context_lines.append(
                f"- **Latest Result:** {latest['prompt_version']} — "
                f"{latest_rate:.1%} pass rate"
            )
            best = opt_state.best_iteration()
            if best and best != latest:
                best_rate = best['passed'] / best['total_cases'] if best.get('total_cases') else 0
                context_lines.append(
                    f"- **Best So Far:** {best['prompt_version']} — "
                    f"{best_rate:.1%} pass rate"
                )

    # Active eval job status
    from app.services.jobs import get_job_tracker
    job_tracker = get_job_tracker()
    active_job = job_tracker.get_active_job(project.id)
    if active_job:
        context_lines.append(
            f"- **Active Eval Job:** {active_job.status} "
            f"(job_id={active_job.job_id[:12]}, prompt={active_job.progress.get('message', '')})"
        )
    else:
        context_lines.append("- **Active Eval Job:** none")

    # Harness config summary. An unreadable config.py must not stop the turn:
    # this context is built before every chat turn, and the chat is how the
    # file gets repaired, so a raise here would leave the project unfixable.
    try:
        harness = load_harness_config(project_dir)
    except Exception as exc:  # noqa: BLE001 — any failure degrades to a note
        logger.warning(
            "[coordinator_prompt] src/config.py unreadable for project %s: %s",
            project.id, exc,
        )
        harness = None
        context_lines.append(
            f"- **Harness config:** src/config.py could not be read: {exc}. "
            "Fix the file before running anything."
        )
    if harness is not None:
        try:
            context_lines.append(f"- **Models:** {', '.join(harness.models) if harness.models else 'not configured'}")
            fm = harness.field_mappings
            if fm:
                mappings_str = ", ".join(f"{k}={v}" for k, v in fm.items())
                context_lines.append(f"- **Field Mappings:** {mappings_str}")
        except Exception:
            pass

    context_lines.append("")
    context_lines.append(
        "NOTE: This context is a snapshot from when your turn started. "
        "For live state during long-running operations, call `get_project_status`. "
        "NEVER guess about job status — always verify with the tool."
    )

    return "\n".join(context_lines)
