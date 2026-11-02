# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for prompt management: save_prompt and analyze_failures.

``save_prompt`` persists a new prompt version as a Markdown file so the agent can
version prompts across optimization iterations. ``analyze_failures`` loads
an eval run result and returns a structured failure analysis (by category, metric,
and individual case) to guide the next prompt revision.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from claude_agent_sdk import ToolAnnotations, tool

from app.services.failure_analyzer import (
    analyze_failures as run_failure_analysis,
    diff_regressions,
    shape_failure_digest,
)
from app.services.mcp_response import mcp_error, mcp_json, mcp_success
from app.services.project_locks import get_project_lock
from app.services.project_paths import resolve_project_dir, safe_path

logger = logging.getLogger(__name__)


# Terminal-FAILURE optimization statuses where a save_prompt call should
# be refused if the optimization has no usable iteration data. Benign
# terminals (converged, plateau, max_iterations) are excluded — those
# states have real data to optimize against, so a save is legitimate.
# Live statuses (starting, running, stopping) are also excluded.
_TERMINAL_FAILURE_STATUSES = frozenset({
    "failed",
    "cancelled",
    "judge_unreliable",
    "generation_unreliable",
    "optimizer_stalled",
    "optimizer_exhausted",
})


async def _is_optimization_degenerate(project_id: str) -> bool:
    """Return True if the project's optimization is in a terminal-failure
    state with no usable iteration data, else False.

    A degenerate optimization is one where:
    - Status is in the terminal-FAILURE set (not benign terminal, not live), AND
    - Iterations list is empty OR newest iteration's total_cases is 0 or missing.

    Returns False (not-degenerate) when the state is absent, when the status
    is not a terminal failure, or when the newest iteration has real case data.
    Also returns False (fail-open) on state-read exceptions — an unreadable
    state must never brick prompt writes.

    Accepted residual: a halt at iteration N>1 leaves iteration N-1's
    real data, which is not degenerate. A save is permitted.
    """
    from app.platform.arch_split import resolve_arch_split

    try:
        project_dir = resolve_project_dir(project_id)

        # Resolve arch_split the same way optimization.py does
        arch_split = await resolve_arch_split(project_id)

        # No user_id resolution here any more: the repository keys the durable read
        # on the project OWNER. This used to resolve the REQUESTER via
        # current_request() with a dev fallback, which in the microVM (no request
        # context) read an empty partition and reported "not degenerate" for every
        # project — silently permitting saves it was meant to refuse.
        #
        # read() rather than read_for_display() deliberately: this guard must
        # distinguish "no state" (not degenerate) from "state unreadable" (fail open,
        # but SAY SO). read_for_display collapses both to None and logs under the
        # repository's logger, which would silently drop this call site's fail-open
        # diagnostic — the log line an operator greps when a save slips past the
        # guard.
        from app.services import optimization_state_repository as opt_repo

        handle = await opt_repo.read(
            project_id, project_dir, arch_split=arch_split
        )
        state = handle.state
        if state is None:
            return False  # No state — not degenerate

        # Check status
        if state.status not in _TERMINAL_FAILURE_STATUSES:
            return False  # Not a terminal failure — not degenerate

        # Check iterations
        if not state.iterations:
            return True  # Terminal failure + no iterations = degenerate

        # Check newest iteration's total_cases
        newest = state.iterations[-1]
        total_cases = newest.get("total_cases", 0)
        if total_cases == 0:
            return True  # Terminal failure + zero-case iteration = degenerate

        return False  # Has real iteration data — not degenerate

    except Exception as exc:
        # Fail-open: log the exception and return False so a state-read
        # failure never bricks prompt writes.
        logger.warning(
            "[save_prompt] project=%s state read failed: %s; failing open (allowing save)",
            project_id[:8] if project_id else "?",
            exc,
        )
        return False


def _normalize_prompt_content(content: str) -> str:
    """Normalize a prompt body for no-op comparison.

    Strips leading/trailing whitespace and the changelog header comment
    the optimizer is instructed to prepend (``<!-- Version: ... -->``)
    so a prompt that only differs in its version header still counts as
    identical to its predecessor.
    """
    stripped = re.sub(r"\A<!--.*?-->\s*", "", content.lstrip(), count=1, flags=re.DOTALL)
    return stripped.strip()


@tool(
    "save_prompt",
    "Save a new prompt version to the project's prompts directory. "
    "Refuses to save if the content (ignoring the changelog header) is "
    "identical to the most recent prompt — re-running the same prompt "
    "wastes an eval cycle. Also refuses when the optimization is in a "
    "terminal-failure state with no usable iteration data (halted without "
    "recording any cases), because a prompt derived from it would poison "
    "the next iteration.",
    {"project_id": str, "version": str, "content": str},
)
async def save_prompt(args: dict[str, Any]) -> dict[str, Any]:
    """Save a prompt version to disk as a Markdown file.

    Writes the given content to ``prompts/<version>.md`` within the project
    directory. Creates the prompts directory if it doesn't exist.

    Returns an error in two cases:
    1. The content matches the most recent existing prompt (dedupe refusal) —
       re-running the same prompt triggers an identical-eval cycle.
    2. The optimization is in a terminal-failure state with no usable iteration
       data (degenerate refusal) — a prompt derived from 0/0 data would poison
       the next iteration.

    Still works when:
    - No optimization state exists (user/agent saving outside a loop).
    - Status is running (in-loop optimizer's normal save on healthy iteration).
    - Terminal failure + real iteration data (e.g., optimizer_exhausted after
      N iterations — the documented "edit the prompt to resume" flow).
    - Benign terminal (converged, plateau) with real iteration data.
    - State read fails (fail-open: unreadable state must never brick writes).

    Args:
        args: Dict with ``project_id``, ``version`` (filename stem), and ``content``.

    Returns:
        Dict with confirmation of the saved filename, or an error when the
        content duplicates the most recent prompt or the state is degenerate.
    """
    try:
        # Degenerate guard: refuse when the optimization is in a terminal-failure
        # state with no usable iteration data. Check before taking the project
        # lock so a refusal costs no lock.
        if await _is_optimization_degenerate(args["project_id"]):
            logger.info(
                "[save_prompt] project=%s REJECTED %s: optimization halted without usable iteration data",
                args["project_id"][:8],
                args["version"],
            )
            return mcp_error(
                f"Refusing to save {args['version']}: the optimization halted without "
                f"usable iteration data (no iterations, or newest iteration has zero cases). "
                f"A prompt derived from it would poison the next iteration. Resume the "
                f"optimization or start a new loop, which will clear the terminal state."
            )

        project_dir = resolve_project_dir(args["project_id"])
        prompts_dir = project_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = safe_path(prompts_dir, f"{args['version']}.md")
        if prompt_path is None:
            return mcp_error(f"Invalid version name: {args['version']}")

        new_normalized = _normalize_prompt_content(args["content"])
        if not new_normalized:
            return mcp_error(
                "Prompt content is empty after stripping whitespace — refusing to save."
            )

        # Serialize the dedupe-check + atomic-rename for this project. Two
        # concurrent save_prompt calls (optimizer + manual edit, or two
        # orchestrator turns) would otherwise each pass the dedupe check
        # and the second os.replace would clobber the first write.
        project_lock = await get_project_lock(args["project_id"])
        async with project_lock:
            # Find the most recent *other* prompt to compare against. Sorting by
            # mtime covers both the baseline and version-name ordering (v1, v2,
            # ...); comparing to all existing prompts catches the "optimizer
            # saved v3 identical to v1" case too.
            latest_existing: tuple[float, Path] | None = None
            for existing in prompts_dir.glob("*.md"):
                if existing.resolve() == prompt_path.resolve():
                    continue
                mtime = existing.stat().st_mtime
                if latest_existing is None or mtime > latest_existing[0]:
                    latest_existing = (mtime, existing)

            if latest_existing is not None:
                prior_normalized = _normalize_prompt_content(
                    latest_existing[1].read_text(encoding="utf-8")
                )
                if prior_normalized == new_normalized:
                    logger.info(
                        "[save_prompt] project=%s REJECTED %s: identical to %s "
                        "(normalized_len=%d)",
                        args["project_id"][:8],
                        prompt_path.name,
                        latest_existing[1].name,
                        len(new_normalized),
                    )
                    return mcp_error(
                        f"Refusing to save {prompt_path.name}: content is identical to "
                        f"the most recent prompt {latest_existing[1].name} (ignoring the "
                        f"changelog header). Re-running the same prompt produces the "
                        f"same eval — revise the prompt before saving."
                    )

            # Atomic write: temp file in the same directory, then os.replace.
            # Protects against partial writes from a crash mid-write.
            with tempfile.NamedTemporaryFile(
                dir=prompts_dir,
                prefix=f".{prompt_path.name}.",
                suffix=".tmp",
                mode="w",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(args["content"])
                # The name is read here, but the file is replaced only after
                # this block, where the context manager has already closed and
                # flushed it.
                # nosemgrep: tempfile-without-flush
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, prompt_path)

        logger.info(
            "[save_prompt] project=%s SAVED %s (bytes=%d)",
            args["project_id"][:8], prompt_path.name, len(args["content"]),
        )
        return mcp_success(f"Prompt saved: {prompt_path.name}")

    except Exception as e:
        logger.exception("[save_prompt] project=%s ERROR: %s", args.get("project_id", "?")[:8], e)
        return mcp_error(f"Error: {e}")


@tool(
    "analyze_failures",
    "Digest of an eval run for prompt optimization. Returns: per-category and "
    "per-metric pass/fail breakdown; full failed-case details (input/expected/"
    "actual + failing-metric judge reasons); a bounded sample of PASSING-case "
    "exemplars (the 'what good looks like' template); and a regression diff vs "
    "the prior run (cases that went pass->fail). This is a lean digest/index — "
    "for full ground truth on every case, Read the raw output/eval_results_*.json "
    "file directly (you have the Read tool and your cwd is the project dir).",
    {"project_id": str, "run_filename": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def analyze_failures(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool wrapper for failure analysis + regression diff.

    Loads eval results and returns the structured digest (breakdowns,
    failed cases, passing exemplars). Additionally locates the
    immediately-preceding run via :meth:`EvalRunner.list_runs` (newest
    first) and attaches a ``regression`` block listing cases that went
    pass->fail since that run, so the optimizer can see what its last
    change broke. Regression data is informational, not a gate.

    Args:
        args: Dict with ``project_id`` and ``run_filename``.

    Returns:
        Dict with the failure digest, plus a ``regression`` key (omitted
        when there's no prior run to diff against).
    """
    try:
        project_dir = resolve_project_dir(args["project_id"])
        output_dir = project_dir / "output"
        results_path = safe_path(output_dir, args["run_filename"])
        if results_path is None:
            return mcp_error(f"Invalid filename: {args['run_filename']}")
        if not results_path.exists():
            return mcp_error(f"Run not found: {args['run_filename']}")

        analysis = run_failure_analysis(results_path)

        # Attach a regression diff against the immediately-preceding run.
        # ``list_runs`` is newest-first; the entry right AFTER the current
        # run in that list is the prior run. Best-effort — any failure to
        # locate/parse the prior run simply omits the regression block.
        try:
            from app.services.eval_runner import EvalRunner
            runs = EvalRunner().list_runs(project_dir)
            current_name = results_path.name
            prior_name = None
            for i, run in enumerate(runs):
                if run.get("_file") == current_name:
                    if i + 1 < len(runs):
                        prior_name = runs[i + 1].get("_file")
                    break
            if prior_name:
                prior_path = safe_path(output_dir, prior_name)
                if prior_path and prior_path.exists():
                    analysis["regression"] = diff_regressions(
                        results_path, prior_path,
                    )
        except Exception:
            # Regression diff is a best-effort enhancement; never let it
            # turn a successful failure analysis into an error.
            pass

        # Shape the digest for the MCP transport (bound + truncate under cap)
        digest = shape_failure_digest(analysis)
        return mcp_json(digest)

    except Exception as e:
        return mcp_error(f"Error: {e}")
