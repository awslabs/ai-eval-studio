# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background invocation of the report-narrator agent.

Two entry points:

- :func:`generate_run_report` — fires after every eval completion. Builds
  the per-run facts-pack, hands it to the narrator, the narrator's
  ``save_report`` writes the sidecar.
- :func:`generate_exec_report` — fires on optimization convergence,
  on winner-change, and on manual regenerate. Same flow, exec scope.

This is a silent, chat-decoupled background job. It never goes through
the coordinator / completion-turn path. Failures (bad creds, agent
error, network blip) log and return — the renderer's lean fallback
covers a missing sidecar.

**Exec narration is serialized per project** (not per run). A single
``asyncio.Lock`` per project gates the build→narrate→write critical
section, and an overflow flag coalesces concurrent triggers into one
"rerun pending" cycle. Without this, two overlapping narrations race
to write the same sidecar — the slower writer wins, but its prose was
written about an older facts snapshot, producing a prose-A/stamp-B
sidecar that survives until convergence. We deliberately do NOT reuse
:func:`app.services.project_locks.get_project_lock` here; that lock is
documented as gating short read-modify-write critical sections (prompt
saves, state writes), and a 20-second narration would block them all.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.prompts import PROMPTS_DIR
from app.services.report import (
    ExecFacts,
    build_exec_facts,
    build_run_facts,
    _facts_to_jsonable,
    write_report_json,
)

logger = logging.getLogger(__name__)


# Per-project narration locks + coalescer state. These are module-private
# and process-scoped — every call site goes through ``generate_exec_report``,
# never holds these directly.
_exec_narration_locks: dict[str, asyncio.Lock] = {}
_exec_narration_pending: dict[str, bool] = {}
_exec_narration_registry_lock = asyncio.Lock()
# Per-project facts snapshot for the in-flight narration. ``save_report``
# reads it so the sidecar's stamped facts come from the same build the
# narrator's prose was written about. None when no narration is in flight
# for the project (or the call is from outside the orchestrator path —
# e.g. the per-run path, which doesn't go through this serializer).
_exec_facts_snapshot: dict[str, ExecFacts] = {}


async def _get_exec_narration_lock(project_id: str) -> asyncio.Lock:
    """Return the shared exec-narration lock for ``project_id``."""
    async with _exec_narration_registry_lock:
        lock = _exec_narration_locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            _exec_narration_locks[project_id] = lock
        return lock


def get_exec_facts_snapshot(project_id: str) -> ExecFacts | None:
    """Read-only access to the in-flight exec facts snapshot.

    ``save_report`` calls this when persisting an executive sidecar so the
    facts written reflect the same winner the narrator was looking at,
    not a fresh build that may have advanced mid-narration.
    """
    return _exec_facts_snapshot.get(project_id)


def _reset_exec_narration_state() -> None:
    """Clear all module-private narration state. Test-only."""
    _exec_narration_locks.clear()
    _exec_narration_pending.clear()
    _exec_facts_snapshot.clear()


# Per-run narrative fields — see report-narrator.md.
_RUN_FIELDS = [
    "context",
    "takeaway",
    "failure_patterns",
    "what_changed_worked",
    "metric_descriptions",
]

# Executive narrative fields.
_EXEC_FIELDS = [
    "intro",
    "recommendation",
    "journey_narrative",
    "data_quality_insight",
    "outlook",
    "metric_descriptions",
]


#: Cap on failed-case rows handed to the NARRATOR. One measured run produced
#: 1,184 rows / 0.66 MB, and a narrator with ``max_turns=8`` instructed to
#: cluster failures into <=5 patterns cannot use them all — it is payload with
#: no consumer, on a path that has a hard 1 MiB per-message ceiling. The UI's
#: facts pack is NOT capped: it keeps every row, and only the narrator's copy is
#: sampled. Capping ``_build_failed_cases`` itself would silently drop rows the
#: report UI renders.
_NARRATOR_MAX_FAILED_CASES = 150


def _sample_one_level(facts_dict: dict[str, Any]) -> dict[str, Any]:
    """Sample ``facts_dict["failed_cases"]`` in place of the full list.

    Uses an even stride rather than the first N rows: the list is grouped by
    model then case, so a head slice would hand the narrator only the first
    model's failures and it would cluster patterns from one model while claiming
    to describe the run.

    The sample is labelled with the true total, because the narrator's
    instructions tell it to quote numbers verbatim and never recompute — an
    unlabelled truncation invites it to state a wrong failure count as fact.

    Args:
        facts_dict: A run-shaped facts dict (one carrying ``failed_cases``).

    Returns:
        A shallow copy with the rows sampled, or the SAME object when no
        sampling was needed. Identity is meaningful — the caller uses it to
        avoid copying parents that did not change.
    """
    rows = facts_dict.get("failed_cases")
    if not isinstance(rows, list) or len(rows) <= _NARRATOR_MAX_FAILED_CASES:
        return facts_dict
    stride = len(rows) / _NARRATOR_MAX_FAILED_CASES
    sampled = [rows[int(i * stride)] for i in range(_NARRATOR_MAX_FAILED_CASES)]
    trimmed = dict(facts_dict)
    trimmed["failed_cases"] = sampled
    trimmed["failed_cases_sampling"] = {
        "shown": len(sampled),
        "total": len(rows),
        "note": (
            "Evenly-strided sample of the failed cases, not the full list. "
            "Every COUNT elsewhere in this pack is complete — use those for "
            "totals, and treat these rows as examples to cluster."
        ),
    }
    return trimmed


def _sample_failed_cases_for_narrator(facts_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the facts pack with failed cases sampled to a usable size.

    Handles BOTH facts shapes, which is the whole point of the split below.
    ``RunFacts`` carries ``failed_cases`` at the top level; ``ExecFacts`` has
    none there and nests a whole ``RunFacts`` under ``winner_run``. This
    function used to look only at the top level, so on the executive path
    ``facts_dict.get("failed_cases")`` was ``None``, the guard returned the dict
    untouched, and the narrator received every row — the opposite of what the
    cap exists for, with nothing in the logs to say so.

    The failure was concrete: under the SDK's default 1 MiB frame ceiling, an
    unsampled executive pack was rejected with ``JSON message exceeded maximum
    buffer size of 1048576 bytes`` and the narrator died without writing a
    report. The ceiling is now raised (``SDK_MAX_STDOUT_FRAME_BYTES``), but the
    sampling still keeps the pack to a size the narrator can actually use.

    Returns shallow copies at each level it changes; the caller's ``facts_dict``
    is also persisted for the UI and must not be mutated.
    """
    sampled = _sample_one_level(facts_dict)

    winner = sampled.get("winner_run")
    if isinstance(winner, dict):
        sampled_winner = _sample_one_level(winner)
        if sampled_winner is not winner:
            # Copy the parent only now, so the untouched case stays copy-free.
            sampled = dict(sampled)
            sampled["winner_run"] = sampled_winner

    return sampled


def _build_task_message(
    project_id: str,
    report_type: str,
    facts_dict: dict[str, Any],
    fields: list[str],
    *,
    run_filename: str | None = None,
) -> str:
    """Compose the user message handed to the narrator agent.

    The message has the static instructions/header at the top (so Bedrock
    prompt caching hits across repeated calls in an optimization loop)
    followed by the run-specific facts-pack JSON.
    """
    facts_dict = _sample_failed_cases_for_narrator(facts_dict)
    header = (
        f"You are writing the prose for a {report_type} report.\n\n"
        f"project_id: {project_id}\n"
    )
    if run_filename:
        header += f"run_filename: {run_filename}\n"
    header += (
        f"\nProduce these narrative fields: {', '.join(fields)}\n\n"
        "Write only the prose fields. Quote numbers verbatim from the "
        "facts-pack — never recompute. Cluster failures into ≤5 patterns. "
        "Deliver your output by calling the save_report tool with:\n"
        f"  project_id: {project_id!r}\n"
        f"  report_type: {report_type!r}\n"
    )
    if run_filename:
        header += f"  run_filename: {run_filename!r}\n"
    header += (
        "  narrative: {your fields here}\n\n"
        "FACTS PACK BELOW. Read every section before you write.\n\n"
        "```json\n"
        f"{json.dumps(facts_dict, indent=2, default=str)}\n"
        "```\n"
    )
    return header


async def _run_narrator(
    project_dir: Path,
    project_id: str,
    task_message: str,
    owner_id: str | None,
) -> bool:
    """Invoke the narrator agent in a standalone SDK client.

    Follows the same standalone-``ClaudeSDKClient`` pattern as the prompt
    optimizer (``_run_prompt_optimizer`` in ``optimization_loop_orchestrator``).
    The narrator's only allowed tool is ``mcp__eval__save_report``; that
    call writes the sidecar. We don't read the agent's reply — the side
    effect on disk is the result.

    Request context is required, not optional: ``save_report`` is gated by
    ``mcp_authz.authorize_tool_call``, which fails CLOSED — with no context
    it returns ``mcp_error("not authenticated")`` and never runs the handler.
    Since ``save_report`` is the narrator's ONLY allowed tool, a missing
    context means the narrator cannot deliver anything at all. The API
    endpoints reach here inside the HTTP request, so the middleware's context
    is already live; the post-eval dispatchers reach here from detached
    ``create_task`` work with no context. See the attribution block below.

    Returns:
        True when the narrator emitted a save_report call that succeeded
        on disk; False on any failure (caller logs and moves on; the
        renderer's lean fallback covers a missing sidecar).
    """
    ctx_token = None
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
        from app.services.mcp_server_cache import get_eval_mcp_server
        from app.services.project_manager import _is_cloud_mode
        from app.services.sdk_subprocess_env import (
            SDK_MAX_STDOUT_FRAME_BYTES,
            sdk_subprocess_env,
        )
        from app.services.sdk_tool_hooks import (
            TOOL_OUTCOME_HOOK_EVENTS,
            ToolCallOutcome,
            make_tool_outcome_hook,
        )

        from app.main import get_project_manager
        pm = get_project_manager()
        project = pm.get_project(project_id)
        model = project.agent_model if project else "opus"

        narrator_system = (PROMPTS_DIR / "report-narrator.md").read_text()

        # This outcome IS this function's return value, so it has to be read
        # correctly — the payload-shape check it replaces never fired for an
        # MCP tool, which reported every working ``save_report`` as a failure.
        # See ``sdk_tool_hooks``.
        save_outcome = ToolCallOutcome()
        _save_hook = make_tool_outcome_hook(save_outcome)

        opts = ClaudeAgentOptions(
            allowed_tools=["mcp__eval__save_report"],
            permission_mode="bypassPermissions",
            cwd=str(project_dir),
            sandbox={"enabled": False},
            session_store_flush="eager",
            system_prompt=narrator_system,
            max_turns=8,
            mcp_servers={"eval": get_eval_mcp_server()},
            setting_sources=["user", "project"] if _is_cloud_mode() else ["project"],
            # SDK subprocess env pins: see sdk_subprocess_env.py for rationale.
            env=sdk_subprocess_env(),
            # Bounds ONE stdout frame. The SDK's 1 MiB default is what killed
            # this narrator in cloud; see SDK_MAX_STDOUT_FRAME_BYTES.
            max_buffer_size=SDK_MAX_STDOUT_FRAME_BYTES,
            hooks={
                event: [
                    HookMatcher(
                        matcher="mcp__eval__save_report",
                        hooks=[_save_hook],
                    ),
                ]
                for event in TOOL_OUTCOME_HOOK_EVENTS
            },
        )
        if model:
            opts.model = model

        from app.platform import get_platform
        _ss = get_platform().session_store
        if _ss is not None:
            opts.session_store = _ss

        # Give the narrator a context its save_report call can clear mcp_authz
        # with. Attribution to the owner happens only when no context exists;
        # on the API path the caller's own identity is already live and is kept.
        # Either way the context gets bound to this project, which the API path
        # needs because the HTTP middleware leaves project_id unset and the
        # authz pin compares against it. Mirrors the detached-turn handling in
        # ``agent_runner`` — "acting on the owner's behalf for a background
        # action", never "the owner made this request".
        #
        # Ordering matters: this must precede ``ClaudeSDKClient`` construction.
        # The client's internal tasks inherit the ContextVar as it stands when
        # they are created, so a set after ``__aenter__`` would not be visible
        # to the MCP tool handlers.
        from app.platform.types import attribute_detached_context, request_context
        ctx_token = attribute_detached_context(
            owner_id=owner_id,
            project_id=project_id,
            request_id=f"detached-report-{project_id[:8]}",
        )

        async with ClaudeSDKClient(options=opts) as client:
            await client.query(task_message)
            async for _message in client.receive_response():
                pass

        if not save_outcome.succeeded:
            logger.warning(
                "[report-narrator] project=%s narrator did not save a report "
                "(attempted=%s). Hooks saw: %s",
                project_id[:8], save_outcome.attempted, save_outcome.describe(),
            )
        return save_outcome.succeeded

    except Exception as e:
        logger.exception(
            "[report-narrator] project=%s failed: %s", project_id[:8], e,
        )
        return False
    finally:
        # Reset rather than relying on every caller being a dedicated
        # ``create_task``. All three detached dispatchers currently are, but an
        # inline ``await`` from a long-lived worker would otherwise leak the
        # owner identity into that task for every later operation.
        if ctx_token is not None:
            request_context.reset(ctx_token)


def _publish_report_outcome(
    project_id: str,
    scope: str,
    ok: bool,
    run_filename: str | None = None,
) -> None:
    """Tell an open Reports tab how a narration ended, without polling.

    ``report``/``updated`` means a new sidecar is on disk, so the tab
    refetches it. ``report``/``failed`` means none was written (the lean
    fallback still renders); it ends a pending regenerate without refetching
    unchanged prose. The regenerate endpoints return before the narrator
    finishes, so this event is the only way the tab learns the result.
    """
    # Lazy import to avoid the jobs ↔ report_generation cycle (jobs imports
    # generation indirectly via the eval executor's exec-dispatch).
    from app.services.jobs import _publish_project_event
    event: dict[str, Any] = {
        "resource": "report",
        "type": "updated" if ok else "failed",
        "scope": scope,
    }
    if run_filename is not None:
        event["run_filename"] = run_filename
    _publish_project_event(project_id, event)


async def generate_run_report(
    project_dir: Path,
    project_id: str,
    run_filename: str,
    owner_id: str | None = None,
) -> bool:
    """Generate the per-run report sidecar for ``run_filename``.

    Builds facts → invokes narrator → narrator's ``save_report`` writes
    the sidecar. Best-effort: returns False on any failure (lean fallback
    covers a missing sidecar at render time). Either way it publishes the
    outcome as a ``report`` project event.
    """
    project_dir = Path(project_dir)
    try:
        facts = build_run_facts(project_dir, run_filename)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s run=%s facts build failed: %s",
            project_id[:8], run_filename, e,
        )
        _publish_report_outcome(project_id, "run", False, run_filename)
        return False

    facts_dict = _facts_to_jsonable(facts)
    task = _build_task_message(
        project_id, "run", facts_dict, _RUN_FIELDS, run_filename=run_filename,
    )
    ok = await _run_narrator(project_dir, project_id, task, owner_id)
    _publish_report_outcome(project_id, "run", ok, run_filename)
    return ok


async def generate_exec_report(
    project_dir: Path,
    project_id: str,
    owner_id: str | None = None,
) -> bool:
    """Generate the executive report sidecar for the project.

    Serialized per project. If a narration is already running, this call
    flips the project's "rerun pending" flag and returns; the in-flight
    narration will run one more cycle when it finishes (which rebuilds
    facts → latest winner → fresh prose). Multiple concurrent triggers
    coalesce into exactly one rerun, never N queued runs.

    Best-effort: lock acquisition wraps a try/except, narration failures
    swallow errors (the renderer's lean fallback covers a missing
    sidecar). Never blocks the eval-completion path.
    """
    project_dir = Path(project_dir)
    try:
        lock = await _get_exec_narration_lock(project_id)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s exec lock unavailable: %s",
            project_id[:8], e,
        )
        _publish_report_outcome(project_id, "executive", False)
        return False

    if lock.locked():
        # Another narration is in flight — register one rerun and bail.
        # The current owner will see the flag on its way out and run
        # exactly one more cycle, which builds facts fresh. Nothing is
        # published here: each cycle publishes its own outcome.
        _exec_narration_pending[project_id] = True
        logger.info(
            "[report-narrator] project=%s exec narration already in flight; "
            "marked rerun pending", project_id[:8],
        )
        return False

    last_result = False
    async with lock:
        # Loop until no rerun is pending. Each iteration rebuilds facts so
        # a winner change that landed during the previous iteration's
        # narration is reflected in this iteration's prose.
        _exec_narration_pending.pop(project_id, None)
        while True:
            last_result = await _run_one_exec_narration_cycle(
                project_dir, project_id, owner_id,
            )
            if not _exec_narration_pending.pop(project_id, None):
                break
            logger.info(
                "[report-narrator] project=%s rerunning exec narration to "
                "pick up changes that landed mid-flight", project_id[:8],
            )
    return last_result


async def _run_one_exec_narration_cycle(
    project_dir: Path,
    project_id: str,
    owner_id: str | None,
) -> bool:
    """One build→stash→narrate→clear-stash pass. Caller owns the lock."""
    try:
        facts = build_exec_facts(project_dir)
    except Exception as e:
        logger.warning(
            "[report-narrator] project=%s exec facts build failed: %s",
            project_id[:8], e,
        )
        _publish_report_outcome(project_id, "executive", False)
        return False

    # Stash the snapshot for ``save_report`` so the persisted facts and
    # the agent's prose come from the same build, not a fresh rebuild
    # that may have advanced past the winner the agent just reasoned
    # about. Always cleared in the ``finally`` even if the agent dies.
    _exec_facts_snapshot[project_id] = facts
    try:
        facts_dict = _facts_to_jsonable(facts)
        task = _build_task_message(
            project_id, "executive", facts_dict, _EXEC_FIELDS,
        )
        ok = await _run_narrator(project_dir, project_id, task, owner_id)
        _publish_report_outcome(project_id, "executive", ok)
        return ok
    finally:
        _exec_facts_snapshot.pop(project_id, None)


def previous_winner_filename(project_dir: Path) -> str | None:
    """Read the current sidecar exec report's winner, if any.

    Used by the convergence trigger to decide whether to regenerate the
    exec report (winner-change semantics): when a new run lands and the
    new trophy run is different from the one stamped on the existing
    exec sidecar, regenerate.
    """
    sidecar_path = project_dir / "output" / "reports" / "executive.report.json"
    if not sidecar_path.exists():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    facts = data.get("facts") or {}
    return facts.get("winner_run_filename")
