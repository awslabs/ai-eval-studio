# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory and lifecycle management for AI Eval Studio.

Creates the FastAPI app, configures CORS, registers API routers, and manages
the ProjectManager singleton. Pre-warms the MCP eval server at startup.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Configure app-level logging with an explicit handler that survives
# uvicorn's logger reconfiguration during --reload.
# Uses uvicorn's colored formatter for consistent terminal output.
import logging

from app.platform.log_context import RequestContextFilter

# Format includes correlation IDs from RequestContextFilter so a log
# operator can grep CloudWatch for ``req=<id>`` to follow a user
# request across services. ``-`` placeholders show when no request
# context is active (startup, background tasks before they bind a
# context).
_LOG_FORMAT = (
    "%(levelprefix)s %(name)s "
    "[req=%(request_id)s user=%(user_id)s proj=%(project_id)s] "
    "%(message)s"
)
_LOG_FORMAT_FALLBACK = (
    "%(levelname)s: %(name)s "
    "[req=%(request_id)s user=%(user_id)s proj=%(project_id)s] "
    "%(message)s"
)
try:
    from uvicorn.logging import DefaultFormatter
    _app_formatter = DefaultFormatter(_LOG_FORMAT)
except ImportError:
    _app_formatter = logging.Formatter(_LOG_FORMAT_FALLBACK)

_app_handler = logging.StreamHandler()
_app_handler.setFormatter(_app_formatter)
# Filter runs once per record on the handler, so every logger that
# eventually emits through this handler picks up correlation fields
# without each call site needing ``extra=``.
_app_handler.addFilter(RequestContextFilter())
_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)
_app_logger.addHandler(_app_handler)
_app_logger.propagate = False  # Don't duplicate to root logger

# Suppress noisy third-party loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Demote uvicorn access logs to WARNING. uvicorn writes every access line at
# INFO whatever the status, so this drops all of them, 4xx and 5xx included;
# the SPA's backstop polling would otherwise drown out real signal.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# Enable SDK logging at INFO (bump to DEBUG for connection/auth troubleshooting)
logging.getLogger("claude_agent_sdk").setLevel(logging.INFO)
logging.getLogger("claude_agent_sdk").addHandler(_app_handler)
logging.getLogger("claude_agent_sdk").propagate = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.platform import get_platform
from app.platform.auth import auth_middleware
from app.services.project_manager import ProjectManager

# Root directory where all project folders are stored.
#
# Cloud mode: ``PROJECTS_DIR`` env var is set by CDK (``compute_stack.py``)
# to ``/mnt/projects`` — the EFS mount point. Project working trees live at
# ``{PROJECTS_DIR}/users/{owner_sub}/projects/{pid}/``.
#
# Dev mode: env var unset, fall back to the repo-relative ``projects/``
# directory. The flat ``{PROJECTS_DIR}/{pid}/`` layout is preserved for
# the existing tests + local workflow; ``ProjectManager`` switches layout
# based on ``EVAL_STUDIO_MODE``.
PROJECTS_DIR = Path(
    os.environ.get(
        "PROJECTS_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "projects"),
    )
)

# Global singleton instance of ProjectManager, initialized on first access
_project_manager: ProjectManager | None = None


def get_project_manager() -> ProjectManager:
    """Get or initialize the singleton ProjectManager instance.

    Uses lazy initialization to create the ProjectManager only when first needed.
    This function is called by API endpoint handlers to access project operations.

    Returns:
        ProjectManager: The singleton instance managing all projects.
    """
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager(PROJECTS_DIR)
    return _project_manager


def _iter_project_dirs() -> "list[Path]":
    """Return every project directory under both dev and cloud layouts.

    Dev:   ``{PROJECTS_DIR}/{pid}/``
    Cloud: ``{PROJECTS_DIR}/users/{sub}/projects/{pid}/``

    The two layouts coexist on disk in dev (a developer who runs the
    cloud test fixture has both); always check both. Returned list is
    de-duplicated and order-stable so logs are deterministic.
    """
    if not PROJECTS_DIR.exists():
        return []

    seen: set[Path] = set()
    project_dirs: list[Path] = []

    # Cloud layout — config.json acts as the project marker so we don't
    # accidentally treat the ``users`` umbrella directory itself or any
    # non-project leaf as a project.
    users_root = PROJECTS_DIR / "users"
    if users_root.exists():
        for cfg in sorted(users_root.glob("*/projects/*/config.json")):
            project_dir = cfg.parent
            if project_dir not in seen:
                seen.add(project_dir)
                project_dirs.append(project_dir)

    # Dev (flat) layout. Filter to dirs that look like project roots —
    # ignore the ``users`` umbrella + any other top-level non-project
    # directory (cache dirs, etc.).
    for child in sorted(PROJECTS_DIR.iterdir()):
        if not child.is_dir() or child.name == "users":
            continue
        if not (child / "config.json").exists():
            continue
        if child not in seen:
            seen.add(child)
            project_dirs.append(child)

    return project_dirs


def _kill_orphan_eval_pid(pid: int, project_id: str) -> bool:
    """Best-effort: kill an orphan harness subprocess from a prior coordinator.

    A coordinator restart wipes ``JobTracker._jobs`` (in-memory). The new
    coordinator has no PID handle on the surviving harness subprocess,
    which keeps running and accumulating Bedrock spend until manual
    intervention. The on-disk ``OptimizationState.active_eval_pid`` is
    the only handle that survives the restart, so we use it here.

    Two defensive guards before sending SIGKILL:

    * ``os.kill(pid, 0)`` — confirms a process with that PID exists.
      A None or already-exited PID is a no-op.
    * ``/proc/{pid}/cmdline`` substring match against ``run_eval.py`` —
      catches the rare case where the PID was recycled by an unrelated
      process during the downtime. We don't want to SIGKILL an
      arbitrary child of init.

    Returns True if a kill signal was actually sent, False otherwise.
    Failures swallow — recovery must never block startup.
    """
    if pid is None or pid <= 0:
        return False
    log = logging.getLogger(__name__)
    try:
        os.kill(pid, 0)  # liveness probe; raises if no such pid
    except (ProcessLookupError, PermissionError, OSError):
        return False
    # Confirm the still-live PID actually belongs to a harness child.
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.exists():
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            cmdline = ""
        if "run_eval.py" not in cmdline:
            log.info(
                "[startup] skipping orphan-pid kill project=%s pid=%d "
                "(cmdline doesn't reference run_eval.py — pid likely "
                "recycled by an unrelated process)",
                project_id[:12], pid,
            )
            return False
    # Best-effort SIGKILL the whole process group (subprocess was
    # spawned with ``start_new_session=True`` so its PID is the PGID).
    import signal
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        pgid = pid
    try:
        os.killpg(pgid, signal.SIGKILL)
        log.warning(
            "[startup] SIGKILL'd orphan harness subprocess project=%s pid=%d pgid=%d",
            project_id[:12], pid, pgid,
        )
        return True
    except (ProcessLookupError, PermissionError, OSError) as e:
        log.warning(
            "[startup] failed to SIGKILL orphan project=%s pid=%d: %s",
            project_id[:12], pid, e,
        )
        return False


async def _recover_orphaned_evals() -> None:
    """SIGKILL harness subprocesses orphaned by a coordinator restart.

    A harness eval runs as a subprocess (``start_new_session=True``)
    whose PID lives in the in-memory ``JobTracker``. A coordinator
    restart wipes that handle, but the subprocess keeps running and
    spending Bedrock credits with nothing able to stop it. The on-disk
    active-eval record (``output/active_eval.json``, written by
    ``harness_runner`` for EVERY eval — manual and optimization alike)
    is the only surviving handle, so we scan it for every project and
    kill the orphan.

    Eval-scoped and runs for projects that never optimized — closing the
    manual-eval orphan leak that optimization-only recovery couldn't
    reach. Stale *optimization loop* records are handled separately by
    the adoption sweep (``adopt_orphan_optimization_loops``), which runs
    after this at startup.
    """
    from app.services.active_eval_state import clear_active_eval, load_active_eval

    log = logging.getLogger(__name__)
    killed = 0
    for project_dir in _iter_project_dirs():
        try:
            record = load_active_eval(project_dir)
        except Exception as exc:  # noqa: BLE001 — never block startup
            log.warning(
                "[startup] failed to load active-eval record for project=%s: %s",
                project_dir.name[:12], exc,
            )
            continue

        if record is None:
            continue

        log.info(
            "[startup] Orphan eval subprocess for project=%s pid=%d job=%s — killing",
            project_dir.name[:12], record.pid, record.job_id[:8] if record.job_id else "?",
        )
        if _kill_orphan_eval_pid(record.pid, project_dir.name):
            killed += 1
        # Clear the record regardless of kill success — the process is
        # either dead or unreachable, and leaving it would make the next
        # restart probe a certainly-stale PID.
        clear_active_eval(project_dir)

    if killed:
        log.warning("[startup] SIGKILL'd %d orphan eval subprocess(es)", killed)


async def _fail_stale_dev_job_records() -> None:
    """Fail every non-terminal job record at dev startup.

    Dev jobs run in-process, so a job record still non-terminal after a
    restart is definitionally a corpse. Flipping them serves two masters:
    honest job history, and the adoption sweep — which correctly defers
    while it sees live-looking optimization children, and would otherwise
    defer forever in dev (no reconciler runs there to flip stuck jobs the
    way cloud's does).

    Never blocks startup; each record failure is logged and skipped.
    """
    from app.platform import get_platform

    log = logging.getLogger(__name__)
    store = get_platform().metadata_store
    try:
        stale = await store.list_non_terminal_jobs()
    except Exception as exc:  # noqa: BLE001 — never block startup
        log.warning("[startup] could not list non-terminal dev jobs: %s", exc)
        return

    failed = 0
    for job in stale:
        try:
            # Record shape per the single put_job writer (jobs/__init__.py):
            # owner_id / project_id / job_id. The dev store's update keys on
            # project_id + job_id only.
            await store.update_job_state(
                job.get("owner_id", ""),
                job["project_id"],
                job["job_id"],
                state="failed",
            )
            failed += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[startup] failed to flip stale dev job=%s: %s",
                str(job.get("job_id") or "?")[:8], exc,
            )
    if failed:
        log.warning(
            "[startup] failed %d stale dev job record(s) from the previous process",
            failed,
        )


async def _audit_project_index_items() -> None:
    """Name every cloud project whose ``PROJECT_INDEX#{id}/META`` item is missing,
    and backfill it.

    ``require_access`` resolves a project through ``find_project_by_id``, which is
    a TWO-STEP read: look up ``PROJECT_INDEX#{id}/META`` to discover the owner,
    then read the real record from the owner's partition. Step 1 missing means the
    whole lookup returns ``None``, and ``require_access`` answers 404 — for every
    caller, the owner included, with no error anywhere else.

    The index item is written by ``put_project`` right after the record itself, so
    a project lacks it only if that second write failed. That is expected to be
    zero or near-zero projects, but "expected to be zero" is exactly the kind of
    assumption that should be checked by a machine rather than spot-checked by hand
    on one project.

    Backfill is safe and cheap here, which is why this does more than log. The
    owner is already known from the EFS path (``users/{owner}/projects/{pid}``),
    and the repair re-puts the record, which idempotently rewrites both the record
    and its index item. On-read backfill inside the store is NOT possible for this
    shape: once step 1 misses, the store has no way to learn the owner, so it
    cannot reconstruct the index. The startup sweep is the only place with both
    halves.

    Never blocks startup: every failure is logged and skipped.
    """
    from app.platform import get_platform

    log = logging.getLogger(__name__)
    store = get_platform().metadata_store
    missing: list[str] = []
    repaired = 0

    for project_dir in _iter_project_dirs():
        project_id = project_dir.name
        parts = project_dir.parts
        try:
            owner_id = parts[parts.index("users") + 1]
        except (ValueError, IndexError):
            # Dev (flat) layout — no durable index to audit.
            continue

        try:
            found = await store.find_project_by_id(project_id)
        except Exception as exc:  # noqa: BLE001 — never block startup
            log.warning(
                "[startup] project-index audit read failed for project=%s: %s",
                project_id[:12], exc,
            )
            continue

        if found is not None:
            continue

        # The lookup missed. Distinguish "index item absent" (repairable) from
        # "project genuinely not in the table" (nothing to repair — an EFS tree
        # with no durable record, e.g. a half-deleted project).
        try:
            record = await store.get_project(owner_id, project_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[startup] project-index audit owner read failed for project=%s: %s",
                project_id[:12], exc,
            )
            continue

        if record is None:
            log.warning(
                "[startup] project=%s (owner=%s) has an EFS tree but NO durable "
                "project record — not an index gap; leaving it alone",
                project_id[:12], owner_id[:8],
            )
            continue

        missing.append(project_id)
        log.error(
            "[startup] project=%s (owner=%s) is MISSING its PROJECT_INDEX item. "
            "require_access resolves projects via find_project_by_id, so every "
            "request for this project was answered 404. "
            "Backfilling the index.",
            project_id[:12], owner_id[:8],
        )

        # Re-put the record: put_project writes both the canonical item and the
        # index entry, and it is idempotent, so this repairs the index without
        # inventing a new write path.
        try:
            await store.put_project(record)
            repaired += 1
        except Exception as exc:  # noqa: BLE001
            log.error(
                "[startup] project-index backfill FAILED for project=%s: %s. "
                "Every request for this project keeps answering 404 until the "
                "index item is repaired.",
                project_id[:12], exc,
            )

    if missing:
        log.error(
            "[startup] project-index audit: %d project(s) were missing their index "
            "item (%s); %d repaired",
            len(missing), ", ".join(p[:12] for p in missing), repaired,
        )
    else:
        log.info("[startup] project-index audit: all projects have an index item")


async def _recover_pending_deletes() -> None:
    """Finish any project delete interrupted by a coordinator restart.

    Project deletion writes ``deleting: true`` to config.json before
    touching any store, then removes the stores. If the coordinator dies
    mid-delete (crash, OOM, deploy), the project is left flagged
    ``deleting`` — invisible to the UI/auth (``get_project`` hides it) but
    not yet fully removed. This sweep completes those deletes on the next
    startup so data is guaranteed gone and no orphan lingers.

    Idempotent: re-runs the same store cleanup the route uses
    (``_complete_project_deletion``); each step tolerates already-gone.
    Failures are logged, not fatal — the marker stays and the next startup
    retries. Runs in both dev and cloud.
    """
    import json as _json

    from app.api.projects import _complete_project_deletion

    log = logging.getLogger(__name__)
    completed = 0
    for project_dir in _iter_project_dirs():
        config_path = project_dir / "config.json"
        try:
            data = _json.loads(config_path.read_text())
        except Exception:  # noqa: BLE001 — never block startup
            continue
        if not data.get("deleting"):
            continue

        project_id = data.get("id") or project_dir.name
        log.warning(
            "[startup] completing interrupted delete: project=%s",
            str(project_id)[:12],
        )
        try:
            await _complete_project_deletion(
                project_id,
                owner_id=data.get("owner_id"),
                session_id=data.get("session_id"),
            )
            completed += 1
        except Exception as exc:  # noqa: BLE001 — marker stays; next boot retries
            log.warning(
                "[startup] failed to complete delete project=%s: %s "
                "(marker kept; will retry next startup)",
                str(project_id)[:12], exc,
            )

    if completed:
        log.warning(
            "[startup] completed %d interrupted project delete(s)", completed,
        )


async def _reattach_surviving_workers() -> None:
    """Re-attach progress pollers to surviving workers after a control-plane restart.

    When the coordinator restarts mid-eval (deploy, OOM, restart), the
    in-memory JobTracker is empty, but detached Fargate workers keep running.
    Each non-terminal DDB job carrying a ``compute_handle`` (the ECS task ARN)
    is a surviving worker whose progress/completion poller must be re-established.

    The poller is the exactly-once completion-trigger owner — its
    ``claim_completion_fired`` conditional write ensures a re-attached poller
    (or a second concurrent observer) cannot double-fire the completion turn.

    Dev jobs (PID-shaped handles, or no handle) are skipped — dev runs in-process,
    and the dev startup branch already killed those orphan subprocesses via
    ``_recover_orphaned_evals``.

    Runs on the cloud startup branch only. Dev startup does NOT call this
    (the orphan-killers run there instead).

    Delegates to ``adopt_unpolled_jobs`` so startup reattach and the periodic
    coordinator sweep are ONE code path — a second copy of the listing/iteration
    would be free to drift out of step with it. The sweep derives
    ``fire_completion`` from the durable record's source field, so a reattached
    optimization-sourced poller no longer claims a completion the loop owns.
    """
    from app.services.job_progress_republisher import adopt_unpolled_jobs

    log = logging.getLogger(__name__)

    try:
        reattached = await adopt_unpolled_jobs()
        if reattached > 0:
            log.info("[startup] re-attached %d surviving worker pollers", reattached)
        else:
            log.info("[startup] no surviving workers to reattach")
    except Exception as exc:  # noqa: BLE001 — never block startup
        log.warning("[startup] failed to adopt surviving workers: %s", exc)


async def _shutdown_active_work() -> None:
    """Force-cancel in-flight async work so ``uvicorn --reload`` restarts cleanly.

    Why this exists: when WatchFiles triggers a reload while an SDK
    subprocess, eval subprocess, or long-running optimization task is in
    flight, the worker blocks on draining those pipes instead of exiting.
    The reloader times out and the port stays half-dead until a new
    client knocks on it. Canceling everything at shutdown — orchestrator
    tasks, eval jobs (with SIGTERM on their PIDs), and agent-turn SDK
    clients — lets the worker exit promptly so the reloader can spawn a
    fresh one.

    Best-effort only: any failure here is swallowed so a broken tracker
    never blocks the shutdown path.
    """
    import os
    import signal

    from app.services.agent_turn import get_agent_turn_tracker
    from app.services.jobs import get_job_tracker

    # 1. Cancel every orchestrator / eval / agent asyncio task by name prefix.
    #    Names we set at creation:
    #      - ``optimization-<prefix>`` (optimization loop)
    #      - ``job-eval-<prefix>`` (eval subprocess driver — see
    #        ``mcp_tools/dispatch.py``; matches BOTH manual and
    #        optimization-driven evals because they share the spawn site)
    #      - ``optimizer-heartbeat-<prefix>`` (optimization-loop heartbeat)
    #      - ``relay-turn-<prefix>`` (AgentCore relay drain — see ``api/chat.py``)
    #    Without the ``job-`` prefix the eval driver task is missed,
    #    its stdout/stderr drain coroutines keep holding the
    #    subprocess pipes, and uvicorn's reload hangs the worker.
    #
    #    ``relay-turn-`` was missing, so a relay drain was never cancelled on a
    #    graceful shutdown: it died with the process without running the
    #    ``finally`` that releases its AgentCore session claim, manufacturing
    #    exactly the orphaned claim the startup sweep exists to clear.
    cancellable_prefixes = (
        "optimization-",
        "eval-",
        "job-",
        "optimizer-heartbeat-",
        "agent-turn-",
        "relay-turn-",
    )
    cancelled_tasks: list[asyncio.Task] = []
    for task in asyncio.all_tasks():
        name = task.get_name()
        if any(name.startswith(p) for p in cancellable_prefixes) and not task.done():
            task.cancel()
            cancelled_tasks.append(task)
    log = logging.getLogger(__name__)
    if cancelled_tasks:
        log.info(
            "[shutdown] Cancelling %d background task(s): %s",
            len(cancelled_tasks),
            [t.get_name() for t in cancelled_tasks],
        )

    # 2. Kill any eval subprocess the tracker knows about. The tracker's
    #    task cancellation (step 1) is not enough — the uv/python child
    #    process lives in its own session and must be SIGTERM'd directly.
    try:
        job_tracker = get_job_tracker()
        for job in list(getattr(job_tracker, "_jobs", {}).values()):
            if job.pid and getattr(job, "status", None) in ("running", "pending"):
                try:
                    # ``start_new_session=True`` at launch → SIGTERM the PGID.
                    os.killpg(os.getpgid(job.pid), signal.SIGTERM)
                    log.info("[shutdown] SIGTERM eval subprocess pgid=%s", job.pid)
                except (ProcessLookupError, OSError, PermissionError):
                    pass
    except Exception as e:
        log.warning("[shutdown] Eval-job kill pass failed: %s", e)

    # 3. Interrupt active SDK clients so AnyIO subprocess pipes unblock.
    try:
        turn_tracker = get_agent_turn_tracker()
        for turn in list(getattr(turn_tracker, "_turns", {}).values()):
            client = getattr(turn, "_client", None)
            if client and getattr(turn, "status", None) == "running":
                try:
                    await client.interrupt()
                    log.info("[shutdown] Interrupted agent turn %s", turn.turn_id[:8])
                except Exception:
                    # No clean API → rely on task cancellation above.
                    pass
    except Exception as e:
        log.warning("[shutdown] Agent-turn interrupt pass failed: %s", e)

    # 4. Give the cancelled tasks a short window to unwind before we return.
    #    Beyond this, uvicorn's own shutdown timer will kill the worker.
    if cancelled_tasks:
        try:
            await asyncio.wait(cancelled_tasks, timeout=3.0)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup (dev mode only):
    - Ensures the local projects directory exists.

    On startup (every mode):
    - Pre-warms the MCP eval server (imports deepeval, pandas, boto3).
    - Adopts orphaned optimization loops for any project on disk
      (any state at ``running``/``stopping`` after a coordinator
      restart has no live orchestrator and would otherwise stick).
    - Completes any project delete interrupted by a prior restart
      (config.json still flagged ``deleting``) so half-deleted projects
      are fully removed instead of lingering as invisible orphans.

    On shutdown:
    - Force-cancels in-flight background work so ``--reload`` restarts
      promptly without leaving the port half-dead.
    - Clears the ProjectManager reference.

    Cloud mode skips the dev FS-mutating ``mkdir`` step. Projects live
    on EFS at ``/mnt/projects`` (the mount point comes from the
    ``PROJECTS_DIR`` env var). The mount itself is provisioned +
    permission-checked by the Fargate task at attach time, so we
    don't need to ``mkdir`` here. Per-user subdirectories under
    ``users/{sub}/projects/{pid}/`` get created on demand at first
    project create. Orphaned-state recovery still runs in cloud mode
    — that's the whole point of the per-deploy recovery sweep.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control during the application's running phase.
    """
    global _project_manager

    # Bound locally, matching every sibling recovery function in this module. The
    # startup-recovery blocks below log through it, so an unbound name here does
    # not degrade to a missing log line — it raises out of lifespan and aborts boot.
    log = logging.getLogger(__name__)

    is_cloud = os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"
    if not is_cloud:
        PROJECTS_DIR.mkdir(exist_ok=True)

    # Verify all packaged prompt resources exist. This check is deliberately NOT inside
    # a try/except, departing from the convention of every sibling recovery block in
    # this function. Those blocks recover transient runtime state where a failure means
    # "could not clean up" — swallowing and logging is appropriate because blocking boot
    # would escalate a cleanup miss into an outage. A missing packaged prompt is a
    # deterministic build defect that will never self-heal, and swallowing it defers
    # the failure to the worst possible moment — mid-optimization, after the user has
    # already paid for an eval run. The check makes the guarantee total and the
    # diagnostic legible.
    #
    # This is consistent with existing behavior rather than a new policy: importing
    # app.main transitively imports app.agents, whose four modules read their .md
    # files at import time, so four of these seven resources already abort process
    # import when absent. The check closes the gap for the three lazily-read resources
    # (coordinator.md, _rules.md, deepeval-metric-reference.md).
    from app.prompts import verify_prompt_resources
    verify_prompt_resources()

    # Pre-warm MCP server at startup so the first chat message doesn't pay
    # the import cost of deepeval, pandas, boto3, etc.
    from app.services.mcp_server_cache import pre_warm_mcp_server
    pre_warm_mcp_server()

    # Start the WebSocket hub's keepalive task so idle ``/ws`` connections
    # stay alive past the ALB's default 60s idle timeout. Idempotent.
    from app.services.ws_hub import get_ws_hub
    get_ws_hub().start_keepalive()

    # Branch on cloud vs dev startup recovery:
    # - Cloud: reattach pollers to surviving workers, adopt orphaned loops and
    #   start the reconciler. Skip the orphan-killers — cloud has no in-process
    #   eval subprocesses; all evals are detached Fargate workers that survive
    #   the restart.
    # - Dev: run the orphan-killers to SIGKILL in-process eval subprocesses,
    #   fail their stale job records, and adopt orphaned loops.
    if is_cloud:
        # Cloud startup: reattach + reconciler
        await _reattach_surviving_workers()
        # Adopt orphaned optimization loops immediately (dispatch-to-owner
        # seam). The reconciler repeats this on its cadence, but its loop
        # sleeps a full interval before the first pass — a deploy that
        # replaced a coordinator mid-loop (or a chat dispatch from the
        # microVM just before boot) should not wait 60s for its loop.
        # Never blocks startup.
        try:
            from app.services.optimization_loop_orchestrator import (
                adopt_orphan_optimization_loops,
            )

            adopted = await adopt_orphan_optimization_loops()
            if adopted:
                logging.getLogger(__name__).info(
                    "[startup] Adopted %d orphaned optimization loop(s)", adopted
                )
        except Exception as exc:  # noqa: BLE001 — never block startup
            logging.getLogger(__name__).warning(
                "[startup] Loop-adoption pass failed: %s", exc
            )
        from app.services.eval_reconciler import start_reconciler_loop
        start_reconciler_loop()

        # Recover orphaned AgentCore session claims. Any persisted claim still
        # marked running after a restart is definitionally orphaned — the relay
        # task that held it died with the previous process. Clear them through the
        # version-fenced reconciler.
        #
        # This belongs on the CLOUD path because that is the only place a relay
        # claim can exist: the relay resolver is forced off outside cloud mode, and
        # the claim is only taken from the relay branch of the chat route. Running
        # it in dev inspects a table that is guaranteed empty, while cloud — where
        # a redeploy really can leave a claim held — would never reconcile at all.
        # An uncleared claim makes the next message for that tenant fail its claim,
        # get refused as busy, and be DROPPED (there is no queue) until the TTL
        # expires hours later.
        #
        # Swallows exceptions so a reconcile failure never blocks boot.
        try:
            from app.services.agentcore_session_reconciler import (
                reconcile_orphaned_agentcore_claims,
            )

            result = await reconcile_orphaned_agentcore_claims()
            if result["claims_cleared"] > 0:
                log.info(
                    "[startup] Cleared %d orphaned AgentCore session claim(s)",
                    result["claims_cleared"],
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[startup] AgentCore claim reconcile failed: %s", exc)

        # Audit (and repair) the PROJECT_INDEX items require_access depends on. A
        # missing index item makes find_project_by_id return None, so every
        # request for that project answers 404 with no error anywhere else.
        try:
            await _audit_project_index_items()
        except Exception as exc:  # noqa: BLE001 — never block boot
            log.warning("[startup] project-index audit failed: %s", exc)
    else:
        # Dev startup (unified with cloud where the mechanics allow):
        # 1. Kill harness subprocesses orphaned by the restart. Dev-only:
        #    dev evals are in-process children; cloud evals are detached
        #    Fargate workers that survive.
        await _recover_orphaned_evals()

        # 2. Fail the killed jobs' durable records. In dev, ANY non-terminal
        #    job record at startup is definitionally a corpse (the process
        #    that ran it just died) — and the adoption sweep below correctly
        #    DEFERS while it sees live-looking children, so leaving corpses
        #    would wedge adoption forever (dev runs no reconciler to flip
        #    them later, unlike cloud).
        await _fail_stale_dev_job_records()

        # 3. ADOPT stale loops — same seam as cloud, not the old
        #    flip-to-cancelled: a running/stopping record is a durable
        #    declaration that a loop should exist, and dev's store reads the
        #    same on-disk state the loop writes, so the sweep works here
        #    unchanged. Continuing the loop beats cancelling it; genuinely
        #    unadoptable records (config gone) are flipped to failed by the
        #    sweep itself.
        try:
            from app.services.optimization_loop_orchestrator import (
                adopt_orphan_optimization_loops,
            )

            adopted = await adopt_orphan_optimization_loops()
            if adopted:
                log.info(
                    "[startup] Adopted %d orphaned optimization loop(s) (dev)",
                    adopted,
                )
        except Exception as exc:  # noqa: BLE001 — never block startup
            log.warning("[startup] Dev loop-adoption pass failed: %s", exc)

    # Finish any project delete interrupted by a prior restart (config.json
    # still flagged ``deleting``). Guarantees a half-deleted project's data
    # is removed rather than lingering as an invisible orphan.
    # Runs in BOTH paths (unrelated to eval/optimization recovery).
    await _recover_pending_deletes()

    # Warm the TLS cert chain for the bedrock-mantle.{region}.api.aws
    # endpoint family in cloud mode (or when MANTLE_BOOT_SMOKE=1 is set
    # in dev). Non-blocking — failure logs WARN and continues so a
    # transient mantle outage doesn't block runtime evals.
    from app.services.bedrock_mantle_model import _verify_mantle_tls
    await _verify_mantle_tls()

    yield

    await _shutdown_active_work()
    get_ws_hub().stop_keepalive()

    # Stop the reconciler loop (idempotent — no-op if not running)
    from app.services.eval_reconciler import stop_reconciler_loop
    stop_reconciler_loop()

    _project_manager = None
    # Clear the Platform singleton so tests that swap PROJECTS_DIR get
    # a fresh build on the next request. In production this is a no-op.
    from app.platform import set_platform as _set_platform
    _set_platform(None)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Sets up:
    - Application metadata (title, version)
    - Lifespan context manager for startup/shutdown
    - CORS middleware (dev mode only — see below)
    - API routers for projects, chat, file upload, and eval
    - Health check endpoint

    Returns:
        FastAPI: Configured application instance ready to serve requests.
    """
    app = FastAPI(title="AI Eval Studio", version="0.1.0", lifespan=lifespan)

    # CORS is dev-only. The cloud deploy serves the SPA from CloudFront
    # and proxies /api + /ws to the same origin via the ALB VPC origin,
    # so browser requests are same-origin and CORS headers are dead
    # weight. Leaving the middleware on with the historical default
    # (``http://localhost:3100``) plus ``allow_credentials=True`` would
    # be a foot-gun: anyone who reached the ALB directly with a
    # spoofed Host header could echo CORS approval to a cross-origin
    # browser session. Skipping the middleware entirely in cloud mode
    # avoids that. Dev keeps the localhost SPA running on :3100, which
    # IS cross-origin against the backend on :8100.
    is_cloud = os.environ.get("EVAL_STUDIO_MODE", "dev").lower() == "cloud"
    if not is_cloud:
        cors_origins = os.environ.get(
            "CORS_ORIGINS", "http://localhost:3100",
        ).split(",")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors_origins],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Every request runs through the auth middleware so
    # ``request_context`` is populated for downstream code (and so cloud
    # mode never sees an unauthenticated request). Dev mode injects
    # ``LOCAL_DEV_USER_ID`` and behaves identically to cloud at the
    # request-context level.
    platform = get_platform()
    app.middleware("http")(auth_middleware(platform.auth))

    # API responses declare themselves non-storable so the browser HTTP
    # cache cannot serve a stale response body. This is the origin half
    # of a client+origin pair — the SPA's `apiFetch` enforces
    # `cache: "no-store"` at the request side, which is the primary
    # guard. This header makes the guarantee independent of any one
    # client so a future non-SPA caller or a CDN behavior change cannot
    # reintroduce it. Scoped to /api paths so the WebSocket route and
    # any future static paths are untouched.
    async def cache_control_middleware(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
    app.middleware("http")(cache_control_middleware)

    # Register API routers
    from app.api.projects import router as projects_router
    app.include_router(projects_router)

    from app.api.chat import router as chat_router
    app.include_router(chat_router)

    from app.api.upload import router as upload_router
    app.include_router(upload_router)

    from app.api.eval import router as eval_router
    app.include_router(eval_router)

    from app.api.files import router as files_router
    app.include_router(files_router)

    from app.api.synthetic import router as synthetic_router
    app.include_router(synthetic_router)

    from app.api.ws import router as ws_router
    app.include_router(ws_router)

    from app.api.config import router as config_router
    app.include_router(config_router)

    from app.api.reports import router as reports_router
    app.include_router(reports_router)

    # Reported as a function defined but never called. The decorator below
    # registers it as a route, which is the call site — the ALB target group
    # polls this path (network_stack.py), so removing it takes the service's
    # health signal with it.
    # nosemgrep: useless-inner-function
    @app.get("/api/health")
    async def health():
        """Health check endpoint to verify service availability."""
        return {"status": "ok"}

    return app


# Create the application instance that will be served by the ASGI server
app = create_app()
