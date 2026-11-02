# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-recovery record of the harness subprocess currently running an eval.

A harness eval runs as a subprocess whose PID lives only in the in-memory
``JobTracker``. When the coordinator restarts (deploy, OOM, force-stop) that
in-memory handle is lost, but the subprocess — spawned with
``start_new_session=True`` — keeps running and burning Bedrock credits with
nothing able to stop it. This module persists the live eval's PID to disk so
the startup recovery hook can find and SIGKILL the orphan.

One record per project (``output/active_eval.json``). Dispatch fences a project
to a single concurrent eval (see ``mcp_tools/dispatch.py``), so one slot is
sufficient. The record is written for EVERY eval — manual and optimization
alike — which is the whole point: previously only optimization evals persisted
a recovery PID (it was stored inside ``OptimizationState``), so a manual eval's
subprocess became an unkillable orphan across a restart. This record is the
single home for "the eval subprocess running for this project," independent of
whether an optimization loop is driving it.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)

#: Marker filename. Public so code that scans ``output/`` for eval runs can
#: recognise the marker without re-spelling the string — a scan that mistakes it
#: for a run parses a PID record as eval results.
ACTIVE_EVAL_FILENAME = "active_eval.json"

# Per-project location. Lives under ``output/`` (always present for a project
# that has run anything) rather than ``output/optimization/`` so it reads as
# eval-scoped, not optimization-scoped.
_ACTIVE_EVAL_FILE = Path("output") / ACTIVE_EVAL_FILENAME


@dataclass
class ActiveEvalRecord:
    """The harness subprocess currently running an eval for a project.

    Attributes:
        pid: OS process ID of the harness subprocess (process-group leader,
            since it's spawned with ``start_new_session=True``).
        started_at: ISO 8601 timestamp when the PID was recorded. Lets
            recovery defensively skip a PID that's suspiciously old relative
            to downtime (a PID the OS may have recycled).
        job_id: The ``JobTracker`` job id this subprocess belongs to, for
            cross-referencing logs.
    """

    pid: int
    started_at: str
    job_id: str

    def to_dict(self) -> dict:
        """Serialize for the on-disk JSON record."""
        return {"pid": self.pid, "started_at": self.started_at, "job_id": self.job_id}


def _record_path(project_dir: Path) -> Path:
    """Return the active-eval record path for a project directory."""
    return Path(project_dir) / _ACTIVE_EVAL_FILE


def record_active_eval(project_dir: Path, pid: int, job_id: str) -> None:
    """Persist the running eval subprocess PID for crash recovery.

    Called by ``harness_runner`` once the subprocess spawns, for every eval
    regardless of source. Atomic write (temp + rename) so a crash mid-write
    can't leave a half-written record that recovery would misparse.

    Best-effort: a failure here MUST NOT block eval startup. The cost of a
    missed record is a recoverable orphan on the next restart, not a crash.

    Args:
        project_dir: The project directory the eval runs in.
        pid: PID of the spawned harness subprocess (process-group leader).
        job_id: Owning ``JobTracker`` job id, for log cross-reference.
    """
    record = ActiveEvalRecord(
        pid=pid,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        job_id=job_id,
    )
    path = _record_path(project_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(record.to_dict(), f)
            # The name is read here, but the file is replaced only after this
            # block, where the context manager has already closed and flushed
            # it. Write-temp-then-rename is the whole point.
            # nosemgrep: tempfile-without-flush
            tmp_path = f.name
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(
            "[active-eval] failed to record pid=%d for project=%s: %s",
            pid, Path(project_dir).name[:12], e,
        )


def clear_active_eval(project_dir: Path) -> None:
    """Remove the active-eval record after an eval terminates.

    Called from every terminal exit path (complete / fail / cancel /
    unhandled) so the on-disk PID never outlives the real process. A stale
    record risks recovery SIGKILLing an unrelated process that recycled the
    same PID — though ``_kill_orphan_eval_pid`` guards that with a
    ``/proc/<pid>/cmdline`` check, clearing promptly is the first line of
    defense.

    Best-effort and idempotent: a missing record is a no-op.

    Args:
        project_dir: The project directory whose record should be cleared.
    """
    path = _record_path(project_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(
            "[active-eval] failed to clear record for project=%s: %s",
            Path(project_dir).name[:12], e,
        )


def load_active_eval(project_dir: Path) -> ActiveEvalRecord | None:
    """Load the active-eval record for a project, or None if absent/unreadable.

    Used by the startup recovery hook to find orphan harness subprocesses.

    Args:
        project_dir: The project directory to read from.

    Returns:
        The parsed :class:`ActiveEvalRecord`, or None when no record exists
        or the file is malformed (treated as "nothing to recover").
    """
    path = _record_path(project_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ActiveEvalRecord(
            pid=int(data["pid"]),
            started_at=str(data.get("started_at", "")),
            job_id=str(data.get("job_id", "")),
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
        logger.warning(
            "[active-eval] ignoring unreadable record for project=%s: %s",
            Path(project_dir).name[:12], e,
        )
        return None
