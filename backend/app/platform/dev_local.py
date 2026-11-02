# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dev-mode platform adapters — backed by the local filesystem.

Each adapter implements its Protocol by delegating to the
``ProjectManager`` / ``OptimizationState`` / ``project_paths`` code, so call
sites see only the Protocol and run unchanged in dev and cloud.

Multi-tenancy in dev: ``user_id`` is accepted by every method but
ignored (single-user local). The fixed ``LOCAL_DEV_USER_ID`` mirrors
what the Auth dev adapter injects, so logs and DynamoDB-shaped data
still carry a sensible value.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from .auth import Auth, get_local_dev_user_id
from .metadata_store import (
    ConcurrentUpdateError,
    MetadataStore,
    check_project_settings,
)
from .session_store import SessionStoreProtocol
from .token_vending import ScopedCredentials, TokenVending
from .types import (
    AccessAction,
    JobId,
    ProjectId,
    TurnId,
    UserId,
)

logger = logging.getLogger(__name__)

# Default root directory. Tests override ``app.main.PROJECTS_DIR`` to a
# tmp_path fixture; the dev adapters resolve through ``_get_projects_dir``
# below so they pick up that override at call time rather than freezing
# the value at import.
DEFAULT_PROJECTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "projects"


def _get_projects_dir() -> Path:
    """Return the active projects directory.

    Reads ``app.main.PROJECTS_DIR`` so test fixtures that swap the
    constant transparently affect the dev adapters. Falls back to
    :data:`DEFAULT_PROJECTS_DIR` if ``app.main`` isn't importable yet
    (e.g. early in module load).
    """
    try:
        from app.main import PROJECTS_DIR as runtime_dir
        return runtime_dir
    except ImportError:
        return DEFAULT_PROJECTS_DIR


# --- ProjectStore (removed) ------------------------------------------
# Project files are read/written directly via real ``Path`` APIs
# (``app.services.project_paths.resolve_project_dir``) in both dev and
# cloud, so there is no filesystem ``ProjectStore`` adapter.


# --- MetadataStore ---------------------------------------------------


class FilesystemMetadataStore(MetadataStore):
    """JSON-file-backed :class:`MetadataStore` for dev mode.

    Project records: ``projects/{guid}/config.json``, carrying the
    project's ``owner_id``.

    Job, turn, optimization, and cost records live under
    ``projects/{guid}/.platform/{kind}.json`` so dev mode doesn't need
    a database. This is intentionally minimal — fancy queries (active
    turns across all projects, etc.) iterate the projects directory.
    Cloud mode uses DynamoDB; nobody calls these dev paths in production.
    """

    def __init__(self, projects_dir: Path | None = None):
        self._explicit_projects_dir = projects_dir
        # Per-(project_id, job_id) locks for claim_completion_fired concurrency
        self._claim_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._claim_locks_lock = asyncio.Lock()  # Protects the locks dict itself
        # Global seq allocator lock (serialize concurrent allocate_job_seq calls)
        self._seq_allocator_lock = asyncio.Lock()
        # Global event log lock (serialize concurrent append_job_event writes)
        self._event_log_lock = asyncio.Lock()
        # Launch gates — keyed by (project_id, job_id), value is the winning claimant
        self._launch_gates: dict[tuple[str, str], str] = {}
        self._launch_gates_lock = asyncio.Lock()
        # Eval concurrency locks — keyed by project_id
        self._eval_locks: dict[str, dict[str, Any]] = {}
        self._eval_lock_versions: dict[str, int] = {}  # Monotonic version counter
        self._eval_locks_lock = asyncio.Lock()
        # Idempotency keys — keyed by (owner_id, idempotency_key)
        self._idempotency_keys: dict[tuple[str, str], tuple[str, float]] = {}  # (job_id, expiry_ts)
        self._idempotency_lock = asyncio.Lock()
        # AgentCore session registry — keyed by (user_id, project_id)
        self._agentcore_sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._agentcore_claim_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._agentcore_claim_locks_lock = asyncio.Lock()
        self._agentcore_claim_versions: dict[tuple[str, str], int] = {}  # Monotonic version counter

    @property
    def projects_dir(self) -> Path:
        if self._explicit_projects_dir is not None:
            return self._explicit_projects_dir
        return _get_projects_dir()

    # ----- Project records ----

    def _config_path(self, project_id: ProjectId) -> Path:
        return self.projects_dir / project_id / "config.json"

    def _platform_path(self, project_id: ProjectId, kind: str) -> Path:
        return self.projects_dir / project_id / ".platform" / f"{kind}.json"

    def _global_platform_path(self, kind: str) -> Path:
        """Global platform file (not scoped to a project)."""
        return self.projects_dir / ".platform" / f"{kind}.json"

    async def get_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        path = self._config_path(project_id)
        if not path.exists():
            return None

        def _read() -> dict[str, Any] | None:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            return data

        return await asyncio.to_thread(_read)

    async def list_user_projects(self, user_id: UserId) -> list[dict[str, Any]]:
        if not self.projects_dir.exists():
            return []

        def _list() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for cfg in sorted(self.projects_dir.glob("*/config.json")):
                try:
                    data = json.loads(cfg.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if data.get("owner_id") == user_id:
                    results.append(data)
            return results

        return await asyncio.to_thread(_list)

    async def find_project_by_id(
        self, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the project record by id (dev: read on-disk config)."""
        path = self._config_path(project_id)
        if not path.exists():
            return None

        def _read() -> dict[str, Any] | None:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            return data

        return await asyncio.to_thread(_read)

    async def put_project(self, project: dict[str, Any]) -> None:
        if "id" not in project or "owner_id" not in project:
            raise ValueError("project record missing id/owner_id")
        path = self._config_path(project["id"])

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(project, indent=2))

        await asyncio.to_thread(_write)

    async def update_project_settings(
        self, owner_id: UserId, project_id: ProjectId, settings: dict[str, Any]
    ) -> bool:
        """Set only the named settings in the project's config.json (dev)."""
        check_project_settings(settings)
        path = self._config_path(project_id)
        if not path.exists():
            return False

        def _update() -> None:
            data = json.loads(path.read_text())
            data.update(settings)
            path.write_text(json.dumps(data, indent=2))

        await asyncio.to_thread(_update)
        return True

    async def delete_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> bool:
        path = self.projects_dir / project_id
        if not path.exists():
            return False
        await asyncio.to_thread(shutil.rmtree, path)
        return True

    # ----- Job records ----

    async def _read_kind(
        self, project_id: ProjectId, kind: str
    ) -> dict[str, Any]:
        path = self._platform_path(project_id, kind)

        def _read() -> dict[str, Any]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}

        return await asyncio.to_thread(_read)

    async def _write_kind(
        self, project_id: ProjectId, kind: str, data: dict[str, Any]
    ) -> None:
        path = self._platform_path(project_id, kind)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def get_job(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId, *, consistent: bool = False
    ) -> dict[str, Any] | None:
        # Dev mode: in-process/synchronous, always consistent. Accept kwarg for parity.
        jobs = await self._read_kind(project_id, "jobs")
        return jobs.get(job_id)

    async def list_jobs(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        jobs = await self._read_kind(project_id, "jobs")
        return sorted(
            jobs.values(),
            key=lambda j: j.get("created_at", ""),
            reverse=True,
        )

    async def list_jobs_by_owner(self, owner_id: UserId) -> list[dict[str, Any]]:
        """Return all dev-store job records for the owner across their projects.

        Dev mode: iterate all projects under projects_dir, read each project's
        jobs.json, filter to the owner, return newest first.
        """

        def _gather() -> list[dict[str, Any]]:
            all_jobs = []
            # Iterate all project directories
            if not self.projects_dir.exists():
                return []
            for project_path in self.projects_dir.iterdir():
                if not project_path.is_dir():
                    continue
                # Skip .platform (global files)
                if project_path.name.startswith("."):
                    continue
                jobs_path = project_path / ".platform" / "jobs.json"
                if not jobs_path.exists():
                    continue
                try:
                    import json

                    jobs_data = json.loads(jobs_path.read_text())
                    for job in jobs_data.values():
                        # Filter to this owner
                        job_owner = job.get("owner_id") or job.get("user_id")
                        if job_owner == owner_id:
                            all_jobs.append(job)
                except (OSError, json.JSONDecodeError):
                    continue
            return sorted(
                all_jobs, key=lambda j: j.get("created_at", ""), reverse=True
            )

        import asyncio

        return await asyncio.to_thread(_gather)

    async def put_job(self, job: dict[str, Any]) -> None:
        if "job_id" not in job or "project_id" not in job:
            raise ValueError("job record missing job_id/project_id")
        jobs = await self._read_kind(job["project_id"], "jobs")
        jobs[job["job_id"]] = job
        await self._write_kind(job["project_id"], "jobs", jobs)

    async def update_job_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        state: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        compute_handle: str | None = None,
        output_file: str | None = None,
    ) -> None:
        jobs = await self._read_kind(project_id, "jobs")
        job = jobs.get(job_id)
        if job is None:
            return
        if state is not None:
            job["state"] = state
        if progress is not None:
            # Write progress top-level to match the cloud adapter
            job["progress"] = progress
        if result is not None:
            job["result"] = result
        if compute_handle is not None:
            job["compute_handle"] = compute_handle
        if output_file is not None:
            job["output_file"] = output_file
        await self._write_kind(project_id, "jobs", jobs)

    async def mark_cancel_requested(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Stamp ``cancel_requested_at`` on the job record, if there is one."""
        jobs = await self._read_kind(project_id, "jobs")
        job = jobs.get(job_id)
        if job is None:
            return
        job["cancel_requested_at"] = datetime.now(UTC).isoformat()
        await self._write_kind(project_id, "jobs", jobs)

    async def claim_completion_fired(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Claim the exactly-once completion turn (dev in-process first-wins).

        Dev mode: read the job record, check if ``completion_fired`` is already
        set; if so return False (loser). Otherwise set it (+ timestamp), persist,
        return True (winner). Single-process first-wins parity with cloud
        conditional write.

        The claim is an ATTRIBUTE on the EXISTING job record, not a
        separate claim item — no orphan cleanup.

        Concurrent access is serialized via per-(project_id, job_id) asyncio.Lock
        so N concurrent claims on the same job yield exactly one True (concurrent
        claims on different jobs proceed in parallel).
        """
        from datetime import datetime, timezone

        # Get or create a lock for this (project_id, job_id) pair
        async with self._claim_locks_lock:
            key = (project_id, job_id)
            if key not in self._claim_locks:
                self._claim_locks[key] = asyncio.Lock()
            lock = self._claim_locks[key]

        # Serialize access to this job's claim
        async with lock:
            jobs = await self._read_kind(project_id, "jobs")
            job = jobs.get(job_id)
            if job is None:
                # Job doesn't exist — can't claim (treat as loser)
                return False

            # Check if already claimed
            if job.get("completion_fired"):
                return False

            # Claim it
            job["completion_fired"] = True
            job["completion_fired_at"] = datetime.now(timezone.utc).isoformat()
            await self._write_kind(project_id, "jobs", jobs)
            return True

    # --- Launch gate ---------------------------------

    async def claim_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> str:
        """First-wins claim on a job's launch gate (dev in-process).

        Dev evals run as a local subprocess rather than a detached worker, so the
        race this settles cannot occur here. It is implemented anyway so the seam
        has one contract: a caller that has to ask which mode it is in before
        trusting an answer is how the dual-path bugs in this codebase started.
        """
        async with self._launch_gates_lock:
            return self._launch_gates.setdefault((project_id, job_id), claimant)

    async def release_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> bool:
        """Drop this claimant's gate claim (dev). Never removes another's."""
        async with self._launch_gates_lock:
            if self._launch_gates.get((project_id, job_id)) != claimant:
                return False
            del self._launch_gates[(project_id, job_id)]
            return True

    # --- Eval concurrency lock -----------------------

    async def claim_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Claim eval lock (dev in-process first-wins).

        Dev mode: in-process dict keyed by project_id. First-wins, stores
        holder+version (monotonic counter) if absent → True, else False.
        Single-process safe.

        The lock_version is a monotonic counter that survives release/reclaim
        cycles (fencing token for reconciler-clear).
        """
        import time

        async with self._eval_locks_lock:
            if project_id in self._eval_locks:
                # Lock already held
                return False

            # Increment version counter (survives release)
            current_version = self._eval_lock_versions.get(project_id, 0)
            new_version = current_version + 1
            self._eval_lock_versions[project_id] = new_version

            # Claim it
            self._eval_locks[project_id] = {
                "lock_holder_job_id": job_id,
                "lock_version": new_version,
                "lock_acquired_at": int(time.time()),
            }
            return True

    async def release_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Release eval lock (dev idempotent).

        Deletes only if holder matches. Idempotent no-op if not held by this
        job or already gone.
        """
        async with self._eval_locks_lock:
            lock = self._eval_locks.get(project_id)
            if lock and lock.get("lock_holder_job_id") == job_id:
                del self._eval_locks[project_id]

    async def get_eval_lock(
        self, owner_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the eval lock or None (dev).

        Returns a dict with keys: lock_holder_job_id, lock_version,
        lock_acquired_at.
        """
        async with self._eval_locks_lock:
            lock = self._eval_locks.get(project_id)
            if lock:
                return dict(lock)  # Return a copy
            return None

    async def clear_stale_eval_lock(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        expected_version: int,
        expected_holder_job_id: JobId,
    ) -> bool:
        """Clear stale lock, fenced on version AND holder (dev).

        Returns True if deleted, False if either fence tripped. Deletes only if
        the lock still has the expected version AND is still held by the job the
        reconciler decided was stale — a lock reclaimed by a newer job must
        survive, or a third eval can start against the same project.
        """
        async with self._eval_locks_lock:
            lock = self._eval_locks.get(project_id)
            if (
                lock
                and lock.get("lock_version") == expected_version
                and lock.get("lock_holder_job_id") == expected_holder_job_id
            ):
                del self._eval_locks[project_id]
                return True
            return False

    # --- AgentCore session registry ---

    async def get_agentcore_session(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the AgentCore session or None (dev).

        Returns a dict with keys: runtime_session_id, app_session_id, status,
        updated_at, claim_version, ttl.
        """
        key = (user_id, project_id)
        async with self._agentcore_claim_locks_lock:
            session = self._agentcore_sessions.get(key)
            if session:
                return dict(session)  # Return a copy
            return None

    async def put_agentcore_session(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        runtime_session_id: str | None,
        app_session_id: str | None,
        ttl: int,
    ) -> None:
        """Upsert runtime/app session ids without touching status (dev).

        Creates the item at status="idle" if absent. Does NOT overwrite
        status if present.
        """
        import time

        key = (user_id, project_id)
        async with self._agentcore_claim_locks_lock:
            if key not in self._agentcore_sessions:
                # Create with idle status
                self._agentcore_sessions[key] = {
                    "runtime_session_id": runtime_session_id,
                    "app_session_id": app_session_id,
                    "status": "idle",
                    "updated_at": int(time.time()),
                    "claim_version": 0,
                    "ttl": ttl,
                }
            else:
                # Update identity fields without touching status. Only
                # overwrite runtime_session_id / app_session_id when non-None,
                # so recording a mapping never nulls out a good stored id.
                session = self._agentcore_sessions[key]
                if runtime_session_id is not None:
                    session["runtime_session_id"] = runtime_session_id
                if app_session_id is not None:
                    session["app_session_id"] = app_session_id
                session["updated_at"] = int(time.time())
                session["ttl"] = ttl

    async def claim_agentcore_session_turn(
        self, user_id: UserId, project_id: ProjectId, *, ttl: int, current_turn_id: str | None = None
    ) -> bool:
        """Claim session turn (dev in-process first-wins).

        Dev mode: per-(user,project) lock + dict. First-wins, stores
        holder+version (monotonic counter) if status=idle → True, else False.
        Single-process safe.

        The claim_version is a monotonic counter that survives release/reclaim
        cycles (fencing token for reconciler-clear).
        """
        import time

        key = (user_id, project_id)
        async with self._agentcore_claim_locks_lock:
            session = self._agentcore_sessions.get(key)

            # Check if already running. An EXPIRED claim is treated as free, matching
            # the cloud conditional write's `#ttl < :now` arm: DynamoDB TTL deletion
            # is best-effort and lags expiry by up to days (AWS-documented), so a
            # leaked claim must not block its (user, project) pair until the item is
            # physically deleted. Keeping the two stores' claim semantics identical is
            # the point of this branch — a dev-only "still held" answer would hide the
            # cloud behavior the tests are meant to pin.
            if session and session.get("status") == "running":
                claim_ttl = session.get("ttl")
                if claim_ttl is None or int(claim_ttl) >= int(time.time()):
                    return False

            # Increment version counter (survives release)
            current_version = self._agentcore_claim_versions.get(key, 0)
            new_version = current_version + 1
            self._agentcore_claim_versions[key] = new_version

            # Create or update session to running
            if key not in self._agentcore_sessions:
                self._agentcore_sessions[key] = {
                    "runtime_session_id": None,
                    "app_session_id": None,
                    "status": "running",
                    "updated_at": int(time.time()),
                    "claim_version": new_version,
                    "ttl": ttl,
                    "current_turn_id": current_turn_id,
                }
            else:
                session = self._agentcore_sessions[key]
                session["status"] = "running"
                session["claim_version"] = new_version
                session["updated_at"] = int(time.time())
                session["ttl"] = ttl
                session["current_turn_id"] = current_turn_id

            return True

    async def release_agentcore_session_turn(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        expected_turn_id: str | None = None,
    ) -> None:
        """Release session turn claim (dev idempotent).

        Sets status to idle. Idempotent no-op if not running, absent, or held by
        a different turn than ``expected_turn_id`` — see the cloud store for why
        that last case matters.
        """
        import time

        key = (user_id, project_id)
        async with self._agentcore_claim_locks_lock:
            session = self._agentcore_sessions.get(key)
            if not session or session.get("status") != "running":
                return
            if (
                expected_turn_id is not None
                and session.get("current_turn_id") != expected_turn_id
            ):
                return
            session["status"] = "idle"
            session["updated_at"] = int(time.time())
            session["ttl"] = int(time.time()) + 86400  # +24h

    async def list_agentcore_sessions(self, user_id: UserId) -> list[dict[str, Any]]:
        """List all AgentCore sessions for a user (dev).

        Returns all sessions where user_id matches.
        """
        async with self._agentcore_claim_locks_lock:
            results = []
            for (uid, pid), session in self._agentcore_sessions.items():
                if uid == user_id:
                    results.append(dict(session))
            return results

    async def list_running_agentcore_sessions(self) -> list[dict[str, Any]]:
        """List ALL running AgentCore sessions across users (dev).

        Returns all sessions with status="running" across all users. On a
        real dev restart the dict is empty (dev orphans cannot survive a
        restart), so this effectively returns [] at startup — kept for
        Protocol parity and testability.
        """
        async with self._agentcore_claim_locks_lock:
            results = []
            for (uid, pid), session in self._agentcore_sessions.items():
                if session.get("status") == "running":
                    # Include user_id and project_id in the result
                    session_dict = dict(session)
                    session_dict["user_id"] = uid
                    session_dict["project_id"] = pid
                    results.append(session_dict)
            return results

    async def clear_stale_agentcore_session_claim(
        self, user_id: UserId, project_id: ProjectId, expected_version: int
    ) -> bool:
        """Clear stale claim with version fence (dev).

        Returns True if cleared, False if claim_version changed. Sets status to
        idle only if claim_version == expected_version.
        """
        import time

        key = (user_id, project_id)
        async with self._agentcore_claim_locks_lock:
            session = self._agentcore_sessions.get(key)
            if session and session.get("claim_version") == expected_version:
                session["status"] = "idle"
                session["updated_at"] = int(time.time())
                return True
            return False

    # --- Dispatch idempotency + reconciler helpers ---

    async def claim_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId, ttl_seconds: int
    ) -> tuple[bool, str | None]:
        """Claim idempotency key (dev in-process first-wins with TTL).

        Returns (claimed, existing_job_id): (True, None) when this call freshly
        claimed the key, (False, <first job's job_id>) when a prior dispatch
        already claimed it within TTL.

        In-process dict keyed (owner_id, idempotency_key) mapping to
        (job_id, expiry_ts); first-wins, expired entries treated as absent.
        """
        import time

        async with self._idempotency_lock:
            now = time.time()
            key = (owner_id, idempotency_key)

            # Check if key exists and is not expired
            if key in self._idempotency_keys:
                stored_job_id, expiry_ts = self._idempotency_keys[key]
                if now < expiry_ts:
                    # Key is still valid — return existing job_id
                    return (False, stored_job_id)
                # Key expired — remove it and treat as fresh
                del self._idempotency_keys[key]

            # Claim it
            self._idempotency_keys[key] = (job_id, now + ttl_seconds)
            return (True, None)

    async def release_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId
    ) -> bool:
        """Release idempotency key (dev in-process with ownership guard).

        Releases the key ONLY if the stored job_id matches the caller's job_id.
        Returns False if the stored job_id does not match (ownership guard tripped)
        or if the key does not exist.

        Under lock: delete (owner_id, key) from _idempotency_keys dict ONLY if
        the stored job_id matches.
        """
        async with self._idempotency_lock:
            key = (owner_id, idempotency_key)
            if key in self._idempotency_keys:
                stored_job_id, _expiry_ts = self._idempotency_keys[key]
                if stored_job_id == job_id:
                    # Ownership matches — release the key
                    del self._idempotency_keys[key]
                    return True
                # Ownership guard: stored job_id does not match
                return False
            # Key doesn't exist (already released or never claimed)
            return False

    async def count_active_jobs_for_owner(self, owner_id: UserId) -> int:
        """Count non-terminal jobs for owner (dev).

        Returns count of jobs in states (queued, running, awaiting_user).
        Terminal states (completed, failed, cancelled) are excluded.

        Calls list_jobs_by_owner and filters to non-terminal.
        """
        jobs = await self.list_jobs_by_owner(owner_id)
        non_terminal_states = {"queued", "running", "awaiting_user"}
        return sum(1 for job in jobs if job.get("state") in non_terminal_states)

    async def list_non_terminal_jobs(self) -> list[dict[str, Any]]:
        """List all non-terminal jobs across users (dev).

        Returns jobs in states (running, queued). Used by the reconciler,
        the account backstop cap, and startup reattach.

        Dev: read every project dir's job records and filter to
        non-terminal; single-process so a simple scan is fine.
        """
        # In dev mode, we don't have multi-tenancy, so we just scan all projects
        projects_dir = self.projects_dir
        results: list[dict[str, Any]] = []

        if not projects_dir.exists():
            return results

        for project_path in projects_dir.iterdir():
            if not project_path.is_dir():
                continue
            project_id = project_path.name

            # Read jobs for this project
            jobs = await self._read_kind(project_id, "jobs")
            for job in jobs.values():
                if job.get("state") in ("running", "queued"):
                    results.append(job)

        return results

    async def list_recent_completed_jobs(
        self, within_seconds: int = 86400
    ) -> list[dict[str, Any]]:
        """List recent terminal jobs for completion turn recovery.

        Returns jobs in states (completed, failed) created within the recency window.
        EXCLUDES cancelled jobs (no turn owed).

        Dev: iterate over all projects and filter to completed/failed within window.
        """
        from datetime import datetime, timezone
        import time

        now = int(time.time())
        since_ts = now - within_seconds

        projects_dir = self.projects_dir
        results: list[dict[str, Any]] = []

        if not projects_dir.exists():
            return results

        for project_path in projects_dir.iterdir():
            if not project_path.is_dir():
                continue
            project_id = project_path.name

            # Read jobs for this project
            jobs = await self._read_kind(project_id, "jobs")
            for job in jobs.values():
                state = job.get("state")
                if state not in ("completed", "failed"):
                    # Exclude cancelled and non-terminal
                    continue

                created_at_iso = job.get("created_at")
                if not created_at_iso:
                    continue

                try:
                    created_at = datetime.fromisoformat(created_at_iso)
                    job_ts = int(created_at.timestamp())
                except (ValueError, TypeError):
                    continue

                if job_ts >= since_ts:
                    results.append(job)

        return results

    # ----- Turn records ----

    async def get_turn(
        self, user_id: UserId, project_id: ProjectId, turn_id: TurnId
    ) -> dict[str, Any] | None:
        turns = await self._read_kind(project_id, "turns")
        return turns.get(turn_id)

    async def list_active_turns(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        turns = await self._read_kind(project_id, "turns")
        return [t for t in turns.values() if t.get("state") in ("running", "interrupted")]

    async def put_turn(self, turn: dict[str, Any]) -> None:
        if "turn_id" not in turn or "project_id" not in turn:
            raise ValueError("turn record missing turn_id/project_id")
        turns = await self._read_kind(turn["project_id"], "turns")
        turns[turn["turn_id"]] = turn
        await self._write_kind(turn["project_id"], "turns", turns)

    async def update_turn_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        turn_id: TurnId,
        *,
        state: str | None = None,
        last_heartbeat: str | None = None,
        completed_at: str | None = None,
        interrupt_reason: str | None = None,
    ) -> None:
        turns = await self._read_kind(project_id, "turns")
        turn = turns.get(turn_id)
        if turn is None:
            return
        if state is not None:
            turn["state"] = state
        if last_heartbeat is not None:
            turn["last_heartbeat"] = last_heartbeat
        if completed_at is not None:
            turn["completed_at"] = completed_at
        if interrupt_reason is not None:
            turn["interrupt_reason"] = interrupt_reason
        await self._write_kind(project_id, "turns", turns)

    # ----- Optimization state ----

    async def get_optimization_state(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        # Delegate to the existing on-disk JSON file from
        # ``OptimizationState.save`` so this dev adapter is read-through.
        from app.services.optimization_state import OPTIMIZATION_STATE_FILE
        path = self.projects_dir / project_id / OPTIMIZATION_STATE_FILE
        if not path.exists():
            return None

        def _read() -> dict[str, Any] | None:
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None

        return await asyncio.to_thread(_read)

    async def put_optimization_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        state: dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> int:
        # Dev mode: ``OptimizationState`` already serializes via per-file
        # threading.Lock. We mirror its behavior here for callers that go
        # through the Protocol directly. If ``expected_version`` is set,
        # we honour it for consistency with the cloud contract.
        from app.services.optimization_state import OPTIMIZATION_STATE_FILE

        path = self.projects_dir / project_id / OPTIMIZATION_STATE_FILE

        def _write() -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            current_version = 0
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                    current_version = existing.get("version", 0)
                except (OSError, json.JSONDecodeError):
                    pass
            if expected_version is not None and current_version != expected_version:
                raise ConcurrentUpdateError(
                    f"optimization state version mismatch: expected={expected_version} actual={current_version}"
                )
            new_version = current_version + 1
            payload = {**state, "version": new_version}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, path)
            return new_version

        return await asyncio.to_thread(_write)

    # ----- Event log ----

    async def _read_global_kind(self, kind: str) -> dict[str, Any]:
        """Read global platform file (not project-scoped)."""
        path = self._global_platform_path(kind)

        def _read() -> dict[str, Any]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}

        return await asyncio.to_thread(_read)

    async def _write_global_kind(self, kind: str, data: dict[str, Any]) -> None:
        """Write global platform file (not project-scoped)."""
        path = self._global_platform_path(kind)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def allocate_job_seq(self, job_id: JobId) -> int:
        """Allocate the next event seq (in-process max+1, single-process safe).

        Dev mode: uses a separate "seq_counters" file to track per-job seq values.
        Read current counter, increment, persist, return. This mirrors DynamoDB's
        atomic counter behavior in single-process dev mode.

        serialize concurrent calls with a lock so concurrent appends from
        multiple asyncio tasks don't race on read-modify-write or mkdir.
        """
        async with self._seq_allocator_lock:
            counters = await self._read_global_kind("seq_counters")
            current = counters.get(job_id, 0)
            next_seq = current + 1
            counters[job_id] = next_seq
            await self._write_global_kind("seq_counters", counters)
            return next_seq

    # ----- Project event outbox — interface parity with cloud ----------
    # Dev has no microVM, so nothing writes this in normal dev operation; the
    # implementation exists so the Protocol is fully satisfied and the drain
    # logic is testable against the dev store.

    async def put_outbox_event(
        self, owner_id: UserId, project_id: ProjectId, event: dict[str, Any]
    ) -> None:
        from app.services.project_event_outbox import outbox_sk

        async with self._event_log_lock:
            data = await self._read_global_kind("event_outbox")
            items = data.setdefault(owner_id, [])
            items.append({
                "SK": outbox_sk(project_id),
                "event": event,
                "project_id": project_id,
                "ttl": int(time.time()) + 3600,
            })
            await self._write_global_kind("event_outbox", data)

    async def list_outbox_events(
        self, owner_id: UserId, project_id: ProjectId, limit: int = 50
    ) -> list[dict[str, Any]]:
        data = await self._read_global_kind("event_outbox")
        items = [
            it for it in data.get(owner_id, [])
            if it.get("SK", "").startswith(f"OUTBOX#{project_id}#")
        ]
        items.sort(key=lambda it: it.get("SK", ""))
        return items[:limit]

    async def delete_outbox_event(self, owner_id: UserId, sk: str) -> None:
        async with self._event_log_lock:
            data = await self._read_global_kind("event_outbox")
            items = data.get(owner_id, [])
            data[owner_id] = [it for it in items if it.get("SK") != sk]
            await self._write_global_kind("event_outbox", data)

    async def append_job_event(
        self, job_id: JobId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one event to the dev JSON store.

        serialize concurrent appends with a lock so background tasks
        from concurrent Job.append_event calls don't race on read-modify-write.
        """
        async with self._event_log_lock:
            events_data = await self._read_global_kind("events")
            job_events = events_data.setdefault(job_id, [])
            job_events.append({"seq": seq, "event": event, "ttl": ttl})
            await self._write_global_kind("events", events_data)

    async def events_since(
        self, job_id: JobId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Return events with seq > since_seq, ordered ascending.

        Returns the event dicts as stored; the job writer stores the published
        envelope, which carries its own ``seq``.
        """
        events_data = await self._read_global_kind("events")
        job_events = events_data.get(job_id, [])

        # Filter and sort
        filtered = [e for e in job_events if e["seq"] > since_seq]
        filtered.sort(key=lambda e: e["seq"])
        return [e["event"] for e in filtered]

    async def allocate_turn_seq(self, turn_id: TurnId) -> int:
        """Allocate the next event seq for a turn (in-process max+1, single-process safe).

        Dev mode: uses a separate "turn_seq_counters" file to track per-turn seq values.
        Read current counter, increment, persist, return. This mirrors DynamoDB's
        atomic counter behavior in single-process dev mode.

        serialize concurrent calls with a lock so concurrent appends from
        multiple asyncio tasks don't race on read-modify-write.
        """
        async with self._seq_allocator_lock:
            counters = await self._read_global_kind("turn_seq_counters")
            current = counters.get(turn_id, 0)
            next_seq = current + 1
            counters[turn_id] = next_seq
            await self._write_global_kind("turn_seq_counters", counters)
            return next_seq

    async def append_turn_event(
        self, turn_id: TurnId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one turn event to the dev JSON store.

        serialize concurrent appends with a lock so background tasks
        from concurrent append calls don't race on read-modify-write.
        """
        async with self._event_log_lock:
            events_data = await self._read_global_kind("turn_events")
            turn_events = events_data.setdefault(turn_id, [])
            turn_events.append({"seq": seq, "event": event, "ttl": ttl})
            await self._write_global_kind("turn_events", events_data)

    async def turn_events_since(
        self, turn_id: TurnId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Return turn events with seq > since_seq, ordered ascending.

        Returns the event dicts as stored; the turn writers store the published
        envelope, which carries its own ``seq``.
        """
        events_data = await self._read_global_kind("turn_events")
        turn_events = events_data.get(turn_id, [])

        # Filter and sort
        filtered = [e for e in turn_events if e["seq"] > since_seq]
        filtered.sort(key=lambda e: e["seq"])
        return [e["event"] for e in filtered]

    async def allocate_project_seq(self, project_id: ProjectId) -> int:
        """Allocate the next event seq for a project stream (cross-instance safe).

        Dev mode: uses a separate "project_seq_counters" file to track per-project
        seq values. Read current counter, increment, persist, return. This mirrors
        DynamoDB's atomic counter behavior and is cross-instance safe via file
        read/write (two separate FilesystemMetadataStore objects reading the same
        projects_dir will continue the same counter).

        Reuses the existing _seq_allocator_lock; no new lock needed.
        """
        async with self._seq_allocator_lock:
            counters = await self._read_global_kind("project_seq_counters")
            current = counters.get(project_id, 0)
            next_seq = current + 1
            counters[project_id] = next_seq
            await self._write_global_kind("project_seq_counters", counters)
            return next_seq

    async def put_turn_metadata(
        self, turn_id: TurnId, *, user_id: UserId, project_id: ProjectId, ttl: int
    ) -> None:
        """Store turn metadata in dev JSON store."""
        metadata = await self._read_global_kind("turn_metadata")
        metadata[turn_id] = {
            "user_id": user_id,
            "project_id": project_id,
            "ttl": ttl,
        }
        await self._write_global_kind("turn_metadata", metadata)

    async def get_turn_metadata(self, turn_id: TurnId) -> dict[str, Any] | None:
        """Retrieve turn metadata from dev JSON store."""
        metadata = await self._read_global_kind("turn_metadata")
        return metadata.get(turn_id)

    # ----- Connection registry ----

    async def put_connection(
        self,
        connection_id: str,
        user_id: UserId,
        *,
        project_id: ProjectId | None = None,
        ttl: int,
    ) -> None:
        """Persist connection to dev JSON store."""
        connections = await self._read_global_kind("connections")
        connections[connection_id] = {
            "connection_id": connection_id,
            "user_id": user_id,
            "ttl": ttl,
        }
        if project_id is not None:
            connections[connection_id]["project_id"] = project_id
        await self._write_global_kind("connections", connections)

    async def delete_connection(self, connection_id: str) -> None:
        """Remove connection from dev JSON store (no-op if missing)."""
        connections = await self._read_global_kind("connections")
        connections.pop(connection_id, None)
        await self._write_global_kind("connections", connections)

    async def list_connections_for_user(
        self, user_id: UserId
    ) -> list[dict[str, Any]]:
        """Return all connections for a user (filter by user_id)."""
        connections = await self._read_global_kind("connections")
        return [
            conn for conn in connections.values() if conn.get("user_id") == user_id
        ]

    # ----- Cost tracking ----

    async def add_bedrock_cost(
        self,
        user_id: UserId,
        *,
        year_month: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        # Dev mode: append to a single per-user JSON file. Single-process,
        # so a plain file write is fine; cloud uses an UpdateItem ADD.
        path = self.projects_dir.parent / ".platform" / "costs" / f"{user_id}.json"

        def _add() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    data = {}
            row = data.get(year_month, {
                "bedrock_input_tokens": 0,
                "bedrock_output_tokens": 0,
                "total_cost_usd": 0.0,
            })
            row["bedrock_input_tokens"] += input_tokens
            row["bedrock_output_tokens"] += output_tokens
            row["total_cost_usd"] += cost_usd
            data[year_month] = row
            path.write_text(json.dumps(data, indent=2))

        await asyncio.to_thread(_add)

    async def get_monthly_costs(
        self, user_id: UserId, year_month: str
    ) -> dict[str, Any] | None:
        path = self.projects_dir.parent / ".platform" / "costs" / f"{user_id}.json"
        if not path.exists():
            return None

        def _read() -> dict[str, Any] | None:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            return data.get(year_month)

        return await asyncio.to_thread(_read)


# --- TokenVending ----------------------------------------------------


class NoOpTokenVending(TokenVending):
    """Dev TokenVending — always returns ``None`` (use ambient creds)."""

    async def mint(
        self,
        user_id: UserId,
        project_id: ProjectId,
        cwd: str,
        *,
        ttl_sec: int = 3600,
        job_id: str | None = None,
    ) -> ScopedCredentials | None:
        return None


# --- Auth ------------------------------------------------------------


class LocalAuth(Auth):
    """Dev :class:`Auth` — single user, no JWT verification.

    Reads ``LOCAL_DEV_USER_ID`` from the environment for ``extract_user_id``
    so the rest of the codebase looks identical to cloud mode.
    """

    def __init__(self, metadata_store: MetadataStore):
        self.metadata_store = metadata_store

    async def extract_user_id(self, request: Request) -> UserId:
        return get_local_dev_user_id()

    async def extract_user_id_from_token(self, token: str) -> UserId | None:
        """Dev mode ignores the token and returns the configured local user.

        Used by the WebSocket route. The token is accepted but never
        validated — dev mode is single-user and JWT-free.
        """
        return get_local_dev_user_id()

    async def require_access(
        self,
        user_id: UserId,
        project_id: ProjectId,
        action: AccessAction,
    ) -> None:
        project = await self.metadata_store.find_project_by_id(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        owner_id = project.get("owner_id")
        if owner_id == user_id:
            return
        raise HTTPException(status_code=403, detail="Forbidden")


# --- SessionStore ----------------------------------------------------


def _build_dev_session_store() -> SessionStoreProtocol | None:
    """Return ``None`` to keep the SDK's default on-disk session store.

    Dev mode is single-host and the SDK already persists sessions to
    ``~/.claude/projects``. Returning ``None`` here means the
    ``ClaudeAgentOptions`` instantiations in
    :mod:`app.services.agent_runner` and friends should NOT pass
    ``session_store=`` in dev — the SDK falls back to its built-in
    behavior. Cloud mode swaps in :class:`S3SessionStore`.
    """
    return None


# --- Bootstrap -------------------------------------------------------


def build_dev_platform():
    """Construct the dev-mode :class:`Platform` (called by ``bootstrap``)."""
    from . import Platform

    metadata = FilesystemMetadataStore()
    auth = LocalAuth(metadata)
    return Platform(
        metadata_store=metadata,
        session_store=_build_dev_session_store(),
        token_vending=NoOpTokenVending(),
        auth=auth,
    )


# --- JobCompute ------------------------------------------------------


class LocalSubprocessCompute:
    """Dev-mode :class:`JobCompute` impl — jobs run in-process, no snapshot.

    The local worktree already IS the working tree, so `launch` does NOT
    snapshot. Jobs spawn via ``asyncio.create_task`` and register in the
    in-process ``JobTracker`` (same as the in-process flow). The PID is
    the cancel handle, read from the job's ``.pid`` field once the harness
    subprocess spawns (PID not known at launch return time).

    Cancel is process-group SIGTERM→SIGKILL via ``harness_runner.kill_harness_subprocess``.
    Status reads the job's current state from the ``JobTracker``.

    No AWS/boto3 import is reachable from this class — dev adapter stays
    AWS-dep-free so tests run without mocking AWS.
    """

    async def launch(
        self,
        job_id: JobId,
        owner_id: UserId,
        project_id: ProjectId,
    ):
        """Return a handle wrapping the job_id (dev-mode no-op).

        The job must already exist in the ``JobTracker`` (created by the
        dispatch site before calling ``JobCompute.launch``). This impl
        returns a handle wrapping the job_id — the actual job execution
        happens on the old in-process path (asyncio.create_task in the
        dispatch site when arch_split is False, which is always in dev
        because resolve_arch_split() forces False outside cloud mode).

        WHY this is a no-op: the local worktree already IS the tree, and
        dev is cloud-gated off the new path, so this method is never the
        dev execution surface. The handle allows dev-mode tests to pass
        the JobCompute Protocol contract without spawning a real worker.
        """
        from .job_compute import JobComputeHandle

        # For dev mode, the handle wraps the job_id
        # The actual PID is recorded on the job later
        return JobComputeHandle(kind="local_pid", value=job_id)

    async def cancel(self, handle) -> None:
        """Cancel a running job via process-group SIGTERM→SIGKILL.

        Reads the job's current PID from the ``JobTracker`` and calls
        ``harness_runner.kill_harness_subprocess`` (process-group
        SIGTERM→SIGKILL). If no PID yet, cancels the asyncio task.
        """
        # Import here to avoid circular dependency
        from app.services.jobs import get_job_tracker

        tracker = get_job_tracker()
        # handle.value is the job_id for LocalSubprocessCompute
        job = tracker.get_job(handle.value)
        if job is None:
            return

        # If harness PID is known, kill the subprocess
        pid = getattr(job, "pid", None)
        if pid:
            from app.services.harness_runner import kill_harness_subprocess

            kill_harness_subprocess(pid)
        # Otherwise cancel the asyncio task
        else:
            task = getattr(job, "_task", None)
            if task:
                task.cancel()

    async def status(self, handle) -> str:
        """Return the job's current state from the JobTracker."""
        from app.services.jobs import get_job_tracker

        tracker = get_job_tracker()
        job = tracker.get_job(handle.value)
        if job is None:
            return "unknown"
        return job.state


__all__ = [
    "FilesystemMetadataStore",
    "LocalAuth",
    "LocalSubprocessCompute",
    "NoOpTokenVending",
    "build_dev_platform",
]
