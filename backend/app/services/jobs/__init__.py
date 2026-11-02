# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job abstraction for long-running background work.

Provides the ``Job`` dataclass, ``JobTracker`` singleton, and ``JobExecutor``
Protocol used by every background job. All job types (eval, harness_build,
optimization_iteration) share the same lifecycle, event buffering, and
completion-turn dispatch.

All executors run as asyncio tasks in-process on the coordinator.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Literal, Protocol

from app.services.resource_error import ResourceError

logger = logging.getLogger(__name__)

_MAX_EVENT_BUFFER = 10_000

# retain durable-append background tasks so they aren't GC'd mid-flight
_durable_append_tasks: set = set()

# retain fire-and-forget WS publish tasks so they aren't GC'd mid-flight
_ws_publish_tasks: set = set()

JobType = Literal["eval", "harness_build", "optimization_iteration"]
JobState = Literal[
    "queued", "running", "completed", "failed", "cancelled", "awaiting_user"
]


def is_loop_owned(job_or_config: "Job | dict[str, Any]") -> bool:
    """Check if a job belongs to the optimization loop.

    A job whose config.source == "optimization" is owned by the loop — the loop
    reports iteration completion, so the job itself must NOT fire a separate
    completion turn. This predicate is the single source of truth for that rule,
    used by both the reconciler and the poller startup paths.

    Implements the same check present in:
    - ``job_progress_republisher.adopt_unpolled_jobs`` (``fire_completion =
      source != "optimization"``)
    - ``optimization_loop_orchestrator._reap_loop_child_eval`` (the child-eval
      cancel gate)

    Args:
        job_or_config: A Job instance or a config dict.

    Returns:
        True if the job is loop-owned (must not fire its own completion turn).
    """
    if isinstance(job_or_config, dict):
        config = job_or_config
    else:
        config = job_or_config.config
    return config.get("source") == "optimization"


@dataclass
class JobResult:
    """Outcome of a job executor run.

    Attributes:
        status: Terminal status the executor produced.
        data: Type-specific payload (eval results, harness validation, etc.).
        error: Structured error if the job failed.
    """

    status: Literal["completed", "failed"]
    data: dict[str, Any] = field(default_factory=dict)
    error: ResourceError | None = None


class JobExecutor(Protocol):
    """Protocol for type-specific job execution."""

    async def run(self, job: "Job") -> JobResult:
        """Execute a job to completion."""
        ...


def _default_progress() -> dict[str, Any]:
    """Default progress shape consumed by the eval dashboard + ActiveJobBar.

    Eval jobs are the only consumers that populate this structured
    payload; harness_build and optimization_iteration jobs leave the
    fields at their defaults.
    """
    return {
        "current_model": None,
        "models_completed": 0,
        "models_total": 0,
        "current_case": 0,
        "cases_total": 0,
        "message": "Pending",
    }


@dataclass
class Job:
    """A single background job with an event buffer.

    Attributes:
        job_id: Unique identifier for this job.
        project_id: Project GUID this job belongs to.
        type: Job type discriminator.
        config: Type-specific payload (model IDs, prompt version, etc.).
        state: Current lifecycle state.
        progress: Free-form progress dict consumed by the eval-jobs UI.
        result: Type-specific result data on completion. For ``eval`` jobs
            this is the list of per-model run dicts.
        output_file: Path/filename of the saved results JSON for eval jobs.
        error: Structured error if the job failed.
        pause_reason: Human-readable question when state == "awaiting_user".
        created_at: ISO 8601 timestamp.
        completed_at: ISO 8601 timestamp when terminal state reached.
        pid: OS process ID for cancellation (eval subprocess).
    """

    job_id: str
    project_id: str
    type: JobType
    config: dict[str, Any] = field(default_factory=dict)
    state: JobState = "queued"
    progress: dict[str, Any] = field(default_factory=_default_progress)
    result: Any = None
    output_file: str | None = None
    error: ResourceError | None = None
    pause_reason: str | None = None
    created_at: str = ""
    completed_at: str | None = None
    pid: int | None = None
    # asyncio.Task handle for cancellation
    _task: asyncio.Task | None = field(default=None, repr=False)
    # In-memory event buffer for WS replay-on-subscribe.
    events: list[dict] = field(default_factory=list, repr=False)
    events_dropped: int = field(default=0, repr=False)
    _condition: asyncio.Condition | None = field(default=None, repr=False)

    @property
    def status(self) -> str:
        """API-facing status string consumed by the frontend serializer.

        The on-the-wire vocabulary uses ``"pending"`` for the initial
        state; internally we call that ``"queued"``. All other values
        pass through unchanged.
        """
        if self.state == "queued":
            return "pending"
        return self.state

    @property
    def condition(self) -> asyncio.Condition:
        """Lazy-initialized asyncio.Condition for event-buffer notification."""
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    @property
    def is_terminal(self) -> bool:
        """Whether the job is in a terminal state."""
        return self.state in ("completed", "failed", "cancelled")

    @property
    def is_active(self) -> bool:
        """Whether the job is still running or awaiting input."""
        return self.state in ("queued", "running", "awaiting_user")

    async def append_event(self, event: dict) -> None:
        """Append a progress event and fan it out to the project owner's WS.

        The buffer (``self.events``) is the source of truth for
        replay-on-reconnect — ``ws_hub.replay_to_connection`` filters
        the buffer on ``isinstance(env.get("seq"), int)`` so we MUST
        store the envelope (with seq), not the raw event dict. Storing
        raw events made every replay return zero events; the SPA
        subscribing mid-eval missed the ``scoring_start`` envelope
        that sets the denominator and the bar froze at "190/0" /
        "model 0 of N".

        Live publish uses the same envelope so live + replay paths
        send the SPA exactly the same shape.

        Args:
            event: Progress event dict (harness JSON shape:
                ``{"event": "scoring_start"|"results"|...}``).
        """
        from app.services.ws_envelope import make_envelope, make_envelope_with_seq, mark_degraded
        from app.services.ws_hub import get_ws_hub

        # ONE SEQ DOMAIN PER STREAM. On the durable path the DDB seq is
        # allocated FIRST and used for the live envelope, the in-memory replay
        # buffer, and the durable append — one number, three consumers. Previously
        # the live/buffered envelope used the in-memory counter while only the
        # durable record carried the DDB seq, so the client's single per-stream
        # cursor saw two disjoint domains and silently dropped frames from whichever
        # one lagged.
        #
        # Allocation happens before the buffer append because the buffered entry is
        # what ``replay_to_connection`` serves; a buffer holding in-memory seqs and a
        # durable log holding DDB seqs is the mismatch itself.
        arch_split = self.config.get("arch_split", False)
        durable_seq: int | None = None
        alloc_failed = False
        loop = None
        if arch_split:
            try:
                # No running loop — old-path fallback (dev/test). The stream
                # has no durable leg there, so the in-memory counter IS its
                # single seq domain: healthy, not degraded. Checked in its
                # OWN try so a store-raised RuntimeError below cannot be
                # misclassified as the benign no-loop case.
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if arch_split and loop is not None:
            try:
                from app.platform import get_platform
                store = get_platform().metadata_store
                # allocate synchronously in EMISSION order, so rapidly
                # emitted events get seqs matching emission rather than
                # task-schedule order.
                durable_seq = await store.allocate_job_seq(self.job_id)
            except Exception as e:  # noqa: BLE001
                # Allocation failed on a stream that HAS a durable domain:
                # this frame cannot join that domain, and minting an
                # in-memory seq would create a second, colliding domain the
                # client's cursor can't reconcile — frames silently dropped
                # as duplicates/stale. Publish DEGRADED instead: the client
                # bypasses its cursor for degraded envelopes and delivers
                # the payload unconditionally. Never drop the event.
                logger.warning(
                    "[job %s] durable seq allocation failed (%s) — publishing "
                    "this event degraded (client bypasses its replay cursor)",
                    self.job_id[:8], e,
                )
                durable_seq = None
                alloc_failed = True

        if durable_seq is None:
            envelope = make_envelope(
                stream=f"job:{self.job_id}", kind="progress", data=event,
            )
            if alloc_failed:
                envelope = mark_degraded(envelope)
        else:
            envelope = make_envelope_with_seq(
                stream=f"job:{self.job_id}", kind="progress", data=event,
                seq=durable_seq,
            )
        self.events.append(envelope)

        if durable_seq is not None and loop is not None:
            try:
                import time
                ttl = int(time.time()) + (7 * 24 * 3600)
                seq = durable_seq

                async def _durable_append(seq: int):
                    try:
                        # The SAME envelope that was published live and buffered for
                        # replay — one domain, so all three agree.
                        await store.append_job_event(self.job_id, seq, envelope, ttl=ttl)
                    except Exception as e:
                        # Durable-append failures must not throw into the agent loop.
                        # Log the error and continue — the live envelope still fans out.
                        logger.warning(
                            "[job %s] durable event append failed: %s",
                            self.job_id[:8], e
                        )
                # retain the task so it isn't GC'd mid-flight
                task = loop.create_task(_durable_append(seq))
                _durable_append_tasks.add(task)
                task.add_done_callback(_durable_append_tasks.discard)
            except RuntimeError:
                # No running loop — skip the durable append (old-path fallback)
                pass

        if len(self.events) > _MAX_EVENT_BUFFER:
            overflow = len(self.events) - _MAX_EVENT_BUFFER
            del self.events[:overflow]
            self.events_dropped += overflow
            if self.events_dropped == overflow or self.events_dropped % 1000 == 0:
                logger.warning(
                    "[job %s] event buffer overflow; dropped %d so far",
                    self.job_id[:8],
                    self.events_dropped,
                )
        async with self.condition:
            self.condition.notify_all()
        # Fan out to the owner's WS connections on the per-job stream so
        # ``useJobStream`` renders per-case progress in the SPA. Owner
        # lookup failure silently skips the publish — the buffer
        # survived and a REST refetch will pick up the latest state.
        try:
            from app.main import get_project_manager
            project = get_project_manager().get_project(self.project_id)
            owner_id = project.owner_id if project else None
        except Exception:
            owner_id = None
        if owner_id is None:
            return
        await get_ws_hub().publish_to_user(owner_id, envelope)

    async def notify_readers(self) -> None:
        """Wake all waiting event readers."""
        async with self.condition:
            self.condition.notify_all()

    async def wait_for_events(self, known_count: int, timeout: float = 15.0) -> bool:
        """Wait until new events are available or timeout expires.

        Args:
            known_count: Absolute count of events the caller has seen.
            timeout: Maximum seconds to wait.

        Returns:
            True if new events available, False on timeout.
        """
        async with self.condition:
            try:
                await asyncio.wait_for(
                    self.condition.wait_for(
                        lambda: self.events_dropped + len(self.events) > known_count
                    ),
                    timeout=timeout,
                )
                return True
            except asyncio.TimeoutError:
                return False


def _try_notify(job: Job) -> None:
    """Schedule event-reader notification if an event loop is running."""
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(job.notify_readers())
        # Retain the fire-and-forget task so it isn't GC'd mid-flight
        _ws_publish_tasks.add(task)
        task.add_done_callback(_ws_publish_tasks.discard)
    except RuntimeError:
        pass


def _publish_per_job_done(
    job: "Job",
    *,
    status: str,
    output_file: str | None = None,
    error: ResourceError | None = None,
) -> None:
    """Push a ``done`` envelope to the per-job WS stream.

    The per-job stream (``job:<job_id>``) is what ``useJobStream`` on the
    SPA subscribes to for live progress + finalization. Emitting a final
    ``done`` envelope here lets the hook flip the eval dashboard out of
    "running" without polling the REST list.

    Called from :meth:`JobTracker.update_state` for every terminal
    transition so cancel paths surface to the SPA — without this, the
    eval-job bar stayed wedged at "model 0 of N" with a live Stop button
    after a cancel because the in-memory tracker had moved on but no
    envelope ever reached the hook.

    Args:
        job: The Job that just terminated.
        status: ``"completed"``, ``"failed"``, or ``"cancelled"``
            (mirrored into envelope.data; the SPA's done handler
            branches on each of the three).
        output_file: Eval-shaped result file when status is ``"completed"``.
        error: Structured failure reason when status is ``"failed"``.
    """
    try:
        from app.main import get_project_manager
        from app.services.ws_envelope import make_envelope, make_envelope_with_seq, mark_degraded
        from app.services.ws_hub import get_ws_hub
        project = get_project_manager().get_project(job.project_id)
        owner_id = project.owner_id if project else None
    except Exception:
        return
    if owner_id is None:
        return
    payload: dict[str, Any] = {"status": status, "job_id": job.job_id}
    if output_file is not None:
        payload["output_file"] = output_file
    if error is not None:
        payload["error"] = error.to_dict()

    arch_split = job.config.get("arch_split", False)

    # ONE SEQ DOMAIN PER STREAM. ``job:<id>`` carries durable seqs on the
    # arch_split path, so this TERMINAL frame must too — publishing an in-memory seq
    # here put the most consequential frame on the stream into a second domain,
    # where the client drops it as a duplicate or as stale. A dropped `done` is the
    # eval bar wedged at "model 0 of N" with a live Stop button.
    #
    # This function is SYNC (called from JobTracker.update_state), so the allocation
    # cannot be awaited inline. It happens inside the same task that already carries
    # the publish — allocate, then publish, in one coroutine, so ordering holds.
    async def _allocate_and_publish() -> None:
        seq: int | None = None
        alloc_failed = False
        if arch_split:
            try:
                from app.platform import get_platform
                seq = await get_platform().metadata_store.allocate_job_seq(job.job_id)
            except Exception as e:  # noqa: BLE001
                # Never drop the terminal frame over an allocation failure —
                # but never mint an in-memory seq into the durable domain
                # either: the cursor comparison would drop the most
                # consequential frame on the stream as a duplicate/stale.
                # Degraded delivery bypasses the client cursor entirely, so
                # the frame is guaranteed to render.
                logger.warning(
                    "[job %s] durable seq allocation failed for the terminal frame "
                    "(%s) — publishing degraded (client bypasses its replay cursor)",
                    job.job_id[:8], e,
                )
                seq = None
                alloc_failed = True
        if seq is None:
            envelope = make_envelope(
                stream=f"job:{job.job_id}", kind="done", data=payload,
            )
            if alloc_failed:
                envelope = mark_degraded(envelope)
        else:
            envelope = make_envelope_with_seq(
                stream=f"job:{job.job_id}", kind="done", data=payload, seq=seq,
            )
        await get_ws_hub().publish_to_user(owner_id, envelope)

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_allocate_and_publish())
        # Retain the fire-and-forget task so it isn't GC'd mid-flight
        _ws_publish_tasks.add(task)
        task.add_done_callback(_ws_publish_tasks.discard)
    except RuntimeError:
        pass


def _publish_project_event(project_id: str, event: dict[str, Any]) -> None:
    """Fan out a project-scoped event to the project owner's WS connections.

    The job tracker's ``complete_job`` / ``fail_job`` paths used to
    publish on the (now-deleted) ``ProjectEventBus``; the WS hub is the
    replacement. Resolves ``owner_id`` from the project metadata before
    publishing so the hub can target the right user.

    Owner lookup failures are swallowed — the WS event is best-effort
    and the underlying job state is already persisted.

    **Execution-home routing.** In the AgentCore microVM the event is handed to
    the coordinator over the turn's SSE stream, and the coordinator publishes it
    on the project stream where the browser is listening. The local publish is
    still attempted and remains harmless (zero connections means zero fan-out).
    This does NOT suppress the local publish — a no-op publish is harmless in
    itself, and suppressing it would remove the only trace that the event was
    emitted at all.

    When the event cannot be routed (no live turn for the project), the AgentCore
    runtime stashes it in the durable project-event outbox, which the
    coordinator's reconciler drains within a pass, so it arrives late rather than
    never. If the owner cannot be resolved nothing can be stashed: a warning
    names the resource and type, and the event is lost. The local publish is
    still attempted in that case too: never suppressed, because a process-local hub with zero connections
    is indistinguishable from a process holding connections that happen not to be
    subscribed to this project, and the latter case is normal operation where
    suppressing the publish would be wrong.

    The routed path is deliberately quiet (``logger.debug`` below); see
    ``project_event_frame.forward_project_event_to_coordinator`` for why
    successful routing is not logged.

    Args:
        project_id: Project the event belongs to.
        event: Free-form event payload (``{"resource": "...", "type": "...", ...}``).
    """
    # Owner resolution FIRST: both the microVM outbox stash and the local
    # fan-out below key on it.
    try:
        from app.main import get_project_manager
        from app.services.ws_hub import get_ws_hub
        project = get_project_manager().get_project(project_id)
        owner_id = project.owner_id if project else None
    except Exception:
        owner_id = None

    try:
        from app.platform.process_role import is_agentcore_runtime

        if is_agentcore_runtime():
            from app.services.project_event_frame import forward_project_event_to_coordinator
            routed = forward_project_event_to_coordinator(project_id, event)
            if routed:
                logger.debug(
                    "[project-event] Routed via coordinator (resource=%s type=%s)",
                    event.get("resource"),
                    event.get("type"),
                )
            elif owner_id is not None:
                # Unroutable: no live turn to carry the event. Stash it
                # in the durable outbox — the coordinator's reconciler drains
                # it within a pass, so the event is LATE (≤~60s), not LOST.
                # Without the outbox such an event reaches nobody. Scheduled
                # fire-and-forget, same retention pattern as the local publish
                # below.
                from app.services.project_event_outbox import stash_unroutable_event
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        stash_unroutable_event(owner_id, project_id, event)
                    )
                    _ws_publish_tasks.add(task)
                    task.add_done_callback(_ws_publish_tasks.discard)
                except RuntimeError:
                    pass  # no loop (sync test context) — same drop as local publish
            else:
                logger.warning(
                    "[project-event] Unroutable microVM event AND owner "
                    "unresolvable (project=%s resource=%s type=%s) — cannot "
                    "stash, event is lost.",
                    project_id[:12], event.get("resource"), event.get("type"),
                )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break the publish
        # Logged rather than swallowed: a bare ``pass`` here would make a thrown
        # forwarding block indistinguishable from a forward that simply had
        # nothing to route. A diagnostic must never break the publish, so this
        # only logs.
        logger.warning(
            "[project-event] Exception in AgentCore routing check/forward: %s",
            exc,
        )

    if owner_id is None:
        return

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(get_ws_hub().publish_project(project_id, owner_id, event))
        # Retain the fire-and-forget task so it isn't GC'd mid-flight
        _ws_publish_tasks.add(task)
        task.add_done_callback(_ws_publish_tasks.discard)
    except RuntimeError:
        # No running loop (sync test context) — drop the event. The
        # tracker state is already persisted; the SPA refetches via
        # REST on its next page load.
        pass


class JobNotFoundError(Exception):
    """Raised when a job_id cannot be resolved for a given project.

    Distinguishes "wrong project" from "really gone" at the type level
    while keeping both with the same outer error message — never leak
    existence of one user's jobs to another via 404 vs 403.
    """


class JobTracker:
    """Singleton in-memory store for all background jobs.

    Covers eval, harness_build, and optimization_iteration jobs behind
    one interface. Thread-safe under the GIL for dict pointer swaps.
    """

    _instance: "JobTracker | None" = None
    _jobs: dict[str, Job]
    _pending_writes: set[asyncio.Task]

    def __new__(cls) -> "JobTracker":
        """Ensure only one instance exists."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._jobs = {}
            instance._pending_writes = set()
            cls._instance = instance
        return cls._instance

    def create_job(
        self,
        project_id: str,
        job_type: "JobType | None" = None,
        *,
        config: dict[str, Any] | None = None,
        job_id: str | None = None,
        durable_write: bool = True,
    ) -> Job:
        """Create and register a new job in ``queued`` state.

        Args:
            project_id: Project GUID.
            job_type: ``JobType`` literal — required. Accepted positionally
                (``create_job("p", "eval", config=...)``) or as a kwarg.
            config: Type-specific configuration payload. Eval jobs expect
                ``{"model_ids": [...], "prompt_version": "..."}`` so the
                queued-progress message renders correctly.
            job_id: Optional explicit job_id (used by tests that need a
                stable id; production callers let the tracker mint a UUID).
            durable_write: When True (default), the ``arch_split`` create-time
                ``put_job`` write-through fires so the durable record is
                persisted. When False, the in-memory Job and the replay
                provider are registered EXACTLY as usual but NO ``put_job``
                write-through fires. This is for the detached Fargate worker:
                it registers the SAME job_id the coordinator already owns, and
                a full-item ``put_job`` from the worker would clobber the
                coordinator's durable record (wiping ``compute_handle``,
                re-freezing ``state`` to ``queued``, and replacing config with
                the worker's sparse shape). Clean ownership: the coordinator
                owns the durable job record; the worker owns the event log.
                Only affects the ``put_job`` write-through — the worker's
                event-log + result writes (via ``update_state`` / ``append_event``)
                are unaffected.
        """
        if job_type is None:
            raise TypeError("create_job requires job_type=")
        # Resolve owner_id at create-time so the replay provider is
        # registered with a fixed owner. Job.append_event also looks up
        # the owner per-event for live fan-out; the values are the same.
        from app.services.agent_turn import _resolve_owner_id_for_project
        owner_id = _resolve_owner_id_for_project(project_id)

        resolved_config: dict[str, Any] = config or {}
        initial_progress = _default_progress()
        if job_type == "eval":
            model_ids = list(resolved_config.get("model_ids", []) or [])
            prompt_version = resolved_config.get("prompt_version", "") or ""
            initial_progress["models_total"] = len(model_ids)
            initial_progress["message"] = (
                f"Queued: {len(model_ids)} model(s), prompt {prompt_version}"
            )

        job = Job(
            job_id=job_id or str(uuid.uuid4()),
            project_id=project_id,
            type=job_type,
            config=resolved_config,
            state="queued",
            progress=initial_progress,
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        self._jobs[job.job_id] = job

        # Write-through to durable store on the new path.
        # The arch_split boolean is threaded from the operation entry point
        # via the job config (never re-resolved). When True,
        # schedule a background write so the job record persists. The ordering
        # invariant: create_job's put_job is enqueued before any update_job_state
        # for this job, so DDB never receives an update on a never-put job
        # so a late publish cannot resurrect a finished job.
        # ``durable_write`` gates ONLY this create-time put_job write-through.
        # The worker passes durable_write=False so it never full-item put_jobs
        # over the coordinator's record (the confirmed cause of the null
        # compute_handle on optimization eval records).
        arch_split = resolved_config.get("arch_split", False)
        if arch_split and durable_write:
            try:
                loop = asyncio.get_running_loop()
                from app.platform import get_platform
                store = get_platform().metadata_store
                # Build the job dict for put_job
                job_dict = {
                    "owner_id": owner_id,
                    "project_id": project_id,
                    "job_id": job.job_id,
                    "type": job_type,
                    "state": job.state,
                    "progress": job.progress,
                    "created_at": job.created_at,
                    "config": resolved_config,
                }
                # Wrap in try/except to log failures and retain the task
                async def _write_through():
                    try:
                        await store.put_job(job_dict)
                    except Exception:
                        logger.exception("[job %s] durable put_job failed", job.job_id[:8])

                task = loop.create_task(_write_through())
                self._pending_writes.add(task)
                task.add_done_callback(self._pending_writes.discard)
            except RuntimeError:
                # No running loop (create_job called off-loop) — skip the durable
                # write and fall back to old-path behavior. This is rare, and it is
                # the same get_running_loop guard ``_try_notify`` uses.
                pass

        # Register as a replay provider so a SPA reconnecting mid-job
        # gets the buffered progress envelopes via the WS subscribe-
        # with-replay path. ``owner_id`` is captured here so the hub
        # can reject cross-tenant replay subscribes (``stream`` is
        # client-supplied).
        from app.services.ws_hub import get_ws_hub
        get_ws_hub().register_replay_provider(
            f"job:{job.job_id}", job, owner_id=owner_id,
        )
        self._evict_old_jobs(max_terminal=50)
        return job

    # Eval-only convenience for the coordinator-prompt summary and the
    # eval cancel route. New call sites that care about other job types
    # should use :meth:`get_active_jobs` or :meth:`get_active_job_by_type`.
    def get_active_job(self, project_id: str) -> Job | None:
        """Return the most-recent active *eval* job for ``project_id``."""
        return self.get_active_job_by_type(project_id, "eval")

    def get_job(self, job_id: str) -> Job | None:
        """Retrieve a job by ID."""
        return self._jobs.get(job_id)

    def get_active_jobs(self, project_id: str) -> list[Job]:
        """Get all active (non-terminal) jobs for a project.

        Args:
            project_id: Project GUID.

        Returns:
            List of active jobs, newest first.
        """
        active = [
            j for j in self._jobs.values()
            if j.project_id == project_id and j.is_active
        ]
        return sorted(active, key=lambda j: j.created_at, reverse=True)

    def get_active_job_by_type(
        self, project_id: str, job_type: JobType
    ) -> Job | None:
        """Get the latest active job of a specific type for a project.

        Args:
            project_id: Project GUID.
            job_type: Job type to filter by.

        Returns:
            The most recent active job of that type, or None.
        """
        active = [
            j for j in self._jobs.values()
            if j.project_id == project_id
            and j.type == job_type
            and j.is_active
        ]
        if not active:
            return None
        return max(active, key=lambda j: j.created_at)

    def list_jobs(self, project_id: str) -> list[Job]:
        """List all jobs for a project, newest first.

        Args:
            project_id: Project GUID.

        Returns:
            All jobs for the project, sorted by creation time descending.
        """
        project_jobs = [
            j for j in self._jobs.values()
            if j.project_id == project_id
        ]
        return sorted(project_jobs, key=lambda j: j.created_at, reverse=True)

    async def rebuild_jobs_from_store(
        self, project_id: str, arch_split: bool, *, refresh_terminal: bool = False
    ) -> None:
        """Rebuild jobs from durable store for missing cache entries.

        On the new path (arch_split=True), read jobs from the durable store
        (DynamoDB) and reconstruct any MISSING Job dataclasses from stored metadata,
        repopulating the in-memory cache (self._jobs). Jobs already in the cache
        are left untouched (cache-as-truth for live jobs; store fills gaps only).

        **refresh_terminal mode (opt-in, replica readers only):** When
        ``refresh_terminal=True`` and a job is already cached, refresh the cached
        Job if and only if the stored state is terminal (completed/failed/cancelled)
        and the cached state is not. Terminal state is a one-way door — a durable
        terminal state can never be a lagging read, so this asymmetry preserves
        cache-as-truth for live jobs while letting the durable store win on terminal
        state. This mode exists for any reader whose cached entry is a replica it
        did not execute (agent-facing tools in the microVM, and the coordinator's
        activity view for a job dispatched elsewhere). Cache-as-truth applies to
        jobs the reading process launched itself — those are live executing objects
        whose in-memory state is authoritative.

        **Scope edge:** Rebuild reconstructs job METADATA ONLY (id, project_id,
        owner_id, type, state, progress, timestamps, config). It does NOT:
        - Relaunch an eval subprocess
        - Reattach to a worker
        - Resume in-flight work
        - Register replay providers for rebuilt jobs

        Rebuild is a cold-start metadata-recovery path only — it makes the job
        record queryable, but does NOT resume the execution.

        Args:
            project_id: Project GUID.
            arch_split: Whether the project uses the new durable-state path.
                        If False, this method is a no-op (old path untouched).
            refresh_terminal: If True, update cached non-terminal jobs when the
                              durable state is terminal. Default False (coordinator
                              path — cache-as-truth).

        **Thread-safety / cross-tenant safety:** The store reads are
        keyed by project_id (the tenant fence) and server-resolved owner_id
        (_resolve_owner_id_for_project — same path create_job uses, never
        client-supplied). REST routes gate on require_project_read before calling
        rebuild, so the caller is authorized to read the project's jobs. JobTracker._jobs
        is a process-global singleton; rebuilt jobs land in a shared cache keyed by
        job_id. get_job(job_id) later returns any cached job whose project_id matches,
        regardless of which partition (owner) the job was rebuilt from. This is acceptable
        because project_id is the real tenant boundary and REST authorization already
        checked the caller's access to the project.
        """
        if not arch_split:
            # Old path: no store calls, zero-cost no-op
            return

        from app.platform import get_platform
        from app.services.agent_turn import _resolve_owner_id_for_project

        # Resolve owner_id for the project (the same resolution create_job uses)
        owner_id = _resolve_owner_id_for_project(project_id)

        store = get_platform().metadata_store
        stored_jobs = await store.list_jobs(owner_id, project_id)

        # Terminal states tuple (matches Job.is_terminal property)
        TERMINAL_STATES = ("completed", "failed", "cancelled")

        for job_dict in stored_jobs:
            job_id = job_dict["job_id"]

            # If job already exists in the cache, conditionally refresh if enabled
            if job_id in self._jobs:
                cached_job = self._jobs[job_id]
                stored_state = job_dict.get("state", "queued")

                # refresh_terminal mode: update cached job only when stored state
                # is terminal and cached state is not (one-way door — terminal
                # state can never be a lagging read)
                if refresh_terminal and stored_state in TERMINAL_STATES and not cached_job.is_terminal:
                    # Mutate cached job in place with direct field assignment.
                    # Do NOT use update_state() — it performs a durable write-through,
                    # fires terminal fan-out (_publish_per_job_done + project event),
                    # and releases the eval lock. Those are write side effects that
                    # must never be triggered by an agent read path.
                    cached_job.state = stored_state
                    cached_job.completed_at = job_dict.get("completed_at")
                    # Coalesce progress/output_file: prefer store value, but keep
                    # cached value if store is missing (never blank a populated field)
                    cached_job.progress = job_dict.get("progress") or cached_job.progress
                    cached_job.output_file = job_dict.get("output_file") or cached_job.output_file
                    # Do NOT copy result or error from the store. The durable job row
                    # deliberately carries neither (result blob excluded for DDB item cap;
                    # there is no error column), so copying would erase good in-memory data.
                continue

            # Reconstruct the Job dataclass from stored metadata (metadata
            # ONLY, no subprocess relaunch, no worker reattach, no replay provider).
            # The stored job_dict has: owner_id, project_id, job_id, type, state,
            # progress, created_at, config, and (if terminal) result, completed_at.
            # Use .get() with defaults for fields that may be missing in partial rows
            # (update-before-put can create a partial item with state/progress
            # but no type/created_at).
            self._jobs[job_id] = Job(
                job_id=job_id,
                project_id=job_dict["project_id"],
                type=job_dict.get("type", "unknown"),
                config=job_dict.get("config", {}),
                state=job_dict.get("state", "queued"),
                progress=job_dict.get("progress", {"message": "Unknown"}),
                result=job_dict.get("result"),
                created_at=job_dict.get("created_at", "1970-01-01T00:00:00Z"),
                completed_at=job_dict.get("completed_at"),
                # no subprocess, no worker, no replay provider for rebuilt jobs
            )

    def update_state(
        self,
        job_id: str,
        state: JobState,
        *,
        result: dict[str, Any] | None = None,
        error: ResourceError | None = None,
        pause_reason: str | None = None,
    ) -> None:
        """Transition a job to a new state.

        Args:
            job_id: Job to update.
            state: New state.
            result: Optional result data (typically on completion).
            error: Optional structured error (on failure).
            pause_reason: Question text (when state == "awaiting_user").

        Raises:
            KeyError: If job_id not found.

        Note:
            The durable write-through to DynamoDB (when arch_split=True)
            persists ONLY pointers (state, progress, output_file) — the
            result blob is intentionally NOT persisted. This is because
            the materialized file on EFS/S3 is the authoritative source
            of truth for per-case eval results, and DynamoDB has a 400KB
            item cap that multi-MB result payloads exceed. The in-memory
            Job.result is preserved for the completion turn and
            get_results MCP tool to consume.
        """
        job = self._jobs[job_id]
        # Capture the prior state BEFORE the assignment so the terminal-
        # transition fan-out below fires exactly once per job — even if a
        # caller redundantly transitions an already-terminal job. The
        # optimization-executor cancel path does this: it calls
        # ``cancel_job`` (which routes through update_state) and then
        # ``update_state(..., "cancelled")`` again on the same id, and
        # without this guard each pair of calls would emit two ``done``
        # envelopes and two ``eval_job`` project events.
        was_terminal = job.is_terminal
        job.state = state
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        if pause_reason is not None:
            job.pause_reason = pause_reason

        # Write-through to durable store on the new path.
        # Mirror the state transition to DDB. The arch_split boolean is threaded
        # from the job config (set at create_job time). Schedule the update
        # via the same loop.create_task seam so it's enqueued after put_job.
        arch_split = job.config.get("arch_split", False)
        if arch_split:
            try:
                loop = asyncio.get_running_loop()
                from app.platform import get_platform
                store = get_platform().metadata_store
                # Resolve the durable OWNER partition key. Optimization-path
                # evals carry config.owner_id (no user_id); manual-path evals
                # carry config.user_id. Coalesce owner_id first so BOTH paths
                # mirror terminal state — previously the user_id-only guard
                # silently skipped the optimization path, leaving the record
                # frozen at queued. Mirrors the same coalesce in
                # hard_cancel_project_eval and the reconciler.
                owner_key = job.config.get("owner_id") or job.config.get("user_id")
                tracker = get_job_tracker()
                if owner_key:
                    # Thread output_file into durable write-through so cross-process
                    # Job rebuilds (poller, reconciler) can render "Results file:" in the seed
                    output_file_val = job.output_file if hasattr(job, "output_file") else None
                    # Wrap in try/except to log failures and retain the task
                    async def _write_through():
                        try:
                            await store.update_job_state(
                                owner_key,
                                job.project_id,
                                job_id,
                                state=state,
                                progress=job.progress,
                                output_file=output_file_val,
                            )
                        except Exception:
                            logger.exception("[job %s] durable update_job_state failed", job_id[:8])

                    task = loop.create_task(_write_through())
                    tracker._pending_writes.add(task)
                    task.add_done_callback(tracker._pending_writes.discard)
            except RuntimeError:
                # No running loop — skip the durable write (old-path fallback)
                pass

        if state in ("completed", "failed", "cancelled") and not was_terminal:
            job.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            # Single chokepoint for terminal-transition fan-out. Every
            # cancel path (JobTracker.cancel_job, hard_cancel_project_eval,
            # optimization_executor's update_state(..., "cancelled"/"failed"))
            # routes through here, so emitting the per-job ``done``
            # envelope at this layer guarantees the SPA's
            # ``useJobStream`` finalizes its state regardless of how the
            # job ended. Before this, only complete_job/fail_job
            # published — cancel paths went silent and the eval bar
            # stuck at "model 0 of N" with a live Stop button forever.
            _publish_per_job_done(
                job,
                status=state,
                output_file=job.output_file if state == "completed" else None,
                error=job.error if state == "failed" else None,
            )
            # Eval jobs also drive the SPA's project-scoped refetch path —
            # keep that signal here so a cancelled eval's "running" state
            # clears at once instead of waiting for the next backstop poll
            # (OptimizationStateProvider subscribes to every eval_job type).
            # Other job types (harness_build, optimization_iteration) don't
            # have a corresponding ``eval_job`` listener and would only
            # confuse the SPA.
            if job.type == "eval":
                event: dict[str, Any] = {
                    "resource": "eval_job",
                    "type": (
                        "completed" if state == "completed"
                        else "error" if state == "failed"
                        else "cancelled"
                    ),
                    "job_id": job_id,
                }
                if state == "completed" and job.output_file:
                    event["output_file"] = job.output_file
                if state == "failed" and job.error is not None:
                    event.update(job.error.to_dict())
                _publish_project_event(job.project_id, event)

            # Release eval lock on every terminal transition
            # This is the ONE chokepoint every terminal path routes through
            # (cancel_job, hard_cancel_project_eval, optimization_executor's
            # update_state, and normal completion). Releasing here ensures
            # the lock is freed on all paths (completed/failed/cancelled),
            # not just happy-path — avoiding the stale-phantom bug.
            # Only on the new path (arch_split=True) and when owner_id is
            # present. Because release_eval_lock is holder-conditional and
            # idempotent, releasing a lock this job never held
            # (e.g., an optimization-iteration job that isn't the project
            # lock holder) is a harmless no-op.
            arch_split = job.config.get("arch_split", False)
            # Resolve the SAME owner partition the claim used. dispatch.py claims
            # the lock under the resolved project OWNER, and the durable
            # write-through earlier in this method already coalesces
            # owner_id-then-user_id for exactly this reason. Reading only
            # user_id here released against the wrong partition for
            # optimization-path evals, which carry owner_id and no user_id.
            # release_eval_lock is
            # holder-conditional, so a wrong-partition release silently no-ops —
            # the real lock then survives to its 24h TTL and blocks every
            # subsequent dispatch on the project with no visible cause.
            owner_id = job.config.get("owner_id") or job.config.get("user_id")
            if arch_split and owner_id:
                try:
                    loop = asyncio.get_running_loop()
                    from app.platform import get_platform
                    store = get_platform().metadata_store
                    tracker = get_job_tracker()

                    async def _release_lock():
                        try:
                            await store.release_eval_lock(owner_id, job.project_id, job_id)
                        except Exception:
                            logger.exception("[job %s] durable release_eval_lock failed", job_id[:8])

                    task = loop.create_task(_release_lock())
                    tracker._pending_writes.add(task)
                    task.add_done_callback(tracker._pending_writes.discard)
                except RuntimeError:
                    # No running loop — skip the durable release (old-path fallback)
                    pass

            self._evict_old_jobs(max_terminal=50)
        _try_notify(job)

    def update_progress(self, job_id: str, *args: Any, **kwargs: Any) -> None:
        """Merge progress fields into ``job.progress`` and notify event readers.

        Two call shapes are supported:

        * ``update_progress(job_id, {"foo": 1, "bar": 2})`` — accepted for
          callers that already build a progress dict.
        * ``update_progress(job_id, foo=1, bar=2)`` — kwargs shape used
          throughout ``harness_runner``. Treated as a merge, not a replace.

        Transitions ``queued`` → ``running`` on the first update so the
        eval dashboard stops showing "Pending" the instant the harness
        produces output. On the durable path (arch_split=True), writes
        state="running" through to the store exactly once on the queued->running
        transition so a cross-process reader sees "running" (not stale "queued").
        The detached worker flips state via update_progress, not update_state, so
        without this the durable record stays frozen at queued while the harness runs.

        Progress alone is written through again each time ``models_completed``
        changes. Every other process reads a detached eval's progress from the
        durable record, and with only the first write it said "0 of N models" for
        the whole run — including when a cancel announced how far the eval got.
        Per model, not per case: a case update is a write per scored case.

        Raises:
            KeyError: ``job_id`` not found.
        """
        job = self._jobs[job_id]

        # Capture transition BEFORE mutating state (reading after makes it always False)
        transitioned_to_running = (job.state == "queued")
        models_completed_before = job.progress.get("models_completed")

        if transitioned_to_running:
            job.state = "running"

        merged: dict[str, Any] = {}
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update_progress takes at most one positional dict; got "
                    f"{len(args)}"
                )
            arg = args[0]
            if not isinstance(arg, dict):
                raise TypeError(
                    f"update_progress expected dict, got {type(arg).__name__}"
                )
            merged.update(arg)
        merged.update(kwargs)

        job.progress.update(merged)
        finished_a_model = (
            "models_completed" in merged
            and merged["models_completed"] != models_completed_before
        )

        # Write state=running through to durable store on queued->running transition,
        # and progress alone when a model finishes (arch_split-gated, owner_key
        # coalesced, no-loop/old-path safe)
        if transitioned_to_running or finished_a_model:
            arch_split = job.config.get("arch_split", False)
            if arch_split:
                # Resolve owner_key (the same coalesce update_state uses)
                owner_key = job.config.get("owner_id") or job.config.get("user_id")
                if owner_key:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running event loop — old-path fallback, skip durable write
                        pass
                    else:
                        # Schedule durable write-through for state=running + progress.
                        # A model boundary writes progress only: the write is
                        # backgrounded, and one landing after a terminal write must
                        # not put the job back to running.
                        from app.platform import get_platform
                        store = get_platform().metadata_store
                        state_to_write = "running" if transitioned_to_running else None

                        async def _write_through():
                            try:
                                await store.update_job_state(
                                    owner_key,
                                    job.project_id,
                                    job_id,
                                    state=state_to_write,
                                    progress=job.progress,
                                )
                            except Exception as e:
                                logger.exception(
                                    "[job %s] durable update_progress write failed",
                                    job_id[:8]
                                )

                        task = loop.create_task(_write_through())
                        # Retain on _pending_writes so drain_pending_writes awaits it
                        self._pending_writes.add(task)
                        task.add_done_callback(self._pending_writes.discard)

        _try_notify(job)

    def complete_job(
        self,
        job_id: str,
        result: Any,
        output_file: str,
    ) -> None:
        """Mark a job as completed with eval-shaped result data.

        Sets the result fields, then routes through :meth:`update_state`
        so the per-job ``done`` envelope and the project-scoped
        ``eval_job/completed`` event fan out from the single terminal
        chokepoint.

        Args:
            job_id: Job to complete.
            result: Per-model run dicts (eval) or any type-specific payload.
            output_file: Filename where results were saved.

        Raises:
            KeyError: ``job_id`` not found.
        """
        job = self._jobs[job_id]
        job.result = result
        job.output_file = output_file
        job.progress["message"] = "Completed"
        self.update_state(job_id, "completed")

    def fail_job(self, job_id: str, error: ResourceError) -> None:
        """Mark a job as failed with a structured error.

        Sets the error + progress message, then routes through
        :meth:`update_state` so the per-job ``done`` envelope and the
        project-scoped error event fan out from the single terminal
        chokepoint.

        Args:
            job_id: Job to fail.
            error: Structured failure reason.

        Raises:
            KeyError: ``job_id`` not found.
        """
        job = self._jobs[job_id]
        job.progress["message"] = f"Failed: {error.message}"
        self.update_state(job_id, "failed", error=error)

    def get_job_for_project(self, project_id: str, job_id: str) -> "Job":
        """Look up a job and verify it belongs to ``project_id``.

        Raises :class:`JobNotFoundError` for both "no such job" and
        "job exists but for a different project" so we don't leak
        existence information across project boundaries.
        """
        job = self.get_job(job_id)
        if job is None or job.project_id != project_id:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job

    def cancel_job(self, job_id: str) -> None:
        """Mark a job as cancelled and propagate cancellation to its workers.

        For eval jobs we also SIGTERM the harness subprocess group via
        :attr:`Job.pid`.

        Args:
            job_id: Job to cancel.

        Raises:
            KeyError: ``job_id`` not found.
        """
        job = self._jobs[job_id]
        self.update_state(job_id, "cancelled")
        job.progress["message"] = "Cancelled by user"

        # Cancel the asyncio task if we have a handle.
        if job._task is not None and not job._task.done():
            job._task.cancel()

        # For eval jobs, SIGTERM the harness subprocess group. The pid is
        # set by harness_runner once the subprocess spawns; if we cancel
        # before then, the post-spawn check in harness_runner observes
        # ``state == "cancelled"`` and kills the proc itself.
        if job.type == "eval" and job.pid:
            try:
                from app.services.harness_runner import kill_harness_subprocess
                loop = asyncio.get_running_loop()
                loop.create_task(kill_harness_subprocess(job.pid))
            except RuntimeError:
                # No running event loop (sync test context). The
                # subprocess will exit when its asyncio.Task gets the
                # cancellation propagated.
                pass

    async def drain_pending_writes(self) -> None:
        """Await all pending durable writes to completion (non-raising).

        Backgrounded durable writes (update_state → update_job_state,
        Job.append_event → append_job_event) are tracked in
        ``self._pending_writes`` and the module-level
        ``_durable_append_tasks`` so they can be awaited before the
        event loop closes. Without this drain, ``asyncio.run()`` cancels
        any unfinished tasks — dropping the terminal state/event that a
        client waiting on job completion depends on.

        Snapshots both sets FIRST into local lists to avoid "set changed
        size during iteration" (add_done_callback discards fire during
        the await). Uses ``return_exceptions=True`` so one failed write
        cannot abort the drain. Non-raising — failures are logged, not
        raised.

        Called by worker_entrypoint.run_worker after the harness returns
        (success or failure) before the event loop closes.
        """
        # Snapshot both sets first so add_done_callback discards during
        # the await cannot raise "set changed size during iteration"
        pending_state_writes = list(self._pending_writes)
        pending_event_writes = list(_durable_append_tasks)
        all_writes = pending_state_writes + pending_event_writes

        if not all_writes:
            return

        # Await all writes with return_exceptions so one failure doesn't
        # abort the drain. Failures are already logged by the background
        # tasks; no need to re-raise here.
        await asyncio.gather(*all_writes, return_exceptions=True)

    def _evict_old_jobs(self, max_terminal: int = 50) -> None:
        """Remove oldest terminal jobs when capacity is exceeded."""
        terminal = [
            (jid, j) for jid, j in self._jobs.items()
            if j.is_terminal
        ]
        if len(terminal) <= max_terminal:
            return
        terminal.sort(key=lambda x: x[1].created_at)
        to_evict = terminal[: len(terminal) - max_terminal]
        from app.services.ws_hub import get_ws_hub
        hub = get_ws_hub()
        for jid, _ in to_evict:
            hub.unregister_replay_provider(f"job:{jid}")
            del self._jobs[jid]

    def _reset(self) -> None:
        """Clear all jobs. For testing only."""
        self._jobs.clear()


def get_job_tracker() -> JobTracker:
    """Module-level singleton accessor for the JobTracker."""
    return JobTracker()


__all__ = [
    "Job",
    "JobExecutor",
    "JobNotFoundError",
    "JobResult",
    "JobState",
    "JobTracker",
    "JobType",
    "get_job_tracker",
]
