# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetadataStore Protocol — projects, jobs, optimization state, costs.

Cloud implementation is single-table DynamoDB. Dev implementation
reads/writes JSON files on disk, matching the on-disk project layout.

All GSI queries on the cloud implementation MUST go through the
DynamoDB store module (the GSI chokepoint rule). A grep test
(``tests/test_no_raw_gsi_queries.py``) enforces no raw ``IndexName=``
calls outside ``app/platform/aws/dynamodb_metadata_store.py``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import JobId, ProjectId, TurnId, UserId


@runtime_checkable
class MetadataStore(Protocol):
    """Project metadata, job records, optimization state, costs, connections.

    The Protocol is split into logical groups by docstring section. Cloud
    and dev implementations are free to denormalize internally as long
    as the contracts hold.
    """

    # --- Project records -------------------------------------------------

    async def get_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the project record or ``None`` if missing.

        The dict shape mirrors :class:`app.models.schemas.ProjectConfig`,
        including the ownership field ``owner_id``.
        """
        ...

    async def list_user_projects(self, user_id: UserId) -> list[dict[str, Any]]:
        """List projects the user owns. Cheap summary, not full records."""
        ...

    async def put_project(self, project: dict[str, Any]) -> None:
        """Create or replace a project record. ``project["owner_id"]`` required.

        Raises:
            ProjectOwnerConflictError: the id already belongs to another owner;
                nothing is written.
        """
        ...

    async def update_project_settings(
        self, owner_id: UserId, project_id: ProjectId, settings: dict[str, Any]
    ) -> bool:
        """Set a project's user-editable settings, and nothing else.

        Only keys in :data:`PROJECT_SETTINGS_FIELDS` are accepted. The owner is
        never written here: it is fixed when the project is created, so a
        settings edit cannot move the project to another owner, whatever the
        caller's copy of the project says.

        Returns:
            True if the record existed and was updated, False if there is none.

        Raises:
            ValueError: a key outside :data:`PROJECT_SETTINGS_FIELDS`.
        """
        ...

    async def delete_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> bool:
        """Delete a project record. Returns ``True`` if a row was deleted."""
        ...

    async def find_project_by_id(
        self, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return a project record by id, regardless of owner.

        Used by ``Auth.require_access``, which is given a project id and a
        caller who may not be the owner, so it cannot know the owner's
        partition key up front.

        Cloud implementation: reads the ``PROJECT_INDEX#{id}`` item to find
        the owner, then the owner's record (no GSI, no scan); dev
        implementation: reads the on-disk config.json directly.

        Returns the same dict shape as :meth:`get_project`, or ``None``
        if no project with that id exists.
        """
        ...

    # --- Job records (eval, harness_build, optimization_iteration) ------

    async def get_job(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId, *, consistent: bool = False
    ) -> dict[str, Any] | None:
        """Return the job record or ``None``.

        Args:
            user_id: Job owner's user_id.
            project_id: Project UUID.
            job_id: Job UUID.
            consistent: If True, perform a strongly-consistent read (cloud mode uses
                ConsistentRead=True for DynamoDB GetItem; dev mode is always consistent).
                Use consistent=True when reading cross-process terminal jobs to avoid
                reading stale data (eventual consistency window).

        Returns:
            Job record dict or None if not found.
        """
        ...

    async def list_jobs(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        """List all job records for a project, newest first."""
        ...

    async def list_jobs_by_owner(self, owner_id: UserId) -> list[dict[str, Any]]:
        """List all job records for an owner across ALL projects, newest first.

        Returns all of an owner's jobs from all of their projects (not scoped to a
        single project). Callers filter ``is_active`` client-side for "active
        jobs by owner"; the store returns all jobs (active + terminal).

        Cloud mode: base-table query on ``PK=USER#{owner_id}`` with JOB-item
        filter (no GSI — the owner IS the base-table partition). Dev mode:
        return all dev-store job records for the owner across their projects.
        """
        ...

    async def put_job(self, job: dict[str, Any]) -> None:
        """Create or replace a job record."""
        ...

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
        """Partial update of a job record. Only non-None fields are written.

        Args:
            compute_handle: the worker's ECS task ARN in cloud (dev jobs carry
                none or a PID-shaped value). Written only after launch, so a
                live job's record can still lack it.
            output_file: Results file path (e.g., "output/eval_results_baseline.json").
                Written durably so cross-process Job rebuilds (poller, reconciler) can
                render the "Results file:" line in the completion seed.
        """
        ...

    async def mark_cancel_requested(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Record on a job that a cancel has started stopping its worker.

        Writes ``cancel_requested_at`` (ISO-8601 UTC, now). A cancel stops the
        worker before it marks the job cancelled, so for that span the job reads
        as running with no worker, which is also how a crashed worker reads. This
        tells the two apart: the reconciler leaves a job carrying it to the cancel
        instead of reaping it as failed.

        Never creates a record. A job that does not exist is left alone.
        """
        ...

    async def claim_completion_fired(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Claim the exactly-once completion turn for a finished job.

        Returns True iff this caller won the exactly-once claim; subsequent calls
        return False. The claim is an ATTRIBUTE on the EXISTING job item
        (PK=USER#{owner_id}, SK=PROJECT#{project_id}#JOB#{job_id}), not a
        separate claim item — no orphan cleanup required.

        Cloud mode: conditional UpdateItem with
        ``attribute_not_exists(completion_fired)``; ConditionalCheckFailedException
        → return False. Dev mode: first-wins semantics via the in-process store
        (single-process safe).

        Used by the poller (job_progress_republisher.py): when a terminal
        job status is observed in the durable log, the poller claims the right to
        fire via this method, and only the winner fires the completion turn. A
        second observer (rolling deploy ⇒ ≥2 stateless tasks) gets False and fires
        nothing.

        Args:
            user_id: Job owner's user_id (for cloud DDB partition key).
            project_id: Project UUID.
            job_id: Job UUID.

        Returns:
            True if this caller won the claim (first call), False otherwise
            (subsequent calls or concurrent losers).
        """
        ...

    # --- Launch gate -------------------------------

    async def claim_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> str:
        """Settle the race between launching a job's worker and cancelling it.

        A detached eval becomes visible — and therefore cancellable — well before
        its worker exists. On the manual path the job is created deliberately
        ahead of a model preflight that can take 15-20s, and the snapshot upload
        and RunTask follow that. Nothing in the span read cancellation state, and
        the durable ``compute_handle`` is only written after RunTask returns, so a
        cancel arriving in the window found no handle, found no tagged task no
        matter how patiently it looked, concluded there was nothing to stop, and
        marked the job cancelled and released its lock. The dispatch then went on
        to start a worker for it. No amount of retrying a task lookup can close
        that, because the task does not exist yet.

        This is the serialization point that does. Both sides claim the same
        per-job gate and exactly one wins:

        * Dispatch claims it immediately before launching. Losing means a cancel
          already settled the job, so it must not start a worker.
        * Cancel claims it only after looking for a task and finding none. Winning
          means no worker exists and none can now be started, so "nothing to stop"
          is finally sound. Losing means one exists or is seconds away, so the
          cancel keeps looking instead of inferring absence.

        First write wins and the claim is never overwritten, so both sides can ask
        repeatedly and get the same answer.

        Cloud mode: conditional PutItem on a dedicated item
        (PK=USER#{owner_id}, SK=LAUNCH_GATE#{project_id}#{job_id}) with
        ``attribute_not_exists(SK)``; on a condition failure, read the item and
        report the claimant already recorded. Dev mode: in-process first-wins dict.

        The item carries a TTL, because it outlives nothing useful once the job is
        finished and an untidied claim would block a later cancel on a recycled id.

        Args:
            owner_id: Job owner's user_id (cloud partition key).
            project_id: Project UUID.
            job_id: Job whose launch is being raced.
            claimant: Who is claiming. Callers pass the two module constants in
                ``app.services.launch_gate`` rather than bare strings.

        Returns:
            The winning claimant. Equal to ``claimant`` when this caller won.
        """
        ...

    async def release_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> bool:
        """Drop a gate claim so the other side can proceed. Idempotent.

        Dispatch calls this when a launch it had claimed the gate for did not
        happen — the snapshot upload failed, RunTask was rejected, the process
        died. Without it the gate would keep reporting that a launch is in flight
        and every later cancel would wait out its window and report failure for a
        worker that never existed.

        Conditional on the claimant, so one side can never release the other's
        claim: a dispatch that lost the race must not be able to clear the cancel
        that beat it.

        Args:
            owner_id: Job owner's user_id.
            project_id: Project UUID.
            job_id: Job whose gate is being released.
            claimant: Who is releasing. The claim survives if it is held by
                anyone else.

        Returns:
            True if a claim by this claimant was removed, False otherwise.
        """
        ...

    # --- Eval concurrency lock ---------------------

    async def claim_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Atomically claim eval lock for project.

        Returns True if this caller won the claim (first call), False if the
        lock is already held. The lock is a dedicated DDB item
        (PK=USER#{owner_id}, SK=EVAL_LOCK#{project_id}) with a lock_version
        attribute and lock_holder_job_id.

        Cloud mode: conditional PutItem with attribute_not_exists(SK);
        ConditionalCheckFailedException → return False. Dev mode: first-wins
        semantics via in-process dict keyed by project_id.

        Used at dispatch to enforce one-eval-per-project across processes. A
        second dispatch on the same project is rejected until the holder
        releases the lock (via release_eval_lock on every terminal transition,
        or the reconciler clears a stale lock.

        Args:
            owner_id: Job owner's user_id (for cloud DDB partition key).
            project_id: Project UUID.
            job_id: Job UUID (the holder).

        Returns:
            True if this caller won the claim, False if already held.
        """
        ...

    async def release_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Release eval lock.

        Idempotent — no-op if lock not held by this job or already gone.
        Cloud mode: conditional DeleteItem on lock_holder_job_id = :job_id;
        swallow ConditionalCheckFailedException. Dev mode: delete only if
        holder matches.

        Called on every terminal transition (completed/failed/cancelled) by
        update_job_state. Ensures the lock is released on all paths, not just
        happy path.

        Args:
            owner_id: Job owner's user_id.
            project_id: Project UUID.
            job_id: Job UUID (the holder attempting release).
        """
        ...

    async def get_eval_lock(
        self, owner_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the eval lock item or None (for reconciler).

        Returns a dict with keys: lock_holder_job_id, lock_version,
        lock_acquired_at. The reconciler reads this to snapshot lock_version
        before deciding staleness, then issues a version-fenced clear via
        clear_stale_eval_lock.

        Args:
            owner_id: Job owner's user_id.
            project_id: Project UUID.

        Returns:
            Lock item dict or None if no lock held.
        """
        ...

    async def clear_stale_eval_lock(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        expected_version: int,
        expected_holder_job_id: JobId,
    ) -> bool:
        """Clear a stale eval lock, fenced on version AND holder.

        Returns True if deleted, False if either fence tripped. Cloud mode:
        conditional DeleteItem on both attributes; ConditionalCheckFailedException
        → return False. Dev mode: delete only if both match.

        BOTH fences are required, and the version alone is not enough. The
        version fence only covers the window between the reconciler reading the
        lock and deleting it — a lock that was released and RECLAIMED by a newer
        job before that read presents its own current version, so the delete
        matched and the reconciler removed a live job's lock, letting a third
        eval start against the same project. The holder fence is what ties the
        delete to the stale job the reconciler actually decided about.

        Args:
            owner_id: Job owner's user_id.
            project_id: Project UUID.
            expected_version: Lock version read by reconciler before deciding.
            expected_holder_job_id: The stale job the reconciler is reconciling.
                A lock held by anyone else must survive.

        Returns:
            True if cleared, False if either fence tripped.
        """
        ...

    # --- AgentCore session registry ---

    async def get_agentcore_session(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the AgentCore session item for this (user, project) or None.

        Returns the durable session registry item keyed PK=USER#{sub},
        SK=AGENTCORE_SESSION#{project_id}, containing runtime_session_id,
        app_session_id, status, updated_at, claim_version, and ttl.

        This is the durable resume mapping (survives
        StopRuntimeSession / microVM recycle): the runtime_session_id enables
        warm affinity (reuse the same runtimeSessionId for < 1h idle), and the
        app_session_id enables transcript resume correctness (S3SessionStore).

        Granularity is per-(user, project), not per-user — each project gets
        its own AgentCore microVM, matching the per-project EFS working tree
        and the per-project transcript.

        Args:
            user_id: User's sub (Cognito user_id).
            project_id: Project UUID.

        Returns:
            Session dict or None if no session exists. Fields (expand-only):
            runtime_session_id (str | None), app_session_id (str | None),
            status ("idle"|"running"), updated_at (int epoch), claim_version
            (int monotonic fence), ttl (int epoch).
        """
        ...

    async def put_agentcore_session(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        runtime_session_id: str | None,
        app_session_id: str | None,
        ttl: int,
    ) -> None:
        """Upsert the AgentCore session identity/resume mapping without claiming.

        Updates the runtime_session_id, app_session_id, and ttl fields on the
        (user, project) session item. Creates the item at status="idle" if
        absent. Does NOT touch status — recording the mapping mid-idle does
        not forge a claim (claim semantics).

        Used after agent turns to persist the mapping for next-turn warm
        affinity + transcript resume. Fields may be added, but existing ones
        are never renamed or removed: stored items outlive the code that
        wrote them.

        Args:
            user_id: User's sub.
            project_id: Project UUID.
            runtime_session_id: AgentCore runtimeSessionId (UUID) or None.
            app_session_id: S3SessionStore session_id (UUID) or None.
            ttl: Expiry epoch (seconds) for TTL-based cleanup backstop.
        """
        ...

    async def claim_agentcore_session_turn(
        self, user_id: UserId, project_id: ProjectId, *, ttl: int, current_turn_id: str | None = None
    ) -> bool:
        """Atomically claim the (user, project) session for one turn.

        Returns True iff this caller won the claim (first-wins). The claim is
        a conditional write status idle→running on the session item itself
        (PK=USER#{sub}, SK=AGENTCORE_SESSION#{project_id}). A second claim on
        the SAME (user, project) while status=running returns False. A claim
        on a DIFFERENT project for the SAME user returns True (per-(user,project)
        grain, NOT per-user).

        This is the PRIMARY, LOAD-BEARING one-turn-at-a-time serializer:
        AgentCore provides no documented native intra-session turn
        serialization, so our DDB claim is the source of truth. If
        AgentCore rejects a concurrent same-session invoke, that is
        belt-and-suspenders only — nothing depends on it.

        Modeled on claim_eval_lock (conditional write), but the
        claim is a status FIELD flip on the persisted item (which also carries
        the durable runtime/app session-id mapping), NOT a separate lock item
        that is deleted on release. Release flips status back to idle;
        release does NOT delete the item (the mapping must survive across
        turns for warm affinity + resume).

        Bumps claim_version (monotonic, survives release — the fence) and
        updated_at on win. TTL backstop if a crashed turn never releases.

        Cloud mode: conditional UpdateItem SET status="running" guarded by
        status="idle" (or attribute_not_exists for a never-claimed field);
        ConditionalCheckFailedException → return False. Dev mode: first-wins
        semantics via in-process per-(user,project) lock + dict.

        Args:
            user_id: User's sub.
            project_id: Project UUID.
            ttl: Expiry epoch (seconds) for the session item.
            current_turn_id: Turn UUID being claimed (optional, for in-flight turn discovery).

        Returns:
            True if this caller won the claim, False if already held.
        """
        ...

    async def release_agentcore_session_turn(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        expected_turn_id: str | None = None,
    ) -> None:
        """Release the (user, project) session turn claim (idempotent).

        Conditional write status running→idle. Idempotent: swallows
        ConditionalCheckFailed when not running / item gone. Never deletes
        the item — the mapping persists for next-turn warm affinity + resume.

        Called on turn completion (success / failure / cancel). Refreshes
        updated_at + ttl on successful release.

        Pass ``expected_turn_id`` whenever the caller knows its own turn. Without
        it the release is fenced only on ``status = running``, which is not a
        fence at all across turns: releases run from a ``finally``, and a turn
        admitted after this one's lease expired sets the item back to running, so
        a late release matched and cleared the SUCCESSOR's live claim.

        Args:
            user_id: User's sub.
            project_id: Project UUID.
            expected_turn_id: Only release if this turn still holds the claim.
        """
        ...

    async def list_agentcore_sessions(self, user_id: UserId) -> list[dict[str, Any]]:
        """List all AgentCore sessions for a user.

        Base-table query PK=USER#{uid} with SK prefix filter
        begins_with("AGENTCORE_SESSION#"). No IndexName= (GSI chokepoint
        invariant preserved).

        Args:
            user_id: User's sub.

        Returns:
            List of session dicts (same shape as get_agentcore_session).
        """
        ...

    async def list_running_agentcore_sessions(self) -> list[dict[str, Any]]:
        """List ALL running AgentCore sessions across users (claim reconcilers).

        Used by the startup reconciler to enumerate orphaned claims and by the
        periodic sweep to find expired ones. Returns all sessions with
        status="running" across all users. Each dict contains user_id,
        project_id, claim_version (and potentially other fields).

        DDB: a GSI1 Query on the sparse ``GSI1PK="AGENTCORE_SESSION#running"``
        partition, which holds only the claims held right now. Dev: iterates
        the in-memory dict (returns [] on fresh process — dev orphans cannot
        survive a restart; kept for parity).

        Returns:
            List of session dicts with user_id, project_id, claim_version.
        """
        ...

    async def clear_stale_agentcore_session_claim(
        self, user_id: UserId, project_id: ProjectId, expected_version: int
    ) -> bool:
        """Clear a stale running claim with version fence (reconciler path).

        Returns True if cleared, False if claim_version changed (the claim was
        released+reclaimed since read). Conditional write status→idle only
        when claim_version = :expected_version.

        The version fence prevents the reconciler from clearing a claim that
        was released and reclaimed while the reconciler was deciding staleness
        (mirrors the reconciler's grace-period reasoning). Uses
        claim_version as the strictly-increasing fence token that survives
        release (monotonic counter incremented on every claim).

        Args:
            user_id: User's sub.
            project_id: Project UUID.
            expected_version: claim_version (int) read by reconciler before
                deciding staleness.

        Returns:
            True if cleared, False if version changed (fence tripped).
        """
        ...

    # --- Dispatch idempotency + reconciler helpers ---

    async def claim_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId, ttl_seconds: int
    ) -> tuple[bool, str | None]:
        """Claim idempotency key.

        Returns (claimed, existing_job_id): (True, None) when this call freshly
        claimed the key, (False, <first job's job_id>) when a prior dispatch
        already claimed it within TTL.

        The losing-claim path returns that existing_job_id so the agent
        can still track/poll/wait/cancel the collapsed-to job.

        Cloud mode: conditional PutItem on SK=IDEMPOTENCY#{key} under PK=USER#{owner_id},
        stores job_id + ttl; on ConditionalCheckFailedException read back the
        stored job_id. Dev mode: in-process dict with TTL expiry.

        Args:
            owner_id: Job owner's user_id (for partition key).
            idempotency_key: Client-supplied key (from SPA "Run eval" click).
            job_id: Job UUID to associate with this key.
            ttl_seconds: TTL for the key (300s recommended).

        Returns:
            (True, None) if fresh claim, (False, existing_job_id) if duplicate.
        """
        ...

    async def release_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId
    ) -> bool:
        """Release an idempotency key after a failed dispatch (ownership-guarded).

        Releases the key ONLY when the stored job_id matches the caller's job_id.
        This ownership invariant ensures a dispatch never releases a key that an
        EARLIER dispatch owns (preventing the "two concurrent clicks both launch"
        corruption).

        Cloud mode: conditional DeleteItem on SK=IDEMPOTENCY#{key} with
        ConditionExpression="job_id = :job_id"; returns False on
        ConditionalCheckFailedException (ownership guard tripped).

        Dev mode: under lock, delete (owner_id, key) from in-process dict ONLY
        if the stored job_id matches; return False otherwise.

        Args:
            owner_id: Job owner's user_id (for partition key).
            idempotency_key: Key to release.
            job_id: This dispatch's job_id (ownership guard).

        Returns:
            True if released, False if ownership guard tripped (stored job_id
            does not match, or key already gone).
        """
        ...

    async def count_active_jobs_for_owner(self, owner_id: UserId) -> int:
        """Count non-terminal jobs for owner (per-user cap).

        Returns count of jobs in states (queued, running, awaiting_user).
        Terminal states (completed, failed, cancelled) are excluded.

        Cloud + dev: call list_jobs_by_owner and filter to non-terminal.
        At 1-10 users a filtered owner list is cheap (no new infra).

        Args:
            owner_id: Job owner's user_id.

        Returns:
            Count of non-terminal jobs.
        """
        ...

    async def list_non_terminal_jobs(self) -> list[dict[str, Any]]:
        """List all non-terminal jobs across users (reconciler).

        Returns jobs in states (running, queued). Used by the reconciler,
        the account backstop cap, and startup reattach.

        Cloud: paginated IndexName="GSI1" query for GSI1PK="JOB#running" and
        GSI1PK="JOB#queued", union the results. Dev: read every project dir's
        job records and filter to non-terminal.

        Returns:
            List of non-terminal job dicts.
        """
        ...

    async def list_recent_completed_jobs(
        self, within_seconds: int = 86400
    ) -> list[dict[str, Any]]:
        """List recent terminal jobs that may need completion turn recovery.

        Returns jobs in states (completed, failed) created within the recency window.
        EXCLUDES cancelled jobs (user-initiated, no turn owed). Used by the reconciler
        to re-fire orphaned completion turns when a job completes while the coordinator
        is down or draining.

        The recency bound is aligned to the event-log TTL (24h default). Beyond that
        window the 'results' event is gone; the job record still has results but no chat
        turn is owed (acceptable edge).

        Cloud: paginated IndexName="GSI1" queries for GSI1PK="JOB#completed" and
        GSI1PK="JOB#failed", with GSI1SK >= JOB#{since_iso} range condition to bound
        by created_at. GSI1SK is `JOB#{created_at_iso}#{job_id}`, so lexicographic
        range works. Dev: read every project dir's job records and filter to
        completed/failed within window.

        Args:
            within_seconds: Recency window in seconds (default 86400 = 24h).

        Returns:
            List of recently terminal job dicts (includes completion_fired and result fields).
        """
        ...

    # --- Turn records (chat sessions) -----------------------------------

    async def get_turn(
        self, user_id: UserId, project_id: ProjectId, turn_id: TurnId
    ) -> dict[str, Any] | None:
        """Return the turn record or ``None``."""
        ...

    async def list_active_turns(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        """Return turns in non-terminal state (``running`` | ``interrupted``)."""
        ...

    async def put_turn(self, turn: dict[str, Any]) -> None:
        """Create or replace a turn record."""
        ...

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
        """Partial update of a turn record."""
        ...

    # --- Optimization state ---------------------------------------------

    async def get_optimization_state(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the optimization state record or ``None``."""
        ...

    async def put_optimization_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        state: dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> int:
        """Conditional write of the optimization state.

        Args:
            expected_version: If provided, the write only succeeds when
                the stored ``version`` field matches — optimistic locking
                for the cloud store (the dev store uses a per-process
                ``threading.Lock`` in ``optimization_state.py``).

        Returns:
            The new version number.

        Raises:
            ConcurrentUpdateError: ``expected_version`` provided and didn't
                match the stored value.
        """
        ...

    # --- Event log ---------------------------------------------

    async def allocate_job_seq(self, job_id: JobId) -> int:
        """Atomically allocate the next event seq for a job (cross-process safe).

        Returns strictly increasing distinct integers across concurrent callers.
        First call returns 1.

        Uses DynamoDB atomic counter (``UPDATE ADD``) in cloud mode; in-process
        max+1 in dev mode (single-process safe).
        """
        ...

    async def append_job_event(
        self, job_id: JobId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one durable event item with TTL.

        Args:
            job_id: Job identifier (opaque UUID).
            seq: Event sequence number (from ``allocate_job_seq``).
            event: Event JSON payload (envelope or event dict).
            ttl: Expiry timestamp (epoch seconds). The table-level ``ttl``
                attribute handles auto-expiry.
        """
        ...

    async def events_since(
        self, job_id: JobId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Return event JSONs whose seq > since_seq, ordered ascending.

        Args:
            job_id: Job identifier.
            since_seq: Return events with seq strictly greater than this value.

        Returns:
            List of event dicts, ordered by seq ascending. Empty list when none.
        """
        ...

    # --- Turn event log -------------------------------------------------

    async def allocate_turn_seq(self, turn_id: TurnId) -> int:
        """Atomically allocate the next event seq for a turn (cross-process safe).

        Returns strictly increasing distinct integers across concurrent callers.
        First call returns 1.

        Uses DynamoDB atomic counter (``UPDATE ADD``) in cloud mode; in-process
        max+1 in dev mode (single-process safe).
        """
        ...

    async def append_turn_event(
        self, turn_id: TurnId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one durable turn event item with TTL.

        Args:
            turn_id: Turn identifier (opaque UUID).
            seq: Event sequence number (from ``allocate_turn_seq``).
            event: Event JSON payload (envelope or event dict).
            ttl: Expiry timestamp (epoch seconds). The table-level ``ttl``
                attribute handles auto-expiry.
        """
        ...

    async def turn_events_since(
        self, turn_id: TurnId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Return turn event JSONs whose seq > since_seq, ordered ascending.

        Args:
            turn_id: Turn identifier.
            since_seq: Return events with seq strictly greater than this value.

        Returns:
            List of event dicts, ordered by seq ascending. Empty list when none.
        """
        ...

    # --- Project event seq (one seq domain for project: streams) -----------

    async def allocate_project_seq(self, project_id: ProjectId) -> int:
        """Atomically allocate the next event seq for a project stream (cross-process safe).

        Returns strictly increasing distinct integers across concurrent AND
        cross-process callers. First call returns 1.

        WHY: more than one process publishes on a `project:<id>` stream (the
        coordinator hub and the microVM worker route), so per-process counters
        can issue the same seq twice within a single coordinator lifetime — a
        collision, not a counter reset. `project:` has no durable event log, but
        its replay buffer (ProjectEventBuffer) and the client cursor compared
        against it are consulted by every publisher, so it still needs one
        cross-process seq authority.

        Uses DynamoDB atomic counter (``UPDATE ADD``) in cloud mode; persisted
        JSON counter in dev mode (cross-instance safe via file read/write).

        Cloud: counter lives under dedicated partition key ``PROJECT_SEQ#{project_id}``
        with ``SK="SEQ"``, NOT under ``USER#{uid}``, so the two `begins_with(SK,
        "PROJECT#")` queries on the user partition are structurally unaffected by
        the counter item (key-namespace isolation).

        Args:
            project_id: Project identifier (UUID).

        Returns:
            Next sequence number (starts at 1, monotonically increasing).
        """
        ...

    async def put_turn_metadata(
        self, turn_id: TurnId, *, user_id: UserId, project_id: ProjectId, ttl: int
    ) -> None:
        """Store turn metadata for owner resolution (turn: replay owner-guard).

        Creates an item at ``PK=TURN#{turn_id}, SK=META`` with ``{user_id, project_id, ttl}``.
        This enables resolving turn_id → (user_id, project_id) for the owner-guard
        in ``_replay_turn_stream_from_durable_log`` without inventing an owner-less
        turn-by-id lookup.

        Args:
            turn_id: Turn identifier.
            user_id: Owner's Cognito sub.
            project_id: Project UUID.
            ttl: Expiry timestamp (epoch seconds), same as turn events (7 days).
        """
        ...

    async def get_turn_metadata(self, turn_id: TurnId) -> dict[str, Any] | None:
        """Retrieve turn metadata for owner resolution.

        Args:
            turn_id: Turn identifier.

        Returns:
            ``{user_id, project_id, ttl}`` or ``None`` if not found.
        """
        ...

    # --- Connection registry ---------------------------------

    async def put_connection(
        self,
        connection_id: str,
        user_id: UserId,
        *,
        project_id: ProjectId | None = None,
        ttl: int,
    ) -> None:
        """Persist a WebSocket connection with TTL.

        Args:
            connection_id: Opaque connection identifier (UUID4 from WS route).
            user_id: Authenticated user ID from request context.
            project_id: Optional project binding (often unknown at connect time).
            ttl: Expiry timestamp (epoch seconds). The table-level ``ttl``
                attribute handles auto-expiry.
        """
        ...

    async def delete_connection(self, connection_id: str) -> None:
        """Remove a connection record.

        Deleting a missing connection is a no-op (no error raised).
        """
        ...

    async def list_connections_for_user(
        self, user_id: UserId
    ) -> list[dict[str, Any]]:
        """Return all connection records for a user (GSI1 by-user lookup).

        Returns:
            List of connection dicts. Empty list when none.
        """
        ...

    # --- Project event outbox ------------------------------------------

    async def put_outbox_event(
        self, owner_id: UserId, project_id: ProjectId, event: dict[str, Any]
    ) -> None:
        """Stash an unroutable microVM project event for coordinator drain.

        Keyed ``PK=USER#{owner}``, ``SK=OUTBOX#{project}#{ns-timestamp}`` —
        inside the owner's partition so the microVM's LeadingKeys-scoped
        write policy covers it without IAM changes. Items carry a 1-hour
        TTL: an event that old is stale by definition and expires unread.
        """
        ...

    async def list_outbox_events(
        self, owner_id: UserId, project_id: ProjectId, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return stashed events for a project, oldest first (SK order).

        Each item carries at least ``SK`` (the delete handle) and ``event``
        (the original payload dict).
        """
        ...

    async def delete_outbox_event(self, owner_id: UserId, sk: str) -> None:
        """Delete one drained outbox item. Missing item is a no-op."""
        ...

    # --- Cost tracking --------------------------------------------------

    async def add_bedrock_cost(
        self,
        user_id: UserId,
        *,
        year_month: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Atomically increment the user's monthly Bedrock cost rollup."""
        ...

    async def get_monthly_costs(
        self, user_id: UserId, year_month: str
    ) -> dict[str, Any] | None:
        """Return the cost rollup for the given ``YYYY-MM`` or ``None``."""
        ...


class ConcurrentUpdateError(Exception):
    """Raised by :meth:`MetadataStore.put_optimization_state` when the
    ``expected_version`` precondition fails."""


class ProjectOwnerConflictError(Exception):
    """Raised by :meth:`MetadataStore.put_project` when the project id is
    already indexed under a different owner, and nothing was written.

    A project's owner is fixed when it is created. A record naming another
    owner for an existing id is a redirected write, never a transfer."""


#: The project fields a user edits directly (see
#: :meth:`MetadataStore.update_project_settings`). Identity and access fields are
#: deliberately absent.
PROJECT_SETTINGS_FIELDS: frozenset[str] = frozenset({"name", "description", "agent_model"})


def check_project_settings(settings: dict[str, Any]) -> None:
    """Refuse any key that is not a user-editable project setting."""
    extra = set(settings) - PROJECT_SETTINGS_FIELDS
    if extra:
        raise ValueError(f"not project settings: {sorted(extra)}")


__all__ = [
    "ConcurrentUpdateError",
    "MetadataStore",
    "PROJECT_SETTINGS_FIELDS",
    "ProjectOwnerConflictError",
    "check_project_settings",
]
