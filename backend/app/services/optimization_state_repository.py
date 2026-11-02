# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single seam for reading and writing :class:`OptimizationState`.

**Why this module exists.** Optimization state has two homes — an
``output/optimization/state.json`` file on EFS, and an ``OPT`` item in DynamoDB —
and which one is authoritative depends on the ``arch_split`` flag, which
``arch_split.resolve_arch_split`` derives from the deployment mode (the durable
item in cloud, the EFS file in dev). Before
this module, every one of ~38 call sites re-implemented that branch by hand:

    if arch_split:
        state_data = await store.get_optimization_state(user_id, project_id)
        state = OptimizationState._from_dict(project_dir, state_data) if state_data else None
        version = state_data.get("version", 0) if state_data else None
    else:
        state = OptimizationState.load(project_dir)
        version = None

Each hand-rolled copy is an independent chance to get it wrong, and the failure is
silent: reading the wrong home returns *plausible* state, so the caller proceeds
against data that does not reflect reality. Eleven sites had simply omitted the
branch and always read EFS — including read-modify-**write** paths, where a bare
EFS write left DynamoDB saying ``running`` while the UI announced ``cancelled``.
Fixing those one at a time is what this codebase has done repeatedly; the class
regenerates because the branch is duplicated rather than owned.

Four concerns live here and nowhere else:

1. **Home selection** — the ``arch_split`` branch, resolved once per operation.
2. **Version threading** — DynamoDB writes are conditional on ``expected_version``.
   Consecutive writes must chain the version *returned* by the previous write;
   guessing ``version + 1`` desynchronizes the chain and loses races. The
   :class:`OptimizationStateHandle` carries that version so callers cannot forget.
3. **Owner partition** — the durable item is keyed on the project OWNER. Keying it
   on the requester reads an empty partition for any caller without request
   context.
4. **Decimal coercion** — the store returns ``Decimal`` for fractional numbers, and
   the convergence math mixes them with floats (``Decimal - float`` raises). This is
   applied on the durable read path only, so the store's ``cost_usd`` precision
   guarantee is untouched.

**The sync ``OptimizationState.load`` is deliberately still public.** Some readers
are genuinely EFS-only — they want the on-disk artifact, not the authoritative
record — and forcing them through an async repository would be dishonest about what
they read. Those sites call ``load`` directly and say why. Everything that reads
*authoritative* state goes through here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.platform.types import ProjectId
from app.services.optimization_state import OptimizationState

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStateHandle:
    """A read of optimization state, plus everything needed to write it back.

    Bundling the state with its version and resolved owner is the point: a caller
    holding a handle cannot perform a conditional write with a guessed version or
    against the wrong partition, because it never has to construct those itself.

    Attributes:
        state: The loaded state, or ``None`` when no state exists.
        version: Durable version this read observed; ``None`` on the EFS path or
            when no record existed (which the store reads as
            ``attribute_not_exists``, i.e. a genuine first write).
        arch_split: Resolved home for this operation. Threaded, never re-resolved.
        owner_id: Owner partition key for durable writes; ``None`` on the EFS path.
        project_id: Project this handle refers to.
        project_dir: Project directory (still needed — ``OptimizationState``
            carries it, and the EFS path writes there).
    """

    state: OptimizationState | None
    version: int | None
    arch_split: bool
    owner_id: str | None
    project_id: ProjectId
    project_dir: Path

    @property
    def exists(self) -> bool:
        """Whether any state was found."""
        return self.state is not None

    @property
    def status(self) -> str | None:
        """Status of the loaded state, or ``None`` when absent."""
        return self.state.status if self.state else None

    def is_live(self) -> bool:
        """Whether a run is currently in flight (``running`` or ``stopping``).

        The single definition of "live" — this predicate was open-coded as
        ``status in ("running", "stopping")`` at a dozen sites, which is how one of
        them ended up checking only ``running`` and missing a stopping loop.
        """
        return self.status in ("running", "stopping")


async def resolve_owner(project_id: ProjectId) -> str:
    """Resolve the OWNER user_id for a project's durable partition.

    Delegates to the same resolver ``dispatch.py`` uses for lock/caps keys, so
    every durable key for a project agrees on one identity.

    Args:
        project_id: Project identifier.

    Returns:
        The project owner's user_id (falls back to the dev user).
    """
    from app.services.agent_turn import _resolve_owner_id_for_project

    return _resolve_owner_id_for_project(project_id)


async def read(
    project_id: ProjectId,
    project_dir: Path,
    *,
    arch_split: bool | None = None,
    owner_id: str | None = None,
) -> OptimizationStateHandle:
    """Read authoritative optimization state for a project.

    Args:
        project_id: Project identifier.
        project_dir: Project directory.
        arch_split: Pre-resolved flag. Pass it when the caller already resolved it
            for this operation — resolve-once-and-thread is what makes an operation
            provably all-old-path or all-new-path. Omit only at an operation entry
            point, where this resolves it.
        owner_id: Pre-resolved owner. Omit to resolve here.

    Returns:
        A handle carrying the state, its version, and the write context.
    """
    if arch_split is None:
        from app.platform.arch_split import resolve_arch_split

        arch_split = await resolve_arch_split(project_id)

    if not arch_split:
        # EFS is authoritative on the old path.
        return OptimizationStateHandle(
            state=OptimizationState.load(project_dir),
            version=None,
            arch_split=False,
            owner_id=None,
            project_id=project_id,
            project_dir=project_dir,
        )

    if owner_id is None:
        owner_id = await resolve_owner(project_id)

    from app.platform import get_platform

    store = get_platform().metadata_store
    state_data: dict[str, Any] | None = await store.get_optimization_state(
        owner_id, project_id
    )

    if not state_data:
        # No durable record. version stays None so a following write uses the
        # store's attribute_not_exists first-write guard rather than asserting a
        # version that was never observed.
        return OptimizationStateHandle(
            state=None,
            version=None,
            arch_split=True,
            owner_id=owner_id,
            project_id=project_id,
            project_dir=project_dir,
        )

    # _from_dict applies the Decimal coercion internally on this path.
    return OptimizationStateHandle(
        state=OptimizationState._from_dict(project_dir, state_data),
        version=state_data.get("version", 0),
        arch_split=True,
        owner_id=owner_id,
        project_id=project_id,
        project_dir=project_dir,
    )


def write(handle: OptimizationStateHandle) -> OptimizationStateHandle:
    """Persist the handle's state to its authoritative home.

    Returns a handle carrying the NEW version, so a caller performing consecutive
    writes chains versions by reassignment:

        handle = repo.write(handle)   # version advances
        handle.state.status = "cancelled"
        handle = repo.write(handle)

    That is the whole reason this returns a handle instead of ``None``. The
    previous convention — ``new_version = state.save(...)`` and remembering to
    reassign a separate ``current_version`` local — is what produced version-race
    bugs when a branch forgot the reassignment.

    Args:
        handle: Handle whose ``state`` has been mutated. Must have a state.

    Returns:
        The same handle with ``version`` advanced (durable path) or unchanged
        (EFS path).

    Raises:
        ValueError: If the handle carries no state to write.
        ConcurrentUpdateError: On a durable version conflict (propagated from the
            store — callers decide whether that is fatal).
    """
    if handle.state is None:
        raise ValueError("cannot write an OptimizationStateHandle with no state")

    if not handle.arch_split:
        handle.state.save()
        return handle

    new_version = handle.state.save(
        arch_split=True,
        user_id=handle.owner_id,
        project_id=handle.project_id,
        expected_version=handle.version,
    )
    return OptimizationStateHandle(
        state=handle.state,
        version=new_version,
        arch_split=True,
        owner_id=handle.owner_id,
        project_id=handle.project_id,
        project_dir=handle.project_dir,
    )


def create(
    handle: OptimizationStateHandle,
    *,
    target_metric_scores: dict[str, float],
    max_iterations: int,
    plateau_window: int,
    plateau_threshold: float | None = None,
    workload: dict[str, Any] | None = None,
) -> OptimizationStateHandle:
    """Create a fresh optimization run and persist it to the authoritative home.

    Takes the handle from a preceding :func:`read` so the create inherits its
    resolved home, owner, and observed version. That version matters: a *second*
    start on a project overwrites the previous run's TERMINAL record, and the
    store's condition is ``attribute_not_exists(version) OR version = :expected``.
    Passing ``None`` when a record already exists fails the condition, so the
    read's version must be threaded — which is automatic here because it rides on
    the handle.

    Args:
        handle: Handle from a preceding read (supplies home/owner/version).
        target_metric_scores: Target pass rates per metric.
        max_iterations: Per-run iteration budget.
        plateau_window: Iterations considered for plateau detection.
        plateau_threshold: Pass-rate spread below which the window is a plateau.
        workload: Workload fingerprint captured at creation.

    Returns:
        A handle carrying the newly created state and its version.
    """
    state = OptimizationState.create(
        project_dir=handle.project_dir,
        target_metric_scores=target_metric_scores,
        max_iterations=max_iterations,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        workload=workload,
        arch_split=handle.arch_split,
        user_id=handle.owner_id,
        project_id=handle.project_id,
        expected_version=handle.version,
    )
    # create() persists internally; the version it wrote is read's version + 1 on
    # the durable path. Re-read is avoided (an extra round trip on the hot start
    # path) — but the version is NOT guessed for a subsequent write either: callers
    # that write again after create must re-read, which is what every current caller
    # does by spawning the loop rather than writing inline.
    return OptimizationStateHandle(
        state=state,
        version=None,
        arch_split=handle.arch_split,
        owner_id=handle.owner_id,
        project_id=handle.project_id,
        project_dir=handle.project_dir,
    )


def record_iteration(
    handle: OptimizationStateHandle, **kwargs: Any
) -> OptimizationStateHandle:
    """Record an iteration and return a handle with the version advanced.

    ``OptimizationState.record_iteration`` persists internally and returns the new
    version, which the caller then had to thread into the NEXT write by hand. That
    hand-threading is the failure mode: a branch that forgot it passed a stale
    version and lost the conditional write. Returning a handle makes the chain
    automatic.

    Args:
        handle: Handle from a preceding read.
        **kwargs: Forwarded to ``OptimizationState.record_iteration`` (the
            iteration payload — prompt_version, run_id, metrics, and so on).

    Returns:
        Handle with ``version`` advanced.
    """
    new_version = handle.state.record_iteration(
        **kwargs,
        arch_split=handle.arch_split,
        user_id=handle.owner_id,
        project_id=handle.project_id,
        expected_version=handle.version,
    )
    return OptimizationStateHandle(
        state=handle.state,
        version=new_version,
        arch_split=handle.arch_split,
        owner_id=handle.owner_id,
        project_id=handle.project_id,
        project_dir=handle.project_dir,
    )


def check_convergence(
    handle: OptimizationStateHandle,
) -> tuple[str | None, OptimizationStateHandle]:
    """Check convergence, returning the reason and a version-advanced handle.

    ``check_convergence`` may WRITE (it persists the terminal status when a
    convergence condition trips), so it participates in the version chain like any
    other write.

    Args:
        handle: Handle from a preceding read/record_iteration.

    Returns:
        ``(convergence_reason_or_None, handle_with_version_advanced)``.
    """
    convergence, new_version = handle.state.check_convergence(
        arch_split=handle.arch_split,
        user_id=handle.owner_id,
        project_id=handle.project_id,
        expected_version=handle.version,
    )
    return convergence, OptimizationStateHandle(
        state=handle.state,
        version=new_version,
        arch_split=handle.arch_split,
        owner_id=handle.owner_id,
        project_id=handle.project_id,
        project_dir=handle.project_dir,
    )


async def read_for_display(
    project_id: ProjectId,
    project_dir: Path,
    *,
    arch_split: bool | None = None,
) -> OptimizationState | None:
    """Read state for a read-only consumer that will not write it back.

    Convenience over :func:`read` for the many display/report callers that want the
    state and nothing else. Deliberately drops the version so such a caller
    *cannot* perform a conditional write with it — a read-only site that later
    grows a write should switch to :func:`read` and thread the handle, not smuggle
    a stale version forward.

    On a durable-store failure this degrades to ``None``, NOT to an EFS read.
    Falling back to the other home returns state from a source that is not
    authoritative for this project, which reads as success and silently shows stale
    data; "unknown" is the honest answer.
    """
    try:
        handle = await read(project_id, project_dir, arch_split=arch_split)
    except Exception as exc:  # noqa: BLE001 — display must not crash the caller
        logger.warning(
            "[opt-state-repo] read failed for project=%s: %s — reporting unknown "
            "rather than degrading to the non-authoritative home",
            str(project_id)[:12],
            exc,
        )
        return None
    return handle.state
