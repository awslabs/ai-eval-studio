# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One admission gate for every path that launches billable compute.

The cost kill switch and the per-user / account job caps used to live only in
``mcp_tools.dispatch``, so they applied to launches the coordinator made and to
nothing else. The REST start and resume routes — which is what the UI calls —
enforced neither, and both caps were additionally skipped on the non-arch-split
execution path. A gate the primary entry point does not pass is not a gate.

This module holds the checks themselves, with no opinion on how a caller reports
a refusal: MCP tools turn one into ``mcp_error`` so the coordinator can react,
REST routes turn one into an HTTP status.

The checks are read-then-act, not leases. Concurrent launches can therefore each
observe the same pre-launch count and each pass. That window is unchanged from
the original implementation and is not what this module set out to fix; the
per-project eval lock (:func:`claim_project_eval_lock`) IS atomic, so the
narrower "one eval per project" invariant does not depend on these counts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: Concurrent non-terminal jobs one user may hold.
PER_USER_MAX_JOBS = 3

#: Concurrent non-terminal jobs across the whole deployment. A backstop against
#: one runaway loop, not a per-tenant quota.
ACCOUNT_MAX_JOBS = 10

LaunchKind = Literal["eval", "optimization"]


@dataclass(frozen=True)
class AdmissionRefusal:
    """Why a launch was refused, in a form any transport can render.

    Attributes:
        code: Stable machine-readable reason. Callers may match on it; the
            existing coordinator-facing strings embed these same codes.
        message: Sentence for the person or agent that asked.
    """

    code: str
    message: str


async def check_launch_admission(
    *, owner_id: str, project_id: str, kind: LaunchKind
) -> AdmissionRefusal | None:
    """Return a refusal if this launch must not proceed, else None.

    Does not claim anything and has no side effects, so a caller may run it
    before doing any work and abandon the launch cheaply.

    Args:
        owner_id: Project owner whose per-user cap applies.
        project_id: Project being launched into. Used only for logging.
        kind: What is being launched, so the message names it.

    Returns:
        ``AdmissionRefusal`` when refused, ``None`` when admitted.
    """
    from app.platform import get_platform
    from app.platform.arch_split import is_cost_kill_switch_active

    noun = "Eval" if kind == "eval" else "Optimization"

    if is_cost_kill_switch_active():
        return AdmissionRefusal(
            code="cost_kill_switch_active",
            message=(
                f"{noun} launches are halted by cost control "
                "(cost_kill_switch_active). Contact the operator."
            ),
        )

    store = get_platform().metadata_store

    owner_active = await store.count_active_jobs_for_owner(owner_id)
    if owner_active >= PER_USER_MAX_JOBS:
        return AdmissionRefusal(
            code="user_job_limit",
            message=(
                f"You have reached the per-user job limit ({PER_USER_MAX_JOBS} "
                "active jobs, user_job_limit). Wait for a job to finish or cancel "
                "one before starting another."
            ),
        )

    all_non_terminal = await store.list_non_terminal_jobs()
    if len(all_non_terminal) >= ACCOUNT_MAX_JOBS:
        return AdmissionRefusal(
            code="account_job_limit",
            message=(
                f"The account has reached the global job limit ({ACCOUNT_MAX_JOBS} "
                "active jobs, account_job_limit). Wait for jobs to finish before "
                "starting another."
            ),
        )

    logger.debug(
        "[admission] admitted %s for owner=%s project=%s (owner_active=%d, "
        "account_active=%d)",
        kind, owner_id[:8], project_id[:8], owner_active, len(all_non_terminal),
    )
    return None


async def claim_project_eval_lock(
    *, owner_id: str, project_id: str, holder_id: str
) -> bool:
    """Atomically claim the project's one-eval-at-a-time lock.

    This is the real mutual exclusion — a DynamoDB conditional put — and it is
    what makes "two evals on one project" impossible rather than unlikely. It is
    separate from :func:`check_launch_admission` because a caller has to be able
    to release it on any later failure, which means owning the claim explicitly.

    Args:
        owner_id: Owner partition the lock lives under.
        project_id: Project to lock.
        holder_id: Who holds it. An eval passes its job id; an optimization run
            passes a run-scoped synthetic id so per-iteration evals cannot
            release the run's lock out from under it.

    Returns:
        True if the lock is now held by ``holder_id``.
    """
    from app.platform import get_platform

    return await get_platform().metadata_store.claim_eval_lock(
        owner_id, project_id, holder_id
    )


async def release_project_eval_lock(
    *, owner_id: str, project_id: str, holder_id: str
) -> None:
    """Release a lock this holder owns. Holder-conditional and idempotent.

    Safe to call on a path that may not hold the lock: releasing one held by
    someone else is a no-op rather than a theft.
    """
    from app.platform import get_platform

    try:
        await get_platform().metadata_store.release_eval_lock(
            owner_id, project_id, holder_id
        )
    except Exception:  # noqa: BLE001 - the reconciler's sweep is the backstop
        logger.exception(
            "[admission] failed to release eval lock for project=%s holder=%s",
            project_id[:8], holder_id[:8],
        )


def optimization_lock_holder_id(project_id: str) -> str:
    """Run-scoped lock holder id for an optimization loop.

    One string, one definition: the REST route and the MCP tool both claim under
    it and the loop's finalizer releases it, so they have to agree exactly.
    """
    return f"optimization-{project_id}"
