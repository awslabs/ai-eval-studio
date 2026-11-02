# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serialization point between launching a detached eval and cancelling it.

A detached eval becomes visible — and therefore cancellable — well before its
worker exists. On the manual path the job record is created deliberately ahead of
a model preflight that can take 15-20s, so the "starting" phase a user is most
likely to cancel during is entirely inside the gap; the snapshot upload and
``RunTask`` follow it, and the durable ``compute_handle`` is written only after
``RunTask`` returns.

Nothing in that span read cancellation state. So a cancel arriving in it found no
handle, looked for a tagged ECS task and found none — correctly, because none
existed yet — concluded there was nothing to stop, marked the job cancelled and
released the project's eval lock. The dispatch then went on to start a worker for
a job the user had already cancelled and the UI had already reported gone. That
worker invoked target and judge models to completion with nothing watching it.

Retrying the task lookup cannot fix this, however patiently: the task is in the
future, not merely invisible. What fixes it is both sides agreeing on one
first-wins claim per job.

    Dispatch claims it immediately before launching. If it loses, a cancel has
    already settled the job, so no worker is started.

    Cancel claims it only after looking for a task and finding none. If it wins,
    no worker exists and none can be started now, so "nothing to stop" is finally
    sound. If it loses, one exists or is seconds away, so it keeps looking rather
    than inferring absence from a miss.

Every failure here is answered conservatively, in the direction that cannot leave
compute running unobserved: a dispatch that cannot settle the gate does not
launch, and a cancel that cannot settle it does not claim to have stopped
anything.
"""
from __future__ import annotations

import logging
from enum import Enum

from app.platform.types import JobId, ProjectId, UserId

logger = logging.getLogger(__name__)

#: Claimed by the dispatch path, immediately before it starts a worker.
LAUNCH_CLAIM = "launch"

#: Claimed by the cancel path, once it has looked for a worker and found none.
CANCEL_CLAIM = "cancel"


class LaunchDecision(Enum):
    """What a dispatch learned when it asked to launch.

    Three outcomes, not two. Collapsing the last two into a single "do not launch"
    was enough to make the dispatch decline safely, but not enough to be honest
    about it or to clean up correctly: the caller reported the evaluation
    *cancelled* when the truth was that the store could not be reached, and it left
    the job for a reconciler to settle because it could not tell that anyone still
    needed to.
    """

    #: No cancel has settled this job. Start the worker.
    PROCEED = "proceed"

    #: A cancel won the gate. It owns the job's terminal state and has already
    #: published it, so the dispatch stops and writes nothing.
    CANCELLED = "cancelled"

    #: The gate could not be settled at all. Nothing is known about cancellation,
    #: so launching is unsafe — but nobody else owns this job either, so the caller
    #: has to fail it rather than leave it pending.
    UNAVAILABLE = "unavailable"


async def claim_for_launch(
    owner_id: UserId, project_id: ProjectId, job_id: JobId
) -> LaunchDecision:
    """Whether this dispatch may start a worker for ``job_id``.

    Call immediately before the launch, not earlier: the value of the claim is
    that nothing cancellable happens between it and ``RunTask``.

    Args:
        owner_id: Job owner, for the durable partition.
        project_id: Project the job belongs to.
        job_id: Job about to be launched.

    Returns:
        A :class:`LaunchDecision`. Both non-``PROCEED`` outcomes mean "do not
        launch", but they differ in who owns the job afterwards and in what the
        caller may truthfully tell the user, so they are reported separately.
    """
    from app.platform import get_platform

    try:
        winner = await get_platform().metadata_store.claim_launch_gate(
            owner_id, project_id, job_id, claimant=LAUNCH_CLAIM
        )
    except Exception as exc:  # noqa: BLE001 - unsettled means do not launch
        logger.error(
            "[launch-gate] could not settle the gate for job=%s, so not "
            "launching: %s",
            job_id[:8], exc,
        )
        return LaunchDecision.UNAVAILABLE

    if winner == LAUNCH_CLAIM:
        return LaunchDecision.PROCEED

    logger.warning(
        "[launch-gate] job=%s was cancelled before its worker started "
        "(gate held by %s); not launching",
        job_id[:8], winner,
    )
    return LaunchDecision.CANCELLED


async def claim_for_cancel(
    owner_id: UserId, project_id: ProjectId, job_id: JobId
) -> bool:
    """Whether a cancel may conclude there is no worker to stop.

    Call only after looking for a running worker and finding none. Winning the
    gate turns that miss into proof: a launch that has not claimed the gate can no
    longer claim it, so it will not start a worker.

    Args:
        owner_id: Job owner, for the durable partition.
        project_id: Project the job belongs to.
        job_id: Job being cancelled.

    Returns:
        True if no worker exists and none can now start. False if a launch holds
        the gate — a worker exists or is imminent — or if the gate could not be
        settled, since neither case supports reporting a stop.
    """
    from app.platform import get_platform

    try:
        winner = await get_platform().metadata_store.claim_launch_gate(
            owner_id, project_id, job_id, claimant=CANCEL_CLAIM
        )
    except Exception as exc:  # noqa: BLE001 - unsettled means do not claim a stop
        logger.error(
            "[launch-gate] could not settle the gate for job=%s, so not "
            "concluding there is nothing to stop: %s",
            job_id[:8], exc,
        )
        return False

    if winner == CANCEL_CLAIM:
        logger.info(
            "[launch-gate] cancel won the gate for job=%s — no worker was ever "
            "launched and none can be now",
            job_id[:8],
        )
        return True

    logger.info(
        "[launch-gate] a launch holds the gate for job=%s, so a worker exists or "
        "is starting; continuing to look for it",
        job_id[:8],
    )
    return False


async def abandon_launch(
    owner_id: UserId, project_id: ProjectId, job_id: JobId
) -> None:
    """Release a launch claim for a launch that did not happen.

    The snapshot upload failed, ``RunTask`` was rejected, the process died between
    the claim and the call. Without this the gate keeps reporting a launch in
    flight, and every later cancel waits out its window and reports failure for a
    worker that never existed.

    Best-effort by construction: the caller is already failing the dispatch, and
    the claim carries a TTL as the backstop.

    Args:
        owner_id: Job owner, for the durable partition.
        project_id: Project the job belongs to.
        job_id: Job whose launch was abandoned.
    """
    from app.platform import get_platform

    try:
        released = await get_platform().metadata_store.release_launch_gate(
            owner_id, project_id, job_id, claimant=LAUNCH_CLAIM
        )
    except Exception as exc:  # noqa: BLE001 - the TTL is the backstop
        logger.warning(
            "[launch-gate] could not release the abandoned claim for job=%s: %s",
            job_id[:8], exc,
        )
        return

    if released:
        logger.info(
            "[launch-gate] released the launch claim for job=%s after a launch "
            "that did not happen",
            job_id[:8],
        )
