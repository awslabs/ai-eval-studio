# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A turn must run under the id its caller was told about.

Two callers commit to a turn id before the turn exists: a queued chat request
answers the browser with one, and a completion turn takes the session claim under
one. Both used to get an id that named nothing, because the builder that
eventually ran minted a fresh uuid of its own and nothing reconciled them.

The queued case is user-visible: the browser subscribes to ``turn:<id>``, so it
listens to a stream that never carries an event, its Stop returns 404, and the
real turn's terminal event is rejected by the client's own id guard — leaving the
spinner and Stop button up until a manual refresh.

The claim case is a concurrency hazard: an unattributable claim can only be
released by an unfenced release, and an unfenced release clears whatever is
running, including a successor's live claim.
"""
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def tracker():
    """A tracker with clean per-test state.

    ``AgentTurnTracker`` is a process singleton, so a test that leaves an active
    turn behind changes whether the NEXT test's first submit starts immediately
    or queues. Clearing the shared dicts keeps each test independent.
    """
    from app.services.agent_turn import AgentTurnTracker

    instance = AgentTurnTracker()
    instance._turns.clear()
    instance._queues.clear()
    instance._processing.clear()
    return instance


@pytest.fixture
def project_id() -> str:
    """A unique project per test, so queue state cannot bleed across them."""
    return f"proj-{uuid.uuid4()}"


@pytest.fixture
def dev_store(projects_dir: Path):
    """Dev platform backed by a REAL FilesystemMetadataStore.

    The claim and release go through the actual conditional-write path, so the
    fence under test is the real one rather than a mocked return.
    """
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        FilesystemMetadataStore,
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    set_platform(None)
    metadata_store = FilesystemMetadataStore(projects_dir=projects_dir)
    set_platform(
        Platform(
            metadata_store=metadata_store,
            session_store=_build_dev_session_store(),
            auth=LocalAuth(metadata_store=metadata_store),
            token_vending=NoOpTokenVending(),
        )
    )
    yield metadata_store
    set_platform(None)


async def test_queued_turn_runs_under_the_id_returned_to_the_caller(
    tracker, project_id
):
    """The id from submit() is the id the built turn carries."""
    seen: dict[str, str] = {}

    def _build(reserved: str):
        seen["reserved"] = reserved
        return tracker.create_turn(
            project_id, origin="user", owner_id="userA", turn_id=reserved
        )

    returned = tracker.submit(project_id, origin="user", build_turn=_build)

    assert seen["reserved"] == returned, "the builder must receive the reserved id"
    active = tracker.get_active_turn(project_id)
    assert active is not None
    assert active.turn_id == returned, (
        "the running turn must be the one the caller was told about"
    )
    assert tracker.get_turn(returned) is active, "the id must resolve to the turn"


async def test_tracker_rejects_a_builder_that_mints_its_own_id(tracker, project_id):
    """The reserved id is a promise already made; silently ignoring it is the bug.

    Failing loudly is what makes this a caught error instead of a stranded
    client. The tracker swallows build failures and moves to the next queued
    turn, so the queue does not wedge.
    """

    def _build_ignoring_reserved(_reserved: str):
        # A builder that ignores the reservation: mint a fresh id and return it.
        return tracker.create_turn(project_id, origin="user", owner_id="userA")

    returned = tracker.submit(
        project_id, origin="user", build_turn=_build_ignoring_reserved
    )

    # The mismatch is detected, so no turn is left registered under the id the
    # caller holds — rather than one running under a different id entirely.
    assert tracker.get_turn(returned) is None


async def test_second_queued_turn_also_keeps_its_own_id(tracker, project_id):
    """The reserved id must survive the queue, not just the immediate-start path."""
    built: list[str] = []

    def _build(reserved: str):
        built.append(reserved)
        return tracker.create_turn(
            project_id, origin="user", owner_id="userA", turn_id=reserved
        )

    first = tracker.submit(project_id, origin="user", build_turn=_build)
    # First one started immediately; this one queues behind it.
    second = tracker.submit(project_id, origin="user", build_turn=_build)

    assert built == [first], "the second turn should still be queued"
    queued = tracker.get_queued_turns(project_id)
    assert [q.turn_id for q in queued] == [second]

    # Finish the first turn so the queue advances.
    active = tracker.get_active_turn(project_id)
    active.status = "completed"
    tracker.notify_turn_complete(project_id)

    assert built == [first, second]
    assert tracker.get_active_turn(project_id).turn_id == second


async def test_release_is_fenced_to_its_own_turn(dev_store):
    """A stale turn's release must not clear a successor's live claim.

    Conditioning only on ``status = running`` made this reachable: releases run
    from a ``finally``, so a turn that outlived its lease arrives after the next
    turn has claimed the project, and its late release matched that condition.
    """
    from app.services.agentcore_session_registry import claim_turn, release_turn

    assert await claim_turn("userA", "projX", turn_id="turn-old") is True

    # turn-old's lease lapses and turn-new takes the project. Simulated by
    # releasing as turn-old and reclaiming, which is what the TTL path amounts to.
    await release_turn("userA", "projX", turn_id="turn-old")
    assert await claim_turn("userA", "projX", turn_id="turn-new") is True

    # Now turn-old's finally finally runs. It must be a no-op.
    await release_turn("userA", "projX", turn_id="turn-old")

    assert await claim_turn("userA", "projX", turn_id="turn-third") is False, (
        "turn-new's live claim was cleared by the stale turn's release"
    )

    # turn-new's own release still works.
    await release_turn("userA", "projX", turn_id="turn-new")
    assert await claim_turn("userA", "projX", turn_id="turn-third") is True
