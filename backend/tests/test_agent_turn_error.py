# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentTurn.error field."""

import asyncio

from app.services.agent_turn import AgentTurn, get_agent_turn_tracker
from app.services.resource_error import Codes, ResourceError


class TestAgentTurnError:
    async def test_fail_without_error_leaves_error_none(self):
        turn = AgentTurn("t-1", "proj", owner_id="u-test")
        await turn.fail()
        assert turn.status == "failed"
        assert turn.error is None

    async def test_fail_with_error_attaches_it(self):
        turn = AgentTurn("t-2", "proj", owner_id="u-test")
        err = ResourceError(
            code=Codes.AGENT_SDK_STREAM_ERROR, message="stream died"
        )
        await turn.fail(err)
        assert turn.status == "failed"
        assert turn.error is err
        assert turn.error.code == Codes.AGENT_SDK_STREAM_ERROR

    async def test_tracker_reads_error_after_fail(self):
        tracker = get_agent_turn_tracker()
        tracker._reset()
        turn = tracker.create_turn("proj-a", origin="user", owner_id="u-test")
        err = ResourceError(code=Codes.AGENT_SDK_STREAM_ERROR, message="boom")
        await turn.fail(err)

        retrieved = tracker.get_turn(turn.turn_id)
        assert retrieved is not None
        assert retrieved.error is err
        tracker._reset()


class TestTurnQueueWedge:
    """A turn left at status=="running" (e.g. torn down by CancelledError,
    which bypasses run_agent_turn's ``except Exception``) must not wedge the
    project's chat queue.

    The wedge mechanism: ``get_active_turn`` returns any turn whose status is
    still "running"; ``_try_process_queue`` refuses to start a queued turn
    while ``get_active_turn`` is non-None. So a turn that never reached a
    terminal status blocks every subsequent turn. The fix forces a terminal
    status at the single ``finally`` chokepoint in ``run_agent_turn`` and,
    symmetrically, on the optimization-cancel interrupt branch.
    """

    def setup_method(self):
        get_agent_turn_tracker()._reset()

    def teardown_method(self):
        get_agent_turn_tracker()._reset()

    def test_running_turn_is_active_and_blocks_queue(self):
        """Baseline: a running turn IS returned by get_active_turn, which is
        what gates the queue. This pins the precondition the fix targets."""
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("proj-wedge", origin="orchestrator", owner_id="u-test")
        assert turn.status == "running"
        assert tracker.get_active_turn("proj-wedge") is turn

    def test_forcing_terminal_status_unblocks_active_lookup(self):
        """Once the stuck turn is marked terminal (what the run_agent_turn
        ``finally`` now guarantees), get_active_turn stops returning it — so
        _try_process_queue is free to start the next turn."""
        tracker = get_agent_turn_tracker()
        turn = tracker.create_turn("proj-wedge", origin="orchestrator", owner_id="u-test")

        # Simulate the hardened finally: a turn reaching cleanup still
        # "running" gets forced to "failed".
        if turn.status == "running":
            turn.status = "failed"

        assert tracker.get_active_turn("proj-wedge") is None

    def test_new_turn_runs_after_prior_left_running(self):
        """End-to-end: even if a prior turn is somehow still "running",
        create_turn force-fails it (agent_turn.py) AND a terminal status
        clears the active-turn gate, so a fresh turn becomes the active one
        rather than queuing forever behind a zombie."""
        tracker = get_agent_turn_tracker()
        stuck = tracker.create_turn("proj-wedge", origin="orchestrator", owner_id="u-test")
        # Leave it "running" — the zombie.
        assert stuck.status == "running"

        # A new turn for the same project must take over, not wedge.
        fresh = tracker.create_turn("proj-wedge", origin="user", owner_id="u-test")
        # create_turn force-fails the prior running turn.
        assert stuck.status == "failed"
        assert fresh.status == "running"
        assert tracker.get_active_turn("proj-wedge") is fresh
