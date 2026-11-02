# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cancel must not be able to finish before the worker it is cancelling starts.

A detached eval is visible, and therefore cancellable, well before its worker
exists. The manual path creates the job record deliberately ahead of a model
preflight that can take 15-20s — so the "starting" phase a user is most likely to
cancel during sits entirely inside the gap — and the snapshot upload and RunTask
follow it. The durable ``compute_handle`` is written only after RunTask returns.

So a cancel arriving in that window made two true observations — no handle, no
tagged ECS task — and drew a false conclusion from them: that there was nothing to
stop. It marked the job cancelled, published that to the SPA and released the
project's eval lock, and the dispatch then launched a worker for it anyway. That
worker ran the full eval, invoking target and judge models, with the UI reporting
the job gone and nothing left watching to stop it.

Retrying the task lookup cannot fix this at any budget: the task is in the future,
not merely invisible yet. These tests cover the thing that does fix it — one
first-wins claim per job that both sides consult — at the store, at the service
helper, and end to end through both dispatch paths and the cancel path.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from botocore.stub import ANY, Stubber

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.services.launch_gate import CANCEL_CLAIM, LAUNCH_CLAIM

pytestmark = pytest.mark.asyncio

OWNER = "owner-1"
PROJECT = "proj-1"
JOB = "job-1"
GATE_SK = f"LAUNCH_GATE#{PROJECT}#{JOB}"
TASK_ARN = "arn:aws:ecs:us-east-1:123456789012:task/cluster/abc123"


def _ddb() -> tuple[DynamoDBMetadataStore, Stubber]:
    client = boto3.client(
        "dynamodb",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    return DynamoDBMetadataStore(ddb_client=client), Stubber(client)


# ---------------------------------------------------------------------------
# Store: the claim itself
# ---------------------------------------------------------------------------


async def test_the_first_claim_wins_and_is_written_conditionally():
    """attribute_not_exists is what makes this a race winner rather than a write."""
    store, stubber = _ddb()
    stubber.add_response(
        "put_item",
        {},
        {
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(SK)",
        },
    )
    stubber.activate()
    try:
        winner = await store.claim_launch_gate(
            OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
        )
        assert winner == LAUNCH_CLAIM
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


async def test_a_claim_carries_a_ttl_so_it_cannot_outlive_the_job():
    """An untidied claim would block a later cancel forever."""
    store, stubber = _ddb()
    captured: dict = {}

    stubber.add_response("put_item", {}, {"TableName": ANY, "Item": ANY,
                                          "ConditionExpression": ANY})
    stubber.activate()
    try:
        with patch.object(
            store.client, "put_item", side_effect=lambda **kw: captured.update(kw) or {}
        ):
            await store.claim_launch_gate(OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM)
    finally:
        stubber.deactivate()

    assert "ttl" in captured["Item"], "the gate item must expire on its own"
    assert int(captured["Item"]["ttl"]["N"]) > 0


async def test_the_second_claim_loses_and_learns_who_holds_it():
    """Losing is not enough — the loser has to know which side won.

    Dispatch and cancel take opposite branches on the answer, so reporting a bare
    "you lost" would leave both of them unable to act.
    """
    store, stubber = _ddb()
    stubber.add_client_error(
        "put_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="already claimed",
    )
    stubber.add_response(
        "get_item",
        {
            "Item": {
                "PK": {"S": f"USER#{OWNER}"},
                "SK": {"S": GATE_SK},
                "claimant": {"S": CANCEL_CLAIM},
            }
        },
        {
            "TableName": "ai-eval-studio-metadata",
            "Key": {"PK": {"S": f"USER#{OWNER}"}, "SK": {"S": GATE_SK}},
            "ConsistentRead": True,
        },
    )
    stubber.activate()
    try:
        winner = await store.claim_launch_gate(
            OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
        )
        assert winner == CANCEL_CLAIM
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


async def test_a_claim_released_mid_read_is_retried_not_reported_as_held():
    """The gate can be released between the failed write and the read.

    That is a dispatch abandoning a launch that never happened. Reporting a holder
    that no longer exists would deadlock the next caller against nothing.
    """
    store, stubber = _ddb()
    stubber.add_client_error(
        "put_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="already claimed",
    )
    stubber.add_response("get_item", {}, None)  # gone by the time we look
    stubber.add_response(
        "put_item",
        {},
        {"TableName": ANY, "Item": ANY, "ConditionExpression": ANY},
    )
    stubber.activate()
    try:
        winner = await store.claim_launch_gate(
            OWNER, PROJECT, JOB, claimant=CANCEL_CLAIM
        )
        assert winner == CANCEL_CLAIM
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


async def test_a_release_cannot_clear_the_other_sides_claim():
    """A dispatch that lost the race must not be able to undo the cancel."""
    store, stubber = _ddb()
    stubber.add_response(
        "delete_item",
        {},
        {
            "TableName": "ai-eval-studio-metadata",
            "Key": {"PK": {"S": f"USER#{OWNER}"}, "SK": {"S": GATE_SK}},
            "ConditionExpression": "claimant = :c",
            "ExpressionAttributeValues": {":c": {"S": LAUNCH_CLAIM}},
        },
    )
    stubber.activate()
    try:
        assert await store.release_launch_gate(
            OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
        )
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


async def test_releasing_a_claim_held_by_the_other_side_is_refused():
    store, stubber = _ddb()
    stubber.add_client_error(
        "delete_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="not yours",
    )
    stubber.activate()
    try:
        assert not await store.release_launch_gate(
            OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
        )
    finally:
        stubber.deactivate()


# ---------------------------------------------------------------------------
# Dev store: same contract, so no caller has to ask which mode it is in
# ---------------------------------------------------------------------------


async def test_dev_store_is_first_wins_too():
    from app.platform.dev_local import FilesystemMetadataStore

    store = FilesystemMetadataStore(projects_dir=Path("/tmp/nonexistent"))

    assert await store.claim_launch_gate(
        OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
    ) == LAUNCH_CLAIM
    assert await store.claim_launch_gate(
        OWNER, PROJECT, JOB, claimant=CANCEL_CLAIM
    ) == LAUNCH_CLAIM, "the second claimant must learn the first one won"
    assert not await store.release_launch_gate(
        OWNER, PROJECT, JOB, claimant=CANCEL_CLAIM
    ), "one side must not release the other's claim"
    assert await store.release_launch_gate(
        OWNER, PROJECT, JOB, claimant=LAUNCH_CLAIM
    )
    assert await store.claim_launch_gate(
        OWNER, PROJECT, JOB, claimant=CANCEL_CLAIM
    ) == CANCEL_CLAIM, "a released gate is claimable again"


async def test_dev_store_keeps_gates_per_job():
    """One job's launch must not gate another's."""
    from app.platform.dev_local import FilesystemMetadataStore

    store = FilesystemMetadataStore(projects_dir=Path("/tmp/nonexistent"))
    await store.claim_launch_gate(OWNER, PROJECT, "job-a", claimant=CANCEL_CLAIM)

    assert await store.claim_launch_gate(
        OWNER, PROJECT, "job-b", claimant=LAUNCH_CLAIM
    ) == LAUNCH_CLAIM


# ---------------------------------------------------------------------------
# Service helper: every failure answered in the safe direction
# ---------------------------------------------------------------------------


def _platform_with(**store_attrs) -> MagicMock:
    store = MagicMock()
    for name, value in store_attrs.items():
        setattr(store, name, value)
    return MagicMock(metadata_store=store)


async def test_a_dispatch_that_wins_the_gate_proceeds():
    from app.services import launch_gate

    platform = _platform_with(
        claim_launch_gate=AsyncMock(side_effect=lambda *a, claimant, **k: claimant)
    )
    with patch("app.platform.get_platform", return_value=platform):
        assert await launch_gate.claim_for_launch(
            OWNER, PROJECT, JOB
        ) is launch_gate.LaunchDecision.PROCEED


async def test_a_dispatch_that_loses_the_gate_is_told_the_job_was_cancelled():
    from app.services import launch_gate

    platform = _platform_with(
        claim_launch_gate=AsyncMock(return_value=CANCEL_CLAIM)
    )
    with patch("app.platform.get_platform", return_value=platform):
        assert await launch_gate.claim_for_launch(
            OWNER, PROJECT, JOB
        ) is launch_gate.LaunchDecision.CANCELLED


async def test_an_unsettled_gate_is_distinguished_from_a_cancelled_job():
    """Both decline the launch, but they are not the same fact.

    Reporting one outcome for both made the caller tell the user the evaluation had
    been cancelled when the truth was that the store could not be reached — and
    left the job unowned, because the caller could not tell that no cancel was
    coming to settle it.
    """
    from app.services import launch_gate

    platform = _platform_with(
        claim_launch_gate=AsyncMock(side_effect=RuntimeError("ddb down"))
    )
    with patch("app.platform.get_platform", return_value=platform):
        assert await launch_gate.claim_for_launch(
            OWNER, PROJECT, JOB
        ) is launch_gate.LaunchDecision.UNAVAILABLE


async def test_a_cancel_that_loses_the_gate_may_not_claim_there_is_nothing_to_stop():
    from app.services import launch_gate

    platform = _platform_with(
        claim_launch_gate=AsyncMock(return_value=LAUNCH_CLAIM)
    )
    with patch("app.platform.get_platform", return_value=platform):
        assert not await launch_gate.claim_for_cancel(OWNER, PROJECT, JOB)


async def test_an_unsettled_gate_stops_a_cancel_reporting_a_stop():
    from app.services import launch_gate

    platform = _platform_with(
        claim_launch_gate=AsyncMock(side_effect=RuntimeError("ddb down"))
    )
    with patch("app.platform.get_platform", return_value=platform):
        assert not await launch_gate.claim_for_cancel(OWNER, PROJECT, JOB)


async def test_abandoning_a_launch_releases_only_the_launch_claim():
    from app.services import launch_gate

    release = AsyncMock(return_value=True)
    platform = _platform_with(release_launch_gate=release)
    with patch("app.platform.get_platform", return_value=platform):
        await launch_gate.abandon_launch(OWNER, PROJECT, JOB)

    assert release.await_args.kwargs["claimant"] == LAUNCH_CLAIM


# ---------------------------------------------------------------------------
# End to end: the cancel path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """Collapse every backoff. None of these tests are about durations."""
    monkeypatch.setattr("app.services.eval_cancel._STOP_CONFIRM_DELAY_S", 0)
    monkeypatch.setattr("app.services.eval_cancel._TAG_LOOKUP_DELAY_S", 0)
    monkeypatch.setattr("app.services.eval_cancel._PENDING_LAUNCH_DELAY_S", 0)


def _tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.rebuild_jobs_from_store = AsyncMock()
    tracker.get_job_for_project.return_value = MagicMock(status="running", pid=None)
    tracker.cancel_job = MagicMock()
    return tracker


async def _cancel(tracker, compute, *, gate_winner, handles=None):
    """Drive cancel_eval_job for a handle-less cloud job with a given gate outcome.

    ``handles`` is the sequence ``_resolve_compute_handle`` returns, so a test can
    make the dispatch publish its handle partway through the cancel's chase.
    """
    from app.services.eval_cancel import cancel_eval_job

    resolve = AsyncMock(side_effect=handles) if handles else AsyncMock(return_value=None)
    with patch(
        "app.services.eval_cancel.get_job_tracker", return_value=tracker
    ), patch(
        "app.api.eval_jobs._resolve_compute_handle", new=resolve
    ), patch(
        "app.platform.job_compute.get_job_compute", return_value=compute
    ), patch(
        "app.services.eval_cancel._stop_optimization_loop", new=AsyncMock()
    ), patch(
        "app.platform.get_platform",
        return_value=_platform_with(
            claim_launch_gate=AsyncMock(return_value=gate_winner)
        ),
    ):
        return await cancel_eval_job(
            PROJECT,
            JOB,
            user_id=OWNER,
            project_dir=Path("/tmp/nonexistent"),
            arch_split=True,
        )


def _compute_with_no_task() -> MagicMock:
    compute = MagicMock()
    compute.cancel = AsyncMock()
    compute.cancel_by_tag = AsyncMock(return_value=None)
    compute.status = AsyncMock(return_value="STOPPED")
    return compute


async def test_winning_the_gate_turns_a_lookup_miss_into_proof():
    """No worker, and none can start now, so cancelling is sound."""
    compute = _compute_with_no_task()
    tracker = _tracker()

    outcome, _ = await _cancel(tracker, compute, gate_winner=CANCEL_CLAIM)

    assert outcome == "cancelled"
    tracker.cancel_job.assert_called_once()


async def test_losing_the_gate_stops_the_worker_once_its_handle_appears():
    """The fix has to still cancel, not merely refuse to lie about it.

    The dispatch won the gate, so it is launching. It publishes the handle right
    after RunTask returns, and the cancel picks it up on a later pass and stops it.
    """
    compute = _compute_with_no_task()
    tracker = _tracker()

    outcome, _ = await _cancel(
        tracker,
        compute,
        gate_winner=LAUNCH_CLAIM,
        handles=[None, None, TASK_ARN],
    )

    assert outcome == "cancelled"
    compute.cancel.assert_awaited_once()
    assert compute.cancel.await_args.args[0].value == TASK_ARN
    tracker.cancel_job.assert_called_once()


async def test_losing_the_gate_never_reports_a_stop_it_could_not_make():
    """This is the bug, stated as a test.

    The dispatch holds the gate and the cancel never finds the worker. Before, the
    lookup misses alone concluded "nothing to stop" and the job went terminal while
    a worker started behind it. Now the job stays visible so a retry and the
    reconciler can still act.
    """
    compute = _compute_with_no_task()
    tracker = _tracker()

    outcome, status = await _cancel(tracker, compute, gate_winner=LAUNCH_CLAIM)

    assert outcome == "stop_failed"
    assert status == "running"
    tracker.cancel_job.assert_not_called()


async def test_a_tagged_worker_is_stopped_without_ever_consulting_the_gate():
    """Positive evidence beats any inference, so the lookup still comes first.

    A long-running handle-less worker is found by its tag. Reaching for the gate
    before looking would let an expired claim conclude there was nothing to stop
    while that worker was plainly running.
    """
    compute = _compute_with_no_task()
    compute.cancel_by_tag = AsyncMock(return_value=TASK_ARN)
    tracker = _tracker()
    claim = AsyncMock(return_value=CANCEL_CLAIM)

    from app.services.eval_cancel import cancel_eval_job

    with patch(
        "app.services.eval_cancel.get_job_tracker", return_value=tracker
    ), patch(
        "app.api.eval_jobs._resolve_compute_handle", new=AsyncMock(return_value=None)
    ), patch(
        "app.platform.job_compute.get_job_compute", return_value=compute
    ), patch(
        "app.services.eval_cancel._stop_optimization_loop", new=AsyncMock()
    ), patch(
        "app.platform.get_platform",
        return_value=_platform_with(claim_launch_gate=claim),
    ):
        outcome, _ = await cancel_eval_job(
            PROJECT, JOB, user_id=OWNER,
            project_dir=Path("/tmp/nonexistent"), arch_split=True,
        )

    assert outcome == "cancelled"
    # A found task settles the question; the gate must not be touched at all.
    claim.assert_not_awaited()


# ---------------------------------------------------------------------------
# End to end: both dispatch paths decline to launch
# ---------------------------------------------------------------------------


async def test_the_optimization_dispatch_does_not_launch_after_losing_the_gate():
    from app.services.optimization_orchestrator import (
        OptimizationConfig,
        dispatch_eval_job,
    )

    cfg = OptimizationConfig(
        project_id=PROJECT,
        owner_id=OWNER,
        model_ids=["m1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="baseline",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=10,
        plateau_window=3,
    )

    tracker = MagicMock()
    tracker.create_job.return_value = MagicMock(job_id=JOB)
    compute = MagicMock()
    compute.launch = AsyncMock()
    store = AsyncMock()
    store.claim_launch_gate = AsyncMock(return_value=CANCEL_CLAIM)

    with patch("app.services.jobs.get_job_tracker", return_value=tracker), \
         patch.object(Path, "exists", return_value=True), \
         patch("app.platform.job_compute.get_job_compute", return_value=compute), \
         patch("app.platform.get_platform") as platform:
        platform.return_value.metadata_store = store
        job_id = await dispatch_eval_job(
            Path("/tmp/proj"), cfg, "baseline", None, arch_split=True
        )

    assert job_id == JOB
    # A cancel already settled this job; starting a worker is the whole bug.
    compute.launch.assert_not_awaited()
    store.update_job_state.assert_not_awaited()


async def test_an_unreachable_gate_fails_the_optimization_eval_rather_than_stranding_it():
    """Nobody else owns a job whose gate could not be settled.

    Unlike the cancelled case there is no cancel coming to publish a terminal state,
    so silently returning would leave a job reading live forever with no worker
    behind it, waiting on a reconciler to notice.
    """
    from app.services.optimization_orchestrator import (
        OptimizationConfig,
        dispatch_eval_job,
    )

    cfg = OptimizationConfig(
        project_id=PROJECT,
        owner_id=OWNER,
        model_ids=["m1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="baseline",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=10,
        plateau_window=3,
    )

    tracker = MagicMock()
    tracker.create_job.return_value = MagicMock(job_id=JOB)
    compute = MagicMock()
    compute.launch = AsyncMock()
    store = AsyncMock()
    store.claim_launch_gate = AsyncMock(side_effect=RuntimeError("ddb down"))

    with patch("app.services.jobs.get_job_tracker", return_value=tracker), \
         patch.object(Path, "exists", return_value=True), \
         patch("app.platform.job_compute.get_job_compute", return_value=compute), \
         patch("app.platform.get_platform") as platform:
        platform.return_value.metadata_store = store
        with pytest.raises(RuntimeError):
            await dispatch_eval_job(
                Path("/tmp/proj"), cfg, "baseline", None, arch_split=True
            )

    compute.launch.assert_not_awaited()


async def test_a_failed_launch_hands_the_gate_back():
    """Otherwise every later cancel waits out its window for a worker that
    never existed, and reports failure for it."""
    from app.services.optimization_orchestrator import (
        OptimizationConfig,
        dispatch_eval_job,
    )

    cfg = OptimizationConfig(
        project_id=PROJECT,
        owner_id=OWNER,
        model_ids=["m1"],
        data_file="train.csv",
        field_mappings={},
        prompt_version="baseline",
        judge_model_id=None,
        target_metric_scores={},
        max_iterations=10,
        plateau_window=3,
    )

    tracker = MagicMock()
    tracker.create_job.return_value = MagicMock(job_id=JOB)
    compute = MagicMock()
    compute.launch = AsyncMock(side_effect=RuntimeError("RunTask rejected"))
    store = AsyncMock()
    store.claim_launch_gate = AsyncMock(side_effect=lambda *a, claimant, **k: claimant)
    store.release_launch_gate = AsyncMock(return_value=True)

    with patch("app.services.jobs.get_job_tracker", return_value=tracker), \
         patch.object(Path, "exists", return_value=True), \
         patch("app.platform.job_compute.get_job_compute", return_value=compute), \
         patch("app.platform.get_platform") as platform:
        platform.return_value.metadata_store = store
        with pytest.raises(RuntimeError):
            await dispatch_eval_job(
                Path("/tmp/proj"), cfg, "baseline", None, arch_split=True
            )

    store.release_launch_gate.assert_awaited_once()
    assert store.release_launch_gate.await_args.kwargs["claimant"] == LAUNCH_CLAIM


async def test_the_manual_dispatch_does_not_launch_after_losing_the_gate():
    """The path with the 15-20s preflight, which is where the window is widest."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _tool

    dispatch_eval = _tool.handler
    compute = MagicMock()
    compute.launch = AsyncMock()

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as tracker,
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
        patch("app.platform.arch_split.resolve_arch_split", return_value=True),
        patch("app.platform.job_compute.get_job_compute", return_value=compute),
        patch("app.services.mcp_tools.dispatch._publish_project_event"),
        patch("app.platform.types.current_request") as req,
    ):
        resolve.return_value = MagicMock()
        resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

        tracker_inst = MagicMock()
        tracker_inst.create_job.return_value = MagicMock(
            job_id=JOB, project_id=PROJECT
        )
        tracker_inst.get_active_job_by_type.return_value = None
        tracker.return_value = tracker_inst
        req.return_value = MagicMock(user_id=OWNER)

        store = AsyncMock()
        store.count_active_jobs_for_owner = AsyncMock(return_value=0)
        store.list_non_terminal_jobs = AsyncMock(return_value=[])
        store.claim_eval_lock = AsyncMock(return_value=True)
        store.claim_idempotency_key = AsyncMock(return_value=(True, None))
        # The optimization-live fence reads this before anything else. An
        # AsyncMock would hand back a MagicMock that the state loader then
        # treats as a real record, which is where the unawaited-coroutine
        # warnings came from.
        store.get_optimization_state = AsyncMock(return_value=None)
        store.claim_launch_gate = AsyncMock(return_value=CANCEL_CLAIM)

        with patch(
            "app.platform.get_platform",
            return_value=MagicMock(metadata_store=store),
        ):
            result = await dispatch_eval({
                "project_id": PROJECT,
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

    # The job was already cancelled; no worker may be started for it.
    compute.launch.assert_not_awaited()
    assert result.get("is_error"), "the caller has to observe that nothing launched"
    assert "cancelled" in result["content"][0]["text"].lower()


async def test_the_manual_dispatch_does_not_call_an_unreachable_gate_a_cancellation():
    """An unreachable store is not a cancellation, and must not be reported as one.

    It also has to fail the job: the cancelled path can leave the terminal write to
    the cancel that made it, but here no cancel exists, so the dispatch owns it.
    """
    from app.services.mcp_tools.dispatch import dispatch_eval as _tool

    dispatch_eval = _tool.handler
    compute = MagicMock()
    compute.launch = AsyncMock()

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as tracker,
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
        patch("app.platform.arch_split.resolve_arch_split", return_value=True),
        patch("app.platform.job_compute.get_job_compute", return_value=compute),
        patch("app.services.mcp_tools.dispatch._publish_project_event"),
        patch("app.platform.types.current_request") as req,
    ):
        resolve.return_value = MagicMock()
        resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

        tracker_inst = MagicMock()
        tracker_inst.create_job.return_value = MagicMock(
            job_id=JOB, project_id=PROJECT
        )
        tracker_inst.get_active_job_by_type.return_value = None
        tracker.return_value = tracker_inst
        req.return_value = MagicMock(user_id=OWNER)

        store = AsyncMock()
        store.count_active_jobs_for_owner = AsyncMock(return_value=0)
        store.list_non_terminal_jobs = AsyncMock(return_value=[])
        store.claim_eval_lock = AsyncMock(return_value=True)
        store.claim_idempotency_key = AsyncMock(return_value=(True, None))
        # The optimization-live fence reads this before anything else. An
        # AsyncMock would hand back a MagicMock that the state loader then
        # treats as a real record, which is where the unawaited-coroutine
        # warnings came from.
        store.get_optimization_state = AsyncMock(return_value=None)
        store.claim_launch_gate = AsyncMock(side_effect=RuntimeError("ddb down"))

        with patch(
            "app.platform.get_platform",
            return_value=MagicMock(metadata_store=store),
        ):
            result = await dispatch_eval({
                "project_id": PROJECT,
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

    compute.launch.assert_not_awaited()
    assert result.get("is_error")
    assert "cancelled" not in result["content"][0]["text"].lower(), (
        "nothing established that this job was cancelled"
    )
    # The terminal transition is what releases the eval lock, so it has to happen
    # here rather than being left to the reconciler.
    states = [c.args[1] for c in tracker_inst.update_state.call_args_list]
    assert "failed" in states, "an unowned job must be failed, not left pending"
