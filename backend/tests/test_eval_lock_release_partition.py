# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The eval-lock release must key on the same partition the claim used.

``dispatch.py`` claims the eval lock under the resolved project OWNER
(``_resolve_owner_id_for_project``), and the durable write-through in
``JobTracker.update_state`` already coalesces ``owner_id`` then ``user_id`` for
exactly that reason. The lock RELEASE 80 lines later read only
``config["user_id"]``.

Two live shapes break on that:

* **A requester other than the owner** — the release targets a partition that
  holds no lock.
* **Optimization-path evals** — they carry ``owner_id`` and no ``user_id`` at
  all, so ``owner_id`` is None and the release is skipped entirely.

``release_eval_lock`` is holder-conditional and idempotent, so a wrong-partition
release *silently no-ops*. The real lock then survives to its 24h TTL and blocks
every subsequent dispatch on that project with no visible cause.

**These tests must use the CLOUD store.** The dev store's eval lock is keyed on
``project_id`` alone and ignores ``owner_id`` completely, so a dev-backed test
cannot observe a wrong-partition release at all — it would pass against the bug.
"""
from __future__ import annotations

import asyncio

import boto3
import pytest
from moto import mock_aws

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema (mirrors test_eval_lock's helper)."""
    ddb_client.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )


async def _drain_pending_writes(tracker) -> None:
    """Await the fire-and-forget durable tasks update_state schedules."""
    for _ in range(20):
        pending = [t for t in list(tracker._pending_writes) if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_extra", "case"),
    [
        ({"owner_id": "owner-1", "user_id": "other-2"}, "requester-is-not-owner"),
        ({"owner_id": "owner-1"}, "optimization-path (no user_id)"),
    ],
    ids=["requester_not_owner", "optimization_path"],
)
async def test_terminal_release_uses_owner_partition(config_extra, case, monkeypatch):
    """A terminal transition must release the lock the OWNER holds.

    Against the buggy code the release targets ``user_id`` — a different
    partition for the first case, and None (release skipped) for the
    optimization case. Either way the owner's lock survives and every later
    dispatch on the project is blocked until the 24h TTL.
    """
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table_name)

        from app.platform import Platform, set_platform
        from app.platform.dev_local import (
            LocalAuth,
            NoOpTokenVending,
            _build_dev_session_store,
        )
        from app.services.jobs import get_job_tracker

        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            tracker = get_job_tracker()
            tracker._reset()

            project_id = "project-1"
            job = tracker.create_job(
                project_id,
                job_type="eval",
                config={
                    "arch_split": True,
                    "model_ids": ["m"],
                    "prompt_version": "baseline",
                    **config_extra,
                },
            )

            # The OWNER holds the lock — this is what dispatch.py does.
            claimed = await store.claim_eval_lock("owner-1", project_id, job.job_id)
            assert claimed is True, "Owner must win the lock"
            assert await store.get_eval_lock("owner-1", project_id) is not None

            # Terminal transition — the single chokepoint that releases the lock.
            tracker.update_state(job.job_id, "completed")
            await _drain_pending_writes(tracker)

            lock = await store.get_eval_lock("owner-1", project_id)
            assert lock is None, (
                f"[{case}] The owner's eval lock MUST be released on terminal. It "
                f"is still held, so every later dispatch on this project is "
                f"blocked until the 24h TTL. Lock: {lock}"
            )
        finally:
            get_job_tracker()._reset()
            set_platform(None)


@pytest.mark.asyncio
async def test_release_does_not_touch_another_projects_lock(monkeypatch):
    """Counterpart: the release must stay scoped to its own project.

    Guards against widening the release into something that frees locks it does
    not hold — the lock is the one-eval-per-project serializer.
    """
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(ddb, table_name)
        store = DynamoDBMetadataStore(ddb_client=ddb, table_name=table_name)

        from app.platform import Platform, set_platform
        from app.platform.dev_local import (
            LocalAuth,
            NoOpTokenVending,
            _build_dev_session_store,
        )
        from app.services.jobs import get_job_tracker

        set_platform(
            Platform(
                metadata_store=store,
                session_store=_build_dev_session_store(),
                auth=LocalAuth(metadata_store=store),
                token_vending=NoOpTokenVending(),
            )
        )
        try:
            tracker = get_job_tracker()
            tracker._reset()

            job = tracker.create_job(
                "project-1",
                job_type="eval",
                config={
                    "arch_split": True,
                    "owner_id": "owner-1",
                    "model_ids": ["m"],
                    "prompt_version": "baseline",
                },
            )
            await store.claim_eval_lock("owner-1", "project-1", job.job_id)
            # A DIFFERENT project's lock, same owner, must be left alone.
            await store.claim_eval_lock("owner-1", "project-2", "other-job")

            tracker.update_state(job.job_id, "completed")
            await _drain_pending_writes(tracker)

            assert await store.get_eval_lock("owner-1", "project-1") is None
            assert await store.get_eval_lock("owner-1", "project-2") is not None, (
                "Another project's lock must NOT be released"
            )
        finally:
            get_job_tracker()._reset()
            set_platform(None)
