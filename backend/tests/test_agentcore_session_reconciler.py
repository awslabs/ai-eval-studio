# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agentcore_session_reconciler (startup orphan recovery)."""
from __future__ import annotations

import time
from typing import Any

import pytest
from moto import mock_aws

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import FilesystemMetadataStore
from app.services.agentcore_session_reconciler import reconcile_orphaned_agentcore_claims


def _create_metadata_table(dynamodb_client: Any) -> str:
    """Create single-table with PK/SK + GSI1 (mirrors test_session_registry helper)."""
    table_name = "test-metadata-table"
    dynamodb_client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
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
    )
    return table_name


@pytest.fixture
def cloud_store() -> DynamoDBMetadataStore:
    """Cloud store (moto) fixture."""
    with mock_aws():
        import boto3

        dynamodb = boto3.client("dynamodb", region_name="us-east-1")
        table_name = _create_metadata_table(dynamodb)
        store = DynamoDBMetadataStore(ddb_client=dynamodb, table_name=table_name)
        yield store


@pytest.fixture
def dev_store(tmp_path: Any) -> FilesystemMetadataStore:
    """Dev store fixture."""
    return FilesystemMetadataStore(projects_dir=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_reconcile_clears_orphaned_running_claim(
    store_fixture: str, request: Any
) -> None:
    """Test: reconciler clears a stranded 'running' claim at startup.

    Strand a running claim (claim without release), run the reconciler,
    assert the claim flips to idle and the count is 1. Proves the startup
    reconciler clears orphaned claims (the recovery guarantee).
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # Strand a running claim (claim, no release)
    won = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert won is True

    # Verify it's running
    session = await store.get_agentcore_session(user_id, project_id)
    assert session is not None
    assert session["status"] == "running"

    # Mock the platform to use this store
    from app.platform import Platform, set_platform
    from app.platform.dev_local import LocalAuth, NoOpTokenVending, _build_dev_session_store

    set_platform(None)
    session_store = _build_dev_session_store()
    auth = LocalAuth(metadata_store=store)
    token_vending = NoOpTokenVending()
    platform = Platform(
        metadata_store=store,
        session_store=session_store,
        auth=auth,
        token_vending=token_vending,
    )
    set_platform(platform)

    # Run the reconciler (simulates startup recovery)
    result = await reconcile_orphaned_agentcore_claims()

    # Verify the claim was cleared
    assert result["claims_cleared"] == 1

    # Verify status is now idle
    session_after = await store.get_agentcore_session(user_id, project_id)
    assert session_after is not None
    assert session_after["status"] == "idle"

    # Cleanup
    set_platform(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_reconcile_version_fence_protects_reclaimed_claim(
    store_fixture: str, request: Any
) -> None:
    """Version fence test: reconciler does NOT clear a claim reclaimed after list.

    Strand a running claim, snapshot its version, then release+reclaim it
    (bumps version), then simulate the reconciler clearing with the OLD
    version. The version fence should prevent the clear (protecting the
    now-legitimate claim from being cleared).
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # Strand a running claim
    won = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert won is True

    # Read the version (simulates reconciler list snapshot)
    session1 = await store.get_agentcore_session(user_id, project_id)
    assert session1 is not None
    version_1 = session1["claim_version"]

    # Release and reclaim (bumps version) — simulates a legitimate reclaim
    # between the reconciler's list and clear
    await store.release_agentcore_session_turn(user_id, project_id)
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)

    # Read the new version
    session2 = await store.get_agentcore_session(user_id, project_id)
    assert session2 is not None
    version_2 = session2["claim_version"]
    assert version_2 > version_1  # Version incremented

    # Reconciler tries to clear with stale version_1 (the version from before reclaim)
    cleared = await store.clear_stale_agentcore_session_claim(
        user_id, project_id, expected_version=version_1
    )
    assert cleared is False, "Clear with stale version should fail (fence tripped)"

    # Claim is still running (fence protected it)
    session3 = await store.get_agentcore_session(user_id, project_id)
    assert session3 is not None
    assert session3["status"] == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_list_running_returns_only_running_sessions(
    store_fixture: str, request: Any
) -> None:
    """list_running_agentcore_sessions filters to status=running only.

    Create one running session and one idle session, assert list_running
    returns only the running one.
    """
    store = request.getfixturevalue(store_fixture)

    user_a = "user-a"
    user_b = "user-b"
    proj_x = "proj-x"
    proj_y = "proj-y"
    ttl = int(time.time()) + 3600

    # User A project X: claim (running)
    await store.claim_agentcore_session_turn(user_a, proj_x, ttl=ttl)

    # User B project Y: claim then release (idle)
    await store.claim_agentcore_session_turn(user_b, proj_y, ttl=ttl)
    await store.release_agentcore_session_turn(user_b, proj_y)

    # List running sessions
    running = await store.list_running_agentcore_sessions()

    # Only user_a/proj_x should be in the list
    assert len(running) == 1
    assert running[0]["user_id"] == user_a
    assert running[0]["project_id"] == proj_x
    assert running[0]["status"] == "running"


# ---------------------------------------------------------------------------
# Least-privilege enforcement
#
# The reconciler shipped correct-looking and inert: it ran, and every claim it
# should have cleared stayed held, because the coordinator role has no table-wide
# dynamodb:Scan and the old implementation scanned. The exception was swallowed by
# design, so startup looked healthy while the redeploy-wedge fix did nothing.
#
# A passing unit test and a passing wiring test both missed it, because neither ran
# the reconciler under the permissions it actually has in production. These tests
# close that: they exercise the real store through a client that refuses anything
# the deployed policy does not allow.
# ---------------------------------------------------------------------------


class _LeastPrivilegeDenied(Exception):
    """Raised when the reconciler attempts an API call the task role lacks."""


def _deny_operations(store: Any, denied: set[str]) -> None:
    """Make named DynamoDB operations raise, mimicking an IAM denial.

    Wraps the store's botocore client so any call to a denied operation raises
    instead of succeeding. Paginators are covered too, because botocore builds
    them from the same client method.

    Args:
        store: DynamoDBMetadataStore whose client should be constrained.
        denied: Operation names to refuse (e.g. ``{"scan"}``).
    """
    real_client = store.client

    class _Constrained:
        def __getattr__(self, name: str) -> Any:
            if name in denied:
                raise _LeastPrivilegeDenied(
                    f"coordinator-task-role is not authorized to perform: "
                    f"dynamodb:{name.title()}"
                )
            return getattr(real_client, name)

        def get_paginator(self, operation_name: str) -> Any:
            if operation_name in denied:
                raise _LeastPrivilegeDenied(
                    f"coordinator-task-role is not authorized to perform: "
                    f"dynamodb:{operation_name.title()}"
                )
            return real_client.get_paginator(operation_name)

    store.client = _Constrained()


@pytest.mark.asyncio
async def test_reconcile_clears_claim_without_scan_permission(
    cloud_store: Any,
) -> None:
    """The reconciler clears an orphaned claim using only Query, never Scan.

    This is the test that fails if the reconciler cannot complete its lookup under
    the deployed policy. `dynamodb:Scan` is absent from the coordinator role, so
    denying it here reproduces production exactly — the previous Scan-based
    implementation fails, and the effect (a cleared claim) is asserted, not just
    that the call returned.
    """
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    user_id, project_id = "user-lp", "proj-lp"

    # Strand a running claim, then take Scan away.
    won = await cloud_store.claim_agentcore_session_turn(
        user_id, project_id, ttl=int(time.time()) + 7200, current_turn_id="turn-lp"
    )
    assert won
    _deny_operations(cloud_store, {"scan"})

    set_platform(None)
    set_platform(
        Platform(
            metadata_store=cloud_store,
            session_store=_build_dev_session_store(),
            auth=LocalAuth(metadata_store=cloud_store),
            token_vending=NoOpTokenVending(),
        )
    )
    try:
        result = await reconcile_orphaned_agentcore_claims()

        # The effect, not the call: the claim must actually be released.
        assert result["claims_cleared"] == 1, (
            "reconciler cleared nothing — it could not complete its lookup under "
            "the coordinator's real permissions (no dynamodb:Scan)"
        )
        session = await cloud_store.get_agentcore_session(user_id, project_id)
        assert session["status"] == "idle"

        # And the (user, project) is usable again — the whole point of the sweep.
        assert await cloud_store.claim_agentcore_session_turn(
            user_id, project_id, ttl=int(time.time()) + 7200
        ), "claim still wedged after reconcile"
    finally:
        set_platform(None)


@pytest.mark.asyncio
async def test_running_claims_index_is_sparse(cloud_store: Any) -> None:
    """Only HELD claims appear in the running-claims index.

    The Query is only bounded if released claims leave the index. If a release did
    not remove the GSI keys, the reconciler would re-find every claim ever taken
    and the index would grow without bound — a Scan by another name.
    """
    ttl = int(time.time()) + 7200

    await cloud_store.claim_agentcore_session_turn("u1", "p1", ttl=ttl)
    await cloud_store.claim_agentcore_session_turn("u2", "p2", ttl=ttl)
    held = await cloud_store.list_running_agentcore_sessions()
    assert {(s["user_id"], s["project_id"]) for s in held} == {("u1", "p1"), ("u2", "p2")}

    # A normal release must drop out of the index.
    await cloud_store.release_agentcore_session_turn("u1", "p1")
    remaining = await cloud_store.list_running_agentcore_sessions()
    assert {(s["user_id"], s["project_id"]) for s in remaining} == {("u2", "p2")}, (
        "a released claim is still in the running-claims index — the release must "
        "REMOVE the GSI keys or the index grows without bound"
    )

    # A reconciler clear must drop out too.
    session = await cloud_store.get_agentcore_session("u2", "p2")
    assert await cloud_store.clear_stale_agentcore_session_claim(
        "u2", "p2", expected_version=session["claim_version"]
    )
    assert await cloud_store.list_running_agentcore_sessions() == [], (
        "a cleared claim is still in the running-claims index"
    )
