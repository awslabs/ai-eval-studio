# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the AgentCore session registry.

Tests the durable (user, project) session registry in both adapters (cloud moto
+ dev) for: first-wins claim, per-(user,project) grain, idempotent release,
identity upsert without forging a claim, and version-fenced stale-claim reaper.
"""
from __future__ import annotations

import time
from typing import Any

import pytest
from moto import mock_aws

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import FilesystemMetadataStore


def _create_metadata_table(dynamodb_client: Any) -> str:
    """Create single-table with PK/SK + GSI1 (mirrors test_eval_lock helper)."""
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
async def test_second_claim_fails_real_conflict(
    store_fixture: str, request: Any
) -> None:
    """First claim on (userA, projX) → True; second claim → False (REAL conflict).

    The second claim hits a REAL conditional-write conflict, not a mocked
    return. Proves the first-wins serialization works end-to-end.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # First claim wins
    result1 = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert result1 is True

    # Second claim on SAME (user, project) while running → False
    result2 = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert result2 is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_grain_per_user_project_not_per_user(
    store_fixture: str, request: Any
) -> None:
    """Claim (userA, projX) True AND claim (userA, projY) True concurrently.

    Proves the grain is per-(user,project), NOT per-user. A per-USER claim
    would make the second False. Both return True (grain proof).
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_x = "proj-x"
    project_y = "proj-y"
    ttl = int(time.time()) + 3600

    # Claim on (userA, projX) → True
    result_x = await store.claim_agentcore_session_turn(user_id, project_x, ttl=ttl)
    assert result_x is True

    # Claim on (userA, projY) while X is running → True (different project)
    result_y = await store.claim_agentcore_session_turn(user_id, project_y, ttl=ttl)
    assert result_y is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_release_then_reclaim(store_fixture: str, request: Any) -> None:
    """Release then reclaim → True.

    Proves release makes the claim available again.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # Claim
    result1 = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert result1 is True

    # Release
    await store.release_agentcore_session_turn(user_id, project_id)

    # Reclaim → True
    result2 = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert result2 is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_release_is_idempotent(store_fixture: str, request: Any) -> None:
    """Release is idempotent (double release, release-when-idle = no raise).

    Proves idempotent release: release when not running / already gone / never
    claimed → no exception, status stays idle.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # Claim then release
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    await store.release_agentcore_session_turn(user_id, project_id)

    # Double release → no exception
    await store.release_agentcore_session_turn(user_id, project_id)

    # Get session → status is idle
    session = await store.get_agentcore_session(user_id, project_id)
    assert session is not None
    assert session["status"] == "idle"

    # Release a never-claimed session → no exception
    await store.release_agentcore_session_turn("user-b", "proj-y")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_put_identity_does_not_forge_claim(
    store_fixture: str, request: Any
) -> None:
    """put_agentcore_session records runtime/app session ids; get returns them;
    putting the mapping while status=idle does NOT set status=running.

    Proves the identity upsert is separate from the claim — updating the
    mapping mid-idle does not forge a claim.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    runtime_session_id = "runtime-123"
    app_session_id = "app-456"
    ttl = int(time.time()) + 3600

    # Put identity while idle (no claim yet)
    await store.put_agentcore_session(
        user_id,
        project_id,
        runtime_session_id=runtime_session_id,
        app_session_id=app_session_id,
        ttl=ttl,
    )

    # Get session → status is idle, not running
    session = await store.get_agentcore_session(user_id, project_id)
    assert session is not None
    assert session["status"] == "idle"
    assert session["runtime_session_id"] == runtime_session_id
    assert session["app_session_id"] == app_session_id

    # Claim still works (status was idle)
    result = await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)
    assert result is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_stale_claim_fence(store_fixture: str, request: Any) -> None:
    """Stale-claim fence: clear with matching claim_version → True + status idle;
    clear with stale version after release+reclaim → False (fence holds).

    Proves the version fence prevents clearing a claim that was
    released+reclaimed since read (reconciler grace). claim_version
    is the monotonic fence token that survives release and increments on every
    claim.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    ttl = int(time.time()) + 3600

    # Claim
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)

    # Read the claim_version
    session1 = await store.get_agentcore_session(user_id, project_id)
    assert session1 is not None
    version_1 = session1["claim_version"]

    # Release then reclaim (bumps claim_version)
    await store.release_agentcore_session_turn(user_id, project_id)
    await store.claim_agentcore_session_turn(user_id, project_id, ttl=ttl)

    # Read the new claim_version
    session2 = await store.get_agentcore_session(user_id, project_id)
    assert session2 is not None
    version_2 = session2["claim_version"]
    assert version_2 > version_1  # Version incremented

    # Clear with stale version_1 (from before reclaim) → False (fence tripped)
    cleared = await store.clear_stale_agentcore_session_claim(
        user_id, project_id, expected_version=version_1
    )
    assert cleared is False, "Clear with stale version should fail"

    # Session is still running (fence prevented the clear)
    session3 = await store.get_agentcore_session(user_id, project_id)
    assert session3 is not None
    assert session3["status"] == "running"

    # Clear with current version_2 → True
    cleared2 = await store.clear_stale_agentcore_session_claim(
        user_id, project_id, expected_version=version_2
    )
    assert cleared2 is True

    # Session is now idle
    session4 = await store.get_agentcore_session(user_id, project_id)
    assert session4 is not None
    assert session4["status"] == "idle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_list_sessions(store_fixture: str, request: Any) -> None:
    """list_agentcore_sessions returns a user's items and is scoped to that user.

    Proves the per-user list works and does not leak across users.
    """
    store = request.getfixturevalue(store_fixture)

    user_a = "user-a"
    user_b = "user-b"
    proj_x = "proj-x"
    proj_y = "proj-y"
    ttl = int(time.time()) + 3600

    # Create sessions for user A on two projects
    await store.claim_agentcore_session_turn(user_a, proj_x, ttl=ttl)
    await store.claim_agentcore_session_turn(user_a, proj_y, ttl=ttl)

    # Create session for user B on one project
    await store.claim_agentcore_session_turn(user_b, proj_x, ttl=ttl)

    # List user A's sessions → 2 items
    sessions_a = await store.list_agentcore_sessions(user_a)
    assert len(sessions_a) == 2

    # List user B's sessions → 1 item
    sessions_b = await store.list_agentcore_sessions(user_b)
    assert len(sessions_b) == 1

    # List user C (no sessions) → 0 items
    sessions_c = await store.list_agentcore_sessions("user-c")
    assert len(sessions_c) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_put_with_none_preserves_existing_ids(
    store_fixture: str, request: Any
) -> None:
    """Regression test: put_agentcore_session with None does not null out a good stored id.

    When a put/record call carries a None for runtime_session_id or app_session_id,
    the existing stored value must be preserved — recording a mapping never destroys
    a previously-good resume id. This is the integration-level behavior.
    """
    store = request.getfixturevalue(store_fixture)

    user_id = "user-a"
    project_id = "proj-x"
    runtime_id_initial = "runtime-initial"
    app_id_initial = "app-initial"
    ttl = int(time.time()) + 3600

    # Put initial mapping with both ids non-None
    await store.put_agentcore_session(
        user_id,
        project_id,
        runtime_session_id=runtime_id_initial,
        app_session_id=app_id_initial,
        ttl=ttl,
    )

    # Verify initial mapping
    session1 = await store.get_agentcore_session(user_id, project_id)
    assert session1 is not None
    assert session1["runtime_session_id"] == runtime_id_initial
    assert session1["app_session_id"] == app_id_initial

    # Put with app_session_id=None (should preserve the stored app_id_initial)
    await store.put_agentcore_session(
        user_id,
        project_id,
        runtime_session_id="runtime-updated",
        app_session_id=None,  # must not clobber the stored app_id_initial
        ttl=ttl,
    )

    # Verify app_session_id was NOT nulled out
    session2 = await store.get_agentcore_session(user_id, project_id)
    assert session2 is not None
    assert session2["runtime_session_id"] == "runtime-updated"
    assert session2["app_session_id"] == app_id_initial  # Preserved

    # Put with runtime_session_id=None (should preserve the stored runtime-updated)
    await store.put_agentcore_session(
        user_id,
        project_id,
        runtime_session_id=None,  # must not clobber the stored runtime-updated
        app_session_id="app-updated",
        ttl=ttl,
    )

    # Verify runtime_session_id was NOT nulled out
    session3 = await store.get_agentcore_session(user_id, project_id)
    assert session3 is not None
    assert session3["runtime_session_id"] == "runtime-updated"  # Preserved
    assert session3["app_session_id"] == "app-updated"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_expired_claim_is_claimable(store_fixture: str, request: Any) -> None:
    """An EXPIRED claim must be re-claimable in BOTH adapters.

    The claim's TTL was assumed to bound a leaked claim ("2h TTL backstop"). It
    does not: the conditional write accepted only ``status = idle OR
    attribute_not_exists(status)``, so an item still marked ``running`` blocked
    every later claim regardless of its ttl — and DynamoDB TTL deletion is a
    best-effort background process AWS documents as "typically" deleting expired
    items within a few DAYS, not at expiry. So the pair stayed wedged for days.

    Parametrized across cloud (real conditional write under moto) and dev so the
    two adapters cannot drift on claim semantics.
    """
    store = request.getfixturevalue(store_fixture)

    # A claim whose expiry has already passed, with the item still present.
    won = await store.claim_agentcore_session_turn(
        "userTTL", "projTTL", ttl=int(time.time()) - 60
    )
    assert won is True, "First claim must win"

    reclaim = await store.claim_agentcore_session_turn(
        "userTTL", "projTTL", ttl=int(time.time()) + 7200
    )
    assert reclaim is True, (
        "An expired claim must be claimable — otherwise the (user, project) pair "
        "is wedged until DynamoDB's best-effort TTL deletion physically removes "
        "the item, which AWS documents as taking up to days."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_fixture", ["cloud_store", "dev_store"], ids=["cloud", "dev"]
)
async def test_unexpired_claim_still_blocks(store_fixture: str, request: Any) -> None:
    """Counterpart to the expiry arm: a LIVE claim must still block.

    Guards against the expiry condition degrading the claim into a no-op — it is
    the primary one-turn-at-a-time serializer that keeps one writer per
    transcript.
    """
    store = request.getfixturevalue(store_fixture)

    assert await store.claim_agentcore_session_turn(
        "userLive", "projLive", ttl=int(time.time()) + 7200
    ) is True
    assert await store.claim_agentcore_session_turn(
        "userLive", "projLive", ttl=int(time.time()) + 7200
    ) is False, "A live claim must still block a concurrent claim"
