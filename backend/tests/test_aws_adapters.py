# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AWS-backed platform adapters.

These tests use ``botocore.stub.Stubber`` and lightweight fakes — no live
AWS calls are made. Real-AWS integration tests are gated by the
``integration`` marker (skipped per ``addopts = "-m 'not integration'"``
in ``pyproject.toml``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import boto3
import pytest
from botocore.stub import ANY, Stubber

from app.platform.aws.dynamodb_metadata_store import (
    DynamoDBMetadataStore,
    _from_dynamo,
    _to_dynamo,
)
from app.platform.aws.sts_token_vending import STSTokenVending
from app.platform.metadata_store import ConcurrentUpdateError


# ---------------------------------------------------------------------------
# DynamoDBMetadataStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_optimization_state_first_write_uses_attribute_not_exists():
    """First write (no expected_version) sets attribute_not_exists guard."""
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    stubber.add_response(
        "put_item",
        {},
        {
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(version)",
        },
    )
    stubber.activate()
    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)
        version = await store.put_optimization_state(
            "u-1", "p-1", {"current_iteration": 0}, expected_version=None
        )
        assert version == 1
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


@pytest.mark.asyncio
async def test_put_optimization_state_uses_versioned_condition_and_raises_on_conflict():
    """expected_version mismatch must raise ConcurrentUpdateError."""
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    stubber.add_client_error(
        "put_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="version mismatch",
        expected_params={
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": (
                "attribute_not_exists(version) OR version = :expected"
            ),
            "ExpressionAttributeValues": {":expected": {"N": "3"}},
        },
    )
    stubber.activate()
    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)
        with pytest.raises(ConcurrentUpdateError):
            await store.put_optimization_state(
                "u-1", "p-1", {"current_iteration": 4}, expected_version=3
            )
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


@pytest.mark.asyncio
async def test_add_bedrock_cost_uses_atomic_add_expression():
    """The cost rollup must use ADD, not SET, so concurrent writers don't clobber."""
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    stubber.add_response(
        "update_item",
        {},
        {
            "TableName": "ai-eval-studio-metadata",
            "Key": {
                "PK": {"S": "USER#u-1"},
                "SK": {"S": "COST#2026-05"},
            },
            "UpdateExpression": (
                "ADD bedrock_input_tokens :it, "
                "bedrock_output_tokens :ot, "
                "total_cost_usd :c"
            ),
            "ExpressionAttributeValues": {
                ":it": {"N": "100"},
                ":ot": {"N": "50"},
                ":c": {"N": "0.123"},
            },
        },
    )
    stubber.activate()
    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)
        await store.add_bedrock_cost(
            "u-1",
            year_month="2026-05",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.123,
        )
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


def test_to_dynamo_accepts_decimal_and_round_trips():
    """
    _to_dynamo must accept Decimal (which _from_dynamo returns for fractional numbers)
    so DDB read-modify-write round-trips without a type crash.
    """
    from decimal import Decimal

    # Round-trip: _from_dynamo returns Decimal, _to_dynamo must accept it
    ddb_number = {"N": "0.95"}
    parsed_value = _from_dynamo(ddb_number)
    assert isinstance(parsed_value, Decimal), "_from_dynamo should return Decimal for fractional numbers"
    assert parsed_value == Decimal("0.95")

    # Direct: _to_dynamo(Decimal) should produce valid DDB format
    serialized = _to_dynamo(Decimal("0.95"))
    assert serialized == {"N": "0.95"}, f"Expected {{'N': '0.95'}}, got {serialized}"

    # Full round-trip: _to_dynamo(_from_dynamo(x)) == x
    round_tripped = _to_dynamo(parsed_value)
    assert round_tripped == ddb_number, f"Round-trip failed: {round_tripped} != {ddb_number}"


@pytest.mark.asyncio
async def test_put_optimization_state_accepts_decimal_bearing_state():
    """
    put_optimization_state with Decimal-bearing state dict (as DDB read path surfaces)
    must serialize without raising 'Unsupported DynamoDB value type: Decimal'.

    This mirrors the live RMW symptom: get_optimization_state → _from_dynamo yields Decimal
    for fractional fields → optimization loop mutates state → put_optimization_state → _to_dynamo
    must accept the Decimal and write it back.
    """
    from decimal import Decimal

    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)

    # Accept any put_item with attribute_not_exists (first-write shape)
    stubber.add_response(
        "put_item",
        {},
        {
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(version)",
        },
    )
    stubber.activate()

    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)

        # Build an optimization state dict containing Decimal fractional values
        # as a DDB read would surface them
        state = {
            "target_metric_scores": {"Correctness": Decimal("0.95")},
            "iterations": [
                {
                    "per_model_pass_rates": {"model-a": Decimal("0.9")},
                    "pass_rate": Decimal("0.88"),
                }
            ],
            "current_iteration": 1,
            "status": "running",
        }

        # _to_dynamo must accept the Decimal fractions a DynamoDB read returns;
        # if it cannot, it raises TypeError before the stub is reached. Accepted,
        # the write returns version 1.
        version = await store.put_optimization_state(
            "u-1", "p-1", state, expected_version=None
        )
        assert version == 1
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


# ---------------------------------------------------------------------------
# STSTokenVending
# ---------------------------------------------------------------------------


def test_sts_token_vending_builds_scoped_session_tags():
    """build_session_tags() returns uid+proj (coordinator) or uid+job+proj (worker).

    The worker carries ``proj`` so the scoped-session role's
    ``PROJECT_SEQ#${aws:PrincipalTag/proj}`` LeadingKeys template resolves at
    authorization time; without it the durable seq allocation is denied.
    """
    sts = boto3.client("sts", region_name="us-east-1")
    tvm = STSTokenVending(
        sts_client=sts,
        role_arn="arn:aws:iam::123:role/ai-eval-studio-scoped-session-role",
        sessions_bucket="my-bucket",
        snapshots_bucket="my-snapshots",
        table_name="ai-eval-studio-metadata",
    )

    # Coordinator path (job_id=None): uid+proj
    coord_tags = tvm.build_session_tags("user-abc", "proj-xyz", job_id=None)
    coord_dict = {t["Key"]: t["Value"] for t in coord_tags}
    assert coord_dict == {"uid": "user-abc", "proj": "proj-xyz"}, \
        f"Coordinator tags should be {{uid, proj}}, got {coord_dict}"

    # Worker path (job_id set): uid+job+proj
    worker_tags = tvm.build_session_tags("user-abc", "proj-xyz", job_id="job-777")
    worker_dict = {t["Key"]: t["Value"] for t in worker_tags}
    assert worker_dict == {
        "uid": "user-abc",
        "job": "job-777",
        "proj": "proj-xyz",
    }, f"Worker tags should be {{uid, job, proj}}, got {worker_dict}"


@pytest.mark.asyncio
async def test_sts_mint_invokes_assume_role_with_tags():
    """mint() must call STS with Tags (not Policy)."""
    sts = boto3.client("sts", region_name="us-east-1")
    stubber = Stubber(sts)
    expiration = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stubber.add_response(
        "assume_role",
        {
            "Credentials": {
                "AccessKeyId": "AKIATESTACCESS123",
                "SecretAccessKey": "secretsecretsecretsecret",
                "SessionToken": "sessiontoken",
                "Expiration": expiration,
            },
            "AssumedRoleUser": {
                "AssumedRoleId": "AROA:fake",
                "Arn": "arn:aws:sts::123:assumed-role/x/y",
            },
            "PackedPolicySize": 31,
        },
        expected_params={
            "RoleArn": "arn:aws:iam::123:role/ai-eval-studio-scoped-session-role",
            "RoleSessionName": ANY,
            "DurationSeconds": 1800,
            "Tags": ANY,  # NOT Policy
        },
    )
    stubber.activate()
    try:
        tvm = STSTokenVending(
            sts_client=sts,
            role_arn="arn:aws:iam::123:role/ai-eval-studio-scoped-session-role",
            sessions_bucket="my-bucket",
            snapshots_bucket="my-snapshots",
        )
        creds = await tvm.mint(
            "user-abc",
            "proj-xyz",
            "/mnt/projects/users/user-abc/projects/proj-xyz",
            ttl_sec=1800,
        )
        assert creds.access_key_id == "AKIATESTACCESS123"
        assert creds.session_token == "sessiontoken"
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()




# ---------------------------------------------------------------------------
# Project event outbox — the write must not overwrite another writer's event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_outbox_event_is_conditional_on_the_key_being_free():
    """A plain put_item destroyed the earlier event on a key collision.

    Silently: no error, no log, and the outbox is the durable queue that exists
    so microVM events are LATE rather than LOST. The condition is what turns a
    collision into something recoverable.
    """
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
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
        store = DynamoDBMetadataStore(ddb_client=ddb)
        await store.put_outbox_event("u-1", "p-1", {"resource": "eval_run"})
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()


@pytest.mark.asyncio
async def test_put_outbox_event_retries_with_a_new_key_after_a_collision():
    """Cross-process collisions are real: two coordinator tasks overlap during a
    rolling deploy by design, and a microVM can stash while the coordinator does.

    The retry has to move to a DIFFERENT key — retrying the taken one would fail
    forever — so this asserts the second attempt carries a strictly greater sort
    key, not merely that a second call happened.
    """
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    stubber.add_client_error(
        "put_item",
        service_error_code="ConditionalCheckFailedException",
        service_message="key taken",
        expected_params={
            "TableName": "ai-eval-studio-metadata",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(SK)",
        },
    )
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
        store = DynamoDBMetadataStore(ddb_client=ddb)
        seen: list[str] = []
        original = ddb.put_item

        def _capture(**kwargs):
            seen.append(kwargs["Item"]["SK"]["S"])
            return original(**kwargs)

        ddb.put_item = _capture  # type: ignore[method-assign]
        await store.put_outbox_event("u-1", "p-1", {"resource": "eval_run"})
        stubber.assert_no_pending_responses()
        assert len(seen) == 2, "the collision must be retried"
        assert seen[1] > seen[0], "the retry must mint a NEW, later key"
    finally:
        stubber.deactivate()


@pytest.mark.asyncio
async def test_put_outbox_event_raises_when_every_key_is_taken():
    """Give up loudly rather than silently: the caller logs a lost event.

    Losing one visibly is strictly better than the old behaviour, which deleted
    someone else's event and reported success.
    """
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    stubber = Stubber(ddb)
    for _ in range(3):
        stubber.add_client_error(
            "put_item",
            service_error_code="ConditionalCheckFailedException",
            service_message="key taken",
            expected_params={
                "TableName": "ai-eval-studio-metadata",
                "Item": ANY,
                "ConditionExpression": "attribute_not_exists(SK)",
            },
        )
    stubber.activate()
    try:
        store = DynamoDBMetadataStore(ddb_client=ddb)
        with pytest.raises(RuntimeError, match="already taken"):
            await store.put_outbox_event("u-1", "p-1", {"resource": "eval_run"})
        stubber.assert_no_pending_responses()
    finally:
        stubber.deactivate()
