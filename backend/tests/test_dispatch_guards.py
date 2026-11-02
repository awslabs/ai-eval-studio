# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the dispatch guards (durable lock, caps, kill-switch, idempotency).

Behavioral tests against moto DDB with arch_split=True. They prove: concurrent-rejection, per-user cap, account
backstop, kill-switch halt, old-path fence unchanged, idempotency dedup returns
original job_id, different-key→lock-rejection, no-key→idempotency skipped,
terminal-release for completed/failed/cancelled, exactly-once, non-holder no-op.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from moto import mock_aws
import boto3

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.services.jobs import get_job_tracker


def _create_metadata_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema (inline helper from integration tests)."""
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


# ---------------------------------------------------------------------------
# Existence checks (keep these cheap grep-gates)
# ---------------------------------------------------------------------------


def test_per_user_cap_constant_is_3():
    """PER_USER_MAX_JOBS constant exists in dispatch.py and is 3."""
    from app.services.mcp_tools import dispatch as dispatch_module

    assert hasattr(dispatch_module, "PER_USER_MAX_JOBS"), "PER_USER_MAX_JOBS constant must exist"
    assert dispatch_module.PER_USER_MAX_JOBS == 3, "PER_USER_MAX_JOBS must be 3"


def test_account_backstop_constant_is_10():
    """ACCOUNT_MAX_JOBS constant exists in dispatch.py and is 10."""
    from app.services.mcp_tools import dispatch as dispatch_module

    assert hasattr(dispatch_module, "ACCOUNT_MAX_JOBS"), "ACCOUNT_MAX_JOBS constant must exist"
    assert dispatch_module.ACCOUNT_MAX_JOBS == 10, "ACCOUNT_MAX_JOBS must be 10"


@pytest.mark.asyncio
async def test_dispatch_eval_has_idempotency_key_arg():
    """dispatch_eval tool schema should include optional idempotency_key."""
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool

    schema = _dispatch_eval_tool.input_schema
    assert "idempotency_key" in schema["properties"], "idempotency_key must be in dispatch_eval schema"
    assert schema["properties"]["idempotency_key"]["type"] == "string"
    assert "idempotency_key" not in schema.get("required", []), "idempotency_key must be optional"


@pytest.mark.asyncio
async def test_coordinator_prompt_mentions_idempotency():
    """Coordinator prompt should instruct agent to use idempotency_key."""
    from pathlib import Path

    # Resolve relative to this file, never an absolute mount path: a hardcoded
    # checkout path passes only on the machine that has it and fails on every
    # other checkout.
    backend_dir = Path(__file__).parent.parent
    coord_prompt = (backend_dir / "app/prompts/coordinator.md").read_text()
    assert "idempotency_key" in coord_prompt, "coordinator.md must mention idempotency_key"


# ---------------------------------------------------------------------------
# Durable lock + caps + kill-switch behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_dispatch_rejected_new_path():
    """NEW PATH: Second concurrent dispatch for locked project is rejected."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup project dir mock
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            # Setup tracker mock
            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-1", project_id="proj-abc")
            mock_job_2 = MagicMock(job_id="job-2", project_id="proj-abc")
            mock_tracker_inst.create_job.side_effect = [mock_job_1, mock_job_2]
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="user-xyz")

            # Mock compute
            fake_compute = AsyncMock()
            mock_compute.return_value = fake_compute

            # First dispatch succeeds
            result1 = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            assert result1_json.get("job_id") == "job-1", "First dispatch should succeed"

            # Second dispatch fails (lock held)
            result2 = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            # Second dispatch returns mcp_error with eval_already_running
            result2_text = result2["content"][0]["text"]
            assert "eval_already_running" in result2_text or "already running" in result2_text.lower(), \
                f"Second dispatch should be rejected with eval_already_running, got: {result2_text}"

            # Verify only ONE job launched (first dispatch)
            assert fake_compute.launch.call_count == 1, "Only first dispatch should launch"


@pytest.mark.asyncio
async def test_per_user_cap_enforced():
    """Per-user cap (3 jobs) rejects 4th dispatch."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Seed 3 non-terminal jobs for owner-1
        await store.put_job({
            "job_id": "job-1",
            "project_id": "proj-1",
            "owner_id": "owner-1",
            "state": "running",
            "created_at": "2026-01-01T00:00:00Z",
        })
        await store.put_job({
            "job_id": "job-2",
            "project_id": "proj-2",
            "owner_id": "owner-1",
            "state": "running",
            "created_at": "2026-01-01T00:01:00Z",
        })
        await store.put_job({
            "job_id": "job-3",
            "project_id": "proj-3",
            "owner_id": "owner-1",
            "state": "queued",
            "created_at": "2026-01-01T00:02:00Z",
        })

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute"),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="owner-1")

            # 4th dispatch should fail with user_job_limit
            result = await dispatch_eval({
                "project_id": "proj-new",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            result_text = result["content"][0]["text"]
            assert "user_job_limit" in result_text or "per-user job limit" in result_text.lower(), \
                f"4th dispatch should be rejected with user_job_limit, got: {result_text}"


@pytest.mark.asyncio
async def test_account_backstop_enforced():
    """Account backstop (10 jobs) rejects next dispatch when limit reached."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Seed ACCOUNT_MAX_JOBS (10) non-terminal jobs across multiple owners
        for i in range(10):
            await store.put_job({
                "job_id": f"job-{i}",
                "project_id": f"proj-{i}",
                "owner_id": f"owner-{i % 5}",  # 5 different owners
                "state": "running",
                "created_at": "2026-01-01T00:00:00Z",
            })

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="new-user"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute"),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_tracker.return_value = mock_tracker_inst

            # New user with no jobs yet
            mock_req.return_value = MagicMock(user_id="new-user")

            # 11th dispatch should fail with account_job_limit
            result = await dispatch_eval({
                "project_id": "proj-new",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            result_text = result["content"][0]["text"]
            assert "account_job_limit" in result_text or "global job limit" in result_text.lower(), \
                f"11th dispatch should be rejected with account_job_limit, got: {result_text}"


@pytest.mark.asyncio
async def test_kill_switch_halts_dispatch():
    """Kill switch active → dispatch rejected before launch."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        fake_compute = AsyncMock()

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=True),  # Kill switch ON
            patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="user-xyz")

            # Dispatch should fail with cost_kill_switch_active
            result = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            result_text = result["content"][0]["text"]
            assert "cost_kill_switch_active" in result_text or "halted by cost control" in result_text.lower(), \
                f"Dispatch should be rejected with cost_kill_switch_active, got: {result_text}"

            # Verify compute.launch was NOT called
            fake_compute.launch.assert_not_called()


@pytest.mark.asyncio
async def test_old_path_fence_unchanged_claim_eval_lock_not_called():
    """OLD PATH (arch_split=False): uses in-memory fence, does NOT call claim_eval_lock."""
    # Use dev store (no moto needed for old path)
    from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
    dispatch_eval = _dispatch_eval_tool.handler

    with (
        patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
        patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
        patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
        patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
        patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
        patch("app.platform.arch_split.resolve_arch_split", return_value=False),  # OLD PATH
        patch("app.platform.job_compute.get_job_compute"),
        patch("app.services.mcp_tools.dispatch.asyncio.create_task"),
        patch("app.services.mcp_tools.dispatch.attach_error_sink"),
        patch("app.services.mcp_tools.dispatch._publish_project_event"),
        patch("app.platform.types.current_request") as mock_req,
        patch("app.platform.get_platform") as mock_platform,
    ):
        # Setup mocks
        mock_resolve.return_value = MagicMock()
        mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

        # Setup tracker with existing job (to trigger old-path fence)
        mock_tracker_inst = MagicMock()
        mock_existing_job = MagicMock(job_id="existing-job")
        mock_tracker_inst.get_active_job_by_type.return_value = mock_existing_job
        mock_tracker.return_value = mock_tracker_inst

        mock_req.return_value = MagicMock(user_id="user-xyz")

        # Setup store mock to verify claim_eval_lock NOT called
        mock_store = AsyncMock()
        mock_plat_inst = MagicMock()
        mock_plat_inst.metadata_store = mock_store
        mock_platform.return_value = mock_plat_inst

        # Dispatch should fail via old-path fence
        result = await dispatch_eval({
            "project_id": "proj-abc",
            "data_file": "test.csv",
            "prompt_version": "baseline",
            "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
        })

        result_text = result["content"][0]["text"]
        assert "already running" in result_text.lower(), \
            f"Old path should reject with in-memory fence message, got: {result_text}"

        # CRITICAL: verify claim_eval_lock was NOT called (old path doesn't use durable lock)
        mock_store.claim_eval_lock.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_deduplicates_and_returns_original_job_id():
    """Same idempotency_key within TTL returns original job_id with deduplicated=True."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        fake_compute = AsyncMock()

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute", return_value=fake_compute),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-1", project_id="proj-abc")
            mock_job_2 = MagicMock(job_id="job-2", project_id="proj-def")
            mock_tracker_inst.create_job.side_effect = [mock_job_1, mock_job_2]
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="user-xyz")

            # First dispatch with idempotency_key
            result1 = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
                "idempotency_key": "click-123",
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            assert result1_json.get("job_id") == "job-1", "First dispatch should succeed with job-1"
            assert result1_json.get("deduplicated") != True, "First dispatch should not be deduplicated"

            # Second dispatch with SAME idempotency_key on different project
            result2 = await dispatch_eval({
                "project_id": "proj-def",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
                "idempotency_key": "click-123",  # SAME KEY
            })

            result2_json = json.loads(result2["content"][0]["text"])
            # CRITICAL: Second dispatch should return ORIGINAL job_id (job-1, NOT job-2)
            assert result2_json.get("job_id") == "job-1", \
                f"Second dispatch with same key should return original job_id (job-1), got: {result2_json.get('job_id')}"
            assert result2_json.get("deduplicated") == True, "Second dispatch should be marked deduplicated"

            # Verify only ONE worker launched
            assert fake_compute.launch.call_count == 1, "Only first dispatch should launch worker"


@pytest.mark.asyncio
async def test_different_idempotency_keys_hit_lock():
    """Different idempotency keys on same project → second hits eval lock."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-1", project_id="proj-abc")
            mock_job_2 = MagicMock(job_id="job-2", project_id="proj-abc")
            mock_tracker_inst.create_job.side_effect = [mock_job_1, mock_job_2]
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="user-xyz")

            fake_compute = AsyncMock()
            mock_compute.return_value = fake_compute

            # First dispatch with key-1
            result1 = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
                "idempotency_key": "click-1",
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            assert result1_json.get("job_id") == "job-1", "First dispatch should succeed"

            # Second dispatch with DIFFERENT key on SAME project
            # Should hit eval lock (project already locked by job-1)
            result2 = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
                "idempotency_key": "click-2",  # DIFFERENT KEY
            })

            result2_text = result2["content"][0]["text"]
            assert "eval_already_running" in result2_text or "already running" in result2_text.lower(), \
                f"Second dispatch with different key should hit eval lock, got: {result2_text}"


@pytest.mark.asyncio
async def test_no_idempotency_key_skips_dedup():
    """Omitting idempotency_key skips the idempotency path entirely."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup mocks
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job = MagicMock(job_id="job-1", project_id="proj-abc")
            mock_tracker_inst.create_job.return_value = mock_job
            mock_tracker.return_value = mock_tracker_inst

            mock_req.return_value = MagicMock(user_id="user-xyz")

            fake_compute = AsyncMock()
            mock_compute.return_value = fake_compute

            # Dispatch WITHOUT idempotency_key
            result = await dispatch_eval({
                "project_id": "proj-abc",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
                # NO idempotency_key
            })

            import json
            result_json = json.loads(result["content"][0]["text"])
            assert result_json.get("job_id") == "job-1", "Dispatch should succeed"

            # Verify no idempotency key was created (path skipped)
            # Scan for any idempotency keys in the table
            response = dynamodb_client.scan(
                TableName=table_name,
                FilterExpression="begins_with(SK, :sk_prefix)",
                ExpressionAttributeValues={":sk_prefix": {"S": "IDEMPOTENCY#"}},
            )
            items = response.get("Items", [])
            assert len(items) == 0, "No idempotency key should exist when key omitted"


# ---------------------------------------------------------------------------
# Terminal release behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_release_frees_lock():
    """Terminal transition (completed/failed/cancelled) releases eval lock."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Test all three terminal states
        for terminal_state in ["completed", "failed", "cancelled"]:
            # Setup
            owner_id = "owner-1"
            project_id = f"proj-{terminal_state}"

            # Create an in-memory job first (JobTracker generates its own UUIDs)
            tracker = get_job_tracker()
            tracker._jobs.clear()  # Clear any prior jobs

            job = tracker.create_job(
                project_id=project_id,
                job_type="eval",
                config={
                    "user_id": owner_id,
                    "arch_split": True,
                },
            )
            job_id = job.job_id  # Use the generated job_id

            # Now claim lock with the real job_id
            claimed = await store.claim_eval_lock(owner_id, project_id, job_id)
            assert claimed, f"Lock should be claimed for {terminal_state} test"

            # Wire the platform to use the real moto store
            from app.platform import get_platform
            with patch("app.platform.get_platform") as mock_platform:
                mock_plat_inst = MagicMock()
                mock_plat_inst.metadata_store = store
                mock_platform.return_value = mock_plat_inst

                # Transition to terminal state
                tracker.update_state(job_id, terminal_state)

                # Drain pending writes (release is fire-and-forget)
                await tracker.drain_pending_writes()

            # Verify lock was released — a fresh claim should succeed
            new_job_id = f"job-new-{terminal_state}"
            claimed_after = await store.claim_eval_lock(owner_id, project_id, new_job_id)
            assert claimed_after, \
                f"Lock should be released after {terminal_state} transition, but new claim failed"


@pytest.mark.asyncio
async def test_terminal_release_exactly_once():
    """Double terminal transition → lock released exactly once."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "proj-once"

        # Create job first
        tracker = get_job_tracker()
        tracker._jobs.clear()

        job = tracker.create_job(
            project_id=project_id,
            job_type="eval",
            config={
                "user_id": owner_id,
                "arch_split": True,
            },
        )
        job_id = job.job_id  # Use generated job_id

        # Claim lock with the real job_id
        claimed = await store.claim_eval_lock(owner_id, project_id, job_id)
        assert claimed

        # Wire platform
        from app.platform import get_platform
        with patch("app.platform.get_platform") as mock_platform:
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # First terminal transition
            tracker.update_state(job_id, "failed")
            await tracker.drain_pending_writes()

            # Verify lock released
            lock = await store.get_eval_lock(owner_id, project_id)
            assert lock is None, "Lock should be released after first terminal"

            # Claim lock again with a different job
            new_job_id = "job-new"
            claimed_new = await store.claim_eval_lock(owner_id, project_id, new_job_id)
            assert claimed_new, "Should be able to claim after release"

            # Second terminal transition on SAME job (was_terminal guard prevents double-release)
            tracker.update_state(job_id, "cancelled")
            await tracker.drain_pending_writes()

            # Verify the NEW job's lock is still held (double-release didn't free it)
            lock_after = await store.get_eval_lock(owner_id, project_id)
            assert lock_after is not None, "New job's lock should still be held"
            assert lock_after["lock_holder_job_id"] == new_job_id, \
                "Lock holder should still be the new job (double-release was a no-op)"


@pytest.mark.asyncio
async def test_non_holder_terminal_is_noop():
    """A job that never held the lock transitioning terminal is a no-op."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "owner-1"
        project_id = "proj-noop"

        # Create tracker and jobs
        tracker = get_job_tracker()
        tracker._jobs.clear()

        # Create holder job first
        holder_job = tracker.create_job(
            project_id=project_id,
            job_type="eval",
            config={
                "user_id": owner_id,
                "arch_split": True,
            },
        )
        holder_job_id = holder_job.job_id  # Use generated job_id

        # Claim lock with holder job
        claimed = await store.claim_eval_lock(owner_id, project_id, holder_job_id)
        assert claimed

        # Create a NON-HOLDER job (e.g., optimization-iteration job)
        non_holder_job = tracker.create_job(
            project_id=project_id,
            job_type="optimization_iteration",  # Different type
            config={
                "user_id": owner_id,
                "arch_split": True,
            },
        )
        non_holder_job_id = non_holder_job.job_id  # Use generated job_id

        # Wire platform
        from app.platform import get_platform
        with patch("app.platform.get_platform") as mock_platform:
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Transition non-holder to terminal
            tracker.update_state(non_holder_job_id, "completed")
            await tracker.drain_pending_writes()

        # Verify HOLDER job's lock is still held (non-holder release was a no-op)
        lock = await store.get_eval_lock(owner_id, project_id)
        assert lock is not None, "Lock should still exist"
        assert lock["lock_holder_job_id"] == holder_job_id, \
            "Lock should still be held by the original holder (non-holder release was a no-op)"


# ---------------------------------------------------------------------------
# Post-claim exception lock leak fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_claim_exception_releases_lock():
    """Post-claim exception fails job and releases lock."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        owner_id = "user-xyz"
        project_id = "proj-abc"

        # Use real tracker to ensure terminal-release fires
        tracker = get_job_tracker()
        tracker._jobs.clear()

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors") as mock_read_descriptors,
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup project dir mock
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_req.return_value = MagicMock(user_id=owner_id)

            # Mock compute
            fake_compute = AsyncMock()
            mock_compute.return_value = fake_compute

            # Force read_descriptors to raise AFTER lock claim
            mock_read_descriptors.side_effect = RuntimeError("Descriptor read failed")

            # Dispatch should fail with mcp_error
            result = await dispatch_eval({
                "project_id": project_id,
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            # Should return mcp_error (text content with error message)
            result_text = result["content"][0]["text"]
            assert "Dispatch error" in result_text or "Descriptor read failed" in result_text, \
                f"Error message should mention the exception: {result_text}"

            # CRITICAL: Verify job was created and failed (terminal state)
            jobs = list(tracker._jobs.values())
            assert len(jobs) > 0, "Job should have been created"
            job = jobs[0]
            assert job.state == "failed", f"Job should be in 'failed' state, got: {job.state}"

            # Drain pending writes to ensure lock release completes
            # (terminal-release is fire-and-forget via asyncio.create_task)
            await tracker.drain_pending_writes()

            # CRITICAL: Verify lock was released (a fresh claim should succeed)
            new_claim = await store.claim_eval_lock(owner_id, project_id, "job-new")
            assert new_claim is True, \
                "Lock should be released after post-claim exception (new claim should succeed)"


# ---------------------------------------------------------------------------
# Eval lock keyed on resolved project owner fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_keys_the_lock_on_the_owner_not_the_request():
    """Dispatch keys the lock on the project owner, whatever user the request
    context names, so every entry point contends for the same lock."""
    with mock_aws():
        # Setup moto DDB
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        project_id = "proj-1"
        owner_id = "owner-alice"
        other_user_id = "other-bob"

        # Use real tracker to track jobs
        tracker = get_job_tracker()
        tracker._jobs.clear()

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project") as mock_resolve_owner,
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            # Wire real moto store
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            # Setup project dir mock
            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            # Mock compute
            fake_compute = AsyncMock()
            mock_compute.return_value = fake_compute

            # Project owned by alice
            mock_resolve_owner.return_value = owner_id

            # FIRST dispatch: the request context names another user
            mock_req.return_value = MagicMock(user_id=other_user_id)

            result1 = await dispatch_eval({
                "project_id": project_id,
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            job_id_1 = result1_json.get("job_id")
            assert job_id_1, "First dispatch should succeed and return job_id"

            # CRITICAL: Verify lock was claimed under OWNER's partition (not the requester's)
            lock_via_owner = await store.get_eval_lock(owner_id, project_id)
            assert lock_via_owner is not None, \
                "Lock should be claimable via OWNER's partition (not requester's)"

            # SECOND dispatch: requester is owner alice
            # Should hit the SAME lock
            mock_req.return_value = MagicMock(user_id=owner_id)

            result2 = await dispatch_eval({
                "project_id": project_id,
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["anthropic.claude-3-sonnet-20240229-v1:0"],
            })

            result2_text = result2["content"][0]["text"]
            assert "eval_already_running" in result2_text or "already running" in result2_text.lower(), \
                f"Second dispatch (by owner) should hit the first lock, got: {result2_text}"


# ============================================================================
# Idempotency Key Release Tests
# ============================================================================


@pytest.mark.asyncio
async def test_idempotency_claim_then_launch_failure_releases_key():
    """claim → launch failure → key RELEASED (assert moto store item GONE)."""
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-fail-1", project_id="proj-1")
            mock_tracker_inst.create_job.side_effect = [mock_job_1]
            mock_tracker.return_value = mock_tracker_inst
            mock_req.return_value = MagicMock(user_id="owner-1")

            # Make compute.launch RAISE to trigger post-claim exception
            mock_compute_inst = MagicMock()
            mock_compute_inst.launch = MagicMock(side_effect=RuntimeError("Launch failed"))
            mock_compute.return_value = mock_compute_inst

            result = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "test-key-1",
            })

            # Dispatch should fail
            assert result.get("is_error") is True

            # Assert the idempotency key item is GONE from the store
            claimed, existing_job_id = await store.claim_idempotency_key(
                "owner-1", "test-key-1", "new-job-id", ttl_seconds=300
            )
            assert claimed is True, "Key should be released (fresh claim succeeds)"


@pytest.mark.asyncio
async def test_idempotency_release_then_retry_succeeds():
    """After failure, retry with SAME key actually launches (not deduplicated)."""
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Use REAL tracker so terminal release fires
        tracker = get_job_tracker()
        tracker._jobs.clear()

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)
            mock_req.return_value = MagicMock(user_id="owner-1")

            # First dispatch: make launch RAISE
            mock_compute_inst = AsyncMock()
            launch_call_count = [0]

            async def launch_side_effect(*args, **kwargs):
                launch_call_count[0] += 1
                if launch_call_count[0] == 1:
                    raise RuntimeError("First launch failed")
                # Second launch succeeds
                return MagicMock(kind="fargate_task", value="arn:task-2")

            mock_compute_inst.launch.side_effect = launch_side_effect
            mock_compute.return_value = mock_compute_inst

            # First dispatch with key — should fail
            result1 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "retry-key",
            })
            assert result1.get("is_error") is True

            # Drain pending writes to ensure lock release completes
            await tracker.drain_pending_writes()

            # Second dispatch with SAME key — should actually launch (not dedup)
            result2 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "retry-key",
            })

            # Assert second dispatch succeeded with DIFFERENT job_id (retry launched new job)
            assert result2.get("is_error") is not True
            import json
            result2_json = json.loads(result2["content"][0]["text"])
            assert "job_id" in result2_json, "Retry should return job_id"
            assert result2_json.get("deduplicated") != True, "Retry should NOT be deduplicated"
            assert launch_call_count[0] == 2, "Launch should be called twice (first failed, second succeeded)"


@pytest.mark.asyncio
async def test_idempotency_dedup_still_works_and_key_survives():
    """Dedup path still dedupes AND the key item still EXISTS after it."""
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.services.job_progress_republisher.maybe_start_poller"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-dedup-1", project_id="proj-1")
            mock_job_2 = MagicMock(job_id="job-dedup-2", project_id="proj-1")
            mock_tracker_inst.create_job.side_effect = [mock_job_1, mock_job_2]
            mock_tracker.return_value = mock_tracker_inst
            mock_req.return_value = MagicMock(user_id="owner-1")

            mock_compute_inst = AsyncMock()
            mock_compute_inst.launch.return_value = MagicMock(kind="fargate_task", value="arn:task-1")
            mock_compute.return_value = mock_compute_inst

            # First dispatch with key — succeeds
            result1 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "dedup-key",
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            job_id_1 = result1_json.get("job_id")
            assert job_id_1 == "job-dedup-1", "First dispatch should succeed with job-dedup-1"

            # Second dispatch with SAME key — should dedup
            result2 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "dedup-key",
            })

            result2_json = json.loads(result2["content"][0]["text"])
            assert result2_json.get("deduplicated") is True
            assert result2_json.get("job_id") == "job-dedup-1", "Dedup should return ORIGINAL job_id"

            # Assert the key item still EXISTS in the store (dedup path does NOT release)
            claimed, existing_job_id = await store.claim_idempotency_key(
                "owner-1", "dedup-key", "new-job-id", ttl_seconds=300
            )
            assert claimed is False, "Key should still be claimed (not released)"
            assert existing_job_id == job_id_1


@pytest.mark.asyncio
async def test_idempotency_ownership_guard_mismatched_job_id():
    """Ownership guard — mismatched job_id returns False and item survives (DynamoDB adapter)."""
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Claim a key with job_id_1
        claimed, _ = await store.claim_idempotency_key("owner-1", "owned-key", "job-id-1", ttl_seconds=300)
        assert claimed is True

        # Try to release with DIFFERENT job_id_2 — should return False
        released = await store.release_idempotency_key("owner-1", "owned-key", "job-id-2")
        assert released is False, "Ownership guard should prevent release (mismatched job_id)"

        # Assert the item still exists
        claimed_again, existing_job_id = await store.claim_idempotency_key(
            "owner-1", "owned-key", "job-id-3", ttl_seconds=300
        )
        assert claimed_again is False, "Key should still be claimed"
        assert existing_job_id == "job-id-1"


@pytest.mark.asyncio
async def test_idempotency_lock_claim_failure_releases_key():
    """Lock-claim-failure path releases the idempotency key."""
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Pre-claim the eval lock for proj-1 (so second dispatch hits lock-claim-failure)
        await store.claim_eval_lock("owner-1", "proj-1", "existing-job-id")

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute"),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_fail = MagicMock(job_id="job-lock-fail", project_id="proj-1")
            mock_tracker_inst.create_job.side_effect = [mock_job_fail]
            mock_tracker.return_value = mock_tracker_inst
            mock_req.return_value = MagicMock(user_id="owner-1")

            # Dispatch with idempotency key — should fail on lock-claim-failure
            result = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "lock-fail-key",
            })

            # Dispatch should fail with eval_already_running
            result_text = result["content"][0]["text"]
            assert "eval_already_running" in result_text, f"Should fail with eval_already_running, got: {result_text}"

            # Assert the idempotency key was RELEASED (fresh claim succeeds)
            claimed, _ = await store.claim_idempotency_key(
                "owner-1", "lock-fail-key", "new-job-id", ttl_seconds=300
            )
            assert claimed is True, "Key should be released on lock-claim-failure"


@pytest.mark.asyncio
async def test_idempotency_pre_launch_raise_releases_key():
    """PRE-LAUNCH raise releases (proves except release is generic, not launch-specific).
    
    Patch read_descriptors to raise inside the post-claim try and assert the key item
    is gone from the store. Without this test, an implementer could wire the release
    next to compute.launch only and every other test would still pass while
    read_descriptors/_preflight_models_invocable/update_job_state/maybe_start_poller
    failures silently keep stranding the key.
    """
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors") as mock_read_desc,
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute"),
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_pre = MagicMock(job_id="job-pre-launch", project_id="proj-1")
            mock_tracker_inst.create_job.side_effect = [mock_job_pre]
            mock_tracker.return_value = mock_tracker_inst
            mock_req.return_value = MagicMock(user_id="owner-1")

            # Make read_descriptors RAISE (before compute.launch is ever called)
            mock_read_desc.side_effect = RuntimeError("Descriptor read failed")

            result = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "pre-launch-key",
            })

            # Dispatch should fail
            assert result.get("is_error") is True

            # Assert the idempotency key was RELEASED (fresh claim succeeds)
            claimed, _ = await store.claim_idempotency_key(
                "owner-1", "pre-launch-key", "new-job-id", ttl_seconds=300
            )
            assert claimed is True, "Key should be released on pre-launch exception"


@pytest.mark.asyncio
async def test_idempotency_deduped_dispatch_never_releases():
    """A deduped dispatch never releases — catches guarding on reassigned `claimed`.
    
    Assert that a dispatch which DEDUPED (never claimed) does not release, even when
    it subsequently fails. This is the test that catches guarding on the reassigned
    `claimed` (the lock's boolean) instead of `idempotency_claimed_by_us`.
    """
    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        from app.services.mcp_tools.dispatch import dispatch_eval as _dispatch_eval_tool
        dispatch_eval = _dispatch_eval_tool.handler

        with (
            patch("app.services.mcp_tools.dispatch.resolve_project_dir") as mock_resolve,
            patch("app.services.mcp_tools.dispatch.get_job_tracker") as mock_tracker,
            patch("app.services.mcp_tools.dispatch.OptimizationState.load", return_value=None),
            patch("app.services.mcp_tools.dispatch.read_descriptors", return_value={}),
            patch("app.services.mcp_tools.dispatch._preflight_models_invocable", return_value=[]),
            patch("app.services.agent_turn._resolve_owner_id_for_project", return_value="owner-1"),
            patch("app.platform.arch_split.resolve_arch_split", return_value=True),
            patch("app.platform.arch_split.is_cost_kill_switch_active", return_value=False),
            patch("app.platform.job_compute.get_job_compute") as mock_compute,
            patch("app.services.mcp_tools.dispatch._publish_project_event"),
            patch("app.services.job_progress_republisher.maybe_start_poller"),
            patch("app.platform.types.current_request") as mock_req,
            patch("app.platform.get_platform") as mock_platform,
        ):
            mock_plat_inst = MagicMock()
            mock_plat_inst.metadata_store = store
            mock_platform.return_value = mock_plat_inst

            mock_resolve.return_value = MagicMock()
            mock_resolve.return_value.__truediv__ = lambda s, o: MagicMock(exists=lambda: True)

            mock_tracker_inst = MagicMock()
            mock_job_1 = MagicMock(job_id="job-dedup-never-1", project_id="proj-1")
            mock_job_2 = MagicMock(job_id="job-dedup-never-2", project_id="proj-1")
            mock_tracker_inst.create_job.side_effect = [mock_job_1, mock_job_2]
            mock_tracker.return_value = mock_tracker_inst
            mock_req.return_value = MagicMock(user_id="owner-1")

            mock_compute_inst = AsyncMock()
            mock_compute_inst.launch.return_value = MagicMock(kind="fargate_task", value="arn:task-1")
            mock_compute.return_value = mock_compute_inst

            # First dispatch — succeeds and claims the key
            result1 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "dedup-never-release",
            })

            import json
            result1_json = json.loads(result1["content"][0]["text"])
            job_id_1 = result1_json.get("job_id")
            assert job_id_1 == "job-dedup-never-1", "First dispatch should succeed"

            # Second dispatch — DEDUPES (never claims the key)
            result2 = await dispatch_eval({
                "project_id": "proj-1",
                "data_file": "test.csv",
                "prompt_version": "baseline",
                "target_model_ids": ["test-model"],
                "idempotency_key": "dedup-never-release",
            })

            result2_json = json.loads(result2["content"][0]["text"])
            assert result2_json.get("deduplicated") is True
            assert result2_json.get("job_id") == "job-dedup-never-1", "Dedup should return ORIGINAL job_id"

            # Assert the key item STILL EXISTS (dedup dispatch did NOT release it)
            claimed, existing_job_id = await store.claim_idempotency_key(
                "owner-1", "dedup-never-release", "new-job-id", ttl_seconds=300
            )
            assert claimed is False, "Dedup dispatch should NOT release the key"
            assert existing_job_id == "job-dedup-never-1"
