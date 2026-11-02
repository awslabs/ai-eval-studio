# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable event log."""
import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from app.services.jobs import Job

@pytest.mark.asyncio
async def test_append_event_calls_allocate_and_append_on_new_path():
    """On new path, Job.append_event calls allocate_job_seq + append_job_event."""
    mock_store = AsyncMock()
    mock_store.allocate_job_seq.return_value = 1
    
    job = Job(
        job_id="test_job", project_id="test_proj", type="eval",
        config={"arch_split": True}, state="running", progress={},
        created_at="2026-07-02T12:00:00Z"
    )
    
    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        with patch("app.services.ws_hub.get_ws_hub") as mock_hub:
            mock_hub.return_value.publish_to_user = AsyncMock()
            with patch("app.main.get_project_manager") as mock_pm:
                mock_project = type("Project", (), {"owner_id": "test_owner"})()
                mock_pm.return_value.get_project.return_value = mock_project
                
                await job.append_event({"event": "test"})
                await asyncio.sleep(0.2)
    
    mock_store.allocate_job_seq.assert_called_once()
    mock_store.append_job_event.assert_called_once()

@pytest.mark.asyncio
async def test_append_event_no_durable_append_on_old_path():
    """On old path, no durable append fires."""
    mock_store = AsyncMock()

    job = Job(
        job_id="test_job", project_id="test_proj", type="eval",
        config={}, state="running", progress={},
        created_at="2026-07-02T12:00:00Z"
    )

    with patch("app.platform.get_platform") as mock_platform:
        mock_platform.return_value.metadata_store = mock_store
        with patch("app.services.ws_hub.get_ws_hub") as mock_hub:
            mock_hub.return_value.publish_to_user = AsyncMock()
            with patch("app.main.get_project_manager") as mock_pm:
                mock_project = type("Project", (), {"owner_id": "test_owner"})()
                mock_pm.return_value.get_project.return_value = mock_project

                await job.append_event({"event": "test"})
                await asyncio.sleep(0.2)

    mock_store.allocate_job_seq.assert_not_called()
    mock_store.append_job_event.assert_not_called()


@pytest.mark.asyncio
async def test_events_since_skips_seq_counter_item():
    """events_since returns ONLY EVT# event items, excluding the SEQ counter.

    The JOB#{job_id} partition co-locates the atomic-seq counter item (SK="SEQ")
    with event items (SK="EVT#..."). Because "SEQ" > "EVT#<any-20-digit>"
    lexicographically, the SEQ item is returned by SK > EVT#{since_seq} queries.

    This test seeds BOTH item shapes (the SEQ counter + real events) and proves
    events_since excludes the SEQ item before decoding item["event"], so no
    KeyError is raised.
    """
    from moto import mock_aws
    import time
    import boto3
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

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

    with mock_aws():
        # Create table
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)

        # Create store instance
        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        job_id = "test-job-seq-item"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Seed the partition with BOTH item shapes:
        # 1. Create the SEQ counter item (via allocate_job_seq)
        await store.allocate_job_seq(job_id)
        await store.allocate_job_seq(job_id)  # Increment to 2

        # 2. Append 3 real events
        for seq in range(1, 4):
            durable_event = {
                "stream": f"job:{job_id}",
                "seq": seq,
                "kind": "progress",
                "data": {"event": "progress", "case": seq, "total": 3}
            }
            await store.append_job_event(job_id, seq, durable_event, ttl=ttl)

        # Query from seq 0 — this MUST return only the 3 events (SEQ item excluded).
        # Decoding the SEQ item as an event would raise KeyError: 'event'.
        events = await store.events_since(job_id, 0)

        # Assert: 3 events returned (not 4 — SEQ excluded)
        assert len(events) == 3, (
            f"Expected 3 events, got {len(events)}. "
            f"SEQ counter item must be excluded from events_since result."
        )

        # Assert: events have correct seq order (ascending 1, 2, 3)
        for idx, event in enumerate(events, start=1):
            assert event["seq"] == idx, (
                f"Event {idx} has seq={event['seq']}, expected {idx}"
            )
            assert event["stream"] == f"job:{job_id}"
            assert event["kind"] == "progress"


@pytest.mark.asyncio
async def test_turn_events_since_suffix_semantics():
    """Turn events_since returns exactly the ordered suffix (seq > since_seq)."""
    from moto import mock_aws
    import time
    import boto3
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    def _create_metadata_table(ddb_client, table_name: str) -> None:
        """Create the single-table schema."""
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

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)

        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "test-turn-123"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Append 5 turn events
        for seq in range(1, 6):
            durable_event = {
                "stream": f"turn:{turn_id}",
                "seq": seq,
                "kind": "text" if seq < 5 else "result",
                "data": {"content": f"event-{seq}"}
            }
            await store.append_turn_event(turn_id, seq, durable_event, ttl=ttl)

        # Query from seq 2 — should return events 3, 4, 5 (seq > 2)
        events = await store.turn_events_since(turn_id, 2)

        assert len(events) == 3, f"Expected 3 events (3, 4, 5), got {len(events)}"
        for idx, event in enumerate(events):
            expected_seq = idx + 3  # 3, 4, 5
            assert event["seq"] == expected_seq, f"Event {idx} has seq={event['seq']}, expected {expected_seq}"
            assert event["stream"] == f"turn:{turn_id}"


@pytest.mark.asyncio
async def test_turn_events_since_skips_seq_counter_item():
    """Turn events_since MUST exclude the SEQ counter item (non-EVT# skip guard).

    The TURN#{turn_id} partition co-locates the atomic-seq counter item (SK="SEQ")
    with event items (SK="EVT#..."). Because "SEQ" > "EVT#<any-20-digit>"
    lexicographically, the SEQ item is returned by SK > EVT#{since_seq} queries.

    This test seeds BOTH the SEQ counter + real events and proves turn_events_since
    excludes the SEQ item before decoding item["event"], so no KeyError.
    """
    from moto import mock_aws
    import time
    import boto3
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    def _create_metadata_table(ddb_client, table_name: str) -> None:
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

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)

        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "test-turn-seq-guard"
        ttl = int(time.time()) + (7 * 24 * 3600)

        # Seed the partition with BOTH item shapes:
        # 1. Create the SEQ counter item (via allocate_turn_seq)
        await store.allocate_turn_seq(turn_id)
        await store.allocate_turn_seq(turn_id)  # Increment to 2

        # 2. Append 3 real events
        for seq in range(1, 4):
            durable_event = {
                "stream": f"turn:{turn_id}",
                "seq": seq,
                "kind": "text",
                "data": {"event": "text", "index": seq}
            }
            await store.append_turn_event(turn_id, seq, durable_event, ttl=ttl)

        # Query from seq 0 — MUST return only the 3 events (SEQ item excluded)
        # Returning the SEQ counter item would raise KeyError: 'event'.
        events = await store.turn_events_since(turn_id, 0)

        # Assert: 3 events returned (not 4 — SEQ excluded)
        assert len(events) == 3, (
            f"Expected 3 events, got {len(events)}. "
            f"SEQ counter item must be excluded from turn_events_since result."
        )

        # Assert: events have correct seq order (ascending 1, 2, 3)
        for idx, event in enumerate(events, start=1):
            assert event["seq"] == idx, f"Event {idx} has seq={event['seq']}, expected {idx}"
            assert event["stream"] == f"turn:{turn_id}"


@pytest.mark.asyncio
async def test_allocate_turn_seq_increments_and_delegates_atomicity():
    """Sequential allocations increment, and the request delegates atomicity to DDB.

    **What this test can and cannot prove.** The previous version fired three
    ``allocate_turn_seq`` calls concurrently through ``asyncio.gather`` and asserted
    three distinct seqs. That assertion was not testing the product code: moto's
    in-memory backend performs the ``ADD`` as a non-atomic read-modify-write and is
    not thread-safe, so three ``asyncio.to_thread`` callers can interleave and all
    observe 1 — which is what happens under macOS thread scheduling. The test was
    green on Linux by scheduling luck alone.

    Real DynamoDB ``UpdateItem`` with an ``ADD`` expression is applied server-side
    and is atomic per AWS's documented atomic-counter semantics; concurrency is
    therefore a property of the SERVICE, not of anything this process does, and no
    mock can demonstrate it. Asserting it against moto only manufactures a flake
    that erodes trust in the suite.

    So this test pins the two things that are actually ours and actually checkable:

    1. sequential allocations return strictly increasing values (the counter is
       wired to the right key and the response is parsed correctly), and
    2. the emitted request uses ``ADD`` with ``ReturnValues=UPDATED_NEW`` — the
       shape that hands atomicity to DynamoDB. A regression to a
       read-then-put-item implementation would still pass (1) but would lose
       cross-process safety, and (2) is what catches it.

    Cross-process atomicity under real concurrency can only be shown against real
    DynamoDB, not in a moto test.
    """
    from moto import mock_aws
    import boto3
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    def _create_metadata_table(ddb_client, table_name: str) -> None:
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

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)

        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        turn_id = "test-turn-atomic"

        # (1) Sequential allocations increment strictly. Deliberately NOT
        # concurrent: see the docstring — moto's ADD is a non-atomic
        # read-modify-write, so a concurrent assertion here tests the mock's
        # thread scheduling, not our code.
        seqs = [
            await store.allocate_turn_seq(turn_id),
            await store.allocate_turn_seq(turn_id),
            await store.allocate_turn_seq(turn_id),
        ]
        assert seqs == [1, 2, 3], f"Sequential seqs must increment, got {seqs}"

        # (2) The request delegates atomicity to DynamoDB. Spy on the client to
        # assert the wire shape: ADD (server-side atomic counter) with
        # UPDATED_NEW. A read-then-put rewrite would pass (1) and fail here.
        captured: dict = {}
        real_update_item = store.client.update_item

        def _spy_update_item(**kwargs):
            captured.update(kwargs)
            return real_update_item(**kwargs)

        store.client.update_item = _spy_update_item
        try:
            fourth = await store.allocate_turn_seq(turn_id)
        finally:
            store.client.update_item = real_update_item

        assert fourth == 4
        assert captured["UpdateExpression"] == "ADD seq :one", (
            "The counter must use an ADD expression so the increment is applied "
            "server-side (atomic across processes), not read-modify-written by the "
            f"caller. Got: {captured.get('UpdateExpression')!r}"
        )
        assert captured["ReturnValues"] == "UPDATED_NEW", (
            "UPDATED_NEW is what returns THIS caller's post-increment value; "
            f"anything else reintroduces a read race. Got: {captured.get('ReturnValues')!r}"
        )
        assert captured["Key"]["SK"] == {"S": "SEQ"}


@pytest.mark.asyncio
async def test_turn_events_since_unknown_turn_empty():
    """turn_events_since on an unknown turn_id returns [] (no crash)."""
    from moto import mock_aws
    import boto3
    from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore

    def _create_metadata_table(ddb_client, table_name: str) -> None:
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

    with mock_aws():
        dynamodb_client = boto3.client("dynamodb", region_name="us-east-1")
        table_name = "test-table"
        _create_metadata_table(dynamodb_client, table_name)

        store = DynamoDBMetadataStore(ddb_client=dynamodb_client, table_name=table_name)

        # Query a turn that was never created
        events = await store.turn_events_since("nonexistent-turn", 0)

        assert events == [], f"Expected empty list for unknown turn, got {events}"


# --- Dev adapter turn event tests ---


@pytest.mark.asyncio
async def test_dev_turn_events_suffix(tmp_path):
    """Dev adapter turn events_since returns seq>k suffix in ascending order."""
    from app.platform.dev_local import FilesystemMetadataStore

    store = FilesystemMetadataStore(projects_dir=tmp_path)

    turn_id = "dev-turn-123"
    ttl = 1234567890

    # Append 5 turn events
    for seq in range(1, 6):
        await store.allocate_turn_seq(turn_id)  # Allocate seq
        durable_event = {
            "stream": f"turn:{turn_id}",
            "seq": seq,
            "kind": "text" if seq < 5 else "result",
            "data": {"content": f"event-{seq}"}
        }
        await store.append_turn_event(turn_id, seq, durable_event, ttl=ttl)

    # Query from seq 2 — should return events 3, 4, 5
    events = await store.turn_events_since(turn_id, 2)

    assert len(events) == 3, f"Expected 3 events (3, 4, 5), got {len(events)}"
    for idx, event in enumerate(events):
        expected_seq = idx + 3  # 3, 4, 5
        assert event["seq"] == expected_seq, f"Event {idx} has seq={event['seq']}, expected {expected_seq}"


@pytest.mark.asyncio
async def test_dev_allocate_turn_seq_serial(tmp_path):
    """Dev adapter allocate_turn_seq is lock-serialized (distinct seqs)."""
    from app.platform.dev_local import FilesystemMetadataStore

    store = FilesystemMetadataStore(projects_dir=tmp_path)

    turn_id = "dev-turn-atomic"

    # Allocate 3 seqs concurrently
    seqs = await asyncio.gather(
        store.allocate_turn_seq(turn_id),
        store.allocate_turn_seq(turn_id),
        store.allocate_turn_seq(turn_id),
    )

    # All 3 must be distinct and increasing
    assert len(seqs) == 3
    assert len(set(seqs)) == 3, f"Seqs must be distinct, got {seqs}"
    assert seqs == sorted(seqs), f"Seqs must be increasing, got {seqs}"


@pytest.mark.asyncio
async def test_dev_turn_events_since_unknown_turn(tmp_path):
    """Dev adapter turn_events_since on unknown turn_id returns [] (no crash)."""
    from app.platform.dev_local import FilesystemMetadataStore

    store = FilesystemMetadataStore(projects_dir=tmp_path)

    # Query a turn that was never created
    events = await store.turn_events_since("nonexistent-turn", 0)

    assert events == [], f"Expected empty list for unknown turn, got {events}"
