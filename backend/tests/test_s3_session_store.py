# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK conformance tests for :class:`S3SessionStore`.

Validates our S3-backed Claude Agent SDK ``SessionStore`` against the SDK's
bundled conformance suite. The helper is
:func:`claude_agent_sdk.testing.run_session_store_conformance`, which asserts
the 14 behavioral contracts of the ``SessionStore`` Protocol (append, load,
optional list/delete/listSubkeys semantics, ordering, idempotency, etc.).

S3 is faked in-process with ``moto``, so nothing here reaches AWS and both
tests run with the default suite.
"""
from __future__ import annotations

import itertools

import boto3
import pytest
from moto import mock_aws

from app.platform.aws.s3_session_store import LOAD_CONCURRENCY, S3SessionStore


@pytest.mark.asyncio
async def test_s3_session_store_passes_sdk_conformance() -> None:
    """Drive ``run_session_store_conformance`` against an S3-backed store.

    The SDK helper invokes ``make_store`` once per contract for isolation.
    We hand it a factory closure that builds a fresh
    :class:`S3SessionStore` against a moto-faked S3 client each call.
    """
    try:
        from claude_agent_sdk.testing import run_session_store_conformance
    except ImportError as exc:  # pragma: no cover — guards against SDK rename
        pytest.skip(
            "claude_agent_sdk.testing.run_session_store_conformance not "
            f"available: {exc}"
        )

    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")

        # The SDK conformance helper invokes ``make_store`` once per
        # contract for *isolation* — the contracts assume a clean store
        # and assert exact append-order outputs. Sharing one bucket
        # across contracts (the previous shape) leaked writes from
        # earlier fragments into later ``load(_KEY)`` calls because
        # they all use the same canonical ``_KEY`` prefix. Hand each
        # call its own bucket so the only state crossing contract
        # boundaries is whatever ``make_store`` itself plumbs through.
        counter = itertools.count()

        def make_store():
            bucket = f"eval-studio-conformance-{next(counter)}"
            s3_client.create_bucket(Bucket=bucket)
            return S3SessionStore(s3_client=s3_client, bucket=bucket)

        # Skip the ``list_session_summaries`` contract — we deliberately
        # don't maintain summary sidecars (the SDK falls back to
        # ``list_sessions`` + per-session ``load``). The S3 implementation
        # raises ``NotImplementedError`` for this method by design;
        # ``skip_optional`` is the documented opt-out for that contract.
        await run_session_store_conformance(
            make_store,
            skip_optional=frozenset({"list_session_summaries"}),
        )


@pytest.mark.asyncio
async def test_load_parallelization_byte_identical_ordering() -> None:
    """Parallel _load() must produce byte-identical sorted-order concat.

    Writes 50+ part files (exceeds LOAD_CONCURRENCY=32), then asserts:
    (a) Loaded entries equal the appended entries in exact order.
    (b) One malformed JSONL line is skipped while other lines survive.
    (c) A GET error propagates out of load() — it does NOT degrade to [].
    """
    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket = "eval-studio-load-test"
        s3_client.create_bucket(Bucket=bucket)

        store = S3SessionStore(s3_client=s3_client, bucket=bucket)
        key = {"project_key": "proj-load", "session_id": "sess-load"}

        # (a) ORDERING / byte-identical concat: write 50+ parts (exceeds
        # LOAD_CONCURRENCY), record appended order, assert loaded == expected.
        expected_entries = []
        num_parts = LOAD_CONCURRENCY + 20  # 52 parts
        for i in range(num_parts):
            entry = {"uuid": f"entry-{i:03d}", "index": i}
            await store.append(key, [entry])
            expected_entries.append(entry)

        loaded = await store.load(key)
        assert loaded is not None, "load() returned None for non-empty session"
        assert loaded == expected_entries, (
            f"Loaded entries do not match appended order. "
            f"Expected {len(expected_entries)} entries, got {len(loaded)}."
        )

        # (b) MALFORMED-LINE SKIP: write one part with valid + malformed lines.
        # The malformed line is skipped with a warning, valid line survives.
        malformed_key = f"proj-load/sess-load/{store._next_seq()}.jsonl"
        malformed_body = b'{"uuid":"ok-after-malformed"}\n{ not json\n'
        s3_client.put_object(Bucket=bucket, Key=malformed_key, Body=malformed_body)

        loaded_with_malformed = await store.load(key)
        assert loaded_with_malformed is not None
        # Should contain all original entries + the valid "ok-after-malformed"
        assert len(loaded_with_malformed) == len(expected_entries) + 1
        assert loaded_with_malformed[-1] == {"uuid": "ok-after-malformed"}

        # (c) GET-ERROR PROPAGATION: patch get_object to raise after listing
        # succeeds. Assert load() raises, does NOT return [].
        original_get_object = s3_client.get_object

        def raising_get_object(**kwargs):
            # Let the first few GETs succeed, then raise for one key
            if "Key" in kwargs and "sess-raise" in kwargs["Key"]:
                raise RuntimeError("Simulated S3 unavailable")
            return original_get_object(**kwargs)

        s3_client.get_object = raising_get_object

        raise_key = {"project_key": "proj-raise", "session_id": "sess-raise"}
        await store.append(raise_key, [{"uuid": "will-fail"}])

        with pytest.raises(RuntimeError, match="Simulated S3 unavailable"):
            await store.load(raise_key)
