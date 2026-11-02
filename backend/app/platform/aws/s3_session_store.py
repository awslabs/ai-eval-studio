# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3-backed Claude Agent SDK :class:`SessionStore`.

The SDK's ``SessionStore`` Protocol mirrors transcript JSONL entries to
an external store. We persist each ``append`` call as one part file:

    s3://{bucket}/{project_key}/{session_id}/{seq}.jsonl

Subagent transcripts (``key["subpath"]`` set) live under:

    s3://{bucket}/{project_key}/{session_id}/{subpath}/{seq}.jsonl

``seq`` is ``{epoch_ms:013d}-{instance_id}-{counter:09d}`` (see
:meth:`S3SessionStore._next_seq`). Reads (``load``) list the prefix, sort by
key (the zero-padded epoch-ms prefix makes lexicographic order time order),
and concatenate all parts.

We deliberately do NOT maintain summary sidecars (the SDK falls back to
``list_sessions`` + per-session ``load`` if ``list_session_summaries``
raises ``NotImplementedError``). They can be added later once ops
experience tells us whether the latency win is worth the extra write
per turn.

The store accepts either a static ``s3_client`` (falls back to the
platform's task-role client) OR a ``client_factory`` callable that
returns an STS-scoped client. The agent_runner constructs a per-turn
store with the scoped factory so SDK transcript writes go through
credentials narrowed to one project's
``project_key_for_directory(cwd)/*`` prefix (the SDK's own cwd-derived
key shape — see :mod:`sts_token_vending`) — defense in depth above
the app-layer ``require_access`` chokepoint.

**Concurrency model:** ``_load()`` issues GET operations concurrently
across up to ``LOAD_CONCURRENCY`` worker threads sharing one boto3 S3
client (boto3 clients are thread-safe). Results are reassembled in
sorted-key order, so the concatenated transcript is byte-identical to a
serial read.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

# Concurrency limit for parallel part-file reads in _load(). Coupled to
# the S3 client connection-pool size — both aws/__init__.py and
# agent_runner.py set max_pool_connections to this value so a 32-wide
# read fan-out never queues on the botocore default pool of 10.
LOAD_CONCURRENCY: int = 32


def _key_path(key: dict[str, Any]) -> str:
    """Compute the S3 prefix (no trailing slash) for a SessionKey.

    Args:
        key: A SessionKey dict with ``project_key``, ``session_id``, and
            optionally ``subpath``.
    """
    parts = [key["project_key"], key["session_id"]]
    sub = key.get("subpath")
    if sub:
        parts.append(sub)
    return "/".join(parts)


class S3SessionStore:
    """SDK :class:`SessionStore` backed by S3 part files.

    Args:
        s3_client: A boto3 S3 client. Used directly when no
            ``client_factory`` is provided. Tests inject a fake here.
        bucket: S3 bucket holding session transcripts.
        client_factory: Optional callable returning an S3 client. When
            set, every ``append`` / ``load`` operation calls this to
            get a fresh client — used by the agent_runner to plug in
            STS-scoped credentials per turn. The factory is responsible
            for caching; this store does not cache the result.
    """

    def __init__(
        self,
        s3_client: "BaseClient",
        bucket: str,
        *,
        client_factory: Callable[[], "BaseClient"] | None = None,
    ) -> None:
        self._fallback_s3 = s3_client
        self._client_factory = client_factory
        self.bucket = bucket
        # Per-instance counter combined with epoch-ms gives monotonically-
        # increasing keys even when the same instance issues multiple
        # appends within the same millisecond. ``_instance_id`` is a random
        # 8-hex slug that disambiguates two processes (coordinator + worker,
        # or two workers) appending to the same session in the same
        # millisecond — without it, both produce identical keys and S3
        # last-write-wins silently drops one part file.
        self._counter = itertools.count()
        self._counter_lock = threading.Lock()
        self._instance_id = uuid.uuid4().hex[:8]

    @property
    def s3(self) -> "BaseClient":
        """Return the S3 client to use for the next operation.

        When a ``client_factory`` is configured, it's called every time
        so the store always uses the freshest scoped credentials. The
        factory itself is expected to cache (5-min TTL via
        :class:`AwsCredentialCache`) so this isn't a per-call STS hit.
        """
        if self._client_factory is not None:
            return self._client_factory()
        return self._fallback_s3

    def _next_seq(self) -> str:
        """Return a strictly-increasing zero-padded sequence string.

        ``{epoch_ms:013d}-{instance_id}-{counter:09d}`` — 13 digits cover
        until the year 2286, the 8-hex instance id disambiguates two
        processes appending in the same millisecond, and 9 counter digits
        handle 1B+ appends per process before we'd worry about a wrap.
        Cross-process ordering still relies on the wall clock — that's
        acceptable because SDK session writes all originate from one
        coordinator turn at a time, so a process is the boundary anyway.
        Lexicographic order matches numeric order on the
        epoch-ms prefix, so list-then-concat reads still produce a
        time-ordered transcript.
        """
        with self._counter_lock:
            n = next(self._counter)
        return f"{int(time.time() * 1000):013d}-{self._instance_id}-{n:09d}"

    # ----- Required SDK Protocol methods -------------------------------------

    async def append(
        self, key: dict[str, Any], entries: list[dict[str, Any]]
    ) -> None:
        """Persist ``entries`` as one new JSONL part file under ``key``.

        Sequence keys come from :meth:`_next_seq` — a monotonic
        ``{epoch_ms}-{instance_id}-{counter}`` so multiple appends within the
        same millisecond never overwrite each other (the case the SDK
        conformance suite exercises with three rapid appends).
        """
        if not entries:
            return
        body = ("\n".join(json.dumps(e) for e in entries) + "\n").encode("utf-8")
        prefix = _key_path(key)
        s3_key = f"{prefix}/{self._next_seq()}.jsonl"

        def _put() -> None:
            self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=body)

        await asyncio.to_thread(_put)

    async def load(
        self, key: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Concatenate all part files under ``key`` and return entries.

        Returns ``None`` if no part files exist.
        """
        prefix = _key_path(key) + "/"

        def _load() -> list[dict[str, Any]] | None:
            # Capture the client once at the top — use the same client for
            # listing and every GET. boto3 clients are thread-safe.
            client = self.s3

            paginator = client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    k = obj["Key"]
                    # Skip nested subagent prefixes — load() is scoped to
                    # the exact (project, session, subpath) tuple.
                    rest = k[len(prefix):]
                    if "/" in rest:
                        continue
                    if k.endswith(".jsonl"):
                        keys.append(k)
            if not keys:
                return None
            keys.sort()

            # Per-key task: read + parse JSONL lines. Returns a list of
            # entries for one key.
            def _read_and_parse_key(k: str) -> list[dict[str, Any]]:
                resp = client.get_object(Bucket=self.bucket, Key=k)
                body = resp["Body"].read().decode("utf-8")
                key_entries: list[dict[str, Any]] = []
                for line in body.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        key_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "[s3-sessions] Skipping malformed JSONL line in %s", k
                        )
                return key_entries

            # Submit all GETs concurrently via a bounded ThreadPoolExecutor.
            # Pre-size a list indexed by the key's position in the sorted
            # keys list, so we can reassemble in sorted-key order.
            results: list[list[dict[str, Any]]] = [None] * len(keys)  # type: ignore
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=LOAD_CONCURRENCY
            ) as executor:
                futures = {
                    executor.submit(_read_and_parse_key, k): i
                    for i, k in enumerate(keys)
                }
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    # Call .result() to propagate GET exceptions out of
                    # _load() and therefore out of load(). Do NOT swallow
                    # transport errors.
                    results[idx] = future.result()

            # Flatten the per-key lists in order into the final entries list.
            # This reassembly is byte-identical to a serial read.
            entries: list[dict[str, Any]] = []
            for key_entries in results:
                entries.extend(key_entries)
            return entries

        return await asyncio.to_thread(_load)

    # ----- Optional SDK Protocol methods -------------------------------------

    async def list_sessions(
        self, project_key: str
    ) -> list[dict[str, Any]]:
        """Return ``[{session_id, mtime}]`` for all sessions under ``project_key``.

        ``mtime`` is the latest part-file ``LastModified`` in epoch ms.
        """
        prefix = project_key + "/"

        def _list() -> list[dict[str, Any]]:
            paginator = self.s3.get_paginator("list_objects_v2")
            # Map session_id -> latest mtime (ms).
            seen: dict[str, int] = {}
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    rel = obj["Key"][len(prefix):]
                    parts = rel.split("/")
                    if len(parts) < 2:
                        continue
                    # Only main transcripts (no subpath): exactly two parts —
                    # session_id/{seq}.jsonl
                    if len(parts) != 2:
                        continue
                    session_id = parts[0]
                    last_modified = obj.get("LastModified")
                    mtime_ms = (
                        int(last_modified.timestamp() * 1000)
                        if last_modified is not None
                        else 0
                    )
                    if mtime_ms > seen.get(session_id, 0):
                        seen[session_id] = mtime_ms
            return [
                {"session_id": sid, "mtime": mt} for sid, mt in seen.items()
            ]

        return await asyncio.to_thread(_list)

    async def list_session_summaries(self, project_key: str):
        # Optional. Not maintained — SDK falls back to list_sessions + load.
        raise NotImplementedError

    async def delete(self, key: dict[str, Any]) -> None:
        """Delete every part file under the SessionKey prefix.

        Per SDK contract, deleting a main-transcript key (no ``subpath``)
        cascades to all subkeys (subagent transcripts) under the session.
        """
        prefix = _key_path(key)
        if key.get("subpath") is None:
            # Cascade delete: every key whose path starts with
            # `{project_key}/{session_id}/` goes away.
            prefix = prefix + "/"
        else:
            prefix = prefix + "/"

        def _delete() -> None:
            paginator = self.s3.get_paginator("list_objects_v2")
            to_delete: list[dict[str, str]] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    to_delete.append({"Key": obj["Key"]})
                    # Batch in groups of 1000 (S3 API limit).
                    if len(to_delete) >= 1000:
                        self.s3.delete_objects(
                            Bucket=self.bucket,
                            Delete={"Objects": to_delete},
                        )
                        to_delete = []
            if to_delete:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": to_delete},
                )

        await asyncio.to_thread(_delete)

    async def list_subkeys(self, key: dict[str, Any]) -> list[str]:
        """Return all subpath suffixes under ``{project_key}/{session_id}``.

        Used by the SDK during resume to discover subagent transcripts.
        """
        prefix = f"{key['project_key']}/{key['session_id']}/"

        def _list() -> list[str]:
            paginator = self.s3.get_paginator("list_objects_v2")
            seen: set[str] = set()
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    rel = obj["Key"][len(prefix):]
                    parts = rel.split("/")
                    # Main transcript files have just 1 part (e.g. "<seq>.jsonl").
                    # Subkeys have 2+ parts; the subpath is everything but
                    # the trailing part-file name.
                    if len(parts) >= 2:
                        seen.add("/".join(parts[:-1]))
            return sorted(seen)

        return await asyncio.to_thread(_list)


__all__ = ["S3SessionStore"]
