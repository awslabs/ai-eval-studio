# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DynamoDB-backed :class:`MetadataStore` (single-table design).

Schema (all items live in one table):

    PK                  | SK                          | GSI1PK       | GSI1SK
    --------------------|-----------------------------|--------------|---------
    USER#{uid}          | PROJECT#{pid}               | -            | -
    USER#{uid}          | PROJECT#{pid}#JOB#{jid}     | JOB#{state}  | JOB#{ts}#{jid}
    USER#{uid}          | PROJECT#{pid}#TURN#{tid}    | -            | -
    USER#{uid}          | PROJECT#{pid}#OPT           | -            | -
    USER#{uid}          | COST#{yyyy-mm}              | -            | -
    USER#{uid}          | AGENTCORE_SESSION#{pid}     | (see below)  | (see below)
    PROJECT_SEQ#{pid}   | SEQ                         | -            | -

The AgentCore session row carries ``GSI1PK="AGENTCORE_SESSION#running"`` /
``GSI1SK="USER#{uid}#{pid}"`` ONLY while its turn claim is held; both keys are
removed on release. That makes the partition a sparse index of live claims, which
the startup reconciler queries to find orphans without a full-table Scan.

Concurrency: :meth:`put_optimization_state` uses a conditional
``UpdateItem`` with ``attribute_not_exists(version) OR version =
:expected`` to provide optimistic locking.

GSI chokepoint: this is the ONLY module in the codebase allowed to
issue ``IndexName=`` queries. The grep test in
``tests/test_no_raw_gsi_queries.py`` enforces this.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ..metadata_store import (
    ConcurrentUpdateError,
    MetadataStore,
    ProjectOwnerConflictError,
    check_project_settings,
)
from ..types import JobId, ProjectId, TurnId, UserId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

#: Attempts to find a free outbox sort key before giving up. Each retry mints a
#: strictly higher key, so a collision only repeats if another writer is minting
#: keys in lockstep — three attempts is generous for a race that needs two
#: writers to land in the same nanosecond twice more.
_OUTBOX_PUT_ATTEMPTS = 3

# GSI1 partition for AgentCore session claims that are currently HELD.
#
# Written when a claim is taken and REMOVEd on every release/clear, which makes the
# partition a sparse index of live claims only. The startup reconciler queries it to
# find orphans, so its cost tracks the number of concurrent turns rather than the
# size of the table — and it needs no table-wide Scan permission.
#
# Both the write sites and the reconciler query must agree on this value, so it is a
# module constant rather than a repeated literal.
_AGENTCORE_SESSION_RUNNING_GSI1PK = "AGENTCORE_SESSION#running"

#: How long a launch-gate claim lives before DynamoDB reaps it.
#:
#: It only has to outlast one launch attempt, which is seconds of snapshot upload
#: plus a RunTask call. A day is chosen for the same reason the eval lock chose it:
#: comfortably longer than anything legitimate, short enough that an abandoned
#: claim cannot outlive interest in the job.
_LAUNCH_GATE_TTL_SECONDS = 86400

#: Attempts to settle a launch gate before giving up.
#:
#: More than one is needed only for the narrow interleaving where a claim's
#: conditional write loses to an existing claim and that claim is then released
#: before this caller can read it.
_LAUNCH_GATE_CLAIM_ATTEMPTS = 3

#: Reported when the gate could not be settled. Matches neither side's claimant,
#: so both callers take their conservative branch.
_LAUNCH_GATE_CONTENDED = "contended"


# ---------------------------------------------------------------------------
# DynamoDB JSON helpers
# ---------------------------------------------------------------------------


def _to_dynamo(value: Any) -> dict[str, Any]:
    """Convert a Python value to a DynamoDB AttributeValue dict.

    Supports ``str``, ``bool``, ``int``, ``float``, ``Decimal``, ``None``,
    ``list``, and ``dict``. We avoid pulling in ``boto3.dynamodb.types.TypeSerializer``
    so this module is straightforward to unit-test with stubbed
    low-level clients.
    """
    if value is None:
        return {"NULL": True}
    if isinstance(value, bool):
        return {"BOOL": value}
    # Accept Decimal so a value read back via _from_dynamo (which returns Decimal
    # for fractional numbers to preserve full DynamoDB precision) round-trips on
    # write without a type crash. str(Decimal) produces a valid DynamoDB N string.
    if isinstance(value, (int, float, Decimal)):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    if isinstance(value, list):
        return {"L": [_to_dynamo(v) for v in value]}
    if isinstance(value, dict):
        return {"M": {k: _to_dynamo(v) for k, v in value.items()}}
    raise TypeError(f"Unsupported DynamoDB value type: {type(value).__name__}")


def _from_dynamo(av: dict[str, Any]) -> Any:
    """Inverse of :func:`_to_dynamo`.

    Numbers are returned as ``int`` when the wire string is a pure integer
    and as :class:`decimal.Decimal` otherwise. ``Decimal`` preserves the
    full DynamoDB precision (38 significant digits) — using ``float`` here
    silently loses bits past 2^53, which would corrupt cumulative
    ``cost_usd`` totals or any future fixed-point field. Callers that
    need ``float`` should cast explicitly at the consumer boundary.
    """
    if "NULL" in av:
        return None
    if "BOOL" in av:
        return av["BOOL"]
    if "S" in av:
        return av["S"]
    if "N" in av:
        s = av["N"]
        return int(s) if s.lstrip("-").isdigit() else Decimal(s)
    if "L" in av:
        return [_from_dynamo(v) for v in av["L"]]
    if "M" in av:
        return {k: _from_dynamo(v) for k, v in av["M"].items()}
    if "SS" in av:
        return list(av["SS"])
    return None


def _item_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a DynamoDB item (full row) into a plain dict."""
    return {k: _from_dynamo(v) for k, v in item.items()}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DynamoDBMetadataStore(MetadataStore):
    """Cloud :class:`MetadataStore` over single-table DynamoDB.

    Args:
        ddb_client: Low-level boto3 DynamoDB client. Inject a stubbed
            client in tests via :class:`botocore.stub.Stubber`.
        table_name: Single-table name (default ``ai-eval-studio-metadata``).
    """

    def __init__(
        self, ddb_client: "BaseClient", table_name: str = "ai-eval-studio-metadata"
    ) -> None:
        self.client = ddb_client
        self.table = table_name

    # ----- Key construction --------------------------------------------------

    @staticmethod
    def _user_pk(user_id: UserId) -> str:
        return f"USER#{user_id}"

    @staticmethod
    def _project_sk(project_id: ProjectId) -> str:
        return f"PROJECT#{project_id}"

    @staticmethod
    def _project_seq_pk(project_id: ProjectId) -> str:
        """Dedicated partition key for the project seq counter.

        Lives under its own PK, not under USER#{uid}, so the two
        `begins_with(SK, "PROJECT#")` queries on the user partition are
        structurally unaffected by the counter item (key-namespace isolation).
        """
        return f"PROJECT_SEQ#{project_id}"

    @staticmethod
    def _job_sk(project_id: ProjectId, job_id: JobId) -> str:
        return f"PROJECT#{project_id}#JOB#{job_id}"

    @staticmethod
    def _turn_sk(project_id: ProjectId, turn_id: TurnId) -> str:
        return f"PROJECT#{project_id}#TURN#{turn_id}"

    @staticmethod
    def _eval_lock_sk(project_id: ProjectId) -> str:
        return f"EVAL_LOCK#{project_id}"

    @staticmethod
    def _launch_gate_sk(project_id: ProjectId, job_id: JobId) -> str:
        return f"LAUNCH_GATE#{project_id}#{job_id}"

    @staticmethod
    def _idempotency_sk(idempotency_key: str) -> str:
        return f"IDEMPOTENCY#{idempotency_key}"

    @staticmethod
    def _opt_sk(project_id: ProjectId) -> str:
        return f"PROJECT#{project_id}#OPT"

    @staticmethod
    def _cost_sk(year_month: str) -> str:
        return f"COST#{year_month}"

    @staticmethod
    def _project_index_pk(project_id: ProjectId) -> str:
        """Project-id → owner index. Used by ``find_project_by_id``."""
        return f"PROJECT_INDEX#{project_id}"

    @staticmethod
    def _agentcore_session_sk(project_id: ProjectId) -> str:
        """AgentCore session registry item under the user's partition."""
        return f"AGENTCORE_SESSION#{project_id}"

    # ----- Project records ---------------------------------------------------

    async def get_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._project_sk(project_id)),
                },
            )
            item = resp.get("Item")
            return _item_to_dict(item) if item else None

        return await asyncio.to_thread(_get)

    async def list_user_projects(
        self, user_id: UserId
    ) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(self._user_pk(user_id)),
                    # Prefix is "PROJECT#{pid}" only — exclude job/turn/opt sub-items.
                    ":sk": _to_dynamo("PROJECT#"),
                },
            ):
                for item in page.get("Items", []):
                    sk = item.get("SK", {}).get("S", "")
                    # Filter out anything that's a sub-record under a project.
                    rest = sk[len("PROJECT#"):]
                    if "#" not in rest:
                        results.append(_item_to_dict(item))
            return results

        return await asyncio.to_thread(_query)

    async def put_project(self, project: dict[str, Any]) -> None:
        if "id" not in project or "owner_id" not in project:
            raise ValueError("project record missing id/owner_id")
        item = dict(project)
        item["PK"] = self._user_pk(project["owner_id"])
        item["SK"] = self._project_sk(project["id"])

        # Index entry so find_project_by_id is O(1). The index carries
        # only owner_id; the full record stays under the owner's PK.
        index_item = {
            "PK": self._project_index_pk(project["id"]),
            "SK": "META",
            "owner_id": project["owner_id"],
            "project_id": project["id"],
        }

        def _put() -> None:
            ddb_item = {k: _to_dynamo(v) for k, v in item.items()}
            # ONE transaction, and the index put is conditioned on the id
            # being new or already this owner's. The index is what
            # ``find_project_by_id`` and so ``require_access`` resolve an id
            # through: an unconditioned put let any record naming an existing
            # id re-point it at a new owner, who then owned that project.
            # Both puts are one transaction, so a refused index write writes
            # no record either.
            try:
                self.client.transact_write_items(
                    TransactItems=[
                        {
                            "Put": {
                                "TableName": self.table,
                                "Item": {k: _to_dynamo(v) for k, v in index_item.items()},
                                "ConditionExpression": (
                                    "attribute_not_exists(PK) OR owner_id = :owner"
                                ),
                                "ExpressionAttributeValues": {
                                    ":owner": _to_dynamo(project["owner_id"]),
                                },
                            }
                        },
                        {"Put": {"TableName": self.table, "Item": ddb_item}},
                    ]
                )
            except self.client.exceptions.TransactionCanceledException as exc:
                reasons = exc.response.get("CancellationReasons") or []
                if (reasons[:1] or [{}])[0].get("Code") == "ConditionalCheckFailed":
                    raise ProjectOwnerConflictError(
                        f"project {project['id']} belongs to another owner"
                    ) from exc
                raise

        await asyncio.to_thread(_put)

    async def update_project_settings(
        self, owner_id: UserId, project_id: ProjectId, settings: dict[str, Any]
    ) -> bool:
        check_project_settings(settings)
        if not settings:
            return True
        names = {f"#f{i}": key for i, key in enumerate(settings)}
        values = {f":v{i}": _to_dynamo(settings[key]) for i, key in enumerate(settings)}

        def _update() -> bool:
            try:
                self.client.update_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(owner_id)),
                        "SK": _to_dynamo(self._project_sk(project_id)),
                    },
                    # SET on the named settings only: the record's owner is
                    # left exactly as the store holds it.
                    UpdateExpression="SET " + ", ".join(
                        f"{name} = :v{i}" for i, name in enumerate(names)
                    ),
                    ExpressionAttributeNames=names,
                    ExpressionAttributeValues=values,
                    ConditionExpression="attribute_exists(SK)",
                )
            except self.client.exceptions.ConditionalCheckFailedException:
                return False
            return True

        return await asyncio.to_thread(_update)

    async def delete_project(
        self, user_id: UserId, project_id: ProjectId
    ) -> bool:
        def _delete() -> bool:
            resp = self.client.delete_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._project_sk(project_id)),
                },
                ReturnValues="ALL_OLD",
            )
            existed = bool(resp.get("Attributes"))
            if existed:
                # Drop the project-id index too; failure here is non-
                # fatal — orphan index entries are tolerable, the
                # ``find_project_by_id`` caller treats a missing real
                # record as "not found".
                try:
                    self.client.delete_item(
                        TableName=self.table,
                        Key={
                            "PK": _to_dynamo(
                                self._project_index_pk(project_id)
                            ),
                            "SK": _to_dynamo("META"),
                        },
                    )
                except Exception:  # noqa: BLE001 - best-effort
                    logger.warning(
                        "[delete_project] failed to drop index for %s",
                        project_id,
                    )
            return existed

        return await asyncio.to_thread(_delete)

    async def find_project_by_id(
        self, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the project record by id, regardless of caller.

        Two-step: read the index entry to discover the owner, then load
        the real record from the owner's partition. Returns ``None``
        if either lookup misses.
        """
        def _find() -> dict[str, Any] | None:
            idx_resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._project_index_pk(project_id)),
                    "SK": _to_dynamo("META"),
                },
            )
            idx = idx_resp.get("Item")
            if not idx:
                return None
            owner_id = _from_dynamo(idx.get("owner_id", {}))
            if not owner_id:
                return None
            real_resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(owner_id)),
                    "SK": _to_dynamo(self._project_sk(project_id)),
                },
            )
            real = real_resp.get("Item")
            return _item_to_dict(real) if real else None

        return await asyncio.to_thread(_find)

    # ----- Job records -------------------------------------------------------

    async def get_job(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId, *, consistent: bool = False
    ) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._job_sk(project_id, job_id)),
                },
                # Strongly consistent only when the caller asks: the completion
                # seed does, so it cannot read a job ahead of its own result.
                ConsistentRead=consistent,
            )
            item = resp.get("Item")
            return _item_to_dict(item) if item else None

        return await asyncio.to_thread(_get)

    async def list_jobs(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            prefix = f"PROJECT#{project_id}#JOB#"
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(self._user_pk(user_id)),
                    ":sk": _to_dynamo(prefix),
                },
                ScanIndexForward=False,
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))
            return results

        return await asyncio.to_thread(_query)

    async def list_jobs_by_owner(self, owner_id: UserId) -> list[dict[str, Any]]:
        """Query base table for all of an owner's jobs across projects.

        Base-table query on ``PK=USER#{owner_id}`` with ``SK begins_with PROJECT#``,
        then filter to JOB items (exclude TURN/OPT/COST items sharing the partition).

        No IndexName= — this is a base-table owner query, so it needs NO GSI.
        GSI1 is already ``JOB#{state}`` for by-state ops; overloading it would
        conflict. The owner IS the base-table PK.
        """

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(self._user_pk(owner_id)),
                    ":sk": _to_dynamo("PROJECT#"),
                },
                ScanIndexForward=False,  # newest first
            ):
                for item in page.get("Items", []):
                    # Filter to JOB items only (SK contains #JOB#)
                    sk = item.get("SK", {}).get("S", "")
                    if "#JOB#" in sk:
                        results.append(_item_to_dict(item))
            return results

        return await asyncio.to_thread(_query)

    async def put_job(self, job: dict[str, Any]) -> None:
        if "job_id" not in job or "project_id" not in job:
            raise ValueError("job record missing job_id/project_id")
        owner_id = job.get("owner_id") or job.get("user_id")
        if not owner_id:
            raise ValueError("job record missing owner_id/user_id")
        item = dict(job)
        item["PK"] = self._user_pk(owner_id)
        item["SK"] = self._job_sk(job["project_id"], job["job_id"])
        # Maintain GSI1 for ops queries (list all running jobs).
        state = job.get("state", "unknown")
        ts = job.get("created_at", "")
        item["GSI1PK"] = f"JOB#{state}"
        item["GSI1SK"] = f"JOB#{ts}#{job['job_id']}"

        def _put() -> None:
            self.client.put_item(
                TableName=self.table,
                Item={k: _to_dynamo(v) for k, v in item.items()},
            )

        await asyncio.to_thread(_put)

    async def update_job_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        state: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        compute_handle: str | None = None,
        output_file: str | None = None,
    ) -> None:
        names: dict[str, str] = {}
        values: dict[str, dict[str, Any]] = {}
        sets: list[str] = []
        if state is not None:
            names["#state"] = "state"
            values[":state"] = _to_dynamo(state)
            sets.append("#state = :state")
            # Keep GSI1PK in sync so "all jobs in state X" queries work.
            names["#gsi1pk"] = "GSI1PK"
            values[":gsi1pk"] = _to_dynamo(f"JOB#{state}")
            sets.append("#gsi1pk = :gsi1pk")
        if progress is not None:
            names["#progress"] = "progress"
            values[":progress"] = _to_dynamo(progress)
            sets.append("#progress = :progress")
        if result is not None:
            names["#result"] = "result"
            values[":result"] = _to_dynamo(result)
            sets.append("#result = :result")
        if compute_handle is not None:
            names["#compute_handle"] = "compute_handle"
            values[":compute_handle"] = _to_dynamo(compute_handle)
            sets.append("#compute_handle = :compute_handle")
        if output_file is not None:
            names["#output_file"] = "output_file"
            values[":output_file"] = _to_dynamo(output_file)
            sets.append("#output_file = :output_file")
        if not sets:
            return

        def _update() -> None:
            self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._job_sk(project_id, job_id)),
                },
                UpdateExpression="SET " + ", ".join(sets),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )

        await asyncio.to_thread(_update)

    async def mark_cancel_requested(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Stamp ``cancel_requested_at`` on the existing job item.

        Conditional on the item existing, so a job id with no record upserts
        nothing — an unconditional SET would create a phantom item under that key.
        """
        from datetime import datetime, timezone

        def _mark() -> None:
            try:
                self.client.update_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(user_id)),
                        "SK": _to_dynamo(self._job_sk(project_id, job_id)),
                    },
                    UpdateExpression="SET cancel_requested_at = :now",
                    ConditionExpression="attribute_exists(PK)",
                    ExpressionAttributeValues={
                        ":now": _to_dynamo(datetime.now(timezone.utc).isoformat()),
                    },
                )
            except self.client.exceptions.ConditionalCheckFailedException:
                pass

        await asyncio.to_thread(_mark)

    async def claim_completion_fired(
        self, user_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Claim the exactly-once completion turn via conditional UpdateItem.

        Writes ``completion_fired = True`` and ``completion_fired_at = <now>``
        on the job item (PK=USER#{owner_id}, SK=PROJECT#{project_id}#JOB#{job_id}) with
        ConditionExpression ``attribute_not_exists(completion_fired)``. First
        caller wins (returns True); subsequent callers get
        ConditionalCheckFailedException (return False).

        The claim is an ATTRIBUTE on the EXISTING job item, not a
        separate claim item — no orphan cleanup.
        """
        from datetime import datetime, timezone

        def _claim() -> bool:
            try:
                self.client.update_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(user_id)),
                        "SK": _to_dynamo(self._job_sk(project_id, job_id)),
                    },
                    UpdateExpression="SET completion_fired = :true, completion_fired_at = :now",
                    # add attribute_exists guard so missing jobs return False
                    # (matches dev contract), not upsert a phantom item
                    ConditionExpression="attribute_exists(PK) AND attribute_not_exists(completion_fired)",
                    ExpressionAttributeValues={
                        ":true": _to_dynamo(True),
                        ":now": _to_dynamo(datetime.now(timezone.utc).isoformat()),
                    },
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Another caller already claimed — return False
                return False

        return await asyncio.to_thread(_claim)

    # --- Launch gate ---------------------------------

    async def claim_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> str:
        """First-wins claim on a job's launch gate. See the protocol for why.

        Returns the winning claimant, which is ``claimant`` when this caller won.
        """
        def _claim() -> str:
            key = {
                "PK": _to_dynamo(self._user_pk(owner_id)),
                "SK": _to_dynamo(self._launch_gate_sk(project_id, job_id)),
            }
            for attempt in range(1, _LAUNCH_GATE_CLAIM_ATTEMPTS + 1):
                try:
                    self.client.put_item(
                        TableName=self.table,
                        Item={
                            **key,
                            "claimant": _to_dynamo(claimant),
                            "claimed_at": _to_dynamo(int(time.time())),
                            "ttl": _to_dynamo(
                                int(time.time()) + _LAUNCH_GATE_TTL_SECONDS
                            ),
                        },
                        ConditionExpression="attribute_not_exists(SK)",
                    )
                    return claimant
                except self.client.exceptions.ConditionalCheckFailedException:
                    pass

                resp = self.client.get_item(
                    TableName=self.table, Key=key, ConsistentRead=True
                )
                item = resp.get("Item")
                if item is not None:
                    return _from_dynamo(item["claimant"])
                # The claim was released between the write and this read — a
                # dispatch abandoning a launch that never happened. Whoever asks
                # next is entitled to win, so try again rather than reporting a
                # holder that no longer exists.
                logger.info(
                    "[launch-gate] claim for job=%s lost a race with a release "
                    "(attempt %d); retrying",
                    job_id[:8], attempt,
                )
            # Both sides cannot keep releasing forever, but refusing to loop
            # without bound matters more than winning: report the gate as held by
            # the other side, which is the conservative answer for both callers —
            # dispatch declines to launch, cancel declines to infer absence.
            other = _LAUNCH_GATE_CONTENDED
            logger.error(
                "[launch-gate] job=%s could not be settled in %d attempts",
                job_id[:8], _LAUNCH_GATE_CLAIM_ATTEMPTS,
            )
            return other

        return await asyncio.to_thread(_claim)

    async def release_launch_gate(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        job_id: JobId,
        *,
        claimant: str,
    ) -> bool:
        """Drop this claimant's gate claim. Conditional, so it cannot clear another's."""
        def _release() -> bool:
            try:
                self.client.delete_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(owner_id)),
                        "SK": _to_dynamo(self._launch_gate_sk(project_id, job_id)),
                    },
                    ConditionExpression="claimant = :c",
                    ExpressionAttributeValues={":c": _to_dynamo(claimant)},
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Absent, or held by the other side. Either way there is nothing
                # this caller may remove.
                return False

        return await asyncio.to_thread(_release)

    # --- Eval concurrency lock -----------------------

    async def claim_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> bool:
        """Atomically claim eval lock for project via conditional PutItem.

        Writes a dedicated lock item (PK=USER#{owner_id},
        SK=EVAL_LOCK#{project_id}) with lock_holder_job_id=job_id,
        lock_version (starts at 1, increments on every claim — NOT reset to 1
        after release), lock_acquired_at=<now>, ttl=<now+86400> (24h cleanup,
        NOT primary release), ConditionExpression=attribute_not_exists(SK).
        First caller wins (returns True); subsequent callers get
        ConditionalCheckFailedException (return False).

        The lock_version is a monotonic counter that survives release/reclaim
        cycles — this is the fencing token for reconciler-clear.
        We achieve this by storing a separate VERSION_COUNTER item on the
        project that survives lock deletion.

        TTL is a 24h leaked-lock backstop only — never the primary release.
        DynamoDB TTL deletes "within a few days", too slow for a guard.
        """
        import time

        def _claim() -> bool:
            pk = _to_dynamo(self._user_pk(owner_id))
            lock_sk = _to_dynamo(self._eval_lock_sk(project_id))
            counter_sk = _to_dynamo(f"EVAL_LOCK_VERSION#{project_id}")

            # Obtain new_version atomically via DynamoDB ADD operation BEFORE
            # attempting the lock Put. This guarantees strictly-increasing
            # distinct versions across all claims, even under concurrent access.
            # A read-modify-write (get_item → compute new_version → Put in
            # transaction) could reuse a version number if two claimants read the
            # same counter value, defeating the reconciler's version fence.
            try:
                # Atomically increment the version counter and get the new value
                counter_resp = self.client.update_item(
                    TableName=self.table,
                    Key={"PK": pk, "SK": counter_sk},
                    UpdateExpression="ADD version :one",
                    ExpressionAttributeValues={":one": _to_dynamo(1)},
                    ReturnValues="UPDATED_NEW",
                )
                new_version = int(_from_dynamo(counter_resp["Attributes"]["version"]))

                # Now attempt the lock claim with the atomically-obtained version
                self.client.put_item(
                    TableName=self.table,
                    Item={
                        "PK": pk,
                        "SK": lock_sk,
                        "lock_holder_job_id": _to_dynamo(job_id),
                        "lock_version": _to_dynamo(new_version),
                        "lock_acquired_at": _to_dynamo(int(time.time())),
                        "ttl": _to_dynamo(int(time.time()) + 86400),
                    },
                    ConditionExpression="attribute_not_exists(SK)",
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Lock already held — the atomically-incremented counter is now
                # "burned" (a gap in the sequence), which is fine — versions
                # must be monotonic, not gapless.
                return False

        return await asyncio.to_thread(_claim)

    async def release_eval_lock(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> None:
        """Release eval lock (idempotent).

        Conditional DeleteItem with lock_holder_job_id = :jid. Swallows
        ConditionalCheckFailedException (idempotent no-op if not held by this
        job or already gone).
        """

        def _release() -> None:
            try:
                self.client.delete_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(owner_id)),
                        "SK": _to_dynamo(self._eval_lock_sk(project_id)),
                    },
                    ConditionExpression="lock_holder_job_id = :jid",
                    ExpressionAttributeValues={":jid": _to_dynamo(job_id)},
                )
            except self.client.exceptions.ConditionalCheckFailedException:
                # Not held by this job or already gone — idempotent no-op
                pass

        await asyncio.to_thread(_release)

    async def get_eval_lock(
        self, owner_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the eval lock item or None.

        Returns a dict with keys: lock_holder_job_id, lock_version,
        lock_acquired_at.
        """

        def _get() -> dict[str, Any] | None:
            response = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(owner_id)),
                    "SK": _to_dynamo(self._eval_lock_sk(project_id)),
                },
            )
            item = response.get("Item")
            if not item:
                return None

            return _item_to_dict(item)

        return await asyncio.to_thread(_get)

    async def clear_stale_eval_lock(
        self,
        owner_id: UserId,
        project_id: ProjectId,
        expected_version: int,
        expected_holder_job_id: JobId,
    ) -> bool:
        """Clear stale lock, fenced on version AND holder (reconciler path).

        Returns True if deleted, False if either fence tripped. The holder fence
        is the load-bearing one: the version only covers the read→delete window,
        so a lock reclaimed by a NEWER job before the read presented its own
        current version and was deleted anyway.
        """

        def _clear() -> bool:
            try:
                self.client.delete_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(owner_id)),
                        "SK": _to_dynamo(self._eval_lock_sk(project_id)),
                    },
                    ConditionExpression=(
                        "lock_version = :v AND lock_holder_job_id = :h"
                    ),
                    ExpressionAttributeValues={
                        ":v": _to_dynamo(expected_version),
                        ":h": _to_dynamo(expected_holder_job_id),
                    },
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Version changed, or the lock belongs to a different job now.
                return False

        return await asyncio.to_thread(_clear)

    # --- AgentCore session registry ---

    async def get_agentcore_session(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        """Return the AgentCore session item or None.

        Returns a dict with keys: runtime_session_id, app_session_id, status,
        updated_at, claim_version, ttl.
        """

        def _get() -> dict[str, Any] | None:
            response = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._agentcore_session_sk(project_id)),
                },
            )
            item = response.get("Item")
            if not item:
                return None

            return _item_to_dict(item)

        return await asyncio.to_thread(_get)

    async def put_agentcore_session(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        runtime_session_id: str | None,
        app_session_id: str | None,
        ttl: int,
    ) -> None:
        """Upsert runtime/app session ids without touching status.

        Creates the item at status="idle" if absent. Does NOT overwrite
        status if present — recording the mapping mid-idle does not forge
        a claim.
        """
        import time

        def _put() -> None:
            pk = _to_dynamo(self._user_pk(user_id))
            sk = _to_dynamo(self._agentcore_session_sk(project_id))

            # Upsert identity fields, initialize status to idle only if absent.
            # Only SET runtime_session_id / app_session_id when non-None, so
            # recording a mapping never nulls out a good stored id.
            update_expr_parts = [
                "updated_at = :now",
                "#ttl = :ttl",
            ]
            expr_attr_values = {
                ":now": _to_dynamo(int(time.time())),
                ":ttl": _to_dynamo(ttl),
                ":idle": _to_dynamo("idle"),
            }

            # Conditionally add runtime_session_id and app_session_id to SET clause
            if runtime_session_id is not None:
                update_expr_parts.append("runtime_session_id = :rsid")
                expr_attr_values[":rsid"] = _to_dynamo(runtime_session_id)
            if app_session_id is not None:
                update_expr_parts.append("app_session_id = :asid")
                expr_attr_values[":asid"] = _to_dynamo(app_session_id)

            # Set status to idle only if it doesn't exist (status is reserved keyword)
            update_expr_parts.append("#status = if_not_exists(#status, :idle)")

            self.client.update_item(
                TableName=self.table,
                Key={"PK": pk, "SK": sk},
                UpdateExpression="SET " + ", ".join(update_expr_parts),
                ExpressionAttributeNames={"#ttl": "ttl", "#status": "status"},
                ExpressionAttributeValues=expr_attr_values,
            )

        await asyncio.to_thread(_put)

    async def claim_agentcore_session_turn(
        self, user_id: UserId, project_id: ProjectId, *, ttl: int, current_turn_id: str | None = None
    ) -> bool:
        """Atomically claim session turn via conditional idle→running write.

        Conditional UpdateItem SET status="running" guarded by status="idle"
        or attribute_not_exists. First caller wins (returns True); concurrent
        callers get ConditionalCheckFailedException (return False).

        Bumps claim_version (monotonic, survives release) and updated_at on win.
        Optionally stores current_turn_id for in-flight turn discovery.
        """
        import time

        def _claim() -> bool:
            pk = _to_dynamo(self._user_pk(user_id))
            sk = _to_dynamo(self._agentcore_session_sk(project_id))
            counter_sk = _to_dynamo(f"AGENTCORE_SESSION_VERSION#{project_id}")

            try:
                # Atomically increment the version counter (survives release)
                counter_resp = self.client.update_item(
                    TableName=self.table,
                    Key={"PK": pk, "SK": counter_sk},
                    UpdateExpression="ADD version :one",
                    ExpressionAttributeValues={":one": _to_dynamo(1)},
                    ReturnValues="UPDATED_NEW",
                )
                new_version = int(_from_dynamo(counter_resp["Attributes"]["version"]))

                # Build UpdateExpression conditionally.
                #
                # GSI1 keys are written on claim and REMOVED on release/clear, so
                # GSI1PK="AGENTCORE_SESSION#running" is a SPARSE index holding only
                # the claims that are actually held right now. That is what lets the
                # startup reconciler find orphans with a bounded Query instead of a
                # full-table Scan: the index is sized by concurrent turns, not by the
                # table. It also keeps the coordinator role free of table-wide Scan.
                update_expr = (
                    "SET #status = :running, "
                    "claim_version = :version, "
                    "updated_at = :now, "
                    "#ttl = :ttl, "
                    "GSI1PK = :gsi1pk, "
                    "GSI1SK = :gsi1sk"
                )
                attr_values = {
                    ":running": _to_dynamo("running"),
                    ":version": _to_dynamo(new_version),
                    ":now": _to_dynamo(int(time.time())),
                    ":ttl": _to_dynamo(ttl),
                    ":idle": _to_dynamo("idle"),
                    ":gsi1pk": _to_dynamo(_AGENTCORE_SESSION_RUNNING_GSI1PK),
                    # user+project makes the sort key unique per claim, so two users
                    # holding claims on different projects never collide.
                    ":gsi1sk": _to_dynamo(f"{self._user_pk(user_id)}#{project_id}"),
                }
                if current_turn_id is not None:
                    update_expr += ", current_turn_id = :turn_id"
                    attr_values[":turn_id"] = _to_dynamo(current_turn_id)

                # Claim the session: status idle→running (or create at running)
                self.client.update_item(
                    TableName=self.table,
                    Key={"PK": pk, "SK": sk},
                    UpdateExpression=update_expr,
                    ExpressionAttributeNames={"#status": "status", "#ttl": "ttl"},
                    ExpressionAttributeValues=attr_values,
                    # An EXPIRED claim must be claimable. DynamoDB TTL deletion is
                    # a best-effort background process: AWS documents that expired
                    # items are "typically" deleted within a few DAYS, not at the
                    # expiry instant, and the documented pattern is to filter
                    # expired items application-side. So `#status = :idle` alone
                    # let a leaked `running` claim block its (user, project) pair
                    # for days on a long-lived coordinator — the TTL was not the
                    # backstop it was assumed to be. Treat an item whose ttl has
                    # passed as free, matching what TTL will eventually do anyway.
                    ConditionExpression=(
                        "#status = :idle "
                        "OR attribute_not_exists(#status) "
                        "OR #ttl < :now "
                        "OR attribute_not_exists(#ttl)"
                    ),
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Status is not idle (already running) or changed concurrently
                return False

        return await asyncio.to_thread(_claim)

    async def release_agentcore_session_turn(
        self,
        user_id: UserId,
        project_id: ProjectId,
        *,
        expected_turn_id: str | None = None,
    ) -> None:
        """Release session turn claim (idempotent running→idle).

        Conditional UpdateItem SET status="idle". Swallows
        ConditionalCheckFailedException (idempotent no-op if not running, item
        gone, or held by another turn). Refreshes updated_at + ttl on success.

        ``expected_turn_id`` fences the release to its own turn, and callers that
        know their turn should always pass it. Conditioning on ``status =
        running`` alone made this a cross-turn hazard rather than a no-op: a
        stale turn's release runs from a ``finally``, and a turn admitted after
        the stale one's lease expired sets the same item back to running — so the
        late release matched the condition and flipped the NEW turn's live claim
        to idle, admitting a third concurrent writer to the same project.
        """
        import time

        def _release() -> None:
            condition = "#status = :running"
            values = {
                ":idle": _to_dynamo("idle"),
                ":running": _to_dynamo("running"),
                ":now": _to_dynamo(int(time.time())),
                ":ttl": _to_dynamo(int(time.time()) + 86400),  # +24h
            }
            if expected_turn_id is not None:
                condition += " AND current_turn_id = :tid"
                values[":tid"] = _to_dynamo(expected_turn_id)

            try:
                self.client.update_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(user_id)),
                        "SK": _to_dynamo(self._agentcore_session_sk(project_id)),
                    },
                    # REMOVE the GSI1 keys so this item drops out of the sparse
                    # running-claims index. Without the removal the reconciler's
                    # Query would keep returning released claims forever, and the
                    # index would grow without bound.
                    UpdateExpression=(
                        "SET #status = :idle, updated_at = :now, #ttl = :ttl "
                        "REMOVE GSI1PK, GSI1SK"
                    ),
                    ExpressionAttributeNames={"#status": "status", "#ttl": "ttl"},
                    ExpressionAttributeValues=values,
                    ConditionExpression=condition,
                )
            except self.client.exceptions.ConditionalCheckFailedException:
                # Not running, already gone, or held by a DIFFERENT turn —
                # idempotent no-op in every case.
                pass

        await asyncio.to_thread(_release)

    async def list_agentcore_sessions(self, user_id: UserId) -> list[dict[str, Any]]:
        """List all AgentCore sessions for a user.

        Base-table query PK=USER#{uid} with SK prefix filter. No IndexName=.
        """

        def _list() -> list[dict[str, Any]]:
            pk = _to_dynamo(self._user_pk(user_id))
            prefix = "AGENTCORE_SESSION#"

            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")

            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
                ExpressionAttributeValues={
                    ":pk": pk,
                    ":prefix": _to_dynamo(prefix),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))

            return results

        return await asyncio.to_thread(_list)

    async def list_running_agentcore_sessions(self) -> list[dict[str, Any]]:
        """List ALL running AgentCore sessions across users (claim reconcilers).

        Paginated ``IndexName="GSI1"`` query on
        ``GSI1PK="AGENTCORE_SESSION#running"``, reusing the existing index — no new
        one is added. Mirrors the jobs-by-state query pattern next door.

        This was a full-table Scan with a filter on the SK prefix. Two problems with
        that: the coordinator role has no table-wide ``dynamodb:Scan`` (and granting
        it would widen the blast radius of a role that is otherwise key-scoped), so
        the reconciler failed with AccessDenied and silently cleared nothing; and a
        Scan reads the whole table to return a handful of items, getting slower and
        costlier as the table grows.

        The claim write adds the GSI1 keys and every release/clear REMOVEs them, so
        this index is SPARSE — it holds only the claims held right now, bounded by
        concurrent turns rather than by table size. Cost is proportional to the
        answer, and the query needs only ``dynamodb:Query`` on the index, which the
        role already has.
        """

        def _list_running() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")

            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :gsi1pk",
                ExpressionAttributeValues={
                    ":gsi1pk": _to_dynamo(_AGENTCORE_SESSION_RUNNING_GSI1PK),
                },
            ):
                for item in page.get("Items", []):
                    session_dict = _item_to_dict(item)
                    # Extract user_id from PK (format: USER#{user_id})
                    pk = session_dict.get("PK", "")
                    if pk.startswith("USER#"):
                        session_dict["user_id"] = pk[5:]  # Strip "USER#" prefix
                    # Extract project_id from SK (format: AGENTCORE_SESSION#{project_id})
                    sk = session_dict.get("SK", "")
                    if sk.startswith("AGENTCORE_SESSION#"):
                        session_dict["project_id"] = sk[18:]  # Strip prefix
                    results.append(session_dict)

            return results

        return await asyncio.to_thread(_list_running)

    async def clear_stale_agentcore_session_claim(
        self, user_id: UserId, project_id: ProjectId, expected_version: int
    ) -> bool:
        """Clear stale running claim with version fence.

        Returns True if cleared, False if claim_version changed (claim was
        released+reclaimed since read). Conditional UpdateItem status→idle
        only when claim_version = :expected_version.
        """
        import time

        def _clear() -> bool:
            try:
                self.client.update_item(
                    TableName=self.table,
                    Key={
                        "PK": _to_dynamo(self._user_pk(user_id)),
                        "SK": _to_dynamo(self._agentcore_session_sk(project_id)),
                    },
                    # REMOVE the GSI1 keys for the same reason as the normal
                    # release: a cleared claim must leave the sparse running-claims
                    # index, or the next reconcile pass finds it again forever.
                    UpdateExpression=(
                        "SET #status = :idle, updated_at = :now "
                        "REMOVE GSI1PK, GSI1SK"
                    ),
                    ExpressionAttributeNames={"#status": "status"},
                    ExpressionAttributeValues={
                        ":idle": _to_dynamo("idle"),
                        ":now": _to_dynamo(int(time.time())),
                        ":expected": _to_dynamo(expected_version),
                    },
                    ConditionExpression="claim_version = :expected",
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # claim_version changed — claim was released+reclaimed
                return False

        return await asyncio.to_thread(_clear)

    # --- Dispatch idempotency + reconciler helpers ---

    async def claim_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId, ttl_seconds: int
    ) -> tuple[bool, str | None]:
        """Claim idempotency key.

        Returns (claimed, existing_job_id): (True, None) when this call freshly
        claimed the key, (False, <first job's job_id>) when a prior dispatch
        already claimed it within TTL.

        Conditional PutItem on SK=IDEMPOTENCY#{key} under PK=USER#{owner_id},
        stores job_id + ttl; on ConditionalCheckFailedException read back the
        stored job_id.
        """
        import time

        def _claim() -> tuple[bool, str | None]:
            pk = _to_dynamo(self._user_pk(owner_id))
            sk = _to_dynamo(self._idempotency_sk(idempotency_key))

            try:
                self.client.put_item(
                    TableName=self.table,
                    Item={
                        "PK": pk,
                        "SK": sk,
                        "job_id": _to_dynamo(job_id),
                        "ttl": _to_dynamo(int(time.time()) + ttl_seconds),
                    },
                    ConditionExpression="attribute_not_exists(SK)",
                    ReturnValuesOnConditionCheckFailure="ALL_OLD",
                )
                return (True, None)
            except self.client.exceptions.ConditionalCheckFailedException as e:
                # Key already exists — read back the stored job_id
                existing_item = e.response.get("Item")
                if existing_item:
                    existing_job_id = _from_dynamo(existing_item.get("job_id", {"S": ""}))
                    return (False, existing_job_id)
                # Fallback: read explicitly if not in exception response
                response = self.client.get_item(
                    TableName=self.table,
                    Key={"PK": pk, "SK": sk},
                )
                item = response.get("Item")
                if item:
                    existing_job_id = _from_dynamo(item.get("job_id", {"S": ""}))
                    return (False, existing_job_id)
                # Edge case: item was there but vanished (expired TTL) — treat as fresh
                return (True, None)

        return await asyncio.to_thread(_claim)

    async def release_idempotency_key(
        self, owner_id: UserId, idempotency_key: str, job_id: JobId
    ) -> bool:
        """Release idempotency key (conditional delete guarded by job_id ownership).

        Releases the key ONLY if the stored job_id matches the caller's job_id.
        Returns False on ConditionalCheckFailedException (ownership guard tripped
        — stored job_id does not match, or key already gone).

        Conditional DeleteItem on SK=IDEMPOTENCY#{key} with
        ConditionExpression="job_id = :job_id".
        """

        def _release() -> bool:
            pk = _to_dynamo(self._user_pk(owner_id))
            sk = _to_dynamo(self._idempotency_sk(idempotency_key))

            try:
                self.client.delete_item(
                    TableName=self.table,
                    Key={"PK": pk, "SK": sk},
                    ConditionExpression="job_id = :job_id",
                    ExpressionAttributeValues={
                        ":job_id": _to_dynamo(job_id),
                    },
                )
                return True
            except self.client.exceptions.ConditionalCheckFailedException:
                # Ownership guard tripped: stored job_id does not match,
                # or key does not exist
                return False

        return await asyncio.to_thread(_release)

    async def count_active_jobs_for_owner(self, owner_id: UserId) -> int:
        """Count non-terminal jobs for owner (per-user cap).

        Returns count of jobs in states (queued, running, awaiting_user).
        Terminal states (completed, failed, cancelled) are excluded.

        Calls list_jobs_by_owner and filters to non-terminal.
        """
        jobs = await self.list_jobs_by_owner(owner_id)
        non_terminal_states = {"queued", "running", "awaiting_user"}
        return sum(1 for job in jobs if job.get("state") in non_terminal_states)

    async def list_non_terminal_jobs(self) -> list[dict[str, Any]]:
        """List all non-terminal jobs across users (reconciler).

        Returns jobs in states (running, queued). Used by the reconciler,
        the account backstop cap, and startup reattach.

        Paginated IndexName="GSI1" query for GSI1PK="JOB#running" and
        GSI1PK="JOB#queued", union the results.
        """

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")

            # Query for running jobs
            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :state",
                ExpressionAttributeValues={
                    ":state": _to_dynamo("JOB#running"),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))

            # Query for queued jobs
            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :state",
                ExpressionAttributeValues={
                    ":state": _to_dynamo("JOB#queued"),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))

            return results

        return await asyncio.to_thread(_query)

    async def list_recent_completed_jobs(
        self, within_seconds: int = 86400
    ) -> list[dict[str, Any]]:
        """List recent terminal jobs for completion turn recovery.

        Returns jobs in states (completed, failed) created within the recency window.
        EXCLUDES cancelled jobs (no turn owed).

        Paginated IndexName="GSI1" queries for GSI1PK="JOB#completed" and
        GSI1PK="JOB#failed", with GSI1SK >= JOB#{since_iso} range condition.
        """
        from datetime import datetime, timezone
        import time

        now = int(time.time())
        since_ts = now - within_seconds
        since_iso = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")

            # Query for completed jobs within window
            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :state AND GSI1SK >= :since",
                ExpressionAttributeValues={
                    ":state": _to_dynamo("JOB#completed"),
                    ":since": _to_dynamo(f"JOB#{since_iso}"),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))

            # Query for failed jobs within window
            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :state AND GSI1SK >= :since",
                ExpressionAttributeValues={
                    ":state": _to_dynamo("JOB#failed"),
                    ":since": _to_dynamo(f"JOB#{since_iso}"),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))

            return results

        return await asyncio.to_thread(_query)

    # ----- Turn records ------------------------------------------------------

    async def get_turn(
        self, user_id: UserId, project_id: ProjectId, turn_id: TurnId
    ) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._turn_sk(project_id, turn_id)),
                },
            )
            item = resp.get("Item")
            return _item_to_dict(item) if item else None

        return await asyncio.to_thread(_get)

    async def list_active_turns(
        self, user_id: UserId, project_id: ProjectId
    ) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            prefix = f"PROJECT#{project_id}#TURN#"
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(self._user_pk(user_id)),
                    ":sk": _to_dynamo(prefix),
                },
            ):
                for item in page.get("Items", []):
                    record = _item_to_dict(item)
                    if record.get("state") in ("running", "interrupted"):
                        results.append(record)
            return results

        return await asyncio.to_thread(_query)

    async def put_turn(self, turn: dict[str, Any]) -> None:
        if "turn_id" not in turn or "project_id" not in turn:
            raise ValueError("turn record missing turn_id/project_id")
        owner_id = turn.get("owner_id") or turn.get("user_id")
        if not owner_id:
            raise ValueError("turn record missing owner_id/user_id")
        item = dict(turn)
        item["PK"] = self._user_pk(owner_id)
        item["SK"] = self._turn_sk(turn["project_id"], turn["turn_id"])
        # Make sure last_heartbeat is always present so the reaper can scan.
        item.setdefault("last_heartbeat", turn.get("started_at", ""))

        def _put() -> None:
            self.client.put_item(
                TableName=self.table,
                Item={k: _to_dynamo(v) for k, v in item.items()},
            )

        await asyncio.to_thread(_put)

    async def update_turn_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        turn_id: TurnId,
        *,
        state: str | None = None,
        last_heartbeat: str | None = None,
        completed_at: str | None = None,
        interrupt_reason: str | None = None,
    ) -> None:
        names: dict[str, str] = {}
        values: dict[str, dict[str, Any]] = {}
        sets: list[str] = []
        for field, val in (
            ("state", state),
            ("last_heartbeat", last_heartbeat),
            ("completed_at", completed_at),
            ("interrupt_reason", interrupt_reason),
        ):
            if val is None:
                continue
            names[f"#{field}"] = field
            values[f":{field}"] = _to_dynamo(val)
            sets.append(f"#{field} = :{field}")
        if not sets:
            return

        def _update() -> None:
            self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._turn_sk(project_id, turn_id)),
                },
                UpdateExpression="SET " + ", ".join(sets),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )

        await asyncio.to_thread(_update)

    # ----- Optimization state ------------------------------------------------

    async def get_optimization_state(
        self, user_id: UserId, project_id: ProjectId
    ) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._opt_sk(project_id)),
                },
            )
            item = resp.get("Item")
            return _item_to_dict(item) if item else None

        return await asyncio.to_thread(_get)

    async def put_optimization_state(
        self,
        user_id: UserId,
        project_id: ProjectId,
        state: dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> int:
        # Compute the new version client-side; the conditional check
        # ensures nobody else committed in between.
        new_version = (expected_version or 0) + 1
        item = {**state, "version": new_version}
        item["PK"] = self._user_pk(user_id)
        item["SK"] = self._opt_sk(project_id)

        def _put() -> int:
            kwargs: dict[str, Any] = {
                "TableName": self.table,
                "Item": {k: _to_dynamo(v) for k, v in item.items()},
            }
            if expected_version is None:
                # First-write protection: only succeed if no record yet.
                kwargs["ConditionExpression"] = "attribute_not_exists(version)"
            else:
                kwargs["ConditionExpression"] = (
                    "attribute_not_exists(version) OR version = :expected"
                )
                kwargs["ExpressionAttributeValues"] = {
                    ":expected": _to_dynamo(expected_version)
                }
            try:
                self.client.put_item(**kwargs)
            except self.client.exceptions.ConditionalCheckFailedException as exc:
                raise ConcurrentUpdateError(
                    f"optimization state version mismatch (expected={expected_version})"
                ) from exc
            return new_version

        return await asyncio.to_thread(_put)

    # ----- Event log ------------------------------------------------

    async def allocate_job_seq(self, job_id: JobId) -> int:
        """Atomically allocate the next event seq using DynamoDB ADD counter.

        Uses ``UpdateItem ADD seq :one`` with ``ReturnValues=UPDATED_NEW`` on
        ``PK=JOB#{job_id}, SK=SEQ``. Each caller gets a distinct, strictly
        increasing value (DynamoDB atomic counter semantics — writes applied
        in received order, each returns the post-increment value).

        Thread-safety: Cross-process safe by design (DynamoDB atomic counter).
        """

        def _add() -> int:
            resp = self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(f"JOB#{job_id}"),
                    "SK": _to_dynamo("SEQ"),
                },
                UpdateExpression="ADD seq :one",
                ExpressionAttributeValues={":one": _to_dynamo(1)},
                ReturnValues="UPDATED_NEW",
            )
            # _from_dynamo returns int for a pure-integer N
            return int(_from_dynamo(resp["Attributes"]["seq"]))

        return await asyncio.to_thread(_add)

    # ----- Project event outbox -----------------------------------------

    async def put_outbox_event(
        self, owner_id: UserId, project_id: ProjectId, event: dict[str, Any]
    ) -> None:
        """Stash an unroutable microVM project event for coordinator drain.

        ``PK=USER#{owner}``, ``SK=OUTBOX#{project}#{ns-timestamp}`` — the
        owner's partition, so the microVM's LeadingKeys-scoped policy covers
        the write with no IAM change (same reasoning as the OPT record the
        dispatch tool already writes from the microVM). 1-hour TTL: an event
        that old is stale by definition and expires unread.

        **Conditional on the key being free.** A plain ``put_item`` overwrites,
        so a sort-key collision destroyed the earlier event with no error and no
        log. ``outbox_sk`` now guarantees uniqueness within a process; this
        condition covers the case it cannot — two writers, which happens by
        design during a rolling deploy (two coordinator tasks overlap) and
        whenever a microVM stashes while the coordinator does. On a collision we
        mint a fresh key and retry, so the loss becomes a retry rather than a
        deletion.

        Raises:
            RuntimeError: If every attempt collides. The caller
                (``stash_unroutable_event``) already treats a raised stash as
                "event lost" and logs it — losing an event LOUDLY is the point;
                the old behavior lost it silently.
        """
        from app.services.project_event_outbox import outbox_sk

        base_item = {
            "PK": self._user_pk(owner_id),
            "event": event,
            "project_id": project_id,
            "ttl": int(time.time()) + 3600,
        }

        def _put(sk: str) -> None:
            item = {**base_item, "SK": sk}
            self.client.put_item(
                TableName=self.table,
                Item={k: _to_dynamo(v) for k, v in item.items()},
                ConditionExpression="attribute_not_exists(SK)",
            )

        # Each attempt mints a NEW key: the retry has to move off the taken one.
        for attempt in range(_OUTBOX_PUT_ATTEMPTS):
            sk = outbox_sk(project_id)
            try:
                await asyncio.to_thread(_put, sk)
                return
            except self.client.exceptions.ConditionalCheckFailedException:
                logger.warning(
                    "[ddb] outbox key already taken (attempt %d/%d) project=%s "
                    "— another writer minted the same key; retrying with a new one",
                    attempt + 1, _OUTBOX_PUT_ATTEMPTS, project_id[:12],
                )
        raise RuntimeError(
            f"Could not stash outbox event for project={project_id[:12]}: "
            f"{_OUTBOX_PUT_ATTEMPTS} sort keys in a row were already taken"
        )

    async def list_outbox_events(
        self, owner_id: UserId, project_id: ProjectId, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Oldest-first stashed events for one project (base-table query)."""

        def _query() -> list[dict[str, Any]]:
            resp = self.client.query(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(self._user_pk(owner_id)),
                    ":sk": _to_dynamo(f"OUTBOX#{project_id}#"),
                },
                ScanIndexForward=True,  # ns-timestamp SK → oldest first
                Limit=limit,
            )
            return [_item_to_dict(item) for item in resp.get("Items", [])]

        return await asyncio.to_thread(_query)

    async def delete_outbox_event(self, owner_id: UserId, sk: str) -> None:
        """Delete one drained outbox item. Missing item is a no-op."""

        def _delete() -> None:
            self.client.delete_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(owner_id)),
                    "SK": _to_dynamo(sk),
                },
            )

        await asyncio.to_thread(_delete)

    async def append_job_event(
        self, job_id: JobId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one durable event item.

        Item shape: ``PK=JOB#{job_id}, SK=EVT#{seq:020d}, event=<dict>, ttl=<int>``.
        Zero-padding ``seq`` to 20 digits makes string-SK ordering == numeric ordering.
        """
        item = {
            "PK": f"JOB#{job_id}",
            "SK": f"EVT#{seq:020d}",
            "event": event,
            "ttl": ttl,
        }

        def _put() -> None:
            self.client.put_item(
                TableName=self.table, Item={k: _to_dynamo(v) for k, v in item.items()}
            )

        await asyncio.to_thread(_put)

    async def events_since(
        self, job_id: JobId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Query ordered event suffix from the base table (no GSI).

        ``KeyConditionExpression="PK = :pk AND SK > :sk"``, ``ScanIndexForward=True``.
        Zero-padded SK ensures lexicographic ordering == numeric seq ordering.

        The JOB#{job_id} partition co-locates the atomic-seq counter item (SK="SEQ")
        with event items (SK="EVT#..."). Because "SEQ" sorts lexicographically AFTER
        any "EVT#" key, an unfiltered SK > :sk query always returns the counter item.
        This method skips non-EVT# items (including the SEQ counter) before decoding
        the event attribute, so the caller receives only event items.
        """

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND SK > :sk",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(f"JOB#{job_id}"),
                    ":sk": _to_dynamo(f"EVT#{since_seq:020d}"),
                },
                ScanIndexForward=True,  # ascending seq
            ):
                for item in page.get("Items", []):
                    # Skip non-event items (SEQ counter, any future non-EVT# items)
                    # before decoding item["event"] to avoid KeyError
                    sk_value = item.get("SK", {}).get("S", "")
                    if not sk_value.startswith("EVT#"):
                        continue
                    results.append(_from_dynamo(item["event"]))
            return results

        return await asyncio.to_thread(_query)

    # ----- Turn event log -----------------------------------------------------

    async def allocate_turn_seq(self, turn_id: TurnId) -> int:
        """Atomically allocate the next event seq for a turn using DynamoDB ADD counter.

        Uses ``UpdateItem ADD seq :one`` with ``ReturnValues=UPDATED_NEW`` on
        ``PK=TURN#{turn_id}, SK=SEQ``. Each caller gets a distinct, strictly
        increasing value (DynamoDB atomic counter semantics).

        Thread-safety: Cross-process safe by design (DynamoDB atomic counter).
        """

        def _add() -> int:
            resp = self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(f"TURN#{turn_id}"),
                    "SK": _to_dynamo("SEQ"),
                },
                UpdateExpression="ADD seq :one",
                ExpressionAttributeValues={":one": _to_dynamo(1)},
                ReturnValues="UPDATED_NEW",
            )
            return int(_from_dynamo(resp["Attributes"]["seq"]))

        return await asyncio.to_thread(_add)

    async def append_turn_event(
        self, turn_id: TurnId, seq: int, event: dict[str, Any], *, ttl: int
    ) -> None:
        """Append one durable turn event item.

        Item shape: ``PK=TURN#{turn_id}, SK=EVT#{seq:020d}, event=<dict>, ttl=<int>``.
        Zero-padding ``seq`` to 20 digits makes string-SK ordering == numeric ordering.
        """
        item = {
            "PK": f"TURN#{turn_id}",
            "SK": f"EVT#{seq:020d}",
            "event": event,
            "ttl": ttl,
        }

        def _put() -> None:
            self.client.put_item(
                TableName=self.table, Item={k: _to_dynamo(v) for k, v in item.items()}
            )

        await asyncio.to_thread(_put)

    async def turn_events_since(
        self, turn_id: TurnId, since_seq: int
    ) -> list[dict[str, Any]]:
        """Query ordered turn event suffix from the base table (no GSI).

        ``KeyConditionExpression="PK = :pk AND SK > :sk"``, ``ScanIndexForward=True``.
        Zero-padded SK ensures lexicographic ordering == numeric seq ordering.

        The TURN#{turn_id} partition co-locates the atomic-seq counter item (SK="SEQ")
        with event items (SK="EVT#..."). Because "SEQ" sorts lexicographically AFTER
        any "EVT#" key, an unfiltered SK > :sk query always returns the counter item.
        This method skips non-EVT# items (including the SEQ counter) before decoding
        the event attribute, so the caller receives only event items.
        """

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            for page in paginator.paginate(
                TableName=self.table,
                KeyConditionExpression="PK = :pk AND SK > :sk",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(f"TURN#{turn_id}"),
                    ":sk": _to_dynamo(f"EVT#{since_seq:020d}"),
                },
                ScanIndexForward=True,  # ascending seq
            ):
                for item in page.get("Items", []):
                    # Skip non-event items (SEQ counter, any future non-EVT# items)
                    # before decoding item["event"] to avoid KeyError
                    sk_value = item.get("SK", {}).get("S", "")
                    if not sk_value.startswith("EVT#"):
                        continue
                    results.append(_from_dynamo(item["event"]))
            return results

        return await asyncio.to_thread(_query)

    # ----- Project event seq ---------------------------------------------------

    async def allocate_project_seq(self, project_id: ProjectId) -> int:
        """Atomically allocate the next event seq for a project stream.

        Uses ``UpdateItem ADD seq :one`` with ``ReturnValues=UPDATED_NEW`` on
        ``PK=PROJECT_SEQ#{project_id}, SK=SEQ``. Each caller gets a distinct,
        strictly increasing value (DynamoDB atomic counter semantics).

        Dedicated partition key (not under USER#{uid}) makes collision with
        project/job/turn records structurally impossible.

        Thread-safety: Cross-process safe by design (DynamoDB atomic counter).
        """

        def _add() -> int:
            resp = self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._project_seq_pk(project_id)),
                    "SK": _to_dynamo("SEQ"),
                },
                UpdateExpression="ADD seq :one",
                ExpressionAttributeValues={":one": _to_dynamo(1)},
                ReturnValues="UPDATED_NEW",
            )
            return int(_from_dynamo(resp["Attributes"]["seq"]))

        return await asyncio.to_thread(_add)

    async def put_turn_metadata(
        self, turn_id: TurnId, *, user_id: UserId, project_id: ProjectId, ttl: int
    ) -> None:
        """Store turn metadata for owner resolution (turn: replay owner-guard).

        Item shape: ``PK=TURN#{turn_id}, SK=META, user_id, project_id, ttl``.
        """
        item = {
            "PK": f"TURN#{turn_id}",
            "SK": "META",
            "user_id": user_id,
            "project_id": project_id,
            "ttl": ttl,
        }

        def _put() -> None:
            self.client.put_item(
                TableName=self.table, Item={k: _to_dynamo(v) for k, v in item.items()}
            )

        await asyncio.to_thread(_put)

    async def get_turn_metadata(self, turn_id: TurnId) -> dict[str, Any] | None:
        """Retrieve turn metadata for owner resolution.

        Returns {user_id, project_id, ttl} or None if not found.
        """

        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(f"TURN#{turn_id}"),
                    "SK": _to_dynamo("META"),
                },
            )
            item = resp.get("Item")
            if not item:
                return None
            return {
                "user_id": _from_dynamo(item.get("user_id")),
                "project_id": _from_dynamo(item.get("project_id")),
                "ttl": _from_dynamo(item.get("ttl")),
            }

        return await asyncio.to_thread(_get)

    # ----- Connection registry ------------------------------------

    async def put_connection(
        self,
        connection_id: str,
        user_id: UserId,
        *,
        project_id: ProjectId | None = None,
        ttl: int,
    ) -> None:
        """Persist connection with GSI1 keys for by-user lookup.

        Item shape: ``PK=CONN#{cid}, SK=META, user_id, project_id?, ttl,
        GSI1PK=CONNUSER#{uid}, GSI1SK=CONN#{cid}``. The GSI1 keys are what
        :meth:`list_connections_for_user` queries.
        """
        item = {
            "PK": f"CONN#{connection_id}",
            "SK": "META",
            "user_id": user_id,
            "ttl": ttl,
            "GSI1PK": f"CONNUSER#{user_id}",
            "GSI1SK": f"CONN#{connection_id}",
        }
        if project_id is not None:
            item["project_id"] = project_id

        def _put() -> None:
            self.client.put_item(
                TableName=self.table, Item={k: _to_dynamo(v) for k, v in item.items()}
            )

        await asyncio.to_thread(_put)

    async def delete_connection(self, connection_id: str) -> None:
        """Remove connection record (no-op if missing)."""

        def _delete() -> None:
            self.client.delete_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(f"CONN#{connection_id}"),
                    "SK": _to_dynamo("META"),
                },
            )

        await asyncio.to_thread(_delete)

    async def list_connections_for_user(
        self, user_id: UserId
    ) -> list[dict[str, Any]]:
        """Query GSI1 (``GSI1PK=CONNUSER#{uid}``) for a user's connections."""

        def _query() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self.client.get_paginator("query")
            for page in paginator.paginate(
                TableName=self.table,
                IndexName="GSI1",
                KeyConditionExpression="GSI1PK = :pk",
                ExpressionAttributeValues={
                    ":pk": _to_dynamo(f"CONNUSER#{user_id}"),
                },
            ):
                for item in page.get("Items", []):
                    results.append(_item_to_dict(item))
            return results

        return await asyncio.to_thread(_query)

    # ----- Cost tracking -----------------------------------------------------

    async def add_bedrock_cost(
        self,
        user_id: UserId,
        *,
        year_month: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        def _add() -> None:
            self.client.update_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._cost_sk(year_month)),
                },
                UpdateExpression=(
                    "ADD bedrock_input_tokens :it, "
                    "bedrock_output_tokens :ot, "
                    "total_cost_usd :c"
                ),
                ExpressionAttributeValues={
                    ":it": _to_dynamo(int(input_tokens)),
                    ":ot": _to_dynamo(int(output_tokens)),
                    ":c": _to_dynamo(float(cost_usd)),
                },
            )

        await asyncio.to_thread(_add)

    async def get_monthly_costs(
        self, user_id: UserId, year_month: str
    ) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            resp = self.client.get_item(
                TableName=self.table,
                Key={
                    "PK": _to_dynamo(self._user_pk(user_id)),
                    "SK": _to_dynamo(self._cost_sk(year_month)),
                },
            )
            item = resp.get("Item")
            return _item_to_dict(item) if item else None

        return await asyncio.to_thread(_get)


__all__ = ["DynamoDBMetadataStore"]
