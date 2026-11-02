# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FargateTaskCompute — isolated ECS RunTask worker backend.

Launches eval + optimization jobs in isolated Fargate workers behind the
``JobCompute`` seam. At dispatch, snapshots the exact project working tree
to S3 and calls ``ecs.run_task`` with minimal env vars (job_id + owner_id +
project_id + scoped-session role ARN). The worker pulls the snapshot, writes
an AWS config whose ``credential_process`` invokes a mint helper that assumes
the scoped-session role with session tags, and reports progress to the
durable event log.

Credential-refresh mechanism: the worker self-mints refreshable credentials
through a ``credential_process`` chain that survives the harness subprocess
boundary. The mint helper assumes the scoped-session role FROM the worker's
task role, which is **role chaining** — AWS hard-caps role-chained sessions at
3600s (1 hour) per mint. A ``DurationSeconds`` > 3600 makes ``assume_role``
FAIL; raising the scoped role's ``MaxSessionDuration`` does NOT lift the
chaining cap. Therefore each individual mint is capped at 3600s, BUT the
credential_process chain refreshes every ~900s (botocore's advisory window),
so an eval of any wall-clock duration stays authorized — no single mint has to
cover the whole run. Each ``launch`` bootstraps its own credential_process
config, and each optimization iteration is its own launch → own config, so a
multi-iteration loop is unaffected.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import project_key_for_directory

from ..job_compute import JobComputeHandle
from ..token_vending import TokenVending
from ..types import JobId, ProjectId, UserId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from botocore.client import BaseClient

logger = logging.getLogger(__name__)

#: Reported by :meth:`FargateTaskCompute.status` when DescribeTasks came back with
#: no task at all.
#:
#: Deliberately not one of ECS's own ``lastStatus`` values: it is not a state the
#: task is in, it is the statement that this call did not establish one. A caller
#: deciding whether a worker stopped must not read it as ``STOPPED`` — an empty
#: response is what ordinary eventual consistency looks like immediately after
#: StopTask, and also what a task reaped an hour after it finished looks like.
TASK_STATUS_UNKNOWN = "UNKNOWN"


class FargateTaskCompute:
    """Cloud ``JobCompute`` adapter via ECS RunTask.

    Launch snapshots the working tree to S3 under the job-derived prefix and
    calls ``ecs.run_task`` with minimal env vars. The worker mints its own
    per-tenant STS-scoped credentials on boot, pulls the snapshot, runs the
    harness, and reports progress to the durable event log.

    Args:
        ecs_client: Low-level boto3 ECS client. Inject in tests.
        cluster_arn: Full ARN of the ECS cluster. No region/account hardcoded.
        task_definition_arn: Worker task definition for RunTask. Accepts either
            a full ``family:revision`` ARN or a bare family name; the deploy
            supplies the FAMILY, and ECS resolves a bare family to its latest
            ACTIVE revision. The family is used because this value is threaded to
            a second execution home as a CloudFormation export, and an export
            carrying a per-deploy revision cannot be updated once imported.
        subnet_ids: List of subnet IDs for awsvpcConfiguration.
        security_group_id: Security group ID for the worker tasks.
        token_vending: Platform TokenVending for minting scoped creds.
        s3_client: Low-level boto3 S3 client for snapshot upload.
        snapshot_bucket: S3 bucket name for snapshot storage.
        projects_dir: Base projects directory (e.g. ``/mnt/projects``).
    """

    def __init__(
        self,
        *,
        ecs_client: "BaseClient",
        cluster_arn: str,
        task_definition_arn: str,
        subnet_ids: list[str],
        security_group_id: str,
        token_vending: TokenVending,
        s3_client: "BaseClient",
        snapshot_bucket: str,
        projects_dir: str,
    ) -> None:
        self.ecs = ecs_client
        self.cluster_arn = cluster_arn
        self.task_definition_arn = task_definition_arn
        self.subnet_ids = subnet_ids
        self.security_group_id = security_group_id
        self.token_vending = token_vending
        self.s3 = s3_client
        self.snapshot_bucket = snapshot_bucket
        self.projects_dir = Path(projects_dir)

    async def launch(
        self,
        job_id: JobId,
        owner_id: UserId,
        project_id: ProjectId,
    ) -> JobComputeHandle:
        """Launch a Fargate RunTask worker and return task ARN handle.

        1. Resolves the worker's project cwd the same way the runtime resolves
           it (so ``project_key_for_directory(cwd)`` yields the tenant prefix
           the worker's session tags will be scoped to).
        2. Snapshots the working tree to S3 under that canonical tenant prefix
           (e.g. ``<prefix>/snapshots/{job_id}.tar.gz``).
        3. Calls ``ecs.run_task`` with container overrides: command override
           to ``python -m app.scripts.worker_main``, environment vars for
           job_id/owner_id/project_id + scoped-session role ARN (NOT minted
           creds — the worker mints its own refreshable chain on boot).
        4. Returns handle wrapping the full task ARN (never the task ID).

        The worker calls ``write_worker_credentials_config`` on boot to set up
        a ``credential_process`` chain that self-refreshes for the life of any
        eval run. No wall-clock duration ceiling.

        The RunTask call sets ``enableExecuteCommand=False`` (security hardening)
        and NEVER logs any credential material.

        Raises:
            Exception: If snapshot upload fails or ``run_task`` returns failures.
                Caller wraps in ``mcp_error`` so agent reacts.
        """
        # 1. Compute snapshot key via shared helper (launch + worker MUST use same key)
        from app.services.snapshot_key import snapshot_key_for_job

        snapshot_key = snapshot_key_for_job(job_id)

        # 2. Snapshot working tree to S3
        project_dir = self.projects_dir / "users" / owner_id / "projects" / project_id
        worker_project_cwd = f"/tmp/workspace/{job_id}"
        canonical_prefix = project_key_for_directory(worker_project_cwd)

        logger.info(
            "Snapshotting project to S3",
            extra={
                "job_id": job_id,
                "project_id": project_id,
                "snapshot_key": snapshot_key,
            },
        )

        # Upload snapshot via shared utility (dedupe with working_tree_snapshot)
        from app.services.working_tree_snapshot import upload_snapshot_to_s3

        await upload_snapshot_to_s3(project_dir, self.s3, self.snapshot_bucket, snapshot_key)

        # 3. RunTask with minimal env (job/owner/project + role ARN)
        # The worker mints its own refreshable credentials on boot via
        # credential_process, so no static credential triple is injected here.
        # NEVER log credential material — only log job_id/owner/project
        logger.info(
            "Launching Fargate worker",
            extra={
                "job_id": job_id,
                "owner_id": owner_id,
                "project_id": project_id,
            },
        )

        def _run_task() -> str:
            response = self.ecs.run_task(
                cluster=self.cluster_arn,
                taskDefinition=self.task_definition_arn,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": self.subnet_ids,
                        "securityGroups": [self.security_group_id],
                        "assignPublicIp": "DISABLED",
                    }
                },
                overrides={
                    "containerOverrides": [
                        {
                            "name": "coordinator",  # Container name from task def
                            "command": ["python", "-m", "app.scripts.worker_main"],
                            "environment": [
                                {"name": "JOB_ID", "value": job_id},
                                {"name": "OWNER_ID", "value": owner_id},
                                {"name": "PROJECT_ID", "value": project_id},
                            ],
                        }
                    ]
                },
                tags=[
                    {"key": "Owner", "value": owner_id},
                    {"key": "Project", "value": project_id},
                    {"key": "Job", "value": job_id},
                ],
                startedBy="eval-studio-controller",
                enableExecuteCommand=False,  # Security hardening
            )

            # Extract full task ARN (never the task ID)
            if response.get("failures"):
                failures = response["failures"]
                raise RuntimeError(f"RunTask failed: {failures}")

            tasks = response.get("tasks", [])
            if not tasks:
                raise RuntimeError("RunTask returned no tasks")

            task_arn = tasks[0]["taskArn"]
            return task_arn

        task_arn = await asyncio.to_thread(_run_task)

        return JobComputeHandle(kind="fargate_task", value=task_arn)

    async def cancel(self, handle: JobComputeHandle) -> None:
        """Cancel a running worker via ecs.stop_task.

        The worker receives SIGTERM, has 30s (``stopTimeout``) to cleanup before
        SIGKILL. The full task ARN is required.

        Args:
            handle: JobComputeHandle wrapping the full task ARN.
        """
        logger.info("Stopping Fargate worker", extra={"task_arn": handle.value})

        def _stop_task() -> None:
            self.ecs.stop_task(
                cluster=self.cluster_arn,
                task=handle.value,
                reason="User canceled evaluation",
            )

        await asyncio.to_thread(_stop_task)

    async def status(self, handle: JobComputeHandle) -> str:
        """Return the worker's current state via ecs.describe_tasks.

        Args:
            handle: JobComputeHandle wrapping the full task ARN.

        Returns:
            ECS ``lastStatus`` string (RUNNING, STOPPED, PROVISIONING, etc.), or
            :data:`TASK_STATUS_UNKNOWN` when the response carries no task.

        An empty response used to be reported as ``STOPPED``. That reads absence
        as termination, and the two are not the same thing: a caller polling right
        after StopTask sees an empty result from ordinary eventual consistency, so
        "missing" there confirmed a stop that had not happened and released the
        project's eval lock while the worker kept invoking models. Only a task that
        is actually present and actually reports ``STOPPED`` is evidence it
        stopped; anything else is the absence of evidence, and the caller decides
        what to do about that.
        """
        def _describe_tasks() -> str:
            response = self.ecs.describe_tasks(
                cluster=self.cluster_arn,
                tasks=[handle.value],
            )

            tasks = response.get("tasks", [])
            if not tasks:
                logger.info(
                    "DescribeTasks returned no task — state is unknown, not stopped",
                    extra={
                        "task_arn": handle.value,
                        "failures": response.get("failures", []),
                    },
                )
                return TASK_STATUS_UNKNOWN

            return tasks[0]["lastStatus"]

        return await asyncio.to_thread(_describe_tasks)

    async def list_running_workers(self) -> list[dict[str, Any]]:
        """List all running eval workers via ecs.list_tasks + describe_tasks.

        This is the ECS sprawl-detection seam the reconciler and the kill
        switch consume. It returns raw task dicts with `taskArn`,
        `lastStatus`, and `tags` so the reconciler can build a `Job`-tag → `job_id`
        map (single responsibility — the reconciler owns the correlation logic).

        The method queries `desiredStatus="RUNNING"`, which includes
        PROVISIONING/PENDING/ACTIVATING/RUNNING states — all non-terminal states.
        It does NOT also query STOPPED.

        Returns:
            List of task detail dicts. Each dict has keys: `taskArn`, `lastStatus`,
            `tags` (list of {key, value} dicts). Returns [] if no tasks.
        """
        def _list_and_describe() -> list[dict[str, Any]]:
            # Page list_tasks to get all task ARNs
            task_arns = []
            paginator = self.ecs.get_paginator("list_tasks")
            for page in paginator.paginate(
                cluster=self.cluster_arn,
                family="ai-eval-studio-worker",  # WORKER_TASK_DEFINITION_FAMILY in compute_stack.py
                desiredStatus="RUNNING",  # Includes PROVISIONING/PENDING/ACTIVATING/RUNNING
            ):
                task_arns.extend(page.get("taskArns", []))

            if not task_arns:
                return []

            # Describe tasks to get tags + lastStatus
            response = self.ecs.describe_tasks(
                cluster=self.cluster_arn,
                tasks=task_arns,
                include=["TAGS"],  # REQUIRED or tags absent from response
            )

            return response.get("tasks", [])

        return await asyncio.to_thread(_list_and_describe)

    async def cancel_by_tag(
        self, owner_id: UserId, project_id: ProjectId, job_id: JobId
    ) -> str | None:
        """StopTask the worker whose ECS ``Job`` tag matches ``job_id`` (tag-first cancel).

        The durable ``compute_handle`` can be null on an eval record: it is
        written only after RunTask returns, so a record can exist before its
        handle does, or keep a null one if that write is lost. ECS task tags
        (Owner/Project/Job) are set on ``run_task`` and are INDEPENDENT of the
        durable record, so they are the reliable addressing mechanism for a
        detached worker.

        ECS ``list_tasks`` cannot filter by tag server-side, so this lists the
        worker family (via :meth:`list_running_workers` — list-by-family +
        describe-with-TAGS) and matches the ``Job`` tag client-side. It
        defensively cross-checks the ``Owner`` and ``Project`` tags before
        stopping so a ``Job``-tag collision can never stop another tenant's
        task.

        Args:
            owner_id: Expected ``Owner`` tag value (tenant cross-check).
            project_id: Expected ``Project`` tag value (cross-check).
            job_id: The ``Job`` tag value to match.

        Returns:
            The ARN of the task StopTask was issued against, or ``None`` when no
            task carried this job's tag.

        The ARN, not a bool, because StopTask returning means ECS accepted the
        request and nothing more. Handing back a bool left the caller with no way
        to confirm the task actually reached ``STOPPED`` — so this path reported a
        stop that the direct-ARN path would have gone on to verify, and the two
        addressing modes gave the caller different guarantees for the same word.
        """
        tasks = await self.list_running_workers()
        for task in tasks:
            tags = {t["key"]: t["value"] for t in task.get("tags", [])}
            if tags.get("Job") != job_id:
                continue
            # Defensive tenant cross-check: never stop a task whose Owner/Project
            # tags don't also match (guards against a Job-tag collision).
            if tags.get("Owner") != owner_id or tags.get("Project") != project_id:
                logger.warning(
                    "Job tag matched but Owner/Project mismatch — not stopping",
                    extra={"job_id": job_id},
                )
                continue
            task_arn = task["taskArn"]
            await self.cancel(JobComputeHandle(kind="fargate_task", value=task_arn))
            logger.info(
                "Issued StopTask for worker found by tag",
                extra={"job_id": job_id, "task_arn": task_arn},
            )
            return task_arn

        logger.info("No running worker found by tag", extra={"job_id": job_id})
        return None


__all__ = ["FargateTaskCompute"]
