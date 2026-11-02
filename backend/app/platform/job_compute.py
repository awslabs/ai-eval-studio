# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JobCompute Protocol — abstraction over 'where to run a job remotely'.

In cloud mode (the durable-state path; see ``resolve_arch_split``), eval and
optimization jobs run in isolated Fargate ``RunTask`` workers (right-sized,
independently surviving compute) behind this seam. The worker bootstraps
from minimal env vars, pulls an S3 snapshot of the project working tree, and
runs under per-tenant STS-scoped credentials it mints for itself.

In dev mode, this returns ``LocalSubprocessCompute`` — jobs run in-process
via ``asyncio.create_task``, so tests and dev need no AWS dependencies.

The seam decouples 'run a job' from 'where it runs', selected once by
config/flag at dispatch entry and threaded down the call chain. Never
re-read the flag mid-operation (dual-writer corruption risk).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .types import JobId, ProjectId, UserId


class MissingWorkerLaunchConfig(RuntimeError):
    """Worker-launch environment variables are missing.

    Raised by ``get_job_compute()`` cloud branch when one or more of the
    required worker-launch variables (ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN,
    WORKER_SUBNET_IDS, WORKER_SECURITY_GROUP_ID, S3_SNAPSHOTS_BUCKET, PROJECTS_DIR)
    are absent from the process environment. The message names ALL missing
    variables (sorted, comma-joined) so an operator sees the full gap in one error.

    The exception is raised (not returned) so it reaches the agent as a clear
    ``mcp_error`` via the existing post-claim try/except in ``dispatch_eval``,
    and so the eval-lock terminal-release behavior stays intact. Non-dispatch
    callers (reconciler, harness cancel, run-output materialization, optimization
    executor) already guard with ``isinstance(compute, FargateTaskCompute)`` or
    run inside try/except, so a raise there degrades rather than crashing an
    agent loop.

    Tradeoff: raising means the check executes at every ``get_job_compute()``
    call site (8 today), but the alternative (a separate pre-flight return at
    dispatch only) would duplicate the check and could drift. The dispatch path's
    existing try/except converts any exception into ``mcp_error``, so the agent
    observes the failure with zero new plumbing.
    """


# Required worker-launch environment variables. Must be present in the
# process environment for ``get_job_compute()`` cloud mode to succeed.
# Parsed by the infra guard test to assert parity with AgentCore runtime env.
_REQUIRED_WORKER_LAUNCH_ENV = (
    "ECS_CLUSTER_ARN",
    "WORKER_TASK_DEFINITION_ARN",
    "WORKER_SUBNET_IDS",
    "WORKER_SECURITY_GROUP_ID",
    "S3_SNAPSHOTS_BUCKET",
    "PROJECTS_DIR",
)


@dataclass(frozen=True)
class JobComputeHandle:
    """Opaque handle to a running job (Fargate task ARN or local PID).

    Returned by :meth:`JobCompute.launch`, stored on the job record as
    ``compute_handle``, and passed to :meth:`JobCompute.cancel` /
    :meth:`JobCompute.status`.

    The ``kind`` discriminator distinguishes Fargate (``"fargate_task"``)
    from local (``"local_pid"``) handles for debugging. The ``value`` is
    the actual task ARN or stringified PID.

    ``str(handle)`` returns the ``value``, so callers can persist
    ``compute_handle=str(handle)`` on the job record.
    """

    kind: Literal["fargate_task", "local_pid"]
    value: str

    def __str__(self) -> str:
        """Return the handle value for persisting as compute_handle."""
        return self.value


@runtime_checkable
class JobCompute(Protocol):
    """Abstraction over 'how to run a job remotely'.

    The cloud implementation (``FargateTaskCompute``) snapshots the
    project working tree to S3 and launches an isolated Fargate ``RunTask``
    worker. The worker mints its own per-tenant STS-scoped credentials,
    pulls the snapshot, runs the harness, and reports progress to the
    durable event log.

    Dev implementation (``LocalSubprocessCompute``) spawns the job
    in-process via ``asyncio.create_task``, reusing the existing harness
    runner without snapshotting (the local working tree already IS the
    tree). Cancel is process-group SIGTERM→SIGKILL.

    Selection: ``get_job_compute()`` returns the impl for the current mode.
    Both dispatch sites (standalone eval + optimization iteration) call
    ``launch`` instead of ``asyncio.create_task`` when ``resolve_arch_split``
    returns True (cloud).
    """

    async def launch(
        self,
        job_id: JobId,
        owner_id: UserId,
        project_id: ProjectId,
    ) -> JobComputeHandle:
        """Launch a job and return a handle for cancel/status.

        For ``FargateTaskCompute`` (cloud): snapshots the working tree to S3
        under the job-derived prefix and calls ``ecs.run_task`` with
        job_id/owner_id/project_id in the container override environment (the
        worker mints its own credentials). Returns a handle wrapping the task
        ARN.

        For ``LocalSubprocessCompute`` (dev): returns a handle wrapping the
        job_id (no-op, as dev is cloud-gated off this path). The job execution
        happens on the old in-process path (asyncio.create_task in the dispatch
        site) when arch_split is False (forced in dev by resolve_arch_split).

        Errors propagate to the caller (dispatch site wraps them in
        ``mcp_error`` so the agent reacts).
        """
        ...

    async def cancel(self, handle: JobComputeHandle) -> None:
        """Cancel a running job (StopTask for Fargate, SIGTERM for local).

        For ``FargateTaskCompute``: calls ``ecs.stop_task`` on the task ARN.
        The worker receives SIGTERM, has 30s to cleanup before SIGKILL.

        For ``LocalSubprocessCompute``: reads the job's current PID from the
        ``JobTracker`` and calls ``harness_runner.kill_harness_subprocess``
        (process-group SIGTERM→SIGKILL). If no PID yet, cancels the asyncio
        task.
        """
        ...

    async def status(self, handle: JobComputeHandle) -> str:
        """Return the job's current state.

        For ``FargateTaskCompute``: calls ``ecs.describe_tasks`` on the task
        ARN and returns the lastStatus (RUNNING, STOPPED, etc.).

        For ``LocalSubprocessCompute``: looks up the job in the ``JobTracker``
        and returns its current state string.
        """
        ...


def get_job_compute() -> JobCompute:
    """Return the JobCompute impl for the current platform mode.

    Cloud mode (``EVAL_STUDIO_MODE=cloud``): returns ``FargateTaskCompute``.
    The import is lazy so dev/test never pull
    AWS deps.

    Dev/test mode: returns ``LocalSubprocessCompute`` from ``dev_local``.

    This selector is called ONCE per dispatch operation, and the resolved
    impl is threaded down. Never re-read the mode flag mid-operation —
    that reintroduces dual-writer corruption risk.
    """
    import os

    mode = os.environ.get("EVAL_STUDIO_MODE", "dev").lower()

    if mode == "cloud":
        # Lazy import so dev/test never hit AWS deps
        from app.platform.aws.fargate_task_compute import FargateTaskCompute
        from app.platform import get_platform
        import boto3
        from botocore.config import Config

        # Read launch-input env vars (written by compute_stack.py cloud_env
        # and agentcore_stack.py cloud_env). Check ALL required keys upfront
        # and fail loud with the complete missing-variable list if any are absent,
        # rather than a bare ``KeyError: 'ECS_CLUSTER_ARN'`` on the first one.
        missing = sorted([k for k in _REQUIRED_WORKER_LAUNCH_ENV if k not in os.environ])
        if missing:
            raise MissingWorkerLaunchConfig(
                f"This process is missing worker-launch configuration and cannot "
                f"launch eval workers. Missing environment variables: {', '.join(missing)}. "
                f"These variables must be supplied by the deploying stack's container "
                f"or runtime environment configuration."
            )

        cluster_arn = os.environ["ECS_CLUSTER_ARN"]
        task_definition_arn = os.environ["WORKER_TASK_DEFINITION_ARN"]
        subnet_ids = os.environ["WORKER_SUBNET_IDS"].split(",")
        security_group_id = os.environ["WORKER_SECURITY_GROUP_ID"]
        snapshot_bucket = os.environ["S3_SNAPSHOTS_BUCKET"]
        projects_dir = os.environ["PROJECTS_DIR"]

        # Construct boto3 clients (mirror build_cloud_platform style)
        region = os.environ.get("AWS_REGION", "us-east-1")
        boto_session = boto3.session.Session(region_name=region)
        ecs_client = boto_session.client("ecs")
        s3_client = boto_session.client(
            "s3", config=Config(max_pool_connections=20)
        )

        # Get platform token_vending
        platform = get_platform()

        return FargateTaskCompute(
            ecs_client=ecs_client,
            cluster_arn=cluster_arn,
            task_definition_arn=task_definition_arn,
            subnet_ids=subnet_ids,
            security_group_id=security_group_id,
            token_vending=platform.token_vending,
            s3_client=s3_client,
            snapshot_bucket=snapshot_bucket,
            projects_dir=projects_dir,
        )
    else:
        from .dev_local import LocalSubprocessCompute

        return LocalSubprocessCompute()


__all__ = [
    "JobCompute",
    "JobComputeHandle",
    "MissingWorkerLaunchConfig",
    "get_job_compute",
    "_REQUIRED_WORKER_LAUNCH_ENV",
]
