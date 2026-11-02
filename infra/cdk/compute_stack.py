# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute stack: the coordinator Fargate service.

- One container image running ``uvicorn``, built from ``backend/Dockerfile``
  and pushed to CDK's asset ECR during ``cdk deploy``.
- **Coordinator service:** ``desiredCount=1`` with
  ``minimumHealthyPercent=100, maximumPercent=200`` — a rolling deploy boots the
  replacement task and waits for its health checks before draining the old one,
  so there is no deploy outage. Two coordinator tasks therefore OVERLAP during a
  deploy, which the in-memory turn queue is not shared across; that cost is
  accepted deliberately and the trade-off is spelled out at the service
  definition below. Eval and optimization jobs run as asyncio tasks inside this
  service, and a loop owned by the outgoing task still dies with it.

The ALB lives in :class:`NetworkStack` so destroying compute
doesn't take the DNS name with it; we receive the target group here and
register the coordinator service against it.
"""

from __future__ import annotations

import os
from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_cognito as cognito,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_efs as efs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct

# Task-definition family for the eval worker.
#
# This is the value handed to other stacks and to RunTask, deliberately NOT the
# revisioned task-definition ARN. The ARN carries a revision number that changes
# on every compute deploy, and CloudFormation refuses to update an export while
# another stack imports it — exporting the ARN made every subsequent compute
# deploy fail with "Cannot update export ... as it is in use by". A family name is
# stable across revisions, so the export never changes.
#
# It is also the behaviour we want: RunTask resolves a bare family to its latest
# ACTIVE revision, so a worker launched after a deploy picks up the new revision
# without anything having to re-read a value.
#
# Module-level because the IAM RunTask scope in ``iam_stack.py`` matches
# ``task-definition/<family>:*``; the two must name the same family.
WORKER_TASK_DEFINITION_FAMILY = "ai-eval-studio-worker"


class ComputeStack(Stack):
    """The coordinator Fargate service (image built via CDK asset ECR).

    Args:
        vpc: VPC from ``NetworkStack``.
        coordinator_role: IAM task role from ``IamStack``.
        worker_role: IAM task role from ``IamStack`` for isolated worker
            tasks.
        worker_execution_role: IAM execution role from ``IamStack`` for
            the worker task def. Created explicitly in IamStack
            (not auto-created) so its ARN is known at PassRole-grant time.
        coordinator_target_group: Target group from ``NetworkStack`` —
            the coordinator Fargate service registers itself against
            this on each deploy. Owning the target group in the network
            stack keeps the ALB's forward destination stable when
            compute is destroyed and re-deployed.
        coordinator_log_group: Log group from ``ObservabilityStack``.
        sessions_bucket: S3 bucket from ``StorageStack`` — name flows
            into ``S3_SESSIONS_BUCKET`` env var on the coordinator container.
        uploads_staging_bucket: S3 bucket from ``StorageStack`` — name
            flows into ``S3_UPLOADS_STAGING_BUCKET`` env var.
        snapshots_bucket: S3 bucket from ``StorageStack`` — name flows
            into ``S3_SNAPSHOTS_BUCKET`` env var on both containers. The
            coordinator uploads working-tree snapshots; the worker downloads
            via STS-scoped credentials.
        table: DynamoDB table from ``StorageStack`` — name flows
            into ``DDB_TABLE`` env var on both containers.
        file_system: EFS file system from ``StorageStack`` — mounted at
            ``/mnt/projects`` on both task definitions.
        scoped_session_role: STS-assumable role from ``IamStack`` —
            ARN flows into ``SCOPED_SESSION_ROLE_ARN`` env var so
            ``STSTokenVending.mint()`` can assume it.
        user_pool: Cognito User Pool from ``AuthStack`` — pool id flows
            into ``COGNITO_USER_POOL_ID`` env var; the in-app JWT
            validator uses it to derive the issuer URL for the JWKS
            fetch (see ``app/platform/aws/auth_cognito_bearer.py``).
        spa_client: SPA Cognito user-pool client from ``AuthStack`` —
            id flows into ``COGNITO_SPA_CLIENT_ID`` env var, used both
            by the JWT validator (must match the access token's
            ``client_id`` claim) and by ``GET /api/config`` so the SPA
            knows which client to use for the OIDC PKCE flow.
        user_pool_domain: Cognito hosted UI domain from ``AuthStack`` —
            full base URL flows into ``COGNITO_DOMAIN`` env var, served
            to the SPA via ``GET /api/config``.

    Attributes:
        worker_launch_env: The four worker-launch environment variables
            (ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN, WORKER_SUBNET_IDS,
            WORKER_SECURITY_GROUP_ID) needed by FargateTaskCompute to launch
            eval workers. Exposed publicly so AgentCoreStack can consume these
            same values — the AgentCore runtime is a second execution home that
            runs the same dispatch code and needs the same launch inputs. The
            values are CloudFormation cross-stack imports when consumed by
            AgentCoreStack, not literals.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        coordinator_role: iam.IRole,
        worker_role: iam.IRole,
        worker_execution_role: iam.IRole,
        coordinator_target_group: elbv2.IApplicationTargetGroup,
        coordinator_log_group: logs.ILogGroup,
        sessions_bucket: s3.IBucket,
        uploads_staging_bucket: s3.IBucket,
        snapshots_bucket: s3.IBucket,
        table: dynamodb.ITable,
        file_system: efs.IFileSystem,
        efs_access_point: efs.IAccessPoint,
        scoped_session_role: iam.IRole,
        user_pool: cognito.IUserPool,
        spa_client: cognito.IUserPoolClient,
        user_pool_domain: cognito.IUserPoolDomain,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- ECS cluster + task definitions ------------------------------
        # ECS Exec sessions (``aws ecs execute-command``, enabled on the
        # coordinator service below) are recorded: each operator shell's input
        # and output lands in the coordinator log group, under its own stream,
        # encrypted with that group's customer-managed key. Without this
        # configuration ECS falls back to logging through the task's
        # ``awslogs`` settings, which the task role cannot write, so sessions
        # went unrecorded. Because the coordinator service enables ECS Exec,
        # CDK grants its task role the log writes this needs, on this group.
        cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
            execute_command_configuration=ecs.ExecuteCommandConfiguration(
                logging=ecs.ExecuteCommandLogging.OVERRIDE,
                log_configuration=ecs.ExecuteCommandLogConfiguration(
                    cloud_watch_log_group=coordinator_log_group,
                    cloud_watch_encryption_enabled=True,
                ),
            ),
        )

        # Image source: build from ``backend/Dockerfile`` and push to CDK's
        # asset ECR during ``cdk deploy``. One repeatable
        # ``cdk deploy --all`` produces a working deployment with no
        # separate docker-push step.
        #
        # Use an absolute path derived from this file's location instead
        # of ``../../backend``. CDK resolves ``directory`` relative to
        # whichever CWD ``cdk synth`` was invoked from, which is *not*
        # always the cdk.json directory — ``cdk synth`` from ``infra/``
        # would otherwise look for a ``backend`` directory beside the repo
        # and fail with CannotFindImageDirectory. ``__file__`` is stable.
        #
        # Platform pinned to LINUX_ARM64 to align the compute stack with the
        # ARM64-only AgentCore microVM the coordinator relays to. Keeping all three
        # (coordinator, worker, AgentCore runtime) on one arch means the
        # compiled-dependency venvs the coordinator and worker share on the EFS
        # project tree (uv cache, .venv) are built for the same arch as AgentCore.
        # Also builds natively on Apple-Silicon (arm64) workstations.
        #
        # CDK shells out to ``docker`` for the build; ``CDK_DOCKER=podman``
        # (set by infra/scripts/dev-prep.sh) makes podman the engine
        # without changing the Dockerfile.
        backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
        image = ecs.ContainerImage.from_asset(
            directory=str(backend_dir),
            file="Dockerfile",
            platform=ecr_assets.Platform.LINUX_ARM64,
        )

        # Cloud-mode environment for the coordinator + worker containers.
        # ``backend/app/platform/aws/__init__.py``'s
        # ``build_cloud_platform()`` reads each of these via ``_require_env``
        # and crashes the boot if any is unset. Container-side defaults
        # match the names ``build_cloud_platform`` expects.
        #
        # Realtime runs as an in-process WS hub on the coordinator,
        # served at ``/ws`` through the same CloudFront → ALB path as
        # REST. The JWT validator pins the access-token ``client_id``
        # claim against ``COGNITO_SPA_CLIENT_ID``, and the SPA
        # bootstraps OIDC config from ``GET /api/config`` which sources
        # ``COGNITO_DOMAIN`` + ``LOGOUT_REDIRECT_URL``.
        cognito_domain = (
            f"https://{user_pool_domain.domain_name}"
            f".auth.{self.region}.amazoncognito.com"
        )
        # Logout redirect URL — the CloudFront DNS that Cognito sends
        # the user back to after the hosted-UI ``/logout`` flow.
        # Resolution order:
        #   1. ``logout_redirect_url`` CDK context — explicit override.
        #      ``dev-postdeploy.sh`` persists this to ``cdk.context.json``
        #      once CloudFront DNS is known so every subsequent
        #      ``cdk deploy`` picks it up without flags.
        #   2. ``cloudfront_dns`` CDK context — same persistence path;
        #      we derive the URL from the DNS so the operator only
        #      has to set one value.
        #   3. ``CLOUDFRONT_DNS`` env var — same shape as #2 but
        #      sourced from the shell.
        #   4. ``"/"`` so synth never fails on a fresh checkout. The
        #      SPA resolves ``/`` against the current origin at
        #      runtime, which works but is fragile if the SPA is
        #      ever served from a path prefix.
        _logout_ctx = self.node.try_get_context("logout_redirect_url")
        _cf_ctx = self.node.try_get_context("cloudfront_dns")
        _cf_env = os.environ.get("CLOUDFRONT_DNS")
        logout_redirect_url = (
            _logout_ctx
            or (f"https://{_cf_ctx}/" if _cf_ctx else None)
            or (f"https://{_cf_env}/" if _cf_env else None)
            or "/"
        )
        cloud_env: dict[str, str] = {
            "EVAL_STUDIO_MODE": "cloud",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": self.region,
            # EFS-mounted root for project working trees. Backend resolves
            # ``/mnt/projects/users/{owner_sub}/projects/{pid}/`` per-request.
            "PROJECTS_DIR": "/mnt/projects",
            # Persistent uv package cache on EFS. Without this, uv defaults
            # to $HOME/.cache/uv on the container's EPHEMERAL layer, so
            # every new/restarted Fargate task starts cold and the first
            # ``uv sync`` (validate_harness / eval run) pays the full
            # ~114s download+unpack. The harness dep set is pinned and
            # identical across all projects (deepeval pinned to the
            # backend's version), so one shared cache on EFS serves every
            # project and survives task restarts — warm after the first
            # sync ever. uv locks cache writes, so concurrent syncs across
            # tasks are safe to share it.
            "UV_CACHE_DIR": "/mnt/projects/.uv-cache",
            "S3_SESSIONS_BUCKET": sessions_bucket.bucket_name,
            "S3_UPLOADS_STAGING_BUCKET": uploads_staging_bucket.bucket_name,
            "S3_SNAPSHOTS_BUCKET": snapshots_bucket.bucket_name,
            "DDB_TABLE": table.table_name,
            "SCOPED_SESSION_ROLE_ARN": scoped_session_role.role_arn,
            "COGNITO_REGION": self.region,
            "COGNITO_USER_POOL_ID": user_pool.user_pool_id,
            "COGNITO_SPA_CLIENT_ID": spa_client.user_pool_client_id,
            "COGNITO_DOMAIN": cognito_domain,
            "LOGOUT_REDIRECT_URL": logout_redirect_url,
        }

        # Coordinator relays chat turns and completion turns to the deployed
        # AgentCore runtime server-side; the runtime ARN flows via CDK
        # context (not a cross-stack ref) so the AgentCore stack stays
        # additive and independently destroyable. Only the relay reads it; the
        # public GET /api/config does not serve it.
        agentcore_runtime_arn = self.node.try_get_context("agentcore_runtime_arn")
        if agentcore_runtime_arn:
            cloud_env["AGENTCORE_RUNTIME_ARN"] = agentcore_runtime_arn

        # ---- EFS volume + mount points ----------------------------------
        # Both task defs mount the same file system at /mnt/projects via
        # an EFS access point that chroots to ``/projects`` and squashes
        # all NFS ops to UID/GID 1000 (matching ``USER app`` in the
        # runtime Dockerfile). Containers therefore run non-root and
        # still have read+write on the mount.
        #
        # The bundled ``claude`` CLI in ``claude-agent-sdk`` refuses to
        # start with ``--dangerously-skip-permissions`` if
        # ``process.getuid() === 0`` (verified by reading the bundled
        # binary). Running as ``app`` is therefore load-bearing for the
        # SDK, not just a hardening preference.
        #
        # Tenant isolation is enforced at the app layer
        # (``require_access``) plus the per-user path layout
        # (``users/{owner_sub}/projects/{pid}/``), for API routes and the
        # path-jailed file tools — not for the agent's ``Bash`` tool, which
        # reads the mount directly (see the README's Security scope). Note
        # EFS access points squash UID by design — every NFS op arrives as
        # UID 1000 regardless of the calling Linux user — so per-UID file
        # permissions are not a distinguishing mechanism at this layer.
        efs_volume_name = "projects"
        efs_mount_path = "/mnt/projects"

        def _add_efs_volume(task_def: ecs.FargateTaskDefinition) -> None:
            """Attach the EFS volume via the access point, over TLS and with
            IAM authorization.

            The file system's policy (``storage_stack.py``) refuses any
            client that is not using TLS or does not present an IAM identity,
            so both settings are required for the mount to succeed. With IAM
            authorization on, EFS evaluates the task role's
            ``elasticfilesystem:Client*`` grants: the worker's role holds no
            ``ClientWrite``, so its mount is read-only at the file system as
            well as by its mount point's ``read_only`` flag."""
            task_def.add_volume(
                name=efs_volume_name,
                efs_volume_configuration=ecs.EfsVolumeConfiguration(
                    file_system_id=file_system.file_system_id,
                    transit_encryption="ENABLED",
                    authorization_config=ecs.AuthorizationConfig(
                        access_point_id=efs_access_point.access_point_id,
                        iam="ENABLED",
                    ),
                ),
            )

        # ---- Explicit coordinator security group to break the cycle ------
        # The coordinator service needs a security group, and the worker-launch
        # vars (including WORKER_SECURITY_GROUP_ID) must be populated into
        # cloud_env BEFORE the coordinator container is created (CDK snapshots
        # environment at add_container time). Creating an explicit SG breaks
        # the dependency cycle that previously forced the wrong order.
        #
        # This SG sets allow_all_outbound=True to preserve NAT egress to
        # Bedrock, bedrock-mantle.{region}.api.aws (mantle raw-HTTPS + SigV4),
        # and knowledge-mcp.global.api.aws (live model-card lookups).
        # If egress is ever tightened to an allowlist, BOTH hosts MUST be
        # permitted or those calls fail closed in prod.
        coordinator_sg = ec2.SecurityGroup(
            self,
            "CoordinatorServiceSg",
            vpc=vpc,
            allow_all_outbound=True,
            description="Coordinator service security group (allow-all outbound for Bedrock + mantle + knowledge-mcp)",
        )
        worker_security_group_id = coordinator_sg.security_group_id

        # ---- Private subnet IDs (PRIVATE_WITH_EGRESS) --------------------
        private_subnet_ids = ",".join(
            [s.subnet_id for s in vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ).subnets]
        )

        # ---- Worker task definition (isolated job compute) --------------
        # Reuses the same coordinator image (one image), same EFS
        # mount, same sizing (CPU=2048/mem=16384 to match the coordinator,
        # whose 4 GB ran out of memory on 500-case runs; matching avoids
        # drift and is worth re-tuning against real load).
        #
        # The worker task def has NO static command — the command is
        # overridden per-launch by RunTask (FargateTaskCompute),
        # which sets it to `["python", "-m", "app.scripts.worker_main"]`.
        # The container name matches the coordinator's ("coordinator") so
        # RunTask container-overrides can target it by name (the RunTask
        # override references the container name "coordinator").
        #
        # Worker task role (from IamStack) holds NO ambient S3 or DynamoDB
        # grant and no Cognito/KMS-admin. The worker reaches S3 and DynamoDB
        # only through the scoped-session credential it mints for itself on
        # boot (credential_process), whose statements are
        # ${aws:PrincipalTag/*}-templated.
        #
        # The /mnt/projects mount is present for parity/optionality and is
        # read-only — the worker executes exclusively against the snapshot
        # unpacked to /tmp/workspace, so it can never mutate the shared tree.
        # Isolation boundary = read-only mount + /tmp snapshot.
        #
        # family="ai-eval-studio-worker" so the generated task-definition name
        # begins with the prefix the RunTask IAM resource scope expects
        # (arn:aws:ecs:...:task-definition/ai-eval-studio-worker:*) in iam_stack.py.
        # execution_role=worker_execution_role (explicit, not auto-created) so
        # PassRole can cover both the task role and the execution role.
        #
        # runtime_platform pins Fargate placement to Graviton to match the
        # ARM64 image asset and the ARM64-only AgentCore microVM.
        worker_task_def = ecs.FargateTaskDefinition(
            self,
            "WorkerTaskDef",
            family=WORKER_TASK_DEFINITION_FAMILY,
            cpu=2048,
            memory_limit_mib=16384,
            task_role=worker_role,
            execution_role=worker_execution_role,
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.ARM64,
            ),
        )
        _add_efs_volume(worker_task_def)

        # ---- Build worker-launch vars for cross-stack consumption ---------
        # The 4 launch vars are needed by ANY execution home that runs
        # dispatch_eval / FargateTaskCompute in cloud mode (coordinator Fargate
        # service AND AgentCore Runtime). Both read them in ``get_job_compute()``
        # (backend/app/platform/job_compute.py) to construct a RunTask call.
        # The worker task ITSELF reads NONE of them — it receives
        # JOB_ID/OWNER_ID/PROJECT_ID via RunTask overrides and uses only
        # base cloud_env (S3_SNAPSHOTS_BUCKET, AWS_REGION, etc.).
        #
        # Exposed as self.worker_launch_env so AgentCoreStack can consume them
        # (the AgentCore runtime is a second execution home that runs the same
        # coordinator turn code and must hold these to dispatch evals).
        #
        # Coordinator env must be assembled BEFORE the coordinator
        # add_container call — CDK snapshots a container's environment at
        # add_container time, so post-add mutations are not reflected.
        #
        # The worker deliberately omits WORKER_TASK_DEFINITION_ARN to avoid
        # a self-referential CloudFormation resource (WorkerTaskDef depends
        # on WorkerTaskDef), which passes synth but fails at changeset
        # creation with a circular dependency error.
        #
        # WORKER_TASK_DEFINITION_ARN carries the task-definition FAMILY, not the
        # revisioned ARN. Every value in this dict is consumed cross-stack as a
        # CloudFormation export, and an export cannot be updated while another
        # stack imports it — so a value that changes on each deploy permanently
        # bricks deploys of this stack. The revisioned ARN is exactly such a
        # value; the family is stable. RunTask accepts a family and resolves the
        # latest ACTIVE revision, which is what we want post-deploy anyway.
        #
        # Every value here must be deploy-stable for the same reason. The other
        # three are: a cluster ARN (name-derived), subnet ids, and a security
        # group id — none carry a revision.
        self.worker_launch_env = {
            "ECS_CLUSTER_ARN": cluster.cluster_arn,
            "WORKER_TASK_DEFINITION_ARN": WORKER_TASK_DEFINITION_FAMILY,
            "WORKER_SUBNET_IDS": private_subnet_ids,
            "WORKER_SECURITY_GROUP_ID": worker_security_group_id,
        }
        coordinator_env = {**cloud_env, **self.worker_launch_env}

        # ---- Coordinator task definition — 2 vCPU / 16 GB ----------------
        # 4 GB runs out of memory on 500-case eval runs (run-data + session
        # materialization spike), hence 16 GB. CPU stays at 2048 (valid
        # Fargate combo with 16 GB; the coordinator is I/O-bound — no need to
        # scale CPU).
        coordinator_task_def = ecs.FargateTaskDefinition(
            self,
            "CoordinatorTaskDef",
            cpu=2048,
            memory_limit_mib=16384,
            task_role=coordinator_role,
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.ARM64,
            ),
        )
        _add_efs_volume(coordinator_task_def)
        coordinator_container = coordinator_task_def.add_container(
            "coordinator",
            image=image,
            # Container runs as ``app`` (Dockerfile default — UID 1000).
            # No ``user="0:0"`` override; see EFS volume comment above
            # for the access-point UID-squash that makes this work.
            # ``--ws-max-size`` caps each INBOUND WebSocket message; outbound
            # envelopes are unaffected. The ``/ws`` route authenticates on the
            # first frame, after the upgrade, so it buffers a whole message from
            # a caller it has not authenticated yet and parses it as JSON — up to
            # 16 MiB at uvicorn's default, and a JSON array that size expands to
            # far more as Python objects. Everything a client legitimately sends
            # is a JWT or a short ``subscribe``, so 64 KiB is generous.
            # ``backend/Dockerfile`` carries the same value.
            #
            # ``app.scripts.serve_tls`` serves uvicorn over TLS with a certificate
            # it generates per task, because the ALB reaches this container over
            # HTTPS (the target group in ``network_stack.py``).
            command=[
                "python",
                "-m",
                "app.scripts.serve_tls",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8100",
                "--ws-max-size",
                "65536",
            ],
            port_mappings=[ecs.PortMapping(container_port=8100)],
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="coordinator", log_group=coordinator_log_group
            ),
            environment=coordinator_env,
        )
        coordinator_container.add_mount_points(
            ecs.MountPoint(
                container_path=efs_mount_path,
                source_volume=efs_volume_name,
                read_only=False,
            )
        )

        # ---- Add worker container (base env only, no launch vars) --------
        # Worker uses base cloud_env WITHOUT the 4 worker-launch vars.
        # RunTask overrides supply JOB_ID/OWNER_ID/PROJECT_ID at launch.
        # Omitting WORKER_TASK_DEFINITION_ARN avoids a self-referential
        # CloudFormation cycle (verified: worker reads none of the 4 vars).
        #
        # Worker mounts EFS read-only, and uv run (invoked even under skip_sync=True)
        # may write cache index/wheel metadata. Redirecting the worker's cache to
        # writable ephemeral /tmp avoids EACCES on the RO mount. The coordinator keeps
        # the shared warm cache on the RW EFS mount.
        worker_env = {**cloud_env, "UV_CACHE_DIR": "/tmp/uv-cache"}
        worker_container = worker_task_def.add_container(
            "coordinator",  # same name as coordinator container
            image=image,  # reuses the shared image asset
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="worker",
                log_group=coordinator_log_group,  # reuse log group
            ),
            environment=worker_env,  # base env + worker UV cache override
            # An init process as PID 1, so a cancel's StopTask actually stops the
            # worker. The RunTask command runs Python directly, which makes it
            # PID 1, and Linux delivers a signal sent from outside the container
            # to PID 1 only if PID 1 installed a handler for it. The worker
            # installs none, so ECS's SIGTERM was dropped and every cancelled
            # worker ran on until the SIGKILL at stopTimeout. Under the init
            # process Python is an ordinary child: SIGTERM ends it, the init
            # exits, and the kernel kills the harness processes left in the
            # container. The coordinator needs none of this — uvicorn handles
            # SIGTERM itself.
            linux_parameters=ecs.LinuxParameters(
                self, "WorkerLinuxParameters", init_process_enabled=True
            ),
        )
        worker_container.add_mount_points(
            ecs.MountPoint(
                container_path=efs_mount_path,
                source_volume=efs_volume_name,
                read_only=True,  # Worker mounts EFS read-only; isolation boundary is the read-only mount + /tmp snapshot
            )
        )

        # ---- Coordinator Fargate Service ---------------------------------
        # ``min_healthy_percent=100, max_healthy_percent=200`` — rolling
        # deploys start the replacement task and wait for its health checks
        # to pass before draining the old one. Removes the deploy outage
        # (the ~30s stop-before-start gap is gone).
        #
        # Accepted consequence: TWO coordinator tasks overlap during a
        # deploy. The coordinator's turn queue is in-memory and not shared
        # across tasks.
        # This overlap is accepted deliberately in exchange for removing
        # the deploy outage.
        #
        # Scope limit: this removes DOWNTIME, not mid-flight loop continuity.
        # A loop owned by the old task still dies when that task stops.
        # Graceful drain-and-resume (SIGTERM → checkpoint → new task's
        # reconciler adopts) is separate, larger work and is explicitly NOT
        # attempted here.
        #
        # Outage profile:
        #   * Healthy deploy: no user-facing downtime (new task boots while
        #     old task serves).
        #   * Bad image / failed health checks: ~20s detection (2 × 10s
        #     unhealthy checks) → circuit breaker rollback. Previously 15–20 min;
        #     now fail-fast to ~20s.
        self.coordinator_service = ecs.FargateService(
            self,
            "CoordinatorService",
            cluster=cluster,
            task_definition=coordinator_task_def,
            desired_count=1,
            min_healthy_percent=100,
            max_healthy_percent=200,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(60),
            assign_public_ip=False,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[coordinator_sg],
            # ECS Exec — lets operators ``aws ecs execute-command`` into
            # a running task to probe environment / DNS / network /
            # process state without redeploying. The matching IAM
            # permissions (``ssmmessages:*``) are granted on the
            # task role in ``iam_stack.py``. Setting this on the
            # service triggers a task replacement on next deploy
            # because the task agent reads the flag at startup.
            enable_execute_command=True,
        )

        # ---- Bind coordinator service to the network-stack target group --
        # The ALB + listener + target-group construct itself live in
        # NetworkStack; we just register the dynamic targets
        # (Fargate task IPs) here. This is the only ALB-touching code path
        # the compute stack still owns.
        self.coordinator_service.attach_to_application_target_group(
            coordinator_target_group
        )

        # ---- EFS network reachability -----------------------------------
        # The EFS access SG (in StorageStack) already allows NFS from the
        # VPC CIDR, which covers our private-subnet Fargate tasks. A
        # more-specific rule here (Fargate SG → EFS access SG) would create a
        # stack-dependency cycle, since compute already depends on storage.

        # ---- Control-plane outputs for FargateTaskCompute ---------------
        # Surface the launch inputs the runtime needs (cluster ARN, worker
        # task-definition ARN, subnets, SG) — no hardcoded ARNs/region.
        CfnOutput(
            self,
            "ClusterArn",
            value=cluster.cluster_arn,
            export_name="ai-eval-studio-cluster-arn",
            description="ECS cluster ARN for worker RunTask launches",
        )
        # Family, not the revisioned ARN. A named export carrying a per-deploy
        # value is a latent trap even with no importer today: the moment anything
        # imports it, this stack can never be updated again. The family is stable
        # and is equally RunTask-launchable.
        CfnOutput(
            self,
            "WorkerTaskDefinitionFamily",
            value=WORKER_TASK_DEFINITION_FAMILY,
            export_name="ai-eval-studio-worker-task-definition-family",
            description=(
                "Worker task-definition family (RunTask launchable; resolves to "
                "the latest ACTIVE revision)"
            ),
        )
        # private_subnet_ids and worker_security_group_id are already computed
        # above (before the worker task def) and added to cloud_env. Reuse them here.
        CfnOutput(
            self,
            "WorkerSubnetIds",
            value=private_subnet_ids,
            export_name="ai-eval-studio-worker-subnet-ids",
            description="Private subnet IDs for worker awsvpcConfiguration (comma-joined)",
        )
        CfnOutput(
            self,
            "WorkerSecurityGroupId",
            value=worker_security_group_id,
            export_name="ai-eval-studio-worker-security-group-id",
            description="Security group ID for worker awsvpcConfiguration",
        )
