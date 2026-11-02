# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore stack: runtime for per-user session isolation.

Provisions an AWS::BedrockAgentCore::Runtime for the orchestrator,
built from the ARM64 image (``backend/Dockerfile.agentcore``). This is an ADDITIVE
stack — it synthesizes alongside the existing Fargate coordinator without modifying
``compute_stack.py``. The runtime is configured with:

- **HTTP protocol** (``serverProtocol=HTTP``) so the AgentCore data-plane
  runtime speaks plain HTTP to the container on port 8080.
- **Cognito JWT authorizer** (reuses the existing user pool, config-only) so the
  browser can call the runtime ARN directly with a bearer token from the SPA's
  OIDC flow. The authorizer validates ``discoveryUrl`` +
  ``allowedClients=[SPA_CLIENT_ID]`` at the runtime edge.
- **Declarative EFS mount** at ``/mnt/projects`` via the access-point config
  (no manual NFS mount, no root). The access point squashes to UID/GID 1000,
  matching the non-root ``app`` user in the runtime Dockerfile.
- **VPC mode** with a PRIVATE_WITH_EGRESS subnet and a security group allowing
  egress (Bedrock, NAT) + NFS (2049) to the EFS mount target SG. The deployed
  subnet MUST be in a supported AZ (use1-az1/az2/az4, NOT az6).
- **Raised idle timeout** (e.g. 3600s) so long streamed turns are not idle-killed
  mid-stream.

**Managed session storage caveat:** The runtime's managed session storage is wiped
on every version update. Durable per-tenant state (sessions, transcripts, working
trees) MUST live on EFS or S3 only — never rely on the managed session storage
for persistence.

Args:
    vpc: VPC from ``NetworkStack``. Runtime needs private subnets with NAT egress.
    efs_access_point: EFS access point from ``StorageStack`` (ProjectsAp,
        uid/gid 1000, mountPath ``/mnt/projects``). The runtime's declarative
        EFS mount references this to mount at ``/mnt/projects`` without root.
    sessions_bucket: S3 bucket from ``StorageStack`` (session transcripts).
    uploads_staging_bucket: S3 bucket from ``StorageStack`` (browser uploads).
    snapshots_bucket: S3 bucket from ``StorageStack`` (job working-tree snapshots).
    table: DynamoDB table from ``StorageStack`` (metadata single-table).
    scoped_session_role: STS-assumable role from ``IamStack`` for per-request
        narrowed credentials.
    runtime_role: dedicated AgentCore Runtime execution role from ``IamStack``
        (trusted by ``bedrock-agentcore.amazonaws.com``; covers ECR pull,
        CloudWatch Logs, and the coordinator turn's data-plane access).
    user_pool: Cognito User Pool from ``AuthStack`` (for JWT discovery URL).
    spa_client: SPA Cognito user-pool client from ``AuthStack`` (allowedClients).
    worker_launch_env: The four worker-launch environment variables from
        ``ComputeStack`` (ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN,
        WORKER_SUBNET_IDS, WORKER_SECURITY_GROUP_ID). The AgentCore runtime
        runs the same coordinator turn code, so an agent-initiated dispatch_eval
        executes FargateTaskCompute in-process and reads these to build the
        RunTask call. The values are CloudFormation cross-stack imports from
        the compute stack, not literals.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Annotations,
    CfnOutput,
    Stack,
    aws_bedrockagentcore as bedrockagentcore,
    aws_cognito as cognito,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_efs as efs,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class AgentCoreStack(Stack):
    """AgentCore Runtime for per-user session isolation.

    This is an ADDITIVE stack — it synthesizes a separate
    ``AWS::BedrockAgentCore::Runtime`` without modifying the existing Fargate
    coordinator in ``ComputeStack``. The runtime serves the ARM64 orchestrator
    image (``backend/Dockerfile.agentcore``) with VPC mode, Cognito JWT
    authorizer, declarative EFS mount, HTTP protocol, and raised idle timeout.

    Attributes:
        runtime_arn: CloudFormation output for the runtime ARN.
            ``dev-postdeploy.sh`` reads it and passes it to the compute stack,
            which puts it in the coordinator's environment for the relay.
        runtime_id: CloudFormation output for the runtime id.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        efs_access_point: efs.IAccessPoint,
        sessions_bucket: s3.IBucket,
        uploads_staging_bucket: s3.IBucket,
        snapshots_bucket: s3.IBucket,
        table: dynamodb.ITable,
        scoped_session_role: iam.IRole,
        runtime_role: iam.IRole,
        user_pool: cognito.IUserPool,
        spa_client: cognito.IUserPoolClient,
        worker_launch_env: dict[str, str],
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- ARM64 image for AgentCore runtime ---------------------------
        # Build from ``backend/Dockerfile.agentcore`` and push to CDK's asset
        # ECR during ``cdk deploy``. Platform pinned to LINUX_ARM64 (AgentCore
        # Runtime requirement). The image serves ``app.agentcore_runtime:app``
        # on port 8080 (not 8100).
        backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
        image = ecr_assets.DockerImageAsset(
            self,
            "AgentCoreImage",
            directory=str(backend_dir),
            file="Dockerfile.agentcore",
            platform=ecr_assets.Platform.LINUX_ARM64,
        )

        # ---- Cloud-mode environment for the AgentCore runtime container ---
        # Same environment keys the coordinator needs (per
        # ``backend/app/platform/aws/__init__.py``'s ``build_cloud_platform``).
        # Discovery URL is built from the user pool id + region. AGENTCORE_RUNTIME_ARN
        # is not set here: the runtime does not need its own ARN, and only the
        # coordinator's relay uses it.
        #
        # The worker-launch vars (from ComputeStack) are merged into cloud_env
        # BEFORE the CfnRuntime is constructed so they land in the runtime's
        # environment_variables. The microVM runs the same coordinator turn code
        # as the Fargate service, so an agent-initiated dispatch_eval executes
        # FargateTaskCompute in-process there and reads these to build the
        # RunTask call. The values are CloudFormation cross-stack imports
        # (Fn::ImportValue / Fn::GetAtt / Fn::Join structures), not literals.
        cognito_discovery_url = (
            f"https://cognito-idp.{self.region}.amazonaws.com/"
            f"{user_pool.user_pool_id}/.well-known/openid-configuration"
        )
        cloud_env: dict[str, str] = {
            "EVAL_STUDIO_MODE": "cloud",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": self.region,
            "PROJECTS_DIR": "/mnt/projects",
            "UV_CACHE_DIR": "/mnt/projects/.uv-cache",
            "S3_SESSIONS_BUCKET": sessions_bucket.bucket_name,
            "S3_UPLOADS_STAGING_BUCKET": uploads_staging_bucket.bucket_name,
            "S3_SNAPSHOTS_BUCKET": snapshots_bucket.bucket_name,
            "DDB_TABLE": table.table_name,
            "SCOPED_SESSION_ROLE_ARN": scoped_session_role.role_arn,
            "COGNITO_REGION": self.region,
            "COGNITO_USER_POOL_ID": user_pool.user_pool_id,
            "COGNITO_SPA_CLIENT_ID": spa_client.user_pool_client_id,
            # Process-role marker: The microVM runs the same coordinator turn code
            # as the Fargate service, so the process needs a way to know it is not
            # the process holding browser connections. Job lifecycle ownership (progress
            # pollers, completion turns, report dispatch) stays on the coordinator.
            "EVAL_STUDIO_PROCESS_ROLE": "agentcore_runtime",
            # Unified observability opt-in: runtimes created before 2026-07-20
            # split telemetry between aws/spans and the per-agent log group.
            # This flag routes traces/prompts into the single per-agent group
            # (/aws/bedrock-agentcore/runtimes/<agent_id>-<endpoint>). Inert
            # until an ADOT collector ships in the image, but correct now so
            # observability work lands against one log group, not two.
            "UNIFIED_TRACES_DESTINATION_ENABLED": "true",
            **worker_launch_env,
        }

        # ---- Security group for the runtime (all outbound) ---------------
        # allow_all_outbound already covers outbound HTTPS (443) for Bedrock +
        # NAT and NFS (2049) to the EFS mount targets, so no explicit egress
        # rule is needed. An added NFS egress rule is DEAD under
        # allow_all_outbound=True (CDK ignores it + warns). The EFS mount-target
        # SG already allows 2049 from the VPC CIDR, so reachability is covered
        # on the ingress side.
        runtime_sg = ec2.SecurityGroup(
            self,
            "RuntimeSg",
            vpc=vpc,
            description="AgentCore Runtime security group (all outbound)",
            allow_all_outbound=True,
        )

        # ---- Subnet selection (EXPLICIT, supported-AZ-ID only) -----------
        # AgentCore only runs in a subset of AZs (by AZ-ID: use1-az1 / use1-az2
        # / use1-az4, NOT use1-az6) and rejects the WHOLE create request if any
        # supplied subnet is in an unsupported AZ. We must NOT pass the full
        # PRIVATE_WITH_EGRESS pool: CDK auto-assigns a created VPC's subnets to
        # the account's first AZs, which can land on an unsupported AZ-ID (in
        # some accounts us-east-1a maps to use1-az6). AZ *names* don't map 1:1 to
        # AZ-*IDs* across accounts and a created VPC's subnet->AZ-ID isn't
        # cleanly resolvable at synth, so the supported subnet IDs are supplied
        # explicitly at deploy time via CDK context and passed verbatim:
        #   cdk deploy ai-eval-studio-agentcore \
        #     -c agentcore_subnet_ids=subnet-aaa,subnet-bbb
        # A single subnet is fine here; the comma-separated form leaves multi-AZ
        # HA open later.
        #
        # WARN-and-omit rather than raise: the value cannot be known until the
        # VPC exists, and this check runs at SYNTH time, so raising would abort
        # ``cdk deploy --all`` on a fresh account before any stack deployed.
        # Same shape as network_stack's cloudfront_vpc_origin_sg_id, for the same
        # reason: discovered post-deploy, then this stack is redeployed.
        raw_subnet_ids = self.node.try_get_context("agentcore_subnet_ids")
        agentcore_subnet_ids = [
            s.strip() for s in str(raw_subnet_ids or "").split(",") if s.strip()
        ]
        if not agentcore_subnet_ids:
            Annotations.of(self).add_warning(
                "No 'agentcore_subnet_ids' in CDK context, so this stack creates "
                "NO AgentCore runtime and the relay path has no invoke target. "
                "Run infra/scripts/dev-postdeploy.sh (it resolves a "
                "PRIVATE_WITH_EGRESS subnet in an AgentCore-supported AZ-ID from "
                "this account's VPC and redeploys this stack) or pass "
                "-c agentcore_subnet_ids=subnet-xxxx. Expected on a first deploy, "
                "when the VPC whose subnets this needs does not exist yet."
            )
            return

        # ---- CfnRuntime: AgentCore Runtime resource -----------------------
        # Synthesizes an AWS::BedrockAgentCore::Runtime (CDK L1) with:
        #   - Container artifact: ARM64 ECR image (port 8080 in Dockerfile CMD)
        #   - serverProtocol: HTTP
        #   - VPC mode: private subnets + runtime SG
        #   - Cognito JWT authorizer: discoveryUrl + allowedClients=[SPA]
        #   - Declarative EFS mount: access-point at /mnt/projects
        #   - Raised idle timeout: 3600s
        #   - Cloud env vars: same as coordinator
        runtime = bedrockagentcore.CfnRuntime(
            self,
            "Runtime",
            agent_runtime_name=f"ai_eval_studio_{self.region.replace('-', '_')}",
            agent_runtime_artifact=bedrockagentcore.CfnRuntime.AgentRuntimeArtifactProperty(
                container_configuration=bedrockagentcore.CfnRuntime.ContainerConfigurationProperty(
                    container_uri=image.image_uri,
                ),
            ),
            protocol_configuration="HTTP",
            network_configuration=bedrockagentcore.CfnRuntime.NetworkConfigurationProperty(
                network_mode="VPC",
                network_mode_config=bedrockagentcore.CfnRuntime.VpcConfigProperty(
                    # EXPLICIT supported-AZ-ID subnets from context — NOT the full
                    # PRIVATE_WITH_EGRESS pool (which can include the unsupported
                    # use1-az6 and would make AgentCore reject the create).
                    subnets=agentcore_subnet_ids,
                    security_groups=[runtime_sg.security_group_id],
                ),
            ),
            role_arn=runtime_role.role_arn,
            authorizer_configuration=bedrockagentcore.CfnRuntime.AuthorizerConfigurationProperty(
                custom_jwt_authorizer=bedrockagentcore.CfnRuntime.CustomJWTAuthorizerConfigurationProperty(
                    discovery_url=cognito_discovery_url,
                    allowed_clients=[spa_client.user_pool_client_id],
                ),
            ),
            # Forward the Authorization header to the container. The JWT
            # authorizer validates + CONSUMES the Authorization header at the
            # runtime edge and does NOT pass it through by default, so the
            # in-container auth (which performs the tenant->project
            # require_access check the edge authorizer does NOT do) sees no
            # bearer token and fail-closes with 401 (reason=missing-bearer).
            # Allowlisting "Authorization" propagates the token to the
            # container for the defense-in-depth + per-project authz check.
            request_header_configuration=bedrockagentcore.CfnRuntime.RequestHeaderConfigurationProperty(
                request_header_allowlist=["Authorization"],
            ),
            filesystem_configurations=[
                bedrockagentcore.CfnRuntime.FilesystemConfigurationProperty(
                    efs_access_point=bedrockagentcore.CfnRuntime.EfsAccessPointConfigurationProperty(
                        access_point_arn=efs_access_point.access_point_arn,
                        mount_path="/mnt/projects",
                    ),
                ),
            ],
            lifecycle_configuration=bedrockagentcore.CfnRuntime.LifecycleConfigurationProperty(
                idle_runtime_session_timeout=3600,  # 1 hour
                max_lifetime=14400,  # 4 hours max ceiling
            ),
            environment_variables=cloud_env,
        )

        # ---- CloudFormation outputs --------------------------------------
        # Emit the runtime ARN + id so post-deploy wiring (dev-postdeploy.sh
        # passes the ARN to the compute stack) can reference them after deploy.
        self.runtime_arn = CfnOutput(
            self,
            "RuntimeArn",
            value=runtime.attr_agent_runtime_arn,
            description="AgentCore Runtime ARN (read by dev-postdeploy.sh for the coordinator)",
        )
        self.runtime_id = CfnOutput(
            self,
            "RuntimeId",
            value=runtime.attr_agent_runtime_id,
            description="AgentCore Runtime id",
        )
        # Echo the chosen subnet IDs so the operator can eyeball them (are they
        # in a supported AZ-ID?) before/after deploy.
        self.runtime_subnets = CfnOutput(
            self,
            "RuntimeSubnetIds",
            value=",".join(agentcore_subnet_ids),
            description=(
                "Subnet IDs passed to the AgentCore Runtime (must all be in "
                "supported AZ-IDs: use1-az1 / use1-az2 / use1-az4)"
            ),
        )
