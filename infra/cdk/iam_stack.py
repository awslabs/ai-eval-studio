# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IAM stack: task roles, the AgentCore runtime role, and the scoped-session role.

Five roles are created here. The worker execution role and the AgentCore
runtime role are described where they are built; the other three:

- **coordinator-task-role** — assumed by the Fargate coordinator. Reads/writes
  the sessions / uploads-staging S3 buckets, mounts EFS at
  ``/mnt/projects``, full DynamoDB on ``ai-eval-studio-metadata``,
  KMS decrypt on the bucket keys, ``bedrock:InvokeModel*``, and
  ``sts:AssumeRole`` + ``sts:TagSession`` on the scoped-session role for
  per-request credential narrowing.
- **scoped-session-role** — assumed via STS by the coordinator to mint
  per-request tag-scoped credentials. Uses ABAC (Attribute-Based Access
  Control): the role's identity policy is fixed and fully tag-templated
  (``${aws:PrincipalTag/uid}``, ``${aws:PrincipalTag/proj}``,
  ``${aws:PrincipalTag/job}``). Per-tenant scoping comes from session tags
  passed on ``assume_role``, not an inline session policy, so the minted
  packed-binary budget stays small and FLAT regardless of action count. This
  eliminates the ``PackedPolicyTooLarge`` outage class.
- **worker-task-role** — assumed by the isolated Fargate eval workers. Holds
  the non-tenant surface only (read-only EFS mount, Bedrock, ECS Exec)
  plus ``sts:AssumeRole``/``sts:TagSession`` on the scoped-session role. It
  carries NO ambient S3-sessions or DynamoDB grant, because the worker
  container executes agent-authored harness code: an ambient grant there is
  reachable by that code and would supersede the ABAC bound.

Which roles get the cross-tenant S3 + DynamoDB data plane is the tenant
boundary in this stack, and it is decided by one flag —
``_attach_shared_policies(minted_data_plane_only=...)``. Control-plane roles
(coordinator, AgentCore runtime) hold it and enforce ownership in application
code; roles whose container runs agent-authored code do not hold it and go
through a minted scoped credential instead. ``infra/tests/test_worker_task_def.py``
asserts both halves.

EFS holds project working trees. The scoped-session role narrows the
sessions S3 access and DynamoDB ``LeadingKeys`` per user; EFS has no
IAM-layer ACL surface, so its isolation is the per-user path layout plus
the app-layer ``require_access`` chokepoint — which the agent's ``Bash``
tool does not pass through (see the README's Security scope).

Each role's ARN is exposed as a CfnOutput so ``compute_stack`` can wire them
into the coordinator Fargate task definition.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_dynamodb as dynamodb,
    aws_efs as efs,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

# The RunTask scope below must name the same task-definition family the worker is
# actually created with, so it is imported rather than restated. ``compute_stack``
# imports no local module, so this direction cannot cycle.
from compute_stack import WORKER_TASK_DEFINITION_FAMILY

# What S3 needs from a caller's identity to read and write SSE-KMS objects.
_BUCKET_KEY_ACTIONS = ["kms:Decrypt", "kms:GenerateDataKey"]

# The minimum DynamoDB documents for a principal using a table encrypted with a
# customer-managed key.
_TABLE_KEY_ACTIONS = [
    "kms:Encrypt",
    "kms:Decrypt",
    "kms:ReEncrypt*",
    "kms:GenerateDataKey*",
    "kms:DescribeKey",
    "kms:CreateGrant",
]


def _customer_managed_key_arn(resource_name: str, key) -> str:
    """Return ``key``'s ARN, refusing a store that has no customer-managed key.

    A store on an AWS-managed key would make the grant below name nothing, and
    the failure would surface at runtime as an AccessDenied naming KMS.
    """
    if key is None:
        raise ValueError(
            f"{resource_name} has no customer-managed encryption key; the roles "
            "here are granted use of that key and nothing else."
        )
    return key.key_arn


def _bucket_key_statement(region: str, *buckets: s3.IBucket) -> iam.PolicyStatement:
    """Allow use of the buckets' keys, only through S3 in ``region``."""
    key_arns = list(dict.fromkeys(
        _customer_managed_key_arn(bucket.node.id, bucket.encryption_key)
        for bucket in buckets
    ))
    return iam.PolicyStatement(
        actions=_BUCKET_KEY_ACTIONS,
        resources=key_arns,
        conditions={"StringEquals": {"kms:ViaService": f"s3.{region}.amazonaws.com"}},
    )


def _table_key_statement(region: str, table: dynamodb.ITable) -> iam.PolicyStatement:
    """Allow use of the table's key, only through DynamoDB in ``region``."""
    return iam.PolicyStatement(
        actions=_TABLE_KEY_ACTIONS,
        resources=[_customer_managed_key_arn(table.node.id, table.encryption_key)],
        conditions={
            "StringEquals": {"kms:ViaService": f"dynamodb.{region}.amazonaws.com"}
        },
    )


class IamStack(Stack):
    """Task roles, the AgentCore runtime role, and the scoped-session IAM role.

    Attributes:
        coordinator_task_role: ECS task role for the coordinator service.
        agentcore_runtime_role: Execution role for the AgentCore Runtime.
            Holds worker-launch parity with the coordinator (both execution
            homes run the same dispatch code and need identical ECS control-plane +
            PassRole + snapshots-bucket permissions).
        scoped_session_role: Role assumed via STS by the coordinator task
            role. ``STSTokenVending.mint()`` narrows each credential to one
            user / one project with session tags that resolve the role's
            ``${aws:PrincipalTag/*}`` templates. Also called the
            token-vending (TVM) target role.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        sessions_bucket: s3.IBucket,
        uploads_staging_bucket: s3.IBucket,
        snapshots_bucket: s3.IBucket,
        table: dynamodb.ITable,
        file_system: efs.IFileSystem,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ECS tasks always need this trust relationship.
        ecs_tasks_principal = iam.ServicePrincipal("ecs-tasks.amazonaws.com")

        # ---- Coordinator task role ---------------------------------------
        self.coordinator_task_role = iam.Role(
            self,
            "CoordinatorTaskRole",
            role_name="ai-eval-studio-coordinator-task-role",
            assumed_by=ecs_tasks_principal,
            description="ECS task role for the Eval Studio coordinator Fargate service.",
        )
        self._attach_shared_policies(
            role=self.coordinator_task_role,
            sessions_bucket=sessions_bucket,
            uploads_staging_bucket=uploads_staging_bucket,
            table=table,
            file_system=file_system,
        )
        # Coordinator uploads the working-tree snapshot to S3 at dispatch
        # (under the tenant's prefix: project_key_for_directory(cwd)/snapshots/{job_id}.tar.gz).
        # The worker downloads via minted STS creds derived from scoped_session_role.
        # s3:GetObject is required for the reverse-sync leg: a completed
        # detached eval worker uploads its output/ results to S3 (worker EFS
        # mount is read-only), and the coordinator downloads that object to EFS
        # at the shared detached-terminal point. The other three verbs cover the
        # inbound snapshot upload.
        self.coordinator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[
                    snapshots_bucket.bucket_arn,
                    f"{snapshots_bucket.bucket_arn}/*",
                ],
            )
        )
        self.coordinator_task_role.add_to_policy(
            _bucket_key_statement(self.region, snapshots_bucket)
        )
        # s3:ListBucket, although nothing here lists the bucket. S3 answers a
        # GetObject for a key that does not exist with 404 only to a caller that
        # may list the bucket, and with 403 AccessDenied to anyone else. The
        # reverse-sync read above routinely asks before the key exists: the
        # harness reports its results before the worker uploads them, and the
        # download retries until the upload lands. It retries quietly on a 404
        # and logs anything else, so without this every completed eval logged an
        # AccessDenied warning, and a real permission failure read the same as an
        # upload still in flight. A bucket-level action, so the bucket ARN alone.
        self.coordinator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[snapshots_bucket.bucket_arn],
            )
        )
        # Coordinator invokes AgentCore runtime server-side for the chat relay
        # and the completion-turn relay. This is the coordinator's capability to
        # invoke the runtime (a capability grant, not gated on the per-project relay
        # flag — the runtime-facing flag lives in app config, not IAM). This is
        # DISTINCT from self.agentcore_runtime_role, which is the runtime's OWN
        # execution role (trusted by bedrock-agentcore.amazonaws.com) and is not
        # the caller's invoke permission.
        self.coordinator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*",
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*/runtime-endpoint/*",
                ],
            )
        )

        # ---- Worker task role (isolated job compute) --------------------
        # The worker container executes agent-authored harness code by
        # design (run_eval.py imports src/config.py, which the scaffolder
        # deliberately never overwrites), so this role is a TENANT
        # BOUNDARY, not just a floor. It therefore carries NO ambient
        # S3-sessions or DynamoDB data-plane grant: every data-plane call
        # the worker makes resolves through the minted scoped-session
        # credential (credential_process → scoped_session_role, whose
        # statements are ${aws:PrincipalTag/*}-templated). See
        # minted_data_plane_only in _attach_shared_policies.
        #
        # Why an ambient grant cannot be a "harmless floor" here: any code
        # in the container can drop AWS_CONFIG_FILE from its environment
        # and fall through botocore's chain to the ECS credential endpoint
        # (mint_worker_credentials.py documents that exact technique for
        # its own recursion break). An ambient statement is therefore
        # reachable by harness code and silently supersedes the ABAC bound
        # that test_sessions_bucket_statements_gated_to_coordinator and
        # test_worker_role_has_no_ambient_s3_or_dynamodb assert
        # (infra/tests/test_worker_task_def.py).
        #
        # The only AWS call the worker makes on THIS role is
        # sts:AssumeRole + sts:TagSession on scoped_session_role (granted
        # below) — the one call that must precede the credential chain.
        #
        # Explicitly NO Cognito (cognito-idp:*) — workers must not reach
        # user pool admin; coordinator-only surface.
        # Explicitly NO KMS admin — workers don't manage keys.
        # NO DeleteBucket / DeleteTable — same reasoning as scoped_session_role.
        self.worker_task_role = iam.Role(
            self,
            "WorkerTaskRole",
            role_name="ai-eval-studio-worker-task-role",
            assumed_by=ecs_tasks_principal,
            description=(
                "ECS task role for isolated Fargate eval workers. Holds no "
                "ambient S3-sessions/DynamoDB data plane; per-tenant scoping "
                "via the minted scoped-session credential it self-mints on boot."
            ),
        )
        # Reuse the shared policy set for the parts that are NOT a tenant
        # boundary (EFS mount, Bedrock, region discovery, ECS Exec). No KMS:
        # the keys go with the data-plane grants it does not hold.
        # writable_efs=False keeps ClientWrite off this role. The ECS volumes
        # mount with IAM authorization (compute_stack.py), so EFS evaluates
        # that grant: a compromised worker running agent-influenced harness
        # code cannot write the shared tenant tree even if it remounts
        # read-write. Isolation boundary = read-only mount + /tmp snapshot.
        self._attach_shared_policies(
            role=self.worker_task_role,
            sessions_bucket=sessions_bucket,
            uploads_staging_bucket=uploads_staging_bucket,
            table=table,
            file_system=file_system,
            writable_efs=False,
            minted_data_plane_only=True,
        )

        # ---- Worker execution role --------------------------------------
        # Created explicitly in IamStack (not auto-created by the task def)
        # so its ARN is known at PassRole-grant time. The cyclic-dependency
        # problem: ComputeStack runs after IamStack, so if the execution role
        # were auto-created by the task def, its ARN wouldn't be known here
        # when we grant PassRole. Creating it here breaks the cycle.
        # Standard ECS execution-role managed policy (ECR pull + CloudWatch Logs).
        self.worker_execution_role = iam.Role(
            self,
            "WorkerExecutionRole",
            role_name="ai-eval-studio-worker-execution-role",
            assumed_by=ecs_tasks_principal,
            description=(
                "ECS execution role for isolated Fargate eval workers. "
                "Created in IamStack (not auto-created) so its ARN is "
                "known at PassRole-grant time."
            ),
        )
        self.worker_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSTaskExecutionRolePolicy"
            )
        )

        # ---- Worker-launch grant helper (coordinator + agentcore parity) --
        # Both the coordinator and agentcore runtime need worker-launch grants
        # (both execution homes run dispatch_eval / FargateTaskCompute). Extracting
        # the four statements into one helper prevents drift and ensures parity.
        def _attach_worker_launch_policies(*, role: iam.IRole) -> None:
            """Attach worker-launch ECS + IAM policies to the given role.

            Grants the four capabilities needed to dispatch eval workers via
            FargateTaskCompute: RunTask (scoped to worker task def), TagResource
            (tag-on-create for the worker task), Stop/Describe/List (cluster-scoped),
            and PassRole (scoped to exactly the worker task + execution role ARNs).

            Applied to both coordinator_task_role AND agentcore_runtime_role
            since both execution homes run the same dispatch code and must
            hold identical launch permissions.
            """
            # RunTask assigns the worker task/execution roles, which requires
            # the caller hold PassRole for exactly those role ARNs — do not
            # use resources=["*"] for PassRole.
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["ecs:RunTask"],
                    # Scope RunTask to the worker task-definition family:revision ARN
                    # (arn:aws:ecs:region:account:task-definition/ai-eval-studio-worker:*)
                    # RunTask authorizes against the family:revision pattern, so
                    # ai-eval-studio-worker-* would never match — use worker:* instead.
                    #
                    # The trailing ``:*`` is what lets a caller pass the bare family
                    # and have RunTask resolve the latest ACTIVE revision: whichever
                    # revision it picks, the resulting family:revision ARN matches.
                    resources=[
                        f"arn:aws:ecs:{Stack.of(self).region}:{Stack.of(self).account}"
                        f":task-definition/{WORKER_TASK_DEFINITION_FAMILY}:*",
                    ],
                )
            )
            # Tag-on-create: AWS-canonical form uses resources=["*"] gated by
            # ecs:CreateAction=RunTask condition. This ties TagResource authorization
            # to the RunTask call that creates the TASK, immune to short/long task-ARN
            # drift and needs no cluster-name ref.
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["ecs:TagResource"],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "ecs:CreateAction": "RunTask",
                        }
                    },
                )
            )
            # StopTask/DescribeTasks/ListTasks take task ARNs or operate on cluster, so scope by cluster condition
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecs:StopTask",
                        "ecs:DescribeTasks",
                        "ecs:ListTasks",
                    ],
                    resources=["*"],
                    conditions={
                        "ArnEquals": {
                            "ecs:cluster": f"arn:aws:ecs:{Stack.of(self).region}:{Stack.of(self).account}:cluster/*",
                        }
                    },
                )
            )
            # PassRole scoped to exactly the worker task + execution roles.
            # RunTask assigns both the task role and the execution role, so
            # PassRole must cover both ARNs (not just the task role).
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[
                        self.worker_task_role.role_arn,
                        self.worker_execution_role.role_arn,
                    ],
                )
            )

        # Apply worker-launch grants to coordinator
        _attach_worker_launch_policies(role=self.coordinator_task_role)

        # ---- AgentCore runtime role -------------------------------------
        # The AgentCore Runtime is assumed by the AgentCore SERVICE principal
        # (bedrock-agentcore.amazonaws.com), NOT ecs-tasks — reusing the
        # coordinator's ecs-tasks-trusted role would fail create->READY. Trust
        # principal + confused-deputy conditions (SourceAccount / SourceArn)
        # and the ECR-pull + CloudWatch-Logs execution permissions are the
        # AWS-documented AgentCore Runtime execution-role contract (verified
        # against docs.aws.amazon.com/bedrock-agentcore .../runtime-permissions).
        #
        # AgentCore uses ONE role (unlike ECS's task/execution split): this
        # single role covers both image-pull/logging AND the runtime's own
        # data-plane access. It therefore gets the same shared data-plane
        # policy set the coordinator turn needs (S3 / EFS / DDB / Bedrock),
        # since it runs the SAME coordinator turn code, plus sts:AssumeRole on
        # the scoped-session role for per-request credential narrowing.
        agentcore_principal = iam.ServicePrincipal(
            "bedrock-agentcore.amazonaws.com",
            conditions={
                "StringEquals": {"aws:SourceAccount": self.account},
                "ArnLike": {
                    "aws:SourceArn": (
                        f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:*"
                    )
                },
            },
        )
        self.agentcore_runtime_role = iam.Role(
            self,
            "AgentCoreRuntimeRole",
            role_name="ai-eval-studio-agentcore-runtime-role",
            assumed_by=agentcore_principal,
            description=(
                "Execution role for the AgentCore Runtime. "
                "Trusted by bedrock-agentcore.amazonaws.com; covers ECR pull, "
                "CloudWatch Logs, and the coordinator turn's data-plane access."
            ),
        )
        # Data-plane access — same shape as the coordinator (runs the same
        # turn code, mounts the shared EFS tree writable).
        #
        # This role keeps the cross-tenant grant (no minted_data_plane_only)
        # because it IS the control plane: the turn code it runs resolves
        # ``require_access``, which reads other users' project records by
        # definition, and it has no
        # credential_process chain to fall back on the way the worker does.
        # Narrowing this role means moving agent execution off the control
        # plane, or routing its data-plane calls through a minted credential.
        # Both are architectural changes, not policy tweaks — a condition here
        # cannot substitute for either.
        self._attach_shared_policies(
            role=self.agentcore_runtime_role,
            sessions_bucket=sessions_bucket,
            uploads_staging_bucket=uploads_staging_bucket,
            table=table,
            file_system=file_system,
        )
        # Per-request credential narrowing: the runtime assumes the
        # scoped-session role the same way the coordinator does.
        # (scoped_session_role is created below; the grant is added after it.)
        #
        # AgentCore execution-role permissions (AWS-documented contract):
        # pull the ARM64 image from the CDK asset ECR repo and write runtime
        # logs to the AgentCore log group.
        self.agentcore_runtime_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRImagePull",
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:GetAuthorizationToken",
                ],
                resources=["*"],
            )
        )
        self.agentcore_runtime_role.add_to_policy(
            iam.PolicyStatement(
                sid="AgentCoreRuntimeLogs",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/bedrock-agentcore/runtimes/*",
                ],
            )
        )
        # EFS filesystem discovery — AgentCore's CreateAgentRuntime validates
        # (at create time) that the execution role can DescribeAccessPoints +
        # DescribeMountTargets whenever a FilesystemConfigurations EFS mount is
        # attached. The shared data-plane policy only grants ClientMount/
        # ClientWrite/DescribeMountTargets scoped to the filesystem ARN;
        # DescribeAccessPoints has no filesystem-ARN resource form, so it must
        # be granted on "*" (read-only describe, low risk). Without this the
        # runtime create fails: "Execution role is missing required filesystem
        # permissions. Ensure the role has elasticfilesystem:DescribeAccessPoints
        # and elasticfilesystem:DescribeMountTargets".
        self.agentcore_runtime_role.add_to_policy(
            iam.PolicyStatement(
                sid="AgentCoreEfsDiscovery",
                actions=[
                    "elasticfilesystem:DescribeAccessPoints",
                    "elasticfilesystem:DescribeMountTargets",
                ],
                resources=["*"],
            )
        )

        # Apply worker-launch grants to agentcore runtime (same grants as coordinator).
        # The microVM runs the same coordinator turn code, so an agent-initiated
        # dispatch_eval executes FargateTaskCompute in-process and needs identical
        # ECS control-plane + PassRole permissions.
        _attach_worker_launch_policies(role=self.agentcore_runtime_role)

        # Snapshots-bucket grant (agentcore runtime, not via helper).
        # FargateTaskCompute.launch uploads the working-tree snapshot with the
        # TASK ROLE's own S3 client (self.s3, upload_snapshot_to_s3), BEFORE
        # minting scoped creds — the snapshot upload is NOT done with minted
        # credentials. So both execution homes need this grant. Applied separately
        # from the worker-launch helper for three reasons: (a) folding it would
        # move the coordinator's existing standalone snapshots-bucket statement
        # (granted above, next to the coordinator task role) into the helper and
        # change the coordinator's rendered policy shape for no functional gain;
        # (b) the coordinator will always hold at least one grant the runtime does
        # not need (bedrock-agentcore:InvokeAgentRuntime, granted to the
        # coordinator above), so perfect symmetry is unachievable anyway;
        # (c) ECS control-plane grants (RunTask/TagResource/PassRole/
        # Stop-Describe-List) and an S3 data-plane grant are separate concerns.
        self.agentcore_runtime_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[
                    snapshots_bucket.bucket_arn,
                    f"{snapshots_bucket.bucket_arn}/*",
                ],
            )
        )
        self.agentcore_runtime_role.add_to_policy(
            _bucket_key_statement(self.region, snapshots_bucket)
        )

        # ---- Scoped-session role -----------------------------------------
        # ABAC (Attribute-Based Access Control) for per-tenant scoping.
        # The role's identity policy is FIXED and fully tag-templated
        # (${aws:PrincipalTag/uid}, ${aws:PrincipalTag/proj}, ${aws:PrincipalTag/job}).
        # Per-tenant scoping comes from session tags passed on assume_role (not
        # an inline session policy), so the minted packed-binary budget stays
        # small and FLAT regardless of action count — eliminating the
        # PackedPolicyTooLarge failure an inline session policy hits as
        # actions are added.
        #
        # Session length: ``STSTokenVending.mint()`` assumes this role FROM the
        # coordinator's or worker's task-role credentials, and AWS hard-caps a
        # role-chained session at 1 hour: ``DurationSeconds > 3600`` makes the
        # call FAIL, and MaxSessionDuration does NOT lift it. 3600s is
        # comfortably above the DeepEval per-run gather budget (600s), and each
        # optimization iteration re-mints.
        # Trusted by the coordinator task role, the AgentCore runtime role, AND
        # the worker task role. The coordinator + runtime mint per-request creds
        # at dispatch; the worker self-mints refreshable creds on boot (so eval
        # runs exceeding 1 hour stay authorized instead of expiring mid-run).
        self.scoped_session_role = iam.Role(
            self,
            "ScopedSessionRole",
            role_name="ai-eval-studio-scoped-session-role",
            assumed_by=iam.CompositePrincipal(
                iam.ArnPrincipal(self.coordinator_task_role.role_arn),
                iam.ArnPrincipal(self.agentcore_runtime_role.role_arn),
                iam.ArnPrincipal(self.worker_task_role.role_arn),
            ),
            # Every mint is chained, which STS caps at 3600s whatever this says.
            max_session_duration=Duration.seconds(3600),
            description=(
                "Target role assumed via STS to mint tag-scoped credentials. "
                "Coordinator/runtime mint per-request at dispatch; worker self-mints "
                "refreshable creds on boot (credential_process chain). "
                "Uses ABAC: fixed tag-templated identity policy + session tags."
            ),
        )

        # Add sts:TagSession to the trust policy (required for session tags).
        # The Null conditions require the uid tag (all paths) and job tag
        # (worker-only) EXIST at assume time — a tagless assume can't start.
        # Conservative: applied to TagSession only.
        #
        # NOTE: the job-tag condition proves a tag was SUPPLIED, not that its
        # VALUE matches the launched job. There is no IAM condition variable
        # that binds an sts:TagSession request to the ECS task tags set at
        # run_task (those are resource tags on the task, not principal/context
        # keys available in an STS authorization). So this condition enforces
        # presence/format but cannot cross-bind the value.
        #
        # This trust policy is NOT the tenancy boundary, and must not be read
        # as one. IAM has no condition key that binds a session-tag VALUE to the
        # job a container was launched for, so tag PRESENCE is the most this
        # policy can require — do not add conditions here expecting them to pin a
        # tenant. Tenancy is enforced by the app-layer ``require_access`` check.
        #
        # Moving that boundary into IAM needs a validator outside the assuming
        # container's own trust boundary. The shape identified for it is
        # tags-in-JWT ABAC via AssumeRoleWithWebIdentity, where the tags ride in
        # a signed JWT claim and STS verifies the signature itself; that also
        # removes the 3600s role-chaining cap. Per-job roles and a hand-rolled
        # credential broker were the alternatives considered.
        self.scoped_session_role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                actions=["sts:TagSession"],
                principals=[
                    iam.ArnPrincipal(self.coordinator_task_role.role_arn),
                    iam.ArnPrincipal(self.agentcore_runtime_role.role_arn),
                ],
                effect=iam.Effect.ALLOW,
                conditions={
                    "Null": {"aws:RequestTag/uid": "false"}
                },
            )
        )
        # Worker TagSession statement: requires BOTH uid and job tags present.
        # The worker self-mints on boot, so it must hold TagSession. The job-tag
        # presence check is an additional constraint vs coordinator/runtime
        # (which use the proj tag, not job).
        self.scoped_session_role.assume_role_policy.add_statements(
            iam.PolicyStatement(
                actions=["sts:TagSession"],
                principals=[
                    iam.ArnPrincipal(self.worker_task_role.role_arn),
                ],
                effect=iam.Effect.ALLOW,
                conditions={
                    "Null": {
                        "aws:RequestTag/uid": "false",
                        "aws:RequestTag/job": "false",
                    }
                },
            )
        )

        # ABAC identity policy: tag-templated statements for S3, DynamoDB, Bedrock.
        # NO per-tenant literals — scoping is enforced by ${aws:PrincipalTag/*}
        # templates that resolve at authorization time from the session tags.

        # Sessions S3 (coordinator-only, inherently gated by missing proj tag on worker).
        # GetObject/PutObject/DeleteObject on the sessions bucket prefix (coordinator-only).
        # Explicitly gated ``Null: {aws:PrincipalTag/job: "true"}`` so the worker's
        # proj tag does NOT gain sessions-bucket access. Previously this was gated
        # incidentally by the worker lacking a proj tag; adding proj for PROJECT_SEQ#
        # authorization makes the gate explicit.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[
                    f"{sessions_bucket.bucket_arn}/-mnt-projects-users-${{aws:PrincipalTag/uid}}-projects-${{aws:PrincipalTag/proj}}/*",
                ],
                conditions={
                    "Null": {"aws:PrincipalTag/job": "true"}
                },
            )
        )
        # Sessions S3 ListBucket (coordinator-only).
        # Same explicit job-tag gate as above.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[sessions_bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": ["-mnt-projects-users-${aws:PrincipalTag/uid}-projects-${aws:PrincipalTag/proj}/*"]
                    },
                    "Null": {"aws:PrincipalTag/job": "true"}
                },
            )
        )

        # Snapshot S3 (worker-only, gated by Null condition).
        # GetObject downloads the inbound working-tree snapshot; PutObject
        # uploads the outbound eval output/ results for the reverse-sync leg
        # (worker EFS is read-only, so completed results go worker→S3→coordinator).
        # Both live under the SAME job-derived -tmp-workspace-${job}/* prefix
        # (project_key_for_directory("/tmp/workspace/{job}") = -tmp-workspace-{job}),
        # so the output key -tmp-workspace-{job}/output/{file} is inside this grant.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[
                    f"{snapshots_bucket.bucket_arn}/-tmp-workspace-${{aws:PrincipalTag/job}}/*",
                ],
                conditions={
                    "Null": {"aws:PrincipalTag/job": "false"}
                },
            )
        )

        # DynamoDB: item-level CRUD + queries (NO Scan).
        # LeadingKeys list includes USER#, JOB#, and PROJECT_SEQ# templates.
        # The worker needs PROJECT_SEQ# authorization to allocate a durable
        # project seq counter. Tag-resolved equality template prevents cross-tenant
        # writes; rejected alternative was StringLike PROJECT_SEQ#* (wildcard),
        # which would let any worker bump ANY tenant's counter. Unresolved tags
        # → resource matches nothing per AWS docs, so both paths safe.
        #
        # TABLE ONLY, no ``/index/*``. Nothing on the minted path queries a GSI —
        # the store's three ``IndexName=`` queries are reconciler/recovery
        # operations that run under the coordinator or AgentCore runtime role,
        # both of which keep the control-plane grant (the worker role itself is
        # attached with ``minted_data_plane_only=True`` and has none). Dropping
        # the unused half also sidesteps a question AWS docs do not settle:
        # whether ``dynamodb:LeadingKeys`` is evaluated at all on a GSI query.
        # If it is not, an index grant here would be a cross-tenant read; with
        # no index grant the question cannot arise.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                ],
                resources=[table.table_arn],
                conditions={
                    "ForAllValues:StringEquals": {
                        "dynamodb:LeadingKeys": [
                            "USER#${aws:PrincipalTag/uid}",
                            "JOB#${aws:PrincipalTag/job}",
                            "PROJECT_SEQ#${aws:PrincipalTag/proj}",
                        ]
                    }
                },
            )
        )

        # The keys behind the S3 and DynamoDB statements above. A credential
        # minted from this role holds no other identity, so without these every
        # transcript read and write in a turn, and the worker's snapshot
        # download and result upload, fails with AccessDenied. The key grants
        # carry no tag condition: what a minted credential may read is bounded
        # by the tag-templated statements above, and a key grant alone opens
        # no object or item.
        self.scoped_session_role.add_to_policy(
            _bucket_key_statement(self.region, sessions_bucket, snapshots_bucket)
        )
        self.scoped_session_role.add_to_policy(_table_key_statement(self.region, table))

        # Bedrock (worker-only, gated by Null condition).
        # InvokeModel on inference profiles and foundation models.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:*:{Stack.of(self).account}:inference-profile/*",
                    f"arn:aws:bedrock:*:{Stack.of(self).account}:application-inference-profile/*",
                    "arn:aws:bedrock:*::foundation-model/*",
                ],
                conditions={
                    "Null": {"aws:PrincipalTag/job": "false"}
                },
            )
        )

        # Bedrock Mantle (worker-only, gated by Null condition).
        # CreateInference on project/* resources.
        self.scoped_session_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-mantle:CreateInference"],
                resources=["arn:aws:bedrock-mantle:*:*:project/*"],
                conditions={
                    "Null": {"aws:PrincipalTag/job": "false"}
                },
            )
        )

        # Bedrock Mantle — AWS-managed policy for inference access.
        # A credential minted from this role can do no more than this role's own
        # policies allow: the chained ``STSTokenVending.mint()`` passes session
        # tags only. So mantle calls made on minted credentials (the worker's harness) are
        # denied unless this role carries the grant, while Converse models keep
        # working through the bedrock:InvokeModel statement above (Converse is
        # authorized as bedrock:InvokeModel).
        #
        # The same managed policy is also attached to the shared task roles
        # (coordinator/worker/agentcore) via _attach_shared_policies. All four
        # roles run mantle-capable code, so all four carry it. One
        # AWS-maintained source of truth.
        self.scoped_session_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonBedrockMantleInferenceAccess"
            )
        )

        # Allow the coordinator task role to assume the scoped-session role
        # AND pass session tags (sts:TagSession required for ABAC).
        self.coordinator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole", "sts:TagSession"],
                resources=[self.scoped_session_role.role_arn],
            )
        )
        # Same grant for the AgentCore runtime role — it mints per-request
        # tag-scoped creds via the same STS chaining path as the coordinator.
        self.agentcore_runtime_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole", "sts:TagSession"],
                resources=[self.scoped_session_role.role_arn],
            )
        )
        # Worker task role: self-mints refreshable credentials on boot (via
        # credential_process) so eval runs exceeding 1 hour stay authorized
        # instead of expiring mid-run. The worker assumes the scoped-session
        # role FROM its own task role, which is role chaining — the same path
        # the coordinator uses, just invoked by the worker itself instead of
        # being delivered as a static RunTask env triple. The scoped role's
        # ABAC identity statements are unchanged (same ${aws:PrincipalTag/*}
        # templates), so the effective permission set is byte-for-byte the same
        # scope — only the expiry moves (the credential refreshes instead of
        # freezing). Grant is scoped to the scoped-session role ARN alone; the
        # worker cannot assume any other role.
        self.worker_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole", "sts:TagSession"],
                resources=[self.scoped_session_role.role_arn],
            )
        )

        # ---- Outputs (consumed by compute_stack via env-var override) -----
        CfnOutput(
            self,
            "CoordinatorTaskRoleArn",
            value=self.coordinator_task_role.role_arn,
            export_name="ai-eval-studio-coordinator-task-role-arn",
        )
        CfnOutput(
            self,
            "ScopedSessionRoleArn",
            value=self.scoped_session_role.role_arn,
            export_name="ai-eval-studio-scoped-session-role-arn",
        )
        CfnOutput(
            self,
            "WorkerTaskRoleArn",
            value=self.worker_task_role.role_arn,
            export_name="ai-eval-studio-worker-task-role-arn",
            description="Worker task role ARN (least-privilege floor for isolated eval workers)",
        )
        CfnOutput(
            self,
            "WorkerExecutionRoleArn",
            value=self.worker_execution_role.role_arn,
            export_name="ai-eval-studio-worker-execution-role-arn",
            description="Worker execution role ARN (ECR pull + CloudWatch Logs)",
        )
        CfnOutput(
            self,
            "AgentCoreRuntimeRoleArn",
            value=self.agentcore_runtime_role.role_arn,
            export_name="ai-eval-studio-agentcore-runtime-role-arn",
            description=(
                "AgentCore Runtime execution role ARN "
                "(bedrock-agentcore trust; ECR pull + Logs + data-plane)"
            ),
        )

    @staticmethod
    def _attach_shared_policies(
        *,
        role: iam.Role,
        sessions_bucket: s3.IBucket,
        uploads_staging_bucket: s3.IBucket,
        table: dynamodb.ITable,
        file_system: efs.IFileSystem,
        writable_efs: bool = True,
        minted_data_plane_only: bool = False,
    ) -> None:
        """Apply the policy set shared by the coordinator, worker, and runtime roles.

        The role needs:
        - Read/write the sessions / uploads-staging buckets, and item-level
          CRUD + Query on the single metadata table. Both are CROSS-TENANT
          and are granted only to control-plane roles, which serve every
          user's requests and must resolve any owner's records
          (``require_access``, email sharing lookup, delete cascades).
          Roles whose container runs agent-authored code pass
          ``minted_data_plane_only=True`` and are skipped — see below.
        - Mount + read (+ write if writable_efs=True) on the EFS file system.
          Tenant isolation on EFS is enforced at the app layer, for API routes
          and the path-jailed file tools (not the agent's ``Bash`` tool). The
          worker receives writable_efs=False. Every mount uses IAM
          authorization, so EFS enforces that: the worker's mount is
          read-only at the file system as well as by the task definition's
          read-only mount (isolation boundary = read-only mount + /tmp
          snapshot).
        - Use of the buckets' and the table's customer-managed keys, each only
          through its own service, for control-plane roles. Roles with
          ``minted_data_plane_only=True`` get no key access, for the same
          reason they get no S3 or DynamoDB grant. EFS needs no key grant: it
          uses its key through a grant the service holds itself.
        - ``bedrock:InvokeModel*`` on any foundation-model, inference-profile
          or application-inference-profile ARN; the allowed model ids are
          pinned in app config, not IAM. Tightening to specific model ARNs is
          future hardening.

        Args:
            writable_efs: If True (default), grant ClientWrite on EFS; if False,
                grant only ClientMount + DescribeMountTargets (read-only mount).
            minted_data_plane_only: When True, omit the cross-tenant S3
                (sessions + uploads-staging) and DynamoDB statements, and the
                KMS grants that go with them, entirely. Use for any role whose container executes
                agent-authored code: such a role must reach S3/DynamoDB
                only through the minted scoped-session credential, whose
                ``${aws:PrincipalTag/*}``-templated statements are the
                tenant bound. An ambient statement here would be
                reachable by that code (botocore falls back to the ECS
                credential endpoint whenever ``AWS_CONFIG_FILE`` is
                dropped) and would silently supersede the bound.
        """
        # S3 + DynamoDB — the cross-tenant data plane. Neither statement
        # carries a tenant condition, and neither CAN: a bare task-role
        # session has no principal tags, so a ${aws:PrincipalTag/uid}
        # template would resolve to nothing and deny outright. Per-tenant
        # scoping is therefore structural — control-plane roles hold the
        # broad grant and enforce ownership in application code at
        # ``require_access``; roles running agent-authored code hold no
        # grant at all and go through the minted credential instead.
        # AbortMultipartUpload + ListMultipartUploadParts are needed for
        # boto3's >8 MB multipart upload path. Uploads staging follows the
        # same shape.
        if not minted_data_plane_only:
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "s3:GetObject",
                        "s3:GetObjectVersion",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:DeleteObjectVersion",
                        "s3:ListBucket",
                        "s3:ListBucketVersions",
                        "s3:GetBucketLocation",
                        "s3:AbortMultipartUpload",
                        "s3:ListMultipartUploadParts",
                    ],
                    resources=[
                        sessions_bucket.bucket_arn,
                        f"{sessions_bucket.bucket_arn}/*",
                        uploads_staging_bucket.bucket_arn,
                        f"{uploads_staging_bucket.bucket_arn}/*",
                    ],
                )
            )

        # EFS — Mount + IO. Tasks mount through the access point, which
        # squashes every NFS op to UID/GID 1000, so there is no per-user POSIX
        # enforcement: every tree is readable and writable by that one
        # identity, and separation between users' trees is the app-layer
        # ``require_access`` check plus the path layout. The container runs as
        # the non-root ``app`` user (UID 1000, see ``backend/Dockerfile``), so
        # ``ClientMount`` + ``ClientWrite`` are sufficient — ``ClientRootAccess``
        # is intentionally omitted. Worker receives writable_efs=False so it
        # never holds ClientWrite. Every mount uses IAM authorization (the ECS
        # volumes in compute_stack.py, and AgentCore's managed mount), so EFS
        # evaluates these grants, and the file system's policy in
        # storage_stack.py grants nothing of its own.
        efs_actions = [
            "elasticfilesystem:ClientMount",
            "elasticfilesystem:DescribeMountTargets",
        ]
        if writable_efs:
            efs_actions.append("elasticfilesystem:ClientWrite")
        role.add_to_policy(
            iam.PolicyStatement(
                actions=efs_actions,
                resources=[file_system.file_system_arn],
            )
        )
        # DynamoDB — the other half of the cross-tenant data plane; same
        # control-plane-only reasoning as the S3 statement above.
        if not minted_data_plane_only:
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "dynamodb:GetItem",
                        "dynamodb:PutItem",
                        "dynamodb:UpdateItem",
                        "dynamodb:DeleteItem",
                        "dynamodb:Query",
                        "dynamodb:BatchGetItem",
                        "dynamodb:BatchWriteItem",
                        "dynamodb:ConditionCheckItem",
                    ],
                    resources=[
                        table.table_arn,
                        f"{table.table_arn}/index/*",
                    ],
                )
            )
        # The keys behind the two statements above. Without them every S3 and
        # DynamoDB call fails with an AccessDenied that names KMS. The
        # coordinator also signs the uploads-staging presigned PUTs, and S3
        # checks the signer's GenerateDataKey on the bucket's key.
        if not minted_data_plane_only:
            region = Stack.of(role).region
            role.add_to_policy(
                _bucket_key_statement(
                    region, sessions_bucket, uploads_staging_bucket,
                )
            )
            role.add_to_policy(_table_key_statement(region, table))
        # Bedrock. The bundled ``claude`` CLI's ``initialize()`` calls
        # ``ListInferenceProfiles`` + ``GetInferenceProfile`` to resolve
        # the configured model id to a backing foundation-model ARN
        # before its first ``InvokeModel`` call. Without those two
        # actions the CLI hangs at initialize until the SDK's 60s
        # control-request timeout fires (verified strings in the
        # bundled binary: ``Access denied. Your IAM role needs
        # bedrock:ListInferenceProfiles permission.`` plus
        # ``paginateListInferenceProfiles`` / ``GetInferenceProfileCommand``).
        #
        # Resource scope covers all three ARN shapes Bedrock recognizes
        # for invocation + inference-profile resolution:
        #   * ``foundation-model/*`` — direct model ARNs.
        #   * ``inference-profile/*`` — region-aware aliases.
        #   * ``application-inference-profile/*`` — customer-managed
        #     wrappers used by Anthropic-via-Bedrock.
        # ``*`` wildcards on the discovery actions because
        # ``ListInferenceProfiles`` doesn't accept a per-profile ARN.
        # Tightening the invocation actions to specific model ARNs is a
        # future hardening step once the allowed-model list stabilizes.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetInferenceProfile",
                ],
                resources=[
                    f"arn:aws:bedrock:*:{Stack.of(role).account}:inference-profile/*",
                    f"arn:aws:bedrock:*:{Stack.of(role).account}:application-inference-profile/*",
                    "arn:aws:bedrock:*::foundation-model/*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:ListFoundationModels",
                ],
                resources=["*"],
            )
        )

        # Bedrock Mantle — AWS-managed policy for inference access.
        # Replaces the hand-written bedrock-mantle:CreateInference/Get*/List*
        # statement with AmazonBedrockMantleInferenceAccess (AWS-maintained
        # source of truth). The managed policy includes bedrock-mantle
        # inference actions + CallWithBearerToken + aws-marketplace surface
        # that the hand-written statement lacked (future-proof; unused today).
        #
        # Attached to ALL THREE shared roles (coordinator/worker/agentcore)
        # via this _attach_shared_policies helper AND to scoped_session_role
        # separately (see the scoped_session_role construction in
        # ``__init__``). The separate attachment is required because a
        # credential minted from scoped_session_role can do no more than that
        # role's own policies allow, so mantle calls on minted credentials are
        # denied unless it carries the grant itself.
        #
        # The wider surface (CallWithBearerToken + aws-marketplace, unused
        # today) is accepted in exchange for tracking AWS's own definition.
        # The managed policy scopes CreateInference to project/* resources
        # (not *); if runtime authorization ever gates on a different
        # resource pattern, those calls are denied, and the harness's stderr
        # [generation-error] logging names the denial.
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonBedrockMantleInferenceAccess"
            )
        )

        # account:ListRegions — dynamic region discovery.
        # This is SEPARATE from the mantle block and is NOT covered by the
        # AWS-managed AmazonBedrockMantleInferenceAccess policy, so it stays
        # as an inline statement. Resources=["*"] is the only valid scope for
        # account:* actions (no per-resource ARN exists for the account
        # service; equivalent to ec2:DescribeRegions scope).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["account:ListRegions"],
                resources=["*"],
            )
        )

        # SSM Messages — required by the ECS Exec data plane. The
        # ``execute-command`` agent inside the Fargate task opens a
        # WebSocket-style channel to ``ssmmessages.<region>.amazonaws.com``
        # for stdin/stdout/stderr forwarding. Without these four actions
        # the channel handshake 403's and ``aws ecs execute-command``
        # times out at "TargetNotConnected". Resources are ``*`` per
        # AWS docs — Session Manager doesn't accept per-resource ARNs
        # for messaging actions.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                resources=["*"],
            )
        )
