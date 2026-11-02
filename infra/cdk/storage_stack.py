# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage stack: S3 buckets, DynamoDB single-table, EFS file system.

Schema:

- ``sessions`` S3 bucket — Claude SDK session JSONL part files, append-only,
  no versioning.
- ``uploads-staging`` S3 bucket — landing zone for ≥50 MB browser uploads
  via presigned PUT. Objects auto-expire after 24h since the
  backend copies them into EFS shortly after upload.
- DynamoDB table ``ai-eval-studio-metadata`` with PK/SK + GSI1, on-demand
  billing, point-in-time recovery, encryption at rest, TTL on the ``ttl``
  attribute (auto-cleans locks, events).
- **EFS file system** for project working trees. Each project
  lives at ``users/{owner_sub}/projects/{pid}/`` (relative to the access
  point root) including per-project ``.venv/``. The access point chroots
  every Fargate mount to ``/projects`` and squashes all NFS ops to UID/
  GID 1000 (matching ``USER app`` in the runtime Dockerfile), so the
  coordinator can read and write without root.

  Tenant isolation is enforced at the **app layer** (``require_access``)
  plus the **per-user path layout** above, for API routes and the
  path-jailed file tools; the agent's ``Bash`` tool passes through neither
  (see the README's Security scope). Note the access point squashes every
  NFS op to UID/GID 1000, so per-UID file permissions are not a
  distinguishing mechanism here.

EFS holds project working trees; no separate S3 bucket is provisioned
for them.

Encryption at rest uses customer-managed KMS keys, one per kind of store: the
three data buckets share one, and the table and the file system have their
own. The deployer owns each key policy, can disable a key to make its data
unreadable at once, and sees every use of it attributed to a key they own.
Separate keys keep each audit trail readable and limit what one grant can
decrypt. Every key rotates yearly and is retained with the store it
encrypts: deleting a key would leave the retained data unreadable.
"""

from __future__ import annotations

import os

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_efs as efs,
    aws_iam as iam,
    aws_kms as kms,
    aws_s3 as s3,
)
from constructs import Construct

from network_stack import S3_ACCESS_LOG_PREFIX, access_logs_bucket_name

# First deploy of a fresh account doesn't have a CloudFront DNS yet —
# ``auth_stack.py`` uses the same placeholder pattern. ``dev-postdeploy.sh``
# resolves the real DNS from the frontend stack's CFN output, writes it
# into ``cdk.context.json``, and re-runs ``cdk deploy ai-eval-studio-auth``
# so the production posture catches up. The CORS rule below behaves the
# same way: bootstrap-empty resolves to no allowed origins (the staging
# bucket is unreachable from the SPA until the second deploy), then real
# DNS locks the rule to the deployed origin.
_PLACEHOLDER_CLOUDFRONT_DNS = "placeholder.cloudfront.net"


def _staging_cors_origins(stack: Stack) -> list[str]:
    """Return the CORS allowed_origins list for the staging bucket.

    Resolves the deployed CloudFront DNS from CDK context (or the
    ``CLOUDFRONT_DNS`` env var as a deploy-time override) and emits
    one ``https://`` origin for it. Falls back to the placeholder DNS
    on a first-account bootstrap (the staging bucket is unreachable
    from the SPA until ``dev-postdeploy.sh`` re-runs with the real
    DNS — same posture ``auth_stack.py`` accepts).

    ``http://localhost:3100`` is always included so a developer running
    the SPA against a real cloud staging bucket can still test uploads.
    """
    cloudfront_dns = (
        stack.node.try_get_context("cloudfront_dns")
        or os.environ.get("CLOUDFRONT_DNS")
        or _PLACEHOLDER_CLOUDFRONT_DNS
    )
    return [
        f"https://{cloudfront_dns}",
        "http://localhost:3100",
    ]


# The NFS client actions. The file-system policy below restricts only these,
# so it can never stand between CloudFormation and its own management calls.
_EFS_CLIENT_ACTIONS = [
    "elasticfilesystem:ClientMount",
    "elasticfilesystem:ClientWrite",
    "elasticfilesystem:ClientRootAccess",
]


def _efs_client_policy() -> iam.PolicyDocument:
    """Return the EFS file-system policy: TLS and a mount target, or no mount.

    Any client that connects without TLS, or other than through a mount
    target in this VPC, is refused. The policy grants nothing: with a policy in
    place EFS no longer lets every reachable client in, so each mount needs an
    IAM identity whose own policy allows the client actions. Those grants are
    in ``iam_stack.py`` (``ClientMount`` for every role, ``ClientWrite`` only
    for the roles that write). The ECS volumes mount with IAM authorization
    for that reason, and AgentCore's managed mount uses TLS and IAM itself.
    """
    return iam.PolicyDocument(
        statements=[
            iam.PolicyStatement(
                sid="DenyInsecureTransport",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=_EFS_CLIENT_ACTIONS,
                resources=["*"],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
            ),
            iam.PolicyStatement(
                sid="DenyAccessOutsideMountTargets",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=_EFS_CLIENT_ACTIONS,
                resources=["*"],
                conditions={
                    "Bool": {"elasticfilesystem:AccessedViaMountTarget": "false"}
                },
            ),
        ],
    )


def _data_key(scope: Construct, construct_id: str, description: str) -> kms.Key:
    """Return a customer-managed key for one kind of data store.

    No alias: a retained key keeps its alias, so a redeploy after a teardown
    would fail on the alias name. The description identifies the key instead.
    The default key policy delegates use of the key to IAM, so each role's
    grant lives in its own policy in ``iam_stack.py``.
    """
    return kms.Key(
        scope,
        construct_id,
        description=description,
        enable_key_rotation=True,
        removal_policy=RemovalPolicy.RETAIN,
    )


class StorageStack(Stack):
    """S3 buckets + DynamoDB single-table + EFS file system for Eval Studio.

    Attributes:
        sessions_bucket: Claude SDK session transcripts (``sessions/{key}/...``).
        uploads_staging_bucket: Landing zone for ≥50 MB browser uploads
            via presigned PUT. 24h object expiry — backend copies content
            into EFS, then deletes.
        snapshots_bucket: Short-lived per-job working-tree tarballs, keyed under
            the job-derived prefix
            ``project_key_for_directory("/tmp/workspace/{job_id}")/snapshots/{job_id}.tar.gz``.
            The worker's minted STS creds reach only the prefix of the ``job``
            session tag they were minted with. The tag's value is chosen by
            whoever mints, and IAM cannot bind it to the job a container was
            launched for (see the trust-policy note in ``iam_stack.py``), so
            this narrows a correctly minted credential; it is not a tenant
            boundary. 1-day expiry cleans up completed/failed eval snapshots
            automatically.
        table: Single DynamoDB table ``ai-eval-studio-metadata`` with one GSI.
        file_system: EFS for project working trees. Mounted on the
            coordinator Fargate task at ``/mnt/projects``.
        access_point: EFS access point through which Fargate mounts the
            file system. Chroots to ``/projects`` and squashes all
            NFS ops to UID/GID 1000.
        access_security_group: Security group attached to EFS mount targets;
            Fargate tasks add this group's ID to their inbound rules.
        data_key: Customer-managed key for the three data buckets.
        table_key: Customer-managed key for the table and its backups.
        file_system_key: Customer-managed key for EFS. EFS uses it through a
            grant it holds itself, so mounting clients need no KMS access.

    Args:
        vpc: VPC from ``NetworkStack``. EFS mount targets need to live in
            the private subnets where Fargate tasks run.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account = self.account
        region = self.region

        self.data_key = _data_key(
            self, "DataBucketsKey",
            "AI Eval Studio: sessions, uploads-staging and snapshots buckets",
        )
        self.table_key = _data_key(self, "TableKey", "AI Eval Studio: metadata table")
        self.file_system_key = _data_key(
            self, "FileSystemKey", "AI Eval Studio: project working trees on EFS",
        )
        # Server access logs for the three data buckets: a per-object record of
        # every read and write, delivered to the access-log bucket the network
        # stack owns (imported by name, so this stack adds no cross-stack
        # export). That bucket's policy already admits S3 log delivery.
        access_logs_bucket = s3.Bucket.from_bucket_name(
            self, "AccessLogsBucket", access_logs_bucket_name(self)
        )

        # Sessions bucket: no versioning. SDK SessionStore writes are append-only
        # JSONL parts; versioning would double cost without recovery benefit.
        #
        # Lifecycle: storage-class transition keeps long-lived
        # transcripts cheap, and the abort-incomplete-multipart rule
        # cleans up failed multipart uploads before they accrue cost.
        # We don't expire objects — a transcript belongs to its
        # project and cascades to delete via the project-delete path
        # (``_delete_session_transcript`` in ``backend/app/api/projects.py``);
        # platform-level expiry would silently lose history under user feet.
        self.sessions_bucket = s3.Bucket(
            self,
            "SessionsBucket",
            bucket_name=f"ai-eval-studio-sessions-{account}-{region}",
            server_access_logs_bucket=access_logs_bucket,
            server_access_logs_prefix=f"{S3_ACCESS_LOG_PREFIX}sessions/",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            # Bucket keys cut KMS requests from one per object to one per
            # bucket key, which is most of the cost of a customer-managed key.
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=False,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="sessions-cost-control",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        ),
                    ],
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
            ],
        )

        # Uploads staging: ≥50 MB browser PUTs land here. Backend copies
        # to EFS within seconds and deletes; the 24h expiry is a safety
        # net for orphaned uploads.
        self.uploads_staging_bucket = s3.Bucket(
            self,
            "UploadsStagingBucket",
            bucket_name=f"ai-eval-studio-uploads-staging-{account}-{region}",
            server_access_logs_bucket=access_logs_bucket,
            server_access_logs_prefix=f"{S3_ACCESS_LOG_PREFIX}uploads-staging/",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            # Bucket keys cut KMS requests from one per object to one per
            # bucket key, which is most of the cost of a customer-managed key.
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=False,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="staging-24h-expiry",
                    enabled=True,
                    expiration=Duration.days(1),
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[
                        s3.HttpMethods.PUT,
                        s3.HttpMethods.POST,
                    ],
                    # Lock the CORS allowlist to the deployed CloudFront
                    # origin (and ``localhost:3100`` for dev-mode SPAs
                    # talking to a real cloud staging bucket). The
                    # presigned-URL signature is the actual gate, but
                    # an explicit allowlist is defense-in-depth: if a
                    # signed URL ever leaks, the browser's CORS check
                    # blocks any other origin from using it.
                    allowed_origins=_staging_cors_origins(self),
                    allowed_headers=["*"],
                    exposed_headers=["ETag"],
                    max_age=3000,
                ),
            ],
        )

        # Snapshots bucket: short-lived per-job working-tree tarballs, keyed
        # under the job-derived prefix
        # project_key_for_directory("/tmp/workspace/{job_id}")/snapshots/{job_id}.tar.gz.
        # The worker's minted STS creds reach only the prefix of the ``job``
        # session tag they were minted with. That narrows a correctly minted
        # credential; it is not a tenant boundary, because the tag's value is
        # chosen by whoever mints and IAM cannot bind it to the launched job
        # (see the trust-policy note in iam_stack.py).
        # 1-day expiry cleans up completed/failed eval snapshots automatically.
        self.snapshots_bucket = s3.Bucket(
            self,
            "SnapshotsBucket",
            bucket_name=f"ai-eval-studio-snapshots-{account}-{region}",
            server_access_logs_bucket=access_logs_bucket,
            server_access_logs_prefix=f"{S3_ACCESS_LOG_PREFIX}snapshots/",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            # Bucket keys cut KMS requests from one per object to one per
            # bucket key, which is most of the cost of a customer-managed key.
            bucket_key_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=False,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="snapshots-1d-expiry",
                    enabled=True,
                    expiration=Duration.days(1),
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
            ],
        )

        # Single-table DynamoDB design.
        self.table = dynamodb.Table(
            self,
            "Table",
            table_name="ai-eval-studio-metadata",
            partition_key=dynamodb.Attribute(
                name="PK", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="SK", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.table_key,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.RETAIN,
            # Refuse any request not made over TLS, as ``enforce_ssl`` does for
            # the buckets. The AWS SDKs always use TLS, so this changes nothing
            # for the app; it closes the door on a misconfigured client.
            # ``*`` is the table itself: a table's policy cannot name its own
            # ARN, which does not exist until the table does.
            resource_policy=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        sid="DenyInsecureTransport",
                        effect=iam.Effect.DENY,
                        principals=[iam.AnyPrincipal()],
                        actions=["dynamodb:*"],
                        resources=["*"],
                        conditions={"Bool": {"aws:SecureTransport": "false"}},
                    ),
                ],
            ),
        )

        # GSI1: inverse lookups — jobs by state for operator queries, and the
        # AgentCore session claims currently held, which the startup reconciler
        # reads. ProjectionType=ALL so those lookups get the full record without
        # a follow-up GetItem.
        self.table.add_global_secondary_index(
            index_name="GSI1",
            partition_key=dynamodb.Attribute(
                name="GSI1PK", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="GSI1SK", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ---- EFS for project working trees -------------------------------
        # Security group on the mount targets. Inbound NFS (TCP/2049) is
        # allowed from the entire VPC CIDR — the only thing in this VPC
        # are our private-subnet Fargate tasks (the ALB lives in private
        # subnets too but doesn't speak NFS). Allowing the Fargate
        # service SG specifically would require a stack reference from
        # storage → compute, but compute already depends on storage
        # (file_system), so that closes a cycle. VPC-CIDR scoping is
        # equivalent in practice given our subnet layout.
        self.access_security_group = ec2.SecurityGroup(
            self,
            "EfsAccessSg",
            vpc=vpc,
            description="EFS mount-target access for AI Eval Studio Fargate tasks",
            allow_all_outbound=False,
        )
        self.access_security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(2049),
            description="NFS from in-VPC Fargate tasks",
        )

        # Elastic throughput keeps small steady-state cost low and bursts
        # for the cold-install case (~3 GB uv sync writes lots of small
        # files). Encrypted at rest with its own customer-managed key, and in
        # transit as the file-system policy below requires.
        # Lifecycle: cold files transition to IA at 30d (auto-billed at
        # ~$0.025/GB-mo vs $0.30/GB-mo standard), bring back to standard
        # on first access.
        self.file_system = efs.FileSystem(
            self,
            "ProjectsFs",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
            security_group=self.access_security_group,
            encrypted=True,
            kms_key=self.file_system_key,
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
            throughput_mode=efs.ThroughputMode.ELASTIC,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_30_DAYS,
            out_of_infrequent_access_policy=efs.OutOfInfrequentAccessPolicy.AFTER_1_ACCESS,
            removal_policy=RemovalPolicy.RETAIN,
            file_system_policy=_efs_client_policy(),
        )

        # EFS access point — Fargate mounts via this rather than the
        # filesystem root. Two effects:
        #
        # 1. ``create_acl`` runs once on first attach, ``chmod``ing the
        #    access-point root (``/projects``) to 0750 owned by UID/GID
        #    1000. Without this the FS root is ``root:root 755`` and the
        #    non-root container can't write.
        # 2. ``posix_user`` squashes every NFS op to UID/GID 1000
        #    regardless of the calling process's UID. The kernel sees
        #    UID 1000 for every read/write — per-user UIDs and ACL/
        #    setfacl-based fences cannot work over an access point with
        #    a fixed posix_user. Tenant isolation is enforced at the
        #    app layer (``require_access``) + the per-user path layout
        #    (``users/{owner_sub}/projects/{pid}/``) — for API routes and
        #    the path-jailed file tools, not for the agent's ``Bash`` tool.
        #
        # 1000:1000 must match ``backend/Dockerfile``'s ``useradd``
        # invocation. Drift would leave the container unable to write.
        self.access_point = efs.AccessPoint(
            self,
            "ProjectsAp",
            file_system=self.file_system,
            path="/projects",
            create_acl=efs.Acl(
                owner_uid="1000",
                owner_gid="1000",
                permissions="750",
            ),
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
        )
