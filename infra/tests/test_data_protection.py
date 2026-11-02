# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level guards on encryption in transit, encryption at rest and access logs.

Each property here is a one-word edit away from reverting with nothing else
visibly changing: the stack still deploys and the app still works, while a hop
quietly goes back to plaintext, a store back to an AWS-managed key, or a
request path back to unlogged. So each is pinned against the synthesized
templates, which is what would really be deployed.
"""

import json
import sys
from pathlib import Path

import pytest
from aws_cdk import App, Environment

TEST_ENV = Environment(account="111111111111", region="us-east-1")
TEST_SUBNET_IDS = "subnet-0123456789abcdef0"


@pytest.fixture(scope="module")
def templates() -> dict[str, dict]:
    """Synthesize the full app and return {stack name: template dict}."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))

    from agentcore_stack import AgentCoreStack
    from auth_stack import AuthStack
    from compute_stack import ComputeStack
    from frontend_stack import FrontendStack
    from iam_stack import IamStack
    from network_stack import NetworkStack
    from observability_stack import ObservabilityStack
    from storage_stack import StorageStack

    # ``aws:cdk:bundling-stacks`` empty skips asset bundling, so the frontend
    # stack synthesizes without running the SPA build in a container.
    app = App(
        context={
            "agentcore_subnet_ids": TEST_SUBNET_IDS,
            "aws:cdk:bundling-stacks": [],
        }
    )
    network = NetworkStack(app, "network", env=TEST_ENV)
    storage = StorageStack(app, "storage", vpc=network.vpc, env=TEST_ENV)
    auth = AuthStack(app, "auth", env=TEST_ENV)
    iam_stack = IamStack(
        app,
        "iam",
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        file_system=storage.file_system,
        env=TEST_ENV,
    )
    observability = ObservabilityStack(app, "observability", env=TEST_ENV)
    compute = ComputeStack(
        app,
        "compute",
        vpc=network.vpc,
        coordinator_role=iam_stack.coordinator_task_role,
        worker_role=iam_stack.worker_task_role,
        worker_execution_role=iam_stack.worker_execution_role,
        coordinator_target_group=network.coordinator_target_group,
        coordinator_log_group=observability.coordinator_log_group,
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        file_system=storage.file_system,
        efs_access_point=storage.access_point,
        scoped_session_role=iam_stack.scoped_session_role,
        user_pool=auth.user_pool,
        spa_client=auth.spa_client,
        user_pool_domain=auth.user_pool_domain,
        env=TEST_ENV,
    )
    FrontendStack(app, "frontend", alb=network.alb, env=TEST_ENV)
    AgentCoreStack(
        app,
        "agentcore",
        vpc=network.vpc,
        efs_access_point=storage.access_point,
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        scoped_session_role=iam_stack.scoped_session_role,
        runtime_role=iam_stack.agentcore_runtime_role,
        user_pool=auth.user_pool,
        spa_client=auth.spa_client,
        worker_launch_env=compute.worker_launch_env,
        env=TEST_ENV,
    )

    assembly = app.synth()
    return {
        artifact.stack_name: json.loads(
            (Path(assembly.directory) / artifact.template_file).read_text()
        )
        for artifact in assembly.stacks
    }


def _resources(template: dict, resource_type: str) -> dict[str, dict]:
    """Return {logical id: resource} for every resource of ``resource_type``."""
    return {
        logical_id: resource
        for logical_id, resource in template["Resources"].items()
        if resource["Type"] == resource_type
    }


def _only(template: dict, resource_type: str) -> dict:
    """Return the single resource of ``resource_type``, failing if not exactly one."""
    found = _resources(template, resource_type)
    assert len(found) == 1, f"expected one {resource_type}, found {sorted(found)}"
    return next(iter(found.values()))


# ---- ALB to coordinator ---------------------------------------------------


def test_alb_reaches_the_coordinator_over_https(templates):
    """The target group, and its health check, use HTTPS rather than HTTP."""
    group = _only(templates["network"], "AWS::ElasticLoadBalancingV2::TargetGroup")
    props = group["Properties"]
    assert props["Protocol"] == "HTTPS", f"target group protocol: {props['Protocol']}"
    assert props["HealthCheckProtocol"] == "HTTPS", (
        f"health check protocol: {props.get('HealthCheckProtocol')}"
    )


def _coordinator_command(compute_template: dict) -> list[str]:
    task_defs = _resources(compute_template, "AWS::ECS::TaskDefinition")
    coordinator = next(
        resource for logical_id, resource in task_defs.items()
        if "Coordinator" in logical_id and "Worker" not in logical_id
    )
    return coordinator["Properties"]["ContainerDefinitions"][0]["Command"]


def test_coordinator_serves_tls(templates):
    """The deployed command and the image's default both start the TLS entrypoint.

    A target group on HTTPS in front of a container serving plain HTTP fails
    every health check, so the two must change together. The image's own CMD
    is pinned too, so running the image directly cannot quietly serve HTTP.
    """
    command = _coordinator_command(templates["compute"])
    assert command[:3] == ["python", "-m", "app.scripts.serve_tls"], command

    dockerfile = Path(__file__).resolve().parents[2] / "backend" / "Dockerfile"
    cmd_line = next(
        line for line in dockerfile.read_text().splitlines() if line.startswith("CMD ")
    )
    assert json.loads(cmd_line[len("CMD "):])[:3] == [
        "python", "-m", "app.scripts.serve_tls",
    ], cmd_line


# ---- EFS ------------------------------------------------------------------


def test_every_ecs_efs_volume_uses_tls_and_iam(templates):
    """Both task definitions mount EFS with TLS and IAM authorization.

    The file system's policy refuses clients without either, so a volume that
    drops one fails to mount; pinning both keeps that failure out of a deploy.
    """
    task_defs = _resources(templates["compute"], "AWS::ECS::TaskDefinition")
    assert len(task_defs) == 2, sorted(task_defs)
    for logical_id, task_def in task_defs.items():
        volumes = task_def["Properties"]["Volumes"]
        assert volumes, f"{logical_id} has no volumes"
        for volume in volumes:
            efs = volume["EFSVolumeConfiguration"]
            assert efs["TransitEncryption"] == "ENABLED", (logical_id, efs)
            assert efs["AuthorizationConfig"]["IAM"] == "ENABLED", (logical_id, efs)


def _deny_conditions(policy: dict) -> list[dict]:
    return [s["Condition"] for s in policy["Statement"] if s["Effect"] == "Deny"]


def test_efs_policy_refuses_plaintext_and_off_vpc_clients(templates):
    """The file-system policy denies non-TLS clients and any path but a mount target."""
    file_system = _only(templates["storage"], "AWS::EFS::FileSystem")
    policy = file_system["Properties"]["FileSystemPolicy"]
    conditions = _deny_conditions(policy)
    assert {"Bool": {"aws:SecureTransport": "false"}} in conditions, conditions
    assert {
        "Bool": {"elasticfilesystem:AccessedViaMountTarget": "false"}
    } in conditions, conditions


def test_efs_policy_grants_nothing(templates):
    """The policy only denies, so each mount stands on its role's own grants.

    An Allow here would let a role mount, or write, without its IAM policy
    saying so, which would undo the worker's read-only grant.
    """
    file_system = _only(templates["storage"], "AWS::EFS::FileSystem")
    effects = {s["Effect"] for s in file_system["Properties"]["FileSystemPolicy"]["Statement"]}
    assert effects == {"Deny"}, effects


# ---- DynamoDB -------------------------------------------------------------


def test_table_refuses_plaintext_requests(templates):
    """The table's resource policy denies any request not made over TLS."""
    table = _only(templates["storage"], "AWS::DynamoDB::Table")
    policy = table["Properties"]["ResourcePolicy"]["PolicyDocument"]
    assert {"Bool": {"aws:SecureTransport": "false"}} in _deny_conditions(policy), policy


# ---- Customer-managed keys --------------------------------------------------


def _logical_id_of(ref) -> str | None:
    """Return the logical id an in-template ``Ref``/``GetAtt`` points at."""
    if isinstance(ref, dict):
        if "Ref" in ref:
            return ref["Ref"]
        if "Fn::GetAtt" in ref:
            return ref["Fn::GetAtt"][0]
    return None


def test_storage_keys_rotate_and_are_retained(templates):
    """Three keys, one per kind of store, each rotating yearly and retained.

    A key deleted with its stack would leave the retained data it encrypts
    unreadable, so its deletion policy must match theirs.
    """
    keys = _resources(templates["storage"], "AWS::KMS::Key")
    assert len(keys) == 3, sorted(keys)
    for logical_id, key in keys.items():
        assert key["Properties"]["EnableKeyRotation"] is True, logical_id
        assert key["DeletionPolicy"] == "Retain", logical_id


def test_data_buckets_use_the_customer_managed_key(templates):
    """Every data bucket encrypts with SSE-KMS under a key in the same stack."""
    keys = set(_resources(templates["storage"], "AWS::KMS::Key"))
    buckets = _resources(templates["storage"], "AWS::S3::Bucket")
    assert len(buckets) == 3, sorted(buckets)
    for logical_id, bucket in buckets.items():
        rule = bucket["Properties"]["BucketEncryption"][
            "ServerSideEncryptionConfiguration"
        ][0]
        default = rule["ServerSideEncryptionByDefault"]
        assert default["SSEAlgorithm"] == "aws:kms", (logical_id, default)
        assert _logical_id_of(default["KMSMasterKeyID"]) in keys, (logical_id, default)
        assert rule["BucketKeyEnabled"] is True, logical_id


def test_table_uses_the_customer_managed_key(templates):
    """The table encrypts under a key in the same stack, not the AWS-managed one."""
    keys = set(_resources(templates["storage"], "AWS::KMS::Key"))
    sse = _only(templates["storage"], "AWS::DynamoDB::Table")["Properties"][
        "SSESpecification"
    ]
    assert sse["SSEEnabled"] is True and sse["SSEType"] == "KMS", sse
    assert _logical_id_of(sse["KMSMasterKeyId"]) in keys, sse


def test_file_system_uses_the_customer_managed_key(templates):
    """EFS encrypts under a key in the same stack."""
    keys = set(_resources(templates["storage"], "AWS::KMS::Key"))
    props = _only(templates["storage"], "AWS::EFS::FileSystem")["Properties"]
    assert props["Encrypted"] is True
    assert _logical_id_of(props["KmsKeyId"]) in keys, props.get("KmsKeyId")


@pytest.mark.parametrize("stack", ["observability"])
def test_log_groups_use_a_key_logs_may_use(templates, stack):
    """Each log group encrypts under a key whose policy lets CloudWatch Logs in.

    CloudWatch Logs uses the key as its own service principal, so a key policy
    without that statement fails the log group's creation.
    """
    template = templates[stack]
    keys = _resources(template, "AWS::KMS::Key")
    groups = _resources(template, "AWS::Logs::LogGroup")
    assert groups, f"no log groups in {stack}"
    for logical_id, group in groups.items():
        key_id = _logical_id_of(group["Properties"].get("KmsKeyId"))
        assert key_id in keys, (logical_id, group["Properties"].get("KmsKeyId"))
        assert group["DeletionPolicy"] == "Retain", logical_id
        statements = keys[key_id]["Properties"]["KeyPolicy"]["Statement"]
        logs_statements = [
            s for s in statements if "logs." in json.dumps(s.get("Principal", {}))
        ]
        assert logs_statements, f"{key_id} lets CloudWatch Logs nothing"
        assert all(
            "kms:EncryptionContext:aws:logs:arn" in s["Condition"]["ArnLike"]
            for s in logs_statements
        ), logs_statements


def _role_statements(template: dict) -> dict[str, list[dict]]:
    """Return {role logical id: every statement of its inline policies}."""
    found: dict[str, list[dict]] = {}
    for policy in _resources(template, "AWS::IAM::Policy").values():
        props = policy["Properties"]
        for role in props.get("Roles", []):
            found.setdefault(_logical_id_of(role), []).extend(
                props["PolicyDocument"]["Statement"]
            )
    return found


def _role_id(template: dict, role_name: str) -> str:
    return next(
        logical_id
        for logical_id, role in _resources(template, "AWS::IAM::Role").items()
        if role["Properties"].get("RoleName") == role_name
    )


def _kms_statements(statements: list[dict]) -> list[dict]:
    def actions(statement):
        action = statement["Action"]
        return action if isinstance(action, list) else [action]

    return [s for s in statements if any(a.startswith("kms:") for a in actions(s))]


def test_no_role_may_use_every_key_in_the_account(templates):
    """KMS grants name the keys; none is on ``*``.

    ``*`` with only a ViaService condition was the pattern for AWS-managed keys,
    whose ARNs are not known at synth. Left in place it would also let a role
    use any customer-managed key in the account through S3 or DynamoDB.
    """
    for statements in _role_statements(templates["iam"]).values():
        for statement in _kms_statements(statements):
            assert statement["Resource"] != "*", statement
            assert "kms:ViaService" in statement["Condition"]["StringEquals"], statement


def _via_services(statements: list[dict]) -> set[str]:
    return {
        s["Condition"]["StringEquals"]["kms:ViaService"]
        for s in _kms_statements(statements)
    }


@pytest.mark.parametrize(
    "role_name",
    [
        "ai-eval-studio-coordinator-task-role",
        "ai-eval-studio-agentcore-runtime-role",
        "ai-eval-studio-scoped-session-role",
    ],
)
def test_data_plane_roles_may_use_the_bucket_and_table_keys(templates, role_name):
    """Every role that reads or writes the buckets and the table holds both keys.

    Without them each S3 and DynamoDB call fails with an AccessDenied naming
    KMS. The scoped-session role is the easy one to miss: it has no other
    identity, and every minted credential depends on it.
    """
    template = templates["iam"]
    statements = _role_statements(template)[_role_id(template, role_name)]
    assert _via_services(statements) == {
        "s3.us-east-1.amazonaws.com",
        "dynamodb.us-east-1.amazonaws.com",
    }, _kms_statements(statements)


def test_worker_role_holds_no_key(templates):
    """The worker reaches data only through its minted credential, so its own
    role holds no key, just as it holds no S3 or DynamoDB grant."""
    template = templates["iam"]
    statements = _role_statements(template).get(
        _role_id(template, "ai-eval-studio-worker-task-role"), []
    )
    assert _kms_statements(statements) == []


# ---- Access logs ------------------------------------------------------------

ACCESS_LOGS_BUCKET_NAME = "ai-eval-studio-access-logs-111111111111-us-east-1"


def _access_logs_bucket(templates: dict) -> tuple[str, dict]:
    buckets = _resources(templates["network"], "AWS::S3::Bucket")
    assert len(buckets) == 1, sorted(buckets)
    return next(iter(buckets.items()))


def test_access_log_bucket_is_private_retained_and_expiring(templates):
    """The log bucket is SSE-S3, private, TLS-only, retained, and expires objects.

    SSE-S3 is required: ALB and S3 server access logs are delivered only to an
    SSE-S3 bucket. ACLs stay available because CloudFront's standard logging
    writes through an ACL grant.
    """
    logical_id, bucket = _access_logs_bucket(templates)
    props = bucket["Properties"]
    assert props["BucketName"] == ACCESS_LOGS_BUCKET_NAME
    default = props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"
    ]
    assert default == {"SSEAlgorithm": "AES256"}, default
    assert all(props["PublicAccessBlockConfiguration"].values()), props
    assert props["OwnershipControls"]["Rules"] == [
        {"ObjectOwnership": "BucketOwnerPreferred"}
    ]
    assert bucket["DeletionPolicy"] == "Retain"
    assert any(
        rule.get("ExpirationInDays") for rule in props["LifecycleConfiguration"]["Rules"]
    ), props["LifecycleConfiguration"]

    policies = _resources(templates["network"], "AWS::S3::BucketPolicy")
    statements = [
        s for p in policies.values() for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]
    assert any(
        s["Effect"] == "Deny" and s["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}
        for s in statements
    ), statements


def test_s3_log_delivery_is_limited_to_this_account(templates):
    """S3 may write server access logs here only for this account's buckets."""
    policies = _resources(templates["network"], "AWS::S3::BucketPolicy")
    statements = [
        s for p in policies.values() for s in p["Properties"]["PolicyDocument"]["Statement"]
        if s.get("Principal") == {"Service": "logging.s3.amazonaws.com"}
    ]
    assert len(statements) == 1, statements
    assert statements[0]["Condition"]["StringEquals"] == {
        "aws:SourceAccount": "111111111111"
    }, statements[0]


def test_alb_writes_access_logs(templates):
    """The ALB has access logging on, into the access-log bucket."""
    logical_id, _ = _access_logs_bucket(templates)
    alb = _only(templates["network"], "AWS::ElasticLoadBalancingV2::LoadBalancer")
    attributes = {
        a["Key"]: a["Value"] for a in alb["Properties"]["LoadBalancerAttributes"]
    }
    assert attributes["access_logs.s3.enabled"] == "true", attributes
    assert attributes["access_logs.s3.bucket"] == {"Ref": logical_id}, attributes


def test_data_buckets_write_server_access_logs(templates):
    """Every data bucket logs each request to the access-log bucket, under its own prefix."""
    buckets = _resources(templates["storage"], "AWS::S3::Bucket")
    prefixes = set()
    for logical_id, bucket in buckets.items():
        logging = bucket["Properties"].get("LoggingConfiguration")
        assert logging, f"{logical_id} has no server access logging"
        assert logging["DestinationBucketName"] == ACCESS_LOGS_BUCKET_NAME, logging
        prefixes.add(logging["LogFilePrefix"])
    assert len(prefixes) == len(buckets), prefixes


def test_cloudfront_writes_access_logs(templates):
    """The distribution's standard logging goes to the access-log bucket."""
    distribution = _only(templates["frontend"], "AWS::CloudFront::Distribution")
    logging = distribution["Properties"]["DistributionConfig"].get("Logging")
    assert logging, "the distribution has no access logging"
    assert logging["Bucket"].startswith(f"{ACCESS_LOGS_BUCKET_NAME}.s3."), logging


def test_ecs_exec_sessions_are_recorded(templates):
    """Operator shell sessions are logged to an encrypted log group.

    Without an explicit configuration ECS falls back to the task's awslogs
    settings, which the task role cannot write, so sessions go unrecorded.
    """
    cluster = _only(templates["compute"], "AWS::ECS::Cluster")
    config = cluster["Properties"]["Configuration"]["ExecuteCommandConfiguration"]
    assert config["Logging"] == "OVERRIDE", config
    assert config["LogConfiguration"]["CloudWatchEncryptionEnabled"] is True, config
    assert config["LogConfiguration"]["CloudWatchLogGroupName"], config
