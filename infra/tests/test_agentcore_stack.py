# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level test for the AgentCore Runtime.

Synthesizes the ISOLATED AgentCoreStack template (not the full app) and asserts:
- Exactly one AWS::BedrockAgentCore::Runtime resource
- ServerProtocol: HTTP
- Cognito JWT authorizer with discoveryUrl + allowedClients
- EFS mount at /mnt/projects via access-point config
- VPC mode with private subnets + security group
- Raised idleRuntimeSessionTimeout > 900s
- Container port 8080
- CLAUDE_CODE_USE_BEDROCK=1 in environmentVariables
- CfnOutput for runtime ARN

This test synthesizes ONLY the AgentCoreStack in a minimal hierarchy (not
the full app with ComputeStack) to avoid pulling in Docker assets; the
worker-launch values ComputeStack would supply are mocked. The combined
compute + agentcore synth is covered by test_microvm_worker_launch_parity.py
and test_cross_stack_export_stability.py.
"""

import json
from pathlib import Path

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Annotations, Match, Template


# A syntactically valid subnet id, supplied the way the operator supplies it
# at deploy: -c agentcore_subnet_ids=... The real value is account-specific
# and is discovered by dev-postdeploy.sh, never committed.
TEST_SUBNET_IDS = "subnet-0123456789abcdef0"

# Concrete env mirrors how app.py deploys (env=cdk.Environment(region=...)).
# The runtime name interpolates ``self.region`` with hyphens replaced by
# underscores; that replacement only resolves when the region is a concrete
# string, not the ``{Ref: AWS::Region}`` token an env-agnostic stack produces.
TEST_ENV = Environment(account="111111111111", region="us-east-1")


@pytest.fixture
def agentcore_template():
    """Synthesize the isolated AgentCoreStack template."""
    # agentcore_subnet_ids is REQUIRED context — supply it like deploy does.
    app = App(context={"agentcore_subnet_ids": TEST_SUBNET_IDS})

    # Import the stacks we need
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack
    from agentcore_stack import AgentCoreStack

    # Create minimal stack hierarchy (no ComputeStack — avoid Docker).
    # All stacks share one concrete env so cross-stack refs resolve and the
    # region-dependent runtime name synthesizes to a concrete string.
    network_stack = NetworkStack(app, "test-network", env=TEST_ENV)
    storage_stack = StorageStack(app, "test-storage", vpc=network_stack.vpc, env=TEST_ENV)
    auth_stack = AuthStack(app, "test-auth", env=TEST_ENV)
    iam_stack = IamStack(
        app,
        "test-iam",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
        env=TEST_ENV,
    )
    # Mock worker_launch_env (in real deployment this comes from ComputeStack)
    mock_worker_launch_env = {
        "ECS_CLUSTER_ARN": "arn:aws:ecs:us-east-1:111111111111:cluster/test-cluster",
        "WORKER_TASK_DEFINITION_ARN": "arn:aws:ecs:us-east-1:111111111111:task-definition/ai-eval-studio-worker:1",
        "WORKER_SUBNET_IDS": "subnet-12345,subnet-67890",
        "WORKER_SECURITY_GROUP_ID": "sg-12345",
    }
    agentcore_stack = AgentCoreStack(
        app,
        "test-agentcore",
        env=TEST_ENV,
        vpc=network_stack.vpc,
        efs_access_point=storage_stack.access_point,
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        scoped_session_role=iam_stack.scoped_session_role,
        runtime_role=iam_stack.agentcore_runtime_role,
        user_pool=auth_stack.user_pool,
        spa_client=auth_stack.spa_client,
        worker_launch_env=mock_worker_launch_env,
    )

    # Synthesize and return the AgentCore template
    cloud_assembly = app.synth()
    template_path = cloud_assembly.get_stack_by_name("test-agentcore").template_full_path
    with open(template_path, "r") as f:
        return json.load(f)


@pytest.fixture
def iam_template():
    """Synthesize the isolated IamStack template (for the runtime-role trust check)."""
    app = App()
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack

    network_stack = NetworkStack(app, "test-network2")
    storage_stack = StorageStack(app, "test-storage2", vpc=network_stack.vpc)
    auth_stack = AuthStack(app, "test-auth2")
    iam_stack = IamStack(
        app,
        "test-iam2",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
    )
    cloud_assembly = app.synth()
    template_path = cloud_assembly.get_stack_by_name("test-iam2").template_full_path
    with open(template_path, "r") as f:
        return json.load(f)


def test_exactly_one_agentcore_runtime(agentcore_template):
    """Exactly one AWS::BedrockAgentCore::Runtime resource."""
    runtimes = [
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    ]
    assert len(runtimes) == 1, f"Expected exactly 1 AgentCore Runtime, found {len(runtimes)}"


def test_server_protocol_http(agentcore_template):
    """ProtocolConfiguration == "HTTP"."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    protocol_config = runtime["Properties"].get("ProtocolConfiguration")
    assert protocol_config == "HTTP", f"Expected ProtocolConfiguration=HTTP, got {protocol_config}"


def test_cognito_jwt_authorizer(agentcore_template):
    """Cognito JWT authorizer with discoveryUrl + allowedClients."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    authz_config = runtime["Properties"].get("AuthorizerConfiguration", {})
    jwt_authz = authz_config.get("CustomJWTAuthorizer", {})

    # Check DiscoveryUrl contains .well-known/openid-configuration
    # (CDK serializes as Fn::Join with tokens, so check str representation)
    discovery_url = jwt_authz.get("DiscoveryUrl", "")
    discovery_url_str = str(discovery_url)
    assert ".well-known/openid-configuration" in discovery_url_str, (
        f"DiscoveryUrl should contain .well-known/openid-configuration, got {discovery_url_str}"
    )
    assert "cognito-idp" in discovery_url_str, (
        f"DiscoveryUrl should reference cognito-idp, got {discovery_url_str}"
    )

    # Check AllowedClients is non-empty
    allowed_clients = jwt_authz.get("AllowedClients", [])
    assert len(allowed_clients) > 0, "AllowedClients should be non-empty"


def test_efs_mount_at_projects(agentcore_template):
    """Filesystem configuration with mountPath == "/mnt/projects"."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    filesystem_configs = runtime["Properties"].get("FilesystemConfigurations", [])
    assert len(filesystem_configs) > 0, "FilesystemConfigurations should not be empty"

    # Check the first filesystem config has an EFS access point with MountPath
    fs_config = filesystem_configs[0]
    efs_ap_config = fs_config.get("EfsAccessPoint", {})
    mount_path = efs_ap_config.get("MountPath")
    assert mount_path == "/mnt/projects", (
        f"Expected MountPath=/mnt/projects, got {mount_path}"
    )

    # Check AccessPointArn is present (token reference is fine)
    access_point_arn = efs_ap_config.get("AccessPointArn")
    assert access_point_arn is not None, "AccessPointArn should be present"


def test_vpc_mode_with_private_subnets(agentcore_template):
    """VPC mode with private subnets + security group."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    network_config = runtime["Properties"].get("NetworkConfiguration", {})
    network_mode = network_config.get("NetworkMode")
    assert network_mode == "VPC", f"Expected NetworkMode=VPC, got {network_mode}"

    # Check VPC config has subnets + security groups
    vpc_config = network_config.get("NetworkModeConfig", {})
    subnets = vpc_config.get("Subnets", [])
    security_groups = vpc_config.get("SecurityGroups", [])

    assert len(subnets) > 0, "Subnets should be non-empty"
    assert len(security_groups) > 0, "SecurityGroups should be non-empty"


def test_subnets_are_exactly_the_context_ids(agentcore_template):
    """The runtime uses EXACTLY the context-supplied subnet IDs, not the full pool.

    Passing the whole PRIVATE_WITH_EGRESS pool can include an unsupported AZ-ID
    (use1-az6) and makes AgentCore reject the create. The stack must pass only
    the explicit, operator-supplied supported-AZ subnet IDs verbatim.
    """
    runtime = next(
        r for r in agentcore_template["Resources"].values()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    subnets = runtime["Properties"]["NetworkConfiguration"]["NetworkModeConfig"]["Subnets"]
    expected = [s.strip() for s in TEST_SUBNET_IDS.split(",") if s.strip()]
    assert subnets == expected, (
        f"Runtime subnets must be exactly the context IDs {expected}, got {subnets} "
        "(the full PRIVATE_WITH_EGRESS pool must NOT be passed)"
    )


def test_subnet_ids_echoed_as_output(agentcore_template):
    """The chosen subnet IDs are emitted as a CfnOutput for operator eyeballing."""
    outputs = agentcore_template.get("Outputs", {})
    subnet_output = next(
        (v for k, v in outputs.items() if "Subnet" in k), None
    )
    assert subnet_output is not None, "Expected a RuntimeSubnetIds output"
    assert TEST_SUBNET_IDS in str(subnet_output.get("Value")), (
        f"Subnet output should echo the chosen IDs, got {subnet_output.get('Value')}"
    )


def test_missing_subnet_context_warns_and_omits_the_runtime():
    """No agentcore_subnet_ids -> NO runtime, plus a warning naming the remedy.

    This check runs at SYNTH, and the value cannot be known until the VPC exists,
    so raising would abort ``cdk deploy --all`` on a fresh account before any
    stack deployed — making the repo undeployable for anyone who does not already
    have the subnet id. Warn-and-omit is the same trade-off, and the same shape,
    as network_stack's ``cloudfront_vpc_origin_sg_id``.

    The RuntimeArn output must also be absent, so ``dev-postdeploy.sh`` fails
    loudly on its read rather than wiring the coordinator to nothing.
    """
    app = App()  # no agentcore_subnet_ids context
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack
    from agentcore_stack import AgentCoreStack

    network_stack = NetworkStack(app, "test-network-noctx")
    storage_stack = StorageStack(app, "test-storage-noctx", vpc=network_stack.vpc)
    auth_stack = AuthStack(app, "test-auth-noctx")
    iam_stack = IamStack(
        app,
        "test-iam-noctx",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
    )
    mock_worker_launch_env = {
        "ECS_CLUSTER_ARN": "arn:aws:ecs:us-east-1:111111111111:cluster/test-cluster",
        "WORKER_TASK_DEFINITION_ARN": "arn:aws:ecs:us-east-1:111111111111:task-definition/ai-eval-studio-worker:1",
        "WORKER_SUBNET_IDS": "subnet-12345,subnet-67890",
        "WORKER_SECURITY_GROUP_ID": "sg-12345",
    }
    stack = AgentCoreStack(
            app,
            "test-agentcore-noctx",
            vpc=network_stack.vpc,
            efs_access_point=storage_stack.access_point,
            sessions_bucket=storage_stack.sessions_bucket,
            uploads_staging_bucket=storage_stack.uploads_staging_bucket,
            snapshots_bucket=storage_stack.snapshots_bucket,
            table=storage_stack.table,
            scoped_session_role=iam_stack.scoped_session_role,
            runtime_role=iam_stack.agentcore_runtime_role,
            user_pool=auth_stack.user_pool,
            spa_client=auth_stack.spa_client,
            worker_launch_env=mock_worker_launch_env,
    )

    template = Template.from_stack(stack)
    assert template.find_resources("AWS::BedrockAgentCore::Runtime") == {}, (
        "no subnet context must mean no runtime — creating one with the full "
        "PRIVATE_WITH_EGRESS pool would land on an unsupported AZ and AWS "
        "rejects the whole create request"
    )
    assert "RuntimeArn" not in template.find_outputs("*"), (
        "RuntimeArn must be absent so dev-postdeploy.sh's read fails loudly"
    )

    annotations = Annotations.from_stack(stack)
    annotations.has_warning("*", Match.string_like_regexp(".*agentcore_subnet_ids.*"))
    annotations.has_warning("*", Match.string_like_regexp(".*dev-postdeploy.*"))


def test_raised_idle_timeout(agentcore_template):
    """IdleRuntimeSessionTimeout present and > 900s."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    lifecycle_config = runtime["Properties"].get("LifecycleConfiguration", {})
    idle_timeout = lifecycle_config.get("IdleRuntimeSessionTimeout")
    assert idle_timeout is not None, "IdleRuntimeSessionTimeout should be present"
    assert idle_timeout > 900, (
        f"IdleRuntimeSessionTimeout should be > 900s, got {idle_timeout}"
    )


def test_container_uri_present(agentcore_template):
    """Container URI is present (port 8080 defined in Dockerfile CMD)."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    artifact = runtime["Properties"].get("AgentRuntimeArtifact", {})
    container_config = artifact.get("ContainerConfiguration", {})
    container_uri = container_config.get("ContainerUri")
    assert container_uri is not None, "ContainerUri should be present"
    # Port 8080 is defined in backend/Dockerfile.agentcore CMD, not in CDK config


def test_claude_code_use_bedrock_env(agentcore_template):
    """CLAUDE_CODE_USE_BEDROCK=1 in environmentVariables."""
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    env_vars = runtime["Properties"].get("EnvironmentVariables", {})

    assert "CLAUDE_CODE_USE_BEDROCK" in env_vars, (
        "CLAUDE_CODE_USE_BEDROCK should be in environmentVariables"
    )
    assert env_vars["CLAUDE_CODE_USE_BEDROCK"] == "1", (
        f"CLAUDE_CODE_USE_BEDROCK should be '1', got {env_vars['CLAUDE_CODE_USE_BEDROCK']}"
    )


def test_runtime_arn_output_exists(agentcore_template):
    """CfnOutput for runtime ARN exists."""
    outputs = agentcore_template.get("Outputs", {})
    assert "RuntimeArn" in outputs, "RuntimeArn output should exist"

    # Check it references the runtime's ARN attribute
    runtime_arn_output = outputs["RuntimeArn"]
    value = runtime_arn_output.get("Value")
    assert value is not None, "RuntimeArn output Value should be present"


def test_agentcore_runtime_role_trusts_agentcore_service(iam_template):
    """The AgentCore runtime role trusts bedrock-agentcore, NOT ecs-tasks.

    Reusing the ecs-tasks-trusted coordinator role would fail runtime
    create->READY, since AgentCore assumes the runtime role as the
    bedrock-agentcore.amazonaws.com service principal (verified against the
    AWS AgentCore Runtime IAM-permissions docs).
    """
    roles = [
        r for r in iam_template["Resources"].values()
        if r["Type"] == "AWS::IAM::Role"
        and r["Properties"].get("RoleName") == "ai-eval-studio-agentcore-runtime-role"
    ]
    assert len(roles) == 1, "Expected exactly one AgentCore runtime role"

    trust = roles[0]["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    principals = [
        s.get("Principal", {}).get("Service")
        for s in trust
        if s.get("Action") == "sts:AssumeRole"
    ]
    # Service principal may serialize as a string or a list.
    flat = []
    for p in principals:
        if isinstance(p, list):
            flat.extend(p)
        elif p is not None:
            flat.append(p)
    assert any("bedrock-agentcore" in str(p) for p in flat), (
        f"AgentCore runtime role must trust bedrock-agentcore, got {flat}"
    )
    assert not any("ecs-tasks" in str(p) for p in flat), (
        f"AgentCore runtime role must NOT trust ecs-tasks, got {flat}"
    )


def test_agentcore_runtime_role_has_ecr_and_logs(iam_template):
    """The runtime role carries the AgentCore ECR-pull + Logs execution perms."""
    template_str = json.dumps(iam_template)
    # ECR pull + AgentCore log group are the documented execution-role perms.
    assert "ecr:BatchGetImage" in template_str
    assert "/aws/bedrock-agentcore/runtimes/" in template_str


def test_runtime_name_matches_agentcore_regex(agentcore_template):
    """AgentRuntimeName matches AgentCore's ^[a-zA-Z][a-zA-Z0-9_]{0,47}$ (deploy-blocker).

    AgentCore rejects the create when the name contains hyphens; the region is
    interpolated with hyphens replaced by underscores. A synth-time regex check
    catches a regression before a doomed create.
    """
    import re

    runtime = next(
        r for r in agentcore_template["Resources"].values()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    name = runtime["Properties"].get("AgentRuntimeName")
    assert name is not None, "AgentRuntimeName should be present"
    assert re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,47}", str(name)), (
        f"AgentRuntimeName must match AgentCore's regex (no hyphens), got {name!r}"
    )


def test_authorization_header_forwarded_to_container(agentcore_template):
    """Authorization is on the request-header allowlist (deploy-blocker).

    The edge JWT authorizer validates + CONSUMES the Authorization header and
    does not pass it through by default, so the in-container per-project authz
    check would 401 with reason=missing-bearer. Allowlisting "Authorization"
    propagates the token to the container.
    """
    runtime = next(
        r for r in agentcore_template["Resources"].values()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    header_config = runtime["Properties"].get("RequestHeaderConfiguration", {})
    allowlist = header_config.get("RequestHeaderAllowlist", [])
    assert "Authorization" in allowlist, (
        f"RequestHeaderAllowlist must include 'Authorization', got {allowlist}"
    )


def test_agentcore_runtime_role_has_efs_discovery(iam_template):
    """The runtime role grants elasticfilesystem Describe{AccessPoints,MountTargets} (deploy-blocker).

    CreateAgentRuntime validates (at create time) that the execution role can
    DescribeAccessPoints + DescribeMountTargets whenever an EFS mount is
    attached. DescribeAccessPoints has no filesystem-ARN resource form, so it is
    granted on "*". Without this the runtime create fails.
    """
    template_str = json.dumps(iam_template)
    assert "elasticfilesystem:DescribeAccessPoints" in template_str, (
        "runtime role must grant elasticfilesystem:DescribeAccessPoints"
    )
    assert "elasticfilesystem:DescribeMountTargets" in template_str, (
        "runtime role must grant elasticfilesystem:DescribeMountTargets"
    )


def test_idle_timeout_is_exactly_3600(agentcore_template):
    """IdleRuntimeSessionTimeout is EXACTLY 3600 (warm-idle lever).

    The idle timeout is what keeps a session warm between turns, so it is pinned
    exactly. The exactness
    matters because the AWS default is 900s (15 min): if the LifecycleConfiguration
    ever drops or the value drifts, the runtime silently falls back to a 15-min
    idle window, quadrupling how often turns pay the cold recreate. The existing
    `test_raised_idle_timeout` only asserts `> 900`, which would not catch a
    regression to e.g. 1200 that shrinks the warm window — this test catches
    any drift from the 3600s value.
    """
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    lifecycle_config = runtime["Properties"].get("LifecycleConfiguration", {})
    idle_timeout = lifecycle_config.get("IdleRuntimeSessionTimeout")
    assert idle_timeout == 3600, (
        f"IdleRuntimeSessionTimeout must be EXACTLY 3600, got {idle_timeout}"
    )


def test_max_lifetime_ceiling_present_and_above_idle(agentcore_template):
    """MaxLifetime is present and strictly greater than the idle timeout (cold boundary).

    This documents the SECOND truly-cold boundary:
    a session is force-recycled (terminated → next turn is cold) either after the
    idle window elapses OR when it hits maxLifetime. Deployed value is 4h / 14400s;
    AWS default is 8h / 28800s. Both paths end in a TERMINATE + in-memory purge, so
    the post-boundary turn is a full microVM recreate + S3-transcript reload, not
    a warm resume.
    """
    runtime = next(
        r for rid, r in agentcore_template["Resources"].items()
        if r["Type"] == "AWS::BedrockAgentCore::Runtime"
    )
    lifecycle_config = runtime["Properties"].get("LifecycleConfiguration", {})
    idle_timeout = lifecycle_config.get("IdleRuntimeSessionTimeout")
    max_lifetime = lifecycle_config.get("MaxLifetime")

    assert max_lifetime is not None, "MaxLifetime should be present"
    assert idle_timeout is not None, "IdleRuntimeSessionTimeout should be present"
    assert max_lifetime > idle_timeout, (
        f"MaxLifetime ({max_lifetime}) must be > IdleRuntimeSessionTimeout ({idle_timeout})"
    )


class TestMmdsV2IsAssertedNotDeclared:
    """MMDSv2 has to be checked after deploy, because it cannot be declared.

    MMDS serves the runtime execution role's credentials to any code running in
    the microVM, the way IMDS does on EC2. MMDSv2 makes that token-first, which
    closes the remote (SSRF) path — not the local one, since a shell in the
    microVM can perform the token PUT itself.

    It is absent from this stack on purpose, and the purpose is checkable:
    ``AWS::BedrockAgentCore::Runtime`` has no ``MetadataConfiguration``
    property, so CloudFormation cannot carry it. The only API that can,
    ``UpdateAgentRuntime``, is a full-replace PUT requiring
    ``agentRuntimeArtifact`` and ``roleArn``, so a script that set the flag
    would have to restate the whole runtime and would clear anything it
    omitted. ``dev-postdeploy.sh`` therefore ASSERTS the flag instead.

    Without these tests the assertion is one grep away from being deleted as
    redundant, and its absence is silent: the app keeps working while the
    credentials become readable without a token.
    """

    SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dev-postdeploy.sh"

    @pytest.fixture(scope="class")
    def script(self) -> str:
        return self.SCRIPT.read_text()

    def test_the_stack_does_not_try_to_declare_it(self, agentcore_template):
        """A future edit adding MetadataConfiguration here would synthesize a
        template CloudFormation rejects. If this ever starts failing, AWS added
        the property and the post-deploy assertion can become a real setting."""
        runtime = next(
            r for r in agentcore_template["Resources"].values()
            if r["Type"] == "AWS::BedrockAgentCore::Runtime"
        )
        assert "MetadataConfiguration" not in runtime["Properties"], (
            "the runtime now declares MetadataConfiguration. If CloudFormation "
            "has gained the property, move the flag into the stack and drop the "
            "assertion from dev-postdeploy.sh — do not keep both."
        )

    def test_postdeploy_reads_the_flag_back(self, script: str):
        assert "get-agent-runtime" in script, (
            "dev-postdeploy.sh no longer reads the runtime back, so a runtime "
            "recreated without MMDSv2 would deploy silently"
        )
        assert "requireMMDSV2" in script, (
            "the script no longer inspects requireMMDSV2 specifically"
        )

    def test_the_assertion_fails_the_deploy(self, script: str):
        """Scoped to the MMDSv2 block. A repo-wide search for ``exit 1`` passes
        on any other section's — the failure this file's sibling documents."""
        start = script.index("# ---- Assert MMDSv2")
        end = script.index("# ---- Verify the API path actually answers")
        block = script[start:end]
        assert "exit 1" in block, (
            "the MMDSv2 check no longer exits non-zero, so it is a log line "
            "nobody reads"
        )
        assert "unreadable" in block, (
            "the check no longer fails closed — an AWS CLI without "
            "get-agent-runtime, or a role lacking GetAgentRuntime, would read "
            "as absent and pass"
        )

    def test_it_runs_before_the_api_probe(self, script: str):
        """Placement is load-bearing twice over: it must follow the agentcore
        leg (so a runtime ARN exists) and precede the API-probe block, whose
        tests extract everything up to the completion banner and assert
        ``exit 1`` appears in it. Nested inside, this block's own ``exit 1``
        would satisfy that assertion even if the probe's were deleted.
        """
        assert script.index("Reading AgentCore runtime ARN") < script.index(
            "# ---- Assert MMDSv2"
        ) < script.index("# ---- Verify the API path actually answers")
