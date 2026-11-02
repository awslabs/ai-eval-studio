# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level test for the worker task definition + IAM grants.

Synthesizes the ComputeStack template and asserts:
- Worker task definition exists alongside the coordinator task def
- Worker container named "coordinator" (for RunTask override)
- CPU=2048, Memory=16384 (matches coordinator sizing)
- CfnOutputs for cluster/task-def/subnets/SG exist
- Coordinator role has ecs:RunTask/StopTask/DescribeTasks + iam:PassRole
- Worker task role has no Cognito/KMS-admin and no destructive verbs
- scoped_session_role MaxSessionDuration equals the STS role-chaining cap
- nothing trusts the scoped-session role through web identity
- Coordinator env has AGENTCORE_RUNTIME_ARN when context set
- Coordinator env omits AGENTCORE_RUNTIME_ARN when context unset
- Coordinator role has bedrock-agentcore:InvokeAgentRuntime permission
"""

import json
import re
from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3

# Test AgentCore runtime ARN for positive env tests
_TEST_AGENTCORE_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/ai_eval_studio_us_east_1-EXAMPLE1234"


@pytest.fixture
def compute_template():
    """Synthesize the ComputeStack template."""
    app = App()

    # Import the stacks we need
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack
    from observability_stack import ObservabilityStack
    from compute_stack import ComputeStack

    # Create minimal stack hierarchy
    network_stack = NetworkStack(app, "test-network")
    storage_stack = StorageStack(app, "test-storage", vpc=network_stack.vpc)
    auth_stack = AuthStack(app, "test-auth")
    iam_stack = IamStack(
        app,
        "test-iam",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
    )
    observability_stack = ObservabilityStack(app, "test-observability")
    compute_stack = ComputeStack(
        app,
        "test-compute",
        vpc=network_stack.vpc,
        coordinator_role=iam_stack.coordinator_task_role,
        worker_role=iam_stack.worker_task_role,
        worker_execution_role=iam_stack.worker_execution_role,
        coordinator_target_group=network_stack.coordinator_target_group,
        coordinator_log_group=observability_stack.coordinator_log_group,
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
        efs_access_point=storage_stack.access_point,
        scoped_session_role=iam_stack.scoped_session_role,
        user_pool=auth_stack.user_pool,
        spa_client=auth_stack.spa_client,
        user_pool_domain=auth_stack.user_pool_domain,
    )

    # Synthesize and return the compute template
    cloud_assembly = app.synth()
    template_path = cloud_assembly.get_stack_by_name("test-compute").template_full_path
    with open(template_path, "r") as f:
        return json.load(f)


@pytest.fixture
def compute_template_with_agentcore_arn():
    """Synthesize the ComputeStack template WITH agentcore_runtime_arn context."""
    app = App(context={"agentcore_runtime_arn": _TEST_AGENTCORE_ARN})

    # Import the stacks we need
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack
    from observability_stack import ObservabilityStack
    from compute_stack import ComputeStack

    # Create minimal stack hierarchy
    network_stack = NetworkStack(app, "test-network")
    storage_stack = StorageStack(app, "test-storage", vpc=network_stack.vpc)
    auth_stack = AuthStack(app, "test-auth")
    iam_stack = IamStack(
        app,
        "test-iam",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
    )
    observability_stack = ObservabilityStack(app, "test-observability")
    compute_stack = ComputeStack(
        app,
        "test-compute",
        vpc=network_stack.vpc,
        coordinator_role=iam_stack.coordinator_task_role,
        worker_role=iam_stack.worker_task_role,
        worker_execution_role=iam_stack.worker_execution_role,
        coordinator_target_group=network_stack.coordinator_target_group,
        coordinator_log_group=observability_stack.coordinator_log_group,
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
        efs_access_point=storage_stack.access_point,
        scoped_session_role=iam_stack.scoped_session_role,
        user_pool=auth_stack.user_pool,
        spa_client=auth_stack.spa_client,
        user_pool_domain=auth_stack.user_pool_domain,
    )

    # Synthesize and return the compute template
    cloud_assembly = app.synth()
    template_path = cloud_assembly.get_stack_by_name("test-compute").template_full_path
    with open(template_path, "r") as f:
        return json.load(f)


def test_worker_task_definition_exists(compute_template):
    """Worker task definition exists alongside coordinator."""
    task_defs = [
        r for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    assert len(task_defs) == 2, "Expected exactly 2 task definitions (coordinator + worker)"


def test_worker_container_name(compute_template):
    """Worker container is named 'coordinator' (for RunTask override)."""
    task_defs = [
        r for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    # Find the worker task def (the one with "Worker" in its logical ID)
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"

    container_defs = worker_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Worker task should have exactly 1 container"
    assert container_defs[0]["Name"] == "coordinator", "Worker container must be named 'coordinator'"


def test_worker_runs_an_init_process(compute_template):
    """The worker container runs an init process, so a StopTask can stop it.

    The worker's command is Python itself, and Linux drops a signal sent from
    outside the container to PID 1 unless PID 1 installed a handler — which the
    worker does not. Without the init process a cancel's SIGTERM did nothing and
    the worker ran until the SIGKILL at stopTimeout.
    """
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"

    container = worker_task_def["Properties"]["ContainerDefinitions"][0]
    assert container.get("LinuxParameters", {}).get("InitProcessEnabled") is True


def test_worker_sizing(compute_template):
    """Worker CPU=2048, Memory=16384 (matches coordinator)."""
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None

    assert worker_task_def["Properties"]["Cpu"] == "2048"
    assert worker_task_def["Properties"]["Memory"] == "16384"


def test_control_plane_outputs_exist(compute_template):
    """CfnOutputs for cluster, task-def, subnets, SG exist."""
    outputs = compute_template.get("Outputs", {})

    assert "ClusterArn" in outputs, "ClusterArn output missing"
    # Family, not the revisioned ARN — a named export carrying a per-deploy value
    # blocks every later update of this stack once anything imports it.
    assert "WorkerTaskDefinitionFamily" in outputs, (
        "WorkerTaskDefinitionFamily output missing"
    )
    assert "WorkerTaskDefinitionArn" not in outputs, (
        "WorkerTaskDefinitionArn export is back — it renders the revisioned ARN, "
        "which cannot be updated while another stack imports it"
    )
    assert "WorkerSubnetIds" in outputs, "WorkerSubnetIds output missing"
    assert "WorkerSecurityGroupId" in outputs, "WorkerSecurityGroupId output missing"


def test_both_task_defs_use_same_image(compute_template):
    """Both task definitions reference the same image asset hash."""
    task_defs = [
        (rid, r) for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    assert len(task_defs) == 2

    # Extract image URIs from both task defs
    images = []
    for rid, task_def in task_defs:
        container_defs = task_def["Properties"]["ContainerDefinitions"]
        images.append(container_defs[0]["Image"])

    # Both should reference the same ECR asset (the URI contains the same asset hash)
    # We can't assert exact equality because the URI includes account/region placeholders,
    # but we can assert that both have the asset ECR pattern and the same imageAssetXXX suffix
    assert len(images) == 2
    # In synth (not deployed), images are CDK token references like {"Fn::Sub": "..."}
    # Both should reference the SAME image token — assert they're identical
    assert images[0] == images[1], f"Both task defs should reference the same image asset: {images[0]} vs {images[1]}"


@pytest.fixture
def iam_template():
    """Synthesize the IamStack template."""
    app = App()

    # Import the stacks we need
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack

    # Create minimal stack hierarchy
    network_stack = NetworkStack(app, "test-network")
    storage_stack = StorageStack(app, "test-storage", vpc=network_stack.vpc)
    auth_stack = AuthStack(app, "test-auth")
    iam_stack = IamStack(
        app,
        "test-iam",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
    )

    # Synthesize and return the IAM template
    cloud_assembly = app.synth()
    template_path = cloud_assembly.get_stack_by_name("test-iam").template_full_path
    with open(template_path, "r") as f:
        return json.load(f)


def test_coordinator_role_has_ecs_runtask(iam_template):
    """Coordinator role has ecs:RunTask/StopTask/DescribeTasks + iam:PassRole."""
    # Find coordinator role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    # Extract all policy statements
    all_statements = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            all_statements.append(statement)

    # Check for ecs:RunTask
    ecs_actions = []
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        ecs_actions.extend([a for a in actions if a.startswith("ecs:")])

    assert "ecs:RunTask" in ecs_actions, "Coordinator missing ecs:RunTask"
    assert "ecs:StopTask" in ecs_actions, "Coordinator missing ecs:StopTask"
    assert "ecs:DescribeTasks" in ecs_actions, "Coordinator missing ecs:DescribeTasks"
    assert "ecs:ListTasks" in ecs_actions, "Coordinator missing ecs:ListTasks"

    # Check for iam:PassRole
    iam_actions = []
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        iam_actions.extend([a for a in actions if a.startswith("iam:")])

    assert "iam:PassRole" in iam_actions, "Coordinator missing iam:PassRole"


def test_coordinator_role_can_invoke_agentcore_runtime(iam_template):
    """Coordinator role has bedrock-agentcore:InvokeAgentRuntime permission."""
    # Find coordinator role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    # Extract all policy statements
    all_statements = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            all_statements.append(statement)

    # Flatten all actions (normalize single-string actions to list)
    all_actions = []
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        all_actions.extend(actions)

    # Assert bedrock-agentcore:InvokeAgentRuntime is present
    assert "bedrock-agentcore:InvokeAgentRuntime" in all_actions, \
        "Coordinator role should have bedrock-agentcore:InvokeAgentRuntime permission"


def test_worker_task_role_has_no_cognito(iam_template):
    """Worker task role has no cognito-idp:* actions."""
    # Find worker role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Worker" in rid
    ]
    assert len(policies) > 0, "Worker role policy not found"

    # Extract all actions
    all_actions = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            all_actions.extend(actions)

    # Assert no cognito-idp actions
    cognito_actions = [a for a in all_actions if a.startswith("cognito-idp:")]
    assert len(cognito_actions) == 0, f"Worker role should have no cognito-idp actions, found: {cognito_actions}"


def test_worker_task_role_has_no_destructive_verbs(iam_template):
    """Worker task role has no DeleteTable/DeleteBucket."""
    # Find worker role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Worker" in rid
    ]
    assert len(policies) > 0, "Worker role policy not found"

    # Extract all actions
    all_actions = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            all_actions.extend(actions)

    # Assert no destructive verbs
    destructive_verbs = ["DeleteTable", "DeleteBucket", "PutBucketPolicy", "CreateTable", "UpdateTable"]
    found_destructive = [v for v in destructive_verbs if any(v in a for a in all_actions)]
    assert len(found_destructive) == 0, f"Worker role should have no destructive verbs, found: {found_destructive}"


def test_scoped_session_role_max_duration_matches_the_chaining_cap(iam_template):
    """MaxSessionDuration equals the 3600s cap STS puts on a chained session.

    Every mint of this role is chained from a task role, and STS refuses a
    chained ``DurationSeconds`` above 3600 whatever MaxSessionDuration says.
    A larger value would only advertise a lifetime no credential can have.
    """
    scoped_role = next(
        (r for rid, r in iam_template["Resources"].items()
         if r["Type"] == "AWS::IAM::Role" and "ScopedSession" in rid),
        None,
    )
    assert scoped_role is not None, "ScopedSessionRole not found"
    assert scoped_role["Properties"].get("MaxSessionDuration") == 3600


def test_worker_task_def_family_matches_runtask_scope(compute_template, iam_template):
    """Worker task-def Family matches the RunTask resource scope prefix."""
    # Find worker task def
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"

    # Check Family
    family = worker_task_def["Properties"].get("Family")
    assert family == "ai-eval-studio-worker", f"Worker task-def Family should be 'ai-eval-studio-worker', got {family}"

    # Find coordinator role policy with RunTask
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    # Extract RunTask statement
    runtask_stmt = None
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            if "ecs:RunTask" in actions:
                runtask_stmt = statement
                break
        if runtask_stmt:
            break

    assert runtask_stmt is not None, "RunTask statement not found"

    # Check resource ARN contains the task-definition family prefix
    resources = runtask_stmt.get("Resource", [])
    if not isinstance(resources, list):
        resources = [resources]

    # The resource should be something like arn:aws:ecs:...:task-definition/ai-eval-studio-worker:*
    task_def_resource = next((r for r in resources if "task-definition/ai-eval-studio-worker" in str(r)), None)
    assert task_def_resource is not None, f"RunTask resource scope should contain 'task-definition/ai-eval-studio-worker', got {resources}"


def test_passrole_covers_both_worker_roles(iam_template):
    """PassRole statement covers both worker task role and worker execution role."""
    # Find coordinator role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    # Find PassRole statement
    passrole_stmt = None
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            if "iam:PassRole" in actions:
                passrole_stmt = statement
                break
        if passrole_stmt:
            break

    assert passrole_stmt is not None, "PassRole statement not found"

    # Check resources list
    resources = passrole_stmt.get("Resource", [])
    if not isinstance(resources, list):
        resources = [resources]

    # Should have 2 resources (worker task role + worker execution role)
    # The resources are Fn::GetAtt references, so we check the count
    assert len(resources) == 2, f"PassRole should cover 2 roles (task + execution), got {len(resources)}: {resources}"

    # Check that one is the task role and one is the execution role
    # by checking if their logical IDs contain "WorkerTaskRole" and "WorkerExecutionRole"
    resource_refs = []
    for r in resources:
        if isinstance(r, dict) and "Fn::GetAtt" in r:
            resource_refs.append(r["Fn::GetAtt"][0])

    has_task_role = any("WorkerTaskRole" in ref for ref in resource_refs)
    has_execution_role = any("WorkerExecutionRole" in ref for ref in resource_refs)
    assert has_task_role, "PassRole should cover WorkerTaskRole"
    assert has_execution_role, "PassRole should cover WorkerExecutionRole"


def test_worker_env_has_snapshots_bucket(compute_template):
    """Worker task def container environment includes S3_SNAPSHOTS_BUCKET."""
    # Find worker task def
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"

    # Check container environment
    container_defs = worker_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Worker task should have exactly 1 container"

    environment = container_defs[0].get("Environment", [])
    env_dict = {item["Name"]: item["Value"] for item in environment}

    assert "S3_SNAPSHOTS_BUCKET" in env_dict, "Worker environment should have S3_SNAPSHOTS_BUCKET"
    # Value should be a token/Ref, not the literal "eval-snapshots"
    snapshots_value = env_dict["S3_SNAPSHOTS_BUCKET"]
    assert snapshots_value != "eval-snapshots", "S3_SNAPSHOTS_BUCKET should not be the hardcoded literal 'eval-snapshots'"


def test_worker_efs_mount_read_only(compute_template):
    """Worker EFS mount is read-only; coordinator mount is read-write."""
    # Find both task defs
    task_defs = [
        (rid, r) for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    assert len(task_defs) == 2, "Expected exactly 2 task definitions"

    # Find worker and coordinator task defs
    worker_task_def = next(
        (r for rid, r in task_defs if "Worker" in rid),
        None,
    )
    coordinator_task_def = next(
        (r for rid, r in task_defs if "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Check worker mount point is read-only
    worker_container = worker_task_def["Properties"]["ContainerDefinitions"][0]
    worker_mounts = worker_container.get("MountPoints", [])
    assert len(worker_mounts) == 1, "Worker should have exactly 1 mount point"
    assert worker_mounts[0].get("ReadOnly") is True, "Worker mount point should be ReadOnly=True"

    # Check coordinator mount point is read-write (False or absent)
    coordinator_container = coordinator_task_def["Properties"]["ContainerDefinitions"][0]
    coordinator_mounts = coordinator_container.get("MountPoints", [])
    assert len(coordinator_mounts) == 1, "Coordinator should have exactly 1 mount point"
    coordinator_readonly = coordinator_mounts[0].get("ReadOnly", False)
    assert coordinator_readonly is False, "Coordinator mount point should be ReadOnly=False"


def test_worker_role_has_no_efs_clientwrite(iam_template):
    """Worker role does NOT have elasticfilesystem:ClientWrite action."""
    # Find worker role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Worker" in rid
    ]
    assert len(policies) > 0, "Worker role policy not found"

    # Extract all actions
    all_actions = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            all_actions.extend(actions)

    # Assert no ClientWrite
    efs_write_actions = [a for a in all_actions if "elasticfilesystem:ClientWrite" in a]
    assert len(efs_write_actions) == 0, f"Worker role should NOT have elasticfilesystem:ClientWrite, found: {efs_write_actions}"


def test_coordinator_env_has_agentcore_runtime_arn_when_context_set(compute_template_with_agentcore_arn):
    """Coordinator env has AGENTCORE_RUNTIME_ARN when agentcore_runtime_arn context is set."""
    # Find coordinator task def (contains "Coordinator" but NOT "Worker")
    coordinator_task_def = next(
        (r for rid, r in compute_template_with_agentcore_arn["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Extract environment variables as a dict
    container_defs = coordinator_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Coordinator task should have exactly 1 container"
    environment = container_defs[0].get("Environment", [])
    env_dict = {item["Name"]: item["Value"] for item in environment}

    # Assert AGENTCORE_RUNTIME_ARN is present and equals the test ARN
    assert "AGENTCORE_RUNTIME_ARN" in env_dict, "Coordinator environment should have AGENTCORE_RUNTIME_ARN when context is set"
    assert env_dict["AGENTCORE_RUNTIME_ARN"] == _TEST_AGENTCORE_ARN, \
        f"AGENTCORE_RUNTIME_ARN should equal {_TEST_AGENTCORE_ARN}, got {env_dict['AGENTCORE_RUNTIME_ARN']}"


def test_coordinator_env_omits_agentcore_runtime_arn_without_context(compute_template):
    """Coordinator env omits AGENTCORE_RUNTIME_ARN when agentcore_runtime_arn context is unset."""
    # Find coordinator task def (contains "Coordinator" but NOT "Worker")
    coordinator_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Extract environment variables as a dict
    container_defs = coordinator_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Coordinator task should have exactly 1 container"
    environment = container_defs[0].get("Environment", [])
    env_dict = {item["Name"]: item["Value"] for item in environment}

    # Assert AGENTCORE_RUNTIME_ARN is NOT present
    assert "AGENTCORE_RUNTIME_ARN" not in env_dict, \
        f"Coordinator environment should NOT have AGENTCORE_RUNTIME_ARN when context is unset, found: {env_dict.get('AGENTCORE_RUNTIME_ARN')}"


def test_worker_env_omits_worker_launch_vars(compute_template):
    """Worker container env omits all 4 worker-launch vars.

    The worker process reads none of the 4 launch vars — only
    ``get_job_compute()`` in backend/app/platform/job_compute.py reads them,
    on the dispatching (coordinator/runtime) side.
    Worker receives JOB_ID/OWNER_ID/PROJECT_ID via RunTask overrides and
    uses base cloud_env only (S3_SNAPSHOTS_BUCKET, AWS_REGION, etc.).

    Giving the worker WORKER_TASK_DEFINITION_ARN = its own task definition
    ARN creates a self-referential CloudFormation resource (WorkerTaskDef
    depends on WorkerTaskDef), which passes synth but fails at changeset
    creation with a circular dependency error.

    This test asserts the worker container Environment omits all 4 vars,
    breaking the self-reference cycle.
    """
    # Find worker task def
    worker_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Worker" in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"

    # Extract environment variable names
    container_defs = worker_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Worker task should have exactly 1 container"
    environment = container_defs[0].get("Environment", [])
    env_names = {item["Name"] for item in environment}

    # Worker-launch vars the worker should NOT have
    forbidden_vars = {
        "ECS_CLUSTER_ARN",
        "WORKER_TASK_DEFINITION_ARN",
        "WORKER_SUBNET_IDS",
        "WORKER_SECURITY_GROUP_ID",
    }

    # Assert none of them are present
    present_forbidden = forbidden_vars & env_names
    assert len(present_forbidden) == 0, \
        f"Worker environment should NOT have worker-launch vars, found: {sorted(present_forbidden)}"


def test_compute_stack_has_no_dependency_cycle(compute_template):
    """ComputeStack template contains no resource dependency cycle.

    This is a general template-wide cycle detector using DFS with WHITE/GRAY/BLACK
    coloring. It catches circular dependencies offline (no AWS, no deploy) by
    building a directed graph of all resource dependencies:
    - Explicit DependsOn edges
    - Implicit dependencies via Ref and Fn::GetAtt to sibling resources
    - Self-references (e.g., a resource Ref'ing its own logical ID)

    A GRAY neighbor (including a self-loop) indicates a cycle. This test serves
    as a durable gate against the entire class of synth-passing but deploy-failing
    circular dependency regressions.

    Root cause prevented: Worker container environment included
    WORKER_TASK_DEFINITION_ARN = worker_task_def.task_definition_arn, creating
    a self-referential resource (WorkerTaskDef -> WorkerTaskDef). CDK synth
    succeeded but CloudFormation rejected at changeset creation with a circular
    dependency error.
    """
    resources = compute_template["Resources"]
    resource_ids = set(resources.keys())

    def extract_refs(obj):
        """Recursively extract all sibling resource references from a value.

        Returns: set of logical IDs referenced via Ref or Fn::GetAtt
        """
        refs = set()
        if isinstance(obj, dict):
            # Handle {"Ref": "LogicalId"}
            if "Ref" in obj:
                ref_id = obj["Ref"]
                if isinstance(ref_id, str) and ref_id in resource_ids:
                    refs.add(ref_id)
            # Handle {"Fn::GetAtt": ["LogicalId", "Attr"]} or {"Fn::GetAtt": "LogicalId.Attr"}
            if "Fn::GetAtt" in obj:
                get_att = obj["Fn::GetAtt"]
                if isinstance(get_att, list) and len(get_att) >= 1:
                    logical_id = get_att[0]
                    if isinstance(logical_id, str) and logical_id in resource_ids:
                        refs.add(logical_id)
                elif isinstance(get_att, str) and "." in get_att:
                    logical_id = get_att.split(".", 1)[0]
                    if logical_id in resource_ids:
                        refs.add(logical_id)
            # Recurse into nested structures
            for value in obj.values():
                refs.update(extract_refs(value))
        elif isinstance(obj, list):
            for item in obj:
                refs.update(extract_refs(item))
        return refs

    # Build dependency graph: resource_id -> set of dependencies
    graph = {}
    for res_id, resource in resources.items():
        deps = set()

        # Explicit DependsOn edges
        depends_on = resource.get("DependsOn")
        if depends_on:
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            for dep in depends_on:
                if dep in resource_ids:
                    deps.add(dep)

        # Implicit dependencies via Ref/GetAtt in Properties
        properties = resource.get("Properties", {})
        deps.update(extract_refs(properties))

        graph[res_id] = deps

    # DFS with WHITE/GRAY/BLACK coloring to detect cycles
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {res_id: WHITE for res_id in resource_ids}

    def dfs(node, path):
        """Visit node and recurse to neighbors. Returns cycle path if found, else None."""
        if color[node] == GRAY:
            # Found a back edge to a node in current path - cycle detected
            cycle_start_idx = path.index(node)
            cycle_path = path[cycle_start_idx:] + [node]
            return cycle_path
        if color[node] == BLACK:
            # Already fully explored
            return None

        color[node] = GRAY
        path.append(node)

        for neighbor in graph.get(node, set()):
            cycle = dfs(neighbor, path)
            if cycle:
                return cycle

        path.pop()
        color[node] = BLACK
        return None

    # Check all nodes (handles disconnected components)
    for start_node in resource_ids:
        if color[start_node] == WHITE:
            cycle = dfs(start_node, [])
            if cycle:
                cycle_str = " -> ".join(cycle)
                assert False, f"CFN dependency cycle detected: {cycle_str}"


def test_coordinator_bounds_inbound_websocket_messages(compute_template):
    """The coordinator's uvicorn caps inbound WebSocket messages far below 16 MiB.

    ``/ws`` authenticates on the first frame, after the upgrade, so the server
    buffers a whole message from a caller it has not authenticated yet and the
    route parses it as JSON. uvicorn's default ceiling is 16 MiB. Both edges are
    pinned: too high restores that exposure, too low refuses a real JWT. The
    image's own ``CMD`` must carry the same value, so running the image directly
    cannot quietly differ from the deployed command.
    """
    coordinator_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert coordinator_task_def is not None, "Coordinator task definition not found"
    command = coordinator_task_def["Properties"]["ContainerDefinitions"][0]["Command"]
    assert "--ws-max-size" in command, f"coordinator command sets no --ws-max-size: {command}"
    limit = int(command[command.index("--ws-max-size") + 1])
    assert 16 * 1024 <= limit <= 1024 * 1024, (
        f"--ws-max-size {limit} is outside [16 KiB, 1 MiB]"
    )

    dockerfile = Path(__file__).resolve().parents[2] / "backend" / "Dockerfile"
    cmd_line = next(
        line for line in dockerfile.read_text().splitlines() if line.startswith("CMD ")
    )
    image_command = json.loads(cmd_line[len("CMD "):])
    assert "--ws-max-size" in image_command, (
        f"backend/Dockerfile CMD sets no --ws-max-size: {image_command}"
    )
    assert int(image_command[image_command.index("--ws-max-size") + 1]) == limit, (
        "backend/Dockerfile CMD and the deployed coordinator command disagree on "
        "--ws-max-size"
    )


def test_coordinator_env_has_worker_launch_vars(compute_template):
    """Coordinator container env carries all 4 worker-launch vars.

    The coordinator process runs dispatch_eval / FargateTaskCompute in cloud
    mode and reads these vars to construct a RunTask call. CDK snapshots a
    container's environment at add_container time, so all 4 vars must be
    populated into cloud_env BEFORE the coordinator container is added.
    This is the regression guard against the env-ordering bug where vars
    added to cloud_env after add_container are not visible in the
    synthesized template.
    """
    # Find coordinator task def (contains "Coordinator" but NOT "Worker")
    coordinator_task_def = next(
        (r for rid, r in compute_template["Resources"].items()
         if r["Type"] == "AWS::ECS::TaskDefinition" and "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Extract environment variables as a set of names
    container_defs = coordinator_task_def["Properties"]["ContainerDefinitions"]
    assert len(container_defs) == 1, "Coordinator task should have exactly 1 container"
    environment = container_defs[0].get("Environment", [])
    env_names = {item["Name"] for item in environment}

    # Required worker-launch vars
    required_vars = {
        "ECS_CLUSTER_ARN",
        "WORKER_TASK_DEFINITION_ARN",
        "WORKER_SUBNET_IDS",
        "WORKER_SECURITY_GROUP_ID",
    }

    # Assert all 4 are present
    missing_vars = required_vars - env_names
    assert len(missing_vars) == 0, \
        f"Coordinator environment missing worker-launch vars: {sorted(missing_vars)}"


def test_scoped_session_role_can_invoke_bedrock(iam_template):
    """scoped_session_role grants bedrock:InvokeModel(+WithResponseStream).

    Regression guard: the worker harness invokes bedrock-runtime.converse
    under minted STS creds. A credential minted from scoped_session_role can do
    no more than the role's own policies allow (the chained mint passes session
    tags only), so the role must grant bedrock invoke or the call is denied.
    """
    # Find scoped_session_role
    scoped_role = next(
        (r for rid, r in iam_template["Resources"].items()
         if r["Type"] == "AWS::IAM::Role" and "ScopedSession" in rid),
        None,
    )
    assert scoped_role is not None, "ScopedSessionRole not found"

    # Find attached policies
    policies = [
        r for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "ScopedSession" in rid
    ]
    assert len(policies) > 0, "ScopedSessionRole policies not found"

    # Extract all policy statements
    all_statements = []
    for policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            all_statements.append(statement)

    # Flatten all actions
    all_actions = []
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        all_actions.extend(actions)

    # Assert bedrock invoke actions are present
    assert "bedrock:InvokeModel" in all_actions, \
        "scoped_session_role should have bedrock:InvokeModel permission"
    assert "bedrock:InvokeModelWithResponseStream" in all_actions, \
        "scoped_session_role should have bedrock:InvokeModelWithResponseStream permission"


# ---------------------------------------------------------------------------
# ABAC: scoped_session_role tag-templated identity policy shape.
#
# These synth-assertion tests prove the ROLE POLICY the session tags resolve
# against is correct — the other half of the ABAC contract (the integration
# tests prove the tags are correct). Without them, a future edit that drops the
# Null-gate or re-adds dynamodb:Scan would regress the tenant fence with the
# suite still green.
#
# NOTE: CDK renders resources/conditions carrying ${aws:PrincipalTag/...} inside
# Fn::Join arrays (the tokenized account/region ARNs), so the tag template is a
# substring inside a joined array element. These tests match on substring
# containment across the flattened JSON (json.dumps(statement)), not a bare
# top-level string.
# ---------------------------------------------------------------------------


def _scoped_session_role_statements(iam_template):
    """Return all inline-policy statements attached to scoped_session_role."""
    policies = [
        r for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "ScopedSession" in rid
    ]
    assert len(policies) > 0, "ScopedSessionRole policies not found"
    statements = []
    for policy in policies:
        statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return statements


def test_scoped_session_role_uses_principal_tag_templates(iam_template):
    """scoped_session_role identity policy uses ${aws:PrincipalTag/*} templates.

    ABAC proof: per-tenant scoping lives in the fixed role policy via
    ${aws:PrincipalTag/uid|proj|job} templates — NO per-tenant literals. The
    presence of the templates is the proof there are no per-tenant literal ids
    (a literal id would replace the template segment).
    """
    statements = _scoped_session_role_statements(iam_template)
    flat = json.dumps(statements)

    # All three tag templates must appear in the role policy
    assert "aws:PrincipalTag/uid" in flat, \
        "scoped_session_role policy missing ${aws:PrincipalTag/uid} template"
    assert "aws:PrincipalTag/proj" in flat, \
        "scoped_session_role policy missing ${aws:PrincipalTag/proj} template"
    assert "aws:PrincipalTag/job" in flat, \
        "scoped_session_role policy missing ${aws:PrincipalTag/job} template"


def test_scoped_session_role_excludes_dynamodb_scan(iam_template):
    """scoped_session_role DDB statement excludes Scan + carries LeadingKeys condition.

    Regression guard: dynamodb:Scan MUST stay EXCLUDED so the ForAllValues
    empty-set bypass cannot apply (all granted actions are partition-keyed). The
    statement must carry a ForAllValues:StringEquals LeadingKeys condition
    referencing the uid+job templates.
    """
    statements = _scoped_session_role_statements(iam_template)

    # Find the DDB statement (contains dynamodb: actions)
    ddb_stmt = None
    for stmt in statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if any(a.startswith("dynamodb:") for a in actions):
            ddb_stmt = stmt
            break
    assert ddb_stmt is not None, "scoped_session_role DDB statement not found"

    ddb_actions = ddb_stmt["Action"]
    if not isinstance(ddb_actions, list):
        ddb_actions = [ddb_actions]

    # Scan MUST be absent
    assert "dynamodb:Scan" not in ddb_actions, \
        "dynamodb:Scan must be EXCLUDED (empty-set ForAllValues bypass guard)"

    # The five partition-keyed actions MUST be present
    for action in [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
    ]:
        assert action in ddb_actions, f"scoped_session_role DDB statement missing {action}"

    # ForAllValues:StringEquals LeadingKeys condition referencing uid+job templates
    condition = ddb_stmt.get("Condition", {})
    assert "ForAllValues:StringEquals" in condition, \
        "DDB statement must use ForAllValues:StringEquals for multi-valued LeadingKeys"
    leading_keys = condition["ForAllValues:StringEquals"]["dynamodb:LeadingKeys"]
    assert "USER#${aws:PrincipalTag/uid}" in leading_keys, \
        f"LeadingKeys missing USER#${{aws:PrincipalTag/uid}}, got {leading_keys}"
    assert "JOB#${aws:PrincipalTag/job}" in leading_keys, \
        f"LeadingKeys missing JOB#${{aws:PrincipalTag/job}}, got {leading_keys}"
    assert "PROJECT_SEQ#${aws:PrincipalTag/proj}" in leading_keys, \
        f"LeadingKeys missing PROJECT_SEQ#${{aws:PrincipalTag/proj}}, got {leading_keys}"


def test_scoped_session_role_ddb_grant_is_table_only_no_index(iam_template):
    """The minted DDB grant must NOT cover ``/index/*``.

    Nothing on the minted path queries a GSI — the three ``IndexName=`` queries
    in the store are reconciler/recovery operations that run under the
    coordinator or AgentCore runtime role, both of which keep the control-plane
    grant. Granting the index here would also raise a question AWS docs do not
    settle: whether ``dynamodb:LeadingKeys`` is evaluated at all on a GSI query.
    If it is not, an index grant is a cross-tenant read; with no index grant
    the question cannot arise.
    """
    statements = _scoped_session_role_statements(iam_template)

    ddb_stmt = None
    for stmt in statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if any(a.startswith("dynamodb:") for a in actions):
            ddb_stmt = stmt
            break
    assert ddb_stmt is not None, "scoped_session_role DDB statement not found"

    resources = ddb_stmt["Resource"]
    if not isinstance(resources, list):
        resources = [resources]
    rendered = json.dumps(resources)
    assert "index/" not in rendered, (
        "minted DDB grant must be table-only; an index grant is unused by the "
        f"worker and reopens the LeadingKeys-on-GSI question. Got: {rendered}"
    )


def test_scoped_session_role_leading_keys_have_no_wildcard(iam_template):
    """Every DDB LeadingKeys entry is a tag-resolved equality template, never a wildcard.

    A ``PROJECT_SEQ#*`` entry (or any other wildcard) would let ANY worker bump
    ANY tenant's seq counter, corrupting another tenant's stream ordering. The
    seq counter's partition key is not under ``USER#{uid}``, so its authorization
    rests entirely on the ``${aws:PrincipalTag/proj}`` template resolving to the
    caller's own project. A wildcard would silently remove that binding while
    leaving the statement looking scoped.
    """
    statements = _scoped_session_role_statements(iam_template)

    ddb_stmt = None
    for stmt in statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if any(a.startswith("dynamodb:") for a in actions):
            ddb_stmt = stmt
            break
    assert ddb_stmt is not None, "scoped_session_role DDB statement not found"

    leading_keys = ddb_stmt["Condition"]["ForAllValues:StringEquals"]["dynamodb:LeadingKeys"]
    if not isinstance(leading_keys, list):
        leading_keys = [leading_keys]

    for entry in leading_keys:
        assert "*" not in entry, (
            f"LeadingKeys entry {entry!r} contains a wildcard — that permits "
            "cross-tenant writes. Entries must be tag-resolved equality templates."
        )
    # The condition operator must stay StringEquals: StringLike would make the
    # wildcard check above meaningless by reinterpreting '*' if one were added.
    assert "ForAllValues:StringLike" not in ddb_stmt.get("Condition", {}), \
        "DDB LeadingKeys must use StringEquals, not StringLike (wildcard semantics)"


def test_sessions_bucket_statements_gated_to_coordinator(iam_template):
    """Sessions-bucket statements require the job tag to be ABSENT.

    The worker now carries a ``proj`` session tag so it can allocate a durable
    project seq. That tag also resolves the sessions-bucket prefix template
    ``-mnt-projects-users-${uid}-projects-${proj}/*``, which previously was
    unreachable for the worker only INCIDENTALLY — because the worker had no
    ``proj`` tag at all. Without an explicit gate, adding ``proj`` silently grants
    the worker get/put/DELETE on its project's session transcripts.

    ``Null: {aws:PrincipalTag/job: "true"}`` requires the job tag to be absent,
    which is true only on the coordinator path, so the worker's effective reach
    is unchanged.

    Exactly one pair exists, templated on the ``proj`` tag, so a session reaches
    one project's transcripts. A statement that wildcards the project segment
    would let a session reach every project of its user, so none may exist.
    """
    statements = _scoped_session_role_statements(iam_template)

    def _s3(stmt) -> bool:
        actions = (
            stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
        )
        return any(a.startswith("s3:") for a in actions)

    def _null(stmt) -> dict:
        return stmt.get("Condition", {}).get("Null", {})

    chained = [
        s for s in statements
        if _s3(s)
        and _null(s).get("aws:PrincipalTag/job") == "true"
        and "aws:PrincipalTag/proj" not in _null(s)
        and "${aws:PrincipalTag/proj}" in json.dumps(s)
    ]
    user_wide = [
        s for s in statements
        if _s3(s) and "-projects-*" in json.dumps(s)
    ]

    assert len(chained) == 2, (
        "Expected exactly 2 chained sessions-bucket statements (object CRUD + "
        f"ListBucket) templated on the proj tag, got {len(chained)}"
    )
    assert user_wide == [], (
        "a sessions-bucket statement wildcards the project segment, so a "
        f"session can reach every project of its user: {user_wide}"
    )

    # Without the job-tag gate the worker's proj tag would resolve the prefix.
    for stmt in chained:
        condition = stmt.get("Condition", {})
        assert "Null" in condition, (
            f"Sessions statement {stmt.get('Action')} missing the Null job-tag gate — "
            "the worker's proj tag would reach session transcripts."
        )
        assert condition["Null"].get("aws:PrincipalTag/job") == "true", (
            "Sessions statement must require the job tag to be ABSENT "
            f'(Null aws:PrincipalTag/job == "true"), got {condition["Null"]}'
        )


def test_scoped_session_role_bedrock_worker_gated(iam_template):
    """bedrock:InvokeModel + bedrock-mantle:CreateInference are Null-job-gated (worker-only).

    Regression guard: the coordinator (no job tag) must NOT get bedrock via
    minted creds. The bedrock + mantle statements carry
    Condition.Null[aws:PrincipalTag/job] == "false" so they apply only when a
    job tag is present (worker path).
    """
    statements = _scoped_session_role_statements(iam_template)

    # Find the bedrock:InvokeModel statement
    bedrock_stmt = None
    mantle_stmt = None
    for stmt in statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if "bedrock:InvokeModel" in actions:
            bedrock_stmt = stmt
        if "bedrock-mantle:CreateInference" in actions:
            mantle_stmt = stmt

    assert bedrock_stmt is not None, "scoped_session_role bedrock:InvokeModel statement not found"
    assert mantle_stmt is not None, "scoped_session_role bedrock-mantle:CreateInference statement not found"

    # Both statements must carry the Null job=false worker-gate
    for name, stmt in [("bedrock", bedrock_stmt), ("mantle", mantle_stmt)]:
        condition = stmt.get("Condition", {})
        assert "Null" in condition, f"{name} statement missing Null condition (worker-gate)"
        assert condition["Null"].get("aws:PrincipalTag/job") == "false", \
            f"{name} statement Null condition should be aws:PrincipalTag/job=false, got {condition['Null']}"


def test_scoped_session_role_sessions_prefix_template(iam_template):
    """Sessions-bucket S3 statement resource uses the uid+proj prefix template.

    Regression guard: the coordinator sessions prefix must be
    -mnt-projects-users-${aws:PrincipalTag/uid}-projects-${aws:PrincipalTag/proj}/
    so the role policy resolves the coordinator's uid+proj tags to the exact
    SDK S3SessionStore prefix.
    """
    statements = _scoped_session_role_statements(iam_template)

    # Find the S3 statement whose flattened JSON contains the sessions prefix template
    expected = "-mnt-projects-users-${aws:PrincipalTag/uid}-projects-${aws:PrincipalTag/proj}/"
    matched = [s for s in statements if expected in json.dumps(s)]
    assert len(matched) > 0, \
        f"No scoped_session_role statement contains the sessions prefix template {expected}"


def test_scoped_session_role_listbucket_prefix_condition(iam_template):
    """Sessions ListBucket statement carries s3:prefix StringLike ABAC condition.

    Additive to test_scoped_session_role_sessions_prefix_template: the existing
    test substring-matches the flattened statements list, so it would still pass
    if the ListBucket Condition were dropped (the GetObject statement carries
    the same prefix substring). This test specifically pins the ListBucket
    s3:prefix StringLike condition that authorization evaluates when a session
    resumes (``S3SessionStore.load()`` lists the transcript prefix).

    The AgentCore TagSession grant is already pinned by
    test_coordinator_and_agentcore_can_tagsession — no duplication here.
    """
    statements = _scoped_session_role_statements(iam_template)

    # Find the ListBucket statement (Action includes s3:ListBucket)
    listbucket_stmt = None
    for stmt in statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if "s3:ListBucket" in actions:
            listbucket_stmt = stmt
            break

    assert listbucket_stmt is not None, \
        "scoped_session_role ListBucket statement not found"

    # Assert it carries a Condition.StringLike on s3:prefix
    condition = listbucket_stmt.get("Condition", {})
    assert "StringLike" in condition, \
        "ListBucket statement must have a StringLike condition (ABAC prefix gate)"

    prefix_condition = condition["StringLike"].get("s3:prefix")
    assert prefix_condition is not None, \
        "ListBucket StringLike condition must include s3:prefix"

    # Assert the prefix condition contains the ABAC template
    expected_template = "-mnt-projects-users-${aws:PrincipalTag/uid}-projects-${aws:PrincipalTag/proj}/*"
    # prefix_condition is a list in the CDK synth output
    if isinstance(prefix_condition, list):
        assert expected_template in prefix_condition, \
            f"s3:prefix condition missing ABAC template {expected_template}, got {prefix_condition}"
    else:
        assert expected_template == prefix_condition, \
            f"s3:prefix condition should be {expected_template}, got {prefix_condition}"


def test_scoped_session_role_snapshot_prefix_template(iam_template):
    """Snapshot-bucket S3 statement resource uses the job prefix template.

    Regression guard: the worker snapshot prefix must be
    -tmp-workspace-${aws:PrincipalTag/job}/ so the role policy resolves the
    worker's job tag to the exact snapshot key prefix.
    """
    statements = _scoped_session_role_statements(iam_template)

    expected = "-tmp-workspace-${aws:PrincipalTag/job}/"
    matched = [s for s in statements if expected in json.dumps(s)]
    assert len(matched) > 0, \
        f"No scoped_session_role statement contains the snapshot prefix template {expected}"


def test_scoped_session_role_trust_allows_tagsession(iam_template):
    """scoped_session_role trust policy grants sts:TagSession (plus AssumeRole).

    ABAC requires sts:TagSession on the trust so the caller can pass session
    tags at assume time. Both AssumeRole and TagSession must be present.
    """
    scoped_role = next(
        (r for rid, r in iam_template["Resources"].items()
         if r["Type"] == "AWS::IAM::Role" and "ScopedSession" in rid),
        None,
    )
    assert scoped_role is not None, "ScopedSessionRole not found"

    trust_statements = scoped_role["Properties"]["AssumeRolePolicyDocument"]["Statement"]

    # Flatten all trust actions
    trust_actions = []
    for stmt in trust_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        trust_actions.extend(actions)

    assert "sts:AssumeRole" in trust_actions, \
        "scoped_session_role trust should allow sts:AssumeRole"
    assert "sts:TagSession" in trust_actions, \
        "scoped_session_role trust should allow sts:TagSession (ABAC session tags)"


def test_coordinator_and_agentcore_can_tagsession(iam_template):
    """Coordinator + agentcore identity policies grant sts:TagSession on scoped role.

    Both caller principals must be able to pass session tags when assuming the
    scoped_session_role — sts:TagSession is required on the identity side too.
    """
    # Find all statements granting sts:TagSession
    tagsession_role_ids = set()
    for rid, r in iam_template["Resources"].items():
        if r["Type"] != "AWS::IAM::Policy":
            continue
        for stmt in r["Properties"]["PolicyDocument"]["Statement"]:
            actions = stmt.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            if "sts:TagSession" in actions:
                # Confirm the resource targets the scoped-session-role ARN
                resource = json.dumps(stmt.get("Resource", ""))
                assert "ScopedSessionRole" in resource, \
                    f"sts:TagSession in {rid} should target ScopedSessionRole ARN, got {resource}"
                tagsession_role_ids.add(rid)

    # Both coordinator and agentcore must carry the grant
    has_coordinator = any("Coordinator" in rid for rid in tagsession_role_ids)
    has_agentcore = any("AgentCore" in rid for rid in tagsession_role_ids)
    assert has_coordinator, \
        f"Coordinator role policy missing sts:TagSession grant (found in {tagsession_role_ids})"
    assert has_agentcore, \
        f"AgentCore role policy missing sts:TagSession grant (found in {tagsession_role_ids})"


def test_worker_uv_cache_dir_is_writable(compute_template):
    """Worker container UV_CACHE_DIR=/tmp/uv-cache; coordinator unchanged.

    Regression guard: the worker mounts EFS read-only, and
    uv run (invoked even under skip_sync=True) may write cache index/wheel
    metadata. Redirecting the worker's cache to writable ephemeral /tmp avoids
    EACCES. The coordinator keeps the shared warm cache on the RW EFS mount.
    """
    # Find both task defs
    task_defs = [
        (rid, r) for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    assert len(task_defs) == 2, "Expected exactly 2 task definitions"

    # Find worker and coordinator task defs
    worker_task_def = next(
        (r for rid, r in task_defs if "Worker" in rid),
        None,
    )
    coordinator_task_def = next(
        (r for rid, r in task_defs if "Coordinator" in rid and "Worker" not in rid),
        None,
    )
    assert worker_task_def is not None, "Worker task definition not found"
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Extract environment variables from both containers
    worker_container = worker_task_def["Properties"]["ContainerDefinitions"][0]
    worker_env = worker_container.get("Environment", [])
    worker_env_dict = {item["Name"]: item["Value"] for item in worker_env}

    coordinator_container = coordinator_task_def["Properties"]["ContainerDefinitions"][0]
    coordinator_env = coordinator_container.get("Environment", [])
    coordinator_env_dict = {item["Name"]: item["Value"] for item in coordinator_env}

    # Assert worker has UV_CACHE_DIR=/tmp/uv-cache
    assert "UV_CACHE_DIR" in worker_env_dict, "Worker environment should have UV_CACHE_DIR"
    assert worker_env_dict["UV_CACHE_DIR"] == "/tmp/uv-cache", \
        f"Worker UV_CACHE_DIR should be /tmp/uv-cache, got {worker_env_dict['UV_CACHE_DIR']}"

    # Assert coordinator still has UV_CACHE_DIR=/mnt/projects/.uv-cache
    assert "UV_CACHE_DIR" in coordinator_env_dict, "Coordinator environment should have UV_CACHE_DIR"
    assert coordinator_env_dict["UV_CACHE_DIR"] == "/mnt/projects/.uv-cache", \
        f"Coordinator UV_CACHE_DIR should be /mnt/projects/.uv-cache (unchanged), got {coordinator_env_dict['UV_CACHE_DIR']}"


def test_both_task_defs_are_arm64(compute_template):
    """Both task definitions render RuntimePlatform with ARM64/LINUX.

    Regression guard: Both Fargate task definitions must run on Graviton to
    match the ARM64 image asset and the ARM64-only AgentCore microVM the
    coordinator relays to. The synthesized CFN template should have
    RuntimePlatform: {CpuArchitecture: ARM64, OperatingSystemFamily: LINUX}
    on BOTH the worker and coordinator task definitions.
    """
    # Find all task definitions
    task_defs = [
        (rid, r) for rid, r in compute_template["Resources"].items()
        if r["Type"] == "AWS::ECS::TaskDefinition"
    ]
    assert len(task_defs) == 2, "Expected exactly 2 task definitions (coordinator + worker)"

    # Identify worker and coordinator by logical ID
    worker_task_def = None
    coordinator_task_def = None
    for rid, task_def in task_defs:
        if "Worker" in rid:
            worker_task_def = (rid, task_def)
        elif "Coordinator" in rid:
            coordinator_task_def = (rid, task_def)

    assert worker_task_def is not None, "Worker task definition not found"
    assert coordinator_task_def is not None, "Coordinator task definition not found"

    # Check worker task def RuntimePlatform
    worker_rid, worker_def = worker_task_def
    worker_platform = worker_def["Properties"].get("RuntimePlatform")
    assert worker_platform is not None, f"Worker task def ({worker_rid}) missing RuntimePlatform"
    assert worker_platform.get("CpuArchitecture") == "ARM64", \
        f"Worker task def ({worker_rid}) CpuArchitecture should be ARM64, got {worker_platform.get('CpuArchitecture')}"
    assert worker_platform.get("OperatingSystemFamily") == "LINUX", \
        f"Worker task def ({worker_rid}) OperatingSystemFamily should be LINUX, got {worker_platform.get('OperatingSystemFamily')}"

    # Check coordinator task def RuntimePlatform
    coordinator_rid, coordinator_def = coordinator_task_def
    coordinator_platform = coordinator_def["Properties"].get("RuntimePlatform")
    assert coordinator_platform is not None, f"Coordinator task def ({coordinator_rid}) missing RuntimePlatform"
    assert coordinator_platform.get("CpuArchitecture") == "ARM64", \
        f"Coordinator task def ({coordinator_rid}) CpuArchitecture should be ARM64, got {coordinator_platform.get('CpuArchitecture')}"
    assert coordinator_platform.get("OperatingSystemFamily") == "LINUX", \
        f"Coordinator task def ({coordinator_rid}) OperatingSystemFamily should be LINUX, got {coordinator_platform.get('OperatingSystemFamily')}"


def test_runtask_and_tagresource_on_separate_statements(iam_template):
    """ecs:RunTask and ecs:TagResource are on separate statements with correct resource scopes.

    Regression guard for the two ways RunTask authorization breaks:
    - ecs:RunTask's resource ARN must match family:revision (worker:*), not family-hyphen-wildcard (worker-*).
    - ecs:TagResource must be on a SEPARATE statement scoped to resources=["*"]
      with Condition: {StringEquals: {ecs:CreateAction: RunTask}}. This is the AWS-canonical
      "grant tag-on-create" form, immune to task-ARN drift and no cluster-name ref needed.

    This test asserts:
    - ecs:RunTask is on a statement with task-definition/ai-eval-studio-worker:* resource
    - ecs:TagResource is on a SEPARATE statement with resources=["*"] + ecs:CreateAction=RunTask condition
    - The colon-revision pattern (worker:*) is present, NOT hyphen-wildcard (worker-*)
    """
    # Find coordinator role policy
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    # Extract all policy statements
    all_statements = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            all_statements.append(statement)

    # Find the RunTask statement
    runtask_stmt = None
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if "ecs:RunTask" in actions:
            runtask_stmt = stmt
            break

    assert runtask_stmt is not None, "RunTask statement not found"

    # Assert RunTask statement does NOT include TagResource (should be separate)
    runtask_actions = runtask_stmt.get("Action", [])
    if not isinstance(runtask_actions, list):
        runtask_actions = [runtask_actions]
    assert "ecs:TagResource" not in runtask_actions, \
        "ecs:TagResource should NOT be in the same statement as ecs:RunTask (must be on a separate statement with resources=['*'] and an ecs:CreateAction condition)"

    # Check RunTask resource ARN is task-definition/ai-eval-studio-worker:* (colon-revision)
    runtask_resources = runtask_stmt.get("Resource", [])
    if not isinstance(runtask_resources, list):
        runtask_resources = [runtask_resources]

    # Flatten Fn::Join tokens to strings for substring checks
    resource_strs = []
    for r in runtask_resources:
        if isinstance(r, dict):
            resource_str = json.dumps(r)
        else:
            resource_str = str(r)
        resource_strs.append(resource_str)

    # Assert the correct colon-revision pattern is present
    has_colon_revision = any("ai-eval-studio-worker:*" in s for s in resource_strs)
    assert has_colon_revision, \
        f"RunTask resource should end with 'ai-eval-studio-worker:*' (colon-revision), got: {runtask_resources}"

    # Assert the broken hyphen-wildcard pattern is absent
    has_hyphen_wildcard = any("ai-eval-studio-worker-*" in s for s in resource_strs)
    assert not has_hyphen_wildcard, \
        f"RunTask resource should NOT contain 'ai-eval-studio-worker-*' (hyphen-wildcard), got: {runtask_resources}"

    # Find the TagResource statement (must be separate from RunTask)
    tagresource_stmt = None
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if "ecs:TagResource" in actions and stmt != runtask_stmt:
            tagresource_stmt = stmt
            break

    assert tagresource_stmt is not None, \
        "ecs:TagResource statement not found (must be on a separate statement from ecs:RunTask)"

    # Assert TagResource uses the AWS-canonical ecs:CreateAction=RunTask condition
    tagresource_condition = tagresource_stmt.get("Condition", {})
    assert "StringEquals" in tagresource_condition, \
        "TagResource statement should have a StringEquals condition"
    assert "ecs:CreateAction" in tagresource_condition["StringEquals"], \
        "TagResource condition should gate on ecs:CreateAction"
    assert tagresource_condition["StringEquals"]["ecs:CreateAction"] == "RunTask", \
        f"TagResource ecs:CreateAction should equal 'RunTask', got {tagresource_condition['StringEquals']['ecs:CreateAction']}"

    # Assert TagResource resource is "*" (or contains "*")
    tagresource_resources = tagresource_stmt.get("Resource", [])
    if not isinstance(tagresource_resources, list):
        tagresource_resources = [tagresource_resources]

    # Should be ["*"] or contain "*"
    has_wildcard = "*" in tagresource_resources or any(r == "*" for r in tagresource_resources)
    assert has_wildcard, \
        f"ecs:TagResource resource should be '*' when using ecs:CreateAction condition, got: {tagresource_resources}"

    # Assert TagResource does NOT contain task-definition (wrong resource type for tag-on-create)
    tagresource_strs = []
    for r in tagresource_resources:
        if isinstance(r, dict):
            resource_str = json.dumps(r)
        else:
            resource_str = str(r)
        tagresource_strs.append(resource_str)

    has_task_definition = any("task-definition" in s for s in tagresource_strs)
    assert not has_task_definition, \
        f"ecs:TagResource resource should NOT contain 'task-definition' (tag-on-create authorizes against the TASK via ecs:CreateAction, not task-definition), got: {tagresource_resources}"


def test_managed_mantle_policy_attached_to_all_four_roles(iam_template):
    """AWS-managed AmazonBedrockMantleInferenceAccess attached to 4 roles.

    Regression guard: the managed mantle policy must be attached
    to ALL 4 roles — scoped_session_role + coordinator/worker/agentcore
    (shared via _attach_shared_policies). The worker calls mantle with
    credentials minted from scoped_session_role (session tags only, no session
    policy), so that role must carry the grant or the worker's calls are denied.

    The managed policy replaces the hand-written bedrock-mantle:CreateInference/
    Get*/List* statements as the single AWS-maintained source of truth.
    """
    # Expected role logical-ID substrings
    role_searches = [
        ("ScopedSessionRole", "ScopedSessionRole"),
        ("CoordinatorTaskRole", "CoordinatorTaskRole"),
        ("WorkerTaskRole", "WorkerTaskRole"),
        ("AgentCoreRuntimeRole", "AgentCoreRuntimeRole"),
    ]

    # Find each role and check ManagedPolicyArns
    for search_substring, role_name in role_searches:
        role = next(
            (r for rid, r in iam_template["Resources"].items()
             if r["Type"] == "AWS::IAM::Role" and search_substring in rid),
            None,
        )
        assert role is not None, f"{role_name} not found in IAM template"

        # ManagedPolicyArns contains CDK token Fn::Join with AWS::Partition
        managed_policy_arns = role["Properties"].get("ManagedPolicyArns", [])
        # Serialize to JSON for substring check (handles Fn::Join tokens)
        arns_json = json.dumps(managed_policy_arns)

        assert "AmazonBedrockMantleInferenceAccess" in arns_json, \
            f"{role_name} should have managed policy AmazonBedrockMantleInferenceAccess attached, got: {managed_policy_arns}"


def test_scoped_session_mantle_create_inference_is_worker_gated(iam_template):
    """bedrock-mantle:CreateInference is tag-gated on scoped_session_role (ABAC).

    The coordinator/worker/agentcore TASK roles carry no hand-written, un-gated
    CreateInference statement; they get mantle through the managed policy. On
    scoped_session_role the inline grant stays, worker-only, behind a Null
    condition (aws:PrincipalTag/job=false).

    This test asserts CreateInference is present on scoped_session_role
    AND carries the worker-gate Null condition.
    """
    # Find the scoped_session_role inline policy
    role_policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy"
        and "ScopedSessionRole" in r["Properties"]["Roles"][0].get("Ref", "")
    ]
    assert len(role_policies) > 0, "No scoped_session_role inline policy found"

    # Flatten all statements from scoped_session_role's policy
    all_statements = []
    for rid, policy in role_policies:
        all_statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])

    # Find the statement with bedrock-mantle:CreateInference
    mantle_statements = [
        s for s in all_statements
        if "bedrock-mantle:CreateInference" in s.get("Action", [])
    ]
    assert len(mantle_statements) > 0, \
        "bedrock-mantle:CreateInference not found in scoped_session_role (should be present, tag-gated)"

    # Assert it has the worker-gate Null condition
    mantle_stmt = mantle_statements[0]
    assert "Condition" in mantle_stmt, \
        "bedrock-mantle:CreateInference statement missing Condition (should be Null-job-gated)"
    assert "Null" in mantle_stmt["Condition"], \
        "bedrock-mantle:CreateInference Condition missing Null operator"
    assert "aws:PrincipalTag/job" in mantle_stmt["Condition"]["Null"], \
        "bedrock-mantle:CreateInference Null condition missing aws:PrincipalTag/job"
    assert mantle_stmt["Condition"]["Null"]["aws:PrincipalTag/job"] == "false", \
        "bedrock-mantle:CreateInference Null condition should be 'false' (worker-only gate)"


def test_scoped_session_role_snapshot_grants_putobject_worker_gated(iam_template):
    """Worker snapshot statement grants s3:PutObject (Null-gated) for the reverse-sync leg.

    Regression guard for the reverse-sync leg: the worker's
    EFS mount is read-only, so a completed detached eval's ``output/`` results
    are uploaded to S3 (worker→S3→coordinator→EFS). The upload key falls under
    the SAME ``-tmp-workspace-${job}/*`` job-derived prefix the worker already
    reads its snapshot from, so the existing worker-only snapshot statement
    must grant ``s3:PutObject`` alongside ``s3:GetObject`` — while keeping the
    Null job-gate (a tagless coordinator assume gets nothing here) and the
    prefix template unchanged.
    """
    statements = _scoped_session_role_statements(iam_template)

    # The worker-only snapshot statement is the one whose flattened JSON
    # carries the job-derived snapshot prefix template.
    snapshot_prefix = "-tmp-workspace-${aws:PrincipalTag/job}/"
    snapshot_stmts = [s for s in statements if snapshot_prefix in json.dumps(s)]
    assert len(snapshot_stmts) > 0, (
        f"No scoped_session_role statement carries the snapshot prefix {snapshot_prefix}"
    )
    snapshot_stmt = snapshot_stmts[0]

    actions = snapshot_stmt["Action"]
    if not isinstance(actions, list):
        actions = [actions]

    # Both read (snapshot pull) and write (output reverse-sync) on the job prefix.
    assert "s3:GetObject" in actions, (
        "worker snapshot statement should retain s3:GetObject (snapshot pull)"
    )
    assert "s3:PutObject" in actions, (
        "worker snapshot statement should grant s3:PutObject (output reverse-sync leg)"
    )

    # The Null job-gate must still be present (worker-only).
    condition = snapshot_stmt.get("Condition", {})
    assert "Null" in condition, "worker snapshot statement missing Null job-gate"
    assert condition["Null"].get("aws:PrincipalTag/job") == "false", (
        f"snapshot statement Null gate should be aws:PrincipalTag/job=false, got {condition['Null']}"
    )


def test_coordinator_snapshot_statement_grants_getobject(iam_template):
    """Coordinator snapshot statement grants s3:GetObject for the download leg.

    Regression guard for the reverse-sync leg: the coordinator
    already holds PutObject/AbortMultipartUpload/ListMultipartUploadParts on the
    snapshots bucket (inbound snapshot upload). To download the worker-uploaded
    output object it also needs ``s3:GetObject`` on that same snapshot statement.

    The snapshot statement is uniquely identified on the coordinator policy as
    the S3 statement carrying ``s3:AbortMultipartUpload`` but NOT ``s3:ListBucket``
    / ``s3:DeleteObject`` (the shared data-plane S3 statement has those).
    """
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and "Coordinator" in rid
    ]
    assert len(policies) > 0, "Coordinator role policy not found"

    all_statements = []
    for rid, policy in policies:
        all_statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])

    snapshot_stmt = None
    for stmt in all_statements:
        actions = stmt.get("Action", [])
        if not isinstance(actions, list):
            actions = [actions]
        if (
            "s3:AbortMultipartUpload" in actions
            and "s3:ListBucket" not in actions
            and "s3:DeleteObject" not in actions
        ):
            snapshot_stmt = stmt
            break

    assert snapshot_stmt is not None, (
        "Coordinator snapshot statement (AbortMultipartUpload without "
        "ListBucket/DeleteObject) not found"
    )

    snapshot_actions = snapshot_stmt["Action"]
    if not isinstance(snapshot_actions, list):
        snapshot_actions = [snapshot_actions]
    assert "s3:GetObject" in snapshot_actions, (
        "Coordinator snapshot statement should grant s3:GetObject (worker-output download leg)"
    )


def test_account_list_regions_still_present(iam_template):
    """account:ListRegions IS still present in inline IAM::Policy statements.

    Regression guard against over-removal: the account:ListRegions
    statement is SEPARATE from the mantle block and is NOT covered by the
    AWS-managed AmazonBedrockMantleInferenceAccess policy. Dynamic region
    discovery needs it, so it must remain in the inline policy.

    This test flattens all actions and asserts account:ListRegions is present.
    """
    # Find all IAM::Policy resources
    policies = [
        (rid, r) for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy"
    ]
    assert len(policies) > 0, "No IAM::Policy resources found"

    # Flatten all actions from all inline policy statements
    all_actions = []
    for rid, policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if not isinstance(actions, list):
                actions = [actions]
            all_actions.extend(actions)

    # Assert account:ListRegions IS present
    has_list_regions = "account:ListRegions" in all_actions
    assert has_list_regions, \
        "account:ListRegions should still be present in inline IAM::Policy (NOT covered by managed mantle policy)"


# ---------------------------------------------------------------------------
# Tenant boundary: which roles hold the cross-tenant S3 + DynamoDB data plane.
#
# The worker container executes agent-authored harness code by design, so its
# task role is a tenant boundary rather than a "least-privilege floor". Any
# ambient S3-sessions/DynamoDB statement on it is reachable by that code —
# botocore falls back to the ECS credential endpoint the moment
# AWS_CONFIG_FILE is dropped, which mint_worker_credentials.py documents as
# its own recursion break — and would silently supersede the ABAC bound the
# scoped_session_role tests above assert.
#
# Those tests inspect only _scoped_session_role_statements, so they cannot see
# the task role. These are the symmetric assertions on the task-role side.
# ---------------------------------------------------------------------------


def _statements_for_role(iam_template, role_marker: str):
    """Return all inline-policy statements for policies matching a role marker."""
    policies = [
        r for rid, r in iam_template["Resources"].items()
        if r["Type"] == "AWS::IAM::Policy" and role_marker in rid
    ]
    assert len(policies) > 0, f"{role_marker} role policy not found"
    statements = []
    for policy in policies:
        statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return statements


def _actions_of(statement) -> list[str]:
    """Normalize a statement's Action to a list."""
    actions = statement.get("Action", [])
    return actions if isinstance(actions, list) else [actions]


def test_worker_role_has_no_ambient_s3_or_dynamodb(iam_template):
    """Worker task role grants NO ambient S3 or DynamoDB action.

    The worker reaches both services exclusively through the minted
    scoped-session credential (credential_process -> scoped_session_role),
    whose statements are ${aws:PrincipalTag/*}-templated. A statement here
    would bypass that bound entirely, because a bare task-role session
    carries no principal tags and so cannot be tenant-conditioned at all.
    """
    offenders = []
    for statement in _statements_for_role(iam_template, "Worker"):
        for action in _actions_of(statement):
            if action.startswith(("s3:", "dynamodb:")):
                offenders.append(action)

    assert offenders == [], (
        "Worker task role must hold no ambient S3/DynamoDB grant — those are "
        f"reachable by agent-authored harness code. Found: {sorted(set(offenders))}. "
        "Route the access through the minted scoped-session credential instead."
    )


def test_worker_role_retains_assume_scoped_session(iam_template):
    """Worker keeps sts:AssumeRole + sts:TagSession — its only ambient AWS call.

    Companion to the test above: removing the data plane must not remove the
    one grant the worker genuinely needs on its own role, or it can never mint
    the scoped credential and every eval fails at boot.
    """
    sts_actions = {
        action
        for statement in _statements_for_role(iam_template, "Worker")
        for action in _actions_of(statement)
        if action.startswith("sts:")
    }
    assert "sts:AssumeRole" in sts_actions, \
        f"Worker role must keep sts:AssumeRole, found: {sorted(sts_actions)}"
    assert "sts:TagSession" in sts_actions, \
        f"Worker role must keep sts:TagSession, found: {sorted(sts_actions)}"


@pytest.mark.parametrize("role_marker", ["Coordinator", "AgentCore"])
def test_control_plane_roles_keep_data_plane(iam_template, role_marker):
    """Coordinator + AgentCore runtime keep the cross-tenant data plane.

    Pins the other side of the boundary so the worker fix cannot be
    "generalized" into the control plane. These two roles serve every user's
    requests — ``require_access`` and the email sharing lookup read other
    users' records by definition — and have no credential_process chain to
    fall back on, so stripping them breaks the app rather than hardening it.
    """
    actions = {
        action
        for statement in _statements_for_role(iam_template, role_marker)
        for action in _actions_of(statement)
    }
    assert "s3:GetObject" in actions, \
        f"{role_marker} role lost its S3 data plane"
    assert "dynamodb:Query" in actions, \
        f"{role_marker} role lost its DynamoDB data plane"


def test_nothing_assumes_the_scoped_session_role_through_web_identity(iam_template):
    """The scoped-session role is reached only by chaining from a task role.

    A web-identity or federated trust statement would be a second route to the
    same credentials, so its appearance must be a deliberate, reviewed change.
    """
    role = next(
        (r for rid, r in iam_template["Resources"].items()
         if r["Type"] == "AWS::IAM::Role" and "ScopedSession" in rid),
        None,
    )
    assert role is not None, "ScopedSessionRole not found"
    for statement in role["Properties"]["AssumeRolePolicyDocument"]["Statement"]:
        actions = statement.get("Action", [])
        actions = actions if isinstance(actions, list) else [actions]
        assert "sts:AssumeRoleWithWebIdentity" not in actions, statement
        assert "Federated" not in statement.get("Principal", {}), statement
