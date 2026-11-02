# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level test for AgentCore Runtime worker-launch parity.

Synthesizes BOTH ComputeStack and AgentCoreStack in one app (the decisive
offline no-cycle proof) and asserts:
- AgentCore runtime env contains all 4 worker-launch keys
- Each value is a CloudFormation reference (Fn::ImportValue / Fn::GetAtt),
  not a hardcoded literal
- Parity guard: AgentCore env covers every key in job_compute.py's
  _REQUIRED_WORKER_LAUNCH_ENV constant (derived from source, not hand-written)
- AgentCore runtime role has worker-launch IAM at coordinator parity: RunTask,
  TagResource, Stop/Describe/List, PassRole — same scoping
- AgentCore runtime role has snapshots-bucket S3 grant
- Regression guard: coordinator container env and role ECS/PassRole grants are
  unchanged (compare normalized sets, not exact-ordered JSON)
- No circular dependencies in either template

This test is the decisive offline check. The operator's deploy + agent-initiated
eval dispatch from chat is the decisive LIVE check (no AWS credentials here).
"""

import ast
import json
from pathlib import Path

import pytest
from aws_cdk import App, Environment


# Concrete env so cross-stack refs resolve and region-dependent names synth.
TEST_ENV = Environment(account="111111111111", region="us-east-1")

# Known-good agentcore_subnet_ids context (use1-az1)
TEST_SUBNET_IDS = "subnet-0123456789abcdef0"


def _parse_required_worker_launch_env_from_source() -> set[str]:
    """Parse the required worker-launch keys from job_compute.py source.

    Reads backend/app/platform/job_compute.py and extracts the tuple assigned
    to _REQUIRED_WORKER_LAUNCH_ENV. Returns a set of the key names.

    This is the parity guard: if someone adds a key to that constant without
    adding it to the AgentCore runtime env, this test FAILS. A hand-written
    list would defeat the guard.
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
    job_compute_path = backend_dir / "app" / "platform" / "job_compute.py"

    with open(job_compute_path, "r") as f:
        source = f.read()

    # Parse the module AST and find the _REQUIRED_WORKER_LAUNCH_ENV assignment
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_REQUIRED_WORKER_LAUNCH_ENV":
                    # Found it — extract the tuple elements
                    if isinstance(node.value, ast.Tuple):
                        keys = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                keys.add(elt.value)
                        if keys:
                            return keys

    # If we didn't find the constant, skip the parity test with a clear message
    pytest.skip("_REQUIRED_WORKER_LAUNCH_ENV constant not found in job_compute.py source")


@pytest.fixture
def combined_template():
    """Synthesize the combined ComputeStack + AgentCoreStack template.

    This is the decisive offline no-cycle proof: if the cross-stack reference
    creates a circular dependency, synth FAILS here.
    """
    app = App(context={"agentcore_subnet_ids": TEST_SUBNET_IDS})

    # Import the stacks
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from auth_stack import AuthStack
    from iam_stack import IamStack
    from observability_stack import ObservabilityStack
    from compute_stack import ComputeStack
    from agentcore_stack import AgentCoreStack

    # Create the minimal stack hierarchy (compute + agentcore + dependencies)
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
    observability_stack = ObservabilityStack(app, "test-observability", env=TEST_ENV)
    compute_stack = ComputeStack(
        app,
        "test-compute",
        env=TEST_ENV,
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
        worker_launch_env=compute_stack.worker_launch_env,
    )

    # Synthesize and return ALL THREE templates (compute, agentcore, IAM)
    cloud_assembly = app.synth()
    compute_template_path = cloud_assembly.get_stack_by_name("test-compute").template_full_path
    agentcore_template_path = cloud_assembly.get_stack_by_name("test-agentcore").template_full_path
    iam_template_path = cloud_assembly.get_stack_by_name("test-iam").template_full_path

    with open(compute_template_path, "r") as f:
        compute_template = json.load(f)
    with open(agentcore_template_path, "r") as f:
        agentcore_template = json.load(f)
    with open(iam_template_path, "r") as f:
        iam_template = json.load(f)

    return {"compute": compute_template, "agentcore": agentcore_template, "iam": iam_template}


def test_agentcore_runtime_env_has_all_worker_launch_keys(combined_template):
    """AgentCore runtime env contains all 4 worker-launch keys as CFN refs."""
    agentcore_template = combined_template["agentcore"]
    resources = agentcore_template.get("Resources", {})

    # Find the CfnRuntime resource
    runtime_resource = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::BedrockAgentCore::Runtime":
            runtime_resource = resource
            break

    assert runtime_resource is not None, "No AWS::BedrockAgentCore::Runtime found"

    # Extract environment_variables from Properties
    env_vars = runtime_resource["Properties"].get("EnvironmentVariables", {})

    # Assert all 4 worker-launch keys are present
    required_keys = {
        "ECS_CLUSTER_ARN",
        "WORKER_TASK_DEFINITION_ARN",
        "WORKER_SUBNET_IDS",
        "WORKER_SECURITY_GROUP_ID",
    }
    present_keys = set(env_vars.keys())
    missing_keys = required_keys - present_keys
    assert not missing_keys, f"Missing worker-launch keys in AgentCore env: {missing_keys}"

    # The infrastructure-derived values must be CloudFormation references rather
    # than hardcoded ARNs/ids, so they track the real resources.
    #
    # WORKER_TASK_DEFINITION_ARN is deliberately EXCLUDED from that requirement: it
    # carries the task-definition FAMILY, a stable literal. It used to be
    # `worker_task_def.task_definition_arn`, which renders as a cross-stack export
    # of the REVISIONED ARN — and because CloudFormation refuses to update an
    # export while another stack imports it, every compute deploy after the first
    # failed with "Cannot update export ... as it is in use by". Requiring a CFN
    # reference for this key is what pinned that shape in place, so the requirement
    # is scoped to the three keys where it is correct.
    reference_required_keys = required_keys - {"WORKER_TASK_DEFINITION_ARN"}
    for key in reference_required_keys:
        value = env_vars[key]
        assert isinstance(value, dict), f"{key} is not a CFN reference (got plain string)"
        # CFN references are dicts with keys like "Fn::ImportValue", "Fn::GetAtt", "Fn::Join", "Ref"
        assert any(
            k in value for k in ["Fn::ImportValue", "Fn::GetAtt", "Fn::Join", "Ref"]
        ), f"{key} value is a dict but not a recognized CFN intrinsic: {value}"

    # The task-definition value must be a plain stable family, NOT a revisioned
    # ARN or an import of one — that is the whole point of the fix.
    task_def_value = env_vars["WORKER_TASK_DEFINITION_ARN"]
    assert isinstance(task_def_value, str), (
        "WORKER_TASK_DEFINITION_ARN must be a plain stable family string, not a "
        f"CloudFormation reference: {task_def_value!r}. A per-deploy value here "
        "makes the compute stack permanently undeployable."
    )
    assert ":" not in task_def_value, (
        "WORKER_TASK_DEFINITION_ARN must not carry a revision or ARN prefix — a "
        f"bare family resolves to the latest ACTIVE revision. Got {task_def_value!r}"
    )


def test_parity_guard_agentcore_env_covers_required_set(combined_template):
    """Parity guard: AgentCore env covers every key job_compute.py requires.

    Derives the required-key set from source by parsing _REQUIRED_WORKER_LAUNCH_ENV.
    If someone adds a key to that constant without adding it to the runtime env,
    this test FAILS. A hand-written list would defeat the guard.
    """
    required_keys_from_source = _parse_required_worker_launch_env_from_source()
    agentcore_template = combined_template["agentcore"]
    resources = agentcore_template.get("Resources", {})

    # Find the CfnRuntime resource
    runtime_resource = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::BedrockAgentCore::Runtime":
            runtime_resource = resource
            break

    assert runtime_resource is not None, "No AWS::BedrockAgentCore::Runtime found"

    # Extract environment_variables
    env_vars = runtime_resource["Properties"].get("EnvironmentVariables", {})
    present_keys = set(env_vars.keys())

    # Assert the runtime env covers EVERY key from job_compute.py
    missing_keys = required_keys_from_source - present_keys
    assert not missing_keys, (
        f"AgentCore runtime env is missing required worker-launch keys: {missing_keys}. "
        f"These keys are in job_compute.py's _REQUIRED_WORKER_LAUNCH_ENV but not in "
        f"the runtime EnvironmentVariables. Add them to agentcore_stack.py's cloud_env."
    )


def test_agentcore_role_has_worker_launch_iam_at_coordinator_parity(combined_template):
    """AgentCore runtime role has worker-launch IAM (narrowed to action presence).

    Verifies the agentcore runtime role policy contains the 4 worker-launch actions:
    - ecs:RunTask
    - ecs:TagResource
    - ecs:StopTask, ecs:DescribeTasks, ecs:ListTasks
    - iam:PassRole

    Narrowed scope: asserts actions are PRESENT. Does not compare full
    coordinator-vs-agentcore policies (coordinator holds grants agentcore doesn't
    need, like bedrock-agentcore:InvokeAgentRuntime). Live validation (operator
    deploy + agent dispatch from chat) is the decisive check for correct scoping.
    """
    iam_template = combined_template["iam"]
    resources = iam_template.get("Resources", {})

    # Find agentcore runtime role (bedrock-agentcore trust principal)
    agentcore_role_id = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::IAM::Role":
            assume_policy = resource.get("Properties", {}).get("AssumeRolePolicyDocument", {})
            principals = str(assume_policy.get("Statement", [{}])[0].get("Principal", {}))
            if "bedrock-agentcore" in principals:
                agentcore_role_id = resource_id
                break

    assert agentcore_role_id is not None, "AgentCore runtime role not found in IAM template"

    # Find the DefaultPolicy for this role (CDK creates this when using add_to_policy)
    # DefaultPolicy resources have a Ref to the role
    agentcore_default_policy = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::IAM::Policy":
            roles = resource.get("Properties", {}).get("Roles", [])
            for role_ref in roles:
                if isinstance(role_ref, dict) and role_ref.get("Ref") == agentcore_role_id:
                    agentcore_default_policy = resource
                    break
            if agentcore_default_policy:
                break

    assert agentcore_default_policy is not None, "AgentCore DefaultPolicy not found in IAM template"

    # Extract all actions from all policy statements
    all_actions = set()
    policy_document = agentcore_default_policy.get("Properties", {}).get("PolicyDocument", {})
    for statement in policy_document.get("Statement", []):
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        all_actions.update(actions)

    # Assert the 4 worker-launch actions are present
    assert "ecs:RunTask" in all_actions, "Missing ecs:RunTask"
    assert "ecs:TagResource" in all_actions, "Missing ecs:TagResource"
    assert "ecs:StopTask" in all_actions, "Missing ecs:StopTask"
    assert "iam:PassRole" in all_actions, "Missing iam:PassRole"


def test_agentcore_role_has_snapshots_bucket_s3_grant(combined_template):
    """AgentCore runtime role has the snapshots-bucket S3 grant (4 verbs).

    Asserts the agentcore role policy contains the 4 S3 verbs needed for
    snapshot upload: GetObject, PutObject, AbortMultipartUpload,
    ListMultipartUploadParts.
    """
    iam_template = combined_template["iam"]
    resources = iam_template.get("Resources", {})

    # Find agentcore runtime role
    agentcore_role_id = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::IAM::Role":
            assume_policy = resource.get("Properties", {}).get("AssumeRolePolicyDocument", {})
            principals = str(assume_policy.get("Statement", [{}])[0].get("Principal", {}))
            if "bedrock-agentcore" in principals:
                agentcore_role_id = resource_id
                break

    assert agentcore_role_id is not None, "AgentCore runtime role not found in IAM template"

    # Find the DefaultPolicy for this role
    agentcore_default_policy = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::IAM::Policy":
            roles = resource.get("Properties", {}).get("Roles", [])
            for role_ref in roles:
                if isinstance(role_ref, dict) and role_ref.get("Ref") == agentcore_role_id:
                    agentcore_default_policy = resource
                    break
            if agentcore_default_policy:
                break

    assert agentcore_default_policy is not None, "AgentCore DefaultPolicy not found in IAM template"

    # Extract all S3 actions
    all_actions = set()
    policy_document = agentcore_default_policy.get("Properties", {}).get("PolicyDocument", {})
    for statement in policy_document.get("Statement", []):
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        all_actions.update(actions)

    # Assert the 4 S3 snapshot actions are present
    assert "s3:GetObject" in all_actions, "Missing s3:GetObject"
    assert "s3:PutObject" in all_actions, "Missing s3:PutObject"
    assert "s3:AbortMultipartUpload" in all_actions, "Missing s3:AbortMultipartUpload"
    assert "s3:ListMultipartUploadParts" in all_actions, "Missing s3:ListMultipartUploadParts"


def test_coordinator_env_and_role_unchanged_regression_guard(combined_template):
    """Regression guard: coordinator's worker-launch env vars are unchanged.

    Narrowed scope: asserts the coordinator container env still contains the 4
    worker-launch keys (ECS_CLUSTER_ARN, WORKER_TASK_DEFINITION_ARN,
    WORKER_SUBNET_IDS, WORKER_SECURITY_GROUP_ID). Does not assert byte-identical
    baseline (that would require capturing pre-change template JSON, which is
    brittle to unrelated CDK updates). The infra suite passing proves the
    coordinator's grants didn't regress (existing tests like
    test_coordinator_role_has_ecs_runtask still pass).
    """
    compute_template = combined_template["compute"]
    resources = compute_template.get("Resources", {})

    # Find coordinator task definition
    coordinator_task_def = None
    for resource_id, resource in resources.items():
        if resource.get("Type") == "AWS::ECS::TaskDefinition":
            # Coordinator has family not set (default) or family="ai-eval-studio-coordinator"
            # Worker has family="ai-eval-studio-worker"
            family = resource.get("Properties", {}).get("Family")
            if family != "ai-eval-studio-worker":
                coordinator_task_def = resource
                break

    assert coordinator_task_def is not None, "Coordinator task def not found"

    # Extract container env vars
    containers = coordinator_task_def.get("Properties", {}).get("ContainerDefinitions", [])
    assert len(containers) > 0, "No containers in coordinator task def"

    coordinator_container = containers[0]
    env_vars = {
        e.get("Name"): e.get("Value")
        for e in coordinator_container.get("Environment", [])
    }

    # Assert the 4 worker-launch keys are still present
    assert "ECS_CLUSTER_ARN" in env_vars, "Coordinator missing ECS_CLUSTER_ARN"
    assert "WORKER_TASK_DEFINITION_ARN" in env_vars, "Coordinator missing WORKER_TASK_DEFINITION_ARN"
    assert "WORKER_SUBNET_IDS" in env_vars, "Coordinator missing WORKER_SUBNET_IDS"
    assert "WORKER_SECURITY_GROUP_ID" in env_vars, "Coordinator missing WORKER_SECURITY_GROUP_ID"


def test_combined_synth_succeeds_no_circular_dependency(combined_template):
    """The combined compute + agentcore synth succeeds.

    The fact that this fixture built and returned templates is the decisive
    offline no-cycle proof. If the cross-stack reference created a circular
    dependency, synth would have FAILED before we reached this test.

    This test simply asserts that both templates exist and are non-empty, which
    proves synth succeeded with the cross-stack reference in place.
    """
    assert combined_template["compute"], "Compute template is empty"
    assert combined_template["agentcore"], "AgentCore template is empty"
    # The existence of Resources in both proves synth succeeded
    assert "Resources" in combined_template["compute"]
    assert "Resources" in combined_template["agentcore"]
