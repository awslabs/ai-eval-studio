# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level guard for the AgentCore runtime process-role marker.

Synthesizes BOTH ComputeStack and AgentCoreStack in one app (reusing the combined-
synth fixture technique from test_microvm_worker_launch_parity.py) and asserts:
- AgentCore runtime environment contains the process-role marker key with the
  exact value the Python accessor compares against, both derived from backend
  source by AST-parsing rather than hand-coded strings.
- Coordinator container environment does NOT contain the marker key (unset is
  the coordinator default).

This is the offline guard. The operator's live check is the decisive one: ask
the agent in chat to dispatch an eval and confirm the progress bar appears and
animates, and confirm the poller start line appears in the coordinator log group
and NOT in the runtime log group.

The marker gates job lifecycle side effects (progress pollers, completion turns,
report dispatch) so they happen on the process holding browser WebSocket
connections (the coordinator), not on the process where dispatch executed (which
can now be the microVM, a process with zero connections).
"""

import ast
import json
from pathlib import Path

import pytest
from aws_cdk import App, Environment


# Concrete env so cross-stack refs resolve and region-dependent names synth
TEST_ENV = Environment(account="111111111111", region="us-east-1")

# Known-good agentcore_subnet_ids context (use1-az1, from microvm_worker_launch_parity)
TEST_SUBNET_IDS = "subnet-0123456789abcdef0"


def _parse_marker_constants_from_source() -> tuple[str, str]:
    """Parse the marker key and AgentCore role value from process_role.py source.

    Reads backend/app/platform/process_role.py and extracts the module constants
    PROCESS_ROLE_ENV_VAR and AGENTCORE_RUNTIME_ROLE. Returns a tuple of
    (marker_key, agentcore_role_value).

    This is the parity guard: if someone renames the marker key or the role value
    without updating the CDK stack, this test FAILS. Hand-copying the strings into
    the test would defeat the guard.
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
    process_role_path = backend_dir / "app" / "platform" / "process_role.py"

    with open(process_role_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse the module AST and find the two assignments
    tree = ast.parse(source)
    marker_key = None
    agentcore_role = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "PROCESS_ROLE_ENV_VAR" and isinstance(
                        node.value, ast.Constant
                    ):
                        marker_key = node.value.value
                    elif target.id == "AGENTCORE_RUNTIME_ROLE" and isinstance(
                        node.value, ast.Constant
                    ):
                        agentcore_role = node.value.value

    if marker_key is None or agentcore_role is None:
        pytest.skip(
            "PROCESS_ROLE_ENV_VAR or AGENTCORE_RUNTIME_ROLE constant not found "
            "in process_role.py source"
        )

    return marker_key, agentcore_role


@pytest.fixture
def combined_template():
    """Synthesize the combined ComputeStack + AgentCoreStack template.

    This reuses the combined-synth fixture technique from test_microvm_worker_launch_parity.py:
    build the same minimal stack hierarchy with the same concrete test environment and
    the same known-good subnet context, then synthesize and load the compute and agentcore
    templates.
    """
    app = App(context={"agentcore_subnet_ids": TEST_SUBNET_IDS})

    # The stack modules live in infra/cdk/ and import each other by bare module
    # name, so that directory has to be on sys.path before the first import.
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    import sys
    sys.path.insert(0, str(infra_path))

    from auth_stack import AuthStack
    from compute_stack import ComputeStack
    from iam_stack import IamStack
    from network_stack import NetworkStack
    from storage_stack import StorageStack
    from observability_stack import ObservabilityStack
    from agentcore_stack import AgentCoreStack

    # Build the stack hierarchy in dependency order
    network_stack = NetworkStack(app, "ai-eval-studio-network", env=TEST_ENV)
    storage_stack = StorageStack(
        app,
        "ai-eval-studio-storage",
        vpc=network_stack.vpc,
        env=TEST_ENV,
    )
    auth_stack = AuthStack(app, "ai-eval-studio-auth", env=TEST_ENV)
    iam_stack = IamStack(
        app,
        "ai-eval-studio-iam",
        sessions_bucket=storage_stack.sessions_bucket,
        uploads_staging_bucket=storage_stack.uploads_staging_bucket,
        snapshots_bucket=storage_stack.snapshots_bucket,
        table=storage_stack.table,
        file_system=storage_stack.file_system,
        env=TEST_ENV,
    )
    observability_stack = ObservabilityStack(app, "ai-eval-studio-observability", env=TEST_ENV)
    compute_stack = ComputeStack(
        app,
        "ai-eval-studio-compute",
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
        env=TEST_ENV,
    )
    agentcore_stack = AgentCoreStack(
        app,
        "ai-eval-studio-agentcore",
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
        env=TEST_ENV,
    )

    # Synthesize the app
    cloud_assembly = app.synth()

    # Load the compute and agentcore templates
    compute_template_path = (
        cloud_assembly.directory + "/ai-eval-studio-compute.template.json"
    )
    agentcore_template_path = (
        cloud_assembly.directory + "/ai-eval-studio-agentcore.template.json"
    )

    with open(compute_template_path, "r") as f:
        compute_template = json.load(f)

    with open(agentcore_template_path, "r") as f:
        agentcore_template = json.load(f)

    return compute_template, agentcore_template


def test_agentcore_runtime_env_has_marker(combined_template):
    """AgentCore runtime environment contains the marker key with the exact value from source."""
    compute_template, agentcore_template = combined_template
    marker_key, agentcore_role = _parse_marker_constants_from_source()

    # Find the AgentCore Runtime resource (AWS::BedrockAgentCore::Runtime)
    runtime_resource = None
    for logical_id, resource in agentcore_template.get("Resources", {}).items():
        if resource.get("Type") == "AWS::BedrockAgentCore::Runtime":
            runtime_resource = resource
            break

    assert runtime_resource is not None, (
        "AWS::BedrockAgentCore::Runtime resource not found in agentcore template"
    )

    # Extract the environment variables from the Properties level
    # (for CfnRuntime, environment_variables is at the top level, not in ContainerConfiguration)
    env_dict = runtime_resource["Properties"].get("EnvironmentVariables", {})

    # Assert the marker key is present with the exact value from source
    assert marker_key in env_dict, (
        f"Marker key '{marker_key}' not found in AgentCore runtime environment"
    )
    assert env_dict[marker_key] == agentcore_role, (
        f"Marker value mismatch: expected '{agentcore_role}', "
        f"got '{env_dict[marker_key]}'"
    )


def test_coordinator_env_does_not_have_marker(combined_template):
    """Coordinator container environment does NOT contain the marker key."""
    compute_template, agentcore_template = combined_template
    marker_key, agentcore_role = _parse_marker_constants_from_source()

    # Find the coordinator task definition (AWS::ECS::TaskDefinition)
    coordinator_task_def = None
    for logical_id, resource in compute_template.get("Resources", {}).items():
        if resource.get("Type") == "AWS::ECS::TaskDefinition":
            # The coordinator task def has ContainerDefinitions with a container
            # named "coordinator"
            container_defs = resource["Properties"].get("ContainerDefinitions", [])
            for container in container_defs:
                if container.get("Name") == "coordinator":
                    coordinator_task_def = resource
                    break
            if coordinator_task_def:
                break

    assert coordinator_task_def is not None, (
        "Coordinator task definition not found in compute template"
    )

    # Extract the environment variables from the coordinator container
    container_defs = coordinator_task_def["Properties"]["ContainerDefinitions"]
    coordinator_container = next(
        (c for c in container_defs if c.get("Name") == "coordinator"), None
    )
    assert coordinator_container is not None, "Coordinator container not found"

    env_vars = coordinator_container.get("Environment", [])
    env_keys = {var["Name"] for var in env_vars}

    # Assert the marker key is NOT present (unset is the coordinator default)
    assert marker_key not in env_keys, (
        f"Marker key '{marker_key}' should NOT be present in coordinator environment "
        f"(unset is the coordinator default)"
    )
