# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test health-check + circuit-breaker tuning.

Verifies:
- Target group health check: interval 10s, healthy 2, unhealthy 2, path /api/health
- Target group deregistration delay: 30s
- Service health_check_grace_period: 60s
- Service deployment config: min 100%, max 200%, desiredCount=1, circuit_breaker rollback=True
"""

from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template


@pytest.fixture
def synth_templates():
    """Synthesize NetworkStack + ComputeStack templates."""
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

    network_template = Template.from_stack(network_stack)
    compute_template = Template.from_stack(compute_stack)

    return {"network": network_template, "compute": compute_template}


def test_target_group_health_check(synth_templates):
    """Target group has tuned health-check (interval 10s, healthy 2, unhealthy 2)."""
    network_template = synth_templates["network"]

    # Find the CoordinatorTargetGroup
    network_template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        {
            "HealthCheckIntervalSeconds": 10,
            "HealthyThresholdCount": 2,
            "UnhealthyThresholdCount": 2,
            "HealthCheckPath": "/api/health",
        }
    )


def test_target_group_deregistration_delay(synth_templates):
    """Target group has 30s deregistration delay."""
    from aws_cdk.assertions import Match

    network_template = synth_templates["network"]

    # Find the TargetGroupAttributes for deregistration_delay
    # Use array_with to match just the element we care about (CDK may add others)
    network_template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        {
            "TargetGroupAttributes": Match.array_with([
                {
                    "Key": "deregistration_delay.timeout_seconds",
                    "Value": "30"
                }
            ])
        }
    )


def test_service_health_check_grace_period(synth_templates):
    """Coordinator service has 60s health_check_grace_period."""
    compute_template = synth_templates["compute"]

    # Find the CoordinatorService
    compute_template.has_resource_properties(
        "AWS::ECS::Service",
        {
            "HealthCheckGracePeriodSeconds": 60,
        }
    )


def test_service_deployment_config_rolling_update(synth_templates):
    """Deployment config: desiredCount=1, min 100%, max 200%, circuit_breaker rollback=True."""
    compute_template = synth_templates["compute"]

    # Find the CoordinatorService and verify deployment config
    compute_template.has_resource_properties(
        "AWS::ECS::Service",
        {
            "DesiredCount": 1,
            "DeploymentConfiguration": {
                "MinimumHealthyPercent": 100,
                "MaximumPercent": 200,
                "DeploymentCircuitBreaker": {
                    "Enable": True,
                    "Rollback": True
                }
            }
        }
    )
