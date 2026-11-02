# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CDK app entry point for AI Eval Studio cloud deployment.

Single environment per app — reads ``CDK_DEFAULT_ACCOUNT`` and
``CDK_DEFAULT_REGION`` from the environment. To deploy to a different account
or region, set those env vars before invoking ``cdk synth`` / ``cdk deploy``.

Two app-wide tags (``Project``, ``Environment``) and an AppRegistry
``Application`` are applied here. Both cascade to every resource in every
stack: the tags give us cost-allocation and resource grouping, the
AppRegistry application + per-stack associations light up Console Home
→ myApplications and the cross-service Application picker.
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from aws_cdk import Tags
from aws_cdk import aws_servicecatalogappregistry as svcatappreg
from aws_cdk import aws_servicecatalogappregistry_alpha as appreg

from agentcore_stack import AgentCoreStack
from auth_stack import AuthStack
from compute_stack import ComputeStack
from frontend_stack import FrontendStack
from iam_stack import IamStack
from network_stack import NetworkStack
from observability_stack import ObservabilityStack
from storage_stack import StorageStack

# Single environment. Multi-region failover is future work.
env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

app = cdk.App()

# App-wide tags — ``Tags.of(app)`` cascades to every taggable resource in
# every stack. Two are sufficient for cost allocation and console grouping;
# anything finer (Component=, Owner=) gets added when we exceed ~15 stacks.
Tags.of(app).add("Project", "ai-eval-studio")
Tags.of(app).add("Environment", os.environ.get("DEPLOY_ENV", "dev"))

# Stack dependency graph (auth no longer feeds network — auth is
# enforced in-app by the Fargate coordinator, not at the ALB):
#   auth      ── (user pool, spa_client) ──▶ iam, compute
#   network   ── (vpc, target_group) ──▶ storage, compute, frontend
#   storage   ── (buckets, table, EFS file system) ──▶ iam, compute
#   iam       ── (task roles) ──▶ compute
#   observability stands alone (log groups consumed by compute via construct ref)
#   compute   ── (worker_launch_env) ──▶ agentcore
#   frontend  ── (CloudFront DNS) ──▶ auth (callback URL via context var
#                                          — first deploy uses the default
#                                          ``cloudfront_dns`` context, then
#                                          ``dev-postdeploy.sh`` re-deploys
#                                          ``ai-eval-studio-auth`` with the
#                                          real CloudFront DNS).

auth_stack = AuthStack(app, "ai-eval-studio-auth", env=env)

network_stack = NetworkStack(
    app,
    "ai-eval-studio-network",
    env=env,
)

storage_stack = StorageStack(
    app,
    "ai-eval-studio-storage",
    env=env,
    vpc=network_stack.vpc,
)

iam_stack = IamStack(
    app,
    "ai-eval-studio-iam",
    env=env,
    sessions_bucket=storage_stack.sessions_bucket,
    uploads_staging_bucket=storage_stack.uploads_staging_bucket,
    snapshots_bucket=storage_stack.snapshots_bucket,
    table=storage_stack.table,
    file_system=storage_stack.file_system,
)

observability_stack = ObservabilityStack(app, "ai-eval-studio-observability", env=env)

compute_stack = ComputeStack(
    app,
    "ai-eval-studio-compute",
    env=env,
    vpc=network_stack.vpc,
    coordinator_role=iam_stack.coordinator_task_role,
    worker_role=iam_stack.worker_task_role,
    worker_execution_role=iam_stack.worker_execution_role,
    coordinator_target_group=network_stack.coordinator_target_group,
    coordinator_log_group=observability_stack.coordinator_log_group,
    # Cloud-mode env-var sources for the Fargate containers:
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

# Frontend: S3 bucket + CloudFront distribution serving the React SPA.
# CloudFront talks to the internal ALB via VPC origin, so the
# user-facing entry point is the CloudFront DNS, not the ALB.
frontend_stack = FrontendStack(
    app,
    "ai-eval-studio-frontend",
    env=env,
    alb=network_stack.alb,
)

# Realtime flows over the same CloudFront → ALB → Fargate path as REST
# (native FastAPI WebSocket route at ``/ws``). No separate WS stack.

# AgentCore: runtime for per-user session isolation.
# Additive stack that synthesizes AWS::BedrockAgentCore::Runtime alongside
# the Fargate coordinator without modifying compute_stack. Runtime role +
# execution role are created by IamStack. Built from the ARM64 orchestrator
# image (backend/Dockerfile.agentcore) with HTTP protocol, Cognito JWT
# authorizer, declarative EFS mount, VPC mode, and raised idle timeout.
# Deployed from a workstation with the infra/scripts helpers, which expect a
# podman machine (as on macOS); there is no CI deploy.
#
# AgentCore imports compute-owned worker-launch values (cross-stack exports),
# so compute cannot drop or change those exports — or be destroyed on its own
# — while agentcore imports them. Un-wind order: switch agentcore off the
# imports (or destroy agentcore entirely) and deploy, then drop the exports
# from compute. The coupling exists to avoid a weaker reference that would
# silently let agentcore reference a nonexistent worker task def.
agentcore_stack = AgentCoreStack(
    app,
    "ai-eval-studio-agentcore",
    env=env,
    vpc=network_stack.vpc,
    efs_access_point=storage_stack.access_point,
    sessions_bucket=storage_stack.sessions_bucket,
    uploads_staging_bucket=storage_stack.uploads_staging_bucket,
    snapshots_bucket=storage_stack.snapshots_bucket,
    table=storage_stack.table,
    scoped_session_role=iam_stack.scoped_session_role,
    # Dedicated AgentCore runtime role (bedrock-agentcore trust principal), NOT
    # the ecs-tasks-trusted coordinator role — the latter would fail create→READY.
    runtime_role=iam_stack.agentcore_runtime_role,
    user_pool=auth_stack.user_pool,
    spa_client=auth_stack.spa_client,
    worker_launch_env=compute_stack.worker_launch_env,
)

# AppRegistry — one Application + 8 stack associations. Auto-applies the
# ``awsApplication`` tag to every resource and lights up:
#   - Console Home → myApplications dashboard (cost, security, ops widgets)
#   - AWS Resource Groups (ai-eval-studio group, auto-generated)
#   - Service Catalog → AppRegistry → Applications
#   - Systems Manager → Application Manager
#   - The cross-service Application picker (top-right of any console).
#
# The Application construct must live in a Stack scope (it produces a real
# CFN resource ``AWS::ServiceCatalogAppRegistry::Application``); we put it in
# a dedicated tiny stack so its lifecycle is independent of the resource
# stacks. ``appregistry_alpha`` is alpha-tier; the underlying CFN resource
# is GA. If the alpha API drifts on a future aws-cdk-lib upgrade, fall back
# to the L1 ``CfnApplication`` construct.
appreg_stack = cdk.Stack(app, "ai-eval-studio-appregistry", env=env)
application = appreg.Application(
    appreg_stack,
    "Application",
    application_name="ai-eval-studio",
    description="AI Eval Studio — LLM evaluation platform",
)

# associate_application_with_stack creates a CFN ResourceAssociation
# pointing at each stack ARN, but the AppRegistry → myApplications →
# Resource Groups discovery chain frequently misses cascading
# sub-resources (Cognito sub-resources, certain IAM children, S3 bucket
# notifications, etc.) — those resources show up in the AWS account but
# do NOT appear in the myApplications view because they aren't tagged
# with ``awsApplication``.
#
# The reliable fix is to apply the ``awsApplication`` tag directly via
# ``Tags.of(stack)``: that cascade tags every taggable resource the
# stack creates, and AppRegistry indexes resources by that tag. The
# alpha L2 ``Application`` doesn't expose the tag value as a property,
# so reach through ``node.default_child`` to the underlying L1
# ``CfnApplication`` whose ``attr_application_tag_value`` is the
# CloudFormation token we need.
cfn_application = application.node.default_child
assert isinstance(cfn_application, svcatappreg.CfnApplication), (
    "Application's underlying L1 must be CfnApplication"
)
application_tag_value = cfn_application.attr_application_tag_value

for stack in (
    network_stack,
    storage_stack,
    auth_stack,
    iam_stack,
    observability_stack,
    compute_stack,
    frontend_stack,
    agentcore_stack,
):
    application.associate_application_with_stack(stack)
    # Belt-and-braces tagging: every resource in this stack gets the
    # awsApplication tag so myApplications can find it even when the
    # AppRegistry resource-discovery chain misses sub-resources.
    # ``exclude_resource_types=['aws:cdk:stack']`` keeps the unresolved
    # ``attr_application_tag_value`` token off the Stack itself (CFN
    # rejects the token as an invalid tag value) while still cascading
    # to every taggable child resource — which is what AppRegistry
    # actually indexes against.
    Tags.of(stack).add(
        "awsApplication",
        application_tag_value,
        exclude_resource_types=["aws:cdk:stack"],
    )

app.synth()
