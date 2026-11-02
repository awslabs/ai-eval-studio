# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability stack: log group only.

Provides the log group so the coordinator Fargate task definition has a
stable destination for ``awslogs`` driver output. A CloudWatch dashboard
will be added when widgets are real (turn p50/p95, job duration, error
rate, Bedrock spend) — an empty placeholder dashboard creates noise in
the AWS Console without adding value, so it's left out for now.

Log groups are created with ``RemovalPolicy.RETAIN`` so a stack delete
doesn't take audit history with it, and are encrypted with a
customer-managed key (``log_group_key``), retained for the same reason.
"""

from __future__ import annotations

from aws_cdk import (
    RemovalPolicy,
    Stack,
    aws_iam as iam,
    aws_kms as kms,
    aws_logs as logs,
)
from constructs import Construct

#: Every log group this app names lives under this prefix, which is what the
#: logs keys' policies are scoped to.
LOG_GROUP_PREFIX = "/ai-eval-studio/"


def log_group_key(scope: Construct, construct_id: str, description: str) -> kms.Key:
    """Return a customer-managed key that CloudWatch Logs may use for this app.

    CloudWatch Logs encrypts with the key as its own service principal, not as
    the writer, so the writers need no KMS access; the key policy has to let
    the service in. That statement is bounded by the encryption context Logs
    sends, which names the log group, so the key works only for this app's log
    groups in this account and region.

    No alias, and retained, for the same reasons as the data keys in
    ``storage_stack.py``: the log groups are retained too.
    """
    stack = Stack.of(scope)
    key = kms.Key(
        scope,
        construct_id,
        description=description,
        enable_key_rotation=True,
        removal_policy=RemovalPolicy.RETAIN,
    )
    key.add_to_resource_policy(
        iam.PolicyStatement(
            sid="AllowCloudWatchLogsForThisApp",
            principals=[iam.ServicePrincipal(f"logs.{stack.region}.amazonaws.com")],
            actions=[
                "kms:Encrypt*",
                "kms:Decrypt*",
                "kms:ReEncrypt*",
                "kms:GenerateDataKey*",
                "kms:Describe*",
            ],
            resources=["*"],
            conditions={
                "ArnLike": {
                    "kms:EncryptionContext:aws:logs:arn": (
                        f"arn:aws:logs:{stack.region}:{stack.account}:"
                        f"log-group:{LOG_GROUP_PREFIX}*"
                    )
                }
            },
        )
    )
    return key


class ObservabilityStack(Stack):
    """CloudWatch log groups for app + agent telemetry.

    Attributes:
        coordinator_log_group: Consumed by the coordinator Fargate task.
        logs_key: Customer-managed key the coordinator log group is encrypted
            with.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.logs_key = log_group_key(
            self, "LogsKey", "AI Eval Studio: coordinator and worker logs"
        )
        self.coordinator_log_group = logs.LogGroup(
            self,
            "CoordinatorLogGroup",
            log_group_name=f"{LOG_GROUP_PREFIX}coordinator",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
            encryption_key=self.logs_key,
        )
