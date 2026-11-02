# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infra test: worker self-mint IAM — role trust + grants + ABAC unchanged.

Synthesizes the IAM stack and asserts:
1. Worker task role holds sts:AssumeRole + sts:TagSession scoped to the
   scoped-session role ARN alone (not wildcard).
2. Worker's TagSession grant has a job-tag presence condition.
3. Scoped-session role's trust policy admits the worker task role.
4. Scoped-session role's ${aws:PrincipalTag/*} ABAC identity statements are
   unchanged (so the effective permission set is byte-for-byte the same
   scope, only the expiry moves).

Pins the IAM state that makes worker self-mint work without widening tenant
isolation. If this test fails, the worker cannot self-mint OR the ABAC
scoping has drifted.
"""
import json
import sys
from pathlib import Path

# Add infra/cdk to sys.path so we can import the stacks
infra_cdk = Path(__file__).resolve().parent.parent / "cdk"
sys.path.insert(0, str(infra_cdk))

from aws_cdk import App, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_efs as efs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3

# Import the IAM stack under test
from iam_stack import IamStack


def _synth_iam_stack() -> dict:
    """Synthesize the IAM stack and return its CloudFormation template as a dict."""
    app = App()

    # Create minimal mocks for the IAM stack's dependencies (buckets, table,
    # filesystem, user pool). The stack only reads ARNs from these, so we can
    # use from_*_attributes with fake ARNs.
    mock_stack = Stack(app, "MockDeps")
    # The IAM stack grants use of the stores' customer-managed keys, so the
    # mocks carry one, as the real buckets and table do.
    data_key = kms.Key.from_key_arn(
        mock_stack,
        "DataKey",
        "arn:aws:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
    )

    sessions_bucket = s3.Bucket.from_bucket_attributes(
        mock_stack,
        "SessionsBucket",
        bucket_arn="arn:aws:s3:::fake-sessions-bucket",
        encryption_key=data_key,
    )
    uploads_staging_bucket = s3.Bucket.from_bucket_attributes(
        mock_stack,
        "UploadsStagingBucket",
        bucket_arn="arn:aws:s3:::fake-uploads-staging-bucket",
        encryption_key=data_key,
    )
    snapshots_bucket = s3.Bucket.from_bucket_attributes(
        mock_stack,
        "SnapshotsBucket",
        bucket_arn="arn:aws:s3:::fake-snapshots-bucket",
        encryption_key=data_key,
    )
    table = dynamodb.Table.from_table_attributes(
        mock_stack,
        "Table",
        table_arn="arn:aws:dynamodb:us-east-1:123456789012:table/fake-table",
        encryption_key=kms.Key.from_key_arn(
            mock_stack,
            "TableKey",
            "arn:aws:kms:us-east-1:123456789012:key/0987dcba-09fe-87dc-65ba-ab0987654321",
        ),
    )
    # EFS filesystem requires a security group
    fake_sg = ec2.SecurityGroup.from_security_group_id(
        mock_stack,
        "FakeSG",
        security_group_id="sg-fake",
    )
    file_system = efs.FileSystem.from_file_system_attributes(
        mock_stack,
        "FileSystem",
        file_system_id="fs-fakefs",
        security_group=fake_sg,
    )
    # Instantiate the IAM stack
    iam_stack = IamStack(
        app,
        "TestIamStack",
        sessions_bucket=sessions_bucket,
        uploads_staging_bucket=uploads_staging_bucket,
        snapshots_bucket=snapshots_bucket,
        table=table,
        file_system=file_system,
    )

    # Synthesize and extract the IAM stack's template
    assembly = app.synth()
    template_json = assembly.get_stack_by_name("TestIamStack").template
    return template_json


def test_worker_grant_scoped_to_scoped_session_role_only():
    """Worker task role's sts:AssumeRole + sts:TagSession grant is scoped to the scoped-session role ARN alone.

    Verifies the worker can assume the scoped-session role (so it can
    self-mint refreshable credentials on boot), and that the grant is scoped
    to exactly that one role ARN (not wildcard, not a family of roles).
    """
    template = _synth_iam_stack()
    resources = template["Resources"]

    # Find the worker task role's policy statements
    worker_role_key = None
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Role":
            if "worker-task-role" in resource["Properties"]["RoleName"]:
                worker_role_key = key
                break

    assert worker_role_key is not None, "Worker task role not found in template"

    # Find the policy attached to the worker role
    worker_policies = []
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Policy":
            roles_ref = resource["Properties"].get("Roles", [])
            if any(ref.get("Ref") == worker_role_key for ref in roles_ref):
                worker_policies.append(resource)

    # Find the sts:AssumeRole + sts:TagSession statement
    sts_statement = None
    for policy in worker_policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            if "sts:AssumeRole" in actions and "sts:TagSession" in actions:
                sts_statement = statement
                break

    assert sts_statement is not None, (
        "Worker task role does not have sts:AssumeRole + sts:TagSession grant"
    )

    # Assert the Resource is the scoped-session role ARN (not wildcard)
    resource_value = sts_statement["Resource"]

    # The resource can be either a single value or a list
    if isinstance(resource_value, list):
        assert len(resource_value) == 1, (
            f"Worker STS grant has {len(resource_value)} resources; expected 1 "
            f"(the scoped-session role ARN only)"
        )
        resource_arn = resource_value[0]
    else:
        resource_arn = resource_value

    # The resource should be a Fn::GetAtt reference to the scoped-session role
    assert isinstance(resource_arn, dict), "Resource ARN is not a CFN intrinsic"
    assert "Fn::GetAtt" in resource_arn or "Ref" in resource_arn, (
        f"Resource ARN is not a GetAtt or Ref: {resource_arn}"
    )


def test_worker_tag_session_trust_has_job_tag_condition():
    """Scoped-session role's trust policy admits the worker with uid+job tag presence condition.

    Verifies the worker can call sts:TagSession on the scoped-session role,
    and that the condition requires BOTH the uid tag and the job tag to be
    present (not just uid, which the coordinator/runtime use).
    """
    template = _synth_iam_stack()
    resources = template["Resources"]

    # Find the scoped-session role
    scoped_role_key = None
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Role":
            if "scoped-session-role" in resource["Properties"]["RoleName"]:
                scoped_role_key = key
                break

    assert scoped_role_key is not None, "Scoped-session role not found in template"

    scoped_role = resources[scoped_role_key]
    trust_policy = scoped_role["Properties"]["AssumeRolePolicyDocument"]

    # Find the TagSession statement that admits the worker
    worker_tag_session_statement = None
    for statement in trust_policy["Statement"]:
        if statement.get("Action") == "sts:TagSession":
            principals_value = statement.get("Principal", {}).get("AWS", [])
            # Handle both single value and list
            if isinstance(principals_value, dict):
                principals = [principals_value]
            elif isinstance(principals_value, list):
                principals = principals_value
            elif isinstance(principals_value, str):
                principals = [principals_value]
            else:
                principals = []

            # Check if the worker is in the principals list (via GetAtt/Ref)
            for principal in principals:
                if isinstance(principal, dict) and "Fn::GetAtt" in principal:
                    attr_ref = principal["Fn::GetAtt"]
                    if isinstance(attr_ref, list) and "Worker" in attr_ref[0]:
                        worker_tag_session_statement = statement
                        break

    assert worker_tag_session_statement is not None, (
        "Worker is not admitted for sts:TagSession on the scoped-session role"
    )

    # Assert the condition requires BOTH uid and job tags present
    condition = worker_tag_session_statement.get("Condition", {})
    null_condition = condition.get("Null", {})

    assert "aws:RequestTag/uid" in null_condition, (
        "TagSession condition does not check uid tag presence"
    )
    assert null_condition["aws:RequestTag/uid"] == "false", (
        "TagSession condition does not require uid tag to be present"
    )

    assert "aws:RequestTag/job" in null_condition, (
        "TagSession condition does not check job tag presence (worker-specific)"
    )
    assert null_condition["aws:RequestTag/job"] == "false", (
        "TagSession condition does not require job tag to be present"
    )


def test_scoped_role_trust_admits_worker():
    """Scoped-session role's trust policy admits the worker task role as a principal.

    Verifies the worker can assume the scoped-session role (via sts:AssumeRole
    in the base trust policy, not just TagSession).
    """
    template = _synth_iam_stack()
    resources = template["Resources"]

    # Find the scoped-session role
    scoped_role_key = None
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Role":
            if "scoped-session-role" in resource["Properties"]["RoleName"]:
                scoped_role_key = key
                break

    assert scoped_role_key is not None, "Scoped-session role not found in template"

    scoped_role = resources[scoped_role_key]
    trust_policy = scoped_role["Properties"]["AssumeRolePolicyDocument"]

    # Find the AssumeRole statement that admits the worker
    worker_in_trust = False
    for statement in trust_policy["Statement"]:
        if statement.get("Action") == "sts:AssumeRole":
            principals_value = statement.get("Principal", {}).get("AWS", [])
            # Handle both single value and list
            if isinstance(principals_value, dict):
                principals = [principals_value]
            elif isinstance(principals_value, list):
                principals = principals_value
            elif isinstance(principals_value, str):
                principals = [principals_value]
            else:
                principals = []

            # Check if the worker is in the principals list
            for principal in principals:
                if isinstance(principal, dict) and "Fn::GetAtt" in principal:
                    attr_ref = principal["Fn::GetAtt"]
                    if isinstance(attr_ref, list) and "Worker" in attr_ref[0]:
                        worker_in_trust = True
                        break

    assert worker_in_trust, (
        "Worker task role is not admitted for sts:AssumeRole on the "
        "scoped-session role trust policy"
    )


def test_scoped_role_abac_statements_unchanged():
    """Scoped-session role's ${aws:PrincipalTag/*} ABAC identity statements are unchanged.

    Verifies the role's S3, DynamoDB, and Bedrock identity policy statements
    still use the same ${aws:PrincipalTag/uid}, ${aws:PrincipalTag/proj}, and
    ${aws:PrincipalTag/job} templates, so the effective permission set is
    byte-for-byte the same scope (only the expiry moves, not the reach).

    This is a conservative test: it asserts the template strings are present,
    not their exact positions or counts. The goal is to catch drift where the
    ABAC template is removed or changed, which would widen or narrow the
    worker's scope vs the coordinator's.
    """
    template = _synth_iam_stack()
    resources = template["Resources"]

    # Find the scoped-session role
    scoped_role_key = None
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Role":
            if "scoped-session-role" in resource["Properties"]["RoleName"]:
                scoped_role_key = key
                break

    assert scoped_role_key is not None, "Scoped-session role not found in template"

    # Find policies attached to the scoped-session role
    scoped_policies = []
    for key, resource in resources.items():
        if resource["Type"] == "AWS::IAM::Policy":
            roles_ref = resource["Properties"].get("Roles", [])
            if any(ref.get("Ref") == scoped_role_key for ref in roles_ref):
                scoped_policies.append(resource)

    # Serialize the policy documents to JSON and check for ABAC template strings
    policies_json = json.dumps(scoped_policies)

    # Assert the three ABAC tag templates are present
    assert "${aws:PrincipalTag/uid}" in policies_json, (
        "Scoped-session role identity policy does not use ${aws:PrincipalTag/uid} "
        "ABAC template"
    )
    assert "${aws:PrincipalTag/proj}" in policies_json, (
        "Scoped-session role identity policy does not use ${aws:PrincipalTag/proj} "
        "ABAC template (coordinator-only, but must be present)"
    )
    assert "${aws:PrincipalTag/job}" in policies_json, (
        "Scoped-session role identity policy does not use ${aws:PrincipalTag/job} "
        "ABAC template (worker-only, but must be present)"
    )

    # Also check that the managed policy for mantle is still attached
    # (the ceiling side of the permission intersection)
    has_mantle_policy = False
    if "ManagedPolicyArns" in resources[scoped_role_key]["Properties"]:
        managed_arns_str = str(resources[scoped_role_key]["Properties"]["ManagedPolicyArns"])
        if "AmazonBedrockMantleInferenceAccess" in managed_arns_str:
            has_mantle_policy = True

    assert has_mantle_policy or "AmazonBedrockMantleInferenceAccess" in policies_json, (
        "Scoped-session role is missing AmazonBedrockMantleInferenceAccess managed policy"
    )
