# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The coordinator can tell a run output still uploading from one it may not read.

A completed detached eval's results travel worker → S3 → coordinator, and the
coordinator's read starts before the worker's upload lands: the harness reports
its results first, and the download retries until the object is there. So it
routinely asks S3 for a key that does not exist yet. S3 answers that with 404
only to a caller holding ``s3:ListBucket`` on the bucket, and with 403
AccessDenied to anyone else, which the download loop treats as a real failure
and logs on every completed eval.

Nothing in the code lists the bucket, so the grant reads as unused. This pins
it, on the bucket itself: ``s3:ListBucket`` is a bucket-level action, and
granted on the ``/*`` object resource it would not apply.
"""
from .test_worker_selfmint_iam import _synth_iam_stack

SNAPSHOTS_BUCKET_ARN = "arn:aws:s3:::fake-snapshots-bucket"


def _coordinator_statements() -> list[dict]:
    """Every policy statement attached to the coordinator task role."""
    resources = _synth_iam_stack()["Resources"]
    role_key = next(
        key
        for key, resource in resources.items()
        if resource["Type"] == "AWS::IAM::Role"
        and resource["Properties"].get("RoleName") == "ai-eval-studio-coordinator-task-role"
    )
    return [
        statement
        for resource in resources.values()
        if resource["Type"] == "AWS::IAM::Policy"
        and any(ref.get("Ref") == role_key for ref in resource["Properties"].get("Roles", []))
        for statement in resource["Properties"]["PolicyDocument"]["Statement"]
    ]


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def test_the_coordinator_may_list_the_snapshots_bucket():
    grants = [
        statement
        for statement in _coordinator_statements()
        if "s3:ListBucket" in _as_list(statement.get("Action", []))
        and SNAPSHOTS_BUCKET_ARN in _as_list(statement.get("Resource", []))
    ]

    assert grants, "the coordinator cannot list the snapshots bucket"
    for statement in grants:
        assert statement["Effect"] == "Allow"
        # No condition on the key prefix. S3 documents only that the 404 needs
        # s3:ListBucket on the bucket, and a condition it did not count as met
        # there would bring the 403 back without failing anything.
        assert "Condition" not in statement
