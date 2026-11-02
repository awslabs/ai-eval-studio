# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adapter-parity guard: one check over the whole durable-state method set.

This test asserts that the seven durable-state Protocol methods (event log +
connection registry + list_jobs_by_owner) are implemented in BOTH the dev
(FilesystemMetadataStore) and cloud (DynamoDBMetadataStore) adapters.

**Why this test exists:** Dev-mode green != cloud-mode green. A method present
in only one adapter breaks dev's ability to run the whole flow in-process
against the same interface cloud uses. Checking the full set here means a
method that reaches only one adapter cannot slip through.

The seven methods:
- allocate_job_seq, append_job_event, events_since
- put_connection, delete_connection, list_connections_for_user
- list_jobs_by_owner
"""

import pytest

from app.platform.aws.dynamodb_metadata_store import DynamoDBMetadataStore
from app.platform.dev_local import FilesystemMetadataStore


# The seven durable-state methods
DURABLE_STATE_METHODS = [
    "allocate_job_seq",
    "append_job_event",
    "events_since",
    "put_connection",
    "delete_connection",
    "list_connections_for_user",
    "list_jobs_by_owner",
]


def test_dev_adapter_implements_durable_state_methods():
    """FilesystemMetadataStore exposes all seven durable-state methods."""
    dev_methods = set(dir(FilesystemMetadataStore))
    for method_name in DURABLE_STATE_METHODS:
        assert method_name in dev_methods, (
            f"FilesystemMetadataStore missing durable-state method: {method_name}"
        )


def test_cloud_adapter_implements_durable_state_methods():
    """DynamoDBMetadataStore exposes all seven durable-state methods."""
    cloud_methods = set(dir(DynamoDBMetadataStore))
    for method_name in DURABLE_STATE_METHODS:
        assert method_name in cloud_methods, (
            f"DynamoDBMetadataStore missing durable-state method: {method_name}"
        )


def test_both_adapters_have_parity():
    """Both adapters expose the same seven durable-state methods (set comparison)."""
    dev_methods = {m for m in DURABLE_STATE_METHODS if hasattr(FilesystemMetadataStore, m)}
    cloud_methods = {m for m in DURABLE_STATE_METHODS if hasattr(DynamoDBMetadataStore, m)}

    missing_from_dev = set(DURABLE_STATE_METHODS) - dev_methods
    missing_from_cloud = set(DURABLE_STATE_METHODS) - cloud_methods

    assert not missing_from_dev, f"Dev adapter missing: {missing_from_dev}"
    assert not missing_from_cloud, f"Cloud adapter missing: {missing_from_cloud}"
    assert dev_methods == cloud_methods == set(DURABLE_STATE_METHODS), (
        "Parity broken: dev and cloud adapters do not expose the same methods"
    )
