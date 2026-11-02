# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests package.

Tests marked with ``@pytest.mark.integration`` are skipped by default
(backend pytest config sets ``addopts = "-m 'not integration'"``) and
only run when explicitly invoked with ``pytest -m integration``.

These tests typically involve mocked AWS services (via moto) or other
heavyweight setup that doesn't belong in the default unit test suite.
"""
