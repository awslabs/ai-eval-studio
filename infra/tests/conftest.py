# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the infra synth tests.

``AuthStack`` refuses to synthesize without ``COGNITO_DOMAIN_PREFIX``, because
in a real deploy a missing prefix is an unprepared shell. Unit tests synthesize
the stack without ``dev-prep.sh``, so a fixed test prefix is set here, before
any test or fixture builds a stack. Tests of the missing-prefix refusal remove
it themselves.
"""

import os

TEST_COGNITO_DOMAIN_PREFIX = "ai-eval-studio-unit-test"


def pytest_configure(config):
    """Set the test prefix before collection, so module-level fixtures see it."""
    os.environ["COGNITO_DOMAIN_PREFIX"] = TEST_COGNITO_DOMAIN_PREFIX
