# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Cognito hosted-domain prefix comes only from the prep environment.

A fixed fallback prefix is guessable, and one was applied silently to a real
deployment when ``cdk deploy`` ran in a shell without ``dev-prep.sh``'s
exports. The auth stack therefore refuses to synthesize without the variable.
"""

import sys
from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))


def _auth_stack():
    from auth_stack import AuthStack

    return AuthStack(App(), "test-auth")


def test_synth_refuses_a_missing_prefix(monkeypatch):
    """Without the variable, synth fails and names the prep script to run."""
    monkeypatch.delenv("COGNITO_DOMAIN_PREFIX", raising=False)
    with pytest.raises(ValueError, match="dev-prep.sh"):
        _auth_stack()


def test_synth_refuses_an_empty_prefix(monkeypatch):
    """An exported-but-empty variable is as unprepared as a missing one."""
    monkeypatch.setenv("COGNITO_DOMAIN_PREFIX", "")
    with pytest.raises(ValueError, match="COGNITO_DOMAIN_PREFIX"):
        _auth_stack()


def test_the_domain_uses_the_given_prefix(monkeypatch):
    """The prefix in the environment is the domain created, unchanged."""
    monkeypatch.setenv("COGNITO_DOMAIN_PREFIX", "ai-eval-studio-k3x9q2ab")
    domain = Template.from_stack(_auth_stack()).find_resources(
        "AWS::Cognito::UserPoolDomain"
    )
    assert [d["Properties"]["Domain"] for d in domain.values()] == [
        "ai-eval-studio-k3x9q2ab"
    ]
