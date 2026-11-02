# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth-level guard on the internal ALB's security group.

The ALB is reachable only through CloudFront's VPC origin. Two properties make
that true, and neither is self-evident from reading the stack:

* the listener is created with ``open=False``. CDK's default adds a
  ``0.0.0.0/0`` ingress rule on the listener port to the ALB's security group.
  The one rule the ALB needs — tcp/80 from CloudFront's VPC-origin service
  group — is declared explicitly in ``network_stack.py``, and only when the
  ``cloudfront_vpc_origin_sg_id`` context is supplied;
  ``cf.VpcOrigin.with_application_load_balancer`` adds no rule at all.
  ``test_alb_cloudfront_ingress.py`` pins that positive shape; this file pins
  the negative half.
* the ALB security group keeps a STABLE logical id. It is CDK's auto-generated
  default group (the stack passes no ``security_group=``). If a future edit
  changes the group's logical id, CloudFormation replaces the group — a new
  group id behind the ``AlbSecurityGroupId`` output — which should be a
  deliberate change, not a side effect of an unrelated edit.

Both are one-word edits away from reverting with no other visible effect, which
is why they are pinned here rather than left to the stack comments.
"""

import sys
from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template


@pytest.fixture(scope="module")
def network_template() -> Template:
    """Synthesize NetworkStack alone — nothing here needs the wider stack graph."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))
    from network_stack import NetworkStack

    return Template.from_stack(NetworkStack(App(), "test-network"))


#: CDK's auto-generated logical id for the ALB's default security group. Pinned
#: because a change here means CloudFormation REPLACES the group (a new group
#: id), which should never happen as a side effect of an unrelated edit.
_ALB_SG_LOGICAL_ID = "AlbSecurityGroup580F65A6"


def _alb_security_group(template: Template) -> dict:
    groups = template.find_resources("AWS::EC2::SecurityGroup")
    assert _ALB_SG_LOGICAL_ID in groups, (
        "the ALB security group's logical id changed, so CloudFormation would "
        f"REPLACE the group (a new group id). Found: {sorted(groups)}"
    )
    return groups[_ALB_SG_LOGICAL_ID]


def test_alb_security_group_has_no_world_ingress(network_template):
    """``open=False`` — the listener must not open the ALB to 0.0.0.0/0.

    Fails if the ``open=False`` argument is dropped from ``add_listener``, which
    is otherwise invisible: the stack still deploys and the site still works,
    while the security group advertises world ingress.
    """
    props = _alb_security_group(network_template)["Properties"]
    for rule in props.get("SecurityGroupIngress", []):
        assert rule.get("CidrIp") != "0.0.0.0/0", (
            "the ALB security group grants world ingress — has open=False been "
            f"dropped from add_listener? rule={rule}"
        )
        assert rule.get("CidrIpv6") != "::/0", f"world IPv6 ingress: {rule}"


def test_alb_security_group_has_no_ingress_without_the_cloudfront_context(network_template):
    """Without the CloudFront SG context, the ALB group has no ingress at all.

    None inline and none as a separate resource. The fixture synthesizes the
    stack with no ``cloudfront_vpc_origin_sg_id`` context, so the one legitimate
    rule — declared in ``network_stack.py`` only when that context is supplied —
    is absent here. An ingress rule appearing anyway would be a second path into
    the ALB.
    """
    props = _alb_security_group(network_template)["Properties"]
    assert props.get("SecurityGroupIngress", []) == []

    standalone = network_template.find_resources("AWS::EC2::SecurityGroupIngress")
    alb_rules = {
        lid: r for lid, r in standalone.items()
        if _ALB_SG_LOGICAL_ID in str(r.get("Properties", {}).get("GroupId", ""))
    }
    assert not alb_rules, f"unexpected ingress into the ALB group: {alb_rules}"


def test_alb_stays_internal(network_template):
    """An internet-facing scheme would make the world-ingress question live.

    The ``open=False`` reasoning leans on there being no public path into these
    subnets, so the scheme is part of the same invariant.
    """
    network_template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        {"Scheme": "internal"},
    )


def test_listener_is_http_on_port_80(network_template):
    """Auth is enforced in the app, so the listener is deliberately HTTP-only.

    Pinned with the ingress rules because "no certificate" and "no world
    ingress" only add up to a safe configuration together.
    """
    network_template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::Listener",
        {"Port": 80, "Protocol": "HTTP"},
    )
