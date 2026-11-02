# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ALB's one way in: CloudFront's VPC-origin security group.

``cf.VpcOrigin.with_application_load_balancer`` does NOT add the ingress rule
CloudFront needs: the frontend template contains an ``AWS::CloudFront::VpcOrigin``
and no ``AWS::EC2::SecurityGroupIngress`` at all. VPC-origin ENIs sit inside the
VPC, so the ``0.0.0.0/0`` rule CDK's ``open=True`` default adds would admit them;
with ``open=False`` and no explicit rule, ``/api/*`` returns 504 while the target
behind it stays healthy.

Three states must be distinguishable, and an assertion on the ABSENCE of world
ingress alone (``test_alb_ingress.py``) cannot tell the first two apart:

* no ingress at all  → API unreachable
* ``0.0.0.0/0``      → reachable but the documented guarantee is false
* CloudFront SG only → correct

So these tests assert the POSITIVE shape. ``test_alb_ingress.py`` keeps the
negative half; the two are complementary and both are needed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk.assertions import Annotations, Match, Template

#: CDK's logical id for the ALB's auto-created security group. Shared with
#: ``test_alb_ingress.py``, which pins it for the replacement hazard.
_ALB_SG_LOGICAL_ID = "AlbSecurityGroup580F65A6"

#: Any syntactically valid group id works — these tests are about the wiring,
#: not about which group a given account happens to have.
_FAKE_CF_SG = "sg-0123456789abcdef0"

_CONTEXT_KEY = "cloudfront_vpc_origin_sg_id"


def _network_template(**context) -> Template:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))
    from network_stack import NetworkStack

    app = App(context=context or None)
    return Template.from_stack(NetworkStack(app, "test-network"))


@pytest.fixture(scope="module")
def wired() -> Template:
    """The deployed configuration: the SG id supplied via context."""
    return _network_template(**{_CONTEXT_KEY: _FAKE_CF_SG})


class TestTheRuleExists:
    def test_exactly_one_ingress_rule(self, wired):
        """Two rules would mean a second, undocumented path to the ALB."""
        rules = wired.find_resources("AWS::EC2::SecurityGroupIngress")
        assert len(rules) == 1, (
            f"expected exactly one ingress rule into the ALB, found {len(rules)}: "
            f"{sorted(rules)}"
        )

    def test_the_rule_sources_the_cloudfront_group_on_port_80(self, wired):
        """The no-rule state — API unreachable — fails here, which a
        world-ingress-absence check alone cannot catch."""
        wired.has_resource_properties(
            "AWS::EC2::SecurityGroupIngress",
            {
                "IpProtocol": "tcp",
                "FromPort": 80,
                "ToPort": 80,
                "SourceSecurityGroupId": _FAKE_CF_SG,
            },
        )

    def test_the_rule_targets_the_alb_group(self, wired):
        """A rule on some other group would pass the shape check and still
        leave the ALB unreachable."""
        rules = wired.find_resources("AWS::EC2::SecurityGroupIngress")
        (props,) = [r["Properties"] for r in rules.values()]
        assert props["GroupId"] == {
            "Fn::GetAtt": ["AlbSecurityGroup580F65A6", "GroupId"]
        }, f"ingress is not attached to the ALB's security group: {props['GroupId']}"

    def test_no_cidr_ingress_into_the_alb(self, wired):
        """Catches a revert to ``open=True`` even alongside a correct SG rule.

        Scoped to the ALB's group on purpose. Other groups here legitimately
        admit the VPC CIDR — the ECR interface endpoint has to, on 443 — so a
        stack-wide sweep would fail for an unrelated and correct reason.
        """
        for name, rule in wired.find_resources(
            "AWS::EC2::SecurityGroupIngress"
        ).items():
            props = rule["Properties"]
            assert "CidrIp" not in props and "CidrIpv6" not in props, (
                f"{name}: CIDR-based ingress into the ALB: {props}"
            )

        alb_group = wired.find_resources("AWS::EC2::SecurityGroup")[
            _ALB_SG_LOGICAL_ID
        ]
        for rule in alb_group["Properties"].get("SecurityGroupIngress", []):
            assert "CidrIp" not in rule and "CidrIpv6" not in rule, (
                f"the ALB group carries inline CIDR ingress — has open=True "
                f"come back? {rule}"
            )


class TestTheCloudFrontGroupIsOnlyReferenced:
    """The group is AWS's, so this stack must consume it and never manage it.

    Note on what is NOT pinned here: ``mutable=False`` on the import. It reads
    like a guard but is a no-op for ``allow_from`` — the synthesized template is
    byte-identical without it, since ``allow_from`` only touches the ALB's own
    group. A test asserting "no egress on the CloudFront group" therefore passes
    whether or not the flag is present, so it would have been decorative and is
    deliberately absent. What IS pinned is the property that can actually
    regress: the group must not become a resource this stack creates.
    """

    def test_the_cloudfront_group_is_not_a_managed_resource(self, wired):
        """Swapping the import for a real ``SecurityGroup`` construct would make
        CDK try to CREATE a group AWS already owns."""
        for name, group in wired.find_resources("AWS::EC2::SecurityGroup").items():
            description = str(group["Properties"].get("GroupDescription", ""))
            assert "CloudFront" not in description, (
                f"{name} looks like this stack is creating CloudFront's own "
                f"security group rather than importing it: {description}"
            )

    def test_the_group_appears_only_as_a_rule_source(self, wired):
        """Its id should occur exactly once in the whole template, as the
        ``SourceSecurityGroupId`` of the one ingress rule."""
        import json

        occurrences = json.dumps(wired.to_json()).count(_FAKE_CF_SG)
        assert occurrences == 1, (
            f"the CloudFront group id appears {occurrences} times in the "
            "template; expected exactly one (the ingress rule's source)"
        )


class TestMissingContextFailsLoudlyRatherThanQuietly:
    """A fallback to something wider would hide the missing value. Absent
    context must produce no rule and a warning, not a guess."""

    def test_no_ingress_rule_without_the_context_key(self):
        template = _network_template()
        assert template.find_resources("AWS::EC2::SecurityGroupIngress") == {}

    def test_a_warning_names_the_consequence_and_the_remedy(self):
        """Silence here is the actual hazard: a deploy that quietly produces an
        unreachable API. The warning has to say what breaks (504) and what fixes
        it (the context key)."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))
        from network_stack import NetworkStack

        app = App()
        stack = NetworkStack(app, "test-network-nocontext")
        annotations = Annotations.from_stack(stack)
        annotations.has_warning("*", Match.string_like_regexp(f".*{_CONTEXT_KEY}.*"))
        annotations.has_warning("*", Match.string_like_regexp(".*504.*"))


class TestPostDeploySuppliesTheValue:
    """The id cannot be a CFN reference — the group is created by the frontend
    stack's VPC origin, which deploys AFTER network. ``dev-postdeploy.sh`` is
    what closes that ordering gap, so it has to keep doing it.
    """

    def test_postdeploy_discovers_persists_and_redeploys(self):
        script = (
            Path(__file__).resolve().parent.parent / "scripts" / "dev-postdeploy.sh"
        ).read_text()

        assert "CloudFront-VPCOrigins-Service-SG" in script, (
            "dev-postdeploy.sh no longer looks the group up by name"
        )
        assert _CONTEXT_KEY in script, (
            f"dev-postdeploy.sh no longer passes/persists {_CONTEXT_KEY}"
        )
        assert "cdk deploy ai-eval-studio-network" in script, (
            "discovering the id is useless without redeploying the network "
            "stack that consumes it"
        )

    def test_the_lookup_is_scoped_to_the_vpc(self):
        """Filtering on group name alone would pick a group from an unrelated
        VPC in a multi-VPC account."""
        script = (
            Path(__file__).resolve().parent.parent / "scripts" / "dev-postdeploy.sh"
        ).read_text()
        assert "Name=vpc-id,Values=" in script

    def test_network_exports_what_the_lookup_needs(self, wired):
        """The script reads VpcId off this stack to scope that filter."""
        outputs = wired.find_outputs("*")
        assert "VpcId" in outputs


class TestTheDeployAssertsTheApiAnswers:
    """Every deploy leg can report success while ``/api/*`` is dead.

    An ALB synthesized with zero ingress rules answers every API call with 504
    while CloudFormation reports success. A context-value check would only
    catch the one cause already understood, so the script asserts the OUTCOME
    over HTTP instead.

    Source-level, because the real assertion needs a deployed environment. The
    intended behaviour: 200 → pass, 504 → exit 1, connection failure → retry
    then exit 1.
    """

    SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dev-postdeploy.sh"

    #: Start of the guard, and the banner it must precede.
    _START = "# ---- Verify the API path actually answers"
    _BANNER = 'echo "# === Post-deploy complete'

    @pytest.fixture(scope="class")
    def script(self) -> str:
        return self.SCRIPT.read_text()

    @pytest.fixture(scope="class")
    def guard(self, script: str) -> str:
        """Just the guard block.

        Scoped deliberately. Searching the whole script for ``exit 1`` or a
        variable name passes on strings that belong to unrelated sections, and
        stays green even when the guard's own ``exit 1`` is replaced with a
        no-op.
        """
        assert self._START in script, (
            f"dev-postdeploy.sh no longer contains {self._START!r}; the API "
            "assertion was removed or renamed"
        )
        assert self._BANNER in script, "the completion banner moved or was renamed"
        start = script.index(self._START)
        end = script.index(self._BANNER)
        assert start < end, "the API probe must gate the completion banner"
        return script[start:end]

    def test_the_api_path_is_probed_over_http(self, guard: str):
        assert "/api/config" in guard, (
            "the guard no longer probes the API path, so a deploy that leaves "
            "/api/* returning 504 reports success"
        )
        assert "curl" in guard, "the probe must be a real request, not a config read"

    def test_a_bad_response_fails_the_script(self, guard: str):
        """Without a non-zero exit this is a log line nobody reads."""
        assert "exit 1" in guard, (
            "the API probe does not exit non-zero, so a 504 is only a warning "
            "and the deploy still looks successful"
        )

    def test_the_probe_retries_rather_than_racing_propagation(self, guard: str):
        """CloudFront config propagation and ALB target registration are both
        eventually-consistent — a single immediate request would flake.

        Matches a shell loop HEADER (``for``/``while`` … ``do``) rather than the
        bare keyword: a search for ``while`` anywhere is satisfied by the
        guard's own comment, "report success while /api/* is dead" — prose, not
        control flow.
        """
        assert "sleep" in guard, "no backoff between attempts"
        assert re.search(r"^\s*(for|while)\b.*\bdo\b", guard, re.MULTILINE), (
            "the probe does not loop, so it races eventually-consistent state"
        )

    def test_the_verdict_is_fail_closed_on_anything_but_200(self, guard: str):
        """The authoritative check is the negative one.

        Asserting the success comparison is not enough: removing it only costs
        the retry, because this check still catches a non-200 and exits. So this
        pins the comparison that actually decides pass/fail — anything other than
        200, including a CloudFront error page, must fail the deploy.
        """
        assert re.search(r'!=\s*"?200"?', guard), (
            "no fail-closed status comparison — the guard cannot distinguish a "
            "healthy API from a 504"
        )

    def test_the_failure_message_names_the_likely_cause_and_the_fix(self, guard: str):
        """An assertion that only says "it failed" leaves the next operator to
        rediscover the cause from scratch."""
        assert "describe-security-group-rules" in guard, (
            "the failure path should hand over the command that confirms the "
            "ALB ingress rule, not just report a bad status code"
        )
        assert _CONTEXT_KEY in guard, (
            "the failure path should name the context value whose absence "
            "produces this exact outage"
        )


class TestTheDiagramStatesTheMechanism:
    """The architecture diagram asserted the CONCLUSION ("Reachable only from
    CloudFront") while nothing implemented it. A conclusion cannot be checked
    against the code; the mechanism can, so the label has to name it.

    This class gates the ``.drawio`` source. The exported ``.svg`` is gated
    separately by ``test_diagram_svg_matches_source.py``, which compares the
    labels draw.io embeds in the export against the source — so a stale export
    cannot survive an edit here. (That check was only possible once the export
    became SVG: the previous PNG carried its copy URL-encoded inside a binary
    chunk, invisible to every text tool.)
    """

    DIAGRAM = (
        Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "diagrams"
        / "architecture-overview.drawio"
    )

    def test_the_alb_label_names_its_ingress_source(self):
        text = self.DIAGRAM.read_text()
        assert "CloudFront VPC-origin SG only" in text, (
            "the ALB box no longer states what its security group admits; a "
            "label like 'Reachable only from CloudFront' is a claim nobody can "
            "verify against the stack"
        )

    def test_the_unverifiable_phrasing_has_not_come_back(self):
        assert "Reachable only from CloudFront" not in self.DIAGRAM.read_text()
