# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network stack: VPC, subnets, NAT, VPC endpoints, **internal ALB (HTTP)**.

Cost-optimized: a single NAT gateway across both AZs (not HA). Acceptable
trade-off — if the AZ hosting the NAT fails, outbound traffic for the other AZ
breaks until the AZ recovers or we redeploy with two NATs. At 10 users this
saves ~$32/month.

VPC endpoints (S3 gateway, DynamoDB gateway, ECR interface)
keep data-plane traffic off the public internet.

The **ALB** lives here so ``cdk destroy ai-eval-studio-compute``
doesn't blow away the ALB DNS name on every iteration.
``RemovalPolicy.RETAIN`` makes the DNS survive
``cdk destroy ai-eval-studio-network`` too.

The ALB is **internal** (``internet_facing=False``, private subnets) and its
only client is CloudFront's VPC origin from the :class:`FrontendStack`. Three
independent things make that true, and the third had to be added by hand:

1. no public IP, private subnets, no peering/VPN/NAT path inbound;
2. the app fails closed on every non-allowlisted route;
3. the security group admits **only** the CloudFront VPC-origin service group
   on port 80 — an explicit rule in this stack, not something the CloudFront
   origin construct provides. The construct adds no rule at all; without this
   one CloudFront answers the API paths with 504. See the ``open=False`` block
   below.

Until (3) existed, the accurate statement was the weaker "no public path in,
and the app fails closed" — the group carried a ``0.0.0.0/0`` rule. Anyone
building a network-trusted path into this VPC on the stronger reading would
still be wrong: (3) constrains who reaches the **ALB**, not who reaches other
workloads in the VPC. The viewer-facing HTTPS layer lives at CloudFront's edge
(default ``*.cloudfront.net`` cert).

**HTTP-only listener on port 80.** Auth is enforced inside the FastAPI
app (in-app JWT against Cognito JWKS — see
``backend/app/platform/aws/auth_cognito_bearer.py``), so the listener
doesn't need ``authenticate-cognito`` and therefore doesn't need TLS:
the ALB listens HTTP on port 80, CloudFront's VPC origin connects with
``OriginProtocolPolicy.HTTP_ONLY``, and the CloudFront ↔ ALB hop rides
the AWS managed plane. Production hardening adds a custom domain + ACM
cert at the CloudFront edge AND on the ALB for end-to-end TLS.

The **target group** is created here too so the ALB has something to
forward to even when the compute stack is destroyed and re-deployed
(the actual *targets* — Fargate tasks — re-bind on each compute deploy
via :class:`ComputeStack`). It speaks HTTPS, so the ALB-to-task hop is
encrypted.

The **access-log bucket** lives here because the ALB, the first thing that
logs into it, does; every other stack that logs into it deploys after this
one. They find it by name (``access_logs_bucket_name``) rather than by a
cross-stack reference, and its policy grants each log-delivery service
itself, so no consumer has to edit it.
"""

from __future__ import annotations

from aws_cdk import (
    Annotations,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Token,
    aws_ec2 as ec2,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

#: Key prefixes in the access-log bucket, one per source.
ALB_ACCESS_LOG_PREFIX = "alb"
CLOUDFRONT_ACCESS_LOG_PREFIX = "cloudfront/"
S3_ACCESS_LOG_PREFIX = "s3/"

#: How long access logs are kept. Long enough to investigate an incident found
#: months later; the volume at this app's scale makes the cost negligible.
ACCESS_LOG_RETENTION = Duration.days(365)


def access_logs_bucket_name(stack: Stack) -> str:
    """Return the access-log bucket's name, which other stacks import it by."""
    return f"ai-eval-studio-access-logs-{stack.account}-{stack.region}"


class NetworkStack(Stack):
    """VPC + endpoints + internal HTTP ALB for Eval Studio.

    Attributes:
        vpc: VPC consumed by the compute stack for Fargate tasks.
        alb: Internal ApplicationLoadBalancer. ``RemovalPolicy.RETAIN``
            so the DNS name survives stack churn.
        coordinator_target_group: Target group the compute stack registers
            its Fargate service against on each deploy.
        access_logs_bucket: Bucket receiving the ALB, CloudFront and S3
            server access logs.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- VPC --------------------------------------------------------
        # 2 AZs, public + private subnets, single NAT gateway (cost-optimized).
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # Gateway endpoints are free (no hourly charge, no data charge).
        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )
        self.vpc.add_gateway_endpoint(
            "DynamoDBEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
        )

        # Interface endpoints (PrivateLink) for ECR. ECR needs both
        # `ECR` (auth) and `ECR_DOCKER` (image pulls), plus S3 (already gateway)
        # for the layer blobs.
        self.vpc.add_interface_endpoint(
            "EcrEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR,
        )
        self.vpc.add_interface_endpoint(
            "EcrDockerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
        )

        # ---- ALB --------------------------------------------------------
        # Internal ALB sitting in private subnets; CloudFront's VPC origin is
        # its only client. HTTP-only listener: auth is enforced in the
        # FastAPI app, not at the listener, so there is no need for an
        # HTTPS listener / certificate. CloudFront's VPC origin uses
        # ``OriginProtocolPolicy.HTTP_ONLY`` to talk to it.
        #
        # The ALB uses CDK's auto-generated default SecurityGroup (no
        # ``security_group=`` arg) so its logical ID + cross-stack export
        # stay stable across listener changes. Its one ingress rule is
        # declared further down, next to the listener — NOT by the CloudFront
        # origin construct, which adds no security-group rule at all.
        self.alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            vpc=self.vpc,
            internet_facing=False,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
        )
        self.alb.apply_removal_policy(RemovalPolicy.RETAIN)

        # ---- Access logs -------------------------------------------------
        # One bucket for every request log: the ALB here, the CloudFront
        # distribution, and S3 server access logs for the data buckets.
        #
        # SSE-S3, not a customer-managed key: ALB access logs and S3 server
        # access logs can only be delivered to an SSE-S3 bucket. The logs hold
        # request metadata (paths, sources, timings, status), not stored data.
        #
        # BUCKET_OWNER_PREFERRED keeps ACLs available because CloudFront's
        # standard logging delivers through an ACL grant it adds itself; the
        # bucket owner still owns every object. Block-public-access stays on.
        self.access_logs_bucket = s3.Bucket(
            self,
            "AccessLogsBucket",
            bucket_name=access_logs_bucket_name(self),
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="access-logs-expiry",
                    enabled=True,
                    expiration=ACCESS_LOG_RETENTION,
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
            ],
        )
        # S3 server access logs from this account's app buckets. The source
        # buckets import this one by name, so CDK cannot add this grant from
        # their side; the conditions keep any other account's or app's
        # buckets from writing here.
        self.access_logs_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowS3ServerAccessLogs",
                principals=[iam.ServicePrincipal("logging.s3.amazonaws.com")],
                actions=["s3:PutObject"],
                resources=[
                    self.access_logs_bucket.arn_for_objects(f"{S3_ACCESS_LOG_PREFIX}*")
                ],
                conditions={
                    "ArnLike": {"aws:SourceArn": "arn:aws:s3:::ai-eval-studio-*"},
                    "StringEquals": {"aws:SourceAccount": self.account},
                },
            )
        )
        # The ALB delivers as a region-specific ELB account in older regions,
        # so its grant needs a concrete region. The deployed app always has
        # one (``app.py`` defaults it); without one, recording an error fails
        # ``cdk synth`` rather than shipping an ALB that logs nothing.
        if Token.is_unresolved(self.region):
            Annotations.of(self).add_error(
                "ALB access logging needs a concrete region: deploy with "
                "CDK_DEFAULT_REGION set, or an explicit env on this stack."
            )
        else:
            self.alb.log_access_logs(self.access_logs_bucket, ALB_ACCESS_LOG_PREFIX)

        # Coordinator target group — created here so the ALB has a forward
        # destination even when compute is torn down. Targets register
        # dynamically when the compute stack deploys the Fargate service.
        #
        # HTTPS, so the ALB-to-task hop is encrypted. The task serves a
        # certificate it generates at start-up (``app.scripts.serve_tls``);
        # ALB encrypts to its targets but does not validate their
        # certificates, so a self-signed one is enough. The health check
        # follows the same protocol.
        self.coordinator_target_group = elbv2.ApplicationTargetGroup(
            self,
            "CoordinatorTargetGroup",
            vpc=self.vpc,
            port=8100,
            protocol=elbv2.ApplicationProtocol.HTTPS,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                protocol=elbv2.Protocol.HTTPS,
                path="/api/health",
                interval=Duration.seconds(10),
                # Tuned for fail-fast: 2 consecutive failures × 10s = ~20s
                # detection window. Shrinks bad-rollout
                # detection from the previous ~90s (3 × 30s). Healthy deploy:
                # 2 passes × 10s = ~20s registration. The 60s grace period
                # (health_check_grace_period on the service below) prevents
                # Fargate cold-start latency from tripping the circuit breaker.
                # The real safeguard against event-loop blocks is still keeping
                # blocking work OFF the loop via ``to_thread``.
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=2,
            ),
            deregistration_delay=Duration.seconds(30),
        )

        # HTTP listener — forwards every request to the coordinator target
        # group. The FastAPI middleware in
        # ``app/platform/auth.py`` validates the bearer JWT before any
        # route handler runs; ``/api/health`` and ``/api/config`` are
        # explicit public exceptions for the LB health-check + the SPA
        # bootstrap, respectively.
        #
        # ``open=False`` — CDK's default adds a ``0.0.0.0/0`` ingress rule on the
        # listener port to the ALB's security group, which would make the group
        # advertise world ingress when CloudFront is meant to be the only thing
        # that can reach the ALB.
        #
        # The rule CloudFront needs does NOT come from
        # ``cf.VpcOrigin.with_application_load_balancer`` in ``frontend_stack.py``:
        # that construct emits an ``AWS::CloudFront::VpcOrigin`` and nothing else —
        # no ``AWS::EC2::SecurityGroupIngress`` anywhere in the frontend template,
        # on create or on update. CloudFront's VPC-origin ENIs live inside this
        # VPC, so a ``0.0.0.0/0`` rule would admit them; with ``open=False`` and
        # no explicit rule, CloudFront answers ``/api/*`` with 504 while the SPA
        # keeps serving from S3 and the target stays healthy.
        #
        # So the rule is declared here explicitly, which is genuinely tighter
        # than ``0.0.0.0/0``.
        self.alb.add_listener(
            "HttpListener",
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
            default_action=elbv2.ListenerAction.forward(
                target_groups=[self.coordinator_target_group]
            ),
        )

        # The peer is CloudFront's service-managed ``CloudFront-VPCOrigins-Service-SG``,
        # created by AWS in this VPC when the first VPC origin appears — i.e. by
        # the FRONTEND stack, which already depends on this one. Its id therefore
        # cannot be a CFN reference (``AWS::CloudFront::VpcOrigin`` exposes no
        # security-group attribute) and referencing it from here would be a cycle.
        #
        # Resolved the same way as every other cross-stack-discovered value in
        # this app: ``dev-postdeploy.sh`` looks it up after the deploy, persists
        # it to ``cdk.context.json``, and redeploys this stack. Absent context
        # means no rule at all rather than a quiet fallback to something wider —
        # a fallback would hide the missing value. The build warns instead,
        # because "the API is unreachable" must not be silent.
        cf_origin_sg_id = self.node.try_get_context("cloudfront_vpc_origin_sg_id")
        if cf_origin_sg_id:
            self.alb.connections.allow_from(
                # ``mutable=False`` records that this group belongs to the
                # CloudFront service and must not be edited from here. It is a
                # no-op for THIS call — verified: the synthesized template is
                # byte-identical with and without it, because ``allow_from``
                # only adds ingress to the ALB's own group. It would start
                # mattering if anyone reached for this handle to add egress.
                ec2.SecurityGroup.from_security_group_id(
                    self,
                    "CloudFrontVpcOriginSg",
                    cf_origin_sg_id,
                    mutable=False,
                ),
                ec2.Port.tcp(80),
                "CloudFront VPC origin to internal ALB",
            )
        else:
            Annotations.of(self).add_warning(
                "No 'cloudfront_vpc_origin_sg_id' in CDK context, so the ALB "
                "security group gets NO ingress rule and CloudFront will return "
                "504 for /api/* and /ws/*. Run infra/scripts/dev-postdeploy.sh "
                "(it discovers the id and redeploys this stack) or pass "
                "-c cloudfront_vpc_origin_sg_id=sg-xxxx. Expected on a first "
                "deploy, when the VPC origin that creates that group does not "
                "exist yet."
            )

        # ALB internal DNS exposed as a CfnOutput so other stacks /
        # debug tooling can reference it without guessing CDK's
        # auto-generated load-balancer name. This is the **internal**
        # DNS — CloudFront talks to the ALB via VPC origin, not via
        # this DNS.
        CfnOutput(
            self,
            "AlbDnsName",
            value=self.alb.load_balancer_dns_name,
            description="Internal DNS of the ALB. Reachable only from inside the VPC.",
            export_name="ai-eval-studio-alb-dns-name",
        )

        # Lets ``dev-postdeploy.sh`` scope its CloudFront-VPCOrigins-Service-SG
        # lookup to this VPC. Filtering on the group name alone would silently
        # pick a group from an unrelated VPC in a multi-VPC account.
        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            description="VPC hosting the ALB. Used to scope the CloudFront VPC-origin SG lookup.",
            export_name="ai-eval-studio-vpc-id",
        )

        # The ALB's own security group, so the post-deploy step can authorize
        # CloudFront's ingress without scraping CDK's generated logical ids.
        CfnOutput(
            self,
            "AlbSecurityGroupId",
            value=self.alb.connections.security_groups[0].security_group_id,
            description="ALB security group. Ingress: CloudFront VPC-origin SG only.",
            export_name="ai-eval-studio-alb-security-group-id",
        )
