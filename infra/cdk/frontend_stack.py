# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend stack: S3 bucket + CloudFront distribution.

Serves the React SPA from S3 via CloudFront, with path-based routing
to forward ``/api/*`` and ``/ws/*`` to the ALB origin via a **VPC
origin** (CloudFront → private ALB inside the VPC, no public exposure).

Result: ``https://<cloudfront-dns>/`` returns the SPA, the browser
makes relative ``/api/*`` and ``wss://.../ws`` calls against the same
hostname, CloudFront forwards to the ALB. No CORS, no env-baked API
base URL, no separate domain.

Trust topology:

- ALB is ``internet_facing=False`` (private subnets) and listens **HTTP
  on port 80**. Auth is enforced in the FastAPI app, so the ALB
  doesn't need the ``authenticate-cognito`` action that would force
  an HTTPS listener. Only CloudFront's VPC origin reaches it.
- CloudFront ↔ ALB: ``OriginProtocolPolicy.HTTP_ONLY`` over the
  AWS-managed VPC-origin tunnel. Browser ↔ CloudFront stays HTTPS via
  the default ``*.cloudfront.net`` cert. CloudFront connects to an HTTPS
  origin only when the origin's certificate is publicly trusted and
  matches its hostname, which needs a custom domain; ``SECURITY.md``
  lists that as a production step. The next hop, ALB ↔ coordinator,
  is HTTPS (``network_stack.py``).
- All viewer behaviors are ``REDIRECT_TO_HTTPS`` so a stale bookmark
  or bare ``http://`` URL gets bounced before any request body flows.
- ``/api/*`` and ``/ws/*`` both go to the ALB origin via the same
  ``api_behavior``: ``CACHING_DISABLED`` + ``ALL_VIEWER`` +
  ``ALLOW_ALL`` methods. ``ALL_VIEWER`` already forwards the
  ``Sec-WebSocket-*`` and ``Upgrade`` headers, so the WS upgrade
  travels through CloudFront unchanged.

CSP: a strict ``ResponseHeadersPolicy`` is attached to the **default**
(SPA HTML) behavior — not the API/WS behaviors, since the backend
sets its own response headers. The policy disallows inline scripts
and pins ``connect-src`` to ``'self'`` plus the Cognito hosted UI
domain so the OIDC redirect chain works.

Bundling: ``s3deploy.BucketDeployment`` runs ``npm ci && npm run build``
inside a Node 20 container during ``cdk deploy``. Uses
``CDK_DOCKER=podman`` (configured by ``infra/scripts/dev-prep.sh``) and
``BundlingFileAccess.VOLUME_COPY`` so the build cannot write into the
host's ``frontend/`` — the default bind mount had ``npm ci`` clobbering
the developer's ``node_modules`` with Linux bindings on every deploy.
VOLUME_COPY also requires ``user="0:0"``; the reason is at the call site.

Cache invalidation: ``distribution=`` on the BucketDeployment
auto-invalidates the named ``distribution_paths`` on every deploy.
Hashed asset filenames (``assets/index-DEADBEEF.js``) need no
invalidation; the entry-point HTML does.
"""

from __future__ import annotations

import os
from pathlib import Path

from aws_cdk import (
    BundlingFileAccess,
    CfnOutput,
    DockerImage,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudfront as cf,
    aws_cloudfront_origins as origins,
    aws_elasticloadbalancingv2 as elbv2,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

from network_stack import CLOUDFRONT_ACCESS_LOG_PREFIX, access_logs_bucket_name


class FrontendStack(Stack):
    """S3 + CloudFront distribution serving the SPA, with VPC origin to ALB.

    Args:
        alb: The internal :class:`IApplicationLoadBalancer` from
            :class:`NetworkStack`. CloudFront connects via VPC origin
            on plain HTTP; the ALB has no public DNS.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        alb: elbv2.IApplicationLoadBalancer,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- S3 bucket ---------------------------------------------------
        # Private bucket; CloudFront-only access via OAC (Origin Access
        # Control — modern replacement for OAI). RemovalPolicy.RETAIN
        # mirrors the rest of the persistent tier; the bucket is cheap
        # to keep and re-uploading is the slow part of
        # ``cdk destroy && cdk deploy``.
        bucket = s3.Bucket(
            self,
            "FrontendBucket",
            bucket_name=f"ai-eval-studio-frontend-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ---- Origins -----------------------------------------------------
        # S3 origin, locked to this distribution via OAC.
        s3_origin = origins.S3BucketOrigin.with_origin_access_control(bucket)

        # ALB origin via VPC origin. Protocol policy is HTTP_ONLY: the
        # ALB listener is plain HTTP on port 80 (no cert, no
        # authenticate-cognito), and the CloudFront ↔ ALB hop rides
        # AWS's managed VPC-origin plane.
        alb_origin = origins.VpcOrigin.with_application_load_balancer(
            alb,
            protocol_policy=cf.OriginProtocolPolicy.HTTP_ONLY,
        )
        # NOTE: the VPC origin's L1 logical id is overridden to a fresh
        # value AFTER ``cf.Distribution`` is constructed (see the
        # ``override_logical_id`` call below). Why: CloudFront refuses
        # an in-place ``OriginProtocolPolicy`` update on a VpcOrigin
        # that's still attached to a distribution (HTTP 409 "specified
        # VPC origin is currently associated with one or more
        # distributions"). Forcing a logical-id rename makes CFN do a
        # create-new + repoint-distribution + delete-old replacement
        # instead of an update-in-place — the only path the service
        # accepts for the protocol-policy flip.

        # ---- ALB-origin behavior (shared by /api/* and /ws/*) -----------
        # CACHING_DISABLED + ALL_VIEWER forwards everything (headers,
        # cookies, query strings — including Authorization and the
        # Sec-WebSocket-* + Upgrade headers needed for the WS handshake)
        # and bypasses cache entirely. The default CACHING_OPTIMIZED
        # would strip Authorization and serve cached responses across
        # users; that's a security hole, hence CACHING_DISABLED.
        # REDIRECT_TO_HTTPS bounces stale ``http://`` URLs at the edge
        # without forwarding any payload over HTTP.
        api_behavior = cf.BehaviorOptions(
            origin=alb_origin,
            allowed_methods=cf.AllowedMethods.ALLOW_ALL,
            viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            cache_policy=cf.CachePolicy.CACHING_DISABLED,
            origin_request_policy=cf.OriginRequestPolicy.ALL_VIEWER,
        )

        # ---- Strict CSP for the SPA HTML --------------------------------
        # Attached only to the *default* behavior (the SPA's HTML +
        # bundles). The API/WS behaviors do not carry this policy —
        # the backend sets its own response headers and the CSP would
        # incorrectly attach to API responses too. ``connect-src`` is
        # widened to the Cognito hosted UI domain so the SPA's OIDC
        # redirect chain (PKCE → /oauth2/authorize → token exchange)
        # is allowed; everything else stays at ``'self'``.
        #
        # Resolution order, first-wins:
        # 1. ``cognito_domain`` CDK context — explicit override
        #    (``dev-postdeploy.sh`` passes this on full runs).
        # 2. ``COGNITO_DOMAIN`` env var — full URL form, e.g.
        #    ``https://<prefix>.auth.<region>.amazoncognito.com``.
        # 3. Constructed from ``COGNITO_DOMAIN_PREFIX`` env var. This
        #    is the path ``dev-prep.sh`` already exports, so a direct
        #    ``cdk deploy ai-eval-studio-frontend`` (without re-running
        #    ``dev-postdeploy.sh``) still produces the correct CSP.
        #    Earlier deploys regressed to the placeholder when the
        #    operator skipped post-deploy.
        # 4. Placeholder ``https://example.invalid`` so synth never
        #    fails — the SPA never reaches a runtime callback against
        #    this URL because it pulls real Cognito config from
        #    ``/api/config`` at boot.
        prefix_env = os.environ.get("COGNITO_DOMAIN_PREFIX")
        cognito_domain = (
            self.node.try_get_context("cognito_domain")
            or os.environ.get("COGNITO_DOMAIN")
            or (
                f"https://{prefix_env}.auth.{self.region}.amazoncognito.com"
                if prefix_env
                else None
            )
            or "https://example.invalid"
        )
        csp_policy = cf.ResponseHeadersPolicy(
            self,
            "SpaResponseHeadersPolicy",
            response_headers_policy_name=f"ai-eval-studio-spa-csp-{self.region}",
            comment="Strict CSP for the SPA HTML / asset behaviors.",
            security_headers_behavior=cf.ResponseSecurityHeadersBehavior(
                content_security_policy=cf.ResponseHeadersContentSecurityPolicy(
                    content_security_policy=(
                        "default-src 'self'; "
                        # ``wasm-unsafe-eval`` permits WebAssembly
                        # compilation only — it does NOT enable
                        # general JS ``eval()``. Required for
                        # ``@react-pdf/renderer``'s WASM layout
                        # engine, which the Reports tab uses to
                        # generate exec / per-run PDFs in the
                        # browser. Without it the deployed site
                        # blocks every Download PDF click with a
                        # CSP violation; local dev never sees the
                        # CSP (it's a CloudFront response header)
                        # so this regressed silently into prod.
                        "script-src 'self' 'wasm-unsafe-eval'; "
                        "style-src 'self' 'unsafe-inline'; "
                        "img-src 'self' data:; "
                        "font-src 'self' data:; "
                        # ``cognito-idp.<region>.amazonaws.com`` is the
                        # OIDC ``jwks_uri``. The auth-code flow trusts
                        # the token endpoint and doesn't fetch JWKS
                        # client-side today, so this isn't load-bearing
                        # — but an ``oidc-client-ts`` version or config
                        # change could start fetching it, and CSP would
                        # silently block login. Pre-emptive landmine
                        # removal. ``self.region`` so it tracks the
                        # stack's region (matches how ``cognito_domain``
                        # is built).
                        f"connect-src 'self' {cognito_domain} "
                        f"https://cognito-idp.{self.region}.amazonaws.com; "
                        "frame-ancestors 'none'; "
                        "base-uri 'self'; "
                        "form-action 'self'"
                    ),
                    override=True,
                ),
                strict_transport_security=cf.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.days(365),
                    include_subdomains=True,
                    override=True,
                ),
                content_type_options=cf.ResponseHeadersContentTypeOptions(
                    override=True,
                ),
                frame_options=cf.ResponseHeadersFrameOptions(
                    frame_option=cf.HeadersFrameOption.DENY,
                    override=True,
                ),
                referrer_policy=cf.ResponseHeadersReferrerPolicy(
                    referrer_policy=cf.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
                    override=True,
                ),
            ),
        )

        # ---- SPA-fallback CloudFront Function (default behavior only) ----
        # Why not ``error_responses`` on the distribution: CloudFront's
        # ``CustomErrorResponses`` are distribution-wide; they rewrite
        # 4xx from any origin/behavior, including the ALB-backed
        # ``/api/*`` and ``/ws/*`` behaviors. That turns every legitimate
        # backend 404 (e.g. ``/api/projects/{id}/harness/config`` on a
        # missing project) into a ``200`` carrying the SPA shell, which
        # the SPA then tries to ``JSON.parse('<!doctype html...')``.
        #
        # Instead, run the SPA fallback at viewer-request on the default
        # behavior only: any path without a file extension gets rewritten
        # to ``/index.html`` *before* origin lookup, so S3 never has to
        # 404 on a deep link in the first place. ``/api/*`` and ``/ws*``
        # match earlier behaviors and never reach this function, so
        # backend 4xx flows back to the browser unchanged.
        spa_fallback_fn = cf.Function(
            self,
            "SpaFallback",
            function_name=f"ai-eval-studio-spa-fallback-{self.region}",
            runtime=cf.FunctionRuntime.JS_2_0,
            # 128-byte hard limit on the CFN ``Comment`` field — keep this
            # short. The full rationale lives in the comment block above
            # this construct, which is where future readers will look.
            comment="SPA fallback: rewrite extensionless URIs to /index.html. Attached to default behavior only.",
            code=cf.FunctionCode.from_inline(
                # Real S3 objects (.js, .css, .html, .png, .ico, .json,
                # .woff2, etc.) carry a 1–5 char extension after the last
                # ``.`` — pass those through untouched. Anything else
                # (``/``, ``/projects/abc``, ``/auth/callback``) is a SPA
                # route; rewrite to the shell. Query strings are
                # preserved automatically — the function only mutates
                # ``request.uri``.
                "function handler(event) {\n"
                "  var request = event.request;\n"
                "  if (/\\.[a-zA-Z0-9]{1,5}$/.test(request.uri)) {\n"
                "    return request;\n"
                "  }\n"
                "  request.uri = '/index.html';\n"
                "  return request;\n"
                "}\n"
            ),
        )

        # ---- Distribution ------------------------------------------------
        self.distribution = cf.Distribution(
            self,
            "FrontendDistribution",
            default_behavior=cf.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cf.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=csp_policy,
                # SPA-fallback runs only on this behavior (S3-origin).
                # /api/* and /ws/* attach no function — backend 4xx
                # responses pass straight back to the browser.
                function_associations=[
                    cf.FunctionAssociation(
                        function=spa_fallback_fn,
                        event_type=cf.FunctionEventType.VIEWER_REQUEST,
                    ),
                ],
            ),
            additional_behaviors={
                "/api/*": api_behavior,
                # Two WS patterns: ``/ws`` is the exact path the SPA
                # opens (``wss://<cf>/ws``); ``/ws/*`` covers
                # any future sub-path. CloudFront's wildcard match for
                # ``/ws/*`` requires at least one character after the
                # slash, so without the exact-``/ws`` rule the upgrade
                # request falls through to the default (S3) behavior
                # and the browser sees ``HTTP 200 server: AmazonS3``
                # instead of ``101 Switching Protocols``. ``/ws*`` is
                # NOT a substitute — it would also match ``/wsx``.
                "/ws": api_behavior,
                "/ws/*": api_behavior,
            },
            default_root_object="index.html",
            # Standard logging: one record per viewer request, including the
            # client IP and TLS version, in the access-log bucket the network
            # stack owns. CloudFront grants itself write access to that bucket
            # through an ACL when the distribution is created.
            enable_logging=True,
            log_bucket=s3.Bucket.from_bucket_attributes(
                self,
                "AccessLogsBucket",
                bucket_name=access_logs_bucket_name(self),
                region=self.region,
            ),
            log_file_prefix=CLOUDFRONT_ACCESS_LOG_PREFIX,
        )

        # Force a full replacement of the auto-created VpcOrigin so CFN
        # does a create-new → repoint-distribution → delete-old swap on
        # the protocol-policy flip. The L2 ``cf.Distribution`` nests the
        # VpcOrigin at construct path
        # ``FrontendDistribution/Origin2/VpcOrigin/Resource``; CDK assigns
        # both the CFN logical id (``FrontendDistributionOrigin2VpcOriginCCF1FB11``)
        # AND the AWS-side ``VpcOriginEndpointConfig.Name`` from the same
        # construct path. Two overrides are required:
        #
        # 1. Logical id → makes CFN treat the synthesized resource as
        #    new. On its own this is not enough; the deploy still fails
        #    without step 2.
        # 2. ``VpcOriginEndpointConfig.Name`` → AWS-side identifier the
        #    CloudFront service uses for collision detection. Without an
        #    override, the new resource derives the *same* name from the
        #    unchanged construct path, so the service rejects the create:
        #
        #        Resource of type 'AWS::CloudFront::VpcOrigin' with
        #        identifier '...' already exists.
        #
        # Both overrides must land together. With both in place, CFN
        # creates a brand-new VpcOrigin (different logical id, different
        # AWS Name), repoints the Distribution's Origin2 at the new id
        # via ``Fn::GetAtt``, and deletes the orphaned old VpcOrigin —
        # the only update path the CloudFront service accepts for an
        # in-use VpcOrigin's protocol policy (CloudFront does not allow an
        # in-place update of it).
        vpc_origin_l1 = self.distribution.node.find_child("Origin2") \
            .node.find_child("VpcOrigin") \
            .node.find_child("Resource")
        vpc_origin_l1.override_logical_id("FrontendDistributionOrigin2VpcOriginHttp1")
        vpc_origin_l1.add_property_override(
            "VpcOriginEndpointConfig.Name",
            "ai-eval-studio-alb-vpc-origin-http",
        )

        # ---- Build + deploy the SPA -------------------------------------
        # CDK runs ``npm ci && npm run build`` inside a Node 20 container
        # during ``cdk deploy ai-eval-studio-frontend`` (or --all). The
        # build context is the repo's frontend/ directory — absolute
        # path computed from __file__ so synth works regardless of CWD
        # (relative paths have bitten us in compute_stack before).
        frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
        s3deploy.BucketDeployment(
            self,
            "FrontendDeploy",
            sources=[
                s3deploy.Source.asset(
                    str(frontend_dir),
                    # Keeps generated directories out of the asset FINGERPRINT.
                    # Without it, the hash covers node_modules, so a local
                    # ``npm install`` changed the asset hash and re-deployed the
                    # SPA (plus a CloudFront invalidation) with no source change.
                    #
                    # It does NOT reduce what the bundling container receives,
                    # and cannot be used to dodge the permission problem the
                    # ``user`` note below describes. ``AssetStaging`` passes
                    # ``{sourcePath: this.sourcePath, bundleDir, ...bundlingOptions}``
                    # to the bundler (core/lib/asset-staging.js) — ``exclude`` is
                    # an ASSET option, not a bundling one, so it is not in that
                    # spread, and ``AssetBundlingVolumeCopy`` then copies
                    # ``${sourcePath}/.`` wholesale. Verified, not assumed: with
                    # this exclude in place and ``user`` removed, bundling still
                    # failed with ``rm: cannot remove
                    # 'node_modules/.bin/acorn': Permission denied`` — so the
                    # directory was still copied into the volume. The 762 MB
                    # copy therefore also remains; the only ways to avoid it are
                    # building locally and uploading ``dist/``, or pointing the
                    # asset at a directory that has no ``node_modules``.
                    exclude=["node_modules", "dist", ".vite", "coverage"],
                    bundling={
                        # Node 20 build image. The ``bash -c`` script
                        # runs inside the container; ``/asset-output``
                        # is the directory CDK syncs to S3.
                        "image": DockerImage.from_registry(
                            "public.ecr.aws/sam/build-nodejs20.x"
                        ),
                        "command": [
                            "bash",
                            "-c",
                            # Drop whatever the host tree happened to have.
                            # ``npm ci`` deletes node_modules itself, but doing
                            # it explicitly also clears a stale ``dist`` that
                            # vite would otherwise try to empty, and makes the
                            # bundle independent of the developer's working
                            # state. Both are root-owned in the volume — see
                            # the ``user`` note below for why that matters.
                            "rm -rf node_modules dist"
                            " && npm ci && npm run build"
                            " && cp -r dist/* /asset-output/",
                        ],
                        # Run the build as root.
                        #
                        # CDK's VOLUME_COPY helper (core/lib/private/
                        # asset-staging.js, ``AssetBundlingVolumeCopy.run``)
                        # chowns /asset-input to the host uid:gid and only THEN
                        # runs ``docker cp <source>/. copyContainer:/asset-input``.
                        # ``docker cp`` without ``--archive`` lands every copied
                        # entry root-owned, so the tree the build sees is
                        # root-owned while the build process would be the host
                        # uid. /asset-input itself stays host-owned, so writing
                        # NEW files works and the mismatch stays invisible —
                        # until something has to DELETE a copied file. ``npm
                        # ci`` starts by removing node_modules, so it dies on
                        # ``EACCES: permission denied, rmdir
                        # '/asset-input/node_modules/.bin'`` and takes the whole
                        # synth down with it (every cdk command synths the full
                        # app, so this blocks deploys of unrelated stacks too).
                        #
                        # Matching the build user to the files' actual owner is
                        # the fix. Nothing escapes the container: the tree is a
                        # throwaway volume, not the host's frontend/.
                        "user": "0:0",
                        # VOLUME_COPY, not the CDK default BIND_MOUNT.
                        #
                        # BIND_MOUNT mounts the HOST's frontend/ directory
                        # read-write at /asset-input, so ``npm ci`` above ran
                        # in the real working tree and replaced the
                        # developer's node_modules with the container's
                        # LINUX-arm64 native bindings. Every frontend deploy
                        # then broke ``npm run build`` and ``npm test`` on the
                        # macOS host with "Cannot find native binding"
                        # (@rolldown/binding-darwin-arm64,
                        # @rollup/rollup-darwin-arm64) until someone
                        # reinstalled them. Lint and ``tsc -b`` kept passing,
                        # being pure JS/TS, so the breakage looked like a
                        # mysterious toolchain fault rather than a deploy
                        # side effect.
                        #
                        # VOLUME_COPY copies the source into a temporary
                        # Docker volume instead, so the container cannot write
                        # through to the tree. It is slower — the copy
                        # includes frontend/node_modules, which ``exclude=``
                        # above does not trim (see the note there). Correctness
                        # over deploy speed; if that cost ever matters, the
                        # fix is to build locally and upload ``dist/`` rather
                        # than to go back to bind-mounting.
                        "bundling_file_access": BundlingFileAccess.VOLUME_COPY,
                    },
                ),
            ],
            destination_bucket=bucket,
            distribution=self.distribution,
            # Hashed bundle filenames (assets/index-DEADBEEF.js) need no
            # invalidation — every build produces fresh hashes. The
            # entry-point HTML and any non-hashed top-level files do.
            distribution_paths=["/index.html", "/", "/manifest.json"],
            # This sizes the LAMBDA that replicates files from the CDK staging
            # bucket into this bucket — not the ``npm run build`` container.
            # Nothing here can starve the SPA build: bundling runs in podman on
            # the operator's machine and CDK imposes no memory cap on it, so a
            # vite/rolldown OOM would not be this number. Worth stating because
            # the two are easy to conflate when a deploy dies during bundling.
            #
            # Default (128 MB) was tight for the old build output and would have
            # OOM'd if we hadn't pruned the syntax-highlighter chunk explosion.
            # 512 MB is cheap insurance against future bundle growth
            # re-thrashing the Lambda — Lambda billing is per-invocation-ms × MB
            # so the incremental cost is a few cents per deploy at most.
            memory_limit=512,
        )

        # ---- Outputs -----------------------------------------------------
        CfnOutput(
            self,
            "DistributionDomainName",
            value=self.distribution.distribution_domain_name,
            description="CloudFront DNS — the public entry point. Use this as the Cognito callback host.",
            export_name="ai-eval-studio-cloudfront-domain",
        )
