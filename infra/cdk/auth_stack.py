# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth stack: Cognito User Pool + SPA client + hosted UI domain.

Single SPA public client. Auth is enforced inside the FastAPI
coordinator against the Cognito JWKS endpoint (see
``backend/app/platform/aws/auth_cognito_bearer.py``); the ALB has no
auth action, so no confidential ALB client is needed.

**Self-signup is disabled.** With Bedrock spend
proportional to usage and no per-user cap, anyone who finds the
CloudFront DNS could otherwise sign up, click an email-verification
link, and start running evals on our account. New users join only via:

    aws cognito-idp admin-create-user \\
        --user-pool-id <pool> \\
        --username alice@example.com \\
        --user-attributes Name=email,Value=alice@example.com \\
                          Name=email_verified,Value=true

Cognito emails the user a temp password; first sign-in forces a reset.
See ``docs/runbooks/deploy.md`` for the full onboarding command.

**SPA client callback URLs.** The SPA's `oidc-client-ts` flow needs the
CloudFront DNS as the OAuth ``redirect_uri`` and ``logout_uri``. Because
CloudFront is created in :class:`FrontendStack` and depending on it
would create a cycle, the CloudFront DNS is passed in via CDK context:

* First deploy uses ``cloudfront_dns`` from ``cdk.context.json`` (or
  the placeholder fallback below if absent), so synth never breaks.
* ``infra/scripts/dev-postdeploy.sh`` reads the real CloudFront DNS
  from the frontend stack's CFN output, writes it into the context
  store, and re-deploys ``ai-eval-studio-auth`` so the SPA client
  picks up the real callback + logout URLs.

**Refresh-token rotation.** Setting
``refresh_token_rotation_grace_period=Duration.seconds(30)`` enables
rotation (any non-``None`` value enables; the value is the grace
window during which the prior refresh token still works to absorb
client-side retries). Property name verified against the bundled CDK
source.

Password policy: 8+ chars with lower/upper/digit.
"""

from __future__ import annotations

import os

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cognito as cognito,
)
from constructs import Construct


# Synth-time placeholder used on the very first deploy (when CloudFront
# DNS isn't known yet). ``dev-postdeploy.sh`` re-runs ``cdk deploy
# ai-eval-studio-auth`` with the real CloudFront DNS once the frontend
# stack is up; the placeholder never reaches a runtime callback because
# the SPA fetches its OIDC config from the backend's ``/api/config``
# endpoint and the OIDC handshake won't even start until that returns
# the production-shaped Cognito metadata.
_PLACEHOLDER_CLOUDFRONT_DNS = "example.invalid"


class AuthStack(Stack):
    """Cognito User Pool + SPA app client + hosted UI domain.

    Attributes:
        user_pool: The shared Cognito User Pool.
        user_pool_domain: Cognito-hosted UI domain — the OIDC IdP for the
            SPA's PKCE flow.
        spa_client: Public client for the React SPA.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Resolve the CloudFront DNS for the SPA's callback + logout URLs.
        # First deploy: missing → placeholder. ``dev-postdeploy.sh`` reads
        # the real CloudFront DNS from the frontend stack's CFN output,
        # writes it into the context, and re-runs ``cdk deploy
        # ai-eval-studio-auth``.
        cloudfront_dns = (
            self.node.try_get_context("cloudfront_dns")
            or os.environ.get("CLOUDFRONT_DNS")
            or _PLACEHOLDER_CLOUDFRONT_DNS
        )
        spa_callback_url = f"https://{cloudfront_dns}/auth/callback"
        spa_logout_url = f"https://{cloudfront_dns}/"

        # RemovalPolicy.RETAIN — Cognito UserPool defaults to RETAIN already
        # but we make it explicit so a future maintainer reading this file
        # doesn't mis-edit the default-DESTROY out of habit. User accounts
        # MUST survive iteration cycles.
        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name="ai-eval-studio-users",
            # Admin-create only. ``auto_verify`` is dropped because there
            # is no self-signup flow to verify; keeping it enabled would
            # just be misleading config.
            #
            # Quirk: CDK still synthesizes ``AutoVerifiedAttributes: [email]``
            # at the CFN level because it auto-fills that from
            # ``sign_in_aliases=email``. With ``AllowAdminCreateUserOnly:
            # true`` (set by ``self_sign_up_enabled=False``) and
            # admin-created users carrying ``email_verified=true``
            # explicitly, the residual attribute is dormant — it only
            # gates self-signup and forgot-password flows that don't apply
            # here. Suppressing it would require passing an explicit
            # ``auto_verify=cognito.AutoVerifiedAttrs()`` and isn't worth
            # fighting CDK over.
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=False),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=False,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            # Declared explicitly rather than left to whatever plan the pool
            # happens to be on, so a pool recreated from this template lands on
            # the same plan as the one it replaces.
            feature_plan=cognito.FeaturePlan.ESSENTIALS,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Cognito-hosted UI domain. Must be CREATED (not looked up via
        # ``UserPoolDomain.from_domain_name``, which was a bug in earlier
        # iterations). The prefix is globally unique across all AWS
        # accounts; ``dev-prep.sh`` reuses the prefix already deployed and
        # otherwise generates ``ai-eval-studio-`` plus 8 random characters,
        # because a prefix derivable from the account id would make the login
        # endpoint findable.
        #
        # No fallback. A fixed default here is exactly such a findable prefix,
        # and it was applied without a word: a deploy from a shell without the
        # prep exports created a real domain with it. Changing the prefix of a
        # live pool replaces its domain and moves the SPA's login URL, so it has
        # to be refused before anything is created, not noticed afterwards.
        domain_prefix = os.environ.get("COGNITO_DOMAIN_PREFIX")
        if not domain_prefix:
            raise ValueError(
                "COGNITO_DOMAIN_PREFIX is not set. Run "
                '`eval "$(./infra/scripts/dev-prep.sh)"` in this shell first: it '
                "reuses the prefix of an existing deployment, or generates an "
                "unguessable one for a new deployment."
            )
        self.user_pool_domain = cognito.UserPoolDomain(
            self,
            "UserPoolDomain",
            user_pool=self.user_pool,
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
        )

        # Public client for the React SPA — no secret, auth code grant +
        # refresh, refresh-token rotation enabled (30s grace window for
        # client-side retries). Design points:
        # * Authorization Code + PKCE against the hosted UI.
        # * Callback URL = CloudFront DNS + ``/auth/callback``.
        # * Logout URL = CloudFront DNS + ``/``.
        # * Refresh-token rotation ON via the ``grace_period`` property.
        # * ``access_token_validity=1h`` matches the SPA's silent-renew
        #   schedule (renews ~5 min before expiry).
        # Notes — properties intentionally NOT set:
        # * ``enable_token_revocation`` — defaults to True.
        # * ``auth_session_validity`` — accept the AWS service default
        #   of 3 minutes. Plenty for the hosted UI flow.
        # * ``enable_propagate_additional_user_context_data`` — requires
        #   ``generate_secret=True``; this client is public.
        self.spa_client = self.user_pool.add_client(
            "SpaClient",
            user_pool_client_name="spa",
            generate_secret=False,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    authorization_code_grant=True,
                ),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL],
                callback_urls=[spa_callback_url],
                logout_urls=[spa_logout_url],
            ),
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
            auth_flows=cognito.AuthFlow(user_srp=True),
            refresh_token_rotation_grace_period=Duration.seconds(30),
            # Return identical "user does not exist or password is
            # wrong" responses regardless of whether the username
            # exists. Without this, the distinct error code leaks the
            # enrolled-user list to anyone who can reach the hosted UI.
            prevent_user_existence_errors=True,
        )

        # ---- Outputs (consumed by the SPA via the backend's /api/config) -----
        # The SPA reads the user pool id, client id, region, and hosted UI
        # domain from ``GET /api/config`` (the backend sources them from env
        # vars set at Fargate task launch — see compute_stack.py).
        CfnOutput(
            self,
            "UserPoolId",
            value=self.user_pool.user_pool_id,
            description="Cognito User Pool id. Backend env var: COGNITO_USER_POOL_ID.",
            export_name="ai-eval-studio-user-pool-id",
        )
        CfnOutput(
            self,
            "SpaClientId",
            value=self.spa_client.user_pool_client_id,
            description="SPA Cognito client id. Backend env var: COGNITO_SPA_CLIENT_ID.",
            export_name="ai-eval-studio-spa-client-id",
        )
        CfnOutput(
            self,
            "UserPoolDomainUrl",
            value=(
                f"https://{self.user_pool_domain.domain_name}"
                f".auth.{self.region}.amazoncognito.com"
            ),
            description="Cognito hosted UI base URL. Backend env var: COGNITO_DOMAIN.",
            export_name="ai-eval-studio-user-pool-domain",
        )
