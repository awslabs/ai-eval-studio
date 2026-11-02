#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# AI Eval Studio — one-time AWS prep for the first ``cdk deploy --all``.
#
# The ALB listens HTTP-only (auth is enforced in the FastAPI app — see
# ``backend/app/platform/aws/auth_cognito_bearer.py``), so there is
# nothing to import into ACM. CloudFront's free ``*.cloudfront.net``
# cert covers the user-facing TLS; the CloudFront ↔ ALB hop rides the
# AWS managed plane.
#
# Idempotent: safe to re-run (cdk bootstrap is a no-op after the first
# call).
#
# Usage:
#   ./infra/scripts/dev-prep.sh         # prints the export commands
#   eval "$(./infra/scripts/dev-prep.sh)"   # captures into the shell
#
# Outputs (export these before ``cdk deploy``):
#   COGNITO_DOMAIN_PREFIX   — globally unique Cognito hosted-domain prefix
#   CDK_DEFAULT_ACCOUNT     — account id from ``aws sts get-caller-identity``
#   CDK_DEFAULT_REGION      — defaults to AWS_REGION or us-east-1
#   DEPLOY_ENV              — defaults to dev (sets the Environment tag)
#   CDK_DOCKER              — podman, so ``cdk deploy``'s from_asset uses
#                             podman as the container engine
#   DOCKER_HOST             — podman socket so the cdk-spawned podman
#                             call connects to the running podman machine
set -euo pipefail
# A failure under ``set -e`` exits with nothing on stdout, and the caller's
# ``eval "$(...)"`` of an empty string succeeds, so the error would otherwise
# surface only later as an unset variable inside ``cdk deploy``. Name the line.
trap 'echo "ERROR: dev-prep.sh failed at line $LINENO" >&2' ERR

REGION="${AWS_REGION:-us-east-1}"
ACCT="$(aws sts get-caller-identity --query Account --output text)"

echo "# === AI Eval Studio dev-prep ==========================================" >&2
echo "# Account: $ACCT  Region: $REGION" >&2

# 1. CDK bootstrap (one-time per account+region; cdk handles the
#    idempotent case automatically). Bootstrap creates the asset ECR
#    that ContainerImage.from_asset pushes into and the staging bucket
#    that BucketDeployment uses.
#
#    ``cdk bootstrap`` re-reads and re-deploys the CDKToolkit stack under
#    the CALLER's own identity, so it needs permissions a deploy does not:
#    a credential that can deploy purely by assuming the cdk-hnb659fds-*
#    roles fails here on ``cloudformation:GetTemplate``. An already-
#    bootstrapped account is the normal case, so treat a failure as fatal
#    only when CDKToolkit genuinely is not there — otherwise ``set -e``
#    aborts the script, no exports reach stdout, and the caller's
#    ``eval "$(...)"`` silently yields an empty environment.
echo "# Step 1: cdk bootstrap aws://$ACCT/$REGION" >&2
if ! cdk bootstrap "aws://${ACCT}/${REGION}" >&2; then
    if aws cloudformation describe-stacks --stack-name CDKToolkit \
            --region "$REGION" >/dev/null 2>&1; then
        echo "# Step 1: bootstrap command failed, but CDKToolkit exists —" >&2
        echo "#         continuing. Its VERSION was not verified, so if a" >&2
        echo "#         later deploy complains about the bootstrap version," >&2
        echo "#         re-run this step with a credential that can update" >&2
        echo "#         the CDKToolkit stack." >&2
    else
        echo "ERROR: bootstrap failed and CDKToolkit does not exist in" >&2
        echo "  aws://${ACCT}/${REGION}. Nothing can deploy until it does." >&2
        exit 1
    fi
fi

# 2. Cognito domain prefix — globally unique across all AWS accounts.
#    A new deployment gets ``ai-eval-studio-`` plus 8 random characters.
# Cognito hosted-UI domains are internet-facing and globally unique, so a
# DERIVABLE prefix makes the live login endpoint findable and echoes part of the
# account id back. Hence random for a new deployment.
#
# But it must stay IDEMPOTENT: re-running this script has to yield the prefix
# already deployed, or the next `cdk deploy` REPLACES the hosted domain and
# breaks the SPA's OIDC config and every bookmarked login URL. So prefer, in
# order: an explicit override, the prefix already deployed (read from the auth
# stack, the same discover-rather-than-hardcode idiom used for the CloudFront
# origin SG), and only then a fresh random one.
#
# On a first deploy the auth stack does not exist and ``describe-stacks`` exits
# non-zero. That is the expected case, not an error, so the ``|| true`` keeps
# ``pipefail`` from turning it into a ``set -e`` abort.
DEPLOYED_PREFIX="$({ aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-auth --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='UserPoolDomainUrl'].OutputValue" \
    --output text 2>/dev/null || true; } | sed -E 's#^https://([^.]+)\..*#\1#')"

if [[ -n "${COGNITO_DOMAIN_PREFIX:-}" ]]; then
    DOMAIN_PREFIX="$COGNITO_DOMAIN_PREFIX"
    echo "# Step 2: Cognito prefix from the environment → $DOMAIN_PREFIX" >&2
elif [[ -n "$DEPLOYED_PREFIX" && "$DEPLOYED_PREFIX" != "None" ]]; then
    DOMAIN_PREFIX="$DEPLOYED_PREFIX"
    echo "# Step 2: Cognito prefix already deployed → $DOMAIN_PREFIX (reused)" >&2
else
    # ``head`` closes the pipe after 8 characters, so ``tr`` always dies of
    # SIGPIPE and ``pipefail`` reports 141. The output is complete by then, so
    # the status is ignored. Reading a fixed block of /dev/urandom instead
    # would not help: only about 14% of bytes survive ``tr -dc``, so the
    # prefix would sometimes come out short.
    DOMAIN_PREFIX="ai-eval-studio-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 8 || true)"
    echo "# Step 2: no deployment found — generated a fresh Cognito prefix" >&2
    echo "#         → $DOMAIN_PREFIX  (random on purpose: a derivable prefix" >&2
    echo "#         makes the login endpoint findable. Keep it somewhere.)" >&2
fi

# 3. Verify podman is reachable (CDK from_asset + BucketDeployment
#    bundling both shell out to it).
if ! command -v podman >/dev/null 2>&1; then
    echo "ERROR: podman not on PATH. Install: brew install podman" >&2
    exit 1
fi
if ! podman info >/dev/null 2>&1; then
    echo "ERROR: podman machine not running. Start: podman machine start" >&2
    exit 1
fi
PODMAN_SOCKET="$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}' 2>/dev/null || true)"
if [[ -z "$PODMAN_SOCKET" ]]; then
    echo "ERROR: could not determine podman socket. Run 'podman machine start' first." >&2
    exit 1
fi
echo "# Step 3: podman OK — socket at $PODMAN_SOCKET" >&2

echo "# === Export the following before 'cdk deploy --all' ===================" >&2

cat <<EOF
export CDK_DEFAULT_ACCOUNT="$ACCT"
export CDK_DEFAULT_REGION="$REGION"
export DEPLOY_ENV="${DEPLOY_ENV:-dev}"
export COGNITO_DOMAIN_PREFIX="$DOMAIN_PREFIX"
export CDK_DOCKER=podman
export DOCKER_HOST="unix://$PODMAN_SOCKET"
EOF
