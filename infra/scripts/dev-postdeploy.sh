#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# AI Eval Studio — post-deploy wiring after the first ``cdk deploy --all``.
#
# Reads the **CloudFront** DNS from the frontend stack's CFN output
# and re-deploys ``ai-eval-studio-auth`` with the real DNS so the SPA
# Cognito client picks up the production callback + logout URLs. The
# auth stack reads ``cloudfront_dns`` from CDK context, so a single
# ``cdk deploy ai-eval-studio-auth -c cloudfront_dns=<dns>`` is enough.
#
# Also reads the **AgentCore runtime ARN** from the agentcore stack's CFN
# output, because the compute leg has to re-pass it or the coordinator
# loses ``AGENTCORE_RUNTIME_ARN``.
#
# And the **CloudFront VPC-origin security group**, which AWS creates in the
# ALB's VPC only once the frontend stack's VPC origin exists. The network stack
# needs it to authorize ingress to the ALB, but network deploys BEFORE frontend,
# so it cannot be a CloudFormation reference. Until this runs, ``/api/*`` and
# ``/ws/*`` return 504.
#
# Every value this script needs is discoverable from CFN outputs or the EC2
# API — no operator-supplied flags.
#
# Idempotent: safe to re-run.
#
# Prerequisite: ``dev-prep.sh`` exports must already be in the shell
# (CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION, COGNITO_DOMAIN_PREFIX,
# CDK_DOCKER, DOCKER_HOST). Fails loudly if any are missing.
set -euo pipefail

# ---- Sanity-check the env from dev-prep ----
missing=()
for v in CDK_DEFAULT_ACCOUNT CDK_DEFAULT_REGION \
         COGNITO_DOMAIN_PREFIX CDK_DOCKER DOCKER_HOST; do
    if [[ -z "${!v:-}" ]]; then
        missing+=("$v")
    fi
done
if (( ${#missing[@]} )); then
    echo "ERROR: missing env vars: ${missing[*]}" >&2
    echo "  Run: eval \"\$(./infra/scripts/dev-prep.sh)\"" >&2
    exit 1
fi

# ---- Read CloudFront DNS from the frontend stack's CFN output ----
echo "# Reading CloudFront DNS from ai-eval-studio-frontend outputs..." >&2
CF_DNS="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-frontend \
    --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
    --output text)"

if [[ -z "$CF_DNS" || "$CF_DNS" == "None" ]]; then
    echo "ERROR: DistributionDomainName output not found on ai-eval-studio-frontend." >&2
    echo "  Has 'cdk deploy --all' completed successfully?" >&2
    exit 1
fi

echo "# CloudFront DNS: $CF_DNS" >&2

# ---- Find the CloudFront VPC-origin security group ----
# AWS creates ``CloudFront-VPCOrigins-Service-SG`` in the ALB's VPC when the
# first VPC origin appears — which happens during the frontend deploy, AFTER
# the network stack that needs to reference it. That ordering is why the id
# cannot be a CloudFormation reference and is discovered here instead.
#
# Without this rule the ALB security group has NO ingress and CloudFront
# returns 504 for every /api/* and /ws/* request while the SPA keeps serving
# from S3.
echo "# Reading VPC id from ai-eval-studio-network outputs..." >&2
VPC_ID="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-network \
    --query "Stacks[0].Outputs[?OutputKey=='VpcId'].OutputValue" \
    --output text)"

if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "ERROR: VpcId output not found on ai-eval-studio-network." >&2
    echo "  Deploy (or redeploy) that stack first." >&2
    exit 1
fi

echo "# Looking up CloudFront-VPCOrigins-Service-SG in $VPC_ID..." >&2
CF_ORIGIN_SG_ID="$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=CloudFront-VPCOrigins-Service-SG" \
              "Name=vpc-id,Values=${VPC_ID}" \
    --query "SecurityGroups[].GroupId" \
    --output text)"

# Scoped to one VPC, so more than one match means the assumption behind this
# lookup no longer holds; picking arbitrarily would authorize the wrong group.
if [[ "$CF_ORIGIN_SG_ID" == *[[:space:]]* ]]; then
    echo "ERROR: multiple CloudFront-VPCOrigins-Service-SG groups in ${VPC_ID}:" >&2
    echo "  $CF_ORIGIN_SG_ID" >&2
    echo "  Pass the right one explicitly with" >&2
    echo "  -c cloudfront_vpc_origin_sg_id=sg-xxxx" >&2
    exit 1
fi

if [[ -z "$CF_ORIGIN_SG_ID" || "$CF_ORIGIN_SG_ID" == "None" ]]; then
    echo "ERROR: no CloudFront-VPCOrigins-Service-SG in ${VPC_ID}." >&2
    echo "  AWS creates it with the first VPC origin, so this means the" >&2
    echo "  frontend stack's VPC origin does not exist yet. Deploy" >&2
    echo "  ai-eval-studio-frontend, then re-run this script." >&2
    exit 1
fi

echo "# CloudFront VPC-origin SG: $CF_ORIGIN_SG_ID" >&2

# Every cdk invocation below must run from infra/cdk, where cdk.json lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../cdk"

# ---- Re-deploy network so the ALB admits CloudFront's VPC origin ----
# First, because it is what makes /api/* and /ws/* reachable at all. Until it
# runs, the ALB security group has no ingress rule and CloudFront answers 504
# on every API request while the SPA still loads from S3 — a failure that looks
# like a backend outage rather than a missing firewall rule.
echo "# Re-deploying ai-eval-studio-network with cloudfront_vpc_origin_sg_id context..." >&2
cdk deploy ai-eval-studio-network \
    --exclusively \
    --context "cloudfront_vpc_origin_sg_id=${CF_ORIGIN_SG_ID}" \
    --require-approval never

# ---- Resolve the AgentCore runtime subnet ----------------------------
# AgentCore Runtime only runs in a subset of Availability Zones, and AWS
# identifies those by AZ-**ID** (``use1-az1``) rather than AZ **name**
# (``us-east-1a``). AZ names are shuffled per account while AZ-IDs are stable, so
# the supported set can only be expressed as IDs — and the name<->ID mapping has
# to be resolved per account, at runtime.
#
# That is why this cannot be a committed default: a subnet id is account-specific,
# so a hardcoded one makes the repo undeployable for everyone else. The supported
# AZ-ID list below is a published property of the service, not of any account.
AGENTCORE_SUPPORTED_AZ_IDS="use1-az1,use1-az2,use1-az4"

echo "# Resolving AgentCore-supported AZs (${AGENTCORE_SUPPORTED_AZ_IDS}) to this account's AZ names..." >&2
AGENTCORE_AZ_NAMES="$(aws ec2 describe-availability-zones \
    --query "AvailabilityZones[?contains('${AGENTCORE_SUPPORTED_AZ_IDS}', ZoneId)].ZoneName" \
    --output text | tr '\t' ',')"

if [[ -z "$AGENTCORE_AZ_NAMES" ]]; then
    echo "# WARNING: none of the AgentCore-supported AZ-IDs" >&2
    echo "#   (${AGENTCORE_SUPPORTED_AZ_IDS}) exist in ${CDK_DEFAULT_REGION}." >&2
    echo "#   Those ids are us-east-1's. Deploying elsewhere therefore skips the" >&2
    echo "#   AgentCore runtime: the app works, the relay path has no invoke" >&2
    echo "#   target. Update AGENTCORE_SUPPORTED_AZ_IDS if AWS has widened" >&2
    echo "#   support for your region." >&2
fi
echo "# Supported AZ names here: $AGENTCORE_AZ_NAMES" >&2

# PRIVATE_WITH_EGRESS only — the runtime needs NAT egress and must not be public.
# CDK tags those subnets ``aws-cdk:subnet-type=Private``.
AGENTCORE_SUBNET_IDS="$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
              "Name=availability-zone,Values=${AGENTCORE_AZ_NAMES}" \
              "Name=tag:aws-cdk:subnet-type,Values=Private" \
    --query "Subnets[].SubnetId" \
    --output text | tr '\t' ',')"

if [[ -z "$AGENTCORE_SUBNET_IDS" ]]; then
    echo "# WARNING: no PRIVATE_WITH_EGRESS subnet in ${VPC_ID} lands in an" >&2
    echo "#   AgentCore-supported AZ. The VPC spans max_azs=2 of the region's" >&2
    echo "#   AZs and AZ names map to AZ-IDs per account, so both can miss." >&2
    echo "#   Skipping the AgentCore runtime; everything else deploys." >&2
else
    echo "# AgentCore runtime subnet(s): $AGENTCORE_SUBNET_IDS" >&2
fi


# Cognito hosted UI domain — derived from the prefix dev-prep.sh
# exported. Frontend stack's CSP ``connect-src`` directive needs this so
# the SPA's OIDC redirect to Cognito isn't blocked by the browser.
COGNITO_DOMAIN="https://${COGNITO_DOMAIN_PREFIX}.auth.${CDK_DEFAULT_REGION}.amazoncognito.com"
echo "# Cognito hosted UI: $COGNITO_DOMAIN" >&2


# Persist context to cdk.context.json so subsequent ``cdk deploy``
# invocations pick up the post-deploy-discovered values without
# requiring the operator to re-pass --context flags. Each per-stack
# redeploy below still uses --context so the merge is explicit, but
# any *future* operator-driven ``cdk deploy ai-eval-studio-frontend``
# (or similar) will read these values from the persisted file
# automatically. Earlier deploys without these persisted regressed
# the CSP and LOGOUT_REDIRECT_URL to their placeholder fallbacks.
#
# agentcore_runtime_arn joins them further down, once the runtime has been
# deployed and its ARN read back.
echo "# Persisting discovered context (cloudfront_dns, cognito_domain, logout_redirect_url, cloudfront_vpc_origin_sg_id, agentcore_subnet_ids; agentcore_runtime_arn once read) to cdk.context.json..." >&2
python3 - <<PYEOF
import json, os
ctx_path = "cdk.context.json"
ctx = {}
if os.path.exists(ctx_path):
    with open(ctx_path) as f:
        ctx = json.load(f)
ctx["cloudfront_dns"] = "${CF_DNS}"
ctx["cognito_domain"] = "${COGNITO_DOMAIN}"
ctx["logout_redirect_url"] = "https://${CF_DNS}/"
ctx["cloudfront_vpc_origin_sg_id"] = "${CF_ORIGIN_SG_ID}"
ctx["agentcore_subnet_ids"] = "${AGENTCORE_SUBNET_IDS}"
with open(ctx_path, "w") as f:
    json.dump(ctx, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF

# ---- Redeploy auth with the real CloudFront DNS in context ----------
# ``--exclusively`` prevents CDK from re-synthesizing dependent stacks
# (frontend bundles the SPA via podman, ~60s). Auth has no dependencies
# of its own to deploy, so excluding the cascade is safe.
echo "# Re-deploying ai-eval-studio-auth with cloudfront_dns context..." >&2
cdk deploy ai-eval-studio-auth \
    --exclusively \
    --context "cloudfront_dns=${CF_DNS}" \
    --require-approval never

# ---- Re-deploy storage so the staging bucket's CORS picks up the real
# CloudFront origin. Without this, ``allowed_origins`` stays pinned to
# the placeholder DNS from the first deploy and any future SPA upload
# via direct-to-S3 PUT gets CORS-blocked. The presigned-URL signature
# is the actual gate, so this is defense-in-depth — but the CORS rule
# is a real browser-side gate the SPA must pass.
echo "# Re-deploying ai-eval-studio-storage with cloudfront_dns context..." >&2
cdk deploy ai-eval-studio-storage \
    --exclusively \
    --context "cloudfront_dns=${CF_DNS}" \
    --require-approval never

# ---- Redeploy frontend with cognito_domain so CSP allows OIDC -------
# Without this, the synthesized CSP keeps the placeholder
# ``https://example.invalid``, which the browser refuses to connect to,
# breaking the SPA's redirect to the Cognito hosted UI.
echo "# Re-deploying ai-eval-studio-frontend with cognito_domain context..." >&2
cdk deploy ai-eval-studio-frontend \
    --exclusively \
    --context "cognito_domain=${COGNITO_DOMAIN}" \
    --require-approval never

# ---- Re-deploy agentcore so backend changes reach the microVM -------
# Closes the coordinator-vs-microVM deploy gap: a backend change is
# automatically rebuilt into the ARM64 microVM image (content-hash asset)
# and the runtime version is bumped, so the deployed agentcore runtime
# serves current code instead of drifting stale. Without this leg, the
# microVM image stays pinned to the original cdk deploy --all build and
# backend changes (like the ABAC session-tag minting) never reach it.
#
# -c agentcore_subnet_ids: the value discovered above. It is NOT a committed
# default — a subnet id is account-specific, so hardcoding one would make
# this repo undeployable in any other account. Until this leg runs, the
# agentcore stack synthesizes with NO runtime and warns saying so.
#
# iam and compute deploy in the same command, BEFORE agentcore. The runtime
# imports three values nothing else imports — the runtime role ARN from iam,
# and the cluster ARN and coordinator security group from compute — and CDK
# creates a cross-stack export only once something references it. On the
# first ``cdk deploy --all`` the agentcore stack has no runtime yet, so those
# exports were never created, and deploying agentcore alone failed with
# ``No export named ...``. With all three named, CDK deploys them in
# dependency order, so the exports exist before agentcore imports them.
#
# --exclusively still limits the deploy to those three stacks; the rest are
# already up. compute reads ``logout_redirect_url`` from the context persisted
# above. It renders ``agentcore_runtime_arn`` only when it is in context. A
# previous run saved it to cdk.context.json, but a checkout without that file has
# none, so pass the deployed runtime's ARN explicitly when there is one:
# otherwise this leg would strip AGENTCORE_RUNTIME_ARN from the coordinator until
# the compute leg below restores it.
if [[ -z "$AGENTCORE_SUBNET_IDS" ]]; then
    echo "# Skipping the agentcore leg — no supported-AZ subnet (see warning above)." >&2
    AGENTCORE_RUNTIME_ARN=""
else
EXISTING_RUNTIME_ARN="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-agentcore \
    --query "Stacks[0].Outputs[?OutputKey=='RuntimeArn'].OutputValue" \
    --output text 2>/dev/null || true)"
agentcore_ctx=( --context "agentcore_subnet_ids=${AGENTCORE_SUBNET_IDS}" )
if [[ -n "$EXISTING_RUNTIME_ARN" && "$EXISTING_RUNTIME_ARN" != "None" ]]; then
    agentcore_ctx+=( --context "agentcore_runtime_arn=${EXISTING_RUNTIME_ARN}" )
fi
echo "# Re-deploying ai-eval-studio-iam, -compute and -agentcore with agentcore_subnet_ids context..." >&2
cdk deploy ai-eval-studio-iam ai-eval-studio-compute ai-eval-studio-agentcore \
    --exclusively \
    "${agentcore_ctx[@]}" \
    --require-approval never
fi

# ---- Read the AgentCore runtime ARN from its stack's CFN output ----
# The compute stack renders ``AGENTCORE_RUNTIME_ARN`` into the coordinator
# container only when ``agentcore_runtime_arn`` is in CDK context
# (compute_stack.py). It is discoverable rather than operator-supplied, so
# read it here instead of asking for a flag: without it the compute leg
# below would re-render the task definition WITHOUT the var and the relay
# path would lose its invoke target — the same regression class this file
# already guards against for the CSP and LOGOUT_REDIRECT_URL.
if [[ -z "$AGENTCORE_SUBNET_IDS" ]]; then
    echo "# Skipping the runtime-ARN read — the agentcore leg was skipped." >&2
    AGENTCORE_RUNTIME_ARN=""
else
    echo "# Reading AgentCore runtime ARN from ai-eval-studio-agentcore outputs..." >&2
    AGENTCORE_RUNTIME_ARN="$(aws cloudformation describe-stacks \
        --stack-name ai-eval-studio-agentcore \
        --query "Stacks[0].Outputs[?OutputKey=='RuntimeArn'].OutputValue" \
        --output text)"

    if [[ -z "$AGENTCORE_RUNTIME_ARN" || "$AGENTCORE_RUNTIME_ARN" == "None" ]]; then
        echo "ERROR: the agentcore leg ran but produced no RuntimeArn output." >&2
        echo "  Deploying compute without it would strip AGENTCORE_RUNTIME_ARN" >&2
        echo "  from the coordinator and break the relay path. Fix the agentcore" >&2
        echo "  stack first, then re-run this script." >&2
        exit 1
    fi
    echo "# AgentCore runtime: $AGENTCORE_RUNTIME_ARN" >&2
fi

# ---- Persist the runtime ARN for later deploys from this checkout ----
# Every compute deploy renders AGENTCORE_RUNTIME_ARN only when
# ``agentcore_runtime_arn`` is in CDK context, so a plain
# ``cdk deploy ai-eval-studio-compute`` without the flag used to strip it and
# break the relay. Saving it here lets later deploys from this checkout read it
# from cdk.context.json, as they already do the keys persisted above. It is
# written only after it has been read and validated, and overwritten on every
# run, so it tracks the runtime that is deployed now. An explicit
# ``-c agentcore_runtime_arn=...`` still overrides it.
#
# When the agentcore leg was skipped there is no ARN, and the file is left as
# it is rather than emptied. The value goes through the environment, not the
# heredoc, so nothing in it is interpreted by the shell or by Python.
if [[ -n "$AGENTCORE_RUNTIME_ARN" ]]; then
    echo "# Persisting agentcore_runtime_arn to cdk.context.json..." >&2
    ARN="$AGENTCORE_RUNTIME_ARN" python3 - <<'PYEOF'
import json, os
ctx_path = "cdk.context.json"
ctx = {}
if os.path.exists(ctx_path):
    with open(ctx_path) as f:
        ctx = json.load(f)
ctx["agentcore_runtime_arn"] = os.environ["ARN"]
with open(ctx_path, "w") as f:
    json.dump(ctx, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF
fi

# ---- Re-deploy compute so LOGOUT_REDIRECT_URL is non-default --------
# Both context values are mandatory on EVERY compute deploy: each one is
# rendered into the container env only when present, so omitting either
# silently drops it. See docs/runbooks/deploy.md.
echo "# Re-deploying ai-eval-studio-compute with logout_redirect_url + agentcore_runtime_arn context..." >&2
compute_ctx=( --context "logout_redirect_url=https://${CF_DNS}/" )
if [[ -n "$AGENTCORE_RUNTIME_ARN" ]]; then
    compute_ctx+=( --context "agentcore_runtime_arn=${AGENTCORE_RUNTIME_ARN}" )
else
    echo "# NOTE: deploying compute WITHOUT agentcore_runtime_arn — the relay" >&2
    echo "#   path will have no invoke target. Everything else works." >&2
fi
cdk deploy ai-eval-studio-compute \
    --exclusively \
    "${compute_ctx[@]}" \
    --require-approval never

# ---- Assert MMDSv2 is required on the AgentCore runtime --------------
# The microVM Metadata Service hands the runtime's execution-role credentials
# to any code running in the VM, the way IMDS does on EC2. MMDSv2 makes that a
# token-first request, which stops an SSRF from reaching them. It does NOT stop
# local code — a shell in the microVM performs the token PUT itself — so this
# is not a tenancy boundary, only the difference between remote and local reach.
#
# ASSERTED, never set, and both reasons are structural:
#
#   * ``AWS::BedrockAgentCore::Runtime`` has no MetadataConfiguration property.
#     CloudFormation cannot express it, so agentcore_stack.py cannot either.
#   * ``UpdateAgentRuntime`` is a full-replace PUT — agentRuntimeArtifact,
#     agentRuntimeId and roleArn are all required. Setting the flag here would
#     mean restating the entire runtime definition in shell, silently clearing
#     any optional field the call omitted, and bumping the runtime version
#     (which resets managed session storage). That is a worse failure than the
#     one it fixes.
#
# AWS turned the flag on during its own rollout, so a runtime that predates
# this check has it. A runtime recreated from our template may not, and the
# regression is invisible from the outside: the app keeps working while the
# credentials become readable without a token. Hence a hard failure with the
# out-of-band remedy spelled out.
if [[ -n "$AGENTCORE_RUNTIME_ARN" ]]; then
    AGENTCORE_RUNTIME_ID="${AGENTCORE_RUNTIME_ARN##*/}"
    echo "# Checking MMDSv2 is required on runtime ${AGENTCORE_RUNTIME_ID}..." >&2
    MMDSV2_REQUIRED="$(aws bedrock-agentcore-control get-agent-runtime \
        --agent-runtime-id "$AGENTCORE_RUNTIME_ID" \
        --query "metadataConfiguration.requireMMDSV2" \
        --output text 2>/dev/null || echo "unreadable")"
    if [[ "$MMDSV2_REQUIRED" != "True" && "$MMDSV2_REQUIRED" != "true" ]]; then
        echo "ERROR: MMDSv2 is not confirmed on the AgentCore runtime." >&2
        echo "  requireMMDSV2 read back as: ${MMDSV2_REQUIRED}" >&2
        echo "  ('unreadable' also means an AWS CLI without get-agent-runtime," >&2
        echo "   or a role lacking bedrock-agentcore:GetAgentRuntime.)" >&2
        echo "" >&2
        echo "  Why this matters: MMDS serves the runtime execution role's" >&2
        echo "  credentials to any code in the microVM. Without MMDSv2 a plain" >&2
        echo "  GET reaches them, with no token step." >&2
        echo "" >&2
        echo "  CloudFormation has no property for this, so it cannot be fixed" >&2
        echo "  in the CDK stack. Set it out of band:" >&2
        echo "    aws bedrock-agentcore-control update-agent-runtime \\" >&2
        echo "        --agent-runtime-id ${AGENTCORE_RUNTIME_ID} \\" >&2
        echo "        --metadata-configuration requireMMDSV2=true \\" >&2
        echo "        <plus agentRuntimeArtifact, roleArn, and EVERY other field" >&2
        echo "         the runtime already has — this call is a full replace>" >&2
        echo "  Read the current values with get-agent-runtime first." >&2
        exit 1
    fi
    echo "# MMDSv2: required" >&2
fi

# ---- Verify the API path actually answers ---------------------------
# The deploy legs above can all report success while ``/api/*`` is dead: the
# ALB's security group is populated from CDK context, so skipping this script
# (or losing ``cloudfront_vpc_origin_sg_id`` from cdk.context.json) synthesizes
# an ALB with ZERO ingress rules and every API call returns 504, while
# CloudFormation reports every stack update as successful.
#
# Deliberately an end-to-end HTTP assertion rather than a check for the context
# value: it catches the 504 class regardless of cause — wrong security group,
# unhealthy target, listener misconfiguration, a cause nobody has thought of —
# instead of only the one failure already understood.
#
# NOT a CDK-level guard. The network stack must stay deployable before the
# CloudFront VPC origin exists (AWS creates that security group only once the
# frontend stack's origin does), so failing synth on a missing context value
# would break the first deploy into a fresh account. Asserting after the fact is
# what works for both the bootstrap case and every later one.
#
# Retries because CloudFront config propagation and ALB target registration are
# both eventually-consistent; a 504 in the first seconds is expected.
API_URL="https://${CF_DNS}/api/config"
API_MAX_ATTEMPTS=18
API_RETRY_DELAY=10

echo "# Verifying the API path answers: ${API_URL}" >&2
api_code="000"
for (( attempt = 1; attempt <= API_MAX_ATTEMPTS; attempt++ )); do
    # ``|| echo 000`` keeps a connection failure from tripping ``set -e`` — a
    # refused connection is a retryable state here, not a script bug.
    api_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$API_URL" || echo "000")"
    if [[ "$api_code" == "200" ]]; then
        echo "# API reachable — HTTP 200 after ${attempt} attempt(s)." >&2
        break
    fi
    if (( attempt < API_MAX_ATTEMPTS )); then
        echo "#   attempt ${attempt}/${API_MAX_ATTEMPTS}: HTTP ${api_code} — retrying in ${API_RETRY_DELAY}s" >&2
        sleep "$API_RETRY_DELAY"
    fi
done

if [[ "$api_code" != "200" ]]; then
    echo "ERROR: ${API_URL} returned HTTP ${api_code} after ~$(( API_MAX_ATTEMPTS * API_RETRY_DELAY ))s." >&2
    echo "The deploy completed but the API is NOT reachable. Do not treat this as a success." >&2
    echo "" >&2
    echo "Most likely cause — the ALB security group has no ingress rule." >&2
    echo "Confirm the rule exists (expect one rule on port 80 from the CloudFront VPC-origin SG):" >&2
    cat <<EOF >&2
   ALB_SG=\$(aws cloudformation describe-stacks --stack-name ai-eval-studio-network \\
       --query "Stacks[0].Outputs[?OutputKey=='AlbSecurityGroupId'].OutputValue" --output text)
   aws ec2 describe-security-group-rules \\
       --filters "Name=group-id,Values=\$ALB_SG" \\
       --query "SecurityGroupRules[?!IsEgress]"
EOF
    echo "" >&2
    echo "If it is missing, cloudfront_vpc_origin_sg_id is absent from cdk.context.json." >&2
    echo "Re-run this script: it discovers the value, persists it, and redeploys the network stack." >&2
    exit 1
fi

echo "# === Post-deploy complete ============================================" >&2
echo "# CloudFront:  https://$CF_DNS/" >&2
echo "# Onboard the first user with:" >&2
cat <<EOF >&2
   aws cognito-idp admin-create-user \\
       --user-pool-id <pool id from auth-stack outputs> \\
       --username you@example.com \\
       --user-attributes Name=email,Value=you@example.com \\
                         Name=email_verified,Value=true
EOF
