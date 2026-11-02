# AI Eval Studio — first cloud deploy

Mirror of [`teardown.md`](teardown.md) for the deploy direction. This is
the dev path — CloudFront's default `*.cloudfront.net` cert, no custom
domain. Production hardening (a real ACM cert in us-east-1, Route 53
alias, WAF, end-to-end TLS to the ALB) is out of scope here; see the path
to production in [`SECURITY.md`](../../SECURITY.md).

**Architecture:** CloudFront → internal ALB (HTTP) → Fargate (HTTPS).
Auth is enforced inside the FastAPI coordinator against Cognito JWKS. The
ALB's listener has no authentication and no TLS cert; the ALB reaches the
coordinator over HTTPS, using a self-signed certificate the container
generates at start-up (`app.scripts.serve_tls`). WebSocket traffic flows
over the same path as REST via the `/ws/*` CloudFront behavior.

**Upgrading a deployment from before customer-managed keys and the HTTPS
target group:** it cannot be updated in place. Both changes make
CloudFormation replace a resource another stack imports — the coordinator
target group, and the EFS file system, whose new key forces a new one — so
the deploy is refused. Tear the old deployment down with
[`teardown.md`](teardown.md), then follow this runbook from the start. Data on
the old deployment does not carry over unless you back it up first.

**Redeploying from a checkout that deployed before:** delete
`infra/cdk/cdk.context.json` first (teardown step 13). It is gitignored and
still holds the old deployment's discovered values — subnet, CloudFront DNS,
CloudFront's origin security group — which this deploy would otherwise reuse.

**Every `cdk` command in this file runs from `infra/cdk`** (where
`cdk.json` is) with `infra/.venv` activated, in the same shell as the
`dev-prep.sh` prep. Shell commands with a repo-relative path (`./infra/...`)
run from the repo root. `infra/tests/test_deploy_docs_cwd.py` pins the first
half of that.

## Pre-flight

**Everything below happens in ONE shell.** `dev-prep.sh` only *prints*
exports — the `eval` is what puts them in the current shell, and they are
gone in the next one. A `cdk deploy` run in a different shell from the prep
silently gets an unconfigured environment.

1. **AWS credentials.** `aws sts get-caller-identity` must return your
   sandbox account. The deploy user needs either `AdministratorAccess`, or
   `sts:AssumeRole` on `cdk-hnb659fds-*` (CDK assumes the bootstrap roles to
   deploy, so the second is enough for `cdk deploy` — but not for
   `cdk bootstrap`, which runs under your own identity and needs
   `cloudformation:GetTemplate` on `CDKToolkit`. `dev-prep.sh` continues past
   a bootstrap failure when `CDKToolkit` already exists).
2. **podman.** `podman --version` should print 5+. If `podman info`
   complains, run `podman machine start`. CDK shells out to podman to
   build the backend image (Dockerfile) AND the SPA bundle
   (BucketDeployment) during `cdk deploy`.
3. **Run the prep script.** It bootstraps CDK, picks a Cognito hosted-domain
   prefix (your `COGNITO_DOMAIN_PREFIX` if set, else the one already deployed,
   else a fresh random one), verifies podman, and prints the exports:
   ```bash
   eval "$(./infra/scripts/dev-prep.sh)"
   ```
   Idempotent — safe to re-run if you change account/region. Diagnostics go
   to stderr and the exports to stdout, so the `eval` captures only the
   exports and you still see the progress lines.
4. **Confirm exports landed.** `echo "$CDK_DEFAULT_ACCOUNT
   $COGNITO_DOMAIN_PREFIX $CDK_DOCKER"` should print three non-empty values.
   Empty means step 3 failed — re-run it without the `eval` to see why.
5. **Activate the CDK venv.** `infra/cdk/cdk.json` declares
   `"app": "python app.py"`, so the `python` on PATH must have `aws_cdk`
   importable. `infra/.venv` is the one to use — the same environment
   `scripts/test-infra.sh` runs the infra tests from:
   ```bash
   source infra/.venv/bin/activate
   ```
   If it does not exist yet: `python3 -m venv infra/.venv && infra/.venv/bin/pip
   install -r infra/requirements.txt`.

## Deploy

### First-time deploy (or any deploy where storage stack is unchanged)

Run `cdk` from `infra/cdk` — that is where `cdk.json` lives. From `infra/`
you get `--app is required either in command-line, in cdk.json or in
~/.cdk.json`.

```bash
cd infra/cdk
cdk deploy --all --require-approval never
```

This needs the `aws-cdk` CLI installed globally, not `npx cdk`: both
`dev-prep.sh` and `dev-postdeploy.sh` call `cdk` directly.

## Post-deploy

Same shell as the deploy — it re-checks the `dev-prep.sh` exports and fails
loudly if they are missing. The path is relative to the repo root, and the
Deploy step left you in `infra/cdk`:

```bash
cd ../..
./infra/scripts/dev-postdeploy.sh
```

What it does, in order:
- Reads the **CloudFront DNS** from the frontend stack's CFN output and
  finds CloudFront's `CloudFront-VPCOrigins-Service-SG` in the ALB's VPC.
- Re-deploys `ai-eval-studio-network` with
  `--context cloudfront_vpc_origin_sg_id=<sg>`, so the ALB security group
  admits CloudFront. Until this runs, `/api/*` and `/ws/*` return 504.
- Resolves a `PRIVATE_WITH_EGRESS` subnet in an AgentCore-supported
  Availability Zone.
- Writes `cloudfront_dns`, `cognito_domain`, `logout_redirect_url`,
  `cloudfront_vpc_origin_sg_id` and `agentcore_subnet_ids` into
  `infra/cdk/cdk.context.json`.
- Re-deploys `ai-eval-studio-auth` with `--context cloudfront_dns=<dns>`
  so the SPA Cognito client picks up the production callback
  (`/auth/callback`) and logout (`/`) URLs.
- Re-deploys `ai-eval-studio-storage` (so the upload-staging bucket's CORS
  allows the CloudFront origin) and `ai-eval-studio-frontend` (so the CSP
  allows the Cognito domain).
- Re-deploys `ai-eval-studio-agentcore` with the discovered subnet and reads
  its `RuntimeArn` output. `ai-eval-studio-iam` and `ai-eval-studio-compute`
  deploy in the same command, first: the runtime imports values from them
  that CDK exports only once the runtime exists, so on a fresh deploy
  agentcore alone would fail with `No export named ...`. Skipped, with a
  warning, when no subnet lands in a supported Availability Zone.
- Re-deploys `ai-eval-studio-compute` with `--context
  logout_redirect_url=https://<dns>/` and, when a runtime exists,
  `--context agentcore_runtime_arn=<arn>`, so the `/api/config` endpoint
  serves the right logout target to the SPA's `auth.ts` and the coordinator
  can relay turns to the runtime.
- Exits non-zero unless the AgentCore runtime (when there is one) requires
  MMDSv2, and unless `/api/config` answers HTTP 200 through CloudFront
  within about three minutes.
- Prints the `admin-create-user` command.

Then onboard the first user (Cognito self-signup is disabled by
design):

```bash
POOL_ID="$(aws cognito-idp list-user-pools --max-results 10 \
    --query "UserPools[?Name=='ai-eval-studio-users'].Id" --output text)"

aws cognito-idp admin-create-user \
    --user-pool-id "$POOL_ID" \
    --username you@example.com \
    --user-attributes Name=email,Value=you@example.com \
                      Name=email_verified,Value=true
```

Cognito emails the user a temp password; first sign-in forces a reset
via Cognito's hosted UI.

## Smoke test

```bash
CF_DNS="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-frontend \
    --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
    --output text)"

# 1. SPA shell loads at the root.
curl -I "https://$CF_DNS/"
# → HTTP/2 200
# → Content-Type: text/html
# → Strict-Transport-Security + Content-Security-Policy headers present
#   (set by the CloudFront ResponseHeadersPolicy on the default behavior).

# 2. /api/config is public — returns auth shape without a token.
curl -s "https://$CF_DNS/api/config" | jq
# → {"auth_required": true, "cognito": {"domain": "...", "client_id": "...",
#    "region": "...", "user_pool_id": "..."}, "logout_redirect": "https://<cf-dns>/"}
#   Nothing else. The AgentCore runtime ARN is never served here.

# 3. /api/* (other than /api/config and /api/health) returns 401 without
#    a Bearer token — the FastAPI middleware enforces JWT validation.
curl -I "https://$CF_DNS/api/projects"
# → HTTP/2 401

# 4. HTTP requests redirect to HTTPS.
curl -I "http://$CF_DNS/"
# → HTTP/1.1 301
# → Location: https://...

# 5. ALB has no public DNS — direct access fails.
ALB_DNS="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-network \
    --query "Stacks[0].Outputs[?OutputKey=='AlbDnsName'].OutputValue" \
    --output text)"
dig +short "$ALB_DNS" | head
# → (private IPs from the VPC; not resolvable from outside)

# 6. Coordinator task is RUNNING.
aws ecs list-tasks \
    --cluster "$(aws ecs list-clusters --query 'clusterArns[?contains(@, `ai-eval-studio`)]|[0]' --output text)" \
    --desired-status RUNNING
```

## Browser walkthrough

1. Open `https://<cloudfront-dns>/` in a browser. **No cert warning** —
   `*.cloudfront.net` is a publicly-trusted domain.
2. The SPA's `AuthGate` mounts, fetches `/api/config`, sees
   `auth_required: true`, and redirects to the Cognito hosted UI via
   `oidc-client-ts` (Authorization Code + PKCE).
3. Cognito hosted UI prompts for username + temp password.
4. First sign-in forces a password reset (`FORCE_CHANGE_PASSWORD` →
   `CONFIRMED`).
5. Cognito redirects to `https://<cloudfront-dns>/auth/callback?code=...`.
   The SPA's `AuthGate` finalizes the code exchange, stores tokens
   (access, ID and refresh, all in `sessionStorage`), and pushes the
   user back to `/`.
6. The SPA opens a WebSocket to `wss://<cloudfront-dns>/ws` and sends the
   access token as its first frame, `{"action": "auth", "token": "<jwt>"}`
   — never in the URL, which every proxy and access log records. The
   FastAPI route validates it against Cognito JWKS, replies `auth_ok`, and
   only then registers the connection in the in-process `ws_hub`.
7. `/api/*` requests carry `Authorization: Bearer <jwt>`; the middleware
   validates and populates `request_context.user_id` for downstream
   handlers.

## Troubleshooting

**CloudFront 502 or 504 on `/api/*` or `/ws/*`.** CloudFront can't reach
the ALB through the VPC origin. Common causes:
- ALB port-80 listener was removed or replaced (the network stack
  expects a single HTTP listener on port 80).
- **ALB security group has no ingress rule.** This is the most likely cause
  of a 504 on `/api/*` when the SPA loads fine and the target is healthy.

  ```bash
  ALB_SG="$(aws cloudformation describe-stacks \
      --stack-name ai-eval-studio-network \
      --query "Stacks[0].Outputs[?OutputKey=='AlbSecurityGroupId'].OutputValue" \
      --output text)"
  aws ec2 describe-security-group-rules \
      --filters "Name=group-id,Values=$ALB_SG" \
      --query "SecurityGroupRules[?IsEgress==\`false\`]"
  ```

  Expect **exactly one** rule: tcp/80 from CloudFront's
  `CloudFront-VPCOrigins-Service-SG`. Two other outcomes and what they mean:

  - **Empty.** The network stack was deployed without
    `cloudfront_vpc_origin_sg_id` in CDK context, so it created no rule (synth
    prints a warning saying exactly this). Fix by running
    `./infra/scripts/dev-postdeploy.sh`, which discovers the group and
    redeploys the network stack. Expected state after a first-ever
    `cdk deploy --all`, since the group does not exist until the frontend
    stack creates the VPC origin.
  - **A `0.0.0.0/0` rule.** The listener was deployed with `open=True`, which
    adds that rule. The API works, but the ALB then accepts traffic from
    anything that can route to it, not only from CloudFront.

  Do **not** expect `VpcOrigin.with_application_load_balancer` to supply this
  rule: the frontend template contains an `AWS::CloudFront::VpcOrigin` and no
  `AWS::EC2::SecurityGroupIngress` anywhere. CloudFront's VPC-origin ENIs live
  inside the VPC, so without the explicit rule their only way in would be a
  `0.0.0.0/0` rule, which the listener's `open=False` leaves out.
  `infra/tests/test_alb_cloudfront_ingress.py` pins the CloudFront-group rule;
  `infra/tests/test_alb_ingress.py` pins the absence of the `0.0.0.0/0` rule.
- Coordinator service has no running tasks. `aws ecs list-tasks ...`
  to confirm.

Fix: `cdk deploy ai-eval-studio-network ai-eval-studio-compute
ai-eval-studio-frontend` in that order, with the compute stack's two context
flags (see
[Compute-Deploy Context Flags](#compute-deploy-context-flags-required-every-time)),
or re-run `./infra/scripts/dev-postdeploy.sh`, which passes them itself.

**WebSocket connect fails with 1006 / connection refused.** Browser
devtools shows the WS upgrade attempt but the connection drops. Common
causes:
- The auth frame's token is missing or stale. The upgrade itself still
  succeeds, because the credential travels in the first frame rather
  than the URL, so this shows as an open followed by a close with code
  4401 rather than a failed upgrade. The SPA's `ws-client.ts` calls
  `getAccessToken()` immediately before opening; if `auth.ts` returned
  `""` (token expired and silent renew failed), every attempt is closed
  4401 and the client backs off between them. Force a re-login.
- CloudFront's `/ws/*` behavior was changed to a cache policy other
  than `CACHING_DISABLED` or an origin-request policy other than
  `ALL_VIEWER` — the `Sec-WebSocket-*` headers won't reach the ALB.
  Confirm in the CloudFront console that `/ws/*` uses the same
  `api_behavior` as `/api/*`.
- ALB target group health is unhealthy (CloudFront doesn't proxy WS
  to unhealthy targets). `aws elbv2 describe-target-health
  --target-group-arn <arn>` to confirm. The health check is HTTPS, so an
  unhealthy target with a running task usually means the container is not
  serving TLS: check the coordinator log group for `app.scripts.serve_tls`
  failing at start-up, before uvicorn's startup lines.

**`CannotPullContainerError`** in ECS task events. Backend image build
failed or push didn't complete. Re-run `cdk deploy ai-eval-studio-compute`
with `CDK_DOCKER=podman` and `DOCKER_HOST` set and the two context flags (see
[Compute-Deploy Context Flags](#compute-deploy-context-flags-required-every-time));
check `podman info` reaches a running machine.

**SPA bundle build fails with `Error: statfs ... no such file or
directory`.** podman machine on macOS doesn't share `/tmp` from the
host. CDK's default `--output /tmp/cdkout` breaks bundling. Synth without
`--output` (defaults to `infra/cdk/cdk.out/` under `/Users/`, which podman
shares). The dev workflow handles this automatically.

**Cognito callback rejection** (`redirect_mismatch`). The SPA's
callback URL on the app client doesn't match what Cognito sees. The
post-deploy script (`dev-postdeploy.sh`) should sync it via the
`cloudfront_dns` context var. If it didn't run, manually re-deploy:
```bash
cdk deploy ai-eval-studio-auth --context "cloudfront_dns=$CF_DNS" \
    --require-approval never
```
Or update directly:
```bash
SPA_CLIENT_ID="$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-auth \
    --query "Stacks[0].Outputs[?OutputKey=='SpaClientId'].OutputValue" \
    --output text)"
aws cognito-idp update-user-pool-client \
    --user-pool-id "$POOL_ID" \
    --client-id "$SPA_CLIENT_ID" \
    --callback-urls "https://$CF_DNS/auth/callback" \
    --logout-urls "https://$CF_DNS/"
```

**`cdk deploy` reports "no changes"** but you expected new code. CDK
fingerprints each asset from the CONTENTS of its build context. If only
`backend/tests/` changed (`backend/.dockerignore` excludes them), the backend
image hash won't change and CDK skips that asset — correctly, since excluded
files never reach the image. Same logic applies to the SPA bundle — changes
inside `frontend/` regenerate it. Touching a file changes its mtime, not its
contents, so it does not force a rebuild; editing a file the image ships
does.

**SPA shows old version after deploy.** CloudFront caches `index.html`
at the edge. The frontend stack wires `BucketDeployment(distribution=...,
distribution_paths=[...])` to invalidate the entry-point paths
automatically; if you bypassed CDK and uploaded directly, invalidate
manually:
```bash
DIST_ID="$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?contains(Aliases.Items[0] || DomainName, \`$CF_DNS\`)].Id" \
    --output text)"
aws cloudfront create-invalidation \
    --distribution-id "$DIST_ID" \
    --paths "/index.html" "/" "/manifest.json"
```

## Compute-Deploy Context Flags (Required Every Time)

**`cdk deploy ai-eval-studio-compute` needs two context values on every
deploy**, or the coordinator environment regresses. `dev-postdeploy.sh` saves
both to this checkout's `cdk.context.json`; from anywhere else, pass them as
`-c` flags.

### The Two Mandatory Flags

1. **`-c agentcore_runtime_arn=<AGENTCORE_RUNTIME_ARN>`**  
   Sourced from the AgentCore stack's `RuntimeArn` CloudFormation output.
   Without it, the coordinator container renders WITHOUT the
   `AGENTCORE_RUNTIME_ARN` env var and the relay path cannot invoke the
   runtime.

2. **`-c logout_redirect_url=<CLOUDFRONT_URL>/`**  
   The `/api/config` logout target the SPA reads.

### Key Facts

- **CDK does not persist either flag on its own.**  
  Context passed with `-c` applies to that one invocation. Without
  `agentcore_runtime_arn`, the coordinator renders with no
  `AGENTCORE_RUNTIME_ARN` at all. Without `logout_redirect_url`, the logout
  target falls back to `https://<cloudfront_dns>/` when `cloudfront_dns` is in
  context (or `CLOUDFRONT_DNS` is set in the environment), and to `/`
  otherwise.

  `dev-postdeploy.sh` passes both on its own compute leg, discovering them
  from CFN outputs (`DistributionDomainName` on the frontend stack,
  `RuntimeArn` on the agentcore stack). It also saves six keys into
  `infra/cdk/cdk.context.json` — `cloudfront_dns`, `cognito_domain`,
  `logout_redirect_url`, `cloudfront_vpc_origin_sg_id`,
  `agentcore_subnet_ids` and `agentcore_runtime_arn` — so later deploys in
  that checkout pick them up from the file. The compute stack reads context with
  `try_get_context`, which reads the file as well as `-c`, so after the script
  has run, a plain `cdk deploy ai-eval-studio-compute` from the same checkout
  keeps `AGENTCORE_RUNTIME_ARN`. A `-c` flag still overrides the file.

  Two limits, both by design:

  - **The file is per checkout.** `cdk.context.json` is gitignored, so another
    checkout, or a CI job, has none of these keys. From there, pass
    `-c agentcore_runtime_arn=...` and `-c logout_redirect_url=...` explicitly,
    as in the invocation below and in
    [Iteration after first deploy](#iteration-after-first-deploy).
  - **The saved ARN goes stale only when the runtime is replaced.** A new
    image bumps the runtime version and keeps its ARN. Replacing the runtime
    (a name change, or deleting and recreating it) gives it a new ARN, and a
    stale one is rendered into the coordinator on the next compute deploy,
    which then relays turns to a runtime that no longer exists. The script
    re-reads and re-saves the ARN on every run, and the documented flow runs it
    right after `cdk deploy --all`, so re-run `dev-postdeploy.sh` after
    replacing the runtime any other way. After a teardown, step 13 of
    [`teardown.md`](teardown.md) removes the whole file.

- **The AgentCore subnet is discovered, not committed.** AgentCore Runtime only
  runs in certain Availability Zones, and AWS identifies those by AZ-**ID**
  (`use1-az1`) while CDK selects subnets by AZ **name** — and the name-to-ID
  mapping differs per account. A subnet id is therefore account-specific and
  cannot be a committed default.

  `dev-postdeploy.sh` resolves the supported AZ-IDs to this account's AZ names,
  finds a `PRIVATE_WITH_EGRESS` subnet in one of them, persists it to
  `cdk.context.json`, and deploys the agentcore stack with it. Until that runs,
  the stack synthesizes with **no runtime** and prints a warning saying exactly
  that — which is why `cdk deploy --all` still works on a fresh account instead
  of failing at synth.

### Working Invocation Reference

```bash
# Environment exports (set once per shell session)
export CDK_DEFAULT_ACCOUNT=<ACCOUNT_ID>
export CDK_DEFAULT_REGION=us-east-1
export DEPLOY_ENV=dev
export COGNITO_DOMAIN_PREFIX=<prefix>
export CDK_DOCKER=podman
export DOCKER_HOST=<podman socket>

# Compute deploy with both context flags
cdk deploy ai-eval-studio-compute \
  -c agentcore_runtime_arn=<AGENTCORE_RUNTIME_ARN> \
  -c logout_redirect_url=https://<CLOUDFRONT_URL>/ \
  --require-approval never
```

**Placeholders:** Replace `<ACCOUNT_ID>`, `<AGENTCORE_RUNTIME_ARN>`,
`<CLOUDFRONT_URL>`, `<prefix>`, and `<podman socket>` with
your environment-specific values.

## Iteration after first deploy

For day-to-day work see
[`infra/README.md`](../../infra/README.md#iteration-workflow).
The deploy below updates the compute and iam stacks in place. The network
(with the ALB), auth, storage, observability, frontend and agentcore stacks
stay as they are, and their persistent resources (the ALB, user pool, buckets,
access-log bucket, table, file system, log groups and the customer-managed KMS
keys) are RETAIN'd besides. Frontend changes re-trigger a CloudFront
invalidation automatically. Backend-only, with both context flags from
[Compute-Deploy Context Flags](#compute-deploy-context-flags-required-every-time):

```bash
RUNTIME_ARN="$(aws cloudformation describe-stacks --stack-name ai-eval-studio-agentcore \
  --query "Stacks[0].Outputs[?OutputKey=='RuntimeArn'].OutputValue" --output text)"
CF_DNS="$(aws cloudformation describe-stacks --stack-name ai-eval-studio-frontend \
  --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" --output text)"
cdk deploy ai-eval-studio-compute ai-eval-studio-iam \
  -c agentcore_runtime_arn="$RUNTIME_ARN" \
  -c logout_redirect_url="https://$CF_DNS/" \
  --require-approval never
```

Do not `cdk destroy` the compute or iam stack to force a fresh start. The
agentcore stack imports values that both of them export, so CDK includes it in
the destroy (`Including depending stacks: ai-eval-studio-agentcore`) and deletes
it, with its runtime, first. That starts the up-to-8-hour wait for the runtime's
network interface described in [`teardown.md`](teardown.md), and the deploy above
does not bring the runtime back.

CloudFront DNS, ALB DNS, S3 data, Cognito users, log history — all
survive.

Frontend-only iteration (UI changes don't need a backend deploy):

```bash
cdk deploy ai-eval-studio-frontend
```

CDK's asset hashing means a no-op deploy if `frontend/` didn't change.
On real changes, CloudFront's `/index.html` cache is invalidated
automatically.
