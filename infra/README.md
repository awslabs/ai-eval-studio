# AI Eval Studio — Infrastructure (CDK)

AWS CDK (Python) for the cloud deployment. `cdk synth` produces
CloudFormation; `cdk deploy` pushes to AWS. Both need podman — the frontend
stack bundles the SPA in a container at synth time.

## Prerequisites

- Python 3.11+
- Node.js and the `aws-cdk` CLI (`npm install -g aws-cdk`) — the scripts invoke
  bare `cdk`, so `npx cdk` is not a substitute for them
- AWS CLI configured with credentials for the target account
- podman, running (`podman machine start`)
- **us-east-1 for the AgentCore runtime.** AgentCore is only available in some
  Availability Zones, identified by AZ-ID; `dev-postdeploy.sh` knows us-east-1's.
  Elsewhere it warns and skips that one stack — the rest deploys and the app
  works, without the relay path.

## Install

`cdk.json` declares `"app": "python app.py"`, so every `cdk` command needs a
`python` on PATH with `aws_cdk` importable. **`infra/.venv` is the one
environment for this** — `scripts/test-infra.sh` also runs the infra tests
from it. Do not create a second venv under `infra/cdk/`; a duplicate there
looks authoritative because it sits beside `cdk.json`, but nothing reads it.

```bash
cd infra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install -g aws-cdk    # if the cdk CLI is not already on PATH
```

## Synth

```bash
source infra/.venv/bin/activate    # cdk.json runs `python app.py`
cd infra/cdk                       # where cdk.json lives

# Required for cdk to know where to deploy. CDK_DEFAULT_ACCOUNT/REGION
# typically come from your AWS profile.
export CDK_DEFAULT_ACCOUNT=123456789012
export CDK_DEFAULT_REGION=us-east-1

# Required: synth fails without it. Globally unique and unguessable; for an
# existing deployment, use its deployed prefix (dev-prep.sh finds it for you).
export COGNITO_DOMAIN_PREFIX=ai-eval-studio-$RANDOM$RANDOM

# Container engine for `cdk deploy`'s from_asset image build.
export CDK_DOCKER=podman
export DOCKER_HOST="unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')"

# Tag every resource with Environment=<value>. Defaults to "dev".
export DEPLOY_ENV=prod

cdk synth
```

## Quickstart

First-time dev deploy from a fresh AWS account, top to bottom. **One shell
throughout** — `dev-prep.sh` only prints the exports, so a `cdk deploy` in a
different shell gets an unconfigured environment. Paths below are from the
**repo root**, not from `infra/`:

```bash
# 1. One-time prep: cdk bootstrap, pick the Cognito prefix (reuse the
#    deployed one, or generate a random one), verify podman.
#    Idempotent. Eval into the shell to capture the exports (they go to
#    stdout; the progress lines go to stderr).
eval "$(./infra/scripts/dev-prep.sh)"

# 2. Put aws_cdk on PATH for cdk.json's `python app.py`.
source infra/.venv/bin/activate

# 3. Deploy everything. ~15 min cold. Backend image builds via from_asset;
#    SPA builds via BucketDeployment bundling. Both shell out to podman.
#    ai-eval-studio-agentcore synthesizes with NO runtime until
#    dev-postdeploy.sh discovers a supported-AZ subnet — it warns and skips
#    rather than failing, so --all still succeeds.
#    Must run from infra/cdk — from infra/ cdk cannot find its app.
(cd infra/cdk && cdk deploy --all --require-approval never)

# 4. Wire the CloudFront DNS as Cognito's callback URL and redeploy auth.
./infra/scripts/dev-postdeploy.sh

# 5. Onboard the first user (self-signup is disabled by design).
POOL_ID="$(aws cognito-idp list-user-pools --max-results 10 \
    --query "UserPools[?Name=='ai-eval-studio-users'].Id" --output text)"
aws cognito-idp admin-create-user \
    --user-pool-id "$POOL_ID" \
    --username you@example.com \
    --user-attributes Name=email,Value=you@example.com \
                      Name=email_verified,Value=true

# 6. Visit the app.
echo "Open: https://$(aws cloudformation describe-stacks \
    --stack-name ai-eval-studio-frontend \
    --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
    --output text)"
```

The public entry point is CloudFront, running on the default
``*.cloudfront.net`` cert. No domain or ACM cert to register; the ALB
is private — no public IP, and its security group admits only CloudFront's
VPC-origin group on port 80. Migration to a
custom domain is a separate hardening step that adds an ACM cert in
us-east-1 and ``domain_names=`` on the distribution.

Full runbook with smoke tests + troubleshooting:
[`docs/runbooks/deploy.md`](../docs/runbooks/deploy.md).

## Stacks

| Stack | Tier | Removal Policy | What it does |
|-------|------|---------------|--------------|
| `ai-eval-studio-auth` | Persistent | RETAIN (UserPool) | Cognito User Pool (`ai-eval-studio-users`) and the SPA public client. |
| `ai-eval-studio-network` | Persistent | RETAIN (ALB) | VPC (2 AZs), public + private subnets, single NAT, gateway endpoints (S3, DynamoDB), interface endpoints (ECR), **internal ALB + HTTP listener + coordinator target group**. ALB is `internet_facing=False` in private subnets with no inbound peering/VPN/NAT path, the listener is created `open=False`, and the security group carries **exactly one ingress rule: tcp/80 from CloudFront's `CloudFront-VPCOrigins-Service-SG`**, declared explicitly in `network_stack.py`. That rule is NOT supplied by `VpcOrigin.with_application_load_balancer`, which adds no ingress rule at all. Its group id arrives via `cloudfront_vpc_origin_sg_id` CDK context, which `dev-postdeploy.sh` discovers, because AWS creates the group only once the frontend stack's VPC origin exists (i.e. after this stack deploys). Without that context there is no rule at all and CloudFront returns 504; synth warns. Listener is plain HTTP on port 80; auth is enforced by the FastAPI app (in-app JWT against Cognito JWKS), and viewer-facing TLS lives at the CloudFront edge. The coordinator target group is **HTTPS** to the container, which serves a self-signed certificate it generates at start-up (`app.scripts.serve_tls`); ALB does not validate target certificates. Also owns the **access-log bucket** (`ai-eval-studio-access-logs-{acct}-{region}`, SSE-S3, RETAIN, one-year expiry) and turns on the ALB's access logs into it. |
| `ai-eval-studio-storage` | Persistent | RETAIN (buckets, table, EFS, KMS keys) | S3 buckets (`ai-eval-studio-sessions-{acct}-{region}` for chat transcripts, `ai-eval-studio-uploads-staging-{acct}-{region}` for ≥50 MB browser uploads with a 24h lifecycle, `ai-eval-studio-snapshots-{acct}-{region}` for job working-tree snapshots + run output with a 1-day expiry), EFS file system for project working trees (mounted at `/mnt/projects`), DynamoDB single table `ai-eval-studio-metadata` with GSI1, on-demand billing, PITR, TTL on `ttl`. Three customer-managed KMS keys (data buckets, table, EFS), rotating yearly and RETAIN'd with the stores they encrypt. An EFS file-system policy refuses clients without TLS or outside a mount target, and a DynamoDB resource policy refuses requests without TLS. The data buckets write S3 server access logs to the network stack's access-log bucket. |
| `ai-eval-studio-iam` | Disposable | DESTROY | Coordinator task role (`ai-eval-studio-coordinator-task-role`), worker task + execution roles (`ai-eval-studio-worker-task-role`, `ai-eval-studio-worker-execution-role`) for per-job eval Fargate tasks, AgentCore runtime execution role (`ai-eval-studio-agentcore-runtime-role`), and the scoped-session role (`ai-eval-studio-scoped-session-role`, also called the token-vending (TVM) target role) assumed via STS for per-request narrowing. |
| `ai-eval-studio-observability` | Persistent | RETAIN (log group, KMS key) | Log group (`/ai-eval-studio/coordinator`) with 30-day retention, encrypted with a customer-managed KMS key that only CloudWatch Logs may use, for this app's log groups. ECS Exec sessions are recorded here too. No dashboard or alarms yet. |
| `ai-eval-studio-compute` | Disposable | DESTROY (services) | Coordinator Fargate task def + service (`desiredCount=1`, `min_healthy_percent=100, max_healthy_percent=200` — no deploy outage, at the cost of two coordinator tasks overlapping during a deploy) plus the eval worker task def (family `ai-eval-studio-worker`, no service — the coordinator launches one Fargate task per eval job via `RunTask` and it exits when the job completes). Both use the container image built from `backend/Dockerfile` and pushed to CDK's bootstrap asset ECR on deploy (no named repo). |
| `ai-eval-studio-agentcore` | Disposable | DESTROY (runtime) | AWS::BedrockAgentCore::Runtime built from the ARM64 `backend/Dockerfile.agentcore` (content-hash asset — rebuilds and bumps the runtime version when the image content changes), VPC mode on a supported-AZ subnet (discovered by `dev-postdeploy.sh`), Cognito JWT authorizer, declarative EFS mount at `/mnt/projects`. Managed session storage is wiped on version bump, but durable state lives on S3/EFS so this is acceptable. |
| `ai-eval-studio-frontend` | Persistent | RETAIN (bucket) | S3 bucket `ai-eval-studio-frontend-{acct}-{region}` + CloudFront distribution. **Public entry point.** S3 origin serves the SPA; VPC-origin pipes `/api/*` and `/ws/*` to the internal ALB (HTTP). SPA built and uploaded via `s3deploy.BucketDeployment` (Node 20 build container) on every deploy. Strict CSP attached to the default behavior. CloudFront standard logging writes every viewer request to the network stack's access-log bucket. |
| `ai-eval-studio-appregistry` | Auto-managed | — | AppRegistry `Application` + 8 stack associations. Lights up Console Home → myApplications. |

## Iteration workflow

The persistent layer (network, auth, storage, observability) is wrapped
in `RemovalPolicy.RETAIN`, and day-to-day changes only redeploy compute and
iam on top of it. Fast iterate — from `infra/cdk`, venv active:

```bash
cd infra/cdk

# Fast iterate (~5 min). Updates the Fargate task defs, the service and the
# IAM task roles in place. Without the runtime ARN in context, the coordinator
# loses AGENTCORE_RUNTIME_ARN and cannot relay agent turns. dev-postdeploy.sh
# saves it to this checkout's cdk.context.json, so a plain deploy from here
# keeps it; from any other checkout, or after replacing the runtime without
# re-running that script, pass it explicitly as below.
RUNTIME_ARN="$(aws cloudformation describe-stacks --stack-name ai-eval-studio-agentcore \
  --query "Stacks[0].Outputs[?OutputKey=='RuntimeArn'].OutputValue" --output text)"
CF_DNS="$(aws cloudformation describe-stacks --stack-name ai-eval-studio-frontend \
  --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" --output text)"
cdk deploy ai-eval-studio-compute ai-eval-studio-iam \
  -c agentcore_runtime_arn="$RUNTIME_ARN" \
  -c logout_redirect_url="https://$CF_DNS/" \
  --require-approval never

# After a backend change, also redeploy agentcore so the change reaches
# the microVM (rebuilds the ARM64 image via podman and bumps the runtime
# version). The subnet comes from cdk.context.json, which dev-postdeploy.sh
# writes — a fresh clone must run that first or the runtime is omitted.
cdk deploy ai-eval-studio-agentcore --exclusively --require-approval never
```

The canonical `./infra/scripts/dev-postdeploy.sh` already includes the agentcore
redeploy leg automatically after `cdk deploy --all`, so this manual step is
only needed for the fast-iterate workflow above.

Do not `cdk destroy` compute or iam to force a fresh start. The agentcore stack
imports values both of them export, so CDK includes it in the destroy and
deletes it, with its runtime, first. That starts the up-to-8-hour wait for the
runtime's network interface, and the deploy above does not bring it back.

Cognito users, S3 data (including the SPA bucket), the DynamoDB table,
the ALB DNS name, the CloudFront distribution, log history, and ECR
images all survive.

For a full clean teardown (rare — usually only before deleting the AWS
account itself), see [`docs/runbooks/teardown.md`](../docs/runbooks/teardown.md).
Allow up to 8 hours: the AgentCore runtime's network interface can outlive
the runtime by that long, and the network stack cannot be deleted until it is
gone.

## Required env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `CDK_DEFAULT_ACCOUNT` | — | Required. AWS account ID. |
| `CDK_DEFAULT_REGION` | `us-east-1` | Region. |
| `DEPLOY_ENV` | `dev` | Value of the app-wide `Environment` tag (cost allocation, console grouping). Set to `staging` / `prod` per environment. |
| `COGNITO_DOMAIN_PREFIX` | — | Required; synth fails without it. Cognito hosted-domain prefix, globally unique. `dev-prep.sh` reuses the deployed prefix, or generates an unguessable one for a new deployment. A different prefix on an existing deployment replaces its domain and moves the login URL. |
| `cloudfront_dns` (CDK context) | placeholder | CloudFront DNS for the SPA Cognito client's callback / logout URLs. `dev-postdeploy.sh` re-runs `cdk deploy ai-eval-studio-auth -c cloudfront_dns=<dns>` after the first deploy. |
| `logout_redirect_url` (CDK context) | `/` | Logout redirect for the SPA. Set to `https://<cloudfront-dns>/` post-deploy via the same script. |
| `CDK_DOCKER` | `docker` | Container engine for `from_asset` builds. Set to `podman` for the dev workflow; `dev-prep.sh` exports it for you. |
| `DOCKER_HOST` | (engine-default socket) | Socket path. With CDK_DOCKER=podman, point at the podman machine's socket. `dev-prep.sh` derives this from `podman machine inspect`. |

## Tagging + AppRegistry

`Tags.of(app).add("Project", "ai-eval-studio")` and
`Tags.of(app).add("Environment", DEPLOY_ENV)` cascade to every taggable
resource in every stack. Verify in the synth output:

```bash
cdk synth    # writes the templates to cdk.out/
grep -c '"Key": "Project"' cdk.out/*.template.json   # one count per stack template
```

The `ai-eval-studio` AppRegistry application + 8 stack associations
auto-apply the `awsApplication` tag and light up:
- Console Home → **myApplications** dashboard
- AWS Resource Groups (auto-generated `ai-eval-studio` group)
- Service Catalog → AppRegistry → Applications
- Systems Manager → Application Manager
- The **All Applications** picker (top-right of any service console)

## Tenant boundary in IAM

Which roles hold the cross-tenant S3-sessions + DynamoDB data plane is the
tenant boundary in `iam_stack.py`, and one flag decides it:
`_attach_shared_policies(minted_data_plane_only=...)`.

- **Control-plane roles** (coordinator, AgentCore runtime) hold the broad
  grant and enforce ownership in application code at `require_access`. They
  have to: resolving `require_access` and the email sharing lookup reads
  other users' records by definition.
- **The worker task role** does not. Its container executes agent-authored
  harness code by design, so an ambient grant there is reachable by that
  code — botocore falls back to the ECS credential endpoint the moment
  `AWS_CONFIG_FILE` is dropped. The worker reaches S3 and DynamoDB only
  through the scoped-session credential it self-mints on boot, whose
  statements are `${aws:PrincipalTag/*}`-templated.

A condition cannot substitute for that split: a bare task-role session
carries no principal tags, so a `${aws:PrincipalTag/uid}` template on a task
role resolves to nothing and denies outright.
`infra/tests/test_worker_task_def.py` asserts both halves — no ambient
S3/DynamoDB on the worker, and the control-plane roles keep theirs.

## Outstanding TODOs

- **Observability:** add a CloudWatch dashboard and alarms.
