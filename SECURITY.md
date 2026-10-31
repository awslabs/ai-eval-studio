# Security

## Reporting a vulnerability

If you discover a potential security issue in this project, notify AWS/Amazon Security
through the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or by email to [aws-security@amazon.com](mailto:aws-security@amazon.com).

Please do **not** open a public issue for a security problem.

Only the latest version on the default branch receives security fixes.

The limitations listed in the README's [Security scope](README.md#security-scope) are
known and stated deliberately, so they are not vulnerabilities to report. A gap not
listed there is in scope.

## This is a prototype

AI Eval Studio is a prototype for demonstration and evaluation, not a production
service. The default cloud deployment is a development deployment: one trust domain,
CloudFront's default `*.cloudfront.net` hostname, and an AWS account you are prepared to
let it run LLM-written code in. Before using it for real data or real users, do your own
independent assessment against your security and quality standards, and work through
the steps below.

## What the default deployment already does

- **Encryption in transit** on every hop but one: browser to CloudFront (HTTPS only,
  HSTS), CloudFront to the frontend bucket, the internal ALB to the coordinator (HTTPS),
  and every task to S3, DynamoDB and EFS (TLS, refused otherwise). The exception is
  CloudFront to the ALB, which stays inside AWS's managed VPC-origin path; step 2 below
  encrypts it.
- **Encryption at rest with customer-managed KMS keys** for the data buckets, the
  DynamoDB table and its backups, the EFS file system, and the app's log groups. Each key
  rotates yearly, and IAM grants name the keys
  ([`storage_stack.py`](infra/cdk/storage_stack.py),
  [`iam_stack.py`](infra/cdk/iam_stack.py)).
- **Access logs** for CloudFront, the ALB and the data buckets, kept for a year in one
  bucket, plus a record of every ECS Exec session
  ([`network_stack.py`](infra/cdk/network_stack.py)).

## Path to production

1. **Treat one deployment as one trust domain.** Onboard only users who may see each
   other's data, as [Security scope](README.md#security-scope) explains. For groups that
   must not see each other's data, run separate deployments in separate AWS accounts.
2. **Use a custom domain.** CloudFront connects to an HTTPS origin only when the
   origin's certificate is publicly trusted and matches its hostname, so the
   CloudFront-to-ALB hop can only be encrypted with a domain of your own. Issue ACM
   certificates for CloudFront and the ALB, add an HTTPS listener to the ALB
   ([`network_stack.py`](infra/cdk/network_stack.py)), and set the VPC origin to
   `HTTPS_ONLY` ([`frontend_stack.py`](infra/cdk/frontend_stack.py)). A custom
   certificate also lets you set the viewer minimum to TLS 1.2; with the default
   CloudFront certificate it is fixed at TLSv1.
3. **Review who can use and administer the keys.** The keys' policies delegate to IAM in
   the deploying account. Restrict key administration to the people who should hold it,
   and decide who may disable a key, which makes its data unreadable at once.
4. **Confirm account-level logging.** The access logs above record requests, not who
   called which AWS API. Enable CloudTrail with data events for the S3 data buckets and
   the DynamoDB table, which record which IAM principal, including which minted session
   and its tags, read or wrote which object or item. Enable VPC flow logs for this VPC.
5. **Harden sign-in.** Require MFA in the Cognito user pool and raise the minimum
   password length; the default is 8 characters with no MFA
   ([`auth_stack.py`](infra/cdk/auth_stack.py)).
6. **Put AWS WAF in front of CloudFront** with the AWS managed rule groups and a
   rate-based rule. Every chat turn spends Bedrock tokens, so rate limiting also caps
   cost.
7. **Limit Bedrock spend and scope.** Bedrock grants access to its serverless models
   automatically, so restrict the models this deployment may call with IAM policies or
   service control policies, and set an AWS Budgets alert on Bedrock.
8. **Remove developer conveniences.** Drop `http://localhost:3100` from the
   uploads-staging bucket's CORS rule (`_staging_cors_origins` in
   [`storage_stack.py`](infra/cdk/storage_stack.py)), and turn off ECS Exec on the
   coordinator service (`enable_execute_command` in
   [`compute_stack.py`](infra/cdk/compute_stack.py)) unless operators need it.
9. **Review log retention** against your own policy: the coordinator log group keeps one
   month ([`observability_stack.py`](infra/cdk/observability_stack.py)), the sign-in
   trigger's two weeks ([`auth_stack.py`](infra/cdk/auth_stack.py)), and the access logs
   a year ([`network_stack.py`](infra/cdk/network_stack.py)).
