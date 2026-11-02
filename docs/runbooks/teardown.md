# AI Eval Studio — full teardown

For the rare full-clean case (sandbox tear-down before destroying the AWS
account, or a clean-slate restart). Day-to-day iteration redeploys the
compute and iam stacks in place and destroys nothing — see
[`infra/README.md`](../../infra/README.md#iteration-workflow).

`cdk destroy --all` alone won't fully clean up because every persistent-tier
resource carries `RemovalPolicy.RETAIN` (S3 buckets, DynamoDB table, EFS
file system, ALB, Cognito User Pool, log groups, the access-log bucket, and the
customer-managed KMS keys that encrypt them). The order below is what
actually removes everything, including leftovers that exist only in accounts
where an older version of the stack was deployed (steps 5, 6 and 12).

**Plan for up to 8 hours.** Deleting the AgentCore runtime leaves its network
interface in the VPC for up to 8 hours, and the runtime's security group and the
private subnet cannot be deleted until it is gone (step 8). Step 1 deletes the
runtime first so that clock starts as early as possible.

**Run this runbook from the repo root.** Its one `cdk` command (step 8) has to
run from `infra/cdk` (where `cdk.json` is) with `infra/.venv` activated —
`cdk.json` declares `"app": "python app.py"`, so a `python` with `aws_cdk`
importable must be on PATH, and from `infra/` you get `--app is required
either in command-line, in cdk.json or in ~/.cdk.json`. That block does the
`cd` in a subshell, so you stay at the repo root. `aws` commands run from
anywhere.

```bash
source infra/.venv/bin/activate
```

Both blocks below use `${ACCOUNT}` and `${REGION}`. Set them first — nothing
else in this runbook does:

```bash
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGION="${AWS_REGION:-us-east-1}"
```

## Pre-flight

1. Confirm no eval/optimization jobs are in flight. Agent turns and the
   optimization loop run inside the coordinator, but each eval job runs
   as its own Fargate worker task — check both via the app (Projects list
   shows nothing actively running) and directly:
   ```bash
   aws ecs list-tasks \
     --cluster "$(aws ecs list-clusters --query 'clusterArns[?contains(@, `ai-eval-studio`)]|[0]' --output text)" \
     --family ai-eval-studio-worker --desired-status RUNNING
   ```
   Should return no task ARNs. Destroying the compute stack kills the
   coordinator service, but a stray worker task holds its ENI until it
   stops — stop it first if one is running.
2. Snapshot DynamoDB if you want to keep the data:
   ```bash
   aws dynamodb create-backup \
     --table-name ai-eval-studio-metadata \
     --backup-name pre-teardown-$(date +%Y%m%d)
   ```
3. Snapshot the data you want to keep:
   ```bash
   # Chat transcripts (S3).
   aws s3 sync s3://ai-eval-studio-sessions-${ACCOUNT}-${REGION}/ ./backup/sessions/
   ```
   Project working trees live on the EFS file system, not S3. To keep them,
   take an AWS Backup recovery point of the EFS file system (or mount it and
   copy off) before teardown. The `ai-eval-studio-uploads-staging` bucket is
   24h-ephemeral — nothing durable lives there, so skip it.

## Teardown order

1. **Delete the AgentCore runtime.** Do this before anything else, for two
   reasons. Its stack cannot be deleted once the Cognito user pool is gone:
   CloudFormation's delete fails with `Failed to fetch discovery document`
   (HTTP 400) from the pool's URL. And deleting it starts the up-to-8-hour wait
   for its network interface (step 8). The runtime id is the agentcore stack's
   `RuntimeId` output:
   ```bash
   RUNTIME_ID="$(aws cloudformation describe-stacks \
     --stack-name ai-eval-studio-agentcore \
     --query "Stacks[0].Outputs[?OutputKey=='RuntimeId'].OutputValue" --output text)"
   aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id "$RUNTIME_ID"
   ```
2. **Empty + delete S3 buckets, delete the EFS file system.** `cdk destroy`
   won't touch a non-empty bucket; `aws s3 rb --force` empties it first.
   ```bash
   aws s3 rb s3://ai-eval-studio-sessions-${ACCOUNT}-${REGION} --force
   aws s3 rb s3://ai-eval-studio-uploads-staging-${ACCOUNT}-${REGION} --force
   aws s3 rb s3://ai-eval-studio-snapshots-${ACCOUNT}-${REGION} --force
   aws s3 rb s3://ai-eval-studio-frontend-${ACCOUNT}-${REGION} --force
   ```
   (Snapshots are 1-day-ephemeral job artifacts and the frontend bucket
   is a rebuildable SPA bundle — nothing to back up in either. Both are
   RETAIN'd, so skipping them leaves orphaned buckets behind.)
   Project working trees are on EFS (RETAIN). Find it by the app-wide
   `Project` tag (the file system has no explicit Name), delete its mount
   targets, then the file system:
   ```bash
   FS_ID=$(aws efs describe-file-systems \
     --query "FileSystems[?Tags[?Key=='Project'&&Value=='ai-eval-studio']].FileSystemId" \
     --output text)
   for mt in $(aws efs describe-mount-targets --file-system-id "$FS_ID" \
     --query 'MountTargets[].MountTargetId' --output text); do
     aws efs delete-mount-target --mount-target-id "$mt"
   done
   # wait for mount targets to drain, then:
   aws efs delete-file-system --file-system-id "$FS_ID"
   ```
3. **Delete the DynamoDB table.**
   ```bash
   aws dynamodb delete-table --table-name ai-eval-studio-metadata
   ```
4. **Delete the Cognito User Pool.** Logs everyone out. Only after step 1.
   The pool cannot be deleted while it still has its hosted-UI domain
   (`It has a domain configured that should be deleted first`), so delete the
   domain first:
   ```bash
   POOL_ID="$(aws cognito-idp list-user-pools --max-results 60 \
     --query "UserPools[?Name=='ai-eval-studio-users'].Id" --output text)"
   POOL_DOMAIN="$(aws cognito-idp describe-user-pool --user-pool-id "$POOL_ID" \
     --query 'UserPool.Domain' --output text)"
   aws cognito-idp delete-user-pool-domain --user-pool-id "$POOL_ID" --domain "$POOL_DOMAIN"
   aws cognito-idp delete-user-pool --user-pool-id "$POOL_ID"
   ```
5. **Delete the `ai-eval-studio-app` ECR repo, if the account has one.** The
   stack does not create it — the backend image is published through CDK's
   bootstrap asset repository — but older versions of the stack created it
   with RETAIN, so it survives in accounts that deployed one. If
   `aws ecr describe-repositories --repository-names ai-eval-studio-app`
   reports `RepositoryNotFoundException`, skip this step.
   ```bash
   aws ecr delete-repository --repository-name ai-eval-studio-app --force
   ```
6. **Delete the log group.** `/ai-eval-studio/coordinator` is CDK-managed
   (RETAIN); worker task logs and ECS Exec sessions also stream into it.
   ```bash
   aws logs delete-log-group --log-group-name /ai-eval-studio/coordinator
   ```
   *Optional:* AWS creates more log groups that no stack owns, so they survive
   the teardown too: the Lambda functions behind CDK custom resources, ECS
   Container Insights, and every AgentCore runtime this account has run. None
   holds anything the app needs. List them, then delete the ones you recognise
   with `aws logs delete-log-group --log-group-name <name>`:
   ```bash
   aws logs describe-log-groups --query "logGroups[?contains(logGroupName, 'ai-eval-studio') \
     || contains(logGroupName, 'ai_eval_studio') || contains(logGroupName, 'aievalstudio')].logGroupName" \
     --output text | tr '\t' '\n'
   ```
7. **Delete the ALB.** It is RETAIN'd, so `cdk destroy` would leave it
   behind. It has no fixed name, so find it by the app-wide `Project` tag:
   ```bash
   ALB_ARN="$(aws resourcegroupstaggingapi get-resources \
     --tag-filters Key=Project,Values=ai-eval-studio \
     --resource-type-filters elasticloadbalancing:loadbalancer \
     --query 'ResourceTagMappingList[].ResourceARN' --output text)"
   aws elbv2 delete-load-balancer --load-balancer-arn "$ALB_ARN"
   ```
8. **Destroy stacks.** Now that the RETAINed resources are gone, CDK can
   clean up the rest. `cdk destroy` synthesizes the app first, and the auth
   stack refuses to synthesize without `COGNITO_DOMAIN_PREFIX`; a destroy
   creates nothing, so outside a `dev-prep.sh` shell any value will do.
   ```bash
   (cd infra/cdk && COGNITO_DOMAIN_PREFIX="${COGNITO_DOMAIN_PREFIX:-teardown}" cdk destroy --all)
   ```
   **Expect this to fail the first time, and rerun it later.** AgentCore's
   network interfaces are shared across agents that use the same subnet and
   security group, and AWS documents that one can persist for up to 8 hours
   after its agent is deleted
   ([AgentCore VPC configuration](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html)).
   Until it goes, the agentcore stack's delete fails on the runtime security
   group (`has a dependent object`) and the network stack's on the private
   subnet (`has dependencies`). Nothing is wrong: wait, then rerun the command.
   Detaching the interface by hand does not work — AWS refuses with
   `not allowed to manage 'ela-attach' attachments` — because AgentCore owns it.

   A redeploy under the same stack names has to wait for the network stack to
   finish deleting.

   *Optional, to finish the stacks sooner:* retain what the interface holds on
   to, so each stack's delete stops waiting on it, then delete those resources
   by hand once the interface is gone. CloudFormation accepts
   `--retain-resources` only on a stack whose delete has already failed, so
   run these after the first failed attempt above.

   The agentcore stack waits on the runtime security group. Retain it, then
   rerun this step; the compute, iam, auth, storage and observability stacks
   then finish at once:
   ```bash
   aws cloudformation delete-stack --stack-name ai-eval-studio-agentcore \
     --retain-resources RuntimeSg1BC2853D
   ```
   The network stack waits on every private subnet that holds one of the
   runtime's interfaces, and on the VPC around them. Which subnets those are
   depends on the availability zones AgentCore placed the interfaces in, so look
   them up rather than assuming one, then retain them and the VPC.

   Every lookup below is filtered on an ID. Do not shorten one to "take the
   first result": `describe-stack-resources --physical-resource-id`, for
   instance, returns every resource of the stack that owns the ID, not that
   one resource, and retaining the wrong one leaves the real subnet blocking.
   The block refuses to retain anything unless each lookup found exactly what it
   should. It works pasted into bash or zsh.
   ```bash
   # The VPC, from the network stack itself.
   VPC_LOGICAL="$(aws cloudformation describe-stack-resources --stack-name ai-eval-studio-network \
     --query "StackResources[?ResourceType=='AWS::EC2::VPC'].LogicalResourceId" --output text)"
   VPC_ID="$(aws cloudformation describe-stack-resources --stack-name ai-eval-studio-network \
     --query "StackResources[?ResourceType=='AWS::EC2::VPC'].PhysicalResourceId" --output text)"
   # The retained runtime group keeps its CloudFormation logical-id tag. Filter
   # on the VPC too: an earlier teardown can leave a group with the same tag.
   RUNTIME_SG="$(aws ec2 describe-security-groups \
     --filters Name=tag:aws:cloudformation:logical-id,Values=RuntimeSg1BC2853D \
               Name=vpc-id,Values="$VPC_ID" \
     --query 'SecurityGroups[].GroupId' --output text)"
   # Every subnet holding one of the group's interfaces; each one blocks the delete.
   ENI_SUBNETS="$(aws ec2 describe-network-interfaces \
     --filters Name=group-id,Values="$RUNTIME_SG" \
     --query 'NetworkInterfaces[].SubnetId' --output text | tr '\t' '\n' | sort -u)"
   SUBNET_LOGICALS=""
   for s in $(printf '%s\n' "$ENI_SUBNETS"); do
     SUBNET_LOGICALS="$SUBNET_LOGICALS $(aws cloudformation describe-stack-resources \
       --stack-name ai-eval-studio-network \
       --query "StackResources[?PhysicalResourceId=='$s'].LogicalResourceId" --output text)"
   done

   words() { printf '%s' "$1" | wc -w | tr -d ' '; }
   if [ "$(words "$VPC_LOGICAL")" = 1 ] && [ "$(words "$VPC_ID")" = 1 ] \
      && [ "$(words "$RUNTIME_SG")" = 1 ] && [ "$RUNTIME_SG" != None ] \
      && [ "$(words "$ENI_SUBNETS")" -ge 1 ] \
      && [ "$(words "$SUBNET_LOGICALS")" = "$(words "$ENI_SUBNETS")" ]; then
     aws cloudformation delete-stack --stack-name ai-eval-studio-network \
       --retain-resources $(printf '%s\n' "$SUBNET_LOGICALS") "$VPC_LOGICAL"
   else
     echo "Lookup failed, not retaining anything:" >&2
     echo "  VPC=$VPC_LOGICAL ($VPC_ID) runtime SG=$RUNTIME_SG" >&2
     echo "  interface subnets=[$ENI_SUBNETS] logical ids=[$SUBNET_LOGICALS]" >&2
   fi
   ```
   Once `describe-network-interfaces` for the runtime security group returns
   nothing, delete the retained resources in this order. The VPC also holds a
   security group named `CloudFront-VPCOrigins-Service-SG`, which CloudFront
   creates for its VPC origin and no stack owns; `delete-vpc` fails with
   `DependencyViolation` until it is gone:
   ```bash
   aws ec2 delete-security-group --group-id "$RUNTIME_SG"
   for s in $(printf '%s\n' "$ENI_SUBNETS"); do aws ec2 delete-subnet --subnet-id "$s"; done
   CF_SG="$(aws ec2 describe-security-groups \
     --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values=CloudFront-VPCOrigins-Service-SG \
     --query 'SecurityGroups[].GroupId' --output text)"
   [ -n "$CF_SG" ] && aws ec2 delete-security-group --group-id "$CF_SG"
   aws ec2 delete-vpc --vpc-id "$VPC_ID"
   ```
   If you let the network stack's delete run to completion instead, it may hit
   the same CloudFront security group. Delete it as above and rerun.
9. **Empty + delete the access-log bucket.** It is RETAIN'd, and only now has
   nothing left writing to it: the ALB, the CloudFront distribution and the
   data buckets all logged into it until they were gone. Back it up first if
   you need the request history.
   ```bash
   aws s3 rb s3://ai-eval-studio-access-logs-${ACCOUNT}-${REGION} --force
   ```
10. **Schedule the KMS keys for deletion.** The buckets, table, file system and
    log groups are encrypted with customer-managed keys, retained with them
    because deleting a key leaves anything it encrypted unreadable. Do this
    last, once nothing they encrypt is left. A key cannot be deleted at once:
    KMS deletes it after a waiting period of 7 to 30 days, and
    `aws kms cancel-key-deletion` undoes it until then. Keys an earlier
    teardown scheduled are still tagged during their waiting period, and
    scheduling one again fails with `KMSInvalidStateException`, so skip them:
    ```bash
    for key_arn in $(aws resourcegroupstaggingapi get-resources \
      --tag-filters Key=Project,Values=ai-eval-studio \
      --resource-type-filters kms:key \
      --query 'ResourceTagMappingList[].ResourceARN' --output text); do
      state="$(aws kms describe-key --key-id "$key_arn" --query KeyMetadata.KeyState --output text)"
      if [ "$state" = PendingDeletion ]; then
        echo "already pending deletion: $key_arn"
      else
        aws kms schedule-key-deletion --key-id "$key_arn" --pending-window-in-days 7
      fi
    done
    ```
11. **Delete the task-definition revisions.** ECS keeps every revision of a
    task definition after its stack is gone, deregistered ones included, and
    each deploy adds one, so an account that deployed often holds hundreds.
    Collect them by tag, and by family too, because revisions from older
    deploys may carry no tag. The worker's family is `ai-eval-studio-worker`;
    the coordinator's is named after its stack, so it starts with
    `aievalstudiocompute`. A revision must be inactive before it can be
    deleted, and `delete-task-definitions` takes at most 10 at a time. The AWS
    CLI pages through each listing itself.
    ```bash
    # Deregister the active revisions.
    for prefix in ai-eval-studio-worker aievalstudiocompute; do
      for arn in $(aws ecs list-task-definitions --family-prefix "$prefix" \
        --status ACTIVE --query 'taskDefinitionArns[]' --output text); do
        aws ecs deregister-task-definition --task-definition "$arn" > /dev/null
      done
    done

    # Collect every revision, tagged or not, then delete them 10 at a time.
    # The list goes to a temporary file, not the repo root this runbook runs from.
    TASK_DEFS="$(mktemp)"
    {
      aws resourcegroupstaggingapi get-resources \
        --tag-filters Key=Project,Values=ai-eval-studio \
        --resource-type-filters ecs:task-definition \
        --query 'ResourceTagMappingList[].ResourceARN' --output text
      for prefix in ai-eval-studio-worker aievalstudiocompute; do
        aws ecs list-task-definitions --family-prefix "$prefix" --status INACTIVE \
          --query 'taskDefinitionArns[]' --output text
      done
    } | tr '\t' '\n' | grep -v '^$' | sort -u > "$TASK_DEFS"
    xargs -n 10 aws ecs delete-task-definitions --task-definitions < "$TASK_DEFS"
    rm -f "$TASK_DEFS"
    ```
12. **Delete leftovers from older versions, if the account has them.** Like
    step 5, these exist only where an older stack, or a failed earlier
    teardown, left them:
    - An empty `ai-eval-studio-projects-${ACCOUNT}-${REGION}` bucket that no
      stack owns. Delete it with `aws s3 rb` as in step 2.
    - A runtime security group, named after `RuntimeSg` in the agentcore stack,
      orphaned by an earlier delete that retained it. Delete it with
      `aws ec2 delete-security-group` once no network interface uses it.

13. **Delete `infra/cdk/cdk.context.json` in every checkout that deployed.**
    `dev-postdeploy.sh` saved the deployment's discovered values there —
    `agentcore_runtime_arn`, `agentcore_subnet_ids`, `cloudfront_dns`,
    `cloudfront_vpc_origin_sg_id`, `cognito_domain` and `logout_redirect_url` —
    and the file is gitignored, so nothing else removes it. Every value names a
    resource this teardown deleted, and a redeploy from the same checkout would
    reuse them: a coordinator relaying turns to a deleted runtime, an AgentCore
    runtime on a subnet that no longer exists, an ALB rule for a security group
    that is gone, callback URLs for the old distribution.
    ```bash
    rm -f infra/cdk/cdk.context.json
    ```

Every step has been run against a full deployment except the optional step 8
block that retains the network stack's subnets and VPC. When you use it, check
its output rather than assuming it worked.

## Verify clean

```bash
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=Project,Values=ai-eval-studio
```

Should return `ResourceTagMappingList: []`, apart from the KMS keys from
step 10 until their waiting period ends, and, if you retained resources in
step 8, the VPC tagged `Project=ai-eval-studio` until you delete it there.

That listing lags: it can still show resources that were deleted minutes or
hours ago. Ask each service directly before concluding anything is left. Each
command should print nothing, except `describe-vpcs` while a VPC retained in
step 8 is still waiting to be deleted:

```bash
aws cloudformation list-stacks --query "StackSummaries[?starts_with(StackName, 'ai-eval-studio') \
  && StackStatus!='DELETE_COMPLETE'].[StackName,StackStatus]" --output text
aws s3api list-buckets --query "Buckets[?starts_with(Name, 'ai-eval-studio')].Name" --output text
aws dynamodb list-tables --query "TableNames[?starts_with(@, 'ai-eval-studio')]" --output text
aws efs describe-file-systems \
  --query "FileSystems[?Tags[?Key=='Project'&&Value=='ai-eval-studio']].FileSystemId" --output text
aws cognito-idp list-user-pools --max-results 60 \
  --query "UserPools[?Name=='ai-eval-studio-users'].Id" --output text
aws bedrock-agentcore-control list-agent-runtimes \
  --query "agentRuntimes[?starts_with(agentRuntimeName, 'ai_eval_studio')].agentRuntimeId" --output text
aws ec2 describe-vpcs --filters Name=tag:Project,Values=ai-eval-studio \
  --query 'Vpcs[].VpcId' --output text
aws logs describe-log-groups --log-group-name-prefix /ai-eval-studio \
  --query 'logGroups[].logGroupName' --output text
```
