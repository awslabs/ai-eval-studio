# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards on the post-deploy AgentCore leg.

The leg deploys every stack its new imports need. On the first
``cdk deploy --all`` there is no ``agentcore_subnet_ids`` context, so the
agentcore stack has no runtime and imports almost nothing. CDK creates a
cross-stack export only once something references it, so the stacks the
runtime will import from go up without those exports. ``dev-postdeploy.sh``
then adds the runtime. If that leg deploys agentcore alone, under
``--exclusively``, the import fails with ``No export named ...`` and a fresh
deploy needs a second ``cdk deploy --all`` to recover. The synth tests below
build both states with the stack names ``app.py`` uses, find which stacks own
the exports that appear only once the runtime exists, and check that the
script's agentcore leg deploys each of them in the same command.

The leg also saves the runtime ARN. The compute stack renders
AGENTCORE_RUNTIME_ARN only when ``agentcore_runtime_arn`` is in CDK context, so
a later ``cdk deploy ai-eval-studio-compute`` without the flag stripped it and
broke the relay. The script tests run a copy of the script against stub
``aws``, ``cdk`` and ``curl`` commands and check what it leaves in
``cdk.context.json``.
"""

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from aws_cdk import App, Environment

TEST_ENV = Environment(account="111111111111", region="us-east-1")
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dev-postdeploy.sh"


def _synth(context: dict) -> dict[str, dict]:
    """Synthesize the app's stacks, named as in app.py; return {stack: template}."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cdk"))
    from agentcore_stack import AgentCoreStack
    from auth_stack import AuthStack
    from compute_stack import ComputeStack
    from iam_stack import IamStack
    from network_stack import NetworkStack
    from observability_stack import ObservabilityStack
    from storage_stack import StorageStack

    app = App(context=context)
    auth = AuthStack(app, "ai-eval-studio-auth", env=TEST_ENV)
    network = NetworkStack(app, "ai-eval-studio-network", env=TEST_ENV)
    storage = StorageStack(app, "ai-eval-studio-storage", vpc=network.vpc, env=TEST_ENV)
    iam_stack = IamStack(
        app,
        "ai-eval-studio-iam",
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        file_system=storage.file_system,
        env=TEST_ENV,
    )
    observability = ObservabilityStack(app, "ai-eval-studio-observability", env=TEST_ENV)
    compute = ComputeStack(
        app,
        "ai-eval-studio-compute",
        vpc=network.vpc,
        coordinator_role=iam_stack.coordinator_task_role,
        worker_role=iam_stack.worker_task_role,
        worker_execution_role=iam_stack.worker_execution_role,
        coordinator_target_group=network.coordinator_target_group,
        coordinator_log_group=observability.coordinator_log_group,
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        file_system=storage.file_system,
        efs_access_point=storage.access_point,
        scoped_session_role=iam_stack.scoped_session_role,
        user_pool=auth.user_pool,
        spa_client=auth.spa_client,
        user_pool_domain=auth.user_pool_domain,
        env=TEST_ENV,
    )
    AgentCoreStack(
        app,
        "ai-eval-studio-agentcore",
        vpc=network.vpc,
        efs_access_point=storage.access_point,
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        scoped_session_role=iam_stack.scoped_session_role,
        runtime_role=iam_stack.agentcore_runtime_role,
        user_pool=auth.user_pool,
        spa_client=auth.spa_client,
        worker_launch_env=compute.worker_launch_env,
        env=TEST_ENV,
    )
    assembly = app.synth()
    return {
        artifact.stack_name: json.loads(
            (Path(assembly.directory) / artifact.template_file).read_text()
        )
        for artifact in assembly.stacks
    }


def _exports(template: dict) -> set[str]:
    return {
        output["Export"]["Name"]
        for output in template.get("Outputs", {}).values()
        if isinstance(output.get("Export", {}).get("Name"), str)
    }


@pytest.fixture(scope="module")
def stacks_that_gain_exports() -> set[str]:
    """Stacks whose exports exist only once the AgentCore runtime does."""
    first = _synth({})
    with_runtime = _synth({"agentcore_subnet_ids": "subnet-0123456789abcdef0"})
    return {
        name
        for name, template in with_runtime.items()
        if _exports(template) - _exports(first.get(name, {}))
    }


def _agentcore_leg_stacks() -> list[str]:
    """The stacks named on the script's ``cdk deploy`` that includes agentcore."""
    # Join backslash continuations so each command is one line.
    text = SCRIPT.read_text().replace("\\\n", " ")
    commands = [
        line for line in text.splitlines()
        if re.match(r"\s*cdk deploy\b", line) and "ai-eval-studio-agentcore" in line
    ]
    assert len(commands) == 1, commands
    words = shlex.split(commands[0], posix=True)
    return [w for w in words[2:] if w.startswith("ai-eval-studio-")]


def test_first_deploy_lacks_exports_the_runtime_imports(stacks_that_gain_exports):
    """The premise: some stack really does gain exports only with the runtime.

    If this ever stops holding, the leg can go back to deploying agentcore
    alone, and the next test is moot.
    """
    assert stacks_that_gain_exports, "no stack gains exports once the runtime exists"


def test_agentcore_leg_deploys_every_stack_that_gains_exports(stacks_that_gain_exports):
    """Each such stack is deployed in the same command as agentcore, and first."""
    leg = _agentcore_leg_stacks()
    missing = stacks_that_gain_exports - set(leg)
    assert not missing, (
        f"the agentcore leg deploys {leg} but not {sorted(missing)}, whose exports "
        "the runtime imports; a fresh deploy would fail with 'No export named'"
    )
    assert leg[-1] == "ai-eval-studio-agentcore", leg


# ---- The saved runtime ARN -------------------------------------------------

_ARN_A = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/ai_eval_studio_us_east_1-AAAAAAAAAA"
_ARN_B = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/ai_eval_studio_us_east_1-BBBBBBBBBB"

# Answers every AWS call dev-postdeploy.sh makes. ``RUNTIME_ARN`` is the
# agentcore stack's RuntimeArn output; ``SKIP_AGENTCORE`` set means no subnet
# lands in a supported AZ, so the script skips the agentcore leg.
_AWS_STUB = """#!/bin/bash
case "$*" in
  *"--stack-name ai-eval-studio-frontend"*) echo d111111abcdef8.cloudfront.net ;;
  *"--stack-name ai-eval-studio-network"*) echo vpc-0123456789abcdef0 ;;
  *"--stack-name ai-eval-studio-agentcore"*) echo "$RUNTIME_ARN" ;;
  *"describe-security-groups"*) echo sg-0123456789abcdef0 ;;
  *"describe-availability-zones"*) [[ -n "${SKIP_AGENTCORE:-}" ]] || printf 'us-east-1a\\tus-east-1b\\n' ;;
  *"describe-subnets"*) [[ -n "${SKIP_AGENTCORE:-}" ]] || echo subnet-0123456789abcdef0 ;;
  *"get-agent-runtime"*) echo True ;;
  *) echo "unexpected aws call: $*" >&2; exit 2 ;;
esac
"""


@pytest.fixture
def postdeploy(tmp_path):
    """A copy of the script under ``<tmp>/scripts``, so it writes ``<tmp>/cdk``.

    The script ``cd``s to its own ``../cdk``. Running the original would write
    the checkout's real cdk.context.json.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "cdk").mkdir()
    script = tmp_path / "scripts" / "dev-postdeploy.sh"
    shutil.copy(SCRIPT, script)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("aws", _AWS_STUB),
        ("cdk", "#!/bin/bash\nexit 0\n"),
        ("curl", "#!/bin/bash\necho 200\n"),
    ):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)

    def run(**env):
        result = subprocess.run(
            ["bash", str(script)],
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "HOME": str(tmp_path),
                "CDK_DEFAULT_ACCOUNT": "123456789012",
                "CDK_DEFAULT_REGION": "us-east-1",
                "COGNITO_DOMAIN_PREFIX": "ai-eval-studio-k3x9q2ab",
                "CDK_DOCKER": "podman",
                "DOCKER_HOST": "unix:///tmp/podman.sock",
                **env,
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads((tmp_path / "cdk" / "cdk.context.json").read_text())

    run.context_file = tmp_path / "cdk" / "cdk.context.json"
    return run


def test_postdeploy_saves_the_runtime_arn(postdeploy):
    """A plain compute deploy afterwards finds the ARN in cdk.context.json."""
    ctx = postdeploy(RUNTIME_ARN=_ARN_A)

    assert ctx["agentcore_runtime_arn"] == _ARN_A
    assert ctx["cloudfront_dns"] == "d111111abcdef8.cloudfront.net"


def test_a_rerun_overwrites_the_saved_arn(postdeploy):
    """The file tracks the runtime deployed now, not the first one it saw."""
    postdeploy(RUNTIME_ARN=_ARN_A)
    ctx = postdeploy(RUNTIME_ARN=_ARN_B)

    assert ctx["agentcore_runtime_arn"] == _ARN_B


@pytest.mark.parametrize("saved", [None, _ARN_A])
def test_a_skipped_agentcore_leg_leaves_the_saved_arn_alone(postdeploy, saved):
    """No runtime ARN this run: the key is neither added nor removed."""
    if saved:
        postdeploy.context_file.write_text(json.dumps({"agentcore_runtime_arn": saved}))

    ctx = postdeploy(SKIP_AGENTCORE="1", RUNTIME_ARN="")

    assert ctx.get("agentcore_runtime_arn") == saved
