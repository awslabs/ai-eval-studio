# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: no cross-stack export may carry a value that changes on every deploy.

CloudFormation refuses to update an export while another stack imports it. So an
export whose value is derived from a resource whose *identity* changes on each
deploy makes the exporting stack permanently undeployable — every subsequent
`cdk deploy` fails with:

    Update canceled. Cannot update export <name> as it is in use by <stack>

That is not a hypothetical. Exporting the worker task definition by `Ref` — which
renders the REVISIONED task-definition ARN, bumped on every deploy — bricked
compute deploys in production. The stable fix is to pass the task-definition
FAMILY, which RunTask resolves to the latest ACTIVE revision.

The class of coupling has to fail at synth, not at deploy, which is what this
module is for. It works on the real synthesized templates rather than on source
text, so it catches the coupling however it is introduced (a direct `Ref`, a
property that renders one, or a value threaded through a constructor).

**Which resource types are per-deploy.** A `Ref` to most resources yields a stable
physical name or id. It is only unsafe for resource types whose `Ref` embeds a
mutable version/revision. ECS task definitions are the case we have; the list is
extensible as others turn up.
"""

import json
import sys
from pathlib import Path

import pytest
from aws_cdk import App, Environment

TEST_ENV = Environment(account="111111111111", region="us-east-1")
TEST_SUBNET_IDS = "subnet-0123456789abcdef0"

# Resource types whose ``Ref`` returns an identity that CHANGES on every deploy.
# Exporting such a value across stacks is the trap this module guards.
#
# AWS::ECS::TaskDefinition: Ref returns the ARN INCLUDING the revision number
# (``.../ai-eval-studio-worker:29``), and CDK bumps the revision whenever the task
# definition changes — which for us is every image build.
PER_DEPLOY_REF_RESOURCE_TYPES: dict[str, str] = {
    "AWS::ECS::TaskDefinition": (
        "Ref yields the revisioned task-definition ARN, which changes on every "
        "deploy. Export the family instead — RunTask resolves a bare family to "
        "the latest ACTIVE revision."
    ),
}


@pytest.fixture(scope="module")
def synthesized():
    """Synthesize the full app and return {stack_artifact_id: template dict}.

    Builds the same stack hierarchy the other combined-synth tests use, so the
    templates inspected here are the ones that would really be deployed.
    """
    infra_path = Path(__file__).resolve().parent.parent / "cdk"
    sys.path.insert(0, str(infra_path))

    from agentcore_stack import AgentCoreStack
    from auth_stack import AuthStack
    from compute_stack import ComputeStack
    from iam_stack import IamStack
    from network_stack import NetworkStack
    from observability_stack import ObservabilityStack
    from storage_stack import StorageStack

    app = App(context={"agentcore_subnet_ids": TEST_SUBNET_IDS})

    network = NetworkStack(app, "network", env=TEST_ENV)
    storage = StorageStack(app, "storage", vpc=network.vpc, env=TEST_ENV)
    auth = AuthStack(app, "auth", env=TEST_ENV)
    iam_stack = IamStack(
        app,
        "iam",
        sessions_bucket=storage.sessions_bucket,
        uploads_staging_bucket=storage.uploads_staging_bucket,
        snapshots_bucket=storage.snapshots_bucket,
        table=storage.table,
        file_system=storage.file_system,
        env=TEST_ENV,
    )
    observability = ObservabilityStack(app, "observability", env=TEST_ENV)
    compute = ComputeStack(
        app,
        "compute",
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
        "agentcore",
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
    templates = {}
    for artifact in assembly.stacks:
        templates[artifact.stack_name] = json.loads(
            (Path(assembly.directory) / artifact.template_file).read_text()
        )
    return templates


def _iter_nodes(node):
    """Yield every dict nested anywhere inside a template fragment."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nodes(item)


def _referenced_logical_ids(value) -> set[str]:
    """Collect logical ids reached by ``Ref`` or ``Fn::GetAtt`` inside a value."""
    found: set[str] = set()
    for node in _iter_nodes(value):
        ref = node.get("Ref")
        if isinstance(ref, str):
            found.add(ref)
        get_att = node.get("Fn::GetAtt")
        if isinstance(get_att, list) and get_att and isinstance(get_att[0], str):
            found.add(get_att[0])
    return found


def _imported_export_names(template: dict) -> set[str]:
    """Collect every export name this template consumes via ``Fn::ImportValue``."""
    found: set[str] = set()
    for node in _iter_nodes(template):
        imported = node.get("Fn::ImportValue")
        if isinstance(imported, str):
            found.add(imported)
    return found


def test_no_exported_value_is_a_per_deploy_resource_ref(synthesized):
    """No stack may EXPORT a value derived from a per-deploy resource identity.

    Checked on exports rather than only on imports, because an unconsumed export
    is a loaded gun: the day something imports it, the exporting stack can never
    be updated again.
    """
    offenders = []

    for stack_name, template in synthesized.items():
        resources = template.get("Resources", {})
        for output_name, output in template.get("Outputs", {}).items():
            export = output.get("Export")
            if not export:
                continue  # a plain output is not cross-stack coupling
            for logical_id in _referenced_logical_ids(output.get("Value")):
                res_type = resources.get(logical_id, {}).get("Type")
                if res_type in PER_DEPLOY_REF_RESOURCE_TYPES:
                    offenders.append(
                        f"{stack_name}.Outputs.{output_name} exports a value derived "
                        f"from {logical_id} ({res_type}) — "
                        f"{PER_DEPLOY_REF_RESOURCE_TYPES[res_type]}"
                    )

    assert not offenders, (
        "Cross-stack export(s) carry a per-deploy value. Once imported, the "
        "exporting stack can never be updated again — CloudFormation reports "
        '"Cannot update export ... as it is in use by". Offenders:\n  '
        + "\n  ".join(offenders)
    )


def test_agentcore_imports_only_stable_compute_exports(synthesized):
    """Every compute export agentcore imports must be deploy-stable.

    Narrower and more direct than the sweep above: it verifies the exact coupling
    that broke — agentcore importing something from compute that compute cannot
    then update.
    """
    agentcore = synthesized["agentcore"]
    compute = synthesized["compute"]
    compute_resources = compute.get("Resources", {})

    imported = _imported_export_names(agentcore)
    compute_exports = {
        output["Export"]["Name"]: output
        for output in compute.get("Outputs", {}).values()
        if output.get("Export")
    }

    offenders = []
    for export_name, output in compute_exports.items():
        # Cross-stack export names are Fn::Sub/Join structures in the exporting
        # template but plain strings on the importing side; match on the literal
        # logical-id suffix CDK generates, which appears in both.
        name_text = json.dumps(export_name)
        if not any(name_text.strip('"') in imp or imp in name_text for imp in imported):
            continue
        for logical_id in _referenced_logical_ids(output.get("Value")):
            res_type = compute_resources.get(logical_id, {}).get("Type")
            if res_type in PER_DEPLOY_REF_RESOURCE_TYPES:
                offenders.append(
                    f"agentcore imports compute export {name_text} derived from "
                    f"{logical_id} ({res_type})"
                )

    assert not offenders, (
        "agentcore imports a per-deploy compute export, which blocks all future "
        "compute deploys:\n  " + "\n  ".join(offenders)
    )


def test_worker_task_definition_is_never_referenced_across_stacks(synthesized):
    """The worker task definition must not be reachable from another stack at all.

    The decisive check for this incident, stated as a property rather than as a
    diff: whatever plumbing exists, no other stack may end up importing an export
    whose value came from the worker task-definition resource.
    """
    compute = synthesized["compute"]
    resources = compute.get("Resources", {})

    worker_task_def_ids = {
        logical_id
        for logical_id, resource in resources.items()
        if resource.get("Type") == "AWS::ECS::TaskDefinition"
        and resource.get("Properties", {}).get("Family") == "ai-eval-studio-worker"
    }
    assert worker_task_def_ids, "worker task definition not found — test is stale"

    leaking = []
    for output_name, output in compute.get("Outputs", {}).items():
        if not output.get("Export"):
            continue
        if _referenced_logical_ids(output.get("Value")) & worker_task_def_ids:
            leaking.append(output_name)

    assert not leaking, (
        "The worker task definition is exported cross-stack via "
        f"{leaking}. Pass the task-definition FAMILY instead: the revisioned ARN "
        "changes every deploy and pins the export permanently."
    )
