# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool registrations for eval operations.

Each submodule owns one functional area. ``create_eval_mcp_server`` wires
every registered tool into a single ``create_sdk_mcp_server`` so the agent
sees the whole toolbox while the code stays partitioned by domain.
"""

from claude_agent_sdk import create_sdk_mcp_server

from app.platform.mcp_authz import apply_authz_to_tools

from .data import describe_data
from .dispatch import (
    dispatch_eval,
    dispatch_harness_build,
    dispatch_optimization,
    request_user_input,
    resume_job,
)
from .eval import (
    cancel_eval,
    check_eval_status,
    delete_eval_run,
    get_case_details,
    get_eval_results,
    list_eval_runs,
)
from .harness import scaffold_harness, set_models, validate_harness
from .models import list_bedrock_models
from .optimization import record_optimization_iteration
from .pricing import refresh_pricing
from .prompt import analyze_failures, save_prompt
from .reports import save_report
from .status import get_project_status
from .synthesis import (
    approve_synthetic_cases,
    generate_synthetic,
    list_synthetic_batches,
)
from .uploads import finalize_upload, list_upload, move_file


# Single source of truth for the eval-server tool list. The
# authz-coverage test in ``tests/test_mcp_authz.py`` reads this constant
# to assert ``TOOL_ACTION_MAP`` is in sync.
REGISTERED_TOOLS = [
    describe_data,
    # Dispatch tools (fire-and-forget, completion turns deliver results)
    dispatch_eval,
    dispatch_optimization,
    dispatch_harness_build,
    request_user_input,
    resume_job,
    # Eval read-side + cancellation
    check_eval_status,
    list_eval_runs,
    get_eval_results,
    get_case_details,
    cancel_eval,
    delete_eval_run,
    # Prompt + optimization helpers
    save_prompt,
    analyze_failures,
    record_optimization_iteration,
    # Report narrative
    save_report,
    # Discovery / scaffolding / synthesis / uploads / pricing / status
    list_bedrock_models,
    scaffold_harness,
    set_models,
    validate_harness,
    generate_synthetic,
    list_synthetic_batches,
    approve_synthetic_cases,
    list_upload,
    move_file,
    finalize_upload,
    refresh_pricing,
    get_project_status,
]


def create_eval_mcp_server():
    """Create the in-process MCP server with all registered eval tools.

    Each handler is wrapped with the authz check from
    :func:`app.platform.mcp_authz.apply_authz_to_tools`.
    The wrapper is transparent — names, schemas, and signatures are
    preserved — but every tool call now consults
    ``platform.auth.require_access`` before executing.
    """
    return create_sdk_mcp_server(
        name="eval",
        version="1.0.0",
        tools=apply_authz_to_tools(REGISTERED_TOOLS),
    )


__all__ = ["REGISTERED_TOOLS", "create_eval_mcp_server"]
