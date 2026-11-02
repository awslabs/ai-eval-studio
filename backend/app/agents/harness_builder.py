# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness builder subagent definition.

Creates and validates the eval harness: scaffolds boilerplate,
writes config.py/dataset_loader.py/metrics, validates imports
and model access.
"""

from claude_agent_sdk import AgentDefinition

from app.prompts import PROMPTS_DIR

harness_builder = AgentDefinition(
    description=(
        "Creates and validates the eval harness: scaffolds boilerplate, "
        "writes config.py/dataset_loader.py/metrics, validates imports and "
        "model access. Delegate after user confirms all Phase 2 decisions."
    ),
    prompt=(PROMPTS_DIR / "harness-builder.md").read_text(),
    tools=[
        "mcp__eval__scaffold_harness", "mcp__eval__validate_harness",
        "mcp__eval__list_bedrock_models",
        # AWS Knowledge MCP doc tools — look up Bedrock model cards
        # (use-case, context window, reasoning, modalities, launch date)
        # so target/judge model picks are grounded in current facts, not
        # training memory. Scoped to the two read/search doc tools only;
        # registered as a remote HTTP MCP on the coordinator (inherited).
        "mcp__aws_knowledge__aws___search_documentation",
        "mcp__aws_knowledge__aws___read_documentation",
        # Fetch a provider model-card / guide page at a URL surfaced by an
        # AWS Knowledge search (or supplied by the user) so model facts come
        # from a current source, not training memory. WebSearch is omitted
        # deliberately — it does not work over Bedrock, so only the
        # fetch-a-known-URL capability is useful here.
        "WebFetch",
        "Read", "Write", "Edit", "Bash",
    ],
    # Inherits from the coordinator. The UI picker is restricted to Sonnet and
    # Opus (both reliable for code-generation), so inheriting is always safe
    # for writing project-specific dataset_loader.py / metrics / config.py.
    model="inherit",
)
