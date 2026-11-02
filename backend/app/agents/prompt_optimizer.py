# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt optimizer subagent definition.

Analyzes eval results (failures, successes, patterns), reads
current prompt and iteration history, and writes an improved
prompt version.
"""

from claude_agent_sdk import AgentDefinition

from app.prompts import PROMPTS_DIR

prompt_optimizer = AgentDefinition(
    description=(
        "Analyzes eval results (failures, successes, patterns), reads "
        "current prompt and iteration history, writes an improved prompt "
        "version. Delegate after an eval run completes and optimization "
        "is needed."
    ),
    prompt=(PROMPTS_DIR / "prompt-optimizer.md").read_text(),
    tools=[
        "mcp__eval__analyze_failures", "mcp__eval__get_eval_results",
        "mcp__eval__save_prompt", "Read",
    ],
    # Inherits from the coordinator. Runs every optimization iteration, so
    # picking Opus at the project level flows through here — which is often
    # where quality compounds matter most (better rewrites → faster convergence).
    model="inherit",
)
