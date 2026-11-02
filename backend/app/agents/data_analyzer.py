# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data analyzer subagent definition.

Analyzes uploaded datasets to identify structure, field types,
data quality issues, and proposes field mappings for evaluation.
"""

from claude_agent_sdk import AgentDefinition

from app.prompts import PROMPTS_DIR

data_analyzer = AgentDefinition(
    description=(
        "Analyzes uploaded datasets to identify structure, field types, "
        "data quality issues, and proposes field mappings for evaluation. "
        "Delegate when user uploads data or asks to analyze their dataset."
    ),
    # Metric reference EMBEDDED alongside the role prompt: data-analyzer.md
    # instructs the analyzer to consult it, and a file-path Read fails in
    # the microVM (repo-relative path, wrong cwd, app root outside the
    # working tree). Same treatment as the coordinator prompt.
    prompt=(
        (PROMPTS_DIR / "data-analyzer.md").read_text()
        + "\n\n# DeepEval metric reference\n\n"
        + (PROMPTS_DIR / "deepeval-metric-reference.md").read_text()
    ),
    tools=[
        "mcp__eval__describe_data",
        "mcp__eval__list_upload",
        "mcp__eval__move_file",
        "mcp__eval__finalize_upload",
        "Bash",
        "Read",
    ],
    # Inherits the coordinator's tier so one picker controls the whole pipeline.
    # Task is mechanical (extension classification + describe_data) so any tier
    # the user picks is reliable enough.
    model="inherit",
)
