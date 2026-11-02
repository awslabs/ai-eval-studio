# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report narrator subagent definition.

Reads a server-prepared facts-pack (computed deterministically from run
files) and writes ONLY the prose fields a report needs: takeaway,
recommendation, journey narrative, failure pattern clusters. Output is
delivered via the ``save_report`` MCP tool — composing prose in the chat
reply accomplishes nothing, same discipline as ``save_prompt``.

Critical: the narrator NEVER supplies numbers. Numbers are recomputed
server-side by the tool. The agent's job is to interpret the facts, not
restate them.
"""

from claude_agent_sdk import AgentDefinition

from app.prompts import PROMPTS_DIR


report_narrator = AgentDefinition(
    description=(
        "Reads a complete facts-pack JSON for an eval report and writes "
        "ONLY the prose fields (takeaway, recommendation, journey narrative, "
        "failure pattern clusters, etc.). Delivers output by calling the "
        "save_report MCP tool. Never invents or recomputes numbers — quotes "
        "them verbatim from the provided facts. Clusters failures into ≤5 "
        "patterns; never emits one card per raw failed case."
    ),
    prompt=(PROMPTS_DIR / "report-narrator.md").read_text(),
    tools=["mcp__eval__save_report"],
    # Inherits the project's chosen agent model — same discipline as the
    # prompt-optimizer subagent.
    model="inherit",
)
