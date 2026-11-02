# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subagent definitions for the coordinator.

Each subagent is defined in its own module and re-exported here
as the SUBAGENTS dict, which gets registered on ClaudeAgentOptions.
"""

from app.agents.data_analyzer import data_analyzer
from app.agents.harness_builder import harness_builder
from app.agents.prompt_optimizer import prompt_optimizer
from app.agents.report_narrator import report_narrator

SUBAGENTS = {
    "data-analyzer": data_analyzer,
    "harness-builder": harness_builder,
    "prompt-optimizer": prompt_optimizer,
    "report-narrator": report_narrator,
}

__all__ = ["SUBAGENTS"]
