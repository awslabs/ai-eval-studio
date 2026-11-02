# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for subagent definitions.

Verifies that all AgentDefinition instances are properly configured
with required fields and that their prompt files load correctly.
"""

from claude_agent_sdk import AgentDefinition

from app.agents import SUBAGENTS


EXPECTED_AGENTS = ["data-analyzer", "harness-builder", "prompt-optimizer"]


class TestSubagentDefinitions:
    """Verify subagent definitions are valid."""

    def test_all_expected_agents_defined(self):
        """All expected subagents should be defined."""
        for name in EXPECTED_AGENTS:
            assert name in SUBAGENTS, f"Missing subagent: {name}"

    def test_definitions_are_agent_definition_instances(self):
        """Each subagent should be an AgentDefinition instance."""
        for name, defn in SUBAGENTS.items():
            assert isinstance(defn, AgentDefinition), (
                f"{name} is {type(defn)}, expected AgentDefinition"
            )

    def test_definitions_have_required_fields(self):
        """Each subagent should have description, prompt, tools, and model."""
        for name, defn in SUBAGENTS.items():
            assert defn.description, f"{name} missing description"
            assert defn.prompt, f"{name} missing prompt"
            assert defn.tools, f"{name} missing tools"
            assert defn.model, f"{name} missing model"

    def test_models_are_valid(self):
        """Each subagent's model should be a valid model literal."""
        valid_models = {"sonnet", "opus", "haiku", "inherit"}
        for name, defn in SUBAGENTS.items():
            assert defn.model in valid_models, (
                f"{name} has invalid model: {defn.model}"
            )


class TestSubagentPrompts:
    """Verify subagent prompts have the right content."""

    def test_data_analyzer_prompt(self):
        """data-analyzer prompt should mention data analysis."""
        prompt = SUBAGENTS["data-analyzer"].prompt
        assert "data" in prompt.lower()
        assert "describe_data" in prompt

    def test_harness_builder_prompt(self):
        """harness-builder prompt should mention config.py fields."""
        prompt = SUBAGENTS["harness-builder"].prompt
        assert "config.py" in prompt
        assert "scaffold_harness" in prompt
        assert "validate_harness" in prompt

    def test_harness_builder_has_webfetch_not_websearch(self):
        """harness-builder should have WebFetch (for known model-card URLs)
        and NOT WebSearch (non-functional over Bedrock)."""
        tools = SUBAGENTS["harness-builder"].tools
        assert "WebFetch" in tools
        assert "WebSearch" not in tools

    def test_prompt_optimizer_has_optimization_knowledge(self):
        """prompt-optimizer prompt should contain failure categories, strategies, versioning."""
        prompt = SUBAGENTS["prompt-optimizer"].prompt
        # Failure categories
        assert "format mismatch" in prompt.lower()
        assert "missing knowledge" in prompt.lower()
        assert "reasoning error" in prompt.lower()
        # Strategies — the prompt must still teach the optimizer that
        # step-by-step reasoning structure is the answer to reasoning
        # errors and that few-shot examples drawn from passing cases
        # are a primary tool. Specific framework names (CO-STAR,
        # Chain-of-Thought, etc.) were removed when the prompt-architect
        # skill was removed; the underlying concepts remain.
        assert "step-by-step" in prompt.lower()
        assert "few-shot" in prompt.lower()
        # Anti-overfitting
        assert "overfitting" in prompt.lower()
        # Versioning
        assert "changelog" in prompt.lower()
