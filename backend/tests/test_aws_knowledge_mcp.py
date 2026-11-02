# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the AWS Knowledge MCP wiring.

The coordinator and the harness-builder subagent are given the AWS
Knowledge MCP (a remote HTTP MCP) so they can look up live Bedrock
model-card capability data at recommendation time instead of relying on
stale training memory. These tests pin the wiring: the server is
registered as a remote HTTP MCP (no subprocess spawn), and only the two
read/search documentation tools are exposed — not the server's broader
surface.
"""

from app.agents import SUBAGENTS
from app.services.agent_runner import (
    AWS_KNOWLEDGE_DOC_TOOLS,
    AWS_KNOWLEDGE_MCP_SERVER,
    AWS_KNOWLEDGE_MCP_URL,
    _aws_knowledge_mcp_config,
)


class TestAwsKnowledgeMcpConfig:
    def test_config_is_remote_http_not_subprocess(self):
        """Must be a remote HTTP MCP (``type: "http"``) — NOT a uvx-spawned
        stdio server. The prior pricing MCP was removed precisely because
        spawning added package-fetch + boot latency to every
        initialize() handshake; a remote MCP has no local process."""
        cfg = _aws_knowledge_mcp_config()
        assert cfg["type"] == "http"
        assert cfg["url"] == AWS_KNOWLEDGE_MCP_URL
        assert cfg["url"].startswith("https://")
        # No ``command``/``args`` keys — those would indicate a stdio
        # subprocess config, the pattern we explicitly avoid.
        assert "command" not in cfg
        assert "args" not in cfg

    def test_url_is_bare_domain_no_mcp_suffix(self):
        """Regression guard: the URL MUST be the bare domain, NOT the
        ``/mcp`` path. Both pass the initialize handshake, but ``/mcp``
        fails on the actual tool-call POST with "Http operation is not
        supported for gateway protocol type MCP" (verified against the
        real CLI streamable-HTTP transport). The bare domain is what
        every official awslabs client config uses."""
        cfg = _aws_knowledge_mcp_config()
        assert cfg["url"] == "https://knowledge-mcp.global.api.aws"
        assert not cfg["url"].rstrip("/").endswith("/mcp")

    def test_config_matches_sdk_http_typeddict(self):
        """Shape must satisfy the installed SDK's McpHttpServerConfig."""
        from claude_agent_sdk.types import McpHttpServerConfig

        cfg = _aws_knowledge_mcp_config()
        # TypedDict required keys present, no unexpected keys.
        required = set(McpHttpServerConfig.__required_keys__)
        allowed = required | set(McpHttpServerConfig.__optional_keys__)
        assert required.issubset(cfg.keys())
        assert set(cfg.keys()).issubset(allowed)

    def test_doc_tools_are_scoped_and_prefixed(self):
        """Only the two read/search doc tools, with the
        ``mcp__<server>__<servertool>`` prefix the SDK applies to a
        remote MCP. Explicitly NOT a wildcard, and NOT the server's
        recommend / regions / skill tools."""
        assert AWS_KNOWLEDGE_DOC_TOOLS == [
            f"mcp__{AWS_KNOWLEDGE_MCP_SERVER}__aws___search_documentation",
            f"mcp__{AWS_KNOWLEDGE_MCP_SERVER}__aws___read_documentation",
        ]
        joined = " ".join(AWS_KNOWLEDGE_DOC_TOOLS)
        for forbidden in ("recommend", "retrieve_skill", "list_regions",
                          "regional_availability", "*"):
            assert forbidden not in joined


class TestHarnessBuilderInheritsDocTools:
    def test_harness_builder_has_scoped_doc_tools(self):
        """The harness-builder subagent also does model selection, so it
        gets the same two doc tools — and nothing broader."""
        tools = SUBAGENTS["harness-builder"].tools
        assert "mcp__aws_knowledge__aws___search_documentation" in tools
        assert "mcp__aws_knowledge__aws___read_documentation" in tools
        # No broader Knowledge MCP surface leaked in.
        assert not any(
            x in t
            for t in tools
            for x in ("recommend", "retrieve_skill", "list_regions",
                      "regional_availability")
        )
