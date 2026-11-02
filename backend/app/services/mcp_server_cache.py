# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server singleton for eval tools.

Pre-warmed at app startup to avoid first-request import cost
(deepeval, pandas, boto3). Shared by agent turns and background
optimization runs.
"""

import logging

logger = logging.getLogger(__name__)

_eval_mcp_server = None


def pre_warm_mcp_server() -> None:
    """Pre-warm the eval MCP server at app startup.

    Eagerly initializes the server so the first chat message doesn't pay
    the import cost of deepeval, pandas, boto3, etc.
    """
    global _eval_mcp_server
    if _eval_mcp_server is None:
        from app.services.mcp_tools import create_eval_mcp_server
        _eval_mcp_server = create_eval_mcp_server()
        logger.info("MCP eval server pre-warmed")


def get_eval_mcp_server():
    """Get or initialize the singleton eval MCP server.

    Returns the pre-warmed instance if available, otherwise creates one
    on first access.
    """
    global _eval_mcp_server
    if _eval_mcp_server is None:
        from app.services.mcp_tools import create_eval_mcp_server
        _eval_mcp_server = create_eval_mcp_server()
    return _eval_mcp_server
