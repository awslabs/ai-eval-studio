# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool for refreshing a project's src/pricing.json after a model change.

Chat-driven entry point: when the user swaps or adds target/judge models
mid-session, the coordinator edits ``src/config.py`` and then calls
``refresh_pricing`` so downstream cost recommendations reflect the new
selection. Thin wrapper around ``write_pricing_file`` — no smoke eval,
no revalidation.
"""
from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import tool

from app.services.bedrock_pricing import write_pricing_file
from app.services.harness_config import load_harness_config
from app.services.mcp_response import mcp_error, mcp_success
from app.services.project_paths import resolve_project_dir

logger = logging.getLogger(__name__)


@tool(
    "refresh_pricing",
    "Refetch Bedrock pricing for the project's configured target + judge "
    "models and overwrite src/pricing.json. Call after editing src/config.py "
    "to change models mid-session so cost estimates stay accurate.",
    {"project_id": str},
)
async def refresh_pricing(args: dict[str, Any]) -> dict[str, Any]:
    """Refresh ``src/pricing.json`` for a project's current model selection.

    Reads ``src/config.py`` to derive the set of target + judge model IDs,
    refetches pricing from the LiteLLM price list, and overwrites the file.
    Returns an error if the project has no config.py or if the fetch fails.

    Args:
        args: Dict with ``project_id``.

    Returns:
        MCP success message listing the models that were repriced, or an
        error if the config is missing or the fetch failed.
    """
    logger.info("[refresh_pricing] project=%s", args.get("project_id", "?")[:12])
    try:
        project_dir = resolve_project_dir(args["project_id"])
    except ValueError as exc:
        return mcp_error(str(exc))

    harness = load_harness_config(project_dir)
    if harness is None:
        return mcp_error(
            "Project has no src/config.py — scaffold and configure the harness first."
        )

    model_ids: list[str] = list(harness.models or [])
    if harness.judge_model_id:
        model_ids.append(harness.judge_model_id)
    # Dedup while preserving order so the log line reads sensibly.
    seen: set[str] = set()
    unique_model_ids = [m for m in model_ids if not (m in seen or seen.add(m))]

    if not unique_model_ids:
        return mcp_error(
            "No models configured — set EvalConfig.models and judge_model_id "
            "before calling refresh_pricing."
        )

    pricing_path = await write_pricing_file(project_dir, unique_model_ids)
    if pricing_path is None:
        return mcp_error(
            "Failed to fetch Bedrock pricing — the LiteLLM price-list fetch "
            "returned an error. Harness will continue using whatever prices "
            "are currently in src/pricing.json (or fallback estimates)."
        )

    model_list = "\n".join(f"  - {m}" for m in unique_model_ids)
    return mcp_success(
        "Pricing refreshed for:\n"
        f"{model_list}\n\n"
        f"Wrote {pricing_path.relative_to(project_dir)}."
    )
