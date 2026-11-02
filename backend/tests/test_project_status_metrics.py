# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""get_project_status reports no metric list from config.py.

A harness defines its metrics as code in src/metrics/metrics.py. The status
tool used to report a ``metrics`` field read from config.py, which was always
empty; an agent read the empty list as a gap and wrote a list of names into
config.py, which the platform then could not parse.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.services.mcp_tools.status import get_project_status

_get_project_status = get_project_status.handler

_CONFIG = '''\
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    models: list[str] = field(default_factory=lambda: ["m1"])
    data_file: str = "cases.csv"
    prompt_version: str = "baseline"
    metrics: list[str] = field(default_factory=lambda: ["Decision Correctness"])
'''


async def test_status_has_no_metric_list_from_config_py(tmp_path: Path):
    project_dir = tmp_path / "status-metrics"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "prompts").mkdir()
    (project_dir / "output").mkdir()
    (project_dir / "src" / "config.py").write_text(_CONFIG)

    with patch("app.platform.arch_split.resolve_arch_split", new=AsyncMock(return_value=False)), \
         patch("app.services.mcp_tools.status.resolve_project_dir", return_value=project_dir):
        result = await _get_project_status({"project_id": project_dir.name})

    assert result.get("is_error") is not True
    harness = json.loads(result["content"][0]["text"])["harness_config"]
    assert "metrics" not in harness
    assert harness["models"] == ["m1"]
