# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the coordinator prompt builder.

Verifies that the coordinator prompt is properly assembled from
markdown files and dynamic project context.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.coordinator_prompt import build_coordinator_prompt, PROMPTS_DIR


EXPECTED_PROMPT_FILES = [
    "coordinator.md",
    "_rules.md",
    "data-analyzer.md",
    "harness-builder.md",
    "prompt-optimizer.md",
    "report-narrator.md",
]


class TestPromptFiles:
    """Verify all prompt markdown files exist."""

    def test_all_prompt_files_exist(self):
        """Every expected .md file should exist in the prompts directory."""
        for filename in EXPECTED_PROMPT_FILES:
            path = PROMPTS_DIR / filename
            assert path.exists(), f"Missing prompt file: {path}"

    def test_prompt_files_are_not_empty(self):
        """Prompt files should have meaningful content."""
        for filename in EXPECTED_PROMPT_FILES:
            content = (PROMPTS_DIR / filename).read_text()
            assert len(content) > 50, f"Prompt file too short: {filename}"


class TestBuildCoordinatorPrompt:
    """Verify build_coordinator_prompt assembles coordinator + rules + context."""

    def _make_project(self, tmp_path: Path) -> tuple[MagicMock, Path]:
        """Create a mock ProjectConfig plus its resolved project directory.

        Returns both, because the builder takes the RESOLVED project dir —
        the same thing ``resolve_project_dir`` hands every real call site.
        """
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        (project_dir / "data" / "raw").mkdir(parents=True)
        (project_dir / "data" / "raw" / "test.csv").write_text("a,b\n1,2")

        project = MagicMock()
        project.id = "test-project"
        project.name = "Test Project"
        project.description = "A test project"
        return project, project_dir

    def test_prompt_has_coordinator_content(self, tmp_path: Path):
        """The assembled prompt should include coordinator.md content.

        ``opt_state=None`` is passed explicitly. It used to be simulated by
        patching ``OptimizationState.load``, but the prompt builder no longer
        reads state at all — its caller resolves the authoritative state and
        passes it in — so the patch would mock a read that never happens.
        """
        project, project_dir = self._make_project(tmp_path)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "AI Eval Studio" in prompt
        assert "get_project_status" in prompt

    def test_prompt_has_rules(self, tmp_path: Path):
        """The assembled prompt should include rules.md content."""
        project, project_dir = self._make_project(tmp_path)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "run_eval" in prompt
        assert "validate_harness" in prompt

    def test_prompt_embeds_metric_reference(self, tmp_path: Path):
        """The DeepEval metric reference is EMBEDDED in the prompt.

        coordinator.md used to instruct the agent to Read the reference by
        repo-relative path — which resolves nowhere the agent runs (microVM
        cwd is the project tree; the packaged copy lives under the app
        install root). The agent then fell back to training memory of
        DeepEval's API, the exact drift the reference exists to prevent.
        Embedding removes the filesystem dependence in every execution home.
        """
        from app.prompts import PROMPTS_DIR

        project, project_dir = self._make_project(tmp_path)
        prompt = build_coordinator_prompt(project, project_dir, None)
        reference = (PROMPTS_DIR / "deepeval-metric-reference.md").read_text()
        assert reference in prompt, "full metric reference must be embedded"
        # And no instruction remains to Read it from a filesystem path.
        assert "backend/app/prompts/deepeval-metric-reference.md" not in prompt

    def test_prompt_has_project_context(self, tmp_path: Path):
        """The assembled prompt should include dynamic project context."""
        project, project_dir = self._make_project(tmp_path)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "test-project" in prompt
        assert "Test Project" in prompt
        assert "test.csv" in prompt


#: A config.py the prompt reads normally.
_VALID_CONFIG = '''\
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    models: list[str] = field(default_factory=lambda: ["m1"])
    data_file: str = "cases.csv"
    field_mappings: dict[str, str] = field(default_factory=lambda: {
        "input": "question",
        "expected_output": "answer",
    })
'''

#: A field the harness config schema rejects: model ids must be strings.
_BAD_MODELS_CONFIG = _VALID_CONFIG.replace('["m1"]', "[1, 2]")

#: config.py as an agent wrote it on a live project, adding a metric list of
#: names. The platform used to read a ``metrics`` field and rejected this
#: shape, which failed every chat turn on that project. Metrics are code in
#: src/metrics/metrics.py, so the platform no longer reads the field at all.
_METRIC_NAMES_CONFIG = _VALID_CONFIG + (
    "    # Declarative metric list, for platform/status display only.\n"
    '    metrics: list[str] = field(default_factory=lambda: ["Decision Correctness"])\n'
)


class TestUnreadableHarnessConfig:
    """A config.py the loader rejects must not stop the chat turn.

    The prompt is built before every turn, and the chat is how a broken
    harness gets repaired; direct file edits are refused. A raise here left the
    project unable to chat at all.
    """

    def _project_with_config(self, tmp_path: Path, source: str) -> tuple[MagicMock, Path]:
        project_dir = tmp_path / "test-project"
        (project_dir / "src").mkdir(parents=True)
        (project_dir / "src" / "config.py").write_text(source)
        project = MagicMock()
        project.id = "test-project"
        project.name = "Test Project"
        project.description = "A test project"
        return project, project_dir

    @pytest.mark.parametrize(
        "source",
        [_BAD_MODELS_CONFIG, "class EvalConfig(:\n"],
        ids=["models-not-strings", "syntax-error"],
    )
    def test_prompt_builds_and_says_the_config_is_unreadable(self, tmp_path, source):
        project, project_dir = self._project_with_config(tmp_path, source)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "test-project" in prompt
        assert "- **Harness config:** src/config.py could not be read: " in prompt
        assert "Fix the file before running anything." in prompt

    def test_a_valid_config_is_summarised_with_no_unreadable_note(self, tmp_path):
        project, project_dir = self._project_with_config(tmp_path, _VALID_CONFIG)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "- **Models:** m1\n" in prompt
        assert "- **Field Mappings:** input=question, expected_output=answer\n" in prompt
        assert "could not be read" not in prompt

    def test_a_metric_list_in_config_py_is_ignored(self, tmp_path):
        """The live project's file reads like a valid one: the field is unused."""
        project, project_dir = self._project_with_config(tmp_path, _METRIC_NAMES_CONFIG)
        prompt = build_coordinator_prompt(project, project_dir, None)
        assert "- **Models:** m1\n" in prompt
        assert "could not be read" not in prompt


class TestAskUserQuestionDisabled:
    """AskUserQuestion is unsupported in this app's UI (renders as an unanswerable
    chip, doesn't survive refresh) — it must be disabled both
    deterministically (SDK config) and probabilistically (prompt)."""

    def test_disallowed_tools_blocks_ask_user_question(self):
        """The SDK ``disallowed_tools`` config must remove AskUserQuestion."""
        from app.services.agent_runner import DISALLOWED_TOOLS

        assert "AskUserQuestion" in DISALLOWED_TOOLS

    def test_ask_user_question_not_in_the_allowed_set(self):
        """Sanity: it's not also listed as an allowed/available tool."""
        from app.services.agent_runner import (
            AWS_KNOWLEDGE_DOC_TOOLS,
            COORDINATOR_TOOLS,
        )

        allowed = set(COORDINATOR_TOOLS) | set(AWS_KNOWLEDGE_DOC_TOOLS)
        assert "AskUserQuestion" not in allowed

    def test_coordinator_prompt_forbids_ask_user_question(self):
        """The prompt guard must tell the coordinator not to use it."""
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        assert "Never use the AskUserQuestion tool" in content


class TestHarnessBuilderMantleAware:
    """harness-builder.md must teach the agent about mantle vs runtime
    models, runtime-first precedence, use-case-fit proposals, and the
    new set_models MCP tool."""

    def test_harness_builder_uses_set_models(self):
        """The prompt must instruct the agent to call set_models, not
        write src/config.py directly for model fields."""
        content = (PROMPTS_DIR / "harness-builder.md").read_text()
        assert "set_models" in content

    def test_harness_builder_mantle_aware(self):
        """The prompt must teach runtime/mantle catalog awareness with
        RUNTIME-first precedence and use-case-fit proposals.

        The precedence direction is asserted, not just the word "precedence":
        the prompt telling the agent the opposite of what
        ``_merge_runtime_first`` does is how the agent starts proposing ids the
        deduped catalog no longer contains.
        """
        content = (PROMPTS_DIR / "harness-builder.md").read_text().lower()
        assert "runtime-first precedence" in content
        assert "mantle-first" not in content
        assert "use-case-fit" in content
        assert "endpoint" in content
        assert "mantle" in content

    def test_harness_builder_no_direct_config_write_for_models(self):
        """Regression: the legacy 'Write src/config.py — EvalConfig
        dataclass' instruction MUST be replaced by the set_models call.
        The agent must NOT instruct the user to pick from a dropdown —
        mantle/runtime ids flow through chat prose from
        list_bedrock_models, and there is no frontend dropdown for target
        or judge models in this app."""
        content = (PROMPTS_DIR / "harness-builder.md").read_text()
        # Use a variable to keep this test file safe against any future
        # negative-grep that scans test sources for the legacy literal.
        legacy_direct_write = "Write `src/config.py` — EvalConfig dataclass"
        assert legacy_direct_write not in content
        assert "dropdown" not in content.lower()


class TestCoordinatorMetricOptimizability:
    """Coordinator must rule on metric/optimization compatibility:
    target metrics MUST read ``actual_output``; ContextualRecall/Precision/
    Relevancy do NOT and cannot be moved by prompt edits."""

    def test_coordinator_optimization_target_rule_present(self):
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        assert "Optimization-target metrics MUST read `actual_output`" in content

    def test_coordinator_lists_non_optimizable_metrics(self):
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        # The non-optimizable triple must be called out by name.
        assert "ContextualRecallMetric" in content
        assert "ContextualPrecisionMetric" in content
        assert "ContextualRelevancyMetric" in content

    def test_coordinator_references_metric_reference(self):
        """Coordinator.md points at the EMBEDDED reference section, not
        a filesystem path (which resolves nowhere the agent runs)."""
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        assert "DeepEval metric reference" in content
        assert "embedded later in this prompt" in content
        assert "deepeval-metric-reference.md" not in content


class TestCoordinatorJudgeSelectionDiscipline:
    """Judge selection must apply the same catalog-grounded, newest-suitable
    discipline as target selection."""

    def test_coordinator_judge_uses_target_discipline(self):
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        assert "Judge model — same discipline as targets" in content


class TestCoordinatorPreLaunchSoundnessCheck:
    """Phase 5 must include a pre-launch soundness checklist that blocks
    ``dispatch_optimization`` until each item passes."""

    def test_pre_launch_soundness_check_present(self):
        content = (PROMPTS_DIR / "coordinator.md").read_text()
        assert "Pre-launch soundness check" in content
        # Each of the four items lives in the checklist.
        assert "Every target metric is optimizable" in content
        assert "Required test-case fields are mapped" in content
        assert "fits the lineup" in content  # max_tokens line
        assert "Targets are reachable" in content


class TestHarnessBuilderReasoningMaxTokens:
    """Harness-builder must size max_tokens for reasoning models, not the
    bare 2048 default."""

    def test_harness_builder_reasoning_aware_max_tokens(self):
        content = (PROMPTS_DIR / "harness-builder.md").read_text()
        # The 2048 default lives in the dataclass field; the guidance must
        # tell the agent to size markedly higher when the lineup includes
        # reasoning models. Cite the runner signal so the agent can recognise
        # the symptom of the wrong call.
        assert "size for the lineup, not the default" in content
        assert "MANTLE_REASONING_TRUNCATED" in content


class TestDeepEvalMetricReference:
    """The prompt set ships a pinned-version DeepEval metric reference and corrects
    the data-analyzer's field-to-metric mapping."""

    def test_metric_reference_file_exists(self):
        path = PROMPTS_DIR / "deepeval-metric-reference.md"
        assert path.exists(), f"Missing metric reference: {path}"
        content = path.read_text()
        assert "deepeval==3.9.9" in content

    def test_metric_reference_lists_required_params(self):
        content = (PROMPTS_DIR / "deepeval-metric-reference.md").read_text()
        # Hits the metric names + required-field columns the coordinator
        # actually needs to read; if any of these drift, the regression
        # script in the doc is the recipe for fixing it.
        for name in ("AnswerRelevancyMetric", "FaithfulnessMetric",
                     "ContextualRecallMetric", "ContextualPrecisionMetric",
                     "HallucinationMetric"):
            assert name in content
        # retrieval_context vs context distinction is the load-bearing fix.
        assert "retrieval_context" in content
        assert "context" in content

    def test_data_analyzer_context_maps_to_hallucination_not_contextual_recall(self):
        content = (PROMPTS_DIR / "data-analyzer.md").read_text()
        # Negative grep — the legacy mapping that confused things must be gone.
        legacy = "context — ground truth context passages (list of strings, for Contextual Recall/Precision)"
        assert legacy not in content
        # Positive grep — context now correctly maps to HallucinationMetric.
        assert "HallucinationMetric" in content
        assert "retrieval_context" in content


class TestNarratorNAPricing:
    """report-narrator.md must teach the narrator to mention models
    with no LiteLLM pricing data (narrator side)."""

    def test_narrator_na_pricing_clause(self):
        """The narrator prompt must reference pricing_available and
        instruct the agent to write 'Pricing data is unavailable' prose
        in both per-run and executive sections."""
        content = (PROMPTS_DIR / "report-narrator.md").read_text()
        assert "pricing_available" in content
        assert content.lower().count("pricing data is unavailable") >= 2


class TestPromptReferenceIntegrity:
    """Two guards born from the metric-reference reachability miss.

    1. Version drift: the metric reference documents the PINNED deepeval;
       bumping the pin without regenerating the reference must fail CI, not
       silently hand the agent stale grounding.
    2. Reachability: agents can only open files inside the project working
       tree. A prompt that points the agent at a repo path (backend/,
       frontend/, infra/) references a file that exists in NO runtime
       environment — the agent degrades to training memory without erroring
       (measured live). Repo paths in prompts are banned outright;
       platform files an agent must read get embedded or scaffolded
       into the project.
    """

    def test_metric_reference_version_matches_installed_deepeval(self):
        import re

        import deepeval

        ref = (PROMPTS_DIR / "deepeval-metric-reference.md").read_text()
        m = re.search(r"deepeval==([\d.]+)", ref)
        assert m, "reference must declare the deepeval version it documents"
        assert m.group(1) == deepeval.__version__, (
            f"deepeval-metric-reference.md documents {m.group(1)} but the "
            f"installed (pinned) deepeval is {deepeval.__version__}. "
            "Regenerate the reference from the installed package "
            "(_required_params per metric class) as part of the version bump."
        )

    def test_no_prompt_points_agents_at_repo_paths(self):
        unreachable_prefixes = ("backend/", "frontend/", "infra/")
        offenders: list[str] = []
        for md in sorted(PROMPTS_DIR.glob("*.md")):
            content = md.read_text()
            for lineno, line in enumerate(content.splitlines(), 1):
                for prefix in unreachable_prefixes:
                    if f"`{prefix}" in line:
                        offenders.append(f"{md.name}:{lineno}: {line.strip()[:100]}")
        assert not offenders, (
            "Prompt(s) reference repo paths the agent cannot reach from the "
            "project working tree (it degrades to training memory without "
            "erroring):\n" + "\n".join(offenders) +
            "\nEmbed the content into the prompt at build time "
            "or point at the scaffolded copy inside the project."
        )
