# DeepEval metric reference

**Pinned to `deepeval==3.9.9`.** Verified by `inspect`-ing `_required_params` on each metric class in the installed package. Re-verify on any DeepEval version bump — the metric-to-field mapping has changed across releases and training-memory drift on this surface has caused production bugs (e.g. mapping `context` to ContextualRecall when ContextualRecall actually keys on `retrieval_context`).

This is steering for the coordinator and data-analyzer. When in doubt, read the source — `from deepeval.metrics import <X>; X._required_params` is authoritative; web search is fallback only.

## Reading this table

For each metric:
- **Required `LLMTestCase` params**: the fields DeepEval will refuse to score without.
- **Reads `actual_output`**: ✅ if optimization-targetable (a prompt change can move the score), ❌ if scoring is independent of the prompt.
- **Scoring direction**: "higher = better" by default; "INVERTED" means 1.0 is the *desirable* end (no hallucination, no bias, no toxicity).
- **Use as optimization target**: ✅ ❌ — same as "Reads `actual_output`" but called out so the coordinator can scan quickly.

## Metrics

### Single-turn metrics (the `LLMTestCase` flow)

| Metric | Required params | Reads `actual_output` | Direction | Optimization target | Notes |
|---|---|---|---|---|---|
| `ExactMatchMetric` | `input`, `actual_output`, `expected_output` | ✅ | higher = better | ✅ | **DETERMINISTIC — no judge, no cost.** Exactly `expected.strip() == actual.strip()`: whitespace-trim only, NO case folding, no punctuation handling. It can never score a wrong answer as correct, so it is the safe default whenever the task wants a literal match. It IS strict — a model that wraps output in backticks or a ```lang fence, or appends a semicolon, scores 0. That failure is visible in the results, which is the right direction to fail. Prefer this over hand-rolling a normalized matcher; if the task genuinely needs tolerance, see the normalizer-validation checks in the harness's `src/metrics/metrics.py` docstring. Records as `Exact Match` in run files. |
| `PatternMatchMetric` | `input`, `actual_output` | ✅ | higher = better | ✅ | **DETERMINISTIC — no judge, no cost.** Regex, and two constraints decide whether it fits: it uses `fullmatch`, so the pattern must match the ENTIRE stripped output (`SELECT .*` will not match trailing prose); and `pattern` is a CONSTRUCTOR argument, so ONE pattern applies to every case — it does not read `expected_output` and therefore cannot express a per-case expected answer. Good for "is the output well-formed" (shape/syntax) checks, NOT for per-case correctness. `ignore_case=True` available; an invalid regex raises `ValueError` at construction. Records as `Pattern Match`. |
| `AnswerRelevancyMetric` | `input`, `actual_output` | ✅ | higher = better | ✅ | Default RAG / QA quality metric — judges whether the answer addresses the question. |
| `FaithfulnessMetric` | `input`, `actual_output`, `retrieval_context` | ✅ | higher = better | ✅ | Grounding check: does `actual_output` contradict `retrieval_context`? Needs the retrieved passages on the test case. |
| `ContextualRecallMetric` | `input`, `retrieval_context`, `expected_output` | ❌ | higher = better | ❌ — RETRIEVER QUALITY | Did the retriever surface enough of `expected_output`'s information? Independent of what the prompt produces. Use as a dataset/retriever-quality check, not a target. |
| `ContextualPrecisionMetric` | `input`, `retrieval_context`, `expected_output` | ❌ | higher = better | ❌ — RETRIEVER QUALITY | Are the relevant passages ranked above irrelevant ones? Independent of the prompt. Same usage rule as Recall. |
| `ContextualRelevancyMetric` | `input`, `retrieval_context` | ❌ | higher = better | ❌ — RETRIEVER QUALITY | Are the retrieved passages relevant to the input? Independent of the prompt. |
| `HallucinationMetric` | `input`, `actual_output`, `context` | ✅ | **INVERTED** — 1.0 = *no* hallucination | ✅ | Note the field is `context` (ground-truth passages), NOT `retrieval_context`. State both numbers when proposing the target ("target 1.0 = your requested 0% hallucination"). |
| `BiasMetric` | `input`, `actual_output` | ✅ | **INVERTED** — 1.0 = no bias | ✅ | |
| `ToxicityMetric` | `input`, `actual_output` | ✅ | **INVERTED** — 1.0 = no toxicity | ✅ | |
| `SummarizationMetric` | `input`, `actual_output` | ✅ | higher = better | ✅ | |
| `JsonCorrectnessMetric` | `input`, `actual_output` | ✅ | higher = better | ✅ | Judge-backed. Requires an `expected_schema` constructor argument — a **Pydantic `BaseModel` class**, not a JSON-schema dict — so it only applies when the task has one fixed output schema. `strict_mode=True` by DEFAULT, which forces `threshold=1`: any schema violation fails the case. Natural fit for structured-output tasks; for a per-case expected VALUE (not just shape) pair it with `ExactMatchMetric`. |
| `PromptAlignmentMetric` | `input`, `actual_output` | ✅ | higher = better | ✅ | Judge-backed. Requires a non-empty `prompt_instructions: list[str]` constructor argument (empty raises `ValueError`) — it scores whether the output obeyed those instructions. Note what this means for optimization: the instructions are pinned at construction while the prompt under optimization changes, so if you optimize the prompt you must keep the two in step or the metric measures adherence to a spec the prompt no longer states. |
| `ToolCorrectnessMetric` | `input`, `tools_called`, `expected_tools` | ❌ — reads `tools_called` | higher = better | depends — only if the harness exposes the model's tool calls into `tools_called`. |
| `GEval` | depends on `evaluation_params` | depends on whether `actual_output` is in `evaluation_params` | depends on the criteria | ✅ when criteria are written against `actual_output` (the default for "Correctness"-style judges). | LLM-as-judge with a criteria string. The `evaluation_params` you pass at construction determines what fields it reads — `[ACTUAL_OUTPUT, EXPECTED_OUTPUT]` is the typical "correctness" shape and IS optimizable. If you only pass `[INPUT, RETRIEVAL_CONTEXT]` it becomes a retrieval quality probe and is NOT optimizable. |

### How to use this table

**Picking a target.** Every entry in `target_metric_scores` must be a metric whose "Optimization target" column is ✅. Anything ❌ goes in the harness as a display-only metric (still scored every run, surfaced in the dashboard) but NOT in `target_metric_scores`.

**Picking field mappings.** Walk the "Required params" column for every selected metric and ensure each one appears in `field_mappings` (Tier 1). Missing required fields surface as null/None scores at runtime, which DeepEval treats as failures — silent until the user wonders why every case failed on a metric.

**Pre-launch soundness (Phase 5).** The coordinator's pre-launch checklist references this table directly. If any check fails, fix the config BEFORE `dispatch_optimization` — that's the cheapest correction point.

## When the table is not enough

**This table is a static snapshot, not a live lookup.** For anything it does not cover — a metric not listed, a constructor argument, whether a metric is judge-backed — read the installed package directly. You have `Bash`, and the project's own venv has DeepEval in it, so this is cheap and authoritative:

```bash
uv run python -c "
import deepeval.metrics as m, inspect
from deepeval.metrics import BaseMetric
for n in sorted(dir(m)):
    c = getattr(m, n)
    if inspect.isclass(c) and issubclass(c, BaseMetric) and not n.startswith('_'):
        rp = getattr(c, '_required_params', None)
        req = [p.name for p in inspect.signature(c.__init__).parameters.values()
               if p.default is inspect.Parameter.empty and p.name != 'self']
        print(n, rp, 'ctor_required=', req)
"
```

Do that before writing a custom metric — the thing you need often already ships. Web search is a last resort: it returns whatever version the page documents, which is not necessarily the one installed here.
