#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation harness CLI entry point.

This script provides a command-line interface for running evaluations with
multiple models, prompt versions, and configuration options. It supports both
interactive mode (terminal output) and JSON progress mode (for platform integration).

Key features:
- Multi-model evaluation with comma-separated or space-separated model lists
- Model override via full Bedrock model IDs
- Judge model override for LLM-as-a-judge metrics
- JSON progress events for platform integration (--json-progress)
- Consistent result file naming: eval_results_{version}_{timestamp}.json

Usage:
    # Evaluate with default models
    python run_eval.py

    # Evaluate specific models — FULL Bedrock ids (comma- or space-separated)
    python run_eval.py --models us.amazon.nova-lite-v1:0,openai.gpt-5.6-luna
    python run_eval.py --models us.amazon.nova-lite-v1:0 openai.gpt-5.6-luna

    # Use specific prompt version
    python run_eval.py --prompt-version v2

    # Platform integration mode (JSON progress events)
    python run_eval.py --json-progress --models us.amazon.nova-lite-v1:0
"""

import argparse
import sys
from pathlib import Path

from src.config import EvalConfig
from src.evaluation.runner import EvalRunner


def parse_model_list(models_arg: str | list[str]) -> list[str]:
    """
    Parse model list from CLI argument.

    Accepts both comma-separated strings and space-separated lists:
    - "us.amazon.nova-lite-v1:0,us.amazon.nova-pro-v1:0"
      → ["us.amazon.nova-lite-v1:0", "us.amazon.nova-pro-v1:0"]
    - ["us.amazon.nova-lite-v1:0", "us.amazon.nova-pro-v1:0"]
      → ["us.amazon.nova-lite-v1:0", "us.amazon.nova-pro-v1:0"]

    Parameters:
        models_arg: CLI argument (string or list)

    Returns:
        List of full Bedrock model IDs, unchanged — no short-name
        resolution is applied.
    """
    if isinstance(models_arg, list):
        # Space-separated: flatten any comma-separated items
        result = []
        for item in models_arg:
            result.extend(m.strip() for m in item.split(",") if m.strip())
        return result
    else:
        # Single string: split by comma
        return [m.strip() for m in models_arg.split(",") if m.strip()]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation Harness - Multi-model LLM evaluation with DeepEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help=(
            "Models to evaluate (comma-separated or space-separated). "
            "FULL Bedrock model ids only — values are passed to Bedrock "
            "verbatim, no short-name resolution exists. "
            "If not specified, uses models from EvalConfig."
        ),
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="baseline",
        help="Prompt version name (maps to prompts/{name}.md). Default: baseline",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock. Default: us-east-1",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (optional)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model for LLM-as-a-judge metrics (optional, defaults to EvalConfig.judge_model_id)",
    )
    parser.add_argument(
        "--json-progress",
        action="store_true",
        help="Emit JSON progress events to stdout (for platform integration)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="manual",
        choices=["manual", "optimization", "validation"],
        help=(
            "Run provenance: ``manual`` (user-triggered eval), "
            "``optimization`` (one iteration of the optimization loop), "
            "or ``validation`` (1-case smoke run from validate_harness — "
            "filtered out of list_runs so smokes never pollute the chart)."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Override DeepEval async metric concurrency. Defaults to "
            "EvalConfig.max_concurrent if set, else 20. Smoke checks pass "
            "1 to keep a misconfigured project from triggering throttling."
        ),
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help=(
            "Override the dataset file. A bare filename is resolved under "
            "data/raw/ (matching EvalConfig.data_file); an absolute path is "
            "used as-is (e.g. a held-out validation set or a synthetic/merged "
            "file the platform passes). Defaults to EvalConfig.data_file."
        ),
    )

    args = parser.parse_args()

    # Parse model list (handles both comma-separated and space-separated)
    models = parse_model_list(args.models) if args.models else None

    try:
        # Initialize config — only pass non-None values so EvalConfig
        # defaults are used for anything not specified on the CLI
        config_kwargs: dict = {
            "prompt_version": args.prompt_version,
            "aws_region": args.region,
        }
        if models is not None:
            config_kwargs["models"] = models
        if args.max_examples is not None:
            config_kwargs["max_examples"] = args.max_examples
        if args.profile is not None:
            config_kwargs["aws_profile"] = args.profile
        if args.judge_model is not None:
            config_kwargs["judge_model_id"] = args.judge_model
        if args.data_file is not None:
            # Pass the override to EvalConfig.data_file. Two shapes, both
            # handled by the dataset_loader contract:
            #   * bare filename  → loader joins it under data/raw/ (the
            #     standard EvalConfig.data_file convention) — this is the
            #     common case: a held-out file uploaded into data/raw/.
            #   * absolute path  → loader opens it as-is (a synthetic/merged
            #     file the platform built outside data/raw/).
            # We pass the value through unchanged rather than pre-joining,
            # so a bare name still flows through the loader's data/raw join
            # exactly as the default data_file does.
            config_kwargs["data_file"] = args.data_file

        config = EvalConfig(**config_kwargs)

        if not args.json_progress:
            print("=" * 80)
            print("Evaluation Harness")
            print("=" * 80)
            print(f"Models: {', '.join(config.models)}")
            print(f"Prompt version: {config.prompt_version}")
            print(f"Max examples: {config.max_examples or 'all'}")
            print(f"Judge model: {config.judge_model_id}")
            print("=" * 80)

        # Run evaluation. ``--max-concurrent`` overrides whatever
        # ``EvalConfig`` declares (or its absence) without forcing every
        # agent-written EvalConfig to add the field.
        runner = EvalRunner(
            config,
            json_progress=args.json_progress,
            source=args.source,
            max_concurrent_override=args.max_concurrent,
        )
        results = runner.run()

        # Save results
        output_path = runner.save_results(results, config.output_dir)

        if not args.json_progress:
            print(f"\n✓ Results saved to: {output_path}")
            print(f"  Total models evaluated: {len(results)}")

    except Exception as e:
        if args.json_progress:
            import json
            print(json.dumps({"event": "error", "error": str(e)}), file=sys.stderr, flush=True)
        else:
            print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
