# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metric extraction for optimization iterations.

Used by ``optimization_executor`` (the per-iteration job body), the
``record_optimization_iteration`` MCP tool, and the report facts builder
(``report.py``) to compute per-model/per-metric pass rates from eval results.
One implementation so the callers can't drift.
"""

from typing import Any


def extract_iteration_metrics(results: list[dict]) -> dict[str, Any]:
    """Extract iteration-level metrics from eval results.

    Computes total/passed cases, per-metric pass rates, per-model
    pass rates, and per-model-per-metric pass rates from the raw
    eval output (list of per-model run dicts).

    Args:
        results: List of per-model run dicts from the eval harness.

    Returns:
        Dict with keys: passed_cases, total_cases, metrics_summary,
        per_model_pass_rates, per_model_metrics.
    """
    total_cases = 0
    passed_cases = 0
    metrics_summary: dict[str, float] = {}
    per_model_pass_rates: dict[str, float] = {}
    per_model_metrics: dict[str, dict[str, float]] = {}

    for model_run in results:
        model_results = model_run.get("results", [])
        model_id = (
            model_run.get("model_id")
            or model_run.get("hyperparameters", {}).get("model")
            or model_run.get("test_run_id")
            or "unknown"
        )
        model_total = len(model_results)
        model_passed = sum(1 for r in model_results if r.get("success", False))
        total_cases += model_total
        passed_cases += model_passed
        if model_total > 0:
            per_model_pass_rates[model_id] = model_passed / model_total

        model_metric_passes: dict[str, int] = {}
        for r in model_results:
            for md in r.get("metrics_data", []):
                raw_name = md["name"]
                name = raw_name.split(" [")[0] if " [" in raw_name else raw_name
                if name not in metrics_summary:
                    metrics_summary[name] = 0.0
                if md.get("success", False):
                    metrics_summary[name] = metrics_summary.get(name, 0) + 1
                    model_metric_passes[name] = model_metric_passes.get(name, 0) + 1
                else:
                    model_metric_passes.setdefault(name, 0)

        if model_total > 0:
            per_model_metrics[model_id] = {
                name: passes / model_total
                for name, passes in model_metric_passes.items()
            }

    if total_cases > 0:
        for name in metrics_summary:
            metrics_summary[name] /= total_cases

    return {
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "metrics_summary": metrics_summary,
        "per_model_pass_rates": per_model_pass_rates,
        "per_model_metrics": per_model_metrics,
    }
