# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One derivation for a metric's base name.

DeepEval appends the metric TYPE to ``__name__`` depending on the installed
version, so the same configured metric can serialize as ``"Answer Correctness"`` or
``"Answer Correctness [GEval]"``. Anything that joins two name spaces — config keys
against run-file names, a report's metric list against the metric audit's
verdicts — has to normalize both sides with the SAME rule, or the join silently
misses and whatever depended on it disappears without an error.

It had three platform-side implementations. Two were the regex below; the third
(``report._short_metric_name``) was ``raw.split(" [")[0]``, which strips from the
first bracket ANYWHERE in the name: ``"Recall [strict] check"`` became
``"Recall"``. The harness template has an explicit test that the same input is
left alone, so the report rendered a name the rest of the system did not agree
with, and a metric named that way would never match its own audit verdict.

The harness template keeps its own copy (``EvalRunner._base_metric_name``) out of
necessity — it runs inside the generated project's venv and cannot import from
here. Keep the two regexes IDENTICAL; ``tests/test_metric_name.py`` pins them
against each other.
"""
from __future__ import annotations

import re

#: A trailing `` [MetricType]`` suffix. Anchored at the end so brackets that are
#: part of the name survive.
_TYPE_SUFFIX = re.compile(r"\s*\[[A-Za-z0-9 _-]+\]$")


def base_metric_name(name: str) -> str:
    """Strip a trailing `` [MetricType]`` suffix from a metric name.

    Args:
        name: A metric name from run-file ``metrics_data``, from harness config,
            or from ``target_metric_scores``. ``None``-ish input yields ``""``.

    Returns:
        The name with a trailing bracketed type suffix removed. Brackets
        elsewhere in the name are content and are preserved.
    """
    return _TYPE_SUFFIX.sub("", name or "")
