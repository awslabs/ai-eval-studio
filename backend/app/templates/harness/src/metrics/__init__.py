# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics package for evaluation. Re-exports build_metrics from metrics module."""

from .metrics import build_metrics

__all__ = ["build_metrics"]
