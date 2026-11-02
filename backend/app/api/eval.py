# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite router for eval-related endpoints.

Concerns are split into eval_jobs, eval_runs, and optimization submodules.
"""

from fastapi import APIRouter

from . import eval_jobs, eval_runs, optimization

router = APIRouter()
router.include_router(eval_jobs.router)
router.include_router(eval_runs.router)
router.include_router(optimization.router)
