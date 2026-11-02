# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RequestContextFilter that injects correlation IDs.

Pinning the contract so a future log-format refactor doesn't accidentally
strip the request_id / user_id / project_id columns operators rely on
to grep CloudWatch.
"""
from __future__ import annotations

import logging

from app.platform.log_context import RequestContextFilter
from app.platform.types import RequestContextValue, request_context


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="m",
        args=None,
        exc_info=None,
    )


def test_filter_uses_placeholder_when_no_context():
    """Outside any request the filter renders ``-`` placeholders.

    Background tasks and process startup hit the logger before any
    auth middleware runs; the filter must never error and must return
    a stable token operators can grep for.
    """
    f = RequestContextFilter()
    record = _make_record()
    assert f.filter(record) is True
    assert record.request_id == "-"
    assert record.user_id == "-"
    assert record.project_id == "-"
    assert record.turn_id == "-"
    assert record.job_id == "-"


def test_filter_attaches_truncated_ids_when_context_present():
    """Inside a request the filter pulls the ContextVar and shortens
    long IDs to 8 chars so log lines stay readable.

    Full IDs are still queryable via Cognito / DDB; the truncation is
    a display concern only.
    """
    ctx = RequestContextValue(
        user_id="cognito-sub-1234567890abcdef",
        request_id="abcd1234-efgh-5678",
        project_id="proj-12345-uuid",
        turn_id="turn-abcdef",
        job_id="job-987654",
    )
    token = request_context.set(ctx)
    try:
        f = RequestContextFilter()
        record = _make_record()
        assert f.filter(record) is True
        assert record.request_id == "abcd1234"
        assert record.user_id == "cognito-"
        assert record.project_id == "proj-123"
        assert record.turn_id == "turn-abc"
        assert record.job_id == "job-9876"
    finally:
        request_context.reset(token)


def test_filter_never_drops_records():
    """The filter is an enrichment, not a gate. ``filter`` must always
    return ``True`` so a buggy ContextVar can't silence diagnostics."""
    f = RequestContextFilter()
    assert f.filter(_make_record()) is True
    ctx = RequestContextValue(user_id="u", request_id="r")
    token = request_context.set(ctx)
    try:
        assert f.filter(_make_record()) is True
    finally:
        request_context.reset(token)
