# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SessionStore protocol — re-export of the Claude Agent SDK's contract.

We re-export the SDK's ``SessionStore`` Protocol here so call sites
depend on ``app.platform`` rather than the SDK package directly — and
so the conformance suite at
``claude_agent_sdk.testing.run_session_store_conformance`` is what
validates our implementations.

Dev-mode uses the SDK's default on-disk store (no override needed).
Cloud-mode uses ``S3SessionStore`` (in :mod:`app.platform.aws`).
"""
from __future__ import annotations

from claude_agent_sdk import SessionStore as SessionStoreProtocol


__all__ = ["SessionStoreProtocol"]
