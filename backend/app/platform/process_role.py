# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-role resolver for execution-home differentiation.

This module provides a resolve-once accessor that identifies which execution home
this process is: the coordinator Fargate service or the AgentCore runtime microVM.
The coordinator owns job lifecycle side effects (progress republishing, completion
turns, report dispatch) because it holds the browser WebSocket connections. The
microVM runs the same turn code but has zero WebSocket connections, so it must
be barred from owning those side effects to prevent progress/completion blindness.

**Shape:** Like ``arch_split.py``, this is a module named after the thing it
resolves (the process role), with a module constant read from the environment at
import time plus a zero-argument accessor.

**Why this exists:** The microVM and the coordinator run the same coordinator turn
code (``services/agent_runner.py``, ``services/mcp_tools/dispatch.py``), but only
the coordinator holds browser WebSocket connections. Job lifecycle side effects
(starting the progress poller, claiming and firing the completion turn, dispatching
reports) must happen on the process that can reach the browser, not on the process
where dispatch happened to execute. This marker gates those side effects so ownership
stays on the coordinator.

**Eval worker task:** The eval worker task (``app.scripts.worker_main``) leaves this
marker unset and that is harmless — it contains no poller call site and runs no
coordinator turn code. The marker is a microVM-vs-coordinator distinction only.

**Testing:** Tests monkeypatch the resolved module constant (``_PROCESS_ROLE``), not
the environment, to flip roles. This is the same pattern used by the cost kill-switch
tests in ``test_cost_kill_switch.py``.
"""

import os

# Environment variable name for the process-role marker
PROCESS_ROLE_ENV_VAR = "EVAL_STUDIO_PROCESS_ROLE"

# Role value for the AgentCore runtime microVM
AGENTCORE_RUNTIME_ROLE = "agentcore_runtime"

# Role value for the coordinator Fargate service (the default when unset)
COORDINATOR_ROLE = "coordinator"

# Resolved process role (read once at import time)
# Defaults to COORDINATOR_ROLE when unset, matching the behavior of the Fargate
# service and the eval worker task (neither sets the marker).
_PROCESS_ROLE = os.environ.get(PROCESS_ROLE_ENV_VAR, COORDINATOR_ROLE).lower().strip()


def is_agentcore_runtime() -> bool:
    """Return whether this process is the AgentCore runtime microVM.

    This accessor takes zero arguments — the role is process-scoped, not
    request-scoped or tenant-scoped. It returns the module constant resolved
    once at import time.

    The microVM and the coordinator run the same coordinator turn code, but only
    the coordinator holds browser WebSocket connections. Job lifecycle side effects
    (starting the progress poller, claiming and firing the completion turn, dispatching
    reports) must be gated on this accessor so they happen on the coordinator, not
    on the microVM.

    Returns:
        True if this process is the AgentCore runtime, False if this process is
        the coordinator or the eval worker task.
    """
    return _PROCESS_ROLE == AGENTCORE_RUNTIME_ROLE
