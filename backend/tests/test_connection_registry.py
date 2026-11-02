# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket connection persistence: not covered, because the route does not do it.

``websocket_endpoint`` (app/api/ws.py) registers each connection with the
in-process hub only. It does not call ``put_connection`` / ``delete_connection``:
a socket is accepted before any project is bound, and resolving ``arch_split``
needs a project id. The store methods themselves exist on every metadata-store
adapter, which ``test_adapter_parity.py`` checks.
"""
