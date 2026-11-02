# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A caller must not be able to choose which AgentCore session its turn joins.

``ChatRequest.session_id`` is an APP session id. The SPA reads it from the
``result`` envelope, which carries the SDK's ``message.session_id``, and the
schema documents it as a fallback for the project's persisted session. The relay
dispatch used to assign it straight to ``runtime_session_id`` and hand it to
AgentCore as the ``runtimeSessionId``.

Two facts make that a cross-tenant hole rather than a type confusion:

* AgentCore treats the id as opaque and does not enforce session-to-user
  mapping — AWS states the client backend must maintain that relationship.
* ``resolve_resume`` only overrides the caller's value when a durable registry
  item exists with a non-empty app session. That item lives 2h
  (``AGENTCORE_SESSION_TTL_SECONDS``), so any caller returning after that
  window had their own value forwarded verbatim.

It went unnoticed because an app session id is a 36-character UUID, which clears
the 33-character minimum the relay enforces on the header — so the wrong id was
always well-formed.

These tests pin the property at the dispatch boundary, which is where the
caller's value enters, plus the defense-in-depth half: a stored runtime id wins
over anything supplied.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.asyncio

#: Stands in for an id the caller should not be able to steer the turn with.
#: Shaped like a real app session id — 36 characters — because that is the
#: property that let the wrong id through the relay's 33-character minimum.
#: Hex letters in every group on purpose: an all-digit group long enough to
#: look like an AWS account id reads as a leaked account id to anyone scanning
#: the repo.
_FOREIGN_SESSION = "a1b2c3d4-e5f6-4a7b-8c9d-e1f2a3b4c5d6"


@pytest.fixture
def dev_platform_with_store(projects_dir: Path, monkeypatch):
    """Dev platform with a REAL metadata store.

    Real store because the registry read inside ``resolve_resume`` is half of
    what is under test; a mocked return would prove nothing about precedence.
    Authz is stubbed to a no-op — it already passed by the time the session id
    is consumed, and it is covered elsewhere.
    """
    from app.platform import Platform, set_platform
    from app.platform.dev_local import (
        FilesystemMetadataStore,
        LocalAuth,
        NoOpTokenVending,
        _build_dev_session_store,
    )

    set_platform(None)

    metadata_store = FilesystemMetadataStore(projects_dir=projects_dir)
    auth = LocalAuth(metadata_store=metadata_store)

    async def _noop_require_access(self, user_id, project_id, action):
        return None

    monkeypatch.setattr(LocalAuth, "require_access", _noop_require_access)

    set_platform(
        Platform(
            metadata_store=metadata_store,
            session_store=_build_dev_session_store(),
            auth=auth,
            token_vending=NoOpTokenVending(),
        )
    )
    yield
    set_platform(None)


def _mock_request():
    request = Mock()
    request.headers.get.return_value = "Bearer test-access-token"
    return request


async def _relay_on(*args, **kwargs):
    return True


async def _dispatch(session_id: str | None, persisted: str | None = "session-123"):
    """Run POST /chat down the relay branch and return the invoke kwargs."""
    from app.api.chat import chat
    from app.models.schemas import ChatRequest

    project = MagicMock()
    project.session_id = persisted

    with patch(
        "app.platform.arch_split.resolve_agentcore_relay", side_effect=_relay_on
    ), patch(
        "app.services.agentcore_relay_client.invoke_agentcore_turn",
        new_callable=AsyncMock,
    ) as mock_invoke, patch(
        "app.api.chat.current_user_id", return_value="userA"
    ), patch("app.main.get_project_manager") as mock_pm_getter:
        mock_pm = MagicMock()
        mock_pm.get_project.return_value = project
        mock_pm_getter.return_value = mock_pm

        resp = await chat(
            ChatRequest(project_id="projX", message="hi", session_id=session_id),
            _mock_request(),
        )

    assert resp["status"] == "running", f"relay branch did not run: {resp}"
    assert mock_invoke.call_count == 1
    return mock_invoke.call_args.kwargs


class TestTheCallerCannotNameTheRuntimeSession:
    async def test_a_supplied_session_id_never_becomes_the_runtime_session_id(
        self, dev_platform_with_store
    ):
        """The authoritative assertion. If this fails, a caller can direct its
        turn into a microVM session it does not own."""
        kwargs = await _dispatch(_FOREIGN_SESSION)
        assert kwargs["runtime_session_id"] is None, (
            "the caller's session_id reached AgentCore as the runtimeSessionId: "
            f"{kwargs['runtime_session_id']!r}. Only resolve_resume may supply "
            "that id — it reads the (user, project)-keyed registry, so its value "
            "can only ever belong to this caller."
        )

    async def test_the_supplied_value_survives_as_the_app_session_fallback(
        self, dev_platform_with_store
    ):
        """Dropping it entirely would be a silent regression, not a fix — the
        schema documents it as the fallback when the backend failed to persist
        the session, so it has to keep arriving in that slot."""
        kwargs = await _dispatch(_FOREIGN_SESSION)
        assert kwargs["app_session_fallback"] == _FOREIGN_SESSION

    async def test_omitting_it_is_still_valid(self, dev_platform_with_store):
        """A project's first turn sends nothing; that path must be unchanged."""
        kwargs = await _dispatch(None)
        assert kwargs["runtime_session_id"] is None
        assert kwargs["app_session_fallback"] is None


class TestAStoredRuntimeIdWinsOverAnythingSupplied:
    """Defense in depth on the resolver itself.

    The dispatch fix means nothing is supplied any more, so this pins the layer
    that would still hold if some future caller passed one.
    """

    async def test_resolve_resume_prefers_the_stored_runtime_id(
        self, dev_platform_with_store
    ):
        from app.services.agentcore_session_registry import (
            record_session,
            resolve_resume,
        )

        await record_session(
            "userA",
            "projX",
            runtime_session_id="stored-runtime-id",
            app_session_id="stored-app-id",
        )

        runtime, app = await resolve_resume("userA", "projX", _FOREIGN_SESSION)

        assert runtime == "stored-runtime-id", (
            f"a supplied runtime id displaced the stored one: {runtime!r}"
        )
        assert app == "stored-app-id"

    async def test_no_stored_session_yields_no_runtime_id_of_its_own(
        self, dev_platform_with_store
    ):
        """With nothing stored, the resolver has no id to offer. It returns what
        it was given — which is exactly why the dispatch must give it None."""
        from app.services.agentcore_session_registry import resolve_resume

        runtime, _ = await resolve_resume("userB", "projY", None)
        assert runtime is None
