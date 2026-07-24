"""Integration tests for Phase 5 Task 5.2: progressive-disclosure mode-transition prompt.

After guided_session.terminal is set (any reason, including COMPLETED), the first
subsequent freeform chat turn uses a layered system prompt:

    [guided_pipeline.md content]
    ## Mode Transition — Guided → Freeform
    ...reason...
    LIFTED
    [pipeline_composer.md content]

Subsequent freeform turns use the freeform skill alone.
``transition_consumed=True`` is persisted after the first layered turn.

Tests exercise two terminal paths:
  1. ``exited_to_freeform`` (reason=user_pressed_exit via manual terminal seeding)
  2. ``completed`` (terminal kind=completed with pipeline_yaml set)
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.state_machine import (
    TerminalKind,
    TerminalReason,
    TerminalState,
)
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Fake LLM response: pure-chat, no tool calls
# ---------------------------------------------------------------------------


def _fake_chat_response(text: str = "Hello from freeform mode!") -> SimpleNamespace:
    """LiteLLM-shaped response with no tool calls (pure chat reply)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=None,
                    content=text,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        model="test-model",
        id="test-id",
    )


# ---------------------------------------------------------------------------
# Fixture: composer_with_freeform_client
#
# Extends the base composer_test_client by wiring a real ComposerServiceImpl
# so that POST /api/sessions/{id}/messages is exercisable.
# ---------------------------------------------------------------------------


@pytest.fixture
def composer_freeform_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with both guided and freeform endpoints wired.

    Differs from composer_test_client (in conftest.py) by setting
    ``app.state.composer_service`` to a real ``ComposerServiceImpl`` and
    adding ``app.state.scoped_secret_resolver = None``.  The composer
    service has ``_litellm_acompletion`` patched to avoid real LLM calls.

    A fake ``OPENAI_API_KEY`` env var is set and both mocked models use that
    provider so the service's boot-time availability checks pass without real
    API keys.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-integration-tests")

    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)

    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.progressive_disclosure"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="gpt-4o-mini",
        composer_advisor_model="openai/gpt-4.1-mini",
        composer_max_composition_turns=5,
        composer_max_discovery_turns=5,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=100,
        shareable_link_signing_key=b"\x00" * 32,
    )
    catalog = create_catalog_service()
    composer_service = ComposerServiceImpl.for_trained_operator(
        catalog=catalog,
        settings=settings,
        sessions_service=session_service,
        session_engine=engine,
    )

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    app.state.session_service = session_service
    app.state.session_engine = engine
    app.state.blob_service = blob_service
    app.state.settings = settings
    app.state.composer_service = composer_service
    app.state.scoped_secret_resolver = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = catalog
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    profiles = MagicMock(spec=OperatorProfileRegistry)
    profiles.public_schema.side_effect = lambda _plugin_id, schema, *, available_aliases: schema
    app.state.operator_profile_registry = profiles
    app.state.plugin_snapshot_factory = lambda _user: snapshot
    app.state.composer_recorder = BufferingRecorder()
    app.state.composer_progress_registry = ComposerProgressRegistry()

    router = create_session_router()
    app.include_router(router)

    blobs_router = create_blobs_router()
    app.include_router(blobs_router)

    client = TestClient(app)
    yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "progressive-disclosure-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _seed_terminal_guided_session(
    client: TestClient,
    session_id: str,
    terminal: TerminalState,
) -> None:
    """Directly persist a GuidedSession with the given terminal into the DB.

    Faster and more deterministic than driving 8+ wizard steps.
    """
    service: SessionServiceImpl = client.app.state.session_service
    session_uuid = UUID(session_id)

    # Load or create current state
    state_record = asyncio.run(service.get_current_state(session_uuid))

    if state_record is None:
        # No state yet — build minimal CompositionState
        from dataclasses import replace

        from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

        state = _initial_composition_state_with_guided_session()
        # Attach the desired terminal
        new_guided = replace(state.guided_session, terminal=terminal)
        state = replace(state, guided_session=new_guided)
        existing_meta: dict = {}
    else:
        from dataclasses import replace

        from elspeth.contracts.freeze import deep_thaw

        state = state_from_record(state_record)
        existing_meta = dict(deep_thaw(state_record.composer_meta)) if state_record.composer_meta else {}
        if state.guided_session is None:
            # Attach a fresh initial guided session with the terminal
            from elspeth.web.composer.guided.state_machine import GuidedSession

            guided = GuidedSession.initial()
            guided = replace(guided, terminal=terminal)
        else:
            guided = replace(state.guided_session, terminal=terminal)
        state = replace(state, guided_session=guided)

    new_composer_meta = {**existing_meta, "guided_session": state.guided_session.to_dict()}
    state_d = state.to_dict()
    state_data = CompositionStateData(
        sources=state_d["sources"],
        nodes=state_d["nodes"],
        edges=state_d["edges"],
        outputs=state_d["outputs"],
        metadata_=state_d["metadata"],
        is_valid=False,
        validation_errors=None,
        composer_meta=new_composer_meta,
    )
    asyncio.run(service.save_composition_state(session_uuid, state_data, provenance="session_seed"))


def _get_current_guided_session(client: TestClient, session_id: str) -> dict:
    """Load the current GuidedSession dict from the DB (bypasses HTTP)."""
    service: SessionServiceImpl = client.app.state.session_service
    state_record = asyncio.run(service.get_current_state(UUID(session_id)))
    if state_record is None:
        return {}
    from elspeth.contracts.freeze import deep_thaw

    meta = deep_thaw(state_record.composer_meta) if state_record.composer_meta else {}
    return dict(meta.get("guided_session") or {})


def _send_message(client: TestClient, session_id: str, content: str) -> dict:
    resp = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": content},
    )
    assert resp.status_code == 200, f"send_message failed: {resp.status_code} {resp.text}"
    return resp.json()


def _seed_user_message(client: TestClient, session_id: str, content: str = "retry this") -> None:
    """Insert a user message directly so recompose finds a valid last-user-turn.

    Post Phase-1A: ``add_message`` requires ``writer_principal`` keyword.
    User-channel messages use ``route_user_message`` to match the production
    POST /sessions/{id}/messages route's writer_principal assignment.
    """
    service: SessionServiceImpl = client.app.state.session_service
    asyncio.run(
        service.add_message(
            UUID(session_id),
            "user",
            content,
            writer_principal="route_user_message",
        )
    )


def _recompose(client: TestClient, session_id: str) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/recompose")
    assert resp.status_code == 200, f"recompose failed: {resp.status_code} {resp.text}"
    return resp.json()


def _inject_one_assistant_insert_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the next assistant insert after any preceding state write."""
    service: SessionServiceImpl = client.app.state.session_service
    original_insert = service._insert_chat_message  # type: ignore[attr-defined]
    failed = False

    def _fail_once(*args, **kwargs):
        nonlocal failed
        if kwargs.get("role") == "assistant" and not failed:
            failed = True
            raise IntegrityError(
                "INSERT chat_messages",
                {},
                RuntimeError("injected transition assistant failure"),
            )
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(service, "_insert_chat_message", _fail_once)


# ---------------------------------------------------------------------------
# Test: first freeform turn after exit uses transition prompt
# ---------------------------------------------------------------------------


class TestFirstFreeformTurnAfterExit:
    def test_first_turn_uses_transition_prompt(self, composer_freeform_client: TestClient) -> None:
        """First freeform chat turn after an exited_to_freeform terminal uses the layered prompt."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)

        captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            _send_message(composer_freeform_client, session_id, "what can you do?")

        assert len(captured_messages) >= 1, "LLM should have been called once"
        messages = captured_messages[0]

        # First message must be the system prompt
        system_messages = [m for m in messages if m["role"] == "system"]
        assert system_messages, "No system messages found"
        system_content = system_messages[0]["content"]

        assert "## Mode Transition" in system_content, f"Expected transition header in system prompt, got: {system_content[:500]}"
        assert "LIFTED" in system_content
        assert "user_pressed_exit" in system_content


class TestSecondFreeformTurnAfterTransition:
    def test_second_turn_uses_freeform_only_prompt(self, composer_freeform_client: TestClient) -> None:
        """After the first (transition) turn, subsequent turns use freeform-only prompt."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)

        all_captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            all_captured_messages.append(kwargs["messages"])
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            # First freeform turn — should use transition prompt
            _send_message(composer_freeform_client, session_id, "first message")

        # After first turn, transition_consumed should be persisted as True
        gs_dict = _get_current_guided_session(composer_freeform_client, session_id)
        assert gs_dict.get("transition_consumed") is True, f"transition_consumed not set to True after first turn. GuidedSession: {gs_dict}"

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            # Second freeform turn — must NOT use transition prompt
            _send_message(composer_freeform_client, session_id, "second message")

        assert len(all_captured_messages) == 2, f"Expected exactly 2 LLM calls, got {len(all_captured_messages)}"

        # First call: transition prompt present
        first_system = next(m for m in all_captured_messages[0] if m["role"] == "system")
        assert "## Mode Transition" in first_system["content"]

        # Second call: transition prompt absent
        second_system = next(m for m in all_captured_messages[1] if m["role"] == "system")
        assert "## Mode Transition" not in second_system["content"], "Second turn should NOT contain the transition header"

    def test_send_message_assistant_failure_rolls_back_transition_consumption(
        self,
        composer_freeform_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The transition prompt remains available when its response cannot persist."""
        session_id = _create_session(composer_freeform_client)
        _seed_terminal_guided_session(
            composer_freeform_client,
            session_id,
            TerminalState(
                kind=TerminalKind.EXITED_TO_FREEFORM,
                reason=TerminalReason.USER_PRESSED_EXIT,
                pipeline_yaml=None,
            ),
        )
        captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _fake_chat_response("durable transition response")

        _inject_one_assistant_insert_failure(composer_freeform_client, monkeypatch)
        with patch("elspeth.web.composer.service._litellm_acompletion", side_effect=_fake_acompletion):
            with pytest.raises(IntegrityError, match="injected transition assistant failure"):
                composer_freeform_client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"content": "leave guided mode"},
                )

            assert _get_current_guided_session(composer_freeform_client, session_id).get("transition_consumed") is False
            body = _send_message(composer_freeform_client, session_id, "retry the transition")
        assert body["message"]["content"] == "durable transition response"
        assert _get_current_guided_session(composer_freeform_client, session_id).get("transition_consumed") is True
        assert len(captured_messages) == 2
        assert all("## Mode Transition" in next(m for m in turn if m["role"] == "system")["content"] for turn in captured_messages)


class TestTransitionPromptAfterCompletedTerminal:
    def test_completed_terminal_uses_transition_prompt(self, composer_freeform_client: TestClient) -> None:
        """After COMPLETED terminal, the first freeform turn uses the layered prompt with 'completed_pipeline'."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.COMPLETED,
            reason=None,
            pipeline_yaml="# stub pipeline yaml",
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)

        captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            _send_message(composer_freeform_client, session_id, "actually change the sink to CSV")

        assert len(captured_messages) >= 1
        messages = captured_messages[0]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert system_messages
        system_content = system_messages[0]["content"]

        assert "## Mode Transition" in system_content, f"COMPLETED terminal must trigger transition prompt. Got: {system_content[:500]}"
        assert "completed_pipeline" in system_content, "COMPLETED terminal must use 'completed_pipeline' reason string"
        assert "LIFTED" in system_content


# ---------------------------------------------------------------------------
# Tests: recompose path mirrors send_message for progressive disclosure
# ---------------------------------------------------------------------------


class TestRecomposeTransitionPrompt:
    """Recompose is a retried freeform chat call — spec §5.5 progressive
    disclosure fires on the same semantics as send_message."""

    def test_recompose_uses_transition_prompt_on_first_freeform_turn(self, composer_freeform_client: TestClient) -> None:
        """First recompose after guided_session.terminal is set uses the layered prompt."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)
        # Recompose requires the last conversation message to be a user turn.
        _seed_user_message(composer_freeform_client, session_id, "try again after exit")

        captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            _recompose(composer_freeform_client, session_id)

        assert len(captured_messages) >= 1, "LLM should have been called once"
        messages = captured_messages[0]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert system_messages, "No system messages found"
        system_content = system_messages[0]["content"]

        assert "## Mode Transition" in system_content, f"Expected transition header in recompose system prompt, got: {system_content[:500]}"
        assert "LIFTED" in system_content
        assert "user_pressed_exit" in system_content

    def test_recompose_persists_transition_consumed(self, composer_freeform_client: TestClient) -> None:
        """After recompose fires the transition prompt, transition_consumed=True is persisted."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)
        _seed_user_message(composer_freeform_client, session_id, "try again")

        async def _fake_acompletion(**kwargs):
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            body = _recompose(composer_freeform_client, session_id)

        gs_dict = _get_current_guided_session(composer_freeform_client, session_id)
        assert gs_dict.get("transition_consumed") is True, f"transition_consumed not set to True after recompose. GuidedSession: {gs_dict}"
        assert body["state"]["validation_errors"] == ["guided_composition_invalid"]

    def test_recompose_guided_session_persisted_in_composer_meta(self, composer_freeform_client: TestClient) -> None:
        """guided_session is included in composer_meta after recompose, not silently dropped."""
        session_id = _create_session(composer_freeform_client)

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        _seed_terminal_guided_session(composer_freeform_client, session_id, terminal)
        _seed_user_message(composer_freeform_client, session_id, "persist check")

        async def _fake_acompletion(**kwargs):
            return _fake_chat_response()

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            side_effect=_fake_acompletion,
        ):
            _recompose(composer_freeform_client, session_id)

        gs_dict = _get_current_guided_session(composer_freeform_client, session_id)
        assert gs_dict, "guided_session must be present in composer_meta after recompose"
        assert "terminal" in gs_dict, "guided_session must retain terminal after recompose"

    def test_recompose_assistant_failure_rolls_back_transition_consumption(
        self,
        composer_freeform_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recompose retries the transition prompt when its response write fails."""
        session_id = _create_session(composer_freeform_client)
        _seed_terminal_guided_session(
            composer_freeform_client,
            session_id,
            TerminalState(
                kind=TerminalKind.EXITED_TO_FREEFORM,
                reason=TerminalReason.USER_PRESSED_EXIT,
                pipeline_yaml=None,
            ),
        )
        _seed_user_message(composer_freeform_client, session_id, "retry after exit")
        captured_messages: list[list[dict]] = []

        async def _fake_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _fake_chat_response("durable recompose transition")

        _inject_one_assistant_insert_failure(composer_freeform_client, monkeypatch)
        with patch("elspeth.web.composer.service._litellm_acompletion", side_effect=_fake_acompletion):
            with pytest.raises(IntegrityError, match="injected transition assistant failure"):
                composer_freeform_client.post(f"/api/sessions/{session_id}/recompose")

            assert _get_current_guided_session(composer_freeform_client, session_id).get("transition_consumed") is False
            body = _recompose(composer_freeform_client, session_id)
        assert body["message"]["content"] == "durable recompose transition"
        assert _get_current_guided_session(composer_freeform_client, session_id).get("transition_consumed") is True
        assert len(captured_messages) == 2
        assert all("## Mode Transition" in next(m for m in turn if m["role"] == "system")["content"] for turn in captured_messages)
