"""Integration tests for POST /api/sessions/{id}/guided/chat (Phase A slice 3).

Verifies the end-to-end contract of the per-step chat endpoint:

- Success path: a fresh session at step_1 with a mocked LLM returns the
  assistant message and echoes the unchanged guided_session.
- Step mismatch: client sends a step_index that doesn't match the
  session's current step → 409 (the wizard advanced under the user).
- Unknown step_index value: malformed enum string → 400.
- Terminal session: chat against a session in a terminal state → 409.
- No guided_session attached: 400 with the "use /messages" guidance.
- Pydantic boundary: empty / oversize message → 422 (the route never
  reaches solve_step_chat with an invalid message).
- Transient LLM failure: LiteLLM timeout returns 200 with the synthetic
  unavailable message; the session is not terminated.

HTTP transport: SyncASGITestClient (in-process, synchronous — same
pattern as the other guided integration tests). Patch target convention:
``elspeth.web.composer.guided.chat_solver._litellm_acompletion`` —
mirrors the chain-solver test convention (see test_auto_drop.py).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from elspeth.web.composer.guided.state_machine import TerminalReason, TerminalState
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "step-chat-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _seed_guided_session(client: TestClient, session_id: str) -> dict:
    """Trigger initial guided turn so guided_session is attached + at step_1."""
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _fake_llm_reply(text: str) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a plain assistant message (no tool calls)."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _post_chat(client: TestClient, session_id: str, **kwargs) -> tuple[int, dict]:
    resp = client.post(f"/api/sessions/{session_id}/guided/chat", json=kwargs)
    return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStepChatSuccess:
    def test_returns_assistant_message_on_step_1(self, composer_test_client: TestClient) -> None:
        """Happy path: LLM reply round-trips through the route as assistant_message."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("CSV columns are typically detected from the header row.")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="what columns are in this CSV?",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "CSV columns are typically detected from the header row."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_echoes_unchanged_guided_session(self, composer_test_client: TestClient) -> None:
        """Slice 3 invariant: chat does not mutate guided_session.history."""
        session_id = _create_session(composer_test_client)
        seeded = _seed_guided_session(composer_test_client, session_id)
        history_before = seeded["guided_session"]["history"]

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("ack")),
        ):
            _, body = _post_chat(
                composer_test_client,
                session_id,
                message="hello",
                step_index="step_1_source",
            )

        assert body["guided_session"]["history"] == history_before


class TestStepChatRejections:
    def test_step_mismatch_returns_409(self, composer_test_client: TestClient) -> None:
        """step_index != session.step → wizard advanced under the user → 409."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        # No LLM patch — request must fail before the solver runs.
        status, body = _post_chat(
            composer_test_client,
            session_id,
            message="hi",
            step_index="step_3_transforms",
        )

        assert status == 409, body
        assert "step_3_transforms" in body["detail"]
        assert "step_1_source" in body["detail"]

    def test_unknown_step_index_returns_400(self, composer_test_client: TestClient) -> None:
        """Stale client sends a value not in the GuidedStep enum → 400 with valid list."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        status, body = _post_chat(
            composer_test_client,
            session_id,
            message="hi",
            step_index="step_42_nope",
        )

        assert status == 400, body
        assert "step_42_nope" in body["detail"]
        assert "step_1_source" in body["detail"]  # valid options listed

    def test_no_guided_session_returns_400(self, composer_test_client: TestClient) -> None:
        """Session with no guided_session attached → 400 with /messages hint.

        We force this by directly clearing the guided_session field on the
        persisted state. (All fresh sessions default to guided per spec §5.2,
        so this is the only way to exercise the rejection path.)
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        # Strip guided_session from the persisted composition_state via the
        # service layer — mirrors the path a freeform-only session would take.
        service = composer_test_client.app.state.session_service
        record = asyncio.run(service.get_current_state(UUID(session_id)))
        assert record is not None
        existing_meta = dict(record.composer_meta or {})
        existing_meta.pop("guided_session", None)
        from elspeth.web.sessions.protocol import CompositionStateData

        new_data = CompositionStateData(
            source=record.source,
            nodes=record.nodes,
            edges=record.edges,
            outputs=record.outputs,
            metadata_=record.metadata_,
            is_valid=record.is_valid,
            validation_errors=record.validation_errors,
            composer_meta=existing_meta,
        )
        asyncio.run(service.save_composition_state(UUID(session_id), new_data))

        status, body = _post_chat(
            composer_test_client,
            session_id,
            message="hi",
            step_index="step_1_source",
        )

        assert status == 400, body
        assert "/api/sessions/{id}/messages" in body["detail"]

    def test_terminal_session_returns_409(self, composer_test_client: TestClient) -> None:
        """Chat against a session in a terminal state → 409.

        We force a terminal state by directly persisting it; mirrors the path
        a completed or exited-to-freeform session would take.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        # Drop the session to terminal via the converter round-trip.
        service = composer_test_client.app.state.session_service
        record = asyncio.run(service.get_current_state(UUID(session_id)))
        assert record is not None
        from elspeth.web.sessions.converters import state_from_record

        state = state_from_record(record)
        assert state.guided_session is not None
        terminal_guided = replace(
            state.guided_session,
            terminal=TerminalState(
                kind=state.guided_session.terminal.kind if state.guided_session.terminal else _terminal_kind_exited(),
                reason=TerminalReason.USER_PRESSED_EXIT,
                pipeline_yaml=None,
            ),
        )
        from elspeth.web.sessions.protocol import CompositionStateData

        existing_meta = dict(record.composer_meta or {})
        existing_meta["guided_session"] = terminal_guided.to_dict()
        new_data = CompositionStateData(
            source=record.source,
            nodes=record.nodes,
            edges=record.edges,
            outputs=record.outputs,
            metadata_=record.metadata_,
            is_valid=record.is_valid,
            validation_errors=record.validation_errors,
            composer_meta=existing_meta,
        )
        asyncio.run(service.save_composition_state(UUID(session_id), new_data))

        status, body = _post_chat(
            composer_test_client,
            session_id,
            message="hi",
            step_index="step_1_source",
        )

        assert status == 409, body
        assert "terminal" in body["detail"].lower()


def _terminal_kind_exited():
    """Return the TerminalKind value matching exited_to_freeform."""
    from elspeth.web.composer.guided.state_machine import TerminalKind

    return TerminalKind.EXITED_TO_FREEFORM


class TestStepChatBoundary:
    def test_empty_message_returns_422(self, composer_test_client: TestClient) -> None:
        """min_length=1 + visible-content validator → 422."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        status, _ = _post_chat(
            composer_test_client,
            session_id,
            message="",
            step_index="step_1_source",
        )
        assert status == 422

    def test_whitespace_only_message_returns_422(self, composer_test_client: TestClient) -> None:
        """has_visible_content rejects whitespace-only strings → 422."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        status, _ = _post_chat(
            composer_test_client,
            session_id,
            message="   \t\n",
            step_index="step_1_source",
        )
        assert status == 422

    def test_oversize_message_returns_422(self, composer_test_client: TestClient) -> None:
        """max_length=4096 → Pydantic 422 before reaching the solver."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        oversize = "a" * 4097
        status, _ = _post_chat(
            composer_test_client,
            session_id,
            message=oversize,
            step_index="step_1_source",
        )
        assert status == 422


class TestStepChatTransientFailure:
    def test_litellm_timeout_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """TimeoutError from the LLM seam → 200 with the synthetic unavailable message.

        The session must NOT be terminated — chat is a non-load-bearing
        helper, unlike the chain solver's auto-drop which marks the
        session ``solver_exhausted``.
        """
        session_id = _create_session(composer_test_client)
        seeded = _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=TimeoutError("upstream LLM timed out")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        # Synthetic message wording — must match _SYNTHETIC_UNAVAILABLE_MESSAGE.
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        # Session is unchanged: still at step_1, no terminal.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None
        # History is not mutated by the failed chat.
        assert body["guided_session"]["history"] == seeded["guided_session"]["history"]

    def test_malformed_litellm_response_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """Empty choices list (IndexError in solve_step_chat) → synthetic message."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=SimpleNamespace(choices=[])),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
