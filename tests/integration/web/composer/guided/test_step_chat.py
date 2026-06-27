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
import json
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
    """Trigger initial guided turn so guided_session is attached + at step_1.

    GET /guided is non-mutating on a fresh session (commit c4e2f69cd,
    May 15 2026) — the latent step_1 turn is built in memory and returned
    but no composition_state row is allocated.  The chat dispatcher
    auto-seeds the TurnRecord on the first POST chat the same way the
    respond endpoint does, so step_chat tests that only need a step_1
    chat surface can rely on this helper as-is.

    Tests that need the persisted composition_state row to exist
    BEFORE the chat call (e.g. to overwrite ``composer_meta`` directly
    via the service layer) must use :func:`_seed_persisted_state`
    instead.
    """
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_persisted_state(client: TestClient, session_id: str) -> dict:
    """Materialise a persisted composition_state version with a guided session.

    Use this in step_chat tests that read or overwrite the persisted
    state through the service layer.  POSTs respond with ``chosen=["csv"]``
    to trigger the auto-seed + persistence path inside the route's
    compose-lock; the session ends up at step_2_blob with the step_1
    TurnRecord recorded in ``guided_session.history``.  Tests that need
    the session positioned at step_1 must use :func:`_seed_persisted_step1`
    instead — that helper writes the initial latent state straight
    through the service layer and leaves the session at step_1_source.
    """
    get_resp = client.get(f"/api/sessions/{session_id}/guided")
    assert get_resp.status_code == 200, get_resp.json()
    respond_resp = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={"chosen": ["csv"]},
    )
    assert respond_resp.status_code == 200, respond_resp.json()
    return respond_resp.json()


def _seed_persisted_step1(client: TestClient, session_id: str) -> None:
    """Persist the latent step_1 guided state via the service layer.

    Unlike :func:`_seed_persisted_state`, this does NOT advance the
    session past step_1 — it writes the same in-memory state that
    ``_initial_composition_state_with_guided_session`` produces through
    the route's lazy-create branches, allocating a real
    composition_state v1 row.  Tests that need an existing
    composition_state_id available on the audit row of a chat call that
    happens at step_1_source (e.g. the InvariantError sanitization
    coverage) must seed this way: the chat endpoint reads but does not
    write state on the failure path, so the audit row's
    composition_state_id needs to point at a pre-existing row.
    """
    from elspeth.web.composer.guided.state_machine import GuidedSession
    from elspeth.web.sessions.protocol import CompositionStateData

    initial_guided = GuidedSession.initial()
    state_data = CompositionStateData(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata_={"name": None, "description": None, "tags": ()},
        is_valid=False,
        validation_errors=None,
        composer_meta={"guided_session": initial_guided.to_dict()},
    )
    service = client.app.state.session_service
    asyncio.run(
        service.save_composition_state(
            UUID(session_id),
            state_data,
            provenance="session_seed",
        )
    )


def _fake_llm_reply(text: str) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a plain assistant message (no tool calls).

    ``tool_calls=None`` mirrors a real LiteLLM message, which always carries the
    attribute (None when the model did not call a tool). The STEP_2/STEP_3 chat
    apply branches drive their step solver first; that solver reads
    ``message.tool_calls or ()`` and classifies a None as advisory prose, so a
    faithful mock must present the attribute rather than omit it.
    """
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def _fake_source_resolution_tool_call(arguments: dict) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a Step-1 source-resolution tool call."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="resolve_source",
            arguments=json.dumps(arguments),
        )
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))])


def _post_chat(client: TestClient, session_id: str, **kwargs) -> tuple[int, dict]:
    resp = client.post(f"/api/sessions/{session_id}/guided/chat", json=kwargs)
    return resp.status_code, resp.json()


def _chat_turn_audit_bodies(client: TestClient, session_id: str) -> list[dict]:
    return [json.loads(row.content) for row in _chat_turn_audit_rows(client, session_id)]


def _chat_turn_audit_rows(client: TestClient, session_id: str) -> list:
    service = client.app.state.session_service
    messages = asyncio.run(service.get_messages(UUID(session_id)))
    return [m for m in messages if m.role == "audit" and '"_kind": "chat_turn_audit"' in m.content]


def _llm_call_audit_bodies(client: TestClient, session_id: str) -> list[dict]:
    service = client.app.state.session_service
    messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    rows: list[dict] = []
    for message in messages:
        if message.role != "audit":
            continue
        try:
            content = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(content, dict):
            continue
        if content.get("_kind") == "llm_call_audit":
            rows.append(content)
    return rows


def _fake_resolve_source_response_csv() -> SimpleNamespace:
    """Canonical zero-arg CSV source resolution stub for phase-entry and refresh tests."""
    source_content = (
        "colour,teal_fit\n"
        "White,good\n"
        "Navy blue,good\n"
        "Coral,good\n"
        "Gold,good\n"
        "Soft gray,good\n"
        "Neon green,bad\n"
        "Bright orange,bad\n"
        "Cherry red,bad\n"
        "Mud brown,bad\n"
        "Fluorescent yellow,bad\n"
    )
    return _fake_source_resolution_tool_call(
        {
            "resolution": "source",
            "plugin": "csv",
            "filename": "teal_colours.csv",
            "mime_type": "text/csv",
            "content": source_content,
            "options": {"schema": {"mode": "observed"}},
            "observed_columns": ["colour", "teal_fit"],
            "sample_rows": [
                {"colour": "White", "teal_fit": "good"},
                {"colour": "Neon green", "teal_fit": "bad"},
            ],
            "assistant_message": "I set this up as a CSV source.",
        }
    )


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
        """Slice 3 invariant: chat does not mutate guided_session.history.

        ``history`` (the wizard turn record list) is distinct from
        ``chat_history`` (the per-step chat).  Slice 5 introduces the latter
        and the chat route MUST append to ``chat_history`` only; the wizard
        ``history`` field is owned by /respond + emitters.
        """
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

    def test_appends_user_and_assistant_turns_to_chat_history(self, composer_test_client: TestClient) -> None:
        """Slice 5: a successful chat appends BOTH turns to chat_history atomically.

        chat_turn_seq advances by 2 (user.seq=0, assistant.seq=1, next=2).
        Both turns carry the same step and a server-recorded ts_iso; the
        sequence guarantees deterministic ordering even when two turns
        share a wall-clock second.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("rows look fine")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="any nulls in col_a?",
                step_index="step_1_source",
            )

        assert status == 200, body
        chat_history = body["guided_session"]["chat_history"]
        assert len(chat_history) == 2

        user_entry, assistant_entry = chat_history
        assert user_entry["role"] == "user"
        assert user_entry["content"] == "any nulls in col_a?"
        assert user_entry["seq"] == 0
        assert user_entry["step"] == "step_1_source"
        assert user_entry["ts_iso"]  # non-empty ISO string

        assert assistant_entry["role"] == "assistant"
        assert assistant_entry["content"] == "rows look fine"
        assert assistant_entry["seq"] == 1
        assert assistant_entry["step"] == "step_1_source"
        # Both turns produced in the same request share a single ts_iso —
        # ordering must come from `seq`, not the timestamp.
        assert assistant_entry["ts_iso"] == user_entry["ts_iso"]

        assert body["guided_session"]["chat_turn_seq"] == 2

    def test_chat_history_round_trips_through_persistence(self, composer_test_client: TestClient) -> None:
        """Slice 5 invariant: chat_history persists across a service reload.

        After a chat exchange, GET /guided must restore the same chat_history
        and chat_turn_seq.  This exercises GuidedSession.to_dict /
        from_dict for the new fields end-to-end through the service layer.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("ack")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
                step_index="step_1_source",
            )

        # Force a fresh load via GET /guided (reads through state_from_record).
        resp = composer_test_client.get(f"/api/sessions/{session_id}/guided")
        assert resp.status_code == 200, resp.json()
        reloaded = resp.json()["guided_session"]

        assert reloaded["chat_turn_seq"] == 2
        assert len(reloaded["chat_history"]) == 2
        assert reloaded["chat_history"][0]["content"] == "ping"
        assert reloaded["chat_history"][1]["content"] == "ack"

    def test_chat_turn_persists_audit_message(self, composer_test_client: TestClient) -> None:
        """Slice 5.1: each chat round-trip persists a ComposerChatTurn audit row.

        The audit message lands as a ``role=audit`` chat message tagged
        ``_kind=chat_turn_audit``.  Per CLAUDE.md auditability standard
        ("no inference - if it's not recorded, it didn't happen"), the
        per-turn audit record MUST persist; constructing it in memory
        and letting it GC at function return would be evidence
        tampering.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("acked")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
                step_index="step_1_source",
            )

        # The content carries the slim summary fields.
        audit_bodies = _chat_turn_audit_bodies(composer_test_client, session_id)
        assert len(audit_bodies) == 1, audit_bodies
        body = audit_bodies[0]
        assert body["_kind"] == "chat_turn_audit"
        assert body["status"] == "success"
        assert body["step"] == "step_1_source"
        assert body["initiator"] == "user"
        assert body["chat_turn_seq"] == 0
        # latency is captured but non-deterministic; assert presence only.
        assert isinstance(body["latency_ms"], int)
        assert body["latency_ms"] >= 0

    def test_successful_step_chat_persists_llm_call_audit_row(self, composer_test_client: TestClient) -> None:
        """The guided chat LLM call must persist a ComposerLLMCall sidecar.

        ``ComposerChatTurn`` proves a user-facing chat turn happened; it is not
        a substitute for the lower-level ``ComposerLLMCall`` row that records
        the outbound model request.  Chain solving already emits that row; the
        per-step chat path must do the same.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("acked")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="ping",
                step_index="step_1_source",
            )

        assert status == 200, body
        audits = _llm_call_audit_bodies(composer_test_client, session_id)
        assert len(audits) == 1, audits
        audit = audits[0]
        assert audit["status"] == "success"
        assert audit["model_requested"]

    def test_synthetic_unavailable_persists_audit_message(self, composer_test_client: TestClient) -> None:
        """Slice 5.1: synthetic-unavailable chat round-trips also persist audit.

        The SYNTHETIC_UNAVAILABLE status is the audit-discriminator that
        tells an auditor "this LLM call failed and the user got the
        unavailable fallback."  It must reach the audit table, otherwise
        the operator cannot distinguish a real LLM reply from a synthetic
        one after the fact.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=TimeoutError("upstream timeout")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
                step_index="step_1_source",
            )

        audit_bodies = _chat_turn_audit_bodies(composer_test_client, session_id)
        assert len(audit_bodies) == 1
        body = audit_bodies[0]
        assert body["status"] == "synthetic_unavailable"
        assert body["error_class"] == "TimeoutError"
        assert body["chat_turn_seq"] == 0
        llm_audits = _llm_call_audit_bodies(composer_test_client, session_id)
        assert len(llm_audits) == 1, llm_audits
        assert llm_audits[0]["status"] == "timeout"

    def test_two_chat_turns_advance_seq_monotonically(self, composer_test_client: TestClient) -> None:
        """Slice 5: chat_turn_seq is monotonic across multiple chat turns in the same step."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("first reply")),
        ):
            _post_chat(composer_test_client, session_id, message="first", step_index="step_1_source")
        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("second reply")),
        ):
            _, body = _post_chat(
                composer_test_client,
                session_id,
                message="second",
                step_index="step_1_source",
            )

        # After two chats: seq advances 0,1,2,3 then next=4.
        chat_history = body["guided_session"]["chat_history"]
        assert [entry["seq"] for entry in chat_history] == [0, 1, 2, 3]
        assert [entry["content"] for entry in chat_history] == ["first", "first reply", "second", "second reply"]
        assert body["guided_session"]["chat_turn_seq"] == 4


class TestStep1SourceResolution:
    def _drive_to_step_1_schema_form(self, client: TestClient, session_id: str) -> None:
        _seed_guided_session(client, session_id)
        resp = client.post(f"/api/sessions/{session_id}/guided/respond", json={"chosen": ["csv"]})
        assert resp.status_code == 200, resp.json()
        assert resp.json()["next_turn"]["type"] == "schema_form"

    def test_step_1_chat_commits_source_in_place_and_rerenders_form(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """A complete Step-1 source request must not be answered as prose-only chat.

        Regression for staging session 221110b7-e709-48cf-99cd-b22e2c9e1f5e:
        after the operator selected ``csv``, a specific data request in the
        guided chat box received a generic color-list answer and left
        ``CompositionState.source`` null.  The Step-1 chat path should instead
        resolve the source, persist it through the normal Step-1 handler, and
        re-render the Step-1 form in place so the user can revise or advance
        explicitly.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_resolve_source_response_csv()),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="can we make it just a list of ten random colours, 5 that go well with teal and 5 that go badly with teal",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I set this up as a CSV source."
        # Apply-in-place: the phase stays STEP_1 (revise model), and the form
        # re-renders populated from the committed source.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["step_index"] == 0
        # The re-rendered form is POPULATED from the committed source (the whole
        # point of LLM-primary: the form shows what was just built).
        assert body["next_turn"]["payload"]["plugin"] == "csv"
        assert body["next_turn"]["payload"]["prefilled"]["path"].endswith("_teal_colours.csv")
        assert body["composition_state"]["sources"]["source"]["plugin"] == "csv"
        source_options = body["composition_state"]["sources"]["source"]["options"]
        assert source_options["schema"]["mode"] == "observed"
        assert source_options["path"].endswith("_teal_colours.csv")
        audits = _llm_call_audit_bodies(composer_test_client, session_id)
        assert len(audits) == 1, audits
        assert audits[0]["status"] == "success"

    def test_step_1_chat_drives_source_from_phase_entry(self, composer_test_client) -> None:
        """A chat submit at STEP_1 entry (no manual plugin pick) drives the source."""
        client = composer_test_client
        session_id = _create_session(client)
        # Seed a persisted composition_state so GET /guided can record the
        # initial SINGLE_SELECT TurnRecord (GET only records when a
        # state_record already exists — non-mutating on a truly fresh session).
        _seed_persisted_step1(client, session_id)
        # GET records the initial SINGLE_SELECT turn; NO _respond plugin pick.
        entry = _seed_guided_session(client, session_id)
        assert entry["guided_session"]["step"] == "step_1_source"
        assert entry["next_turn"]["type"] == "single_select"
        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_resolve_source_response_csv()),
        ):
            status, body = _post_chat(
                client,
                session_id,
                message="make a csv source with a text column",
                step_index="step_1_source",
            )
        assert status == 200, body
        # Drove the phase from entry: committed in place, stayed STEP_1, populated form.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["step_index"] == 0
        assert body["next_turn"]["payload"]["plugin"] == "csv"
        assert body["composition_state"]["sources"]["source"]["plugin"] == "csv"
        # M6: the SINGLE_SELECT entry record, resolved via chat, carries a DISPLAY
        # summary so "Decisions so far" reads "Configured: csv" instead of a bare
        # "Decided". response_hash stays None — the chat IS the answer (recorded as
        # a ChatTurn), not a widget response, so we do not fabricate one.
        step1_single_selects = [
            r for r in body["guided_session"]["history"] if r["step"] == "step_1_source" and r["turn_type"] == "single_select"
        ]
        assert step1_single_selects, body["guided_session"]["history"]
        assert step1_single_selects[-1]["summary"] == "Configured: csv"
        assert step1_single_selects[-1]["response_hash"] is None

    def test_step_1_chat_advisory_prose_at_phase_entry_does_not_mutate_source(self, composer_test_client) -> None:
        """Prose-only chat at STEP_1 phase entry (SINGLE_SELECT) stays advisory.

        Hardening for the guard-widening in Task 3 (2a): SINGLE_SELECT is now
        a valid entry type for ``resolve_step_1_source_chat_with_auto_drop``.
        When the resolver returns no tool call (``source_resolution is None``)
        the route MUST remain advisory:
          (a) ``next_turn`` is None — no re-render
          (b) ``guided.step`` is unchanged (still ``step_1_source``)
          (c) no source committed (``step_1_result`` unmutated)
          (d) the reply lands in ``chat_history``, NOT wizard ``history``

        The pin catches a regression where ``if source_resolution is not None:``
        is removed: the route would then crash at ``source_resolution.filename``
        with AttributeError → 500, failing assertion (a).
        """
        client = composer_test_client
        session_id = _create_session(client)
        # Phase-entry precondition: same setup as test_step_1_chat_drives_source_from_phase_entry.
        _seed_persisted_step1(client, session_id)
        entry = _seed_guided_session(client, session_id)
        history_before = entry["guided_session"]["history"]
        assert len(history_before) == 1, "expected exactly 1 TurnRecord after seeded GET"

        # _fake_llm_reply has no tool_calls attribute; maybe_resolve_step_1_source_chat
        # hits AttributeError on message.tool_calls, which resolve_step_1_source_chat_with_auto_drop
        # catches and maps to the synthetic-unavailable fallback
        # (source_resolution=None, fallback_chat=StepChatResult(synthetic_message)).
        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("Try using the wizard plugin picker.")),
        ):
            status, body = _post_chat(
                client,
                session_id,
                message="what should I do?",
                step_index="step_1_source",
            )
        assert status == 200, body
        # (a) advisory: no re-render
        assert body["next_turn"] is None
        # (b) step unchanged
        assert body["guided_session"]["step"] == "step_1_source"
        # (c) no source committed
        source_slot = (body["composition_state"]["sources"] or {}).get("source")
        assert source_slot is None, f"expected no committed source, got {source_slot!r}"
        # (d) reply in chat_history (user+assistant), wizard history unchanged
        assert len(body["guided_session"]["chat_history"]) == 2
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
        _seed_persisted_step1(composer_test_client, session_id)

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
        asyncio.run(service.save_composition_state(UUID(session_id), new_data, provenance="session_seed"))

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
        _seed_persisted_step1(composer_test_client, session_id)

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
        asyncio.run(service.save_composition_state(UUID(session_id), new_data, provenance="session_seed"))

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
        # Wizard `history` is not mutated by the failed chat; the chat
        # round-trip lives in `chat_history` only.
        assert body["guided_session"]["history"] == seeded["guided_session"]["history"]
        # Slice 5: even on synthetic-message, the chat IS appended to
        # chat_history.  The auditor must be able to see "this user typed
        # X and got the synthetic reply" — that's the audit value of
        # SYNTHETIC_UNAVAILABLE in ComposerChatTurnStatus.  ChatTurnStatus
        # is the discriminator; on the wire (Phase A) the user sees the
        # synthetic content directly.
        chat_history = body["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[0]["content"] == "anything"
        assert chat_history[1]["content"] == "I'm unavailable right now; you can still use the wizard controls."

    def test_litellm_budget_exceeded_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """BudgetExceededError from the LLM seam → 200 with the synthetic unavailable message.

        Regression for elspeth-4cec1a03b9: ``BudgetExceededError`` is a direct
        ``Exception`` subclass (NOT an ``APIError`` descendant), so the
        wrapper's old narrow catch tuple let it escape into the route as an
        unhandled 500. It is an operational provider-budget failure that must
        be absorbed into the synthetic-unavailable contract — parity with the
        non-guided ``_explain_run_diagnostics`` absorb set (sibling ab3ad30e87).
        """
        from litellm.exceptions import BudgetExceededError

        session_id = _create_session(composer_test_client)
        seeded = _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=BudgetExceededError(current_cost=10.0, max_budget=5.0)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None
        assert body["guided_session"]["history"] == seeded["guided_session"]["history"]
        chat_history = body["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[0]["content"] == "anything"
        assert chat_history[1]["content"] == "I'm unavailable right now; you can still use the wizard controls."

    def test_litellm_guardrail_raised_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """GuardrailRaisedException from the LLM seam → 200 with the synthetic message.

        Companion to the budget case (elspeth-4cec1a03b9): the content-policy
        guardrail failure is likewise a direct ``Exception`` subclass that
        bypassed the old catch tuple. Absorbed into the synthetic-unavailable
        contract; the session is not terminated.
        """
        from litellm.exceptions import GuardrailRaisedException

        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=GuardrailRaisedException(guardrail_name="pii", message="blocked")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_litellm_blocked_pii_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """BlockedPiiEntityError from the LLM seam → 200 with the synthetic message.

        Third member of the elspeth-4cec1a03b9 absorb set (parity with
        ab3ad30e87): a direct ``Exception`` subclass that bypassed the old
        catch tuple. Absorbed into the synthetic-unavailable contract.
        """
        from litellm.exceptions import BlockedPiiEntityError

        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=BlockedPiiEntityError(entity_type="email", guardrail_name="pii")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_litellm_guardrail_intervention_normal_string_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """GuardrailInterventionNormalStringError from the LLM seam → 200 with the synthetic message.

        Fourth member of the elspeth-4cec1a03b9 named absorb set. The ticket
        enumerates this class explicitly; its MRO is a direct ``Exception``
        subclass (NOT a subclass of ``GuardrailRaisedException`` nor
        ``APIError``), so before the catch-tuple widening it escaped
        ``solve_step_chat_with_auto_drop`` to an unhandled 500 instead of the
        synthetic-unavailable audit contract.
        """
        from litellm.exceptions import GuardrailInterventionNormalStringError

        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(side_effect=GuardrailInterventionNormalStringError(message="intervention")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

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


class TestStepChatServerInvariants:
    """Offensive-programming paths — defective LLM responses we cannot recover from.

    Distinct from :class:`TestStepChatTransientFailure`: transient failures
    (network, auth, malformed-response-shape) are absorbed by
    ``solve_step_chat_with_auto_drop`` into the synthetic unavailable
    message and surface as HTTP 200.  Server-invariant violations
    (empty / whitespace content from an otherwise well-shaped LLM
    response) are NOT absorbed — they raise :class:`InvariantError` from
    ``solve_step_chat``, which the route converts to a B1-sanitized HTTP
    500 with a structured slog event carrying only ``exc_class`` and
    safe frame strings (no ``str(exc)`` — the InvariantError message
    embeds the model name + step value).

    Mirror of the ``post_guided_respond`` InvariantError handling at
    ``routes.py``'s ``step_advance`` call site.
    """

    _STATIC_DETAIL = "Server invariant violated. See application audit log for diagnostic detail."

    def test_empty_content_returns_sanitized_500(self, composer_test_client: TestClient) -> None:
        """Empty content string → InvariantError → 500 with B1-sanitized detail + slog."""
        from structlog.testing import capture_logs

        session_id = _create_session(composer_test_client)
        # Pre-persist step_1 state so the audit row written on the failure
        # path can reference a real composition_state_id (the chat
        # InvariantError handler reads state but does not write — without a
        # prior persistence the audit row's composition_state_id would be
        # null, hiding the failure from per-state audit replays).
        _seed_persisted_step1(composer_test_client, session_id)

        with (
            patch(
                "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
                new=AsyncMock(return_value=_fake_llm_reply("")),
            ),
            capture_logs() as cap_logs,
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 500, body
        assert body["detail"] == self._STATIC_DETAIL
        # Locate the slog event from the route's InvariantError handler.
        # Asserting site="solve_step_chat" pins this to the chat path
        # (the post_guided_respond handler emits site="step_advance"
        # from an identically named "guided.invariant_violated" event).
        invariant_events = [e for e in cap_logs if e.get("event") == "guided.invariant_violated" and e.get("site") == "solve_step_chat"]
        assert len(invariant_events) == 1, f"expected one guided.invariant_violated event from chat path, got: {cap_logs!r}"
        event = invariant_events[0]
        assert event["exc_class"] == "InvariantError"
        # Frames are tuple-shaped and start with the safe-frame format.
        # No source lines, no locals reprs, no exception message — only
        # the path:line:func triple (B1 convention).
        assert isinstance(event["frames"], tuple) and len(event["frames"]) > 0
        assert all(f.startswith("frame=") for f in event["frames"])
        audit_bodies = _chat_turn_audit_bodies(composer_test_client, session_id)
        assert len(audit_bodies) == 1
        audit_body = audit_bodies[0]
        assert audit_body["status"] == "invariant_violated"
        assert audit_body["initiator"] == "user"
        assert audit_body["step"] == "step_1_source"
        assert audit_body["error_class"] == "InvariantError"

        reloaded = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["guided_session"]
        assert reloaded["chat_history"] == []
        assert reloaded["chat_turn_seq"] == 0
        audit_rows = _chat_turn_audit_rows(composer_test_client, session_id)
        assert audit_rows[0].composition_state_id is not None

    def test_chat_audit_unwind_preserves_primary_error_if_slog_raises(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """A logger failure in the final audit-drain fallback must not replace the primary chat error.

        The route is already unwinding a sanitized HTTPException from the
        chat solver. If the final ``slog.error`` fallback also raises,
        Python would otherwise let that secondary logger exception replace
        the in-flight HTTPException from the ``finally`` block.
        """
        from elspeth.web.sessions.routes.composer import guided as composer_module

        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)

        async def _raising_persist_chat_turns(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated chat-audit persist failure")

        def _raising_only_for_unwind_log(event: str, *args: object, **kwargs: object) -> None:
            if event == "guided.chat_turn_persist_failed_during_exception_handling":
                raise RuntimeError("simulated logger failure")

        monkeypatch.setattr(composer_module, "_persist_chat_turns", _raising_persist_chat_turns)
        monkeypatch.setattr(composer_module.slog, "error", _raising_only_for_unwind_log)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 500, body
        assert body["detail"] == self._STATIC_DETAIL

    def test_whitespace_only_content_returns_sanitized_500(self, composer_test_client: TestClient) -> None:
        """Whitespace-only content → same path as empty content (``.strip()`` is empty)."""
        session_id = _create_session(composer_test_client)
        # See the sibling empty-content test for why pre-persisting is
        # required on the InvariantError failure path.
        _seed_persisted_step1(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("   \n\t  ")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
                step_index="step_1_source",
            )

        assert status == 500, body
        assert body["detail"] == "Server invariant violated. See application audit log for diagnostic detail."
        audit_bodies = _chat_turn_audit_bodies(composer_test_client, session_id)
        assert len(audit_bodies) == 1
        audit_body = audit_bodies[0]
        assert audit_body["status"] == "invariant_violated"
        assert audit_body["initiator"] == "user"
        assert audit_body["step"] == "step_1_source"
        assert audit_body["error_class"] == "InvariantError"

        reloaded = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["guided_session"]
        assert reloaded["chat_history"] == []
        assert reloaded["chat_turn_seq"] == 0
        audit_rows = _chat_turn_audit_rows(composer_test_client, session_id)
        assert audit_rows[0].composition_state_id is not None


class TestStepChatCrossStep:
    """chat_history persists across step transitions (Phase A integration coverage).

    The slice 5 round-trip test exercises persistence within a single
    step; this class exercises the cross-step invariant: a session that
    chats at step_1, advances to step_2, then chats at step_2 must
    accumulate four chat_history entries spanning two step values, in
    seq order.  Documented gap from elspeth-obs-791077020c.

    Step advance is via the existing ``/api/sessions/{id}/guided/respond``
    seam.  The pattern mirrors helpers in ``test_audit_emission.py``
    (``_seed_blob`` + two ``_respond`` calls drive step_1_source →
    step_2_sink): seed a CSV blob, accept the CSV source plugin, then
    submit an edited_values payload that closes the INSPECT_AND_CONFIRM
    turn and advances to step_2_sink SINGLE_SELECT.
    """

    @staticmethod
    def _seed_csv_blob(client: TestClient, session_id: str) -> str:
        """Seed an inline CSV blob and return its storage_path on disk."""
        content = "text,note\nHello,world\n"
        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
        )
        assert resp.status_code == 201, resp.json()
        blob_id = resp.json()["id"]
        blob_service = client.app.state.blob_service
        record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
        return record.storage_path

    @staticmethod
    def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
        resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
        assert resp.status_code == 200, resp.json()
        return resp.json()

    def test_chat_history_accumulates_across_step_transition(self, composer_test_client: TestClient) -> None:
        """Chat at step_1, advance to step_2, chat at step_2, GET /guided → 4 entries with mixed step values."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)
        storage_path = self._seed_csv_blob(composer_test_client, session_id)

        # 1. Chat at step_1_source.  This records two ChatTurn entries
        #    (user + assistant, seq 0 + 1, step=step_1_source).
        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("step-1 advice")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="how do I describe my CSV?",
                step_index="step_1_source",
            )

        # 2. Advance through step_1 SINGLE_SELECT → INSPECT_AND_CONFIRM.
        #    The wizard remains at step_1_source for the INSPECT turn.
        self._respond(composer_test_client, session_id, chosen=["csv"])

        # 3. Commit the source config — this transitions the wizard to
        #    step_2_sink SINGLE_SELECT.  The chat_history persisted in
        #    composer_meta carries through the to_dict / from_dict round
        #    trip on the GuidedSession update.
        self._respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text", "note"],
                "sample_rows": [{"text": "Hello", "note": "world"}],
            },
        )

        # Confirm the wizard advanced — load-bearing precondition for
        # the cross-step assertion below.  Without this assertion, a
        # silent revert to step_1_source would mask the test as passing
        # while never exercising the cross-step transition.
        guided_after_advance = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()
        assert guided_after_advance["guided_session"]["step"] == "step_2_sink", guided_after_advance

        # 4. Chat at step_2_sink.  Two more ChatTurn entries (user +
        #    assistant, seq 2 + 3, step=step_2_sink).
        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=AsyncMock(return_value=_fake_llm_reply("step-2 advice")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="which sink for JSON output?",
                step_index="step_2_sink",
            )

        # 5. GET /guided and assert chat_history has 4 entries with the
        #    expected step+seq+role+content pattern.  This is the
        #    cross-step invariant: a single chat_history list, monotonic
        #    seq, step values reflecting where each turn happened.
        final = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()
        chat_history = final["guided_session"]["chat_history"]
        assert len(chat_history) == 4, chat_history
        # Tuple-shaped assertion so a regression on any axis (step/seq/role/content)
        # surfaces in the failure message.
        observed = [(entry["role"], entry["step"], entry["seq"], entry["content"]) for entry in chat_history]
        assert observed == [
            ("user", "step_1_source", 0, "how do I describe my CSV?"),
            ("assistant", "step_1_source", 1, "step-1 advice"),
            ("user", "step_2_sink", 2, "which sink for JSON output?"),
            ("assistant", "step_2_sink", 3, "step-2 advice"),
        ]
        assert final["guided_session"]["chat_turn_seq"] == 4
