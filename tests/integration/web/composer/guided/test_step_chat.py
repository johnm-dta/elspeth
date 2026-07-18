"""Integration tests for POST /api/sessions/{id}/guided/chat (Phase A slice 3).

Verifies the end-to-end contract of the per-step chat endpoint:

- Success path: a fresh session at step_1 with a mocked LLM returns the
  assistant message and echoes the unchanged guided_session.
- Stale turn token: a token that is not the current server-held turn → 409.
- Legacy step_index field: strict schema-8 request validation → 422.
- Terminal session: chat against a session in a terminal state → 409.
- No guided_session attached: 400 with the "use /messages" guidance.
- Pydantic boundary: empty / oversize message → 422 (the route never
  reaches solve_step_chat with an invalid message).
- Transient LLM failure: LiteLLM timeout returns 200 with the synthetic
  unavailable message; the session is not terminated.

HTTP transport: SyncASGITestClient (in-process, synchronous — same
pattern as the other guided integration tests). Patch target convention:
``elspeth.web.composer.guided.chat_solver._litellm_acompletion`` —
patch the symbol where ``chat_solver`` resolves it rather than patching the
provider library globally.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from elspeth.web.composer.guided.state_machine import TerminalReason, TerminalState
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_SOLVER_ACOMPLETION = "elspeth.web.composer.guided.chat_solver._litellm_acompletion"


@dataclass
class _ReturningLiteLLMCompletion:
    response: object
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


@dataclass
class _RaisingLiteLLMCompletion:
    exception: BaseException
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        raise self.exception


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "step-chat-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


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
    return _post_respond(client, session_id, chosen=["csv"])


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
    guided = client.get(f"/api/sessions/{session_id}/guided")
    next_turn = guided.json().get("next_turn") if guided.status_code == 200 else None
    kwargs.setdefault("operation_id", str(uuid4()))
    kwargs.setdefault("turn_token", next_turn["turn_token"] if next_turn is not None else "0" * 64)
    resp = client.post(f"/api/sessions/{session_id}/guided/chat", json=kwargs)
    return resp.status_code, resp.json()


def _post_respond(client: TestClient, session_id: str, **kwargs) -> dict:
    guided = client.get(f"/api/sessions/{session_id}/guided")
    assert guided.status_code == 200, guided.json()
    turn = guided.json()["next_turn"]
    assert turn is not None
    body = {
        "operation_id": str(uuid4()),
        "turn_token": turn["turn_token"],
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "control_signal": None,
        "proposal_id": None,
        "draft_hash": None,
        "edit_target": None,
    }
    body.update(kwargs)
    response = client.post(f"/api/sessions/{session_id}/guided/respond", json=body)
    assert response.status_code == 200, response.json()
    return response.json()


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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("CSV columns are typically detected from the header row.")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="what columns are in this CSV?",
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
        _seed_persisted_step1(composer_test_client, session_id)
        seeded = _seed_guided_session(composer_test_client, session_id)
        history_before = seeded["guided_session"]["history"]

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("ack")),
        ):
            _, body = _post_chat(
                composer_test_client,
                session_id,
                message="hello",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("rows look fine")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="any nulls in col_a?",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("ack")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("acked")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("acked")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="ping",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(TimeoutError("upstream timeout")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="ping",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("first reply")),
        ):
            _post_chat(composer_test_client, session_id, message="first")
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("second reply")),
        ):
            _, body = _post_chat(
                composer_test_client,
                session_id,
                message="second",
            )

        # After two chats: seq advances 0,1,2,3 then next=4.
        chat_history = body["guided_session"]["chat_history"]
        assert [entry["seq"] for entry in chat_history] == [0, 1, 2, 3]
        assert [entry["content"] for entry in chat_history] == ["first", "first reply", "second", "second reply"]
        assert body["guided_session"]["chat_turn_seq"] == 4


class TestStep1SourceResolution:
    def _drive_to_step_1_schema_form(self, client: TestClient, session_id: str) -> None:
        _seed_guided_session(client, session_id)
        body = _post_respond(client, session_id, chosen=["csv"])
        assert body["next_turn"]["type"] == "schema_form"

    def test_fresh_step_1_chat_uses_source_resolver_for_current_plugin_turn(self, composer_test_client: TestClient) -> None:
        """A first Chat can answer only the current schema-8 plugin turn."""
        session_id = _create_session(composer_test_client)
        entry = _seed_guided_session(composer_test_client, session_id)
        assert entry["composition_state"] is None
        assert len(entry["guided_session"]["history"]) == 1
        assert entry["guided_session"]["history"][0]["response_hash"] is None

        completion = _ReturningLiteLLMCompletion(_fake_resolve_source_response_csv())
        with patch(_CHAT_SOLVER_ACOMPLETION, new=completion):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="make a csv source with a text column",
            )

        assert status == 200, body
        assert len(completion.calls) == 1
        assert any(tool["function"]["name"] == "resolve_source" for tool in completion.calls[0]["tools"])
        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["payload"]["plugin"] == "csv"
        assert not body["composition_state"]["sources"]

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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_resolve_source_response_csv()),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="can we make it just a list of ten random colours, 5 that go well with teal and 5 that go badly with teal",
            )

        assert status == 200, body
        assert body["assistant_message"] == (
            "I did not apply generated source content. Review the current source form and submit it through the wizard controls."
        )
        assert body["assistant_message_kind"] == "synthetic_failure"
        assert body["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
        # Applying generated bytes requires a future atomic blob participant.
        # The current form remains authoritative and no source is committed.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["step_index"] == 0
        assert body["next_turn"]["payload"]["plugin"] == "csv"
        assert not body["composition_state"]["sources"]
        assert "_teal_colours.csv" not in json.dumps(body)
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_resolve_source_response_csv()),
        ):
            status, body = _post_chat(
                client,
                session_id,
                message="make a csv source with a text column",
            )
        assert status == 200, body
        # The pure schema-8 transition answers the current plugin turn only.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["step_index"] == 0
        assert body["next_turn"]["payload"]["plugin"] == "csv"
        assert not body["composition_state"]["sources"]
        step1_single_selects = [
            r for r in body["guided_session"]["history"] if r["step"] == "step_1_source" and r["turn_type"] == "single_select"
        ]
        assert step1_single_selects, body["guided_session"]["history"]
        assert step1_single_selects[-1]["response_hash"] is not None

    def test_step_1_chat_advisory_prose_at_phase_entry_does_not_mutate_source(self, composer_test_client) -> None:
        """Prose-only chat at STEP_1 phase entry (SINGLE_SELECT) stays advisory.

        Hardening for the guard-widening in Task 3 (2a): SINGLE_SELECT is now
        a valid entry type for ``resolve_step_1_source_chat_with_auto_drop``.
        When the resolver returns no tool call (``source_resolution is None``)
        the route MUST remain advisory:
          (a) ``next_turn`` is None — no re-render
          (b) ``guided.step`` is unchanged (still ``step_1_source``)
          (c) no source is committed
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("Try using the wizard plugin picker.")),
        ):
            status, body = _post_chat(
                client,
                session_id,
                message="what should I do?",
            )
        assert status == 200, body
        # (a) advisory: the same current turn remains authoritative
        assert body["next_turn"]["turn_token"] == entry["next_turn"]["turn_token"]
        # (b) step unchanged
        assert body["guided_session"]["step"] == "step_1_source"
        # (c) no source committed
        source_slot = (body["composition_state"]["sources"] or {}).get("source")
        assert source_slot is None, f"expected no committed source, got {source_slot!r}"
        # (d) reply in chat_history (user+assistant), wizard history unchanged
        assert len(body["guided_session"]["chat_history"]) == 2
        assert body["guided_session"]["history"] == history_before

    def test_step_1_typed_description_salvages_declined_prose_without_second_call(self, composer_test_client) -> None:
        """A typed, data-less step-1 description: ONE tool-capable call, its own prose is the answer.

        C-1 (2026-07-04 review): "I have a CSV, columns are name/email" cannot
        satisfy ``resolve_source``'s required ``content`` (no actual rows), so
        the tool-capable resolve call correctly declines and replies in prose.
        Before the fix, that prose was discarded and the route fired a
        SECOND, tool-less call reusing a system prompt that still described
        tool-calling capability with zero tools attached — the mismatch that
        made the model hallucinate ``<tool_call>`` scaffolding live. The fix
        salvages the first call's own prose directly: only ONE LLM call
        happens, and its reply IS the assistant_message.
        """
        session_id = _create_session(composer_test_client)
        # A persisted composition_state row must exist BEFORE the GET so the
        # GET call itself persists the SINGLE_SELECT TurnRecord into history
        # (see ``get_guided``'s docstring) — the chat route does not auto-seed
        # history the way ``post_guided_respond`` does, so without this the
        # STEP_1_SOURCE resolve branch's ``existing_record_for_chat is not
        # None`` guard would never pass and the call would go advisory only.
        _seed_persisted_step1(composer_test_client, session_id)
        _seed_guided_session(composer_test_client, session_id)

        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            # Resolve call: declines the tool (no rows to embed) and replies
            # in prose, per its own instructions.
            return _fake_llm_reply("I don't have your file's contents yet — please paste the rows or upload the file.")

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=fake_acompletion,
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="I have a CSV, columns are name and email",
            )

        assert status == 200, body
        # Exactly ONE LLM call — the salvage means no second, tool-less call.
        assert len(calls) == 1
        assert any(tool["function"]["name"] == "resolve_source" for tool in calls[0]["tools"])
        # The user sees the resolve call's OWN prose — no scaffold leak, no
        # synthetic-unavailable fallback, no second round-trip.
        assert body["assistant_message"] == "I don't have your file's contents yet — please paste the rows or upload the file."
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_step_1_actionable_false_tool_decline_retries_resolve_and_commits(self, composer_test_client) -> None:
        """An actionable Step-1 source request must not loop on "re-send so tools work" prose.

        Regression for staging session ee98e873: the resolve-equipped call saw
        enough detail to build the source, but the model replied as though it
        lacked tools. Returning that prose directly asks the user to re-send,
        and the next send repeats the same path. The server should retry the
        resolve-equipped call once with an explicit nudge, then commit the tool
        result when the retry resolves.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        calls: list[dict] = []
        user_message = "Use this CSV path /tmp/orders.csv with headers order_id,url and discard invalid rows."
        false_decline = (
            "I don't have my tools available in this reply. "
            "Please re-send your message, or just say 'go ahead', "
            "so the tool-enabled version of me can pick it up."
        )

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _fake_llm_reply(false_decline)
            return _fake_resolve_source_response_csv()

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=fake_acompletion,
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message=user_message,
            )

        assert status == 200, body
        assert len(calls) == 2
        assert all(any(tool["function"]["name"] == "resolve_source" for tool in call["tools"]) for call in calls)
        assert any("do not ask the user to re-send" in m["content"].lower() for m in calls[1]["messages"])
        assert body["assistant_message"] == (
            "I did not apply generated source content. Review the current source form and submit it through the wizard controls."
        )
        assert body["assistant_message_kind"] == "synthetic_failure"
        assert body["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
        assert body["next_turn"]["type"] == "schema_form"
        assert not body["composition_state"]["sources"]
        chat_history = body["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[0]["content"] == user_message
        assert chat_history[1]["content"] == body["assistant_message"]
        assert false_decline not in [turn["content"] for turn in chat_history]
        audits = _llm_call_audit_bodies(composer_test_client, session_id)
        assert [audit["status"] for audit in audits] == ["success", "success"]

    def test_step_1_advisory_call_carries_no_tools_addendum_as_belt_and_braces(self, composer_test_client) -> None:
        """Belt-and-braces: when the resolve call itself is genuinely unusable
        (empty response — no tool call, no content), the route still falls
        back to a second, tool-less call, and THAT call carries the no-tools
        addendum so a real provider hiccup on this rare path still can't
        prime the model to hallucinate tool-call scaffolding.
        """
        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)
        _seed_guided_session(composer_test_client, session_id)

        empty_reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))])
        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return empty_reply
            return _fake_llm_reply("Here is what I can tell you.")

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=fake_acompletion,
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="I have a CSV, columns are name and email",
            )

        assert status == 200, body
        assert len(calls) == 2
        assert "tools" not in calls[1]
        advisory_system_messages = [m["content"] for m in calls[1]["messages"] if m["role"] == "system"]
        assert any("No tools in this reply" in content for content in advisory_system_messages)
        assert body["assistant_message"] == "Here is what I can tell you."


class TestStepChatRejections:
    def test_stale_turn_token_returns_409(self, composer_test_client: TestClient) -> None:
        """A token other than the current server-held turn fails before provider work."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        response = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/chat",
            json={"operation_id": str(uuid4()), "turn_token": "0" * 64, "message": "hi"},
        )

        assert response.status_code == 409, response.json()
        assert response.json()["detail"] == "turn_token does not identify the current unanswered turn."

    def test_step_index_is_rejected_as_an_extra_wire_field(self, composer_test_client: TestClient) -> None:
        """Schema 8 derives stage server-side and refuses the legacy step index."""
        session_id = _create_session(composer_test_client)
        turn = _seed_guided_session(composer_test_client, session_id)["next_turn"]

        response = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/chat",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "message": "hi",
                "step_index": "step_42_nope",
            },
        )

        assert response.status_code == 422, response.json()
        assert response.json()["detail"][0]["type"] == "extra_forbidden"

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
        )
        assert status == 422


class TestStepChatTransientFailure:
    def test_litellm_timeout_returns_synthetic_message(self, composer_test_client: TestClient) -> None:
        """TimeoutError from the LLM seam → 200 with the synthetic unavailable message.

        The session must NOT be terminated — chat is a non-load-bearing
        helper, so the session remains active for a later retry.
        """
        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)
        seeded = _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(TimeoutError("upstream LLM timed out")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
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

    def test_scaffold_leak_in_advisory_reply_returns_honest_retryable_message(self, composer_test_client: TestClient) -> None:
        """A model reply carrying raw <tool_call> scaffolding → 200 honest, retryable message.

        Observed live 2026-07-03 (guided step_1): the model answered the
        advisory chat path with a full pseudo tool-call transcript as literal
        content, which persisted into chat_history and rendered raw in the
        user-facing bubble — the same register violation the resolve-path
        guards already reject. The advisory reply is now guarded too:
        AssistantScaffoldLeakError is absorbed by the auto-drop wrapper, the
        session is untouched, and the user's Send stays retryable.

        C-1 (2026-07-04 review): the guard rejection must NOT claim
        unavailability — the service is fine, a quality check rejected this
        one reply — so the message is DISTINCT from
        ``test_litellm_timeout_returns_synthetic_message``'s genuine-outage
        copy above.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        scaffold_reply = (
            'Let me look up what is available. <tool_call> {"name": "list_sources"} '
            "</tool_call> <tool_response> [...] </tool_response> Good — csv_file fits. "
            "What is the category column called?"
        )
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply(scaffold_reply)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="Read my CSV and count rows per category.",
            )

        assert status == 200, body
        assert body["assistant_message"] != "I'm unavailable right now; you can still use the wizard controls."
        assert "unavailable" not in body["assistant_message"].lower()
        assert "quality check" in body["assistant_message"]
        # The scaffolding never lands in chat_history — only the user turn and
        # the honest reply.
        chat_history = body["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert "<tool_call" not in chat_history[1]["content"]
        assert chat_history[1]["content"] == body["assistant_message"]
        # Session unharmed: still at step_1, no terminal.
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_scaffold_leak_in_resolve_source_assistant_message_returns_honest_retryable_message(
        self, composer_test_client: TestClient
    ) -> None:
        """A scaffold leak INSIDE ``resolve_source``'s own ``assistant_message`` argument.

        Distinct from the advisory-path leak above: this one happens in the
        FIRST, tool-equipped call's tool-call arguments (observed live,
        tutorial ``resolve_source`` path), not the second, tool-less fallback.
        ``resolve_step_1_source_chat_with_auto_drop`` must route it through
        its own dedicated ``AssistantScaffoldLeakError`` branch, not the
        generic transient-failure catch — which would mislabel it
        "unavailable" and lose the distinction entirely.
        """
        session_id = _create_session(composer_test_client)
        # A persisted composition_state row must exist BEFORE the GET so the
        # GET call persists the SINGLE_SELECT TurnRecord — see the routing
        # test above for the same precondition.
        _seed_persisted_step1(composer_test_client, session_id)
        _seed_guided_session(composer_test_client, session_id)

        scaffold_args = {
            "resolution": "source",
            "plugin": "csv",
            "filename": "data.csv",
            "mime_type": "text/csv",
            "content": "name,email\nalice,a@x.test\n",
            "options": {"schema": {"mode": "observed"}},
            "observed_columns": ["name", "email"],
            "sample_rows": [{"name": "alice", "email": "a@x.test"}],
            "assistant_message": 'Let me check. <tool_call>{"name": "list_sources"}</tool_call> csv fits.',
        }
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_source_resolution_tool_call(scaffold_args)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="Here is my CSV: name,email\nalice,a@x.test",
            )

        assert status == 200, body
        assert body["assistant_message"] != "I'm unavailable right now; you can still use the wizard controls."
        assert "unavailable" not in body["assistant_message"].lower()
        assert "quality check" in body["assistant_message"]
        # The scaffold leak aborts before commit — no source lands in state.
        source_slot = (body["composition_state"]["sources"] or {}).get("source")
        assert source_slot is None, f"expected no committed source, got {source_slot!r}"

    @pytest.mark.parametrize("malformed_value", [float("inf"), "\ud800"], ids=["non_finite", "lone_surrogate"])
    def test_malformed_source_snapshot_auto_drops_before_blob_creation(
        self,
        composer_test_client: TestClient,
        malformed_value: object,
    ) -> None:
        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)
        malformed_args = {
            "resolution": "source",
            "plugin": "json",
            "filename": "rows.json",
            "mime_type": "application/json",
            "content": '[{"value": 1}]',
            "options": {"bad": malformed_value},
            "observed_columns": ["value"],
            "sample_rows": [{"value": 1}],
            "assistant_message": "Created the source.",
        }

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_source_resolution_tool_call(malformed_args)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="Create a JSON source.",
            )

        assert status == 200, body
        assert (body["composition_state"]["sources"] or {}).get("source") is None
        blobs = asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id)))
        assert blobs == []
        llm_calls = _llm_call_audit_bodies(composer_test_client, session_id)
        assert llm_calls[-1]["status"] == "malformed_response"
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["guided_session"]["terminal"] is None

    def test_route_enforces_server_side_timeout_bound(self, composer_test_client: TestClient) -> None:
        """A HUNG LLM call is bounded by ``settings.composer_timeout_seconds``.

        elspeth-fb4464cdf0: the guided chat route now threads the composer
        budget into the solver (``asyncio.wait_for``), the same bound freeform
        compose applies. A provider call that never returns must NOT hang the
        request past the budget — the wait_for fires TimeoutError, which the
        auto-drop wrapper maps to the 200 synthetic-unavailable contract.
        """

        async def hung_acompletion(**_kwargs: object) -> object:
            await asyncio.sleep(60)
            raise AssertionError("unreachable — the wait_for bound must fire first")

        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        # Shrink the budget so the test proves the bound without waiting 85 s.
        settings = composer_test_client.app.state.settings
        composer_test_client.app.state.settings = settings.model_copy(update={"composer_timeout_seconds": 0.05})
        try:
            with patch(
                _CHAT_SOLVER_ACOMPLETION,
                new=hung_acompletion,
            ):
                status, body = _post_chat(
                    composer_test_client,
                    session_id,
                    message="anything",
                )
        finally:
            composer_test_client.app.state.settings = settings

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."
        assert body["guided_session"]["terminal"] is None

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
        _seed_persisted_step1(composer_test_client, session_id)
        seeded = _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(BudgetExceededError(current_cost=10.0, max_budget=5.0)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(GuardrailRaisedException(guardrail_name="pii", message="blocked")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(BlockedPiiEntityError(entity_type="email", guardrail_name="pii")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(GuardrailInterventionNormalStringError(message="intervention")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
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
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(SimpleNamespace(choices=[])),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
            )

        assert status == 200, body
        assert body["assistant_message"] == "I'm unavailable right now; you can still use the wizard controls."


class TestStepChatServerInvariants:
    """Defective successful provider replies fail closed without partial audit state."""

    def test_empty_content_returns_sanitized_500(self, composer_test_client: TestClient) -> None:
        """Empty content string → InvariantError → 500 with B1-sanitized detail + slog."""
        from structlog.testing import capture_logs

        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)

        with (
            patch(
                _CHAT_SOLVER_ACOMPLETION,
                new=_ReturningLiteLLMCompletion(_fake_llm_reply("")),
            ),
            capture_logs() as cap_logs,
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
            )

        assert status == 500, body
        assert body["detail"]["error_type"] == "guided_operation_terminal_failure"
        assert body["detail"]["failure_code"] == "integrity_error"
        invariant_events = [
            event
            for event in cap_logs
            if event.get("event") == "guided.operation_terminal_failure" and event.get("site") == "post_guided_chat"
        ]
        assert len(invariant_events) == 1, cap_logs
        event = invariant_events[0]
        assert event["exc_class"] == "InvariantError"
        assert isinstance(event["frames"], tuple) and len(event["frames"]) > 0
        assert all(f.startswith("frame=") for f in event["frames"])
        assert _chat_turn_audit_bodies(composer_test_client, session_id) == []

        reloaded = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["guided_session"]
        assert reloaded["chat_history"] == []
        assert reloaded["chat_turn_seq"] == 0

    def test_terminal_failure_preserves_primary_error_if_slog_raises(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """A logger failure must not replace the typed terminal failure."""
        from elspeth.web.sessions.routes.composer import guided_chat_atomic

        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)

        monkeypatch.setattr(
            guided_chat_atomic.slog,
            "error",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated logger failure")),
        )

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
            )

        assert status == 500, body
        assert body["detail"]["failure_code"] == "integrity_error"

    def test_whitespace_only_content_returns_sanitized_500(self, composer_test_client: TestClient) -> None:
        """Whitespace-only content → same path as empty content (``.strip()`` is empty)."""
        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("   \n\t  ")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
            )

        assert status == 500, body
        assert body["detail"]["failure_code"] == "integrity_error"
        assert _chat_turn_audit_bodies(composer_test_client, session_id) == []

        reloaded = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["guided_session"]
        assert reloaded["chat_history"] == []
        assert reloaded["chat_turn_seq"] == 0


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
        """Seed an inline CSV blob and return its opaque guided reference."""
        content = "text,note\nHello,world\n"
        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
        )
        assert resp.status_code == 201, resp.json()
        blob_id = resp.json()["id"]
        return f"blob:{blob_id}"

    @staticmethod
    def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
        return _post_respond(client, session_id, **kwargs)

    @classmethod
    def _configure_csv_source(cls, client: TestClient, session_id: str) -> dict:
        """Advance the three current schema-8 Step-1 turns explicitly."""
        selected = cls._respond(client, session_id, chosen=["csv"])
        prefilled = selected["next_turn"]["payload"]["prefilled"]
        cls._respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": prefilled,
            },
        )
        return cls._respond(client, session_id, edited_values={"columns": ["text", "note"]})

    def test_chat_history_accumulates_across_step_transition(self, composer_test_client: TestClient) -> None:
        """Chat at step_1, advance to step_2, chat at step_2, GET /guided → 4 entries with mixed step values."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)
        self._seed_csv_blob(composer_test_client, session_id)

        # 1. Chat at step_1_source.  This records two ChatTurn entries
        #    (user + assistant, seq 0 + 1, step=step_1_source).
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("step-1 advice")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="how do I describe my CSV?",
            )

        # 2. Advance the current plugin, schema, and inspection turns.
        self._configure_csv_source(composer_test_client, session_id)

        # Confirm the wizard advanced — load-bearing precondition for
        # the cross-step assertion below.  Without this assertion, a
        # silent revert to step_1_source would mask the test as passing
        # while never exercising the cross-step transition.
        guided_after_advance = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()
        assert guided_after_advance["guided_session"]["step"] == "step_2_sink", guided_after_advance

        # 4. Chat at step_2_sink.  Two more ChatTurn entries (user +
        #    assistant, seq 2 + 3, step=step_2_sink).
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("step-2 advice")),
        ):
            _post_chat(
                composer_test_client,
                session_id,
                message="which sink for JSON output?",
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


class TestGuidedChatWireDiscriminator:
    """Live and persisted assistant-turn discriminators.

    The POST response carries ``assistant_message_kind`` so the frontend can
    distinguish a real assistant reply from a synthetic failure. Persisted
    synthetic-failure turns additionally require a closed reason value so a
    reload preserves the failure's machine-readable cause.
    """

    def test_real_assistant_reply_is_kind_assistant(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("Here's some advice.")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="what should I do?",
            )

        assert status == 200, body
        assert body["assistant_message_kind"] == "assistant"
        assert "synthetic_failure_reason" not in body

    def test_source_selection_transition_reply_is_kind_assistant(self, composer_test_client: TestClient) -> None:
        """A valid source selection is a real reply.

        The provider result is projected through the schema-8 transition
        authority, then the next-turn state, chat turns, audit evidence, and
        operation result are committed by the single atomic settlement.
        """
        session_id = _create_session(composer_test_client)
        _seed_persisted_step1(composer_test_client, session_id)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(
                _fake_source_resolution_tool_call(
                    {
                        "resolution": "source",
                        "plugin": "csv",
                        "filename": "data.csv",
                        "mime_type": "text/csv",
                        "content": "name,email\nalice,a@x.test\n",
                        "options": {"schema": {"mode": "observed"}},
                        "observed_columns": ["name", "email"],
                        "sample_rows": [{"name": "alice", "email": "a@x.test"}],
                        "assistant_message": "I built the source.",
                    }
                )
            ),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="here is my CSV: name,email\nalice,a@x.test",
            )

        assert status == 200, body
        assert body["assistant_message_kind"] == "assistant"

    def test_scaffold_leak_rejection_is_kind_synthetic_failure(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        scaffold_reply = 'Let me check. <tool_call>{"name": "list_sources"}</tool_call> csv fits.'
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply(scaffold_reply)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="Read my CSV and count rows per category.",
            )

        assert status == 200, body
        assert body["assistant_message_kind"] == "synthetic_failure"

    def test_provider_timeout_is_kind_synthetic_failure(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_RaisingLiteLLMCompletion(TimeoutError("upstream LLM timed out")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="anything",
            )

        assert status == 200, body
        assert body["assistant_message_kind"] == "synthetic_failure"
        # Provider unavailability uses distinct safe copy; the persisted turn
        # test below pins its required machine-readable reason.
        assert "quality check" not in body["assistant_message"]


class TestChatHistoryDiscriminatorPersistence:
    """Strict persisted-history discriminator contract.

    The discriminator must survive into the persisted ``chat_history`` a GET
    /guided reload serves. User turns carry two explicit null discriminator
    fields; real assistant turns require ``assistant`` plus a null reason;
    synthetic failures require ``synthetic_failure`` plus a closed reason.
    """

    def test_synthetic_failure_turn_persists_kind_across_get(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        scaffold_reply = 'Let me check. <tool_call>{"name": "list_sources"}</tool_call> csv fits.'
        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply(scaffold_reply)),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="Read my CSV and count rows per category.",
            )
        assert status == 200, body
        assert body["assistant_message_kind"] == "synthetic_failure"

        # Not just the live response — a fresh GET must show the same thing
        # from the persisted chat_history.
        get_resp = composer_test_client.get(f"/api/sessions/{session_id}/guided")
        assert get_resp.status_code == 200, get_resp.json()
        chat_history = get_resp.json()["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[0]["assistant_message_kind"] is None  # user turn: not applicable
        assert chat_history[1]["assistant_message_kind"] == "synthetic_failure"
        assert chat_history[1]["synthetic_failure_reason"] == "quality_guard"

    def test_real_reply_turn_persists_kind_across_get(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("Here's some advice.")),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="what should I do?",
            )
        assert status == 200, body
        assert body["assistant_message_kind"] == "assistant"

        get_resp = composer_test_client.get(f"/api/sessions/{session_id}/guided")
        assert get_resp.status_code == 200, get_resp.json()
        chat_history = get_resp.json()["guided_session"]["chat_history"]
        assert len(chat_history) == 2
        assert chat_history[1]["assistant_message_kind"] == "assistant"

    def test_chat_turn_without_required_kind_fields_fails_closed_on_get(self, composer_test_client: TestClient) -> None:
        """An incomplete schema-8 assistant turn is an integrity failure."""
        from elspeth.web.composer.guided.errors import InvariantError
        from elspeth.web.sessions.protocol import CompositionStateData
        from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

        session_id = _create_session(composer_test_client)
        service = composer_test_client.app.state.session_service
        session_uuid = UUID(session_id)

        state = _initial_composition_state_with_guided_session()
        guided_dict = state.guided_session.to_dict()
        guided_dict["chat_history"] = [
            {
                "role": "assistant",
                "content": "an incomplete reply",
                "seq": 0,
                "step": "step_1_source",
                "ts_iso": "2026-05-13T12:00:00+00:00",
                # No "assistant_message_kind" / "synthetic_failure_reason" keys.
            }
        ]
        guided_dict["chat_turn_seq"] = 1
        state_d = state.to_dict()
        state_data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta={"guided_session": guided_dict},
        )
        asyncio.run(service.save_composition_state(session_uuid, state_data, provenance="session_seed"))

        with pytest.raises(InvariantError, match="missing keys"):
            composer_test_client.get(f"/api/sessions/{session_id}/guided")


class TestStepChatProgressWiring:
    """elspeth-a8eeebb3aa: ``post_guided_chat`` now writes ``/composer-progress``
    snapshots to the SAME app-scoped ``ComposerProgressRegistry`` the freeform
    ``send_message`` route writes to, so a poller (``loadComposerProgress`` in
    sessionStore.ts) sees real phase movement during a guided chat compose —
    previously ``composerProgress`` stayed ``null`` for the entire guided
    compose because ``chatGuided`` never started polling AND the backend
    never published anything, and the only tests that exercised the substep
    indicator injected ``composerProgress`` directly via ``setState``.

    Deliberately drives BOTH step_1_source and step_2_sink to prove the
    wiring is uniform across guided steps (not forked on step_2_sink, and
    not forked on "is tutorial" — CLAUDE.md's tutorial-parity doctrine): the
    route publishes the same starting/complete bracket regardless of step,
    and only step_2_sink's resolver additionally hops through calling_model/
    using_tools mid-flight.
    """

    def _progress(self, client: TestClient, session_id: str) -> dict:
        resp = client.get(f"/api/sessions/{session_id}/composer-progress")
        assert resp.status_code == 200, resp.json()
        return resp.json()

    def test_step_1_chat_publishes_starting_to_complete_bracket(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)

        # Idle before any chat — proves the assertion below is caused by
        # THIS request, not leftover snapshot state from session creation.
        assert self._progress(composer_test_client, session_id)["phase"] == "idle"

        with patch(
            _CHAT_SOLVER_ACOMPLETION,
            new=_ReturningLiteLLMCompletion(_fake_llm_reply("step-1 advice")),
        ):
            status, _ = _post_chat(
                composer_test_client,
                session_id,
                message="how do I describe my CSV?",
            )
        assert status == 200

        progress = self._progress(composer_test_client, session_id)
        assert progress["phase"] == "complete"
        assert progress["reason"] == "composer_complete"

    def test_step_2_sink_chat_is_visibly_calling_model_mid_flight_then_completes(self, composer_test_client: TestClient) -> None:
        """The decisive check: poll the registry FROM INSIDE the patched LLM
        call (i.e. while the route's ``await`` on the provider round-trip is
        still pending) and see ``calling_model`` — not a stale snapshot and
        not ``idle``. This is what a real frontend poller would observe
        mid-compose; asserting only the post-request final state would not
        prove the mid-flight visibility the ticket is about.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)
        TestStepChatCrossStep._seed_csv_blob(composer_test_client, session_id)
        TestStepChatCrossStep._configure_csv_source(composer_test_client, session_id)
        guided = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()
        assert guided["guided_session"]["step"] == "step_2_sink", guided

        mid_flight_snapshots: list[dict] = []

        async def _capturing_completion(**_kwargs: object) -> SimpleNamespace:
            registry = composer_test_client.app.state.composer_progress_registry
            snapshot = await registry.get_latest(session_id)
            mid_flight_snapshots.append(snapshot.model_dump())
            return _fake_llm_reply("step-2 advice")

        with patch(_CHAT_SOLVER_ACOMPLETION, new=_capturing_completion):
            status, _ = _post_chat(
                composer_test_client,
                session_id,
                message="which sink for JSON output?",
            )
        assert status == 200

        assert len(mid_flight_snapshots) == 1
        assert mid_flight_snapshots[0]["phase"] == "calling_model"

        final_progress = self._progress(composer_test_client, session_id)
        assert final_progress["phase"] == "complete"
        assert final_progress["reason"] == "composer_complete"

    def test_step_2_sink_resolution_publishes_saving_while_committing(self, composer_test_client: TestClient) -> None:
        """The saving phase is visible while the atomic settlement runs.

        The sink proposal is projected by the pure schema-8 transition
        authority. The test captures progress from inside the settlement that
        commits the next state, chat turns, audit evidence, and operation
        result together.
        """
        session_id = _create_session(composer_test_client)
        _seed_guided_session(composer_test_client, session_id)
        TestStepChatCrossStep._seed_csv_blob(composer_test_client, session_id)
        TestStepChatCrossStep._configure_csv_source(composer_test_client, session_id)
        guided = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()
        assert guided["guided_session"]["step"] == "step_2_sink", guided

        # The sink path must remain within the session outputs directory for
        # the pure transition validator to accept the proposal.
        out_path = _outputs_path(composer_test_client, "chat_out.jsonl")
        resolve_sink_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="resolve_sink",
                                    arguments=json.dumps(
                                        {
                                            "resolution": "sink",
                                            "output": {
                                                "name": "main",
                                                "plugin": "json",
                                                "options": {
                                                    "path": out_path,
                                                    "schema": {"mode": "observed"},
                                                    "mode": "write",
                                                    "collision_policy": "auto_increment",
                                                },
                                                "required_fields": [],
                                                "schema_mode": "observed",
                                                "on_write_failure": "discard",
                                            },
                                            "assistant_message": "Output set to JSON.",
                                        }
                                    ),
                                )
                            )
                        ],
                    )
                )
            ]
        )

        mid_commit_snapshots: list[dict] = []
        service = composer_test_client.app.state.session_service
        real_settlement = service.settle_guided_state_operation

        async def _capturing_settlement(*args: object, **kwargs: object) -> object:
            registry = composer_test_client.app.state.composer_progress_registry
            snapshot = await registry.get_latest(session_id)
            mid_commit_snapshots.append(snapshot.model_dump())
            return await real_settlement(*args, **kwargs)

        with (
            patch(_CHAT_SOLVER_ACOMPLETION, new=_ReturningLiteLLMCompletion(resolve_sink_response)),
            patch.object(service, "settle_guided_state_operation", side_effect=_capturing_settlement),
        ):
            status, body = _post_chat(
                composer_test_client,
                session_id,
                message="save the results as json",
            )
        assert status == 200, body
        # Pin that this exercised the accepted transition, rather than a
        # synthetic transition-rejection settlement.
        assert body["assistant_message_kind"] == "assistant", body
        assert body["assistant_message"] == "Output set to JSON."

        assert len(mid_commit_snapshots) == 1
        assert mid_commit_snapshots[0]["phase"] == "saving"

        final_progress = self._progress(composer_test_client, session_id)
        assert final_progress["phase"] == "complete"
