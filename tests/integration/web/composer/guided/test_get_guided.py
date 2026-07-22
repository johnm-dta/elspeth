"""Integration tests for GET /api/sessions/{id}/guided.

Verifies:
- First fetch projects one deterministic prospective TurnRecord without
  persisting a state version, payload, or audit message.
- Re-fetch is idempotent: the same payload hash and turn token are returned.
- 400 on freeform sessions (no guided_session attached — not currently
  exercised since all new sessions default to guided per spec §5.2).
- Schema-8 pending source/output intents rebuild their exact intra-step turn.
- Later authoring stages do not synthesize a legacy embedded proposal.

HTTP transport: SyncASGITestClient (in-process, synchronous — same pattern
as test_fixture_smoke.py).  The full roundtrip exercises:
  - route handler lock + load-or-create branch
  - emitters.build_initial_step_1_turn (single_select)
  - GuidedSession serialisation into composer_meta
  - state_from_record restoring guided_session
  - audit drain via _persist_tool_invocations

Per spec §5.2 / errata C7: all fresh sessions default to guided mode.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    """Create a session and return its id."""
    resp = client.post("/api/sessions", json={"title": "guided-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    """Call GET /api/sessions/{id}/guided."""
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _start_guided(client: TestClient, session_id: str) -> dict:
    response = client.post(
        f"/api/sessions/{session_id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid4())},
    )
    assert response.status_code == 200, response.json()
    return response.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    """POST one closed schema-8 response to the currently projected turn."""
    current = _get_guided(client, session_id)
    turn = current["next_turn"]
    payload = {
        "operation_id": str(uuid4()),
        "turn_token": turn["turn_token"] if turn is not None else None,
        **kwargs,
    }
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=payload)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_first_turn(client: TestClient, session_id: str) -> dict:
    """Force first-turn persistence under the non-mutating GET semantics.

    GET /guided is non-mutating on a fresh session (commit c4e2f69cd,
    May 15 2026, "Fix composer and session bug regressions"). It returns
    the deterministic step_1 turn in memory but does NOT persist a
    TurnRecord or composition_state version — the frontend auto-fetches
    GET /guided on session load, and creating a v1 graph version per
    auto-fetch would pollute every session's state history with an empty
    bootstrap version.

    Tests that need persisted history / composition_state version / audit
    rows for the step_1 surface must trigger a mutating action. This
    helper does the cheapest valid one: GET /guided to expose the step_1
    turn (idempotent, no state change), then POST /guided/respond with
    ``chosen=["csv"]`` to answer it. The session advances to step_2_blob;
    the step_1 TurnRecord lands in ``guided_session.history`` as the first
    record (with ``response_hash`` populated by the answer); composition
    state version 1 is allocated; the ``guided_turn_emitted`` audit row
    is persisted.
    """
    _get_guided(client, session_id)
    return _respond(client, session_id, chosen=["csv"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetGuidedFirstFetch:
    def test_returns_200_with_guided_session(self, composer_test_client: TestClient) -> None:
        """GET /guided on a fresh session returns HTTP 200 with guided_session."""
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        assert "guided_session" in body
        assert body["guided_session"]["step"] == "step_1_source"

    def test_next_turn_is_single_select(self, composer_test_client: TestClient) -> None:
        """Without a blob, next_turn is a single_select over source plugins."""
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "single_select"
        payload = body["next_turn"]["payload"]
        assert "options" in payload
        assert len(payload["options"]) > 0  # at least one source plugin registered
        # CSV plugin must be listed (canonical source for test data)
        option_ids = [o["id"] for o in payload["options"]]
        assert "csv" in option_ids, f"csv not in option_ids: {option_ids}"

    def test_history_projects_first_occurrence_without_persisting_state(self, composer_test_client: TestClient) -> None:
        """A fresh GET projects the exact prospective turn occurrence.

        GET remains non-mutating (no composition-state version), while the
        response history includes the in-memory occurrence needed to derive
        the required turn token. RESPOND reconstructs this same occurrence.
        """
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        assert len(body["guided_session"]["history"]) == 1
        assert body["guided_session"]["history"][0]["response_hash"] is None
        assert body["next_turn"] is not None
        assert len(body["next_turn"]["turn_token"]) == 64
        assert body["composition_state"] is None

    def test_history_has_one_record_after_first_mutation(self, composer_test_client: TestClient) -> None:
        """After the first mutating respond, history records the step_1 turn.

        The first ``POST /guided/respond`` on a fresh session auto-seeds the
        step_1 TurnRecord (so the dispatcher knows what turn the answer is
        for), applies the user's choice, advances to step_2, and persists.
        The persisted ``guided_session.history`` therefore carries the
        step_1 record with a populated ``response_hash``.
        """
        session_id = _create_session(composer_test_client)
        _seed_first_turn(composer_test_client, session_id)
        body = _get_guided(composer_test_client, session_id)

        history = body["guided_session"]["history"]
        assert len(history) >= 1
        step_1 = next(r for r in history if r["step"] == "step_1_source")
        assert step_1["turn_type"] == "single_select"
        assert step_1["emitter"] == "server"
        assert step_1["payload_hash"]  # non-empty hash string
        assert step_1["response_hash"] is not None  # answered by _seed_first_turn

    def test_payload_hash_matches_deterministic_turn(self, composer_test_client: TestClient) -> None:
        """``next_turn["payload"]`` is the deterministic step-1 surface; its
        ``stable_hash`` matches the persisted ``payload_hash`` after the
        first mutating respond.
        """
        from elspeth.web.sessions.protocol import guided_json_payload_id

        session_id = _create_session(composer_test_client)
        first = _get_guided(composer_test_client, session_id)
        returned_payload = first["next_turn"]["payload"]
        expected_hash = guided_json_payload_id("turn", returned_payload)

        _seed_first_turn(composer_test_client, session_id)
        body = _get_guided(composer_test_client, session_id)
        step_1 = next(r for r in body["guided_session"]["history"] if r["step"] == "step_1_source")
        assert step_1["payload_hash"] == expected_hash


class TestGetGuidedIdempotency:
    def test_second_fetch_returns_same_payload_hash_after_mutation(self, composer_test_client: TestClient) -> None:
        """Re-fetching after the first mutation returns the identical
        step_1 ``payload_hash`` (deterministic rebuild)."""
        session_id = _create_session(composer_test_client)
        _seed_first_turn(composer_test_client, session_id)
        body1 = _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        hash1 = next(r for r in body1["guided_session"]["history"] if r["step"] == "step_1_source")["payload_hash"]
        hash2 = next(r for r in body2["guided_session"]["history"] if r["step"] == "step_1_source")["payload_hash"]
        assert hash1 == hash2

    def test_repeated_non_mutating_fetches_reuse_prospective_occurrence(self, composer_test_client: TestClient) -> None:
        """Re-fetching a never-mutated session does not grow or rotate its token."""
        session_id = _create_session(composer_test_client)

        body1 = _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        assert len(body2["guided_session"]["history"]) == 1
        assert body2["guided_session"]["history"] == body1["guided_session"]["history"]
        assert body2["next_turn"]["turn_token"] == body1["next_turn"]["turn_token"]
        assert body2["composition_state"] is None

    def test_second_fetch_has_same_turn_type(self, composer_test_client: TestClient) -> None:
        """Re-fetching returns the same turn type as the first fetch."""
        session_id = _create_session(composer_test_client)

        body1 = _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        assert body1["next_turn"]["type"] == body2["next_turn"]["type"]


class TestGetGuidedStatePersistence:
    def test_guided_session_survives_roundtrip(self, composer_test_client: TestClient) -> None:
        """guided_session restored from DB on a post-mutation fetch is equal
        to the in-process state.

        This is the key Tier 1 round-trip test: the GuidedSession serialised
        into composer_meta on the first mutating respond must deserialise
        identically on the following fetch.  In-process identity is not
        sufficient — state_from_record must reconstruct the same
        GuidedSession.

        The test verifies the history record fields (step_1) to catch
        serialisation gaps (missing field, type drift, enum coercion
        failure).
        """
        session_id = _create_session(composer_test_client)
        _seed_first_turn(composer_test_client, session_id)  # mutating: persists state
        body2 = _get_guided(composer_test_client, session_id)  # loads from DB

        history = body2["guided_session"]["history"]
        step_1 = next((r for r in history if r["step"] == "step_1_source"), None)
        assert step_1 is not None, f"step_1_source record missing from history: {history!r}"
        assert step_1["turn_type"] == "single_select"
        assert step_1["emitter"] == "server"
        assert step_1["response_hash"] is not None  # answered by _seed_first_turn

    def test_composition_state_is_none_until_first_mutation(self, composer_test_client: TestClient) -> None:
        """composition_state in the response is None until the first respond.

        The May 15 non-mutating-GET contract is observable here: a fresh
        session's GET /guided returns ``composition_state=None`` because no
        v1 graph version has been allocated. The first mutating respond
        creates v1 and subsequent fetches return a non-None
        composition_state.
        """
        session_id = _create_session(composer_test_client)
        first = _get_guided(composer_test_client, session_id)
        assert first["composition_state"] is None

        _seed_first_turn(composer_test_client, session_id)
        after = _get_guided(composer_test_client, session_id)
        assert after["composition_state"] is not None


class TestGetGuidedAuditTrail:
    def _get_tool_messages(self, client: TestClient, session_id: str) -> list:
        """Query chat messages from service layer directly (bypasses API filter).

        Audit-bearing messages are deliberately excluded from the public
        GET /messages API response — they are internal audit rows, not
        conversation turns.  Query the service layer directly to verify
        persistence.

        Post Phase-1B/rev-4 (`_persist_tool_invocations`): guided audit
        invocations land on ``role='audit'`` when there is no parent
        assistant message (the GET /guided path emits server-side without
        a paired assistant chat). Include both ``role='tool'`` and
        ``role='audit'`` here so the test's persistence check sees what
        the production endpoint actually writes.

        Uses ``asyncio.run()`` since SyncASGITestClient runs in a thread pool
        that may not have an active event loop.
        """
        import asyncio

        app = client.app
        service = app.state.session_service

        msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        return [m for m in msgs if m.role in ("tool", "audit")]

    def test_no_audit_message_after_non_mutating_fetch(self, composer_test_client: TestClient) -> None:
        """Non-mutating GET /guided on a fresh session emits no audit row.

        Under the May 15 non-mutating-GET contract, no state changed, so
        nothing to audit.  Once the operator mutates (POST respond),
        ``test_audit_message_persisted_after_first_mutation`` confirms
        the audit row lands.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        _get_guided(composer_test_client, session_id)  # idempotent re-fetch

        tool_messages = self._get_tool_messages(composer_test_client, session_id)
        assert tool_messages == [], (
            f"Expected no audit messages after non-mutating fetches, got {len(tool_messages)}: {[m.tool_calls for m in tool_messages]}"
        )

    def test_audit_message_persisted_after_first_mutation(self, composer_test_client: TestClient) -> None:
        """A ``guided_turn_emitted`` audit row is persisted after the first
        mutating respond.

        The first mutating respond auto-seeds the step_1 TurnRecord (so the
        dispatcher knows what turn the operator's answer is for) and emits
        ``guided_turn_emitted`` for that record. The auto-seed runs inside
        the respond endpoint's compose-lock, so the audit row is part of
        the same atomic state transition.
        """
        session_id = _create_session(composer_test_client)
        _seed_first_turn(composer_test_client, session_id)

        tool_messages = self._get_tool_messages(composer_test_client, session_id)
        assert len(tool_messages) >= 1, f"Expected at least one audit message, got {len(tool_messages)}"

        def _has_guided_turn_emitted(msg) -> bool:
            tool_calls = msg.tool_calls
            if not tool_calls:
                return False
            for tc in tool_calls:
                invocation = tc.get("invocation", {})
                if invocation.get("tool_name") == "guided_turn_emitted":
                    return True
            return False

        assert any(_has_guided_turn_emitted(m) for m in tool_messages), (
            f"No guided_turn_emitted found in tool messages: {[m.tool_calls for m in tool_messages]}"
        )


class TestGetGuidedNotFound:
    def test_returns_404_for_unknown_session(self, composer_test_client: TestClient) -> None:
        """GET /guided on a non-existent session returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = composer_test_client.get(f"/api/sessions/{fake_id}/guided")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# M2: GET /guided after step advance returns step-2 turn, not step-1
# ---------------------------------------------------------------------------


class TestGetGuidedAfterStepAdvance:
    """M2: Verifies C1 fix — GET /guided dispatches per-step, not unconditionally step-1.

    Before the fix, GET /guided always called build_initial_step_1_turn regardless
    of guided.step.  A session advanced to step 2 would receive a step-1
    single_select (listing source plugins) instead of the step-2 single_select
    (listing sink plugins).  The test drives through step 1 via POST /respond and
    then GETs /guided, asserting the response reflects step 2.
    """

    def _seed_blob(self, client: TestClient, session_id: str) -> tuple[str, str]:
        """Seed a CSV blob and return (blob_id, storage_path)."""
        import asyncio
        from uuid import UUID

        content = "col_a,col_b\n1,x\n2,y\n"
        resp = client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
        )
        assert resp.status_code == 201, resp.json()
        blob_id = resp.json()["id"]
        blob_service = client.app.state.blob_service
        record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
        return blob_id, record.storage_path

    def _outputs_path(self, client: TestClient, filename: str) -> str:
        """Return an absolute path under {data_dir}/outputs/."""
        from pathlib import Path

        data_dir: Path = client.app.state.settings.data_dir
        outputs_dir = data_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return str(outputs_dir / filename)

    def test_get_guided_after_step_1_advance_returns_step_2_turn(self, composer_test_client: TestClient) -> None:
        """GET /guided after step-1 → step-2 advance returns a step-2 next_turn.

        Sequence:
        1. Create session.
        2. GET /guided → step-1 single_select.
        3. POST /respond chosen=["csv"] → intra-step schema_form.
        4. POST /respond with the server-prefilled source form.
        5. POST /respond with reviewed inspection columns → source review.
        6. POST /respond finish source review → advance to step 2.
        7. GET /guided → must return step-2 single_select (sink plugins), NOT step-1.

        Distinguishes step-1 from step-2 single_select via step_index (0 vs 1)
        and payload.question text; both must indicate step 2 (index=1, sink list).
        """
        session_id = _create_session(composer_test_client)
        self._seed_blob(composer_test_client, session_id)

        # Step 1: initialise
        get1 = _get_guided(composer_test_client, session_id)
        assert get1["next_turn"]["type"] == "single_select"
        assert get1["next_turn"]["step_index"] == 0  # STEP_1_SOURCE

        # Step 1: pick csv source
        selected = _respond(composer_test_client, session_id, chosen=["csv"])

        # Step 1: submit the strict server-prefilled form, then review inspection.
        inspected = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": selected["next_turn"]["payload"]["prefilled"],
            },
        )
        observed_columns = inspected["next_turn"]["payload"]["observed"]["columns"]
        assert observed_columns == ["col_a", "col_b"]
        _respond(composer_test_client, session_id, edited_values={"columns": observed_columns})
        _respond(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "source"},
        )

        # GET /guided after step advance must reflect step 2, not step 1.
        get2 = _get_guided(composer_test_client, session_id)

        assert get2["guided_session"]["step"] == "step_2_sink", (
            f"guided_session.step should be step_2_sink but got {get2['guided_session']['step']!r}"
        )
        assert get2["next_turn"] is not None, "next_turn must not be None at step 2"
        # step_index must be 1 (STEP_2_SINK), not 0 (STEP_1_SOURCE).
        assert get2["next_turn"]["step_index"] == 1, (
            f"Expected step_index=1 (STEP_2_SINK) but got {get2['next_turn']['step_index']!r} — "
            "C1 regression: GET /guided returned step-1 turn after step advance"
        )
        # Turn type must be single_select listing sink plugins (not source plugins).
        assert get2["next_turn"]["type"] == "single_select"
        option_ids = [o["id"] for o in get2["next_turn"]["payload"]["options"]]
        # Step-1 single_select lists source plugins (e.g. "csv").
        # Step-2 single_select lists sink plugins (e.g. "json").
        assert "json" in option_ids, f"Sink plugin 'json' missing from options — may be getting step-1 source list: {option_ids}"

    def test_get_guided_after_step_advance_is_idempotent(self, composer_test_client: TestClient) -> None:
        """Re-fetching GET /guided at step 2 returns the same turn (idempotency).

        After step 1 advances to step 2, the first GET /guided at step 2 emits a
        TurnRecord and persists it.  A second GET must return the same step_index
        and turn type without appending a new record.
        """
        session_id = _create_session(composer_test_client)
        self._seed_blob(composer_test_client, session_id)

        _get_guided(composer_test_client, session_id)
        selected = _respond(composer_test_client, session_id, chosen=["csv"])
        inspected = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": selected["next_turn"]["payload"]["prefilled"],
            },
        )
        observed_columns = inspected["next_turn"]["payload"]["observed"]["columns"]
        assert observed_columns == ["col_a", "col_b"]
        _respond(composer_test_client, session_id, edited_values={"columns": observed_columns})
        _respond(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "source"},
        )

        get_a = _get_guided(composer_test_client, session_id)
        get_b = _get_guided(composer_test_client, session_id)

        # Both fetches must return the same step-2 turn.
        assert get_a["next_turn"]["step_index"] == get_b["next_turn"]["step_index"] == 1
        assert get_a["next_turn"]["type"] == get_b["next_turn"]["type"] == "single_select"
        # Idempotency: second fetch must not append an extra TurnRecord at step 2.
        step_2_records = [r for r in get_b["guided_session"]["history"] if r["step"] == "step_2_sink"]
        assert len(step_2_records) == 1, f"Expected exactly 1 step_2_sink TurnRecord on re-fetch, got {len(step_2_records)}"


# ---------------------------------------------------------------------------
# M3/M4/M5: GET /guided full-state rebuild (Codex #5, #10, #14)
#
# These tests seed GuidedSession state directly via save_composition_state
# to place the session at a specific intra-step position, then verify that
# GET /guided returns the correct turn for that position.
# ---------------------------------------------------------------------------


def _seed_guided_session(client: TestClient, session_id: str, guided_session_dict: dict) -> None:
    """Persist a pre-built guided session dict into composer_meta.

    Seeds state directly via save_composition_state — the same pattern used
    by test_respond._seed_inspect_and_confirm_history.  This bypasses the
    normal POST /respond flow so tests can place the session at arbitrary
    intra-step positions without driving the full wizard sequence.
    """
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.web.composer.guided.state_machine import GuidedSession
    from elspeth.web.sessions.converters import state_from_record
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

    service = client.app.state.session_service
    session_uuid = UUID(session_id)
    state_record = asyncio.run(service.get_current_state(session_uuid))

    if state_record is None:
        state = _initial_composition_state_with_guided_session()
        existing_meta: dict = {}
    else:
        state = state_from_record(state_record)
        existing_meta = dict(deep_thaw(state_record.composer_meta)) if state_record.composer_meta else {}

    current_guided_shape = GuidedSession.initial().to_dict()
    current_guided_shape.update(guided_session_dict)

    new_composer_meta = {**existing_meta, "guided_session": current_guided_shape}
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


class _FailOncePayloadStore:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.store_calls = 0

    def store(self, content: bytes) -> str:
        self.store_calls += 1
        if self.store_calls == 1:
            raise RuntimeError("payload store unavailable")
        return self._delegate.store(content)

    def retrieve(self, content_hash: str) -> bytes:
        return self._delegate.retrieve(content_hash)

    def exists(self, content_hash: str) -> bool:
        return self._delegate.exists(content_hash)

    def delete(self, content_hash: str) -> bool:
        return self._delegate.delete(content_hash)


class _CorruptingPayloadStore:
    def __init__(self, delegate, corrupt_payload_id: str) -> None:
        self._delegate = delegate
        self._corrupt_payload_id = corrupt_payload_id

    def store(self, content: bytes) -> str:
        return self._delegate.store(content)

    def retrieve(self, content_hash: str) -> bytes:
        content = self._delegate.retrieve(content_hash)
        return content + b"corrupt" if content_hash == self._corrupt_payload_id else content

    def exists(self, content_hash: str) -> bool:
        return self._delegate.exists(content_hash)

    def delete(self, content_hash: str) -> bool:
        return self._delegate.delete(content_hash)


def _guided_turn_emitted_args(client: TestClient, session_id: str) -> list[dict]:
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    events: list[dict] = []
    for msg in msgs:
        if msg.role not in ("tool", "audit") or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            invocation = tool_call.get("invocation", {})
            if invocation.get("tool_name") == "guided_turn_emitted":
                events.append(json.loads(invocation["arguments_canonical"]))
    return events


class TestGetGuidedAuditPayloadOrdering:
    def test_persisted_occurrence_projects_exact_cas_across_catalog_availability_drift(
        self,
        composer_test_client: TestClient,
    ) -> None:
        from dataclasses import replace

        from elspeth.web.auth.models import UserIdentity

        session_id = _create_session(composer_test_client)
        started = _start_guided(composer_test_client, session_id)
        snapshot = composer_test_client.app.state.plugin_snapshot_factory(UserIdentity(user_id="alice", username="alice"))
        composer_test_client.app.state.plugin_snapshot_factory = lambda _user: replace(snapshot, available=frozenset())

        with patch(
            "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
            side_effect=AssertionError("persisted GET must not consult the live catalog"),
        ):
            fetched = _get_guided(composer_test_client, session_id)

        assert fetched["next_turn"] == started["next_turn"]
        assert fetched["next_turn"]["turn_token"] == started["next_turn"]["turn_token"]

    @pytest.mark.parametrize("failure_mode", ["missing", "corrupt"])
    def test_persisted_occurrence_cas_failure_is_fail_closed(
        self,
        composer_test_client: TestClient,
        failure_mode: str,
    ) -> None:
        session_id = _create_session(composer_test_client)
        started = _start_guided(composer_test_client, session_id)
        payload_id = started["guided_session"]["history"][-1]["payload_hash"]
        payload_store = composer_test_client.app.state.payload_store

        if failure_mode == "missing":
            assert payload_store.delete(payload_id)
        else:
            composer_test_client.app.state.payload_store = _CorruptingPayloadStore(payload_store, payload_id)

        with pytest.raises(AuditIntegrityError, match="Guided replay payload"):
            composer_test_client.get(f"/api/sessions/{session_id}/guided")

    def test_persisted_lazy_get_is_prospective_and_never_splits_history_from_evidence(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """A read-only GET does not touch CAS, state history, or audit rows."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(
            composer_test_client,
            session_id,
            {
                "step": "step_2_sink",
                "history": [],
            },
        )
        payload_store = _FailOncePayloadStore(composer_test_client.app.state.payload_store)
        composer_test_client.app.state.payload_store = payload_store

        body = _get_guided(composer_test_client, session_id)
        assert body["next_turn"]["type"] == "single_select"
        assert len(body["guided_session"]["history"]) == 1
        assert payload_store.store_calls == 0
        assert _guided_turn_emitted_args(composer_test_client, session_id) == []
        versions = asyncio.run(composer_test_client.app.state.session_service.get_state_versions(UUID(session_id)))
        assert [version.version for version in versions] == [1]


class TestGetGuidedFullStateRebuild:
    """M3/M4/M5: GET /guided correctly rebuilds turn state for all intra-step positions.

    Codex #5  (P1): STEP_3_TRANSFORMS returns next_turn: null before fix.
    Codex #10 (P2): STEP_2_SINK always rebuilds initial SINGLE_SELECT before fix.
    Codex #14 (P2): STEP_1_SOURCE never reaches INSPECT_AND_CONFIRM before fix.
    """

    def _make_source_resolved_dict(self) -> dict:
        return {
            "name": "source_1",
            "plugin": "csv",
            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
            "observed_columns": ["col_a", "col_b"],
            "sample_rows": [{"col_a": "x", "col_b": "1"}],
            "on_validation_failure": "discard",
        }

    def test_step_3_checkpoint_does_not_rebuild_external_proposal(self, composer_test_client: TestClient) -> None:
        """Schema 8 does not duplicate proposal payloads in guided custody.

        Durable proposal payloads are owned by the proposal service.  A plain
        guided checkpoint therefore cannot synchronously reconstruct a Step 3
        turn and must project ``next_turn: null`` instead of inventing one.
        """
        session_id = _create_session(composer_test_client)
        guided_dict = {
            "step": "step_3_transforms",
            "history": [],
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"] is None

    def test_step_4_wire_without_active_proposal_fails_closed(self, composer_test_client: TestClient) -> None:
        """Schema 9 rejects a wire checkpoint without its bound proposal."""
        session_id = _create_session(composer_test_client)

        guided_dict = {
            "step": "step_4_wire",
            "history": [],
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        response = composer_test_client.get(f"/api/sessions/{session_id}/guided")

        assert response.status_code == 500
        assert response.json()["detail"] == "Server invariant violated. See application audit log for diagnostic detail."

    # ------------------------------------------------------------------
    # M4: Step 2 intra-step rebuild (Codex #10)
    # ------------------------------------------------------------------

    def test_step_2_with_chosen_plugin_returns_schema_form(self, composer_test_client: TestClient) -> None:
        """A plugin-options output intent rebuilds its schema form.

        Codex #10 fix: before the fix, GET /guided always returned the initial
        SINGLE_SELECT regardless of intra-step position.  When the user had
        already picked a plugin, refresh would force them back to step start.
        """
        session_id = _create_session(composer_test_client)
        output_id = str(uuid4())

        guided_dict = {
            "step": "step_2_sink",
            "history": [],
            "output_order": [output_id],
            "pending_output_intents": {
                output_id: {
                    "name": "output_1",
                    "phase": "plugin_options",
                    "plugin": "json",
                    "options": None,
                }
            },
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "schema_form", (
            f"Expected schema_form but got {body['next_turn']['type']!r} — "
            "Codex #10 regression: GET /guided returned single_select instead of schema_form"
        )
        assert body["next_turn"]["payload"]["plugin"] == "json"

    def test_step_2_with_sink_intent_returns_multi_select(self, composer_test_client: TestClient) -> None:
        """A field-review output intent rebuilds its field-selection turn.

        After plugin options are reviewed, GET must preserve the current
        field-selection phase rather than restarting at plugin selection.
        """
        session_id = _create_session(composer_test_client)
        source_id = str(uuid4())
        output_id = str(uuid4())

        guided_dict = {
            "step": "step_2_sink",
            "history": [],
            "source_order": [source_id],
            "reviewed_sources": {source_id: self._make_source_resolved_dict()},
            "output_order": [output_id],
            "pending_output_intents": {
                output_id: {
                    "name": "output_1",
                    "phase": "field_review",
                    "plugin": "json",
                    "options": {"path": "/data/out.jsonl", "schema": {"mode": "observed"}},
                }
            },
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "multi_select_with_custom", (
            f"Expected multi_select_with_custom but got {body['next_turn']['type']!r} — "
            "Codex #10 regression: GET /guided returned single_select instead of multi_select"
        )
        # Defaults come from the server-held reviewed source projection.
        payload = body["next_turn"]["payload"]
        assert "col_a" in payload["default_chosen"]
        assert "col_b" in payload["default_chosen"]

    # ------------------------------------------------------------------
    # M5: Step 1 INSPECT_AND_CONFIRM rebuild (Codex #14)
    # ------------------------------------------------------------------

    def test_step_1_with_source_intent_returns_inspect_and_confirm(self, composer_test_client: TestClient) -> None:
        """An inspection-review source intent rebuilds inspect-and-confirm.

        The pending intent holds server-derived inspection facts and must
        reconstruct the same review turn after refresh.
        """
        session_id = _create_session(composer_test_client)
        source_id = str(uuid4())
        blob_id = str(uuid4())

        guided_dict = {
            "step": "step_1_source",
            "history": [],
            "source_order": [source_id],
            "pending_source_intents": {
                source_id: {
                    "name": "source_1",
                    "phase": "inspection_review",
                    "plugin": "csv",
                    "options": {"path": f"blob:{blob_id}", "schema": {"mode": "observed"}},
                    "inspection_facts": {
                        "source_kind": "csv",
                        "redacted_identity": {"blob_id": blob_id},
                        "byte_range_inspected": [0, 16],
                        "sample_row_count": 1,
                        "observed_headers": ["col_a", "col_b"],
                        "inferred_types": {"col_a": "str", "col_b": "int"},
                        "url_candidates": [],
                        "warnings": [],
                    },
                    "observed_columns": ["col_a", "col_b"],
                    "sample_rows": [{"col_a": "x", "col_b": "1"}],
                }
            },
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "inspect_and_confirm", (
            f"Expected inspect_and_confirm but got {body['next_turn']['type']!r} — "
            "Codex #14 regression: GET /guided returned single_select instead of inspect_and_confirm"
        )
        # Server-held inspection headers must appear in the payload.
        observed = body["next_turn"]["payload"]["observed"]
        assert "col_a" in observed["columns"]
        assert "col_b" in observed["columns"]
