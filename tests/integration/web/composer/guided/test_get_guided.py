"""Integration tests for GET /api/sessions/{id}/guided.

Verifies:
- First fetch emits a turn, persists a TurnRecord to guided_session.history,
  and saves the updated state via save_composition_state.
- Re-fetch is idempotent: same payload_hash returned, no second TurnRecord
  appended, no second audit event persisted.
- Audit message (role=tool) is persisted after first fetch.
- 400 on freeform sessions (no guided_session attached — not currently
  exercised since all new sessions default to guided per spec §5.2).
- M3: GET /guided rebuilds Step 3 propose_chain from staged step_3_proposal
  (Codex #5 fix — previously returned next_turn: null at STEP_3_TRANSFORMS).
- M4: GET /guided rebuilds intra-step Step 2 turns from staged fields
  (Codex #10 fix — previously always returned the initial SINGLE_SELECT).
- M5: GET /guided passes blob inspection facts to build_initial_step_1_turn
  when step_1_source_intent is set (Codex #14 fix — previously passed None).

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
from uuid import UUID

import pytest

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


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    """POST /guided/respond and assert 200."""
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
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

    def test_history_is_empty_after_first_fetch_non_mutating(self, composer_test_client: TestClient) -> None:
        """After first fetch on a fresh session, history is empty (non-mutating).

        Commit c4e2f69cd made GET /guided non-mutating on fresh sessions to
        avoid allocating a v1 composition_state version on the frontend's
        auto-fetch. ``next_turn`` is returned in memory; ``history`` stays
        empty until the first mutating respond seeds the TurnRecord.  See
        ``test_history_has_one_record_after_first_mutation`` for the
        complementary post-mutation assertion.
        """
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["history"] == []
        assert body["next_turn"] is not None
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
        from elspeth.core.canonical import stable_hash

        session_id = _create_session(composer_test_client)
        first = _get_guided(composer_test_client, session_id)
        returned_payload = first["next_turn"]["payload"]
        expected_hash = stable_hash(returned_payload)

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

    def test_repeated_non_mutating_fetches_leave_history_empty(self, composer_test_client: TestClient) -> None:
        """Re-fetching a never-mutated session does not grow the history."""
        session_id = _create_session(composer_test_client)

        _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        assert body2["guided_session"]["history"] == []
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
        4. POST /respond edited_values={plugin, options, ...} → advance to step 2.
        5. GET /guided → must return step-2 single_select (sink plugins), NOT step-1.

        Distinguishes step-1 from step-2 single_select via step_index (0 vs 1)
        and payload.question text; both must indicate step 2 (index=1, sink list).
        """
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = self._seed_blob(composer_test_client, session_id)

        # Step 1: initialise
        get1 = _get_guided(composer_test_client, session_id)
        assert get1["next_turn"]["type"] == "single_select"
        assert get1["next_turn"]["step_index"] == 0  # STEP_1_SOURCE

        # Step 1: pick csv source
        _respond(composer_test_client, session_id, chosen=["csv"])

        # Step 1: submit schema_form → advances to step 2
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["col_a", "col_b"],
                "sample_rows": [],
            },
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
        _blob_id, storage_path = self._seed_blob(composer_test_client, session_id)

        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["col_a"],
                "sample_rows": [],
            },
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
    def test_payload_store_failure_before_first_get_emit_does_not_orphan_history(self, composer_test_client: TestClient) -> None:
        """A failed payload ref write must not make retry skip guided audit emission."""
        session_id = _create_session(composer_test_client)
        _seed_guided_session(
            composer_test_client,
            session_id,
            {
                "step": "step_2_sink",
                "history": [],
                "step_1_result": {
                    "plugin": "csv",
                    "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                    "observed_columns": ["col_a"],
                    "sample_rows": [{"col_a": "x"}],
                    "on_validation_failure": "discard",
                },
                "step_2_result": None,
                "step_3_proposal": None,
                "terminal": None,
                "transition_consumed": False,
                "step_1_source_intent": None,
                "step_2_sink_intent": None,
                "step_2_5_recipe_offer": None,
                "step_2_chosen_plugin": None,
                "chat_history": [],
                "chat_turn_seq": 0,
            },
        )
        payload_store = _FailOncePayloadStore(composer_test_client.app.state.payload_store)
        composer_test_client.app.state.payload_store = payload_store

        with pytest.raises(RuntimeError, match="payload store unavailable"):
            composer_test_client.get(f"/api/sessions/{session_id}/guided")

        body = _get_guided(composer_test_client, session_id)
        assert body["next_turn"]["type"] == "single_select"
        events = _guided_turn_emitted_args(composer_test_client, session_id)
        assert len(events) == 1
        payload_ref = events[0]["payload_payload_id"]
        assert payload_ref == events[0]["payload_hash"]
        assert payload_store.retrieve(payload_ref)


class TestGetGuidedFullStateRebuild:
    """M3/M4/M5: GET /guided correctly rebuilds turn state for all intra-step positions.

    Codex #5  (P1): STEP_3_TRANSFORMS returns next_turn: null before fix.
    Codex #10 (P2): STEP_2_SINK always rebuilds initial SINGLE_SELECT before fix.
    Codex #14 (P2): STEP_1_SOURCE never reaches INSPECT_AND_CONFIRM before fix.
    """

    def _make_source_resolved_dict(self) -> dict:
        return {
            "plugin": "csv",
            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
            "observed_columns": ["col_a", "col_b"],
            "sample_rows": [{"col_a": "x", "col_b": "1"}],
            "on_validation_failure": "discard",
        }

    def _make_sink_resolved_dict(self) -> dict:
        return {
            "outputs": [
                {
                    "plugin": "json",
                    "options": {"path": "/data/out.jsonl", "schema": {"mode": "observed"}},
                    "required_fields": ["col_a"],
                    "schema_mode": "observed",
                }
            ]
        }

    # ------------------------------------------------------------------
    # M3: Step 3 propose_chain rebuild (Codex #5)
    # ------------------------------------------------------------------

    def test_step_3_with_proposal_returns_propose_chain_turn(self, composer_test_client: TestClient) -> None:
        """GET /guided at STEP_3_TRANSFORMS with step_3_proposal returns propose_chain.

        Codex #5 fix: before the fix this returned next_turn=null and the
        frontend fell back to freeform UI despite the guided session being
        non-terminal.  The seeded proposal must appear in the response.
        """
        session_id = _create_session(composer_test_client)

        proposal_dict = {
            "steps": [{"plugin": "rename", "options": {"mappings": {}}, "rationale": "normalise names"}],
            "why": "Normalise column names before JSON output.",
        }
        guided_dict = {
            "step": "step_3_transforms",
            "history": [],
            "step_1_result": self._make_source_resolved_dict(),
            "step_2_result": self._make_sink_resolved_dict(),
            "step_3_proposal": proposal_dict,
            "terminal": None,
            "transition_consumed": False,
            "step_1_source_intent": None,
            "step_2_sink_intent": None,
            "step_2_5_recipe_offer": None,
            "step_2_chosen_plugin": None,
            "chat_history": [],
            "chat_turn_seq": 0,
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"] is not None, (
            "next_turn must not be null at STEP_3_TRANSFORMS when step_3_proposal is set — "
            "Codex #5 regression: GET /guided returned null instead of propose_chain"
        )
        assert body["next_turn"]["type"] == "propose_chain", f"Expected propose_chain but got {body['next_turn']['type']!r}"
        assert body["next_turn"]["step_index"] == 3  # STEP_3_TRANSFORMS index
        payload = body["next_turn"]["payload"]
        assert payload["why"] == proposal_dict["why"]
        assert len(payload["steps"]) == 1
        assert payload["steps"][0]["plugin"] == "rename"

    def test_step_4_wire_returns_confirm_wiring_turn_idempotently(self, composer_test_client: TestClient) -> None:
        """GET /guided at STEP_4_WIRE rebuilds the skeleton confirm_wiring turn."""
        session_id = _create_session(composer_test_client)

        guided_dict = {
            "step": "step_4_wire",
            "history": [],
            "step_1_result": self._make_source_resolved_dict(),
            "step_2_result": self._make_sink_resolved_dict(),
            "step_3_proposal": {
                "steps": [
                    {
                        "plugin": "passthrough",
                        "options": {"schema": {"mode": "observed"}},
                        "rationale": "identity chain",
                    }
                ],
                "why": "Rows already match.",
            },
            "terminal": None,
            "transition_consumed": False,
            "step_1_source_intent": None,
            "step_2_sink_intent": None,
            "step_2_5_recipe_offer": None,
            "step_2_chosen_plugin": None,
            "chat_history": [],
            "chat_turn_seq": 0,
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        first = _get_guided(composer_test_client, session_id)
        second = _get_guided(composer_test_client, session_id)

        assert first["guided_session"]["step"] == "step_4_wire"
        assert first["next_turn"] is not None
        assert first["next_turn"]["type"] == "confirm_wiring"
        assert first["next_turn"]["step_index"] == 4
        assert set(first["next_turn"]["payload"]) == {
            "topology",
            "edge_contracts",
            "semantic_contracts",
            "warnings",
        }
        assert first["next_turn"]["payload"]["topology"] == {
            "sources": {},
            "nodes": [],
            "outputs": [],
        }
        wire_records = [r for r in second["guided_session"]["history"] if r["step"] == "step_4_wire"]
        assert len(wire_records) == 1
        assert wire_records[0]["turn_type"] == "confirm_wiring"
        assert second["next_turn"]["type"] == "confirm_wiring"

    # ------------------------------------------------------------------
    # M4: Step 2 intra-step rebuild (Codex #10)
    # ------------------------------------------------------------------

    def test_step_2_with_chosen_plugin_returns_schema_form(self, composer_test_client: TestClient) -> None:
        """GET /guided at STEP_2_SINK with step_2_chosen_plugin returns schema_form.

        Codex #10 fix: before the fix, GET /guided always returned the initial
        SINGLE_SELECT regardless of intra-step position.  When the user had
        already picked a plugin, refresh would force them back to step start.
        """
        session_id = _create_session(composer_test_client)

        guided_dict = {
            "step": "step_2_sink",
            "history": [],
            "step_1_result": self._make_source_resolved_dict(),
            "step_2_result": None,
            "step_3_proposal": None,
            "terminal": None,
            "transition_consumed": False,
            "step_1_source_intent": None,
            "step_2_sink_intent": None,
            "step_2_5_recipe_offer": None,
            # Placed in the SINGLE_SELECT→SCHEMA_FORM window.
            "step_2_chosen_plugin": "json",
            "chat_history": [],
            "chat_turn_seq": 0,
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
        """GET /guided at STEP_2_SINK with step_2_sink_intent returns multi_select_with_custom.

        Codex #10 fix: after the SCHEMA_FORM was submitted (step_2_sink_intent is
        set), GET /guided must return multi_select_with_custom so the client can
        complete the field-selection step, not the initial single_select.
        """
        session_id = _create_session(composer_test_client)

        guided_dict = {
            "step": "step_2_sink",
            "history": [],
            "step_1_result": self._make_source_resolved_dict(),
            "step_2_result": None,
            "step_3_proposal": None,
            "terminal": None,
            "transition_consumed": False,
            "step_1_source_intent": None,
            # Placed in the SCHEMA_FORM→MULTI_SELECT window.
            "step_2_sink_intent": {
                "plugin": "json",
                "options": {"path": "/data/out.jsonl", "schema": {"mode": "observed"}},
            },
            "step_2_5_recipe_offer": None,
            "step_2_chosen_plugin": None,
            "chat_history": [],
            "chat_turn_seq": 0,
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "multi_select_with_custom", (
            f"Expected multi_select_with_custom but got {body['next_turn']['type']!r} — "
            "Codex #10 regression: GET /guided returned single_select instead of multi_select"
        )
        # default_chosen should include the columns from step_1_result.
        payload = body["next_turn"]["payload"]
        assert "col_a" in payload["default_chosen"]
        assert "col_b" in payload["default_chosen"]

    # ------------------------------------------------------------------
    # M5: Step 1 INSPECT_AND_CONFIRM rebuild (Codex #14)
    # ------------------------------------------------------------------

    def test_step_1_with_source_intent_returns_inspect_and_confirm(self, composer_test_client: TestClient) -> None:
        """GET /guided at STEP_1_SOURCE with step_1_source_intent returns inspect_and_confirm.

        Codex #14 fix: before the fix, build_initial_step_1_turn was called with
        blob_inspection=None unconditionally, making INSPECT_AND_CONFIRM unreachable
        via GET.  When step_1_source_intent is set, the server must rebuild the
        inspect_and_confirm turn from the staged observed columns.
        """
        session_id = _create_session(composer_test_client)

        guided_dict = {
            "step": "step_1_source",
            "history": [],
            "step_1_result": None,
            "step_2_result": None,
            "step_3_proposal": None,
            "terminal": None,
            "transition_consumed": False,
            # The SCHEMA_FORM was submitted — source intent is staged.
            "step_1_source_intent": {
                "plugin": "csv",
                "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                "observed_columns": ["col_a", "col_b"],
                "sample_rows": [{"col_a": "x", "col_b": "1"}],
            },
            "step_2_sink_intent": None,
            "step_2_5_recipe_offer": None,
            "step_2_chosen_plugin": None,
            "chat_history": [],
            "chat_turn_seq": 0,
        }
        _seed_guided_session(composer_test_client, session_id, guided_dict)

        body = _get_guided(composer_test_client, session_id)

        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "inspect_and_confirm", (
            f"Expected inspect_and_confirm but got {body['next_turn']['type']!r} — "
            "Codex #14 regression: GET /guided returned single_select instead of inspect_and_confirm"
        )
        # The observed columns from step_1_source_intent must appear in the payload.
        observed = body["next_turn"]["payload"]["observed"]
        assert "col_a" in observed["columns"]
        assert "col_b" in observed["columns"]
