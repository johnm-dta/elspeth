"""Integration tests for GET /api/sessions/{id}/guided.

Verifies:
- First fetch emits a turn, persists a TurnRecord to guided_session.history,
  and saves the updated state via save_composition_state.
- Re-fetch is idempotent: same payload_hash returned, no second TurnRecord
  appended, no second audit event persisted.
- Audit message (role=tool) is persisted after first fetch.
- 400 on freeform sessions (no guided_session attached — not currently
  exercised since all new sessions default to guided per spec §5.2).

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

from uuid import UUID

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

    def test_history_has_one_record_after_first_fetch(self, composer_test_client: TestClient) -> None:
        """After first fetch, guided_session.history contains exactly one TurnRecord."""
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        history = body["guided_session"]["history"]
        assert len(history) == 1
        record = history[0]
        assert record["step"] == "step_1_source"
        assert record["turn_type"] == "single_select"
        assert record["emitter"] == "server"
        assert record["payload_hash"]  # non-empty hash string
        assert record["response_hash"] is None

    def test_payload_hash_matches_deterministic_turn(self, composer_test_client: TestClient) -> None:
        """payload_hash is the stable_hash of the turn payload."""
        from elspeth.core.canonical import stable_hash

        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        recorded_hash = body["guided_session"]["history"][0]["payload_hash"]
        returned_payload = body["next_turn"]["payload"]
        expected_hash = stable_hash(returned_payload)
        assert recorded_hash == expected_hash


class TestGetGuidedIdempotency:
    def test_second_fetch_returns_same_payload_hash(self, composer_test_client: TestClient) -> None:
        """Re-fetching the same session returns the identical payload_hash."""
        session_id = _create_session(composer_test_client)

        body1 = _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        hash1 = body1["guided_session"]["history"][0]["payload_hash"]
        hash2 = body2["guided_session"]["history"][0]["payload_hash"]
        assert hash1 == hash2

    def test_second_fetch_does_not_append_extra_record(self, composer_test_client: TestClient) -> None:
        """Re-fetching does not grow the history — idempotent."""
        session_id = _create_session(composer_test_client)

        _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        assert len(body2["guided_session"]["history"]) == 1

    def test_second_fetch_has_same_turn_type(self, composer_test_client: TestClient) -> None:
        """Re-fetching returns the same turn type as the first fetch."""
        session_id = _create_session(composer_test_client)

        body1 = _get_guided(composer_test_client, session_id)
        body2 = _get_guided(composer_test_client, session_id)

        assert body1["next_turn"]["type"] == body2["next_turn"]["type"]


class TestGetGuidedStatePersistence:
    def test_guided_session_survives_roundtrip(self, composer_test_client: TestClient) -> None:
        """guided_session restored from DB on second fetch is equal to first-fetch state.

        This is the key Tier 1 round-trip test: the GuidedSession serialised
        into composer_meta on first fetch must deserialise identically on
        second fetch.  In-process identity is not sufficient — state_from_record
        must reconstruct the same GuidedSession.

        The test verifies the history length and record fields to catch
        serialisation gaps (missing field, type drift, enum coercion failure).
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)  # first fetch: saves state
        body2 = _get_guided(composer_test_client, session_id)  # second: loads from DB

        # Restored session must have history.
        history = body2["guided_session"]["history"]
        assert len(history) == 1
        r = history[0]
        assert r["step"] == "step_1_source"
        assert r["turn_type"] == "single_select"
        assert r["emitter"] == "server"
        assert r["response_hash"] is None

    def test_composition_state_returned_after_first_fetch(self, composer_test_client: TestClient) -> None:
        """composition_state in response is non-None after first fetch."""
        session_id = _create_session(composer_test_client)
        body = _get_guided(composer_test_client, session_id)

        assert body["composition_state"] is not None


class TestGetGuidedAuditTrail:
    def _get_tool_messages(self, client: TestClient, session_id: str) -> list:
        """Query chat messages from service layer directly (bypasses API filter).

        Audit messages (role=tool, _kind=audit) are deliberately excluded from
        the public GET /messages API response — they are internal audit rows,
        not conversation turns.  Query the service layer directly to verify
        persistence.

        Uses ``asyncio.run()`` since SyncASGITestClient runs in a thread pool
        that may not have an active event loop.
        """
        import asyncio

        app = client.app
        service = app.state.session_service

        msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        return [m for m in msgs if m.role == "tool"]

    def test_audit_message_persisted_after_first_fetch(self, composer_test_client: TestClient) -> None:
        """An audit role=tool message is persisted after the first GET /guided call.

        Verifies that _persist_tool_invocations was called with the recorder's
        contents.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        tool_messages = self._get_tool_messages(composer_test_client, session_id)
        assert len(tool_messages) >= 1, f"Expected at least one role=tool audit message, got {len(tool_messages)}"

        # Verify guided_turn_emitted is present.
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

    def test_no_second_audit_message_on_refetch(self, composer_test_client: TestClient) -> None:
        """Re-fetching does not add a second role=tool audit message."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)  # first
        _get_guided(composer_test_client, session_id)  # re-fetch

        tool_messages = self._get_tool_messages(composer_test_client, session_id)
        # Only one guided_turn_emitted audit message — re-fetch must not re-emit.
        assert len(tool_messages) == 1


class TestGetGuidedNotFound:
    def test_returns_404_for_unknown_session(self, composer_test_client: TestClient) -> None:
        """GET /guided on a non-existent session returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = composer_test_client.get(f"/api/sessions/{fake_id}/guided")
        assert resp.status_code == 404
