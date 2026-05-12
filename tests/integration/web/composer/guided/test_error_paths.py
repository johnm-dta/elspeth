"""Integration tests for POST /api/sessions/{id}/guided/respond — error paths.

Covers:
  - exit_to_freeform control signal terminates the guided session (§5.3)
  - POST /guided/respond after terminal state returns 409 (§9.4)
  - POST /guided/respond before GET /guided (no TurnRecord) returns 400
  - POST /guided/respond on a session not in guided mode returns 400

HTTP transport: SyncASGITestClient (in-process, synchronous — same pattern
as test_respond.py).

Per spec §5.3, §9.4:
  - Any control_signal="exit_to_freeform" → terminal.kind="exited_to_freeform",
    terminal.reason="user_pressed_exit", next_turn=None
  - Any respond after terminal → 409 Conflict
"""

from __future__ import annotations

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_respond.py — these are self-contained tests
# that must not import from another test module to avoid coupling)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    """Create a session and return its string id."""
    resp = client.post("/api/sessions", json={"title": "error-path-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    """Fetch GET /guided and assert 200."""
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond_raw(client: TestClient, session_id: str, **kwargs) -> object:
    """POST /guided/respond and return the raw response (any status)."""
    return client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    """POST /guided/respond and assert 200."""
    resp = _respond_raw(client, session_id, **kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


# ---------------------------------------------------------------------------
# exit_to_freeform — §5.3 manual exit
# ---------------------------------------------------------------------------


class TestExitToFreeform:
    def test_exit_from_step_1_terminates(self, composer_test_client: TestClient) -> None:
        """exit_to_freeform from step 1 (SINGLE_SELECT) terminates with user_pressed_exit."""
        session_id = _create_session(composer_test_client)
        # Seed the first TurnRecord by fetching GET /guided.
        _get_guided(composer_test_client, session_id)

        resp = _respond_raw(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )

        assert resp.status_code == 200, resp.json()
        body = resp.json()

        # Top-level terminal field.
        assert body["terminal"] is not None
        assert body["terminal"]["kind"] == "exited_to_freeform"
        assert body["terminal"]["reason"] == "user_pressed_exit"
        assert body["terminal"]["pipeline_yaml"] is None

        # guided_session.terminal is also set.
        gs = body["guided_session"]
        assert gs["terminal"] is not None
        assert gs["terminal"]["kind"] == "exited_to_freeform"
        assert gs["terminal"]["reason"] == "user_pressed_exit"

        # No further turn is emitted — wizard is done.
        assert body["next_turn"] is None

    def test_exit_from_step_1_intra_step_terminates(self, composer_test_client: TestClient) -> None:
        """exit_to_freeform works from within an intra-step turn (SCHEMA_FORM).

        Drives step 1 to SCHEMA_FORM (after SINGLE_SELECT response), then exits.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        # Advance to SCHEMA_FORM.
        _respond(composer_test_client, session_id, chosen=["csv"])

        # Now exit from the SCHEMA_FORM turn.
        resp = _respond_raw(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )

        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert body["terminal"]["kind"] == "exited_to_freeform"
        assert body["terminal"]["reason"] == "user_pressed_exit"
        assert body["next_turn"] is None

    def test_exit_terminal_is_persisted(self, composer_test_client: TestClient) -> None:
        """After exit_to_freeform, GET /guided reflects the terminal state."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        _respond(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )

        # Fetch guided state again — terminal must be persisted.
        resp = composer_test_client.get(f"/api/sessions/{session_id}/guided")
        assert resp.status_code == 200, resp.json()
        guided = resp.json()["guided_session"]
        assert guided["terminal"] is not None
        assert guided["terminal"]["kind"] == "exited_to_freeform"


# ---------------------------------------------------------------------------
# 409 after terminal — §9.4 error matrix
# ---------------------------------------------------------------------------


class TestRespondAfterTerminal:
    def _drive_to_terminal(self, client: TestClient, session_id: str) -> None:
        """Drive the session to terminal via exit_to_freeform."""
        _get_guided(client, session_id)
        resp = _respond(client, session_id, control_signal="exit_to_freeform")
        assert resp["terminal"] is not None

    def test_respond_after_exit_returns_409(self, composer_test_client: TestClient) -> None:
        """A second POST /guided/respond after terminal returns 409 Conflict."""
        session_id = _create_session(composer_test_client)
        self._drive_to_terminal(composer_test_client, session_id)

        # Any subsequent respond must be rejected.
        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "terminal" in detail.lower()

    def test_respond_after_exit_returns_409_regardless_of_payload(self, composer_test_client: TestClient) -> None:
        """409 is returned even if the payload would be otherwise valid."""
        session_id = _create_session(composer_test_client)
        self._drive_to_terminal(composer_test_client, session_id)

        # Try another exit_to_freeform — still 409.
        resp = _respond_raw(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )

        assert resp.status_code == 409

    def test_repeated_409_responses_are_stable(self, composer_test_client: TestClient) -> None:
        """Multiple responds after terminal all return 409, not 500 (idempotent rejection)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_terminal(composer_test_client, session_id)

        for _ in range(3):
            resp = _respond_raw(
                composer_test_client,
                session_id,
                chosen=["csv"],
            )
            assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Pre-condition violations — respond without prior GET /guided
# ---------------------------------------------------------------------------


class TestRespondPreconditions:
    def test_respond_without_prior_get_guided_returns_400(self, composer_test_client: TestClient) -> None:
        """POST /guided/respond before GET /guided (no TurnRecord) returns 400.

        The route requires at least one TurnRecord to exist for the current
        step — it cannot infer the turn type without one.  Callers must always
        fetch GET /guided first to seed the initial turn.
        """
        session_id = _create_session(composer_test_client)
        # Do NOT call _get_guided — no TurnRecord exists yet.

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Should mention fetching GET /guided.
        assert "guided" in detail.lower()

    def test_respond_on_non_guided_session_returns_400(self, composer_test_client: TestClient) -> None:
        """POST /guided/respond on a session with no guided_session returns 400.

        Sessions only enter guided mode when created with guided=True (or when
        the default guided session is initialised by the first GET /guided call).
        A bare POST /api/sessions without the guided flag has no guided_session
        and must be rejected with 400, not 500.

        NOTE: In the current implementation, GET /guided auto-initialises the
        guided session on first call, so this test creates a session and
        bypasses the guided initialisation entirely by calling respond
        against the initial empty state.
        """
        session_id = _create_session(composer_test_client)
        # Do not call GET /guided — guided_session is None in initial state.

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        # Either 400 (no guided mode) or 400 (no TurnRecord) are acceptable —
        # both indicate the call was rejected before doing any state mutation.
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Security — InvariantError 500 detail must not leak corrupted-record content
# (PR #37 review finding B1)
# ---------------------------------------------------------------------------


class TestInvariantError500DetailIsSanitized:
    """Pin the B1 fix: the InvariantError → 500 catch must NOT interpolate
    ``str(exc)`` into the HTTP response body.

    Several ``from_dict`` raise sites in ``state_machine.py`` build their
    message as ``f"...: malformed record {d!r}"`` — and ``d`` for
    ``GuidedSession.from_dict`` / ``SourceResolved.from_dict`` contains
    the Tier-3 ``sample_rows`` payload from the source plugin.  Echoing
    that into the HTTP detail would surface user/PII content (e.g. CSV
    cell values such as customer SSNs) to the browser-side JSON 500 body.

    The catch sites at ``routes.py:`` (step_advance / dispatch_guided_respond)
    now use a static detail message and route diagnostic content through
    ``slog`` under the audit-system-failure exemption.
    """

    _SENTINEL = "AAA-BB-CCCC-SENTINEL-DO-NOT-LEAK"
    _STATIC_DETAIL = "Server invariant violated. See application audit log for diagnostic detail."

    def _seed_guided_session(self, client: TestClient, session_id: str) -> None:
        """Seed the initial Step 1 SINGLE_SELECT TurnRecord so the respond
        handler will load the session and reach ``step_advance``.
        """
        _get_guided(client, session_id)

    def test_step_advance_invariant_does_not_leak_to_response(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """When ``step_advance`` raises ``InvariantError`` with a sentinel
        message, the HTTP 500 body contains the static detail and **not**
        the sentinel content nor the field name ``sample_rows``.
        """
        from elspeth.web.composer.guided.errors import InvariantError
        from elspeth.web.sessions import routes as routes_module

        session_id = _create_session(composer_test_client)
        self._seed_guided_session(composer_test_client, session_id)

        # Replace ``step_advance`` as the route module imported it (top-level
        # ``from ... import step_advance``).  The patched function builds a
        # message shaped like ``GuidedSession.from_dict``'s ``{d!r}`` repr so
        # this test reproduces the leak vector the unpatched code would have
        # surfaced into ``detail``.
        leaky_record_repr = "{'sample_rows': [{'ssn': '" + self._SENTINEL + "', 'name': 'Alice'}], 'step': 'step_1_source'}"

        def _raise_invariant_with_leak(*args, **kwargs):
            raise InvariantError(f"GuidedSession.from_dict: malformed record {leaky_record_repr}")

        monkeypatch.setattr(routes_module, "step_advance", _raise_invariant_with_leak)

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 500
        body = resp.json()
        detail = body["detail"]

        # Static-detail pin.
        assert detail == self._STATIC_DETAIL, f"InvariantError 500 detail must be the static B1 message; got {detail!r}"

        # Negative pins: neither the sentinel nor structural leak markers
        # may appear anywhere in the serialised body.  Using the full body
        # text (not just ``detail``) catches accidental leaks via response
        # headers or auxiliary fields.
        body_text = resp.text
        assert self._SENTINEL not in body_text, "B1 leak: sentinel content escaped into the 500 response body"
        assert "sample_rows" not in body_text, "B1 leak: structural field name 'sample_rows' surfaced in the 500 response body"
        assert "malformed record" not in body_text, "B1 leak: from_dict error prefix surfaced in the 500 response body"

    def test_dispatcher_invariant_does_not_leak_to_response(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """When the dispatcher path raises ``InvariantError`` with a sentinel
        message, the HTTP 500 body contains the static detail and **not**
        the sentinel content.

        Exercises the second catch site (after ``step_advance`` succeeds but
        ``_dispatch_guided_respond`` raises).  Patches the dispatcher
        directly to inject the leaky message.
        """
        from elspeth.web.composer.guided.errors import InvariantError
        from elspeth.web.sessions import routes as routes_module

        session_id = _create_session(composer_test_client)
        self._seed_guided_session(composer_test_client, session_id)

        leaky_record_repr = "{'sample_rows': [{'ssn': '" + self._SENTINEL + "', 'name': 'Bob'}], 'plugin': 'csv'}"

        async def _raise_invariant_in_dispatcher(*args, **kwargs):
            raise InvariantError(f"SourceResolved.from_dict: malformed record {leaky_record_repr}")

        monkeypatch.setattr(routes_module, "_dispatch_guided_respond", _raise_invariant_in_dispatcher)

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail == self._STATIC_DETAIL, f"Dispatcher InvariantError 500 detail must be the static B1 message; got {detail!r}"

        body_text = resp.text
        assert self._SENTINEL not in body_text
        assert "sample_rows" not in body_text
        assert "malformed record" not in body_text


# ---------------------------------------------------------------------------
# Server-invariant gates raise InvariantError (PR #37 review finding I1)
#
# Four sites previously raised ``RuntimeError`` for server-bug invariant
# conditions (Step-3 repair guards + post-compose impossible-state guards).
# RuntimeError lands at FastAPI's default 500 handler with no structured log
# and no audit-system-failure exemption record. Converting to ``InvariantError``
# routes the failure through the B1 sanitized handler: static 500 detail +
# ``guided.invariant_violated`` slog event that on-call dashboards filter on.
# ---------------------------------------------------------------------------


class TestI1InvariantErrorStructuredLogging:
    """Pin that the 4 sites converted in I1 fire the structured slog event
    when the invariant is violated, not a generic FastAPI 500.

    The B1 static-detail pin in ``TestInvariantError500DetailIsSanitized``
    proves the response body is sanitized; this class proves the parallel
    requirement — that the failure is observable via the structured log
    channel keyed on ``guided.invariant_violated``.
    """

    _STATIC_DETAIL = "Server invariant violated. See application audit log for diagnostic detail."

    def test_dispatcher_invariant_emits_structured_slog_event(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """When ``_dispatch_guided_respond`` raises ``InvariantError`` (the
        class the Step-3 repair guards at routes.py:~2396/2398 now raise),
        the route emits a ``guided.invariant_violated`` slog event with the
        B1-convention field set (``exc_class`` + ``frames`` + ``site``, no
        raw exception message).

        Without the RuntimeError → InvariantError conversion, the guard
        would escape as a plain RuntimeError to FastAPI's default 500 with
        no structured log — operators would see only "Internal Server
        Error" with no audit-derivation trail.
        """
        from structlog.testing import capture_logs

        from elspeth.web.composer.guided.errors import InvariantError
        from elspeth.web.sessions import routes as routes_module

        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        # Reproduce the exact text the converted Step-3 repair site raises.
        # The site is unreachable from a well-formed client (Step 3 can only
        # be entered after Steps 1+2 commit), so we patch the dispatcher to
        # synthesise the InvariantError shape that ``raise InvariantError(...)``
        # at routes.py:~2396 now produces.
        async def _raise_invariant_in_dispatcher(*args, **kwargs):
            raise InvariantError("repair: step_1_result is None — STEP_3 unreachable without Step 1 commit")

        monkeypatch.setattr(routes_module, "_dispatch_guided_respond", _raise_invariant_in_dispatcher)

        with capture_logs() as cap_logs:
            resp = _respond_raw(
                composer_test_client,
                session_id,
                chosen=["csv"],
            )

        # The dispatcher's outer except InvariantError handler converts to
        # a sanitized 500 with the B1 static detail.
        assert resp.status_code == 500
        assert resp.json()["detail"] == self._STATIC_DETAIL

        # The structured slog event must be emitted under the B1 convention.
        matches = [entry for entry in cap_logs if entry.get("event") == "guided.invariant_violated"]
        assert matches, f"expected guided.invariant_violated slog event for converted dispatcher InvariantError; got entries: {cap_logs}"
        entry = matches[0]

        # Field-set pin. B1 convention: exc_class + frames + site +
        # session_id + user_id. Never str(exc) or exc_info.
        assert entry["exc_class"] == "InvariantError", entry
        assert entry["site"] == "dispatch_guided_respond", entry
        assert "frames" in entry, entry
        assert "session_id" in entry, entry
        assert "user_id" in entry, entry

        # Forbidden keys: any field carrying the raw exception message is
        # a B1 leak vector. The Step-3 repair message itself is harmless,
        # but the dispatcher might one day raise an InvariantError whose
        # message embeds {d!r} (cf. ``from_dict`` callers).
        forbidden_keys = {"exc_message", "exception", "exc_info"}
        assert not (forbidden_keys & entry.keys()), f"slog entry must not carry the raw exception message under the B1 convention: {entry}"
