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
  - Any NON-exit respond after terminal → 409 Conflict
  - control_signal="exit_to_freeform" against a kind="completed" terminal →
    200 + terminal transitions COMPLETED → EXITED_TO_FREEFORM (wizard-teardown
    exemption -- pinned by ``TestExitFromCompletedTerminal``).  The exemption
    is narrow: it does NOT apply to already-exited sessions and does NOT apply
    to any non-exit payload.
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


def _reenter_raw(client: TestClient, session_id: str) -> object:
    """POST /guided/reenter and return the raw response (any status)."""
    return client.post(f"/api/sessions/{session_id}/guided/reenter")


def _seed_first_turn(client: TestClient, session_id: str) -> dict:
    """Force first-turn persistence under the non-mutating GET semantics.

    See ``test_get_guided.py:_seed_first_turn`` for the rationale.  Briefly:
    GET /guided is non-mutating on a fresh session (commit c4e2f69cd,
    May 15 2026); tests that need a persisted composition_state version
    must trigger a mutating respond.
    """
    _get_guided(client, session_id)
    resp = _respond_raw(client, session_id, chosen=["csv"])
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


class TestReenterGuided:
    def test_reenter_user_exit_clears_terminal_and_returns_live_turn(self, composer_test_client: TestClient) -> None:
        """A user-pressed freeform exit can be reversed from the command palette.

        Re-entry is not a normal turn response: it clears the
        ``exited_to_freeform/user_pressed_exit`` terminal marker and rebuilds
        the current guided turn from server state, giving the frontend a live
        ``next_turn`` to render again.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        exited = _respond(composer_test_client, session_id, control_signal="exit_to_freeform")
        assert exited["terminal"]["kind"] == "exited_to_freeform"

        resp = _reenter_raw(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert body["terminal"] is None
        assert body["guided_session"]["terminal"] is None
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "single_select"
        current_step_records = [record for record in body["guided_session"]["history"] if record["step"] == body["guided_session"]["step"]]
        assert current_step_records[-1]["response_hash"] is None
        assert current_step_records[-1]["summary"] is None

        persisted = _get_guided(composer_test_client, session_id)
        assert persisted["guided_session"]["terminal"] is None
        assert persisted["next_turn"] is not None

    def test_reenter_rejects_auto_drop_terminals(self, composer_test_client: TestClient) -> None:
        """Only deliberate user exits are reversible.

        Solver exhaustion and protocol-violation terminals represent failed
        guided runs, not a mode switch, so the re-entry command must not revive
        them.
        """
        import asyncio
        from dataclasses import replace
        from uuid import UUID

        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.state_machine import (
            TerminalKind,
            TerminalReason,
            TerminalState,
        )
        from elspeth.web.sessions.converters import state_from_record
        from elspeth.web.sessions.protocol import CompositionStateData

        session_id = _create_session(composer_test_client)
        # Seed a persisted composition_state version — GET /guided is
        # non-mutating on fresh sessions (commit c4e2f69cd), so a mutating
        # respond is required to materialise the state we then overwrite
        # with a SOLVER_EXHAUSTED terminal to drive the reenter-rejection
        # path.
        _seed_first_turn(composer_test_client, session_id)
        service = composer_test_client.app.state.session_service
        session_uuid = UUID(session_id)
        state_record = asyncio.run(service.get_current_state(session_uuid))
        assert state_record is not None
        state = state_from_record(state_record)
        assert state.guided_session is not None

        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.SOLVER_EXHAUSTED,
            pipeline_yaml=None,
        )
        guided = replace(state.guided_session, terminal=terminal)
        state = replace(state, guided_session=guided)
        existing_meta = dict(deep_thaw(state_record.composer_meta)) if state_record.composer_meta else {}
        state_d = state.to_dict()
        state_data = CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta={**existing_meta, "guided_session": guided.to_dict()},
        )
        asyncio.run(service.save_composition_state(session_uuid, state_data, provenance="session_seed"))

        resp = _reenter_raw(composer_test_client, session_id)

        assert resp.status_code == 409
        assert "user exit" in resp.json()["detail"].lower()


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
# Exit-from-COMPLETED — wizard-teardown exemption from the 409 guard.
#
# When the wizard reaches ``terminal.kind == "completed"``, the
# CompletionSummary surface (frontend: CompletionSummary.tsx) shows two
# action buttons that fire ``control_signal=exit_to_freeform`` to tear the
# wizard down and return the user to the freeform chat surface.  Without
# the exemption added to ``post_guided_respond``, those clicks hit the
# blanket 409 guard and the user is stuck on the summary (the ChatPanel
# discriminator keeps the completed surface visible until
# terminal.kind transitions to ``exited_to_freeform``).
#
# These tests pin the narrow exemption shape:
#   * EXIT_TO_FREEFORM from COMPLETED -> 200 + terminal kind transitions.
#   * Non-exit payloads from COMPLETED -> still 409.
#   * EXIT_TO_FREEFORM from EXITED_TO_FREEFORM -> still 409 (already-exited).
#   * No ``guided_turn_answered`` is emitted on this path (no turn was
#     answered); ``guided_dropped_to_freeform`` IS emitted.
# ---------------------------------------------------------------------------


class TestExitFromCompletedTerminal:
    """Pin the wizard-teardown exemption: exit_to_freeform from kind=COMPLETED
    must transition to kind=EXITED_TO_FREEFORM instead of returning 409.
    """

    def _seed_completed_terminal(self, client: TestClient, session_id: str) -> None:
        """Seed a GuidedSession with terminal.kind=COMPLETED.

        Direct state injection mirrors the
        ``test_respond.py::_seed_inspect_and_confirm_history`` pattern: we
        bypass the full wizard drive (which would require LLM stubs and
        recipe-accept slot wiring) and write the desired terminal state
        straight into composer_meta.  The route under test reads
        ``state.guided_session`` from that record and exercises the
        exit-from-COMPLETED branch.
        """
        import asyncio
        from dataclasses import replace
        from uuid import UUID

        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
        from elspeth.web.composer.guided.state_machine import (
            GuidedSession,
            TerminalKind,
            TerminalState,
            TurnRecord,
        )
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

        # Build a GuidedSession resting at STEP_3_TRANSFORMS with COMPLETED
        # terminal carrying a non-trivial pipeline_yaml.  The handler must
        # NOT propagate ``pipeline_yaml`` to the new EXITED_TO_FREEFORM
        # terminal (the TerminalState invariant restricts pipeline_yaml to
        # kind=COMPLETED); the test asserts that elision explicitly.
        guided = state.guided_session if state.guided_session is not None else GuidedSession.initial()
        record = TurnRecord(
            step=GuidedStep.STEP_3_TRANSFORMS,
            turn_type=TurnType.PROPOSE_CHAIN,
            payload_hash="seed-payload-hash",
            response_hash="seed-response-hash",
            emitter="server",
        )
        completed_terminal = TerminalState(
            kind=TerminalKind.COMPLETED,
            reason=None,
            pipeline_yaml="version: 1\nsource:\n  plugin: csv\n",
        )
        guided = replace(
            guided,
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(*guided.history, record),
            terminal=completed_terminal,
        )
        state = replace(state, guided_session=guided)

        new_composer_meta = {**existing_meta, "guided_session": guided.to_dict()}
        state_d = state.to_dict()
        state_data = CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta=new_composer_meta,
        )
        asyncio.run(service.save_composition_state(session_uuid, state_data, provenance="session_seed"))

    def test_exit_from_completed_transitions_to_exited_to_freeform(self, composer_test_client: TestClient) -> None:
        """POST /guided/respond with control_signal=exit_to_freeform against a
        kind=COMPLETED terminal must return 200 and transition the terminal.

        Bug ref: CompletionSummary.tsx:70-75 buttons fire exit_to_freeform; the
        prior implementation 409'd on this combination because the terminal
        guard ran before reading control_signal, leaving the user with no chat
        input and no way out of the summary.
        """
        session_id = _create_session(composer_test_client)
        self._seed_completed_terminal(composer_test_client, session_id)

        body = _respond(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )

        # Top-level terminal field.
        assert body["terminal"] is not None
        assert body["terminal"]["kind"] == "exited_to_freeform"
        assert body["terminal"]["reason"] == "user_pressed_exit"
        # TerminalState invariant: pipeline_yaml is set only on kind=COMPLETED.
        # The COMPLETED terminal had non-null yaml; the new EXITED_TO_FREEFORM
        # terminal must drop it (yaml is recoverable from composition_state).
        assert body["terminal"]["pipeline_yaml"] is None

        # guided_session.terminal mirrors the top-level field.
        gs = body["guided_session"]
        assert gs["terminal"]["kind"] == "exited_to_freeform"
        assert gs["terminal"]["reason"] == "user_pressed_exit"

        # No live turn after teardown.
        assert body["next_turn"] is None

    def test_exit_from_completed_is_persisted(self, composer_test_client: TestClient) -> None:
        """After exit-from-COMPLETED, GET /guided reflects the new terminal AND
        the surrounding ``composition_state`` is preserved intact.

        The route's persistence call is ``_replace(state, guided_session=...)``,
        which leaves ``source``/``nodes``/``edges``/``outputs`` untouched.  The
        production-code comment justifies dropping ``terminal.pipeline_yaml`` on
        the grounds that the yaml is recoverable from ``composition_state``; this
        assertion pins that claim by verifying the surrounding state survives.
        """
        session_id = _create_session(composer_test_client)
        self._seed_completed_terminal(composer_test_client, session_id)
        composition_before = _get_guided(composer_test_client, session_id)["composition_state"]

        _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

        body = _get_guided(composer_test_client, session_id)
        guided = body["guided_session"]
        assert guided["terminal"] is not None
        assert guided["terminal"]["kind"] == "exited_to_freeform"
        assert guided["terminal"]["reason"] == "user_pressed_exit"
        assert guided["terminal"]["pipeline_yaml"] is None

        # Composition-state survives the wizard-teardown: the yaml is
        # recoverable from these fields.
        composition_after = body["composition_state"]
        assert composition_after["source"] == composition_before["source"]
        assert composition_after["nodes"] == composition_before["nodes"]
        assert composition_after["edges"] == composition_before["edges"]
        assert composition_after["outputs"] == composition_before["outputs"]

    def test_non_exit_payload_from_completed_still_returns_409(self, composer_test_client: TestClient) -> None:
        """Sending any non-exit payload (chosen, edited_values, ...) against a
        kind=COMPLETED terminal must still return 409.  The exemption is
        narrow: it ONLY applies to control_signal=exit_to_freeform.
        """
        session_id = _create_session(composer_test_client)
        self._seed_completed_terminal(composer_test_client, session_id)

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 409
        assert "terminal" in resp.json()["detail"].lower()

    def test_exit_from_already_exited_terminal_still_returns_409(self, composer_test_client: TestClient) -> None:
        """exit_to_freeform against a kind=EXITED_TO_FREEFORM terminal is
        idempotently rejected with 409 -- exiting an already-exited session
        is a no-op.  This guards against accidentally widening the exemption
        when refactoring the terminal-rejection logic.
        """
        session_id = _create_session(composer_test_client)
        # Drive the session terminal via the normal exit flow.
        _get_guided(composer_test_client, session_id)
        first = _respond(composer_test_client, session_id, control_signal="exit_to_freeform")
        assert first["terminal"]["kind"] == "exited_to_freeform"

        # A second exit should NOT take the exemption path -- only COMPLETED is exempt.
        resp = _respond_raw(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )
        assert resp.status_code == 409

    def test_exit_from_completed_audit_shape(self, composer_test_client: TestClient) -> None:
        """The exit-from-COMPLETED path emits ``guided_dropped_to_freeform``
        and does NOT emit ``guided_turn_answered`` (no turn was being
        answered -- the wizard had already completed).
        """
        import asyncio
        import json
        from uuid import UUID

        session_id = _create_session(composer_test_client)
        self._seed_completed_terminal(composer_test_client, session_id)

        _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

        # Collect guided-mode tool invocations from the audit log.
        service = composer_test_client.app.state.session_service
        msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        tool_names: list[str] = []
        for msg in msgs:
            # Post Phase-1B/rev-4: guided audit invocations on this exit-from-
            # COMPLETED path land with role='audit' (no parent assistant — the
            # exit signal is server-side-only, no LLM call). See the matching
            # _get_tool_messages docstrings in test_audit_emission.py and
            # test_auto_drop.py for the full rationale.
            if msg.role not in ("tool", "audit") or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                invocation = tc.get("invocation", {})
                name = invocation.get("tool_name")
                if name in {
                    "guided_turn_emitted",
                    "guided_turn_answered",
                    "guided_step_advanced",
                    "guided_dropped_to_freeform",
                }:
                    tool_names.append(name)
                    # For dropped_to_freeform, also assert the directive's
                    # prev_step captures the step at which the wizard had
                    # completed (the seeded STEP_3_TRANSFORMS) and the reason
                    # matches USER_PRESSED_EXIT.
                    if name == "guided_dropped_to_freeform":
                        args = json.loads(invocation.get("arguments_canonical", "{}"))
                        assert args["prev_step"] == "step_3_transforms"
                        assert args["drop_reason"] == "user_pressed_exit"

        assert "guided_dropped_to_freeform" in tool_names, f"expected guided_dropped_to_freeform to be emitted; got: {tool_names}"
        assert "guided_turn_answered" not in tool_names, (
            f"guided_turn_answered must NOT be emitted on the exit-from-COMPLETED path (no turn is being answered); got: {tool_names}"
        )


# ---------------------------------------------------------------------------
# Pre-condition violations — respond without prior GET /guided
# ---------------------------------------------------------------------------


class TestRespondPreconditions:
    def test_respond_without_prior_get_auto_seeds_and_succeeds(self, composer_test_client: TestClient) -> None:
        """POST /guided/respond before GET /guided auto-seeds the TurnRecord.

        Per the May 15 design (commit c4e2f69cd), respond auto-seeds the
        current step's TurnRecord when history is empty so a client that
        skips the GET still produces a complete, auditable turn record on
        the server side.  This replaces the old pre-condition that rejected
        respond without a prior GET — the rejection was a client-protocol
        nicety, but at the cost of making the respond surface fragile to
        race conditions and dropped GETs.  The auto-seed runs inside the
        same compose-lock as the response application, so the seed and the
        answer land atomically.
        """
        session_id = _create_session(composer_test_client)
        # Do NOT call _get_guided — verify respond still works.

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 200, resp.json()
        body = resp.json()
        # Auto-seeded step_1 record must be present in the persisted history.
        step_1 = next(
            (r for r in body["guided_session"]["history"] if r["step"] == "step_1_source"),
            None,
        )
        assert step_1 is not None, body["guided_session"]["history"]
        assert step_1["emitter"] == "server"
        # The answer must have been applied (response_hash populated).
        assert step_1["response_hash"] is not None

    def test_respond_on_fresh_session_succeeds_via_latent_guided(self, composer_test_client: TestClient) -> None:
        """POST /guided/respond on a fresh session uses the latent guided session.

        ``_initial_composition_state_with_guided_session`` always pre-attaches
        a latent :class:`GuidedSession.initial` so every server lazy-create
        branch reaches a uniformly-shaped state (commit ba9ffefd0 docstring,
        line 1903ff).  The freeform-default frontend means users may never
        click "Switch to guided", but a direct POST respond still produces a
        valid answer surface; the 400-on-no-guided-mode path is unreachable
        from the public API and is retained only as a defensive guard
        against future state-loading regressions that would set
        ``guided_session=None``.
        """
        session_id = _create_session(composer_test_client)

        resp = _respond_raw(
            composer_test_client,
            session_id,
            chosen=["csv"],
        )

        assert resp.status_code == 200, resp.json()


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


# ---------------------------------------------------------------------------
# Unwind-path audit disposition flag (regression)
#
# The guided ``finally`` blocks drain the recorder on BOTH the success path
# (``primary_exc is None``) and the exception-unwind path (``else``). The
# ``plugin_crash_pending`` flag selects the audit-persist disposition:
#   - False  -> success disposition: a persist failure is a Tier-1 audit
#               corruption that MUST raise ``AuditIntegrityError``.
#   - True   -> unwind disposition: record via the "persist failed during
#               unwind" counter + slog and CONTINUE, so the persist failure
#               does not mask the primary exception the operator needs to see.
#
# The bug: the unwind (``else``) branch passed ``plugin_crash_pending=False``
# at three handlers (get_guided, post_guided_respond, post_guided_chat). On
# unwind that wrongly selects the success disposition — incrementing the
# Tier-1 violation counter and raising a (then-swallowed) AuditIntegrityError,
# i.e. falsely recording an audit-corruption event during a routine
# best-effort unwind persist. post_guided_chat's sibling _persist_chat_turns
# call in the SAME branch already used ``request_unwinding=True``, proving the
# disposition was set inconsistently within one branch. All three sites are
# fixed to pass ``True`` on unwind; this pins post_guided_respond, the
# canonical site flagged in review.
# ---------------------------------------------------------------------------


class TestUnwindAuditDispositionFlag:
    def test_unwind_persist_helpers_receive_plugin_crash_pending_true(
        self,
        composer_test_client: TestClient,
        monkeypatch,
    ) -> None:
        """On the exception-unwind path, both audit-persist helpers must be
        invoked with ``plugin_crash_pending=True`` (record-and-continue), never
        ``False`` (the success disposition that raises AuditIntegrityError and
        masks the primary failure).

        Forces the unwind branch by patching ``_dispatch_guided_respond`` to
        raise, and captures the disposition flag by patching the two persist
        helpers in the composer module namespace (where the finally block
        resolves them at call time). The flag is the caller-side contract this
        fix corrects.
        """
        from elspeth.web.composer.guided.errors import InvariantError
        from elspeth.web.sessions.routes import composer as composer_module

        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        captured: list[tuple[str, bool]] = []

        async def _raise_in_dispatcher(*args, **kwargs):
            raise InvariantError("synthetic primary failure to force the unwind branch")

        async def _spy_tool(*args, plugin_crash_pending, **kwargs):
            captured.append(("tool_invocations", plugin_crash_pending))

        async def _spy_llm(*args, plugin_crash_pending, **kwargs):
            captured.append(("llm_calls", plugin_crash_pending))

        monkeypatch.setattr(composer_module, "_dispatch_guided_respond", _raise_in_dispatcher)
        monkeypatch.setattr(composer_module, "_persist_tool_invocations", _spy_tool)
        monkeypatch.setattr(composer_module, "_persist_llm_calls", _spy_llm)

        resp = _respond_raw(composer_test_client, session_id, chosen=["csv"])

        # The primary failure surfaces as a sanitized 500 — it is NOT masked.
        assert resp.status_code == 500, resp.text

        # Both finally-block persist helpers ran on the unwind path with the
        # record-and-continue disposition...
        assert ("tool_invocations", True) in captured, captured
        assert ("llm_calls", True) in captured, captured
        # ...and NEITHER was called with the success disposition (the bug).
        assert ("tool_invocations", False) not in captured, captured
        assert ("llm_calls", False) not in captured, captured
