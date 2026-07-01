"""Integration tests for Phase 5 Task 5.3: full-session audit emission contract.

Asserts that spec §9.1 audit events appear (or do NOT appear) across two
complete session lifecycles:

1. Recipe-match happy path:
   - guided_turn_emitted fires at least once
   - guided_turn_answered fires at least once
   - guided_step_advanced fires at least once
   - guided_dropped_to_freeform fires ZERO times

2. Auto-drop path (chain solver exhausted):
   - guided_dropped_to_freeform fires at least once
   - The drop event's ``drop_reason == "solver_exhausted"``
   - The drop event's ``prev_step == "step_3_transforms"``
   - The drop event carries a ``validation_result`` field (spec §9.1 MUST)

These tests cover the audit-emission contract, not HTTP response semantics;
for response-shape tests see test_respond.py and test_auto_drop.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Guided audit discriminators (spec §9.1)
# ---------------------------------------------------------------------------

_GUIDED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "guided_turn_emitted",
        "guided_turn_answered",
        "guided_step_advanced",
        "guided_dropped_to_freeform",
    }
)

# ---------------------------------------------------------------------------
# Low-level helpers (mirrors test_auto_drop.py — no cross-file imports)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "audit-emission-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    """Seed a CSV blob for recipe-match path.  Returns (blob_id, storage_path).

    Uses ``text`` + ``category`` columns so the ``classify-rows-llm-jsonl``
    recipe predicate is satisfied when the user declares ``category`` as a
    required field (the classify keyword is a substring of "category").
    """
    content = "text,category\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _seed_blob_no_recipe(client: TestClient, session_id: str) -> tuple[str, str]:
    """Seed a CSV blob for the auto-drop path (no recipe match).

    Uses ``text`` + ``note`` columns — ``note`` does not satisfy any
    classify/label/category keyword, so no recipe matches and the
    chain-solver path fires.
    """
    content = "text,note\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


# ---------------------------------------------------------------------------
# Audit-extraction helpers
# ---------------------------------------------------------------------------


def _get_tool_messages(client: TestClient, session_id: str) -> list:
    """Return audit-bearing messages for this session.

    Post Phase-1B/rev-4 (`_persist_tool_invocations`), audit rows split by
    parent linkage:
    - ``role='tool'`` when paired with a parent assistant message (compose
      success path).
    - ``role='audit'`` when no parent assistant exists (convergence /
      preflight / guided-endpoint paths — see ``_persist_tool_invocations``
      docstring at routes.py:850).

    Guided-mode audit invocations (``guided_turn_emitted`` /
    ``guided_turn_answered`` / ``guided_step_advanced`` /
    ``guided_dropped_to_freeform``) ride on the ``role='audit'`` channel
    because the guided GET/POST endpoints emit the events server-side
    without a paired assistant chat message. Filtering only ``role='tool'``
    here would silently exclude every guided event the tests assert on.
    """
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    return [m for m in msgs if m.role in ("tool", "audit")]


def _extract_guided_invocations(client: TestClient, session_id: str) -> dict[str, list[dict]]:
    """Return a mapping of guided-mode tool_name → list of parsed argument payloads.

    Filters to the four guided-mode discriminators from spec §9.1.  Other tool
    names (``set_source``, ``set_output``, ``apply_pipeline_recipe``, etc.) are
    excluded so callers only see guided-protocol events.

    Returns:
        A dict keyed by tool_name, each value a list of parsed argument dicts
        (``arguments_canonical`` decoded from JSON).  Missing keys indicate zero
        events of that type.
    """
    tool_messages = _get_tool_messages(client, session_id)
    result: dict[str, list[dict]] = {name: [] for name in _GUIDED_TOOL_NAMES}
    for msg in tool_messages:
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            invocation = tc.get("invocation", {})
            tool_name = invocation.get("tool_name")
            if tool_name not in _GUIDED_TOOL_NAMES:
                continue
            args_canonical = invocation.get("arguments_canonical", "{}")
            result[tool_name].append(json.loads(args_canonical))
    return result


def _assert_payload_ref_retrievable(client: TestClient, *, payload_ref: str, expected_hash: str) -> None:
    assert payload_ref, "guided audit payload refs must be non-empty"
    assert payload_ref == expected_hash
    stored = client.app.state.payload_store.retrieve(payload_ref)
    assert stored, "guided audit payload ref must retrieve stored canonical payload bytes"


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> tuple[dict, str, str]:
    """Drive the wizard to the Step 3 ``propose_chain`` turn (no recipe).

    Uses ``required_fields=["text"]`` (no classify/label/category keyword) so
    no recipe matches and the chain-solver entry seam fires.

    Returns (response_body_at_step_3, blob_id, output_path).
    """
    blob_id, storage_path = _seed_blob_no_recipe(client, session_id)
    output_path = _outputs_path(client, "out_drop.jsonl")

    _get_guided(client, session_id)
    _respond(client, session_id, chosen=["csv"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {"path": storage_path, "schema": {"mode": "observed"}},
            "observed_columns": ["text", "note"],
            "sample_rows": [{"text": "Hello world", "note": "greeting"}],
        },
    )
    _respond(client, session_id, chosen=["json"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": output_path,
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            "observed_columns": [],
            "sample_rows": [],
        },
    )
    body = _respond(
        client,
        session_id,
        chosen=["text"],
        custom_inputs=[],
    )
    return body, blob_id, output_path


def _fake_llm_bad_plugin() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a nonexistent plugin (validation will fail)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": "definitely_not_a_real_plugin_xyzzy",
                                                    "options": {},
                                                    "rationale": "stub: guaranteed to fail validation",
                                                }
                                            ],
                                            "why": "stub that forces preview_pipeline failure",
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


def _fake_llm_passthrough() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a valid single-step passthrough chain."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": "passthrough",
                                                    "options": {"schema": {"mode": "observed"}},
                                                    "rationale": "identity chain",
                                                }
                                            ],
                                            "why": "rows already match the sink schema",
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


def _chat_propose_chain(client: TestClient, session_id: str) -> dict:
    """Drive the per-stage transforms chat that emits the ``propose_chain`` turn.

    CHANGE 1: committing the sink no longer auto-runs ``solve_chain``.  The
    sink-commit now lands at ``step_3_transforms`` with ``next_turn: null`` and
    NO proposal.  The transform chain — and its ``guided_turn_emitted`` audit —
    now ride the per-stage transforms chat posted here, which calls
    ``solve_chain`` and therefore fires the SAME
    ``chain_solver._litellm_acompletion`` mock the caller has patched.

    Any transforms-intent string works; the mock ignores it.  The call MUST be
    made inside the caller's ``with patch(...)`` block so the mock covers it.

    Asserting ``propose_chain`` here (not just status 200) is diagnostic: if
    ``solve_chain`` ever fell through to advisory prose the proposal would stay
    unset and a downstream accept would 400 with "No turn has been emitted".
    The bad-plugin proposal surfaced as a ``propose_chain`` turn in the old
    auto-build path, so it must surface the same way through the chat.
    """
    resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
    )
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["guided_session"]["step"] == "step_3_transforms", body
    assert body["next_turn"] is not None, body
    assert body["next_turn"]["type"] == "propose_chain", body
    return body


# ---------------------------------------------------------------------------
# Test 1 — Recipe-match happy path audit emission
# ---------------------------------------------------------------------------


class TestWireAuditEmission:
    def test_direct_chain_accept_to_wire_emits_user_advanced(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)
            # Sink-commit no longer auto-builds; the propose_chain turn now
            # rides the per-stage transforms chat (same chain_solver mock).
            _chat_propose_chain(composer_test_client, session_id)
            body = _respond(composer_test_client, session_id, chosen=["accept"])

        assert body["next_turn"]["type"] == "confirm_wiring"
        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert any(
            event["prev_step"] == "step_3_transforms" and event["next_step"] == "step_4_wire" and event["reason"] == "user_advanced"
            for event in guided_invocations["guided_step_advanced"]
        )

    def test_repair_success_to_wire_emits_auto_advanced(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)
            # The (bad) proposal now rides the per-stage transforms chat.
            _chat_propose_chain(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_passthrough()],
        ):
            body = _respond(composer_test_client, session_id, chosen=["accept"])

        assert body["next_turn"]["type"] == "confirm_wiring"
        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert any(
            event["prev_step"] == "step_3_transforms" and event["next_step"] == "step_4_wire" and event["reason"] == "auto_advanced"
            for event in guided_invocations["guided_step_advanced"]
        )


# ---------------------------------------------------------------------------
# Test 2 — Auto-drop path audit emission
# ---------------------------------------------------------------------------


class TestAutoDropAuditEmission:
    """Assert spec §9.1 audit events fire correctly on the solver-exhausted auto-drop path.

    Both the initial chain-solver call and the repair call propose an
    invalid plugin name.  The wizard auto-drops to freeform after the
    repair fails.
    """

    def test_auto_drop_emits_drop_event_with_correct_fields(self, composer_test_client: TestClient) -> None:
        """guided_dropped_to_freeform fires with required fields on solver-exhausted path.

        Asserts:
        - At least one guided_dropped_to_freeform event is present.
        - drop_reason == "solver_exhausted"
        - prev_step == "step_3_transforms"
        - validation_result is present (spec §9.1 MUST when drop_reason=solver_exhausted)
        - validation_result.is_valid is False
        - validation_result.errors is present (may be empty list, not absent)
        """
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 to step_3_transforms, then build the (bad) chain
        # via the per-stage transforms chat (CHANGE 1: sink-commit no longer
        # auto-builds — the propose_chain turn rides the chat solve).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)
            _chat_propose_chain(composer_test_client, session_id)

        # Accept the (bad) chain — initial commit fails, repair also fails → auto-drop.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_bad_plugin()],
        ):
            final_resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert final_resp.status_code == 200, final_resp.json()
        body = final_resp.json()

        # Confirm the session actually dropped (not a different terminal).
        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in response body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform", f"expected terminal.kind=exited_to_freeform, got: {terminal}"

        # Extract guided-mode audit events.
        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        drop_events = guided_invocations["guided_dropped_to_freeform"]

        assert len(drop_events) >= 1, (
            f"expected at least one guided_dropped_to_freeform audit event; got none. All guided events: {guided_invocations}"
        )

        drop_args = drop_events[0]

        assert drop_args["drop_reason"] == "solver_exhausted", f"expected drop_reason=solver_exhausted; got: {drop_args}"

        assert drop_args["prev_step"] == "step_3_transforms", f"expected prev_step=step_3_transforms; got: {drop_args}"

        assert "validation_result" in drop_args, (
            f"spec §9.1 requires validation_result on solver_exhausted drops; field is absent. Drop event payload: {drop_args}"
        )

        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict), f"validation_result must be a dict; got {type(validation_result)}: {validation_result}"
        assert validation_result["is_valid"] is False, (
            f"validation_result.is_valid must be False (pipeline was invalid); got: {validation_result}"
        )
        assert "errors" in validation_result, f"validation_result must have an 'errors' key; got: {validation_result}"

    def test_auto_drop_also_emits_turn_answered_events(self, composer_test_client: TestClient) -> None:
        """guided_turn_answered fires at least once on the auto-drop path (responds were made)."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)
            # Build the (bad) chain via the chat so the accept below genuinely
            # reaches the auto-drop path this test claims to exercise.
            _chat_propose_chain(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_bad_plugin()],
        ):
            composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        guided_invocations = _extract_guided_invocations(composer_test_client, session_id)
        assert len(guided_invocations["guided_turn_answered"]) >= 1, (
            f"expected at least one guided_turn_answered event on auto-drop path; got none. All guided events: {guided_invocations}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Rejection-path audit drain (PR-review B3 regression pin)
# ---------------------------------------------------------------------------


class TestRejectionPathAuditDrain:
    """Pin the PR-review B3 invariant: handler-level recorders are drained on
    every exit path, including ``raise HTTPException`` rejections.

    Before the B3 fix, ``post_guided_respond`` created a ``BufferingRecorder``,
    wrote ``guided_turn_answered`` to it, and then only persisted on the
    success path.  Every 400/409/500 raised between the emit and the
    success-path drain silently dropped buffered audit events — a CLAUDE.md
    auditability violation ("rejected requests are facts worth recording").

    These tests verify:

    1. **The load-bearing pin** — a 400 raised AFTER ``emit_turn_answered``
       still persists the buffered event.  Concretely: POST a malformed
       ``INSPECT_AND_CONFIRM`` response (``edited_values=None``), which
       triggers ``ValueError`` inside ``step_advance`` → HTTP 400.  The
       ``guided_turn_answered`` event for this rejected attempt must land
       in the audit DB so an auditor can reconstruct "this client tried
       to advance with payload X and was rejected."

    2. **The 409 path** — terminal-state rejection fires BEFORE
       ``emit_turn_answered``, so the recorder is empty on this exit.  The
       drain must still be a clean no-op (no crash from draining an empty
       buffer).

    3. **The GET 400 path** — freeform-session rejection in ``get_guided``
       fires immediately after recorder construction, before any emit.
       The drain must be a clean no-op.

    4. **Audit-persist failure does NOT suppress the original HTTPException**
       — if ``_persist_tool_invocations`` itself raises inside ``finally``,
       Python's default behaviour would let the inner exception replace
       the in-flight 400/409.  The inner ``try/except`` around the persist
       call prevents that.  The audit-persist failure is logged via
       ``slog.error`` under the audit-system-failure exemption (CLAUDE.md
       primacy order, matching the B1 convention).
    """

    def test_post_guided_respond_400_persists_prior_emits(self, composer_test_client: TestClient) -> None:
        """A 400 raised AFTER ``emit_turn_answered`` still persists the buffered event.

        Drives the wizard to the INSPECT_AND_CONFIRM turn (after ``chosen=["csv"]``),
        then sends ``edited_values=None`` which ``step_advance`` rejects with
        ``ValueError`` → HTTP 400.  Asserts the recorder's
        ``guided_turn_answered`` event landed in the audit DB despite the
        rejection.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        # Advance past STEP_1 SINGLE_SELECT → INSPECT_AND_CONFIRM turn.
        _respond(composer_test_client, session_id, chosen=["csv"])

        # Snapshot the answered-event count BEFORE the rejected request so
        # we can prove the rejected request added exactly one more.
        baseline = len(_extract_guided_invocations(composer_test_client, session_id)["guided_turn_answered"])

        # Send a malformed INSPECT_AND_CONFIRM response: edited_values=None.
        # step_advance raises ValueError("inspect_and_confirm response must
        # carry edited_values; got None") which the handler maps to 400.
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )
        assert resp.status_code == 400, resp.json()

        # The load-bearing assertion: the guided_turn_answered emitted before
        # the rejection must be in the audit DB.  The 400 fires either from
        # ``step_advance`` (state-machine guard) or from the step-1 handler
        # (handle_step_1_source / dispatcher ValueError), both of which run
        # AFTER ``emit_turn_answered``.  Either way the buffered event
        # must persist.
        post_reject = _extract_guided_invocations(composer_test_client, session_id)["guided_turn_answered"]
        assert len(post_reject) == baseline + 1, (
            f"PR-review B3 regression: the guided_turn_answered event for "
            f"the rejected INSPECT_AND_CONFIRM turn was dropped on the 400 "
            f"path. baseline={baseline} post_reject={len(post_reject)} "
            f"(expected baseline+1). Response body: {resp.json()}"
        )

    def test_post_guided_respond_409_terminal_drains_cleanly(self, composer_test_client: TestClient) -> None:
        """The 409 terminal-state path drains a clean (empty-or-prior-only) recorder.

        The 409 is raised BEFORE ``emit_turn_answered`` for this request — see
        the handler: terminal-state rejection sits between
        ``service.get_current_state`` and the emit.  So the recorder has no
        new event from this rejected request; the test simply verifies that
        the drain doesn't crash (i.e. empty drains are safe and the original
        HTTPException is not masked).
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        # Exit immediately so the next respond hits the 409 terminal guard.
        _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

        # Second respond on the now-terminal session: 409.
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert resp.status_code == 409, resp.json()

    def test_get_guided_400_freeform_session_drains_empty_recorder(self, composer_test_client: TestClient) -> None:
        """GET /guided on a freeform session drains an empty recorder without crashing.

        Constructs a session with composition state that has no
        ``guided_session`` attached (a freeform session), then GETs /guided.
        The handler raises 400 immediately after recorder construction —
        no audit emit has occurred, so the recorder is empty.  The finally
        block must drain it without crashing, and the 400 must reach the
        client (not be replaced by a finally-block exception).
        """
        import asyncio
        from uuid import UUID

        # Build a freeform session: create then overwrite the composition
        # state with one whose composer_meta has no ``guided_session`` key.
        session_id = _create_session(composer_test_client)
        service = composer_test_client.app.state.session_service

        # Persist a freeform composition state directly.
        from elspeth.web.sessions.protocol import CompositionStateData

        freeform_state = CompositionStateData(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata_={"name": "Untitled Pipeline", "description": ""},
            is_valid=False,
            validation_errors=None,
            composer_meta={},  # No guided_session key → freeform.
        )
        asyncio.run(service.save_composition_state(UUID(session_id), freeform_state, provenance="session_seed"))

        resp = composer_test_client.get(f"/api/sessions/{session_id}/guided")
        assert resp.status_code == 400, resp.json()
        assert "not in guided mode" in resp.json()["detail"]

    def test_audit_persist_failure_does_not_suppress_http_exception(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``_persist_tool_invocations`` raises inside finally, the
        original HTTPException must still surface.  The inner
        ``try/except`` in the finally block prevents Python's default
        behaviour (the inner raise replaces the outer) from masking the
        intended 400/409.

        Concretely: drive to INSPECT_AND_CONFIRM, then patch
        ``_persist_tool_invocations`` to raise.  Send a malformed payload
        that triggers the 400 path.  Assert the response is still 400 —
        not 500 from the persist failure.
        """
        from elspeth.web.sessions.routes.composer import guided as composer_module

        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])

        async def _raising_persist(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated audit-system failure")

        # _persist_tool_invocations is called bare from the composer.py
        # /guided/respond handler; patch the calling module.
        monkeypatch.setattr(composer_module, "_persist_tool_invocations", _raising_persist)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )

        # The original 400 must surface — NOT a 500 from the persist failure.
        assert resp.status_code == 400, (
            f"PR-review B3 regression: audit-persist failure inside finally "
            f"replaced the original HTTPException. Expected 400, got "
            f"{resp.status_code}: {resp.json()}"
        )

    def test_audit_persist_failure_logs_via_slog(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The audit-persist failure is logged via slog.error under the
        audit-system-failure exemption.  Pins the
        ``guided.audit_persist_failed_during_exception_handling`` event
        name and confirms the failure is observable to operators.

        Uses ``structlog.testing.capture_logs`` because the project uses
        structlog (not stdlib logging) for ``slog.error``; ``caplog``
        does not see structlog events unless the logger is bridged.
        """
        from structlog.testing import capture_logs

        from elspeth.web.sessions.routes.composer import guided as composer_module

        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])

        async def _raising_persist(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated audit-system failure")

        # _persist_tool_invocations is called bare from the composer.py
        # /guided/respond handler; patch the calling module.
        monkeypatch.setattr(composer_module, "_persist_tool_invocations", _raising_persist)

        with capture_logs() as cap_logs:
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"edited_values": None},
            )
            # Original HTTPException still surfaces.
            assert resp.status_code == 400

        matches = [entry for entry in cap_logs if entry.get("event") == "guided.audit_persist_failed_during_exception_handling"]
        assert matches, f"expected the audit-persist-failure slog event; got entries: {cap_logs}"

        # Match the B1 convention: exc_class + frames + site + session_id +
        # user_id, never str(exc) or exc_info.  Pin the field set so a
        # future refactor that swaps in str(exc) is caught immediately.
        entry = matches[0]
        assert entry["exc_class"] == "RuntimeError", entry
        assert entry["site"] == "post_guided_respond", entry
        assert "frames" in entry and isinstance(entry["frames"], tuple), entry
        # No raw exception message field — that's the leak vector the B1
        # convention forbids.
        forbidden_keys = {"exc_message", "exception", "exc_info"}
        assert not (forbidden_keys & entry.keys()), f"slog entry has forbidden message/exc_info field: {entry}"

    def test_audit_persist_failure_preserves_original_error_if_slog_raises(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken logger must not replace the original guided-response error.

        The finally-drain ``except`` block has already reached the last-resort
        logging channel. If structlog itself raises there, no further channel
        exists that can safely report the audit-of-audit failure, so the route
        must preserve the original HTTPException instead of surfacing a logger
        pipeline error to the browser.
        """
        from elspeth.web.sessions.routes.composer import guided as composer_module

        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])

        async def _raising_persist(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated audit-system failure")

        def _raising_slog_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated logger failure")

        # _persist_tool_invocations is called bare from composer.py; patch the
        # calling module. ``slog`` is the same shared logger object across
        # modules (re-exported from _helpers), so patching ``.error`` on
        # composer_module.slog mutates that shared object — the route handler's
        # slog.error call sees it regardless of which module reference is used.
        monkeypatch.setattr(composer_module, "_persist_tool_invocations", _raising_persist)
        monkeypatch.setattr(composer_module.slog, "error", _raising_slog_error)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )

        assert resp.status_code == 400, resp.json()
