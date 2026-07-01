"""Integration tests for Phase 5 Task 5.1: auto-drop to freeform on solver-exhausted.

Drives a full session through Steps 1 + 2 + Step 3 PROPOSE_CHAIN and then
exercises the repair-then-drop flow:

- Both-fail test: first LLM call proposes a chain that fails preview, second
  (repair) also fails → HTTP 200 with terminal kind=exited_to_freeform,
  reason=solver_exhausted; audit record for guided_dropped_to_freeform emitted.

- Repair-succeeds test: first LLM call proposes a bad chain, second proposes a
  working passthrough chain → HTTP 200 with confirm_wiring, then confirm stamps
  terminal kind=completed; no drop event emitted.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from litellm.exceptions import (
    BlockedPiiEntityError,
    BudgetExceededError,
    GuardrailInterventionNormalStringError,
    GuardrailRaisedException,
    OpenAIError,
)

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (mirrors test_step_3_e2e.py — kept local; no cross-file imports)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient, *, profile: str | None = None) -> str:
    resp = client.post("/api/sessions", json={"title": "auto-drop-test"})
    assert resp.status_code == 201, resp.json()
    session_id = resp.json()["id"]
    if profile is not None:
        start_resp = client.post(f"/api/sessions/{session_id}/guided/start", json={"profile": profile})
        assert start_resp.status_code == 200, start_resp.json()
    return session_id


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _confirm_wiring(client: TestClient, session_id: str) -> dict:
    return _respond(
        client,
        session_id,
        chosen=["confirm"],
        edited_values=None,
        custom_inputs=None,
        accepted_step_index=None,
        edit_step_index=None,
        control_signal=None,
    )


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
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


def _fake_llm_response_for_bad_plugin() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a nonexistent plugin.

    ``definitely_not_a_real_plugin_xyzzy`` causes ``_validate_plugin_name`` to
    fail in ``_execute_set_pipeline``, making ``handle_step_3_chain_accept``
    return success=False — the failure mode exercised by the repair flow.
    """
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


def _fake_llm_response_for_passthrough() -> SimpleNamespace:
    """LiteLLM-shaped response proposing a valid passthrough chain (will succeed)."""
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
                                                    "rationale": "pass rows through unchanged",
                                                }
                                            ],
                                            "why": "source rows already match sink schema",
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


def _post_step_3_transforms_chat(client: TestClient, session_id: str, message: str = "fetch each page and summarise it") -> object:
    """Send the per-stage transforms prompt via /guided/chat at STEP_3.

    Post sink->step_3 auto-build removal, the transform chain is built here
    (the STEP_3_TRANSFORMS cold-start path in ``post_guided_chat``), NOT at
    the sink commit. The ``chain_solver._litellm_acompletion`` mock the caller
    installed fires on THIS call; ``message`` is arbitrary (the mock ignores
    it). On a returning (non-raising) mock the response carries the
    ``propose_chain`` turn in ``next_turn``.
    """
    return client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": message, "step_index": "step_3_transforms"},
    )


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> tuple[dict, str, str]:
    """Drive the wizard to the Step 3 ``propose_chain`` turn.

    Returns (chat_response_body_at_step_3, blob_id, output_path).
    Uses ``required_fields=["text"]`` (no classifier keyword) so no recipe
    matches and the chain-solver entry seam fires.

    Post sink->step_3 auto-build removal: committing the sink (the final
    MULTI_SELECT_WITH_CUSTOM ``chosen=["text"]`` respond) now advances to
    STEP_3_TRANSFORMS with ``next_turn=None`` and DOES NOT solve a chain. The
    transform chain is built by the per-stage transforms prompt sent via
    ``/guided/chat`` (step_index="step_3_transforms"). The SAME
    ``chain_solver._litellm_acompletion`` mock installed by the caller's
    ``with patch(...)`` block fires on THIS chat call and produces the
    ``propose_chain`` turn.
    """
    blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out.jsonl")

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
    # Sink commit: advances to STEP_3_TRANSFORMS with next_turn=None (no
    # auto-build). The mock does NOT fire here.
    _respond(
        client,
        session_id,
        chosen=["text"],
        custom_inputs=[],
    )
    # Per-stage transforms prompt builds the chain via /guided/chat. The
    # caller's mock fires here and stages the propose_chain turn.
    resp = _post_step_3_transforms_chat(client, session_id)
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    return body, blob_id, output_path


def _get_tool_messages(client: TestClient, session_id: str) -> list:
    """Return audit-bearing messages for this session.

    Post Phase-1B/rev-4: guided audit invocations land on ``role='audit'``
    (no parent assistant) — see the docstring at the matching helper in
    test_audit_emission.py and ``_persist_tool_invocations`` at
    src/elspeth/web/sessions/routes.py:850.
    """
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    return [m for m in msgs if m.role in ("tool", "audit")]


def _extract_guided_drop_invocations(client: TestClient, session_id: str) -> list[dict]:
    """Return all guided_dropped_to_freeform invocation payloads for this session."""
    tool_messages = _get_tool_messages(client, session_id)
    drop_invocations = []
    for msg in tool_messages:
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            invocation = tc.get("invocation", {})
            if invocation.get("tool_name") == "guided_dropped_to_freeform":
                # arguments_canonical is the JSON-encoded payload dict
                args_canonical = invocation.get("arguments_canonical", "{}")
                drop_invocations.append(json.loads(args_canonical))
    return drop_invocations


# ---------------------------------------------------------------------------
# Test: both initial and repair attempts fail → auto-drop to freeform (200)
# ---------------------------------------------------------------------------


class TestAutoDropOnSolverExhausted:
    def test_both_attempts_fail_returns_200_with_terminal_exited_to_freeform(self, composer_test_client: TestClient) -> None:
        """Both chain-solver attempts fail validation → HTTP 200, terminal=exited_to_freeform."""
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 to reach PROPOSE_CHAIN.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept the (bad) chain. The initial commit fails, triggering one LLM
        # repair call (also a bad plugin).  Both fail → auto-drop.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_bad_plugin()],
        ) as mock_llm:
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        # 1. Status code must be 200 — auto-drop is a clean wizard outcome.
        assert resp.status_code == 200, resp.json()

        body = resp.json()

        # 2. Terminal in body must carry exited_to_freeform / solver_exhausted.
        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform", f"unexpected kind: {terminal}"
        assert terminal["reason"] == "solver_exhausted", f"unexpected reason: {terminal}"
        assert terminal["pipeline_yaml"] is None

        # 3. GuidedSession terminal also set correctly.
        gs = body["guided_session"]
        assert gs["terminal"]["kind"] == "exited_to_freeform"
        assert gs["terminal"]["reason"] == "solver_exhausted"

        # 4. Repair call was made exactly once (one repair attempt).
        assert mock_llm.call_count == 1

        # 5. Audit record for guided_dropped_to_freeform must be present.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, (
            f"expected exactly one guided_dropped_to_freeform audit record, got {len(drop_invocations)}: {drop_invocations}"
        )
        drop_args = drop_invocations[0]
        assert drop_args["drop_reason"] == "solver_exhausted"
        assert drop_args["prev_step"] == "step_3_transforms"
        assert "validation_result" in drop_args, f"spec §9.1 requires validation_result on solver_exhausted drops; got: {drop_args}"
        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict)
        assert validation_result["is_valid"] is False
        assert "errors" in validation_result  # may be empty list, not absent

    def test_no_next_turn_after_auto_drop(self, composer_test_client: TestClient) -> None:
        """After auto-drop the next_turn must be None — wizard is terminal."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_bad_plugin()],
        ):
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["next_turn"] is None


# ---------------------------------------------------------------------------
# Test: first attempt fails, repair succeeds → COMPLETED (no drop event)
# ---------------------------------------------------------------------------


class TestRepairSucceeds:
    def test_first_fails_repair_succeeds_returns_confirm_wiring_then_completed(self, composer_test_client: TestClient) -> None:
        """First chain fails, repair succeeds → confirm_wiring, then completed."""
        session_id = _create_session(composer_test_client, profile="tutorial")

        # Drive Steps 1 + 2 to reach PROPOSE_CHAIN (initial solve: bad plugin).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept. Initial commit fails → repair attempt returns passthrough (valid).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[_fake_llm_response_for_passthrough()],
        ) as mock_llm:
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert resp.status_code == 200, resp.json()
        body = resp.json()

        assert body["terminal"] is None
        assert body["guided_session"]["step"] == "step_4_wire"
        assert body["next_turn"]["type"] == "confirm_wiring"

        body = _confirm_wiring(composer_test_client, session_id)

        terminal = body.get("terminal")
        assert terminal is not None
        assert terminal["kind"] == "completed", f"unexpected kind: {terminal}"
        assert terminal["reason"] is None
        assert terminal["pipeline_yaml"] is not None
        assert "passthrough" in terminal["pipeline_yaml"]

        # next_turn is None — wizard is complete.
        assert body["next_turn"] is None

        # Exactly one repair call made.
        assert mock_llm.call_count == 1

        # No guided_dropped_to_freeform audit event emitted.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert drop_invocations == [], f"unexpected drop event in repair-succeeds path: {drop_invocations}"


# ---------------------------------------------------------------------------
# I2: transient LLM failure at the live solve_chain auto-drop site → 200, not 500.
# ---------------------------------------------------------------------------
#
# HISTORY — site removal (sink->step_3 auto-build removed):
# The sink commit (``step_2_sink_initial_solve``) used to auto-build the
# transform chain via ``solve_chain``; a transient there auto-dropped to
# freeform. That auto-build is GONE. The sink commit now advances to
# STEP_3_TRANSFORMS with ``next_turn=None`` and runs NO solver, and the
# transform chain is built by the per-stage transforms prompt on the
# ``/guided/chat`` STEP_3 cold-start path. That chat path is **non-load-bearing**:
# it calls a PLAIN ``solve_chain`` (NOT ``solve_chain_with_auto_drop``), and on a
# transient it catches the error, abandons the proposal, and falls through to
# advisory prose — it DOES NOT auto-drop the session to freeform. The new
# behaviour is pinned by ``test_step_3_chat_coldstart_transient_does_not_auto_drop``.
#
# The remaining LIVE ``solve_chain_with_auto_drop`` site reachable from
# ``/guided/respond`` is the Step-3 **repair** re-solve (``site="step_3_repair_solve"``,
# ``_helpers.py``): after an accept whose initial commit fails validation, the
# repair re-solve runs through the auto-drop wrapper, and a transient there
# exhausts the solver and drops to freeform. These tests therefore drive to a
# staged (bad-plugin) ``propose_chain`` and then accept so the failing mock fires
# on the repair re-solve — the live expression of the auto-drop contract these
# tests have always pinned. (A regenerate re-solve at ``site="step_3_reject_solve"``
# shares the same wrapper.)
#
# Pre-I2 the solve sites were unwrapped: a LiteLLM API/auth error, timeout, or
# malformed-response shape (IndexError on empty ``choices``) escaped as an
# unstructured 500 — bypassing ``mark_solver_exhausted`` and leaving the session
# wedged mid-step with no ``guided_dropped_to_freeform`` audit record. These
# tests pin the auto-drop contract from ``solve_chain_with_auto_drop``.


def _accept_propose_chain(client: TestClient, session_id: str) -> object:
    """Accept the staged propose_chain turn (triggers commit + repair re-solve)."""
    return client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={"chosen": ["accept"]},
    )


def _drive_to_step_3_propose_chain_bad(client: TestClient, session_id: str) -> None:
    """Drive to a staged (bad-plugin) ``propose_chain`` turn ready for accept.

    The bad-plugin response is a structurally valid ``propose_chain`` (it parses
    cleanly into a ``ChainProposal``); the plugin invalidity only surfaces at
    preview/commit time. So the drive's ``/guided/chat`` solve SUCCEEDS (no drop,
    one success llm_call audit row) and stages the proposal. The subsequent
    accept fails its commit and triggers the live ``step_3_repair_solve`` re-solve
    — the seam these I2 tests provoke.
    """
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_response_for_bad_plugin(),
    ):
        _drive_to_step_3_propose_chain(client, session_id)


def _drive_to_step_2_sink_initial_solve_pre_call(client: TestClient, session_id: str) -> str:
    """Drive to the sink-commit request, then on to STEP_3_TRANSFORMS.

    Returns ``session_id`` after Steps 1 and 2 SINGLE_SELECT + SCHEMA_FORM have
    been answered. The NEXT respond (MULTI_SELECT ``chosen=['text']``, via
    :func:`_post_step_2_sink_intent`) commits the sink and advances to
    STEP_3_TRANSFORMS with ``next_turn=None`` — it no longer runs a solver
    (the sink->step_3 auto-build was removed).
    """
    _blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out.jsonl")

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
    return session_id


def _post_step_2_sink_intent(client: TestClient, session_id: str) -> object:
    """Commit the sink (MULTI_SELECT) → advance to STEP_3_TRANSFORMS.

    Post auto-build removal this runs NO solver; it returns 200 with the step
    at STEP_3_TRANSFORMS and ``next_turn=None``.
    """
    return client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={"chosen": ["text"], "custom_inputs": []},
    )


class TestI2ChainSolverTransientFailure:
    """Pin: external-LLM failures at each solve_chain site auto-drop cleanly.

    "Transient" in the class name is historical — the class now covers two
    overlapping categories that both route through SOLVER_EXHAUSTED auto-drop:

      * **Transient** — LiteLLM provider/network errors, timeouts, malformed
        response shape at the LiteLLM-response level (``IndexError`` from
        empty ``choices``, ``AttributeError`` from missing ``message``,
        ``json.JSONDecodeError`` from invalid tool-call arguments).  These
        are retry-class failures in principle, but the wrapper single-shots
        the drop.

      * **LLM-shape violations** — the LLM responded but produced an
        ``emit_turn`` shape that violated the chain-solver contract (wrong
        tool name, wrong ``turn_type``, missing/malformed ``payload``).
        Routed via :class:`ChainSolverResponseShapeError`, sibling-bucketed
        with the LiteLLM-level shape failures because both are "external
        system produced unexpected response."

    The category distinction matters for the spec §5.4 retry-then-
    PROTOCOL_VIOLATION flow (filed as observation
    ``elspeth-obs-8e5b614ca3``) but does not change current behaviour.

    SITE: the original ``step_2_sink_initial_solve`` site no longer exists (the
    sink->step_3 auto-build was removed). The live ``solve_chain_with_auto_drop``
    site reachable from ``/guided/respond`` is the Step-3 **repair** re-solve
    (``site="step_3_repair_solve"``), so these tests now drive to a staged
    (bad-plugin) ``propose_chain`` (:func:`_drive_to_step_3_propose_chain_bad`)
    and ACCEPT it (:func:`_accept_propose_chain`): the initial commit fails
    validation and the repair re-solve fires the failing mock. ``solve_chain``
    is the same function regardless of site, so every failure mode below raises
    identically on the repair re-solve. The (non-auto-dropping) chat cold-start
    path is pinned separately by
    ``test_step_3_chat_coldstart_transient_does_not_auto_drop``.

    Patches ``chain_solver._litellm_acompletion`` (the seam the existing
    auto-drop tests use) to provoke each failure mode. Asserts:
      - HTTP 200 (not 500) with terminal=exited_to_freeform / solver_exhausted.
      - ``guided_dropped_to_freeform`` audit emitted exactly once with the
        ``validation_result`` payload mandated by spec §9.1. (The drive's
        bad-plugin chat solve succeeds and emits no drop; the single drop comes
        from the repair re-solve.)
      - No exception ``str(exc)`` leaks into the response body.
      - ``InvariantError`` from inside ``solve_chain``'s callee chain (a
        genuine server-side invariant violation, NOT an LLM shape failure)
        still propagates as 500 — real server bugs must crash.
    """

    def test_step_3_chat_coldstart_transient_does_not_auto_drop(self, composer_test_client: TestClient) -> None:
        """Contrast case (the removed site's new behavior): a transient at the
        STEP_3 ``/guided/chat`` cold-start solve does NOT auto-drop.

        The sink->step_3 auto-build (old ``step_2_sink_initial_solve`` auto-drop
        site) is gone. The transform chain is now built on the
        ``/guided/chat`` STEP_3 cold-start path, which calls a PLAIN
        ``solve_chain`` (NOT ``solve_chain_with_auto_drop``). On a transient it
        abandons the proposal and falls through to advisory prose via the
        non-load-bearing ``solve_step_chat_with_auto_drop`` — the session is
        NOT terminated to freeform and NO ``guided_dropped_to_freeform`` is
        emitted.

        Double-mock: ``chain_solver._litellm_acompletion`` raises (cold-start
        solve abandons the proposal) AND ``chat_solver._litellm_acompletion``
        raises a LiteLLM-typed error (so the advisory ``solve_step_chat`` returns
        the synthetic-unavailable message → clean 200). A single mock would 500
        on the unmocked advisory ``solve_step_chat`` real network call.
        """
        from litellm.exceptions import APIError as LiteLLMAPIError

        session_id = _create_session(composer_test_client)
        _drive_to_step_2_sink_initial_solve_pre_call(composer_test_client, session_id)
        # Commit the sink → advance to STEP_3_TRANSFORMS (no solver runs here).
        sink_resp = _post_step_2_sink_intent(composer_test_client, session_id)
        assert sink_resp.status_code == 200, sink_resp.json()
        assert sink_resp.json()["guided_session"]["step"] == "step_3_transforms"
        assert sink_resp.json()["next_turn"] is None

        api_err = LiteLLMAPIError(
            status_code=500,
            message="cold-start solve failed: SECRET_API_KEY=sk-leaks-here",
            llm_provider="openai",
            model="gpt-4o",
        )
        chat_err = LiteLLMAPIError(
            status_code=500,
            message="advisory chat failed: SECRET_API_KEY=sk-leaks-here",
            llm_provider="openai",
            model="gpt-4o",
        )
        with (
            patch(
                "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=api_err,
            ),
            patch(
                "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=chat_err,
            ),
        ):
            resp = _post_step_3_transforms_chat(composer_test_client, session_id)

        # Non-load-bearing: a transient here is a 200 advisory fall-through, NOT
        # an auto-drop. The session stays at STEP_3_TRANSFORMS; no terminal.
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert body["terminal"] is None, f"chat cold-start transient must NOT auto-drop; got terminal={body['terminal']}"
        assert body["guided_session"]["terminal"] is None
        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"] is None
        # The advisory fall-through surfaces the synthetic-unavailable message.
        assert "unavailable" in body["assistant_message"].lower(), body["assistant_message"]

        # No drop event — the chat path never marks the solver exhausted.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert drop_invocations == [], f"chat cold-start transient must not emit a drop; got {drop_invocations}"

        # No-leak: neither the cold-start nor the advisory error str escapes.
        body_text = json.dumps(body)
        assert "SECRET_API_KEY" not in body_text, "exc str leaked into response body"
        assert "sk-leaks-here" not in body_text, "exc str leaked into response body"

    def test_step_3_repair_solve_api_error_auto_drops(self, composer_test_client: TestClient) -> None:
        """Repair re-solve: a LiteLLM APIError on the repair attempt auto-drops."""
        from litellm.exceptions import APIError as LiteLLMAPIError

        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        api_err = LiteLLMAPIError(
            status_code=500,
            message="upstream auth failure: SECRET_API_KEY=sk-leaks-here",
            llm_provider="openai",
            model="gpt-4o",
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=api_err,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        # Must be 200 (auto-drop is a clean wizard outcome) not 500.
        assert resp.status_code == 200, resp.json()
        body = resp.json()

        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

        # No-leak invariant: the LiteLLM error message embedded a fake
        # secret-shaped string. The response body must not echo it.
        body_text = json.dumps(body)
        assert "SECRET_API_KEY" not in body_text, "exc str leaked into response body"
        assert "sk-leaks-here" not in body_text, "exc str leaked into response body"

        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, f"expected exactly one drop audit event; got {drop_invocations}"
        drop_args = drop_invocations[0]
        assert drop_args["drop_reason"] == "solver_exhausted"
        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict)
        assert validation_result["is_valid"] is False
        errors = validation_result["errors"]
        assert isinstance(errors, list) and len(errors) == 1
        # The wrapper records the exception class only, in the structured
        # ``error_class`` field — no str(exc), and no free-form ``message``
        # (the guided audit emitter drops ``message`` by allowlist).
        assert errors[0]["error_class"] == "APIError", errors
        assert "message" not in errors[0], errors

    @pytest.mark.parametrize(
        ("exc_factory", "exc_name"),
        [
            (
                lambda: BudgetExceededError(
                    current_cost=10.0,
                    max_budget=5.0,
                    message="budget exhausted: SECRET_API_KEY=sk-leaks-here",
                ),
                "BudgetExceededError",
            ),
            (
                lambda: BlockedPiiEntityError(entity_type="email", guardrail_name="pii"),
                "BlockedPiiEntityError",
            ),
            (
                lambda: GuardrailRaisedException(
                    guardrail_name="pii",
                    message="guardrail blocked SECRET_API_KEY=sk-leaks-here",
                ),
                "GuardrailRaisedException",
            ),
            (
                lambda: GuardrailInterventionNormalStringError(message="intervention SECRET_API_KEY=sk-leaks-here"),
                "GuardrailInterventionNormalStringError",
            ),
            (
                lambda: OpenAIError(original_exception=RuntimeError("provider SECRET_API_KEY=sk-leaks-here")),
                "OpenAIError",
            ),
        ],
    )
    def test_step_3_repair_solve_non_api_litellm_errors_auto_drop(
        self,
        composer_test_client: TestClient,
        exc_factory: Callable[[], Exception],
        exc_name: str,
    ) -> None:
        """Non-APIError LiteLLM operational failures follow the auto-drop path."""
        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=exc_factory(),
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        body = resp.json()
        terminal = body.get("terminal")
        assert terminal is not None, f"expected terminal in body, got: {body}"
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

        body_text = json.dumps(body)
        assert "SECRET_API_KEY" not in body_text, "exc str leaked into response body"
        assert "sk-leaks-here" not in body_text, "exc str leaked into response body"

        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, f"expected exactly one drop audit event; got {drop_invocations}"
        drop_args = drop_invocations[0]
        assert drop_args["drop_reason"] == "solver_exhausted"
        validation_result = drop_args["validation_result"]
        assert isinstance(validation_result, dict)
        errors = validation_result["errors"]
        assert isinstance(errors, list) and len(errors) == 1
        assert errors[0]["error_class"] == exc_name, errors
        assert "message" not in errors[0], errors

    def test_step_3_repair_solve_malformed_response_auto_drops(self, composer_test_client: TestClient) -> None:
        """Site 3: an IndexError (empty choices) on the repair re-solve auto-drops.

        Repair is already attempt #2; a transient on repair means we are done
        trying. Asserts the single-drop short-circuit — no double-emit through
        the downstream ``mark_solver_exhausted`` at the bottom of the repair block.
        """
        session_id = _create_session(composer_test_client)

        # Drive to PROPOSE_CHAIN with a bad-plugin initial proposal.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_bad_plugin(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        # Accept the bad chain → handler fails → repair call → empty choices
        # array → solve_chain raises IndexError at ``response.choices[0]``.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(choices=[]),
        ):
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"]},
            )

        assert resp.status_code == 200, resp.json()
        body = resp.json()
        terminal = body.get("terminal")
        assert terminal is not None
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

        # Exactly one drop event — the wrapper's short-circuit prevents the
        # downstream mark_solver_exhausted from firing a second one.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, (
            f"site-3 transient must emit exactly one drop event (no double-emit "
            f"through the downstream repair-failure path); got {len(drop_invocations)}"
        )
        errors = drop_invocations[0]["validation_result"]["errors"]
        assert errors[0]["error_class"] == "IndexError", errors
        assert "message" not in errors[0], errors

    def test_chain_solver_wrong_tool_name_auto_drops(self, composer_test_client: TestClient) -> None:
        """LLM-shape failure: a tool_call name other than ``emit_turn`` is an
        external-system (LLM) shape violation, not a server-side programming
        bug.  ``solve_chain`` raises :class:`ChainSolverResponseShapeError`
        and the wrapper routes the request through the SOLVER_EXHAUSTED
        auto-drop path -- HTTP 200 with terminal kind=exited_to_freeform,
        reason=solver_exhausted, and a ``guided_dropped_to_freeform`` event.

        Prior to the schema-tightening fix, this case raised
        :class:`InvariantError` from inside ``solve_chain``, which escaped
        the wrapper's caught set and surfaced as HTTP 500.  The sibling
        test ``test_invariant_error_from_solve_chain_propagates_to_500``
        below pins the wrapper's not-absorb-InvariantError contract using a
        direct ``solve_chain`` stub (decoupled from chain_solver internals).
        """
        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        # Construct a LiteLLM response with a tool_call name OTHER than
        # ``emit_turn`` — on the repair re-solve this triggers
        # ``raise ChainSolverResponseShapeError`` in ``solve_chain``, which the
        # auto-drop wrapper catches.
        wrong_name_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="not_emit_turn",
                                    arguments="{}",
                                )
                            )
                        ],
                    )
                )
            ]
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=wrong_name_response,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        terminal = resp.json()["terminal"]
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

        # The drop event must carry the exc_class in validation_result.errors
        # via the structured ``error_class`` field (per the wrapper contract) —
        # no exc str leakage, and no free-form ``message``.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drop_invocations) == 1, drop_invocations
        errors = drop_invocations[0]["validation_result"]["errors"]
        assert errors[0]["error_class"] == "ChainSolverResponseShapeError", errors
        assert "message" not in errors[0], errors

    def test_chain_solver_wrong_turn_type_auto_drops(self, composer_test_client: TestClient) -> None:
        """LLM-shape failure: a turn_type other than ``propose_chain`` is an
        external-system shape violation routed through auto-drop.

        Before the fix, the schema permitted six turn types and the consumer
        raised :class:`InvariantError` for any non-``propose_chain`` value
        (escaping as HTTP 500).  The schema-tightening reduces the allowed
        set to ``["propose_chain"]``, and the backstop routes any survivor
        through auto-drop.
        """
        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        wrong_turn_type_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="emit_turn",
                                    arguments=json.dumps({"turn_type": "single_select", "payload": {"options": []}}),
                                )
                            )
                        ],
                    )
                )
            ]
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=wrong_turn_type_response,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        terminal = resp.json()["terminal"]
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

    def test_chain_solver_missing_payload_steps_auto_drops(self, composer_test_client: TestClient) -> None:
        """LLM-shape failure: ``propose_chain`` payload missing ``steps`` key.

        Prior to the fix, ``payload["steps"]`` raised :class:`KeyError`,
        which escaped the wrapper's caught set as an unhandled exception
        and surfaced as HTTP 500.  The backstop now converts shape failures
        into :class:`ChainSolverResponseShapeError` and routes them through
        auto-drop.
        """
        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        missing_steps_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="emit_turn",
                                    arguments=json.dumps({"turn_type": "propose_chain", "payload": {"why": "no steps"}}),
                                )
                            )
                        ],
                    )
                )
            ]
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=missing_steps_response,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        terminal = resp.json()["terminal"]
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

    def test_chain_solver_malformed_steps_element_auto_drops(self, composer_test_client: TestClient) -> None:
        """LLM-shape failure: ``payload.steps`` element is not dict-coercible
        (e.g., a bare integer).  The ``tuple(dict(s) for s in steps_raw)``
        coercion raises ``TypeError``, which the backstop converts to
        :class:`ChainSolverResponseShapeError`.
        """
        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        malformed_step_response = SimpleNamespace(
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
                                            "payload": {"steps": [42], "why": "garbage step"},
                                        }
                                    ),
                                )
                            )
                        ],
                    )
                )
            ]
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=malformed_step_response,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 200, resp.json()
        terminal = resp.json()["terminal"]
        assert terminal["kind"] == "exited_to_freeform"
        assert terminal["reason"] == "solver_exhausted"

    def test_invariant_error_from_solve_chain_propagates_to_500(self, composer_test_client: TestClient) -> None:
        """The wrapper's not-absorb-:class:`InvariantError` contract is preserved.

        Stubs :func:`solve_chain` directly (not ``_litellm_acompletion``) to
        raise :class:`InvariantError` -- representing a *genuine* server-side
        invariant violation as opposed to an LLM shape failure (which now
        raises :class:`ChainSolverResponseShapeError`).  The wrapper must
        NOT absorb this class; the request surfaces as HTTP 500 with the
        B1-sanitised static detail.
        """
        from elspeth.web.composer.guided.errors import InvariantError

        session_id = _create_session(composer_test_client)
        # Drive to a staged proposal FIRST (the drive's chat solve binds
        # ``chain_solver.solve_chain``, NOT the ``_guided_solve_chain`` binding
        # patched below — so the drive is unaffected). The accept's repair
        # re-solve calls ``_guided_solve_chain.solve_chain``, which we stub to
        # raise InvariantError.
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        async def _raise_invariant(**_kwargs):
            raise InvariantError("genuine server-side invariant violation")

        with patch(
            "elspeth.web.sessions._guided_solve_chain.solve_chain",
            new=_raise_invariant,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        assert resp.status_code == 500, resp.json()
        detail = resp.json().get("detail", "")
        assert "invariant" in detail.lower(), detail
        # No drop event must have been emitted -- the wrapper short-circuits
        # before reaching mark_solver_exhausted.
        drop_invocations = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert drop_invocations == [], f"InvariantError must not trigger guided_dropped_to_freeform; got {drop_invocations}"


# ---------------------------------------------------------------------------
# End-to-end LLM-call audit persistence
#
# These tests prove the drain wiring: a solve_chain invocation records a
# ComposerLLMCall into the recorder, the route handler's ``finally`` block
# fires ``_persist_llm_calls(recorder.llm_calls)``, and the audit row lands
# in the session message table with ``_kind=llm_call_audit``.
#
# Mock-based unit tests cover the recorder side of that boundary
# (test_chain_solver.py:TestSolveChainLLMCallAudit).  These tests cover the
# persist side end-to-end against the route handler + session-service stack.
# ---------------------------------------------------------------------------


def _extract_llm_call_audits(client: TestClient, session_id: str) -> list[dict]:
    """Return all ``_kind=llm_call_audit`` content payloads for this session.

    Mirrors :func:`_extract_guided_drop_invocations` but filters on the
    LLM-call audit channel (persisted by ``_persist_llm_calls``) rather
    than the tool-invocation channel (persisted by
    ``_persist_tool_invocations``).
    """
    tool_messages = _get_tool_messages(client, session_id)
    rows: list[dict] = []
    for msg in tool_messages:
        # _persist_llm_calls writes the compact summary into ``content`` and
        # the full envelope into ``tool_calls[0]``.  Parse content and filter
        # by ``_kind`` so we don't accidentally match guided directive rows.
        try:
            content = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(content, dict):
            continue
        if content.get("_kind") == "llm_call_audit":
            rows.append(content)
    return rows


class TestE2ELLMCallAuditPersisted:
    """End-to-end persistence: solve_chain → recorder → _persist_llm_calls → DB."""

    def test_successful_solve_chain_persists_audit_row_with_status_success(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """Driving to STEP_3_TRANSFORMS via a working chain fires solve_chain
        exactly once; the SUCCESS audit row must appear in the DB after the
        route handler exits."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_passthrough(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

        audits = _extract_llm_call_audits(composer_test_client, session_id)
        # The wizard fires solve_chain exactly once — on the STEP_3 cold-start
        # ``/guided/chat`` call that builds the chain (the drive does not accept,
        # so there is no repair re-solve). One SUCCESS row, no others.
        assert len(audits) == 1, f"expected exactly one llm_call_audit row; got {audits}"
        row = audits[0]
        assert row["status"] == "success", row
        # Model is whatever ``settings.composer_model`` resolves to in the
        # test client.  We don't pin a specific string here — the existence
        # of a non-empty value proves the contract is wired, the specific
        # value is set by fixture config.
        assert row["model_requested"], "model_requested must be present in the audit row"

    def test_litellm_api_error_persists_audit_row_with_status_api_error(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """A LiteLLMAPIError at the repair re-solve site fires the auto-drop
        AND lands an api_error audit row.  Both audit channels (tool_invocations
        for the drop directive, llm_calls for the LLM-call record) must drain
        in the same ``finally`` block.

        Auto-drop is only reachable after a prior SUCCESSFUL proposal (the
        drive's bad-plugin chat solve), so the persisted llm_calls carry BOTH a
        ``success`` row (the drive) and an ``api_error`` row (the failed repair).
        We assert the api_error row EXISTS rather than pinning a total count.
        """
        from litellm.exceptions import APIError as LiteLLMAPIError

        session_id = _create_session(composer_test_client)
        _drive_to_step_3_propose_chain_bad(composer_test_client, session_id)

        api_err = LiteLLMAPIError(
            status_code=500,
            message="upstream provider failure",
            llm_provider="openai",
            model="gpt-4o",
        )
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=api_err,
        ):
            resp = _accept_propose_chain(composer_test_client, session_id)

        # Sanity: the existing auto-drop behavior is preserved (200, dropped).
        assert resp.status_code == 200, resp.json()

        # The whole point of this fix: the failed repair lands an api_error
        # LLM-call audit row alongside the drop invocation row.
        audits = _extract_llm_call_audits(composer_test_client, session_id)
        api_error_rows = [a for a in audits if a["status"] == "api_error"]
        assert len(api_error_rows) == 1, f"expected exactly one api_error llm_call_audit row; got {audits}"
        # The drive's successful chat solve also persisted its own row — proves
        # the channel drains every invocation, success and failure alike.
        assert any(a["status"] == "success" for a in audits), f"expected a success row from the drive solve; got {audits}"

        # And the sibling channel (drop invocation) still fired — proves the
        # two drain calls inside the same finally block are independent.
        drops = _extract_guided_drop_invocations(composer_test_client, session_id)
        assert len(drops) == 1, f"expected exactly one drop invocation; got {drops}"
