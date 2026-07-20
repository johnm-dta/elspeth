"""Integration coverage for current-schema guided audit emission."""

from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid4

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
    current = _get_guided(client, session_id)
    turn = current["next_turn"]
    resp = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"] if turn is not None else None,
            **kwargs,
        },
    )
    assert resp.status_code == 200, resp.json()
    return resp.json()


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


def test_successful_turn_settlement_persists_exact_audit_cohort(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    turn = _get_guided(composer_test_client, session_id)["next_turn"]

    body = _respond(
        composer_test_client,
        session_id,
        chosen=[turn["payload"]["options"][0]["id"]],
    )

    assert body["next_turn"]["type"] == "schema_form"
    events = _extract_guided_invocations(composer_test_client, session_id)
    assert len(events["guided_turn_emitted"]) == 2
    assert len(events["guided_turn_answered"]) == 1
    assert events["guided_step_advanced"] == []
    assert events["guided_dropped_to_freeform"] == []


def test_invalid_preflight_never_records_an_answered_turn(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    turn = _get_guided(composer_test_client, session_id)["next_turn"]
    _respond(
        composer_test_client,
        session_id,
        chosen=[turn["payload"]["options"][0]["id"]],
    )
    current = _get_guided(composer_test_client, session_id)["next_turn"]
    baseline = _extract_guided_invocations(composer_test_client, session_id)

    response = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": current["turn_token"],
            "edited_values": {"plugin": current["payload"]["plugin"], "options": {}},
        },
    )

    assert response.status_code == 400
    after = _extract_guided_invocations(composer_test_client, session_id)
    assert after == baseline


def test_explicit_exit_persists_one_redacted_drop_event(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)

    body = _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

    assert body["terminal"]["kind"] == "exited_to_freeform"
    drops = _extract_guided_invocations(composer_test_client, session_id)["guided_dropped_to_freeform"]
    assert drops == [
        {
            "drop_reason": "user_pressed_exit",
            "prev_step": "step_1_source",
        }
    ]


def test_get_guided_rejects_freeform_session_without_masking_http_error(composer_test_client: TestClient) -> None:
    from elspeth.web.sessions.protocol import CompositionStateData

    session_id = _create_session(composer_test_client)
    service = composer_test_client.app.state.session_service
    freeform_state = CompositionStateData(
        sources={},
        nodes=(),
        edges=(),
        outputs=(),
        metadata_={"name": "Untitled Pipeline", "description": ""},
        is_valid=False,
        validation_errors=None,
        composer_meta={},
    )
    asyncio.run(service.save_composition_state(UUID(session_id), freeform_state, provenance="session_seed"))

    response = composer_test_client.get(f"/api/sessions/{session_id}/guided")

    assert response.status_code == 400
    assert "not in guided mode" in response.json()["detail"]
