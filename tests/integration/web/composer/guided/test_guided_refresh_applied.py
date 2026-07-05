"""p1 Task 3 — apply via /guided/chat then GET /guided agree (in-place state)."""

from __future__ import annotations

from unittest.mock import patch

from tests.integration.web.composer.guided.test_step_3_e2e import (
    _create_session,
    _get_guided,
)
from tests.integration.web.composer.guided.test_step_chat import (
    _fake_resolve_source_response_csv,
    _seed_persisted_step1,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _post_chat(client: TestClient, session_id: str, *, message: str, step_index: str):
    resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": message, "step_index": step_index},
    )
    return resp.status_code, resp.json()


async def _fake_acompletion(*_args: object, **_kwargs: object) -> object:
    return _fake_resolve_source_response_csv()


def test_chat_apply_then_get_render_the_same_step_1_turn(composer_test_client: TestClient) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    # Seed persisted state so GET /guided records the initial SINGLE_SELECT turn.
    _seed_persisted_step1(client, session_id)
    _get_guided(client, session_id)  # records the initial SINGLE_SELECT turn
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_fake_acompletion,
    ):
        status, apply_body = _post_chat(client, session_id, message="make a csv source", step_index="step_1_source")
    assert status == 200, apply_body
    assert apply_body["guided_session"]["step"] == "step_1_source"
    apply_turn = apply_body["next_turn"]
    assert apply_turn["type"] == "schema_form"
    # Refresh: GET must re-render the SAME populated turn (staging fields cleared,
    # so _build_get_guided_turn hits the from-resolved sub-case, not the empty form).
    get_body = _get_guided(client, session_id)
    assert get_body["guided_session"]["step"] == "step_1_source"
    assert get_body["next_turn"]["type"] == "schema_form"
    assert get_body["next_turn"]["step_index"] == apply_turn["step_index"]
    assert get_body["next_turn"]["payload"]["plugin"] == apply_turn["payload"]["plugin"]
    assert get_body["next_turn"]["payload"]["prefilled"] == apply_turn["payload"]["prefilled"]
