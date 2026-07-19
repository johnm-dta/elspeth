"""A pure schema-8 Chat transition and GET expose one authoritative turn."""

from __future__ import annotations

from uuid import uuid4

import pytest

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.web.composer.guided.chat_solver import Step1SourceChatResolution
from elspeth.web.sessions._guided_step_chat import Step1SourceResolvedResult, StepChatResult
from elspeth.web.sessions.routes.composer import guided as guided_route
from elspeth.web.sessions.routes.composer.guided_chat_atomic import GuidedChatProviderOutcome
from tests.integration.web.composer.guided.test_step_chat import _create_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


async def _source_selection_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
    resolution = Step1SourceChatResolution(
        assistant_message="I prepared the CSV source form.",
        plugin="csv",
        filename="data.csv",
        mime_type="text/csv",
        content="name,email\nalice,a@example.test\n",
        options={"schema": {"mode": "observed"}},
        observed_columns=("name", "email"),
        sample_rows=({"name": "alice", "email": "a@example.test"},),
        on_validation_failure="discard",
    )
    return Step1SourceResolvedResult(
        chat=StepChatResult(
            assistant_message=resolution.assistant_message,
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=1,
            error_class=None,
        ),
        resolution=resolution,
    )


def test_chat_transition_then_get_render_the_same_step_1_turn(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    initial = client.get(f"/api/sessions/{session_id}/guided").json()
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _source_selection_provider)

    response = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={
            "operation_id": str(uuid4()),
            "turn_token": initial["next_turn"]["turn_token"],
            "message": "Use CSV.",
        },
    )

    assert response.status_code == 200, response.json()
    response_json = response.json()
    assert response_json["guided_session"]["step"] == "step_1_source"
    assert response_json["next_turn"]["type"] == "schema_form"
    assert response_json["composition_state"]["sources"] == {}
    refreshed = client.get(f"/api/sessions/{session_id}/guided").json()
    assert refreshed["next_turn"] == response_json["next_turn"]
    assert refreshed["composition_state"] == response_json["composition_state"]
