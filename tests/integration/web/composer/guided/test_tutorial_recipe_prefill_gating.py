"""p4 Task 8b — tutorial recipe-slot prefill gating (end-to-end through the route).

Pins that the STEP_2.5 ``recipe_offer`` is TUTORIAL-gated at the offer-build seam:

  * TUTORIAL session  -> the operator-fillable LLM slots (``model``,
    ``api_key_secret``) are PREFILLED (read-only) and DROP OUT of the editable
    ``knobs`` set, so the passive learner's single "Apply recipe" click is
    enabled with nothing required-empty.
  * NON-tutorial (live) session -> the offer is UNCHANGED: those slots still
    surface as editable ``knobs`` for the operator to fill.

Driven through the real HTTP guided route with the classify-rows recipe (csv ->
single json with a ``category`` keyword field), whose unsatisfied required slots
are ``{classifier_template, model, api_key_secret}``. ``classifier_template`` is
NOT tutorial-fillable, so the tutorial offer keeps it editable while moving
``model``/``api_key_secret`` to prefilled — the cleanest gating signal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "prefill-gating-test"})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _start(client: TestClient, session_id: str, profile: str) -> None:
    resp = client.post(f"/api/sessions/{session_id}/guided/start", json={"profile": profile})
    assert resp.status_code == 200, resp.text


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    content = "text,category\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.text
    blob_id = resp.json()["id"]
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


def _drive_to_classify_recipe_offer(client: TestClient, session_id: str) -> dict:
    """Drive a started session to the classify-rows RECIPE_OFFER and return the body."""
    _, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out.jsonl")

    client.get(f"/api/sessions/{session_id}/guided")
    _respond(client, session_id, chosen=["csv"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {"path": storage_path, "schema": {"mode": "observed"}},
            "observed_columns": ["text", "category"],
            "sample_rows": [{"text": "Hello", "category": "greeting"}],
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
    body = _respond(client, session_id, chosen=["text", "category"], custom_inputs=[])
    assert body["next_turn"]["type"] == "recipe_offer", body["next_turn"]
    assert body["next_turn"]["payload"]["recipe_context"]["recipe_name"] == "classify-rows-llm-jsonl"
    return body


def _knob_names(body: dict) -> set[str]:
    return {entry["name"] for entry in body["next_turn"]["payload"]["knobs"]["fields"]}


def test_tutorial_offer_prefills_llm_slots(composer_test_client: TestClient) -> None:
    """TUTORIAL session -> model/api_key_secret prefilled, dropped from editable knobs."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "tutorial")
    body = _drive_to_classify_recipe_offer(composer_test_client, session_id)

    payload = body["next_turn"]["payload"]
    # The two LLM slots moved out of the editable knobs...
    assert _knob_names(body) == {"classifier_template"}
    # ...into the read-only prefilled set with honest config-sourced values.
    prefilled = payload["prefilled"]
    assert prefilled["model"] == composer_test_client.app.state.settings.composer_model
    assert prefilled["api_key_secret"] == "OPENROUTER_API_KEY"


def test_non_tutorial_offer_is_unchanged(composer_test_client: TestClient) -> None:
    """LIVE (non-tutorial) session -> the LLM slots stay editable; offer UNCHANGED."""
    session_id = _create_session(composer_test_client)
    _start(composer_test_client, session_id, "live")
    body = _drive_to_classify_recipe_offer(composer_test_client, session_id)

    # All three required slots remain editable knobs — no prefill leaked in.
    assert _knob_names(body) == {"classifier_template", "model", "api_key_secret"}
    prefilled = body["next_turn"]["payload"]["prefilled"]
    assert "model" not in prefilled
    assert "api_key_secret" not in prefilled
