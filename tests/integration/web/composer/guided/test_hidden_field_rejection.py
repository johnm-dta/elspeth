"""Integration coverage for hidden schema-form field rejection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def test_hidden_field_submission_returns_400(
    composer_test_client: TestClient,
    session_at_step_3_llm: str,
) -> None:
    """Operator submits a value for a hidden variant field; backend rejects."""
    sess_id = session_at_step_3_llm

    resp = composer_test_client.post(
        f"/api/sessions/{sess_id}/guided/respond",
        json={
            "chosen": None,
            "edited_values": {
                "plugin": "llm",
                "options": {
                    "provider": "azure",
                    "schema_config": {"mode": "observed"},
                    "deployment_name": "gpt-4",
                    "endpoint": "https://example.openai.azure.com/",
                    "api_key": "test-key",
                    "prompt_template": "Summarize {{ row.text }}",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "hidden_field_submitted"
    assert body["detail"]["field"] == "base_url"
    audit_rows = _audit_events_for_session(composer_test_client, sess_id)
    assert any(row.tool_name == "guided_hidden_field_rejected" for row in audit_rows)


@pytest.fixture
def session_at_step_3_llm(composer_test_client: TestClient) -> str:
    sess = _create_session(composer_test_client)
    schema_turn = _drive_guided_flow_to_step_3_transform_schema(composer_test_client, sess, transform_plugin="llm")
    assert schema_turn["guided_session"]["step"] == "step_3_transforms"
    assert schema_turn["next_turn"]["type"] == "schema_form"
    assert schema_turn["next_turn"]["payload"]["plugin"] == "llm"
    return sess


def _audit_events_for_session(client: TestClient, session_id: str) -> list[SimpleNamespace]:
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    rows: list[SimpleNamespace] = []
    for msg in msgs:
        if msg.role not in ("tool", "audit") or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            invocation = tool_call.get("invocation", {})
            if invocation.get("tool_name"):
                rows.append(SimpleNamespace(**invocation))
    return rows


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "hidden-field-rejection"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _drive_guided_flow_to_step_3_transform_schema(client: TestClient, session_id: str, *, transform_plugin: str) -> dict:
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_response_for_transform(transform_plugin),
    ):
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
        # Committing the sink no longer auto-builds the transform chain: it
        # advances to step_3_transforms with no proposal (next_turn=None). The
        # chain is built by the per-stage transforms chat prompt, on which the
        # SAME chain_solver mock fires.
        body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
        assert body["next_turn"] is None
        assert body["guided_session"]["step"] == "step_3_transforms"

        chat_resp = client.post(
            f"/api/sessions/{session_id}/guided/chat",
            json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
        )
        assert chat_resp.status_code == 200, chat_resp.json()
        chat_body = chat_resp.json()
        assert chat_body["guided_session"]["step"] == "step_3_transforms"
        assert chat_body["next_turn"]["type"] == "propose_chain"

        edit_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": None,
                "edited_values": None,
                "custom_inputs": None,
                "accepted_step_index": None,
                "edit_step_index": 0,
                "control_signal": None,
            },
        )
        assert edit_resp.status_code == 200, edit_resp.json()
        return edit_resp.json()


def _fake_llm_response_for_transform(plugin: str) -> SimpleNamespace:
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
                                                    "plugin": plugin,
                                                    "options": {
                                                        "provider": "azure",
                                                        "schema_config": "observed",
                                                        "deployment_name": "gpt-4",
                                                        "endpoint": "https://example.openai.azure.com/",
                                                        "api_key": "test-key",
                                                        "prompt_template": "Summarize {{ row.text }}",
                                                    },
                                                    "rationale": "exercise variant visibility",
                                                }
                                            ],
                                            "why": "single transform proposal for hidden-field rejection test",
                                            "blockers": [],
                                        },
                                    }
                                ),
                            )
                        )
                    ]
                )
            )
        ]
    )


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": "text,note\nHello,greeting\n", "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    record = asyncio.run(client.app.state.blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)
