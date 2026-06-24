"""End-to-end Step 3 chain-solver tests.

Walks the wizard from new-session through Step 1 (CSV source) + Step 2
(JSON sink, no recipe match by virtue of non-classifier ``required_fields``)
+ Step 3 (LLM-proposed chain, accepted) to the wire-confirm stage and then a
COMPLETED terminal with rendered YAML.  The LLM is stubbed by patching
``_litellm_acompletion`` on
the chain_solver module the same way the dedicated chain-solver tests do.

Reject and clarifying-question paths are not implemented in this endpoint;
they raise ``HTTPException`` 501, which the second test exercises.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers (mirrors test_respond.py — kept local to avoid cross-file imports)
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "step-3-e2e"})
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


def _fake_llm_response_for_passthrough() -> SimpleNamespace:
    """A LiteLLM-shaped response that proposes a single passthrough transform.

    Passthrough is the safest stub target: minimal options
    (only ``schema: {mode: observed}`` required) so the
    ``_execute_set_pipeline`` step inside ``handle_step_3_chain_accept`` is
    exercising the wiring, not plugin-specific validation.
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
                                                    "plugin": "passthrough",
                                                    "options": {"schema": {"mode": "observed"}},
                                                    "rationale": "no transformation needed; pass rows through unchanged",
                                                }
                                            ],
                                            "why": "source rows already match sink schema; identity chain is sufficient",
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


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> tuple[dict, str, str]:
    """Drive the wizard to the Step 3 ``propose_chain`` turn.

    Returns (response_body_at_step_3, blob_id, output_path).

    Picks ``required_fields=["text"]`` so the deterministic recipe matcher
    finds no match (no classifier keyword present, single JSON output —
    neither ``classify-rows-llm-jsonl`` nor ``split-by-numeric-threshold``
    fires), forcing the chain-solver entry seam at the no-recipe branch.
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
    # No classifier keyword (no category/label/tag/classification),
    # not exactly two outputs → no recipe matches → chain solver entry seam fires.
    body = _respond(
        client,
        session_id,
        chosen=["text"],
        custom_inputs=[],
    )
    return body, blob_id, output_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStep3ChainAccept:
    def test_csv_to_json_step_3_accept_returns_confirm_wiring_then_completes_session(self, composer_test_client: TestClient) -> None:
        """End-to-end: Step 3 ACCEPT → confirm_wiring → terminal=COMPLETED."""
        session_id = _create_session(composer_test_client)

        # Drive Steps 1 + 2 + the auto-advance to Step 3 (with chain-solver stubbed).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_passthrough(),
        ):
            body, _blob_id, _output_path = _drive_to_step_3_propose_chain(composer_test_client, session_id)

            # Verify we are now at Step 3 with a server-emitted propose_chain turn.
            assert body["guided_session"]["step"] == "step_3_transforms"
            next_turn = body["next_turn"]
            assert next_turn is not None
            assert next_turn["type"] == "propose_chain"
            assert next_turn["step_index"] == 3
            payload = next_turn["payload"]
            assert set(payload.keys()) == {"steps", "why", "blockers"}
            assert payload["blockers"] == []
            assert payload["steps"][0]["plugin"] == "passthrough"

            # Accept the chain.  handle_step_3_chain_accept commits via
            # _execute_set_pipeline and redirects to the wire stage.
            accept_body = _respond(composer_test_client, session_id, chosen=["accept"])

        assert accept_body["terminal"] is None
        assert accept_body["guided_session"]["step"] == "step_4_wire"
        assert accept_body["next_turn"]["type"] == "confirm_wiring"

        accept_body = _confirm_wiring(composer_test_client, session_id)

        terminal = accept_body["terminal"]
        assert terminal is not None
        assert terminal["kind"] == "completed"
        assert terminal["pipeline_yaml"] is not None
        assert "source:" in terminal["pipeline_yaml"]
        assert "passthrough" in terminal["pipeline_yaml"]
        assert accept_body["next_turn"] is None

        gs = accept_body["guided_session"]
        assert gs["terminal"]["kind"] == "completed"


class TestStep3RejectNotImplemented:
    def test_csv_to_json_step_3_reject_returns_501(self, composer_test_client: TestClient) -> None:
        """Rejecting the chain proposal returns 501 with the freeform escape hatch."""
        session_id = _create_session(composer_test_client)

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_response_for_passthrough(),
        ):
            _drive_to_step_3_propose_chain(composer_test_client, session_id)

            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["reject"]},
            )

        assert resp.status_code == 501
        detail = resp.json()["detail"]
        assert "not yet implemented" in detail
        assert "exit-to-freeform" in detail
