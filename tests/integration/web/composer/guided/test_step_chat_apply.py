"""p1 Task 4 — STEP_2/STEP_3 /guided/chat apply branches (in-place)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# Reuse the verbatim drive helpers from the sibling e2e test.
from tests.integration.web.composer.guided.test_step_3_e2e import (
    _create_session,
    _drive_to_step_3_propose_chain,
    _fake_llm_response_for_passthrough,
    _get_guided,
    _outputs_path,
    _respond,
    _seed_blob,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _post_chat(client: TestClient, session_id: str, *, message: str, step_index: str):
    resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": message, "step_index": step_index},
    )
    return resp.status_code, resp.json()


def _fake_resolve_sink_response(path: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_sink",
                                arguments=json.dumps(
                                    {
                                        "resolution": "sink",
                                        "outputs": [
                                            {
                                                "plugin": "json",
                                                "options": {
                                                    "path": path,
                                                    "schema": {"mode": "observed"},
                                                    "mode": "write",
                                                    "collision_policy": "auto_increment",
                                                },
                                                "required_fields": [],
                                                "schema_mode": "observed",
                                            }
                                        ],
                                        "assistant_message": "Output set to JSON Lines.",
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


def _fake_chain_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
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
                                                    "rationale": "no transform needed; pass rows through",
                                                }
                                            ],
                                            "why": "the rows already match the sink",
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


def _drive_to_step_2(client: TestClient, session_id: str) -> None:
    _blob_id, storage_path = _seed_blob(client, session_id)
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
    body = _get_guided(client, session_id)
    assert body["guided_session"]["step"] == "step_2_sink"


def test_step_2_chat_applies_sink_in_place(composer_test_client: TestClient) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    _drive_to_step_2(client, session_id)
    out = _outputs_path(client, "chat_out.jsonl")
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_resolve_sink_response(out)),
    ):
        status, body = _post_chat(client, session_id, message="write the rows to a jsonl file", step_index="step_2_sink")
    assert status == 200, body
    # Apply-in-place: phase stays STEP_2, sink committed, form re-rendered.
    assert body["guided_session"]["step"] == "step_2_sink"
    assert body["next_turn"]["type"] == "schema_form"
    assert body["next_turn"]["step_index"] == 1
    outputs = body["composition_state"]["outputs"]
    assert any(o["plugin"] == "json" for o in outputs)


def test_step_2_chat_apply_then_get_render_the_same_step_2_turn(composer_test_client: TestClient) -> None:
    """apply↔GET equality: the next_turn the STEP_2 apply emits is byte-identical
    to the turn GET /guided rebuilds from the committed (rehydrated, frozen) sink.

    The GET side rehydrates ``step_2_result`` from persisted composer_meta, whose
    options are deep-frozen ``mappingproxy`` — so this test is the load-bearing
    proof of the ``build_step_2_schema_form_turn_from_resolved`` deep_thaw fix:
    without it, the rehydrated render raises on the nested frozen options.
    """
    client = composer_test_client
    session_id = _create_session(client)
    _drive_to_step_2(client, session_id)
    out = _outputs_path(client, "equality_out.jsonl")
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_resolve_sink_response(out)),
    ):
        status, applied = _post_chat(client, session_id, message="write the rows to a jsonl file", step_index="step_2_sink")
    assert status == 200, applied
    apply_turn = applied["next_turn"]
    assert apply_turn is not None
    assert apply_turn["type"] == "schema_form"

    # GET rehydrates from frozen composer_meta and rebuilds the SAME turn.
    got = _get_guided(client, session_id)
    get_turn = got["next_turn"]
    assert get_turn is not None
    assert get_turn == apply_turn


def test_step_2_chat_prose_is_advisory_no_mutation(composer_test_client: TestClient) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    _drive_to_step_2(client, session_id)
    before = _get_guided(client, session_id)
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A sink writes rows out.", tool_calls=None))])
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=prose),
    ):
        status, body = _post_chat(client, session_id, message="what is a sink?", step_index="step_2_sink")
    assert status == 200, body
    # Advisory: no mutation, no next_turn, phase unchanged.
    assert body["next_turn"] is None
    assert body["guided_session"]["step"] == "step_2_sink"
    # No outputs committed by an advisory message.
    after = _get_guided(client, session_id)
    assert before["composition_state"]["outputs"] == after["composition_state"]["outputs"]


def test_step_2_chat_from_schema_form_stamps_prior_record_answered(composer_test_client: TestClient) -> None:
    """A STEP_2 chat apply from the SCHEMA_FORM sub-state stamps the prior record
    answered (response_hash non-None) — audit parity with STEP_1."""
    client = composer_test_client
    session_id = _create_session(client)
    _drive_to_step_2(client, session_id)
    # Advance to the STEP_2 SCHEMA_FORM sub-state by picking a sink plugin
    # (SINGLE_SELECT -> SCHEMA_FORM), so the existing record is a SCHEMA_FORM turn.
    _respond(client, session_id, chosen=["json"])
    before = _get_guided(client, session_id)
    assert before["next_turn"]["type"] == "schema_form"
    out = _outputs_path(client, "answered_out.jsonl")
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_resolve_sink_response(out)),
    ):
        status, body = _post_chat(client, session_id, message="write the rows to a jsonl file", step_index="step_2_sink")
    assert status == 200, body
    assert body["guided_session"]["step"] == "step_2_sink"
    # The prior STEP_2 SCHEMA_FORM record is now answered (not response_hash=None).
    step_2_schema_records = [r for r in body["guided_session"]["history"] if r["step"] == "step_2_sink" and r["turn_type"] == "schema_form"]
    assert step_2_schema_records, body["guided_session"]["history"]
    # The earliest STEP_2 schema_form record (the answered one) carries a hash;
    # the freshly re-rendered one is the new unanswered turn.
    assert any(r["response_hash"] is not None for r in step_2_schema_records)


def test_step_3_chat_reproposes_in_place_without_committing(composer_test_client: TestClient) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    # Drive to STEP_3 under the HELPER's own chain-solver patch (which exits at
    # the end of this block), so it does NOT nest with this test's patch below.
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_response_for_passthrough(),
    ):
        _drive_to_step_3_propose_chain(client, session_id)
    # Now a fresh, non-nested patch controls the STEP_3 chat re-solve.
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_chain_response()),
    ):
        status, body = _post_chat(client, session_id, message="actually just pass the rows through", step_index="step_3_transforms")
    assert status == 200, body
    # In-place: phase stays STEP_3, a fresh propose_chain turn is re-rendered,
    # and the pipeline is NOT committed/advanced to wire.
    assert body["guided_session"]["step"] == "step_3_transforms"
    assert body["next_turn"]["type"] == "propose_chain"
    assert body["terminal"] is None
