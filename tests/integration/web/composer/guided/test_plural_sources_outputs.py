"""Real schema-8 journey for plural reviewed source/output controllers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions.models import composition_proposals_table, guided_operations_table, proposal_events_table
from elspeth.web.sessions.routes.composer.guided_chat_atomic import _current_sink
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "plural source/output review"})
    assert response.status_code == 201, response.json()
    return response.json()["id"]


def _get(client: TestClient, session_id: str) -> dict:
    response = client.get(f"/api/sessions/{session_id}/guided")
    assert response.status_code == 200, response.json()
    return response.json()


def _hydrate(client: TestClient, session_id: str) -> GuidedSession:
    record = asyncio.run(client.app.state.session_service.get_current_state(UUID(session_id)))
    if record is None:
        return GuidedSession.initial()
    assert record.composer_meta is not None
    return GuidedSession.from_dict(deep_thaw(record.composer_meta)["guided_session"])


def _respond(client: TestClient, session_id: str, **response_fields: object) -> dict:
    current = _get(client, session_id)
    _hydrate(client, session_id)
    turn = current["next_turn"]
    assert turn is not None
    response = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            **response_fields,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def _output_path(client: TestClient, filename: str) -> str:
    path = Path(client.app.state.settings.data_dir) / "outputs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _source_path(client: TestClient, filename: str) -> str:
    path = Path(client.app.state.settings.data_dir) / "blobs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,label\n1,reviewed\n", encoding="utf-8")
    return str(path)


def _review_items(body: dict) -> list[dict]:
    assert body["next_turn"]["type"] == "review_components"
    return body["next_turn"]["payload"]["items"]


def _stage_minimal_plural_proposal(client: TestClient, *, suffix: str) -> tuple[str, dict]:
    """Stage a fresh ordered 2-source/2-output proposal through public HTTP."""

    session_id = _create_session(client)
    source_ids: list[str] = []
    for index in (1, 2):
        if index == 2:
            _respond(client, session_id, component_action={"action": "add", "component_kind": "source"})
        _respond(client, session_id, chosen=["csv"])
        reviewed = _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {
                    "path": f"{suffix}-input-{index}.csv",
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                },
            },
        )
        source_ids = [item["stable_id"] for item in _review_items(reviewed)]
    _respond(
        client,
        session_id,
        component_action={
            "action": "reorder",
            "component_kind": "source",
            "stable_ids": list(reversed(source_ids)),
        },
    )
    _respond(client, session_id, component_action={"action": "finish", "component_kind": "source"})

    output_ids: list[str] = []
    for index in (1, 2):
        if index == 2:
            _respond(client, session_id, component_action={"action": "add", "component_kind": "output"})
        _respond(client, session_id, chosen=["json"])
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _output_path(client, f"{suffix}-output-{index}.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                    "on_write_failure": "discard",
                },
            },
        )
        reviewed = _respond(client, session_id, control_signal="passthrough")
        output_ids = [item["stable_id"] for item in _review_items(reviewed)]
    _respond(
        client,
        session_id,
        component_action={
            "action": "reorder",
            "component_kind": "output",
            "stable_ids": list(reversed(output_ids)),
        },
    )
    staged = _respond(client, session_id, component_action={"action": "finish", "component_kind": "output"})
    proposal = staged["next_turn"]["payload"]
    assert [item["stable_id"] for item in proposal["graph"]["sources"]] == list(reversed(source_ids))
    assert [item["stable_id"] for item in proposal["outputs"]] == list(reversed(output_ids))
    return session_id, staged


def test_plural_sources_outputs_survive_hydration_and_stage_ordered_proposal(
    composer_test_client: TestClient,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    source_a_path = _source_path(client, "input-a.csv")
    source_b_path = _source_path(client, "input-b.csv")

    # Source A resolves but remains in source review.
    _respond(client, session_id, chosen=["csv"])
    source_review = _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {
                "path": source_a_path,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
        },
    )
    source_a = _review_items(source_review)[0]["stable_id"]
    assert source_review["guided_session"]["step"] == "step_1_source"

    # Add uses the operation-attempt UUID once and same-operation replay returns
    # the exact pending identity rather than allocating a replacement.
    current = _get(client, session_id)
    add_operation = str(uuid4())
    add_request = {
        "operation_id": add_operation,
        "turn_token": current["next_turn"]["turn_token"],
        "component_action": {"action": "add", "component_kind": "source"},
    }
    added = client.post(f"/api/sessions/{session_id}/guided/respond", json=add_request)
    assert added.status_code == 200, added.json()
    replayed = client.post(f"/api/sessions/{session_id}/guided/respond", json=add_request)
    assert replayed.status_code == 200, replayed.json()
    assert replayed.json() == added.json()
    pending_after_add = _hydrate(client, session_id)
    source_b = next(iter(pending_after_add.pending_source_intents))
    assert source_b != source_a

    _respond(client, session_id, chosen=["csv"])
    source_review = _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {
                "path": source_b_path,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
        },
    )
    assert [item["stable_id"] for item in _review_items(source_review)] == [source_a, source_b]

    source_review = _respond(
        client,
        session_id,
        component_action={
            "action": "reorder",
            "component_kind": "source",
            "stable_ids": [source_b, source_a],
        },
    )
    assert [item["stable_id"] for item in _review_items(source_review)] == [source_b, source_a]

    # Model a literal service restart: a new app, service, engine, and client
    # must recover the same durable controller before the journey can continue.
    before_restart = _hydrate(client, session_id).to_dict()
    old_app = client.app
    old_service = client.app.state.session_service
    old_engine = client.app.state.session_engine
    restart_test_client = getattr(client.app.state, "restart_test_client", None)
    assert callable(restart_test_client), "guided integration fixture must support a literal app/service restart"
    client = restart_test_client()
    assert client.app is not old_app
    assert client.app.state.session_service is not old_service
    assert client.app.state.session_engine is not old_engine
    assert _hydrate(client, session_id).to_dict() == before_restart
    restarted_review = _get(client, session_id)
    assert [item["stable_id"] for item in _review_items(restarted_review)] == [source_b, source_a]

    editing = _respond(
        client,
        session_id,
        component_action={"action": "edit", "target": {"kind": "source", "stable_id": source_b}},
    )
    assert editing["next_turn"]["type"] == "schema_form"
    assert editing["next_turn"]["payload"]["prefilled"]["path"] == source_b_path
    revised_source_b_path = _source_path(client, "input-b-revised.csv")
    source_review = _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {
                "path": revised_source_b_path,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
        },
    )
    hydrated_sources = _hydrate(client, session_id)
    assert hydrated_sources.source_order == (source_b, source_a)
    assert hydrated_sources.reviewed_sources[source_b].name == "source_2"
    assert dict(hydrated_sources.reviewed_sources[source_b].options)["path"] == revised_source_b_path
    assert hydrated_sources.reviewed_sources[source_b].on_validation_failure == "discard"
    assert hydrated_sources.active_edit_target is None

    source_review = _respond(
        client,
        session_id,
        component_action={"action": "remove", "target": {"kind": "source", "stable_id": source_a}},
    )
    assert [item["stable_id"] for item in _review_items(source_review)] == [source_b]
    _respond(client, session_id, component_action={"action": "add", "component_kind": "source"})
    _respond(client, session_id, chosen=["csv"])
    source_c_path = _source_path(client, "input-c.csv")
    source_review = _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {
                "path": source_c_path,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
        },
    )
    source_c = next(item["stable_id"] for item in _review_items(source_review) if item["stable_id"] != source_b)
    assert [item["stable_id"] for item in _review_items(source_review)] == [source_b, source_c]
    output_select = _respond(
        client,
        session_id,
        component_action={"action": "finish", "component_kind": "source"},
    )
    assert output_select["guided_session"]["step"] == "step_2_sink"
    assert output_select["next_turn"]["type"] == "single_select"

    # Repeat the same controller lifecycle for two outputs.
    _respond(client, session_id, chosen=["json"])
    output_c_path = _output_path(client, "output-c.jsonl")
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": _output_path(client, "output-a.jsonl"),
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
                "on_write_failure": "discard",
            },
        },
    )
    output_review = _respond(client, session_id, control_signal="passthrough")
    output_a = _review_items(output_review)[0]["stable_id"]

    _respond(client, session_id, component_action={"action": "add", "component_kind": "output"})
    _respond(client, session_id, chosen=["json"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": _output_path(client, "output-b.jsonl"),
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
                "on_write_failure": "discard",
            },
        },
    )
    output_review = _respond(client, session_id, control_signal="passthrough")
    output_b = next(item["stable_id"] for item in _review_items(output_review) if item["stable_id"] != output_a)
    output_review = _respond(
        client,
        session_id,
        component_action={
            "action": "reorder",
            "component_kind": "output",
            "stable_ids": [output_b, output_a],
        },
    )
    assert [item["stable_id"] for item in _review_items(output_review)] == [output_b, output_a]
    hydrated_outputs = _hydrate(client, session_id)
    sink = _current_sink(hydrated_outputs)
    assert sink is not None
    assert tuple(output.name for output in sink.outputs) == ("output_2", "output")

    editing = _respond(
        client,
        session_id,
        component_action={"action": "edit", "target": {"kind": "output", "stable_id": output_b}},
    )
    assert editing["next_turn"]["payload"]["prefilled"]["on_write_failure"] == "discard"
    revised_path = _output_path(client, "output-b-revised.jsonl")
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": revised_path,
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
                "on_write_failure": "discard",
            },
        },
    )
    output_review = _respond(client, session_id, control_signal="passthrough")
    hydrated_outputs = _hydrate(client, session_id)
    assert hydrated_outputs.output_order == (output_b, output_a)
    assert hydrated_outputs.reviewed_outputs[output_b].name == "output_2"
    assert dict(hydrated_outputs.reviewed_outputs[output_b].options)["path"] == revised_path
    assert hydrated_outputs.reviewed_outputs[output_b].on_write_failure == "discard"

    output_review = _respond(
        client,
        session_id,
        component_action={"action": "remove", "target": {"kind": "output", "stable_id": output_a}},
    )
    assert [item["stable_id"] for item in _review_items(output_review)] == [output_b]
    _respond(client, session_id, component_action={"action": "add", "component_kind": "output"})
    _respond(client, session_id, chosen=["json"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": output_c_path,
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
                "on_write_failure": "discard",
            },
        },
    )
    output_review = _respond(client, session_id, control_signal="passthrough")
    output_c = next(item["stable_id"] for item in _review_items(output_review) if item["stable_id"] != output_b)
    assert [item["stable_id"] for item in _review_items(output_review)] == [output_b, output_c]
    staged = _respond(
        client,
        session_id,
        component_action={"action": "finish", "component_kind": "output"},
    )
    assert staged["guided_session"]["step"] == "step_3_transforms"
    assert staged["next_turn"]["type"] == "propose_pipeline"
    assert staged["next_turn"]["payload"]["proposal_id"]
    proposal_payload = staged["next_turn"]["payload"]
    assert [item["stable_id"] for item in proposal_payload["graph"]["sources"]] == [source_b, source_c]
    assert [item["stable_id"] for item in proposal_payload["outputs"]] == [output_b, output_c]
    assert _hydrate(client, session_id).active_proposal is not None

    accepted = _respond(
        client,
        session_id,
        chosen=["accept"],
        proposal_id=proposal_payload["proposal_id"],
        draft_hash=proposal_payload["draft_hash"],
    )
    assert accepted["guided_session"]["step"] == "step_4_wire"
    assert accepted["next_turn"]["type"] == "confirm_wiring"
    assert _hydrate(client, session_id).active_proposal is None
    assert [output["name"] for output in accepted["composition_state"]["outputs"]] == ["output_2", "output"]
    assert [output["options"]["path"] for output in accepted["composition_state"]["outputs"]] == [
        revised_path,
        output_c_path,
    ]
    assert [output["sink_name"] for output in accepted["next_turn"]["payload"]["topology"]["outputs"]] == [
        "output_2",
        "output",
    ]
    authoritative = _get(client, session_id)
    assert authoritative["guided_session"] == accepted["guided_session"]
    assert authoritative["terminal"] == accepted["terminal"]
    assert authoritative["composition_state"] == accepted["composition_state"]
    assert [output["name"] for output in authoritative["composition_state"]["outputs"]] == [
        "output_2",
        "output",
    ]
    assert authoritative["next_turn"] == accepted["next_turn"]


def test_rejected_plural_proposal_is_terminal_and_cannot_execute(composer_test_client: TestClient) -> None:
    client = composer_test_client
    session_id, staged = _stage_minimal_plural_proposal(client, suffix="reject-plural")
    proposal_turn = staged["next_turn"]
    proposal = proposal_turn["payload"]
    before_state = staged["composition_state"]

    rejected = _respond(
        client,
        session_id,
        control_signal="reject",
        proposal_id=proposal["proposal_id"],
        draft_hash=proposal["draft_hash"],
    )

    assert rejected["next_turn"] is None
    assert rejected["composition_state"]["id"] != before_state["id"]
    assert rejected["composition_state"]["sources"] == before_state["sources"]
    assert rejected["composition_state"]["outputs"] == before_state["outputs"]
    assert _hydrate(client, session_id).active_proposal is None
    with client.app.state.session_engine.connect() as connection:
        assert (
            connection.execute(
                select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == proposal["proposal_id"])
            ).scalar_one()
            == "rejected"
        )
        assert connection.execute(
            select(proposal_events_table.c.event_type)
            .where(proposal_events_table.c.proposal_id == proposal["proposal_id"])
            .order_by(proposal_events_table.c.created_at)
        ).scalars().all() == ["proposal.created", "proposal.rejected"]

    cannot_execute = client.post(
        f"/api/sessions/{session_id}/proposals/{proposal['proposal_id']}/accept",
        json={"draft_hash": proposal["draft_hash"]},
    )
    assert cannot_execute.status_code == 409
    assert _get(client, session_id)["composition_state"] == rejected["composition_state"]


def test_blob_backed_source_edit_reinspects_exact_target_and_preserves_identity(
    composer_test_client: TestClient,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    upload = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={
            "filename": "reviewed.csv",
            "content": "id,label\n1,first\n",
            "mime_type": "text/csv",
        },
    )
    assert upload.status_code == 201, upload.json()
    blob_id = upload.json()["id"]

    selected = _respond(client, session_id, chosen=["csv"])
    prefilled = selected["next_turn"]["payload"]["prefilled"]
    assert prefilled["path"] == f"blob:{blob_id}"
    _respond(client, session_id, edited_values={"plugin": "csv", "options": prefilled})
    reviewed = _respond(client, session_id, edited_values={"columns": ["id", "label"]})
    stable_id = _review_items(reviewed)[0]["stable_id"]

    editing = _respond(
        client,
        session_id,
        component_action={"action": "edit", "target": {"kind": "source", "stable_id": stable_id}},
    )
    assert editing["next_turn"]["payload"]["prefilled"]["path"] == f"blob:{blob_id}"
    inspecting = _respond(
        client,
        session_id,
        edited_values={"plugin": "csv", "options": editing["next_turn"]["payload"]["prefilled"]},
    )
    assert inspecting["next_turn"]["type"] == "inspect_and_confirm"
    staged = _hydrate(client, session_id)
    assert staged.active_edit_target is not None
    assert staged.active_edit_target.stable_id == stable_id
    assert stable_id in staged.reviewed_sources
    assert staged.pending_source_intents[stable_id].phase == "inspection_review"

    reviewed = _respond(client, session_id, edited_values={"columns": ["record_id", "display_label"]})
    assert _review_items(reviewed)[0]["stable_id"] == stable_id
    hydrated = _hydrate(client, session_id)
    assert hydrated.source_order == (stable_id,)
    assert hydrated.reviewed_sources[stable_id].name == "source"
    assert hydrated.reviewed_sources[stable_id].observed_columns == ("record_id", "display_label")
    assert hydrated.active_edit_target is None


@pytest.mark.parametrize(
    "component_action",
    [
        {"action": "finish", "component_kind": "output"},
        {
            "action": "remove",
            "target": {"kind": "source", "stable_id": "99999999-9999-4999-8999-999999999999"},
        },
    ],
    ids=["wrong-kind", "stale-id"],
)
def test_invalid_component_action_is_atomic_before_provider(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    component_action: dict[str, object],
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    _respond(client, session_id, chosen=["csv"])
    reviewed = _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {
                "path": "input.csv",
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
        },
    )
    before = _hydrate(client, session_id).to_dict()
    with client.app.state.session_engine.connect() as connection:
        operations_before = connection.execute(
            select(func.count()).select_from(guided_operations_table).where(guided_operations_table.c.session_id == session_id)
        ).scalar_one()

    async def forbidden_provider(**_kwargs: object) -> object:
        raise AssertionError("invalid component action reached planner provider")

    monkeypatch.setattr(client.app.state.composer_service, "plan_guided_pipeline", forbidden_provider)
    response = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": reviewed["next_turn"]["turn_token"],
            "component_action": component_action,
        },
    )

    assert response.status_code == 400
    assert _hydrate(client, session_id).to_dict() == before
    with client.app.state.session_engine.connect() as connection:
        operations_after = connection.execute(
            select(func.count()).select_from(guided_operations_table).where(guided_operations_table.c.session_id == session_id)
        ).scalar_one()
    assert operations_after == operations_before


def test_component_action_on_non_review_and_stale_turn_are_rejected_without_state(
    composer_test_client: TestClient,
) -> None:
    client = composer_test_client
    session_id = _create_session(client)
    current = _get(client, session_id)
    before_versions = asyncio.run(client.app.state.session_service.get_state_versions(UUID(session_id)))

    wrong_turn = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": current["next_turn"]["turn_token"],
            "component_action": {"action": "finish", "component_kind": "source"},
        },
    )
    stale_turn = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": "0" * 64,
            "component_action": {"action": "finish", "component_kind": "source"},
        },
    )

    assert wrong_turn.status_code == 400
    assert stale_turn.status_code == 409
    assert asyncio.run(client.app.state.session_service.get_state_versions(UUID(session_id))) == before_versions
