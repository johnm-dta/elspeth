"""Current-schema integration tests for guided response transitions.

These tests exercise the live Step 1 and Step 2 turn contracts, server-held
blob inspection facts, reviewed source/output projections, fail-closed input
validation, and atomic settlement failure behavior. Later authoring stages are
covered separately as their schema-8 response handlers are implemented.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    """Create a session and return its string id."""
    resp = client.post("/api/sessions", json={"title": "respond-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    """Fetch GET /guided and assert 200."""
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _post_current_response(client: TestClient, session_id: str, **kwargs):
    """POST one strict schema-8 response to the current turn."""
    current = _get_guided(client, session_id)
    turn = current["next_turn"]
    payload = {
        "operation_id": str(uuid4()),
        "turn_token": turn["turn_token"] if turn is not None else None,
        **kwargs,
    }
    return client.post(f"/api/sessions/{session_id}/guided/respond", json=payload)


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    """POST one strict schema-8 response and require success."""
    resp = _post_current_response(client, session_id, **kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _full_guided_session(body: dict) -> dict:
    """The top-level ``guided_session`` wire projection (``GuidedSessionResponse``)
    deliberately omits Tier-3-bearing reviewed source/output options. The full
    schema-8 checkpoint is nested under
    ``composition_state.composer_meta.guided_session``.
    """
    return body["composition_state"]["composer_meta"]["guided_session"]


@pytest.mark.parametrize(
    "stable_id",
    [
        "",
        "0" * 35,
        "0" * 37,
        "00000000-0000-4000-8000-00000000000A",
        "../source",
    ],
)
def test_malformed_edit_target_stable_id_is_http_400(
    composer_test_client: TestClient,
    stable_id: str,
) -> None:
    session_id = _create_session(composer_test_client)

    response = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": "a" * 64,
            "proposal_id": "00000000-0000-4000-8000-000000000002",
            "draft_hash": "b" * 64,
            "edit_target": {"kind": "source", "stable_id": stable_id},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "edit_target.stable_id must be a canonical UUID"


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    """Seed a CSV blob and return (blob_id, storage_path).

    The ``storage_path`` is the authoritative file path under
    ``{data_dir}/blobs/{session_id}/`` and can be passed directly as the
    ``path`` option in a source SCHEMA_FORM response (it's already under
    the allowed source directories).

    The route obtains inspection and custody facts from this server-held blob;
    clients never submit those facts in the schema-8 response body.
    """
    content = "text,category\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]

    # Retrieve storage_path from the blob service (not exposed by API).
    blob_service = client.app.state.blob_service
    record = asyncio.run(blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    """Return an absolute path under {data_dir}/outputs/ for use as a sink path.

    Sink paths are validated by _validate_sink_path to be under {data_dir}/outputs/
    or {data_dir}/blobs/.  Tests must use this helper instead of bare relative paths
    like "out.jsonl" to pass the path allowlist check.
    """
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


# ---------------------------------------------------------------------------
# Step 1 intra-step — SINGLE_SELECT → SCHEMA_FORM
# ---------------------------------------------------------------------------


class TestStep1IntraStep:
    def test_single_select_response_emits_schema_form(self, composer_test_client: TestClient) -> None:
        """POST /respond with a SINGLE_SELECT response emits SCHEMA_FORM for the chosen plugin."""
        session_id = _create_session(composer_test_client)
        get_body = _get_guided(composer_test_client, session_id)
        assert get_body["next_turn"]["type"] == "single_select"

        # Respond to the single_select: pick "csv"
        body = _respond(composer_test_client, session_id, chosen=["csv"])

        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "schema_form"
        payload = body["next_turn"]["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "csv"
        assert "knobs" in payload
        assert "schema_block" not in payload
        assert "prefilled" in payload
        # schema.mode defaults to "observed"
        assert payload["prefilled"].get("schema", {}).get("mode") == "observed"

    def test_single_select_response_records_response_hash(self, composer_test_client: TestClient) -> None:
        """The TurnRecord for the SINGLE_SELECT turn is updated with a response_hash."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        body = _respond(composer_test_client, session_id, chosen=["csv"])

        history = body["guided_session"]["history"]
        # First record: the SINGLE_SELECT turn (now has response_hash)
        ss_record = next(r for r in history if r["turn_type"] == "single_select")
        assert ss_record["response_hash"] is not None
        # Second record: the new SCHEMA_FORM turn (no response yet)
        sf_record = next(r for r in history if r["turn_type"] == "schema_form")
        assert sf_record["response_hash"] is None
        assert sf_record["emitter"] == "server"

    def test_schema_form_step_index_matches_step_1(self, composer_test_client: TestClient) -> None:
        """The emitted schema_form is still at step_index 0 (step_1_source)."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        body = _respond(composer_test_client, session_id, chosen=["csv"])

        assert body["next_turn"]["step_index"] == 0  # STEP_1_SOURCE is index 0
        assert body["guided_session"]["step"] == "step_1_source"

    def test_uploaded_csv_prefills_blob_path_and_commits_observed_schema(self, composer_test_client: TestClient) -> None:
        """A Step-1 CSV pick after upload is deterministic without asking chat to infer the file."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        upload = composer_test_client.post(
            f"/api/sessions/{session_id}/blobs/inline",
            json={
                "filename": "guided_inventory.csv",
                "content": "sku,color,quantity\nSKU-001,red,2\nSKU-002,blue,5\nSKU-003,green,4\n",
                "mime_type": "text/csv",
            },
        )
        assert upload.status_code == 201, upload.json()
        blob_id = upload.json()["id"]

        body = _respond(composer_test_client, session_id, chosen=["csv"])

        assert body["next_turn"]["type"] == "schema_form"
        prefilled = body["next_turn"]["payload"]["prefilled"]
        assert prefilled["path"] == f"blob:{blob_id}"
        assert prefilled["on_validation_failure"] == "discard"
        assert prefilled["schema"]["mode"] == "flexible"
        assert {field.split(":", 1)[0] for field in prefilled["schema"]["fields"]} == {
            "sku",
            "color",
            "quantity",
        }

        advanced = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": prefilled,
            },
        )
        assert advanced["next_turn"]["type"] == "inspect_and_confirm"
        advanced = _respond(
            composer_test_client,
            session_id,
            edited_values={"columns": ["sku", "color", "quantity"]},
        )

        assert advanced["guided_session"]["step"] == "step_2_sink"
        reviewed_sources = _full_guided_session(advanced)["reviewed_sources"]
        assert len(reviewed_sources) == 1
        source = next(iter(reviewed_sources.values()))
        assert source["options"]["path"] == f"blob:{blob_id}"
        assert source["observed_columns"] == ["sku", "color", "quantity"]


# ---------------------------------------------------------------------------
# Step 1 completing — SCHEMA_FORM → handle_step_1_source → Step 2
# ---------------------------------------------------------------------------


class TestStep1Advance:
    def _drive_to_schema_form(self, client: TestClient, session_id: str) -> dict:
        """Drive to the Step 1 SCHEMA_FORM state. Returns the last /respond body."""
        _get_guided(client, session_id)
        return _respond(client, session_id, chosen=["csv"])

    def test_schema_form_response_advances_to_step_2(self, composer_test_client: TestClient) -> None:
        """A SCHEMA_FORM response calls handle_step_1_source and advances to STEP_2_SINK."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        # Submit the strict schema-8 plugin/options form.
        body = _respond(
            composer_test_client,
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

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "single_select"
        assert body["next_turn"]["step_index"] == 1  # STEP_2_SINK is index 1

    def test_schema_form_response_reviews_source_without_writing_topology(self, composer_test_client: TestClient) -> None:
        """Step 1 stores reviewed facts; proposal commit owns topology writes."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        body = _respond(
            composer_test_client,
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

        cs = body["composition_state"]
        assert cs is not None
        assert cs["sources"] == {}
        reviewed = cs["composer_meta"]["guided_session"]["reviewed_sources"]
        assert len(reviewed) == 1
        source = next(iter(reviewed.values()))
        assert source["plugin"] == "csv"
        assert source["options"]["path"] == "input.csv"

    def test_schema_form_rejects_blob_custody_tampering_without_path_leak(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        _blob_id, _storage_path = _seed_blob(composer_test_client, session_id)
        self._drive_to_schema_form(composer_test_client, session_id)
        rejected_path = "/tmp/not-under-elspeth-blobs/rows.csv"
        current = _get_guided(composer_test_client, session_id)
        turn = current["next_turn"]
        options = dict(turn["payload"]["prefilled"])
        options["path"] = rejected_path

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "edited_values": {
                    "plugin": "csv",
                    "options": options,
                },
            },
        )

        assert resp.status_code == 400, resp.json()
        detail = json.dumps(resp.json()["detail"])
        assert rejected_path not in detail
        assert "/tmp" not in detail

    def test_step_2_single_select_lists_sink_plugins(self, composer_test_client: TestClient) -> None:
        """The step-2 initial turn is single_select listing registered sink plugins."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        body = _respond(
            composer_test_client,
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

        options = body["next_turn"]["payload"]["options"]
        ids = [o["id"] for o in options]
        assert "json" in ids, f"json sink not found in options: {ids}"


# ---------------------------------------------------------------------------
# Step 2 intra-step — sink SINGLE_SELECT → SCHEMA_FORM → MULTI_SELECT
# ---------------------------------------------------------------------------


class TestStep2IntraStep:
    def _drive_to_step_2_single_select(self, client: TestClient, session_id: str) -> dict:
        """Drive to the Step 2 initial SINGLE_SELECT state."""
        _seed_blob(client, session_id)
        _get_guided(client, session_id)
        selected = _respond(client, session_id, chosen=["csv"])
        prefilled = selected["next_turn"]["payload"]["prefilled"]
        inspected = _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": prefilled,
            },
        )
        assert inspected["next_turn"]["type"] == "inspect_and_confirm"
        return _respond(client, session_id, edited_values={"columns": ["text", "category"]})

    def test_step_2_single_select_response_emits_schema_form(self, composer_test_client: TestClient) -> None:
        """Picking a sink plugin emits SCHEMA_FORM for the sink options."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)

        body = _respond(composer_test_client, session_id, chosen=["json"])

        assert body["next_turn"]["type"] == "schema_form"
        assert body["next_turn"]["payload"]["plugin"] == "json"
        assert body["next_turn"]["step_index"] == 1

    def test_schema_form_at_step_2_emits_multi_select(self, composer_test_client: TestClient) -> None:
        """Filling in sink options emits MULTI_SELECT_WITH_CUSTOM for required fields."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out.jsonl")

        # The strict form echoes the selected plugin with its validated options.
        body = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        assert body["next_turn"]["type"] == "multi_select_with_custom"
        payload = body["next_turn"]["payload"]
        assert "options" in payload
        assert "default_chosen" in payload
        # Observed columns from step 1 appear as options
        option_ids = [o["id"] for o in payload["options"]]
        assert "text" in option_ids
        assert "category" in option_ids

    def test_multi_select_response_advances_to_step_3_without_embedded_proposal(self, composer_test_client: TestClient) -> None:
        """Reviewed sink facts advance to Step 3; proposal creation is a later operation."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out.jsonl")
        # Step-2 SCHEMA_FORM: structured shape with plugin + options.
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        body = _respond(
            composer_test_client,
            session_id,
            chosen=["text", "category"],
            custom_inputs=[],
        )
        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"] is None

    def test_multi_select_response_reviews_output_without_writing_topology(self, composer_test_client: TestClient) -> None:
        """Step 2 stores reviewed output facts; proposal commit owns topology writes."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out_sink_commit.jsonl")
        # Step-2 SCHEMA_FORM: structured shape with plugin + options.
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        body = _respond(
            composer_test_client,
            session_id,
            chosen=["text", "category"],
            custom_inputs=[],
        )

        # Step advanced to 3.
        assert body["guided_session"]["step"] == "step_3_transforms"

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs["outputs"] == []
        full_guided = _full_guided_session(body)
        assert len(full_guided["reviewed_outputs"]) == 1
        output = next(iter(full_guided["reviewed_outputs"].values()))
        assert output["plugin"] == "json"
        assert output["required_fields"] == ["text", "category"]

    def test_multi_select_passthrough_signal_commits_with_no_required_fields(self, composer_test_client: TestClient) -> None:
        """control_signal='passthrough' is the explicit escape-hatch contract (C-3(a)):
        an empty chosen + custom_inputs pair commits successfully when paired with
        the passthrough signal, and composition_state gains the sink.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out_passthrough.jsonl")
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        body = _respond(
            composer_test_client,
            session_id,
            control_signal="passthrough",
        )

        assert body["guided_session"]["step"] == "step_3_transforms"
        output = next(iter(_full_guided_session(body)["reviewed_outputs"].values()))
        assert output["required_fields"] == []
        assert output["schema_mode"] == "observed"

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs["outputs"] == []

    def test_multi_select_bare_empty_chosen_returns_structured_400(self, composer_test_client: TestClient) -> None:
        """Fail-closed contract (C-3(a)): an empty chosen + custom_inputs pair
        WITHOUT the passthrough signal is ambiguous and must be rejected — with a
        structured, plain-language reason, not the old reasonless
        "Step 2 sink commit failed" string.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out_bare_empty.jsonl")
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        resp = _post_current_response(
            composer_test_client,
            session_id,
            chosen=[],
            custom_inputs=[],
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_multi_select_passthrough_with_chosen_fields_is_contradictory_400(self, composer_test_client: TestClient) -> None:
        """control_signal='passthrough' combined with explicit chosen fields is a
        contradictory payload — rejected with a structured reason rather than
        silently preferring one signal over the other.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out_contradiction.jsonl")
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        resp = _post_current_response(
            composer_test_client,
            session_id,
            chosen=["text"],
            custom_inputs=[],
            control_signal="passthrough",
        )
        assert resp.status_code == 422, resp.json()
        assert "control_signal cannot be combined with turn response fields" in resp.text

    def test_multi_select_settlement_failure_does_not_diverge_guided_session_from_state(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed atomic settlement leaves reviewed facts and topology unchanged."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "failed-settlement.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        before = _get_guided(composer_test_client, session_id)
        assert before["guided_session"]["step"] == "step_2_sink"
        assert _full_guided_session(before)["reviewed_outputs"] == {}
        before_outputs = before["composition_state"]["outputs"]

        async def fail_settlement(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("safe synthetic settlement failure")

        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "settle_guided_state_operation",
            fail_settlement,
        )
        resp = _post_current_response(
            composer_test_client,
            session_id,
            chosen=["text", "category"],
            custom_inputs=[],
        )
        assert resp.status_code == 500, resp.json()
        assert resp.json()["detail"]["failure_code"] == "operation_failed"

        after = _get_guided(composer_test_client, session_id)
        assert after["guided_session"]["step"] == "step_2_sink"
        assert _full_guided_session(after)["reviewed_outputs"] == {}
        assert after["composition_state"]["outputs"] == before_outputs


# ---------------------------------------------------------------------------
# Step 1 SCHEMA_FORM — contract-violation negative tests (Pair 4)
# ---------------------------------------------------------------------------
# The schema-8 form accepts exactly a plugin echo and options mapping. Blob
# inspection facts are server-held and deliberately absent from this contract.


class TestStep1SchemaFormAccept:
    def _drive_to_schema_form(self, client: TestClient, session_id: str) -> dict:
        """Drive to the Step 1 SCHEMA_FORM state. Returns the last /respond body."""
        _get_guided(client, session_id)
        return _respond(client, session_id, chosen=["csv"])

    def _assert_invalid(self, client: TestClient, session_id: str, edited_values: object) -> None:
        response = _post_current_response(client, session_id, edited_values=edited_values)
        assert response.status_code == 400, response.json()
        assert response.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_step_1_schema_form_missing_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """The strict schema-8 form requires the plugin echo."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)
        self._assert_invalid(composer_test_client, session_id, {"options": {}})

    def test_step_1_schema_form_missing_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``options`` key is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": "csv"})

    def test_step_1_schema_form_empty_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Empty plugin names fail at the current turn boundary."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)
        self._assert_invalid(composer_test_client, session_id, {"plugin": "", "options": {}})

    def test_step_1_schema_form_non_string_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-string ``plugin`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": 42, "options": {}})

    def test_step_1_schema_form_non_mapping_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-Mapping ``options`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": "csv", "options": ["not", "a", "mapping"]})


# ---------------------------------------------------------------------------
# Step 2 SCHEMA_FORM — contract-violation negative tests (Pair 4)
# ---------------------------------------------------------------------------
# The sink form uses the same strict plugin/options shape as the source form.


class TestStep2SchemaFormAccept:
    def _drive_to_step_2_schema_form(self, client: TestClient, session_id: str) -> None:
        """Drive to the Step 2 SCHEMA_FORM state (post-sink-pick)."""
        _seed_blob(client, session_id)
        _get_guided(client, session_id)
        selected = _respond(client, session_id, chosen=["csv"])
        prefilled = selected["next_turn"]["payload"]["prefilled"]
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": prefilled,
            },
        )
        _respond(client, session_id, edited_values={"columns": ["text", "category"]})
        _respond(client, session_id, chosen=["json"])

    def _assert_invalid(self, client: TestClient, session_id: str, edited_values: object) -> None:
        response = _post_current_response(client, session_id, edited_values=edited_values)
        assert response.status_code == 400, response.json()
        assert response.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_step_2_schema_form_missing_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``plugin`` key at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"options": {}})

    def test_step_2_schema_form_missing_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``options`` key at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": "json"})

    def test_step_2_schema_form_empty_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Empty-string ``plugin`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": "", "options": {}})

    def test_step_2_schema_form_non_string_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-string ``plugin`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": 42, "options": {}})

    def test_step_2_schema_form_non_mapping_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-Mapping ``options`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        self._assert_invalid(composer_test_client, session_id, {"plugin": "json", "options": "not a mapping"})


# ---------------------------------------------------------------------------
# Step 1 INSPECT_AND_CONFIRM — contract tests (Pair 5a)
# ---------------------------------------------------------------------------
# The live upload flow carries server-held source inspection facts into an
# INSPECT_AND_CONFIRM turn. The response contains only the reviewed columns;
# source review and session state settle atomically.


class TestStep1InspectAndConfirmAccept:
    def _seed_inspect_and_confirm_history(
        self,
        client: TestClient,
        session_id: str,
    ) -> dict:
        """Drive the real schema-8 upload flow to INSPECT_AND_CONFIRM."""
        _seed_blob(client, session_id)
        _get_guided(client, session_id)
        selected = _respond(client, session_id, chosen=["csv"])
        prefilled = selected["next_turn"]["payload"]["prefilled"]
        return _respond(
            client,
            session_id,
            edited_values={"plugin": "csv", "options": prefilled},
        )

    def test_inspect_and_confirm_non_list_columns_returns_400(self, composer_test_client: TestClient) -> None:
        """The current inspection response requires an exact column list."""
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)

        resp = _post_current_response(
            composer_test_client,
            session_id,
            edited_values={"columns": "text,category"},
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_inspect_and_confirm_commits_source_and_advances_to_step_2(self, composer_test_client: TestClient) -> None:
        """Success path: a valid INSPECT_AND_CONFIRM commits the source and
        advances to STEP_2_SINK, with both authoritative surfaces agreeing
        (elspeth-948eb9c0b8 C-3(b), Step-1 mirror of the Step-2 fix).
        """
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)
        body = _respond(composer_test_client, session_id, edited_values={"columns": ["text", "category"]})

        assert body["guided_session"]["step"] == "step_2_sink"
        full_guided = _full_guided_session(body)
        assert full_guided["pending_source_intents"] == {}
        assert len(full_guided["reviewed_sources"]) == 1
        source = next(iter(full_guided["reviewed_sources"].values()))
        assert source["plugin"] == "csv"
        assert source["observed_columns"] == ["text", "category"]

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs["sources"] == {}

    def test_inspect_and_confirm_commit_failure_does_not_diverge_guided_session_from_state(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed atomic settlement leaves the inspection intent current."""
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)

        before = _get_guided(composer_test_client, session_id)
        assert before["guided_session"]["step"] == "step_1_source"
        assert _full_guided_session(before)["reviewed_sources"] == {}
        before_sources = before["composition_state"]["sources"]

        async def fail_settlement(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("safe synthetic settlement failure")

        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "settle_guided_state_operation",
            fail_settlement,
        )
        resp = _post_current_response(
            composer_test_client,
            session_id,
            edited_values={"columns": ["text", "category"]},
        )
        assert resp.status_code == 500, resp.json()
        assert resp.json()["detail"]["failure_code"] == "operation_failed"

        after = _get_guided(composer_test_client, session_id)
        assert after["guided_session"]["step"] == "step_1_source"
        assert _full_guided_session(after)["reviewed_sources"] == {}
        assert after["composition_state"]["sources"] == before_sources


# ---------------------------------------------------------------------------
# Error paths: 400 on no GET /guided first, 404 unknown session
# ---------------------------------------------------------------------------


class TestRespondErrorPaths:
    def test_respond_unknown_session_returns_404(self, composer_test_client: TestClient) -> None:
        """POST /respond for a non-existent session returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = composer_test_client.post(
            f"/api/sessions/{fake_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": "a" * 64,
                "chosen": ["csv"],
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Unknown catalog choices fail closed at the response boundary and do not
# mutate the current guided turn.
# ---------------------------------------------------------------------------


class TestValueErrorMappedTo400:
    """Unknown source and sink plugins map to generic HTTP 400 responses."""

    def test_site_c_unknown_source_plugin_returns_400(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """An unknown source plugin fails without exposing catalog internals."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = _post_current_response(
            composer_test_client,
            session_id,
            chosen=["nonexistent_source_plugin_xyz"],
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_site_c_unknown_sink_plugin_returns_400(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """An unknown sink plugin fails without exposing catalog internals."""
        session_id = _create_session(composer_test_client)
        _seed_blob(composer_test_client, session_id)
        _get_guided(composer_test_client, session_id)
        selected = _respond(composer_test_client, session_id, chosen=["csv"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": selected["next_turn"]["payload"]["prefilled"],
            },
        )
        _respond(composer_test_client, session_id, edited_values={"columns": ["text", "category"]})
        # Now at step 2 SINGLE_SELECT — send a bogus sink plugin name.
        resp = _post_current_response(
            composer_test_client,
            session_id,
            chosen=["nonexistent_sink_plugin_xyz"],
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."


# ---------------------------------------------------------------------------
# Wire-validation tests — Codex #12 (control_signal enum validation)
# ---------------------------------------------------------------------------


class TestCodex12ControlSignalValidation:
    """Codex #12: control_signal validated against the ControlSignal enum at
    the wire boundary.

    ``GuidedRespondRequest`` stores control_signal as str | None
    intentionally; this guard is the route handler's enforcement point.
    """

    def test_unknown_control_signal_returns_400(self, composer_test_client: TestClient) -> None:
        """An unsupported signal fails closed before reservation."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = _post_current_response(
            composer_test_client,
            session_id,
            control_signal="invalid_value",
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."

    def test_exit_to_freeform_is_accepted(self, composer_test_client: TestClient) -> None:
        """exit_to_freeform is a valid signal and settles the terminal state."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = _post_current_response(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform",
        )
        assert resp.status_code == 200, resp.json()
        assert resp.json()["terminal"]["kind"] == "exited_to_freeform"

    def test_typo_in_known_signal_returns_400(self, composer_test_client: TestClient) -> None:
        """Asymmetry probe: a near-miss typo is rejected -- exact-value matching only."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = _post_current_response(
            composer_test_client,
            session_id,
            control_signal="exit_to_freeform_",
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Guided response does not satisfy the current turn contract."
