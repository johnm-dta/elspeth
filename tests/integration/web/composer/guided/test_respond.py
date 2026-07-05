"""Integration tests for POST /api/sessions/{id}/guided/respond.

Verifies the dispatcher's step routing, intra-step turn progression, and
the happy-path walk from step 1 (SINGLE_SELECT) through step 2.5
(RECIPE_OFFER) to the wire-confirm stage. Valid wiring confirmation is covered
by direct chain tests; invalid recipe wiring re-emits the wire turn.

HTTP transport: SyncASGITestClient (in-process, synchronous — same pattern
as test_get_guided.py).  The full roundtrip exercises:
  - route handler lock + guided session load
  - _dispatch_guided_respond per-step routing
  - step_advance (pure) + step handlers (side-effectful)
  - GuidedSession serialisation round-trip through composer_meta
  - audit drain via _persist_tool_invocations

Per spec §3.1, §3.2, §3.3, §5.3:
  - SINGLE_SELECT at step 1 → server emits SCHEMA_FORM (intra-step)
  - SCHEMA_FORM at step 1 → handle_step_1_source + advance to step 2;
    server emits SINGLE_SELECT (step 2 initial)
  - SINGLE_SELECT at step 2 → server emits SCHEMA_FORM (intra-step)
  - SCHEMA_FORM at step 2 → server emits MULTI_SELECT_WITH_CUSTOM (intra-step)
  - MULTI_SELECT_WITH_CUSTOM at step 2 → handle_step_2_sink + advance to step 3;
    server solves the transform chain and emits PROPOSE_CHAIN
  - PROPOSE_CHAIN chosen=["accept"] → handle_step_3_chain_accept;
    server emits CONFIRM_WIRING
  - CONFIRM_WIRING chosen=["confirm"] → terminal=COMPLETED

Error paths (exit_to_freeform, 409 after terminal) live in test_error_paths.py
(Task 3.6).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from elspeth.web.composer.redaction import REDACTED_BLOB_SOURCE_PATH
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


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    """POST /guided/respond and assert 200."""
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _full_guided_session(body: dict) -> dict:
    """The top-level ``guided_session`` wire projection (``GuidedSessionResponse``)
    deliberately omits ``step_1_result``/``step_2_result`` (Tier-3-bearing sink/source
    options). The full snapshot — including ``step_2_result`` — is nested under
    ``composition_state.composer_meta.guided_session`` on both GET /guided and
    POST /guided/respond responses.
    """
    return body["composition_state"]["composer_meta"]["guided_session"]


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
    """Seed a CSV blob and return (blob_id, storage_path).

    The ``storage_path`` is the authoritative file path under
    ``{data_dir}/blobs/{session_id}/`` and can be passed directly as the
    ``path`` option in a source SCHEMA_FORM response (it's already under
    the allowed source directories).

    Needed for:
    - Step 1 SCHEMA_FORM advance: ``_execute_set_source`` validates that
      ``path`` is under ``{data_dir}/blobs/``.
    - Step 2.5 recipe-apply: ``blob_id`` must resolve to a real file in
      the session DB for ``_resolve_source_blob``.
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
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)

        # Submit SCHEMA_FORM response with source options — path must be under {data_dir}/blobs/
        body = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text", "category"],
                "sample_rows": [{"text": "Hello", "category": "greeting"}],
            },
        )

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "single_select"
        assert body["next_turn"]["step_index"] == 1  # STEP_2_SINK is index 1

    def test_schema_form_response_commits_source_to_state(self, composer_test_client: TestClient) -> None:
        """After SCHEMA_FORM advance, composition_state has a source set."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)

        body = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["col_a"],
                "sample_rows": [],
            },
        )

        cs = body["composition_state"]
        assert cs is not None
        assert cs["sources"].get("source") is not None
        assert cs["sources"]["source"]["plugin"] == "csv"
        # Egress (the OTHER path the original leak was demonstrated on — the
        # schema_form submit via /guided/respond, not just /guided/chat): this
        # blob-backed source's absolute storage_path must NOT reach the wire. The
        # committed source carries no blob_ref (manual set_source strips it), so
        # redact_guided_snapshot_storage_paths cross-references the GuidedSession
        # snapshot's retained blob_ref to mask both the committed source path AND
        # the composer_meta snapshot path.
        assert cs["sources"]["source"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
        assert cs["composer_meta"]["guided_session"]["step_1_result"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
        assert storage_path not in json.dumps(body)

    def test_step_2_single_select_lists_sink_plugins(self, composer_test_client: TestClient) -> None:
        """The step-2 initial turn is single_select listing registered sink plugins."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)

        body = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["x"],
                "sample_rows": [],
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
        _get_guided(client, session_id)
        _respond(client, session_id, chosen=["csv"])
        _blob_id, storage_path = _seed_blob(client, session_id)
        return _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text", "label"],
                "sample_rows": [],
            },
        )

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

        # Step-2 SCHEMA_FORM: must include "plugin" in edited_values so the dispatcher
        # can persist the sink intent into GuidedSession.step_2_sink_intent.
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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        assert body["next_turn"]["type"] == "multi_select_with_custom"
        payload = body["next_turn"]["payload"]
        assert "options" in payload
        assert "default_chosen" in payload
        # Observed columns from step 1 appear as options
        option_ids = [o["id"] for o in payload["options"]]
        assert "text" in option_ids
        assert "label" in option_ids

    def test_multi_select_response_advances_to_step_3_propose_chain(self, composer_test_client: TestClient) -> None:
        """MULTI_SELECT_WITH_CUSTOM response advances straight to STEP_3 with a PROPOSE_CHAIN turn.

        The recipe-offer deviation was removed: the sink commit now hops directly
        to the chain solver, which builds the transform chain from the request.
        """
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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        # Confirm required fields via chosen. The backend reconstructs
        # SinkOutputResolved from step_2_sink_intent (plugin + options) plus these
        # required_fields and commits the sink. Committing the sink no longer
        # auto-builds the transform chain: it advances to step_3_transforms with
        # NO proposal (next_turn=None). The chain is built by the per-stage
        # transforms chat prompt, on which the SAME chain_solver mock fires.
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            body = _respond(
                composer_test_client,
                session_id,
                chosen=["text", "label"],
                custom_inputs=[],
            )
            assert body["next_turn"] is None
            assert body["guided_session"]["step"] == "step_3_transforms"

            chat_resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
            )
            assert chat_resp.status_code == 200, chat_resp.json()
            body = chat_resp.json()

        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "propose_chain"
        payload = body["next_turn"]["payload"]
        assert payload["steps"][0]["plugin"] == "passthrough"
        assert payload["blockers"] == []

    def test_multi_select_response_commits_sink_to_state(self, composer_test_client: TestClient) -> None:
        """MULTI_SELECT_WITH_CUSTOM → STEP_3 transition commits the sink to composition_state.outputs.

        handle_step_2_sink IS called on the MULTI_SELECT_WITH_CUSTOM → STEP_3
        transition (before the chain solver fires); state.outputs must be
        non-empty so the proposed chain has a sink to wire into.
        """
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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        # The MULTI_SELECT response that triggers step 2 → step 3 advance.
        # chosen carries the required field names; the backend reads plugin + options
        # from GuidedSession.step_2_sink_intent (persisted by the SCHEMA_FORM dispatcher).
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            body = _respond(
                composer_test_client,
                session_id,
                chosen=["text", "label"],
                custom_inputs=[],
            )

        # Step advanced to 3.
        assert body["guided_session"]["step"] == "step_3_transforms"

        # Composition state must have at least one output — sink was committed.
        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        outputs = cs.get("outputs", [])
        assert outputs, "composition_state.outputs is empty after MULTI_SELECT advance — handle_step_2_sink was not called"
        committed_schema = outputs[0]["options"]["schema"]
        assert committed_schema["mode"] == "observed"
        assert committed_schema["required_fields"] == ["text", "label"]

        # Both authoritative surfaces agree on success (elspeth-948eb9c0b8 C-3(b)):
        # the decisions-ledger side (guided_session.step_2_result) and the
        # validator side (composition_state.outputs) both reflect the commit.
        full_guided = _full_guided_session(body)
        assert full_guided["step_2_result"] is not None
        assert full_guided["step_2_result"]["outputs"]

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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={
                    "chosen": [],
                    "edited_values": None,
                    "custom_inputs": [],
                    "accepted_step_index": None,
                    "edit_step_index": None,
                    "control_signal": "passthrough",
                },
            )
        assert resp.status_code == 200, resp.json()
        body = resp.json()

        assert body["guided_session"]["step"] == "step_3_transforms"
        output = _full_guided_session(body)["step_2_result"]["outputs"][0]
        assert output["required_fields"] == []
        assert output["schema_mode"] == "observed"

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        outputs = cs.get("outputs", [])
        assert outputs, "composition_state.outputs is empty — passthrough commit did not reach handle_step_2_sink"
        committed_schema = outputs[0]["options"]["schema"]
        assert committed_schema["mode"] == "observed"
        assert committed_schema["required_fields"] == []

    def test_multi_select_schema_config_alias_is_canonicalised_on_commit(self, composer_test_client: TestClient) -> None:
        """Step-2 commits preserve schema_config alias compatibility without persisting duplicate aliases."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        output_path = _outputs_path(composer_test_client, "out_schema_config_alias.jsonl")
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema_config": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            body = _respond(
                composer_test_client,
                session_id,
                chosen=["text"],
                custom_inputs=["summary"],
            )

        outputs = body["composition_state"]["outputs"]
        committed_options = outputs[0]["options"]
        assert "schema_config" not in committed_options
        assert committed_options["schema"]["mode"] == "observed"
        assert committed_options["schema"]["required_fields"] == ["text", "summary"]

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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": [],
                "edited_values": None,
                "custom_inputs": [],
                "accepted_step_index": None,
                "edit_step_index": None,
                "control_signal": None,
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert detail["code"] == "guided_step2_no_fields_selected"
        # "detail" (not "message"): parseResponse (frontend/src/api/client.ts) reads
        # nestedDetail.detail as the human-readable string, with no "message" fallback.
        assert "pass all fields through" in detail["detail"].lower() or "let source decide" in detail["detail"].lower()

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
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": ["text"],
                "edited_values": None,
                "custom_inputs": [],
                "accepted_step_index": None,
                "edit_step_index": None,
                "control_signal": "passthrough",
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert detail["code"] == "guided_step2_passthrough_conflict"

    def test_multi_select_commit_failure_does_not_diverge_guided_session_from_state(self, composer_test_client: TestClient) -> None:
        """State-integrity test (elspeth-948eb9c0b8 C-3(b)): a Step-2 sink commit
        failure must NOT leave guided_session.step_2_result / step advanced while
        composition_state.outputs stays behind. Force a genuine commit failure
        (a sink path outside the allowed {data_dir}/outputs/ and {data_dir}/blobs/
        directories — a real handle_step_2_sink rejection, not the ambiguous-empty
        case) and assert neither store moved.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": "/tmp/definitely-not-an-allowed-sink-directory/out.jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        before = _get_guided(composer_test_client, session_id)
        assert before["guided_session"]["step"] == "step_2_sink"
        assert _full_guided_session(before)["step_2_result"] is None
        before_outputs = before["composition_state"]["outputs"]

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": ["text", "label"],
                "edited_values": None,
                "custom_inputs": [],
                "accepted_step_index": None,
                "edit_step_index": None,
                "control_signal": None,
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert detail["code"] == "guided_step2_sink_commit_failed"

        after = _get_guided(composer_test_client, session_id)
        # step_2_result and outputs must both stay exactly where they were —
        # not just "guided" as a whole (history[-1].response_hash IS expected to
        # get stamped on the rejected turn; that's the audit trail, not the
        # commit-relevant surfaces this test guards).
        assert after["guided_session"]["step"] == "step_2_sink"
        assert _full_guided_session(after)["step_2_result"] is None
        assert after["composition_state"]["outputs"] == before_outputs


# ---------------------------------------------------------------------------
# Step 1 SCHEMA_FORM — contract-violation negative tests (Pair 4)
# ---------------------------------------------------------------------------
# Defends against silent ``.get()``-with-default and bare ``str()`` coercion
# rewriting a malformed Step-1 SCHEMA_FORM ``edited_values`` payload into
# bogus-but-shape-valid data. The SchemaFormTurn widget contract is
# ``{"plugin": str, "options": Mapping, "observed_columns": list,
# "sample_rows": list}``; any deviation MUST surface as an HTTP 400 with
# the contract cited in the message — not flow downstream into
# ``SourceResolved`` and fail far away in handle_step_1_source as a
# misleading plugin-not-found or schema-mismatch error.


class TestStep1SchemaFormAccept:
    def _drive_to_schema_form(self, client: TestClient, session_id: str) -> dict:
        """Drive to the Step 1 SCHEMA_FORM state. Returns the last /respond body."""
        _get_guided(client, session_id)
        return _respond(client, session_id, chosen=["csv"])

    def test_step_1_schema_form_null_edited_values_returns_400(self, composer_test_client: TestClient) -> None:
        """``edited_values=null`` at Step 1 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form" in detail
        assert "step 1" in detail
        assert "null" in detail

    def test_step_1_schema_form_missing_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``plugin`` key is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"options": {}, "observed_columns": [], "sample_rows": []}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Boundary-message marker pins the dispatcher's missing-keys guard
        # against any downstream 400 that might also mention "plugin".
        assert "schema_form response at step 1" in detail
        assert "missing required keys" in detail
        assert "'plugin'" in detail

    def test_step_1_schema_form_missing_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``options`` key is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "csv", "observed_columns": [], "sample_rows": []}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 1" in detail
        assert "missing required keys" in detail
        assert "'options'" in detail

    def test_step_1_schema_form_missing_observed_columns_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``observed_columns`` key is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "csv", "options": {}, "sample_rows": []}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 1" in detail
        assert "missing required keys" in detail
        assert "'observed_columns'" in detail

    def test_step_1_schema_form_missing_sample_rows_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``sample_rows`` key is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "csv", "options": {}, "observed_columns": []}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 1" in detail
        assert "missing required keys" in detail
        assert "'sample_rows'" in detail

    def test_step_1_schema_form_empty_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Empty-string ``plugin`` is a protocol violation (HTTP 400).

        Defends specifically against the silent ``str(edited["plugin"])`` pattern
        that would have stringified ``""`` and routed it into ``SourceResolved``,
        failing far downstream as a misleading "plugin not registered: ''" error.
        """
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "",
                    "options": {},
                    "observed_columns": [],
                    "sample_rows": [],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Assert the boundary-message marker, not a bare substring — the bare
        # substring "plugin" matches the downstream ToolResult repr (which
        # echoes the offending value into a "plugin not registered" error),
        # so a pre-validator code path would also satisfy ``"plugin" in detail``.
        # Only the dispatcher's contract-citing 400 emits "must be a non-empty
        # string", which mechanically distinguishes pre/post-validation paths.
        assert "schema_form response at step 1" in detail
        assert "must be a non-empty string" in detail
        assert "''" in detail  # the offending empty-string value is echoed in the detail

    def test_step_1_schema_form_non_string_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-string ``plugin`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": 42,
                    "options": {},
                    "observed_columns": [],
                    "sample_rows": [],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Boundary-message marker pins the dispatcher 400 against the
        # downstream ToolResult repr that also contains "plugin".
        assert "schema_form response at step 1" in detail
        assert "must be a non-empty string" in detail
        assert "42" in detail  # the offending integer value is echoed in the detail

    def test_step_1_schema_form_non_mapping_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-Mapping ``options`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": ["not", "a", "mapping"],
                    "observed_columns": [],
                    "sample_rows": [],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Boundary-message marker pins the dispatcher's Mapping-guard against
        # any downstream 400 that may also mention "options" in its repr.
        assert "schema_form response at step 1" in detail
        assert "must be an object" in detail
        assert "list" in detail  # the offending type is named in the detail

    def test_step_1_schema_form_non_list_observed_columns_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-list ``observed_columns`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {},
                    "observed_columns": "abc",
                    "sample_rows": [],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Boundary-message marker pins the dispatcher's list-guard.
        assert "schema_form response at step 1" in detail
        assert "must be a list" in detail
        assert "str" in detail  # offending type is named

    def test_step_1_schema_form_non_list_sample_rows_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-list ``sample_rows`` is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {},
                    "observed_columns": [],
                    "sample_rows": {"not": "a list"},
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 1" in detail
        assert "must be a list" in detail
        assert "dict" in detail  # offending type is named


# ---------------------------------------------------------------------------
# Step 2 SCHEMA_FORM — contract-violation negative tests (Pair 4)
# ---------------------------------------------------------------------------
# Defends the Step-2 (sink) SCHEMA_FORM dispatcher branch. The SchemaFormTurn
# widget contract is identical to Step 1 — ``{"plugin": str, "options": Mapping,
# "observed_columns": list, "sample_rows": list}`` — but Step 2 only consumes
# ``plugin`` + ``options`` for ``step_2_sink_intent``. The dispatcher still
# rejects missing/wrong-typed ``plugin`` and ``options`` with HTTP 400.
# ``observed_columns`` and ``sample_rows`` are not validated at Step 2 (they
# are payload-shape echo, not consumed for sink state).


class TestStep2SchemaFormAccept:
    def _drive_to_step_2_schema_form(self, client: TestClient, session_id: str) -> None:
        """Drive to the Step 2 SCHEMA_FORM state (post-sink-pick)."""
        _get_guided(client, session_id)
        _respond(client, session_id, chosen=["csv"])
        _blob_id, storage_path = _seed_blob(client, session_id)
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text", "label"],
                "sample_rows": [],
            },
        )
        _respond(client, session_id, chosen=["json"])

    def test_step_2_schema_form_null_edited_values_returns_400(self, composer_test_client: TestClient) -> None:
        """``edited_values=null`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form" in detail
        assert "step 2" in detail
        assert "null" in detail

    def test_step_2_schema_form_missing_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``plugin`` key at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"options": {}}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # Boundary-message marker pins the dispatcher's missing-keys guard.
        assert "schema_form response at step 2" in detail
        assert "missing required keys" in detail
        assert "'plugin'" in detail

    def test_step_2_schema_form_missing_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``options`` key at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "json"}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 2" in detail
        assert "missing required keys" in detail
        assert "'options'" in detail

    def test_step_2_schema_form_empty_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Empty-string ``plugin`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "", "options": {}}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 2" in detail
        assert "must be a non-empty string" in detail
        assert "''" in detail

    def test_step_2_schema_form_non_string_plugin_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-string ``plugin`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": 42, "options": {}}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 2" in detail
        assert "must be a non-empty string" in detail
        assert "42" in detail

    def test_step_2_schema_form_non_mapping_options_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-Mapping ``options`` at Step 2 SCHEMA_FORM is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"plugin": "json", "options": "not a mapping"}},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "schema_form response at step 2" in detail
        assert "must be an object" in detail
        assert "str" in detail  # offending type is named


# ---------------------------------------------------------------------------
# Step 1 INSPECT_AND_CONFIRM — contract tests (Pair 5a)
# ---------------------------------------------------------------------------
# The INSPECT_AND_CONFIRM dispatcher branch (routes.py STEP_1 → STEP_2 path)
# is currently HTTP-unreachable in normal flow because ``get_guided`` always
# passes ``blob_inspection=None`` — the server never emits an
# INSPECT_AND_CONFIRM turn (see emitters.build_initial_step_1_turn).  The
# dispatch branch is exercised here via state injection (the same pattern
# used by test_progressive_disclosure._seed_terminal_state at line 170): a
# TurnRecord with turn_type=INSPECT_AND_CONFIRM AND step_1_source_intent is
# seeded into guided state before POST /respond fires.
#
# New wire contract (Pair 5a):
#   edited_values = {"columns": list[str]}
#
# Plugin, options, observed_columns, and sample_rows are now held server-side
# in step_1_source_intent (state_machine.SourceIntent).  The old 4-key wire
# shape is gone; tests for the removed plugin/options/observed_columns guards
# are removed.
#
# elspeth-948eb9c0b8 C-3(b) mirror fix: ``_advance_step_1`` is now a pure
# self-loop for INSPECT_AND_CONFIRM too (mirroring the Step 2 fix) — the
# resolve (step_1_source_intent + edited_values["columns"] -> SourceResolved),
# validation, and handle_step_1_source commit all happen in the dispatcher,
# and guided.step / step_1_result are only ever set after the commit is known
# to have succeeded. The former "shadowing" architecture (_advance_step_1
# running BEFORE the dispatcher, its own ValueError/KeyError propagating as
# a distinct HTTP 500 before the dispatcher's guards could even run) no
# longer exists: every guard below is now the single, live validation path.


class TestStep1InspectAndConfirmAccept:
    def _seed_inspect_and_confirm_history(
        self,
        client: TestClient,
        session_id: str,
        *,
        path: str = "/data/input.csv",
    ) -> None:
        """Seed an INSPECT_AND_CONFIRM TurnRecord + step_1_source_intent into the session.

        This bypasses GET /guided because the server-side initial-turn builder
        passes blob_inspection=None unconditionally and so never emits an
        inspect_and_confirm turn.  We inject the state directly via
        save_composition_state — the same pattern used by
        test_progressive_disclosure._seed_terminal_state (line 170).

        step_1_source_intent is populated so the dispatcher can recover
        plugin/options/sample_rows when the POST /respond fires.

        ``path`` defaults to "/data/input.csv" — outside any allowed source
        directory under the test's data_dir, so a commit against the default
        genuinely fails handle_step_1_source's path validation (used by the
        state-integrity test below). Pass a real storage_path (e.g. from
        _seed_blob) for tests that need the commit to succeed.
        """
        from dataclasses import replace

        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import TurnType
        from elspeth.web.composer.guided.state_machine import (
            GuidedSession,
            GuidedStep,
            SourceIntent,
            TurnRecord,
        )
        from elspeth.web.sessions.converters import state_from_record
        from elspeth.web.sessions.protocol import CompositionStateData
        from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session

        service = client.app.state.session_service
        session_uuid = UUID(session_id)
        state_record = asyncio.run(service.get_current_state(session_uuid))

        if state_record is None:
            state = _initial_composition_state_with_guided_session()
            existing_meta: dict = {}
        else:
            state = state_from_record(state_record)
            existing_meta = dict(deep_thaw(state_record.composer_meta)) if state_record.composer_meta else {}

        # Build the GuidedSession with an INSPECT_AND_CONFIRM TurnRecord
        # at STEP_1_SOURCE so the dispatcher's current_turn_type read picks
        # it up on POST /respond.  Also seed step_1_source_intent so the
        # dispatcher can build SourceResolved from it.
        guided = state.guided_session if state.guided_session is not None else GuidedSession.initial()
        record = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.INSPECT_AND_CONFIRM,
            payload_hash="seed-payload-hash",
            response_hash=None,
            emitter="server",
        )
        intent = SourceIntent(
            plugin="csv",
            options={"path": path, "schema": {"mode": "observed"}},
            observed_columns=("id", "name", "score"),
            sample_rows=({"id": 1, "name": "Alice", "score": 99},),
        )
        guided = replace(guided, history=(*guided.history, record), step_1_source_intent=intent)
        state = replace(state, guided_session=guided)

        new_composer_meta = {**existing_meta, "guided_session": guided.to_dict()}
        state_d = state.to_dict()
        state_data = CompositionStateData(
            sources=state_d["sources"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta=new_composer_meta,
        )
        asyncio.run(service.save_composition_state(session_uuid, state_data, provenance="session_seed"))

    def test_inspect_and_confirm_non_list_columns_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-list ``columns`` at post-advance INSPECT_AND_CONFIRM is a protocol violation (HTTP 400).

        The dispatcher's ``isinstance(columns_raw, list)`` guard fires the 400
        on the raw scalar value.  This guard must catch a non-list value before
        the dispatcher's own ``tuple(str(c) for c in columns_raw)`` coercion
        runs — that coercion would otherwise silently accept a scalar string
        by iterating its characters.
        """
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "columns": "id,name,score",  # string, not a list
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "inspect_and_confirm response at step 1" in detail
        assert "must be a list" in detail
        assert "str" in detail  # offending type is named

    def test_inspect_and_confirm_null_edited_values_returns_400(self, composer_test_client: TestClient) -> None:
        """edited_values=None on an INSPECT_AND_CONFIRM response returns a live HTTP 400.

        Before elspeth-948eb9c0b8's mirror fix this exact condition was
        raised as a ValueError inside ``_advance_step_1`` (which ran BEFORE
        the dispatcher) and shadowed the dispatcher's own null-guard,
        surfacing as a 400 via a different code path. Now the dispatcher's
        guard is the only one that exists.
        """
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": None},
        )
        assert resp.status_code == 400, resp.json()
        assert "requires edited_values" in resp.json()["detail"]

    def test_inspect_and_confirm_commits_source_and_advances_to_step_2(self, composer_test_client: TestClient) -> None:
        """Success path: a valid INSPECT_AND_CONFIRM commits the source and
        advances to STEP_2_SINK, with both authoritative surfaces agreeing
        (elspeth-948eb9c0b8 C-3(b), Step-1 mirror of the Step-2 fix).
        """
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id, path=storage_path)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"columns": ["id", "name", "score"]}},
        )
        assert resp.status_code == 200, resp.json()
        body = resp.json()

        assert body["guided_session"]["step"] == "step_2_sink"
        full_guided = _full_guided_session(body)
        assert full_guided["step_1_result"] is not None
        assert full_guided["step_1_result"]["plugin"] == "csv"
        assert full_guided["step_1_source_intent"] is None  # consumed

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs.get("sources"), "composition_state.sources is empty — handle_step_1_source was not called"

    def test_inspect_and_confirm_coerces_numeric_columns_to_str(self, composer_test_client: TestClient) -> None:
        """The dispatcher's ``tuple(str(c) for c in columns_raw)`` Tier-3
        coercion: a widget submitting non-string column labels must land as
        strings in the committed source's observed_columns, never as raw
        ints/bools. This coverage moved out of the unit suite when
        _advance_step_1 became a pure self-loop; without it, dropping the
        str() coercion would land ints in persisted composition state with the
        full suite still green. Regression for the fp-review test-adequacy gap.
        """
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id, path=storage_path)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"columns": [42, "name", True]}},
        )
        assert resp.status_code == 200, resp.json()
        body = resp.json()

        step_1_result = _full_guided_session(body)["step_1_result"]
        assert step_1_result is not None
        assert step_1_result["observed_columns"] == ["42", "name", "True"]

    def test_inspect_and_confirm_commit_failure_does_not_diverge_guided_session_from_state(self, composer_test_client: TestClient) -> None:
        """State-integrity test (elspeth-948eb9c0b8 C-3(b), Step-1 mirror of the Step-2
        test): a Step-1 source commit failure must NOT leave guided_session.step_1_result
        / step advanced while composition_state.sources stays behind. The default seed
        path ("/data/input.csv") is outside the test's allowed source directories, so
        handle_step_1_source genuinely rejects it — not the ambiguous-empty case, a real
        commit rejection.
        """
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)

        before = _get_guided(composer_test_client, session_id)
        assert before["guided_session"]["step"] == "step_1_source"
        assert _full_guided_session(before)["step_1_result"] is None
        before_sources = before["composition_state"]["sources"]

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"edited_values": {"columns": ["id", "name", "score"]}},
        )
        assert resp.status_code == 400, resp.json()
        assert resp.json()["detail"] == "Step 1 source commit failed"

        after = _get_guided(composer_test_client, session_id)
        assert after["guided_session"]["step"] == "step_1_source"
        assert _full_guided_session(after)["step_1_result"] is None
        assert after["composition_state"]["sources"] == before_sources


# ---------------------------------------------------------------------------
# Error paths: 400 on no GET /guided first, 404 unknown session
# ---------------------------------------------------------------------------


class TestRespondErrorPaths:
    def test_respond_without_prior_get_auto_seeds_and_succeeds(self, composer_test_client: TestClient) -> None:
        """POST /respond without prior GET auto-seeds the step_1 TurnRecord.

        Sibling of
        :py:meth:`TestRespondPreconditions.test_respond_without_prior_get_auto_seeds_and_succeeds`
        in ``test_error_paths.py``; the old 400 pre-condition was replaced
        by the route's auto-seed (commit c4e2f69cd) so a fresh-session
        respond produces a complete TurnRecord + audit row atomically.
        """
        session_id = _create_session(composer_test_client)
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert resp.status_code == 200, resp.json()

    def test_respond_unknown_session_returns_404(self, composer_test_client: TestClient) -> None:
        """POST /respond for a non-existent session returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = composer_test_client.post(
            f"/api/sessions/{fake_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Codex #8, #11, #15 — ValueError → HTTP 400 mapping in guided dispatcher
#
# S3 (commit f2478b6b) added `except InvariantError → 500` to distinguish
# server bugs from client faults.  The symmetric `except ValueError → 400`
# was missing; these tests verify it is now present at both catch sites.
#
# Site B: solve_chain() — chain_solver raises only InvariantError (server
#   bugs); no client-fault ValueError path exists there.  The new catch
#   provides defence-in-depth but has no direct client trigger.
#
# Site C: build_step_1/2_schema_form_turn() — catalog.get_schema() raises
#   ValueError for unknown plugin names (service.py line 114).  Fires inside
#   _dispatch_guided_respond, caught by the dispatcher try/except.
# ---------------------------------------------------------------------------


class TestValueErrorMappedTo400:
    """ValueError from step_advance or the dispatcher maps to HTTP 400 (Codex #8, #15)."""

    def test_site_c_unknown_source_plugin_returns_400(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """Site C (Codex #15): unknown source plugin name in chosen maps to HTTP 400.

        The step-1 SINGLE_SELECT handler calls build_step_1_schema_form_turn,
        which calls catalog.get_schema("source", plugin_name).  For an unknown
        plugin name that raises ValueError (service.py line 114).  The
        dispatcher try/except must catch this and re-raise as HTTPException(400).
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["nonexistent_source_plugin_xyz"]},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "Guided-mode protocol error" in detail
        assert "nonexistent_source_plugin_xyz" in detail

    def test_site_c_unknown_sink_plugin_returns_400(
        self,
        composer_test_client: TestClient,
    ) -> None:
        """Site C mirror (Codex #15): unknown sink plugin name in chosen maps to HTTP 400.

        The step-2 SINGLE_SELECT handler calls build_step_2_schema_form_turn,
        which calls catalog.get_schema("sink", plugin_name).  For an unknown
        plugin name that raises ValueError (service.py line 114).  The
        dispatcher try/except must catch this and re-raise as HTTPException(400).
        """
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)

        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}},
                "observed_columns": ["text"],
                "sample_rows": [],
            },
        )
        # Now at step 2 SINGLE_SELECT — send a bogus sink plugin name.
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["nonexistent_sink_plugin_xyz"]},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "Guided-mode protocol error" in detail
        assert "nonexistent_sink_plugin_xyz" in detail


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
        """An unknown string value for control_signal -> 400 with valid-values message."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": "invalid_value"},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "Unknown control_signal" in detail
        assert "invalid_value" in detail
        assert "exit_to_freeform" in detail  # valid values listed

    def test_null_control_signal_passes_through(self, composer_test_client: TestClient) -> None:
        """control_signal=null is the normal path -- guard must not reject it."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        # null control_signal + valid chosen -> normal step-1 advance -> 200
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": None},
        )
        assert resp.status_code == 200, resp.json()

    def test_exit_to_freeform_is_accepted(self, composer_test_client: TestClient) -> None:
        """exit_to_freeform is a valid ControlSignal value -> guard passes, step_advance runs."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": "exit_to_freeform"},
        )
        # step_advance handles exit_to_freeform by dropping to freeform terminal.
        # The guard must not reject it; the response may be 200 or 4xx depending on
        # the step_advance / dispatcher logic, but not due to our validation guard.
        # We only assert it is NOT a 400 caused by "Unknown control_signal".
        if resp.status_code == 400:
            assert "Unknown control_signal" not in resp.json().get("detail", "")

    def test_reject_signal_is_accepted(self, composer_test_client: TestClient) -> None:
        """reject is a valid ControlSignal value -> guard passes."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": "reject"},
        )
        # Guard must not return "Unknown control_signal".
        if resp.status_code == 400:
            assert "Unknown control_signal" not in resp.json().get("detail", "")

    def test_request_advisor_signal_is_accepted(self, composer_test_client: TestClient) -> None:
        """request_advisor is a valid ControlSignal value -> guard passes."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": "request_advisor"},
        )
        if resp.status_code == 400:
            assert "Unknown control_signal" not in resp.json().get("detail", "")

    def test_typo_in_known_signal_returns_400(self, composer_test_client: TestClient) -> None:
        """Asymmetry probe: a near-miss typo is rejected -- exact-value matching only."""
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "control_signal": "exit_to_freeform_"},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "Unknown control_signal" in detail


# ---------------------------------------------------------------------------
# Wire-validation tests — Codex #16 (per-element sample_rows validation)
# ---------------------------------------------------------------------------


class TestCodex16SampleRowsElementValidation:
    """Codex #16: per-element Mapping check for sample_rows at Step 1 SCHEMA_FORM.

    The outer-container check (list vs non-list) existed before this change.
    The new guard catches non-Mapping elements inside the list, which
    previously triggered an uncontrolled TypeError from dict(r) -> 500.
    """

    def _drive_to_step_1_schema_form(self, client: TestClient, session_id: str) -> None:
        """Drive to the Step 1 SCHEMA_FORM state."""
        _get_guided(client, session_id)
        _respond(client, session_id, chosen=["csv"])

    def test_non_mapping_elements_in_sample_rows_return_400(self, composer_test_client: TestClient) -> None:
        """sample_rows containing non-Mapping elements (int, str, list) -> 400 naming the index."""
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {"path": storage_path, "schema": {"mode": "observed"}},
                    "observed_columns": ["text"],
                    "sample_rows": [42, "string", []],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        # The error message must name the offending list position.
        # Detail text form: edited_values['sample_rows'][N] must be an object
        assert "'sample_rows'][" in detail
        assert "must be an object" in detail

    def test_single_non_mapping_at_index_1_names_index(self, composer_test_client: TestClient) -> None:
        """One valid row followed by a non-Mapping element: error names index 1."""
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {"path": storage_path, "schema": {"mode": "observed"}},
                    "observed_columns": ["text"],
                    "sample_rows": [{"text": "ok"}, 99],
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "'sample_rows'][1]" in detail

    def test_valid_sample_rows_pass_through(self, composer_test_client: TestClient) -> None:
        """Asymmetry probe: a well-formed sample_rows list of Mapping elements does not
        trigger the guard and allows the request to reach the step handler."""
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {"path": storage_path, "schema": {"mode": "observed"}},
                    "observed_columns": ["text"],
                    "sample_rows": [{"text": "Hello"}, {"text": "World"}],
                },
            },
        )
        # Guard passes; step handler runs and advances to Step 2 -> 200.
        assert resp.status_code == 200, resp.json()

    def test_empty_sample_rows_list_is_valid(self, composer_test_client: TestClient) -> None:
        """An empty sample_rows list has no elements to fail the per-element check."""
        session_id = _create_session(composer_test_client)
        _blob_id, storage_path = _seed_blob(composer_test_client, session_id)
        self._drive_to_step_1_schema_form(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "edited_values": {
                    "plugin": "csv",
                    "options": {"path": storage_path, "schema": {"mode": "observed"}},
                    "observed_columns": [],
                    "sample_rows": [],
                },
            },
        )
        assert resp.status_code == 200, resp.json()


# ---------------------------------------------------------------------------
# Wire-validation tests — Codex #7 (step-index validation)
# ---------------------------------------------------------------------------

# Helpers for driving to Step 3 (mirrors test_step_3_e2e.py pattern).


def _fake_llm_passthrough_for_step_index_tests() -> SimpleNamespace:
    """Single-step passthrough proposal stub."""
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
                                                    "rationale": "identity chain",
                                                }
                                            ],
                                            "why": "rows already match sink schema",
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


def _drive_to_step_3_propose_chain_for_step_idx(
    client: TestClient,
    session_id: str,
) -> None:
    """Drive to the Step 3 PROPOSE_CHAIN turn (no-recipe path)."""
    _blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out_idx.jsonl")

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
    # Non-classifier required field -> sink commit. Committing the sink no
    # longer auto-builds the transform chain: it advances to step_3_transforms
    # with no proposal (next_turn=None).
    sink_body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
    assert sink_body["next_turn"] is None
    assert sink_body["guided_session"]["step"] == "step_3_transforms"

    # The per-stage transforms chat prompt drives the chain solver and emits
    # the propose_chain turn (the SAME chain_solver mock fires on this call).
    chat_resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
    )
    assert chat_resp.status_code == 200, chat_resp.json()
    chat_body = chat_resp.json()
    assert chat_body["guided_session"]["step"] == "step_3_transforms"
    assert chat_body["next_turn"]["type"] == "propose_chain"


class TestCodex7StepIndexValidation:
    """Codex #7: accepted_step_index / edit_step_index validated against the
    current proposal at the wire boundary.

    Guards fire before response_hash computation and audit emission, so an
    invalid request returns 400 without recording a partial event.
    """

    def test_accepted_step_index_minus_one_returns_400(self, composer_test_client: TestClient) -> None:
        """Negative accepted_step_index is out of range -> 400."""
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            _drive_to_step_3_propose_chain_for_step_idx(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["accept"], "accepted_step_index": -1},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "accepted_step_index" in detail
        assert "out of range" in detail

    def test_accepted_step_index_beyond_end_returns_400(self, composer_test_client: TestClient) -> None:
        """accepted_step_index=999 is far beyond proposal length -> 400."""
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            _drive_to_step_3_propose_chain_for_step_idx(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["accept"], "accepted_step_index": 999},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "accepted_step_index" in detail
        assert "out of range" in detail

    def test_edit_step_index_out_of_range_returns_400(self, composer_test_client: TestClient) -> None:
        """edit_step_index=999 is out of range -> 400."""
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            _drive_to_step_3_propose_chain_for_step_idx(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["accept"], "edit_step_index": 999},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "edit_step_index" in detail
        assert "out of range" in detail

    def test_valid_accepted_step_index_zero_is_accepted(self, composer_test_client: TestClient) -> None:
        """Asymmetry probe: accepted_step_index=0 is valid for a 1-step proposal (no error).

        The guard must not reject valid indices. accepted_step_index is not
        consumed by the accept handler (the dispatcher uses the full proposal),
        so the request reaches its normal path and returns 200.
        """
        session_id = _create_session(composer_test_client)
        with patch(
            "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_fake_llm_passthrough_for_step_index_tests(),
        ):
            _drive_to_step_3_propose_chain_for_step_idx(composer_test_client, session_id)
            resp = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={"chosen": ["accept"], "accepted_step_index": 0},
            )
        # 200 -> the guard passed; the underlying handler ran to completion.
        assert resp.status_code == 200, resp.json()

    def test_non_null_step_index_at_step_1_returns_400(self, composer_test_client: TestClient) -> None:
        """Step-1 SINGLE_SELECT is not PROPOSE_CHAIN -> accepted_step_index must be None.

        Cross-step stale probe: a request with accepted_step_index set at Step 1
        (where it is semantically irrelevant) is caught before it can corrupt
        step-machine state.
        """
        session_id = _create_session(composer_test_client)
        _get_guided(composer_test_client, session_id)
        # Currently at Step 1 SINGLE_SELECT.
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"], "accepted_step_index": 0},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "accepted_step_index" in detail
        assert "null" in detail
