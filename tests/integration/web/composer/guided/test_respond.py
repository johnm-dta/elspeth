"""Integration tests for POST /api/sessions/{id}/guided/respond.

Verifies the dispatcher's step routing, intra-step turn progression, and
the happy-path walk from step 1 (SINGLE_SELECT) through step 2.5
(RECIPE_OFFER) to a COMPLETED terminal state.

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
  - MULTI_SELECT_WITH_CUSTOM at step 2 → handle_step_2_sink + advance;
    server emits RECIPE_OFFER if recipe matched
  - RECIPE_OFFER chosen=["accept"] → handle_step_2_5_recipe_apply;
    terminal=COMPLETED

Error paths (exit_to_freeform, 409 after terminal) live in test_error_paths.py
(Task 3.6).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

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
        assert payload["plugin"] == "csv"
        assert "schema_block" in payload
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
        assert cs["source"] is not None
        assert cs["source"]["plugin"] == "csv"

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

    def test_multi_select_response_advances_to_step_2_5_with_recipe(self, composer_test_client: TestClient) -> None:
        """MULTI_SELECT_WITH_CUSTOM response advances to step 2.5 with a RECIPE_OFFER turn."""
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
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        # Confirm required fields via chosen — "label" matches the classify-rows recipe's
        # keyword set.  The backend reconstructs SinkOutputResolved from step_2_sink_intent
        # (plugin + options) plus these required_fields.
        body = _respond(
            composer_test_client,
            session_id,
            chosen=["text", "label"],
            custom_inputs=[],
        )

        assert body["guided_session"]["step"] == "step_2_5_recipe_match"
        assert body["next_turn"] is not None
        assert body["next_turn"]["type"] == "recipe_offer"
        payload = body["next_turn"]["payload"]
        assert "recipe_name" in payload
        assert payload["recipe_name"] == "classify-rows-llm-jsonl"
        # Gap 6 wire-shape: payload exposes the unsatisfied required slots
        # so the frontend can render an editable form for them.  classify-rows
        # has three required slots not derivable from (source, sink):
        # classifier_template, model, api_key_secret.
        assert "unsatisfied_slots" in payload
        unsatisfied_names = {entry["name"] for entry in payload["unsatisfied_slots"]}
        assert unsatisfied_names == {
            "classifier_template",
            "model",
            "api_key_secret",
        }
        # ``required`` is absent from the wire shape: the RecipeMatch invariant
        # guarantees every entry is required, so the field would be dead information.
        for entry in payload["unsatisfied_slots"]:
            assert set(entry.keys()) == {"name", "slot_type", "description"}
            assert "required" not in entry

    def test_multi_select_response_commits_sink_to_state(self, composer_test_client: TestClient) -> None:
        """M1: MULTI_SELECT_WITH_CUSTOM → step 2.5 transition commits sink to composition_state.outputs.

        Verifies C2 fix: handle_step_2_sink IS called on the MULTI_SELECT_WITH_CUSTOM
        → step 2.5 transition.  Before the fix, state.outputs was empty because
        handle_step_2_sink was never called; the recipe-apply path (step 2.5) would
        then fail with a missing sink when generating YAML.
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
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )

        # The MULTI_SELECT response that triggers step 2 → 2.5 advance.
        # chosen carries the required field names; the backend reads plugin + options
        # from GuidedSession.step_2_sink_intent (persisted by the SCHEMA_FORM dispatcher).
        body = _respond(
            composer_test_client,
            session_id,
            chosen=["text", "label"],
            custom_inputs=[],
        )

        # Step advanced to 2.5.
        assert body["guided_session"]["step"] == "step_2_5_recipe_match"

        # Composition state must have at least one output — sink was committed.
        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        outputs = cs.get("outputs", {})
        assert outputs, "composition_state.outputs is empty after MULTI_SELECT advance — handle_step_2_sink was not called (C2 regression)"


# ---------------------------------------------------------------------------
# Step 2.5 — RECIPE_OFFER accept → terminal=COMPLETED
# ---------------------------------------------------------------------------


class TestStep25RecipeAccept:
    def _drive_to_recipe_offer(self, client: TestClient, session_id: str) -> tuple[dict, str]:
        """Drive to the RECIPE_OFFER state. Returns (last body, blob_id).

        blob_id is returned for use as ``source_blob_id`` in the recipe-accept
        slots, where _execute_apply_pipeline_recipe resolves it via the session DB.
        storage_path is used as the source ``path`` option for the step-1 commit
        (``_execute_set_source`` requires the path to be under {data_dir}/blobs/).
        """
        blob_id, storage_path = _seed_blob(client, session_id)
        output_path = _outputs_path(client, "out.jsonl")

        _get_guided(client, session_id)
        # Step 1: pick csv
        _respond(client, session_id, chosen=["csv"])
        # Step 1: fill csv options — path must be under {data_dir}/blobs/
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
        # Step 2: pick json sink
        _respond(client, session_id, chosen=["json"])
        # Step 2: fill json options — path must be under {data_dir}/outputs/.
        # collision_policy is required by the json sink validator.
        # Must include "plugin" so the dispatcher persists step_2_sink_intent.
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )
        # Step 2: declare required fields via chosen ("category" matches classify keyword).
        # The backend reads plugin + options from GuidedSession.step_2_sink_intent and
        # combines with chosen to construct SinkOutputResolved.
        # output_path must be under {data_dir}/outputs/ and collision_policy must
        # be set (handle_step_2_sink calls _execute_set_output which validates).
        body = _respond(
            client,
            session_id,
            chosen=["text", "category"],
            custom_inputs=[],
        )
        return body, blob_id

    def test_recipe_offer_accept_produces_terminal_completed(self, composer_test_client: TestClient) -> None:
        """Accepting the recipe offer produces terminal.kind=completed with pipeline YAML."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = self._drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out.jsonl")

        # Verify recipe was offered
        assert recipe_body["next_turn"]["type"] == "recipe_offer"
        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        # Accept the recipe — output_path must be under {data_dir}/outputs/
        body = _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        assert body["terminal"] is not None
        assert body["terminal"]["kind"] == "completed"
        assert body["terminal"]["pipeline_yaml"] is not None
        assert "source:" in body["terminal"]["pipeline_yaml"]
        assert body["next_turn"] is None

    def test_completed_terminal_reflected_in_guided_session(self, composer_test_client: TestClient) -> None:
        """After recipe accept, guided_session.terminal is COMPLETED."""
        session_id = _create_session(composer_test_client)
        _recipe_body, blob_id = self._drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out.jsonl")

        body = _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": "classify-rows-llm-jsonl",
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        gs = body["guided_session"]
        assert gs["terminal"] is not None
        assert gs["terminal"]["kind"] == "completed"
        assert gs["terminal"]["reason"] is None

    def test_recipe_state_survives_roundtrip(self, composer_test_client: TestClient) -> None:
        """After the full walk, a GET /guided re-fetch returns terminal state from DB."""
        session_id = _create_session(composer_test_client)
        recipe_body, blob_id = self._drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out.jsonl")
        offered_recipe = recipe_body["next_turn"]["payload"]["recipe_name"]

        _respond(
            composer_test_client,
            session_id,
            chosen=["accept"],
            edited_values={
                "recipe_name": offered_recipe,
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "required_input_fields": ["text"],
                    "label_field": "category",
                    "output_path": output_path,
                },
            },
        )

        # Re-fetch from DB
        get_body = _get_guided(composer_test_client, session_id)
        gs = get_body["guided_session"]
        assert gs["terminal"] is not None
        assert gs["terminal"]["kind"] == "completed"

    # ---- Contract-violation negative tests for chosen=["accept"] -----------
    # Defends against silent ``.get()`` defaults that rewrite a malformed
    # ``edited_values`` payload into bogus-but-shape-valid data (e.g.
    # ``recipe_name=""``). The widget contract is
    # ``{"recipe_name": str, "slots": Mapping}``; missing keys, a null
    # payload, or a non-string ``recipe_name`` MUST surface as an HTTP 400
    # with the contract cited in the message — not flow downstream as an
    # "unknown recipe ''" error.

    def test_recipe_offer_accept_null_edited_values_returns_400(self, composer_test_client: TestClient) -> None:
        """``edited_values=null`` on accept is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_recipe_offer(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["accept"], "edited_values": None},
        )
        assert resp.status_code == 400, resp.json()
        detail = resp.json()["detail"]
        assert "edited_values" in detail
        assert "null" in detail

    def test_recipe_offer_accept_missing_slots_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``slots`` key on accept is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        self._drive_to_recipe_offer(composer_test_client, session_id)

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": ["accept"],
                "edited_values": {"recipe_name": "classify-rows-llm-jsonl"},
            },
        )
        assert resp.status_code == 400, resp.json()
        assert "slots" in resp.json()["detail"]

    def test_recipe_offer_accept_missing_recipe_name_returns_400(self, composer_test_client: TestClient) -> None:
        """Missing ``recipe_name`` key on accept is a protocol violation (HTTP 400)."""
        session_id = _create_session(composer_test_client)
        _recipe_body, blob_id = self._drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out.jsonl")

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": ["accept"],
                "edited_values": {
                    "slots": {
                        "source_blob_id": blob_id,
                        "classifier_template": "Classify: {{ row['text'] }}",
                        "model": "anthropic/claude-3.5-sonnet",
                        "api_key_secret": "OPENROUTER_API_KEY",
                        "required_input_fields": ["text"],
                        "label_field": "category",
                        "output_path": output_path,
                    },
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        assert "recipe_name" in resp.json()["detail"]

    def test_recipe_offer_accept_empty_recipe_name_returns_400(self, composer_test_client: TestClient) -> None:
        """Empty-string ``recipe_name`` on accept is a protocol violation (HTTP 400).

        Specifically defends against the silent ``str(edited.get("recipe_name", ""))``
        pattern that would have routed ``""`` downstream into ``_RecipeMatch`` and
        failed far away with "unknown recipe ''".
        """
        session_id = _create_session(composer_test_client)
        _recipe_body, blob_id = self._drive_to_recipe_offer(composer_test_client, session_id)
        output_path = _outputs_path(composer_test_client, "out.jsonl")

        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": ["accept"],
                "edited_values": {
                    "recipe_name": "",
                    "slots": {
                        "source_blob_id": blob_id,
                        "classifier_template": "Classify: {{ row['text'] }}",
                        "model": "anthropic/claude-3.5-sonnet",
                        "api_key_secret": "OPENROUTER_API_KEY",
                        "required_input_fields": ["text"],
                        "label_field": "category",
                        "output_path": output_path,
                    },
                },
            },
        )
        assert resp.status_code == 400, resp.json()
        assert "recipe_name" in resp.json()["detail"]


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
# Shadowing note:
#   ``_advance_step_1`` runs BEFORE the dispatcher and raises ValueError on
#   null ``edited_values`` and KeyError on missing ``columns``.  Those errors
#   propagate as HTTP 500, not the dispatcher's HTTP 400.  The dispatcher's
#   null guard and missing-key guard remain as defense-in-depth.  The guard
#   that IS HTTP-reachable as 400 is the ``isinstance(columns_raw, list)``
#   check: ``_advance_step_1``'s ``tuple(str(c) for c in columns_raw)``
#   silently accepts a scalar string (iterating its characters), so the
#   dispatcher catches non-list here before that coercion runs.


class TestStep1InspectAndConfirmAccept:
    def _seed_inspect_and_confirm_history(
        self,
        client: TestClient,
        session_id: str,
    ) -> None:
        """Seed an INSPECT_AND_CONFIRM TurnRecord + step_1_source_intent into the session.

        This bypasses GET /guided because the server-side initial-turn builder
        passes blob_inspection=None unconditionally and so never emits an
        inspect_and_confirm turn.  We inject the state directly via
        save_composition_state — the same pattern used by
        test_progressive_disclosure._seed_terminal_state (line 170).

        step_1_source_intent is populated so that _advance_step_1 can recover
        plugin/options/sample_rows when the POST /respond fires — without it
        _advance_step_1 raises ValueError before the dispatcher can validate
        edited_values.
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
        # it up on POST /respond.  Also seed step_1_source_intent so that
        # _advance_step_1 can build SourceResolved from it.
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
            options={"path": "/data/input.csv"},
            observed_columns=("id", "name", "score"),
            sample_rows=({"id": 1, "name": "Alice", "score": 99},),
        )
        guided = replace(guided, history=(*guided.history, record), step_1_source_intent=intent)
        state = replace(state, guided_session=guided)

        new_composer_meta = {**existing_meta, "guided_session": guided.to_dict()}
        state_d = state.to_dict()
        state_data = CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=False,
            validation_errors=None,
            composer_meta=new_composer_meta,
        )
        asyncio.run(service.save_composition_state(session_uuid, state_data))

    def test_inspect_and_confirm_non_list_columns_returns_400(self, composer_test_client: TestClient) -> None:
        """Non-list ``columns`` at post-advance INSPECT_AND_CONFIRM is a protocol violation (HTTP 400).

        The dispatcher's ``isinstance(columns_raw, list)`` guard fires the 400
        on the raw scalar value.  This guard IS HTTP-reachable as 400 because
        ``_advance_step_1``'s ``tuple(str(c) for c in columns_raw)`` coercion
        would silently accept a scalar string (iterating its characters),
        so the dispatcher must catch it before that coercion runs.
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


# ---------------------------------------------------------------------------
# Error paths: 400 on no GET /guided first, 404 unknown session
# ---------------------------------------------------------------------------


class TestRespondErrorPaths:
    def test_respond_without_prior_get_returns_400(self, composer_test_client: TestClient) -> None:
        """POST /respond without prior GET /guided returns 400 (no turn emitted)."""
        session_id = _create_session(composer_test_client)
        resp = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        # No TurnRecord emitted yet — dispatcher crashes with 400.
        assert resp.status_code == 400

    def test_respond_unknown_session_returns_404(self, composer_test_client: TestClient) -> None:
        """POST /respond for a non-existent session returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = composer_test_client.post(
            f"/api/sessions/{fake_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert resp.status_code == 404
