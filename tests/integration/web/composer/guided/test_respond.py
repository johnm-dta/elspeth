"""Current-schema integration tests for guided response transitions.

These tests exercise the live Step 1 and Step 2 turn contracts, server-held
blob inspection facts, reviewed source/output projections, fail-closed input
validation, and atomic settlement failure behavior. Later authoring stages are
covered separately as their schema-8 response handlers are implemented.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.models import (
    blobs_table,
    composition_states_table,
    guided_operation_events_table,
    guided_operations_table,
)
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.routes._helpers import _SessionComposeLockRegistry
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
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


def _finish_review(client: TestClient, session_id: str, component_kind: str) -> dict:
    """Explicitly finish the current plural component-review stage."""
    return _respond(
        client,
        session_id,
        component_action={"action": "finish", "component_kind": component_kind},
    )


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


def _seed_blob(client: TestClient, session_id: str, *, content: str | None = None) -> tuple[str, str]:
    """Seed a CSV blob and return (blob_id, storage_path).

    The ``storage_path`` is the authoritative file path under
    ``{data_dir}/blobs/{session_id}/`` and can be passed directly as the
    ``path`` option in a source SCHEMA_FORM response (it's already under
    the allowed source directories).

    The route obtains inspection and custody facts from this server-held blob;
    clients never submit those facts in the schema-8 response body.
    """
    content = content or "text,category\nHello world,greeting\nGoodbye,farewell\n"
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


def _independent_guided_peer_app(primary: TestClient) -> FastAPI:
    """Build a second web worker over the same Postgres engine and payloads."""

    primary_app = primary.app
    engine = primary_app.state.session_engine
    data_dir = Path(primary_app.state.settings.data_dir)
    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_engine = engine
    app.state.session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.independent-peer"),
    )
    app.state.blob_service = BlobServiceImpl(engine, data_dir)
    app.state.payload_store = FilesystemPayloadStore(data_dir / "payloads")
    app.state.scoped_secret_resolver = primary_app.state.scoped_secret_resolver
    app.state.settings = primary_app.state.settings
    app.state.composer_service = type(primary_app.state.composer_service)()
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = primary_app.state.catalog_service
    app.state.web_plugin_policy = primary_app.state.web_plugin_policy
    app.state.operator_profile_registry = primary_app.state.operator_profile_registry
    app.state.plugin_snapshot_factory = primary_app.state.plugin_snapshot_factory
    app.state.composer_recorder = BufferingRecorder()
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.session_compose_lock_registry = _SessionComposeLockRegistry()
    app.include_router(create_session_router())
    return app


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

        assert advanced["guided_session"]["step"] == "step_1_source"
        assert advanced["next_turn"]["type"] == "review_components"
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

    def test_schema_form_response_requires_explicit_finish_to_advance_to_step_2(self, composer_test_client: TestClient) -> None:
        """A resolved source remains reviewable until the operator finishes."""
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

        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "review_components"
        finished = _finish_review(composer_test_client, session_id, "source")
        assert finished["guided_session"]["step"] == "step_2_sink"
        assert finished["next_turn"]["type"] == "single_select"
        assert finished["next_turn"]["step_index"] == 1

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

        body = _finish_review(composer_test_client, session_id, "source")
        options = body["next_turn"]["payload"]["options"]
        ids = [o["id"] for o in options]
        assert "json" in ids, f"json sink not found in options: {ids}"


# ---------------------------------------------------------------------------
# Step 2 intra-step — sink SINGLE_SELECT → SCHEMA_FORM → MULTI_SELECT
# ---------------------------------------------------------------------------


class TestStep2IntraStep:
    def _drive_to_step_2_single_select(
        self,
        client: TestClient,
        session_id: str,
        *,
        blob_content: str | None = None,
    ) -> dict:
        """Drive to the Step 2 initial SINGLE_SELECT state."""
        _seed_blob(client, session_id, content=blob_content)
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
        _respond(client, session_id, edited_values={"columns": ["text", "category"]})
        return _finish_review(client, session_id, "source")

    def _stage_proposal(
        self,
        client: TestClient,
        session_id: str,
        *,
        filename: str,
        blob_content: str | None = None,
    ) -> dict:
        self._drive_to_step_2_single_select(client, session_id, blob_content=blob_content)
        _respond(client, session_id, chosen=["json"])
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(client, filename),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(client, session_id, chosen=["text"], custom_inputs=[])
        return _finish_review(client, session_id, "output")

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

    def test_multi_select_response_atomically_stages_step_3_proposal(self, composer_test_client: TestClient) -> None:
        """Reviewed sink facts and the private proposal become durable together."""
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
        body = _finish_review(composer_test_client, session_id, "output")
        assert body["guided_session"]["step"] == "step_3_transforms"
        assert body["next_turn"]["type"] == "propose_pipeline"
        proposal = body["next_turn"]["payload"]
        active = _full_guided_session(body)["active_proposal"]
        assert active["proposal_id"] == proposal["proposal_id"]
        assert active["draft_hash"] == proposal["draft_hash"]

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

        assert body["guided_session"]["step"] == "step_2_sink"
        assert body["next_turn"]["type"] == "review_components"

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs["outputs"] == []
        full_guided = _full_guided_session(body)
        assert len(full_guided["reviewed_outputs"]) == 1
        output = next(iter(full_guided["reviewed_outputs"].values()))
        assert output["plugin"] == "json"
        assert output["required_fields"] == ["text", "category"]

    def test_operator_reject_atomically_preserves_reviewed_facts_and_clears_reference(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "reject.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        staged = _finish_review(composer_test_client, session_id, "output")
        turn = staged["next_turn"]

        rejected = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": turn["payload"]["proposal_id"],
                "draft_hash": turn["payload"]["draft_hash"],
                "control_signal": "reject",
            },
        )

        assert rejected.status_code == 200, rejected.json()
        body = rejected.json()
        guided = _full_guided_session(body)
        assert guided["active_proposal"] is None
        assert guided["active_edit_target"] is None
        assert guided["reviewed_sources"]
        assert guided["reviewed_outputs"]
        assert body["next_turn"] is None

    def test_exact_reviewed_non_blob_source_path_can_stage_and_accept(self, composer_test_client: TestClient) -> None:
        session_id = _create_session(composer_test_client)
        data_dir = Path(composer_test_client.app.state.settings.data_dir)
        source_path = data_dir / "blobs" / "operator-reviewed.csv"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("text,category\nHello,greeting\n", encoding="utf-8")

        _get_guided(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["csv"])
        source_reviewed = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {
                    "path": str(source_path),
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                },
            },
        )
        assert source_reviewed["next_turn"]["type"] == "review_components"
        _finish_review(composer_test_client, session_id, "source")
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "non-blob-accepted.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=[], custom_inputs=["text"])
        staged = _finish_review(composer_test_client, session_id, "output")
        proposal = staged["next_turn"]["payload"]

        accepted = _post_current_response(
            composer_test_client,
            session_id,
            proposal_id=proposal["proposal_id"],
            draft_hash=proposal["draft_hash"],
            chosen=["accept"],
        )

        assert accepted.status_code == 200, accepted.json()
        accepted_source = next(iter(accepted.json()["composition_state"]["sources"].values()))
        assert accepted_source["options"]["path"] == str(source_path)

    def test_target_only_revision_atomically_supersedes_old_and_publishes_successor(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="revise.jsonl")
        old_turn = staged["next_turn"]
        old_payload = old_turn["payload"]
        old_guided = _full_guided_session(staged)
        target = old_payload["edit_targets"][0]

        revised = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": old_turn["turn_token"],
                "proposal_id": old_payload["proposal_id"],
                "draft_hash": old_payload["draft_hash"],
                "edit_target": target,
            },
        )

        assert revised.status_code == 200, revised.json()
        body = revised.json()
        successor_turn = body["next_turn"]
        assert successor_turn["type"] == "propose_pipeline"
        successor = successor_turn["payload"]
        assert successor["proposal_id"] != old_payload["proposal_id"]
        assert successor["draft_hash"] != old_payload["draft_hash"]
        guided = _full_guided_session(body)
        assert guided["reviewed_sources"] == old_guided["reviewed_sources"]
        assert guided["reviewed_outputs"] == old_guided["reviewed_outputs"]
        assert guided["deferred_intents"] == old_guided["deferred_intents"]
        assert guided["active_proposal"]["proposal_id"] == successor["proposal_id"]
        assert guided["active_proposal"]["supersedes_proposal_id"] == old_payload["proposal_id"]
        assert guided["active_proposal"]["supersedes_draft_hash"] == old_payload["draft_hash"]

        proposals = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert proposals[old_payload["proposal_id"]].status == "rejected"
        assert proposals[successor["proposal_id"]].status == "pending"
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        old_events = [event for event in events if str(event.proposal_id) == old_payload["proposal_id"]]
        assert [event.event_type for event in old_events] == ["proposal.created", "proposal.rejected"]
        assert old_events[-1].payload["reason_code"] == "superseded"

        old_accept = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": old_turn["turn_token"],
                "proposal_id": old_payload["proposal_id"],
                "draft_hash": old_payload["draft_hash"],
                "chosen": ["accept"],
            },
        )
        assert old_accept.status_code == 409, old_accept.json()
        proposals_after = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert proposals_after[old_payload["proposal_id"]].status == "rejected"

    def test_revision_rejects_non_null_edited_values_without_mutation(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="revise-shape.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]

        rejected = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": payload["proposal_id"],
                "draft_hash": payload["draft_hash"],
                "edit_target": payload["edit_targets"][0],
                "edited_values": {"action": "regenerate"},
            },
        )

        assert rejected.status_code == 400, rejected.json()
        assert rejected.json()["detail"] == "Guided proposal action has an invalid closed shape."
        current = _get_guided(composer_test_client, session_id)
        assert current["next_turn"]["payload"] == payload
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1
        assert proposals[0].status == "pending"

    def test_proposal_lifecycle_surfaces_never_expose_private_review_canaries(
        self,
        composer_test_client: TestClient,
    ) -> None:
        canaries = (
            "RAW-INLINE-CONTENT-CANARY",
            "CREDENTIAL-LITERAL-CANARY",
            "RESOLVED-SECRET-CANARY",
            "RAW-VALIDATION-TEXT-CANARY",
            "RAW-PROVIDER-ERROR-CANARY",
        )
        blob_content = "text,category\n" + "\n".join(f"{canary},private" for canary in canaries) + "\n"

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(
            composer_test_client,
            session_id,
            filename="canary-accept.jsonl",
            blob_content=blob_content,
        )
        old_turn = staged["next_turn"]
        old_payload = old_turn["payload"]
        revised = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": old_turn["turn_token"],
                "proposal_id": old_payload["proposal_id"],
                "draft_hash": old_payload["draft_hash"],
                "edit_target": old_payload["edit_targets"][0],
            },
        )
        assert revised.status_code == 200, revised.json()
        restored = _get_guided(composer_test_client, session_id)
        successor_turn = revised.json()["next_turn"]
        successor = successor_turn["payload"]
        accepted = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": successor_turn["turn_token"],
                "proposal_id": successor["proposal_id"],
                "draft_hash": successor["draft_hash"],
                "chosen": ["accept"],
            },
        )
        assert accepted.status_code == 200, accepted.json()
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))

        rejected_session_id = _create_session(composer_test_client)
        rejected_stage = self._stage_proposal(
            composer_test_client,
            rejected_session_id,
            filename="canary-reject.jsonl",
            blob_content=blob_content,
        )
        rejected_turn = rejected_stage["next_turn"]
        rejected_payload = rejected_turn["payload"]
        rejected = composer_test_client.post(
            f"/api/sessions/{rejected_session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": rejected_turn["turn_token"],
                "proposal_id": rejected_payload["proposal_id"],
                "draft_hash": rejected_payload["draft_hash"],
                "control_signal": "reject",
            },
        )
        assert rejected.status_code == 200, rejected.json()
        rejected_restored = _get_guided(composer_test_client, rejected_session_id)
        rejected_events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(rejected_session_id)))

        public_surfaces = (
            staged,
            revised.json(),
            restored,
            accepted.json(),
            rejected_stage,
            rejected.json(),
            rejected_restored,
            *(event.payload for event in (*events, *rejected_events)),
        )
        rendered = repr(public_surfaces)
        assert all(canary not in rendered for canary in canaries)

    @pytest.mark.parametrize(
        "composer_test_client",
        (
            pytest.param("sqlite", id="sqlite"),
            pytest.param("postgres", id="postgres", marks=pytest.mark.testcontainer),
        ),
        indirect=True,
    )
    def test_failed_proposal_planning_retains_only_closed_failure_code(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failure_canaries = (
            "RAW-VALIDATION-TEXT-CANARY",
            "RAW-PROVIDER-ERROR-CANARY",
            "CREDENTIAL-LITERAL-CANARY",
            "RESOLVED-SECRET-CANARY",
        )
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "failed-plan.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )

        async def fail_planner(*, recorder, **_kwargs):
            now = datetime.now(UTC)
            recorder.record_llm_call(
                ComposerLLMCall(
                    model_requested="planner-model",
                    model_returned=None,
                    status=ComposerLLMCallStatus.API_ERROR,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    latency_ms=1,
                    provider_request_id="failed-planner-request",
                    messages_hash="a" * 64,
                    tools_spec_hash=None,
                    declared_tool_names=(),
                    started_at=now,
                    finished_at=now,
                    error_class="ProviderError",
                    error_message=" | ".join(failure_canaries),
                    temperature=None,
                    seed=None,
                    reasoning_content=failure_canaries[0],
                    reasoning_details={"provider": failure_canaries[1]},
                    thinking_blocks={"validation": failure_canaries[2]},
                )
            )
            raise RuntimeError(" | ".join(failure_canaries))

        monkeypatch.setattr(
            composer_test_client.app.state.composer_service,
            "plan_guided_pipeline",
            fail_planner,
        )
        _respond(
            composer_test_client,
            session_id,
            chosen=["text"],
            custom_inputs=[],
        )
        failed = _post_current_response(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "output"},
        )
        assert failed.status_code == 500, failed.json()
        restored = _get_guided(composer_test_client, session_id)
        with composer_test_client.app.state.session_engine.connect() as conn:
            operation_rows = (
                conn.execute(select(guided_operations_table).where(guided_operations_table.c.session_id == session_id)).mappings().all()
            )
            operation_events = (
                conn.execute(select(guided_operation_events_table).where(guided_operation_events_table.c.session_id == session_id))
                .mappings()
                .all()
            )
        audit_messages = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
        failed_calls = [
            envelope["call"]
            for message in audit_messages
            for envelope in message.tool_calls
            if envelope.get("_kind") == "llm_call_audit" and envelope.get("call", {}).get("status") == ComposerLLMCallStatus.API_ERROR.value
        ]
        failed_rows = [row for row in operation_rows if row["status"] == "failed"]
        assert len(failed_rows) == 1
        assert failed_rows[0]["failure_code"] == "operation_failed"
        assert len(failed_calls) == 1
        assert failed_calls[0]["error_class"] == "ProviderError"
        assert failed_calls[0]["error_message"] is None
        assert failed_calls[0]["reasoning_content"] is None
        assert failed_calls[0]["reasoning_details"] is None
        assert failed_calls[0]["thinking_blocks"] is None
        rendered = repr((failed.json(), restored, operation_rows, operation_events, audit_messages))
        assert all(canary not in rendered for canary in failure_canaries)

    @pytest.mark.parametrize(
        "composer_test_client",
        (
            pytest.param("sqlite", id="sqlite"),
            pytest.param("postgres", id="postgres", marks=pytest.mark.testcontainer),
        ),
        indirect=True,
    )
    def test_failed_planning_audit_insert_failure_rolls_back_operation_failure(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failure_canary = "FAILED-PLANNER-AUDIT-ROLLBACK-CANARY"
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "failed-plan-audit-rollback.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        messages_before = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))

        async def fail_planner(*, recorder, **_kwargs):
            now = datetime.now(UTC)
            recorder.record_llm_call(
                ComposerLLMCall(
                    model_requested="planner-model",
                    model_returned=None,
                    status=ComposerLLMCallStatus.API_ERROR,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    latency_ms=1,
                    provider_request_id="failed-planner-request",
                    messages_hash="b" * 64,
                    tools_spec_hash=None,
                    declared_tool_names=(),
                    started_at=now,
                    finished_at=now,
                    error_class="ProviderError",
                    error_message=failure_canary,
                    temperature=None,
                    seed=None,
                )
            )
            raise RuntimeError(failure_canary)

        def fail_audit_insert(*_args, **_kwargs):
            raise RuntimeError("safe audit insert failure")

        monkeypatch.setattr(
            composer_test_client.app.state.composer_service,
            "plan_guided_pipeline",
            fail_planner,
        )
        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "_insert_prepared_guided_audit_rows_on_connection",
            fail_audit_insert,
        )
        operation_id = str(uuid4())
        current = _get_guided(composer_test_client, session_id)
        with pytest.raises(
            AuditIntegrityError,
            match="Guided RESPOND could not record its terminal failure",
        ) as exc_info:
            composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={
                    "operation_id": operation_id,
                    "turn_token": current["next_turn"]["turn_token"],
                    "component_action": {"action": "finish", "component_kind": "output"},
                },
            )

        with composer_test_client.app.state.session_engine.connect() as conn:
            operation = (
                conn.execute(
                    select(guided_operations_table)
                    .where(guided_operations_table.c.session_id == session_id)
                    .where(guided_operations_table.c.operation_id == operation_id)
                )
                .mappings()
                .one()
            )
        assert operation["status"] == "in_progress"
        assert operation["failure_code"] is None
        assert operation["settled_at"] is None
        assert asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None)) == messages_before
        assert failure_canary not in repr((exc_info.value, operation))

    @pytest.mark.parametrize(
        "composer_test_client",
        (
            pytest.param("sqlite", id="sqlite"),
            pytest.param("postgres", id="postgres", marks=pytest.mark.testcontainer),
        ),
        indirect=True,
    )
    def test_accept_vs_revise_race_has_one_winner_and_exact_revision_replay(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="revise-race.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]
        target = payload["edit_targets"][0]
        planner_calls = 0
        original_planner = composer_test_client.app.state.composer_service.plan_guided_pipeline
        revision_request = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "proposal_id": payload["proposal_id"],
            "draft_hash": payload["draft_hash"],
            "edit_target": target,
        }
        accept_request = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "proposal_id": payload["proposal_id"],
            "draft_hash": payload["draft_hash"],
            "chosen": ["accept"],
        }

        async def race():
            nonlocal planner_calls
            planner_entered = asyncio.Event()
            release_planner = asyncio.Event()

            async def blocking_planner(**kwargs):
                nonlocal planner_calls
                planner_calls += 1
                planner_entered.set()
                await release_planner.wait()
                return await original_planner(**kwargs)

            monkeypatch.setattr(
                composer_test_client.app.state.composer_service,
                "plan_guided_pipeline",
                blocking_planner,
            )
            async with AsyncClient(
                transport=ASGITransport(app=composer_test_client.app),
                base_url="http://test",
            ) as client:
                revision_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=revision_request,
                    )
                )
                await asyncio.wait_for(planner_entered.wait(), timeout=5)
                # Revision planning runs while both guided admission and the
                # per-session compose lock are held. Reaching this point
                # therefore fences accept behind revision publication; once
                # released, accept must fail stale preflight rather than race
                # the old pending row inside either settlement transaction.
                during_planning = await composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id))
                accept_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=accept_request,
                    )
                )
                await asyncio.sleep(0)
                release_planner.set()
                revised, accepted = await asyncio.wait_for(
                    asyncio.gather(revision_task, accept_task),
                    timeout=10,
                )
                replayed = await client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json=revision_request,
                )
                return during_planning, revised, accepted, replayed

        during_planning, revised, accepted, replayed = asyncio.run(race())
        assert len(during_planning) == 1
        assert during_planning[0].status == "pending"

        assert revised.status_code == 200, revised.json()
        assert accepted.status_code == 409, accepted.json()
        winner_body = revised.json()
        successor_id = winner_body["next_turn"]["payload"]["proposal_id"]
        proposals = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert set(proposals) == {payload["proposal_id"], successor_id}
        assert proposals[payload["proposal_id"]].status == "rejected"
        assert proposals[successor_id].status == "pending"
        events_before_replay = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        assert [event.event_type for event in events_before_replay if str(event.proposal_id) == payload["proposal_id"]] == [
            "proposal.created",
            "proposal.rejected",
        ]
        assert [event.event_type for event in events_before_replay if str(event.proposal_id) == successor_id] == ["proposal.created"]

        assert replayed.status_code == 200, replayed.json()
        assert replayed.json() == winner_body
        assert planner_calls == 1
        events_after_replay = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        assert events_after_replay == events_before_replay

    @pytest.mark.parametrize(
        "composer_test_client",
        (pytest.param("postgres", marks=pytest.mark.testcontainer),),
        indirect=True,
    )
    @pytest.mark.parametrize("db_winner", ("accept", "revise"))
    def test_independent_postgres_workers_serialize_accept_vs_revise_at_settlement(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        db_winner: str,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename=f"independent-{db_winner}.jsonl")
        turn = staged["next_turn"]
        proposal_payload = turn["payload"]
        target = proposal_payload["edit_targets"][0]
        accept_operation_id = str(uuid4())
        revise_operation_id = str(uuid4())
        accept_request = {
            "operation_id": accept_operation_id,
            "turn_token": turn["turn_token"],
            "proposal_id": proposal_payload["proposal_id"],
            "draft_hash": proposal_payload["draft_hash"],
            "chosen": ["accept"],
        }
        revise_request = {
            "operation_id": revise_operation_id,
            "turn_token": turn["turn_token"],
            "proposal_id": proposal_payload["proposal_id"],
            "draft_hash": proposal_payload["draft_hash"],
            "edit_target": target,
        }

        primary_app = composer_test_client.app
        peer_app = _independent_guided_peer_app(composer_test_client)
        assert primary_app.state.session_service is not peer_app.state.session_service
        assert primary_app.state.composer_service is not peer_app.state.composer_service
        assert primary_app.state.session_engine is peer_app.state.session_engine
        assert primary_app.state.session_compose_lock_registry is not peer_app.state.session_compose_lock_registry

        accept_service = primary_app.state.session_service
        revise_service = peer_app.state.session_service
        original_accept = accept_service.accept_guided_pipeline_proposal
        original_stage = revise_service.stage_guided_pipeline_proposal
        original_accept_lock = accept_service._session_write_lock
        original_revise_lock = revise_service._session_write_lock
        accept_ready = asyncio.Event()
        revise_ready = asyncio.Event()
        release_winner = asyncio.Event()
        winner_db_locked = threading.Event()
        loser_db_attempted = threading.Event()
        accept_armed = threading.Event()
        revise_armed = threading.Event()
        accept_commands = []
        revise_commands = []

        async def gated_accept(command, *, payload_store=None):
            accept_commands.append(command)
            assert command.invocation.tool_name == "set_pipeline"
            assert command.invocation.status.value == "success"
            accept_ready.set()
            await release_winner.wait()
            if db_winner != "accept":
                assert await asyncio.to_thread(winner_db_locked.wait, 10)
            accept_armed.set()
            try:
                return await original_accept(command, payload_store=payload_store)
            finally:
                accept_armed.clear()

        async def gated_stage(command, *, payload_store=None):
            revise_commands.append(command)
            assert str(command.supersedes_proposal_id) == proposal_payload["proposal_id"]
            revise_ready.set()
            await release_winner.wait()
            if db_winner != "revise":
                assert await asyncio.to_thread(winner_db_locked.wait, 10)
            revise_armed.set()
            try:
                return await original_stage(command, payload_store=payload_store)
            finally:
                revise_armed.clear()

        @contextmanager
        def accept_db_lock(conn, locked_session_id):
            if not accept_armed.is_set():
                with original_accept_lock(conn, locked_session_id):
                    yield
                return
            if db_winner == "accept":
                with original_accept_lock(conn, locked_session_id):
                    winner_db_locked.set()
                    assert loser_db_attempted.wait(10)
                    yield
                return
            loser_db_attempted.set()
            with original_accept_lock(conn, locked_session_id):
                yield

        @contextmanager
        def revise_db_lock(conn, locked_session_id):
            if not revise_armed.is_set():
                with original_revise_lock(conn, locked_session_id):
                    yield
                return
            if db_winner == "revise":
                with original_revise_lock(conn, locked_session_id):
                    winner_db_locked.set()
                    assert loser_db_attempted.wait(10)
                    yield
                return
            loser_db_attempted.set()
            with original_revise_lock(conn, locked_session_id):
                yield

        monkeypatch.setattr(accept_service, "accept_guided_pipeline_proposal", gated_accept)
        monkeypatch.setattr(revise_service, "stage_guided_pipeline_proposal", gated_stage)
        monkeypatch.setattr(accept_service, "_session_write_lock", accept_db_lock)
        monkeypatch.setattr(revise_service, "_session_write_lock", revise_db_lock)

        engine = primary_app.state.session_engine
        with engine.connect() as conn:
            state_count_before = conn.execute(
                select(composition_states_table.c.id).where(composition_states_table.c.session_id == session_id)
            ).all()

        async def race_and_replay():
            async with (
                AsyncClient(transport=ASGITransport(app=primary_app), base_url="http://accept-worker") as accept_client,
                AsyncClient(transport=ASGITransport(app=peer_app), base_url="http://revise-worker") as revise_client,
            ):
                accept_task = asyncio.create_task(accept_client.post(f"/api/sessions/{session_id}/guided/respond", json=accept_request))
                revise_task = asyncio.create_task(revise_client.post(f"/api/sessions/{session_id}/guided/respond", json=revise_request))
                await asyncio.wait_for(asyncio.gather(accept_ready.wait(), revise_ready.wait()), timeout=10)
                release_winner.set()
                accept_response, revise_response = await asyncio.wait_for(
                    asyncio.gather(accept_task, revise_task),
                    timeout=20,
                )
                accept_replay = await revise_client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json=accept_request,
                )
                revise_replay = await accept_client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json=revise_request,
                )
                accept_conflict = await revise_client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json={**revise_request, "operation_id": accept_operation_id},
                )
                revise_conflict = await accept_client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json={**accept_request, "operation_id": revise_operation_id},
                )
                return (
                    accept_response,
                    revise_response,
                    accept_replay,
                    revise_replay,
                    accept_conflict,
                    revise_conflict,
                )

        (
            accept_response,
            revise_response,
            accept_replay,
            revise_replay,
            accept_conflict,
            revise_conflict,
        ) = asyncio.run(race_and_replay())

        responses = {"accept": accept_response, "revise": revise_response}
        replays = {"accept": accept_replay, "revise": revise_replay}
        loser = "revise" if db_winner == "accept" else "accept"
        assert responses[db_winner].status_code == 200, responses[db_winner].json()
        assert responses[loser].status_code == 409, responses[loser].json()
        assert responses[loser].json()["detail"]["failure_code"] == "stale_conflict"
        for action in ("accept", "revise"):
            assert replays[action].status_code == responses[action].status_code
            assert replays[action].json() == responses[action].json()
        assert accept_conflict.status_code == revise_conflict.status_code == 409
        assert len(accept_commands) == len(revise_commands) == 1
        assert winner_db_locked.is_set() and loser_db_attempted.is_set()

        service = primary_app.state.session_service
        proposals = {str(proposal.id): proposal for proposal in asyncio.run(service.list_composition_proposals(UUID(session_id)))}
        events = asyncio.run(service.list_proposal_events(UUID(session_id)))
        messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        dispatches = [
            envelope
            for message in messages
            for envelope in message.tool_calls
            if envelope.get("invocation", {}).get("tool_name") == "set_pipeline"
            and envelope.get("invocation", {}).get("status") == "success"
        ]
        with engine.connect() as conn:
            operation_rows = (
                conn.execute(
                    select(guided_operations_table)
                    .where(guided_operations_table.c.session_id == session_id)
                    .where(guided_operations_table.c.operation_id.in_((accept_operation_id, revise_operation_id)))
                )
                .mappings()
                .all()
            )
            operation_event_rows = (
                conn.execute(
                    select(guided_operation_events_table)
                    .where(guided_operation_events_table.c.session_id == session_id)
                    .where(guided_operation_events_table.c.operation_id.in_((accept_operation_id, revise_operation_id)))
                    .order_by(guided_operation_events_table.c.operation_id, guided_operation_events_table.c.sequence)
                )
                .mappings()
                .all()
            )
            state_count_after = conn.execute(
                select(composition_states_table.c.id).where(composition_states_table.c.session_id == session_id)
            ).all()
        operations = {row["operation_id"]: row for row in operation_rows}
        operation_events = {
            operation_id: [row["event_kind"] for row in operation_event_rows if row["operation_id"] == operation_id]
            for operation_id in (accept_operation_id, revise_operation_id)
        }
        winner_operation_id = accept_operation_id if db_winner == "accept" else revise_operation_id
        loser_operation_id = revise_operation_id if db_winner == "accept" else accept_operation_id
        assert set(operations) == {accept_operation_id, revise_operation_id}
        assert operations[winner_operation_id]["status"] == "completed"
        assert operations[winner_operation_id]["failure_code"] is None
        assert operations[winner_operation_id]["result_kind"] == "composition_state"
        assert operations[winner_operation_id]["result_state_id"] is not None
        assert operations[loser_operation_id]["status"] == "failed"
        assert operations[loser_operation_id]["failure_code"] == "stale_conflict"
        assert operations[loser_operation_id]["result_kind"] is None
        assert operations[loser_operation_id]["result_state_id"] is None
        assert operations[loser_operation_id]["proposal_id"] is None
        assert operation_events[winner_operation_id] == ["claimed", "renewed", "completed"]
        assert operation_events[loser_operation_id] == ["claimed", "renewed", "failed"]
        assert len(state_count_after) == len(state_count_before) + 1

        original_id = proposal_payload["proposal_id"]
        original_events = [event.event_type for event in events if str(event.proposal_id) == original_id]
        current = _get_guided(composer_test_client, session_id)
        if db_winner == "accept":
            assert set(proposals) == {original_id}
            assert len(events) == 2
            assert proposals[original_id].status == "committed"
            assert original_events == ["proposal.created", "proposal.accepted"]
            assert len(dispatches) == 1
            assert current["next_turn"]["type"] == "confirm_wiring"
        else:
            successor_id = revise_response.json()["next_turn"]["payload"]["proposal_id"]
            assert set(proposals) == {original_id, successor_id}
            assert len(events) == 3
            assert proposals[original_id].status == "rejected"
            assert proposals[successor_id].status == "pending"
            assert original_events == ["proposal.created", "proposal.rejected"]
            assert [event.event_type for event in events if str(event.proposal_id) == successor_id] == ["proposal.created"]
            assert dispatches == []
            assert current["next_turn"]["type"] == "propose_pipeline"
            assert current["next_turn"]["payload"]["proposal_id"] == successor_id

    def test_accept_atomically_commits_pipeline_consumes_coverage_and_advances_to_wire(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_id = _create_session(composer_test_client)
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "accepted.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        staged = _finish_review(composer_test_client, session_id, "output")
        turn = staged["next_turn"]
        with composer_test_client.app.state.session_engine.connect() as conn:
            storage_paths = tuple(conn.execute(select(blobs_table.c.storage_path).where(blobs_table.c.session_id == session_id)).scalars())
        assert storage_paths
        proposal = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))[0]
        assert "blob:" in repr(proposal.arguments_json)
        for storage_path in storage_paths:
            assert storage_path not in json.dumps(staged)
            assert storage_path not in repr(proposal.arguments_json)

        operation_id = str(uuid4())
        accept_request = {
            "operation_id": operation_id,
            "turn_token": turn["turn_token"],
            "proposal_id": turn["payload"]["proposal_id"],
            "draft_hash": turn["payload"]["draft_hash"],
            "chosen": ["accept"],
        }
        accepted = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=accept_request,
        )

        assert accepted.status_code == 200, accepted.json()
        body = accepted.json()
        guided = _full_guided_session(body)
        assert guided["active_proposal"] is None
        assert guided["deferred_intents"] == []
        assert body["guided_session"]["step"] == "step_4_wire"
        assert body["next_turn"]["type"] == "confirm_wiring"
        assert body["composition_state"]["outputs"]
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        replayed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=accept_request,
        )
        assert replayed.status_code == 200, replayed.json()
        assert replayed.json() == body
        restored = _get_guided(composer_test_client, session_id)
        for storage_path in storage_paths:
            assert storage_path not in json.dumps(body)
            assert storage_path not in replayed.text
            assert storage_path not in json.dumps(restored)
            assert all(storage_path not in repr(event.payload) for event in events)

    def test_accept_failure_after_dispatch_audit_insert_rolls_back_entire_cohort(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from elspeth.web.sessions import service as service_module

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-audit-rollback.jsonl")
        turn = staged["next_turn"]
        messages_before = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
        state_before = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        events_before = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))

        def fail_after_dispatch_audit_insert(*_args, **_kwargs):
            raise RuntimeError("safe failure after dispatch audit insert")

        monkeypatch.setattr(
            service_module,
            "_persisted_pipeline_dispatch_content_hashes",
            fail_after_dispatch_audit_insert,
        )
        failed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": turn["payload"]["proposal_id"],
                "draft_hash": turn["payload"]["draft_hash"],
                "chosen": ["accept"],
            },
        )

        assert failed.status_code == 500, failed.json()
        assert asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None)) == messages_before
        assert asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id))) == events_before
        state_after = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        assert state_before is not None and state_after is not None and state_after.id == state_before.id
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"

    def test_accept_cancellation_before_service_settlement_persists_no_dispatch_audit(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-cancel-before-service.jsonl")
        turn = staged["next_turn"]
        messages_before = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
        entered = asyncio.Event()
        never = asyncio.Event()

        async def blocked_accept(*_args, **_kwargs):
            entered.set()
            await never.wait()

        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "accept_guided_pipeline_proposal",
            blocked_accept,
        )

        async def cancel_before_settlement() -> None:
            async with AsyncClient(
                transport=ASGITransport(app=composer_test_client.app),
                base_url="http://test",
            ) as client:
                task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json={
                            "operation_id": str(uuid4()),
                            "turn_token": turn["turn_token"],
                            "proposal_id": turn["payload"]["proposal_id"],
                            "draft_hash": turn["payload"]["draft_hash"],
                            "chosen": ["accept"],
                        },
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(cancel_before_settlement())

        assert asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None)) == messages_before
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        assert [event.event_type for event in events] == ["proposal.created"]
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"

    def test_accept_cancellation_during_worker_settlement_is_exactly_replayable(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-cancel-during-worker.jsonl")
        turn = staged["next_turn"]
        request_body = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "proposal_id": turn["payload"]["proposal_id"],
            "draft_hash": turn["payload"]["draft_hash"],
            "chosen": ["accept"],
        }
        service = composer_test_client.app.state.session_service
        original_insert = service._insert_prepared_guided_audit_rows_on_connection
        audit_inserted = threading.Event()
        release_worker = threading.Event()

        def pause_after_audit_insert(*args, **kwargs):
            records = original_insert(*args, **kwargs)
            audit_inserted.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release guided accept worker")
            return records

        monkeypatch.setattr(
            service,
            "_insert_prepared_guided_audit_rows_on_connection",
            pause_after_audit_insert,
        )

        async def cancel_and_replay():
            async with AsyncClient(
                transport=ASGITransport(app=composer_test_client.app),
                base_url="http://test",
            ) as client:
                first = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=request_body,
                    )
                )
                inserted = await asyncio.to_thread(audit_inserted.wait, 5)
                assert inserted is True
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                release_worker.set()
                return await asyncio.wait_for(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=request_body,
                    ),
                    timeout=10,
                )

        replayed = asyncio.run(cancel_and_replay())

        assert replayed.status_code == 200, replayed.json()
        proposals = asyncio.run(service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "committed"
        messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        dispatches = [
            envelope
            for message in messages
            for envelope in message.tool_calls
            if envelope.get("invocation", {}).get("tool_name") == "set_pipeline"
            and envelope.get("invocation", {}).get("status") == "success"
        ]
        assert len(dispatches) == 1

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
        body = _finish_review(composer_test_client, session_id, "output")

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

        _respond(
            composer_test_client,
            session_id,
            chosen=["text", "category"],
            custom_inputs=[],
        )
        before = _get_guided(composer_test_client, session_id)
        assert before["guided_session"]["step"] == "step_2_sink"
        before_reviewed_outputs = _full_guided_session(before)["reviewed_outputs"]
        assert before_reviewed_outputs
        before_outputs = before["composition_state"]["outputs"]

        async def fail_settlement(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("safe synthetic settlement failure")

        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "stage_guided_pipeline_proposal",
            fail_settlement,
        )
        resp = _post_current_response(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "output"},
        )
        assert resp.status_code == 500, resp.json()
        assert resp.json()["detail"]["failure_code"] == "operation_failed"

        after = _get_guided(composer_test_client, session_id)
        assert after["guided_session"]["step"] == "step_2_sink"
        assert _full_guided_session(after)["reviewed_outputs"] == before_reviewed_outputs
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
        _finish_review(client, session_id, "source")
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

    def test_inspect_and_confirm_commits_source_then_finish_advances_to_step_2(self, composer_test_client: TestClient) -> None:
        """A valid inspection commits review custody; explicit finish advances.

        Both authoritative surfaces agree after the inspection response
        (elspeth-948eb9c0b8 C-3(b), Step-1 mirror of the Step-2 fix).
        """
        session_id = _create_session(composer_test_client)
        self._seed_inspect_and_confirm_history(composer_test_client, session_id)
        body = _respond(composer_test_client, session_id, edited_values={"columns": ["text", "category"]})

        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "review_components"
        full_guided = _full_guided_session(body)
        assert full_guided["pending_source_intents"] == {}
        assert len(full_guided["reviewed_sources"]) == 1
        source = next(iter(full_guided["reviewed_sources"].values()))
        assert source["plugin"] == "csv"
        assert source["observed_columns"] == ["text", "category"]

        cs = body["composition_state"]
        assert cs is not None, "composition_state missing from response"
        assert cs["sources"] == {}
        finished = _finish_review(composer_test_client, session_id, "source")
        assert finished["guided_session"]["step"] == "step_2_sink"

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
