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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.sql.dml import Insert, Update

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.capability_skill import PlannerCapabilityManifest
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    guided_operation_events_table,
    guided_operations_table,
    proposal_events_table,
)
from elspeth.web.sessions.protocol import CompositionStateData, GuidedOperationClaimed
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.routes._helpers import _SessionComposeLockRegistry
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _PlannerFunction:
    name: str
    arguments: str


@dataclass
class _PlannerToolCall:
    id: str
    function: _PlannerFunction


@dataclass
class _PlannerMessage:
    content: str | None
    tool_calls: list[_PlannerToolCall]


@dataclass
class _PlannerChoice:
    message: _PlannerMessage


@dataclass
class _PlannerResponse:
    choices: list[_PlannerChoice]
    usage: Mapping[str, object]
    model: str = "provider/guided-planner-v1"
    id: str = "guided-planner-request-1"


def _planner_terminal_response() -> _PlannerResponse:
    return _PlannerResponse(
        choices=[
            _PlannerChoice(
                message=_PlannerMessage(
                    content=None,
                    tool_calls=[
                        _PlannerToolCall(
                            id="guided-terminal",
                            function=_PlannerFunction(
                                name="emit_pipeline_proposal",
                                arguments=json.dumps({"pipeline": {}}),
                            ),
                        )
                    ],
                )
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
    )


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


def _review_wiring(client: TestClient, session_id: str) -> dict:
    """Review the current pending proposal without committing it."""
    current = _get_guided(client, session_id)
    turn = current["next_turn"]
    assert turn["type"] == "propose_pipeline"
    payload = turn["payload"]
    return _respond(
        client,
        session_id,
        proposal_id=payload["proposal_id"],
        draft_hash=payload["draft_hash"],
        chosen=["review_wiring"],
    )


def _confirm_wiring(client: TestClient, session_id: str) -> dict:
    """Confirm the reviewed wire projection and commit its proposal."""
    current = _get_guided(client, session_id)
    turn = current["next_turn"]
    assert turn["type"] == "confirm_wiring"
    payload = turn["payload"]
    return _respond(
        client,
        session_id,
        proposal_id=payload["proposal_id"],
        draft_hash=payload["draft_hash"],
        chosen=["confirm_wiring"],
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

    @pytest.mark.parametrize("profile", ("live", "tutorial"))
    def test_rootless_step_3_entry_synthesizes_the_sketch_without_a_provider_call(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        profile: str,
    ) -> None:
        """The discarded starting sketch is server-synthesized, not planned.

        The step-2→3 auto-proposal on a rootless walk (no root intent, no
        deferred intents, one reviewed source and output) is always the same
        passthrough sketch, withheld from acceptance (supersedes_draft_hash
        null) and discarded by design once the transforms instruction lands.
        Tutorial final3 spent 222s of provider time producing it (op
        424021cd). It must now seal server-side through the same canonical
        final gate (prepare_pipeline_plan) with zero provider calls.
        """
        app = composer_test_client.app
        session_id = _create_session(composer_test_client)
        if profile == "tutorial":
            started = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/start",
                json={"profile": "tutorial", "operation_id": str(uuid4())},
            )
            assert started.status_code == 200, started.json()
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, f"sketch-{profile}.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])

        monkeypatch.setattr(
            ComposerServiceImpl,
            "_compute_availability",
            lambda _self: ComposerAvailability(
                available=True,
                provider="test",
                model="test/guided-planner",
                reason=None,
            ),
        )
        app.state.composer_service = ComposerServiceImpl(
            app.state.catalog_service,
            app.state.settings.model_copy(update={"composer_model": "test/guided-planner"}),
            sessions_service=app.state.session_service,
            session_engine=app.state.session_engine,
            secret_service=app.state.scoped_secret_resolver,
            plugin_snapshot_factory=lambda user_id: app.state.plugin_snapshot_factory(UserIdentity(user_id=user_id, username=user_id)),
            operator_profile_registry=app.state.operator_profile_registry,
        )

        async def poisoned_completion(**_kwargs: Any) -> _PlannerResponse:
            raise AssertionError("the rootless starting sketch must never call the provider")

        monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", poisoned_completion)

        settled = _post_current_response(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "output"},
        )

        assert settled.status_code == 200, settled.json()
        body = settled.json()
        assert body["next_turn"]["type"] == "propose_pipeline"
        with app.state.session_engine.connect() as conn:
            proposal = conn.execute(
                select(
                    composition_proposals_table.c.composer_model_identifier,
                    composition_proposals_table.c.composer_provider,
                ).where(composition_proposals_table.c.session_id == session_id)
            ).one()
        assert proposal.composer_model_identifier == "composer-guided-passthrough-synthesis"
        assert proposal.composer_provider == "server"
        audit_messages = asyncio.run(app.state.session_service.get_messages(UUID(session_id), limit=None))
        llm_audits = [
            envelope for message in audit_messages for envelope in (message.tool_calls or ()) if envelope.get("_kind") == "llm_call_audit"
        ]
        assert llm_audits == []

    @pytest.mark.parametrize(
        ("profile", "expected_surface"),
        (("live", "guided_staged"),),
    )
    @pytest.mark.parametrize(
        ("provider_outcome", "expected_status"),
        (
            ("success", ComposerLLMCallStatus.SUCCESS),
            ("error", ComposerLLMCallStatus.API_ERROR),
            ("cancel", ComposerLLMCallStatus.CANCELLED),
        ),
    )
    def test_actual_guided_planner_manifest_mismatch_is_durable_before_failure(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        profile: str,
        expected_surface: str,
        provider_outcome: str,
        expected_status: ComposerLLMCallStatus,
    ) -> None:
        import elspeth.web.composer.pipeline_planner as planner_module

        session_id = _create_session(composer_test_client)
        # A ROOT INTENT keeps the step-2→3 entry on the provider planner path:
        # a rootless walk now server-synthesizes the starting sketch without a
        # provider call (see test_rootless_step_3_entry_synthesizes_the_sketch),
        # which would make this provider-outcome matrix unreachable.
        started = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={
                "operation_id": str(uuid4()),
                "intent": "Build a pipeline that annotates each row before saving.",
            },
        )
        assert started.status_code == 200, started.json()
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, f"{profile}-{provider_outcome}.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])

        app = composer_test_client.app
        monkeypatch.setattr(
            ComposerServiceImpl,
            "_compute_availability",
            lambda _self: ComposerAvailability(
                available=True,
                provider="test",
                model="test/guided-planner",
                reason=None,
            ),
        )
        app.state.composer_service = ComposerServiceImpl(
            app.state.catalog_service,
            app.state.settings.model_copy(update={"composer_model": "test/guided-planner"}),
            sessions_service=app.state.session_service,
            session_engine=app.state.session_engine,
            secret_service=app.state.scoped_secret_resolver,
            plugin_snapshot_factory=lambda user_id: app.state.plugin_snapshot_factory(UserIdentity(user_id=user_id, username=user_id)),
            operator_profile_registry=app.state.operator_profile_registry,
        )
        requests: list[dict[str, object]] = []
        manifests: list[PlannerCapabilityManifest] = []
        real_builder = planner_module.build_planner_capability_manifest  # type: ignore[attr-defined]

        def capture_manifest(**kwargs: Any) -> PlannerCapabilityManifest:
            manifest = real_builder(**kwargs)
            manifests.append(manifest)
            return manifest

        async def mutating_completion(**kwargs: Any) -> _PlannerResponse:
            kwargs["messages"][0]["content"] += "\nprovider-side mutation"
            requests.append(kwargs)
            if provider_outcome == "error":
                raise RuntimeError("provider unavailable")
            if provider_outcome == "cancel":
                raise asyncio.CancelledError()
            return _planner_terminal_response()

        monkeypatch.setattr(planner_module, "build_planner_capability_manifest", capture_manifest)  # type: ignore[attr-defined]
        monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", mutating_completion)

        failed = _post_current_response(
            composer_test_client,
            session_id,
            component_action={"action": "finish", "component_kind": "output"},
        )
        assert failed.status_code == 500, failed.json()

        audit_messages = asyncio.run(app.state.session_service.get_messages(UUID(session_id), limit=None))
        planner_calls = [
            envelope["call"]
            for message in audit_messages
            for envelope in (message.tool_calls or ())
            if envelope.get("_kind") == "llm_call_audit" and envelope.get("call", {}).get("planner_call_ordinal") == 1
        ]
        with app.state.session_engine.connect() as conn:
            failed_operations = (
                conn.execute(
                    select(guided_operations_table.c.failure_code)
                    .where(guided_operations_table.c.session_id == session_id)
                    .where(guided_operations_table.c.status == "failed")
                )
                .scalars()
                .all()
            )
            proposals = conn.execute(
                select(composition_proposals_table.c.id).where(composition_proposals_table.c.session_id == session_id)
            ).all()

        assert len(requests) == len(manifests) == len(planner_calls) == 1
        manifest = manifests[0]
        assert manifest.surface.value == expected_surface
        assert manifest.profile == ("tutorial" if profile == "tutorial" else "ordinary")
        assert planner_calls[0]["status"] == expected_status.value
        assert planner_calls[0]["messages_hash"] != manifest.rendered_prompt_hash
        assert planner_calls[0]["tools_spec_hash"] == manifest.effective_tool_hash
        assert failed_operations == ["integrity_error"]
        assert proposals == []

    def test_actual_guided_planner_repeated_cancellation_waits_for_failure_settlement(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        # Root intent keeps the step-2→3 entry on the provider planner path
        # (rootless walks now server-synthesize the sketch without a call).
        started = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={
                "operation_id": str(uuid4()),
                "intent": "Build a pipeline that annotates each row before saving.",
            },
        )
        assert started.status_code == 200, started.json()
        self._drive_to_step_2_single_select(composer_test_client, session_id)
        _respond(composer_test_client, session_id, chosen=["json"])
        _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": _outputs_path(composer_test_client, "matching-cancel.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        current = _get_guided(composer_test_client, session_id)
        operation_id = str(uuid4())

        app = composer_test_client.app
        monkeypatch.setattr(
            ComposerServiceImpl,
            "_compute_availability",
            lambda _self: ComposerAvailability(
                available=True,
                provider="test",
                model="test/guided-planner",
                reason=None,
            ),
        )
        app.state.composer_service = ComposerServiceImpl(
            app.state.catalog_service,
            app.state.settings.model_copy(update={"composer_model": "test/guided-planner"}),
            sessions_service=app.state.session_service,
            session_engine=app.state.session_engine,
            secret_service=app.state.scoped_secret_resolver,
            plugin_snapshot_factory=lambda user_id: app.state.plugin_snapshot_factory(UserIdentity(user_id=user_id, username=user_id)),
            operator_profile_registry=app.state.operator_profile_registry,
        )

        async def cancelling_completion(**_kwargs: Any) -> _PlannerResponse:
            raise asyncio.CancelledError("provider cancelled matching planner request")

        monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", cancelling_completion)
        with app.state.session_engine.connect() as conn:
            state_ids_before = conn.execute(
                select(composition_states_table.c.id).where(composition_states_table.c.session_id == session_id)
            ).all()

        audit_inserted = threading.Event()
        release_audit_worker = threading.Event()
        original_insert = app.state.session_service._insert_prepared_guided_audit_rows_on_connection

        def pause_after_audit_insert(*args: Any, **kwargs: Any) -> Any:
            records = original_insert(*args, **kwargs)
            audit_inserted.set()
            if not release_audit_worker.wait(timeout=5.0):
                raise TimeoutError("test did not release guided cancellation audit worker")
            return records

        monkeypatch.setattr(
            app.state.session_service,
            "_insert_prepared_guided_audit_rows_on_connection",
            pause_after_audit_insert,
        )

        async def invoke_cancelled_route() -> None:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                request_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json={
                            "operation_id": operation_id,
                            "turn_token": current["next_turn"]["turn_token"],
                            "component_action": {"action": "finish", "component_kind": "output"},
                        },
                    )
                )
                assert await asyncio.to_thread(audit_inserted.wait, 5.0), "guided cancellation audit worker did not start"
                request_task.cancel("shutdown cancelled guided failure settlement")
                await asyncio.sleep(0)
                cancellation_escaped_before_settlement = request_task.done()
                release_audit_worker.set()
                with pytest.raises(
                    asyncio.CancelledError,
                    match="provider cancelled matching planner request",
                ) as caught:
                    await asyncio.wait_for(request_task, timeout=5.0)
                assert not cancellation_escaped_before_settlement
                assert caught.value.args == ("provider cancelled matching planner request",)
                assert caught.value.__cause__ is None

        asyncio.run(invoke_cancelled_route())

        audit_messages = asyncio.run(app.state.session_service.get_messages(UUID(session_id), limit=None))
        planner_calls = [
            envelope["call"]
            for message in audit_messages
            for envelope in (message.tool_calls or ())
            if envelope.get("_kind") == "llm_call_audit" and envelope.get("call", {}).get("planner_call_ordinal") == 1
        ]
        with app.state.session_engine.connect() as conn:
            operation = conn.execute(
                select(
                    guided_operations_table.c.status,
                    guided_operations_table.c.failure_code,
                    guided_operations_table.c.proposal_id,
                    guided_operations_table.c.result_state_id,
                )
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one()
            failed_events = conn.execute(
                select(guided_operation_events_table.c.event_kind)
                .where(guided_operation_events_table.c.session_id == session_id)
                .where(guided_operation_events_table.c.operation_id == operation_id)
                .where(guided_operation_events_table.c.event_kind == "failed")
            ).all()
            state_ids_after = conn.execute(
                select(composition_states_table.c.id).where(composition_states_table.c.session_id == session_id)
            ).all()
            proposals = conn.execute(
                select(composition_proposals_table.c.id).where(composition_proposals_table.c.session_id == session_id)
            ).all()

        assert len(planner_calls) == 1
        assert planner_calls[0]["status"] == ComposerLLMCallStatus.CANCELLED.value
        assert operation.status == "failed"
        assert operation.failure_code == "operation_failed"
        assert operation.proposal_id is None
        assert operation.result_state_id is None
        assert len(failed_events) == 1
        assert state_ids_after == state_ids_before
        assert proposals == []

    def test_external_cancellation_before_proposal_staging_is_request_cancelled(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
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
                    "path": _outputs_path(composer_test_client, "request-cancelled-before-stage.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        current = _get_guided(composer_test_client, session_id)
        operation_id = str(uuid4())
        planner_entered = asyncio.Event()

        async def blocked_planner(**_kwargs: Any):
            planner_entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(composer_test_client.app.state.composer_service, "plan_guided_pipeline", blocked_planner)
        state_before = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))

        async def cancel_request() -> asyncio.CancelledError:
            async with AsyncClient(transport=ASGITransport(app=composer_test_client.app), base_url="http://test") as client:
                request_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json={
                            "operation_id": operation_id,
                            "turn_token": current["next_turn"]["turn_token"],
                            "component_action": {"action": "finish", "component_kind": "output"},
                        },
                    )
                )
                await asyncio.wait_for(planner_entered.wait(), timeout=5)
                request_task.cancel("operator cancelled before proposal staging")
                with pytest.raises(asyncio.CancelledError, match="operator cancelled before proposal staging") as caught:
                    await asyncio.wait_for(request_task, timeout=5)
                return caught.value

        caught = asyncio.run(cancel_request())

        assert caught.args == ("operator cancelled before proposal staging",)
        state_after = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        assert state_before is not None and state_after is not None and state_after.id == state_before.id
        with composer_test_client.app.state.session_engine.connect() as conn:
            operation = conn.execute(
                select(guided_operations_table.c.status, guided_operations_table.c.failure_code)
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one()
            failed_events = conn.execute(
                select(guided_operation_events_table.c.event_kind)
                .where(guided_operation_events_table.c.session_id == session_id)
                .where(guided_operation_events_table.c.operation_id == operation_id)
                .where(guided_operation_events_table.c.event_kind == "failed")
            ).all()
            proposals = conn.execute(
                select(composition_proposals_table.c.id).where(composition_proposals_table.c.session_id == session_id)
            ).all()
        assert operation.status == "failed"
        assert operation.failure_code == "request_cancelled"
        assert len(failed_events) == 1
        assert proposals == []

    def test_cancellation_during_proposal_staging_drains_to_completed_replay(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
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
                    "path": _outputs_path(composer_test_client, "request-cancelled-during-stage.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
            },
        )
        _respond(composer_test_client, session_id, chosen=["text"], custom_inputs=[])
        current = _get_guided(composer_test_client, session_id)
        operation_id = str(uuid4())
        request_body = {
            "operation_id": operation_id,
            "turn_token": current["next_turn"]["turn_token"],
            "component_action": {"action": "finish", "component_kind": "output"},
        }
        service = composer_test_client.app.state.session_service
        original_insert = service._insert_prepared_guided_audit_rows_on_connection
        audit_inserted = threading.Event()
        release_worker = threading.Event()

        def pause_after_audit_insert(*args: Any, **kwargs: Any) -> Any:
            records = original_insert(*args, **kwargs)
            audit_inserted.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release guided proposal staging worker")
            return records

        monkeypatch.setattr(service, "_insert_prepared_guided_audit_rows_on_connection", pause_after_audit_insert)

        async def cancel_and_replay():
            async with AsyncClient(transport=ASGITransport(app=composer_test_client.app), base_url="http://test") as client:
                request_task = asyncio.create_task(client.post(f"/api/sessions/{session_id}/guided/respond", json=request_body))
                assert await asyncio.to_thread(audit_inserted.wait, 5), "guided proposal staging worker did not start"
                request_task.cancel("operator cancelled atomic proposal staging")
                await asyncio.sleep(0)
                assert not request_task.done()
                release_worker.set()
                with pytest.raises(asyncio.CancelledError, match="operator cancelled atomic proposal staging") as caught:
                    await asyncio.wait_for(request_task, timeout=5)
                replay = await asyncio.wait_for(
                    client.post(f"/api/sessions/{session_id}/guided/respond", json=request_body),
                    timeout=5,
                )
                return caught.value, replay

        caught, replay = asyncio.run(cancel_and_replay())

        assert caught.args == ("operator cancelled atomic proposal staging",)
        assert replay.status_code == 200, replay.json()
        assert replay.json()["next_turn"]["type"] == "propose_pipeline"
        with composer_test_client.app.state.session_engine.connect() as conn:
            operation = conn.execute(
                select(guided_operations_table.c.status, guided_operations_table.c.failure_code)
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one()
            proposals = conn.execute(
                select(composition_proposals_table.c.id).where(composition_proposals_table.c.session_id == session_id)
            ).all()
        assert operation.status == "completed"
        assert operation.failure_code is None
        assert len(proposals) == 1

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
        _finish_review(composer_test_client, session_id, "output")
        _review_wiring(composer_test_client, session_id)
        accepted = _confirm_wiring(composer_test_client, session_id)

        accepted_source = next(iter(accepted["composition_state"]["sources"].values()))
        assert accepted_source["options"]["path"] == str(source_path)

    @pytest.mark.parametrize("target_kind", ("source", "output"))
    def test_source_output_back_edit_atomically_supersedes_proposal_without_planning(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        target_kind: str,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="revise.jsonl")
        old_turn = staged["next_turn"]
        old_payload = old_turn["payload"]
        old_guided = _full_guided_session(staged)
        target = next(candidate for candidate in old_payload["edit_targets"] if candidate["kind"] == target_kind)
        reviewed_key = "reviewed_sources" if target_kind == "source" else "reviewed_outputs"
        reviewed_before = old_guided[reviewed_key][target["stable_id"]]
        planner_calls = 0

        async def forbidden_planner(**_kwargs):
            nonlocal planner_calls
            planner_calls += 1
            raise AssertionError("source/output proposal back-edit must not call the planner")

        monkeypatch.setattr(
            composer_test_client.app.state.composer_service,
            "plan_guided_pipeline",
            forbidden_planner,
        )
        operation_id = str(uuid4())
        request_payload = {
            "operation_id": operation_id,
            "turn_token": old_turn["turn_token"],
            "proposal_id": old_payload["proposal_id"],
            "draft_hash": old_payload["draft_hash"],
            "edit_target": target,
        }

        revised = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=request_payload,
        )

        assert revised.status_code == 200, revised.json()
        body = revised.json()
        assert planner_calls == 0
        edit_turn = body["next_turn"]
        assert edit_turn["type"] == "schema_form"
        assert edit_turn["step_index"] == (0 if target_kind == "source" else 1)
        guided = _full_guided_session(body)
        assert guided["reviewed_sources"] == old_guided["reviewed_sources"]
        assert guided["reviewed_outputs"] == old_guided["reviewed_outputs"]
        assert guided["deferred_intents"] == old_guided["deferred_intents"]
        assert guided["active_proposal"] is None
        assert guided["active_edit_target"] == target
        assert guided["step"] == ("step_1_source" if target_kind == "source" else "step_2_sink")
        assert guided[reviewed_key][target["stable_id"]] == reviewed_before

        reentered = _get_guided(composer_test_client, session_id)
        assert reentered["next_turn"] == edit_turn
        replayed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=request_payload,
        )
        assert replayed.status_code == 200, replayed.json()
        assert replayed.json() == body

        proposals = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert proposals[old_payload["proposal_id"]].status == "rejected"
        assert set(proposals) == {old_payload["proposal_id"]}
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        old_events = [event for event in events if str(event.proposal_id) == old_payload["proposal_id"]]
        assert [event.event_type for event in old_events] == ["proposal.created", "proposal.rejected"]
        assert old_events[-1].payload["reason_code"] == "superseded"

        old_review = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": old_turn["turn_token"],
                "proposal_id": old_payload["proposal_id"],
                "draft_hash": old_payload["draft_hash"],
                "chosen": ["review_wiring"],
            },
        )
        assert old_review.status_code == 409, old_review.json()
        proposals_after = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert proposals_after[old_payload["proposal_id"]].status == "rejected"

        continued = _respond(
            composer_test_client,
            session_id,
            edited_values={
                "plugin": edit_turn["payload"]["plugin"],
                "options": {key: value for key, value in edit_turn["payload"]["prefilled"].items() if value is not None},
            },
        )
        expected_review_type = "inspect_and_confirm" if target_kind == "source" else "multi_select_with_custom"
        assert continued["next_turn"]["type"] == expected_review_type
        assert _get_guided(composer_test_client, session_id)["next_turn"] == continued["next_turn"]
        if target_kind == "source":
            reviewed = _respond(
                composer_test_client,
                session_id,
                edited_values={"columns": reviewed_before["observed_columns"]},
            )
            policy_key = "on_validation_failure"
        else:
            reviewed = _respond(
                composer_test_client,
                session_id,
                chosen=reviewed_before["required_fields"],
                custom_inputs=[],
            )
            policy_key = "on_write_failure"
        assert reviewed["next_turn"]["type"] == "review_components"
        reviewed_after = _full_guided_session(reviewed)[reviewed_key][target["stable_id"]]
        assert reviewed_after["name"] == reviewed_before["name"]
        assert reviewed_after["plugin"] == reviewed_before["plugin"]
        assert reviewed_after[policy_key] == reviewed_before[policy_key]

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

    def test_prose_revision_replans_full_pipeline_with_instruction_as_intent(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A docked-composer instruction re-plans the whole pipeline.

        The proposal-review turn accepts a prose ``revision_instruction`` (no
        ``edit_target``). It supersedes the pending proposal and re-plans the
        full pipeline with the instruction as the planner intent. With no root
        intent (the tutorial / auto-proposal case), the planner originating
        content is the instruction alone.
        """
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="prose-revise.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]
        old_proposal_id = payload["proposal_id"]
        assert _full_guided_session(staged)["deferred_intents"] == []

        captured: dict[str, object] = {}
        original_planner = composer_test_client.app.state.composer_service.plan_guided_pipeline

        async def spy_planner(**kwargs: object) -> object:
            captured.update(kwargs)
            return await original_planner(**kwargs)

        monkeypatch.setattr(composer_test_client.app.state.composer_service, "plan_guided_pipeline", spy_planner)

        instruction = "Add a deduplication transform before the output."
        operation_id = str(uuid4())
        request_payload = {
            "operation_id": operation_id,
            "turn_token": turn["turn_token"],
            "proposal_id": payload["proposal_id"],
            "draft_hash": payload["draft_hash"],
            "edited_values": {"revision_instruction": instruction},
        }

        revised = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=request_payload,
        )

        assert revised.status_code == 200, revised.json()
        body = revised.json()
        assert body["next_turn"]["type"] == "propose_pipeline"
        successor_id = body["next_turn"]["payload"]["proposal_id"]
        assert successor_id != old_proposal_id
        # The instruction is the planner intent verbatim, and with no root intent
        # the originating content is exactly the instruction (root-absent branch).
        assert captured["intent"] == instruction
        assert captured["originating_message"].content == instruction

        guided = _full_guided_session(body)
        assert guided["deferred_intents"] == []
        assert guided["active_proposal"]["proposal_id"] == successor_id

        proposals = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert proposals[old_proposal_id].status == "rejected"
        assert proposals[successor_id].status == "pending"

        # The operation is idempotent: replaying the same request returns the
        # settled body without a second planner call.
        captured.clear()
        replay = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=request_payload,
        )
        assert replay.status_code == 200, replay.json()
        assert replay.json() == body
        assert captured == {}

    def test_competing_respond_answers_fast_coded_conflict_during_planner_settlement(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A different-body respond never queues behind an in-flight planner run.

        post_guided_respond holds the session admission lock across the whole
        settlement, including the in-request planner (200s+ observed live).
        A genuinely concurrent competing respond (different operation_id —
        double-click race, second tab) would silently queue for minutes and
        then die stale. It must instead answer FAST with the coded conflict
        envelope. Same-body retries are safe by construction: they join or
        replay via the pre-admission replay lookup and never wait on the
        admission lock, so the bounded wait cannot corrupt replay semantics.
        """
        from elspeth.web.composer.pipeline_planner import PipelinePlannerError
        from elspeth.web.sessions.routes import guided_operations as guided_ops_module

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="admission-conflict.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]

        planner_entered = asyncio.Event()
        planner_release = asyncio.Event()

        async def slow_planner(**_kwargs: object) -> object:
            planner_entered.set()
            await planner_release.wait()
            raise PipelinePlannerError(
                "planner released for teardown",
                code="REPAIR_EXHAUSTED",
                detail_codes=(),
            )

        monkeypatch.setattr(composer_test_client.app.state.composer_service, "plan_guided_pipeline", slow_planner)
        monkeypatch.setattr(guided_ops_module, "GUIDED_RESPOND_ADMISSION_WAIT_SECONDS", 0.2, raising=False)

        def _revision_body(instruction: str) -> dict[str, object]:
            return {
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": payload["proposal_id"],
                "draft_hash": payload["draft_hash"],
                "edited_values": {"revision_instruction": instruction},
            }

        async def run() -> tuple[object, float, object]:
            async with AsyncClient(transport=ASGITransport(app=composer_test_client.app), base_url="http://test") as client:
                owner_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=_revision_body("First revision: add a summarizing transform."),
                    )
                )
                await asyncio.wait_for(planner_entered.wait(), timeout=5)

                loop = asyncio.get_running_loop()
                started = loop.time()
                competitor_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=_revision_body("Second revision: rename the output file."),
                    )
                )
                try:
                    competitor = await asyncio.wait_for(competitor_task, timeout=5)
                except TimeoutError:
                    competitor_task.cancel()
                    competitor = None
                elapsed = loop.time() - started

                planner_release.set()
                owner = await asyncio.wait_for(owner_task, timeout=15)
                return competitor, elapsed, owner

        competitor, elapsed, owner = asyncio.run(run())

        assert competitor is not None, "competing respond queued behind the in-flight planner instead of answering"
        assert competitor.status_code == 409, competitor.text
        detail = competitor.json()["detail"]
        assert detail["error_type"] == "guided_operation_conflict"
        assert detail["code"] == "operation_in_progress"
        assert elapsed < 3, f"competing respond took {elapsed:.1f}s; must answer well under the planner runtime"
        # The in-flight owner is unaffected: it settles through its own path
        # (coded planner-failure envelope from the stubbed exhaustion).
        assert owner.status_code == 502, owner.text

    def test_prose_revision_planner_exhaustion_is_coded_502_not_500(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Planner exhaustion answers the closed coded envelope, not a raw 500.

        Tutorial op 18b4cee7 (session c98e8561, 2026-07-22): REPAIR_EXHAUSTED
        on the step-3 replan fell through post_guided_respond's failure-code
        selection to 'operation_failed' — the generic 500 banner — while the
        sibling /guided/plan route already maps PipelinePlannerError through
        _guided_full_failure_code to 'invalid_provider_response' (502, with a
        retry instruction). The respond route must answer the same closed
        shape for the same failure class.
        """
        from elspeth.web.composer.pipeline_planner import PipelinePlannerError

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="prose-exhaust.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]

        async def exhausted_planner(**kwargs: object) -> object:
            raise PipelinePlannerError(
                "planner repair budget exhausted",
                code="REPAIR_EXHAUSTED",
                detail_codes=("interpretation_review_contract_unsatisfied",),
            )

        monkeypatch.setattr(
            composer_test_client.app.state.composer_service,
            "plan_guided_pipeline",
            exhausted_planner,
        )

        response = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": payload["proposal_id"],
                "draft_hash": payload["draft_hash"],
                "edited_values": {"revision_instruction": "Summarize each row before saving."},
            },
        )

        assert response.status_code == 502, response.text
        detail = response.json()["detail"]
        assert detail["error_type"] == "guided_operation_terminal_failure"
        assert detail["failure_code"] == "invalid_provider_response"
        assert "retry" in detail["detail"].lower()  # actionable, no planner internals

    def test_prose_revision_appends_instruction_to_root_intent(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a live root intent, the instruction is appended after it.

        The planner originating content is the root intent first, then the
        instruction on a new paragraph (root-present branch).
        """
        session_id = _create_session(composer_test_client)
        intent = "Author a pipeline that ingests the CSV and writes JSON results."
        started = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={"profile": "live", "intent": intent, "operation_id": str(uuid4())},
        )
        assert started.status_code == 200, started.json()
        staged = self._stage_proposal(composer_test_client, session_id, filename="prose-root.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]

        captured: dict[str, object] = {}
        original_planner = composer_test_client.app.state.composer_service.plan_guided_pipeline

        async def spy_planner(**kwargs: object) -> object:
            captured.update(kwargs)
            return await original_planner(**kwargs)

        monkeypatch.setattr(composer_test_client.app.state.composer_service, "plan_guided_pipeline", spy_planner)

        instruction = "Insert a deduplication transform between the source and the sink."
        revised = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": payload["proposal_id"],
                "draft_hash": payload["draft_hash"],
                "edited_values": {"revision_instruction": instruction},
            },
        )

        assert revised.status_code == 200, revised.json()
        assert revised.json()["next_turn"]["type"] == "propose_pipeline"
        assert captured["intent"] == instruction
        assert captured["originating_message"].content == f"{intent}\n\n{instruction}"

    @pytest.mark.parametrize("instruction", ["", "   ", "x" * 8193, 123])
    def test_prose_revision_rejects_invalid_instruction_without_mutation(
        self,
        composer_test_client: TestClient,
        instruction: object,
    ) -> None:
        """Blank, oversized, or non-string instructions are a stable 400."""
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="prose-invalid.jsonl")
        turn = staged["next_turn"]
        payload = turn["payload"]

        rejected = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": payload["proposal_id"],
                "draft_hash": payload["draft_hash"],
                "edited_values": {"revision_instruction": instruction},
            },
        )

        assert rejected.status_code == 400, rejected.json()
        assert rejected.json()["detail"] == ("Guided proposal revision instruction must be a non-empty string of at most 8192 characters.")
        current = _get_guided(composer_test_client, session_id)
        assert current["next_turn"]["payload"] == payload
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1
        assert proposals[0].status == "pending"

    def test_component_back_edit_rejects_stale_target_without_mutation(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="stale-target.jsonl")
        turn = staged["next_turn"]
        proposal = turn["payload"]
        operation_id = str(uuid4())
        target = next(candidate for candidate in proposal["edit_targets"] if candidate["kind"] == "source")

        rejected = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": operation_id,
                "turn_token": turn["turn_token"],
                "proposal_id": proposal["proposal_id"],
                "draft_hash": proposal["draft_hash"],
                "edit_target": {**target, "stable_id": str(uuid4())},
            },
        )

        assert rejected.status_code == 409, rejected.json()
        assert _get_guided(composer_test_client, session_id)["next_turn"] == turn
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"
        with composer_test_client.app.state.session_engine.connect() as conn:
            operation = conn.execute(
                select(guided_operations_table.c.operation_id)
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one_or_none()
        assert operation is None

    def test_component_back_edit_rejects_cross_session_proposal_binding_without_mutation(
        self,
        composer_test_client: TestClient,
    ) -> None:
        session_a = _create_session(composer_test_client)
        session_b = _create_session(composer_test_client)
        staged_a = self._stage_proposal(composer_test_client, session_a, filename="cross-a.jsonl")
        staged_b = self._stage_proposal(composer_test_client, session_b, filename="cross-b.jsonl")
        turn_a = staged_a["next_turn"]
        proposal_a = turn_a["payload"]
        proposal_b = staged_b["next_turn"]["payload"]
        target_a = next(candidate for candidate in proposal_a["edit_targets"] if candidate["kind"] == "source")

        rejected = composer_test_client.post(
            f"/api/sessions/{session_a}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn_a["turn_token"],
                "proposal_id": proposal_b["proposal_id"],
                "draft_hash": proposal_b["draft_hash"],
                "edit_target": target_a,
            },
        )

        assert rejected.status_code == 409, rejected.json()
        assert _get_guided(composer_test_client, session_a)["next_turn"] == turn_a
        for session_id in (session_a, session_b):
            proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
            assert len(proposals) == 1 and proposals[0].status == "pending"

    def test_component_back_edit_rejects_proposal_base_bound_to_older_head_atomically(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        session_uuid = UUID(session_id)
        staged = self._stage_proposal(composer_test_client, session_id, filename="stale-base.jsonl")
        turn = staged["next_turn"]
        proposal = turn["payload"]
        target = next(candidate for candidate in proposal["edit_targets"] if candidate["kind"] == "source")
        service = composer_test_client.app.state.session_service
        original_back_edit = service.back_edit_guided_pipeline_proposal
        captured_commands = []

        async def capture_without_settlement(command, *, payload_store=None):
            del payload_store
            captured_commands.append(command)
            raise RuntimeError("capture command before settlement")

        monkeypatch.setattr(service, "back_edit_guided_pipeline_proposal", capture_without_settlement)
        captured = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": proposal["proposal_id"],
                "draft_hash": proposal["draft_hash"],
                "edit_target": target,
            },
        )
        assert captured.status_code == 500, captured.json()
        assert len(captured_commands) == 1
        command = captured_commands[0]

        older_head = asyncio.run(service.get_current_state(session_uuid))
        assert older_head is not None
        later_head = asyncio.run(
            service.save_composition_state(
                session_uuid,
                CompositionStateData(
                    sources=older_head.sources,
                    nodes=older_head.nodes,
                    edges=older_head.edges,
                    outputs=older_head.outputs,
                    metadata_=older_head.metadata_,
                    is_valid=older_head.is_valid,
                    validation_errors=older_head.validation_errors,
                    composer_meta=older_head.composer_meta,
                ),
                provenance="convergence_persist",
            )
        )
        later_state = state_from_record(later_head)
        assert later_state.guided_session is not None
        assert later_state.guided_session.active_proposal is not None
        assert later_state.guided_session.active_proposal.base.state_id == older_head.id
        assert later_state.guided_session.active_proposal.base.state_id != later_head.id

        operation_id = str(uuid4())
        claimed = asyncio.run(
            service.reserve_guided_operation(
                session_id=session_uuid,
                operation_id=operation_id,
                kind="guided_respond",
                request_hash="f" * 64,
                actor="test_route",
                lease_seconds=300,
            )
        )
        assert isinstance(claimed, GuidedOperationClaimed)
        stale_base_command = replace(
            command,
            fence=claimed.fence,
            expected_current_state_id=later_head.id,
            expected_current_state_version=later_head.version,
            expected_current_content_hash=composition_content_hash(later_state),
        )

        engine = composer_test_client.app.state.session_engine
        with engine.connect() as conn:
            state_count_before = conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            ).scalar_one()
            message_count_before = conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == session_id)
            ).scalar_one()
            proposal_before = conn.execute(
                select(composition_proposals_table.c.status, composition_proposals_table.c.audit_event_id).where(
                    composition_proposals_table.c.id == proposal["proposal_id"]
                )
            ).one()
            proposal_events_before = (
                conn.execute(
                    select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal["proposal_id"])
                )
                .scalars()
                .all()
            )

        with pytest.raises(AuditIntegrityError, match="base"):
            asyncio.run(
                original_back_edit(
                    stale_base_command,
                    payload_store=composer_test_client.app.state.payload_store,
                )
            )

        with engine.connect() as conn:
            assert (
                conn.execute(
                    select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
                ).scalar_one()
                == state_count_before
            )
            assert (
                conn.execute(
                    select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == session_id)
                ).scalar_one()
                == message_count_before
            )
            proposal_row = conn.execute(
                select(composition_proposals_table.c.status, composition_proposals_table.c.audit_event_id).where(
                    composition_proposals_table.c.id == proposal["proposal_id"]
                )
            ).one()
            proposal_events_after = (
                conn.execute(
                    select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal["proposal_id"])
                )
                .scalars()
                .all()
            )
            operation = conn.execute(
                select(
                    guided_operations_table.c.status,
                    guided_operations_table.c.proposal_id,
                    guided_operations_table.c.result_kind,
                    guided_operations_table.c.result_state_id,
                    guided_operations_table.c.response_hash,
                )
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one()
            messages = (
                conn.execute(select(chat_messages_table.c.tool_calls).where(chat_messages_table.c.session_id == session_id)).scalars().all()
            )
            operation_events = (
                conn.execute(
                    select(guided_operation_events_table.c.event_kind)
                    .where(guided_operation_events_table.c.session_id == session_id)
                    .where(guided_operation_events_table.c.operation_id == operation_id)
                    .order_by(guided_operation_events_table.c.sequence)
                )
                .scalars()
                .all()
            )
        assert proposal_row == proposal_before
        assert proposal_row.status == "pending"
        assert proposal_events_before == ["proposal.created"]
        assert proposal_events_after == proposal_events_before
        assert operation.status == "in_progress"
        assert operation.proposal_id is None
        assert operation.result_kind is None
        assert operation.result_state_id is None
        assert operation.response_hash is None
        assert operation_events == ["claimed"]
        assert all(payload.payload_id not in repr(messages) for payload in stale_base_command.payloads)

    @pytest.mark.parametrize(
        "fault_point",
        (
            "state_insert",
            "proposal_event",
            "proposal_update",
            "audit_insert",
            "operation_bind",
            "operation_complete",
            "operation_event",
        ),
    )
    def test_component_back_edit_fault_rolls_back_every_settlement_surface(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        fault_point: str,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename=f"back-edit-{fault_point}.jsonl")
        turn = staged["next_turn"]
        proposal_payload = turn["payload"]
        target = next(candidate for candidate in proposal_payload["edit_targets"] if candidate["kind"] == "source")
        operation_id = str(uuid4())
        engine = composer_test_client.app.state.session_engine
        service = composer_test_client.app.state.session_service
        original = service.back_edit_guided_pipeline_proposal
        captured_commands = []

        async def capture(command, *, payload_store=None):
            captured_commands.append(command)
            return await original(command, payload_store=payload_store)

        monkeypatch.setattr(service, "back_edit_guided_pipeline_proposal", capture)
        with engine.connect() as conn:
            state_count_before = conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            ).scalar_one()

        armed = True
        observed_writes: set[str] = set()
        operation_event_predecessors: set[str] | None = None

        def inject_fault(_conn, _cursor, _statement, _parameters, context, _executemany):
            nonlocal armed, operation_event_predecessors
            if not armed:
                return
            compiled = context.compiled
            statement = compiled.statement if compiled is not None else None
            table_name = getattr(getattr(statement, "table", None), "name", None)
            value_keys = set(compiled.params) if compiled is not None else set()
            operation: str | None = None
            if isinstance(statement, Insert):
                operation = {
                    "composition_states": "state_insert",
                    "proposal_events": "proposal_event",
                    "chat_messages": "audit_insert",
                    "guided_operation_events": "operation_event",
                }.get(table_name)
            elif isinstance(statement, Update):
                if table_name == "composition_proposals":
                    operation = "proposal_update"
                elif table_name == "guided_operations" and "originating_message_id" in value_keys:
                    operation = "operation_bind"
                elif table_name == "guided_operations" and "status" in value_keys:
                    operation = "operation_complete"
            if operation is not None:
                observed_writes.add(operation)
            matched = fault_point == operation and (operation != "operation_event" or "operation_complete" in observed_writes)
            if matched:
                if fault_point == "operation_event":
                    operation_event_predecessors = set(observed_writes)
                armed = False
                raise RuntimeError(f"injected {fault_point}")

        event.listen(engine, "before_cursor_execute", inject_fault)
        try:
            failed = composer_test_client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={
                    "operation_id": operation_id,
                    "turn_token": turn["turn_token"],
                    "proposal_id": proposal_payload["proposal_id"],
                    "draft_hash": proposal_payload["draft_hash"],
                    "edit_target": target,
                },
            )
        finally:
            event.remove(engine, "before_cursor_execute", inject_fault)

        assert failed.status_code == 500, failed.json()
        assert not armed, f"fault point {fault_point} was not reached"
        if fault_point == "operation_event":
            assert operation_event_predecessors is not None
            assert {
                "state_insert",
                "proposal_event",
                "proposal_update",
                "audit_insert",
                "operation_bind",
                "operation_complete",
            } <= operation_event_predecessors
        assert len(captured_commands) == 1
        prepared_payload_ids = {payload.payload_id for payload in captured_commands[0].payloads}
        assert _get_guided(composer_test_client, session_id)["next_turn"] == turn
        with engine.connect() as conn:
            assert (
                conn.execute(
                    select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
                ).scalar_one()
                == state_count_before
            )
            proposal = conn.execute(
                select(composition_proposals_table.c.status, composition_proposals_table.c.audit_event_id).where(
                    composition_proposals_table.c.id == proposal_payload["proposal_id"]
                )
            ).one()
            events = conn.execute(
                select(proposal_events_table.c.event_type).where(proposal_events_table.c.proposal_id == proposal_payload["proposal_id"])
            ).all()
            operation = conn.execute(
                select(
                    guided_operations_table.c.status,
                    guided_operations_table.c.proposal_id,
                    guided_operations_table.c.result_kind,
                    guided_operations_table.c.result_state_id,
                    guided_operations_table.c.response_hash,
                    guided_operations_table.c.failure_code,
                )
                .where(guided_operations_table.c.session_id == session_id)
                .where(guided_operations_table.c.operation_id == operation_id)
            ).one()
            messages = (
                conn.execute(select(chat_messages_table.c.tool_calls).where(chat_messages_table.c.session_id == session_id)).scalars().all()
            )
            operation_events = (
                conn.execute(
                    select(guided_operation_events_table.c.event_kind)
                    .where(guided_operation_events_table.c.session_id == session_id)
                    .where(guided_operation_events_table.c.operation_id == operation_id)
                    .order_by(guided_operation_events_table.c.sequence)
                )
                .scalars()
                .all()
            )
        assert proposal.status == "pending"
        assert [row.event_type for row in events] == ["proposal.created"]
        assert operation.status == "failed"
        assert operation.failure_code == "operation_failed"
        assert operation.proposal_id is None
        assert operation.result_kind is None
        assert operation.result_state_id is None
        assert operation.response_hash is None
        assert operation_events == ["claimed", "renewed", "failed"]
        assert all(payload_id not in repr(messages) for payload_id in prepared_payload_ids)

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
        assert revised.json()["next_turn"]["type"] == "schema_form"
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
            rejected_stage,
            rejected.json(),
            rejected_restored,
            *(event.payload for event in (*events, *rejected_events)),
        )
        rendered = repr(public_surfaces)
        assert all(canary not in rendered for canary in canaries)

    @pytest.mark.parametrize(
        "composer_test_client",
        (pytest.param("postgres", id="postgres", marks=pytest.mark.testcontainer),),
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
    def test_review_vs_revise_race_has_one_winner_and_exact_revision_replay(
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
        back_edit_calls = 0
        original_back_edit = composer_test_client.app.state.session_service.back_edit_guided_pipeline_proposal
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
            "chosen": ["review_wiring"],
        }

        async def race():
            nonlocal back_edit_calls
            back_edit_entered = asyncio.Event()
            release_back_edit = asyncio.Event()

            async def forbidden_planner(**_kwargs):
                nonlocal planner_calls
                planner_calls += 1
                raise AssertionError("source/output proposal back-edit must not call the planner")

            async def blocking_back_edit(command, *, payload_store=None):
                nonlocal back_edit_calls
                back_edit_calls += 1
                back_edit_entered.set()
                await release_back_edit.wait()
                return await original_back_edit(command, payload_store=payload_store)

            monkeypatch.setattr(
                composer_test_client.app.state.composer_service,
                "plan_guided_pipeline",
                forbidden_planner,
            )
            monkeypatch.setattr(
                composer_test_client.app.state.session_service,
                "back_edit_guided_pipeline_proposal",
                blocking_back_edit,
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
                await asyncio.wait_for(back_edit_entered.wait(), timeout=5)
                # The atomic back-edit settlement is invoked while guided
                # admission and the per-session compose lock are held. Accept
                # queues behind it and then fails stale preflight.
                during_back_edit = await composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id))
                accept_task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=accept_request,
                    )
                )
                await asyncio.sleep(0)
                release_back_edit.set()
                revised, accepted = await asyncio.wait_for(
                    asyncio.gather(revision_task, accept_task),
                    timeout=10,
                )
                replayed = await client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json=revision_request,
                )
                return during_back_edit, revised, accepted, replayed

        during_back_edit, revised, accepted, replayed = asyncio.run(race())
        assert len(during_back_edit) == 1
        assert during_back_edit[0].status == "pending"

        assert revised.status_code == 200, revised.json()
        assert accepted.status_code == 409, accepted.json()
        winner_body = revised.json()
        assert winner_body["next_turn"]["type"] == "schema_form"
        proposals = {
            str(proposal.id): proposal
            for proposal in asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        }
        assert set(proposals) == {payload["proposal_id"]}
        assert proposals[payload["proposal_id"]].status == "rejected"
        events_before_replay = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        assert [event.event_type for event in events_before_replay if str(event.proposal_id) == payload["proposal_id"]] == [
            "proposal.created",
            "proposal.rejected",
        ]
        assert replayed.status_code == 200, replayed.json()
        assert replayed.json() == winner_body
        assert planner_calls == 0
        assert back_edit_calls == 1
        events_after_replay = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        assert events_after_replay == events_before_replay

    @pytest.mark.parametrize(
        "composer_test_client",
        (
            pytest.param("sqlite", id="sqlite"),
            pytest.param("postgres", id="postgres", marks=pytest.mark.testcontainer),
        ),
        indirect=True,
    )
    @pytest.mark.parametrize("db_winner", ("confirm", "correct"))
    def test_independent_workers_serialize_confirm_vs_correction_at_admission(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        db_winner: str,
    ) -> None:
        from elspeth.web.composer import pipeline_commit

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename=f"admission-{db_winner}.jsonl")
        proposal_payload = staged["next_turn"]["payload"]
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
        accept_operation_id = str(uuid4())
        revise_operation_id = str(uuid4())
        requests = {
            "confirm": {
                "operation_id": accept_operation_id,
                "turn_token": turn["turn_token"],
                "proposal_id": proposal_payload["proposal_id"],
                "draft_hash": proposal_payload["draft_hash"],
                "chosen": ["confirm_wiring"],
            },
            "correct": {
                "operation_id": revise_operation_id,
                "turn_token": turn["turn_token"],
                "proposal_id": proposal_payload["proposal_id"],
                "draft_hash": proposal_payload["draft_hash"],
                "edit_target": turn["payload"]["connections"][0]["from_endpoint"],
                "correction_feedback": "Route this source through a corrected topology.",
            },
        }

        primary_app = composer_test_client.app
        peer_app = _independent_guided_peer_app(composer_test_client)
        accept_service = primary_app.state.session_service
        revise_service = peer_app.state.session_service
        assert accept_service is not revise_service
        assert primary_app.state.session_engine is peer_app.state.session_engine
        assert primary_app.state.session_compose_lock_registry is not peer_app.state.session_compose_lock_registry

        original_admit = accept_service.admit_guided_pipeline_confirmation
        original_stage = revise_service.stage_guided_pipeline_proposal
        original_prepare = pipeline_commit.prepare_pipeline_proposal_commit
        original_execute = pipeline_commit.execute_tool
        entered = {"confirm": asyncio.Event(), "correct": asyncio.Event()}
        release = asyncio.Event()
        winner_settled = asyncio.Event()
        admission_commands = []
        stage_commands = []
        prepare_calls = []
        execute_calls = []

        async def gated_admit(command):
            admission_commands.append(command)
            entered["confirm"].set()
            await release.wait()
            if db_winner != "confirm":
                await winner_settled.wait()
            try:
                return await original_admit(command)
            finally:
                if db_winner == "confirm":
                    winner_settled.set()

        async def gated_stage(command, *, payload_store=None):
            stage_commands.append(command)
            entered["correct"].set()
            await release.wait()
            if db_winner != "correct":
                await winner_settled.wait()
            try:
                return await original_stage(command, payload_store=payload_store)
            finally:
                if db_winner == "correct":
                    winner_settled.set()

        async def counted_prepare(**kwargs):
            prepare_calls.append(kwargs)
            return await original_prepare(**kwargs)

        def counted_execute(*args, **kwargs):
            execute_calls.append((args, kwargs))
            return original_execute(*args, **kwargs)

        monkeypatch.setattr(accept_service, "admit_guided_pipeline_confirmation", gated_admit)
        monkeypatch.setattr(revise_service, "stage_guided_pipeline_proposal", gated_stage)
        monkeypatch.setattr(pipeline_commit, "prepare_pipeline_proposal_commit", counted_prepare)
        monkeypatch.setattr(pipeline_commit, "execute_tool", counted_execute)

        engine = primary_app.state.session_engine
        with engine.connect() as conn:
            state_count_before = conn.scalar(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            )

        async def race_and_replay():
            async with (
                AsyncClient(transport=ASGITransport(app=primary_app), base_url="http://confirm-worker") as accept_client,
                AsyncClient(transport=ASGITransport(app=peer_app), base_url="http://correct-worker") as revise_client,
            ):
                tasks = {
                    "confirm": asyncio.create_task(
                        accept_client.post(f"/api/sessions/{session_id}/guided/respond", json=requests["confirm"])
                    ),
                    "correct": asyncio.create_task(
                        revise_client.post(f"/api/sessions/{session_id}/guided/respond", json=requests["correct"])
                    ),
                }
                await asyncio.wait_for(
                    asyncio.gather(entered["confirm"].wait(), entered["correct"].wait()),
                    timeout=10,
                )
                release.set()
                responses = dict(
                    zip(
                        ("confirm", "correct"),
                        await asyncio.wait_for(asyncio.gather(tasks["confirm"], tasks["correct"]), timeout=20),
                        strict=True,
                    )
                )
                replays = {
                    "confirm": await revise_client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=requests["confirm"],
                    ),
                    "correct": await accept_client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=requests["correct"],
                    ),
                }
                return responses, replays

        responses, replays = asyncio.run(race_and_replay())
        loser = "correct" if db_winner == "confirm" else "confirm"
        assert responses[db_winner].status_code == 200, responses[db_winner].json()
        assert responses[loser].status_code == 409, responses[loser].json()
        assert responses[loser].json()["detail"]["failure_code"] == "stale_conflict"
        for action in ("confirm", "correct"):
            assert replays[action].status_code == responses[action].status_code
            assert replays[action].json() == responses[action].json()
        assert len(admission_commands) == len(stage_commands) == 1

        service = primary_app.state.session_service
        proposals = {str(item.id): item for item in asyncio.run(service.list_composition_proposals(UUID(session_id)))}
        events = asyncio.run(service.list_proposal_events(UUID(session_id)))
        messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        dispatches = [
            envelope
            for message in messages
            for envelope in (message.tool_calls or ())
            if envelope.get("invocation", {}).get("tool_name") == "set_pipeline"
            and envelope.get("invocation", {}).get("status") == "success"
        ]
        with engine.connect() as conn:
            operations = {
                row["operation_id"]: row
                for row in conn.execute(
                    select(guided_operations_table)
                    .where(guided_operations_table.c.session_id == session_id)
                    .where(guided_operations_table.c.operation_id.in_((accept_operation_id, revise_operation_id)))
                ).mappings()
            }
            state_count_after = conn.scalar(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == session_id)
            )
        assert len(operations) == 2
        winner_operation_id = accept_operation_id if db_winner == "confirm" else revise_operation_id
        loser_operation_id = revise_operation_id if db_winner == "confirm" else accept_operation_id
        assert operations[winner_operation_id]["status"] == "completed"
        assert operations[loser_operation_id]["status"] == "failed"
        assert operations[loser_operation_id]["failure_code"] == "stale_conflict"
        assert state_count_before is not None and state_count_after == state_count_before + 1

        original_id = proposal_payload["proposal_id"]
        original_events = [event.event_type for event in events if str(event.proposal_id) == original_id]
        current = _get_guided(composer_test_client, session_id)
        if db_winner == "confirm":
            assert len(prepare_calls) == len(execute_calls) == 1
            assert set(proposals) == {original_id}
            assert proposals[original_id].status == "committed"
            assert original_events == ["proposal.created", "proposal.accepted"]
            assert len(dispatches) == 1
            assert current["next_turn"] is None
        else:
            assert prepare_calls == execute_calls == []
            assert len(proposals) == 2
            assert proposals[original_id].status == "rejected"
            assert proposals[original_id].committed_state_id is None
            assert original_events == ["proposal.created", "proposal.rejected"]
            assert all(event.event_type != "proposal.accepted" for event in events)
            assert dispatches == []
            successor = next(item for proposal_id, item in proposals.items() if proposal_id != original_id)
            assert successor.status == "pending"
            assert current["next_turn"]["type"] == "confirm_wiring"

    def test_confirm_wiring_atomically_commits_pipeline_consumes_coverage_and_completes_guided_authoring(
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
        assert staged["next_turn"]["type"] == "propose_pipeline"
        with composer_test_client.app.state.session_engine.connect() as conn:
            storage_paths = tuple(conn.execute(select(blobs_table.c.storage_path).where(blobs_table.c.session_id == session_id)).scalars())
        assert storage_paths
        proposal = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))[0]
        assert "blob:" in repr(proposal.arguments_json)
        for storage_path in storage_paths:
            assert storage_path not in json.dumps(staged)
            assert storage_path not in repr(proposal.arguments_json)

        reviewed = _review_wiring(composer_test_client, session_id)
        wire_turn = reviewed["next_turn"]
        operation_id = str(uuid4())
        confirm_request = {
            "operation_id": operation_id,
            "turn_token": wire_turn["turn_token"],
            "proposal_id": wire_turn["payload"]["proposal_id"],
            "draft_hash": wire_turn["payload"]["draft_hash"],
            "chosen": ["confirm_wiring"],
        }
        accepted = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=confirm_request,
        )

        assert accepted.status_code == 200, accepted.json()
        body = accepted.json()
        guided = _full_guided_session(body)
        assert guided["active_proposal"] is None
        assert guided["deferred_intents"] == []
        assert body["terminal"]["kind"] == "completed"
        assert body["next_turn"] is None
        assert body["composition_state"]["outputs"]
        events = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        replayed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json=confirm_request,
        )
        assert replayed.status_code == 200, replayed.json()
        assert replayed.json() == body
        restored = _get_guided(composer_test_client, session_id)
        for storage_path in storage_paths:
            assert storage_path not in json.dumps(body)
            assert storage_path not in replayed.text
            assert storage_path not in json.dumps(restored)
            assert all(storage_path not in repr(event.payload) for event in events)

    def test_respond_planner_call_threads_a_live_progress_sink(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The respond route's planner runs must publish live phase progress.

        The decision-progress indicator (6996bdb38) had no phase text on the
        respond path: post_guided_respond wired no progress sink into its
        plan_guided_pipeline calls (guided_plan.py does). The sink must reach
        the planner and its events must land in the app's progress registry so
        GET /composer/progress serves phase text during the multi-minute
        planner run.
        """
        from elspeth.contracts.composer_progress import ComposerProgressEvent

        session_id = _create_session(composer_test_client)
        app = composer_test_client.app
        original = app.state.composer_service.plan_guided_pipeline
        seen: dict[str, Any] = {}

        async def capturing_planner(**kwargs: Any):
            seen["progress"] = kwargs.get("progress")
            if seen["progress"] is not None:
                await seen["progress"](
                    ComposerProgressEvent(
                        phase="calling_model",
                        headline="Planning the pipeline against the reviewed components.",
                    )
                )
            return await original(**{k: v for k, v in kwargs.items() if k != "progress"})

        monkeypatch.setattr(app.state.composer_service, "plan_guided_pipeline", capturing_planner)
        self._stage_proposal(composer_test_client, session_id, filename="progress_sink.jsonl")

        assert seen.get("progress") is not None, "plan_guided_pipeline received no progress sink"
        snapshot = asyncio.run(app.state.composer_progress_registry.get_latest(session_id))
        assert snapshot.phase == "calling_model"
        assert snapshot.headline == "Planning the pipeline against the reviewed components."

    def test_confirm_wiring_surfaces_pending_interpretation_events_for_committed_llm_prompts(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The wire-confirm commit must surface the events freeform settlement surfaces.

        Tutorial session e1332b5a: the guided walk completed with a committed
        llm node carrying a canonicalized pending ``llm_prompt_template``
        requirement, but ZERO interpretation_events rows existed — the guided
        wire-confirm path never ran the post-commit surfacing pass that both
        the freeform settlement route (pipeline_settlement.py) and the guided
        CHAT dispatcher run. No Accept card could ever render, and /execute
        then refused with ``UnresolvedInterpretationPlaceholderError``. After
        confirm, a pending event must exist for the committed node, bound to
        the accepted durable state (the writer boundary validates the node
        against that state's row).
        """
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.core.canonical import stable_hash
        from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
        from elspeth.web.composer.pipeline_planner import PipelinePlanResult
        from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface

        session_id = _create_session(composer_test_client)
        prompt = "Summarise this row in one short sentence."

        async def llm_planner(
            *,
            guided,
            base,
            supersedes_draft_hash,
            recorder,
            correction_target=None,
            **_kwargs,
        ):
            del recorder, correction_target
            source = guided.reviewed_sources[guided.source_order[0]]
            output = guided.reviewed_outputs[guided.output_order[0]]
            pipeline = {
                "sources": {
                    source.name: {
                        "plugin": source.plugin,
                        "options": deep_thaw(source.options),
                        "on_success": "llm_rows",
                        "on_validation_failure": source.on_validation_failure,
                    }
                },
                "nodes": [
                    {
                        "id": "summarize_rows",
                        "node_type": "transform",
                        "plugin": "llm",
                        "input": "llm_rows",
                        "on_success": output.name,
                        "on_error": "discard",
                        "options": {
                            "schema": {"mode": "observed"},
                            "profile": "task-role",
                            "prompt_template": prompt,
                            "response_field": "summary",
                            "interpretation_requirements": [
                                {
                                    "id": "llm_prompt_template:summarize_rows:summarize_rows",
                                    "kind": "llm_prompt_template",
                                    "user_term": "llm_prompt_template:summarize_rows",
                                    "status": "pending",
                                    "draft": prompt,
                                }
                            ],
                        },
                    }
                ],
                "edges": [],
                "outputs": [
                    {
                        "sink_name": output.name,
                        "plugin": output.plugin,
                        "options": deep_thaw(output.options),
                        "on_write_failure": output.on_write_failure,
                    }
                ],
            }
            proposal = PipelineProposal.create(
                pipeline=pipeline,
                base=base,
                reviewed_facts=guided_private_reviewed_facts(guided),
                surface=PlannerSurface.GUIDED_STAGED,
                repair_count=0,
                skill_hash=stable_hash("llm-prompt-review-test-planner"),
                covered_deferred_intent_ids=(),
                supersedes_draft_hash=supersedes_draft_hash,
            )
            return (
                PipelinePlanResult(
                    proposal=proposal,
                    tool_call_id=f"guided-test-{proposal.draft_hash[:16]}",
                    custody_result="not_required",
                    model_identifier="llm-prompt-review-test-planner",
                    model_version="v1",
                    provider="test",
                ),
                {
                    "source": frozenset({source.plugin}),
                    "transform": frozenset({"llm"}),
                    "sink": frozenset({output.plugin}),
                },
            )

        monkeypatch.setattr(
            composer_test_client.app.state.composer_service,
            "plan_guided_pipeline",
            llm_planner,
        )
        staged = self._stage_proposal(composer_test_client, session_id, filename="llm_reviewed.jsonl")
        assert staged["next_turn"]["type"] == "propose_pipeline"

        _review_wiring(composer_test_client, session_id)
        accepted = _confirm_wiring(composer_test_client, session_id)
        assert accepted["terminal"]["kind"] == "completed"

        session_service = composer_test_client.app.state.session_service
        events = asyncio.run(session_service.list_interpretation_events(UUID(session_id), status="pending"))
        prompt_events = [event for event in events if event.affected_node_id == "summarize_rows"]
        assert len(prompt_events) == 1, [(event.affected_node_id, str(event.kind), event.user_term) for event in events]
        event = prompt_events[0]
        assert event.kind is not None and event.kind.value == "llm_prompt_template"
        assert event.llm_draft == prompt
        # Bound to the accepted durable state — the writer boundary validated
        # the node against that state's persisted nodes JSON.
        current = asyncio.run(session_service.get_current_state(UUID(session_id)))
        assert current is not None
        assert str(event.composition_state_id) == str(current.id)

    def test_confirm_wiring_failure_after_dispatch_audit_insert_preserves_failure_evidence_only(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from elspeth.web.sessions import service as service_module

        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-audit-rollback.jsonl")
        assert staged["next_turn"]["type"] == "propose_pipeline"
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
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
                "chosen": ["confirm_wiring"],
            },
        )

        assert failed.status_code == 500, failed.json()
        messages_after = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
        new_messages = [message for message in messages_after if message.id not in {item.id for item in messages_before}]
        assert len(new_messages) == 1
        assert new_messages[0].role == "audit"
        assert len(new_messages[0].tool_calls or ()) == 1
        invocation = (new_messages[0].tool_calls or ())[0]["invocation"]
        assert invocation["tool_name"] == "set_pipeline"
        assert invocation["status"] == "success"
        assert "accept-audit-rollback.jsonl" not in repr(new_messages[0].tool_calls)
        assert asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id))) == events_before
        state_after = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        assert state_before is not None and state_after is not None and state_after.id == state_before.id
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"

    def test_confirm_wiring_prepare_failure_persists_its_dispatch_evidence_with_the_failed_operation(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer import pipeline_commit
        from elspeth.web.composer.audit import begin_dispatch, finish_success
        from elspeth.web.composer.pipeline_commit import PipelineCommitError

        session_id = _create_session(composer_test_client)
        self._stage_proposal(composer_test_client, session_id, filename="confirm-prepare-audit.jsonl")
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
        operation_id = str(uuid4())

        async def fail_after_recording_dispatch(**kwargs: Any):
            authority = kwargs["authority"]
            recorder = kwargs["recorder"]
            audit = begin_dispatch(
                authority.row.tool_call_id,
                "set_pipeline",
                deep_thaw(authority.proposal.pipeline),
                version_before=0,
                actor=kwargs["actor"],
            )
            invocation = finish_success(
                audit,
                result_payload={"success": False, "failure_code": "validation_failed"},
                version_after=0,
            )
            recorder.record(invocation)
            raise PipelineCommitError(
                "pipeline proposal failed current executor validation",
                code="VALIDATION_FAILED",
                invocation=invocation,
            )

        monkeypatch.setattr(pipeline_commit, "prepare_pipeline_proposal_commit", fail_after_recording_dispatch)

        failed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": operation_id,
                "turn_token": turn["turn_token"],
                "proposal_id": turn["payload"]["proposal_id"],
                "draft_hash": turn["payload"]["draft_hash"],
                "chosen": ["confirm_wiring"],
            },
        )

        assert failed.status_code == 500, failed.json()
        assert failed.json()["detail"]["failure_code"] == "operation_failed"
        service = composer_test_client.app.state.session_service
        proposals = asyncio.run(service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"
        messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        dispatches = [
            envelope
            for message in messages
            for envelope in (message.tool_calls or ())
            if envelope.get("invocation", {}).get("tool_name") == "set_pipeline"
        ]
        assert len(dispatches) == 1
        assert dispatches[0]["invocation"]["status"] == "success"
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
        assert operation["status"] == "failed"
        assert operation["failure_code"] == "operation_failed"

    @pytest.mark.parametrize("revalidation", ("message", "mechanical"))
    def test_confirm_wiring_revalidates_deferred_authority_before_any_write(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        revalidation: str,
    ) -> None:
        from elspeth.web.composer.guided import planning
        from elspeth.web.composer.guided.deferred_intents import DeferredIntentAction
        from elspeth.web.composer.guided.stage_subjects import ComponentCountConstraint
        from elspeth.web.sessions import service as service_module
        from elspeth.web.sessions.routes.composer import guided as guided_route
        from tests.integration.web.composer.guided.test_wrong_stage_intent import _provider

        session_id = _create_session(composer_test_client)
        started = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={
                "profile": "live",
                "intent": "Begin guided authoring before refining topology requirements.",
                "operation_id": str(uuid4()),
            },
        )
        assert started.status_code == 200, started.json()
        current = _get_guided(composer_test_client, session_id)
        action = DeferredIntentAction(
            target_stage="topology",
            catalog_kind="transform",
            catalog_name="passthrough",
            redacted_summary="Retain a mechanically testable topology constraint.",
            constraints=(
                ComponentCountConstraint(
                    kind="component_count",
                    component_kind="node",
                    plugin_kind="transform",
                    plugin_name="passthrough",
                    operator="at_most",
                    count=0,
                ),
            ),
        )
        monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _provider(action))
        retained = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/chat",
            json={
                "operation_id": str(uuid4()),
                "turn_token": current["next_turn"]["turn_token"],
                "message": "Do not include passthrough later.",
            },
        )
        assert retained.status_code == 200, retained.json()
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-deferred-revalidation.jsonl")
        assert staged["next_turn"]["type"] == "propose_pipeline"
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
        state_before = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        events_before = asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id)))
        messages_before = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))

        if revalidation == "message":

            def reject_corrupt_authority(*_args, **_kwargs):
                raise AuditIntegrityError("guided deferred intent message content hash mismatch")

            monkeypatch.setattr(service_module, "_verify_guided_deferred_message_authority", reject_corrupt_authority)
        else:
            original_verifier = planning.verified_remaining_deferred_intents
            verification_calls = 0

            def reject_second_verification(**kwargs):
                nonlocal verification_calls
                verification_calls += 1
                if verification_calls == 2:
                    raise AuditIntegrityError("guided deferred mechanical coverage drifted before acceptance")
                return original_verifier(**kwargs)

            monkeypatch.setattr(planning, "verified_remaining_deferred_intents", reject_second_verification)
        failed = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "operation_id": str(uuid4()),
                "turn_token": turn["turn_token"],
                "proposal_id": turn["payload"]["proposal_id"],
                "draft_hash": turn["payload"]["draft_hash"],
                "chosen": ["confirm_wiring"],
            },
        )

        assert failed.status_code == 500, failed.json()
        state_after = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
        assert state_before is not None and state_after is not None and state_after.id == state_before.id
        assert asyncio.run(composer_test_client.app.state.session_service.list_proposal_events(UUID(session_id))) == events_before
        messages_after = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
        new_messages = [message for message in messages_after if message.id not in {item.id for item in messages_before}]
        assert len(new_messages) == 1
        assert new_messages[0].role == "audit"
        assert len(new_messages[0].tool_calls or ()) == 1
        invocation = (new_messages[0].tool_calls or ())[0]["invocation"]
        assert invocation["tool_name"] == "set_pipeline"
        assert invocation["status"] == "success"
        assert "accept-deferred-revalidation.jsonl" not in repr(new_messages[0].tool_calls)
        proposals = asyncio.run(composer_test_client.app.state.session_service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "pending"
        if revalidation == "mechanical":
            assert verification_calls == 2

    def test_confirm_wiring_cancellation_after_dispatch_cannot_orphan_acceptance_audit(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-cancel-before-service.jsonl")
        assert staged["next_turn"]["type"] == "propose_pipeline"
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
        service = composer_test_client.app.state.session_service
        original_accept = service.accept_guided_pipeline_proposal
        entered = asyncio.Event()
        release = asyncio.Event()
        request_body = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "proposal_id": turn["payload"]["proposal_id"],
            "draft_hash": turn["payload"]["draft_hash"],
            "chosen": ["confirm_wiring"],
        }

        async def blocked_accept(command, *, payload_store=None):
            entered.set()
            await release.wait()
            return await original_accept(command, payload_store=payload_store)

        monkeypatch.setattr(
            composer_test_client.app.state.session_service,
            "accept_guided_pipeline_proposal",
            blocked_accept,
        )

        async def cancel_after_dispatch():
            async with AsyncClient(
                transport=ASGITransport(app=composer_test_client.app),
                base_url="http://test",
            ) as client:
                task = asyncio.create_task(
                    client.post(
                        f"/api/sessions/{session_id}/guided/respond",
                        json=request_body,
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                return await client.post(
                    f"/api/sessions/{session_id}/guided/respond",
                    json=request_body,
                )

        replayed = asyncio.run(cancel_after_dispatch())

        assert replayed.status_code == 200, replayed.json()
        events = asyncio.run(service.list_proposal_events(UUID(session_id)))
        assert [event.event_type for event in events] == ["proposal.created", "proposal.accepted"]
        proposals = asyncio.run(service.list_composition_proposals(UUID(session_id)))
        assert len(proposals) == 1 and proposals[0].status == "committed"
        messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
        dispatches = [
            envelope
            for message in messages
            for envelope in (message.tool_calls or ())
            if envelope.get("invocation", {}).get("tool_name") == "set_pipeline"
            and envelope.get("invocation", {}).get("status") == "success"
        ]
        assert len(dispatches) == 1

    def test_confirm_wiring_cancellation_during_worker_settlement_is_exactly_replayable(
        self,
        composer_test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id = _create_session(composer_test_client)
        staged = self._stage_proposal(composer_test_client, session_id, filename="accept-cancel-during-worker.jsonl")
        assert staged["next_turn"]["type"] == "propose_pipeline"
        reviewed = _review_wiring(composer_test_client, session_id)
        turn = reviewed["next_turn"]
        request_body = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "proposal_id": turn["payload"]["proposal_id"],
            "draft_hash": turn["payload"]["draft_hash"],
            "chosen": ["confirm_wiring"],
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
                release_worker.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
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
