"""Tests for session API routes -- CRUD, IDOR, fork, YAML."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.blobs import BlobNotFoundError, BlobServiceProtocol
from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolStatus,
)
from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus, ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.composer_progress import ComposerProgressEvent
from elspeth.contracts.enums import CreationModality, TerminalOutcome, TerminalPath
from elspeth.contracts.hashing import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
    transform_errors_table,
    validation_errors_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import TurnResponse, TurnType
from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.state_machine import GuidedSession, GuidedStep, TerminalKind, TerminalReason, TerminalState
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.protocol import ComposerPluginCrashError, ComposerResult, ComposerService, PipelineCommitIntent
from elspeth.web.composer.redaction import REDACTED_BLOB_SOURCE_PATH
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec, ValidationSummary
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
)
from elspeth.web.execution.schemas import (
    ValidationResult as ValidationResultModel,
)
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.provider_config_policy import AWS_S3_ENDPOINT_URL_POLICY_ERROR
from elspeth.web.sessions._guided_step_chat import Step1SourceChatResult, StepChatResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import (
    ChatMessageRecord,
    ChatMessageRole,
    CompositionStateData,
    CompositionStateRecord,
    SessionRecord,
)
from elspeth.web.sessions.routes import _summarize_guided_response, create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# Sentinel empty state for mock composer responses
_EMPTY_STATE = CompositionState(
    source=None,
    nodes=(),
    edges=(),
    outputs=(),
    metadata=PipelineMetadata(),
    version=1,
)


class _RecordedSyncCall:
    def __init__(self, side_effect: Any | None = None) -> None:
        self.side_effect = side_effect
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None:
            return self.side_effect(*args, **kwargs)
        return None

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert self.calls == [(args, kwargs)]


class _ExecutionServiceStub:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self.cleanup_session_lock = _RecordedSyncCall()

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]


def _async_return(value: Any):
    async def _return_value(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _return_value


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _blocked_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[])


def ValidationResult(
    *,
    is_valid: bool,
    checks: list[ValidationCheck],
    errors: list[ValidationError],
    readiness: ValidationReadiness | None = None,
    **kwargs: Any,
) -> ValidationResultModel:
    return ValidationResultModel(
        is_valid=is_valid,
        checks=checks,
        errors=errors,
        readiness=readiness or (_ready_readiness() if is_valid else _blocked_readiness()),
        **kwargs,
    )


def test_summarize_guided_response_rejects_unhandled_turn_type() -> None:
    response: TurnResponse = {
        "chosen": None,
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }

    with pytest.raises(InvariantError, match="unhandled turn_type"):
        _summarize_guided_response(cast(TurnType, object()), response)


def _make_composer_mock(
    response_text: str = "Sure, I can help.",
    state: CompositionState | None = None,
) -> SimpleNamespace:
    """Create a mock ComposerServiceImpl.compose that returns a fixed result."""
    mock = SimpleNamespace()
    mock.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message=response_text,
            state=state or _EMPTY_STATE,
        ),
    )
    return mock


def _audit_tool_calls(tool_call_id: str = "call-audit") -> list[dict[str, Any]]:
    return [
        {
            "_kind": "audit",
            "invocation": {
                "tool_call_id": tool_call_id,
            },
        }
    ]


def _llm_call(**overrides: Any) -> ComposerLLMCall:
    defaults: dict[str, Any] = {
        "model_requested": "openrouter/openai/gpt-5.5",
        "model_returned": "openai/gpt-5.5-2026-05-01",
        "status": ComposerLLMCallStatus.SUCCESS,
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "total_tokens": 21,
        "latency_ms": 42,
        "provider_request_id": "chatcmpl-route",
        "messages_hash": "m" * 64,
        "tools_spec_hash": "t" * 64,
        "declared_tool_names": ("set_pipeline",),
        "started_at": datetime.now(UTC),
        "finished_at": datetime.now(UTC),
        "error_class": None,
        "error_message": None,
        "temperature": 0.0,
        "seed": 42,
    }
    defaults.update(overrides)
    return ComposerLLMCall(**defaults)


def _cancelled_error_with_llm_call(call: ComposerLLMCall) -> asyncio.CancelledError:
    exc = asyncio.CancelledError()
    exc_with_calls = cast(Any, exc)
    exc_with_calls.llm_calls = (call,)
    return exc


def _llm_call_audit_tool_calls(call: ComposerLLMCall | None = None) -> list[dict[str, Any]]:
    return [
        {
            "_kind": "llm_call_audit",
            "call": (call or _llm_call()).to_dict(),
        }
    ]


def _llm_call_audit_rows(messages: Sequence[ChatMessageRecord]) -> list[tuple[ChatMessageRecord, Mapping[str, Any]]]:
    """Rev-4: LLM-call audit sidecars are stored with ``role="audit"``."""
    rows: list[tuple[ChatMessageRecord, Mapping[str, Any]]] = []
    for message in messages:
        if message.role != "audit" or message.tool_calls is None:
            continue
        first_tool_call = message.tool_calls[0]
        if first_tool_call.get("_kind") == "llm_call_audit":
            rows.append((message, first_tool_call))
    return rows


class _BlockingRecordingComposer:
    """Composer stub that lets tests observe and gate concurrent compose() calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.first_call_started = asyncio.Event()
        self.second_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()

    async def compose(
        self,
        message: str,
        chat_messages: list[dict[str, object]],
        state: CompositionState,
        *,
        session_id: str | None = None,
        current_state_id: str | None = None,
        user_id: str | None = None,
        progress=None,
        guided_terminal=None,
        user_message_id: str | None = None,
    ) -> ComposerResult:
        del state, session_id, current_state_id, user_id, progress, guided_terminal, user_message_id

        self.calls.append(
            {
                "message": message,
                "chat_messages": [dict(entry) for entry in chat_messages],
            }
        )

        if len(self.calls) == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
            reply = "Reply to first"
        else:
            self.second_call_started.set()
            reply = "Reply to second"

        return ComposerResult(message=reply, state=_EMPTY_STATE)


class _ProgressAwareComposer:
    """Composer stub that proves routes provide a progress sink."""

    def __init__(self, response_text: str = "Progress-aware reply") -> None:
        self.response_text = response_text
        self.progress_sink_seen = False

    async def compose(
        self,
        message: str,
        chat_messages: list[dict[str, object]],
        state: CompositionState,
        *,
        session_id: str | None = None,
        current_state_id: str | None = None,
        user_id: str | None = None,
        progress=None,
        guided_terminal=None,
        user_message_id: str | None = None,
    ) -> ComposerResult:
        del message, chat_messages, session_id, current_state_id, user_id, guided_terminal, user_message_id
        assert progress is not None, "session routes must pass a composer progress sink"
        self.progress_sink_seen = True
        await progress(
            ComposerProgressEvent(
                phase="calling_model",
                headline="I'm asking the model to choose a pipeline update.",
                evidence=("The route supplied a session-scoped progress sink.",),
                likely_next="ELSPETH will save any accepted pipeline update.",
            )
        )
        updated_state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="progress test"),
            version=state.version + 1,
        )
        return ComposerResult(message=self.response_text, state=updated_state)


class _ProgressRouteSessionService:
    """Minimal async session service for progress route tests."""

    def __init__(self, *, user_id: str = "alice", auth_provider_type: str = "local") -> None:
        now = datetime.now(UTC)
        self.session = SessionRecord(
            id=uuid.uuid4(),
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title="Pipeline",
            created_at=now,
            updated_at=now,
        )
        self.messages: list[ChatMessageRecord] = []
        self.current_state: CompositionStateRecord | None = None

    async def get_session(self, session_id: uuid.UUID) -> SessionRecord:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        return self.session

    async def get_current_state(self, session_id: uuid.UUID) -> CompositionStateRecord | None:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        return self.current_state

    async def add_message(
        self,
        session_id: uuid.UUID,
        role: ChatMessageRole,
        content: str,
        *,
        writer_principal: str,
        tool_calls=None,
        composition_state_id: uuid.UUID | None = None,
        raw_content: str | None = None,
        tool_call_id: str | None = None,
        parent_assistant_id: uuid.UUID | None = None,
    ) -> ChatMessageRecord:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        message = ChatMessageRecord(
            id=uuid.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            tool_calls=tool_calls,
            created_at=datetime.now(UTC),
            composition_state_id=composition_state_id,
            writer_principal=writer_principal,
            tool_call_id=tool_call_id,
            parent_assistant_id=parent_assistant_id,
        )
        self.messages.append(message)
        return message

    async def get_messages(
        self,
        session_id: uuid.UUID,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[ChatMessageRecord]:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        del offset
        if limit is None:
            return list(self.messages)
        return list(self.messages[:limit])

    async def save_composition_state(
        self,
        session_id: uuid.UUID,
        data: CompositionStateData,
        *,
        provenance: str,
    ) -> CompositionStateRecord:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        version = 1 if self.current_state is None else self.current_state.version + 1
        # Stub records the provenance label so handler tests can assert on it
        # (the production INSERT writes it to the ``provenance`` column under
        # the ``ck_composition_states_provenance`` CHECK; the stub mirrors the
        # contract through the ``last_save_provenance`` attribute).
        self.last_save_provenance = provenance
        record = CompositionStateRecord(
            id=uuid.uuid4(),
            session_id=session_id,
            version=version,
            source=None,
            nodes=data.nodes,
            edges=data.edges,
            outputs=data.outputs,
            metadata_=data.metadata_,
            is_valid=data.is_valid,
            validation_errors=data.validation_errors,
            created_at=datetime.now(UTC),
            derived_from_state_id=self.current_state.id if self.current_state is not None else None,
            composer_meta=data.composer_meta,
            sources=data.sources,
        )
        self.current_state = record
        return record

    async def list_composition_proposals(
        self,
        session_id: uuid.UUID,
        status: str | None = None,
    ) -> list[Any]:
        if session_id != self.session.id:
            raise ValueError("Session not found")
        del status
        return []


def _make_progress_route_app(
    tmp_path: Path,
    *,
    user_id: str = "alice",
) -> tuple[FastAPI, _ProgressRouteSessionService]:
    app = FastAPI()
    service = _ProgressRouteSessionService(user_id=user_id)
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = service
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.payload_store = FilesystemPayloadStore(app.state.settings.get_payload_store_path())
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = _ExecutionServiceStub()
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.scoped_secret_resolver = None
    _install_restricted_plugin_policy(app)
    app.include_router(create_session_router())
    return app, service


def _make_app(
    tmp_path: Path,
    user_id: str = "alice",
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> tuple[FastAPI, SessionServiceImpl]:
    """Create a test app with session routes and a mock auth user."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    telemetry = build_sessions_telemetry()
    service = SessionServiceImpl(
        engine,
        telemetry=telemetry,
        log=structlog.get_logger("test"),
    )

    app = FastAPI()

    # Override auth dependency to return a fixed user
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    # Set up app state
    app.state.session_service = service
    # Phase 6A B3 — YAML export route's audit-write reaches into
    # ``app.state.session_engine`` for the composer_completion_events insert.
    app.state.session_engine = engine
    # Phase 8 Sub-task 7c — the YAML-export route emits
    # ``composer.session.completed_total`` via
    # ``request.app.state.sessions_telemetry``. Mirror the same
    # container the service holds so route-emit and service-emit
    # observe a single counter (matches production wiring in
    # ``web/app.py:586``).
    app.state.sessions_telemetry = telemetry
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        max_upload_bytes=max_upload_bytes,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.payload_store = FilesystemPayloadStore(app.state.settings.get_payload_store_path())
    # composer_service is set to None here; tests that POST messages
    # must replace it with a mock before sending requests.
    app.state.composer_service = None

    from elspeth.web.middleware.rate_limit import ComposerRateLimiter

    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.scoped_secret_resolver = None
    _install_restricted_plugin_policy(app)

    # Minimal stub for execution service — delete_session coordinates with
    # the per-session execution lock and then cleans it up after archiving.
    app.state.execution_service = _ExecutionServiceStub()

    router = create_session_router()
    app.include_router(router)

    return app, service


def _install_restricted_plugin_policy(app: FastAPI, *hidden: PluginId) -> PluginAvailabilitySnapshot:
    """Install one deterministic principal policy on a hand-rolled route app."""
    catalog = create_catalog_service()
    unrestricted = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="session-route-policy",
        principal_scope="local:alice",
        available=unrestricted.available - set(hidden),
        unavailable=(),
        selected=unrestricted.selected,
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="session-route-policy-generation",
    )
    app.state.catalog_service = catalog
    profile_registry = MagicMock(spec=OperatorProfileRegistry)
    profile_registry.public_schema.side_effect = lambda _plugin_id, full_schema, *, available_aliases: full_schema
    app.state.operator_profile_registry = profile_registry
    app.state.plugin_snapshot_factory = lambda _user: snapshot
    return snapshot


def test_get_composer_preferences_returns_defaults(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Prefs"}).json()

    response = test_client.get(f"/api/sessions/{session['id']}/composer/preferences")

    assert response.status_code == 200
    # Default trust_mode is auto_commit (commit c4e2f69cd reverted from
    # explicit_approve — see sessions_table.trust_mode comment in models.py).
    assert response.json()["trust_mode"] == "auto_commit"
    assert response.json()["density_default"] == "high"
    assert response.json()["interpretation_review_disabled"] is False


def test_patch_composer_preferences_records_event(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Prefs"}).json()

    response = test_client.patch(
        f"/api/sessions/{session['id']}/composer/preferences",
        json={"trust_mode": "auto_commit", "density_default": "medium"},
    )

    assert response.status_code == 200
    assert response.json()["trust_mode"] == "auto_commit"
    assert response.json()["interpretation_review_disabled"] is False
    events = test_client.get(f"/api/sessions/{session['id']}/proposal-events").json()
    assert events[-1]["event_type"] == "trust_mode.changed"


def test_list_proposals_is_session_scoped(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Proposals"}).json()

    response = test_client.get(f"/api/sessions/{session['id']}/proposals")

    assert response.status_code == 200
    assert response.json() == []


def test_send_message_response_includes_empty_proposals_array(tmp_path) -> None:
    mock_composer = _make_composer_mock(response_text="Got it!")
    app, _service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Chat"}).json()

    response = client.post(
        f"/api/sessions/{session['id']}/messages",
        json={"content": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["proposals"] == []


def test_send_message_response_includes_pending_proposals_created_during_compose(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    mock_composer = SimpleNamespace()

    async def _compose_with_pending_proposal(*args: object, **kwargs: object) -> ComposerResult:
        del args
        session_id = uuid.UUID(str(kwargs["session_id"]))
        await service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline",
            tool_name="set_pipeline",
            summary="Replace the pipeline.",
            rationale="Requested by the current composer turn.",
            affects=("graph", "yaml"),
            arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
            arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
            base_state_id=None,
            actor="composer-web:alice",
        )
        return ComposerResult(message="Needs approval.", state=_EMPTY_STATE)

    mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=_compose_with_pending_proposal)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Atomic proposals"}).json()

    response = client.post(
        f"/api/sessions/{session['id']}/messages",
        json={"content": "Build a csv pipeline"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["content"] == "Needs approval."
    assert body["proposals"][0]["tool_call_id"] == "call_set_pipeline"
    assert body["proposals"][0]["status"] == "pending"


def test_accept_unknown_proposal_returns_404(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Accept"}).json()

    response = test_client.post(f"/api/sessions/{session['id']}/proposals/00000000-0000-0000-0000-000000000000/accept")

    assert response.status_code == 404


def test_accept_proposal_executes_tool_and_commits_state(tmp_path, monkeypatch) -> None:
    from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary

    app, service = _make_app(tmp_path)
    app.state.session_engine = service._engine
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="passthrough", description="Passthrough", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="csv", description="CSV sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    app.state.catalog_service = catalog
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    input_path = tmp_path / "blobs" / "input.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("value\n1\n", encoding="utf-8")
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Accept"}).json()
    session_id = uuid.UUID(session["id"])
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline",
            tool_name="set_pipeline",
            summary="Replace the pipeline.",
            rationale="Requested by the current composer turn.",
            affects=("graph", "validation", "yaml"),
            arguments_json={
                "sources": {
                    "primary": {
                        "plugin": "csv",
                        "on_success": "source_out",
                        "options": {"path": str(input_path), "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    }
                },
                "nodes": [
                    {
                        "id": "t1",
                        "node_type": "transform",
                        "plugin": "passthrough",
                        "input": "source_out",
                        "on_success": "main",
                        "on_error": "discard",
                        "options": {"schema": {"mode": "observed"}},
                    }
                ],
                "edges": [
                    {
                        "id": "e1",
                        "from_node": "source",
                        "to_node": "t1",
                        "edge_type": "on_success",
                        "label": None,
                    }
                ],
                "outputs": [
                    {
                        "sink_name": "main",
                        "plugin": "csv",
                        "options": {
                            "path": str(tmp_path / "outputs" / "output.csv"),
                            "schema": {"mode": "observed"},
                            "mode": "write",
                            "collision_policy": "auto_increment",
                        },
                        "on_write_failure": "discard",
                    }
                ],
                "metadata": {"name": "accepted-proposal"},
            },
            arguments_redacted_json={"summary": "redacted"},
            base_state_id=None,
            actor="composer-web:user:alice",
        )
    )

    response = client.post(f"/api/sessions/{session['id']}/proposals/{proposal.id}/accept")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "committed"
    assert body["committed_state_id"] is not None
    persisted = asyncio.run(service.get_current_state(session_id))
    assert persisted is not None
    from sqlalchemy import select

    from elspeth.web.sessions.models import composition_states_table

    with service._engine.begin() as conn:
        provenance = conn.execute(
            select(composition_states_table.c.provenance).where(composition_states_table.c.id == str(persisted.id))
        ).scalar_one()
    assert provenance == "tool_call"


async def _create_canonical_pipeline_route_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_call_id: str,
) -> tuple[FastAPI, SessionServiceImpl, dict[str, Any], uuid.UUID, Any, str]:
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

    app, service = _make_app(tmp_path)
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    input_path = tmp_path / "blobs" / f"{tool_call_id}.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("value\n1\n", encoding="utf-8")
    pipeline: dict[str, Any] = {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(input_path), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "rows",
                "plugin": "json",
                "options": {
                    "path": str(tmp_path / "outputs" / f"{tool_call_id}.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
    }
    proposal = PipelineProposal.create(
        pipeline=pipeline,
        base=AbsentBase(),
        reviewed_facts={},
        surface=PlannerSurface.FREEFORM,
        repair_count=0,
        skill_hash=stable_hash("planner-skill"),
    )
    session = await service.create_session("alice", "Canonical accept", "local")
    session_id = session.id
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=PipelinePlanResult(
            proposal=proposal,
            tool_call_id=tool_call_id,
            custody_result="not_required",
            model_identifier="planner-model",
            model_version="planner-model-v1",
            provider="test",
        ),
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph", "validation"),
        arguments_redacted_json=redact_tool_call_arguments(
            "set_pipeline",
            pipeline,
            telemetry=NoopRedactionTelemetry(),
        ),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )
    endpoint = f"/api/sessions/{session.id}/proposals/{row.id}/accept"
    return app, service, pipeline, session_id, row, endpoint


def test_send_message_auto_commit_settles_exact_pipeline_intent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, _pipeline, session_id, row, _endpoint = asyncio.run(
        _create_canonical_pipeline_route_proposal(tmp_path, monkeypatch, tool_call_id="send-auto-pipeline")
    )
    assert row.pipeline_metadata is not None
    composer = SimpleNamespace()
    composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="Pipeline prepared.",
            state=_EMPTY_STATE,
            repair_turns_used=2,
            pipeline_commit_intent=PipelineCommitIntent(
                proposal_id=row.id,
                draft_hash=row.pipeline_metadata.draft_hash,
            ),
        ),
    )
    app.state.composer_service = composer
    client = TestClient(app)

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build the pipeline."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] is not None
    assert body["proposals"] == []
    settled = asyncio.run(service.list_composition_proposals(session_id))
    assert len(settled) == 1
    assert settled[0].status == "committed"
    assert settled[0].committed_state_id == uuid.UUID(body["state"]["id"])
    current_state = asyncio.run(service.get_current_state(session_id))
    assert current_state is not None
    assert current_state.composer_meta is not None
    assert current_state.composer_meta["repair_turns_used"] == 2
    from sqlalchemy import func, select

    from elspeth.web.sessions.models import composition_states_table

    with service._engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == 1
    messages = asyncio.run(service.get_messages(session_id, limit=None))
    assistant = next(message for message in reversed(messages) if message.role == "assistant")
    assert assistant.composition_state_id == settled[0].committed_state_id


def test_send_message_explicit_approval_leaves_canonical_pipeline_pending(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, _pipeline, session_id, row, _endpoint = asyncio.run(
        _create_canonical_pipeline_route_proposal(tmp_path, monkeypatch, tool_call_id="send-explicit-pipeline")
    )
    asyncio.run(
        service.update_composer_preferences(
            session_id,
            trust_mode="explicit_approve",
            density_default="high",
            actor="user:alice",
        )
    )
    composer = SimpleNamespace()
    composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(message="Pipeline prepared for review.", state=_EMPTY_STATE),
    )
    app.state.composer_service = composer
    client = TestClient(app)

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build the pipeline."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] is None
    assert len(body["proposals"]) == 1
    assert body["proposals"][0]["id"] == str(row.id)
    assert body["proposals"][0]["status"] == "pending"
    assert asyncio.run(service.get_current_state(session_id)) is None


def test_recompose_auto_commit_uses_shared_pipeline_settlement(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, _pipeline, session_id, row, _endpoint = asyncio.run(
        _create_canonical_pipeline_route_proposal(tmp_path, monkeypatch, tool_call_id="recompose-auto-pipeline")
    )
    asyncio.run(
        service.add_message(
            session_id,
            "user",
            "Build the pipeline.",
            writer_principal="route_user_message",
        )
    )
    assert row.pipeline_metadata is not None
    composer = SimpleNamespace()
    composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="Pipeline prepared.",
            state=_EMPTY_STATE,
            repair_turns_used=1,
            pipeline_commit_intent=PipelineCommitIntent(
                proposal_id=row.id,
                draft_hash=row.pipeline_metadata.draft_hash,
            ),
        ),
    )
    app.state.composer_service = composer
    client = TestClient(app)

    response = client.post(f"/api/sessions/{session_id}/recompose")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] is not None
    assert body["proposals"] == []
    settled = asyncio.run(service.list_composition_proposals(session_id))
    assert settled[0].status == "committed"
    current_state = asyncio.run(service.get_current_state(session_id))
    assert current_state is not None
    assert current_state.composer_meta is not None
    assert current_state.composer_meta["repair_turns_used"] == 1
    from sqlalchemy import func, select

    from elspeth.web.sessions.models import composition_states_table

    with service._engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == 1


@pytest.mark.parametrize(
    ("failure_mode", "reason_code"),
    [
        ("validation", "validation_failed"),
        ("mismatch", "candidate_executor_mismatch"),
    ],
)
def test_canonical_pipeline_accept_terminalizes_audited_executor_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    reason_code: str,
) -> None:
    import elspeth.web.composer.pipeline_commit as commit_module

    app, service, _pipeline, session_id, row, endpoint = asyncio.run(
        _create_canonical_pipeline_route_proposal(
            tmp_path,
            monkeypatch,
            tool_call_id=f"canonical-{failure_mode}-call",
        )
    )
    original_execute = commit_module.execute_tool

    def failing_execute(*args: Any, **kwargs: Any):
        result = original_execute(*args, **kwargs)
        if failure_mode == "validation":
            return replace(result, success=False)
        return replace(
            result,
            updated_state=CompositionState(
                source=None,
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(name="executor mismatch"),
                version=result.updated_state.version,
            ),
        )

    monkeypatch.setattr(commit_module, "execute_tool", failing_execute)

    response = TestClient(app).post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash})

    assert response.status_code == 422, response.text
    assert asyncio.run(service.get_current_state(session_id)) is None
    terminal_row = asyncio.run(service.get_authoritative_composition_proposal(session_id=session_id, proposal_id=row.id, reviewed_facts={}))
    assert terminal_row.row.status == "rejected"
    events = asyncio.run(service.list_proposal_events(session_id))
    assert events[-1].payload["reason_code"] == reason_code
    assert events[-1].payload["dispatch"] is not None
    audit_messages = [message for message in asyncio.run(service.get_messages(session_id, limit=None)) if message.role == "audit"]
    assert len(audit_messages) == 1
    assert audit_messages[0].tool_calls is not None
    invocation = audit_messages[0].tool_calls[0]["invocation"]
    assert invocation["result_canonical"] is not None
    result_payload = json.loads(invocation["result_canonical"])
    assert result_payload["pipeline_content_hash_schema"] == "composer.pipeline-dispatch-result.v1"
    assert events[-1].payload["dispatch"]["result_hash"] == invocation["result_hash"]


@pytest.mark.asyncio
async def test_canonical_pipeline_cancel_then_prepare_failure_terminalizes_before_reraising(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.web.composer.pipeline_commit import PipelineCommitError

    app, service, _pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-cancel-prepare-call",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_prepare(**_kwargs: Any):
        started.set()
        await release.wait()
        raise PipelineCommitError("candidate validation failed", code="VALIDATION_FAILED")

    monkeypatch.setattr(
        "elspeth.web.sessions.routes.composer.pipeline_settlement.prepare_pipeline_proposal_commit",
        fail_prepare,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash}))
        await started.wait()
        request_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    proposals = await service.list_composition_proposals(session_id)
    assert proposals[0].status == "rejected"
    assert (await service.list_proposal_events(session_id))[-1].payload["reason_code"] == "validation_failed"


@pytest.mark.asyncio
async def test_canonical_pipeline_cancel_then_prepare_failure_preserves_binding_cleanup_failure_as_cause(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.web.composer.audit import begin_dispatch, finish_success
    from elspeth.web.composer.pipeline_commit import (
        PipelineCommitError,
        PipelineDispatchAuditBinding,
    )
    from elspeth.web.sessions.routes.composer import pipeline_settlement as proposal_routes

    app, service, pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-cancel-binding-cleanup-call",
    )
    audit = begin_dispatch(
        row.tool_call_id,
        "set_pipeline",
        pipeline,
        version_before=0,
        actor="user:alice",
    )
    invocation = finish_success(
        audit,
        result_payload={
            "success": False,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": stable_hash({"executor": "validation-failed"}),
        },
        version_after=0,
    )
    dispatch = PipelineDispatchAuditBinding.from_invocation(invocation)
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()

    async def fail_prepare(**_kwargs: Any):
        prepare_started.set()
        await prepare_release.wait()
        raise PipelineCommitError(
            "executor validation failed",
            code="VALIDATION_FAILED",
            invocation=invocation,
            dispatch=dispatch,
        )

    async def lose_binding(*_args: Any, **_kwargs: Any):
        return ()

    monkeypatch.setattr(proposal_routes, "prepare_pipeline_proposal_commit", fail_prepare)
    monkeypatch.setattr(proposal_routes, "_persist_tool_invocations", lose_binding)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash}))
        await prepare_started.wait()
        request_task.cancel()
        prepare_release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await request_task

    cleanup_failure = caught.value.__cause__
    assert isinstance(cleanup_failure, RuntimeError)
    assert "did not persist exactly one rebound binding" in str(cleanup_failure)
    assert isinstance(cleanup_failure.__cause__ or cleanup_failure.__context__, PipelineCommitError)
    assert (await service.list_composition_proposals(session_id))[0].status == "pending"


@pytest.mark.asyncio
async def test_canonical_pipeline_cancel_then_prepare_failure_preserves_rejection_cleanup_failure_as_cause(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.web.composer.pipeline_commit import PipelineCommitError
    from elspeth.web.sessions.routes.composer import pipeline_settlement as proposal_routes

    app, service, _pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-cancel-rejection-cleanup-call",
    )
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()

    async def fail_prepare(**_kwargs: Any):
        prepare_started.set()
        await prepare_release.wait()
        raise PipelineCommitError("candidate validation failed", code="VALIDATION_FAILED")

    async def fail_rejection(**_kwargs: Any):
        raise RuntimeError("rejection cleanup failed")

    monkeypatch.setattr(proposal_routes, "prepare_pipeline_proposal_commit", fail_prepare)
    monkeypatch.setattr(service, "reject_pipeline_composition_proposal", fail_rejection)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash}))
        await prepare_started.wait()
        request_task.cancel()
        prepare_release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await request_task

    cleanup_failure = caught.value.__cause__
    assert isinstance(cleanup_failure, RuntimeError)
    assert str(cleanup_failure) == "rejection cleanup failed"
    assert isinstance(cleanup_failure.__cause__ or cleanup_failure.__context__, PipelineCommitError)
    assert (await service.list_composition_proposals(session_id))[0].status == "pending"


@pytest.mark.asyncio
async def test_canonical_pipeline_cancel_during_failed_dispatch_audit_persist_terminalizes_before_reraising(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.core.canonical import stable_hash as canonical_stable_hash
    from elspeth.web.composer.audit import begin_dispatch, finish_success
    from elspeth.web.composer.pipeline_commit import (
        PipelineCommitError,
        PipelineDispatchAuditBinding,
    )
    from elspeth.web.sessions.routes.composer import pipeline_settlement as proposal_routes

    app, service, pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-cancel-audit-call",
    )
    audit = begin_dispatch(
        row.tool_call_id,
        "set_pipeline",
        pipeline,
        version_before=0,
        actor="user:alice",
    )
    recorder_invocation = finish_success(
        audit,
        result_payload={"success": False},
        version_after=0,
    )
    executor_hash = canonical_stable_hash({"executor": "validation-failed"})
    rebound_invocation = finish_success(
        audit,
        result_payload={
            "success": False,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": executor_hash,
        },
        version_after=0,
    )
    dispatch = PipelineDispatchAuditBinding.from_invocation(rebound_invocation)

    async def fail_after_dispatch(*, recorder, **_kwargs: Any):
        recorder.record(recorder_invocation)
        raise PipelineCommitError(
            "executor validation failed",
            code="VALIDATION_FAILED",
            invocation=rebound_invocation,
            dispatch=dispatch,
        )

    persist_started = asyncio.Event()
    persist_release = asyncio.Event()
    original_persist = proposal_routes._persist_tool_invocations

    async def gated_persist(*args: Any, **kwargs: Any):
        persist_started.set()
        await persist_release.wait()
        return await original_persist(*args, **kwargs)

    monkeypatch.setattr(proposal_routes, "prepare_pipeline_proposal_commit", fail_after_dispatch)
    monkeypatch.setattr(proposal_routes, "_persist_tool_invocations", gated_persist)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash}))
        await persist_started.wait()
        request_task.cancel()
        persist_release.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    proposals = await service.list_composition_proposals(session_id)
    assert proposals[0].status == "rejected"
    terminal = (await service.list_proposal_events(session_id))[-1].payload
    assert terminal["reason_code"] == "validation_failed"
    audit_messages = [message for message in await service.get_messages(session_id, limit=None) if message.role == "audit"]
    assert len(audit_messages) == 1
    assert audit_messages[0].tool_calls is not None
    persisted_envelope = audit_messages[0].tool_calls[0]
    persisted_invocation = persisted_envelope["invocation"]
    assert terminal["dispatch"] == {
        "tool_call_id": persisted_invocation["tool_call_id"],
        "tool_name": persisted_invocation["tool_name"],
        "status": persisted_invocation["status"],
        "arguments_hash": persisted_invocation["arguments_hash"],
        "result_hash": persisted_invocation["result_hash"],
    }
    assert persisted_invocation["result_hash"] == dispatch.result_hash


@pytest.mark.asyncio
async def test_canonical_pipeline_cancel_then_settlement_failure_preserves_cancellation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, service, _pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-cancel-settlement-call",
    )
    settle_started = asyncio.Event()
    settle_release = asyncio.Event()

    async def fail_settlement(**_kwargs: Any):
        settle_started.set()
        await settle_release.wait()
        raise RuntimeError("settlement failed")

    monkeypatch.setattr(service, "settle_pipeline_composition_proposal", fail_settlement)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        request_task = asyncio.create_task(client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash}))
        await settle_started.wait()
        request_task.cancel()
        settle_release.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    proposals = await service.list_composition_proposals(session_id)
    assert proposals[0].status == "pending"
    assert await service.get_current_state(session_id) is None
    audit_messages = [message for message in await service.get_messages(session_id, limit=None) if message.role == "audit"]
    assert len(audit_messages) == 1


def test_canonical_pipeline_accept_requires_and_echoes_draft_hash(tmp_path, monkeypatch) -> None:
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

    app, service = _make_app(tmp_path)
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    input_path = tmp_path / "blobs" / "canonical.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("value\n1\n", encoding="utf-8")
    pipeline = {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(input_path), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "rows",
                "plugin": "json",
                "options": {
                    "path": str(tmp_path / "outputs" / "canonical.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
    }
    proposal_envelope = PipelineProposal.create(
        pipeline=pipeline,
        base=AbsentBase(),
        reviewed_facts={},
        surface=PlannerSurface.FREEFORM,
        repair_count=0,
        skill_hash=stable_hash("planner-skill"),
    )
    plan = PipelinePlanResult(
        proposal=proposal_envelope,
        tool_call_id="canonical-terminal-call",
        custody_result="not_required",
        model_identifier="planner-model",
        model_version="planner-model-v1",
        provider="test",
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Canonical accept"}).json()
    session_id = uuid.UUID(session["id"])
    row = asyncio.run(
        service.create_pipeline_composition_proposal(
            session_id=session_id,
            plan=plan,
            summary="Replace the pipeline.",
            rationale="Requested by the operator.",
            affects=("graph", "validation"),
            arguments_redacted_json=redact_tool_call_arguments(
                "set_pipeline",
                pipeline,
                telemetry=NoopRedactionTelemetry(),
            ),
            actor="composer-web:user:alice",
            composer_model_identifier="planner-model",
            composer_model_version="planner-model-v1",
            composer_provider="provider",
        )
    )
    endpoint = f"/api/sessions/{session['id']}/proposals/{row.id}/accept"

    listed = client.get(f"/api/sessions/{session['id']}/proposals").json()
    assert listed[0]["pipeline_metadata"]["draft_hash"] == proposal_envelope.draft_hash
    assert client.post(endpoint).status_code == 422
    assert client.post(endpoint, json={"draft_hash": "0" * 64}).status_code == 409

    settle = service.settle_pipeline_composition_proposal

    async def interrupt_before_settlement(**kwargs: Any):
        del kwargs
        raise RuntimeError("interrupted before atomic settlement")

    monkeypatch.setattr(service, "settle_pipeline_composition_proposal", interrupt_before_settlement)
    with pytest.raises(RuntimeError, match="interrupted before atomic settlement"):
        client.post(endpoint, json={"draft_hash": proposal_envelope.draft_hash})
    assert asyncio.run(service.get_current_state(session_id)) is None
    audit_rows_before_retry = [
        message for message in asyncio.run(service.get_messages(session_id, limit=None)) if message.role == "audit" and message.tool_calls
    ]
    assert len(audit_rows_before_retry) == 1
    monkeypatch.setattr(service, "settle_pipeline_composition_proposal", settle)
    accepted = client.post(endpoint, json={"draft_hash": proposal_envelope.draft_hash})

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["status"] == "committed"
    assert accepted.json()["pipeline_metadata"]["draft_hash"] == proposal_envelope.draft_hash
    committed_state = asyncio.run(service.get_current_state(session_id))
    assert committed_state is not None

    retried = client.post(endpoint, json={"draft_hash": proposal_envelope.draft_hash})

    assert retried.status_code == 200
    assert retried.json()["committed_state_id"] == str(committed_state.id)
    current_after_retry = asyncio.run(service.get_current_state(session_id))
    assert current_after_retry is not None
    assert current_after_retry.id == committed_state.id
    audit_rows_after_retry = [
        message for message in asyncio.run(service.get_messages(session_id, limit=None)) if message.role == "audit" and message.tool_calls
    ]
    assert len(audit_rows_after_retry) == 1


@pytest.mark.asyncio
async def test_canonical_pipeline_recovery_rejects_tampered_bound_content_hash(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select, update

    from elspeth.core.canonical import canonical_json
    from elspeth.web.sessions.models import chat_messages_table

    app, service, _pipeline, session_id, row, endpoint = await _create_canonical_pipeline_route_proposal(
        tmp_path,
        monkeypatch,
        tool_call_id="canonical-bound-tamper-call",
    )
    settle = service.settle_pipeline_composition_proposal

    async def interrupt_before_settlement(**kwargs: Any):
        del kwargs
        raise RuntimeError("interrupted before atomic settlement")

    monkeypatch.setattr(service, "settle_pipeline_composition_proposal", interrupt_before_settlement)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with pytest.raises(RuntimeError, match="interrupted before atomic settlement"):
            await client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash})

        with service._engine.begin() as conn:
            audit_row = conn.execute(select(chat_messages_table).where(chat_messages_table.c.role == "audit")).one()
            envelopes = list(audit_row.tool_calls)
            invocation = envelopes[0]["invocation"]
            result_payload = json.loads(invocation["result_canonical"])
            result_payload["pipeline_content_hash"] = "0" * 64
            invocation["result_canonical"] = canonical_json(result_payload)
            invocation["result_hash"] = stable_hash(result_payload)
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == audit_row.id).values(tool_calls=envelopes))

        monkeypatch.setattr(service, "settle_pipeline_composition_proposal", settle)
        rejected = await client.post(endpoint, json={"draft_hash": row.pipeline_metadata.draft_hash})

    assert rejected.status_code == 422
    assert "candidate/executor content mismatch" in rejected.text
    assert await service.get_current_state(session_id) is None
    assert (await service.list_composition_proposals(session_id))[0].status == "rejected"


@pytest.mark.parametrize("surface_name", ["GUIDED_STAGED", "TUTORIAL_PROFILE"])
def test_generic_accept_rejects_guided_pipeline_surfaces_before_dispatch(tmp_path, monkeypatch, surface_name) -> None:
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

    app, service = _make_app(tmp_path)
    pipeline = {"sources": {}, "nodes": [], "edges": [], "outputs": []}
    proposal_envelope = PipelineProposal.create(
        pipeline=pipeline,
        base=AbsentBase(),
        reviewed_facts={"checkpoint": "server-owned"},
        surface=getattr(PlannerSurface, surface_name),
        repair_count=0,
        skill_hash=stable_hash("planner-skill"),
    )
    plan = PipelinePlanResult(
        proposal=proposal_envelope,
        tool_call_id=f"{surface_name.lower()}-call",
        custody_result="not_required",
        model_identifier="planner-model",
        model_version="planner-model-v1",
        provider="test",
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Guided pipeline"}).json()
    session_id = uuid.UUID(session["id"])
    row = asyncio.run(
        service.create_pipeline_composition_proposal(
            session_id=session_id,
            plan=plan,
            summary="Replace the pipeline.",
            rationale="Requested by the guided workflow.",
            affects=("graph",),
            arguments_redacted_json=redact_tool_call_arguments(
                "set_pipeline",
                pipeline,
                telemetry=NoopRedactionTelemetry(),
            ),
            actor="composer-web:user:alice",
            composer_model_identifier="planner-model",
            composer_model_version="planner-model-v1",
            composer_provider="provider",
        )
    )
    prepare = AsyncMock(side_effect=AssertionError("generic route dispatched guided proposal"))
    monkeypatch.setattr(
        "elspeth.web.sessions.routes.composer.pipeline_settlement.prepare_pipeline_proposal_commit",
        prepare,
    )

    response = client.post(
        f"/api/sessions/{session['id']}/proposals/{row.id}/accept",
        json={"draft_hash": proposal_envelope.draft_hash},
    )

    assert response.status_code == 409
    prepare.assert_not_awaited()
    assert asyncio.run(service.get_current_state(session_id)) is None
    assert asyncio.run(service.get_messages(session_id, limit=None)) == []
    assert (asyncio.run(service.list_composition_proposals(session_id)))[0].status == "pending"

    rejected = client.post(
        f"/api/sessions/{session['id']}/proposals/{row.id}/reject",
        json={},
    )

    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "This pipeline proposal must be rejected through its guided workflow."
    events = asyncio.run(service.list_proposal_events(session_id))
    assert [event.event_type for event in events] == ["proposal.created"]
    assert (asyncio.run(service.list_composition_proposals(session_id)))[0].status == "pending"


def test_malformed_canonical_creation_event_fails_closed_without_legacy_fallback(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select, update

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
    from elspeth.web.sessions.models import proposal_events_table

    app, service = _make_app(tmp_path)
    pipeline = {"sources": {}, "nodes": [], "edges": [], "outputs": []}
    proposal_envelope = PipelineProposal.create(
        pipeline=pipeline,
        base=AbsentBase(),
        reviewed_facts={},
        surface=PlannerSurface.FREEFORM,
        repair_count=0,
        skill_hash=stable_hash("planner-skill"),
    )
    plan = PipelinePlanResult(
        proposal=proposal_envelope,
        tool_call_id="malformed-canonical-call",
        custody_result="not_required",
        model_identifier="planner-model",
        model_version="planner-model-v1",
        provider="test",
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Malformed canonical"}).json()
    session_id = uuid.UUID(session["id"])
    row = asyncio.run(
        service.create_pipeline_composition_proposal(
            session_id=session_id,
            plan=plan,
            summary="Replace the pipeline.",
            rationale="Requested by the operator.",
            affects=("graph",),
            arguments_redacted_json=redact_tool_call_arguments(
                "set_pipeline",
                pipeline,
                telemetry=NoopRedactionTelemetry(),
            ),
            actor="composer-web:user:alice",
            composer_model_identifier="planner-model",
            composer_model_version="planner-model-v1",
            composer_provider="provider",
        )
    )
    with service._engine.begin() as conn:
        event = conn.execute(select(proposal_events_table).where(proposal_events_table.c.proposal_id == str(row.id))).one()
        malformed = ["schema", "pipeline_proposal_created.v1"]
        conn.execute(update(proposal_events_table).where(proposal_events_table.c.id == event.id).values(payload=malformed))
    legacy_execute = MagicMock(side_effect=AssertionError("malformed canonical proposal used legacy replay"))
    monkeypatch.setattr("elspeth.web.sessions.routes.composer.proposals.execute_tool", legacy_execute)

    with pytest.raises(AuditIntegrityError):
        client.post(
            f"/api/sessions/{session['id']}/proposals/{row.id}/accept",
            json={"draft_hash": proposal_envelope.draft_hash},
        )

    legacy_execute.assert_not_called()
    assert asyncio.run(service.get_current_state(session_id)) is None


def test_accept_proposal_threads_originating_message_id_to_inline_blob(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    from elspeth.web.catalog.schemas import PluginSchemaInfo
    from elspeth.web.interpretation_state import SOURCE_AUTHORING_KEY
    from elspeth.web.sessions.models import blobs_table

    app, service = _make_app(tmp_path)
    app.state.session_engine = service._engine
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    app.state.catalog_service = catalog
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Accept inline blob"}).json()
    session_id = uuid.UUID(session["id"])
    user_message = asyncio.run(
        service.add_message(
            session_id,
            "user",
            "Build a generated CSV pipeline with one Ada score row.",
            writer_principal="route_user_message",
        )
    )
    arguments = {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"schema": {"mode": "observed"}},
            "inline_blob": {
                "filename": "ada.csv",
                "mime_type": "text/csv",
                "content": "name,score\nada,42\n",
            },
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
        "metadata": {"name": "accepted-inline-blob-proposal"},
    }
    arguments_hash = stable_hash(arguments)
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline_inline_blob",
            tool_name="set_pipeline",
            summary="Replace the pipeline with inline CSV.",
            rationale="Requested by the current composer turn.",
            affects=("graph", "blob"),
            arguments_json=arguments,
            arguments_redacted_json={"summary": "redacted"},
            base_state_id=None,
            actor="composer-web:user:alice",
            user_message_id=user_message.id,
            composer_model_identifier="openai/gpt-5-mini",
            composer_model_version="gpt-5-mini-2026-05-01",
            composer_provider="openai",
            composer_skill_hash="a" * 64,
            tool_arguments_hash=arguments_hash,
        )
    )

    response = client.post(f"/api/sessions/{session['id']}/proposals/{proposal.id}/accept")

    assert response.status_code == 200
    with service._engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session["id"])).one()
    assert row.created_from_message_id == str(user_message.id)
    assert row.creation_modality == CreationModality.LLM_GENERATED.value
    assert row.creating_model_identifier == "openai/gpt-5-mini"
    assert row.creating_model_version == "gpt-5-mini-2026-05-01"
    assert row.creating_provider == "openai"
    assert row.creating_composer_skill_hash == "a" * 64
    assert row.creating_arguments_hash == arguments_hash

    persisted = asyncio.run(service.get_current_state(session_id))
    assert persisted is not None
    assert "source" in persisted.sources
    source_authoring = persisted.sources["source"]["options"][SOURCE_AUTHORING_KEY]
    assert source_authoring == {
        "modality": CreationModality.LLM_GENERATED.value,
        "content_hash": row.content_hash,
        "review_event_id": None,
        "resolved_kind": None,
    }


def test_accept_inline_blob_proposal_without_composer_provenance_fails_closed(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    from elspeth.web.catalog.schemas import PluginSchemaInfo
    from elspeth.web.sessions.models import blobs_table

    app, service = _make_app(tmp_path)
    app.state.session_engine = service._engine
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    app.state.catalog_service = catalog
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Legacy inline blob proposal"}).json()
    session_id = uuid.UUID(session["id"])
    user_message = asyncio.run(
        service.add_message(
            session_id,
            "user",
            "Build a generated CSV pipeline with one Ada score row.",
            writer_principal="route_user_message",
        )
    )
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline_inline_blob_legacy",
            tool_name="set_pipeline",
            summary="Replace the pipeline with inline CSV.",
            rationale="Legacy proposal missing composer provenance.",
            affects=("graph", "blob"),
            arguments_json={
                "source": {
                    "plugin": "csv",
                    "on_success": "rows",
                    "options": {"schema": {"mode": "observed"}},
                    "inline_blob": {
                        "filename": "ada.csv",
                        "mime_type": "text/csv",
                        "content": "name,score\nada,42\n",
                    },
                },
                "nodes": [],
                "edges": [],
                "outputs": [],
                "metadata": {"name": "legacy-inline-blob-proposal"},
            },
            arguments_redacted_json={"summary": "redacted"},
            base_state_id=None,
            actor="composer-web:user:alice",
            user_message_id=user_message.id,
        )
    )

    response = client.post(f"/api/sessions/{session['id']}/proposals/{proposal.id}/accept")

    assert response.status_code == 409
    assert "missing composer provenance" in response.json()["detail"]
    assert asyncio.run(service.get_current_state(session_id)) is None
    with service._engine.begin() as conn:
        blob_count = conn.execute(select(blobs_table.c.id).where(blobs_table.c.session_id == session["id"])).fetchall()
    assert blob_count == []


def test_accept_empty_inline_blob_proposal_without_composer_provenance_fails_closed(tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    from elspeth.web.catalog.schemas import PluginSchemaInfo
    from elspeth.web.sessions.models import blobs_table

    app, service = _make_app(tmp_path)
    app.state.session_engine = service._engine
    catalog = MagicMock(spec=CatalogService)
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="text",
        plugin_type="source",
        description="Text source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    app.state.catalog_service = catalog
    monkeypatch.setattr(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        _async_return(ValidationResult(is_valid=True, checks=[], errors=[])),
    )
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Legacy empty inline blob proposal"}).json()
    session_id = uuid.UUID(session["id"])
    user_message = asyncio.run(
        service.add_message(
            session_id,
            "user",
            "Create an empty text source.",
            writer_principal="route_user_message",
        )
    )
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline_empty_inline_blob_legacy",
            tool_name="set_pipeline",
            summary="Replace the pipeline with an empty inline source.",
            rationale="Legacy proposal missing composer provenance.",
            affects=("graph", "blob"),
            arguments_json={
                "source": {
                    "plugin": "text",
                    "on_success": "rows",
                    "options": {
                        "column": "text",
                        "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                    },
                    "inline_blob": {
                        "filename": "empty.txt",
                        "mime_type": "text/plain",
                        "content": "",
                    },
                },
                "nodes": [],
                "edges": [],
                "outputs": [],
                "metadata": {"name": "legacy-empty-inline-blob-proposal"},
            },
            arguments_redacted_json={"summary": "redacted"},
            base_state_id=None,
            actor="composer-web:user:alice",
            user_message_id=user_message.id,
        )
    )

    response = client.post(f"/api/sessions/{session['id']}/proposals/{proposal.id}/accept")

    assert response.status_code == 409
    assert "missing composer provenance" in response.json()["detail"]
    assert asyncio.run(service.get_current_state(session_id)) is None
    with service._engine.begin() as conn:
        blob_count = conn.execute(select(blobs_table.c.id).where(blobs_table.c.session_id == session["id"])).fetchall()
    assert blob_count == []


def _insert_discard_audit_records(settings: WebSettings, run_id: str) -> None:
    """Create audit records that route three rows to the virtual discard sink."""
    (settings.data_dir / "runs").mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    with (
        LandscapeDB.from_url(settings.get_landscape_url()) as db,
        db.connection() as conn,
    ):
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                completed_at=now,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="test",
                status="completed",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        conn.execute(
            nodes_table.insert(),
            [
                {
                    "node_id": "source",
                    "run_id": run_id,
                    "plugin_name": "csv",
                    "node_type": "source",
                    "plugin_version": "test",
                    "determinism": "deterministic",
                    "config_hash": "source-cfg",
                    "config_json": "{}",
                    "registered_at": now,
                },
                {
                    "node_id": "transform",
                    "run_id": run_id,
                    "plugin_name": "mapper",
                    "node_type": "transform",
                    "plugin_version": "test",
                    "determinism": "deterministic",
                    "config_hash": "transform-cfg",
                    "config_json": "{}",
                    "registered_at": now,
                },
            ],
        )
        conn.execute(
            rows_table.insert(),
            [
                {
                    "row_id": "row-transform",
                    "run_id": run_id,
                    "source_node_id": "source",
                    "row_index": 0,
                    "source_row_index": 0,
                    "ingest_sequence": 0,
                    "source_data_hash": "hash-transform",
                    "created_at": now,
                },
                {
                    "row_id": "row-sink",
                    "run_id": run_id,
                    "source_node_id": "source",
                    "row_index": 1,
                    "source_row_index": 1,
                    "ingest_sequence": 1,
                    "source_data_hash": "hash-sink",
                    "created_at": now,
                },
            ],
        )
        conn.execute(
            tokens_table.insert(),
            [
                {
                    "token_id": "token-transform",
                    "row_id": "row-transform",
                    "run_id": run_id,
                    "created_at": now,
                },
                {
                    "token_id": "token-sink",
                    "row_id": "row-sink",
                    "run_id": run_id,
                    "created_at": now,
                },
            ],
        )
        conn.execute(
            validation_errors_table.insert().values(
                error_id="verr_discard",
                run_id=run_id,
                node_id="source",
                row_hash="hash-validation",
                row_data_json="{}",
                error="invalid row",
                schema_mode="fixed",
                destination="discard",
                created_at=now,
            )
        )
        conn.execute(
            transform_errors_table.insert().values(
                error_id="terr_discard",
                run_id=run_id,
                token_id="token-transform",
                transform_id="transform",
                row_hash="hash-transform",
                row_data_json="{}",
                error_details_json='{"reason":"validation_failed"}',
                destination="discard",
                created_at=now,
            )
        )
        conn.execute(
            token_outcomes_table.insert(),
            [
                {
                    "outcome_id": "tout_transform_error",
                    "run_id": run_id,
                    "token_id": "token-transform",
                    "outcome": TerminalOutcome.FAILURE,
                    "path": TerminalPath.ON_ERROR_ROUTED,
                    "completed": 1,
                    "recorded_at": now,
                    "sink_name": None,
                },
                {
                    "outcome_id": "tout_discard",
                    "run_id": run_id,
                    "token_id": "token-sink",
                    "outcome": TerminalOutcome.FAILURE,
                    "path": TerminalPath.SINK_DISCARDED,
                    "completed": 1,
                    "recorded_at": now,
                    "sink_name": "__discard__",
                },
            ],
        )


def _fanout_accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(
            emitted=9324,
            terminal=9324,
            succeeded=9323,
            failed=0,
            structural=1,
            pending=0,
        ),
        routing=RunAccountingRouting(
            routed_success=0,
            routed_failure=0,
            quarantined=0,
            discarded=0,
        ),
        integrity=RunAccountingIntegrity(
            closure="closed",
            missing_terminal_outcomes=0,
            duplicate_terminal_outcomes=0,
        ),
    )


def _open_completed_accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(
            emitted=2,
            terminal=1,
            succeeded=1,
            failed=0,
            structural=0,
            pending=1,
        ),
        routing=RunAccountingRouting(
            routed_success=0,
            routed_failure=0,
            quarantined=0,
            discarded=0,
        ),
        integrity=RunAccountingIntegrity(
            closure="open",
            missing_terminal_outcomes=1,
            duplicate_terminal_outcomes=0,
        ),
    )


class TestSessionCRUDRoutes:
    """Tests for session create, list, get, delete endpoints."""

    def test_create_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post(
            "/api/sessions",
            json={"title": "My Pipeline"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "My Pipeline"
        assert body["user_id"] == "alice"
        assert "id" in body

    def test_create_session_default_title(self, tmp_path) -> None:
        from elspeth.web.sessions.titles import is_default_session_title

        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post("/api/sessions", json={})
        assert response.status_code == 201
        title = response.json()["title"]
        # Server-side minted default: "Session — <date>" (elspeth-ef8c18a6cb).
        assert title.startswith("Session — ")
        assert is_default_session_title(title)

    def test_create_session_default_titles_auto_disambiguate(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        first = client.post("/api/sessions", json={}).json()["title"]
        second = client.post("/api/sessions", json={}).json()["title"]
        third = client.post("/api/sessions", json={}).json()["title"]
        assert second == f"{first} (2)"
        assert third == f"{first} (3)"

    def test_create_session_null_title_mints_default(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post("/api/sessions", json={"title": None})
        assert response.status_code == 201
        assert response.json()["title"].startswith("Session — ")

    def test_list_sessions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        client.post("/api/sessions", json={"title": "S1"})
        client.post("/api/sessions", json={"title": "S2"})

        response = client.get("/api/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

    def test_get_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Test"},
        )
        session_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == session_id

    def test_update_session_title(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Original title"},
        )
        session_id = create_resp.json()["id"]

        update_resp = client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "Renamed pipeline"},
        )

        assert update_resp.status_code == 200
        assert update_resp.json()["title"] == "Renamed pipeline"
        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.json()["title"] == "Renamed pipeline"

    def test_update_session_title_rejects_blank_title(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Original title"},
        )
        session_id = create_resp.json()["id"]

        update_resp = client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "   "},
        )

        assert update_resp.status_code == 422
        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.json()["title"] == "Original title"

    def test_get_session_not_found(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.get(f"/api/sessions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_delete_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions",
            json={"title": "To Delete"},
        )
        session_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 204

        # Verify cleanup_session_lock was called with the correct session ID
        app.state.execution_service.cleanup_session_lock.assert_called_once_with(session_id)

        # Verify it's gone
        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_blocked_by_active_run(self, tmp_path) -> None:
        """Deleting a session with a pending/running run returns 409.

        Without this guard, archive_session() deletes run rows and blob
        directories out from under the background pipeline worker.
        """
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post("/api/sessions", json={"title": "Active Run"})
        session_id = uuid.UUID(create_resp.json()["id"])

        # Create a pending run via the service layer
        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.create_run(session_id, state.id)

        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 409
        assert "active" in del_resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_delete_session_serializes_active_run_check_with_execution_lock(self, tmp_path) -> None:
        """Delete must share execute()'s per-session lock across check+archive.

        Holding the execution lock simulates a concurrent execute() already in
        the active-run check/create-run critical section. The delete route must
        wait before it even calls get_active_run(); otherwise a run can be
        created between delete's check and archive.
        """
        app, service = _make_app(tmp_path)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            create_resp = await client.post("/api/sessions", json={"title": "Delete Race"})
            assert create_resp.status_code == 201
            session_id = create_resp.json()["id"]

            execution_lock = app.state.execution_service.get_session_lock(session_id)
            await execution_lock.acquire()

            ownership_checked = asyncio.Event()
            original_get_session = service.get_session
            active_run_checked = asyncio.Event()
            original_get_active_run = service.get_active_run

            async def _get_session_spy(session_id_arg: uuid.UUID) -> Any:
                result = await original_get_session(session_id_arg)
                ownership_checked.set()
                return result

            async def _get_active_run_spy(session_id_arg: uuid.UUID) -> Any:
                active_run_checked.set()
                return await original_get_active_run(session_id_arg)

            service.get_session = _get_session_spy  # type: ignore[method-assign]
            service.get_active_run = _get_active_run_spy  # type: ignore[method-assign]

            delete_task = asyncio.create_task(client.delete(f"/api/sessions/{session_id}"))
            await asyncio.wait_for(ownership_checked.wait(), timeout=2.0)
            await asyncio.sleep(0)

            assert not active_run_checked.is_set()
            assert not delete_task.done()

            execution_lock.release()
            del_resp = await delete_task

        assert del_resp.status_code == 204
        assert active_run_checked.is_set()

    @pytest.mark.asyncio
    async def test_delete_session_allowed_after_run_completes(self, tmp_path) -> None:
        """After a run reaches a terminal state, deletion is allowed."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post("/api/sessions", json={"title": "Completed Run"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-delete-allowed")

        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 204

    @pytest.mark.asyncio
    async def test_list_session_runs_includes_failed_run_error(self, tmp_path) -> None:
        """Failed run cards need the stored terminal error, not just status."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post("/api/sessions", json={"title": "Failed Run"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "failed",
            error="Pipeline execution failed (FrameworkBugError)",
            rows_processed=1,
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 200
        runs = runs_resp.json()
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        assert runs[0]["error"] == "Pipeline execution failed (FrameworkBugError)"

    @pytest.mark.asyncio
    async def test_session_run_list_returns_accounting_for_fanout_run(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post("/api/sessions", json={"title": "Fanout Run"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed",
            landscape_run_id=str(run.id),
            rows_processed=1,
            rows_succeeded=9323,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        monkeypatch.setattr(
            "elspeth.web.sessions.routes.runs.load_run_accounting_for_settings",
            lambda settings, run_ids: {str(run.id): _fanout_accounting()},
            raising=False,
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 200
        runs = runs_resp.json()
        assert len(runs) == 1
        payload = runs[0]
        assert "rows_processed" not in payload
        assert "rows_failed" not in payload
        assert payload["accounting"]["source"]["rows_processed"] == 1
        assert payload["accounting"]["tokens"]["succeeded"] == 9323

    @pytest.mark.asyncio
    async def test_session_run_list_fails_closed_when_completed_accounting_missing(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        create_resp = client.post("/api/sessions", json={"title": "Missing Accounting"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed",
            landscape_run_id=str(run.id),
            rows_processed=1,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        monkeypatch.setattr(
            "elspeth.web.sessions.routes.runs.load_run_accounting_for_settings",
            lambda settings, run_ids: {},
            raising=False,
        )
        monkeypatch.setattr(
            "elspeth.web.execution.discard_summary.load_discard_summaries_for_settings",
            lambda settings, run_ids: {},
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 500
        assert runs_resp.json()["detail"]["code"] == "run_integrity_error"

    @pytest.mark.asyncio
    async def test_session_run_list_fails_closed_when_completed_accounting_is_open(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        create_resp = client.post("/api/sessions", json={"title": "Open Accounting"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed",
            landscape_run_id=str(run.id),
            rows_processed=1,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        monkeypatch.setattr(
            "elspeth.web.sessions.routes.runs.load_run_accounting_for_settings",
            lambda settings, run_ids: {str(run.id): _open_completed_accounting()},
            raising=False,
        )
        monkeypatch.setattr(
            "elspeth.web.execution.discard_summary.load_discard_summaries_for_settings",
            lambda settings, run_ids: {},
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 500
        body = runs_resp.json()
        assert body["detail"]["code"] == "run_integrity_error"
        assert "requires closed token accounting" in str(body["detail"]["validation_errors"])

    @pytest.mark.asyncio
    async def test_list_session_runs_includes_virtual_discard_summary(self, tmp_path) -> None:
        """Run cards must show rows routed to the virtual discard sink."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post("/api/sessions", json={"title": "Discarded Rows"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        landscape_run_id = "lscape-discard-summary"
        _insert_discard_audit_records(app.state.settings, landscape_run_id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "failed",
            landscape_run_id=landscape_run_id,
            error="No row reached a success path.",
            rows_processed=2,
            rows_succeeded=0,
            rows_failed=2,
            rows_routed_success=0,
            rows_routed_failure=1,
            rows_quarantined=0,
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 200
        runs = runs_resp.json()
        assert runs[0]["discard_summary"] == {
            "total": 3,
            "validation_errors": 1,
            "transform_errors": 1,
            "sink_discards": 1,
            "stages": [
                {
                    "stage": "source_validation",
                    "node_id": "source",
                    "count": 1,
                },
                {
                    "stage": "transform_validation",
                    "node_id": "transform",
                    "count": 1,
                },
                {
                    "stage": "sink_discard",
                    "node_id": None,
                    "count": 1,
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_list_session_runs_skips_discard_summary_for_running_run(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running run cards must not inspect an audit DB that may still be initializing."""
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        create_resp = client.post("/api/sessions", json={"title": "Running Run"})
        session_id = uuid.UUID(create_resp.json()["id"])

        state = await service.save_composition_state(session_id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session_id, state.id)
        await service.update_run_status(run.id, "running", landscape_run_id="lscape-running")

        def fail_if_called(*args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("discard summary lookup should not run for non-terminal runs")

        monkeypatch.setattr(
            "elspeth.web.execution.discard_summary.load_discard_summaries_for_settings",
            fail_if_called,
        )

        runs_resp = client.get(f"/api/sessions/{session_id}/runs")

        assert runs_resp.status_code == 200
        runs = runs_resp.json()
        assert len(runs) == 1
        assert runs[0]["status"] == "running"
        assert runs[0]["discard_summary"] is None


def _collect_ownership_call_site_identities(module: ModuleType, helper_name: str) -> set[str]:
    """Walk ``module``'s AST and return enclosing function names for each call to ``helper_name``.

    Shared implementation for the IDOR drift guards across the three
    session-scoped routers (sessions/, execution/, blobs/).  Each
    router has its own ownership-check helper — the drift guard is
    parametrized per (module, helper) pair so the failure message
    points at one specific inventory that drifted, rather than
    reporting an aggregate mismatch across all three.

    Why a shared walker (not a per-module assertion) — the AST-walking
    logic is identical across routers; duplicating it per drift test
    would itself be a drift surface (a future fix to one copy would
    silently leave the others subtly different).
    """
    import ast

    if module.__file__ is None:
        raise AssertionError(f"Module {module.__name__!r} has no source file")
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Build a child->parent map so each call can be attributed to
    # its SMALLEST enclosing function def.  ``ast.walk`` alone would
    # also match the outer factory (``create_session_router`` etc.)
    # — which contains every nested endpoint — bloating the set with
    # a non-endpoint name.  Walking upward from the call to the
    # nearest FunctionDef / AsyncFunctionDef is the only correct way
    # to attribute ownership to the handler that actually contains it.
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    identities: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == helper_name):
            continue
        # Walk up to the nearest function def.  A call outside any
        # function (module level) is a structural anomaly we want the
        # assertion to surface, not silently absorb — so we only
        # record names of function-scoped calls, and let the set
        # comparison below flag any missing endpoint.
        current: ast.AST | None = parents.get(node)
        while current is not None and not isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
            current = parents.get(current)
        if current is not None:
            identities.add(current.name)

    # The helper's own def may contain a self-reference if a future
    # refactor introduces recursion/delegation; drop it explicitly so
    # the drift guard stays focused on ROUTE HANDLERS, not plumbing.
    identities.discard(helper_name)
    return identities


class TestIDORCoverageDrift:
    """Drift guard: every session-scoped endpoint across all four routers must invoke an ownership check.

    ``TestIDORProtection.test_idor_session_crud`` walks one
    cross-session request for each session-scoped endpoint.  The risk
    is that someone adds a new route that calls an ownership-check
    helper but forgets to add a matching IDOR assertion — the
    ownership primitive is in place, but its coverage in this suite
    silently rots.

    Session-scoped endpoints live in four routers, each with its
    own ownership-check helper:

    * ``sessions/routes.py`` via ``_verify_session_ownership`` — the
      chat-and-state endpoints (``GET``/``DELETE``/``POST`` under
      ``/api/sessions/{id}``).  This router still hosts a file-local
      helper; the shared extraction landed only for the routers that
      were rewritten in Task 5.
    * ``execution/routes.py`` via ``verify_session_ownership`` —
      ``/validate`` and ``/execute``.  Calls the shared helper
      extracted to ``elspeth.web.sessions.ownership``.  Also hosts
      run-scoped endpoints which use ``_verify_run_ownership`` (not a
      session-ownership helper, but the same drift risk for run
      identities; covered by a separate inventory).
    * ``blobs/routes.py`` via ``_verify_session_and_get_blob_service``
      — blob upload/list/metadata/download/delete.  This helper is
      dual-role (checks ownership AND returns the service); both
      branches of its callers depend on the ownership check for
      IDOR safety.
    * ``audit_readiness/routes.py`` via ``verify_session_ownership``
      — snapshot and explain endpoints under
      ``/api/sessions/{id}/audit-readiness``.  Uses the shared helper
      from ``elspeth.web.sessions.ownership``.

    Each (module, helper, inventory) tuple is pinned independently so
    a failure message names exactly which router's audit drifted.  A
    pure count across all four would satisfy ``{add endpoint X in
    router A, drop endpoint Y in router B}`` — the count stays
    constant while the audit silently swaps a covered endpoint for an
    uncovered one in a different router.
    """

    EXPECTED_SESSIONS_OWNERSHIP_ENDPOINTS: frozenset[str] = frozenset(
        {
            "get_session",
            "update_session",
            "delete_session",
            "get_messages",
            "send_message",
            "recompose",
            "get_composer_preferences",
            "update_composer_preferences",
            "get_composer_progress",
            "list_composition_proposals",
            "list_proposal_events",
            "accept_composition_proposal",
            "reject_composition_proposal",
            "list_session_runs",
            "get_current_state",
            "get_state_versions",
            "revert_state",
            "get_state_yaml",
            "import_state_yaml",
            "fork_from_message",
            "get_guided",
            "post_guided_start",
            "post_guided_reenter",
            "post_guided_respond",
            "post_guided_chat",
            # Freeform→guided convert endpoint (``POST /api/sessions/
            # {session_id}/guided/convert``, elspeth-e2c3dba6b5). Gates on
            # ``_verify_session_ownership`` like every other session-scoped
            # guided route; the inventory entry was missed when the endpoint
            # landed and caught by this drift-guard during the 0.7.1 review
            # sweep.
            "post_guided_convert",
            # 0.7.0 synthetic-scrape tutorial redesign added the
            # ``GET /api/sessions/{session_id}/guided/tutorial-sample``
            # endpoint (runtime-derived sample-page URLs + SSRF host-class
            # for an active tutorial session). Like every other
            # session-scoped guided route it gates on
            # ``_verify_session_ownership`` as its first line, so it joins
            # this inventory and the cross-session walk in
            # ``test_idor_session_crud``.
            "get_guided_tutorial_sample",
            # Phase 5b Task 6 / Task 7: interpretation event HTTP surface
            # (resolve / list) and opt-out endpoints, added in
            # ``sessions/routes.py`` and gated through
            # ``verify_session_ownership`` like every other session-scoped
            # route. Inventory updated alongside this drift-guard so the
            # IDOR audit reflects the production handler set.
            "resolve_interpretation",
            "list_interpretations",
            "opt_out_of_interpretations",
            "opt_out_summary",
            # Phase 4 hello-world tutorial (commit ca9bc05bd) added the
            # audit-story endpoint at ``GET /api/sessions/{session_id}
            # /runs/{run_id}/audit-story``. Like every other session-scoped
            # route it routes through ``_verify_session_ownership``; this
            # inventory entry was missed at the original PR open and is
            # caught now as part of the post-merge residual closeout.
            "get_run_audit_story",
            # Playwright E2E state-seed endpoint (``POST /{session_id}/state/
            # e2e-seed``, hidden from the schema and gated behind
            # ``e2e_state_seed_enabled``). The feature-flag 404 fires first,
            # but for deployments that enable the flag the endpoint is a
            # full state-write, so it gates on ``_verify_session_ownership``
            # like every other session-scoped route. Cross-session walk
            # coverage lives in ``test_idor_session_crud`` (which enables
            # the flag so the ownership gate — not the flag gate — is what
            # returns the 404).
            "seed_state_for_e2e",
        }
    )

    EXPECTED_EXECUTION_SESSION_OWNERSHIP_ENDPOINTS: frozenset[str] = frozenset(
        {
            "validate_session_pipeline",
            "execute_pipeline",
        }
    )

    EXPECTED_EXECUTION_RUN_OWNERSHIP_ENDPOINTS: frozenset[str] = frozenset(
        {
            "get_run_status",
            "get_run_diagnostics",
            "evaluate_run_diagnostics",
            "get_run_outputs",
            "get_run_output_content",
            "get_run_output_preview",
            "create_run_websocket_ticket",
            "cancel_run",
            "get_run_results",
        }
    )

    EXPECTED_BLOBS_OWNERSHIP_ENDPOINTS: frozenset[str] = frozenset(
        {
            "create_blob_upload",
            "create_blob_inline",
            "list_blobs",
            "get_blob_metadata",
            "download_blob_content",
            # 0.6.0: bounded inline-preview endpoint
            # (``GET .../blobs/{blob_id}/preview``). Session-scoped like the
            # rest of the blob family; routes through
            # ``_verify_session_and_get_blob_service`` for IDOR safety. The
            # cross-session 404 assertion lives in
            # ``tests/unit/web/blobs/test_routes.py::TestIDORProtection``.
            "preview_blob_content",
            "delete_blob",
        }
    )

    EXPECTED_AUDIT_READINESS_OWNERSHIP_ENDPOINTS: frozenset[str] = frozenset(
        {
            "snapshot",
            "explain",
        }
    )

    @staticmethod
    def _assert_inventory(router_label: str, helper_name: str, expected: frozenset[str], found: set[str]) -> None:
        """Render the drift-diagnostic message and assert set-equality."""
        missing = expected - found
        unexpected = found - expected
        assert found == expected, (
            f"IDOR audit drift detected in {router_label}.\n"
            f"  Expected endpoints calling {helper_name!r}: {sorted(expected)}\n"
            f"  Found: {sorted(found)}\n"
            f"  Missing (endpoint advertised in audit but no call found): {sorted(missing)}\n"
            f"  Unexpected (endpoint calls helper but not in audit inventory): {sorted(unexpected)}\n"
            "Update BOTH the corresponding IDOR assertion walk AND the "
            "inventory here in the SAME commit when endpoints enter or "
            "leave this set. Bumping one without the other leaves the "
            "audit in a lying state — the whole point of this drift "
            "guard is to force the three locations (inventory, "
            "assertion, handler set) to stay in sync."
        )

    def test_sessions_routes_ownership_call_sites(self) -> None:
        """sessions/routes/ — _verify_session_ownership inventory."""
        from elspeth.web.sessions.routes import interpretation, messages, runs, sessions
        from elspeth.web.sessions.routes.composer import compose, guided, proposals, state

        found = set()
        for routes_module in (sessions, state, proposals, compose, guided, messages, runs, interpretation):
            found.update(_collect_ownership_call_site_identities(routes_module, "_verify_session_ownership"))
        self._assert_inventory(
            "sessions/routes/",
            "_verify_session_ownership",
            self.EXPECTED_SESSIONS_OWNERSHIP_ENDPOINTS,
            found,
        )

    def test_execution_routes_session_ownership_call_sites(self) -> None:
        """execution/routes.py — verify_session_ownership inventory.

        Phase 2A.5 extracted the session-ownership helper into
        ``elspeth.web.sessions.ownership.verify_session_ownership`` so
        ``execution/routes.py`` and ``audit_readiness/routes.py`` share a
        single IDOR-safe implementation.  ``execution/routes.py`` no
        longer hosts a file-local ``_verify_session_ownership`` symbol;
        the drift guard now walks calls to the imported
        ``verify_session_ownership`` name.
        """
        from elspeth.web.execution import routes

        found = _collect_ownership_call_site_identities(routes, "verify_session_ownership")
        self._assert_inventory(
            "execution/routes.py",
            "verify_session_ownership",
            self.EXPECTED_EXECUTION_SESSION_OWNERSHIP_ENDPOINTS,
            found,
        )

    def test_execution_routes_run_ownership_call_sites(self) -> None:
        """execution/routes.py — _verify_run_ownership inventory.

        Run-scoped endpoints (``/api/runs/{run_id}``) verify a
        different identity dimension (run ownership, resolved through
        the run's parent session).  The IDOR surface is identical in
        principle: an authenticated user probing run_id UUIDs against
        their own endpoints must not be able to distinguish "doesn't
        exist" from "exists in another user's session".
        """
        from elspeth.web.execution import routes

        found = _collect_ownership_call_site_identities(routes, "_verify_run_ownership")
        self._assert_inventory(
            "execution/routes.py",
            "_verify_run_ownership",
            self.EXPECTED_EXECUTION_RUN_OWNERSHIP_ENDPOINTS,
            found,
        )

    def test_blobs_routes_ownership_call_sites(self) -> None:
        """blobs/routes.py — _verify_session_and_get_blob_service inventory.

        The helper is dual-role (ownership check + service lookup).
        Every blob-management endpoint that operates under a
        ``/api/sessions/{session_id}/blobs`` path MUST call it.  A
        handler that acquires the blob service directly from
        ``request.app.state`` without the ownership check would
        bypass the session IDOR guard entirely — that is the drift
        this inventory exists to catch.
        """
        from elspeth.web.blobs import routes

        found = _collect_ownership_call_site_identities(routes, "_verify_session_and_get_blob_service")
        self._assert_inventory(
            "blobs/routes.py",
            "_verify_session_and_get_blob_service",
            self.EXPECTED_BLOBS_OWNERSHIP_ENDPOINTS,
            found,
        )

    def test_audit_readiness_routes_session_ownership_call_sites(self) -> None:
        """audit_readiness/routes.py — verify_session_ownership inventory.

        Both audit-readiness endpoints (snapshot + explain) are
        session-scoped under ``/api/sessions/{session_id}/audit-readiness``
        and depend on the shared ``verify_session_ownership`` helper
        from ``web/sessions/ownership.py`` for IDOR safety.  Any new
        endpoint added to this router must call the helper AND be
        added to the inventory above.
        """
        # The router-factory closure is defined inside
        # ``create_audit_readiness_router``; the AST walker needs the
        # module source, which it loads from the module's __file__.
        from elspeth.web.audit_readiness import routes

        found = _collect_ownership_call_site_identities(routes, "verify_session_ownership")
        self._assert_inventory(
            "audit_readiness/routes.py",
            "verify_session_ownership",
            self.EXPECTED_AUDIT_READINESS_OWNERSHIP_ENDPOINTS,
            found,
        )


class TestIDORProtection:
    """Tests for W5 -- IDOR protection on all session-scoped routes.

    Creates a session as user A, then attempts to access it as user B.
    All should return 404 (not 403).

    Inventory of session-scoped routes audited here (must match the set
    of callers of ``_verify_session_ownership`` in
    ``src/elspeth/web/sessions/routes.py``). If a new session-scoped
    route is added upstream, its cross-session request MUST be added
    to ``test_idor_session_crud`` — the test's purpose is to walk
    EVERY endpoint that depends on the ownership primitive, so a new
    route added without a matching assertion here is a silent
    coverage regression.

    Audited endpoints:

    - ``GET  /{session_id}``                 (get_session)
    - ``PATCH /{session_id}``                (update_session)
    - ``DELETE /{session_id}``               (delete_session)
    - ``GET  /{session_id}/messages``        (get_messages)
    - ``POST /{session_id}/messages``        (send_message)
    - ``POST /{session_id}/recompose``       (recompose)
    - ``GET  /{session_id}/runs``            (list_session_runs)
    - ``GET  /{session_id}/state``           (get_current_state)
    - ``GET  /{session_id}/state/versions``  (get_state_versions)
    - ``POST /{session_id}/state/revert``    (revert_state)
    - ``GET  /{session_id}/state/yaml``      (get_state_yaml)
    - ``POST /{session_id}/state/yaml``      (import_state_yaml)
    - ``POST /{session_id}/fork``            (fork_from_message)
    - ``GET  /{session_id}/guided``          (get_guided)
    - ``GET  /{session_id}/guided/tutorial-sample`` (get_guided_tutorial_sample)
    - ``POST /{session_id}/state/e2e-seed``  (seed_state_for_e2e)
    - ``POST /{session_id}/guided/reenter``  (post_guided_reenter)
    - ``POST /{session_id}/guided/respond``  (post_guided_respond)
    - ``POST /{session_id}/guided/chat``     (post_guided_chat)

    Counter-test: alice's own access continues to return 200 at the end,
    guarding against the regression where an over-eager 404 breaks
    legitimate access.
    """

    def test_idor_session_crud(self, tmp_path) -> None:
        """Shared-DB IDOR test: alice creates, bob tries to access."""
        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        service = SessionServiceImpl(
            engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )

        # Create two apps sharing the same service
        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)

            async def mock_user():
                return identity

            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = WebSettings(
                data_dir=tmp_path,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
                # ON so the e2e-seed walk assertion below exercises the
                # ownership gate: the handler's feature-flag 404 fires
                # BEFORE _verify_session_ownership, so with the flag off
                # bob's 404 would prove nothing about IDOR.
                e2e_state_seed_enabled=True,
            )
            app.state.catalog_service = None

            from elspeth.web.middleware.rate_limit import ComposerRateLimiter

            app.state.rate_limiter = ComposerRateLimiter(limit=100)
            app.state.composer_progress_registry = ComposerProgressRegistry()
            app.include_router(create_session_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        alice_client = TestClient(alice_app)
        bob_client = TestClient(bob_app)

        # Alice creates a session
        resp = alice_client.post(
            "/api/sessions",
            json={"title": "Alice Only"},
        )
        assert resp.status_code == 201
        session_id = resp.json()["id"]

        # Bob tries to GET it -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to PATCH the user-visible session title -- should be
        # 404. A title update is low-bandwidth but still proves session
        # existence and mutates Alice's workspace if the ownership check
        # is missing.
        resp = bob_client.patch(
            f"/api/sessions/{session_id}",
            json={"title": "Bob was here"},
        )
        assert resp.status_code == 404

        # Bob tries to DELETE it -- should be 404
        resp = bob_client.delete(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to GET messages -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/messages")
        assert resp.status_code == 404

        # Bob tries to POST a message -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "hacked"},
        )
        assert resp.status_code == 404

        # Bob tries to GET state -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state")
        assert resp.status_code == 404

        # Bob tries to GET state versions -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state/versions")
        assert resp.status_code == 404

        # Bob tries to revert state -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/state/revert",
            json={"operation_id": str(uuid.uuid4()), "state_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

        # Bob tries to POST recompose -- should be 404.  The ownership
        # check runs before the rate limiter's side effects in the
        # ``recompose`` route handler, so an attacker cannot use this
        # endpoint to probe for session existence through rate-limit
        # timing either.
        resp = bob_client.post(f"/api/sessions/{session_id}/recompose")
        assert resp.status_code == 404

        # Bob tries to GET runs -- should be 404.  Without this guard,
        # an attacker could enumerate run IDs / timings for sessions
        # belonging to other users and correlate them with activity
        # signals (response size, latency).
        resp = bob_client.get(f"/api/sessions/{session_id}/runs")
        assert resp.status_code == 404

        # Bob tries to GET state/yaml -- should be 404.  The YAML export
        # is the most information-dense state projection (full plugin
        # options, source/sink names, routing); missing this guard would
        # be the highest-bandwidth IDOR leak of the state-read family.
        resp = bob_client.get(f"/api/sessions/{session_id}/state/yaml")
        assert resp.status_code == 404

        # Bob tries to POST state/yaml -- should be 404.  Import is a
        # high-impact mutation endpoint: without the same ownership gate as
        # export, an attacker could overwrite Alice's composition state.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/state/yaml",
            json={
                "yaml": (
                    "sources:\n"
                    "  source:\n"
                    "    plugin: csv\n"
                    "    options:\n"
                    "      path: inputs/data.csv\n"
                    "      schema:\n"
                    "        mode: observed\n"
                    "sinks:\n"
                    "  main:\n"
                    "    plugin: csv\n"
                    "    options:\n"
                    "      path: outputs/out.csv\n"
                )
            },
        )
        assert resp.status_code == 404

        # Bob tries to POST fork -- should be 404.  A successful fork
        # would create a new session owned by Bob but seeded from
        # Alice's state history, cross-contaminating audit lineage.
        # The ownership check runs before ``fork_session()`` is called
        # in the ``fork_from_message`` route handler, so no rows are
        # written on denial.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/fork",
            json={
                "from_message_id": str(uuid.uuid4()),
                "new_message_content": "hijacked",
            },
        )
        assert resp.status_code == 404

        # Bob tries to GET guided state -- should be 404.  The guided
        # endpoint reveals wizard step, history, and pending turn payloads.
        # An ownership bypass would let an attacker read Alice's pipeline
        # wizard state without authorization.  The ownership check in
        # ``get_guided`` runs before the compose lock or catalog access.
        resp = bob_client.get(f"/api/sessions/{session_id}/guided")
        assert resp.status_code == 404

        # Bob tries to POST guided/start -- should be 404. The start entry
        # endpoint (P6.4) constructs a server-owned WorkflowProfile and seeds
        # the guided session; an ownership bypass would let an attacker
        # re-seed / reset the profile and CompositionState of Alice's session.
        # A VALID body is sent so the request reaches the in-handler ownership
        # check (a missing body would 422 at FastAPI validation first); with a
        # valid body, the ownership check returns 404 for the non-owner.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/guided/start",
            json={"profile": "live"},
        )
        assert resp.status_code == 404

        # Bob tries to POST guided/reenter -- should be 404. Re-entry is
        # a mode transition that can reveal and mutate Alice's guided
        # session terminal state if ownership is bypassed.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/guided/reenter",
            json={"operation_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

        # Bob tries to POST guided/respond — should be 404.  The respond
        # endpoint can mutate pipeline state by driving step handlers.
        # An ownership bypass would let an attacker submit guided responses
        # against Alice's session and corrupt her pipeline state.  The
        # ownership check in ``post_guided_respond`` runs before any state
        # load or dispatch.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"control_signal": "exit_to_freeform"},
        )
        assert resp.status_code == 404

        # Bob tries to POST guided/chat — should be 404.  Phase A slice 3
        # introduced the per-step chat endpoint; it sends user-typed text to
        # an LLM scoped to Alice's session step, costing LLM credits and
        # surfacing Alice's wizard step in the reply path.  Slice 5 also
        # made it mutate chat_history on the GuidedSession (an audit-write).
        # An ownership bypass would let bob burn LLM budget against Alice's
        # session AND inject conversational turns into her audit trail.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/guided/chat",
            json={"message": "hi", "step_index": "step_1_source"},
        )
        assert resp.status_code == 404

        # Bob tries to GET guided/tutorial-sample — should be 404. The
        # 0.7.0 synthetic-scrape redesign added this read-only endpoint,
        # which exposes the runtime-derived sample-page URLs and the SSRF
        # host-class for an active tutorial session. The ownership check
        # in ``get_guided_tutorial_sample`` runs FIRST (before the guided/
        # tutorial-state 400 branches), so a non-owner gets 404 regardless
        # of whether a tutorial session exists. An ownership bypass would
        # let an attacker learn Alice's resolved sample origin.
        resp = bob_client.get(f"/api/sessions/{session_id}/guided/tutorial-sample")
        assert resp.status_code == 404

        # Bob tries to POST state/e2e-seed — should be 404. The Playwright
        # seed endpoint is a full composition-state write; with the flag
        # enabled (see WebSettings above) an ownership bypass would let an
        # attacker overwrite Alice's state wholesale. The ownership check
        # runs before the body is even parsed, so bob's 404 fires on
        # ownership, not on request validation.
        resp = bob_client.post(
            f"/api/sessions/{session_id}/state/e2e-seed",
            json={"state": {}},
        )
        assert resp.status_code == 404

        # Alice can still access her own session
        resp = alice_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200


class TestSendMessageStateIdValidation:
    """Route-layer IDOR + information-leak coverage for ``POST /messages``
    ``state_id`` validation.

    The ``send_message`` handler accepts an optional ``state_id`` in the
    request body (see ``sessions/routes.py``'s ``send_message`` around
    the ``if body.state_id is not None`` block). That value is used as
    the ``composition_state_id`` stamped onto the persisted user
    message (AD-2 provenance), so any path that lets a client assert
    a state owned by a *different* session would corrupt Tier 1
    audit lineage — a message in session B claiming to have been
    composed against session A's state.

    The outer ``_verify_session_ownership`` check only gates the
    session itself. The ``state_id`` gate is an independent check that
    runs AFTER ownership passes, so it needs its own assertions:

    * ``test_cross_session_state_id_rejected`` (Gap 16): Bob owns his
      own session but supplies a ``state_id`` owned by Alice's session.
      The route must return 404 — not 200 (silently stamp the user
      message with cross-session provenance), not 403 (acknowledges
      the state exists), and not 500 (a ``RuntimeError`` from the
      service-layer defensive guard would indicate the route check
      was bypassed — that would be a separate bug, tested elsewhere).

    * ``test_404_body_is_identical_for_unknown_and_cross_session``
      (Gap 17): the commit that introduced this validation called
      the 404 mapping "load-bearing ... to avoid leaking other
      sessions' state existence". A distinguishable 404 body (for
      example ``"State not found"`` for unknown UUIDs vs ``"State
      not found for this session"`` for owned-by-other) would defeat
      that claim — an attacker who held any UUID could tell from the
      response text whether the UUID exists in a different session,
      reviving the IDOR information leak. This test pins the two
      responses to byte-for-byte parity.

    Sibling ``test_revert_state_not_belonging_to_session`` already
    covers the analogous case for ``POST /state/revert``; the send-
    message handler has its own ``state_id`` check and needs its own
    pins.
    """

    def _alice_plus_bob_with_state(
        self,
        tmp_path: Path,
    ) -> tuple[TestClient, str, str]:
        """Seed the shared-DB IDOR scenario used by both tests below.

        Returns ``(bob_client, bob_session_id, alice_state_id)``:

        * ``alice_state_id`` is a composition_state owned by ``alice``
          in a session ``alice`` alone can access. Bob never touched it.
        * ``bob_session_id`` is a session owned by ``bob`` with a
          composition_state of its own — so ``get_current_state``
          returns non-None in the send_message handler and the
          ``state_id`` validation branch is actually exercised (an
          empty session would route through the ``state_record is
          None`` branch and SKIP the cross-session check, which
          would make the test vacuous).

        The helper owns the engine and service so both users share the
        same underlying DB — the only way the cross-session lookup can
        resolve at all.
        """
        import asyncio

        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        service = SessionServiceImpl(
            engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )

        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)

            async def mock_user():
                return identity

            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = WebSettings(
                data_dir=tmp_path,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )
            app.state.catalog_service = None
            # Composer MUST NOT be called — state_id validation fails
            # before compose is reached. Set to None so any
            # regression that skips validation surfaces as an
            # AttributeError in the test run rather than silently
            # succeeding against a mock.
            app.state.composer_service = None

            from elspeth.web.middleware.rate_limit import ComposerRateLimiter

            app.state.rate_limiter = ComposerRateLimiter(limit=100)
            app.state.composer_progress_registry = ComposerProgressRegistry()
            app.include_router(create_session_router())
            return app

        # Alice creates her own session and a state in it.
        loop = asyncio.new_event_loop()
        try:
            alice_session = loop.run_until_complete(
                service.create_session("alice", "Alice Only", "local"),
            )
            alice_state = loop.run_until_complete(
                service.save_composition_state(
                    alice_session.id,
                    CompositionStateData(
                        metadata_={"name": "Alice", "description": ""},
                        is_valid=True,
                    ),
                    provenance="session_seed",
                ),
            )
            # Bob creates his own session AND seeds a composition state
            # so get_current_state on bob's session returns non-None —
            # otherwise the send_message handler skips the state_id
            # validation branch and the test is vacuous. ``metadata_``
            # must be a non-None mapping because the route goes through
            # ``_state_from_record`` which Tier-1 crashes on ``None``
            # (see ``converters.state_from_record``).
            bob_session = loop.run_until_complete(
                service.create_session("bob", "Bob's Own", "local"),
            )
            loop.run_until_complete(
                service.save_composition_state(
                    bob_session.id,
                    CompositionStateData(
                        metadata_={"name": "Bob", "description": ""},
                        is_valid=True,
                    ),
                    provenance="session_seed",
                ),
            )
        finally:
            loop.close()

        bob_app = make_app_for_user("bob")
        bob_client = TestClient(bob_app)
        return bob_client, str(bob_session.id), str(alice_state.id)

    def test_cross_session_state_id_rejected(self, tmp_path) -> None:
        """Gap 16: POST /messages with a state_id owned by ANOTHER session → 404.

        This closes the specific IDOR vector where Bob's session is
        legitimately his (``_verify_session_ownership`` passes), but
        the ``state_id`` in the request body points at a state from
        a session he does NOT own. Without the route-layer cross-
        session check, the persisted user message would record
        Alice's state_id as its ``composition_state_id`` — corrupting
        audit lineage. The outer session-ownership check does not
        catch this (Bob legitimately owns the session he is posting
        to); the state_id check is a separate, independent guard.

        Mirrors the existing ``test_revert_state_not_belonging_to_session``
        for ``POST /state/revert`` — every endpoint that accepts a
        client-supplied ``state_id`` needs its own cross-session
        assertion, since the primitive isn't shared.
        """
        bob_client, bob_session_id, alice_state_id = self._alice_plus_bob_with_state(tmp_path)

        resp = bob_client.post(
            f"/api/sessions/{bob_session_id}/messages",
            json={"content": "hello", "state_id": alice_state_id},
        )
        assert resp.status_code == 404, (
            f"Expected 404 for cross-session state_id, got {resp.status_code}. "
            f"Body: {resp.text!r}. Without this guard, the persisted user "
            "message would claim provenance from a session Bob does not own."
        )

    def test_404_body_is_identical_for_unknown_and_cross_session(self, tmp_path) -> None:
        """Gap 17: pin the "load-bearing" 404 parity claim that underpins
        the cross-session IDOR guard on ``/messages``.

        Two distinct failure modes MUST produce byte-identical response
        bodies:

        1. The UUID does not exist anywhere (``service.get_state``
           raises ``ValueError`` → caught by the route).
        2. The UUID exists but belongs to a session the requester
           does not own (route's ``client_state.session_id !=
           session.id`` branch fires).

        If these two paths return distinguishable bodies, an attacker
        holding any UUID can tell whether it maps to a real state in
        *some other user's* session — the exact IDOR information leak
        the commit claimed to prevent. Bytes must match, not just
        status codes.

        Both requests use Bob's client against Bob's own session, so
        ``_verify_session_ownership`` passes for both — the
        differentiating factor is purely the ``state_id`` value. The
        ``offset=1`` placeholder UUID is constructed to be astronomically
        unlikely to collide with alice_state_id (the only real
        composition_state a UUID could match in this scenario).
        """
        bob_client, bob_session_id, alice_state_id = self._alice_plus_bob_with_state(tmp_path)
        unknown_state_id = str(uuid.uuid4())
        # Sanity: guard against the minuscule probability of collision
        # — if uuid4 ever produced alice's id, the test would
        # accidentally exercise the same branch twice.
        assert unknown_state_id != alice_state_id, "uuid4 collided — retry the test"

        unknown_resp = bob_client.post(
            f"/api/sessions/{bob_session_id}/messages",
            json={"content": "hello", "state_id": unknown_state_id},
        )
        cross_session_resp = bob_client.post(
            f"/api/sessions/{bob_session_id}/messages",
            json={"content": "hello", "state_id": alice_state_id},
        )

        assert unknown_resp.status_code == 404
        assert cross_session_resp.status_code == 404

        # Byte-identical bodies — the strict assertion. FastAPI serialises
        # the HTTPException detail into ``{"detail": "..."}``, and any
        # divergence in the detail string shows up here as a byte-diff.
        assert unknown_resp.content == cross_session_resp.content, (
            "404 body parity broken — unknown-UUID vs cross-session UUID "
            "responses differ. This re-introduces the IDOR information "
            "leak the route-level ownership check was added to prevent.\n"
            f"  unknown:       {unknown_resp.content!r}\n"
            f"  cross-session: {cross_session_resp.content!r}\n"
            "Unify the HTTPException detail strings in send_message's "
            "state_id validation block."
        )
        # Also pin the detail text explicitly so a future refactor that
        # renames BOTH strings symmetrically (preserving parity but
        # introducing a new leak vector through a different channel
        # like response headers) still surfaces as a diff here.
        assert unknown_resp.json() == {"detail": "State not found"}, (
            f"Unexpected 404 body shape: {unknown_resp.json()!r}. "
            "The route's 404 detail must remain the generic "
            '"State not found" — anything more specific (e.g. '
            '"State not found in session", "Wrong owner") leaks '
            "membership information."
        )


class TestMessageRoutes:
    """Tests for message send and retrieval endpoints."""

    def test_send_message(self, tmp_path) -> None:
        mock_composer = _make_composer_mock(response_text="Got it!")

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello, build me a pipeline"},
        )
        assert msg_resp.status_code == 200
        body = msg_resp.json()
        assert body["message"]["content"] == "Got it!"
        assert body["message"]["role"] == "assistant"
        # State unchanged (version stayed at 1) -> no state in response
        assert body["state"] is None

    def test_send_message_with_state_id(self, tmp_path) -> None:
        """Message with state_id references a specific composition state snapshot.

        Exercises the UUID-typed state_id field in SendMessageRequest end-to-end:
        FastAPI parses the JSON string into a UUID, the route validates the state
        belongs to the session, and the user message is persisted with the
        client-asserted state_id as its composition_state_id (AD-2 provenance).
        """
        import asyncio

        mock_composer = _make_composer_mock(response_text="Acknowledged")

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app)

        # Create a session
        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        # Create a composition state via the service (the mock composer
        # returns version=1 which won't trigger state persistence in the
        # route, so we seed one directly).
        loop = asyncio.new_event_loop()
        state_record = loop.run_until_complete(
            service.save_composition_state(
                uuid.UUID(session_id),
                CompositionStateData(
                    metadata_={"name": "Test", "description": ""},
                    is_valid=True,
                ),
                provenance="session_seed",
            ),
        )
        loop.close()
        state_id = str(state_record.id)

        # Send message WITH state_id as UUID string in JSON body
        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello", "state_id": state_id},
        )
        assert msg_resp.status_code == 200
        body = msg_resp.json()
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "Acknowledged"

        # Verify provenance: the user message was persisted with the
        # client-asserted state_id as its composition_state_id.
        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        messages = msgs_resp.json()
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["composition_state_id"] == state_id

    def test_send_message_with_stale_state_id_composes_against_head(self, tmp_path) -> None:
        """A stale (but session-owned) client state_id must not poison the
        compose loop's optimistic-concurrency baseline.

        Regression test for elspeth-e08063c3a5: after a client-aborted
        turn, the SPA's ``compositionState.id`` lags the DB head (the
        aborted turn's response never arrived). The follow-up send
        carries the stale id. The route must:

        * keep the client-asserted id for USER-MESSAGE provenance
          (AD-2/AD-7 — it records what the user saw), and
        * seed ``composer.compose(current_state_id=...)`` — which the
          compose loop threads into ``persist_compose_turn`` as
          ``expected_current_state_id`` — from the ACTUAL head loaded
          under the compose lock, exactly as ``/recompose`` does.

        Seeding the loop from the stale client id made every follow-up
        send fail with 409 stale_compose_state ("The session changed
        while the compose turn was running.") until page reload.
        """
        head_version_state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="v2"),
            version=2,
        )
        mock_composer = _make_composer_mock(
            response_text="Recovered",
            # version matches the seeded head so the route's
            # version-changed save path stays out of this test's scope.
            state=head_version_state,
        )
        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        try:
            stale_record = loop.run_until_complete(
                service.save_composition_state(
                    uuid.UUID(session_id),
                    CompositionStateData(
                        metadata_={"name": "v1", "description": ""},
                        is_valid=True,
                    ),
                    provenance="session_seed",
                ),
            )
            head_record = loop.run_until_complete(
                service.save_composition_state(
                    uuid.UUID(session_id),
                    CompositionStateData(
                        metadata_={"name": "v2", "description": ""},
                        is_valid=True,
                    ),
                    provenance="session_seed",
                ),
            )
        finally:
            loop.close()

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "And now add a sink", "state_id": str(stale_record.id)},
        )
        assert msg_resp.status_code == 200

        compose_kwargs = mock_composer.compose.await_args.kwargs
        assert compose_kwargs["current_state_id"] == str(head_record.id), (
            "compose loop's optimistic-concurrency baseline must be the DB "
            "head loaded under the compose lock, not the client-asserted "
            "state_id — a stale client id turns into an unrecoverable 409 "
            "stale_compose_state on every follow-up send"
        )

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        user_msg = next(m for m in msgs_resp.json() if m["role"] == "user")
        assert user_msg["composition_state_id"] == str(stale_record.id), (
            "user-message provenance must still record the client-asserted state (AD-2)"
        )

    def test_client_disconnect_cancels_compose_turn(self, tmp_path) -> None:
        """A client disconnect mid-compose must cancel the server-side turn.

        Regression test for elspeth-e08063c3a5 (zombie half): uvicorn's
        ``connection_lost`` only flags the cycle as disconnected and
        Starlette's ``request_response`` has no disconnect watcher, so
        without ``_cancel_on_client_disconnect`` a client abort (Stop
        button / SPA compose timeout / closed tab) left the compose loop
        running to completion — burning LLM budget, holding the
        per-session compose lock for minutes, and advancing composition
        state the client never sees.

        Drives the ASGI app directly: the request body is delivered,
        then — once the composer stub signals it has started — the
        receive channel yields ``http.disconnect``. The watcher must
        cancel the route task; the route's cancelled-path bookkeeping
        converts the disconnect-initiated cancel into a quiet 499 (the
        client is gone; uvicorn discards the bytes — the conversion
        exists so CancelledError does not escape the app and get logged
        as an ASGI crash on every Stop click).
        """

        class _HangingComposer:
            """compose() parks forever; records whether it was cancelled."""

            def __init__(self, started: asyncio.Event) -> None:
                self.started = started
                self.cancelled = False

            async def compose(self, *args: Any, **kwargs: Any) -> ComposerResult:
                del args, kwargs
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                raise AssertionError("unreachable — compose never completes")

        app, _service = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        async def drive() -> tuple[list[dict[str, Any]], _HangingComposer]:
            compose_started = asyncio.Event()
            composer = _HangingComposer(compose_started)
            app.state.composer_service = composer

            request_messages = [
                {
                    "type": "http.request",
                    "body": json.dumps({"content": "build a pipeline"}).encode(),
                    "more_body": False,
                }
            ]
            sent: list[dict[str, Any]] = []

            async def receive() -> dict[str, Any]:
                if request_messages:
                    return request_messages.pop(0)
                # Second receive() is the disconnect watcher: report the
                # client gone as soon as the compose loop is running.
                await compose_started.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                sent.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": f"/api/sessions/{session_id}/messages",
                "raw_path": f"/api/sessions/{session_id}/messages".encode(),
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            }
            # Pre-fix behaviour is an unbounded hang (nothing observes the
            # disconnect); the wait_for turns that into a bounded failure.
            await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
            return sent, composer

        loop = asyncio.new_event_loop()
        try:
            sent, composer = loop.run_until_complete(drive())
        finally:
            loop.close()

        assert composer.cancelled is True, "compose loop must be cancelled when the client disconnects"
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        assert status == 499, f"disconnect-cancel must unwind as a quiet 499, got {status}"

        # The turn must NOT have produced an assistant message — the user
        # message persists (it was committed before compose started), the
        # zombie's would-be reply must not.
        msgs = client.get(f"/api/sessions/{session_id}/messages").json()
        assert [m["role"] for m in msgs] == ["user"]

        # And the session must not be wedged: a fresh send composes fine.
        app.state.composer_service = _make_composer_mock(response_text="Still alive")
        follow_up = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "try again"},
        )
        assert follow_up.status_code == 200
        assert follow_up.json()["message"]["content"] == "Still alive"

    def test_external_cancel_racing_disconnect_keeps_unwinding(self) -> None:
        """An external cancel racing a client disconnect must keep unwinding.

        ``triggered`` only proves a disconnect happened — not that the caught
        CancelledError belongs to the watcher alone. When server shutdown or
        an operator cancel races ``http.disconnect``, the route task carries
        TWO cancellation requests; consuming one and marking the exception as
        disconnect-initiated would let the route convert it into a handled
        499 and swallow the external cancellation. The watcher may consume
        only its own request: with an external cancel still pending, the
        exception must stay unmarked so the route keeps unwinding as
        genuinely cancelled (mirroring the else-branch's ``cancelling()``
        re-check for the completion race).
        """
        from starlette.requests import Request

        from elspeth.web.sessions.routes._helpers import (
            _cancel_on_client_disconnect,
            _is_client_disconnect_cancel,
        )

        async def drive() -> tuple[bool, bool]:
            started = asyncio.Event()
            allow_disconnect = asyncio.Event()

            async def receive() -> dict[str, Any]:
                await allow_disconnect.wait()
                return {"type": "http.disconnect"}

            request = Request({"type": "http"}, receive)
            captured: dict[str, bool] = {}

            async def guarded() -> None:
                try:
                    async with _cancel_on_client_disconnect(request):
                        started.set()
                        await asyncio.Event().wait()
                except asyncio.CancelledError as exc:
                    captured["marked"] = _is_client_disconnect_cancel(exc)
                    raise
                raise AssertionError("unreachable — the guarded block never completes")

            task = asyncio.get_running_loop().create_task(guarded())
            await started.wait()
            allow_disconnect.set()
            # Let the watcher observe the disconnect and file its cancel...
            while task.cancelling() == 0:
                await asyncio.sleep(0)
            # ...then land the external cancel (server shutdown / operator)
            # before the parked task has processed the watcher's.
            task.cancel()
            assert task.cancelling() == 2
            with pytest.raises(asyncio.CancelledError):
                await task
            return captured["marked"], task.cancelled()

        loop = asyncio.new_event_loop()
        try:
            marked, cancelled = loop.run_until_complete(drive())
        finally:
            loop.close()

        assert marked is False, (
            "the CancelledError must NOT be marked disconnect-initiated while "
            "an external cancellation request is still pending — the mark lets "
            "the route swallow a server-shutdown/operator cancel as a quiet 499"
        )
        assert cancelled is True, "the task must finish as genuinely cancelled"

    def test_composer_progress_reports_inflight_request_count(self, tmp_path) -> None:
        """GET /composer-progress carries the live in-flight compose count.

        The count is the SPA's correlated settlement signal after a client
        abort (elspeth-06a23adfcc): the snapshot phase cannot distinguish
        "the aborted route is still running but has not published progress
        yet" (queued on the compose lock, immediate Stop) from "everything
        settled" — the registry may hold the previous turn's terminal
        snapshot in both. The count spans the whole request and drops to
        zero only after the route fully unwinds.
        """
        app, _service = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        async def drive() -> None:
            compose_started = asyncio.Event()
            release = asyncio.Event()
            inner = _make_composer_mock(response_text="Done")

            class _ParkedComposer:
                async def compose(self, *args: Any, **kwargs: Any) -> Any:
                    compose_started.set()
                    await release.wait()
                    return await inner.compose(*args, **kwargs)

            app.state.composer_service = _ParkedComposer()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as http:
                send_task = asyncio.create_task(
                    http.post(
                        f"/api/sessions/{session_id}/messages",
                        json={"content": "build a pipeline"},
                    )
                )
                await asyncio.wait_for(compose_started.wait(), timeout=5.0)
                parked = await http.get(f"/api/sessions/{session_id}/composer-progress")
                assert parked.status_code == 200
                assert parked.json()["inflight_requests"] == 1

                release.set()
                send_resp = await asyncio.wait_for(send_task, timeout=10.0)
                assert send_resp.status_code == 200
                settled = await http.get(f"/api/sessions/{session_id}/composer-progress")
                assert settled.json()["inflight_requests"] == 0

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(drive())
        finally:
            loop.close()

    def test_get_messages(self, tmp_path) -> None:
        mock_composer = _make_composer_mock()

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "First"},
        )
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Second"},
        )

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        # Each POST creates a user message + assistant message = 4 total
        assert len(messages) == 4
        assert messages[0]["content"] == "First"
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Sure, I can help."
        assert messages[1]["role"] == "assistant"

    def test_get_messages_returns_stored_tool_call_arrays(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        app.state.composer_service = _make_composer_mock()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(
                session_id,
                "assistant",
                "Calling a tool",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "list_sources",
                            "arguments": '{"kind":"csv"}',
                        },
                    }
                ],
                writer_principal="compose_loop",
            )
        )
        loop.close()

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        assert len(messages) == 1
        assert messages[0]["tool_calls"] == [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "list_sources",
                    "arguments": '{"kind":"csv"}',
                },
            }
        ]

    def test_get_messages_hides_audit_tool_rows_from_chat_response(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        app.state.composer_service = _make_composer_mock()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(service.add_message(session_id, "user", "Build it", writer_principal="route_user_message"))
            # Rev-4: audit-only breadcrumb rows (no real assistant parent)
            # are persisted with ``role="audit"`` so the
            # ``ck_chat_messages_parent_role`` biconditional is satisfied.
            # Pre-rev-4 stored these as ``role="tool"`` with no parent.
            loop.run_until_complete(
                service.add_message(
                    session_id,
                    "audit",
                    '{"success": true}',
                    tool_calls=_audit_tool_calls("call-1"),
                    writer_principal="compose_loop",
                )
            )
            loop.run_until_complete(
                service.add_message(session_id, "assistant", "I updated the pipeline.", writer_principal="compose_loop")
            )
            persisted = loop.run_until_complete(service.get_messages(session_id, limit=None))
        finally:
            loop.close()

        assert [message.role for message in persisted] == ["user", "audit", "assistant"]

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert all(message["content"] != '{"success": true}' for message in messages)

    def test_send_message_persists_llm_call_audit_sidecars_with_precompose_state(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        llm_calls = (
            _llm_call(provider_request_id="chatcmpl-a", prompt_tokens=13, completion_tokens=8, total_tokens=21, reasoning_tokens=12),
            _llm_call(provider_request_id="chatcmpl-b", prompt_tokens=5, completion_tokens=16, total_tokens=21),
        )
        composer = SimpleNamespace()
        composer.compose = AsyncMock(
            spec=ComposerService.compose, return_value=ComposerResult(message="Saved with audit.", state=_EMPTY_STATE, llm_calls=llm_calls)
        )
        app.state.composer_service = composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])

        loop = asyncio.new_event_loop()
        try:
            pre_state = loop.run_until_complete(
                service.save_composition_state(
                    session_id,
                    CompositionStateData(metadata_={"name": "Precompose", "description": ""}, is_valid=True),
                    provenance="session_seed",
                )
            )
        finally:
            loop.close()

        send_resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Build it"})

        assert send_resp.status_code == 200
        loop = asyncio.new_event_loop()
        try:
            persisted = loop.run_until_complete(service.get_messages(session_id, limit=None))
        finally:
            loop.close()

        llm_audit_rows = _llm_call_audit_rows(persisted)
        assert len(llm_audit_rows) == 2
        assert {row.composition_state_id for row, _tool_call in llm_audit_rows} == {pre_state.id}
        assert [tool_call["call"]["provider_request_id"] for _row, tool_call in llm_audit_rows] == ["chatcmpl-a", "chatcmpl-b"]
        assert sum(tool_call["call"]["total_tokens"] for _row, tool_call in llm_audit_rows) == 42
        assert llm_audit_rows[0][1]["call"]["reasoning_tokens"] == 12
        assert json.loads(llm_audit_rows[0][0].content)["reasoning_tokens"] == 12

        messages_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status_code == 200
        assert [message["role"] for message in messages_resp.json()] == ["user", "assistant"]

    def test_get_messages_can_include_llm_audit_sidecars_without_tool_audit_rows(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(service.add_message(session_id, "user", "Build it", writer_principal="route_user_message"))
            # Rev-4: dispatch-trail audit envelopes (no real assistant
            # parent) and LLM-call audit sidecars are persisted with
            # ``role="audit"`` (the parent-CHECK biconditional rejects
            # ``role="tool"`` here).
            loop.run_until_complete(
                service.add_message(
                    session_id,
                    "audit",
                    '{"success": true}',
                    tool_calls=_audit_tool_calls("call-tool"),
                    writer_principal="compose_loop",
                )
            )
            loop.run_until_complete(
                service.add_message(
                    session_id,
                    "audit",
                    '{"_kind": "llm_call_audit", "total_tokens": 21, "provider_cost": 0.0037}',
                    tool_calls=_llm_call_audit_tool_calls(
                        _llm_call(
                            provider_request_id="chatcmpl-cost",
                            provider_cost=0.0037,
                            provider_cost_source="response_usage.cost",
                        )
                    ),
                    writer_principal="compose_loop",
                )
            )
            loop.run_until_complete(service.add_message(session_id, "assistant", "Done.", writer_principal="compose_loop"))
        finally:
            loop.close()

        hidden_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert hidden_resp.status_code == 200
        assert [message["role"] for message in hidden_resp.json()] == ["user", "assistant"]

        audit_resp = client.get(f"/api/sessions/{session_id}/messages?include_llm_audit=true")
        assert audit_resp.status_code == 200
        messages = audit_resp.json()
        # Rev-4: the LLM-call audit sidecar surfaces as ``role="audit"``
        # in the response (it was previously surfaced as ``role="tool"``
        # because that's how it was persisted).
        assert [message["role"] for message in messages] == ["user", "audit", "assistant"]
        tool_calls = messages[1]["tool_calls"]
        assert tool_calls[0]["_kind"] == "llm_call_audit"
        assert tool_calls[0]["call"]["provider_cost"] == 0.0037
        assert all("call-tool" not in str(message.get("tool_calls")) for message in messages)

    def test_get_messages_can_include_raw_content_for_intercepted_assistant_turns(self, tmp_path) -> None:
        """raw_content (model's actual prose) is exposed only when explicitly requested.

        Server-side synthesis at ``service._finalize_no_tool_response`` replaces the
        model's actual content with a synthetic blocker message (or augments it with an
        operator-facing suffix) and stashes the original in ``raw_content``.
        The eval harness needs the original prose to diagnose whether the model converged;
        the SPA does not. Mirrors the ``include_llm_audit`` query-param pattern.
        """
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])

        synthetic = "[ELSPETH-SYSTEM] No composition-state mutation completed successfully…"
        actual_prose = "I confirmed the file shape from your sample: id→string, message→string, approved→boolean-like."

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(service.add_message(session_id, "user", "Build it", writer_principal="route_user_message"))
            loop.run_until_complete(
                service.add_message(
                    session_id,
                    "assistant",
                    synthetic,
                    raw_content=actual_prose,
                    writer_principal="compose_loop",
                )
            )
        finally:
            loop.close()

        default_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert default_resp.status_code == 200
        default_messages = default_resp.json()
        assistant_default = next(m for m in default_messages if m["role"] == "assistant")
        assert assistant_default["content"] == synthetic
        assert assistant_default.get("raw_content") is None

        included_resp = client.get(f"/api/sessions/{session_id}/messages?include_raw_content=true")
        assert included_resp.status_code == 200
        included_messages = included_resp.json()
        assistant_included = next(m for m in included_messages if m["role"] == "assistant")
        assert assistant_included["content"] == synthetic
        assert assistant_included["raw_content"] == actual_prose

        user_included = next(m for m in included_messages if m["role"] == "user")
        assert user_included["raw_content"] is None

    def test_send_message_llm_call_persistence_failure_raises_on_success_path(self, tmp_path) -> None:
        """Success-path LLM-call audit-row persist failure MUST raise (Tier-1 audit corruption).

        After CLAUDE.md audit-primacy enforcement, a SQLAlchemyError on the
        success-path LLM-call audit-sidecar insert is a Tier-1 audit
        corruption: the assistant row already exists in the audit trail but
        the LLM-call audit row that proves what the model returned is
        missing. ``_persist_llm_calls`` is invoked with
        ``plugin_crash_pending=False`` on the success path; the helper
        raises :class:`AuditIntegrityError` chained through the
        ``OperationalError`` so the request 500s with the diagnostic visible
        to the operator.

        The "fail-soft on persist failure" behaviour previously asserted
        here was the bug: silently swallowing the audit-row write
        rationalised silent failure as mercy and violated CLAUDE.md
        Auditability Standard ("'I don't know what happened' is never an
        acceptable answer for any output").
        """
        app, service = _make_app(tmp_path)
        composer = SimpleNamespace()
        composer.compose = AsyncMock(
            spec=ComposerService.compose,
            return_value=ComposerResult(
                message="Assistant still saved.",
                state=_EMPTY_STATE,
                llm_calls=(_llm_call(provider_request_id="chatcmpl-fail-persist"),),
            ),
        )
        app.state.composer_service = composer
        client = TestClient(app, raise_server_exceptions=False)

        original_add_message = service.add_message

        async def flaky_add_message(*args: Any, **kwargs: Any) -> ChatMessageRecord:
            role = args[1]
            tool_calls = kwargs.get("tool_calls")
            # LLM-call audit sidecars persist with role="audit" — trigger
            # only on that specific insert so the assistant row succeeds
            # first (which is the precondition for the Tier-1 corruption
            # the helper now guards against).
            if role == "audit" and tool_calls and tool_calls[0].get("_kind") == "llm_call_audit":
                raise OperationalError("INSERT INTO chat_messages", {}, Exception("db unavailable"))
            return await original_add_message(*args, **kwargs)

        service.add_message = flaky_add_message  # type: ignore[method-assign]

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])
        send_resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Build it"})

        assert send_resp.status_code == 500

    def test_send_message_tool_invocation_persistence_failure_raises_on_success_path(self, tmp_path) -> None:
        """Symmetric to the LLM-call audit Tier-1 test.

        ``_persist_tool_invocations`` writes ``role="tool"`` audit
        breadcrumbs on the success path (with ``parent_assistant_id`` and
        ``plugin_crash_pending=False``). A SQLAlchemyError from that
        sidecar insert is a Tier-1 audit corruption (assistant row exists,
        tool row missing) — the helper raises
        :class:`AuditIntegrityError` and the request 500s. This replaces
        the pre-fix "fail-soft" expectation; see the LLM-call sibling
        test above for the full doctrine link.
        """
        app, service = _make_app(tmp_path)
        invocation = ComposerToolInvocation(
            tool_call_id="call_test_001",
            tool_name="preview_pipeline",
            arguments_canonical="{}",
            arguments_hash="0" * 64,
            result_canonical='{"ok":true}',
            result_hash="1" * 64,
            status=ComposerToolStatus.SUCCESS,
            error_class=None,
            error_message=None,
            version_before=0,
            version_after=0,
            started_at=datetime(2026, 5, 9, tzinfo=UTC),
            finished_at=datetime(2026, 5, 9, tzinfo=UTC),
            latency_ms=1,
            actor="composer-web:user-test",
        )
        composer = SimpleNamespace()
        composer.compose = AsyncMock(
            spec=ComposerService.compose,
            return_value=ComposerResult(
                message="Assistant still saved.",
                state=_EMPTY_STATE,
                tool_invocations=(invocation,),
            ),
        )
        app.state.composer_service = composer
        client = TestClient(app, raise_server_exceptions=False)

        original_add_message = service.add_message

        async def flaky_add_message(*args: Any, **kwargs: Any) -> ChatMessageRecord:
            role = args[1]
            tool_calls = kwargs.get("tool_calls")
            if role == "tool" and tool_calls and tool_calls[0].get("_kind") == "audit":
                raise OperationalError("INSERT INTO chat_messages", {}, Exception("db unavailable"))
            return await original_add_message(*args, **kwargs)

        service.add_message = flaky_add_message  # type: ignore[method-assign]

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = uuid.UUID(resp.json()["id"])
        send_resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Build it"})

        assert send_resp.status_code == 500

    def test_guided_respond_tool_invocation_persistence_failure_raises_on_success_path(self, tmp_path) -> None:
        """Guided turn audit sidecar failures must not be swallowed after a successful state write."""
        app, service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = []
        app.state.catalog_service = catalog
        app.state.session_engine = service._engine
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided"})
        session_id = uuid.UUID(resp.json()["id"])

        guided_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert guided_resp.status_code == 200

        original_add_message = service.add_message

        async def flaky_add_message(*args: Any, **kwargs: Any) -> ChatMessageRecord:
            role = args[1]
            tool_calls = kwargs.get("tool_calls")
            if role == "audit" and tool_calls and tool_calls[0].get("_kind") == "audit":
                raise OperationalError("INSERT INTO chat_messages", {}, Exception("db unavailable"))
            return await original_add_message(*args, **kwargs)

        service.add_message = flaky_add_message  # type: ignore[method-assign]

        send_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"control_signal": "exit_to_freeform"},
        )

        assert send_resp.status_code == 500

    def test_guided_chat_turn_persistence_failure_raises_on_success_path(self, tmp_path) -> None:
        """Guided chat audit rows must not disappear after chat_history is persisted."""
        app, service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = []
        app.state.catalog_service = catalog
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided chat"})
        session_id = uuid.UUID(resp.json()["id"])

        guided_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert guided_resp.status_code == 200

        original_add_message = service.add_message

        async def flaky_add_message(*args: Any, **kwargs: Any) -> ChatMessageRecord:
            role = args[1]
            tool_calls = kwargs.get("tool_calls")
            if role == "audit" and tool_calls and tool_calls[0].get("_kind") == "chat_turn_audit":
                raise OperationalError("INSERT INTO chat_messages", {}, Exception("db unavailable"))
            return await original_add_message(*args, **kwargs)

        service.add_message = flaky_add_message  # type: ignore[method-assign]

        with patch(
            "elspeth.web.sessions.routes.composer.guided.solve_step_chat_with_auto_drop",
            new=_async_return(
                StepChatResult(
                    assistant_message="Use the source form first.",
                    status=ComposerChatTurnStatus.SUCCESS,
                    latency_ms=7,
                    error_class=None,
                )
            ),
        ):
            send_resp = client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "help me", "step_index": "step_1_source"},
            )

        assert send_resp.status_code == 500

    def test_client_disconnect_cancels_guided_chat_turn(self, tmp_path) -> None:
        """A client disconnect mid-guided-chat must cancel the server turn.

        Guided sibling of test_client_disconnect_cancels_compose_turn
        (elspeth-b2d9e4d084): without the watcher, an aborted guided chat
        (Stop button / SPA compose timeout / closed tab) left the step
        solver running to completion as a zombie — burning LLM budget,
        holding the per-session compose lock, and appending chat turns the
        client never sees. The route's cancelled-path bookkeeping (shielded
        cancelled progress publish + unwind audit drain) already existed;
        the watcher supplies the missing trigger, and the 499 conversion
        keeps the unwind quiet instead of an ASGI crash log per Stop click.
        """
        app, _service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = []
        app.state.catalog_service = catalog
        client = TestClient(app)
        resp = client.post("/api/sessions", json={"title": "Guided chat"})
        session_id = resp.json()["id"]
        guided_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert guided_resp.status_code == 200

        async def drive() -> tuple[list[dict[str, Any]], bool]:
            solver_started = asyncio.Event()
            solver_cancelled = {"flag": False}

            async def hanging_solver(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                solver_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    solver_cancelled["flag"] = True
                    raise
                raise AssertionError("unreachable — the solver never completes")

            request_messages = [
                {
                    "type": "http.request",
                    "body": json.dumps({"message": "help me", "step_index": "step_1_source"}).encode(),
                    "more_body": False,
                }
            ]
            sent: list[dict[str, Any]] = []

            async def receive() -> dict[str, Any]:
                if request_messages:
                    return request_messages.pop(0)
                # Second receive() is the disconnect watcher: report the
                # client gone once the solver is running.
                await solver_started.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                sent.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": f"/api/sessions/{session_id}/guided/chat",
                "raw_path": f"/api/sessions/{session_id}/guided/chat".encode(),
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            }
            with patch(
                # The fresh step-1 session takes the step-1 source-chat
                # resolver (awaited inline in the route task), not the
                # generic advisory solver.
                "elspeth.web.sessions.routes.composer.guided.resolve_step_1_source_chat_with_auto_drop",
                new=hanging_solver,
            ):
                # Pre-fix behaviour is an unbounded hang (nothing observes
                # the disconnect); the wait_for turns that into a bounded
                # failure.
                await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
            return sent, solver_cancelled["flag"]

        loop = asyncio.new_event_loop()
        try:
            sent, solver_cancelled = loop.run_until_complete(drive())
        finally:
            loop.close()

        assert solver_cancelled, "guided chat solver must be cancelled when the client disconnects"
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        assert status == 499, f"disconnect-cancel must unwind as a quiet 499, got {status}"

    def test_guided_chat_source_commit_failure_does_not_leak_tool_result_repr(self, tmp_path) -> None:
        """Step-1 chat source commit failures must not return ToolResult reprs."""
        from elspeth.contracts.blobs import BlobRecord
        from elspeth.contracts.enums import CreationModality
        from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
        from elspeth.web.composer.guided.chat_solver import Step1SourceChatResolution
        from elspeth.web.composer.tools import ToolResult

        app, _ = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
        ]
        catalog.list_sinks.return_value = []
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV source",
            json_schema={"title": "CSV", "type": "object", "properties": {}},
            knob_schema={"fields": []},
        )
        app.state.catalog_service = catalog
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.list_blobs.return_value = []
        app.state.blob_service.create_blob.return_value = BlobRecord(
            id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            filename="source.csv",
            mime_type="text/csv",
            size_bytes=18,
            content_hash="0" * 64,
            storage_path="sessions/raw-secret-source.csv",
            created_at=datetime.now(UTC),
            created_by="assistant",
            source_description="test",
            status="ready",
            creation_modality=CreationModality.VERBATIM,
            created_from_message_id=None,
            creating_model_identifier=None,
            creating_model_version=None,
            creating_provider=None,
            creating_composer_skill_hash=None,
            creating_arguments_hash=None,
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided chat source failure"})
        session_id = uuid.UUID(resp.json()["id"])
        guided_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert guided_resp.status_code == 200
        choose_source_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert choose_source_resp.status_code == 200
        assert choose_source_resp.json()["next_turn"]["type"] == "schema_form"

        raw_row_secret = "raw-customer-ssn-123-45-6789"
        tool_result_private_detail = "REDACTED tool result detail"
        failing_tool_result = ToolResult(
            success=False,
            updated_state=_EMPTY_STATE,
            validation=ValidationSummary(is_valid=False, errors=()),
            affected_nodes=(),
            data={"internal_detail": tool_result_private_detail},
        )

        with (
            patch(
                "elspeth.web.sessions.routes.composer.guided.solve_step_chat_with_auto_drop",
                new=_async_return(
                    StepChatResult(
                        assistant_message="I can use that source.",
                        status=ComposerChatTurnStatus.SUCCESS,
                        latency_ms=5,
                        error_class=None,
                    )
                ),
            ),
            patch(
                "elspeth.web.sessions.routes.composer.guided.resolve_step_1_source_chat_with_auto_drop",
                new=_async_return(
                    Step1SourceChatResult(
                        source_resolution=Step1SourceChatResolution(
                            assistant_message="I created the source.",
                            plugin="csv",
                            filename="source.csv",
                            mime_type="text/csv",
                            content="name,value\nalice,1\n",
                            options={"path": "inline://source.csv"},
                            observed_columns=("name", "value"),
                            sample_rows=({"name": raw_row_secret, "value": "1"},),
                            on_validation_failure="discard",
                        ),
                        fallback_chat=None,
                    )
                ),
            ),
            patch(
                "elspeth.web.sessions.routes.composer.guided.handle_step_1_source",
                return_value=SimpleNamespace(tool_result=failing_tool_result),
            ),
        ):
            send_resp = client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "Use this source", "step_index": "step_1_source"},
            )

        # The strict commit seam rejected the proposal. The source step degrades
        # to advisory (parity with the sink reject path) instead of a fatal 400,
        # so a second Send is never terminal. The egress guarantee is unchanged:
        # the raw tool_result (which can carry Tier-3 row data) must NOT reach the
        # response body on ANY exit path.
        assert send_resp.status_code == 200
        body = send_resp.text
        assert "ToolResult(" not in body
        assert raw_row_secret not in body
        assert tool_result_private_detail not in body
        # No mutation: the rejected commit must not advance or apply.
        assert send_resp.json()["guided_session"]["step"] == "step_1_source"

    def test_guided_respond_source_commit_failure_does_not_leak_tool_result_repr(self, tmp_path) -> None:
        """Step-1 RESPOND (accept) source commit failures must not return ToolResult
        reprs — symmetric with the /guided/chat egress control. The respond path is
        load-bearing (a deliberate accept), so it KEEPS the 400; only the leaky detail
        is redacted to the generic string. The default ToolResult repr dumps
        updated_state + data, which for inline-content sources can carry raw row data.
        """
        from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
        from elspeth.web.composer.tools import ToolResult

        app, _ = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
        ]
        catalog.list_sinks.return_value = []
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV source",
            json_schema={
                "title": "CSV",
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "schema": {"type": "object"},
                },
            },
            knob_schema={
                "fields": [
                    {
                        "name": "path",
                        "label": "Path",
                        "kind": "text",
                        "required": True,
                        "nullable": False,
                    },
                    {
                        "name": "schema",
                        "label": "Schema",
                        "kind": "json-object",
                        "required": True,
                        "nullable": False,
                    },
                ]
            },
        )
        app.state.catalog_service = catalog
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.list_blobs.return_value = []
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided respond source failure"})
        session_id = uuid.UUID(resp.json()["id"])
        assert client.get(f"/api/sessions/{session_id}/guided").status_code == 200
        choose = client.post(f"/api/sessions/{session_id}/guided/respond", json={"chosen": ["csv"]})
        assert choose.status_code == 200
        assert choose.json()["next_turn"]["type"] == "schema_form"

        raw_row_secret = "raw-customer-ssn-123-45-6789"
        tool_result_private_detail = "REDACTED tool result detail"
        failing_tool_result = ToolResult(
            success=False,
            updated_state=_EMPTY_STATE,
            validation=ValidationSummary(is_valid=False, errors=()),
            affected_nodes=(),
            data={"internal_detail": tool_result_private_detail, "row": raw_row_secret},
        )
        with patch(
            "elspeth.web.sessions.routes._helpers.handle_step_1_source",
            return_value=SimpleNamespace(tool_result=failing_tool_result),
        ):
            commit = client.post(
                f"/api/sessions/{session_id}/guided/respond",
                json={
                    "edited_values": {
                        "plugin": "csv",
                        "options": {"path": "inline://source.csv", "schema": {"mode": "observed"}},
                        "observed_columns": ["name"],
                        "sample_rows": [{"name": "alice"}],
                    }
                },
            )

        # Load-bearing accept path KEEPS the 400, but the detail must be the generic
        # string with NO ToolResult repr / Tier-3 data leaked.
        assert commit.status_code == 400, commit.text
        body = commit.text
        assert "ToolResult(" not in body
        assert raw_row_secret not in body
        assert tool_result_private_detail not in body
        assert commit.json()["detail"] == "Step 1 source commit failed"

    def test_guided_chat_malformed_source_tool_args_return_synthetic_unavailable(self, tmp_path) -> None:
        """Malformed Step-1 source resolver tool output must not escape as HTTP 500."""
        from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
        from elspeth.web.sessions._guided_step_chat import _SYNTHETIC_UNAVAILABLE_MESSAGE

        app, service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
        ]
        catalog.list_sinks.return_value = []
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV source",
            json_schema={"title": "CSV", "type": "object", "properties": {}},
            knob_schema={"fields": []},
        )
        app.state.catalog_service = catalog
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.list_blobs.return_value = []
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided malformed source tool args"})
        session_id = uuid.UUID(resp.json()["id"])
        assert client.get(f"/api/sessions/{session_id}/guided").status_code == 200
        choose_source_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert choose_source_resp.status_code == 200
        assert choose_source_resp.json()["next_turn"]["type"] == "schema_form"

        malformed_tool_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="resolve_source",
                                    arguments="{not-json",
                                )
                            )
                        ]
                    )
                )
            ]
        )

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=_async_return(malformed_tool_response),
        ):
            send_resp = client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "Use this CSV: name,value\\nalice,1", "step_index": "step_1_source"},
            )

        assert send_resp.status_code == 200
        body = send_resp.json()
        assert body["assistant_message"] == _SYNTHETIC_UNAVAILABLE_MESSAGE
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"] is None
        app.state.blob_service.create_blob.assert_not_called()

        loop = asyncio.new_event_loop()
        try:
            persisted = loop.run_until_complete(service.get_messages(session_id, limit=None))
        finally:
            loop.close()
        llm_audit_rows = _llm_call_audit_rows(persisted)
        assert len(llm_audit_rows) == 1
        assert llm_audit_rows[0][1]["call"]["status"] == ComposerLLMCallStatus.MALFORMED_RESPONSE.value
        assert llm_audit_rows[0][1]["call"]["error_class"] == "JSONDecodeError"

    def test_guided_chat_source_plugin_mismatch_returns_synthetic_unavailable(self, tmp_path) -> None:
        """Step-1 source resolver plugin mismatch must not commit source state."""
        from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
        from elspeth.web.sessions._guided_step_chat import _SYNTHETIC_UNAVAILABLE_MESSAGE

        app, service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = [
            PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
        ]
        catalog.list_sinks.return_value = []
        catalog.get_schema.return_value = PluginSchemaInfo(
            name="csv",
            plugin_type="source",
            description="CSV source",
            json_schema={"title": "CSV", "type": "object", "properties": {}},
            knob_schema={"fields": []},
        )
        app.state.catalog_service = catalog
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.list_blobs.return_value = []
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Guided source plugin mismatch"})
        session_id = uuid.UUID(resp.json()["id"])
        assert client.get(f"/api/sessions/{session_id}/guided").status_code == 200
        choose_source_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"chosen": ["csv"]},
        )
        assert choose_source_resp.status_code == 200

        valid_but_mismatched_args = json.dumps(
            {
                "resolution": "source",
                "plugin": "json",
                "filename": "source.csv",
                "mime_type": "text/csv",
                "content": "name,value\\nalice,1\\n",
                "options": {},
                "observed_columns": ["name", "value"],
                "sample_rows": [{"name": "alice", "value": "1"}],
                "assistant_message": "I created the source.",
            }
        )
        mismatched_tool_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="resolve_source",
                                    arguments=valid_but_mismatched_args,
                                )
                            )
                        ]
                    )
                )
            ]
        )

        with patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=_async_return(mismatched_tool_response),
        ):
            send_resp = client.post(
                f"/api/sessions/{session_id}/guided/chat",
                json={"message": "Use this JSON file", "step_index": "step_1_source"},
            )

        assert send_resp.status_code == 200
        body = send_resp.json()
        assert body["assistant_message"] == _SYNTHETIC_UNAVAILABLE_MESSAGE
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"] is None
        app.state.blob_service.create_blob.assert_not_called()

        loop = asyncio.new_event_loop()
        try:
            current_state = loop.run_until_complete(service.get_current_state(session_id))
            persisted = loop.run_until_complete(service.get_messages(session_id, limit=None))
        finally:
            loop.close()
        assert current_state is not None
        assert not current_state.sources
        llm_audit_rows = _llm_call_audit_rows(persisted)
        assert len(llm_audit_rows) == 1
        assert llm_audit_rows[0][1]["call"]["status"] == ComposerLLMCallStatus.MALFORMED_RESPONSE.value
        assert llm_audit_rows[0][1]["call"]["error_class"] == "ValueError"

    @pytest.mark.asyncio
    async def test_send_message_serializes_concurrent_requests_per_session(self, tmp_path) -> None:
        """Concurrent sends must not compose against an in-flight partial transcript."""
        composer = _BlockingRecordingComposer()

        app, _ = _make_app(tmp_path)
        app.state.composer_service = composer

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            create_resp = await client.post("/api/sessions", json={"title": "Chat"})
            assert create_resp.status_code == 201
            session_id = create_resp.json()["id"]

            async def send(content: str):
                return await client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={"content": content},
                )

            first_task = asyncio.create_task(send("First"))
            await asyncio.wait_for(composer.first_call_started.wait(), timeout=1.0)

            second_task = asyncio.create_task(send("Second"))

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(composer.second_call_started.wait(), timeout=0.3)

            composer.release_first_call.set()

            first_resp, second_resp = await asyncio.gather(first_task, second_task)

        assert first_resp.status_code == 200
        assert second_resp.status_code == 200
        assert [call["message"] for call in composer.calls] == ["First", "Second"]
        assert composer.calls[0]["chat_messages"] == []
        assert composer.calls[1]["chat_messages"] == [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply to first"},
        ]


class TestLiteLLMErrorRedaction:
    """LiteLLM exception bodies must not leak provider internals.

    ``str(LiteLLMAuthError)`` / ``str(LiteLLMAPIError)`` includes the
    provider name, model ID, request payload fragments, and — on
    certain provider code paths — the upstream HTTP response body
    which has been observed to echo the ``Authorization`` header.
    The 502 HTTP body must therefore carry only the class name, not
    ``str(exc)``.  Paired send_message + recompose assertions pin
    the two mirror paths; any future divergence becomes a selective
    leak surface.
    """

    # Distinct, recognisably synthetic canaries per leakage surface.  Each
    # probes a different way LiteLLM's ``str(exc)`` could expose provider
    # internals:
    #
    # * ``_CANARY_MESSAGE_*`` — the ``message`` constructor argument, the
    #   most common leak vector (Authorization headers, request payload
    #   fragments, upstream response bodies).
    # * ``_CANARY_PROVIDER`` / ``_CANARY_MODEL`` — fields LiteLLM embeds
    #   in its ``__str__`` rendering; even though these are operator-
    #   chosen today, a future provider name that carries credentials
    #   (e.g. tenant-scoped Azure deployments) must not flow through.
    # * ``_CANARY_CAUSE`` — the ``__cause__`` chain from ``raise ... from``;
    #   a 502 body that serialises ``exc.__cause__`` / ``exc.__context__``
    #   would leak upstream DB URLs, credentials, or internal tracebacks
    #   that never appeared in the LiteLLM exception itself.  Mirror of
    #   the SQLAlchemy-side canary coverage (see
    #   ``test_recompose_convergence_save_failure_redacts_sqlalchemy_internals``).
    _CANARY_MESSAGE_TOKEN = "__CANARY_LITELLM_MSG_sk_leaked_token_abc123__"
    _CANARY_MESSAGE_AUTH_HEADER = "__CANARY_LITELLM_MSG_Authorization_Bearer__"
    _CANARY_MESSAGE_PAYLOAD = "__CANARY_LITELLM_MSG_request_payload_opaque__"
    _CANARY_PROVIDER = "__CANARY_LITELLM_PROVIDER_internal__"
    _CANARY_MODEL = "__CANARY_LITELLM_MODEL_secret_deployment__"
    _CANARY_CAUSE = "__CANARY_LITELLM_CAUSE_upstream_conn_str__"

    @classmethod
    def _canary_message(cls) -> str:
        """Assemble a message that packs every message-field canary.

        Kept as a single concatenated string so every canary rides
        through the same ``message`` constructor argument — the exact
        path the redaction contract is designed to sever.
        """
        return (
            f"Auth failed token={cls._CANARY_MESSAGE_TOKEN} header={cls._CANARY_MESSAGE_AUTH_HEADER} payload={cls._CANARY_MESSAGE_PAYLOAD}"
        )

    @classmethod
    def _all_canaries(cls) -> tuple[tuple[str, str], ...]:
        """Canary name → value pairs for the leak-surface sweep.

        Returned as ordered tuples so assertion failures identify the
        specific leak surface (e.g. ``__cause__`` chain vs ``model`` field)
        rather than a generic "something leaked" signal.
        """
        return (
            ("message.token", cls._CANARY_MESSAGE_TOKEN),
            ("message.auth_header", cls._CANARY_MESSAGE_AUTH_HEADER),
            ("message.payload", cls._CANARY_MESSAGE_PAYLOAD),
            ("llm_provider", cls._CANARY_PROVIDER),
            ("model", cls._CANARY_MODEL),
            ("__cause__", cls._CANARY_CAUSE),
        )

    def _make_auth_error(self) -> Exception:
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        exc = LiteLLMAuthError(
            message=self._canary_message(),
            llm_provider=self._CANARY_PROVIDER,
            model=self._CANARY_MODEL,
        )
        # ``raise ... from cause`` chained manually so the test can run
        # without an actual DB/network cause — what matters is that a
        # serialiser that walks ``__cause__`` will encounter the canary.
        exc.__cause__ = RuntimeError(f"upstream: {self._CANARY_CAUSE}")
        return exc

    def _make_api_error(self) -> Exception:
        from litellm.exceptions import APIError as LiteLLMAPIError

        exc = LiteLLMAPIError(
            status_code=503,
            message=self._canary_message(),
            llm_provider=self._CANARY_PROVIDER,
            model=self._CANARY_MODEL,
        )
        exc.__cause__ = RuntimeError(f"upstream: {self._CANARY_CAUSE}")
        return exc

    def _make_bad_request_error(self) -> Exception:
        from elspeth.web.composer.service import _BadRequestLLMError

        return _BadRequestLLMError(
            "LLM request rejected (BadRequestError)",
            provider_detail="Provider rejected composer prompt: model does not support temperature",
            provider_status_code=400,
        )

    def _assert_redacted(self, resp, expected_error_type: str, expected_exc_class: str) -> None:
        """Assert the 502 body is class-name-only and contains no canary strings."""
        assert resp.status_code == 502
        body = resp.json()
        detail = body["detail"]
        assert detail["error_type"] == expected_error_type
        # The ``detail`` field now carries ONLY the exception class name,
        # not the leaky ``str(exc)``.  Byte equality is the load-bearing
        # assertion — a substring check would admit "AuthenticationError:
        # Auth failed for provider=openai ..." which is precisely the
        # shape the redaction exists to prevent.
        assert detail["detail"] == expected_exc_class
        # Defence-in-depth: sweep every canary across the full serialised
        # body.  Per-surface failure messages so a regression names which
        # leak channel opened (``message`` field vs ``__cause__`` chain vs
        # ``model`` field) — mirrors the SQLAlchemy-side coverage in
        # ``test_recompose_convergence_save_failure_redacts_sqlalchemy_internals``.
        serialised = resp.text
        for surface, canary in self._all_canaries():
            assert canary not in serialised, (
                f"LiteLLM canary leaked into HTTP response body via "
                f"{surface!r} surface: {canary!r} appears in serialised 502 body. "
                "The redaction contract requires the body to carry only the "
                "exception class name — inspect the handler for a code path "
                "that re-introduced str(exc), exc.__cause__, or an individual "
                "exception field into the response."
            )

    def test_send_message_auth_error_body_carries_class_name_only(self, tmp_path) -> None:
        """LiteLLMAuthError from compose() → 502 with class-name-only detail."""
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_auth_error())

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello"},
        )
        self._assert_redacted(msg_resp, "llm_auth_error", "AuthenticationError")

    def test_send_message_api_error_body_carries_class_name_only(self, tmp_path) -> None:
        """LiteLLMAPIError from compose() → 502 with class-name-only detail."""
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_api_error())

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello"},
        )
        self._assert_redacted(msg_resp, "llm_unavailable", "APIError")

    def test_send_message_bad_request_provider_detail_is_exposed_when_enabled(self, tmp_path) -> None:
        """_BadRequestLLMError must use its dedicated provider-detail carrier at the route layer."""
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_bad_request_error())

        app, _ = _make_app(tmp_path)
        app.state.settings = app.state.settings.model_copy(update={"composer_expose_provider_errors": True})
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello"},
        )

        assert msg_resp.status_code == 502
        detail = msg_resp.json()["detail"]
        assert detail["error_type"] == "llm_unavailable"
        assert detail["detail"] == "_BadRequestLLMError"
        assert detail["provider_detail"] == "Provider rejected composer prompt: model does not support temperature"
        assert detail["provider_status_code"] == 400

    def test_api_error_debug_detail_is_exposed_when_enabled(self) -> None:
        """Opt-in debug mode exposes scrubbed provider detail for staging triage."""
        from litellm.exceptions import APIError as LiteLLMAPIError

        from elspeth.web.sessions.routes import _litellm_error_detail

        exc = LiteLLMAPIError(
            status_code=402,
            message="OpenRouter upstream rejected request: insufficient credits",
            llm_provider="openrouter",
            model="openai/gpt-5.5",
        )

        detail = _litellm_error_detail(
            "llm_unavailable",
            exc,
            expose_provider_error=True,
        )

        assert detail["error_type"] == "llm_unavailable"
        assert detail["detail"] == "APIError"
        assert detail["provider_status_code"] == 402
        assert detail["provider_detail"] == ("litellm.APIError: OpenRouter upstream rejected request: insufficient credits")

    def test_api_error_debug_detail_does_not_invoke_dynamic_status_code(self) -> None:
        """Third-party error status extraction must not invoke synthetic attributes."""
        from elspeth.web.sessions.routes import _litellm_error_detail

        class _DynamicStatusCodeError(Exception):
            @property
            def status_code(self) -> int:
                raise AssertionError("status_code property must not be invoked")

        detail = _litellm_error_detail(
            "llm_unavailable",
            _DynamicStatusCodeError("provider unavailable"),
            expose_provider_error=True,
        )

        assert detail["provider_detail"] == "provider unavailable"
        assert "provider_status_code" not in detail

    def test_api_error_debug_detail_scrubs_secret_values(self) -> None:
        """Debug detail must not turn provider errors into a secret leak channel."""
        from litellm.exceptions import APIError as LiteLLMAPIError

        from elspeth.web.sessions.routes import _litellm_error_detail

        secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"  # secret-scan: allow-this-line
        exc = LiteLLMAPIError(
            status_code=503,
            message=f"Provider echoed Authorization Bearer {secret}",
            llm_provider="openrouter",
            model="openai/gpt-5.5",
        )

        detail = _litellm_error_detail(
            "llm_unavailable",
            exc,
            expose_provider_error=True,
        )

        body_text = str(detail)
        assert secret not in body_text
        assert detail["provider_detail"] == ("Provider detail redacted because it may contain secrets.")

    def test_bad_request_llm_error_dedicated_attributes_consumed(self) -> None:
        """_BadRequestLLMError carries the raw provider text on a dedicated attribute.

        ``str(exc)`` on this class returns only the redacted wrap message
        ("LLM request rejected (BadRequestError)"), so the route helper must
        prefer ``exc.provider_detail`` / ``exc.provider_status_code``. Without
        this branch the attributes were dead infrastructure.
        """
        from elspeth.web.composer.service import _BadRequestLLMError
        from elspeth.web.sessions.routes import _litellm_error_detail

        exc = _BadRequestLLMError(
            "LLM request rejected (BadRequestError)",
            provider_detail="Anthropic API: prompt too long (200K token limit)",
            provider_status_code=400,
        )

        detail = _litellm_error_detail(
            "llm_unavailable",
            exc,
            expose_provider_error=True,
        )

        assert detail["error_type"] == "llm_unavailable"
        assert detail["detail"] == "_BadRequestLLMError"
        assert detail["provider_status_code"] == 400
        assert detail["provider_detail"] == "Anthropic API: prompt too long (200K token limit)"

    def test_bad_request_llm_error_attributes_scrubbed_for_secrets(self) -> None:
        """Provider text from _BadRequestLLMError must pass through the scrubber too."""
        from elspeth.web.composer.service import _BadRequestLLMError
        from elspeth.web.sessions.routes import _litellm_error_detail

        secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"  # secret-scan: allow-this-line
        exc = _BadRequestLLMError(
            "LLM request rejected (BadRequestError)",
            provider_detail=f"Provider echoed Authorization Bearer {secret}",
            provider_status_code=400,
        )

        detail = _litellm_error_detail(
            "llm_unavailable",
            exc,
            expose_provider_error=True,
        )

        assert secret not in str(detail)
        assert detail["provider_detail"] == "Provider detail redacted because it may contain secrets."

    def test_guided_source_commit_failure_requires_tool_result_contract(self) -> None:
        """The commit-failure helper must not inspect forged dynamic result shapes."""
        from elspeth.web.sessions.routes._helpers import _guided_source_commit_failure_detail

        class _ForgedToolResult:
            def __getattr__(self, name: str) -> object:
                raise AssertionError(f"dynamic attribute {name!r} must not be invoked")

        with pytest.raises(TypeError, match="ToolResult"):
            _guided_source_commit_failure_detail(_ForgedToolResult())

    def test_bad_request_llm_error_without_detail_yields_no_provider_fields(self) -> None:
        """When the carrier has no provider text, omit the optional fields rather than fabricating."""
        from elspeth.web.composer.service import _BadRequestLLMError
        from elspeth.web.sessions.routes import _litellm_error_detail

        exc = _BadRequestLLMError("LLM request rejected (BadRequestError)")

        detail = _litellm_error_detail(
            "llm_unavailable",
            exc,
            expose_provider_error=True,
        )

        assert detail["detail"] == "_BadRequestLLMError"
        assert "provider_detail" not in detail
        assert "provider_status_code" not in detail

    def test_recompose_auth_error_body_carries_class_name_only(self, tmp_path) -> None:
        """recompose path must mirror send_message's redaction."""
        import asyncio

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_auth_error())

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        # recompose precondition: last message must be user turn.
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a pipeline", writer_principal="route_user_message")
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")
        self._assert_redacted(recompose_resp, "llm_auth_error", "AuthenticationError")

    def test_recompose_api_error_body_carries_class_name_only(self, tmp_path) -> None:
        """recompose path must mirror send_message's redaction for APIError too."""
        import asyncio

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_api_error())

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a pipeline", writer_principal="route_user_message")
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")
        self._assert_redacted(recompose_resp, "llm_unavailable", "APIError")

    def test_recompose_bad_request_provider_detail_is_exposed_when_enabled(self, tmp_path) -> None:
        """recompose must mirror send_message for _BadRequestLLMError provider detail."""
        import asyncio

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(spec=ComposerService.compose, side_effect=self._make_bad_request_error())

        app, service = _make_app(tmp_path)
        app.state.settings = app.state.settings.model_copy(update={"composer_expose_provider_errors": True})
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a pipeline", writer_principal="route_user_message")
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert recompose_resp.status_code == 502
        detail = recompose_resp.json()["detail"]
        assert detail["error_type"] == "llm_unavailable"
        assert detail["detail"] == "_BadRequestLLMError"
        assert detail["provider_detail"] == "Provider rejected composer prompt: model does not support temperature"
        assert detail["provider_status_code"] == 400


class TestRecomposeConvergencePartialState:
    """Tests for partial state persistence on composer convergence failure."""

    def test_recompose_convergence_preserves_partial_state(self, tmp_path) -> None:
        """When recompose hits convergence error with partial state,
        the state is persisted and included in the 422 response."""
        import asyncio

        from elspeth.web.composer.protocol import ComposerConvergenceError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=2,  # > initial (1), so it's a real mutation
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        # Create session
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        # Simulate a failed send_message: user message saved, no assistant
        # response. This is the precondition for recompose — the last
        # message must be a user turn.
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a CSV pipeline", writer_principal="route_user_message")
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert recompose_resp.status_code == 422
        detail = recompose_resp.json()["detail"]
        assert detail["error_type"] == "convergence"
        assert "partial_state" in detail
        persisted_id, persisted_version = _read_persisted_state_identity(service, session_id)
        assert detail["partial_state"]["id"] == persisted_id
        assert detail["partial_state"]["version"] == persisted_version

    def test_recompose_convergence_without_partial_state(self, tmp_path) -> None:
        """When convergence error has no partial state (no mutations),
        response omits partial_state key."""
        import asyncio

        from elspeth.web.composer.protocol import ComposerConvergenceError

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=3,
                budget_exhausted="discovery",
                partial_state=None,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build something", writer_principal="route_user_message")
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert recompose_resp.status_code == 422
        detail = recompose_resp.json()["detail"]
        assert detail["error_type"] == "convergence"
        assert "partial_state" not in detail

    def test_convergence_response_body_carries_distinct_reason_for_each_budget(self, tmp_path) -> None:
        """The 422 body must carry a stable ``reason`` and ``recovery_text``
        per ``budget_exhausted`` value so the chat-side error UX cannot drift
        from the /composer-progress snapshot. Regression for the second-order
        risk in elspeth-5030f7373d (response/progress drift)."""
        from elspeth.web.composer.protocol import ComposerConvergenceError

        budgets_to_codes: dict[str, tuple[str, str]] = {
            "composition": ("convergence_composition_budget", "smaller turns"),
            "discovery": ("convergence_discovery_budget", "narrow"),
            "timeout": ("convergence_wall_clock_timeout", "wall-clock"),
        }

        for budget, (expected_reason, recovery_keyword) in budgets_to_codes.items():
            mock_composer = SimpleNamespace()
            mock_composer.compose = AsyncMock(
                spec=ComposerService.compose,
                side_effect=ComposerConvergenceError(
                    max_turns=5,
                    budget_exhausted=budget,  # type: ignore[arg-type]
                    partial_state=None,
                ),
            )
            app, _service = _make_app(tmp_path / f"convergence_body_{budget}")
            app.state.composer_service = mock_composer
            client = TestClient(app, raise_server_exceptions=False)

            resp = client.post("/api/sessions", json={"title": f"convergence-{budget}"})
            assert resp.status_code in (200, 201), resp.text
            session_id = resp.json()["id"]

            response = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "Build me a pipeline"},
            )

            assert response.status_code == 422, response.text
            detail = response.json()["detail"]
            assert detail["error_type"] == "convergence"
            assert detail["budget_exhausted"] == budget
            assert detail["reason"] == expected_reason, (
                f"422 body reason for budget={budget!r} must be {expected_reason!r}, got {detail.get('reason')!r}"
            )
            assert detail["recovery_text"], "422 body must include actionable recovery_text"
            assert recovery_keyword in detail["recovery_text"].lower(), (
                f"recovery_text for budget={budget!r} must mention {recovery_keyword!r}; got {detail['recovery_text']!r}"
            )

            progress_resp = client.get(f"/api/sessions/{session_id}/composer-progress")
            assert progress_resp.status_code == 200
            snapshot = progress_resp.json()
            assert snapshot["phase"] == "failed"
            assert snapshot["reason"] == expected_reason, (
                "Progress snapshot reason must match the 422 body reason — drift would "
                "re-introduce the elspeth-5030f7373d split-brain symptom at a different layer."
            )

    def test_convergence_redacts_blob_path_from_response_but_preserves_in_db(self, tmp_path) -> None:
        """When partial_state has a blob-backed source, the HTTP response must
        redact the internal storage path while the DB copy retains it."""
        import asyncio

        from elspeth.contracts.freeze import deep_freeze
        from elspeth.web.composer.protocol import ComposerConvergenceError
        from elspeth.web.composer.state import SourceSpec

        partial = CompositionState(
            source=SourceSpec(
                plugin="csv",
                options=deep_freeze(
                    {
                        "path": "/internal/blobs/data.csv",
                        "blob_ref": "abc123",
                        "schema": {"mode": "observed"},
                    }
                ),
                on_success="t1",
                on_validation_failure="quarantine",
            ),
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=2,
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        # Create session and seed a user message for recompose precondition
        resp = client.post("/api/sessions", json={"title": "Blob test"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Load my CSV", writer_principal="route_user_message"),
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert recompose_resp.status_code == 422
        detail = recompose_resp.json()["detail"]
        assert detail["error_type"] == "convergence"

        # HTTP response: path key remains contract-visible, but the internal
        # storage value must be redacted.
        response_source_opts = detail["partial_state"]["sources"]["source"]["options"]
        assert response_source_opts["path"] == REDACTED_BLOB_SOURCE_PATH
        assert response_source_opts["blob_ref"] == "abc123"

        # DB copy: path must be preserved alongside blob_ref
        loop = asyncio.new_event_loop()
        db_record = loop.run_until_complete(
            service.get_current_state(uuid.UUID(session_id)),
        )
        loop.close()

        assert db_record is not None
        db_source_opts = db_record.sources["source"]["options"]
        assert db_source_opts["path"] == "/internal/blobs/data.csv"
        assert db_source_opts["blob_ref"] == "abc123"

    def test_recompose_convergence_save_operational_error_preserves_422_body(self, tmp_path) -> None:
        """Regression (elspeth-303f751204): when save_composition_state
        raises an ``OperationalError`` (lock timeout / pool disconnect)
        while persisting partial_state from a convergence error, the
        handler MUST still return the structured 422 body with
        ``partial_state_save_failed=True`` rather than upgrading the
        user-driven 422 to an uncaught 500.

        Before the fix, the handler's ``except`` clause caught only
        ``IntegrityError``; any other ``SQLAlchemyError`` subclass escaped
        and the user received a generic 500 with no structured diagnostic.
        """
        import asyncio

        from sqlalchemy.exc import OperationalError

        from elspeth.web.composer.protocol import ComposerConvergenceError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=2,
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        async def _raise_operational(*_args, **_kwargs):
            raise OperationalError(
                "INSERT INTO composition_states ...",
                {},
                Exception("server has gone away"),
            )

        service.save_composition_state = _raise_operational  # type: ignore[method-assign]

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a pipeline", writer_principal="route_user_message"),
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        # Structured 422 body is preserved despite the secondary save failure.
        assert recompose_resp.status_code == 422
        detail = recompose_resp.json()["detail"]
        assert detail["error_type"] == "convergence"
        assert detail["partial_state_save_failed"] is True
        # partial_state is not populated on save failure (no successful row).
        assert "partial_state" not in detail

    def test_recompose_convergence_save_failure_redacts_sqlalchemy_internals(self, tmp_path) -> None:
        """Regression: the 422 body's ``partial_state_save_error`` field must
        carry only the exception class name — never SQL statements, parameter
        tuples, or ``__cause__`` text. ``str(SQLAlchemyError)`` includes
        ``[SQL: ...]`` and ``[parameters: ...]`` which on this code path carry
        the composition-state JSON payload (potential secret refs); on
        ``OperationalError`` the ``__cause__`` message can carry DB URLs or
        credentials. The slog side already redacts
        (``exc_class=type(save_err).__name__``); the HTTP response body must
        match that redaction so a 422 cannot become a leak channel.

        Paired with the sibling ``_handle_plugin_crash`` contract (same file),
        which emits no response-body diagnostic at all.
        """
        import asyncio

        from sqlalchemy.exc import OperationalError

        from elspeth.web.composer.protocol import ComposerConvergenceError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=2,
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        # Canary strings chosen to be recognisably synthetic — if any appear
        # anywhere in the JSON response body, redaction failed.
        sql_canary = "__CANARY_SQL_INSERT_composition_states_source_options__"
        params_canary = "__CANARY_PARAM_secret_ref_opaque_token__"
        cause_canary = "__CANARY_CAUSE_postgresql_conn_str_password__"

        async def _raise_operational(*_args, **_kwargs):
            raise OperationalError(
                f"INSERT INTO composition_states (id, session_id, source) VALUES (...) -- {sql_canary}",
                {"source_options": params_canary},
                Exception(f"connection closed: {cause_canary}"),
            )

        service.save_composition_state = _raise_operational  # type: ignore[method-assign]

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Redaction"})
        session_id = resp.json()["id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build a pipeline", writer_principal="route_user_message"),
        )
        loop.close()

        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert recompose_resp.status_code == 422
        body_text = recompose_resp.text
        # Primary assertion: no canary substring anywhere in the serialised body.
        assert sql_canary not in body_text, "SQL statement leaked into HTTP response body"
        assert params_canary not in body_text, "parameter tuple leaked into HTTP response body"
        assert cause_canary not in body_text, "DBAPI __cause__ text leaked into HTTP response body"

        detail = recompose_resp.json()["detail"]
        # The boolean signal is preserved so clients can still distinguish
        # "partial state saved" from "partial state lost to a save failure".
        assert detail["partial_state_save_failed"] is True
        # The diagnostic field carries ONLY the exception class name.
        assert detail.get("partial_state_save_error") == "OperationalError"

    def test_send_message_convergence_threads_user_id_to_preflight(self, tmp_path) -> None:
        """I3 regression: _handle_convergence_error MUST pass the authenticated
        user_id to _state_data_from_composer_state so the runtime preflight
        on the partial state can resolve user-scoped secret_refs.

        Failure mode (pre-fix): user_id=None means
        web/execution/validation.py:248 skips the secret-ref validation/
        resolution block entirely. Unresolved {secret_ref: ...} dicts
        flow into plugin instantiation, where typical plugin code like
        ``config["api_key"].lower()`` raises AttributeError. That
        AttributeError is not in validate_pipeline's typed catches, so
        it escapes _state_data_from_composer_state's persist_invalid
        guard, gets wrapped as _RuntimePreflightFailed("AttributeError"),
        and the persisted audit row carries the bare
        ``["runtime_preflight_failed"]`` sentinel — true but
        uninformative for the operator triaging the failure.

        Fix verifies the structural change: _state_data_from_composer_state
        is called with user_id="alice" (the authenticated user) instead
        of user_id=None.
        """
        from elspeth.web.composer.protocol import ComposerConvergenceError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="convergence-with-secrets"),
            version=2,
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, _service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        # Capture the user_id forwarded into _state_data_from_composer_state.
        # The convergence handler is the only caller routed through this
        # patch on the convergence path, so any captured invocation comes
        # from _handle_convergence_error.
        captured_user_ids: list[Any] = []

        original_fn = None

        async def capture_user_id(state, **kwargs):
            captured_user_ids.append(kwargs.get("user_id"))
            assert original_fn is not None
            return await original_fn(state, **kwargs)

        from elspeth.web.sessions.routes import _helpers as routes_module

        original_fn = routes_module._state_data_from_composer_state
        with patch(
            "elspeth.web.sessions.routes._helpers._state_data_from_composer_state",
            side_effect=capture_user_id,
        ):
            resp = client.post("/api/sessions", json={"title": "Convergence + secrets"})
            session_id = resp.json()["id"]
            convergence_resp = client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "Build a pipeline with secrets"},
            )

        assert convergence_resp.status_code == 422
        assert convergence_resp.json()["detail"]["error_type"] == "convergence"
        # The convergence handler MUST have invoked _state_data_from_composer_state
        # with the authenticated user_id, not None. Without this, validation.py:248
        # skips secret-ref resolution and downstream plugin code crashes on
        # unresolved {secret_ref: ...} dicts.
        assert captured_user_ids == ["alice"], (
            "_handle_convergence_error must thread the authenticated user_id "
            "to _state_data_from_composer_state so runtime preflight can "
            f"resolve user-scoped secrets. captured_user_ids={captured_user_ids}"
        )

    def test_recompose_convergence_threads_user_id_to_preflight(self, tmp_path) -> None:
        """Recompose mirror of test_send_message_convergence_threads_user_id_to_preflight.

        The convergence handler is shared by both endpoints, so the I3
        wiring change must take effect on both. Asserting the recompose
        path independently catches a regression that would re-introduce
        user_id=None on one of the two call sites only.
        """
        import asyncio

        from elspeth.web.composer.protocol import ComposerConvergenceError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="recompose-convergence-with-secrets"),
            version=2,
        )

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerConvergenceError(
                max_turns=5,
                budget_exhausted="composition",
                partial_state=partial,
            ),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Recompose convergence + secrets"})
        session_id = uuid.UUID(resp.json()["id"])

        # Recompose precondition: last persisted message must be a user turn.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline", writer_principal="route_user_message"))
        finally:
            loop.close()

        captured_user_ids: list[Any] = []
        original_fn = None

        async def capture_user_id(state, **kwargs):
            captured_user_ids.append(kwargs.get("user_id"))
            assert original_fn is not None
            return await original_fn(state, **kwargs)

        from elspeth.web.sessions.routes import _helpers as routes_module

        original_fn = routes_module._state_data_from_composer_state
        with patch(
            "elspeth.web.sessions.routes._helpers._state_data_from_composer_state",
            side_effect=capture_user_id,
        ):
            convergence_resp = client.post(f"/api/sessions/{session_id}/recompose")

        assert convergence_resp.status_code == 422
        assert convergence_resp.json()["detail"]["error_type"] == "convergence"
        assert captured_user_ids == ["alice"], (
            "_handle_convergence_error from the recompose path must thread "
            "the authenticated user_id to _state_data_from_composer_state. "
            f"captured_user_ids={captured_user_ids}"
        )


class TestStateRoutes:
    """Tests for composition state endpoints."""

    def test_get_state_empty(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Empty"})
        session_id = resp.json()["id"]

        state_resp = client.get(f"/api/sessions/{session_id}/state")
        assert state_resp.status_code == 200
        assert state_resp.json() is None

    def test_get_state_versions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Pipeline"})
        session_id = resp.json()["id"]

        versions_resp = client.get(
            f"/api/sessions/{session_id}/state/versions",
        )
        assert versions_resp.status_code == 200
        assert versions_resp.json() == []


class TestGuidedBootstrapStateVersions:
    """Regression coverage for guided bootstrap not polluting state history."""

    def test_get_guided_does_not_persist_empty_initial_state(self, tmp_path) -> None:
        """Auto-starting guided mode must not create an empty v1 graph.

        The frontend calls GET /guided automatically when a session is created
        or selected. That read path may return the deterministic first turn,
        but it must not allocate a composition_state version before the user
        submits an actual guided response.
        """
        app, service = _make_app(tmp_path)
        catalog = MagicMock(spec=CatalogService)
        catalog.list_sources.return_value = []
        app.state.catalog_service = catalog
        app.state.session_engine = service._engine
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Guided"})
        session_id = resp.json()["id"]

        guided_resp = client.get(f"/api/sessions/{session_id}/guided")
        assert guided_resp.status_code == 200
        guided_body = guided_resp.json()
        assert guided_body["next_turn"] is not None
        assert guided_body["guided_session"]["history"] == []
        assert guided_body["composition_state"] is None

        state_resp = client.get(f"/api/sessions/{session_id}/state")
        assert state_resp.status_code == 200
        assert state_resp.json() is None

        versions_resp = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions_resp.status_code == 200
        assert versions_resp.json() == []

        respond_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={"control_signal": "exit_to_freeform"},
        )
        assert respond_resp.status_code == 200
        respond_body = respond_resp.json()
        assert respond_body["composition_state"]["version"] == 1
        assert respond_body["composition_state"]["composer_meta"]["guided_session"]["transition_consumed"] is True

        versions_after_resp = client.get(f"/api/sessions/{session_id}/state/versions")
        assert versions_after_resp.status_code == 200
        assert [version["version"] for version in versions_after_resp.json()] == [1]


class TestRevertEndpoint:
    """Tests for POST /api/sessions/{id}/state/revert (R1)."""

    @pytest.mark.asyncio
    async def test_revert_creates_new_version(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        # Create session and two state versions via the service
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "csv"}, is_valid=True), provenance="session_seed"
        )
        await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "api"}, is_valid=True), provenance="session_seed"
        )

        # Revert to v1
        operation_id = str(uuid.uuid4())
        resp = client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"operation_id": operation_id, "state_id": str(v1.id)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == 3
        # Should match v1's source, not v2's
        assert body["sources"] == {"source": {"type": "csv"}}
        # Lineage: new version derives from v1
        assert body["derived_from_state_id"] == str(v1.id)

        replay = client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"operation_id": operation_id, "state_id": str(v1.id)},
        )
        assert replay.status_code == 200
        assert replay.json() == body
        versions = await service.get_state_versions(session.id)
        assert [state.version for state in versions] == [1, 2, 3]
        messages = await service.get_messages(session.id, limit=None)
        assert [message.content for message in messages] == ["Pipeline reverted to version 1."]

    @pytest.mark.asyncio
    async def test_revert_injects_system_message(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")

        client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"operation_id": str(uuid.uuid4()), "state_id": str(v1.id)},
        )

        # Check that a system message was injected
        msgs_resp = client.get(f"/api/sessions/{session.id}/messages")
        messages = msgs_resp.json()
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Pipeline reverted to version 1."

    @pytest.mark.asyncio
    async def test_revert_idor_protection(self, tmp_path) -> None:
        """Revert to a state in another user's session returns 404."""
        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        service = SessionServiceImpl(
            engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )

        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)

            async def mock_user():
                return identity

            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = WebSettings(
                data_dir=tmp_path,
                composer_max_composition_turns=15,
                composer_max_discovery_turns=10,
                composer_timeout_seconds=85.0,
                composer_rate_limit_per_minute=10,
                shareable_link_signing_key=b"\x00" * 32,
            )

            from elspeth.web.middleware.rate_limit import ComposerRateLimiter

            app.state.rate_limiter = ComposerRateLimiter(limit=100)
            app.state.composer_progress_registry = ComposerProgressRegistry()
            app.include_router(create_session_router())
            return app

        bob_app = make_app_for_user("bob")
        bob_client = TestClient(bob_app)

        # Alice creates a session with a state
        session = await service.create_session("alice", "Alice Only", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")

        # Bob tries to revert -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"operation_id": str(uuid.uuid4()), "state_id": str(v1.id)},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_revert_state_not_belonging_to_session(self, tmp_path) -> None:
        """Revert with a state_id from a different session returns 404."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        s1 = await service.create_session("alice", "Session 1", "local")
        s2 = await service.create_session("alice", "Session 2", "local")
        v1_s2 = await service.save_composition_state(s2.id, CompositionStateData(is_valid=True), provenance="session_seed")

        # Try to revert s1 using s2's state -- should fail
        resp = client.post(
            f"/api/sessions/{s1.id}/state/revert",
            json={"operation_id": str(uuid.uuid4()), "state_id": str(v1_s2.id)},
        )
        assert resp.status_code == 404


class TestYamlEndpoint:
    """Tests for GET /api/sessions/{id}/state/yaml."""

    @pytest.mark.asyncio
    async def test_post_state_yaml_with_disabled_plugin_is_atomic(self, tmp_path: Path) -> None:
        app, service = _make_app(tmp_path)
        _install_restricted_plugin_policy(app, PluginId("sink", "database"))
        client = TestClient(app)
        session = await service.create_session("alice", "Policy atomicity", "local")
        before = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True),
            provenance="session_seed",
        )
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      schema:
        mode: observed
    on_validation_failure: discard
sinks:
  main:
    plugin: database
    options: {}
    on_write_failure: discard
"""

        response = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert response.status_code == 422
        assert response.json()["detail"]["error_code"] == "plugin_not_enabled"
        after = await service.get_current_state(session.id)
        assert after is not None
        assert after.id == before.id
        assert after.version == before.version

    @pytest.mark.asyncio
    async def test_saved_disabled_state_is_readable_and_exported_as_authored(self, tmp_path: Path) -> None:
        app, service = _make_app(tmp_path)
        snapshot = _install_restricted_plugin_policy(app, PluginId("transform", "llm"))
        app.state.operator_profile_registry.lower_options.side_effect = AssertionError("export must not lower private bindings")
        client = TestClient(app)
        session = await service.create_session("alice", "Historical disabled plugin", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={
                    "source": {
                        "plugin": "csv",
                        "on_success": "score",
                        "options": {"schema": {"mode": "observed"}},
                        "on_validation_failure": "discard",
                    }
                },
                nodes=[
                    {
                        "id": "score",
                        "node_type": "transform",
                        "plugin": "llm",
                        "input": "source",
                        "on_success": "main",
                        "on_error": "discard",
                        "options": {
                            "profile": "task-role",
                            "prompt_template": "Score {{ row }}",
                            "schema": {"mode": "observed"},
                        },
                    }
                ],
                outputs=[
                    {
                        "name": "main",
                        "plugin": "json",
                        "options": {"path": "outputs/scored.jsonl"},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Historical disabled plugin", "description": None},
                is_valid=False,
            ),
            provenance="session_seed",
        )

        state_response = client.get(f"/api/sessions/{session.id}/state")
        export_response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert state_response.status_code == 200
        body = state_response.json()
        assert body["nodes"][0]["plugin"] == "llm"
        assert body["plugin_policy_findings"] == [
            {
                "component_id": "score",
                "plugin_id": "transform:llm",
                "reason_code": "plugin_not_enabled",
                "snapshot_fingerprint": snapshot.snapshot_hash,
            }
        ]
        assert export_response.status_code == 200
        exported_yaml = export_response.json()["yaml"]
        assert "plugin: llm" in exported_yaml
        assert "profile: task-role" in exported_yaml
        assert "bedrock" not in exported_yaml
        assert "credential" not in exported_yaml

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_component", ["source", "sink"])
    async def test_post_state_yaml_persists_aws_s3_endpoint_url_as_invalid(
        self,
        tmp_path: Path,
        invalid_component: str,
    ) -> None:
        endpoint_sentinel = "https://yaml-canary.attacker.invalid/private"
        source_options = {"endpoint_url": endpoint_sentinel} if invalid_component == "source" else {}
        sink_options = {"endpoint_url": endpoint_sentinel} if invalid_component == "sink" else {}
        yaml_text = yaml.safe_dump(
            {
                "sources": {
                    "source": {
                        "plugin": "aws_s3" if invalid_component == "source" else "csv",
                        "on_success": "main",
                        "options": source_options,
                        "on_validation_failure": "discard",
                    }
                },
                "sinks": {
                    "main": {
                        "plugin": "aws_s3" if invalid_component == "sink" else "json",
                        "options": sink_options,
                        "on_write_failure": "discard",
                    }
                },
            }
        )
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "AWS S3 policy import", "local")

        response = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_valid"] is False
        assert body["validation_errors"] == [AWS_S3_ENDPOINT_URL_POLICY_ERROR]
        assert endpoint_sentinel not in repr(body["validation_errors"])
        record = await service.get_current_state(session.id)
        assert record is not None
        assert record.is_valid is False
        assert list(record.validation_errors or ()) == [AWS_S3_ENDPOINT_URL_POLICY_ERROR]
        assert endpoint_sentinel not in repr(record.validation_errors)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_component", ["source", "sink"])
    async def test_e2e_seed_persists_aws_s3_endpoint_url_as_invalid(
        self,
        tmp_path: Path,
        invalid_component: str,
    ) -> None:
        endpoint_sentinel = "https://seed-canary.attacker.invalid/private"
        source = SourceSpec(
            plugin="aws_s3" if invalid_component == "source" else "csv",
            on_success="main",
            options={"endpoint_url": endpoint_sentinel} if invalid_component == "source" else {},
            on_validation_failure="discard",
        )
        output = OutputSpec(
            name="main",
            plugin="aws_s3" if invalid_component == "sink" else "json",
            options={"endpoint_url": endpoint_sentinel} if invalid_component == "sink" else {},
            on_write_failure="discard",
        )
        seeded_state = CompositionState(
            source=source,
            nodes=(),
            edges=(),
            outputs=(output,),
            metadata=PipelineMetadata(name="AWS S3 policy seed"),
            version=1,
        )
        app, service = _make_app(tmp_path)
        app.state.settings = app.state.settings.model_copy(update={"e2e_state_seed_enabled": True})
        client = TestClient(app)
        session = await service.create_session("alice", "AWS S3 policy seed", "local")

        response = client.post(
            f"/api/sessions/{session.id}/state/e2e-seed",
            json={"state": seeded_state.to_dict()},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["is_valid"] is False
        assert body["validation_errors"] == [AWS_S3_ENDPOINT_URL_POLICY_ERROR]
        assert endpoint_sentinel not in repr(body["validation_errors"])
        record = await service.get_current_state(session.id)
        assert record is not None
        assert record.is_valid is False
        assert list(record.validation_errors or ()) == [AWS_S3_ENDPOINT_URL_POLICY_ERROR]
        assert endpoint_sentinel not in repr(record.validation_errors)

    @pytest.mark.asyncio
    async def test_post_state_yaml_imports_exported_runtime_yaml(self, tmp_path) -> None:
        """Replay can seed a fresh session from captured final_yaml.json."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        blob_id = uuid.uuid4()
        blob_path = tmp_path / "blobs" / str(session.id) / f"{blob_id}_input.csv"
        blob_path.parent.mkdir(parents=True)
        blob_path.write_text("id\n1\n")
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.get_blob.return_value = SimpleNamespace(
            id=blob_id,
            session_id=session.id,
            storage_path=str(blob_path),
        )
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: /old/blob.csv
      schema:
        mode: observed
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
      schema:
        mode: observed
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(
                f"/api/sessions/{session.id}/state/yaml",
                json={"yaml": yaml_text, "source_blob_ids": {"source": str(blob_id)}},
            )

        assert resp.status_code == 200, resp.text
        record = await service.get_current_state(session.id)
        assert record is not None
        assert record.sources["source"]["plugin"] == "csv"
        assert record.sources["source"]["options"]["path"] == str(blob_path)
        assert record.outputs[0]["name"] == "main"

    @pytest.mark.asyncio
    async def test_post_state_yaml_surfaces_llm_review_events(self, tmp_path) -> None:
        """Importing YAML with an llm node must surface resolvable pending
        interpretation EVENTS, not just fail-closed requirements
        (elspeth-ae5160c3cb). Without the events the run gate blocks while no
        review card renders and nothing can resolve the block."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: score
    options:
      schema:
        mode: observed
transforms:
- name: score
  plugin: llm
  input: source
  on_success: main
  on_error: discard
  options:
    model: anthropic/claude-haiku-4.5
    prompt_template: 'Score this: {{ row.value }}'
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 200, resp.text
        record = await service.get_current_state(session.id)
        assert record is not None

        events = await service.list_interpretation_events(session.id, status="pending")
        by_kind = {event.kind: event for event in events}
        assert set(by_kind) == {InterpretationKind.LLM_PROMPT_TEMPLATE, InterpretationKind.LLM_MODEL_CHOICE}
        pt = by_kind[InterpretationKind.LLM_PROMPT_TEMPLATE]
        assert pt.affected_node_id == "score"
        assert pt.llm_draft == "Score this: {{ row.value }}"
        assert pt.user_term == "llm_prompt_template:score"
        assert str(pt.composition_state_id) == str(record.id)
        assert pt.tool_call_id is not None and pt.tool_call_id.startswith("backend_auto_surface:")
        assert pt.model_identifier == "yaml_import"
        assert pt.model_version == "yaml_import"
        assert pt.provider == "yaml_import"
        assert pt.composer_skill_hash == "yaml_import"
        mc = by_kind[InterpretationKind.LLM_MODEL_CHOICE]
        assert mc.affected_node_id == "score"
        assert mc.llm_draft == "anthropic/claude-haiku-4.5"
        assert mc.user_term == "llm_model_choice:score"

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_malformed_interpretation_requirements(self, tmp_path) -> None:
        """Hand-written interpretation_requirements rows the schema would
        refuse are rejected 400 before persistence (elspeth-ae5160c3cb)."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
transforms:
- name: score
  plugin: llm
  input: source
  on_success: main
  on_error: discard
  options:
    model: anthropic/claude-haiku-4.5
    prompt_template: 'Score this: {{ row.value }}'
    interpretation_requirements:
    - kind: not_a_kind
      user_term: x
      status: pending
"""
        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})
        assert resp.status_code == 400
        assert "score" in resp.json()["detail"]
        assert await service.get_current_state(session.id) is None

    @pytest.mark.asyncio
    async def test_post_state_yaml_reviews_are_resolvable_and_unblock_execution(self, tmp_path) -> None:
        """The defect in elspeth-ae5160c3cb was an UNRESOLVABLE block: the run
        gate held while nothing existed to resolve. Lock in the full loop —
        import, accept both surfaced cards, requirements patch to resolved,
        no pending events remain, and the interpretation run gate clears."""
        from elspeth.web.interpretation_state import InterpretationReviewPending, materialize_state_for_execution
        from elspeth.web.sessions.routes._helpers import _state_from_record

        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: score
    options:
      schema:
        mode: observed
transforms:
- name: score
  plugin: llm
  input: source
  on_success: main
  on_error: discard
  options:
    model: anthropic/claude-haiku-4.5
    prompt_template: 'Score this: {{ row.value }}'
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})
        assert resp.status_code == 200, resp.text

        events = await service.list_interpretation_events(session.id, status="pending")
        assert len(events) == 2
        for event in events:
            resolve_resp = client.post(
                f"/api/sessions/{session.id}/interpretations/{event.id}/resolve",
                json={"choice": "accepted_as_drafted"},
            )
            assert resolve_resp.status_code == 200, resolve_resp.text

        assert await service.list_interpretation_events(session.id, status="pending") == []
        record = await service.get_current_state(session.id)
        assert record is not None
        node = next(n for n in record.nodes if n["id"] == "score")
        statuses = {r["kind"]: r["status"] for r in node["options"]["interpretation_requirements"]}
        assert statuses == {"llm_prompt_template": "resolved", "llm_model_choice": "resolved"}

        gate_result = materialize_state_for_execution(_state_from_record(record))
        assert not isinstance(gate_result, InterpretationReviewPending)

    @pytest.mark.asyncio
    async def test_post_state_yaml_reimport_does_not_duplicate_pending_reviews(self, tmp_path) -> None:
        """Re-importing the same YAML must not surface twin pending events
        (elspeth-1fcaec9b63). The resolve path demands exactly one pending
        requirement per (node, kind, user_term), so a duplicate event is
        permanently unresolvable once its twin resolves: the card 422s with
        interpretation_placeholder_unavailable in a loop reload never clears."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: score
    options:
      schema:
        mode: observed
transforms:
- name: score
  plugin: llm
  input: source
  on_success: main
  on_error: discard
  options:
    model: anthropic/claude-haiku-4.5
    prompt_template: 'Score this: {{ row.value }}'
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            for _ in range(2):
                resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})
                assert resp.status_code == 200, resp.text

        events = await service.list_interpretation_events(session.id, status="pending")
        kinds = sorted(event.kind.value for event in events)
        assert kinds == ["llm_model_choice", "llm_prompt_template"], kinds

        for event in events:
            resolve_resp = client.post(
                f"/api/sessions/{session.id}/interpretations/{event.id}/resolve",
                json={"choice": "accepted_as_drafted"},
            )
            assert resolve_resp.status_code == 200, resolve_resp.text

        assert await service.list_interpretation_events(session.id, status="pending") == []

    @pytest.mark.asyncio
    async def test_post_state_yaml_allows_wired_secret_ref_marker(self, tmp_path) -> None:
        """Hardening (T-1) no-false-positive lock-in: a legitimately wired
        {secret_ref: NAME} marker in a credential field -- the form export
        itself produces -- must import cleanly, not trip the new
        fabricated-literal-credential rejection."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      api_key: {secret_ref: OPENAI_API_KEY}
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 200, resp.text
        record = await service.get_current_state(session.id)
        assert record is not None
        assert record.sources["source"]["options"]["api_key"] == {"secret_ref": "OPENAI_API_KEY"}

    @pytest.mark.asyncio
    async def test_post_state_yaml_allows_structural_key_literal(self, tmp_path) -> None:
        """Regression (elspeth-61f2c0732e): a literal ``data_key`` — the JSON
        source's structural extraction key — must import cleanly rather than
        tripping the fabricated-literal-credential rejection via the bare
        ``_key`` suffix heuristic."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: json
    on_success: main
    options:
      data_key: results
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 200, resp.text
        record = await service.get_current_state(session.id)
        assert record is not None
        assert record.sources["source"]["options"]["data_key"] == "results"

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_blob_storage_path_without_sidecar(self, tmp_path) -> None:
        """Path-only imports must not bypass source blob ownership checks."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        blob_path = tmp_path / "blobs" / "other-session" / "old.csv"
        blob_path.parent.mkdir(parents=True)
        blob_path.write_text("id\n1\n")
        yaml_text = f"""
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: {blob_path}
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 400
        assert "source_blob_ids" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_source_path_outside_allowed_directories(self, tmp_path) -> None:
        """Imports must fail before persistence when a source path cannot run."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  primary:
    plugin: json
    on_success: out
    options:
      path: examples/json_explode/input.json
      on_validation_failure: discard
sinks:
  out:
    plugin: json
    on_write_failure: discard
"""

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 400
        assert (
            resp.json()["detail"]
            == "Path traversal blocked: source 'primary' path='examples/json_explode/input.json' resolves outside allowed directories"
        )
        assert await service.get_current_state(session.id) is None

    @pytest.mark.asyncio
    async def test_post_state_yaml_remaps_source_blob_ids_to_owned_blob(self, tmp_path) -> None:
        """Replay imports bind captured source blobs to the newly uploaded blob row."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        blob_id = uuid.uuid4()
        blob_path = tmp_path / "blobs" / str(session.id) / f"{blob_id}_input.csv"
        blob_path.parent.mkdir(parents=True)
        blob_path.write_text("id\n1\n")
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.get_blob.return_value = SimpleNamespace(
            id=blob_id,
            session_id=session.id,
            storage_path=str(blob_path),
        )
        old_blob_path = tmp_path / "blobs" / "old-session" / "old.csv"
        yaml_text = f"""
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: {old_blob_path}
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(
                f"/api/sessions/{session.id}/state/yaml",
                json={"yaml": yaml_text, "source_blob_ids": {"source": str(blob_id)}},
            )

        assert resp.status_code == 200, resp.text
        app.state.blob_service.get_blob.assert_awaited_once_with(blob_id)
        record = await service.get_current_state(session.id)
        assert record is not None
        source_options = record.sources["source"]["options"]
        assert source_options["blob_ref"] == str(blob_id)
        assert source_options["path"] == str(blob_path)

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_unknown_source_blob_sidecar_entry(self, tmp_path) -> None:
        """source_blob_ids cannot bind blobs to sources absent from the YAML."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: /old/blob.csv
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        resp = client.post(
            f"/api/sessions/{session.id}/state/yaml",
            json={"yaml": yaml_text, "source_blob_ids": {"other": str(uuid.uuid4())}},
        )

        assert resp.status_code == 400
        assert "unknown source" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_malformed_source_blob_id(self, tmp_path) -> None:
        """source_blob_ids values must be UUIDs before any blob lookup."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: /old/blob.csv
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        resp = client.post(
            f"/api/sessions/{session.id}/state/yaml",
            json={"yaml": yaml_text, "source_blob_ids": {"source": "not-a-uuid"}},
        )

        assert resp.status_code == 400
        assert "must be a UUID" in resp.json()["detail"]
        app.state.blob_service.get_blob.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_cross_session_source_blob(self, tmp_path) -> None:
        """Replay sidecars cannot attach a blob owned by another session."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        other_session_id = uuid.uuid4()
        blob_id = uuid.uuid4()
        app.state.blob_service = MagicMock(spec=BlobServiceProtocol)
        app.state.blob_service.get_blob.return_value = SimpleNamespace(
            id=blob_id,
            session_id=other_session_id,
            storage_path=str(tmp_path / "blobs" / str(other_session_id) / "input.csv"),
        )
        yaml_text = """
sources:
  source:
    plugin: csv
    on_success: main
    options:
      path: /old/blob.csv
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        resp = client.post(
            f"/api/sessions/{session.id}/state/yaml",
            json={"yaml": yaml_text, "source_blob_ids": {"source": str(blob_id)}},
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Blob not found"
        app.state.blob_service.get_blob.assert_awaited_once_with(blob_id)

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_oversized_document(self, tmp_path) -> None:
        """Hardening (T-1): a paste over the Pydantic body cap is rejected as 422,
        and (production parity) the oversized content is not echoed back in the
        error body.

        The egress redaction is a global ``RequestValidationError`` handler in
        ``web/app.py`` (``handle_validation_error`` -- allowlists only
        type/loc/msg, dropping FastAPI's default ``input`` echo) rather than
        anything endpoint-specific, so it is registered here explicitly:
        ``_make_app`` builds a bare ``FastAPI()`` without the production
        app's exception handlers wired up.
        """
        app, service = _make_app(tmp_path)

        _SAFE_VALIDATION_ERROR_KEYS = frozenset({"type", "loc", "msg"})

        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse

        @app.exception_handler(RequestValidationError)
        async def _handle_validation_error(request, exc: RequestValidationError) -> JSONResponse:
            safe_errors = [{k: v for k, v in error.items() if k in _SAFE_VALIDATION_ERROR_KEYS} for error in exc.errors()]
            return JSONResponse(status_code=422, content={"detail": safe_errors})

        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        oversized = "sources:\n  source:\n    plugin: csv\n" + ("x" * 300_000)

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": oversized})

        assert resp.status_code == 422
        assert "x" * 100 not in resp.text

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_malformed_yaml_syntax(self, tmp_path) -> None:
        """Hardening (T-1): non-YAML paste is a categorized 400, never a 500,
        and the error body does not echo the pasted content or raw parser prose."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        not_yaml = "sources: [unterminated\n  plugin: csv"

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": not_yaml})

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail == "YAML parse failed: ParserError"
        assert "unterminated" not in resp.text
        assert "plugin" not in resp.text

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_non_pipeline_mapping(self, tmp_path) -> None:
        """Hardening (T-1): a syntactically valid YAML mapping that names no
        pipeline section must not silently import as an empty composition --
        that would be a silent destructive replace of the session's prior work."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        not_a_pipeline = "shopping_list:\n  - milk\n  - eggs\nnotes: just some random yaml\n"

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": not_a_pipeline})

        assert resp.status_code == 400
        assert "must define at least one pipeline section" in resp.json()["detail"]
        # The session's current state must remain unset -- nothing was persisted.
        assert await service.get_current_state(session.id) is None

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_aliased_yaml(self, tmp_path) -> None:
        """Hardening (T-1): anchors/aliases are rejected outright (billion-laughs
        defense) rather than silently expanded server-side."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        aliased = """
sources:
  source: &src
    plugin: csv
    on_success: out
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
sinks:
  out:
    plugin: csv
    on_write_failure: discard
also: *src
"""

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": aliased})

        assert resp.status_code == 400
        assert resp.json()["detail"].startswith("YAML parse failed: ")

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_literal_credential_value(self, tmp_path) -> None:
        """Hardening (T-1): a pasted literal credential (not a {secret_ref: ...}
        marker) in a credential-bearing field is rejected outright, before any
        persistence -- unlike the tool-call composer surface, which only
        catches this at /validate or runtime-preflight time (after the value
        is already written into CompositionState), pasted YAML has no prior
        tool-call gate, so it would otherwise reach save_composition_state and
        get echoed back in the response verbatim. The error names the field
        only, never the value (parity with the runtime-preflight
        fabricated_secret discipline's audit hygiene)."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        secret_value = "sk-live-totally-real-credential-do-not-leak-1234567890"
        yaml_text = f"""
sources:
  source:
    plugin: csv
    on_success: main
    options:
      api_key: {secret_value}
      on_validation_failure: discard
sinks:
  main:
    plugin: csv
    on_write_failure: discard
"""

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "api_key" in detail
        assert secret_value not in resp.text
        # Nothing was persisted -- the session has no composition state at all.
        assert await service.get_current_state(session.id) is None

    @pytest.mark.asyncio
    async def test_post_state_yaml_rejects_plugin_specific_credential_field(self, tmp_path) -> None:
        """Hardening (T-1 review follow-up): the fabricated-secret import gate
        must match plugin-specific credential fields, not just the name/suffix
        heuristic. The database sink's whole-DSN ``url`` field carries an
        embedded password but does not end in a secret suffix, so the heuristic
        predicate alone lets it through -- the import gate feeds
        allowed_secret_ref_fields per component (mirroring the set_output tool
        gate) so a pasted plaintext DSN is rejected here rather than persisted
        and echoed back. The error names the field only, never the value."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Replay", "local")
        dsn = "postgresql://app:S3cretPw@db.internal/prod"  # secret-scan: allow-this-line (synthetic fixture asserting rejection)
        yaml_text = f"""
sources:
  source:
    plugin: csv
    on_success: main
    options:
      on_validation_failure: discard
sinks:
  main:
    plugin: database
    on_write_failure: discard
    options:
      url: {dsn}
"""

        resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "url" in detail
        assert "S3cretPw" not in resp.text
        assert dsn not in resp.text
        # Nothing was persisted -- the DSN never reached the DB.
        assert await service.get_current_state(session.id) is None

    @pytest.mark.asyncio
    async def test_yaml_returns_yaml_when_state_exists(self, tmp_path) -> None:
        """Returns generated YAML for a valid state even when edge_contracts is empty."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"plugin": "csv", "on_success": "out", "options": {"path": "/data.csv"}, "on_validation_failure": "quarantine"},
                outputs=[
                    {
                        "name": "out",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Test Pipeline", "description": ""},
                is_valid=False,
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")
        assert resp.status_code == 200
        body = resp.json()
        assert "yaml" in body
        assert "csv" in body["yaml"]

    @pytest.mark.asyncio
    async def test_yaml_response_preserves_source_blob_identity_outside_engine_yaml(self, tmp_path) -> None:
        """Final YAML artifacts must retain blob custody even though YAML strips blob_ref."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        blob_id = "98b1357d-5aab-4fb3-85b4-5ad643912e84"

        session = await service.create_session("alice", "Pipeline", "local")
        app.state.blob_service = SimpleNamespace(
            get_blob=AsyncMock(
                return_value=SimpleNamespace(
                    id=uuid.UUID(blob_id),
                    session_id=session.id,
                    storage_path="/data/blobs/session/contact_form_submissions.csv",
                    status="ready",
                )
            )
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {
                        "path": "/data/blobs/session/contact_form_submissions.csv",
                        "blob_ref": blob_id,
                        "mode": "bind_source",
                        "schema": {"mode": "observed"},
                    },
                    "on_validation_failure": "quarantine",
                },
                outputs=[
                    {
                        "name": "out",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Blob-backed source", "description": ""},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_blob_ids"] == {"source": blob_id}
        assert blob_id not in body["yaml"]
        assert "/data/blobs/session/contact_form_submissions.csv" not in body["yaml"]
        exported_source_options = yaml.safe_load(body["yaml"])["sources"]["source"]["options"]
        assert "path" not in exported_source_options
        assert "mode" not in exported_source_options
        assert exported_source_options["schema"] == {"mode": "observed"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custody_failure",
        [
            "foreign_session",
            "wrong_id",
            "wrong_path",
            "non_ready",
            "missing",
            "malformed_record",
            "service_unavailable",
            "noncanonical",
        ],
    )
    async def test_yaml_export_rejects_invalid_blob_custody_with_safe_500(self, tmp_path, custody_failure: str) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        blob_id = uuid.UUID("98b1357d-5aab-4fb3-85b4-5ad643912e84")
        session = await service.create_session("alice", "Pipeline", "local")
        foreign_session_id = uuid.uuid4()
        storage_path = "/data/blobs/foreign/private.csv"
        get_blob = AsyncMock(
            return_value=SimpleNamespace(
                id=uuid.uuid4() if custody_failure == "wrong_id" else blob_id,
                session_id=foreign_session_id if custody_failure == "foreign_session" else session.id,
                storage_path="/data/blobs/same-session/wrong.csv" if custody_failure == "wrong_path" else storage_path,
                status="pending" if custody_failure == "non_ready" else "ready",
            )
        )
        if custody_failure == "missing":
            get_blob.side_effect = BlobNotFoundError(str(blob_id))
        if custody_failure == "malformed_record":
            get_blob.return_value = SimpleNamespace()
        if custody_failure != "service_unavailable":
            app.state.blob_service = SimpleNamespace(
                get_blob=get_blob,
            )
        blob_ref = "NOT-A-CANONICAL-UUID" if custody_failure == "noncanonical" else str(blob_id)
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {"path": storage_path, "blob_ref": blob_ref, "mode": "bind_source"},
                    "on_validation_failure": "discard",
                },
                outputs=[{"name": "out", "plugin": "csv", "options": {}, "on_write_failure": "discard"}],
                metadata_={"name": "Foreign binding", "description": ""},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert str(blob_id) not in response.text
        assert storage_path not in response.text
        if custody_failure in {"noncanonical", "service_unavailable"}:
            get_blob.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_blob_ref",
        [None, "", 123, "98B1357D-5AAB-4FB3-85B4-5AD643912E84"],
        ids=["none", "empty", "wrong_type", "noncanonical_uuid"],
    )
    async def test_yaml_export_rejects_present_invalid_reviewed_blob_ref_before_audit(
        self,
        tmp_path: Path,
        invalid_blob_ref: object,
    ) -> None:
        from sqlalchemy import select

        from elspeth.web.sessions.models import composer_completion_events_table

        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        stable_id = "98b1357d-5aab-4fb3-85b4-5ad643912e84"
        storage_path = "/data/blobs/foreign/private.csv"
        session = await service.create_session("alice", "Pipeline", "local")
        get_blob = AsyncMock()
        app.state.blob_service = SimpleNamespace(get_blob=get_blob)
        guided = replace(
            GuidedSession.initial(),
            source_order=(stable_id,),
            reviewed_sources={
                stable_id: SourceResolved(
                    name="source",
                    plugin="csv",
                    options={"path": storage_path, "blob_ref": invalid_blob_ref},
                    observed_columns=("value",),
                    sample_rows=(),
                    on_validation_failure="discard",
                )
            },
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {"path": storage_path},
                    "on_validation_failure": "discard",
                },
                outputs=[{"name": "out", "plugin": "csv", "options": {}, "on_write_failure": "discard"}],
                metadata_={"name": "Invalid reviewed binding", "description": ""},
                is_valid=True,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert storage_path not in response.text
        if type(invalid_blob_ref) is str and invalid_blob_ref:
            assert invalid_blob_ref not in response.text
        get_blob.assert_not_awaited()
        with app.state.session_engine.connect() as conn:
            export_events = conn.execute(
                select(composer_completion_events_table).where(
                    composer_completion_events_table.c.session_id == str(session.id),
                    composer_completion_events_table.c.event_type == "export_yaml",
                )
            ).all()
        assert export_events == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_carriers",
        [
            {"path": ""},
            {"file": ""},
            {"path": None},
            {"file": 123},
            {"path": "/data/blobs/foreign/private.csv", "file": None},
            {"path": "/data/blobs/foreign/pri\x00vate.csv"},
        ],
        ids=["empty_path", "empty_file", "none_path", "wrong_type_file", "valid_path_invalid_file", "nul_path"],
    )
    async def test_yaml_export_rejects_invalid_reviewed_path_carriers_before_audit(
        self,
        tmp_path: Path,
        invalid_carriers: dict[str, object],
    ) -> None:
        from sqlalchemy import select

        from elspeth.web.sessions.models import composer_completion_events_table

        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        stable_id = "98b1357d-5aab-4fb3-85b4-5ad643912e84"
        storage_path = "/data/blobs/foreign/private.csv"
        session = await service.create_session("alice", "Pipeline", "local")
        get_blob = AsyncMock()
        app.state.blob_service = SimpleNamespace(get_blob=get_blob)
        guided = replace(
            GuidedSession.initial(),
            source_order=(stable_id,),
            reviewed_sources={
                stable_id: SourceResolved(
                    name="source",
                    plugin="csv",
                    options={**invalid_carriers, "blob_ref": stable_id},
                    observed_columns=("value",),
                    sample_rows=(),
                    on_validation_failure="discard",
                )
            },
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {"path": storage_path},
                    "on_validation_failure": "discard",
                },
                outputs=[{"name": "out", "plugin": "csv", "options": {}, "on_write_failure": "discard"}],
                metadata_={"name": "Invalid reviewed carrier", "description": ""},
                is_valid=True,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert stable_id not in response.text
        assert storage_path not in response.text
        get_blob.assert_not_awaited()
        with app.state.session_engine.connect() as conn:
            export_events = conn.execute(
                select(composer_completion_events_table).where(
                    composer_completion_events_table.c.session_id == str(session.id),
                    composer_completion_events_table.c.event_type == "export_yaml",
                )
            ).all()
        assert export_events == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reviewed_carriers", "live_options"),
        [
            ({"path": " /data/blobs/foreign/bogus.csv "}, {"path": "/data/blobs/foreign/live.csv"}),
            ({"path": " /data/blobs/foreign/bogus.csv "}, {"schema": {"mode": "observed"}}),
            (
                {"path": "/data/blobs/foreign/live.csv"},
                {"path": "/data/blobs/foreign/live.csv", "file": "/data/blobs/foreign/secret.csv"},
            ),
            (
                {"path": "/data/blobs/foreign/live.csv", "file": "/data/blobs/foreign/live-alias.csv"},
                {"path": "/data/blobs/foreign/live.csv"},
            ),
        ],
        ids=["mismatched_path", "missing_live_carrier", "extra_live_carrier", "missing_live_reviewed_carrier"],
    )
    async def test_yaml_export_rejects_same_name_without_exact_reviewed_path_before_audit(
        self,
        tmp_path: Path,
        reviewed_carriers: dict[str, object],
        live_options: dict[str, object],
    ) -> None:
        from sqlalchemy import select

        from elspeth.web.sessions.models import composer_completion_events_table

        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        stable_id = "98b1357d-5aab-4fb3-85b4-5ad643912e84"
        session = await service.create_session("alice", "Pipeline", "local")
        get_blob = AsyncMock(
            return_value=SimpleNamespace(
                id=uuid.UUID(stable_id),
                session_id=session.id,
                storage_path="/data/blobs/foreign/live.csv",
                status="ready",
            )
        )
        app.state.blob_service = SimpleNamespace(get_blob=get_blob)
        guided = replace(
            GuidedSession.initial(),
            source_order=(stable_id,),
            reviewed_sources={
                stable_id: SourceResolved(
                    name="source",
                    plugin="csv",
                    options={**reviewed_carriers, "blob_ref": stable_id},
                    observed_columns=("value",),
                    sample_rows=(),
                    on_validation_failure="discard",
                )
            },
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": live_options,
                    "on_validation_failure": "discard",
                },
                outputs=[{"name": "out", "plugin": "csv", "options": {}, "on_write_failure": "discard"}],
                metadata_={"name": "Inconsistent reviewed mapping", "description": ""},
                is_valid=True,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert stable_id not in response.text
        for value in (*reviewed_carriers.values(), *live_options.values()):
            if type(value) is str and value:
                assert value not in response.text
        get_blob.assert_not_awaited()
        with app.state.session_engine.connect() as conn:
            export_events = conn.execute(
                select(composer_completion_events_table).where(
                    composer_completion_events_table.c.session_id == str(session.id),
                    composer_completion_events_table.c.event_type == "export_yaml",
                )
            ).all()
        assert export_events == []

    @pytest.mark.asyncio
    async def test_yaml_export_rejects_reviewed_blob_ref_without_path_before_audit(self, tmp_path: Path) -> None:
        from sqlalchemy import select

        from elspeth.web.sessions.models import composer_completion_events_table

        app, service = _make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        blob_id = uuid.UUID("98b1357d-5aab-4fb3-85b4-5ad643912e84")
        storage_path = "/data/blobs/foreign/private.csv"
        session = await service.create_session("alice", "Pipeline", "local")
        get_blob = AsyncMock()
        app.state.blob_service = SimpleNamespace(get_blob=get_blob)
        guided = replace(
            GuidedSession.initial(),
            source_order=(str(blob_id),),
            reviewed_sources={
                str(blob_id): SourceResolved(
                    name="source",
                    plugin="csv",
                    options={"blob_ref": str(blob_id)},
                    observed_columns=("value",),
                    sample_rows=(),
                    on_validation_failure="discard",
                )
            },
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {"path": storage_path},
                    "on_validation_failure": "discard",
                },
                outputs=[{"name": "out", "plugin": "csv", "options": {}, "on_write_failure": "discard"}],
                metadata_={"name": "Pathless reviewed binding", "description": ""},
                is_valid=True,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            response = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert str(blob_id) not in response.text
        assert storage_path not in response.text
        get_blob.assert_not_awaited()
        with app.state.session_engine.connect() as conn:
            export_events = conn.execute(
                select(composer_completion_events_table).where(
                    composer_completion_events_table.c.session_id == str(session.id),
                    composer_completion_events_table.c.event_type == "export_yaml",
                )
            ).all()
        assert export_events == []

    @pytest.mark.asyncio
    async def test_yaml_allows_connection_valid_state_without_ui_edges(self, tmp_path) -> None:
        """Connection-defined pipelines should export even when the editor graph is incomplete."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "text",
                    "on_success": "mapper_in",
                    "options": {
                        "path": "/data/input.txt",
                        "column": "text",
                        "schema": {"mode": "observed"},
                    },
                    "on_validation_failure": "quarantine",
                },
                nodes=[
                    {
                        "id": "map_body",
                        "node_type": "transform",
                        "plugin": "field_mapper",
                        "input": "mapper_in",
                        "on_success": "main",
                        "on_error": "discard",
                        "options": {
                            "schema": {"mode": "observed", "guaranteed_fields": ["text"], "required_fields": ["text"]},
                            "mapping": {"text": "body"},
                        },
                        "condition": None,
                        "routes": None,
                        "fork_to": None,
                        "branches": None,
                        "policy": None,
                        "merge": None,
                    },
                ],
                edges=[],
                outputs=[
                    {
                        "name": "main",
                        "plugin": "csv",
                        "options": {
                            "path": "outputs/out.csv",
                            "schema": {"mode": "observed", "required_fields": ["body"]},
                        },
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Connection-only Pipeline", "description": ""},
                is_valid=False,
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")
        assert resp.status_code == 200
        assert "field_mapper" in resp.json()["yaml"]
        assert "body" in resp.json()["yaml"]

    @pytest.mark.asyncio
    async def test_yaml_serializes_coalesce_on_success_runtime_route(self, tmp_path) -> None:
        """Coalesce terminal routing must survive export/reload parity checks."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "gate_in",
                    "options": {"path": "/data/input.csv"},
                    "on_validation_failure": "quarantine",
                },
                nodes=[
                    {
                        "id": "fork_gate",
                        "node_type": "gate",
                        "plugin": None,
                        "input": "gate_in",
                        "on_success": None,
                        "on_error": None,
                        "options": {},
                        "condition": "True",
                        "routes": {},
                        "fork_to": ["path_a", "path_b"],
                        "branches": None,
                        "policy": None,
                        "merge": None,
                    },
                    {
                        "id": "merge_point",
                        "node_type": "coalesce",
                        "plugin": None,
                        "input": "join",
                        "on_success": "main",
                        "on_error": None,
                        "options": {},
                        "condition": None,
                        "routes": None,
                        "fork_to": None,
                        "branches": ["path_a", "path_b"],
                        "policy": "require_all",
                        "merge": "nested",
                    },
                ],
                edges=[],
                outputs=[
                    {
                        "name": "main",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv"},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Fork and merge", "description": ""},
                is_valid=False,
            ),
            provenance="session_seed",
        )

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 200
        doc = yaml.safe_load(resp.json()["yaml"])
        assert doc["coalesce"][0]["on_success"] == "main"

    def test_yaml_returns_404_when_no_state(self, tmp_path) -> None:
        """No composition state yet -> 404."""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Empty"})
        session_id = resp.json()["id"]

        yaml_resp = client.get(f"/api/sessions/{session_id}/state/yaml")
        assert yaml_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_state_yaml_validates_exact_state_snapshot(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "main",
                    "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "quarantine",
                },
                outputs=[
                    {
                        "name": "main",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Snapshot", "description": ""},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        captured_states: list[CompositionState] = []

        async def fake_runtime_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            captured_states.append(state)
            return ValidationResult(
                is_valid=False,
                checks=[],
                errors=[
                    ValidationError(
                        component_id="source",
                        component_type="source",
                        message="runtime preflight failed for captured state",
                        suggestion=None,
                        error_code=None,
                    )
                ],
            )

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=fake_runtime_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "Current composition state failed runtime preflight. Fix validation errors before exporting YAML."
        assert "runtime preflight failed for captured state" not in resp.text
        assert len(captured_states) == 1
        assert captured_states[0].sources["source"] is not None

    @pytest.mark.asyncio
    async def test_get_state_yaml_does_not_echo_preflight_error_messages(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )
        leaked_value = "REDACTED-preflight-error-canary"

        async def fake_runtime_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            del state, settings, secret_service, user_id, session_id
            return ValidationResult(
                is_valid=False,
                checks=[],
                errors=[
                    ValidationError(
                        component_id="scraper",
                        component_type="transform",
                        message=f"Invalid CIDR in allowed_hosts: {leaked_value!r}",
                        suggestion=None,
                        error_code=None,
                    )
                ],
            )

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=fake_runtime_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "Current composition state failed runtime preflight. Fix validation errors before exporting YAML."
        assert leaked_value not in resp.text

    @pytest.mark.asyncio
    async def test_get_state_yaml_threads_session_id_to_runtime_preflight(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )
        seen_session_ids: list[uuid.UUID] = []

        async def capture_session_id(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            del state, settings, secret_service, user_id
            seen_session_ids.append(session_id)
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=capture_session_id):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 200, resp.text
        assert seen_session_ids == [session.id]

    @pytest.mark.asyncio
    async def test_get_state_yaml_emits_yaml_export_telemetry_on_passed_preflight(self, tmp_path, monkeypatch) -> None:
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[str, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((str(value), dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )

        async def pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=pass_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 200
        assert any(attrs.get("source") == "yaml_export" and attrs.get("result") == "passed" for _, attrs in emitted)

    @pytest.mark.asyncio
    async def test_get_state_yaml_emits_yaml_export_telemetry_on_failed_preflight(self, tmp_path, monkeypatch) -> None:
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[str, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((str(value), dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )

        failure = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[ValidationError(component_id=None, component_type=None, message="bad runtime", suggestion=None, error_code=None)],
        )

        async def fail_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return failure

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=fail_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 409
        assert any(attrs.get("source") == "yaml_export" and attrs.get("result") == "failed" for _, attrs in emitted)

    @pytest.mark.asyncio
    async def test_get_state_yaml_handles_preflight_exception_with_telemetry_and_409(self, tmp_path, monkeypatch) -> None:
        """Preflight exceptions must surface as 409 with bounded telemetry, not 500 with raw exception text."""
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[str, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((str(value), dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )

        secret_canary = "this-text-must-not-appear-in-the-response-body"

        async def boom(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            raise TimeoutError(secret_canary)

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=boom):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 409
        # The raw exception text must not leak into the response body.
        assert secret_canary not in resp.text
        assert "Runtime preflight could not complete" in resp.json()["detail"]
        # Telemetry recorded the exception class as a bounded attribute.
        assert any(
            attrs.get("source") == "yaml_export" and attrs.get("result") == "exception" and attrs.get("exception_class") == "TimeoutError"
            for _, attrs in emitted
        )

    @pytest.mark.asyncio
    async def test_get_state_yaml_propagates_programmer_bugs_uncaught(self, tmp_path, monkeypatch) -> None:
        """I4a lock-in: the YAML-export catch is narrowed to user-fixable
        exception classes (TimeoutError, OSError, PluginConfigError,
        PluginNotFoundError, GraphValidationError). Programmer-bug classes
        (AttributeError, TypeError, KeyError, RuntimeError, ImportError)
        MUST propagate uncaught so:

        * Operators see the real traceback via FastAPI's default 500
          handler, not "Runtime preflight could not complete; YAML export
          aborted." which falsely implies a user-fixable failure.
        * No yaml_export telemetry is emitted for programmer bugs — the
          exception counter is reserved for the user-fixable bucket so
          dashboards measure the real preflight failure rate, not bugs
          we introduced ourselves.

        Per CLAUDE.md offensive-programming policy: programmer bugs MUST
        crash. Conflating them with user-fixable failures destroys the
        operator's ability to diagnose.
        """
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[str, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((str(value), dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
        app, service = _make_app(tmp_path)
        # raise_server_exceptions=False so Starlette returns the 500 body
        # rather than re-raising the AttributeError into the test runner.
        client = TestClient(app, raise_server_exceptions=False)
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(metadata_={"name": "Snapshot", "description": ""}, is_valid=True), provenance="session_seed"
        )

        async def programmer_bug(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            # A real bug we'd see if e.g. a refactor accidentally broke
            # an attribute lookup inside validate_pipeline. AttributeError
            # is in the canonical "programmer bug" set per CLAUDE.md
            # offensive-programming policy.
            raise AttributeError("'NoneType' object has no attribute 'plugin'")

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=programmer_bug):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        # FastAPI's default 500 handler emits the bare 'Internal Server
        # Error' string body when an unhandled exception escapes a route.
        assert resp.status_code == 500
        assert "Runtime preflight could not complete" not in resp.text, (
            f"Programmer bug must not be relabeled as a user-fixable preflight failure; body was: {resp.text!r}"
        )
        # No yaml_export telemetry should fire — the exception bucket is
        # reserved for user-fixable preflight outcomes.
        assert not any(attrs.get("source") == "yaml_export" and attrs.get("result") == "exception" for _, attrs in emitted), (
            "yaml_export exception telemetry fired for a programmer bug — "
            "the catch should not have intercepted AttributeError. "
            f"emitted={emitted}"
        )

    @pytest.mark.asyncio
    async def test_get_state_yaml_preserves_secret_ref_markers_in_output(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        resolved_secret = "__RESOLVED_SECRET_CANARY_DO_NOT_EXPORT__"  # secret-scan: allow-this-line

        class FakeResolvedSecretService:
            resolved_value = resolved_secret

        app.state.scoped_secret_resolver = FakeResolvedSecretService()
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "main",
                    "options": {
                        "path": "/data/blobs/input.csv",
                        "schema": {"mode": "observed"},
                        "api_key": {"secret_ref": "OPENAI_API_KEY"},
                    },
                    "on_validation_failure": "quarantine",
                },
                outputs=[
                    {
                        "name": "main",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Secret export", "description": ""},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        async def fake_runtime_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            assert secret_service is None
            # YAML export preflight must not receive the scoped resolver. It only
            # serializes the original state snapshot with the secret_ref marker.
            assert state.to_dict()["sources"]["source"]["options"]["api_key"] == {"secret_ref": "OPENAI_API_KEY"}
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=fake_runtime_preflight):
            resp = client.get(f"/api/sessions/{session.id}/state/yaml")

        assert resp.status_code == 200
        exported_yaml = resp.json()["yaml"]
        assert resolved_secret not in exported_yaml
        parsed = yaml.safe_load(exported_yaml)
        assert parsed["sources"]["source"]["options"]["api_key"] == {"secret_ref": "OPENAI_API_KEY"}

    @pytest.mark.asyncio
    async def test_post_state_yaml_imports_and_persists_queue(self, tmp_path) -> None:
        """A pasted runtime queue section survives import and is persisted as a
        canonical structural queue node (elspeth-a5b86149d4)."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Queue import", "local")
        yaml_text = """
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      schema:
        mode: observed
queues:
  inbound:
    description: Orders and refunds interleave here
transforms:
- name: normalize
  plugin: passthrough
  input: inbound
  on_success: main
  on_error: discard
  options:
    schema:
      mode: observed
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            resp = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": yaml_text})

        assert resp.status_code == 200, resp.text
        record = await service.get_current_state(session.id)
        assert record is not None
        queue_nodes = [node for node in record.nodes if node["node_type"] == "queue"]
        assert len(queue_nodes) == 1
        assert queue_nodes[0]["id"] == "inbound"
        assert queue_nodes[0]["options"] == {"description": "Orders and refunds interleave here"}

    @pytest.mark.asyncio
    async def test_post_state_yaml_malformed_queue_is_rejected_atomically(self, tmp_path) -> None:
        """Malformed queue YAML is a 400 that leaves the prior persisted state and
        version untouched — the import boundary is all-or-nothing."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Queue atomicity", "local")
        valid_yaml = """
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      schema:
        mode: observed
queues:
  inbound: {}
transforms:
- name: normalize
  plugin: passthrough
  input: inbound
  on_success: main
  on_error: discard
  options:
    schema:
      mode: observed
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
    on_write_failure: discard
"""
        malformed_yaml = valid_yaml.replace("  inbound: {}", "  inbound:\n    priority: 5")

        async def _pass_preflight(state, *, settings, secret_service, user_id, session_id, **_policy_context):
            return ValidationResult(is_valid=True, checks=[], errors=[])

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            ok = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": valid_yaml})
        assert ok.status_code == 200, ok.text
        before = await service.get_current_state(session.id)
        assert before is not None
        version_before = before.version
        nodes_before = list(before.nodes)

        with patch("elspeth.web.sessions.routes.composer.state._runtime_preflight_for_state", side_effect=_pass_preflight):
            bad = client.post(f"/api/sessions/{session.id}/state/yaml", json={"yaml": malformed_yaml})

        assert bad.status_code == 400
        assert "priority" in bad.json()["detail"]
        after = await service.get_current_state(session.id)
        assert after is not None
        assert after.version == version_before
        assert list(after.nodes) == nodes_before


class TestRunAlreadyActiveError:
    """Tests for seam contract D: RunAlreadyActiveError → 409 with error_type.

    The create_run endpoint does not exist yet (Sub-5), but the exception
    handler is wired. These tests exercise it via direct service calls +
    app-level exception propagation to verify the contract.
    """

    @pytest.mark.asyncio
    async def test_run_already_active_returns_409(self, tmp_path) -> None:
        """RunAlreadyActiveError produces 409 with error_type field."""
        from elspeth.web.sessions.protocol import RunAlreadyActiveError

        app, service = _make_app(tmp_path)

        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        # Create a run to block the session
        await service.create_run(session.id, v1.id)

        # Register the app-level exception handler (wired in create_app,
        # but our test app uses create_session_router directly). Wire it here.
        from fastapi.responses import JSONResponse

        @app.exception_handler(RunAlreadyActiveError)
        async def handle_run_already_active(
            request,
            exc: RunAlreadyActiveError,
        ) -> JSONResponse:
            return JSONResponse(
                status_code=409,
                content={"detail": str(exc), "error_type": "run_already_active"},
            )

        # Add a test endpoint that triggers the error
        @app.post("/api/_test_create_run")
        async def _test_create_run():
            await service.create_run(session.id, v1.id)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/_test_create_run")
        assert resp.status_code == 409
        body = resp.json()
        assert body["error_type"] == "run_already_active"
        assert "detail" in body


class TestNewStateHasNoLineage:
    """Test that fresh composition states have null derived_from_state_id."""

    @pytest.mark.asyncio
    async def test_fresh_state_has_null_derived_from(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "csv"}, is_valid=True), provenance="session_seed"
        )

        resp = client.get(f"/api/sessions/{session.id}/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["derived_from_state_id"] is None

    @pytest.mark.asyncio
    async def test_state_response_exposes_redacted_named_sources(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)
        session = await service.create_session("alice", "Multi-source", "local")
        sources = {
            "orders": {
                "plugin": "csv",
                "on_success": "orders_rows",
                "on_validation_failure": "discard",
                "options": {"blob_ref": "blob-1", "path": str(tmp_path / "internal" / "orders.csv")},
            },
            "refunds": {
                "plugin": "csv",
                "on_success": "refunds_rows",
                "on_validation_failure": "discard",
                "options": {"path": "refunds.csv"},
            },
        }
        await service.save_composition_state(
            session.id,
            CompositionStateData(sources=sources, is_valid=True),
            provenance="session_seed",
        )

        resp = client.get(f"/api/sessions/{session.id}/state")

        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"]["orders"]["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
        assert body["sources"]["refunds"]["options"]["path"] == "refunds.csv"


class TestComposerProgressRoutes:
    @pytest.mark.asyncio
    async def test_progress_endpoint_returns_latest_snapshot_for_owned_session(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        registry = app.state.composer_progress_registry
        await registry.publish(
            session_id=str(service.session.id),
            request_id="message-1",
            user_id=service.session.user_id,
            event=ComposerProgressEvent(
                phase="using_tools",
                headline="The model requested plugin schemas.",
                evidence=("Checking available source, transform, and sink tools.",),
                likely_next="ELSPETH will use the schemas to choose a pipeline shape.",
            ),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/sessions/{service.session.id}/composer-progress")

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == str(service.session.id)
        assert body["request_id"] == "message-1"
        assert body["phase"] == "using_tools"
        assert body["headline"] == "The model requested plugin schemas."
        assert body["evidence"] == ["Checking available source, transform, and sink tools."]

    @pytest.mark.asyncio
    async def test_progress_endpoint_enforces_session_ownership(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)

        async def bob_user():
            return UserIdentity(user_id="bob", username="bob")

        app.dependency_overrides[get_current_user] = bob_user

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/sessions/{service.session.id}/composer-progress")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_marks_terminal_progress_with_user_message_id(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        composer = _ProgressAwareComposer()
        app.state.composer_service = composer

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/sessions/{service.session.id}/messages",
                json={"content": "Exploit HTML into JSON"},
            )
            messages = (await client.get(f"/api/sessions/{service.session.id}/messages")).json()
            progress = (await client.get(f"/api/sessions/{service.session.id}/composer-progress")).json()

        assert response.status_code == 200
        assert composer.progress_sink_seen is True
        user_message_id = next(message["id"] for message in messages if message["role"] == "user")
        assert progress["request_id"] == user_message_id
        assert progress["phase"] == "complete"
        assert progress["headline"] == "The composer has updated the pipeline."

    @pytest.mark.asyncio
    async def test_send_message_returns_transition_only_state_row(self, tmp_path) -> None:
        """A guided->freeform transition save is still the session's new current state.

        Regression for the live stale_compose_state loop: the first freeform
        compose turn may only persist ``guided_session.transition_consumed`` and
        leave ``CompositionState.version`` unchanged. The route must still return
        the newly persisted state row so the SPA's next message uses the current
        state id.
        """
        from elspeth.web.composer.state import CompositionState

        app, service = _make_progress_route_app(tmp_path)
        guided = GuidedSession(
            step=GuidedStep.STEP_1_SOURCE,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=TerminalState(
                kind=TerminalKind.EXITED_TO_FREEFORM,
                reason=TerminalReason.USER_PRESSED_EXIT,
                pipeline_yaml=None,
            ),
            transition_consumed=False,
        )
        initial_state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
            guided_session=guided,
        )
        initial_state_d = initial_state.to_dict()
        await service.save_composition_state(
            service.session.id,
            CompositionStateData(
                sources=initial_state_d["sources"],
                nodes=initial_state_d["nodes"],
                edges=initial_state_d["edges"],
                outputs=initial_state_d["outputs"],
                metadata_=initial_state_d["metadata"],
                is_valid=False,
                validation_errors=None,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )

        class _NoGraphChangeComposer:
            async def compose(
                self,
                message: str,
                chat_messages: list[dict[str, object]],
                state: CompositionState,
                *,
                session_id: str | None = None,
                current_state_id: str | None = None,
                user_id: str | None = None,
                progress=None,
                guided_terminal=None,
                user_message_id: str | None = None,
            ) -> ComposerResult:
                del message, chat_messages, session_id, current_state_id, user_id, progress, user_message_id
                assert guided_terminal == guided.terminal
                return ComposerResult(message="Freeform response", state=state)

        app.state.composer_service = _NoGraphChangeComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/sessions/{service.session.id}/messages",
                json={"content": "continue in freeform"},
            )
            messages = (await client.get(f"/api/sessions/{service.session.id}/messages")).json()

        assert response.status_code == 200
        body = response.json()
        assert body["state"] is not None
        assert body["state"]["version"] == 2
        assert body["state"]["id"] == messages[-1]["composition_state_id"]
        assert body["state"]["composer_meta"]["guided_session"]["transition_consumed"] is True

    @pytest.mark.asyncio
    async def test_recompose_marks_terminal_progress_with_last_user_message_id(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        composer = _ProgressAwareComposer("Retry reply")
        app.state.composer_service = composer
        user_msg = await service.add_message(service.session.id, "user", "Exploit HTML into JSON", writer_principal="route_user_message")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/api/sessions/{service.session.id}/recompose")
            progress = (await client.get(f"/api/sessions/{service.session.id}/composer-progress")).json()

        assert response.status_code == 200
        assert composer.progress_sink_seen is True
        assert progress["request_id"] == str(user_msg.id)
        assert progress["phase"] == "complete"


class TestComposerInFlightEndpoint:
    """The /_active endpoint is the cross-session view added with elspeth-29e8bd8a1f.

    Per-session /composer-progress polling already worked, but answered "what's
    happening with session X?". Operators needed "what's running on this server
    right now?" — without that view, an in-flight composer request was invisible
    in the journal until the response completed (uvicorn access log timing).
    """

    @pytest.mark.asyncio
    async def test_returns_only_authenticated_users_in_flight_sessions(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        registry = app.state.composer_progress_registry

        # alice has one in-flight and one completed session.
        await registry.publish(
            session_id=str(service.session.id),
            request_id="msg-1",
            user_id="alice",
            event=ComposerProgressEvent(
                phase="calling_model",
                headline="The model is composing.",
                evidence=("Prompt was built.",),
            ),
        )
        await registry.publish(
            session_id="alice-completed-session",
            request_id="msg-completed",
            user_id="alice",
            event=ComposerProgressEvent(
                phase="complete",
                headline="The composer response is ready.",
                evidence=("Saved.",),
                reason="composer_complete",
            ),
        )
        # bob has one in-flight session — must not appear in alice's view.
        await registry.publish(
            session_id="bob-active-session",
            request_id="msg-bob",
            user_id="bob",
            event=ComposerProgressEvent(
                phase="using_tools",
                headline="Bob's request.",
                evidence=("Active.",),
            ),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/sessions/_active")

        assert resp.status_code == 200
        body = resp.json()
        assert {snap["session_id"] for snap in body} == {str(service.session.id)}
        # phase must be a non-terminal value; counters depend on this invariant.
        assert all(snap["phase"] in {"starting", "calling_model", "using_tools", "validating", "saving"} for snap in body)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_sessions_in_flight(self, tmp_path) -> None:
        """Empty list, not 404 — it's a successful query with zero results."""
        app, _ = _make_progress_route_app(tmp_path)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/sessions/_active")

        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_route_path_does_not_collide_with_session_id_route(self, tmp_path) -> None:
        """_active must beat the /{session_id} pattern. Otherwise FastAPI tries
        to parse 'underscore_active' as a UUID and returns 422 before the
        operator ever sees the snapshot list.
        """
        app, _ = _make_progress_route_app(tmp_path)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/sessions/_active")

        # If the route order were wrong, this would be 422 (UUID parse error).
        assert resp.status_code != 422


class TestComposerCancellationLifecycle:
    """elspeth-29e8bd8a1f acceptance criteria: client cancellations are
    distinguished from server timeouts, both in the progress snapshot
    (phase=cancelled, reason=client_cancelled) and in the terminal
    counter (status=cancelled vs status=timed_out).
    """

    @pytest.mark.asyncio
    async def test_send_message_publishes_cancelled_snapshot_on_cancellation(self, tmp_path) -> None:
        """When compose() is cancelled mid-flight (real disconnect or asyncio
        cancel), the route MUST publish a phase=cancelled / reason=client_cancelled
        snapshot before re-raising. This proves the except-CancelledError +
        asyncio.shield contract.

        The fake composer raises CancelledError directly. A live ASGI
        http.disconnect propagates as a CancelledError on the route task
        — the route handler cannot tell the two apart, so this test
        exercises the contract that matters.
        """
        app, service = _make_progress_route_app(tmp_path)

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise asyncio.CancelledError()

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(
                    f"/api/sessions/{service.session.id}/messages",
                    json={"content": "Will be cancelled"},
                )

        registry = app.state.composer_progress_registry
        snapshot = await registry.get_latest(str(service.session.id))
        assert snapshot.phase == "cancelled"
        assert snapshot.reason == "client_cancelled"
        # list_active must NOT include cancelled phases — otherwise the
        # /_active operator view would persistently show stale cancellations.
        active = await registry.list_active(user_id=service.session.user_id)
        assert active == ()

    @pytest.mark.asyncio
    async def test_send_message_persists_cancelled_llm_call_audit_sidecar(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        llm_call = _llm_call(
            status=ComposerLLMCallStatus.CANCELLED,
            model_returned=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            provider_request_id=None,
            error_class="CancelledError",
            error_message="CancelledError",
        )
        cancelled = _cancelled_error_with_llm_call(llm_call)

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise cancelled

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(
                    f"/api/sessions/{service.session.id}/messages",
                    json={"content": "Will be cancelled"},
                )

        llm_audit_rows = _llm_call_audit_rows(service.messages)
        assert len(llm_audit_rows) == 1
        _row, tool_call = llm_audit_rows[0]
        assert tool_call["call"]["status"] == "cancelled"
        assert tool_call["call"]["messages_hash"] == llm_call.messages_hash

    @pytest.mark.asyncio
    async def test_send_message_increments_terminal_counter_with_cancelled_status(self, tmp_path, monkeypatch) -> None:
        """The terminal counter must record status=cancelled separately from
        status=timed_out so a Grafana board can distinguish "the client gave
        up" from "the server gave up".
        """
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[int, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((value, dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_REQUEST_TERMINAL_COUNTER", FakeCounter())

        app, service = _make_progress_route_app(tmp_path)

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise asyncio.CancelledError()

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(
                    f"/api/sessions/{service.session.id}/messages",
                    json={"content": "Will be cancelled"},
                )

        terminal_events = [(v, attrs) for v, attrs in emitted if attrs.get("endpoint") == "send_message"]
        assert len(terminal_events) == 1, f"expected 1 terminal event, got {terminal_events}"
        value, attrs = terminal_events[0]
        assert value == 1
        assert attrs == {"endpoint": "send_message", "status": "cancelled"}

    @pytest.mark.asyncio
    async def test_send_message_increments_terminal_counter_with_completed_status(self, tmp_path, monkeypatch) -> None:
        """Successful path: status=completed. Pinned so a future refactor that
        forgets to set terminal_status before return falls back to the
        pessimistic default of 'failed' — and this test catches it.
        """
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[int, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((value, dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_REQUEST_TERMINAL_COUNTER", FakeCounter())

        app, service = _make_progress_route_app(tmp_path)
        app.state.composer_service = _ProgressAwareComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/api/sessions/{service.session.id}/messages",
                json={"content": "Hello"},
            )

        assert resp.status_code == 200
        terminal_events = [(v, attrs) for v, attrs in emitted if attrs.get("endpoint") == "send_message"]
        assert terminal_events == [(1, {"endpoint": "send_message", "status": "completed"})]

    @pytest.mark.asyncio
    async def test_recompose_publishes_cancelled_snapshot_on_cancellation(self, tmp_path) -> None:
        """Drift guard: recompose mirrors send_message structurally, so the
        cancellation contract MUST remain identical. A 4-line parallel test
        is cheap insurance against future refactors that fix one path and
        forget the other.
        """
        app, service = _make_progress_route_app(tmp_path)
        await service.add_message(service.session.id, "user", "Will be cancelled on retry", writer_principal="route_user_message")

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise asyncio.CancelledError()

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(f"/api/sessions/{service.session.id}/recompose")

        registry = app.state.composer_progress_registry
        snapshot = await registry.get_latest(str(service.session.id))
        assert snapshot.phase == "cancelled"
        assert snapshot.reason == "client_cancelled"

    @pytest.mark.asyncio
    async def test_recompose_persists_cancelled_llm_call_audit_sidecar(self, tmp_path) -> None:
        app, service = _make_progress_route_app(tmp_path)
        await service.add_message(service.session.id, "user", "Will be cancelled on retry", writer_principal="route_user_message")
        llm_call = _llm_call(
            status=ComposerLLMCallStatus.CANCELLED,
            model_returned=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            provider_request_id=None,
            error_class="CancelledError",
            error_message="CancelledError",
        )
        cancelled = _cancelled_error_with_llm_call(llm_call)

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise cancelled

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(f"/api/sessions/{service.session.id}/recompose")

        llm_audit_rows = _llm_call_audit_rows(service.messages)
        assert len(llm_audit_rows) == 1
        _row, tool_call = llm_audit_rows[0]
        assert tool_call["call"]["status"] == "cancelled"
        assert tool_call["call"]["messages_hash"] == llm_call.messages_hash

    @pytest.mark.asyncio
    async def test_recompose_increments_terminal_counter_with_cancelled_status(self, tmp_path, monkeypatch) -> None:
        """Counter parity with send_message — the endpoint attribute on the
        terminal counter must be 'recompose' (not 'send_message'), so a
        Grafana board can split the two routes' cancellation rates.
        """
        from elspeth.web.sessions.routes import _helpers as routes_module

        emitted: list[tuple[int, dict[str, str]]] = []

        class FakeCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((value, dict(attributes)))

        monkeypatch.setattr(routes_module, "_COMPOSER_REQUEST_TERMINAL_COUNTER", FakeCounter())

        app, service = _make_progress_route_app(tmp_path)
        await service.add_message(service.session.id, "user", "Will be cancelled on retry", writer_principal="route_user_message")

        class _CancellingComposer:
            async def compose(self, *args, **kwargs) -> None:
                del args, kwargs
                raise asyncio.CancelledError()

        app.state.composer_service = _CancellingComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(f"/api/sessions/{service.session.id}/recompose")

        terminal_events = [(v, attrs) for v, attrs in emitted if attrs.get("endpoint") == "recompose"]
        assert terminal_events == [(1, {"endpoint": "recompose", "status": "cancelled"})]

    @pytest.mark.asyncio
    async def test_inflight_gauge_increments_then_decrements_across_request(self, tmp_path, monkeypatch) -> None:
        """The UpDownCounter must net to zero across one successful request.
        If the finally clause is removed or the increment is mis-placed,
        the gauge drifts and a Grafana 'currently in flight' panel lies.
        """
        # ``_COMPOSER_REQUESTS_INFLIGHT.add`` is called from the send_message
        # route handler in messages.py (the /messages route this test drives),
        # which bound the name at import — so the patch must target that
        # calling module, not the defining _helpers module.
        from elspeth.web.sessions.routes import messages as messages_module

        emitted: list[tuple[int, dict[str, str]]] = []

        class FakeUpDownCounter:
            def add(self, value: int, attributes: dict[str, str]) -> None:
                emitted.append((value, dict(attributes)))

        monkeypatch.setattr(messages_module, "_COMPOSER_REQUESTS_INFLIGHT", FakeUpDownCounter())

        app, service = _make_progress_route_app(tmp_path)
        app.state.composer_service = _ProgressAwareComposer()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/api/sessions/{service.session.id}/messages",
                json={"content": "Hello"},
            )

        assert resp.status_code == 200
        send_events = [(v, attrs) for v, attrs in emitted if attrs.get("endpoint") == "send_message"]
        assert send_events == [
            (1, {"endpoint": "send_message"}),
            (-1, {"endpoint": "send_message"}),
        ], "in-flight gauge must net to zero across one request"


class TestPaginationRoutes:
    """Tests for limit/offset query parameters on list endpoints."""

    def test_list_sessions_pagination(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        for i in range(5):
            client.post("/api/sessions", json={"title": f"S{i}"})

        resp = client.get("/api/sessions?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp = client.get("/api/sessions?limit=2&offset=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_sessions_pagination_validation(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        # limit < 1
        resp = client.get("/api/sessions?limit=0")
        assert resp.status_code == 422

        # limit > 200
        resp = client.get("/api/sessions?limit=201")
        assert resp.status_code == 422

        # offset < 0
        resp = client.get("/api/sessions?offset=-1")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_messages_pagination(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        # Add messages directly via service to avoid composer dependency
        session = await service.get_session(uuid.UUID(session_id))
        for i in range(5):
            await service.add_message(session.id, "user", f"Msg {i}", writer_principal="route_user_message")

        resp = client.get(f"/api/sessions/{session_id}/messages?limit=2")
        assert resp.status_code == 200
        messages = resp.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "Msg 0"

        resp = client.get(
            f"/api/sessions/{session_id}/messages?limit=2&offset=3",
        )
        assert resp.status_code == 200
        messages = resp.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "Msg 3"

    def test_get_messages_pagination_validation(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        resp = client.get(f"/api/sessions/{session_id}/messages?limit=0")
        assert resp.status_code == 422

        resp = client.get(f"/api/sessions/{session_id}/messages?limit=501")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_state_versions_pagination(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        for _ in range(5):
            await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")

        resp = client.get(
            f"/api/sessions/{session.id}/state/versions?limit=2",
        )
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) == 2
        assert versions[0]["version"] == 1

        resp = client.get(
            f"/api/sessions/{session.id}/state/versions?limit=2&offset=3",
        )
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) == 2
        assert versions[0]["version"] == 4

    def test_get_state_versions_pagination_validation(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        resp = client.get(
            f"/api/sessions/{session_id}/state/versions?limit=0",
        )
        assert resp.status_code == 422

        resp = client.get(
            f"/api/sessions/{session_id}/state/versions?limit=201",
        )
        assert resp.status_code == 422


class TestComposePluginCrashResponse:
    """Plugin TypeError/ValueError from compose() must produce a structured 500.

    After the Task 4 narrowing, plugin bugs escape the service layer instead
    of being laundered as LLM retries. The route handler MUST shape these
    into a documented response rather than letting FastAPI's default handler
    emit an arbitrary traceback.

    Audit-integrity invariant: exception message content — especially
    fragments from __cause__-chained exceptions that may include DB URLs,
    filesystem paths, or secret material — MUST NOT appear in the response
    body. Only the documented error_type + generic detail string is echoed.
    """

    SECRET_PATH = "/etc/elspeth/secrets/bootstrap.key"

    class _StructuredLogRecorder:
        """Minimal slog stand-in for assertions that must ignore global structlog state."""

        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def error(self, event: str, **fields: object) -> None:
            self.events.append({"event": event, **fields})

    @classmethod
    def _capture_route_slogs(cls, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        recorder = cls._StructuredLogRecorder()
        monkeypatch.setattr("elspeth.web.sessions.routes._helpers.slog", recorder)
        return recorder.events

    def test_compose_plugin_value_error_returns_structured_500(self, tmp_path) -> None:
        original = ValueError(f"plugin bug: {self.SECRET_PATH}")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=None),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        body = response.json()
        # FastAPI serializes HTTPException(detail={...}) as {"detail": {...}}.
        assert isinstance(body.get("detail"), dict), body
        assert body["detail"]["error_type"] == "composer_plugin_error"
        assert "user-retryable" in body["detail"]["detail"].lower()

        # Audit-integrity: exception message and cause content MUST NOT leak.
        body_text = response.text
        assert "plugin bug" not in body_text
        assert self.SECRET_PATH not in body_text
        assert "ValueError" not in body_text  # exception class also redacted

    def test_recompose_plugin_type_error_returns_structured_500(self, tmp_path) -> None:
        import asyncio

        original = TypeError(f"plugin bug: NoneType has no attribute 'read' from {self.SECRET_PATH}")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=None),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        # Recompose requires a pre-existing trailing user message (see
        # TestRecomposeConvergencePartialState for the template).
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            service.add_message(uuid.UUID(session_id), "user", "Build something", writer_principal="route_user_message")
        )
        loop.close()

        response = client.post(f"/api/sessions/{session_id}/recompose")

        assert response.status_code == 500
        body = response.json()
        assert isinstance(body.get("detail"), dict), body
        assert body["detail"]["error_type"] == "composer_plugin_error"

        body_text = response.text
        assert "plugin bug" not in body_text
        assert self.SECRET_PATH not in body_text
        assert "NoneType" not in body_text
        assert "TypeError" not in body_text

    def test_compose_plugin_crash_persists_partial_state(self, tmp_path) -> None:
        """P1 regression fix: when a plugin crashes AFTER one or more tool
        calls succeeded in the same request, the accumulated ``partial_state``
        MUST be persisted into ``composition_states`` before the 500 is
        returned.  Without this, recompose restarts from the stale
        pre-request state and silently reverts the LLM's successful mutations.

        Symmetric with ``TestRecomposeConvergencePartialState`` for the
        convergence-error path.
        """
        import asyncio

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="after-first-mutation"),
            version=5,
        )
        original = ValueError(f"plugin bug after mutation: {self.SECRET_PATH}")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=partial),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )
        assert response.status_code == 500
        body = response.json()
        assert body["detail"]["error_type"] == "composer_plugin_error"
        # Response body still fully redacted — persisting partial_state
        # into composition_states does NOT echo it on the failure response.
        assert self.SECRET_PATH not in response.text

        # The partial_state row MUST exist in composition_states now.
        loop = asyncio.new_event_loop()
        try:
            persisted = loop.run_until_complete(service.get_current_state(uuid.UUID(session_id)))
        finally:
            loop.close()
        assert persisted is not None, "partial_state must be persisted to composition_states on plugin crash"
        assert persisted.metadata_ is not None
        assert persisted.metadata_.get("name") == "after-first-mutation"

    def test_compose_plugin_crash_no_partial_state_persists_nothing(self, tmp_path) -> None:
        """When a plugin crashes BEFORE any mutation (partial_state is None),
        no new ``composition_states`` row is written. The 500 response shape
        is identical to the persisted-partial case.
        """
        import asyncio

        original = ValueError("plugin bug on first call")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=None),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )
        assert response.status_code == 500

        loop = asyncio.new_event_loop()
        try:
            persisted = loop.run_until_complete(service.get_current_state(uuid.UUID(session_id)))
        finally:
            loop.close()
        # A brand-new session with no successful mutations → no composition
        # state row should have been created by the crash path.
        assert persisted is None

    def test_compose_plugin_crash_log_has_no_traceback_fields(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2 regression fix: the plugin-crash structured log MUST NOT
        carry traceback-shaped fields. ``exc_info=True`` was dropped
        because plugin exception ``__cause__`` chains may include DB
        URLs, filesystem paths, or secret fragments.
        """
        original = ValueError(f"plugin bug with secret {self.SECRET_PATH}")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=None),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        cap_logs = self._capture_route_slogs(monkeypatch)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )
        assert response.status_code == 500

        crash_events = [e for e in cap_logs if e.get("event") == "compose_plugin_crash"]
        assert len(crash_events) == 1, cap_logs
        event = crash_events[0]
        # Triage fields present.
        assert event["exc_class"] == "ValueError"
        assert event["session_id"] == session_id
        # Traceback-shaped fields absent.
        assert "exc_info" not in event
        assert "exception" not in event
        assert "stack_info" not in event
        # Exception message / secret fragments MUST NOT appear anywhere
        # in the structured event (defense-in-depth).
        serialised = str(event)
        assert self.SECRET_PATH not in serialised
        assert "plugin bug" not in serialised

    def test_compose_plugin_crash_sentinel_leak(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-sentinel test: inject an exception whose ``__str__`` and
        whose ``__cause__.__str__`` each carry a distinct secret sentinel.
        Neither must appear in the HTTP response body nor in any captured
        log record. This guards against future regressions where a
        structlog processor or log field addition inadvertently serialises
        exception content.
        """
        message_secret = "postgres://user:p4ss@prod-db.internal:5432/audit"  # secret-scan: allow-this-line
        cause_secret = "/var/secrets/elspeth/bootstrap-key.pem"

        original = RuntimeError(f"upstream failure: {message_secret}")
        original.__cause__ = FileNotFoundError(cause_secret)

        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=None),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        cap_logs = self._capture_route_slogs(monkeypatch)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )
        assert response.status_code == 500

        # Neither sentinel in response body.
        assert message_secret not in response.text
        assert cause_secret not in response.text

        # Neither sentinel in any captured log record.
        assert cap_logs, "plugin-crash path should emit a structured event"
        for event in cap_logs:
            serialised = str(event)
            assert message_secret not in serialised, event
            assert cause_secret not in serialised, event

    def test_compose_plugin_crash_save_operational_error_preserves_500_body(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression (elspeth-303f751204): when save_composition_state
        raises an ``OperationalError`` (lock timeout, pool disconnect,
        deadlock) while persisting partial_state during a plugin crash,
        the handler MUST still return the structured ``composer_plugin_error``
        500 body rather than letting the secondary DB failure mask the
        primary crash.  The save failure is recorded via
        ``_plugin_crash_partial_state_save_failed`` slog.

        Before the fix, the handler's ``except`` clause caught only
        ``IntegrityError``; any other ``SQLAlchemyError`` subclass escaped,
        producing a generic (unstructured) 500 and losing the redacted
        response path entirely.
        """
        from sqlalchemy.exc import OperationalError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="mid-crash-mutation"),
            version=3,
        )
        original = ValueError(f"plugin bug: {self.SECRET_PATH}")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=partial),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        # Patch save_composition_state to raise a non-IntegrityError
        # SQLAlchemyError subclass — simulates lock timeout / deadlock /
        # pool disconnect / schema drift.
        async def _raise_operational(*_args, **_kwargs):
            raise OperationalError(
                "UPDATE composition_states ...",
                {},
                Exception("lock wait timeout exceeded"),
            )

        service.save_composition_state = _raise_operational  # type: ignore[method-assign]

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        cap_logs = self._capture_route_slogs(monkeypatch)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        # Structured 500 body is preserved despite the secondary save failure.
        assert response.status_code == 500
        body = response.json()
        assert body["detail"]["error_type"] == "composer_plugin_error"
        assert self.SECRET_PATH not in response.text

        # The secondary failure is recorded via slog.
        save_fail_events = [e for e in cap_logs if e.get("event") == "compose_plugin_crash_partial_state_save_failed"]
        assert len(save_fail_events) == 1, cap_logs
        assert save_fail_events[0]["exc_class"] == "OperationalError"

        # And the primary plugin_crash slog still fires.
        crash_events = [e for e in cap_logs if e.get("event") == "compose_plugin_crash"]
        assert len(crash_events) == 1, cap_logs

    def test_compose_plugin_crash_save_failure_sets_partial_state_save_failed_flag(self, tmp_path) -> None:
        """Regression (P2d): when partial-state persistence fails during a
        plugin crash, the 500 response body MUST include
        ``partial_state_save_failed=True`` and ``partial_state_save_error``
        symmetric with the 422 convergence-error response. The frontend
        recovery UX branches on this flag to distinguish "state is
        captured, safe to retry later" from "state is lost, start over."
        Without the flag, the two plugin-crash outcomes (save success
        vs save failure) are indistinguishable to the client.
        """
        from sqlalchemy.exc import OperationalError

        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="mid-crash-mutation"),
            version=3,
        )
        original = ValueError("plugin bug")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=partial),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        async def _raise_operational(*_args, **_kwargs):
            raise OperationalError(
                "UPDATE composition_states ...",
                {},
                Exception("lock wait timeout exceeded"),
            )

        service.save_composition_state = _raise_operational  # type: ignore[method-assign]

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["error_type"] == "composer_plugin_error"
        # The two symmetry fields introduced to match _handle_convergence_error.
        assert detail.get("partial_state_save_failed") is True
        assert detail.get("partial_state_save_error") == "OperationalError"

    def test_compose_plugin_crash_save_successful_omits_partial_state_save_failed_flag(self, tmp_path) -> None:
        """Regression (P2d, negative): when partial-state persistence
        SUCCEEDS, the 500 response body MUST NOT carry a false
        ``partial_state_save_failed`` flag. The flag is a signal of
        recovery failure, not a constant field."""
        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="mid-crash-mutation"),
            version=3,
        )
        original = ValueError("plugin bug")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=partial),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["error_type"] == "composer_plugin_error"
        # Both keys absent on the success path — no false signal.
        assert "partial_state_save_failed" not in detail
        assert "partial_state_save_error" not in detail

    def test_compose_plugin_crash_save_typeerror_propagates_tier1_crash(self, tmp_path) -> None:
        """Regression (P2b): when ``save_composition_state`` raises a
        ``TypeError`` (our own code, Tier 1), the handler MUST let it
        propagate as an unstructured 500 rather than laundering it into
        ``partial_state_save_failed=True``. Pre-fix behaviour caught
        (ValueError, TypeError, KeyError, SQLAlchemyError) and produced
        a soft 500 with the flag set — which is exactly the silent-
        wrong-result pattern CLAUDE.md forbids for our own data.
        """
        partial = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="mid-crash-mutation"),
            version=3,
        )
        original = ValueError("plugin bug")
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=ComposerPluginCrashError(original, partial_state=partial),
        )

        app, service = _make_app(tmp_path)
        app.state.composer_service = mock_composer

        # TypeError from our own dataclass path — a Tier 1 bug that must propagate.
        async def _raise_type_error(*_args, **_kwargs):
            raise TypeError("dataclass field contract violated")

        service.save_composition_state = _raise_type_error  # type: ignore[method-assign]

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        # 500 from FastAPI's default handler — NOT the composer_plugin_error
        # structured body. The crashed request is the correct outcome; a
        # soft partial_state_save_failed=True would hide the Tier 1 bug.
        assert response.status_code == 500
        assert "composer_plugin_error" not in response.text
        assert "partial_state_save_failed" not in response.text

    def test_compose_unknown_exception_class_is_not_absorbed(self, tmp_path) -> None:
        """Deliberately narrow typed catch: RuntimeError (not in the handler's
        catch list) must propagate past the composer_plugin_error handler.
        With raise_server_exceptions=False, TestClient returns FastAPI's
        default 500 response; the critical invariant is that the structured
        composer_plugin_error body is NOT produced for unknown classes.
        """
        mock_composer = SimpleNamespace()
        mock_composer.compose = AsyncMock(
            spec=ComposerService.compose,
            side_effect=RuntimeError("unknown failure class"),
        )

        app, _ = _make_app(tmp_path)
        app.state.composer_service = mock_composer
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/sessions", json={"title": "Test"})
        session_id = resp.json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

        assert response.status_code == 500
        # Unconditional: the composer_plugin_error marker MUST NOT appear
        # anywhere in the response body, regardless of whether FastAPI
        # renders detail as a dict or a string.  This closes the vacuous-
        # pass risk of an `if isinstance(...)` guard.
        assert "composer_plugin_error" not in response.text


# ---------------------------------------------------------------------------
# Task 5: Persist Runtime Validity And Raw Model Text Across All Composer
# Write Paths
# ---------------------------------------------------------------------------


def test_runtime_preflight_errors_are_used_for_composition_state_persistence() -> None:
    from elspeth.web.sessions.routes import _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())
    runtime = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="plugin_instantiation",
                passed=False,
                detail="Invalid configuration for transform 'batch_stats'",
                affected_nodes=(),
                outcome_code=None,
            )
        ],
        errors=[
            ValidationError(
                component_id="agg1",
                component_type="transform",
                message="Invalid configuration for transform 'batch_stats'",
                suggestion="Remove required_input_fields from batch-aware transform options.",
                error_code=None,
            )
        ],
    )

    is_valid, messages = _composer_persisted_validation(authoring, runtime)

    assert is_valid is False
    assert messages == ["Invalid configuration for transform 'batch_stats'"]


def test_authoring_validity_is_not_marked_valid_when_runtime_preflight_failed_internally() -> None:
    """Legacy zero-arg sentinel still produces the bare ``runtime_preflight_failed``.

    The :data:`_RUNTIME_PREFLIGHT_FAILED` module-level constant is a
    no-diagnostics fallback constructed at import time. Production code
    paths now use :func:`_capture_runtime_preflight_failure` to populate
    the structured fields, but the bare sentinel is preserved as a
    defensive default and locks in the contract that authoring-valid +
    opaque-runtime-fail still persists as ``is_valid=False``.
    """
    from elspeth.web.sessions.routes import _RUNTIME_PREFLIGHT_FAILED, _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())

    is_valid, messages = _composer_persisted_validation(authoring, _RUNTIME_PREFLIGHT_FAILED)

    assert is_valid is False
    assert messages == ["runtime_preflight_failed"]


def test_runtime_preflight_failed_with_diagnostics_emits_structured_errors() -> None:
    """elspeth-2c3d63037c: replace opaque ``["runtime_preflight_failed"]``.

    Verifies that a populated :class:`_RuntimePreflightFailed` (the
    production path produced by :func:`_capture_runtime_preflight_failure`)
    yields a structured ``validation_errors`` list. The first entry MUST
    remain the legacy sentinel so SPA / LLM recovery loops keying on it
    still match; subsequent entries carry attribution that the prior
    opaque path forced operators to ``/state/revert`` to discover.
    """
    from elspeth.web.sessions.routes import _composer_persisted_validation, _RuntimePreflightFailed

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())
    runtime = _RuntimePreflightFailed(
        exception_class="AttributeError",
        exception_message_first_line="'NoneType' object has no attribute 'lower'",
        frames=(
            "frame=src/elspeth/web/execution/validation.py:436:validate_pipeline",
            "frame=src/elspeth/cli_helpers.py:120:instantiate_plugins_from_config",
        ),
    )

    is_valid, messages = _composer_persisted_validation(authoring, runtime)

    assert is_valid is False
    assert messages is not None
    assert messages[0] == "runtime_preflight_failed", (
        "Legacy sentinel must remain at index 0 — SPA / LLM parsers "
        "key on this exact string and would silently fail to detect "
        "the failure class if it moved or changed."
    )
    assert "exception_class=AttributeError" in messages
    assert any(m.startswith("exception_message=") for m in messages)
    assert sum(1 for m in messages if m.startswith("frame=")) == 2


def test_capture_runtime_preflight_failure_redacts_locals_and_source() -> None:
    """Frames must contain only file:line:function — no source, no values.

    Frame strings flow into the persisted audit row and a structured
    server log; secret-bearing plugin config values, DB connection
    strings, and bound SQL parameters travel through ``str(exc)`` /
    ``__cause__`` chains and through ``traceback.format_exception``'s
    source-line and locals-render output. The capture helper's contract
    is to redact everything except the structural ``frame=path:line:func``
    triple. This test pins the contract.
    """
    from elspeth.web.sessions.routes import _capture_runtime_preflight_failure

    secret = "API_KEY=sk-LIVE-MUST-NOT-LEAK-ABCDEF"  # secret-scan: allow-this-line

    def deeper() -> None:
        local_secret = secret
        raise RuntimeError(f"plugin bug referencing {local_secret}")

    try:
        deeper()
    except RuntimeError as exc:
        captured = _capture_runtime_preflight_failure(exc)

    # Exception class and a redacted message line are captured.
    assert captured.exception_class == "RuntimeError"
    assert "plugin bug" in captured.exception_message_first_line
    assert "sk-LIVE" in captured.exception_message_first_line, (
        "Test relies on knowing the message text contains the secret — if this assertion fails the test setup is wrong, not the code."
    )

    # Frames are file:line:function ONLY. Source-line text and local
    # repr (which would include ``secret`` and ``local_secret``) MUST
    # NOT appear in the frame strings.
    assert captured.frames, "expected at least one frame from the live traceback"
    for frame in captured.frames:
        assert frame.startswith("frame="), frame
        assert "sk-LIVE" not in frame
        assert "API_KEY" not in frame
        # No source-line capture — frames are structural only.
        assert "raise RuntimeError" not in frame
        assert "local_secret" not in frame


def test_state_data_carries_structured_errors_before_save_for_atomicity() -> None:
    """elspeth-2c3d63037c sub-fix 3: structured errors must be baked into
    the :class:`CompositionStateData` DTO BEFORE
    :meth:`SessionServiceProtocol.save_composition_state` is called.

    Atomicity contract: ``save_composition_state`` writes
    ``version``, ``is_valid``, and ``validation_errors`` in a single
    SQLAlchemy ``engine.begin()`` transaction (see
    :meth:`SessionServiceImpl.save_composition_state`). Any path that
    would let the version bump commit without the structured errors
    would re-introduce the opaque-sentinel symptom by accident — a
    reader of the new row would see ``is_valid=False`` with no
    attribution.

    This test pins the contract at the helper's seam: if a future
    refactor moves the ``_RuntimePreflightFailed`` build site out from
    under :func:`_state_data_from_composer_state` so the structured
    errors are filled in *after* the DTO is constructed, the resulting
    DTO would carry the legacy bare sentinel and this assertion would
    catch it.
    """
    import asyncio

    from elspeth.web.sessions.routes import _state_data_from_composer_state

    # Authoring-valid state — otherwise runtime preflight is skipped and the
    # structured-errors path under test is never exercised.
    state = _make_authoring_valid_partial("atomicity-test")

    async def boom(state, *, settings, secret_service, user_id, session_id, plugin_snapshot, profile_registry, catalog):
        del plugin_snapshot, profile_registry, catalog
        raise AttributeError("'NoneType' has no attribute 'something_that_failed'")

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom):
        state_data, _validation = asyncio.run(
            _state_data_from_composer_state(
                state,
                settings=object(),
                secret_service=None,
                user_id="alice",
                session_id="session-123",
                plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
                profile_registry=MagicMock(spec=OperatorProfileRegistry),
                catalog=create_catalog_service(),
                runtime_preflight=None,
                preflight_exception_policy="persist_invalid",
                initial_version=None,
                telemetry_source="compose",
            ),
        )

    # The DTO that will be passed to save_composition_state already
    # carries the structured attribution. The DB-side INSERT is
    # transactional (see SessionServiceImpl.save_composition_state),
    # so no observable row at v(N+1) can lack these fields.
    assert state_data.is_valid is False
    errors = list(state_data.validation_errors or ())
    assert errors[0] == "runtime_preflight_failed"
    assert "exception_class=AttributeError" in errors
    assert any(e.startswith("exception_message=") for e in errors)
    assert any(e.startswith("frame=") for e in errors)


@pytest.mark.asyncio
async def test_runtime_preflight_for_state_threads_session_id_to_validate_pipeline(monkeypatch) -> None:
    """The runtime wrapper must preserve the session-scoped sink allowlist."""
    from elspeth.web.sessions.routes import _helpers as routes

    state = _make_authoring_valid_partial("runtime-wrapper-session")
    settings = SimpleNamespace(composer_runtime_preflight_timeout_seconds=1)
    plugin_snapshot = MagicMock(spec=PluginAvailabilitySnapshot)
    profile_registry = MagicMock(spec=OperatorProfileRegistry)
    catalog = create_catalog_service()
    seen_kwargs: list[dict[str, object]] = []

    async def fake_run_sync_in_worker(func, *args, **kwargs):
        assert func is routes.validate_pipeline
        del args
        seen_kwargs.append(dict(kwargs))
        return ValidationResult(is_valid=True, checks=[], errors=[])

    monkeypatch.setattr(routes, "run_sync_in_worker", fake_run_sync_in_worker)

    await routes._runtime_preflight_for_state(
        state,
        settings=settings,
        secret_service=None,
        user_id="alice",
        session_id="session-123",
        plugin_snapshot=plugin_snapshot,
        profile_registry=profile_registry,
        catalog=catalog,
    )

    assert seen_kwargs == [
        {
            "secret_service": None,
            "user_id": "alice",
            "session_id": "session-123",
            "plugin_snapshot": plugin_snapshot,
            "profile_registry": profile_registry,
            "catalog": catalog,
        }
    ]


@pytest.mark.asyncio
async def test_state_data_threads_session_id_to_runtime_preflight() -> None:
    """Persisting state must validate blob sinks against the owning session."""
    from elspeth.web.sessions.routes import _helpers as routes

    state = _make_authoring_valid_partial("session-scoped-preflight")
    seen_session_ids: list[str] = []

    async def capture_session_id(state, *, settings, secret_service, user_id, session_id, plugin_snapshot, profile_registry, catalog):
        del state, settings, secret_service, user_id, plugin_snapshot, profile_registry, catalog
        seen_session_ids.append(session_id)
        return ValidationResult(is_valid=True, checks=[], errors=[])

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=capture_session_id):
        await routes._state_data_from_composer_state(
            state,
            settings=object(),
            secret_service=None,
            user_id="alice",
            session_id="session-123",
            plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
            catalog=create_catalog_service(),
            runtime_preflight=None,
            preflight_exception_policy="persist_invalid",
            initial_version=None,
            telemetry_source="compose",
        )

    assert seen_session_ids == ["session-123"]


@pytest.mark.asyncio
async def test_state_data_from_composer_state_uses_profile_aware_authoring_validation() -> None:
    from elspeth.web.sessions.routes import _helpers as routes

    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    state = _make_authoring_valid_partial("profile-aware-state-data")
    runtime = ValidationResult(is_valid=True, checks=[], errors=[])

    with patch(
        "elspeth.web.sessions.routes._helpers.validate_authored_composition_state",
        wraps=routes.validate_authored_composition_state,
    ) as validate:
        _state_data, summary = await routes._state_data_from_composer_state(
            state,
            settings=object(),
            secret_service=None,
            user_id="alice",
            session_id="session-123",
            plugin_snapshot=snapshot,
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
            catalog=catalog,
            runtime_preflight=runtime,
            preflight_exception_policy="persist_invalid",
            initial_version=None,
            telemetry_source="compose",
        )

    validate.assert_called_once()
    assert summary == state.validate()


@pytest.mark.asyncio
async def test_state_data_persists_structured_implicit_decisions_report() -> None:
    """elspeth-457c8688ef: successful compose saves must carry a structured
    implicit-decision report in ``composition_states.composer_meta``.

    The skill prompt already tells the LLM to include a natural-language
    "Decisions I made on your behalf" section, but auditors need a persisted
    machine-readable sidecar on reload. The report is generated from the state
    that is about to be saved, before the DB write, so the new version and its
    disclosure are atomic.
    """
    from elspeth.contracts.freeze import deep_freeze
    from elspeth.web.composer.state import EdgeSpec, NodeSpec, OutputSpec, SourceSpec
    from elspeth.web.sessions.routes import _state_data_from_composer_state

    state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            options=deep_freeze(
                {
                    "path": "blobs/session-input.csv",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                }
            ),
            on_success="url_rows",
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="fetch_pages",
                node_type="transform",
                plugin="web_scrape",
                input="url_rows",
                on_success="scraped_content",
                on_error="discard",
                options=deep_freeze(
                    {
                        "schema": {"mode": "fixed", "fields": ["url: str"]},
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "format": "markdown",
                        "http": {
                            "abuse_contact": "ops@agency.gov.au",
                            "scraping_reason": "Front-page summarisation of three .gov.au sites",
                            "allowed_hosts": "public_only",
                        },
                    }
                ),
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="summarise",
                node_type="transform",
                plugin="llm",
                input="scraped_content",
                on_success="summaries",
                on_error="discard",
                options=deep_freeze(
                    {
                        "provider": "openrouter",
                        "model": "anthropic/claude-sonnet-4",
                        "temperature": 0,
                        "pool_size": 1,
                        "prompt_template": "Summarise {{ row.content }} as JSON.",
                    }
                ),
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(
            EdgeSpec(id="e1", from_node="source", to_node="fetch_pages", edge_type="on_success", label=None),
            EdgeSpec(id="e2", from_node="fetch_pages", to_node="summarise", edge_type="on_success", label=None),
            EdgeSpec(id="e3", from_node="summarise", to_node="summaries_out", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="summaries_out",
                plugin="json",
                options=deep_freeze({"path": "outputs/summaries_out.json", "collision_policy": "auto_increment"}),
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Gov AU summaries"),
        version=2,
    )

    state_data, _validation = await _state_data_from_composer_state(
        state,
        settings=object(),
        secret_service=None,
        user_id="alice",
        session_id="session-123",
        plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
        profile_registry=MagicMock(spec=OperatorProfileRegistry),
        catalog=create_catalog_service(),
        runtime_preflight=ValidationResult(is_valid=True, checks=[], errors=[]),
        preflight_exception_policy="raise",
        initial_version=1,
        telemetry_source="compose",
        composer_meta={"repair_turns_used": 0},
    )

    assert state_data.composer_meta is not None
    report = state_data.composer_meta["implicit_decisions"]
    assert report["schema_version"] == 1
    by_path = {entry["path"]: entry for entry in report["entries"]}

    assert by_path["node.fetch_pages.options.http.abuse_contact"]["value"] == "ops@agency.gov.au"
    assert by_path["node.fetch_pages.options.http.abuse_contact"]["provenance"] == "explicit_source_required"
    assert list(by_path["node.fetch_pages.options.format"]["candidate_alternatives"]) == ["html", "markdown", "text"]
    assert by_path["node.summarise.options.model"]["value"] == "anthropic/claude-sonnet-4"
    assert by_path["node.summarise.options.temperature"]["provenance"] == "picked"
    assert by_path["output.summaries_out.options.path"]["value"] == "outputs/summaries_out.json"
    assert by_path["output.summaries_out.options.collision_policy"]["provenance"] == "default"
    assert by_path["node.fetch_pages.on_error"]["category"] == "error_routing"
    assert list(report["normalization_events"]) == []


def test_runtime_preflight_failure_500_detail_does_not_promise_journal_traceback() -> None:
    """elspeth-2c3d63037c: 500 detail must not claim "see server logs".

    The prior wording promised a traceback in journald that the helpers
    deliberately do not emit (per the secret-leak guard on
    ``slog.error(... exc_info omitted)``). Operators following the
    promise found nothing. The new wording must point at the persisted
    state's ``validation_errors`` row — the actual diagnostic surface.
    """
    import asyncio
    from uuid import UUID as _UUID

    from elspeth.web.composer.protocol import ComposerRuntimePreflightError
    from elspeth.web.sessions.routes import _handle_runtime_preflight_failure

    exc = ComposerRuntimePreflightError(
        original_exc=RuntimeError("boom"),
        partial_state=None,
    )
    service = SimpleNamespace()

    body = asyncio.run(
        _handle_runtime_preflight_failure(
            exc,
            service,
            _UUID("00000000-0000-4000-8000-000000000001"),
            "user-id",
            "compose",
            None,
            settings=object(),
            secret_service=None,
            plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
            catalog=create_catalog_service(),
        ),
    )

    assert body["error_type"] == "composer_plugin_error"
    detail_text = str(body["detail"]).lower()
    assert "see server logs" not in detail_text, (
        "The 500 detail must not promise a journald traceback that the "
        "helpers deliberately do not emit — that's exactly the diagnostic "
        "lie elspeth-2c3d63037c was filed for."
    )
    assert "validation_errors" in detail_text or "persisted state" in detail_text


def test_composer_persisted_validation_has_no_split_runtime_failed_knob() -> None:
    import inspect

    from elspeth.web.sessions.routes import _composer_persisted_validation

    params = inspect.signature(_composer_persisted_validation).parameters

    assert "preflight_failed" not in params
    assert list(params) == ["authoring", "runtime_preflight"]


def test_authoring_valid_state_without_runtime_outcome_is_rejected() -> None:
    from elspeth.web.sessions.routes import _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())

    with pytest.raises(ValueError, match="requires runtime preflight outcome"):
        _composer_persisted_validation(authoring, None)


@pytest.mark.asyncio
async def test_state_data_from_composer_state_propagates_to_dict_errors() -> None:
    from elspeth.web.composer.state import ValidationEntry
    from elspeth.web.sessions.routes import _state_data_from_composer_state

    state = MagicMock(spec=CompositionState)
    state.version = 1
    state.validate.return_value = ValidationSummary(
        is_valid=False,
        errors=(ValidationEntry("validation", "validation_failed", "high"),),
        warnings=(),
        suggestions=(),
    )
    state.to_dict.side_effect = TypeError("broken Tier 1 state")

    with pytest.raises(TypeError, match="broken Tier 1 state"):
        await _state_data_from_composer_state(
            state,
            settings=object(),
            secret_service=None,
            user_id="user-1",
            session_id="session-123",
            plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
            catalog=create_catalog_service(),
            runtime_preflight=None,
            preflight_exception_policy="persist_invalid",
            initial_version=None,
            telemetry_source="compose",
        )


def test_runtime_preflight_telemetry_uses_bounded_attributes(monkeypatch) -> None:
    from elspeth.web.sessions.routes import _helpers as routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    monkeypatch.setattr(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
    monkeypatch.setattr(routes, "_COMPOSER_AUTHORING_VALIDATION_COUNTER", FakeCounter())

    routes._record_composer_runtime_preflight_telemetry(
        "exception",
        source="compose",
        exception_class="RuntimeError",
    )
    routes._record_composer_runtime_preflight_telemetry(
        "exception",
        source="compose",
        exception_class="AdversarialPluginFailure_9c5dbf3e",
    )
    routes._record_composer_authoring_validation_telemetry(
        "exception",
        source="compose",
        exception_class="RuntimeError",
    )

    assert emitted == [
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "RuntimeError",
            },
        ),
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "other",
            },
        ),
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "RuntimeError",
            },
        ),
    ]


def _runtime_preflight_failed_result(message: str = "runtime preflight blocked export") -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="plugin_instantiation",
                passed=False,
                detail=message,
                affected_nodes=(),
                outcome_code=None,
            )
        ],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message=message,
                suggestion=None,
                error_code=None,
            )
        ],
    )


def test_recompose_success_persists_runtime_invalid_state(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline", writer_principal="route_user_message"))
    finally:
        loop.close()

    changed_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="runtime-invalid-recompose"),
        version=_EMPTY_STATE.version + 1,
    )
    runtime_preflight = _runtime_preflight_failed_result("runtime failure from recompose")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="I cannot mark this pipeline complete yet.",
            state=changed_state,
            runtime_preflight=runtime_preflight,
            # I6 invariant: failed runtime_preflight requires the original LLM
            # text to be parked in raw_assistant_content so the audit trail
            # can recover what the LLM actually said before the synthetic
            # replacement was substituted into ``message``.
            raw_assistant_content="The pipeline is complete and valid.",
        ),
    )
    app.state.composer_service = mock_composer

    recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

    assert recompose_resp.status_code == 200
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "runtime-invalid-recompose"
    assert persisted.is_valid is False
    assert persisted.validation_errors is not None
    assert list(persisted.validation_errors) == ["runtime failure from recompose"]


def test_recompose_convergence_persists_runtime_invalid_partial_state(tmp_path) -> None:
    from elspeth.contracts.freeze import deep_freeze
    from elspeth.web.composer.protocol import ComposerConvergenceError
    from elspeth.web.composer.state import EdgeSpec, OutputSpec, SourceSpec

    # Authoring-valid partial state so runtime preflight is invoked.
    # Source → "out" sink with a single edge satisfies authoring validation
    # (source present, output present, edge references valid nodes).
    partial = CompositionState(
        source=SourceSpec(
            plugin="csv",
            options=deep_freeze({"path": "/x.csv"}),
            on_success="out",
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(EdgeSpec(id="e1", from_node="source", to_node="out", edge_type="on_success", label=None),),
        outputs=(OutputSpec(name="out", plugin="json", options=deep_freeze({}), on_write_failure="discard"),),
        metadata=PipelineMetadata(name="partial-after-convergence"),
        version=2,
    )
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerConvergenceError(
            max_turns=5,
            budget_exhausted="composition",
            partial_state=partial,
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline", writer_principal="route_user_message"))
    finally:
        loop.close()

    runtime_preflight = _runtime_preflight_failed_result("runtime failure from convergence")
    with patch(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        new=_async_return(runtime_preflight),
    ):
        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

    assert recompose_resp.status_code == 422
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-convergence"
    assert persisted.is_valid is False
    assert persisted.validation_errors is not None
    assert list(persisted.validation_errors) == ["runtime failure from convergence"]


def test_compose_plugin_crash_persists_runtime_invalid_partial_state(tmp_path) -> None:
    from elspeth.contracts.freeze import deep_freeze
    from elspeth.web.composer.state import EdgeSpec, OutputSpec, SourceSpec

    # Authoring-valid partial state so runtime preflight is invoked (same
    # pattern as the convergence test above).
    partial = CompositionState(
        source=SourceSpec(
            plugin="csv",
            options=deep_freeze({"path": "/x.csv"}),
            on_success="out",
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(EdgeSpec(id="e1", from_node="source", to_node="out", edge_type="on_success", label=None),),
        outputs=(OutputSpec(name="out", plugin="json", options=deep_freeze({}), on_write_failure="discard"),),
        metadata=PipelineMetadata(name="partial-after-plugin-crash"),
        version=5,
    )
    original = ValueError("plugin bug after mutation")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerPluginCrashError(original, partial_state=partial),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    runtime_preflight = _runtime_preflight_failed_result("runtime failure from plugin crash")
    with patch(
        "elspeth.web.sessions.routes._helpers._runtime_preflight_for_state",
        new=_async_return(runtime_preflight),
    ):
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "composer_plugin_error"
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-plugin-crash"
    assert persisted.is_valid is False
    assert persisted.validation_errors is not None
    assert list(persisted.validation_errors) == ["runtime failure from plugin crash"]


# ---------------------------------------------------------------------------
# ComposerRuntimePreflightError handler — C1 regression coverage
# ---------------------------------------------------------------------------
#
# When _state_data_from_composer_state raises ComposerRuntimePreflightError
# (preflight_exception_policy="raise" path inside _state_data_from_composer_state),
# the route-level catch handlers (send_message at routes.py:1039 and
# recompose at routes.py:1344) MUST be symmetric with _handle_plugin_crash:
# persist rpf_exc.partial_state into composition_states, increment the
# composer.runtime_preflight.total{result=exception, source=runtime_preflight}
# counter, and surface the partial_state_save_failed/_save_error fields on
# the 500 body when DB persistence itself fails.
#
# The original stubs raised HTTPException(500) directly without persisting,
# silently dropping accumulated tool-call mutations from the audit trail —
# an audit-primacy violation per CLAUDE.md.


def _make_authoring_valid_partial(name: str, version: int = 5) -> CompositionState:
    """Build a partial CompositionState whose authoring validation passes,
    so the runtime-preflight branch inside _state_data_from_composer_state
    is reached when the state is re-validated by the recovery handler.
    """
    from elspeth.contracts.freeze import deep_freeze
    from elspeth.web.composer.state import EdgeSpec, OutputSpec, SourceSpec

    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            options=deep_freeze({"path": "/x.csv"}),
            on_success="out",
            on_validation_failure="quarantine",
        ),
        nodes=(),
        edges=(EdgeSpec(id="e1", from_node="source", to_node="out", edge_type="on_success", label=None),),
        outputs=(OutputSpec(name="out", plugin="json", options=deep_freeze({}), on_write_failure="discard"),),
        metadata=PipelineMetadata(name=name),
        version=version,
    )


def test_compose_runtime_preflight_persists_partial_state(tmp_path) -> None:
    """C1: send_message's ComposerRuntimePreflightError catch (routes.py:1039)
    MUST persist rpf_exc.partial_state with is_valid=False, mirroring
    _handle_plugin_crash. Without persistence the tool-call mutations
    accumulated in result.state are silently dropped from the audit trail.
    """
    partial = _make_authoring_valid_partial("partial-after-runtime-preflight")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="The composer mutated state but preflight then failed.",
            state=partial,
            runtime_preflight=None,  # forces the route to re-run preflight via _state_data_from_composer_state
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    async def boom(state, *, settings, secret_service, user_id, session_id, **_policy_context):
        raise RuntimeError("preflight blew up during state persistence")

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom):
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error_type"] == "composer_plugin_error"

    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None, (
        "partial_state from ComposerRuntimePreflightError MUST be persisted to "
        "composition_states; without this, accumulated tool-call mutations are "
        "silently dropped from the audit trail."
    )
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-runtime-preflight"
    assert persisted.is_valid is False
    # _composer_persisted_validation maps _RuntimePreflightFailed →
    # structured diagnostic strings (elspeth-2c3d63037c). The first entry
    # remains the legacy sentinel so external parsers keying on it still
    # match; subsequent entries carry exception_class + first-line message
    # + bounded file:line:function frames.
    assert persisted.validation_errors is not None
    errors = list(persisted.validation_errors)
    assert errors[0] == "runtime_preflight_failed"
    assert "exception_class=RuntimeError" in errors
    assert any(e.startswith("exception_message=") for e in errors)
    assert any(e.startswith("frame=") for e in errors), (
        "Frame strings (file:line:function) must be persisted for triage; "
        "without them the operator only sees the exception class and is "
        "back to the opaque-sentinel state."
    )


def test_recompose_runtime_preflight_persists_partial_state(tmp_path) -> None:
    """C1: recompose's ComposerRuntimePreflightError catch (routes.py:1344)
    MUST behave identically to send_message's catch — partial_state persistence
    is part of the contract, not a per-route concern.
    """
    partial = _make_authoring_valid_partial("partial-after-recompose-preflight")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="Recompose mutated state but preflight then failed.",
            state=partial,
            runtime_preflight=None,
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Recompose runtime preflight"})
    session_id = uuid.UUID(resp.json()["id"])

    # Recompose precondition: last persisted message must be a user turn.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline", writer_principal="route_user_message"))
    finally:
        loop.close()

    async def boom(state, *, settings, secret_service, user_id, session_id, **_policy_context):
        raise RuntimeError("preflight blew up on recompose")

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom):
        response = client.post(f"/api/sessions/{session_id}/recompose")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error_type"] == "composer_plugin_error"

    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-recompose-preflight"
    assert persisted.is_valid is False
    # See sibling test in test_compose_runtime_preflight_persists_partial_state
    # for the structured-error rationale (elspeth-2c3d63037c).
    assert persisted.validation_errors is not None
    errors = list(persisted.validation_errors)
    assert errors[0] == "runtime_preflight_failed"
    assert "exception_class=RuntimeError" in errors
    assert any(e.startswith("exception_message=") for e in errors)
    assert any(e.startswith("frame=") for e in errors)


def test_compose_cached_runtime_preflight_persists_partial_state(tmp_path) -> None:
    """C1 path-1 lock-in: when composer.compose() itself raises
    ComposerRuntimePreflightError (the cached-preflight path in
    web/composer/service.py:_raise_cached_runtime_preflight_failure), the
    catch at routes.py:1039 MUST persist partial_state via the shared
    handler. Without this lock-in, a future refactor that breaks the helper
    wiring on path-1 (compose-time) would slip through coverage that only
    exercises path-2 (post-compose state-save).
    """
    from elspeth.web.composer.protocol import ComposerRuntimePreflightError

    partial = _make_authoring_valid_partial("partial-from-cached-preflight")
    original = RuntimeError("cached preflight failure from composer.compose")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerRuntimePreflightError(original_exc=original, partial_state=partial),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Cached preflight"})
    session_id = uuid.UUID(resp.json()["id"])

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "composer_plugin_error"

    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None, (
        "partial_state from cached-preflight ComposerRuntimePreflightError "
        "(raised inside composer.compose()) MUST be persisted by the "
        "routes.py:1039 catch handler."
    )
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-from-cached-preflight"
    assert persisted.is_valid is False


def test_runtime_preflight_handler_records_exception_telemetry(tmp_path, monkeypatch) -> None:
    """C1: The handler MUST increment composer.runtime_preflight.total with
    {result=exception, source=runtime_preflight, exception_class=...} so
    operators can distinguish recovery-path failures from primary-path
    failures on dashboards. Without telemetry, runtime-preflight crashes
    are invisible to monitoring.
    """
    from elspeth.web.sessions.routes import _helpers as routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    monkeypatch.setattr(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())

    partial = _make_authoring_valid_partial("telemetry-runtime-preflight")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="ok",
            state=partial,
            runtime_preflight=None,
        ),
    )

    app, _service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Telemetry"})
    session_id = uuid.UUID(resp.json()["id"])

    async def boom(state, *, settings, secret_service, user_id, session_id, **_policy_context):
        raise RuntimeError("preflight crashed")

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom):
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )
    assert response.status_code == 500

    # I1 dual-emission contract — path-2 failures must emit telemetry at TWO
    # distinct attribution points:
    #
    #   (a) Primary failure site: _state_data_from_composer_state's raise arm
    #       fires telemetry with source="compose" before propagating the
    #       ComposerRuntimePreflightError. This preserves the originating-route
    #       attribution that would otherwise be lost when the route handler
    #       relabels the recovery emission as source="runtime_preflight".
    #
    #   (b) Recovery site: _handle_runtime_preflight_failure's re-call to
    #       _state_data_from_composer_state with policy="persist_invalid" and
    #       telemetry_source="runtime_preflight" emits a second event,
    #       attributed to the recovery handler.
    #
    # Operators distinguish "primary failure rate" (filter source∈{compose,
    # recompose,plugin_crash,convergence,yaml_export}) from "recovery handler
    # invocations" (filter source=runtime_preflight) using these two emissions.
    primary_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "compose" and attrs.get("result") == "exception" and attrs.get("exception_class") == "RuntimeError"
    ]
    recovery_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "runtime_preflight"
        and attrs.get("result") == "exception"
        and attrs.get("exception_class") == "RuntimeError"
    ]
    assert primary_emissions, (
        "Primary failure attribution missing: _state_data_from_composer_state "
        "raise arm MUST emit composer.runtime_preflight.total{source=compose, "
        f"result=exception, exception_class=RuntimeError}}; emitted={emitted}"
    )
    assert recovery_emissions, (
        "Recovery attribution missing: _handle_runtime_preflight_failure MUST "
        "emit composer.runtime_preflight.total{source=runtime_preflight, "
        f"result=exception, exception_class=RuntimeError}}; emitted={emitted}"
    )


def test_compose_cached_runtime_preflight_no_partial_state_records_telemetry(tmp_path, monkeypatch) -> None:
    """Path-1 silent-failure lock-in (elspeth-0891e8da73): when
    composer.compose() re-raises a previously-cached runtime-preflight
    failure (web/composer/service.py:_raise_cached_runtime_preflight_failure)
    AND the LLM never mutated state before the cached re-raise (so
    partial_state is None per ComposerRuntimePreflightError.capture's rule),
    the route catch handler MUST still emit telemetry on
    composer.runtime_preflight.total. Without this, dashboards under-count
    cached-preflight failures by exactly the count of "no LLM mutation
    before cached failure re-raise" events — operators see neither a
    primary nor a recovery emission, violating CLAUDE.md telemetry primacy
    ("every telemetry emission point must send or explicitly acknowledge
    'nothing to send.'").

    Two emissions are required:
      (a) Primary: source="cached_preflight" — attributes the failure to
          the cached re-raise raise site, distinct from path-2's
          source="compose" attribution. The outer
          ``except ComposerRuntimePreflightError`` at routes.py reaches the
          cached path only (path-2 is caught inline around
          _state_data_from_composer_state).
      (b) Recovery: source="runtime_preflight" — the recovery handler
          ran, even though it had no partial_state to persist. Without
          this acknowledgment, the recovery counter under-counts handler
          invocations.

    No source="compose" emission is allowed: that label belongs to the
    path-2 primary site (_state_data_from_composer_state's raise arm) and
    must not be relabeled onto the cached path.
    """
    from elspeth.web.composer.protocol import ComposerRuntimePreflightError
    from elspeth.web.sessions.routes import _helpers as routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    monkeypatch.setattr(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())

    original = RuntimeError("cached preflight failure with no LLM mutation")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerRuntimePreflightError(
            original_exc=original,
            partial_state=None,
        ),
    )

    app, _service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Cached preflight no partial"})
    session_id = uuid.UUID(resp.json()["id"])

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "composer_plugin_error"

    cached_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "cached_preflight"
        and attrs.get("result") == "exception"
        and attrs.get("exception_class") == "RuntimeError"
    ]
    recovery_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "runtime_preflight"
        and attrs.get("result") == "exception"
        and attrs.get("exception_class") == "RuntimeError"
    ]
    compose_emissions = [attrs for _, attrs in emitted if attrs.get("source") == "compose"]

    assert cached_emissions, (
        "Primary cached_preflight attribution missing: route catch MUST emit "
        "composer.runtime_preflight.total{result=exception, source=cached_preflight, "
        f"exception_class=RuntimeError}}; emitted={emitted}"
    )
    assert recovery_emissions, (
        "Recovery acknowledgment missing: _handle_runtime_preflight_failure with "
        "partial_state=None MUST still emit composer.runtime_preflight.total"
        "{result=exception, source=runtime_preflight, exception_class=...} so the "
        f"recovery handler invocation count remains complete; emitted={emitted}"
    )
    assert not compose_emissions, (
        "Path-2 attribution leakage: cached path MUST NOT relabel as source=compose "
        f"(that's _state_data_from_composer_state's primary site); emitted={emitted}"
    )


def test_recompose_cached_runtime_preflight_no_partial_state_records_telemetry(tmp_path, monkeypatch) -> None:
    """Recompose mirror of
    test_compose_cached_runtime_preflight_no_partial_state_records_telemetry.
    The recompose endpoint's outer ComposerRuntimePreflightError catch
    handles the same path-1 case symmetrically with send_message; the
    telemetry contract is identical.
    """
    from elspeth.web.composer.protocol import ComposerRuntimePreflightError
    from elspeth.web.sessions.routes import _helpers as routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    monkeypatch.setattr(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())

    original = RuntimeError("cached preflight failure with no LLM mutation on recompose")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerRuntimePreflightError(
            original_exc=original,
            partial_state=None,
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post(
        "/api/sessions",
        json={"title": "Cached preflight no partial recompose"},
    )
    session_id = uuid.UUID(resp.json()["id"])

    # Recompose precondition: last persisted message must be a user turn.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline", writer_principal="route_user_message"))
    finally:
        loop.close()

    response = client.post(f"/api/sessions/{session_id}/recompose")

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "composer_plugin_error"

    cached_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "cached_preflight"
        and attrs.get("result") == "exception"
        and attrs.get("exception_class") == "RuntimeError"
    ]
    recovery_emissions = [
        attrs
        for _, attrs in emitted
        if attrs.get("source") == "runtime_preflight"
        and attrs.get("result") == "exception"
        and attrs.get("exception_class") == "RuntimeError"
    ]
    recompose_emissions = [attrs for _, attrs in emitted if attrs.get("source") == "recompose"]

    assert cached_emissions, f"Primary cached_preflight attribution missing on recompose path; emitted={emitted}"
    assert recovery_emissions, f"Recovery acknowledgment missing on recompose path; emitted={emitted}"
    assert not recompose_emissions, (
        "Path-2 attribution leakage: cached path MUST NOT relabel as source=recompose "
        f"(that's _state_data_from_composer_state's primary site); emitted={emitted}"
    )


@pytest.mark.asyncio
async def test_state_data_raise_arm_emits_telemetry_before_propagating() -> None:
    """I1 unit-level lock-in: when _state_data_from_composer_state runs with
    preflight_exception_policy="raise" and the underlying preflight raises,
    the function MUST emit composer.runtime_preflight.total with the source
    label passed in (e.g. "compose" or "recompose") BEFORE propagating
    ComposerRuntimePreflightError. Without this emission, primary failure
    attribution is lost — the only remaining telemetry comes from the
    recovery handler's re-call, which uses source="runtime_preflight" and
    cannot reveal which originating route triggered the failure.
    """
    from elspeth.web.composer.protocol import ComposerRuntimePreflightError
    from elspeth.web.sessions.routes import _helpers as routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    state = _make_authoring_valid_partial("raise-arm-telemetry")

    async def boom(state, *, settings, secret_service, user_id, session_id, plugin_snapshot, profile_registry, catalog):
        del plugin_snapshot, profile_registry, catalog
        raise RuntimeError("preflight raised at primary site")

    with (
        patch.object(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter()),
        patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom),
        pytest.raises(ComposerRuntimePreflightError) as excinfo,
    ):
        await routes._state_data_from_composer_state(
            state,
            settings=object(),
            secret_service=None,
            user_id="user-1",
            session_id="session-123",
            plugin_snapshot=PluginAvailabilitySnapshot.for_trained_operator(create_catalog_service()),
            profile_registry=MagicMock(spec=OperatorProfileRegistry),
            catalog=create_catalog_service(),
            runtime_preflight=None,
            preflight_exception_policy="raise",
            initial_version=1,
            telemetry_source="compose",
        )

    # The exception still propagates with the original cause attached, so
    # downstream callers can extract partial_state and log __cause__.
    assert isinstance(excinfo.value.original_exc, RuntimeError)

    # Telemetry MUST have fired with the source the caller passed in
    # ("compose"), NOT the recovery-handler's "runtime_preflight" label.
    assert any(
        attrs.get("source") == "compose" and attrs.get("result") == "exception" and attrs.get("exception_class") == "RuntimeError"
        for _, attrs in emitted
    ), (
        "raise arm MUST emit composer.runtime_preflight.total{source=compose, "
        f"result=exception, exception_class=RuntimeError}} before propagating; emitted={emitted}"
    )


def test_runtime_preflight_handler_save_failure_sets_partial_state_save_failed_flag(
    tmp_path,
) -> None:
    """C1: When DB persistence of partial_state fails inside the recovery
    handler, the 500 response MUST carry partial_state_save_failed=True and
    partial_state_save_error=<class name>, matching the convergence/plugin-crash
    contract so the frontend can distinguish "state captured" from
    "state lost". The save failure MUST NOT mask the primary 500 response.
    """
    from sqlalchemy.exc import OperationalError

    partial = _make_authoring_valid_partial("save-fail-runtime-preflight")
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        return_value=ComposerResult(
            message="ok",
            state=partial,
            runtime_preflight=None,
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer

    # Canary strings to verify the response body never echoes SQL/parameter
    # internals (mirrors the convergence/plugin-crash redaction contract).
    sql_canary = "__CANARY_SQL_INSERT_composition_states_runtime_preflight__"
    params_canary = "__CANARY_PARAM_runtime_preflight_payload__"
    cause_canary = "__CANARY_CAUSE_runtime_preflight_db_url__"

    async def _raise_operational(*_args, **_kwargs):
        raise OperationalError(
            f"INSERT INTO composition_states (...) -- {sql_canary}",
            {"source_options": params_canary},
            Exception(f"connection closed: {cause_canary}"),
        )

    service.save_composition_state = _raise_operational  # type: ignore[method-assign]

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/sessions", json={"title": "Save fail"})
    session_id = uuid.UUID(resp.json()["id"])

    async def boom(state, *, settings, secret_service, user_id, session_id, **_policy_context):
        raise RuntimeError("preflight crashed before save")

    with patch("elspeth.web.sessions.routes._helpers._runtime_preflight_for_state", side_effect=boom):
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

    assert response.status_code == 500
    body_text = response.text
    # Redaction parity: no SQL / parameter / __cause__ leakage in the body.
    assert sql_canary not in body_text, "SQL statement leaked into HTTP response body"
    assert params_canary not in body_text, "parameter tuple leaked into HTTP response body"
    assert cause_canary not in body_text, "DBAPI __cause__ text leaked into HTTP response body"

    detail = response.json()["detail"]
    assert detail["error_type"] == "composer_plugin_error"
    assert detail["partial_state_save_failed"] is True
    assert detail.get("partial_state_save_error") == "OperationalError"


def test_assistant_raw_content_is_persisted_but_not_returned(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)

    session_resp = client.post("/api/sessions", json={"title": "Chat"})
    session_id = uuid.UUID(session_resp.json()["id"])

    changed_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="runtime preflight failed"),
        version=_EMPTY_STATE.version + 1,
    )
    composer_result = ComposerResult(
        message="I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
        state=changed_state,
        runtime_preflight=ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="bad config",
                    suggestion=None,
                    error_code=None,
                )
            ],
        ),
        raw_assistant_content="The pipeline is complete and valid.",
    )
    composer = SimpleNamespace()
    composer.compose = AsyncMock(spec=ComposerService.compose, return_value=composer_result)
    app.state.composer_service = composer

    resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "build it"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["content"].startswith("I cannot mark this pipeline complete")
    # POST /messages never populates raw_content. The field is part of the schema
    # (so eval tooling has a stable shape on the parallel GET endpoint), but is
    # null here — opt-in retrieval is GET-only via ?include_raw_content=true.
    assert body["message"]["raw_content"] is None

    loop = asyncio.new_event_loop()
    try:
        messages = loop.run_until_complete(service.get_messages(session_id, limit=None))
    finally:
        loop.close()
    assistant = next(message for message in messages if message.role == "assistant")
    assert assistant.raw_content == "The pipeline is complete and valid."


def test_non_augmentation_assistant_history_raises_audit_integrity_error() -> None:
    """Read-path defense: a persisted assistant row whose ``content`` does
    not start with ``raw_content`` violates the augmentation prefix
    invariant. All composer synthesis shapes are augmentations
    post-elspeth-9cfbad6901, so a row that breaks the contract is an
    audit-integrity violation. Crash on read rather than silently
    misroute synthesized operator-facing text into LLM history.

    Per project_db_migration_policy, the operator deletes the audit DB
    on schema-shape changes — so this case should not occur in
    production after the augmentation-only migration. The defensive
    check exists to surface a producer regression early rather than
    quietly serve corrupted history to the LLM.
    """
    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content="Synthetic content that does not start with raw_content.",
        raw_content="Model's actual prose.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        composition_state_id=None,
        writer_principal="compose_loop",
    )

    with pytest.raises(AuditIntegrityError) as exc_info:
        _composer_chat_history([message])

    detail = str(exc_info.value)
    assert "Tier 1" in detail
    assert "augmentation prefix invariant" in detail


def test_augmented_assistant_history_returns_unmodified_model_prose() -> None:
    """Augmentation path: the model's prose is preserved verbatim and an
    operator-facing suffix is appended. The LLM should see its own prose
    unmodified on subsequent turns — without this, the model cannot
    recover its own diagnostic context across turns.

    Discriminator is structural: ``content.startswith(raw_content)``
    detects synthesis and returns ``raw_content``. All composer
    synthesis shapes are augmentations post-elspeth-9cfbad6901; a row
    that breaks the contract raises AuditIntegrityError on read.
    """
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    model_prose = (
        "I tried to build the workflow but couldn't bind the CSV source — "
        "the schema requirement is not met. To move forward I need the file."
    )
    augmented_content = (
        model_prose + "\n\n---\n\n[ELSPETH-SYSTEM] The pipeline is still empty — "
        "the composer did not complete a valid build this turn.\n\n"
        "Cause: set_pipeline returned success=false: schema: Field required\n\n"
        "To continue: refine your request..."
    )
    message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content=augmented_content,
        raw_content=model_prose,
        tool_calls=None,
        created_at=datetime.now(UTC),
        composition_state_id=None,
        writer_principal="compose_loop",
    )

    history = _composer_chat_history([message])

    # Augmented turns: LLM sees its own prose verbatim, no system suffix —
    # the suffix is operator-facing only.
    assert history == [{"role": "assistant", "content": model_prose}]
    assert "[ELSPETH-SYSTEM]" not in history[0]["content"]


def test_augmented_assistant_history_handles_content_equal_to_raw_content() -> None:
    """Equality case: ``content == raw_content`` is augmentation with an empty
    suffix (or augmentation that no-ops). The LLM must see its own prose
    verbatim. Pre-fix a ``len(content) > len(raw_content)`` guard routed
    equal-length cases away from augmentation; the structural rule
    ``startswith`` handles the equality case correctly because every
    string startswith itself.
    """
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    model_prose = "Done — the pipeline validates and is ready to run."
    message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content=model_prose,
        raw_content=model_prose,
        tool_calls=None,
        created_at=datetime.now(UTC),
        composition_state_id=None,
        writer_principal="compose_loop",
    )

    history = _composer_chat_history([message])

    assert history == [{"role": "assistant", "content": model_prose}]


def test_augmented_assistant_history_treats_empty_raw_content_as_augmentation() -> None:
    """Empty-raw-content case for empty-state augmentation: when the model
    produces empty prose AND the empty-state synthesizer appends an
    operator-facing suffix, ``raw_content == ""`` and ``content == "<suffix>"``.
    The LLM must see an empty prior turn (its own actual output), not the
    operator-facing suffix.

    Pre-fix the discriminator's ``raw_content != ""`` guard short-circuited
    this case to ``return message.content`` — routing the suffix-only
    synthetic text into LLM history as if it were the model's own prior
    answer. The structural discriminator now treats empty-raw as
    augmentation (``"".startswith("")`` is always True) and returns the
    empty prose.

    Per elspeth-9cfbad6901, all composer synthesis shapes are
    augmentations including the non-empty-state preflight-invalid case
    that previously emitted a replacement; the empty-content + empty-raw
    + non-empty-state shape now degenerates cleanly to suffix-only
    output without the prior ambiguity.
    """
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    operator_facing_suffix = (
        "[ELSPETH-SYSTEM] The pipeline is still empty — the composer did "
        "not complete a valid build this turn.\n\n"
        "To continue: refine your request..."
    )
    message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content=operator_facing_suffix,
        raw_content="",
        tool_calls=None,
        created_at=datetime.now(UTC),
        composition_state_id=None,
        writer_principal="compose_loop",
    )

    history = _composer_chat_history([message])

    # LLM sees its empty prior turn, not the operator-facing suffix.
    # The suffix stays out of LLM context.
    assert history == [{"role": "assistant", "content": ""}]
    assert "[ELSPETH-SYSTEM]" not in history[0]["content"]


def test_composer_chat_history_skips_audit_tool_messages() -> None:
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    user_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="user",
        content="Build a CSV pipeline.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        writer_principal="route_user_message",
    )
    tool_audit_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="tool",
        content='{"success": true}',
        tool_calls=_audit_tool_calls("call-1"),
        created_at=datetime.now(UTC),
        writer_principal="compose_loop",
    )
    assistant_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content="I updated the pipeline.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        writer_principal="compose_loop",
    )

    history = _composer_chat_history([user_message, tool_audit_message, assistant_message])

    assert history == [
        {"role": "user", "content": "Build a CSV pipeline."},
        {"role": "assistant", "content": "I updated the pipeline."},
    ]


def test_composer_chat_history_skips_llm_call_audit_and_unknown_audit_kinds() -> None:
    from elspeth.web.sessions.routes import _composer_chat_history

    session_id = uuid.uuid4()
    user_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="user",
        content="Build a CSV pipeline.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        writer_principal="route_user_message",
    )
    llm_audit_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="tool",
        content="llm audit sidecar",
        tool_calls=_llm_call_audit_tool_calls(_llm_call(provider_request_id="chatcmpl-history")),
        created_at=datetime.now(UTC),
        writer_principal="compose_loop",
    )
    unknown_audit_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="tool",
        content="future audit sidecar",
        tool_calls=[{"_kind": "future_audit", "payload": {"id": "future"}}],
        created_at=datetime.now(UTC),
        writer_principal="compose_loop",
    )
    assistant_message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content="I updated the pipeline.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        writer_principal="compose_loop",
    )

    history = _composer_chat_history([user_message, llm_audit_message, unknown_audit_message, assistant_message])

    assert history == [
        {"role": "user", "content": "Build a CSV pipeline."},
        {"role": "assistant", "content": "I updated the pipeline."},
    ]


def test_send_message_does_not_replay_audit_tool_rows_to_composer(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    composer = _make_composer_mock(response_text="Continuing without audit rows.")
    app.state.composer_service = composer
    client = TestClient(app)

    session_resp = client.post("/api/sessions", json={"title": "Chat"})
    session_id = uuid.UUID(session_resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build it", writer_principal="route_user_message"))
        loop.run_until_complete(service.add_message(session_id, "assistant", "I started.", writer_principal="compose_loop"))
        # Rev-4: dispatch-trail audit envelopes without a real assistant
        # parent are persisted with ``role="audit"`` so the parent-CHECK
        # biconditional is satisfied. Pre-rev-4 used ``role="tool"``.
        loop.run_until_complete(
            service.add_message(
                session_id,
                "audit",
                '{"success": true}',
                tool_calls=_audit_tool_calls("call-1"),
                writer_principal="compose_loop",
            )
        )
    finally:
        loop.close()

    resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Continue"})

    assert resp.status_code == 200
    history = composer.compose.call_args.args[1]
    assert [entry["role"] for entry in history] == ["user", "assistant"]
    assert all(entry["role"] != "tool" for entry in history)
    assert all(entry["role"] != "audit" for entry in history)


def test_recompose_uses_last_conversational_user_before_audit_tool_rows(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    composer = _make_composer_mock(response_text="Retrying failed turn.")
    app.state.composer_service = composer
    client = TestClient(app, raise_server_exceptions=False)

    session_resp = client.post("/api/sessions", json={"title": "Retry"})
    session_id = uuid.UUID(session_resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        user_message = loop.run_until_complete(
            service.add_message(
                session_id,
                "user",
                "Build a CSV pipeline",
                writer_principal="route_user_message",
            )
        )
        # Rev-4 audit-only breadcrumb (no assistant parent).
        loop.run_until_complete(
            service.add_message(
                session_id,
                "audit",
                '{"success": true}',
                tool_calls=_audit_tool_calls("call-1"),
                writer_principal="compose_loop",
            )
        )
    finally:
        loop.close()

    resp = client.post(f"/api/sessions/{session_id}/recompose")

    assert resp.status_code == 200
    composer.compose.assert_awaited_once()
    assert composer.compose.call_args.args[0] == "Build a CSV pipeline"
    assert composer.compose.call_args.args[1] == []
    assert composer.compose.call_args.kwargs["user_message_id"] == str(user_message.id)


# ---------------------------------------------------------------------------
# elspeth-obs-f217c634aa: provenance discriminator at the three handler sites.
#
# Pre-fix: ``service.save_composition_state`` hardcoded
# ``provenance="session_seed"`` in the INSERT, so the three
# ``_handle_*`` partial-state captures all wrote ``session_seed`` into the
# audit DB — silently conflating four distinct event categories under one
# label and weakening the §4.1.2 audit-attribution contract.
#
# Post-fix: the public API takes ``provenance`` as a required keyword
# argument; the three handlers pass the discriminator value matching their
# error class. These tests assert the persisted ``provenance`` column
# carries the correct value end-to-end (handler → service →
# composition_states INSERT). The assertion is on the raw column rather
# than ``CompositionStateRecord`` because the field is not surfaced on
# the dataclass per Schedule 1A scope (DB-only audit column).
# ---------------------------------------------------------------------------


def _read_persisted_provenance(service: SessionServiceImpl, session_id: str) -> str:
    """Read the ``provenance`` column for the session's most-recent state row.

    The field is not exposed on ``CompositionStateRecord`` per spec
    §4.1.2 (DB-only audit column, Schedule 1A); a direct SELECT against
    the engine is the only way to verify the persisted label without
    breaking the deliberate read-side surface restriction.
    """
    from sqlalchemy import text

    with service._engine.begin() as conn:
        row = conn.execute(
            text("SELECT provenance FROM composition_states WHERE session_id = :sid ORDER BY version DESC LIMIT 1"),
            {"sid": session_id},
        ).fetchone()
    assert row is not None, f"no composition_states row for session {session_id}"
    return row[0]


def _read_persisted_state_identity(service: SessionServiceImpl, session_id: str) -> tuple[str, int]:
    """Read id/version for the session's most-recent state row."""

    from sqlalchemy import text

    with service._engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, version FROM composition_states WHERE session_id = :sid ORDER BY version DESC LIMIT 1"),
            {"sid": session_id},
        ).fetchone()
    assert row is not None, f"no composition_states row for session {session_id}"
    return str(row[0]), int(row[1])


def test_handle_convergence_error_persists_convergence_persist_provenance(tmp_path: Path) -> None:
    """``_handle_convergence_error`` must persist captured ``partial_state``
    with ``provenance='convergence_persist'`` so an auditor counting
    convergence-budget exhaustions gets the right answer.

    Pre-fix the row was written with ``provenance='session_seed'`` because
    ``save_composition_state`` hardcoded the label; this test pins the
    fix in place.
    """
    from elspeth.web.composer.protocol import ComposerConvergenceError

    partial = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="convergence-partial"),
        version=2,
    )
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerConvergenceError(
            max_turns=15,
            budget_exhausted="composition",
            partial_state=partial,
            tool_invocations=(),
            llm_calls=(),
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )
    assert response.status_code == 422

    assert _read_persisted_provenance(service, session_id) == "convergence_persist"


def test_handle_plugin_crash_persists_plugin_crash_persist_provenance(tmp_path: Path) -> None:
    """``_handle_plugin_crash`` must persist captured ``partial_state`` with
    ``provenance='plugin_crash_persist'`` — distinct from convergence and
    preflight partial-state captures so remediation triage can discriminate
    bug-fix-required from retry/budget-tunable.
    """
    partial = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="plugin-crash-partial"),
        version=3,
    )
    mock_composer = SimpleNamespace()
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerPluginCrashError(
            ValueError("plugin bug"),
            partial_state=partial,
        ),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )
    assert response.status_code == 500

    detail = response.json()["detail"]
    persisted_id, persisted_version = _read_persisted_state_identity(service, session_id)
    assert detail["partial_state"]["id"] == persisted_id
    assert detail["partial_state"]["version"] == persisted_version
    assert _read_persisted_provenance(service, session_id) == "plugin_crash_persist"


def test_handle_runtime_preflight_failure_persists_preflight_persist_provenance(tmp_path: Path) -> None:
    """``_handle_runtime_preflight_failure`` must persist captured
    ``partial_state`` with ``provenance='preflight_persist'`` so
    misconfiguration-class failures (preflight rejected the composed
    pipeline) are distinguishable from runtime execution failures in the
    audit DB.

    The preflight handler is reached when a successful compose result's
    state advance triggers a runtime preflight that crashes — see the
    ``preflight_exception_policy="raise"`` branch in routes.py and the
    matching ``ComposerRuntimePreflightError`` arm.
    """
    from elspeth.web.composer.protocol import ComposerRuntimePreflightError

    partial = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="preflight-partial"),
        version=4,
    )
    advanced_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="advanced"),
        version=2,
    )
    mock_composer = SimpleNamespace()
    # Compose succeeds (state advanced) but the post-compose
    # ``_state_data_from_composer_state`` call with
    # ``preflight_exception_policy="raise"`` would raise
    # ``ComposerRuntimePreflightError``; here we drive the same handler
    # via the cleaner top-level raise path that routes.py's path-1
    # ``_handle_runtime_preflight_failure`` invocation also reaches.
    mock_composer.compose = AsyncMock(
        spec=ComposerService.compose,
        side_effect=ComposerRuntimePreflightError(
            original_exc=ValueError("preflight rejected"),
            partial_state=partial,
            tool_invocations=(),
            llm_calls=(),
        ),
    )
    del advanced_state  # only declared to document the intended path-2 shape

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )
    assert response.status_code == 500

    detail = response.json()["detail"]
    persisted_id, persisted_version = _read_persisted_state_identity(service, session_id)
    assert detail["partial_state"]["id"] == persisted_id
    assert detail["partial_state"]["version"] == persisted_version
    assert _read_persisted_provenance(service, session_id) == "preflight_persist"


def test_send_message_post_compose_state_advance_persists_post_compose_provenance(tmp_path: Path) -> None:
    """A successful send-message state advance is not a session seed.

    The row is committed after the composer returns a newer state version, so
    auditors must be able to distinguish it from initial session creation and
    explicit state reselection.
    """
    advanced_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="post-compose"),
        version=2,
    )
    mock_composer = _make_composer_mock(response_text="Updated.", state=advanced_state)

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)

    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Build me a pipeline"},
    )

    assert response.status_code == 200
    assert _read_persisted_provenance(service, session_id) == "post_compose"


def test_send_message_state_advance_preserves_existing_composer_meta(tmp_path: Path) -> None:
    """Version-changing freeform messages retain opaque guided lifecycle metadata."""
    guided = GuidedSession.initial()
    advanced_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="post-compose"),
        version=2,
        guided_session=guided,
    )
    mock_composer = _make_composer_mock(response_text="Updated.", state=advanced_state)
    marker = {"composition_hash": "seed-hash"}

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    asyncio.run(
        service.save_composition_state(
            uuid.UUID(session_id),
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={"name": "before", "description": ""},
                is_valid=False,
                validation_errors=None,
                composer_meta={
                    "guided_session": guided.to_dict(),
                    "guided_completed_terminal_before_user_exit": marker,
                },
            ),
            provenance="session_seed",
        )
    )

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Update it"})

    assert response.status_code == 200
    current = asyncio.run(service.get_current_state(uuid.UUID(session_id)))
    assert current is not None
    assert current.composer_meta["guided_completed_terminal_before_user_exit"] == marker


def test_recompose_post_compose_state_advance_persists_post_compose_provenance(tmp_path: Path) -> None:
    """The recompose mirror path uses the same post-compose provenance."""
    advanced_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="post-compose"),
        version=2,
    )
    mock_composer = _make_composer_mock(response_text="Updated.", state=advanced_state)

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)

    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            service.add_message(
                uuid.UUID(session_id),
                "user",
                "Build me a pipeline",
                writer_principal="route_user_message",
            )
        )
    finally:
        loop.close()

    response = client.post(f"/api/sessions/{session_id}/recompose")

    assert response.status_code == 200
    assert _read_persisted_provenance(service, session_id) == "post_compose"


def test_recompose_state_advance_preserves_existing_composer_meta(tmp_path: Path) -> None:
    """Version-changing recompose retains opaque guided lifecycle metadata."""
    guided = GuidedSession.initial()
    advanced_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="post-compose"),
        version=2,
        guided_session=guided,
    )
    mock_composer = _make_composer_mock(response_text="Updated.", state=advanced_state)
    marker = {"composition_hash": "seed-hash"}

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session_id = client.post("/api/sessions", json={"title": "T"}).json()["id"]
    asyncio.run(
        service.save_composition_state(
            uuid.UUID(session_id),
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={"name": "before", "description": ""},
                is_valid=False,
                validation_errors=None,
                composer_meta={
                    "guided_session": guided.to_dict(),
                    "guided_completed_terminal_before_user_exit": marker,
                },
            ),
            provenance="session_seed",
        )
    )
    asyncio.run(
        service.add_message(
            uuid.UUID(session_id),
            "user",
            "Update it",
            writer_principal="route_user_message",
        )
    )

    response = client.post(f"/api/sessions/{session_id}/recompose")

    assert response.status_code == 200
    current = asyncio.run(service.get_current_state(uuid.UUID(session_id)))
    assert current is not None
    assert current.composer_meta["guided_completed_terminal_before_user_exit"] == marker


def test_composition_state_provenance_python_and_sql_enums_agree() -> None:
    """The ``ck_composition_states_provenance`` CHECK constraint and the
    :data:`CompositionStateProvenance` Literal are paired contracts:
    extending one without the other lets the Python writer pass while
    the DB rejects the row (or vice versa). This test pins them equal
    by parsing the CHECK SQL and comparing against the Literal's
    ``frozenset``.

    The save-time enforcement of unknown values lives at the DB layer
    (the CHECK fires on INSERT). Exercising that path through
    ``save_composition_state`` is awkward because the production
    INSERT path retries on ``IntegrityError`` (B3 belt-and-suspenders
    in service.py); the symmetry assertion here is the cleaner
    contract pin-down.
    """
    import re

    from elspeth.web.sessions.models import composition_states_table
    from elspeth.web.sessions.protocol import COMPOSITION_STATE_PROVENANCE_VALUES

    check = next(c for c in composition_states_table.constraints if getattr(c, "name", None) == "ck_composition_states_provenance")
    sql_text = str(check.sqltext)  # type: ignore[attr-defined]
    sql_values = frozenset(re.findall(r"'([a-z_]+)'", sql_text))
    assert sql_values == COMPOSITION_STATE_PROVENANCE_VALUES, (
        f"CHECK enum {sorted(sql_values)} drifted from CompositionStateProvenance Literal {sorted(COMPOSITION_STATE_PROVENANCE_VALUES)}"
    )
