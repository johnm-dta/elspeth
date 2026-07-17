"""Freeform empty-pipeline requests use the canonical proposal lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.pipeline_planner import PipelinePlannerError
from elspeth.web.composer.pipeline_proposal import composition_content_hash
from elspeth.web.composer.protocol import ComposerResult
from elspeth.web.composer.recipe_intent_routing import FreeformRecipeIntentMatch, InlineRecipeBlob
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    proposal_events_table,
)
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Function


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall]


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: Mapping[str, object]
    model: str = "provider/planner-v1"
    id: str = "planner-request-1"


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _pipeline(data_dir: Path) -> dict[str, Any]:
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(data_dir / "blobs" / "input.csv"), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "rows",
                "plugin": "json",
                "options": {
                    "path": str(data_dir / "outputs" / "result.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
    }


def _terminal_response(data_dir: Path) -> _Response:
    return _Response(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="freeform-terminal",
                            function=_Function(
                                name="emit_pipeline_proposal",
                                arguments=json.dumps({"pipeline": _pipeline(data_dir)}),
                            ),
                        )
                    ],
                )
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("trust_mode", ["auto_commit", "explicit_approve"])
@pytest.mark.parametrize("persisted_base", [False, True], ids=["absent-base", "present-base"])
async def test_empty_build_stages_one_canonical_pipeline_proposal_for_both_trust_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trust_mode: str,
    persisted_base: bool,
) -> None:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    sessions = SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))
    session = await sessions.create_session("planner-user", "Planner", "local")
    await sessions.update_composer_preferences(
        session.id,
        trust_mode=trust_mode,  # type: ignore[arg-type]
        density_default="high",
        actor="test",
    )
    user_message = await sessions.add_message(
        session.id,
        "user",
        "Build a CSV to JSONL pipeline.",
        writer_principal="route_user_message",
    )
    current_state = None
    if persisted_base:
        current_state = await sessions.save_composition_state(
            session.id,
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={"name": "Untitled Pipeline", "description": ""},
                is_valid=False,
            ),
            provenance="session_seed",
        )
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="test/planner",
        composer_boot_probe_enabled=False,
        composer_max_composition_turns=3,
        composer_max_discovery_turns=2,
        composer_timeout_seconds=20.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(available=True, provider="test", model="test/planner", reason=None),
    )
    composer = ComposerServiceImpl.for_trained_operator(
        create_catalog_service(),
        settings,
        sessions_service=sessions,
        session_engine=engine,
    )
    requests: list[dict[str, Any]] = []

    async def completion(**kwargs: Any) -> _Response:
        requests.append(kwargs)
        return _terminal_response(tmp_path)

    monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)

    result = await composer.compose(
        "Build a CSV to JSONL pipeline.",
        [],
        _empty_state(),
        session_id=str(session.id),
        current_state_id=str(current_state.id) if current_state is not None else None,
        user_id="planner-user",
        user_message_id=str(user_message.id),
    )

    proposals = await sessions.list_composition_proposals(session.id, status="pending")
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "set_pipeline"
    assert deep_thaw(proposal.arguments_json) == _pipeline(tmp_path)
    assert proposal.pipeline_metadata is not None
    assert proposal.pipeline_metadata.base["kind"] == ("present" if persisted_base else "absent")
    assert proposal.base_state_id == (current_state.id if current_state is not None else None)
    if current_state is not None:
        assert proposal.pipeline_metadata.base == {
            "kind": "present",
            "state_id": str(current_state.id),
            "composition_content_hash": composition_content_hash(_empty_state()),
        }
    assert result.state == _empty_state()
    if trust_mode == "auto_commit":
        assert result.pipeline_commit_intent is not None
        assert result.pipeline_commit_intent.proposal_id == proposal.id
        assert result.pipeline_commit_intent.draft_hash == proposal.pipeline_metadata.draft_hash  # type: ignore[union-attr]
    else:
        assert result.pipeline_commit_intent is None

    with engine.connect() as conn:
        audit_rows = conn.execute(select(chat_messages_table.c.role, chat_messages_table.c.tool_calls)).all()
    assert any(role == "audit" and calls and calls[0].get("_kind") == "llm_call_audit" for role, calls in audit_rows)
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == settings.composer_planner_max_completion_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "state"),
    [
        ("What can you help me build?", _empty_state()),
        (
            "Add a JSONL output to this pipeline.",
            CompositionState(
                source=SourceSpec(
                    plugin="csv",
                    on_success="rows",
                    options={"path": "/tmp/input.csv", "schema": {"mode": "observed"}},
                    on_validation_failure="discard",
                ),
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(),
                version=1,
            ),
        ),
    ],
    ids=["empty-informational", "nonempty-incremental"],
)
async def test_requests_outside_empty_mutation_gate_use_ordinary_compose_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    state: CompositionState,
) -> None:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    sessions = SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))
    session = await sessions.create_session("planner-user", "Planner", "local")
    user_message = await sessions.add_message(
        session.id,
        "user",
        message,
        writer_principal="route_user_message",
    )
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="test/planner",
        composer_boot_probe_enabled=False,
        composer_max_composition_turns=3,
        composer_max_discovery_turns=2,
        composer_timeout_seconds=20.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(available=True, provider="test", model="test/planner", reason=None),
    )
    composer = ComposerServiceImpl.for_trained_operator(
        create_catalog_service(),
        settings,
        sessions_service=sessions,
        session_engine=engine,
    )
    expected = ComposerResult(message="ordinary loop", state=state)
    ordinary_loop = AsyncMock(return_value=expected)
    monkeypatch.setattr(composer, "_compose_loop", ordinary_loop)

    result = await composer.compose(
        message,
        [],
        state,
        session_id=str(session.id),
        user_id="planner-user",
        user_message_id=str(user_message.id),
    )

    assert result is expected
    ordinary_loop.assert_awaited_once()
    assert await sessions.list_composition_proposals(session.id) == []


@pytest.mark.asyncio
async def test_planner_audit_failure_publishes_no_proposal_authority_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    sessions = SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))
    session = await sessions.create_session("planner-user", "Planner", "local")
    user_message = await sessions.add_message(
        session.id,
        "user",
        "Build a CSV to JSONL pipeline.",
        writer_principal="route_user_message",
    )
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="test/planner",
        composer_boot_probe_enabled=False,
        composer_max_composition_turns=3,
        composer_max_discovery_turns=2,
        composer_timeout_seconds=20.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(available=True, provider="test", model="test/planner", reason=None),
    )
    composer = ComposerServiceImpl.for_trained_operator(
        create_catalog_service(),
        settings,
        sessions_service=sessions,
        session_engine=engine,
    )

    async def completion(**_kwargs: Any) -> _Response:
        return _terminal_response(tmp_path)

    monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)
    monkeypatch.setattr(sessions, "add_message", AsyncMock(side_effect=SQLAlchemyError("audit write failed")))

    with pytest.raises(AuditIntegrityError, match="audit persistence failed before proposal creation"):
        await composer.compose(
            "Build a CSV to JSONL pipeline.",
            [],
            _empty_state(),
            session_id=str(session.id),
            user_id="planner-user",
            user_message_id=str(user_message.id),
        )

    assert await sessions.list_composition_proposals(session.id) == []
    with engine.connect() as conn:
        assert conn.execute(select(composition_proposals_table)).all() == []
        assert conn.execute(select(proposal_events_table)).all() == []
        assert conn.execute(select(composition_states_table)).all() == []


async def _recipe_composer_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    message: str,
) -> tuple[Any, SessionServiceImpl, Any, Any, ComposerServiceImpl]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    sessions = SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))
    session = await sessions.create_session("planner-user", "Planner", "local")
    user_message = await sessions.add_message(
        session.id,
        "user",
        message,
        writer_principal="route_user_message",
    )
    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="test/planner",
        composer_boot_probe_enabled=False,
        composer_max_composition_turns=3,
        composer_max_discovery_turns=2,
        composer_timeout_seconds=20.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(available=True, provider="test", model="test/planner", reason=None),
    )
    composer = ComposerServiceImpl.for_trained_operator(
        create_catalog_service(),
        settings,
        sessions_service=sessions,
        session_engine=engine,
    )
    return engine, sessions, session, user_message, composer


def _assert_no_pipeline_side_effects(engine: Any) -> None:
    with engine.connect() as conn:
        assert conn.execute(select(blobs_table)).all() == []
        assert conn.execute(select(composition_proposals_table)).all() == []
        assert conn.execute(select(proposal_events_table)).all() == []
        assert conn.execute(select(composition_states_table)).all() == []


@pytest.mark.asyncio
async def test_invalid_server_recipe_falls_back_before_custody_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "Build the requested pipeline."
    engine, _sessions, session, user_message, composer = await _recipe_composer_context(
        tmp_path,
        monkeypatch,
        message=message,
    )
    invalid_match = FreeformRecipeIntentMatch(
        recipe_name="fork-coalesce-truncate-jsonl",
        inline_blob=InlineRecipeBlob(filename="rows.csv", mime_type="text/csv", content="name,description\na,hello"),
        slots={"output_path": "outputs/result.jsonl"},
    )
    monkeypatch.setattr("elspeth.web.composer.service.match_freeform_recipe_intent", lambda _message: invalid_match)
    prepare = AsyncMock(side_effect=AssertionError("invalid recipe must not reach custody preparation"))
    monkeypatch.setattr("elspeth.web.composer.service.prepare_pipeline_plan", prepare)
    fallback = AsyncMock(side_effect=PipelinePlannerError("fallback stopped", code="TEST_STOP"))
    monkeypatch.setattr("elspeth.web.composer.service.plan_pipeline", fallback)

    with pytest.raises(PipelinePlannerError, match="fallback stopped"):
        await composer.compose(
            message,
            [],
            _empty_state(),
            session_id=str(session.id),
            user_id="planner-user",
            user_message_id=str(user_message.id),
        )

    prepare.assert_not_awaited()
    fallback.assert_awaited_once()
    _assert_no_pipeline_side_effects(engine)


@pytest.mark.asyncio
async def test_recipe_custody_failure_cannot_publish_reviewable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "Build the requested pipeline."
    engine, _sessions, session, user_message, composer = await _recipe_composer_context(
        tmp_path,
        monkeypatch,
        message=message,
    )
    valid_match = FreeformRecipeIntentMatch(
        recipe_name="fork-coalesce-truncate-jsonl",
        inline_blob=InlineRecipeBlob(filename="rows.csv", mime_type="text/csv", content="name,description\na,hello"),
        slots={
            "truncate_field": "description",
            "max_chars": 30,
            "truncation_suffix": "...",
            "output_path": "outputs/result.jsonl",
            "key_a": "path_a",
            "key_b": "path_b",
        },
    )
    monkeypatch.setattr("elspeth.web.composer.service.match_freeform_recipe_intent", lambda _message: valid_match)
    monkeypatch.setattr(
        "elspeth.web.composer.pipeline_planner.finalize_pipeline_custody",
        AsyncMock(side_effect=AuditIntegrityError("custody failed")),
    )

    with pytest.raises(AuditIntegrityError, match="custody failed"):
        await composer.compose(
            message,
            [],
            _empty_state(),
            session_id=str(session.id),
            user_id="planner-user",
            user_message_id=str(user_message.id),
        )

    _assert_no_pipeline_side_effects(engine)
