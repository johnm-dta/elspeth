"""Concurrency proofs for atomic composer transform splices."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Engine
from testcontainers.postgres import PostgresContainer
from tests.integration.web.conftest import _make_session
from tests.unit.web.sessions.test_routes import _make_app

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.protocol import ComposerResult
from elspeth.web.composer.state import CompositionState, EdgeSpec, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.composer.tools import execute_tool
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions._persist_payload import RedactedToolRow, StatePayload
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl, StaleComposeStateError
from elspeth.web.sessions.telemetry import build_sessions_telemetry

pytestmark = pytest.mark.testcontainer


def _node(node_id: str, *, input_name: str, output_name: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="passthrough",
        input=input_name,
        on_success=output_name,
        on_error="discard",
        options={"schema": {"mode": "observed"}},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _initial_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": "rows.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(
            _node("before", input_name="rows", output_name="middle"),
            _node("after", input_name="middle", output_name="result"),
        ),
        edges=(
            EdgeSpec(id="source-before", from_node="source", to_node="before", edge_type="on_success", label=None),
            EdgeSpec(id="before-after", from_node="before", to_node="after", edge_type="on_success", label=None),
            EdgeSpec(id="after-output", from_node="after", to_node="result", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="result",
                plugin="json",
                options={
                    "path": "out.jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="splice concurrency"),
        version=1,
    )


def _splice_arguments() -> dict[str, object]:
    return {
        "predecessor_id": "before",
        "successor_id": "after",
        "node": {
            "id": "inserted",
            "plugin": "passthrough",
            "options": {"schema": {"mode": "observed"}},
            "on_error": "discard",
        },
    }


def _policy_context() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot), snapshot


def _state_data(state: CompositionState) -> CompositionStateData:
    payload = state.to_dict()
    return CompositionStateData(
        sources=payload["sources"],
        nodes=payload["nodes"],
        edges=payload["edges"],
        outputs=payload["outputs"],
        metadata_=payload["metadata"],
        is_valid=state.validate().is_valid,
    )


class _BlockingSpliceComposer:
    """Execute the real splice handler while exposing route-lock ordering."""

    def __init__(self) -> None:
        self._catalog, self._snapshot = _policy_context()
        self.input_versions: list[int] = []
        self.already_applied: list[bool] = []
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
        progress: Any = None,
        guided_terminal: Any = None,
        user_message_id: str | None = None,
    ) -> ComposerResult:
        del message, chat_messages, session_id, current_state_id, user_id, progress, guided_terminal, user_message_id
        result = execute_tool(
            "splice_transform",
            _splice_arguments(),
            state,
            self._catalog,
            plugin_snapshot=self._snapshot,
        )
        assert result.success, result.data
        self.input_versions.append(state.version)
        replayed = bool(result.data["already_applied"])
        self.already_applied.append(replayed)
        if len(self.input_versions) == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
        else:
            self.second_call_started.set()
        return ComposerResult(
            message=f"already_applied={str(replayed).lower()}",
            state=result.updated_state,
        )


@pytest.mark.asyncio
async def test_concurrent_http_splices_serialize_reload_and_apply_once(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path)
    composer = _BlockingSpliceComposer()
    app.state.composer_service = composer

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post(
            "/api/sessions",
            json={"title": "Splice concurrency proof"},
        )
        assert created.status_code == 201
        session_id = uuid.UUID(created.json()["id"])
        initial_record = await service.save_composition_state(
            session_id,
            _state_data(_initial_state()),
            provenance="session_seed",
        )

        async def send(content: str):
            return await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": content},
            )

        first_task = asyncio.create_task(send("Insert the transform"))
        await asyncio.wait_for(composer.first_call_started.wait(), timeout=2.0)
        second_task = asyncio.create_task(send("Insert the same transform"))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(composer.second_call_started.wait(), timeout=0.3)

        composer.release_first_call.set()
        first_response, second_response = await asyncio.gather(first_task, second_task)

    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text
    assert composer.input_versions == [initial_record.version, initial_record.version + 1]
    assert composer.already_applied == [False, True]
    assert first_response.json()["message"]["content"] == "already_applied=false"
    assert second_response.json()["message"]["content"] == "already_applied=true"
    assert first_response.json()["state"]["version"] == initial_record.version + 1
    assert second_response.json()["state"] is None

    current = await service.get_current_state(session_id)
    versions = await service.get_state_versions(session_id)
    assert current is not None
    assert current.version == initial_record.version + 1
    assert len(versions) == 2
    assert [node.id for node in state_from_record(current).nodes] == ["before", "inserted", "after"]


@pytest.fixture(scope="module")
def postgres_engine() -> Iterator[Engine]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        engine = create_session_engine(postgres.get_connection_url())
        initialize_session_schema(engine)
        try:
            yield engine
        finally:
            engine.dispose()


@pytest.fixture
def postgres_service(postgres_engine: Engine, tmp_path: Path) -> SessionServiceImpl:
    return SessionServiceImpl(
        postgres_engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.composer.splice.concurrency"),
    )


def test_concurrent_postgres_splice_persistence_rejects_stale_writer_without_lost_update(
    postgres_service: SessionServiceImpl,
) -> None:
    session_id = uuid.uuid4()
    with postgres_service._engine.begin() as connection:
        _make_session(
            connection,
            session_id=str(session_id),
            title="PostgreSQL splice concurrency proof",
        )

    initial_state = _initial_state()
    initial_record = asyncio.run(
        postgres_service.save_composition_state(
            session_id,
            _state_data(initial_state),
            provenance="session_seed",
        )
    )
    catalog, snapshot = _policy_context()
    splice = execute_tool(
        "splice_transform",
        _splice_arguments(),
        initial_state,
        catalog,
        plugin_snapshot=snapshot,
    )
    assert splice.success, splice.data
    winning_data = _state_data(splice.updated_state)

    barrier = threading.Barrier(2)
    successes: list[str | None] = []
    stale_rejections: list[StaleComposeStateError] = []
    unexpected: list[BaseException] = []

    def persist(writer: str) -> None:
        tool_call_id = f"splice-{writer}"
        try:
            barrier.wait(timeout=10)
            outcome = postgres_service.persist_compose_turn(
                session_id=str(session_id),
                assistant_content="Splice complete.",
                redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "splice_transform"}},),
                redacted_tool_rows=(
                    RedactedToolRow(
                        tool_call_id,
                        '{"success":true}',
                        StatePayload(
                            data=winning_data,
                            derived_from_state_id=str(initial_record.id),
                        ),
                    ),
                ),
                parent_composition_state_id=str(initial_record.id),
                expected_current_state_id=str(initial_record.id),
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
            successes.append(outcome.current_state_id)
        except StaleComposeStateError as exc:
            stale_rejections.append(exc)
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            unexpected.append(exc)

    workers = [
        threading.Thread(target=persist, args=("a",), name="splice-writer-a"),
        threading.Thread(target=persist, args=("b",), name="splice-writer-b"),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)

    assert all(not worker.is_alive() for worker in workers)
    assert unexpected == []
    assert len(successes) == 1
    assert len(stale_rejections) == 1

    current = asyncio.run(postgres_service.get_current_state(session_id))
    versions = asyncio.run(postgres_service.get_state_versions(session_id))
    assert current is not None
    assert current.id == uuid.UUID(successes[0])
    assert current.version == initial_record.version + 1
    assert len(versions) == 2
    assert state_from_record(current).to_dict() == splice.updated_state.to_dict()
