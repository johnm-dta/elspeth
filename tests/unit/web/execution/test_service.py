"""Tests for ExecutionServiceImpl — background execution with thread safety.

Each test class targets a specific review fix:
- TestExecutionFlow: Basic lifecycle (pending -> running -> completed)
- TestB2ShutdownEvent: shutdown_event always passed to Orchestrator.run()
- TestB3Construction: LandscapeDB/PayloadStore from WebSettings
- TestB7ExceptionHandling: BaseException catch + done_callback safety net
- TestB8AsyncBridging: _call_async() bridges sync thread to async event loop
- TestCancelMechanism: Event-based cancellation
- TestOneActiveRun: B6 constraint enforcement
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import threading
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    RateLimitSettings,
    TelemetrySettings,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import run_attributions_table, runs_table
from elspeth.web.blobs.protocol import BlobFinalizationResult
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
)
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, PROMPT_TEMPLATE_PARTS_KEY
from elspeth.web.sessions.protocol import (
    LEGAL_RUN_TRANSITIONS,
    CompositionStateRecord,
    IllegalRunTransitionError,
    RunAlreadyActiveError,
    SessionRunStatus,
)
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value

# ── Fixtures ───────────────────────────────────────────────────────────

_TEST_PIPELINE_YAML = "source:\n  plugin: csv\n  options: {}\n"


@pytest.fixture
def mock_pipeline_config_assembly() -> Iterator[MagicMock]:
    """Opt-in patch for ``assemble_and_validate_pipeline_config``.

    Service tests that exercise ``_run_pipeline`` and mock the plugin bundle
    and graph would otherwise have the helper's route-target validators fire
    against MagicMock attributes and raise ``RouteValidationError``
    spuriously. Tests using real settings + real plugin instantiation must
    NOT use this fixture — the real helper is required for those flows.
    Issue elspeth-127de6865a introduced the helper.
    """
    with patch(
        "elspeth.web.execution.service.assemble_and_validate_pipeline_config",
        return_value=MagicMock(),
    ) as mock_assemble:
        yield mock_assemble


@pytest.fixture
def mock_loop() -> MagicMock:
    return MagicMock(spec=asyncio.AbstractEventLoop)


@pytest.fixture
def broadcaster(mock_loop: MagicMock) -> ProgressBroadcaster:
    return ProgressBroadcaster(mock_loop)


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.get_landscape_url.return_value = "sqlite:///test_audit.db"
    settings.get_payload_store_path.return_value = Path("/tmp/test_payloads")
    settings.landscape_passphrase = None
    # data_dir is consumed by the source/sink path allowlist and the
    # blob source-path read guard.  Pin it to a known string so tests
    # that exercise blob-backed sources can compute matching canonical
    # paths (elspeth-07089fbaa3).
    settings.data_dir = "/tmp/data"
    return settings


def _mock_pipeline_settings() -> MagicMock:
    """Return settings-shaped test data for patched pipeline loading.

    _run_pipeline() now builds the same runtime infrastructure as the CLI path,
    so tests that patch YAML loading must still provide real config-contract
    objects for the runtime conversion boundary.
    """
    settings = MagicMock()
    settings.gates = []
    settings.coalesce = []
    settings.rate_limit = RateLimitSettings(enabled=False)
    settings.concurrency = ConcurrencySettings()
    settings.checkpoint = CheckpointSettings(enabled=False)
    settings.telemetry = TelemetrySettings(enabled=False)
    return settings


def _run_accounting_for_status(status: RunStatus) -> RunAccounting:
    if status == RunStatus.EMPTY:
        return RunAccounting(
            source=RunAccountingSource(rows_processed=0),
            tokens=RunAccountingTokens(emitted=0, terminal=0, succeeded=0, failed=0, structural=0, pending=0),
            routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
            integrity=RunAccountingIntegrity(closure="closed", missing_terminal_outcomes=0, duplicate_terminal_outcomes=0),
        )
    if status == RunStatus.COMPLETED_WITH_FAILURES:
        return RunAccounting(
            source=RunAccountingSource(rows_processed=10),
            tokens=RunAccountingTokens(emitted=10, terminal=10, succeeded=8, failed=2, structural=0, pending=0),
            routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
            integrity=RunAccountingIntegrity(closure="closed", missing_terminal_outcomes=0, duplicate_terminal_outcomes=0),
        )
    return RunAccounting(
        source=RunAccountingSource(rows_processed=10),
        tokens=RunAccountingTokens(emitted=10, terminal=10, succeeded=10, failed=0, structural=0, pending=0),
        routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
        integrity=RunAccountingIntegrity(closure="closed", missing_terminal_outcomes=0, duplicate_terminal_outcomes=0),
    )


def _with_resolved_model_choice(node: dict[str, Any]) -> dict[str, Any]:
    """Pre-stage a resolved ``llm_model_choice`` interpretation requirement.

    Tests that construct an LLM node by raw dict bypass the composer's
    mutation-time auto-stager (which would create a pending
    requirement). Without this resolution, the validator's interpretation
    gate short-circuits before any downstream check runs. Tests
    exercising downstream behavior (fanout guard, blob-inline,
    placeholder gate, etc.) get the gate resolved here so the test stays
    focused on its actual subject.
    """
    if node.get("plugin") != "llm":
        return node
    options = node.get("options")
    if not isinstance(options, dict):
        return node
    model = options.get("model")
    if not isinstance(model, str) or not model:
        return node
    requirements = list(options.get(INTERPRETATION_REQUIREMENTS_KEY) or ())
    requirements.append(
        {
            "id": f"model_choice_review:{node['id']}",
            "kind": "llm_model_choice",
            "user_term": f"llm_model_choice:{node['id']}",
            "status": "resolved",
            "draft": model,
            "event_id": f"model-choice-accepted:{node['id']}",
            "accepted_value": model,
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": stable_hash(model),
        }
    )
    return {
        **node,
        "options": {**options, INTERPRETATION_REQUIREMENTS_KEY: requirements},
    }


def _composition_state_record(
    *,
    session_id: UUID,
    source_path: Path,
    output_path: Path,
    nodes: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]] | None = None,
) -> CompositionStateRecord:
    source = {
        "plugin": "text",
        "on_success": "source_rows",
        "on_validation_failure": "discard",
        "options": {
            "path": str(source_path),
            "column": "body",
            "schema": {"mode": "observed"},
        },
    }
    return CompositionStateRecord(
        id=uuid4(),
        session_id=session_id,
        version=1,
        source=source if sources is None else next(iter(sources.values())),
        sources=sources,
        nodes=[_with_resolved_model_choice(node) for node in nodes],
        edges=[],
        outputs=[
            {
                "name": "out",
                "plugin": "json",
                "options": {
                    "path": str(output_path),
                    "format": "jsonl",
                    "mode": "write",
                    "schema": {"mode": "observed"},
                },
                "on_write_failure": "discard",
            }
        ],
        metadata_={"name": "Test", "description": ""},
        is_valid=True,
        validation_errors=None,
        created_at=datetime.now(UTC),
        derived_from_state_id=None,
    )


@pytest.fixture
def mock_session_service() -> MagicMock:
    svc = MagicMock()
    state = MagicMock()
    state.yaml_content = "source:\n  plugin: csv_source"
    # SessionService methods are async — use AsyncMock for awaitable returns
    # state_record needs fields that state_from_record() accesses
    state.id = uuid4()
    state.session_id = uuid4()
    state.version = 1
    state.source = None  # No source → path allowlist check skips
    state.sources = None
    state.nodes = None
    state.edges = None
    state.outputs = None
    state.metadata_ = {"name": "Test", "description": ""}
    svc.get_state = AsyncMock(return_value=state)
    svc.get_current_state = AsyncMock(return_value=state)
    svc.get_active_run = AsyncMock(return_value=None)
    svc.create_run = AsyncMock(return_value=MagicMock(id=uuid4()))
    svc.get_run = AsyncMock(return_value=MagicMock(status="pending"))
    svc.update_run_status = AsyncMock()
    return svc


@pytest.fixture
def service(
    mock_loop: MagicMock,
    broadcaster: ProgressBroadcaster,
    mock_settings: MagicMock,
    mock_session_service: MagicMock,
) -> Iterator[ExecutionServiceImpl]:
    # AC #17: All Run CRUD goes through SessionService — no direct DB access.
    mock_yaml_generator = MagicMock()
    mock_yaml_generator.generate_yaml.return_value = _TEST_PIPELINE_YAML
    svc = ExecutionServiceImpl(
        loop=mock_loop,
        broadcaster=broadcaster,
        settings=mock_settings,
        session_service=mock_session_service,
        yaml_generator=mock_yaml_generator,
        telemetry=build_sessions_telemetry(),
    )
    # Patch _call_async for tests that call _run_pipeline directly (sync).
    # The real _call_async uses asyncio.run_coroutine_threadsafe which needs
    # a running event loop. In unit tests, we bridge by running the coroutine
    # synchronously via asyncio.get_event_loop().run_until_complete().
    # TestB8AsyncBridging tests _call_async itself with its own mocking.
    _real_loop = asyncio.new_event_loop()

    def _mock_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            return _real_loop.run_until_complete(coro)
        except RuntimeError:
            # If no event loop is available, just close the coroutine
            coro.close()
            return None

    cast(Any, svc)._call_async = _mock_call_async
    yield svc
    _real_loop.close()


# ── Basic Lifecycle ────────────────────────────────────────────────────


class TestExecutionFlow:
    @pytest.mark.asyncio
    async def test_execute_returns_run_id_immediately(self, service: ExecutionServiceImpl) -> None:
        """execute() returns a UUID without blocking on pipeline completion."""
        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_execute_rejects_non_string_yaml_generator_output(self, service: ExecutionServiceImpl) -> None:
        """YamlGenerator contract violations must fail fast, not spin in PyYAML."""
        # _yaml_generator is a MagicMock in the fixture (see service fixture
        # above); the production type is Callable[[CompositionState], str]
        # which has no .return_value attribute.  Cast for mypy.
        cast(MagicMock, service._yaml_generator).generate_yaml.return_value = MagicMock()

        with pytest.raises(TypeError, match="must return str"):
            await service.execute(session_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_creates_run_via_session_service(self, service: ExecutionServiceImpl, mock_session_service: MagicMock) -> None:
        """AC #17: Run creation delegates to session_service.create_run()
        with R6 expanded params (session_id, state_id, pipeline_yaml)."""
        with patch.object(service, "_run_pipeline"):
            await service.execute(session_id=uuid4())
        mock_session_service.create_run.assert_called_once()
        create_call = mock_session_service.create_run.call_args
        assert "session_id" in create_call[1] or len(create_call[0]) >= 1
        assert "pipeline_yaml" in create_call[1] or len(create_call[0]) >= 2

    @pytest.mark.asyncio
    async def test_get_status_returns_run_status(self, service: ExecutionServiceImpl, mock_session_service: MagicMock) -> None:
        run_id = uuid4()
        mock_session_service.get_run.return_value = MagicMock(
            id=run_id,  # B7: RunRecord uses `id`, not `run_id`
            status="running",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            rows_processed=50,
            rows_succeeded=48,
            rows_failed=2,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id=None,
        )
        status = await service.get_status(run_id)
        assert status.status == "running"
        assert status.accounting is None


class TestExecutionFanoutGuard:
    @pytest.mark.asyncio
    async def test_line_explode_to_llm_requires_ack_before_run_creation(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A deaggregation transform upstream of LLM must stop at launch."""
        from elspeth.web.execution.fanout_guard import ExecutionFanoutGuardRequired

        data_dir = tmp_path
        blob_dir = data_dir / "blobs"
        output_dir = data_dir / "outputs"
        blob_dir.mkdir()
        output_dir.mkdir()
        source_path = blob_dir / "input.txt"
        source_path.write_text("alpha\nbeta\n", encoding="utf-8")
        session_id = uuid4()
        mock_settings.data_dir = data_dir
        mock_session_service.get_current_state.return_value = _composition_state_record(
            session_id=session_id,
            source_path=source_path,
            output_path=output_dir / "out.jsonl",
            nodes=[
                {
                    "id": "explode_lines",
                    "node_type": "transform",
                    "plugin": "line_explode",
                    "input": "source_rows",
                    "on_success": "line_rows",
                    "on_error": "errors",
                    "options": {"source_field": "body", "schema": {"mode": "observed"}},
                },
                {
                    "id": "classify_line",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "line_rows",
                    "on_success": "out",
                    "on_error": "errors",
                    "options": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    },
                },
            ],
        )

        with (
            patch.object(service, "_run_pipeline"),
            patch("elspeth.web.execution.service.validate_semantic_contracts", return_value=((), ())),
            pytest.raises(ExecutionFanoutGuardRequired) as raised,
        ):
            await service.execute(session_id=session_id)

        guard = raised.value.guard
        assert guard.ack_token
        assert guard.risks[0].node_id == "classify_line"
        assert guard.risks[0].provider == "openrouter"
        assert guard.risks[0].model == "openai/gpt-4o-mini"
        assert guard.risks[0].credential_ref == "secret_ref:OPENROUTER_API_KEY"
        assert guard.risks[0].estimated_provider_calls is None
        assert mock_session_service.create_run.await_count == 0

    @pytest.mark.asyncio
    async def test_acknowledged_line_explode_to_llm_records_guard_in_run_yaml(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Accepted fanout warnings are persisted with the run launch record."""
        from elspeth.web.execution.fanout_guard import ExecutionFanoutGuardRequired

        data_dir = tmp_path
        blob_dir = data_dir / "blobs"
        output_dir = data_dir / "outputs"
        blob_dir.mkdir()
        output_dir.mkdir()
        source_path = blob_dir / "input.txt"
        source_path.write_text("alpha\nbeta\n", encoding="utf-8")
        session_id = uuid4()
        mock_settings.data_dir = data_dir
        mock_session_service.get_current_state.return_value = _composition_state_record(
            session_id=session_id,
            source_path=source_path,
            output_path=output_dir / "out.jsonl",
            nodes=[
                {
                    "id": "explode_lines",
                    "node_type": "transform",
                    "plugin": "line_explode",
                    "input": "source_rows",
                    "on_success": "line_rows",
                    "on_error": "errors",
                    "options": {"source_field": "body", "schema": {"mode": "observed"}},
                },
                {
                    "id": "classify_line",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "line_rows",
                    "on_success": "out",
                    "on_error": "errors",
                    "options": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    },
                },
            ],
        )

        with (
            patch.object(service, "_run_pipeline"),
            patch("elspeth.web.execution.service.validate_semantic_contracts", return_value=((), ())),
            pytest.raises(ExecutionFanoutGuardRequired) as raised,
        ):
            await service.execute(session_id=session_id)

        run_id = uuid4()
        mock_session_service.create_run.return_value = MagicMock(id=run_id)
        await service.execute(
            session_id=session_id,
            fanout_ack_token=raised.value.guard.ack_token,
        )

        create_call = mock_session_service.create_run.await_args_list[-1]
        persisted_yaml = create_call.kwargs["pipeline_yaml"]
        assert "elspeth_execution_fanout_guard" in persisted_yaml
        assert '"accepted":true' in persisted_yaml
        assert raised.value.guard.ack_token in persisted_yaml
        assert '"node_id":"classify_line"' in persisted_yaml

    @pytest.mark.asyncio
    async def test_direct_small_text_source_to_llm_executes_without_ack(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A direct low-cardinality source->LLM path remains a one-click run."""
        data_dir = tmp_path
        blob_dir = data_dir / "blobs"
        output_dir = data_dir / "outputs"
        blob_dir.mkdir()
        output_dir.mkdir()
        source_path = blob_dir / "input.txt"
        source_path.write_text("alpha\nbeta\n", encoding="utf-8")
        session_id = uuid4()
        mock_settings.data_dir = data_dir
        mock_session_service.get_current_state.return_value = _composition_state_record(
            session_id=session_id,
            source_path=source_path,
            output_path=output_dir / "out.jsonl",
            nodes=[
                {
                    "id": "classify_row",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "source_rows",
                    "on_success": "out",
                    "on_error": "errors",
                    "options": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    },
                },
            ],
        )

        with (
            patch.object(service, "_run_pipeline"),
            patch("elspeth.web.execution.service.validate_semantic_contracts", return_value=((), ())),
        ):
            run_id = await service.execute(session_id=session_id)

        assert isinstance(run_id, UUID)
        persisted_yaml = mock_session_service.create_run.await_args.kwargs["pipeline_yaml"]
        assert "elspeth_execution_fanout_guard" not in persisted_yaml

    @pytest.mark.asyncio
    async def test_non_first_named_source_to_llm_uses_its_own_cardinality(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Named source fanout accounting must not inspect only the compatibility source."""
        from elspeth.web.execution.fanout_guard import ExecutionFanoutGuardRequired

        data_dir = tmp_path
        blob_dir = data_dir / "blobs"
        output_dir = data_dir / "outputs"
        blob_dir.mkdir()
        output_dir.mkdir()
        orders_path = blob_dir / "orders.txt"
        refunds_path = blob_dir / "refunds.txt"
        orders_path.write_text("one\n", encoding="utf-8")
        refunds_path.write_text("\n".join(f"refund-{i}" for i in range(101)) + "\n", encoding="utf-8")
        session_id = uuid4()
        mock_settings.data_dir = data_dir
        mock_session_service.get_current_state.return_value = _composition_state_record(
            session_id=session_id,
            source_path=orders_path,
            output_path=output_dir / "out.jsonl",
            sources={
                "orders": {
                    "plugin": "text",
                    "on_success": "orders_rows",
                    "on_validation_failure": "discard",
                    "options": {"path": str(orders_path), "column": "body", "schema": {"mode": "observed"}},
                },
                "refunds": {
                    "plugin": "text",
                    "on_success": "refunds_rows",
                    "on_validation_failure": "discard",
                    "options": {"path": str(refunds_path), "column": "body", "schema": {"mode": "observed"}},
                },
            },
            nodes=[
                {
                    "id": "classify_refund",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "refunds_rows",
                    "on_success": "out",
                    "on_error": "errors",
                    "options": {
                        "provider": "openrouter",
                        "model": "openai/gpt-4o-mini",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    },
                },
            ],
        )

        with (
            patch.object(service, "_run_pipeline"),
            patch("elspeth.web.execution.service.validate_semantic_contracts", return_value=((), ())),
            pytest.raises(ExecutionFanoutGuardRequired) as raised,
        ):
            await service.execute(session_id=session_id)

        risk = raised.value.guard.risks[0]
        assert risk.estimated_provider_calls == 101
        assert risk.upstream_fanout == ("source:refunds:text:estimated_rows=101",)
        assert mock_session_service.create_run.await_count == 0


class TestWebRuntimeInfrastructure:
    """Regression coverage for web execution's orchestrator runtime wiring."""

    def test_run_pipeline_records_web_user_attribution_in_landscape(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Web execution must persist who initiated the Landscape run."""
        source_path = tmp_path / "input.txt"
        source_path.write_text("alpha\n", encoding="utf-8")
        output_path = tmp_path / "out.jsonl"
        run_id = str(uuid4())
        mock_settings.get_landscape_url.return_value = f"sqlite:///{tmp_path / 'audit.db'}"
        mock_settings.get_payload_store_path.return_value = tmp_path / "payloads"

        pipeline_yaml = f"""
sources:
  primary:
    plugin: text
    on_success: output
    options:
      path: {source_path}
      column: value
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
        - "value: str"
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      mode: write
      schema:
        mode: observed
"""

        service._run_pipeline(
            run_id,
            pipeline_yaml,
            threading.Event(),
            user_id="alice",
            auth_provider_type="local",
        )

        db = LandscapeDB.from_url(mock_settings.get_landscape_url.return_value, create_tables=False)
        try:
            with db.read_only_connection() as conn:
                attribution_row = conn.execute(select(run_attributions_table).where(run_attributions_table.c.run_id == run_id)).one()
                run_row = conn.execute(select(runs_table.c.settings_json).where(runs_table.c.run_id == run_id)).one()
        finally:
            db.close()

        settings_json = json.loads(run_row.settings_json)
        assert settings_json["sources"]["primary"]["plugin"] == "text"
        assert settings_json["sinks"]["output"]["plugin"] == "json"
        assert attribution_row.initiated_by_user_id == "alice"
        assert attribution_row.auth_provider_type == "local"
        assert output_path.exists()

    def test_web_scrape_pipeline_receives_rate_limit_registry(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Web execution must provide runtime infrastructure required by external-call transforms."""
        import socket
        from datetime import UTC, datetime

        import httpx

        from elspeth.contracts import CallStatus, CallType
        from elspeth.contracts.audit import Call
        from elspeth.contracts.contexts import TransformContext
        from elspeth.core.security.web import SSRFSafeRequest
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        source_path = tmp_path / "input.txt"
        source_path.write_text("https://example.com/page\n", encoding="utf-8")
        output_path = tmp_path / "out.jsonl"
        mock_settings.get_landscape_url.return_value = f"sqlite:///{tmp_path / 'audit.db'}"
        mock_settings.get_payload_store_path.return_value = tmp_path / "payloads"

        def fake_getaddrinfo(
            host: str,
            port: object,
            family: int = 0,
            type: int = 0,
            proto: int = 0,
            flags: int = 0,
        ) -> list[tuple[object, ...]]:
            assert host == "example.com"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        def fake_fetch_url(
            self: WebScrapeTransform,
            safe_request: SSRFSafeRequest,
            ctx: TransformContext,
        ) -> tuple[httpx.Response, str, Call]:
            del self
            return (
                httpx.Response(
                    200,
                    text="<html><body><h1>ok</h1></body></html>",
                    request=httpx.Request("GET", safe_request.connection_url),
                ),
                safe_request.original_url,
                Call(
                    call_id="call-web-runtime",
                    call_index=0,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_hash="request-hash",
                    created_at=datetime.now(UTC),
                    state_id=ctx.state_id or "state-web-runtime",
                    request_ref="request-ref",
                    response_hash="response-hash",
                    response_ref="response-ref",
                    latency_ms=1.0,
                ),
            )

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(WebScrapeTransform, "_fetch_url", fake_fetch_url)

        pipeline_yaml = f"""
sources:
  primary:
    plugin: text
    on_success: scrape_in
    options:
      path: {source_path}
      column: url
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
        - "url: str"
transforms:
- name: scrape_page
  plugin: web_scrape
  input: scrape_in
  on_success: scraped
  on_error: errors
  options:
    schema:
      mode: flexible
      fields:
      - "url: str"
    required_input_fields:
    - url
    url_field: url
    content_field: html
    fingerprint_field: html_fingerprint
    format: raw
    fingerprint_mode: content
    strip_elements: []
    http:
      abuse_contact: tests@example.com
      scraping_reason: test runtime wiring
      timeout: 30
      allowed_hosts: public_only
sinks:
  scraped:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      mode: write
      schema:
        mode: observed
  errors:
    plugin: json
    on_write_failure: discard
    options:
      path: {tmp_path / "errors.jsonl"}
      format: jsonl
      mode: write
      schema:
        mode: observed
"""

        service._run_pipeline(str(uuid4()), pipeline_yaml, threading.Event())

        completed_calls = [
            call for call in mock_session_service.update_run_status.await_args_list if call.kwargs.get("status") == "completed"
        ]
        assert completed_calls
        assert output_path.exists()


# ── B2: shutdown_event Always Passed ───────────────────────────────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestB2ShutdownEvent:
    """B2 fix: _run_pipeline() MUST pass shutdown_event to orchestrator.run().

    If shutdown_event is omitted, the Orchestrator calls signal.signal()
    from the worker thread, raising ValueError: signal only works in main thread.
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_shutdown_event_passed_to_orchestrator_run(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock(spec=object)
        mock_bundle.source = MagicMock(spec=object)
        mock_bundle.sources = {"source": mock_bundle.source}
        mock_bundle.source_settings = MagicMock(spec=object)
        mock_bundle.source_settings_map = {"source": mock_bundle.source_settings}
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock(spec=object)}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        shutdown_event = threading.Event()
        run_id = uuid4()

        mock_orch = MagicMock(spec=["run"])
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock(spec=object)
        mock_result.run_id = str(run_id)
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 10
        mock_result.rows_succeeded = 10
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED),
        ):
            service._run_pipeline(str(run_id), "source:\n  plugin: csv", shutdown_event)

        # B2 invariant: shutdown_event was passed
        orch_run_call = mock_orch.run.call_args
        assert orch_run_call[1].get("shutdown_event") is shutdown_event, (
            "B2 VIOLATION: shutdown_event not passed to orchestrator.run(). This will cause ValueError: signal only works in main thread."
        )
        assert orch_run_call[1].get("run_id") == str(run_id), (
            "Run diagnostics require the web run UUID to be the Landscape run_id while the run is still active."
        )

        running_calls = [call for call in mock_session_service.update_run_status.await_args_list if call.kwargs.get("status") == "running"]
        assert running_calls
        assert running_calls[0].kwargs.get("landscape_run_id") == str(run_id)


# ── B3: LandscapeDB and PayloadStore Construction ─────────────────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestB3Construction:
    """B3 fix: Construct LandscapeDB and FilesystemPayloadStore from WebSettings.

    _run_pipeline() does NOT use hardcoded paths. It calls
    self._settings.get_landscape_url() and self._settings.get_payload_store_path().
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_landscape_db_constructed_from_settings(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_settings: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MagicMock(
            run_id="r1",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=10,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED),
        ):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # B3: LandscapeDB constructed from settings URL
        mock_landscape_cls.assert_called_once_with(connection_string="sqlite:///test_audit.db", passphrase=None)
        # B3: PayloadStore constructed from settings path
        mock_payload_cls.assert_called_once_with(base_path=Path("/tmp/test_payloads"))


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestInlineBlobRuntimePreflight:
    """Inline-content blob refs resolve before plugin construction.

    Bug verification: remove the ``record_blob_inline_resolutions`` call
    from ``ExecutionServiceImpl._run_pipeline`` and this class loses the
    audit-before-settings invariant.
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_run_pipeline_resolves_inline_content_and_records_audit_before_settings_load(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_load: MagicMock,
        mock_runtime_graph: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        content = b"You are an audited prompt."
        blob_id = uuid4()
        run_id = uuid4()
        sha256 = hashlib.sha256(content).hexdigest()
        order: list[str] = []

        blob_record = MagicMock(spec=object)
        blob_record.status = "ready"
        blob_record.content_hash = sha256
        blob_record.mime_type = "text/plain"
        blob_record.size_bytes = len(content)

        async def link_blob_to_run(*_args: Any, **_kwargs: Any) -> None:
            order.append("link")

        async def read_blob_content(_blob_id: UUID) -> bytes:
            order.append("read")
            return content

        async def get_blob(_blob_id: UUID) -> Any:
            order.append("metadata")
            return blob_record

        async def record_blob_inline_resolutions(*_args: Any, **_kwargs: Any) -> None:
            order.append("record")

        blob_service = MagicMock(spec=object)
        blob_service.link_blob_to_run = AsyncMock(side_effect=link_blob_to_run)
        blob_service.read_blob_content = AsyncMock(side_effect=read_blob_content)
        blob_service.get_blob = AsyncMock(side_effect=get_blob)
        blob_service.finalize_run_output_blobs = AsyncMock(return_value=BlobFinalizationResult(finalized=(), errors=()))
        cast(Any, service)._blob_service = blob_service
        mock_session_service.record_blob_inline_resolutions = AsyncMock(side_effect=record_blob_inline_resolutions)

        def load_settings(yaml_text: str) -> MagicMock:
            assert "record" in order, "audit row must be recorded before settings/plugin construction"
            assert "You are an audited prompt." in yaml_text
            assert "blob_ref" not in yaml_text
            assert "inline_content" not in yaml_text
            order.append("load")
            return _mock_pipeline_settings()

        mock_load.side_effect = load_settings

        mock_bundle = MagicMock(spec=object)
        mock_bundle.source = MagicMock(spec=object)
        mock_bundle.sources = {"source": mock_bundle.source}
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock(spec=object)}
        mock_bundle.aggregations = {}
        mock_runtime = MagicMock(spec=object)
        mock_runtime.plugin_bundle = mock_bundle
        mock_runtime.graph = MagicMock(spec=object)
        mock_runtime_graph.return_value = mock_runtime

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.run_id = str(run_id)
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 1
        mock_result.rows_succeeded = 1
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

        pipeline_yaml = f"""
source:
  plugin: csv
  options:
    path: input.csv
transforms:
  - name: classify
    plugin: llm
    options:
      system_prompt:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {sha256}
sinks:
  primary:
    plugin: json
    options:
      path: output.jsonl
"""

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED),
        ):
            service._run_pipeline(str(run_id), pipeline_yaml, threading.Event())

        assert order.index("link") < order.index("read")
        assert order.index("metadata") < order.index("record") < order.index("load")
        blob_service.link_blob_to_run.assert_awaited_once_with(blob_id=blob_id, run_id=run_id, direction="input")
        blob_service.read_blob_content.assert_awaited_once_with(blob_id)
        mock_session_service.record_blob_inline_resolutions.assert_awaited_once()
        resolutions = mock_session_service.record_blob_inline_resolutions.await_args.kwargs["resolutions"]
        assert len(resolutions) == 1
        assert resolutions[0].field_path == "node:classify.options.system_prompt"
        assert resolutions[0].content_hash == sha256

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_audit_write_failure_prevents_settings_load(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_load: MagicMock,
        mock_runtime_graph: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        del mock_payload_cls, mock_landscape_cls, mock_runtime_graph
        content = b"You are an audited prompt."
        blob_id = uuid4()
        run_id = uuid4()
        sha256 = hashlib.sha256(content).hexdigest()

        blob_record = MagicMock(spec=object)
        blob_record.status = "ready"
        blob_record.content_hash = sha256
        blob_record.mime_type = "text/plain"
        blob_record.size_bytes = len(content)

        blob_service = MagicMock(spec=object)
        blob_service.link_blob_to_run = AsyncMock(return_value=None)
        blob_service.read_blob_content = AsyncMock(return_value=content)
        blob_service.get_blob = AsyncMock(return_value=blob_record)
        blob_service.finalize_run_output_blobs = AsyncMock(return_value=BlobFinalizationResult(finalized=(), errors=()))
        cast(Any, service)._blob_service = blob_service
        mock_session_service.record_blob_inline_resolutions = AsyncMock(side_effect=AuditIntegrityError("audit write refused"))

        pipeline_yaml = f"""
source:
  plugin: csv
  options:
    path: input.csv
transforms:
  - name: classify
    plugin: llm
    options:
      system_prompt:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {sha256}
sinks:
  primary:
    plugin: json
    options:
      path: output.jsonl
"""

        with pytest.raises(AuditIntegrityError, match="audit write refused"):
            service._run_pipeline(str(run_id), pipeline_yaml, threading.Event())

        mock_load.assert_not_called()
        mock_orch_cls.assert_not_called()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_oversized_inline_content_metadata_fails_before_blob_read(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_load: MagicMock,
        mock_runtime_graph: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        from elspeth.contracts.blobs_inline import BlobContentResolutionError

        del mock_payload_cls, mock_landscape_cls, mock_runtime_graph
        blob_id = uuid4()
        run_id = uuid4()
        sha256 = hashlib.sha256(b"small prompt").hexdigest()

        blob_record = MagicMock(spec=object)
        blob_record.status = "ready"
        blob_record.content_hash = sha256
        blob_record.mime_type = "text/plain"
        blob_record.size_bytes = 256 * 1024 + 1

        blob_service = MagicMock(spec=object)
        blob_service.link_blob_to_run = AsyncMock(return_value=None)
        blob_service.read_blob_content = AsyncMock(return_value=b"small prompt")
        blob_service.get_blob = AsyncMock(return_value=blob_record)
        blob_service.finalize_run_output_blobs = AsyncMock(return_value=BlobFinalizationResult(finalized=(), errors=()))
        cast(Any, service)._blob_service = blob_service
        mock_session_service.record_blob_inline_resolutions = AsyncMock(return_value=None)

        pipeline_yaml = f"""
source:
  plugin: csv
  options:
    path: input.csv
transforms:
  - name: classify
    plugin: llm
    options:
      system_prompt:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {sha256}
sinks:
  primary:
    plugin: json
    options:
      path: output.jsonl
"""

        with pytest.raises(BlobContentResolutionError) as exc_info:
            service._run_pipeline(str(run_id), pipeline_yaml, threading.Event())

        assert exc_info.value.oversized == (("node:classify.options.system_prompt", 256 * 1024 + 1, 256 * 1024),)
        blob_service.read_blob_content.assert_not_awaited()
        blob_service.link_blob_to_run.assert_not_awaited()
        mock_session_service.record_blob_inline_resolutions.assert_not_called()
        mock_load.assert_not_called()
        mock_orch_cls.assert_not_called()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_aggregate_inline_content_metadata_fails_before_blob_read(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_load: MagicMock,
        mock_runtime_graph: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        from elspeth.contracts.blobs_inline import BlobContentResolutionError

        del mock_payload_cls, mock_landscape_cls, mock_runtime_graph
        blob_ids = [uuid4() for _ in range(5)]
        run_id = uuid4()
        hashes = [hashlib.sha256(f"blob-{index}".encode()).hexdigest() for index in range(5)]

        records_by_id: dict[UUID, Any] = {}
        for blob_id, blob_hash in zip(blob_ids, hashes, strict=True):
            record = MagicMock(spec=object)
            record.status = "ready"
            record.content_hash = blob_hash
            record.mime_type = "text/plain"
            record.size_bytes = 220 * 1024
            records_by_id[blob_id] = record

        async def get_blob(blob_id: UUID) -> Any:
            if blob_id in records_by_id:
                return records_by_id[blob_id]
            raise AssertionError(f"unexpected blob_id {blob_id}")

        blob_service = MagicMock(spec=object)
        blob_service.link_blob_to_run = AsyncMock(return_value=None)
        blob_service.read_blob_content = AsyncMock(return_value=b"content")
        blob_service.get_blob = AsyncMock(side_effect=get_blob)
        blob_service.finalize_run_output_blobs = AsyncMock(return_value=BlobFinalizationResult(finalized=(), errors=()))
        cast(Any, service)._blob_service = blob_service
        mock_session_service.record_blob_inline_resolutions = AsyncMock(return_value=None)

        inline_options = "\n".join(
            f"""      prompt_{index}:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {blob_hash}"""
            for index, (blob_id, blob_hash) in enumerate(zip(blob_ids, hashes, strict=True))
        )
        pipeline_yaml = f"""
source:
  plugin: csv
  options:
    path: input.csv
transforms:
  - name: classify
    plugin: llm
    options:
{inline_options}
sinks:
  primary:
    plugin: json
    options:
      path: output.jsonl
"""

        with pytest.raises(BlobContentResolutionError) as exc_info:
            service._run_pipeline(str(run_id), pipeline_yaml, threading.Event())

        assert exc_info.value.oversized == (("(aggregate)", 5 * 220 * 1024, 1024 * 1024),)
        blob_service.read_blob_content.assert_not_awaited()
        blob_service.link_blob_to_run.assert_not_awaited()
        mock_session_service.record_blob_inline_resolutions.assert_not_called()
        mock_load.assert_not_called()
        mock_orch_cls.assert_not_called()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_hash_mismatch_increments_zero_threshold_counter(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_load: MagicMock,
        mock_runtime_graph: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from elspeth.web.blobs.protocol import BlobIntegrityError
        from elspeth.web.execution import service as service_module

        del mock_payload_cls, mock_landscape_cls, mock_runtime_graph
        content = b"actual prompt bytes"
        blob_id = uuid4()
        run_id = uuid4()
        hash_counter = MagicMock(spec=["add"])
        monkeypatch.setattr(service_module, "_BLOB_INLINE_HASH_MISMATCH_TOTAL", hash_counter)

        blob_record = MagicMock(spec=object)
        blob_record.status = "ready"
        blob_record.content_hash = hashlib.sha256(content).hexdigest()
        blob_record.mime_type = "text/plain"
        blob_record.size_bytes = len(content)

        blob_service = MagicMock(spec=object)
        blob_service.link_blob_to_run = AsyncMock(return_value=None)
        blob_service.read_blob_content = AsyncMock(return_value=content)
        blob_service.get_blob = AsyncMock(return_value=blob_record)
        blob_service.finalize_run_output_blobs = AsyncMock(return_value=BlobFinalizationResult(finalized=(), errors=()))
        cast(Any, service)._blob_service = blob_service

        pipeline_yaml = f"""
source:
  plugin: csv
  options:
    path: input.csv
transforms:
  - name: classify
    plugin: llm
    options:
      system_prompt:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {"b" * 64}
sinks:
  primary:
    plugin: json
    options:
      path: output.jsonl
"""

        with pytest.raises(BlobIntegrityError):
            service._run_pipeline(str(run_id), pipeline_yaml, threading.Event())

        hash_counter.add.assert_called_once_with(1, {"run_id": str(run_id)})
        mock_load.assert_not_called()
        mock_orch_cls.assert_not_called()


class TestWebRuntimeConfigLoading:
    """Web execution rejects file-backed config options before runtime graph construction."""

    @patch("elspeth.web.execution.service.build_validated_runtime_graph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_file_backed_template_options_fail_before_runtime_graph(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_runtime_graph: MagicMock,
        service: ExecutionServiceImpl,
    ) -> None:
        del mock_payload_cls, mock_landscape_cls
        pipeline_yaml = """
source:
  plugin: csv
  on_success: transform_in
  options: {}
transforms:
  - name: classify
    plugin: llm
    input: transform_in
    on_success: results
    on_error: results
    options:
      template_file: prompt.txt
      lookup_file: lookup.yaml
      system_prompt_file: system.txt
sinks:
  primary:
    plugin: json
    on_write_failure: discard
    options:
      path: output.jsonl
"""
        mock_runtime_graph.side_effect = AssertionError("runtime graph must not be built")

        with pytest.raises(ValueError, match="template_file"):
            service._run_pipeline(str(uuid4()), pipeline_yaml, threading.Event())

        mock_runtime_graph.assert_not_called()


# ── B7: BaseException + Done Callback ─────────────────────────────────


class TestB7ExceptionHandling:
    """B7 fix: _run_pipeline() catches BaseException, not Exception.

    Layer 1: try/except BaseException updates run to failed status.
    Layer 2: future.add_done_callback() logs as safety net.
    """

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_keyboard_interrupt_skips_failed_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """R6 fix: KeyboardInterrupt skips _call_async for the 'failed' update.

        The initial 'running' update succeeds (before LandscapeDB raises).
        The except block skips the 'failed' update — orphan cleanup handles it.
        """
        mock_landscape.side_effect = KeyboardInterrupt("ctrl-c")

        with pytest.raises(KeyboardInterrupt):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # R6: The 'running' update went through, but the 'failed' update was skipped
        calls = mock_session_service.update_run_status.call_args_list
        assert len(calls) == 1  # Only the initial "running" call
        assert calls[0][1].get("status") == "running"

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_system_exit_skips_failed_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """R6 fix: SystemExit skips _call_async for the 'failed' update."""
        mock_landscape.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # R6: The 'running' update went through, but the 'failed' update was skipped
        calls = mock_session_service.update_run_status.call_args_list
        assert len(calls) == 1  # Only the initial "running" call
        assert calls[0][1].get("status") == "running"

    def test_shutdown_event_cleaned_up_in_finally(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """finally clause removes shutdown event from _shutdown_events dict."""
        run_id = str(uuid4())
        event = threading.Event()
        service._shutdown_events[run_id] = event

        with patch("elspeth.web.execution.service.LandscapeDB") as mock_db:
            mock_db.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError):
                service._run_pipeline(run_id, "yaml", event)

        # finally must have removed the event
        assert run_id not in service._shutdown_events

    def test_done_callback_logs_last_resort_on_exception(self, service: ExecutionServiceImpl) -> None:
        """Callback logs a last-resort diagnostic when the pipeline future
        carries an exception.  This covers the edge case where _run_pipeline's
        own except block failed (e.g. update_run_status raised).
        """
        future: Future[None] = Future()
        future.set_exception(RuntimeError("unhandled"))

        with patch("elspeth.web.execution.service.slog") as mock_slog:
            service._on_pipeline_done(future)
            mock_slog.error.assert_called_once()
            call_kwargs = mock_slog.error.call_args
            assert call_kwargs[0][0] == "pipeline_done_callback_exception"
            assert call_kwargs[1]["exc_type"] == "RuntimeError"
            # Redaction contract: the slog emits ONLY class names via
            # ``exc_class_chain``. ``exc_msg`` (length-truncated
            # ``str(exc)``) is forbidden because pipeline exceptions may
            # chain SQLAlchemyError payloads, Tier-3 sanitizer text, or
            # source-rendering fragments through ``__cause__`` /
            # ``__context__``.
            assert "exc_msg" not in call_kwargs[1]
            assert call_kwargs[1]["exc_class_chain"] == ["RuntimeError"]

    def test_done_callback_walks_exception_chain(self, service: ExecutionServiceImpl) -> None:
        """Chained exceptions surface as a class-name chain — no payloads.

        Regression: ``exc_msg=str(exc)[:200]`` leaked truncated-but-still-
        sensitive text. The chain walk visits ``__cause__`` / ``__context__``
        and records only ``type(current).__name__``.
        """
        try:
            try:
                raise ValueError("secret=deadbeef")  # Tier-3-ish payload
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as outer:
            future: Future[None] = Future()
            future.set_exception(outer)

        with patch("elspeth.web.execution.service.slog") as mock_slog:
            service._on_pipeline_done(future)
            call_kwargs = mock_slog.error.call_args[1]
            assert call_kwargs["exc_type"] == "RuntimeError"
            assert call_kwargs["exc_class_chain"] == ["RuntimeError", "ValueError"]
            # No ``str(exc)`` text should appear in any field.
            for value in call_kwargs.values():
                if isinstance(value, str):
                    assert "secret" not in value
                    assert "deadbeef" not in value

    def test_done_callback_noop_on_success(self, service: ExecutionServiceImpl) -> None:
        """done_callback does not log on successful completion."""
        future: Future[None] = Future()
        future.set_result(None)

        with patch("elspeth.web.execution.service.slog") as mock_slog:
            service._on_pipeline_done(future)
            mock_slog.error.assert_not_called()

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_pydantic_validation_error_emits_schema_contract_diagnostic(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Strict schema crashes need operator diagnostics, not just generic failure."""
        from pydantic import BaseModel
        from pydantic import ValidationError as PydanticValidationError

        class SchemaContractProbe(BaseModel):
            internal_required_field: int

        try:
            SchemaContractProbe()
        except PydanticValidationError as exc:
            validation_error = exc
        else:
            raise AssertionError("expected SchemaContractProbe() to raise")

        run_id = str(uuid4())
        mock_session_service.get_run.return_value = MagicMock(status="running")

        with (
            patch(
                "elspeth.web.execution.service.load_settings_from_yaml_string",
                side_effect=validation_error,
            ),
            patch("elspeth.web.execution.service.slog") as mock_slog,
            pytest.raises(PydanticValidationError),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv\n", threading.Event())

        schema_calls = [call for call in mock_slog.error.call_args_list if call.args[0] == "run_schema_contract_violation"]
        assert len(schema_calls) == 1
        schema_kwargs = schema_calls[0].kwargs
        assert schema_kwargs["run_id"] == run_id
        assert schema_kwargs["exc_class"] == "ValidationError"
        assert schema_kwargs["error_count"] == 1
        assert schema_kwargs["schema_errors"] == [{"loc": "internal_required_field", "type": "missing"}]

        failed_calls = [call for call in mock_session_service.update_run_status.call_args_list if call.kwargs.get("status") == "failed"]
        assert failed_calls
        assert failed_calls[-1].kwargs["error"] == "Pipeline execution failed (ValidationError)"
        assert "internal_required_field" not in failed_calls[-1].kwargs["error"]


# ── Cancel Mechanism ───────────────────────────────────────────────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestCancelMechanism:
    @pytest.mark.asyncio
    async def test_cancel_active_run_sets_event(self, service: ExecutionServiceImpl) -> None:
        run_id = uuid4()
        event = threading.Event()
        service._shutdown_events[str(run_id)] = event

        await service.cancel(run_id)

        assert event.is_set(), "cancel() must set the threading.Event so the Orchestrator detects it during row processing"

    @pytest.mark.asyncio
    async def test_get_status_marks_active_set_event_as_cancel_requested(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        run_id = uuid4()
        event = threading.Event()
        event.set()
        service._shutdown_events[str(run_id)] = event
        mock_session_service.get_run.return_value = MagicMock(
            id=run_id,
            status="running",
            started_at=datetime.now(UTC),
            finished_at=None,
            error=None,
            landscape_run_id=None,
        )

        status = await service.get_status(run_id)

        assert status.status == "running"
        assert status.cancel_requested is True

    @pytest.mark.asyncio
    async def test_cancel_pending_run_updates_status(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When no shutdown event exists (pending), update status directly."""
        run_id = uuid4()
        # No event in _shutdown_events — run is pending
        await service.cancel(run_id)
        mock_session_service.update_run_status.assert_called()

    @pytest.mark.parametrize("terminal_status", ["completed", "completed_with_failures", "failed", "empty", "cancelled"])
    @pytest.mark.asyncio
    async def test_cancel_terminal_run_is_noop(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        terminal_status: str,
    ) -> None:
        """Cancelling any terminal run does nothing."""
        run_id = uuid4()
        mock_session_service.get_run.return_value = MagicMock(status=terminal_status)
        await service.cancel(run_id)
        mock_session_service.update_run_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_idempotent_on_set_event(self, service: ExecutionServiceImpl) -> None:
        """Setting an already-set event is safe."""
        run_id = uuid4()
        event = threading.Event()
        event.set()
        service._shutdown_events[str(run_id)] = event

        # Should not raise
        await service.cancel(run_id)
        assert event.is_set()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_cancelled_run_broadcasts_cancelled_event(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When orchestrator raises GracefulShutdownError, _run_pipeline
        broadcasts 'cancelled' and updates status accordingly."""
        from elspeth.contracts.errors import GracefulShutdownError

        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        # Orchestrator raises GracefulShutdownError on actual cancellation
        mock_orch.run.side_effect = GracefulShutdownError(
            rows_processed=50,
            run_id="test-run-001",
            rows_succeeded=48,
            rows_failed=2,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        shutdown_event = threading.Event()
        shutdown_event.set()
        run_id = str(uuid4())

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED_WITH_FAILURES),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        # The test sets shutdown_event BEFORE _run_pipeline, so the early
        # shutdown check (line 534) fires — no orchestrator runs, no row counts.
        # Verify status updated to "cancelled" via the early-exit path.
        status_calls = mock_session_service.update_run_status.call_args_list
        final_status_call = status_calls[-1]
        assert final_status_call.kwargs["status"] == "cancelled"

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_graceful_shutdown_forwards_row_counts(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """GracefulShutdownError row counts are forwarded to update_run_status.

        Regression: prior test only asserted status=='cancelled' but did
        not verify that rows_processed, rows_succeeded, rows_failed, and
        rows_quarantined were propagated from the GSE to the session service.
        """
        from elspeth.contracts.errors import GracefulShutdownError

        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.side_effect = GracefulShutdownError(
            rows_processed=50,
            run_id="test-run-gse",
            rows_succeeded=48,
            rows_failed=2,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        # Do NOT set shutdown_event — let _run_pipeline proceed past the
        # early check so orchestrator.run() fires and raises the GSE.
        shutdown_event = threading.Event()
        run_id = str(uuid4())

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED_WITH_FAILURES),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        status_calls = mock_session_service.update_run_status.call_args_list
        # Second call is the GSE handler (first is running transition)
        gse_call = status_calls[-1]
        assert gse_call.kwargs["status"] == "cancelled"
        assert gse_call.kwargs["rows_processed"] == 50
        assert gse_call.kwargs["rows_succeeded"] == 48
        assert gse_call.kwargs["rows_failed"] == 2
        assert gse_call.kwargs["rows_routed_success"] == 0
        assert gse_call.kwargs["rows_routed_failure"] == 0
        assert gse_call.kwargs["rows_quarantined"] == 0

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_completed_run_not_misclassified_when_event_set_late(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Race guard: if shutdown_event is set AFTER orchestrator completes
        (returns normally), the run must still be classified as 'completed',
        not 'cancelled'."""
        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED_WITH_FAILURES
        mock_result.rows_processed = 50
        mock_result.rows_succeeded = 48
        mock_result.rows_failed = 2
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-late-cancel"
        mock_orch.run.return_value = mock_result

        # Simulate late cancel: event is set DURING orchestrator.run()
        # (after it returns its result), not before _run_pipeline starts.
        # This tests the race where cancel() fires after the orchestrator
        # finishes but before status is persisted.
        shutdown_event = threading.Event()

        original_return = mock_result

        def set_event_on_run(*args: object, **kwargs: object) -> MagicMock:
            shutdown_event.set()
            return original_return

        mock_orch.run.side_effect = set_event_on_run
        run_id = str(uuid4())

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED_WITH_FAILURES),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        # Must be "completed", NOT "cancelled"
        status_calls = mock_session_service.update_run_status.call_args_list
        final_status_call = status_calls[-1]
        assert "completed" in str(final_status_call), f"Expected 'completed' status update, got: {final_status_call}"

    # ── Race condition: cancel() before _run_pipeline starts ──────────

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_run_pipeline_exits_gracefully_when_already_cancelled(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Race fix: if cancel() set DB to 'cancelled' before _run_pipeline
        starts, the pending→running transition fails. _run_pipeline must
        detect this and exit cleanly — no Orchestrator, no crash."""
        run_id = str(uuid4())

        # Simulate: update_run_status("running") raises because status is "cancelled"
        mock_session_service.update_run_status.side_effect = IllegalRunTransitionError("cancelled", "running", frozenset())
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

        # Should NOT raise — graceful exit
        service._run_pipeline(run_id, "yaml", threading.Event())

        # No Orchestrator or LandscapeDB instantiated (early return)
        mock_landscape.assert_not_called()
        mock_payload.assert_not_called()

        # Only the one failed "running" attempt — no "failed" status update
        assert mock_session_service.update_run_status.call_count == 1

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_run_pipeline_early_shutdown_skips_setup(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If shutdown_event is already set when _run_pipeline starts,
        skip all setup and immediately transition to cancelled."""
        run_id = str(uuid4())

        shutdown_event = threading.Event()
        shutdown_event.set()

        service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        # No LandscapeDB or PayloadStore constructed (skipped setup)
        mock_landscape.assert_not_called()
        mock_payload.assert_not_called()

        # Status updated to "cancelled"
        status_calls = mock_session_service.update_run_status.call_args_list
        assert len(status_calls) == 1
        assert "cancelled" in str(status_calls[0])

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_run_pipeline_reraises_valueerror_when_not_cancelled(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If update_run_status raises ValueError for a reason other than
        'already cancelled', _run_pipeline must re-raise (offensive programming)."""
        run_id = str(uuid4())

        mock_session_service.update_run_status.side_effect = IllegalRunTransitionError("completed", "running", frozenset())
        mock_session_service.get_run.return_value = MagicMock(status="completed")

        with pytest.raises(ValueError, match="completed"):
            service._run_pipeline(run_id, "yaml", threading.Event())

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_running_transition_does_not_swallow_non_illegal_value_errors(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Tier-1 invariant: the four non-illegal-transition ValueError sites in
        update_run_status (run-not-found, landscape_run_id overwrite,
        completed-without-landscape, failed-without-error) must NOT be caught by
        the cancelled-race recovery at the running-transition site (originally
        684).  The only catchable class is IllegalRunTransitionError.

        Discriminator (regression-resistant): under the *old* broad
        ``except ValueError`` the cancelled-race recovery would (a) consult
        get_run, (b) see status == "cancelled", (c) broadcast a "cancelled" SSE
        event, and (d) ``return`` silently — masking the Tier-1 breach as a
        normal cancellation.  The bare ValueError would never propagate.

        Under the narrowed catch, the bare ValueError propagates verbatim
        (preserving the original message), no "cancelled" SSE is emitted from
        the cancelled-race path, and the only get_run call comes from the
        downstream BaseException post-terminal recovery (separate audit-primacy
        machinery — its get_run is correct and expected).

        Without this test a future maintainer could re-widen
        ``except IllegalRunTransitionError`` back to ``except ValueError`` and
        the recovery-path tests would still pass.
        """
        run_id = str(uuid4())

        sentinel_message = "landscape_run_id already set to 'sentinel-existing-id'; cannot overwrite"
        mock_session_service.update_run_status.side_effect = ValueError(sentinel_message)
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

        broadcast_calls: list[tuple[str, Any]] = []
        original_broadcast = service._broadcaster.broadcast

        def spy_broadcast(rid: str, event: Any) -> None:
            broadcast_calls.append((rid, event))
            original_broadcast(rid, event)

        service._broadcaster.broadcast = spy_broadcast  # type: ignore[assignment]

        with pytest.raises(ValueError, match="sentinel-existing-id") as exc_info:
            service._run_pipeline(run_id, "yaml", threading.Event())

        # Discriminator #1: the propagating exception is bare ValueError, not the
        # narrow subclass — proves the catch did not match.
        assert not isinstance(exc_info.value, IllegalRunTransitionError)
        # Discriminator #2: no "cancelled" SSE was broadcast.  Old broad-catch
        # behaviour would emit one before silently returning.
        cancelled_events = [event for (_, event) in broadcast_calls if event.event_type == "cancelled"]
        assert cancelled_events == [], f"unexpected cancelled SSE — masking window re-opened: {cancelled_events}"

    @pytest.mark.asyncio
    async def test_shutdown_event_registered_before_blob_linkage(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Race fix part 2: _shutdown_events registration must happen before
        blob linkage, so cancel() finds the event during the blob window."""
        session_id = uuid4()
        run_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"
        mock_session_service.create_run.return_value = MagicMock(id=run_id)

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))

        async def tracking_link(*args: Any, **kwargs: Any) -> None:
            # At the time blob linkage runs, the event MUST already exist
            assert str(run_id) in service._shutdown_events, "RACE: _shutdown_events not registered before blob linkage"

        blob_service.link_blob_to_run = AsyncMock(side_effect=tracking_link)
        cast(Any, service)._blob_service = blob_service

        # Set up state record with a source containing a blob_ref.
        # Use a real dict so state_from_record → deep_thaw works correctly.
        # path must equal blob.storage_path to satisfy the Tier 1 read
        # guard for blob-backed sources (elspeth-07089fbaa3).
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref, "path": canonical_path},
            "on_validation_failure": "quarantine",
        }

        with patch.object(service, "_run_pipeline"):
            await service.execute(session_id=session_id)

    @pytest.mark.asyncio
    async def test_shutdown_event_cleaned_up_on_blob_linkage_failure(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If blob linkage raises after event registration, the event must
        be cleaned up to avoid leaking into _shutdown_events."""
        session_id = uuid4()
        run_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"
        mock_session_service.create_run.return_value = MagicMock(id=run_id)

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        blob_service.link_blob_to_run = AsyncMock(side_effect=RuntimeError("blob storage unavailable"))
        cast(Any, service)._blob_service = blob_service

        # Use a real dict so state_from_record → deep_thaw works correctly.
        # path must equal blob.storage_path to satisfy the Tier 1 read
        # guard for blob-backed sources (elspeth-07089fbaa3).
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref, "path": canonical_path},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(RuntimeError, match="blob storage unavailable"):
            await service.execute(session_id=session_id)

        assert str(run_id) not in service._shutdown_events


class TestP2aCleanupCatchNarrowing:
    """Regression (P2a): cleanup catches in ExecutionServiceImpl must not
    launder exception strings into slog.

    ``except Exception`` over ``update_run_status`` previously logged
    ``cleanup_error=str(cleanup_err)``. On SQLAlchemyError subclasses that
    expands to ``[SQL: ...] [parameters: ...]`` plus a ``__cause__`` chain
    that can carry DB URLs / credentials. Canonical pattern (commits
    b8ba2214/127417cb): narrow to ``(SQLAlchemyError, OSError)`` and log
    ``exc_class`` only.
    """

    @pytest.mark.asyncio
    async def test_setup_failure_cleanup_slog_uses_exc_class_not_str(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When a setup failure triggers cleanup and cleanup's own
        ``update_run_status`` raises a ``SQLAlchemyError``, the slog
        record must carry ``cleanup_exc_class`` + ``original_exc_class``
        (class names) — not the legacy ``cleanup_error``/``original_error``
        string fields."""
        from sqlalchemy.exc import OperationalError

        session_id = uuid4()
        run_id = uuid4()
        mock_session_service.create_run.return_value = MagicMock(id=run_id)

        # First update_run_status call (in cleanup) raises OperationalError.
        mock_session_service.update_run_status.side_effect = OperationalError(
            "UPDATE runs ...",
            {"id": str(run_id), "error": "Setup failed: SuperSecretDSN://u:p@h/d"},
            Exception("lock wait timeout exceeded — __cause__ carries DSN"),
        )

        # Force the setup path to fail so the cleanup catch fires.
        # _executor.submit raising is the simplest route.
        service._executor.submit = MagicMock(side_effect=RuntimeError("pool shutdown"))  # type: ignore[method-assign]

        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            pytest.raises(RuntimeError, match="pool shutdown"),
        ):
            await service.execute(session_id=session_id)

        # slog.error was called for the cleanup failure.
        slog_calls = [c for c in mock_slog.error.call_args_list if c[0] and c[0][0] == "run_cleanup_status_update_failed"]
        assert len(slog_calls) == 1, mock_slog.error.call_args_list
        kwargs = slog_calls[0][1]

        # The narrow-catch kwargs are class names, not strings.
        assert kwargs["cleanup_exc_class"] == "OperationalError"
        assert kwargs["original_exc_class"] == "RuntimeError"

        # Legacy string-valued fields are GONE — this is the redaction
        # regression guard. Any reintroduction re-opens the str(exc) leak.
        assert "cleanup_error" not in kwargs
        assert "original_error" not in kwargs

    @pytest.mark.asyncio
    async def test_setup_cleanup_narrow_catch_lets_runtimeerror_escape(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Narrow catch semantics: a RuntimeError from update_run_status
        (programmer bug, not a DB/filesystem failure) MUST propagate
        instead of being swallowed. Pre-narrowing, the broad
        ``except Exception`` masked such bugs."""
        session_id = uuid4()
        run_id = uuid4()
        mock_session_service.create_run.return_value = MagicMock(id=run_id)

        # First update_run_status (in cleanup) raises RuntimeError — outside
        # the narrow (SQLAlchemyError, OSError) catch. It must escape.
        mock_session_service.update_run_status.side_effect = RuntimeError("dataclass contract violated inside update_run_status")

        # Force setup to fail so cleanup fires.
        service._executor.submit = MagicMock(side_effect=RuntimeError("pool shutdown"))  # type: ignore[method-assign]

        # The RuntimeError from update_run_status escapes the narrow catch.
        # The outer `raise` is bypassed — the cleanup RuntimeError wins
        # (Python's implicit exception chaining preserves both via
        # __context__, but the foreground exception is the cleanup one).
        # We accept either RuntimeError here — the key invariant is that
        # a RuntimeError propagates rather than being swallowed.
        with pytest.raises(RuntimeError):
            await service.execute(session_id=session_id)


# ── Completion-Path Guard ─────────────────────────────────────────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestCompletionPathExternalCancellation:
    """Defence-in-depth: if the DB says 'cancelled' when _run_pipeline
    tries to write 'completed', exit gracefully — no 'failed' broadcast,
    no BaseException cascade, no re-raise."""

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_run_pipeline_exits_gracefully_when_completed_but_db_cancelled(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Pipeline completes, but orphan cleanup already set DB to 'cancelled'.
        _run_pipeline must detect this and return cleanly."""
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED_WITH_FAILURES
        mock_result.rows_processed = 100
        mock_result.rows_succeeded = 95
        mock_result.rows_failed = 5
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-123"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())

        # First call: update_run_status("running") succeeds.
        # Second call: update_run_status("completed") raises ValueError
        # because the DB was externally set to "cancelled".
        call_count = 0

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise IllegalRunTransitionError("cancelled", "completed", frozenset())

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

        # Should NOT raise — graceful exit
        service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # The "failed" path should NOT have been entered: check that
        # update_run_status was called exactly twice (running + completed),
        # NOT three times (running + completed + failed).
        assert mock_session_service.update_run_status.call_count == 2

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_cancelled_compensating_event_broadcast_on_external_cancel(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When pipeline completes but DB says 'cancelled', exactly one
        terminal event must be broadcast: 'cancelled' (the DB is authoritative).
        No 'completed' event should be emitted — finalize-first ordering
        ensures the terminal broadcast reflects the actual DB state."""
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED_WITH_FAILURES
        mock_result.rows_processed = 100
        mock_result.rows_succeeded = 95
        mock_result.rows_failed = 5
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-789"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())

        call_count = 0

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise IllegalRunTransitionError("cancelled", "completed", frozenset())

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

        broadcast_calls: list[tuple[str, Any]] = []
        original_broadcast = service._broadcaster.broadcast

        def spy_broadcast(rid: str, event: Any) -> None:
            broadcast_calls.append((rid, event))
            original_broadcast(rid, event)

        service._broadcaster.broadcast = spy_broadcast  # type: ignore[assignment]

        service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        event_types = [call[1].event_type for call in broadcast_calls]
        terminal_types = [et for et in event_types if et in ("completed", "failed", "cancelled")]
        assert terminal_types == ["cancelled"], f"Expected exactly one 'cancelled' terminal, got: {terminal_types}"

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_external_cancel_finalizes_output_blobs_as_error(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Cancelled runs must not leave output blobs finalized as ready."""
        from elspeth.web.blobs.protocol import BlobFinalizationResult

        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 7
        mock_result.rows_succeeded = 7
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-blob-cancel"
        mock_orch.run.return_value = mock_result

        blob_state = {"status": "pending"}
        blob_calls: list[bool] = []

        async def finalize_run_output_blobs(run_id: UUID, success: bool) -> BlobFinalizationResult:
            del run_id
            blob_calls.append(success)
            if blob_state["status"] == "pending":
                blob_state["status"] = "ready" if success else "error"
            return BlobFinalizationResult(finalized=[], errors=[])

        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=finalize_run_output_blobs)
        cast(Any, service)._blob_service = blob_service

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            if kwargs.get("status") == "completed":
                raise IllegalRunTransitionError("cancelled", "completed", frozenset())

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED),
        ):
            service._run_pipeline(str(uuid4()), "source:\n  plugin: csv", threading.Event())

        assert blob_calls == [False]
        assert blob_state["status"] == "error"

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_completion_guard_reraises_for_non_cancelled_status(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If update_run_status('completed') raises ValueError for a reason
        other than 'already cancelled', the error must propagate (offensive)."""
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 10
        mock_result.rows_succeeded = 10
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-456"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())

        call_count = 0

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise IllegalRunTransitionError("completed", "completed", frozenset())

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        # DB says "completed" (not "cancelled") — this should re-raise
        mock_session_service.get_run.return_value = MagicMock(status="completed")

        with pytest.raises(ValueError, match="completed"):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_completion_guard_does_not_swallow_non_illegal_value_errors(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Tier-1 invariant for the completion-transition catch (originally
        ~915): a bare ValueError raised for a non-illegal-transition reason
        (run-not-found, landscape_run_id overwrite, completed-without-landscape,
        failed-without-error) must propagate verbatim and must NOT trigger the
        cancelled-race silent-swallow path.

        Discriminator (regression-resistant): under the old broad
        ``except ValueError`` the cancelled-race recovery would (a) consult
        get_run, (b) see status == "cancelled", (c) broadcast a "cancelled"
        SSE event, and (d) ``return`` silently — masking the Tier-1 breach as
        a normal cancellation.  The bare ValueError would never propagate.

        Under the narrowed catch the bare ValueError propagates and no
        "cancelled" SSE is emitted from the cancelled-race path.  (The
        downstream BaseException post-terminal recovery may consult get_run
        as part of separate audit-primacy machinery; that's expected and
        correct, so we assert on broadcast shape rather than get_run call
        count.)

        Without this test a future widening of
        ``except IllegalRunTransitionError`` back to ``except ValueError`` would
        silently re-open the masking window identified by silent-failure-hunter
        (H1) — and the existing recovery-path tests would still pass.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 10
        mock_result.rows_succeeded = 10
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-sentinel"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())
        call_count = 0
        sentinel_message = "landscape_run_id already set to 'sentinel-existing-id'; cannot overwrite"

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Bare ValueError simulating one of the four non-illegal-transition
                # invariant breaches in update_run_status.
                raise ValueError(sentinel_message)

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        mock_session_service.get_run = AsyncMock(return_value=MagicMock(status="cancelled"))

        broadcast_calls: list[tuple[str, Any]] = []
        original_broadcast = service._broadcaster.broadcast

        def spy_broadcast(rid: str, event: Any) -> None:
            broadcast_calls.append((rid, event))
            original_broadcast(rid, event)

        service._broadcaster.broadcast = spy_broadcast  # type: ignore[assignment]

        with pytest.raises(ValueError, match="sentinel-existing-id") as exc_info:
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # Discriminator #1: bare ValueError, not the narrow subclass.
        assert not isinstance(exc_info.value, IllegalRunTransitionError)
        # Discriminator #2: no "cancelled" SSE — old broad-catch behaviour
        # would emit one before silently returning.
        cancelled_events = [event for (_, event) in broadcast_calls if event.event_type == "cancelled"]
        assert cancelled_events == [], f"unexpected cancelled SSE — masking window re-opened: {cancelled_events}"


# ── Post-Completion Exception Recovery (elspeth-879f6de6bd) ───────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestPostCompletionExceptionRecovery:
    """Defence-in-depth: when a BaseException fires AFTER ``update_run_status``
    has already committed a terminal state, the recovery must not attempt an
    illegal terminal→failed transition.

    elspeth-879f6de6bd: ``LEGAL_RUN_TRANSITIONS`` makes all five terminal
    statuses (``completed``, ``completed_with_failures``, ``failed``,
    ``empty``, ``cancelled``) outgoing-empty.  The pre-fix recovery at
    ``_run_pipeline``'s ``except BaseException`` handler attempted
    ``update_run_status(status="failed", ...)`` unconditionally, raising
    ``ValueError("Illegal run transition: 'completed' → 'failed'. Allowed: []")``
    and losing the original exception in ``__context__``.

    The fix consults ``get_run`` first; if the run is already terminal it
    skips both the status update and the misleading ``failed`` SSE broadcast
    (audit primacy: SSE must not contradict the audit row).
    """

    @staticmethod
    def _make_completed_orchestrator(status: RunStatus = RunStatus.COMPLETED) -> MagicMock:
        """Build an orchestrator stub whose ``run`` returns a terminal result.

        Counts are status-aware so the SSE-payload status/count cross-consistency
        validator (``CompletedData._check_status_consistency``) accepts the
        result.  COMPLETED_WITH_FAILURES requires both a success and a failure
        indicator; EMPTY requires zero rows; COMPLETED requires success only;
        FAILED tolerates any shape.
        """
        mock_orch = MagicMock()
        mock_result = MagicMock()
        mock_result.status = status
        if status == RunStatus.COMPLETED_WITH_FAILURES:
            mock_result.rows_processed = 10
            mock_result.rows_succeeded = 8
            mock_result.rows_failed = 2
            mock_result.rows_routed_success = 0
            mock_result.rows_routed_failure = 0
            mock_result.rows_quarantined = 0
        elif status == RunStatus.EMPTY:
            mock_result.rows_processed = 0
            mock_result.rows_succeeded = 0
            mock_result.rows_failed = 0
            mock_result.rows_routed_success = 0
            mock_result.rows_routed_failure = 0
            mock_result.rows_quarantined = 0
        else:
            mock_result.rows_processed = 10
            mock_result.rows_succeeded = 10
            mock_result.rows_failed = 0
            mock_result.rows_routed_success = 0
            mock_result.rows_routed_failure = 0
            mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-postcompletion"
        mock_orch.run.return_value = mock_result
        return mock_orch

    @staticmethod
    def _wrap_broadcaster_to_raise(
        service: ExecutionServiceImpl,
        *,
        on_event_type: str,
        exc: BaseException,
    ) -> list[tuple[str, Any]]:
        """Replace the broadcaster with a spy that records all calls and raises
        ``exc`` when the broadcast event_type matches ``on_event_type``.

        Returns the list that will be appended to as broadcasts occur.  Tests
        can assert on event_types after ``_run_pipeline`` returns.
        """
        original_broadcast = service._broadcaster.broadcast
        broadcast_calls: list[tuple[str, Any]] = []

        def crashing_broadcast(rid: str, event: Any) -> None:
            broadcast_calls.append((rid, event))
            if event.event_type == on_event_type:
                raise exc
            original_broadcast(rid, event)

        service._broadcaster.broadcast = crashing_broadcast  # type: ignore[assignment]
        return broadcast_calls

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_post_completion_broadcast_crash_skips_failed_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Run completes; success-path ``broadcast("completed", ...)`` raises
        ``RuntimeError``.  The recovery must NOT attempt a third
        ``update_run_status(status="failed", ...)`` because the audit row is
        already ``completed`` (illegal terminal→failed transition).
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch_cls.return_value = self._make_completed_orchestrator(RunStatus.COMPLETED)

        # update_run_status is permissive (just records calls); the audit row
        # is conceptually "completed" after the second call.  The recovery's
        # third call (with status="failed") MUST NOT happen.
        mock_session_service.get_run.return_value = MagicMock(status="completed")

        broadcast_calls = self._wrap_broadcaster_to_raise(
            service,
            on_event_type="completed",
            exc=RuntimeError("simulated SSE crash"),
        )

        run_id = str(uuid4())
        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            patch(
                "elspeth.web.execution.service.load_run_accounting_from_db",
                return_value=_run_accounting_for_status(RunStatus.COMPLETED),
            ),
            pytest.raises(RuntimeError, match="simulated SSE crash"),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # Exactly two status updates: "running" (line 650) and the terminal
        # "completed" (line 857).  No third "failed" call.
        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "completed"], f"Expected [running, completed], got {statuses}"

        # Audit-primacy guarantees enforced at this site:
        #   1. No third update_run_status (asserted above).
        #   2. No "failed" SSE broadcast (asserted below) — would contradict
        #      the audit row's true terminal status.
        # Post-audit-exception observability is provided by two channels
        # outside _run_pipeline (see service.py post-terminal-exception
        # comment block):
        #   - The audit ``runs`` row (status="completed" — verified by the
        #     ``statuses`` assertion above by transitive proof).
        #   - ``_on_pipeline_done``'s safety-net slog
        #     (``pipeline_done_callback_exception``) — fires against the
        #     re-raised exc once the Future completes.  Tested separately
        #     in the _on_pipeline_done test class.
        # This test must NOT pin a slog at the post-terminal-exception
        # site itself: per ``logging-telemetry-policy`` the logger is
        # not the correct surface for post-audit operational signal.
        post_terminal_logs = [
            c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_terminal_exception_in_run_pipeline"
        ]
        assert post_terminal_logs == [], (
            "post_terminal_exception_in_run_pipeline slog has been removed "
            "(audit-primacy fix); the post-audit signal is captured by the "
            "audit row + _on_pipeline_done safety-net log."
        )

        # No "failed" SSE event must be emitted from the recovery — the run
        # actually completed; broadcasting "failed" would diverge from audit.
        recovery_failed_events = [event for (_, event) in broadcast_calls if event.event_type == "failed"]
        assert recovery_failed_events == [], f"Recovery must not broadcast 'failed' when run is terminal; got: {recovery_failed_events}"

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_post_completion_with_failures_also_skips_failed_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Regression catcher for the partial-tuple bug pattern.

        Several call sites in the codebase use a hardcoded
        ``("completed", "failed", "cancelled")`` tuple to detect terminality,
        which omits ``completed_with_failures`` and ``empty``.  The fix MUST
        use the full terminal set derived from ``LEGAL_RUN_TRANSITIONS`` /
        ``SESSION_TERMINAL_RUN_STATUS_VALUES``.  This test proves the guard
        fires for ``completed_with_failures`` — a state the partial tuple
        would miss.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch_cls.return_value = self._make_completed_orchestrator(RunStatus.COMPLETED_WITH_FAILURES)

        mock_session_service.get_run.return_value = MagicMock(status="completed_with_failures")

        self._wrap_broadcaster_to_raise(
            service,
            on_event_type="completed",
            exc=RuntimeError("simulated SSE crash"),
        )

        run_id = str(uuid4())
        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            patch(
                "elspeth.web.execution.service.load_run_accounting_from_db",
                return_value=_run_accounting_for_status(RunStatus.COMPLETED_WITH_FAILURES),
            ),
            pytest.raises(RuntimeError),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "completed_with_failures"], f"Expected [running, completed_with_failures], got {statuses}"

        # Audit-primacy fix: the post-audit slog has been removed from
        # this site.  See test_post_completion_broadcast_crash_skips_failed_status_update
        # for the rationale; the assertion here pins the same invariant for
        # the COMPLETED_WITH_FAILURES branch (the non-completed branch the
        # partial-tuple bug pattern would have missed).
        post_terminal_logs = [
            c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_terminal_exception_in_run_pipeline"
        ]
        assert post_terminal_logs == [], "post_terminal_exception_in_run_pipeline slog has been removed (audit-primacy fix)."

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_post_completion_get_run_probe_failure_falls_through(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When ``get_run`` itself raises during the recovery probe, fall
        through to the existing best-effort recovery (attempt
        ``update_run_status("failed", ...)``).

        Documents a **known gap**: this test exercises the fall-through with
        a permissive ``update_run_status`` mock that does NOT enforce
        ``LEGAL_RUN_TRANSITIONS``.  In production, if the run is genuinely
        terminal AND the probe fails AND ``update_run_status`` therefore
        raises ``ValueError``, the original exception is still lost — same
        as today's behaviour.  Closing that gap requires a larger design
        change (fail-closed on probe failure, or post-hoc reconcile of the
        ValueError) and is out of scope for elspeth-879f6de6bd.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch_cls.return_value = self._make_completed_orchestrator(RunStatus.COMPLETED)

        # Probe raises a generic SQLAlchemy-family-ish error.  Use the actual
        # SQLAlchemyError to match the recovery's narrow catch.
        mock_session_service.get_run.side_effect = SQLAlchemyError("simulated DB hiccup")

        self._wrap_broadcaster_to_raise(
            service,
            on_event_type="completed",
            exc=RuntimeError("simulated SSE crash"),
        )

        run_id = str(uuid4())
        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            patch(
                "elspeth.web.execution.service.load_run_accounting_from_db",
                return_value=_run_accounting_for_status(RunStatus.COMPLETED),
            ),
            pytest.raises(RuntimeError, match="simulated SSE crash"),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # The probe failure must surface as a structured log so it's observable.
        probe_failed_logs = [c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_exception_run_state_probe_failed"]
        assert len(probe_failed_logs) == 1, f"Expected one post_exception_run_state_probe_failed log, got {len(probe_failed_logs)}"

        # Fall-through: recovery DID attempt the failed update.  Three calls
        # total: running, completed, failed.
        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "completed", "failed"], (
            f"Probe-failure path must fall through to update_run_status('failed', ...); got {statuses}"
        )

        # The post_terminal_exception_in_run_pipeline log must NOT have
        # been emitted (we couldn't determine the run was terminal).
        post_terminal_logs = [
            c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_terminal_exception_in_run_pipeline"
        ]
        assert post_terminal_logs == []

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_post_completion_probe_failure_with_legal_transitions_preserves_original_exception(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Lock the residual data-loss path into the test surface (pr-test-analyzer IG-2).

        Sister to ``test_post_completion_get_run_probe_failure_falls_through``,
        but wires real ``LEGAL_RUN_TRANSITIONS`` semantics into the
        ``update_run_status`` mock so the gap acknowledged in the production
        comment block on the post-exception probe (``run_already_terminal``
        false-on-probe-failure) is actually exercised by the test surface —
        not just documented in prose.

        Scenario:
          1. Pipeline runs to completion → ``update_run_status('completed', ...)``
             commits the audit row.
          2. Success-path ``broadcast('completed', ...)`` raises ``RuntimeError``.
          3. Recovery probes ``get_run`` → SQLAlchemyError (probe failure).
          4. ``run_already_terminal`` stays ``False`` (probe couldn't determine).
          5. Recovery falls through to ``update_run_status('failed', ...)`` against
             a row whose true status is ``completed``.
          6. Real ``LEGAL_RUN_TRANSITIONS`` mock raises
             ``IllegalRunTransitionError`` (matching production
             ``SessionService.update_run_status``'s transition guard).
          7. Production recovery's narrow ``except (SQLAlchemyError, OSError)``
             does NOT catch the ``ValueError`` subclass; it propagates and
             shadows the original ``RuntimeError``.

        Correct behaviour: the ``RuntimeError("simulated SSE crash")`` is the
        operationally-relevant signal and MUST be the surfacing exception —
        the ``IllegalRunTransitionError`` is an artefact of a recovery attempt
        that should never have been made against an already-terminal row.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch_cls.return_value = self._make_completed_orchestrator(RunStatus.COMPLETED)

        # Stateful mock that mirrors SessionService.update_run_status's
        # transition validation (sessions/service.py:665-667). Driven by the
        # real ``LEGAL_RUN_TRANSITIONS`` table so the test stays in lockstep
        # with the production validator without duplicating its terminal-set
        # closure.
        audit_row_status: dict[str, SessionRunStatus] = {"current": "pending"}

        def _legal_transitions_update(_run_id: Any, *, status: str, **__: Any) -> None:
            current = audit_row_status["current"]
            allowed = LEGAL_RUN_TRANSITIONS[current]
            if status not in allowed:
                raise IllegalRunTransitionError(current, status, allowed)
            audit_row_status["current"] = cast(SessionRunStatus, status)

        mock_session_service.update_run_status.side_effect = _legal_transitions_update

        # Probe fails — same as the sibling test.  Drives the recovery into
        # the fall-through branch where the residual data-loss occurs.
        mock_session_service.get_run.side_effect = SQLAlchemyError("simulated DB hiccup")

        self._wrap_broadcaster_to_raise(
            service,
            on_event_type="completed",
            exc=RuntimeError("simulated SSE crash"),
        )

        run_id = str(uuid4())
        # Closes the elspeth-879f6de6bd gap: the IllegalRunTransitionError
        # raised by the fall-through update_run_status('failed', ...) against
        # an already-terminal row is now caught narrowly in
        # ``ExecutionServiceImpl._run_pipeline``'s BaseException recovery
        # (branch 3 — see the prelude comment block at that site).  The
        # narrow catch promotes the run into the audit-primacy stance via
        # ``irte.current_status`` as in-band proof of terminality, so the
        # original SSE-crash RuntimeError surfaces and the ``failed`` SSE
        # broadcast is suppressed.
        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            patch(
                "elspeth.web.execution.service.load_run_accounting_from_db",
                return_value=_run_accounting_for_status(RunStatus.COMPLETED),
            ),
            pytest.raises(RuntimeError, match="simulated SSE crash"),
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # Audit-primacy completion: branch 3 must NOT have broadcast a
        # ``failed`` SSE event (the audit row is in a real terminal status
        # — broadcasting ``failed`` would contradict it).  The branch-3 log
        # ``post_exception_recovery_aborted_run_terminal`` is the
        # SRE-discoverable resolution of the probe-failure ambiguity.
        recovery_aborted_logs = [
            c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_exception_recovery_aborted_run_terminal"
        ]
        assert len(recovery_aborted_logs) == 1, f"branch-3 recovery slog must fire exactly once; got {len(recovery_aborted_logs)}"
        # The probe-failure slog still fires upstream — the IRTE catch is
        # the *resolution*, not a replacement for the probe-failure record.
        probe_failed_logs = [c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_exception_run_state_probe_failed"]
        assert len(probe_failed_logs) == 1, f"probe-failure slog must still fire upstream of the IRTE catch; got {len(probe_failed_logs)}"
        # Three update_run_status attempts: running, completed, failed (the
        # third one raises IRTE which is caught).  The attempt is recorded
        # on the mock even though the side_effect raised.
        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "completed", "failed"], (
            f"branch-3 recovery path must attempt the failed update (caught by IRTE); got {statuses}"
        )

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_post_completion_get_run_probe_value_error_propagates(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """ValueError from the post-exception ``get_run`` probe must propagate,
        not be absorbed.

        Audit-primacy contract (CLAUDE.md tier model): ``get_run`` can raise
        ``ValueError`` only via Tier 1 audit-data corruption — "Run not found"
        (the row vanished mid-run), malformed UUID columns, or non-UTC
        ``started_at`` / ``finished_at``.  All three are Tier 1 invariant
        violations that MUST crash immediately.

        Pre-fix behaviour absorbed ValueError in the probe catch alongside
        ``SQLAlchemyError``/``OSError`` and fell through to the best-effort
        ``update_run_status`` recovery, which would re-encounter the same
        corruption.  The narrow catch (mirroring the sibling pattern at the
        ``update_run_status`` recovery — commits b8ba2214/127417cb) keeps
        Tier 1 corruption visible at the call site.

        This test pins that contract so future re-widening of the catch is
        caught at review time.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch_cls.return_value = self._make_completed_orchestrator(RunStatus.COMPLETED)

        # Probe raises ValueError — the canonical Tier 1 signal from get_run
        # ("Run not found", malformed UUID, or non-UTC datetime).  All three
        # share this exception class and must surface, not be absorbed.
        mock_session_service.get_run.side_effect = ValueError("Run not found: simulated Tier 1 corruption")

        self._wrap_broadcaster_to_raise(
            service,
            on_event_type="completed",
            exc=RuntimeError("simulated SSE crash"),
        )

        run_id = str(uuid4())
        # The visible exception MUST be the probe's ValueError (Tier 1
        # corruption surfaces as itself).  The original RuntimeError is
        # preserved on ``__context__`` by Python's normal exception chaining.
        with (
            patch("elspeth.web.execution.service.slog") as mock_slog,
            patch(
                "elspeth.web.execution.service.load_run_accounting_from_db",
                return_value=_run_accounting_for_status(RunStatus.COMPLETED),
            ),
            pytest.raises(ValueError, match="Run not found") as exc_info,
        ):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        # Exception chain pins the original cause — the probe ValueError
        # is raised while handling the RuntimeError, so __context__ MUST
        # carry the original SSE crash.  Without this, debugging the
        # post-completion exception gets harder, not easier.
        assert isinstance(exc_info.value.__context__, RuntimeError)
        assert "simulated SSE crash" in str(exc_info.value.__context__)

        # Probe-failure slog MUST NOT fire — the ValueError exits the try
        # block via propagation, not through the narrow except.  If a future
        # change re-widens the catch to include ValueError, this assertion
        # will fail and surface the regression.
        probe_failed_logs = [c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_exception_run_state_probe_failed"]
        assert probe_failed_logs == [], (
            "ValueError from get_run must propagate (Tier 1 corruption); "
            f"probe-failed slog should not fire, got {len(probe_failed_logs)} call(s)."
        )

        # The fall-through update_run_status("failed", ...) MUST NOT have
        # been called — control left the BaseException handler before the
        # recovery branch.  Only the "running" and "completed" updates from
        # the happy path are present.
        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "completed"], f"ValueError-propagation path must skip the recovery update_run_status; got {statuses}"

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_running_state_exception_still_records_failed(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_load: MagicMock,
        mock_instantiate: MagicMock,
        mock_graph_cls: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Negative control: when the orchestrator raises BEFORE the terminal
        transition (run is still ``running``), the existing happy-path
        recovery must still land — the new guard MUST NOT short-circuit
        non-terminal states.
        """
        mock_bundle = MagicMock()
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_load.return_value = _mock_pipeline_settings()
        mock_orch = MagicMock()
        mock_orch.run.side_effect = RuntimeError("orchestrator blew up mid-run")
        mock_orch_cls.return_value = mock_orch

        # DB still holds "running" because we never reached the terminal
        # transition.
        mock_session_service.get_run.return_value = MagicMock(status="running")

        run_id = str(uuid4())
        with patch("elspeth.web.execution.service.slog") as mock_slog, pytest.raises(RuntimeError, match="orchestrator blew up"):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())

        statuses = [c.kwargs.get("status") for c in mock_session_service.update_run_status.call_args_list]
        assert statuses == ["running", "failed"], f"Non-terminal path must still record failure; got {statuses}"

        post_terminal_logs = [
            c for c in mock_slog.error.call_args_list if c.args and c.args[0] == "post_terminal_exception_in_run_pipeline"
        ]
        assert post_terminal_logs == [], "Guard must NOT fire for non-terminal current status"


# ── Liveness Registry ─────────────────────────────────────────────────


class TestGetLiveRunIds:
    """Tests for get_live_run_ids — used by periodic orphan cleanup."""

    def test_returns_empty_when_no_active_runs(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """No runs registered → empty frozenset."""
        assert service.get_live_run_ids() == frozenset()

    def test_returns_registered_run_ids(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """Manually registered shutdown events appear in live run IDs."""
        event = threading.Event()
        with service._shutdown_events_lock:
            service._shutdown_events["run-abc"] = event
            service._shutdown_events["run-def"] = event
        assert service.get_live_run_ids() == frozenset({"run-abc", "run-def"})

    def test_includes_signalled_events_until_worker_exits(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """Signalled runs stay live until _run_pipeline() removes them.

        A set shutdown event means cancellation was requested, not that the
        worker thread has finished its GracefulShutdownError unwinding.
        Periodic orphan cleanup must keep excluding the run until the
        worker's finally block removes the registry entry.
        """
        live_event = threading.Event()
        signalled_event = threading.Event()
        signalled_event.set()
        with service._shutdown_events_lock:
            service._shutdown_events["run-live"] = live_event
            service._shutdown_events["run-signalled"] = signalled_event
        assert service.get_live_run_ids() == frozenset({"run-live", "run-signalled"})

    def test_returns_snapshot_not_live_reference(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """Returned frozenset is a snapshot — later changes don't affect it."""
        event = threading.Event()
        with service._shutdown_events_lock:
            service._shutdown_events["run-1"] = event
        snapshot = service.get_live_run_ids()
        with service._shutdown_events_lock:
            service._shutdown_events["run-2"] = event
        # Snapshot should not include run-2
        assert snapshot == frozenset({"run-1"})


# ── Blob Ref Pre-Validation ───────────────────────────────────────────


class TestBlobRefPreValidation:
    """Malformed blob_ref must raise BEFORE create_run() to avoid
    orphaning a pending run that blocks future executions."""

    @pytest.mark.asyncio
    async def test_malformed_blob_ref_raises_before_run_creation(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """A non-UUID blob_ref raises typed validation before create_run()
        is called, so no pending run is orphaned."""
        from elspeth.web.execution.errors import MalformedBlobRefError

        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": "not-a-uuid"},
            "on_validation_failure": "quarantine",
        }

        blob_service = MagicMock()
        cast(Any, service)._blob_service = blob_service

        with pytest.raises(MalformedBlobRefError):
            await service.execute(session_id=uuid4())

        # The critical invariant: create_run() was never called,
        # so no stale pending run exists.
        mock_session_service.create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_blob_ref_still_links_correctly(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Valid UUID blob_ref is parsed early and passed to link_blob_to_run."""
        session_id = uuid4()
        run_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"
        mock_session_service.create_run.return_value = MagicMock(id=run_id)

        blob_service = MagicMock()
        blob_service.link_blob_to_run = AsyncMock()
        # get_blob returns a record matching the executing session
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        cast(Any, service)._blob_service = blob_service

        # path must equal blob.storage_path to satisfy the Tier 1 read
        # guard for blob-backed sources (elspeth-07089fbaa3).
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref, "path": canonical_path},
            "on_validation_failure": "quarantine",
        }

        with patch.object(service, "_run_pipeline"):
            await service.execute(session_id=session_id)

        blob_service.link_blob_to_run.assert_called_once_with(
            blob_id=UUID(blob_ref),
            run_id=run_id,
            direction="input",
        )


# ── Blob Ownership (Cross-Session IDOR) ──────────────────────────────


class TestBlobOwnership:
    """P2 defense-in-depth: blob_ref must belong to the executing session.

    Without this, a crafted composition state could reference another
    session's blob path — the shared-root path allowlist would pass it.
    """

    @pytest.mark.asyncio
    async def test_cross_session_blob_ref_rejected(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Cross-session blob_ref raises ``BlobNotFoundError`` (IDOR collapse).

        The exception type is load-bearing: the route handler relies
        on cross-session and nonexistent blobs BOTH surfacing as
        ``BlobNotFoundError`` so they produce byte-identical 404
        responses.  Earlier this branch raised ``ValueError`` with a
        "does not belong to session" message — a distinguishable
        body AND a distinguishable status (404 vs the 500 that an
        uncaught ``BlobNotFoundError`` produced for the nonexistent
        case).  Do not revert to ``ValueError`` or add a specialised
        subclass without also updating the route handler in
        lockstep.
        """
        from elspeth.web.blobs.protocol import BlobNotFoundError

        executing_session_id = uuid4()
        other_session_id = uuid4()
        blob_ref = str(uuid4())

        blob_service = MagicMock()
        # Blob belongs to other_session_id, not executing_session_id
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=other_session_id))
        cast(Any, service)._blob_service = blob_service

        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(BlobNotFoundError):
            await service.execute(session_id=executing_session_id)

        # Critical: create_run was never called (rejected before run creation)
        mock_session_service.create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_session_blob_ref_accepted(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Blob belonging to the same session passes ownership check."""
        session_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        blob_service.link_blob_to_run = AsyncMock()
        cast(Any, service)._blob_service = blob_service

        # path must equal blob.storage_path to satisfy the Tier 1 read
        # guard for blob-backed sources (elspeth-07089fbaa3).
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref, "path": canonical_path},
            "on_validation_failure": "quarantine",
        }

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=session_id)
        assert isinstance(run_id, UUID)


# ── Blob Source Path Read Guard (Tier 1, elspeth-07089fbaa3) ─────────


class TestBlobSourcePathReadGuard:
    """Closes elspeth-07089fbaa3 (runtime read guard).

    The composer's write-side defenses make wrong-shape blob source paths
    impossible to persist going forward, but the audit-integrity contract
    also requires that runtime crash informatively if a previously-
    persisted state row carries a path that disagrees with the canonical
    ``BlobRecord.storage_path``.  Per CLAUDE.md "no defensive programming",
    the runtime must not silently coerce or fall back to ``FileNotFoundError``.

    Bug-verification protocol (cf.
    ``tests/integration/pipeline/test_composer_runtime_agreement.py``
    module docstring lines 76-88): manually revert the
    ``if stored_path != canonical_path: raise BlobSourcePathMismatchError``
    block in ``ExecutionServiceImpl._execute_locked`` and confirm the
    mismatch test below fails with the canonical-path branch silently
    accepting the divergent stored path.  Then restore.
    """

    @pytest.mark.asyncio
    async def test_diverging_stored_path_raises_structured_error(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Tier 1: stored path != blob.storage_path crashes at execute time.

        Reproduces the captured staging defect (session
        588b94c8-919c-43ab-ae2c-8a3033de8109): the persisted
        ``source.options.path`` does not match the canonical
        ``BlobRecord.storage_path``.  The captured shape was
        ``data/blobs/<bid>/<filename>`` (rejected first by the source
        path allowlist after the legacy resolver was removed); this test
        exercises the divergence case where the path is allowlist-valid
        but still not the canonical one (e.g. a stale absolute path
        pointing at a different file under ``data_dir/blobs/``).  The
        guard fires before the run record is created so the session is
        not poisoned with a pending run.
        """
        from elspeth.web.execution.errors import BlobSourcePathMismatchError

        session_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"
        # Allowlist-valid (under /tmp/data/blobs/) but not equal to
        # canonical_path — the divergence the read guard targets.
        diverging_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_OTHER.csv"

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        blob_service.link_blob_to_run = AsyncMock()
        cast(Any, service)._blob_service = blob_service

        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": blob_ref, "path": diverging_path},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(BlobSourcePathMismatchError) as exc_info:
            await service.execute(session_id=session_id)

        assert exc_info.value.stored_path == diverging_path
        assert exc_info.value.canonical_path == canonical_path
        assert exc_info.value.blob_id == blob_ref
        assert "elspeth-07089fbaa3" in str(exc_info.value)

        # Critical: create_run was never called — the session is not
        # poisoned with a pending run that the operator must clean up.
        mock_session_service.create_run.assert_not_called()
        # link_blob_to_run was never called either — guard fires before
        # any side effects.
        blob_service.link_blob_to_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_stored_path_raises_structured_error(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Tier 1: stored path is None for a blob-backed source crashes.

        A composition state with ``blob_ref`` set but no ``path`` is
        structurally invalid — the blob binding requires the canonical
        path to be present.  This branch protects against a regression
        where a future composer-side bug omits the path entirely while
        still persisting the blob_ref.
        """
        from elspeth.web.execution.errors import BlobSourcePathMismatchError

        session_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_input.csv"

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        cast(Any, service)._blob_service = blob_service

        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            # No path key at all
            "options": {"blob_ref": blob_ref},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(BlobSourcePathMismatchError) as exc_info:
            await service.execute(session_id=session_id)

        assert exc_info.value.stored_path is None
        assert exc_info.value.canonical_path == canonical_path

    @pytest.mark.asyncio
    async def test_named_blob_source_path_mismatch_raises_structured_error(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Every named source blob_ref gets the same ownership/path guard."""
        from elspeth.web.execution.errors import BlobSourcePathMismatchError

        session_id = uuid4()
        blob_ref = str(uuid4())
        canonical_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_orders.csv"
        diverging_path = f"/tmp/data/blobs/{session_id}/{blob_ref}_OTHER.csv"

        blob_service = MagicMock()
        blob_service.get_blob = AsyncMock(return_value=MagicMock(session_id=session_id, storage_path=canonical_path))
        blob_service.link_blob_to_run = AsyncMock()
        cast(Any, service)._blob_service = blob_service

        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.sources = {
            "orders": {
                "plugin": "csv",
                "on_success": "orders_rows",
                "options": {"blob_ref": blob_ref, "path": diverging_path},
                "on_validation_failure": "quarantine",
            }
        }

        with pytest.raises(BlobSourcePathMismatchError) as exc_info:
            await service.execute(session_id=session_id)

        assert exc_info.value.stored_path == diverging_path
        assert exc_info.value.canonical_path == canonical_path
        mock_session_service.create_run.assert_not_called()
        blob_service.link_blob_to_run.assert_not_called()


# ── One Active Run (B6) ───────────────────────────────────────────────


class TestOneActiveRun:
    @pytest.mark.asyncio
    async def test_second_execute_raises_run_already_active(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """B6: Only one pending/running run per session."""
        session_id = uuid4()
        mock_session_service.get_active_run.return_value = MagicMock(status="running")

        with pytest.raises(RunAlreadyActiveError):
            await service.execute(session_id=session_id)

    @pytest.mark.asyncio
    async def test_execute_after_completed_run_succeeds(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """After a run completes, a new one can start."""
        mock_session_service.get_active_run.return_value = None
        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)


# ── EventBus Bridge ───────────────────────────────────────────────────


class TestEventBusBridge:
    """Verify that ProgressEvent from the Orchestrator's EventBus
    is translated to RunEvent and broadcast via the ProgressBroadcaster."""

    def test_progress_event_translated_to_run_event(self, service: ExecutionServiceImpl) -> None:
        """_to_run_event maps ProgressEvent fields to RunEvent.data.

        elspeth-5069612f3c — assert the routed split (MOVE / DIVERT) is
        plumbed verbatim through the translator. Pre-fix the engine emitter
        folded ``rows_routed_success`` into ``rows_succeeded`` and dropped
        ``rows_routed_failure`` entirely; the wire payload then lacked the
        fields. This test guards against regression to that shape.
        """
        from elspeth.contracts.cli import ProgressEvent
        from elspeth.web.execution.schemas import ProgressData

        progress = ProgressEvent(
            rows_processed=100,
            rows_succeeded=92,
            rows_failed=5,
            rows_quarantined=3,
            rows_routed_success=7,
            rows_routed_failure=2,
            elapsed_seconds=10.5,
        )
        run_id = "run-123"
        run_event = service._to_run_event(run_id, progress)

        assert run_event.event_type == "progress"
        assert isinstance(run_event.data, ProgressData)
        # S-8: assert every counter passes through with its real producer
        # value.  Non-zero values in every slot guard against a future
        # producer that hardcodes any single counter to 0 (which Pydantic
        # cannot detect — making the upstream ProgressEvent require all six
        # is the structural defense, this assertion is the test surface).
        assert run_event.data.source_rows_processed == 100
        assert run_event.data.tokens_succeeded == 92
        assert run_event.data.tokens_failed == 5
        assert run_event.data.tokens_quarantined == 3
        assert run_event.data.tokens_routed_success == 7
        assert run_event.data.tokens_routed_failure == 2
        assert run_event.run_id == "run-123"

    def test_progress_broadcast_closed_loop_records_drop_telemetry(self, service: ExecutionServiceImpl) -> None:
        """Loop-closed progress drops are operational telemetry, not slog-only."""
        from elspeth.contracts.cli import ProgressEvent

        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            broadcaster.subscribe("run-123")
            loop.close()
            service._broadcaster = broadcaster

            service._broadcast_progress_event(
                "run-123",
                ProgressEvent(
                    rows_processed=100,
                    rows_succeeded=92,
                    rows_failed=5,
                    rows_quarantined=3,
                    rows_routed_success=7,
                    rows_routed_failure=2,
                    elapsed_seconds=10.5,
                ),
            )
        finally:
            if not loop.is_closed():
                loop.close()

        assert observed_value(service._telemetry.progress_broadcast_dropped_total) == 1


# ── B10: _call_async() Bridge Tests ──────────────────────────────────


class TestB8AsyncBridging:
    """B8/C1 fix: _call_async() bridges sync thread to async event loop.

    These tests need the REAL _call_async (not the test fixture's mock),
    so they construct a fresh service with a mock loop whose
    run_coroutine_threadsafe is controlled.
    """

    def test_call_async_returns_coroutine_result(
        self,
        broadcaster: ProgressBroadcaster,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """_call_async() schedules coroutine and returns its result."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        svc = ExecutionServiceImpl(
            loop=mock_loop,
            broadcaster=broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )
        mock_future = MagicMock()
        mock_future.result.return_value = "test_result"

        async def dummy_coro() -> str:
            return "test_result"

        coro = dummy_coro()
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = svc._call_async(coro)
        coro.close()
        assert result == "test_result"
        mock_future.result.assert_called_once_with(timeout=30.0)

    def test_call_async_propagates_coroutine_exception(
        self,
        broadcaster: ProgressBroadcaster,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """If the coroutine raises, _call_async re-raises from future.result()."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        svc = ExecutionServiceImpl(
            loop=mock_loop,
            broadcaster=broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )
        mock_future = MagicMock()
        mock_future.result.side_effect = ValueError("db error")

        async def failing_coro() -> None:
            raise ValueError("db error")

        coro = failing_coro()
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future), pytest.raises(ValueError, match="db error"):
            svc._call_async(coro)
        coro.close()

    def test_call_async_raises_timeout_error(
        self,
        broadcaster: ProgressBroadcaster,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """R6 fix: _call_async raises TimeoutError after 30s, preventing deadlock."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        svc = ExecutionServiceImpl(
            loop=mock_loop,
            broadcaster=broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()

        async def hanging_coro() -> None:
            pass

        coro = hanging_coro()
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future), pytest.raises(concurrent.futures.TimeoutError):
            svc._call_async(coro)
        coro.close()


class TestAsyncShutdown:
    """Shutdown must keep the event loop available for worker cleanup."""

    @pytest.mark.asyncio
    async def test_shutdown_keeps_loop_available_for_worker_cleanup(
        self,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """Regression: draining the executor must not strand worker _call_async calls."""
        loop = asyncio.get_running_loop()
        svc = ExecutionServiceImpl(
            loop=loop,
            broadcaster=ProgressBroadcaster(loop),
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )

        run_id = str(uuid4())
        shutdown_event = threading.Event()
        with svc._shutdown_events_lock:
            svc._shutdown_events[run_id] = shutdown_event

        cleanup_applied = asyncio.Event()

        async def update_run_status(*args: Any, **kwargs: Any) -> None:
            cleanup_applied.set()

        mock_session_service.update_run_status = AsyncMock(side_effect=update_run_status)

        def short_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=1.0)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise

        cast(Any, svc)._call_async = short_call_async

        worker_done = threading.Event()
        worker_errors: list[str] = []

        def worker() -> None:
            shutdown_event.wait()
            try:
                svc._call_async(mock_session_service.update_run_status(uuid4(), status="cancelled"))
            except BaseException as exc:
                worker_errors.append(type(exc).__name__)
            finally:
                worker_done.set()

        svc._executor.submit(worker)

        await svc.shutdown()

        assert worker_done.is_set()
        assert worker_errors == []
        assert cleanup_applied.is_set()


# ── W15: Running Status Failure Path ─────────────────────────────────


class TestRunningStatusFailure:
    """W15: What happens when the initial status update to 'running' fails."""

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_running_status_failure_marks_run_failed(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If update_run_status('running') fails, the except BaseException
        block attempts to set 'failed'. Run stays 'pending' if both fail."""
        # Make the first _call_async raise (simulating event loop issues)
        original_call_async = service._call_async
        call_count = 0

        def failing_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call = update to "running"
                coro.close()
                raise ConnectionError("DB connection lost")
            return original_call_async(coro)

        cast(Any, service)._call_async = failing_call_async

        with pytest.raises(ConnectionError):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # The except block tried to set "failed" via the second _call_async call
        assert call_count >= 2


# ── IDOR Protection: verify_run_ownership ─────────────────────────────


class TestVerifyRunOwnership:
    """IDOR protection — verify_run_ownership checks user_id + auth_provider.

    Criticality 9/10: This is the gate between "attacker can watch other
    users' pipeline progress via WebSocket" and "access denied."
    """

    @pytest.fixture
    def idor_service(
        self,
        mock_loop: MagicMock,
        broadcaster: ProgressBroadcaster,
    ) -> tuple[ExecutionServiceImpl, MagicMock]:
        """ExecutionServiceImpl with controllable session service."""
        session_svc = MagicMock()
        settings = MagicMock()
        settings.auth_provider = "local"
        settings.get_landscape_url.return_value = "sqlite:///test.db"
        settings.get_payload_store_path.return_value = Path("/tmp/test")

        svc = ExecutionServiceImpl(
            loop=mock_loop,
            broadcaster=broadcaster,
            settings=settings,
            session_service=session_svc,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )
        return svc, session_svc

    @pytest.mark.asyncio
    async def test_owner_match_returns_true(self, idor_service) -> None:
        """Correct user + correct provider → access granted."""
        svc, session_svc = idor_service
        session_id = uuid4()
        run = MagicMock(session_id=session_id)
        session = MagicMock(user_id="alice", auth_provider_type="local")
        session_svc.get_run = AsyncMock(return_value=run)
        session_svc.get_session = AsyncMock(return_value=session)

        user = MagicMock(user_id="alice")
        assert await svc.verify_run_ownership(user, str(uuid4())) is True

    @pytest.mark.asyncio
    async def test_wrong_user_returns_false(self, idor_service) -> None:
        """Wrong user_id → access denied."""
        svc, session_svc = idor_service
        run = MagicMock(session_id=uuid4())
        session = MagicMock(user_id="alice", auth_provider_type="local")
        session_svc.get_run = AsyncMock(return_value=run)
        session_svc.get_session = AsyncMock(return_value=session)

        user = MagicMock(user_id="eve")
        assert await svc.verify_run_ownership(user, str(uuid4())) is False

    @pytest.mark.asyncio
    async def test_cross_provider_returns_false(self, idor_service) -> None:
        """Same user_id but different auth provider → access denied.

        This prevents "alice" in local auth from accessing runs belonging
        to "alice" in OIDC. Cross-provider user_id collision is the
        non-obvious IDOR vector.
        """
        svc, session_svc = idor_service
        run = MagicMock(session_id=uuid4())
        # Session was created under OIDC, but server is now configured for "local"
        session = MagicMock(user_id="alice", auth_provider_type="oidc")
        session_svc.get_run = AsyncMock(return_value=run)
        session_svc.get_session = AsyncMock(return_value=session)

        user = MagicMock(user_id="alice")
        assert await svc.verify_run_ownership(user, str(uuid4())) is False

    @pytest.mark.asyncio
    async def test_nonexistent_run_raises(self, idor_service) -> None:
        """Run not found → ValueError propagates (caller handles)."""
        svc, session_svc = idor_service
        session_svc.get_run = AsyncMock(side_effect=ValueError("Run not found"))

        user = MagicMock(user_id="alice")
        with pytest.raises(ValueError, match="Run not found"):
            await svc.verify_run_ownership(user, str(uuid4()))

    @pytest.mark.asyncio
    async def test_str_vs_non_str_user_id_rejects(self, idor_service) -> None:
        """Regression: if session.user_id were stored as UUID, str comparison must reject."""
        svc, session_svc = idor_service
        run = MagicMock(session_id=uuid4())
        user_uuid = uuid4()
        session = MagicMock(user_id=user_uuid, auth_provider_type="local")
        session_svc.get_run = AsyncMock(return_value=run)
        session_svc.get_session = AsyncMock(return_value=session)

        user = MagicMock(user_id=str(user_uuid))
        assert await svc.verify_run_ownership(user, str(uuid4())) is False


# ── Sink Path Restriction ─────────────────────────────────────────────


class TestSinkPathRestriction:
    """P1 security fix: Sink output paths must be confined to allowed directories.

    Without this, a client can set sink options.path to an arbitrary absolute
    or ../ path and /execute will write there — turning the executor into an
    arbitrary file-write surface.
    """

    @pytest.mark.asyncio
    async def test_sink_path_outside_allowed_dirs_raises(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Sink with path pointing outside data_dir/outputs must be rejected."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.outputs = [
            {
                "name": "primary",
                "plugin": "csv",
                "options": {"path": "/etc/cron.d/backdoor.csv"},
                "on_write_failure": "discard",
            }
        ]
        state.nodes = None
        state.edges = None

        from elspeth.web.execution.errors import PathAllowlistViolationError

        with pytest.raises(PathAllowlistViolationError, match="resolves outside allowed output directories"):
            await service.execute(session_id=uuid4())

    @pytest.mark.asyncio
    async def test_sink_path_traversal_rejected(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Sink with ../ traversal in path must be rejected."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.outputs = [
            {
                "name": "results",
                "plugin": "json",
                "options": {"path": "/tmp/elspeth_data/outputs/../../etc/passwd"},
                "on_write_failure": "discard",
            }
        ]
        state.nodes = None
        state.edges = None

        from elspeth.web.execution.errors import PathAllowlistViolationError

        with pytest.raises(PathAllowlistViolationError, match="resolves outside allowed output directories"):
            await service.execute(session_id=uuid4())

    @pytest.mark.asyncio
    async def test_sink_path_under_outputs_accepted(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Sink with path under data_dir/outputs is allowed."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.outputs = [
            {
                "name": "primary",
                "plugin": "csv",
                "options": {"path": "/tmp/elspeth_data/outputs/result.csv"},
                "on_write_failure": "discard",
            }
        ]
        state.nodes = None
        state.edges = None

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_sink_without_path_option_passes(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Sink with no path/file options (e.g. database sink) passes check."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.outputs = [
            {
                "name": "db_sink",
                "plugin": "database",
                "options": {"connection_string": "sqlite:///out.db"},
                "on_write_failure": "discard",
            }
        ]
        state.nodes = None
        state.edges = None

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)


# ── Transform Framing Restriction ─────────────────────────────────────


class TestExecuteSemanticContractViolation:
    """Execution must reject transform pairings that violate semantic contracts.

    Replaces the legacy TestTransformFramingRestriction. The new
    SemanticContractViolationError carries structured ``entries`` and
    ``contracts`` records; the regex assertions still anchor on
    ``line_explode``/``Semantic contract`` because the diagnostic now
    names the consumer plugin and the contract code in the message,
    not the option that the operator must edit (``text_separator``).
    """

    @staticmethod
    def _set_web_scrape_line_explode_state(
        mock_session_service: MagicMock,
        *,
        scrape_options: dict[str, Any] | None = None,
    ) -> None:
        state = mock_session_service.get_current_state.return_value
        web_scrape_options = {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "content_fingerprint",
            "format": "text",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "pipeline@example.com",
                "scraping_reason": "test scrape",
                "allowed_hosts": "public_only",
            },
        }
        web_scrape_options.update(scrape_options or {})
        state.source = {
            "plugin": "text",
            "on_success": "scrape_in",
            "options": {
                "path": "blobs/urls.txt",
                "column": "url",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            "on_validation_failure": "discard",
        }
        state.nodes = [
            {
                "id": "scrape_page",
                "node_type": "transform",
                "plugin": "web_scrape",
                "input": "scrape_in",
                "on_success": "explode_in",
                "on_error": "discard",
                "options": web_scrape_options,
            },
            {
                "id": "split_lines",
                "node_type": "transform",
                "plugin": "line_explode",
                "input": "explode_in",
                "on_success": "results",
                "on_error": "discard",
                "options": {
                    "schema": {
                        "mode": "flexible",
                        "fields": [
                            "url: str",
                            "content: str",
                            "content_fingerprint: str",
                        ],
                    },
                    "required_input_fields": ["content"],
                    "source_field": "content",
                    "output_field": "line",
                    "include_index": True,
                    "index_field": "line_index",
                },
            },
        ]
        state.edges = None
        state.outputs = [
            {
                "name": "results",
                "plugin": "json",
                "options": {"path": "outputs/lines.json", "format": "json"},
                "on_write_failure": "discard",
            }
        ]

    @pytest.mark.asyncio
    async def test_execute_rejects_compact_web_scrape_text_before_creating_run(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_web_scrape_line_explode_state(mock_session_service)

        # SemanticContractViolationError IS a ValueError, so legacy
        # ``except ValueError`` paths still catch it. New callers should
        # catch the specific type and read .entries/.contracts.
        with pytest.raises(ValueError, match="line_explode"):
            await service.execute(session_id=uuid4())

        mock_session_service.create_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_compact_text_raises_structured_exception(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Verify the structured payload — the whole point of the new exception.

        Frontend banners and MCP error renderers consume entries and
        contracts directly; falling back to ``str(exc)`` parsing would
        make this surface as fragile as the pre-Phase-4 string concat.
        """
        from elspeth.web.execution.errors import SemanticContractViolationError

        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_web_scrape_line_explode_state(mock_session_service)

        with pytest.raises(SemanticContractViolationError) as excinfo:
            await service.execute(session_id=uuid4())

        exc = excinfo.value
        assert len(exc.entries) >= 1
        assert any("Semantic contract" in e.message for e in exc.entries)
        assert any(c.outcome.value == "conflict" for c in exc.contracts)
        assert any(c.consumer_plugin == "line_explode" for c in exc.contracts)

    @pytest.mark.asyncio
    async def test_execute_allows_newline_framed_web_scrape_text(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_web_scrape_line_explode_state(
            mock_session_service,
            scrape_options={"text_separator": "\n"},
        )

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())

        assert isinstance(run_id, UUID)


# ── F-17 / F-21: Unresolved Interpretation Placeholder Gate ─────────────


class TestExecuteUnresolvedInterpretationPlaceholderGate:
    """``/execute`` must refuse to run an LLM transform whose prompt_template
    still carries ``{{interpretation:<term>}}`` placeholders (F-17 / F-21 —
    Phase 5b Task 5 follow-on).

    Operates under the operator-acknowledged assumption that 18a Task 0
    (empirical LLM gate ≥ 8/10 staging runs emit
    ``{{interpretation:<term>}}``) passes; this gate is the runtime-safety
    net catching cases where the LLM under-fires.
    """

    @staticmethod
    def _set_unresolved_placeholder_state(
        mock_session_service: MagicMock,
        *,
        term: str = "cool",
        node_id: str = "rate_node",
    ) -> None:
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "rate_in",
            "options": {"path": "blobs/rows.csv"},
            "on_validation_failure": "discard",
        }
        state.nodes = [
            {
                "id": node_id,
                "node_type": "transform",
                "plugin": "llm",
                "input": "rate_in",
                "on_success": "results",
                "on_error": "discard",
                "options": {
                    "prompt_template": f"Rate {{{{interpretation:{term}}}}} aspects.",
                    "model": "test-model",
                    # Pre-resolve the model-choice review so this fixture
                    # exercises ONLY the unresolved-vague-term /
                    # unresolved-prompt-template gates the test class
                    # targets. Without this, the auto-enumerated
                    # model-choice site shows up as a third pending
                    # interpretation and contaminates the gate's
                    # observed telemetry list.
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": f"model_choice_review:{node_id}",
                            "kind": "llm_model_choice",
                            "user_term": f"llm_model_choice:{node_id}",
                            "status": "resolved",
                            "draft": "test-model",
                            "event_id": "model-choice-accepted",
                            "accepted_value": "test-model",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("test-model"),
                        }
                    ],
                },
            }
        ]
        state.edges = None
        state.outputs = [
            {
                "name": "results",
                "plugin": "json",
                "options": {"path": "outputs/scored.json", "format": "json"},
                "on_write_failure": "discard",
            }
        ]

    @staticmethod
    def _set_structured_pending_interpretation_state(
        mock_session_service: MagicMock,
        *,
        term: str = "cool",
        node_id: str = "rate_node",
    ) -> None:
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "rate_in",
            "options": {"path": "blobs/rows.csv"},
            "on_validation_failure": "discard",
        }
        state.nodes = [
            {
                "id": node_id,
                "node_type": "transform",
                "plugin": "llm",
                "input": "rate_in",
                "on_success": "results",
                "on_error": "discard",
                "options": {
                    "prompt_template": "Rate pending interpretation aspects.",
                    "model": "test-model",
                    PROMPT_TEMPLATE_PARTS_KEY: [
                        {"kind": "text", "text": "Rate "},
                        {"kind": "interpretation_ref", "requirement_id": term},
                        {"kind": "text", "text": " aspects."},
                    ],
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": term,
                            "kind": "vague_term",
                            "user_term": term,
                            "status": "pending",
                            "draft": "visually appealing",
                            "event_id": "event-1",
                            "accepted_value": None,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": None,
                        },
                        # Pre-resolve the model-choice review so this
                        # fixture exercises only the structured pending
                        # vague_term scenario.
                        {
                            "id": f"model_choice_review:{node_id}",
                            "kind": "llm_model_choice",
                            "user_term": f"llm_model_choice:{node_id}",
                            "status": "resolved",
                            "draft": "test-model",
                            "event_id": "model-choice-accepted",
                            "accepted_value": "test-model",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("test-model"),
                        },
                    ],
                },
            }
        ]
        state.edges = None
        state.outputs = [
            {
                "name": "results",
                "plugin": "json",
                "options": {"path": "outputs/scored.json", "format": "json"},
                "on_write_failure": "discard",
            }
        ]

    @pytest.mark.asyncio
    async def test_execute_rejects_unresolved_placeholder_before_creating_run(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """F-17: an unresolved placeholder blocks execution and raises a typed error.

        The detector runs AFTER semantic-contract validation and BEFORE
        path-allowlist / YAML generation, so the gate fires before any
        ``Run`` row is created in the sessions DB.
        """
        from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError

        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_unresolved_placeholder_state(mock_session_service)

        with pytest.raises(UnresolvedInterpretationPlaceholderError) as excinfo:
            await service.execute(session_id=uuid4())

        # The typed payload carries (node_id, term) — no prompt_template.
        assert excinfo.value.placeholders == (("rate_node", "cool"),)

        # The actionable message names both the term and the node so the
        # frontend banner / MCP error renderer can echo it directly.
        assert "{{interpretation:cool}}" in str(excinfo.value)
        assert "rate_node" in str(excinfo.value)

        # No Run was created (fail-fast before run record persistence).
        mock_session_service.create_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_rejects_structured_pending_interpretation_before_creating_run(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError

        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_structured_pending_interpretation_state(mock_session_service)

        with patch.object(service, "_run_pipeline"), pytest.raises(UnresolvedInterpretationPlaceholderError) as excinfo:
            await service.execute(session_id=uuid4())

        assert excinfo.value.placeholders == (("rate_node", "cool"),)
        mock_session_service.create_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_emits_telemetry_per_unresolved_site(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """F-21: each unresolved interpretation site emits one counter increment.

        Attributes MUST identify kind and component without including the
        prompt_template value (which may carry user-supplied content —
        operational telemetry must be PII-clean).
        """
        from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError
        from elspeth.web.sessions.telemetry import _FakeCounter

        mock_settings.data_dir = "/tmp/elspeth_data"
        self._set_unresolved_placeholder_state(mock_session_service)

        with pytest.raises(UnresolvedInterpretationPlaceholderError):
            await service.execute(session_id=uuid4())

        counter = service._telemetry.interpretation_placeholder_unresolved_at_runtime_total
        # Test fixture uses fake counters — type-narrow to access ``calls``.
        assert isinstance(counter, _FakeCounter)
        assert len(counter.calls) == 2
        observed = []
        for amount, attrs, _context in counter.calls:
            assert amount == 1
            observed.append(attrs)
        assert observed == [
            {
                "component_id": "rate_node",
                "component_type": "transform",
                "kind": "vague_term",
            },
            {
                "component_id": "rate_node",
                "component_type": "transform",
                "kind": "llm_prompt_template",
            },
        ]
        # Explicit negative assertion: prompt/user-authored text must
        # never appear in telemetry attributes.
        for attrs in observed:
            assert attrs is not None
            assert "prompt_template" not in attrs
            assert "user_term" not in attrs
            assert "cool" not in attrs.values()

    @pytest.mark.asyncio
    async def test_execute_passes_when_placeholder_resolved(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """An LLM transform whose prompt_template has no placeholder runs normally.

        Negative-space test: confirms the gate does not fire spuriously
        when the compose loop did its job and the placeholder was
        replaced by a concrete term via the interpretation_events
        resolve flow.
        """
        from elspeth.web.sessions.telemetry import _FakeCounter

        mock_settings.data_dir = "/tmp/elspeth_data"
        prompt = "Rate visually-appealing aspects."
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "rate_in",
            "options": {"path": "blobs/rows.csv"},
            "on_validation_failure": "discard",
        }
        state.nodes = [
            {
                "id": "rate_node",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rate_in",
                "on_success": "results",
                "on_error": "discard",
                "options": {
                    # Placeholder resolved — no ``{{interpretation:…}}`` text.
                    "prompt_template": prompt,
                    "model": "test-model",
                    "resolved_prompt_template_hash": stable_hash(prompt),
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        {
                            "id": "prompt-template-review",
                            "kind": "llm_prompt_template",
                            "user_term": "rating prompt",
                            "status": "resolved",
                            "draft": prompt,
                            "event_id": "event-2",
                            "accepted_value": prompt,
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash(prompt),
                        },
                        # Model-choice review also resolved — the gate fires
                        # on any unresolved llm_model_choice site so the
                        # "all reviews resolved" negative-space test must
                        # cover this requirement explicitly.
                        {
                            "id": "model-choice-review",
                            "kind": "llm_model_choice",
                            "user_term": "llm_model_choice:rate_node",
                            "status": "resolved",
                            "draft": "test-model",
                            "event_id": "event-3",
                            "accepted_value": "test-model",
                            "accepted_artifact_hash": None,
                            "resolved_prompt_template_hash": stable_hash("test-model"),
                        },
                    ],
                },
            }
        ]
        state.edges = None
        state.outputs = [
            {
                "name": "results",
                "plugin": "json",
                "options": {"path": "outputs/scored.json", "format": "json"},
                "on_write_failure": "discard",
            }
        ]

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())

        assert isinstance(run_id, UUID)
        # Counter was NOT incremented.
        counter = service._telemetry.interpretation_placeholder_unresolved_at_runtime_total
        assert isinstance(counter, _FakeCounter)
        assert counter.calls == []


# ── Relative Path Resolution ──────────────────────────────────────────


class TestRelativePathResolution:
    """Path resolution must use data_dir as the base for relative paths.

    Without this, ``Path(value).resolve()`` resolves against the server's CWD,
    which diverges from the validation layer's behaviour.
    """

    @pytest.mark.asyncio
    async def test_relative_sink_path_resolves_against_data_dir(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Sink with a relative path under outputs/ passes when resolved against data_dir."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = None
        state.outputs = [
            {
                "name": "primary",
                "plugin": "csv",
                "options": {"path": "outputs/result.csv"},
                "on_write_failure": "discard",
            }
        ]
        state.nodes = None
        state.edges = None

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_relative_source_path_resolves_against_data_dir(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Source with a relative path under blobs/ passes when resolved against data_dir."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"path": "blobs/data.csv"},
            "on_validation_failure": "quarantine",
        }
        state.outputs = None
        state.nodes = None
        state.edges = None

        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_second_named_source_path_outside_allowed_dirs_raises(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Every named source path must pass the direct /execute allowlist."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "orders_out",
            "options": {"path": "blobs/orders.csv"},
            "on_validation_failure": "quarantine",
        }
        state.sources = {
            "orders": state.source,
            "refunds": {
                "plugin": "csv",
                "on_success": "refunds_out",
                "options": {"path": "/etc/passwd"},
                "on_validation_failure": "quarantine",
            },
        }
        state.outputs = None
        state.nodes = None
        state.edges = None

        from elspeth.web.execution.errors import PathAllowlistViolationError

        with pytest.raises(PathAllowlistViolationError, match=r"Source 'refunds'.*resolves outside allowed directories"):
            await service.execute(session_id=uuid4())

    @pytest.mark.asyncio
    async def test_relative_traversal_still_blocked(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Source with ../ traversal is rejected even when relative."""
        mock_settings.data_dir = "/tmp/elspeth_data"
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"path": "../etc/passwd"},
            "on_validation_failure": "quarantine",
        }
        state.outputs = None
        state.nodes = None
        state.edges = None

        from elspeth.web.execution.errors import PathAllowlistViolationError

        with pytest.raises(PathAllowlistViolationError, match="resolves outside allowed directories"):
            await service.execute(session_id=uuid4())


# ── Edge Compatibility in _run_pipeline ───────────────────────────────


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestEdgeCompatibility:
    """P2 fix: _run_pipeline must call validate_edge_compatibility() so that
    schema-incompatible pipelines are rejected before execution begins."""

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_validate_edge_compatibility_called(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
    ) -> None:
        """_run_pipeline must call graph.validate_edge_compatibility()
        after graph.validate() to catch schema mismatches."""
        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MagicMock(
            run_id="r1",
            status=RunStatus.COMPLETED,
            rows_processed=10,
            rows_succeeded=10,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        with patch(
            "elspeth.web.execution.service.load_run_accounting_from_db",
            return_value=_run_accounting_for_status(RunStatus.COMPLETED),
        ):
            service._run_pipeline(str(uuid4()), "source:\n  plugin: csv", threading.Event())

        mock_graph.validate.assert_called_once()
        mock_graph.validate_edge_compatibility.assert_called_once()

    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_edge_compatibility_failure_crashes_pipeline(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        service: ExecutionServiceImpl,
    ) -> None:
        """If edge compatibility fails, the pipeline must not execute."""
        from elspeth.core.dag.models import GraphValidationError

        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        mock_graph.validate_edge_compatibility.side_effect = GraphValidationError(
            "Schema mismatch: source outputs str but transform expects int"
        )

        with pytest.raises(GraphValidationError, match="Schema mismatch"):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())


# ── Blob Finalization Catch Widening ──────────────────────────────────


def _make_strict_call_async() -> tuple[Callable[[Coroutine[Any, Any, Any]], Any], asyncio.AbstractEventLoop]:
    """Create a _call_async bridge that propagates all exceptions faithfully.

    The standard test fixture's _mock_call_async catches RuntimeError to
    handle "event loop is closed" issues. For finalize tests, that masks
    the exact exception we're trying to test. This version propagates
    everything.
    """
    loop = asyncio.new_event_loop()

    def _call_async(coro: Coroutine[Any, Any, Any]) -> Any:
        return loop.run_until_complete(coro)

    return _call_async, loop


class TestFinalizeOutputBlobsCatchWidening:
    """Bug: elspeth-25df1be367 — _finalize_output_blobs only catches
    OSError and SQLAlchemyError, but finalize_run_output_blobs can raise
    BlobNotFoundError and RuntimeError from _finalize_blob_sync.

    These escaping exceptions trigger a second terminal event via the
    outer except BaseException, violating the "exactly one terminal state"
    invariant.

    Uses a strict _call_async that does NOT swallow RuntimeError (unlike
    the standard test fixture).
    """

    @pytest.fixture(autouse=True)
    def _cleanup_loops(self) -> Iterator[None]:
        self._loops_to_close: list[asyncio.AbstractEventLoop] = []
        yield
        for loop in self._loops_to_close:
            loop.close()

    def _make_service_with_blob(
        self, blob_service: MagicMock, mock_settings: MagicMock, mock_session_service: MagicMock
    ) -> ExecutionServiceImpl:
        svc = ExecutionServiceImpl(
            loop=MagicMock(spec=asyncio.AbstractEventLoop),
            broadcaster=MagicMock(spec=ProgressBroadcaster),
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
            blob_service=blob_service,
        )
        call_async, loop = _make_strict_call_async()
        self._loops_to_close.append(loop)
        cast(Any, svc)._call_async = call_async
        return svc

    def test_suppresses_blob_not_found_error(self, mock_settings: MagicMock, mock_session_service: MagicMock) -> None:
        from elspeth.web.blobs.protocol import BlobNotFoundError

        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=BlobNotFoundError("missing-blob"))
        svc = self._make_service_with_blob(blob_service, mock_settings, mock_session_service)
        svc._finalize_output_blobs(str(uuid4()), success=True)

    def test_propagates_runtime_error_from_blob_lifecycle(self, mock_settings: MagicMock, mock_session_service: MagicMock) -> None:
        """RuntimeError is no longer suppressed — it's too broad and would
        catch Tier 1 anomaly signals.  Blob lifecycle errors should use
        BlobStateError or BlobNotFoundError instead.
        """
        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(
            side_effect=RuntimeError("Cannot finalize — status is 'ready', expected 'pending'")
        )
        svc = self._make_service_with_blob(blob_service, mock_settings, mock_session_service)
        with pytest.raises(RuntimeError, match="Cannot finalize"):
            svc._finalize_output_blobs(str(uuid4()), success=True)

    def test_suppresses_blob_quota_exceeded_error(self, mock_settings: MagicMock, mock_session_service: MagicMock) -> None:
        from elspeth.web.blobs.protocol import BlobQuotaExceededError

        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=BlobQuotaExceededError("sess-1", current_bytes=100, limit_bytes=50))
        svc = self._make_service_with_blob(blob_service, mock_settings, mock_session_service)
        svc._finalize_output_blobs(str(uuid4()), success=True)

    def test_propagates_type_error(self, mock_settings: MagicMock, mock_session_service: MagicMock) -> None:
        """Programmer bugs (TypeError, AttributeError, etc.) must still crash."""
        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=TypeError("unexpected keyword argument"))
        svc = self._make_service_with_blob(blob_service, mock_settings, mock_session_service)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            svc._finalize_output_blobs(str(uuid4()), success=True)

    def test_propagates_attribute_error(self, mock_settings: MagicMock, mock_session_service: MagicMock) -> None:
        """AttributeError is a programmer bug — must crash."""
        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=AttributeError("'NoneType' object has no attribute 'id'"))
        svc = self._make_service_with_blob(blob_service, mock_settings, mock_session_service)
        with pytest.raises(AttributeError):
            svc._finalize_output_blobs(str(uuid4()), success=True)


# ── Terminal Ordering Invariant ───────────────────────────────────────


def _collect_terminal_types(mock_broadcaster: MagicMock) -> list[str]:
    """Extract terminal event types from a mock broadcaster's call log."""
    terminals = []
    for call in mock_broadcaster.broadcast.call_args_list:
        _, event = call[0]
        if event.event_type in ("completed", "failed", "cancelled"):
            terminals.append(event.event_type)
    return terminals


@pytest.mark.usefixtures("mock_pipeline_config_assembly")
class TestTerminalOrderingInvariant:
    """Bug: elspeth-25df1be367 — run termination is published before output
    blob finalization. A late finalize failure triggers a second terminal event
    via except BaseException.

    CLAUDE.md invariant: "Every row reaches exactly one terminal state."
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_single_terminal_when_finalize_raises_blob_not_found(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """When finalize_run_output_blobs raises BlobNotFoundError after
        a successful orchestrator.run(), exactly one terminal event must
        be broadcast — not completed-then-failed."""
        from elspeth.web.blobs.protocol import BlobNotFoundError

        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.run_id = "landscape-run-1"
        mock_result.status = RunStatus.COMPLETED_WITH_FAILURES
        mock_result.rows_processed = 10
        mock_result.rows_succeeded = 9
        mock_result.rows_failed = 1
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

        mock_broadcaster = MagicMock(spec=ProgressBroadcaster)
        blob_service = MagicMock()
        blob_service.finalize_run_output_blobs = AsyncMock(side_effect=BlobNotFoundError("blob-vanished"))

        svc = ExecutionServiceImpl(
            loop=MagicMock(spec=asyncio.AbstractEventLoop),
            broadcaster=mock_broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
            blob_service=blob_service,
        )
        _real_loop = asyncio.new_event_loop()
        try:
            cast(Any, svc)._call_async = lambda coro: _real_loop.run_until_complete(coro)

            with contextlib.suppress(Exception):
                svc._run_pipeline(str(uuid4()), "yaml", threading.Event())

            terminals = _collect_terminal_types(mock_broadcaster)
            assert len(terminals) == 1, (
                f"Exactly one terminal event expected, got {terminals}. A finalize failure must not trigger a second terminal broadcast."
            )
        finally:
            _real_loop.close()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.preflight.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.preflight.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_externally_cancelled_run_emits_single_cancelled_terminal(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        mock_settings: MagicMock,
        mock_session_service: MagicMock,
    ) -> None:
        """When a run completes but the DB status is already 'cancelled'
        (external orphan cleanup raced), exactly one terminal event must
        be emitted — not completed-then-cancelled."""
        mock_load.return_value = _mock_pipeline_settings()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.run_id = "landscape-run-2"
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 5
        mock_result.rows_succeeded = 5
        mock_result.rows_failed = 0
        mock_result.rows_routed_success = 0
        mock_result.rows_routed_failure = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

        mock_broadcaster = MagicMock(spec=ProgressBroadcaster)

        # Simulate external cancel: update_run_status("running") succeeds,
        # then update_run_status("completed") raises ValueError because
        # orphan cleanup already set the DB status to "cancelled".
        async def _selective_update(run_id, *, status="", **kwargs):
            if status == "completed":
                raise IllegalRunTransitionError("cancelled", "completed", frozenset())
            return None

        mock_session_service.update_run_status = AsyncMock(side_effect=_selective_update)
        mock_session_service.get_run = AsyncMock(return_value=MagicMock(status="cancelled"))

        svc = ExecutionServiceImpl(
            loop=MagicMock(spec=asyncio.AbstractEventLoop),
            broadcaster=mock_broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
            telemetry=build_sessions_telemetry(),
        )
        _real_loop = asyncio.new_event_loop()
        try:
            cast(Any, svc)._call_async = lambda coro: _real_loop.run_until_complete(coro)

            svc._run_pipeline(str(uuid4()), "yaml", threading.Event())

            terminals = _collect_terminal_types(mock_broadcaster)
            assert len(terminals) == 1, (
                f"Exactly one terminal event expected, got {terminals}. "
                "External cancellation must produce a single 'cancelled', "
                "not 'completed' followed by 'cancelled'."
            )
            assert terminals[0] == "cancelled", f"Terminal should be 'cancelled' (DB is authoritative), got '{terminals[0]}'."
        finally:
            _real_loop.close()


# ── Session Lock Cleanup ──────────────────────────────────────────────


class TestSessionLockCleanup:
    """Tests that cleanup_session_lock removes per-session asyncio.Lock entries."""

    def test_cleanup_removes_existing_lock(self, service: ExecutionServiceImpl) -> None:
        """cleanup_session_lock removes the lock for a known session."""
        session_id = str(uuid4())
        service._session_locks[session_id] = asyncio.Lock()
        service.cleanup_session_lock(session_id)
        assert session_id not in service._session_locks

    def test_cleanup_noop_for_unknown_session(self, service: ExecutionServiceImpl) -> None:
        """cleanup_session_lock is a no-op for an unknown session."""
        service.cleanup_session_lock("nonexistent")  # Should not raise

    def test_cleanup_does_not_affect_other_sessions(self, service: ExecutionServiceImpl) -> None:
        """Cleaning up one session leaves other sessions' locks intact."""
        session_a = str(uuid4())
        session_b = str(uuid4())
        service._session_locks[session_a] = asyncio.Lock()
        service._session_locks[session_b] = asyncio.Lock()
        service.cleanup_session_lock(session_a)
        assert session_a not in service._session_locks
        assert session_b in service._session_locks


# ── T1: _sanitize_error_for_client ────────────────────────────────────


class TestSanitizeErrorForClient:
    """Security boundary: error messages exposed to WebSocket clients
    and persisted in runs.error must not leak internal details."""

    def test_secret_resolution_error_returns_safe_message(self) -> None:
        """SecretResolutionError must NEVER leak secret names."""
        from elspeth.core.secrets import SecretResolutionError
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = SecretResolutionError(["DB_PASSWORD", "API_KEY"])
        result = _sanitize_error_for_client(exc)
        assert "DB_PASSWORD" not in result
        assert "API_KEY" not in result
        assert "secret" in result.lower()

    def test_value_error_does_not_leak_validation_structure(self) -> None:
        """ValueError can carry Pydantic/config internals and must be generic."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = ValueError("2 validation errors for PipelineSettings\nsource.options.internal_token_path\n  Field required [type=missing]")
        result = _sanitize_error_for_client(exc)
        assert result == "Pipeline execution failed (ValueError)"
        assert "PipelineSettings" not in result
        assert "internal_token_path" not in result

    def test_type_error_does_not_leak_function_signature(self) -> None:
        """TypeError can carry function signatures and must be generic."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = TypeError("build_pipeline() got an unexpected keyword argument 'internal_model_state'")
        result = _sanitize_error_for_client(exc)
        assert result == "Pipeline execution failed (TypeError)"
        assert "build_pipeline" not in result
        assert "internal_model_state" not in result

    def test_key_error_does_not_leak_internal_names(self) -> None:
        """KeyError is NOT allowlisted — str(KeyError) leaks dict key names."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = KeyError("_SCOPE_TO_AUDIT_SOURCE")
        result = _sanitize_error_for_client(exc)
        assert "_SCOPE_TO_AUDIT_SOURCE" not in result
        assert "KeyError" in result

    def test_runtime_error_returns_generic_message(self) -> None:
        """Unexpected exceptions get a generic message with class name only."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = RuntimeError("internal traceback details here /home/john/elspeth/src")
        result = _sanitize_error_for_client(exc)
        assert "/home/john" not in result
        assert "RuntimeError" in result

    def test_os_error_returns_generic_message(self) -> None:
        """OSError with file paths must not leak."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = OSError("[Errno 13] Permission denied: '/var/secrets/key.pem'")
        result = _sanitize_error_for_client(exc)
        assert "/var/secrets" not in result
        assert "OSError" in result


# ── T2: _resolve_yaml_paths ───────────────────────────────────────────


class TestResolveYamlPaths:
    """Path rewriting from relative to absolute before YAML reaches plugins."""

    def test_source_relative_path_rewritten(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "source:\n  plugin: csv\n  options:\n    path: data/input.csv\n"
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "/srv/data/data/input.csv" in result

    def test_source_absolute_path_unchanged(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "source:\n  plugin: csv\n  options:\n    path: /absolute/input.csv\n"
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "/absolute/input.csv" in result

    def test_sink_relative_path_rewritten(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "source:\n  plugin: csv\n  options:\n    path: /abs/in.csv\nsinks:\n  primary:\n    plugin: csv\n    options:\n      file: output/results.csv\n"
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "/srv/data/output/results.csv" in result

    def test_non_string_input_raises_type_error(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        with pytest.raises(TypeError, match="must return str"):
            _resolve_yaml_paths(123, "/srv/data")  # type: ignore[arg-type]

    def test_non_dict_yaml_raises_type_error(self) -> None:
        """YAML that parses to a scalar (not a dict) is a generator bug."""
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        with pytest.raises(TypeError, match="non-dict top-level"):
            _resolve_yaml_paths("just a string", "/srv/data")

    def test_no_source_or_sinks_is_noop(self) -> None:
        """YAML with no source/sinks passes through without error."""
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "metadata:\n  name: test\n"
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "name: test" in result

    def test_source_without_options_raises_type_error(self) -> None:
        """A present ``source`` missing its ``options`` key is a generator-contract
        violation, not optional data.

        ``yaml_generator`` emits ``source.options`` unconditionally
        (yaml_generator.py:92) and both production callers feed
        ``generate_yaml()`` output here — never hand-authored YAML. So a
        ``source`` without ``options`` can only mean a generator bug, and
        ``resolve_runtime_yaml_paths`` asserts it loudly rather than masking
        the absence with ``.get()`` (sinks differ — the generator emits sink
        options conditionally, so the sink path tolerates absence by design).
        """
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "source:\n  plugin: csv\n"
        with pytest.raises(TypeError, match="without required 'options'"):
            _resolve_yaml_paths(yaml_str, "/srv/data")

    def test_plural_source_relative_paths_rewritten(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = (
            "sources:\n"
            "  orders:\n"
            "    plugin: csv\n"
            "    options:\n"
            "      path: data/orders.csv\n"
            "  refunds:\n"
            "    plugin: csv\n"
            "    options:\n"
            "      file: /absolute/refunds.csv\n"
        )
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "/srv/data/data/orders.csv" in result
        assert "/absolute/refunds.csv" in result

    @pytest.mark.parametrize(
        ("yaml_str", "message"),
        [
            ("sources: []\n", "non-dict 'sources'"),
            ("sources:\n  orders: csv\n", "non-dict source 'sources.orders'"),
            ("sources:\n  orders:\n    plugin: csv\n    options: []\n", "non-dict 'sources.orders.options'"),
        ],
    )
    def test_plural_source_malformed_shapes_fail_closed(self, yaml_str: str, message: str) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        with pytest.raises(TypeError, match=message):
            _resolve_yaml_paths(yaml_str, "/srv/data")


# ── Phase 2.2 propagation: _partial_completion_message ───────────────


class TestPartialCompletionMessage:
    """Sibling to ``_structural_failure_message`` for COMPLETED_WITH_FAILURES.

    Populated into ``RunRecord.error`` so the frontend can render failure
    evidence for partial-success runs without re-implementing the L0
    ``failure_indicator`` predicate.  The RunRecord invariant at
    ``sessions/protocol.py:237-238`` permits ``error`` on any status; only
    ``failed`` *requires* it.
    """

    def test_returns_non_empty_string(self) -> None:
        from elspeth.web.execution.service import _partial_completion_message

        msg = _partial_completion_message(
            rows_succeeded=7,
            rows_failed=3,
            rows_routed_failure=1,
            rows_quarantined=2,
        )
        assert msg
        assert isinstance(msg, str)

    def test_includes_all_count_fields(self) -> None:
        """Operator must be able to read the failure breakdown directly from
        the runs row.  The four counts are the structural failure-evidence
        surface in ``RunRecord``."""
        from elspeth.web.execution.service import _partial_completion_message

        msg = _partial_completion_message(
            rows_succeeded=7,
            rows_failed=3,
            rows_routed_failure=1,
            rows_quarantined=2,
        )
        assert "rows_succeeded=7" in msg
        assert "rows_failed=3" in msg
        assert "rows_routed_failure=1" in msg
        assert "rows_quarantined=2" in msg

    def test_points_at_user_visible_affordance_when_no_samples(self) -> None:
        """Without enrichment samples, the message must direct the operator to
        the actual UI surface — the in-page expand panel — not a backend API
        path the user cannot navigate to (was '/diagnostics')."""
        from elspeth.web.execution.service import _partial_completion_message

        msg = _partial_completion_message(
            rows_succeeded=1,
            rows_failed=1,
            rows_routed_failure=0,
            rows_quarantined=0,
        )
        assert "Expand this run" in msg
        assert "/diagnostics" not in msg

    def test_inlines_failure_samples_when_supplied(self) -> None:
        """When the caller supplies a pre-formatted samples block, the
        message inlines it under a 'Top per-row failures' heading so the
        runs view shows the dominant cause without needing the expand."""
        from elspeth.web.execution.service import _partial_completion_message

        samples = "  • 3x SSRFBlockedError: URL is missing a scheme"
        msg = _partial_completion_message(
            rows_succeeded=0,
            rows_failed=3,
            rows_routed_failure=0,
            rows_quarantined=0,
            failure_samples=samples,
        )
        assert "Top per-row failures:" in msg
        assert samples in msg
        assert "Expand this run" not in msg

    def test_deterministic_given_inputs(self) -> None:
        """No timestamps, no random IDs — the message is a pure function of
        its counts so audit-trail comparisons are stable."""
        from elspeth.web.execution.service import _partial_completion_message

        a = _partial_completion_message(rows_succeeded=5, rows_failed=2, rows_routed_failure=1, rows_quarantined=0)
        b = _partial_completion_message(rows_succeeded=5, rows_failed=2, rows_routed_failure=1, rows_quarantined=0)
        assert a == b

    def test_does_not_echo_user_row_data(self) -> None:
        """Same security posture as ``_structural_failure_message`` — the
        message is structural facts only; no row keys, no LLM prompts, no
        secret-resolution candidates."""
        from elspeth.web.execution.service import _partial_completion_message

        msg = _partial_completion_message(
            rows_succeeded=1,
            rows_failed=1,
            rows_routed_failure=0,
            rows_quarantined=0,
        )
        # Sanity: only structural words.  If a future change inlines a row
        # value, this assertion would fail loudly.
        forbidden_substrings = ["row_id=", "key=", "value=", "prompt=", "secret"]
        for forbidden in forbidden_substrings:
            assert forbidden not in msg.lower(), f"_partial_completion_message must not include {forbidden!r} (row-data leak)"


class TestSetOpenrouterCatalogSnapshotValidation:
    """Pin the sha256-hex validator at the snapshot setter site.

    ``set_openrouter_catalog_snapshot()`` is called once by the FastAPI
    lifespan; a non-hex string passing the old ``not sha256`` guard would
    propagate into the runs row and corrupt the audit trail. The validator
    now uses the canonical ``is_valid_sha256_hex`` shared with the
    Landscape write-side guards.
    """

    def test_setter_rejects_non_hex_sha256(self, service: ExecutionServiceImpl) -> None:
        """A non-empty non-hex string fails the hex shape check."""
        with pytest.raises(RuntimeError, match="64 lowercase hex chars"):
            service.set_openrouter_catalog_snapshot(sha256="not-a-sha", source="bundled")

    def test_setter_accepts_canonical_digest(self, service: ExecutionServiceImpl) -> None:
        """A real hashlib.sha256 hex digest passes."""
        import hashlib

        digest = hashlib.sha256(b"catalog-anchor").hexdigest()
        service.set_openrouter_catalog_snapshot(sha256=digest, source="bundled")
        # No exception — the setter accepted the value.

    def test_setter_rejects_bad_source(self, service: ExecutionServiceImpl) -> None:
        with pytest.raises(RuntimeError, match="must be 'live' or 'bundled'"):
            service.set_openrouter_catalog_snapshot(sha256="0" * 64, source="oops")
