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
import threading
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.enums import RunStatus
from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    RateLimitSettings,
    TelemetrySettings,
)
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.sessions.protocol import RunAlreadyActiveError

# ── Fixtures ───────────────────────────────────────────────────────────

_TEST_PIPELINE_YAML = "source:\n  plugin: csv\n"


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
            rows_routed=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id=None,
        )
        status = await service.get_status(run_id)
        assert status.status == "running"
        assert status.rows_processed == 50
        assert status.rows_routed == 0


class TestWebRuntimeInfrastructure:
    """Regression coverage for web execution's orchestrator runtime wiring."""

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
source:
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
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        shutdown_event = threading.Event()
        run_id = uuid4()

        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.run_id = str(run_id)
        mock_result.status = RunStatus.COMPLETED
        mock_result.rows_processed = 10
        mock_result.rows_succeeded = 10
        mock_result.rows_failed = 0
        mock_result.rows_routed = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

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
            rows_routed=0,
            rows_quarantined=0,
        )

        service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # B3: LandscapeDB constructed from settings URL
        mock_landscape_cls.assert_called_once_with(connection_string="sqlite:///test_audit.db", passphrase=None)
        # B3: PayloadStore constructed from settings path
        mock_payload_cls.assert_called_once_with(base_path=Path("/tmp/test_payloads"))


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

    @pytest.mark.asyncio
    async def test_cancel_terminal_run_is_noop(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Cancelling completed/failed/cancelled run does nothing."""
        run_id = uuid4()
        mock_session_service.get_run.return_value = MagicMock(status="completed")
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
            rows_routed=0,
            rows_quarantined=0,
        )

        shutdown_event = threading.Event()
        shutdown_event.set()
        run_id = str(uuid4())

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
            rows_routed=0,
            rows_quarantined=0,
        )

        # Do NOT set shutdown_event — let _run_pipeline proceed past the
        # early check so orchestrator.run() fires and raises the GSE.
        shutdown_event = threading.Event()
        run_id = str(uuid4())

        service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        status_calls = mock_session_service.update_run_status.call_args_list
        # Second call is the GSE handler (first is running transition)
        gse_call = status_calls[-1]
        assert gse_call.kwargs["status"] == "cancelled"
        assert gse_call.kwargs["rows_processed"] == 50
        assert gse_call.kwargs["rows_succeeded"] == 48
        assert gse_call.kwargs["rows_failed"] == 2
        assert gse_call.kwargs["rows_routed"] == 0
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
        mock_result.rows_routed = 0
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
        mock_session_service.update_run_status.side_effect = ValueError("Illegal run transition: 'cancelled' → 'running'. Allowed: []")
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

        mock_session_service.update_run_status.side_effect = ValueError("Illegal run transition: 'completed' → 'running'. Allowed: []")
        mock_session_service.get_run.return_value = MagicMock(status="completed")

        with pytest.raises(ValueError, match="completed"):
            service._run_pipeline(run_id, "yaml", threading.Event())

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
        mock_result.rows_routed = 0
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
                raise ValueError("Illegal run transition: 'cancelled' → 'completed'. Allowed: []")

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
        mock_result.rows_routed = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-789"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())

        call_count = 0

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Illegal run transition: 'cancelled' → 'completed'. Allowed: []")

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
        mock_result.rows_routed = 0
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
                raise ValueError("Illegal run transition: 'cancelled' → 'completed'. Allowed: []")

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        mock_session_service.get_run.return_value = MagicMock(status="cancelled")

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
        mock_result.rows_routed = 0
        mock_result.rows_quarantined = 0
        mock_result.run_id = "landscape-run-456"
        mock_orch.run.return_value = mock_result

        run_id = str(uuid4())

        call_count = 0

        async def status_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Illegal run transition: 'completed' → 'completed'. Allowed: []")

        mock_session_service.update_run_status = AsyncMock(side_effect=status_side_effect)
        # DB says "completed" (not "cancelled") — this should re-raise
        mock_session_service.get_run.return_value = MagicMock(status="completed")

        with pytest.raises(ValueError, match="completed"):
            service._run_pipeline(run_id, "source:\n  plugin: csv", threading.Event())


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
        """A non-UUID blob_ref raises ValueError before create_run()
        is called, so no pending run is orphaned."""
        state = mock_session_service.get_current_state.return_value
        state.source = {
            "plugin": "csv",
            "on_success": "continue",
            "options": {"blob_ref": "not-a-uuid"},
            "on_validation_failure": "quarantine",
        }

        blob_service = MagicMock()
        cast(Any, service)._blob_service = blob_service

        with pytest.raises(ValueError):
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
        """_to_run_event maps ProgressEvent fields to RunEvent.data."""
        from elspeth.contracts.cli import ProgressEvent
        from elspeth.web.execution.schemas import ProgressData

        progress = ProgressEvent(
            rows_processed=100,
            rows_succeeded=95,
            rows_failed=5,
            rows_quarantined=3,
            elapsed_seconds=10.5,
        )
        run_id = "run-123"
        run_event = service._to_run_event(run_id, progress)

        assert run_event.event_type == "progress"
        assert isinstance(run_event.data, ProgressData)
        assert run_event.data.rows_processed == 100
        assert run_event.data.rows_failed == 5
        assert run_event.run_id == "run-123"


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

        with pytest.raises(ValueError, match="resolves outside allowed output directories"):
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

        with pytest.raises(ValueError, match="resolves outside allowed output directories"):
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

        with pytest.raises(ValueError, match="resolves outside allowed directories"):
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
            rows_routed=0,
            rows_quarantined=0,
        )

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
        mock_result.rows_routed = 0
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
        mock_result.rows_routed = 0
        mock_result.rows_quarantined = 0
        mock_orch.run.return_value = mock_result

        mock_broadcaster = MagicMock(spec=ProgressBroadcaster)

        # Simulate external cancel: update_run_status("running") succeeds,
        # then update_run_status("completed") raises ValueError because
        # orphan cleanup already set the DB status to "cancelled".
        async def _selective_update(run_id, *, status="", **kwargs):
            if status == "completed":
                raise ValueError("Invalid transition: cancelled -> completed")
            return None

        mock_session_service.update_run_status = AsyncMock(side_effect=_selective_update)
        mock_session_service.get_run = AsyncMock(return_value=MagicMock(status="cancelled"))

        svc = ExecutionServiceImpl(
            loop=MagicMock(spec=asyncio.AbstractEventLoop),
            broadcaster=mock_broadcaster,
            settings=mock_settings,
            session_service=mock_session_service,
            yaml_generator=MagicMock(),
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

    def test_value_error_passes_through(self) -> None:
        """ValueError is allowlisted — user-actionable config errors."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = ValueError("Invalid source path: /tmp/data.csv")
        assert _sanitize_error_for_client(exc) == "Invalid source path: /tmp/data.csv"

    def test_type_error_passes_through(self) -> None:
        """TypeError is allowlisted — type mismatches in config/YAML."""
        from elspeth.web.execution.service import _sanitize_error_for_client

        exc = TypeError("Expected str, got int")
        assert _sanitize_error_for_client(exc) == "Expected str, got int"

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

    def test_source_without_options_is_noop(self) -> None:
        from elspeth.web.execution.preflight import resolve_runtime_yaml_paths as _resolve_yaml_paths

        yaml_str = "source:\n  plugin: csv\n"
        result = _resolve_yaml_paths(yaml_str, "/srv/data")
        assert "plugin: csv" in result
