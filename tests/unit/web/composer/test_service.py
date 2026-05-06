"""Tests for ComposerServiceImpl — LLM tool-use loop with mock LLM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.composer.progress import ComposerProgressEvent
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerResult,
    ComposerRuntimePreflightError,
    ComposerServiceError,
    ToolArgumentError,
)
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult
from elspeth.web.composer.tools import execute_tool as _execute_tool
from elspeth.web.config import WebSettings
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeLLMResponse:
    choices: list[FakeChoice]


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="uppercase",
            description="Uppercase",
            plugin_type="transform",
            config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv",
            description="CSV sink",
            plugin_type="sink",
            config_fields=[],
        ),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
    )
    return catalog


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> FakeLLMResponse:
    """Build a typed fake LiteLLM response.

    Uses typed dataclasses instead of MagicMock so tests fail if production
    code accesses an attribute that doesn't exist on the real response shape.
    """
    fake_tool_calls: list[FakeToolCall] | None = None
    if tool_calls:
        fake_tool_calls = [
            FakeToolCall(
                id=tc["id"],
                function=FakeFunction(
                    name=tc["name"],
                    arguments=json.dumps(tc["arguments"]),
                ),
            )
            for tc in tool_calls
        ]

    message = FakeMessage(content=content, tool_calls=fake_tool_calls)
    return FakeLLMResponse(choices=[FakeChoice(message=message)])


def _make_settings(**overrides: Any) -> WebSettings:
    """Build WebSettings with Pydantic-enforced defaults.

    Use keyword arguments to override specific fields for a test.
    Defaults come from the Pydantic model — no drift possible.

    data_dir defaults to /data (absolute) so test paths like
    /data/blobs/file.csv pass S2 path validation.
    """
    defaults: dict[str, Any] = {
        "data_dir": Path("/data"),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)


def _session_engine_with_session() -> tuple[Any, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="Test Session",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep service tests focused on compose behavior, not local API keys."""

    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.fixture(autouse=True)
def _composer_to_thread_uses_test_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run composer to_thread seams through a deterministic test worker.

    These unit tests assert the composer offloads synchronous tool and audit
    work; they do not need to test asyncio's executor implementation. The shim
    keeps the off-event-loop-thread property visible while avoiding local
    executor hangs from masking composer behavior.
    """

    async def test_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        import threading

        result: list[Any] = []
        failures: list[BaseException] = []

        def run() -> None:
            try:
                result.append(func(*args, **kwargs))
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, name="composer-test-worker")
        worker.start()
        while worker.is_alive():
            await asyncio.sleep(0.001)
        worker.join()
        if failures:
            raise failures[0]
        if result:
            return result[0]
        return None

    monkeypatch.setattr("asyncio.to_thread", test_to_thread)


class TestComposerTextOnlyResponse:
    @pytest.mark.asyncio
    async def test_text_only_returns_immediately(self) -> None:
        """LLM responds with text only — no tool calls, loop terminates."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
        )
        state = _empty_state()

        llm_response = _make_llm_response(content="I'll help you build a pipeline!")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await service.compose("Build me a CSV pipeline", [], state)

        assert isinstance(result, ComposerResult)
        assert result.message == "I'll help you build a pipeline!"
        assert result.state.version == 1  # unchanged


class TestComposerSingleToolCall:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_text(self) -> None:
        """LLM makes one tool call, then responds with text."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: tool call to set_source
        tool_response = _make_llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/data.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Turn 2: text response
        text_response = _make_llm_response(content="I've set up a CSV source.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [tool_response, text_response]
            result = await service.compose("Use CSV as source", [], state)

        assert result.message == "I've set up a CSV source."
        assert result.state.source is not None
        assert result.state.source.plugin == "csv"
        assert result.state.version == 2

    @pytest.mark.asyncio
    async def test_progress_reports_visible_boundaries_without_tool_payloads(self) -> None:
        """Progress summaries derive from visible lifecycle/tool names, not raw args."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        tool_response = _make_llm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_secret",
                    "name": "validate_secret_ref",
                    "arguments": {"name": "OPENROUTER_TOP_SECRET_VALUE"},
                }
            ],
        )
        text_response = _make_llm_response(content="I checked the available credentials.")

        async def inline_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch("elspeth.web.composer.service.asyncio.to_thread", side_effect=inline_to_thread),
        ):
            mock_llm.side_effect = [tool_response, text_response]
            result = await service.compose(
                "Use my OpenRouter secret",
                [],
                state,
                progress=record_progress,
            )

        assert result.message == "I checked the available credentials."
        phases = [event.phase for event in progress_events]
        assert phases[:2] == ["starting", "calling_model"]
        assert "using_tools" in phases
        assert "complete" in phases

        progress_text = "\n".join(line for event in progress_events for line in (event.headline, event.likely_next or "", *event.evidence))
        assert "OPENROUTER_TOP_SECRET_VALUE" not in progress_text
        assert '"name"' not in progress_text
        assert "checking available secret references" in progress_text.lower()

    @pytest.mark.asyncio
    async def test_inline_complete_pipeline_replay_uses_atomic_tool_shape(self, tmp_path: Path) -> None:
        """Fake-model replay for simple inline-data builds stays provider-bounded."""
        catalog = _mock_catalog()
        engine, session_id = _session_engine_with_session()
        settings = _make_settings(data_dir=tmp_path)
        service = ComposerServiceImpl(catalog=catalog, settings=settings, session_engine=engine)
        state = _empty_state()
        output_path = tmp_path / "outputs" / "append.csv"
        pipeline_args = {
            "source": {
                "plugin": "text",
                "on_success": "source_out",
                "options": {
                    "column": "text",
                    "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                "inline_blob": {
                    "filename": "input.txt",
                    "mime_type": "text/plain",
                    "content": "hello",
                },
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "append_world",
                    "node_type": "transform",
                    "plugin": "value_transform",
                    "input": "source_out",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": {
                        "schema": {
                            "mode": "observed",
                            "guaranteed_fields": ["text"],
                            "required_fields": ["text"],
                        },
                        "operations": [{"target": "text", "expression": "row['text'] + ' world'"}],
                    },
                }
            ],
            "edges": [
                {"id": "source_to_append", "from_node": "source", "to_node": "append_world", "edge_type": "on_success"},
                {"id": "append_to_main", "from_node": "append_world", "to_node": "main", "edge_type": "on_success"},
            ],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": str(output_path),
                        "schema": {"mode": "observed", "required_fields": ["text"]},
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
            "metadata": {"name": "Append literal text"},
        }
        build_turn = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_build",
                    "name": "set_pipeline",
                    "arguments": pipeline_args,
                }
            ],
        )
        preview_turn = _make_llm_response(tool_calls=[{"id": "call_preview", "name": "preview_pipeline", "arguments": {}}])
        final_turn = _make_llm_response(content="Pipeline configured.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(
                service,
                "_cached_runtime_preflight",
                new_callable=AsyncMock,
                return_value=ValidationResult(is_valid=True, checks=[], errors=[]),
            ),
        ):
            mock_llm.side_effect = [build_turn, preview_turn, final_turn]
            result = await service.compose(
                "I want a pipeline that takes the string 'hello' and appends ' world' to it.",
                [],
                state,
                session_id=session_id,
            )

        assert result.message == "Pipeline configured."
        assert mock_llm.call_count == 3
        tool_names = [inv.tool_name for inv in result.tool_invocations]
        assert tool_names == ["set_pipeline", "preview_pipeline"]
        assert "create_blob" not in tool_names
        assert "set_source_from_blob" not in tool_names
        assert "upsert_node" not in tool_names
        assert "set_output" not in tool_names
        assert len(result.llm_calls) == 3
        assert result.state.source is not None
        assert "blob_ref" in result.state.source.options


class TestComposerMultiTurnToolCalls:
    @pytest.mark.asyncio
    async def test_multi_turn_state_accumulates(self) -> None:
        """Multiple tool calls across turns — state accumulates."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source
        turn1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Turn 2: set_metadata
        turn2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "My Pipeline"}},
                }
            ],
        )
        # Turn 3: text
        turn3 = _make_llm_response(content="Pipeline configured.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [turn1, turn2, turn3]
            result = await service.compose("Build a pipeline", [], state)

        assert result.state.source is not None
        assert result.state.metadata.name == "My Pipeline"
        assert result.state.version == 3  # two mutations


class TestComposerConvergence:
    @pytest.mark.asyncio
    async def test_discovery_budget_exceeded_raises(self) -> None:
        """Discovery-only turns exhaust the discovery budget."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Two different discovery tools to avoid cache hits
        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Loop forever", [], state)
            assert exc_info.value.budget_exhausted == "discovery"

    @pytest.mark.asyncio
    async def test_composition_budget_exceeded_raises(self) -> None:
        """Mutation turns exhaust the composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "test"}},
                }
            ],
        )
        # Bonus call also returns tool calls — convergence error
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "test2"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Keep mutating", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_self_correction_on_final_composition_turn_succeeds(self) -> None:
        """B-4D-3: LLM makes mutation on final turn, then text on bonus call."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Final"}},
                }
            ],
        )
        text = _make_llm_response(content="Done after final correction.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [mut, text]
            result = await service.compose("Do it", [], state)

        assert result.message == "Done after final correction."
        assert result.state.metadata.name == "Final"

    @pytest.mark.asyncio
    async def test_mixed_turns_charge_correct_budgets(self) -> None:
        """Mixed discovery/mutation turns are classified independently.

        Discovery turns charge discovery budget, mutation turns charge
        composition budget. Neither exhausts the other.
        """
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=2,
            composer_max_discovery_turns=2,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: discovery (list_sources) — discovery counter = 1
        disc = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        # Turn 2: mutation (set_metadata) — composition counter = 1
        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Works"}},
                }
            ],
        )
        # Turn 3: text response — loop terminates
        text = _make_llm_response(content="Done.")

        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])
        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [disc, mut, text]
            result = await service.compose("Build", [], state)

        assert result.message == "Done."
        assert result.state.metadata.name == "Works"


class TestFailedMutationBudgetClassification:
    """Failed mutation tool calls must charge composition budget, not discovery."""

    @pytest.mark.asyncio
    async def test_failed_mutation_charges_composition_budget(self) -> None:
        """A mutation tool that fails with KeyError/TypeError should exhaust
        composition budget, not discovery budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=1,
            composer_max_discovery_turns=10,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source with missing required key → KeyError
        # This is a mutation tool, so even though it fails, it should
        # charge composition budget (1/1 → exhausted).
        bad_mutation = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing on_success
                }
            ],
        )
        # Bonus call (composition exhausted gives one last chance) also
        # returns a tool call → convergence error
        bad_mutation2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_mutation, bad_mutation2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Setup source", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_failed_mutation_json_parse_charges_composition_budget(self) -> None:
        """Mutation tool with unparseable JSON arguments should still
        charge composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=1,
            composer_max_discovery_turns=10,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Build a tool call with invalid JSON manually
        call = FakeToolCall(
            id="c1",
            function=FakeFunction(
                name="set_source",
                arguments="{invalid json",
            ),
        )
        msg = FakeMessage(content=None, tool_calls=[call])
        response = FakeLLMResponse(choices=[FakeChoice(message=msg)])

        # Bonus call also fails
        call2 = FakeToolCall(
            id="c2",
            function=FakeFunction(
                name="set_source",
                arguments="{still invalid",
            ),
        )
        msg2 = FakeMessage(content=None, tool_calls=[call2])
        response2 = FakeLLMResponse(choices=[FakeChoice(message=msg2)])

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [response, response2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Setup source", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_failed_discovery_does_not_charge_composition_budget(self) -> None:
        """A failed discovery tool should still charge discovery budget,
        not composition budget."""
        catalog = _mock_catalog()
        settings = _make_settings(
            composer_max_composition_turns=10,
            composer_max_discovery_turns=1,
        )
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # list_sources with invalid JSON → still a discovery turn
        call = FakeToolCall(
            id="c1",
            function=FakeFunction(
                name="list_sources",
                arguments="{bad json",
            ),
        )
        msg = FakeMessage(content=None, tool_calls=[call])
        response = FakeLLMResponse(choices=[FakeChoice(message=msg)])

        # Turn 2: another discovery call
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [response, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Explore", [], state)
            assert exc_info.value.budget_exhausted == "discovery"


class TestComposerErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_to_llm(self) -> None:
        """Unknown tool name returns error message, LLM can retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: invalid tool
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "nonexistent_tool",
                    "arguments": {},
                }
            ],
        )
        # Turn 2: text response (self-corrected)
        text = _make_llm_response(content="Sorry, let me try again.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Do something", [], state)

        assert result.message == "Sorry, let me try again."
        # State unchanged — the bad tool call didn't modify anything
        assert result.state.version == 1

    @pytest.mark.asyncio
    async def test_malformed_arguments_returns_error(self) -> None:
        """Malformed tool arguments return error, not crash."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source with missing required field
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing on_success
                }
            ],
        )
        # Turn 2: text
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Fixed."

    @pytest.mark.asyncio
    async def test_wrong_type_tool_arg_returns_error(self) -> None:
        """ToolArgumentError from Tier 3 type guard in tool handler is caught, not crash.

        Tool handlers validate LLM-provided argument types at the Tier 3
        boundary, raising ToolArgumentError for wrong types (e.g. int where
        str expected). The compose loop catches this typed exception and
        feeds the error back to the LLM so it can retry with a corrected
        argument.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: tool call that triggers ToolArgumentError from Tier 3 type guard
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": {"plugin": "csv", "on_success": "out"},
                }
            ],
        )
        # Turn 2: LLM self-corrects
        text = _make_llm_response(content="Fixed.")

        with (
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ToolArgumentError(
                    argument="content",
                    expected="a string",
                    actual_type="int",
                ),
            ),
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Fixed."

    @pytest.mark.asyncio
    async def test_malformed_set_pipeline_missing_top_level_required_field_returns_error(self) -> None:
        """Top-level required field omission in set_pipeline returns a Tier-3 arg error.

        Renamed from test_malformed_set_pipeline_nested_required_field_returns_error
        — the body always tested top-level ``source.plugin`` omission, not a nested
        required field. Coverage of nested-optional + inner-required semantics lives
        in the two tests below.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "on_success": "source_out",
                            "options": {"path": "/data/in.csv", "schema": {"mode": "observed"}},
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "uppercase",
                                "input": "source_out",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Recovered."
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "source.plugin" in error_content["error"]
        assert "missing required" in error_content["error"].lower()
        # Regression guard: the conditional-on-presence walker must NOT
        # surface false-positive errors for the optional ``source.inline_blob``
        # branch when no inline blob was supplied. See elspeth-4e79436719 §Bug A.
        assert "inline_blob" not in error_content["error"]

    @pytest.mark.asyncio
    async def test_set_pipeline_without_inline_blob_passes_predispatch_validation(self) -> None:
        """No-inline set_pipeline payloads must reach the handler.

        Regression guard for elspeth-4e79436719 Bug A: the required-paths
        walker previously lifted ``source.inline_blob.{filename,mime_type,content}``
        into top-level required paths because it recursed into nested
        ``properties`` blocks unconditionally — even when the containing
        property (``inline_blob``) was optional at its parent. Correct
        JSON-Schema semantics: nested ``required`` applies only when the
        containing object is present in the value.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        good_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_ok",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "csv",
                            "on_success": "t1",
                            "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "uppercase",
                                "input": "main",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Pipeline ready.")
        passing_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch.object(service, "_runtime_preflight", return_value=passing_preflight),
        ):
            mock_llm.side_effect = [good_call, text]
            await service.compose("Setup", [], state)

        # The pre-dispatch validator must not have rejected the call.
        # Verify by inspecting the tool-result message content: the
        # MissingRequiredPaths code path produces a JSON object whose
        # ``error`` field starts with ``Tool 'set_pipeline' missing required
        # argument(s):``. A no-inline payload must not trigger that path,
        # regardless of what the tool handler returns afterwards.
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        assert tool_msg["role"] == "tool"
        try:
            error_content = json.loads(tool_msg["content"])
            error_text = error_content.get("error", "") if isinstance(error_content, dict) else ""
        except json.JSONDecodeError:
            error_text = ""
        assert "missing required argument" not in error_text.lower()
        assert "inline_blob" not in error_text

    @pytest.mark.asyncio
    async def test_set_pipeline_with_partial_inline_blob_returns_tier3_arg_error(self) -> None:
        """When ``inline_blob`` is present but incomplete, the inner required fields ARE enforced.

        Conditional-on-presence semantics: an absent optional object skips
        its inner ``required`` checks; a present-but-incomplete optional
        object surfaces them as Tier-3 arg errors. See elspeth-4e79436719
        §Bug A.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        partial_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_partial",
                    "name": "set_pipeline",
                    "arguments": {
                        "source": {
                            "plugin": "csv",
                            "on_success": "t1",
                            "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                            "inline_blob": {"filename": "data.csv"},
                        },
                        "nodes": [
                            {
                                "id": "t1",
                                "node_type": "transform",
                                "plugin": "uppercase",
                                "input": "main",
                                "on_success": "main",
                                "options": {},
                            }
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "from_node": "source",
                                "to_node": "t1",
                                "edge_type": "on_success",
                            }
                        ],
                        "outputs": [
                            {
                                "sink_name": "main",
                                "plugin": "csv",
                                "options": {"path": "/data/out.csv", "schema": {"mode": "observed"}},
                            }
                        ],
                    },
                }
            ],
        )
        text = _make_llm_response(content="Adjusted.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [partial_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Adjusted."
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "source.inline_blob.mime_type" in error_content["error"]
        assert "source.inline_blob.content" in error_content["error"]
        # filename WAS supplied — must not be reported.
        assert "source.inline_blob.filename" not in error_content["error"]

    @pytest.mark.asyncio
    async def test_internal_key_error_is_not_swallowed(self) -> None:
        """KeyError from tool handler internals must crash, not be sent to LLM.

        Previously, KeyError from missing LLM arguments and KeyError from
        internal bugs were both caught and fed back to the LLM. Internal
        bugs should crash immediately — the LLM cannot self-correct our code.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Provide all required arguments so pre-validation passes,
        # but patch execute_tool to raise a KeyError from "internal logic"
        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=KeyError("internal_state_key"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        # The underlying KeyError is preserved on the wrapper so callers
        # (server logs, route handler, capture_logs assertions) can still
        # identify the original plugin-internal class.
        assert isinstance(exc_info.value.original_exc, KeyError)
        assert exc_info.value.exc_class == "KeyError"
        assert "internal_state_key" in str(exc_info.value.original_exc)

    @pytest.mark.asyncio
    async def test_missing_args_error_message_lists_keys(self) -> None:
        """Missing required arguments should produce a clear error listing the keys."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # set_source requires plugin, on_success, options, on_validation_failure
        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {"plugin": "csv"},  # missing 3 required args
                }
            ],
        )
        text = _make_llm_response(content="Ok.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            await service.compose("Setup", [], state)

        # Verify the error message sent back to the LLM mentions the missing keys
        tool_msg = mock_llm.call_args_list[1][0][0][-1]  # last message in second call
        error_content = json.loads(tool_msg["content"])
        assert "on_success" in error_content["error"]
        assert "missing required" in error_content["error"].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", [[], None, "oops"])
    async def test_top_level_non_object_mutation_arguments_return_tool_error(self, arguments: Any) -> None:
        """Non-object mutation arguments are Tier-3 tool errors, not plugin crashes."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source",
                    "arguments": arguments,
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.service.execute_tool",
                wraps=_execute_tool,
            ) as mock_execute_tool,
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Recovered."
        mock_execute_tool.assert_not_called()
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "arguments must be a JSON object" in error_content["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", [[], None, "oops"])
    async def test_top_level_non_object_discovery_arguments_return_tool_error(self, arguments: Any) -> None:
        """Non-object discovery arguments are rejected before cache lookup or execution."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "list_sources",
                    "arguments": arguments,
                }
            ],
        )
        text = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.service.execute_tool",
                wraps=_execute_tool,
            ) as mock_execute_tool,
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        ):
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Explore", [], state)

        assert result.message == "Recovered."
        mock_execute_tool.assert_not_called()
        tool_msg = mock_llm.call_args_list[1][0][0][-1]
        error_content = json.loads(tool_msg["content"])
        assert "arguments must be a JSON object" in error_content["error"]


class TestProviderCacheTokenAudit:
    """Audit-write capture of provider prompt-cache statistics (elspeth-4e79436719 Bug C).

    Provider responses can carry cache-hit information in two distinct
    shapes:
    - OpenAI / OpenRouter: ``usage.prompt_tokens_details.cached_tokens``
    - Anthropic: ``usage.cache_creation_input_tokens`` and
      ``usage.cache_read_input_tokens`` as siblings on usage

    Both must land on the ``ComposerLLMCall`` audit record with their
    canonical names, so the audit DB records what the provider actually
    reported. A missing field must remain ``None`` per the
    "absence as evidence" rule from CLAUDE.md.

    LiteLLM-shape deduplication: when LiteLLM is the wire to an
    Anthropic-family provider (Anthropic direct, Bedrock-Claude,
    Vertex-Claude/Gemini), it populates BOTH the nested OpenAI shape and
    the Anthropic siblings on the same response, deriving
    ``prompt_tokens_details.cached_tokens`` from the sibling
    ``cache_read_input_tokens``. The two are aliases, not independent
    signals — recording both would double-count cache hits in the audit
    sidecar, and creation-only responses (which carry ``cached_tokens=0``
    by LiteLLM default) would fabricate a zero into ``cached_prompt_tokens``
    that the provider never asserted. Presence of an Anthropic sibling is
    the provenance signal that the nested shape is LiteLLM-synthesized.
    """

    @staticmethod
    def _response_with_usage(usage: dict[str, Any]) -> FakeLLMResponse:
        """Build a fake response carrying a usage object as a Mapping.

        FakeLLMResponse only carries .choices; add .usage so the audit
        extractor's Mapping branch fires. Using a separate dataclass keeps
        the typed FakeLLMResponse intact for the rest of the test suite.
        """

        @dataclass
        class FakeResponseWithUsage:
            choices: list[FakeChoice]
            usage: dict[str, Any]
            model: str = "openrouter/openai/gpt-5.5"
            id: str = "chatcmpl-test"

        text = _make_llm_response(content="Done.")
        return FakeResponseWithUsage(choices=text.choices, usage=usage)  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_openai_nested_cached_tokens_lands_on_audit_record(self) -> None:
        from elspeth.web.composer.service import _token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "total_tokens": 1280,
                "prompt_tokens_details": {"cached_tokens": 1024},
            }
        )
        usage = _token_usage_from_response(response)
        assert usage.prompt_tokens == 1200
        assert usage.cached_prompt_tokens == 1024
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    @pytest.mark.asyncio
    async def test_anthropic_sibling_cache_fields_land_on_audit_record(self) -> None:
        from elspeth.web.composer.service import _token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 8200,
                "completion_tokens": 120,
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 1100,
            }
        )
        usage = _token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_normalized_anthropic_does_not_double_record_cache_hit(self) -> None:
        """LiteLLM-normalized Anthropic-family response: nested shape is suppressed.

        Mirrors the wire shape produced by
        ``litellm/llms/anthropic/chat/transformation.py:1294-1317`` —
        ``prompt_tokens_details.cached_tokens`` and ``cache_read_input_tokens``
        carry the SAME value because LiteLLM derives the former from the
        latter. The audit row must record only the Anthropic-shape signal.
        """
        from elspeth.web.composer.service import _token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 8200,
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 1100},
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 1100,
            }
        )
        usage = _token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_creation_only_does_not_leak_zero_into_cached_prompt_tokens(self) -> None:
        """LiteLLM creation-only Anthropic response: nested cached_tokens=0 is suppressed.

        ``PromptTokensDetailsWrapper`` preserves ``cached_tokens=0`` (verified
        empirically), so a cache-creation-only Anthropic call still emits a
        nested shape with ``cached_tokens=0``. Without dedup, the audit row
        would carry ``cached_prompt_tokens=0`` — a fabricated zero that
        misleads the auditor into thinking the provider reported a zero-hit
        cache read, when in fact the provider only reported cache creation.
        """
        from elspeth.web.composer.service import _token_usage_from_response

        response = self._response_with_usage(
            {
                "prompt_tokens": 7000,
                "completion_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 0,
            }
        )
        usage = _token_usage_from_response(response)
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 0
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_litellm_normalized_dual_shape_is_deduped_on_attribute_branch(self) -> None:
        """Pydantic-shaped (attribute) usage object also dedups when siblings present.

        Real LiteLLM responses are Pydantic ``Usage`` objects, not Mappings.
        Verifies the elif branch in ``_token_usage_from_response`` honors the
        same dedup rule: nested ``prompt_tokens_details.cached_tokens`` is
        dropped when an Anthropic sibling is present on the attribute object.
        """
        from elspeth.web.composer.service import _token_usage_from_response

        @dataclass
        class FakePromptTokensDetails:
            cached_tokens: int | None

        @dataclass
        class FakePydanticUsage:
            prompt_tokens: int
            completion_tokens: int
            total_tokens: int
            prompt_tokens_details: FakePromptTokensDetails
            cache_creation_input_tokens: int | None
            cache_read_input_tokens: int | None

        @dataclass
        class FakeResponseWithPydanticUsage:
            choices: list[FakeChoice]
            usage: FakePydanticUsage
            model: str = "anthropic/claude-3-5-sonnet"
            id: str = "msg_test"

        text = _make_llm_response(content="Done.")
        response = FakeResponseWithPydanticUsage(
            choices=text.choices,
            usage=FakePydanticUsage(
                prompt_tokens=8200,
                completion_tokens=120,
                total_tokens=8320,
                prompt_tokens_details=FakePromptTokensDetails(cached_tokens=1100),
                cache_creation_input_tokens=7000,
                cache_read_input_tokens=1100,
            ),
        )
        usage = _token_usage_from_response(response)  # type: ignore[arg-type]
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        assert usage.cached_prompt_tokens is None

    @pytest.mark.asyncio
    async def test_no_cache_metadata_leaves_fields_none(self) -> None:
        """Absent cache metadata must NOT be fabricated to zero."""
        from elspeth.web.composer.service import _token_usage_from_response

        response = self._response_with_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        usage = _token_usage_from_response(response)
        assert usage.cached_prompt_tokens is None
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    @pytest.mark.asyncio
    async def test_cache_fields_propagate_through_compose_to_llm_call_record(self) -> None:
        """End-to-end: cache fields appear on the buffered ComposerLLMCall record.

        Patches the LLM call to return a response with OpenAI-shape cache
        metadata, then checks the recorded ComposerLLMCall on the
        ``ComposerResult.llm_calls`` tuple.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        response = self._response_with_usage(
            {
                "prompt_tokens": 500,
                "completion_tokens": 12,
                "total_tokens": 512,
                "prompt_tokens_details": {"cached_tokens": 256},
            }
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = response
            result = await service.compose("Hi", [], state)

        assert len(result.llm_calls) == 1
        call_record = result.llm_calls[0]
        assert call_record.prompt_tokens == 500
        assert call_record.cached_prompt_tokens == 256
        assert call_record.cache_creation_input_tokens is None
        assert call_record.cache_read_input_tokens is None


class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_build_messages_returns_new_list(self) -> None:
        """_build_messages must return a new list on every call."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        msgs1 = service._build_messages([], state, "Hello")
        msgs2 = service._build_messages([], state, "Hello")

        assert msgs1 is not msgs2  # different list objects
        assert msgs1 == msgs2  # same content

    @pytest.mark.asyncio
    async def test_build_messages_oserror_redacts_filename(self) -> None:
        """OSError from deployment-skill loading MUST NOT leak its filename.

        ``str(OSError)`` expands to "[Errno N] <strerror>: '<absolute
        path>'".  That filename reveals the operator's data-dir layout
        and — when the deployment skill lives under a user-scoped
        subdirectory — the user identifier itself.  The wrapper
        ``ComposerServiceError`` flows into the 502 response body in
        ``sessions/routes.py::send_message`` and ``recompose``, so the
        message MUST contain only the exception class name.

        This test pins the redaction contract: the ``str(exc)`` of the
        raised ``ComposerServiceError`` contains no substring of the
        provoking OSError's filename or its strerror text.  Mirrors
        the regression assertion on the SQLAlchemy-family 422 path in
        ``_handle_convergence_error`` (web/sessions/routes.py).
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        secret_path = "/var/lib/elspeth/users/alice/skills/secret-deployment.md"

        def _raise_oserror(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
            raise PermissionError(13, "Permission denied", secret_path)

        with (
            patch("elspeth.web.composer.service.build_messages", side_effect=_raise_oserror),
            pytest.raises(ComposerServiceError) as excinfo,
        ):
            service._build_messages([], state, "Hi")

        body = str(excinfo.value)
        # The filename MUST NOT appear in the wrapper message. Test
        # against the full path AND its directory fragments, because
        # partial leaks (e.g. "/var/lib/elspeth/users/alice") are just
        # as damaging as the full path.
        assert secret_path not in body
        assert "alice" not in body
        assert "Permission denied" not in body
        # The class name IS part of the safe surface — operators
        # reading the 502 still need to distinguish PermissionError
        # from IsADirectoryError from FileNotFoundError.
        assert "PermissionError" in body
        # __cause__ preservation: full detail reaches server-side
        # machinery even though the HTTP body is redacted.
        assert isinstance(excinfo.value.__cause__, PermissionError)
        assert excinfo.value.__cause__.filename == secret_path


class TestComposerMultipleToolCallsPerTurn:
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_turn(self) -> None:
        """LLM returns multiple tool calls in one response — all executed."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: two tool calls in one response
        multi_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                },
                {
                    "id": "call_2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Dual Call Pipeline"}},
                },
            ],
        )
        # Turn 2: text
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [multi_call, text]
            result = await service.compose("Setup", [], state)

        assert result.state.source is not None
        assert result.state.metadata.name == "Dual Call Pipeline"
        assert result.state.version == 3  # two mutations


class TestDiscoveryCache:
    """Tests for the discovery cache (F1)."""

    @pytest.mark.asyncio
    async def test_cacheable_tool_returns_cached_result(self) -> None:
        """Repeated cacheable discovery calls return cached results
        without incrementing any budget counter."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=2)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: list_sources (first call — executes, charges discovery: 1/2)
        # Turn 2: list_sources AGAIN (cache hit — no budget charge)
        # Turn 3: text
        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_sources", "arguments": {}}],
        )
        text = _make_llm_response(content="Found sources.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2, text]
            result = await service.compose("List sources", [], state)

        # Should NOT have raised — second list_sources was a cache hit
        assert result.message == "Found sources."
        # Catalog list_sources is called once by build_messages (prompt
        # context) and once by execute_tool (first discovery call).
        # The second discovery call is a cache hit — no catalog call.
        # Total: 2, not 3.
        assert catalog.list_sources.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_hit_rebuilds_result_envelope_from_current_state(self) -> None:
        """Cacheable discovery data is reused, but validation/version stay current."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        mutate = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c3", "name": "list_sources", "arguments": {}}],
        )
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, mutate, disc2, text]
            result = await service.compose("Build", [], state)

        assert result.state.version == 2
        cached_tool_message = mock_llm.call_args_list[3][0][0][-1]
        cached_payload = json.loads(cached_tool_message["content"])
        assert cached_payload["version"] == 2
        expected_validation = ToolResult(
            success=True,
            updated_state=result.state,
            validation=result.state.validate(),
            affected_nodes=(),
        ).to_dict()["validation"]
        assert cached_payload["validation"] == expected_validation
        assert catalog.list_sources.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_key_includes_arguments(self) -> None:
        """Different arguments = different cache entries = both execute."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        schema1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "get_plugin_schema",
                    "arguments": {"plugin_type": "source", "name": "csv"},
                }
            ],
        )
        schema2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "get_plugin_schema",
                    "arguments": {"plugin_type": "source", "name": "json"},
                }
            ],
        )
        text = _make_llm_response(content="Got schemas.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [schema1, schema2, text]
            await service.compose("Get schemas", [], state)

        # Both calls should have executed (different arguments)
        assert catalog.get_schema.call_count == 2

    @pytest.mark.asyncio
    async def test_mutation_tools_never_cached(self) -> None:
        """Mutation tool results are never cached — always execute."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        mut1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "X"}},
                }
            ],
        )
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Y"}},
                }
            ],
        )
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut1, mut2, text]
            result = await service.compose("Update metadata", [], state)

        assert result.state.metadata.name == "Y"


class TestComposeTimeout:
    """Tests for the server-side compose timeout (F1)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_convergence_error(self) -> None:
        """Exceeding composer_timeout_seconds raises ComposerConvergenceError
        with budget_exhausted='timeout'."""
        import asyncio

        catalog = _mock_catalog()
        settings = _make_settings(composer_timeout_seconds=0.1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        async def slow_llm(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="Too late.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = slow_llm
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Slow pipeline", [], state)
            assert exc_info.value.budget_exhausted == "timeout"

    @pytest.mark.asyncio
    async def test_mutation_tool_state_preserved_on_timeout(self) -> None:
        """Mutation tools that complete before timeout must have their
        state reflected in partial_state.

        Regression test for the cancel-safety concern: with cooperative
        timeout, the deadline is checked AFTER tool execution completes,
        so side effects and state publication are never split. The
        partial_state must include the mutation that completed.
        """
        import time

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        call_count = 0

        def _slow_mutation_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            # Simulate a blob mutation that takes time
            time.sleep(0.2)
            from elspeth.web.composer.state import SourceSpec

            new_state = current_state.with_source(
                SourceSpec(
                    plugin="csv",
                    on_success="out",
                    options={"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                    on_validation_failure="quarantine",
                )
            )
            return ToolResult(
                success=True,
                updated_state=new_state,
                validation=new_state.validate(),
                affected_nodes=("source",),
                data=None,
            )

        async def first_tool_then_timeout_llm(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return tool call (fast)
                return _make_llm_response(
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "set_source",
                            "arguments": {
                                "plugin": "csv",
                                "on_success": "out",
                                "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                                "on_validation_failure": "quarantine",
                            },
                        }
                    ],
                )
            # Second call: exercise _call_llm_before_deadline's timeout
            # capture path without depending on wall-clock scheduling.
            raise TimeoutError

        with (
            patch.object(service, "_call_llm", new=first_tool_then_timeout_llm),
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=_slow_mutation_tool,
            ),
            pytest.raises(ComposerConvergenceError) as exc_info,
        ):
            await service.compose("Build pipeline", [], state)

        assert exc_info.value.budget_exhausted == "timeout"
        assert call_count == 2
        # The mutation tool completed BEFORE the timeout fired on the
        # second LLM call.  With cooperative timeout, partial_state must
        # reflect the completed mutation.
        assert exc_info.value.partial_state is not None, (
            "partial_state is None — mutation tool's state was lost on timeout. "
            "This is the cancel-safety regression: side effects committed but "
            "state was not published."
        )
        assert exc_info.value.partial_state.source is not None
        assert exc_info.value.partial_state.source.plugin == "csv"


class TestConvergenceProgressDispatch:
    """Each ComposerConvergenceError sub-cause must surface a discriminated
    progress event through the sink — covering the contract that fixes
    elspeth-5030f7373d.

    The original symptom: wall-clock timeout, mutation-turn budget, and
    discovery-turn budget all collapsed into one generic ``phase: failed``
    event with the same headline / evidence / likely_next text. These tests
    end-to-end through ``compose()`` to confirm the sink receives the
    discriminated event for each budget value.
    """

    @pytest.mark.asyncio
    async def test_composition_budget_emits_distinct_failure_event(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        mut1 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "x"}},
                }
            ],
        )
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "y"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut1, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("loop forever", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "composition"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_composition_budget"
        assert "mutation turn budget" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_discovery_budget_emits_distinct_failure_event(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("discover forever", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "discovery"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_discovery_budget"
        assert "discovery turn budget" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_emits_distinct_failure_event(self) -> None:
        import asyncio

        catalog = _mock_catalog()
        settings = _make_settings(composer_timeout_seconds=0.1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        progress_events: list[ComposerProgressEvent] = []

        async def record_progress(event: ComposerProgressEvent) -> None:
            progress_events.append(event)

        async def slow_llm(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="Too late.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = slow_llm
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("slow pipeline", [], state, progress=record_progress)
        assert exc_info.value.budget_exhausted == "timeout"

        failed_events = [e for e in progress_events if e.phase == "failed"]
        assert len(failed_events) == 1
        assert failed_events[0].reason == "convergence_wall_clock_timeout"
        assert "timed out" in failed_events[0].headline.lower()

    @pytest.mark.asyncio
    async def test_three_sub_causes_produce_three_distinct_progress_reasons(self) -> None:
        """Regression guard against UX-text collapse — the three convergence
        sub-causes must surface three distinct reason codes via the sink."""
        import asyncio
        import contextlib

        async def collect_failure_reason(
            settings: Any,
            llm_side_effect: Any,
        ) -> str | None:
            catalog = _mock_catalog()
            service = ComposerServiceImpl(catalog=catalog, settings=settings)
            state = _empty_state()
            events: list[ComposerProgressEvent] = []

            async def record(event: ComposerProgressEvent) -> None:
                events.append(event)

            with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = llm_side_effect
                with contextlib.suppress(ComposerConvergenceError):
                    await service.compose("x", [], state, progress=record)
            failed = [e for e in events if e.phase == "failed"]
            return failed[-1].reason if failed else None

        composition_reason = await collect_failure_reason(
            _make_settings(composer_max_composition_turns=1),
            [
                _make_llm_response(tool_calls=[{"id": "c1", "name": "set_metadata", "arguments": {"patch": {"name": "x"}}}]),
                _make_llm_response(tool_calls=[{"id": "c2", "name": "set_metadata", "arguments": {"patch": {"name": "y"}}}]),
            ],
        )
        discovery_reason = await collect_failure_reason(
            _make_settings(composer_max_discovery_turns=1),
            [
                _make_llm_response(tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}]),
                _make_llm_response(tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}]),
            ],
        )

        async def slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)
            return _make_llm_response(content="late")

        timeout_reason = await collect_failure_reason(
            _make_settings(composer_timeout_seconds=0.1),
            slow,
        )

        codes = {composition_reason, discovery_reason, timeout_reason}
        assert len(codes) == 3, (
            "Convergence sub-causes collapsed into fewer than three distinct progress reasons "
            "— elspeth-5030f7373d regression. Got: " + repr(codes)
        )


class TestPartialStatePreservation:
    """Tests for partial state preservation on convergence failure (F2)."""

    @pytest.mark.asyncio
    async def test_convergence_includes_partial_state_when_mutated(self) -> None:
        """When mutations occurred before convergence failure,
        partial_state is attached to the exception."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: mutation (set_source) — composition budget exhausted (1/1)
        mut = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "t1",
                        "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Bonus call also returns tool calls — convergence error
        mut2 = _make_llm_response(
            tool_calls=[
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "nope"}},
                }
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut, mut2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Build pipeline", [], state)

            assert exc_info.value.partial_state is not None
            assert exc_info.value.partial_state.source is not None
            assert exc_info.value.partial_state.source.plugin == "csv"
            assert exc_info.value.partial_state.version == 2

    @pytest.mark.asyncio
    async def test_convergence_no_partial_state_when_no_mutations(self) -> None:
        """When no mutations occurred, partial_state is None."""
        catalog = _mock_catalog()
        settings = _make_settings(composer_max_discovery_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_transforms", "arguments": {}}],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Just looking", [], state)

            assert exc_info.value.partial_state is None


class TestComposerSamplingDeterminism:
    """Composer LLM calls must use temperature=0.0 and seed=42.

    RGR investigation 2026-05-06 §4.4: uncontrolled sampling at the
    LiteLLM/OpenRouter default (~1.0) was the largest single explanation
    for schema-construction nondeterminism. Pipeline composition is an
    extraction task, not a creative one — temperature 0 is the right
    setpoint per the LLM-debugging-skill temperature guide.

    Audit primacy: the values that flow to LiteLLM are also recorded on
    each ComposerLLMCall, so a reviewer can correlate any individual
    failure with the precise sampling regime that produced it.
    """

    @pytest.mark.asyncio
    async def test_call_llm_passes_temperature_zero_and_seed_42(self) -> None:
        """The tool-loop LLM call site sends temperature=0.0 and seed=42."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Single text-only response converges the loop immediately.
        completion = _make_llm_response(content="acknowledged")

        # Service may reject because no pipeline was built; we only
        # care about what reached LiteLLM on the first (and possibly only) call.
        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=completion,
            ) as mock_acomp,
            contextlib.suppress(ComposerServiceError),
        ):
            await service.compose("Hello", [], state)

        assert mock_acomp.call_count >= 1
        first_call_kwargs = mock_acomp.call_args_list[0].kwargs
        assert first_call_kwargs["temperature"] == 0.0, f"composer must send temperature=0.0, got {first_call_kwargs.get('temperature')!r}"
        assert first_call_kwargs["seed"] == 42, f"composer must send seed=42, got {first_call_kwargs.get('seed')!r}"

    @pytest.mark.asyncio
    async def test_call_text_llm_passes_temperature_zero_and_seed_42(self) -> None:
        """The diagnostics text-LLM call site also sends temperature=0.0 and seed=42.

        run-diagnostics text generation goes through _call_text_llm. Since the
        skill prescribes determinism for the user's LLM transforms, the host
        composer should be at least as deterministic.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)

        completion = _make_llm_response(content="diagnostic text")

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=completion,
        ) as mock_acomp:
            await service._call_text_llm([{"role": "user", "content": "explain"}])

        assert mock_acomp.call_count == 1
        kwargs = mock_acomp.call_args_list[0].kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == 42


class TestEmptyChoicesValidation:
    """Tier 3 boundary: LiteLLM can return empty choices."""

    @pytest.mark.asyncio
    async def test_empty_choices_raises_service_error(self) -> None:
        """LiteLLM returning empty choices must raise ComposerServiceError.

        Empty choices can occur on content-filter blocks, rate-limit
        responses, or malformed upstream responses.  Without validation,
        this causes an IndexError at response.choices[0].message.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Patch the lazy LiteLLM wrapper (not _call_llm) so the validation
        # inside _call_llm is exercised through the production code path.
        empty_response = FakeLLMResponse(choices=[])
        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                return_value=empty_response,
            ),
            pytest.raises(ComposerServiceError, match="empty choices"),
        ):
            await service.compose("Hello", [], state)

    @pytest.mark.asyncio
    async def test_empty_choices_on_bonus_turn_raises_service_error(self) -> None:
        """Empty choices on the bonus turn (budget exhaustion) also raises.

        The bonus turn at composition budget exhaustion goes through the
        same _call_llm() path, so the validation protects both sites.
        """
        catalog = _mock_catalog()
        # Budget of 1 composition turn — first mutation exhausts it,
        # then the bonus _call_llm returns empty choices.
        settings = _make_settings(composer_max_composition_turns=1)
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # First call: valid response with a mutation tool call
        mutation_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        # Second call (bonus turn): empty choices
        empty_response = FakeLLMResponse(choices=[])

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=[mutation_call, empty_response],
            ) as mock_acomp,
            pytest.raises(ComposerServiceError, match="empty choices"),
        ):
            await service.compose("Setup CSV", [], state)

        # Confirm both LLM calls happened — the error is from the bonus
        # turn (second call), not from a tool handler fault on the first.
        assert mock_acomp.call_count == 2


class TestComposerAvailabilityAndBadRequest:
    """Readiness and LiteLLM bad-request failures are normalized by the service."""

    @pytest.mark.asyncio
    async def test_unavailable_composer_short_circuits_before_llm_call(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        service._availability = ComposerAvailability(
            available=False,
            model="bad-model",
            provider=None,
            reason="Composer model bad-model is unavailable.",
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            pytest.raises(ComposerServiceError, match="bad-model is unavailable"),
        ):
            mock_llm.return_value = _make_llm_response(content="unexpected")
            await service.compose("Hello", [], _empty_state())

        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_litellm_bad_request_raises_redacted_service_error(self) -> None:
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        bad_request = LiteLLMBadRequestError(
            message="bad model leaked-detail",
            model="bad-model",
            llm_provider="bad-provider",
        )

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=bad_request,
            ),
            pytest.raises(ComposerServiceError) as exc_info,
        ):
            await service.compose("Hello", [], state)

        assert str(exc_info.value) == "LLM request rejected (BadRequestError)"
        assert "leaked-detail" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_litellm_api_error_is_retried_before_unavailable(self) -> None:
        from litellm.exceptions import APIError as LiteLLMAPIError

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        transient_error = LiteLLMAPIError(
            status_code=503,
            message="provider temporarily unavailable leaked-detail",
            model="openrouter/openai/gpt-5.5",
            llm_provider="openrouter",
        )
        success = _make_llm_response(content="Recovered.")

        with (
            patch(
                "elspeth.web.composer.service._litellm_acompletion",
                new_callable=AsyncMock,
                side_effect=[transient_error, success],
            ) as mock_llm,
            patch("elspeth.web.composer.service.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await service.compose("Hello", [], state)

        assert result.message == "Recovered."
        assert mock_llm.call_count == 2
        mock_sleep.assert_awaited_once()


class TestPluginBugCrashesFromToolExecution:
    """Plugin-internal TypeError/ValueError/UnicodeError must crash.

    The compose loop catches ONLY ToolArgumentError around execute_tool.
    Any other TypeError/ValueError/UnicodeError is a plugin bug — per
    CLAUDE.md, silently laundering a plugin bug as an LLM-argument error
    is worse than crashing, because the audit trail records a confident
    but wrong Tier-3 story.

    Mirrors test_internal_key_error_is_not_swallowed.
    """

    @pytest.mark.asyncio
    async def test_plugin_value_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("invalid expression syntax — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        # Crash on first tool call → no prior mutations → partial_state is None.
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)
        assert exc_info.value.exc_class == "ValueError"

    @pytest.mark.asyncio
    async def test_plugin_type_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=TypeError("NoneType + int — plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, TypeError)
        assert "plugin bug" in str(exc_info.value.original_exc)
        assert exc_info.value.exc_class == "TypeError"

    @pytest.mark.asyncio
    async def test_plugin_unicode_error_is_not_swallowed(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "plugin bug"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)
        assert exc_info.value.partial_state is None
        assert isinstance(exc_info.value.original_exc, UnicodeDecodeError)
        assert exc_info.value.exc_class == "UnicodeDecodeError"

    @pytest.mark.asyncio
    async def test_plugin_crash_after_successful_mutation_preserves_partial_state(
        self,
    ) -> None:
        """When a plugin crashes AFTER at least one prior mutation succeeded
        in the same request, ``ComposerPluginCrashError.partial_state`` MUST
        carry the accumulated post-mutation state so the route handler can
        persist it into composition_states.

        This closes the P1 regression flagged in review: the narrowed
        ``except`` in compose() used to re-raise bare exceptions without
        threading the loop-local ``state``, so any successful mutations
        prior to the crash were silently dropped and recompose restarted
        from the stale pre-request state.
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        initial_version = state.version

        # Two tool calls in a single LLM turn: first succeeds (mutates
        # state), second raises a plugin-bug exception.
        two_calls = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                },
                {
                    "id": "c2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "after-mutation"}},
                },
            ],
        )

        mutated_state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(name="after-mutation"),
            version=initial_version + 1,
        )
        successful_result = ToolResult(
            success=True,
            updated_state=mutated_state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=(),
        )

        call_count = {"n": 0}

        def _fake_execute_tool(*args: Any, **kwargs: Any) -> ToolResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return successful_result
            raise ValueError("plugin bug: crash AFTER first mutation")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=_fake_execute_tool,
            ),
        ):
            mock_llm.return_value = two_calls
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state)

        assert call_count["n"] == 2, "both tool calls should have been attempted"
        crash = exc_info.value
        assert crash.partial_state is not None, "partial_state MUST be populated when a mutation succeeded before the crash"
        assert crash.partial_state.version == initial_version + 1
        assert crash.partial_state.metadata.name == "after-mutation"
        assert isinstance(crash.original_exc, ValueError)
        assert crash.exc_class == "ValueError"

    @pytest.mark.asyncio
    async def test_tool_argument_error_is_caught_and_fed_to_llm(self) -> None:
        """Positive case: ToolArgumentError IS caught, error fed back for LLM retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it, trying again.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ToolArgumentError(
                    argument="plugin",
                    expected="a string",
                    actual_type="int",
                ),
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            result = await service.compose("Setup", [], state)

        assert isinstance(result, ComposerResult)
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "'plugin' must be a string, got int" in error_payload["error"]

    @pytest.mark.asyncio
    async def test_tool_argument_error_subclass_cannot_leak_cause_to_llm(self) -> None:
        """Defense-in-depth: if a subclass overrides __str__ to embed the
        __cause__ chain, the LLM-echo path must still use args[0] only.

        Simulates a future regression where a helpful-looking subclass does
        `def __str__(self): return f"{self.args[0]}: caused by {self.__cause__}"`.
        A DB URL or file path leaked through __cause__ would then reach the
        LLM API. The compose loop MUST short-circuit __str__ and emit
        args[0] verbatim, isolating the cause chain to __cause__ (audit-only).
        """

        class LeakyToolArgumentError(ToolArgumentError):
            def __str__(self) -> str:
                return f"{self.args[0]}: caused by {self.__cause__}"

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        secret_path = "/etc/elspeth/secrets/bootstrap.key"
        secret_cause = ValueError(f"bad path: {secret_path}")
        leaky = LeakyToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        leaky.__cause__ = secret_cause

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Got it.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=leaky,
            ),
        ):
            mock_llm.side_effect = [valid_call, text]
            await service.compose("Setup", [], state)

        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_payload = json.loads(tool_messages[0]["content"])
        assert "'content' must be a string, got int" in error_payload["error"]
        # The crucial assertion: the cause-chain content NEVER appears.
        assert secret_path not in error_payload["error"]
        assert "caused by" not in error_payload["error"]


class TestPluginCrashSessionPersistence:
    """Plugin-bug crash must leave a durable session-row breadcrumb.

    "No silent drops" for session records: a plugin crash that leaves
    the session in no recorded terminal state is as bad for audit
    integrity as the laundering behaviour this plan eliminates.

    Given the current sessions_table schema (no status / crashed_at /
    last_exc_class columns), the breadcrumb is a bump of updated_at.
    This test asserts that bump, plus the invariant that NO exception
    message leaks into any column. The follow-up filigree issue tracks
    the schema migration that adds richer crash markers.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        # Seed the sessions row with a DELIBERATELY OLD updated_at so the
        # crash-path bump is unambiguously distinguishable from the seed.
        self.seeded_at = datetime(2020, 1, 1, tzinfo=UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=self.seeded_at,
                    updated_at=self.seeded_at,
                )
            )

    @pytest.mark.asyncio
    async def test_plugin_crash_bumps_session_updated_at(self) -> None:
        from elspeth.web.sessions.models import sessions_table

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("plugin bug: /etc/secrets/bootstrap.key is bad"),
            ),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        # The underlying plugin exception is preserved on the wrapper.
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)

        # Assertion 1: session row was touched on the crash path.
        with self.engine.begin() as conn:
            row = conn.execute(sessions_table.select().where(sessions_table.c.id == self.session_id)).one()

        # SQLite DateTime(timezone=True) strips tzinfo on read — normalize
        # both sides of the comparison to the same tz-naive representation.
        row_updated_at = row.updated_at
        if row_updated_at.tzinfo is None:
            seed_for_compare = self.seeded_at.replace(tzinfo=None)
        else:
            seed_for_compare = self.seeded_at
        assert row_updated_at > seed_for_compare, "crash path must bump updated_at as audit breadcrumb"

        # Assertion 2: NO column holds the exception message. Stringify
        # the entire row and verify secret fragments / class hints are
        # absent. This is the load-bearing audit-integrity invariant —
        # if a future refactor adds a 'last_error' column, the assertion
        # will catch any attempt to persist the raw message.
        row_text = " | ".join(str(v) for v in row._mapping.values())
        assert "plugin bug" not in row_text
        assert "/etc/secrets" not in row_text
        assert "ValueError" not in row_text

    @pytest.mark.asyncio
    async def test_persist_crashed_session_failure_does_not_mask_plugin_bug(
        self,
    ) -> None:
        """If _persist_crashed_session itself raises a recoverable audit-path
        exception (SQLAlchemyError / OSError), slog.error fires and the
        original plugin-bug exception still propagates unchanged.

        Uses sqlalchemy.exc.OperationalError as the stand-in for a realistic
        DB-write failure (connection drop, locking timeout, disk I/O
        translated to SQLAlchemy layer). The catch at
        service.py ComposerServiceImpl.compose is narrowed to
        (SQLAlchemyError, OSError); substituting RuntimeError here would
        assert the wrong invariant because RuntimeError deliberately
        propagates past this catch (see the programmer-bug companion test).

        Two invariants asserted:
        1. The ORIGINAL ValueError reaches the caller (not the
           OperationalError from the persistence failure).
        2. slog.error is called with the `composer_crash_persistence_failed`
           event — guarantees that an accidental removal of Step 4a-pre's
           structlog import would be caught (without this assertion, a
           regression where slog.error silently fails as NameError would
           pass the test because the original exception still propagates).
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("original plugin bug"),
            ),
            patch.object(
                service,
                "_persist_crashed_session",
                side_effect=OperationalError("UPDATE sessions", {}, Exception("db unavailable")),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        # Original plugin exception survives the wrap.
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "original plugin bug" in str(exc_info.value.original_exc)

        # The crash-persistence-failure slog.error MUST fire. This closes
        # the regression risk where Step 4a-pre's structlog import is
        # accidentally removed — the method would then raise NameError
        # inside the except, masking the original ValueError.
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert len(persistence_failure_events) == 1, cap_logs
        event = persistence_failure_events[0]
        assert event["session_id"] == self.session_id
        assert event["original_exc_class"] == "ValueError"
        # audit_exc_class is the class of the *persistence* failure, not the
        # original plugin bug. Present so operators can distinguish "DB
        # write failed with IntegrityError" from "DB write failed with
        # OperationalError" without needing the traceback.
        assert event["audit_exc_class"] == "OperationalError"
        # No traceback / exception message fields — exc_info was deliberately
        # dropped from this slog call to prevent __cause__-chain secret
        # leakage into server logs.
        assert "exc_info" not in event
        assert "exception" not in event
        assert "stack_info" not in event
        # Exception messages MUST NOT appear anywhere in the structured
        # event (defense-in-depth against accidental re-addition of a
        # message= field in a future refactor).
        assert "original plugin bug" not in str(event)
        # The OperationalError carries its SQL statement and __cause__
        # ("db unavailable") — neither may appear in the structured event.
        assert "db unavailable" not in str(event)
        assert "UPDATE sessions" not in str(event)

    @pytest.mark.asyncio
    async def test_persist_crashed_session_real_path_slog_emission(self) -> None:
        """Smoke test for Step 4a-pre: exercise the real _persist_crashed_session
        path (no patching of the private method).  If structlog is not
        imported in service.py, this test will surface the NameError that
        `test_persist_crashed_session_failure_does_not_mask_plugin_bug`
        misses (because that test patches the method itself).

        The real _persist_crashed_session should succeed here (the sessions
        engine is live), so we assert the crash propagates without any
        persistence-failure slog event.
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("plugin bug"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)
        assert isinstance(exc_info.value.original_exc, ValueError)
        assert "plugin bug" in str(exc_info.value.original_exc)

        # No persistence-failure event — the real path succeeded.
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert persistence_failure_events == [], cap_logs

    @pytest.mark.asyncio
    async def test_persist_crashed_session_programmer_bug_propagates_past_catch(
        self,
    ) -> None:
        """Programmer-bug exceptions inside _persist_crashed_session MUST NOT
        be absorbed by the audit-cleanup catch in compose().

        This test is the guardrail for the narrowed catch at
        ComposerServiceImpl.compose: replacing ``except Exception`` with
        ``except (SQLAlchemyError, OSError)`` means AttributeError, TypeError,
        AssertionError, NameError and the like now escape the handler.
        A future regression that re-widens the catch (e.g., "catch everything
        so audit never crashes the request") would silently pass the sibling
        ``test_persist_crashed_session_failure_does_not_mask_plugin_bug``
        test because that path raises an audit-family exception. This test
        closes the loop by asserting AttributeError — a canonical Tier 1/2
        programmer bug — bubbles out of the compose() call unchanged, NOT
        wrapped as ComposerPluginCrashError and NOT logged as
        ``composer_crash_persistence_failed``.

        The original plugin-bug ValueError becomes the ``__context__`` of the
        escaping AttributeError because Python chains implicit exception
        context through the re-raise site; we do not assert on ``__context__``
        directly since that coupling is an implementation detail, but we do
        verify the headline exception type flipped from
        ComposerPluginCrashError to AttributeError.
        """
        from structlog.testing import capture_logs

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("original plugin bug"),
            ),
            patch.object(
                service,
                "_persist_crashed_session",
                side_effect=AttributeError("sessions_table has no attribute 'c'"),
            ),
            capture_logs() as cap_logs,
        ):
            mock_llm.return_value = valid_call
            # AttributeError escapes the narrowed catch; the outer
            # ComposerPluginCrashError is never re-raised because the
            # audit-site AttributeError propagates first.
            with pytest.raises(AttributeError) as exc_info:
                await service.compose("Setup", [], state, session_id=self.session_id)

        assert "sessions_table" in str(exc_info.value)

        # No slog event — the catch did not fire, so the structured-logging
        # path was not reached. A regression that re-widens the catch would
        # cause this assertion to fail (the event would appear).
        persistence_failure_events = [entry for entry in cap_logs if entry.get("event") == "composer_crash_persistence_failed"]
        assert persistence_failure_events == [], cap_logs

    @pytest.mark.asyncio
    async def test_persist_crashed_session_runs_off_event_loop(self) -> None:
        """_persist_crashed_session must execute in a worker thread, not
        on the event loop thread.

        The method performs a synchronous ``Engine.begin()`` + UPDATE,
        which holds the GIL and (more importantly) blocks the asyncio
        event loop for the duration of the DB round-trip. Every other
        sync DB path in the compose flow is already wrapped in
        ``asyncio.to_thread(...)``; the crash-path persistence was
        hoisted out of the main loop but not wrapped.

        Blast radius: a stalled persist blocks websocket heartbeats,
        rate-limit checks, and the per-session progress broadcasts for
        every concurrent request. Cold path, but the partial DoS
        matches the same class of regression that the tool-execution
        offloading test already guards against.
        """
        import threading

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        event_loop_thread = threading.current_thread()
        persist_thread: threading.Thread | None = None

        original_persist = service._persist_crashed_session

        def capture_thread(session_id: str) -> None:
            nonlocal persist_thread
            persist_thread = threading.current_thread()
            original_persist(session_id)

        valid_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=ValueError("plugin bug"),
            ),
            patch.object(service, "_persist_crashed_session", side_effect=capture_thread),
        ):
            mock_llm.return_value = valid_call
            with pytest.raises(ComposerPluginCrashError):
                await service.compose("Setup", [], state, session_id=self.session_id)

        assert persist_thread is not None, "_persist_crashed_session was never called"
        assert persist_thread is not event_loop_thread, (
            "_persist_crashed_session ran on the event loop thread — "
            "the synchronous Engine.begin() call blocks all concurrent "
            "requests. It must be offloaded via asyncio.to_thread(...)"
        )


class TestToolExecutionThreadOffloading:
    """execute_tool() must run in a worker thread, not the event loop thread.

    Tests capture actual thread identity rather than checking whether
    asyncio.to_thread was called — testing the behavioral property
    (event loop not blocked) regardless of the offloading mechanism.
    """

    @staticmethod
    async def _assert_tool_runs_off_event_loop(
        tool_call_response: FakeLLMResponse,
        text_response: FakeLLMResponse,
        user_message: str,
    ) -> None:
        """Shared helper: verify a tool call executes in a worker thread."""
        import threading

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        event_loop_thread = threading.current_thread()
        tool_execution_thread: threading.Thread | None = None

        def _capture_thread(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            nonlocal tool_execution_thread
            tool_execution_thread = threading.current_thread()
            return ToolResult(
                success=True,
                updated_state=current_state,
                validation=current_state.validate(),
                affected_nodes=(),
                data={"sources": []},
            )

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=_capture_thread,
            ),
        ):
            mock_llm.side_effect = [tool_call_response, text_response]
            await service.compose(user_message, [], state)

        assert tool_execution_thread is not None, "execute_tool was never called"
        assert tool_execution_thread is not event_loop_thread, (
            "execute_tool ran on the event loop thread — must be offloaded to a worker thread to avoid blocking"
        )

    @pytest.mark.asyncio
    async def test_discovery_tool_runs_off_event_loop_thread(self) -> None:
        """Discovery tools run in a worker thread (read-only I/O)."""
        await self._assert_tool_runs_off_event_loop(
            tool_call_response=_make_llm_response(
                tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
            ),
            text_response=_make_llm_response(content="Here are the sources."),
            user_message="List sources",
        )

    @pytest.mark.asyncio
    async def test_mutation_tool_runs_off_event_loop_thread(self) -> None:
        """Mutation tools run in a worker thread (blob/secret I/O).

        Previously only discovery tools were offloaded; mutation tools
        ran synchronously on the event loop, blocking all concurrent
        requests in the single-process server.
        """
        await self._assert_tool_runs_off_event_loop(
            tool_call_response=_make_llm_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "set_source",
                        "arguments": {
                            "plugin": "csv",
                            "on_success": "out",
                            "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                            "on_validation_failure": "quarantine",
                        },
                    }
                ],
            ),
            text_response=_make_llm_response(content="Source configured."),
            user_message="Set CSV source",
        )

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_tool_execution(self) -> None:
        """Heartbeat regression: compose() must not block the event loop.

        Runs compose() alongside an async heartbeat coroutine. If the
        heartbeat fires on schedule (not delayed by blocking tool work),
        the event loop was free during tool execution.
        """
        import time

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Blocking duration must be much larger than the gap threshold
        # to avoid false positives on slow/shared CI runners where OS
        # scheduler jitter can delay asyncio.sleep wakeups by 50-100ms.
        tool_block_seconds = 1.0
        heartbeat_interval = 0.05

        def _blocking_tool(
            _tool_name: str,
            _arguments: dict[str, Any],
            current_state: CompositionState,
            _catalog: Any,
            **kwargs: Any,
        ) -> ToolResult:
            time.sleep(tool_block_seconds)
            return ToolResult(
                success=True,
                updated_state=current_state,
                validation=current_state.validate(),
                affected_nodes=("source",),
                data=None,
            )

        heartbeat_times: list[float] = []

        async def heartbeat() -> None:
            while True:
                heartbeat_times.append(time.monotonic())
                await asyncio.sleep(heartbeat_interval)

        # Use a mutation tool — the original bug was specifically about
        # mutation tools running synchronously on the event loop.
        tool_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "c1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv",
                        "on_success": "out",
                        "options": {"path": "/data/blobs/f.csv", "schema": {"mode": "observed"}},
                        "on_validation_failure": "quarantine",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Done.")

        with (
            patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "elspeth.web.composer.service.execute_tool",
                side_effect=_blocking_tool,
            ),
        ):
            mock_llm.side_effect = [tool_call, text]
            hb_task = asyncio.create_task(heartbeat())
            try:
                await service.compose("List sources", [], state)
            finally:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hb_task

        # With 1.0s block and 50ms interval we expect ~20 heartbeats.
        # Require at least 4 to catch partial stalls, not just total seizure.
        min_expected = int(tool_block_seconds / heartbeat_interval) - 2
        assert len(heartbeat_times) >= min(min_expected, 4), (
            f"Only {len(heartbeat_times)} heartbeat(s) fired during {tool_block_seconds}s tool execution — event loop was likely blocked"
        )

        # Check that no heartbeat interval exceeds a generous threshold.
        # If the event loop were blocked, one interval would be ≈ tool_block_seconds.
        # The 5x multiplier (250ms) gives wide margin for OS scheduler jitter
        # on shared CI runners while still catching a 1.0s event loop block.
        max_allowed_gap = heartbeat_interval * 5  # 250ms threshold vs 1.0s block (4x safety)
        for i in range(1, len(heartbeat_times)):
            gap = heartbeat_times[i] - heartbeat_times[i - 1]
            assert gap < max_allowed_gap, (
                f"Heartbeat gap {gap:.3f}s exceeds {max_allowed_gap:.3f}s — event loop was blocked (tool takes {tool_block_seconds}s)"
            )


class TestToolArgumentError:
    """ToolArgumentError is a composer-domain exception for Tier-3 boundary failures.

    It signals that a tool handler received arguments of the wrong type or
    with semantically invalid values that could not be coerced. The compose
    loop catches this and feeds the message back to the LLM for retry. Any
    OTHER exception escaping execute_tool is a plugin bug and must crash.

    The class is deliberately a structured DTO rather than a free-form
    ``Exception`` subclass: its composed message is echoed verbatim to
    the LLM API AND recorded in the Landscape audit trail, so any
    channel that could carry an LLM-supplied value would be a secret/PII
    leak pathway. Tests below lock in the "safe by construction" shape.
    """

    def test_inherits_from_exception_directly_not_composer_service_error(self) -> None:
        """ToolArgumentError must NOT inherit from ComposerServiceError.

        If it did, the route-level ``except ComposerServiceError`` block
        in ``send_message`` (sessions/routes.py) would silently absorb
        any escaped ToolArgumentError as a 502, recreating the
        silent-laundering channel this plan closes.
        Inheriting from Exception directly ensures an escaped
        ToolArgumentError (a compose-loop bug) surfaces loudly via FastAPI's
        default handler rather than being masked.
        """
        assert issubclass(ToolArgumentError, Exception)
        assert not issubclass(ToolArgumentError, ComposerServiceError)

    def test_structured_fields_compose_canonical_message(self) -> None:
        """Constructor composes args[0] deterministically from the three fields.

        The compose loop reads ``exc.args[0]`` to build the LLM-echo
        payload, so the composition template is a documented wire
        contract — a change here is a change to what the LLM and
        Landscape see.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        assert exc.argument == "content"
        assert exc.expected == "a string"
        assert exc.actual_type == "int"
        assert exc.args[0] == "'content' must be a string, got int"
        assert str(exc) == "'content' must be a string, got int"

    def test_constructor_is_keyword_only(self) -> None:
        """Positional construction must fail — structural leak prevention.

        The whole point of the DTO shape is that there is no way to
        sneak a raw LLM-supplied value into the message. A positional
        ``ToolArgumentError(f"bad: {user_input!r}")`` would defeat
        that. Making the constructor keyword-only forces every call
        site through the three-field safe channel.
        """
        with pytest.raises(TypeError):
            cast(Any, ToolArgumentError)("content must be a string, got int")

    def test_empty_argument_rejected(self) -> None:
        """Blank ``argument`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="argument must be a non-empty"):
            ToolArgumentError(argument="", expected="a string", actual_type="int")

    def test_empty_expected_rejected(self) -> None:
        """Blank ``expected`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="expected must be a non-empty"):
            ToolArgumentError(argument="content", expected="", actual_type="int")

    def test_empty_actual_type_rejected(self) -> None:
        """Blank ``actual_type`` produces a nonsensical audit record and must be rejected."""
        with pytest.raises(ValueError, match="actual_type must be a non-empty"):
            ToolArgumentError(argument="content", expected="a string", actual_type="")

    def test_declared_fields_frozen_after_construction(self) -> None:
        """Declared fields must not be mutable after construction.

        The exception flows into ``composition_states`` / LLM echo as
        an immutable audit artefact. Mirror the _FROZEN_ATTRS pattern
        used by ComposerConvergenceError and ComposerPluginCrashError
        so no intermediate layer can silently rewrite what downstream
        consumers see.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.argument = "other"
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.expected = "a dict"
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.actual_type = "str"

    def test_exception_chain_dunders_remain_writable(self) -> None:
        """__cause__, __context__, __traceback__, __notes__ must stay writable.

        ``raise ... from ...`` and ``add_note()`` rely on these being
        assignable. The freeze guard covers only the three declared
        fields — the rest of the exception machinery must work
        unchanged.
        """
        exc = ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type="int",
        )
        cause = ValueError("deep cause")
        exc.__cause__ = cause
        assert exc.__cause__ is cause
        exc.add_note("diagnostic note")
        assert "diagnostic note" in exc.__notes__

    def test_supports_exception_chaining(self) -> None:
        """raise ToolArgumentError(...) from exc must preserve __cause__.

        Audit-grade error reporting depends on the cause chain surviving
        asyncio.to_thread re-raise and the service-level catch. The
        cause is carried on ``__cause__`` for debug/audit but NEVER
        echoed to the LLM (see test_tool_argument_error_subclass_
        cannot_leak_cause_to_llm).
        """
        original = ValueError("bad input")
        try:
            try:
                raise original
            except ValueError as exc:
                raise ToolArgumentError(
                    argument="content",
                    expected="a string",
                    actual_type="int",
                ) from exc
        except ToolArgumentError as wrapped:
            assert wrapped.__cause__ is original


class TestToolArgumentErrorAcrossThreadBoundary:
    """End-to-end: ToolArgumentError raised inside the worker thread is caught
    correctly by the service-level catch, with message preserved.

    Closes the sleepy-assertion gap in the mocked service-level tests
    (which raise synchronously on the mock and never exercise the real
    asyncio.to_thread re-raise path).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        self.engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(self.engine)

        self.session_id = str(uuid4())
        self.data_dir = tmp_path
        now = datetime.now(UTC)
        with self.engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=self.session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

    @pytest.mark.asyncio
    async def test_real_create_blob_type_guard_feeds_error_to_llm(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "create_blob",
                    "arguments": {
                        "filename": "x.txt",
                        "mime_type": "text/plain",
                        "content": 42,  # wrong type
                    },
                }
            ],
        )
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state, session_id=self.session_id)

        assert result.message == "Fixed."
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        assert "'content' must be a string, got int" in error_content["error"]

    @pytest.mark.asyncio
    async def test_real_set_source_from_blob_options_guard_feeds_error_to_llm(self) -> None:
        from elspeth.web.composer.tools import execute_tool

        catalog = _mock_catalog()
        settings = _make_settings(data_dir=self.data_dir)
        service = ComposerServiceImpl(
            catalog=catalog,
            settings=settings,
            session_engine=self.engine,
        )
        state = _empty_state()

        create_result = execute_tool(
            "create_blob",
            {"filename": "seed.txt", "mime_type": "text/plain", "content": "hello"},
            state,
            catalog,
            data_dir=str(self.data_dir),
            session_engine=self.engine,
            session_id=self.session_id,
        )
        blob_id = create_result.data["blob_id"]

        bad_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_bad",
                    "name": "set_source_from_blob",
                    "arguments": {
                        "blob_id": blob_id,
                        "on_success": "out",
                        "options": "column=text",
                    },
                }
            ],
        )
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state, session_id=self.session_id)

        assert result.message == "Fixed."
        second_call_messages = mock_llm.call_args_list[1].args[0]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        assert "'options' must be an object, got str" in error_content["error"]


class TestComposerErrorConstructionInvariants:
    """Type-level invariants for composer service exceptions.

    These exceptions flow into HTTP responses (as error_type/detail bodies)
    and into Landscape (via partial_state persistence in composition_states
    and structured-log exc_class correlation). Post-construction attribute
    reassignment would let any layer silently rewrite what downstream HTTP
    and audit consumers see. The class-level freeze and the ``capture()``
    classmethod encode the "partial_state only when state.version >
    initial_version" invariant mechanically rather than relying on each
    raise site to apply the rule by hand.
    """

    def test_plugin_crash_error_attributes_are_frozen_after_construction(self) -> None:
        exc = ComposerPluginCrashError(ValueError("boom"), partial_state=None)

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("original_exc", RuntimeError("replaced"))

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("partial_state", _empty_state())

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("exc_class", "PrettyException")

    def test_plugin_crash_error_allows_exception_chain_machinery(self) -> None:
        # __cause__, __context__, __suppress_context__, __traceback__, and
        # add_note() target BaseException-managed slots, not our declared
        # attrs. The freeze MUST NOT break `raise X from Y` or add_note.
        root = RuntimeError("underlying")
        exc = ComposerPluginCrashError(ValueError("boom"))
        exc.__cause__ = root
        exc.__suppress_context__ = True
        exc.add_note("operator triage hint")

        assert exc.__cause__ is root
        assert exc.__suppress_context__ is True
        assert "operator triage hint" in exc.__notes__

    def test_convergence_error_attributes_are_frozen_after_construction(self) -> None:
        exc = ComposerConvergenceError(
            max_turns=3,
            budget_exhausted="composition",
            partial_state=None,
        )

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("max_turns", 99)

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("budget_exhausted", "timeout")

        with pytest.raises(AttributeError, match="frozen"):
            exc.__setattr__("partial_state", _empty_state())

    def test_plugin_crash_capture_returns_none_when_state_not_mutated(self) -> None:
        # Invariant: partial_state is None when state.version == initial_version
        # (no tool call successfully mutated state before the crash).
        state = _empty_state()  # version=1
        exc = ComposerPluginCrashError.capture(
            KeyError("missing"),
            state=state,
            initial_version=state.version,
        )

        assert exc.partial_state is None
        assert exc.exc_class == "KeyError"

    def test_plugin_crash_capture_returns_state_when_mutated(self) -> None:
        # Invariant: partial_state IS the state when state.version moved
        # beyond initial_version (at least one tool call persisted).
        initial = _empty_state()  # version=1
        mutated = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=initial.version + 2,
        )
        exc = ComposerPluginCrashError.capture(
            ValueError("boom"),
            state=mutated,
            initial_version=initial.version,
        )

        assert exc.partial_state is mutated
        assert exc.exc_class == "ValueError"

    def test_convergence_capture_returns_none_when_state_not_mutated(self) -> None:
        state = _empty_state()
        exc = ComposerConvergenceError.capture(
            max_turns=5,
            budget_exhausted="composition",
            state=state,
            initial_version=state.version,
        )

        assert exc.partial_state is None
        assert exc.max_turns == 5
        assert exc.budget_exhausted == "composition"

    def test_convergence_capture_returns_state_when_mutated(self) -> None:
        initial = _empty_state()
        mutated = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=initial.version + 1,
        )
        exc = ComposerConvergenceError.capture(
            max_turns=7,
            budget_exhausted="timeout",
            state=mutated,
            initial_version=initial.version,
        )

        assert exc.partial_state is mutated
        assert exc.budget_exhausted == "timeout"


class TestComposerRuntimePreflightCacheAndTimeout:
    @pytest.mark.asyncio
    async def test_runtime_preflight_cache_reuses_same_state_version_and_settings_hash(self) -> None:
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=preflight) as mock_preflight:
            first = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )
            second = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )

        assert first is preflight
        assert second is preflight
        mock_preflight.assert_called_once_with(state, "user-1")

    @pytest.mark.asyncio
    async def test_runtime_preflight_timeout_is_cached_for_compose_call(self) -> None:
        settings = _make_settings(composer_runtime_preflight_timeout_seconds=0.01)
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        started = threading.Event()
        release = threading.Event()

        def slow_preflight(candidate: CompositionState, user_id: str | None) -> ValidationResult:
            started.set()
            release.wait(timeout=30)
            return ValidationResult(is_valid=True, checks=[], errors=[])

        try:
            with patch.object(service, "_runtime_preflight", side_effect=slow_preflight) as mock_preflight:
                with pytest.raises(ComposerRuntimePreflightError) as first:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )
                assert first.value.exc_class == "TimeoutError"
                assert started.is_set()

                with pytest.raises(ComposerRuntimePreflightError) as second:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )

                assert second.value.exc_class == "TimeoutError"
                mock_preflight.assert_called_once()
        finally:
            release.set()

    def test_runtime_preflight_settings_hash_is_non_secret(self) -> None:
        class FakeSettings:
            data_dir = Path("/tmp/elspeth-data")
            landscape_passphrase = "SECRET_CANARY_SHOULD_NOT_APPEAR"

        digest = runtime_preflight_settings_hash(FakeSettings())

        assert "SECRET_CANARY" not in digest
        assert len(digest) == 64


class TestComposerRuntimePreflightFinalGate:
    @pytest.mark.asyncio
    async def test_changed_state_completion_is_replaced_when_runtime_preflight_fails(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        changed_state = replace(changed_state, version=state.version + 1)

        llm_response = _make_llm_response(content="The pipeline is complete and valid.")
        failed_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                )
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            with patch.object(service, "_runtime_preflight", return_value=failed_preflight) as mock_preflight:
                result = await service._finalize_no_tool_response(
                    content="The pipeline is complete and valid.",
                    state=changed_state,
                    initial_version=state.version,
                    user_id="user-1",
                    last_runtime_preflight=None,
                    runtime_preflight_cache=service._new_runtime_preflight_cache(),
                    session_scope="session:test",
                )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is failed_preflight
        mock_preflight.assert_called_once_with(changed_state, "user-1")

    @pytest.mark.asyncio
    async def test_unchanged_text_without_preview_does_not_run_preflight(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="I can help with that.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "I can help with that."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is None
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchanged_state_reuses_preview_preflight_without_rerun(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        preview_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=preview_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is preview_preflight
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchanged_state_reuses_valid_preview_preflight_without_replacement(self) -> None:
        """Reuse path with is_valid=True must preserve the LLM message verbatim.

        Complement to test_unchanged_state_reuses_preview_preflight_without_rerun:
        that test exercises the invalid-cached branch (which replaces the
        message). This test pins the valid-cached branch (which must keep the
        LLM message intact and not populate raw_assistant_content).
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        passing_preview_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=passing_preview_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "The pipeline is complete and valid."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is passing_preview_preflight
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_passing_preflight_preserves_original_message_verbatim(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)
        passed_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=passed_preflight):
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "The pipeline is complete and valid."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is passed_preflight

    def test_runtime_preflight_failure_message_uses_failed_check_when_errors_empty(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Graph has no path from source to sink",
                )
            ],
            errors=[],
        )

        message = ComposerServiceImpl._runtime_preflight_failure_message(result)

        assert "runtime preflight failed" in message
        assert "Graph has no path from source to sink" in message

    def test_runtime_preflight_failure_message_has_bare_fallback(self) -> None:
        result = ValidationResult(is_valid=False, checks=[], errors=[])

        message = ComposerServiceImpl._runtime_preflight_failure_message(result)

        assert message == "I cannot mark this pipeline complete yet because runtime preflight failed."

    @pytest.mark.asyncio
    async def test_unexpected_preflight_exception_preserves_partial_state(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)

        with (
            patch.object(service, "_runtime_preflight", side_effect=RuntimeError("boom")),
            pytest.raises(ComposerRuntimePreflightError) as exc_info,
        ):
            await service._finalize_no_tool_response(
                content="The pipeline is complete.",
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert exc_info.value.partial_state == changed_state
        assert exc_info.value.exc_class == "RuntimeError"

    def test_runtime_preflight_error_original_exception_is_frozen(self) -> None:
        original = RuntimeError("boom")
        error = ComposerRuntimePreflightError(original_exc=original, partial_state=None)

        with pytest.raises(AttributeError):
            error.original_exc = RuntimeError("replacement")


class TestEmptyStateFinalizePassthrough:
    """Tier 1.5 §7.6 followup — empty-state finalize-time passthrough.

    Audit-DB inspection of three captured rag-text-llm REDs (2026-05-06
    cohort, sessions 2cf59016, 12f061d9, 29ef178e) showed the model spent
    20+ tool calls trying to converge on a valid set_pipeline build, gave
    up, and produced honest prose explaining what it tried and what's
    blocking. The synthesizer was discarding this prose and replacing it
    with raw Pydantic noise ("source: Field required, sinks: Field
    required") that looked like a system bug to a viewer.

    The structural fix: when state is structurally empty (no source, no
    nodes, no outputs) at finalize time AND the cached/computed runtime
    preflight is invalid, the model isn't lying about completion — it's
    reporting honest failure. Pass through its content with a
    system-attributed suffix telling the user what to do next.

    These tests pin the new behaviour. The original synthesizer path
    (non-empty state with invalid preflight) is left unchanged and tested
    by sister tests above.
    """

    # ── Standalone tests on the helpers ─────────────────────────────────

    def test_state_is_structurally_empty_returns_true_for_empty_state(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty
        state = _empty_state()
        assert _state_is_structurally_empty(state) is True

    def test_state_is_structurally_empty_false_with_source(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty
        state = _empty_state().with_source(
            SourceSpec(plugin="csv", on_success="t1", options={"path": "/tmp/x.csv"}, on_validation_failure="discard")
        )
        assert _state_is_structurally_empty(state) is False

    def test_state_is_structurally_empty_false_with_output(self) -> None:
        from elspeth.web.composer.service import _state_is_structurally_empty
        state = _empty_state().with_output(
            OutputSpec(name="main", plugin="csv", options={"path": "/tmp/y.csv", "schema": {"mode": "observed"}}, on_write_failure="discard")
        )
        assert _state_is_structurally_empty(state) is False

    def test_compose_empty_state_message_appends_system_suffix(self) -> None:
        from elspeth.web.composer.service import _compose_empty_state_message
        content = "I tried to build the pipeline but couldn't converge."
        msg = _compose_empty_state_message(content)
        # Original content preserved (model's prose is the truthful part).
        assert content in msg
        # System-attributed marker present (UI / scorers can detect).
        assert "[ELSPETH-SYSTEM]" in msg
        # Concrete next-step guidance.
        assert "refine" in msg.lower() or "retry" in msg.lower()

    def test_compose_empty_state_message_handles_empty_content(self) -> None:
        """Edge case: model produced no content at all. Suffix becomes the
        whole message (better than silence)."""
        from elspeth.web.composer.service import _compose_empty_state_message
        msg = _compose_empty_state_message("")
        assert "[ELSPETH-SYSTEM]" in msg
        assert msg.startswith("[ELSPETH-SYSTEM]") or msg.lstrip().startswith("[ELSPETH-SYSTEM]")

    # ── End-to-end through _finalize_no_tool_response ────────────────────

    @pytest.mark.asyncio
    async def test_empty_state_invalid_preflight_passes_through_model_content(self) -> None:
        """The captured rag-text-llm RED scenario: empty state, invalid
        cached preflight, model gave up with honest prose. Synthesizer
        must NOT fire."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Pipeline state is empty",
                )
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="2 validation errors for ElspethSettings — source: Field required, sinks: Field required",
                    suggestion=None,
                )
            ],
        )
        model_prose = (
            "I did discover the needed plugin requirements: web_scrape needs explicit "
            "schema, url_field, content_field, fingerprint_field, and http. The setup "
            "needs one more valid build pass to supply the right options."
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content=model_prose,
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=invalid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Model's prose preserved as the load-bearing part of the message.
        assert model_prose in result.message
        # System suffix appended.
        assert "[ELSPETH-SYSTEM]" in result.message
        # Synthesized Pydantic-noise message did NOT replace the prose.
        assert "Field required" not in result.message
        assert "I cannot mark this pipeline complete yet" not in result.message
        # Audit fields preserved (matches synthesizer-path semantics).
        assert result.raw_assistant_content == model_prose
        assert result.runtime_preflight is invalid_preflight
        # No re-run of runtime preflight (state.version unchanged).
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_state_invalid_preflight_after_state_mutation_run_preflight(self) -> None:
        """Even when the preflight is computed fresh (state.version > initial),
        the empty-state branch fires if the resulting state is empty."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        # Simulate state.version bump without populating any state fields
        # (e.g., the model called a mutation that net-emptied the state).
        bumped_state = replace(state, version=state.version + 1)
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="Pipeline empty",
                    suggestion=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight", return_value=invalid_preflight):
            result = await service._finalize_no_tool_response(
                content="Model prose explaining the gap.",
                state=bumped_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert "Model prose explaining the gap." in result.message
        assert "[ELSPETH-SYSTEM]" in result.message
        assert "Pipeline empty" not in result.message  # synthesizer skipped

    @pytest.mark.asyncio
    async def test_non_empty_state_invalid_preflight_still_synthesizes(self) -> None:
        """Regression — when the state has actually been populated AND
        runtime preflight is invalid, the synthesizer still fires
        (preserving the original "model lied about completion" semantics).
        This branch is the load-bearing safety net for the empty-state
        special case above."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="t1",
                options={"path": "/data/inputs/in.csv"},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        bumped_state = replace(state, version=state.version + 1)
        invalid_preflight = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="t1",
                    component_type="transform",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                )
            ],
        )

        with patch.object(service, "_runtime_preflight", return_value=invalid_preflight):
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=bumped_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        # Synthesizer fired — original message replaced.
        assert result.message != "The pipeline is complete and valid."
        assert "I cannot mark this pipeline complete" in result.message
        assert "Forbidden name" in result.message
        # No empty-state suffix.
        assert "[ELSPETH-SYSTEM]" not in result.message
        # raw_assistant_content preserved (matches synthesizer semantics).
        assert result.raw_assistant_content == "The pipeline is complete and valid."

    @pytest.mark.asyncio
    async def test_empty_state_valid_preflight_preserves_message_verbatim(self) -> None:
        """Sanity: empty state but valid preflight is a degenerate but
        well-defined case (an empty pipeline that "passes" preflight is
        only possible if validation is bypassed). The model's content
        must NOT be touched in this case — neither synthesizer nor
        empty-state suffix."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        valid_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight"):
            result = await service._finalize_no_tool_response(
                content="All good.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=valid_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "All good."
        assert result.raw_assistant_content is None  # no replacement happened
