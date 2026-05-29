"""Replay scoreboard for the 2026-04-28 composer LLM evaluation.

These tests intentionally describe the desired post-remediation behavior for
known defects as strict xfails. Run this module with ``--runxfail`` to get the
red characterization failures that later child tickets must turn green.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_progress import ComposerProgressEvent
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.core.secrets import SecretResolutionError, resolve_secret_refs
from elspeth.core.security.secret_loader import EnvSecretLoader
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.testing import make_field, make_row
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerPluginCrashError
from elspeth.web.composer.redaction import (
    REDACTED_UNKNOWN_RESPONSE_KEY,
    redact_tool_call_arguments,
)
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult, execute_tool
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value
from tests.fixtures.factories import make_context
from tests.integration.web.conftest import _make_session

pytestmark = pytest.mark.composer_llm_eval


SOURCE_REPORT = "docs/composer/evidence/composer-llm-eval-2026-04-28.md"
EVAL_MODEL = "openrouter/openai/gpt-5.5"
EVAL_USER_ID = "dta_user"

ISSUE_CHARACTERIZATION = "elspeth-a5481032bd"
ISSUE_BLOB_PATH = "elspeth-411435710b"
ISSUE_RUNTIME_PREFLIGHT = "elspeth-34baf10c01"
ISSUE_TRIGGER_END_OF_SOURCE = "elspeth-fa94309e28"
ISSUE_BATCH_STATS_REQUIRED_FIELDS = "elspeth-178f765792"
ISSUE_BATCH_STATS_GROUP_BY = "elspeth-95904149b2"
ISSUE_SECRET_AVAILABILITY = "elspeth-cd5d811121"
ISSUE_INTROSPECTION_PATH = "elspeth-0380d5119f"
ISSUE_PROGRESS_CLASSIFICATION = "elspeth-5030f7373d"

EXPECTED_REDACTED_BLOB_SOURCE_PATH = "<redacted-blob-source-path>"

SCENARIO_1A_SESSION_ID = "c549bb63-47e9-427f-9a27-35467f877395"
SCENARIO_1B_SESSION_ID = "6472ff67-1052-406c-98c3-b3278e9ef4ea"
SCENARIO_2_SESSION_ID = "ae6816aa-1f75-4103-b176-886d14f9e104"
SCENARIO_3_SESSION_ID = "9002ed1f-3046-4c00-86be-2f1e3b3bd932"


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


class _ReplayLLM:
    """Callable fake LLM for CL-PP compose-loop characterization cases."""

    def __init__(self, responses: tuple[FakeLLMResponse, ...]) -> None:
        self._responses = list(responses)

    async def __call__(self, _messages: Any, _tools: Any) -> FakeLLMResponse:
        if not self._responses:
            return _make_llm_response(content="Done.")
        return self._responses.pop(0)


class _NoUserSecretStore:
    """User secret store stub for server-secret-only replay cases."""

    def list_secrets(self, *, user_id: str, auth_provider_type: str) -> list[SecretInventoryItem]:
        del user_id, auth_provider_type
        return []

    def has_secret_record(self, name: str, *, user_id: str, auth_provider_type: str) -> bool:
        del name, user_id, auth_provider_type
        return False

    def has_secret(self, name: str, *, user_id: str, auth_provider_type: str) -> bool:
        del name, user_id, auth_provider_type
        return False


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="batch_stats", description="Batch stats", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="json", description="JSON sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> FakeLLMResponse:
    fake_tool_calls: list[FakeToolCall] | None = None
    if tool_calls is not None:
        fake_tool_calls = [
            FakeToolCall(
                id=tool_call["id"],
                function=FakeFunction(
                    name=tool_call["name"],
                    arguments=json.dumps(tool_call["arguments"]),
                ),
            )
            for tool_call in tool_calls
        ]
    return FakeLLMResponse(choices=[FakeChoice(message=FakeMessage(content=content, tool_calls=fake_tool_calls))])


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name=f"Composer LLM eval replay ({ISSUE_CHARACTERIZATION})"),
        version=1,
    )


def _web_settings(data_dir: Path, **overrides: Any) -> WebSettings:
    defaults: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_model": EVAL_MODEL,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 180.0,
        "composer_rate_limit_per_minute": 60,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)


def _session_service_for_characterization(
    *,
    data_dir: Path,
    session_id: str,
) -> SessionServiceImpl:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        data_dir=data_dir,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.composer-llm-eval"),
    )
    with engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id=EVAL_USER_ID)
        conn.execute(text("UPDATE sessions SET trust_mode = 'auto_commit' WHERE id = :session_id"), {"session_id": session_id})
    return service


def _composer_for_characterization(
    *,
    data_dir: Path,
    session_id: str,
    **settings_overrides: Any,
) -> tuple[ComposerServiceImpl, SessionServiceImpl]:
    settings = _web_settings(data_dir, **settings_overrides)
    sessions_service = _session_service_for_characterization(
        data_dir=data_dir,
        session_id=session_id,
    )
    return (
        ComposerServiceImpl(
            catalog=_mock_catalog(),
            settings=settings,
            sessions_service=sessions_service,
        ),
        sessions_service,
    )


async def _run_one_turn_for_characterization(
    service: ComposerServiceImpl,
    *,
    llm: _ReplayLLM,
    session_id: str,
    initial_state: CompositionState | None = None,
) -> Any:
    driver = cast(Any, service)
    return await driver._run_one_turn_for_test(
        llm=llm,
        session_id=session_id,
        initial_state=initial_state,
    )


def _chat_rows(sessions_service: SessionServiceImpl, *, session_id: str) -> list[Any]:
    with sessions_service._engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT id, role, content, tool_calls, tool_call_id, parent_assistant_id, composition_state_id "
                    "FROM chat_messages WHERE session_id = :session_id ORDER BY sequence_no"
                ),
                {"session_id": session_id},
            ).fetchall()
        )


def _composition_state_rows(sessions_service: SessionServiceImpl, *, session_id: str) -> list[Any]:
    with sessions_service._engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT id, version, derived_from_state_id, provenance "
                    "FROM composition_states WHERE session_id = :session_id ORDER BY version"
                ),
                {"session_id": session_id},
            ).fetchall()
        )


@contextlib.contextmanager
def _force_commit_failure(engine: Engine) -> Iterator[None]:
    original_do_commit = engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise sqlite3.OperationalError("simulated COMMIT failure (CL-PP characterization)")
        original_do_commit(dbapi_conn)

    engine.dialect.do_commit = _fail_once
    try:
        yield
    finally:
        engine.dialect.do_commit = original_do_commit


def _write_scenario_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ticket_id,customer_tier,amount\nT-001,gold,10.0\nT-002,silver,20.0\n",
        encoding="utf-8",
    )


def _output(name: str, path: str | Path) -> OutputSpec:
    return OutputSpec(
        name=name,
        plugin="json",
        options={"path": str(path), "format": "jsonl", "schema": {"mode": "observed"}},
        on_write_failure="discard",
    )


def _source(source_path: str | Path, *, on_success: str, extra_options: dict[str, Any] | None = None) -> SourceSpec:
    options: dict[str, Any] = {
        "path": str(source_path),
        "schema": {"mode": "fixed", "fields": ["ticket_id: str", "customer_tier: str", "amount: float"]},
    }
    options.update(extra_options or {})
    # ``on_validation_failure`` accepts ``"discard"`` or a real sink name
    # (route-target validators reject anything else). The eval scenarios do
    # not exercise quarantine routing, so use ``discard`` to keep fixtures
    # internally consistent. Issue elspeth-127de6865a closed the silent-pass
    # behaviour that previously let dangling quarantine targets through.
    return SourceSpec(
        plugin="csv",
        on_success=on_success,
        options=options,
        on_validation_failure="discard",
    )


def _direct_source_state(
    source_path: str | Path,
    output_path: str | Path,
    *,
    blob_ref: str | None = None,
) -> CompositionState:
    extra_options = {"blob_ref": blob_ref} if blob_ref is not None else None
    return CompositionState(
        source=_source(source_path, on_success="summary", extra_options=extra_options),
        nodes=(),
        edges=(EdgeSpec(id="e_source_summary", from_node="source", to_node="summary", edge_type="on_success", label=None),),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 1B"),
        version=1,
    )


def _aggregation_state(
    source_path: str | Path,
    output_path: str | Path,
    *,
    trigger: dict[str, Any] | None,
    aggregation_options: dict[str, Any],
) -> CompositionState:
    return CompositionState(
        source=_source(source_path, on_success="aggregate_in"),
        nodes=(
            NodeSpec(
                id="tier_summary",
                node_type="aggregation",
                plugin="batch_stats",
                input="aggregate_in",
                on_success="summary",
                on_error="discard",
                options=aggregation_options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger=trigger,
                output_mode="transform",
                expected_output_count=1,
            ),
        ),
        edges=(
            EdgeSpec(id="e_source_agg", from_node="source", to_node="tier_summary", edge_type="on_success", label=None),
            EdgeSpec(id="e_agg_summary", from_node="tier_summary", to_node="summary", edge_type="on_success", label=None),
        ),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 2"),
        version=1,
    )


def _scenario_2_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    source_path = data_dir / "blobs" / SCENARIO_2_SESSION_ID / "tickets.csv"
    output_path = data_dir / "outputs" / "tier_summary.jsonl"
    _write_scenario_csv(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return data_dir, source_path, output_path


@pytest.mark.asyncio
async def test_cl_pp_9_mixed_redaction_policy_persists_sensitive_sentinel_and_public_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-9: mixed sensitive/non-sensitive arguments persist through the manifest walker."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-000000000009"
    service, _sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    raw_arguments = {
        "filename": "secret.txt",
        "mime_type": "text/plain",
        "content": "top-secret",
    }
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_sensitive",
                        "name": "create_blob",
                        "arguments": raw_arguments,
                    }
                ]
            ),
            _make_llm_response(content="Done."),
        )
    )

    result = await _run_one_turn_for_characterization(service, llm=llm, session_id=session_id)

    persisted_arguments = json.loads(result.persisted_assistant_tool_calls[0]["function"]["arguments"])
    expected_arguments = redact_tool_call_arguments(
        "create_blob",
        raw_arguments,
        telemetry=service._redaction_telemetry,  # type: ignore[attr-defined]
    )
    assert persisted_arguments == expected_arguments
    assert persisted_arguments["filename"] == "secret.txt"
    assert persisted_arguments["mime_type"] == "text/plain"
    assert persisted_arguments["content"] != "top-secret"
    assert persisted_arguments["content"] == "<inline-blob:10-bytes>"


@pytest.mark.asyncio
async def test_cl_pp_10a_commit_failure_without_plugin_crash_raises_audit_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-10a: COMMIT failure without plugin crash is Tier-1 audit failure."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-00000000010a"
    service, sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    telemetry = build_sessions_telemetry()
    service._telemetry = telemetry  # type: ignore[attr-defined]
    sessions_service._telemetry = telemetry
    # Phase 5b Task 5 follow-on: prime the F-5c per-instance gate so the
    # first compose-loop entry doesn't issue the skill_markdown_history
    # upsert (its commit would be caught by ``_force_commit_failure``
    # below, displacing the persist_compose_turn commit this test
    # actually targets).
    service._skill_markdown_history_upserted = True  # type: ignore[attr-defined]
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_ok",
                        "name": "set_metadata",
                        "arguments": {"patch": {"name": "commit failure replay"}},
                    }
                ]
            ),
            _make_llm_response(content="Done."),
        )
    )

    with (
        _force_commit_failure(sessions_service._engine),
        pytest.raises(AuditIntegrityError) as exc_info,
    ):
        await _run_one_turn_for_characterization(service, llm=llm, session_id=session_id)

    assert isinstance(exc_info.value.__cause__, OperationalError)
    assert observed_value(telemetry.tool_row_tier1_violation_total) == 1
    assert _chat_rows(sessions_service, session_id=session_id) == []


@pytest.mark.asyncio
async def test_cl_pp_10b_commit_failure_during_plugin_crash_preserves_plugin_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-10b: COMMIT failure on unwind increments counter, then plugin crash propagates."""

    from structlog.testing import capture_logs

    import elspeth.web.composer.service as composer_service_module

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-00000000010b"
    service, sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    telemetry = build_sessions_telemetry()
    sessions_service._telemetry = telemetry
    # See CL-PP-10a above for the F-5c gate-priming rationale.
    service._skill_markdown_history_upserted = True  # type: ignore[attr-defined]
    original_execute_tool = composer_service_module.execute_tool
    calls = 0

    def _crash_on_second(tool_name: str, *args: Any, **kwargs: Any) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("CL-PP-10b synthetic plugin crash")
        return original_execute_tool(tool_name, *args, **kwargs)

    # The compose-loop tool dispatch was extracted into tool_batch.run_tool_batch,
    # which binds execute_tool in its own module namespace. Patch both the service
    # (inline-blob recipe path) and tool_batch (dispatch path) seams so the intercept
    # fires regardless of which path executes the tool.
    import elspeth.web.composer.tool_batch as _composer_tool_batch_module

    monkeypatch.setattr(composer_service_module, "execute_tool", _crash_on_second)
    monkeypatch.setattr(_composer_tool_batch_module, "execute_tool", _crash_on_second)
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_ok",
                        "name": "set_metadata",
                        "arguments": {"patch": {"name": "before crash"}},
                    },
                    {
                        "id": "call_crash",
                        "name": "set_metadata",
                        "arguments": {"patch": {"description": "crash"}},
                    },
                ]
            ),
        )
    )

    with (
        _force_commit_failure(sessions_service._engine),
        capture_logs() as cap_logs,
        pytest.raises(ComposerPluginCrashError) as exc_info,
    ):
        await _run_one_turn_for_characterization(service, llm=llm, session_id=session_id)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert exc_info.value.failed_turn is not None
    assert exc_info.value.failed_turn.assistant_message_id is None
    assert observed_value(telemetry.tool_row_persist_failed_during_unwind_total) == 1
    assert any(log["event"] == "audit_insert_failed_during_tool_failure_unwind" for log in cap_logs)
    assert _chat_rows(sessions_service, session_id=session_id) == []


@pytest.mark.asyncio
async def test_cl_pp_10c_cancellation_during_shielded_sync_dispatch_commits_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-10c: caller cancellation during shielded sync dispatch does not interrupt COMMIT."""

    import threading

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-00000000010c"
    service, sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    telemetry = build_sessions_telemetry()
    sessions_service._telemetry = telemetry
    starting_integrity_violations = observed_value(telemetry.tool_row_integrity_violation_total)

    real_persist = sessions_service.persist_compose_turn
    release_worker = threading.Event()
    worker_started = threading.Event()
    worker_finished = threading.Event()
    worker_errors: list[BaseException] = []

    def _gated_persist(*args: Any, **kwargs: Any) -> Any:
        try:
            worker_started.set()
            if not release_worker.wait(timeout=10.0):
                pytest.fail("CL-PP-10c worker gate was not released within 10s")
            return real_persist(*args, **kwargs)
        except BaseException as exc:
            worker_errors.append(exc)
            raise
        finally:
            worker_finished.set()

    sessions_service.persist_compose_turn = _gated_persist  # type: ignore[method-assign]
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_cancel_during_dispatch",
                        "name": "set_metadata",
                        "arguments": {"patch": {"name": "cancel during dispatch"}},
                    }
                ]
            ),
            _make_llm_response(content="Done."),
        )
    )

    compose_task = asyncio.create_task(_run_one_turn_for_characterization(service, llm=llm, session_id=session_id))

    for _ in range(200):
        if worker_started.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("CL-PP-10c worker did not enter shielded dispatch within 2s")

    compose_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await compose_task

    release_worker.set()
    for _ in range(1000):
        if worker_finished.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("CL-PP-10c shielded worker did not finish within 10s")

    if worker_errors:
        raise AssertionError("CL-PP-10c shielded worker failed after caller cancellation") from worker_errors[0]

    rows = _chat_rows(sessions_service, session_id=session_id)
    assert len(rows) == 2
    assert [row.role for row in rows] == ["assistant", "tool"]
    assert rows[0].tool_calls is not None
    assert rows[1].tool_call_id == "call_cancel_during_dispatch"
    assert rows[1].parent_assistant_id == rows[0].id
    assert observed_value(telemetry.tool_row_integrity_violation_total) == starting_integrity_violations


@pytest.mark.asyncio
async def test_cl_pp_10d_cancellation_after_commit_before_response_yield_keeps_single_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-10d: cancellation after COMMIT preserves rows and does not duplicate tool drains."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-00000000010d"
    service, sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    commit_returned = asyncio.Event()
    release_response = asyncio.Event()
    real_persist_async = sessions_service.persist_compose_turn_async

    async def _gated_after_commit(*args: Any, **kwargs: Any) -> Any:
        outcome = await real_persist_async(*args, **kwargs)
        commit_returned.set()
        await release_response.wait()
        return outcome

    sessions_service.persist_compose_turn_async = _gated_after_commit  # type: ignore[method-assign]
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_cancel_after_commit",
                        "name": "set_metadata",
                        "arguments": {"patch": {"name": "cancel after commit"}},
                    }
                ]
            ),
            _make_llm_response(content="Done."),
        )
    )

    compose_task = asyncio.create_task(_run_one_turn_for_characterization(service, llm=llm, session_id=session_id))

    await asyncio.wait_for(commit_returned.wait(), timeout=2.0)
    rows_after_commit = _chat_rows(sessions_service, session_id=session_id)
    assert [row.role for row in rows_after_commit] == ["assistant", "tool"]

    compose_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await compose_task
    release_response.set()

    rows = _chat_rows(sessions_service, session_id=session_id)
    assert [row.role for row in rows] == ["assistant", "tool"]
    assert rows[0].tool_calls is not None
    assert rows[1].tool_call_id == "call_cancel_after_commit"
    assert rows[1].parent_assistant_id == rows[0].id
    assert rows[1].composition_state_id is not None
    state_rows = _composition_state_rows(sessions_service, session_id=session_id)
    assert [row.provenance for row in state_rows] == ["tool_call"]
    assert rows[1].composition_state_id == state_rows[0].id
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_cl_pp_12_tool_call_cap_exceeded_writes_no_rows_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-12: over-cap assistant turns fail before any tool executes or persists."""

    import elspeth.web.composer.service as composer_service_module

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-000000000012"
    service, sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    telemetry = build_sessions_telemetry()
    service._telemetry = telemetry  # type: ignore[attr-defined]
    service._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]
    execute_calls = 0

    def _counting_execute_tool(*_args: Any, **_kwargs: Any) -> ToolResult:
        nonlocal execute_calls
        execute_calls += 1
        raise AssertionError("execute_tool must not run when CL-PP-12 cap fires")

    # Patch both seams (see CL-PP-10b): dispatch resolves execute_tool via
    # tool_batch.run_tool_batch's own module binding, so a service-only patch
    # would not catch a tool that slipped past the cap.
    import elspeth.web.composer.tool_batch as _composer_tool_batch_module

    monkeypatch.setattr(composer_service_module, "execute_tool", _counting_execute_tool)
    monkeypatch.setattr(_composer_tool_batch_module, "execute_tool", _counting_execute_tool)
    llm = _ReplayLLM(
        (_make_llm_response(tool_calls=[{"id": f"call_{idx}", "name": "get_pipeline_state", "arguments": {}} for idx in range(17)]),)
    )

    with pytest.raises(ComposerConvergenceError) as exc_info:
        await _run_one_turn_for_characterization(service, llm=llm, session_id=session_id)

    assert exc_info.value.reason == "tool_call_cap_exceeded"
    assert exc_info.value.evidence == {"observed": 17, "cap": 16}
    assert execute_calls == 0
    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 1
    assert _chat_rows(sessions_service, session_id=session_id) == []


@pytest.mark.asyncio
async def test_cl_pp_13_unknown_response_key_redacted_in_persisted_tool_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CL-PP-13: declarative response-shape drift is sentinelized before persistence."""

    import elspeth.web.composer.service as composer_service_module
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

    class _StrayToolResult(ToolResult):
        def to_dict(self) -> dict[str, Any]:
            payload = super().to_dict()
            payload["stray_provider_field"] = "do-not-persist"
            return payload

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    session_id = "00000000-0000-4000-8000-000000000013"
    service, _sessions_service = _composer_for_characterization(data_dir=tmp_path / "data", session_id=session_id)
    redaction_telemetry = NoopRedactionTelemetry()
    service._redaction_telemetry = redaction_telemetry  # type: ignore[attr-defined]

    def _return_stray_result(
        tool_name: str, _arguments: dict[str, Any], state: CompositionState, *_args: Any, **_kwargs: Any
    ) -> ToolResult:
        if tool_name != "set_metadata":
            raise AssertionError(f"unexpected tool {tool_name!r}")
        return _StrayToolResult(
            success=True,
            updated_state=state,
            validation=state.validate(),
            affected_nodes=(),
        )

    # Patch both seams (see CL-PP-10b): the dispatch path that persists the tool
    # row resolves execute_tool via tool_batch.run_tool_batch's own module binding.
    import elspeth.web.composer.tool_batch as _composer_tool_batch_module

    monkeypatch.setattr(composer_service_module, "execute_tool", _return_stray_result)
    monkeypatch.setattr(_composer_tool_batch_module, "execute_tool", _return_stray_result)
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[
                    {
                        "id": "call_stray",
                        "name": "set_metadata",
                        "arguments": {"patch": {"name": "unknown response key replay"}},
                    }
                ]
            ),
            _make_llm_response(content="Done."),
        )
    )

    result = await _run_one_turn_for_characterization(service, llm=llm, session_id=session_id)

    persisted_tool_row = json.loads(result.persisted_tool_row_content[0])
    assert persisted_tool_row["stray_provider_field"] == REDACTED_UNKNOWN_RESPONSE_KEY
    assert persisted_tool_row["stray_provider_field"] == "<redacted-unknown-response-key>"
    assert "do-not-persist" not in result.persisted_tool_row_content[0]
    assert redaction_telemetry.unknown_response_key_calls == [{"tool_name": "set_metadata"}]


def _format_validation_errors(result: ValidationResult) -> str:
    return "\n".join(error.message for error in result.errors)


def _format_composer_errors(summary: ValidationSummary) -> str:
    return "\n".join(entry.message for entry in summary.errors)


def _make_pipeline_row(data: dict[str, Any]):
    fields = tuple(
        make_field(key, type(value) if value is not None else object, original_name=key, required=False, source="inferred")
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


def test_scenario_1b_blob_service_storage_path_validates_through_runtime_path_allowlist(
    tmp_path: Path,
) -> None:
    """Protects the Scenario 1B/3 blob-path failure captured in the report.

    Originally characterized that a relative ``data/blobs/<sid>/<bid>_<filename>``
    path would resolve via the legacy CWD branch in ``web/paths.py`` and pass
    the runtime path allowlist.  After elspeth-07089fbaa3 closed the
    composer-stored blob source path defect, the canonical contract is that
    blob-backed source paths are absolute (``BlobRecord.storage_path``); the
    legacy relative form is no longer accepted.  This test now pins the
    *post-fix* contract: an absolute canonical path under the configured
    ``data_dir`` validates cleanly through the runtime allowlist.
    """
    data_dir = tmp_path / "data"
    blob_id = "11111111-1111-4111-8111-111111111111"
    canonical_storage_path = data_dir / "blobs" / SCENARIO_1B_SESSION_ID / f"{blob_id}_tickets.csv"
    _write_scenario_csv(canonical_storage_path)

    state = _direct_source_state(
        str(canonical_storage_path),
        str(data_dir / "outputs" / "scenario_1b_summary.jsonl"),
        blob_ref=blob_id,
    )

    composer_summary = state.validate()
    assert composer_summary.is_valid, _format_composer_errors(composer_summary)

    runtime_result = validate_pipeline(
        state,
        _web_settings(data_dir),
        composer_yaml_generator,
    )

    assert runtime_result.is_valid, _format_validation_errors(runtime_result)


def test_scenario_2_end_of_source_condition_rejected_before_runtime_settings_load(tmp_path: Path) -> None:
    """Protects Scenario 2 aggregation trigger shape drift."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert not runtime_result.is_valid
    assert "end_of_source" in _format_validation_errors(runtime_result)

    composer_summary = state.validate()
    assert not composer_summary.is_valid, "composer accepted an end_of_source token in the boolean condition slot"
    assert "end_of_source" in _format_composer_errors(composer_summary)


def test_scenario_2_omitted_trigger_is_end_of_source_only_contract(tmp_path: Path) -> None:
    """Composer and runtime agree that omitted trigger means end-of-source-only aggregation."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger=None,
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )

    composer_summary = state.validate()
    assert composer_summary.is_valid, _format_composer_errors(composer_summary)

    yaml_doc = yaml.safe_load(composer_yaml_generator.generate_yaml(state))
    assert "trigger" not in yaml_doc["aggregations"][0]

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert runtime_result.is_valid, _format_validation_errors(runtime_result)


def test_scenario_2_batch_stats_required_input_fields_returns_pre_execution_validation_error(tmp_path: Path) -> None:
    """Protects the ADR-013 batch-aware dispatch gap from Scenario 2."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"count": 100},
        aggregation_options={
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "required_input_fields": ["amount"],
        },
    )

    runtime_result = validate_pipeline(state, _web_settings(data_dir), composer_yaml_generator)
    assert not runtime_result.is_valid
    assert "batch-aware" in _format_validation_errors(runtime_result)

    composer_summary = state.validate()
    assert not composer_summary.is_valid
    assert "required_input_fields" in _format_composer_errors(composer_summary)


def test_scenario_2_batch_stats_group_by_emits_per_tier_rollups() -> None:
    """Protects elspeth-95904149b2 with the batch_stats per-group rollup contract."""
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "group_by": "customer_tier",
        }
    )
    rows = [
        _make_pipeline_row({"ticket_id": "T-001", "customer_tier": "gold", "amount": 10.0}),
        _make_pipeline_row({"ticket_id": "T-002", "customer_tier": "silver", "amount": 20.0}),
        _make_pipeline_row({"ticket_id": "T-003", "customer_tier": "gold", "amount": 30.0}),
    ]

    result = transform.process(rows, make_context())

    assert result.status == "success"
    assert result.is_multi_row
    assert result.rows is not None
    rollups = {row["customer_tier"]: row for row in result.rows}
    assert set(rollups) == {"gold", "silver"}
    assert rollups["gold"]["count"] == 2
    assert rollups["gold"]["sum"] == 40.0
    assert rollups["silver"]["count"] == 1
    assert rollups["silver"]["sum"] == 20.0


def test_known_secret_env_marker_cannot_bypass_unavailable_web_secret_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects elspeth-cd5d811121 without any live provider call."""
    secret_name = "OPENROUTER_API_KEY"
    secret_value = "test-openrouter-key"
    monkeypatch.setenv(secret_name, secret_value)
    monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

    loaded_value, secret_ref = EnvSecretLoader().get_secret(secret_name)
    assert loaded_value == secret_value
    assert secret_ref.source == "env"

    settings = _web_settings(
        tmp_path / "data",
        composer_model=EVAL_MODEL,
        server_secret_allowlist=(secret_name,),
    )
    composer = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    assert composer._availability.available is True
    assert composer._availability.provider == "openrouter"

    web_secret_service = WebSecretService(
        user_store=_NoUserSecretStore(),  # type: ignore[arg-type]
        server_store=ServerSecretStore(allowlist=(secret_name,)),
    )
    resolver = ScopedSecretResolver(web_secret_service, auth_provider_type=settings.auth_provider)

    result = execute_tool(
        "validate_secret_ref",
        {"name": secret_name},
        _empty_state(),
        _mock_catalog(),
        secret_service=resolver,
        user_id=EVAL_USER_ID,
    )

    assert result.success is True
    assert result.to_dict()["data"] == {"name": secret_name, "available": False}

    with pytest.raises(SecretResolutionError) as exc_info:
        resolve_secret_refs(
            {"api_key": f"${{{secret_name}}}"},
            resolver,
            EVAL_USER_ID,
            env_ref_names=frozenset({secret_name}),
        )

    assert exc_info.value.missing == [secret_name]


def test_scenario_3_get_pipeline_state_preserves_redacted_patched_blob_path_that_yaml_preserves(tmp_path: Path) -> None:
    """Characterizes elspeth-0380d5119f redaction across get_pipeline_state and YAML.

    Originally exercised ``patch_source_options`` to set the blob source
    path; after elspeth-07089fbaa3 closed the composer-stored blob source
    path defect, that flow is forbidden — patches against a blob-backed
    source may not touch ``path`` or ``blob_ref`` because the binding is
    immutable.  The redaction contract (LLM/HTTP surfaces see the sentinel;
    YAML emission preserves the real path for the runtime) is now pinned
    against an initial state that already carries the canonical path
    (the post-fix shape produced by ``set_source_from_blob``).
    """
    data_dir = tmp_path / "data"
    blob_id = "33333333-3333-4333-8333-333333333333"
    source_path = data_dir / "blobs" / SCENARIO_3_SESSION_ID / f"{blob_id}_tickets.csv"
    output_path = data_dir / "outputs" / "scenario_3_summary.jsonl"
    _write_scenario_csv(source_path)

    initial_state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="summary",
            options={
                "blob_ref": blob_id,
                "path": str(source_path),
                "schema": {"mode": "observed"},
            },
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(EdgeSpec(id="e_source_summary", from_node="source", to_node="summary", edge_type="on_success", label=None),),
        outputs=(_output("summary", output_path),),
        metadata=PipelineMetadata(name=f"{SOURCE_REPORT} scenario 3 patched path"),
        version=1,
    )

    # patch_source_options against a blob-backed source must reject any
    # patch that touches the immutable (path, blob_ref) binding — see
    # elspeth-07089fbaa3.  Re-binding requires set_source_from_blob.
    rejected = execute_tool(
        "patch_source_options",
        {"patch": {"path": str(source_path)}},
        initial_state,
        _mock_catalog(),
        data_dir=str(data_dir),
    )
    assert rejected.success is False
    assert "blob-backed source" in rejected.data["error"]

    # The redaction contract still holds for canonical-path blob sources
    # (the shape set_source_from_blob produces).
    introspection = execute_tool(
        "get_pipeline_state",
        {"component": "source"},
        initial_state,
        _mock_catalog(),
    )
    assert introspection.success is True
    introspected_source = introspection.to_dict()["data"]["source"]
    assert introspected_source["options"]["path"] == EXPECTED_REDACTED_BLOB_SOURCE_PATH
    assert introspected_source["options"]["blob_ref"] == blob_id
    assert str(source_path) not in json.dumps(introspection.to_dict()["data"])

    yaml_doc = yaml.safe_load(composer_yaml_generator.generate_yaml(initial_state))
    assert yaml_doc["source"]["options"]["path"] == str(source_path)


async def _failed_progress_for_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ComposerProgressEvent:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    settings = _web_settings(tmp_path / "data", composer_timeout_seconds=0.05)
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    events: list[ComposerProgressEvent] = []

    async def record_progress(event: ComposerProgressEvent) -> None:
        events.append(event)

    async def slow_llm(*args: Any, **kwargs: Any) -> FakeLLMResponse:
        del args, kwargs
        await asyncio.sleep(1.0)
        return _make_llm_response(content="too late")

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = slow_llm
        with pytest.raises(ComposerConvergenceError) as exc_info:
            await service.compose(
                "Scenario 1A monolithic request",
                [],
                _empty_state(),
                session_id=SCENARIO_1A_SESSION_ID,
                user_id=EVAL_USER_ID,
                progress=record_progress,
            )

    assert exc_info.value.budget_exhausted == "timeout"
    return next(event for event in reversed(events) if event.phase == "failed")


async def _failed_progress_for_composition_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ComposerProgressEvent:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    data_dir = tmp_path / "data"
    settings = _web_settings(data_dir, composer_max_composition_turns=1)
    sessions_service = _session_service_for_characterization(
        data_dir=data_dir,
        session_id=SCENARIO_1A_SESSION_ID,
    )
    service = ComposerServiceImpl(
        catalog=_mock_catalog(),
        settings=settings,
        sessions_service=sessions_service,
    )
    events: list[ComposerProgressEvent] = []

    async def record_progress(event: ComposerProgressEvent) -> None:
        events.append(event)

    mutation = _make_llm_response(
        tool_calls=[
            {
                "id": "call_1",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "budget replay"}},
            }
        ]
    )
    bonus_mutation = _make_llm_response(
        tool_calls=[
            {
                "id": "call_2",
                "name": "set_metadata",
                "arguments": {"patch": {"description": "still mutating"}},
            }
        ]
    )

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [mutation, bonus_mutation]
        with pytest.raises(ComposerConvergenceError) as exc_info:
            await service.compose(
                "Keep editing forever",
                [],
                _empty_state(),
                session_id=SCENARIO_1A_SESSION_ID,
                user_id=EVAL_USER_ID,
                progress=record_progress,
            )

    assert exc_info.value.budget_exhausted == "composition"
    return next(event for event in reversed(events) if event.phase == "failed")


def _progress_copy(event: ComposerProgressEvent) -> tuple[str, tuple[str, ...], str | None]:
    return (event.headline, event.evidence, event.likely_next)


@pytest.mark.asyncio
async def test_long_running_compose_failures_expose_distinct_progress_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects the report's long-running failure-classification finding.

    Originally an xfail-strict guard for ``{ISSUE_PROGRESS_CLASSIFICATION}``;
    flipped to a passing characterization test once the discriminator
    landed. Now serves as the regression guard against re-collapsing the
    three convergence sub-causes into a single generic event.
    """
    timeout_event = await _failed_progress_for_timeout(tmp_path, monkeypatch)
    composition_event = await _failed_progress_for_composition_budget(tmp_path, monkeypatch)

    assert _progress_copy(timeout_event) != _progress_copy(composition_event)
    # Tighter assertion than text inequality: the discriminator field must
    # carry the per-sub-cause Literal so frontend / response body / LLM
    # recovery can branch on a stable taxonomy.
    assert timeout_event.reason == "convergence_wall_clock_timeout"
    assert composition_event.reason == "convergence_composition_budget"


def test_runtime_preflight_preview_blocks_scenario_2_invalid_trigger(tmp_path: Path) -> None:
    """Scenario 2: preview must show runtime failure, not authoring-only validity."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    settings = _web_settings(data_dir)

    def runtime_preflight(candidate: CompositionState) -> ValidationResult:
        return validate_pipeline(candidate, settings, composer_yaml_generator)

    preview = execute_tool(
        "preview_pipeline",
        {},
        state,
        _mock_catalog(),
        data_dir=str(data_dir),
        runtime_preflight=runtime_preflight,
    )

    preview_data = preview.to_dict()["data"]
    assert preview.success is True
    assert preview_data["is_valid"] is False
    assert preview_data["runtime_preflight"]["is_valid"] is False
    assert "end_of_source" in json.dumps(preview_data["runtime_preflight"])


@pytest.mark.asyncio
async def test_final_completion_claim_is_augmented_with_runtime_preflight_failure(tmp_path: Path) -> None:
    """The composer must augment (not discard) the LLM's prose with the validator's
    objection after a dry-run failure. Issue elspeth-9cfbad6901 unified the
    preflight-fail policy on augmentation: the model's prose is preserved
    verbatim and the technical preflight error is appended as a system-
    attributed suffix.
    """
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    settings = _web_settings(data_dir)
    composer = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    changed_state = replace(state, version=state.version + 1)
    model_prose = "The pipeline is complete and valid."

    result = await composer._finalize_no_tool_response(
        content=model_prose,
        state=changed_state,
        initial_version=state.version,
        user_id=EVAL_USER_ID,
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope="session:eval",
    )

    # Augmentation prefix invariant: model prose preserved verbatim at start.
    assert result.message.startswith(model_prose)
    # System-attributed suffix carries the validator's specific objection.
    # The first ValidationError.message for the end_of_source trigger case
    # contains "end_of_source", so a regression that omits the validator
    # detail from the suffix would fail this check.
    assert "[ELSPETH-SYSTEM]" in result.message
    assert "end_of_source" in result.message
    # raw_assistant_content carries the unaugmented prose for the audit
    # trail and LLM history replay.
    assert result.raw_assistant_content == model_prose
    assert result.runtime_preflight is not None
    assert result.runtime_preflight.is_valid is False
