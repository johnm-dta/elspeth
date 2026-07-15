"""Stateful property surface for Phase 3 compose-loop audit invariants."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import structlog
from hypothesis import example, given, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool
from tests.integration.web.conftest import _make_session
from tests.property.web.composer.strategies import (
    CANCELLATION_ARRIVAL_TIMES,
    FAILURE_INJECTION_POINTS,
    CancellationArrivalTime,
    FailureInjectionPoint,
    RedactionPolicy,
    SessionState,
    ToolCallSpec,
    st_argument_dict,
    st_cancellation_arrival_time,
    st_failure_injection_point,
    st_redaction_policy,
    st_session_state,
    st_tool_call,
)

import elspeth.web.composer.tool_batch as composer_tool_batch_module
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.protocol import ComposerConvergenceError
from elspeth.web.composer.redaction import REDACTED_UNKNOWN_RESPONSE_KEY
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import ToolResult
from elspeth.web.config import WebSettings
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value

EVAL_MODEL = "openrouter/openai/gpt-5.5"
EVAL_USER_ID = "dta_user"


@dataclass(frozen=True, slots=True)
class _FakeFunction:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass(frozen=True, slots=True)
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass(frozen=True, slots=True)
class _FakeChoice:
    message: _FakeMessage


@dataclass(frozen=True, slots=True)
class _FakeLLMResponse:
    choices: list[_FakeChoice]


class _ReplayLLM:
    def __init__(self, responses: tuple[_FakeLLMResponse, ...]) -> None:
        self._responses = list(responses)

    async def __call__(self, _messages: Any, _tools: Any) -> _FakeLLMResponse:
        if not self._responses:
            return _make_llm_response(content="Done.")
        return self._responses.pop(0)


class _BlockingLLM:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, _messages: Any, _tools: Any) -> _FakeLLMResponse:
        self.entered.set()
        await self.release.wait()
        return _make_llm_response(content="Done.")


@dataclass(frozen=True, slots=True)
class _Harness:
    service: ComposerServiceImpl
    sessions_service: SessionServiceImpl
    session_id: str


@dataclass(frozen=True, slots=True)
class _TraceOutcome:
    cancellation_arrival_time: CancellationArrivalTime
    cancelled: bool
    assistant_ids: frozenset[str]
    tool_call_ids: frozenset[str]
    rows: tuple[Any, ...]
    state_rows: tuple[Any, ...]
    tool_row_tier1_violations: int
    tool_row_integrity_violations: int
    tool_call_cap_exceeded: int


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = []
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
) -> _FakeLLMResponse:
    fake_tool_calls: list[_FakeToolCall] | None = None
    if tool_calls is not None:
        fake_tool_calls = [
            _FakeToolCall(
                id=tool_call["id"],
                function=_FakeFunction(
                    name=tool_call["name"],
                    arguments=json.dumps(tool_call["arguments"]),
                ),
            )
            for tool_call in tool_calls
        ]
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=fake_tool_calls))])


def _web_settings(data_dir: Path) -> WebSettings:
    return WebSettings(
        data_dir=data_dir,
        composer_model=EVAL_MODEL,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=180.0,
        composer_rate_limit_per_minute=60,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _make_harness(tmp_path: Path, *, session_state: SessionState = "empty") -> _Harness:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    data_dir = tmp_path / "data"
    telemetry = build_sessions_telemetry()
    sessions_service = SessionServiceImpl(
        engine,
        data_dir=data_dir,
        telemetry=telemetry,
        log=structlog.get_logger("test.property.compose-loop"),
    )
    session_id = str(uuid.uuid4())
    with engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id=EVAL_USER_ID)
        conn.execute(text("UPDATE sessions SET trust_mode = 'auto_commit' WHERE id = :session_id"), {"session_id": session_id})
    service = ComposerServiceImpl.for_trained_operator(
        catalog=_mock_catalog(),
        settings=_web_settings(data_dir),
        sessions_service=sessions_service,
    )
    service._telemetry = telemetry  # type: ignore[attr-defined]
    # The property harness injects DB failures around compose-turn audit
    # persistence. Skip the service-instance bootstrap upsert so the
    # one-time skill-history write does not consume those fault injections.
    service._skill_markdown_history_upserted = True  # type: ignore[attr-defined]
    if session_state == "has_prior_state":
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO chat_messages "
                    "(id, session_id, role, content, raw_content, tool_calls, tool_call_id, sequence_no, "
                    " writer_principal, created_at, composition_state_id, parent_assistant_id) "
                    "VALUES (:id, :session_id, 'user', 'prior', NULL, NULL, NULL, 1, "
                    "        'route_user_message', CURRENT_TIMESTAMP, NULL, NULL)"
                ),
                {"id": f"prior_{uuid.uuid4().hex}", "session_id": session_id},
            )
    return _Harness(service=service, sessions_service=sessions_service, session_id=session_id)


async def _run_one_turn(harness: _Harness, llm: Any) -> Any:
    driver = cast(Any, harness.service)
    return await driver._run_one_turn_for_test(
        llm=llm,
        session_id=harness.session_id,
        initial_state=CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        ),
    )


def _chat_rows(harness: _Harness) -> tuple[Any, ...]:
    with harness.sessions_service._engine.connect() as conn:
        return tuple(
            conn.execute(
                text(
                    "SELECT id, role, sequence_no, tool_calls, tool_call_id, parent_assistant_id, composition_state_id, content "
                    "FROM chat_messages WHERE session_id = :session_id ORDER BY sequence_no"
                ),
                {"session_id": harness.session_id},
            ).fetchall()
        )


def _state_rows(harness: _Harness) -> tuple[Any, ...]:
    with harness.sessions_service._engine.connect() as conn:
        return tuple(
            conn.execute(
                text("SELECT id, version, provenance FROM composition_states WHERE session_id = :session_id ORDER BY version"),
                {"session_id": harness.session_id},
            ).fetchall()
        )


def _assert_no_tool_row_without_parent_assistant(harness: _Harness) -> None:
    with harness.sessions_service._engine.connect() as conn:
        violations = conn.execute(
            text(
                "SELECT tool.id FROM chat_messages tool "
                "LEFT JOIN chat_messages assistant "
                "  ON assistant.id = tool.parent_assistant_id "
                " AND assistant.session_id = tool.session_id "
                " AND assistant.role = 'assistant' "
                "WHERE tool.session_id = :session_id "
                "  AND tool.role = 'tool' "
                "  AND assistant.id IS NULL"
            ),
            {"session_id": harness.session_id},
        ).fetchall()
    assert violations == []


def _assert_no_tool_state_without_tool_row(harness: _Harness) -> None:
    with harness.sessions_service._engine.connect() as conn:
        violations = conn.execute(
            text(
                "SELECT cs.id FROM composition_states cs "
                "LEFT JOIN chat_messages cm "
                "  ON cm.composition_state_id = cs.id AND cm.role = 'tool' "
                "WHERE cs.session_id = :session_id "
                "  AND cs.provenance = 'tool_call' AND cs.version > 0 "
                "  AND cm.id IS NULL"
            ),
            {"session_id": harness.session_id},
        ).fetchall()
    assert violations == []


def _assert_sequence_numbers_unique_and_monotonic(harness: _Harness) -> None:
    rows = _chat_rows(harness)
    sequence_numbers = [row.sequence_no for row in rows]
    assert sequence_numbers == sorted(sequence_numbers)
    assert len(sequence_numbers) == len(set(sequence_numbers))


def _assert_persisted_tool_payloads_match_manifest(harness: _Harness) -> None:
    for row in _chat_rows(harness):
        if row.role != "tool":
            continue
        payload = json.loads(row.content)
        assert "do-not-persist" not in row.content
        if "stray_provider_field" in payload:
            assert payload["stray_provider_field"] == REDACTED_UNKNOWN_RESPONSE_KEY


def _tool_call_payload(tool_call: ToolCallSpec, argument_dict: Mapping[str, object]) -> dict[str, Any]:
    if tool_call.name == "set_metadata":
        return {
            "id": tool_call.call_id,
            "name": "set_metadata",
            "arguments": argument_dict if "patch" in argument_dict else {"patch": {"name": "property trace"}},
        }
    return {"id": tool_call.call_id, "name": "get_pipeline_state", "arguments": {}}


async def _drive_trace_async(
    harness: _Harness,
    *,
    cancellation_arrival_time: CancellationArrivalTime,
    tool_call: ToolCallSpec,
    argument_dict: Mapping[str, object],
) -> bool:
    tool_payload = _tool_call_payload(tool_call, argument_dict)
    llm = _ReplayLLM(
        (
            _make_llm_response(tool_calls=[tool_payload]),
            _make_llm_response(content="Done."),
        )
    )

    if cancellation_arrival_time == "before_llm_call":
        task = asyncio.create_task(_run_one_turn(harness, llm))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return True

    if cancellation_arrival_time == "during_llm_call":
        blocking_llm = _BlockingLLM()
        task = asyncio.create_task(_run_one_turn(harness, blocking_llm))
        await asyncio.wait_for(blocking_llm.entered.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return True

    if cancellation_arrival_time in {"after_llm_before_tool", "during_tool_dispatch"}:

        def _cancel_at_tool_boundary(*_args: Any, **_kwargs: Any) -> ToolResult:
            raise asyncio.CancelledError()

        with patch.object(composer_tool_batch_module, "execute_tool", _cancel_at_tool_boundary), pytest.raises(asyncio.CancelledError):
            await _run_one_turn(harness, llm)
        return True

    if cancellation_arrival_time == "after_tool_before_sync_dispatch":
        real_persist_async = harness.sessions_service.persist_compose_turn_async

        async def _cancel_before_sync(**_kwargs: Any) -> Any:
            raise asyncio.CancelledError()

        harness.sessions_service.persist_compose_turn_async = _cancel_before_sync  # type: ignore[method-assign]
        try:
            with pytest.raises(asyncio.CancelledError):
                await _run_one_turn(harness, llm)
        finally:
            harness.sessions_service.persist_compose_turn_async = real_persist_async  # type: ignore[method-assign]
        return True

    if cancellation_arrival_time == "during_run_sync_between_insert_and_commit":
        original_do_commit = harness.sessions_service._engine.dialect.do_commit
        commit_started = threading.Event()
        release_commit = threading.Event()
        commit_finished = threading.Event()
        commit_errors: list[BaseException] = []

        def _gated_commit(dbapi_conn: object) -> None:
            try:
                commit_started.set()
                if not release_commit.wait(timeout=10.0):
                    pytest.fail("property worker commit gate was not released within 10s")
                original_do_commit(dbapi_conn)
            except BaseException as exc:
                commit_errors.append(exc)
                raise
            finally:
                commit_finished.set()

        harness.sessions_service._engine.dialect.do_commit = _gated_commit
        try:
            task = asyncio.create_task(_run_one_turn(harness, llm))
            for _ in range(200):
                if commit_started.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("property trace did not reach COMMIT gate within 2s")
            task.cancel()
            # Persistence is shielded from caller cancellation. Release the
            # in-flight critical section before awaiting the deferred cancel.
            release_commit.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(1000):
                if commit_finished.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("property trace did not finish shielded COMMIT within 10s")
            if commit_errors:
                raise AssertionError("property trace shielded COMMIT failed after caller cancellation") from commit_errors[0]
            await _wait_for_tool_rows(harness)
        finally:
            harness.sessions_service._engine.dialect.do_commit = original_do_commit
        return True

    if cancellation_arrival_time == "during_advisory_lock_acquisition":
        real_session_write_lock = harness.sessions_service._session_write_lock
        lock_entered = threading.Event()
        release_lock = threading.Event()

        @contextlib.contextmanager
        def _blocked_session_write_lock(*_args: Any, **_kwargs: Any) -> Any:
            lock_entered.set()
            if not release_lock.wait(timeout=2.0):
                pytest.fail("property advisory-lock gate was not released within 2s")
            raise OperationalError(
                "session write lock acquisition",
                {},
                sqlite3.OperationalError("simulated session/advisory lock unavailable"),
            )
            yield

        harness.sessions_service._session_write_lock = _blocked_session_write_lock  # type: ignore[method-assign]
        try:
            task = asyncio.create_task(_run_one_turn(harness, llm))
            for _ in range(200):
                if lock_entered.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("property trace did not reach advisory-lock gate within 2s")
            task.cancel()
            # The lock acquisition runs inside the shielded persistence task;
            # let it finish before asserting that cancellation is re-raised.
            release_lock.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
        finally:
            harness.sessions_service._session_write_lock = real_session_write_lock  # type: ignore[method-assign]
        return True

    if cancellation_arrival_time == "after_commit_before_response_yielded":
        commit_returned = asyncio.Event()
        release_response = asyncio.Event()
        real_persist_async = harness.sessions_service.persist_compose_turn_async

        async def _gated_after_commit(**kwargs: Any) -> Any:
            outcome = await real_persist_async(**kwargs)
            commit_returned.set()
            await release_response.wait()
            return outcome

        harness.sessions_service.persist_compose_turn_async = _gated_after_commit  # type: ignore[method-assign]
        try:
            task = asyncio.create_task(_run_one_turn(harness, llm))
            await asyncio.wait_for(commit_returned.wait(), timeout=2.0)
            task.cancel()
            # Model response publication after the durable write, then observe
            # the cancellation deferred across that critical section.
            release_response.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            harness.sessions_service.persist_compose_turn_async = real_persist_async  # type: ignore[method-assign]
        return True

    if cancellation_arrival_time == "after_response_yielded":
        task = asyncio.create_task(_run_one_turn(harness, llm))
        await task
        task.cancel()
        return False

    raise AssertionError(f"unhandled cancellation arrival time: {cancellation_arrival_time}")


async def _wait_for_tool_rows(harness: _Harness) -> None:
    for _ in range(1000):
        # Tail check, not exact-equality: when session_state='has_prior_state'
        # the harness pre-inserts a legitimate 'user' chat row, so a successful
        # turn persists ['user', 'assistant', 'tool']. The invariant under test
        # is that the assistant+tool pair was persisted, regardless of prior rows.
        if [row.role for row in _chat_rows(harness)][-2:] == ["assistant", "tool"]:
            return
        await asyncio.sleep(0.01)
    pytest.fail("property trace did not persist assistant/tool rows within 10s")


def _drive_single_example_trace(
    *,
    cancellation_arrival_time: CancellationArrivalTime,
    tool_call: ToolCallSpec | None = None,
    argument_dict: Mapping[str, object] | None = None,
    failure_injection_point: FailureInjectionPoint = "none",
    session_state: SessionState = "empty",
) -> _TraceOutcome:
    if failure_injection_point != "none":
        cancellation_arrival_time = "after_response_yielded"
    tmp_path = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / f"cl_pp_prop_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    harness = _make_harness(tmp_path, session_state=session_state)
    chosen_tool_call = tool_call or ToolCallSpec(name="set_metadata", call_id=f"call_{uuid.uuid4().hex[:12]}")
    chosen_arguments = argument_dict or {"patch": {"name": "property trace"}}
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}):
        if failure_injection_point == "audit_raises_OperationalError_on_commit":
            _drive_commit_failure_trace(harness, chosen_tool_call, chosen_arguments)
            cancelled = False
        elif failure_injection_point == "advisory_lock_unavailable":
            _drive_advisory_lock_failure_trace(harness, chosen_tool_call, chosen_arguments)
            cancelled = False
        elif failure_injection_point == "tool_call_cap_exceeded":
            _drive_tool_call_cap_trace(harness)
            cancelled = False
        elif failure_injection_point == "unknown_response_key":
            _drive_unknown_response_key_trace(harness, chosen_tool_call, chosen_arguments)
            cancelled = False
        else:
            cancelled = asyncio.run(
                _drive_trace_async(
                    harness,
                    cancellation_arrival_time=cancellation_arrival_time,
                    tool_call=chosen_tool_call,
                    argument_dict=chosen_arguments,
                )
            )
    outcome = _TraceOutcome(
        cancellation_arrival_time=cancellation_arrival_time,
        cancelled=cancelled,
        assistant_ids=frozenset(row.id for row in _chat_rows(harness) if row.role == "assistant"),
        tool_call_ids=frozenset(row.tool_call_id for row in _chat_rows(harness) if row.role == "tool"),
        rows=_chat_rows(harness),
        state_rows=_state_rows(harness),
        tool_row_tier1_violations=observed_value(harness.sessions_service._telemetry.tool_row_tier1_violation_total),
        tool_row_integrity_violations=observed_value(harness.sessions_service._telemetry.tool_row_integrity_violation_total),
        tool_call_cap_exceeded=observed_value(harness.service._telemetry.tool_call_cap_exceeded_total),  # type: ignore[attr-defined]
    )
    _assert_common_invariants(harness)
    return outcome


def _drive_advisory_lock_failure_trace(
    harness: _Harness,
    tool_call: ToolCallSpec,
    argument_dict: Mapping[str, object],
) -> None:
    from elspeth.contracts.errors import AuditIntegrityError

    real_session_write_lock = harness.sessions_service._session_write_lock

    @contextlib.contextmanager
    def _unavailable_session_write_lock(*_args: Any, **_kwargs: Any) -> Any:
        raise OperationalError(
            "session write lock acquisition",
            {},
            sqlite3.OperationalError("simulated session/advisory lock unavailable"),
        )
        yield

    payload = _tool_call_payload(tool_call, argument_dict)
    payload["name"] = "set_metadata"
    payload["arguments"] = {"patch": {"name": "advisory unavailable property"}}
    llm = _ReplayLLM(
        (
            _make_llm_response(tool_calls=[payload]),
            _make_llm_response(content="Done."),
        )
    )
    harness.sessions_service._session_write_lock = _unavailable_session_write_lock  # type: ignore[method-assign]
    try:
        with pytest.raises(AuditIntegrityError):
            asyncio.run(_run_one_turn(harness, llm))
    finally:
        harness.sessions_service._session_write_lock = real_session_write_lock  # type: ignore[method-assign]


def _drive_commit_failure_trace(
    harness: _Harness,
    tool_call: ToolCallSpec,
    argument_dict: Mapping[str, object],
) -> None:
    original_do_commit = harness.sessions_service._engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise sqlite3.OperationalError("simulated property COMMIT failure")
        original_do_commit(dbapi_conn)

    payload = _tool_call_payload(tool_call, argument_dict)
    payload["name"] = "set_metadata"
    payload["arguments"] = {"patch": {"name": "commit failure property"}}
    llm = _ReplayLLM(
        (
            _make_llm_response(tool_calls=[payload]),
            _make_llm_response(content="Done."),
        )
    )
    harness.sessions_service._engine.dialect.do_commit = _fail_once
    try:
        from elspeth.contracts.errors import AuditIntegrityError

        with pytest.raises(AuditIntegrityError):
            asyncio.run(_run_one_turn(harness, llm))
    finally:
        harness.sessions_service._engine.dialect.do_commit = original_do_commit


def _drive_tool_call_cap_trace(harness: _Harness) -> None:
    llm = _ReplayLLM(
        (
            _make_llm_response(
                tool_calls=[{"id": f"call_cap_{index}", "name": "get_pipeline_state", "arguments": {}} for index in range(17)]
            ),
        )
    )
    harness.service._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]
    with pytest.raises(ComposerConvergenceError):
        asyncio.run(_run_one_turn(harness, llm))


def _drive_unknown_response_key_trace(
    harness: _Harness,
    tool_call: ToolCallSpec,
    argument_dict: Mapping[str, object],
) -> None:
    class _StrayToolResult(ToolResult):
        def to_dict(self) -> dict[str, Any]:
            payload = super().to_dict()
            payload["stray_provider_field"] = "do-not-persist"
            return payload

    def _return_stray_result(
        _tool_name: str,
        _arguments: dict[str, Any],
        state: CompositionState,
        *_args: Any,
        **_kwargs: Any,
    ) -> ToolResult:
        return _StrayToolResult(
            success=True,
            updated_state=state,
            validation=state.validate(),
            affected_nodes=(),
        )

    payload = _tool_call_payload(tool_call, argument_dict)
    payload["name"] = "set_metadata"
    payload["arguments"] = {"patch": {"name": "unknown response key property"}}
    llm = _ReplayLLM(
        (
            _make_llm_response(tool_calls=[payload]),
            _make_llm_response(content="Done."),
        )
    )
    with patch.object(composer_tool_batch_module, "execute_tool", _return_stray_result):
        asyncio.run(_run_one_turn(harness, llm))


def _assert_common_invariants(harness: _Harness) -> None:
    _assert_no_tool_row_without_parent_assistant(harness)
    _assert_no_tool_state_without_tool_row(harness)
    _assert_sequence_numbers_unique_and_monotonic(harness)
    _assert_persisted_tool_payloads_match_manifest(harness)


class ComposeLoopAuditMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.outcomes: list[_TraceOutcome] = []
        self.cancellation_points_seen: set[CancellationArrivalTime] = set()

    @initialize()
    def start_session(self) -> None:
        self.outcomes = []
        self.cancellation_points_seen = set()

    @rule(
        tool_call=st_tool_call(),
        argument_dict=st_argument_dict(),
        redaction_policy=st_redaction_policy(),
        failure_injection_point=st_failure_injection_point(),
        cancellation_arrival_time=st_cancellation_arrival_time(),
        session_state=st_session_state(),
    )
    def compose_turn(
        self,
        tool_call: ToolCallSpec,
        argument_dict: Mapping[str, object],
        redaction_policy: RedactionPolicy,
        failure_injection_point: FailureInjectionPoint,
        cancellation_arrival_time: CancellationArrivalTime,
        session_state: SessionState,
    ) -> None:
        if redaction_policy == "unknown_response_key":
            failure_injection_point = "unknown_response_key"
        outcome = _drive_single_example_trace(
            cancellation_arrival_time=cancellation_arrival_time,
            tool_call=tool_call,
            argument_dict=argument_dict,
            failure_injection_point=failure_injection_point,
            session_state=session_state,
        )
        self.outcomes.append(outcome)
        self.cancellation_points_seen.add(outcome.cancellation_arrival_time)

    @invariant()
    def audit_rows_are_bidirectional(self) -> None:
        for outcome in self.outcomes:
            assert not [row for row in outcome.state_rows if row.provenance == "tool_call" and not outcome.tool_call_ids]

    @invariant()
    def cancellation_outcomes_are_classified(self) -> None:
        for outcome in self.outcomes:
            assert outcome.cancellation_arrival_time in CANCELLATION_ARRIVAL_TIMES

    @invariant()
    def counters_stay_non_negative(self) -> None:
        for outcome in self.outcomes:
            assert outcome.tool_row_tier1_violations >= 0
            assert outcome.tool_row_integrity_violations >= 0
            assert outcome.tool_call_cap_exceeded >= 0


TestComposeLoopAuditMachine = ComposeLoopAuditMachine.TestCase
TestComposeLoopAuditMachine.settings = settings(max_examples=25, stateful_step_count=2, deadline=None)


@example(cancellation_arrival_time="before_llm_call")
@example(cancellation_arrival_time="during_llm_call")
@example(cancellation_arrival_time="after_llm_before_tool")
@example(cancellation_arrival_time="during_tool_dispatch")
@example(cancellation_arrival_time="after_tool_before_sync_dispatch")
@example(cancellation_arrival_time="during_run_sync_between_insert_and_commit")
@example(cancellation_arrival_time="during_advisory_lock_acquisition")
@example(cancellation_arrival_time="after_commit_before_response_yielded")
@example(cancellation_arrival_time="after_response_yielded")
@settings(max_examples=50, deadline=None)
@given(cancellation_arrival_time=st_cancellation_arrival_time())
def test_compose_loop_audit_machine_examples(cancellation_arrival_time: CancellationArrivalTime) -> None:
    outcome = _drive_single_example_trace(cancellation_arrival_time=cancellation_arrival_time)
    assert outcome.cancellation_arrival_time == cancellation_arrival_time
    if cancellation_arrival_time in {
        "during_run_sync_between_insert_and_commit",
        "after_commit_before_response_yielded",
        "after_response_yielded",
    }:
        assert [row.role for row in outcome.rows] == ["assistant", "tool"]
    if cancellation_arrival_time in {
        "before_llm_call",
        "during_llm_call",
        "after_llm_before_tool",
        "during_tool_dispatch",
        "after_tool_before_sync_dispatch",
        "during_advisory_lock_acquisition",
    }:
        assert outcome.rows == ()


def test_failure_injection_strategy_contains_required_arms() -> None:
    assert set(FAILURE_INJECTION_POINTS) == {
        "none",
        "audit_raises_OperationalError_on_commit",
        "advisory_lock_unavailable",
        "tool_call_cap_exceeded",
        "unknown_response_key",
    }


@pytest.mark.parametrize("failure_injection_point", FAILURE_INJECTION_POINTS)
def test_failure_injection_arms_are_mechanically_drivable(failure_injection_point: FailureInjectionPoint) -> None:
    outcome = _drive_single_example_trace(
        cancellation_arrival_time="after_response_yielded",
        failure_injection_point=failure_injection_point,
    )
    if failure_injection_point in {
        "audit_raises_OperationalError_on_commit",
        "advisory_lock_unavailable",
        "tool_call_cap_exceeded",
    }:
        assert outcome.rows == ()


def test_otel_counter_postconditions() -> None:
    """§1.4 SLO counters stay dark except for explicit injected faults."""

    success = _drive_single_example_trace(cancellation_arrival_time="after_response_yielded")
    assert success.tool_row_tier1_violations == 0
    assert success.tool_row_integrity_violations == 0
    assert success.tool_call_cap_exceeded == 0

    cap = _drive_single_example_trace(
        cancellation_arrival_time="after_response_yielded",
        failure_injection_point="tool_call_cap_exceeded",
    )
    assert cap.tool_row_tier1_violations == 0
    assert cap.tool_row_integrity_violations == 0
    assert cap.tool_call_cap_exceeded == 1

    tier1 = _drive_single_example_trace(
        cancellation_arrival_time="after_response_yielded",
        failure_injection_point="audit_raises_OperationalError_on_commit",
    )
    assert tier1.tool_row_tier1_violations == 1
    assert tier1.tool_row_integrity_violations == 0
    assert tier1.rows == ()
