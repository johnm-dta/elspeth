"""Compose-loop Step 1/2/3 unit tests (spec §5.2.1)."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import select, text

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.redaction import redact_tool_call_arguments, redact_tool_call_response
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import ValidationSummary
from elspeth.web.composer.tools._common import ToolResult
from elspeth.web.sessions.models import chat_messages_table
from elspeth.web.sessions.protocol import ComposerSessionPreferencesRecord, CompositionStateData
from tests.unit.web.composer._helpers import _stub_advisor_end_gate_clean  # noqa: F401  (autouse end-gate CLEAN stub)


async def _run_one_turn(
    service: ComposerServiceImpl,
    *,
    llm: Any,
    session_id: str,
    current_state_id: str | None = None,
) -> Any:
    driver = cast(Any, service)
    return await driver._run_one_turn_for_test(llm=llm, session_id=session_id, current_state_id=current_state_id)


def _patch_auto_commit_preferences(monkeypatch: pytest.MonkeyPatch, sessions_service: Any) -> None:
    async def _get_composer_preferences(session_id: UUID) -> ComposerSessionPreferencesRecord:
        return ComposerSessionPreferencesRecord(
            session_id=session_id,
            trust_mode="auto_commit",
            density_default="high",
            interpretation_review_disabled=False,
            updated_at=datetime.now(UTC),
        )

    monkeypatch.setattr(sessions_service, "get_composer_preferences", _get_composer_preferences)


def _advisor_tool_call_response(call_id: str, *, extra_args: dict[str, Any] | None = None) -> Any:
    arguments = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck on llm config with private schema",
        "recent_errors": [
            "validator rejected the private column",
            "validator rejected the private column",
        ],
        "attempted_actions": [
            "set_pipeline with sensitive options",
            "checked the relevant schema",
        ],
    }
    if extra_args is not None:
        arguments.update(extra_args)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            function=SimpleNamespace(
                                name="request_advisor_hint",
                                arguments=json.dumps(arguments),
                            ),
                        )
                    ],
                )
            )
        ],
    )


def _text_response(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


def _metadata_tool_response(call_id: str, name: str) -> Any:
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="set_metadata",
            arguments=json.dumps({"patch": {"name": name}}),
        ),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))])


def _advisor_model_response(content: str = "Try setting `provider: azure` with the deployment name.") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        model="anthropic/claude-sonnet-4-6",
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165),
    )


@pytest.mark.asyncio
async def test_step1_three_tools_all_succeed_accumulates_three_outcomes(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_three_tool_calls: Any,
    result_session_id: str,
) -> None:
    """Three successful tools produce three outcomes with response set."""

    result = await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_three_tool_calls,
        session_id=result_session_id,
    )

    outcomes = result.tool_outcomes_for_assertion
    assert len(outcomes) == 3
    assert all(outcome.error_class is None for outcome in outcomes)
    assert all(outcome.response is not None for outcome in outcomes)
    assert outcomes[0].post_version <= outcomes[1].post_version <= outcomes[2].post_version


@pytest.mark.asyncio
async def test_step1_tool_argument_error_continues_loop(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_tool_argument_error_on_second: Any,
    result_session_id: str,
) -> None:
    """ToolArgumentError on call 2 of 3 records an error and continues."""

    result = await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_tool_argument_error_on_second,
        session_id=result_session_id,
    )

    outcomes = result.tool_outcomes_for_assertion
    assert len(outcomes) == 3
    assert outcomes[0].error_class is None
    assert outcomes[1].error_class == "ToolArgumentError"
    assert outcomes[2].error_class is None


@pytest.mark.asyncio
async def test_step1_assertion_error_reraises_before_persist(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_assertion_error_on_second: Any,
    result_session_id: str,
) -> None:
    """AssertionError is re-raised before any compose-turn DB write runs."""

    with pytest.raises(AssertionError):
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=fake_llm_assertion_error_on_second,
            session_id=result_session_id,
        )

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text("SELECT id FROM chat_messages WHERE session_id = :session_id AND role IN ('assistant', 'tool')"),
            {"session_id": result_session_id},
        ).fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_step1_plugin_bug_captures_crash_breaks_loop(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_runtime_error_on_second: Any,
    result_session_id: str,
) -> None:
    """RuntimeError on call 2 of 3 records the crash and skips call 3."""

    with pytest.raises(ComposerPluginCrashError) as excinfo:
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=fake_llm_runtime_error_on_second,
            session_id=result_session_id,
        )

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    outcomes = composer_service_with_real_sessions._phase3_last_tool_outcomes  # type: ignore[attr-defined]
    assert len(outcomes) == 2
    assert outcomes[0].error_class is None
    assert outcomes[1].error_class == "RuntimeError"
    assert outcomes[1].error_message == "RuntimeError"
    assert "phase3 synthetic runtime error" not in (outcomes[1].error_message or "")


@pytest.mark.asyncio
async def test_step2_redacts_via_manifest_walker(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_with_sensitive_tool_call: Any,
    result_session_id: str,
) -> None:
    """Assistant tool_calls are redacted with the Phase 2 manifest walker."""

    result = await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_with_sensitive_tool_call,
        session_id=result_session_id,
    )

    expected = tuple(
        redact_tool_call_arguments(
            outcome.call.function.name,
            json.loads(outcome.call.function.arguments),
            telemetry=composer_service_with_real_sessions._redaction_telemetry,  # type: ignore[attr-defined]
        )
        for outcome in result.tool_outcomes
    )
    persisted = tuple(json.loads(call["function"]["arguments"]) for call in result.persisted_assistant_tool_calls)
    assert persisted == expected


@pytest.mark.asyncio
async def test_step2_redacts_response_with_summarizer(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_summarizer_active: Any,
    result_session_id: str,
) -> None:
    """Tool-row content is serialized from redact_tool_call_response output."""

    result = await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_summarizer_active,
        session_id=result_session_id,
    )

    expected_content = redact_tool_call_response(
        tool_name=result.tool_outcomes[0].call.function.name,
        response=result.tool_outcomes[0].response.to_dict(),
        telemetry=composer_service_with_real_sessions._redaction_telemetry,  # type: ignore[attr-defined]
    )
    assert json.loads(result.persisted_tool_row_content[0]) == expected_content


@pytest.mark.asyncio
async def test_step2_persists_intercepted_advisor_tool_call_rows(
    composer_service_with_real_sessions: ComposerServiceImpl,
    result_session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intercepted advisor calls still persist assistant tool_calls and tool rows."""

    service = composer_service_with_real_sessions
    service._settings = service._settings.model_copy(  # type: ignore[attr-defined]
        update={
            "composer_advisor_max_calls_per_compose": 3,
        }
    )
    responses = [
        _advisor_tool_call_response("call_advisor_phase3"),
        _text_response("Done."),
    ]

    async def _fake_llm(_messages: Any, _tools: Any) -> Any:
        return responses.pop(0)

    async def _fake_advisor(**_kwargs: Any) -> Any:
        return _advisor_model_response()

    monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", _fake_advisor)

    result = await _run_one_turn(
        service,
        llm=_fake_llm,
        session_id=result_session_id,
    )

    advisor_invocations = [inv for inv in result.tool_invocations if inv.tool_name == "request_advisor_hint"]
    assert len(advisor_invocations) == 1
    assert len(result.persisted_assistant_tool_calls) == 1
    assert len(result.persisted_tool_row_content) == 1
    persisted_call = result.persisted_assistant_tool_calls[0]
    assert persisted_call["id"] == "call_advisor_phase3"
    assert persisted_call["function"]["name"] == "request_advisor_hint"
    persisted_args = json.loads(persisted_call["function"]["arguments"])
    assert persisted_args["problem_summary"].startswith("<advisor-problem-summary:")
    persisted_content = json.loads(result.persisted_tool_row_content[0])
    assert persisted_content["status"] == "SUCCESS"
    assert persisted_content["guidance"] == "<redacted>"
    assert persisted_content["model"] == "anthropic/claude-sonnet-4-6"

    sessions_service = service._sessions_service  # type: ignore[attr-defined]
    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            select(
                chat_messages_table.c.role,
                chat_messages_table.c.tool_calls,
                chat_messages_table.c.tool_call_id,
                chat_messages_table.c.content,
            )
            .where(chat_messages_table.c.session_id == result_session_id)
            .where(chat_messages_table.c.role.in_(("assistant", "tool")))
            .order_by(chat_messages_table.c.sequence_no)
        ).mappings()
        persisted_rows = list(rows)

    assert [row["role"] for row in persisted_rows] == ["assistant", "tool"]
    assert persisted_rows[0]["tool_calls"][0]["id"] == "call_advisor_phase3"
    assert persisted_rows[0]["tool_calls"][0]["function"]["name"] == "request_advisor_hint"
    assert persisted_rows[1]["tool_call_id"] == "call_advisor_phase3"
    assert json.loads(persisted_rows[1]["content"])["guidance"] == "<redacted>"


@pytest.mark.asyncio
async def test_step2_redacts_intercepted_advisor_unknown_arguments_before_persist(
    composer_service_with_real_sessions: ComposerServiceImpl,
    result_session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advisor ARG_ERROR rows must not mirror unknown LLM argument values."""

    service = composer_service_with_real_sessions
    raw_extra_context = "RAW_EXTRA_CONTEXT: private traceback and source excerpt"
    responses = [
        _advisor_tool_call_response(
            "call_advisor_extra_arg",
            extra_args={"full_context": raw_extra_context},
        ),
        _text_response("Done."),
    ]

    async def _fake_llm(_messages: Any, _tools: Any) -> Any:
        return responses.pop(0)

    async def _fake_advisor(**_kwargs: Any) -> Any:
        return _advisor_model_response()

    monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", _fake_advisor)

    result = await _run_one_turn(
        service,
        llm=_fake_llm,
        session_id=result_session_id,
    )

    assert len(result.persisted_assistant_tool_calls) == 1
    persisted_call = result.persisted_assistant_tool_calls[0]
    persisted_args = json.loads(persisted_call["function"]["arguments"])
    assert "full_context" not in persisted_args
    assert persisted_args["_unknown_arguments"] == "<redacted-unknown-argument-key>"
    persisted_blob = json.dumps(
        {
            "assistant_tool_calls": result.persisted_assistant_tool_calls,
            "tool_rows": result.persisted_tool_row_content,
        },
        sort_keys=True,
    )
    assert "full_context" not in persisted_blob
    assert raw_extra_context not in persisted_blob


@pytest.mark.asyncio
async def test_step2_preserves_absent_raw_content_as_none(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_tool_call_with_no_content: Any,
    result_session_id: str,
) -> None:
    """Missing assistant content remains NULL in raw_content."""

    await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_tool_call_with_no_content,
        session_id=result_session_id,
    )

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    audit_outcome = composer_service_with_real_sessions._phase3_last_audit_outcome  # type: ignore[attr-defined]
    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        row = conn.execute(
            text("SELECT raw_content FROM chat_messages WHERE id = :id"),
            {"id": audit_outcome.assistant_id},
        ).one()
    assert row.raw_content is None


@pytest.mark.asyncio
async def test_step2_first_tool_turn_uses_existing_current_state_id(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_two_tool_calls: Any,
    result_session_id: str,
) -> None:
    """First tool-call persistence must guard against the current state row."""

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    state_record = await sessions_service.save_composition_state(
        result_session_id,
        CompositionStateData(is_valid=False),
        provenance="session_seed",
    )

    await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_two_tool_calls,
        session_id=result_session_id,
        current_state_id=str(state_record.id),
    )

    assert composer_service_with_real_sessions._phase3_last_expected_current_state_id == str(state_record.id)  # type: ignore[attr-defined]
    audit_outcome = composer_service_with_real_sessions._phase3_last_audit_outcome  # type: ignore[attr-defined]
    assert audit_outcome.current_state_id == str(state_record.id)


@pytest.mark.asyncio
async def test_step2_dispatches_one_persist_compose_turn_async_per_turn(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_two_tool_calls: Any,
    result_session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One tool-call turn is committed by one persist_compose_turn_async call.

    The invariant is the service call boundary, not the incidental number
    of SQLAlchemy transactions opened by adjacent preference or audit
    bookkeeping.  Count the production persistence method directly so
    legitimate neighbouring DB reads/writes do not make this test brittle.
    """

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    _patch_auto_commit_preferences(monkeypatch, sessions_service)
    persist_calls = 0
    original_persist = sessions_service.persist_compose_turn_async

    async def _count_persist_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal persist_calls
        persist_calls += 1
        return await original_persist(*args, **kwargs)

    monkeypatch.setattr(sessions_service, "persist_compose_turn_async", _count_persist_call)

    await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_two_tool_calls,
        session_id=result_session_id,
    )

    assert persist_calls == 1


@pytest.mark.asyncio
async def test_cancellation_during_sync_tool_waits_for_result_audit_persist(
    composer_service_with_real_sessions: ComposerServiceImpl,
    result_session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled awaiter must not split a sync side effect from P4 persist."""

    service = composer_service_with_real_sessions
    sessions_service = service._sessions_service  # type: ignore[attr-defined]
    _patch_auto_commit_preferences(monkeypatch, sessions_service)

    loop = asyncio.get_running_loop()
    worker_started = asyncio.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    worker_invocations = 0

    def _blocking_tool(*args: Any, **_kwargs: Any) -> ToolResult:
        nonlocal worker_invocations
        worker_invocations += 1
        state = cast(Any, args[2])
        loop.call_soon_threadsafe(worker_started.set)
        if not release_worker.wait(timeout=5.0):
            raise TimeoutError("test worker was never released")
        worker_finished.set()
        return ToolResult(
            success=True,
            updated_state=replace(state, version=state.version + 1),
            validation=ValidationSummary(
                is_valid=True,
                errors=(),
                warnings=(),
                suggestions=(),
                semantic_contracts=(),
            ),
            affected_nodes=(),
        )

    monkeypatch.setattr("elspeth.web.composer.tool_batch.execute_tool", _blocking_tool)

    llm_calls = 0

    async def _llm(_messages: Any, _tools: Any) -> Any:
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_cancel_during_worker",
                                    function=SimpleNamespace(
                                        name="set_metadata",
                                        arguments=json.dumps({"patch": {"name": "Committed before cancel"}}),
                                    ),
                                ),
                                SimpleNamespace(
                                    id="call_must_not_start_after_cancel",
                                    function=SimpleNamespace(
                                        name="set_metadata",
                                        arguments=json.dumps({"patch": {"name": "Must not run"}}),
                                    ),
                                ),
                            ],
                        )
                    )
                ]
            )
        return _text_response("must not be reached after cancellation")

    compose_task = asyncio.create_task(
        _run_one_turn(
            service,
            llm=_llm,
            session_id=result_session_id,
        )
    )
    await asyncio.wait_for(worker_started.wait(), timeout=2.0)
    compose_task.cancel()
    await asyncio.sleep(0)

    try:
        assert not compose_task.done(), "cancellation escaped while the synchronous tool still owned an in-flight side effect"
    finally:
        release_worker.set()
        await asyncio.to_thread(worker_finished.wait, 5.0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(compose_task, timeout=5.0)

    assert worker_invocations == 1, "deferred cancellation must not start another tool"
    assert llm_calls == 1, "deferred cancellation must stop before another model turn"
    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        persisted_rows = list(
            conn.execute(
                select(
                    chat_messages_table.c.role,
                    chat_messages_table.c.tool_calls,
                    chat_messages_table.c.tool_call_id,
                    chat_messages_table.c.content,
                )
                .where(chat_messages_table.c.session_id == result_session_id)
                .where(chat_messages_table.c.role.in_(("assistant", "tool")))
                .order_by(chat_messages_table.c.sequence_no)
            ).mappings()
        )

    assert [row["role"] for row in persisted_rows] == ["assistant", "tool"]
    assert len(persisted_rows[0]["tool_calls"]) == 1
    assert persisted_rows[0]["tool_calls"][0]["id"] == "call_cancel_during_worker"
    assert persisted_rows[1]["tool_call_id"] == "call_cancel_during_worker"
    assert json.loads(persisted_rows[1]["content"])["success"] is True
    assert await sessions_service.get_current_state(UUID(result_session_id)) is not None


@pytest.mark.asyncio
async def test_deferred_cancellation_survives_child_failure() -> None:
    """A child failure after cancellation is deferred must not swallow the cancel.

    When a disconnect or external cancellation arrives while the shielded
    dispatch/persist section runs and the child then raises (e.g. an audit
    persistence failure), the exception from ``asyncio.shield(task)`` would
    bypass ``deferred``. Python never redelivers a caught CancelledError on
    its own, so the route would finish on the child's error path with the
    task's cancellation requests still pending — swallowing an operator or
    shutdown cancel. Cancellation must win; the child failure rides along
    as ``__cause__`` for diagnosis.
    """
    from elspeth.web.composer.service import _await_tool_turn_with_deferred_cancellation

    cancellation_requested = asyncio.Event()
    proceed_to_fail = asyncio.Event()

    async def child() -> str:
        await proceed_to_fail.wait()
        raise RuntimeError("audit persistence failed")

    captured: dict[str, BaseException] = {}

    async def awaiter() -> None:
        try:
            await _await_tool_turn_with_deferred_cancellation(
                child(),
                cancellation_requested=cancellation_requested,
            )
        except BaseException as exc:
            captured["exc"] = exc
            raise
        raise AssertionError("unreachable — the child never succeeds")

    task = asyncio.get_running_loop().create_task(awaiter())
    await asyncio.sleep(0)  # awaiter parked on the shield
    task.cancel()
    # The helper catches the CancelledError, defers it, and re-awaits the
    # shielded child; cancellation_requested is the deterministic sync point.
    await asyncio.wait_for(cancellation_requested.wait(), timeout=2.0)
    proceed_to_fail.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    exc = captured["exc"]
    assert isinstance(exc, asyncio.CancelledError), f"the child failure replaced the deferred cancellation: {exc!r}"
    assert isinstance(exc.__cause__, RuntimeError), "the child failure must stay diagnosable as the cancellation's __cause__"
    assert task.cancelled(), "the awaiting task must finish as genuinely cancelled"


@pytest.mark.asyncio
async def test_step2_does_not_call_legacy_add_message_inside_loop(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_two_tool_calls: Any,
    result_session_id: str,
    add_message_spy: list[str],
) -> None:
    """The compose loop does not use SessionService.add_message for tool rows."""

    await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_two_tool_calls,
        session_id=result_session_id,
    )

    assert not any(frame.endswith(":_compose_loop") for frame in add_message_spy)


@pytest.mark.asyncio
async def test_step2_plugin_crash_carries_failed_turn_metadata(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_runtime_error_on_second: Any,
    result_session_id: str,
) -> None:
    """Plugin crashes raised after dispatch expose the persisted assistant id."""

    with pytest.raises(ComposerPluginCrashError) as excinfo:
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=fake_llm_runtime_error_on_second,
            session_id=result_session_id,
        )

    failed_turn = excinfo.value.failed_turn
    assert failed_turn is not None
    assert failed_turn.assistant_message_id is not None
    assert failed_turn.tool_calls_attempted == 3


@pytest.mark.asyncio
async def test_step2_audit_integrity_error_carries_failed_turn_metadata(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_two_tool_calls: Any,
    result_session_id: str,
    inject_commit_OperationalError: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit failures from the single dispatch keep route-visible turn context."""

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    _patch_auto_commit_preferences(monkeypatch, sessions_service)
    # Phase 5b Task 5 follow-on: skip the F-5c skill_markdown_history
    # upsert so the next-commit-OperationalError listener catches the
    # persist_compose_turn_async commit (the test's actual target), not
    # the audit-archive upsert that fires once per service instance.
    composer_service_with_real_sessions._skill_markdown_history_upserted = True  # type: ignore[attr-defined]
    inject_commit_OperationalError(sessions_service._engine)  # type: ignore[attr-defined]

    with pytest.raises(AuditIntegrityError) as excinfo:
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=fake_llm_two_tool_calls,
            session_id=result_session_id,
        )

    assert excinfo.value.failed_turn is not None
    assert excinfo.value.failed_turn.assistant_message_id is None
    assert excinfo.value.failed_turn.tool_calls_attempted == 2
    assert excinfo.value.failed_turn.tool_responses_persisted == 0


@pytest.mark.asyncio
async def test_plugin_crash_unwind_commit_failure_remains_unpersisted_and_retains_current_invocations(
    composer_service_with_real_sessions: ComposerServiceImpl,
    fake_llm_runtime_error_on_second: Any,
    result_session_id: str,
    inject_commit_OperationalError: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rolled-back unwind write cannot suppress the crash audit evidence."""

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    _patch_auto_commit_preferences(monkeypatch, sessions_service)
    composer_service_with_real_sessions._skill_markdown_history_upserted = True  # type: ignore[attr-defined]

    persisted_flags: list[bool] = []
    original_persist = composer_service_with_real_sessions._persist_turn_audit  # type: ignore[attr-defined]

    async def _capture_persist_outcome(**kwargs: Any) -> Any:
        outcome = await original_persist(**kwargs)
        persisted_flags.append(outcome.persisted_tool_call_turn)
        return outcome

    monkeypatch.setattr(composer_service_with_real_sessions, "_persist_turn_audit", _capture_persist_outcome)
    inject_commit_OperationalError(sessions_service._engine)  # type: ignore[attr-defined]

    with pytest.raises(ComposerPluginCrashError) as excinfo:
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=fake_llm_runtime_error_on_second,
            session_id=result_session_id,
        )

    assert persisted_flags == [False]
    assert [invocation.tool_call_id for invocation in excinfo.value.tool_invocations] == ["call_ok", "call_crash"]
    assert excinfo.value.failed_turn is not None
    assert excinfo.value.failed_turn.assistant_message_id is None
    assert excinfo.value.failed_turn.tool_calls_attempted == 3
    assert excinfo.value.failed_turn.tool_responses_persisted == 0

    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text("SELECT role, tool_call_id FROM chat_messages WHERE session_id = :session_id AND role IN ('assistant', 'tool')"),
            {"session_id": result_session_id},
        ).fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_unwind_failure_retains_only_current_turn_after_committed_prefix(
    composer_service_with_real_sessions: ComposerServiceImpl,
    result_session_id: str,
    inject_commit_OperationalError: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earlier committed invocations are not duplicated on unwind recovery."""

    from elspeth.web.composer import tool_batch

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    _patch_auto_commit_preferences(monkeypatch, sessions_service)
    composer_service_with_real_sessions._skill_markdown_history_upserted = True  # type: ignore[attr-defined]

    original_execute = tool_batch.execute_tool
    execute_calls = 0

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 2:
            raise RuntimeError("second-turn plugin crash")
        return original_execute(tool_name, *args, **kwargs)

    monkeypatch.setattr(tool_batch, "execute_tool", _execute)

    original_persist = sessions_service.persist_compose_turn_async
    persist_calls = 0

    async def _fail_second_persist(**kwargs: Any) -> Any:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            inject_commit_OperationalError(sessions_service._engine)  # type: ignore[attr-defined]
        return await original_persist(**kwargs)

    monkeypatch.setattr(sessions_service, "persist_compose_turn_async", _fail_second_persist)

    responses = [
        _metadata_tool_response("call_committed", "committed"),
        _metadata_tool_response("call_unpersisted_crash", "crash"),
    ]

    async def _llm(_messages: Any, _tools: Any) -> Any:
        return responses.pop(0)

    with pytest.raises(ComposerPluginCrashError) as excinfo:
        await _run_one_turn(
            composer_service_with_real_sessions,
            llm=_llm,
            session_id=result_session_id,
        )

    assert persist_calls == 2
    assert [invocation.tool_call_id for invocation in excinfo.value.tool_invocations] == ["call_unpersisted_crash"]
    with sessions_service._engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text(
                "SELECT role, tool_call_id FROM chat_messages WHERE session_id = :session_id AND role IN ('assistant', 'tool') ORDER BY sequence_no"
            ),
            {"session_id": result_session_id},
        ).fetchall()
    assert rows == [("assistant", None), ("tool", "call_committed")]
