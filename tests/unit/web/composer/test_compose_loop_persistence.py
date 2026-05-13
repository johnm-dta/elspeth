"""Compose-loop Step 1/2/3 unit tests (spec §5.2.1)."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from sqlalchemy import text

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.redaction import redact_tool_call_arguments, redact_tool_call_response
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.sessions.protocol import CompositionStateData


async def _run_one_turn(
    service: ComposerServiceImpl,
    *,
    llm: Any,
    session_id: str,
    current_state_id: str | None = None,
) -> Any:
    driver = cast(Any, service)
    return await driver._run_one_turn_for_test(llm=llm, session_id=session_id, current_state_id=current_state_id)


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
    sqlalchemy_event_listener: Any,
) -> None:
    """One tool-call turn is committed by one persist_compose_turn_async call."""

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
    counts = sqlalchemy_event_listener(sessions_service._engine)  # type: ignore[attr-defined]

    await _run_one_turn(
        composer_service_with_real_sessions,
        llm=fake_llm_two_tool_calls,
        session_id=result_session_id,
    )

    assert counts["begin"] == 1
    assert counts["commit"] == 1
    assert counts["rollback"] == 0


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
) -> None:
    """Audit failures from the single dispatch keep route-visible turn context."""

    sessions_service = composer_service_with_real_sessions._sessions_service  # type: ignore[attr-defined]
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
