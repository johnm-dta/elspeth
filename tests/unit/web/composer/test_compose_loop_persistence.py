"""Compose-loop Step 1/2/3 unit tests (spec §5.2.1)."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from sqlalchemy import text

from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.redaction import redact_tool_call_arguments, redact_tool_call_response
from elspeth.web.composer.service import ComposerServiceImpl


async def _run_one_turn(service: ComposerServiceImpl, *, llm: Any, session_id: str) -> Any:
    driver = cast(Any, service)
    return await driver._run_one_turn_for_test(llm=llm, session_id=session_id)


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
