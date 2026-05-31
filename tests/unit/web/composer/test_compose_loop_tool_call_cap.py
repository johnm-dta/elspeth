"""Tests for the per-turn tool-call cap (spec §1.4 NFR / §5.2.1 Step 0)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from elspeth.web.composer.protocol import ComposerConvergenceError
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value


async def _run_one_turn(service: object, *, llm: object, session_id: str) -> Any:
    driver = cast(Any, service)
    return await driver._run_one_turn_for_test(llm=llm, session_id=session_id)


@pytest.mark.asyncio
async def test_cap_exceeded_raises_before_any_tool_execution(
    fake_composer_service: object,
    fake_llm_emitting_n_tool_calls: Any,
    result_session_id: str,
) -> None:
    """The compose loop rejects over-cap turns before dispatching tools."""

    fake_llm = fake_llm_emitting_n_tool_calls(n=17)
    fake_composer_service._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]

    with pytest.raises(ComposerConvergenceError) as excinfo:
        await _run_one_turn(fake_composer_service, llm=fake_llm, session_id=result_session_id)

    assert excinfo.value.reason == "tool_call_cap_exceeded"
    assert excinfo.value.evidence["observed"] == 17
    assert excinfo.value.evidence["cap"] == 16
    assert fake_llm.execute_tool_invocations == 0


@pytest.mark.asyncio
async def test_cap_exceeded_increments_counter(
    fake_composer_service: object,
    fake_llm_emitting_n_tool_calls: Any,
    result_session_id: str,
) -> None:
    """The cap breach increments the composer tool-call-cap counter."""

    telemetry = build_sessions_telemetry()
    fake_composer_service._telemetry = telemetry  # type: ignore[attr-defined]
    fake_composer_service._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]
    fake_llm = fake_llm_emitting_n_tool_calls(n=17)

    with pytest.raises(ComposerConvergenceError):
        await _run_one_turn(fake_composer_service, llm=fake_llm, session_id=result_session_id)

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 1


@pytest.mark.asyncio
async def test_cap_not_exceeded_does_not_increment(
    fake_composer_service: object,
    fake_llm_emitting_n_tool_calls: Any,
    result_session_id: str,
) -> None:
    """At-cap turns are allowed and do not increment the cap counter."""

    telemetry = build_sessions_telemetry()
    fake_composer_service._telemetry = telemetry  # type: ignore[attr-defined]
    fake_composer_service._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]
    fake_llm = fake_llm_emitting_n_tool_calls(n=16)

    await _run_one_turn(fake_composer_service, llm=fake_llm, session_id=result_session_id)

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 0
