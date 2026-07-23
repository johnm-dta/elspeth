"""Tests for ``ComposerServiceImpl.explain_run_diagnostics`` error handling.

The exception-catching surface is widened from ``LiteLLMAPIError`` alone to
include LiteLLM's policy-gate exceptions (``BudgetExceededError``,
``BlockedPiiEntityError``, ``GuardrailRaisedException``). These do not subclass
``LiteLLMAPIError`` at the pinned LiteLLM version — without the wider catch,
they would escape this method as raw ``litellm.exceptions`` types and break
the route-layer contract that diagnostics failures surface as
``ComposerServiceError``.

Ticket: elspeth-ab3ad30e87.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from litellm.exceptions import (
    BlockedPiiEntityError,
    BudgetExceededError,
    GuardrailRaisedException,
)

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.protocol import ComposerServiceError
from elspeth.web.composer.service import ComposerServiceImpl


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        pytest.param("success", ComposerLLMCallStatus.SUCCESS, id="success"),
        pytest.param("timeout", ComposerLLMCallStatus.TIMEOUT, id="timeout"),
        pytest.param("malformed", ComposerLLMCallStatus.MALFORMED_RESPONSE, id="malformed"),
        pytest.param("provider_error", ComposerLLMCallStatus.API_ERROR, id="provider-error"),
    ],
)
@pytest.mark.asyncio
async def test_explain_run_diagnostics_records_each_outbound_call_once(
    composer_service_without_sessions_service: ComposerServiceImpl,
    outcome: str,
    expected_status: ComposerLLMCallStatus,
) -> None:
    service = composer_service_without_sessions_service
    recorder = BufferingRecorder()
    snapshot: dict[str, object] = {
        "run_id": "SECRET_DIAGNOSTICS_RUN_ID",
        "status": "failed",
        "rows": [],
    }

    async def fake_call_text_llm(_messages: list[dict[str, str]]) -> object:
        if outcome == "timeout":
            raise TimeoutError
        if outcome == "provider_error":
            raise BudgetExceededError(current_cost=10.0, max_budget=1.0, message="provider secret must not persist")
        if outcome == "malformed":
            return SimpleNamespace(model="test/diagnostics-model", choices=[])
        return SimpleNamespace(
            model="test/diagnostics-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content="The run is processing one row."))],
        )

    with patch.object(service, "_call_text_llm", new=fake_call_text_llm):
        if outcome == "success":
            assert await service.explain_run_diagnostics(snapshot, recorder=recorder) == "The run is processing one row."
        else:
            with pytest.raises(ComposerServiceError):
                await service.explain_run_diagnostics(snapshot, recorder=recorder)

    assert len(recorder.llm_calls) == 1
    call = recorder.llm_calls[0]
    assert call.status is expected_status
    assert call.tools_spec_hash is None
    assert call.declared_tool_names == ()
    assert "SECRET_DIAGNOSTICS_RUN_ID" not in repr(call)
    assert "provider secret must not persist" not in repr(call)


@pytest.mark.parametrize(
    "exc_factory",
    [
        pytest.param(
            lambda: BudgetExceededError(current_cost=10.0, max_budget=1.0, message="budget exceeded"),
            id="BudgetExceededError",
        ),
        pytest.param(
            lambda: BlockedPiiEntityError(entity_type="EMAIL_ADDRESS", guardrail_name="presidio"),
            id="BlockedPiiEntityError",
        ),
        pytest.param(
            lambda: GuardrailRaisedException(
                guardrail_name="bedrock-guardrails",
                message="content blocked",
            ),
            id="GuardrailRaisedException",
        ),
    ],
)
@pytest.mark.asyncio
async def test_explain_run_diagnostics_wraps_litellm_policy_exceptions(
    composer_service_without_sessions_service: ComposerServiceImpl,
    exc_factory: Any,
) -> None:
    """Policy-gate LiteLLM exceptions surface as ``ComposerServiceError``.

    Without the widened catch, the raw ``litellm.exceptions`` class would
    escape ``explain_run_diagnostics`` and the route handler would translate
    it into a 500 with provider-specific messaging instead of a 502 with the
    composer's normalised error envelope.
    """
    service = composer_service_without_sessions_service
    litellm_exception = exc_factory()
    snapshot: dict[str, object] = {
        "run_id": "run-test-1",
        "status": "failed",
        "row_count": 0,
        "rows": [],
    }

    async def fake_call_text_llm(_messages: list[dict[str, str]]) -> object:
        raise litellm_exception

    with (
        patch.object(
            service,
            "_call_text_llm",
            new=fake_call_text_llm,
        ),
        pytest.raises(ComposerServiceError) as exc_info,
    ):
        await service.explain_run_diagnostics(snapshot)

    # Wrap message mirrors the existing ``LLM unavailable ({type})`` pattern.
    assert "LLM unavailable" in str(exc_info.value)
    assert type(litellm_exception).__name__ in str(exc_info.value)


@pytest.mark.parametrize("exc_cls", [RuntimeError, ValueError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_unrelated_exceptions_propagate(
    composer_service_without_sessions_service: ComposerServiceImpl,
    exc_cls: type[BaseException],
) -> None:
    """Widened catch must NOT swallow unrelated exception types.

    Pins the negative side of commit 4dacddfda's widening — RuntimeError,
    ValueError, asyncio.CancelledError, etc. MUST escape unwrapped so the
    route layer's normalised error envelope only covers the documented
    LiteLLM policy-gate set.
    """
    service = composer_service_without_sessions_service
    unrelated_exception = exc_cls("boom")
    snapshot: dict[str, object] = {"run_id": "run-x", "status": "failed", "row_count": 0, "rows": []}

    async def fake_call_text_llm(_messages: list[dict[str, str]]) -> object:
        raise unrelated_exception

    with (
        patch.object(service, "_call_text_llm", new=fake_call_text_llm),
        pytest.raises(exc_cls),
    ):
        await service.explain_run_diagnostics(snapshot)
