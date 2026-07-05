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
from typing import Any
from unittest.mock import patch

import pytest
from litellm.exceptions import (
    BlockedPiiEntityError,
    BudgetExceededError,
    GuardrailRaisedException,
)

from elspeth.web.composer.protocol import ComposerServiceError
from elspeth.web.composer.service import ComposerServiceImpl


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
