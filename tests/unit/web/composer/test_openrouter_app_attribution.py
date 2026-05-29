"""OpenRouter app-attribution headers on the composer LiteLLM path.

The composer reaches OpenRouter through LiteLLM, which brands every request
with its own attribution headers (``HTTP-Referer: https://litellm.ai`` /
``X-Title: liteLLM``) unless the caller overrides them. Without an override the
OpenRouter dashboard attributes all composer ("orchestrator") traffic to
LiteLLM instead of ELSPETH. The LLM transform *plugins* avoid this because they
speak raw HTTP and set the identity headers themselves; this suite pins the
parity fix for the LiteLLM-routed composer path.

Primary source (https://openrouter.ai/docs/app-attribution): ``HTTP-Referer``
is the primary ranking identifier and ``X-OpenRouter-Title`` is the current
display-name header, with ``X-Title`` "still supported for backwards
compatibility" — which is the spelling LiteLLM defaults to ``liteLLM``. The fix
therefore overrides all three so no LiteLLM branding survives whichever header
OpenRouter honours.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.plugins.transforms.llm.providers.openrouter import (
    OPENROUTER_APP_REFERER,
    OPENROUTER_APP_TITLE,
)
from elspeth.web.composer.service import (
    _apply_openrouter_app_identity,
    _litellm_acompletion,
)


def test_injects_elspeth_identity_for_openrouter_model() -> None:
    kwargs: dict[str, Any] = {
        "model": "openrouter/openai/gpt-5.4-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }

    _apply_openrouter_app_identity(kwargs)

    headers = kwargs["extra_headers"]
    assert headers["HTTP-Referer"] == OPENROUTER_APP_REFERER
    # Current OpenRouter header (matches the plugin path) ...
    assert headers["X-OpenRouter-Title"] == OPENROUTER_APP_TITLE
    # ... plus the legacy spelling LiteLLM defaults to "liteLLM", overridden so
    # no LiteLLM branding leaks through.
    assert headers["X-Title"] == OPENROUTER_APP_TITLE


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "anthropic/claude-sonnet-4-6",
        "azure/gpt-4o",
    ],
)
def test_does_not_touch_non_openrouter_models(model: str) -> None:
    kwargs: dict[str, Any] = {"model": model, "messages": []}

    _apply_openrouter_app_identity(kwargs)

    assert "extra_headers" not in kwargs


def test_missing_model_is_left_for_litellm_to_reject() -> None:
    # The wrapper must not invent attribution for a malformed call; LiteLLM
    # owns the "model is required" error.
    kwargs: dict[str, Any] = {"messages": []}

    _apply_openrouter_app_identity(kwargs)

    assert "extra_headers" not in kwargs


def test_caller_supplied_headers_win() -> None:
    kwargs: dict[str, Any] = {
        "model": "openrouter/openai/gpt-5.4-mini",
        "messages": [],
        "extra_headers": {"HTTP-Referer": "https://example.test/custom"},
    }

    _apply_openrouter_app_identity(kwargs)

    headers = kwargs["extra_headers"]
    # An explicit caller override is preserved; we only fill identity keys we own.
    assert headers["HTTP-Referer"] == "https://example.test/custom"
    assert headers["X-OpenRouter-Title"] == OPENROUTER_APP_TITLE
    assert headers["X-Title"] == OPENROUTER_APP_TITLE


@pytest.mark.asyncio
async def test_wrapper_forwards_identity_headers_to_litellm() -> None:
    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "ok"

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
        result = await _litellm_acompletion(
            model="openrouter/openai/gpt-5.4-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result == "ok"
    headers = captured["extra_headers"]
    assert headers["HTTP-Referer"] == OPENROUTER_APP_REFERER
    assert headers["X-OpenRouter-Title"] == OPENROUTER_APP_TITLE
    assert headers["X-Title"] == OPENROUTER_APP_TITLE
