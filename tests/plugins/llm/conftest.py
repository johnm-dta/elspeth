# tests/plugins/llm/conftest.py
"""Shared fixtures and helpers for LLM plugin tests."""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

if TYPE_CHECKING:
    import httpx

from elspeth.contracts.identity import TokenInfo
from elspeth.plugins.context import PluginContext

# Common schema config used across LLM tests
DYNAMIC_SCHEMA = {"fields": "dynamic"}


def make_azure_multi_query_config(**overrides: Any) -> dict[str, Any]:
    """Create valid Azure Multi-Query config with optional overrides.

    This is the canonical config factory for AzureMultiQueryLLMTransform tests.
    Use overrides to customize specific fields.
    """
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Input: {{ row.input_1 }}\nCriterion: {{ row.criterion.name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"]},
            {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym", "cs2_hist"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG"},
            {"name": "treatment", "code": "TREAT"},
        ],
        "response_format": "standard",
        "output_mapping": {
            "score": {"suffix": "score", "type": "integer"},
            "rationale": {"suffix": "rationale", "type": "string"},
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for tests
        "pool_size": 4,
    }
    config.update(overrides)
    return config


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data={},  # Not used in these tests
    )


def make_plugin_context(
    state_id: str = "state-123",
    token: TokenInfo | None = None,
) -> PluginContext:
    """Create a PluginContext with mocked landscape."""
    mock_landscape = Mock()
    mock_landscape.record_external_call = Mock()
    mock_landscape.record_call = Mock()
    return PluginContext(
        run_id="run-123",
        landscape=mock_landscape,
        state_id=state_id,
        config={},
        token=token or make_token("row-1"),
    )


def _build_chaosllm_response(
    chaosllm_server,
    request: dict[str, Any],
    *,
    mode_override: str | None = None,
    template_override: str | None = None,
    usage_override: dict[str, int] | None = None,
) -> Mock:
    response_dict = chaosllm_server.server._response_generator.generate(
        request,
        mode_override=mode_override,
        template_override=template_override,
    ).to_dict()

    if usage_override is not None:
        response_dict["usage"] = {
            "prompt_tokens": usage_override["prompt_tokens"],
            "completion_tokens": usage_override["completion_tokens"],
        }

    mock_usage = Mock()
    mock_usage.prompt_tokens = response_dict["usage"]["prompt_tokens"]
    mock_usage.completion_tokens = response_dict["usage"]["completion_tokens"]

    mock_message = Mock()
    mock_message.content = response_dict["choices"][0]["message"]["content"]

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.model = response_dict["model"]
    mock_response.usage = mock_usage
    mock_response.model_dump = Mock(return_value=response_dict)
    return mock_response


@contextmanager
def chaosllm_azure_openai_client(
    chaosllm_server,
    *,
    mode: str = "echo",
    template_override: str | None = None,
    usage_override: dict[str, int] | None = None,
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Patch AzureOpenAI to use ChaosLLM response generation (no HTTP)."""

    def make_response(**kwargs: Any) -> Mock:
        if side_effect is not None:
            raise side_effect
        request = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"],
        }
        if "max_tokens" in kwargs:
            request["max_tokens"] = kwargs["max_tokens"]
        return _build_chaosllm_response(
            chaosllm_server,
            request,
            mode_override=mode,
            template_override=template_override,
            usage_override=usage_override,
        )

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client


@contextmanager
def chaosllm_azure_openai_responses(
    chaosllm_server,
    responses: list[dict[str, Any] | str],
    *,
    usage_override: dict[str, int] | None = None,
) -> Generator[Mock, None, None]:
    """Patch AzureOpenAI to return a sequence of ChaosLLM-generated JSON responses."""
    response_cycle = itertools.cycle(responses)
    lock = threading.Lock()

    def make_response(**kwargs: Any) -> Mock:
        with lock:
            payload = next(response_cycle)
        template_override = payload if isinstance(payload, str) else json.dumps(payload)
        request = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"],
        }
        if "max_tokens" in kwargs:
            request["max_tokens"] = kwargs["max_tokens"]
        return _build_chaosllm_response(
            chaosllm_server,
            request,
            mode_override="template",
            template_override=template_override,
            usage_override=usage_override,
        )

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client


@contextmanager
def chaosllm_azure_openai_sequence(
    chaosllm_server,
    response_factory,
    *,
    usage_override: dict[str, int] | None = None,
) -> Generator[tuple[Mock, list[int], Mock], None, None]:
    """Patch AzureOpenAI with a response factory (supports delays)."""
    call_count = [0]
    lock = threading.Lock()

    def make_response(**kwargs: Any) -> Mock:
        with lock:
            call_count[0] += 1
            current = call_count[0]
        request = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"],
        }
        if "max_tokens" in kwargs:
            request["max_tokens"] = kwargs["max_tokens"]

        payload = response_factory(current, request)
        delay_ms = 0.0
        if isinstance(payload, tuple):
            payload, delay_ms = payload
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

        template_override = payload if isinstance(payload, str) else json.dumps(payload)
        return _build_chaosllm_response(
            chaosllm_server,
            request,
            mode_override="template",
            template_override=template_override,
            usage_override=usage_override,
        )

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client, call_count, mock_azure_class


def _build_chaosllm_httpx_response(
    chaosllm_server,
    request: dict[str, Any],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    mode_override: str | None = None,
    template_override: str | None = None,
    raw_body: str | bytes | None = None,
    usage_override: dict[str, int] | None = None,
) -> httpx.Response:
    import httpx

    response_dict = chaosllm_server.server._response_generator.generate(
        request,
        mode_override=mode_override,
        template_override=template_override,
    ).to_dict()

    if usage_override is not None:
        response_dict["usage"] = {
            "prompt_tokens": usage_override["prompt_tokens"],
            "completion_tokens": usage_override["completion_tokens"],
        }

    if raw_body is None:
        content = json.dumps(response_dict).encode("utf-8")
        response_headers = headers or {"content-type": "application/json"}
    else:
        if isinstance(raw_body, bytes):
            content = raw_body
        else:
            content = raw_body.encode("utf-8")
        response_headers = headers or {"content-type": "text/plain"}

    request_obj = httpx.Request("POST", "http://testserver/v1/chat/completions")
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=response_headers,
        request=request_obj,
    )


@contextmanager
def chaosllm_openrouter_http_responses(
    chaosllm_server,
    responses: list[dict[str, Any] | str | httpx.Response],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    usage_override: dict[str, int] | None = None,
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Patch httpx.Client to return ChaosLLM-generated responses (no HTTP)."""
    import httpx

    response_cycle = itertools.cycle(responses)
    lock = threading.Lock()

    def make_response(*args: Any, **kwargs: Any) -> httpx.Response:
        if side_effect is not None:
            raise side_effect
        with lock:
            payload = next(response_cycle)

        if isinstance(payload, httpx.Response):
            return payload

        template_override = payload if isinstance(payload, str) else json.dumps(payload)
        request_body = kwargs.get("json") or {}
        request = {
            "model": request_body.get("model"),
            "messages": request_body.get("messages", []),
            "temperature": request_body.get("temperature"),
            "max_tokens": request_body.get("max_tokens"),
        }
        return _build_chaosllm_httpx_response(
            chaosllm_server,
            request,
            status_code=status_code,
            headers=headers,
            mode_override="template",
            template_override=template_override,
            usage_override=usage_override,
        )

    with patch("httpx.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.post.side_effect = make_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=None)
        yield mock_client


def chaosllm_openrouter_httpx_response(
    chaosllm_server,
    request: dict[str, Any],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    template_override: str | None = None,
    raw_body: str | bytes | None = None,
    usage_override: dict[str, int] | None = None,
) -> httpx.Response:
    """Create a single httpx.Response using ChaosLLM response generation."""
    return _build_chaosllm_httpx_response(
        chaosllm_server,
        request,
        status_code=status_code,
        headers=headers,
        mode_override="template",
        template_override=template_override,
        raw_body=raw_body,
        usage_override=usage_override,
    )
