# tests/unit/plugins/llm/conftest.py
"""Shared fixtures and helpers for LLM plugin tests."""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import httpx

from elspeth.contracts.identity import TokenInfo
from elspeth.testing import make_pipeline_row

# Re-export chaosllm_server fixture so LLM tests can use it
# ruff: noqa: F811  # Helper functions receive chaosllm_server as parameter, shadowing the re-export
from tests.fixtures.chaosllm import chaosllm_server  # noqa: F401

# Common schema config used across LLM tests
DYNAMIC_SCHEMA = {"mode": "observed"}


class _CallRecord:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs


class _SyncCallRecorder:
    def __init__(self, return_value: Any = None, side_effect: Callable[..., Any] | Exception | None = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args: _CallRecord | None = None
        self.call_args_list: list[_CallRecord] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        record = _CallRecord(args, kwargs)
        self.call_args = record
        self.call_args_list.append(record)
        if isinstance(self.side_effect, Exception):
            raise self.side_effect
        if self.side_effect is not None:
            return self.side_effect(*args, **kwargs)
        return self.return_value

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)


class _AzureOpenAIClientDouble:
    def __init__(self, create_response: Callable[..., Any]) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_SyncCallRecorder(side_effect=create_response)))
        self.close = _SyncCallRecorder()


class _OpenRouterHTTPClientDouble:
    def __init__(self, post_response: Callable[..., Any]) -> None:
        self.post = _SyncCallRecorder(side_effect=post_response)
        self.close = _SyncCallRecorder()

    def __enter__(self) -> _OpenRouterHTTPClientDouble:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row({}),
    )


def _build_chaosllm_response(
    chaosllm_server,
    request: dict[str, Any],
    *,
    mode_override: str | None = None,
    template_override: str | None = None,
    usage_override: dict[str, int] | None = None,
) -> SimpleNamespace:
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

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=response_dict["choices"][0]["message"]["content"]),
                finish_reason=response_dict["choices"][0].get("finish_reason"),
            )
        ],
        model=response_dict["model"],
        usage=SimpleNamespace(
            prompt_tokens=response_dict["usage"]["prompt_tokens"],
            completion_tokens=response_dict["usage"]["completion_tokens"],
        ),
        model_dump=lambda *_args, **_kwargs: response_dict,
    )


@contextmanager
def chaosllm_azure_openai_client(
    chaosllm_server,
    *,
    mode: str = "echo",
    template_override: str | None = None,
    usage_override: dict[str, int] | None = None,
    side_effect: Exception | None = None,
) -> Iterator[_AzureOpenAIClientDouble]:
    """Patch AzureOpenAI to use ChaosLLM response generation (no HTTP)."""

    def make_response(**kwargs: Any) -> SimpleNamespace:
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
        mock_client = _AzureOpenAIClientDouble(make_response)
        mock_azure_class.return_value = mock_client
        yield mock_client


@contextmanager
def chaosllm_azure_openai_responses(
    chaosllm_server,
    responses: list[dict[str, Any] | str],
    *,
    usage_override: dict[str, int] | None = None,
) -> Iterator[_AzureOpenAIClientDouble]:
    """Patch AzureOpenAI to return a sequence of ChaosLLM-generated JSON responses."""
    response_cycle = itertools.cycle(responses)
    lock = threading.Lock()

    def make_response(**kwargs: Any) -> SimpleNamespace:
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
        mock_client = _AzureOpenAIClientDouble(make_response)
        mock_azure_class.return_value = mock_client
        yield mock_client


@contextmanager
def chaosllm_azure_openai_sequence(
    chaosllm_server,
    response_factory,
    *,
    usage_override: dict[str, int] | None = None,
) -> Iterator[tuple[_AzureOpenAIClientDouble, list[int], Any]]:
    """Patch AzureOpenAI with a response factory (supports delays)."""
    call_count = [0]
    lock = threading.Lock()

    def make_response(**kwargs: Any) -> SimpleNamespace:
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
        mock_client = _AzureOpenAIClientDouble(make_response)
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
) -> Iterator[_OpenRouterHTTPClientDouble]:
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
        # httpx.Client() returns mock_client in two usage patterns:
        # 1. Direct: self._client = httpx.Client(...)  (AuditedHTTPClient)
        # 2. Context manager: with httpx.Client(...) as client:  (openrouter_batch)
        # Both must route to the same mock with side_effects configured.
        mock_client = _OpenRouterHTTPClientDouble(make_response)
        mock_client_class.return_value = mock_client
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
