"""OpenRouter LLM provider.

Handles raw HTTP transport with full Tier 3 boundary validation:
- JSON parsing with NaN/Infinity rejection
- Content extraction from choices[0].message.content
- Null content → ContentPolicyError
- Non-finite usage values → LLMClientError
- HTTP status code → typed exception mapping

Client caching is per-state_id with a threading lock. Uses AuditedHTTPClient
for transport-level HTTP audit recording and records a logical LLM call row for
chat-completion semantics.
"""

from __future__ import annotations

import json
import math
import time
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import Field, field_validator

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.call_data import LLMCallError, LLMCallRequest, LLMCallResponse
from elspeth.contracts.token_usage import TokenUsage
from elspeth.contracts.value_source import CatalogValueSource, ValueSource
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient
from elspeth.plugins.infrastructure.clients.llm import (
    CONTEXT_LENGTH_PATTERNS,
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url
from elspeth.plugins.transforms.llm.base import LLMConfig
from elspeth.plugins.transforms.llm.model_catalog import MODEL_CATALOG_OPENROUTER
from elspeth.plugins.transforms.llm.provider import LLMQueryResult, ParsedFinishReason, parse_finish_reason
from elspeth.plugins.transforms.llm.validation import reject_nonfinite_constant

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.clients.base import TelemetryEmitCallback

__all__ = [
    "OPENROUTER_APP_REFERER",
    "OPENROUTER_APP_TITLE",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_BASE_URL_APPLIES_WHEN",
    "OpenRouterConfig",
    "OpenRouterLLMProvider",
    "normalize_openrouter_base_url",
]


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
"""Canonical OpenRouter HTTP API base URL used by direct OpenRouter providers."""

OPENROUTER_BASE_URL_APPLIES_WHEN = (("base_url", OPENROUTER_BASE_URL),)
"""Value-source predicate for configs targeting the canonical OpenRouter API."""

OPENROUTER_APP_REFERER = "https://github.com/johnm-dta/elspeth"
"""Canonical public project URL used for OpenRouter app attribution."""

OPENROUTER_APP_TITLE = "Elspeth"
"""Canonical OpenRouter app display title."""


def _http_error_body_text(error: httpx.HTTPStatusError) -> str:
    """Return buffered HTTP error text for internal classification only."""
    try:
        return error.response.text
    except (httpx.ResponseNotRead, RuntimeError):
        return ""


def _summarize_http_error_body(error: httpx.HTTPStatusError) -> str:
    """Return a redacted provider-error-body suffix, or '' when no body exists.

    Provider response bodies are Tier-3 remote text and can contain credentials,
    request echoes, or private provider diagnostics. The full HTTP response body
    remains available through the audited call payload; web-visible exception
    text only records bounded metadata that a body existed.
    """
    body = _http_error_body_text(error)
    if not body:
        return ""
    return f" | provider error body redacted (body_present=true; chars={len(body)})"


def normalize_openrouter_base_url(value: str) -> str:
    """Normalize base URL spellings that runtime HTTP joining treats as identical."""
    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _validate_chat_completion_response(response: httpx.Response) -> tuple[dict[str, Any], str, TokenUsage, ParsedFinishReason, str]:
    """Parse and validate an OpenRouter chat-completion response body."""
    try:
        data = json.loads(response.content, parse_constant=reject_nonfinite_constant)
    except (ValueError, TypeError) as e:
        raise LLMClientError(
            f"Response is not valid JSON: {e}",
            retryable=False,
        ) from e

    if not isinstance(data, dict):
        raise LLMClientError(
            f"Empty or missing choices in response: {type(data).__name__}",
            retryable=False,
        )

    choices = data.get("choices")
    if not choices:
        raise LLMClientError(
            f"Empty or missing choices in response: {list(data.keys())}",
            retryable=False,
        )

    try:
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(
            f"Malformed response structure: {type(e).__name__}: {e}",
            retryable=False,
        ) from e

    if content is None:
        raise ContentPolicyError("LLM returned null content (likely content-filtered by provider)")

    if not isinstance(content, str):
        raise LLMClientError(
            f"Expected string content, got {type(content).__name__}",
            retryable=False,
        )

    if not content.strip():
        raw_fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        if raw_fr == "tool_calls":
            raise LLMClientError(
                "LLM returned tool_calls response (not supported by ELSPETH)",
                retryable=False,
            )
        raise ContentPolicyError(
            f"LLM returned empty content (finish_reason={raw_fr})",
        )

    raw_usage = data.get("usage")
    if isinstance(raw_usage, dict):
        for usage_key, usage_val in raw_usage.items():
            if isinstance(usage_val, float) and not math.isfinite(usage_val):
                raise LLMClientError(
                    f"Non-finite value in usage.{usage_key}: {usage_val}",
                    retryable=False,
                )
    usage = TokenUsage.from_dict(raw_usage)

    raw_finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    finish_reason = parse_finish_reason(str(raw_finish_reason)) if raw_finish_reason is not None else None

    # The provider MUST report which model served the request. Substituting the
    # requested model would fabricate a Tier-3 datum; callers record/use the
    # response model as an audited fact.
    raw_model = data["model"] if "model" in data else None
    if not isinstance(raw_model, str) or not raw_model.strip():
        missing_desc = "missing" if raw_model is None else f"{type(raw_model).__name__}/empty"
        raise LLMClientError(
            f"LLM response 'model' is {missing_desc}, expected non-empty str. Provider returned malformed data at the Tier 3 boundary.",
            retryable=False,
        )
    response_model = raw_model

    return data, content, usage, finish_reason, response_model


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration.

    Extends LLMConfig with OpenRouter API settings:
    - api_key: OpenRouter API key (required)
    - base_url: API base URL (default: https://openrouter.ai/api/v1)
    - timeout_seconds: Request timeout (default: 60.0)

    Tier 2 tracing:
    - tracing: Optional Langfuse configuration (azure_ai not supported for OpenRouter)
    """

    # OpenRouter configs always have provider="openrouter" — narrowed Literal prevents misconfiguration
    provider: Literal["openrouter"] = Field(default="openrouter", description="LLM provider")

    # Override base model to make it required — OpenRouter has no deployment_name fallback
    model: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o')")

    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default=OPENROUTER_BASE_URL,
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Request timeout")

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return normalize_openrouter_base_url(validate_credential_safe_https_url(value, field_name="base_url"))

    # Catalog membership for ``model`` is enforced as a value-source concern,
    # NOT in config construction: the ``CatalogValueSource`` declaration below
    # is checked by the bundle walker at instantiation
    # (engine/orchestrator/preflight.validate_value_source_compliance) and by the
    # composer's per-node prevalidation
    # (web/composer/tools._prevalidate_plugin_options via
    # check_config_value_sources). Both yield a structured, node-attributed
    # finding with a list_models hint — which the composer repair loop consumes —
    # rather than the opaque construction-time PluginConfigError a model_validator
    # would raise. Keeping it out of construction also keeps OpenRouterConfig
    # deterministically constructible without the optional, runtime-primed
    # catalog being available.

    # Tier 2: Plugin-internal tracing (optional, Langfuse only)
    # Azure AI tracing is NOT supported - it auto-instruments the OpenAI SDK,
    # but OpenRouter uses HTTP directly via httpx.
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (langfuse only - azure_ai not supported)",
    )

    # Value-source declaration: ``model`` must appear in the OpenRouter
    # slice of ``litellm.model_list``, BUT only when this config targets
    # the canonical OpenRouter endpoint. When an operator overrides
    # ``base_url`` (e.g. to a private HTTPS OpenAI-compatible gateway), the model identifier
    # semantics are owned by that endpoint — not by litellm's OpenRouter
    # slug list. The ``applies_when`` predicate keeps the catalog check
    # in lock-step with the actual HTTP boundary the runtime targets.
    # ClassVar so Pydantic v2 ignores it.
    VALUE_SOURCES: ClassVar[tuple[ValueSource, ...]] = (
        CatalogValueSource(
            field_name="model",
            catalog_id=MODEL_CATALOG_OPENROUTER,
            applies_when=OPENROUTER_BASE_URL_APPLIES_WHEN,
        ),
    )


class OpenRouterLLMProvider:
    """OpenRouter provider — raw HTTP with Tier 3 validation.

    Responsibilities:
    1. Create/cache AuditedHTTPClient per state_id (thread-safe)
    2. Make HTTP POST to /chat/completions
    3. Parse JSON response with NaN rejection
    4. Validate content, usage, finish_reason at Tier 3 boundary
    5. Map HTTP errors to typed exceptions
    6. Record a logical LLM audit row with the validated request/response
    7. Let validated data flow as LLMQueryResult

    The underlying OpenRouter transport is HTTP, so AuditedHTTPClient records
    the raw transport row. The LLM semantic row is recorded here so
    ``calls.resolved_prompt_template_hash`` remains attached only to
    ``CallType.LLM`` rows.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        timeout_seconds: float = 60.0,
        recorder: PluginAuditWriter,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        limiter: Any = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> None:
        # Pre-build auth headers — avoids storing the raw API key as a named attribute
        self._request_headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": OPENROUTER_APP_REFERER,
            "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
        }
        self._base_url = normalize_openrouter_base_url(validate_credential_safe_https_url(base_url, field_name="base_url"))
        self._timeout = timeout_seconds
        self._recorder = recorder
        self._run_id = run_id
        self._telemetry_emit = telemetry_emit
        self._limiter = limiter
        # Phase 5b Task 9 — cross-DB hash anchor. Forwarded to every HTTP
        # post() call so the Landscape ``calls`` row carries the matching
        # SHA-256.
        self._resolved_prompt_template_hash = resolved_prompt_template_hash

        # Client cache with reference counting for parallel multi-query safety.
        # Multiple parallel queries share the same state_id, so _get_http_client()
        # returns the same cached client. Reference counting ensures the client is
        # only closed when the last query releases it.
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_client_refs: dict[str, int] = {}
        self._http_clients_lock = Lock()

    def execute_query(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        state_id: str,
        token_id: str,
        response_format: dict[str, Any] | None = None,
    ) -> LLMQueryResult:
        """Execute LLM query via OpenRouter HTTP API.

        Full Tier 3 validation pipeline:
        1. HTTP POST with error classification
        2. JSON parse with NaN rejection
        3. Content extraction and null check
        4. Usage validation (non-finite rejection)
        5. Finish reason normalization

        Args:
            response_format: OpenAI-compatible response_format dict
                (e.g., {"type": "json_object"})

        Raises:
            RateLimitError: HTTP 429 (retryable)
            ServerError: HTTP 5xx (retryable)
            NetworkError: Connection/timeout failures (retryable)
            ContentPolicyError: Null content from provider (not retryable)
            LLMClientError: Other failures (not retryable)
        """
        snapshot_state_id = state_id
        llm_request_payload = self._build_llm_request_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        logical_start = time.perf_counter()

        http_client = self._get_http_client(snapshot_state_id, token_id=token_id)
        try:
            # Build request body
            request_body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request_body["max_tokens"] = max_tokens
            if response_format is not None:
                request_body["response_format"] = response_format

            # HTTP call — raise_for_status for error classification
            try:
                response = http_client.post(
                    "/chat/completions",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                detail = _summarize_http_error_body(e)
                if status_code == 429:
                    raise RateLimitError(f"Rate limited (HTTP {status_code}){detail}") from e
                elif status_code >= 500:
                    raise ServerError(f"Server error (HTTP {status_code}){detail}") from e
                else:
                    # Check response body for context length indicators before
                    # falling through to generic LLMClientError. Imports the shared
                    # pattern tuple so the provider classifier and AuditedLLMClient
                    # cannot drift — adding a wording in one place reaches both.
                    error_body = _http_error_body_text(e).lower()
                    if any(p in error_body for p in CONTEXT_LENGTH_PATTERNS):
                        raise ContextLengthError(
                            f"Context length exceeded{detail}",
                        ) from e
                    raise LLMClientError(
                        f"HTTP {status_code}{detail}",
                        retryable=False,
                    ) from e
            except httpx.RequestError as e:
                raise NetworkError(f"Network error: {e}") from e

            data, content, usage, finish_reason, response_model = _validate_chat_completion_response(response)

            result = LLMQueryResult(
                content=content,
                usage=usage,
                model=response_model,
                finish_reason=finish_reason,
            )
            self._record_logical_llm_success(
                state_id=snapshot_state_id,
                started_at=logical_start,
                request_payload=llm_request_payload,
                content=content,
                model=response_model,
                usage=usage,
                raw_response=data,
            )
            return result
        except LLMClientError as exc:
            self._record_logical_llm_error(
                state_id=snapshot_state_id,
                started_at=logical_start,
                request_payload=llm_request_payload,
                exc=exc,
            )
            raise
        finally:
            self._release_http_client(snapshot_state_id)

    def _build_llm_request_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> LLMCallRequest:
        extra_kwargs: dict[str, Any] = {}
        if response_format is not None:
            extra_kwargs["response_format"] = response_format
        return LLMCallRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            provider="openrouter",
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )

    def _record_logical_llm_success(
        self,
        *,
        state_id: str,
        started_at: float,
        request_payload: LLMCallRequest,
        content: str,
        model: str,
        usage: TokenUsage,
        raw_response: dict[str, Any],
    ) -> None:
        """Record the semantic LLM call that the HTTP transport fulfilled."""
        call_index = self._recorder.allocate_call_index(state_id)
        self._recorder.record_call(
            state_id=state_id,
            call_index=call_index,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_payload,
            response_data=LLMCallResponse(
                content=content,
                model=model,
                usage=usage,
                raw_response=raw_response,
            ),
            latency_ms=(time.perf_counter() - started_at) * 1000,
            resolved_prompt_template_hash=self._resolved_prompt_template_hash,
        )

    def _record_logical_llm_error(
        self,
        *,
        state_id: str,
        started_at: float,
        request_payload: LLMCallRequest,
        exc: LLMClientError,
    ) -> None:
        call_index = self._recorder.allocate_call_index(state_id)
        message = str(exc) or type(exc).__name__
        self._recorder.record_call(
            state_id=state_id,
            call_index=call_index,
            call_type=CallType.LLM,
            status=CallStatus.ERROR,
            request_data=request_payload,
            error=LLMCallError(
                type=type(exc).__name__,
                message=message,
                retryable=bool(getattr(exc, "retryable", False)),
            ),
            latency_ms=(time.perf_counter() - started_at) * 1000,
            resolved_prompt_template_hash=self._resolved_prompt_template_hash,
        )

    def runtime_preflight(self, *, operation_id: str, model: str) -> None:
        """Run a minimal audited OpenRouter call under an operation parent."""
        http_client = AuditedHTTPClient(
            execution=self._recorder,
            state_id=None,
            operation_id=operation_id,
            run_id=self._run_id,
            telemetry_emit=self._telemetry_emit,
            timeout=self._timeout,
            base_url=self._base_url,
            headers=self._request_headers,
            limiter=self._limiter,
        )
        try:
            request_body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": "This is a pre-flight smoke test. Please reply with ok."}],
                "temperature": 0.0,
                # Underlying providers behind OpenRouter enforce different minimums
                # on max_output_tokens. Azure-backed routes require >= 16; values
                # below that 400 with "integer_below_min_value". 32 gives margin
                # without materially affecting smoke-test cost.
                "max_tokens": 32,
            }
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            _validate_chat_completion_response(response)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            detail = _summarize_http_error_body(e)
            if status_code == 429:
                raise RateLimitError(f"Rate limited (HTTP {status_code}){detail}") from e
            if status_code >= 500:
                raise ServerError(f"Server error (HTTP {status_code}){detail}") from e
            raise LLMClientError(f"HTTP {status_code}{detail}", retryable=False) from e
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}") from e
        finally:
            http_client.close()

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> AuditedHTTPClient:
        """Get or create AuditedHTTPClient for a state_id (thread-safe).

        Increments reference count so parallel queries sharing a state_id
        keep the client alive until the last query releases it.
        """
        with self._http_clients_lock:
            if state_id not in self._http_clients:
                self._http_clients[state_id] = AuditedHTTPClient(
                    execution=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=self._timeout,
                    base_url=self._base_url,
                    headers=self._request_headers,
                    limiter=self._limiter,
                    token_id=token_id,
                )
                self._http_client_refs[state_id] = 0
            self._http_client_refs[state_id] += 1
            return self._http_clients[state_id]

    def _release_http_client(self, state_id: str) -> None:
        """Decrement reference count and close client when last user releases it."""
        client_to_close: AuditedHTTPClient | None = None
        with self._http_clients_lock:
            if state_id not in self._http_client_refs:
                raise RuntimeError(
                    f"_release_http_client called for unknown state_id={state_id!r}. "
                    f"This is a refcount underflow — _get_http_client() was never called "
                    f"for this state_id, or it was already fully released."
                )
            count = self._http_client_refs[state_id] - 1
            self._http_client_refs[state_id] = count
            if count <= 0:
                client_to_close = self._http_clients.pop(state_id, None)
                self._http_client_refs.pop(state_id, None)
        if client_to_close is not None:
            client_to_close.close()

    def close(self) -> None:
        """Release all cached clients."""
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()
            self._http_client_refs.clear()
