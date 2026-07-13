"""AWS Bedrock LLM provider implemented through LiteLLM."""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.plugins.infrastructure.clients.llm import (
    AuditedLLMClient,
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.transforms.llm.base import LLMConfig
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult, parse_finish_reason

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.clients.base import TelemetryEmitCallback

__all__ = ["BedrockConfig", "BedrockLLMProvider"]

_STATIC_BEDROCK_ERROR = "Bedrock LLM request failed"


class BedrockConfig(LLMConfig):
    """Keyless LiteLLM Bedrock configuration using the AWS default chain."""

    provider: Literal["bedrock"] = Field(default="bedrock", description="LLM provider")
    model: str = Field(
        ...,
        min_length=9,
        max_length=512,
        description="LiteLLM Bedrock model id in bedrock/<id> form",
    )
    region_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="AWS region override; default AWS region resolution otherwise",
    )
    tracing: dict[str, Any] | None = Field(default=None, description="Tier 2 tracing (langfuse only)")

    @field_validator("model")
    @classmethod
    def _require_bedrock_prefix(cls, value: str) -> str:
        if value != value.strip() or not value.startswith("bedrock/") or not value.removeprefix("bedrock/"):
            raise ValueError("Bedrock model must be a non-empty LiteLLM 'bedrock/<model-id>' value without surrounding whitespace")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            raise ValueError("Bedrock model must not contain control characters")
        return value


class _LiteLLMSDKAdapter:
    """Expose ``litellm.completion`` through the SDK-shaped audited client API."""

    def __init__(self, *, region_name: str | None) -> None:
        self._region_name = region_name
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        import litellm

        if self._region_name is not None:
            kwargs.setdefault("aws_region_name", self._region_name)
        return litellm.completion(**kwargs)

    def close(self) -> None:
        """LiteLLM completion calls hold no provider client owned here."""


def _redacted_bedrock_error(error: LLMClientError) -> LLMClientError:
    """Preserve ELSPETH's typed category without provider-controlled text."""
    if isinstance(error, RateLimitError):
        return RateLimitError(_STATIC_BEDROCK_ERROR)
    if isinstance(error, ContentPolicyError):
        return ContentPolicyError(_STATIC_BEDROCK_ERROR)
    if isinstance(error, ContextLengthError):
        return ContextLengthError(_STATIC_BEDROCK_ERROR)
    if isinstance(error, ServerError):
        return ServerError(_STATIC_BEDROCK_ERROR)
    if isinstance(error, NetworkError):
        return NetworkError(_STATIC_BEDROCK_ERROR)
    return LLMClientError(_STATIC_BEDROCK_ERROR, retryable=error.retryable)


class BedrockLLMProvider:
    """LiteLLM Bedrock provider with audited calls and bounded error egress."""

    def __init__(
        self,
        *,
        region_name: str | None,
        recorder: PluginAuditWriter,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        limiter: Any = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> None:
        self._region_name = region_name
        self._recorder = recorder
        self._run_id = run_id
        self._telemetry_emit = telemetry_emit
        self._limiter = limiter
        self._resolved_prompt_template_hash = resolved_prompt_template_hash
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        self._underlying_client: _LiteLLMSDKAdapter | None = None
        self._underlying_client_lock = Lock()

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
        """Execute one Bedrock request through the authoritative audit client."""
        snapshot_state_id = state_id
        redacted_error: LLMClientError | None = None
        response = None
        try:
            client = self._get_llm_client(snapshot_state_id, token_id=token_id)
            try:
                response = client.chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    resolved_prompt_template_hash=self._resolved_prompt_template_hash,
                )
            except LLMClientError as error:
                redacted_error = _redacted_bedrock_error(error)
        finally:
            with self._llm_clients_lock:
                self._llm_clients.pop(snapshot_state_id, None)

        if redacted_error is not None:
            raise redacted_error from None
        if response is None:
            raise RuntimeError("Bedrock response absent without a typed client error")

        finish_reason = None
        if response.raw_response is not None:
            choices = response.raw_response.get("choices")
            if choices:
                raw_finish_reason = choices[0].get("finish_reason")
                if raw_finish_reason is not None:
                    finish_reason = parse_finish_reason(str(raw_finish_reason))

        if not response.content or not response.content.strip():
            if finish_reason == FinishReason.TOOL_CALLS:
                raise LLMClientError("Bedrock returned tool_calls response (not supported by ELSPETH)", retryable=False)
            raise ContentPolicyError(f"Bedrock LLM returned empty content (finish_reason={finish_reason})")

        return LLMQueryResult(
            content=response.content,
            usage=response.usage,
            model=response.model,
            finish_reason=finish_reason,
        )

    def runtime_preflight(self, *, operation_id: str, model: str) -> None:
        """Run a minimal audited Bedrock call under an operation parent."""
        client = AuditedLLMClient(
            execution=self._recorder,
            state_id=None,
            operation_id=operation_id,
            run_id=self._run_id,
            telemetry_emit=self._telemetry_emit,
            underlying_client=self._get_underlying_client(),
            provider="bedrock",
            limiter=self._limiter,
        )
        redacted_error: LLMClientError | None = None
        try:
            try:
                client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "This is a pre-flight smoke test. Please reply with ok."}],
                    temperature=0.0,
                    max_tokens=32,
                )
            except LLMClientError as error:
                redacted_error = _redacted_bedrock_error(error)
        finally:
            client.close()
        if redacted_error is not None:
            raise redacted_error from None

    def _get_underlying_client(self) -> _LiteLLMSDKAdapter:
        with self._underlying_client_lock:
            if self._underlying_client is None:
                self._underlying_client = _LiteLLMSDKAdapter(region_name=self._region_name)
            return self._underlying_client

    def _get_llm_client(self, state_id: str, *, token_id: str | None = None) -> AuditedLLMClient:
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                self._llm_clients[state_id] = AuditedLLMClient(
                    execution=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    underlying_client=self._get_underlying_client(),
                    provider="bedrock",
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._llm_clients[state_id]

    def close(self) -> None:
        """Release cached audited clients and the stateless adapter."""
        with self._llm_clients_lock:
            self._llm_clients.clear()
        with self._underlying_client_lock:
            if self._underlying_client is not None:
                self._underlying_client.close()
            self._underlying_client = None
