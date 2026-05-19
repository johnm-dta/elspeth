"""Azure OpenAI LLM provider.

Thin wrapper over AuditedLLMClient that normalizes LLMResponse into
LLMQueryResult. All audit recording, telemetry, and error classification
happen inside AuditedLLMClient — this provider just manages client lifecycle
and response normalization.

Client caching is per-state_id with a threading lock. The state_id is
snapshot at method entry (not read from a mutable context) to prevent
evicting the wrong cache entry during retry races.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import structlog
from pydantic import Field, field_validator, model_validator

from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.value_source import DerivedFromSiblingValueSource, ValueSource
from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient, ContentPolicyError, LLMClientError
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url
from elspeth.plugins.transforms.llm.base import LLMConfig
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult, parse_finish_reason
from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig, TracingConfig

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.clients.base import TelemetryEmitCallback

logger = structlog.get_logger(__name__)


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI-specific configuration.

    Extends LLMConfig with Azure-specific settings:
    - deployment_name: Azure deployment name (required) - used as model identifier
    - endpoint: Azure OpenAI endpoint URL (required)
    - api_key: Azure OpenAI API key (required)
    - api_version: Azure API version (default: 2024-10-21)

    Pooling options (inherited from LLMConfig):
    - pool_size: Number of concurrent workers (1=sequential, >1=pooled)
    - max_dispatch_delay_ms: Maximum AIMD backoff delay
    - max_capacity_retry_seconds: Timeout for capacity error retries

    Note: The 'model' field from LLMConfig is automatically set to
    deployment_name if not explicitly provided.
    """

    # Azure configs always have provider="azure" — narrowed Literal prevents misconfiguration
    provider: Literal["azure"] = Field(default="azure", description="LLM provider")

    # Override model to make it optional - will default to deployment_name
    model: str = Field(default="", description="Model identifier (defaults to deployment_name)")

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    # Tier 2: Plugin-internal tracing (optional)
    # Use environment variables for secrets: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (azure_ai, langfuse, or none)",
    )

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint_url(cls, value: str) -> str:
        return validate_credential_safe_https_url(value, field_name="endpoint")

    @model_validator(mode="before")
    @classmethod
    def _set_model_from_deployment(cls, data: Any) -> Any:
        """Set model to deployment_name if not explicitly provided."""
        if isinstance(data, dict) and not data.get("model"):
            deployment = data.get("deployment_name")
            if deployment:
                data["model"] = deployment
        return data

    # Value-source declaration: ``model`` is derived from ``deployment_name``.
    # The ``_set_model_from_deployment`` validator above fills the field when
    # empty; the value-source compliance walker confirms post-validation that
    # ``model == deployment_name`` (or that ``model`` was empty in the original
    # config — accepted because the validator substitutes the sibling).
    VALUE_SOURCES: ClassVar[tuple[ValueSource, ...]] = (
        DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        ),
    )


class AzureLLMProvider:
    """Azure OpenAI provider — wraps AuditedLLMClient.

    Responsibilities:
    1. Create/cache AuditedLLMClient per state_id (thread-safe)
    2. Create/cache underlying AzureOpenAI SDK client (thread-safe)
    3. Map LLMResponse → LLMQueryResult (content, usage, model, finish_reason)
    4. Let LLMClientError subclasses propagate unchanged

    Does NOT own:
    - Audit recording (AuditedLLMClient does this)
    - Error classification (AuditedLLMClient does this)
    - Tracing setup (transform lifecycle manages azure_ai tracing)
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        recorder: PluginAuditWriter,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        limiter: Any = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key: str | None = api_key
        self._api_version = api_version
        self._deployment_name = deployment_name
        self._recorder = recorder
        self._run_id = run_id
        self._telemetry_emit = telemetry_emit
        self._limiter = limiter
        # Phase 5b Task 9 — cross-DB hash anchor. Forwarded to every
        # ``client.chat_completion`` call so the Landscape ``calls`` row
        # carries the matching SHA-256.
        self._resolved_prompt_template_hash = resolved_prompt_template_hash

        # Client caches — lock ordering: _llm_clients_lock → _underlying_client_lock
        # (always acquire _llm_clients_lock first to prevent deadlock)
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        self._underlying_client: Any = None  # AzureOpenAI | None
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
        """Execute LLM query via Azure OpenAI SDK.

        Args:
            messages: Chat messages (system + user)
            model: Model/deployment name
            temperature: Sampling temperature
            max_tokens: Max response tokens (None = provider default)
            state_id: Snapshot of state_id for client caching
            token_id: Token identity for audit correlation
            response_format: OpenAI response_format dict (e.g., {"type": "json_object"})

        Returns:
            Normalized LLMQueryResult

        Raises:
            RateLimitError, NetworkError, ServerError: Retryable
            ContentPolicyError, ContextLengthError, LLMClientError: Not retryable
        """
        # Snapshot state_id — do not read from mutable ctx later.
        # This prevents the openrouter.py bug where ctx.state_id was read
        # in the finally block, evicting the wrong cache entry during retries.
        snapshot_state_id = state_id

        try:
            client = self._get_llm_client(snapshot_state_id, token_id=token_id)

            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                resolved_prompt_template_hash=self._resolved_prompt_template_hash,
            )

            # Extract finish_reason from raw_response.
            # raw_response is the Azure SDK's deserialized API response (Tier 3
            # external boundary — validate structure, then safe access on optional keys).
            # Missing/empty choices or absent raw_response → finish_reason stays None.
            # The full raw_response is already recorded in the audit trail via
            # AuditedLLMClient.record_call(), so anomalies are diagnosable there.
            finish_reason = None
            if response.raw_response is not None:
                choices = response.raw_response.get("choices")
                if choices:
                    raw_fr = choices[0].get("finish_reason")
                    if raw_fr is not None:
                        finish_reason = parse_finish_reason(str(raw_fr))

            # Empty/whitespace content — AuditedLLMClient converts None→""
            # (known fabrication). Detect here and raise typed errors so the
            # transform's except LLMClientError handler catches them.
            if not response.content or not response.content.strip():
                if finish_reason == FinishReason.TOOL_CALLS:
                    raise LLMClientError(
                        "Azure returned tool_calls response (not supported by ELSPETH)",
                        retryable=False,
                    )
                raise ContentPolicyError(
                    f"LLM returned empty content (finish_reason={finish_reason})",
                )

            return LLMQueryResult(
                content=response.content,
                usage=response.usage,
                model=response.model,
                finish_reason=finish_reason,
            )
        finally:
            # Clean up cached client for this state_id to prevent unbounded growth.
            # Uses snapshot (not state_id parameter) to avoid evicting wrong entry.
            with self._llm_clients_lock:
                self._llm_clients.pop(snapshot_state_id, None)

    def runtime_preflight(self, *, operation_id: str, model: str) -> None:
        """Run a minimal audited Azure OpenAI call under an operation parent."""
        client = AuditedLLMClient(
            execution=self._recorder,
            state_id=None,
            operation_id=operation_id,
            run_id=self._run_id,
            telemetry_emit=self._telemetry_emit,
            underlying_client=self._get_underlying_client(),
            provider="azure",
            limiter=self._limiter,
        )
        try:
            client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Respond with OK only."}],
                temperature=0.0,
                # Azure OpenAI requires max_output_tokens >= 16. Values below
                # the floor return HTTP 400 with "integer_below_min_value"
                # before any model work, killing the entire pipeline at
                # preflight. 32 gives margin without materially affecting
                # smoke-test cost.
                max_tokens=32,
            )
        finally:
            client.close()

    def _get_underlying_client(self) -> Any:
        """Get or create the underlying AzureOpenAI SDK client (thread-safe)."""
        with self._underlying_client_lock:
            if self._underlying_client is None:
                from openai import AzureOpenAI

                self._underlying_client = AzureOpenAI(
                    azure_endpoint=self._endpoint,
                    api_key=self._api_key,
                    api_version=self._api_version,
                )
                # Clear plaintext key — SDK client holds its own copy
                self._api_key = None
            return self._underlying_client

    def _get_llm_client(self, state_id: str, *, token_id: str | None = None) -> AuditedLLMClient:
        """Get or create AuditedLLMClient for a state_id (thread-safe)."""
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                self._llm_clients[state_id] = AuditedLLMClient(
                    execution=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    underlying_client=self._get_underlying_client(),
                    provider="azure",
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._llm_clients[state_id]

    def close(self) -> None:
        """Release all cached clients."""
        with self._llm_clients_lock:
            self._llm_clients.clear()
        with self._underlying_client_lock:
            if self._underlying_client is not None:
                self._underlying_client.close()
            self._underlying_client = None


# Optional SDK import — module-level so tests can mock it.
_imported_configure_azure_monitor: Callable[..., None] | None
try:
    from azure.monitor.opentelemetry import (
        configure_azure_monitor as _azure_monitor_sdk_configure,
    )
except ImportError:
    _imported_configure_azure_monitor = None
else:
    _imported_configure_azure_monitor = _azure_monitor_sdk_configure

configure_azure_monitor: Callable[..., None] | None = _imported_configure_azure_monitor

# Module-level idempotency guard — Azure Monitor is process-global.
_azure_monitor_configured: bool = False


def _reset_azure_monitor_state() -> None:
    """Reset module state for testing only."""
    global _azure_monitor_configured
    _azure_monitor_configured = False


def _configure_azure_monitor(config: TracingConfig) -> bool:
    """Configure Azure Monitor (module-level to allow mocking).

    Returns True on success, False on failure.
    Idempotent: second call logs a warning and returns True.
    """
    global _azure_monitor_configured

    if not isinstance(config, AzureAITracingConfig):
        return False

    if _azure_monitor_configured:
        logger.warning(
            "Azure Monitor already configured — skipping duplicate initialization",
        )
        return True

    if configure_azure_monitor is None:
        raise ImportError(
            "azure-monitor-opentelemetry is not installed but azure_ai tracing was configured. "
            "This should not happen — the SDK is bundled with elspeth[azure]. "
            "Reinstall with: uv pip install 'elspeth[azure]'"
        )

    configure_azure_monitor(
        connection_string=config.connection_string,
        enable_live_metrics=config.enable_live_metrics,
    )

    # Wire enable_content_recording to the Azure AI Inference tracing SDK.
    # Without this, the config field is accepted and logged but never applied,
    # leaving operators with a false sense of their content recording policy.
    try:
        from azure.ai.inference.tracing import AIInferenceInstrumentor

        AIInferenceInstrumentor().instrument(enable_content_recording=config.enable_content_recording)
    except ImportError:
        # azure-ai-inference not installed — fall back to environment variable
        # which the OpenAI SDK instrumentor reads at trace emission time.
        import os

        logger.warning(
            "azure-ai-inference not installed — falling back to environment variable for content recording",
            hint="Install azure-ai-inference for full tracing support",
            fallback_env_var="AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED",
        )
        os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = str(config.enable_content_recording).lower()

    _azure_monitor_configured = True
    return True
