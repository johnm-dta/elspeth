# src/elspeth/plugins/llm/base.py
"""Base class for LLM transforms.

Provides the foundation for all LLM-based transforms with:
- Typed configuration (LLMConfig extending TransformDataConfig)
- Jinja2 prompt templating with audit metadata
- Self-contained client creation (subclasses implement _get_llm_client)
- Three-tier trust model error handling
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import LLMClientError, RateLimitError
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.plugins.clients.llm import AuditedLLMClient


class LLMConfig(TransformDataConfig):
    """Configuration for LLM transforms.

    Extends TransformDataConfig to get:
    - schema: Input/output schema configuration (REQUIRED)
    - on_error: Sink for failed rows (optional)

    LLM-specific fields:
    - model: Model identifier (required)
    - template: Jinja2 prompt template (required)
    - system_prompt: Optional system message
    - temperature: Sampling temperature (default 0.0 for determinism)
    - max_tokens: Maximum response tokens
    - response_field: Field name for LLM response in output
    """

    model: str = Field(
        ..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')"
    )
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")
    response_field: str = Field(
        "llm_response", description="Field name for LLM response in output"
    )

    # New fields for file-based templates
    lookup: dict[str, Any] | None = Field(
        None, description="Lookup data loaded from YAML file"
    )
    template_source: str | None = Field(
        None, description="Template file path for audit (None if inline)"
    )
    lookup_source: str | None = Field(
        None, description="Lookup file path for audit (None if no lookup)"
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template is non-empty and syntactically valid."""
        if not v or not v.strip():
            raise ValueError("template cannot be empty")
        # Validate template syntax at config time
        try:
            PromptTemplate(v)
        except TemplateError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e
        return v


class BaseLLMTransform(BaseTransform):
    """Abstract base class for LLM transforms.

    Provides shared functionality for all LLM transforms:
    - Configuration parsing (LLMConfig)
    - Template rendering with audit metadata
    - Error handling following Three-Tier Trust Model
    - Output row building with usage tracking

    Subclasses MUST implement:
    - `name`: Unique plugin identifier (class attribute)
    - `_get_llm_client(ctx)`: Create/return an AuditedLLMClient

    The `_get_llm_client()` method allows subclasses to be self-contained,
    creating their own audited clients using `ctx.landscape` and `ctx.state_id`.

    Example:
        class MyLLMTransform(BaseLLMTransform):
            name = "my_llm_transform"

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self._api_key = config["api_key"]

            def _get_llm_client(self, ctx: PluginContext) -> AuditedLLMClient:
                from openai import OpenAI
                underlying = OpenAI(api_key=self._api_key)
                return AuditedLLMClient(
                    recorder=ctx.landscape,
                    state_id=ctx.state_id,
                    underlying_client=underlying,
                    provider="openai",
                )
    """

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = LLMConfig.from_dict(config)
        self._model = cfg.model
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._on_error = cfg.on_error

        # Schema from config
        # TransformDataConfig validates schema_config is not None
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

    # name must be set by subclasses - this is enforced by BaseTransform
    # Example: name = "my_llm_transform"

    @abstractmethod
    def _get_llm_client(self, ctx: PluginContext) -> AuditedLLMClient:
        """Create or return an AuditedLLMClient for this transform.

        Subclasses MUST implement this to be self-contained. The client
        should be created using ctx.landscape and ctx.state_id for
        automatic audit trail recording.

        Args:
            ctx: Plugin context with landscape and state_id

        Returns:
            An AuditedLLMClient configured for this transform's provider

        Raises:
            RuntimeError: If ctx.landscape or ctx.state_id is None
        """
        ...

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row through the LLM.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - catch TemplateError, return error
        2. LLM call (EXTERNAL) - catch LLMClientError, return error
        3. Internal logic (OUR CODE) - let it crash

        Args:
            row: Input row matching input_schema
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with processed row or error
        """
        # 1. Render template with row data
        # This operates on THEIR DATA - wrap in try/catch
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                }
            )

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Get LLM client from subclass (self-contained pattern)
        llm_client = self._get_llm_client(ctx)

        # 4. Call LLM via audited client
        # This is an EXTERNAL SYSTEM - wrap in try/catch
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            # Rate limit - retryable
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            # Other LLM error - check if retryable
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # 5. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage

        # 6. Add audit metadata for template traceability
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source

        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources."""
        pass
