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

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import LLMClientError
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.pooling import PoolConfig
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.plugins.clients.llm import AuditedLLMClient


class LLMConfig(TransformDataConfig):
    """Configuration for LLM transforms.

    Extends TransformDataConfig to get:
    - schema: Input/output schema configuration (REQUIRED)
    - on_error: Sink for failed rows (optional)
    - required_input_fields: Fields this transform requires (optional but recommended)

    IMPORTANT: Template Field Requirements
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If your template references row fields (e.g., {{ row.customer_id }}),
    you SHOULD declare them in `required_input_fields`. This enables DAG
    validation to catch missing fields at config time rather than runtime.

    Use the helper utility to discover fields:

        from elspeth.core.templates import extract_jinja2_fields

        fields = extract_jinja2_fields(your_template)
        # Returns: frozenset({'customer_id', 'amount'})
        # Then add to config: required_input_fields: [customer_id, amount]

    For templates with conditional logic ({% if row.x %}...{% endif %}),
    only declare the fields that are TRULY required (always accessed).

    LLM-specific fields:
    - model: Model identifier (required)
    - template: Jinja2 prompt template (required)
    - system_prompt: Optional system message
    - temperature: Sampling temperature (default 0.0 for determinism)
    - max_tokens: Maximum response tokens
    - response_field: Field name for LLM response in output

    Pool configuration (flat fields assembled into PoolConfig when pool_size > 1):
    - pool_size: Number of concurrent requests (1 = sequential, no pooling)
    - min_dispatch_delay_ms: Floor for delay between dispatches
    - max_dispatch_delay_ms: Ceiling for delay
    - backoff_multiplier: Multiply delay on capacity error (must be > 1)
    - recovery_step_ms: Subtract from delay on success
    - max_capacity_retry_seconds: Max time to retry capacity errors per row
    """

    model: str = Field(..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")
    response_field: str = Field("llm_response", description="Field name for LLM response in output")

    # File-based content with source paths for audit trail
    lookup: dict[str, Any] | None = Field(None, description="Lookup data loaded from YAML file")
    template_source: str | None = Field(None, description="Template file path for audit (None if inline)")
    lookup_source: str | None = Field(None, description="Lookup file path for audit (None if no lookup)")
    system_prompt_source: str | None = Field(None, description="System prompt file path for audit (None if inline)")

    # Pool configuration fields (flat - assembled into PoolConfig by pool_config property)
    pool_size: int = Field(1, ge=1, description="Number of concurrent requests (1 = sequential)")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    @property
    def pool_config(self) -> PoolConfig | None:
        """Get pool configuration if pooling is enabled.

        Returns None if pool_size <= 1 (sequential mode).
        Otherwise returns a PoolConfig built from flat fields.

        Returns:
            PoolConfig instance or None if sequential mode.
        """
        if self.pool_size <= 1:
            return None
        return PoolConfig(
            pool_size=self.pool_size,
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
            max_capacity_retry_seconds=self.max_capacity_retry_seconds,
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

    @model_validator(mode="after")
    def _validate_required_input_fields_declared(self) -> LLMConfig:
        """Require explicit field declaration when template references row fields.

        This enforces the "explicit contracts" pattern from ELSPETH's audit philosophy.
        If a template accesses row.field, the user MUST declare what fields are required.

        Opt-out mechanism:
        - required_input_fields: [field_a, field_b]  # Declare specific requirements
        - required_input_fields: []                   # Explicit opt-out (accept runtime risk)

        Omitting required_input_fields entirely when template has row references is an error.
        This prevents "Drifting Goals" pattern where teams deploy without thinking about contracts.
        """
        # None means "not specified" - this triggers the check
        # Empty list [] means "explicit opt-out" - this is allowed
        fields_not_declared = self.required_input_fields is None

        if fields_not_declared:
            # Use AST parser to detect row references - catches ALL Jinja2 patterns
            # including {% if row.x %}, {{ filter(row.y) }}, row['field'], etc.
            from elspeth.core.templates import extract_jinja2_fields

            extracted = extract_jinja2_fields(self.template)
            if extracted:
                raise ValueError(
                    f"LLM template references row fields {sorted(extracted)} but "
                    f"required_input_fields is not declared.\n\n"
                    f"You must explicitly declare field requirements:\n"
                    f"  required_input_fields: {sorted(extracted)}  # Require these fields\n"
                    f"  required_input_fields: []                    # Accept runtime risk (opt-out)\n\n"
                    f"Use extract_jinja2_fields() from elspeth.core.templates to discover fields.\n"
                    f"This explicit declaration enables DAG validation to catch missing fields at config time."
                )
        return self


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
                self._limiter = None  # Set in on_start

            def on_start(self, ctx: PluginContext) -> None:
                # Capture rate limiter for throttling
                self._limiter = (
                    ctx.rate_limit_registry.get_limiter("openai")
                    if ctx.rate_limit_registry is not None
                    else None
                )

            def _get_llm_client(self, ctx: PluginContext) -> AuditedLLMClient:
                from openai import OpenAI
                underlying = OpenAI(api_key=self._api_key)
                return AuditedLLMClient(
                    recorder=ctx.landscape,
                    state_id=ctx.state_id,
                    run_id=ctx.run_id,
                    telemetry_emit=ctx.telemetry_emit,
                    underlying_client=underlying,
                    provider="openai",
                    limiter=self._limiter,
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
        self._system_prompt_source = cfg.system_prompt_source
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._on_error = cfg.on_error

        # Schema from config (TransformDataConfig guarantees schema_config is not None)
        schema_config = cfg.schema_config
        schema = create_schema_from_config(
            schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Build output schema config with field categorization
        guaranteed = get_llm_guaranteed_fields(self._response_field)
        audit = get_llm_audit_fields(self._response_field)

        # Merge with any existing fields from base schema
        base_guaranteed = schema_config.guaranteed_fields or ()
        base_audit = schema_config.audit_fields or ()

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            is_dynamic=schema_config.is_dynamic,
            guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
            audit_fields=tuple(set(base_audit) | set(audit)),
            required_fields=schema_config.required_fields,
        )

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
        # Retryable errors (RateLimitError, NetworkError, ServerError) are re-raised
        # to let the engine's RetryManager handle them. Non-retryable errors
        # (ContentPolicyError, ContextLengthError) return TransformResult.error().
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LLMClientError as e:
            if e.retryable:
                # Re-raise for engine retry (RateLimitError, NetworkError, ServerError)
                raise
            # Non-retryable error - return error result
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=False,
            )

        # 5. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_model"] = response.model
        output[f"{self._response_field}_usage"] = response.usage

        # 6. Add audit metadata for template traceability
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source

        return TransformResult.success(
            output,
            success_reason={"action": "enriched", "fields_added": [self._response_field]},
        )

    def close(self) -> None:
        """Release resources."""
        pass
