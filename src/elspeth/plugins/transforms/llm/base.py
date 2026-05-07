"""LLM configuration model.

Provides LLMConfig extending TransformDataConfig with LLM-specific fields:
model, template, system_prompt, temperature, max_tokens, response_field,
and pool configuration (flat fields assembled into PoolConfig).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.pooling import PoolConfig
from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.plugins.transforms.llm.templates import PromptTemplate


class LLMConfig(TransformDataConfig):
    """Configuration for LLM transforms.

    Extends TransformDataConfig to get:
    - schema: Input/output schema configuration (REQUIRED)
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
    - provider: LLM provider ("azure" or "openrouter")
    - model: Model identifier (optional — Azure uses deployment_name instead)
    - template: Jinja2 prompt template (required)
    - system_prompt: Optional system message
    - temperature: Sampling temperature (default 0.0 for determinism)
    - max_tokens: Maximum response tokens
    - response_field: Field name for LLM response in output
    - queries: Multi-query specs (None = single-query mode)

    Pool configuration (flat fields assembled into PoolConfig when pool_size > 1):
    - pool_size: Number of concurrent requests (1 = sequential, no pooling)
    - min_dispatch_delay_ms: Floor for delay between dispatches
    - max_dispatch_delay_ms: Ceiling for delay
    - backoff_multiplier: Multiply delay on capacity error (must be > 1)
    - recovery_step_ms: Subtract from delay on success
    - max_capacity_retry_seconds: Max time to retry capacity errors per row
    """

    provider: Literal["azure", "openrouter"] = Field(..., description="LLM provider")
    model: str | None = Field(None, description="Model identifier (optional — Azure uses deployment_name)")
    queries: list[dict[str, Any]] | dict[str, dict[str, Any]] | None = Field(
        None, description="Multi-query specs (None = single-query mode)"
    )
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

    @field_validator("response_field")
    @classmethod
    def validate_response_field(cls, v: str) -> str:
        """Validate response_field is a valid Python identifier."""
        if not v or not v.strip():
            raise ValueError("response_field must be non-empty")
        if not v.isidentifier():
            raise ValueError(f"response_field must be a valid Python identifier, got {v!r}")
        return v

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

        In multi-query mode, required fields are derived from the union of all
        query specs' input_fields values (the row column names), plus any row
        references in the top-level template and per-query template overrides.

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
            from elspeth.core.templates import extract_jinja2_fields

            if self.queries is not None:
                # Multi-query mode: required row fields are the union of all
                # query specs' input_fields values (row column names), plus
                # any row.* references in the top-level and per-query templates.
                extracted: set[str] = set()
                # Collect row column names from input_fields mappings
                if isinstance(self.queries, dict):
                    for defn in self.queries.values():
                        if isinstance(defn, dict) and "input_fields" in defn:
                            extracted.update(defn["input_fields"].values())
                            if "template" in defn and defn["template"]:
                                extracted.update(extract_jinja2_fields(defn["template"]))
                elif isinstance(self.queries, list):
                    for item in self.queries:
                        if isinstance(item, dict) and "input_fields" in item:
                            extracted.update(item["input_fields"].values())
                            if "template" in item and item["template"]:
                                extracted.update(extract_jinja2_fields(item["template"]))
                # Also check the top-level template for row references
                extracted.update(extract_jinja2_fields(self.template))
            else:
                # Single-query mode: detect row references in the template
                extracted = set(extract_jinja2_fields(self.template))

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
