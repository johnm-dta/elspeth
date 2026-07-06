"""LLM configuration model.

Provides LLMConfig extending TransformDataConfig with LLM-specific fields:
model, prompt_template, system_prompt, temperature, max_tokens, response_field,
and pool configuration (flat fields assembled into PoolConfig).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from jinja2 import TemplateSyntaxError
from pydantic import Field, field_validator, model_validator

from elspeth.contracts.hashing import stable_hash
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
    - prompt_template: Jinja2 prompt template (required)
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
    prompt_template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")
    response_field: str = Field("llm_response", description="Field name for LLM response in output")

    # File-based content with source paths for audit trail
    lookup: dict[str, Any] | None = Field(None, description="Lookup data loaded from YAML file")
    prompt_template_source: str | None = Field(None, description="Prompt template file path for audit (None if inline)")
    lookup_source: str | None = Field(None, description="Lookup file path for audit (None if no lookup)")
    system_prompt_source: str | None = Field(None, description="System prompt file path for audit (None if inline)")

    # Phase 5b Task 9 — cross-DB hash anchor for interpretation events.
    # When this LLM transform is downstream of a resolved interpretation
    # event, the session service writes ``stable_hash(resolved prompt
    # template)`` here (via ``resolve_interpretation_event`` →
    # ``_patch_llm_transform_prompt`` → ``composition_states.nodes[i].options``).
    # The runtime reads this field and forwards it to every LLM call so the
    # Landscape ``calls.resolved_prompt_template_hash`` column is populated
    # — the cross-DB anchor an auditor uses to join a Landscape call back to
    # the session-DB interpretation_events row. ``None`` is the legitimate
    # value for non-interpretation LLM transforms (most LLM nodes never go
    # through an interpretation surface).
    resolved_prompt_template_hash: str | None = Field(
        None,
        description="Cross-DB hash anchor for interpretation events (Phase 5b Task 9)",
        json_schema_extra={"composer_hidden": True},
    )

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

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, v: str) -> str:
        """Validate prompt_template is non-empty and syntactically valid."""
        if not v or not v.strip():
            raise ValueError("prompt_template cannot be empty")
        # Validate template syntax at config time
        try:
            PromptTemplate(v)
        except TemplateError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e
        return v

    @model_validator(mode="after")
    def _validate_resolved_prompt_template_hash_matches_template(self) -> LLMConfig:
        """Refuse runtime configs whose interpretation hash anchor drifted."""
        if self.resolved_prompt_template_hash is None:
            return self

        expected_hash = stable_hash(self.prompt_template)
        if self.resolved_prompt_template_hash != expected_hash:
            raise ValueError(
                "resolved_prompt_template_hash must equal stable_hash(prompt_template); "
                f"expected {expected_hash!r}, got {self.resolved_prompt_template_hash!r}"
            )
        return self

    def _field_extraction_templates(self) -> tuple[tuple[str, str], ...]:
        """Return (label, template) for every LLM Jinja2 template that can interpolate row data."""
        templates = [("prompt_template", self.prompt_template)]
        if isinstance(self.queries, dict):
            for name, defn in self.queries.items():
                if isinstance(defn, dict) and defn.get("template"):
                    templates.append((f"query {name!r} template", defn["template"]))
        elif isinstance(self.queries, list):
            for index, item in enumerate(self.queries):
                if isinstance(item, dict) and item.get("template"):
                    label = item.get("name", index)
                    templates.append((f"query {label!r} template", item["template"]))
        return tuple(templates)

    @model_validator(mode="after")
    def _validate_dynamic_row_access_requires_explicit_opt_out(self) -> LLMConfig:
        """Fail closed when row fields are accessed through parse-time dynamic keys."""
        if self.required_input_fields == []:
            return self

        from elspeth.core.templates import extract_jinja2_field_usage

        dynamic_accesses: list[str] = []
        for label, template in self._field_extraction_templates():
            try:
                extraction = extract_jinja2_field_usage(template)
            except TemplateSyntaxError as e:
                # An unparseable template cannot be proven free of dynamic row
                # access, so it must fail here — as the structured TemplateError
                # the constructor advertises, not a raw jinja2 exception.
                raise TemplateError(f"Invalid template syntax in {label}: {e}") from e
            dynamic_accesses.extend(extraction.dynamic_accesses)

        if not dynamic_accesses:
            return self

        access_kinds = sorted(set(dynamic_accesses))
        raise ValueError(
            "LLM prompt_template uses dynamic row field access "
            f"({', '.join(access_kinds)} via row[expr] or row.get(expr)). "
            "Dynamic row keys cannot be audited against options.required_input_fields. "
            "Use static row.field or row['field'] references, or set "
            "options.required_input_fields: [] to explicitly opt out and accept runtime risk."
        )

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
                extracted.update(extract_jinja2_fields(self.prompt_template))
            else:
                # Single-query mode: detect row references in the template
                extracted = set(extract_jinja2_fields(self.prompt_template))

            if extracted:
                required_fields = sorted(extracted)
                required_fields_json = json.dumps(required_fields)
                raise ValueError(
                    f"LLM prompt_template references row fields {required_fields} but "
                    f"options.required_input_fields is not declared.\n\n"
                    "You must explicitly declare field requirements inside the LLM node options:\n"
                    f"  options.required_input_fields: {required_fields_json}  # Require these fields\n"
                    "  options.required_input_fields: []                    # Accept runtime risk (opt-out)\n\n"
                    "Composer repair examples:\n"
                    f'  patch_node_options({{"node_id": "<node_id>", "patch": {{"required_input_fields": {required_fields_json}}}}})\n'
                    f"  set_pipeline/upsert_node: include options.required_input_fields={required_fields_json} on the llm node.\n\n"
                    "Use extract_jinja2_fields() from elspeth.core.templates to discover fields. "
                    "This explicit declaration enables DAG validation to catch missing fields at config time."
                )
        return self

    @model_validator(mode="after")
    def _validate_required_input_fields_appear_in_template(self) -> LLMConfig:
        """Reject single-query configs that declare row-field requirements the template never uses.

        Dual of `_validate_required_input_fields_declared`. That check catches the
        "template uses row.X but contract is undeclared" footgun. This check catches
        the inverse "contract declares X but template never references row.X" footgun
        — a prompt body that does not interpolate any row data, so every row receives
        the same static prompt and the model has no per-row context to reason about.

        Scope:
        - Single-query mode only (``queries is None``). Multi-query mode flows row
          data via per-query ``input_fields`` mappings, so an empty ``row.*`` set in
          the top-level template is not by itself diagnostic.
        - Empty ``required_input_fields: []`` is the explicit opt-out and passes.
        - ``required_input_fields is None`` is handled by the sibling validator;
          this check only fires when fields are declared.
        """
        if self.queries is not None:
            return self
        if self.required_input_fields is None or len(self.required_input_fields) == 0:
            return self

        from elspeth.core.templates import extract_jinja2_fields

        template_fields = extract_jinja2_fields(self.prompt_template)
        if template_fields:
            return self

        declared = sorted(self.required_input_fields)
        declared_json = json.dumps(declared)
        example_interpolations = " ".join(f"{{{{ row.{f} }}}}" for f in declared)
        raise ValueError(
            f"LLM options.required_input_fields declares {declared} but the "
            "prompt_template does not interpolate any row.* fields. "
            "Every row would receive the same static prompt and the model would "
            "have no row-specific context to reason about.\n\n"
            "Fix one of the following:\n"
            f"  (a) Reference the declared fields inside prompt_template using "
            f"Jinja2 row-namespace syntax, e.g. {example_interpolations}\n"
            "  (b) If the fields are required for runtime presence but intentionally "
            "not interpolated into the prompt, set\n"
            "      options.required_input_fields: []   # explicit opt-out\n"
            "      and document the presence assertion elsewhere.\n\n"
            "Composer repair example:\n"
            f'  patch_node_options({{"node_id": "<node_id>", "patch": '
            f'{{"prompt_template": "<...includes {example_interpolations}...>"}}}})\n\n'
            f"Declared fields: {declared_json}. Template row.* references: []."
        )
