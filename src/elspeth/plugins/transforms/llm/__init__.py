"""LLM transform plugins for ELSPETH.

Provides transforms for Azure OpenAI, OpenRouter, and other LLM providers.
Includes support for pooled execution, batch processing, and multi-query evaluation.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    transform = manager.get_transform_by_name("llm")

For testing or advanced usage, import directly from module paths:
    from elspeth.plugins.transforms.llm.transform import LLMTransform
    from elspeth.plugins.transforms.llm.templates import PromptTemplate
    from elspeth.contracts.schema_contract import PipelineRow

Metadata Field Categories
=========================

guaranteed_fields: Contract-stable fields downstream can depend on
    - <response_field>: The LLM response content
    - <response_field>_usage: Token usage (for cost/quota management)
    - <response_field>_model: Model identifier (for routing/reporting)

audit_fields: Provenance metadata for audit trail (may change between versions)
    - <response_field>_template_hash: SHA256 of prompt template
    - <response_field>_variables_hash: SHA256 of rendered variables
    - <response_field>_template_source: Config file path
    - <response_field>_lookup_hash: SHA256 of lookup data
    - <response_field>_lookup_source: Config file path
    - <response_field>_system_prompt_source: Config file path

WARNING: Do not build production logic that depends on audit_fields.
These fields exist for audit trail reconstruction (explain() queries)
and may change between versions without notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from elspeth.contracts.token_usage import TokenUsage

if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema
    from elspeth.contracts.schema import SchemaConfig

# Metadata field suffixes for contract-stable fields (downstream can depend on these)
LLM_GUARANTEED_SUFFIXES: tuple[str, ...] = (
    "",  # The response content field itself
    "_usage",  # Token usage dict {prompt_tokens, completion_tokens, total_tokens}
    "_model",  # Model identifier that actually responded
)

# Type alias matching FieldDefinition.field_type
_FieldType = Literal["str", "int", "float", "bool", "any"]

# Schema types for each guaranteed suffix — used by output schema builders
# to advertise real types instead of erasing to "any".
_SUFFIX_SCHEMA_TYPES: dict[str, _FieldType] = {
    "": "str",  # Response content is always str
    "_usage": "any",  # Token usage dict — shape varies by provider
    "_model": "str",  # Model identifier is always str
}

# Maps OutputFieldType enum values to schema field_type strings
_OUTPUT_FIELD_TYPE_TO_SCHEMA: dict[str, _FieldType] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "enum": "str",  # Enum values are string labels in output rows
}

# Multi-query transforms emit suffixed fields only (no base field)
# e.g., category_score, category_rationale, category_usage, category_model
# NOT category (the base field with empty suffix)
MULTI_QUERY_GUARANTEED_SUFFIXES: tuple[str, ...] = (
    "_usage",  # Token usage dict {prompt_tokens, completion_tokens, total_tokens}
    "_model",  # Model identifier that actually responded
)

# Metadata field suffixes for audit-only fields (exist but may change between versions)
LLM_AUDIT_SUFFIXES: tuple[str, ...] = (
    "_template_hash",  # SHA256 of prompt template
    "_variables_hash",  # SHA256 of rendered template variables
    "_template_source",  # File path of template (None if inline)
    "_lookup_hash",  # SHA256 of lookup data
    "_lookup_source",  # File path of lookup data (None if no lookup)
    "_system_prompt_source",  # File path of system prompt (None if inline)
)


def get_llm_guaranteed_fields(response_field: str) -> tuple[str, ...]:
    """Return contract-stable metadata field names for LLM transforms.

    These fields are part of the stable API. Downstream transforms can
    safely declare dependencies on them via required_fields.

    Args:
        response_field: Base field name (e.g., "llm_response"). Must not be empty
            or whitespace-only.

    Returns:
        Tuple of field names that are guaranteed to exist.

    Raises:
        ValueError: If response_field is empty or whitespace-only.
    """
    if not response_field or not response_field.strip():
        raise ValueError("response_field cannot be empty or whitespace-only")
    if not response_field.isidentifier():
        raise ValueError(
            f"response_field '{response_field}' is not a valid Python identifier. "
            f"Use only letters, digits, and underscores, starting with a letter or underscore."
        )
    return tuple(f"{response_field}{suffix}" for suffix in LLM_GUARANTEED_SUFFIXES)


def get_llm_audit_fields(response_field: str) -> tuple[str, ...]:
    """Return audit-only metadata field names for LLM transforms.

    These fields exist for audit trail reconstruction but are NOT part
    of the stability contract. They may change between versions.

    Args:
        response_field: Base field name (e.g., "llm_response"). Must not be empty
            or whitespace-only.

    Returns:
        Tuple of field names for audit purposes.

    Raises:
        ValueError: If response_field is empty or whitespace-only.
    """
    if not response_field or not response_field.strip():
        raise ValueError("response_field cannot be empty or whitespace-only")
    if not response_field.isidentifier():
        raise ValueError(
            f"response_field '{response_field}' is not a valid Python identifier. "
            f"Use only letters, digits, and underscores, starting with a letter or underscore."
        )
    return tuple(f"{response_field}{suffix}" for suffix in LLM_AUDIT_SUFFIXES)


def populate_llm_operational_fields(
    output: dict[str, object],
    field_prefix: str,
    *,
    usage: TokenUsage | None,
    model: str | None,
) -> None:
    """Populate operational metadata into the output row (stays in pipeline data).

    These fields have legitimate downstream use (budgeting, routing).
    Audit provenance fields go to success_reason["metadata"] via
    build_llm_audit_metadata() instead.

    Args:
        output: Mutable row dict to populate.
        field_prefix: Response field name (e.g., "llm_response").
        usage: Token usage (TokenUsage or None).
        model: Model identifier that actually responded.
    """
    output[f"{field_prefix}_usage"] = usage.to_dict() if usage is not None else None
    output[f"{field_prefix}_model"] = model


def build_llm_audit_metadata(
    field_prefix: str,
    *,
    template_hash: str,
    variables_hash: str | None,
    template_source: str | None,
    lookup_hash: str | None,
    lookup_source: str | None,
    system_prompt_source: str | None,
) -> dict[str, str | None]:
    """Build audit provenance dict for inclusion in success_reason["metadata"].

    Does NOT write to the output row — audit provenance lives in the Landscape only.

    Args:
        field_prefix: Response field name (e.g., "llm_response").
        template_hash: SHA-256 of prompt template.
        variables_hash: SHA-256 of rendered template variables (None for batch-level
            metadata where per-row hashes are recorded in the calls table).
        template_source: Config file path of template (None if inline).
        lookup_hash: SHA-256 of lookup data (None if no lookup).
        lookup_source: Config file path of lookup data (None if no lookup).
        system_prompt_source: Config file path of system prompt (None if inline).

    Returns:
        Dict of audit field names to values, ready to merge into
        success_reason["metadata"].
    """
    return {
        f"{field_prefix}_template_hash": template_hash,
        f"{field_prefix}_variables_hash": variables_hash,
        f"{field_prefix}_template_source": template_source,
        f"{field_prefix}_lookup_hash": lookup_hash,
        f"{field_prefix}_lookup_source": lookup_source,
        f"{field_prefix}_system_prompt_source": system_prompt_source,
    }


def _build_augmented_output_schema(
    base_schema_config: SchemaConfig,
    response_field: str,
    schema_name: str,
) -> type[PluginSchema]:
    """Build an output schema that includes LLM-added fields.

    LLM transforms add response, usage, and model fields to output rows.
    The output schema must include these fields for DAG type validation to pass
    when downstream consumers have explicit schemas requiring LLM output fields.

    For observed schemas this returns the same dynamic schema (no fields to add).
    For explicit schemas (fixed/flexible) this augments the base fields with
    optional LLM output fields typed as ``object`` (Any).

    Args:
        base_schema_config: The base schema config from plugin options.
        response_field: Base field name (e.g., "llm_response").
        schema_name: Name for the generated Pydantic model class.

    Returns:
        A PluginSchema subclass with input fields plus LLM output fields.
    """
    from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

    if base_schema_config.is_observed:
        # Observed schemas accept anything — no augmentation needed
        return create_schema_from_config(base_schema_config, schema_name, allow_coercion=False)

    # For explicit schemas, build an augmented SchemaConfig that includes
    # LLM output fields as optional fields.
    from elspeth.contracts.schema import FieldDefinition, SchemaConfig

    base_fields = base_schema_config.fields or ()
    existing_names = {f.name for f in base_fields}

    # Add LLM fields (guaranteed only) with their real types
    # Audit provenance fields go to success_reason["metadata"], not the output schema
    extra_fields = tuple(
        FieldDefinition(
            name=f"{response_field}{suffix}",
            field_type=_SUFFIX_SCHEMA_TYPES[suffix],
            required=False,
        )
        for suffix in LLM_GUARANTEED_SUFFIXES
        if f"{response_field}{suffix}" not in existing_names
    )

    augmented_config = SchemaConfig(
        # Use flexible mode so extra fields from upstream are accepted
        mode="flexible",
        fields=(*base_fields, *extra_fields),
        guaranteed_fields=base_schema_config.guaranteed_fields,
        required_fields=base_schema_config.required_fields,
        audit_fields=base_schema_config.audit_fields,
    )
    return create_schema_from_config(augmented_config, schema_name, allow_coercion=False)


def _build_multi_query_output_schema(
    base_schema_config: SchemaConfig,
    response_field: str,
    query_names: tuple[str, ...],
    schema_name: str,
    extracted_fields: dict[str, tuple[tuple[str, _FieldType], ...]] | None = None,
) -> type[PluginSchema]:
    """Build an output schema for multi-query mode with prefixed fields.

    Multi-query transforms emit query-prefixed fields (e.g., quality_llm_response,
    sentiment_llm_response) instead of the unprefixed single-query fields. The output
    schema must include these fields for DAG type validation.

    For observed schemas this returns the same dynamic schema (no fields to add).
    For explicit schemas this augments the base fields with optional prefixed
    LLM output fields with their real types.

    Args:
        base_schema_config: The base schema config from plugin options.
        response_field: Base field name (e.g., "llm_response").
        query_names: Tuple of query names (e.g., ("quality", "sentiment")).
        schema_name: Name for the generated Pydantic model class.
        extracted_fields: Mapping of query name to (field_name, schema_type) tuples
            from structured output_fields config. These are fields extracted from
            LLM JSON responses (e.g., {"quality": (("quality_score", "int"), ("quality_label", "str"))}).

    Returns:
        A PluginSchema subclass with input fields plus prefixed LLM output fields.
    """
    from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

    if base_schema_config.is_observed:
        return create_schema_from_config(base_schema_config, schema_name, allow_coercion=False)

    from elspeth.contracts.schema import FieldDefinition
    from elspeth.contracts.schema import SchemaConfig as _SchemaConfig

    if extracted_fields is not None:
        unknown_queries = set(extracted_fields) - set(query_names)
        if unknown_queries:
            raise ValueError(f"extracted_fields references unknown query names: {unknown_queries}")

    base_fields = base_schema_config.fields or ()
    existing_names = {f.name for f in base_fields}

    # Add prefixed LLM fields for each query with real types
    extra_fields: list[FieldDefinition] = []
    for query_name in query_names:
        prefix = f"{query_name}_{response_field}"
        # The response field itself is str
        if prefix not in existing_names:
            extra_fields.append(FieldDefinition(name=prefix, field_type="str", required=False))
            existing_names.add(prefix)
        # Guaranteed metadata fields with real types
        for suffix in MULTI_QUERY_GUARANTEED_SUFFIXES:
            name = f"{prefix}{suffix}"
            if name not in existing_names:
                extra_fields.append(FieldDefinition(name=name, field_type=_SUFFIX_SCHEMA_TYPES[suffix], required=False))
                existing_names.add(name)

        # Add structured output_fields with their declared types
        if extracted_fields is not None and query_name in extracted_fields:
            for field_name, field_type in extracted_fields[query_name]:
                if field_name not in existing_names:
                    extra_fields.append(FieldDefinition(name=field_name, field_type=field_type, required=False))
                    existing_names.add(field_name)

    augmented_config = _SchemaConfig(
        mode="flexible",
        fields=(*base_fields, *extra_fields),
        guaranteed_fields=base_schema_config.guaranteed_fields,
        required_fields=base_schema_config.required_fields,
        audit_fields=base_schema_config.audit_fields,
    )
    return create_schema_from_config(augmented_config, schema_name, allow_coercion=False)


__all__ = [
    "LLM_AUDIT_SUFFIXES",
    "LLM_GUARANTEED_SUFFIXES",
    "MULTI_QUERY_GUARANTEED_SUFFIXES",
    "_OUTPUT_FIELD_TYPE_TO_SCHEMA",
    "_SUFFIX_SCHEMA_TYPES",
    "_build_augmented_output_schema",
    "_build_multi_query_output_schema",
    "build_llm_audit_metadata",
    "get_llm_audit_fields",
    "get_llm_guaranteed_fields",
    "populate_llm_operational_fields",
]
