# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH.

Provides transforms for Azure OpenAI, OpenRouter, and other LLM providers.
Includes support for pooled execution, batch processing, and multi-query evaluation.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    transform = manager.get_transform_by_name("azure_llm")

For testing or advanced usage, import directly from module paths:
    from elspeth.plugins.llm.azure import AzureLLMTransform
    from elspeth.plugins.llm.templates import PromptTemplate

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

# Metadata field suffixes for contract-stable fields (downstream can depend on these)
LLM_GUARANTEED_SUFFIXES: tuple[str, ...] = (
    "",  # The response content field itself
    "_usage",  # Token usage dict {prompt_tokens, completion_tokens, total_tokens}
    "_model",  # Model identifier that actually responded
)

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
    return tuple(f"{response_field}{suffix}" for suffix in LLM_AUDIT_SUFFIXES)


def get_multi_query_guaranteed_fields(output_prefix: str) -> tuple[str, ...]:
    """Return contract-stable metadata field names for multi-query LLM transforms.

    Multi-query transforms emit suffixed output fields (e.g., category_score,
    category_rationale) but NOT the base field itself. This function returns
    only the metadata fields (_usage, _model) without the base field.

    For the output mapping fields (score, rationale, etc.), multi-query
    computes those separately from the output_mapping config.

    Args:
        output_prefix: Output field prefix (e.g., "category"). Must not be empty
            or whitespace-only.

    Returns:
        Tuple of metadata field names that are guaranteed to exist.

    Raises:
        ValueError: If output_prefix is empty or whitespace-only.
    """
    if not output_prefix or not output_prefix.strip():
        raise ValueError("output_prefix cannot be empty or whitespace-only")
    return tuple(f"{output_prefix}{suffix}" for suffix in MULTI_QUERY_GUARANTEED_SUFFIXES)


__all__ = [
    "LLM_AUDIT_SUFFIXES",
    "LLM_GUARANTEED_SUFFIXES",
    "MULTI_QUERY_GUARANTEED_SUFFIXES",
    "get_llm_audit_fields",
    "get_llm_guaranteed_fields",
    "get_multi_query_guaranteed_fields",
]
