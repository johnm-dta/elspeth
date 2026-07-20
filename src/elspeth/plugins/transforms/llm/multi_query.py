"""Multi-query LLM support — query specifications and resolution.

Provides QuerySpec (named variable mapping) and resolve_queries() for
multi-query LLM transforms. Output field configuration (OutputFieldConfig,
ResponseFormat) used by both single and multi-query modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.config_base import PluginConfig


class OutputFieldType(StrEnum):
    """Supported types for structured output fields."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


class ResponseFormat(StrEnum):
    """LLM response format modes.

    - STANDARD: Uses {"type": "json_object"} - model outputs JSON but no schema enforcement
    - STRUCTURED: Uses {"type": "json_schema", ...} - API enforces exact schema compliance
    """

    STANDARD = "standard"
    STRUCTURED = "structured"


class OutputFieldConfig(PluginConfig):
    """Configuration for a single output field in the LLM response.

    Attributes:
        suffix: Column suffix in output row (e.g., "score" -> "{prefix}_score")
        type: Data type for schema enforcement
        values: Required for enum type - list of allowed values
    """

    suffix: str = Field(..., description="Column suffix in output row")
    type: OutputFieldType = Field(..., description="Data type for schema enforcement")
    values: list[str] | None = Field(None, description="Allowed values (required for enum type)")

    @model_validator(mode="after")
    def validate_enum_has_values(self) -> OutputFieldConfig:
        """Ensure enum type has values list."""
        if self.type == OutputFieldType.ENUM:
            if not self.values or len(self.values) == 0:
                raise ValueError("enum type requires non-empty 'values' list")
        elif self.values is not None:
            raise ValueError(f"'values' is only valid for enum type, not {self.type.value}")
        return self

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema property definition.

        Returns:
            JSON Schema dict for this field
        """
        if self.type == OutputFieldType.ENUM:
            # Enum uses 'enum' keyword with allowed values
            return {"type": "string", "enum": self.values}
        else:
            # Direct type mapping
            return {"type": self.type.value}


class QueryDefinition(BaseModel):
    """Typed authoring model for a single multi-query LLM query.

    This is the *public discovery* shape for one entry of ``LLMConfig.queries``.
    It replaces the previous untyped ``dict[str, Any]`` so the catalog can
    advertise the structured-output contract (``name``, ``input_fields``,
    ``template``, ``response_format``, ``output_fields``) and so malformed
    drafts fail closed with a safe ``PluginConfigError`` rather than a bare
    ``ValueError`` escaping as a 500.

    Both accepted authoring forms normalize to this model:

    * **Mapping form** — ``queries: {query_name: {...}}``. The value carries no
      ``name``; ``LLMConfig`` injects the mapping key as ``name`` before
      validation (see ``LLMConfig._inject_mapping_query_names``).
    * **List form** — ``queries: [{name: query_name, ...}]``. Each entry must
      carry its own ``name``; ``resolve_queries`` rejects a list entry with a
      missing name as a safe configuration error.

    ``name`` is therefore ``str | None`` at the model level (the mapping value
    legitimately omits it under ``extra=forbid``); the list-form requirement is
    enforced downstream in ``resolve_queries``. The frozen runtime spec remains
    :class:`QuerySpec`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(
        default=None,
        description="Unique query identifier (used in output field prefixes). Omitted in mapping form, where the mapping key supplies it.",
    )
    input_fields: dict[str, str] = Field(
        ...,
        description="Mapping of template variable name to row column name.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.STANDARD,
        description="LLM response format mode (standard JSON object vs. enforced json_schema).",
    )
    output_fields: list[OutputFieldConfig] | None = Field(
        default=None,
        description="Typed structured-output field definitions (None = unstructured response).",
    )
    template: str | None = Field(
        default=None,
        description="Per-query Jinja2 template override (None = use the config-level prompt_template).",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Per-query max_tokens override (None = use the config-level max_tokens).",
    )


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Domain-agnostic query specification for multi-query transforms.

    Uses named input_fields (dict mapping template variable name to row
    column name) for flexible variable binding in templates.

    Attributes:
        name: Unique query identifier (used in output field prefixes)
        input_fields: Mapping of template variable → row column name
        response_format: LLM response format mode
        output_fields: Typed output field definitions (None = unstructured)
        template: Per-query template override (None = use config-level template)
        max_tokens: Per-query max_tokens override (None = use config-level)
    """

    name: str
    input_fields: MappingProxyType[str, str]
    response_format: ResponseFormat = ResponseFormat.STANDARD
    output_fields: tuple[OutputFieldConfig, ...] | None = None
    template: str | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.input_fields:
            raise ValueError("input_fields must be non-empty")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool):
                raise TypeError(f"max_tokens must be int, got bool ({self.max_tokens!r})")
            if not isinstance(self.max_tokens, int):
                raise TypeError(f"max_tokens must be int, got {type(self.max_tokens).__name__}")
            if self.max_tokens <= 0:
                raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        object.__setattr__(self, "input_fields", MappingProxyType(dict(self.input_fields)))
        if self.output_fields is not None:
            object.__setattr__(self, "output_fields", tuple(self.output_fields))

    def build_template_context(self, row: PipelineRow | dict[str, Any]) -> dict[str, Any]:
        """Build template context mapping named variables to row values.

        Args:
            row: Full row data (dict or PipelineRow)

        Returns:
            Context dict with named variables and source_row reference

        Raises:
            KeyError: If a required row column is missing
        """
        context: dict[str, Any] = {}
        for template_var, row_column in self.input_fields.items():
            context[template_var] = row[row_column]
        context["source_row"] = row
        return context


# Pattern for detecting legacy positional template variables {{ input_N }}
_POSITIONAL_VAR_PATTERN = re.compile(r"\{\{\s*input_\d+\s*\}\}")


def _query_spec_from_definition(name: str, definition: QueryDefinition) -> QuerySpec:
    """Build the frozen runtime :class:`QuerySpec` from a typed authoring model."""
    output_fields = tuple(definition.output_fields) if definition.output_fields is not None else None
    return QuerySpec(
        name=name,
        input_fields=MappingProxyType(dict(definition.input_fields)),
        response_format=definition.response_format,
        output_fields=output_fields,
        template=definition.template,
        max_tokens=definition.max_tokens,
    )


def resolve_queries(
    queries: list[QuerySpec] | list[QueryDefinition] | dict[str, QueryDefinition],
) -> list[QuerySpec]:
    """Normalize typed query definitions into a list of QuerySpec.

    Accepts:
    - ``dict[str, QueryDefinition]``: mapping key becomes the query name
      (the value carries no ``name``).
    - ``list[QueryDefinition]``: each entry must carry its own ``name``.
    - ``list[QuerySpec]``: already-resolved specs, passed through as-is.

    Validates (cross-query rules that a per-field Pydantic model cannot express):
    - Non-empty input
    - List-form entries carry a ``name``
    - No duplicate query names
    - No legacy positional template variables (``{{ input_N }}``)
    - No output field suffix collisions across queries, and no collision with a
      reserved LLM suffix (``usage``/``model``/``error``)

    Per-field shape (missing ``input_fields``, invalid ``output_fields``, enum
    without ``values``, ...) is already enforced by ``QueryDefinition`` /
    ``OutputFieldConfig`` at model-validation time, so those never reach here.

    Args:
        queries: Typed query definitions in any supported form.

    Returns:
        List of validated QuerySpec instances.

    Raises:
        ValueError: If queries is empty, a list entry omits ``name``, names
            collide, positional variables are used, or output keys collide.
    """
    specs: list[QuerySpec] = []

    if isinstance(queries, dict):
        if not queries:
            raise ValueError("no queries configured")
        for name, definition in queries.items():
            specs.append(_query_spec_from_definition(name, definition))
    elif isinstance(queries, list):
        if not queries:
            raise ValueError("no queries configured")
        for item in queries:
            if isinstance(item, QuerySpec):
                specs.append(item)
            else:
                if item.name is None:
                    raise ValueError(
                        "List-form query entries must include a 'name'. Add 'name' to each "
                        "query, or author the queries as a mapping keyed by query name."
                    )
                specs.append(_query_spec_from_definition(item.name, item))
    else:
        raise TypeError(f"queries must be list or dict, got {type(queries).__name__}")

    # Validate: reject duplicate query names.
    # Dict-form configs are naturally unique (Python dict keys), but list-form
    # configs can have duplicate "name" fields. Duplicate names cause silent
    # data loss: per-query output keys ({name}_response, {name}_metadata) collide,
    # and dict.update() overwrites earlier query results.
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"Duplicate query name '{spec.name}'. Each query must have a unique name to prevent output field collisions.")
        seen_names.add(spec.name)

    # Validate: reject positional template variables
    for spec in specs:
        if spec.template and _POSITIONAL_VAR_PATTERN.search(spec.template):
            raise ValueError(
                f"Query '{spec.name}' template uses positional variables "
                f"(e.g., {{{{ input_1 }}}}). Use named input_fields instead: "
                f"map template variables to row columns via input_fields dict."
            )

    # Validate: check output field suffix collisions across queries
    from elspeth.plugins.transforms.llm import MULTI_QUERY_GUARANTEED_SUFFIXES

    reserved_suffixes = set()
    for suffix in MULTI_QUERY_GUARANTEED_SUFFIXES:
        if suffix:
            reserved_suffixes.add(suffix.lstrip("_"))
    # System-reserved suffixes used by multi-query error handling
    reserved_suffixes.add("error")

    seen_output_keys: dict[str, str] = {}  # full output key → first query name
    for spec in specs:
        if spec.output_fields:
            for field in spec.output_fields:
                # Warn on reserved suffixes
                if field.suffix in reserved_suffixes:
                    raise ValueError(
                        f"Query '{spec.name}' output field suffix '{field.suffix}' collides with "
                        f"reserved LLM suffix. Reserved suffixes: {sorted(reserved_suffixes)}. "
                        f"Choose a different suffix to prevent silent data loss."
                    )
                # Check for cross-query output key collisions.
                # Output keys are "{query_name}_{suffix}" (see MultiQueryStrategy),
                # so different queries MAY share the same suffix as long as the full
                # key is unique.
                output_key = f"{spec.name}_{field.suffix}"
                if output_key in seen_output_keys:
                    raise ValueError(
                        f"Output field key collision: key '{output_key}' "
                        f"used by both query '{seen_output_keys[output_key]}' "
                        f"and query '{spec.name}'"
                    )
                seen_output_keys[output_key] = spec.name

    return specs
