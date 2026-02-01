"""Multi-query LLM support for case study x criteria cross-product evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.llm.azure import AzureOpenAIConfig


class OutputFieldType(str, Enum):
    """Supported types for structured output fields."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


class ResponseFormat(str, Enum):
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


@dataclass
class QuerySpec:
    """Specification for a single query in the cross-product.

    Represents one (case_study, criterion) pair to be evaluated.

    Attributes:
        case_study_name: Name of the case study (e.g., "cs1")
        criterion_name: Name of the criterion (e.g., "diagnosis")
        input_fields: List of row field names to map to input_1, input_2, etc.
        output_prefix: Prefix for output fields (e.g., "cs1_diagnosis")
        criterion_data: Full criterion object for template injection
        case_study_data: Full case study object for template injection
    """

    case_study_name: str
    criterion_name: str
    input_fields: list[str]
    output_prefix: str
    criterion_data: dict[str, Any]
    case_study_data: dict[str, Any]
    max_tokens: int | None = None  # Per-query override for max_tokens

    def build_template_context(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build template context for this query.

        Maps input_fields to input_1, input_2, etc. and injects criterion and case_study data.

        Args:
            row: Full row data

        Returns:
            Context dict with input_N, criterion, case_study, and row
        """
        context: dict[str, Any] = {}

        # Map input fields to positional variables
        # Access directly - missing field is a config error, should crash
        for i, field_name in enumerate(self.input_fields, start=1):
            if field_name not in row:
                raise KeyError(f"Required field '{field_name}' not found in row for query {self.output_prefix}")
            context[f"input_{i}"] = row[field_name]

        # Inject criterion data
        context["criterion"] = self.criterion_data

        # Inject case study data (name, context, metadata)
        context["case_study"] = self.case_study_data

        # Include full row for row-based lookups
        context["row"] = row

        return context


class CaseStudyConfig(PluginConfig):
    """Configuration for a single case study.

    Attributes:
        name: Unique identifier for this case study (used in output field prefix)
        input_fields: Row field names to map to input_1, input_2, etc.
        context: Context type (e.g., "Capability", "Capacity") for template use
        description: Human-readable description of this case study type
    """

    name: str = Field(..., description="Case study identifier")
    input_fields: list[str] = Field(..., description="Row fields to map to input_N")
    context: str | None = Field(None, description="Context type (e.g., Capability, Capacity)")
    description: str | None = Field(None, description="Human-readable description")

    @field_validator("input_fields")
    @classmethod
    def validate_input_fields_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one input field."""
        if not v:
            raise ValueError("input_fields cannot be empty")
        return v

    def to_template_data(self) -> dict[str, Any]:
        """Convert to dict for template injection."""
        return {
            "name": self.name,
            "context": self.context,
            "description": self.description,
            "input_fields": self.input_fields,
        }


class CriterionConfig(PluginConfig):
    """Configuration for a single evaluation criterion.

    All fields except 'name' are optional and available in templates
    via {{ row.criterion.field_name }}.

    Attributes:
        name: Unique identifier (used in output field prefix)
        code: Short code for lookups (e.g., "DIAG")
        description: Human-readable description
        subcriteria: List of subcriteria names/descriptions
    """

    name: str = Field(..., description="Criterion identifier")
    code: str | None = Field(None, description="Short code for lookups")
    description: str | None = Field(None, description="Human-readable description")
    subcriteria: list[str] = Field(default_factory=list, description="Subcriteria list")
    max_tokens: int | None = Field(None, gt=0, description="Per-criterion max_tokens override")

    def to_template_data(self) -> dict[str, Any]:
        """Convert to dict for template injection."""
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "subcriteria": self.subcriteria,
        }


class MultiQueryConfig(AzureOpenAIConfig):
    """Configuration for multi-query LLM transform.

    Extends AzureOpenAIConfig with:
    - case_studies: List of case study definitions
    - criteria: List of criterion definitions
    - output_mapping: JSON field -> typed output field configuration
    - response_format: Response format mode (standard or structured)

    The cross-product of case_studies x criteria defines all queries.

    Example:
        output_mapping:
          score:
            suffix: score
            type: integer
          rationale:
            suffix: rationale
            type: string
          confidence:
            suffix: confidence
            type: enum
            values: [low, medium, high]
    """

    case_studies: list[CaseStudyConfig] = Field(
        ...,
        description="Case study definitions",
        min_length=1,
    )
    criteria: list[CriterionConfig] = Field(
        ...,
        description="Criterion definitions",
        min_length=1,
    )
    output_mapping: dict[str, OutputFieldConfig] = Field(
        ...,
        description="JSON field -> typed output field configuration",
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.STANDARD,
        description="Response format: 'standard' (JSON mode) or 'structured' (schema-enforced)",
    )

    @field_validator("output_mapping", mode="before")
    @classmethod
    def parse_output_mapping(cls, v: Any) -> dict[str, Any]:
        """Parse output_mapping from config dict format."""
        if not isinstance(v, dict):
            raise ValueError("output_mapping must be a dict")
        if not v:
            raise ValueError("output_mapping cannot be empty")
        # Pydantic will handle nested OutputFieldConfig parsing
        return v

    @model_validator(mode="after")
    def validate_no_output_key_collisions(self) -> MultiQueryConfig:
        """Validate no duplicate names or reserved suffix collisions.

        Checks:
        1. No duplicate case_study names
        2. No duplicate criterion names
        3. No output_mapping suffixes that collide with reserved LLM suffixes

        Raises:
            ValueError: If duplicates or collisions are detected.
        """
        from elspeth.plugins.llm import LLM_AUDIT_SUFFIXES, LLM_GUARANTEED_SUFFIXES

        # Check for duplicate case_study names
        case_study_names: set[str] = set()
        for cs in self.case_studies:
            if cs.name in case_study_names:
                raise ValueError(f"Duplicate case_study name: '{cs.name}'. Each case_study must have a unique name.")
            case_study_names.add(cs.name)

        # Check for duplicate criterion names
        criterion_names: set[str] = set()
        for crit in self.criteria:
            if crit.name in criterion_names:
                raise ValueError(f"Duplicate criterion name: '{crit.name}'. Each criterion must have a unique name.")
            criterion_names.add(crit.name)

        # Build set of reserved suffixes (strip leading underscore for comparison)
        reserved_suffixes = set()
        for suffix in LLM_GUARANTEED_SUFFIXES + LLM_AUDIT_SUFFIXES:
            if suffix:  # Skip empty string
                # Reserved suffixes are stored as "_usage", we compare against "usage"
                reserved_suffixes.add(suffix.lstrip("_"))

        # Check output_mapping suffixes don't collide with reserved suffixes
        for json_field, field_config in self.output_mapping.items():
            if field_config.suffix in reserved_suffixes:
                raise ValueError(
                    f"Output mapping '{json_field}' has suffix '{field_config.suffix}' that collides with reserved LLM suffix '_{field_config.suffix}'. "
                    f"Reserved suffixes: {sorted('_' + s for s in reserved_suffixes)}"
                )

        return self

    def build_json_schema(self) -> dict[str, Any]:
        """Build JSON Schema for structured outputs.

        Returns:
            Complete JSON Schema dict for the response format
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for json_field, field_config in self.output_mapping.items():
            properties[json_field] = field_config.to_json_schema()
            required.append(json_field)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def build_response_format(self) -> dict[str, Any]:
        """Build the response_format parameter for the LLM API.

        Returns:
            Dict to pass as response_format to the LLM API
        """
        if self.response_format == ResponseFormat.STRUCTURED:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "query_response",
                    "strict": True,
                    "schema": self.build_json_schema(),
                },
            }
        else:
            # Standard JSON mode
            return {"type": "json_object"}

    def expand_queries(self) -> list[QuerySpec]:
        """Expand config into QuerySpec list (case_studies x criteria).

        Returns:
            List of QuerySpec, one per (case_study, criterion) pair
        """
        specs: list[QuerySpec] = []

        for case_study in self.case_studies:
            for criterion in self.criteria:
                spec = QuerySpec(
                    case_study_name=case_study.name,
                    criterion_name=criterion.name,
                    input_fields=case_study.input_fields,
                    output_prefix=f"{case_study.name}_{criterion.name}",
                    criterion_data=criterion.to_template_data(),
                    case_study_data=case_study.to_template_data(),
                    max_tokens=criterion.max_tokens,
                )
                specs.append(spec)

        return specs
