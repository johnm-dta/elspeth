"""JSONExplode deaggregation transform.

Transforms one row containing an array field into multiple rows, one for each
element in the array. This is the inverse of aggregation (1-to-N expansion).

THREE-TIER TRUST MODEL COMPLIANCE:

Per the plugin protocol, transforms TRUST that pipeline data types are correct:
- Source validates that required fields exist and have correct types
- Transforms access fields directly without defensive checks
- Type violations (missing field, wrong type) indicate UPSTREAM BUGS and should CRASH

JSONExplode does NOT return TransformResult.error() for type violations because:
1. Missing field = source should have validated -> crash surfaces config bug
2. Wrong type = source should have validated -> crash surfaces config bug
3. There are no VALUE-level operations that can fail in this transform

Therefore, JSONExplode inherits from DataPluginConfig (NOT TransformDataConfig)
and has no on_error configuration.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import DataPluginConfig, PluginConfigError
from elspeth.plugins.infrastructure.results import TransformResult

if TYPE_CHECKING:
    from elspeth.contracts.plugin_assistance import PluginAssistance
    from elspeth.contracts.plugin_semantics import InputSemanticRequirements


def _build_json_explode_input_requirements(
    *,
    array_field: str,
) -> InputSemanticRequirements:
    from elspeth.contracts.plugin_semantics import (
        FieldSemanticRequirement,
        InputSemanticRequirements,
        SemanticValueType,
        UnknownSemanticPolicy,
    )

    return InputSemanticRequirements(
        fields=(
            FieldSemanticRequirement(
                field_name=array_field,
                accepted_content_kinds=frozenset(),
                accepted_text_framings=frozenset(),
                requirement_code="json_explode.array_field.list",
                accepted_value_types=frozenset({SemanticValueType.LIST}),
                severity="high",
                unknown_policy=UnknownSemanticPolicy.FAIL,
                configured_by=("array_field",),
            ),
        ),
    )


class JSONExplodeConfig(DataPluginConfig):
    """Configuration for JSON explode transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.

    Extends DataPluginConfig (not TransformDataConfig) because JSONExplode
    has no on_error behavior -- type violations crash to surface upstream bugs.
    Routing fields such as on_success are owned by TransformSettings at the
    pipeline settings layer, not plugin options.

    _plugin_component_type overrides DataPluginConfig (None) because this
    config extends DataPluginConfig directly, bypassing TransformDataConfig.

    Attributes:
        array_field: Name of the array field to explode (required)
        output_field: Name for the exploded element (default: "item")
        include_index: Whether to include item_index field (default: True)
    """

    _plugin_component_type: ClassVar[str | None] = "transform"

    array_field: str = Field(..., description="Name of the array field to explode")
    output_field: str = Field(default="item", description="Name for the exploded element")
    include_index: bool = Field(default=True, description="Whether to include item_index field")

    @field_validator("array_field")
    @classmethod
    def _validate_array_field(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return v

    @field_validator("output_field")
    @classmethod
    def _validate_output_field(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("output_field must be non-empty")
        if not v.isidentifier():
            raise ValueError(f"output_field must be a valid Python identifier, got {v!r}")
        return v

    @model_validator(mode="after")
    def _reject_field_collision(self) -> JSONExplodeConfig:
        if self.output_field == self.array_field:
            raise ValueError(f"output_field and array_field must differ, both are '{self.output_field}'")
        if self.include_index and self.output_field == "item_index":
            raise ValueError(
                "output_field='item_index' conflicts with the auto-generated index field "
                "when include_index=True — the index would overwrite the exploded item"
            )
        return self

    @model_validator(mode="after")
    def _require_statically_resolvable_array_field_when_output_contract_depends_on_it(self) -> JSONExplodeConfig:
        """Fail closed when static output-contract derivation cannot resolve aliases.

        Runtime row access can resolve original headers through ``PipelineRow.contract``,
        but constructor-time contract propagation only sees normalized schema metadata.
        When guaranteed fields or explicit schema fields participate in output-contract
        derivation, the consumed array field must already be expressed in that same
        normalized namespace.
        """
        static_output_inputs: set[str] = set(self.schema_config.guaranteed_fields or ())
        if self.schema_config.fields is not None:
            static_output_inputs.update(field.name for field in self.schema_config.fields)

        if static_output_inputs and self.array_field not in static_output_inputs:
            raise ValueError(
                "array_field must use the normalized field name when schema declares "
                "guaranteed_fields or explicit fields for output-contract propagation. "
                f"Got {self.array_field!r}; known normalized fields are {sorted(static_output_inputs)!r}."
            )

        return self


class JSONExplode(BaseTransform):
    """Explode a JSON array field into multiple rows.

    This is a deaggregation transform that expands one input row into multiple
    output rows, one for each element in the array field. The creates_tokens=True
    flag signals to the engine that new token IDs should be created for each
    output row with parent linkage to the input token.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        array_field: Required. Name of the array field to explode
        output_field: Name for the exploded element (default: "item")
        include_index: Whether to include item_index field (default: True)

    Example:
        Input:  {"id": 1, "items": [{"name": "a"}, {"name": "b"}]}
        Output: [
            {"id": 1, "item": {"name": "a"}, "item_index": 0},
            {"id": 1, "item": {"name": "b"}, "item_index": 1},
        ]

    TRUST MODEL:
        This transform trusts that the source validated:
        - array_field exists in the row
        - array_field value is a list/array

        If these invariants are violated, the transform CRASHES (KeyError, TypeError)
        to surface the upstream bug. This is intentional - see module docstring.
    """

    name = "json_explode"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:6b8ffd45b12ab18f"
    config_model = JSONExplodeConfig
    creates_tokens = True  # CRITICAL: enables new token creation for deaggregation

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "array_field": "json_explode_items",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the JSONExplode transform.

        Args:
            config: Configuration dict containing array_field and optional settings

        Raises:
            PluginConfigError: If required config is missing or invalid
        """
        if "on_success" in config:
            raise PluginConfigError(
                "JSONExplode does not accept 'on_success' in plugin options. "
                "Set routing at the settings layer with transforms[].on_success.",
                plugin_class="JSONExplodeConfig",
                component_type="transform",
            )
        super().__init__(config)
        cfg = JSONExplodeConfig.from_dict(config, plugin_name=self.name)
        self._array_field = cfg.array_field
        self._output_field = cfg.output_field
        self._include_index = cfg.include_index

        # Declare output fields for centralized collision detection in TransformExecutor.
        fields = [cfg.output_field]
        if cfg.include_index:
            fields.append("item_index")
        self.declared_output_fields = frozenset(fields)

        self._schema_config = cfg.schema_config

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "JSONExplode",
            adds_fields=True,
        )
        self._output_schema_config = self._build_json_explode_output_schema_config(cfg)

    def input_semantic_requirements(self) -> InputSemanticRequirements:
        return _build_json_explode_input_requirements(array_field=self._array_field)

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> PluginAssistance | None:
        from elspeth.contracts.plugin_assistance import PluginAssistance

        if issue_code is None:
            return PluginAssistance(
                plugin_name="json_explode",
                issue_code=None,
                summary="Deaggregate a list-valued field — emits one row per array element, preserving sibling fields and optionally adding an item_index.",
                composer_hints=(
                    "array_field MUST hold a real list (value_type=list). A JSON-looking string does NOT count — insert a parser first.",
                    "Single-query LLM responses are strings. Do not wire response_field directly to json_explode; explode after parsing.",
                    "Sibling fields are duplicated onto every emitted row — that's by design for fan-out lineage.",
                ),
            )
        if issue_code != "json_explode.array_field.list":
            return None
        return PluginAssistance(
            plugin_name="json_explode",
            issue_code="json_explode.array_field.list",
            summary=(
                "json_explode expands a field that is already a real list-shaped "
                "pipeline value. A string is not a valid array_field, even when it "
                "contains JSON-looking text."
            ),
            suggested_fixes=(
                "Point array_field at a source or transform output that declares value_type=list.",
                "If the upstream value is JSON text, insert an explicit parser/validator transform that emits a real list before json_explode.",
                "For single-query llm output, do not wire response_field directly to json_explode; the response_field is a string.",
            ),
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the real array-consumption path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._array_field,
                value=["only-item"],
            )
        ]

    def _build_json_explode_output_schema_config(self, cfg: JSONExplodeConfig) -> SchemaConfig:
        """Build output schema config excluding array_field.

        JSONExplode removes array_field from output and adds output_field
        (and optionally item_index). The base _build_output_schema_config()
        incorrectly retains array_field in guaranteed_fields.
        """
        base_guaranteed = set(cfg.schema_config.guaranteed_fields or ())
        output_field_defs = cfg.schema_config.fields

        # Remove array_field from output guarantees (it's consumed at runtime)
        base_guaranteed.discard(cfg.array_field)

        if cfg.schema_config.fields is not None:
            kept_fields = tuple(field for field in cfg.schema_config.fields if field.name != cfg.array_field)
            extra_fields: list[FieldDefinition] = []
            if all(field.name != cfg.output_field for field in kept_fields):
                extra_fields.append(
                    FieldDefinition(
                        name=cfg.output_field,
                        field_type="any",
                        required=True,
                        nullable=False,
                    )
                )
            if cfg.include_index and all(field.name != "item_index" for field in kept_fields):
                extra_fields.append(
                    FieldDefinition(
                        name="item_index",
                        field_type="int",
                        required=True,
                        nullable=False,
                    )
                )
            output_field_defs = (*kept_fields, *extra_fields)

        # Add declared output fields (output_field + optionally item_index)
        output_fields = base_guaranteed | self.declared_output_fields

        # Preserve None-vs-empty-tuple semantics: None = abstain, () = explicitly empty.
        # If upstream declared guarantees or we computed non-empty output, declare explicitly.
        upstream_declared = cfg.schema_config.guaranteed_fields is not None
        if upstream_declared or output_fields:
            guaranteed_fields_result = tuple(sorted(output_fields))
        else:
            guaranteed_fields_result = None

        return SchemaConfig(
            mode=cfg.schema_config.mode,
            fields=output_field_defs,
            guaranteed_fields=guaranteed_fields_result,
            audit_fields=cfg.schema_config.audit_fields,
            required_fields=cfg.schema_config.required_fields,
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Explode array field into multiple rows.

        Args:
            row: Input row containing the array field
            ctx: Plugin context

        Returns:
            TransformResult with multiple output rows (success_multi) or
            single row for empty arrays (success)

        Raises:
            KeyError: If array_field is missing (upstream bug)
            TypeError: If array_field is not a list (upstream bug)
        """
        # Direct access - TRUST that source validated field exists
        # KeyError here = upstream bug (source didn't validate field exists)
        array_value = row[self._array_field]

        # Contract enforcement: array_field must be list or tuple.
        # PipelineRow deep-freezes data (list→tuple), so both are valid.
        # Strings/dicts are iterable but would produce garbage - fail explicitly.
        if not isinstance(array_value, (list, tuple)):
            raise TypeError(
                f"Field '{self._array_field}' must be a list, got {type(array_value).__name__}. "
                f"This indicates an upstream validation bug - check source schema or prior transforms."
            )

        row_data = row.to_dict()
        if self._array_field in row_data:
            normalized_array_field = self._array_field
        else:
            # row[self._array_field] above already validated resolvability.
            normalized_array_field = row.contract.resolve_name(self._array_field)
        base = {k: v for k, v in row_data.items() if k != normalized_array_field}

        # Empty array: nothing to deaggregate — quarantine with clear audit trail
        if len(array_value) == 0:
            return TransformResult.error(
                {"reason": "invalid_input", "field": self._array_field, "error": "empty array"},
                retryable=False,
            )

        # Explode array into multiple rows
        # Deep copy base for each row to prevent cross-row mutation via shared
        # nested references (e.g., downstream mutating row["metadata"]["key"]
        # would corrupt sibling rows if they shared the same dict object).
        output_rows: list[dict[str, Any]] = []
        for i, item in enumerate(array_value):
            output = copy.deepcopy(base)
            output[self._output_field] = item
            if self._include_index:
                output["item_index"] = i
            output_rows.append(output)

        fields_added = [self._output_field]
        if self._include_index:
            fields_added.append("item_index")

        if len(output_rows) > 1:
            first_keys = set(output_rows[0].keys())
            for i, output_row in enumerate(output_rows[1:], start=1):
                row_keys = set(output_row.keys())
                if row_keys != first_keys:
                    raise ValueError(
                        f"Multi-row output has heterogeneous schema: "
                        f"row 0 has fields {sorted(first_keys)}, "
                        f"row {i} has fields {sorted(row_keys)}"
                    )

        # Determine the contract type for the output field.
        # If the exploded array contains heterogeneous types (e.g., ["a", {"k": 1}]),
        # the output field type must be `object` (the universal type) rather than
        # the type inferred from only the first element. This prevents downstream
        # components from relying on a contract type that doesn't hold for all rows.
        item_types = {type(item) for item in array_value}
        output_field_is_heterogeneous = len(item_types) > 1

        # Update contract using first output row (all rows have same schema)
        output_contract = narrow_contract_to_output(
            input_contract=row.contract,
            output_row=output_rows[0],
        )
        if output_contract.find_field(self._output_field) is None:
            output_contract = SchemaContract(
                mode=output_contract.mode,
                fields=(
                    *output_contract.fields,
                    FieldContract(
                        normalized_name=self._output_field,
                        original_name=self._output_field,
                        python_type=object,
                        required=False,
                        source="inferred",
                    ),
                ),
                locked=True,
            )
        elif output_field_is_heterogeneous:
            # Override the inferred type to `object` when items have mixed types.
            # narrow_contract_to_output inferred the type from the first element only,
            # which would be wrong for subsequent rows with different element types.
            patched_fields = tuple(
                FieldContract(
                    normalized_name=fc.normalized_name,
                    original_name=fc.original_name,
                    python_type=object if fc.normalized_name == self._output_field else fc.python_type,
                    required=fc.required,
                    source=fc.source,
                    nullable=fc.nullable,
                )
                for fc in output_contract.fields
            )
            output_contract = SchemaContract(
                mode=output_contract.mode,
                fields=patched_fields,
                locked=True,
            )
        output_contract = self._align_output_contract(output_contract)

        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in output_rows],
            success_reason={
                "action": "transformed",
                "fields_added": fields_added,
                "fields_removed": [self._array_field],
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
