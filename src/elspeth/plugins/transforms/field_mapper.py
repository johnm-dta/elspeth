"""FieldMapper transform plugin.

Renames, selects, and reorganizes row fields.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

import copy
from typing import Any

from pydantic import Field

from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.sentinels import MISSING
from elspeth.plugins.utils import get_nested_field


class FieldMapperConfig(TransformDataConfig):
    """Configuration for field mapper transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    mapping: dict[str, str] = Field(default_factory=dict)
    select_only: bool = False
    strict: bool = False
    validate_input: bool = False  # Optional input validation


class FieldMapper(BaseTransform):
    """Map, rename, and select row fields.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        mapping: Dict of source_field -> target_field
            - Simple: {"old": "new"} renames old to new
            - Nested: {"meta.source": "origin"} extracts nested field
        select_only: If True, only include mapped fields (default: False)
        strict: If True, error on missing source fields (default: False)
        validate_input: If True, validate input against schema (default: False)
    """

    name = "field_mapper"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FieldMapperConfig.from_dict(config)
        self._mapping: dict[str, str] = cfg.mapping
        self._select_only: bool = cfg.select_only
        self._strict: bool = cfg.strict
        self._validate_input: bool = cfg.validate_input

        self._schema_config = cfg.schema_config

        # Create input schema from config
        # CRITICAL: allow_coercion=False - wrong types are source bugs
        self.input_schema = create_schema_from_config(
            cfg.schema_config,
            "FieldMapperInputSchema",
            allow_coercion=False,
        )

        # Output schema MUST be dynamic because FieldMapper changes row shape:
        # - With mapping, fields can be renamed
        # - With select_only=True, only mapped fields appear in output
        # The output shape depends on config, not input schema.
        # Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch
        self.output_schema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "FieldMapperOutputSchema",
            allow_coercion=False,
        )

    def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
        """Apply field mapping to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with mapped row data

        Raises:
            ValidationError: If validate_input=True and row fails schema validation.
                This indicates a bug in the upstream source/transform.
        """
        # Keep a normalized dict view only for validation and dotted-path lookups.
        row_data = row.to_dict()

        # Optional input validation - crash on wrong types (source bug!)
        if self._validate_input and not self._schema_config.is_observed:
            self.input_schema.model_validate(row_data)  # Raises on failure

        # Start with empty or copy depending on select_only
        if self._select_only:
            output: dict[str, Any] = {}
        else:
            output = copy.deepcopy(row_data)

        # Apply mappings
        applied_mappings: dict[str, str] = {}
        for source, target in self._mapping.items():
            if "." in source:
                value = get_nested_field(row_data, source)
            elif source in row:
                value = row[source]
            else:
                value = MISSING

            if value is MISSING:
                if self._strict:
                    return TransformResult.error(
                        {"reason": "missing_field", "field": source, "message": f"Required field '{source}' not found in row"}
                    )
                continue  # Skip missing fields in non-strict mode

            # Remove old key if it exists (for rename within same dict)
            if not self._select_only and "." not in source and source in row:
                if source in output:
                    del output[source]
                else:
                    normalized_source = row.contract.resolve_name(source)
                    if normalized_source in output:
                        del output[normalized_source]

            output[target] = value
            applied_mappings[source] = target

        # Track field changes
        fields_modified: list[str] = []
        fields_added: list[str] = []
        for target in applied_mappings.values():
            if target in row:
                fields_modified.append(target)
            else:
                fields_added.append(target)

        # Update contract to reflect field mapping (renames and removals)
        output_contract = narrow_contract_to_output(
            input_contract=row.contract,
            output_row=output,
            renamed_fields=applied_mappings,
        )

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "mapped",
                "fields_modified": fields_modified,
                "fields_added": fields_added,
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
