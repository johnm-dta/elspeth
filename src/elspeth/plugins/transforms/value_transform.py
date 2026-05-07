"""ValueTransform transform plugin.

Applies expressions to compute new or modified field values.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.expression_parser import (
    ExpressionEvaluationError,
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class OperationSpec(BaseModel):
    """Single value transform operation specification."""

    model_config = {"extra": "forbid", "frozen": True}

    target: str
    expression: str
    # Parsed expression stored after validation
    _parsed_expression: ExpressionParser | None = None

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target field name must not be empty")
        return v

    @field_validator("expression")
    @classmethod
    def _validate_expression(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("expression must not be empty")
        return v

    @model_validator(mode="after")
    def _parse_expression(self) -> OperationSpec:
        """Parse and validate expression at config time."""
        try:
            parser = ExpressionParser(self.expression)
            # Store the parsed expression for later use
            object.__setattr__(self, "_parsed_expression", parser)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Expression syntax error: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Expression contains forbidden constructs: {e}") from e
        return self

    def get_parser(self) -> ExpressionParser:
        """Get the pre-parsed expression parser."""
        if self._parsed_expression is None:
            # Re-parse if needed (shouldn't happen after validation)
            return ExpressionParser(self.expression)
        return self._parsed_expression


class ValueTransformConfig(TransformDataConfig):
    """Configuration for value transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    operations: list[OperationSpec] = Field(
        ...,
        description="List of operations to apply (target + expression pairs)",
    )

    @model_validator(mode="after")
    def _validate_operations_not_empty(self) -> ValueTransformConfig:
        if not self.operations:
            raise ValueError("operations must contain at least one operation")
        return self


# =============================================================================
# ValueTransform Plugin Class
# =============================================================================


class ValueTransform(BaseTransform):
    """Apply expressions to compute new or modified field values.

    Operations are evaluated in order on a working copy of the row.
    Each operation sees the results of prior operations (sequential visibility).
    If all operations succeed, the updated row is emitted.
    If any operation fails, the original row is returned as an error
    and no partial changes are emitted on the success path.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        operations: List of {target, expression} specs defining field computations
    """

    name = "value_transform"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:5a371ef8d6e0a5e9"
    config_model = ValueTransformConfig
    passes_through_input = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = ValueTransformConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._operations = cfg.operations
        self._configured_targets = frozenset(op.target for op in self._operations)
        self._schema_config = cfg.schema_config

        # declared_output_fields intentionally empty — we can't statically know which
        # targets are new vs overwrites, and overwrites are an intentional feature.
        # The executor's field collision check only runs when this is non-empty.
        self.declared_output_fields: frozenset[str] = frozenset()

        self._output_schema_config = self._build_value_transform_output_schema_config(cfg)

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "ValueTransform",
            adds_fields=True,
        )

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "operations": [
                {
                    "target": "value_transform_probe_added_1",
                    "expression": "1",
                }
            ],
        }

    def _build_value_transform_output_schema_config(
        self,
        cfg: ValueTransformConfig,
    ) -> SchemaConfig:
        """Build output guarantees for configured targets without forcing collision checks."""

        base_guaranteed = set(cfg.schema_config.guaranteed_fields or ())
        output_fields = base_guaranteed | self._configured_targets

        # Preserve None-vs-empty-tuple semantics: None = abstain, () = explicitly empty.
        # If upstream declared guarantees or this transform always writes targets,
        # declare the effective guarantees explicitly for DAG validation.
        upstream_declared = cfg.schema_config.guaranteed_fields is not None
        if upstream_declared or output_fields:
            guaranteed_fields_result = tuple(sorted(output_fields))
        else:
            guaranteed_fields_result = None

        return SchemaConfig(
            mode=cfg.schema_config.mode,
            fields=cfg.schema_config.fields,
            guaranteed_fields=guaranteed_fields_result,
            audit_fields=cfg.schema_config.audit_fields,
            required_fields=cfg.schema_config.required_fields,
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Apply expression operations to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with computed field values, or error if any operation fails
        """
        # Work on a copy to support atomic rollback
        working_data = copy.deepcopy(row.to_dict())
        working_contract = row.contract
        fields_modified: list[str] = []
        fields_added: list[str] = []
        original_fields = set(row.to_dict().keys())

        for op in self._operations:
            target = op.target
            parser = op.get_parser()

            # Create PipelineRow for evaluation to preserve dual-name access
            # (expressions can use original headers like row['Price USD'])
            working_row = PipelineRow(working_data, working_contract)

            try:
                result = parser.evaluate(working_row)
            except ExpressionEvaluationError as e:
                return TransformResult.error(
                    {
                        "reason": "invalid_input",
                        "field": target,
                        "message": str(e),
                    }
                )

            # Track field changes
            if target in original_fields:
                if target not in fields_modified:
                    fields_modified.append(target)
            else:
                if target not in fields_added:
                    fields_added.append(target)

            # Write result to working copy
            working_data[target] = result
            if working_contract.find_field(target) is None:
                working_contract = working_contract.with_field(target, target, result)

        output_contract = self._align_output_contract(working_contract)
        return TransformResult.success(
            PipelineRow(working_data, output_contract),
            success_reason={
                "action": "transformed",
                "fields_modified": fields_modified,
                "fields_added": fields_added,
                "metadata": {
                    "operations_applied": len(self._operations),
                },
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
