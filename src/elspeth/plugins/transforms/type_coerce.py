"""TypeCoerce transform plugin.

Performs explicit, strict, per-field type normalization.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.contracts.contexts import TransformContext


class CoercionError(Exception):
    """Raised when type coercion fails."""

    def __init__(self, value: Any, target_type: str, reason: str) -> None:
        self.value = value
        self.target_type = target_type
        self.reason = reason
        super().__init__(f"Cannot coerce {type(value).__name__} to {target_type}: {reason}")


def coerce_to_int(value: Any) -> int:
    """Coerce value to int with strict rules.

    Accepts:
        - int (unchanged)
        - float with no fractional part (3.0 -> 3)
        - string of integer after trim ("42", " -7 ")

    Rejects:
        - float with fractional part (3.9 -> error)
        - string with decimal ("3.5" -> error)
        - scientific notation string ("1e3" -> error)
        - empty/whitespace string
        - bool (True/False are technically ints but rejected)
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "int", "None cannot be converted to int")

    # Reject bool explicitly (before int check, since bool is subclass of int)
    if type(value) is bool:
        raise CoercionError(value, "int", "bool cannot be converted to int")

    # int passes through
    if type(value) is int:
        return value

    # float: only if no fractional part
    if type(value) is float:
        if not math.isfinite(value):
            raise CoercionError(value, "int", "non-finite float cannot be converted to int")
        if value != int(value):
            raise CoercionError(value, "int", f"float {value} has fractional part")
        return int(value)

    # string: parse as integer
    if type(value) is str:
        trimmed = value.strip()
        if not trimmed:
            raise CoercionError(value, "int", "empty string cannot be converted to int")
        try:
            return int(trimmed)
        except ValueError as exc:
            raise CoercionError(value, "int", f"'{trimmed}' is not a valid integer string") from exc

    raise CoercionError(value, "int", f"unsupported type {type(value).__name__}")


def coerce_to_float(value: Any) -> float:
    """Coerce value to float with strict rules.

    Accepts:
        - float (unchanged, must be finite)
        - int -> float
        - numeric string after trim ("12.5", "1e3")

    Rejects:
        - non-finite floats (NaN, inf, -inf)
        - empty/whitespace string
        - bool
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "float", "None cannot be converted to float")

    # Reject bool explicitly
    if type(value) is bool:
        raise CoercionError(value, "float", "bool cannot be converted to float")

    # float: check finite
    if type(value) is float:
        if not math.isfinite(value):
            raise CoercionError(value, "float", "non-finite float values are not allowed")
        return value

    # int -> float
    if type(value) is int:
        return float(value)

    # string: parse as float
    if type(value) is str:
        trimmed = value.strip()
        if not trimmed:
            raise CoercionError(value, "float", "empty string cannot be converted to float")
        try:
            result = float(trimmed)
        except ValueError as exc:
            raise CoercionError(value, "float", f"'{trimmed}' is not a valid numeric string") from exc
        if not math.isfinite(result):
            raise CoercionError(value, "float", f"'{trimmed}' produces non-finite value")
        return result

    raise CoercionError(value, "float", f"unsupported type {type(value).__name__}")


# Boolean string mappings (case-insensitive after trim)
_BOOL_TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes", "y", "on"})
_BOOL_FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no", "n", "off", ""})


def coerce_to_bool(value: Any) -> bool:
    """Coerce value to bool with strict rules.

    Accepts:
        - bool (unchanged)
        - int 0 -> False, int 1 -> True
        - string true set (case-insensitive): true, 1, yes, y, on
        - string false set (case-insensitive): false, 0, no, n, off, "" (empty/whitespace-only)

    Rejects:
        - other integers (2, -1, etc.)
        - other strings
        - float
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "bool", "None cannot be converted to bool")

    # bool passes through
    if type(value) is bool:
        return value

    # int: only 0 and 1
    if type(value) is int:
        if value == 0:
            return False
        if value == 1:
            return True
        raise CoercionError(value, "bool", f"only 0 and 1 can be converted to bool, got {value}")

    # float: reject
    if type(value) is float:
        raise CoercionError(value, "bool", "float cannot be converted to bool")

    # string: check against true/false sets
    if type(value) is str:
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE_STRINGS:
            return True
        if normalized in _BOOL_FALSE_STRINGS:
            return False
        raise CoercionError(value, "bool", f"'{value}' is not a valid boolean string")

    raise CoercionError(value, "bool", f"unsupported type {type(value).__name__}")


# Scalar types accepted for string conversion
_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool)


def coerce_to_str(value: Any) -> str:
    """Coerce value to str with strict rules.

    Accepts:
        - str (unchanged)
        - int, float, bool -> Python str()

    Rejects:
        - list, dict, objects, bytes (not scalars)
        - None
    """
    # Reject None first
    if value is None:
        raise CoercionError(value, "str", "None cannot be converted to str")

    # Only accept scalar types
    if type(value) not in _SCALAR_TYPES:
        raise CoercionError(value, "str", f"{type(value).__name__} is not a scalar type")

    return str(value)


class ConversionSpec(BaseModel):
    """Single field conversion specification."""

    model_config = {"extra": "forbid", "frozen": True}

    field: str
    to: Literal["int", "float", "bool", "str"]

    @field_validator("field")
    @classmethod
    def _validate_field_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field name must not be empty")
        return v


class TypeCoerceConfig(TransformDataConfig):
    """Configuration for type coercion transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """

    conversions: list[ConversionSpec] = Field(
        ...,
        description="List of field type conversions to apply",
    )

    @model_validator(mode="after")
    def _validate_conversions_not_empty(self) -> TypeCoerceConfig:
        if not self.conversions:
            raise ValueError("conversions must contain at least one conversion")
        return self


# Conversion function dispatch table
_COERCION_FUNCS: dict[str, Any] = {
    "int": coerce_to_int,
    "float": coerce_to_float,
    "bool": coerce_to_bool,
    "str": coerce_to_str,
}

# Target type checks for idempotency
_TARGET_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


class TypeCoerce(BaseTransform):
    """Perform explicit, strict, per-field type normalization.

    Conversions are evaluated in order on a working copy of the row.
    If all conversions succeed, the updated row is emitted.
    If any conversion fails, the original row is returned as an error
    and no partial changes are emitted on the success path.

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        conversions: List of {field, to} specs defining type conversions
    """

    name = "type_coerce"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:cd665aaedabd43a5"
    config_model = TypeCoerceConfig
    passes_through_input = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TypeCoerceConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._conversions = cfg.conversions
        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_type_coerce_output_schema_config(cfg.schema_config)

        self.input_schema = create_schema_from_config(
            cfg.schema_config,
            "TypeCoerceInput",
            allow_coercion=False,
        )
        self.output_schema = create_schema_from_config(
            self._output_schema_config,
            "TypeCoerceOutput",
            allow_coercion=False,
        )

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "conversions": [{"field": "type_coerce_probe_1", "to": "str"}],
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="type_coerce_probe_1",
                value=7,
            )
        ]

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Apply type conversions to row fields.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with converted field values, or error if any conversion fails
        """
        # Work on a copy to support atomic rollback
        output = copy.deepcopy(row.to_dict())
        fields_coerced: list[str] = []
        fields_unchanged: list[str] = []
        conversion_targets: dict[str, Literal["int", "float", "bool", "str"]] = {}

        for spec in self._conversions:
            config_field = spec.field  # Field name from config (may be original header)
            target_type_name = spec.to

            # Check field exists (PipelineRow resolves both original and normalized names)
            if config_field not in row:
                return TransformResult.error(
                    {
                        "reason": "missing_field",
                        "field": config_field,
                        "message": f"Field '{config_field}' not found in row",
                    }
                )

            # Resolve to normalized key for output dict (handles original header names)
            normalized_key = row.contract.find_name(config_field)
            if normalized_key is None:
                # Field exists in row but not in contract — use config name as-is
                # (shouldn't happen for valid rows, but defensive for edge cases)
                normalized_key = config_field
            conversion_targets[normalized_key] = target_type_name

            value = row[config_field]

            # Check for None
            if value is None:
                return TransformResult.error(
                    {
                        "reason": "type_mismatch",
                        "field": config_field,
                        "expected": target_type_name,
                        "actual": "None",
                        "message": f"Field '{config_field}' is None",
                    }
                )

            # Check if already correct type (idempotent)
            target_type = _TARGET_TYPES[target_type_name]
            # Use type() not isinstance() to avoid bool matching int
            if type(value) is target_type:
                fields_unchanged.append(normalized_key)
                continue

            # Apply conversion
            coerce_func = _COERCION_FUNCS[target_type_name]
            try:
                converted = coerce_func(value)
            except CoercionError as e:
                return TransformResult.error(
                    {
                        "reason": "type_mismatch",
                        "field": config_field,
                        "expected": target_type_name,
                        "actual": type(value).__name__,
                        "message": e.reason,
                    }
                )

            output[normalized_key] = converted
            fields_coerced.append(normalized_key)

        output_contract = self._build_output_contract(row.contract, conversion_targets)
        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "coerced",
                "fields_modified": fields_coerced,
                "metadata": {
                    "fields_coerced": fields_coerced,
                    "fields_unchanged": fields_unchanged,
                    "rules_evaluated": len(self._conversions),
                },
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass

    def _build_type_coerce_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Return the output schema config after applying configured target types."""
        if schema_config.fields is None:
            return self._build_output_schema_config(schema_config)

        target_types = {spec.field: spec.to for spec in self._conversions}
        output_fields = tuple(
            FieldDefinition(
                name=field.name,
                field_type=target_types.get(field.name, field.field_type),
                required=field.required,
                nullable=False if field.name in target_types else field.nullable,
            )
            for field in schema_config.fields
        )
        return SchemaConfig(
            mode=schema_config.mode,
            fields=output_fields,
            guaranteed_fields=schema_config.guaranteed_fields,
            required_fields=schema_config.required_fields,
            audit_fields=schema_config.audit_fields,
        )

    def _build_output_contract(
        self,
        contract: SchemaContract,
        conversion_targets: dict[str, Literal["int", "float", "bool", "str"]],
    ) -> SchemaContract:
        """Return an aligned output contract whose field types match the emitted row."""
        if not conversion_targets:
            return self._align_output_contract(contract)

        changed = False
        output_fields: list[FieldContract] = []
        for field in contract.fields:
            target_type_name = conversion_targets.get(field.normalized_name)
            if target_type_name is None:
                output_fields.append(field)
                continue

            target_type = _TARGET_TYPES[target_type_name]
            new_field = FieldContract(
                normalized_name=field.normalized_name,
                original_name=field.original_name,
                python_type=target_type,
                required=field.required,
                source=field.source,
                nullable=False,
            )
            output_fields.append(new_field)
            if new_field != field:
                changed = True

        evolved_contract = contract
        if changed:
            evolved_contract = SchemaContract(
                mode=contract.mode,
                fields=tuple(output_fields),
                locked=contract.locked,
            )

        return self._align_output_contract(evolved_contract)
