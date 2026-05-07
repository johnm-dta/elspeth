"""Runtime verification of transform ``_output_schema_config`` mode semantics (ADR-014).

This contract registers for TWO dispatch sites:

    * ``post_emission_check`` — single-token path from ``TransformExecutor``.
    * ``batch_flush_check``   — batch-aware flush path from ``RowProcessor``.

The contract is intentionally scoped to transforms that expose
``_output_schema_config``. ``applies_to`` is O(1): it reads only that single
flag and does not inspect the config shape until the runtime check runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from elspeth.contracts.declaration_contracts import (
    BatchFlushInputs,
    BatchFlushOutputs,
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    FrameworkBugError,
    OrchestrationInvariantError,
    SchemaConfigFieldMetadataMismatch,
    SchemaConfigModePayload,
    SchemaConfigModeViolation,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import (
    FieldContract,
    PipelineRow,
    SchemaContract,
)
from elspeth.contracts.schema_contract_factory import create_contract_from_config


def _build_contract(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
) -> SchemaContract:
    return SchemaContract(
        mode=mode,  # type: ignore[arg-type]  # test/example helper uses closed-set literals
        fields=tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=str,
                required=True,
                source="inferred",
                nullable=False,
            )
            for name in fields
        ),
        locked=locked,
    )


def _build_row(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
) -> PipelineRow:
    return PipelineRow(dict.fromkeys(fields, "v"), _build_contract(fields, mode=mode, locked=locked))


def _allowed_declared_fields(output_schema_config: SchemaConfig) -> frozenset[str]:
    declared_fields = frozenset(fd.name for fd in output_schema_config.fields) if output_schema_config.fields is not None else frozenset()
    explicit_guarantees = frozenset(output_schema_config.guaranteed_fields or ())
    audit_fields = frozenset(output_schema_config.audit_fields or ())
    return declared_fields | explicit_guarantees | audit_fields


def _declared_field_contracts(output_schema_config: SchemaConfig) -> dict[str, FieldContract]:
    return {field.normalized_name: field for field in create_contract_from_config(output_schema_config).fields}


def _collect_field_metadata_mismatches(
    *,
    output_schema_config: SchemaConfig,
    emitted_row: PipelineRow,
) -> list[SchemaConfigFieldMetadataMismatch]:
    if output_schema_config.fields is None:
        return []

    mismatches: list[SchemaConfigFieldMetadataMismatch] = []
    declared_fields = _declared_field_contracts(output_schema_config)
    runtime_row_fields = frozenset(emitted_row.keys())

    for field_name, declared_field in declared_fields.items():
        if field_name not in runtime_row_fields:
            continue

        observed_field = emitted_row.contract.find_field(field_name)
        if (
            observed_field is not None
            and observed_field.python_type is declared_field.python_type
            and observed_field.required == declared_field.required
            and observed_field.nullable == declared_field.nullable
        ):
            continue

        mismatches.append(
            {
                "field": field_name,
                "expected_type": declared_field.python_type.__name__,
                "observed_type": observed_field.python_type.__name__ if observed_field is not None else "<missing>",
                "expected_required": declared_field.required,
                "observed_required": observed_field.required if observed_field is not None else False,
                "expected_nullable": declared_field.nullable,
                "observed_nullable": observed_field.nullable if observed_field is not None else False,
                "observed_present": observed_field is not None,
            }
        )

    return mismatches


def verify_schema_config_mode(
    *,
    output_schema_config: SchemaConfig,
    emitted_rows: Sequence[PipelineRow],
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify emitted contracts reflect the transform's declared schema mode."""
    if not emitted_rows:
        return

    declared_mode = output_schema_config.mode
    declared_locked = True
    allowed_declared_fields = _allowed_declared_fields(output_schema_config)
    required_output_fields = output_schema_config.get_effective_guaranteed_fields()

    for emitted in emitted_rows:
        if emitted.contract is None:
            raise FrameworkBugError(f"Transform {plugin_name!r} emitted row with no contract. Framework invariant violated.")

        observed_mode = emitted.contract.mode.lower()
        observed_locked = emitted.contract.locked
        runtime_row_fields = frozenset(emitted.keys())
        runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)

        undeclared_extra_fields: list[str] | None = None
        if declared_mode == "fixed":
            runtime_fields = runtime_contract_fields | runtime_row_fields
            extras = runtime_fields - allowed_declared_fields
            if extras:
                undeclared_extra_fields = sorted(extras)

        missing_required_fields: list[str] | None = None
        missing = required_output_fields - runtime_row_fields
        if missing:
            missing_required_fields = sorted(missing)

        field_metadata_mismatches = _collect_field_metadata_mismatches(
            output_schema_config=output_schema_config,
            emitted_row=emitted,
        )

        if (
            observed_mode == declared_mode
            and observed_locked == declared_locked
            and undeclared_extra_fields is None
            and missing_required_fields is None
            and not field_metadata_mismatches
        ):
            continue

        payload: SchemaConfigModePayload = {
            "declared_mode": declared_mode,
            "observed_mode": observed_mode,
            "declared_locked": declared_locked,
            "observed_locked": observed_locked,
        }
        problem_details: list[str] = []
        if observed_mode != declared_mode:
            problem_details.append(f"mode {observed_mode!r} != declared {declared_mode!r}")
        if observed_locked != declared_locked:
            problem_details.append(f"locked={observed_locked!r} != declared {declared_locked!r}")
        if undeclared_extra_fields is not None:
            payload["undeclared_extra_fields"] = undeclared_extra_fields
            problem_details.append(f"undeclared extra fields {undeclared_extra_fields!r}")
        if missing_required_fields is not None:
            payload["runtime_observed_fields"] = sorted(runtime_row_fields)
            payload["missing_required_fields"] = missing_required_fields
            problem_details.append(f"missing required fields {missing_required_fields!r}")
        if field_metadata_mismatches:
            payload["field_metadata_mismatches"] = field_metadata_mismatches
            mismatch_fields = [mismatch["field"] for mismatch in field_metadata_mismatches]
            problem_details.append(f"field metadata mismatches for {mismatch_fields!r}")

        raise SchemaConfigModeViolation(
            plugin=plugin_name,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            payload=payload,
            message=(
                f"Transform {plugin_name!r} (node {node_id!r}) emitted output schema "
                f"semantics inconsistent with its declaration for row {row_id!r}: "
                f"{'; '.join(problem_details)}."
            ),
        )


class SchemaConfigModeContract(DeclarationContract):
    """ADR-014 adopter for transform output schema mode declarations."""

    name: ClassVar[str] = "schema_config_mode"
    payload_schema: ClassVar[type] = SchemaConfigModePayload

    def applies_to(self, plugin: Any) -> bool:
        return plugin._output_schema_config is not None

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        output_schema_config = inputs.plugin._output_schema_config
        if output_schema_config is None:
            return
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {inputs.plugin.name!r} has no node_id set at schema-config-mode check time.")
        verify_schema_config_mode(
            output_schema_config=output_schema_config,
            emitted_rows=outputs.emitted_rows,
            plugin_name=inputs.plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        output_schema_config = inputs.plugin._output_schema_config
        if output_schema_config is None:
            return
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(
                f"Transform {inputs.plugin.name!r} has no node_id set at schema-config-mode batch-flush check time."
            )
        verify_schema_config_mode(
            output_schema_config=output_schema_config,
            emitted_rows=outputs.emitted_rows,
            plugin_name=inputs.plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        class _MinimalTransform:
            name = "NegativeSchemaConfigModeExample"
            node_id = "schema-config-mode-neg-1"
            passes_through_input = False
            declared_input_fields: ClassVar[frozenset[str]] = frozenset()
            is_batch_aware = False
            _output_schema_config = SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": [{"name": "source", "type": "str", "required": True, "nullable": False}],
                }
            )

        inputs = PostEmissionInputs(
            plugin=_MinimalTransform(),
            node_id="schema-config-mode-neg-1",
            run_id="schema-config-mode-neg-run",
            row_id="schema-config-mode-neg-row",
            token_id="schema-config-mode-neg-token",
            input_row=_build_row(("source",), mode="FIXED"),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_build_row(("source",), mode="OBSERVED"),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def negative_example_batch_flush(cls) -> ExampleBundle:
        class _MinimalBatchTransform:
            name = "NegativeSchemaConfigModeBatchExample"
            node_id = "schema-config-mode-batch-neg-1"
            passes_through_input = False
            declared_input_fields: ClassVar[frozenset[str]] = frozenset()
            is_batch_aware = True
            _output_schema_config = SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": [{"name": "source", "type": "str", "required": True, "nullable": False}],
                }
            )

        token_row = _build_row(("source",), mode="FIXED")
        inputs = BatchFlushInputs(
            plugin=_MinimalBatchTransform(),
            node_id="schema-config-mode-batch-neg-1",
            run_id="schema-config-mode-batch-neg-run",
            row_id="schema-config-mode-batch-neg-row",
            token_id="schema-config-mode-batch-neg-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_build_row(("source", "extra"), mode="FIXED"),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _NonApplyingTransform:
            name = "NonApplyingSchemaConfigModeExample"
            node_id = "schema-config-mode-non-fire-1"
            passes_through_input = False
            declared_input_fields: ClassVar[frozenset[str]] = frozenset()
            is_batch_aware = False
            _output_schema_config = None

        inputs = PostEmissionInputs(
            plugin=_NonApplyingTransform(),
            node_id="schema-config-mode-non-fire-1",
            run_id="schema-config-mode-non-fire-run",
            row_id="schema-config-mode-non-fire-row",
            token_id="schema-config-mode-non-fire-token",
            input_row=_build_row(("source",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_build_row(("source",), mode="FIXED"),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply_batch_flush(cls) -> ExampleBundle:
        class _NonApplyingBatchTransform:
            name = "NonApplyingSchemaConfigModeBatchExample"
            node_id = "schema-config-mode-batch-non-fire-1"
            passes_through_input = False
            declared_input_fields: ClassVar[frozenset[str]] = frozenset()
            is_batch_aware = True
            _output_schema_config = None

        token_row = _build_row(("source",))
        inputs = BatchFlushInputs(
            plugin=_NonApplyingBatchTransform(),
            node_id="schema-config-mode-batch-non-fire-1",
            run_id="schema-config-mode-batch-non-fire-run",
            row_id="schema-config-mode-batch-non-fire-row",
            token_id="schema-config-mode-batch-non-fire-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_build_row(("source",), mode="FIXED"),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))


register_declaration_contract(SchemaConfigModeContract())
