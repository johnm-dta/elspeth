"""Runtime verification of ``declared_output_fields`` declarations (ADR-011).

This contract registers for TWO dispatch sites:

    * ``post_emission_check`` — single-token path from ``TransformExecutor``.
    * ``batch_flush_check``   — batch-aware flush path from ``RowProcessor``.

Both sites share the same runtime invariant: every emitted row must expose
every field named in ``plugin.declared_output_fields``. Runtime observation is
the intersection of the emitted row's contract fields and payload keys,
mirroring the pass-through contract's "contract ∩ payload" posture.
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
    DeclaredOutputFieldRowViolationPayload,
    DeclaredOutputFieldsPayload,
    DeclaredOutputFieldsViolation,
    FrameworkBugError,
    OrchestrationInvariantError,
)
from elspeth.contracts.plugin_roles import require_declared_output_fields_plugin
from elspeth.contracts.schema_contract import (
    FieldContract,
    PipelineRow,
    SchemaContract,
)


def _build_contract(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
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
        locked=True,
    )


def _build_row(fields: tuple[str, ...]) -> PipelineRow:
    return PipelineRow(dict.fromkeys(fields, "v"), _build_contract(fields))


def verify_declared_output_fields(
    *,
    declared_output_fields: frozenset[str],
    emitted_rows: Sequence[PipelineRow],
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify every emitted row carries the transform's declared output fields."""
    if not emitted_rows:
        return

    violations: list[DeclaredOutputFieldRowViolationPayload] = []
    for emitted_index, emitted in enumerate(emitted_rows):
        if emitted.contract is None:
            raise FrameworkBugError(f"Transform {plugin_name!r} emitted row with no contract. Framework invariant violated.")
        runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
        runtime_payload_fields = frozenset(emitted.keys())
        runtime_observed = runtime_contract_fields & runtime_payload_fields
        missing = declared_output_fields - runtime_observed
        if not missing:
            continue
        violations.append(
            {
                "emitted_index": emitted_index,
                "runtime_observed": sorted(runtime_observed),
                "missing": sorted(missing),
            }
        )

    if not violations:
        return

    failing_indexes = [entry["emitted_index"] for entry in violations]
    raise DeclaredOutputFieldsViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "declared": sorted(declared_output_fields),
            "violations": violations,
        },
        message=(
            f"Transform {plugin_name!r} (node {node_id!r}) declared output fields "
            f"{sorted(declared_output_fields)!r} but emitted row(s) at indexes "
            f"{failing_indexes!r} that violated the declaration for row {row_id!r}."
        ),
    )


class DeclaredOutputFieldsContract(DeclarationContract):
    """ADR-011 adopter for transform ``declared_output_fields`` declarations."""

    name: ClassVar[str] = "declared_output_fields"
    payload_schema: ClassVar[type] = DeclaredOutputFieldsPayload

    def applies_to(self, plugin: Any) -> bool:
        typed_plugin = require_declared_output_fields_plugin(plugin)
        return bool(typed_plugin.declared_output_fields)

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        plugin = require_declared_output_fields_plugin(inputs.plugin)
        transform_node_id = plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {plugin.name!r} has no node_id set at declared-output-fields check time.")
        verify_declared_output_fields(
            declared_output_fields=plugin.declared_output_fields,
            emitted_rows=outputs.emitted_rows,
            plugin_name=plugin.name,
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
        plugin = require_declared_output_fields_plugin(inputs.plugin)
        transform_node_id = plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(
                f"Transform {plugin.name!r} has no node_id set at declared-output-fields batch-flush check time."
            )
        verify_declared_output_fields(
            declared_output_fields=plugin.declared_output_fields,
            emitted_rows=outputs.emitted_rows,
            plugin_name=plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        class _MinimalTransform:
            name = "NegativeDeclaredOutputFieldsExample"
            node_id = "declared-output-neg-1"
            declared_output_fields: frozenset[str] = frozenset({"new_a", "new_b"})
            passes_through_input = False
            _output_schema_config = None

        input_row = _build_row(("source",))
        inputs = PostEmissionInputs(
            plugin=_MinimalTransform(),
            node_id="declared-output-neg-1",
            run_id="declared-output-neg-run",
            row_id="declared-output-neg-row",
            token_id="declared-output-neg-token",
            input_row=input_row,
            static_contract=frozenset({"new_a", "new_b"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_build_row(("source", "new_a")),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def negative_example_batch_flush(cls) -> ExampleBundle:
        class _MinimalBatchTransform:
            name = "NegativeDeclaredOutputFieldsBatchExample"
            node_id = "declared-output-batch-neg-1"
            declared_output_fields: frozenset[str] = frozenset({"new_a", "new_b"})
            passes_through_input = False
            _output_schema_config = None

        token_row = _build_row(("source",))
        inputs = BatchFlushInputs(
            plugin=_MinimalBatchTransform(),
            node_id="declared-output-batch-neg-1",
            run_id="declared-output-batch-neg-run",
            row_id="declared-output-batch-neg-row",
            token_id="declared-output-batch-neg-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"new_a", "new_b"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_build_row(("source", "new_a")),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _NonApplyingTransform:
            name = "NonApplyingDeclaredOutputFieldsExample"
            node_id = "declared-output-non-fire-1"
            declared_output_fields: frozenset[str] = frozenset()
            passes_through_input = False
            _output_schema_config = None

        input_row = _build_row(("source",))
        inputs = PostEmissionInputs(
            plugin=_NonApplyingTransform(),
            node_id="declared-output-non-fire-1",
            run_id="declared-output-non-fire-run",
            row_id="declared-output-non-fire-row",
            token_id="declared-output-non-fire-token",
            input_row=input_row,
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_build_row(("source", "new_a", "new_b")),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply_batch_flush(cls) -> ExampleBundle:
        class _NonApplyingBatchTransform:
            name = "NonApplyingDeclaredOutputFieldsBatchExample"
            node_id = "declared-output-batch-non-fire-1"
            declared_output_fields: frozenset[str] = frozenset()
            passes_through_input = False
            _output_schema_config = None

        token_row = _build_row(("source",))
        inputs = BatchFlushInputs(
            plugin=_NonApplyingBatchTransform(),
            node_id="declared-output-batch-non-fire-1",
            run_id="declared-output-batch-non-fire-run",
            row_id="declared-output-batch-non-fire-row",
            token_id="declared-output-batch-non-fire-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_build_row(("source", "new_a", "new_b")),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))


register_declaration_contract(DeclaredOutputFieldsContract())
