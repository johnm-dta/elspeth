"""Runtime verification of transform ``can_drop_rows`` declarations (ADR-012).

This contract registers for TWO dispatch sites:

    * ``post_emission_check`` — single-token path from ``TransformExecutor``.
    * ``batch_flush_check``   — batch-flush path from ``RowProcessor``.

The contract is intentionally scoped to pass-through transforms only. It
retires ADR-009 Clause 3's empty-emission carve-out mechanically: empty
success output is now allowed only when the plugin explicitly declares
``can_drop_rows=True``.
"""

from __future__ import annotations

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
    OrchestrationInvariantError,
    UnexpectedEmptyEmissionPayload,
    UnexpectedEmptyEmissionViolation,
    ZeroEmissionSuccessContractViolation,
)
from elspeth.contracts.schema_contract import (
    FieldContract,
    PipelineRow,
    SchemaContract,
)


def _build_row(fields: tuple[str, ...]) -> PipelineRow:
    contract = SchemaContract(
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
    return PipelineRow(dict.fromkeys(fields, "v"), contract)


def _require_bool_flag(plugin: Any, *, attr_name: str) -> bool:
    """Return a declaration flag only when it is an exact ``bool``."""

    value = getattr(plugin, attr_name)
    if type(value) is not bool:
        raise TypeError(f"{type(plugin).__name__}.{attr_name} must be bool, got {type(value).__name__!r}.")
    return value


def verify_can_drop_rows(
    *,
    plugin: Any,
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
    emitted_count: int,
) -> None:
    """Raise when a pass-through transform emitted zero rows without opt-in."""
    if emitted_count != 0:
        return
    passes_through_input = _require_bool_flag(plugin, attr_name="passes_through_input")
    can_drop_rows = _require_bool_flag(plugin, attr_name="can_drop_rows")

    raise UnexpectedEmptyEmissionViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "passes_through_input": passes_through_input,
            "can_drop_rows": can_drop_rows,
            "emitted_count": emitted_count,
        },
        message=(
            f"Transform {plugin_name!r} (node {node_id!r}) emitted zero rows for "
            f"row {row_id!r} but declares passes_through_input=True and "
            f"can_drop_rows=False."
        ),
    )


def verify_zero_emission_declaration_path(
    *,
    plugin: Any,
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
    emitted_count: int,
    used_success_empty: bool,
) -> None:
    """Reject ``success_empty()`` outside the explicit filter declaration path.

    ADR-012 scopes ``can_drop_rows`` to pass-through filters. The dispatcher-
    owned contract covers the ``passes_through_input=True`` / missing-opt-in
    case; this inline guard closes the non-pass-through escape hatch so
    ``success_empty()`` cannot silently route as ``DROPPED_BY_FILTER``.
    """
    if not used_success_empty:
        return
    if emitted_count != 0:
        return
    passes_through_input = _require_bool_flag(plugin, attr_name="passes_through_input")
    can_drop_rows = _require_bool_flag(plugin, attr_name="can_drop_rows")
    if passes_through_input:
        return

    raise ZeroEmissionSuccessContractViolation(
        transform=plugin_name,
        transform_node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        passes_through_input=passes_through_input,
        can_drop_rows=can_drop_rows,
        emitted_count=emitted_count,
        message=(
            f"Transform {plugin_name!r} (node {node_id!r}) returned "
            f"TransformResult.success_empty() for row {row_id!r}, but zero-"
            f"emission success is reserved for pass-through filters declaring "
            f"passes_through_input=True and can_drop_rows=True."
        ),
    )


class CanDropRowsContract(DeclarationContract):
    """ADR-012 adopter for empty-emission governance."""

    name: ClassVar[str] = "can_drop_rows"
    payload_schema: ClassVar[type] = UnexpectedEmptyEmissionPayload

    def applies_to(self, plugin: Any) -> bool:
        passes_through_input = _require_bool_flag(plugin, attr_name="passes_through_input")
        can_drop_rows = _require_bool_flag(plugin, attr_name="can_drop_rows")
        return passes_through_input and not can_drop_rows

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        if not self.applies_to(inputs.plugin):
            return
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {inputs.plugin.name!r} has no node_id set at can-drop-rows check time.")
        verify_can_drop_rows(
            plugin=inputs.plugin,
            plugin_name=inputs.plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            emitted_count=len(outputs.emitted_rows),
        )

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        if not self.applies_to(inputs.plugin):
            return
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(
                f"Transform {inputs.plugin.name!r} has no node_id set at can-drop-rows batch-flush check time."
            )
        verify_can_drop_rows(
            plugin=inputs.plugin,
            plugin_name=inputs.plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            emitted_count=len(outputs.emitted_rows),
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        class _MinimalTransform:
            name = "NegativeCanDropRowsExample"
            node_id = "can-drop-rows-neg-1"
            passes_through_input = True
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            declared_input_fields: frozenset[str] = frozenset()
            is_batch_aware = False
            _output_schema_config = None

        inputs = PostEmissionInputs(
            plugin=_MinimalTransform(),
            node_id="can-drop-rows-neg-1",
            run_id="can-drop-rows-neg-run",
            row_id="can-drop-rows-neg-row",
            token_id="can-drop-rows-neg-token",
            input_row=_build_row(("source",)),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=())
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _OutOfScopeTransform:
            name = "NonApplyingCanDropRowsExample"
            node_id = "can-drop-rows-non-fire-1"
            passes_through_input = False
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            declared_input_fields: frozenset[str] = frozenset()
            is_batch_aware = False
            _output_schema_config = None

        inputs = PostEmissionInputs(
            plugin=_OutOfScopeTransform(),
            node_id="can-drop-rows-non-fire-1",
            run_id="can-drop-rows-non-fire-run",
            row_id="can-drop-rows-non-fire-row",
            token_id="can-drop-rows-non-fire-token",
            input_row=_build_row(("source",)),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=())
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def negative_example_batch_flush(cls) -> ExampleBundle:
        class _MinimalBatchTransform:
            name = "NegativeCanDropRowsBatchExample"
            node_id = "can-drop-rows-batch-neg-1"
            passes_through_input = True
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            declared_input_fields: frozenset[str] = frozenset()
            is_batch_aware = True
            _output_schema_config = None

        token_row = _build_row(("source",))
        inputs = BatchFlushInputs(
            plugin=_MinimalBatchTransform(),
            node_id="can-drop-rows-batch-neg-1",
            run_id="can-drop-rows-batch-neg-run",
            row_id="can-drop-rows-batch-neg-row",
            token_id="can-drop-rows-batch-neg-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=())
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply_batch_flush(cls) -> ExampleBundle:
        class _OutOfScopeBatchTransform:
            name = "NonApplyingCanDropRowsBatchExample"
            node_id = "can-drop-rows-batch-non-fire-1"
            passes_through_input = False
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            declared_input_fields: frozenset[str] = frozenset()
            is_batch_aware = True
            _output_schema_config = None

        token_row = _build_row(("source",))
        inputs = BatchFlushInputs(
            plugin=_OutOfScopeBatchTransform(),
            node_id="can-drop-rows-batch-non-fire-1",
            run_id="can-drop-rows-batch-non-fire-run",
            row_id="can-drop-rows-batch-non-fire-row",
            token_id="can-drop-rows-batch-non-fire-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=())
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))


register_declaration_contract(CanDropRowsContract())
