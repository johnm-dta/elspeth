"""Runtime verification of sink required-field declarations (ADR-017).

This contract registers for ONE dispatch site:

    * ``boundary_check`` — sink-side row boundary before schema validation and
      before external sink I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from elspeth.contracts.declaration_contracts import (
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    OrchestrationInvariantError,
    SinkRequiredFieldsPayload,
    SinkRequiredFieldsViolation,
)
from elspeth.contracts.plugin_roles import sink_declared_required_fields
from elspeth.contracts.schema_contract import (
    FieldContract,
    SchemaContract,
)


def _build_contract(
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> SchemaContract:
    fields = tuple(
        FieldContract(
            normalized_name=name,
            original_name=name,
            python_type=str,
            required=True,
            source="inferred",
            nullable=False,
        )
        for name in required_fields
    ) + tuple(
        FieldContract(
            normalized_name=name,
            original_name=name,
            python_type=str,
            required=False,
            source="inferred",
            nullable=False,
        )
        for name in optional_fields
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def verify_sink_required_fields(
    *,
    declared_required_fields: frozenset[str],
    row_data: Mapping[str, object],
    row_contract: SchemaContract | None,
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify the row satisfies the sink's required-field declaration."""
    runtime_observed = frozenset(row_data.keys())
    missing = declared_required_fields - runtime_observed
    if not missing:
        return

    contract_context = ""
    if row_contract is not None:
        required_in_contract = row_contract.required_field_names
        optional_in_contract: list[str] = []
        for missing_name in missing:
            normalized = row_contract.find_name(missing_name)
            if normalized is not None and normalized not in required_in_contract:
                optional_in_contract.append(missing_name)
        if optional_in_contract:
            contract_context = (
                f" Fields {optional_in_contract} are optional in the row's schema contract "
                f"(likely from coalesce merge). Fix: ensure all branches produce these fields as required."
            )

    raise SinkRequiredFieldsViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "declared": sorted(declared_required_fields),
            "runtime_observed": sorted(runtime_observed),
            "missing": sorted(missing),
        },
        message=(
            f"Sink {plugin_name!r} (node {node_id!r}) declared required fields "
            f"{sorted(declared_required_fields)!r} but row {row_id!r} only exposed "
            f"{sorted(runtime_observed)!r}; missing {sorted(missing)!r}.{contract_context}"
        ),
    )


class SinkRequiredFieldsContract(DeclarationContract):
    """ADR-017 adopter for sink ``declared_required_fields``."""

    name: ClassVar[str] = "sink_required_fields"
    payload_schema: ClassVar[type] = SinkRequiredFieldsPayload

    def applies_to(self, plugin: Any) -> bool:
        return bool(sink_declared_required_fields(plugin))

    @implements_dispatch_site("boundary_check")
    def boundary_check(
        self,
        inputs: BoundaryInputs,
        outputs: BoundaryOutputs,
    ) -> None:
        declared_required_fields = inputs.plugin.declared_required_fields
        sink_node_id = inputs.plugin.node_id
        if sink_node_id is None:
            raise OrchestrationInvariantError(
                f"Sink {inputs.plugin.name!r} has no node_id set at sink-required-fields boundary check time."
            )
        verify_sink_required_fields(
            declared_required_fields=declared_required_fields,
            row_data=cast(Mapping[str, object], inputs.row_data),
            row_contract=cast(SchemaContract | None, inputs.row_contract),
            plugin_name=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        class _MinimalSink:
            name = "NegativeSinkRequiredFieldsExample"
            node_id = "sink-required-neg-1"
            declared_required_fields: frozenset[str] = frozenset({"customer_id", "amount"})

        inputs = BoundaryInputs(
            plugin=_MinimalSink(),
            node_id="sink-required-neg-1",
            run_id="sink-required-neg-run",
            row_id="sink-required-neg-row",
            token_id="sink-required-neg-token",
            static_contract=frozenset({"customer_id", "amount"}),
            row_data={"customer_id": "v"},
            row_contract=_build_contract(required_fields=("customer_id",), optional_fields=("amount",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _NonApplyingSink:
            name = "NonApplyingSinkRequiredFieldsExample"
            node_id = "sink-required-non-fire-1"
            declared_required_fields: frozenset[str] = frozenset()

        inputs = BoundaryInputs(
            plugin=_NonApplyingSink(),
            node_id="sink-required-non-fire-1",
            run_id="sink-required-non-fire-run",
            row_id="sink-required-non-fire-row",
            token_id="sink-required-non-fire-token",
            static_contract=frozenset(),
            row_data={"customer_id": "v", "amount": "1"},
            row_contract=_build_contract(required_fields=("customer_id", "amount")),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))


register_declaration_contract(SinkRequiredFieldsContract())
