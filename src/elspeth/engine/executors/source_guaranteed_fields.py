"""Runtime verification of source guaranteed-field declarations (ADR-016).

This contract registers for ONE dispatch site:

    * ``boundary_check`` — source-side row boundary after token creation and
      before source node-state completion.
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
    FrameworkBugError,
    OrchestrationInvariantError,
    SourceGuaranteedFieldsPayload,
    SourceGuaranteedFieldsViolation,
)
from elspeth.contracts.plugin_roles import source_declared_guaranteed_fields
from elspeth.contracts.schema_contract import (
    FieldContract,
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


def verify_source_guaranteed_fields(
    *,
    declared_guaranteed_fields: frozenset[str],
    row_data: Mapping[str, object],
    row_contract: SchemaContract | None,
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify the emitted source row satisfies the source's guarantees."""
    if row_contract is None:
        raise FrameworkBugError(
            f"Source {plugin_name!r} emitted a valid row without a schema contract at the source boundary. Framework invariant violated."
        )

    runtime_contract_fields = frozenset(fc.normalized_name for fc in row_contract.fields)
    runtime_payload_fields = frozenset(row_data.keys())
    runtime_observed = runtime_contract_fields & runtime_payload_fields
    missing = declared_guaranteed_fields - runtime_observed
    if not missing:
        return

    raise SourceGuaranteedFieldsViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "declared": sorted(declared_guaranteed_fields),
            "runtime_observed": sorted(runtime_observed),
            "missing": sorted(missing),
        },
        message=(
            f"Source {plugin_name!r} (node {node_id!r}) declared guaranteed fields "
            f"{sorted(declared_guaranteed_fields)!r} but emitted row {row_id!r} with "
            f"runtime-observed fields {sorted(runtime_observed)!r}; missing {sorted(missing)!r}."
        ),
    )


class SourceGuaranteedFieldsContract(DeclarationContract):
    """ADR-016 adopter for source ``declared_guaranteed_fields``."""

    name: ClassVar[str] = "source_guaranteed_fields"
    payload_schema: ClassVar[type] = SourceGuaranteedFieldsPayload

    def applies_to(self, plugin: Any) -> bool:
        return bool(source_declared_guaranteed_fields(plugin))

    @implements_dispatch_site("boundary_check")
    def boundary_check(
        self,
        inputs: BoundaryInputs,
        outputs: BoundaryOutputs,
    ) -> None:
        declared_guaranteed_fields = inputs.plugin.declared_guaranteed_fields
        source_node_id = inputs.plugin.node_id
        if source_node_id is None:
            raise OrchestrationInvariantError(
                f"Source {inputs.plugin.name!r} has no node_id set at source-guaranteed-fields boundary check time."
            )
        verify_source_guaranteed_fields(
            declared_guaranteed_fields=declared_guaranteed_fields,
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
        class _MinimalSource:
            name = "NegativeSourceGuaranteedFieldsExample"
            node_id = "source-guaranteed-neg-1"
            declared_guaranteed_fields: frozenset[str] = frozenset({"customer_id", "account_id"})

        inputs = BoundaryInputs(
            plugin=_MinimalSource(),
            node_id="source-guaranteed-neg-1",
            run_id="source-guaranteed-neg-run",
            row_id="source-guaranteed-neg-row",
            token_id="source-guaranteed-neg-token",
            static_contract=frozenset({"customer_id", "account_id"}),
            row_data={"customer_id": "v", "account_id": "v"},
            row_contract=_build_contract(("account_id",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _NonApplyingSource:
            name = "NonApplyingSourceGuaranteedFieldsExample"
            node_id = "source-guaranteed-non-fire-1"
            declared_guaranteed_fields: frozenset[str] = frozenset()

        inputs = BoundaryInputs(
            plugin=_NonApplyingSource(),
            node_id="source-guaranteed-non-fire-1",
            run_id="source-guaranteed-non-fire-run",
            row_id="source-guaranteed-non-fire-row",
            token_id="source-guaranteed-non-fire-token",
            static_contract=frozenset(),
            row_data={"customer_id": "v"},
            row_contract=_build_contract(("customer_id",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))


register_declaration_contract(SourceGuaranteedFieldsContract())
