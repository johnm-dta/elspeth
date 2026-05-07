"""Runtime verification of ``declared_input_fields`` declarations (ADR-013).

This contract registers for ONE dispatch site:

    * ``pre_emission_check`` — single-row path from ``TransformExecutor``
      before ``transform.process()`` runs.

This contract applies only to single-row transforms. The framework has no
batch-pre-execution dispatch surface, so a batch-aware transform that declares
``declared_input_fields`` fails closed via both construction-time
normalization and this contract's ``applies_to`` guard.
"""

from __future__ import annotations

from typing import Any, ClassVar

from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    PreEmissionInputs,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    DeclaredRequiredInputFieldsPayload,
    DeclaredRequiredInputFieldsViolation,
    FrameworkBugError,
    OrchestrationInvariantError,
)
from elspeth.contracts.plugin_roles import require_declared_input_fields_plugin
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


def verify_declared_required_fields(
    *,
    declared_input_fields: frozenset[str],
    effective_input_fields: frozenset[str],
    plugin_name: str,
    node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify the input row satisfies the transform's declared prerequisites."""
    missing = declared_input_fields - effective_input_fields
    if not missing:
        return

    raise DeclaredRequiredInputFieldsViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "declared": sorted(declared_input_fields),
            "effective_input_fields": sorted(effective_input_fields),
            "missing": sorted(missing),
        },
        message=(
            f"Transform {plugin_name!r} (node {node_id!r}) declared required "
            f"input fields {sorted(declared_input_fields)!r} but row {row_id!r} "
            f"only exposed {sorted(effective_input_fields)!r}; missing "
            f"{sorted(missing)!r}."
        ),
    )


class DeclaredRequiredFieldsContract(DeclarationContract):
    """ADR-013 adopter for transform ``declared_input_fields`` declarations."""

    name: ClassVar[str] = "declared_required_fields"
    payload_schema: ClassVar[type] = DeclaredRequiredInputFieldsPayload

    def applies_to(self, plugin: Any) -> bool:
        typed_plugin = require_declared_input_fields_plugin(plugin)
        declared_input_fields = typed_plugin.declared_input_fields
        if not declared_input_fields:
            return False
        if typed_plugin.is_batch_aware:
            raise FrameworkBugError(
                f"Transform {typed_plugin.name!r} declares declared_input_fields "
                f"{sorted(declared_input_fields)!r} but is batch-aware. No "
                f"batch-pre-execution dispatch site exists; ADR-013 scopes this "
                f"contract to non-batch transforms until an ADR-010 amendment lands."
            )
        return True

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
        plugin = require_declared_input_fields_plugin(inputs.plugin)
        transform_node_id = plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {plugin.name!r} has no node_id set at declared-required-fields check time.")
        verify_declared_required_fields(
            declared_input_fields=plugin.declared_input_fields,
            effective_input_fields=inputs.effective_input_fields,
            plugin_name=plugin.name,
            node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        class _MinimalTransform:
            name = "NegativeDeclaredRequiredFieldsExample"
            node_id = "declared-required-neg-1"
            declared_input_fields: frozenset[str] = frozenset({"customer_id", "account_id"})
            is_batch_aware = False

        inputs = PreEmissionInputs(
            plugin=_MinimalTransform(),
            node_id="declared-required-neg-1",
            run_id="declared-required-neg-run",
            row_id="declared-required-neg-row",
            token_id="declared-required-neg-token",
            input_row=_build_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )
        return ExampleBundle(site=DispatchSite.PRE_EMISSION, args=(inputs,))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        class _NonApplyingTransform:
            name = "NonApplyingDeclaredRequiredFieldsExample"
            node_id = "declared-required-non-fire-1"
            declared_input_fields: frozenset[str] = frozenset()
            is_batch_aware = False

        inputs = PreEmissionInputs(
            plugin=_NonApplyingTransform(),
            node_id="declared-required-non-fire-1",
            run_id="declared-required-non-fire-run",
            row_id="declared-required-non-fire-row",
            token_id="declared-required-non-fire-token",
            input_row=_build_row(("customer_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"customer_id"}),
        )
        return ExampleBundle(site=DispatchSite.PRE_EMISSION, args=(inputs,))


register_declaration_contract(DeclaredRequiredFieldsContract())
