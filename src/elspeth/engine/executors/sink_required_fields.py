"""Runtime verification of sink required-field declarations (ADR-017).

This contract registers for ONE dispatch site:

    * ``boundary_check`` — sink-side row boundary before schema validation and
      before external sink I/O.
"""

from __future__ import annotations

import hashlib
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

_MAX_RUNTIME_OBSERVED_FIELDS = 20
_MAX_RUNTIME_OBSERVED_FIELD_DISPLAY_CHARS = 64
_MAX_RUNTIME_OBSERVED_OMITTED_HASHES = 20
_FIELD_NAME_HASH_CHARS = 16


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


def _field_name_hash(field_name: str) -> str:
    return hashlib.sha256(field_name.encode("utf-8", errors="surrogatepass")).hexdigest()[:_FIELD_NAME_HASH_CHARS]


def _bounded_field_name(field_name: str) -> str:
    if len(field_name) <= _MAX_RUNTIME_OBSERVED_FIELD_DISPLAY_CHARS:
        return field_name
    suffix = f"...#{_field_name_hash(field_name)}"
    prefix_len = _MAX_RUNTIME_OBSERVED_FIELD_DISPLAY_CHARS - len(suffix)
    return f"{field_name[:prefix_len]}{suffix}"


def _bounded_runtime_observed_payload(runtime_observed: frozenset[str]) -> dict[str, object]:
    sorted_observed = sorted(runtime_observed)
    sample_names = sorted_observed[:_MAX_RUNTIME_OBSERVED_FIELDS]
    omitted_names = sorted_observed[_MAX_RUNTIME_OBSERVED_FIELDS:]
    payload: dict[str, object] = {
        "runtime_observed": [_bounded_field_name(name) for name in sample_names],
        "runtime_observed_count": len(sorted_observed),
        "runtime_observed_truncated": bool(omitted_names),
    }
    if omitted_names:
        omitted_hashes = [_field_name_hash(name) for name in omitted_names[:_MAX_RUNTIME_OBSERVED_OMITTED_HASHES]]
        payload.update(
            {
                "runtime_observed_omitted_count": len(omitted_names),
                "runtime_observed_omitted_hashes": omitted_hashes,
                "runtime_observed_omitted_hashes_truncated": len(omitted_names) > len(omitted_hashes),
            }
        )
    return payload


def _runtime_observed_message(observed_payload: Mapping[str, object]) -> str:
    observed_sample = observed_payload["runtime_observed"]
    observed_count = observed_payload["runtime_observed_count"]
    if observed_payload["runtime_observed_truncated"]:
        omitted_count = observed_payload["runtime_observed_omitted_count"]
        omitted_hashes = observed_payload["runtime_observed_omitted_hashes"]
        hashes_truncated = observed_payload["runtime_observed_omitted_hashes_truncated"]
        hash_note = "truncated " if hashes_truncated else ""
        return (
            f"{observed_count} runtime field(s), sampled {observed_sample!r}; "
            f"omitted {omitted_count} field name(s), {hash_note}omitted hashes {omitted_hashes!r}"
        )
    return f"{observed_count} runtime field(s), sampled {observed_sample!r}"


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

    observed_payload = _bounded_runtime_observed_payload(runtime_observed)
    raise SinkRequiredFieldsViolation(
        plugin=plugin_name,
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        payload={
            "declared": sorted(declared_required_fields),
            **observed_payload,
            "missing": sorted(missing),
        },
        message=(
            f"Sink {plugin_name!r} (node {node_id!r}) declared required fields "
            f"{sorted(declared_required_fields)!r} but row {row_id!r} only exposed "
            f"{_runtime_observed_message(observed_payload)}; missing {sorted(missing)!r}.{contract_context}"
        ),
    )


class SinkRequiredFieldsContract(DeclarationContract):
    """ADR-017 adopter for sink ``declared_required_fields``."""

    name: ClassVar[str] = "sink_required_fields"
    payload_schema: ClassVar[type] = SinkRequiredFieldsPayload
    violation_class: ClassVar[type[SinkRequiredFieldsViolation]] = SinkRequiredFieldsViolation

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
        if inputs.node_id != sink_node_id:
            raise OrchestrationInvariantError(
                f"Sink {inputs.plugin.name!r} node_id drift at sink-required-fields boundary check time: "
                f"dispatcher passed {inputs.node_id!r}, plugin has {sink_node_id!r}."
            )
        verify_sink_required_fields(
            declared_required_fields=declared_required_fields,
            row_data=cast(Mapping[str, object], inputs.row_data),
            row_contract=cast(SchemaContract | None, inputs.row_contract),
            plugin_name=inputs.plugin.name,
            node_id=sink_node_id,
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
