"""SchemaConfigModeContract — ADR-014 behaviour and dispatcher coverage."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.data import PluginSchema as _PermissiveSchema
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    contract_sites,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    SchemaConfigModeViolation,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_dispatch import (
    run_post_emission_checks,
)
from elspeth.engine.executors.pass_through import PassThroughDeclarationContract
from elspeth.engine.executors.schema_config_mode import SchemaConfigModeContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig


def _contract(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
    python_type: type | dict[str, type] = str,
    required: bool | dict[str, bool] = True,
    nullable: bool | dict[str, bool] = False,
) -> SchemaContract:
    return SchemaContract(
        mode=mode,  # type: ignore[arg-type]  # test helper uses closed-set literals
        fields=tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=python_type[name] if isinstance(python_type, dict) else python_type,
                required=required[name] if isinstance(required, dict) else required,
                source="inferred",
                nullable=nullable[name] if isinstance(nullable, dict) else nullable,
            )
            for name in fields
        ),
        locked=locked,
    )


def _row(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
    python_type: type | dict[str, type] = str,
    required: bool | dict[str, bool] = True,
    nullable: bool | dict[str, bool] = False,
) -> PipelineRow:
    return PipelineRow(
        dict.fromkeys(fields, "v"),
        _contract(
            fields,
            mode=mode,
            locked=locked,
            python_type=python_type,
            required=required,
            nullable=nullable,
        ),
    )


def _schema_config(
    *,
    mode: str,
    fields: tuple[str, ...] | None = None,
    guaranteed_fields: tuple[str, ...] = (),
) -> SchemaConfig:
    config: dict[str, object] = {"mode": mode}
    if fields is not None:
        config["fields"] = [{"name": name, "type": "str", "required": True, "nullable": False} for name in fields]
    if guaranteed_fields:
        config["guaranteed_fields"] = list(guaranteed_fields)
    return SchemaConfig.from_dict(config)


def _plugin(
    *,
    name: str = "SchemaConfigModeTransform",
    node_id: str | None = "schema-config-mode-node",
    output_schema_config: SchemaConfig | None = None,
    passes_through_input: bool = False,
    can_drop_rows: bool = False,
    is_batch_aware: bool = False,
) -> Any:
    plugin = type("SchemaConfigModePlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin._output_schema_config = output_schema_config
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_output_fields = frozenset()
    plugin.declared_input_fields = frozenset()
    plugin.is_batch_aware = is_batch_aware
    return plugin


class _DummyTransformConfig(TransformDataConfig):
    pass


class _AligningTransform(BaseTransform):
    name = "aligning_schema_mode_transform"
    config_model = _DummyTransformConfig
    input_schema = _PermissiveSchema
    output_schema = _PermissiveSchema

    def __init__(self) -> None:
        super().__init__({})
        self._output_schema_config = _schema_config(mode="fixed", fields=("source",))

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        raise NotImplementedError


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attribute() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(output_schema_config=_schema_config(mode="observed"))
    assert contract.applies_to(plugin) is True
    plugin._output_schema_config = None
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_crashes() -> None:
    contract = SchemaConfigModeContract()

    class _NoAttr:
        pass

    with pytest.raises(AttributeError):
        contract.applies_to(_NoAttr())


def test_contract_claims_both_dispatch_sites() -> None:
    assert contract_sites(SchemaConfigModeContract()) == frozenset({"post_emission_check", "batch_flush_check"})


def test_post_emission_check_raises_on_mode_mismatch() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(output_schema_config=_schema_config(mode="fixed", fields=("source",)))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",), mode="FIXED"),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    assert exc_info.value.payload == {
        "declared_mode": "fixed",
        "observed_mode": "observed",
        "declared_locked": True,
        "observed_locked": True,
    }


def test_post_emission_check_raises_on_fixed_mode_undeclared_extra_fields() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(output_schema_config=_schema_config(mode="fixed", fields=("source",)))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",), mode="FIXED"),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "extra"), mode="FIXED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    assert tuple(exc_info.value.payload["undeclared_extra_fields"]) == ("extra",)
    assert exc_info.value.payload["declared_mode"] == "fixed"
    assert exc_info.value.payload["observed_mode"] == "fixed"


def test_post_emission_check_returns_none_for_valid_observed_mode() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(output_schema_config=_schema_config(mode="observed"))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",), mode="OBSERVED"),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "dynamic"), mode="OBSERVED"),))

    assert contract.post_emission_check(inputs, outputs) is None


def test_batch_flush_check_raises_on_fixed_mode_undeclared_extra_fields() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(
        output_schema_config=_schema_config(mode="fixed", fields=("source",)),
        is_batch_aware=True,
    )
    token_row = _row(("source",), mode="FIXED")
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("source", "extra"), mode="FIXED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.batch_flush_check(inputs, outputs)

    assert tuple(exc_info.value.payload["undeclared_extra_fields"]) == ("extra",)


def test_post_emission_check_raises_on_observed_mode_missing_guaranteed_field() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(output_schema_config=_schema_config(mode="observed", guaranteed_fields=("source", "guaranteed_missing")))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",), mode="OBSERVED"),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    assert tuple(exc_info.value.payload["runtime_observed_fields"]) == ("source",)
    assert tuple(exc_info.value.payload["missing_required_fields"]) == ("guaranteed_missing",)


def test_batch_flush_check_raises_on_fixed_mode_missing_required_declared_field() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(
        output_schema_config=SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": [
                    {"name": "source", "type": "str", "required": True, "nullable": False},
                    {"name": "required_out", "type": "str", "required": True, "nullable": False},
                ],
            }
        ),
        is_batch_aware=True,
    )
    token_row = _row(("source",), mode="FIXED")
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("source",), mode="FIXED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.batch_flush_check(inputs, outputs)

    assert tuple(exc_info.value.payload["runtime_observed_fields"]) == ("source",)
    assert tuple(exc_info.value.payload["missing_required_fields"]) == ("required_out",)


def test_post_emission_check_raises_on_declared_field_metadata_mismatch() -> None:
    contract = SchemaConfigModeContract()
    plugin = _plugin(
        output_schema_config=SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": [{"name": "score", "type": "int", "required": True, "nullable": False}],
            }
        )
    )
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("score",), mode="FIXED"),
        static_contract=frozenset({"score"}),
        effective_input_fields=frozenset({"score"}),
    )
    outputs = PostEmissionOutputs(
        emitted_rows=(
            _row(
                ("score",),
                mode="FIXED",
                python_type={"score": str},
                required={"score": False},
                nullable={"score": True},
            ),
        )
    )

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    assert len(exc_info.value.payload["field_metadata_mismatches"]) == 1
    mismatch = exc_info.value.payload["field_metadata_mismatches"][0]
    assert mismatch["field"] == "score"
    assert mismatch["expected_type"] == "int"
    assert mismatch["observed_type"] == "str"
    assert mismatch["expected_required"] is True
    assert mismatch["observed_required"] is False
    assert mismatch["expected_nullable"] is False
    assert mismatch["observed_nullable"] is True
    assert mismatch["observed_present"] is True


def test_dispatcher_raises_single_violation_for_schema_config_mode_only(_isolated_registry) -> None:
    register_declaration_contract(SchemaConfigModeContract())
    plugin = _plugin(output_schema_config=_schema_config(mode="fixed", fields=("source",)))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-single",
        row_id="row-single",
        token_id="token-single",
        input_row=_row(("source",), mode="FIXED"),
        static_contract=frozenset({"source"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

    with pytest.raises(SchemaConfigModeViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=outputs)

    assert exc_info.value.contract_name == "schema_config_mode"


def test_dispatcher_aggregates_with_pass_through(_isolated_registry) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(SchemaConfigModeContract())

    plugin = _plugin(
        output_schema_config=_schema_config(mode="fixed", fields=("source", "carry")),
        passes_through_input=True,
    )
    input_row = _row(("source", "carry"), mode="FIXED")
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-aggregate",
        row_id="row-aggregate",
        token_id="token-aggregate",
        input_row=input_row,
        static_contract=frozenset({"source", "carry"}),
        effective_input_fields=frozenset({"source", "carry"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=outputs)

    child_types = {type(child).__name__ for child in exc_info.value.violations}
    assert child_types == {"PassThroughContractViolation", "SchemaConfigModeViolation"}

    schema_child = next(child for child in exc_info.value.violations if isinstance(child, SchemaConfigModeViolation))
    assert schema_child.contract_name == "schema_config_mode"
    assert schema_child.payload["observed_mode"] == "observed"


def test_base_transform_alignment_normalizes_mode_and_lock_state() -> None:
    transform = _AligningTransform()
    observed_contract = _contract(("source",), mode="OBSERVED", locked=False)

    aligned = transform._align_output_contract(observed_contract)

    assert aligned.mode == "FIXED"
    assert aligned.locked is True
    assert tuple(fc.normalized_name for fc in aligned.fields) == ("source",)


def test_base_transform_alignment_wraps_pipeline_row_with_declared_mode() -> None:
    transform = _AligningTransform()
    row = _row(("source",), mode="OBSERVED", locked=False)

    aligned_row = transform._align_output_row_contract(row)

    assert aligned_row.contract is not None
    assert aligned_row.contract.mode == "FIXED"
    assert aligned_row.contract.locked is True
