"""Production-path aggregate declaration-contract coverage.

Exercises ``ExecutionGraph.from_plugin_instances()`` -> ``Orchestrator.run()``
for a single transform that violates two applicable declaration contracts on
the same row. This guards the integration path, not just the direct
``run_post_emission_checks(...)`` helper.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.declaration_contracts import AggregateDeclarationContractViolation
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline


def _contract_for(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=tuple(
            FieldContract(
                normalized_name=field,
                original_name=field,
                python_type=object,
                required=True,
                source="inferred",
                nullable=False,
            )
            for field in fields
        ),
        locked=True,
    )


class _TwoContractViolatingTransform(BaseTransform):
    """Misdeclares both pass-through preservation and output-field emission."""

    name = "two-contract-violating-transform"
    input_schema = _TestSchema
    output_schema = _TestSchema
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    passes_through_input = True
    declared_output_fields = frozenset({"new_a", "new_b"})
    on_success = "default"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._output_schema_config = self._build_output_schema_config(
            SchemaConfig(
                mode="observed",
                fields=None,
                guaranteed_fields=("new_a", "new_b"),
            )
        )

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        data.pop("carry", None)
        data["new_a"] = "present"
        return TransformResult.success(
            PipelineRow(data, _contract_for(tuple(data.keys()))),
            success_reason={"action": "violate_two_declarations"},
        )


@pytest.mark.integration
def test_aggregate_declaration_contract_violation_reaches_orchestrator(payload_store) -> None:
    transform = _TwoContractViolatingTransform()
    source, transforms, sinks, graph = build_linear_pipeline(
        [{"source": "v", "carry": "must-remain"}],
        transforms=[transform],
    )
    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in transforms],
        sinks={"default": as_sink(sinks["default"])},
    )

    orchestrator = Orchestrator(LandscapeDB("sqlite:///:memory:"))
    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        orchestrator.run(config, graph=graph, payload_store=payload_store)

    aggregate = exc_info.value
    child_types = {type(child).__name__ for child in aggregate.violations}
    assert {"DeclaredOutputFieldsViolation", "PassThroughContractViolation"} <= child_types
    assert len(child_types) >= 2
    assert sinks["default"].results == []
