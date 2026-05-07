from __future__ import annotations

from typing import Any

from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context
from tests.invariants.test_pass_through_invariants import _probe_instantiate


class _SingleRowEcho(BaseTransform):
    name = "single_row_echo"
    input_schema = None
    output_schema = None

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "echo"})

    def close(self) -> None:
        pass


class _BatchEcho(BaseTransform):
    name = "batch_echo"
    input_schema = None
    output_schema = None
    is_batch_aware = True

    def process(self, rows: list[PipelineRow], ctx: Any) -> TransformResult:  # type: ignore[override]
        return TransformResult.success(rows[0], success_reason={"action": "echo"})

    def close(self) -> None:
        pass


class _CustomExecution(BaseTransform):
    name = "custom_execution"
    input_schema = None
    output_schema = None

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        raise AssertionError("default process() path must not be used")

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        return TransformResult.success(probe_rows[0], success_reason={"action": "custom"})

    def close(self) -> None:
        pass


class _PositionalOnlyInit(BaseTransform):
    name = "positional_only_init"
    input_schema = None
    output_schema = None
    passes_through_input = True

    def __init__(self, config: dict[str, Any], /) -> None:
        super().__init__(config)

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {"schema": {"mode": "observed"}}

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "echo"})

    def close(self) -> None:
        pass


def test_default_forward_execution_calls_process_for_single_row() -> None:
    transform = _SingleRowEcho(config={})
    row = make_pipeline_row({"baseline": "kept"})
    result = transform.execute_forward_invariant_probe([row], make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"


def test_default_forward_execution_calls_process_for_batch_aware() -> None:
    transform = _BatchEcho(config={})
    rows = [make_pipeline_row({"baseline": "kept"})]
    result = transform.execute_forward_invariant_probe(rows, make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"


def test_custom_forward_execution_can_bypass_default_process_path() -> None:
    transform = _CustomExecution(config={})
    row = make_pipeline_row({"baseline": "kept"})
    result = transform.execute_forward_invariant_probe([row], make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"


def test_probe_instantiation_uses_runtime_positional_constructor_contract() -> None:
    transform = _probe_instantiate(_PositionalOnlyInit)
    assert isinstance(transform, _PositionalOnlyInit)
