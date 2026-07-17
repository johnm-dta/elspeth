"""Plugin registry and deterministic recovery transform for the DAG corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from elspeth.contracts import Determinism, PipelineRow, PluginSchema
from elspeth.contracts.errors import TransformSuccessReason
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl
from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.plugins.infrastructure.results import TransformResult


class CorpusInputSchema(PluginSchema):
    id: int
    value: int


class CorpusOutputSchema(PluginSchema):
    value: int
    count: int


class CorpusFailOnceEOFBatchTransform(BaseTransform):
    name = "dag_corpus_fail_once_eof_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = CorpusInputSchema
    output_schema = CorpusOutputSchema
    is_batch_aware = True
    on_error = "discard"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        marker = config.get("fault_marker_path")
        if not isinstance(marker, str) or not marker:
            raise ValueError("fault_marker_path must be a non-empty string")
        self._fault_marker = Path(marker)

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        del ctx
        if not isinstance(row, list):
            return TransformResult.success(
                row,
                success_reason=cast(TransformSuccessReason, {"action": "buffer"}),
            )

        self._fault_marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fault_marker.touch(exist_ok=False)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("injected DAG corpus EOF flush crash")

        total = sum(int(member["value"]) for member in row)
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="value",
                    original_name="value",
                    python_type=int,
                    required=False,
                    source="inferred",
                ),
                FieldContract(
                    normalized_name="count",
                    original_name="count",
                    python_type=int,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )
        return TransformResult.success(
            PipelineRow({"value": total, "count": len(row)}, contract),
            success_reason=cast(TransformSuccessReason, {"action": "batch_sum"}),
        )


def make_corpus_plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    manager.register(create_dynamic_hookimpl([CorpusFailOnceEOFBatchTransform], "elspeth_get_transforms"))
    return manager


def install_corpus_plugin_manager(monkeypatch: pytest.MonkeyPatch) -> PluginManager:
    manager = make_corpus_plugin_manager()
    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: manager,
    )
    return manager
