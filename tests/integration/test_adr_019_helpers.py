"""ADR-019 phase 3 integration-helper assembly tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import PipelineConfig
from tests.integration._helpers import (
    build_test_pipeline_with_discard_sink,
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
)

PipelineBundle = tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        (
            build_test_pipeline_with_discard_sink,
            {"success_row_count": 2, "discard_row_count": 1},
        ),
        (
            build_test_pipeline_with_gate_route,
            {"routed_row_count": 2, "default_flow_row_count": 1},
        ),
        (
            build_test_pipeline_with_on_error_route,
            {"on_error_routed_count": 2, "success_count": 1},
        ),
    ],
)
def test_adr019_helpers_build_through_production_instantiation(
    builder: Callable[..., PipelineBundle],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: dict[str, int],
) -> None:
    """Phase 3 helpers must use production plugin and graph assembly."""
    config, graph, db, store = builder(tmp_path, monkeypatch, **args)

    assert config.sinks
    assert graph is not None
    assert db is not None
    assert store is not None
