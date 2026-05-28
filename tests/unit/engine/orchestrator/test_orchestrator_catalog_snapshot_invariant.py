# tests/unit/engine/orchestrator/test_orchestrator_catalog_snapshot_invariant.py
"""Pin the offensive guard for the OpenRouter catalog snapshot kwargs.

``Orchestrator.run`` raises ``OrchestrationInvariantError`` when either
``openrouter_catalog_sha256`` or ``openrouter_catalog_source`` is
``None`` — the L3 entry point (web lifespan, CLI bootstrap) is the
canonical resolver of these values, and arriving at L2 without them is
a wiring bug, not a runtime condition to tolerate.

The autouse conftest fixture
``_freeze_runtime_val_registries_before_begin_run`` defaults both
kwargs to a synthetic snapshot (``"0" * 64`` / ``"bundled"``) so that
the bulk of unit tests need not thread the value through every call
site. This file's tests bypass that default by passing ``None``
explicitly — ``dict.setdefault`` is a no-op when the key is already
present in ``kwargs``.
"""

from __future__ import annotations

import threading
from typing import cast

import pytest

from elspeth.contracts import SinkProtocol, SourceProtocol
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore


def _build_minimal_pipeline() -> tuple[PipelineConfig, ExecutionGraph]:
    """Construct a trivial source → sink pipeline.

    The orchestrator's snapshot guard fires before any source rows are
    pulled, so we never actually run this pipeline — we just need a
    valid ``ExecutionGraph`` to pass the earlier ``graph is None`` and
    ``payload_store is None`` invariant checks.
    """
    source = ListSource(data=[{"k": "v"}])
    sink = CollectSink("default")
    source_settings = SourceSettings(
        plugin=source.name,
        on_success="default",
        options={},
    )
    config = PipelineConfig(
        sources={"source": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(sink)},
    )
    graph = ExecutionGraph.from_plugin_instances(
        sources={"source": cast(SourceProtocol, source)},
        source_settings_map={"source": source_settings},
        transforms=[],
        sinks=cast("dict[str, SinkProtocol]", {"default": sink}),
        aggregations={},
        gates=[],
    )
    return config, graph


class TestCatalogSnapshotInvariant:
    """Offensive guards for the OpenRouter catalog snapshot kwargs."""

    def test_run_raises_when_sha256_is_none(self) -> None:
        """``openrouter_catalog_sha256=None`` crashes loudly with the named field."""
        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db)
        config, graph = _build_minimal_pipeline()

        with pytest.raises(OrchestrationInvariantError, match="openrouter_catalog_sha256"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=MockPayloadStore(),
                shutdown_event=threading.Event(),
                openrouter_catalog_sha256=None,
                openrouter_catalog_source="bundled",
            )

    def test_run_raises_when_source_is_none(self) -> None:
        """``openrouter_catalog_source=None`` also fails the same guard."""
        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db)
        config, graph = _build_minimal_pipeline()

        with pytest.raises(OrchestrationInvariantError, match="openrouter_catalog_sha256"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=MockPayloadStore(),
                shutdown_event=threading.Event(),
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source=None,
            )

    def test_run_raises_when_both_are_none(self) -> None:
        """Both ``None`` — the production-typical wiring-bug case."""
        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db)
        config, graph = _build_minimal_pipeline()

        with pytest.raises(OrchestrationInvariantError, match="openrouter_catalog_sha256"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=MockPayloadStore(),
                shutdown_event=threading.Event(),
                openrouter_catalog_sha256=None,
                openrouter_catalog_source=None,
            )
