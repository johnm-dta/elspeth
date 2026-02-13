# tests/unit/core/test_explicit_sink_routing_safeguards.py
"""Safeguard tests for the explicit sink routing (on_success) feature.

ADR: docs/architecture/adr/004-adr-explicit-sink-routing.md

These tests prevent regressions of the default_sink removal and verify
the on_success wiring chain remains connected end-to-end.

Safeguards:
- 8m9d: Automated zero-default_sink grep verification
- vd7j: on_success config alignment test (end-to-end wiring)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest

from elspeth.contracts import RunStatus
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.config import SourceSettings, TransformSettings
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "elspeth"


# ---------------------------------------------------------------------------
# 8m9d: Automated zero-default_sink grep verification
# ---------------------------------------------------------------------------

# Patterns that are ALLOWED to reference "default_sink" in src/elspeth/
# These are rejection validators and comments explaining the removal.
ALLOWED_DEFAULT_SINK_PATTERNS = [
    # config.py: rejection validators that tell users to migrate
    re.compile(r"reject_default_sink"),
    re.compile(r"'default_sink' has been removed"),
    re.compile(r'"default_sink" in'),
    re.compile(r"default_sink.*migration|remove.*default_sink.*line", re.IGNORECASE),
    re.compile(r'""".*default_sink'),  # docstrings referencing the concept
    # Comments/docstrings explaining the removal
    re.compile(r"#.*default_sink"),
    re.compile(r"no default_sink fallback"),
    re.compile(r"old default_sink"),
    re.compile(r"replaces.*default_sink"),
    re.compile(r"default_sink_name"),  # parameter name in docstrings
]


class TestZeroDefaultSinkGrep:
    """Automated verification that default_sink has been fully removed from source code.

    Any reference to default_sink in src/elspeth/ must be either:
    (a) a rejection validator in config.py that tells users to migrate, or
    (b) a docstring/comment explaining the removal.

    This prevents accidental re-introduction of the concept.
    """

    def test_no_operational_default_sink_references(self) -> None:
        """All default_sink references in src/ must be rejection validators or comments."""
        violations: list[str] = []

        for py_file in sorted(SRC_ROOT.rglob("*.py")):
            rel_path = py_file.relative_to(SRC_ROOT.parent.parent)
            for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
                if "default_sink" not in line:
                    continue

                # Check if this line matches an allowed pattern
                if any(pat.search(line) for pat in ALLOWED_DEFAULT_SINK_PATTERNS):
                    continue

                violations.append(f"{rel_path}:{lineno}: {line.strip()}")

        if violations:
            msg = (
                f"Found {len(violations)} operational default_sink reference(s) in src/.\n"
                "default_sink has been removed — use on_success routing instead.\n"
                "If this is a rejection validator or explanatory comment, "
                "add the pattern to ALLOWED_DEFAULT_SINK_PATTERNS.\n\n" + "\n".join(violations)
            )
            pytest.fail(msg)


# ---------------------------------------------------------------------------
# vd7j: on_success config alignment test (end-to-end wiring)
# ---------------------------------------------------------------------------


class _OnSuccessTracingTransform(BaseTransform):
    """Transform used to verify on_success wiring end-to-end."""

    name = "on_success_tracer"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> Any:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(
            make_pipeline_row(row.to_dict()),
            success_reason={"action": "trace"},
        )


class TestOnSuccessConfigAlignment:
    """Verify on_success flows end-to-end from config to RowResult.

    The wiring chain:
    1. TransformDataConfig.on_success (Pydantic validation)
    2. BaseTransform.on_success (set from config or test helper)
    3. TransformProtocol.on_success property (read by DAG and processor)
    4. _validate_on_success_routing() in dag.py (DAG construction)
    5. RowProcessor tracks last_on_success_sink (runtime routing)
    6. RowResult(sink_name=effective_sink) (terminal outcome)

    If any link in this chain breaks, rows would fail at
    RowResult.__post_init__ with "COMPLETED outcome requires sink_name".
    """

    def test_on_success_from_transform_reaches_row_result(self, payload_store) -> None:
        """Terminal transform on_success flows through to RowResult sink_name."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 42}], on_success="target_sink")
        transform = _OnSuccessTracingTransform()
        transform.on_success = "target_sink"
        sink = CollectSink(name="target_sink")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"target_sink": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1
        assert sink.results[0]["value"] == 42

    def test_on_success_from_source_reaches_row_result_no_transforms(self, payload_store) -> None:
        """Source on_success flows through to RowResult when no transforms exist."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 99}], on_success="direct_sink")
        sink = CollectSink(name="direct_sink")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"direct_sink": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1

    def test_on_success_attribute_readable_and_writable(self) -> None:
        """TransformProtocol.on_success is a plain attribute."""
        transform = _OnSuccessTracingTransform()
        assert transform.on_success is None

        transform.on_success = "my_sink"
        assert transform.on_success == "my_sink"

    def test_config_on_success_parsed_into_transform(self) -> None:
        """TransformSettings.on_success is accepted by Pydantic and validated.

        Note: on_success was lifted from TransformDataConfig (options layer)
        to TransformSettings (settings level) as part of Phase 3 DAG wiring.
        """
        cfg = TransformSettings(
            name="test_transform",
            plugin="passthrough",
            input="source_out",
            on_success="output_sink",
            on_error="discard",
        )
        assert cfg.on_success == "output_sink"

    def test_config_on_success_rejects_empty_string(self) -> None:
        """Empty on_success string is rejected by validator."""
        with pytest.raises(ValueError, match="on_success must be a connection name"):
            TransformSettings(
                name="test_transform",
                plugin="passthrough",
                input="source_out",
                on_success="   ",
                on_error="discard",
            )

    def test_pydantic_rejects_missing_on_success(self) -> None:
        """TransformSettings requires on_success — Pydantic rejects omission."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransformSettings(
                name="tracer_0",
                plugin="on_success_tracer",
                input="source_out",
                on_error="discard",
            )

    def test_multiple_sinks_routes_to_correct_one(self, payload_store) -> None:
        """With multiple sinks, rows route to the on_success-declared sink."""
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 1}], on_success="source_out")
        transform = _OnSuccessTracingTransform()
        transform.on_success = "sink_b"
        sink_a = CollectSink(name="sink_a")
        sink_b = CollectSink(name="sink_b")

        source_settings = SourceSettings(plugin=source.name, on_success="source_out", options={})
        wired = wire_transforms(
            [cast(TransformProtocol, transform)],
            source_connection="source_out",
            final_sink="sink_b",
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=cast(SourceProtocol, source),
            source_settings=source_settings,
            transforms=wired,
            sinks=cast("dict[str, SinkProtocol]", {"sink_a": sink_a, "sink_b": sink_b}),
            aggregations={},
            gates=[],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert len(sink_a.results) == 0
        assert len(sink_b.results) == 1
