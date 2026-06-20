# tests/e2e/pipelines/test_json_to_json.py
"""E2E: JSON source -> transform -> JSON sink roundtrip.

Verifies real JSON file I/O through the full pipeline path,
including nested data preservation.
"""

from __future__ import annotations

import json
from pathlib import Path

from elspeth.contracts import RunStatus
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sources.json_source import JSONSource
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import PassTransform


class TestJSONToJSON:
    """JSON source -> PassTransform -> JSON sink roundtrip."""

    def test_json_source_to_json_sink_roundtrip(self, tmp_path: Path) -> None:
        """Create JSON file, run through pipeline, verify output matches input."""
        # -- Arrange: write input JSON --
        input_json = tmp_path / "input.json"
        output_json = tmp_path / "output.json"

        input_data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Carol", "score": 91},
            {"id": 4, "name": "Dave", "score": 78},
            {"id": 5, "name": "Eve", "score": 88},
        ]
        input_json.write_text(json.dumps(input_data))

        # Build real plugins
        source = JSONSource(
            {
                "path": str(input_json),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "json_out"
        source_settings = SourceSettings(plugin=source.name, on_success="json_out", options={})
        transform = PassTransform()
        wired = wire_transforms([transform], source_connection="json_out", final_sink="default")
        sink = JSONSink(
            {
                "path": str(output_json),
                "schema": {"mode": "observed"},
            }
        )

        # Build graph via production path (BUG-LINEAGE-01)
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},
            source_settings_map={"primary": source_settings},
            transforms=wired,
            sinks={"default": sink},
            aggregations={},
            gates=[],
        )

        # -- Act: run pipeline --
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert --
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5

        # Verify output JSON exists and contains correct data
        assert output_json.exists()
        output_data = json.loads(output_json.read_text())
        assert len(output_data) == 5

        for i, row in enumerate(output_data):
            assert row["id"] == input_data[i]["id"]
            assert row["name"] == input_data[i]["name"]
            assert row["score"] == input_data[i]["score"]

    def test_json_pipeline_preserves_nested_data(self, tmp_path: Path) -> None:
        """Test with nested JSON objects declared as 'any' type to verify no data loss.

        The OBSERVED schema mode infers types from the first row, and complex
        types (dict, list) are not supported in type inference. Use a FLEXIBLE
        schema with 'any' type for nested fields to allow them through.
        """
        # -- Arrange: write input JSON with nested structures --
        input_json = tmp_path / "input_nested.json"
        output_json = tmp_path / "output_nested.json"

        input_data = [
            {
                "id": 1,
                "name": "Alice",
                "address": {"city": "Portland", "state": "OR"},
                "tags": ["admin", "active"],
            },
            {
                "id": 2,
                "name": "Bob",
                "address": {"city": "Seattle", "state": "WA"},
                "tags": ["user"],
            },
            {
                "id": 3,
                "name": "Carol",
                "address": {"city": "Denver", "state": "CO"},
                "tags": ["admin", "reviewer"],
            },
            {
                "id": 4,
                "name": "Dave",
                "address": {"city": "Austin", "state": "TX"},
                "tags": [],
            },
            {
                "id": 5,
                "name": "Eve",
                "address": {"city": "Miami", "state": "FL"},
                "tags": ["moderator"],
            },
        ]
        input_json.write_text(json.dumps(input_data))

        # Use FLEXIBLE schema declaring nested fields as 'any' type
        # to avoid type inference failures on dict/list values
        source = JSONSource(
            {
                "path": str(input_json),
                "schema": {
                    "mode": "flexible",
                    "fields": [
                        "id: int",
                        "name: str",
                        "address: any",
                        "tags: any",
                    ],
                },
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "json_out"
        source_settings = SourceSettings(plugin=source.name, on_success="json_out", options={})
        transform = PassTransform()
        wired = wire_transforms([transform], source_connection="json_out", final_sink="default")
        sink = JSONSink(
            {
                "path": str(output_json),
                "schema": {"mode": "observed"},
            }
        )

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},
            source_settings_map={"primary": source_settings},
            transforms=wired,
            sinks={"default": sink},
            aggregations={},
            gates=[],
        )

        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert --
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5

        output_data = json.loads(output_json.read_text())
        assert len(output_data) == 5

        # Verify nested data is preserved exactly
        for i, row in enumerate(output_data):
            assert row["address"] == input_data[i]["address"]
            assert row["tags"] == input_data[i]["tags"]
            assert row["name"] == input_data[i]["name"]
