# tests/integration/test_deaggregation.py
"""Integration tests for deaggregation (JSONExplode) pipeline.

Tests the complete multi-row output system working end-to-end:
1. Pipeline execution with JSONExplode transform
2. Audit trail verification for token expansion
3. Three-tier trust model demonstration
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


class TestDeaggregationPipeline:
    """Pipeline execution tests for deaggregation."""

    @pytest.fixture
    def input_data(self, tmp_path: Path) -> Path:
        """Create test input JSON file with orders containing items."""
        input_file = tmp_path / "input.json"
        data = [
            {
                "order_id": 1,
                "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}],
            },
            {"order_id": 2, "items": [{"sku": "C3", "qty": 5}]},
            {
                "order_id": 3,
                "items": [
                    {"sku": "A1", "qty": 1},
                    {"sku": "D4", "qty": 3},
                    {"sku": "E5", "qty": 2},
                ],
            },
        ]
        input_file.write_text(json.dumps(data))
        return input_file

    @pytest.fixture
    def pipeline_config(self, tmp_path: Path, input_data: Path) -> Path:
        """Create pipeline configuration for deaggregation."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = {
            "source": {
                "plugin": "json",
                "options": {
                    "path": str(input_data),
                    "schema": {
                        "mode": "strict",
                        "fields": ["order_id: int", "items: any"],
                    },
                    "on_validation_failure": "discard",
                },
            },
            "transforms": [
                {
                    "plugin": "json_explode",
                    "options": {
                        "array_field": "items",
                        "output_field": "item",
                        "include_index": True,
                        "schema": {"fields": "dynamic"},
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_dir / "order_items.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'audit.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_explodes_orders_into_items(self, pipeline_config: Path, tmp_path: Path) -> None:
        """3 orders with 2+1+3 items should produce 6 output rows."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--execute"])
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"
        assert "completed" in result.output.lower()

        # Check output file
        output_file = tmp_path / "output" / "order_items.json"
        assert output_file.exists(), "Output file should exist"

        data = json.loads(output_file.read_text())
        assert len(data) == 6, f"Expected 6 item rows, got {len(data)}"

        # Verify structure of output rows
        for row in data:
            assert "order_id" in row, "Each row should have order_id"
            assert "item" in row, "Each row should have item field"
            assert "item_index" in row, "Each row should have item_index"

    def test_preserves_item_order(self, pipeline_config: Path, tmp_path: Path, plugin_manager) -> None:
        """Verify item_index ordering is preserved within each order."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--execute"])
        assert result.exit_code == 0

        output_file = tmp_path / "output" / "order_items.json"
        data = json.loads(output_file.read_text())

        # Group by order_id
        orders: dict[int, list[dict[str, Any]]] = {}
        for row in data:
            order_id = row["order_id"]
            if order_id not in orders:
                orders[order_id] = []
            orders[order_id].append(row)

        # Verify item_index is sequential within each order
        for order_id, items in orders.items():
            # Sort by item_index to check sequence
            sorted_items = sorted(items, key=lambda x: x["item_index"])
            indices = [item["item_index"] for item in sorted_items]
            expected = list(range(len(items)))
            assert indices == expected, f"Order {order_id} has non-sequential indices: {indices}"


class TestDeaggregationAuditTrail:
    """Audit trail verification tests for deaggregation."""

    @pytest.fixture
    def input_data(self, tmp_path: Path) -> Path:
        """Create test input JSON file."""
        input_file = tmp_path / "input.json"
        data = [
            {
                "order_id": 1,
                "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}],
            },
            {"order_id": 2, "items": [{"sku": "C3", "qty": 5}]},
            {
                "order_id": 3,
                "items": [
                    {"sku": "A1", "qty": 1},
                    {"sku": "D4", "qty": 3},
                    {"sku": "E5", "qty": 2},
                ],
            },
        ]
        input_file.write_text(json.dumps(data))
        return input_file

    @pytest.fixture
    def run_pipeline(self, tmp_path: Path, input_data: Path, plugin_manager, payload_store) -> tuple[str, "LandscapeDB"]:
        """Run pipeline and return run_id and database for verification.

        Returns:
            Tuple of (run_id, LandscapeDB instance)
        """
        from elspeth.core.config import load_settings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.json_source import JSONSource
        from elspeth.plugins.transforms.json_explode import JSONExplode

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Build config programmatically for test
        config_dict = {
            "source": {
                "plugin": "json",
                "options": {
                    "path": str(input_data),
                    "schema": {
                        "mode": "strict",
                        "fields": ["order_id: int", "items: any"],
                    },
                    "on_validation_failure": "discard",
                },
            },
            "transforms": [
                {
                    "plugin": "json_explode",
                    "options": {
                        "array_field": "items",
                        "output_field": "item",
                        "include_index": True,
                        "schema": {"fields": "dynamic"},
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_dir / "order_items.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'audit.db'}"},
        }

        # Save and load config via settings loader
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config_dict))
        settings = load_settings(config_file)

        # Create database
        db = LandscapeDB.from_url(settings.landscape.url)

        # Build graph
        from elspeth.cli_helpers import instantiate_plugins_from_config

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        # Instantiate plugins
        source = JSONSource(dict(settings.source.options))
        transform = JSONExplode(dict(settings.transforms[0].options))
        sink = JSONSink(dict(settings.sinks["output"].options))

        # Build pipeline config
        from elspeth.core.config import resolve_config

        pipeline_config = PipelineConfig(
            source=source,  # type: ignore[arg-type]
            transforms=[transform],  # type: ignore[arg-type]
            sinks={"output": sink},  # type: ignore[arg-type]
            config=resolve_config(settings),
        )

        # Run pipeline
        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph, settings=settings, payload_store=payload_store)

        return (result.run_id, db)

    def test_records_token_expansion(self, run_pipeline: tuple[str, "LandscapeDB"]) -> None:
        """9 tokens created: 3 source tokens + 6 expanded tokens."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        run_id, db = run_pipeline
        recorder = LandscapeRecorder(db)

        # Get all rows for this run
        rows = recorder.get_rows(run_id)
        assert len(rows) == 3, f"Expected 3 source rows, got {len(rows)}"

        # Count all tokens across all rows
        all_tokens = []
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            all_tokens.extend(tokens)

        # 3 source tokens + 6 expanded tokens = 9 total
        assert len(all_tokens) == 9, f"Expected 9 tokens, got {len(all_tokens)}"

    def test_records_parent_relationships(self, run_pipeline: tuple[str, "LandscapeDB"]) -> None:
        """6 parent relationships in token_parents (one per expanded token)."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        run_id, db = run_pipeline
        recorder = LandscapeRecorder(db)

        # Get all rows and their tokens
        rows = recorder.get_rows(run_id)

        # Count parent relationships
        parent_count = 0
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                parents = recorder.get_token_parents(token.token_id)
                parent_count += len(parents)

        # 6 expanded tokens each have 1 parent = 6 relationships
        assert parent_count == 6, f"Expected 6 parent relationships, got {parent_count}"

    def test_expand_group_id_set(self, run_pipeline: tuple[str, "LandscapeDB"]) -> None:
        """6 tokens have expand_group_id set (the expanded tokens)."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        run_id, db = run_pipeline
        recorder = LandscapeRecorder(db)

        # Get all rows and their tokens
        rows = recorder.get_rows(run_id)

        # Count tokens with expand_group_id
        tokens_with_expand_group = 0
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                if token.expand_group_id is not None:
                    tokens_with_expand_group += 1

        # 6 expanded tokens should have expand_group_id
        assert tokens_with_expand_group == 6, f"Expected 6 tokens with expand_group_id, got {tokens_with_expand_group}"


class TestSourceSchemaValidation:
    """Three-tier trust model demonstration tests."""

    @pytest.fixture
    def valid_and_invalid_input(self, tmp_path: Path) -> Path:
        """Create input with both valid and invalid rows."""
        input_file = tmp_path / "input.json"
        data = [
            {"order_id": 1, "items": [{"sku": "A1", "qty": 2}]},  # Valid
            {"order_id": 2},  # Invalid: missing 'items' field
            {"order_id": 3, "items": [{"sku": "B2", "qty": 1}]},  # Valid
        ]
        input_file.write_text(json.dumps(data))
        return input_file

    def test_invalid_row_quarantined_at_source(self, tmp_path: Path, valid_and_invalid_input: Path) -> None:
        """Missing items field causes quarantine at source, not transform crash.

        This test demonstrates the three-tier trust model:
        - Source validates schema and quarantines invalid rows
        - Transform trusts that data has correct structure
        - Invalid row never reaches transform (would crash if it did)
        """
        from elspeth.cli import app

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = {
            "source": {
                "plugin": "json",
                "options": {
                    "path": str(valid_and_invalid_input),
                    "schema": {
                        "mode": "strict",
                        "fields": ["order_id: int", "items: any"],
                    },
                    # Use discard so invalid row is dropped, not routed
                    "on_validation_failure": "discard",
                },
            },
            "transforms": [
                {
                    "plugin": "json_explode",
                    "options": {
                        "array_field": "items",
                        "output_field": "item",
                        "include_index": True,
                        "schema": {"fields": "dynamic"},
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_dir / "order_items.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'audit.db'}"},
        }

        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))

        # Pipeline should complete successfully (invalid row discarded at source)
        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"

        # Only valid rows should be in output
        output_file = output_dir / "order_items.json"
        data = json.loads(output_file.read_text())

        # 2 valid orders with 1 item each = 2 output rows
        assert len(data) == 2, f"Expected 2 rows from valid orders, got {len(data)}"

        # Verify the invalid order (id=2) is not present
        order_ids = {row["order_id"] for row in data}
        assert 2 not in order_ids, "Invalid order (id=2) should not be in output"
        assert order_ids == {1, 3}, "Only valid orders should be in output"


# Type annotation for imports
if __name__ == "__main__":
    from elspeth.core.landscape import LandscapeDB
