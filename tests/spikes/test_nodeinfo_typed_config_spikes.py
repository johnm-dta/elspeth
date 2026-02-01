"""Risk reduction spikes for NodeInfo typed config refactor.

These spikes validate assumptions before the main implementation.
Run with: pytest tests/spikes/test_nodeinfo_typed_config_spikes.py -v

Spikes:
1. Hash stability - verify None-filtering produces correct semantic hashes
2. Schema presence - verify nodes have schema after graph construction
3. Plugin config access - verify plugin init pattern is unchanged
4. JSON-safe config - verify plugin configs survive serialization
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from elspeth.core.canonical import canonical_json, stable_hash

# =============================================================================
# SPIKE 1: Hash Stability (CRITICAL)
# =============================================================================


@dataclass(frozen=True, slots=True)
class MockGateNodeConfig:
    """Mock of GateNodeConfig for hash stability testing."""

    routes: dict[str, str]
    schema: dict[str, Any]
    condition: str | None = None
    fork_to: list[str] | None = None
    plugin_config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MockTransformNodeConfig:
    """Mock of TransformNodeConfig for hash stability testing."""

    plugin_config: dict[str, Any]
    schema: dict[str, Any]
    required_input_fields: list[str] | None = None


@dataclass(frozen=True, slots=True)
class MockCoalesceNodeConfig:
    """Mock of CoalesceNodeConfig for hash stability testing."""

    branches: list[str]
    policy: str
    merge: str
    schema: dict[str, Any]
    timeout_seconds: float | None = None
    quorum_count: int | None = None
    select_branch: str | None = None


def config_to_dict_filtered(config: Any) -> dict[str, Any]:
    """Proposed config_to_dict with None filtering."""
    result = asdict(config)
    return {k: v for k, v in result.items() if v is not None}


def config_to_dict_unfiltered(config: Any) -> dict[str, Any]:
    """asdict without filtering - shows the problem."""
    return asdict(config)


class TestHashStability:
    """Verify that None-filtering produces semantically correct hashes.

    INVARIANT: None fields are non-semantic and excluded from serialization.

    This is the CORRECT semantic definition:
    - A field set to None means "not applicable for this node"
    - Non-applicable fields have no semantic meaning
    - Only meaningful (non-None) fields affect hashes

    The critical insight: asdict() includes ALL fields, even None ones.
    Filtering them out produces the semantically correct hash.
    """

    def test_none_filtering_required_for_hash_stability(self) -> None:
        """Demonstrate that unfiltered asdict() changes hashes."""
        # Current dict format (only explicit fields)
        current_config = {
            "routes": {"true": "continue", "false": "sink_a"},
            "schema": {"fields": "dynamic"},
            "condition": "row['amount'] > 100",
        }

        # Typed config with None for optional fields
        typed_config = MockGateNodeConfig(
            routes={"true": "continue", "false": "sink_a"},
            schema={"fields": "dynamic"},
            condition="row['amount'] > 100",
            fork_to=None,  # Optional, not set
            plugin_config=None,  # Optional, not set
        )

        # Unfiltered asdict CHANGES the hash (includes null fields)
        unfiltered = config_to_dict_unfiltered(typed_config)
        assert "fork_to" in unfiltered, "Unfiltered should include None fields"
        assert unfiltered["fork_to"] is None

        unfiltered_hash = stable_hash(unfiltered)
        current_hash = stable_hash(current_config)

        # This FAILS - demonstrating the problem
        assert unfiltered_hash != current_hash, "Unfiltered should differ (has extra null fields)"

    def test_filtered_asdict_matches_current_hash(self) -> None:
        """Verify filtered asdict() produces identical hash to current dict."""
        # Current dict format
        current_config = {
            "routes": {"true": "continue", "false": "sink_a"},
            "schema": {"fields": "dynamic"},
            "condition": "row['amount'] > 100",
        }

        # Typed config
        typed_config = MockGateNodeConfig(
            routes={"true": "continue", "false": "sink_a"},
            schema={"fields": "dynamic"},
            condition="row['amount'] > 100",
            fork_to=None,
            plugin_config=None,
        )

        # Filtered asdict should match
        filtered = config_to_dict_filtered(typed_config)
        assert "fork_to" not in filtered, "Filtered should exclude None fields"

        filtered_hash = stable_hash(filtered)
        current_hash = stable_hash(current_config)

        assert filtered_hash == current_hash, (
            f"Hash mismatch!\nFiltered: {json.dumps(filtered, sort_keys=True)}\nCurrent:  {json.dumps(current_config, sort_keys=True)}"
        )

    def test_hash_stability_with_nested_dicts(self) -> None:
        """Verify nested dicts (like schema) hash correctly."""
        current_config = {
            "routes": {"a": "sink_1", "b": "sink_2"},
            "schema": {
                "fields": "dynamic",
                "guaranteed_fields": ["id", "name"],
            },
        }

        typed_config = MockGateNodeConfig(
            routes={"a": "sink_1", "b": "sink_2"},
            schema={
                "fields": "dynamic",
                "guaranteed_fields": ["id", "name"],
            },
            condition=None,
            fork_to=None,
            plugin_config=None,
        )

        filtered = config_to_dict_filtered(typed_config)
        assert stable_hash(filtered) == stable_hash(current_config)

    def test_hash_stability_with_optional_fields_set(self) -> None:
        """Verify hash matches when optional fields ARE set."""
        current_config = {
            "routes": {"true": "continue"},
            "schema": {"fields": "dynamic"},
            "condition": "x > 0",
            "fork_to": ["branch_a", "branch_b"],
        }

        typed_config = MockGateNodeConfig(
            routes={"true": "continue"},
            schema={"fields": "dynamic"},
            condition="x > 0",
            fork_to=["branch_a", "branch_b"],
            plugin_config=None,  # Still None
        )

        filtered = config_to_dict_filtered(typed_config)
        assert "fork_to" in filtered  # Should be included when set
        assert stable_hash(filtered) == stable_hash(current_config)

    def test_transform_config_hash_stability(self) -> None:
        """Verify transform config hash stability."""
        current_config = {
            "plugin_config": {"model": "gpt-4", "temperature": 0.7},
            "schema": {"fields": "dynamic"},
        }

        typed_config = MockTransformNodeConfig(
            plugin_config={"model": "gpt-4", "temperature": 0.7},
            schema={"fields": "dynamic"},
            required_input_fields=None,
        )

        filtered = config_to_dict_filtered(typed_config)
        assert stable_hash(filtered) == stable_hash(current_config)

    def test_coalesce_config_hash_stability(self) -> None:
        """Verify coalesce config hash stability."""
        current_config = {
            "branches": ["path_a", "path_b"],
            "policy": "require_all",
            "merge": "union",
            "schema": {"fields": "dynamic"},
        }

        typed_config = MockCoalesceNodeConfig(
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            schema={"fields": "dynamic"},
            timeout_seconds=None,
            quorum_count=None,
            select_branch=None,
        )

        filtered = config_to_dict_filtered(typed_config)
        assert stable_hash(filtered) == stable_hash(current_config)

    def test_coalesce_config_with_optional_fields(self) -> None:
        """Verify coalesce with optional fields set."""
        current_config = {
            "branches": ["a", "b"],
            "policy": "quorum",
            "merge": "select",
            "schema": {"fields": "dynamic"},
            "quorum_count": 1,
            "select_branch": "a",
        }

        typed_config = MockCoalesceNodeConfig(
            branches=["a", "b"],
            policy="quorum",
            merge="select",
            schema={"fields": "dynamic"},
            timeout_seconds=None,  # Not set
            quorum_count=1,  # Set
            select_branch="a",  # Set
        )

        filtered = config_to_dict_filtered(typed_config)
        assert "quorum_count" in filtered
        assert "select_branch" in filtered
        assert "timeout_seconds" not in filtered
        assert stable_hash(filtered) == stable_hash(current_config)


# =============================================================================
# SPIKE 2: Schema Presence After Graph Construction
# =============================================================================


class TestSchemaPresence:
    """Verify that nodes have schema in their config after graph construction.

    This validates the assumption that we can track branch→schema mappings
    during gate construction.
    """

    def test_transform_has_schema_after_construction(self) -> None:
        """Verify transforms have schema in their config."""
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough

        source = NullSource({"schema": {"fields": "dynamic"}})
        transform = PassThrough({"schema": {"fields": "dynamic"}})
        # Sinks require schema_config for validation
        sink = JSONSink({"path": "/tmp/test.json", "schema": {"fields": "dynamic"}})

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
        )

        # Find transform node by type (node IDs are hash-based, not "transform_0")
        transform_nodes = [node_id for node_id, data in graph._graph.nodes(data=True) if data["info"].node_type == "transform"]
        assert len(transform_nodes) == 1, f"Expected 1 transform, found {len(transform_nodes)}"

        # Transform should have schema in config
        transform_node = graph.get_node_info(transform_nodes[0])
        assert "schema" in transform_node.config, "Transform missing schema in config"
        assert transform_node.config["schema"] == {"fields": "dynamic"}

    def test_config_gate_has_schema_after_construction(self) -> None:
        """Verify config-driven gates have schema from upstream."""
        from elspeth.core.config import GateSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource

        source = NullSource({"schema": {"fields": "dynamic"}})
        # Sinks require schema_config
        sink_a = JSONSink({"path": "/tmp/a.json", "schema": {"fields": "dynamic"}})
        sink_b = JSONSink({"path": "/tmp/b.json", "schema": {"fields": "dynamic"}})

        gate = GateSettings(
            name="router",
            condition="True",
            routes={"true": "sink_a", "false": "sink_b"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[],
            sinks={"sink_a": sink_a, "sink_b": sink_b},
            aggregations={},
            gates=[gate],
            default_sink="sink_a",
        )

        # Find the gate node (use internal _graph to iterate nodes)
        gate_nodes = [node_id for node_id, data in graph._graph.nodes(data=True) if data["info"].node_type == "gate"]
        assert len(gate_nodes) == 1, f"Expected 1 gate, found {len(gate_nodes)}"

        gate_node = graph.get_node_info(gate_nodes[0])
        assert "schema" in gate_node.config, "Gate missing schema in config"
        assert "routes" in gate_node.config, "Gate missing routes in config"


# =============================================================================
# SPIKE 3: Plugin Config Access Pattern
# =============================================================================


class TestPluginConfigAccess:
    """Verify plugin initialization pattern is unchanged.

    Key insight: Plugins receive config dict at instantiation time,
    before NodeInfo exists. They store it as self.config and read
    from that dict directly. This is separate from NodeInfo.config.
    """

    def test_plugin_receives_config_dict(self) -> None:
        """Verify plugins receive and store config dict."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        config = {"schema": {"fields": "dynamic"}, "validate_input": True}
        plugin = PassThrough(config)

        # Plugin stores config as dict
        assert hasattr(plugin, "config")
        assert isinstance(plugin.config, dict)
        assert plugin.config["validate_input"] is True

    def test_plugin_config_separate_from_nodeinfo(self) -> None:
        """Verify plugin.config and NodeInfo.config are separate."""
        from elspeth.core.dag import ExecutionGraph
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough

        source = NullSource({"schema": {"fields": "dynamic"}})
        transform = PassThrough({"schema": {"fields": "dynamic"}, "validate_input": True})
        sink = JSONSink({"path": "/tmp/test.json", "schema": {"fields": "dynamic"}})

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            aggregations={},
            gates=[],
            default_sink="output",
        )

        # Plugin still has its own config
        assert transform.config["validate_input"] is True

        # Find transform node by type (node IDs are hash-based)
        transform_nodes = [node_id for node_id, data in graph._graph.nodes(data=True) if data["info"].node_type == "transform"]
        assert len(transform_nodes) == 1

        # NodeInfo.config is a COPY, with framework fields
        node_info = graph.get_node_info(transform_nodes[0])
        assert "schema" in node_info.config

        # They reference different dicts (modification doesn't propagate)
        original_node_config = dict(node_info.config)
        transform.config["new_field"] = "new_value"
        assert node_info.config == original_node_config  # Unchanged

    def test_sink_config_path_access(self) -> None:
        """Verify sink.config['path'] access pattern works.

        This is the pattern at orchestrator.py:1740.
        It accesses the plugin's config, not NodeInfo.config.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        # CSVSink requires strict schema (fixed columns, no dynamic)
        config = {"path": "/tmp/output.csv", "schema": {"fields": ["data: str"], "mode": "strict"}}
        sink = CSVSink(config)

        # Direct access works
        assert sink.config["path"] == "/tmp/output.csv"


# =============================================================================
# SPIKE 4: JSON-Safe Config Serialization
# =============================================================================


class TestJsonSafeConfig:
    """Verify plugin configs survive serialization.

    Plugin configs pass through asdict() and canonical_json().
    Non-JSON-safe values will fail here, not at runtime.

    IMPORTANT: Uses REAL plugin instances to catch non-JSON or heavy
    objects that wouldn't be caught with synthetic test data.

    This is a go/no-go gate for the typed config refactor.
    """

    def test_real_source_configs_json_safe(self) -> None:
        """Verify real source plugin configs serialize correctly."""
        from elspeth.plugins.sources.csv_source import CSVSource
        from elspeth.plugins.sources.json_source import JSONSource
        from elspeth.plugins.sources.null_source import NullSource

        # Create real plugin instances with correct required fields
        sources = [
            CSVSource(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "discard",
                }
            ),
            JSONSource(
                {
                    "path": "/tmp/test.json",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "discard",
                }
            ),
            NullSource({"schema": {"fields": "dynamic"}, "rows": [{"id": 1}]}),
        ]

        for source in sources:
            # Use the actual plugin config (what would go into NodeInfo)
            typed = MockTransformNodeConfig(
                plugin_config=dict(source.config),
                schema={"fields": "dynamic"},
            )
            serialized = config_to_dict_filtered(typed)
            # Must survive canonical_json (the actual hot path)
            result = canonical_json(serialized)
            assert isinstance(result, str), f"Failed for {source.name}"

    def test_real_transform_configs_json_safe(self) -> None:
        """Verify real transform plugin configs serialize correctly."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper
        from elspeth.plugins.transforms.passthrough import PassThrough
        from elspeth.plugins.transforms.truncate import Truncate

        # Create real plugin instances with correct field names
        transforms = [
            PassThrough({"schema": {"fields": "dynamic"}, "validate_input": True}),
            FieldMapper(
                {
                    "schema": {"fields": "dynamic"},
                    "mapping": {"old_field": "new_field"},
                }
            ),
            Truncate(
                {
                    "schema": {"fields": "dynamic"},
                    "fields": {"description": 100},  # Correct: fields is a dict
                }
            ),
        ]

        for transform in transforms:
            typed = MockTransformNodeConfig(
                plugin_config=dict(transform.config),
                schema={"fields": "dynamic"},
            )
            serialized = config_to_dict_filtered(typed)
            result = canonical_json(serialized)
            assert isinstance(result, str), f"Failed for {transform.name}"

    def test_real_sink_configs_json_safe(self) -> None:
        """Verify real sink plugin configs serialize correctly."""
        from elspeth.plugins.sinks.csv_sink import CSVSink
        from elspeth.plugins.sinks.json_sink import JSONSink

        # Create real plugin instances and extract their configs
        sinks = [
            CSVSink({"path": "/tmp/output.csv", "schema": {"fields": ["data: str"], "mode": "strict"}}),
            JSONSink({"path": "/tmp/output.json", "schema": {"fields": "dynamic"}}),
        ]

        for sink in sinks:
            # Sinks use opaque plugin_config
            typed = MockTransformNodeConfig(
                plugin_config=dict(sink.config),
                schema={"fields": "dynamic"},
            )
            serialized = config_to_dict_filtered(typed)
            result = canonical_json(serialized)
            assert isinstance(result, str), f"Failed for {sink.name}"

    def test_coalesce_config_json_safe(self) -> None:
        """Verify coalesce configs serialize correctly."""
        typed = MockCoalesceNodeConfig(
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
            schema={"fields": "dynamic", "guaranteed_fields": ["id"]},
            timeout_seconds=30.0,
            quorum_count=None,
            select_branch=None,
        )
        serialized = config_to_dict_filtered(typed)
        result = canonical_json(serialized)
        assert isinstance(result, str)

        # Verify None fields are excluded
        assert "quorum_count" not in serialized
        assert "select_branch" not in serialized
        assert "timeout_seconds" in serialized
