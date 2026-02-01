"""Tests for explicit node_id assignment with validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from elspeth.contracts import NodeID, SinkName
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator


class TestNodeIdAssignment:
    """Test node_id assignment validation."""

    def test_assign_plugin_node_ids_validates_source(self) -> None:
        """Should raise if source lacks node_id attribute."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        # Source without node_id attribute
        source = object()  # Plain object, no node_id

        with pytest.raises(AttributeError):
            orchestrator._assign_plugin_node_ids(
                source=source,  # type: ignore[arg-type]
                transforms=[],
                sinks={},
                source_id=NodeID("source-1"),
                transform_id_map={},
                sink_id_map={},
            )

    def test_assign_plugin_node_ids_sets_source_id(self) -> None:
        """Should set node_id on source plugin."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[],
            sinks={},
            source_id=NodeID("source-1"),
            transform_id_map={},
            sink_id_map={},
        )

        assert source.node_id == "source-1"

    def test_assign_plugin_node_ids_sets_transform_ids(self) -> None:
        """Should set node_id on all transforms."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        t1 = MagicMock()
        t1.node_id = None
        t2 = MagicMock()
        t2.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[t1, t2],
            sinks={},
            source_id=NodeID("source-1"),
            transform_id_map={0: NodeID("transform-0"), 1: NodeID("transform-1")},
            sink_id_map={},
        )

        assert t1.node_id == "transform-0"
        assert t2.node_id == "transform-1"

    def test_assign_plugin_node_ids_sets_sink_ids(self) -> None:
        """Should set node_id on all sinks."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        sink1 = MagicMock()
        sink1.node_id = None
        sink2 = MagicMock()
        sink2.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[],
            sinks={"output": sink1, "errors": sink2},
            source_id=NodeID("source-1"),
            transform_id_map={},
            sink_id_map={SinkName("output"): NodeID("sink-output"), SinkName("errors"): NodeID("sink-errors")},
        )

        assert sink1.node_id == "sink-output"
        assert sink2.node_id == "sink-errors"

    def test_assign_plugin_node_ids_raises_for_missing_transform(self) -> None:
        """Should raise ValueError if transform sequence not in map."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        t1 = MagicMock()
        t1.node_id = None

        with pytest.raises(ValueError, match="Transform at sequence 0 not found"):
            orchestrator._assign_plugin_node_ids(
                source=source,
                transforms=[t1],
                sinks={},
                source_id=NodeID("source-1"),
                transform_id_map={},  # Missing mapping for sequence 0
                sink_id_map={},
            )

    def test_assign_plugin_node_ids_raises_for_missing_sink(self) -> None:
        """Should raise ValueError if sink name not in map."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        sink = MagicMock()
        sink.node_id = None

        with pytest.raises(ValueError, match="Sink 'output' not found"):
            orchestrator._assign_plugin_node_ids(
                source=source,
                transforms=[],
                sinks={"output": sink},
                source_id=NodeID("source-1"),
                transform_id_map={},
                sink_id_map={},  # Missing mapping for "output"
            )

    def test_assign_plugin_node_ids_all_plugins_together(self) -> None:
        """Should correctly assign node_ids to all plugin types at once."""
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        t1 = MagicMock()
        t1.node_id = None
        t2 = MagicMock()
        t2.node_id = None

        sink1 = MagicMock()
        sink1.node_id = None
        sink2 = MagicMock()
        sink2.node_id = None

        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[t1, t2],
            sinks={"default": sink1, "errors": sink2},
            source_id=NodeID("src-001"),
            transform_id_map={0: NodeID("xform-0"), 1: NodeID("xform-1")},
            sink_id_map={SinkName("default"): NodeID("sink-default"), SinkName("errors"): NodeID("sink-errors")},
        )

        # Verify all assignments
        assert source.node_id == "src-001"
        assert t1.node_id == "xform-0"
        assert t2.node_id == "xform-1"
        assert sink1.node_id == "sink-default"
        assert sink2.node_id == "sink-errors"

    def test_assign_plugin_node_ids_preserves_preassigned_transform_ids(self) -> None:
        """Should preserve pre-assigned node_ids on transforms (aggregation path).

        Aggregation transforms have their node_id assigned by CLI from
        aggregation_id_map before _assign_plugin_node_ids is called.
        The method should skip these transforms without error.
        """
        db = MagicMock(spec=LandscapeDB)
        orchestrator = Orchestrator(db)

        source = MagicMock()
        source.node_id = None

        # Regular transform without node_id
        t1 = MagicMock()
        t1.node_id = None

        # Aggregation transform with pre-assigned node_id
        t2 = MagicMock()
        t2.node_id = "agg-1"  # Pre-assigned by CLI

        # Regular transform without node_id
        t3 = MagicMock()
        t3.node_id = None

        # Note: transform_id_map only has entries for transforms WITHOUT pre-assigned node_ids
        # Sequence 1 (t2) is not in the map because it already has node_id
        orchestrator._assign_plugin_node_ids(
            source=source,
            transforms=[t1, t2, t3],
            sinks={},
            source_id=NodeID("source-1"),
            transform_id_map={0: NodeID("transform-0"), 2: NodeID("transform-2")},  # No entry for seq 1
            sink_id_map={},
        )

        # Verify pre-assigned node_id remains unchanged
        assert t2.node_id == "agg-1"
        # Verify other transforms get assigned from map
        assert t1.node_id == "transform-0"
        assert t3.node_id == "transform-2"
