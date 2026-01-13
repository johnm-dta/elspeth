# tests/core/landscape/test_exports.py
"""Tests for Landscape module exports."""


class TestLandscapeExports:
    """Public API exports."""

    def test_can_import_database(self) -> None:
        from elspeth.core.landscape import LandscapeDB

        assert LandscapeDB is not None

    def test_can_import_recorder(self) -> None:
        from elspeth.core.landscape import LandscapeRecorder

        assert LandscapeRecorder is not None

    def test_can_import_models(self) -> None:
        from elspeth.core.landscape import (
            Artifact,
            Edge,
            Node,
            NodeState,
            Row,
            Run,
            Token,
        )

        assert Run is not None
        assert Node is not None
        assert Edge is not None
        assert Row is not None
        assert Token is not None
        assert NodeState is not None
        assert Artifact is not None

    def test_can_import_recorder_types(self) -> None:
        from elspeth.core.landscape import Batch, BatchMember, RoutingEvent

        assert Batch is not None
        assert BatchMember is not None
        assert RoutingEvent is not None

    def test_can_import_exporter(self) -> None:
        from elspeth.core.landscape import LandscapeExporter

        assert LandscapeExporter is not None

    def test_can_import_all_exports(self) -> None:
        """Verify __all__ exports are importable."""
        from elspeth.core.landscape import (
            # Models
            Artifact,
            Batch,
            BatchMember,
            BatchOutput,
            Call,
            Edge,
            LandscapeDB,
            LandscapeExporter,
            LandscapeRecorder,
            Node,
            NodeState,
            RoutingEvent,
            Row,
            Run,
            Token,
            TokenParent,
            # Tables
            artifacts_table,
            batch_members_table,
            batch_outputs_table,
            batches_table,
            calls_table,
            edges_table,
            metadata,
            node_states_table,
            nodes_table,
            routing_events_table,
            rows_table,
            runs_table,
            token_parents_table,
            tokens_table,
        )

        # Just verify they're all non-None
        assert all(
            x is not None
            for x in [
                LandscapeDB,
                LandscapeExporter,
                LandscapeRecorder,
                Artifact,
                Batch,
                BatchMember,
                BatchOutput,
                Call,
                Edge,
                Node,
                NodeState,
                RoutingEvent,
                Row,
                Run,
                Token,
                TokenParent,
                artifacts_table,
                batch_members_table,
                batch_outputs_table,
                batches_table,
                calls_table,
                edges_table,
                metadata,
                node_states_table,
                nodes_table,
                routing_events_table,
                rows_table,
                runs_table,
                token_parents_table,
                tokens_table,
            ]
        )
