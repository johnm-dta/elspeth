"""Tests for node detail panel widget."""


class TestNodeDetailPanel:
    """Tests for NodeDetailPanel widget."""

    def test_can_import_widget(self) -> None:
        """NodeDetailPanel can be imported."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        assert NodeDetailPanel is not None

    def test_display_transform_state(self) -> None:
        """Display details for a transform node state."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-001",
            "node_id": "node-001",
            "token_id": "token-001",
            "plugin_name": "filter",
            "node_type": "transform",
            "status": "completed",
            "input_hash": "abc123",
            "output_hash": "def456",
            "duration_ms": 12.5,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.012Z",
            "error_json": None,
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "filter" in content
        assert "completed" in content
        assert "abc123" in content
        assert "12.5" in content

    def test_display_failed_state(self) -> None:
        """Display details for a failed node state with error."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-002",
            "node_id": "node-001",
            "token_id": "token-001",
            "plugin_name": "transform",
            "node_type": "transform",
            "status": "failed",
            "input_hash": "abc123",
            "output_hash": None,
            "duration_ms": 5.2,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.005Z",
            "error_json": '{"type": "ValueError", "message": "Invalid input"}',
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "failed" in content.lower()
        assert "ValueError" in content
        assert "Invalid input" in content

    def test_display_source_state(self) -> None:
        """Display details for a source node."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-003",
            "node_id": "source-001",
            "token_id": "token-001",
            "plugin_name": "csv_source",
            "node_type": "source",
            "status": "completed",
            "input_hash": None,  # Sources have no input
            "output_hash": "xyz789",
            "duration_ms": 100.0,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.100Z",
            "error_json": None,
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "csv_source" in content
        assert "source" in content.lower()

    def test_display_sink_state(self) -> None:
        """Display details for a sink node."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        node_state = {
            "state_id": "state-004",
            "node_id": "sink-001",
            "token_id": "token-001",
            "plugin_name": "csv_sink",
            "node_type": "sink",
            "status": "completed",
            "input_hash": "final123",
            "output_hash": None,  # Sinks produce artifacts, not output_hash
            "duration_ms": 25.0,
            "started_at": "2024-01-01T10:00:00Z",
            "completed_at": "2024-01-01T10:00:00.025Z",
            "error_json": None,
            "artifact": {
                "artifact_id": "artifact-001",
                "path_or_uri": "/output/result.csv",
                "content_hash": "artifact_hash_789",
                "size_bytes": 1024,
            },
        }

        panel = NodeDetailPanel(node_state)
        content = panel.render_content()

        assert "csv_sink" in content
        assert "artifact" in content.lower()
        assert "/output/result.csv" in content

    def test_empty_state(self) -> None:
        """Handle empty/null state gracefully."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        content = panel.render_content()

        assert "No node selected" in content or "Select a node" in content

    def test_update_state(self) -> None:
        """Can update displayed state."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        assert "No node selected" in panel.render_content()

        panel.update_state(
            {
                "state_id": "state-001",
                "plugin_name": "filter",
                "node_type": "transform",
                "status": "completed",
            }
        )
        content = panel.render_content()
        assert "filter" in content

    def test_format_size_bytes(self) -> None:
        """Formats file sizes correctly."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(
            {
                "state_id": "state-001",
                "plugin_name": "sink",
                "node_type": "sink",
                "status": "completed",
                "artifact": {
                    "artifact_id": "a-001",
                    "path_or_uri": "/out.csv",
                    "content_hash": "hash",
                    "size_bytes": 1536,  # 1.5 KB
                },
            }
        )
        content = panel.render_content()
        assert "1.5 KB" in content

    def test_malformed_error_json_logs_warning(self) -> None:
        """Malformed error_json is logged but still displayed raw."""
        from unittest.mock import MagicMock, patch

        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        malformed_json = "not valid json {{{{"
        node_state = {
            "state_id": "state-005",
            "node_id": "node-001",
            "token_id": "token-001",
            "plugin_name": "transform",
            "node_type": "transform",
            "status": "failed",
            "input_hash": "abc123",
            "output_hash": None,
            "error_json": malformed_json,
        }

        mock_logger = MagicMock()
        with patch("structlog.get_logger", return_value=mock_logger):
            panel = NodeDetailPanel(node_state)
            content = panel.render_content()

        # Raw JSON still displayed (trust boundary fallback)
        assert malformed_json in content

        # Warning was logged with context
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Failed to parse error_json" in call_args[0][0]
        assert call_args[1]["state_id"] == "state-005"
        assert call_args[1]["error_json_preview"] == malformed_json
        assert "decode_error" in call_args[1]
