"""Property-based tests for TUI widget graceful degradation.

These tests verify that TUI widgets handle incomplete or malformed Landscape
data without crashing. The widgets display audit data that may be incomplete
for failed or partial runs. The `.get()` patterns in the widgets are
intentional trust boundary handling, not defensive bug-hiding.

Property tests generate arbitrary data to prove graceful degradation works
for any possible incomplete state from the Landscape.

Note: LineageTree tests have been removed as that widget now requires
strict LineageData contracts. See test_lineage_types.py for LineageTree tests.
"""

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

# Strategy for generating arbitrary node state dictionaries
node_state_strategy = st.one_of(
    st.none(),  # Explicitly None
    st.dictionaries(
        keys=st.sampled_from(
            [
                "state_id",
                "node_id",
                "token_id",
                "plugin_name",
                "node_type",
                "status",
                "input_hash",
                "output_hash",
                "duration_ms",
                "started_at",
                "completed_at",
                "error_json",
                "artifact",
                "unknown_field",  # Test unknown fields too
            ]
        ),
        values=st.one_of(
            st.none(),
            st.text(max_size=100),
            st.integers(min_value=-1000, max_value=1000000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            # Nested dict for artifact
            st.dictionaries(
                keys=st.sampled_from(
                    ["artifact_id", "path_or_uri", "content_hash", "size_bytes"]
                ),
                values=st.one_of(st.none(), st.text(max_size=50), st.integers()),
                max_size=4,
            ),
        ),
        max_size=15,
    ),
)


class TestNodeDetailPanelGracefulDegradation:
    """Property tests for NodeDetailPanel graceful degradation."""

    @given(node_state=node_state_strategy)
    @settings(max_examples=100)
    def test_handles_arbitrary_incomplete_data(
        self, node_state: dict[str, Any] | None
    ) -> None:
        """NodeDetailPanel should not crash on incomplete/malformed data.

        The widget uses .get() for graceful degradation at the trust boundary
        where Landscape data enters the TUI. This test proves that ANY
        combination of missing, null, or malformed fields renders safely.
        """
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(node_state)
        # Should render without raising
        content = panel.render_content()
        # Must return a string (even if mostly defaults)
        assert isinstance(content, str)

    @given(node_state=node_state_strategy)
    @settings(max_examples=50)
    def test_update_state_handles_arbitrary_data(
        self, node_state: dict[str, Any] | None
    ) -> None:
        """NodeDetailPanel.update_state should accept arbitrary data without crashing."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        # Update to arbitrary state
        panel.update_state(node_state)
        # Should render without raising
        content = panel.render_content()
        assert isinstance(content, str)

    @given(
        initial_state=node_state_strategy,
        updated_state=node_state_strategy,
    )
    @settings(max_examples=50)
    def test_state_transitions_are_safe(
        self,
        initial_state: dict[str, Any] | None,
        updated_state: dict[str, Any] | None,
    ) -> None:
        """Transitioning between any two states should not crash."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(initial_state)
        content1 = panel.render_content()
        assert isinstance(content1, str)

        panel.update_state(updated_state)
        content2 = panel.render_content()
        assert isinstance(content2, str)
