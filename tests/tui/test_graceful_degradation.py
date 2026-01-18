"""Property-based tests for TUI widget graceful degradation.

These tests verify that TUI widgets handle incomplete or malformed data
for OPTIONAL fields without crashing. The widgets display audit data that
may have optional execution state fields missing.

Per CLAUDE.md Three-Tier Trust Model:
- Required fields (node_id, plugin_name, node_type): Must always be present.
  Missing = bug in _load_node_state(), should crash.
- Optional fields (state_id, token_id, status, timing, hashes, etc.):
  May be missing when execution hasn't occurred yet.

Property tests generate arbitrary data for OPTIONAL fields only, while
ensuring required fields are always present.

Note: LineageTree tests have been removed as that widget now requires
strict LineageData contracts. See test_lineage_types.py for LineageTree tests.
"""

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

# Strategy for optional field values (can be None, various types)
optional_field_values = st.one_of(
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
)

# Strategy for optional fields only
optional_fields_strategy = st.dictionaries(
    keys=st.sampled_from(
        [
            "state_id",
            "token_id",
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
    values=optional_field_values,
    max_size=12,
)


def make_node_state_with_required_fields(
    optional_fields: dict[str, Any],
) -> dict[str, Any]:
    """Create a NodeStateInfo with required fields + arbitrary optional fields."""
    required = {
        "node_id": "test-node-001",
        "plugin_name": "test-plugin",
        "node_type": "transform",
    }
    return {**required, **optional_fields}


# Strategy for generating node state dictionaries with required fields
node_state_strategy = st.one_of(
    st.none(),  # Explicitly None (no node selected)
    optional_fields_strategy.map(make_node_state_with_required_fields),
)


class TestNodeDetailPanelGracefulDegradation:
    """Property tests for NodeDetailPanel graceful degradation.

    These tests verify that optional fields can have arbitrary values
    without crashing, while required fields (node_id, plugin_name, node_type)
    are always present.
    """

    @given(node_state=node_state_strategy)
    @settings(max_examples=100)
    def test_handles_arbitrary_optional_fields(
        self, node_state: dict[str, Any] | None
    ) -> None:
        """NodeDetailPanel handles arbitrary optional field values.

        Required fields (node_id, plugin_name, node_type) are always present.
        Optional fields can have any value type or be missing entirely.
        The widget uses .get() + explicit fallback for optional fields.
        """
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(node_state)
        # Should render without raising
        content = panel.render_content()
        # Must return a string (even if mostly defaults)
        assert isinstance(content, str)

    @given(node_state=node_state_strategy)
    @settings(max_examples=50)
    def test_update_state_handles_arbitrary_optional_data(
        self, node_state: dict[str, Any] | None
    ) -> None:
        """NodeDetailPanel.update_state handles arbitrary optional field values."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        # Update to state with arbitrary optional fields
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
