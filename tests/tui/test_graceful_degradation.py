# tests/tui/test_graceful_degradation.py
"""Property-based tests for TUI widget optional field handling.

These tests verify that TUI widgets handle MISSING optional fields correctly.
Per CLAUDE.md Three-Tier Trust Model, optional fields may be absent, but when
present they MUST conform to their schema (Tier 1 - crash on corruption).

Field categories:
- Required fields (node_id, plugin_name, node_type): MUST always be present.
  Missing = bug in _load_node_state(), should crash.
- Simple optional fields (state_id, token_id, status, timing, hashes): May be
  absent when execution hasn't occurred. When present, type must match.
- Structured optional fields (error_json, artifact): May be absent. When present,
  MUST conform to schema (ExecutionError/TransformErrorReason, ArtifactDisplay).

Note: LineageTree tests have been removed as that widget requires strict
LineageData contracts. See test_lineage_types.py for LineageTree tests.
"""

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

# =============================================================================
# Strategies for VALID optional field values
# =============================================================================

# Valid ExecutionError format
execution_error_strategy = st.fixed_dictionaries(
    {
        "type": st.text(min_size=1, max_size=50),
        "exception": st.text(min_size=1, max_size=200),
    }
).map(json.dumps)

# Valid TransformErrorReason format
transform_error_strategy = st.fixed_dictionaries(
    {
        "reason": st.sampled_from(
            [
                "api_error",
                "missing_field",
                "validation_failed",
                "json_parse_failed",
            ]
        ),
    }
).map(json.dumps)

# Valid error_json: either ExecutionError or TransformErrorReason
valid_error_json_strategy = st.one_of(
    execution_error_strategy,
    transform_error_strategy,
)

# Valid artifact format (all required fields per ArtifactDisplay)
valid_artifact_strategy = st.fixed_dictionaries(
    {
        "artifact_id": st.text(min_size=1, max_size=50),
        "path_or_uri": st.text(min_size=1, max_size=200),
        "content_hash": st.text(min_size=1, max_size=64),
        "size_bytes": st.integers(min_value=0, max_value=10_000_000_000),
    }
)

# Simple optional field values (for fields that don't have complex schemas)
simple_optional_values = st.one_of(
    st.none(),
    st.text(max_size=100),
    st.integers(min_value=0, max_value=1000000),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
)


def make_valid_node_state(
    include_error: bool = False,
    include_artifact: bool = False,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a valid NodeStateInfo with required fields."""
    state: dict[str, Any] = {
        "node_id": "test-node-001",
        "plugin_name": "test-plugin",
        "node_type": "transform",
    }
    if extra_fields:
        state.update(extra_fields)
    return state


# Strategy for simple optional fields only (not error_json or artifact)
simple_optional_fields_strategy = st.dictionaries(
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
        ]
    ),
    values=simple_optional_values,
    max_size=8,
)


class TestNodeDetailPanelOptionalFields:
    """Property tests for NodeDetailPanel optional field handling.

    Tests that MISSING optional fields work correctly. When structured fields
    (error_json, artifact) ARE present, they must have valid schemas.
    """

    @given(optional_fields=simple_optional_fields_strategy)
    @settings(max_examples=100)
    def test_handles_missing_simple_optional_fields(self, optional_fields: dict[str, Any]) -> None:
        """NodeDetailPanel handles missing simple optional fields.

        Required fields (node_id, plugin_name, node_type) are always present.
        Simple optional fields can be missing or have basic values.
        Structured fields (error_json, artifact) are excluded from this test.
        """
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = make_valid_node_state(extra_fields=optional_fields)
        panel = NodeDetailPanel(state)  # type: ignore[arg-type]
        content = panel.render_content()
        assert isinstance(content, str)
        # Required fields always displayed
        assert "test-plugin" in content

    @given(include_error=st.booleans())
    @settings(max_examples=20)
    def test_error_json_present_or_absent(self, include_error: bool) -> None:
        """error_json can be absent, but when present must be valid schema."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = make_valid_node_state()
        if include_error:
            # Valid ExecutionError schema
            state["error_json"] = json.dumps(
                {
                    "type": "TestError",
                    "exception": "Test exception message",
                }
            )

        panel = NodeDetailPanel(state)  # type: ignore[arg-type]
        content = panel.render_content()
        assert isinstance(content, str)
        if include_error:
            assert "TestError" in content
            assert "Test exception message" in content

    @given(include_artifact=st.booleans())
    @settings(max_examples=20)
    def test_artifact_present_or_absent(self, include_artifact: bool) -> None:
        """artifact can be absent, but when present must be valid schema."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = make_valid_node_state()
        state["node_type"] = "sink"  # Artifacts are for sinks
        if include_artifact:
            # Valid Artifact schema (all required fields)
            state["artifact"] = {
                "artifact_id": "test-artifact-001",
                "path_or_uri": "/output/test.csv",
                "content_hash": "abc123def456",
                "size_bytes": 1024,
            }

        panel = NodeDetailPanel(state)  # type: ignore[arg-type]
        content = panel.render_content()
        assert isinstance(content, str)
        if include_artifact:
            assert "test-artifact-001" in content
            assert "/output/test.csv" in content

    @given(error_json=valid_error_json_strategy)
    @settings(max_examples=50)
    def test_valid_error_json_formats(self, error_json: str) -> None:
        """All valid error_json formats render without crashing."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = make_valid_node_state()
        state["error_json"] = error_json

        panel = NodeDetailPanel(state)  # type: ignore[arg-type]
        content = panel.render_content()
        assert isinstance(content, str)
        assert "Error:" in content

    @given(artifact=valid_artifact_strategy)
    @settings(max_examples=50)
    def test_valid_artifact_formats(self, artifact: dict[str, Any]) -> None:
        """All valid artifact formats render without crashing."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = make_valid_node_state()
        state["node_type"] = "sink"
        state["artifact"] = artifact

        panel = NodeDetailPanel(state)  # type: ignore[arg-type]
        content = panel.render_content()
        assert isinstance(content, str)
        assert "Artifact:" in content

    def test_none_state_renders(self) -> None:
        """None state (no node selected) renders correctly."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        content = panel.render_content()
        assert "No node selected" in content

    @given(
        initial_has_error=st.booleans(),
        updated_has_error=st.booleans(),
    )
    @settings(max_examples=20)
    def test_state_transitions_with_valid_data(
        self,
        initial_has_error: bool,
        updated_has_error: bool,
    ) -> None:
        """Transitioning between valid states should not crash."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        initial_state: dict[str, Any] | None = make_valid_node_state()
        if initial_has_error:
            initial_state["error_json"] = json.dumps(
                {
                    "reason": "api_error",
                }
            )

        updated_state: dict[str, Any] | None = make_valid_node_state()
        if updated_has_error:
            updated_state["error_json"] = json.dumps(
                {
                    "type": "RuntimeError",
                    "exception": "Something went wrong",
                }
            )

        panel = NodeDetailPanel(initial_state)  # type: ignore[arg-type]
        content1 = panel.render_content()
        assert isinstance(content1, str)

        panel.update_state(updated_state)  # type: ignore[arg-type]
        content2 = panel.render_content()
        assert isinstance(content2, str)
