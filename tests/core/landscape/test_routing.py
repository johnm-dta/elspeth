# tests/core/landscape/test_routing.py
"""Tests for RoutingSpec."""

import pytest

from elspeth.core.landscape.models import RoutingSpec


class TestRoutingSpec:
    """Tests for RoutingSpec dataclass."""

    def test_valid_move_mode(self) -> None:
        """Move mode should be valid."""
        spec = RoutingSpec(edge_id="edge-1", mode="move")
        assert spec.edge_id == "edge-1"
        assert spec.mode == "move"

    def test_valid_copy_mode(self) -> None:
        """Copy mode should be valid."""
        spec = RoutingSpec(edge_id="edge-2", mode="copy")
        assert spec.mode == "copy"

    def test_frozen(self) -> None:
        """RoutingSpec should be immutable."""
        spec = RoutingSpec(edge_id="edge-1", mode="move")
        with pytest.raises(AttributeError):
            spec.edge_id = "changed"  # type: ignore

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            RoutingSpec(edge_id="edge-1", mode="invalid")  # type: ignore
