# tests/core/landscape/test_routing.py
"""Tests for RoutingSpec imported via landscape/models.py.

These tests verify the re-export path works correctly.
The canonical tests are in tests/contracts/test_routing.py.
"""

import pytest

from elspeth.contracts import RoutingMode

# Test the re-export path from landscape __init__.py
from elspeth.core.landscape import RoutingSpec


class TestRoutingSpecReExport:
    """Tests for RoutingSpec re-exported from landscape package."""

    def test_can_import_from_landscape(self) -> None:
        """RoutingSpec should be importable from landscape package."""
        # This import above confirms it works
        assert RoutingSpec is not None

    def test_create_with_move_mode(self) -> None:
        """Move mode should be valid with enum."""
        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)
        assert spec.edge_id == "edge-1"
        assert spec.mode == RoutingMode.MOVE

    def test_create_with_copy_mode(self) -> None:
        """Copy mode should be valid with enum."""
        spec = RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY)
        assert spec.mode == RoutingMode.COPY

    def test_frozen(self) -> None:
        """RoutingSpec should be immutable."""
        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)
        with pytest.raises(AttributeError):
            spec.edge_id = "changed"  # type: ignore
