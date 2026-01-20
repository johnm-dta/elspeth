"""Tests for TUI constants."""

import pytest


class TestWidgetIDs:
    """Tests for WidgetIDs constants."""

    def test_widget_ids_class_exists(self) -> None:
        """WidgetIDs class can be imported."""
        from elspeth.tui.constants import WidgetIDs

        assert WidgetIDs is not None

    def test_lineage_tree_constant_exists(self) -> None:
        """LINEAGE_TREE constant is defined."""
        from elspeth.tui.constants import WidgetIDs

        assert hasattr(WidgetIDs, "LINEAGE_TREE")
        assert isinstance(WidgetIDs.LINEAGE_TREE, str)
        assert WidgetIDs.LINEAGE_TREE == "lineage-tree"

    def test_detail_panel_constant_exists(self) -> None:
        """DETAIL_PANEL constant is defined."""
        from elspeth.tui.constants import WidgetIDs

        assert hasattr(WidgetIDs, "DETAIL_PANEL")
        assert isinstance(WidgetIDs.DETAIL_PANEL, str)
        assert WidgetIDs.DETAIL_PANEL == "detail-panel"

    def test_constants_are_valid_css_ids(self) -> None:
        """Constants are valid CSS ID selectors (no spaces, start with letter)."""
        from elspeth.tui.constants import WidgetIDs

        for attr in ["LINEAGE_TREE", "DETAIL_PANEL"]:
            value = getattr(WidgetIDs, attr)
            # CSS IDs must not contain spaces
            assert " " not in value, f"{attr} contains spaces"
            # CSS IDs should start with a letter or hyphen
            assert value[0].isalpha() or value[0] == "-", f"{attr} does not start with letter or hyphen"


class TestWidgetIDsIntegration:
    """Integration tests verifying constants are used correctly in ExplainApp."""

    @pytest.mark.asyncio
    async def test_widgets_use_constant_ids(self) -> None:
        """ExplainApp widgets have IDs matching WidgetIDs constants."""
        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as _pilot:
            # Query widgets by their constant IDs
            lineage_tree = app.query_one(f"#{WidgetIDs.LINEAGE_TREE}")
            detail_panel = app.query_one(f"#{WidgetIDs.DETAIL_PANEL}")

            assert lineage_tree is not None
            assert detail_panel is not None

    @pytest.mark.asyncio
    async def test_css_targets_correct_widgets(self) -> None:
        """CSS selectors in ExplainApp reference the WidgetIDs constants."""
        from elspeth.tui.constants import WidgetIDs
        from elspeth.tui.explain_app import ExplainApp

        # Verify CSS contains the widget ID references
        assert f"#{WidgetIDs.LINEAGE_TREE}" in ExplainApp.CSS
        assert f"#{WidgetIDs.DETAIL_PANEL}" in ExplainApp.CSS
