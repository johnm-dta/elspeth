"""Tests for TUI constants."""

import pytest


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
