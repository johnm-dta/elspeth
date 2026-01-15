# src/elspeth/tui/constants.py
"""TUI constants for ELSPETH.

Provides centralized widget IDs and other constants to avoid magic strings.
"""


class WidgetIDs:
    """Widget ID constants for Textual CSS and compose() alignment.

    Using constants ensures CSS selectors match widget IDs - a typo in either
    location would otherwise cause silent styling failures.
    """

    LINEAGE_TREE = "lineage-tree"
    DETAIL_PANEL = "detail-panel"
