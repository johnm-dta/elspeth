"""Composer sinks plane — sink discovery."""

from __future__ import annotations

from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import (
    CompositionState,
)

# Slice 2 — moved to ._common; re-imported so the helpers/classes still in this
# file resolve them via the in-module namespace as before.
# Slice 4 — moved to ._common; re-imported so helpers still in this file
# resolve them via the in-module namespace as before.
from elspeth.web.composer.tools._common import (
    ToolResult,
    _discovery_result,
)

# Slice 3 — moved to .blobs; re-imported so helpers/handlers still in this
# file resolve them via the in-module namespace as before.


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sinks())
