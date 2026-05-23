"""Composer sinks plane — sink discovery."""

from __future__ import annotations

from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolResult,
    _discovery_result,
)


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sinks())
