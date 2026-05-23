"""Composer sinks plane — sink discovery."""

from __future__ import annotations

from typing import Any

from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
)


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _discovery_result(state, context.catalog.list_sinks())
