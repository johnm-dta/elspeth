"""Discovery result cache and tool-result serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import ToolResult
from elspeth.web.execution.runtime_preflight import RuntimePreflightEntry, RuntimePreflightKey


@dataclass(frozen=True, slots=True)
class CachedDiscoveryPayload:
    """State-independent portion of a cacheable discovery tool result."""

    success: bool
    affected_nodes: tuple[str, ...]
    data: Any


RuntimePreflightCache = dict[RuntimePreflightKey, RuntimePreflightEntry]


def tool_result_mutated_composition_state(
    *,
    version_before: int,
    result: ToolResult,
) -> bool:
    """Return True when a successful tool advanced the CompositionState version."""
    return result.success and result.updated_state.version > version_before


def pydantic_default(obj: Any) -> Any:
    """JSON serializer fallback for Pydantic models in tool results."""
    try:
        return obj.model_dump()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable") from None


def serialize_tool_result(result: Any) -> str:
    """Serialize a ToolResult to JSON, handling Pydantic models in data."""
    return json.dumps(result.to_dict(), default=pydantic_default)


def cached_discovery_payload(result: ToolResult) -> CachedDiscoveryPayload:
    """Extract the state-independent fields from a cacheable discovery result."""
    return CachedDiscoveryPayload(
        success=result.success,
        affected_nodes=result.affected_nodes,
        data=result.data,
    )


def result_from_cached_discovery_payload(
    state: CompositionState,
    cached: CachedDiscoveryPayload,
) -> ToolResult:
    """Rebuild a cached discovery result with the current state envelope."""
    return ToolResult(
        success=cached.success,
        updated_state=state,
        validation=state.validate(),
        affected_nodes=cached.affected_nodes,
        data=cached.data,
    )


def make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name + arguments."""
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
