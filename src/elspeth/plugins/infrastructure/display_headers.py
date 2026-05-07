"""Display header resolution for sink plugins.

Provides shared display header infrastructure used by CSV, JSON, and Azure Blob
sinks. The functions in this module operate on sink instances that have been
initialized with init_display_headers().

Design: Functions rather than mixin class. This avoids MRO complexity and makes
the dependency on sink state explicit. Each function takes the sink instance as
its first argument, typed via DisplayHeaderHost protocol.

Ordering requirement: The host sink MUST have _output_contract already
initialized (by BaseSink.__init__) BEFORE calling init_display_headers().
init_display_headers() asserts this precondition.

State attributes set by init_display_headers():
    _headers_mode: HeaderMode — which header mode is active
    _headers_custom_mapping: dict | None — explicit field→display mapping (CUSTOM mode)
    _resolved_display_headers: dict | None — lazily populated from Landscape (ORIGINAL mode)
    _display_headers_resolved: bool — guard flag for lazy resolution
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from elspeth.contracts.header_modes import HeaderMode, resolve_headers
from elspeth.contracts.plugin_protocols import DisplayHeaderHost

if TYPE_CHECKING:
    from elspeth.contracts.contexts import SinkContext


def init_display_headers(
    sink: DisplayHeaderHost,
    headers_mode: HeaderMode,
    headers_custom_mapping: dict[str, str] | None = None,
) -> None:
    """Initialize display header state on a sink instance.

    Called from sink __init__ to set up the attributes that the other
    functions in this module depend on.

    Args:
        sink: The sink instance to initialize.
        headers_mode: The resolved HeaderMode (NORMALIZED, ORIGINAL, CUSTOM).
        headers_custom_mapping: Explicit mapping for CUSTOM mode.

    Raises:
        RuntimeError: If _output_contract is not set on the sink (forgot super().__init__()).
    """
    # Offensive assertion: _output_contract must already be set by BaseSink.__init__().
    # Access directly — if it's missing, AttributeError IS the correct crash.
    # Per CLAUDE.md: "hasattr() is unconditionally banned" and try/except AttributeError
    # is equivalent banned defensive programming.
    _ = sink._output_contract  # Crashes with AttributeError if super().__init__() not called

    if headers_mode == HeaderMode.CUSTOM and headers_custom_mapping is None:
        raise ValueError(
            "CUSTOM header mode requires an explicit headers_custom_mapping dict. "
            "Got headers_mode=CUSTOM with headers_custom_mapping=None — "
            "this would silently fall back to NORMALIZED mode."
        )

    sink._headers_mode = headers_mode
    sink._headers_custom_mapping = headers_custom_mapping
    sink._resolved_display_headers = None
    sink._display_headers_resolved = False
    sink._needs_resume_field_resolution = headers_mode == HeaderMode.ORIGINAL


def get_effective_display_headers(sink: DisplayHeaderHost) -> dict[str, str] | None:
    """Get the effective display header mapping.

    Priority order:
    1. NORMALIZED mode — no mapping
    2. CUSTOM mode — use custom mapping from 'headers' config
    3. ORIGINAL mode with contract — use resolve_headers()
    4. ORIGINAL mode with resolved headers (from Landscape query)
    5. None (fallback to normalized names)

    Args:
        sink: Sink instance with display header state.

    Returns:
        Dict mapping normalized field name → display name, or None if no
        display headers are configured or if using NORMALIZED mode.
    """
    if sink._headers_mode == HeaderMode.NORMALIZED:
        return None

    if sink._headers_mode == HeaderMode.CUSTOM:
        return sink._headers_custom_mapping

    # ORIGINAL mode — use contract to resolve headers
    if sink._output_contract is not None:
        return resolve_headers(
            contract=sink._output_contract,
            mode=HeaderMode.ORIGINAL,
            custom_mapping=None,
        )
    # Fallback to lazily-resolved display headers from Landscape
    if sink._resolved_display_headers is not None:
        return sink._resolved_display_headers
    return None


def resolve_contract_from_context_if_needed(sink: DisplayHeaderHost, ctx: SinkContext) -> None:
    """Lazily resolve output contract from context for headers: original mode.

    Called on first write() to capture ctx.contract if _output_contract is not
    already set. This allows headers: original mode to work without explicit
    orchestrator wiring of set_output_contract().

    Args:
        sink: Sink instance with display header state.
        ctx: Plugin context with potential contract from orchestrator.
    """
    if sink._headers_mode != HeaderMode.ORIGINAL:
        return
    if sink._output_contract is not None:
        return  # Already set explicitly
    if ctx.contract is not None:
        sink._output_contract = ctx.contract


def set_resume_field_resolution(sink: DisplayHeaderHost, resolution_mapping: dict[str, str]) -> None:
    """Set field resolution mapping for resume validation.

    Called by CLI during `elspeth resume` to provide the source field resolution
    mapping BEFORE calling validate_output_target(). This allows validation to
    correctly compare expected display names against existing output when
    headers mode is ORIGINAL.

    Args:
        sink: Sink instance with display header state.
        resolution_mapping: Dict mapping original header name → normalized field name.
            This is the same format returned by Landscape.get_source_field_resolution().
    """
    if sink._headers_mode != HeaderMode.ORIGINAL:
        return  # Only needed for ORIGINAL mode

    # Build reverse mapping: normalized → original (display name)
    reverse = {v: k for k, v in resolution_mapping.items()}
    if len(reverse) != len(resolution_mapping):
        # Duplicate normalized names in the mapping — two original names collapsed
        # to the same normalized name. This is Tier 1 data (from Landscape), so crash.
        duplicates = [k for k, count in Counter(resolution_mapping.values()).items() if count > 1]
        raise ValueError(
            f"Field resolution mapping has duplicate normalized names: {duplicates}. "
            f"Multiple original names map to the same normalized name — data integrity issue."
        )
    sink._resolved_display_headers = reverse
    sink._display_headers_resolved = True


def resolve_display_headers_if_needed(sink: DisplayHeaderHost, ctx: SinkContext) -> None:
    """Lazily resolve display headers from Landscape if headers mode is ORIGINAL.

    Called on first write() to fetch field resolution mapping. This MUST be lazy
    because the orchestrator calls sink.on_start() BEFORE source.load() iterates,
    and record_source_field_resolution() only happens after the first source row.

    The _display_headers_resolved flag is set AFTER successful resolution (not
    before) so that if the Landscape query raises, the next write() call retries.

    Args:
        sink: Sink instance with display header state.
        ctx: Plugin context with Landscape access.

    Raises:
        ValueError: If Landscape is unavailable or source didn't record resolution.
    """
    if sink._display_headers_resolved:
        return  # Already resolved (or not needed)

    if sink._headers_mode != HeaderMode.ORIGINAL:
        sink._display_headers_resolved = True  # Nothing to resolve — mark done
        return

    # Skip if contract already provides header resolution (takes precedence)
    if sink._output_contract is not None:
        sink._display_headers_resolved = True
        return

    # Fetch source field resolution from Landscape
    if ctx.landscape is None:
        raise ValueError(
            "headers: original requires Landscape to be available. This is a framework bug - context should have landscape set."
        )

    resolution_mapping = ctx.landscape.get_source_field_resolution(ctx.run_id)
    if resolution_mapping is None:
        raise ValueError(
            "headers: original but source did not record field resolution. "
            "Field normalization records the resolution mapping automatically."
        )

    # Build reverse mapping: final (normalized) → original
    reverse = {v: k for k, v in resolution_mapping.items()}
    if len(reverse) != len(resolution_mapping):
        duplicates = [k for k, count in Counter(resolution_mapping.values()).items() if count > 1]
        raise ValueError(
            f"Field resolution mapping has duplicate normalized names: {duplicates}. "
            f"Multiple original names map to the same normalized name — data integrity issue."
        )
    sink._resolved_display_headers = reverse
    # Flag set AFTER success — if the query raised, next write() retries
    sink._display_headers_resolved = True


def apply_display_headers(sink: DisplayHeaderHost, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply display header mapping to row keys.

    Args:
        sink: Sink instance with display header state.
        rows: List of row dicts with normalized field names.

    Returns:
        List of row dicts with display names as keys. If no display headers
        are configured, returns the original rows unchanged.

    Raises:
        ValueError: If multiple fields map to the same output key (collision).
    """
    display_map = get_effective_display_headers(sink)
    if display_map is None:
        return rows

    # Transform each row's keys to display names
    # In ORIGINAL mode, fields not in the mapping keep their original names
    # (transform-added fields that weren't in the source contract).
    # In CUSTOM mode, ALL fields must be explicitly mapped — unmapped fields
    # are a configuration error per the header contract.
    is_custom = sink._headers_mode == HeaderMode.CUSTOM
    result_rows = []
    for row in rows:
        mapped: dict[str, Any] = {}
        for k, v in row.items():
            if k in display_map:
                display_key = display_map[k]
            elif is_custom:
                raise ValueError(
                    f"CUSTOM header mode has no mapping for field '{k}'. "
                    f"All fields must be explicitly mapped — silent fallback to normalized "
                    f"names risks data corruption in external system handover. "
                    f"Mapped fields: {sorted(display_map.keys())}"
                )
            else:
                display_key = k
            if display_key in mapped:
                raise ValueError(
                    f"Header collision: multiple fields map to output key '{display_key}'. "
                    f"Check display_headers mapping for duplicate targets."
                )
            mapped[display_key] = v
        result_rows.append(mapped)
    return result_rows
