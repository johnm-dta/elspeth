"""Plugin lifecycle teardown for the orchestrator.

This module owns the finally-block cleanup contract: ``on_complete(ctx)`` then
``close()`` on every plugin (transforms, sinks, optionally the source), with
each hook individually guarded so one plugin's failure does not prevent the
others from cleaning up. Tier-1 errors (``TIER_1_ERRORS``) bypass the guard
and propagate immediately — they signal system-level corruption, not a
recoverable cleanup failure.

Pure delegation target for the Orchestrator (same pattern as
graph_wiring.py / runtime_preflight.py): it holds no orchestrator state and
operates only on the parameters passed in.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import structlog

import elspeth.contracts.errors as contract_errors

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.engine.orchestrator.types import PipelineConfig

slog = structlog.get_logger(__name__)


def cleanup_plugins(
    config: PipelineConfig,
    ctx: PluginContext,
    *,
    include_source: bool = True,
) -> None:
    """Clean up all plugins in the finally block.

    Implements the lifecycle teardown contract:
    1. on_complete(ctx) on all plugins (transforms, sinks, optionally source)
    2. close() on all plugins (source, transforms, sinks)

    on_complete() is called even on pipeline error -- it signals "processing
    is done" (success or failure), not "processing succeeded". close() is
    pure resource teardown and always follows on_complete().

    Each call is individually try/excepted so one plugin's failure does not
    prevent other plugins from cleaning up. All errors are collected and
    raised together after all cleanup completes.

    Extracted from _execute_run() and _process_resumed_rows() to eliminate
    duplication of the finally-block cleanup pattern.

    Args:
        config: Pipeline configuration
        ctx: Plugin context
        include_source: If True (default), calls on_complete() and close()
            on the source. Set to False for resume path where source wasn't opened.

    Raises:
        RuntimeError: If any plugin cleanup hook fails. Chained from the
            pending exception if one exists.
    """
    logger = slog
    pending_exc = sys.exc_info()[1]
    cleanup_errors: list[str] = []

    def record_cleanup_error(hook: str, plugin_name: str, error: Exception) -> None:
        logger.warning(
            "Plugin cleanup hook failed",
            hook=hook,
            plugin=plugin_name,
            error=str(error),
            error_type=type(error).__name__,
            exc_info=error,
        )
        cleanup_errors.append(f"{hook}({plugin_name}): {type(error).__name__}: {error}")

    def run_hook(hook_label: str, plugin_name: str, fn: Callable[[], None]) -> None:
        # Plugin cleanup MUST attempt every hook even when one fails — broad
        # catch is required by the best-effort lifecycle contract documented
        # above. FrameworkBugError / AuditIntegrityError (TIER_1_ERRORS) signal
        # system-level corruption that must crash immediately, so they are
        # re-raised by the dedicated Tier-1 clause BEFORE the broad catch can
        # downgrade them to a cleanup warning. Everything else is recorded and
        # folded into the RuntimeError raised after all hooks finish.
        try:
            fn()
        except contract_errors.TIER_1_ERRORS:
            raise
        except Exception as exc:
            record_cleanup_error(hook_label, plugin_name, exc)

    # Call on_complete for all plugins (even on error).
    # Base classes provide no-op implementations, so no hasattr needed.
    # functools.partial preserves the bound-method type for mypy and avoids
    # the loop-variable closure trap that lambdas would otherwise need
    # default-argument workarounds for.
    for transform in config.transforms:
        run_hook("transform.on_complete", transform.name, partial(transform.on_complete, ctx))
    for sink in config.sinks.values():
        run_hook("sink.on_complete", sink.name, partial(sink.on_complete, ctx))
    if include_source:
        for source in config.sources.values():
            run_hook("source.on_complete", source.name, partial(source.on_complete, ctx))

    # Close source (if included) and all sinks
    if include_source:
        for source in config.sources.values():
            run_hook("source.close", source.name, source.close)

    # Close all transforms (release resources - file handles, connections, etc.)
    for transform in config.transforms:
        run_hook("transform.close", transform.name, transform.close)

    # Close all sinks
    for sink in config.sinks.values():
        run_hook("sink.close", sink.name, sink.close)

    if cleanup_errors:
        error_summary = "; ".join(cleanup_errors)
        if pending_exc is not None:
            raise RuntimeError(f"Plugin cleanup failed: {error_summary}") from pending_exc
        raise RuntimeError(f"Plugin cleanup failed: {error_summary}")
