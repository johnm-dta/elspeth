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

import hashlib
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts.secret_scrub import scrub_text_for_audit

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol, TransformProtocol
    from elspeth.engine.orchestrator.types import PipelineConfig

slog = structlog.get_logger(__name__)

_CLEANUP_ERROR_PREVIEW_CHARS = 160
_UNSCRUBBED_CLEANUP_ERROR_TEXT = "<redacted-plugin-error>"


@contextmanager
def plugin_node_scope(ctx: PluginContext, node_id: str | None) -> Iterator[None]:
    """Temporarily scope lifecycle hooks to a plugin node."""
    previous_node_id = ctx.node_id
    ctx.node_id = node_id
    try:
        yield
    finally:
        ctx.node_id = previous_node_id


def _safe_cleanup_error_text(error: Exception) -> tuple[str, str, int]:
    """Return public-safe plugin exception text, digest, and raw length."""
    try:
        raw_text = str(error)
    except Exception:
        raw_text = f"<unrepresentable {type(error).__name__}>"

    digest = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:16]
    scrubbed = scrub_text_for_audit(raw_text)
    public_text = scrubbed if scrubbed != raw_text else _UNSCRUBBED_CLEANUP_ERROR_TEXT
    if len(public_text) > _CLEANUP_ERROR_PREVIEW_CHARS:
        public_text = public_text[:_CLEANUP_ERROR_PREVIEW_CHARS] + "..."
    return public_text, digest, len(raw_text)


def cleanup_plugins(
    config: PipelineConfig,
    ctx: PluginContext,
    *,
    include_source: bool = True,
    started_sources: Mapping[str, SourceProtocol] | None = None,
    started_transforms: Sequence[TransformProtocol] | None = None,
    started_sinks: Mapping[str, SinkProtocol] | None = None,
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
    raised together after all cleanup completes — unless an exception is
    already propagating (this function runs in ``finally`` blocks), in which
    case the collected cleanup errors are logged and the in-flight exception
    is preserved as the primary outcome.

    Extracted from _execute_run() and _process_resumed_rows() to eliminate
    duplication of the finally-block cleanup pattern.

    Args:
        config: Pipeline configuration
        ctx: Plugin context
        include_source: If True (default), calls on_complete() and close()
            on the source. Set to False for resume path where source wasn't opened.
        started_sources: Optional subset of sources whose on_start() completed.
            Defaults to all configured sources for steady-state run cleanup.
        started_transforms: Optional subset of transforms whose on_start()
            completed. Defaults to all configured transforms for steady-state
            run cleanup.
        started_sinks: Optional subset of sinks whose on_start() completed.
            Defaults to all configured sinks for steady-state run cleanup.

    Raises:
        RuntimeError: If any plugin cleanup hook fails and no exception is
            already propagating. When a pending exception exists, cleanup
            failures are logged instead so they never mask it.
    """
    logger = slog
    pending_exc = sys.exc_info()[1]
    cleanup_errors: list[str] = []
    sources_for_cleanup = started_sources if started_sources is not None else config.sources
    transforms_for_cleanup = started_transforms if started_transforms is not None else config.transforms
    sinks_for_cleanup = started_sinks if started_sinks is not None else config.sinks

    def record_cleanup_error(hook: str, plugin_name: str, error: Exception) -> None:
        public_error, error_digest, error_length = _safe_cleanup_error_text(error)
        logger.warning(
            "Plugin cleanup hook failed",
            hook=hook,
            plugin=plugin_name,
            error=public_error,
            error_type=type(error).__name__,
            error_sha256=error_digest,
            error_length=error_length,
        )
        cleanup_errors.append(
            f"{hook}({plugin_name}): {type(error).__name__}: {public_error} [message_sha256={error_digest}, message_length={error_length}]"
        )

    def run_hook(hook_label: str, plugin_name: str, node_id: str | None, fn: Callable[[], None]) -> None:
        # Plugin cleanup MUST attempt every hook even when one fails — broad
        # catch is required by the best-effort lifecycle contract documented
        # above. FrameworkBugError / AuditIntegrityError (TIER_1_ERRORS) signal
        # system-level corruption that must crash immediately, so they are
        # re-raised by the dedicated Tier-1 clause BEFORE the broad catch can
        # downgrade them to a cleanup warning. Everything else is recorded and
        # folded into the RuntimeError raised after all hooks finish.
        try:
            with plugin_node_scope(ctx, node_id):
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
    for transform in transforms_for_cleanup:
        run_hook(
            "transform.on_complete",
            transform.name,
            getattr(transform, "node_id", None),
            partial(transform.on_complete, ctx),
        )
    for sink in sinks_for_cleanup.values():
        run_hook(
            "sink.on_complete",
            sink.name,
            getattr(sink, "node_id", None),
            partial(sink.on_complete, ctx),
        )
    if include_source:
        for source in sources_for_cleanup.values():
            run_hook(
                "source.on_complete",
                source.name,
                getattr(source, "node_id", None),
                partial(source.on_complete, ctx),
            )

    # Close source (if included) and all sinks
    if include_source:
        for source in sources_for_cleanup.values():
            run_hook("source.close", source.name, getattr(source, "node_id", None), source.close)

    # Close all transforms (release resources - file handles, connections, etc.)
    for transform in transforms_for_cleanup:
        run_hook("transform.close", transform.name, getattr(transform, "node_id", None), transform.close)

    # Close all sinks
    for sink in sinks_for_cleanup.values():
        run_hook("sink.close", sink.name, getattr(sink, "node_id", None), sink.close)

    if cleanup_errors:
        error_summary = "; ".join(cleanup_errors)
        if pending_exc is not None:
            # An exception is already propagating through the caller's finally
            # block. Raising here would REPLACE it — e.g. swap a
            # _RunFailedWithPartialResultError (real partial counters, failed
            # ceremony) for a cleanup RuntimeError handled by the generic
            # ceremony. The per-hook failures are already logged above; record
            # the aggregate and let the original exception continue.
            logger.error(
                "Plugin cleanup failed during exception propagation; original error preserved",
                cleanup_errors=tuple(cleanup_errors),
                pending_error_type=type(pending_exc).__name__,
            )
            return
        raise RuntimeError(f"Plugin cleanup failed: {error_summary}")
