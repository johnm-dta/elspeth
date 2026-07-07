"""Operation lifecycle management for source/sink I/O.

Operations are the run/node equivalent of node_states - they provide
a parent context for external calls made outside per-row transform state,
including source.load(), sink.write(), and runtime preflight checks.

This module provides the track_operation context manager which handles:
- Operation creation and completion
- Context wiring (ctx.operation_id)
- Duration calculation
- Exception capture with proper status
- Guaranteed completion (even on DB failure)
- Context cleanup (clears operation_id after completion)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import elspeth.contracts.errors as contract_errors
from elspeth.contracts.secret_scrub import scrub_payload_for_audit, scrub_text_for_audit

if TYPE_CHECKING:
    from elspeth.contracts import Operation, OperationType
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.landscape.execution_repository import ExecutionRepository

logger = logging.getLogger(__name__)


def _render_exception(exc: BaseException) -> str:
    """Render an exception to a non-empty, secret-scrubbed audit message.

    The returned string is persisted verbatim as ``operations.error_message``
    (``track_operation`` -> ``complete_operation``), a Tier-1 audit record. This
    is the single chokepoint through which *every* operation error is rendered,
    so the scrub here guarantees no operation error message reaches the audit
    trail unscrubbed — regardless of which exception type failed. Provider and
    runtime exceptions can interpolate bearer tokens / API keys into their
    message text (e.g. ``RuntimePreflightFailedError`` embeds the underlying
    client error), and ``str(exc)`` would otherwise persist that secret raw.

    Best-effort: ``scrub_text_for_audit`` is pattern-based, so a novel secret
    format absent from its rule set can still slip through — mitigation, not a
    guarantee. If rendering *or* scrubbing fails, fall back to the (secret-free)
    exception type name rather than risk leaking an unscrubbed string.
    """
    try:
        message = scrub_text_for_audit(str(exc))
    except BaseException:
        return type(exc).__name__
    return message if message else type(exc).__name__


class OperationHandle:
    """Mutable handle for capturing operation output within context manager.

    Allows the caller to set output_data during the operation, which will
    be recorded when the operation completes.

    The `operation` field is read-only after construction — mutating it would
    corrupt audit trail linkage. Only `output_data` is writable (it's the
    write slot for the context manager pattern).

    Usage:
        with track_operation(...) as handle:
            result = sink.write(rows, ctx)
            handle.output_data = {"artifact_path": result.path}  # Explicit!
    """

    __slots__ = ("_operation", "output_data")

    def __init__(self, operation: Operation, output_data: dict[str, Any] | None = None) -> None:
        self._operation = operation
        self.output_data = output_data

    @property
    def operation(self) -> Operation:
        return self._operation


@contextmanager
def track_operation(
    recorder: ExecutionRepository,
    run_id: str,
    node_id: str,
    operation_type: OperationType,
    ctx: PluginContext,
    *,
    input_data: dict[str, Any] | None = None,
) -> Iterator[OperationHandle]:
    """Context manager for operation lifecycle tracking.

    Handles:
    - Operation creation
    - Context wiring (ctx.operation_id)
    - Duration calculation
    - Exception capture with proper status
    - Audit integrity enforcement (fail run if audit write fails)
    - Context cleanup (clears operation_id after completion)

    The context manager pattern ensures operations are always completed,
    even when exceptions occur. This is critical for audit integrity -
    orphaned operations in 'open' status indicate framework bugs.

    Audit Integrity:
        If complete_operation() fails (DB error), the run MUST fail.
        A successful operation with a missing audit record violates
        Tier-1 trust rules - audit data must be 100% pristine.

        - If original operation failed: original exception propagates (DB error logged)
        - If original operation succeeded but audit fails: DB error is raised

    Usage:
        active_source = config.sources[source_name]
        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_id,
            operation_type="source_load",
            ctx=ctx,
            input_data={"source_plugin": active_source.name},
        ) as handle:
            source_iterator = active_source.load(ctx)
            for row_index, source_item in enumerate(source_iterator):
                # ... process rows ...
            # No finally needed - context manager handles everything
            # No output_data for sources (row count tracked elsewhere)

    Args:
        recorder: ExecutionRepository for audit recording
        run_id: Run ID this operation belongs to
        node_id: Source or sink node performing the operation
        operation_type: Type of operation ('source_load', 'sink_write', or
            'runtime_preflight')
        ctx: PluginContext to wire with operation_id
        input_data: Optional input context to record

    Yields:
        OperationHandle with the Operation object and mutable output_data field
    """
    scrubbed_input_data = scrub_payload_for_audit(input_data) if input_data is not None else None
    operation = recorder.begin_operation(
        run_id=run_id,
        node_id=node_id,
        operation_type=operation_type,
        input_data=scrubbed_input_data,
    )

    handle = OperationHandle(operation=operation)

    # Wire context for call recording
    previous_operation_id = ctx.operation_id
    ctx.operation_id = operation.operation_id

    start_time = time.perf_counter()
    status: Literal["completed", "failed"] = "completed"
    error_msg: str | None = None
    original_exception: BaseException | None = None

    try:
        yield handle
    except Exception as e:
        status = "failed"
        error_msg = _render_exception(e)
        original_exception = e
        raise
    except BaseException as e:
        # Catch system interrupts (KeyboardInterrupt, SystemExit, etc.)
        # These are NOT Exception subclasses, so they bypass the above handler.
        # Without this, interrupted operations would be recorded as "completed".
        # Must come AFTER except Exception (more specific handlers first).
        status = "failed"
        error_msg = _render_exception(e)
        original_exception = e
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            scrubbed_output_data = scrub_payload_for_audit(handle.output_data) if handle.output_data is not None else None
            recorder.complete_operation(
                operation_id=operation.operation_id,
                status=status,
                output_data=scrubbed_output_data,
                error=error_msg,
                duration_ms=duration_ms,
            )
        except Exception as db_error:
            # Audit integrity: if we can't record the operation, the run must fail.
            # A successful operation with missing audit record violates Tier-1 trust.
            logger.critical(
                "Failed to complete operation - audit trail incomplete",
                extra={
                    "operation_id": operation.operation_id,
                    "db_error": _render_exception(db_error),
                    "db_error_type": type(db_error).__name__,
                    "original_status": status,
                    "original_error": error_msg,
                },
            )
            # Tier 1 errors (corruption, framework bugs, invariant violations)
            # must ALWAYS propagate regardless of whether there was an original
            # exception — audit corruption is categorically worse than any
            # operation-level error.
            if isinstance(db_error, contract_errors.TIER_1_ERRORS):
                raise db_error from original_exception
            # If there was an original exception, let it propagate (DB error is logged).
            # If the operation succeeded but audit failed, we MUST raise the DB error.
            if original_exception is None:
                raise
            # Otherwise let the original exception propagate (DB error is logged)
        finally:
            # Always restore previous operation_id to prevent accidental reuse
            ctx.operation_id = previous_operation_id
