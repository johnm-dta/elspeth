# src/elspeth/core/operations.py
"""Operation lifecycle management for source/sink I/O.

Operations are the source/sink equivalent of node_states - they provide
a parent context for external calls made during source.load() or sink.write().

This module provides the track_operation context manager which handles:
- Operation creation and completion
- Context wiring (ctx.operation_id)
- Duration calculation
- Exception capture with proper status
- Guaranteed completion (even on DB failure)
- Context cleanup (clears operation_id after completion)
- BatchPendingError handling (marks as 'pending', not 'failed')
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from elspeth.contracts import BatchPendingError

if TYPE_CHECKING:
    from elspeth.contracts import Operation
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.plugins.context import PluginContext

logger = logging.getLogger(__name__)


@dataclass
class OperationHandle:
    """Mutable handle for capturing operation output within context manager.

    Allows the caller to set output_data during the operation, which will
    be recorded when the operation completes.

    Usage:
        with track_operation(...) as handle:
            result = sink.write(rows, ctx)
            handle.output_data = {"artifact_path": result.path}  # Explicit!
    """

    operation: Operation
    output_data: dict[str, Any] | None = None


@contextmanager
def track_operation(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    operation_type: Literal["source_load", "sink_write"],
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
        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_id,
            operation_type="source_load",
            ctx=ctx,
            input_data={"source_plugin": config.source.name},
        ) as handle:
            source_iterator = config.source.load(ctx)
            for row_index, source_item in enumerate(source_iterator):
                # ... process rows ...
            # No finally needed - context manager handles everything
            # No output_data for sources (row count tracked elsewhere)

    Args:
        recorder: LandscapeRecorder for audit recording
        run_id: Run ID this operation belongs to
        node_id: Source or sink node performing the operation
        operation_type: Type of operation ('source_load' or 'sink_write')
        ctx: PluginContext to wire with operation_id
        input_data: Optional input context to record

    Yields:
        OperationHandle with the Operation object and mutable output_data field
    """
    operation = recorder.begin_operation(
        run_id=run_id,
        node_id=node_id,
        operation_type=operation_type,
        input_data=input_data,
    )

    handle = OperationHandle(operation=operation)

    # Wire context for call recording
    previous_operation_id = ctx.operation_id
    ctx.operation_id = operation.operation_id

    start_time = time.perf_counter()
    status: Literal["completed", "failed", "pending"] = "completed"
    error_msg: str | None = None
    original_exception: BaseException | None = None

    try:
        yield handle
    except BatchPendingError:
        # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
        # A batch transform needs to wait for async results.
        # Mark as "pending" to distinguish from actual failures.
        status = "pending"
        raise
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        original_exception = e
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            recorder.complete_operation(
                operation_id=operation.operation_id,
                status=status,
                output_data=handle.output_data,
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
                    "db_error": str(db_error),
                    "db_error_type": type(db_error).__name__,
                    "original_status": status,
                    "original_error": error_msg,
                },
            )
            # If there was an original exception, let it propagate (DB error is logged).
            # If the operation succeeded but audit failed, we MUST raise the DB error.
            if original_exception is None:
                raise
            # Otherwise let the original exception propagate (DB error is logged)
        finally:
            # Always restore previous operation_id to prevent accidental reuse
            ctx.operation_id = previous_operation_id
