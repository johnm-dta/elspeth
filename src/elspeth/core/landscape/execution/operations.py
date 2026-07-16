"""Source/sink operation lifecycle persistence (split from ``ExecutionRepository``).

Owns the ``operations`` table: operations are the source/sink equivalent of
node_states — a parent context for external calls made during load() or
write(). Operation-parented call rows themselves live with
:class:`~elspeth.core.landscape.execution.calls.CallAuditRepository`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from sqlalchemy import select

from elspeth.contracts import FrameworkBugError, Operation, OperationType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapePostCommitError
from elspeth.core.landscape.model_loaders import OperationLoader
from elspeth.core.landscape.schema import operations_table

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore

_COMPLETABLE_OPERATION_STATUSES = frozenset({"completed", "failed"})


class OperationRepository:
    """Source/sink operation lifecycle for the audit trail."""

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        operation_loader: OperationLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._operation_loader = operation_loader
        self._payload_store = payload_store

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: OperationType,
        *,
        input_data: Mapping[str, object] | None = None,
        sink_effect_id: str | None = None,
    ) -> Operation:
        """Begin an operation for source/sink I/O.

        Operations are the source/sink equivalent of node_states - they provide
        a parent context for external calls made during load() or write().

        Args:
            run_id: Run this operation belongs to
            node_id: Source or sink node performing the operation
            operation_type: Type of operation
            input_data: Optional input context (stored via payload store)
            sink_effect_id: Stable epoch-26 effect identity for sink writes

        Returns:
            Operation with operation_id for call attribution
        """
        # Use pure UUID for operation_id - run_id + node_id can exceed 64 chars
        # (run_id=36 + node_id=45 + prefixes would be 94+ chars)
        operation_id = f"op_{uuid4().hex}"  # "op_" + 32 hex = 35 chars, well under 64

        input_ref = None
        input_hash = None
        if input_data is not None:
            input_hash = stable_hash(input_data)
            if self._payload_store is not None:
                input_bytes = canonical_json(input_data).encode("utf-8")
                input_ref = self._payload_store.store(input_bytes)

        timestamp = now()
        operation = Operation(
            operation_id=operation_id,
            run_id=run_id,
            node_id=node_id,
            operation_type=operation_type,
            started_at=timestamp,
            status="open",
            sink_effect_id=sink_effect_id,
            input_data_ref=input_ref,
            input_data_hash=input_hash,
        )

        self._ops.execute_insert(operations_table.insert().values(**operation.to_dict()))
        return operation

    def complete_operation(
        self,
        operation_id: str,
        status: Literal["completed", "failed"],
        *,
        output_data: Mapping[str, object] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Complete an operation.

        Args:
            operation_id: Operation to complete
            status: Final status ('completed' or 'failed')
            output_data: Optional output context
            error: Error message if failed
            duration_ms: Operation duration

        Raises:
            FrameworkBugError: If operation doesn't exist or is already completed
        """
        # Validate lifecycle invariants before persisting — matches Operation.__post_init__.
        # These are Tier 1 guards: impossible states in the audit trail are framework bugs.
        if status not in _COMPLETABLE_OPERATION_STATUSES:
            raise FrameworkBugError(
                f"complete_operation({operation_id!r}): unsupported status {status!r}; "
                f"expected one of {sorted(_COMPLETABLE_OPERATION_STATUSES)}"
            )
        if duration_ms is None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status={status!r} but duration_ms is None")
        if status == "completed" and error is not None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='completed' but error is set")
        if status == "failed" and error is None:
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='failed' but error is None")
        if status == "failed" and error == "":
            raise FrameworkBugError(f"complete_operation({operation_id!r}): status='failed' but error must not be empty")

        # Atomic check-and-update: WHERE constrains both identity and status
        # to eliminate the TOCTOU race between separate SELECT and UPDATE.
        # Payload storage is deferred until AFTER the status check succeeds
        # to avoid orphaned blobs on duplicate-completion races or invalid IDs.
        timestamp = now()
        output_hash = stable_hash(output_data) if output_data is not None else None
        row = None
        stmt = (
            operations_table.update()
            .where((operations_table.c.operation_id == operation_id) & (operations_table.c.status == "open"))
            .values(
                completed_at=timestamp,
                status=status,
                error_message=error,
                duration_ms=duration_ms,
                output_data_hash=output_hash,
            )
        )
        with self._db.write_connection() as conn:
            result = conn.execute(stmt)
            if result.rowcount == 0:
                # Distinguish "doesn't exist" from "already completed" for diagnostics
                check = conn.execute(select(operations_table.c.status).where(operations_table.c.operation_id == operation_id)).fetchone()
                if check is None:
                    raise FrameworkBugError(f"Completing non-existent operation: {operation_id}")
                raise FrameworkBugError(
                    f"Completing already-completed operation {operation_id}: current status={check.status}, new status={status}"
                )

            # Store payload only after confirming the operation row was updated
            if output_data is not None and self._payload_store is not None:
                output_bytes = canonical_json(output_data).encode("utf-8")
                output_ref = self._payload_store.store(output_bytes)
                ref_result = conn.execute(
                    operations_table.update().where(operations_table.c.operation_id == operation_id).values(output_data_ref=output_ref)
                )
                if ref_result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"complete_operation: output_data_ref UPDATE affected zero rows for "
                        f"operation {operation_id} — row disappeared between status update "
                        f"and ref update (database corruption)"
                    )
            row = conn.execute(select(operations_table).where(operations_table.c.operation_id == operation_id)).fetchone()

        if row is None:
            raise LandscapePostCommitError(
                f"Operation {operation_id} not found after completion — database corruption or transaction failure"
            )
        try:
            self._operation_loader.load(row)
        except AuditIntegrityError as exc:
            raise LandscapePostCommitError(f"Operation {operation_id} became unreadable immediately after completion: {exc}") from exc

    def get_operation(self, operation_id: str) -> Operation | None:
        """Get an operation by ID.

        Args:
            operation_id: Operation ID to retrieve

        Returns:
            Operation model or None if not found
        """
        query = select(operations_table).where(operations_table.c.operation_id == operation_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None

        return self._operation_loader.load(row)

    def get_operations_for_run(self, run_id: str) -> list[Operation]:
        """Get all operations for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Operation models, ordered by started_at
        """
        query = select(operations_table).where(operations_table.c.run_id == run_id).order_by(operations_table.c.started_at)
        db_rows = self._ops.execute_fetchall(query)
        return [self._operation_loader.load(row) for row in db_rows]
