# src/elspeth/core/landscape/_call_recording.py
"""Call and operation recording methods for LandscapeRecorder."""

from __future__ import annotations

import json
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import func, select

from elspeth.contracts import (
    Call,
    CallStatus,
    CallType,
    FrameworkBugError,
    Operation,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    operations_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape._database_ops import DatabaseOps
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.landscape.repositories import CallRepository


class CallRecordingMixin:
    """Call and operation recording methods. Mixed into LandscapeRecorder."""

    # Shared state annotations (set by LandscapeRecorder.__init__)
    _db: LandscapeDB
    _ops: DatabaseOps
    _call_repo: CallRepository
    _payload_store: PayloadStore | None
    _call_indices: dict[str, int]
    _call_index_lock: Lock
    _operation_call_indices: dict[str, int]

    def allocate_call_index(self, state_id: str) -> int:
        """Allocate next call index for a state_id (thread-safe).

        Provides centralized call index allocation ensuring UNIQUE(state_id, call_index)
        across all client types (HTTP, LLM) and retry attempts.

        This is the single source of truth for call numbering. All AuditedClient
        instances MUST delegate to this method rather than maintaining their own counters.

        Thread Safety:
            Uses a lock to prevent race conditions when multiple threads allocate
            indices concurrently. Safe for pooled execution scenarios.

        Persistence:
            Counter seeds from the database on first access per state_id,
            so it survives recorder recreation (e.g., on resume). Subsequent
            allocations are pure in-memory for performance.

        Args:
            state_id: Node state ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this state_id

        Example:
            # Two different client types, same state_id
            http_client = AuditedHTTPClient(recorder, state_id="state-001")
            llm_client = AuditedLLMClient(recorder, state_id="state-001")

            # Both delegate to same recorder - indices coordinate correctly
            http_client.post(...)  # allocates index 0
            llm_client.query(...)   # allocates index 1 (not 0!)
        """
        with self._call_index_lock:
            if state_id not in self._call_indices:
                # Seed from database to survive recorder recreation on resume.
                # Without this, a new recorder would restart indices at 0 for
                # any state_id that already has recorded calls, causing
                # UNIQUE(state_id, call_index) violations.
                row = self._ops.execute_fetchone(select(func.max(calls_table.c.call_index)).where(calls_table.c.state_id == state_id))
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._call_indices[state_id] = existing_max + 1
            idx = self._call_indices[state_id]
            self._call_indices[state_id] += 1
            return idx

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
    ) -> Call:
        """Record an external call for a node state.

        Args:
            state_id: The node_state this call belongs to
            call_index: 0-based index of this call within the state
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (will be hashed, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response

        Returns:
            The recorded Call model

        Note:
            Duplicate (state_id, call_index) will raise IntegrityError from SQLAlchemy.
            Invalid state_id will raise IntegrityError due to foreign key constraint.
            Call indices should be allocated via allocate_call_index() for coordination.
        """
        call_id = generate_id()
        timestamp = now()

        # Hash request (always required)
        request_hash = stable_hash(request_data)

        # Hash response (optional - None for errors without response)
        response_hash = stable_hash(response_data) if response_data is not None else None

        # Auto-persist request to payload store if available and ref not provided
        # This enables replay/verify modes to retrieve the original request
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_data).encode("utf-8")
            request_ref = self._payload_store.store(request_bytes)

        # Auto-persist response to payload store if available and ref not provided
        # This enables replay/verify modes to retrieve the original response
        if response_data is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_data).encode("utf-8")
            response_ref = self._payload_store.store(response_bytes)

        # Serialize error if present
        error_json = canonical_json(error) if error is not None else None

        values = {
            "call_id": call_id,
            "state_id": state_id,
            "operation_id": None,  # State call, not operation call
            "call_index": call_index,
            "call_type": call_type.value,  # Store enum value
            "status": status.value,  # Store enum value
            "request_hash": request_hash,
            "request_ref": request_ref,
            "response_hash": response_hash,
            "response_ref": response_ref,
            "error_json": error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,  # Pass enum directly per Call contract
            status=status,  # Pass enum directly per Call contract
            request_hash=request_hash,
            created_at=timestamp,
            state_id=state_id,  # Parent is node_state
            operation_id=None,  # Not an operation call
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            latency_ms=latency_ms,
        )

    # === Operations (Source/Sink I/O) ===

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: Literal["source_load", "sink_write"],
        *,
        input_data: dict[str, Any] | None = None,
    ) -> Operation:
        """Begin an operation for source/sink I/O.

        Operations are the source/sink equivalent of node_states - they provide
        a parent context for external calls made during load() or write().

        Args:
            run_id: Run this operation belongs to
            node_id: Source or sink node performing the operation
            operation_type: Type of operation
            input_data: Optional input context (stored via payload store)

        Returns:
            Operation with operation_id for call attribution
        """
        from uuid import uuid4

        # Use pure UUID for operation_id - run_id + node_id can exceed 64 chars
        # (run_id=36 + node_id=45 + prefixes would be 94+ chars)
        operation_id = f"op_{uuid4().hex}"  # "op_" + 32 hex = 35 chars, well under 64

        input_ref = None
        input_hash = None
        if input_data:
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
            input_data_ref=input_ref,
            input_data_hash=input_hash,
        )

        self._ops.execute_insert(operations_table.insert().values(**operation.to_dict()))
        return operation

    def complete_operation(
        self,
        operation_id: str,
        status: Literal["completed", "failed", "pending"],
        *,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Complete an operation.

        Args:
            operation_id: Operation to complete
            status: Final status ('completed', 'failed', or 'pending' for BatchPendingError)
            output_data: Optional output context
            error: Error message if failed
            duration_ms: Operation duration

        Raises:
            FrameworkBugError: If operation doesn't exist or is already completed
        """
        # Atomic check-and-update: WHERE constrains both identity and status
        # to eliminate the TOCTOU race between separate SELECT and UPDATE.
        # Payload storage is deferred until AFTER the status check succeeds
        # to avoid orphaned blobs on duplicate-completion races or invalid IDs.
        timestamp = now()
        output_hash = stable_hash(output_data) if output_data else None
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
        with self._ops._db.connection() as conn:
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
            if output_data and self._payload_store is not None:
                output_bytes = canonical_json(output_data).encode("utf-8")
                output_ref = self._payload_store.store(output_bytes)
                conn.execute(
                    operations_table.update().where(operations_table.c.operation_id == operation_id).values(output_data_ref=output_ref)
                )

    def allocate_operation_call_index(self, operation_id: str) -> int:
        """Allocate next call index for an operation_id (thread-safe).

        Provides centralized call index allocation ensuring unique call numbering
        within each operation. Parallel to allocate_call_index() for node_states.

        Args:
            operation_id: Operation ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this operation_id
        """
        with self._call_index_lock:  # Reuse existing lock
            if operation_id not in self._operation_call_indices:
                # Seed from database to survive recorder recreation on resume.
                row = self._ops.execute_fetchone(
                    select(func.max(calls_table.c.call_index)).where(calls_table.c.operation_id == operation_id)
                )
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._operation_call_indices[operation_id] = existing_max + 1
            idx = self._operation_call_indices[operation_id]
            self._operation_call_indices[operation_id] += 1
            return idx

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        provider: str | None = None,
    ) -> Call:
        """Record an external call made during an operation.

        This is the operation equivalent of record_call() - attributes calls
        to operations instead of node_states.

        Args:
            operation_id: The operation this call belongs to
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (will be hashed, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response
            provider: Optional provider name for telemetry

        Returns:
            The recorded Call model
        """
        call_index = self.allocate_operation_call_index(operation_id)
        call_id = f"call_{operation_id}_{call_index}"
        timestamp = now()

        # Hash request (always required)
        request_hash = stable_hash(request_data)

        # Hash response (optional - None for errors without response)
        response_hash = stable_hash(response_data) if response_data is not None else None

        # Auto-persist request to payload store if available and ref not provided
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_data).encode("utf-8")
            request_ref = self._payload_store.store(request_bytes)

        # Auto-persist response to payload store if available and ref not provided
        if response_data is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_data).encode("utf-8")
            response_ref = self._payload_store.store(response_bytes)

        # Serialize error if present
        error_json = canonical_json(error) if error is not None else None

        values = {
            "call_id": call_id,
            "state_id": None,  # NOT a node_state call
            "operation_id": operation_id,  # Operation call
            "call_index": call_index,
            "call_type": call_type.value,
            "status": status.value,
            "request_hash": request_hash,
            "request_ref": request_ref,
            "response_hash": response_hash,
            "response_ref": response_ref,
            "error_json": error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        self._ops.execute_insert(calls_table.insert().values(**values))

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=request_hash,
            created_at=timestamp,
            state_id=None,  # Not a node_state call
            operation_id=operation_id,  # Parent is operation
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            latency_ms=latency_ms,
        )

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

        return Operation(
            operation_id=row.operation_id,
            run_id=row.run_id,
            node_id=row.node_id,
            operation_type=row.operation_type,
            started_at=row.started_at,
            completed_at=row.completed_at,
            status=row.status,
            input_data_ref=row.input_data_ref,
            input_data_hash=row.input_data_hash,
            output_data_ref=row.output_data_ref,
            output_data_hash=row.output_data_hash,
            error_message=row.error_message,
            duration_ms=row.duration_ms,
        )

    def get_operation_calls(self, operation_id: str) -> list[Call]:
        """Get external calls for an operation.

        Args:
            operation_id: Operation ID

        Returns:
            List of Call models, ordered by call_index
        """
        query = select(calls_table).where(calls_table.c.operation_id == operation_id).order_by(calls_table.c.call_index)
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

    def get_operations_for_run(self, run_id: str) -> list[Operation]:
        """Get all operations for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Operation models, ordered by started_at
        """
        query = select(operations_table).where(operations_table.c.run_id == run_id).order_by(operations_table.c.started_at)
        db_rows = self._ops.execute_fetchall(query)
        return [
            Operation(
                operation_id=row.operation_id,
                run_id=row.run_id,
                node_id=row.node_id,
                operation_type=row.operation_type,
                started_at=row.started_at,
                completed_at=row.completed_at,
                status=row.status,
                input_data_ref=row.input_data_ref,
                input_data_hash=row.input_data_hash,
                output_data_ref=row.output_data_ref,
                output_data_hash=row.output_data_hash,
                error_message=row.error_message,
                duration_ms=row.duration_ms,
            )
            for row in db_rows
        ]

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Call]:
        """Get all operation-parented calls for a run (batch query).

        Fetches all calls where operation_id is NOT NULL and the operation
        belongs to the given run. This replaces per-operation get_operation_calls()
        loops in the exporter.

        Args:
            run_id: Run ID

        Returns:
            List of Call models, ordered by operation_id then call_index
        """
        query = (
            select(calls_table)
            .join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)
            .where(operations_table.c.run_id == run_id)
            .order_by(calls_table.c.operation_id, calls_table.c.call_index)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [self._call_repo.load(r) for r in db_rows]

    def find_call_by_request_hash(
        self,
        run_id: str,
        call_type: str,
        request_hash: str,
        *,
        sequence_index: int = 0,
    ) -> Call | None:
        """Find a call by its request hash within a run.

        Used for replay mode to look up previously recorded calls by
        the hash of their request data.

        Args:
            run_id: Run ID to search within
            call_type: Type of call (llm, http, etc.)
            request_hash: SHA-256 hash of the canonical request data
            sequence_index: 0-based index for duplicate request hashes.
                When the same request is made multiple times in a run
                (e.g., retries, loops), use sequence_index to get the
                Nth occurrence (0=first, 1=second, etc.).

        Returns:
            Call model if found, None otherwise

        Note:
            Calls are ordered chronologically by created_at. The sequence_index
            parameter allows disambiguation when the same request was made
            multiple times (each returning a different response).
        """
        # Join to node_states to filter by run_id
        # NOTE: Use node_states.run_id directly (denormalized column) instead of
        # joining through nodes table. The nodes table has composite PK (node_id, run_id),
        # so joining on node_id alone would be ambiguous when node_id is reused across runs.
        query = (
            select(calls_table)
            .join(
                node_states_table,
                calls_table.c.state_id == node_states_table.c.state_id,
            )
            .where(node_states_table.c.run_id == run_id)
            .where(calls_table.c.call_type == call_type)
            .where(calls_table.c.request_hash == request_hash)
            .order_by(calls_table.c.created_at)
            .limit(1)
            .offset(sequence_index)
        )
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._call_repo.load(row)

    def get_call_response_data(self, call_id: str) -> dict[str, Any] | None:
        """Retrieve the response data for a call.

        Fetches response data from the payload store if response_ref is set,
        otherwise returns None.

        Args:
            call_id: The call ID to get response data for

        Returns:
            Response data dict if available, None if no response was recorded
            or if payload store is not configured

        Note:
            Returns None if:
            - Call not found
            - No response_ref set on the call (error calls may not have response)
            - Payload store not configured
            - Response data has been purged from payload store
        """
        # Get the call record first
        query = select(calls_table).where(calls_table.c.call_id == call_id)
        row = self._ops.execute_fetchone(query)

        if row is None:
            return None

        if row.response_ref is None:
            return None

        if self._payload_store is None:
            return None

        try:
            payload_bytes = self._payload_store.retrieve(row.response_ref)
            data: dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))
            return data
        except KeyError:
            # Payload has been purged
            return None
