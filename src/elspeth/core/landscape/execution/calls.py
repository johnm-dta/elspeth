"""External call audit persistence (split from ``ExecutionRepository``).

Owns every write and read of the ``calls`` table — both node-state-parented
and operation-parented calls — plus thread-safe call index allocation
(one Lock + per-state and per-operation dicts), database-owned collision
recovery across independent recorders, and payload staging / post-insert
materialization into the payload store.
"""

from __future__ import annotations

import hashlib
import json
from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import structlog
from sqlalchemy import Insert, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Call, CallStatus, CallType, FrameworkBugError
from elspeth.contracts.audit import validate_resolved_prompt_template_hash
from elspeth.contracts.call_data import CallPayload
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.payload_store import IntegrityError as PayloadIntegrityError
from elspeth.contracts.payload_store import PayloadNotFoundError
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.ids import generate_id
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.errors import LandscapePostCommitError, LandscapeRecordError
from elspeth.core.landscape.model_loaders import CallLoader
from elspeth.core.landscape.row_data import CallDataResult, CallDataState
from elspeth.core.landscape.schema import calls_table, node_states_table, operations_table

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape.database import LandscapeDB


logger = structlog.get_logger(__name__)


class _PreparedCallData(NamedTuple):
    """Intermediate result from _prepare_call_payloads.

    Carries stable hashes plus any payload bytes that still need post-insert
    materialization into the payload store.
    """

    request_hash: str
    request_ref: str | None
    response_hash: str | None
    response_ref: str | None
    error_json: str | None
    request_bytes: bytes | None
    response_bytes: bytes | None


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


class CallAuditRepository:
    """External call recording and call index allocation for the audit trail."""

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        call_loader: CallLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._call_loader = call_loader
        self._payload_store = payload_store

        # Thread-safe call index allocation (internal state, not injected)
        self._call_indices: dict[str, int] = {}
        self._call_index_lock = Lock()
        self._operation_call_indices: dict[str, int] = {}
        self._pending_call_indices: set[tuple[str, int]] = set()
        self._pending_operation_call_indices: set[tuple[str, int]] = set()

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
            allocations are process-local proposals. ``record_call`` remembers
            which proposals came from this repository and atomically remaps a
            uniqueness collision at INSERT time, after the external effect,
            without asking the caller to repeat that effect.

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
                # Slow path (once per state_id): seed from database to survive
                # recorder recreation on resume. Without this, a new recorder
                # would restart indices at 0 for any state_id that already has
                # recorded calls, causing UNIQUE(state_id, call_index) violations.
                # The DB query is serialized under the lock — acceptable because
                # it only fires once per state_id per recorder lifetime. All
                # subsequent allocations hit the fast path (no DB access).
                row = self._ops.execute_fetchone(select(func.max(calls_table.c.call_index)).where(calls_table.c.state_id == state_id))
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._call_indices[state_id] = existing_max + 1
            # Fast path: allocate from in-memory counter (no DB access)
            idx = self._call_indices[state_id]
            self._call_indices[state_id] += 1
            self._pending_call_indices.add((state_id, idx))
            return idx

    def allocate_operation_call_index(self, operation_id: str) -> int:
        """Allocate next call index for an operation_id (thread-safe).

        Provides process-local call index proposals within each operation.
        ``record_operation_call`` resolves a cross-process proposal collision
        at the database INSERT boundary. Parallel to ``allocate_call_index``
        for node states.

        Args:
            operation_id: Operation ID to allocate index for

        Returns:
            Sequential call index (0-based), unique within this operation_id
        """
        with self._call_index_lock:  # Reuse existing lock
            if operation_id not in self._operation_call_indices:
                # Slow path (once per operation_id): seed from database to survive
                # recorder recreation on resume. Serialized under lock — acceptable
                # because it fires only once per operation_id per recorder lifetime.
                row = self._ops.execute_fetchone(
                    select(func.max(calls_table.c.call_index)).where(calls_table.c.operation_id == operation_id)
                )
                existing_max = row[0] if row is not None and row[0] is not None else -1
                self._operation_call_indices[operation_id] = existing_max + 1
            # Fast path: allocate from in-memory counter (no DB access)
            idx = self._operation_call_indices[operation_id]
            self._operation_call_indices[operation_id] += 1
            self._pending_operation_call_indices.add((operation_id, idx))
            return idx

    @staticmethod
    def _collision_tolerant_insert(
        conn: Connection,
        values: dict[str, object],
        *,
        parent_column: str,
    ) -> Insert:
        """Suppress only the parent/index uniqueness collision."""
        column = calls_table.c[parent_column]
        if conn.dialect.name == "sqlite":
            return (
                sqlite_insert(calls_table)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=[column, calls_table.c.call_index],
                    index_where=column.is_not(None),
                )
            )
        if conn.dialect.name == "postgresql":
            return (
                postgresql_insert(calls_table)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=[column, calls_table.c.call_index],
                    index_where=column.is_not(None),
                )
            )
        raise LandscapeRecordError(
            f"call-index collision recovery is unsupported for database dialect {conn.dialect.name!r}; refusing an ambiguous audit write"
        )

    def _insert_allocated_call(
        self,
        values: dict[str, object],
        *,
        parent_column: str,
        parent_id: str,
        allocation_is_owned: bool,
    ) -> dict[str, object]:
        """Insert once, remapping only a repository-allocated index collision."""
        proposed_index = values["call_index"]
        if not isinstance(proposed_index, int):
            raise FrameworkBugError("prepared call_index must be an integer")

        try:
            with self._db.write_connection() as conn:
                if not allocation_is_owned:
                    conn.execute(calls_table.insert().values(**values))
                    return values

                candidate = proposed_index
                for _attempt in range(1_000):
                    candidate_values = dict(values)
                    candidate_values["call_index"] = candidate
                    if parent_column == "operation_id":
                        candidate_values["call_id"] = f"call_{parent_id}_{candidate}"
                    inserted_id = conn.execute(
                        self._collision_tolerant_insert(
                            conn,
                            candidate_values,
                            parent_column=parent_column,
                        ).returning(calls_table.c.call_id)
                    ).scalar_one_or_none()
                    if inserted_id is not None:
                        return candidate_values

                    existing_max = conn.execute(
                        select(func.max(calls_table.c.call_index)).where(calls_table.c[parent_column] == parent_id)
                    ).scalar_one()
                    candidate = (existing_max if existing_max is not None else -1) + 1
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"record_call failed for {parent_column}={parent_id!r} — database rejected audit write: {type(exc).__name__}"
            ) from exc

        raise LandscapeRecordError(f"record_call could not allocate a durable index for {parent_column}={parent_id!r} after 1000 conflicts")

    def _allocation_context(
        self,
        *,
        parent_id: str,
        call_index: int,
        operation: bool,
    ) -> bool:
        """Return whether this repository allocated the proposed index."""
        with self._call_index_lock:
            pending = self._pending_operation_call_indices if operation else self._pending_call_indices
            return (parent_id, call_index) in pending

    def _finish_allocation(
        self,
        *,
        parent_id: str,
        proposed_index: int,
        recorded_index: int | None,
        operation: bool,
    ) -> None:
        """Release the local reservation and advance the cache past remaps."""
        with self._call_index_lock:
            pending = self._pending_operation_call_indices if operation else self._pending_call_indices
            pending.discard((parent_id, proposed_index))
            if recorded_index is None:
                return
            counters = self._operation_call_indices if operation else self._call_indices
            counters[parent_id] = max(counters.get(parent_id, 0), recorded_index + 1)

    def _prepare_call_payloads(
        self,
        request_data: CallPayload,
        response_data: CallPayload | None,
        error: CallPayload | None,
        request_ref: str | None,
        response_ref: str | None,
    ) -> _PreparedCallData:
        """Serialize, hash, and stage call payloads for post-insert storage.

        Shared logic for record_call() and record_operation_call(). Converts
        CallPayload objects to dicts, computes stable hashes, prepares any
        payload bytes that still need storing after the call row exists, and
        serializes the error payload.
        """
        request_dict = request_data.to_dict()
        response_dict = response_data.to_dict() if response_data is not None else None

        request_hash = stable_hash(request_dict)
        response_hash = stable_hash(response_dict) if response_dict is not None else None

        request_bytes = None
        response_bytes = None
        if request_ref is None and self._payload_store is not None:
            request_bytes = canonical_json(request_dict).encode("utf-8")
        if response_dict is not None and response_ref is None and self._payload_store is not None:
            response_bytes = canonical_json(response_dict).encode("utf-8")

        error_json = canonical_json(error.to_dict()) if error is not None else None

        return _PreparedCallData(
            request_hash=request_hash,
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
        )

    def _materialize_call_ref_after_insert(
        self,
        *,
        call_id: str,
        column_name: str,
        payload_bytes: bytes | None,
    ) -> str | None:
        """Store one call payload and update the already-recorded call row."""
        if payload_bytes is None:
            return None
        if self._payload_store is None:
            raise FrameworkBugError(f"_materialize_call_ref_after_insert({call_id!r}, {column_name!r}) requires a payload store")

        try:
            payload_ref = self._payload_store.store(payload_bytes)
            self._ops.execute_update(
                calls_table.update().where(calls_table.c.call_id == call_id).values(**{column_name: payload_ref}),
                context=f"calls.{column_name} for {call_id}",
            )
        except Exception as exc:
            raise LandscapePostCommitError(
                f"Call {call_id} was recorded, but {column_name} materialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        return payload_ref

    def _materialize_call_refs_after_insert(self, call_id: str, prepared: _PreparedCallData) -> tuple[str | None, str | None]:
        """Store staged call payloads after the call row has been committed."""
        request_ref = prepared.request_ref
        response_ref = prepared.response_ref

        materialized_request_ref = self._materialize_call_ref_after_insert(
            call_id=call_id,
            column_name="request_ref",
            payload_bytes=prepared.request_bytes,
        )
        if materialized_request_ref is not None:
            request_ref = materialized_request_ref

        materialized_response_ref = self._materialize_call_ref_after_insert(
            call_id=call_id,
            column_name="response_ref",
            payload_bytes=prepared.response_bytes,
        )
        if materialized_response_ref is not None:
            response_ref = materialized_response_ref

        return request_ref, response_ref

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        """Record an external call for a node state.

        Args:
            state_id: The node_state this call belongs to
            call_index: 0-based index of this call within the state
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (CallPayload — serialized internally)
            response_data: Response payload (CallPayload — serialized internally, optional for errors)
            error: Error payload if status is ERROR (CallPayload — serialized internally)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response
            resolved_prompt_template_hash: Cross-DB hash anchor (Phase 5b Task 9).
                When this LLM-transform call is downstream of an interpretation
                event the L3 plugin forwards the SHA-256 of the resolved prompt
                template string here; the value MUST equal
                ``interpretation_events.resolved_prompt_template_hash`` in the
                session audit DB for the same resolved string. ``None`` for
                non-LLM calls or for LLM transforms not downstream of an
                interpretation event.

        Returns:
            The recorded Call model

        Note:
            A duplicate repository-allocated ``(state_id, call_index)`` is
            remapped atomically because the represented external effect has
            already occurred. A duplicate explicit index that bypassed this
            repository's allocator remains an integrity error.
            Invalid state_id will raise IntegrityError due to foreign key constraint.
            Call indices should be allocated via allocate_call_index() for coordination.
        """
        # Validate the cross-DB prompt-hash anchor BEFORE inserting. The Call
        # dataclass enforces the same invariant in __post_init__, but that runs
        # AFTER execute_insert — a bad hash would commit to `calls` and only then
        # raise. Checking here keeps the audit trail pristine (Tier 1): a bad
        # hash leaves zero rows (elspeth-a94e626a36).
        validate_resolved_prompt_template_hash(call_type, resolved_prompt_template_hash)

        call_id = generate_id()
        timestamp = now()
        prepared = self._prepare_call_payloads(
            request_data,
            response_data,
            error,
            request_ref,
            response_ref,
        )

        values: dict[str, object] = {
            "call_id": call_id,
            "state_id": state_id,
            "operation_id": None,  # State call, not operation call
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_hash": prepared.request_hash,
            "request_ref": prepared.request_ref,
            "response_hash": prepared.response_hash,
            "response_ref": prepared.response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
            "error_json": prepared.error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        proposed_call_index = call_index
        allocation_is_owned = self._allocation_context(
            parent_id=state_id,
            call_index=proposed_call_index,
            operation=False,
        )
        recorded_index: int | None = None
        try:
            values = self._insert_allocated_call(
                values,
                parent_column="state_id",
                parent_id=state_id,
                allocation_is_owned=allocation_is_owned,
            )
            call_id = str(values["call_id"])
            recorded_value = values["call_index"]
            if type(recorded_value) is not int:
                raise FrameworkBugError("inserted state call returned a non-integer call_index")
            recorded_index = recorded_value
            call_index = recorded_index
        finally:
            self._finish_allocation(
                parent_id=state_id,
                proposed_index=proposed_call_index,
                recorded_index=recorded_index,
                operation=False,
            )
        request_ref, response_ref = self._materialize_call_refs_after_insert(call_id, prepared)

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=prepared.request_hash,
            created_at=timestamp,
            state_id=state_id,
            operation_id=None,
            request_ref=request_ref,
            response_hash=prepared.response_hash,
            response_ref=response_ref,
            error_json=prepared.error_json,
            latency_ms=latency_ms,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        """Record an external call made during an operation.

        This is the operation equivalent of record_call() - attributes calls
        to operations instead of node_states.

        Args:
            operation_id: The operation this call belongs to
            call_type: Type of external call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome of the call (SUCCESS, ERROR)
            request_data: Request payload (CallPayload — serialized internally)
            response_data: Response payload (CallPayload — serialized internally, optional for errors)
            error: Error details if status is ERROR (stored as JSON)
            latency_ms: Call duration in milliseconds
            request_ref: Optional payload store reference for request
            response_ref: Optional payload store reference for response

        Returns:
            The recorded Call model
        """
        # Validate the cross-DB prompt-hash anchor BEFORE inserting (see the
        # state-parented record_call for rationale): a bad hash must leave zero
        # `calls` rows rather than commit and then raise from Call.__post_init__
        # (elspeth-a94e626a36).
        validate_resolved_prompt_template_hash(call_type, resolved_prompt_template_hash)

        if call_index is None:
            call_index = self.allocate_operation_call_index(operation_id)
        call_id = f"call_{operation_id}_{call_index}"
        timestamp = now()
        prepared = self._prepare_call_payloads(
            request_data,
            response_data,
            error,
            request_ref,
            response_ref,
        )

        values: dict[str, object] = {
            "call_id": call_id,
            "state_id": None,  # NOT a node_state call
            "operation_id": operation_id,  # Operation call
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_hash": prepared.request_hash,
            "request_ref": prepared.request_ref,
            "response_hash": prepared.response_hash,
            "response_ref": prepared.response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
            "error_json": prepared.error_json,
            "latency_ms": latency_ms,
            "created_at": timestamp,
        }

        proposed_call_index = call_index
        allocation_is_owned = self._allocation_context(
            parent_id=operation_id,
            call_index=proposed_call_index,
            operation=True,
        )
        recorded_index = None
        try:
            values = self._insert_allocated_call(
                values,
                parent_column="operation_id",
                parent_id=operation_id,
                allocation_is_owned=allocation_is_owned,
            )
            call_id = str(values["call_id"])
            recorded_value = values["call_index"]
            if type(recorded_value) is not int:
                raise FrameworkBugError("inserted operation call returned a non-integer call_index")
            recorded_index = recorded_value
            call_index = recorded_index
        finally:
            self._finish_allocation(
                parent_id=operation_id,
                proposed_index=proposed_call_index,
                recorded_index=recorded_index,
                operation=True,
            )
        request_ref, response_ref = self._materialize_call_refs_after_insert(call_id, prepared)

        return Call(
            call_id=call_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=prepared.request_hash,
            created_at=timestamp,
            state_id=None,
            operation_id=operation_id,
            request_ref=request_ref,
            response_hash=prepared.response_hash,
            response_ref=response_ref,
            error_json=prepared.error_json,
            latency_ms=latency_ms,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
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
        return [self._call_loader.load(r) for r in db_rows]

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
        return [self._call_loader.load(r) for r in db_rows]

    def find_call_by_request_hash(
        self,
        run_id: str,
        call_type: CallType,
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
        return self._call_loader.load(row)

    def get_call_response_data(self, call_id: str) -> CallDataResult:
        """Retrieve the response data for a call with explicit state.

        Returns a CallDataResult with explicit state indicating why data
        may be unavailable. Callers match on state instead of guessing
        why the previous `None` return occurred.

        Args:
            call_id: The call ID to get response data for

        Returns:
            CallDataResult with state and data (if available)
        """
        # Get the call record first
        query = select(calls_table).where(calls_table.c.call_id == call_id)
        row = self._ops.execute_fetchone(query)

        if row is None:
            return CallDataResult(state=CallDataState.CALL_NOT_FOUND, data=None)

        if row.response_ref is None:
            if row.response_hash is not None:
                return CallDataResult(state=CallDataState.HASH_ONLY, data=None)
            return CallDataResult(state=CallDataState.NEVER_STORED, data=None)

        if self._payload_store is None:
            return CallDataResult(state=CallDataState.STORE_NOT_CONFIGURED, data=None)

        # Retrieve from payload store — PayloadNotFoundError means purged by
        # retention policy, PayloadIntegrityError means hash mismatch
        # (corruption/tampering), OSError means storage backend failure
        # (permissions, disk, etc.). All non-purge paths translate to
        # AuditIntegrityError with context, matching
        # query_repository._retrieve_and_parse_payload().
        try:
            payload_bytes = self._payload_store.retrieve(row.response_ref)
        except PayloadNotFoundError:
            return CallDataResult(state=CallDataState.PURGED, data=None)
        except PayloadIntegrityError as e:
            raise AuditIntegrityError(f"Payload integrity check failed for call_id={call_id} (ref={row.response_ref}): {e}") from e
        except OSError as e:
            logger.warning(
                "call_response_payload_retrieval_failed",
                call_id=call_id,
                response_ref=row.response_ref,
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise AuditIntegrityError(f"Payload retrieval failed for call_id={call_id}: reason=payload_store_os_error") from e

        # Everything below is Tier 1: our data, crash on anomaly
        try:
            decoded = json.loads(payload_bytes.decode("utf-8"), parse_constant=_reject_non_finite_json_constant)
        except (UnicodeDecodeError, ValueError) as e:
            raise AuditIntegrityError(f"Corrupt call response payload for call_id={call_id} (ref={row.response_ref}): {e}") from e
        if type(decoded) is not dict:
            raise AuditIntegrityError(
                f"Corrupt call response payload for call_id={call_id} (ref={row.response_ref}): "
                f"expected JSON object, got {type(decoded).__name__}"
            )
        response_hash = row.response_hash
        if response_hash is None:
            raise AuditIntegrityError(f"Call response payload ref without response hash for call_id={call_id} (ref={row.response_ref})")
        decoded_hash = hashlib.sha256(payload_bytes).hexdigest()
        if decoded_hash != response_hash:
            raise AuditIntegrityError(f"Call response payload hash mismatch for call_id={call_id} (ref={row.response_ref})")
        return CallDataResult(state=CallDataState.AVAILABLE, data=decoded)
