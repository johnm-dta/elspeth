"""Validation and transform error audit persistence (split from ``DataFlowRepository``).

Owns the ``validation_errors`` and ``transform_errors`` audit aggregates:
recording quarantine/validation failures and legitimate transform errors
(Tier-3 coerce-and-record on the external row data), row linkage for
persisted quarantine rows, and the error read models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from elspeth.contracts import (
    TransformErrorReason,
    TransformErrorRecord,
    ValidationErrorRecord,
    ValidationErrorWithContract,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.data_flow.ownership import RowTokenOwnership
from elspeth.core.landscape.data_flow.serialization import (
    canonical_or_recorded_error_details_json,
    canonical_or_recorded_hash,
    canonical_or_recorded_json,
)
from elspeth.core.landscape.model_loaders import TransformErrorLoader, ValidationErrorLoader
from elspeth.core.landscape.schema import transform_errors_table, validation_errors_table

if TYPE_CHECKING:
    from elspeth.contracts.errors import ContractViolation
    from elspeth.contracts.schema_contract import PipelineRow

__all__ = ["ErrorAuditRepository"]


class ErrorAuditRepository:
    """Validation and transform error recording and read models."""

    def __init__(
        self,
        ops: DatabaseOps,
        *,
        validation_error_loader: ValidationErrorLoader,
        transform_error_loader: TransformErrorLoader,
        ownership: RowTokenOwnership,
    ) -> None:
        self._ops = ops
        self._validation_error_loader = validation_error_loader
        self._transform_error_loader = transform_error_loader
        self._ownership = ownership

    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        row_id: str | None = None,
        contract_violation: ContractViolation | None = None,
    ) -> str:
        """Record a validation error in the audit trail.

        Called when a source row fails schema validation. The row is
        quarantined (not processed further) but we record what we saw
        for complete audit coverage.

        Args:
            run_id: Current run ID
            node_id: Node where validation failed
            row_data: The row that failed validation (may be non-dict or contain non-finite values)
            error: Error description
            schema_mode: Schema mode that caught the error ("fixed", "flexible", "observed")
            destination: Where row was routed ("discard" or sink name)
            contract_violation: Optional contract violation details for structured auditing

        Returns:
            error_id for tracking
        """
        error_id = f"verr_{generate_id()[:12]}"

        # Tier-3 (external data) trust boundary: row_data may be non-canonical.
        # The coerce-and-record helpers return the canonical representation or an
        # explicit non-canonical fallback recorded in the audit trail.
        row_hash = canonical_or_recorded_hash(row_data)
        row_data_json = canonical_or_recorded_json(row_data)

        # Extract contract violation details if provided
        violation_type: str | None = None
        normalized_field_name: str | None = None
        original_field_name: str | None = None
        expected_type: str | None = None
        actual_type: str | None = None

        if contract_violation is not None:
            violation_record = ValidationErrorWithContract.from_violation(contract_violation)
            violation_type = violation_record.violation_type
            normalized_field_name = violation_record.normalized_field_name
            original_field_name = violation_record.original_field_name
            expected_type = violation_record.expected_type
            actual_type = violation_record.actual_type

        self._ops.execute_insert(
            validation_errors_table.insert().values(
                error_id=error_id,
                run_id=run_id,
                node_id=node_id,
                row_id=row_id,
                row_hash=row_hash,
                row_data_json=row_data_json,
                error=error,
                schema_mode=schema_mode,
                destination=destination,
                created_at=now(),
                violation_type=violation_type,
                normalized_field_name=normalized_field_name,
                original_field_name=original_field_name,
                expected_type=expected_type,
                actual_type=actual_type,
            )
        )

        return error_id

    def link_validation_error_to_row(
        self,
        *,
        run_id: str,
        error_id: str,
        row_id: str,
    ) -> None:
        """Attach a persisted quarantine row to an existing validation error."""
        actual_run_id = self._ownership.resolve_run_id_for_row(row_id)
        if actual_run_id != run_id:
            raise AuditIntegrityError(
                f"Validation error linkage prevented cross-run contamination: row {row_id!r} belongs to "
                f"run {actual_run_id!r}, but caller supplied run_id={run_id!r}."
            )

        error_row = self._ops.execute_fetchone(
            select(
                validation_errors_table.c.run_id,
                validation_errors_table.c.row_id,
            ).where(validation_errors_table.c.error_id == error_id)
        )
        if error_row is None:
            raise AuditIntegrityError(f"Validation error {error_id!r} does not exist in validation_errors. This is Tier 1 data corruption.")
        if error_row.run_id != run_id:
            raise AuditIntegrityError(
                f"Validation error linkage prevented cross-run contamination: error {error_id!r} belongs to "
                f"run {error_row.run_id!r}, but caller supplied run_id={run_id!r}."
            )
        if error_row.row_id is not None:
            if error_row.row_id != row_id:
                raise AuditIntegrityError(
                    f"Validation error {error_id!r} is already linked to row {error_row.row_id!r}; refusing to relink it to {row_id!r}."
                )
            return

        self._ops.execute_update(
            validation_errors_table.update()
            .where(
                validation_errors_table.c.error_id == error_id,
                validation_errors_table.c.run_id == run_id,
            )
            .values(row_id=row_id),
            context="validation_errors.row_id linkage",
        )

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        """Record a transform processing error in the audit trail.

        Called when a transform returns TransformResult.error().
        This is for legitimate errors, NOT transform bugs.

        Validates that the token belongs to the specified run_id before recording.
        Cross-run contamination crashes immediately per Tier 1 trust model.

        Args:
            ref: TokenRef bundling token_id and run_id
            transform_id: Transform that returned the error
            row_data: The row that could not be processed
            error_details: Error details from TransformResult (TransformErrorReason TypedDict)
            destination: Where row was routed ("discard" or sink name)

        Returns:
            error_id for tracking

        Raises:
            AuditIntegrityError: If token does not belong to the specified run
        """
        # Validate token belongs to the specified run (Tier 1 invariant)
        self._ownership.validate_token_run_ownership(ref)

        # Validate reason is a known TransformErrorCategory (Tier 1 write guard).
        # TypedDict has zero runtime enforcement — the Literal annotation only
        # helps at compile time. Invalid reasons must crash before persisting.
        from typing import get_args

        from elspeth.contracts.errors import TransformErrorCategory

        reason = error_details["reason"]
        valid_reasons = get_args(TransformErrorCategory)
        if reason not in valid_reasons:
            raise AuditIntegrityError(
                f"Invalid TransformErrorCategory '{reason}' at Tier 1 write boundary. "
                f"This is a plugin bug — transforms must use a valid error category. "
                f"Valid categories: {sorted(valid_reasons)}"
            )

        error_id = f"terr_{generate_id()[:12]}"

        # error_details may contain NaN/Infinity or non-serializable values
        # (e.g. from exception context in row operations). Tier-3 boundary:
        # error_details originates from transform results which may carry
        # arbitrary row-derived data. The helper returns canonical JSON or an
        # explicit __non_canonical__ envelope recorded in the audit trail.
        error_details_json = canonical_or_recorded_error_details_json(error_details)

        # row_data may contain NaN/Infinity (valid floats that passed source
        # validation). The coerce-and-record helpers return the canonical
        # representation or an explicit non-canonical fallback — losing the
        # error record is worse than recording a repr-based hash.
        row_hash = canonical_or_recorded_hash(row_data)
        row_data_json = canonical_or_recorded_json(row_data)

        self._ops.execute_insert(
            transform_errors_table.insert().values(
                error_id=error_id,
                run_id=ref.run_id,
                token_id=ref.token_id,
                transform_id=transform_id,
                row_hash=row_hash,
                row_data_json=row_data_json,
                error_details_json=error_details_json,
                destination=destination,
                created_at=now(),
            )
        )

        return error_id

    def get_validation_errors_for_row(
        self,
        run_id: str,
        row_hash: str | None = None,
        *,
        row_id: str | None = None,
    ) -> list[ValidationErrorRecord]:
        """Get validation errors for a row by stable row linkage or legacy hash.

        Args:
            run_id: Run ID to query
            row_hash: Legacy hash of the row data (used for historical/fallback lookup)
            row_id: Persisted row identifier for quarantined rows when available

        Returns:
            List of ValidationErrorRecord models
        """
        if row_id is not None:
            row_query = select(validation_errors_table).where(
                validation_errors_table.c.run_id == run_id,
                validation_errors_table.c.row_id == row_id,
            )
            row_rows = self._ops.execute_fetchall(row_query)
            if row_rows or row_hash is None:
                return [self._validation_error_loader.load(r) for r in row_rows]

        if row_hash is None:
            raise ValueError("get_validation_errors_for_row requires row_id or row_hash")

        hash_query = select(validation_errors_table).where(
            validation_errors_table.c.run_id == run_id,
            validation_errors_table.c.row_hash == row_hash,
        )
        hash_rows = self._ops.execute_fetchall(hash_query)
        return [self._validation_error_loader.load(r) for r in hash_rows]

    def get_validation_errors_for_run(self, run_id: str) -> list[ValidationErrorRecord]:
        """Get all validation errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of ValidationErrorRecord models, ordered by created_at
        """
        query = (
            select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._validation_error_loader.load(r) for r in rows]

    def get_transform_errors_for_token(self, token_id: str) -> list[TransformErrorRecord]:
        """Get transform errors for a specific token.

        Args:
            token_id: Token ID to query

        Returns:
            List of TransformErrorRecord models
        """
        query = select(transform_errors_table).where(
            transform_errors_table.c.token_id == token_id,
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_loader.load(r) for r in rows]

    def get_transform_errors_for_run(self, run_id: str) -> list[TransformErrorRecord]:
        """Get all transform errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of TransformErrorRecord models, ordered by created_at
        """
        query = (
            select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_loader.load(r) for r in rows]
