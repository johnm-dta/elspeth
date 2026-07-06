"""Row/token identity resolution and ownership guards (split from ``DataFlowRepository``).

Owns the Tier-1 identity reads shared across the data-flow components:
resolving which run owns a row, which (row, run) owns a token, and the
cross-run / cross-row contamination guards that must crash before any
dependent write.
"""

from __future__ import annotations

from sqlalchemy import select

from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.schema import rows_table, tokens_table

__all__ = ["RowTokenOwnership"]


class RowTokenOwnership:
    """Tier-1 row/token identity reads and ownership validation guards."""

    def __init__(self, ops: DatabaseOps) -> None:
        self._ops = ops

    def resolve_run_id_for_row(self, row_id: str) -> str:
        """Resolve the run_id that owns a given row_id.

        This is Tier 1 (our data). If the row doesn't exist, it's a bug
        in our code or database corruption -- crash immediately.

        Args:
            row_id: Row ID to look up

        Returns:
            run_id that owns the row

        Raises:
            AuditIntegrityError: If row_id not found (Tier 1 corruption)
        """
        query = select(rows_table.c.run_id).where(rows_table.c.row_id == row_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Token references row_id={row_id!r} which does not exist in the rows table. "
                f"This is Tier 1 data corruption -- the row should have been created before any token."
            )
        run_id: str = result.run_id
        return run_id

    def resolve_row_ingest_sequence(self, row_id: str) -> int:
        """Resolve a row's global ingest ordering for scheduler fairness.

        This is Tier 1 audit data. A token continuation can only exist for a
        persisted row, so a missing row is corruption or an orchestration bug.
        """
        query = select(rows_table.c.ingest_sequence).where(rows_table.c.row_id == row_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Cannot schedule work for row_id={row_id!r}: row does not exist. "
                "This is Tier 1 data corruption -- scheduler work must reference persisted rows."
            )
        ingest_sequence: int = result.ingest_sequence
        return ingest_sequence

    def resolve_token_ownership(self, token_id: str) -> tuple[str, str]:
        """Resolve the (row_id, run_id) that owns a given token_id.

        Looks up token -> row_id, then row -> run_id. This is Tier 1 (our data).
        If the token or its row doesn't exist, it's a bug or database corruption.

        Args:
            token_id: Token ID to look up

        Returns:
            Tuple of (row_id, run_id) that own the token

        Raises:
            AuditIntegrityError: If token or its row not found (Tier 1 corruption)
        """
        query = select(tokens_table.c.row_id, tokens_table.c.run_id).where(tokens_table.c.token_id == token_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Token {token_id!r} does not exist in the tokens table. "
                f"This is Tier 1 data corruption -- the token should have been created before recording outcomes."
            )
        return result.row_id, result.run_id

    def validate_token_run_ownership(self, ref: TokenRef) -> None:
        """Validate that a token belongs to the specified run.

        Per Tier 1 trust model: cross-run contamination of audit records is
        evidence tampering. Crash immediately if the invariant is violated.

        Args:
            ref: TokenRef to validate — token_id must belong to run_id

        Raises:
            AuditIntegrityError: If token does not belong to the specified run
        """
        _row_id, actual_run_id = self.resolve_token_ownership(ref.token_id)
        if actual_run_id != ref.run_id:
            raise AuditIntegrityError(
                f"Cross-run contamination prevented: token {ref.token_id!r} belongs to "
                f"run {actual_run_id!r}, but caller supplied run_id={ref.run_id!r}. "
                f"This would corrupt the audit trail by attributing records to the wrong run."
            )

    def validate_token_row_ownership(self, token_id: str, row_id: str) -> None:
        """Validate that a token belongs to the specified row.

        Per Tier 1 trust model: cross-row lineage corruption makes the audit
        trail unreliable. Crash immediately if the invariant is violated.

        Args:
            token_id: Token to validate
            row_id: Expected row ID

        Raises:
            AuditIntegrityError: If token does not belong to the specified row
        """
        actual_row_id, _run_id = self.resolve_token_ownership(token_id)
        if actual_row_id != row_id:
            raise AuditIntegrityError(
                f"Cross-row lineage corruption prevented: token {token_id!r} belongs to "
                f"row {actual_row_id!r}, but caller supplied row_id={row_id!r}. "
                f"This would create invalid parent-child lineage across different rows."
            )
