"""Barrier restore read models over token outcomes.

These queries encode ADR-030 crash-window semantics for journal restore. They
live with scheduler/barrier recovery rather than the generic token-outcome
writer so the persistence layer does not own restore policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select

from elspeth.contracts import TokenOutcome
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.model_loaders import TokenOutcomeLoader
from elspeth.core.landscape.schema import token_outcomes_table


class BarrierRestoreReadModel:
    """Read-only token-outcome queries used by barrier journal restore."""

    def __init__(
        self,
        ops: DatabaseOps,
        *,
        token_outcome_loader: TokenOutcomeLoader,
    ) -> None:
        self._ops = ops
        self._token_outcome_loader = token_outcome_loader

    def list_live_buffered_outcomes(self, ref: TokenRef) -> list[TokenOutcome]:
        """All live BUFFERED outcomes for one token.

        "Live" means the token has no completed outcome; a flushed token's
        BUFFERED row is dead history and exempt. Multiple live rows signal a
        duplicate barrier acceptance that restore must refuse loudly.
        """
        terminal = token_outcomes_table.alias("terminal_outcomes")
        terminal_witness = (
            select(terminal.c.outcome_id)
            .where(terminal.c.token_id == ref.token_id)
            .where(terminal.c.run_id == ref.run_id)
            .where(terminal.c.completed == 1)
            .exists()
        )
        query = (
            select(token_outcomes_table)
            .where(token_outcomes_table.c.token_id == ref.token_id)
            .where(token_outcomes_table.c.run_id == ref.run_id)
            .where(token_outcomes_table.c.completed == 0)
            .where(token_outcomes_table.c.path == TerminalPath.BUFFERED.value)
            .where(~terminal_witness)
            .order_by(token_outcomes_table.c.recorded_at, token_outcomes_table.c.outcome_id)
        )
        return [self._token_outcome_loader.load(row) for row in self._ops.execute_fetchall(query)]

    def find_failed_unrouted_terminal_token_ids(self, run_id: str, token_ids: Sequence[str]) -> frozenset[str]:
        """Token ids holding terminal ``(FAILURE, UNROUTED)`` outcomes.

        This is the ADR-030 aggregation restore reconcile signature for a crash
        after failed-flush terminal writes but before BLOCKED scheduler rows are
        released.
        """
        if not token_ids:
            return frozenset()
        query = (
            select(token_outcomes_table.c.token_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.token_id.in_(tuple(token_ids)))
            .where(token_outcomes_table.c.completed == 1)
            .where(token_outcomes_table.c.outcome == TerminalOutcome.FAILURE.value)
            .where(token_outcomes_table.c.path == TerminalPath.UNROUTED.value)
        )
        return frozenset(row.token_id for row in self._ops.execute_fetchall(query))

    def find_duplicate_live_buffered_acceptances(self, run_id: str) -> list[tuple[str, int]]:
        """Run-wide sweep for tokens with more than one live BUFFERED outcome."""
        terminal = token_outcomes_table.alias("terminal_outcomes")
        terminal_witness = (
            select(terminal.c.outcome_id)
            .where(terminal.c.token_id == token_outcomes_table.c.token_id)
            .where(terminal.c.run_id == run_id)
            .where(terminal.c.completed == 1)
            .exists()
        )
        query = (
            select(token_outcomes_table.c.token_id, func.count())
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.completed == 0)
            .where(token_outcomes_table.c.path == TerminalPath.BUFFERED.value)
            .where(~terminal_witness)
            .group_by(token_outcomes_table.c.token_id)
            .having(func.count() > 1)
            .order_by(token_outcomes_table.c.token_id)
        )
        return [(str(row[0]), int(row[1])) for row in self._ops.execute_fetchall(query)]
