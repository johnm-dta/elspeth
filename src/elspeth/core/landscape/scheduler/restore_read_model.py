"""Barrier restore read models over token outcomes.

These queries encode ADR-030 crash-window semantics for journal restore. They
live with scheduler/barrier recovery rather than the generic token-outcome
writer so the persistence layer does not own restore policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select

from elspeth.contracts import NodeStateStatus, TokenOutcome
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.model_loaders import TokenOutcomeLoader
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table, tokens_table

_TOKEN_ID_CHUNK_SIZE = 500


class BarrierRestoreReadModel:
    """Read-only audit queries used by barrier journal restore."""

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

    def get_max_node_state_attempts(
        self,
        run_id: str,
        token_ids: Sequence[str],
        *,
        step_index: int | None = None,
    ) -> dict[str, int]:
        """Max ``node_states.attempt`` per token for resume attempt offsets."""
        result: dict[str, int] = {}
        for i in range(0, len(token_ids), _TOKEN_ID_CHUNK_SIZE):
            chunk = list(token_ids[i : i + _TOKEN_ID_CHUNK_SIZE])
            query = (
                select(node_states_table.c.token_id, func.max(node_states_table.c.attempt).label("max_attempt"))
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.token_id.in_(chunk))
                .group_by(node_states_table.c.token_id)
            )
            if step_index is not None:
                query = query.where(node_states_table.c.step_index == step_index)
            for row in self._ops.execute_fetchall(query):
                result[row.token_id] = int(row.max_attempt)
        return result

    def get_open_node_state_ids(
        self,
        run_id: str,
        *,
        node_ids: Sequence[str],
        token_ids: Sequence[str],
    ) -> dict[str, str]:
        """Outstanding OPEN coalesce-hold node_state ids per token."""
        if not node_ids:
            return {}
        result: dict[str, str] = {}
        for i in range(0, len(token_ids), _TOKEN_ID_CHUNK_SIZE):
            chunk = list(token_ids[i : i + _TOKEN_ID_CHUNK_SIZE])
            query = (
                select(node_states_table.c.token_id, node_states_table.c.state_id)
                .where(node_states_table.c.run_id == run_id)
                .where(node_states_table.c.node_id.in_(list(node_ids)))
                .where(node_states_table.c.token_id.in_(chunk))
                .where(node_states_table.c.status == NodeStateStatus.OPEN.value)
                .order_by(node_states_table.c.token_id, node_states_table.c.attempt)
            )
            for row in self._ops.execute_fetchall(query):
                result[row.token_id] = row.state_id
        return result

    def get_completed_row_ids_for_nodes(
        self,
        run_id: str,
        node_ids: frozenset[str],
    ) -> set[tuple[str, str]]:
        """Completed ``(node_id, row_id)`` pairs for coalesce restore."""
        if not node_ids:
            return set()

        query = (
            select(node_states_table.c.node_id, tokens_table.c.row_id)
            .select_from(
                node_states_table.join(
                    tokens_table,
                    node_states_table.c.token_id == tokens_table.c.token_id,
                )
            )
            .where(
                node_states_table.c.run_id == run_id,
                node_states_table.c.node_id.in_(node_ids),
                node_states_table.c.completed_at.isnot(None),
            )
            .distinct()
        )
        rows = self._ops.execute_fetchall(query)
        return {(row.node_id, row.row_id) for row in rows}

    def has_completed_row_for_node(self, *, run_id: str, node_id: str, row_id: str) -> bool:
        """Return whether one coalesce row completed at one node in one run."""
        query = (
            select(node_states_table.c.state_id)
            .select_from(
                node_states_table.join(
                    tokens_table,
                    node_states_table.c.token_id == tokens_table.c.token_id,
                )
            )
            .where(
                node_states_table.c.run_id == run_id,
                node_states_table.c.node_id == node_id,
                tokens_table.c.row_id == row_id,
                node_states_table.c.completed_at.isnot(None),
            )
            .limit(1)
        )
        return self._ops.execute_fetchone(query) is not None

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
