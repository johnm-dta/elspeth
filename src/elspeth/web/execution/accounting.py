"""Landscape-derived run accounting for the web execution API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from sqlalchemy import and_, func, select

from elspeth.contracts.audit import DISCARD_SINK_NAME
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, run_sources_table, runs_table, token_outcomes_table, tokens_table
from elspeth.web.config import WebSettings
from elspeth.web.execution.discard_summary import _sqlite_database_file_missing, _unique_run_ids
from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
)


def load_run_accounting_for_settings(
    settings: WebSettings,
    landscape_run_ids: Iterable[str | None],
) -> dict[str, RunAccounting]:
    """Load run accounting from the configured Landscape database."""
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    landscape_url = settings.get_landscape_url()
    if _sqlite_database_file_missing(landscape_url):
        return {}

    with LandscapeDB.from_url(
        landscape_url,
        passphrase=settings.landscape_passphrase,
        create_tables=False,
    ) as db:
        return load_run_accounting_map_from_db(db, run_ids)


def load_run_accounting_map_from_db(
    db: LandscapeDB,
    landscape_run_ids: Iterable[str],
) -> dict[str, RunAccounting]:
    """Derive accounting for multiple Landscape runs from an open database."""
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    present_run_ids: tuple[str, ...] = ()

    with db.read_only_connection() as conn:
        present_stmt = select(runs_table.c.run_id).where(runs_table.c.run_id.in_(run_ids)).order_by(runs_table.c.run_id.asc())
        present_run_ids = tuple(str(row.run_id) for row in conn.execute(present_stmt))
        if not present_run_ids:
            return {}

        source_rows_by_source: dict[str, dict[str, int]] = {run_id: {} for run_id in present_run_ids}
        emitted_tokens = _zero_counts(present_run_ids)
        terminal_tokens = _zero_counts(present_run_ids)
        succeeded_tokens = _zero_counts(present_run_ids)
        failed_tokens = _zero_counts(present_run_ids)
        structural_tokens = _zero_counts(present_run_ids)
        routed_success = _zero_counts(present_run_ids)
        routed_failure = _zero_counts(present_run_ids)
        quarantined = _zero_counts(present_run_ids)
        discarded = _zero_counts(present_run_ids)
        missing_terminal_outcomes = _zero_counts(present_run_ids)
        duplicate_terminal_outcomes = _zero_counts(present_run_ids)

        source_stmt = (
            select(
                rows_table.c.run_id,
                func.coalesce(run_sources_table.c.source_name, rows_table.c.source_node_id).label("source_name"),
                func.count().label("count"),
            )
            .select_from(
                rows_table.outerjoin(
                    run_sources_table,
                    and_(
                        run_sources_table.c.run_id == rows_table.c.run_id,
                        run_sources_table.c.source_node_id == rows_table.c.source_node_id,
                    ),
                )
            )
            .where(rows_table.c.run_id.in_(present_run_ids))
            .group_by(rows_table.c.run_id, "source_name")
        )
        for run_id, source_name, count in conn.execute(source_stmt):
            source_rows_by_source[str(run_id)][str(source_name)] = int(count)

        emitted_stmt = (
            select(tokens_table.c.run_id, func.count().label("count"))
            .where(tokens_table.c.run_id.in_(present_run_ids))
            .group_by(tokens_table.c.run_id)
        )
        for run_id, count in conn.execute(emitted_stmt):
            emitted_tokens[str(run_id)] = int(count)

        terminal_stmt = (
            select(
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.sink_name,
                func.count().label("count"),
            )
            .where(token_outcomes_table.c.run_id.in_(present_run_ids))
            .where(token_outcomes_table.c.completed == 1)
            .group_by(
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.sink_name,
            )
        )
        for run_id_value, outcome, path, sink_name, count in conn.execute(terminal_stmt):
            run_id = str(run_id_value)
            value = int(count)
            terminal_tokens[run_id] += value
            if outcome == TerminalOutcome.SUCCESS.value:
                succeeded_tokens[run_id] += value
            elif outcome == TerminalOutcome.FAILURE.value:
                failed_tokens[run_id] += value
            elif outcome == TerminalOutcome.TRANSIENT.value:
                structural_tokens[run_id] += value

            if outcome == TerminalOutcome.SUCCESS.value and path == TerminalPath.GATE_ROUTED.value:
                routed_success[run_id] += value
            elif outcome == TerminalOutcome.FAILURE.value and path == TerminalPath.ON_ERROR_ROUTED.value:
                routed_failure[run_id] += value
            elif outcome == TerminalOutcome.FAILURE.value and path == TerminalPath.QUARANTINED_AT_SOURCE.value:
                quarantined[run_id] += value
            elif outcome == TerminalOutcome.FAILURE.value and (path == TerminalPath.SINK_DISCARDED.value or sink_name == DISCARD_SINK_NAME):
                discarded[run_id] += value

        completed_by_emitted_token = (
            select(
                tokens_table.c.run_id.label("run_id"),
                tokens_table.c.token_id.label("token_id"),
                func.count(token_outcomes_table.c.outcome_id).label("completed_count"),
            )
            .select_from(
                tokens_table.outerjoin(
                    token_outcomes_table,
                    and_(
                        token_outcomes_table.c.run_id == tokens_table.c.run_id,
                        token_outcomes_table.c.token_id == tokens_table.c.token_id,
                        token_outcomes_table.c.completed == 1,
                    ),
                )
            )
            .where(tokens_table.c.run_id.in_(present_run_ids))
            .group_by(tokens_table.c.run_id, tokens_table.c.token_id)
            .subquery()
        )

        missing_stmt = (
            select(completed_by_emitted_token.c.run_id, func.count().label("count"))
            .where(completed_by_emitted_token.c.completed_count == 0)
            .group_by(completed_by_emitted_token.c.run_id)
        )
        for run_id, count in conn.execute(missing_stmt):
            missing_terminal_outcomes[str(run_id)] = int(count)

        duplicate_stmt = (
            select(completed_by_emitted_token.c.run_id, func.count().label("count"))
            .where(completed_by_emitted_token.c.completed_count > 1)
            .group_by(completed_by_emitted_token.c.run_id)
        )
        for run_id, count in conn.execute(duplicate_stmt):
            duplicate_terminal_outcomes[str(run_id)] = int(count)

    accounting: dict[str, RunAccounting] = {}
    for run_id in present_run_ids:
        if duplicate_terminal_outcomes[run_id] > 0:
            raise ValueError(
                "Landscape has duplicate completed terminal outcomes "
                f"for run {run_id!r}: duplicate_terminal_outcomes={duplicate_terminal_outcomes[run_id]}"
            )
        pending_tokens = missing_terminal_outcomes[run_id]
        source_rows = source_rows_by_source[run_id]
        source_row_total = sum(source_rows.values())

        closure: Literal["closed", "open", "unknown"] = (
            "closed"
            if emitted_tokens[run_id] == terminal_tokens[run_id]
            and missing_terminal_outcomes[run_id] == 0
            and duplicate_terminal_outcomes[run_id] == 0
            else "open"
        )
        accounting[run_id] = RunAccounting(
            source=RunAccountingSource(rows_processed=source_row_total),
            sources={source_name: RunAccountingSource(rows_processed=count) for source_name, count in sorted(source_rows.items())},
            tokens=RunAccountingTokens(
                emitted=emitted_tokens[run_id],
                terminal=terminal_tokens[run_id],
                succeeded=succeeded_tokens[run_id],
                failed=failed_tokens[run_id],
                structural=structural_tokens[run_id],
                pending=pending_tokens,
            ),
            routing=RunAccountingRouting(
                routed_success=routed_success[run_id],
                routed_failure=routed_failure[run_id],
                quarantined=quarantined[run_id],
                discarded=discarded[run_id],
            ),
            integrity=RunAccountingIntegrity(
                closure=closure,
                missing_terminal_outcomes=missing_terminal_outcomes[run_id],
                duplicate_terminal_outcomes=duplicate_terminal_outcomes[run_id],
            ),
        )
    return accounting


def load_run_accounting_from_db(db: LandscapeDB, *, landscape_run_id: str) -> RunAccounting:
    """Derive source/token/routing/integrity accounting for one Landscape run."""
    return load_run_accounting_map_from_db(db, (landscape_run_id,))[landscape_run_id]


def _zero_counts(run_ids: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(run_ids, 0)
