"""Tests for Landscape-derived web run accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.enums import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import run_sources_table, token_outcomes_table, tokens_table
from elspeth.web.execution.accounting import (
    load_run_accounting_for_settings,
    load_run_accounting_from_db,
    load_run_accounting_map_from_db,
)

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_NOW = datetime(2026, 5, 6, tzinfo=UTC)


@dataclass(frozen=True)
class _SettingsFake:
    landscape_url: str
    landscape_passphrase: str | None = None

    def get_landscape_url(self) -> str:
        return self.landscape_url


def _setup_run_with_row(db: LandscapeDB, *, run_id: str, row_id: str = "row-1") -> None:
    factory = RecorderFactory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        node_id="source",
        plugin_name="text",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_OBSERVED_SCHEMA,
    )
    factory.data_flow.create_row(run_id, "source", 0, {"value": "input"}, row_id=row_id, source_row_index=0, ingest_sequence=0)


def _setup_empty_run(db: LandscapeDB, *, run_id: str) -> None:
    RecorderFactory(db).run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)


def _insert_tokens(db: LandscapeDB, *, run_id: str, row_id: str, token_ids: list[str]) -> None:
    with db.write_connection() as conn:
        conn.execute(
            tokens_table.insert(),
            [
                {
                    "token_id": token_id,
                    "row_id": row_id,
                    "run_id": run_id,
                    "fork_group_id": None,
                    "join_group_id": None,
                    "expand_group_id": None,
                    "branch_name": None,
                    "step_in_pipeline": 0,
                    "created_at": _NOW,
                }
                for token_id in token_ids
            ],
        )


def _insert_completed_outcomes(
    db: LandscapeDB,
    *,
    run_id: str,
    token_ids: list[str],
    outcome: TerminalOutcome,
    path: TerminalPath,
) -> None:
    with db.write_connection() as conn:
        conn.execute(
            token_outcomes_table.insert(),
            [
                {
                    "outcome_id": f"outcome-{token_id}-{path.value}",
                    "run_id": run_id,
                    "token_id": token_id,
                    "outcome": outcome.value,
                    "path": path.value,
                    "completed": 1,
                    "recorded_at": _NOW,
                    "sink_name": None,
                    "batch_id": None,
                    "fork_group_id": None,
                    "join_group_id": None,
                    "expand_group_id": None,
                    "error_hash": None,
                    "context_json": None,
                    "expected_branches_json": None,
                }
                for token_id in token_ids
            ],
        )


def _insert_outcome(
    db: LandscapeDB,
    *,
    run_id: str,
    token_id: str,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    completed: int,
) -> None:
    with db.write_connection() as conn:
        conn.execute(
            token_outcomes_table.insert().values(
                outcome_id=f"outcome-{token_id}-{path.value}-{completed}",
                run_id=run_id,
                token_id=token_id,
                outcome=outcome.value if outcome is not None else None,
                path=path.value,
                completed=completed,
                recorded_at=_NOW,
                sink_name=None,
                batch_id=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                error_hash=None,
                context_json=None,
                expected_branches_json=None,
            )
        )


def test_one_source_row_expands_to_many_tokens_and_closes() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-1")
        success_token_ids = [f"child-{i}" for i in range(9323)]
        _insert_tokens(db, run_id="run-1", row_id="row-1", token_ids=["parent", *success_token_ids])
        _insert_completed_outcomes(
            db,
            run_id="run-1",
            token_ids=success_token_ids,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        _insert_completed_outcomes(
            db,
            run_id="run-1",
            token_ids=["parent"],
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.EXPAND_PARENT,
        )

        accounting = load_run_accounting_from_db(db, landscape_run_id="run-1")

        assert accounting.source.rows_processed == 1
        assert accounting.tokens.emitted == 9324
        assert accounting.tokens.terminal == 9324
        assert accounting.tokens.succeeded == 9323
        assert accounting.tokens.failed == 0
        assert accounting.tokens.structural == 1
        assert accounting.tokens.pending == 0
        assert accounting.integrity.closure == "closed"
        assert accounting.integrity.missing_terminal_outcomes == 0
        assert accounting.integrity.duplicate_terminal_outcomes == 0
    finally:
        db.close()


def test_source_accounting_projects_named_sources_and_aggregate_total() -> None:
    db = LandscapeDB.in_memory()
    try:
        factory = RecorderFactory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-multi")
        for source_name, source_node_id in (("orders", "source-orders"), ("refunds", "source-refunds")):
            factory.data_flow.register_node(
                run_id="run-multi",
                node_id=source_node_id,
                plugin_name="csv",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={},
                schema_config=_OBSERVED_SCHEMA,
            )
            with db.write_connection() as conn:
                conn.execute(
                    run_sources_table.insert().values(
                        run_id="run-multi",
                        source_node_id=source_node_id,
                        source_name=source_name,
                        plugin_name="csv",
                        lifecycle_state="loaded",
                        config_hash=f"hash-{source_name}",
                        schema_json=None,
                        schema_contract_json=None,
                        schema_contract_hash=None,
                        field_resolution_json=None,
                        recorded_at=_NOW,
                    )
                )

        factory.data_flow.create_row(
            "run-multi",
            "source-orders",
            0,
            {"id": "order-1"},
            row_id="orders-0",
            source_row_index=0,
            ingest_sequence=0,
        )
        factory.data_flow.create_row(
            "run-multi",
            "source-orders",
            1,
            {"id": "order-2"},
            row_id="orders-1",
            source_row_index=1,
            ingest_sequence=1,
        )
        factory.data_flow.create_row(
            "run-multi",
            "source-refunds",
            2,
            {"id": "refund-1"},
            row_id="refunds-0",
            source_row_index=0,
            ingest_sequence=2,
        )

        accounting = load_run_accounting_from_db(db, landscape_run_id="run-multi")

        assert accounting.source.rows_processed == 3
        assert accounting.sources["orders"].rows_processed == 2
        assert accounting.sources["refunds"].rows_processed == 1
    finally:
        db.close()


def test_missing_completed_terminal_outcome_marks_token_pending_and_open() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-2")
        _insert_tokens(db, run_id="run-2", row_id="row-1", token_ids=["token-1"])

        accounting = load_run_accounting_from_db(db, landscape_run_id="run-2")

        assert accounting.tokens.emitted == 1
        assert accounting.tokens.terminal == 0
        assert accounting.tokens.pending == 1
        assert accounting.integrity.missing_terminal_outcomes == 1
        assert accounting.integrity.duplicate_terminal_outcomes == 0
        assert accounting.integrity.closure == "open"
    finally:
        db.close()


def test_batch_loader_returns_accounting_for_requested_runs() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-a", row_id="row-a")
        _insert_tokens(db, run_id="run-a", row_id="row-a", token_ids=["token-a"])
        _insert_completed_outcomes(
            db,
            run_id="run-a",
            token_ids=["token-a"],
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        _setup_run_with_row(db, run_id="run-b", row_id="row-b")
        _insert_tokens(db, run_id="run-b", row_id="row-b", token_ids=["token-b"])

        accounting = load_run_accounting_map_from_db(db, ("run-b", "run-a", "run-a"))

        assert tuple(accounting) == ("run-a", "run-b")
        assert accounting["run-a"].integrity.closure == "closed"
        assert accounting["run-a"].tokens.succeeded == 1
        assert accounting["run-b"].integrity.closure == "open"
        assert accounting["run-b"].integrity.missing_terminal_outcomes == 1
    finally:
        db.close()


def test_batch_loader_does_not_fabricate_absent_landscape_runs() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_empty_run(db, run_id="empty-run")

        accounting = load_run_accounting_map_from_db(db, ("missing-run", "empty-run"))

        assert set(accounting) == {"empty-run"}
        assert accounting["empty-run"].source.rows_processed == 0
        assert accounting["empty-run"].tokens.emitted == 0
        assert accounting["empty-run"].integrity.closure == "closed"
    finally:
        db.close()


def test_settings_loader_returns_empty_for_missing_sqlite_database(tmp_path: Path) -> None:
    settings = _SettingsFake(landscape_url=f"sqlite:///{tmp_path / 'missing-audit.db'}")

    assert load_run_accounting_for_settings(settings, ("run-1", None)) == {}


def test_routing_subset_classification_counts_completed_terminal_pairs() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-3")
        token_ids = ["gate", "error", "quarantine", "discard"]
        _insert_tokens(db, run_id="run-3", row_id="row-1", token_ids=token_ids)
        _insert_completed_outcomes(
            db,
            run_id="run-3",
            token_ids=["gate"],
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
        )
        _insert_completed_outcomes(
            db,
            run_id="run-3",
            token_ids=["error"],
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
        )
        _insert_completed_outcomes(
            db,
            run_id="run-3",
            token_ids=["quarantine"],
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
        )
        _insert_completed_outcomes(
            db,
            run_id="run-3",
            token_ids=["discard"],
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
        )

        accounting = load_run_accounting_from_db(db, landscape_run_id="run-3")

        assert accounting.tokens.emitted == 4
        assert accounting.tokens.terminal == 4
        assert accounting.tokens.succeeded == 1
        assert accounting.tokens.failed == 3
        assert accounting.routing.routed_success == 1
        assert accounting.routing.routed_failure == 1
        assert accounting.routing.quarantined == 1
        assert accounting.routing.discarded == 1
        assert accounting.integrity.closure == "closed"
    finally:
        db.close()


def test_buffered_non_completed_outcomes_do_not_count_as_terminal() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-4")
        _insert_tokens(db, run_id="run-4", row_id="row-1", token_ids=["token-1"])
        _insert_outcome(
            db,
            run_id="run-4",
            token_id="token-1",
            outcome=None,
            path=TerminalPath.BUFFERED,
            completed=0,
        )

        accounting = load_run_accounting_from_db(db, landscape_run_id="run-4")

        assert accounting.tokens.emitted == 1
        assert accounting.tokens.terminal == 0
        assert accounting.tokens.pending == 1
        assert accounting.integrity.missing_terminal_outcomes == 1
        assert accounting.integrity.closure == "open"
    finally:
        db.close()


def test_duplicate_completed_outcomes_raise_clear_integrity_error() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-6")
        _insert_tokens(db, run_id="run-6", row_id="row-1", token_ids=["duplicate-token", "missing-token"])
        with db.write_connection() as conn:
            conn.execute(text("DROP INDEX ix_token_outcomes_terminal_unique"))
        _insert_completed_outcomes(
            db,
            run_id="run-6",
            token_ids=["duplicate-token"],
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        _insert_completed_outcomes(
            db,
            run_id="run-6",
            token_ids=["duplicate-token"],
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
        )

        with pytest.raises(ValueError, match="duplicate completed terminal outcomes"):
            load_run_accounting_from_db(db, landscape_run_id="run-6")
    finally:
        db.close()


def test_sqlite_blocks_duplicate_completed_terminal_outcomes() -> None:
    db = LandscapeDB.in_memory()
    try:
        _setup_run_with_row(db, run_id="run-5")
        _insert_tokens(db, run_id="run-5", row_id="row-1", token_ids=["token-1"])
        _insert_completed_outcomes(
            db,
            run_id="run-5",
            token_ids=["token-1"],
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )

        with pytest.raises(IntegrityError):
            _insert_completed_outcomes(
                db,
                run_id="run-5",
                token_ids=["token-1"],
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )
    finally:
        db.close()
