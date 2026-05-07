"""Tests for reproducibility grade computation.

Tests cover:
- compute_grade with all-deterministic nodes → FULL_REPRODUCIBLE
- compute_grade with nondeterministic node → REPLAY_REPRODUCIBLE
- compute_grade with empty pipeline (no nodes) → FULL_REPRODUCIBLE
- compute_grade with seeded nodes → FULL_REPRODUCIBLE
- compute_grade crashes on NULL/invalid determinism (Tier 1)
- compute_grade raises for nonexistent run
- update_grade_after_purge degrades REPLAY→ATTRIBUTABLE, leaves others
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import CallStatus, CallType, Determinism, NodeStateStatus, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.reproducibility import (
    ReproducibilityGrade,
    compute_grade,
    update_grade_after_purge,
)
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    rows_table,
    runs_table,
    tokens_table,
)
from tests.fixtures.landscape import make_factory, make_landscape_db

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    """Create in-memory DB with a run."""
    db = make_landscape_db()
    factory = make_factory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    return db, factory


class TestComputeGrade:
    """Tests for compute_grade — determines reproducibility from node determinism."""

    def test_all_deterministic_returns_full(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_seeded_returns_full(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="sampler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.SEEDED,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_nondeterministic_returns_replay(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="llm",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.NON_DETERMINISTIC,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_external_call_returns_replay(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="api",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.EXTERNAL_CALL,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_io_read_returns_replay(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="reader",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.IO_READ,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_io_write_returns_replay(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="writer",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.IO_WRITE,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_mixed_deterministic_and_nondeterministic(self) -> None:
        db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="n1",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.DETERMINISTIC,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="llm",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="n2",
            schema_config=_DYNAMIC_SCHEMA,
            determinism=Determinism.NON_DETERMINISTIC,
        )
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_empty_pipeline_returns_full(self) -> None:
        db, _factory = _setup()
        grade = compute_grade(db, "run-1")
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_nonexistent_run_raises(self) -> None:
        db, _factory = _setup()
        with pytest.raises(AuditIntegrityError, match="does not exist"):
            compute_grade(db, "nonexistent-run")


def _set_grade(db: LandscapeDB, run_id: str, grade: ReproducibilityGrade) -> None:
    """Set reproducibility grade via direct SQL (test helper)."""
    with db.connection() as conn:
        conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(reproducibility_grade=grade.value))


def _create_nondeterministic_call(
    db: LandscapeDB,
    factory: RecorderFactory,
    *,
    determinism: Determinism = Determinism.NON_DETERMINISTIC,
    response_ref: str | None = None,
    response_hash: str | None = "resp_hash",
    node_id: str = "nd-node",
    state_id: str = "st-1",
    call_id: str = "call-1",
) -> None:
    """Create a node + node_state + call chain for purge testing."""
    factory.data_flow.register_node(
        run_id="run-1",
        plugin_name="llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id=node_id,
        schema_config=_DYNAMIC_SCHEMA,
        determinism=determinism,
    )
    with db.connection() as conn:
        # Row and token are needed for node_state FK
        conn.execute(
            rows_table.insert().values(
                row_id=f"row-{node_id}",
                run_id="run-1",
                source_node_id=node_id,
                row_index=0,
                source_data_hash="src_hash",
                created_at=datetime.now(UTC),
            )
        )
        conn.execute(
            tokens_table.insert().values(
                token_id=f"tok-{node_id}",
                row_id=f"row-{node_id}",
                run_id="run-1",
                created_at=datetime.now(UTC),
            )
        )
        conn.execute(
            node_states_table.insert().values(
                state_id=state_id,
                token_id=f"tok-{node_id}",
                run_id="run-1",
                node_id=node_id,
                step_index=0,
                attempt=0,
                status=NodeStateStatus.COMPLETED,
                input_hash="in_hash",
                output_hash="out_hash",
                started_at=datetime.now(UTC),
            )
        )
        conn.execute(
            calls_table.insert().values(
                call_id=call_id,
                state_id=state_id,
                operation_id=None,
                call_index=0,
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_hash="req_hash",
                response_hash=response_hash,
                response_ref=response_ref,
                created_at=datetime.now(UTC),
            )
        )


class TestUpdateGradeAfterPurge:
    """Tests for update_grade_after_purge — degrades REPLAY → ATTRIBUTABLE."""

    def test_replay_degrades_when_nondeterministic_response_purged(self) -> None:
        """Downgrade when a nondeterministic node's response payload has been purged."""
        db, factory = _setup()
        # Create a nondeterministic node with a purged response
        # (response_hash set but response_ref is None = purged)
        _create_nondeterministic_call(db, factory, response_ref=None, response_hash="resp_hash")
        _set_grade(db, "run-1", ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

    def test_replay_unchanged_when_only_deterministic_payloads_purged(self) -> None:
        """Do NOT downgrade when only deterministic node payloads are purged."""
        db, factory = _setup()
        # Create a deterministic node with a purged response — not replay-critical
        _create_nondeterministic_call(
            db,
            factory,
            determinism=Determinism.DETERMINISTIC,
            response_ref=None,
            response_hash="resp_hash",
        )
        _set_grade(db, "run-1", ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_replay_unchanged_when_nondeterministic_response_intact(self) -> None:
        """Do NOT downgrade when nondeterministic responses are still present."""
        db, factory = _setup()
        # Create a nondeterministic node with response still present
        _create_nondeterministic_call(db, factory, response_ref="ref://still-there", response_hash="resp_hash")
        _set_grade(db, "run-1", ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_replay_unchanged_when_no_calls_exist(self) -> None:
        """Do NOT downgrade when run has no calls at all (nothing to purge)."""
        db, factory = _setup()
        _set_grade(db, "run-1", ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_full_unchanged_after_purge(self) -> None:
        db, factory = _setup()
        _set_grade(db, "run-1", ReproducibilityGrade.FULL_REPRODUCIBLE)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_attributable_unchanged_after_purge(self) -> None:
        db, factory = _setup()
        _set_grade(db, "run-1", ReproducibilityGrade.ATTRIBUTABLE_ONLY)
        update_grade_after_purge(db, "run-1")
        run = factory.run_lifecycle.get_run("run-1")
        assert run is not None
        assert run.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

    def test_nonexistent_run_raises(self) -> None:
        """Purging a nonexistent run is a caller bug — must crash."""
        db, _factory = _setup()
        with pytest.raises(AuditIntegrityError, match="does not exist"):
            update_grade_after_purge(db, "nonexistent")

    def test_null_grade_raises(self) -> None:
        """NULL reproducibility_grade is Tier 1 corruption — must crash."""
        db, _factory = _setup()
        # begin_run doesn't set a grade by default, so it's NULL
        with pytest.raises(AuditIntegrityError, match="NULL reproducibility_grade"):
            update_grade_after_purge(db, "run-1")
