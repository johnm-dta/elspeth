# tests/property/core/test_landscape_recording_properties.py
"""Property-based tests for landscape recording invariants.

The landscape recording system is the audit backbone of ELSPETH. These tests
verify invariants that must hold for the audit trail to be trustworthy:

1. Run lifecycle: begin→get round-trip preserves all fields
2. Token outcome contracts: each ADR-019 terminal pair requires specific fields
3. Schema contract round-trip: store→retrieve preserves contract content
4. Config hash determinism: same config → same hash
5. Row recording referential integrity: rows link to valid runs
6. Run completion timestamp ordering: completed_at >= started_at

Testing approach:
- Uses LandscapeDB.in_memory() via RecorderFactory for isolated, fast property tests
- Hypothesis generates varied configs, field names, and outcome types
- Tests verify database-level invariants (not just API-level behavior)
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts import (
    NodeType,
    RunStatus,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run

# Minimal contract for tests that only care about token lifecycle, not contract content.
_MINIMAL_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)

# =============================================================================
# Strategies
# =============================================================================

# Simple config dicts for run creation
simple_configs = st.fixed_dictionaries(
    {
        "source": st.fixed_dictionaries({"plugin": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")}),
    }
)

# Canonical version strings
canonical_versions = st.from_regex(r"[0-9]+\.[0-9]+", fullmatch=True)

# Valid sink names for token outcomes
sink_names = st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_")

# Dummy group IDs
group_ids = st.text(min_size=8, max_size=32, alphabet="0123456789abcdef")

# Dummy error hashes
error_hashes = st.text(min_size=32, max_size=64, alphabet="0123456789abcdef")

# Row data for testing
row_data = st.fixed_dictionaries(
    {
        "value": st.one_of(st.integers(min_value=-1000, max_value=1000), st.text(min_size=0, max_size=50)),
    }
)

# Field names for schema contracts
field_names = st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_")


def _make_schema_config() -> SchemaConfig:
    """Create a dynamic schema config for testing."""
    return SchemaConfig.from_dict({"mode": "observed"})


# =============================================================================
# Run Lifecycle Round-Trip Properties
# =============================================================================


class TestRunLifecycleProperties:
    """begin_run → get_run must preserve all fields."""

    @given(config=simple_configs, version=canonical_versions)
    @settings(max_examples=100, deadline=None)
    def test_begin_get_round_trip(self, config: dict[str, Any], version: str) -> None:
        """Property: get_run returns the same data that begin_run stored."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(config=config, canonical_version=version)

            retrieved = factory.run_lifecycle.get_run(run.run_id)
            assert retrieved is not None
            assert retrieved.run_id == run.run_id
            assert retrieved.config_hash == run.config_hash
            assert retrieved.canonical_version == version
            assert retrieved.status == RunStatus.RUNNING

    @given(config=simple_configs)
    @settings(max_examples=50, deadline=None)
    def test_run_id_uniqueness(self, config: dict[str, Any]) -> None:
        """Property: Auto-generated run IDs are unique across multiple begin_run calls."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            ids = set()
            for _ in range(5):
                run = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
                assert run.run_id not in ids, f"Duplicate run_id: {run.run_id}"
                ids.add(run.run_id)

    @given(config=simple_configs)
    @settings(max_examples=50, deadline=None)
    def test_config_hash_determinism(self, config: dict[str, Any]) -> None:
        """Property: Same config always produces the same config_hash."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run1 = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            run2 = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            assert run1.config_hash == run2.config_hash

    @given(config=simple_configs)
    @settings(max_examples=50, deadline=None)
    def test_complete_run_sets_status(self, config: dict[str, Any]) -> None:
        """Property: complete_run transitions status to the specified value."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")

            completed = factory.run_lifecycle.complete_run(run.run_id, RunStatus.COMPLETED)
            assert completed.status == RunStatus.COMPLETED

    @given(config=simple_configs)
    @settings(max_examples=50, deadline=None)
    def test_complete_run_sets_completed_at(self, config: dict[str, Any]) -> None:
        """Property: complete_run sets completed_at timestamp."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")

            completed = factory.run_lifecycle.complete_run(run.run_id, RunStatus.COMPLETED)
            assert completed.completed_at is not None

    @given(config=simple_configs)
    @settings(max_examples=30, deadline=None)
    def test_completed_at_after_started_at(self, config: dict[str, Any]) -> None:
        """Property: completed_at >= started_at (temporal ordering)."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")

            completed = factory.run_lifecycle.complete_run(run.run_id, RunStatus.COMPLETED)
            assert completed.completed_at is not None
            assert completed.completed_at >= completed.started_at

    @given(config=simple_configs)
    @settings(max_examples=30, deadline=None)
    def test_initial_status_is_running(self, config: dict[str, Any]) -> None:
        """Property: Newly created runs have RUNNING status."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            assert run.status == RunStatus.RUNNING

    def test_get_nonexistent_run_returns_none(self) -> None:
        """Property: get_run for nonexistent ID returns None, not crash."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            assert factory.run_lifecycle.get_run("nonexistent-id") is None

    @given(config=simple_configs)
    @settings(max_examples=30, deadline=None)
    def test_list_runs_includes_created(self, config: dict[str, Any]) -> None:
        """Property: list_runs includes all created runs."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run1 = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            run2 = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")

            runs = factory.run_lifecycle.list_runs()
            run_ids = {r.run_id for r in runs}
            assert run1.run_id in run_ids
            assert run2.run_id in run_ids

    @given(config=simple_configs)
    @settings(max_examples=30, deadline=None)
    def test_list_runs_filters_by_status(self, config: dict[str, Any]) -> None:
        """Property: list_runs(status=X) returns only runs with that status."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            run2 = factory.run_lifecycle.begin_run(config=config, canonical_version="1.0")
            factory.run_lifecycle.complete_run(run2.run_id, RunStatus.COMPLETED)

            running = factory.run_lifecycle.list_runs(status=RunStatus.RUNNING)
            assert all(r.status == RunStatus.RUNNING for r in running)

            completed = factory.run_lifecycle.list_runs(status=RunStatus.COMPLETED)
            assert all(r.status == RunStatus.COMPLETED for r in completed)


# =============================================================================
# Token Outcome Contract Enforcement Properties
# =============================================================================


class TestTokenOutcomeContractProperties:
    """Each terminal pair requires specific fields — missing ones must raise ValueError."""

    def _setup(self) -> tuple[LandscapeDB, RecorderFactory, str, str]:
        """Create a run with a source row and token for testing outcomes."""
        setup = make_recorder_with_run(canonical_version="1.0", source_plugin_name="test_source")
        db, factory, run_id, source_node_id = setup.db, setup.factory, setup.run_id, setup.source_node_id
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"value": 1},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        return db, factory, run_id, token.token_id

    def test_completed_requires_sink_name(self) -> None:
        """Property: (SUCCESS, DEFAULT_FLOW) without sink_name raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(SUCCESS, DEFAULT_FLOW\) outcome requires sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
            )
        db.close()

    @given(sink=sink_names)
    @settings(max_examples=30, deadline=None)
    def test_completed_with_sink_name_succeeds(self, sink: str) -> None:
        """Property: (SUCCESS, DEFAULT_FLOW) with valid sink_name succeeds."""
        db, factory, run_id, token_id = self._setup()
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name=sink,
        )
        assert outcome_id is not None
        db.close()

    def test_routed_requires_sink_name(self) -> None:
        """Property: (SUCCESS, GATE_ROUTED) without sink_name raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(SUCCESS, GATE_ROUTED\) outcome requires sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
            )
        db.close()

    def test_forked_requires_fork_group_id(self) -> None:
        """Property: (TRANSIENT, FORK_PARENT) without fork_group_id raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(TRANSIENT, FORK_PARENT\) outcome requires fork_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
            )
        db.close()

    def test_failed_requires_error_hash(self) -> None:
        """Property: (FAILURE, UNROUTED) without error_hash raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(FAILURE, UNROUTED\) outcome requires error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )
        db.close()

    def test_quarantined_requires_error_hash(self) -> None:
        """Property: (FAILURE, QUARANTINED_AT_SOURCE) without error_hash raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(FAILURE, QUARANTINED_AT_SOURCE\) outcome requires error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )
        db.close()

    def test_consumed_in_batch_requires_batch_id(self) -> None:
        """Property: (TRANSIENT, BATCH_CONSUMED) without batch_id raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(TRANSIENT, BATCH_CONSUMED\) outcome requires batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.BATCH_CONSUMED,
            )
        db.close()

    def test_coalesced_requires_join_group_id(self) -> None:
        """Property: (SUCCESS, COALESCED) without join_group_id raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(SUCCESS, COALESCED\) outcome requires join_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.COALESCED,
                sink_name="default",
            )
        db.close()

    def test_expanded_requires_expand_group_id(self) -> None:
        """Property: (TRANSIENT, EXPAND_PARENT) without expand_group_id raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(TRANSIENT, EXPAND_PARENT\) outcome requires expand_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.EXPAND_PARENT,
            )
        db.close()

    def test_buffered_requires_batch_id(self) -> None:
        """Property: (NULL, BUFFERED) without batch_id raises ValueError."""
        db, factory, run_id, token_id = self._setup()
        with pytest.raises(ValueError, match=r"\(NULL, BUFFERED\) outcome requires batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=None,
                path=TerminalPath.BUFFERED,
            )
        db.close()

    def test_dropped_by_filter_requires_no_auxiliary_fields(self) -> None:
        """Property: DROPPED_BY_FILTER is terminal with no extra required fields."""
        db, factory, run_id, token_id = self._setup()
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.FILTER_DROPPED,
        )
        assert outcome_id is not None
        db.close()


# =============================================================================
# Schema Contract Round-Trip Properties
# =============================================================================


class TestSchemaContractRoundTripProperties:
    """Schema contracts stored in runs must survive round-trip."""

    @given(
        field_name=field_names,
        field_type=st.sampled_from([str, int, float, bool]),
    )
    @settings(max_examples=50, deadline=None)
    def test_contract_round_trip(self, field_name: str, field_type: type) -> None:
        """Property: update_run_contract → get_run_contract preserves field info."""
        field = FieldContract(
            normalized_name=field_name,
            original_name=field_name,
            python_type=field_type,
            required=True,
            source="declared",
        )
        contract = SchemaContract(mode="FIXED", fields=(field,), locked=True)

        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            factory.run_lifecycle.update_run_contract(run.run_id, contract)
            restored = factory.run_lifecycle.get_run_contract(run.run_id)

            assert restored is not None
            assert restored.mode == "FIXED"
            assert len(restored.fields) == 1
            assert restored.fields[0].normalized_name == field_name
            assert restored.fields[0].python_type == field_type

    @given(
        n_fields=st.integers(min_value=1, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=30, deadline=None)
    def test_multi_field_contract_round_trip(self, n_fields: int, data: st.DataObject) -> None:
        """Property: Multi-field contracts survive round-trip."""
        # Generate unique field names
        names = data.draw(
            st.lists(
                field_names,
                min_size=n_fields,
                max_size=n_fields,
                unique=True,
            )
        )
        fields_list = [
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=str,
                required=True,
                source="declared",
            )
            for name in names
        ]
        contract = SchemaContract(mode="FIXED", fields=tuple(fields_list), locked=True)

        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            factory.run_lifecycle.update_run_contract(run.run_id, contract)
            restored = factory.run_lifecycle.get_run_contract(run.run_id)

            assert restored is not None
            assert len(restored.fields) == n_fields
            restored_names = {f.normalized_name for f in restored.fields}
            for name in names:
                assert name in restored_names

    def test_no_contract_returns_none(self) -> None:
        """Property: get_run_contract returns None when no contract stored."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            assert factory.run_lifecycle.get_run_contract(run.run_id) is None


# =============================================================================
# Row-Token Referential Integrity Properties
# =============================================================================


class TestReferentialIntegrityProperties:
    """Tokens must always link to valid rows, rows to valid runs."""

    @given(n_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_all_tokens_have_valid_rows(self, n_rows: int) -> None:
        """Property: Every token's row_id exists in the rows table."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )

            for i in range(n_rows):
                row = factory.data_flow.create_row(
                    run_id=run.run_id,
                    source_node_id=source.node_id,
                    row_index=i,
                    data={"i": i},
                )
                factory.data_flow.create_token(row_id=row.row_id)

            # Verify no orphan tokens
            with db.connection() as conn:
                orphans = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM tokens t
                        LEFT JOIN rows r ON r.row_id = t.row_id
                        WHERE r.row_id IS NULL
                    """)
                ).scalar()
                assert orphans == 0

    @given(n_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_all_rows_have_valid_runs(self, n_rows: int) -> None:
        """Property: Every row's run_id exists in the runs table."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )

            for i in range(n_rows):
                factory.data_flow.create_row(
                    run_id=run.run_id,
                    source_node_id=source.node_id,
                    row_index=i,
                    data={"i": i},
                )

            # Verify no orphan rows
            with db.connection() as conn:
                orphans = conn.execute(
                    text("""
                        SELECT COUNT(*) FROM rows r
                        LEFT JOIN runs ru ON ru.run_id = r.run_id
                        WHERE ru.run_id IS NULL
                    """)
                ).scalar()
                assert orphans == 0

    @given(branch_count=st.integers(min_value=2, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_fork_children_share_row_id(self, branch_count: int) -> None:
        """Property: All fork children reference the same row_id as parent."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )

            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data={"v": 1},
            )
            parent = factory.data_flow.create_token(row_id=row.row_id)

            branches = [f"b_{i}" for i in range(branch_count)]
            children, _fork_group_id = factory.data_flow.fork_token(
                parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
                row_id=row.row_id,
                branches=branches,
            )

            assert len(children) == branch_count
            for child in children:
                assert child.row_id == row.row_id

    def test_fork_with_empty_branches_raises(self) -> None:
        """Property: fork_token rejects empty branch list."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data={"v": 1},
            )
            parent = factory.data_flow.create_token(row_id=row.row_id)

            with pytest.raises(ValueError, match="at least one branch"):
                factory.data_flow.fork_token(
                    parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
                    row_id=row.row_id,
                    branches=[],
                )

    @given(count=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_expand_creates_correct_children(self, count: int) -> None:
        """Property: expand_token creates exactly N children."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data={"v": 1},
            )
            parent = factory.data_flow.create_token(row_id=row.row_id)

            children, expand_group_id = factory.data_flow.expand_token(
                parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
                row_id=row.row_id,
                child_payloads=[{"item": i} for i in range(count)],
                output_contract=_MINIMAL_CONTRACT,
            )

            assert len(children) == count
            assert expand_group_id is not None
            for child in children:
                assert child.row_id == row.row_id
                assert child.expand_group_id == expand_group_id


# =============================================================================
# Source Field Resolution Round-Trip Properties
# =============================================================================


class TestFieldResolutionProperties:
    """Source field resolution must survive round-trip."""

    @given(
        data=st.data(),
        n_fields=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_resolution_mapping_round_trip(self, data: st.DataObject, n_fields: int) -> None:
        """Property: record→get field resolution preserves mapping."""
        originals = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        finals = data.draw(st.lists(field_names, min_size=n_fields, max_size=n_fields, unique=True))
        mapping = dict(zip(originals, finals, strict=False))

        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            factory.run_lifecycle.record_source_field_resolution(
                run_id=run.run_id,
                resolution_mapping=mapping,
                normalization_version="v1",
            )

            retrieved = factory.run_lifecycle.get_source_field_resolution(run.run_id)
            assert retrieved == mapping

    def test_no_resolution_returns_none(self) -> None:
        """Property: get_source_field_resolution returns None when not recorded."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            assert factory.run_lifecycle.get_source_field_resolution(run.run_id) is None


# =============================================================================
# Row Data Hash Determinism Properties
# =============================================================================


class TestRowHashProperties:
    """Row data hashes must be deterministic and unique for different data."""

    @given(data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_same_data_same_hash(self, data: dict[str, Any]) -> None:
        """Property: Identical data produces identical source_data_hash."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )

            row1 = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data=data,
            )
            row2 = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=1,
                data=data,
            )

            assert row1.source_data_hash == row2.source_data_hash

    @given(
        data1=st.fixed_dictionaries({"value": st.just(1)}),
        data2=st.fixed_dictionaries({"value": st.just(2)}),
    )
    @settings(max_examples=10, deadline=None)
    def test_different_data_different_hash(self, data1: dict[str, Any], data2: dict[str, Any]) -> None:
        """Property: Different data produces different source_data_hash."""
        with make_landscape_db() as db:
            factory = make_factory(db)
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )
            source = factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="src",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=_make_schema_config(),
            )

            row1 = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data=data1,
            )
            row2 = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=1,
                data=data2,
            )

            assert row1.source_data_hash != row2.source_data_hash
