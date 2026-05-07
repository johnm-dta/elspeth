# tests/property/core/test_reproducibility_properties.py
"""Property-based tests for reproducibility grade computation.

These tests verify the invariants of ELSPETH's reproducibility classification:

Grade Hierarchy Properties:
- FULL_REPRODUCIBLE > REPLAY_REPRODUCIBLE > ATTRIBUTABLE_ONLY
- Grade ordering is meaningful (higher = more reproducible)

Determinism Classification Properties:
- DETERMINISTIC, SEEDED -> FULL_REPRODUCIBLE
- IO_READ, IO_WRITE, EXTERNAL_CALL, NON_DETERMINISTIC -> REPLAY_REPRODUCIBLE

Degradation Properties:
- REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY (after purge)
- FULL_REPRODUCIBLE -> unchanged
- ATTRIBUTABLE_ONLY -> unchanged

Enum Integrity Properties:
- Exactly 3 grades exist
- Values are lowercase strings
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import count

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from elspeth.contracts.enums import CallStatus, CallType, Determinism, NodeStateStatus, NodeType, RunStatus
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.reproducibility import (
    ReproducibilityGrade,
    compute_grade,
    update_grade_after_purge,
)
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

# =============================================================================
# Strategies for reproducibility testing
# =============================================================================

# All reproducibility grades
all_grades = st.sampled_from(list(ReproducibilityGrade))

# All determinism values
all_determinism = st.sampled_from(list(Determinism))

# Determinism values that allow full reproducibility
reproducible_determinism = st.sampled_from(
    [
        Determinism.DETERMINISTIC,
        Determinism.SEEDED,
    ]
)

# Determinism values that require replay (need to capture runtime data)
non_reproducible_determinism = st.sampled_from(
    [
        Determinism.IO_READ,
        Determinism.IO_WRITE,
        Determinism.EXTERNAL_CALL,
        Determinism.NON_DETERMINISTIC,
    ]
)

reproducible_lists = st.lists(reproducible_determinism, min_size=1, max_size=6)


@st.composite
def lists_with_non_reproducible(draw: st.DrawFn) -> list[Determinism]:
    """Generate determinism lists with at least one non-reproducible value."""
    non_repro = draw(non_reproducible_determinism)
    others = draw(st.lists(all_determinism, min_size=0, max_size=5))
    return [non_repro, *others]


# =============================================================================
# Helpers for DB-backed tests
# =============================================================================

_REFERENCE_TIME = datetime(2025, 1, 1, tzinfo=UTC)
_RUN_COUNTER = count()


def _create_run(db: LandscapeDB) -> str:
    run_id = f"run-{next(_RUN_COUNTER):06d}"
    now = _REFERENCE_TIME
    with db.connection() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash=stable_hash({"run_id": run_id}),
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.RUNNING,
            )
        )
    return run_id


def _insert_nodes(db: LandscapeDB, run_id: str, determinisms: list[Determinism]) -> None:
    now = _REFERENCE_TIME
    with db.connection() as conn:
        for idx, det in enumerate(determinisms):
            node_id = f"node_{idx}"
            config = {"node_id": node_id, "determinism": det.value}
            conn.execute(
                nodes_table.insert().values(
                    node_id=node_id,
                    run_id=run_id,
                    plugin_name="test_plugin",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=det.value,
                    config_hash=stable_hash(config),
                    config_json=json.dumps(config),
                    registered_at=now,
                )
            )


def _set_run_grade(db: LandscapeDB, run_id: str, grade: ReproducibilityGrade) -> None:
    with db.connection() as conn:
        conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(reproducibility_grade=grade.value))


def _get_run_grade(db: LandscapeDB, run_id: str) -> ReproducibilityGrade:
    with db.connection() as conn:
        row = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id)).fetchone()
    assert row is not None
    assert row[0] is not None
    return ReproducibilityGrade(row[0])


def _insert_purged_call(
    db: LandscapeDB,
    run_id: str,
    node_id: str,
    determinism: Determinism,
) -> None:
    """Insert a node + row + token + node_state + call chain with a purged response.

    Creates the minimal set of records needed to trigger update_grade_after_purge
    degradation: a call belonging to a nondeterministic node where response_hash
    is set (payload once existed) but response_ref is NULL (payload has been purged).

    Args:
        db: LandscapeDB instance
        run_id: Run ID to insert records for
        node_id: Unique node ID to use
        determinism: Determinism value for the node (should be non-reproducible)
    """
    now = _REFERENCE_TIME
    row_id = f"row-{node_id}"
    token_id = f"tok-{node_id}"
    state_id = f"st-{node_id}"
    call_id = f"call-{node_id}"

    with db.connection() as conn:
        conn.execute(
            nodes_table.insert().values(
                node_id=node_id,
                run_id=run_id,
                plugin_name="test_plugin",
                node_type=NodeType.TRANSFORM.value,
                plugin_version="1.0",
                determinism=determinism.value,
                config_hash=stable_hash({"node_id": node_id}),
                config_json="{}",
                registered_at=now,
            )
        )
        conn.execute(
            rows_table.insert().values(
                row_id=row_id,
                run_id=run_id,
                source_node_id=node_id,
                row_index=0,
                source_data_hash="src_hash",
                created_at=now,
            )
        )
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row_id,
                run_id=run_id,
                created_at=now,
            )
        )
        conn.execute(
            node_states_table.insert().values(
                state_id=state_id,
                token_id=token_id,
                run_id=run_id,
                node_id=node_id,
                step_index=0,
                attempt=0,
                status=NodeStateStatus.COMPLETED.value,
                input_hash="in_hash",
                output_hash="out_hash",
                started_at=now,
            )
        )
        conn.execute(
            calls_table.insert().values(
                call_id=call_id,
                state_id=state_id,
                operation_id=None,
                call_index=0,
                call_type=CallType.HTTP.value,
                status=CallStatus.SUCCESS.value,
                request_hash="req_hash",
                response_hash="resp_hash",  # Proof the payload once existed
                response_ref=None,  # NULL = payload has been purged
                created_at=now,
            )
        )


# =============================================================================
# Enum Integrity Property Tests
# =============================================================================


class TestReproducibilityGradeEnumProperties:
    """Property tests for ReproducibilityGrade enum integrity."""

    def test_exactly_three_grades_exist(self) -> None:
        """Property: Exactly 3 reproducibility grades are defined.

        Canary test - adding a new grade requires updating this test
        and the degradation logic.
        """
        grades = list(ReproducibilityGrade)
        assert len(grades) == 3, f"Expected 3 grades, got {len(grades)}: {[g.name for g in grades]}"

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_grade_values_are_lowercase(self, grade: ReproducibilityGrade) -> None:
        """Property: Grade values are lowercase strings (ELSPETH convention)."""
        assert grade.value == grade.value.lower()

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_grade_round_trip_through_value(self, grade: ReproducibilityGrade) -> None:
        """Property: Grade round-trips through string value."""
        recovered = ReproducibilityGrade(grade.value)
        assert recovered is grade

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_grade_is_string_subclass(self, grade: ReproducibilityGrade) -> None:
        """Property: Grade instances are strings (for JSON serialization)."""
        assert isinstance(grade, str)
        assert grade == grade.value

    def test_no_duplicate_values(self) -> None:
        """Property: All grade values are unique."""
        values = [g.value for g in ReproducibilityGrade]
        assert len(values) == len(set(values))


# =============================================================================
# Determinism Classification Property Tests
# =============================================================================


class TestDeterminismClassificationProperties:
    """Property tests for determinism -> grade classification logic.

    The classification used by compute_grade():
    - {DETERMINISTIC, SEEDED} -> FULL_REPRODUCIBLE
    - {IO_READ, IO_WRITE, EXTERNAL_CALL, NON_DETERMINISTIC} -> REPLAY_REPRODUCIBLE
    """

    @given(determinisms=reproducible_lists)
    @settings(max_examples=50)
    def test_reproducible_determinism_yields_full_grade(self, determinisms: list[Determinism]) -> None:
        """Property: Reproducible determinism values yield FULL_REPRODUCIBLE.

        DETERMINISTIC and SEEDED can be re-executed with identical results
        (SEEDED needs the seed captured beforehand).
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, determinisms)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    @given(determinisms=lists_with_non_reproducible())
    @settings(max_examples=50)
    def test_non_reproducible_determinism_yields_replay_grade(self, determinisms: list[Determinism]) -> None:
        """Property: Any non-reproducible determinism yields REPLAY_REPRODUCIBLE.

        IO_READ, IO_WRITE, EXTERNAL_CALL, and NON_DETERMINISTIC require
        recorded runtime data to replay - they can't be re-executed from
        scratch with identical results.
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, determinisms)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_empty_pipeline_is_full_reproducible(self) -> None:
        """Property: Empty pipeline is trivially FULL_REPRODUCIBLE."""
        with make_landscape_db() as db:
            run_id = _create_run(db)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE


# =============================================================================
# Grade Hierarchy Property Tests
# =============================================================================


class TestGradeHierarchyProperties:
    """Property tests for grade ordering and hierarchy."""

    def test_grade_ordering_is_meaningful(self) -> None:
        """Property: Grades have a meaningful ordering (more -> less reproducible).

        FULL_REPRODUCIBLE > REPLAY_REPRODUCIBLE > ATTRIBUTABLE_ONLY
        """
        # Define the hierarchy (index = reproducibility level, higher = better)
        hierarchy = [
            ReproducibilityGrade.ATTRIBUTABLE_ONLY,  # 0 - lowest
            ReproducibilityGrade.REPLAY_REPRODUCIBLE,  # 1 - middle
            ReproducibilityGrade.FULL_REPRODUCIBLE,  # 2 - highest
        ]

        # Verify all grades are in hierarchy
        assert set(hierarchy) == set(ReproducibilityGrade)

        # The ordering represents reproducibility level
        full_idx = hierarchy.index(ReproducibilityGrade.FULL_REPRODUCIBLE)
        replay_idx = hierarchy.index(ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        attr_idx = hierarchy.index(ReproducibilityGrade.ATTRIBUTABLE_ONLY)

        assert full_idx > replay_idx > attr_idx

    def test_full_reproducible_is_highest(self) -> None:
        """Property: FULL_REPRODUCIBLE is the highest (best) grade."""
        # FULL means we can re-run with identical results
        assert ReproducibilityGrade.FULL_REPRODUCIBLE.value == "full_reproducible"

    def test_attributable_only_is_lowest(self) -> None:
        """Property: ATTRIBUTABLE_ONLY is the lowest grade."""
        # ATTRIBUTABLE means we can only verify via hashes, not replay
        assert ReproducibilityGrade.ATTRIBUTABLE_ONLY.value == "attributable_only"


# =============================================================================
# Degradation Logic Property Tests
# =============================================================================


class TestGradeDegradationProperties:
    """Property tests for grade degradation after purge.

    The degradation rules from update_grade_after_purge():
    - REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY
    - FULL_REPRODUCIBLE -> unchanged
    - ATTRIBUTABLE_ONLY -> unchanged
    """

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_degradation_rule_applies(self, grade: ReproducibilityGrade) -> None:
        """Property: Degradation rule applied via update_grade_after_purge().

        With no calls recorded, REPLAY_REPRODUCIBLE is unchanged — there are no
        replay-critical payloads to purge, so the condition for downgrade is never met.
        FULL_REPRODUCIBLE and ATTRIBUTABLE_ONLY are always unchanged.
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            _set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            # update_grade_after_purge only downgrades REPLAY_REPRODUCIBLE when
            # replay-critical payloads (nondeterministic node responses) have been purged.
            # With no calls in the DB, nothing is purged, so all grades are unchanged.
            assert _get_run_grade(db, run_id) == grade

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_degradation_is_idempotent(self, grade: ReproducibilityGrade) -> None:
        """Property: Applying degradation twice gives same result as once.

        degrade(degrade(x)) == degrade(x) for all grades.
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            _set_run_grade(db, run_id, grade)

            update_grade_after_purge(db, run_id)
            once = _get_run_grade(db, run_id)

            update_grade_after_purge(db, run_id)
            twice = _get_run_grade(db, run_id)

            assert once == twice

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_degradation_never_increases_grade(self, grade: ReproducibilityGrade) -> None:
        """Property: Degradation never results in a higher grade.

        Purging data can only reduce reproducibility, never improve it.
        """
        hierarchy = {
            ReproducibilityGrade.ATTRIBUTABLE_ONLY: 0,
            ReproducibilityGrade.REPLAY_REPRODUCIBLE: 1,
            ReproducibilityGrade.FULL_REPRODUCIBLE: 2,
        }

        with make_landscape_db() as db:
            run_id = _create_run(db)
            _set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)
            result = _get_run_grade(db, run_id)

        assert hierarchy[result] <= hierarchy[grade]


# =============================================================================
# Combined Classification and Degradation Property Tests
# =============================================================================


class TestClassificationDegradationInteractionProperties:
    """Property tests for interaction between classification and degradation."""

    @given(det=reproducible_determinism)
    @settings(max_examples=50)
    def test_reproducible_determinism_survives_purge(self, det: Determinism) -> None:
        """Property: Reproducible determinism stays FULL even after purge.

        DETERMINISTIC nodes don't need recorded payloads to re-execute.
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, [det])

            grade = compute_grade(db, run_id)
            _set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            assert _get_run_grade(db, run_id) == ReproducibilityGrade.FULL_REPRODUCIBLE

    @given(det=non_reproducible_determinism)
    @settings(max_examples=50)
    def test_non_reproducible_determinism_degrades_after_purge(self, det: Determinism) -> None:
        """Property: Non-reproducible determinism degrades to ATTRIBUTABLE after purge.

        Without recorded responses, we can only prove what happened via hashes.
        Degradation requires evidence of purged replay-critical payloads: a call
        belonging to a nondeterministic node where response_hash is set but
        response_ref is NULL.
        """
        with make_landscape_db() as db:
            run_id = _create_run(db)
            # Insert the nondeterministic node and a purged call record so
            # update_grade_after_purge has evidence that replay-critical payloads
            # have been purged (response_hash set, response_ref NULL).
            _insert_purged_call(db, run_id, node_id="nd-node-0", determinism=det)

            grade = compute_grade(db, run_id)
            _set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            assert _get_run_grade(db, run_id) == ReproducibilityGrade.ATTRIBUTABLE_ONLY
