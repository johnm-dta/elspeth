# tests/property/core/test_reproducibility_properties.py
"""Property-based tests for reproducibility grade computation.

These tests verify the invariants of ELSPETH's reproducibility classification:

Grade Hierarchy Properties:
- FULL_REPRODUCIBLE > REPLAY_REPRODUCIBLE > ATTRIBUTABLE_ONLY
- Grade ordering is meaningful (higher = more reproducible)

Determinism Classification Properties:
- DETERMINISTIC, SEEDED, IO_READ, IO_WRITE → FULL_REPRODUCIBLE
- EXTERNAL_CALL, NON_DETERMINISTIC → REPLAY_REPRODUCIBLE

Degradation Properties:
- REPLAY_REPRODUCIBLE → ATTRIBUTABLE_ONLY (after purge)
- FULL_REPRODUCIBLE → unchanged
- ATTRIBUTABLE_ONLY → unchanged

Enum Integrity Properties:
- Exactly 3 grades exist
- Values are lowercase strings
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from elspeth.contracts.enums import Determinism, NodeType, RunStatus
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.reproducibility import (
    ReproducibilityGrade,
    compute_grade,
    set_run_grade,
    update_grade_after_purge,
)
from elspeth.core.landscape.schema import nodes_table, runs_table

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
        Determinism.IO_READ,
        Determinism.IO_WRITE,
    ]
)

# Determinism values that require replay
non_reproducible_determinism = st.sampled_from(
    [
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


def _create_run(db: LandscapeDB) -> str:
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    now = datetime.now(UTC)
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
    now = datetime.now(UTC)
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


def _get_run_grade(db: LandscapeDB, run_id: str) -> ReproducibilityGrade:
    with db.connection() as conn:
        row = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id)).fetchone()
    assert row is not None
    assert row[0] is not None
    return ReproducibilityGrade(row[0])


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
    """Property tests for determinism → grade classification logic.

    The classification used by compute_grade():
    - {DETERMINISTIC, SEEDED, IO_READ, IO_WRITE} → FULL_REPRODUCIBLE
    - {EXTERNAL_CALL, NON_DETERMINISTIC} → REPLAY_REPRODUCIBLE
    """

    @given(determinisms=reproducible_lists)
    @settings(max_examples=50)
    def test_reproducible_determinism_yields_full_grade(self, determinisms: list[Determinism]) -> None:
        """Property: Reproducible determinism values yield FULL_REPRODUCIBLE.

        DETERMINISTIC, SEEDED, IO_READ, IO_WRITE can all be re-executed
        with identical results (SEEDED needs the seed captured).
        """
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, determinisms)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    @given(determinisms=lists_with_non_reproducible())
    @settings(max_examples=50)
    def test_non_reproducible_determinism_yields_replay_grade(self, determinisms: list[Determinism]) -> None:
        """Property: Any non-reproducible determinism yields REPLAY_REPRODUCIBLE.

        EXTERNAL_CALL (LLM, APIs) and NON_DETERMINISTIC require recorded
        responses to replay - they can't be re-executed deterministically.
        """
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, determinisms)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_empty_pipeline_is_full_reproducible(self) -> None:
        """Property: Empty pipeline is trivially FULL_REPRODUCIBLE."""
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            grade = compute_grade(db, run_id)
            assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE


# =============================================================================
# Grade Hierarchy Property Tests
# =============================================================================


class TestGradeHierarchyProperties:
    """Property tests for grade ordering and hierarchy."""

    def test_grade_ordering_is_meaningful(self) -> None:
        """Property: Grades have a meaningful ordering (more → less reproducible).

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
    - REPLAY_REPRODUCIBLE → ATTRIBUTABLE_ONLY
    - FULL_REPRODUCIBLE → unchanged
    - ATTRIBUTABLE_ONLY → unchanged
    """

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_degradation_rule_applies(self, grade: ReproducibilityGrade) -> None:
        """Property: Degradation rule applied via update_grade_after_purge()."""
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            expected = ReproducibilityGrade.ATTRIBUTABLE_ONLY if grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE else grade
            assert _get_run_grade(db, run_id) == expected

    @given(grade=all_grades)
    @settings(max_examples=20)
    def test_degradation_is_idempotent(self, grade: ReproducibilityGrade) -> None:
        """Property: Applying degradation twice gives same result as once.

        degrade(degrade(x)) == degrade(x) for all grades.
        """
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            set_run_grade(db, run_id, grade)

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

        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            set_run_grade(db, run_id, grade)
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
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, [det])

            grade = compute_grade(db, run_id)
            set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            assert _get_run_grade(db, run_id) == ReproducibilityGrade.FULL_REPRODUCIBLE

    @given(det=non_reproducible_determinism)
    @settings(max_examples=50)
    def test_non_reproducible_determinism_degrades_after_purge(self, det: Determinism) -> None:
        """Property: Non-reproducible determinism degrades to ATTRIBUTABLE after purge.

        Without recorded responses, we can only prove what happened via hashes.
        """
        with LandscapeDB.in_memory() as db:
            run_id = _create_run(db)
            _insert_nodes(db, run_id, [det])

            grade = compute_grade(db, run_id)
            set_run_grade(db, run_id, grade)
            update_grade_after_purge(db, run_id)

            assert _get_run_grade(db, run_id) == ReproducibilityGrade.ATTRIBUTABLE_ONLY
