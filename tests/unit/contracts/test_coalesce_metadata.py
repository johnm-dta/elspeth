# tests/unit/contracts/test_coalesce_metadata.py
"""Tests for CoalesceMetadata factory methods — truthiness conflation bugs."""

from types import MappingProxyType

import pytest

from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry, CoalesceMetadata


def _base_merge_metadata() -> CoalesceMetadata:
    """Build a minimal successful-merge base instance for provenance tests."""
    return CoalesceMetadata.for_merge(
        policy=CoalescePolicy.REQUIRE_ALL,
        merge_strategy=MergeStrategy.UNION,
        expected_branches=["a", "b"],
        branches_arrived=["a", "b"],
        branches_lost={},
        arrival_order=[
            ArrivalOrderEntry(branch="a", arrival_offset_ms=0.0),
            ArrivalOrderEntry(branch="b", arrival_offset_ms=10.0),
        ],
        wait_duration_ms=10.0,
    )


class TestForFailureTruthinessConflation:
    """for_failure must distinguish empty dict from None for branches_lost.

    Empty dict ({}) means "zero branches lost" — a known, meaningful state.
    None means "unknown/not applicable." Truthiness conflates them.
    """

    def test_empty_dict_branches_lost_preserved_as_empty_mapping(self) -> None:
        """branches_lost={} must become MappingProxyType({}), not None."""
        meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["a", "b"],
            branches_arrived=["a"],
            branches_lost={},
        )
        assert meta.branches_lost is not None
        assert meta.branches_lost == MappingProxyType({})

    def test_none_branches_lost_stays_none(self) -> None:
        """branches_lost=None must stay None."""
        meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["a", "b"],
            branches_arrived=["a"],
            branches_lost=None,
        )
        assert meta.branches_lost is None

    def test_populated_branches_lost_preserved(self) -> None:
        """Non-empty branches_lost dict is wrapped correctly."""
        meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["a", "b"],
            branches_arrived=["a"],
            branches_lost={"b": "timeout"},
        )
        assert meta.branches_lost is not None
        assert dict(meta.branches_lost) == {"b": "timeout"}


class TestBranchesLostAuditShapeConsistency:
    """Regression: elspeth-a7a6526b14, elspeth-de2e51b30e.

    to_dict() output for branches_lost must have a consistent shape between
    for_failure and for_merge paths. Empty dict must serialize as
    "branches_lost": {}, not be omitted entirely.
    """

    def test_for_failure_empty_dict_emits_branches_lost_key(self) -> None:
        """for_failure with empty branches_lost={} must emit the key in to_dict()."""
        meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["a", "b"],
            branches_arrived=["a", "b"],
            branches_lost={},
        )
        d = meta.to_dict()
        assert "branches_lost" in d, "branches_lost key omitted from to_dict() output"
        assert d["branches_lost"] == {}

    def test_for_merge_empty_dict_emits_branches_lost_key(self) -> None:
        """for_merge with empty branches_lost={} must emit the key in to_dict()."""
        from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry

        meta = CoalesceMetadata.for_merge(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
            expected_branches=["a", "b"],
            branches_arrived=["a", "b"],
            branches_lost={},
            arrival_order=[
                ArrivalOrderEntry(branch="a", arrival_offset_ms=0.0),
                ArrivalOrderEntry(branch="b", arrival_offset_ms=100.0),
            ],
            wait_duration_ms=100.0,
        )
        d = meta.to_dict()
        assert "branches_lost" in d
        assert d["branches_lost"] == {}

    def test_failure_and_merge_shape_consistent_for_empty_branches_lost(self) -> None:
        """Both paths must produce identical shape for 'no branches lost'."""
        from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry

        failure_meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["a"],
            branches_arrived=["a"],
            branches_lost={},
        )
        merge_meta = CoalesceMetadata.for_merge(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
            expected_branches=["a"],
            branches_arrived=["a"],
            branches_lost={},
            arrival_order=[ArrivalOrderEntry(branch="a", arrival_offset_ms=0.0)],
            wait_duration_ms=0.0,
        )
        failure_has_key = "branches_lost" in failure_meta.to_dict()
        merge_has_key = "branches_lost" in merge_meta.to_dict()
        assert failure_has_key == merge_has_key, (
            f"Shape inconsistency: for_failure emits branches_lost={failure_has_key}, for_merge emits branches_lost={merge_has_key}"
        )


class TestCoalesceMetadataEnumFields:
    def test_to_dict_emits_string_value_for_policy(self) -> None:
        meta = CoalesceMetadata(policy=CoalescePolicy.REQUIRE_ALL)
        d = meta.to_dict()
        assert d["policy"] == "require_all"
        assert isinstance(d["policy"], str)

    def test_to_dict_emits_string_value_for_merge_strategy(self) -> None:
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
        )
        d = meta.to_dict()
        assert d["merge_strategy"] == "union"
        assert isinstance(d["merge_strategy"], str)


class TestCoalesceMetadataFreezeGuards:
    def test_direct_construction_with_raw_dict_freezes(self) -> None:
        """Even if someone bypasses factories, freeze_fields catches it."""
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            branches_lost={"a": "timeout"},
        )
        assert isinstance(meta.branches_lost, MappingProxyType)

    def test_direct_construction_with_raw_collisions_freezes(self) -> None:
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            union_field_collisions={"x": ("a", "b")},
        )
        assert isinstance(meta.union_field_collisions, MappingProxyType)


class TestCoalesceMetadataDetachment:
    """Regression: factory methods must not alias caller-owned dicts.

    Bug: for_failure() and for_merge() wrapped branches_lost with
    MappingProxyType(branches_lost), creating a read-only VIEW of the
    caller's dict. deep_freeze then returned it unchanged (identity
    optimization), so caller mutations leaked into frozen audit metadata.
    """

    def test_for_failure_detaches_branches_lost(self) -> None:
        caller_dict = {"branch_a": "timeout"}
        meta = CoalesceMetadata.for_failure(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=["branch_a", "branch_b"],
            branches_arrived=["branch_b"],
            branches_lost=caller_dict,
        )
        caller_dict["branch_a"] = "mutated"
        caller_dict["branch_c"] = "added"

        assert meta.branches_lost is not None
        assert meta.branches_lost["branch_a"] == "timeout", "Caller mutation leaked into frozen CoalesceMetadata.branches_lost"
        assert "branch_c" not in meta.branches_lost

    def test_for_merge_detaches_branches_lost(self) -> None:
        caller_dict: dict[str, str] = {"branch_a": "not_arrived"}
        meta = CoalesceMetadata.for_merge(
            policy=CoalescePolicy.BEST_EFFORT,
            merge_strategy=MergeStrategy.UNION,
            expected_branches=["branch_a", "branch_b"],
            branches_arrived=["branch_b"],
            branches_lost=caller_dict,
            arrival_order=[],
            wait_duration_ms=100.0,
        )
        caller_dict["branch_a"] = "mutated"

        assert meta.branches_lost is not None
        assert meta.branches_lost["branch_a"] == "not_arrived"


class TestCoalesceMetadataDeepFreeze:
    """Sequence fields must be deeply frozen on direct construction."""

    def test_expected_branches_frozen(self) -> None:
        branches: list[str] = ["a", "b"]
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            expected_branches=branches,  # type: ignore[arg-type]
        )
        branches.append("mutated")
        assert meta.expected_branches is not None
        assert isinstance(meta.expected_branches, tuple)
        assert "mutated" not in meta.expected_branches

    def test_branches_arrived_frozen(self) -> None:
        arrived: list[str] = ["a"]
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            branches_arrived=arrived,  # type: ignore[arg-type]
        )
        arrived.append("mutated")
        assert meta.branches_arrived is not None
        assert isinstance(meta.branches_arrived, tuple)
        assert "mutated" not in meta.branches_arrived

    def test_arrival_order_frozen(self) -> None:
        from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry

        entries = [ArrivalOrderEntry(branch="a", arrival_offset_ms=100.0)]
        meta = CoalesceMetadata(
            policy=CoalescePolicy.REQUIRE_ALL,
            arrival_order=entries,  # type: ignore[arg-type]
        )
        entries.append(ArrivalOrderEntry(branch="mutated", arrival_offset_ms=200.0))
        assert meta.arrival_order is not None
        assert isinstance(meta.arrival_order, tuple)
        assert len(meta.arrival_order) == 1


class TestWithUnionResult:
    """with_union_result layers union-merge provenance onto a base instance."""

    def test_with_union_result_records_origins(self) -> None:
        base = _base_merge_metadata()
        origins = {"field_x": "a", "field_y": "b"}

        meta = CoalesceMetadata.with_union_result(base, field_origins=origins)

        assert meta.union_field_origins is not None
        assert dict(meta.union_field_origins) == {"field_x": "a", "field_y": "b"}

    def test_with_union_result_records_collision_values(self) -> None:
        base = _base_merge_metadata()
        origins = {"field_x": "b"}
        collisions = {"field_x": ["a", "b"]}
        collision_values = {"field_x": [("a", 1), ("b", 2)]}

        meta = CoalesceMetadata.with_union_result(
            base,
            field_origins=origins,
            collisions=collisions,
            collision_values=collision_values,
        )

        assert meta.union_field_collisions is not None
        assert meta.union_field_collisions["field_x"] == ("a", "b")
        assert isinstance(meta.union_field_collisions["field_x"], tuple)
        assert meta.union_field_collision_values is not None
        entries = meta.union_field_collision_values["field_x"]
        assert entries == (("a", 1), ("b", 2))
        # Deeply tuple-ified: outer is a tuple, each entry is a tuple.
        assert isinstance(entries, tuple)
        for entry in entries:
            assert isinstance(entry, tuple)

    def test_with_union_result_no_collisions(self) -> None:
        base = _base_merge_metadata()
        origins = {"field_x": "a", "field_y": "b"}

        meta = CoalesceMetadata.with_union_result(base, field_origins=origins)

        assert meta.union_field_origins is not None
        assert dict(meta.union_field_origins) == origins
        assert meta.union_field_collisions is None
        assert meta.union_field_collision_values is None

    def test_union_field_origins_is_deep_frozen(self) -> None:
        base = _base_merge_metadata()
        meta = CoalesceMetadata.with_union_result(base, field_origins={"field_x": "a"})

        assert meta.union_field_origins is not None
        assert isinstance(meta.union_field_origins, MappingProxyType)
        with pytest.raises(TypeError):
            meta.union_field_origins["field_x"] = "mutated"  # type: ignore[index]

    def test_union_field_collision_values_is_deep_frozen(self) -> None:
        base = _base_merge_metadata()
        meta = CoalesceMetadata.with_union_result(
            base,
            field_origins={"field_x": "b"},
            collisions={"field_x": ["a", "b"]},
            collision_values={"field_x": [("a", 1), ("b", 2)]},
        )

        assert meta.union_field_collision_values is not None
        assert isinstance(meta.union_field_collision_values, MappingProxyType)
        with pytest.raises(TypeError):
            meta.union_field_collision_values["field_x"] = (  # type: ignore[index]
                ("c", 3),
            )
        # Inner tuple is immutable by type — verify it's actually a tuple.
        assert isinstance(meta.union_field_collision_values["field_x"], tuple)

    def test_metadata_to_dict_includes_origins_and_collision_values(self) -> None:
        base = _base_merge_metadata()
        meta = CoalesceMetadata.with_union_result(
            base,
            field_origins={"field_x": "b", "field_y": "a"},
            collisions={"field_x": ["a", "b"]},
            collision_values={"field_x": [("a", 1), ("b", 2)]},
        )

        d = meta.to_dict()

        assert d["union_field_origins"] == {"field_x": "b", "field_y": "a"}
        assert isinstance(d["union_field_origins"], dict)
        assert d["union_field_collision_values"] == {
            "field_x": [["a", 1], ["b", 2]],
        }
        # Inner entries must be plain lists, not tuples, for JSON shape.
        entries = d["union_field_collision_values"]["field_x"]
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, list)

    def test_with_union_result_preserves_empty_collisions_dict(self) -> None:
        """Empty ``collisions``/``collision_values`` dicts must be preserved.

        Regression guard mirroring ``TestForFailureTruthinessConflation``:
        ``{}`` means "no collisions observed" (a known state), which is
        distinct from ``None`` (not applicable / not yet populated).
        Truthiness predicates would conflate the two.
        """
        base = _base_merge_metadata()
        meta = CoalesceMetadata.with_union_result(
            base,
            field_origins={"field_x": "a"},
            collisions={},
            collision_values={},
        )

        assert meta.union_field_collisions is not None
        assert isinstance(meta.union_field_collisions, MappingProxyType)
        assert dict(meta.union_field_collisions) == {}
        assert meta.union_field_collision_values is not None
        assert isinstance(meta.union_field_collision_values, MappingProxyType)
        assert dict(meta.union_field_collision_values) == {}

    def test_union_field_collision_values_freezes_dict_valued_entries(self) -> None:
        """Dict-valued entries in collision_values must be deeply frozen.

        The value slot is typed ``Any`` because real union merges carry
        arbitrary row values — including dicts and lists. This test
        exercises the ``deep_freeze`` contract for that path.
        """
        base = _base_merge_metadata()
        meta = CoalesceMetadata.with_union_result(
            base,
            field_origins={"payload": "b"},
            collisions={"payload": ["a", "b"]},
            collision_values={
                "payload": [
                    ("a", {"nested": [1, 2]}),
                    ("b", {"nested": [3, 4]}),
                ]
            },
        )
        assert meta.union_field_collision_values is not None
        entries = meta.union_field_collision_values["payload"]

        # Inner dict value is deep-frozen to MappingProxyType.
        _, value_a = entries[0]
        assert isinstance(value_a, MappingProxyType)
        with pytest.raises(TypeError):
            value_a["nested"] = "mutated"  # type: ignore[index]

        # Nested list inside the dict is frozen to a tuple.
        assert isinstance(value_a["nested"], tuple)

    def test_metadata_to_dict_omits_origins_when_none(self) -> None:
        base = _base_merge_metadata()

        d = base.to_dict()

        assert "union_field_origins" not in d
        assert "union_field_collision_values" not in d
