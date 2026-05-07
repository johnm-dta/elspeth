"""Unit tests for ``compose_propagation`` — the ADR-009 §Clause 1 primitive.

The 11 cases below pin the ADR-007 propagation semantics table. Any behaviour
change here must be accompanied by an ADR amendment; the table is the
specification, not an implementation detail.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.guarantee_propagation import compose_propagation


class TestComposePropagation:
    """ADR-007 propagation semantics table — 11 pinned cases."""

    def test_01_single_predecessor_abstains_returns_self(self) -> None:
        """None in predecessor list is skipped; result is self_fields."""
        assert compose_propagation(frozenset({"a"}), [None]) == frozenset({"a"})

    def test_02_single_predecessor_empty_collapses_to_self(self) -> None:
        """Empty frozenset is explicit-zero participation; intersection is empty."""
        assert compose_propagation(frozenset({"a"}), [frozenset()]) == frozenset({"a"})

    def test_03_single_predecessor_nonempty_unioned_with_self(self) -> None:
        """Single participant intersection is the participant itself."""
        assert compose_propagation(frozenset({"a"}), [frozenset({"b", "c"})]) == frozenset({"a", "b", "c"})

    def test_04_two_participating_with_overlap_intersects(self) -> None:
        """Multiple participants: intersect them, then union with self."""
        assert compose_propagation(
            frozenset({"a"}),
            [frozenset({"b", "c", "d"}), frozenset({"c", "d", "e"})],
        ) == frozenset({"a", "c", "d"})

    def test_05_two_predecessors_one_abstains_intersects_over_one(self) -> None:
        """Abstaining predecessor is skipped; single remaining participant used."""
        assert compose_propagation(
            frozenset(),
            [None, frozenset({"b", "c"})],
        ) == frozenset({"b", "c"})

    def test_06_all_predecessors_abstain_returns_self_only(self) -> None:
        """All None: result is self_fields unchanged."""
        assert compose_propagation(frozenset({"a"}), [None, None, None]) == frozenset({"a"})

    def test_07_closed_mode_upstream_only_its_effective_set_propagates(self) -> None:
        """Closed-mode upstream: only its effective guarantees propagate (computed at caller)."""
        # Caller computes the upstream's effective set; compose_propagation
        # receives the already-computed frozenset. For a closed-mode upstream,
        # that's the full declared set.
        upstream_effective = frozenset({"x", "y"})
        assert compose_propagation(
            frozenset(),
            [upstream_effective],
        ) == frozenset({"x", "y"})

    def test_08_coalesce_upstream_single_participant(self) -> None:
        """Coalesce node presents a single pre-computed participant."""
        coalesce_output = frozenset({"merged_a", "merged_b"})
        assert (
            compose_propagation(
                frozenset(),
                [coalesce_output],
            )
            == coalesce_output
        )

    def test_09_output_schema_config_none_self_abstains_returns_inherited(self) -> None:
        """If self_fields is empty (schema is None), result is the inherited intersection."""
        assert compose_propagation(
            frozenset(),
            [frozenset({"a", "b"}), frozenset({"b", "c"})],
        ) == frozenset({"b"})

    def test_10_creates_tokens_union_at_caller_compose_sees_normal_inputs(self) -> None:
        """creates_tokens=True is a caller-level concern; compose_propagation is oblivious."""
        # The caller decides the self_fields and predecessor_guarantees. If the
        # caller has already unioned in creates_tokens-derived fields, they
        # arrive here as normal inputs.
        self_with_created_tokens = frozenset({"a", "synthetic_token_id"})
        upstream = frozenset({"b"})
        assert compose_propagation(self_with_created_tokens, [upstream]) == frozenset({"a", "synthetic_token_id", "b"})

    def test_11_both_self_and_intersection_empty_yields_empty(self) -> None:
        """Degenerate: no self fields, intersection collapses to empty set."""
        assert (
            compose_propagation(
                frozenset(),
                [frozenset({"a"}), frozenset({"b"})],
            )
            == frozenset()
        )

    # --- property-style sanity tests (not Hypothesis; fixed inputs) ----

    def test_result_is_frozenset(self) -> None:
        """Return type is always frozenset[str] — not set, not list."""
        result = compose_propagation(frozenset({"a"}), [frozenset({"b"})])
        assert isinstance(result, frozenset)

    def test_empty_sequence_returns_self_fields(self) -> None:
        """No predecessor_guarantees at all: return self unchanged."""
        assert compose_propagation(frozenset({"a", "b"}), []) == frozenset({"a", "b"})

    def test_pure_function_same_inputs_same_output(self) -> None:
        """Determinism: same inputs always return same output."""
        self_fields = frozenset({"x"})
        preds = [frozenset({"y"}), None, frozenset({"y", "z"})]
        assert compose_propagation(self_fields, preds) == compose_propagation(self_fields, preds)

    @pytest.mark.parametrize(
        "order",
        [
            [frozenset({"a", "b"}), frozenset({"b", "c"})],
            [frozenset({"b", "c"}), frozenset({"a", "b"})],
        ],
    )
    def test_intersection_order_invariant(self, order: list[frozenset[str]]) -> None:
        """Intersection is commutative: order of participants does not matter."""
        assert compose_propagation(frozenset(), order) == frozenset({"b"})
