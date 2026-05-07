"""Unit tests for ``SchemaConfig.participates_in_propagation`` — ADR-009 §Clause 1.

The property is the canonical participation predicate for ADR-007 pass-through
propagation. Its truth table is derived from ``has_effective_guarantees``: a
schema participates iff it has at least one source of guarantees (explicit
``guaranteed_fields`` OR typed ``fields``).
"""

from __future__ import annotations

from elspeth.contracts.schema import SchemaConfig


class TestParticipatesInPropagation:
    """Truth table of ``declares_guaranteed_fields`` x ``has_effective_guarantees``."""

    def test_declares_true_typed_true_participates(self) -> None:
        """Explicit guaranteed_fields AND typed fields → participates."""
        schema = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: str", "name: str"],
                "guaranteed_fields": ["id"],
            }
        )
        assert schema.declares_guaranteed_fields is True
        assert schema.has_effective_guarantees is True
        assert schema.participates_in_propagation is True

    def test_declares_false_typed_true_participates(self) -> None:
        """No explicit guarantees, but typed fields → still participates.

        This is the case ``declares_guaranteed_fields`` would miss. Fixed-mode
        schemas with required fields implicitly guarantee those fields via the
        type system; a downstream pass-through intersection should include
        them.
        """
        schema = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: str"],
            }
        )
        assert schema.declares_guaranteed_fields is False
        assert schema.has_effective_guarantees is True
        assert schema.participates_in_propagation is True

    def test_declares_true_typed_false_participates(self) -> None:
        """Explicit empty guarantees (None-vs-empty-tuple semantic) → participates.

        Constructed directly rather than via ``from_dict`` because the parser
        normalises ``guaranteed_fields=[]`` back to ``None``. The tuple form
        ``guaranteed_fields=()`` preserves the "explicit zero" distinction that
        downstream intersection code relies on.
        """
        schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=(),  # explicit empty tuple — not None
            required_fields=None,
            audit_fields=None,
        )
        assert schema.declares_guaranteed_fields is True
        assert schema.participates_in_propagation is True

    def test_declares_false_typed_false_abstains(self) -> None:
        """No explicit guarantees AND no typed fields → abstains from propagation.

        Fully-observed schemas with no declared fields abstain from the
        intersection.
        """
        schema = SchemaConfig.from_dict({"mode": "observed"})
        assert schema.declares_guaranteed_fields is False
        assert schema.has_effective_guarantees is False
        assert schema.participates_in_propagation is False

    def test_matches_has_effective_guarantees(self) -> None:
        """participates_in_propagation is currently an alias for has_effective_guarantees.

        The named property exists so future changes to propagation participation
        (e.g., Track 2's ``can_drop_rows``) can diverge from
        ``has_effective_guarantees`` without touching every caller. Today they
        agree; tomorrow they may not.
        """
        cases = [
            {"mode": "fixed", "fields": ["x: str"]},
            {"mode": "observed"},
            {"mode": "observed", "guaranteed_fields": ["y"]},
            {"mode": "flexible", "fields": ["a: int"], "guaranteed_fields": ["a"]},
        ]
        for cfg in cases:
            schema = SchemaConfig.from_dict(cfg)
            assert schema.participates_in_propagation == schema.has_effective_guarantees, (
                f"Drift detected between participates_in_propagation and has_effective_guarantees "
                f"for config {cfg!r}. Intentional divergence must be documented and reflected in "
                f"ADR-009 §Clause 1."
            )
