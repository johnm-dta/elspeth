"""Tests for lineage.explain() — complete lineage query composition.

Tests cover:
- explain() with token_id (direct lookup)
- explain() with row_id (single terminal token resolution)
- explain() with row_id + sink (disambiguation)
- explain() raises when neither token_id nor row_id provided
- explain() raises when row_id has multiple terminal tokens without sink
- explain() returns None for unknown token/row
- LineageResult fields populated correctly
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from elspeth.contracts.audit import (
    RowLineage,
    Token,
    TokenOutcome,
    TokenParent,
)
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.lineage import LineageResult, explain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeQuery:
    token: Token | None = None
    row_lineage: RowLineage | None = None
    node_states: list[object] = field(default_factory=list)
    routing_events: list[object] = field(default_factory=list)
    calls: list[object] = field(default_factory=list)
    token_parents: list[TokenParent] = field(default_factory=list)
    token_side_effect: Callable[[str], Token | None] | None = None
    get_token_calls: list[str] = field(default_factory=list)
    get_tokens_by_ids_calls: list[tuple[str, ...]] = field(default_factory=list)
    explain_row_calls: list[tuple[str, str]] = field(default_factory=list)

    def get_token(self, token_id: str) -> Token | None:
        self.get_token_calls.append(token_id)
        if self.token_side_effect is not None:
            return self.token_side_effect(token_id)
        return self.token

    def get_token_for_run(self, run_id: str, token_id: str) -> Token | None:
        self.get_token_calls.append(token_id)
        if self.token_side_effect is not None:
            token = self.token_side_effect(token_id)
        else:
            token = self.token
        if token is None or token.run_id != run_id:
            return None
        return token

    def get_tokens_by_ids(self, token_ids: list[str]) -> list[Token]:
        self.get_tokens_by_ids_calls.append(tuple(token_ids))
        tokens: list[Token] = []
        for token_id in token_ids:
            if self.token_side_effect is not None:
                token = self.token_side_effect(token_id)
            else:
                token = self.token
            if token is not None:
                tokens.append(token)
        return tokens

    def explain_row(self, _run_id: str, _row_id: str) -> RowLineage | None:
        self.explain_row_calls.append((_run_id, _row_id))
        return self.row_lineage

    def get_node_states_for_token(self, _token_id: str) -> list[object]:
        return self.node_states

    def get_routing_events_for_states(self, _state_ids: list[str]) -> list[object]:
        return self.routing_events

    def get_calls_for_states(self, _state_ids: list[str]) -> list[object]:
        return self.calls

    def get_token_parents(self, _token_id: str) -> list[TokenParent]:
        return self.token_parents

    def get_scheduler_events(self, *, run_id: str, token_id: str) -> list[object]:
        return []

    def assert_get_token_called_with(self, token_id: str) -> None:
        assert self.get_token_calls
        assert self.get_token_calls[-1] == token_id


@dataclass
class _FakeDataFlow:
    token_outcomes: list[TokenOutcome] = field(default_factory=list)
    validation_errors: list[object] = field(default_factory=list)
    transform_errors: list[object] = field(default_factory=list)
    token_outcome: TokenOutcome | None = None

    def get_token_outcomes_for_row(self, _run_id: str, _row_id: str) -> list[TokenOutcome]:
        return self.token_outcomes

    def get_validation_errors_for_row(
        self,
        _run_id: str,
        _source_data_hash: str,
        *,
        row_id: str,
    ) -> list[object]:
        return self.validation_errors

    def get_transform_errors_for_token(self, _token_id: str) -> list[object]:
        return self.transform_errors

    def get_token_outcome(self, _token_id: str) -> TokenOutcome | None:
        return self.token_outcome


@dataclass
class _FakeFactory:
    query: _FakeQuery
    data_flow: _FakeDataFlow


def _make_factory(
    *,
    token: Token | None = None,
    row_lineage: RowLineage | None = None,
    node_states: list[object] | None = None,
    routing_events: list[object] | None = None,
    calls: list[object] | None = None,
    token_parents: list[TokenParent] | None = None,
    token_outcomes: list[TokenOutcome] | None = None,
    validation_errors: list[object] | None = None,
    transform_errors: list[object] | None = None,
    token_outcome: TokenOutcome | None = None,
) -> _FakeFactory:
    """Create a fake repository factory with configurable return values."""
    return _FakeFactory(
        query=_FakeQuery(
            token=token,
            row_lineage=row_lineage,
            node_states=node_states or [],
            routing_events=routing_events or [],
            calls=calls or [],
            token_parents=token_parents or [],
        ),
        data_flow=_FakeDataFlow(
            token_outcomes=token_outcomes or [],
            validation_errors=validation_errors or [],
            transform_errors=transform_errors or [],
            token_outcome=token_outcome,
        ),
    )


def _make_token(
    token_id: str = "tok-1",
    row_id: str = "row-1",
    *,
    run_id: str = "run-1",
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
) -> Token:
    from datetime import UTC, datetime

    return Token(
        token_id=token_id,
        row_id=row_id,
        created_at=datetime(2026, 1, 15, tzinfo=UTC),
        run_id=run_id,
        fork_group_id=fork_group_id,
        join_group_id=join_group_id,
        expand_group_id=expand_group_id,
    )


def _make_outcome(
    token_id: str = "tok-1",
    outcome: TerminalOutcome | None = TerminalOutcome.SUCCESS,
    path: TerminalPath = TerminalPath.DEFAULT_FLOW,
    completed: bool = True,
    sink_name: str | None = None,
) -> TokenOutcome:
    from datetime import UTC, datetime

    return TokenOutcome(
        outcome_id="out-1",
        run_id="run-1",
        token_id=token_id,
        outcome=outcome,
        path=path,
        completed=completed,
        recorded_at=datetime(2026, 1, 15, tzinfo=UTC),
        sink_name=sink_name,
    )


def _make_row_lineage() -> RowLineage:
    from datetime import UTC, datetime

    return RowLineage(
        row_id="row-1",
        run_id="run-1",
        source_node_id="source-0",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        source_data_hash="hash-1",
        created_at=datetime(2026, 1, 15, tzinfo=UTC),
        source_data={"id": 1, "name": "test"},
        payload_available=True,
    )


# ===========================================================================
# Validation
# ===========================================================================


class TestExplainValidation:
    """Tests for explain() argument validation."""

    def test_raises_when_neither_token_nor_row_provided(self) -> None:
        factory = _make_factory()
        with pytest.raises(ValueError, match="Must provide either token_id or row_id"):
            explain(factory.query, factory.data_flow, "run-1")

    def test_raises_when_multiple_terminal_tokens_without_sink(self) -> None:
        """Row with multiple terminal tokens and no sink filter raises."""
        outcomes = [
            _make_outcome(token_id="tok-1", sink_name="output"),
            _make_outcome(token_id="tok-2", sink_name="errors"),
        ]
        factory = _make_factory(token_outcomes=outcomes)
        with pytest.raises(ValueError, match="has 2 terminal tokens"):
            explain(factory.query, factory.data_flow, "run-1", row_id="row-1")

    def test_raises_when_multiple_tokens_at_same_sink(self) -> None:
        """Multiple tokens reaching same sink raises (fork ambiguity)."""
        outcomes = [
            _make_outcome(token_id="tok-1", sink_name="output"),
            _make_outcome(token_id="tok-2", sink_name="output"),
        ]
        factory = _make_factory(token_outcomes=outcomes)
        with pytest.raises(ValueError, match="has 2 tokens at sink"):
            explain(factory.query, factory.data_flow, "run-1", row_id="row-1", sink="output")


# ===========================================================================
# Token ID lookup
# ===========================================================================


class TestExplainByTokenId:
    """Tests for explain() with direct token_id."""

    def test_returns_lineage_for_known_token(self) -> None:
        token = _make_token()
        row_lineage = _make_row_lineage()
        factory = _make_factory(token=token, row_lineage=row_lineage)

        result = explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")
        assert result is not None
        assert result.token.token_id == "tok-1"
        assert result.source_row.row_id == "row-1"

    def test_returns_none_for_unknown_token(self) -> None:
        factory = _make_factory(token=None)
        result = explain(factory.query, factory.data_flow, "run-1", token_id="nonexistent")
        assert result is None

    def test_raises_when_row_not_found_for_known_token(self) -> None:
        """Token exists but its row_id doesn't — Tier 1 corruption."""
        token = _make_token()
        factory = _make_factory(token=token, row_lineage=None)
        with pytest.raises(AuditIntegrityError, match="does not exist in rows table"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")


# ===========================================================================
# Row ID resolution
# ===========================================================================


class TestExplainByRowId:
    """Tests for explain() with row_id resolution."""

    def test_resolves_single_terminal_token(self) -> None:
        token = _make_token()
        row_lineage = _make_row_lineage()
        outcomes = [_make_outcome(token_id="tok-1")]
        factory = _make_factory(
            token=token,
            row_lineage=row_lineage,
            token_outcomes=outcomes,
        )
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1")
        assert result is not None
        assert result.token.token_id == "tok-1"

    def test_returns_none_when_no_outcomes(self) -> None:
        factory = _make_factory(token_outcomes=[])
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1")
        assert result is None

    def test_returns_none_when_all_non_terminal(self) -> None:
        outcomes = [_make_outcome(outcome=None, path=TerminalPath.BUFFERED, completed=False)]
        factory = _make_factory(token_outcomes=outcomes)
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1")
        assert result is None

    def test_filters_by_sink(self) -> None:
        token = _make_token(token_id="tok-2")
        row_lineage = _make_row_lineage()
        outcomes = [
            _make_outcome(token_id="tok-1", sink_name="output"),
            _make_outcome(token_id="tok-2", sink_name="errors"),
        ]
        factory = _make_factory(
            token=token,
            row_lineage=row_lineage,
            token_outcomes=outcomes,
        )
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1", sink="errors")
        assert result is not None
        # Verify get_token was called with tok-2 (the errors sink token)
        factory.query.assert_get_token_called_with("tok-2")

    def test_returns_none_when_sink_has_no_tokens(self) -> None:
        outcomes = [_make_outcome(token_id="tok-1", sink_name="output")]
        factory = _make_factory(token_outcomes=outcomes)
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1", sink="nonexistent")
        assert result is None


# ===========================================================================
# Tier 1 corruption detection
# ===========================================================================


class TestExplainTier1Corruption:
    """Tests for explain() detecting Tier 1 audit data corruption."""

    def test_resolved_token_missing_raises_audit_integrity(self) -> None:
        """Token resolved from token_outcomes but missing from tokens table."""
        outcomes = [_make_outcome(token_id="tok-resolved")]
        factory = _make_factory(token=None, token_outcomes=outcomes)

        with pytest.raises(AuditIntegrityError, match="resolved from token_outcomes"):
            explain(factory.query, factory.data_flow, "run-1", row_id="row-1")

    def test_lineage_result_row_id_mismatch_raises_audit_integrity(self) -> None:
        """LineageResult rejects mismatched token/row IDs as corruption."""
        row_lineage = _make_row_lineage()  # has row_id="row-1"
        # Manually create a mismatch by making a token with different row_id
        mismatched_token = _make_token(token_id="tok-1", row_id="row-WRONG")

        with pytest.raises(AuditIntegrityError, match="row_id mismatch"):
            LineageResult(
                token=mismatched_token,
                source_row=row_lineage,
                node_states=(),
                routing_events=(),
                calls=(),
                parent_tokens=(),
            )

    def test_lineage_result_run_id_mismatch_raises_audit_integrity(self) -> None:
        """LineageResult rejects mismatched token/source row runs as corruption."""
        row_lineage = _make_row_lineage()  # has run_id="run-1"
        mismatched_token = _make_token(token_id="tok-1", row_id="row-1", run_id="run-2")

        with pytest.raises(AuditIntegrityError, match="run_id mismatch"):
            LineageResult(
                token=mismatched_token,
                source_row=row_lineage,
                node_states=(),
                routing_events=(),
                calls=(),
                parent_tokens=(),
            )

    def test_explain_direct_token_run_mismatch_returns_none(self) -> None:
        """Caller-provided token IDs are scoped to the requested run."""
        token = _make_token(token_id="tok-cross-run", row_id="row-1", run_id="run-2")
        row_lineage = _make_row_lineage()  # same row_id, requested/source run is run-1
        factory = _make_factory(token=token, row_lineage=row_lineage)

        result = explain(factory.query, factory.data_flow, "run-1", token_id="tok-cross-run")

        assert result is None
        assert factory.query.explain_row_calls == []


# ===========================================================================
# Parent token integrity
# ===========================================================================


class TestExplainParentIntegrity:
    """Tests for explain() parent token validation."""

    def test_fork_token_without_parents_raises(self) -> None:
        """Token with fork_group_id but no parents is audit corruption."""
        token = _make_token(fork_group_id="fg-1")
        row_lineage = _make_row_lineage()
        factory = _make_factory(
            token=token,
            row_lineage=row_lineage,
            token_parents=[],
        )
        with pytest.raises(AuditIntegrityError, match="Audit integrity violation"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_parent_token_not_found_raises(self) -> None:
        """Missing parent token is audit corruption."""
        token = _make_token(fork_group_id="fork-group-1")
        row_lineage = _make_row_lineage()
        parent_ref = TokenParent(token_id="tok-1", parent_token_id="missing-parent", ordinal=0)
        factory = _make_factory(
            token=token,
            row_lineage=row_lineage,
            token_parents=[parent_ref],
        )
        # get_token returns the main token for tok-1, None for missing-parent
        factory.query.token_side_effect = lambda tid: token if tid == "tok-1" else None

        with pytest.raises(AuditIntegrityError, match=r"parent token.*not found"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_parent_from_different_run_raises(self) -> None:
        """Parent token from a different run is audit corruption.

        Regression test for elspeth-4126073aa8: explain() previously only checked
        that parent tokens exist, not that they belong to the same run.
        """
        child_token = _make_token(token_id="child-1", fork_group_id="fg-1")
        # Parent belongs to a different run
        parent_token = Token(
            token_id="parent-1",
            row_id="row-1",
            created_at=child_token.created_at,
            run_id="DIFFERENT-RUN",  # <-- cross-run corruption
        )
        row_lineage = _make_row_lineage()
        parent_ref = TokenParent(token_id="child-1", parent_token_id="parent-1", ordinal=0)
        factory = _make_factory(
            token=child_token,
            row_lineage=row_lineage,
            token_parents=[parent_ref],
        )
        factory.query.token_side_effect = lambda tid: child_token if tid == "child-1" else parent_token

        with pytest.raises(AuditIntegrityError, match="Cross-run parent lineage"):
            explain(factory.query, factory.data_flow, "run-1", token_id="child-1")

    def test_parent_tokens_are_hydrated_in_one_batch(self) -> None:
        """Parent hydration must not call get_token once per parent."""
        child_token = _make_token(token_id="child-1", join_group_id="join-1")
        parent_1 = _make_token(token_id="parent-1", row_id="row-parent-1")
        parent_2 = _make_token(token_id="parent-2", row_id="row-parent-2")
        parent_refs = [
            TokenParent(token_id="child-1", parent_token_id="parent-1", ordinal=0),
            TokenParent(token_id="child-1", parent_token_id="parent-2", ordinal=1),
        ]
        factory = _make_factory(
            token=child_token,
            row_lineage=_make_row_lineage(),
            token_parents=parent_refs,
        )
        tokens_by_id = {
            child_token.token_id: child_token,
            parent_1.token_id: parent_1,
            parent_2.token_id: parent_2,
        }
        factory.query.token_side_effect = tokens_by_id.get

        result = explain(factory.query, factory.data_flow, "run-1", token_id="child-1")

        assert result is not None
        assert [token.token_id for token in result.parent_tokens] == ["parent-1", "parent-2"]
        assert factory.query.get_tokens_by_ids_calls == [("parent-1", "parent-2")]
        assert factory.query.get_token_calls == ["child-1"]

    def test_join_token_without_parents_raises(self) -> None:
        """Token with join_group_id but no parents is audit corruption."""
        token = _make_token(join_group_id="jg-1")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
            token_parents=[],
        )
        with pytest.raises(AuditIntegrityError, match=r"join_group_id.*no parent"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_expand_token_without_parents_raises(self) -> None:
        """Token with expand_group_id but no parents is audit corruption."""
        token = _make_token(expand_group_id="eg-1")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
            token_parents=[],
        )
        with pytest.raises(AuditIntegrityError, match=r"expand_group_id.*no parent"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_parents_without_any_group_id_raises(self) -> None:
        """Token with parent records but no group ID is audit corruption.

        Exercises lineage.py lines 217-222: if parents exist but no
        fork/join/expand group_id is set, the parent relationships have
        no lineage operation to belong to.
        """
        token = _make_token()  # All group IDs are None
        parent_ref = TokenParent(token_id="tok-1", parent_token_id="parent-1", ordinal=0)
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
            token_parents=[parent_ref],
        )
        with pytest.raises(AuditIntegrityError, match=r"parent relationships.*but no group ID"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")


# ===========================================================================
# LineageResult structure
# ===========================================================================


class TestLineageResult:
    """Tests for LineageResult dataclass."""

    def test_has_expected_fields(self) -> None:
        token = _make_token()
        row_lineage = _make_row_lineage()
        result = LineageResult(
            token=token,
            source_row=row_lineage,
            node_states=(),
            routing_events=(),
            calls=(),
            parent_tokens=(),
        )
        assert result.token is token
        assert result.source_row is row_lineage
        assert result.validation_errors == ()
        assert result.transform_errors == ()
        assert result.outcome is None


# ===========================================================================
# Group ID validation — kill ZeroIterationForLoop & comparison survivors
# ===========================================================================


class TestExplainGroupIdValidation:
    """Kill mutants on lineage.py lines 195-196.

    ZeroIterationForLoop: the entire group ID validation for-loop can be
    deleted without any test failing. These tests prove the loop is
    exercised and catches corrupted empty-string group IDs.

    Comparison mutations on ``gval == ""``: ``is ""``, ``< ""``,
    ``is not None → is None`` all survive without these tests.
    """

    def test_empty_fork_group_id_raises(self) -> None:
        """Token with fork_group_id='' is audit corruption — must raise."""
        token = _make_token(fork_group_id="")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
        )
        with pytest.raises(AuditIntegrityError, match=r"empty.*fork_group_id"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_empty_join_group_id_raises(self) -> None:
        """Token with join_group_id='' is audit corruption — must raise."""
        token = _make_token(join_group_id="")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
        )
        with pytest.raises(AuditIntegrityError, match=r"empty.*join_group_id"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_empty_expand_group_id_raises(self) -> None:
        """Token with expand_group_id='' is audit corruption — must raise."""
        token = _make_token(expand_group_id="")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
        )
        with pytest.raises(AuditIntegrityError, match=r"empty.*expand_group_id"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")

    def test_none_group_ids_accepted(self) -> None:
        """Token with all group IDs as None is valid (no fork/join/expand)."""
        token = _make_token()  # All group IDs default to None
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
        )
        result = explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")
        assert result is not None
        assert result.token.token_id == "tok-1"

    def test_two_group_ids_set_raises(self) -> None:
        """Token with both fork_group_id and join_group_id set is corruption.

        Kill mutant: ``len(set_groups) > 1`` → ``len(set_groups) > 2``.
        With ``> 2``, exactly 2 group IDs set would slip through without raising.
        """
        token = _make_token(fork_group_id="fg-1", join_group_id="jg-1")
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
        )
        with pytest.raises(AuditIntegrityError, match=r"multiple group IDs"):
            explain(factory.query, factory.data_flow, "run-1", token_id="tok-1")


# ===========================================================================
# Sink filter mutation kills — kill `==` → `is` and `==` → `<=` survivors
# ===========================================================================


class TestExplainSinkFilterEquality:
    """Kill mutants on line 122: ``o.sink_name == sink``.

    Mutant 1: ``==`` → ``is``. Survives due to Python string interning —
    short literal strings used in tests share the same object, so ``is``
    returns True. In production, strings from SQLAlchemy are separate objects.

    Mutant 2: ``==`` → ``<=``. Survives when only one outcome exists.
    With multiple sinks, ``<=`` would match sinks alphabetically before
    the target, changing the filter result set.
    """

    def test_sink_filter_uses_equality_not_identity(self) -> None:
        """Kill mutant: ``==`` → ``is`` on sink_name comparison.

        Construct the sink name at runtime from parts to prevent
        Python's string interning from making ``is`` equivalent to ``==``.
        """
        # Build "output" from parts — runtime concatenation avoids interning
        sink_name_for_query = "".join(["o", "u", "t", "p", "u", "t"])
        # Verify it's a different object (interning defeated)
        sink_name_in_outcome = "output"
        assert sink_name_for_query == sink_name_in_outcome
        # Not guaranteed to be different objects, but join() usually avoids interning.
        # The real defense: if ``is`` mutant is active and objects differ, filter returns [].

        token = _make_token(token_id="tok-1")
        outcomes = [_make_outcome(token_id="tok-1", sink_name=sink_name_in_outcome)]
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
            token_outcomes=outcomes,
        )

        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1", sink=sink_name_for_query)
        assert result is not None
        assert result.token.token_id == "tok-1"

    def test_sink_filter_uses_equality_not_lte(self) -> None:
        """Kill mutant: ``==`` → ``<=`` on sink_name comparison.

        With ``<=``, filtering for "beta" would also match "alpha"
        (since "alpha" <= "beta" is True), producing 2 matches instead
        of 1. The code would raise ValueError for multiple tokens at
        the same sink, when it should succeed with a single match.
        """
        token_beta = _make_token(token_id="tok-beta")
        outcomes = [
            _make_outcome(token_id="tok-alpha", sink_name="alpha_sink"),
            _make_outcome(token_id="tok-beta", sink_name="beta_sink"),
        ]
        factory = _make_factory(
            token=token_beta,
            row_lineage=_make_row_lineage(),
            token_outcomes=outcomes,
        )

        # With ==: only "beta_sink" matches → resolves tok-beta → success
        # With <=: "alpha_sink" <= "beta_sink" also matches → 2 results → raises ValueError
        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1", sink="beta_sink")
        assert result is not None
        factory.query.assert_get_token_called_with("tok-beta")

    def test_terminal_filter_excludes_non_terminal(self) -> None:
        """Verify non-terminal outcomes are excluded when mixed with terminal.

        Kills mutant: ``o.completed`` → ``not o.completed``.
        With the mutant, only BUFFERED tokens are kept and the real
        terminal token is dropped, returning None instead of a result.
        """
        token = _make_token(token_id="tok-terminal")
        outcomes = [
            _make_outcome(token_id="tok-buffered", outcome=None, path=TerminalPath.BUFFERED, completed=False),
            _make_outcome(token_id="tok-terminal", outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW, completed=True),
        ]
        factory = _make_factory(
            token=token,
            row_lineage=_make_row_lineage(),
            token_outcomes=outcomes,
        )

        result = explain(factory.query, factory.data_flow, "run-1", row_id="row-1")
        assert result is not None
        factory.query.assert_get_token_called_with("tok-terminal")
