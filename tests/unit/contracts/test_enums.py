"""Tests for contracts enums."""

import pytest

from elspeth.contracts.enums import (
    _LEGAL_TERMINAL_PAIRS,
    _NON_TERMINAL_PATHS,
    RowOutcome,
    TerminalOutcome,
    TerminalPath,
)


class TestRowOutcome:
    """Tests for RowOutcome enum - stored in token_outcomes table (AUD-001)."""

    def test_terminal_mappings(self) -> None:
        """RowOutcome terminal/non-terminal mappings are correct."""
        from elspeth.contracts import RowOutcome

        terminal_outcomes = {
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.ROUTED_ON_ERROR,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.DIVERTED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.DROPPED_BY_FILTER,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
        }
        non_terminal_outcomes = {RowOutcome.BUFFERED}

        assert {o for o in RowOutcome if o.is_terminal} == terminal_outcomes
        assert {o for o in RowOutcome if not o.is_terminal} == non_terminal_outcomes


# ADR-019 mapping table — see docs/architecture/adr/019-two-axis-terminal-model.md
# § Mapping table (lines 99-115).  Each ``RowOutcome`` member maps to one or
# more ``(TerminalOutcome | None, TerminalPath)`` pairs.  ``DIVERTED`` is the
# sole RowOutcome that maps to TWO pairs (failsink-mode TRANSIENT vs
# discard-mode FAILURE — see ADR-019 Sub-decision 5).  ``BUFFERED`` is the
# sole non-terminal entry; its outcome is ``None`` per ADR-019 § Decision.
#
# This dict is the machine-checkable backstop for Stage 4's mechanical
# translation of test assertion sites: when a test currently asserts
# ``outcome == RowOutcome.X``, Stage 4 will rewrite it as
# ``(outcome, path) == ROW_OUTCOME_TO_TWO_AXIS_MAPPING[RowOutcome.X]``
# (or its DIVERTED-case sibling).  If the dict and the recorder/accumulator
# disagree at Stage 2/3, the property test below catches it before merge.
_ROW_OUTCOME_TO_TWO_AXIS_MAPPING: dict[RowOutcome, frozenset[tuple[TerminalOutcome | None, TerminalPath]]] = {
    RowOutcome.COMPLETED: frozenset({(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)}),
    RowOutcome.ROUTED: frozenset({(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)}),
    RowOutcome.ROUTED_ON_ERROR: frozenset({(TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED)}),
    RowOutcome.DROPPED_BY_FILTER: frozenset({(TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED)}),
    RowOutcome.COALESCED: frozenset({(TerminalOutcome.SUCCESS, TerminalPath.COALESCED)}),
    RowOutcome.FAILED: frozenset({(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)}),
    RowOutcome.QUARANTINED: frozenset({(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)}),
    RowOutcome.DIVERTED: frozenset(
        {
            (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK),
            (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED),
        }
    ),
    RowOutcome.FORKED: frozenset({(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)}),
    RowOutcome.EXPANDED: frozenset({(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT)}),
    RowOutcome.CONSUMED_IN_BATCH: frozenset({(TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED)}),
    RowOutcome.BUFFERED: frozenset({(None, TerminalPath.BUFFERED)}),
}


class TestTwoAxisMapping:
    """Property tests for ADR-019 (RowOutcome ↔ (TerminalOutcome, TerminalPath)) mapping.

    These tests are the machine-checkable backstop for the migration's
    Stage 4 mechanical-translation step.  A drift between the mapping table
    and the recorder/accumulator is a Stage 2/3 bug to be fixed in those
    PRs, not absorbed by silent test relaxation.
    """

    def test_every_row_outcome_is_mapped(self) -> None:
        """Every RowOutcome member appears as a key in the mapping table."""
        assert set(_ROW_OUTCOME_TO_TWO_AXIS_MAPPING.keys()) == set(RowOutcome)

    def test_terminal_pairs_match_legal_set(self) -> None:
        """Terminal (outcome, path) pairs in the mapping match _LEGAL_TERMINAL_PAIRS.

        Excludes the BUFFERED row (whose outcome is None, not a TerminalOutcome).
        """
        terminal_pairs_in_mapping: set[tuple[TerminalOutcome, TerminalPath]] = set()
        for pairs in _ROW_OUTCOME_TO_TWO_AXIS_MAPPING.values():
            for outcome, path in pairs:
                if outcome is None:
                    # Non-terminal row (only BUFFERED today).
                    continue
                terminal_pairs_in_mapping.add((outcome, path))
        assert terminal_pairs_in_mapping == set(_LEGAL_TERMINAL_PAIRS)

    def test_non_terminal_path_is_buffered(self) -> None:
        """The only RowOutcome with a None outcome maps to a non-terminal path."""
        non_terminal_pairs: set[tuple[None, TerminalPath]] = set()
        for pairs in _ROW_OUTCOME_TO_TWO_AXIS_MAPPING.values():
            for outcome, path in pairs:
                if outcome is None:
                    non_terminal_pairs.add((outcome, path))
        assert non_terminal_pairs == {(None, TerminalPath.BUFFERED)}
        assert {path for _, path in non_terminal_pairs} == set(_NON_TERMINAL_PATHS)

    def test_diverted_has_two_flavors(self) -> None:
        """DIVERTED maps to BOTH failsink-mode (TRANSIENT) and discard-mode (FAILURE).

        See ADR-019 § Sub-decisions Resolved by Panel Review (verdict 5):
        discard-mode DIVERTED is reclassified as a predicate-input FAILURE
        because no failsink absorbs the row and the primary node_state is
        FAILED (sink.py:991).  Failsink-mode DIVERTED remains TRANSIENT
        because its lifecycle answer lives on the failsink's COMPLETED
        node_state plus registered artifact.
        """
        assert _ROW_OUTCOME_TO_TWO_AXIS_MAPPING[RowOutcome.DIVERTED] == frozenset(
            {
                (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK),
                (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED),
            }
        )

    def test_terminal_pair_uniqueness_within_mapping(self) -> None:
        """No (outcome, path) pair appears under two different RowOutcome keys.

        Bijection invariant: each legal (outcome, path) pair has exactly one
        canonical RowOutcome ancestor (DIVERTED owns two pairs; no other
        sharing).  A pair appearing under two keys would mean Stage 4 has
        two valid translations for the same legacy assertion site, which is
        ambiguous and would block the mechanical edit.
        """
        seen: dict[tuple[TerminalOutcome | None, TerminalPath], RowOutcome] = {}
        for row_outcome, pairs in _ROW_OUTCOME_TO_TWO_AXIS_MAPPING.items():
            for pair in pairs:
                if pair in seen:
                    pytest.fail(f"Pair {pair} is mapped to both {seen[pair]} and {row_outcome} — bijection violated.")
                seen[pair] = row_outcome


class TestTerminalOutcome:
    """Tests for TerminalOutcome enum (ADR-019)."""

    def test_three_values(self) -> None:
        """TerminalOutcome has exactly SUCCESS, FAILURE, TRANSIENT."""
        assert {o.value for o in TerminalOutcome} == {
            "success",
            "failure",
            "transient",
        }

    def test_coercion_from_string(self) -> None:
        """TerminalOutcome can be created from string values (DB read path)."""
        assert TerminalOutcome("success") == TerminalOutcome.SUCCESS
        assert TerminalOutcome("failure") == TerminalOutcome.FAILURE
        assert TerminalOutcome("transient") == TerminalOutcome.TRANSIENT

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError — no silent fallback."""
        with pytest.raises(ValueError):
            TerminalOutcome("invalid")


class TestTerminalPath:
    """Tests for TerminalPath enum (ADR-019)."""

    def test_thirteen_values(self) -> None:
        """TerminalPath has 13 values: 12 terminal paths + BUFFERED."""
        assert {p.value for p in TerminalPath} == {
            "default_flow",
            "gate_routed",
            "on_error_routed",
            "filter_dropped",
            "coalesced",
            "unrouted",
            "quarantined_at_source",
            "sink_fallback_to_failsink",
            "sink_discarded",
            "fork_parent",
            "expand_parent",
            "batch_consumed",
            "buffered",
        }

    def test_coercion_from_string(self) -> None:
        """TerminalPath can be created from string values (DB read path)."""
        assert TerminalPath("default_flow") == TerminalPath.DEFAULT_FLOW
        assert TerminalPath("buffered") == TerminalPath.BUFFERED

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError — no silent fallback."""
        with pytest.raises(ValueError):
            TerminalPath("invalid")


class TestEnumCoercion:
    """Verify enums that ARE stored can be created from string values."""

    def test_run_status_from_string(self) -> None:
        """Can create RunStatus from string (for DB reads)."""
        from elspeth.contracts import RunStatus

        assert RunStatus("running") == RunStatus.RUNNING
        assert RunStatus("completed") == RunStatus.COMPLETED

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError - no silent fallback."""
        from elspeth.contracts import RunStatus

        with pytest.raises(ValueError):
            RunStatus("invalid")


class TestTriggerType:
    """TriggerType must match the set of trigger causes the engine can actually emit."""

    def test_values_match_current_engine_producers(self) -> None:
        from elspeth.contracts import TriggerType

        assert {trigger.value for trigger in TriggerType} == {
            "count",
            "timeout",
            "condition",
            "end_of_source",
        }
