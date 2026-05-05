"""Tests for contracts enums."""

import pytest

import elspeth.contracts as contracts
import elspeth.contracts.enums as enums
from elspeth.contracts.enums import (
    _LEGAL_TERMINAL_PAIRS,
    _NON_TERMINAL_PATHS,
    TerminalOutcome,
    TerminalPath,
)


class TestLegacySingleAxisEnumRemoved:
    """Stage 5 removes the old single-axis terminal enum from public exports."""

    def test_legacy_enum_is_not_exported(self) -> None:
        legacy_name = "Row" + "Outcome"

        assert not hasattr(enums, legacy_name)
        assert not hasattr(contracts, legacy_name)
        assert legacy_name not in contracts.__all__


class TestTwoAxisMapping:
    """Property tests for ADR-019 terminal pair coverage."""

    def test_terminal_pairs_match_legal_set(self) -> None:
        """The legal terminal pair set matches the ADR-019 mapping table."""
        assert set(_LEGAL_TERMINAL_PAIRS) == {
            (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW),
            (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED),
            (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED),
            (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED),
            (TerminalOutcome.SUCCESS, TerminalPath.COALESCED),
            (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
            (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE),
            (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK),
            (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED),
            (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT),
            (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT),
            (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED),
        }

    def test_non_terminal_path_is_buffered(self) -> None:
        """The only non-terminal path is BUFFERED."""
        assert set(_NON_TERMINAL_PATHS) == {TerminalPath.BUFFERED}

    def test_diverted_has_two_flavors(self) -> None:
        """Sink fallback and discard mode remain distinct terminal pair flavors."""
        assert (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK) in _LEGAL_TERMINAL_PAIRS
        assert (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED) in _LEGAL_TERMINAL_PAIRS

    def test_terminal_pair_uniqueness_within_mapping(self) -> None:
        """No path appears under two different lifecycle outcomes."""
        seen_paths: set[TerminalPath] = set()
        for _, path in _LEGAL_TERMINAL_PAIRS:
            if path in seen_paths:
                pytest.fail(f"Path {path} appears in more than one terminal pair.")
            seen_paths.add(path)


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
