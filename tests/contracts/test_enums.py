"""Tests for contracts enums."""

import pytest


class TestDeterminism:
    """Tests for Determinism enum - critical for replay/verify."""

    def test_has_all_required_values(self) -> None:
        """Determinism has exactly 6 values from architecture."""
        from elspeth.contracts import Determinism

        expected = {"DETERMINISTIC", "SEEDED", "IO_READ", "IO_WRITE", "EXTERNAL_CALL", "NON_DETERMINISTIC"}
        assert {e.name for e in Determinism} == expected

    def test_no_unknown_value(self) -> None:
        """Determinism must NOT have 'unknown' - we crash instead."""
        from elspeth.contracts import Determinism

        values = [d.value for d in Determinism]
        assert "unknown" not in values

    def test_string_values_match_architecture(self) -> None:
        """String values match architecture specification."""
        from elspeth.contracts import Determinism

        assert Determinism.DETERMINISTIC.value == "deterministic"
        assert Determinism.SEEDED.value == "seeded"
        assert Determinism.IO_READ.value == "io_read"
        assert Determinism.IO_WRITE.value == "io_write"
        assert Determinism.EXTERNAL_CALL.value == "external_call"
        assert Determinism.NON_DETERMINISTIC.value == "non_deterministic"


class TestRowOutcome:
    """Tests for RowOutcome enum - stored in token_outcomes table (AUD-001)."""

    def test_is_str_enum(self) -> None:
        """RowOutcome IS a (str, Enum) for database storage via token_outcomes table."""
        # AUD-001: RowOutcome is now explicitly recorded, not derived at query time.
        # The (str, Enum) base allows direct database storage.
        from elspeth.contracts import RowOutcome

        # (str, Enum) allows direct string comparison for database serialization
        assert RowOutcome.COMPLETED == "completed"  # type: ignore[comparison-overlap]
        # Can be created from string values (for DB reads)
        assert RowOutcome("completed") == RowOutcome.COMPLETED  # type: ignore[unreachable]

    def test_has_all_terminal_states(self) -> None:
        """RowOutcome has all values and correct string representations."""
        from elspeth.contracts import RowOutcome

        expected_values = {
            RowOutcome.COMPLETED: "completed",
            RowOutcome.ROUTED: "routed",
            RowOutcome.FORKED: "forked",
            RowOutcome.FAILED: "failed",
            RowOutcome.QUARANTINED: "quarantined",
            RowOutcome.CONSUMED_IN_BATCH: "consumed_in_batch",
            RowOutcome.COALESCED: "coalesced",
            RowOutcome.EXPANDED: "expanded",
            RowOutcome.BUFFERED: "buffered",
        }
        assert {o: o.value for o in RowOutcome} == expected_values

    def test_terminal_mappings(self) -> None:
        """RowOutcome terminal/non-terminal mappings are correct."""
        from elspeth.contracts import RowOutcome

        terminal_outcomes = {
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
        }
        non_terminal_outcomes = {RowOutcome.BUFFERED}

        assert {o for o in RowOutcome if o.is_terminal} == terminal_outcomes
        assert {o for o in RowOutcome if not o.is_terminal} == non_terminal_outcomes

    def test_row_outcome_expanded_exists(self) -> None:
        """RowOutcome.EXPANDED is available for deaggregation."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.EXPANDED.value == "expanded"

    def test_row_outcome_buffered_exists(self) -> None:
        """RowOutcome.BUFFERED is available for passthrough batching."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.BUFFERED.value == "buffered"

    # NOTE: Individual is_terminal tests removed - test_terminal_mappings (line ~66)
    # comprehensively tests ALL terminal/non-terminal outcomes in one assertion

    def test_all_outcomes_have_is_terminal(self) -> None:
        """All RowOutcome values have is_terminal property."""
        from elspeth.contracts.enums import RowOutcome

        for outcome in RowOutcome:
            # Should not raise - property exists for all values
            _ = outcome.is_terminal


class TestRoutingMode:
    """Tests for RoutingMode enum."""

    def test_routing_mode_values(self) -> None:
        """RoutingMode has move and copy."""
        from elspeth.contracts import RoutingMode

        assert RoutingMode.MOVE.value == "move"
        assert RoutingMode.COPY.value == "copy"


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
    """Tests for TriggerType enum."""

    def test_trigger_type_exists(self) -> None:
        """TriggerType can be imported."""
        from elspeth.contracts.enums import TriggerType

        assert TriggerType is not None

    def test_trigger_type_values(self) -> None:
        """TriggerType has all required values."""
        from elspeth.contracts.enums import TriggerType

        assert TriggerType.COUNT.value == "count"
        assert TriggerType.TIMEOUT.value == "timeout"
        assert TriggerType.CONDITION.value == "condition"
        assert TriggerType.END_OF_SOURCE.value == "end_of_source"
        assert TriggerType.MANUAL.value == "manual"

    def test_trigger_type_is_str_enum(self) -> None:
        """TriggerType can be used as string (for database serialization)."""
        from elspeth.contracts.enums import TriggerType

        # (str, Enum) allows direct string comparison for database serialization
        assert TriggerType.COUNT == "count"  # type: ignore[comparison-overlap]
        assert TriggerType.TIMEOUT == "timeout"  # type: ignore[unreachable]
        # Can be created from string values (for DB reads)
        assert TriggerType("count") == TriggerType.COUNT
        assert TriggerType("end_of_source") == TriggerType.END_OF_SOURCE
