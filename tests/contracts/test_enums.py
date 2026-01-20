"""Tests for contracts enums."""

import pytest


class TestDeterminism:
    """Tests for Determinism enum - critical for replay/verify."""

    def test_has_all_required_values(self) -> None:
        """Determinism has all 6 values from architecture."""
        from elspeth.contracts import Determinism

        assert hasattr(Determinism, "DETERMINISTIC")
        assert hasattr(Determinism, "SEEDED")
        assert hasattr(Determinism, "IO_READ")
        assert hasattr(Determinism, "IO_WRITE")
        assert hasattr(Determinism, "EXTERNAL_CALL")
        assert hasattr(Determinism, "NON_DETERMINISTIC")
        # Explicit count verification - architecture specifies exactly 6 values
        assert len(list(Determinism)) == 6

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
        assert RowOutcome.COMPLETED.value == "completed"
        # Can be created from string values (for DB reads)
        assert RowOutcome("completed") == RowOutcome.COMPLETED

    def test_has_all_terminal_states(self) -> None:
        """RowOutcome has all terminal states from architecture."""
        from elspeth.contracts import RowOutcome

        assert hasattr(RowOutcome, "COMPLETED")
        assert hasattr(RowOutcome, "ROUTED")
        assert hasattr(RowOutcome, "FORKED")
        assert hasattr(RowOutcome, "FAILED")
        assert hasattr(RowOutcome, "QUARANTINED")
        assert hasattr(RowOutcome, "CONSUMED_IN_BATCH")
        assert hasattr(RowOutcome, "COALESCED")

    def test_row_outcome_expanded_exists(self) -> None:
        """RowOutcome.EXPANDED is available for deaggregation."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.EXPANDED.value == "expanded"

    def test_row_outcome_buffered_exists(self) -> None:
        """RowOutcome.BUFFERED is available for passthrough batching."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.BUFFERED.value == "buffered"

    def test_row_outcome_buffered_is_not_terminal(self) -> None:
        """BUFFERED is non-terminal - token will reappear with final outcome."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.BUFFERED.is_terminal is False

    def test_row_outcome_consumed_in_batch_is_terminal(self) -> None:
        """CONSUMED_IN_BATCH is terminal - token is absorbed into aggregate."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.CONSUMED_IN_BATCH.is_terminal is True

    def test_row_outcome_expanded_is_terminal(self) -> None:
        """EXPANDED is terminal - parent token's journey ends, children continue."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.EXPANDED.is_terminal is True

    def test_row_outcome_completed_is_terminal(self) -> None:
        """COMPLETED is terminal."""
        from elspeth.contracts.enums import RowOutcome

        assert RowOutcome.COMPLETED.is_terminal is True

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
