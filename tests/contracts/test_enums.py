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
    """Tests for RowOutcome enum - derived, not stored."""

    def test_is_not_str_enum(self) -> None:
        """RowOutcome should NOT be a str subclass - it's derived, not stored."""
        # RowOutcome.COMPLETED should not be equal to string without .value
        # Note: mypy correctly detects this comparison can never be equal since
        # RowOutcome is not a (str, Enum). We cast to Any to verify at runtime.
        from typing import Any, cast

        from elspeth.contracts import RowOutcome

        assert cast(Any, RowOutcome.COMPLETED) != "completed"
        assert RowOutcome.COMPLETED.value == "completed"

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
