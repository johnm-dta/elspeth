"""Tests for contracts/engine.py DTOs."""

from dataclasses import FrozenInstanceError

import pytest

from elspeth.contracts.engine import BufferEntry, PendingOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath


class TestBufferEntry:
    def test_frozen(self) -> None:
        """BufferEntry must be immutable — audit timing metadata cannot change after construction."""
        entry = BufferEntry(
            submit_index=0,
            complete_index=1,
            result="test",
            submit_timestamp=1.0,
            complete_timestamp=2.0,
            buffer_wait_ms=5.0,
        )
        with pytest.raises(FrozenInstanceError):
            entry.submit_index = 99  # type: ignore[misc]

    def test_slots(self) -> None:
        """BufferEntry uses __slots__ for memory efficiency — no instance __dict__."""
        entry = BufferEntry(
            submit_index=0,
            complete_index=0,
            result="test",
            submit_timestamp=1.0,
            complete_timestamp=2.0,
            buffer_wait_ms=0.5,
        )
        assert not hasattr(entry, "__dict__"), "Slots dataclass should not have __dict__"


class TestBufferEntryPostInit:
    """Tests for BufferEntry __post_init__ validation."""

    def test_rejects_negative_submit_index(self) -> None:
        with pytest.raises(ValueError, match="submit_index must be >= 0"):
            BufferEntry(submit_index=-1, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)

    def test_rejects_negative_complete_index(self) -> None:
        with pytest.raises(ValueError, match="complete_index must be >= 0"):
            BufferEntry(submit_index=0, complete_index=-1, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)

    def test_rejects_nan_submit_timestamp(self) -> None:
        with pytest.raises(ValueError, match="submit_timestamp must be non-negative and finite"):
            BufferEntry(
                submit_index=0, complete_index=0, result="x", submit_timestamp=float("nan"), complete_timestamp=0.0, buffer_wait_ms=0.0
            )

    def test_rejects_inf_complete_timestamp(self) -> None:
        with pytest.raises(ValueError, match="complete_timestamp must be non-negative and finite"):
            BufferEntry(
                submit_index=0, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=float("inf"), buffer_wait_ms=0.0
            )

    def test_rejects_negative_buffer_wait_ms(self) -> None:
        with pytest.raises(ValueError, match="buffer_wait_ms must be non-negative and finite"):
            BufferEntry(submit_index=0, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=-1.0)

    def test_rejects_nan_buffer_wait_ms(self) -> None:
        with pytest.raises(ValueError, match="buffer_wait_ms must be non-negative and finite"):
            BufferEntry(
                submit_index=0, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=float("nan")
            )

    def test_accepts_zero_values(self) -> None:
        entry = BufferEntry(submit_index=0, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)
        assert entry.submit_index == 0
        assert entry.buffer_wait_ms == 0.0

    # --- Type guards (elspeth-eadb3d18ba) ---

    def test_rejects_float_submit_index(self) -> None:
        """Regression: float 0.5 passes >= 0 check but corrupts index."""
        with pytest.raises(TypeError, match="submit_index must be int"):
            BufferEntry(submit_index=0.5, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)  # type: ignore[arg-type]

    def test_rejects_bool_submit_index(self) -> None:
        """bool is subclass of int — True (value 1) must not pass as index."""
        with pytest.raises(TypeError, match="submit_index must be int"):
            BufferEntry(submit_index=True, complete_index=0, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)

    def test_rejects_float_complete_index(self) -> None:
        with pytest.raises(TypeError, match="complete_index must be int"):
            BufferEntry(submit_index=0, complete_index=1.5, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)  # type: ignore[arg-type]  # intentionally invalid type

    def test_rejects_bool_complete_index(self) -> None:
        with pytest.raises(TypeError, match="complete_index must be int"):
            BufferEntry(submit_index=0, complete_index=False, result="x", submit_timestamp=0.0, complete_timestamp=0.0, buffer_wait_ms=0.0)


class TestPendingOutcomePostInit:
    """Tests for PendingOutcome __post_init__ validation."""

    def test_quarantined_requires_error_hash(self) -> None:
        with pytest.raises(ValueError, match=r"QUARANTINED_AT_SOURCE.*error_hash"):
            PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.QUARANTINED_AT_SOURCE, error_hash=None)

    def test_failed_requires_error_hash(self) -> None:
        with pytest.raises(ValueError, match=r"UNROUTED.*error_hash"):
            PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED, error_hash=None)

    def test_completed_rejects_error_hash(self) -> None:
        with pytest.raises(ValueError, match=r"DEFAULT_FLOW.*must not have error_hash"):
            PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW, error_hash="abc123")

    def test_quarantined_with_error_hash_accepted(self) -> None:
        po = PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.QUARANTINED_AT_SOURCE, error_hash="abc123")
        assert po.error_hash == "abc123"

    def test_completed_without_error_hash_accepted(self) -> None:
        po = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)
        assert po.error_hash is None

    def test_scheduler_pending_sink_requires_bool(self) -> None:
        with pytest.raises(ValueError, match="scheduler_pending_sink must be a bool"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                scheduler_pending_sink=1,  # type: ignore[arg-type]
            )

    def test_rejects_non_enum_outcome_before_pair_formatting(self) -> None:
        with pytest.raises(ValueError, match="outcome must be TerminalOutcome or None"):
            PendingOutcome(
                outcome="success",  # type: ignore[arg-type]
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_rejects_non_enum_path_before_name_formatting(self) -> None:
        with pytest.raises(ValueError, match="path must be TerminalPath"):
            PendingOutcome(
                outcome=None,
                path="buffered",  # type: ignore[arg-type]
            )

    def test_routed_without_error_hash_accepted(self) -> None:
        po = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.GATE_ROUTED)
        assert po.error_hash is None

    def test_routed_rejects_error_hash(self) -> None:
        """GATE_ROUTED is not in _REQUIRES_ERROR_HASH_PATHS — error_hash forbidden."""
        with pytest.raises(ValueError, match=r"GATE_ROUTED.*must not have error_hash"):
            PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.GATE_ROUTED, error_hash="abc123")

    def test_routed_on_error_requires_error_hash(self) -> None:
        """ON_ERROR_ROUTED joins _REQUIRES_ERROR_HASH_PATHS — error_hash REQUIRED."""
        with pytest.raises(ValueError, match="requires non-empty error_hash"):
            PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.ON_ERROR_ROUTED, error_hash=None)

    def test_routed_on_error_with_error_hash_accepted(self) -> None:
        """ON_ERROR_ROUTED + non-empty error_hash is the contract-conforming shape."""
        po = PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.ON_ERROR_ROUTED, error_hash="abc123")
        assert po.error_hash == "abc123"

    def test_routed_on_error_with_empty_error_hash_rejected(self) -> None:
        """Empty-string error_hash counts as missing per __post_init__ whitespace check."""
        with pytest.raises(ValueError, match="requires non-empty error_hash"):
            PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.ON_ERROR_ROUTED, error_hash="")


class TestPendingOutcomeTwoAxis:
    """ADR-019 Phase 1: PendingOutcome carries (outcome, path) for sink-durable recording."""

    def test_pending_outcome_completed(self) -> None:
        po = PendingOutcome(
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        assert po.outcome == TerminalOutcome.SUCCESS
        assert po.path == TerminalPath.DEFAULT_FLOW
        assert po.error_hash is None

    def test_pending_outcome_routed_on_error_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                error_hash=None,
            )

    def test_pending_outcome_failed_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=None,
            )

    def test_pending_outcome_quarantined_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
                error_hash=None,
            )

    def test_pending_outcome_completed_must_not_have_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                error_hash="abcd1234abcd1234",
            )

    def test_pending_outcome_rejects_illegal_completed_pair(self) -> None:
        with pytest.raises(ValueError, match="legal"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
            )

    def test_pending_outcome_none_requires_buffered_path(self) -> None:
        with pytest.raises(ValueError, match="BUFFERED"):
            PendingOutcome(
                outcome=None,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_pending_outcome_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            PendingOutcome(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)  # type: ignore[misc]
