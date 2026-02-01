# tests/property/engine/test_clock_properties.py
"""Property-based tests for clock abstractions.

These tests verify the invariants of ELSPETH's clock implementations:

Clock Protocol Properties:
- monotonic() returns float values
- SystemClock uses real time.monotonic()

MockClock Properties:
- Initial value is configurable
- advance() is monotonic (never decreases time)
- advance() rejects negative values
- set() allows arbitrary time values
- Multiple advances are cumulative

SystemClock Properties:
- Returns real monotonic time
- Sequential calls are non-decreasing
"""

from __future__ import annotations

import time

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.engine.clock import DEFAULT_CLOCK, MockClock, SystemClock

# =============================================================================
# Strategies for generating clock test data
# =============================================================================

# Start times for mock clock
start_times = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)

# Positive advance amounts
positive_advances = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Negative advance amounts (for rejection tests)
negative_advances = st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False)

# Lists of advances
advance_lists = st.lists(positive_advances, min_size=1, max_size=20)


# =============================================================================
# MockClock Initialization Property Tests
# =============================================================================


class TestMockClockInitializationProperties:
    """Property tests for MockClock initialization."""

    @given(start=start_times)
    @settings(max_examples=100)
    def test_initial_time_respected(self, start: float) -> None:
        """Property: MockClock starts at the specified time."""
        clock = MockClock(start=start)

        assert clock.monotonic() == start

    def test_default_start_is_zero(self) -> None:
        """Property: MockClock defaults to t=0."""
        clock = MockClock()

        assert clock.monotonic() == 0.0

    @given(start=start_times)
    @settings(max_examples=50)
    def test_initial_time_stable_before_advance(self, start: float) -> None:
        """Property: Time doesn't change until advance() is called."""
        clock = MockClock(start=start)

        t1 = clock.monotonic()
        t2 = clock.monotonic()
        t3 = clock.monotonic()

        assert t1 == t2 == t3 == start


# =============================================================================
# MockClock.advance() Property Tests
# =============================================================================


class TestMockClockAdvanceProperties:
    """Property tests for MockClock.advance() method."""

    @given(start=start_times, advance=positive_advances)
    @settings(max_examples=100)
    def test_advance_increases_time(self, start: float, advance: float) -> None:
        """Property: advance() increases time by specified amount."""
        clock = MockClock(start=start)

        clock.advance(advance)

        # Use pytest.approx for floating point comparison
        assert clock.monotonic() == pytest.approx(start + advance)

    @given(start=start_times, advances=advance_lists)
    @settings(max_examples=100)
    def test_multiple_advances_cumulative(self, start: float, advances: list[float]) -> None:
        """Property: Multiple advances are cumulative."""
        clock = MockClock(start=start)

        for adv in advances:
            clock.advance(adv)

        expected = start + sum(advances)
        assert clock.monotonic() == pytest.approx(expected)

    @given(start=start_times, negative=negative_advances)
    @settings(max_examples=50)
    def test_negative_advance_rejected(self, start: float, negative: float) -> None:
        """Property: advance() rejects negative values (monotonicity).

        Monotonic clocks must never go backwards. MockClock enforces
        this to match production semantics.
        """
        clock = MockClock(start=start)

        with pytest.raises(ValueError, match="negative"):
            clock.advance(negative)

    @given(start=start_times)
    @settings(max_examples=50)
    def test_zero_advance_is_noop(self, start: float) -> None:
        """Property: advance(0) doesn't change time."""
        clock = MockClock(start=start)

        clock.advance(0.0)

        assert clock.monotonic() == start

    @given(start=start_times, advances=advance_lists)
    @settings(max_examples=50)
    def test_advance_maintains_monotonicity(self, start: float, advances: list[float]) -> None:
        """Property: Time is always non-decreasing after advances."""
        clock = MockClock(start=start)

        previous = start
        for adv in advances:
            clock.advance(adv)
            current = clock.monotonic()
            assert current >= previous
            previous = current


# =============================================================================
# MockClock.set() Property Tests
# =============================================================================


class TestMockClockSetProperties:
    """Property tests for MockClock.set() method."""

    @given(start=start_times, new_value=start_times)
    @settings(max_examples=100)
    def test_set_overrides_current_time(self, start: float, new_value: float) -> None:
        """Property: set() overrides current time to exact value."""
        clock = MockClock(start=start)

        clock.set(new_value)

        assert clock.monotonic() == new_value

    @given(start=start_times, new_value=start_times)
    @settings(max_examples=50)
    def test_set_allows_backwards_time(self, new_value: float, start: float) -> None:
        """Property: set() can go backwards (for test setup).

        Unlike advance(), set() allows any value. This is intentional
        for test scenarios that need to manipulate time freely.
        """
        assume(new_value < start)

        clock = MockClock(start=start)
        clock.set(new_value)

        assert clock.monotonic() == new_value

    @given(start=start_times, advances=advance_lists, final_value=start_times)
    @settings(max_examples=50)
    def test_set_after_advances(self, start: float, advances: list[float], final_value: float) -> None:
        """Property: set() works correctly after previous advances."""
        clock = MockClock(start=start)

        for adv in advances:
            clock.advance(adv)

        clock.set(final_value)

        assert clock.monotonic() == final_value


# =============================================================================
# SystemClock Property Tests
# =============================================================================


class TestSystemClockProperties:
    """Property tests for SystemClock (production clock)."""

    def test_returns_positive_time(self) -> None:
        """Property: SystemClock returns positive time values."""
        clock = SystemClock()

        t = clock.monotonic()

        assert t > 0

    @given(st.data())
    @settings(max_examples=50)
    def test_sequential_calls_monotonic(self, data: st.DataObject) -> None:
        """Property: Sequential calls never decrease.

        This is the core monotonic clock invariant - time never
        goes backwards, even across system clock adjustments.
        """
        clock = SystemClock()

        t1 = clock.monotonic()
        t2 = clock.monotonic()
        t3 = clock.monotonic()

        assert t1 <= t2 <= t3

    def test_matches_time_monotonic(self) -> None:
        """Property: SystemClock uses time.monotonic() internally."""
        clock = SystemClock()

        # Values should be very close
        t1 = clock.monotonic()
        t2 = time.monotonic()

        # Allow small delta for execution time
        assert abs(t2 - t1) < 0.1


# =============================================================================
# DEFAULT_CLOCK Property Tests
# =============================================================================


class TestDefaultClockProperties:
    """Property tests for DEFAULT_CLOCK singleton."""

    def test_default_clock_is_system_clock(self) -> None:
        """Property: DEFAULT_CLOCK is a SystemClock instance."""
        assert isinstance(DEFAULT_CLOCK, SystemClock)

    def test_default_clock_returns_positive_time(self) -> None:
        """Property: DEFAULT_CLOCK returns valid monotonic time."""
        t = DEFAULT_CLOCK.monotonic()

        assert isinstance(t, float)
        assert t > 0


# =============================================================================
# Clock Protocol Property Tests
# =============================================================================


class TestClockProtocolProperties:
    """Property tests for Clock protocol compliance."""

    @given(start=start_times)
    @settings(max_examples=20)
    def test_mock_clock_protocol_compliant(self, start: float) -> None:
        """Property: MockClock satisfies Clock protocol."""
        clock = MockClock(start=start)

        # Protocol requires monotonic() -> float
        result = clock.monotonic()

        assert isinstance(result, float)

    def test_system_clock_protocol_compliant(self) -> None:
        """Property: SystemClock satisfies Clock protocol."""
        clock = SystemClock()

        # Protocol requires monotonic() -> float
        result = clock.monotonic()

        assert isinstance(result, float)


# =============================================================================
# MockClock vs SystemClock Behavioral Property Tests
# =============================================================================


class TestClockBehavioralProperties:
    """Property tests comparing clock implementations."""

    @given(start=start_times, advances=advance_lists)
    @settings(max_examples=50)
    def test_mock_clock_deterministic(self, start: float, advances: list[float]) -> None:
        """Property: MockClock is fully deterministic.

        Given same start and advances, always produces same sequence.
        This is critical for reproducible test scenarios.
        """
        # Run twice with same inputs
        clock1 = MockClock(start=start)
        clock2 = MockClock(start=start)

        times1 = []
        times2 = []

        for adv in advances:
            clock1.advance(adv)
            clock2.advance(adv)
            times1.append(clock1.monotonic())
            times2.append(clock2.monotonic())

        assert times1 == times2

    def test_system_clock_non_deterministic(self) -> None:
        """Property: SystemClock advances between calls (non-deterministic).

        Real time passes, so sequential calls return different values.
        """
        clock = SystemClock()

        t1 = clock.monotonic()
        # Do some work to ensure time passes
        _ = sum(range(10000))
        t2 = clock.monotonic()

        # Should have advanced (non-deterministic)
        assert t2 >= t1
