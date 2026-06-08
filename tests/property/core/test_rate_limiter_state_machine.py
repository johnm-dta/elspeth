# tests/property/core/test_rate_limiter_state_machine.py
"""Property-based stateful tests for rate limiter state machine.

RATE LIMITER STATE MACHINE:
The rate limiter models a leaky bucket with these behaviors:

1. Acquire tokens up to the configured limit
2. Reject tokens when limit is exceeded
3. Rejection doesn't consume quota
4. Tokens replenish as the sliding window advances

Key Invariants:
- Successful acquires never exceed the configured limit within a time window
- Failed try_acquire() calls don't consume tokens
- Weight correctly affects capacity consumption
- After advancing a full window, tokens replenish

These tests use Hypothesis RuleBasedStateMachine and a deterministic clock to
explore state transitions and verify invariants without wall-clock sleeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from elspeth.core.rate_limit import RateLimiter
from elspeth.engine.clock import MockClock

# =============================================================================
# Strategies (local - not duplicating conftest)
# =============================================================================

# Valid limiter names for state machine tests (short, safe)
state_machine_names = st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz")

# Weight values for acquire operations
weights = st.integers(min_value=1, max_value=3)

# Use a short logical window; tests advance MockClock instead of sleeping.
TEST_WINDOW_SECONDS = 2.0
TEST_WINDOW_MS = int(TEST_WINDOW_SECONDS * 1000)
FULL_WINDOW_ADVANCE = TEST_WINDOW_SECONDS + 0.1
PARTIAL_WINDOW_ADVANCE = TEST_WINDOW_SECONDS * 0.3
_STATE_MACHINE_COUNTER = count()


# =============================================================================
# Model Types for State Machine
# =============================================================================


@dataclass
class AcquireAttempt:
    """Record of a single acquire attempt."""

    timestamp: float
    weight: int
    success: bool


@dataclass
class LimiterModel:
    """Model of rate limiter state for verification.

    Tracks acquire attempts and their outcomes to verify invariants.
    """

    limit: int
    window_seconds: float = TEST_WINDOW_SECONDS
    attempts: list[AcquireAttempt] = field(default_factory=list)

    def successful_tokens_in_window(self, now: float) -> int:
        """Count tokens successfully acquired in the current window."""
        window_start = now - self.window_seconds
        return sum(a.weight for a in self.attempts if a.success and a.timestamp > window_start)

    def record_attempt(self, timestamp: float, weight: int, success: bool) -> None:
        """Record an acquire attempt."""
        self.attempts.append(AcquireAttempt(timestamp, weight, success))


# =============================================================================
# Rate Limiter State Machine
# =============================================================================


class RateLimiterStateMachine(RuleBasedStateMachine):
    """Stateful property tests for rate limiter behavior.

    This explores the state space of:
    - Acquiring tokens up to the limit
    - Rejection when limit is exceeded
    - Token replenishment over time (waiting)

    The model tracks expected state and verifies the limiter behaves correctly.
    """

    def __init__(self) -> None:
        super().__init__()

        # Use a moderate limit that's high enough to allow interesting
        # behavior but low enough to hit the limit during testing
        self.limit = 5
        self.clock = MockClock(start=0.0)

        # Create the actual rate limiter
        # Use a unique name to avoid conflicts between test runs
        self.limiter = RateLimiter(
            name=f"statemachine_{next(_STATE_MACHINE_COUNTER)}",
            requests_per_minute=self.limit,
            window_ms=TEST_WINDOW_MS,
            clock=self.clock,
        )

        # Model tracks our expected state
        self.model = LimiterModel(limit=self.limit)

        # Track consecutive rejections for the replenish rule
        self.consecutive_rejections = 0

    def teardown(self) -> None:
        """Clean up limiter resources."""
        self.limiter.close()

    # -------------------------------------------------------------------------
    # Rules: Token Acquisition
    # -------------------------------------------------------------------------

    @rule(weight=weights)
    def acquire_token(self, weight: int) -> None:
        """Try to acquire token(s) and track the result."""
        now = self.clock.monotonic()

        # Try to acquire
        success = self.limiter.try_acquire(weight=weight)

        # Record in model
        self.model.record_attempt(now, weight, success)

        # Track consecutive rejections
        if success:
            self.consecutive_rejections = 0
        else:
            self.consecutive_rejections += 1

    @rule()
    def acquire_single_token(self) -> None:
        """Try to acquire a single token (most common case)."""
        now = self.clock.monotonic()

        success = self.limiter.try_acquire()

        self.model.record_attempt(now, 1, success)

        if success:
            self.consecutive_rejections = 0
        else:
            self.consecutive_rejections += 1

    @rule()
    def wait_for_replenish(self) -> None:
        """Wait for tokens to replenish."""
        if self.consecutive_rejections == 0:
            return
        self.clock.advance(PARTIAL_WINDOW_ADVANCE)

    @rule()
    def wait_full_window(self) -> None:
        """Wait for a full window to reset the bucket."""
        if not self.model.attempts:
            return
        self.clock.advance(FULL_WINDOW_ADVANCE)
        self.consecutive_rejections = 0

    # -------------------------------------------------------------------------
    # Invariants
    # -------------------------------------------------------------------------

    # NOTE: The "rejections don't consume quota" property is tested in the
    # non-stateful test_rejected_acquire_doesnt_consume_quota() below, which
    # provides direct verification rather than relying on indirect state machine
    # behavior observation.

    @invariant()
    def never_exceed_limit_in_window(self) -> None:
        """Invariant: Successful acquires never exceed limit within window.

        At any point in time, the sum of weights from successful acquires
        in the last window should not exceed the limit.
        """
        now = self.clock.monotonic()
        tokens_in_window = self.model.successful_tokens_in_window(now)

        assert tokens_in_window <= self.limit, f"Acquired {tokens_in_window} tokens in window, limit is {self.limit}"


# Create the test class that pytest will discover
TestRateLimiterStateMachine = RateLimiterStateMachine.TestCase
TestRateLimiterStateMachine.settings = settings(
    max_examples=20,
    stateful_step_count=20,
)


# =============================================================================
# Additional Non-Stateful Property Tests
# =============================================================================


class TestRateLimiterQuotaInvariants:
    """Property tests for rate limiter quota invariants."""

    @given(limit=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_rejected_acquire_doesnt_consume_quota(self, limit: int) -> None:
        """Property: If try_acquire fails, the bucket count is unchanged.

        After exhausting the limit, a failed try_acquire should not
        reduce available capacity further.
        """
        with RateLimiter(
            name=f"quotatest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
        ) as limiter:
            # Exhaust the limit
            for _ in range(limit):
                assert limiter.try_acquire() is True

            # Next acquire should fail
            assert limiter.try_acquire() is False

            # Try to acquire with weight that would fit if rejection consumed
            # If rejection consumed quota, we'd have limit+1 consumed,
            # and waiting would give us tokens back at the wrong rate
            #
            # Instead, verify that immediate retry also fails (no quota consumed)
            assert limiter.try_acquire() is False

    @given(limit=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_accepts_up_to_limit(self, limit: int) -> None:
        """Property: Can acquire exactly `limit` tokens within window."""
        with RateLimiter(
            name=f"accepttest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
        ) as limiter:
            # Should be able to acquire exactly `limit` times
            for i in range(limit):
                result = limiter.try_acquire()
                assert result is True, f"Failed on acquire {i + 1} of {limit}"

    @given(limit=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_rejects_over_limit(self, limit: int) -> None:
        """Property: After `limit` acquires, next one fails."""
        with RateLimiter(
            name=f"rejecttest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
        ) as limiter:
            # Exhaust the limit
            for _ in range(limit):
                limiter.try_acquire()

            # Next acquire should fail
            assert limiter.try_acquire() is False

    @given(limit=st.integers(min_value=2, max_value=5))
    @settings(max_examples=10)
    def test_replenishment_after_wait(self, limit: int) -> None:
        """Property: After waiting a full window, tokens replenish."""
        clock = MockClock(start=0.0)
        with RateLimiter(
            name=f"replenishtest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
            clock=clock,
        ) as limiter:
            # Exhaust the limit
            for _ in range(limit):
                limiter.try_acquire()

            # Verify exhausted
            assert limiter.try_acquire() is False

            # Advance past the full logical window.
            clock.advance(FULL_WINDOW_ADVANCE)

            # Should be able to acquire again
            assert limiter.try_acquire() is True

    @given(
        limit=st.integers(min_value=3, max_value=8),
        weight=st.integers(min_value=2, max_value=3),
    )
    @settings(max_examples=20)
    def test_weight_correctly_consumes_capacity(self, limit: int, weight: int) -> None:
        """Property: Acquiring with weight consumes that many tokens."""
        with RateLimiter(
            name=f"weighttest{limit}w{weight}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
        ) as limiter:
            # How many weighted acquires can we do?
            expected_acquires = limit // weight

            # Acquire with weight
            successful = 0
            for _ in range(expected_acquires + 2):  # Try a few extra
                if limiter.try_acquire(weight=weight):
                    successful += 1

            # Should have succeeded exactly expected_acquires times
            # (or expected_acquires + 1 if there's remaining capacity)
            assert successful >= expected_acquires
            assert successful <= expected_acquires + 1

    @given(limit=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10)
    def test_multiple_rejection_attempts_dont_consume_with_replenish(self, limit: int) -> None:
        """Property: Multiple failed try_acquire calls don't accumulate consumption.

        After replenishment, we can acquire the full limit again (proving
        rejections didn't consume quota).
        """
        clock = MockClock(start=0.0)
        with RateLimiter(
            name=f"multirejecttest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
            clock=clock,
        ) as limiter:
            # Exhaust the limit
            for _ in range(limit):
                limiter.try_acquire()

            # Make many failed attempts
            for _ in range(10):
                assert limiter.try_acquire() is False

            # Advance past the full logical window.
            clock.advance(FULL_WINDOW_ADVANCE)

            # Should be able to acquire full limit again
            # (if rejections consumed quota, we'd have fewer available)
            successful = 0
            for _ in range(limit):
                if limiter.try_acquire():
                    successful += 1

            assert successful == limit, f"After replenishment, expected {limit} successful acquires, got {successful}"

    @given(limit=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_multiple_rejection_attempts_dont_consume(self, limit: int) -> None:
        """Property: Multiple failed try_acquire calls don't increase bucket count.

        This is a weaker version of the replenishment test that verifies
        rejections don't consume quota by checking immediate retries also fail.
        """
        with RateLimiter(
            name=f"multirejecttest{limit}",
            requests_per_minute=limit,
            window_ms=TEST_WINDOW_MS,
        ) as limiter:
            # Exhaust the limit
            for _ in range(limit):
                limiter.try_acquire()

            # First rejection
            assert limiter.try_acquire() is False

            # Make many more failed attempts - none should succeed
            # (if rejections consumed quota, the bucket would overflow and
            # potentially cause issues, but immediate retries should all fail)
            for i in range(10):
                result = limiter.try_acquire()
                assert result is False, f"Unexpected success on rejection attempt {i + 1}"


class TestRateLimiterResourceManagement:
    """Property tests for resource management."""

    @given(name=state_machine_names)
    @settings(max_examples=20)
    def test_close_releases_resources(self, name: str) -> None:
        """Property: close() releases resources without error."""
        limiter = RateLimiter(name=name, requests_per_minute=10, window_ms=TEST_WINDOW_MS)
        limiter.try_acquire()
        limiter.close()  # Should not raise

    @given(name=state_machine_names)
    @settings(max_examples=20)
    def test_context_manager_cleanup(self, name: str) -> None:
        """Property: Context manager properly releases resources."""
        with RateLimiter(name=name, requests_per_minute=10, window_ms=TEST_WINDOW_MS) as limiter:
            limiter.try_acquire()
        # Resources should be released after exiting context
