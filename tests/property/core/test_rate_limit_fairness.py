"""Property-based tests for rate limiter fairness under deterministic scheduling.

These tests verify the fairness property ELSPETH can actually guarantee:
when competing requestors are scheduled fairly, the limiter does not let stale
window state permanently starve any requestor. A small real-thread stress test
remains at the bottom of the file for scheduler smoke coverage.
"""

from __future__ import annotations

import threading
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.rate_limit import RateLimiter
from elspeth.engine.clock import MockClock

# =============================================================================
# Strategies for fairness testing
# =============================================================================

fairness_names = st.text(
    min_size=1,
    max_size=10,
    alphabet="abcdefghijklmnopqrstuvwxyz",
)
requestor_counts = st.integers(min_value=2, max_value=6)
logical_windows = st.integers(min_value=2, max_value=5)

TEST_WINDOW_SECONDS = 0.5
TEST_WINDOW_MS = int(TEST_WINDOW_SECONDS * 1000)
FULL_WINDOW_ADVANCE = TEST_WINDOW_SECONDS + 0.001
THREAD_POLL_SLEEP_SECONDS = 0.005


def _advance_window(clock: MockClock) -> None:
    clock.advance(FULL_WINDOW_ADVANCE)


class TestRateLimiterFairnessProperties:
    """Property tests for deterministic fairness under contention."""

    @given(name=fairness_names, num_requestors=requestor_counts, windows=logical_windows)
    @settings(max_examples=30)
    def test_no_requestor_starved_under_round_robin_schedule(
        self,
        name: str,
        num_requestors: int,
        windows: int,
    ) -> None:
        """Every requestor makes progress when scheduled once per window."""
        clock = MockClock(start=0.0)
        acquires_per_requestor = dict.fromkeys(range(num_requestors), 0)

        with RateLimiter(
            name=name,
            requests_per_minute=num_requestors,
            window_ms=TEST_WINDOW_MS,
            clock=clock,
        ) as limiter:
            for _ in range(windows):
                for requestor in range(num_requestors):
                    if limiter.try_acquire():
                        acquires_per_requestor[requestor] += 1
                assert limiter.try_acquire() is False
                _advance_window(clock)

        assert set(acquires_per_requestor.values()) == {windows}

    @given(name=fairness_names, num_requestors=requestor_counts, windows=logical_windows)
    @settings(max_examples=30)
    def test_round_robin_schedule_prevents_monopoly(
        self,
        name: str,
        num_requestors: int,
        windows: int,
    ) -> None:
        """No requestor exceeds one token per logical window in fair rotation."""
        clock = MockClock(start=0.0)
        acquires_per_requestor = dict.fromkeys(range(num_requestors), 0)

        with RateLimiter(
            name=name,
            requests_per_minute=num_requestors,
            window_ms=TEST_WINDOW_MS,
            clock=clock,
        ) as limiter:
            for _ in range(windows):
                for requestor in range(num_requestors):
                    assert limiter.try_acquire() is True
                    acquires_per_requestor[requestor] += 1

                for _ in range(num_requestors):
                    assert limiter.try_acquire() is False
                _advance_window(clock)

        total = sum(acquires_per_requestor.values())
        fair_share = 1 / num_requestors
        for count in acquires_per_requestor.values():
            assert count / total == pytest.approx(fair_share)

    @given(name=fairness_names, windows=logical_windows)
    @settings(max_examples=30)
    def test_weighted_acquires_dont_starve_lightweight_under_fair_schedule(self, name: str, windows: int) -> None:
        """A weight=1 requestor progresses alongside a weight=3 requestor."""
        clock = MockClock(start=0.0)
        light_acquires = 0
        heavy_acquires = 0

        with RateLimiter(
            name=name,
            requests_per_minute=4,
            window_ms=TEST_WINDOW_MS,
            clock=clock,
        ) as limiter:
            for _ in range(windows):
                assert limiter.try_acquire(weight=3) is True
                heavy_acquires += 1
                assert limiter.try_acquire(weight=1) is True
                light_acquires += 1
                assert limiter.try_acquire(weight=1) is False
                _advance_window(clock)

        assert light_acquires == windows
        assert heavy_acquires == windows


class TestRateLimiterFairnessStress:
    """Small real-thread smoke test for scheduler interaction."""

    @pytest.mark.slow
    def test_threads_make_progress_under_contention(self) -> None:
        """Bounded stress check: competing threads all acquire at least once."""
        num_threads = 3
        rate = num_threads * 5
        acquires_per_thread: dict[int, int] = {}
        lock = threading.Lock()
        stop_event = threading.Event()

        def worker(thread_id: int, limiter: RateLimiter) -> None:
            count = 0
            while not stop_event.is_set():
                if limiter.try_acquire():
                    count += 1
                time.sleep(THREAD_POLL_SLEEP_SECONDS)
            with lock:
                acquires_per_thread[thread_id] = count

        with RateLimiter(name="fairness_stress", requests_per_minute=rate, window_ms=TEST_WINDOW_MS) as limiter:
            threads = [threading.Thread(target=worker, args=(i, limiter), name=f"fairness-{i}") for i in range(num_threads)]
            for thread in threads:
                thread.start()

            time.sleep(TEST_WINDOW_SECONDS * 2 + 0.1)
            stop_event.set()

            for thread in threads:
                thread.join(timeout=1.0)

        assert set(acquires_per_thread) == set(range(num_threads))
        assert all(count > 0 for count in acquires_per_thread.values())
