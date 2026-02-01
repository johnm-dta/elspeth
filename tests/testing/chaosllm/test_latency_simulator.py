# tests/testing/chaosllm/test_latency_simulator.py
"""Tests for ChaosLLM latency simulator."""

import random
import threading

import pytest

from elspeth.testing.chaosllm.config import LatencyConfig
from elspeth.testing.chaosllm.latency_simulator import LatencySimulator


class FixedRandom(random.Random):
    """A Random instance that returns a fixed value for testing."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._fixed_value = value

    def random(self) -> float:
        return self._fixed_value


class TestLatencySimulatorBasic:
    """Basic tests for LatencySimulator."""

    def test_default_config_produces_delay(self) -> None:
        """Default config produces non-negative delay."""
        config = LatencyConfig()  # base_ms=50, jitter_ms=30
        simulator = LatencySimulator(config)

        delay = simulator.simulate()
        # With base=50, jitter=30, range is [20, 80] ms = [0.02, 0.08] sec
        assert delay >= 0.0
        assert delay <= 0.08  # max is (50+30)/1000

    def test_zero_base_zero_jitter_returns_zero(self) -> None:
        """With zero base and zero jitter, delay is zero."""
        config = LatencyConfig(base_ms=0, jitter_ms=0)
        simulator = LatencySimulator(config)

        for _ in range(10):
            delay = simulator.simulate()
            assert delay == 0.0

    def test_returns_seconds_not_milliseconds(self) -> None:
        """Verify the delay is returned in seconds, not milliseconds."""
        config = LatencyConfig(base_ms=100, jitter_ms=0)
        simulator = LatencySimulator(config)

        delay = simulator.simulate()
        # 100ms = 0.1 seconds
        assert delay == 0.1

    def test_no_negative_delays(self) -> None:
        """Delay is clamped to non-negative even with large negative jitter."""
        config = LatencyConfig(base_ms=10, jitter_ms=50)
        # With base=10 and jitter=50, minimum would be 10-50=-40
        # But we clamp to 0

        # Use fixed random that returns 0.0 (maps to -jitter_ms in uniform)
        class NegativeJitterRandom(random.Random):
            def uniform(self, a: float, b: float) -> float:
                # Always return the minimum (a) which is -jitter_ms
                return a

        simulator = LatencySimulator(config, rng=NegativeJitterRandom())
        delay = simulator.simulate()
        assert delay == 0.0  # Clamped from -40ms to 0


class TestJitterBehavior:
    """Tests for jitter behavior."""

    def test_jitter_adds_variation(self) -> None:
        """Jitter causes variation in delay values."""
        config = LatencyConfig(base_ms=50, jitter_ms=20)
        simulator = LatencySimulator(config)

        delays = [simulator.simulate() for _ in range(100)]
        unique_delays = set(delays)

        # Should have multiple different delay values
        assert len(unique_delays) > 1

    def test_jitter_range_correct(self) -> None:
        """Jitter is uniformly distributed in [-jitter_ms, +jitter_ms]."""
        config = LatencyConfig(base_ms=100, jitter_ms=50)
        simulator = LatencySimulator(config)

        delays = [simulator.simulate() for _ in range(1000)]

        # Expected range: [50, 150] ms = [0.05, 0.15] sec
        min_delay = min(delays)
        max_delay = max(delays)

        # Should see values near both ends of the range
        assert min_delay < 0.07  # Should be close to 0.05
        assert max_delay > 0.13  # Should be close to 0.15

        # All values should be within theoretical range
        for delay in delays:
            assert 0.05 <= delay <= 0.15

    def test_zero_jitter_gives_constant_delay(self) -> None:
        """With zero jitter, delay is always exactly base_ms."""
        config = LatencyConfig(base_ms=75, jitter_ms=0)
        simulator = LatencySimulator(config)

        for _ in range(50):
            delay = simulator.simulate()
            assert delay == 0.075  # 75ms in seconds


class TestDeterministicBehavior:
    """Tests for deterministic behavior with seeded random."""

    def test_seeded_random_deterministic(self) -> None:
        """Same seed produces same sequence of delays."""
        config = LatencyConfig(base_ms=50, jitter_ms=30)

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        sim1 = LatencySimulator(config, rng=rng1)
        sim2 = LatencySimulator(config, rng=rng2)

        for _ in range(20):
            d1 = sim1.simulate()
            d2 = sim2.simulate()
            assert d1 == d2

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different sequences."""
        config = LatencyConfig(base_ms=50, jitter_ms=30)

        rng1 = random.Random(42)
        rng2 = random.Random(123)

        sim1 = LatencySimulator(config, rng=rng1)
        sim2 = LatencySimulator(config, rng=rng2)

        delays1 = [sim1.simulate() for _ in range(10)]
        delays2 = [sim2.simulate() for _ in range(10)]

        # Highly unlikely to be identical with different seeds
        assert delays1 != delays2

    def test_fixed_random_gives_predictable_delay(self) -> None:
        """FixedRandom allows exact prediction of delay."""
        config = LatencyConfig(base_ms=100, jitter_ms=20)

        # Create a random that returns specific values for uniform()
        class PredictableRandom(random.Random):
            def uniform(self, a: float, b: float) -> float:
                # Return midpoint (0 jitter)
                return (a + b) / 2

        simulator = LatencySimulator(config, rng=PredictableRandom())
        delay = simulator.simulate()
        # jitter = ((-20) + 20) / 2 = 0
        # delay = (100 + 0) / 1000 = 0.1
        assert delay == 0.1


class TestSlowResponseSimulation:
    """Tests for slow response delay simulation."""

    def test_slow_response_within_range(self) -> None:
        """Slow response delay is within specified range."""
        config = LatencyConfig()
        simulator = LatencySimulator(config)

        for _ in range(50):
            delay = simulator.simulate_slow_response(min_sec=10, max_sec=30)
            assert 10.0 <= delay <= 30.0

    def test_slow_response_variation(self) -> None:
        """Slow response delays vary across calls."""
        config = LatencyConfig()
        simulator = LatencySimulator(config)

        delays = [simulator.simulate_slow_response(5, 15) for _ in range(50)]
        unique_delays = set(delays)

        # Should have variation
        assert len(unique_delays) > 1

    def test_slow_response_deterministic_with_seed(self) -> None:
        """Slow response is deterministic with seeded random."""
        config = LatencyConfig()

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        sim1 = LatencySimulator(config, rng=rng1)
        sim2 = LatencySimulator(config, rng=rng2)

        for _ in range(10):
            d1 = sim1.simulate_slow_response(10, 60)
            d2 = sim2.simulate_slow_response(10, 60)
            assert d1 == d2

    def test_slow_response_exact_range_endpoints(self) -> None:
        """Slow response can hit range endpoints."""
        config = LatencyConfig()
        simulator = LatencySimulator(config)

        delays = [simulator.simulate_slow_response(5, 10) for _ in range(1000)]

        min_delay = min(delays)
        max_delay = max(delays)

        # Should see values close to both endpoints
        assert min_delay < 5.5
        assert max_delay > 9.5


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_simulate_calls(self) -> None:
        """Multiple threads can call simulate() safely."""
        config = LatencyConfig(base_ms=50, jitter_ms=30)
        simulator = LatencySimulator(config)

        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(100):
                delay = simulator.simulate()
                with lock:
                    results.append(delay)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads completed successfully
        assert len(results) == 1000

        # All delays should be valid (non-negative)
        for delay in results:
            assert delay >= 0.0

    def test_concurrent_slow_response_calls(self) -> None:
        """Multiple threads can call simulate_slow_response() safely."""
        config = LatencyConfig()
        simulator = LatencySimulator(config)

        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(100):
                delay = simulator.simulate_slow_response(5, 15)
                with lock:
                    results.append(delay)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1000

        # All delays should be within range
        for delay in results:
            assert 5.0 <= delay <= 15.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_large_base_ms(self) -> None:
        """Handles large base_ms values correctly."""
        config = LatencyConfig(base_ms=5000, jitter_ms=0)  # 5 seconds
        simulator = LatencySimulator(config)

        delay = simulator.simulate()
        assert delay == 5.0  # 5000ms = 5 seconds

    def test_equal_min_max_slow_response(self) -> None:
        """When min_sec equals max_sec, returns that exact value."""
        config = LatencyConfig()
        simulator = LatencySimulator(config)

        for _ in range(10):
            delay = simulator.simulate_slow_response(10, 10)
            assert delay == 10.0

    def test_small_jitter_precision(self) -> None:
        """Small jitter values don't cause precision issues."""
        config = LatencyConfig(base_ms=1, jitter_ms=1)
        simulator = LatencySimulator(config)

        for _ in range(100):
            delay = simulator.simulate()
            # Range: [0, 2] ms = [0.0, 0.002] sec
            assert delay >= 0.0
            assert delay <= 0.002


class TestConfigIntegration:
    """Tests for integration with LatencyConfig."""

    def test_reads_config_values(self) -> None:
        """Correctly reads base_ms and jitter_ms from config."""
        config = LatencyConfig(base_ms=200, jitter_ms=50)

        # Use a fixed random to verify calculation
        class ZeroJitterRandom(random.Random):
            def uniform(self, a: float, b: float) -> float:
                return 0.0  # Zero jitter

        simulator = LatencySimulator(config, rng=ZeroJitterRandom())
        delay = simulator.simulate()

        # base=200, jitter=0, delay = 200/1000 = 0.2
        assert delay == 0.2

    def test_frozen_config_works(self) -> None:
        """Works correctly with frozen (immutable) LatencyConfig."""
        from pydantic import ValidationError

        config = LatencyConfig(base_ms=100, jitter_ms=10)

        # Verify config is frozen
        with pytest.raises(ValidationError):
            config.base_ms = 200  # type: ignore[misc]

        # Simulator still works
        simulator = LatencySimulator(config)
        delay = simulator.simulate()
        assert delay >= 0.0
