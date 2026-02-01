# src/elspeth/testing/chaosllm/latency_simulator.py
"""Latency simulation for ChaosLLM server.

The LatencySimulator adds configurable artificial delays to make the fake
server behave more like a real LLM API with network latency and processing time.
"""

import random as random_module

from elspeth.testing.chaosllm.config import LatencyConfig


class LatencySimulator:
    """Adds artificial delays to simulate real LLM API latency.

    Thread-safe and stateless - just configuration + random number generation.

    Usage:
        config = LatencyConfig(base_ms=50, jitter_ms=30)
        simulator = LatencySimulator(config)
        delay = simulator.simulate()  # Returns seconds for asyncio.sleep()
        await asyncio.sleep(delay)
    """

    def __init__(
        self,
        config: LatencyConfig,
        *,
        rng: random_module.Random | None = None,
    ) -> None:
        """Initialize the latency simulator.

        Args:
            config: Latency simulation configuration
            rng: Random instance for testing (default: creates new Random instance).
                 Inject a seeded random.Random() for deterministic testing.
        """
        self._config = config
        self._rng = rng if rng is not None else random_module.Random()

    def simulate(self) -> float:
        """Calculate a simulated latency delay.

        Returns the delay in seconds (suitable for asyncio.sleep()).
        The delay is: (base_ms + random_jitter) / 1000.0

        Where random_jitter is uniformly distributed in [-jitter_ms, +jitter_ms].
        The result is clamped to a minimum of 0 (no negative delays).

        Returns:
            Delay in seconds (float)
        """
        base_ms = self._config.base_ms
        jitter_ms = self._config.jitter_ms

        # Calculate jitter as +/- jitter_ms
        jitter = self._rng.uniform(-jitter_ms, jitter_ms)

        # Calculate total delay in ms, clamped to non-negative
        delay_ms = max(0.0, base_ms + jitter)

        # Convert to seconds for asyncio.sleep()
        return delay_ms / 1000.0

    def simulate_slow_response(self, min_sec: int, max_sec: int) -> float:
        """Calculate a slow response delay.

        Used for slow_response error injection where the delay is specified
        in seconds as a [min, max] range.

        Args:
            min_sec: Minimum delay in seconds
            max_sec: Maximum delay in seconds

        Returns:
            Delay in seconds (float)
        """
        return self._rng.uniform(min_sec, max_sec)
