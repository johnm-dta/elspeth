# src/elspeth/plugins/llm/aimd_throttle.py
"""AIMD (Additive Increase, Multiplicative Decrease) throttle for LLM API calls.

Implements TCP-style congestion control:
- On capacity error: multiply delay (fast ramp down)
- On success: subtract fixed amount (slow ramp up)

This prevents "riding the edge" where you're constantly hitting capacity limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class ThrottleConfig:
    """Configuration for AIMD throttle behavior.

    Note: This is a runtime dataclass, not a Pydantic model, because it's
    internal state configuration built from validated PoolConfig, not
    user-provided YAML config.

    Attributes:
        min_dispatch_delay_ms: Floor for delay between dispatches (default: 0)
        max_dispatch_delay_ms: Ceiling for delay (default: 5000)
        backoff_multiplier: Multiply delay on capacity error (default: 2.0)
        recovery_step_ms: Subtract from delay on success (default: 50)
    """

    min_dispatch_delay_ms: int = 0
    max_dispatch_delay_ms: int = 5000
    backoff_multiplier: float = 2.0
    recovery_step_ms: int = 50


class AIMDThrottle:
    """Thread-safe AIMD throttle state machine.

    Usage:
        throttle = AIMDThrottle()

        # Before dispatching request
        delay = throttle.current_delay_ms
        time.sleep(delay / 1000)

        # After request completes
        if is_capacity_error:
            throttle.on_capacity_error()
        else:
            throttle.on_success()
    """

    def __init__(self, config: ThrottleConfig | None = None) -> None:
        """Initialize throttle with optional config.

        Args:
            config: Throttle configuration (uses defaults if None)
        """
        self._config = config or ThrottleConfig()
        self._current_delay_ms: float = 0.0
        self._lock = Lock()

    @property
    def config(self) -> ThrottleConfig:
        """Get throttle configuration."""
        return self._config

    @property
    def current_delay_ms(self) -> float:
        """Get current delay in milliseconds (thread-safe)."""
        with self._lock:
            return self._current_delay_ms

    def on_capacity_error(self) -> None:
        """Record capacity error - multiply delay (thread-safe).

        If current delay is 0, bootstraps to recovery_step_ms.
        Otherwise multiplies by backoff_multiplier, capped at max.
        """
        with self._lock:
            if self._current_delay_ms == 0:
                # Bootstrap: start with recovery_step as initial backoff
                self._current_delay_ms = float(self._config.recovery_step_ms)
            else:
                # Multiplicative decrease
                self._current_delay_ms *= self._config.backoff_multiplier

            # Cap at maximum
            if self._current_delay_ms > self._config.max_dispatch_delay_ms:
                self._current_delay_ms = float(self._config.max_dispatch_delay_ms)

    def on_success(self) -> None:
        """Record successful request - subtract recovery step (thread-safe).

        Subtracts recovery_step_ms from current delay, floored at min.
        """
        with self._lock:
            self._current_delay_ms -= self._config.recovery_step_ms

            # Floor at minimum
            if self._current_delay_ms < self._config.min_dispatch_delay_ms:
                self._current_delay_ms = float(self._config.min_dispatch_delay_ms)
