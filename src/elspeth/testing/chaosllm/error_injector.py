# src/elspeth/testing/chaosllm/error_injector.py
"""Error injection logic and burst state machine for ChaosLLM.

The ErrorInjector decides per-request whether to inject an error based on
configured percentages. It supports HTTP-level errors, connection-level
failures, and malformed responses. A burst state machine elevates error
rates periodically to simulate real-world LLM provider stress.
"""

import random as random_module
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from elspeth.testing.chaosllm.config import ErrorInjectionConfig


class ErrorCategory(Enum):
    """Categories of errors the injector can produce."""

    HTTP = "http"  # Returns an HTTP error status code
    CONNECTION = "connection"  # Connection-level failures (timeout, reset, slow)
    MALFORMED = "malformed"  # Returns 200 but with bad content


@dataclass(frozen=True, slots=True)
class ErrorDecision:
    """Result of an error injection decision.

    Attributes:
        error_type: The type of error to inject, or None for a successful response
        status_code: HTTP status code (HTTP errors or timeout responses)
        retry_after_sec: Value for Retry-After header (429/529 only)
        delay_sec: Delay before responding or disconnecting (timeout/slow_response/stall)
        start_delay_sec: Lead time before a connection failure or stall
        category: The category of error (HTTP, CONNECTION, or MALFORMED)
        malformed_type: Specific malformation for MALFORMED category
    """

    error_type: str | None
    status_code: int | None = None
    retry_after_sec: int | None = None
    delay_sec: float | None = None
    start_delay_sec: float | None = None
    category: ErrorCategory | None = None
    malformed_type: str | None = None

    @classmethod
    def success(cls) -> "ErrorDecision":
        """Create a decision for a successful (no error) response."""
        return cls(error_type=None)

    @classmethod
    def http_error(
        cls,
        error_type: str,
        status_code: int,
        retry_after_sec: int | None = None,
    ) -> "ErrorDecision":
        """Create a decision for an HTTP-level error."""
        return cls(
            error_type=error_type,
            status_code=status_code,
            retry_after_sec=retry_after_sec,
            category=ErrorCategory.HTTP,
        )

    @classmethod
    def connection_error(
        cls,
        error_type: str,
        delay_sec: float | None = None,
        start_delay_sec: float | None = None,
        status_code: int | None = None,
    ) -> "ErrorDecision":
        """Create a decision for a connection-level failure."""
        return cls(
            error_type=error_type,
            category=ErrorCategory.CONNECTION,
            delay_sec=delay_sec,
            start_delay_sec=start_delay_sec,
            status_code=status_code,
        )

    @classmethod
    def malformed_response(cls, malformed_type: str) -> "ErrorDecision":
        """Create a decision for a malformed response (200 with bad content)."""
        return cls(
            error_type="malformed",
            status_code=200,
            category=ErrorCategory.MALFORMED,
            malformed_type=malformed_type,
        )

    @property
    def should_inject(self) -> bool:
        """Return True if an error should be injected."""
        return self.error_type is not None

    @property
    def is_connection_level(self) -> bool:
        """Return True if error is a connection-level error (spec compatibility)."""
        return self.category == ErrorCategory.CONNECTION

    @property
    def is_malformed(self) -> bool:
        """Return True if error is a malformed response (spec compatibility)."""
        return self.category == ErrorCategory.MALFORMED


# HTTP error types with their status codes.
# Exported for external consumers (e.g., response generators, test assertions).
HTTP_ERRORS: dict[str, int] = {
    "rate_limit": 429,
    "capacity_529": 529,
    "service_unavailable": 503,
    "bad_gateway": 502,
    "gateway_timeout": 504,
    "internal_error": 500,
    "forbidden": 403,
    "not_found": 404,
}

# Connection-level error types.
# Exported for external consumers (e.g., response generators, test assertions).
CONNECTION_ERRORS: set[str] = {
    "timeout",
    "connection_failed",
    "connection_stall",
    "connection_reset",
    "slow_response",
}

# Malformed response types.
# Exported for external consumers (e.g., response generators, test assertions).
MALFORMED_TYPES: set[str] = {
    "invalid_json",
    "truncated",
    "empty_body",
    "missing_fields",
    "wrong_content_type",
}


class ErrorInjector:
    """Decides per-request whether to inject an error.

    Thread-safe implementation with burst state machine that elevates
    error rates periodically to simulate LLM provider stress.

    Usage:
        config = ErrorInjectionConfig(rate_limit_pct=5.0)
        injector = ErrorInjector(config)
        decision = injector.decide()
        if decision.should_inject:
            # Handle the error injection
            ...
    """

    def __init__(
        self,
        config: ErrorInjectionConfig,
        *,
        time_func: Callable[[], float] | None = None,
        rng: random_module.Random | None = None,
    ) -> None:
        """Initialize the error injector.

        Args:
            config: Error injection configuration
            time_func: Time function for testing (default: time.monotonic)
            rng: Random instance for testing (default: module-level random).
                 Inject a seeded random.Random() for deterministic testing.
        """
        self._config = config
        self._time_func = time_func if time_func is not None else time.monotonic
        self._rng = rng if rng is not None else random_module.Random()

        # Burst state machine
        self._lock = threading.Lock()
        self._start_time: float | None = None

    def _get_current_time(self) -> float:
        """Get current time, initializing start time if needed."""
        with self._lock:
            current = self._time_func()
            if self._start_time is None:
                self._start_time = current
            return current - self._start_time

    def _is_in_burst(self, elapsed: float) -> bool:
        """Determine if we're currently in a burst period.

        Bursts occur periodically:
        - Every burst.interval_sec seconds, a burst starts
        - Each burst lasts for burst.duration_sec seconds
        """
        if not self._config.burst.enabled:
            return False

        interval = self._config.burst.interval_sec
        duration = self._config.burst.duration_sec

        # Calculate position within the current interval
        position_in_interval = elapsed % interval

        # We're in burst if we're within the first `duration` seconds of each interval
        return position_in_interval < duration

    def _get_burst_rate_limit_pct(self, elapsed: float) -> float:
        """Get rate limit percentage, using burst rate if in burst mode."""
        if self._is_in_burst(elapsed):
            return self._config.burst.rate_limit_pct
        return self._config.rate_limit_pct

    def _get_burst_capacity_pct(self, elapsed: float) -> float:
        """Get capacity (529) percentage, using burst rate if in burst mode."""
        if self._is_in_burst(elapsed):
            return self._config.burst.capacity_pct
        return self._config.capacity_529_pct

    def _pick_retry_after(self) -> int:
        """Pick a random Retry-After value from the configured range."""
        min_sec, max_sec = self._config.retry_after_sec
        return self._rng.randint(min_sec, max_sec)

    def _pick_timeout_delay(self) -> float:
        """Pick a random timeout delay from the configured range."""
        min_sec, max_sec = self._config.timeout_sec
        return self._rng.uniform(min_sec, max_sec)

    def _build_timeout_decision(self) -> ErrorDecision:
        """Build a timeout decision with a mix of disconnects and 504 responses."""
        delay = self._pick_timeout_delay()
        # 50/50 mix: some timeouts respond with 504, others drop the connection.
        return_504 = self._should_trigger(50.0)
        status_code = 504 if return_504 else None
        return ErrorDecision.connection_error("timeout", delay_sec=delay, status_code=status_code)

    def _pick_connection_failed_lead(self) -> float:
        """Pick a lead time before a connection failure."""
        min_sec, max_sec = self._config.connection_failed_lead_sec
        return self._rng.uniform(min_sec, max_sec)

    def _pick_connection_stall_start(self) -> float:
        """Pick a start delay before stalling the connection."""
        min_sec, max_sec = self._config.connection_stall_start_sec
        return self._rng.uniform(min_sec, max_sec)

    def _pick_connection_stall_delay(self) -> float:
        """Pick a stall duration before disconnect."""
        min_sec, max_sec = self._config.connection_stall_sec
        return self._rng.uniform(min_sec, max_sec)

    def _pick_slow_response_delay(self) -> float:
        """Pick a random slow response delay from the configured range."""
        min_sec, max_sec = self._config.slow_response_sec
        return self._rng.uniform(min_sec, max_sec)

    def _should_trigger(self, percentage: float) -> bool:
        """Determine if an error should trigger based on percentage.

        Args:
            percentage: Error percentage (0-100)

        Returns:
            True if the error should trigger
        """
        if percentage <= 0:
            return False
        return self._rng.random() * 100 < percentage

    def decide(self) -> ErrorDecision:
        """Decide whether to inject an error for this request.

        Errors are evaluated based on selection_mode:
        - priority: first matching error wins (current default)
        - weighted: errors are chosen by configured weights

        Priority order (when selection_mode=priority):
        1. Connection-level errors (connection_failed, connection_stall, timeout, connection_reset, slow_response)
        2. HTTP errors (rate_limit, capacity_529, service_unavailable, etc.)
        3. Malformed responses

        Returns:
            ErrorDecision indicating what error (if any) to inject
        """
        elapsed = self._get_current_time()
        if self._config.selection_mode == "weighted":
            return self._decide_weighted(elapsed)
        return self._decide_priority(elapsed)

    def _decide_priority(self, elapsed: float) -> ErrorDecision:
        """Priority-based decision (first matching error wins)."""

        # === Connection-level errors (highest priority) ===

        # Connection failed: accept then disconnect after a lead time
        if self._should_trigger(self._config.connection_failed_pct):
            return ErrorDecision.connection_error(
                "connection_failed",
                start_delay_sec=self._pick_connection_failed_lead(),
            )

        # Connection stall: accept, delay, stall, then disconnect
        if self._should_trigger(self._config.connection_stall_pct):
            return ErrorDecision.connection_error(
                "connection_stall",
                delay_sec=self._pick_connection_stall_delay(),
                start_delay_sec=self._pick_connection_stall_start(),
            )

        # Timeout: Sometimes respond with 504, sometimes drop the connection
        if self._should_trigger(self._config.timeout_pct):
            return self._build_timeout_decision()

        # Connection reset: RST the TCP connection
        if self._should_trigger(self._config.connection_reset_pct):
            return ErrorDecision.connection_error("connection_reset")

        # Slow response: Respond but with artificial delay
        if self._should_trigger(self._config.slow_response_pct):
            return ErrorDecision.connection_error(
                "slow_response",
                delay_sec=self._pick_slow_response_delay(),
            )

        # === HTTP-level errors ===

        # Rate limit (429) - uses burst rate if in burst
        if self._should_trigger(self._get_burst_rate_limit_pct(elapsed)):
            return ErrorDecision.http_error(
                "rate_limit",
                429,
                retry_after_sec=self._pick_retry_after(),
            )

        # Capacity/Model overloaded (529) - uses burst rate if in burst
        if self._should_trigger(self._get_burst_capacity_pct(elapsed)):
            return ErrorDecision.http_error(
                "capacity_529",
                529,
                retry_after_sec=self._pick_retry_after(),
            )

        # Service unavailable (503)
        if self._should_trigger(self._config.service_unavailable_pct):
            return ErrorDecision.http_error("service_unavailable", 503)

        # Bad gateway (502)
        if self._should_trigger(self._config.bad_gateway_pct):
            return ErrorDecision.http_error("bad_gateway", 502)

        # Gateway timeout (504)
        if self._should_trigger(self._config.gateway_timeout_pct):
            return ErrorDecision.http_error("gateway_timeout", 504)

        # Internal error (500)
        if self._should_trigger(self._config.internal_error_pct):
            return ErrorDecision.http_error("internal_error", 500)

        # Forbidden (403)
        if self._should_trigger(self._config.forbidden_pct):
            return ErrorDecision.http_error("forbidden", 403)

        # Not found (404)
        if self._should_trigger(self._config.not_found_pct):
            return ErrorDecision.http_error("not_found", 404)

        # === Malformed responses (return 200 but with bad content) ===

        # Invalid JSON
        if self._should_trigger(self._config.invalid_json_pct):
            return ErrorDecision.malformed_response("invalid_json")

        # Truncated response
        if self._should_trigger(self._config.truncated_pct):
            return ErrorDecision.malformed_response("truncated")

        # Empty body
        if self._should_trigger(self._config.empty_body_pct):
            return ErrorDecision.malformed_response("empty_body")

        # Missing fields
        if self._should_trigger(self._config.missing_fields_pct):
            return ErrorDecision.malformed_response("missing_fields")

        # Wrong content type
        if self._should_trigger(self._config.wrong_content_type_pct):
            return ErrorDecision.malformed_response("wrong_content_type")

        # No error - success!
        return ErrorDecision.success()

    def _decide_weighted(self, elapsed: float) -> ErrorDecision:
        """Weighted mix decision (errors chosen by configured weights)."""
        choices: list[tuple[float, Callable[[], ErrorDecision]]] = []

        def _add(weight: float, builder: Callable[[], ErrorDecision]) -> None:
            if weight > 0:
                choices.append((weight, builder))

        # Connection-level
        _add(
            self._config.connection_failed_pct,
            lambda: ErrorDecision.connection_error(
                "connection_failed",
                start_delay_sec=self._pick_connection_failed_lead(),
            ),
        )
        _add(
            self._config.connection_stall_pct,
            lambda: ErrorDecision.connection_error(
                "connection_stall",
                delay_sec=self._pick_connection_stall_delay(),
                start_delay_sec=self._pick_connection_stall_start(),
            ),
        )
        _add(self._config.timeout_pct, self._build_timeout_decision)
        _add(
            self._config.connection_reset_pct,
            lambda: ErrorDecision.connection_error("connection_reset"),
        )
        _add(
            self._config.slow_response_pct,
            lambda: ErrorDecision.connection_error("slow_response", delay_sec=self._pick_slow_response_delay()),
        )

        # HTTP-level (burst-adjusted for rate limit and capacity)
        _add(
            self._get_burst_rate_limit_pct(elapsed),
            lambda: ErrorDecision.http_error(
                "rate_limit",
                429,
                retry_after_sec=self._pick_retry_after(),
            ),
        )
        _add(
            self._get_burst_capacity_pct(elapsed),
            lambda: ErrorDecision.http_error(
                "capacity_529",
                529,
                retry_after_sec=self._pick_retry_after(),
            ),
        )
        _add(self._config.service_unavailable_pct, lambda: ErrorDecision.http_error("service_unavailable", 503))
        _add(self._config.bad_gateway_pct, lambda: ErrorDecision.http_error("bad_gateway", 502))
        _add(self._config.gateway_timeout_pct, lambda: ErrorDecision.http_error("gateway_timeout", 504))
        _add(self._config.internal_error_pct, lambda: ErrorDecision.http_error("internal_error", 500))
        _add(self._config.forbidden_pct, lambda: ErrorDecision.http_error("forbidden", 403))
        _add(self._config.not_found_pct, lambda: ErrorDecision.http_error("not_found", 404))

        # Malformed responses
        _add(self._config.invalid_json_pct, lambda: ErrorDecision.malformed_response("invalid_json"))
        _add(self._config.truncated_pct, lambda: ErrorDecision.malformed_response("truncated"))
        _add(self._config.empty_body_pct, lambda: ErrorDecision.malformed_response("empty_body"))
        _add(self._config.missing_fields_pct, lambda: ErrorDecision.malformed_response("missing_fields"))
        _add(self._config.wrong_content_type_pct, lambda: ErrorDecision.malformed_response("wrong_content_type"))

        total_weight = sum(weight for weight, _ in choices)
        if total_weight <= 0:
            return ErrorDecision.success()

        success_weight = max(0.0, 100.0 - total_weight)
        roll = self._rng.random() * (total_weight + success_weight)
        if roll >= total_weight:
            return ErrorDecision.success()

        threshold = 0.0
        for weight, builder in choices:
            threshold += weight
            if roll < threshold:
                return builder()

        return ErrorDecision.success()

    def reset(self) -> None:
        """Reset the injector state (clears burst timing)."""
        with self._lock:
            self._start_time = None

    def is_in_burst(self) -> bool:
        """Check if currently in burst mode (for observability)."""
        elapsed = self._get_current_time()
        return self._is_in_burst(elapsed)
