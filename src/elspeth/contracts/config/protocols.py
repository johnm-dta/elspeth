"""Runtime protocols for Settings -> Runtime enforcement.

These protocols define what engine components EXPECT from runtime config.
By having runtime config classes implement these protocols, we get:
1. Compile-time verification that Settings fields reach runtime
2. Clear documentation of what each component needs

Protocol Pattern:
    - Protocol defines minimal interface a component needs
    - Runtime config dataclass implements the protocol
    - Component accepts the protocol, not the concrete class
    - mypy verifies structural compatibility

Example:
    class RuntimeRetryConfig:
        max_attempts: int
        base_delay: float
        ...

    def __init__(self, config: RuntimeRetryProtocol):
        # mypy verifies RuntimeRetryConfig satisfies this
        self._config = config

Note on jitter:
    jitter is INTERNAL to RuntimeRetryConfig (hardcoded to 1.0, not in Settings).
    It's included in RuntimeRetryProtocol because RetryManager needs to access it
    for tenacity's wait_exponential_jitter(). However, it's not a Settings field -
    the value is always provided by RuntimeRetryConfig's factory methods.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.contracts.config.runtime import ExporterConfig
    from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity


class RetrySettingsProtocol(Protocol):
    """Settings-side shape needed to build RuntimeRetryConfig.

    Members are read-only ``@property`` getters because the concrete settings
    object (``RetrySettings``) is a frozen Pydantic model: a mutable protocol
    attribute would not be satisfied by a read-only field. A read-only protocol
    member accepts both frozen and mutable implementations.
    """

    @property
    def max_attempts(self) -> int:
        """Maximum number of attempts (includes the initial try)."""
        ...

    @property
    def initial_delay_seconds(self) -> float:
        """Initial backoff delay in seconds."""
        ...

    @property
    def max_delay_seconds(self) -> float:
        """Maximum backoff delay in seconds."""
        ...

    @property
    def exponential_base(self) -> float:
        """Exponential backoff multiplier (e.g. 2.0 for doubling)."""
        ...


class ServiceRateLimitSettingsProtocol(Protocol):
    """Settings-side shape for one service rate-limit override.

    Read-only ``@property`` members (like the other ``*SettingsProtocol``
    classes here) so the frozen Pydantic settings models satisfy them.
    """

    @property
    def requests_per_minute(self) -> int:
        """Service-specific requests-per-minute limit."""
        ...


class RateLimitSettingsProtocol(Protocol):
    """Settings-side shape needed to build RuntimeRateLimitConfig."""

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is active."""
        ...

    @property
    def default_requests_per_minute(self) -> int:
        """Default requests per minute for unconfigured services."""
        ...

    @property
    def persistence_path(self) -> str | None:
        """Optional SQLite path for cross-process persistence."""
        ...

    @property
    def services(self) -> Mapping[str, ServiceRateLimitSettingsProtocol]:
        """Per-service rate-limit overrides keyed by service name."""
        ...


class ConcurrencySettingsProtocol(Protocol):
    """Settings-side shape needed to build RuntimeConcurrencyConfig."""

    @property
    def max_workers(self) -> int:
        """Maximum number of parallel workers."""
        ...


class CheckpointSettingsProtocol(Protocol):
    """Settings-side shape needed to build RuntimeCheckpointConfig."""

    @property
    def enabled(self) -> bool:
        """Whether checkpointing is active."""
        ...

    @property
    def frequency(self) -> str:
        """Checkpoint frequency token (e.g. 'every_row', 'every_n')."""
        ...

    @property
    def checkpoint_interval(self) -> int | None:
        """Row interval for frequency='every_n'; None otherwise."""
        ...


class TelemetryExporterSettingsProtocol(Protocol):
    """Settings-side shape for one telemetry exporter config."""

    @property
    def name(self) -> str:
        """Exporter name (e.g. 'console', 'otlp')."""
        ...

    @property
    def options(self) -> Mapping[str, Any]:
        """Exporter-specific options."""
        ...


class TelemetrySettingsProtocol(Protocol):
    """Settings-side shape needed to build RuntimeTelemetryConfig."""

    @property
    def enabled(self) -> bool:
        """Whether telemetry is active."""
        ...

    @property
    def granularity(self) -> str:
        """Granularity token (parsed to TelemetryGranularity)."""
        ...

    @property
    def backpressure_mode(self) -> str:
        """Backpressure-mode token (parsed to BackpressureMode)."""
        ...

    @property
    def fail_on_total_exporter_failure(self) -> bool:
        """Whether to fail the run if all exporters fail."""
        ...

    @property
    def max_consecutive_failures(self) -> int:
        """Consecutive total failures before disabling or raising."""
        ...

    @property
    def exporters(self) -> Sequence[TelemetryExporterSettingsProtocol]:
        """Configured telemetry exporters."""
        ...


@runtime_checkable
class RuntimeRetryProtocol(Protocol):
    """What RetryManager expects from retry configuration.

    These fields come from RetrySettings (possibly with name mapping):
    - max_attempts: RetrySettings.max_attempts (direct)
    - base_delay: RetrySettings.initial_delay_seconds (renamed)
    - max_delay: RetrySettings.max_delay_seconds (renamed)
    - exponential_base: RetrySettings.exponential_base (direct)
    - jitter: INTERNAL - hardcoded to 1.0 second, not from Settings

    Note: jitter is internal to RuntimeRetryConfig (hardcoded, not in Settings)
    but is included in the protocol because RetryManager needs it for tenacity.
    """

    @property
    def max_attempts(self) -> int:
        """Maximum number of attempts (includes initial try)."""
        ...

    @property
    def base_delay(self) -> float:
        """Initial backoff delay in seconds."""
        ...

    @property
    def max_delay(self) -> float:
        """Maximum backoff delay in seconds."""
        ...

    @property
    def exponential_base(self) -> float:
        """Exponential backoff multiplier (e.g., 2.0 for doubling)."""
        ...

    @property
    def jitter(self) -> float:
        """Jitter to add to backoff delay in seconds (internal, not from Settings)."""
        ...


@runtime_checkable
class ServiceRateLimitProtocol(Protocol):
    """Minimal service-level rate limit interface used by RateLimitRegistry."""

    @property
    def requests_per_minute(self) -> int:
        """Service-specific requests-per-minute limit."""
        ...


@runtime_checkable
class RuntimeRateLimitProtocol(Protocol):
    """What RateLimitRegistry expects from rate limit configuration.

    These fields come from RateLimitSettings:
    - enabled: Whether rate limiting is active
    - default_requests_per_minute: Per-minute rate limit for services
    - persistence_path: Optional SQLite path for cross-process persistence

    Plus one method needed by RateLimitRegistry:
    - get_service_config(service_name): resolve per-service override or default
    """

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is active."""
        ...

    @property
    def default_requests_per_minute(self) -> int:
        """Default requests per minute for unconfigured services."""
        ...

    @property
    def persistence_path(self) -> str | None:
        """Optional SQLite path for cross-process rate limit persistence."""
        ...

    def get_service_config(self, service_name: str) -> ServiceRateLimitProtocol:
        """Get service-specific rate limit config, with fallback to defaults."""
        ...


@runtime_checkable
class RuntimeConcurrencyProtocol(Protocol):
    """What ThreadPoolExecutor/Orchestrator expects from concurrency config.

    Simple - just needs max_workers from ConcurrencySettings.
    """

    @property
    def max_workers(self) -> int:
        """Maximum number of parallel workers."""
        ...


@runtime_checkable
class RuntimeCheckpointProtocol(Protocol):
    """What checkpoint system expects from checkpoint configuration.

    Maps CheckpointSettings fields:
    - enabled: CheckpointSettings.enabled (direct)
    - frequency: CheckpointSettings.frequency mapped to int

    Note: checkpoint_interval is conditional on frequency="every_n" and
    handled during construction, not as a protocol field.
    """

    @property
    def enabled(self) -> bool:
        """Whether checkpointing is active."""
        ...

    @property
    def frequency(self) -> int:
        """Checkpoint every N rows (1 = every row, 0 = aggregation only)."""
        ...


@runtime_checkable
class RuntimeTelemetryProtocol(Protocol):
    """What TelemetryManager expects from telemetry configuration.

    These fields come from TelemetrySettings (direct mapping unless noted):
    - enabled: TelemetrySettings.enabled
    - granularity: TelemetrySettings.granularity (parsed to TelemetryGranularity enum)
    - backpressure_mode: TelemetrySettings.backpressure_mode (parsed to BackpressureMode enum)
    - fail_on_total_exporter_failure: TelemetrySettings.fail_on_total_exporter_failure
    - max_consecutive_failures: TelemetrySettings.max_consecutive_failures (direct)
    - exporter_configs: TelemetrySettings.exporters (converted to tuple of ExporterConfig)

    Note: The from_settings() factory validates that backpressure_mode is
    implemented before returning. Unimplemented modes (like 'slow') cause
    NotImplementedError at config load time, not at runtime.
    """

    @property
    def enabled(self) -> bool:
        """Whether telemetry is active."""
        ...

    @property
    def granularity(self) -> "TelemetryGranularity":
        """Granularity of events to emit (lifecycle, rows, or full)."""
        ...

    @property
    def backpressure_mode(self) -> "BackpressureMode":
        """How to handle backpressure when exporters can't keep up."""
        ...

    @property
    def fail_on_total_exporter_failure(self) -> bool:
        """Whether to fail the run if all exporters fail."""
        ...

    @property
    def max_consecutive_failures(self) -> int:
        """Number of consecutive total failures before disabling or raising."""
        ...

    @property
    def exporter_configs(self) -> "tuple[ExporterConfig, ...]":
        """Immutable sequence of exporter configurations."""
        ...
