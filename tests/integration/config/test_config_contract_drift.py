# tests/integration/config/test_config_contract_drift.py
"""Integration tests for Settings -> RuntimeConfig contract drift detection.

Verifies that every Settings field reaches its RuntimeConfig counterpart
through from_settings(), and that non-default values survive the conversion.

Motivation: The P2-2026-01-21 bug showed that exponential_base existed in
RetrySettings but was never mapped in from_settings(). Users configured it,
Pydantic validated it, but the engine silently ignored it. These tests catch
that class of bug by verifying end-to-end value propagation.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

from pydantic import BaseModel

from elspeth.contracts.config.alignment import FIELD_MAPPINGS, SETTINGS_TO_RUNTIME
from elspeth.contracts.config.defaults import INTERNAL_DEFAULTS
from elspeth.contracts.config.runtime import (
    RuntimeCheckpointConfig,
    RuntimeConcurrencyConfig,
    RuntimeRateLimitConfig,
    RuntimeRetryConfig,
    RuntimeTelemetryConfig,
)
from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    ExporterSettings,
    RateLimitSettings,
    RetrySettings,
    ServiceRateLimit,
    TelemetrySettings,
)
from elspeth.engine.retry import RetryManager


class TestRetryConfigPropagation:
    """Verify RetrySettings values propagate through RuntimeRetryConfig to RetryManager."""

    def test_non_default_values_reach_retry_manager(self) -> None:
        """Non-default RetrySettings values must be accessible on RetryManager's config.

        Invariant: Every user-configurable retry field that passes Pydantic
        validation must affect RetryManager behavior. If a field exists in
        RetrySettings but RetryManager never sees it, that field is orphaned.
        """
        settings = RetrySettings(
            max_attempts=7,
            initial_delay_seconds=0.5,
            max_delay_seconds=30.0,
            exponential_base=5.0,
        )

        runtime_config = RuntimeRetryConfig.from_settings(settings)
        manager = RetryManager(runtime_config)

        assert manager._config.max_attempts == 7
        assert manager._config.base_delay == 0.5
        assert manager._config.max_delay == 30.0
        assert manager._config.exponential_base == 5.0

    def test_initial_delay_rename_preserves_value(self) -> None:
        """initial_delay_seconds (Settings) maps to base_delay (Runtime).

        Invariant: The FIELD_MAPPINGS rename from initial_delay_seconds to
        base_delay must preserve the numeric value exactly. A rename that
        drops the value would silently fall back to defaults.
        """
        settings = RetrySettings(initial_delay_seconds=0.5)
        runtime_config = RuntimeRetryConfig.from_settings(settings)

        assert runtime_config.base_delay == 0.5


class TestRateLimitConfigPropagation:
    """Verify RateLimitSettings values propagate through RuntimeRateLimitConfig."""

    def test_non_default_values_reach_config(self) -> None:
        """Non-default RateLimitSettings values must survive from_settings().

        Invariant: enabled, default_requests_per_minute, persistence_path,
        and services must all be faithfully transferred. Missing any field
        means the engine uses a stale default instead of the user's config.
        """
        service = ServiceRateLimit(requests_per_minute=200)
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=120,
            persistence_path="/tmp/rate_limits.db",
            services={"openai": service},
        )

        runtime_config = RuntimeRateLimitConfig.from_settings(settings)

        assert runtime_config.enabled is True
        assert runtime_config.default_requests_per_minute == 120
        assert runtime_config.persistence_path == "/tmp/rate_limits.db"
        assert "openai" in runtime_config.services
        assert runtime_config.services["openai"].requests_per_minute == 200

    def test_disabled_rate_limit_propagates(self) -> None:
        """enabled=False must reach RuntimeRateLimitConfig unchanged.

        Invariant: A user who disables rate limiting must not have it
        silently re-enabled by a default somewhere in the conversion chain.
        """
        settings = RateLimitSettings(enabled=False)
        runtime_config = RuntimeRateLimitConfig.from_settings(settings)

        assert runtime_config.enabled is False


class TestCheckpointConfigPropagation:
    """Verify CheckpointSettings values propagate through RuntimeCheckpointConfig."""

    def test_non_default_values_reach_config(self) -> None:
        """Non-default CheckpointSettings values must survive from_settings().

        Invariant: enabled, frequency, checkpoint_interval, and
        aggregation_boundaries must all be faithfully transferred. The
        frequency field undergoes a type transformation (str -> int) but
        the semantic meaning must be preserved.
        """
        settings = CheckpointSettings(
            enabled=True,
            frequency="every_n",
            checkpoint_interval=50,
            aggregation_boundaries=False,
        )

        runtime_config = RuntimeCheckpointConfig.from_settings(settings)

        assert runtime_config.enabled is True
        assert runtime_config.frequency == 50
        assert runtime_config.checkpoint_interval == 50
        assert runtime_config.aggregation_boundaries is False

    def test_frequency_value_preserved(self) -> None:
        """The frequency field's semantic value must survive type transformation.

        Invariant: "every_row" -> 1, "aggregation_only" -> 0, "every_n" -> N.
        If the mapping is wrong, checkpoints fire at wrong intervals.
        """
        every_row = CheckpointSettings(frequency="every_row")
        assert RuntimeCheckpointConfig.from_settings(every_row).frequency == 1

        agg_only = CheckpointSettings(frequency="aggregation_only")
        assert RuntimeCheckpointConfig.from_settings(agg_only).frequency == 0

        every_10 = CheckpointSettings(frequency="every_n", checkpoint_interval=10)
        assert RuntimeCheckpointConfig.from_settings(every_10).frequency == 10


class TestConcurrencyConfigPropagation:
    """Verify ConcurrencySettings values propagate through RuntimeConcurrencyConfig."""

    def test_non_default_max_workers_preserved(self) -> None:
        """Non-default max_workers must reach RuntimeConcurrencyConfig.

        Invariant: A user who sets max_workers=8 must get exactly 8 workers,
        not the default of 4. This is a simple direct mapping but the test
        ensures no intermediate layer drops or overrides the value.
        """
        settings = ConcurrencySettings(max_workers=8)
        runtime_config = RuntimeConcurrencyConfig.from_settings(settings)

        assert runtime_config.max_workers == 8


class TestTelemetryConfigPropagation:
    """Verify TelemetrySettings values propagate through RuntimeTelemetryConfig."""

    def test_non_default_values_reach_config(self) -> None:
        """Non-default TelemetrySettings values must survive from_settings().

        Invariant: enabled, granularity, backpressure_mode,
        fail_on_total_exporter_failure, max_consecutive_failures, and
        exporters must all be faithfully transferred. The granularity and
        backpressure_mode fields undergo str -> enum parsing.
        """
        from elspeth.contracts.enums import BackpressureMode, TelemetryGranularity

        settings = TelemetrySettings(
            enabled=True,
            granularity="full",
            backpressure_mode="drop",
            fail_on_total_exporter_failure=False,
            max_consecutive_failures=5,
            exporters=[
                ExporterSettings(name="console", options={"pretty": True}),
            ],
        )

        runtime_config = RuntimeTelemetryConfig.from_settings(settings)

        assert runtime_config.enabled is True
        assert runtime_config.granularity == TelemetryGranularity.FULL
        assert runtime_config.backpressure_mode == BackpressureMode.DROP
        assert runtime_config.fail_on_total_exporter_failure is False
        assert runtime_config.max_consecutive_failures == 5
        assert len(runtime_config.exporter_configs) == 1
        assert runtime_config.exporter_configs[0].name == "console"
        assert runtime_config.exporter_configs[0].options == {"pretty": True}


class TestAllFromSettingsMapEveryField:
    """Verify no Settings field is orphaned (missing from RuntimeConfig).

    For each Settings -> RuntimeConfig pair, every Settings field must appear
    in the RuntimeConfig (possibly renamed via FIELD_MAPPINGS) or be documented
    in INTERNAL_DEFAULTS as intentionally internal. This is the "no orphan
    fields" integration test.
    """

    @staticmethod
    def _get_settings_fields(settings_cls: type[BaseModel]) -> set[str]:
        """Get all field names from a Pydantic Settings model."""
        return set(settings_cls.model_fields.keys())

    @staticmethod
    def _get_runtime_fields(runtime_cls: type) -> set[str]:
        """Get all field names from a frozen dataclass."""
        return {f.name for f in dataclasses.fields(runtime_cls)}

    @staticmethod
    def _get_internal_fields(settings_class_name: str) -> set[str]:
        """Get runtime fields that are internal-only (not from Settings).

        Internal fields are documented in INTERNAL_DEFAULTS and have no
        corresponding Settings field. They are hardcoded in from_settings().
        """
        from elspeth.contracts.config.alignment import RUNTIME_TO_SUBSYSTEM

        runtime_class_name = SETTINGS_TO_RUNTIME[settings_class_name]
        if runtime_class_name not in RUNTIME_TO_SUBSYSTEM:
            return set()

        subsystem = RUNTIME_TO_SUBSYSTEM[runtime_class_name]
        if subsystem not in INTERNAL_DEFAULTS:
            return set()

        return set(INTERNAL_DEFAULTS[subsystem].keys())

    def _assert_no_orphaned_fields(
        self,
        settings_cls: type,
        runtime_cls: type,
        settings_class_name: str,
    ) -> None:
        """Assert every Settings field maps to a RuntimeConfig field.

        A Settings field is accounted for if:
        1. It has the same name in RuntimeConfig (direct mapping), OR
        2. It is renamed via FIELD_MAPPINGS and the target name exists, OR
        3. It is a conditional/computed field consumed by from_settings() logic
           (e.g., checkpoint_interval feeds into frequency computation)

        A RuntimeConfig field is accounted for if:
        1. It has the same name in Settings (direct mapping), OR
        2. It is the target of a FIELD_MAPPINGS rename, OR
        3. It is documented in INTERNAL_DEFAULTS (hardcoded, not from Settings)
        """
        settings_fields = self._get_settings_fields(settings_cls)
        runtime_fields = self._get_runtime_fields(runtime_cls)
        internal_fields = self._get_internal_fields(settings_class_name)
        field_renames: dict[str, str] | MappingProxyType[str, str] = FIELD_MAPPINGS.get(settings_class_name, {})

        # Check: every Settings field must map to a RuntimeConfig field
        for field_name in settings_fields:
            runtime_name = field_renames.get(field_name, field_name)
            assert runtime_name in runtime_fields, (
                f"{settings_class_name}.{field_name} has no corresponding "
                f"field in {runtime_cls.__name__} "
                f"(expected '{runtime_name}'). "
                f"Add it to {runtime_cls.__name__} or document in FIELD_MAPPINGS/INTERNAL_DEFAULTS."
            )

        # Check: every RuntimeConfig field must come from Settings or be internal
        rename_targets = set(field_renames.values())
        for field_name in runtime_fields:
            from_settings = field_name in settings_fields
            from_rename = field_name in rename_targets
            from_internal = field_name in internal_fields
            assert from_settings or from_rename or from_internal, (
                f"{runtime_cls.__name__}.{field_name} has no source. "
                f"Not in {settings_class_name} fields, not a FIELD_MAPPINGS target, "
                f"and not in INTERNAL_DEFAULTS."
            )

    def test_retry_settings_all_fields_mapped(self) -> None:
        """Every RetrySettings field must reach RuntimeRetryConfig."""
        self._assert_no_orphaned_fields(RetrySettings, RuntimeRetryConfig, "RetrySettings")

    def test_rate_limit_settings_all_fields_mapped(self) -> None:
        """Every RateLimitSettings field must reach RuntimeRateLimitConfig."""
        self._assert_no_orphaned_fields(RateLimitSettings, RuntimeRateLimitConfig, "RateLimitSettings")

    def test_checkpoint_settings_all_fields_mapped(self) -> None:
        """Every CheckpointSettings field must reach RuntimeCheckpointConfig."""
        self._assert_no_orphaned_fields(CheckpointSettings, RuntimeCheckpointConfig, "CheckpointSettings")

    def test_concurrency_settings_all_fields_mapped(self) -> None:
        """Every ConcurrencySettings field must reach RuntimeConcurrencyConfig."""
        self._assert_no_orphaned_fields(ConcurrencySettings, RuntimeConcurrencyConfig, "ConcurrencySettings")

    def test_telemetry_settings_all_fields_mapped(self) -> None:
        """Every TelemetrySettings field must reach RuntimeTelemetryConfig."""
        self._assert_no_orphaned_fields(TelemetrySettings, RuntimeTelemetryConfig, "TelemetrySettings")
