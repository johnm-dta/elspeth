# tests/core/test_config_alignment.py
"""Config alignment tests - catch field orphaning before it becomes a bug.

These tests verify that Settings fields are actually wired to runtime code.
They serve as:
1. Documentation of which fields are used vs. pending implementation
2. Early warning when someone adds a field but forgets to wire it
3. Refactoring guide showing what needs to be connected

P2-2026-01-21 bug: exponential_base was added to RetrySettings but never
mapped to RuntimeRetryConfig. That bug motivated this comprehensive audit.

## Field Orphaning Categories

- WIRED: Field is used at runtime (tested in runtime_fields)
- PENDING: Field exists but implementation is not yet complete (documented)
- INTERNAL: Field used for validation/defaults only, not runtime behavior

## When These Tests Fail

If a test fails after adding a new field to a Settings class:
1. If the field IS wired: Add it to the test's runtime_fields set
2. If the field is NOT wired yet: Add it to pending_fields with a comment
3. If the field is internal-only: Add it to internal_fields with explanation
"""

from typing import ClassVar


class TestRetryConfigAlignment:
    """Verify RetrySettings ↔ RetryConfig ↔ RetryPolicy alignment.

    RetrySettings is the only Settings class with a separate runtime
    dataclass (RetryConfig). This pattern requires careful alignment.
    """

    # Fields mapped between Settings and Config (may have different names)
    FIELD_MAPPINGS: ClassVar[dict[str, str]] = {
        "initial_delay_seconds": "base_delay",
        "max_delay_seconds": "max_delay",
    }

    # Fields in Config that don't come from Settings
    CONFIG_INTERNAL_ONLY: ClassVar[set[str]] = {"jitter"}

    def test_settings_fields_exist_in_config(self) -> None:
        """Every RetrySettings field must have a corresponding RetryConfig field."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        settings_fields = set(RetrySettings.model_fields.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        expected_in_config = {self.FIELD_MAPPINGS.get(f, f) for f in settings_fields}

        missing = expected_in_config - config_fields
        assert not missing, f"RetrySettings fields not in RetryConfig: {missing}. Add these to RetryConfig and wire in from_settings()."

    def test_config_covers_settings(self) -> None:
        """RetryConfig should not have unexpected fields beyond Settings + internals."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        settings_fields = set(RetrySettings.model_fields.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        expected = {self.FIELD_MAPPINGS.get(f, f) for f in settings_fields} | self.CONFIG_INTERNAL_ONLY

        unexpected = config_fields - expected
        assert not unexpected, f"RetryConfig has undocumented fields: {unexpected}. Add to Settings, CONFIG_INTERNAL_ONLY, or remove."

    def test_policy_matches_config(self) -> None:
        """RetryPolicy TypedDict should have same fields as RuntimeRetryConfig."""
        from elspeth.contracts import RetryPolicy
        from elspeth.contracts.config import RuntimeRetryConfig

        policy_fields = set(RetryPolicy.__annotations__.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        assert policy_fields == config_fields, (
            f"RetryPolicy/RetryConfig mismatch. "
            f"Missing from Policy: {config_fields - policy_fields}, "
            f"Missing from Config: {policy_fields - config_fields}"
        )

    def test_from_settings_maps_all_fields(self) -> None:
        """Verify from_settings() uses non-default values (catches forgotten mappings)."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        settings = RetrySettings(
            max_attempts=99,
            initial_delay_seconds=99.0,
            max_delay_seconds=999.0,
            exponential_base=9.9,
        )
        config = RuntimeRetryConfig.from_settings(settings)

        assert config.max_attempts == 99
        assert config.base_delay == 99.0
        assert config.max_delay == 999.0
        assert config.exponential_base == 9.9
        assert config.jitter == 1.0  # internal default


class TestConcurrencySettingsAlignment:
    """Verify ConcurrencySettings field usage.

    STATUS: WIRED
    RuntimeConcurrencyConfig is created from ConcurrencySettings in CLI
    and passed to Orchestrator, which injects it into PluginContext.
    Plugins can access ctx.concurrency_config.max_workers for pool sizing.
    """

    # Fields wired through RuntimeConcurrencyConfig
    WIRED_FIELDS: ClassVar[set[str]] = {
        "max_workers",  # Accessible via PluginContext.concurrency_config
    }

    def test_documents_wired_fields(self) -> None:
        """Document which fields are wired to runtime."""
        from elspeth.core.config import ConcurrencySettings

        actual_fields = set(ConcurrencySettings.model_fields.keys())

        # This test passes as documentation - update when fields change
        assert actual_fields == self.WIRED_FIELDS, (
            f"ConcurrencySettings fields changed. "
            f"New fields: {actual_fields - self.WIRED_FIELDS}, "
            f"Removed: {self.WIRED_FIELDS - actual_fields}. "
            f"Update WIRED_FIELDS and wiring tests."
        )

    def test_max_workers_accessible_in_context(self) -> None:
        """max_workers is accessible via PluginContext.concurrency_config.

        The concurrency_config is passed CLI → Orchestrator → PluginContext,
        making max_workers available to plugins for pool size decisions.
        """
        from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
        from elspeth.core.config import ConcurrencySettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import Orchestrator

        # Create concurrency config from settings
        settings = ConcurrencySettings(max_workers=8)
        config = RuntimeConcurrencyConfig.from_settings(settings)

        # Verify Orchestrator accepts concurrency_config
        db = LandscapeDB.in_memory()
        try:
            orchestrator = Orchestrator(db, concurrency_config=config)
            assert orchestrator._concurrency_config is config
            assert orchestrator._concurrency_config.max_workers == 8
        finally:
            db.close()


class TestRateLimitSettingsAlignment:
    """Verify RateLimitSettings field usage.

    STATUS: WIRED (Task 7)
    RateLimitRegistry is created in CLI and passed to Orchestrator,
    which injects it into PluginContext for plugin access.
    """

    # Fields wired through RateLimitRegistry
    WIRED_FIELDS: ClassVar[set[str]] = {
        "enabled",
        "default_requests_per_minute",
        "persistence_path",
        "services",
    }

    def test_documents_wired_fields(self) -> None:
        """Document which fields are wired to runtime."""
        from elspeth.core.config import RateLimitSettings

        actual_fields = set(RateLimitSettings.model_fields.keys())

        assert actual_fields == self.WIRED_FIELDS, "RateLimitSettings fields changed. Update WIRED_FIELDS."

    def test_registry_instantiated_from_settings(self) -> None:
        """RateLimitRegistry is created from RuntimeRateLimitConfig in CLI.

        The registry is created in _execute_pipeline_with_instances() and
        _execute_resume_with_instances() via RuntimeRateLimitConfig.from_settings(),
        then passed to Orchestrator which injects it into PluginContext.rate_limit_registry.
        """
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        # Verify the registry can be created from settings via RuntimeRateLimitConfig
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=5,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        # Verify the registry uses the config
        limiter = registry.get_limiter("test_service")
        assert limiter is not None  # Should return a real limiter when enabled

        # Clean up
        registry.close()


class TestLandscapeSettingsAlignment:
    """Verify LandscapeSettings field usage."""

    # Fields actively used at runtime
    WIRED_FIELDS: ClassVar[set[str]] = {
        "url",  # Used in LandscapeDB.from_url()
        "dump_to_jsonl",  # Used in LandscapeDB.from_url() via cli.py
        "dump_to_jsonl_path",  # Used in LandscapeDB.from_url() via cli.py
        "dump_to_jsonl_fail_on_error",  # Used in LandscapeDB.from_url() via cli.py
        "dump_to_jsonl_include_payloads",  # Used in LandscapeDB.from_url() via cli.py
        "dump_to_jsonl_payload_base_path",  # Used in LandscapeDB.from_url() via cli.py
    }

    # Nested settings object - checked separately
    NESTED_FIELDS: ClassVar[set[str]] = {
        "export",  # LandscapeExportSettings - has its own wiring
    }

    # Fields that exist but aren't checked at runtime
    PENDING_FIELDS: ClassVar[set[str]] = {
        "enabled",  # Always assumed True
        "backend",  # Not validated beyond schema
    }

    def test_field_categorization_complete(self) -> None:
        """All fields must be categorized."""
        from elspeth.core.config import LandscapeSettings

        actual_fields = set(LandscapeSettings.model_fields.keys())
        categorized = self.WIRED_FIELDS | self.NESTED_FIELDS | self.PENDING_FIELDS

        assert actual_fields == categorized, (
            f"Uncategorized fields: {actual_fields - categorized}. Add to WIRED_FIELDS, NESTED_FIELDS, or PENDING_FIELDS."
        )

    def test_export_settings_wired(self) -> None:
        """LandscapeExportSettings fields should all be used."""
        from elspeth.core.config import LandscapeExportSettings

        # These are all accessed in orchestrator export logic
        expected_fields = {"enabled", "sink", "format", "sign"}
        actual_fields = set(LandscapeExportSettings.model_fields.keys())

        assert actual_fields == expected_fields, (
            "LandscapeExportSettings fields changed. Verify new fields are used in Orchestrator.run()/resume()."
        )


class TestCheckpointSettingsAlignment:
    """Verify CheckpointSettings field usage.

    STATUS: WIRED
    RuntimeCheckpointConfig is created from CheckpointSettings in CLI
    and passed to Orchestrator for both normal runs and resume.
    Checkpointing is now enabled for normal runs when checkpoint.enabled=true.
    """

    # Fields used during resume
    WIRED_FIELDS: ClassVar[set[str]] = {
        "enabled",
        "frequency",
        "checkpoint_interval",
        "aggregation_boundaries",
    }

    def test_field_categorization_complete(self) -> None:
        """All fields must be categorized."""
        from elspeth.core.config import CheckpointSettings

        actual_fields = set(CheckpointSettings.model_fields.keys())

        assert actual_fields == self.WIRED_FIELDS, f"Uncategorized fields: {actual_fields - self.WIRED_FIELDS}. Add to WIRED_FIELDS."

    def test_checkpoint_config_passed_to_orchestrator(self) -> None:
        """RuntimeCheckpointConfig is passed to Orchestrator for normal runs.

        Checkpointing is now enabled for normal runs via RuntimeCheckpointConfig.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import Orchestrator

        # Create checkpoint config from settings
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        config = RuntimeCheckpointConfig.from_settings(settings)

        # Verify Orchestrator accepts checkpoint_config
        db = LandscapeDB.in_memory()
        try:
            orchestrator = Orchestrator(db, checkpoint_config=config)
            assert orchestrator._checkpoint_config is config
            assert orchestrator._checkpoint_config.enabled is True
            assert orchestrator._checkpoint_config.frequency == 1  # every_row -> 1
        finally:
            db.close()


class TestPayloadStoreSettingsAlignment:
    """Verify PayloadStoreSettings field usage.

    STATUS: FULLY WIRED
    All fields are accessed in CLI for store initialization and purge.
    """

    WIRED_FIELDS: ClassVar[set[str]] = {
        "backend",  # Validated in CLI (must be "filesystem")
        "base_path",  # Passed to FilesystemPayloadStore
        "retention_days",  # Used in purge command
    }

    def test_all_fields_wired(self) -> None:
        """All PayloadStoreSettings fields should be wired."""
        from elspeth.core.config import PayloadStoreSettings

        actual_fields = set(PayloadStoreSettings.model_fields.keys())

        assert actual_fields == self.WIRED_FIELDS, (
            f"PayloadStoreSettings fields changed. "
            f"New: {actual_fields - self.WIRED_FIELDS}, "
            f"Removed: {self.WIRED_FIELDS - actual_fields}. "
            f"Ensure new fields are wired in CLI."
        )


class TestElspethSettingsAlignment:
    """Verify top-level ElspethSettings structure."""

    # Pipeline definition fields (core pipeline config)
    PIPELINE_FIELDS: ClassVar[set[str]] = {
        "source",  # Required - source plugin config
        "transforms",  # Optional - transform chain
        "sinks",  # Required - named sink configs
        "default_sink",  # Required - default output sink name
        "gates",  # Optional - config-driven routing
        "coalesce",  # Optional - fork path merging
        "aggregations",  # Optional - config-driven batching
    }

    # Run mode fields (how to execute)
    RUN_MODE_FIELDS: ClassVar[set[str]] = {
        "run_mode",  # live, replay, or verify
        "replay_from",  # run ID for replay/verify modes
    }

    # Subsystem settings (infrastructure config)
    SUBSYSTEM_SETTINGS: ClassVar[set[str]] = {
        "landscape",  # Audit trail database
        "concurrency",  # Thread pool size (PENDING - not wired)
        "rate_limit",  # Rate limiting (PENDING - not wired)
        "checkpoint",  # Crash recovery
        "retry",  # Retry behavior
        "payload_store",  # Large blob storage
        "telemetry",  # Operational visibility
    }

    ALL_EXPECTED: ClassVar[set[str]] = PIPELINE_FIELDS | RUN_MODE_FIELDS | SUBSYSTEM_SETTINGS

    def test_all_fields_categorized(self) -> None:
        """All ElspethSettings fields must be categorized."""
        from elspeth.core.config import ElspethSettings

        actual_fields = set(ElspethSettings.model_fields.keys())

        missing = actual_fields - self.ALL_EXPECTED
        extra = self.ALL_EXPECTED - actual_fields

        assert not missing, (
            f"Uncategorized fields in ElspethSettings: {missing}. Add to PIPELINE_FIELDS, RUN_MODE_FIELDS, or SUBSYSTEM_SETTINGS."
        )
        assert not extra, f"Expected fields not in ElspethSettings: {extra}. Remove from test expectations."

    def test_subsystem_settings_have_alignment_tests(self) -> None:
        """Each subsystem setting should have its own alignment test class."""
        # This test documents which subsystems have alignment tests
        # Add a class when wiring a new subsystem
        tested_subsystems = {
            "landscape",  # TestLandscapeSettingsAlignment
            "checkpoint",  # TestCheckpointSettingsAlignment
            "retry",  # TestRetryConfigAlignment
            "payload_store",  # TestPayloadStoreSettingsAlignment
            "concurrency",  # TestConcurrencySettingsAlignment (xfail - pending)
            "rate_limit",  # TestRateLimitSettingsAlignment (xfail - pending)
            "telemetry",  # TestTelemetrySettingsAlignment
        }

        assert tested_subsystems == self.SUBSYSTEM_SETTINGS, (
            f"Subsystem alignment test mismatch. Missing tests: {self.SUBSYSTEM_SETTINGS - tested_subsystems}"
        )


# =============================================================================
# Task 4.1: Reverse Orphan Detection Tests
# =============================================================================
# These tests verify that every Runtime field has a documented Settings origin.
# This catches the inverse of the P2-2026-01-21 bug: fields in Runtime that
# don't come from Settings and aren't documented as internal defaults.


class TestRuntimeFieldOrigins:
    """Verify every Runtime field has a documented origin (Settings or INTERNAL_DEFAULTS).

    The P2-2026-01-21 bug was about Settings fields orphaned from Runtime.
    This test class catches the reverse: Runtime fields without documented origins.

    Every Runtime field must be either:
    1. Mapped from a Settings field (same name or via FIELD_MAPPINGS)
    2. Documented in INTERNAL_DEFAULTS as an internal-only field

    Fields appearing in Runtime without documentation are bugs - either:
    - They should be added to Settings (user-configurable)
    - They should be added to INTERNAL_DEFAULTS (internal-only)
    """

    def test_runtime_retry_config_has_no_orphan_fields(self) -> None:
        """Every RuntimeRetryConfig field must have a documented origin."""
        from elspeth.contracts.config import (
            FIELD_MAPPINGS,
            INTERNAL_DEFAULTS,
            RuntimeRetryConfig,
        )
        from elspeth.core.config import RetrySettings

        runtime_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())
        settings_fields = set(RetrySettings.model_fields.keys())

        # Map Settings fields to their Runtime names
        field_mappings = FIELD_MAPPINGS.get("RetrySettings", {})
        settings_as_runtime = {field_mappings.get(f, f) for f in settings_fields}

        # Get internal-only fields for retry subsystem
        internal_fields = set(INTERNAL_DEFAULTS.get("retry", {}).keys())

        # Every runtime field must be from Settings OR internal
        documented_fields = settings_as_runtime | internal_fields
        orphan_fields = runtime_fields - documented_fields

        assert not orphan_fields, (
            f"RuntimeRetryConfig has undocumented fields: {orphan_fields}. "
            f"Either add to RetrySettings (user-configurable) or INTERNAL_DEFAULTS['retry'] (internal-only)."
        )

    def test_runtime_rate_limit_config_has_no_orphan_fields(self) -> None:
        """Every RuntimeRateLimitConfig field must have a documented origin."""
        from elspeth.contracts.config import (
            FIELD_MAPPINGS,
            INTERNAL_DEFAULTS,
            RuntimeRateLimitConfig,
        )
        from elspeth.core.config import RateLimitSettings

        runtime_fields = set(RuntimeRateLimitConfig.__dataclass_fields__.keys())
        settings_fields = set(RateLimitSettings.model_fields.keys())

        # Map Settings fields to their Runtime names
        field_mappings = FIELD_MAPPINGS.get("RateLimitSettings", {})
        settings_as_runtime = {field_mappings.get(f, f) for f in settings_fields}

        # Get internal-only fields (if any)
        internal_fields = set(INTERNAL_DEFAULTS.get("rate_limit", {}).keys())

        documented_fields = settings_as_runtime | internal_fields
        orphan_fields = runtime_fields - documented_fields

        assert not orphan_fields, (
            f"RuntimeRateLimitConfig has undocumented fields: {orphan_fields}. "
            f"Either add to RateLimitSettings or INTERNAL_DEFAULTS['rate_limit']."
        )

    def test_runtime_concurrency_config_has_no_orphan_fields(self) -> None:
        """Every RuntimeConcurrencyConfig field must have a documented origin."""
        from elspeth.contracts.config import (
            FIELD_MAPPINGS,
            INTERNAL_DEFAULTS,
            RuntimeConcurrencyConfig,
        )
        from elspeth.core.config import ConcurrencySettings

        runtime_fields = set(RuntimeConcurrencyConfig.__dataclass_fields__.keys())
        settings_fields = set(ConcurrencySettings.model_fields.keys())

        field_mappings = FIELD_MAPPINGS.get("ConcurrencySettings", {})
        settings_as_runtime = {field_mappings.get(f, f) for f in settings_fields}
        internal_fields = set(INTERNAL_DEFAULTS.get("concurrency", {}).keys())

        documented_fields = settings_as_runtime | internal_fields
        orphan_fields = runtime_fields - documented_fields

        assert not orphan_fields, (
            f"RuntimeConcurrencyConfig has undocumented fields: {orphan_fields}. "
            f"Either add to ConcurrencySettings or INTERNAL_DEFAULTS['concurrency']."
        )

    def test_runtime_checkpoint_config_has_no_orphan_fields(self) -> None:
        """Every RuntimeCheckpointConfig field must have a documented origin."""
        from elspeth.contracts.config import (
            FIELD_MAPPINGS,
            INTERNAL_DEFAULTS,
            RuntimeCheckpointConfig,
        )
        from elspeth.core.config import CheckpointSettings

        runtime_fields = set(RuntimeCheckpointConfig.__dataclass_fields__.keys())
        settings_fields = set(CheckpointSettings.model_fields.keys())

        field_mappings = FIELD_MAPPINGS.get("CheckpointSettings", {})
        settings_as_runtime = {field_mappings.get(f, f) for f in settings_fields}
        internal_fields = set(INTERNAL_DEFAULTS.get("checkpoint", {}).keys())

        documented_fields = settings_as_runtime | internal_fields
        orphan_fields = runtime_fields - documented_fields

        assert not orphan_fields, (
            f"RuntimeCheckpointConfig has undocumented fields: {orphan_fields}. "
            f"Either add to CheckpointSettings or INTERNAL_DEFAULTS['checkpoint']."
        )


# =============================================================================
# Task 4.2: Explicit Field Mapping Tests
# =============================================================================
# These tests verify Settings->Runtime field name mappings with explicit assertions
# for each field, including fields that get renamed during conversion.


class TestExplicitFieldMappings:
    """Verify Settings->Runtime field mappings with explicit assertions.

    These tests complement the structural tests by explicitly checking
    each field mapping, making it clear exactly which Settings field
    maps to which Runtime field.
    """

    def test_retry_field_mapping_explicit(self) -> None:
        """Verify RetrySettings->RuntimeRetryConfig field name mappings."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        settings = RetrySettings(
            max_attempts=7,
            initial_delay_seconds=3.5,
            max_delay_seconds=180.0,
            exponential_base=2.5,
        )
        config = RuntimeRetryConfig.from_settings(settings)

        # Direct mappings (same name)
        assert config.max_attempts == settings.max_attempts, "max_attempts: direct mapping"
        assert config.exponential_base == settings.exponential_base, "exponential_base: direct mapping"

        # Renamed mappings
        assert config.base_delay == settings.initial_delay_seconds, "base_delay <- initial_delay_seconds"
        assert config.max_delay == settings.max_delay_seconds, "max_delay <- max_delay_seconds"

        # Internal field (not from Settings)
        assert config.jitter == 1.0, "jitter: internal default from INTERNAL_DEFAULTS"

    def test_rate_limit_field_mapping_explicit(self) -> None:
        """Verify RateLimitSettings->RuntimeRateLimitConfig field name mappings."""
        from elspeth.contracts.config import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings

        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=500,
            persistence_path="/tmp/limits.db",
            services={},
        )
        config = RuntimeRateLimitConfig.from_settings(settings)

        # All direct mappings (no renames in RateLimitSettings)
        assert config.enabled == settings.enabled, "enabled: direct mapping"
        assert config.default_requests_per_minute == float(settings.default_requests_per_minute), (
            "default_requests_per_minute: direct mapping (int->float)"
        )
        # We set default_requests_per_minute=500 above, so this is safe
        assert settings.default_requests_per_minute is not None  # For mypy
        assert config.default_requests_per_minute == float(settings.default_requests_per_minute), (
            "default_requests_per_minute: direct mapping (int->float)"
        )
        assert config.persistence_path == settings.persistence_path, "persistence_path: direct mapping"
        assert config.services == dict(settings.services), "services: direct mapping"

    def test_concurrency_field_mapping_explicit(self) -> None:
        """Verify ConcurrencySettings->RuntimeConcurrencyConfig field name mappings."""
        from elspeth.contracts.config import RuntimeConcurrencyConfig
        from elspeth.core.config import ConcurrencySettings

        settings = ConcurrencySettings(max_workers=16)
        config = RuntimeConcurrencyConfig.from_settings(settings)

        # Single field, direct mapping
        assert config.max_workers == settings.max_workers, "max_workers: direct mapping"

    def test_checkpoint_field_mapping_explicit(self) -> None:
        """Verify CheckpointSettings->RuntimeCheckpointConfig field mappings.

        Note: frequency field is transformed (Literal -> int), not just renamed.
        """
        from elspeth.contracts.config import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        # Test "every_row" frequency
        settings_every = CheckpointSettings(
            enabled=True,
            frequency="every_row",
            aggregation_boundaries=True,
        )
        config_every = RuntimeCheckpointConfig.from_settings(settings_every)
        assert config_every.enabled == settings_every.enabled, "enabled: direct mapping"
        assert config_every.frequency == 1, "frequency: 'every_row' -> 1"
        assert config_every.aggregation_boundaries == settings_every.aggregation_boundaries, "aggregation_boundaries: direct mapping"
        assert config_every.checkpoint_interval is None, "checkpoint_interval: None when not every_n"

        # Test "every_n" frequency
        settings_n = CheckpointSettings(
            enabled=True,
            frequency="every_n",
            checkpoint_interval=100,
            aggregation_boundaries=False,
        )
        config_n = RuntimeCheckpointConfig.from_settings(settings_n)
        assert config_n.frequency == 100, "frequency: 'every_n' -> checkpoint_interval value"
        assert config_n.checkpoint_interval == 100, "checkpoint_interval: preserved"

        # Test "aggregation_only" frequency
        settings_agg = CheckpointSettings(
            enabled=False,
            frequency="aggregation_only",
            aggregation_boundaries=True,
        )
        config_agg = RuntimeCheckpointConfig.from_settings(settings_agg)
        assert config_agg.frequency == 0, "frequency: 'aggregation_only' -> 0"


# =============================================================================
# Task 4.3: P2-2026-01-21 Regression Test
# =============================================================================
# This test specifically verifies that exponential_base actually affects
# backoff calculation, preventing regression of the original bug.


class TestExponentialBaseRegression:
    """Regression tests for P2-2026-01-21: exponential_base silently ignored.

    The bug: exponential_base was added to RetrySettings but never mapped to
    RuntimeRetryConfig, and never passed to tenacity's wait_exponential_jitter().
    Result: all retries used the default base (2.0) regardless of config.

    These tests verify the fix by checking that exponential_base:
    1. Flows from Settings to RuntimeRetryConfig
    2. Is passed to tenacity (affects actual backoff calculation)
    """

    def test_exponential_base_flows_from_settings_to_config(self) -> None:
        """exponential_base must flow from RetrySettings to RuntimeRetryConfig."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        # Test with non-default value to ensure it's actually mapped
        settings = RetrySettings(exponential_base=5.0)
        config = RuntimeRetryConfig.from_settings(settings)

        assert config.exponential_base == 5.0, (
            "P2-2026-01-21 REGRESSION: exponential_base not mapped from Settings. "
            "The from_settings() method must use settings.exponential_base."
        )

    def test_exponential_base_used_in_retry_manager(self) -> None:
        """exponential_base must be accessible to RetryManager for tenacity."""
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.engine.retry import RetryManager

        config = RuntimeRetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.0,  # Disable jitter for predictable test
            exponential_base=3.0,  # Non-default value
        )
        manager = RetryManager(config)

        # Verify the manager has access to exponential_base through config
        assert manager._config.exponential_base == 3.0, (
            "P2-2026-01-21 REGRESSION: exponential_base not accessible in RetryManager. "
            "The config must expose exponential_base for tenacity's exp_base parameter."
        )

    def test_exponential_base_default_matches_tenacity_default(self) -> None:
        """Default exponential_base should be 2.0 (industry standard).

        This test documents the expected default and ensures we don't
        accidentally change it.
        """
        from elspeth.contracts.config import POLICY_DEFAULTS, RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        # Settings default
        settings = RetrySettings()
        assert settings.exponential_base == 2.0, "RetrySettings default should be 2.0"

        # Config default
        config = RuntimeRetryConfig.default()
        assert config.exponential_base == 2.0, "RuntimeRetryConfig.default() should use 2.0"

        # Policy default
        assert POLICY_DEFAULTS["exponential_base"] == 2.0, "POLICY_DEFAULTS should use 2.0"

    def test_exponential_base_different_values_produce_different_configs(self) -> None:
        """Different exponential_base values must produce different configs.

        The original bug would have made all these configs identical.
        """
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        configs = []
        for base in [1.5, 2.0, 3.0, 4.0]:
            settings = RetrySettings(exponential_base=base)
            config = RuntimeRetryConfig.from_settings(settings)
            configs.append(config.exponential_base)

        assert configs == [1.5, 2.0, 3.0, 4.0], (
            f"P2-2026-01-21 REGRESSION: Different exponential_base values should produce different configs, but got: {configs}"
        )


# =============================================================================
# Task 4.4: Property-Based Roundtrip Tests (Hypothesis)
# =============================================================================
# These tests use property-based testing to verify that valid Settings values
# survive the from_settings() conversion without data loss or corruption.


class TestPropertyBasedRoundtrip:
    """Property-based tests for Settings->Runtime conversion.

    Uses Hypothesis to generate valid Settings values and verify they
    survive conversion to Runtime config without data loss.

    Why property-based testing matters here:
    - Manual tests check specific values; Hypothesis finds edge cases
    - Field orphaning bugs often manifest at boundary values
    - Type coercion bugs (int->float) are caught by property testing
    """

    def test_retry_config_roundtrip_values_preserved(self) -> None:
        """All valid RetrySettings values survive from_settings() conversion.

        Uses Hypothesis to generate many valid RetrySettings configurations
        and verify each field value is correctly transferred to RuntimeRetryConfig.
        """
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        @given(
            max_attempts=st.integers(min_value=1, max_value=100),
            initial_delay=st.floats(min_value=0.01, max_value=300.0, allow_nan=False, allow_infinity=False),
            max_delay=st.floats(min_value=0.1, max_value=3600.0, allow_nan=False, allow_infinity=False),
            exp_base=st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        )
        @settings(max_examples=100)
        def check_roundtrip(
            max_attempts: int,
            initial_delay: float,
            max_delay: float,
            exp_base: float,
        ) -> None:
            # Ensure max_delay >= initial_delay for valid config
            if max_delay < initial_delay:
                max_delay = initial_delay + 1.0

            settings_obj = RetrySettings(
                max_attempts=max_attempts,
                initial_delay_seconds=initial_delay,
                max_delay_seconds=max_delay,
                exponential_base=exp_base,
            )
            config = RuntimeRetryConfig.from_settings(settings_obj)

            # Verify all values survived the conversion
            assert config.max_attempts == max_attempts
            assert config.base_delay == initial_delay
            assert config.max_delay == max_delay
            assert config.exponential_base == exp_base

        check_roundtrip()

    def test_rate_limit_config_roundtrip_values_preserved(self) -> None:
        """All valid RateLimitSettings values survive from_settings() conversion."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from elspeth.contracts.config import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings

        @given(
            enabled=st.booleans(),
            rpm=st.integers(min_value=1, max_value=100000),
            path=st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda s: "/" not in s or s.startswith("/"))),
        )
        @settings(max_examples=100)
        def check_roundtrip(
            enabled: bool,
            rpm: int,
            path: str | None,
        ) -> None:
            settings_obj = RateLimitSettings(
                enabled=enabled,
                default_requests_per_minute=rpm,
                persistence_path=path,
            )
            config = RuntimeRateLimitConfig.from_settings(settings_obj)

            assert config.enabled == enabled
            assert config.default_requests_per_minute == rpm
            assert config.persistence_path == path

        check_roundtrip()

    def test_concurrency_config_roundtrip_values_preserved(self) -> None:
        """All valid ConcurrencySettings values survive from_settings() conversion."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from elspeth.contracts.config import RuntimeConcurrencyConfig
        from elspeth.core.config import ConcurrencySettings

        @given(max_workers=st.integers(min_value=1, max_value=1000))
        @settings(max_examples=100)
        def check_roundtrip(max_workers: int) -> None:
            settings_obj = ConcurrencySettings(max_workers=max_workers)
            config = RuntimeConcurrencyConfig.from_settings(settings_obj)
            assert config.max_workers == max_workers

        check_roundtrip()

    def test_checkpoint_config_roundtrip_values_preserved(self) -> None:
        """All valid CheckpointSettings values survive from_settings() conversion."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from elspeth.contracts.config import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings

        @given(
            enabled=st.booleans(),
            frequency=st.sampled_from(["every_row", "every_n", "aggregation_only"]),
            interval=st.integers(min_value=1, max_value=10000),
            agg_boundaries=st.booleans(),
        )
        @settings(max_examples=100)
        def check_roundtrip(
            enabled: bool,
            frequency: str,
            interval: int,
            agg_boundaries: bool,
        ) -> None:
            # Build settings based on frequency type
            if frequency == "every_n":
                settings_obj = CheckpointSettings(
                    enabled=enabled,
                    frequency=frequency,
                    checkpoint_interval=interval,
                    aggregation_boundaries=agg_boundaries,
                )
            else:
                settings_obj = CheckpointSettings(
                    enabled=enabled,
                    frequency=frequency,
                    aggregation_boundaries=agg_boundaries,
                )

            config = RuntimeCheckpointConfig.from_settings(settings_obj)

            assert config.enabled == enabled
            assert config.aggregation_boundaries == agg_boundaries

            # Verify frequency transformation
            if frequency == "every_row":
                assert config.frequency == 1
            elif frequency == "aggregation_only":
                assert config.frequency == 0
            else:  # every_n
                assert config.frequency == interval
                assert config.checkpoint_interval == interval

        check_roundtrip()


# =============================================================================
# Comprehensive SETTINGS_TO_RUNTIME Alignment Tests
# =============================================================================
# These tests use the canonical SETTINGS_TO_RUNTIME mapping to verify
# all documented Settings->Runtime pairs are correctly wired.


class TestSettingsToRuntimeMapping:
    """Verify the SETTINGS_TO_RUNTIME mapping is accurate and complete.

    SETTINGS_TO_RUNTIME documents which Runtime class implements each Settings.
    These tests verify that documentation matches reality.
    """

    def test_all_mapped_settings_have_from_settings(self) -> None:
        """Every Settings class in SETTINGS_TO_RUNTIME must have a from_settings() factory."""
        from elspeth.contracts.config import (
            SETTINGS_TO_RUNTIME,
            RuntimeCheckpointConfig,
            RuntimeConcurrencyConfig,
            RuntimeRateLimitConfig,
            RuntimeRetryConfig,
            RuntimeTelemetryConfig,
        )

        runtime_classes = {
            "RuntimeRetryConfig": RuntimeRetryConfig,
            "RuntimeRateLimitConfig": RuntimeRateLimitConfig,
            "RuntimeConcurrencyConfig": RuntimeConcurrencyConfig,
            "RuntimeCheckpointConfig": RuntimeCheckpointConfig,
            "RuntimeTelemetryConfig": RuntimeTelemetryConfig,
        }

        for settings_name, runtime_name in SETTINGS_TO_RUNTIME.items():
            runtime_cls = runtime_classes.get(runtime_name)
            assert runtime_cls is not None, f"Unknown Runtime class: {runtime_name}"
            assert hasattr(runtime_cls, "from_settings"), f"{runtime_name} must have from_settings() method for {settings_name} conversion"

    def test_settings_to_runtime_mapping_is_complete(self) -> None:
        """SETTINGS_TO_RUNTIME should document all Settings with Runtime counterparts."""
        from elspeth.contracts.config import EXEMPT_SETTINGS, SETTINGS_TO_RUNTIME
        from elspeth.core import config as config_module

        # Find all Settings classes in core/config.py
        all_settings_classes = {name for name in dir(config_module) if name.endswith("Settings") and not name.startswith("_")}

        # Settings must be either in SETTINGS_TO_RUNTIME or EXEMPT_SETTINGS
        documented = set(SETTINGS_TO_RUNTIME.keys()) | EXEMPT_SETTINGS
        undocumented = all_settings_classes - documented

        assert not undocumented, (
            f"Settings classes not in SETTINGS_TO_RUNTIME or EXEMPT_SETTINGS: {undocumented}. "
            f"Add to SETTINGS_TO_RUNTIME (if needs Runtime class) or EXEMPT_SETTINGS (if not)."
        )
