# tests/unit/chaosllm/test_config.py
"""Unit tests for ChaosLLM configuration module."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from elspeth.testing.chaosllm.config import (
    DEFAULT_MEMORY_DB,
    BurstConfig,
    ChaosLLMConfig,
    ErrorInjectionConfig,
    LatencyConfig,
    MetricsConfig,
    RandomResponseConfig,
    ResponseConfig,
    ServerConfig,
    TemplateResponseConfig,
    list_presets,
    load_config,
    load_preset,
)


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_defaults(self) -> None:
        """ServerConfig has sensible defaults."""
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 4

    def test_custom_values(self) -> None:
        """ServerConfig accepts custom values."""
        config = ServerConfig(host="0.0.0.0", port=9000, workers=8)
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.workers == 8

    def test_port_validation(self) -> None:
        """Port must be in valid range."""
        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            ServerConfig(port=70000)
        with pytest.raises(ValidationError, match="greater than 0"):
            ServerConfig(port=0)

    def test_workers_validation(self) -> None:
        """Workers must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ServerConfig(workers=0)

    def test_frozen(self) -> None:
        """ServerConfig is immutable."""
        config = ServerConfig()
        with pytest.raises(ValidationError):
            config.port = 9000  # type: ignore[misc]


class TestMetricsConfig:
    """Tests for MetricsConfig model."""

    def test_defaults(self) -> None:
        """MetricsConfig has sensible defaults."""
        config = MetricsConfig()
        assert config.database == DEFAULT_MEMORY_DB
        assert config.timeseries_bucket_sec == 1

    def test_custom_values(self) -> None:
        """MetricsConfig accepts custom values."""
        config = MetricsConfig(database="/tmp/test.db", timeseries_bucket_sec=5)
        assert config.database == "/tmp/test.db"
        assert config.timeseries_bucket_sec == 5

    def test_bucket_validation(self) -> None:
        """Bucket size must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            MetricsConfig(timeseries_bucket_sec=0)


class TestRandomResponseConfig:
    """Tests for RandomResponseConfig model."""

    def test_defaults(self) -> None:
        """RandomResponseConfig has sensible defaults."""
        config = RandomResponseConfig()
        assert config.min_words == 10
        assert config.max_words == 100
        assert config.vocabulary == "english"

    def test_word_range_validation(self) -> None:
        """min_words must be <= max_words."""
        with pytest.raises(ValidationError, match=r"min_words .* must be <= max_words"):
            RandomResponseConfig(min_words=100, max_words=10)

    def test_valid_range(self) -> None:
        """Equal min/max is valid (fixed word count)."""
        config = RandomResponseConfig(min_words=50, max_words=50)
        assert config.min_words == config.max_words == 50

    def test_vocabulary_literal(self) -> None:
        """Vocabulary must be 'english' or 'lorem'."""
        config = RandomResponseConfig(vocabulary="lorem")
        assert config.vocabulary == "lorem"

        with pytest.raises(ValidationError, match="literal"):
            RandomResponseConfig(vocabulary="klingon")  # type: ignore[arg-type]


class TestLatencyConfig:
    """Tests for LatencyConfig model."""

    def test_defaults(self) -> None:
        """LatencyConfig has sensible defaults."""
        config = LatencyConfig()
        assert config.base_ms == 50
        assert config.jitter_ms == 30

    def test_zero_latency(self) -> None:
        """Zero latency is valid (for max throughput tests)."""
        config = LatencyConfig(base_ms=0, jitter_ms=0)
        assert config.base_ms == 0
        assert config.jitter_ms == 0

    def test_negative_validation(self) -> None:
        """Latency values cannot be negative."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            LatencyConfig(base_ms=-10)
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            LatencyConfig(jitter_ms=-5)


class TestBurstConfig:
    """Tests for BurstConfig model."""

    def test_defaults(self) -> None:
        """BurstConfig defaults to disabled."""
        config = BurstConfig()
        assert config.enabled is False
        assert config.interval_sec == 30
        assert config.duration_sec == 5
        assert config.rate_limit_pct == 80.0
        assert config.capacity_pct == 50.0

    def test_enabled_burst(self) -> None:
        """BurstConfig can be enabled with custom settings."""
        config = BurstConfig(
            enabled=True,
            interval_sec=15,
            duration_sec=3,
            rate_limit_pct=90.0,
            capacity_pct=60.0,
        )
        assert config.enabled is True
        assert config.interval_sec == 15
        assert config.duration_sec == 3

    def test_percentage_validation(self) -> None:
        """Percentages must be 0-100."""
        with pytest.raises(ValidationError, match="less than or equal to 100"):
            BurstConfig(rate_limit_pct=150.0)
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            BurstConfig(capacity_pct=-5.0)

    def test_interval_validation(self) -> None:
        """Interval and duration must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            BurstConfig(interval_sec=0)
        with pytest.raises(ValidationError, match="greater than 0"):
            BurstConfig(duration_sec=0)


class TestErrorInjectionConfig:
    """Tests for ErrorInjectionConfig model."""

    def test_defaults_all_zero(self) -> None:
        """All error percentages default to 0."""
        config = ErrorInjectionConfig()
        assert config.rate_limit_pct == 0.0
        assert config.capacity_529_pct == 0.0
        assert config.internal_error_pct == 0.0
        assert config.invalid_json_pct == 0.0
        assert config.timeout_pct == 0.0

    def test_percentage_bounds(self) -> None:
        """All percentage fields are bounded 0-100."""
        # Test a few representative fields
        with pytest.raises(ValidationError):
            ErrorInjectionConfig(rate_limit_pct=101.0)
        with pytest.raises(ValidationError):
            ErrorInjectionConfig(timeout_pct=-1.0)

    def test_retry_after_range(self) -> None:
        """retry_after_sec is parsed as [min, max] tuple."""
        config = ErrorInjectionConfig(retry_after_sec=[2, 10])
        assert config.retry_after_sec == (2, 10)

    def test_timeout_range(self) -> None:
        """timeout_sec is parsed as [min, max] tuple."""
        config = ErrorInjectionConfig(timeout_sec=[15, 45])
        assert config.timeout_sec == (15, 45)

    def test_slow_response_range(self) -> None:
        """slow_response_sec is parsed as [min, max] tuple."""
        config = ErrorInjectionConfig(slow_response_sec=[5, 20])
        assert config.slow_response_sec == (5, 20)

    def test_range_validation(self) -> None:
        """Range min must be <= max."""
        with pytest.raises(ValidationError, match=r"retry_after_sec min .* must be <= max"):
            ErrorInjectionConfig(retry_after_sec=[10, 5])
        with pytest.raises(ValidationError, match=r"timeout_sec min .* must be <= max"):
            ErrorInjectionConfig(timeout_sec=[60, 30])
        with pytest.raises(ValidationError, match=r"slow_response_sec min .* must be <= max"):
            ErrorInjectionConfig(slow_response_sec=[30, 10])

    def test_nested_burst_config(self) -> None:
        """ErrorInjectionConfig includes nested BurstConfig."""
        config = ErrorInjectionConfig(burst=BurstConfig(enabled=True, interval_sec=20))
        assert config.burst.enabled is True
        assert config.burst.interval_sec == 20


class TestResponseConfig:
    """Tests for ResponseConfig model."""

    def test_defaults(self) -> None:
        """ResponseConfig defaults to random mode."""
        config = ResponseConfig()
        assert config.mode == "random"

    def test_mode_literal(self) -> None:
        """Mode must be one of the valid literals."""
        for mode in ["random", "template", "echo", "preset"]:
            config = ResponseConfig(mode=mode)  # type: ignore[arg-type]
            assert config.mode == mode

        with pytest.raises(ValidationError, match="literal"):
            ResponseConfig(mode="invalid")  # type: ignore[arg-type]

    def test_nested_configs(self) -> None:
        """ResponseConfig has nested random/template/preset configs."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body='{"test": true}'),
        )
        assert config.template.body == '{"test": true}'


class TestChaosLLMConfig:
    """Tests for top-level ChaosLLMConfig model."""

    def test_defaults(self) -> None:
        """ChaosLLMConfig builds with all defaults."""
        config = ChaosLLMConfig()
        assert config.server.port == 8000
        assert config.metrics.timeseries_bucket_sec == 1
        assert config.response.mode == "random"
        assert config.latency.base_ms == 50
        assert config.error_injection.rate_limit_pct == 0.0

    def test_nested_override(self) -> None:
        """Nested configs can be overridden."""
        config = ChaosLLMConfig(
            server=ServerConfig(port=9000),
            latency=LatencyConfig(base_ms=100),
        )
        assert config.server.port == 9000
        assert config.latency.base_ms == 100
        # Other defaults preserved
        assert config.metrics.database == DEFAULT_MEMORY_DB

    def test_frozen(self) -> None:
        """ChaosLLMConfig is immutable."""
        config = ChaosLLMConfig()
        with pytest.raises(ValidationError):
            config.server = ServerConfig(port=9999)  # type: ignore[misc]


class TestPresetLoading:
    """Tests for preset loading functionality."""

    def test_list_presets(self) -> None:
        """list_presets returns available preset names."""
        presets = list_presets()
        assert "gentle" in presets
        assert "realistic" in presets
        assert "stress_aimd" in presets
        assert "chaos" in presets
        assert "silent" in presets

    def test_load_gentle_preset(self) -> None:
        """Load gentle preset and verify key settings."""
        config_dict = load_preset("gentle")
        assert "error_injection" in config_dict
        assert config_dict["error_injection"]["rate_limit_pct"] == 1.0
        assert config_dict["latency"]["base_ms"] == 50

    def test_load_stress_aimd_preset(self) -> None:
        """Load stress_aimd preset and verify burst settings."""
        config_dict = load_preset("stress_aimd")
        assert config_dict["error_injection"]["rate_limit_pct"] == 15.0
        assert config_dict["error_injection"]["burst"]["enabled"] is True
        assert config_dict["error_injection"]["burst"]["interval_sec"] == 30

    def test_load_silent_preset(self) -> None:
        """Load silent preset and verify zero errors."""
        config_dict = load_preset("silent")
        assert config_dict["error_injection"]["rate_limit_pct"] == 0.0
        assert config_dict["error_injection"]["timeout_pct"] == 0.0
        assert config_dict["latency"]["base_ms"] == 10

    def test_load_chaos_preset(self) -> None:
        """Load chaos preset and verify ~25% total error rate."""
        config_dict = load_preset("chaos")
        # Scaled to ~25% total (was 40%, multiplied by 0.625)
        assert config_dict["error_injection"]["rate_limit_pct"] == 6.25
        assert config_dict["error_injection"]["invalid_json_pct"] == 1.25
        assert config_dict["error_injection"]["burst"]["enabled"] is True

    def test_load_nonexistent_preset(self) -> None:
        """Loading nonexistent preset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Preset 'nonexistent' not found"):
            load_preset("nonexistent")


class TestLoadConfig:
    """Tests for load_config with precedence handling."""

    def test_load_defaults_only(self) -> None:
        """load_config with no args returns defaults."""
        config = load_config()
        assert config.server.port == 8000
        assert config.error_injection.rate_limit_pct == 0.0

    def test_load_from_preset(self) -> None:
        """load_config with preset applies preset values."""
        config = load_config(preset="gentle")
        assert config.error_injection.rate_limit_pct == 1.0
        assert config.latency.base_ms == 50

    def test_cli_overrides_preset(self) -> None:
        """CLI overrides take precedence over preset."""
        config = load_config(
            preset="gentle",
            cli_overrides={"error_injection": {"rate_limit_pct": 25.0}},
        )
        # CLI override wins
        assert config.error_injection.rate_limit_pct == 25.0
        # Preset value preserved for non-overridden fields
        assert config.latency.base_ms == 50

    def test_config_file_overrides_preset(self, tmp_path: Path) -> None:
        """Config file overrides preset but not CLI."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
error_injection:
  rate_limit_pct: 50.0
latency:
  base_ms: 200
""")
        config = load_config(
            preset="gentle",
            config_file=config_file,
        )
        # File overrides preset
        assert config.error_injection.rate_limit_pct == 50.0
        assert config.latency.base_ms == 200

    def test_full_precedence_chain(self, tmp_path: Path) -> None:
        """Test full precedence: CLI > file > preset > defaults."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
error_injection:
  rate_limit_pct: 50.0
  capacity_529_pct: 10.0
latency:
  base_ms: 200
""")
        config = load_config(
            preset="gentle",  # rate_limit=1.0, base_ms=50
            config_file=config_file,  # rate_limit=50.0, capacity_529=10.0, base_ms=200
            cli_overrides={"error_injection": {"rate_limit_pct": 99.0}},  # rate_limit=99.0
        )
        # CLI wins for rate_limit
        assert config.error_injection.rate_limit_pct == 99.0
        # File wins for capacity_529 and base_ms (not overridden by CLI)
        assert config.error_injection.capacity_529_pct == 10.0
        assert config.latency.base_ms == 200
        # Preset wins for other gentle-specific settings
        assert config.latency.jitter_ms == 20

    def test_nonexistent_config_file(self, tmp_path: Path) -> None:
        """Loading nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_file=tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_file(self, tmp_path: Path) -> None:
        """Malformed YAML raises appropriate error."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("this: is: not: valid: yaml: [")

        with pytest.raises(yaml.YAMLError):
            load_config(config_file=config_file)


class TestPresetValidation:
    """Tests that all preset files produce valid configs."""

    @pytest.mark.parametrize("preset_name", list_presets())
    def test_preset_produces_valid_config(self, preset_name: str) -> None:
        """Each preset file produces a valid ChaosLLMConfig."""
        config = load_config(preset=preset_name)
        assert isinstance(config, ChaosLLMConfig)
        # Validate nested configs are present
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.error_injection, ErrorInjectionConfig)
        assert isinstance(config.error_injection.burst, BurstConfig)

    @pytest.mark.parametrize("preset_name", list_presets())
    def test_preset_has_all_fields(self, preset_name: str) -> None:
        """Each preset explicitly sets all error injection fields."""
        config_dict = load_preset(preset_name)
        error_config = config_dict["error_injection"]

        # All HTTP error fields present
        assert "rate_limit_pct" in error_config
        assert "capacity_529_pct" in error_config
        assert "service_unavailable_pct" in error_config
        assert "internal_error_pct" in error_config

        # All connection-level fields present
        assert "timeout_pct" in error_config
        assert "connection_reset_pct" in error_config
        assert "slow_response_pct" in error_config

        # All malformed response fields present
        assert "invalid_json_pct" in error_config
        assert "truncated_pct" in error_config
        assert "empty_body_pct" in error_config

        # Burst config present
        assert "burst" in error_config
        assert "enabled" in error_config["burst"]
