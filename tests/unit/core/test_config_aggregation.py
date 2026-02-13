"""Tests for AggregationSettings configuration."""

import pytest
from pydantic import ValidationError


class TestTriggerConfig:
    """Tests for TriggerConfig model.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    """

    def test_count_trigger_only(self) -> None:
        """Count-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(count=100)
        assert trigger.count == 100
        assert trigger.timeout_seconds is None
        assert trigger.condition is None

    def test_timeout_trigger_only(self) -> None:
        """Timeout-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(timeout_seconds=30.0)
        assert trigger.timeout_seconds == 30.0
        assert trigger.count is None
        assert trigger.condition is None

    def test_condition_trigger_only(self) -> None:
        """Condition-only trigger configuration."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(condition="row['batch_count'] >= 50")
        assert trigger.condition == "row['batch_count'] >= 50"
        assert trigger.count is None
        assert trigger.timeout_seconds is None

    def test_combined_count_and_timeout(self) -> None:
        """Combined count + timeout triggers (first to fire wins)."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(count=100, timeout_seconds=60.0)
        assert trigger.count == 100
        assert trigger.timeout_seconds == 60.0
        assert trigger.condition is None

    def test_combined_all_triggers(self) -> None:
        """All three trigger types combined."""
        from elspeth.core.config import TriggerConfig

        trigger = TriggerConfig(
            count=1000,
            timeout_seconds=3600.0,  # 1 hour
            condition="row['batch_count'] >= 1000 and row['batch_age_seconds'] < 30.0",
        )
        assert trigger.count == 1000
        assert trigger.timeout_seconds == 3600.0
        assert trigger.condition == "row['batch_count'] >= 1000 and row['batch_age_seconds'] < 30.0"

    def test_at_least_one_trigger_required(self) -> None:
        """At least one trigger must be specified."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError, match="at least one trigger"):
            TriggerConfig()

    def test_condition_validates_expression(self) -> None:
        """Condition trigger validates expression syntax."""
        from elspeth.core.config import TriggerConfig

        # Invalid Python syntax
        with pytest.raises(ValidationError, match=r"Invalid.*syntax"):
            TriggerConfig(condition="batch_count >=")

    def test_count_must_be_positive(self) -> None:
        """Count threshold must be positive."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(count=0)

        with pytest.raises(ValidationError):
            TriggerConfig(count=-1)

    def test_timeout_must_be_positive(self) -> None:
        """Timeout must be positive."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(timeout_seconds=0)

        with pytest.raises(ValidationError):
            TriggerConfig(timeout_seconds=-1.0)

    def test_has_count_property(self) -> None:
        """has_count property indicates count trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_count = TriggerConfig(count=100)
        without_count = TriggerConfig(timeout_seconds=30.0)

        assert with_count.has_count is True
        assert without_count.has_count is False

    def test_has_timeout_property(self) -> None:
        """has_timeout property indicates timeout trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_timeout = TriggerConfig(timeout_seconds=30.0)
        without_timeout = TriggerConfig(count=100)

        assert with_timeout.has_timeout is True
        assert without_timeout.has_timeout is False

    def test_has_condition_property(self) -> None:
        """has_condition property indicates condition trigger configured."""
        from elspeth.core.config import TriggerConfig

        with_condition = TriggerConfig(condition="row['batch_count'] > 0")
        without_condition = TriggerConfig(count=100)

        assert with_condition.has_condition is True
        assert without_condition.has_condition is False

    def test_condition_rejects_forbidden_constructs(self) -> None:
        """Condition trigger rejects security-forbidden constructs."""
        from elspeth.core.config import TriggerConfig

        # Forbidden function call
        with pytest.raises(ValidationError, match="Forbidden"):
            TriggerConfig(condition="__import__('os')")

    def test_condition_rejects_non_batch_row_keys(self) -> None:
        """Condition trigger only allows batch-level keys."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError, match="unsupported row keys"):
            TriggerConfig(condition="row['type'] == 'flush_signal'")

        with pytest.raises(ValidationError, match="unsupported row keys"):
            TriggerConfig(condition="row.get('status') == 'ready'")

    def test_condition_rejects_non_literal_row_key_access(self) -> None:
        """Condition row key access must use string literals."""
        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError, match="must be string literals"):
            TriggerConfig(condition="row.get(123) == 1")


class TestAggregationSettings:
    """Tests for AggregationSettings model."""

    def test_aggregation_settings_valid(self) -> None:
        """Valid aggregation settings with all fields."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            output_mode="transform",
            input="source_out",
        )
        assert settings.name == "batch_stats"
        assert settings.plugin == "stats_aggregation"
        assert settings.trigger.count == 100
        assert settings.output_mode == "transform"

    def test_aggregation_settings_combined_triggers(self) -> None:
        """Aggregation with combined triggers (per plugin-protocol.md)."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=1000, timeout_seconds=3600.0),
            output_mode="transform",
            input="source_out",
        )
        assert settings.trigger.count == 1000
        assert settings.trigger.timeout_seconds == 3600.0

    def test_aggregation_settings_default_output_mode(self) -> None:
        """Output mode defaults to 'transform'."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            input="source_out",
        )
        assert settings.output_mode == "transform"

    def test_aggregation_settings_passthrough_mode(self) -> None:
        """Passthrough output mode is valid."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(timeout_seconds=60.0),
            output_mode="passthrough",
            input="source_out",
        )
        assert settings.output_mode == "passthrough"

    def test_aggregation_settings_transform_mode(self) -> None:
        """Transform output mode is valid."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=50),
            output_mode="transform",
            input="source_out",
        )
        assert settings.output_mode == "transform"

    def test_aggregation_settings_invalid_output_mode(self) -> None:
        """Invalid output mode is rejected."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="output_mode"):
            AggregationSettings(
                name="batch_stats",
                plugin="stats_aggregation",
                trigger=TriggerConfig(count=100),
                output_mode="invalid",  # type: ignore[arg-type]
                input="source_out",
            )

    def test_aggregation_settings_requires_name(self) -> None:
        """Name is required."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        with pytest.raises(ValidationError, match="name"):
            AggregationSettings(  # type: ignore[call-arg]
                plugin="stats_aggregation",
                trigger=TriggerConfig(count=100),
                input="source_out",
            )

    def test_aggregation_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            input="source_out",
        )
        assert settings.options == {}

    def test_aggregation_settings_with_options(self) -> None:
        """Options can be provided."""
        from elspeth.core.config import AggregationSettings, TriggerConfig

        settings = AggregationSettings(
            name="batch_stats",
            plugin="stats_aggregation",
            trigger=TriggerConfig(count=100),
            options={"fields": ["value"], "compute_mean": True},
            input="source_out",
        )
        assert settings.options == {"fields": ["value"], "compute_mean": True}


class TestElspethSettingsAggregations:
    """Tests for aggregations in ElspethSettings."""

    def test_elspeth_settings_aggregations_default_empty(self) -> None:
        """Aggregations defaults to empty list."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="csv", on_success="output"),
            sinks={"output": SinkSettings(plugin="csv")},
        )
        assert settings.aggregations == []

    def test_elspeth_settings_with_aggregations(self) -> None:
        """Aggregations can be configured."""
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TriggerConfig,
        )

        settings = ElspethSettings(
            source=SourceSettings(plugin="csv", on_success="output"),
            sinks={"output": SinkSettings(plugin="csv")},
            aggregations=[
                AggregationSettings(
                    name="batch_stats",
                    plugin="stats",
                    trigger=TriggerConfig(count=100),
                    input="source_out",
                ),
            ],
        )
        assert len(settings.aggregations) == 1
        assert settings.aggregations[0].name == "batch_stats"

    def test_elspeth_settings_rejects_duplicate_aggregation_names(self) -> None:
        """Aggregation names must be unique."""
        from elspeth.core.config import (
            AggregationSettings,
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TriggerConfig,
        )

        with pytest.raises(ValidationError, match=r"Node name 'batch_stats' is used by both"):
            ElspethSettings(
                source=SourceSettings(plugin="csv", on_success="output"),
                sinks={"output": SinkSettings(plugin="csv")},
                aggregations=[
                    AggregationSettings(
                        name="batch_stats",
                        plugin="stats",
                        trigger=TriggerConfig(count=100),
                        input="source_out",
                    ),
                    AggregationSettings(
                        name="batch_stats",  # Duplicate!
                        plugin="other_stats",
                        trigger=TriggerConfig(timeout_seconds=30),
                        input="source_out",
                    ),
                ],
            )
