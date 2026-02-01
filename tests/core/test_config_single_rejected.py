"""Test that output_mode='single' is rejected with helpful error."""

import pytest
from pydantic import ValidationError

from elspeth.core.config import AggregationSettings, TriggerConfig


def test_aggregation_config_rejects_single_mode() -> None:
    """Config validation must reject 'single' mode with migration hint."""
    with pytest.raises(ValidationError) as exc_info:
        AggregationSettings(
            name="test_agg",
            plugin="test_plugin",
            trigger=TriggerConfig(count=5),
            output_mode="single",
        )

    error_msg = str(exc_info.value)
    assert "single" in error_msg.lower()
    assert "transform" in error_msg.lower()  # Migration hint


def test_aggregation_config_accepts_transform_mode() -> None:
    """Config validation must accept 'transform' mode."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="transform",
    )
    assert settings.output_mode == "transform"


def test_aggregation_config_accepts_passthrough_mode() -> None:
    """Config validation must accept 'passthrough' mode."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="passthrough",
    )
    assert settings.output_mode == "passthrough"


def test_aggregation_config_default_is_transform() -> None:
    """Default output_mode should be 'transform' (not 'single')."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
    )
    assert settings.output_mode == "transform"


def test_aggregation_config_expected_output_count() -> None:
    """expected_output_count validates output cardinality."""
    settings = AggregationSettings(
        name="test_agg",
        plugin="test_plugin",
        trigger=TriggerConfig(count=5),
        output_mode="transform",
        expected_output_count=1,
    )
    assert settings.expected_output_count == 1
