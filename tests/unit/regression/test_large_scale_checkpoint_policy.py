"""Regression guards for large-scale example checkpoint policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.core.config import CheckpointSettings

LARGE_SCALE_SETTINGS = Path("examples/large_scale_test/settings.yaml")
EXPECTED_CHECKPOINT_POLICY = {
    "enabled": True,
    "frequency": "every_n",
    "checkpoint_interval": 100,
    "aggregation_boundaries": True,
}


def _load_example_settings() -> dict[str, Any]:
    loaded = yaml.safe_load(LARGE_SCALE_SETTINGS.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_large_scale_example_declares_checkpoint_policy_explicitly() -> None:
    """Benchmark configs must not rely on hidden checkpoint defaults."""
    settings = _load_example_settings()

    assert settings.get("checkpoint") == EXPECTED_CHECKPOINT_POLICY


def test_checkpoint_policy_variants_have_distinct_runtime_semantics() -> None:
    """Performance comparisons must name the active checkpoint policy."""
    hidden_default = RuntimeCheckpointConfig.from_settings(CheckpointSettings())
    coarse = RuntimeCheckpointConfig.from_settings(CheckpointSettings(frequency="every_n", checkpoint_interval=100))
    off = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=False))

    assert (hidden_default.enabled, hidden_default.frequency) == (True, 1)
    assert (coarse.enabled, coarse.frequency) == (True, 100)
    assert (off.enabled, off.frequency) == (False, 1)
    assert len({hidden_default, coarse, off}) == 3
