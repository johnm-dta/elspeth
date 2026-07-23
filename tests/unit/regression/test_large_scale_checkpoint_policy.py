"""Regression guards for large-scale example checkpoint policy."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any

import yaml

from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.core.config import CheckpointSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
LARGE_SCALE_DIR = REPO_ROOT / "examples" / "large_scale_test"
LARGE_SCALE_INPUT = LARGE_SCALE_DIR / "input.csv"
LARGE_SCALE_SETTINGS = LARGE_SCALE_DIR / "settings.yaml"
EXPECTED_CHECKPOINT_POLICY = {
    "enabled": True,
    "frequency": "every_n",
    "checkpoint_interval": 100,
}


def _load_example_settings() -> dict[str, Any]:
    loaded = yaml.safe_load(LARGE_SCALE_SETTINGS.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_large_scale_example_input_is_shippable() -> None:
    """The advertised 10k-row fixture must be present in clean checkouts."""
    relative_input = LARGE_SCALE_INPUT.relative_to(REPO_ROOT)
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", str(relative_input)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert ignored.returncode == 1, (
        f"{relative_input} is excluded from distributable source by: {ignored.stdout.strip() or ignored.stderr.strip()}"
    )

    with LARGE_SCALE_INPUT.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        assert rows.fieldnames == ["id", "value", "category", "priority", "timestamp"]
        assert sum(1 for _ in rows) == 10_000


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
