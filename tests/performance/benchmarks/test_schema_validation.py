"""Performance benchmarks for schema validation pipeline.

Migrated from tests/performance/test_baseline_schema_validation.py.
Measures plugin instantiation, graph construction, and end-to-end validation time.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.core.dag import ExecutionGraph
from tests.performance.conftest import benchmark_timer


@pytest.mark.performance
def test_plugin_instantiation_performance() -> None:
    """Measure plugin instantiation time.

    Baseline: Instantiation should be < 500ms for a simple pipeline with
    3 passthrough transforms (includes on_error routing validation).
    """
    config_yaml = """
sources:
  primary:
    plugin: csv
    on_success: t0_in
    options:
      path: test.csv
      schema:
        mode: observed
      on_validation_failure: discard

transforms:
  - plugin: passthrough
    name: t0
    input: t0_in
    on_success: t1_in
    on_error: discard
    options:
      schema:
        mode: observed
  - plugin: passthrough
    name: t1
    input: t1_in
    on_success: t2_in
    on_error: discard
    options:
      schema:
        mode: observed
  - plugin: passthrough
    name: t2
    input: t2_in
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        mode: observed
      format: jsonl
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)

        with benchmark_timer() as timing:
            _ = instantiate_plugins_from_config(config)

        assert timing.wall_seconds < 0.500, f"Plugin instantiation took {timing.wall_seconds * 1000:.2f}ms (expected < 500ms)"
    finally:
        config_file.unlink()


@pytest.mark.performance
def test_graph_construction_performance() -> None:
    """Measure graph construction and validation time.

    Baseline: Graph construction + validation should be < 100ms.
    """
    config_yaml = """
sources:
  primary:
    plugin: csv
    on_success: t0_in
    options:
      path: test.csv
      schema:
        mode: observed
      on_validation_failure: discard

transforms:
  - plugin: passthrough
    name: t0
    input: t0_in
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        mode: observed
      format: jsonl
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        with benchmark_timer() as timing:
            graph = ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
            )
            graph.validate()

        assert timing.wall_seconds < 0.100, f"Graph construction took {timing.wall_seconds * 1000:.2f}ms (expected < 100ms)"
    finally:
        config_file.unlink()


@pytest.mark.performance
def test_end_to_end_validation_performance() -> None:
    """Measure end-to-end validation performance.

    Baseline: Full config load + plugin instantiation + graph build + validate
    should be < 200ms.
    """
    config_yaml = """
sources:
  primary:
    plugin: csv
    on_success: t0_in
    options:
      path: test.csv
      schema:
        mode: observed
      on_validation_failure: discard

transforms:
  - plugin: passthrough
    name: t0
    input: t0_in
    on_success: t1_in
    on_error: discard
    options:
      schema:
        mode: observed
  - plugin: passthrough
    name: t1
    input: t1_in
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        mode: observed
      format: jsonl
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        with benchmark_timer() as timing:
            config = load_settings(config_file)
            plugins = instantiate_plugins_from_config(config)
            graph = ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
            )
            graph.validate()

        assert timing.wall_seconds < 0.200, f"End-to-end validation took {timing.wall_seconds * 1000:.2f}ms (expected < 200ms)"
    finally:
        config_file.unlink()
