"""Performance baseline tests for schema validation refactor.

Measures validation time and plugin instantiation overhead.
Critical for validating architectural change doesn't degrade performance.
"""

import tempfile
import time
from pathlib import Path

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.core.dag import ExecutionGraph


@pytest.mark.performance
def test_plugin_instantiation_performance():
    """Measure plugin instantiation time."""

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        fields: dynamic
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)

        start = time.perf_counter()
        _ = instantiate_plugins_from_config(config)
        instantiation_time = time.perf_counter() - start

        # Baseline: Instantiation should be < 100ms for simple pipeline
        # Note: Includes plugin discovery, class loading, and schema initialization
        assert instantiation_time < 0.100, f"Plugin instantiation took {instantiation_time * 1000:.2f}ms (expected < 100ms)"

        print(f"\nPlugin instantiation: {instantiation_time * 1000:.2f}ms")

    finally:
        config_file.unlink()


@pytest.mark.performance
def test_graph_construction_performance():
    """Measure graph construction and validation time."""

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        fields: dynamic
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        start = time.perf_counter()
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
        graph.validate()
        graph_time = time.perf_counter() - start

        # Baseline: Graph construction + validation should be < 100ms
        assert graph_time < 0.100, f"Graph construction took {graph_time * 1000:.2f}ms (expected < 100ms)"

        print(f"\nGraph construction + validation: {graph_time * 1000:.2f}ms")

    finally:
        config_file.unlink()


@pytest.mark.performance
def test_end_to_end_validation_performance():
    """Measure end-to-end validation performance."""

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: json
    options:
      path: output.json
      schema:
        fields: dynamic
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        start = time.perf_counter()

        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
        graph.validate()

        total_time = time.perf_counter() - start

        # Baseline: End-to-end validation should be < 200ms
        assert total_time < 0.200, f"End-to-end validation took {total_time * 1000:.2f}ms (expected < 200ms)"

        print(f"\nEnd-to-end validation: {total_time * 1000:.2f}ms")

    finally:
        config_file.unlink()
