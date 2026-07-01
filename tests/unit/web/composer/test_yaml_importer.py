from __future__ import annotations

import pytest
import yaml

from elspeth.web.composer.yaml_generator import generate_yaml
from elspeth.web.composer.yaml_importer import RuntimeYamlImportError, composition_state_from_runtime_yaml


def test_composition_state_from_runtime_yaml_rejects_non_mapping_root() -> None:
    with pytest.raises(RuntimeYamlImportError, match="pipeline YAML must be a mapping"):
        composition_state_from_runtime_yaml("- not\n- a\n- mapping\n")


def test_composition_state_from_runtime_yaml_rejects_inline_blob_ref() -> None:
    with pytest.raises(RuntimeYamlImportError, match="blob_ref must be supplied via source_blob_ids"):
        composition_state_from_runtime_yaml(
            """
sources:
  source:
    plugin: csv
    on_success: out
    options:
      path: /data/blobs/input.csv
      blob_ref: 98b1357d-5aab-4fb3-85b4-5ad643912e84
sinks:
  out:
    plugin: csv
"""
        )


def test_composition_state_from_runtime_yaml_allows_terminal_aggregation() -> None:
    state = composition_state_from_runtime_yaml(
        """
sources:
  source:
    plugin: csv
    on_success: batch_in
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
aggregations:
  - name: batch
    plugin: batch_top_k
    input: batch_in
    on_error: discard
    options:
      field: category
sinks:
  out:
    plugin: csv
    on_write_failure: discard
"""
    )

    assert state.nodes[0].node_type == "aggregation"
    assert state.nodes[0].on_success is None


def test_composition_state_from_runtime_yaml_rejects_missing_transform_error_route() -> None:
    with pytest.raises(RuntimeYamlImportError, match=r"transforms\[0\]\.on_error"):
        composition_state_from_runtime_yaml(
            """
sources:
  source:
    plugin: csv
    on_success: transform_in
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
transforms:
  - name: normalize
    plugin: field_mapper
    input: transform_in
    on_success: out
    options:
      mapping:
        body: text
sinks:
  out:
    plugin: csv
    on_write_failure: discard
"""
        )


def test_composition_state_from_runtime_yaml_rejects_missing_aggregation_error_route() -> None:
    with pytest.raises(RuntimeYamlImportError, match=r"aggregations\[0\]\.on_error"):
        composition_state_from_runtime_yaml(
            """
sources:
  source:
    plugin: csv
    on_success: batch_in
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
aggregations:
  - name: batch
    plugin: batch_top_k
    input: batch_in
    on_success: out
    options:
      field: category
sinks:
  out:
    plugin: csv
    on_write_failure: discard
"""
        )


def test_composition_state_from_runtime_yaml_rejects_missing_sink_write_failure_route() -> None:
    with pytest.raises(RuntimeYamlImportError, match=r"sinks\.out\.on_write_failure"):
        composition_state_from_runtime_yaml(
            """
sources:
  source:
    plugin: csv
    on_success: out
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
sinks:
  out:
    plugin: csv
"""
        )


def test_composition_state_from_runtime_yaml_rejects_unpreservable_coalesce_fields() -> None:
    with pytest.raises(RuntimeYamlImportError, match="unsupported coalesce field"):
        composition_state_from_runtime_yaml(
            """
sources:
  source:
    plugin: csv
    on_success: a
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
coalesce:
  - name: joined
    branches:
      - a
      - b
    policy: quorum
    merge: nested
    quorum_count: 2
sinks:
  out:
    plugin: csv
"""
        )


def test_composition_state_from_runtime_yaml_round_trips_runtime_sections() -> None:
    pipeline_yaml = """
sources:
  source:
    plugin: csv
    on_success: gate_in
    options:
      path: /data/blobs/input.csv
      schema:
        mode: observed
      on_validation_failure: discard
transforms:
  - name: normalize
    plugin: field_mapper
    input: gate_in
    on_success: main
    on_error: discard
    options:
      mapping:
        body: text
      schema:
        mode: observed
gates:
  - name: split
    input: main
    condition: text != ''
    routes:
      true: accepted
      false: rejected
aggregations:
  - name: batch
    plugin: batch_top_k
    input: accepted
    on_success: batch_out
    on_error: discard
    trigger:
      count: 10
    options:
      field: category
coalesce:
  - name: joined
    branches:
      - batch_out
      - rejected
    policy: best_effort
    merge: nested
    on_success: audit
sinks:
  audit:
    plugin: json
    options:
      path: outputs/audit.json
    on_write_failure: discard
  rejected:
    plugin: csv
    options:
      path: outputs/rejected.csv
    on_write_failure: discard
"""

    state = composition_state_from_runtime_yaml(pipeline_yaml)
    exported = yaml.safe_load(generate_yaml(state))

    assert exported["sources"]["source"]["plugin"] == "csv"
    assert exported["transforms"][0]["name"] == "normalize"
    assert exported["gates"][0]["routes"] == {"true": "accepted", "false": "rejected"}
    assert exported["aggregations"][0]["trigger"] == {"count": 10}
    assert exported["coalesce"][0]["on_success"] == "audit"
    assert set(exported["sinks"]) == {"audit", "rejected"}
