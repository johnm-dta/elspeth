from __future__ import annotations

import pytest
import yaml

from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.composer.yaml_generator import generate_public_yaml, generate_yaml
from elspeth.web.composer.yaml_importer import (
    MAX_RUNTIME_YAML_IMPORT_CHARS,
    RuntimeYamlImportError,
    _nodes_from_runtime_list,
    _outputs_from_runtime_sinks,
    _require_str,
    _source_from_runtime_entry,
    composition_state_from_runtime_yaml,
)


def test_require_str_rejects_non_string_value() -> None:
    with pytest.raises(RuntimeYamlImportError, match=r"sources\.s\.plugin must be a non-empty string"):
        _require_str({"plugin": 7}, "plugin", "sources.s")


def test_source_from_runtime_entry_rejects_non_mapping_entry() -> None:
    with pytest.raises(RuntimeYamlImportError, match=r"sources\.s must be a mapping"):
        _source_from_runtime_entry("s", ["not", "a", "mapping"])


def test_nodes_from_runtime_list_rejects_non_sequence_section() -> None:
    with pytest.raises(RuntimeYamlImportError, match="transforms must be a list"):
        _nodes_from_runtime_list("not-a-list", "transforms", "transform")


def test_outputs_from_runtime_sinks_rejects_non_mapping_sinks() -> None:
    with pytest.raises(RuntimeYamlImportError, match="sinks must be a mapping"):
        _outputs_from_runtime_sinks(["not", "a", "mapping"])


def test_composition_state_from_runtime_yaml_rejects_non_mapping_root() -> None:
    with pytest.raises(RuntimeYamlImportError, match="pipeline YAML must be a mapping"):
        composition_state_from_runtime_yaml("- not\n- a\n- mapping\n")


def test_composition_state_from_runtime_yaml_rejects_oversized_document() -> None:
    """Hardening: a paste over the character cap is rejected before parsing."""
    oversized = "sources:\n  source:\n    plugin: csv\n" + ("#" * (MAX_RUNTIME_YAML_IMPORT_CHARS + 1))
    with pytest.raises(RuntimeYamlImportError, match="exceeds the 262144 character import limit"):
        composition_state_from_runtime_yaml(oversized)


def test_composition_state_from_runtime_yaml_rejects_malformed_yaml_syntax() -> None:
    """Hardening: genuinely non-YAML input is a categorized error, not a raw parser echo."""
    not_yaml = "sources: [unterminated\n  plugin: csv"
    with pytest.raises(RuntimeYamlImportError, match=r"^YAML parse failed: \w+Error$") as exc_info:
        composition_state_from_runtime_yaml(not_yaml)
    # Egress discipline: the error must name the exception class only, never
    # echo the pasted content back to the caller.
    assert "unterminated" not in str(exc_info.value)
    assert "plugin" not in str(exc_info.value)


def test_composition_state_from_runtime_yaml_rejects_non_pipeline_mapping() -> None:
    """Hardening: a valid YAML mapping that describes no pipeline section at all
    must not silently import as an empty (destructive-replace) composition."""
    not_a_pipeline = "shopping_list:\n  - milk\n  - eggs\nnotes: just some random yaml\n"
    with pytest.raises(RuntimeYamlImportError, match="must define at least one pipeline section"):
        composition_state_from_runtime_yaml(not_a_pipeline)


def test_composition_state_from_runtime_yaml_rejects_empty_mapping() -> None:
    with pytest.raises(RuntimeYamlImportError, match="must define at least one pipeline section"):
        composition_state_from_runtime_yaml("{}\n")


def test_composition_state_from_runtime_yaml_rejects_aliases() -> None:
    """Hardening: anchors/aliases are rejected outright (billion-laughs defense).

    A small document can otherwise compose into a vastly larger logical
    structure via alias object-sharing once something downstream (dict()
    copies, state.to_dict(), JSON persistence) walks every reference.
    """
    aliased = """
sources:
  source: &src
    plugin: csv
    on_success: out
    options:
      path: /data/blobs/input.csv
      on_validation_failure: discard
sinks:
  out:
    plugin: csv
    on_write_failure: discard
also: *src
"""
    with pytest.raises(RuntimeYamlImportError, match=r"^YAML parse failed: \w+Error$"):
        composition_state_from_runtime_yaml(aliased)


def test_composition_state_from_runtime_yaml_rejects_deeply_nested_document() -> None:
    """Hardening: deep-but-textually-small nesting must not crash with RecursionError.

    ~500 levels of ``a:\\n  a:\\n    a:\\n ...`` fits comfortably under
    MAX_RUNTIME_YAML_IMPORT_CHARS but exhausts CPython's default recursion
    limit inside PyYAML's pure-Python composer/constructor if uncaught.
    """
    depth = 500
    lines = ["  " * i + "a:" for i in range(depth)]
    lines.append("  " * depth + "1")
    deeply_nested = "\n".join(lines)
    assert len(deeply_nested) < MAX_RUNTIME_YAML_IMPORT_CHARS
    with pytest.raises(RuntimeYamlImportError, match=r"^YAML parse failed: \w+Error$"):
        composition_state_from_runtime_yaml(deeply_nested)


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


def test_composition_state_from_runtime_yaml_reimports_generate_public_yaml_output() -> None:
    """Hardening regression guard: T-1's whole point is the export -> import
    round trip, so the hardening added in this pass (no-alias pre-scan,
    recursion-safety, non-pipeline-mapping gate) must not break re-importing
    what the composer's own public export (``generate_public_yaml``, the
    function ``GET /state/yaml`` actually calls) produces.

    Also empirically confirms ``generate_public_yaml`` never emits YAML
    anchors/aliases for a representative multi-node/multi-sink state --
    if it ever did (e.g. from a future change that shares an options dict
    object across nodes), the no-alias pre-scan added in this pass would
    reject the composer's own export, and this test would catch it.
    """
    state = CompositionState(
        version=1,
        sources={
            "source": SourceSpec(
                plugin="csv",
                on_success="gate_in",
                options={"path": "/data/blobs/input.csv"},
                on_validation_failure="discard",
            ),
        },
        nodes=(
            NodeSpec(
                id="normalize",
                node_type="transform",
                plugin="field_mapper",
                input="gate_in",
                on_success="main",
                on_error="discard",
                options={"mapping": {"body": "text"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="split",
                node_type="gate",
                plugin=None,
                input="main",
                on_success=None,
                on_error=None,
                options={},
                condition="text != ''",
                routes={"true": "accepted", "false": "rejected"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="batch",
                node_type="aggregation",
                plugin="batch_top_k",
                input="accepted",
                on_success="batch_out",
                on_error="discard",
                options={"field": "category"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="audit", plugin="json", options={"path": "outputs/audit.json"}, on_write_failure="discard"),
            OutputSpec(name="rejected", plugin="csv", options={"path": "outputs/rejected.csv"}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(name="Round trip test", description=""),
    )

    exported_yaml = generate_public_yaml(state)

    # No anchor/alias markers -- confirms the no-alias pre-scan's rejection
    # of aliases (added for the billion-laughs defense) has no cost against
    # the composer's own export shape.
    assert "&" not in exported_yaml.split("\n")[0:1][0]  # sanity: not YAML-document-start noise
    assert not any(line.strip().split(":")[-1].strip().startswith(("&", "*")) for line in exported_yaml.splitlines() if ":" in line)

    reimported = composition_state_from_runtime_yaml(exported_yaml)
    assert set(reimported.sources) == {"source"}
    assert [n.id for n in reimported.nodes] == ["normalize", "split", "batch"]
    assert {o.name for o in reimported.outputs} == {"audit", "rejected"}
