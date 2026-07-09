"""Run-level field-resolution record must carry the cross-row union.

Regression (elspeth-fb108a77c9): the main processing loop recorded the
run-level source field resolution once, after the first row, and the EOF
finalizer was gated on "not yet recorded" — so sparse JSON/JSONL rows that
introduced new raw keys after row 1 never reached the run-level
``runs.source_field_resolution_json`` column that ``headers: original``
restoration and audit-recovery read.
"""

from __future__ import annotations

from pathlib import Path

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from tests.fixtures.landscape import make_factory


def test_run_level_field_resolution_includes_fields_first_seen_after_row_one(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.jsonl"
    # Row 2 introduces "Extra Field", whose original name differs from its
    # normalized form — the exact shape the run-level record used to drop.
    input_path.write_text('{"id": 1, "Name Field": "alice"}\n{"id": 2, "Extra Field": "bonus"}\n')

    settings = load_settings_from_yaml_string(
        f"""
sources:
  rows:
    plugin: json
    on_success: inbound
    options:
      path: {input_path}
      format: jsonl
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound: {{}}
transforms:
  - name: pass_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
"""
    )
    bundle = instantiate_plugins_from_config(settings)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        queues=settings.queues,
    )
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=settings,
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )
    assert result.rows_processed == 2

    resolution = make_factory(db).run_lifecycle.get_source_field_resolution(result.run_id)
    assert resolution is not None
    assert resolution.get("Name Field") == "name_field", f"row-1 field missing from run-level record: {resolution}"
    assert resolution.get("Extra Field") == "extra_field", f"field first seen on row 2 missing from run-level record: {resolution}"
