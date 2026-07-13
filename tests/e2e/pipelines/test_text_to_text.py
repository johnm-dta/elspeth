"""E2E: text source -> passthrough -> text sink roundtrip."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from elspeth.contracts import RunStatus
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import token_outcomes_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.text_sink import TextSink
from elspeth.plugins.sources.text_source import TextSource
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import PassTransform


def test_text_source_to_text_sink_exact_roundtrip_and_terminal_outcomes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    expected = b" alpha \n\nomega\n"
    input_path.write_bytes(expected)

    source = TextSource(
        {
            "path": str(input_path),
            "column": "line_text",
            "encoding": "utf-8",
            "strip_whitespace": False,
            "skip_blank_lines": False,
            "schema": {"mode": "observed"},
            "on_validation_failure": "discard",
        }
    )
    source.on_success = "text_rows"
    source_settings = SourceSettings(plugin="text", on_success="text_rows", options={})
    transform = PassTransform()
    wired = wire_transforms([transform], source_connection="text_rows", final_sink="default")
    sink = TextSink(
        {
            "path": str(output_path),
            "field": "line_text",
            "encoding": "utf-8",
            "schema": {"mode": "observed"},
        }
    )
    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": source},
        source_settings_map={"primary": source_settings},
        transforms=wired,
        sinks={"default": sink},
        aggregations={},
        gates=[],
    )
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    orchestrator = Orchestrator(db)
    result = orchestrator.run(
        PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        ),
        graph=graph,
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )

    assert result.status == RunStatus.COMPLETED
    assert result.rows_processed == 3
    assert result.rows_succeeded == 3
    assert output_path.read_bytes() == expected
    with db.engine.connect() as connection:
        terminal_count = connection.execute(
            select(func.count())
            .select_from(token_outcomes_table)
            .where(token_outcomes_table.c.run_id == result.run_id, token_outcomes_table.c.completed == 1)
        ).scalar_one()
    assert terminal_count == 3
