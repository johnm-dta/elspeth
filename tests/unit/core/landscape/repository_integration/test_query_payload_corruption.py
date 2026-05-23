"""Integration regressions for row payload corruption at query boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.payload_store import FilesystemPayloadStore

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _record_row_with_payload(tmp_path: Path) -> tuple[RecorderFactory, str, str, str, FilesystemPayloadStore]:
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    factory = RecorderFactory(LandscapeDB.in_memory(), payload_store=payload_store)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source.node_id,
        row_index=0,
        data={"field": "value", "number": 42},
    )

    assert row.source_data_ref is not None
    return factory, run.run_id, row.row_id, row.source_data_ref, payload_store


def test_corrupted_row_payload_raises_with_row_and_ref_context(tmp_path: Path) -> None:
    """Real payload-store corruption must crash both row-data read surfaces."""
    factory, run_id, row_id, source_data_ref, payload_store = _record_row_with_payload(tmp_path)

    payload_path = payload_store.base_path / source_data_ref[:2] / source_data_ref
    payload_path.write_bytes(b"tampered payload bytes")

    read_paths: dict[str, Callable[[], object]] = {
        "get_row_data": lambda: factory.query.get_row_data(row_id),
        "explain_row": lambda: factory.query.explain_row(run_id, row_id),
    }

    for read_path, read in read_paths.items():
        with pytest.raises(AuditIntegrityError) as exc_info:
            read()

        message = str(exc_info.value)
        assert "Payload integrity check failed" in message, read_path
        assert f"row {row_id}" in message, read_path
        assert f"ref={source_data_ref}" in message, read_path
