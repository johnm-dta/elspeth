"""Integration coverage for CSV source proof diagnostics on blob-backed previews."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.blobs.service import content_hash
from elspeth.web.catalog.service import CatalogServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import execute_tool
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema

_HEADER_MISMATCH_CODE = "csv_source_blob_header_mismatch"
_HEADER_RESOLUTION_ERROR_CODE = "csv_source_field_resolution_error"


def _catalog() -> CatalogServiceImpl:
    manager = PluginManager()
    manager.register_builtin_plugins()
    return CatalogServiceImpl(manager)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _session_engine() -> tuple[Engine, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=session_id,
                user_id="test-user",
                auth_provider_type="local",
                title="CSV proof diagnostic test",
                created_at=now,
                updated_at=now,
            )
        )
    return engine, session_id


def _insert_blob(
    engine: Engine,
    session_id: str,
    tmp_path: Path,
    *,
    filename: str,
    mime_type: str,
    content: bytes,
) -> str:
    blob_id = str(uuid4())
    storage_dir = tmp_path / "blobs" / session_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{blob_id}_{filename}"
    storage_path.write_bytes(content)
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            blobs_table.insert().values(
                id=blob_id,
                session_id=session_id,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(content),
                content_hash=content_hash(content),
                storage_path=str(storage_path),
                created_at=now,
                created_by="user",
                source_description=None,
                status="ready",
            )
        )
    return blob_id


def _state_with_blob_source(
    engine: Engine,
    session_id: str,
    blob_id: str,
    *,
    plugin: str,
    options: dict[str, Any],
) -> CompositionState:
    catalog = _catalog()
    result = execute_tool(
        "set_source_from_blob",
        {
            "blob_id": blob_id,
            "plugin": plugin,
            "on_success": "out",
            "on_validation_failure": "discard",
            "options": options,
        },
        _empty_state(),
        catalog,
        session_engine=engine,
        session_id=session_id,
    )
    assert result.success is True, result.data

    result = execute_tool(
        "set_output",
        {
            "sink_name": "out",
            "plugin": "json",
            "options": {
                "path": "outputs/out.json",
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            "on_write_failure": "discard",
        },
        result.updated_state,
        catalog,
    )
    assert result.success is True, result.data
    return result.updated_state


def _preview_data(engine: Engine, session_id: str, state: CompositionState) -> dict[str, Any]:
    result = execute_tool(
        "preview_pipeline",
        {},
        state,
        _catalog(),
        session_engine=engine,
        session_id=session_id,
    )
    assert result.success is True, result.data
    return result.data


@pytest.mark.parametrize("schema_mode", ["fixed", "flexible"])
def test_csv_blob_without_header_and_no_declared_overlap_blocks(schema_mode: str, tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="web_pages.txt",
        mime_type="text/plain",
        content=b"https://www.finance.gov.au\nhttps://www.ato.gov.au\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={"schema": {"mode": schema_mode, "fields": ["url: str"]}},
    )

    data = _preview_data(engine, session_id, state)

    diagnostics = data["proof_diagnostics"]
    matching = [item for item in diagnostics if item["code"] == _HEADER_MISMATCH_CODE]
    assert matching
    assert matching[0]["severity"] == "blocking"
    diagnostic_blob = repr(matching[0])
    assert "https://www.finance.gov.au" not in diagnostic_blob
    assert "https://www.ato.gov.au" not in diagnostic_blob
    assert matching[0]["evidence_locator"]["observed_header_count"] == 1
    assert matching[0]["evidence_locator"]["observed_headers_redacted"] is True
    assert "observed_headers" not in matching[0]["evidence_locator"]
    assert data["is_valid"] is False


def test_csv_blob_with_matching_header_does_not_block(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="web_pages.csv",
        mime_type="text/csv",
        content=b"url\nhttps://www.finance.gov.au\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={"schema": {"mode": "fixed", "fields": ["url: str"]}},
    )

    data = _preview_data(engine, session_id, state)

    codes = [item["code"] for item in data["proof_diagnostics"]]
    assert _HEADER_MISMATCH_CODE not in codes
    assert data["is_valid"] is True


def test_csv_blob_with_normalized_header_does_not_block(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="customers.csv",
        mime_type="text/csv",
        content=b"Customer ID\n123\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={"schema": {"mode": "fixed", "fields": ["customer_id: str"]}},
    )

    data = _preview_data(engine, session_id, state)

    codes = [item["code"] for item in data["proof_diagnostics"]]
    assert _HEADER_MISMATCH_CODE not in codes
    assert data["is_valid"] is True


def test_csv_blob_with_field_mapping_header_does_not_block(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="customers.csv",
        mime_type="text/csv",
        content=b"External ID\n123\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={
            "field_mapping": {"external_id": "customer_id"},
            "schema": {"mode": "fixed", "fields": ["customer_id: str"]},
        },
    )

    data = _preview_data(engine, session_id, state)

    codes = [item["code"] for item in data["proof_diagnostics"]]
    assert _HEADER_MISMATCH_CODE not in codes
    assert data["is_valid"] is True


def test_csv_blob_with_normalization_collision_returns_blocking_diagnostic(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="customers.csv",
        mime_type="text/csv",
        content=b"Customer ID,customer_id\n123,456\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={"schema": {"mode": "fixed", "fields": ["customer_id: str"]}},
    )

    data = _preview_data(engine, session_id, state)

    matching = [item for item in data["proof_diagnostics"] if item["code"] == _HEADER_RESOLUTION_ERROR_CODE]
    assert matching
    assert matching[0]["severity"] == "blocking"
    assert "Field name collision" in matching[0]["message"]
    assert data["is_valid"] is False


def test_csv_blob_with_invalid_field_mapping_returns_blocking_diagnostic(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="customers.csv",
        mime_type="text/csv",
        content=b"External ID\n123\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={
            "field_mapping": {"missing_header": "customer_id"},
            "schema": {"mode": "fixed", "fields": ["customer_id: str"]},
        },
    )

    data = _preview_data(engine, session_id, state)

    matching = [item for item in data["proof_diagnostics"] if item["code"] == _HEADER_RESOLUTION_ERROR_CODE]
    assert matching
    assert matching[0]["severity"] == "blocking"
    assert "field_mapping keys not found" in matching[0]["message"]
    assert data["is_valid"] is False


def test_csv_blob_headerless_columns_mode_does_not_block(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="web_pages.txt",
        mime_type="text/plain",
        content=b"https://www.finance.gov.au\nhttps://www.ato.gov.au\n",
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="csv",
        options={
            "columns": ["url"],
            "schema": {"mode": "fixed", "fields": ["url: str"]},
        },
    )

    data = _preview_data(engine, session_id, state)

    codes = [item["code"] for item in data["proof_diagnostics"]]
    assert _HEADER_MISMATCH_CODE not in codes
    assert data["is_valid"] is True


def test_jsonl_blob_does_not_fire_csv_header_mismatch(tmp_path: Path) -> None:
    engine, session_id = _session_engine()
    blob_id = _insert_blob(
        engine,
        session_id,
        tmp_path,
        filename="web_pages.jsonl",
        mime_type="application/x-jsonlines",
        content=b'{"url": "https://www.finance.gov.au"}\n',
    )
    state = _state_with_blob_source(
        engine,
        session_id,
        blob_id,
        plugin="json",
        options={"schema": {"mode": "fixed", "fields": ["url: str"]}},
    )

    data = _preview_data(engine, session_id, state)

    codes = [item["code"] for item in data["proof_diagnostics"]]
    assert _HEADER_MISMATCH_CODE not in codes
    assert data["is_valid"] is True
