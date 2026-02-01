"""Integration tests for Azure Blob source/sink using Azurite."""

from __future__ import annotations

import pytest

from elspeth.plugins.azure.blob_sink import AzureBlobSink
from elspeth.plugins.azure.blob_source import AzureBlobSource
from elspeth.plugins.context import PluginContext

DYNAMIC_SCHEMA = {"fields": "dynamic"}


@pytest.mark.integration
def test_blob_sink_source_roundtrip_with_azurite(azurite_blob_container) -> None:
    """AzureBlobSink writes to Azurite and AzureBlobSource reads back."""
    pytest.importorskip("azure.storage.blob")

    connection_string = azurite_blob_container["connection_string"]
    container = azurite_blob_container["container"]

    blob_path = "roundtrip/data.jsonl"
    rows = [
        {"id": 1, "value": "alpha"},
        {"id": 2, "value": "beta"},
        {"id": 3, "value": "gamma"},
    ]

    ctx = PluginContext(run_id="test-run-azurite", config={})

    sink = AzureBlobSink(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "jsonl",
            "overwrite": True,
            "schema": DYNAMIC_SCHEMA,
        }
    )

    sink.write(rows, ctx)

    source = AzureBlobSource(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "jsonl",
            "on_validation_failure": "discard",
            "schema": DYNAMIC_SCHEMA,
        }
    )

    loaded = list(source.load(ctx))
    loaded_rows = [item.row for item in loaded if not item.is_quarantined]

    assert loaded_rows == rows


@pytest.mark.integration
def test_blob_sink_source_roundtrip_json_array(azurite_blob_container) -> None:
    """AzureBlobSink JSON array format round-trips through Azurite."""
    pytest.importorskip("azure.storage.blob")

    connection_string = azurite_blob_container["connection_string"]
    container = azurite_blob_container["container"]

    blob_path = "roundtrip/data.json"
    rows = [
        {"id": 10, "value": 1},
        {"id": 20, "value": 2},
        {"id": 30, "value": 3},
    ]

    ctx = PluginContext(run_id="test-run-azurite-json", config={})

    sink = AzureBlobSink(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "json",
            "overwrite": True,
            "schema": DYNAMIC_SCHEMA,
        }
    )
    sink.write(rows, ctx)

    source = AzureBlobSource(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "json",
            "on_validation_failure": "discard",
            "schema": DYNAMIC_SCHEMA,
        }
    )

    loaded = list(source.load(ctx))
    loaded_rows = [item.row for item in loaded if not item.is_quarantined]

    assert loaded_rows == rows


@pytest.mark.integration
def test_blob_sink_source_roundtrip_csv(azurite_blob_container) -> None:
    """AzureBlobSink CSV format round-trips through Azurite."""
    pytest.importorskip("azure.storage.blob")

    connection_string = azurite_blob_container["connection_string"]
    container = azurite_blob_container["container"]

    blob_path = "roundtrip/data.csv"
    rows = [
        {"id": "1", "value": "alpha"},
        {"id": "2", "value": "beta"},
        {"id": "3", "value": "gamma"},
    ]

    ctx = PluginContext(run_id="test-run-azurite-csv", config={})

    sink = AzureBlobSink(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "csv",
            "overwrite": True,
            "schema": DYNAMIC_SCHEMA,
        }
    )
    sink.write(rows, ctx)

    source = AzureBlobSource(
        {
            "connection_string": connection_string,
            "container": container,
            "blob_path": blob_path,
            "format": "csv",
            "on_validation_failure": "discard",
            "schema": DYNAMIC_SCHEMA,
        }
    )

    loaded = list(source.load(ctx))
    loaded_rows = [item.row for item in loaded if not item.is_quarantined]

    assert loaded_rows == rows
