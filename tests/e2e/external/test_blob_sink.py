"""Azurite-backed E2E tests for the Azure Blob Storage sink plugin."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest

from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context

pytestmark = pytest.mark.e2e


def _blob_client(container_info: dict[str, str], blob_path: str):
    from azure.storage.blob import BlobServiceClient

    service_client = BlobServiceClient.from_connection_string(container_info["connection_string"])
    return service_client.get_container_client(container_info["container"]).get_blob_client(blob_path)


def _read_blob(container_info: dict[str, str], blob_path: str) -> bytes:
    return _blob_client(container_info, blob_path).download_blob().readall()


def _upload_blob(container_info: dict[str, str], blob_path: str, payload: bytes) -> None:
    _blob_client(container_info, blob_path).upload_blob(payload, overwrite=True)


def _sink_config(container_info: dict[str, str], *, blob_path: str, format_: str = "csv", **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "connection_string": container_info["connection_string"],
        "container": container_info["container"],
        "blob_path": blob_path,
        "format": format_,
        "schema": {"mode": "observed"},
        "overwrite": True,
    }
    config.update(overrides)
    return config


def _sink_context():
    return make_operation_context(
        operation_type="sink_write",
        node_id="azure-sink",
        node_type="SINK",
        plugin_name="azure_blob",
    )


class TestBlobSink:
    """Azure Blob Storage sink plugin E2E tests."""

    def test_blob_sink_writes_csv_data(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "outputs/customers.csv"
        sink = inject_write_failure(AzureBlobSink(_sink_config(azurite_blob_container, blob_path=blob_path, format_="csv")))

        result = sink.write([{"id": "1", "name": "Ada"}, {"id": "2", "name": "Grace"}], _sink_context())
        content = _read_blob(azurite_blob_container, blob_path).decode("utf-8")

        assert result.artifact.path_or_uri == f"azure://{azurite_blob_container['container']}/{blob_path}"
        assert result.artifact.size_bytes == len(content.encode("utf-8"))
        assert list(csv.DictReader(io.StringIO(content))) == [
            {"id": "1", "name": "Ada"},
            {"id": "2", "name": "Grace"},
        ]

    def test_blob_sink_writes_json_data(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "outputs/customers.json"
        sink = inject_write_failure(AzureBlobSink(_sink_config(azurite_blob_container, blob_path=blob_path, format_="json")))

        sink.write([{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}], _sink_context())

        assert json.loads(_read_blob(azurite_blob_container, blob_path)) == [
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Grace"},
        ]

    def test_blob_sink_overwrites_existing_blob(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "outputs/replace.csv"
        _upload_blob(azurite_blob_container, blob_path, b"old,content\n")
        sink = inject_write_failure(AzureBlobSink(_sink_config(azurite_blob_container, blob_path=blob_path, format_="csv")))

        sink.write([{"id": "1", "name": "Replacement"}], _sink_context())

        assert "Replacement" in _read_blob(azurite_blob_container, blob_path).decode("utf-8")

    def test_blob_sink_creates_new_blob(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "outputs/new.jsonl"
        sink = inject_write_failure(AzureBlobSink(_sink_config(azurite_blob_container, blob_path=blob_path, format_="jsonl")))

        result = sink.write([{"id": 1, "name": "Ada"}], _sink_context())

        assert result.artifact.path_or_uri == f"azure://{azurite_blob_container['container']}/{blob_path}"
        assert _read_blob(azurite_blob_container, blob_path).decode("utf-8") == '{"id": 1, "name": "Ada"}'
