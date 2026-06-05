"""Azurite-backed E2E tests for the Azure Blob Storage source plugin."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.plugins.sources.azure_blob_source import AzureBlobSource
from tests.fixtures.factories import make_operation_context

pytestmark = pytest.mark.e2e


def _upload_blob(container_info: dict[str, str], blob_path: str, payload: bytes) -> None:
    from azure.storage.blob import BlobServiceClient

    service_client = BlobServiceClient.from_connection_string(container_info["connection_string"])
    blob_client = service_client.get_container_client(container_info["container"]).get_blob_client(blob_path)
    blob_client.upload_blob(payload, overwrite=True)


def _source_config(container_info: dict[str, str], *, blob_path: str, format_: str = "csv", **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "connection_string": container_info["connection_string"],
        "container": container_info["container"],
        "blob_path": blob_path,
        "format": format_,
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
    }
    config.update(overrides)
    return config


def _source_context():
    return make_operation_context(
        operation_type="source_load",
        node_id="azure-source",
        node_type="SOURCE",
        plugin_name="azure_blob",
    )


class TestBlobSource:
    """Azure Blob Storage source plugin E2E tests."""

    def test_blob_source_reads_csv_data(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "inputs/customers.csv"
        _upload_blob(azurite_blob_container, blob_path, b"id,name\n1,Ada\n2,Grace\n")

        source = AzureBlobSource(_source_config(azurite_blob_container, blob_path=blob_path, format_="csv"))
        rows = list(source.load(_source_context()))

        assert [row.row for row in rows] == [
            {"id": "1", "name": "Ada"},
            {"id": "2", "name": "Grace"},
        ]
        assert not any(row.is_quarantined for row in rows)

    def test_blob_source_reads_json_data(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "inputs/customers.json"
        _upload_blob(azurite_blob_container, blob_path, b'[{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]')

        source = AzureBlobSource(_source_config(azurite_blob_container, blob_path=blob_path, format_="json"))
        rows = list(source.load(_source_context()))

        assert [row.row for row in rows] == [
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Grace"},
        ]
        assert not any(row.is_quarantined for row in rows)

    def test_blob_source_handles_missing_blob(self, azurite_blob_container: dict[str, str]) -> None:
        source = AzureBlobSource(
            _source_config(
                azurite_blob_container,
                blob_path="inputs/missing.csv",
                format_="csv",
            )
        )

        with pytest.raises(RuntimeError, match=r"Failed to download blob 'inputs/missing\.csv'"):
            list(source.load(_source_context()))

    def test_blob_source_handles_empty_blob(self, azurite_blob_container: dict[str, str]) -> None:
        blob_path = "inputs/empty.csv"
        _upload_blob(azurite_blob_container, blob_path, b"")

        source = AzureBlobSource(_source_config(azurite_blob_container, blob_path=blob_path, format_="csv"))
        rows = list(source.load(_source_context()))

        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_destination == "quarantine"
        assert "empty file contains no header row" in (rows[0].quarantine_error or "")
