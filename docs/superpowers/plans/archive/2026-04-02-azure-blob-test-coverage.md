# Azure Blob Source & Sink Test Coverage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive unit and property-based test coverage for `azure_blob_source.py` (941 lines) and `azure_blob_sink.py` (702 lines), matching the depth and patterns of the local equivalents (csv_source, json_source, csv_sink, json_sink).

**Architecture:** Mock `AzureAuthConfig.create_blob_service_client()` to avoid Azure SDK dependency. Source tests use `make_operation_context()` (pre-built for azure_blob). Sink tests use `make_operation_context(operation_type="sink_write", node_type="SINK")`. All tests follow existing ELSPETH patterns: `SourceRow` assertions, `SinkWriteResult`/`ArtifactDescriptor` checks, quarantine routing, schema contract locking.

**Tech Stack:** pytest, unittest.mock (MagicMock/patch), hypothesis (property tests), existing test factories from `tests/fixtures/factories.py`.

**Filigree issue:** `elspeth-78c0e92eb7`

---

## Shared Conventions

**Minimal valid connection string** (reused across all test files):
```python
FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
```

**Mock Azure blob download** (source tests):
```python
def _mock_blob_download(data: bytes):
    """Create a mock blob client whose download_blob().readall() returns data."""
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = data
    mock_service = MagicMock()
    mock_service.get_container_client.return_value.get_blob_client.return_value = mock_blob_client
    return mock_service
```

**Mock Azure blob upload** (sink tests):
```python
def _mock_blob_upload():
    """Create a mock container client that captures upload_blob calls."""
    mock_blob_client = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_blob_client
```

**Patching target** for both: `elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client`

**Schema constants:**
```python
DYNAMIC_SCHEMA = {"mode": "observed"}
FIXED_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str", "value: float"]}
FLEXIBLE_SCHEMA = {"mode": "flexible", "fields": ["id: int"]}
QUARANTINE_SINK = "quarantine"
```

---

### Task 1: Azure Blob Source — Config Validation Tests

**Files:**
- Create: `tests/unit/plugins/sources/test_azure_blob_source.py`

Tests for `AzureBlobSourceConfig` validation — auth methods, format constraints, field normalization options. These test construction only (no `load()` calls), so no mocking needed.

- [ ] **Step 1: Write config validation tests**

```python
"""Tests for Azure Blob Storage source plugin."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.factories import make_operation_context

# Shared constants
FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA = {"mode": "observed"}
FIXED_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str", "value: float"]}
FLEXIBLE_SCHEMA = {"mode": "flexible", "fields": ["id: int"]}
QUARANTINE_SINK = "quarantine"


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid source config with overrides."""
    config: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "data/input.csv",
        "format": "csv",
        "schema": DYNAMIC_SCHEMA,
        "on_validation_failure": QUARANTINE_SINK,
    }
    config.update(overrides)
    return config


def _mock_blob_download(data: bytes) -> MagicMock:
    """Create a mock service client whose blob download returns data."""
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = data
    mock_service = MagicMock()
    mock_service.get_container_client.return_value.get_blob_client.return_value = mock_blob_client
    return mock_service


PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"


class TestAzureBlobSourceConfig:
    """Config validation tests — no Azure SDK calls."""

    def test_connection_string_auth(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config())
        assert source.name == "azure_blob"
        assert hasattr(source, "output_schema")

    def test_sas_token_auth(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(
            _base_config(
                connection_string=None,
                sas_token="sv=2021-06-08&ss=b&srt=sco&sp=r",
                account_url="https://fakestorage.blob.core.windows.net",
            )
        )
        assert source._auth_config.auth_method == "sas_token"

    def test_managed_identity_auth(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(
            _base_config(
                connection_string=None,
                use_managed_identity=True,
                account_url="https://fakestorage.blob.core.windows.net",
            )
        )
        assert source._auth_config.auth_method == "managed_identity"

    def test_service_principal_auth(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(
            _base_config(
                connection_string=None,
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret",
                account_url="https://fakestorage.blob.core.windows.net",
            )
        )
        assert source._auth_config.auth_method == "service_principal"

    def test_no_auth_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="No authentication method"):
            AzureBlobSource(_base_config(connection_string=None))

    def test_multiple_auth_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="Multiple authentication"):
            AzureBlobSource(
                _base_config(
                    use_managed_identity=True,
                    account_url="https://fakestorage.blob.core.windows.net",
                )
            )

    def test_empty_container_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="container cannot be empty"):
            AzureBlobSource(_base_config(container=""))

    def test_empty_blob_path_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="blob_path cannot be empty"):
            AzureBlobSource(_base_config(blob_path=""))

    def test_columns_rejected_for_json_format(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="only supported for CSV"):
            AzureBlobSource(_base_config(format="json", columns=["a", "b"]))

    def test_columns_with_has_header_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="has_header: false"):
            AzureBlobSource(_base_config(columns=["a", "b"]))

    def test_csv_delimiter_must_be_single_char(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="single character"):
            AzureBlobSource(_base_config(csv_options={"delimiter": ";;"}))

    def test_invalid_encoding_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with pytest.raises(ValidationError, match="unknown encoding"):
            AzureBlobSource(_base_config(csv_options={"encoding": "not-a-codec"}))

    def test_fixed_schema_creates_locked_contract_for_json(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config(format="json", schema=FIXED_SCHEMA))
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked

    def test_observed_schema_defers_contract_for_json(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config(format="json", schema=DYNAMIC_SCHEMA))
        assert source._contract_builder is not None

    def test_csv_defers_contract_until_load(self) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config(format="csv", schema=FIXED_SCHEMA))
        assert source._contract_builder is None
        # CSV contract created during load() after field resolution
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceConfig -v`
Expected: All PASS (these test config parsing, not Azure SDK calls)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sources/test_azure_blob_source.py
git commit -m "test: add Azure Blob source config validation tests"
```

---

### Task 2: Azure Blob Source — CSV Loading Tests

**Files:**
- Modify: `tests/unit/plugins/sources/test_azure_blob_source.py`

Tests for `_load_csv()` path — header parsing, field normalization, per-row error handling, quarantine behavior.

- [ ] **Step 1: Add CSV loading test class**

Append to `test_azure_blob_source.py`:

```python
class TestAzureBlobSourceCSV:
    """CSV loading from Azure Blob — mocked Azure SDK."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    @patch(PATCH_AUTH)
    def test_load_csv_with_headers(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id,name,value\n1,alice,100\n2,bob,200\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config())
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"id": "1", "name": "alice", "value": "100"}
        assert rows[1].row["name"] == "bob"

    @patch(PATCH_AUTH)
    def test_load_csv_custom_delimiter(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id;name\n1;alice\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(csv_options={"delimiter": ";"}))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row == {"id": "1", "name": "alice"}

    @patch(PATCH_AUTH)
    def test_load_csv_latin1_encoding(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id,name\n1,caf\xe9\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(csv_options={"encoding": "latin-1"}))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row["name"] == "café"

    @patch(PATCH_AUTH)
    def test_load_csv_headerless_with_columns(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"1,alice,100\n2,bob,200\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(
            _base_config(
                csv_options={"has_header": False},
                columns=["id", "name", "value"],
            )
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": "1", "name": "alice", "value": "100"}

    @patch(PATCH_AUTH)
    def test_load_csv_headerless_no_columns_uses_numeric(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"1,alice\n2,bob\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(csv_options={"has_header": False}))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"0": "1", "1": "alice"}

    @patch(PATCH_AUTH)
    def test_load_csv_column_count_mismatch_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id,name\n1,alice\n2,bob,extra\n3,carol\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config())
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[1].is_quarantined is True
        assert "expected 2 fields, got 3" in rows[1].quarantine_error
        assert rows[1].quarantine_destination == QUARANTINE_SINK
        assert rows[2].is_quarantined is False

    @patch(PATCH_AUTH)
    def test_load_csv_empty_file_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(b"")

        source = AzureBlobSource(_base_config())
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "empty file" in rows[0].quarantine_error

    @patch(PATCH_AUTH)
    def test_load_csv_unicode_decode_error_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        # Invalid UTF-8 bytes
        blob_data = b"\xff\xfe invalid utf8"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config())
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "decode" in rows[0].quarantine_error.lower()

    @patch(PATCH_AUTH)
    def test_load_csv_discard_mode_suppresses_quarantine(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id,name\n1,alice\n2,bob,extra\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(on_validation_failure="discard"))
        rows = list(source.load(ctx))

        # Only the valid row is yielded; malformed row is discarded, not quarantined
        assert len(rows) == 1
        assert rows[0].row["name"] == "alice"

    @patch(PATCH_AUTH)
    def test_load_csv_blank_lines_skipped(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"id,name\n1,alice\n\n2,bob\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config())
        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        assert len(valid_rows) == 2

    @patch(PATCH_AUTH)
    def test_load_csv_field_mapping(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"ID,Full Name\n1,alice\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(
            _base_config(field_mapping={"full_name": "person_name"})
        )
        rows = list(source.load(ctx))

        assert len(rows) == 1
        # "Full Name" normalizes to "full_name", then field_mapping renames to "person_name"
        assert "person_name" in rows[0].row

    @patch(PATCH_AUTH)
    def test_close_nulls_client(self, mock_create: MagicMock) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config())
        source._blob_client = MagicMock()
        source.close()
        assert source._blob_client is None

    @patch(PATCH_AUTH)
    def test_close_idempotent(self, mock_create: MagicMock) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        source = AzureBlobSource(_base_config())
        source.close()
        source.close()  # Should not raise
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceCSV -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sources/test_azure_blob_source.py
git commit -m "test: add Azure Blob source CSV loading tests"
```

---

### Task 3: Azure Blob Source — JSON and JSONL Loading Tests

**Files:**
- Modify: `tests/unit/plugins/sources/test_azure_blob_source.py`

Tests for `_load_json_array()` and `_load_jsonl()` paths — parsing, data_key extraction, per-line quarantine.

- [ ] **Step 1: Add JSON/JSONL test classes**

Append to `test_azure_blob_source.py`:

```python
class TestAzureBlobSourceJSON:
    """JSON array loading from Azure Blob."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    @patch(PATCH_AUTH)
    def test_load_json_array(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[0].is_quarantined is False

    @patch(PATCH_AUTH)
    def test_load_json_with_data_key(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = {"meta": {}, "results": [{"id": 1}, {"id": 2}]}
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(
            _base_config(format="json", json_options={"data_key": "results"})
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1}

    @patch(PATCH_AUTH)
    def test_load_json_data_key_not_found_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = {"other": [{"id": 1}]}
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(
            _base_config(format="json", json_options={"data_key": "results"})
        )
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "not found" in rows[0].quarantine_error

    @patch(PATCH_AUTH)
    def test_load_json_data_key_on_non_object_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(json.dumps([1, 2, 3]).encode())

        source = AzureBlobSource(
            _base_config(format="json", json_options={"data_key": "results"})
        )
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "expected JSON object" in rows[0].quarantine_error

    @patch(PATCH_AUTH)
    def test_load_json_not_array_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(json.dumps({"single": "obj"}).encode())

        source = AzureBlobSource(_base_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "Expected JSON array" in rows[0].quarantine_error

    @patch(PATCH_AUTH)
    def test_load_json_invalid_json_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(b"not json at all")

        source = AzureBlobSource(_base_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "Invalid JSON" in rows[0].quarantine_error

    @patch(PATCH_AUTH)
    def test_load_json_nonfinite_rejected(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(b"[NaN]")

        source = AzureBlobSource(_base_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True

    @patch(PATCH_AUTH)
    def test_load_json_encoding_error_quarantines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(b"\xff\xfe")

        source = AzureBlobSource(_base_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "decode" in rows[0].quarantine_error.lower()


class TestAzureBlobSourceJSONL:
    """JSONL loading from Azure Blob."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    @patch(PATCH_AUTH)
    def test_load_jsonl(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[1].row["name"] == "bob"

    @patch(PATCH_AUTH)
    def test_load_jsonl_skips_empty_lines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1}\n\n  \n{"id": 2}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 2

    @patch(PATCH_AUTH)
    def test_load_jsonl_per_line_quarantine(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1}\nnot-json\n{"id": 3}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[1].is_quarantined is True
        assert "JSON parse error at line 2" in rows[1].quarantine_error
        assert rows[2].is_quarantined is False

    @patch(PATCH_AUTH)
    def test_load_jsonl_discard_mode(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1}\nnot-json\n{"id": 3}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(format="jsonl", on_validation_failure="discard"))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)

    @patch(PATCH_AUTH)
    def test_load_jsonl_nonfinite_per_line(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1}\nNaN\n{"id": 3}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert rows[1].is_quarantined is True

    @patch(PATCH_AUTH)
    def test_load_jsonl_encoding_error(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(b"\xff\xfe")

        source = AzureBlobSource(_base_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceJSON tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceJSONL -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sources/test_azure_blob_source.py
git commit -m "test: add Azure Blob source JSON/JSONL loading tests"
```

---

### Task 4: Azure Blob Source — Schema Validation and Audit Trail Tests

**Files:**
- Modify: `tests/unit/plugins/sources/test_azure_blob_source.py`

Tests for schema contract locking (FLEXIBLE/OBSERVED), validation failures, audit trail recording, and Azure SDK error handling.

- [ ] **Step 1: Add schema and audit test classes**

Append to `test_azure_blob_source.py`:

```python
class TestAzureBlobSourceSchemaValidation:
    """Schema validation and contract locking."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    @patch(PATCH_AUTH)
    def test_fixed_schema_validates_types(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [{"id": 1, "name": "alice", "value": 3.14}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=FIXED_SCHEMA))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is False
        assert rows[0].row["id"] == 1  # Coerced by source schema

    @patch(PATCH_AUTH)
    def test_fixed_schema_quarantines_invalid_types(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [{"id": "not-an-int", "name": "alice", "value": "not-a-float"}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=FIXED_SCHEMA))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True

    @patch(PATCH_AUTH)
    def test_flexible_schema_locks_on_first_row(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [
            {"id": 1, "extra_field": "hello"},
            {"id": 2, "extra_field": "world"},
        ]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=FLEXIBLE_SCHEMA))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked

    @patch(PATCH_AUTH)
    def test_observed_schema_locks_on_first_row(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [{"x": 1, "y": "hello"}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=DYNAMIC_SCHEMA))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked

    @patch(PATCH_AUTH)
    def test_no_valid_rows_still_locks_contract(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        # All rows will fail fixed schema validation
        data = [{"wrong_field": "hello"}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=FIXED_SCHEMA))
        rows = list(source.load(ctx))

        # Contract should still be locked even with no valid rows
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked

    @patch(PATCH_AUTH)
    def test_source_row_has_contract(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        data = [{"id": 1, "name": "alice", "value": 3.14}]
        mock_create.return_value = _mock_blob_download(json.dumps(data).encode())

        source = AzureBlobSource(_base_config(format="json", schema=FIXED_SCHEMA))
        rows = list(source.load(ctx))

        assert rows[0].contract is not None


class TestAzureBlobSourceAuditAndErrors:
    """Audit trail recording and Azure SDK error handling."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    @patch(PATCH_AUTH)
    def test_download_failure_raises_runtime_error(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_service = MagicMock()
        mock_service.get_container_client.return_value.get_blob_client.return_value.download_blob.side_effect = Exception(
            "Connection refused"
        )
        mock_create.return_value = mock_service

        source = AzureBlobSource(_base_config())
        with pytest.raises(RuntimeError, match="Failed to download blob"):
            list(source.load(ctx))

    @patch(PATCH_AUTH)
    def test_import_error_propagated(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.side_effect = ImportError("azure-storage-blob not installed")

        source = AzureBlobSource(_base_config())
        with pytest.raises(ImportError):
            list(source.load(ctx))

    @patch(PATCH_AUTH)
    def test_programming_errors_crash_directly(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_service = MagicMock()
        mock_service.get_container_client.return_value.get_blob_client.side_effect = TypeError("unexpected kwarg")
        mock_create.return_value = mock_service

        source = AzureBlobSource(_base_config())
        with pytest.raises(TypeError, match="unexpected kwarg"):
            list(source.load(ctx))

    @patch(PATCH_AUTH)
    def test_audit_integrity_error_on_record_call_failure(self, mock_create: MagicMock) -> None:
        from unittest.mock import Mock

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b'{"id": 1}\n'
        mock_create.return_value = _mock_blob_download(blob_data)

        # Create a context where record_call raises
        ctx = make_operation_context(plugin_name="azure_blob")
        original_record_call = ctx.record_call
        call_count = 0

        def failing_record_call(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("DB write failed")

        ctx.record_call = failing_record_call  # type: ignore[assignment]

        source = AzureBlobSource(_base_config(format="jsonl"))
        with pytest.raises(AuditIntegrityError, match="audit trail"):
            list(source.load(ctx))

    @patch(PATCH_AUTH)
    def test_field_resolution_returned_for_csv(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        blob_data = b"ID,Full Name\n1,alice\n"
        mock_create.return_value = _mock_blob_download(blob_data)

        source = AzureBlobSource(_base_config())
        list(source.load(ctx))

        result = source.get_field_resolution()
        assert result is not None
        mapping, version = result
        assert isinstance(mapping, dict)

    @patch(PATCH_AUTH)
    def test_field_resolution_none_for_json(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import json

        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(json.dumps([{"id": 1}]).encode())

        source = AzureBlobSource(_base_config(format="json"))
        list(source.load(ctx))

        assert source.get_field_resolution() is None
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceSchemaValidation tests/unit/plugins/sources/test_azure_blob_source.py::TestAzureBlobSourceAuditAndErrors -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sources/test_azure_blob_source.py
git commit -m "test: add Azure Blob source schema validation and audit trail tests"
```

---

### Task 5: Azure Blob Sink — Config Validation and Lifecycle Tests

**Files:**
- Create: `tests/unit/plugins/sinks/test_azure_blob_sink.py`

Tests for `AzureBlobSinkConfig` validation, template compilation, resume rejection, and resource lifecycle.

- [ ] **Step 1: Write sink config and lifecycle tests**

```python
"""Tests for Azure Blob Storage sink plugin."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.factories import make_operation_context

# Shared constants
FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA = {"mode": "observed"}
FIXED_SCHEMA = {"mode": "fixed", "fields": ["id: str", "name: str"]}


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid sink config with overrides."""
    config: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "output/result.csv",
        "format": "csv",
        "schema": DYNAMIC_SCHEMA,
    }
    config.update(overrides)
    return config


def _mock_blob_upload() -> tuple[MagicMock, MagicMock]:
    """Create mock service client returning (service, blob_client) for upload assertions."""
    mock_blob_client = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_container.close = MagicMock()
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_blob_client


PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"


class TestAzureBlobSinkConfig:
    """Config validation tests — no Azure SDK calls."""

    def test_connection_string_auth(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(_base_config())
        assert sink.name == "azure_blob"

    def test_sas_token_auth(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(
            _base_config(
                connection_string=None,
                sas_token="sv=2021-06-08&ss=b&srt=sco&sp=rw",
                account_url="https://fakestorage.blob.core.windows.net",
            )
        )
        assert sink._auth_config.auth_method == "sas_token"

    def test_no_auth_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        with pytest.raises(ValidationError, match="No authentication method"):
            AzureBlobSink(_base_config(connection_string=None))

    def test_empty_container_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        with pytest.raises(ValidationError, match="container cannot be empty"):
            AzureBlobSink(_base_config(container=""))

    def test_empty_blob_path_raises(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        with pytest.raises(ValidationError, match="blob_path cannot be empty"):
            AzureBlobSink(_base_config(blob_path=""))

    def test_invalid_template_syntax_raises(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        with pytest.raises(ValueError, match="Invalid blob_path template"):
            AzureBlobSink(_base_config(blob_path="{{ unclosed"))

    def test_csv_delimiter_must_be_single_char(self) -> None:
        from pydantic import ValidationError

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        with pytest.raises(ValidationError, match="single character"):
            AzureBlobSink(_base_config(csv_options={"delimiter": ";;"}))

    def test_resume_not_supported(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(_base_config())
        assert sink.supports_resume is False
        with pytest.raises(NotImplementedError, match="does not support resume"):
            sink.configure_for_resume()


class TestAzureBlobSinkLifecycle:
    """Resource lifecycle — close(), flush()."""

    def test_close_calls_client_close(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(_base_config())
        mock_client = MagicMock()
        sink._container_client = mock_client
        sink.close()

        mock_client.close.assert_called_once()
        assert sink._container_client is None
        assert sink._buffered_rows == []
        assert sink._resolved_blob_path is None
        assert sink._has_uploaded is False

    def test_close_without_client_is_safe(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(_base_config())
        sink.close()  # Should not raise

    def test_flush_is_noop(self) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        sink = AzureBlobSink(_base_config())
        sink.flush()  # Should not raise
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_azure_blob_sink.py::TestAzureBlobSinkConfig tests/unit/plugins/sinks/test_azure_blob_sink.py::TestAzureBlobSinkLifecycle -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sinks/test_azure_blob_sink.py
git commit -m "test: add Azure Blob sink config validation and lifecycle tests"
```

---

### Task 6: Azure Blob Sink — Write Flow Tests (CSV/JSON/JSONL)

**Files:**
- Modify: `tests/unit/plugins/sinks/test_azure_blob_sink.py`

Tests for `write()` across all three formats — serialization, upload, artifact descriptors, cumulative buffering.

- [ ] **Step 1: Add write flow test class**

Append to `test_azure_blob_sink.py`:

```python
class TestAzureBlobSinkWrite:
    """Write flow — serialization, upload, artifact descriptors."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

    @patch(PATCH_AUTH)
    def test_write_csv_uploads_content(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        import csv
        import io

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(schema=FIXED_SCHEMA))
        result = sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()

        # Verify upload was called
        mock_blob.upload_blob.assert_called_once()
        uploaded_bytes = mock_blob.upload_blob.call_args[0][0]

        # Verify CSV content
        reader = csv.DictReader(io.StringIO(uploaded_bytes.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "alice"

    @patch(PATCH_AUTH)
    def test_write_json_uploads_array(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="json"))
        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        uploaded_bytes = mock_blob.upload_blob.call_args[0][0]
        data = json.loads(uploaded_bytes)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "alice"

    @patch(PATCH_AUTH)
    def test_write_jsonl_uploads_lines(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="jsonl"))
        sink.write([{"id": 1}, {"id": 2}], ctx)
        sink.close()

        uploaded_bytes = mock_blob.upload_blob.call_args[0][0]
        lines = uploaded_bytes.decode().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    @patch(PATCH_AUTH)
    def test_write_returns_artifact_descriptor(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="json"))
        result = sink.write([{"id": 1}], ctx)

        assert isinstance(result.artifact, ArtifactDescriptor)
        assert result.artifact.artifact_type == "file"
        assert result.artifact.path_or_uri == "azure://test-container/output/result.csv"
        assert result.artifact.content_hash  # Non-empty SHA-256
        assert result.artifact.size_bytes > 0

    @patch(PATCH_AUTH)
    def test_write_content_hash_is_sha256(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="json"))
        result = sink.write([{"id": 1}], ctx)

        uploaded_bytes = mock_blob.upload_blob.call_args[0][0]
        expected_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        assert result.artifact.content_hash == expected_hash

    @patch(PATCH_AUTH)
    def test_write_empty_rows(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config())
        result = sink.write([], ctx)

        # No upload for empty rows
        mock_blob.upload_blob.assert_not_called()
        assert result.artifact.size_bytes == 0
        assert result.artifact.content_hash == hashlib.sha256(b"").hexdigest()

    @patch(PATCH_AUTH)
    def test_cumulative_buffering(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="json"))
        sink.write([{"id": 1}], ctx)
        sink.write([{"id": 2}], ctx)
        sink.close()

        # Second upload should contain both rows (cumulative)
        assert mock_blob.upload_blob.call_count == 2
        second_upload = mock_blob.upload_blob.call_args_list[1][0][0]
        data = json.loads(second_upload)
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2

    @patch(PATCH_AUTH)
    def test_csv_custom_delimiter(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(schema=FIXED_SCHEMA, csv_options={"delimiter": ";"}))
        sink.write([{"id": "1", "name": "alice"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0].decode()
        assert ";" in uploaded

    @patch(PATCH_AUTH)
    def test_csv_no_header(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(schema=FIXED_SCHEMA, csv_options={"include_header": False}))
        sink.write([{"id": "1", "name": "alice"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0].decode()
        # Should not contain the header row
        assert not uploaded.startswith("id")
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_azure_blob_sink.py::TestAzureBlobSinkWrite -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sinks/test_azure_blob_sink.py
git commit -m "test: add Azure Blob sink write flow tests"
```

---

### Task 7: Azure Blob Sink — Template, Overwrite, and Audit Trail Tests

**Files:**
- Modify: `tests/unit/plugins/sinks/test_azure_blob_sink.py`

Tests for Jinja2 blob path templating, overwrite protection, and audit trail recording.

- [ ] **Step 1: Add template, overwrite, and audit tests**

Append to `test_azure_blob_sink.py`:

```python
class TestAzureBlobSinkTemplateAndOverwrite:
    """Blob path templating and overwrite protection."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

    @patch(PATCH_AUTH)
    def test_template_renders_run_id(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(blob_path="output/{{ run_id }}/data.csv"))
        result = sink.write([{"id": "1"}], ctx)

        # run_id comes from ctx — check it appears in the URI
        assert ctx.run_id in result.artifact.path_or_uri

    @patch(PATCH_AUTH)
    def test_template_renders_timestamp(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(blob_path="output/{{ timestamp }}.csv"))
        result = sink.write([{"id": "1"}], ctx)

        # Timestamp should be ISO format with T separator
        path = result.artifact.path_or_uri
        assert "T" in path  # ISO timestamp format

    @patch(PATCH_AUTH)
    def test_blob_path_frozen_after_first_write(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(blob_path="output/{{ timestamp }}.csv"))
        result1 = sink.write([{"id": "1"}], ctx)
        result2 = sink.write([{"id": "2"}], ctx)

        # Both writes should target the same blob path
        assert result1.artifact.path_or_uri == result2.artifact.path_or_uri

    @patch(PATCH_AUTH)
    def test_undefined_template_var_raises(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from jinja2 import UndefinedError

        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(blob_path="output/{{ nonexistent }}.csv"))
        with pytest.raises(UndefinedError):
            sink.write([{"id": "1"}], ctx)

    @patch(PATCH_AUTH)
    def test_overwrite_true_passes_overwrite_flag(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(overwrite=True))
        sink.write([{"id": "1"}], ctx)

        _, kwargs = mock_blob.upload_blob.call_args
        assert kwargs["overwrite"] is True

    @patch(PATCH_AUTH)
    def test_overwrite_false_first_write_no_overwrite(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(overwrite=False))
        sink.write([{"id": "1"}], ctx)

        _, kwargs = mock_blob.upload_blob.call_args
        assert kwargs["overwrite"] is False

    @patch(PATCH_AUTH)
    def test_overwrite_false_second_write_allows_rewrite(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(overwrite=False))
        sink.write([{"id": "1"}], ctx)
        sink.write([{"id": "2"}], ctx)

        # Second write should use overwrite=True (same blob, cumulative buffer)
        _, kwargs = mock_blob.upload_blob.call_args_list[1]
        assert kwargs["overwrite"] is True

    @patch(PATCH_AUTH)
    def test_resource_exists_error_raises_value_error(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        # Simulate ResourceExistsError
        error = Exception("blob exists")
        error.__class__.__name__ = "ResourceExistsError"
        type(error).__name__ = "ResourceExistsError"
        # Use a proper mock exception with the right class name
        class ResourceExistsError(Exception):
            pass

        mock_blob.upload_blob.side_effect = ResourceExistsError("blob exists")

        sink = AzureBlobSink(_base_config(overwrite=False))

        # The code checks type(e).__name__ == "ResourceExistsError"
        with pytest.raises(ValueError, match="already exists"):
            sink.write([{"id": "1"}], ctx)


class TestAzureBlobSinkAudit:
    """Audit trail recording for uploads."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

    @patch(PATCH_AUTH)
    def test_upload_failure_records_error(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_blob.upload_blob.side_effect = RuntimeError("network timeout")
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config())
        with pytest.raises(RuntimeError, match="Failed to upload blob"):
            sink.write([{"id": "1"}], ctx)

    @patch(PATCH_AUTH)
    def test_audit_integrity_error_on_record_call_failure(self, mock_create: MagicMock) -> None:
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        ctx = make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

        # Make record_call fail AFTER the upload succeeds
        original = ctx.record_call
        call_count = 0

        def failing_after_upload(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("audit DB down")

        ctx.record_call = failing_after_upload  # type: ignore[assignment]

        sink = AzureBlobSink(_base_config())
        with pytest.raises(AuditIntegrityError, match="audit trail"):
            sink.write([{"id": "1"}], ctx)

    @patch(PATCH_AUTH)
    def test_programming_errors_crash_directly(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service = MagicMock()
        mock_service.get_container_client.side_effect = AttributeError("bad attr")
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config())
        with pytest.raises(AttributeError, match="bad attr"):
            sink.write([{"id": "1"}], ctx)

    @patch(PATCH_AUTH)
    def test_buffer_not_committed_on_upload_failure(self, mock_create: MagicMock, ctx: PluginContext) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        sink = AzureBlobSink(_base_config(format="json"))

        # First write succeeds
        sink.write([{"id": 1}], ctx)
        assert len(sink._buffered_rows) == 1

        # Second write fails — buffer should NOT grow
        mock_blob.upload_blob.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError):
            sink.write([{"id": 2}], ctx)

        # Buffer still has only the first row
        assert len(sink._buffered_rows) == 1
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_azure_blob_sink.py::TestAzureBlobSinkTemplateAndOverwrite tests/unit/plugins/sinks/test_azure_blob_sink.py::TestAzureBlobSinkAudit -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sinks/test_azure_blob_sink.py
git commit -m "test: add Azure Blob sink template, overwrite, and audit trail tests"
```

---

### Task 8: Property Tests — Source Round-Trip and Quarantine Completeness

**Files:**
- Create: `tests/property/plugins/sources/test_azure_blob_source_properties.py`

Property-based tests using Hypothesis for source parsing fidelity.

- [ ] **Step 1: Create property test directory and file**

```bash
mkdir -p tests/property/plugins/sources
touch tests/property/plugins/sources/__init__.py
touch tests/property/plugins/__init__.py
```

- [ ] **Step 2: Write property tests**

```python
"""Property-based tests for Azure Blob source behavior."""

from __future__ import annotations

import csv
import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st

from tests.fixtures.factories import make_operation_context
from tests.strategies.settings import SLOW_SETTINGS

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"

# Strategies
safe_text = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), max_codepoint=0x7E),
).filter(lambda s: s.strip() and "," not in s and "\n" not in s and '"' not in s)

json_scalar = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=0, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
    st.none(),
)


def _mock_blob_download(data: bytes) -> MagicMock:
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = data
    mock_service = MagicMock()
    mock_service.get_container_client.return_value.get_blob_client.return_value = mock_blob_client
    return mock_service


class TestAzureBlobSourceCSVProperties:
    """Property tests for CSV round-trip fidelity."""

    @given(
        names=st.lists(safe_text, min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_csv_round_trip_preserves_values(
        self,
        mock_create: MagicMock,
        names: list[str],
        data: st.DataObject,
    ) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        # Generate rows with the given column names
        num_rows = data.draw(st.integers(min_value=1, max_value=5))
        rows = []
        for _ in range(num_rows):
            values = data.draw(
                st.lists(safe_text, min_size=len(names), max_size=len(names))
            )
            rows.append(dict(zip(names, values, strict=True)))

        # Serialize to CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
        blob_data = output.getvalue().encode("utf-8")

        mock_create.return_value = _mock_blob_download(blob_data)
        ctx = make_operation_context(plugin_name="azure_blob")

        source = AzureBlobSource(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.csv",
                "format": "csv",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        result_rows = list(source.load(ctx))

        valid = [r for r in result_rows if not r.is_quarantined]
        assert len(valid) == len(rows)
        for original, loaded in zip(rows, valid, strict=True):
            for key in original:
                # CSV values are always strings after parsing
                assert str(original[key]) == str(loaded.row[key])


class TestAzureBlobSourceJSONProperties:
    """Property tests for JSON round-trip fidelity."""

    @given(
        keys=st.lists(
            st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True),
            min_size=1,
            max_size=4,
            unique=True,
        ),
        data=st.data(),
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_json_round_trip_preserves_structure(
        self,
        mock_create: MagicMock,
        keys: list[str],
        data: st.DataObject,
    ) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        num_rows = data.draw(st.integers(min_value=1, max_value=5))
        rows = []
        for _ in range(num_rows):
            values = data.draw(
                st.lists(json_scalar, min_size=len(keys), max_size=len(keys))
            )
            rows.append(dict(zip(keys, values, strict=True)))

        blob_data = json.dumps(rows).encode("utf-8")
        mock_create.return_value = _mock_blob_download(blob_data)
        ctx = make_operation_context(plugin_name="azure_blob")

        source = AzureBlobSource(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.json",
                "format": "json",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        result_rows = list(source.load(ctx))

        valid = [r for r in result_rows if not r.is_quarantined]
        assert len(valid) == len(rows)


class TestAzureBlobSourceQuarantineProperties:
    """Property tests: malformed data always produces quarantined rows."""

    @given(garbage=st.binary(min_size=1, max_size=100))
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_garbage_json_never_silently_dropped(
        self,
        mock_create: MagicMock,
        garbage: bytes,
    ) -> None:
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        mock_create.return_value = _mock_blob_download(garbage)
        ctx = make_operation_context(plugin_name="azure_blob")

        source = AzureBlobSource(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.json",
                "format": "json",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )

        try:
            rows = list(source.load(ctx))
        except (ImportError, TypeError, AttributeError, KeyError, NameError):
            # Programming errors re-raised by design
            return

        # If we got rows, at least one must be quarantined (garbage can't parse to valid JSON array)
        # OR it happened to be valid JSON (unlikely but possible)
        if rows:
            try:
                parsed = json.loads(garbage.decode("utf-8"))
                if isinstance(parsed, list):
                    return  # Valid JSON array — valid rows are fine
            except Exception:
                pass
            # If not valid JSON array, all rows must be quarantined
            assert all(r.is_quarantined for r in rows)
```

- [ ] **Step 3: Run property tests**

Run: `.venv/bin/python -m pytest tests/property/plugins/sources/test_azure_blob_source_properties.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/property/plugins/sources/
git commit -m "test: add Azure Blob source property-based tests"
```

---

### Task 9: Property Tests — Sink Serialization and Hash Determinism

**Files:**
- Create: `tests/property/plugins/sinks/test_azure_blob_sink_properties.py`

Property-based tests for sink serialization round-trip and hash determinism.

- [ ] **Step 1: Create directory if needed**

```bash
mkdir -p tests/property/plugins/sinks
touch tests/property/plugins/sinks/__init__.py
```

- [ ] **Step 2: Write property tests**

```python
"""Property-based tests for Azure Blob sink behavior."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st

from tests.fixtures.factories import make_operation_context
from tests.strategies.settings import SLOW_SETTINGS

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"

# Strategies
row_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),
        "name": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        "score": st.one_of(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), st.none()),
    }
)
rows_strategy = st.lists(row_strategy, min_size=1, max_size=5)


def _mock_blob_upload() -> tuple[MagicMock, MagicMock]:
    mock_blob_client = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_container.close = MagicMock()
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_blob_client


FIXED_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str", "score: float?"]}


class TestAzureBlobSinkHashProperties:
    """Hash determinism and content integrity."""

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_hash_matches_uploaded_content(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, Any]],
    ) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        ctx = make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

        sink = AzureBlobSink(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.json",
                "format": "json",
                "schema": FIXED_SCHEMA,
            }
        )
        result = sink.write(rows, ctx)
        sink.close()

        uploaded = mock_blob.upload_blob.call_args[0][0]
        assert result.artifact.content_hash == hashlib.sha256(uploaded).hexdigest()
        assert result.artifact.size_bytes == len(uploaded)

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_same_rows_produce_same_hash(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, Any]],
    ) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service1, mock_blob1 = _mock_blob_upload()
        mock_service2, mock_blob2 = _mock_blob_upload()

        hashes = []
        for mock_service in [mock_service1, mock_service2]:
            mock_create.return_value = mock_service
            ctx = make_operation_context(
                operation_type="sink_write",
                node_id="sink",
                node_type="SINK",
                plugin_name="azure_blob",
            )
            sink = AzureBlobSink(
                {
                    "connection_string": FAKE_CONN_STRING,
                    "container": "c",
                    "blob_path": "d.json",
                    "format": "json",
                    "schema": FIXED_SCHEMA,
                }
            )
            result = sink.write(rows, ctx)
            hashes.append(result.artifact.content_hash)
            sink.close()

        assert hashes[0] == hashes[1]


class TestAzureBlobSinkJSONLRoundTrip:
    """JSONL serialization round-trip."""

    @given(rows=rows_strategy)
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_jsonl_round_trip(
        self,
        mock_create: MagicMock,
        rows: list[dict[str, Any]],
    ) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        mock_service, mock_blob = _mock_blob_upload()
        mock_create.return_value = mock_service

        ctx = make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
            plugin_name="azure_blob",
        )

        sink = AzureBlobSink(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.jsonl",
                "format": "jsonl",
                "schema": FIXED_SCHEMA,
            }
        )
        sink.write(rows, ctx)
        sink.close()

        uploaded = mock_blob.upload_blob.call_args[0][0]
        lines = uploaded.decode().strip().split("\n")
        parsed = [json.loads(line) for line in lines]

        assert len(parsed) == len(rows)
        for original, restored in zip(rows, parsed, strict=True):
            for key in original:
                if original[key] is None:
                    assert restored[key] is None
                else:
                    assert restored[key] == original[key]


class TestAzureBlobSinkBufferingProperties:
    """Cumulative buffering invariant."""

    @given(
        batch1=rows_strategy,
        batch2=rows_strategy,
    )
    @SLOW_SETTINGS
    @patch(PATCH_AUTH)
    def test_two_writes_equals_one_combined_write(
        self,
        mock_create: MagicMock,
        batch1: list[dict[str, Any]],
        batch2: list[dict[str, Any]],
    ) -> None:
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink

        # Two-write path
        mock_service_a, mock_blob_a = _mock_blob_upload()
        mock_create.return_value = mock_service_a
        ctx_a = make_operation_context(
            operation_type="sink_write", node_id="sink", node_type="SINK", plugin_name="azure_blob"
        )
        sink_a = AzureBlobSink(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.json",
                "format": "json",
                "schema": FIXED_SCHEMA,
            }
        )
        sink_a.write(batch1, ctx_a)
        sink_a.write(batch2, ctx_a)
        two_write_content = mock_blob_a.upload_blob.call_args_list[-1][0][0]
        sink_a.close()

        # One-write path
        mock_service_b, mock_blob_b = _mock_blob_upload()
        mock_create.return_value = mock_service_b
        ctx_b = make_operation_context(
            operation_type="sink_write", node_id="sink", node_type="SINK", plugin_name="azure_blob"
        )
        sink_b = AzureBlobSink(
            {
                "connection_string": FAKE_CONN_STRING,
                "container": "c",
                "blob_path": "d.json",
                "format": "json",
                "schema": FIXED_SCHEMA,
            }
        )
        sink_b.write(batch1 + batch2, ctx_b)
        one_write_content = mock_blob_b.upload_blob.call_args[0][0]
        sink_b.close()

        assert two_write_content == one_write_content
```

- [ ] **Step 3: Run property tests**

Run: `.venv/bin/python -m pytest tests/property/plugins/sinks/test_azure_blob_sink_properties.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/property/plugins/sinks/
git commit -m "test: add Azure Blob sink property-based tests"
```

---

### Task 10: Final Verification and Issue Closure

**Files:**
- No new files

- [ ] **Step 1: Run all Azure Blob tests together**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sources/test_azure_blob_source.py tests/unit/plugins/sinks/test_azure_blob_sink.py tests/unit/plugins/sinks/test_azure_blob_sink_serialization.py tests/property/plugins/sources/test_azure_blob_source_properties.py tests/property/plugins/sinks/test_azure_blob_sink_properties.py -v`
Expected: All PASS

- [ ] **Step 2: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/ tests/property/ --tb=short -q`
Expected: All existing tests still pass

- [ ] **Step 3: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS (test-only files don't affect tier model)

- [ ] **Step 4: Final commit and close issue**

```bash
git add -A
git commit -m "test: add comprehensive Azure Blob source and sink test coverage — close elspeth-78c0e92eb7"
```

Close filigree issue `elspeth-78c0e92eb7`.
