"""Tests for Azure Blob Storage source plugin."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from tests.fixtures.factories import make_operation_context

# Shared constants
FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA: dict[str, Any] = {"mode": "observed"}
FIXED_SCHEMA: dict[str, Any] = {
    "mode": "fixed",
    "fields": ["id: int", "name: str", "value: float"],
}
FLEXIBLE_SCHEMA: dict[str, Any] = {
    "mode": "flexible",
    "fields": ["id: int"],
}
QUARANTINE_SINK = "quarantine"
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"

ACCOUNT_URL = "https://fakestorage.blob.core.windows.net"


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid config with connection_string auth."""
    config: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "data/input.csv",
        "schema": DYNAMIC_SCHEMA,
        "on_validation_failure": QUARANTINE_SINK,
    }
    config.update(overrides)
    return config


def _mock_blob_download(data: bytes) -> MagicMock:
    """Create a mock service client that returns data from download_blob().readall()."""
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value.readall.return_value = data
    mock_service = MagicMock()
    mock_service.get_container_client.return_value.get_blob_client.return_value = mock_blob_client
    return mock_service


def _make_source(config: dict[str, Any]) -> Any:
    """Create AzureBlobSource with patched auth so no real Azure calls happen."""
    from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

    with patch(PATCH_AUTH, return_value=MagicMock()):
        return AzureBlobSource(config)


# ---------------------------------------------------------------------------
# Task 1: Config Validation
# ---------------------------------------------------------------------------


class TestAzureBlobSourceConfig:
    """Config validation tests -- no Azure SDK calls needed."""

    def test_connection_string_auth(self) -> None:
        """Connection string config sets name and output_schema."""
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        with patch(PATCH_AUTH, return_value=MagicMock()):
            source = AzureBlobSource(_base_config())

        assert source.name == "azure_blob"
        assert source.output_schema is not None

    def test_sas_token_auth(self) -> None:
        """SAS token auth method accepted."""
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        cfg = _base_config(
            connection_string=None,
            sas_token="sv=2021-06-08&ss=b",
            account_url=ACCOUNT_URL,
        )
        with patch(PATCH_AUTH, return_value=MagicMock()):
            source = AzureBlobSource(cfg)

        assert source._auth_config.auth_method == "sas_token"

    def test_managed_identity_auth(self) -> None:
        """Managed identity auth method accepted."""
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        cfg = _base_config(
            connection_string=None,
            use_managed_identity=True,
            account_url=ACCOUNT_URL,
        )
        with patch(PATCH_AUTH, return_value=MagicMock()):
            source = AzureBlobSource(cfg)

        assert source._auth_config.auth_method == "managed_identity"

    def test_service_principal_auth(self) -> None:
        """Service principal auth method accepted."""
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSource

        cfg = _base_config(
            connection_string=None,
            tenant_id="tid",
            client_id="cid",
            client_secret="csec",
            account_url=ACCOUNT_URL,
        )
        with patch(PATCH_AUTH, return_value=MagicMock()):
            source = AzureBlobSource(cfg)

        assert source._auth_config.auth_method == "service_principal"

    def test_no_auth_raises(self) -> None:
        """No auth method configured raises PluginConfigError."""
        cfg = _base_config(connection_string=None)
        with pytest.raises(PluginConfigError, match="authentication"):
            _make_source(cfg)

    def test_multiple_auth_raises(self) -> None:
        """Multiple auth methods configured raises PluginConfigError."""
        cfg = _base_config(
            sas_token="sv=2021",
            account_url=ACCOUNT_URL,
        )
        with pytest.raises(PluginConfigError, match="Multiple"):
            _make_source(cfg)

    def test_empty_container_raises(self) -> None:
        """Empty container raises PluginConfigError."""
        cfg = _base_config(container="")
        with pytest.raises(PluginConfigError, match="container"):
            _make_source(cfg)

    def test_empty_blob_path_raises(self) -> None:
        """Empty blob_path raises PluginConfigError."""
        cfg = _base_config(blob_path="")
        with pytest.raises(PluginConfigError, match="blob_path"):
            _make_source(cfg)

    def test_columns_rejected_for_json(self) -> None:
        """columns option rejected for JSON format."""
        cfg = _base_config(format="json", columns=["a", "b"])
        with pytest.raises(PluginConfigError, match="columns"):
            _make_source(cfg)

    def test_columns_with_has_header_raises(self) -> None:
        """columns with has_header=True raises PluginConfigError."""
        cfg = _base_config(
            columns=["a", "b"],
            csv_options={"has_header": True},
        )
        with pytest.raises(PluginConfigError, match="has_header"):
            _make_source(cfg)

    def test_csv_delimiter_must_be_single_char(self) -> None:
        """Multi-char delimiter raises PluginConfigError."""
        cfg = _base_config(csv_options={"delimiter": "||"})
        with pytest.raises(PluginConfigError, match="delimiter"):
            _make_source(cfg)

    def test_invalid_encoding_raises(self) -> None:
        """Unknown encoding raises PluginConfigError."""
        cfg = _base_config(csv_options={"encoding": "bogus-999"})
        with pytest.raises(PluginConfigError, match="encoding"):
            _make_source(cfg)

    def test_fixed_schema_creates_locked_contract_for_json(self) -> None:
        """Fixed schema for JSON creates a locked contract immediately."""
        source = _make_source(_base_config(format="json", schema=FIXED_SCHEMA))
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True

    def test_observed_schema_defers_contract_builder_until_json_field_resolution(self) -> None:
        """Observed JSON defers contract builder until first-row field resolution."""
        source = _make_source(_base_config(format="json", schema=DYNAMIC_SCHEMA))
        assert source.get_schema_contract() is None
        assert source._contract_builder is None

    def test_csv_defers_contract_until_load(self) -> None:
        """CSV format defers contract creation until load() (needs field resolution)."""
        source = _make_source(_base_config(format="csv", schema=DYNAMIC_SCHEMA))
        assert source._contract_builder is None  # Created in load()


# ---------------------------------------------------------------------------
# Task 2: CSV Loading
# ---------------------------------------------------------------------------


class TestAzureBlobSourceCSV:
    """CSV loading from Azure Blob -- mocked Azure SDK."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    def test_load_csv_with_headers(self, ctx: PluginContext) -> None:
        """Load 3-row CSV with headers, verify dict contents."""
        csv_bytes = b"id,name,value\n1,alice,100\n2,bob,200\n3,carol,300\n"
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"id": "1", "name": "alice", "value": "100"}
        assert rows[1].row["name"] == "bob"
        assert rows[2].row["value"] == "300"

    def test_custom_delimiter(self, ctx: PluginContext) -> None:
        """CSV with semicolon delimiter."""
        csv_bytes = b"id;name\n1;alice\n"
        source = _make_source(_base_config(csv_options={"delimiter": ";"}))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row["name"] == "alice"

    def test_latin1_encoding(self, ctx: PluginContext) -> None:
        """CSV with latin-1 encoding."""
        csv_bytes = b"id,name\n1,caf\xe9\n"
        source = _make_source(_base_config(csv_options={"encoding": "latin-1"}))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row["name"] == "caf\u00e9"

    def test_headerless_with_explicit_columns(self, ctx: PluginContext) -> None:
        """Headerless CSV with explicit columns config."""
        csv_bytes = b"1,alice,100\n2,bob,200\n"
        source = _make_source(
            _base_config(
                columns=["id", "name", "value"],
                csv_options={"has_header": False},
            )
        )

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": "1", "name": "alice", "value": "100"}

    def test_headerless_no_columns_uses_numeric(self, ctx: PluginContext) -> None:
        """Headerless CSV without columns uses numeric column names."""
        csv_bytes = b"1,alice,100\n2,bob,200\n"
        source = _make_source(_base_config(csv_options={"has_header": False}))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"0": "1", "1": "alice", "2": "100"}

    def test_column_count_mismatch_quarantines_row(self, ctx: PluginContext) -> None:
        """Column count mismatch quarantines individual row, continues processing."""
        csv_bytes = b"id,name,value\n1,alice,100\n2,bob\n3,carol,300\n"
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        # 3 total rows: 2 valid + 1 quarantined
        assert len(rows) == 3
        valid = [r for r in rows if not r.is_quarantined]
        quarantined = [r for r in rows if r.is_quarantined]
        assert len(valid) == 2
        assert len(quarantined) == 1
        assert "expected" in quarantined[0].quarantine_error
        assert quarantined[0].quarantine_destination == QUARANTINE_SINK

    def test_empty_file_quarantines(self, ctx: PluginContext) -> None:
        """Empty CSV file quarantines (no header row)."""
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(b"")):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "empty" in rows[0].quarantine_error.lower()

    def test_unicode_decode_error_quarantines(self, ctx: PluginContext) -> None:
        """Encoding error quarantines the entire file."""
        # Invalid UTF-8 bytes
        bad_bytes = b"\xff\xfe\x00\x01"
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(bad_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "decode" in rows[0].quarantine_error.lower()

    def test_discard_mode_suppresses_quarantine_yield(self, ctx: PluginContext) -> None:
        """on_validation_failure='discard' suppresses quarantine row yield."""
        csv_bytes = b"id,name\n1,alice\n2\n3,carol\n"
        source = _make_source(_base_config(on_validation_failure="discard"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        # Only valid rows yielded; quarantined row discarded
        assert all(not r.is_quarantined for r in rows)
        assert len(rows) == 2

    def test_blank_lines_skipped(self, ctx: PluginContext) -> None:
        """Blank lines in CSV are skipped."""
        csv_bytes = b"id,name\n1,alice\n\n2,bob\n"
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        valid = [r for r in rows if not r.is_quarantined]
        assert len(valid) == 2

    def test_field_mapping(self, ctx: PluginContext) -> None:
        """field_mapping overrides normalized header names."""
        csv_bytes = b"ID,Full Name\n1,Alice\n"
        source = _make_source(_base_config(field_mapping={"full_name": "display_name"}))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert "display_name" in rows[0].row

    def test_close_nulls_client(self, ctx: PluginContext) -> None:
        """close() sets _blob_client to None."""
        source = _make_source(_base_config())
        source._blob_client = MagicMock()
        source.close()
        assert source._blob_client is None

    def test_close_idempotent(self, ctx: PluginContext) -> None:
        """close() can be called multiple times without error."""
        source = _make_source(_base_config())
        source.close()
        source.close()  # Should not raise


# ---------------------------------------------------------------------------
# Task 3: JSON Array Loading
# ---------------------------------------------------------------------------


class TestAzureBlobSourceJSON:
    """JSON array loading."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    def test_load_json_array(self, ctx: PluginContext) -> None:
        """Load 2-row JSON array."""
        data = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[1].row["name"] == "bob"

    def test_data_key_extraction(self, ctx: PluginContext) -> None:
        """data_key extracts nested array from JSON object."""
        data = {"meta": {}, "results": [{"id": 1}, {"id": 2}]}
        blob_bytes = json.dumps(data).encode()
        source = _make_source(
            _base_config(
                format="json",
                json_options={"data_key": "results"},
            )
        )

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1}

    def test_data_key_not_found_quarantines(self, ctx: PluginContext) -> None:
        """Missing data_key quarantines."""
        blob_bytes = json.dumps({"other": []}).encode()
        source = _make_source(
            _base_config(
                format="json",
                json_options={"data_key": "results"},
            )
        )

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "not found" in rows[0].quarantine_error

    def test_data_key_on_non_object_quarantines(self, ctx: PluginContext) -> None:
        """data_key on non-object (array) quarantines."""
        blob_bytes = json.dumps([1, 2, 3]).encode()
        source = _make_source(
            _base_config(
                format="json",
                json_options={"data_key": "results"},
            )
        )

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "expected JSON object" in rows[0].quarantine_error

    def test_not_array_quarantines(self, ctx: PluginContext) -> None:
        """Non-array top-level JSON quarantines."""
        blob_bytes = json.dumps({"a": 1}).encode()
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "Expected JSON array" in rows[0].quarantine_error

    def test_invalid_json_quarantines(self, ctx: PluginContext) -> None:
        """Invalid JSON quarantines."""
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(b"{invalid json")):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True

    def test_nonfinite_rejected(self, ctx: PluginContext) -> None:
        """NaN/Infinity in JSON is rejected (quarantined)."""
        # NaN is not valid JSON but Python's json.loads accepts it by default
        blob_bytes = b'[{"value": NaN}]'
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True

    def test_encoding_error_quarantines(self, ctx: PluginContext) -> None:
        """Encoding error quarantines entire file."""
        bad_bytes = b"\xff\xfe\x00\x01"
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(bad_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "decode" in rows[0].quarantine_error.lower()

    @pytest.mark.parametrize(
        ("source_format", "blob_bytes"),
        [
            ("json", json.dumps([{"Customer Name": "Alice", "Order ID": 101}]).encode()),
            ("jsonl", b'{"Customer Name": "Alice", "Order ID": 101}\n'),
        ],
    )
    def test_json_formats_normalize_keys_and_expose_field_resolution(
        self,
        source_format: str,
        blob_bytes: bytes,
        ctx: PluginContext,
    ) -> None:
        """JSON and JSONL normalize external keys and retain audit mapping."""
        source = _make_source(_base_config(format=source_format))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"customer_name": "Alice", "order_id": 101}

        field_resolution = source.get_field_resolution()
        assert field_resolution is not None
        resolution_mapping, normalization_version = field_resolution
        assert resolution_mapping == {
            "Customer Name": "customer_name",
            "Order ID": "order_id",
        }
        assert normalization_version == "1.0.0"

    def test_json_field_mapping_overrides_normalized_keys(self, ctx: PluginContext) -> None:
        """JSON field_mapping can override names after source-boundary normalization."""
        data = [{"Customer Name": "Alice", "Order ID": 101}]
        source = _make_source(
            _base_config(
                format="json",
                field_mapping={"customer_name": "display_name"},
            )
        )

        with patch(PATCH_AUTH, return_value=_mock_blob_download(json.dumps(data).encode())):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row == {"display_name": "Alice", "order_id": 101}
        assert source.get_field_resolution() is not None


# ---------------------------------------------------------------------------
# Task 3: JSONL Loading
# ---------------------------------------------------------------------------


class TestAzureBlobSourceJSONL:
    """JSONL (newline-delimited JSON) loading."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    def test_load_jsonl(self, ctx: PluginContext) -> None:
        """Load 2-row JSONL."""
        blob_bytes = b'{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n'
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[1].row["name"] == "bob"

    def test_skips_empty_lines(self, ctx: PluginContext) -> None:
        """JSONL skips blank lines."""
        blob_bytes = b'{"id": 1}\n\n{"id": 2}\n\n'
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_per_line_quarantine(self, ctx: PluginContext) -> None:
        """Good, bad, good -- all 3 yielded (bad quarantined)."""
        blob_bytes = b'{"id": 1}\n{bad json}\n{"id": 3}\n'
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[1].is_quarantined is True
        assert rows[2].is_quarantined is False

    def test_discard_mode(self, ctx: PluginContext) -> None:
        """Discard mode suppresses quarantine yield for bad JSONL lines."""
        blob_bytes = b'{"id": 1}\n{bad}\n{"id": 3}\n'
        source = _make_source(_base_config(format="jsonl", on_validation_failure="discard"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        # Only valid rows
        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)

    def test_nonfinite_per_line(self, ctx: PluginContext) -> None:
        """NaN in individual JSONL line is quarantined."""
        blob_bytes = b'{"id": 1}\n{"value": NaN}\n'
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].is_quarantined is False
        assert rows[1].is_quarantined is True

    def test_encoding_error(self, ctx: PluginContext) -> None:
        """Encoding error in a single-line JSONL blob quarantines that line."""
        bad_bytes = b"\xff\xfe\x00\x01"
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(bad_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert "encoding" in rows[0].quarantine_error.lower()

    def test_invalid_encoding_line_quarantined_and_neighboring_rows_continue(self, ctx: PluginContext) -> None:
        """A single invalid-encoding line does not drop valid JSONL neighbors."""
        blob_bytes = b'{"id": 1}\n\xff\xfe\n{"id": 3}\n'
        source = _make_source(_base_config(format="jsonl"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"id": 1}
        assert rows[1].is_quarantined is True
        assert rows[1].quarantine_error is not None
        assert "line 2" in rows[1].quarantine_error
        assert "utf-8" in rows[1].quarantine_error.lower()
        assert rows[1].row["__raw_bytes_hex__"] == "fffe"
        assert rows[1].row["__line_number__"] == 2
        assert rows[2].is_quarantined is False
        assert rows[2].row == {"id": 3}


# ---------------------------------------------------------------------------
# Task 4: Schema Validation
# ---------------------------------------------------------------------------


class TestAzureBlobSourceSchemaValidation:
    """Schema contract locking tests."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    def test_fixed_schema_validates_types(self, ctx: PluginContext) -> None:
        """Fixed schema validates and coerces types."""
        data = [{"id": 1, "name": "alice", "value": 3.14}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json", schema=FIXED_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is False
        assert rows[0].row["id"] == 1
        assert rows[0].row["name"] == "alice"
        assert rows[0].row["value"] == 3.14

    def test_fixed_schema_quarantines_invalid_types(self, ctx: PluginContext) -> None:
        """Fixed schema quarantines rows that fail validation."""
        # "not_a_number" cannot coerce to int
        data = [{"id": "not_a_number", "name": "alice", "value": 3.14}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json", schema=FIXED_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined is True

    def test_flexible_schema_locks_on_first_row(self, ctx: PluginContext) -> None:
        """Flexible schema locks contract after first valid row."""
        data = [{"id": 1, "extra": "val"}, {"id": 2, "extra": "val2"}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json", schema=FLEXIBLE_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
        assert len(rows) == 2

    def test_observed_schema_locks_on_first_row(self, ctx: PluginContext) -> None:
        """Observed schema locks contract after first valid row."""
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json", schema=DYNAMIC_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
        assert len(rows) == 2

    def test_no_valid_rows_still_locks_contract(self, ctx: PluginContext) -> None:
        """All-invalid input still locks contract (empty schema)."""
        blob_bytes = b"not valid json"
        source = _make_source(_base_config(format="json", schema=DYNAMIC_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            list(source.load(ctx))

        # Contract should be locked even with no valid rows
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True

    def test_source_row_has_contract(self, ctx: PluginContext) -> None:
        """Valid SourceRow includes contract reference."""
        data = [{"id": 1, "name": "alice"}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json", schema=DYNAMIC_SCHEMA))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            rows = list(source.load(ctx))

        assert len(rows) == 1
        assert not rows[0].is_quarantined
        assert rows[0].contract is not None
        assert rows[0].contract.locked is True


# ---------------------------------------------------------------------------
# Task 4: Audit Trail and Error Handling
# ---------------------------------------------------------------------------


class TestAzureBlobSourceAuditAndErrors:
    """Audit trail and error handling tests."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_operation_context(plugin_name="azure_blob")

    def test_download_failure_raises_runtime_error(self, ctx: PluginContext) -> None:
        """Azure download failure raises RuntimeError."""
        source = _make_source(_base_config())

        mock_service = MagicMock()
        mock_service.get_container_client.return_value.get_blob_client.return_value.download_blob.side_effect = Exception(
            "connection refused"
        )

        with (
            patch(PATCH_AUTH, return_value=mock_service),
            pytest.raises(RuntimeError, match="Failed to download blob"),
        ):
            list(source.load(ctx))

    def test_import_error_propagated(self, ctx: PluginContext) -> None:
        """ImportError from missing azure SDK propagates directly."""
        source = _make_source(_base_config())

        with (
            patch(PATCH_AUTH, side_effect=ImportError("no azure")),
            pytest.raises(ImportError, match="no azure"),
        ):
            list(source.load(ctx))

    def test_programming_errors_crash_directly(self, ctx: PluginContext) -> None:
        """Programming errors (TypeError) crash through, not caught."""
        source = _make_source(_base_config())

        mock_service = MagicMock()
        mock_service.get_container_client.return_value.get_blob_client.return_value.download_blob.side_effect = TypeError("bad argument")

        with (
            patch(PATCH_AUTH, return_value=mock_service),
            pytest.raises(TypeError, match="bad argument"),
        ):
            list(source.load(ctx))

    def test_audit_integrity_error_on_record_call_failure(self, ctx: PluginContext) -> None:
        """AuditIntegrityError when record_call fails after successful download."""
        source = _make_source(_base_config())

        blob_bytes = b"id,name\n1,alice\n"
        mock_service = _mock_blob_download(blob_bytes)

        # Make record_call raise to simulate audit failure
        ctx.record_call = MagicMock(side_effect=Exception("db write failed"))  # type: ignore[method-assign]

        with (
            patch(PATCH_AUTH, return_value=mock_service),
            pytest.raises(AuditIntegrityError, match="audit trail"),
        ):
            list(source.load(ctx))

    def test_field_resolution_returned_for_csv(self, ctx: PluginContext) -> None:
        """get_field_resolution returns mapping for CSV after load."""
        csv_bytes = b"ID,Full Name\n1,Alice\n"
        source = _make_source(_base_config())

        with patch(PATCH_AUTH, return_value=_mock_blob_download(csv_bytes)):
            list(source.load(ctx))

        result = source.get_field_resolution()
        assert result is not None
        resolution_map, _version = result
        assert isinstance(resolution_map, Mapping)
        assert len(resolution_map) > 0

    def test_field_resolution_returned_for_json(self, ctx: PluginContext) -> None:
        """get_field_resolution returns mapping for JSON format after load."""
        data = [{"Customer Name": "Alice"}]
        blob_bytes = json.dumps(data).encode()
        source = _make_source(_base_config(format="json"))

        with patch(PATCH_AUTH, return_value=_mock_blob_download(blob_bytes)):
            list(source.load(ctx))

        result = source.get_field_resolution()
        assert result is not None
        resolution_map, _version = result
        assert resolution_map == {"Customer Name": "customer_name"}
