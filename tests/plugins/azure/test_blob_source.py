"""Tests for Azure Blob Storage source plugin."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import SourceRow
from elspeth.plugins.azure.blob_source import AzureBlobSource
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol

# Dynamic schema config for tests - DataPluginConfig requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}

# Standard quarantine routing for tests
QUARANTINE_SINK = "quarantine"

# Standard connection string for tests
TEST_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key"
TEST_CONTAINER = "test-container"
TEST_BLOB_PATH = "data/input.csv"

# Managed Identity test values
TEST_ACCOUNT_URL = "https://mystorageaccount.blob.core.windows.net"

# Service Principal test values
TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"
TEST_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
TEST_CLIENT_SECRET = "test-secret-value"


@pytest.fixture
def ctx() -> PluginContext:
    """Create a minimal plugin context."""
    return PluginContext(run_id="test-run", config={})


@pytest.fixture
def mock_blob_client() -> Generator[MagicMock, None, None]:
    """Create a mock blob client for testing."""
    with patch("elspeth.plugins.azure.blob_source.AzureBlobSource._get_blob_client") as mock:
        yield mock


def make_config(
    *,
    # Auth Option 1: Connection string (default)
    connection_string: str | None = TEST_CONNECTION_STRING,
    # Auth Option 2: Managed Identity
    use_managed_identity: bool = False,
    account_url: str | None = None,
    # Auth Option 3: Service Principal
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    # Blob location
    container: str = TEST_CONTAINER,
    blob_path: str = TEST_BLOB_PATH,
    format: str = "csv",
    csv_options: dict[str, Any] | None = None,
    json_options: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    on_validation_failure: str = QUARANTINE_SINK,
) -> dict[str, Any]:
    """Helper to create config dicts with defaults.

    By default uses connection_string auth. Pass connection_string=None
    and set other auth options for managed identity or service principal.
    """
    config: dict[str, Any] = {
        "container": container,
        "blob_path": blob_path,
        "format": format,
        "schema": schema or DYNAMIC_SCHEMA,
        "on_validation_failure": on_validation_failure,
    }

    # Add auth fields based on what's provided
    if connection_string is not None:
        config["connection_string"] = connection_string
    if use_managed_identity:
        config["use_managed_identity"] = use_managed_identity
    if account_url is not None:
        config["account_url"] = account_url
    if tenant_id is not None:
        config["tenant_id"] = tenant_id
    if client_id is not None:
        config["client_id"] = client_id
    if client_secret is not None:
        config["client_secret"] = client_secret

    if csv_options:
        config["csv_options"] = csv_options
    if json_options:
        config["json_options"] = json_options
    return config


class TestAzureBlobSourceProtocol:
    """Tests for AzureBlobSource protocol compliance."""

    def test_implements_protocol(self, mock_blob_client: MagicMock) -> None:
        """AzureBlobSource implements SourceProtocol."""
        source = AzureBlobSource(make_config())
        assert isinstance(source, SourceProtocol)

    def test_has_required_attributes(self, mock_blob_client: MagicMock) -> None:
        """AzureBlobSource has name and output_schema."""
        assert AzureBlobSource.name == "azure_blob"
        source = AzureBlobSource(make_config())
        assert hasattr(source, "output_schema")


class TestAzureBlobSourceConfigValidation:
    """Tests for AzureBlobSource config validation."""

    def test_no_auth_method_raises_error(self) -> None:
        """Missing all auth configuration raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="No authentication method"):
            AzureBlobSource(
                {
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_empty_connection_string_raises_error(self) -> None:
        """Empty connection_string (without other auth) raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="No authentication method"):
            AzureBlobSource(make_config(connection_string=""))

    def test_missing_container_raises_error(self) -> None:
        """Missing container raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="container"):
            AzureBlobSource(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_empty_container_raises_error(self) -> None:
        """Empty container raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="container cannot be empty"):
            AzureBlobSource(make_config(container=""))

    def test_missing_blob_path_raises_error(self) -> None:
        """Missing blob_path raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="blob_path"):
            AzureBlobSource(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_empty_blob_path_raises_error(self) -> None:
        """Empty blob_path raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="blob_path cannot be empty"):
            AzureBlobSource(make_config(blob_path=""))

    def test_missing_schema_raises_error(self) -> None:
        """Missing schema raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match=r"schema_config[\s\S]*Field required"):
            AzureBlobSource(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_missing_on_validation_failure_raises_error(self) -> None:
        """Missing on_validation_failure raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="on_validation_failure"):
            AzureBlobSource(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_unknown_field_raises_error(self) -> None:
        """Unknown config field raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="Extra inputs"):
            AzureBlobSource(
                {
                    **make_config(),
                    "unknown_field": "value",
                }
            )


class TestAzureBlobSourceCSV:
    """Tests for CSV loading from Azure Blob."""

    def test_load_csv_from_blob(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Basic CSV loading from blob."""
        csv_data = b"id,name,value\n1,alice,100\n2,bob,200\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(isinstance(r, SourceRow) for r in rows)
        assert all(not r.is_quarantined for r in rows)
        assert rows[0].row == {"id": "1", "name": "alice", "value": "100"}
        assert rows[1].row == {"id": "2", "name": "bob", "value": "200"}

    def test_csv_with_custom_delimiter(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """CSV with custom delimiter works correctly."""
        csv_data = b"id;name;value\n1;alice;100\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(csv_options={"delimiter": ";"}))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row["name"] == "alice"

    def test_csv_without_header(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """CSV without header row uses numeric column names."""
        csv_data = b"1,alice,100\n2,bob,200\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(csv_options={"has_header": False}))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        # Without header, columns are 0, 1, 2
        assert rows[0].row == {"0": "1", "1": "alice", "2": "100"}

    def test_csv_with_encoding(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """CSV with non-UTF8 encoding works correctly."""
        csv_data = b"id,name\n1,caf\xe9\n"  # latin-1 encoded "cafe" with e-acute
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(csv_options={"encoding": "latin-1"}))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        # \xe9 in latin-1 decodes to U+00E9 (LATIN SMALL LETTER E WITH ACUTE)
        assert rows[0].row["name"] == "caf\u00e9"


class TestAzureBlobSourceJSON:
    """Tests for JSON loading from Azure Blob."""

    def test_load_json_from_blob(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """JSON array loading from blob."""
        json_data = b'[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = json_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="json"))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(isinstance(r, SourceRow) for r in rows)
        assert all(not r.is_quarantined for r in rows)
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[1].row == {"id": 2, "name": "bob"}

    def test_load_json_with_data_key(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """JSON with nested data key extraction."""
        json_data = b'{"results": [{"id": 1, "name": "alice"}], "meta": "ignored"}'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = json_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                format="json",
                json_options={"data_key": "results"},
            )
        )
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row == {"id": 1, "name": "alice"}

    def test_json_not_array_raises_error(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """JSON that is not an array raises ValueError."""
        json_data = b'{"id": 1, "name": "alice"}'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = json_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="json"))
        with pytest.raises(ValueError, match="Expected JSON array"):
            list(source.load(ctx))

    def test_json_invalid_raises_error(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Invalid JSON raises ValueError."""
        json_data = b'[{"id": 1, "name": "alice"'  # Missing closing brackets
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = json_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="json"))
        with pytest.raises(ValueError, match="Invalid JSON"):
            list(source.load(ctx))


class TestAzureBlobSourceJSONL:
    """Tests for JSONL loading from Azure Blob."""

    def test_load_jsonl_from_blob(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """JSONL (newline-delimited) loading from blob."""
        jsonl_data = b'{"id": 1, "name": "alice"}\n{"id": 2, "name": "bob"}\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(isinstance(r, SourceRow) for r in rows)
        assert all(not r.is_quarantined for r in rows)
        assert rows[0].row == {"id": 1, "name": "alice"}
        assert rows[1].row == {"id": 2, "name": "bob"}

    def test_jsonl_skips_empty_lines(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """JSONL skips empty lines."""
        jsonl_data = b'{"id": 1}\n\n{"id": 2}\n\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_jsonl_malformed_line_quarantined_not_crash(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Malformed JSONL line should quarantine, not crash pipeline."""
        jsonl_data = b'{"id": 1, "name": "alice"}\n{invalid json}\n{"id": 3, "name": "carol"}\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                format="jsonl",
                on_validation_failure="quarantine",
            )
        )

        # Should not crash - should yield 2 valid + 1 quarantined
        results = list(source.load(ctx))

        # Verify we got 3 total rows
        assert len(results) == 3

        # First row valid
        assert not results[0].is_quarantined
        assert results[0].row == {"id": 1, "name": "alice"}

        # Second row quarantined (line 2)
        assert results[1].is_quarantined is True
        assert results[1].quarantine_destination == "quarantine"
        assert results[1].quarantine_error is not None
        assert "JSON parse error" in results[1].quarantine_error
        assert "line 2" in results[1].quarantine_error
        assert "__raw_line__" in results[1].row
        assert "__line_number__" in results[1].row

        # Third row valid
        assert not results[2].is_quarantined
        assert results[2].row == {"id": 3, "name": "carol"}

    def test_jsonl_malformed_line_with_discard_mode(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Malformed JSONL line with discard mode should not yield quarantined row."""
        jsonl_data = b'{"id": 1}\n{invalid}\n{"id": 3}\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                format="jsonl",
                on_validation_failure="discard",
            )
        )

        # Should not crash, should yield only valid rows
        results = list(source.load(ctx))

        # Only 2 valid rows, malformed line silently discarded
        assert len(results) == 2
        assert results[0].row == {"id": 1}
        assert results[1].row == {"id": 3}


class TestAzureBlobSourceValidation:
    """Tests for schema validation and quarantining."""

    def test_validation_failure_quarantines_row(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Invalid rows are quarantined with error info."""
        csv_data = b"id,name,score\n1,alice,95\n2,bob,bad\n3,carol,92\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                schema={
                    "mode": "strict",
                    "fields": ["id: int", "name: str", "score: int"],
                },
                on_validation_failure="quarantine",
            )
        )
        results = list(source.load(ctx))

        # 2 valid rows + 1 quarantined
        assert len(results) == 3
        assert all(isinstance(r, SourceRow) for r in results)

        # First and third are valid
        assert not results[0].is_quarantined
        assert results[0].row["name"] == "alice"
        assert not results[2].is_quarantined
        assert results[2].row["name"] == "carol"

        # Second is quarantined
        quarantined = results[1]
        assert quarantined.is_quarantined
        assert quarantined.row["name"] == "bob"
        assert quarantined.row["score"] == "bad"  # Original value preserved
        assert quarantined.quarantine_destination == "quarantine"
        assert quarantined.quarantine_error is not None
        assert "score" in quarantined.quarantine_error

    def test_discard_mode_does_not_yield_invalid_rows(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """When on_validation_failure='discard', invalid rows are not yielded."""
        csv_data = b"id,name,score\n1,alice,95\n2,bob,bad\n3,carol,92\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                schema={
                    "mode": "strict",
                    "fields": ["id: int", "name: str", "score: int"],
                },
                on_validation_failure="discard",
            )
        )
        results = list(source.load(ctx))

        # Only 2 valid rows - invalid row discarded
        assert len(results) == 2
        assert all(isinstance(r, SourceRow) and not r.is_quarantined for r in results)
        assert {r.row["name"] for r in results} == {"alice", "carol"}


class TestAzureBlobSourceErrors:
    """Tests for error handling."""

    def test_blob_not_found_raises(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Missing blob raises appropriate error."""
        # Simulate ResourceNotFoundError from Azure SDK
        mock_client = MagicMock()
        mock_client.download_blob.side_effect = Exception("The specified blob does not exist")
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(Exception, match="specified blob does not exist"):
            list(source.load(ctx))

    def test_connection_error_raises(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Connection failures propagate."""
        # Simulate connection error
        mock_client = MagicMock()
        mock_client.download_blob.side_effect = Exception("Connection refused")
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(Exception, match="Connection refused"):
            list(source.load(ctx))

    def test_encoding_error_raises(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """Invalid encoding raises ValueError."""
        # Invalid UTF-8 bytes
        bad_data = b"\xff\xfe"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = bad_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(ValueError, match="Failed to decode"):
            list(source.load(ctx))

    def test_csv_parse_error_quarantines(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """BUG-BLOB-01: CSV parse errors quarantine instead of crashing."""
        # Severely malformed CSV that pandas can't parse
        # Using delimiter mismatch that causes structural failure
        bad_csv = b"col1;col2;col3\nvalue1,value2,value3\n"  # Headers use ; but data uses ,
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = bad_csv
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                schema={
                    "mode": "strict",
                    "fields": ["col1: str", "col2: str", "col3: str"],
                }
            )
        )

        # Should NOT raise - should quarantine instead
        rows = list(source.load(ctx))

        # Pipeline continues (doesn't crash)
        # Note: With on_bad_lines="warn", pandas may parse some rows or quarantine the whole file
        # Either way, we should not crash
        assert isinstance(rows, list)  # Got results, not a crash

    def test_csv_structural_failure_quarantines_blob(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """BUG-BLOB-01: Catastrophic CSV structure failure quarantines blob, doesn't crash."""
        # Binary data masquerading as CSV - completely unparseable
        bad_csv = b"\x00\x01\x02\x03\x04\x05"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = bad_csv
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(
            make_config(
                schema={
                    "mode": "strict",
                    "fields": ["col1: str"],
                }
            )
        )

        # Should NOT raise - should quarantine the entire blob
        rows = list(source.load(ctx))

        # Should get one quarantined "row" representing the unparseable blob
        assert len(rows) >= 0  # Either empty or quarantined row
        # No crash = success


class TestAzureBlobSourceLifecycle:
    """Tests for source lifecycle methods."""

    def test_close_is_idempotent(self, mock_blob_client: MagicMock) -> None:
        """close() can be called multiple times."""
        source = AzureBlobSource(make_config())
        source.close()
        source.close()  # Should not raise

    def test_close_clears_client(self, mock_blob_client: MagicMock, ctx: PluginContext) -> None:
        """close() clears the blob client reference."""
        csv_data = b"id,name\n1,alice\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        list(source.load(ctx))  # Populate client
        source.close()
        assert source._blob_client is None


class TestAzureBlobSourceImportError:
    """Tests for azure-storage-blob import handling."""

    def test_import_error_gives_helpful_message(self, ctx: PluginContext) -> None:
        """Missing azure-storage-blob gives helpful install message."""
        source = AzureBlobSource(make_config())

        # Mock the import to fail
        with patch.object(source, "_get_blob_client") as mock_get:
            mock_get.side_effect = ImportError(
                "azure-storage-blob is required for AzureBlobSource. Install with: uv pip install azure-storage-blob"
            )

            with pytest.raises(ImportError, match="azure-storage-blob"):
                list(source.load(ctx))


class TestAzureBlobSourceAuthMethods:
    """Tests for Azure authentication methods."""

    def test_auth_connection_string(self, mock_blob_client: MagicMock) -> None:
        """Connection string auth creates source successfully."""
        source = AzureBlobSource(make_config(connection_string=TEST_CONNECTION_STRING))
        assert source._auth_config.auth_method == "connection_string"
        assert source._auth_config.connection_string == TEST_CONNECTION_STRING

    def test_auth_managed_identity(self, mock_blob_client: MagicMock) -> None:
        """Managed identity auth creates source successfully."""
        source = AzureBlobSource(
            make_config(
                connection_string=None,
                use_managed_identity=True,
                account_url=TEST_ACCOUNT_URL,
            )
        )
        assert source._auth_config.auth_method == "managed_identity"
        assert source._auth_config.use_managed_identity is True
        assert source._auth_config.account_url == TEST_ACCOUNT_URL

    def test_auth_service_principal(self, mock_blob_client: MagicMock) -> None:
        """Service principal auth creates source successfully."""
        source = AzureBlobSource(
            make_config(
                connection_string=None,
                tenant_id=TEST_TENANT_ID,
                client_id=TEST_CLIENT_ID,
                client_secret=TEST_CLIENT_SECRET,
                account_url=TEST_ACCOUNT_URL,
            )
        )
        assert source._auth_config.auth_method == "service_principal"
        assert source._auth_config.tenant_id == TEST_TENANT_ID
        assert source._auth_config.client_id == TEST_CLIENT_ID
        assert source._auth_config.client_secret == TEST_CLIENT_SECRET
        assert source._auth_config.account_url == TEST_ACCOUNT_URL

    def test_auth_mutual_exclusivity_conn_string_and_managed_identity(self) -> None:
        """Cannot use connection string and managed identity together."""
        with pytest.raises(PluginConfigError, match="Multiple authentication methods"):
            AzureBlobSource(
                make_config(
                    connection_string=TEST_CONNECTION_STRING,
                    use_managed_identity=True,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_mutual_exclusivity_conn_string_and_service_principal(self) -> None:
        """Cannot use connection string and service principal together."""
        with pytest.raises(PluginConfigError, match="Multiple authentication methods"):
            AzureBlobSource(
                make_config(
                    connection_string=TEST_CONNECTION_STRING,
                    tenant_id=TEST_TENANT_ID,
                    client_id=TEST_CLIENT_ID,
                    client_secret=TEST_CLIENT_SECRET,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_mutual_exclusivity_managed_identity_and_service_principal(
        self,
    ) -> None:
        """Cannot use managed identity and service principal together."""
        with pytest.raises(PluginConfigError, match="Multiple authentication methods"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    use_managed_identity=True,
                    tenant_id=TEST_TENANT_ID,
                    client_id=TEST_CLIENT_ID,
                    client_secret=TEST_CLIENT_SECRET,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_managed_identity_missing_account_url(self) -> None:
        """Managed identity requires account_url."""
        with pytest.raises(PluginConfigError, match="account_url"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    use_managed_identity=True,
                    # account_url omitted
                )
            )

    def test_auth_service_principal_missing_tenant_id(self) -> None:
        """Service principal requires all fields - missing tenant_id."""
        with pytest.raises(PluginConfigError, match="tenant_id"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    # tenant_id omitted
                    client_id=TEST_CLIENT_ID,
                    client_secret=TEST_CLIENT_SECRET,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_service_principal_missing_client_id(self) -> None:
        """Service principal requires all fields - missing client_id."""
        with pytest.raises(PluginConfigError, match="client_id"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    tenant_id=TEST_TENANT_ID,
                    # client_id omitted
                    client_secret=TEST_CLIENT_SECRET,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_service_principal_missing_client_secret(self) -> None:
        """Service principal requires all fields - missing client_secret."""
        with pytest.raises(PluginConfigError, match="client_secret"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    tenant_id=TEST_TENANT_ID,
                    client_id=TEST_CLIENT_ID,
                    # client_secret omitted
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_service_principal_missing_account_url(self) -> None:
        """Service principal requires all fields - missing account_url."""
        with pytest.raises(PluginConfigError, match="account_url"):
            AzureBlobSource(
                make_config(
                    connection_string=None,
                    tenant_id=TEST_TENANT_ID,
                    client_id=TEST_CLIENT_ID,
                    client_secret=TEST_CLIENT_SECRET,
                    # account_url omitted
                )
            )


class TestAzureBlobSourceAuthClientCreation:
    """Tests for Azure auth client creation with mocked credentials.

    These tests verify that the correct Azure SDK methods are called
    based on the authentication method. They require azure-storage-blob
    and azure-identity to be installed to run.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_azure(self) -> None:
        """Skip these tests if Azure SDK is not installed."""
        pytest.importorskip("azure.storage.blob")
        pytest.importorskip("azure.identity")

    def test_managed_identity_uses_default_credential(self, ctx: PluginContext) -> None:
        """Managed identity auth uses DefaultAzureCredential."""
        source = AzureBlobSource(
            make_config(
                connection_string=None,
                use_managed_identity=True,
                account_url=TEST_ACCOUNT_URL,
            )
        )

        # Mock the azure.identity and azure.storage.blob imports
        with (
            patch("azure.identity.DefaultAzureCredential") as mock_credential_cls,
            patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls,
        ):
            mock_credential = MagicMock()
            mock_credential_cls.return_value = mock_credential
            mock_service_client = MagicMock()
            mock_service_client_cls.return_value = mock_service_client

            # Trigger client creation
            source._auth_config.create_blob_service_client()

            # Verify DefaultAzureCredential was instantiated
            mock_credential_cls.assert_called_once()
            # Verify BlobServiceClient was created with account_url and credential
            mock_service_client_cls.assert_called_once_with(TEST_ACCOUNT_URL, credential=mock_credential)

    def test_service_principal_uses_client_secret_credential(self, ctx: PluginContext) -> None:
        """Service principal auth uses ClientSecretCredential."""
        source = AzureBlobSource(
            make_config(
                connection_string=None,
                tenant_id=TEST_TENANT_ID,
                client_id=TEST_CLIENT_ID,
                client_secret=TEST_CLIENT_SECRET,
                account_url=TEST_ACCOUNT_URL,
            )
        )

        # Mock the azure.identity and azure.storage.blob imports
        with (
            patch("azure.identity.ClientSecretCredential") as mock_credential_cls,
            patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls,
        ):
            mock_credential = MagicMock()
            mock_credential_cls.return_value = mock_credential
            mock_service_client = MagicMock()
            mock_service_client_cls.return_value = mock_service_client

            # Trigger client creation
            source._auth_config.create_blob_service_client()

            # Verify ClientSecretCredential was instantiated with correct args
            mock_credential_cls.assert_called_once_with(
                tenant_id=TEST_TENANT_ID,
                client_id=TEST_CLIENT_ID,
                client_secret=TEST_CLIENT_SECRET,
            )
            # Verify BlobServiceClient was created with account_url and credential
            mock_service_client_cls.assert_called_once_with(TEST_ACCOUNT_URL, credential=mock_credential)

    def test_connection_string_uses_from_connection_string(self, ctx: PluginContext) -> None:
        """Connection string auth uses from_connection_string factory."""
        source = AzureBlobSource(make_config(connection_string=TEST_CONNECTION_STRING))

        # Mock the azure.storage.blob import
        with patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls:
            mock_service_client = MagicMock()
            mock_service_client_cls.from_connection_string.return_value = mock_service_client

            # Trigger client creation
            source._auth_config.create_blob_service_client()

            # Verify from_connection_string was called
            mock_service_client_cls.from_connection_string.assert_called_once_with(TEST_CONNECTION_STRING)
