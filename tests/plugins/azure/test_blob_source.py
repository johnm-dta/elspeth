"""Tests for Azure Blob Storage source plugin."""

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
TEST_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key"
)
TEST_CONTAINER = "test-container"
TEST_BLOB_PATH = "data/input.csv"


@pytest.fixture
def ctx() -> PluginContext:
    """Create a minimal plugin context."""
    return PluginContext(run_id="test-run", config={})


@pytest.fixture
def mock_blob_client():
    """Create a mock blob client for testing."""
    with patch(
        "elspeth.plugins.azure.blob_source.AzureBlobSource._get_blob_client"
    ) as mock:
        yield mock


def make_config(
    *,
    connection_string: str = TEST_CONNECTION_STRING,
    container: str = TEST_CONTAINER,
    blob_path: str = TEST_BLOB_PATH,
    format: str = "csv",
    csv_options: dict | None = None,
    json_options: dict | None = None,
    schema: dict | None = None,
    on_validation_failure: str = QUARANTINE_SINK,
) -> dict:
    """Helper to create config dicts with defaults."""
    config: dict = {
        "connection_string": connection_string,
        "container": container,
        "blob_path": blob_path,
        "format": format,
        "schema": schema or DYNAMIC_SCHEMA,
        "on_validation_failure": on_validation_failure,
    }
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

    def test_missing_connection_string_raises_error(self) -> None:
        """Missing connection_string raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="connection_string"):
            AzureBlobSource(
                {
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                    "on_validation_failure": QUARANTINE_SINK,
                }
            )

    def test_empty_connection_string_raises_error(self) -> None:
        """Empty connection_string raises PluginConfigError."""
        with pytest.raises(
            PluginConfigError, match="connection_string cannot be empty"
        ):
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
        with pytest.raises(PluginConfigError, match=r"require.*schema"):
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

    def test_load_csv_from_blob(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_csv_with_custom_delimiter(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """CSV with custom delimiter works correctly."""
        csv_data = b"id;name;value\n1;alice;100\n"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = csv_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(csv_options={"delimiter": ";"}))
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].row["name"] == "alice"

    def test_csv_without_header(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_csv_with_encoding(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_load_json_from_blob(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_load_json_with_data_key(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_json_not_array_raises_error(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """JSON that is not an array raises ValueError."""
        json_data = b'{"id": 1, "name": "alice"}'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = json_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="json"))
        with pytest.raises(ValueError, match="Expected JSON array"):
            list(source.load(ctx))

    def test_json_invalid_raises_error(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_load_jsonl_from_blob(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_jsonl_skips_empty_lines(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """JSONL skips empty lines."""
        jsonl_data = b'{"id": 1}\n\n{"id": 2}\n\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="jsonl"))
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_jsonl_invalid_line_raises_error(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Invalid JSON line in JSONL raises ValueError with line number."""
        jsonl_data = b'{"id": 1}\n{invalid}\n{"id": 3}\n'
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = jsonl_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config(format="jsonl"))
        with pytest.raises(ValueError, match="line 2"):
            list(source.load(ctx))


class TestAzureBlobSourceValidation:
    """Tests for schema validation and quarantining."""

    def test_validation_failure_quarantines_row(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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
        assert "score" in quarantined.quarantine_error

    def test_discard_mode_does_not_yield_invalid_rows(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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

    def test_blob_not_found_raises(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Missing blob raises appropriate error."""
        # Simulate ResourceNotFoundError from Azure SDK
        mock_client = MagicMock()
        mock_client.download_blob.side_effect = Exception(
            "The specified blob does not exist"
        )
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(Exception, match="specified blob does not exist"):
            list(source.load(ctx))

    def test_connection_error_raises(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Connection failures propagate."""
        # Simulate connection error
        mock_client = MagicMock()
        mock_client.download_blob.side_effect = Exception("Connection refused")
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(Exception, match="Connection refused"):
            list(source.load(ctx))

    def test_encoding_error_raises(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Invalid encoding raises ValueError."""
        # Invalid UTF-8 bytes
        bad_data = b"\xff\xfe"
        mock_client = MagicMock()
        mock_client.download_blob.return_value.readall.return_value = bad_data
        mock_blob_client.return_value = mock_client

        source = AzureBlobSource(make_config())
        with pytest.raises(ValueError, match="Failed to decode"):
            list(source.load(ctx))


class TestAzureBlobSourceLifecycle:
    """Tests for source lifecycle methods."""

    def test_close_is_idempotent(self, mock_blob_client: MagicMock) -> None:
        """close() can be called multiple times."""
        source = AzureBlobSource(make_config())
        source.close()
        source.close()  # Should not raise

    def test_close_clears_client(
        self, mock_blob_client: MagicMock, ctx: PluginContext
    ) -> None:
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
                "azure-storage-blob is required for AzureBlobSource. "
                "Install with: uv pip install azure-storage-blob"
            )

            with pytest.raises(ImportError, match="azure-storage-blob"):
                list(source.load(ctx))
