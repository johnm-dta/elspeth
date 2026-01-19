"""Tests for Azure Blob Storage sink plugin."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import ArtifactDescriptor
from elspeth.plugins.azure.blob_sink import AzureBlobSink
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol

# Dynamic schema config for tests - DataPluginConfig requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}

# Standard connection string for tests
TEST_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key"
)
TEST_CONTAINER = "output-container"
TEST_BLOB_PATH = "results/output.csv"


@pytest.fixture
def ctx() -> PluginContext:
    """Create a minimal plugin context."""
    return PluginContext(run_id="test-run-123", config={})


@pytest.fixture
def mock_container_client():
    """Create a mock container client for testing."""
    with patch(
        "elspeth.plugins.azure.blob_sink.AzureBlobSink._get_container_client"
    ) as mock:
        yield mock


def make_config(
    *,
    connection_string: str = TEST_CONNECTION_STRING,
    container: str = TEST_CONTAINER,
    blob_path: str = TEST_BLOB_PATH,
    format: str = "csv",
    overwrite: bool = True,
    csv_options: dict | None = None,
    schema: dict | None = None,
) -> dict:
    """Helper to create config dicts with defaults."""
    config: dict = {
        "connection_string": connection_string,
        "container": container,
        "blob_path": blob_path,
        "format": format,
        "overwrite": overwrite,
        "schema": schema or DYNAMIC_SCHEMA,
    }
    if csv_options:
        config["csv_options"] = csv_options
    return config


class TestAzureBlobSinkProtocol:
    """Tests for AzureBlobSink protocol compliance."""

    def test_implements_protocol(self, mock_container_client: MagicMock) -> None:
        """AzureBlobSink implements SinkProtocol."""
        sink = AzureBlobSink(make_config())
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self, mock_container_client: MagicMock) -> None:
        """AzureBlobSink has name and input_schema."""
        assert AzureBlobSink.name == "azure_blob"
        sink = AzureBlobSink(make_config())
        assert hasattr(sink, "input_schema")


class TestAzureBlobSinkConfigValidation:
    """Tests for AzureBlobSink config validation."""

    def test_missing_connection_string_raises_error(self) -> None:
        """Missing connection_string raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="connection_string"):
            AzureBlobSink(
                {
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_empty_connection_string_raises_error(self) -> None:
        """Empty connection_string raises PluginConfigError."""
        with pytest.raises(
            PluginConfigError, match="connection_string cannot be empty"
        ):
            AzureBlobSink(make_config(connection_string=""))

    def test_missing_container_raises_error(self) -> None:
        """Missing container raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="container"):
            AzureBlobSink(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_empty_container_raises_error(self) -> None:
        """Empty container raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="container cannot be empty"):
            AzureBlobSink(make_config(container=""))

    def test_missing_blob_path_raises_error(self) -> None:
        """Missing blob_path raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="blob_path"):
            AzureBlobSink(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_empty_blob_path_raises_error(self) -> None:
        """Empty blob_path raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="blob_path cannot be empty"):
            AzureBlobSink(make_config(blob_path=""))

    def test_missing_schema_raises_error(self) -> None:
        """Missing schema raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match=r"require.*schema"):
            AzureBlobSink(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                }
            )

    def test_unknown_field_raises_error(self) -> None:
        """Unknown config field raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="Extra inputs"):
            AzureBlobSink(
                {
                    **make_config(),
                    "unknown_field": "value",
                }
            )


class TestAzureBlobSinkWriteCSV:
    """Tests for CSV writing to Azure Blob."""

    def test_write_csv_to_blob(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Basic CSV writing to blob."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config())
        rows = [
            {"id": 1, "name": "alice", "value": 100},
            {"id": 2, "name": "bob", "value": 200},
        ]

        result = sink.write(rows, ctx)

        # Verify blob_client.upload_blob was called
        mock_blob_client.upload_blob.assert_called_once()
        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]

        # Verify CSV content
        assert b"id,name,value" in uploaded_content  # header
        assert b"1,alice,100" in uploaded_content
        assert b"2,bob,200" in uploaded_content

        # Verify returns ArtifactDescriptor
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"
        assert result.content_hash == hashlib.sha256(uploaded_content).hexdigest()
        assert result.size_bytes == len(uploaded_content)

    def test_csv_with_custom_delimiter(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """CSV with custom delimiter works correctly."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(csv_options={"delimiter": ";"}))
        rows = [{"id": 1, "name": "alice"}]

        sink.write(rows, ctx)

        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]
        assert b"id;name" in uploaded_content
        assert b"1;alice" in uploaded_content

    def test_csv_without_header(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """CSV without header row when include_header=False."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(csv_options={"include_header": False}))
        rows = [{"id": 1, "name": "alice"}]

        sink.write(rows, ctx)

        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]
        # Should NOT have header
        lines = uploaded_content.decode().strip().split("\n")
        assert len(lines) == 1
        assert "1,alice" in lines[0]


class TestAzureBlobSinkWriteJSON:
    """Tests for JSON writing to Azure Blob."""

    def test_write_json_to_blob(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """JSON array writing to blob."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(format="json"))
        rows = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]

        result = sink.write(rows, ctx)

        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]

        # Verify JSON content (should be pretty-printed array)
        import json

        parsed = json.loads(uploaded_content.decode())
        assert parsed == rows

        # Verify ArtifactDescriptor
        assert isinstance(result, ArtifactDescriptor)
        assert result.content_hash == hashlib.sha256(uploaded_content).hexdigest()


class TestAzureBlobSinkWriteJSONL:
    """Tests for JSONL writing to Azure Blob."""

    def test_write_jsonl_to_blob(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """JSONL (newline-delimited) writing to blob."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(format="jsonl"))
        rows = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]

        result = sink.write(rows, ctx)

        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]

        # Verify JSONL content
        lines = uploaded_content.decode().strip().split("\n")
        assert len(lines) == 2

        import json

        assert json.loads(lines[0]) == {"id": 1, "name": "alice"}
        assert json.loads(lines[1]) == {"id": 2, "name": "bob"}

        # Verify ArtifactDescriptor
        assert isinstance(result, ArtifactDescriptor)


class TestAzureBlobSinkPathTemplating:
    """Tests for Jinja2 path templating."""

    def test_blob_path_with_run_id_template(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Blob path with {{ run_id }} template renders correctly."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(blob_path="results/{{ run_id }}/output.csv"))
        rows = [{"id": 1, "name": "alice"}]

        result = sink.write(rows, ctx)

        # Verify rendered path was used
        mock_container.get_blob_client.assert_called_once()
        rendered_path = mock_container.get_blob_client.call_args[0][0]
        assert rendered_path == "results/test-run-123/output.csv"

        # Verify artifact descriptor uses rendered path
        assert "test-run-123" in result.path_or_uri

    def test_blob_path_with_timestamp_template(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Blob path with {{ timestamp }} template renders correctly."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(
            make_config(blob_path="results/{{ timestamp }}/output.csv")
        )
        rows = [{"id": 1, "name": "alice"}]

        sink.write(rows, ctx)

        # Verify rendered path contains timestamp-like string
        rendered_path = mock_container.get_blob_client.call_args[0][0]
        # Timestamp should look like 2024-01-15T... (ISO format)
        assert rendered_path.startswith("results/20")
        assert "T" in rendered_path  # ISO format has T separator


class TestAzureBlobSinkOverwriteBehavior:
    """Tests for overwrite behavior."""

    def test_overwrite_true_succeeds_when_blob_exists(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """With overwrite=True, writing succeeds even if blob exists."""
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = True
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(overwrite=True))
        rows = [{"id": 1, "name": "alice"}]

        # Should not raise
        result = sink.write(rows, ctx)
        assert isinstance(result, ArtifactDescriptor)

        # Should upload with overwrite=True
        mock_blob_client.upload_blob.assert_called_once()
        call_kwargs = mock_blob_client.upload_blob.call_args[1]
        assert call_kwargs["overwrite"] is True

    def test_overwrite_false_raises_if_blob_exists(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """With overwrite=False, raises ValueError if blob exists."""
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = True
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(overwrite=False))
        rows = [{"id": 1, "name": "alice"}]

        with pytest.raises(ValueError, match="already exists"):
            sink.write(rows, ctx)

    def test_overwrite_false_succeeds_if_blob_not_exists(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """With overwrite=False, succeeds if blob does not exist."""
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = False
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(overwrite=False))
        rows = [{"id": 1, "name": "alice"}]

        result = sink.write(rows, ctx)
        assert isinstance(result, ArtifactDescriptor)


class TestAzureBlobSinkArtifactDescriptor:
    """Tests for ArtifactDescriptor correctness."""

    def test_returns_artifact_descriptor_with_hash(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Write returns ArtifactDescriptor with correct content hash."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config())
        rows = [{"id": 1, "name": "alice"}]

        result = sink.write(rows, ctx)

        # Get the actual uploaded content to verify hash
        uploaded_content = mock_blob_client.upload_blob.call_args[0][0]
        expected_hash = hashlib.sha256(uploaded_content).hexdigest()

        assert result.artifact_type == "file"
        assert result.content_hash == expected_hash
        assert result.size_bytes == len(uploaded_content)
        assert "azure://" in result.path_or_uri
        assert TEST_CONTAINER in result.path_or_uri

    def test_artifact_descriptor_contains_rendered_path(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """ArtifactDescriptor path_or_uri contains rendered blob path."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config(blob_path="data/{{ run_id }}/file.csv"))
        rows = [{"id": 1}]

        result = sink.write(rows, ctx)

        # Should contain rendered path, not template
        assert "{{ run_id }}" not in result.path_or_uri
        assert "test-run-123" in result.path_or_uri


class TestAzureBlobSinkEmptyRows:
    """Tests for empty rows edge case."""

    def test_empty_rows_returns_empty_descriptor(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Empty rows list returns descriptor with empty content hash."""
        sink = AzureBlobSink(make_config())
        rows: list[dict] = []

        result = sink.write(rows, ctx)

        # Should return descriptor without uploading
        mock_container_client.assert_not_called()

        assert isinstance(result, ArtifactDescriptor)
        assert result.content_hash == hashlib.sha256(b"").hexdigest()
        assert result.size_bytes == 0


class TestAzureBlobSinkErrors:
    """Tests for error handling."""

    def test_upload_error_propagates_with_context(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Azure upload errors propagate with context message."""
        mock_blob_client = MagicMock()
        mock_blob_client.upload_blob.side_effect = Exception("Network error")
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config())
        rows = [{"id": 1}]

        with pytest.raises(Exception, match="Failed to upload blob"):
            sink.write(rows, ctx)

    def test_connection_error_propagates(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """Connection errors propagate to caller."""
        mock_container_client.side_effect = Exception("Connection refused")

        sink = AzureBlobSink(make_config())
        rows = [{"id": 1}]

        with pytest.raises(Exception, match="Connection refused"):
            sink.write(rows, ctx)


class TestAzureBlobSinkLifecycle:
    """Tests for sink lifecycle methods."""

    def test_close_is_idempotent(self, mock_container_client: MagicMock) -> None:
        """close() can be called multiple times."""
        sink = AzureBlobSink(make_config())
        sink.close()
        sink.close()  # Should not raise

    def test_close_clears_client(
        self, mock_container_client: MagicMock, ctx: PluginContext
    ) -> None:
        """close() clears the container client reference."""
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_container_client.return_value = mock_container

        sink = AzureBlobSink(make_config())
        sink.write([{"id": 1}], ctx)  # Populate client
        sink.close()
        assert sink._container_client is None

    def test_flush_is_noop(self, mock_container_client: MagicMock) -> None:
        """flush() is a no-op (uploads are synchronous)."""
        sink = AzureBlobSink(make_config())
        sink.flush()  # Should not raise


class TestAzureBlobSinkImportError:
    """Tests for azure-storage-blob import handling."""

    def test_import_error_gives_helpful_message(self, ctx: PluginContext) -> None:
        """Missing azure-storage-blob gives helpful install message."""
        sink = AzureBlobSink(make_config())

        # Mock the import to fail
        with patch.object(sink, "_get_container_client") as mock_get:
            mock_get.side_effect = ImportError(
                "azure-storage-blob is required for AzureBlobSink. "
                "Install with: uv pip install azure-storage-blob"
            )

            with pytest.raises(ImportError, match="azure-storage-blob"):
                sink.write([{"id": 1}], ctx)
