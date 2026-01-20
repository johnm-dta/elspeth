"""Tests for Azure Blob Storage sink plugin."""

import hashlib
from collections.abc import Generator
from typing import Any
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

# Managed Identity test values
TEST_ACCOUNT_URL = "https://mystorageaccount.blob.core.windows.net"

# Service Principal test values
TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"
TEST_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
TEST_CLIENT_SECRET = "test-secret-value"


@pytest.fixture
def ctx() -> PluginContext:
    """Create a minimal plugin context."""
    return PluginContext(run_id="test-run-123", config={})


@pytest.fixture
def mock_container_client() -> Generator[MagicMock, None, None]:
    """Create a mock container client for testing."""
    with patch(
        "elspeth.plugins.azure.blob_sink.AzureBlobSink._get_container_client"
    ) as mock:
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
    overwrite: bool = True,
    csv_options: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to create config dicts with defaults.

    By default uses connection_string auth. Pass connection_string=None
    and set other auth options for managed identity or service principal.
    """
    config: dict[str, Any] = {
        "container": container,
        "blob_path": blob_path,
        "format": format,
        "overwrite": overwrite,
        "schema": schema or DYNAMIC_SCHEMA,
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

    def test_no_auth_method_raises_error(self) -> None:
        """Missing all auth configuration raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="No authentication method"):
            AzureBlobSink(
                {
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_empty_connection_string_raises_error(self) -> None:
        """Empty connection_string (without other auth) raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="No authentication method"):
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
        rows: list[dict[str, Any]] = []

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


class TestAzureBlobSinkAuthMethods:
    """Tests for Azure authentication methods."""

    def test_auth_connection_string(self, mock_container_client: MagicMock) -> None:
        """Connection string auth creates sink successfully."""
        sink = AzureBlobSink(make_config(connection_string=TEST_CONNECTION_STRING))
        assert sink._auth_config.auth_method == "connection_string"
        assert sink._auth_config.connection_string == TEST_CONNECTION_STRING

    def test_auth_managed_identity(self, mock_container_client: MagicMock) -> None:
        """Managed identity auth creates sink successfully."""
        sink = AzureBlobSink(
            make_config(
                connection_string=None,
                use_managed_identity=True,
                account_url=TEST_ACCOUNT_URL,
            )
        )
        assert sink._auth_config.auth_method == "managed_identity"
        assert sink._auth_config.use_managed_identity is True
        assert sink._auth_config.account_url == TEST_ACCOUNT_URL

    def test_auth_service_principal(self, mock_container_client: MagicMock) -> None:
        """Service principal auth creates sink successfully."""
        sink = AzureBlobSink(
            make_config(
                connection_string=None,
                tenant_id=TEST_TENANT_ID,
                client_id=TEST_CLIENT_ID,
                client_secret=TEST_CLIENT_SECRET,
                account_url=TEST_ACCOUNT_URL,
            )
        )
        assert sink._auth_config.auth_method == "service_principal"
        assert sink._auth_config.tenant_id == TEST_TENANT_ID
        assert sink._auth_config.client_id == TEST_CLIENT_ID
        assert sink._auth_config.client_secret == TEST_CLIENT_SECRET
        assert sink._auth_config.account_url == TEST_ACCOUNT_URL

    def test_auth_mutual_exclusivity_conn_string_and_managed_identity(self) -> None:
        """Cannot use connection string and managed identity together."""
        with pytest.raises(PluginConfigError, match="Multiple authentication methods"):
            AzureBlobSink(
                make_config(
                    connection_string=TEST_CONNECTION_STRING,
                    use_managed_identity=True,
                    account_url=TEST_ACCOUNT_URL,
                )
            )

    def test_auth_mutual_exclusivity_conn_string_and_service_principal(self) -> None:
        """Cannot use connection string and service principal together."""
        with pytest.raises(PluginConfigError, match="Multiple authentication methods"):
            AzureBlobSink(
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
            AzureBlobSink(
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
            AzureBlobSink(
                make_config(
                    connection_string=None,
                    use_managed_identity=True,
                    # account_url omitted
                )
            )

    def test_auth_service_principal_missing_tenant_id(self) -> None:
        """Service principal requires all fields - missing tenant_id."""
        with pytest.raises(PluginConfigError, match="tenant_id"):
            AzureBlobSink(
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
            AzureBlobSink(
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
            AzureBlobSink(
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
            AzureBlobSink(
                make_config(
                    connection_string=None,
                    tenant_id=TEST_TENANT_ID,
                    client_id=TEST_CLIENT_ID,
                    client_secret=TEST_CLIENT_SECRET,
                    # account_url omitted
                )
            )


class TestAzureBlobSinkAuthClientCreation:
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
        sink = AzureBlobSink(
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
            sink._auth_config.create_blob_service_client()

            # Verify DefaultAzureCredential was instantiated
            mock_credential_cls.assert_called_once()
            # Verify BlobServiceClient was created with account_url and credential
            mock_service_client_cls.assert_called_once_with(
                TEST_ACCOUNT_URL, credential=mock_credential
            )

    def test_service_principal_uses_client_secret_credential(
        self, ctx: PluginContext
    ) -> None:
        """Service principal auth uses ClientSecretCredential."""
        sink = AzureBlobSink(
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
            sink._auth_config.create_blob_service_client()

            # Verify ClientSecretCredential was instantiated with correct args
            mock_credential_cls.assert_called_once_with(
                tenant_id=TEST_TENANT_ID,
                client_id=TEST_CLIENT_ID,
                client_secret=TEST_CLIENT_SECRET,
            )
            # Verify BlobServiceClient was created with account_url and credential
            mock_service_client_cls.assert_called_once_with(
                TEST_ACCOUNT_URL, credential=mock_credential
            )

    def test_connection_string_uses_from_connection_string(
        self, ctx: PluginContext
    ) -> None:
        """Connection string auth uses from_connection_string factory."""
        sink = AzureBlobSink(make_config(connection_string=TEST_CONNECTION_STRING))

        # Mock the azure.storage.blob import
        with patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls:
            mock_service_client = MagicMock()
            mock_service_client_cls.from_connection_string.return_value = (
                mock_service_client
            )

            # Trigger client creation
            sink._auth_config.create_blob_service_client()

            # Verify from_connection_string was called
            mock_service_client_cls.from_connection_string.assert_called_once_with(
                TEST_CONNECTION_STRING
            )
