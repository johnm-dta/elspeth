"""Tests for Azure Blob Storage sink plugin."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context

# Dynamic schema config for tests - DataPluginConfig requires schema
DYNAMIC_SCHEMA = {"mode": "observed"}

# Standard connection string for tests
TEST_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key"
TEST_CONTAINER = "output-container"
TEST_BLOB_PATH = "results/output.csv"

# Managed Identity test values
TEST_ACCOUNT_URL = "https://mystorageaccount.blob.core.windows.net"

# Service Principal test values
TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"
TEST_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
TEST_CLIENT_SECRET = "test-secret-value"


@dataclass(frozen=True)
class _RecordedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.args
        if index == 1:
            return self.kwargs
        raise IndexError(index)


class _CallRecorder:
    def __init__(self, *, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args_list: list[_RecordedCall] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args_list.append(_RecordedCall(args=args, kwargs=kwargs))
        if self.side_effect is None:
            return self.return_value

        effect = self.side_effect
        if isinstance(effect, list):
            effect = effect.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, type) and issubclass(effect, BaseException):
            raise effect()
        if callable(effect):
            return effect(*args, **kwargs)
        return effect

    @property
    def called(self) -> bool:
        return bool(self.call_args_list)

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    @property
    def call_args(self) -> _RecordedCall:
        if not self.call_args_list:
            raise AssertionError("Recorder was not called")
        return self.call_args_list[-1]

    def assert_called_once(self) -> None:
        assert self.call_count == 1

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_called_once()
        assert self.call_args.args == args
        assert self.call_args.kwargs == kwargs


class _BlobClient:
    def __init__(self) -> None:
        self.exists = _CallRecorder(return_value=False)
        self.upload_blob = _CallRecorder()


class _ContainerClient:
    def __init__(self, blob_client: _BlobClient) -> None:
        self.get_blob_client = _CallRecorder(return_value=blob_client)
        self.close = _CallRecorder()


def _install_blob_client(mock_container_client: Any) -> tuple[_BlobClient, _ContainerClient]:
    blob_client = _BlobClient()
    container = _ContainerClient(blob_client)
    mock_container_client.return_value = container
    return blob_client, container


@pytest.fixture
def ctx() -> PluginContext:
    """Create a plugin context with proper operation records for Azure blob audit trail."""
    return make_operation_context(
        run_id="test-run-123",
        node_id="sink",
        plugin_name="azure_blob_sink",
        node_type="SINK",
        operation_type="sink_write",
    )


@pytest.fixture
def mock_container_client() -> Generator[Any, None, None]:
    """Patch container client creation for testing."""
    with patch("elspeth.plugins.sinks.azure_blob_sink.AzureBlobSink._get_container_client") as mock:
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
    headers: str | dict[str, str] | None = None,
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
    if headers is not None:
        config["headers"] = headers
    return config


def _make_sink(**kwargs: Any) -> AzureBlobSink:
    """Create an AzureBlobSink with _on_write_failure injected."""
    return inject_write_failure(AzureBlobSink(make_config(**kwargs)))


class TestAzureBlobSinkProtocol:
    """Tests for AzureBlobSink protocol compliance."""

    def test_has_required_attributes(self, mock_container_client: Any) -> None:
        """AzureBlobSink has name and input_schema."""
        assert AzureBlobSink.name == "azure_blob"
        sink = _make_sink()
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

    def test_invalid_blob_path_template_raises_error(self) -> None:
        """Invalid Jinja2 template in blob_path raises PluginConfigError at config time.

        Structural template errors must stop the run at setup, not be deferred
        to the first write() call (Pipeline Templates as Tier 2 Data pattern).
        Validation is now in the Pydantic model_validator (from_dict catches it).
        """
        with pytest.raises(PluginConfigError, match="Invalid blob_path template"):
            AzureBlobSink(make_config(blob_path="results/{% if unclosed %}/output.csv"))

    def test_missing_schema_raises_error(self) -> None:
        """Missing schema raises PluginConfigError with actionable mode guidance."""
        with pytest.raises(PluginConfigError) as exc_info:
            AzureBlobSink(
                {
                    "connection_string": TEST_CONNECTION_STRING,
                    "container": TEST_CONTAINER,
                    "blob_path": TEST_BLOB_PATH,
                }
            )
        message = str(exc_info.value)
        assert "schema" in message
        assert "Field required" in message
        assert "mode: observed" in message
        assert "fixed" in message
        assert "flexible" in message

    def test_unknown_field_raises_error(self) -> None:
        """Unknown config field raises PluginConfigError."""
        with pytest.raises(PluginConfigError, match="Extra inputs"):
            AzureBlobSink(
                {
                    **make_config(),
                    "unknown_field": "value",
                }
            )

    def test_old_config_keys_rejected(self) -> None:
        """Old restore_source_headers and display_headers keys are rejected by extra=forbid."""
        config = make_config()
        config["restore_source_headers"] = True
        with pytest.raises(PluginConfigError):
            AzureBlobSink(config)

        config2 = make_config()
        config2["display_headers"] = {"id": "ID"}
        with pytest.raises(PluginConfigError):
            AzureBlobSink(config2)


class TestAzureBlobSinkAuthMethods:
    """Tests for Azure authentication methods."""

    def test_auth_connection_string(self, mock_container_client: Any) -> None:
        """Connection string auth creates sink successfully."""
        sink = _make_sink(connection_string=TEST_CONNECTION_STRING)
        assert sink._auth_config.auth_method == "connection_string"
        assert sink._auth_config.connection_string == TEST_CONNECTION_STRING

    def test_auth_managed_identity(self, mock_container_client: Any) -> None:
        """Managed identity auth creates sink successfully."""
        sink = _make_sink(
            connection_string=None,
            use_managed_identity=True,
            account_url=TEST_ACCOUNT_URL,
        )
        assert sink._auth_config.auth_method == "managed_identity"
        assert sink._auth_config.use_managed_identity is True
        assert sink._auth_config.account_url == TEST_ACCOUNT_URL

    def test_auth_service_principal(self, mock_container_client: Any) -> None:
        """Service principal auth creates sink successfully."""
        sink = _make_sink(
            connection_string=None,
            tenant_id=TEST_TENANT_ID,
            client_id=TEST_CLIENT_ID,
            client_secret=TEST_CLIENT_SECRET,
            account_url=TEST_ACCOUNT_URL,
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
        sink = _make_sink(
            connection_string=None,
            use_managed_identity=True,
            account_url=TEST_ACCOUNT_URL,
        )

        # Mock the azure.identity and azure.storage.blob imports
        with (
            patch("azure.identity.DefaultAzureCredential") as mock_credential_cls,
            patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls,
        ):
            mock_credential = object()
            mock_credential_cls.return_value = mock_credential
            mock_service_client = object()
            mock_service_client_cls.return_value = mock_service_client

            # Trigger client creation
            sink._auth_config.create_blob_service_client()

            # Verify DefaultAzureCredential was instantiated
            mock_credential_cls.assert_called_once()
            # Verify BlobServiceClient was created with account_url and credential
            mock_service_client_cls.assert_called_once_with(TEST_ACCOUNT_URL, credential=mock_credential)

    def test_service_principal_uses_client_secret_credential(self, ctx: PluginContext) -> None:
        """Service principal auth uses ClientSecretCredential."""
        sink = _make_sink(
            connection_string=None,
            tenant_id=TEST_TENANT_ID,
            client_id=TEST_CLIENT_ID,
            client_secret=TEST_CLIENT_SECRET,
            account_url=TEST_ACCOUNT_URL,
        )

        # Mock the azure.identity and azure.storage.blob imports
        with (
            patch("azure.identity.ClientSecretCredential") as mock_credential_cls,
            patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls,
        ):
            mock_credential = object()
            mock_credential_cls.return_value = mock_credential
            mock_service_client = object()
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
            mock_service_client_cls.assert_called_once_with(TEST_ACCOUNT_URL, credential=mock_credential)

    def test_connection_string_uses_from_connection_string(self, ctx: PluginContext) -> None:
        """Connection string auth uses from_connection_string factory."""
        sink = _make_sink(connection_string=TEST_CONNECTION_STRING)

        # Mock the azure.storage.blob import
        with patch("azure.storage.blob.BlobServiceClient") as mock_service_client_cls:
            mock_service_client = object()
            mock_service_client_cls.from_connection_string.return_value = mock_service_client

            # Trigger client creation
            sink._auth_config.create_blob_service_client()

            # Verify from_connection_string was called
            mock_service_client_cls.from_connection_string.assert_called_once_with(TEST_CONNECTION_STRING)
