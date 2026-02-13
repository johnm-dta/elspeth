"""Tests for Azure authentication configuration.

Tests the four mutually exclusive authentication methods:
1. Connection string
2. SAS token + account_url
3. Managed Identity + account_url
4. Service Principal (tenant_id + client_id + client_secret + account_url)
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from elspeth.plugins.azure.auth import AzureAuthConfig


class TestAzureAuthConfigValid:
    """Tests for valid AzureAuthConfig configurations."""

    def test_connection_string_auth(self) -> None:
        """Connection string auth is valid with just connection_string."""
        config = AzureAuthConfig(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
        )
        assert config.auth_method == "connection_string"

    def test_sas_token_auth(self) -> None:
        """SAS token auth requires sas_token and account_url."""
        config = AzureAuthConfig(
            sas_token="sv=2022-11-02&ss=b&srt=sco&sp=rwdlacu&se=2030-01-01",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )
        assert config.auth_method == "sas_token"

    def test_sas_token_with_question_mark(self) -> None:
        """SAS token can start with ? prefix."""
        config = AzureAuthConfig(
            sas_token="?sv=2022-11-02&ss=b&srt=sco&sp=rwdlacu&se=2030-01-01",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )
        assert config.auth_method == "sas_token"

    def test_managed_identity_auth(self) -> None:
        """Managed Identity auth requires use_managed_identity=True and account_url."""
        config = AzureAuthConfig(
            use_managed_identity=True,
            account_url="https://mystorageaccount.blob.core.windows.net",
        )
        assert config.auth_method == "managed_identity"

    def test_service_principal_auth(self) -> None:
        """Service Principal auth requires all four fields."""
        config = AzureAuthConfig(
            tenant_id="00000000-0000-0000-0000-000000000000",
            client_id="11111111-1111-1111-1111-111111111111",
            client_secret="secret-value",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )
        assert config.auth_method == "service_principal"


class TestAzureAuthConfigInvalid:
    """Tests for invalid AzureAuthConfig configurations."""

    def test_no_auth_method_raises(self) -> None:
        """No auth method configured raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig()
        assert "No authentication method configured" in str(exc_info.value)

    def test_empty_connection_string_raises(self) -> None:
        """Empty connection string is treated as no auth method."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(connection_string="")
        assert "No authentication method configured" in str(exc_info.value)

    def test_whitespace_connection_string_raises(self) -> None:
        """Whitespace-only connection string is treated as no auth method."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(connection_string="   ")
        assert "No authentication method configured" in str(exc_info.value)

    def test_multiple_auth_methods_raises(self) -> None:
        """Multiple auth methods configured raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test",
                sas_token="sv=2022-11-02",
                account_url="https://mystorageaccount.blob.core.windows.net",
            )
        assert "Multiple authentication methods configured" in str(exc_info.value)

    def test_sas_token_without_account_url_raises(self) -> None:
        """SAS token without account_url raises ValidationError.

        Note: The validator treats incomplete configs as 'no auth method'
        because has_sas_token requires BOTH sas_token AND account_url.
        """
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(sas_token="sv=2022-11-02&ss=b")
        # Incomplete method = no complete method found
        assert "No authentication method configured" in str(exc_info.value)

    def test_managed_identity_without_account_url_raises(self) -> None:
        """Managed Identity without account_url raises ValidationError.

        Note: The validator treats incomplete configs as 'no auth method'
        because has_managed_identity requires BOTH flag AND account_url.
        """
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(use_managed_identity=True)
        # Incomplete method = no complete method found
        assert "No authentication method configured" in str(exc_info.value)

    def test_partial_service_principal_missing_tenant_id_raises(self) -> None:
        """Partial service principal config raises ValidationError.

        Note: When no complete auth method is found, the validator raises
        'no auth method configured' before checking partial configs.
        """
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(
                client_id="11111111-1111-1111-1111-111111111111",
                client_secret="secret-value",
                account_url="https://mystorageaccount.blob.core.windows.net",
            )
        # Incomplete method = no complete method found
        assert "No authentication method configured" in str(exc_info.value)

    def test_partial_service_principal_missing_client_secret_raises(self) -> None:
        """Partial service principal config raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(
                tenant_id="00000000-0000-0000-0000-000000000000",
                client_id="11111111-1111-1111-1111-111111111111",
                account_url="https://mystorageaccount.blob.core.windows.net",
            )
        assert "No authentication method configured" in str(exc_info.value)

    def test_partial_service_principal_missing_account_url_raises(self) -> None:
        """Partial service principal config raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(
                tenant_id="00000000-0000-0000-0000-000000000000",
                client_id="11111111-1111-1111-1111-111111111111",
                client_secret="secret-value",
            )
        assert "No authentication method configured" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are forbidden (extra='forbid')."""
        with pytest.raises(ValidationError):
            AzureAuthConfig(
                connection_string="valid-conn-string",
                unknown_field="should fail",  # type: ignore[call-arg]
            )


class TestAzureAuthConfigWhitespaceConsistency:
    """Tests for consistent whitespace handling between validator and runtime.

    The validator uses .strip() to treat whitespace-only strings as empty.
    Runtime methods (auth_method, create_blob_service_client) must match.

    Regression test for P2-2026-01-31-azure-auth-method-selection.
    """

    def test_whitespace_connection_string_with_managed_identity_uses_managed_identity(self) -> None:
        """Whitespace connection_string doesn't shadow valid managed identity auth.

        Bug scenario: whitespace-only connection_string passes validator as empty,
        but runtime if-checks treated it as truthy, shadowing the real auth method.
        """
        config = AzureAuthConfig(
            connection_string="   ",  # whitespace only - validator treats as empty
            use_managed_identity=True,
            account_url="https://test.blob.core.windows.net",
        )
        # Runtime must match validator: managed_identity is the real auth method
        assert config.auth_method == "managed_identity"

    def test_whitespace_sas_token_with_managed_identity_uses_managed_identity(self) -> None:
        """Whitespace sas_token doesn't shadow valid managed identity auth."""
        config = AzureAuthConfig(
            sas_token="   ",  # whitespace only - validator treats as empty
            use_managed_identity=True,
            account_url="https://test.blob.core.windows.net",
        )
        assert config.auth_method == "managed_identity"

    def test_whitespace_connection_string_with_service_principal_uses_service_principal(self) -> None:
        """Whitespace connection_string doesn't shadow valid service principal auth."""
        config = AzureAuthConfig(
            connection_string="   ",  # whitespace only
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            account_url="https://test.blob.core.windows.net",
        )
        assert config.auth_method == "service_principal"


class TestAzureAuthConfigAuthMethodProperty:
    """Tests for auth_method property."""

    def test_connection_string_returns_correct_method(self) -> None:
        """auth_method returns 'connection_string' for connection string auth."""
        config = AzureAuthConfig(connection_string="conn-string")
        assert config.auth_method == "connection_string"

    def test_sas_token_returns_correct_method(self) -> None:
        """auth_method returns 'sas_token' for SAS token auth."""
        config = AzureAuthConfig(
            sas_token="token",
            account_url="https://test.blob.core.windows.net",
        )
        assert config.auth_method == "sas_token"

    def test_managed_identity_returns_correct_method(self) -> None:
        """auth_method returns 'managed_identity' for managed identity auth."""
        config = AzureAuthConfig(
            use_managed_identity=True,
            account_url="https://test.blob.core.windows.net",
        )
        assert config.auth_method == "managed_identity"

    def test_service_principal_returns_correct_method(self) -> None:
        """auth_method returns 'service_principal' for service principal auth."""
        config = AzureAuthConfig(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            account_url="https://test.blob.core.windows.net",
        )
        assert config.auth_method == "service_principal"


class TestAzureAuthConfigCreateClient:
    """Tests for create_blob_service_client method.

    These tests mock the Azure SDK to verify correct client construction
    without requiring actual Azure credentials.
    """

    def test_connection_string_creates_client(self) -> None:
        """Connection string auth creates client from_connection_string."""
        config = AzureAuthConfig(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
        )

        mock_client = MagicMock()
        mock_blob_service_client = MagicMock()
        mock_blob_service_client.from_connection_string.return_value = mock_client

        with patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}):
            client = config.create_blob_service_client()

            mock_blob_service_client.from_connection_string.assert_called_once_with(config.connection_string)
            assert client is mock_client

    def test_sas_token_creates_client_with_url(self) -> None:
        """SAS token auth creates client with SAS appended to URL."""
        config = AzureAuthConfig(
            sas_token="sv=2022-11-02&ss=b",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )

        mock_client = MagicMock()
        mock_blob_service_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}):
            client = config.create_blob_service_client()

            # SAS token should be appended with ? prefix
            expected_url = "https://mystorageaccount.blob.core.windows.net?sv=2022-11-02&ss=b"
            mock_blob_service_client.assert_called_once_with(expected_url)
            assert client is mock_client

    def test_sas_token_with_question_mark_creates_client(self) -> None:
        """SAS token with ? prefix doesn't get double ?."""
        config = AzureAuthConfig(
            sas_token="?sv=2022-11-02&ss=b",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )

        mock_client = MagicMock()
        mock_blob_service_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}):
            client = config.create_blob_service_client()

            # Should NOT have double ?
            expected_url = "https://mystorageaccount.blob.core.windows.net?sv=2022-11-02&ss=b"
            mock_blob_service_client.assert_called_once_with(expected_url)
            assert client is mock_client

    def test_sas_token_strips_trailing_slash_from_url(self) -> None:
        """SAS token auth strips trailing slash from account_url."""
        config = AzureAuthConfig(
            sas_token="sv=2022-11-02",
            account_url="https://mystorageaccount.blob.core.windows.net/",
        )

        mock_blob_service_client = MagicMock()

        with patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}):
            config.create_blob_service_client()

            # Should NOT have double slash before ?
            expected_url = "https://mystorageaccount.blob.core.windows.net?sv=2022-11-02"
            mock_blob_service_client.assert_called_once_with(expected_url)

    def test_managed_identity_creates_client_with_credential(self) -> None:
        """Managed Identity auth creates client with DefaultAzureCredential."""
        config = AzureAuthConfig(
            use_managed_identity=True,
            account_url="https://mystorageaccount.blob.core.windows.net",
        )

        mock_client = MagicMock()
        mock_blob_service_client = MagicMock(return_value=mock_client)
        mock_credential = MagicMock()
        mock_default_azure_credential = MagicMock(return_value=mock_credential)

        with (
            patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}),
            patch.dict("sys.modules", {"azure.identity": MagicMock(DefaultAzureCredential=mock_default_azure_credential)}),
        ):
            client = config.create_blob_service_client()

            mock_default_azure_credential.assert_called_once()
            mock_blob_service_client.assert_called_once_with(config.account_url, credential=mock_credential)
            assert client is mock_client

    def test_service_principal_creates_client_with_credential(self) -> None:
        """Service Principal auth creates client with ClientSecretCredential."""
        config = AzureAuthConfig(
            tenant_id="tenant-id",
            client_id="client-id",
            client_secret="client-secret",
            account_url="https://mystorageaccount.blob.core.windows.net",
        )

        mock_client = MagicMock()
        mock_blob_service_client = MagicMock(return_value=mock_client)
        mock_credential = MagicMock()
        mock_client_secret_credential = MagicMock(return_value=mock_credential)

        with (
            patch.dict("sys.modules", {"azure.storage.blob": MagicMock(BlobServiceClient=mock_blob_service_client)}),
            patch.dict("sys.modules", {"azure.identity": MagicMock(ClientSecretCredential=mock_client_secret_credential)}),
        ):
            client = config.create_blob_service_client()

            mock_client_secret_credential.assert_called_once_with(
                tenant_id="tenant-id",
                client_id="client-id",
                client_secret="client-secret",
            )
            mock_blob_service_client.assert_called_once_with(config.account_url, credential=mock_credential)
            assert client is mock_client

    def test_connection_string_with_partial_service_principal_raises(self) -> None:
        """Partial service principal fields are invalid even with connection string."""
        with pytest.raises(ValidationError) as exc_info:
            AzureAuthConfig(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net",
                tenant_id="00000000-0000-0000-0000-000000000000",
            )

        assert "Service Principal auth requires all fields" in str(exc_info.value)
