# src/elspeth/plugins/azure/auth.py
"""Azure authentication configuration for ELSPETH Azure plugins.

Supports three authentication methods (mutually exclusive):
1. Connection string - Simple connection string auth (default)
2. Managed Identity - For Azure-hosted workloads
3. Service Principal - For automated/CI scenarios

IMPORTANT: This module handles external system credentials. Connection strings
and service principal secrets should be passed via environment variables, not
hardcoded in configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient


class AzureAuthConfig(BaseModel):
    """Azure authentication configuration.

    Supports three methods (mutually exclusive):
    1. connection_string - Simple connection string auth
    2. use_managed_identity + account_url - Azure Managed Identity
    3. tenant_id + client_id + client_secret + account_url - Service Principal

    Example configurations:

        # Option 1: Connection string (simplest)
        connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"

        # Option 2: Managed Identity (for Azure-hosted workloads)
        use_managed_identity: true
        account_url: "https://mystorageaccount.blob.core.windows.net"

        # Option 3: Service Principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
    """

    model_config = {"extra": "forbid"}

    # Option 1: Connection string
    connection_string: str | None = None

    # Option 2: Managed Identity
    use_managed_identity: bool = False
    account_url: str | None = None

    # Option 3: Service Principal
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    @model_validator(mode="after")
    def validate_auth_method(self) -> Self:
        """Ensure exactly one auth method is configured.

        Validates that exactly one of the three authentication methods is
        properly configured:
        - Connection string requires connection_string field
        - Managed Identity requires use_managed_identity=True AND account_url
        - Service Principal requires all of: tenant_id, client_id, client_secret, account_url

        Raises:
            ValueError: If zero or multiple auth methods are configured.
        """
        has_conn_string = self.connection_string is not None and bool(
            self.connection_string.strip()
        )
        has_managed_identity = (
            self.use_managed_identity and self.account_url is not None
        )
        has_service_principal = all(
            [
                self.tenant_id is not None,
                self.client_id is not None,
                self.client_secret is not None,
                self.account_url is not None,
            ]
        )

        methods = [has_conn_string, has_managed_identity, has_service_principal]
        active_count = sum(methods)

        if active_count == 0:
            raise ValueError(
                "No authentication method configured. Provide one of: "
                "connection_string, "
                "managed identity (use_managed_identity + account_url), or "
                "service principal (tenant_id + client_id + client_secret + account_url)"
            )

        if active_count > 1:
            raise ValueError(
                "Multiple authentication methods configured. Provide exactly one of: "
                "connection_string, "
                "managed identity (use_managed_identity + account_url), or "
                "service principal (tenant_id + client_id + client_secret + account_url)"
            )

        # Additional validation for partial configurations
        if self.use_managed_identity and not self.account_url:
            raise ValueError(
                "Managed Identity auth requires account_url. "
                "Example: https://mystorageaccount.blob.core.windows.net"
            )

        sp_fields = [self.tenant_id, self.client_id, self.client_secret]
        sp_field_count = sum(1 for f in sp_fields if f is not None)
        if 0 < sp_field_count < 3 and not has_conn_string and not has_managed_identity:
            missing = []
            if self.tenant_id is None:
                missing.append("tenant_id")
            if self.client_id is None:
                missing.append("client_id")
            if self.client_secret is None:
                missing.append("client_secret")
            if self.account_url is None:
                missing.append("account_url")
            raise ValueError(
                f"Service Principal auth requires all fields. Missing: {', '.join(missing)}"
            )

        return self

    def create_blob_service_client(self) -> BlobServiceClient:
        """Create BlobServiceClient using the configured auth method.

        Returns:
            BlobServiceClient configured with the appropriate credentials.

        Raises:
            ImportError: If azure-storage-blob or azure-identity is not installed.
        """
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required for Azure plugins. "
                "Install with: uv pip install azure-storage-blob"
            ) from e

        if self.connection_string:
            return BlobServiceClient.from_connection_string(self.connection_string)

        elif self.use_managed_identity:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required for Managed Identity auth. "
                    "Install with: uv pip install azure-identity"
                ) from e

            credential = DefaultAzureCredential()
            # account_url is validated to be not None by the model_validator
            assert self.account_url is not None  # Validated by model_validator
            return BlobServiceClient(self.account_url, credential=credential)

        else:
            # Service principal auth
            try:
                from azure.identity import ClientSecretCredential
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required for Service Principal auth. "
                    "Install with: uv pip install azure-identity"
                ) from e

            # All fields are validated to be not None by the model_validator
            assert self.tenant_id is not None  # Validated by model_validator
            assert self.client_id is not None  # Validated by model_validator
            assert self.client_secret is not None  # Validated by model_validator
            assert self.account_url is not None  # Validated by model_validator
            sp_credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            return BlobServiceClient(self.account_url, credential=sp_credential)

    @property
    def auth_method(self) -> str:
        """Return the active authentication method name.

        Returns:
            One of: 'connection_string', 'managed_identity', 'service_principal'
        """
        if self.connection_string:
            return "connection_string"
        elif self.use_managed_identity:
            return "managed_identity"
        else:
            return "service_principal"
