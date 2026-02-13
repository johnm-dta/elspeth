# src/elspeth/plugins/azure/auth.py
"""Azure authentication configuration for ELSPETH Azure plugins.

Supports four authentication methods (mutually exclusive):
1. Connection string - Simple connection string auth (default)
2. SAS token - Shared Access Signature token with account_url
3. Managed Identity - For Azure-hosted workloads
4. Service Principal - For automated/CI scenarios

IMPORTANT: This module handles external system credentials. Connection strings,
SAS tokens, and service principal secrets should be passed via environment
variables, not hardcoded in configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient


class AzureAuthConfig(BaseModel):
    """Azure authentication configuration.

    Supports four methods (mutually exclusive):
    1. connection_string - Simple connection string auth
    2. sas_token + account_url - Shared Access Signature token
    3. use_managed_identity + account_url - Azure Managed Identity
    4. tenant_id + client_id + client_secret + account_url - Service Principal

    Example configurations:

        # Option 1: Connection string (simplest)
        connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"

        # Option 2: SAS token
        sas_token: "${AZURE_STORAGE_SAS_TOKEN}"
        account_url: "https://mystorageaccount.blob.core.windows.net"

        # Option 3: Managed Identity (for Azure-hosted workloads)
        use_managed_identity: true
        account_url: "https://mystorageaccount.blob.core.windows.net"

        # Option 4: Service Principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
    """

    model_config = {"extra": "forbid"}

    # Option 1: Connection string
    connection_string: str | None = None

    # Option 2: SAS token
    sas_token: str | None = None

    # Option 3: Managed Identity
    use_managed_identity: bool = False
    account_url: str | None = None

    # Option 4: Service Principal
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    @model_validator(mode="after")
    def validate_auth_method(self) -> Self:
        """Ensure exactly one auth method is configured.

        Validates that exactly one of the four authentication methods is
        properly configured:
        - Connection string requires connection_string field
        - SAS token requires sas_token AND account_url
        - Managed Identity requires use_managed_identity=True AND account_url
        - Service Principal requires all of: tenant_id, client_id, client_secret, account_url

        Raises:
            ValueError: If zero or multiple auth methods are configured.
        """
        has_conn_string = self.connection_string is not None and bool(self.connection_string.strip())
        has_sas_token = (
            self.sas_token is not None and bool(self.sas_token.strip()) and self.account_url is not None and bool(self.account_url.strip())
        )
        has_managed_identity = self.use_managed_identity and self.account_url is not None and bool(self.account_url.strip())
        has_service_principal = all(
            [
                self.tenant_id is not None and bool(self.tenant_id.strip()),
                self.client_id is not None and bool(self.client_id.strip()),
                self.client_secret is not None and bool(self.client_secret.strip()),
                self.account_url is not None and bool(self.account_url.strip()),
            ]
        )

        methods = [has_conn_string, has_sas_token, has_managed_identity, has_service_principal]
        active_count = sum(methods)

        if active_count == 0:
            raise ValueError(
                "No authentication method configured. Provide one of: "
                "connection_string, "
                "sas_token + account_url, "
                "managed identity (use_managed_identity + account_url), or "
                "service principal (tenant_id + client_id + client_secret + account_url)"
            )

        if active_count > 1:
            raise ValueError(
                "Multiple authentication methods configured. Provide exactly one of: "
                "connection_string, "
                "sas_token + account_url, "
                "managed identity (use_managed_identity + account_url), or "
                "service principal (tenant_id + client_id + client_secret + account_url)"
            )

        # Additional validation for partial configurations
        if self.sas_token and not self.account_url:
            raise ValueError("SAS token auth requires account_url. Example: https://mystorageaccount.blob.core.windows.net")

        if self.use_managed_identity and not self.account_url:
            raise ValueError("Managed Identity auth requires account_url. Example: https://mystorageaccount.blob.core.windows.net")

        sp_fields = [self.tenant_id, self.client_id, self.client_secret]
        sp_field_count = sum(1 for f in sp_fields if f is not None)
        if 0 < sp_field_count < 3:
            missing = []
            if self.tenant_id is None:
                missing.append("tenant_id")
            if self.client_id is None:
                missing.append("client_id")
            if self.client_secret is None:
                missing.append("client_secret")
            if self.account_url is None:
                missing.append("account_url")
            raise ValueError(f"Service Principal auth requires all fields. Missing: {', '.join(missing)}")

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
            raise ImportError("azure-storage-blob is required for Azure plugins. Install with: uv pip install azure-storage-blob") from e

        if self._is_set(self.connection_string):
            # _is_set guarantees non-None non-whitespace
            connection_string = cast(str, self.connection_string)
            return BlobServiceClient.from_connection_string(connection_string)

        elif self._is_set(self.sas_token):
            # SAS token auth: append token to account URL
            # model_validator guarantees account_url is non-None when sas_token is set
            # _is_set guarantees sas_token is non-None non-whitespace
            account_url = cast(str, self.account_url)
            sas_token = cast(str, self.sas_token)
            # Ensure SAS token starts with '?' for URL concatenation
            sas = sas_token if sas_token.startswith("?") else f"?{sas_token}"
            sas_url = f"{account_url.rstrip('/')}{sas}"
            return BlobServiceClient(sas_url)

        elif self.use_managed_identity:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required for Managed Identity auth. Install with: uv pip install azure-identity"
                ) from e

            # model_validator guarantees account_url is non-None when use_managed_identity is True
            account_url = cast(str, self.account_url)
            credential = DefaultAzureCredential()
            return BlobServiceClient(account_url, credential=credential)

        else:
            # Service principal auth
            try:
                from azure.identity import ClientSecretCredential
            except ImportError as e:
                raise ImportError(
                    "azure-identity is required for Service Principal auth. Install with: uv pip install azure-identity"
                ) from e

            # model_validator guarantees all service principal fields are non-None in this branch
            tenant_id = cast(str, self.tenant_id)
            client_id = cast(str, self.client_id)
            client_secret = cast(str, self.client_secret)
            account_url = cast(str, self.account_url)
            sp_credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            return BlobServiceClient(account_url, credential=sp_credential)

    def _is_set(self, value: str | None) -> bool:
        """Check if a string field is non-empty after stripping whitespace.

        Matches the validator's semantics: whitespace-only strings are treated as empty.
        This ensures runtime behavior is consistent with validation.
        """
        return value is not None and bool(value.strip())

    @property
    def auth_method(self) -> str:
        """Return the active authentication method name.

        Returns:
            One of: 'connection_string', 'sas_token', 'managed_identity', 'service_principal'
        """
        if self._is_set(self.connection_string):
            return "connection_string"
        elif self._is_set(self.sas_token):
            return "sas_token"
        elif self.use_managed_identity:
            return "managed_identity"
        else:
            return "service_principal"
