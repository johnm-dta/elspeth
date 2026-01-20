# src/elspeth/plugins/azure/blob_source.py
"""Azure Blob Storage source plugin for ELSPETH.

Loads rows from Azure Blob containers. Supports CSV, JSON array, and JSONL formats.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.plugins.azure.auth import AzureAuthConfig
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient


class CSVOptions(BaseModel):
    """CSV parsing options."""

    model_config = {"extra": "forbid"}

    delimiter: str = ","
    has_header: bool = True
    encoding: str = "utf-8"


class JSONOptions(BaseModel):
    """JSON parsing options."""

    model_config = {"extra": "forbid"}

    encoding: str = "utf-8"
    data_key: str | None = None


class AzureBlobSourceConfig(DataPluginConfig):
    """Configuration for Azure Blob source plugin.

    Extends DataPluginConfig which requires schema configuration.
    Unlike file-based sources, does not extend PathConfig (no local file path).

    Supports three authentication methods (mutually exclusive):
    1. connection_string - Simple connection string auth (default)
    2. use_managed_identity + account_url - Azure Managed Identity
    3. tenant_id + client_id + client_secret + account_url - Service Principal

    Example configurations:

        # Option 1: Connection string (simplest)
        connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
        container: "my-container"
        blob_path: "data/input.csv"

        # Option 2: Managed Identity (for Azure-hosted workloads)
        use_managed_identity: true
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "data/input.csv"

        # Option 3: Service Principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "data/input.csv"
    """

    # Auth Option 1: Connection string
    connection_string: str | None = Field(
        default=None,
        description="Azure Storage connection string",
    )

    # Auth Option 2: Managed Identity
    use_managed_identity: bool = Field(
        default=False,
        description="Use Azure Managed Identity for authentication",
    )
    account_url: str | None = Field(
        default=None,
        description="Azure Storage account URL (e.g., https://mystorageaccount.blob.core.windows.net)",
    )

    # Auth Option 3: Service Principal
    tenant_id: str | None = Field(
        default=None,
        description="Azure AD tenant ID for Service Principal auth",
    )
    client_id: str | None = Field(
        default=None,
        description="Azure AD client ID for Service Principal auth",
    )
    client_secret: str | None = Field(
        default=None,
        description="Azure AD client secret for Service Principal auth",
    )

    # Blob location (required for all auth methods)
    container: str = Field(
        ...,
        description="Azure Blob container name",
    )
    blob_path: str = Field(
        ...,
        description="Path to blob within container",
    )
    format: Literal["csv", "json", "jsonl"] = Field(
        default="csv",
        description="Data format: csv, json (array), or jsonl (newline-delimited)",
    )
    csv_options: CSVOptions = Field(
        default_factory=CSVOptions,
        description="CSV parsing options (delimiter, has_header, encoding)",
    )
    json_options: JSONOptions = Field(
        default_factory=JSONOptions,
        description="JSON parsing options (encoding, data_key)",
    )
    on_validation_failure: str = Field(
        ...,
        description="Sink name for non-conformant rows, or 'discard' for explicit drop",
    )

    @model_validator(mode="after")
    def validate_auth_config(self) -> Self:
        """Validate authentication configuration via AzureAuthConfig.

        Delegates to AzureAuthConfig for comprehensive auth validation,
        ensuring exactly one auth method is configured.
        """
        # Create AzureAuthConfig to validate auth fields
        # This will raise ValueError with descriptive messages if invalid
        AzureAuthConfig(
            connection_string=self.connection_string,
            use_managed_identity=self.use_managed_identity,
            account_url=self.account_url,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        return self

    def get_auth_config(self) -> AzureAuthConfig:
        """Get the AzureAuthConfig for this source configuration.

        Returns:
            AzureAuthConfig instance with the auth fields from this config.
        """
        return AzureAuthConfig(
            connection_string=self.connection_string,
            use_managed_identity=self.use_managed_identity,
            account_url=self.account_url,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    @field_validator("container")
    @classmethod
    def validate_container_not_empty(cls, v: str) -> str:
        """Validate that container is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("container cannot be empty")
        return v

    @field_validator("blob_path")
    @classmethod
    def validate_blob_path_not_empty(cls, v: str) -> str:
        """Validate that blob_path is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("blob_path cannot be empty")
        return v

    @field_validator("on_validation_failure")
    @classmethod
    def validate_on_validation_failure(cls, v: str) -> str:
        """Ensure on_validation_failure is not empty."""
        if not v or not v.strip():
            raise ValueError("on_validation_failure must be a sink name or 'discard'")
        return v.strip()


class AzureBlobSource(BaseSource):
    """Load rows from Azure Blob Storage.

    Config options:
        Authentication (exactly one required):
        - connection_string: Azure Storage connection string
        - use_managed_identity + account_url: Azure Managed Identity
        - tenant_id + client_id + client_secret + account_url: Service Principal

        Blob location:
        - container: Blob container name (required)
        - blob_path: Path to blob within container (required)
        - format: "csv", "json" (array), or "jsonl" (lines). Default: "csv"

        Parsing options:
        - csv_options: CSV parsing options (delimiter, has_header, encoding)
        - json_options: JSON parsing options (encoding, data_key)
        - schema: Schema configuration (required, via DataPluginConfig)
        - on_validation_failure: Sink name or "discard" (required)

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]}
        - Free: {"mode": "free", "fields": ["id: int"]} - at least these fields

    Three-tier trust model:
        - Azure Blob SDK calls = EXTERNAL SYSTEM -> wrap with try/except
        - Row data parsing/validation = THEIR DATA -> wrap, quarantine failures
        - Our internal state = OUR CODE -> let it crash
    """

    name = "azure_blob"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AzureBlobSourceConfig.from_dict(config)

        # Store auth config for creating clients
        self._auth_config = cfg.get_auth_config()
        self._container = cfg.container
        self._blob_path = cfg.blob_path
        self._format = cfg.format
        self._csv_options = cfg.csv_options
        self._json_options = cfg.json_options

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Store quarantine routing destination
        self._on_validation_failure = cfg.on_validation_failure

        # CRITICAL: allow_coercion=True for sources (external data boundary)
        # Sources are the ONLY place where type coercion is allowed
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "AzureBlobRowSchema",
            allow_coercion=True,
        )

        # Set output_schema for protocol compliance
        self.output_schema = self._schema_class

        # Lazy-loaded blob client
        self._blob_client: BlobClient | None = None

    def _get_blob_client(self) -> BlobClient:
        """Get or create the Azure Blob client.

        Uses the configured authentication method (connection string,
        managed identity, or service principal) to create the client.

        Returns:
            BlobClient for the configured blob.

        Raises:
            ImportError: If azure-storage-blob (or azure-identity for
                managed identity/service principal) is not installed.
        """
        if self._blob_client is None:
            # Use shared auth config to create the service client
            service_client = self._auth_config.create_blob_service_client()
            container_client = service_client.get_container_client(self._container)
            self._blob_client = container_client.get_blob_client(self._blob_path)

        return self._blob_client

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from Azure Blob Storage.

        Each row is validated against the configured schema:
        - Valid rows are yielded as SourceRow.valid()
        - Invalid rows are yielded as SourceRow.quarantined()

        Yields:
            SourceRow for each row (valid or quarantined).

        Raises:
            ImportError: If azure-storage-blob is not installed.
            azure.core.exceptions.ResourceNotFoundError: If blob does not exist.
            azure.core.exceptions.ClientAuthenticationError: If connection fails.
        """
        # EXTERNAL SYSTEM: Azure Blob SDK calls - wrap with try/except
        try:
            blob_client = self._get_blob_client()
            blob_data = blob_client.download_blob().readall()
        except ImportError:
            # Re-raise ImportError as-is for clear dependency messaging
            raise
        except Exception as e:
            # Azure SDK errors (ResourceNotFoundError, ClientAuthenticationError, etc.)
            # are external system errors - propagate with context
            raise type(e)(
                f"Failed to download blob '{self._blob_path}' from container "
                f"'{self._container}': {e}"
            ) from e

        # Parse blob content based on format
        if self._format == "csv":
            yield from self._load_csv(blob_data, ctx)
        elif self._format == "json":
            yield from self._load_json_array(blob_data, ctx)
        elif self._format == "jsonl":
            yield from self._load_jsonl(blob_data, ctx)

    def _load_csv(self, blob_data: bytes, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from CSV blob data.

        Args:
            blob_data: Raw bytes from blob download.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow for each row (valid or quarantined).
        """
        # Access typed config fields directly - they have defaults from CSVOptions
        delimiter = self._csv_options.delimiter
        has_header = self._csv_options.has_header
        encoding = self._csv_options.encoding

        # THEIR DATA: Parsing blob content - wrap operations
        try:
            text_data = blob_data.decode(encoding)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode blob as {encoding}: {e}") from e

        # Use pandas for robust CSV parsing (consistent with CSVSource)
        header_arg = 0 if has_header else None
        df = pd.read_csv(
            io.StringIO(text_data),
            delimiter=delimiter,
            header=header_arg,
            dtype=str,  # Keep all values as strings for consistent handling
            keep_default_na=False,  # Don't convert empty strings to NaN
        )

        # DataFrame columns are strings from CSV headers
        for record in df.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}
            yield from self._validate_and_yield(row, ctx)

    def _load_json_array(
        self, blob_data: bytes, ctx: PluginContext
    ) -> Iterator[SourceRow]:
        """Load rows from JSON array blob data.

        Args:
            blob_data: Raw bytes from blob download.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow for each row (valid or quarantined).

        Raises:
            ValueError: If JSON is invalid or not an array.
        """
        # Access typed config fields directly - they have defaults from JSONOptions
        encoding = self._json_options.encoding
        data_key = self._json_options.data_key

        # THEIR DATA: JSON parsing - wrap operations
        try:
            text_data = blob_data.decode(encoding)
            data = json.loads(text_data)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode blob as {encoding}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in blob: {e}") from e

        # Extract from nested key if specified
        if data_key:
            if not isinstance(data, dict) or data_key not in data:
                raise ValueError(
                    f"Expected JSON object with key '{data_key}', got {type(data).__name__}"
                )
            data = data[data_key]

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        for row in data:
            yield from self._validate_and_yield(row, ctx)

    def _load_jsonl(self, blob_data: bytes, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from JSONL (newline-delimited JSON) blob data.

        Args:
            blob_data: Raw bytes from blob download.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow for each row (valid or quarantined).
        """
        # Access typed config field directly - it has a default from JSONOptions
        encoding = self._json_options.encoding

        # THEIR DATA: Decoding blob content - wrap operations
        try:
            text_data = blob_data.decode(encoding)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode blob as {encoding}: {e}") from e

        for line_num, line in enumerate(text_data.splitlines(), start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # THEIR DATA: JSON parsing per line - wrap operations
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            yield from self._validate_and_yield(row, ctx)

    def _validate_and_yield(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> Iterator[SourceRow]:
        """Validate a row and yield if valid, otherwise quarantine.

        Args:
            row: Row data to validate.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow.valid() if valid, SourceRow.quarantined() if invalid.
        """
        try:
            # Validate and potentially coerce row data
            validated = self._schema_class.model_validate(row)
            yield SourceRow.valid(validated.to_row())
        except ValidationError as e:
            # Record validation failure in audit trail
            # This is a trust boundary: external data may be invalid
            ctx.record_validation_error(
                row=row,
                error=str(e),
                schema_mode=self._schema_config.mode or "dynamic",
                destination=self._on_validation_failure,
            )

            # Yield quarantined row for routing to configured sink
            # If "discard", don't yield - row is intentionally dropped
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(e),
                    destination=self._on_validation_failure,
                )

    def close(self) -> None:
        """Release resources (no-op for Azure Blob source)."""
        self._blob_client = None
