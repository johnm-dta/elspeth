# src/elspeth/plugins/azure/blob_sink.py
"""Azure Blob Storage sink plugin for ELSPETH.

Writes rows to Azure Blob containers. Supports CSV, JSON array, and JSONL formats.

IMPORTANT: Sinks use allow_coercion=False - wrong types are upstream bugs.
This is NOT the trust boundary (Sources are). Sinks receive PIPELINE DATA.

Three-tier trust model:
    - Azure Blob SDK calls = EXTERNAL SYSTEM -> wrap with try/except
    - Serialization of rows = OUR CODE -> let it crash (rows already validated)
    - Internal state = OUR CODE -> let it crash
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Self

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.plugins.azure.auth import AzureAuthConfig
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient


class CSVWriteOptions(BaseModel):
    """CSV writing options."""

    model_config = {"extra": "forbid"}

    delimiter: str = ","
    encoding: str = "utf-8"
    include_header: bool = True


class AzureBlobSinkConfig(DataPluginConfig):
    """Configuration for Azure Blob sink plugin.

    Extends DataPluginConfig which requires schema configuration.
    Unlike file-based sinks, does not extend PathConfig (no local file path).

    Supports four authentication methods (mutually exclusive):
    1. connection_string - Simple connection string auth (default)
    2. sas_token + account_url - Shared Access Signature token
    3. use_managed_identity + account_url - Azure Managed Identity
    4. tenant_id + client_id + client_secret + account_url - Service Principal

    Example configurations:

        # Option 1: Connection string (simplest)
        connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
        container: "my-container"
        blob_path: "results/{{ run_id }}/output.csv"

        # Option 2: SAS token
        sas_token: "${AZURE_STORAGE_SAS_TOKEN}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "results/{{ run_id }}/output.csv"

        # Option 3: Managed Identity (for Azure-hosted workloads)
        use_managed_identity: true
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "results/{{ run_id }}/output.csv"

        # Option 4: Service Principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "results/{{ run_id }}/output.csv"
    """

    # Auth Option 1: Connection string
    connection_string: str | None = Field(
        default=None,
        description="Azure Storage connection string",
    )

    # Auth Option 2: SAS token
    sas_token: str | None = Field(
        default=None,
        description="Azure Storage SAS token (with or without leading '?')",
    )

    # Auth Option 3: Managed Identity
    use_managed_identity: bool = Field(
        default=False,
        description="Use Azure Managed Identity for authentication",
    )
    account_url: str | None = Field(
        default=None,
        description="Azure Storage account URL (e.g., https://mystorageaccount.blob.core.windows.net)",
    )

    # Auth Option 4: Service Principal
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
        description="Path to blob within container (supports Jinja2 templates)",
    )
    format: Literal["csv", "json", "jsonl"] = Field(
        default="csv",
        description="Data format: csv, json (array), or jsonl (newline-delimited)",
    )
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite existing blob (if False, raises if exists)",
    )
    csv_options: CSVWriteOptions = Field(
        default_factory=CSVWriteOptions,
        description="CSV writing options (delimiter, encoding, include_header)",
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
            sas_token=self.sas_token,
            use_managed_identity=self.use_managed_identity,
            account_url=self.account_url,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        return self

    def get_auth_config(self) -> AzureAuthConfig:
        """Get the AzureAuthConfig for this sink configuration.

        Returns:
            AzureAuthConfig instance with the auth fields from this config.
        """
        return AzureAuthConfig(
            connection_string=self.connection_string,
            sas_token=self.sas_token,
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


# Rebuild model to resolve forward references for dynamic module loading
AzureBlobSinkConfig.model_rebuild()


class AzureBlobSink(BaseSink):
    """Write rows to Azure Blob Storage.

    Config options:
        Authentication (exactly one required):
        - connection_string: Azure Storage connection string
        - use_managed_identity + account_url: Azure Managed Identity
        - tenant_id + client_id + client_secret + account_url: Service Principal

        Blob location:
        - container: Blob container name (required)
        - blob_path: Path to blob within container, supports Jinja2 (required)
        - format: "csv", "json" (array), or "jsonl" (lines). Default: "csv"
        - overwrite: Whether to overwrite existing blob. Default: True

        Writing options:
        - csv_options: CSV writing options (delimiter, encoding, include_header)
        - schema: Schema configuration (required, via DataPluginConfig)

    Blob path templating:
        The blob_path can contain Jinja2 templates for dynamic paths:
        - {{ run_id }} - The current run ID
        - {{ timestamp }} - ISO format timestamp at write time

    Three-tier trust model:
        - Azure Blob SDK calls = EXTERNAL SYSTEM -> wrap with try/except
        - Serialization of rows = OUR CODE -> let it crash (already validated)
        - Our internal state = OUR CODE -> let it crash
    """

    name = "azure_blob"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AzureBlobSinkConfig.from_dict(config)

        # Store auth config for creating clients
        self._auth_config = cfg.get_auth_config()
        self._container = cfg.container
        self._blob_path_template = cfg.blob_path
        self._format = cfg.format
        self._overwrite = cfg.overwrite

        # CSV options are already validated Pydantic model
        self._csv_options = cfg.csv_options

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "AzureBlobRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        # Lazy-loaded clients
        self._container_client: ContainerClient | None = None

    def _get_container_client(self) -> ContainerClient:
        """Get or create the Azure container client.

        Uses the configured authentication method (connection string,
        managed identity, or service principal) to create the client.

        Returns:
            ContainerClient for the configured container.

        Raises:
            ImportError: If azure-storage-blob (or azure-identity for
                managed identity/service principal) is not installed.
        """
        if self._container_client is None:
            # Use shared auth config to create the service client
            service_client = self._auth_config.create_blob_service_client()
            self._container_client = service_client.get_container_client(self._container)

        return self._container_client

    def _render_blob_path(self, ctx: PluginContext) -> str:
        """Render blob path template with context variables.

        Args:
            ctx: Plugin context containing run_id and other metadata.

        Returns:
            Rendered blob path string.

        Raises:
            jinja2.UndefinedError: If template references undefined variables.
                This is intentional fail-fast behavior to catch config typos
                (e.g., {{ runid }} instead of {{ run_id }}).
        """
        # Use StrictUndefined to fail fast on typos in blob_path template.
        # A typo like {{ runid }} should error, not silently become empty.
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(self._blob_path_template)
        return template.render(
            run_id=ctx.run_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    def _serialize_rows(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to bytes based on format.

        This is OUR CODE operating on validated data. Let it crash on bugs.

        Args:
            rows: List of row dicts to serialize.

        Returns:
            Serialized bytes content.
        """
        if self._format == "csv":
            return self._serialize_csv(rows)
        elif self._format == "json":
            return self._serialize_json(rows)
        elif self._format == "jsonl":
            return self._serialize_jsonl(rows)
        else:
            # Unreachable due to Pydantic Literal validation, but satisfies static analysis
            raise AssertionError(f"Unsupported format: {self._format}")

    def _serialize_csv(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to CSV bytes."""
        output = io.StringIO()

        # Determine fieldnames from first row
        fieldnames = list(rows[0].keys())

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter=self._csv_options.delimiter,
        )

        if self._csv_options.include_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

        return output.getvalue().encode(self._csv_options.encoding)

    def _serialize_json(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to JSON array bytes."""
        return json.dumps(rows, indent=2).encode("utf-8")

    def _serialize_jsonl(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to JSONL bytes (newline-delimited JSON)."""
        lines = [json.dumps(row) for row in rows]
        return "\n".join(lines).encode("utf-8")

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write a batch of rows to Azure Blob Storage.

        Args:
            rows: List of row dicts to write.
            ctx: Plugin context.

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes.

        Raises:
            ImportError: If azure-storage-blob is not installed.
            ValueError: If overwrite=False and blob exists.
            azure.core.exceptions.*: On Azure SDK errors.
        """
        if not rows:
            # Still render the path for consistent audit trail
            rendered_path = self._render_blob_path(ctx)
            return ArtifactDescriptor(
                artifact_type="file",
                path_or_uri=f"azure://{self._container}/{rendered_path}",
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        # Render the blob path with context variables
        rendered_path = self._render_blob_path(ctx)

        # Serialize rows to bytes (OUR CODE - let it crash on bugs)
        content = self._serialize_rows(rows)

        # Compute content hash before upload
        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)

        # EXTERNAL SYSTEM: Azure Blob SDK calls - wrap with try/except
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(rendered_path)

            # Check overwrite policy
            if not self._overwrite and blob_client.exists():
                raise ValueError(f"Blob '{rendered_path}' already exists and overwrite=False")

            # Upload the content
            blob_client.upload_blob(content, overwrite=self._overwrite)

        except ImportError:
            # Re-raise ImportError as-is for clear dependency messaging
            raise
        except ValueError:
            # Re-raise our own ValueError (overwrite check)
            raise
        except Exception as e:
            # Azure SDK errors are external system errors - propagate with context
            raise type(e)(f"Failed to upload blob '{rendered_path}' to container '{self._container}': {e}") from e

        return ArtifactDescriptor(
            artifact_type="file",
            path_or_uri=f"azure://{self._container}/{rendered_path}",
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def flush(self) -> None:
        """Flush buffered data.

        Azure Blob uploads are synchronous, so this is a no-op.
        """
        pass

    def close(self) -> None:
        """Release resources."""
        self._container_client = None

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
