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
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from jinja2 import StrictUndefined, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import ArtifactDescriptor, CallStatus, CallType, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.header_modes import HeaderMode, parse_header_mode
from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import DataPluginConfig, validate_headers_value
from elspeth.plugins.infrastructure.display_headers import (
    apply_display_headers,
    get_effective_display_headers,
    init_display_headers,
    resolve_contract_from_context_if_needed,
    resolve_display_headers_if_needed,
    set_resume_field_resolution,
)
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient


class CSVWriteOptions(BaseModel):
    """CSV writing options."""

    model_config = {"extra": "forbid"}

    delimiter: str = ","
    encoding: str = "utf-8"
    include_header: bool = True

    @field_validator("delimiter")
    @classmethod
    def _validate_delimiter(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError(f"delimiter must be a single character, got {v!r}")
        return v

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, v: str) -> str:
        import codecs

        try:
            codecs.lookup(v)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {v!r}") from exc
        return v


class AzureBlobSinkConfig(DataPluginConfig):
    """Configuration for Azure Blob sink plugin.

    Extends DataPluginConfig which requires schema configuration.
    Unlike file-based sinks, does not extend PathConfig (no local file path).

    _plugin_component_type overrides DataPluginConfig (None) because this
    config extends DataPluginConfig directly, bypassing SinkPathConfig.

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

    _plugin_component_type: ClassVar[str | None] = "sink"

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
    headers: str | dict[str, str] | None = Field(
        default=None,
        description="Header output mode: 'normalized', 'original', or {field: header} mapping",
    )

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, v: str | dict[str, str] | None) -> str | dict[str, str] | None:
        return validate_headers_value(v)

    @property
    def headers_mode(self) -> HeaderMode:
        if self.headers is not None:
            return parse_header_mode(self.headers)
        return HeaderMode.NORMALIZED

    @property
    def headers_mapping(self) -> dict[str, str] | None:
        if isinstance(self.headers, dict):
            return self.headers
        return None

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

    @model_validator(mode="after")
    def validate_blob_path_template(self) -> Self:
        """Pre-compile blob_path as Jinja2 template to catch syntax errors.

        Moved from AzureBlobSink.__init__ so from_dict() catches it
        (pre-validation / engine-validation agreement).
        """
        env = SandboxedEnvironment(undefined=StrictUndefined)
        try:
            env.from_string(self.blob_path)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid blob_path template: {e}") from e
        return self


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
    source_file_hash: str | None = "sha256:1816d1d5730f7f52"
    config_model = AzureBlobSinkConfig
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: Azure Blobs are immutable - cannot append
    supports_resume: bool = False

    def configure_for_resume(self) -> None:
        """Azure Blob sink does not support resume.

        Azure Blobs are immutable - once uploaded, they cannot be appended to.
        A new blob would need to be created with combined content, which is
        not supported in the resume flow.

        Raises:
            NotImplementedError: Always, as Azure Blobs cannot be appended.
        """
        raise NotImplementedError(
            "AzureBlobSink does not support resume. "
            "Azure Blobs are immutable and cannot be appended to. "
            "Consider using a different blob_path template (e.g., '{{ run_id }}/output.csv') "
            "to create unique blobs per run, or use a local file sink for resumable pipelines."
        )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AzureBlobSinkConfig.from_dict(config, plugin_name=self.name)

        # Store auth config for creating clients
        self._auth_config = cfg.get_auth_config()
        self._container = cfg.container
        self._blob_path_template = cfg.blob_path
        self._format = cfg.format
        self._overwrite = cfg.overwrite

        # Pre-compile blob path template at init for runtime use.
        # Syntax validation is now handled by AzureBlobSinkConfig.validate_blob_path_template
        # model_validator — from_dict() above already proved the template is valid.
        env = SandboxedEnvironment(undefined=StrictUndefined)
        self._blob_path_compiled = env.from_string(self._blob_path_template)

        # CSV options are already validated Pydantic model
        self._csv_options = cfg.csv_options

        # Display header state (shared module handles all modes)
        init_display_headers(self, cfg.headers_mode, cfg.headers_mapping)

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
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

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        # Lazy-loaded clients
        self._container_client: ContainerClient | None = None
        # Buffer rows across write() calls so each upload represents full run output.
        self._buffered_rows: list[dict[str, Any]] = []
        # Freeze rendered path on first write; subsequent writes target same blob.
        self._resolved_blob_path: str | None = None
        # Track whether this sink instance has successfully uploaded at least once.
        # Needed to preserve overwrite=False first-write protection while allowing
        # in-run rewrites of the same blob for accumulation.
        self._has_uploaded: bool = False

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

    def _render_blob_path(self, ctx: SinkContext) -> str:
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
        # Use pre-compiled template (structurally validated in __init__).
        # Render-time UndefinedError from typos like {{ runid }} still fails
        # fast here — StrictUndefined was set at compile time.
        return self._blob_path_compiled.render(
            run_id=ctx.run_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    def _get_or_init_blob_path(self, ctx: SinkContext) -> str:
        """Get stable blob path for this sink instance.

        The path is rendered once on first write and reused thereafter so
        repeated write() calls in the same run update the same blob.
        """
        if self._resolved_blob_path is None:
            self._resolved_blob_path = self._render_blob_path(ctx)
        return self._resolved_blob_path

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

    def _get_fieldnames_from_schema_or_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        """Get fieldnames from schema or cumulative row keys.

        Field selection depends on schema mode:
        - fixed: Only declared fields (extras rejected)
        - flexible: Declared fields first, then extras seen across rows
        - observed: All fields seen across rows
        """
        ordered_keys: list[str] = []
        seen_keys: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen_keys:
                    seen_keys.add(key)
                    ordered_keys.append(key)

        if self._schema_config.is_observed:
            # Observed mode: infer all fields from all row keys in first-seen order.
            return ordered_keys
        elif self._schema_config.fields:
            # Explicit schema: start with declared field names in schema order
            declared_fields = [field_def.name for field_def in self._schema_config.fields]
            declared_set = set(declared_fields)

            if self._schema_config.mode == "flexible":
                # Flexible mode: declared fields first, then extras from all rows.
                extras = [key for key in ordered_keys if key not in declared_set]
                return declared_fields + extras
            else:
                # Fixed mode: only declared fields
                return declared_fields
        else:
            # Fallback (shouldn't happen with valid config): use all seen keys.
            return ordered_keys

    def _get_field_names_and_display(self, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """Get data field names and display names for CSV output."""
        data_fields = self._get_fieldnames_from_schema_or_rows(rows)

        display_map = get_effective_display_headers(self)
        if display_map is None:
            return data_fields, data_fields

        display_fields = [display_map[field] if field in display_map else field for field in data_fields]
        return data_fields, display_fields

    def _serialize_csv(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to CSV bytes.

        Validates that all rows conform to the established fieldnames BEFORE
        any serialization occurs. This prevents partial serialization failures
        that would leave the buffer in an inconsistent state.
        """
        output = io.StringIO()

        data_fields, display_fields = self._get_field_names_and_display(rows)

        # Preflight validation: reject extra fields before serialization.
        # Without this, DictWriter raises mid-batch on extras, producing
        # partial CSV content in the buffer.
        if not self._schema_config.is_observed:
            allowed = set(data_fields)
            for i, row in enumerate(rows):
                extra = sorted(set(row) - allowed)
                if extra:
                    raise ValueError(
                        f"AzureBlobSink CSV row {i} has unexpected fields: {extra}. This indicates an upstream transform/schema bug."
                    )

        writer = csv.DictWriter(
            output,
            fieldnames=data_fields,
            delimiter=self._csv_options.delimiter,
        )

        if self._csv_options.include_header:
            if display_fields != data_fields:
                header_writer = csv.writer(output, delimiter=self._csv_options.delimiter)
                header_writer.writerow(display_fields)
            else:
                writer.writeheader()

        for row in rows:
            writer.writerow(row)

        return output.getvalue().encode(self._csv_options.encoding)

    def _serialize_json(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to JSON array bytes."""
        return json.dumps(rows, indent=2, allow_nan=False).encode("utf-8")

    def _serialize_jsonl(self, rows: list[dict[str, Any]]) -> bytes:
        """Serialize rows to JSONL bytes (newline-delimited JSON)."""
        lines = [json.dumps(row, allow_nan=False) for row in rows]
        return "\n".join(lines).encode("utf-8")

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        set_resume_field_resolution(self, resolution_mapping)

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
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
        resolve_contract_from_context_if_needed(self, ctx)
        resolve_display_headers_if_needed(self, ctx)

        if not rows:
            # Still render the path for consistent audit trail
            rendered_path = self._get_or_init_blob_path(ctx)
            return SinkWriteResult(
                artifact=ArtifactDescriptor(
                    artifact_type="file",
                    path_or_uri=f"azure://{self._container}/{rendered_path}",
                    content_hash=hashlib.sha256(b"").hexdigest(),
                    size_bytes=0,
                )
            )

        output_rows = rows
        if self._format in {"json", "jsonl"}:
            output_rows = apply_display_headers(self, rows)

        # Render the blob path once per instance and reuse it across writes.
        rendered_path = self._get_or_init_blob_path(ctx)

        # Build candidate cumulative rows for this upload, but only commit them
        # to sink state after external upload succeeds (retry-idempotent).
        candidate_rows = [*self._buffered_rows, *(row.copy() for row in output_rows)]

        # Serialize rows to bytes (OUR CODE - let it crash on bugs)
        content = self._serialize_rows(candidate_rows)
        # Compute content hash before upload
        content_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)

        # EXTERNAL SYSTEM: Azure Blob SDK calls - wrap with try/except
        # Record call for audit trail (ctx.operation_id is set by executor)
        # Capture overwrite flag BEFORE the try block so it reflects the actual
        # value used for the upload call, not the post-mutation state.
        upload_overwrite = self._overwrite or self._has_uploaded
        start_time = time.perf_counter()
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(rendered_path)

            # Upload with overwrite policy enforced atomically by Azure SDK.
            # When overwrite=False, upload_blob raises ResourceExistsError server-side,
            # avoiding the TOCTOU race of a separate exists() check.
            blob_client.upload_blob(content, overwrite=upload_overwrite)
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Mark external blob existence immediately after upload so retries
            # can safely overwrite the same blob if post-upload steps fail.
            self._has_uploaded = True

        except ImportError:
            # Re-raise ImportError as-is for clear dependency messaging
            raise
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors in our auth/client code — crash to surface the bug
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_data: dict[str, Any] = {"type": type(e).__name__, "message": str(e)}
            if type(e).__name__ == "ResourceExistsError":
                # Preserve explicit reason for overwrite=False conflicts.
                error_data["reason"] = "blob_exists"

            # Record failed blob upload in audit trail
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data={
                    "operation": "upload_blob",
                    "container": self._container,
                    "blob_path": rendered_path,
                    "overwrite": upload_overwrite,
                },
                error=error_data,
                latency_ms=latency_ms,
                provider="azure_blob_storage",
            )

            # Convert ResourceExistsError (overwrite=False) to ValueError
            # for consistent API. Check class name to avoid importing azure SDK at top level.
            if type(e).__name__ == "ResourceExistsError":
                raise ValueError(f"Blob '{rendered_path}' already exists and overwrite=False") from e

            # Azure SDK errors are external system errors - propagate with context.
            # Use RuntimeError wrapper instead of type(e)(...) because Azure SDK
            # exceptions (HttpResponseError, ResourceExistsError, etc.) have
            # multi-parameter constructors that won't accept a single string.
            raise RuntimeError(f"Failed to upload blob '{rendered_path}' to container '{self._container}': {e}") from e

        # Record successful blob upload in audit trail.
        # Outside the upload try/except so ctx.record_call errors don't
        # silently lose the audit record for a successful upload.
        try:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data={
                    "operation": "upload_blob",
                    "container": self._container,
                    "blob_path": rendered_path,
                    "overwrite": upload_overwrite,
                },
                response_data={
                    "size_bytes": size_bytes,
                    "content_hash": content_hash,
                },
                latency_ms=latency_ms,
                provider="azure_blob_storage",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful blob upload to audit trail "
                f"(container={self._container!r}, blob={rendered_path!r}). "
                f"Upload completed but audit record is missing."
            ) from exc
        # Commit cumulative in-memory buffer only after full success path
        # (upload + audit recording) to keep write retries idempotent.
        self._buffered_rows = candidate_rows

        return SinkWriteResult(
            artifact=ArtifactDescriptor(
                artifact_type="file",
                path_or_uri=f"azure://{self._container}/{rendered_path}",
                content_hash=content_hash,
                size_bytes=size_bytes,
            )
        )

    def flush(self) -> None:
        """Flush buffered data.

        No-op for Azure Blob sink - durability is guaranteed by synchronous upload in write().

        Azure Blob Storage uploads in write() are synchronous and complete before
        returning. The blob is committed to Azure's redundant storage (LRS/GRS) when
        write() returns, providing the same durability guarantee as an explicit flush().

        This means data survives:
        - Process crash (blob upload already completed)
        - Azure datacenter failure (redundant storage)
        - Network interruption (upload completed or failed, no partial state)

        Future enhancement: Support async uploads with explicit flush() for batching.
        """
        pass

    def close(self) -> None:
        """Release resources."""
        if self._container_client is not None:
            self._container_client.close()
        self._container_client = None
        self._buffered_rows = []
        self._resolved_blob_path = None
        self._has_uploaded = False
