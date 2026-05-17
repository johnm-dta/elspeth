"""Azure Blob Storage source plugin for ELSPETH.

Loads rows from Azure Blob containers. Supports CSV, JSON array, and JSONL formats.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import time
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import structlog
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts import CallStatus, CallType, PluginSchema, SourceRow
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.contract_builder import ContractBuilder
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.core.identifiers import validate_field_names
from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sources.field_normalization import FieldResolution, normalize_field_name, resolve_field_names
from elspeth.plugins.sources.json_source import (
    _contains_surrogateescape_chars,
    _reject_nonfinite_constant,
    _surrogateescape_line_to_bytes,
)

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient

logger = structlog.get_logger(__name__)


class CSVOptions(BaseModel):
    """CSV parsing options."""

    model_config = {"extra": "forbid"}

    delimiter: str = ","
    has_header: bool = True
    encoding: str = "utf-8"

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


class JSONOptions(BaseModel):
    """JSON parsing options."""

    model_config = {"extra": "forbid"}

    encoding: str = "utf-8"
    data_key: str | None = None

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, v: str) -> str:
        import codecs

        try:
            codecs.lookup(v)
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {v!r}") from exc
        return v


class AzureBlobSourceConfig(DataPluginConfig):
    """Configuration for Azure Blob source plugin.

    Extends DataPluginConfig which requires schema configuration.
    Unlike file-based sources, does not extend PathConfig (no local file path).

    Supports four authentication methods (mutually exclusive):
    1. connection_string - Simple connection string auth (default)
    2. sas_token + account_url - Shared Access Signature token
    3. use_managed_identity + account_url - Azure Managed Identity
    4. tenant_id + client_id + client_secret + account_url - Service Principal

    _plugin_component_type overrides DataPluginConfig (None) because this
    config extends DataPluginConfig directly, bypassing SourceDataConfig.

    Example configurations:

        # Option 1: Connection string (simplest)
        connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
        container: "my-container"
        blob_path: "data/input.csv"

        # Option 2: SAS token
        sas_token: "${AZURE_STORAGE_SAS_TOKEN}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "data/input.csv"

        # Option 3: Managed Identity (for Azure-hosted workloads)
        use_managed_identity: true
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "data/input.csv"

        # Option 4: Service Principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
        account_url: "https://mystorageaccount.blob.core.windows.net"
        container: "my-container"
        blob_path: "data/input.csv"
    """

    _plugin_component_type: ClassVar[str | None] = "source"

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
    columns: list[str] | None = Field(
        default=None,
        description="Explicit column names for headerless CSV blobs",
    )
    field_mapping: dict[str, str] | None = Field(
        default=None,
        description="Override specific normalized field names",
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
            sas_token=self.sas_token,
            use_managed_identity=self.use_managed_identity,
            account_url=self.account_url,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        return self

    @model_validator(mode="after")
    def validate_field_normalization_options(self) -> Self:
        """Validate source-boundary field normalization options."""
        if self.format != "csv":
            if self.columns is not None:
                raise ValueError("columns is only supported for CSV format")
        else:
            if self.csv_options.has_header and self.columns is not None:
                raise ValueError("columns requires csv_options.has_header: false for headerless CSV blobs.")

        if self.format == "csv" and self.columns is not None:
            validate_field_names(self.columns, "columns")

        if self.field_mapping is not None and self.field_mapping:
            validate_field_names(list(self.field_mapping.values()), "field_mapping values")

        return self

    def get_auth_config(self) -> AzureAuthConfig:
        """Get the AzureAuthConfig for this source configuration.

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

    @field_validator("on_validation_failure")
    @classmethod
    def validate_on_validation_failure(cls, v: str) -> str:
        """Ensure on_validation_failure is not empty."""
        if not v or not v.strip():
            raise ValueError("on_validation_failure must be a sink name or 'discard'")
        return v.strip()


# Rebuild model to resolve forward references for dynamic module loading
AzureBlobSourceConfig.model_rebuild()


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
        - Observed: {"mode": "observed"} - accept any fields
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]}
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - at least these fields

    Three-tier trust model:
        - Azure Blob SDK calls = EXTERNAL SYSTEM -> wrap with try/except
        - Row data parsing/validation = THEIR DATA -> wrap, quarantine failures
        - Our internal state = OUR CODE -> let it crash
    """

    name = "azure_blob"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:525397539f159571"
    config_model = AzureBlobSourceConfig

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AzureBlobSourceConfig.from_dict(config, plugin_name=self.name)

        # Store auth config for creating clients
        self._auth_config = cfg.get_auth_config()
        self._container = cfg.container
        self._blob_path = cfg.blob_path
        self._format = cfg.format
        self._csv_options = cfg.csv_options
        self._json_options = cfg.json_options
        self._columns = cfg.columns
        self._field_mapping = cfg.field_mapping
        self._field_resolution: FieldResolution | None = None
        self._contract_builder: ContractBuilder | None = None

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        self._schema_config = cfg.schema_config
        self._initialize_declared_guaranteed_fields(self._schema_config)

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

        # Create schema contract for PipelineRow support.
        # CSV and JSON/JSONL need source-boundary field_resolution before
        # unlocked contracts can infer fields, so those builders are created
        # during load() once the first valid row establishes the mapping.
        if self._format == "csv":
            # CSV needs field_resolution from headers - defer contract creation to load()
            pass
        else:
            # JSON/JSONL normalize row keys when the first row is processed.
            # FIXED schemas can be locked immediately; FLEXIBLE/OBSERVED defer
            # builder creation until field_resolution is known.
            initial_contract = create_contract_from_config(self._schema_config)
            if initial_contract.locked:
                self.set_schema_contract(initial_contract)

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

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        """Load rows from Azure Blob Storage.

        Each row is validated against the configured schema:
        - Valid rows are yielded as SourceRow.valid()
        - Invalid rows are yielded as SourceRow.quarantined()

        For FLEXIBLE/OBSERVED schemas, the first valid row locks the contract with
        inferred types. Subsequent rows validate against the locked contract.

        Yields:
            SourceRow for each row (valid or quarantined).

        Raises:
            ImportError: If azure-storage-blob is not installed.
            azure.core.exceptions.ResourceNotFoundError: If blob does not exist.
            azure.core.exceptions.ClientAuthenticationError: If connection fails.
        """
        # Track first valid row for FLEXIBLE/OBSERVED type inference
        self._first_valid_row_processed = False
        # EXTERNAL SYSTEM: Azure Blob SDK calls - wrap with try/except
        # Record call for audit trail (ctx.operation_id is set by orchestrator)
        start_time = time.perf_counter()
        try:
            blob_client = self._get_blob_client()
            blob_data = blob_client.download_blob().readall()
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record successful blob download in audit trail.
            try:
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data={
                        "operation": "download_blob",
                        "container": self._container,
                        "blob_path": self._blob_path,
                    },
                    response_data={"size_bytes": len(blob_data)},
                    latency_ms=latency_ms,
                    provider="azure_blob_storage",
                )
            except Exception as exc:
                raise AuditIntegrityError(
                    f"Failed to record successful blob download to audit trail "
                    f"(container={self._container!r}, blob_path={self._blob_path!r}). "
                    f"Download completed but audit record is missing."
                ) from exc
        except AuditIntegrityError:
            raise  # Audit failure — do not misattribute as download error
        except ImportError:
            # Re-raise ImportError as-is for clear dependency messaging
            raise
        except (TypeError, AttributeError, KeyError, NameError, ValueError):
            raise  # Programming errors in our auth/client code — crash to surface the bug
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record failed blob download in audit trail
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data={
                    "operation": "download_blob",
                    "container": self._container,
                    "blob_path": self._blob_path,
                },
                error={"type": type(e).__name__, "message": str(e)},
                latency_ms=latency_ms,
                provider="azure_blob_storage",
            )

            # Azure SDK errors (ResourceNotFoundError, ClientAuthenticationError, etc.)
            # are external system errors - propagate with context
            raise RuntimeError(f"Failed to download blob '{self._blob_path}' from container '{self._container}': {e}") from e

        # Log blob download for operator visibility
        blob_size_kb = len(blob_data) / 1024
        if blob_size_kb >= 1024:
            size_str = f"{blob_size_kb / 1024:.1f} MB"
        else:
            size_str = f"{blob_size_kb:.1f} KB"
        logger.info(
            "blob_downloaded",
            blob_path=self._blob_path,
            container=self._container,
            size=size_str,
        )

        # Parse blob content based on format
        if self._format == "csv":
            yield from self._load_csv(blob_data, ctx)
        elif self._format == "json":
            yield from self._load_json_array(blob_data, ctx)
        elif self._format == "jsonl":
            yield from self._load_jsonl(blob_data, ctx)

        # CRITICAL: keep contract state consistent when no valid rows were seen.
        # Mirrors CSVSource behavior for all-invalid/empty inputs.
        if not self._first_valid_row_processed:
            if self._contract_builder is not None:
                self.set_schema_contract(self._contract_builder.contract.with_locked())
            elif self.get_schema_contract() is None:
                self.set_schema_contract(create_contract_from_config(self._schema_config).with_locked())

    def _load_csv(self, blob_data: bytes, ctx: SourceContext) -> Iterator[SourceRow]:
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
            error_msg = f"Failed to decode CSV blob as {encoding}: {e}"
            raw_row = {
                "container": self._container,
                "blob_path": self._blob_path,
                "error": error_msg,
            }
            ctx.record_validation_error(
                row=raw_row,
                error=error_msg,
                schema_mode="parse",
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=raw_row,
                    error=error_msg,
                    destination=self._on_validation_failure,
                )
            return

        # Parse CSV row-by-row using csv.reader for per-row error handling.
        # This allows quarantining individual bad rows instead of the entire file.
        reader = csv.reader(io.StringIO(text_data), delimiter=delimiter)

        # Track a peeked first data row (used for headerless CSV with no schema)
        first_data_row: list[str] | None = None

        # Determine headers
        if has_header:
            try:
                raw_headers = next(reader)
            except StopIteration:
                # Empty file — quarantine as structural failure (Tier 3)
                raw_row = {
                    "__raw_blob_preview__": text_data[:200],
                    "__encoding__": encoding,
                }
                error_msg = "CSV parse error: empty file contains no header row"
                ctx.record_validation_error(
                    row=raw_row,
                    error=error_msg,
                    schema_mode="parse",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=raw_row,
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                return
            except csv.Error as e:
                # Header parse failure at source boundary (Tier 3)
                raw_row = {
                    "__raw_blob_preview__": text_data[:200],
                    "__encoding__": encoding,
                }
                error_msg = f"CSV parse error in blob header: {e}"
                ctx.record_validation_error(
                    row=raw_row,
                    error=error_msg,
                    schema_mode="parse",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=raw_row,
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                return

            self._field_resolution = resolve_field_names(
                raw_headers=raw_headers,
                field_mapping=self._field_mapping,
                columns=None,
            )
            headers = self._field_resolution.final_headers
        elif self._columns is not None:
            self._field_resolution = resolve_field_names(
                raw_headers=None,
                field_mapping=self._field_mapping,
                columns=self._columns,
            )
            headers = self._field_resolution.final_headers
        else:
            # Headerless CSV with schema-defined field names
            if not self._schema_config.is_observed and self._schema_config.fields:
                schema_names = [field_def.name for field_def in self._schema_config.fields]
                self._field_resolution = resolve_field_names(
                    raw_headers=None,
                    field_mapping=self._field_mapping,
                    columns=schema_names,
                )
                headers = self._field_resolution.final_headers
            else:
                # No headers, no columns, no schema — peek at first row
                # to generate numeric column names (matching pandas behavior)
                try:
                    first_row = next(reader)
                except StopIteration:
                    return  # Empty headerless file — no data to process
                numeric_names = [str(i) for i in range(len(first_row))]
                headers = tuple(numeric_names)
                # Push the first row back by re-creating the reader chain
                # We'll process first_row manually, then continue with reader
                first_data_row = first_row

        expected_count = len(headers)

        # Create contract now that field_resolution is known (CSV path)
        if self._contract_builder is None and self._format == "csv":
            initial_contract = create_contract_from_config(
                self._schema_config,
                field_resolution=self._field_resolution.resolution_mapping if self._field_resolution else None,
            )
            if initial_contract.locked:
                self.set_schema_contract(initial_contract)
            else:
                self._contract_builder = ContractBuilder(initial_contract)

        # Process data rows with per-row error handling.
        # csv.Error can corrupt parser state, so we stop on csv.Error (matching CSVSource).
        # Column count mismatches quarantine the individual row and continue.
        # If first_data_row was peeked (headerless, no schema), process it first.
        row_source: Iterator[list[str]] = itertools.chain([first_data_row], reader) if first_data_row is not None else reader
        row_count = 0
        while True:
            try:
                values = next(row_source)
            except StopIteration:
                break
            except csv.Error as e:
                # csv.Error can leave parser in corrupted state — stop processing
                row_count += 1
                physical_line = reader.line_num
                raw_row = {
                    "__raw_line__": "(unparseable due to csv.Error)",
                    "__line_number__": str(physical_line),
                    "__row_number__": str(row_count),
                }
                error_msg = (
                    f"CSV parse error at line {physical_line}: {e}. "
                    f"Stopping blob processing — csv.Error can corrupt parser state, "
                    f"making subsequent rows untrustworthy."
                )
                ctx.record_validation_error(
                    row=raw_row,
                    error=error_msg,
                    schema_mode="parse",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=raw_row,
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                break

            # Skip blank lines
            if not values:
                continue

            row_count += 1
            physical_line = reader.line_num

            # Column count validation — quarantine malformed rows individually
            if len(values) != expected_count:
                raw_row = {
                    "__raw_line__": delimiter.join(values),
                    "__line_number__": str(physical_line),
                    "__row_number__": str(row_count),
                }
                error_msg = f"CSV parse error at line {physical_line}: expected {expected_count} fields, got {len(values)}"
                ctx.record_validation_error(
                    row=raw_row,
                    error=error_msg,
                    schema_mode="parse",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=raw_row,
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                continue

            # Build row dict — csv.reader returns strings, matching dtype=str behavior
            row = dict(zip(headers, values, strict=True))
            yield from self._validate_and_yield(row, ctx)

        # Log row count for operator visibility
        logger.info("csv_blob_parsed", rows_encountered=row_count, blob_path=self._blob_path)

    def _load_json_array(self, blob_data: bytes, ctx: SourceContext) -> Iterator[SourceRow]:
        """Load rows from JSON array blob data.

        Args:
            blob_data: Raw bytes from blob download.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow for each row (valid or quarantined).

        """
        # Access typed config fields directly - they have defaults from JSONOptions
        encoding = self._json_options.encoding
        data_key = self._json_options.data_key

        def _record_file_level_error(error_msg: str, schema_mode: str) -> Iterator[SourceRow]:
            raw_row = {
                "container": self._container,
                "blob_path": self._blob_path,
                "error": error_msg,
            }
            ctx.record_validation_error(
                row=raw_row,
                error=error_msg,
                schema_mode=schema_mode,
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=raw_row,
                    error=error_msg,
                    destination=self._on_validation_failure,
                )

        # THEIR DATA: JSON parsing - quarantine failures at boundary
        try:
            text_data = blob_data.decode(encoding)
        except UnicodeDecodeError as e:
            error_msg = f"Failed to decode blob as {encoding}: {e}"
            yield from _record_file_level_error(error_msg, "parse")
            return

        try:
            data = json.loads(text_data, parse_constant=_reject_nonfinite_constant)
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Invalid JSON in blob: {e}"
            yield from _record_file_level_error(error_msg, "parse")
            return

        # Extract from nested key if specified
        if data_key:
            if not isinstance(data, dict):
                error_msg = f"Cannot extract data_key '{data_key}': expected JSON object, got {type(data).__name__}"
                yield from _record_file_level_error(error_msg, "parse")
                return
            if data_key not in data:
                error_msg = f"data_key '{data_key}' not found in JSON object"
                yield from _record_file_level_error(error_msg, "parse")
                return
            data = data[data_key]

        if not isinstance(data, list):
            error_msg = f"Expected JSON array, got {type(data).__name__}"
            yield from _record_file_level_error(error_msg, "parse")
            return

        # Log row count for operator visibility
        logger.info("json_blob_parsed", row_count=len(data), blob_path=self._blob_path)

        for row in data:
            yield from self._validate_and_yield(row, ctx)

    def _load_jsonl(self, blob_data: bytes, ctx: SourceContext) -> Iterator[SourceRow]:
        """Load rows from JSONL (newline-delimited JSON) blob data.

        Args:
            blob_data: Raw bytes from blob download.
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow for each row (valid or quarantined).
        """
        # Access typed config field directly - it has a default from JSONOptions
        encoding = self._json_options.encoding

        non_empty_count = 0
        line_num = 0
        try:
            text_stream = io.TextIOWrapper(
                io.BytesIO(blob_data),
                encoding=encoding,
                errors="surrogateescape",
                newline="",
            )
            for line_num, raw_line in enumerate(text_stream, start=1):
                line = raw_line.strip()
                if not line:  # Skip empty lines
                    continue

                non_empty_count += 1

                if _contains_surrogateescape_chars(raw_line):
                    raw_bytes = _surrogateescape_line_to_bytes(raw_line.rstrip("\r\n"), encoding)
                    raw_row = {
                        "__raw_bytes_hex__": raw_bytes.hex(),
                        "__line_number__": line_num,
                    }
                    error_msg = f"JSON parse error at line {line_num}: invalid {encoding} encoding"
                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",
                        destination=self._on_validation_failure,
                    )
                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    continue

                # Catch JSON parse errors at the trust boundary
                try:
                    row = json.loads(line, parse_constant=_reject_nonfinite_constant)
                except (json.JSONDecodeError, ValueError) as e:
                    # External data parse failure - quarantine, don't crash
                    # Store raw line + metadata for audit traceability
                    raw_row = {"__raw_line__": line, "__line_number__": str(line_num)}
                    error_msg = f"JSON parse error at line {line_num}: {e}"

                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",  # Distinct from schema validation
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    continue

                yield from self._validate_and_yield(row, ctx)
        except UnicodeDecodeError as e:
            # Some codecs can still raise on truncated byte sequences while
            # reading. Treat that as a line-level external parse failure.
            error_line = line_num + 1
            error_msg = f"JSON parse error at line {error_line}: invalid {encoding} encoding ({e})"
            raw_row = {
                "container": self._container,
                "blob_path": self._blob_path,
                "__line_number__": error_line,
            }
            ctx.record_validation_error(
                row=raw_row,
                error=error_msg,
                schema_mode="parse",
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=raw_row,
                    error=error_msg,
                    destination=self._on_validation_failure,
                )
            return

        logger.info("jsonl_blob_parsed", line_count=non_empty_count, blob_path=self._blob_path)

    def _normalize_row_keys(self, row: Any) -> Mapping[str, Any]:
        """Normalize JSON/JSONL object keys at the source boundary."""
        try:
            row_items = list(row.items())
        except AttributeError:
            raise ValueError(f"Expected JSON object, got {type(row).__name__}") from None

        raw_keys = [key for key, _ in row_items]

        if self._field_resolution is None:
            self._field_resolution = resolve_field_names(
                raw_headers=raw_keys,
                field_mapping=self._field_mapping,
                columns=None,
            )

            if self._contract_builder is None and self.get_schema_contract() is None:
                initial_contract = create_contract_from_config(
                    self._schema_config,
                    field_resolution=self._field_resolution.resolution_mapping,
                )
                self._contract_builder = ContractBuilder(initial_contract)

        mapping = self._field_resolution.resolution_mapping
        normalized: dict[str, Any] = {}
        has_unmapped_fields = False
        for key, value in row_items:
            if key in mapping:
                normalized[mapping[key]] = value
            else:
                normalized[normalize_field_name(key)] = value
                has_unmapped_fields = True

        if has_unmapped_fields:
            self._field_resolution = resolve_field_names(
                raw_headers=list(row.keys()),
                field_mapping=self._field_mapping,
                columns=None,
            )

        return normalized

    def _validate_and_yield(self, row: Any, ctx: SourceContext) -> Iterator[SourceRow]:
        """Validate a row and yield if valid, otherwise quarantine.

        For FLEXIBLE/OBSERVED schemas, the first valid row triggers type inference and
        locks the contract. Subsequent rows validate against the locked contract.

        Args:
            row: Row data to validate. May be non-dict for malformed external
                 data (e.g., JSON arrays containing primitives).
            ctx: Plugin context for recording validation errors.

        Yields:
            SourceRow.valid() if valid, SourceRow.quarantined() if invalid.
        """
        try:
            row_to_validate = self._normalize_row_keys(row) if self._format in ("json", "jsonl") else row
        except ValueError as e:
            error_msg = f"Field normalization failed: {e}"
            ctx.record_validation_error(
                row=row,
                error=error_msg,
                schema_mode="field_normalization",
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=error_msg,
                    destination=self._on_validation_failure,
                )
            return

        try:
            # Validate and potentially coerce row data
            validated = self._schema_class.model_validate(row_to_validate)
            validated_row = validated.to_row()

            # For FLEXIBLE/OBSERVED schemas, process first valid row to lock contract
            if self._contract_builder is not None and not self._first_valid_row_processed:
                if self._field_resolution is None:
                    if self._format in ("json", "jsonl"):
                        raise ValueError("field_resolution must be established before first-row contract inference")
                    field_resolution_map: Mapping[str, str] = {k: k for k in validated_row}
                else:
                    field_resolution_map = self._field_resolution.resolution_mapping

                self._contract_builder.process_first_row(validated_row, field_resolution_map)
                self.set_schema_contract(self._contract_builder.contract)
                self._first_valid_row_processed = True

            # Validate against locked contract to catch type drift on inferred
            # fields. Pydantic extra="allow" accepts any type for extras — the
            # contract knows inferred types from the first row and enforces here.
            contract = self.require_schema_contract()
            if contract.locked:
                violations = contract.validate(validated_row)
                if violations:
                    error_msg = "; ".join(str(v) for v in violations)
                    ctx.record_validation_error(
                        row=validated_row,
                        error=error_msg,
                        schema_mode=self._schema_config.mode,
                        destination=self._on_validation_failure,
                    )
                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=validated_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    return

            yield SourceRow.valid(validated_row, contract=contract)
        except ValidationError as e:
            # Record validation failure in audit trail
            # This is a trust boundary: external data may be invalid
            ctx.record_validation_error(
                row=row_to_validate,
                error=str(e),
                schema_mode=self._schema_config.mode,
                destination=self._on_validation_failure,
            )

            # Yield quarantined row for routing to configured sink
            # If "discard", don't yield - row is intentionally dropped
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row_to_validate,
                    error=str(e),
                    destination=self._on_validation_failure,
                )

    def close(self) -> None:
        """Release resources (no-op for Azure Blob source)."""
        self._blob_client = None

    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
        """Return field resolution mapping for audit trail."""
        if self._field_resolution is None:
            return None

        return (
            self._field_resolution.resolution_mapping,
            self._field_resolution.normalization_version,
        )
