# src/elspeth/plugins/azure/blob_source.py
"""Azure Blob Storage source plugin for ELSPETH.

Loads rows from Azure Blob containers. Supports CSV, JSON array, and JSONL formats.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts import CallStatus, CallType, PluginSchema, SourceRow
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.identifiers import validate_field_names
from elspeth.plugins.azure.auth import AzureAuthConfig
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.sources.field_normalization import FieldResolution, resolve_field_names

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient

logger = logging.getLogger(__name__)


def _reject_nonfinite_constant(value: str) -> None:
    """Reject non-standard JSON constants (NaN, Infinity, -Infinity)."""
    raise ValueError(f"Non-standard JSON constant '{value}' not allowed. Use null for missing values, not NaN/Infinity.")


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

    Supports four authentication methods (mutually exclusive):
    1. connection_string - Simple connection string auth (default)
    2. sas_token + account_url - Shared Access Signature token
    3. use_managed_identity + account_url - Azure Managed Identity
    4. tenant_id + client_id + client_secret + account_url - Service Principal

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
    normalize_fields: bool = Field(
        default=False,
        description="Normalize CSV headers to valid identifiers (only for CSV format)",
    )
    field_mapping: dict[str, str] | None = Field(
        default=None,
        description="Override specific normalized field names (requires normalize_fields or columns)",
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
        """Validate field normalization options for CSV format."""
        if self.format != "csv":
            if self.normalize_fields or self.columns is not None or self.field_mapping is not None:
                raise ValueError("normalize_fields, columns, and field_mapping are only supported for CSV format")
            return self

        if not self.csv_options.has_header and self.normalize_fields:
            raise ValueError("normalize_fields requires csv_options.has_header: true. Use columns for headerless CSV blobs.")

        if self.csv_options.has_header and self.columns is not None:
            raise ValueError("columns requires csv_options.has_header: false for headerless CSV blobs.")

        if self.columns is not None and self.normalize_fields:
            raise ValueError("normalize_fields cannot be used with columns config. The columns config already provides clean names.")

        if self.field_mapping is not None and not self.normalize_fields and self.columns is None:
            raise ValueError("field_mapping requires normalize_fields: true or columns config")

        if self.columns is not None:
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
        self._columns = cfg.columns
        self._normalize_fields = cfg.normalize_fields
        self._field_mapping = cfg.field_mapping
        self._field_resolution: FieldResolution | None = None

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
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

        # Create schema contract for PipelineRow support
        # Strategy depends on format and schema mode:
        # - CSV: needs field_resolution, so ContractBuilder created during load()
        # - JSON/JSONL with FIXED: contract locked immediately
        # - JSON/JSONL with FLEXIBLE/OBSERVED: ContractBuilder for first-row inference
        from elspeth.contracts.contract_builder import ContractBuilder
        from elspeth.contracts.schema_contract_factory import create_contract_from_config

        if self._format == "csv":
            # CSV needs field_resolution from headers - defer contract creation to load()
            self._contract_builder = None
        else:
            # JSON/JSONL - no field normalization, identity mapping
            initial_contract = create_contract_from_config(self._schema_config)
            if initial_contract.locked:
                # FIXED - contract ready immediately
                self.set_schema_contract(initial_contract)
                self._contract_builder = None
            else:
                # FLEXIBLE/OBSERVED - will lock after first valid row
                self._contract_builder = ContractBuilder(initial_contract)

        # Lazy-loaded blob client
        self._blob_client: BlobClient | None = None

        # PHASE 1: Validate self-consistency

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

            # Record successful blob download in audit trail
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
        except ImportError:
            # Re-raise ImportError as-is for clear dependency messaging
            raise
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
            raise type(e)(f"Failed to download blob '{self._blob_path}' from container '{self._container}': {e}") from e

        # Log blob download for operator visibility
        blob_size_kb = len(blob_data) / 1024
        if blob_size_kb >= 1024:
            size_str = f"{blob_size_kb / 1024:.1f} MB"
        else:
            size_str = f"{blob_size_kb:.1f} KB"
        logger.info(
            "Downloaded blob '%s' from container '%s' (%s)",
            self._blob_path,
            self._container,
            size_str,
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
        if not self._first_valid_row_processed and self._contract_builder is not None:
            self.set_schema_contract(self._contract_builder.contract.with_locked())

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

        # BUG-BLOB-01 fix: Wrap pandas CSV parsing to quarantine on structural errors
        # Even with pandas' robustness, severely malformed CSVs can cause parse failures
        try:
            # Use pandas for robust CSV parsing (consistent with CSVSource)
            header_arg = 0 if has_header else None

            # When headerless CSV with explicit schema, use schema field names
            names_arg = None
            if not has_header:
                if self._columns is not None:
                    names_arg = self._columns
                elif not self._schema_config.is_observed and self._schema_config.fields:
                    names_arg = [field_def.name for field_def in self._schema_config.fields]

            df = pd.read_csv(
                io.StringIO(text_data),
                delimiter=delimiter,
                header=header_arg,
                names=names_arg,  # Map columns to schema field names
                dtype=str,  # Keep all values as strings for consistent handling
                keep_default_na=False,  # Don't convert empty strings to NaN
                on_bad_lines="error",  # Quarantine parse failures; never silently drop malformed lines
            )
        except Exception as e:
            # Catastrophic CSV structure failure - entire file unparseable
            # This is rare with pandas but can happen with severely malformed files
            error_msg = f"CSV parse error: Unable to parse blob structure: {e}"
            raw_row = {"__raw_blob_preview__": text_data[:200], "__encoding__": encoding}

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
            return  # No rows to process if file is completely unparseable

        if has_header:
            raw_headers = [str(header) for header in df.columns]
            self._field_resolution = resolve_field_names(
                raw_headers=raw_headers,
                normalize_fields=self._normalize_fields,
                field_mapping=self._field_mapping,
                columns=None,
            )
            final_headers = self._field_resolution.final_headers
            if final_headers != raw_headers:
                df.columns = final_headers
        elif self._columns is not None:
            self._field_resolution = resolve_field_names(
                raw_headers=None,
                normalize_fields=self._normalize_fields,
                field_mapping=self._field_mapping,
                columns=self._columns,
            )
            final_headers = self._field_resolution.final_headers
            if list(df.columns) != final_headers:
                df.columns = final_headers

        # Create contract now that field_resolution is known (CSV path)
        if self._contract_builder is None and self._format == "csv":
            from elspeth.contracts.contract_builder import ContractBuilder
            from elspeth.contracts.schema_contract_factory import create_contract_from_config

            initial_contract = create_contract_from_config(
                self._schema_config,
                field_resolution=self._field_resolution.resolution_mapping if self._field_resolution else None,
            )
            if initial_contract.locked:
                self.set_schema_contract(initial_contract)
            else:
                self._contract_builder = ContractBuilder(initial_contract)

        # Log row count for operator visibility
        row_count = len(df)
        logger.info("Parsed %d rows from CSV blob '%s'", row_count, self._blob_path)

        # DataFrame columns are strings from CSV headers
        for record in df.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}
            yield from self._validate_and_yield(row, ctx)

    def _load_json_array(self, blob_data: bytes, ctx: PluginContext) -> Iterator[SourceRow]:
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
                yield from _record_file_level_error(error_msg, "structure")
                return
            if data_key not in data:
                error_msg = f"data_key '{data_key}' not found in JSON object"
                yield from _record_file_level_error(error_msg, "structure")
                return
            data = data[data_key]

        if not isinstance(data, list):
            error_msg = f"Expected JSON array, got {type(data).__name__}"
            yield from _record_file_level_error(error_msg, "structure")
            return

        # Log row count for operator visibility
        logger.info("Parsed %d rows from JSON array blob '%s'", len(data), self._blob_path)

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

        # Split lines and count non-empty for logging
        lines = text_data.splitlines()
        non_empty_count = sum(1 for line in lines if line.strip())
        logger.info("Parsed %d lines from JSONL blob '%s'", non_empty_count, self._blob_path)

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Catch JSON parse errors at the trust boundary
            try:
                row = json.loads(line, parse_constant=_reject_nonfinite_constant)
            except (json.JSONDecodeError, ValueError) as e:
                # External data parse failure - quarantine, don't crash
                # Store raw line + metadata for audit traceability
                raw_row = {"__raw_line__": line, "__line_number__": line_num}
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

    def _validate_and_yield(self, row: Any, ctx: PluginContext) -> Iterator[SourceRow]:
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
            # Validate and potentially coerce row data
            validated = self._schema_class.model_validate(row)
            validated_row = validated.to_row()

            # For FLEXIBLE/OBSERVED schemas, process first valid row to lock contract
            if self._contract_builder is not None and not self._first_valid_row_processed:
                # Use field_resolution from CSV if available, else identity mapping for JSON/JSONL
                if self._field_resolution is not None:
                    field_resolution_map = self._field_resolution.resolution_mapping
                else:
                    # JSON/JSONL without normalization - identity mapping
                    field_resolution_map = {k: k for k in validated_row}

                self._contract_builder.process_first_row(validated_row, field_resolution_map)
                self.set_schema_contract(self._contract_builder.contract)
                self._first_valid_row_processed = True

            # Validate against locked contract to catch type drift on inferred
            # fields. Pydantic extra="allow" accepts any type for extras — the
            # contract knows inferred types from the first row and enforces here.
            contract = self.get_schema_contract()
            if contract is not None and contract.locked:
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
                row=row,
                error=str(e),
                schema_mode=self._schema_config.mode,
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

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        """Return field resolution mapping for audit trail (CSV only)."""
        if self._field_resolution is None:
            return None

        return (
            self._field_resolution.resolution_mapping,
            self._field_resolution.normalization_version,
        )
