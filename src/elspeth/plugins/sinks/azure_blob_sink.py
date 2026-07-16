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

import base64
import csv
import io
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

from jinja2 import StrictUndefined, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.header_modes import HeaderMode, parse_header_mode
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionRequest,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import DataPluginConfig, validate_headers_value
from elspeth.plugins.infrastructure.display_headers import (
    apply_display_headers,
    get_effective_display_headers,
    init_display_headers,
    set_resume_field_resolution,
)
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sinks._remote_object_effects import (
    RemoteObjectEffectError,
    RemoteObjectObservation,
    RemoteObjectPreconditionError,
    inspect_remote_object,
    prepare_remote_object,
    reconcile_remote_observation,
    remote_commit_result,
    validate_remote_plan,
)

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
    max_blob_bytes: int = Field(
        default=256 * 1024 * 1024,
        gt=0,
        le=1024 * 1024 * 1024,
        strict=True,
        description="Maximum serialized Azure blob bytes",
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
        # Dispatch on the canonical header mode (single source of truth) rather
        # than re-inspecting the runtime type of the validated union field.
        # parse_header_mode() returns CUSTOM iff headers is a non-empty dict, so
        # the narrowing is exact.
        if self.headers_mode is HeaderMode.CUSTOM:
            return cast("dict[str, str]", self.headers)
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
        return reject_operator_required_placeholder_value(v, field_name="container")

    @field_validator("blob_path")
    @classmethod
    def validate_blob_path_not_empty(cls, v: str) -> str:
        """Validate that blob_path is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("blob_path cannot be empty")
        return reject_operator_required_placeholder_value(v, field_name="blob_path")

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
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:bfa2e83797938e0d"
    config_model = AzureBlobSinkConfig
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: Azure Blobs are immutable - cannot append
    supports_resume: bool = False

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        del cls, config, purpose
        return ResolvedSinkEffectMode("write")

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Writes pipeline rows to Azure Blob Storage as CSV, JSON, or JSONL.",
                composer_hints=(
                    "Configure exactly one auth path: connection_string, sas_token+account_url, managed identity+account_url, or service principal+account_url.",
                    "blob_path is a Jinja2 template; use stable run metadata such as run_id or timestamp instead of row values.",
                    "For CSV output, csv_options.include_header controls whether a header row is written.",
                    "Set headers to normalized, original, or an explicit mapping when downstream consumers need display names.",
                    "Azure Blob sink cannot resume append writes; choose unique blob paths for reruns or resumable workflows.",
                ),
            )
        return None

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
        self._max_blob_bytes = cfg.max_blob_bytes

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

    def _effect_blob_path(self, ctx: RestrictedSinkEffectContext) -> str:
        return self._blob_path_compiled.render(
            run_id=ctx.run_id,
            timestamp=ctx.run_started_at.isoformat(),
        )

    @staticmethod
    def _is_missing(error: BaseException) -> bool:
        return type(error).__name__ == "ResourceNotFoundError" or getattr(error, "status_code", None) == 404

    @staticmethod
    def _observation_from_properties(properties: object) -> RemoteObjectObservation:
        size = getattr(properties, "size", None)
        etag = getattr(properties, "etag", None)
        metadata_value = getattr(properties, "metadata", None)
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        content_hash = metadata.get("elspeth_content_sha256")
        effect_id = metadata.get("elspeth_effect_id")
        plan_hash = metadata.get("elspeth_plan_hash")
        protocol_version = metadata.get("elspeth_protocol_version")
        content_settings = getattr(properties, "content_settings", None)
        content_md5 = getattr(content_settings, "content_md5", None)
        checksum_b64 = base64.b64encode(content_md5).decode("ascii") if isinstance(content_md5, (bytes, bytearray)) else None
        return RemoteObjectObservation(
            exists=True,
            etag=etag if isinstance(etag, str) and etag else None,
            content_hash=content_hash if isinstance(content_hash, str) else None,
            size_bytes=size if type(size) is int and size >= 0 else None,
            effect_id=effect_id if isinstance(effect_id, str) else None,
            plan_hash=plan_hash if isinstance(plan_hash, str) else None,
            protocol_version=protocol_version if isinstance(protocol_version, str) else None,
            checksum_algorithm="md5" if checksum_b64 is not None else None,
            checksum_b64=checksum_b64,
        )

    def _observe_effect_target(self, blob_path: str) -> RemoteObjectObservation:
        try:
            properties = self._get_container_client().get_blob_client(blob_path).get_blob_properties()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            if self._is_missing(error):
                return RemoteObjectObservation(False, None, None, None)
            raise RemoteObjectPreconditionError("Azure blob inspection failed before effect dispatch") from None
        return self._observation_from_properties(properties)

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        blob_path = self._effect_blob_path(ctx)
        target = f"azure://{self._container}/{blob_path}"
        observation = self._observe_effect_target(blob_path)
        if observation.exists and not self._overwrite and request.predecessor_descriptor is None:
            raise ValueError(f"Blob '{blob_path}' already exists and overwrite=False") from None
        return inspect_remote_object(
            provider="azure_blob",
            target=target,
            request=request,
            observation=observation,
        )

    def _preflight_effect_members(
        self,
        effect_input: SinkEffectPipelineMembersInput,
    ) -> tuple[list[dict[str, Any]], tuple[int, ...], tuple[int, ...]]:
        accepted: list[int] = []
        diverted: list[int] = []
        diverted_keys: set[tuple[str, str]] = set()
        for member in effect_input.members:
            row = dict(member.row)
            output_row = apply_display_headers(self, [row])[0] if self._format in {"json", "jsonl"} else row
            try:
                self._serialize_rows([output_row])
            except (ValueError, TypeError, csv.Error, UnicodeError) as exc:
                reason = (
                    f"CSV encoding ({self._csv_options.encoding}) failed: {exc}"
                    if self._format == "csv"
                    else f"JSON serialization failed: {exc}"
                )
                self._divert_row(row, row_index=member.ordinal, reason=reason)
                diverted.append(member.ordinal)
                diverted_keys.add((member.token_id, member.row_id))
            else:
                accepted.append(member.ordinal)
        rows = [
            dict(member.row) for member in effect_input.target_snapshot_members if (member.token_id, member.row_id) not in diverted_keys
        ]
        if self._format in {"json", "jsonl"}:
            rows = apply_display_headers(self, rows)
        return rows, tuple(accepted), tuple(diverted)

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if type(request.effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("AzureBlobSink effects require pipeline member input")
        rows, accepted, diverted = self._preflight_effect_members(request.effect_input)
        content = self._serialize_rows(rows)
        evidence = request.inspection.evidence
        predecessor: ArtifactDescriptor | None = None
        if evidence.get("predecessor_declared") is True:
            observed_hash = evidence.get("observed_content_hash")
            observed_size = evidence.get("observed_size")
            if not isinstance(observed_hash, str) or type(observed_size) is not int:
                raise RemoteObjectPreconditionError("Azure predecessor inspection lacks exact content identity")
            predecessor = ArtifactDescriptor(
                artifact_type="file",
                path_or_uri=request.inspection.reference,
                content_hash=observed_hash,
                size_bytes=observed_size,
            )
        return prepare_remote_object(
            effect_id=request.effect_id,
            provider="azure_blob",
            inspection=request.inspection,
            body_chunks=(content,),
            format_name=self._format,
            max_bytes=self._max_blob_bytes,
            accepted_ordinals=accepted,
            diverted_ordinals=diverted,
            predecessor_descriptor=predecessor,
            checksum_algorithm="md5",
        )

    def _blob_path_from_target(self, target: str) -> str:
        prefix = f"azure://{self._container}/"
        blob_path = target.removeprefix(prefix)
        if not blob_path or prefix + blob_path != target:
            raise RemoteObjectPreconditionError("Azure effect target does not match configured container")
        return blob_path

    @staticmethod
    def _is_conditional_failure(error: BaseException) -> bool:
        return type(error).__name__ in {"ResourceExistsError", "ResourceModifiedError"} or getattr(error, "status_code", None) in {409, 412}

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        evidence, stage = validate_remote_plan(plan, provider="azure_blob", require_stage=True)
        expected_target = f"azure://{self._container}/{self._effect_blob_path(ctx)}"
        if evidence.target != expected_target:
            raise RemoteObjectPreconditionError("Azure effect target diverges from the configured run target")
        blob_client = self._get_container_client().get_blob_client(self._blob_path_from_target(evidence.target))
        metadata = {
            "elspeth_content_sha256": evidence.staged_hash,
            "elspeth_effect_id": plan.effect_id,
            "elspeth_plan_hash": plan.plan_hash,
            "elspeth_protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
        }
        from azure.storage.blob import ContentSettings

        content_settings = ContentSettings(content_md5=bytearray(base64.b64decode(evidence.checksum_b64, validate=True)))
        try:
            with stage.open("rb") as body:
                if evidence.precondition == "if_none_match":
                    blob_client.upload_blob(
                        body,
                        overwrite=False,
                        if_none_match="*",
                        metadata=metadata,
                        content_settings=content_settings,
                        validate_content=True,
                    )
                else:
                    from azure.core import MatchConditions

                    blob_client.upload_blob(
                        body,
                        overwrite=True,
                        etag=evidence.predecessor_etag,
                        match_condition=MatchConditions.IfNotModified,
                        metadata=metadata,
                        content_settings=content_settings,
                        validate_content=True,
                    )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            if self._is_conditional_failure(error):
                raise RemoteObjectPreconditionError("Azure conditional blob upload was rejected") from None
            raise RemoteObjectEffectError("Azure blob upload outcome is unknown; reconciliation is required") from None
        return remote_commit_result(plan, evidence)

    def reconcile_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        evidence, _stage = validate_remote_plan(plan, provider="azure_blob", require_stage=False)
        expected_target = f"azure://{self._container}/{self._effect_blob_path(ctx)}"
        if evidence.target != expected_target:
            raise RemoteObjectPreconditionError("Azure effect target diverges from the configured run target")
        observation = self._observe_effect_target(self._blob_path_from_target(evidence.target))
        return reconcile_remote_observation(plan, evidence, observation)

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

        if self._headers_mode == HeaderMode.CUSTOM:
            for field in data_fields:
                if field not in display_map:
                    raise ValueError(
                        f"CUSTOM header mode has no mapping for field '{field}'. "
                        f"All fields must be explicitly mapped — silent fallback to normalized "
                        f"names risks data corruption in external system handover. "
                        f"Mapped fields: {sorted(display_map.keys())}"
                    )

        # ORIGINAL mode falls back to the normalized field name for
        # transform-added fields that have no source header.
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
        del rows, ctx
        raise RuntimeError("AzureBlobSink publication requires the recoverable sink effect coordinator") from None

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
