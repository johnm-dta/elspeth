"""Dataverse sink plugin for ELSPETH.

Writes rows to Microsoft Dataverse entities via OData v4 REST API.
Day-one: upsert-only (PATCH with alternate key). Create and update
modes are deferred per the design spec.
"""

from __future__ import annotations

import hashlib
import time
import urllib.parse
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import CallStatus, CallType, Determinism, PluginSchema
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.contexts import LifecycleContext, SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseAuthConfig,
    DataverseClient,
    DataverseClientError,
    validate_additional_domain,
)
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

logger = structlog.get_logger(__name__)


class LookupConfig(BaseModel):
    """Configuration for a lookup field binding."""

    model_config = {"extra": "forbid", "frozen": True}

    target_entity: str  # Dataverse entity to bind to (e.g., "accounts")
    target_field: str  # Navigation property name (e.g., "parentcustomerid")

    @field_validator("target_entity")
    @classmethod
    def validate_target_entity_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target_entity cannot be empty")
        return v.strip()

    @field_validator("target_field")
    @classmethod
    def validate_target_field_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target_field cannot be empty")
        return v.strip()


class DataverseSinkConfig(DataPluginConfig):
    """Configuration for Dataverse sink plugin.

    Extends DataPluginConfig which requires schema configuration.
    """

    _plugin_component_type: ClassVar[str | None] = "sink"

    environment_url: str = Field(
        ...,
        description="Dataverse environment URL (e.g., https://myorg.crm.dynamics.com)",
    )
    auth: DataverseAuthConfig = Field(
        ...,
        description="Authentication configuration",
    )
    api_version: str = Field(
        default="v9.2",
        description="Dataverse Web API version",
    )

    entity: str = Field(
        ...,
        description="Target entity logical name",
    )
    mode: Literal["upsert"] = Field(
        default="upsert",
        description="Write mode (day-one: upsert only)",
    )

    # Field mapping (mandatory — no passthrough)
    field_mapping: dict[str, str] = Field(
        ...,
        description="Pipeline field → Dataverse column mapping",
    )

    # Key field (required for upsert)
    alternate_key: str = Field(
        ...,
        description="Business key field for upsert (PATCH with alternate key)",
    )

    # Lookup field declarations
    lookups: dict[str, LookupConfig] | None = Field(
        default=None,
        description="Lookup field bindings for navigation properties",
    )

    # Additional SSRF domain patterns
    additional_domains: list[str] | None = Field(
        default=None,
        description="Additional Dataverse domain patterns for SSRF allowlist",
    )

    @field_validator("environment_url")
    @classmethod
    def validate_environment_url_https(cls, v: str) -> str:
        """HTTPS required — same validator as DataverseSourceConfig."""
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme != "https":
            raise ValueError(
                f"environment_url must use HTTPS scheme, got {parsed.scheme!r}. "
                f"Bearer tokens are sent in Authorization headers — HTTP would expose them in transit."
            )
        return v

    @field_validator("additional_domains")
    @classmethod
    def validate_additional_domains(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for pattern in v:
                validate_additional_domain(pattern)
        return v

    @field_validator("entity")
    @classmethod
    def validate_entity_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("entity cannot be empty")
        return v.strip()

    @field_validator("alternate_key")
    @classmethod
    def validate_alternate_key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("alternate_key cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_no_outbound_key_collisions(self) -> Self:
        """Reject configs where multiple source fields map to the same Dataverse key.

        Checks three collision types:
        1. Duplicate field_mapping values (two pipeline fields → same Dataverse column)
        2. Duplicate lookup target_field values (two lookups → same navigation property)
        3. Lookup bind key colliding with a field_mapping target
        """
        # 1. field_mapping value uniqueness
        targets = list(self.field_mapping.values())
        seen: dict[str, str] = {}
        for pipeline_field, dv_column in self.field_mapping.items():
            if dv_column in seen:
                raise ValueError(
                    f"field_mapping collision: pipeline fields '{seen[dv_column]}' and "
                    f"'{pipeline_field}' both map to Dataverse column '{dv_column}'"
                )
            seen[dv_column] = pipeline_field

        if self.lookups:
            # 2. lookup target_field uniqueness
            lookup_targets: dict[str, str] = {}
            for pipeline_field, lookup in self.lookups.items():
                bind_key = f"{lookup.target_field}@odata.bind"
                if lookup.target_field in lookup_targets:
                    raise ValueError(
                        f"lookup collision: fields '{lookup_targets[lookup.target_field]}' and "
                        f"'{pipeline_field}' both target navigation property '{lookup.target_field}'"
                    )
                lookup_targets[lookup.target_field] = pipeline_field

                # 3. bind key vs field_mapping target collision
                if bind_key in targets:
                    raise ValueError(
                        f"lookup/field_mapping collision: lookup for '{pipeline_field}' produces "
                        f"bind key '{bind_key}' which collides with a field_mapping target"
                    )

        return self

    @model_validator(mode="after")
    def validate_alternate_key_in_field_mapping(self) -> Self:
        """Reject configs where alternate_key is not in field_mapping values.

        The alternate_key must name a Dataverse column that appears as a value
        in field_mapping (pipeline_field -> dataverse_column). Moved from
        DataverseSink.__init__ so from_dict() catches it (pre-validation /
        engine-validation agreement).
        """
        if self.alternate_key not in self.field_mapping.values():
            raise ValueError(
                f"alternate_key '{self.alternate_key}' not found in field_mapping values. "
                f"The alternate_key must be a Dataverse column that appears as a value in field_mapping. "
                f"Available field_mapping values: {sorted(self.field_mapping.values())}"
            )
        return self


# Rebuild model to resolve forward references
DataverseSinkConfig.model_rebuild()


class DataverseSink(BaseSink):
    """Write rows to Microsoft Dataverse via OData v4 REST API.

    Day-one supports upsert mode only (PATCH with alternate key).
    PATCH is naturally idempotent — safe for retryable pipelines
    and crash recovery re-runs.
    """

    name = "dataverse"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:cf259d9d2a666fb9"
    determinism = Determinism.EXTERNAL_CALL
    config_model = DataverseSinkConfig
    idempotent = True  # PATCH upsert is idempotent — safe for retries and crash recovery (engine does not yet read this flag)
    supports_resume = False  # Dataverse writes are not locally staged

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DataverseSinkConfig.from_dict(config, plugin_name=self.name)

        # Store config
        self._environment_url = cfg.environment_url
        self._auth_config = cfg.auth
        self._api_version = cfg.api_version
        self._entity = cfg.entity
        self._mode = cfg.mode
        self._field_mapping = cfg.field_mapping
        self._alternate_key = cfg.alternate_key
        self._lookups = cfg.lookups
        self._additional_domains = tuple(cfg.additional_domains) if cfg.additional_domains else ()

        # Schema setup — sinks do NOT coerce (Tier 2 data)
        self._schema_config = cfg.schema_config
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "DataverseSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        # Resolve the pipeline field name for the alternate key.
        # field_mapping is pipeline_field → dataverse_column; we need the reverse.
        # Presence of alternate_key in field_mapping values is guaranteed by
        # DataverseSinkConfig.validate_alternate_key_in_field_mapping model_validator.
        self._alternate_key_pipeline_field: str | None = None
        for pipeline_field, dataverse_col in self._field_mapping.items():
            if dataverse_col == self._alternate_key:
                self._alternate_key_pipeline_field = pipeline_field
                break

        # Lazy-constructed client (needs lifecycle context)
        self._client: DataverseClient | None = None
        self._telemetry_emit: Any = None
        self._run_id: str | None = None

    def on_start(self, ctx: LifecycleContext) -> None:
        """Construct credential and DataverseClient."""
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit

        credential = self._auth_config.create_credential()

        # Obtain rate limiter (with null guard)
        limiter = ctx.rate_limit_registry.get_limiter("dataverse_sink") if ctx.rate_limit_registry is not None else None

        self._client = DataverseClient(
            environment_url=self._environment_url,
            credential=credential,
            api_version=self._api_version,
            limiter=limiter,
            additional_domains=self._additional_domains,
        )

    def _build_upsert_url(self, key_value: str) -> str:
        """Build PATCH URL for upsert with alternate key.

        URL-encodes entity name, alternate key name, and key value to prevent
        injection via special characters.
        key_value is guaranteed str by the isinstance check in write().
        """
        encoded_entity = urllib.parse.quote(self._entity, safe="")
        encoded_key_name = urllib.parse.quote(self._alternate_key, safe="")
        encoded_value = urllib.parse.quote(key_value, safe="")
        return f"{self._environment_url.rstrip('/')}/api/data/{self._api_version}/{encoded_entity}({encoded_key_name}='{encoded_value}')"

    def _emit_telemetry(
        self,
        *,
        ctx: SinkContext,
        status: CallStatus,
        latency_ms: float,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
    ) -> None:
        """Emit ExternalCallCompleted telemetry after successful audit recording.

        Telemetry fires AFTER audit (primacy rule). Failures are logged,
        not propagated — telemetry must never corrupt the audit trail.
        """
        if self._telemetry_emit is None:
            return
        try:
            assert self._run_id is not None, "run_id is None during telemetry emission — on_start() must set _run_id before write()"
            req_payload = RawCallPayload(request_data)
            resp_payload = RawCallPayload(response_data) if response_data else None
            self._telemetry_emit(
                ExternalCallCompleted(
                    timestamp=datetime.now(UTC),
                    run_id=self._run_id,
                    call_type=CallType.HTTP,
                    provider="dataverse",
                    status=status,
                    latency_ms=latency_ms,
                    operation_id=ctx.operation_id,
                    request_hash=stable_hash(request_data),
                    response_hash=stable_hash(response_data) if response_data else None,
                    request_payload=req_payload,
                    response_payload=resp_payload,
                )
            )
        except contract_errors.TIER_1_ERRORS:
            raise
        except Exception as tel_err:
            logger.warning(
                "telemetry_emit_failed",
                error=str(tel_err),
                error_type=type(tel_err).__name__,
                call_type="http",
                exc_info=True,
            )

    def _map_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Apply field mapping and lookup bindings.

        Args:
            row: Pipeline row with normalized field names

        Returns:
            Dataverse-ready payload with OData column names and bind syntax
        """
        payload: dict[str, Any] = {}

        for pipeline_field, dataverse_column in self._field_mapping.items():
            # Tier 2: schema guarantees field exists. KeyError = upstream bug.
            value = row[pipeline_field]

            # Check if this field has a lookup binding
            if self._lookups and pipeline_field in self._lookups:
                lookup = self._lookups[pipeline_field]
                if value is not None:
                    # OData bind syntax: "field@odata.bind": "/entity(guid)"
                    bind_key = f"{lookup.target_field}@odata.bind"
                    payload[bind_key] = f"/{lookup.target_entity}({value})"
                # None value = don't include bind (leaves lookup unset)
            else:
                payload[dataverse_column] = value

        return payload

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        """Write batch of rows to Dataverse via individual PATCH requests.

        Processes rows serially. On success, returns a single ArtifactDescriptor.
        On failure, raises on the first failing row (engine retries entire batch,
        PATCH idempotency makes re-sends safe).

        Args:
            rows: List of row dicts to upsert
            ctx: Sink context for audit recording

        Returns:
            ArtifactDescriptor with batch metadata

        Raises:
            RuntimeError: If any row fails to upsert
        """
        if not rows:
            return SinkWriteResult(
                artifact=ArtifactDescriptor(
                    artifact_type="webhook",
                    path_or_uri=f"dataverse://{self._entity}@{self._environment_url}",
                    content_hash=hashlib.sha256(b"").hexdigest(),
                    size_bytes=0,
                    metadata=MappingProxyType({"row_count": 0, "entity": self._entity}),
                )
            )

        # Client and key field must be set by on_start/__init__
        assert self._client is not None, "on_start() must be called before write()"
        assert self._alternate_key_pipeline_field is not None

        # Pre-process ALL rows before making any HTTP calls.  If _map_row or
        # key validation fails on row N, we must not have already written rows
        # 1..N-1 — that would leave audit states as FAILED while Dataverse data
        # was actually modified (partial success = audit inconsistency).
        prepared: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            # Tier 2: field_mapping guarantees the field exists. Direct access
            # — KeyError if absent is an upstream bug.
            key_value = row[self._alternate_key_pipeline_field]

            # Offensive guard: empty/blank key produces a valid-looking OData
            # URL (entity(key='')) that Dataverse would accept or reject
            # ambiguously. Crash here with a clear message instead.
            if not isinstance(key_value, str) or not key_value.strip():
                raise ValueError(
                    f"alternate_key field '{self._alternate_key_pipeline_field}' has "
                    f"empty or non-string value {key_value!r} — cannot construct "
                    f"PATCH URL for entity '{self._entity}'"
                )

            url = self._build_upsert_url(key_value)
            payload = self._map_row(row)
            prepared.append((url, payload))

        # Compute content hash from the mapped payloads (what we actually send
        # to Dataverse), not the full pipeline rows. This allows an auditor to
        # independently verify the hash against the Dataverse-side data.
        mapped_payloads = [payload for _, payload in prepared]
        canonical_payload = canonical_json(mapped_payloads).encode("utf-8")
        content_hash = hashlib.sha256(canonical_payload).hexdigest()
        total_size = len(canonical_payload)

        # All pre-processing succeeded — safe to make HTTP calls
        for url, payload in prepared:
            # Execute upsert with audit recording + telemetry
            start_time = time.perf_counter()
            try:
                response = self._client.upsert(url, payload)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Audit first (primacy), then telemetry
                request_data = {
                    "method": "PATCH",
                    "url": url,
                    "headers": response.request_headers,
                    "json": payload,
                }
                response_data = {"status_code": response.status_code}
                try:
                    ctx.record_call(
                        call_type=CallType.HTTP,
                        status=CallStatus.SUCCESS,
                        request_data=request_data,
                        response_data=response_data,
                        latency_ms=latency_ms,
                        provider="dataverse",
                    )
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Failed to record successful Dataverse upsert to audit trail "
                        f"(url={url!r}). "
                        f"Upsert completed but audit record is missing."
                    ) from exc
                self._emit_telemetry(
                    ctx=ctx,
                    status=CallStatus.SUCCESS,
                    latency_ms=latency_ms,
                    request_data=request_data,
                    response_data=response_data,
                )
            except DataverseClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Audit first, then telemetry
                request_data = {
                    "method": "PATCH",
                    "url": url,
                    "json": payload,
                }
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data=request_data,
                    error={
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "status_code": e.status_code,
                        "retryable": e.retryable,
                    },
                    latency_ms=latency_ms,
                    provider="dataverse",
                )
                self._emit_telemetry(
                    ctx=ctx,
                    status=CallStatus.ERROR,
                    latency_ms=latency_ms,
                    request_data=request_data,
                )
                # 401 with retryable=True: reconstruct credential before engine retry
                if e.status_code == 401 and e.retryable:
                    assert self._client is not None
                    self._client.reconstruct_credential(self._auth_config)
                # Re-raise original error — engine sink executor records
                # exception_type for audit diagnostics, and DataverseClientError
                # preserves the retryable/status_code metadata in the chain.
                raise

        return SinkWriteResult(
            artifact=ArtifactDescriptor(
                artifact_type="webhook",
                path_or_uri=f"dataverse://{self._entity}@{self._environment_url}",
                content_hash=content_hash,
                size_bytes=total_size,
                metadata=MappingProxyType(
                    {
                        "row_count": len(rows),
                        "entity": self._entity,
                        "mode": self._mode,
                    }
                ),
            )
        )

    def flush(self) -> None:
        """No-op — Dataverse writes are immediate, no local staging buffer."""
        pass

    def close(self) -> None:
        """Release DataverseClient resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
