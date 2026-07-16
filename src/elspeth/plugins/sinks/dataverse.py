"""Dataverse sink plugin for ELSPETH.

Writes rows to Microsoft Dataverse entities via OData v4 REST API.
Day-one: upsert-only (PATCH with alternate key). Create and update
modes are deferred per the design spec.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import CallStatus, CallType, Determinism, PluginSchema
from elspeth.contracts.contexts import LifecycleContext, SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import AuditIntegrityError, FrameworkBugError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseAuthConfig,
    DataverseClient,
    DataverseClientError,
    validate_additional_domain,
)
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url
from elspeth.plugins.sinks._diversion_attribution import build_diversion_attribution

# HTTP status codes that may be single-row-attributable when Dataverse also
# provides a structured row-data classification. These statuses alone are not
# enough: Dataverse uses the same 4xx family for sink-wide configuration faults
# such as invalid entity sets, bad API paths, and missing alternate-key setup.
# Status-only failures must raise so misconfigured sinks cannot silently discard
# every row via on_write_failure=discard.
#
# CLOSED LIST — extend only with a status code that is (a) non-retryable and
# (b) attributable to the individual row's request, not the batch or the
# connection. 422 is included even though the client classifies it as
# "Unexpected HTTP status" rather than an explicit client error, because an
# Unprocessable-Entity response is about this row's payload.
_DIVERTABLE_STATUS_CODES: frozenset[int] = frozenset({400, 404, 409, 412, 422})
_ROW_ATTRIBUTABLE_ERROR_CATEGORIES: frozenset[str] = frozenset({"row_data_error"})


def _is_row_attributable_write_error(error: DataverseClientError) -> bool:
    return (
        not error.retryable and error.status_code in _DIVERTABLE_STATUS_CODES and error.error_category in _ROW_ATTRIBUTABLE_ERROR_CATEGORIES
    )


# An @odata.bind value sits in the UNQUOTED entity-key position of the bind
# reference (``/entity(value)``), unlike the alternate-key URL where the value
# is a quoted string literal. Row values are Tier-3 data; a value containing
# OData/URI structural characters (e.g. ``abc)/contacts(emailaddress1='x')``)
# could change the bind reference's navigation shape — an injection. Restrict
# bind values to a record-reference key token (alphanumerics + hyphen, which
# covers Dataverse record GUIDs) and reject anything else clearly at the sink
# boundary. Validate-and-reject is safe whether or not Dataverse percent-decodes
# the bind reference; percent-encoding in an unquoted position would not be
# (elspeth-e7d31117df).
_SAFE_LOOKUP_BIND_VALUE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9-]+$")


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
        return reject_operator_required_placeholder_value(v.strip(), field_name="target_entity")

    @field_validator("target_field")
    @classmethod
    def validate_target_field_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target_field cannot be empty")
        return reject_operator_required_placeholder_value(v.strip(), field_name="target_field")


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
        return validate_credential_safe_https_url(v, field_name="environment_url")

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
        return reject_operator_required_placeholder_value(v.strip(), field_name="entity")

    @field_validator("alternate_key")
    @classmethod
    def validate_alternate_key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("alternate_key cannot be empty")
        return reject_operator_required_placeholder_value(v.strip(), field_name="alternate_key")

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
            if not dv_column or not dv_column.strip():
                raise ValueError(f"field_mapping target for '{pipeline_field}' cannot be empty")
            reject_operator_required_placeholder_value(dv_column, field_name=f"field_mapping target for '{pipeline_field}'")
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
    source_file_hash: str | None = "sha256:5a66501c68b516d3"
    determinism = Determinism.EXTERNAL_CALL
    config_model = DataverseSinkConfig
    idempotent = True  # PATCH upsert is idempotent — safe for retries and crash recovery (engine does not yet read this flag)
    supports_resume = False  # Dataverse writes are not locally staged
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"upsert"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})
    supports_member_effects = True

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        del cls
        if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
            return None
        mode = config.get("mode", "upsert")
        return ResolvedSinkEffectMode(mode) if isinstance(mode, str) else None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Upserts rows into Microsoft Dataverse via OData using an alternate key.",
                composer_hints=(
                    "Dataverse sink supports upsert mode only; alternate_key must be one of the field_mapping target columns.",
                    "field_mapping maps pipeline field names to Dataverse column names and must not produce duplicate targets.",
                    "lookups map pipeline fields to navigation-property bindings; avoid collisions with field_mapping targets.",
                    "environment_url must be HTTPS and within the allowed Dataverse domain patterns.",
                    "Alternate-key values must be non-empty strings at write time; an empty or non-string key crashes the run (it cannot form a valid OData URL).",
                    "Per-row HTTP failures route via on_write_failure: a non-retryable 4xx about the row (400/404/409/412/422) diverts that row and the batch continues; auth/authz (401/403), rate limit (429), retryable, and 5xx errors raise so the engine retries or aborts.",
                    "Set on_write_failure to a quarantine sink so single-row 4xx don't abort the batch.",
                ),
            )
        return None

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

    def on_start(self, ctx: LifecycleContext) -> None:
        """Construct credential and DataverseClient."""
        super().on_start(ctx)

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
                    # OData bind syntax: "field@odata.bind": "/entity(guid)".
                    # The value is interpolated into the UNQUOTED key position,
                    # so reject any value that isn't a plain record-reference
                    # token before it can change the bind URI's shape. Mirrors
                    # the offensive alternate_key guard in write(): a structurally
                    # unsafe bind value fails clearly at the boundary rather than
                    # producing an ambiguous/injectable outbound payload.
                    bind_value = str(value)
                    if not _SAFE_LOOKUP_BIND_VALUE.match(bind_value):
                        raise ValueError(
                            f"lookup field {pipeline_field!r} value {value!r} is "
                            f"not a valid record reference for @odata.bind to "
                            f"entity {lookup.target_entity!r}: only alphanumerics "
                            f"and hyphens are allowed (e.g. a record GUID). Values "
                            f"with OData/URI structural characters are rejected to "
                            f"prevent bind URI injection."
                        )
                    bind_key = f"{lookup.target_field}@odata.bind"
                    payload[bind_key] = f"/{lookup.target_entity}({bind_value})"
                # None value = don't include bind (leaves lookup unset)
            else:
                payload[dataverse_column] = value

        return payload

    @property
    def _effect_target(self) -> str:
        environment = urllib.parse.urlsplit(self._environment_url)
        assert environment.hostname is not None
        encoded_entity = urllib.parse.quote(self._entity, safe="")
        encoded_api_version = urllib.parse.quote(self._api_version, safe="")
        return f"dataverse://{environment.hostname}/{encoded_entity}?api_version={encoded_api_version}"

    def _member_effect_material(
        self,
        effect_id: str,
        effect_input: SinkEffectPipelineMembersInput,
    ) -> tuple[ArtifactDescriptor, str, str, tuple[dict[str, object], ...]]:
        payloads: list[dict[str, object]] = []
        member_bindings: list[dict[str, object]] = []
        seen_keys: set[str] = set()
        if self._alternate_key_pipeline_field is None:  # pragma: no cover - config validator establishes it
            raise FrameworkBugError("Dataverse alternate-key pipeline field was not resolved")
        for member in effect_input.members:
            row = deep_thaw(member.row)
            if not isinstance(row, dict):  # pragma: no cover - member contract guarantees a mapping
                raise FrameworkBugError("Dataverse effect member row is not an object")
            key_value = row[self._alternate_key_pipeline_field]
            if not isinstance(key_value, str) or not key_value.strip():
                raise ValueError(
                    f"alternate_key field '{self._alternate_key_pipeline_field}' has empty or non-string value "
                    f"{key_value!r} — cannot construct PATCH URL for entity '{self._entity}'"
                )
            if key_value in seen_keys:
                raise ValueError(f"Dataverse effect members require unique alternate-key values; duplicate {key_value!r}")
            seen_keys.add(key_value)
            payload = self._map_row(row)
            payloads.append(payload)
            member_bindings.append(
                {
                    "member_effect_id": member.member_effect_id,
                    "ordinal": member.ordinal,
                    "payload_hash": stable_hash(payload),
                    "target_hash": stable_hash(self._build_upsert_url(key_value)),
                }
            )
        canonical_payload = canonical_json(payloads).encode("utf-8")
        payload_hash = hashlib.sha256(canonical_payload).hexdigest()
        descriptor = ArtifactDescriptor(
            artifact_type="webhook",
            path_or_uri=self._effect_target,
            content_hash=payload_hash,
            size_bytes=len(canonical_payload),
            metadata=MappingProxyType({"row_count": len(payloads), "entity": self._entity, "mode": self._mode}),
        )
        bindings_hash = stable_hash(member_bindings)
        plan_hash = stable_hash(
            {
                "bindings_hash": bindings_hash,
                "descriptor_hash": stable_hash(
                    {
                        "artifact_type": descriptor.artifact_type,
                        "content_hash": descriptor.content_hash,
                        "metadata": deep_thaw(descriptor.metadata),
                        "path_or_uri": descriptor.path_or_uri,
                        "size_bytes": descriptor.size_bytes,
                    }
                ),
                "effect_id": effect_id,
                "schema": "dataverse-member-effect-plan-v1",
            }
        )
        return descriptor, payload_hash, plan_hash, tuple(payloads)

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del request, ctx
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference="no-inspection-required:v1",
            evidence={},
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if type(request.effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("Dataverse effects require pipeline member input")
        descriptor, payload_hash, plan_hash, payloads = self._member_effect_material(request.effect_id, request.effect_input)
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=self._effect_target,
            plan_hash=plan_hash,
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence={
                "member_count": len(payloads),
                "member_plans_hash": stable_hash(
                    [
                        {
                            "member_effect_id": member.member_effect_id,
                            "ordinal": member.ordinal,
                            "payload_hash": stable_hash(payload),
                        }
                        for member, payload in zip(request.effect_input.members, payloads, strict=True)
                    ]
                ),
                "schema": "dataverse-member-effect-plan-v1",
            },
        )

    def _validate_member_effect(
        self,
        plan: SinkEffectPlan,
        member: SinkEffectMember,
        effect_input: SinkEffectPipelineMembersInput,
    ) -> tuple[str, dict[str, object], ArtifactDescriptor]:
        descriptor, payload_hash, plan_hash, payloads = self._member_effect_material(plan.effect_id, effect_input)
        if (
            plan.protocol_version != SINK_EFFECT_PROTOCOL_VERSION
            or plan.input_kind is not SinkEffectInputKind.PIPELINE_MEMBERS
            or plan.descriptor_mode is not SinkEffectDescriptorMode.PRECOMPUTED
            or plan.target != self._effect_target
            or plan.payload_hash != payload_hash
            or plan.plan_hash != plan_hash
            or plan.expected_descriptor != descriptor
        ):
            raise ValueError("Dataverse member effect plan is divergent from the bound input and target")
        if member.ordinal >= len(effect_input.members) or effect_input.members[member.ordinal] != member:
            raise ValueError("Dataverse member does not match its exact stored ordinal")
        if member.member_effect_id is None:
            raise ValueError("Dataverse member effect requires a durable member_effect_id")
        row = deep_thaw(member.row)
        assert isinstance(row, dict)
        key_field = self._alternate_key_pipeline_field
        if key_field is None:  # pragma: no cover - config validator establishes it
            raise FrameworkBugError("Dataverse alternate-key pipeline field was not resolved")
        key_value = row[key_field]
        assert isinstance(key_value, str)
        return self._build_upsert_url(key_value), payloads[member.ordinal], descriptor

    @staticmethod
    def _member_group_evidence(plan: SinkEffectPlan, classification: str) -> dict[str, object]:
        return {
            "classification": classification,
            "effect_id": plan.effect_id,
            "plan_hash": plan.plan_hash,
            "schema": "dataverse-member-effect-result-v1",
        }

    def commit_member_effect(
        self,
        plan: SinkEffectPlan,
        member: SinkEffectMember,
        effect_input: SinkEffectPipelineMembersInput,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        del ctx
        if self._client is None:
            raise FrameworkBugError("Dataverse client is unavailable — on_start() was not called")
        url, payload, descriptor = self._validate_member_effect(plan, member, effect_input)
        try:
            self._client.upsert(url, payload)
        except DataverseClientError as exc:
            # Mirror write(): only explicitly row-attributable, non-retryable
            # responses divert; batch-integrity/unknown failures still raise so
            # the engine retries or crashes instead of silently dropping rows.
            if not _is_row_attributable_write_error(exc):
                raise
            reason = f"Dataverse PATCH failed with non-retryable HTTP {exc.status_code}: {exc}"
            row = deep_thaw(member.row)
            assert isinstance(row, dict)
            # Live diversion log BEFORE the durable result: fails closed with
            # FrameworkBugError when no on_write_failure policy is configured.
            self._divert_row(row, row_index=member.ordinal, reason=reason)
            attribution = build_diversion_attribution(ordinal=member.ordinal, reason=reason)
            return SinkEffectCommitResult(
                descriptor=descriptor,
                evidence={
                    **self._member_group_evidence(plan, "diverted"),
                    "diversion_attribution": [attribution.as_mapping()],
                },
                accepted_ordinals=tuple(item.ordinal for item in effect_input.members if item.ordinal != member.ordinal),
                diverted_ordinals=(member.ordinal,),
            )
        return SinkEffectCommitResult(
            descriptor=descriptor,
            evidence=self._member_group_evidence(plan, "committed"),
            accepted_ordinals=tuple(item.ordinal for item in effect_input.members),
            diverted_ordinals=(),
        )

    def reconcile_member_effect(
        self,
        plan: SinkEffectPlan,
        member: SinkEffectMember,
        effect_input: SinkEffectPipelineMembersInput,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        del ctx
        if self._client is None:
            raise FrameworkBugError("Dataverse client is unavailable — on_start() was not called")
        url, expected, descriptor = self._validate_member_effect(plan, member, effect_input)
        try:
            response = self._client.get_page(url)
        except DataverseClientError as exc:
            if exc.status_code == 404:
                return SinkEffectReconcileResult.not_applied(evidence=self._member_group_evidence(plan, "missing"))
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "unverifiable"))
        if len(response.rows) == 0:
            return SinkEffectReconcileResult.not_applied(evidence=self._member_group_evidence(plan, "missing"))
        if len(response.rows) != 1:
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "ambiguous"))
        actual = response.rows[0]
        exact = all(key in actual and actual[key] == value for key, value in expected.items())
        if not exact:
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "divergent"))
        return SinkEffectReconcileResult.applied(
            descriptor,
            evidence=self._member_group_evidence(plan, "exact"),
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del plan, ctx
        raise FrameworkBugError("Dataverse publication requires durable member-effect coordination")

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del plan, ctx
        raise FrameworkBugError("Dataverse reconciliation requires durable member-effect coordination")

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
        # Each entry carries the original row and its index into the input batch
        # so a per-row write failure can be diverted with correct row_data and
        # row_index (the executor correlates the diversion back to the row token).
        prepared: list[tuple[str, dict[str, Any], dict[str, Any], int]] = []
        for i, row in enumerate(rows):
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
            prepared.append((url, payload, row, i))

        # Payloads actually written to Dataverse (excludes per-row diversions).
        # The content hash and row_count must describe only what we wrote, so an
        # auditor can independently verify the hash against the Dataverse-side
        # data — a diverted row was never written and must not appear here.
        written_payloads: list[dict[str, Any]] = []

        # All pre-processing succeeded — safe to make HTTP calls
        for url, payload, original_row, row_index in prepared:
            # Execute upsert with audit recording + telemetry
            start_time = time.perf_counter()
            try:
                response = self._client.upsert(url, payload)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Audit first (primacy), then telemetry
                request_data: dict[str, Any] = {
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
                written_payloads.append(payload)
            except DataverseClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Audit first, then telemetry
                request_data = {
                    "method": "PATCH",
                    "url": url,
                    "headers": e.request_headers,  # Fingerprinted by client; mirrors the success path
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
                        "error_category": e.error_category,
                    },
                    latency_ms=latency_ms,
                    provider="dataverse",
                )
                # 401 with retryable=True: reconstruct credential before engine retry
                if e.status_code == 401 and e.retryable:
                    assert self._client is not None
                    self._client.reconstruct_credential(self._auth_config)

                # Classify the failure by structured Dataverse semantics:
                #
                #   DIVERT — explicitly row-attributable: this row's payload or
                #     alternate key is bad and a retry will not help. The row is
                #     routed to on_write_failure and the batch continues.
                #
                #   RAISE — batch-integrity or unknown: authn/authz (401/403),
                #     rate limit (429), retryable errors, 5xx server errors, and
                #     generic 4xx protocol/configuration errors. Diverting these
                #     can silently drop rows from a misconfigured sink.
                #
                # Fail safe: a missing/None status_code cannot be attributed to a
                # single row, so it falls through to RAISE.
                if _is_row_attributable_write_error(e):
                    self._divert_row(
                        original_row,
                        row_index=row_index,
                        reason=(f"Dataverse PATCH failed with non-retryable HTTP {e.status_code}: {e}"),
                    )
                    continue

                # Re-raise original error — engine sink executor records
                # exception_type for audit diagnostics, and DataverseClientError
                # preserves the retryable/status_code metadata in the chain.
                raise

        # Compute the content hash over only the payloads we actually wrote to
        # Dataverse, so the hash verifies against the Dataverse-side data.
        canonical_payload = canonical_json(written_payloads).encode("utf-8")
        content_hash = hashlib.sha256(canonical_payload).hexdigest()
        total_size = len(canonical_payload)

        return SinkWriteResult(
            artifact=ArtifactDescriptor(
                artifact_type="webhook",
                path_or_uri=f"dataverse://{self._entity}@{self._environment_url}",
                content_hash=content_hash,
                size_bytes=total_size,
                metadata=MappingProxyType(
                    {
                        "row_count": len(written_payloads),
                        "entity": self._entity,
                        "mode": self._mode,
                    }
                ),
            ),
            diversions=self._get_diversions(),
        )

    def flush(self) -> None:
        """No-op — Dataverse writes are immediate, no local staging buffer."""
        pass

    def close(self) -> None:
        """Release DataverseClient resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
