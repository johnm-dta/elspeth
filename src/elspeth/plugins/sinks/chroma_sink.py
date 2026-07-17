"""ChromaDB vector store sink plugin.

Writes pipeline rows into a ChromaDB collection. Each row becomes a
document with ChromaDB's default embedding function.
"""

from __future__ import annotations

import hashlib
import math
import urllib.parse
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import chromadb
import chromadb.api
import chromadb.errors
import structlog
from pydantic import BaseModel, Field, model_validator

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import Determinism
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import CallType
from elspeth.contracts.errors import (
    FrameworkBugError,
)
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
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sinks._diversion_attribution import build_diversion_attribution

slog = structlog.get_logger(__name__)


class _ChromaPayloadRejection(Exception):
    """Chroma API rejected the payload with ValueError.

    Wraps ValueError from Chroma write calls (upsert/add) so the error handler
    can distinguish "Chroma rejected our data" (Tier 3) from "bug in our code"
    (framework error). Without this, a broad ValueError catch would suppress
    framework bugs in the surrounding code.
    """


if TYPE_CHECKING:
    from elspeth.contracts.contexts import LifecycleContext, SinkContext
    from elspeth.contracts.data import PluginSchema


class FieldMappingConfig(BaseModel):
    """Maps row field names to ChromaDB document concepts.

    Field values are names of row fields, not literal content.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    document_field: str = Field(description="Row field containing text to embed")
    id_field: str = Field(description="Row field containing document ID")
    metadata_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Row fields to include as ChromaDB metadata",
    )


class ChromaSinkConfig(DataPluginConfig):
    """Configuration for ChromaDB vector store sink.

    Connection fields are flat (matching YAML config convention for sinks)
    and validated by constructing a ChromaConnectionConfig in the model
    validator. This is the same delegation pattern used by
    ChromaSearchProviderConfig.
    """

    _plugin_component_type: ClassVar[str | None] = "sink"

    collection: str = Field(description="ChromaDB collection name")
    mode: Literal["persistent", "client"] = Field(description="Connection mode")
    persist_directory: str | None = Field(default=None, description="Local persistence directory for persistent Chroma mode.")
    host: str | None = Field(default=None, description="Chroma server host for client mode.")
    port: int = Field(default=8000, ge=1, le=65535, description="Chroma server port for client mode.")
    ssl: bool = Field(default=True, description="Whether to use TLS when connecting to a Chroma server.")
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Vector distance function used when creating the Chroma collection.",
    )

    field_mapping: FieldMappingConfig = Field(description="Maps row fields to ChromaDB document/id/metadata")
    on_duplicate: Literal["overwrite", "skip", "error"] = Field(
        default="overwrite",
        description="Behaviour when a document ID already exists",
    )

    @model_validator(mode="after")
    def validate_connection(self) -> ChromaSinkConfig:
        """Delegate connection validation to ChromaConnectionConfig."""
        ChromaConnectionConfig(
            collection=self.collection,
            mode=self.mode,
            persist_directory=self.persist_directory,
            host=self.host,
            port=self.port,
            ssl=self.ssl,
            distance_function=self.distance_function,
        )
        return self

    @model_validator(mode="after")
    def validate_field_mapping_against_schema(self) -> ChromaSinkConfig:
        """Cross-reference field_mapping field names against schema_config.

        For fixed/flexible schemas, validates that referenced fields exist and
        have compatible types. For observed schemas, fields are unknown at config
        time so validation defers to runtime.
        """
        if self.schema_config.is_observed:
            return self

        fields = self.schema_config.fields
        if fields is None:
            return self

        field_types = {f.name: f.field_type for f in fields}
        fm = self.field_mapping

        # document_field and id_field must exist and be str-compatible
        for attr_name, label in [("document_field", "document_field"), ("id_field", "id_field")]:
            field_name = getattr(fm, attr_name)
            if field_name not in field_types:
                raise ValueError(
                    f"field_mapping.{label} references '{field_name}' which is not in the schema. Declared fields: {sorted(field_types)}"
                )
            ft = field_types[field_name]
            if ft not in ("str", "any"):
                raise ValueError(f"field_mapping.{label} references '{field_name}' which has type '{ft}' — ChromaDB requires str")

        # metadata_fields must exist and have ChromaDB-compatible types
        chroma_metadata_types = {"str", "int", "float", "bool", "any"}
        for mf in fm.metadata_fields:
            if mf not in field_types:
                raise ValueError(
                    f"field_mapping.metadata_fields references '{mf}' which is not in the schema. Declared fields: {sorted(field_types)}"
                )
            ft = field_types[mf]
            if ft not in chroma_metadata_types:
                raise ValueError(
                    f"field_mapping.metadata_fields references '{mf}' which has type '{ft}' — ChromaDB metadata requires str/int/float/bool"
                )

        return self


class ChromaSink(BaseSink):
    """Write pipeline rows into a ChromaDB collection.

    Each row maps to a ChromaDB document via the configured field_mapping.
    Content is hashed (canonical JSON of the actual payload sent) before
    write for audit integrity.

    Trust boundary: Row data arriving at this sink is Tier 2 (types validated
    upstream). ChromaDB itself is an external system — SDK errors
    (chromadb.errors.ChromaError) are caught as infrastructure failures;
    other exceptions crash through as plugin bugs per CLAUDE.md plugin
    ownership rules.
    """

    name = "chroma_sink"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:998381796385fde1"
    config_model = ChromaSinkConfig
    supports_resume = False
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.VECTOR
    supported_effect_modes = frozenset({"overwrite"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})
    supports_member_effects = True
    effect_mode_remediation = "set on_duplicate=overwrite or choose a sink with a target-side effect marker"

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
        mode = config.get("on_duplicate", "overwrite")
        return ResolvedSinkEffectMode(mode) if isinstance(mode, str) else None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Writes rows to a ChromaDB collection as embedded documents with metadata.",
                composer_hints=(
                    "Set field_mapping.id_field and document_field to row fields holding string values; Chroma requires str ids and documents. A row missing either required field is diverted per row, but a present-yet-non-string id or document is an upstream type error that aborts the whole batch.",
                    "metadata_fields may contain only Chroma-compatible scalar values: str, int, float, bool, or None.",
                    "Use mode=persistent with persist_directory for local storage, or mode=client with host/port/ssl for a server.",
                    "Choose on_duplicate as overwrite, skip, or error before running; the default overwrites document IDs.",
                    "Chroma sink does not support resume, so plan reruns around idempotent document IDs.",
                ),
            )
        return None

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._config = ChromaSinkConfig.from_dict(config, plugin_name=self.name)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._config.schema_config,
            "ChromaSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class
        self.declared_required_fields = self._config.schema_config.get_effective_required_fields()

        self._client: chromadb.api.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._telemetry_emit: Callable[[Any], None] | None = None
        self._total_written = 0
        self._total_bytes = 0

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        self._telemetry_emit = ctx.telemetry_emit

        if self._config.mode == "persistent":
            # Validated by ChromaConnectionConfig: persist_directory is not None
            if self._config.persist_directory is None:
                raise FrameworkBugError(
                    "ChromaSinkConfig.persist_directory is None in 'persistent' mode "
                    "— ChromaConnectionConfig validation should have rejected this"
                )
            self._client = chromadb.PersistentClient(
                path=self._config.persist_directory,
            )
        else:
            # Validated by ChromaConnectionConfig: host is not None
            if self._config.host is None:
                raise FrameworkBugError(
                    "ChromaSinkConfig.host is None in 'client' mode — ChromaConnectionConfig validation should have rejected this"
                )
            self._client = chromadb.HttpClient(
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.ssl,
            )
            self._client.heartbeat()

        self._collection = self._client.get_or_create_collection(
            name=self._config.collection,
            metadata={"hnsw:space": self._config.distance_function},
        )

    @staticmethod
    def _compute_payload_hash(
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any] | None] | None,
    ) -> tuple[str, int]:
        """Compute canonical hash and size for the actual payload being sent."""
        payload = canonical_json({"ids": ids, "documents": documents, "metadatas": metadatas})
        payload_bytes = payload.encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest(), len(payload_bytes)

    def _extract_effect_member(self, member: SinkEffectMember) -> tuple[str, str, dict[str, object] | None]:
        row = deep_thaw(member.row)
        if not isinstance(row, dict):  # pragma: no cover - member contract guarantees a mapping
            raise FrameworkBugError("Chroma effect member row is not an object")
        fm = self._config.field_mapping
        try:
            raw_id = row[fm.id_field]
            raw_document = row[fm.document_field]
        except KeyError as exc:
            raise ValueError(f"Chroma effect member is missing required field {exc.args[0]!r}") from exc
        if not isinstance(raw_id, str) or not isinstance(raw_document, str):
            raise ValueError("Chroma effect member id and document fields must be strings")
        metadata: dict[str, object] = {}
        for field_name in fm.metadata_fields:
            if field_name not in row:
                continue
            value = row[field_name]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Chroma effect member metadata field {field_name!r} is not a supported scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Chroma effect member metadata field {field_name!r} must be finite")
            metadata[field_name] = value
        return raw_id, raw_document, metadata or None

    @property
    def _effect_target(self) -> str:
        if self._config.mode == "persistent":
            target_binding: dict[str, object] = {
                "mode": "persistent",
                "persist_directory": self._config.persist_directory,
            }
        else:
            target_binding = {
                "host": self._config.host,
                "mode": "client",
                "port": self._config.port,
                "ssl": self._config.ssl,
            }
        binding_hash = stable_hash(target_binding)
        collection = urllib.parse.quote(self._config.collection, safe="")
        return f"chromadb://target/{collection}?binding={binding_hash}"

    def _member_effect_material(
        self,
        effect_id: str,
        effect_input: SinkEffectPipelineMembersInput,
    ) -> tuple[ArtifactDescriptor, str, str, dict[int, tuple[str, str, dict[str, object] | None]], tuple[tuple[int, str], ...]]:
        """Deterministically partition members into extractable payloads and diversions.

        An invalid member (missing/non-string ID or document, unsupported or
        non-finite metadata) diverts individually with its exact reason instead
        of aborting the plan and blocking valid siblings (elspeth-32bf1a9b63).
        The descriptor and payload hash cover only the members that will be
        published. Recomputation over the same members is exact, so plan
        validation on commit/reconcile still binds.
        """
        extracted_by_ordinal: dict[int, tuple[str, str, dict[str, object] | None]] = {}
        diversions: list[tuple[int, str]] = []
        for member in effect_input.members:
            try:
                extracted_by_ordinal[member.ordinal] = self._extract_effect_member(member)
            except ValueError as exc:
                diversions.append((member.ordinal, str(exc)))
        ids = [item[0] for item in extracted_by_ordinal.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("Chroma effect members require unique stable document IDs")
        documents = [item[1] for item in extracted_by_ordinal.values()]
        aligned_metadata = [item[2] for item in extracted_by_ordinal.values()]
        hash_metadata = aligned_metadata if any(item is not None for item in aligned_metadata) else None
        payload_hash, payload_size = self._compute_payload_hash(ids, documents, hash_metadata)
        descriptor = ArtifactDescriptor.for_database(
            url=SanitizedDatabaseUrl.from_raw_url(self._effect_target),
            table=self._config.collection,
            content_hash=payload_hash,
            payload_size=payload_size,
            row_count=len(extracted_by_ordinal),
        )
        bindings = [
            {
                "document_id_hash": stable_hash(extracted_by_ordinal[member.ordinal][0]),
                "member_effect_id": member.member_effect_id,
                "ordinal": member.ordinal,
                "payload_hash": stable_hash(
                    {
                        "document": extracted_by_ordinal[member.ordinal][1],
                        "document_id": extracted_by_ordinal[member.ordinal][0],
                        "metadata": extracted_by_ordinal[member.ordinal][2],
                    }
                ),
            }
            for member in effect_input.members
            if member.ordinal in extracted_by_ordinal
        ]
        plan_hash = stable_hash(
            {
                "bindings": bindings,
                "collection": self._config.collection,
                "descriptor_hash": stable_hash(
                    {
                        "artifact_type": descriptor.artifact_type,
                        "content_hash": descriptor.content_hash,
                        "metadata": deep_thaw(descriptor.metadata),
                        "path_or_uri": descriptor.path_or_uri,
                        "size_bytes": descriptor.size_bytes,
                    }
                ),
                "diverted": [build_diversion_attribution(ordinal=ordinal, reason=reason).as_mapping() for ordinal, reason in diversions],
                "effect_id": effect_id,
                "schema": "chroma-member-effect-plan-v1",
            }
        )
        return descriptor, payload_hash, plan_hash, extracted_by_ordinal, tuple(diversions)

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
        if self._config.on_duplicate != "overwrite":
            raise ValueError(
                "Recoverable Chroma publication requires on_duplicate=overwrite; "
                "use overwrite or choose a sink with a target-side effect marker"
            )
        if type(request.effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("Chroma effects require pipeline member input")
        descriptor, payload_hash, plan_hash, extracted_by_ordinal, diversions = self._member_effect_material(
            request.effect_id, request.effect_input
        )
        member_by_ordinal = {member.ordinal: member for member in request.effect_input.members}
        diversion_attribution = []
        for ordinal, reason in diversions:
            row = deep_thaw(member_by_ordinal[ordinal].row)
            assert isinstance(row, dict)
            # Live diversion log BEFORE the plan binds: fails closed with
            # FrameworkBugError when no on_write_failure policy is configured.
            self._divert_row(row, row_index=ordinal, reason=reason)
            diversion_attribution.append(build_diversion_attribution(ordinal=ordinal, reason=reason).as_mapping())
        diversion_attribution.sort(key=lambda item: item["ordinal"])  # type: ignore[arg-type,return-value]
        safe_evidence: dict[str, object] = {
            "accepted_ordinals": sorted(extracted_by_ordinal),
            "diversion_attribution": diversion_attribution,
            "diverted_ordinals": sorted(ordinal for ordinal, _reason in diversions),
            "member_count": len(request.effect_input.members),
            "member_plans_hash": stable_hash(
                [
                    {
                        "document_id_hash": stable_hash(item[0]),
                        "member_effect_id": member_by_ordinal[ordinal].member_effect_id,
                        "ordinal": ordinal,
                        "payload_hash": stable_hash({"document": item[1], "document_id": item[0], "metadata": item[2]}),
                    }
                    for ordinal, item in sorted(extracted_by_ordinal.items())
                ]
            ),
            "schema": "chroma-member-effect-plan-v1",
        }
        # A group with no publishable member performs no external I/O at all:
        # finalize it as a no-publication effect so the diverted members do
        # not wedge the batch waiting for an external attempt that can never
        # exist.
        descriptor_mode = SinkEffectDescriptorMode.PRECOMPUTED
        if not extracted_by_ordinal:
            descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
            safe_evidence["publication_kind"] = "virtual"
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=descriptor_mode,
            inspection_mode=request.inspection.mode,
            target=self._effect_target,
            plan_hash=plan_hash,
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence=safe_evidence,
        )

    def _validate_member_effect(
        self,
        plan: SinkEffectPlan,
        member: SinkEffectMember,
        effect_input: SinkEffectPipelineMembersInput,
    ) -> tuple[str, str, dict[str, object] | None, ArtifactDescriptor]:
        descriptor, payload_hash, plan_hash, extracted_by_ordinal, _diversions = self._member_effect_material(plan.effect_id, effect_input)
        if (
            plan.protocol_version != SINK_EFFECT_PROTOCOL_VERSION
            or plan.input_kind is not SinkEffectInputKind.PIPELINE_MEMBERS
            or plan.descriptor_mode is not SinkEffectDescriptorMode.PRECOMPUTED
            or plan.target != self._effect_target
            or plan.payload_hash != payload_hash
            or plan.plan_hash != plan_hash
            or plan.expected_descriptor != descriptor
        ):
            raise ValueError("Chroma member effect plan is divergent from the bound input and target")
        if member.ordinal >= len(effect_input.members) or effect_input.members[member.ordinal] != member:
            raise ValueError("Chroma member does not match its exact stored ordinal")
        if member.member_effect_id is None:
            raise ValueError("Chroma member effect requires a durable member_effect_id")
        if member.ordinal not in extracted_by_ordinal:
            raise ValueError("Chroma member was diverted during preparation and must not reach external I/O")
        document_id, document, metadata = extracted_by_ordinal[member.ordinal]
        return document_id, document, metadata, descriptor

    @staticmethod
    def _member_group_evidence(plan: SinkEffectPlan, classification: str) -> dict[str, object]:
        return {
            "classification": classification,
            "effect_id": plan.effect_id,
            "plan_hash": plan.plan_hash,
            "schema": "chroma-member-effect-result-v1",
        }

    def commit_member_effect(
        self,
        plan: SinkEffectPlan,
        member: SinkEffectMember,
        effect_input: SinkEffectPipelineMembersInput,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        del ctx
        if self._collection is None:
            raise FrameworkBugError("Chroma collection is unavailable — on_start() was not called")
        document_id, document, metadata, descriptor = self._validate_member_effect(plan, member, effect_input)
        try:
            self._collection.upsert(
                ids=[document_id],
                documents=[document],
                metadatas=None if metadata is None else [metadata],  # type: ignore[list-item]
            )
        except ValueError as exc:
            raise _ChromaPayloadRejection(str(exc)) from exc
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
        if self._collection is None:
            raise FrameworkBugError("Chroma collection is unavailable — on_start() was not called")
        document_id, document, metadata, descriptor = self._validate_member_effect(plan, member, effect_input)
        try:
            result = self._collection.get(ids=[document_id], include=["documents", "metadatas"])
        except (chromadb.errors.ChromaError, ValueError):
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "unverifiable"))
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        if ids == []:
            return SinkEffectReconcileResult.not_applied(evidence=self._member_group_evidence(plan, "missing"))
        if (
            ids != [document_id]
            or not isinstance(documents, list)
            or len(documents) != 1
            or not isinstance(metadatas, list)
            or len(metadatas) != 1
        ):
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "ambiguous"))
        if documents[0] != document or metadatas[0] != metadata:
            return SinkEffectReconcileResult.unknown(evidence=self._member_group_evidence(plan, "divergent"))
        return SinkEffectReconcileResult.applied(
            descriptor,
            evidence=self._member_group_evidence(plan, "exact"),
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del plan, ctx
        raise FrameworkBugError("Chroma publication requires durable member-effect coordination")

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del plan, ctx
        raise FrameworkBugError("Chroma reconciliation requires durable member-effect coordination")

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        del rows, ctx
        raise RuntimeError("ChromaSink publication requires the recoverable sink effect coordinator") from None

    def flush(self) -> None:
        """No-op: ChromaDB publication is synchronous in the member-effect commit path."""

    def on_complete(self, ctx: LifecycleContext) -> None:
        super().on_complete(ctx)
        if self._telemetry_emit is None:
            return
        # Telemetry is best-effort operational visibility emitted AFTER the writes
        # and their audit record have completed; an emit failure must not fail
        # completion (elspeth-ee69831e4c). Tier-1/audit-integrity errors still
        # propagate (audit corruption outranks) — same guard shape as
        # plugins/sinks/dataverse.py's post-audit telemetry emission.
        try:
            self._telemetry_emit(
                {
                    "event": "chroma_sink_complete",
                    "collection": self._config.collection,
                    "total_written": self._total_written,
                    "total_bytes": self._total_bytes,
                }
            )
        except contract_errors.TIER_1_ERRORS:
            raise
        except Exception as tel_err:
            slog.warning(
                "telemetry_emit_failed",
                sink="chroma",
                collection=self._config.collection,
                error=str(tel_err),
                error_type=type(tel_err).__name__,
                exc_info=True,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.clear_system_cache()
        self._client = None
        self._collection = None
