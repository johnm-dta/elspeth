"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine, insert, select, text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import DataError, IntegrityError, SQLAlchemyError

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeEngine

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import CallType
from elspeth.contracts.errors import SinkEffectCapabilityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_assistance import PluginAssistance
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
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sinks._diversion_attribution import DiversionAttribution, build_diversion_attribution

# Map schema field types to SQLAlchemy column types.
# Text (not String) is used for string columns because String() without a length
# argument causes truncation or errors on MySQL/MSSQL — Text maps to TEXT on all
# backends and accepts arbitrary-length values without portability issues.
SCHEMA_TYPE_TO_SQLALCHEMY: Mapping[str, type[TypeEngine[Any]]] = MappingProxyType(
    {
        "str": Text,
        "int": Integer,
        "float": Float,
        "bool": Boolean,
        "any": Text,  # Fallback to Text for 'any' type
    }
)

_DATABASE_EFFECT_LEDGER_SCHEMA_VERSION = 1
_DATABASE_EFFECT_LEDGER_PERMISSIONS = frozenset({"insert", "select"})
_DATABASE_EFFECT_SUPPORTED_DIALECTS = frozenset({"postgresql", "sqlite"})
_DATABASE_EFFECT_TABLE_NAME = re.compile(r"_elspeth_[A-Za-z0-9_]+\Z")
_DATABASE_EFFECT_LEDGER_COLUMNS = frozenset(
    {
        "accepted_ordinals_json",
        "accepted_payload_hash",
        "completed",
        "descriptor_json",
        "diversion_hashes_json",
        "diverted_ordinals_json",
        "effect_id",
        "evidence_json",
        "payload_hash",
        "plan_hash",
        "protocol_version",
        "schema_version",
    }
)


class DatabaseEffectLedgerError(RuntimeError):
    """The configured target-side Database effect ledger is absent or divergent."""


class DatabaseEffectMarkerDivergence(DatabaseEffectLedgerError):
    """A durable target marker exists but does not exactly bind the requested plan."""


class DatabaseEffectLedgerConfig(BaseModel):
    """Operator declaration for an already-provisioned target-side ledger."""

    model_config = {"extra": "forbid", "frozen": True}

    table: str = Field(description="Namespaced operator-provisioned target-side effect ledger table.")
    schema_version: Literal[1] = Field(
        default=1,
        description="Expected target-side effect ledger schema version.",
    )
    permissions: frozenset[Literal["insert", "select"]] = Field(
        description="Operator-declared permissions granted to the runtime identity.",
    )

    @field_validator("table")
    @classmethod
    def _validate_ledger_table(cls, value: str) -> str:
        if _DATABASE_EFFECT_TABLE_NAME.fullmatch(value) is None:
            raise ValueError("effect ledger table must be a namespaced identifier beginning with '_elspeth_'")
        return reject_operator_required_placeholder_value(value, field_name="effect_ledger.table")

    @model_validator(mode="after")
    def _require_exact_permissions(self) -> "DatabaseEffectLedgerConfig":
        if self.permissions != _DATABASE_EFFECT_LEDGER_PERMISSIONS:
            raise ValueError("effect ledger permissions must declare exactly 'select' and 'insert'")
        return self


def database_effect_ledger_table(metadata: MetaData, table_name: str) -> Table:
    """Build the version-1 operator provisioning table; runtime never calls create_all()."""
    if _DATABASE_EFFECT_TABLE_NAME.fullmatch(table_name) is None:
        raise ValueError("effect ledger table must be a namespaced identifier beginning with '_elspeth_'")
    existing = metadata.tables.get(table_name)
    if existing is not None:
        return existing
    return Table(
        table_name,
        metadata,
        Column("effect_id", String(64), primary_key=True),
        Column("schema_version", Integer, nullable=False),
        Column("protocol_version", String(32), nullable=False),
        Column("plan_hash", String(64), nullable=False),
        Column("payload_hash", String(64), nullable=False),
        Column("completed", Boolean, nullable=False),
        Column("descriptor_json", Text, nullable=True),
        Column("accepted_ordinals_json", Text, nullable=True),
        Column("diverted_ordinals_json", Text, nullable=True),
        Column("accepted_payload_hash", String(64), nullable=True),
        Column("diversion_hashes_json", Text, nullable=True),
        Column("evidence_json", Text, nullable=True),
    )


class DatabaseSinkConfig(DataPluginConfig):
    """Configuration for database sink plugin.

    Inherits from DataPluginConfig, which requires schema configuration.
    """

    _plugin_component_type: ClassVar[str | None] = "sink"

    url: str = Field(description="Database connection URL for SQLAlchemy.")
    table: str = Field(description="Database table name to write rows into.")
    if_exists: Literal["append", "replace"] = Field(
        default="append",
        description="Whether to append to an existing table or replace it before writing.",
    )
    effect_ledger: DatabaseEffectLedgerConfig | None = Field(
        default=None,
        description="Operator declaration for the provisioned target-side exactly-once effect ledger.",
    )

    @field_validator("table")
    @classmethod
    def _reject_empty_table(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("table name must not be empty")
        return reject_operator_required_placeholder_value(v, field_name="table")


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Creates the table on first write. When schema is explicit, columns are
    derived from schema field definitions with proper type mapping. When schema
    is dynamic, columns are inferred from the first row's keys.

    Uses SQLAlchemy Core for direct SQL control.

    Returns ArtifactDescriptor with a SHA-256 hash of the canonical JSON
    payload of the rows that were COMMITTED (after per-row diversion). The
    database may further transform data (timestamps, auto-increment IDs);
    the hash proves what was durably written. Rows diverted on a per-row
    constraint failure are NOT in this hash — they are audited via the
    diversion log and routed to the configured failsink.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        schema: Schema configuration (required, via DataPluginConfig)
        if_exists: "append" or "replace" (default: "append")

    Input validation is always enabled. Incoming rows are validated against
    the schema before INSERT — wrong types indicate an upstream plugin bug
    and will crash the pipeline (Tier 2 contract).

    The schema can be:
        - Observed: {"mode": "observed"} - accept any fields (columns inferred from first row)
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]} - columns from schema
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - columns from schema, extras allowed
    """

    name = "database"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:4cb64957f47335f4"
    config_model = DatabaseSinkConfig
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.SQL
    supported_effect_modes = frozenset({"append"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: Database can append to existing tables
    supports_resume: bool = True

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode | None:
        del purpose
        cfg = DatabaseSinkConfig.from_dict(dict(config), plugin_name=cls.name)
        cls._validate_effect_config(cfg)
        return ResolvedSinkEffectMode(cfg.if_exists)

    @classmethod
    def _validate_effect_config(cls, cfg: DatabaseSinkConfig) -> None:
        if cfg.effect_ledger is None:
            raise SinkEffectCapabilityError("Database sink recoverable publication requires an operator-declared target-side effect ledger")
        if cfg.if_exists != "append":
            raise SinkEffectCapabilityError(
                "Database sink effect mode 'replace' is not supported; provision a new target table and use if_exists='append'"
            )
        dialect = make_url(cfg.url).get_backend_name()
        if dialect not in _DATABASE_EFFECT_SUPPORTED_DIALECTS:
            raise SinkEffectCapabilityError(
                f"Database sink effect dialect {dialect!r} is unsupported; use SQLite or PostgreSQL with transactional markers"
            )

    def _validate_sink_effect_capability_configuration(
        self,
        *,
        mode: str,
        required_input_kind: SinkEffectInputKind,
    ) -> None:
        del required_input_kind
        if mode != self._if_exists:
            raise SinkEffectCapabilityError("Database sink effect mode does not match the configured if_exists mode")
        cfg = DatabaseSinkConfig.from_dict(dict(self.config), plugin_name=self.name)
        self._validate_effect_config(cfg)

    def configure_for_resume(self) -> None:
        """Configure database sink for resume mode.

        Switches from replace mode to append mode so resume operations
        add to existing table instead of dropping and recreating.
        """
        self._if_exists = "append"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config, plugin_name=self.name)

        # Honor ELSPETH_ALLOW_RAW_SECRETS for dev environments. Process environment
        # is a Tier-3 boundary; absence of the flag is meaning-preserving (feature
        # off → do not allow raw secrets), so we read it as an explicit membership
        # test rather than a defaulting .get() (mirrors data_flow_repository.py).
        allow_raw = "ELSPETH_ALLOW_RAW_SECRETS" in os.environ and os.environ["ELSPETH_ALLOW_RAW_SECRETS"].lower() == "true"
        fail_if_no_key = not allow_raw

        self._url = cfg.url  # Raw URL for database connection
        self._sanitized_url = SanitizedDatabaseUrl.from_raw_url(cfg.url, fail_if_no_key=fail_if_no_key)  # For audit trail
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists
        self._effect_ledger = cfg.effect_ledger

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        self._schema_config = cfg.schema_config

        # Database supports all schema modes via infer-and-lock:
        # - mode='fixed': columns from config, extras rejected at insert time
        # - mode='flexible': declared columns + extras from first row, then locked
        # - mode='observed': columns from first row, then locked
        #
        # Table schema is created on first write; subsequent rows must match.

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "DatabaseRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        # Track which fields have 'any' type so we can serialize dict/list values
        # to JSON strings before INSERT (SQL TEXT columns can't store Python dicts).
        self._any_typed_fields: frozenset[str] = self._compute_any_typed_fields()

        self._engine: Engine | None = None
        self._metadata: MetaData | None = None

    def _target_reference(self) -> str:
        ledger = self._require_effect_ledger_config()
        identity_hash = hashlib.sha256(
            canonical_json(
                {
                    "ledger_schema_version": ledger.schema_version,
                    "ledger_table": ledger.table,
                    "sanitized_url": self._sanitized_url.sanitized_url,
                    "target_table": self._table_name,
                }
            ).encode("utf-8")
        ).hexdigest()
        return f"database-target:sha256:{identity_hash}"

    def _require_effect_ledger_config(self) -> DatabaseEffectLedgerConfig:
        if self._effect_ledger is None:
            raise DatabaseEffectLedgerError("Database sink requires an operator-provisioned target-side effect ledger")
        return self._effect_ledger

    def _inspect_target_contract(self) -> tuple[str, tuple[str, ...], str]:
        """Verify target and ledger through read-only SQL/introspection only."""
        from sqlalchemy import inspect as sqlalchemy_inspect

        ledger_config = self._require_effect_ledger_config()
        self._ensure_engine_and_metadata_initialized()
        engine = self._engine
        if engine is None:  # pragma: no cover - paired initializer invariant
            raise RuntimeError("Database sink effect inspection called before engine initialization")
        dialect = engine.dialect.name
        if dialect not in _DATABASE_EFFECT_SUPPORTED_DIALECTS:
            raise DatabaseEffectLedgerError(f"Database effect dialect {dialect!r} is unsupported")

        inspector = sqlalchemy_inspect(engine)
        if not inspector.has_table(ledger_config.table):
            raise DatabaseEffectLedgerError(
                f"Database target-side effect ledger {ledger_config.table!r} is missing; provision schema version {ledger_config.schema_version}"
            )
        ledger_columns = {column["name"]: column for column in inspector.get_columns(ledger_config.table)}
        if set(ledger_columns) != _DATABASE_EFFECT_LEDGER_COLUMNS:
            raise DatabaseEffectLedgerError(
                f"Database target-side effect ledger {ledger_config.table!r} does not match schema version {ledger_config.schema_version}"
            )
        primary_key = inspector.get_pk_constraint(ledger_config.table).get("constrained_columns")
        if primary_key != ["effect_id"]:
            raise DatabaseEffectLedgerError("Database target-side effect ledger must use effect_id as its sole primary key")
        required_not_null = _DATABASE_EFFECT_LEDGER_COLUMNS - {
            "accepted_ordinals_json",
            "accepted_payload_hash",
            "descriptor_json",
            "diversion_hashes_json",
            "diverted_ordinals_json",
            "evidence_json",
        }
        if any(bool(ledger_columns[name].get("nullable")) for name in required_not_null):
            raise DatabaseEffectLedgerError("Database target-side effect ledger required columns must be NOT NULL")

        if not inspector.has_table(self._table_name):
            raise DatabaseEffectLedgerError(
                f"Database target table {self._table_name!r} is missing; provision it before recoverable append publication"
            )
        target_columns = tuple(column["name"] for column in inspector.get_columns(self._table_name))
        if not self._schema_config.is_observed and self._schema_config.fields is not None:
            expected = {field.name for field in self._schema_config.fields}
            observed = set(target_columns)
            if self._schema_config.mode == "fixed" and observed != expected:
                raise DatabaseEffectLedgerError("Database target table columns do not match the fixed sink schema")
            if self._schema_config.mode == "flexible" and not expected <= observed:
                raise DatabaseEffectLedgerError("Database target table is missing flexible sink schema columns")

        with engine.connect() as conn:
            ledger = Table(ledger_config.table, MetaData(), autoload_with=conn)
            conn.execute(select(ledger.c.effect_id).where(text("1 = 0"))).all()
            if dialect == "postgresql":
                permissions = {
                    permission: bool(
                        conn.scalar(
                            text("SELECT has_table_privilege(current_user, :table_name, :permission)"),
                            {"table_name": ledger_config.table, "permission": permission.upper()},
                        )
                    )
                    for permission in _DATABASE_EFFECT_LEDGER_PERMISSIONS
                }
                if not all(permissions.values()):
                    raise DatabaseEffectLedgerError(
                        "Database target-side effect ledger runtime identity lacks declared SELECT/INSERT permissions"
                    )
                if not bool(
                    conn.scalar(
                        text("SELECT has_table_privilege(current_user, :table_name, 'INSERT')"),
                        {"table_name": self._table_name},
                    )
                ):
                    raise DatabaseEffectLedgerError("Database target table runtime identity lacks INSERT permission")
        contract_hash = hashlib.sha256(
            canonical_json(
                {
                    "columns": sorted(_DATABASE_EFFECT_LEDGER_COLUMNS),
                    "primary_key": ["effect_id"],
                    "schema_version": ledger_config.schema_version,
                    "table": ledger_config.table,
                }
            ).encode("utf-8")
        ).hexdigest()
        return dialect, target_columns, contract_hash

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del ctx
        ledger_config = self._require_effect_ledger_config()
        dialect, target_columns, contract_hash = self._inspect_target_contract()
        target = self._target_reference()
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.INSPECTED,
            reference=target,
            evidence={
                "dialect": dialect,
                "effect_id": request.effect_id,
                "ledger_contract_hash": contract_hash,
                "ledger_schema_version": ledger_config.schema_version,
                "ledger_table": ledger_config.table,
                "permissions": sorted(ledger_config.permissions),
                "target_columns": list(target_columns),
                "target_table": self._table_name,
            },
        )

    @staticmethod
    def _database_effect_plan_hash(
        *,
        effect_id: str,
        payload_hash: str,
        target: str,
        safe_evidence: Mapping[str, object],
    ) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "effect_id": effect_id,
                    "payload_hash": payload_hash,
                    "safe_evidence": deep_thaw(safe_evidence),
                    "schema": "database-effect-plan-envelope-v1",
                    "target": target,
                }
            ).encode("utf-8")
        ).hexdigest()

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        effect_input = request.effect_input
        if type(effect_input) is not SinkEffectPipelineMembersInput:
            raise TypeError("DatabaseSink effects require pipeline member input")
        inspection = request.inspection
        if inspection.mode is not SinkEffectInspectionMode.INSPECTED:
            raise DatabaseEffectLedgerError("Database sink effects require a completed read-only target inspection")
        target = self._target_reference()
        if inspection.reference != target or inspection.evidence.get("effect_id") != request.effect_id:
            raise DatabaseEffectLedgerError("Database sink effect inspection does not bind this exact target and effect")
        ledger = self._require_effect_ledger_config()
        if (
            inspection.evidence.get("ledger_table") != ledger.table
            or inspection.evidence.get("ledger_schema_version") != ledger.schema_version
            or inspection.evidence.get("target_table") != self._table_name
            or inspection.evidence.get("dialect") not in _DATABASE_EFFECT_SUPPORTED_DIALECTS
        ):
            raise DatabaseEffectLedgerError("Database sink effect inspection is divergent from configured target authority")
        target_columns_value = inspection.evidence.get("target_columns")
        if not isinstance(target_columns_value, tuple) or any(not isinstance(value, str) for value in target_columns_value):
            raise DatabaseEffectLedgerError("Database sink effect inspection lacks exact target columns")
        target_columns = set(target_columns_value)

        members: list[dict[str, object]] = []
        source_rows = [deep_thaw(member.row) for member in effect_input.members]
        serialized_rows = self._serialize_any_typed_fields(source_rows)
        for member, serialized_row in zip(effect_input.members, serialized_rows, strict=True):
            extra = sorted(set(serialized_row) - target_columns)
            if extra:
                raise DatabaseEffectLedgerError(
                    f"Database sink effect member {member.ordinal} has fields absent from target table: {extra}"
                )
            members.append(
                {
                    "ordinal": member.ordinal,
                    "payload_hash": member.payload_hash,
                    "row": serialized_row,
                }
            )
        payload_hash = hashlib.sha256(canonical_json(members).encode("utf-8")).hexdigest()
        safe_evidence: dict[str, object] = {
            "dialect": inspection.evidence["dialect"],
            "diversion_policy": "constraint-savepoint-v1",
            "ledger_contract_hash": inspection.evidence["ledger_contract_hash"],
            "ledger_schema_version": ledger.schema_version,
            "ledger_table": ledger.table,
            "members": members,
            "schema": "database-effect-plan-v1",
            "serializer": "rfc8785-sql-payload-v1",
            "target_columns": sorted(target_columns),
            "target_table": self._table_name,
        }
        plan_hash = self._database_effect_plan_hash(
            effect_id=request.effect_id,
            payload_hash=payload_hash,
            target=target,
            safe_evidence=safe_evidence,
        )
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED,
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target=target,
            plan_hash=plan_hash,
            payload_hash=payload_hash,
            expected_descriptor=None,
            safe_evidence=safe_evidence,
        )

    @staticmethod
    def _require_canonical_json(value: object, *, field_name: str) -> object:
        if not isinstance(value, str):
            raise DatabaseEffectMarkerDivergence(f"Database effect marker {field_name} must be text")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DatabaseEffectMarkerDivergence(f"Database effect marker {field_name} is not JSON") from exc
        if canonical_json(decoded) != value:
            raise DatabaseEffectMarkerDivergence(f"Database effect marker {field_name} is not canonical JSON")
        return decoded

    def _parse_database_effect_plan(self, plan: SinkEffectPlan) -> list[tuple[int, dict[str, Any]]]:
        if (
            plan.protocol_version != SINK_EFFECT_PROTOCOL_VERSION
            or plan.input_kind is not SinkEffectInputKind.PIPELINE_MEMBERS
            or plan.descriptor_mode is not SinkEffectDescriptorMode.RESULT_DERIVED
            or plan.inspection_mode is not SinkEffectInspectionMode.INSPECTED
            or plan.expected_descriptor is not None
            or plan.target != self._target_reference()
        ):
            raise DatabaseEffectLedgerError("Database effect plan does not match the supported result-derived protocol")
        evidence = deep_thaw(plan.safe_evidence)
        if type(evidence) is not dict or set(evidence) != {
            "dialect",
            "diversion_policy",
            "ledger_contract_hash",
            "ledger_schema_version",
            "ledger_table",
            "members",
            "schema",
            "serializer",
            "target_columns",
            "target_table",
        }:
            raise DatabaseEffectLedgerError("Database effect plan evidence has a divergent field set")
        ledger = self._require_effect_ledger_config()
        dialect = make_url(self._url).get_backend_name()
        if (
            evidence["schema"] != "database-effect-plan-v1"
            or evidence["serializer"] != "rfc8785-sql-payload-v1"
            or evidence["diversion_policy"] != "constraint-savepoint-v1"
            or evidence["target_table"] != self._table_name
            or evidence["ledger_table"] != ledger.table
            or evidence["ledger_schema_version"] != ledger.schema_version
            or evidence["dialect"] != dialect
        ):
            raise DatabaseEffectLedgerError("Database effect plan evidence diverges from configured target authority")
        raw_members = evidence["members"]
        if type(raw_members) is not list or not raw_members:
            raise DatabaseEffectLedgerError("Database effect plan requires a non-empty ordered member list")
        members: list[tuple[int, dict[str, Any]]] = []
        payload_members: list[dict[str, object]] = []
        for expected_ordinal, raw_member in enumerate(raw_members):
            if type(raw_member) is not dict or set(raw_member) != {"ordinal", "payload_hash", "row"}:
                raise DatabaseEffectLedgerError("Database effect plan member has a divergent field set")
            ordinal = raw_member["ordinal"]
            payload_hash = raw_member["payload_hash"]
            row = raw_member["row"]
            if type(ordinal) is not int or ordinal != expected_ordinal:
                raise DatabaseEffectLedgerError("Database effect plan member ordinals must be dense and ordered")
            if not isinstance(payload_hash, str) or re.fullmatch(r"[0-9a-f]{64}", payload_hash) is None:
                raise DatabaseEffectLedgerError("Database effect plan member payload hash is invalid")
            if type(row) is not dict or any(not isinstance(key, str) for key in row):
                raise DatabaseEffectLedgerError("Database effect plan member row must be a canonical object")
            detached_row = dict(row)
            members.append((ordinal, detached_row))
            payload_members.append({"ordinal": ordinal, "payload_hash": payload_hash, "row": detached_row})
        expected_payload_hash = hashlib.sha256(canonical_json(payload_members).encode("utf-8")).hexdigest()
        if plan.payload_hash != expected_payload_hash:
            raise DatabaseEffectLedgerError("Database effect plan payload hash does not bind the ordered members")
        expected_plan_hash = self._database_effect_plan_hash(
            effect_id=plan.effect_id,
            payload_hash=plan.payload_hash,
            target=plan.target,
            safe_evidence=evidence,
        )
        if plan.plan_hash != expected_plan_hash:
            raise DatabaseEffectLedgerError("Database effect plan hash does not bind its exact evidence")
        return members

    @staticmethod
    def _descriptor_json(descriptor: ArtifactDescriptor) -> str:
        return canonical_json(
            {
                "artifact_type": descriptor.artifact_type,
                "content_hash": descriptor.content_hash,
                "metadata": None if descriptor.metadata is None else deep_thaw(descriptor.metadata),
                "path_or_uri": descriptor.path_or_uri,
                "size_bytes": descriptor.size_bytes,
            }
        )

    @staticmethod
    def _descriptor_from_json(value: object) -> ArtifactDescriptor:
        decoded = DatabaseSink._require_canonical_json(value, field_name="descriptor_json")
        if type(decoded) is not dict or set(decoded) != {
            "artifact_type",
            "content_hash",
            "metadata",
            "path_or_uri",
            "size_bytes",
        }:
            raise DatabaseEffectMarkerDivergence("Database effect marker descriptor has a divergent field set")
        try:
            return ArtifactDescriptor(
                artifact_type=decoded["artifact_type"],
                path_or_uri=decoded["path_or_uri"],
                content_hash=decoded["content_hash"],
                size_bytes=decoded["size_bytes"],
                metadata=decoded["metadata"],
            )
        except (TypeError, ValueError) as exc:
            raise DatabaseEffectMarkerDivergence("Database effect marker descriptor is invalid") from exc

    def _insert_effect_rows(
        self,
        conn: Connection,
        target: Table,
        members: Sequence[tuple[int, dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], tuple[int, ...], tuple[int, ...], tuple[DiversionAttribution, ...]]:
        rows = [row for _ordinal, row in members]
        batch_savepoint = conn.begin_nested()
        try:
            conn.execute(insert(target), rows)
            batch_savepoint.commit()
        except (IntegrityError, DataError):
            batch_savepoint.rollback()
        else:
            return rows, tuple(ordinal for ordinal, _row in members), (), ()

        accepted_rows: list[dict[str, Any]] = []
        accepted_ordinals: list[int] = []
        diverted_ordinals: list[int] = []
        diversion_attribution: list[DiversionAttribution] = []
        for ordinal, row in members:
            row_savepoint = conn.begin_nested()
            try:
                conn.execute(insert(target), [row])
                row_savepoint.commit()
            except (IntegrityError, DataError) as exc:
                row_savepoint.rollback()
                reason = f"Constraint violation: {exc.orig}"
                # Populate the live diversion log BEFORE the marker commits, so
                # the executor can route the diverted row with its real reason.
                # If no on_write_failure policy is configured this raises inside
                # the outer transaction, rolling everything back (fail closed) —
                # exactly mirroring the streaming write path.
                self._divert_row(row, row_index=ordinal, reason=reason)
                diverted_ordinals.append(ordinal)
                diversion_attribution.append(build_diversion_attribution(ordinal=ordinal, reason=reason))
            else:
                accepted_rows.append(row)
                accepted_ordinals.append(ordinal)
        return accepted_rows, tuple(accepted_ordinals), tuple(diverted_ordinals), tuple(diversion_attribution)

    def _result_from_marker(
        self,
        marker: Mapping[str, object],
        plan: SinkEffectPlan,
        *,
        member_count: int,
    ) -> SinkEffectCommitResult:
        if (
            marker.get("effect_id") != plan.effect_id
            or marker.get("schema_version") != _DATABASE_EFFECT_LEDGER_SCHEMA_VERSION
            or marker.get("protocol_version") != SINK_EFFECT_PROTOCOL_VERSION
            or marker.get("plan_hash") != plan.plan_hash
            or marker.get("payload_hash") != plan.payload_hash
            or marker.get("completed") is not True
        ):
            raise DatabaseEffectMarkerDivergence("Database effect marker does not exactly bind this effect plan")
        accepted_value = self._require_canonical_json(marker.get("accepted_ordinals_json"), field_name="accepted_ordinals_json")
        diverted_value = self._require_canonical_json(marker.get("diverted_ordinals_json"), field_name="diverted_ordinals_json")
        diversion_hashes = self._require_canonical_json(marker.get("diversion_hashes_json"), field_name="diversion_hashes_json")
        evidence_value = self._require_canonical_json(marker.get("evidence_json"), field_name="evidence_json")
        if (
            type(accepted_value) is not list
            or type(diverted_value) is not list
            or type(diversion_hashes) is not list
            or type(evidence_value) is not dict
        ):
            raise DatabaseEffectMarkerDivergence("Database effect marker result fields have invalid JSON shapes")
        accepted = tuple(accepted_value)
        diverted = tuple(diverted_value)
        if (
            any(type(value) is not int or value < 0 for value in (*accepted, *diverted))
            or accepted != tuple(sorted(set(accepted)))
            or diverted != tuple(sorted(set(diverted)))
            or set(accepted) & set(diverted)
            or set(accepted) | set(diverted) != set(range(member_count))
        ):
            raise DatabaseEffectMarkerDivergence("Database effect marker ordinals are not a complete disjoint result partition")
        if len(diversion_hashes) != len(diverted):
            raise DatabaseEffectMarkerDivergence("Database effect marker diversion hashes do not cover diverted ordinals")
        for ordinal, item in zip(diverted, diversion_hashes, strict=True):
            if (
                type(item) is not dict
                or set(item) != {"error_hash", "ordinal", "reason_hash"}
                or item["ordinal"] != ordinal
                or not isinstance(item["reason_hash"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["reason_hash"]) is None
                or not isinstance(item["error_hash"], str)
                or re.fullmatch(r"[0-9a-f]{16}", item["error_hash"]) is None
            ):
                raise DatabaseEffectMarkerDivergence("Database effect marker diversion hashes are invalid")
        descriptor = self._descriptor_from_json(marker.get("descriptor_json"))
        accepted_payload_hash = marker.get("accepted_payload_hash")
        if accepted_payload_hash != descriptor.content_hash:
            raise DatabaseEffectMarkerDivergence("Database effect marker accepted payload hash diverges from descriptor")
        metadata = None if descriptor.metadata is None else deep_thaw(descriptor.metadata)
        if (
            descriptor.artifact_type != "database"
            or type(metadata) is not dict
            or metadata.get("table") != self._table_name
            or metadata.get("row_count") != len(accepted)
        ):
            raise DatabaseEffectMarkerDivergence("Database effect marker descriptor does not bind the configured target/result")
        expected_descriptor = ArtifactDescriptor.for_database(
            url=self._sanitized_url,
            table=self._table_name,
            content_hash=descriptor.content_hash,
            payload_size=descriptor.size_bytes,
            row_count=len(accepted),
        )
        if descriptor != expected_descriptor:
            raise DatabaseEffectMarkerDivergence("Database effect marker descriptor does not bind the configured database URL")
        expected_evidence = {
            "accepted_ordinals": list(accepted),
            "descriptor": self._require_canonical_json(marker.get("descriptor_json"), field_name="descriptor_json"),
            "diversion_attribution": list(diversion_hashes),
            "diverted_ordinals": list(diverted),
        }
        if evidence_value != expected_evidence:
            raise DatabaseEffectMarkerDivergence("Database effect marker evidence does not bind its exact result")
        return SinkEffectCommitResult(
            descriptor=descriptor,
            evidence=evidence_value,
            accepted_ordinals=accepted,
            diverted_ordinals=diverted,
        )

    def _read_effect_marker(self, conn: Connection, ledger: Table, effect_id: str) -> Mapping[str, object] | None:
        marker = conn.execute(select(ledger).where(ledger.c.effect_id == effect_id)).mappings().one_or_none()
        return None if marker is None else cast("Mapping[str, object]", marker)

    def _read_committed_effect_result(
        self,
        plan: SinkEffectPlan,
        *,
        member_count: int,
    ) -> SinkEffectCommitResult | None:
        self._ensure_engine_and_metadata_initialized()
        engine = self._engine
        if engine is None:  # pragma: no cover - paired initializer invariant
            raise RuntimeError("Database sink effect marker read called before engine initialization")
        ledger_config = self._require_effect_ledger_config()
        with engine.connect() as conn:
            ledger = Table(ledger_config.table, MetaData(), autoload_with=conn)
            marker = self._read_effect_marker(conn, ledger, plan.effect_id)
        if marker is None:
            return None
        return self._result_from_marker(marker, plan, member_count=member_count)

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        del ctx
        members = self._parse_database_effect_plan(plan)
        self._ensure_engine_and_metadata_initialized()
        engine = self._engine
        if engine is None:  # pragma: no cover - paired initializer invariant
            raise RuntimeError("Database sink effect commit called before engine initialization")
        ledger_config = self._require_effect_ledger_config()
        try:
            with engine.begin() as conn:
                ledger = Table(ledger_config.table, MetaData(), autoload_with=conn)
                existing = self._read_effect_marker(conn, ledger, plan.effect_id)
                if existing is not None:
                    return self._result_from_marker(existing, plan, member_count=len(members))
                target = Table(self._table_name, MetaData(), autoload_with=conn)
                if engine.dialect.name == "sqlite":
                    # Python's sqlite3 legacy transaction mode does not BEGIN
                    # for SELECT/DDL, and a first SAVEPOINT can otherwise become
                    # the outer transaction whose RELEASE commits accepted rows
                    # before the marker insert. Establish a real outer write
                    # transaction so every savepoint and the marker share it.
                    driver_connection = conn.connection.driver_connection
                    if not bool(getattr(driver_connection, "in_transaction", False)):
                        conn.exec_driver_sql("BEGIN IMMEDIATE")
                target_columns = set(target.columns.keys())
                planned_columns = set(deep_thaw(plan.safe_evidence)["target_columns"])
                if target_columns != planned_columns:
                    raise DatabaseEffectLedgerError("Database target table columns changed after effect inspection")
                accepted_rows, accepted, diverted, diversion_attribution = self._insert_effect_rows(conn, target, members)
                attribution_payload = [item.as_mapping() for item in diversion_attribution]
                canonical_payload = canonical_json(accepted_rows).encode("utf-8")
                content_hash = hashlib.sha256(canonical_payload).hexdigest()
                descriptor = ArtifactDescriptor.for_database(
                    url=self._sanitized_url,
                    table=self._table_name,
                    content_hash=content_hash,
                    payload_size=len(canonical_payload),
                    row_count=len(accepted_rows),
                )
                evidence: dict[str, object] = {
                    "accepted_ordinals": list(accepted),
                    "descriptor": json.loads(self._descriptor_json(descriptor)),
                    "diversion_attribution": attribution_payload,
                    "diverted_ordinals": list(diverted),
                }
                conn.execute(
                    insert(ledger).values(
                        effect_id=plan.effect_id,
                        schema_version=ledger_config.schema_version,
                        protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
                        plan_hash=plan.plan_hash,
                        payload_hash=plan.payload_hash,
                        completed=True,
                        descriptor_json=self._descriptor_json(descriptor),
                        accepted_ordinals_json=canonical_json(list(accepted)),
                        diverted_ordinals_json=canonical_json(list(diverted)),
                        accepted_payload_hash=content_hash,
                        diversion_hashes_json=canonical_json(attribution_payload),
                        evidence_json=canonical_json(evidence),
                    )
                )
                result = SinkEffectCommitResult(
                    descriptor=descriptor,
                    evidence=evidence,
                    accepted_ordinals=accepted,
                    diverted_ordinals=diverted,
                )
        except IntegrityError:
            winner = self._read_committed_effect_result(plan, member_count=len(members))
            if winner is None:
                raise
            return winner
        return result

    def reconcile_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        del ctx
        members = self._parse_database_effect_plan(plan)
        try:
            result = self._read_committed_effect_result(plan, member_count=len(members))
        except (DatabaseEffectLedgerError, SQLAlchemyError):
            return SinkEffectReconcileResult.unknown(
                evidence={"ledger_table": self._require_effect_ledger_config().table, "reason": "marker_divergent_or_unreadable"}
            )
        if result is None:
            return SinkEffectReconcileResult.not_applied(
                evidence={"ledger_table": self._require_effect_ledger_config().table, "marker": "absent"}
            )
        return SinkEffectReconcileResult.applied(
            result.descriptor,
            evidence=result.evidence,
            accepted_ordinals=result.accepted_ordinals,
            diverted_ordinals=result.diverted_ordinals,
        )

    def _compute_any_typed_fields(self) -> frozenset[str]:
        """Identify fields with 'any' type from the schema config.

        These fields may contain dict/list values that must be serialized
        to JSON strings before INSERT into TEXT columns.
        """
        if self._schema_config.is_observed or not self._schema_config.fields:
            return frozenset()
        return frozenset(f.name for f in self._schema_config.fields if f.field_type == "any")

    def _serialize_any_typed_fields(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serialize dict/list values in 'any'-typed fields to JSON strings.

        SQL TEXT columns cannot store Python dicts or lists. This method
        converts non-scalar values to their JSON string representation
        before INSERT, ensuring valid 'any' payloads (e.g., {"k": 1})
        are stored as '{"k": 1}' rather than crashing with a driver error.

        For observed-mode schemas, ALL fields are checked since any field
        could contain a complex value when the schema is inferred.

        For flexible-mode schemas, ALL fields are also checked: declared fields
        have their configured types, but extra columns are added as plain Text
        at table-creation time and never enter _any_typed_fields -- so a dict/list
        value in an extra column would hit the driver raw and raise ProgrammingError.
        Serializing all fields in flexible mode mirrors the observed-mode path.

        Scalar values (str, int, float, bool, None) are left unchanged.
        """
        is_flexible = self._schema_config.mode == "flexible"
        if not self._any_typed_fields and not self._schema_config.is_observed and not is_flexible:
            return rows

        result = []
        for row in rows:
            new_row = dict(row)
            # Observed and flexible modes must check every key: any column may
            # hold a complex value that the Text column cannot accept raw.
            if self._schema_config.is_observed or is_flexible:
                fields_to_check: frozenset[str] | set[str] = set(row.keys())
            else:
                fields_to_check = self._any_typed_fields
            for field in fields_to_check:
                if field in new_row:
                    value = new_row[field]
                    if isinstance(value, (dict, list)):
                        new_row[field] = json.dumps(value)
            result.append(new_row)
        return result

    def _ensure_engine_and_metadata_initialized(self) -> None:
        """Initialize engine/metadata pair together.

        Invariant: if self._engine is set, self._metadata must also be set.
        This keeps validate_output_target() and the effect lifecycle paths consistent.
        """
        if self._engine is None:
            self._engine = create_engine(self._url)
        if self._metadata is None:
            self._metadata = MetaData()

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing database table columns against configured schema.

        Checks that:
        - Strict mode: Table columns match schema fields exactly (set comparison)
        - Free mode: All schema fields present as columns (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing columns)

        Note: Unlike CSV, column order is not validated for databases.

        Returns:
            OutputValidationResult indicating compatibility.
        """
        from sqlalchemy import inspect

        from elspeth.contracts.sink import OutputValidationResult

        # Ensure engine/metadata are initialized consistently before inspection.
        self._ensure_engine_and_metadata_initialized()
        if self._engine is None:
            raise RuntimeError("Database sink validation called before initialization")

        inspector = inspect(self._engine)
        if not inspector.has_table(self._table_name):
            return OutputValidationResult.success()  # Will create table

        # Get existing columns
        columns = inspector.get_columns(self._table_name)
        existing = [col["name"] for col in columns]

        # Dynamic schema = no validation needed
        if self._schema_config.is_observed:
            return OutputValidationResult.success(target_fields=existing)

        # Get expected fields from schema (guaranteed non-None when not dynamic)
        fields = self._schema_config.fields
        if fields is None:
            return OutputValidationResult.success(target_fields=existing)
        expected = [f.name for f in fields]
        existing_set, expected_set = set(existing), set(expected)

        if self._schema_config.mode == "fixed":
            # Fixed: exact column match (set comparison, no order)
            if existing_set != expected_set:
                return OutputValidationResult.failure(
                    message="Table columns do not match schema (fixed mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                )
        else:  # mode == "flexible"
            # Flexible: schema fields must exist as columns (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="Table missing required schema columns (flexible mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        del rows, ctx
        raise RuntimeError("DatabaseSink publication requires the recoverable sink effect coordinator") from None

    def flush(self) -> None:
        """No-op: ``commit_effect`` commits its transaction before returning, so there are no pending operations."""

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._metadata = None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="database",
                issue_code=None,
                summary="Write rows to a SQL table (Postgres, SQLite, SQL Server) via SQLAlchemy. Write behaviour is set by if_exists: 'append' (default) or 'replace'.",
                composer_hints=(
                    "if_exists: 'append' (default — insert into the existing table, creating it if missing) or 'replace' (drops and recreates the table on the FIRST write — destructive). These are the only two values; there is no insert/upsert mode.",
                    "url is sanitised and audit-recorded — never put credentials inline; use the secrets store.",
                    "Schema fields map to column types: string→TEXT, int→Integer, float→Float, bool→Boolean. Other types need explicit type_coerce upstream.",
                    "if_exists='replace' is irreversible — it drops the table on first write. Confirm with the operator before declaring 'replace' on existing data.",
                    "on_write_failure routes per-row constraint violations (UNIQUE / NOT NULL / CHECK / foreign-key) — the offending row is diverted and the rest of the batch commits. Batch-integrity failures (connection loss, lock timeout, bad SQL) are not row-attributable and crash the run.",
                ),
            )
        return None

    @classmethod
    def get_post_call_hints(
        cls,
        *,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        hints: list[str] = []
        if config_snapshot.get("if_exists") == "replace":
            hints.append(
                "if_exists: 'replace' DROPS and recreates the target table on the first write. Confirm with the operator that the existing data is expendable before running."
            )
        return tuple(hints)
