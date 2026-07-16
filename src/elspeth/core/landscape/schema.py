"""SQLAlchemy table definitions for Landscape.

Uses SQLAlchemy Core (not ORM) for explicit control over queries
and compatibility with multiple database backends.
"""

from enum import StrEnum

from sqlalchemy import (
    DDL,
    Boolean,
    CheckConstraint,
    Column,
    ColumnElement,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    event,
    false,
    or_,
    select,
    text,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler

from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.contracts.types import NODE_ID_MAX_LENGTH
from elspeth.core.schema_identity import create_schema_identity_table

# Shared metadata for all tables
metadata = MetaData()


class _LowerHex64Check(ColumnElement[bool]):
    """Dialect-exact lowercase SHA-256 CHECK expression."""

    inherit_cache = True

    def __init__(self, column_name: str) -> None:
        super().__init__()
        self.column_name = column_name


@compiles(_LowerHex64Check, "sqlite")
def _compile_sqlite_lower_hex(element: _LowerHex64Check, _compiler: SQLCompiler, **_kw: object) -> str:
    name = element.column_name
    return f"length({name})=64 AND {name} NOT GLOB '*[^0-9a-f]*'"


@compiles(_LowerHex64Check, "postgresql")
def _compile_postgres_lower_hex(element: _LowerHex64Check, _compiler: SQLCompiler, **_kw: object) -> str:
    return f"{element.column_name} ~ '^[0-9a-f]{{64}}$'"


class _SigningTupleCheck(ColumnElement[bool]):
    """Dialect-exact closed HMAC/UNSIGNED snapshot signing tuple."""

    inherit_cache = True


@compiles(_SigningTupleCheck, "sqlite")
def _compile_sqlite_signing_tuple(_element: _SigningTupleCheck, _compiler: SQLCompiler, **_kw: object) -> str:
    return (
        "(signing_mode = 'hmac_sha256' AND signer_key_id <> 'UNSIGNED' "
        "AND length(trim(signer_key_id)) > 0 AND signature_hex IS NOT NULL "
        "AND length(signature_hex)=64 AND signature_hex NOT GLOB '*[^0-9a-f]*' "
        "AND record_chain_algorithm = 'sha256_concat_hmac_sha256_signatures_v1') OR "
        "(signing_mode = 'unsigned' AND signer_key_id = 'UNSIGNED' AND signature_hex IS NULL "
        "AND record_chain_algorithm = 'sha256_concat_record_sha256_v1')"
    )


@compiles(_SigningTupleCheck, "postgresql")
def _compile_postgres_signing_tuple(_element: _SigningTupleCheck, _compiler: SQLCompiler, **_kw: object) -> str:
    return (
        "(signing_mode = 'hmac_sha256' AND signer_key_id <> 'UNSIGNED' "
        "AND length(trim(signer_key_id)) > 0 AND signature_hex IS NOT NULL "
        "AND signature_hex ~ '^[0-9a-f]{64}$' "
        "AND record_chain_algorithm = 'sha256_concat_hmac_sha256_signatures_v1') OR "
        "(signing_mode = 'unsigned' AND signer_key_id = 'UNSIGNED' AND signature_hex IS NULL "
        "AND record_chain_algorithm = 'sha256_concat_record_sha256_v1')"
    )


class _OptionalLowerHex64Check(ColumnElement[bool]):
    inherit_cache = True

    def __init__(self, column_name: str) -> None:
        super().__init__()
        self.column_name = column_name


@compiles(_OptionalLowerHex64Check, "sqlite")
def _compile_sqlite_optional_lower_hex(element: _OptionalLowerHex64Check, _compiler: SQLCompiler, **_kw: object) -> str:
    name = element.column_name
    return f"{name} IS NULL OR (length({name})=64 AND {name} NOT GLOB '*[^0-9a-f]*')"


@compiles(_OptionalLowerHex64Check, "postgresql")
def _compile_postgres_optional_lower_hex(element: _OptionalLowerHex64Check, _compiler: SQLCompiler, **_kw: object) -> str:
    name = element.column_name
    return f"{name} IS NULL OR {name} ~ '^[0-9a-f]{{64}}$'"


def _sql_string_literal(value: str) -> str:
    """Render one deterministic SQL string literal for generated CHECK clauses."""
    return "'" + value.replace("'", "''") + "'"


def _enum_value_list(enum_type: type[StrEnum]) -> str:
    return ", ".join(_sql_string_literal(member.value) for member in enum_type)


def _enum_in_check(column_name: str, enum_type: type[StrEnum]) -> str:
    """Render a SQL IN fragment from a StrEnum's persisted values."""
    return f"{column_name} IN ({_enum_value_list(enum_type)})"


def _optional_enum_in_check(column_name: str, enum_type: type[StrEnum]) -> str:
    return f"{column_name} IS NULL OR {_enum_in_check(column_name, enum_type)}"


# Explicit Landscape schema epoch for pre-1.0 compatibility policy.
# Stored in the cross-dialect identity row and, on SQLite, PRAGMA user_version
# so future releases can distinguish "intentionally old schema, needs
# migration" from "runtime-required field". The constant retains its
# historical name for compatibility.
#
# Epoch history (pre-1.0 policy — bumps require DB recreation unless an epoch
# entry explicitly names an in-place migration):
#   1 → initial
#   2 → Phase 5 schema contracts + operation I/O hashes (pre-ADR-010)
#   3 → ADR-010 §Decision 3 M3: runtime_val_manifest_json
#        column on runs_table records the declaration + Tier-1 registries
#        in effect at run start, enabling auditor queries like "which VAL
#        contracts were in force during run X?"
#   4 → validation_errors.row_id links quarantine errors to persisted rows,
#        allowing explain() to resolve the exact validation failures for
#        quarantined tokens even when the stored row payload is normalized
#        before persistence.
#   5 → batch/artifact audit links are mechanically run-scoped: batch_members
#        carries run_id, token_outcomes.batch_id is scoped by run_id,
#        batches.aggregation_state_id is scoped by run_id, and
#        artifacts.produced_by_state_id is scoped by run_id.
#   6 → batches.retry_of_batch_id records retry lineage so retry_batch()
#        can deduplicate per failed batch rather than per aggregation node.
#   7 → ADR-019 Stage 2/3: token_outcomes stores the two-axis terminal model
#        (`outcome`, `path`, `completed`) instead of the old single-axis outcome + is_terminal.
#   8 → Phase 5b interpretation-review audit anchor:
#        calls.resolved_prompt_template_hash records the runtime-side hash used
#        to join Landscape LLM calls back to session interpretation_events.
#   9 → Phase 4 hello-world tutorial audit-story fields:
#        runs.llm_call_count, runs.seeded_from_cache, and runs.cache_key.
#  10 → OpenRouter catalog snapshot anchor (audit-completeness):
#        runs.openrouter_catalog_sha256 and runs.openrouter_catalog_source
#        record which model catalog blessed each run's decisions.
#  11 → resume fork/expand/coalesce re-emit fix: tokens.token_data_ref persists per-token
#        payloads (expand children + coalesce merged tokens) and node_states.resume_checkpoint_id
#        marks resume re-drives, so incomplete tokens are reconstructable + attributable.
#  12 → Multi-source foundation: per-source run records, source-scoped row
#        indexes, global ingest sequence ordering, and durable token work items.
#  13 → Durable scheduler continuation routing: token_work_items.on_success_sink
#        preserves sink-bound continuations across resume/recovery.
#  14 → Durable scheduler resume identity: token_work_items stores token lineage
#        and coalesce cursor fields so resumed workers do not depend on in-memory
#        pending_items state.
#  15 → Durable scheduler sink handoff: token_work_items can persist a
#        pending sink outcome without replaying the transform that produced it;
#        run_sources.lifecycle_state is mechanically constrained.
#  15 → Multi-worker scheduler durability: token_work_items grows a
#        CHECK constraint that LEASED rows have a non-empty lease_owner
#        (closes the wedge that elspeth-28aaa36a62's recovery sweep tolerates)
#        and a covering index on (run_id, status, lease_owner,
#        lease_expires_at) so the RC6 multi-worker drain sweep is index-only
#        rather than O(workers²) per drain wave (elspeth-9990c81e14).
#  16 → Scheduler state-transition audit: scheduler_events records immutable
#        enqueue, claim, recovery, lease-loss, and terminalization transitions
#        for durable lease attribution (elspeth-2b608abbd3,
#        elspeth-9030f34c32).
#  17 → Scheduler event lease-expiry evidence: scheduler_events stores
#        from/to lease_expires_at timestamps and enforces event/status/attempt
#        domains so lease recovery and heartbeat loss are exportable facts.
#   17 → Token-scoped pending-sink lookup: token_work_items gains
#        ix_token_work_items_pending_sink_token for post-sink terminalization.
#   18 → Source exhaustion lifecycle evidence: run_sources.lifecycle_state
#        distinguishes exhausted-before-EOF-engine-work from interrupted source
#        iteration so resume can safely drain recoverable engine state.
#   19 → Dead-lane and vestigial-anchor subtraction (F4+F2): scheduler WAITING
#        status and mark_waiting/release_waiting event types removed from the
#        scheduler_events CHECK constraints; checkpoints lose the token_id/
#        node_id anchor columns and checkpoint_node_config_hash (full-topology
#        hash subsumes per-node compatibility validation).
#   20 → F1 durability unification: token_work_items.barrier_blocked_at added;
#        checkpoints aggregation_state_json/coalesce_state_json replaced by
#        barrier_scalars_json; restore_blocked event type removed from
#        scheduler_events CHECK.
#   21 → Option-C multi-worker coordination substrate (ADR-030, slice 2):
#        run_coordination / run_workers / run_coordination_events /
#        coalesce_branch_losses tables; token_work_items gains
#        barrier_adopted_epoch (adoption CAS marker, written only by the
#        slice-3 fenced adoption verb; NULL = intake-pending).
#   22 → Routing events are run-scoped: routing_events carries run_id and
#        composite FKs to node_states(state_id, run_id) and edges(edge_id, run_id)
#        so state/edge route decisions cannot cross audit-run boundaries.
#   23 → Web plugin-policy audit evidence: run_web_plugin_policy stores one
#        optional, sanitized policy/snapshot decision row per web run. This is
#        a deliberate pre-1.0 drop/recreate boundary, not an in-place migration.
#   24 → elspeth_schema_identity gives SQLite and PostgreSQL the same explicit
#        application/store/epoch proof, including semantic-only schema bumps;
#        token ownership is also mechanically row-authoritative: tokens(row_id,
#        run_id) references rows(row_id, run_id). Populated epoch-23 SQLite
#        databases receive a narrow, transactional tokens-table rebuild after
#        a mismatch preflight; all older epochs retain the recreate boundary.
#   25 → Artifact logical-effect idempotency: non-null artifact keys are unique
#        within a run, so retries converge on one immutable audit identity.
#        Populated exact epoch-24 SQLite databases receive a narrow,
#        transactional partial-index migration after a duplicate preflight;
#        exact epoch-23 databases take the ordered 23→24→25 chain.
#   26 → Recoverable sink-effect ledger and immutable audit-export snapshot
#        registry; artifacts/operations gain exclusive effect linkage.
SQLITE_SCHEMA_EPOCH = 26

schema_identity_table = create_schema_identity_table(metadata)

# Column width for node_id across all tables. The cross-layer identifier limit
# lives in contracts.types; changing this value requires an Alembic migration.
NODE_ID_COLUMN_LENGTH = NODE_ID_MAX_LENGTH


class RunSourceLifecycleState(StrEnum):
    """Finite lifecycle states for per-source run metadata."""

    READY = "ready"
    LOADING = "loading"
    EXHAUSTED = "exhausted"
    LOADED = "loaded"
    INTERRUPTED = "interrupted"


# === Runs and Configuration ===

runs_table = Table(
    "runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("config_hash", String(64), nullable=False),
    Column("settings_json", Text, nullable=False),
    Column("reproducibility_grade", String(32)),
    Column("canonical_version", String(64), nullable=False),
    # Source schema for resume type restoration
    # Stores serialized PluginSchema class info to enable proper type coercion
    # when resuming from payloads (datetime/Decimal string -> typed values)
    Column("source_schema_json", Text),  # Nullable — populated on new runs, NULL for runs created before this column
    # Field resolution mapping from source.load() - captures original→final header mapping
    # Field normalization is mandatory; mapping records original→final header names. Stored at run level since one source per run.
    Column("source_field_resolution_json", Text),  # Nullable — populated on new runs, NULL for runs created before this column
    Column("status", String(32), nullable=False),
    # Export tracking - separate from run status so export failures
    # don't mask successful pipeline completion
    Column("export_status", String(32)),  # pending, completed, failed, None if not configured
    Column("export_error", Text),  # Error message if export failed
    Column("exported_at", DateTime(timezone=True)),  # When export completed
    Column("export_format", String(16)),  # csv, json
    Column("export_sink", String(128)),  # Sink name used for export
    # Runtime-VAL manifest for audit trail (ADR-010 §Decision 3 M3).
    # Captures the set of DeclarationContract and Tier-1 error classes
    # registered at bootstrap, serialized as canonical JSON. Enables auditor
    # queries like "which VAL contracts were in force during run X?" and
    # regression detection across runs ("are TIER_1_ERRORS the same between
    # runs X and Y?"). The column is nullable for resume paths and for
    # tests that skip the full bootstrap path.
    Column("runtime_val_manifest_json", Text),
    # Phase 4 audit-story projection fields. `started_at` already exists
    # above; source_data_hash and plugin_versions remain on rows/nodes and
    # are aggregated by the read side instead of denormalized here.
    Column("llm_call_count", Integer, nullable=True),
    Column("seeded_from_cache", Boolean, nullable=False, default=False, server_default=false()),
    Column("cache_key", String(64), nullable=True),
    # OpenRouter model catalog snapshot anchor (audit-completeness):
    # records which catalog blessed the run's model decisions.  The sha
    # is canonical (sorted utf-8 ids joined on '\n') so it is invariant
    # under prime-order; source ∈ {"live", "bundled"} distinguishes
    # online-probed snapshots from the bundled litellm fallback.  Both
    # NOT NULL — see ``read_openrouter_catalog_snapshot_id`` for the
    # reader the orchestrator uses to populate them.  Per CLAUDE.md
    # Tier-1 doctrine no ``server_default`` is set: a synthetic
    # placeholder in the audit trail would be indistinguishable from a
    # real hash to any downstream reader, violating the fabrication
    # test.  Production goes through :meth:`RunLifecycleRepository.begin_run`
    # which validates both fields; direct ``runs_table.insert()`` from
    # test fixtures must supply them explicitly.
    Column("openrouter_catalog_sha256", String(64), nullable=False),
    Column("openrouter_catalog_source", String(8), nullable=False),
    CheckConstraint(
        "openrouter_catalog_source IN ('live', 'bundled')",
        name="ck_runs_openrouter_catalog_source",
    ),
)
Index("uq_runs_export_witness", runs_table.c.run_id, runs_table.c.status, runs_table.c.completed_at, unique=True)

run_attributions_table = Table(
    "run_attributions",
    metadata,
    Column("run_id", String(64), ForeignKey("runs.run_id"), primary_key=True),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("initiated_by_user_id", String(255), nullable=False),
    Column("auth_provider_type", String(32), nullable=False),
    CheckConstraint("auth_provider_type IN ('local', 'oidc', 'entra')", name="ck_run_attributions_auth_provider_type"),
)
Index("ix_run_attributions_user", run_attributions_table.c.initiated_by_user_id, run_attributions_table.c.auth_provider_type)

run_web_plugin_policy_table = Table(
    "run_web_plugin_policy",
    metadata,
    Column("run_id", String(64), ForeignKey("runs.run_id"), primary_key=True),
    Column("schema_version", Integer, nullable=False),
    Column("policy_hash", String(64), nullable=False),
    Column("snapshot_hash", String(64), nullable=False),
    Column("authorized_plugin_ids_json", Text, nullable=False),
    Column("available_plugin_ids_json", Text, nullable=False),
    Column("control_modes_json", Text, nullable=False),
    Column("selected_implementations_json", Text, nullable=False),
    Column("selected_profile_aliases_json", Text, nullable=False),
    Column("plugin_code_identities_json", Text, nullable=False),
    Column("binding_generation_fingerprint", String(64), nullable=False),
    Column("decision_codes_json", Text, nullable=False),
    CheckConstraint("schema_version >= 1", name="ck_run_web_plugin_policy_schema_version"),
)

run_sources_table = Table(
    "run_sources",
    metadata,
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("source_name", String(64), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("lifecycle_state", String(32), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("schema_json", Text),
    Column("schema_contract_json", Text),
    # SchemaContract.version_hash() is a 32-character (128-bit) hex digest.
    # PostgreSQL enforces VARCHAR widths (SQLite does not), so this column must
    # match the runtime contract rather than the historical 16-char width.
    Column("schema_contract_hash", String(32)),
    Column("field_resolution_json", Text),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("run_id", "source_node_id"),
    UniqueConstraint("run_id", "source_name"),
    CheckConstraint(
        _enum_in_check("lifecycle_state", RunSourceLifecycleState),
        name="ck_run_sources_lifecycle_state",
    ),
    ForeignKeyConstraint(["source_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
Index("ix_run_sources_run", run_sources_table.c.run_id)
Index("ix_run_sources_source_name", run_sources_table.c.run_id, run_sources_table.c.source_name)

# === Nodes (Plugin Instances) ===

nodes_table = Table(
    "nodes",
    metadata,
    Column("node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("node_type", String(32), nullable=False),
    Column("plugin_version", String(32), nullable=False),
    Column("source_file_hash", String(32), nullable=True),
    Column("determinism", String(32), nullable=False),  # deterministic, seeded, nondeterministic (from Determinism enum)
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64)),
    Column("sequence_in_pipeline", Integer),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    # Schema configuration for audit trail (WP-11.99)
    Column("schema_mode", String(16)),  # "observed", "fixed", "flexible", "parse", or NULL
    Column("schema_fields_json", Text),  # JSON array of field definitions, or NULL
    # Schema contracts for audit trail (Phase 5: Unified Schema Contracts)
    # Input contract: what the node requires (field names and types)
    Column("input_contract_json", Text),
    # Output contract: what the node guarantees (field names and types)
    Column("output_contract_json", Text),
    # Composite PK: same node config can exist in multiple runs
    # This allows running the same pipeline multiple times against the same database
    PrimaryKeyConstraint("node_id", "run_id"),
)

# === Edges ===

edges_table = Table(
    "edges",
    metadata,
    Column("edge_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("from_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("to_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("label", String(64), nullable=False),
    Column("default_mode", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite unique target for run-scoped FKs to edges.
    UniqueConstraint("edge_id", "run_id"),
    UniqueConstraint("run_id", "from_node_id", "label"),
    # Composite FKs to nodes (node_id, run_id)
    ForeignKeyConstraint(["from_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    ForeignKeyConstraint(["to_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# === Source Rows ===
#
# Audit-DB invariants for the rows table (filigree elspeth-56c3cda89b,
# ADR-bundle systems-thinker Finding 5, "Tragedy of the Commons"). The shared
# resource is the audit database's invariant surface; the actors are 25+
# downstream consumers (audit-readiness panel, ``elspeth explain``, MCP
# failure-context, redaction policy, composer state, resume orchestrator,
# replay verifier...). Each consumer carries its own assumption about
# per-source provenance; without a central guarantor, the failure mode is
# *simultaneous* across consumers rather than progressive.
#
# Invariants by mechanical enforcement status:
#
# 1. ``source_node_id NOT NULL`` (this table, line 225). STRUCTURALLY
#    ENFORCED at schema level. Every row in the audit trail attributes to a
#    specific source node, never NULL.
#
# 2. ``ingest_sequence`` is monotone per run and globally unique within a
#    run. STRUCTURALLY ENFORCED via ``UniqueConstraint("run_id",
#    "ingest_sequence")`` (line 234) plus the ``NOT NULL`` on the column.
#
# 3. Compound row identity ``(run_id, source_node_id, source_row_index)``
#    is unique. STRUCTURALLY ENFORCED via ``UniqueConstraint`` at line 233.
#    Two sources emitting the same ``source_row_index`` value produce
#    distinct rows because ``source_node_id`` discriminates.
#
# Invariants NOT YET structurally enforced (gaps below the schema):
#
# A. ``source_row_index`` and ``ingest_sequence`` fabrication ban (G22).
#    ``data_flow_repository.create_row`` raises ``AuditIntegrityError`` when
#    these values are not explicitly provided ("Do not fabricate
#    source_row_index or ingest_sequence from row_index"), but the
#    prohibition lives in an exception string at one write boundary. The
#    cache-replay write path (``write_repository.record_synthesised_run``)
#    uses the row-index fallback only for single-source runs; multi-source
#    synthesised rows must provide explicit source_node_index,
#    source_row_index, ingest_sequence, and source_data_hash before the
#    writer inserts them. Tracked under filigree elspeth-92afea0d23
#    (elspeth-lints rule with the same enforcement status as
#    ``trust_tier.tier_model``).
#
# B. Scheduler lease-ownership transitions (G29). ``token_work_items``
#    carries the current lease state but not its transition history; a
#    lease-expiry event during multi-worker execution leaves no per-worker
#    audit attribution. Tracked under filigree elspeth-9030f34c32
#    (``scheduler_events`` table).

rows_table = Table(
    "rows",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("row_index", Integer),
    Column("source_row_index", Integer, nullable=False),
    Column("ingest_sequence", Integer, nullable=False),
    Column("source_data_hash", String(64), nullable=False),
    Column("source_data_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("row_id", "run_id"),
    UniqueConstraint("run_id", "source_node_id", "source_row_index"),
    UniqueConstraint("run_id", "ingest_sequence"),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["source_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# === Tokens ===

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(64), primary_key=True),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),  # Run ownership for cross-run contamination prevention
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(32), nullable=True, index=True),  # For deaggregation
    Column("branch_name", String(64)),
    Column("step_in_pipeline", Integer),  # Step where this token was created (fork/coalesce/expand)
    # Payload-store ref for a token whose row_data differs from its source row:
    # expand/deaggregation children (independently-transformed data) AND post-coalesce
    # merged tokens (the merged row, computed in memory at barrier time). NULL for fork
    # children, which share the parent/source payload retrievable by row_id. Enables
    # faithful reconstruction of incomplete expand/coalesce tokens on resume (epoch 11).
    Column("token_data_ref", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite unique target for downstream composite FKs (token_id, run_id)
    UniqueConstraint("token_id", "run_id"),
    # Epoch 24: row ownership is authoritative. The independent run FK proves
    # the run exists; this composite FK proves it is the row's run.
    ForeignKeyConstraint(["row_id", "run_id"], ["rows.row_id", "rows.run_id"]),
)
Index("uq_tokens_identity_row_run", tokens_table.c.token_id, tokens_table.c.row_id, tokens_table.c.run_id, unique=True)

# === Token Outcomes (AUD-001: Explicit terminal state recording) ===

token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False, index=True),
    # Composite FK: enforces token_id and run_id belong together (prevents cross-run contamination)
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    # ADR-019 two-axis terminal model. ``completed`` mirrors the prior
    # ``is_terminal`` column (sub-decision 3). ``outcome`` value space changed
    # from the old single-axis outcome (12 values, non-NULL) to TerminalOutcome (3 values:
    # success / failure / transient) with NULL when completed=False
    # (only ``BUFFERED`` today). ``path`` is producer-declared per ADR-019
    # § "Classification is producer-declared, not topology-derivable" and
    # always populated, including ``path="buffered"`` for non-terminal rows.
    Column("outcome", String(32), nullable=True),
    Column("path", String(64), nullable=False),
    Column("completed", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    # Outcome-specific fields (nullable based on (outcome, path) pair)
    Column("sink_name", String(128)),
    Column("batch_id", String(64)),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    # Optional extended context
    Column("context_json", Text),
    # Branch contract for FORKED/EXPANDED outcomes (enables recovery validation)
    Column("expected_branches_json", Text),
    # Composite FK: batch outcomes must point at a batch from the same run.
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)

# Partial unique index: exactly one terminal outcome per token
# Note: SQLite partial index syntax differs; SQLAlchemy handles this
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.completed == 1),
    postgresql_where=(token_outcomes_table.c.completed == 1),
)

token_work_items_table = Table(
    "token_work_items",
    metadata,
    Column("work_item_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False),
    Column("row_id", String(64), nullable=False),
    Column("node_id", String(NODE_ID_COLUMN_LENGTH)),
    Column("step_index", Integer, nullable=False),
    Column("ingest_sequence", Integer, nullable=False),
    Column("row_payload_json", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("queue_key", String(128)),
    Column("barrier_key", String(128)),
    Column("on_success_sink", String(128)),
    Column("pending_sink_name", String(128)),
    Column("pending_outcome", String(32)),
    Column("pending_path", String(64)),
    Column("pending_error_hash", String(64)),
    Column("pending_error_message", Text),
    Column("branch_name", String(128)),
    Column("fork_group_id", String(128)),
    Column("join_group_id", String(128)),
    Column("expand_group_id", String(128)),
    Column("coalesce_node_id", String(NODE_ID_COLUMN_LENGTH)),
    Column("coalesce_name", String(128)),
    Column("attempt", Integer, nullable=False),
    Column("lease_owner", String(128)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("available_at", DateTime(timezone=True), nullable=False),
    Column("barrier_blocked_at", DateTime(timezone=True), nullable=True),
    # Epoch 21: adoption CAS marker (§C.4 row 6a) — set only by the fenced
    # barrier-adoption verb (slice 3). NULL means intake-pending. Column lands
    # at epoch 21 so the schema is stable across slices 2→3.
    Column("barrier_adopted_epoch", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "token_id", "node_id", "attempt"),
    # ``status=leased`` must imply a non-empty ``lease_owner``. The
    # ``recover_expired_leases`` sweep's OR-NULL predicate (elspeth-28aaa36a62)
    # treats ``lease_owner=NULL`` as a recoverable wedge, so the CHECK closes
    # the structural gap by preventing the wedge from being written in the
    # first place (filigree elspeth-9990c81e14, embedded-database-reviewer).
    # The literal MUST match ``TokenWorkStatus.LEASED.value`` exactly — the
    # enum is a ``StrEnum`` whose ``.value`` is lowercase ``"leased"`` and
    # every write site persists ``.value`` (e.g. ``scheduler_repository.py``
    # ``status=TokenWorkStatus.LEASED.value``). A mismatched literal here
    # would make both arms of the CHECK trivially satisfied for every row
    # and silently nullify the Tier-1 invariant (elspeth-36d5635402).
    # The constraint runs independently of ``PRAGMA foreign_keys`` in SQLite.
    CheckConstraint(
        f"(status = {_sql_string_literal(TokenWorkStatus.LEASED.value)} "
        f"AND lease_owner IS NOT NULL AND length(lease_owner) > 0) OR status != {_sql_string_literal(TokenWorkStatus.LEASED.value)}",
        name="ck_token_work_items_lease_owner_required_when_leased",
    ),
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    ForeignKeyConstraint(["row_id", "run_id"], ["rows.row_id", "rows.run_id"]),
    ForeignKeyConstraint(["node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    ForeignKeyConstraint(["coalesce_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
Index("ix_token_work_items_ready", token_work_items_table.c.run_id, token_work_items_table.c.status, token_work_items_table.c.available_at)
Index(
    "ix_token_work_items_lease", token_work_items_table.c.run_id, token_work_items_table.c.status, token_work_items_table.c.lease_expires_at
)
# Covering index for the multi-worker drain ``recover_expired_leases`` sweep.
# ``ix_token_work_items_lease`` covers ``(run_id, status, lease_expires_at)``
# which served the pre-multi-worker predicate; the worker-skipping predicate
# ``lease_owner != caller_owner`` (elspeth-28aaa36a62, elspeth-941f1508f5)
# leaves SQLite to apply the inequality as a row filter against the existing
# index. Bounded by run-LEASED rows today, but the RC6 multi-worker target
# runs the sweep per-worker-per-iteration: O(workers²) per drain wave. The
# wider index puts ``lease_owner`` into the seek key so the sweep is index-
# only (filigree elspeth-9990c81e14, embedded-database-reviewer MED).
Index(
    "ix_token_work_items_recovery",
    token_work_items_table.c.run_id,
    token_work_items_table.c.status,
    token_work_items_table.c.lease_owner,
    token_work_items_table.c.lease_expires_at,
)
# Token-scoped lookup for the sink durability callback. Without this, SQLite
# plans ``mark_pending_sink_terminal`` as a run/status scan for every token,
# turning large sink batches into an O(n^2) post-write grind.
Index(
    "ix_token_work_items_pending_sink_token",
    token_work_items_table.c.run_id,
    token_work_items_table.c.token_id,
    token_work_items_table.c.status,
    token_work_items_table.c.pending_sink_name,
)
Index(
    "uq_token_work_items_terminal_identity",
    token_work_items_table.c.run_id,
    token_work_items_table.c.token_id,
    token_work_items_table.c.attempt,
    unique=True,
    sqlite_where=token_work_items_table.c.node_id.is_(None),
    postgresql_where=token_work_items_table.c.node_id.is_(None),
)


def pending_sink_bundle_clause() -> ColumnElement[bool]:
    """Return the complete durable sink-redrive subtype predicate.

    ``PENDING_SINK`` is not merely a status: the dedicated redrive path must
    be able to rebuild a legal sink-bound ``RowResult`` without replaying its
    producer.  The opaque row payload and sink identity must therefore be
    present, the persisted outcome/path pair must be one of the four
    sink-bound terminal pairs, and pair-specific evidence must be complete.

    Keep this predicate at the schema boundary so claim selection and its CAS
    UPDATE use the exact same SQL on SQLite and PostgreSQL.  Payload *shape*
    validation remains the separately owned scheduler-payload contract; this
    subtype guard only rejects the representable empty payload sentinel.
    """
    no_error_evidence = and_(
        token_work_items_table.c.pending_error_hash.is_(None),
        token_work_items_table.c.pending_error_message.is_(None),
    )
    return and_(
        token_work_items_table.c.row_payload_json != "",
        token_work_items_table.c.pending_sink_name.is_not(None),
        token_work_items_table.c.pending_sink_name != "",
        or_(
            and_(
                token_work_items_table.c.pending_outcome == TerminalOutcome.SUCCESS.value,
                token_work_items_table.c.pending_path == TerminalPath.DEFAULT_FLOW.value,
                no_error_evidence,
            ),
            and_(
                token_work_items_table.c.pending_outcome == TerminalOutcome.SUCCESS.value,
                token_work_items_table.c.pending_path == TerminalPath.GATE_ROUTED.value,
                no_error_evidence,
            ),
            and_(
                token_work_items_table.c.pending_outcome == TerminalOutcome.FAILURE.value,
                token_work_items_table.c.pending_path == TerminalPath.ON_ERROR_ROUTED.value,
                token_work_items_table.c.pending_error_hash.is_not(None),
                token_work_items_table.c.pending_error_hash != "",
                token_work_items_table.c.pending_error_message.is_not(None),
            ),
            and_(
                token_work_items_table.c.pending_outcome == TerminalOutcome.SUCCESS.value,
                token_work_items_table.c.pending_path == TerminalPath.COALESCED.value,
                token_work_items_table.c.join_group_id.is_not(None),
                token_work_items_table.c.join_group_id != "",
                no_error_evidence,
            ),
        ),
    )


def blocked_barrier_hold_clause() -> ColumnElement[bool]:
    """Predicate selecting journal BLOCKED rows that are BARRIER holds.

    BLOCKED rows are dual-use (F1 design D1): barrier holds carry a non-NULL
    ``barrier_key`` (coalesce_name for coalesce, str(node_id) for
    aggregation), while ADR-028 queue-holds carry only a ``queue_key``. The
    ``barrier_key IS NOT NULL`` filter is what keeps queue-holds out of
    barrier sweeps (restore, resume work-set exclusion, quiescence counting).

    Single source of truth for the dual-use predicate — shared by
    ``TokenSchedulerRepository.list_blocked_barrier_items`` and
    ``RecoveryManager._get_buffered_journal_token_ids`` /
    ``count_blocked_barrier_items``. The literal ``'blocked'`` MUST match
    ``TokenWorkStatus.BLOCKED.value`` (a lowercase ``StrEnum``), consistent
    with the status literals in this module's CHECK constraints.
    """
    return and_(
        token_work_items_table.c.status == "blocked",
        token_work_items_table.c.barrier_key.is_not(None),
    )


scheduler_events_table = Table(
    "scheduler_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("token_id", String(64), nullable=False),
    # ``work_item_id`` is a forensic scheduler identity, not a foreign key:
    # recover_expired_leases intentionally rotates token_work_items.work_item_id
    # on attempt bump. A FK or ON UPDATE CASCADE would rewrite immutable
    # history or reject the lease-loss event this table exists to preserve.
    Column("work_item_id", String(64), nullable=False),
    Column("node_id", String(NODE_ID_COLUMN_LENGTH)),
    Column("event_type", String(64), nullable=False),
    Column("from_status", String(32)),
    Column("to_status", String(32), nullable=False),
    Column("from_lease_owner", String(128)),
    Column("to_lease_owner", String(128)),
    Column("from_lease_expires_at", DateTime(timezone=True)),
    Column("to_lease_expires_at", DateTime(timezone=True)),
    Column("from_attempt", Integer),
    Column("to_attempt", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("caller_owner", String(128)),
    Column("context_json", Text, nullable=False, server_default=text("'{}'")),
    CheckConstraint(
        _enum_in_check("event_type", SchedulerEventType),
        name="ck_scheduler_events_event_type",
    ),
    CheckConstraint(
        _optional_enum_in_check("from_status", TokenWorkStatus),
        name="ck_scheduler_events_from_status",
    ),
    CheckConstraint(
        _enum_in_check("to_status", TokenWorkStatus),
        name="ck_scheduler_events_to_status",
    ),
    CheckConstraint("from_attempt IS NULL OR from_attempt >= 0", name="ck_scheduler_events_from_attempt_non_negative"),
    CheckConstraint("to_attempt >= 0", name="ck_scheduler_events_to_attempt_non_negative"),
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    ForeignKeyConstraint(["node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
Index(
    "ix_scheduler_events_run_token_time",
    scheduler_events_table.c.run_id,
    scheduler_events_table.c.token_id,
    scheduler_events_table.c.recorded_at,
    scheduler_events_table.c.event_id,
)
Index(
    "ix_scheduler_events_work_item",
    scheduler_events_table.c.run_id,
    scheduler_events_table.c.work_item_id,
    scheduler_events_table.c.recorded_at,
)

# === Multi-worker run coordination (epoch 21, ADR-030) ===

run_coordination_table = Table(
    "run_coordination",
    metadata,
    # Exactly one row per run, created by begin_run (uniformity rule: an N=1
    # worker is leader of its own run at epoch 1).
    Column("run_id", String(64), ForeignKey("runs.run_id"), primary_key=True),
    Column("leader_worker_id", String(128)),  # NULL = vacant seat
    # THE fencing token; monotonic, bumps on every acquisition CAS (§B.4).
    Column("leader_epoch", Integer, nullable=False, server_default=text("0")),
    Column("leader_heartbeat_expires_at", DateTime(timezone=True)),  # run-level leader liveness clock
    Column("updated_at", DateTime(timezone=True), nullable=False),
    # Vacant seat ⇔ no liveness clock; occupied seat ⇔ clock present.
    CheckConstraint(
        "(leader_worker_id IS NULL) = (leader_heartbeat_expires_at IS NULL)",
        name="ck_run_coordination_seat_liveness_paired",
    ),
)

run_workers_table = Table(
    "run_workers",
    metadata,
    Column("worker_id", String(128), primary_key=True),  # 'worker:{run_id}:{uuid4().hex}'
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("role", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    Column("heartbeat_expires_at", DateTime(timezone=True), nullable=False),  # run-level worker liveness clock
    Column("departed_at", DateTime(timezone=True)),
    Column("evicted_at", DateTime(timezone=True)),  # forensics (graft, Design 1)
    Column("evicted_by_worker_id", String(128)),  # forensics: who ran the eviction CAS
    # Forensic only — EXCEPT pid, surfaced by the BUSY-takeover diagnostic
    # (§B.4 WriteLockHeldError carries registered pids).
    Column("pid", Integer),
    Column("hostname", String(255)),
    Column("entry_point", String(255)),
    CheckConstraint("role IN ('leader','follower')", name="ck_run_workers_role"),
    CheckConstraint("status IN ('active','departed','evicted')", name="ck_run_workers_status"),
    CheckConstraint("(status = 'evicted') = (evicted_at IS NOT NULL)", name="ck_run_workers_evicted_at_paired"),
)
# Serves both hot paths: the slice-4 EXISTS membership fence probes
# (run_id, status='active', worker_id) — (run_id, status) prefix seek then
# worker_id filter on a tiny table — and the liveness reap scans
# (run_id, status, heartbeat_expires_at) index-only. Worker_id point lookups
# (heartbeat CAS) use the PK.
Index(
    "ix_run_workers_liveness",
    run_workers_table.c.run_id,
    run_workers_table.c.status,
    run_workers_table.c.heartbeat_expires_at,
)

run_coordination_events_table = Table(
    "run_coordination_events",
    metadata,
    # Authoritative replay order. AUTOINCREMENT (not bare rowid) so seq values
    # are never reused after deletion and are strictly monotonic for the life
    # of the ledger — process-stamped recorded_at can invert commit order
    # under busy_timeout stalls; seq cannot.
    Column("seq", Integer, primary_key=True, autoincrement=True),
    # sha256(canonical_json(identity)) — same dedup recipe as scheduler
    # events (scheduler_repository.py:434-455).
    Column("event_id", String(64), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("event_type", String(32), nullable=False),
    Column("worker_id", String(128), nullable=False),
    Column("leader_epoch", Integer),
    Column("recorded_at", DateTime(timezone=True), nullable=False),  # forensic wall-clock; NOT the replay order
    Column("context_json", Text, nullable=False, server_default=text("'{}'")),
    # All 10 event types from the design DDL (§A.2), including the slice-4
    # producers worker_stalled and heartbeat_degraded — pinned into the
    # epoch-21 CHECK now so slice 4 needs no schema change.
    CheckConstraint(
        "event_type IN ('worker_register', 'worker_depart', 'worker_evict', 'worker_stalled', "
        "'leader_acquire', 'leader_release', 'leadership_lost', "
        "'fence_refusal', 'heartbeat_degraded', 'finalize')",
        name="ck_run_coordination_events_event_type",
    ),
    # Mandatory: without the table kwarg, SQLAlchemy emits a bare INTEGER
    # PRIMARY KEY (rowid alias, values reusable after delete); with it the DDL
    # is `seq INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT`. Inert on Postgres
    # (Integer PK becomes IDENTITY — also monotonic).
    sqlite_autoincrement=True,
)
# event_id uniqueness is a named unique Index, not UniqueConstraint/unique=True,
# deliberately: SQLAlchemy's SQLite inspector get_indexes() does not report
# sqlite_autoindex_* indexes created by inline UNIQUE constraints, so a
# UniqueConstraint cannot be verified by the _REQUIRED_INDEXES loop. Same
# pattern as uq_token_work_items_terminal_identity above.
Index(
    "uq_run_coordination_events_event_id",
    run_coordination_events_table.c.event_id,
    unique=True,
)
Index(
    "ix_run_coordination_events_run",
    run_coordination_events_table.c.run_id,
    run_coordination_events_table.c.seq,
)

coalesce_branch_losses_table = Table(
    "coalesce_branch_losses",
    metadata,
    # §E.5: durable cross-worker branch-loss hand-off. Table only at epoch 21;
    # record/replay verbs land in slice 3.
    Column("loss_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("coalesce_name", String(128), nullable=False),
    Column("row_id", String(64), nullable=False),
    Column("branch_name", String(128), nullable=False),
    Column("token_id", String(64), nullable=False),
    Column("reason", String(64), nullable=False),  # failed / quarantined / error_routed / ...
    Column("recorded_by", String(128), nullable=False),  # worker_id
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("adopted_epoch", Integer),  # NULL = not yet replayed into leader memory
)
# Natural-key idempotency (design §G: record_coalesce_branch_loss is
# "idempotent on the natural key"). Named unique Index rather than a
# UniqueConstraint for _REQUIRED_INDEXES verifiability; doubles as the hot
# lookup index (replay scans by run_id + coalesce_name prefix).
Index(
    "uq_coalesce_branch_losses_natural",
    coalesce_branch_losses_table.c.run_id,
    coalesce_branch_losses_table.c.coalesce_name,
    coalesce_branch_losses_table.c.row_id,
    coalesce_branch_losses_table.c.branch_name,
    unique=True,
)


def active_worker_fence_clause(*, worker_id: ColumnElement[str] | str, run_id: ColumnElement[str] | str) -> ColumnElement[bool]:
    """Membership fence: the acting worker holds an *active* run_workers row.

    Single source of truth for the EXISTS predicate that slice 4 compiles
    into enqueue_ready's INSERT guard (design §G verb table). Strict variant:
    ABSENT worker → False (the caller explicitly supplied worker_id so absence
    is a definite membership failure). Single-use identity doctrine:
    'departed'/'evicted' rows never return to 'active', so a False result is
    a permanent fence. The literal 'active' MUST match the worker-status
    CHECK on run_workers.

    For claim_ready / claim_pending_sink CAS UPDATEs use
    ``claim_verb_fence_clause`` which adds the backward-compat OR-branch that
    passes when the run has no registered workers at all (N=0 unit-test mode).
    """
    return (
        select(run_workers_table.c.worker_id)
        .where(
            run_workers_table.c.worker_id == worker_id,
            run_workers_table.c.run_id == run_id,
            run_workers_table.c.status == "active",
        )
        .exists()
    )


def claim_verb_fence_clause(*, worker_id: ColumnElement[str] | str, run_id: ColumnElement[str] | str) -> ColumnElement[bool]:
    """Lenient membership fence for claim_ready / claim_pending_sink CAS UPDATEs.

    Passes when either:

    * the acting worker holds an ACTIVE run_workers row for this run, or
    * the run has no registered workers at all (N=0 unit-test compatibility).

    Semantics by case:
    (a) ABSENT worker and N=0 run registry: no registered workers exist for the
        run → pass. Backward-compat for unit tests that call claim_ready with
        fictional lease_owner IDs without populating run_workers.
    (b) ABSENT worker and N>0 run registry: registered workers exist for the run
        but this caller is not one of them → fail closed with zero mutation.
    (c) ACTIVE run_workers row for this worker and run → pass.
    (d) EVICTED or DEPARTED row for this worker and run → fail; the claim verb
        re-probe raises RunWorkerEvictedError.

    This is strictly weaker than ``active_worker_fence_clause`` (strict):
    ABSENT passes here only when the run has no registry at all, and is refused
    once any worker has registered for the run. Use the strict variant for
    ``enqueue_ready``'s explicit-worker_id guard where ABSENT is always wrong
    (caller supplied worker_id, registration is mandatory).
    """
    run_has_no_registered_workers = ~select(run_workers_table.c.worker_id).where(run_workers_table.c.run_id == run_id).exists()
    return or_(
        active_worker_fence_clause(worker_id=worker_id, run_id=run_id),
        run_has_no_registered_workers,
    )


# === Token Parents (for multi-parent joins) ===

token_parents_table = Table(
    "token_parents",
    metadata,
    Column("token_id", String(64), ForeignKey("tokens.token_id"), primary_key=True),
    Column(
        "parent_token_id",
        String(64),
        ForeignKey("tokens.token_id"),
        primary_key=True,
    ),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("token_id", "ordinal"),
)

# === Node States ===

node_states_table = Table(
    "node_states",
    metadata,
    Column("state_id", String(64), primary_key=True),
    Column("token_id", String(64), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("step_index", Integer, nullable=False),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("output_hash", String(64)),
    Column("context_before_json", Text),
    Column("context_after_json", Text),
    Column("duration_ms", Float),
    Column("error_json", Text),
    Column("success_reason_json", Text),  # TransformSuccessReason for successful transforms
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    # Resume provenance marker (epoch 11): NULL for every node_state written during the
    # original run; set to the resumed-from checkpoint id for every node_state written
    # while re-driving a reconstructed incomplete token on resume. Makes a resume re-drive
    # (which records at attempt = max+1 under the SAME run_id) provably distinguishable
    # from a run-1 tenacity retry — explain() filters on resume_checkpoint_id IS NULL.
    #
    # MARKER-ONLY (no FK): the id is a durable provenance fact, like a content hash — it
    # endures even after its checkpoint row is purged. Checkpoints are deletable progress
    # state (delete_checkpoints clears them unconditionally on successful completion); the
    # marker on node_states does NOT keep them alive and carries NO referential constraint
    # to the checkpoints table. explain() distinguishes resume re-drives from run-1 retries
    # purely by this column's NULL-ness (resume_checkpoint_id IS NOT NULL), which survives
    # checkpoint purge. (Operator decision 2026-05-30, faithful to "hashes survive payload
    # deletion".)
    Column("resume_checkpoint_id", String(64), nullable=True),
    # Composite unique target for run-scoped FKs to node_states.
    UniqueConstraint("state_id", "run_id"),
    UniqueConstraint("token_id", "node_id", "attempt"),
    UniqueConstraint("token_id", "step_index", "attempt"),
    # Composite FK: enforces token_id and run_id belong together (prevents cross-run contamination)
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# === Recoverable sink effects (epoch 26) ===

sink_effect_streams_table = Table(
    "sink_effect_streams",
    metadata,
    Column("stream_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False),
    Column("sink_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("role", String(16), nullable=False),
    Column("requested_target_hash", String(64), nullable=False),
    Column("resolved_target", String(512)),
    Column("next_sequence", Integer, nullable=False),
    Column("tail_effect_id", String(64)),
    Column("head_effect_id", String(64)),
    Column("head_descriptor_hash", String(64)),
    UniqueConstraint("stream_id", "run_id"),
    CheckConstraint("role IN ('primary','failsink')", name="ck_sink_effect_streams_role"),
    CheckConstraint("next_sequence >= 0", name="ck_sink_effect_streams_next_sequence"),
    ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
Index(
    "uq_sink_effect_stream_identity",
    sink_effect_streams_table.c.run_id,
    sink_effect_streams_table.c.sink_node_id,
    sink_effect_streams_table.c.role,
    sink_effect_streams_table.c.requested_target_hash,
    unique=True,
)

audit_export_snapshots_table = Table(
    "audit_export_snapshots",
    metadata,
    Column("snapshot_id", String(64), primary_key=True),
    Column("source_run_id", String(64), nullable=False),
    Column("source_status", String(32), nullable=False),
    Column("source_completed_at", DateTime(timezone=True), nullable=False),
    Column("exported_at", DateTime(timezone=True), nullable=False),
    Column("registry_key_hash", String(64), nullable=False),
    Column("exporter_version", String(64), nullable=False),
    Column("serialization_version", String(64), nullable=False),
    Column("export_format", String(16), nullable=False),
    Column("signing_mode", String(16), nullable=False),
    Column("signer_key_id", String(128), nullable=False),
    Column("derivation_version", String(64), nullable=False),
    Column("public_export_config_hash", String(64), nullable=False),
    Column("chunking_algorithm_version", String(64), nullable=False),
    Column("per_chunk_record_limit", Integer, nullable=False),
    Column("per_chunk_byte_limit", Integer, nullable=False),
    Column("record_count", Integer, nullable=False),
    Column("total_bytes", Integer, nullable=False),
    Column("chunk_count", Integer, nullable=False),
    Column("terminal_chunk_ordinal", Integer, nullable=False),
    Column("content_store_id", String(128), nullable=False),
    Column("manifest_hash", String(64), nullable=False),
    Column("last_chunk_seal_hash", String(64), nullable=False),
    Column("snapshot_hash", String(64), nullable=False),
    Column("snapshot_seal_hash", String(64), nullable=False),
    Column("signature_hex", String(64)),
    Column("record_chain_algorithm", String(64), nullable=False),
    Column("final_hash", String(64), nullable=False),
    Column("signed_manifest_schema", String(64), nullable=False),
    Column("signed_manifest_hash", String(64), nullable=False),
    Column("signed_manifest_ref", String(71), nullable=False),
    Column("signed_manifest_size_bytes", Integer, nullable=False),
    UniqueConstraint("snapshot_id", "source_run_id"),
    CheckConstraint(
        "source_status IN ('completed','completed_with_failures','empty') AND source_completed_at IS NOT NULL",
        name="ck_audit_export_snapshots_terminal_witness",
    ),
    CheckConstraint("record_count > 0 AND total_bytes > 0 AND chunk_count > 0", name="ck_audit_export_snapshots_positive_totals"),
    CheckConstraint("terminal_chunk_ordinal = chunk_count - 1", name="ck_audit_export_snapshots_terminal_ordinal"),
    CheckConstraint(_LowerHex64Check("manifest_hash"), name="ck_audit_export_snapshots_manifest_hash_hex"),
    CheckConstraint(_LowerHex64Check("snapshot_hash"), name="ck_audit_export_snapshots_snapshot_hash_hex"),
    CheckConstraint(_LowerHex64Check("snapshot_seal_hash"), name="ck_audit_export_snapshots_snapshot_seal_hash_hex"),
    CheckConstraint(_LowerHex64Check("last_chunk_seal_hash"), name="ck_audit_export_snapshots_last_chunk_seal_hash_hex"),
    CheckConstraint(_LowerHex64Check("final_hash"), name="ck_audit_export_snapshots_final_hash_hex"),
    CheckConstraint(_LowerHex64Check("signed_manifest_hash"), name="ck_audit_export_snapshots_signed_manifest_hash_hex"),
    CheckConstraint(
        "signed_manifest_ref = 'sha256:' || signed_manifest_hash",
        name="ck_audit_export_snapshots_signed_manifest_ref",
    ),
    CheckConstraint(
        "signed_manifest_size_bytes BETWEEN 1 AND 65536",
        name="ck_audit_export_snapshots_signed_manifest_size",
    ),
    CheckConstraint(
        "signed_manifest_schema = 'elspeth.audit-export-manifest.v2'",
        name="ck_audit_export_snapshots_manifest_schema",
    ),
    CheckConstraint(
        "derivation_version = 'audit-export-derivation-v1'",
        name="ck_audit_export_snapshots_derivation_version",
    ),
    CheckConstraint(_SigningTupleCheck(), name="ck_audit_export_snapshots_signing_tuple"),
    ForeignKeyConstraint(
        ["source_run_id", "source_status", "source_completed_at"],
        ["runs.run_id", "runs.status", "runs.completed_at"],
    ),
    ForeignKeyConstraint(
        ["snapshot_id", "terminal_chunk_ordinal", "last_chunk_seal_hash", "record_count", "total_bytes"],
        [
            "audit_export_snapshot_chunks.snapshot_id",
            "audit_export_snapshot_chunks.ordinal",
            "audit_export_snapshot_chunks.chunk_seal_hash",
            "audit_export_snapshot_chunks.cumulative_records",
            "audit_export_snapshot_chunks.cumulative_bytes",
        ],
        deferrable=True,
        initially="DEFERRED",
    ),
)
Index(
    "uq_audit_export_snapshots_registry_key",
    audit_export_snapshots_table.c.source_run_id,
    audit_export_snapshots_table.c.exporter_version,
    audit_export_snapshots_table.c.serialization_version,
    audit_export_snapshots_table.c.export_format,
    audit_export_snapshots_table.c.signing_mode,
    audit_export_snapshots_table.c.signer_key_id,
    audit_export_snapshots_table.c.public_export_config_hash,
    unique=True,
)
Index("ix_audit_export_snapshots_registry_key_hash", audit_export_snapshots_table.c.registry_key_hash, unique=True)

audit_export_snapshot_chunks_table = Table(
    "audit_export_snapshot_chunks",
    metadata,
    Column("snapshot_id", String(64), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("content_ref", String(71), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("record_count", Integer, nullable=False),
    Column("predecessor_seal_hash", String(64)),
    Column("cumulative_records", Integer, nullable=False),
    Column("cumulative_bytes", Integer, nullable=False),
    Column("chunk_seal_hash", String(64), nullable=False),
    PrimaryKeyConstraint("snapshot_id", "ordinal"),
    CheckConstraint("ordinal >= 0", name="ck_audit_export_snapshot_chunks_ordinal"),
    CheckConstraint("size_bytes > 0 AND size_bytes <= 67108864", name="ck_audit_export_snapshot_chunks_size"),
    CheckConstraint("record_count > 0 AND record_count <= 1000000", name="ck_audit_export_snapshot_chunks_records"),
    CheckConstraint("cumulative_records > 0 AND cumulative_bytes > 0", name="ck_audit_export_snapshot_chunks_cumulative"),
    CheckConstraint(_LowerHex64Check("content_hash"), name="ck_audit_export_snapshot_chunks_content_hash_hex"),
    CheckConstraint(_LowerHex64Check("chunk_seal_hash"), name="ck_audit_export_snapshot_chunks_seal_hash_hex"),
    CheckConstraint(_OptionalLowerHex64Check("predecessor_seal_hash"), name="ck_audit_export_snapshot_chunks_predecessor_hash"),
    CheckConstraint("content_ref = 'sha256:' || content_hash", name="ck_audit_export_snapshot_chunks_content_ref"),
    ForeignKeyConstraint(["snapshot_id"], ["audit_export_snapshots.snapshot_id"], deferrable=True, initially="DEFERRED"),
)
Index(
    "uq_audit_export_snapshot_chunks_terminal",
    audit_export_snapshot_chunks_table.c.snapshot_id,
    audit_export_snapshot_chunks_table.c.ordinal,
    audit_export_snapshot_chunks_table.c.chunk_seal_hash,
    audit_export_snapshot_chunks_table.c.cumulative_records,
    audit_export_snapshot_chunks_table.c.cumulative_bytes,
    unique=True,
)

sink_effects_table = Table(
    "sink_effects",
    metadata,
    Column("effect_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False),
    Column("sink_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("role", String(16), nullable=False),
    Column("state", String(16), nullable=False),
    Column("protocol_version", String(64), nullable=False),
    Column("input_kind", String(32), nullable=False),
    Column("required_member_ordinal", Integer),
    Column("required_snapshot_slot", Integer),
    Column("config_hash", String(64), nullable=False),
    Column("membership_or_manifest_hash", String(64), nullable=False),
    Column("group_payload_hash", String(64), nullable=False),
    Column("artifact_id", String(64), nullable=False),
    Column("artifact_idempotency_key", String(256), nullable=False),
    Column("target_json", Text, nullable=False),
    Column("inspection_mode", String(32)),
    Column("inspection_attempt_id", String(64)),
    Column("plan_json", Text),
    Column("plan_hash", String(64)),
    Column("descriptor_mode", String(32)),
    Column("expected_descriptor_hash", String(64)),
    Column("precondition_hash", String(64)),
    Column("prepared_at", DateTime(timezone=True)),
    Column("lease_owner", String(128)),
    Column("generation", Integer, nullable=False),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("lease_heartbeat_at", DateTime(timezone=True)),
    Column("reconcile_kind", String(48)),
    Column("reconcile_evidence_hash", String(64)),
    Column("result_descriptor_hash", String(64)),
    Column("publication_performed", Boolean),
    Column("publication_evidence_kind", String(32)),
    Column("primary_effect_id", String(64)),
    Column("stream_id", String(64)),
    Column("stream_sequence", Integer),
    Column("predecessor_effect_id", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finalized_at", DateTime(timezone=True)),
    UniqueConstraint("effect_id", "input_kind"),
    UniqueConstraint("effect_id", "run_id", "sink_node_id"),
    UniqueConstraint("effect_id", "stream_id"),
    CheckConstraint("role IN ('primary','failsink')", name="ck_sink_effects_role"),
    CheckConstraint("state IN ('reserved','prepared','in_flight','finalized')", name="ck_sink_effects_state"),
    CheckConstraint(
        "(input_kind = 'pipeline_members' AND required_member_ordinal = 0 AND required_snapshot_slot IS NULL) OR "
        "(input_kind = 'audit_export_snapshot' AND required_member_ordinal IS NULL AND required_snapshot_slot = 0)",
        name="ck_sink_effects_input_kind_xor",
    ),
    CheckConstraint(
        "(state = 'reserved' AND generation = 0 AND inspection_mode IS NULL AND inspection_attempt_id IS NULL "
        "AND plan_json IS NULL AND plan_hash IS NULL AND descriptor_mode IS NULL AND expected_descriptor_hash IS NULL "
        "AND precondition_hash IS NULL AND prepared_at IS NULL AND lease_owner IS NULL AND lease_expires_at IS NULL "
        "AND lease_heartbeat_at IS NULL AND reconcile_kind IS NULL AND reconcile_evidence_hash IS NULL "
        "AND result_descriptor_hash IS NULL AND publication_performed IS NULL AND publication_evidence_kind IS NULL "
        "AND finalized_at IS NULL) OR "
        "(state = 'prepared' AND generation = 0 AND plan_hash IS NOT NULL AND plan_json IS NOT NULL "
        "AND inspection_mode IS NOT NULL AND inspection_attempt_id IS NOT NULL AND descriptor_mode IS NOT NULL "
        "AND precondition_hash IS NOT NULL AND prepared_at IS NOT NULL AND lease_owner IS NULL "
        "AND lease_expires_at IS NULL AND lease_heartbeat_at IS NULL AND reconcile_kind IS NULL "
        "AND reconcile_evidence_hash IS NULL AND result_descriptor_hash IS NULL AND publication_performed IS NULL "
        "AND publication_evidence_kind IS NULL AND finalized_at IS NULL) OR "
        "(state = 'in_flight' AND plan_hash IS NOT NULL AND plan_json IS NOT NULL AND inspection_mode IS NOT NULL "
        "AND inspection_attempt_id IS NOT NULL AND descriptor_mode IS NOT NULL AND precondition_hash IS NOT NULL "
        "AND prepared_at IS NOT NULL AND lease_owner IS NOT NULL AND generation > 0 AND lease_expires_at IS NOT NULL "
        "AND lease_heartbeat_at IS NOT NULL AND reconcile_kind IS NULL AND reconcile_evidence_hash IS NULL "
        "AND result_descriptor_hash IS NULL AND publication_performed IS NULL AND publication_evidence_kind IS NULL "
        "AND finalized_at IS NULL) OR "
        "(state = 'finalized' AND plan_hash IS NOT NULL AND plan_json IS NOT NULL AND inspection_mode IS NOT NULL "
        "AND inspection_attempt_id IS NOT NULL AND descriptor_mode IS NOT NULL AND precondition_hash IS NOT NULL "
        "AND prepared_at IS NOT NULL AND result_descriptor_hash IS NOT NULL AND publication_performed IS NOT NULL "
        "AND publication_evidence_kind IS NOT NULL AND lease_owner IS NULL AND lease_expires_at IS NULL "
        "AND lease_heartbeat_at IS NULL AND finalized_at IS NOT NULL AND "
        "((publication_performed IS TRUE AND publication_evidence_kind = 'returned' AND reconcile_kind IS NULL "
        "AND reconcile_evidence_hash IS NULL) OR "
        "(publication_performed IS TRUE AND publication_evidence_kind = 'reconciled' "
        "AND reconcile_kind = 'applied_with_exact_descriptor' AND reconcile_evidence_hash IS NOT NULL) OR "
        "(publication_performed IS FALSE AND publication_evidence_kind IN ('inherited','virtual') "
        "AND reconcile_kind IS NULL AND reconcile_evidence_hash IS NULL)))",
        name="ck_sink_effects_lifecycle",
    ),
    CheckConstraint("generation >= 0", name="ck_sink_effects_generation"),
    CheckConstraint(
        "lease_expires_at IS NULL OR lease_heartbeat_at IS NULL OR lease_expires_at >= lease_heartbeat_at",
        name="ck_sink_effects_lease_window",
    ),
    CheckConstraint(
        "(stream_id IS NULL AND stream_sequence IS NULL AND predecessor_effect_id IS NULL) OR "
        "(stream_id IS NOT NULL AND stream_sequence = 0 AND predecessor_effect_id IS NULL) OR "
        "(stream_id IS NOT NULL AND stream_sequence > 0 AND predecessor_effect_id IS NOT NULL)",
        name="ck_sink_effects_stream_shape",
    ),
    CheckConstraint(
        "descriptor_mode IS NULL OR "
        "(descriptor_mode = 'precomputed' AND expected_descriptor_hash IS NOT NULL) OR "
        "(descriptor_mode = 'result_derived' AND expected_descriptor_hash IS NULL) OR "
        "(descriptor_mode = 'no_publication' AND expected_descriptor_hash IS NOT NULL)",
        name="ck_sink_effects_descriptor_mode",
    ),
    CheckConstraint(
        "inspection_mode IS NULL OR inspection_mode IN ('inspected','no_inspection_required')",
        name="ck_sink_effects_inspection_mode",
    ),
    CheckConstraint(
        "reconcile_kind IS NULL OR reconcile_kind IN ('not_applied','applied_with_exact_descriptor','unknown')",
        name="ck_sink_effects_reconcile_kind",
    ),
    ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    ForeignKeyConstraint(["primary_effect_id"], ["sink_effects.effect_id"]),
    ForeignKeyConstraint(["stream_id", "run_id"], ["sink_effect_streams.stream_id", "sink_effect_streams.run_id"]),
    ForeignKeyConstraint(["predecessor_effect_id", "stream_id"], ["sink_effects.effect_id", "sink_effects.stream_id"]),
    ForeignKeyConstraint(
        ["effect_id", "input_kind", "required_member_ordinal"],
        ["sink_effect_members.effect_id", "sink_effect_members.input_kind", "sink_effect_members.ordinal"],
        deferrable=True,
        initially="DEFERRED",
    ),
    ForeignKeyConstraint(
        ["effect_id", "input_kind", "required_snapshot_slot"],
        [
            "sink_effect_export_snapshots.effect_id",
            "sink_effect_export_snapshots.input_kind",
            "sink_effect_export_snapshots.slot",
        ],
        deferrable=True,
        initially="DEFERRED",
    ),
)
Index("ix_sink_effects_stream_sequence", sink_effects_table.c.stream_id, sink_effects_table.c.stream_sequence, unique=True)
Index("ix_sink_effects_run_state", sink_effects_table.c.run_id, sink_effects_table.c.state)

sink_effect_members_table = Table(
    "sink_effect_members",
    metadata,
    Column("effect_id", String(64), nullable=False),
    Column("input_kind", String(32), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("sink_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("role", String(16), nullable=False),
    Column("token_id", String(64), nullable=False),
    Column("row_id", String(64), nullable=False),
    Column("ingest_sequence", Integer, nullable=False),
    Column("lineage_json", Text, nullable=False),
    Column("lineage_hash", String(64), nullable=False),
    Column("payload_hash", String(64), nullable=False),
    Column("prepared_disposition", String(16)),
    Column("reason_hash", String(64)),
    Column("member_effect_id", String(64)),
    Column("member_state", String(16)),
    Column("descriptor_hash", String(64)),
    Column("evidence_hash", String(64)),
    PrimaryKeyConstraint("effect_id", "ordinal"),
    UniqueConstraint("effect_id", "input_kind", "ordinal"),
    CheckConstraint("input_kind = 'pipeline_members'", name="ck_sink_effect_members_input_kind"),
    CheckConstraint("ordinal >= 0 AND ingest_sequence >= 0", name="ck_sink_effect_members_order"),
    CheckConstraint(
        "prepared_disposition IS NULL OR prepared_disposition IN ('accepted','diverted')",
        name="ck_sink_effect_members_disposition",
    ),
    CheckConstraint(
        "member_state IS NULL OR member_state IN ('reserved','prepared','in_flight','finalized')",
        name="ck_sink_effect_members_state",
    ),
    ForeignKeyConstraint(["effect_id", "input_kind"], ["sink_effects.effect_id", "sink_effects.input_kind"]),
    ForeignKeyConstraint(["token_id", "row_id", "run_id"], ["tokens.token_id", "tokens.row_id", "tokens.run_id"]),
    ForeignKeyConstraint(["row_id", "run_id"], ["rows.row_id", "rows.run_id"]),
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)
Index(
    "uq_sink_effect_member_binding",
    sink_effect_members_table.c.run_id,
    sink_effect_members_table.c.sink_node_id,
    sink_effect_members_table.c.role,
    sink_effect_members_table.c.token_id,
    unique=True,
)

sink_effect_export_snapshots_table = Table(
    "sink_effect_export_snapshots",
    metadata,
    Column("effect_id", String(64), nullable=False),
    Column("input_kind", String(32), nullable=False),
    Column("slot", Integer, nullable=False),
    Column("snapshot_id", String(64), nullable=False),
    PrimaryKeyConstraint("effect_id", "slot"),
    UniqueConstraint("effect_id", "input_kind", "slot"),
    UniqueConstraint("effect_id", "snapshot_id"),
    CheckConstraint("input_kind = 'audit_export_snapshot'", name="ck_sink_effect_export_snapshots_input_kind"),
    CheckConstraint("slot = 0", name="ck_sink_effect_export_snapshots_slot"),
    ForeignKeyConstraint(["effect_id", "input_kind"], ["sink_effects.effect_id", "sink_effects.input_kind"]),
    ForeignKeyConstraint(["snapshot_id"], ["audit_export_snapshots.snapshot_id"]),
)

sink_effect_attempts_table = Table(
    "sink_effect_attempts",
    metadata,
    Column("attempt_id", String(64), primary_key=True),
    Column("effect_id", String(64), nullable=False),
    Column("member_ordinal", Integer),
    Column("generation", Integer, nullable=False),
    Column("action", String(16), nullable=False),
    Column("call_kind", String(64), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("state", String(32), nullable=False),
    Column("evidence_json", Text),
    Column("evidence_hash", String(64)),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("latency_ms", Float),
    CheckConstraint("generation >= 0", name="ck_sink_effect_attempts_generation"),
    CheckConstraint("action IN ('inspect','commit','reconcile')", name="ck_sink_effect_attempts_action"),
    CheckConstraint("state IN ('intent','returned','response_lost','error')", name="ck_sink_effect_attempts_state"),
    ForeignKeyConstraint(["effect_id"], ["sink_effects.effect_id"]),
    ForeignKeyConstraint(["effect_id", "member_ordinal"], ["sink_effect_members.effect_id", "sink_effect_members.ordinal"]),
)
Index("ix_sink_effect_attempts_effect", sink_effect_attempts_table.c.effect_id, sink_effect_attempts_table.c.started_at)

sink_effect_streams_table.append_constraint(
    ForeignKeyConstraint(["tail_effect_id", "stream_id"], ["sink_effects.effect_id", "sink_effects.stream_id"])
)
sink_effect_streams_table.append_constraint(
    ForeignKeyConstraint(["head_effect_id", "stream_id"], ["sink_effects.effect_id", "sink_effects.stream_id"])
)

_SQLITE_AUDIT_EXPORT_TRIGGERS: tuple[str, ...] = (
    """
    CREATE TRIGGER trg_audit_export_chunk_insert_validate
    BEFORE INSERT ON audit_export_snapshot_chunks
    BEGIN
      SELECT CASE WHEN NEW.content_ref <> 'sha256:' || NEW.content_hash
        THEN RAISE(ABORT, 'audit export chunk content ref/hash mismatch') END;
      SELECT CASE WHEN EXISTS (
        SELECT 1 FROM audit_export_snapshots WHERE snapshot_id = NEW.snapshot_id
      ) THEN RAISE(ABORT, 'sealed audit export snapshot cannot accept chunks') END;
      SELECT CASE WHEN NEW.ordinal = 0 AND (
        NEW.predecessor_seal_hash IS NOT NULL OR
        NEW.cumulative_records <> NEW.record_count OR
        NEW.cumulative_bytes <> NEW.size_bytes
      ) THEN RAISE(ABORT, 'invalid audit export genesis chunk') END;
      SELECT CASE WHEN NEW.ordinal > 0 AND NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks AS prior
        WHERE prior.snapshot_id = NEW.snapshot_id
          AND prior.ordinal = NEW.ordinal - 1
          AND NEW.predecessor_seal_hash = prior.chunk_seal_hash
          AND NEW.cumulative_records = prior.cumulative_records + NEW.record_count
          AND NEW.cumulative_bytes = prior.cumulative_bytes + NEW.size_bytes
      ) THEN RAISE(ABORT, 'invalid audit export chunk predecessor/totals') END;
    END
    """,
    """
    CREATE TRIGGER trg_audit_export_snapshot_insert_seal
    BEFORE INSERT ON audit_export_snapshots
    BEGIN
      SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks
        WHERE snapshot_id = NEW.snapshot_id
        GROUP BY snapshot_id
        HAVING COUNT(*) = NEW.chunk_count
           AND MIN(ordinal) = 0
           AND MAX(ordinal) = NEW.chunk_count - 1
           AND SUM(record_count) = NEW.record_count
           AND SUM(size_bytes) = NEW.total_bytes
      ) THEN RAISE(ABORT, 'audit export snapshot chunk graph is incomplete') END;
      SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks
        WHERE snapshot_id = NEW.snapshot_id
          AND ordinal = NEW.terminal_chunk_ordinal
          AND chunk_seal_hash = NEW.last_chunk_seal_hash
          AND cumulative_records = NEW.record_count
          AND cumulative_bytes = NEW.total_bytes
      ) THEN RAISE(ABORT, 'audit export terminal descriptor mismatch') END;
    END
    """,
    """
    CREATE TRIGGER trg_audit_export_snapshot_immutable
    BEFORE UPDATE ON audit_export_snapshots
    BEGIN SELECT RAISE(ABORT, 'sealed audit export snapshot is immutable'); END
    """,
    """
    CREATE TRIGGER trg_audit_export_snapshot_immutable_delete
    BEFORE DELETE ON audit_export_snapshots
    BEGIN SELECT RAISE(ABORT, 'sealed audit export snapshot is immutable'); END
    """,
    """
    CREATE TRIGGER trg_audit_export_chunk_immutable
    BEFORE UPDATE ON audit_export_snapshot_chunks
    WHEN EXISTS (SELECT 1 FROM audit_export_snapshots WHERE snapshot_id = OLD.snapshot_id)
    BEGIN SELECT RAISE(ABORT, 'sealed audit export chunk is immutable'); END
    """,
    """
    CREATE TRIGGER trg_audit_export_chunk_immutable_delete
    BEFORE DELETE ON audit_export_snapshot_chunks
    WHEN EXISTS (SELECT 1 FROM audit_export_snapshots WHERE snapshot_id = OLD.snapshot_id)
    BEGIN SELECT RAISE(ABORT, 'sealed audit export chunk is immutable'); END
    """,
)
for _trigger_ddl in _SQLITE_AUDIT_EXPORT_TRIGGERS:
    event.listen(
        audit_export_snapshot_chunks_table,
        "after_create",
        DDL(_trigger_ddl).execute_if(dialect="sqlite"),  # type: ignore[no-untyped-call]
    )

_POSTGRES_AUDIT_EXPORT_TRIGGER_DDL: tuple[str, ...] = (
    """
    CREATE FUNCTION fn_audit_export_chunk_insert_validate() RETURNS trigger AS $$
    BEGIN
      IF NEW.content_ref <> 'sha256:' || NEW.content_hash THEN
        RAISE EXCEPTION 'audit export chunk content ref/hash mismatch';
      END IF;
      IF EXISTS (SELECT 1 FROM audit_export_snapshots WHERE snapshot_id=NEW.snapshot_id) THEN
        RAISE EXCEPTION 'sealed audit export snapshot cannot accept chunks';
      END IF;
      IF NEW.ordinal=0 THEN
        IF NEW.predecessor_seal_hash IS NOT NULL OR NEW.cumulative_records<>NEW.record_count
           OR NEW.cumulative_bytes<>NEW.size_bytes THEN
          RAISE EXCEPTION 'invalid audit export genesis chunk';
        END IF;
      ELSIF NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks prior
        WHERE prior.snapshot_id=NEW.snapshot_id AND prior.ordinal=NEW.ordinal-1
          AND NEW.predecessor_seal_hash=prior.chunk_seal_hash
          AND NEW.cumulative_records=prior.cumulative_records+NEW.record_count
          AND NEW.cumulative_bytes=prior.cumulative_bytes+NEW.size_bytes
      ) THEN RAISE EXCEPTION 'invalid audit export chunk predecessor/totals';
      END IF;
      RETURN NEW;
    END; $$ LANGUAGE plpgsql
    """,
    """
    CREATE TRIGGER trg_audit_export_chunk_insert_validate
    BEFORE INSERT ON audit_export_snapshot_chunks
    FOR EACH ROW EXECUTE FUNCTION fn_audit_export_chunk_insert_validate()
    """,
    """
    CREATE FUNCTION fn_audit_export_snapshot_insert_seal() RETURNS trigger AS $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks WHERE snapshot_id=NEW.snapshot_id
        GROUP BY snapshot_id HAVING COUNT(*)=NEW.chunk_count AND MIN(ordinal)=0
          AND MAX(ordinal)=NEW.chunk_count-1 AND SUM(record_count)=NEW.record_count
          AND SUM(size_bytes)=NEW.total_bytes
      ) THEN RAISE EXCEPTION 'audit export snapshot chunk graph is incomplete'; END IF;
      IF NOT EXISTS (
        SELECT 1 FROM audit_export_snapshot_chunks WHERE snapshot_id=NEW.snapshot_id
          AND ordinal=NEW.terminal_chunk_ordinal AND chunk_seal_hash=NEW.last_chunk_seal_hash
          AND cumulative_records=NEW.record_count AND cumulative_bytes=NEW.total_bytes
      ) THEN RAISE EXCEPTION 'audit export terminal descriptor mismatch'; END IF;
      RETURN NEW;
    END; $$ LANGUAGE plpgsql
    """,
    """
    CREATE TRIGGER trg_audit_export_snapshot_insert_seal
    BEFORE INSERT ON audit_export_snapshots
    FOR EACH ROW EXECUTE FUNCTION fn_audit_export_snapshot_insert_seal()
    """,
    """
    CREATE FUNCTION fn_audit_export_snapshot_immutable() RETURNS trigger AS $$
    BEGIN RAISE EXCEPTION 'sealed audit export snapshot is immutable'; END; $$ LANGUAGE plpgsql
    """,
    """
    CREATE TRIGGER trg_audit_export_snapshot_immutable
    BEFORE UPDATE OR DELETE ON audit_export_snapshots
    FOR EACH ROW EXECUTE FUNCTION fn_audit_export_snapshot_immutable()
    """,
    """
    CREATE FUNCTION fn_audit_export_chunk_immutable() RETURNS trigger AS $$
    BEGIN
      IF EXISTS (SELECT 1 FROM audit_export_snapshots WHERE snapshot_id=OLD.snapshot_id) THEN
        RAISE EXCEPTION 'sealed audit export chunk is immutable';
      END IF;
      RETURN OLD;
    END; $$ LANGUAGE plpgsql
    """,
    """
    CREATE TRIGGER trg_audit_export_chunk_immutable
    BEFORE UPDATE OR DELETE ON audit_export_snapshot_chunks
    FOR EACH ROW EXECUTE FUNCTION fn_audit_export_chunk_immutable()
    """,
)
for _trigger_ddl in _POSTGRES_AUDIT_EXPORT_TRIGGER_DDL:
    event.listen(
        audit_export_snapshot_chunks_table,
        "after_create",
        DDL(_trigger_ddl).execute_if(dialect="postgresql"),  # type: ignore[no-untyped-call]
    )

# === Operations (Source/Sink I/O) ===
# Operations are the source/sink equivalent of node_states - they provide
# a parent context for external calls made during source.load() or sink.write().
# Unlike node_states (which require a token_id), operations exist at the
# run/node level because sources CREATE tokens rather than processing them.

operations_table = Table(
    "operations",
    metadata,
    Column("operation_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("operation_type", String(32), nullable=False),  # 'source_load' | 'sink_write' | 'runtime_preflight'
    Column("sink_effect_id", String(64), ForeignKey("sink_effects.effect_id"), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),  # 'open' | 'completed' | 'failed' | 'pending'
    Column("input_data_ref", String(256)),  # Payload store reference for operation input
    Column("input_data_hash", String(64)),  # SHA-256 of canonical JSON (survives purge)
    Column("output_data_ref", String(256)),  # Payload store reference for operation output
    Column("output_data_hash", String(64)),  # SHA-256 of canonical JSON (survives purge)
    Column("error_message", Text),  # Error details if failed
    Column("duration_ms", Float),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    CheckConstraint(
        "sink_effect_id IS NULL OR operation_type = 'sink_write'",
        name="ck_operations_sink_effect_type",
    ),
)
Index("uq_operations_sink_effect_id", operations_table.c.sink_effect_id, unique=True)

# === External Calls ===
# Calls can be parented by either a node_state (transform processing) or an
# operation (source/sink I/O). Exactly one parent must be set (XOR constraint).

calls_table = Table(
    "calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=True),  # NULL for operation calls
    Column("operation_id", String(64), ForeignKey("operations.operation_id"), nullable=True),  # NULL for state calls
    Column("call_index", Integer, nullable=False),
    Column("call_type", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("request_ref", String(256)),
    Column("response_hash", String(64)),
    Column("response_ref", String(256)),
    # Cross-DB hash anchor for interpretation events (Option A — Phase 5b).
    # Populated by the LLM-transform plugin at execution time when the runtime
    # node config contains a ``resolved_prompt_template_hash`` sibling field
    # (written by ``resolve_interpretation_event`` at compose time and committed
    # into ``composition_states.nodes``). If the sibling field is absent (the
    # LLM transform is NOT downstream of an interpretation event), this column
    # is NULL.
    #
    # When non-NULL, this hash MUST equal the corresponding
    # ``interpretation_events.resolved_prompt_template_hash`` in the session
    # audit DB for the same resolved prompt string. An inequality indicates
    # tampering or a composition-to-execution coherence failure. Checked by the
    # audit-tooling layer; a mismatch is a Tier-1 crash-on-anomaly.
    #
    # Hash scheme: SHA-256 over rfc8785 canonical JSON of the resolved
    # prompt-template string, using ``CANONICAL_VERSION = "sha256-rfc8785-v1"``
    # (contracts/hashing.py:CANONICAL_VERSION). Identical scheme used by both
    # the session service (write at resolve time) and the runtime plugin (write
    # at execution time), so the hashes are comparable byte-for-byte.
    Column("resolved_prompt_template_hash", String(64), nullable=True),
    Column("error_json", Text),
    Column("latency_ms", Float),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # XOR constraint: exactly one parent (state OR operation)
    CheckConstraint(
        "(state_id IS NOT NULL AND operation_id IS NULL) OR (state_id IS NULL AND operation_id IS NOT NULL)",
        name="calls_has_parent",
    ),
)

# Partial unique indexes for call_index uniqueness within each parent type.
# Since calls can be parented by EITHER state_id OR operation_id (XOR),
# we need separate uniqueness constraints for each parent type.
# This preserves the original UNIQUE(state_id, call_index) semantics while
# also enforcing UNIQUE(operation_id, call_index) for operation calls.
Index(
    "ix_calls_state_call_index_unique",
    calls_table.c.state_id,
    calls_table.c.call_index,
    unique=True,
    sqlite_where=(calls_table.c.state_id.isnot(None)),
    postgresql_where=(calls_table.c.state_id.isnot(None)),
)

Index(
    "ix_calls_operation_call_index_unique",
    calls_table.c.operation_id,
    calls_table.c.call_index,
    unique=True,
    sqlite_where=(calls_table.c.operation_id.isnot(None)),
    postgresql_where=(calls_table.c.operation_id.isnot(None)),
)

# === Artifacts ===

artifacts_table = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column(
        "produced_by_state_id",
        String(64),
        nullable=True,
    ),
    Column("sink_effect_id", String(64), nullable=True),
    Column("sink_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("idempotency_key", String(256)),  # For retry deduplication
    Column("publication_performed", Boolean, nullable=False),
    Column("publication_evidence_kind", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite FK: producer node state must belong to the artifact run.
    ForeignKeyConstraint(["produced_by_state_id", "run_id"], ["node_states.state_id", "node_states.run_id"]),
    ForeignKeyConstraint(
        ["sink_effect_id", "run_id", "sink_node_id"],
        ["sink_effects.effect_id", "sink_effects.run_id", "sink_effects.sink_node_id"],
    ),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    CheckConstraint(
        "(produced_by_state_id IS NOT NULL AND sink_effect_id IS NULL) OR (produced_by_state_id IS NULL AND sink_effect_id IS NOT NULL)",
        name="ck_artifacts_producer_xor",
    ),
    CheckConstraint(
        "publication_evidence_kind IN ('returned','reconciled','inherited','virtual','legacy_returned')",
        name="ck_artifacts_publication_evidence_kind",
    ),
)
Index(
    "uq_artifacts_run_idempotency_key",
    artifacts_table.c.run_id,
    artifacts_table.c.idempotency_key,
    unique=True,
    sqlite_where=artifacts_table.c.idempotency_key.isnot(None),
    postgresql_where=artifacts_table.c.idempotency_key.isnot(None),
)

# === Routing Events ===

routing_events_table = Table(
    "routing_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("state_id", String(64), nullable=False),
    Column("edge_id", String(64), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("routing_group_id", String(64), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("mode", String(16), nullable=False),  # move, copy
    Column("reason_hash", String(64)),
    Column("reason_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("routing_group_id", "ordinal"),
    # Composite FKs: routed state and edge must belong to the same run.
    ForeignKeyConstraint(["state_id", "run_id"], ["node_states.state_id", "node_states.run_id"]),
    ForeignKeyConstraint(["edge_id", "run_id"], ["edges.edge_id", "edges.run_id"]),
)

# === Batches (Aggregation) ===

batches_table = Table(
    "batches",
    metadata,
    Column("batch_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("aggregation_node_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),
    Column("aggregation_state_id", String(64)),
    Column("retry_of_batch_id", String(64)),
    Column("trigger_reason", String(128)),
    Column("trigger_type", String(32)),  # TriggerType enum value
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # draft, executing, completed, failed
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    # Composite unique target for run-scoped batch FKs.
    UniqueConstraint("batch_id", "run_id"),
    UniqueConstraint("retry_of_batch_id"),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["aggregation_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
    # Composite FK: aggregation state must belong to the batch run.
    ForeignKeyConstraint(["aggregation_state_id", "run_id"], ["node_states.state_id", "node_states.run_id"]),
    # Retry lineage is same-run and one-to-one at the direct-retry level.
    ForeignKeyConstraint(["retry_of_batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)

batch_members_table = Table(
    "batch_members",
    metadata,
    Column("batch_id", String(64), nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("token_id", String(64), nullable=False),
    Column("ordinal", Integer, nullable=False),
    PrimaryKeyConstraint("batch_id", "token_id"),  # Natural key: token belongs to batch once
    UniqueConstraint("batch_id", "ordinal"),  # Ordering uniqueness within a batch
    # Composite FKs: member token and batch must belong to the same run.
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
)

batch_outputs_table = Table(
    "batch_outputs",
    metadata,
    Column("batch_output_id", String(64), primary_key=True),  # Surrogate PK
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("output_type", String(32), nullable=False),  # token, artifact
    Column("output_id", String(64), nullable=False),
    UniqueConstraint("batch_id", "output_type", "output_id"),  # Prevent duplicates
)

# === Indexes for Query Performance ===

Index("ix_routing_events_state", routing_events_table.c.state_id)
Index("ix_routing_events_run_state", routing_events_table.c.run_id, routing_events_table.c.state_id)
Index("ix_routing_events_group", routing_events_table.c.routing_group_id)
Index("ix_batches_run_status", batches_table.c.run_id, batches_table.c.status)
Index("ix_batch_members_batch", batch_members_table.c.batch_id)
Index("ix_batch_outputs_batch", batch_outputs_table.c.batch_id)

# Indexes for existing Phase 1 tables
Index("ix_nodes_run_id", nodes_table.c.run_id)
Index("ix_edges_run_id", edges_table.c.run_id)
Index("ix_rows_run_id", rows_table.c.run_id)
Index("ix_rows_run_ingest_sequence", rows_table.c.run_id, rows_table.c.ingest_sequence)
Index("ix_rows_run_source_row", rows_table.c.run_id, rows_table.c.source_node_id, rows_table.c.source_row_index)
Index("ix_tokens_row_id", tokens_table.c.row_id)
# Performance index for run-accounting API projections and session-list batch
# reads. This is additive and intentionally does not advance the SQLite schema
# epoch or participate in the required-schema compatibility gate.
Index("ix_tokens_run_id", tokens_table.c.run_id)
Index("ix_token_parents_parent", token_parents_table.c.parent_token_id)
Index("ix_node_states_token", node_states_table.c.token_id)
Index("ix_node_states_node", node_states_table.c.node_id)
Index("ix_calls_state", calls_table.c.state_id)
Index("ix_calls_operation", calls_table.c.operation_id)  # For operation call lookups
# Phase 5b — supports the cross-DB anchor lookup: "given a session-side
# interpretation_events.resolved_prompt_template_hash, find the matching
# Landscape calls row". The index is sparse on NULL (SQLite excludes NULL
# keys from B-tree indexes by default), so the storage cost is proportional
# to the number of LLM-transform calls downstream of an interpretation event.
Index(
    "ix_calls_resolved_prompt_template_hash",
    calls_table.c.resolved_prompt_template_hash,
)
Index("ix_operations_node_run", operations_table.c.node_id, operations_table.c.run_id)
Index("ix_artifacts_run", artifacts_table.c.run_id)

# === Validation Errors (WP-11.99: Config-Driven Plugin Schemas) ===

validation_errors_table = Table(
    "validation_errors",
    metadata,
    Column("error_id", String(32), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("node_id", String(NODE_ID_COLUMN_LENGTH)),  # Source node where validation failed (nullable)
    Column("row_id", String(64), ForeignKey("rows.row_id")),  # Persisted quarantine row when one exists
    Column("row_hash", String(64), nullable=False),
    Column("row_data_json", Text),  # Store the row for debugging
    Column("error", Text, nullable=False),
    Column("schema_mode", String(16), nullable=False),  # "fixed", "flexible", "observed", "parse"
    Column("destination", String(255), nullable=False),  # Sink name or "discard"
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Schema contract violation details (Phase 5: Unified Schema Contracts)
    # These columns provide structured violation data for auditing
    Column("violation_type", String(32)),  # "type_mismatch", "missing_field", "extra_field"
    Column("original_field_name", String(256)),  # "'Amount USD'" for display
    Column("normalized_field_name", String(256)),  # "amount_usd" for code reference
    Column("expected_type", String(32)),  # "int", "str", etc.
    Column("actual_type", String(32)),  # Type of actual value
    # Composite FK to nodes (node_id, run_id) - nullable node_id supported
    ForeignKeyConstraint(
        ["node_id", "run_id"],
        ["nodes.node_id", "nodes.run_id"],
        ondelete="RESTRICT",
    ),
)

Index("ix_validation_errors_run", validation_errors_table.c.run_id)
Index("ix_validation_errors_node", validation_errors_table.c.node_id)
Index("ix_validation_errors_run_row", validation_errors_table.c.run_id, validation_errors_table.c.row_id)

# === Transform Errors (WP-11.99b: Transform Error Routing) ===

transform_errors_table = Table(
    "transform_errors",
    metadata,
    Column("error_id", String(32), primary_key=True),
    Column("run_id", String(64), nullable=False),
    Column("token_id", String(64), nullable=False),
    Column("transform_id", String(NODE_ID_COLUMN_LENGTH), nullable=False),  # Part of composite FK to nodes
    Column("row_hash", String(64), nullable=False),
    Column("row_data_json", Text),
    Column("error_details_json", Text),  # From TransformResult.error()
    Column("destination", String(255), nullable=False),  # Sink name or "discard"
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite FK to tokens (token_id, run_id) — enforces token/run ownership
    ForeignKeyConstraint(
        ["token_id", "run_id"],
        ["tokens.token_id", "tokens.run_id"],
        ondelete="RESTRICT",
    ),
    # Composite FK to nodes (transform_id, run_id)
    ForeignKeyConstraint(
        ["transform_id", "run_id"],
        ["nodes.node_id", "nodes.run_id"],
        ondelete="RESTRICT",
    ),
)

Index("ix_transform_errors_run", transform_errors_table.c.run_id)
Index("ix_transform_errors_token", transform_errors_table.c.token_id)
Index("ix_transform_errors_transform", transform_errors_table.c.transform_id)

# === Checkpoints (Phase 5: Production Hardening) ===

checkpoints_table = Table(
    "checkpoints",
    metadata,
    Column("checkpoint_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("sequence_number", Integer, nullable=False),  # Monotonic progress marker
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Topology validation (topological checkpoint compatibility). The full
    # topology hash embeds every node's config hash, so no per-node anchor
    # columns are needed for compatibility checking (epoch 19).
    Column("upstream_topology_hash", String(64), nullable=False),  # Hash of ALL nodes + edges in the DAG
    # Format version for compatibility checking (replaces hardcoded date check)
    # Version 1: Pre-deterministic node IDs (legacy, rejected)
    # Version 2: Deterministic node IDs (2026-01-24+)
    # Version 3: Phase 2 traversal refactor checkpoint break
    # Version 4: Pending coalesce state persisted in checkpoints
    # Version 5: F1 durability unification — buffered tokens live in journal
    #            BLOCKED rows; the checkpoint carries only scalar barrier
    #            metadata (barrier_scalars_json)
    Column("format_version", Integer, nullable=True),  # Nullable — populated on new runs, NULL for checkpoints created before this column
    # Epoch 20: F1 durability unification — scalar barrier metadata replaces
    # the dropped per-barrier buffer blob columns; buffered tokens live in
    # token_work_items journal BLOCKED rows.
    Column("barrier_scalars_json", Text, nullable=True),
)

Index("ix_checkpoints_run", checkpoints_table.c.run_id)
Index(
    "ix_checkpoints_run_seq",
    checkpoints_table.c.run_id,
    checkpoints_table.c.sequence_number,
)
Index(
    "ix_checkpoints_run_sequence_unique",
    checkpoints_table.c.run_id,
    checkpoints_table.c.sequence_number,
    unique=True,
)

# === Secret Resolutions (P2-10: Key Vault Secret Audit Trail) ===
# Records which secrets were loaded from where during pipeline startup.
# NOTE: Records are inserted AFTER run is created, though secrets load before.
# Allows auditors to answer: "Which Key Vault did this secret come from?"
# without exposing actual secret values (stores fingerprint only).

secret_resolutions_table = Table(
    "secret_resolutions",
    metadata,
    Column("resolution_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("timestamp", Float, nullable=False),  # When secret was loaded (before run)
    Column("env_var_name", String(256), nullable=False),  # Target environment variable
    Column("source", String(32), nullable=False),  # 'keyvault' (env source doesn't record)
    Column("vault_url", Text, nullable=True),  # Key Vault URL (NULL if source != keyvault)
    Column("secret_name", String(256), nullable=True),  # Secret name in vault
    Column("fingerprint", String(64), nullable=False),  # HMAC fingerprint of secret value
    Column("resolution_latency_ms", Float, nullable=True),  # Time to fetch from vault
)

Index("ix_secret_resolutions_run", secret_resolutions_table.c.run_id)


# === Web Auth Events ===
# Additive, non-run-scoped Landscape table for web authentication audit.
# Records must never contain passwords, bearer tokens, raw JWTs, or unredacted
# provider exception text.

auth_events_table = Table(
    "auth_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("event_type", String(32), nullable=False),
    Column("outcome", String(16), nullable=False),
    Column("provider", String(16), nullable=False),
    Column("user_id", String(256)),
    Column("username", String(256)),
    Column("failure_category", String(64)),
    Column("request_id", String(64)),
    Column("client_host", String(128)),
    Column("user_agent", Text),
    Column("metadata_json", Text, nullable=False),
    CheckConstraint(
        "event_type IN ('login', 'token_issued', 'auth_failure')",
        name="ck_auth_events_event_type",
    ),
    CheckConstraint(
        "outcome IN ('success', 'failure')",
        name="ck_auth_events_outcome",
    ),
    CheckConstraint(
        "provider IN ('local', 'oidc', 'entra')",
        name="ck_auth_events_provider",
    ),
)

Index("ix_auth_events_occurred_at", auth_events_table.c.occurred_at)
Index("ix_auth_events_type_outcome", auth_events_table.c.event_type, auth_events_table.c.outcome)
Index("ix_auth_events_user", auth_events_table.c.user_id)

# === Pre-flight Results (Pipeline Dependencies & Commencement Gates) ===

preflight_results_table = Table(
    "preflight_results",
    metadata,
    Column("result_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("result_type", String(32), nullable=False),
    Column("name", String(256), nullable=False),  # Dependency name or gate name
    Column("result_json", Text, nullable=False),  # Full result as canonical JSON
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "result_type IN ('dependency_run', 'commencement_gate', 'readiness_check')",
        name="ck_preflight_result_type",
    ),
)

Index("ix_preflight_results_run", preflight_results_table.c.run_id)
