"""SQLAlchemy table definitions for Landscape.

Uses SQLAlchemy Core (not ORM) for explicit control over queries
and compatibility with multiple database backends.
"""

from enum import StrEnum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
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
    text,
)

# Shared metadata for all tables
metadata = MetaData()

# Explicit SQLite schema epoch for pre-1.0 compatibility policy.
# Stored in PRAGMA user_version so future releases can distinguish
# "intentionally old schema, needs migration" from "runtime-required field".
#
# Epoch history (pre-1.0 policy — bumps require DB recreation):
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
SQLITE_SCHEMA_EPOCH = 18

# Column width for node_id across all tables. Referenced by dag.py
# for validation — changing this value requires an Alembic migration.
NODE_ID_COLUMN_LENGTH = 64


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
    Column("seeded_from_cache", Boolean, nullable=False, default=False, server_default=text("0")),
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

run_sources_table = Table(
    "run_sources",
    metadata,
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(64), nullable=False),
    Column("source_name", String(64), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("lifecycle_state", String(32), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("schema_json", Text),
    Column("schema_contract_json", Text),
    Column("schema_contract_hash", String(16)),
    Column("field_resolution_json", Text),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("run_id", "source_node_id"),
    UniqueConstraint("run_id", "source_name"),
    CheckConstraint(
        "lifecycle_state IN ('ready', 'loading', 'exhausted', 'loaded', 'interrupted')",
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
    Column("node_id", String(64), nullable=False),
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
    Column("from_node_id", String(64), nullable=False),
    Column("to_node_id", String(64), nullable=False),
    Column("label", String(64), nullable=False),
    Column("default_mode", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
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
#    intentionally sets all three equal because there is exactly one source;
#    a future contributor adding a multi-source synthesised-run path could
#    silently drift. Tracked under filigree elspeth-92afea0d23 (elspeth-lints
#    rule with the same enforcement status as ``trust_tier.tier_model``).
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
    Column("source_node_id", String(64), nullable=False),
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
)

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
    Column("node_id", String(64)),
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
        "(status = 'leased' AND lease_owner IS NOT NULL AND length(lease_owner) > 0) OR status != 'leased'",
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
        "event_type IN ('enqueue', 'restore_blocked', 'claim_ready', 'claim_pending_sink', "
        "'recover_expired_lease', 'lease_lost', 'mark_waiting', 'release_waiting', 'mark_blocked', "
        "'mark_terminal', 'mark_failed', 'mark_pending_sink', 'mark_pending_sink_terminal', "
        "'mark_blocked_barrier_terminal')",
        name="ck_scheduler_events_event_type",
    ),
    CheckConstraint(
        "from_status IS NULL OR from_status IN ('ready', 'leased', 'waiting', 'blocked', 'pending_sink', 'terminal', 'failed')",
        name="ck_scheduler_events_from_status",
    ),
    CheckConstraint(
        "to_status IN ('ready', 'leased', 'waiting', 'blocked', 'pending_sink', 'terminal', 'failed')",
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
    Column("node_id", String(64), nullable=False),
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
    Column("node_id", String(64), nullable=False),
    Column("operation_type", String(32), nullable=False),  # 'source_load' | 'sink_write' | 'runtime_preflight'
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
)

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
        nullable=False,
    ),
    Column("sink_node_id", String(64), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("idempotency_key", String(256)),  # For retry deduplication
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite FK: producer node state must belong to the artifact run.
    ForeignKeyConstraint(["produced_by_state_id", "run_id"], ["node_states.state_id", "node_states.run_id"]),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["sink_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# === Routing Events ===

routing_events_table = Table(
    "routing_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
    Column("edge_id", String(64), ForeignKey("edges.edge_id"), nullable=False),
    Column("routing_group_id", String(64), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("mode", String(16), nullable=False),  # move, copy
    Column("reason_hash", String(64)),
    Column("reason_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("routing_group_id", "ordinal"),
)

# === Batches (Aggregation) ===

batches_table = Table(
    "batches",
    metadata,
    Column("batch_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("aggregation_node_id", String(64), nullable=False),
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
    Column("node_id", String(64)),  # Source node where validation failed (nullable)
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
    Column("transform_id", String(64), nullable=False),  # Part of composite FK to nodes
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
    Column("token_id", String(64), nullable=False),
    Column("node_id", String(64), nullable=False),  # Part of composite FK to nodes
    Column("sequence_number", Integer, nullable=False),  # Monotonic progress marker
    Column("aggregation_state_json", Text),  # Serialized aggregation buffers (if any)
    Column("coalesce_state_json", Text),  # Serialized pending coalesce state (if any)
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Topology validation (topological checkpoint compatibility)
    Column("upstream_topology_hash", String(64), nullable=False),  # Hash of nodes + edges upstream of checkpoint
    Column("checkpoint_node_config_hash", String(64), nullable=False),  # Hash of checkpoint node config only
    # Format version for compatibility checking (replaces hardcoded date check)
    # Version 1: Pre-deterministic node IDs (legacy, rejected)
    # Version 2: Deterministic node IDs (2026-01-24+)
    # Version 3: Phase 2 traversal refactor checkpoint break
    # Version 4: Pending coalesce state persisted in checkpoints
    Column("format_version", Integer, nullable=True),  # Nullable — populated on new runs, NULL for checkpoints created before this column
    # Composite FK: enforces token_id and run_id belong together (prevents cross-run contamination)
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(
        ["node_id", "run_id"],
        ["nodes.node_id", "nodes.run_id"],
    ),
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
