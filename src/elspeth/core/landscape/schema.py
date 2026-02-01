# src/elspeth/core/landscape/schema.py
"""SQLAlchemy table definitions for Landscape.

Uses SQLAlchemy Core (not ORM) for explicit control over queries
and compatibility with multiple database backends.
"""

from sqlalchemy import (
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
)

# Shared metadata for all tables
metadata = MetaData()

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
    Column("source_schema_json", Text),  # Nullable for backward compatibility
    # Field resolution mapping from source.load() - captures original→final header mapping
    # when normalize_fields or field_mapping is used. Stored at run level since one source per run.
    Column("source_field_resolution_json", Text),  # Nullable for backward compatibility
    Column("status", String(32), nullable=False),
    # Export tracking - separate from run status so export failures
    # don't mask successful pipeline completion
    Column("export_status", String(32)),  # pending, completed, failed, None if not configured
    Column("export_error", Text),  # Error message if export failed
    Column("exported_at", DateTime(timezone=True)),  # When export completed
    Column("export_format", String(16)),  # csv, json
    Column("export_sink", String(128)),  # Sink name used for export
)

# === Nodes (Plugin Instances) ===

nodes_table = Table(
    "nodes",
    metadata,
    Column("node_id", String(64), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("node_type", String(32), nullable=False),
    Column("plugin_version", String(32), nullable=False),
    Column("determinism", String(32), nullable=False),  # deterministic, seeded, nondeterministic (from Determinism enum)
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64)),
    Column("sequence_in_pipeline", Integer),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    # Schema configuration for audit trail (WP-11.99)
    Column("schema_mode", String(16)),  # "dynamic", "strict", "free", "parse", or NULL
    Column("schema_fields_json", Text),  # JSON array of field definitions, or NULL
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

rows_table = Table(
    "rows",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(64), nullable=False),
    Column("row_index", Integer, nullable=False),
    Column("source_data_hash", String(64), nullable=False),
    Column("source_data_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "row_index"),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["source_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# === Tokens ===

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(64), primary_key=True),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(32), nullable=True, index=True),  # For deaggregation
    Column("branch_name", String(64)),
    Column("step_in_pipeline", Integer),  # Step where this token was created (fork/coalesce/expand)
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# === Token Outcomes (AUD-001: Explicit terminal state recording) ===

token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False, index=True),
    # Core outcome
    Column("outcome", String(32), nullable=False),
    Column("is_terminal", Integer, nullable=False),  # SQLite doesn't have Boolean, use Integer
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    # Outcome-specific fields (nullable based on outcome type)
    Column("sink_name", String(128)),
    Column("batch_id", String(64), ForeignKey("batches.batch_id")),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    # Optional extended context
    Column("context_json", Text),
    # Branch contract for FORKED/EXPANDED outcomes (enables recovery validation)
    Column("expected_branches_json", Text),
)

# Partial unique index: exactly one terminal outcome per token
# Note: SQLite partial index syntax differs; SQLAlchemy handles this
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
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
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),  # Added for composite FK
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
    UniqueConstraint("token_id", "node_id", "attempt"),
    UniqueConstraint("token_id", "step_index", "attempt"),
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
    Column("operation_type", String(32), nullable=False),  # 'source_load' | 'sink_write'
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),  # 'open' | 'completed' | 'failed' | 'pending'
    Column("input_data_ref", String(256)),  # Payload store reference for operation input
    Column("output_data_ref", String(256)),  # Payload store reference for operation output
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
        ForeignKey("node_states.state_id"),
        nullable=False,
    ),
    Column("sink_node_id", String(64), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("idempotency_key", String(256)),  # For retry deduplication
    Column("created_at", DateTime(timezone=True), nullable=False),
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
    Column("aggregation_state_id", String(64), ForeignKey("node_states.state_id")),
    Column("trigger_reason", String(128)),
    Column("trigger_type", String(32)),  # TriggerType enum value
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # draft, executing, completed, failed
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    # Composite FK to nodes (node_id, run_id)
    ForeignKeyConstraint(["aggregation_node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

batch_members_table = Table(
    "batch_members",
    metadata,
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("batch_id", "ordinal"),
    UniqueConstraint("batch_id", "token_id"),  # Prevent duplicate token in same batch
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
Index("ix_tokens_row_id", tokens_table.c.row_id)
Index("ix_token_parents_parent", token_parents_table.c.parent_token_id)
Index("ix_node_states_token", node_states_table.c.token_id)
Index("ix_node_states_node", node_states_table.c.node_id)
Index("ix_calls_state", calls_table.c.state_id)
Index("ix_calls_operation", calls_table.c.operation_id)  # For operation call lookups
Index("ix_operations_node_run", operations_table.c.node_id, operations_table.c.run_id)
Index("ix_artifacts_run", artifacts_table.c.run_id)

# === Validation Errors (WP-11.99: Config-Driven Plugin Schemas) ===

validation_errors_table = Table(
    "validation_errors",
    metadata,
    Column("error_id", String(32), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("node_id", String(64)),  # Source node where validation failed (nullable)
    Column("row_hash", String(64), nullable=False),
    Column("row_data_json", Text),  # Store the row for debugging
    Column("error", Text, nullable=False),
    Column("schema_mode", String(16), nullable=False),  # "strict", "free", "dynamic", "parse"
    Column("destination", String(255), nullable=False),  # Sink name or "discard"
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite FK to nodes (node_id, run_id) - nullable node_id supported
    ForeignKeyConstraint(
        ["node_id", "run_id"],
        ["nodes.node_id", "nodes.run_id"],
        ondelete="RESTRICT",
    ),
)

Index("ix_validation_errors_run", validation_errors_table.c.run_id)
Index("ix_validation_errors_node", validation_errors_table.c.node_id)

# === Transform Errors (WP-11.99b: Transform Error Routing) ===

transform_errors_table = Table(
    "transform_errors",
    metadata,
    Column("error_id", String(32), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id", ondelete="RESTRICT"), nullable=False),
    Column("transform_id", String(64), nullable=False),  # Part of composite FK to nodes
    Column("row_hash", String(64), nullable=False),
    Column("row_data_json", Text),
    Column("error_details_json", Text),  # From TransformResult.error()
    Column("destination", String(255), nullable=False),  # Sink name or "discard"
    Column("created_at", DateTime(timezone=True), nullable=False),
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
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("node_id", String(64), nullable=False),  # Part of composite FK to nodes
    Column("sequence_number", Integer, nullable=False),  # Monotonic progress marker
    Column("aggregation_state_json", Text),  # Serialized aggregation buffers (if any)
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Topology validation (topological checkpoint compatibility)
    Column("upstream_topology_hash", String(64), nullable=False),  # Hash of nodes + edges upstream of checkpoint
    Column("checkpoint_node_config_hash", String(64), nullable=False),  # Hash of checkpoint node config only
    # Format version for compatibility checking (replaces hardcoded date check)
    # Version 1: Pre-deterministic node IDs (legacy, rejected)
    # Version 2: Deterministic node IDs (2026-01-24+)
    Column("format_version", Integer, nullable=True),  # Nullable for backwards compat with existing checkpoints
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
