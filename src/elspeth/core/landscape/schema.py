# src/elspeth/core/landscape/schema.py
"""SQLAlchemy table definitions for Landscape.

Uses SQLAlchemy Core (not ORM) for explicit control over queries
and compatibility with multiple database backends.
"""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
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
    Column("status", String(32), nullable=False),
    # Export tracking - separate from run status so export failures
    # don't mask successful pipeline completion
    Column(
        "export_status", String(32)
    ),  # pending, completed, failed, None if not configured
    Column("export_error", Text),  # Error message if export failed
    Column("exported_at", DateTime(timezone=True)),  # When export completed
    Column("export_format", String(16)),  # csv, json
    Column("export_sink", String(128)),  # Sink name used for export
)

# === Nodes (Plugin Instances) ===

nodes_table = Table(
    "nodes",
    metadata,
    Column("node_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("node_type", String(32), nullable=False),
    Column("plugin_version", String(32), nullable=False),
    Column(
        "determinism", String(32), nullable=False
    ),  # deterministic, seeded, nondeterministic (from Determinism enum)
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64)),
    Column("sequence_in_pipeline", Integer),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    # Schema configuration for audit trail (WP-11.99)
    Column("schema_mode", String(16)),  # "dynamic", "strict", "free", or NULL
    Column("schema_fields_json", Text),  # JSON array of field definitions, or NULL
)

# === Edges ===

edges_table = Table(
    "edges",
    metadata,
    Column("edge_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("from_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("to_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("label", String(64), nullable=False),
    Column("default_mode", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "from_node_id", "label"),
)

# === Source Rows ===

rows_table = Table(
    "rows",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("row_index", Integer, nullable=False),
    Column("source_data_hash", String(64), nullable=False),
    Column("source_data_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "row_index"),
)

# === Tokens ===

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(64), primary_key=True),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("branch_name", String(64)),
    Column(
        "step_in_pipeline", Integer
    ),  # Step where this token was created (fork/coalesce)
    Column("created_at", DateTime(timezone=True), nullable=False),
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
    Column("node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("step_index", Integer, nullable=False),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),
    Column("input_hash", String(64), nullable=False),
    Column("output_hash", String(64)),
    Column("context_before_json", Text),
    Column("context_after_json", Text),
    Column("duration_ms", Float),
    Column("error_json", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    UniqueConstraint("token_id", "node_id", "attempt"),
    UniqueConstraint("token_id", "step_index", "attempt"),
)

# === External Calls ===

calls_table = Table(
    "calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
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
    UniqueConstraint("state_id", "call_index"),
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
    Column("sink_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", String(512), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
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
    Column(
        "aggregation_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False
    ),
    Column("aggregation_state_id", String(64), ForeignKey("node_states.state_id")),
    Column("trigger_reason", String(128)),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # draft, executing, completed, failed
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
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
Index("ix_artifacts_run", artifacts_table.c.run_id)

# === Validation Errors (WP-11.99: Config-Driven Plugin Schemas) ===

validation_errors_table = Table(
    "validation_errors",
    metadata,
    Column("error_id", String(32), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("node_id", String(64)),  # Source node where validation failed
    Column("row_hash", String(64), nullable=False),
    Column("row_data_json", Text),  # Store the row for debugging
    Column("error", Text, nullable=False),
    Column("schema_mode", String(16), nullable=False),  # "strict", "free", "dynamic"
    Column("destination", String(255), nullable=False),  # Sink name or "discard"
    Column("created_at", DateTime(timezone=True), nullable=False),
)

Index("ix_validation_errors_run", validation_errors_table.c.run_id)
Index("ix_validation_errors_node", validation_errors_table.c.node_id)

# === Checkpoints (Phase 5: Production Hardening) ===

checkpoints_table = Table(
    "checkpoints",
    metadata,
    Column("checkpoint_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("sequence_number", Integer, nullable=False),  # Monotonic progress marker
    Column("aggregation_state_json", Text),  # Serialized aggregation buffers (if any)
    Column("created_at", DateTime(timezone=True), nullable=False),
)

Index("ix_checkpoints_run", checkpoints_table.c.run_id)
Index(
    "ix_checkpoints_run_seq",
    checkpoints_table.c.run_id,
    checkpoints_table.c.sequence_number,
)
