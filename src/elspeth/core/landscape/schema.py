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
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64)),
    Column("sequence_in_pipeline", Integer),
    Column("registered_at", DateTime(timezone=True), nullable=False),
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
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# === Token Parents (for multi-parent joins) ===

token_parents_table = Table(
    "token_parents",
    metadata,
    Column(
        "token_id", String(64), ForeignKey("tokens.token_id"), primary_key=True
    ),
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
    UniqueConstraint("token_id", "step_index"),
)

# === External Calls ===

calls_table = Table(
    "calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column(
        "state_id", String(64), ForeignKey("node_states.state_id"), nullable=False
    ),
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
