"""SQLAlchemy Core table definitions for the session database.

Tables: sessions, chat_messages, composition_states, runs, run_events,
blobs, blob_run_links.

Current schema bootstrap lives in ``sessions/schema.py``. Pre-release
session databases are created from this metadata and stale runtime DBs
are deleted/recreated rather than migrated.

All tables live in a dedicated session database, separate from the
Landscape audit database.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import JSON

metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("auth_provider_type", String, nullable=False, default="local"),
    Column("title", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column(
        "forked_from_session_id",
        String,
        ForeignKey("sessions.id"),
        nullable=True,
    ),
    Column("forked_from_message_id", String, nullable=True),
)

chat_messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("raw_content", Text, nullable=True),
    Column("tool_calls", JSON, nullable=True),
    Column("tool_call_id", String, nullable=True),
    Column("sequence_no", Integer, nullable=False),
    Column("writer_principal", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Composite FK forces same-session ownership: a message in session B
    # cannot reference a composition state owned by session A. When
    # composition_state_id is NULL, standard SQL partial-null semantics
    # skip FK enforcement, which is the intended behavior.
    Column("composition_state_id", String, nullable=True),
    Column("parent_assistant_id", String, nullable=True),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_chat_messages_composition_state_session",
    ),
    # Composite same-session FK on parent_assistant_id closes the
    # cross-session lineage hole: a tool row in session B cannot
    # reference an assistant row in session A. ON DELETE CASCADE
    # removes child tool rows when the assistant is deleted, preventing
    # orphan tool rows from accumulating in the audit DB. The schema
    # cannot mechanically enforce that the referenced row has
    # role='assistant'; Task 9's _assert_parent_assistant_message guard
    # adds that check at the helper-call boundary.
    ForeignKeyConstraint(
        ["parent_assistant_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_chat_messages_parent_assistant_session",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "id",
        "session_id",
        name="uq_chat_messages_id_session",
    ),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool', 'audit')",
        name="ck_chat_messages_role",
    ),
    CheckConstraint(
        "(role = 'tool') = (tool_call_id IS NOT NULL)",
        name="ck_chat_messages_tool_call_id_role",
    ),
    CheckConstraint(
        "(role = 'tool') = (parent_assistant_id IS NOT NULL)",
        name="ck_chat_messages_parent_role",
    ),
    CheckConstraint(
        "writer_principal IN ('compose_loop', 'route_user_message', 'route_system_message', 'admin_tool', 'session_fork')",
        name="ck_chat_messages_writer_principal",
    ),
    Index(
        "ix_chat_messages_session_sequence",
        "session_id",
        "sequence_no",
        unique=True,
    ),
    Index(
        "ix_chat_messages_session_tool_call_id",
        "session_id",
        "tool_call_id",
    ),
)

# Partial unique index: tool_call_id must be unique within
# (session_id, role='tool') scope. Two tool rows in the same session
# cannot share a provider tool_call_id (would conflate distinct LLM
# tool calls), but the same tool_call_id may legally appear in two
# different sessions, and non-tool rows (NULL tool_call_id) must not
# collide on NULL with each other. The same predicate is supplied to
# both ``sqlite_where`` (SQLite 3.8.0+) and ``postgresql_where``
# (PostgreSQL >= 9.5) so the index is equivalent across dialects.
# Mirrors the project pattern at ``uq_runs_one_active_per_session``
# below.
Index(
    "uq_chat_messages_tool_call_id",
    chat_messages_table.c.session_id,
    chat_messages_table.c.tool_call_id,
    unique=True,
    sqlite_where=chat_messages_table.c.role == "tool",
    postgresql_where=chat_messages_table.c.role == "tool",
)

composition_states_table = Table(
    "composition_states",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("version", Integer, nullable=False),
    Column("source", JSON, nullable=True),
    Column("nodes", JSON, nullable=True),
    Column("edges", JSON, nullable=True),
    Column("outputs", JSON, nullable=True),
    Column("metadata_", JSON, nullable=True),
    Column("is_valid", Boolean, nullable=False, default=False),
    Column("validation_errors", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column(
        "derived_from_state_id",
        String,
        ForeignKey("composition_states.id"),
        nullable=True,
    ),
    # ``provenance`` records WHY this state row was written. The CHECK
    # below is a closed enum — extending it requires design review and
    # a corresponding spec §4.1.2 amendment, not a silent value
    # addition. Schedule 1A treats this as a DB-only audit column: it
    # is NOT surfaced on ``CompositionStateRecord`` /
    # ``CompositionStateResponse``. Read-side hydration is deferred to
    # Schedule 1B+ per plan §1053-1061.
    Column("provenance", String, nullable=False),
    UniqueConstraint("session_id", "version", name="uq_composition_state_version"),
    # Composite uniqueness target for composite FKs on chat_messages /
    # runs. The primary key already makes `id` unique on its own; this
    # constraint exists solely so SQL engines (including Postgres) will
    # accept (id, session_id) as an FK reference.
    UniqueConstraint("id", "session_id", name="uq_composition_state_id_session"),
    # Closed enum: every value corresponds to a documented writer path
    # in spec §4.1.2 (as amended by the Phase 1 plan supersession
    # marker — ``session_fork`` is the cross-session fork-copy value,
    # and ``session_seed`` is broadened to mean "any state row written
    # outside the compose loop's tool-call path"). Adding a value here
    # without amending the spec creates an untraceable writer category
    # in the audit DB.
    CheckConstraint(
        "provenance IN ('tool_call', 'convergence_persist', 'plugin_crash_persist', 'preflight_persist', 'session_seed', 'session_fork')",
        name="ck_composition_states_provenance",
    ),
)

runs_table = Table(
    "runs",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    # Composite FK forces same-session ownership: a run in session B
    # cannot reference a composition state owned by session A. state_id
    # is NOT NULL so no partial-null concerns.
    Column("state_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("rows_processed", Integer, nullable=False, default=0),
    Column("rows_succeeded", Integer, nullable=False, default=0),
    Column("rows_failed", Integer, nullable=False, default=0),
    Column("rows_routed_success", Integer, nullable=False, default=0),
    Column("rows_routed_failure", Integer, nullable=False, default=0),
    Column("rows_quarantined", Integer, nullable=False, default=0),
    Column("error", Text, nullable=True),
    Column("landscape_run_id", String, nullable=True),
    Column("pipeline_yaml", Text, nullable=True),
    ForeignKeyConstraint(
        ["state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_runs_state_session",
    ),
    # Phase 2.2 (elspeth-0de989c56d): four-value terminal taxonomy.
    # The constraint mirrors SessionRunStatus in web/sessions/protocol.py;
    # adding a value to the Literal without updating this CheckConstraint
    # would let the dataclass validator pass while the DB rejects the row,
    # so widen both in lockstep.
    CheckConstraint(
        "status IN ('pending', 'running', 'completed', 'completed_with_failures', 'failed', 'empty', 'cancelled')",
        name="ck_runs_status",
    ),
)

# Partial unique index: at most one active (pending/running) run per session.
# Enforces the one-active-run invariant at the database level, eliminating
# the TOCTOU race in the service-level check-and-insert.
Index(
    "uq_runs_one_active_per_session",
    runs_table.c.session_id,
    unique=True,
    sqlite_where=runs_table.c.status.in_(["pending", "running"]),
)

blobs_table = Table(
    "blobs",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("filename", String, nullable=False),
    Column("mime_type", String, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("content_hash", String, nullable=True),
    Column("storage_path", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("created_by", String, nullable=False),
    Column("source_description", String, nullable=True),
    Column("status", String, nullable=False, server_default="ready"),
    CheckConstraint(
        "created_by IN ('user', 'assistant', 'pipeline')",
        name="ck_blobs_created_by",
    ),
    CheckConstraint(
        "status IN ('ready', 'pending', 'error')",
        name="ck_blobs_status",
    ),
    # Integrity invariant: a blob that claims to be ready MUST carry a
    # SHA-256 hex content_hash (exactly 64 lowercase hex characters).
    # Without this, a defective finalization path — or a direct SQL
    # write — could persist a "ready" row whose hash either is NULL
    # (no integrity check possible) or is a malformed string like
    # "abc123" (will never match any real bytes, so every download
    # raises BlobIntegrityError).  Either failure mode leaves the
    # audit trail asserting "this blob is ready" while the bytes are
    # unverifiable in practice (AD-5/AD-7 in
    # docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md).
    #
    # The shape rule mirrors ``_validate_finalize_hash`` at the write
    # side (``re.compile(r"^[a-f0-9]{64}$")``).  The DDL here uses
    # SQLite GLOB syntax because the session DB is SQLite (see
    # ``sessions/engine.py`` and the StaticPool test harness). If a
    # different dialect is introduced, add its V0 check expression here
    # instead of adding a migration path.
    CheckConstraint(
        "status != 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 64 AND content_hash NOT GLOB '*[^a-f0-9]*')",
        name="ck_blobs_ready_hash",
    ),
)

blob_run_links_table = Table(
    "blob_run_links",
    metadata,
    Column(
        "blob_id",
        String,
        ForeignKey("blobs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "run_id",
        String,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("direction", String, nullable=False),
    UniqueConstraint("blob_id", "run_id", "direction", name="uq_blob_run_link"),
    CheckConstraint(
        "direction IN ('input', 'output')",
        name="ck_blob_run_links_direction",
    ),
)
Index("ix_blob_run_links_blob_id", blob_run_links_table.c.blob_id)
Index("ix_blob_run_links_run_id", blob_run_links_table.c.run_id)

run_events_table = Table(
    "run_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "run_id",
        String,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("event_type", String, nullable=False),
    Column("data", JSON, nullable=False),
    CheckConstraint(
        "event_type IN ('progress', 'error', 'completed', 'cancelled', 'failed')",
        name="ck_run_events_type",
    ),
)

user_secrets_table = Table(
    "user_secrets",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("user_id", String, nullable=False),
    Column("auth_provider_type", String, nullable=False),
    Column("encrypted_value", LargeBinary, nullable=False),
    Column("salt", LargeBinary, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("name", "user_id", "auth_provider_type", name="uq_user_secret_name_user_provider"),
)
Index("ix_user_secrets_user_provider", user_secrets_table.c.user_id, user_secrets_table.c.auth_provider_type)

# ``audit_access_log`` — INERT IN PHASE 1A.
#
# This table records who viewed audit-grade message data (the eventual
# ``include_tool_rows=true`` route surface). 1A lands the table SCHEMA
# ONLY: no route writes it, no service method writes it, no fixture
# writes it. Phase 1A is the destructive session-DB schema reset
# boundary, so deferring this table to a later phase would force a
# second staging DB recreation for a table whose ownership, FK shape,
# and writer_principal enum are already known.
#
# DO NOT ADD A WRITER WITHOUT THE PRIVACY GATE. The table holds
# privacy-sensitive request context (``requesting_principal``,
# ``request_path``, ``query_args``, ``ip_address``). Before any later
# schedule adds a writer, that schedule MUST:
#
# 1. Define and test an allowlist for ``query_args`` keys. The writer
#    must never store request headers, request bodies, secrets,
#    provider tokens, or arbitrary exception strings. The allowlist
#    is a closed set, with every accepted key justified.
# 2. Choose an explicit IP retention policy. ``ip_address`` is
#    nullable for service-to-service calls and for retention
#    truncation. The policy must be stated in writing
#    (literal storage, /24 truncation, or keyed hash) and pinned by
#    test before the writer ships.
# 3. Prove via integration test that no out-of-allowlist payload can
#    reach the writer call site, even via misconfigured routes or
#    unhandled exception paths.
#
# CLOSED-LIST WRITER PRINCIPAL ENUM. The two values
# ``('audit_grade_view', 'admin_tool')`` are the entire universe of
# permitted writers. Adding a third value here is a governance
# action, not a coding action: it requires (a) a design review of
# the new writer's privacy posture, (b) a destructive session-DB
# recreation per ``project_db_migration_policy`` (no Alembic in this
# project), and (c) a corresponding spec amendment. The friction is
# the design — do not extend silently.
audit_access_log_table = Table(
    "audit_access_log",
    metadata,
    Column("id", String, primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("requesting_principal", String, nullable=False),
    Column("request_path", String, nullable=False),
    Column("query_args", JSON, nullable=False),
    Column("ip_address", String, nullable=True),
    Column("writer_principal", String, nullable=False),
    CheckConstraint(
        "writer_principal IN ('audit_grade_view', 'admin_tool')",
        name="ck_audit_access_log_writer_principal",
    ),
    Index("ix_audit_access_log_session_timestamp", "session_id", "timestamp"),
)
