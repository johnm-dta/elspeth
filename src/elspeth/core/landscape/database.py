"""Database connection management for Landscape.

Handles SQLite (development) and PostgreSQL (production) backends
with appropriate settings for each.
"""

import os
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NewType, Self, cast
from urllib.parse import quote
from weakref import WeakKeyDictionary

from sqlalchemy import Connection, Table, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.url import SENSITIVE_PARAMS, _scrub_odbc_connect_value
from elspeth.core.landscape.journal import LandscapeJournal
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, metadata, schema_identity_table
from elspeth.core.schema_identity import (
    SCHEMA_IDENTITY_TABLE_NAME,
    SchemaIdentityMismatch,
    insert_schema_identity,
    read_schema_identities,
    schema_identity_mismatch,
)
from elspeth.core.schema_shape import collect_metadata_shape_issues

# Tier-1 branded Engine type.
#
# ``Tier1Engine`` is a NewType wrapper around :class:`sqlalchemy.engine.Engine`
# that carries a static guarantee: the engine was created through
# :class:`LandscapeDB` and passed the backend-appropriate audit-integrity
# checks. For SQLite that includes the PRAGMA integrity probe
# (:meth:`LandscapeDB._verify_sqlite_pragmas`). The wrapper has zero runtime
# overhead (``NewType`` is erased at runtime); it is a type-checker signal only.
#
# Only :meth:`LandscapeDB.engine` may mint a ``Tier1Engine`` via ``cast()``.
# Call sites that accept ``Tier1Engine`` (e.g.
# :class:`~elspeth.core.landscape.scheduler_repository.TokenSchedulerRepository`)
# are guaranteed to receive a Tier-1 engine — any call site that tries to pass
# a bare :class:`Engine` will be caught by mypy.
Tier1Engine = NewType("Tier1Engine", Engine)

# Execution-option key that marks a connection's next transaction as carrying
# WRITE INTENT.  On writable SQLite engines the engine-level ``begin`` event
# (installed by :meth:`LandscapeDB._configure_sqlite`) inspects this option and
# begins the transaction with ``BEGIN IMMEDIATE`` — taking the single WAL write
# lock at BEGIN — instead of the default DEFERRED ``BEGIN``.  This closes the
# cross-process read-then-write hazard where a transaction that starts as a
# reader and later upgrades to a writer aborts with the non-retryable
# ``SQLITE_BUSY_SNAPSHOT`` (ADR-030 §D5; option-c design F10).
#
# Only engine-side write verbs set this option (via
# :func:`begin_write` / :meth:`LandscapeDB.write_connection`); read and
# dashboard connections never carry it, so they never contend for the write
# lock at BEGIN.
WRITE_INTENT_OPTION = "elspeth_write_intent"

_JOURNAL_WORKER_SUFFIX_RE = re.compile(r"[0-9a-f]+")

# Canonical SQLite PRAGMA invariants for the Landscape audit DB.
#
# Tier-1 doctrine: the audit DB is the legal record. If SQLite doesn't accept
# any of these PRAGMAs, the crash-safety / concurrency guarantees the audit
# subsystem depends on are unmet and we MUST refuse to open the database.
#
# - journal_mode=WAL — file-backed DBs MUST run in WAL mode (better
#   concurrency, fsync-on-checkpoint instead of fsync-per-write).  SQLite
#   silently downgrades to 'memory' for ``:memory:`` databases (WAL has no
#   meaning without an on-disk journal file); the probe accepts that.
# - synchronous=NORMAL — paired with WAL, this is the canonical
#   write-heavy crash-safe shape: durable across application crashes and
#   sufficiently durable across OS crashes when fsync semantics are honoured
#   by the filesystem.  FULL is slower without meaningful added durability
#   under WAL.  See https://www.sqlite.org/pragma.html#pragma_synchronous
# - foreign_keys=ON — referential integrity is mandatory for the audit
#   schema (run_id / token_id / node_id FKs must enforce).
# - busy_timeout=5000 ms — tolerate brief contention between recorders
#   without surfacing SQLITE_BUSY to the caller.
_SQLITE_PRAGMA_INVARIANTS_FILE: tuple[tuple[str, str], ...] = (
    ("journal_mode", "wal"),
    ("synchronous", "1"),
    ("foreign_keys", "1"),
    ("busy_timeout", "5000"),
)
_SQLITE_PRAGMA_INVARIANTS_MEMORY: tuple[tuple[str, str], ...] = (
    # ``:memory:`` databases have no on-disk journal file; SQLite reports
    # ``journal_mode=memory`` regardless of what we requested.  That is
    # the only sanctioned deviation from the file-backed contract.
    ("journal_mode", "memory"),
    ("synchronous", "1"),
    ("foreign_keys", "1"),
    ("busy_timeout", "5000"),
)


def verify_sqlite_tier1_pragmas(engine: Engine, *, owner: str) -> None:
    """Refuse SQLite engines that bypassed LandscapeDB's PRAGMA gate.

    Repository constructors use this as a runtime defense against casts or
    type ignores that smuggle a bare SQLite engine past the ``Tier1Engine``
    type. Non-SQLite engines deliberately skip this probe: PostgreSQL has no
    SQLite PRAGMA surface, and issuing these statements there is a backend
    syntax error.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        fk_result = conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one_or_none()
        jm_result = conn.exec_driver_sql("PRAGMA journal_mode").scalar_one_or_none()

    foreign_keys = "" if fk_result is None else str(fk_result).lower()
    journal_mode = "" if jm_result is None else str(jm_result).lower()

    violations: list[str] = []
    if foreign_keys != "1":
        violations.append(f"PRAGMA foreign_keys: expected '1', observed {foreign_keys!r}")
    if journal_mode not in ("wal", "memory"):
        violations.append(f"PRAGMA journal_mode: expected 'wal' (or 'memory' for :memory: DBs), observed {journal_mode!r}")

    if violations:
        raise AuditIntegrityError(
            f"{owner} received an engine that does not meet Tier-1 audit-integrity "
            "requirements; the engine was not opened through LandscapeDB. " + "; ".join(violations)
        )


class SchemaCompatibilityError(Exception):
    """Raised when the Landscape database schema is incompatible with current code."""

    pass


ADR019_CUTOVER_GUIDE = "docs/operator/migrations/adr-019.md"

_EPOCH_24_TOKEN_ROW_RUN_FK: tuple[str, tuple[str, ...], str, tuple[str, ...]] = (
    "tokens",
    ("row_id", "run_id"),
    "rows",
    ("row_id", "run_id"),
)


def _query_base_param_name(key: str) -> str:
    """Return the base query parameter name for diagnostic scrubbing."""
    return key.split("[", 1)[0].split(".", 1)[0]


def _safe_database_descriptor(connection_string: str) -> str:
    """Return a diagnostic database URL with credentials removed."""
    try:
        parsed = make_url(connection_string)
    except (ArgumentError, TypeError, ValueError):
        return "<unparseable database URL redacted>"

    safe_query: dict[str, str | tuple[str, ...]] = {}
    for key, value in parsed.query.items():
        base_key = _query_base_param_name(key).lower()
        if base_key in SENSITIVE_PARAMS:
            continue
        if base_key == "odbc_connect":
            values = value if isinstance(value, tuple) else (value,)
            scrubbed_values = tuple(_scrub_odbc_connect_value(connect_value)[0] for connect_value in values)
            safe_query[key] = scrubbed_values if isinstance(value, tuple) else scrubbed_values[0]
            continue
        safe_query[key] = value

    return parsed.set(query=safe_query).render_as_string(hide_password=True)


# StaticPool engines (``LandscapeDB.in_memory()``, tests only) share ONE DBAPI
# connection across every thread (``check_same_thread=False`` + StaticPool).  When
# a helper thread — e.g. the idle-timeout aggregation poller in
# ``source_iteration.py`` — drives audit writes concurrently with the main source
# thread (which may itself write audit rows via ``ctx.record_call`` during
# ``next()``), both would drive that single shared connection at once and SQLite
# raises "recursive use of cursors not allowed" / "cannot start a transaction
# within a transaction".  File-backed production engines use the default
# ``QueuePool`` (one connection PER thread) + WAL/BEGIN IMMEDIATE/busy_timeout, so
# they are already safe and MUST NOT pay any app-level lock (it would regress the
# epoch-21 multi-writer design).  We therefore serialize connection acquisition
# with a per-engine reentrant lock that engages ONLY for StaticPool engines; every
# other engine takes the no-op (lock-free) path.
_SHARED_CONNECTION_LOCKS: "WeakKeyDictionary[Engine, threading.RLock]" = WeakKeyDictionary()
_SHARED_CONNECTION_LOCKS_GUARD = threading.Lock()


def _shared_connection_lock(engine: Engine) -> "threading.RLock | None":
    """Return a serialization lock for engines whose pool shares one DBAPI
    connection across threads (``StaticPool``); ``None`` for per-thread-connection
    engines (the production ``QueuePool`` path), which need no app-level lock.

    The lock is reentrant so a single thread that nests connection context
    managers (e.g. ``connection()`` calling into ``write_connection()``) does not
    self-deadlock, and per-engine so all callers sharing one StaticPool engine
    contend on the same lock.
    """
    if not isinstance(engine.pool, StaticPool):
        return None
    with _SHARED_CONNECTION_LOCKS_GUARD:
        lock = _SHARED_CONNECTION_LOCKS.get(engine)
        if lock is None:
            lock = threading.RLock()
            _SHARED_CONNECTION_LOCKS[engine] = lock
        return lock


@contextmanager
def _maybe_serialize_shared_connection(engine: Engine) -> Iterator[None]:
    """Hold the StaticPool serialization lock for the duration of a transaction,
    or do nothing on per-thread-connection (production) engines.
    """
    lock = _shared_connection_lock(engine)
    if lock is None:
        yield
        return
    with lock:
        yield


@contextmanager
def begin_write(engine: Engine) -> Iterator[Connection]:
    """Drop-in replacement for ``engine.begin()`` that carries write intent.

    On writable Landscape SQLite engines the transaction begins with
    ``BEGIN IMMEDIATE``, taking the WAL write lock at BEGIN so cross-process
    read-then-write transactions cannot abort with the non-retryable
    ``SQLITE_BUSY_SNAPSHOT`` (the lock is held from transaction start, so the
    snapshot can never go stale under a concurrent writer).  Under contention
    the BEGIN itself polls for up to ``busy_timeout`` (5000 ms) before raising
    a retryable ``OperationalError`` ("database is locked").

    Commit/rollback semantics are identical to ``engine.begin()``: commit on
    successful block exit, rollback on exception.  On non-SQLite engines the
    write-intent option is inert.

    Exists as a module-level helper because some callers hold a bare
    :class:`Engine` / :data:`Tier1Engine` rather than a :class:`LandscapeDB`
    (e.g. ``TokenSchedulerRepository``); LandscapeDB holders should prefer
    :meth:`LandscapeDB.write_connection`.
    """
    with _maybe_serialize_shared_connection(engine), engine.connect() as conn:
        conn.execution_options(**{WRITE_INTENT_OPTION: True})
        with conn.begin():
            yield conn


# Required columns that have been added since initial schema.
# Used by _validate_schema() to detect outdated SQLite databases.
_REQUIRED_COLUMNS: tuple[tuple[str, str], ...] = (
    # Web auth audit trail - records login, token, and auth failure events.
    ("auth_events", "event_id"),
    ("auth_events", "occurred_at"),
    ("auth_events", "event_type"),
    ("auth_events", "outcome"),
    ("auth_events", "provider"),
    ("auth_events", "metadata_json"),
    # Web run attribution - records the authenticated user that initiated a run.
    ("run_attributions", "run_id"),
    ("run_attributions", "recorded_at"),
    ("run_attributions", "initiated_by_user_id"),
    ("run_attributions", "auth_provider_type"),
    # Epoch 23: one optional, sanitized plugin-policy evidence row per web run.
    ("run_web_plugin_policy", "run_id"),
    ("run_web_plugin_policy", "schema_version"),
    ("run_web_plugin_policy", "policy_hash"),
    ("run_web_plugin_policy", "snapshot_hash"),
    ("run_web_plugin_policy", "authorized_plugin_ids_json"),
    ("run_web_plugin_policy", "available_plugin_ids_json"),
    ("run_web_plugin_policy", "control_modes_json"),
    ("run_web_plugin_policy", "selected_implementations_json"),
    ("run_web_plugin_policy", "selected_profile_aliases_json"),
    ("run_web_plugin_policy", "plugin_code_identities_json"),
    ("run_web_plugin_policy", "binding_generation_fingerprint"),
    ("run_web_plugin_policy", "decision_codes_json"),
    ("tokens", "expand_group_id"),
    # Added for run ownership — prevents cross-run contamination of token-linked records
    ("tokens", "run_id"),
    # Added for composite FK to nodes (node_id, run_id) - enables run-isolated queries
    ("node_states", "run_id"),
    # Source schema persists typed resume metadata; runtime always writes this on new runs
    ("runs", "source_schema_json"),
    # Field resolution audit trail - captures original→final header mapping
    ("runs", "source_field_resolution_json"),
    # Fork/expand branch contract - enables recovery validation
    ("token_outcomes", "expected_branches_json"),
    # Transform success reason audit trail - captures why transform succeeded
    ("node_states", "success_reason_json"),
    # Operation call linkage - enables source/sink call tracking
    ("calls", "operation_id"),
    # Phase 5: Plugin contract audit trail - captures input/output contracts per node
    ("nodes", "input_contract_json"),
    ("nodes", "output_contract_json"),
    # Operation I/O hashes - survive payload purge for integrity verification
    ("operations", "input_data_hash"),
    ("operations", "output_data_hash"),
    # Checkpoint compatibility gate - runtime always stamps the checkpoint format version
    ("checkpoints", "format_version"),
    # Epoch 20: F1 durability unification.
    # barrier_scalars_json carries shrunken checkpoint barrier metadata (Task 1.2).
    ("checkpoints", "barrier_scalars_json"),
    # Mechanical change detection — hash of plugin source file at registration
    ("nodes", "source_file_hash"),
    # ADR-010 §Decision 3 M3: runtime VAL manifest — set of
    # declaration contracts + Tier-1 error classes registered at bootstrap,
    # serialized as canonical JSON on the runs row.
    ("runs", "runtime_val_manifest_json"),
    # Quarantine lineage exactness - links validation errors to persisted rows
    ("validation_errors", "row_id"),
    # Batch membership run ownership - enables composite FK enforcement to both
    # batches and tokens so cross-run batch contamination fails at the database.
    ("batch_members", "run_id"),
    # Routing decisions must bind the chosen state and edge to the same audit run.
    ("routing_events", "run_id"),
    # Retry lineage exactness - retry_batch() must deduplicate per failed batch.
    ("batches", "retry_of_batch_id"),
    # ADR-019 two-axis terminal model: old is_terminal DBs must fail fast.
    ("token_outcomes", "completed"),
    ("token_outcomes", "path"),
    # Phase 5b interpretation-review audit anchor — runtime LLM calls must
    # carry the resolved prompt hash used to join back to session DB events.
    ("calls", "resolved_prompt_template_hash"),
    # Phase 4 tutorial audit-story projection fields.
    ("runs", "llm_call_count"),
    ("runs", "seeded_from_cache"),
    ("runs", "cache_key"),
    # Preflight results are written after run creation; partial stale tables
    # otherwise pass schema validation and fail later on insert.
    ("preflight_results", "result_id"),
    ("preflight_results", "run_id"),
    ("preflight_results", "result_type"),
    ("preflight_results", "name"),
    ("preflight_results", "result_json"),
    ("preflight_results", "created_at"),
    # Epoch 11: resume fork/expand/coalesce re-emit fix (F1).
    # token_data_ref persists per-token payloads for expand children + coalesce merged tokens.
    # resume_checkpoint_id marks resume re-drives so explain() can distinguish them from
    # run-1 tenacity retries (filters on resume_checkpoint_id IS NULL).
    ("tokens", "token_data_ref"),
    ("node_states", "resume_checkpoint_id"),
    # F3 co-fix: the OpenRouter catalog columns (epoch 10) were never added to the
    # Postgres staleness backstop, so a stale Postgres DB would slip past validation.
    ("runs", "openrouter_catalog_sha256"),
    ("runs", "openrouter_catalog_source"),
    # Epoch 12: multi-source foundation.
    ("run_sources", "run_id"),
    ("run_sources", "source_node_id"),
    ("run_sources", "source_name"),
    ("run_sources", "lifecycle_state"),
    ("rows", "source_row_index"),
    ("rows", "ingest_sequence"),
    ("token_work_items", "work_item_id"),
    ("token_work_items", "run_id"),
    ("token_work_items", "token_id"),
    ("token_work_items", "row_id"),
    ("token_work_items", "node_id"),
    ("token_work_items", "step_index"),
    ("token_work_items", "ingest_sequence"),
    ("token_work_items", "status"),
    ("token_work_items", "queue_key"),
    ("token_work_items", "barrier_key"),
    ("token_work_items", "available_at"),
    ("token_work_items", "row_payload_json"),
    ("token_work_items", "on_success_sink"),
    ("token_work_items", "pending_sink_name"),
    ("token_work_items", "pending_outcome"),
    ("token_work_items", "pending_path"),
    ("token_work_items", "pending_error_hash"),
    ("token_work_items", "pending_error_message"),
    ("token_work_items", "branch_name"),
    ("token_work_items", "fork_group_id"),
    ("token_work_items", "join_group_id"),
    ("token_work_items", "expand_group_id"),
    ("token_work_items", "coalesce_node_id"),
    ("token_work_items", "coalesce_name"),
    ("token_work_items", "attempt"),
    ("token_work_items", "lease_owner"),
    ("token_work_items", "lease_expires_at"),
    ("token_work_items", "created_at"),
    ("token_work_items", "updated_at"),
    # Epoch 20: F1 durability unification.
    # barrier_blocked_at records when a work item was blocked at a barrier (Task 1.3).
    ("token_work_items", "barrier_blocked_at"),
    ("scheduler_events", "event_id"),
    ("scheduler_events", "run_id"),
    ("scheduler_events", "token_id"),
    ("scheduler_events", "work_item_id"),
    ("scheduler_events", "node_id"),
    ("scheduler_events", "event_type"),
    ("scheduler_events", "from_status"),
    ("scheduler_events", "to_status"),
    ("scheduler_events", "from_lease_owner"),
    ("scheduler_events", "to_lease_owner"),
    ("scheduler_events", "from_lease_expires_at"),
    ("scheduler_events", "to_lease_expires_at"),
    ("scheduler_events", "from_attempt"),
    ("scheduler_events", "to_attempt"),
    ("scheduler_events", "recorded_at"),
    ("scheduler_events", "caller_owner"),
    ("scheduler_events", "context_json"),
    # Epoch 21: multi-worker coordination substrate (ADR-030).
    ("token_work_items", "barrier_adopted_epoch"),
    ("run_coordination", "run_id"),
    ("run_coordination", "leader_worker_id"),
    ("run_coordination", "leader_epoch"),
    ("run_coordination", "leader_heartbeat_expires_at"),
    ("run_coordination", "updated_at"),
    ("run_workers", "worker_id"),
    ("run_workers", "run_id"),
    ("run_workers", "role"),
    ("run_workers", "status"),
    ("run_workers", "registered_at"),
    ("run_workers", "heartbeat_expires_at"),
    ("run_workers", "departed_at"),
    ("run_workers", "evicted_at"),
    ("run_workers", "evicted_by_worker_id"),
    ("run_workers", "pid"),
    ("run_workers", "hostname"),
    ("run_workers", "entry_point"),
    ("run_coordination_events", "seq"),
    ("run_coordination_events", "event_id"),
    ("run_coordination_events", "run_id"),
    ("run_coordination_events", "event_type"),
    ("run_coordination_events", "worker_id"),
    ("run_coordination_events", "leader_epoch"),
    ("run_coordination_events", "recorded_at"),
    ("run_coordination_events", "context_json"),
    ("coalesce_branch_losses", "loss_id"),
    ("coalesce_branch_losses", "run_id"),
    ("coalesce_branch_losses", "coalesce_name"),
    ("coalesce_branch_losses", "row_id"),
    ("coalesce_branch_losses", "branch_name"),
    ("coalesce_branch_losses", "token_id"),
    ("coalesce_branch_losses", "reason"),
    ("coalesce_branch_losses", "recorded_by"),
    ("coalesce_branch_losses", "recorded_at"),
    ("coalesce_branch_losses", "adopted_epoch"),
)

_EPOCH_26_REQUIRED_TABLES = (
    "sink_effect_streams",
    "sink_effects",
    "sink_effect_members",
    "sink_effect_attempts",
    "audit_export_snapshots",
    "audit_export_snapshot_chunks",
    "sink_effect_export_snapshots",
)
_REQUIRED_COLUMNS += (
    *((table_name, column.name) for table_name in _EPOCH_26_REQUIRED_TABLES for column in metadata.tables[table_name].columns),
    ("operations", "sink_effect_id"),
    ("artifacts", "sink_effect_id"),
    ("artifacts", "publication_performed"),
    ("artifacts", "publication_evidence_kind"),
)

_EPOCH_27_REQUIRED_TABLES = ("coalesce_effects", "coalesce_effect_members")
_REQUIRED_COLUMNS += tuple(
    (table_name, column.name) for table_name in _EPOCH_27_REQUIRED_TABLES for column in metadata.tables[table_name].columns
)

# Required foreign keys for audit integrity (Tier 1 trust).
# Format: (table_name, column_name, referenced_table)
# Use this only for exact single-column contracts. Run-scoped contracts belong in
# _REQUIRED_COMPOSITE_FOREIGN_KEYS so stale single-column FKs cannot satisfy them.
_REQUIRED_FOREIGN_KEYS: tuple[tuple[str, str, str], ...] = (
    ("run_web_plugin_policy", "run_id", "runs"),
    ("validation_errors", "row_id", "rows"),
    ("preflight_results", "run_id", "runs"),
    ("scheduler_events", "run_id", "runs"),
    # Epoch 21: multi-worker coordination substrate (ADR-030).
    ("run_coordination", "run_id", "runs"),
    ("run_workers", "run_id", "runs"),
    ("run_coordination_events", "run_id", "runs"),
    ("coalesce_branch_losses", "run_id", "runs"),
    ("operations", "sink_effect_id", "sink_effects"),
    ("audit_export_snapshot_chunks", "snapshot_id", "audit_export_snapshots"),
    ("sink_effect_export_snapshots", "snapshot_id", "audit_export_snapshots"),
    ("sink_effect_attempts", "effect_id", "sink_effects"),
    ("coalesce_effects", "run_id", "runs"),
)

# Required composite foreign keys for run-scoped audit integrity.
# Format: (table_name, constrained_columns, referenced_table, referenced_columns)
_REQUIRED_COMPOSITE_FOREIGN_KEYS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    # Epoch 24: a token's run is derived from, and must match, its row.
    _EPOCH_24_TOKEN_ROW_RUN_FK,
    ("token_outcomes", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_outcomes", ("batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("node_states", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("node_states", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("validation_errors", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("transform_errors", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("transform_errors", ("transform_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("artifacts", ("produced_by_state_id", "run_id"), "node_states", ("state_id", "run_id")),
    ("artifacts", ("sink_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("routing_events", ("state_id", "run_id"), "node_states", ("state_id", "run_id")),
    ("routing_events", ("edge_id", "run_id"), "edges", ("edge_id", "run_id")),
    ("run_sources", ("source_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("batches", ("aggregation_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("batches", ("aggregation_state_id", "run_id"), "node_states", ("state_id", "run_id")),
    ("batches", ("retry_of_batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("batch_members", ("batch_id", "run_id"), "batches", ("batch_id", "run_id")),
    ("batch_members", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_work_items", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("token_work_items", ("row_id", "run_id"), "rows", ("row_id", "run_id")),
    ("token_work_items", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("token_work_items", ("coalesce_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("scheduler_events", ("token_id", "run_id"), "tokens", ("token_id", "run_id")),
    ("scheduler_events", ("node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("artifacts", ("sink_effect_id", "run_id", "sink_node_id"), "sink_effects", ("effect_id", "run_id", "sink_node_id")),
    ("sink_effects", ("sink_node_id", "run_id"), "nodes", ("node_id", "run_id")),
    ("sink_effect_members", ("token_id", "row_id", "run_id"), "tokens", ("token_id", "row_id", "run_id")),
    ("sink_effect_members", ("effect_id", "input_kind"), "sink_effects", ("effect_id", "input_kind")),
    ("sink_effect_members", ("primary_effect_id", "run_id"), "sink_effects", ("effect_id", "run_id")),
    (
        "audit_export_snapshots",
        ("source_run_id", "source_status", "source_completed_at"),
        "runs",
        ("run_id", "status", "completed_at"),
    ),
    ("sink_effect_export_snapshots", ("effect_id", "input_kind"), "sink_effects", ("effect_id", "input_kind")),
    ("coalesce_effects", ("row_id", "run_id"), "rows", ("row_id", "run_id")),
    (
        "coalesce_effects",
        ("result_token_id", "run_id", "result_join_group_id"),
        "tokens",
        ("token_id", "run_id", "join_group_id"),
    ),
    (
        "coalesce_effect_members",
        ("effect_id", "run_id"),
        "coalesce_effects",
        ("effect_id", "run_id"),
    ),
    (
        "coalesce_effect_members",
        ("parent_token_id", "run_id"),
        "tokens",
        ("token_id", "run_id"),
    ),
    (
        "coalesce_effect_members",
        ("parent_state_id", "run_id", "parent_token_id"),
        "node_states",
        ("state_id", "run_id", "token_id"),
    ),
)

# Foreign keys that belonged to older schema shapes but are incompatible with
# current runtime semantics. Exact matches must fail startup validation.
_FORBIDDEN_FOREIGN_KEYS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...], str], ...] = (
    (
        "node_states",
        ("resume_checkpoint_id",),
        "checkpoints",
        ("checkpoint_id",),
        "resume_checkpoint_id is marker-only; checkpoints are deletable progress state",
    ),
)

# Required check constraints for audit integrity.
# Format: (table_name, constraint_name)
_REQUIRED_CHECK_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("auth_events", "ck_auth_events_event_type"),
    ("auth_events", "ck_auth_events_outcome"),
    ("auth_events", "ck_auth_events_provider"),
    ("run_attributions", "ck_run_attributions_auth_provider_type"),
    ("run_web_plugin_policy", "ck_run_web_plugin_policy_schema_version"),
    ("run_sources", "ck_run_sources_lifecycle_state"),
    ("token_work_items", "ck_token_work_items_lease_owner_required_when_leased"),
    ("scheduler_events", "ck_scheduler_events_event_type"),
    ("scheduler_events", "ck_scheduler_events_from_status"),
    ("scheduler_events", "ck_scheduler_events_to_status"),
    ("scheduler_events", "ck_scheduler_events_from_attempt_non_negative"),
    ("scheduler_events", "ck_scheduler_events_to_attempt_non_negative"),
    ("calls", "calls_has_parent"),
    ("preflight_results", "ck_preflight_result_type"),
    ("runs", "ck_runs_openrouter_catalog_source"),
    # Epoch 21: multi-worker coordination substrate (ADR-030).
    ("run_coordination", "ck_run_coordination_seat_liveness_paired"),
    ("run_workers", "ck_run_workers_role"),
    ("run_workers", "ck_run_workers_status"),
    ("run_workers", "ck_run_workers_evicted_at_paired"),
    ("run_coordination_events", "ck_run_coordination_events_event_type"),
    ("sink_effect_streams", "ck_sink_effect_streams_role"),
    ("sink_effect_streams", "ck_sink_effect_streams_next_sequence"),
    ("sink_effects", "ck_sink_effects_role"),
    ("sink_effects", "ck_sink_effects_state"),
    ("sink_effects", "ck_sink_effects_input_kind_xor"),
    ("sink_effects", "ck_sink_effects_lifecycle"),
    ("sink_effects", "ck_sink_effects_generation"),
    ("sink_effects", "ck_sink_effects_lease_window"),
    ("sink_effects", "ck_sink_effects_stream_shape"),
    ("sink_effects", "ck_sink_effects_descriptor_mode"),
    ("sink_effects", "ck_sink_effects_inspection_mode"),
    ("sink_effects", "ck_sink_effects_reconcile_kind"),
    ("sink_effect_members", "ck_sink_effect_members_input_kind"),
    ("sink_effect_members", "ck_sink_effect_members_order"),
    ("sink_effect_members", "ck_sink_effect_members_primary_linkage"),
    ("sink_effect_members", "ck_sink_effect_members_disposition"),
    ("sink_effect_members", "ck_sink_effect_members_state"),
    ("sink_effect_export_snapshots", "ck_sink_effect_export_snapshots_input_kind"),
    ("sink_effect_export_snapshots", "ck_sink_effect_export_snapshots_slot"),
    ("sink_effect_attempts", "ck_sink_effect_attempts_generation"),
    ("sink_effect_attempts", "ck_sink_effect_attempts_action"),
    ("sink_effect_attempts", "ck_sink_effect_attempts_state"),
    ("operations", "ck_operations_sink_effect_type"),
    ("artifacts", "ck_artifacts_producer_xor"),
    ("artifacts", "ck_artifacts_publication_evidence_kind"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_terminal_witness"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_positive_totals"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_terminal_ordinal"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_manifest_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_snapshot_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_snapshot_seal_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_last_chunk_seal_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_final_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_signed_manifest_hash_hex"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_signed_manifest_ref"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_signed_manifest_size"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_manifest_schema"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_derivation_version"),
    ("audit_export_snapshots", "ck_audit_export_snapshots_signing_tuple"),
    ("audit_export_snapshot_chunks", "ck_audit_export_snapshot_chunks_content_ref"),
    ("coalesce_effects", "ck_coalesce_effects_lifecycle"),
    ("coalesce_effects", "ck_coalesce_effects_parent_set_hash_hex"),
    ("coalesce_effects", "ck_coalesce_effects_effect_hash_hex"),
    ("coalesce_effects", "ck_coalesce_effects_payload_ref_hex"),
    ("coalesce_effect_members", "ck_coalesce_effect_members_ordinal"),
)

# Required indexes (including partial unique indexes) for audit integrity.
# Format: (table_name, index_name)
_REQUIRED_INDEXES: tuple[tuple[str, str], ...] = (
    ("auth_events", "ix_auth_events_occurred_at"),
    ("auth_events", "ix_auth_events_type_outcome"),
    ("auth_events", "ix_auth_events_user"),
    ("run_attributions", "ix_run_attributions_user"),
    ("calls", "ix_calls_state_call_index_unique"),
    ("calls", "ix_calls_operation_call_index_unique"),
    ("calls", "ix_calls_resolved_prompt_template_hash"),
    ("checkpoints", "ix_checkpoints_run_sequence_unique"),
    ("preflight_results", "ix_preflight_results_run"),
    ("token_outcomes", "ix_token_outcomes_terminal_unique"),
    ("token_work_items", "ix_token_work_items_ready"),
    ("token_work_items", "ix_token_work_items_lease"),
    ("token_work_items", "ix_token_work_items_recovery"),
    ("token_work_items", "ix_token_work_items_pending_sink_token"),
    ("token_work_items", "uq_token_work_items_terminal_identity"),
    ("scheduler_events", "ix_scheduler_events_run_token_time"),
    ("scheduler_events", "ix_scheduler_events_work_item"),
    ("validation_errors", "ix_validation_errors_run_row"),
    ("artifacts", "uq_artifacts_run_idempotency_key"),
    # Epoch 21: multi-worker coordination substrate (ADR-030).
    ("run_workers", "ix_run_workers_liveness"),
    ("run_coordination_events", "uq_run_coordination_events_event_id"),
    ("run_coordination_events", "ix_run_coordination_events_run"),
    ("coalesce_branch_losses", "uq_coalesce_branch_losses_natural"),
    ("runs", "uq_runs_export_witness"),
    ("tokens", "uq_tokens_identity_row_run"),
    ("tokens", "uq_tokens_coalesce_result_identity"),
    ("node_states", "uq_node_states_coalesce_member_identity"),
    ("operations", "uq_operations_sink_effect_id"),
    ("sink_effect_streams", "uq_sink_effect_stream_identity"),
    ("sink_effect_members", "uq_sink_effect_member_binding"),
    ("audit_export_snapshots", "uq_audit_export_snapshots_registry_key"),
    ("audit_export_snapshots", "ix_audit_export_snapshots_registry_key_hash"),
    ("audit_export_snapshot_chunks", "uq_audit_export_snapshot_chunks_terminal"),
)

_REQUIRED_TRIGGERS: tuple[str, ...] = (
    "trg_audit_export_chunk_insert_validate",
    "trg_audit_export_snapshot_insert_seal",
    "trg_audit_export_snapshot_immutable",
    "trg_audit_export_snapshot_immutable_delete",
    "trg_audit_export_chunk_immutable",
    "trg_audit_export_chunk_immutable_delete",
)

_ADDITIVE_INDEX_OWNERS: Mapping[str, str] = MappingProxyType({"ix_tokens_run_id": "tokens"})
_ADDITIVE_INDEX_NAMES: frozenset[str] = frozenset(_ADDITIVE_INDEX_OWNERS)
_ADDITIVE_TABLE_NAMES: frozenset[str] = frozenset({"auth_events", "run_attributions"})


class LandscapeSchemaShape(Enum):
    """The non-mutating structural state of a Landscape target."""

    EMPTY = "empty"
    FOREIGN = "foreign"
    INCOMPLETE = "incomplete"
    DIVERGENT = "divergent"
    MATCHES = "matches"


def _sqlite_epoch_is_incompatible(bind: Engine | Connection) -> bool:
    """Return whether a SQLite target carries a non-current, non-zero epoch."""
    if bind.dialect.name != "sqlite":
        return False
    if isinstance(bind, Connection):
        epoch = int(bind.exec_driver_sql("PRAGMA user_version").scalar_one())
    else:
        with bind.connect() as conn:
            epoch = int(conn.exec_driver_sql("PRAGMA user_version").scalar_one())
    return epoch not in (0, SQLITE_SCHEMA_EPOCH)


def _landscape_identity_issue(
    bind: Engine | Connection,
    inspector: Inspector,
    existing_tables: set[str],
    *,
    schema_epoch: int = SQLITE_SCHEMA_EPOCH,
    identity_required: bool = True,
) -> SchemaIdentityMismatch | Literal["identity_table", "identity_shape"] | None:
    """Return a static issue code for cross-dialect Landscape identity drift."""
    if SCHEMA_IDENTITY_TABLE_NAME not in existing_tables:
        other_landscape_tables = existing_tables.intersection(set(metadata.tables) - {SCHEMA_IDENTITY_TABLE_NAME})
        return "identity_table" if identity_required and other_landscape_tables else None

    columns = {str(column["name"]) for column in inspector.get_columns(SCHEMA_IDENTITY_TABLE_NAME)}
    if columns != {"singleton_id", "application_id", "store_kind", "schema_epoch"}:
        return "identity_shape"

    if isinstance(bind, Connection):
        rows = read_schema_identities(bind, schema_identity_table)
    else:
        with bind.connect() as connection:
            rows = read_schema_identities(connection, schema_identity_table)
    return schema_identity_mismatch(rows, store_kind="landscape", schema_epoch=schema_epoch)


def _missing_additive_indexes(inspector: Inspector, present_tables: set[str]) -> frozenset[str]:
    missing: set[str] = set()
    for index_name, table_name in _ADDITIVE_INDEX_OWNERS.items():
        if table_name not in present_tables:
            missing.add(index_name)
            continue
        found = {str(index["name"]) for index in inspector.get_indexes(table_name) if index.get("name") is not None}
        if index_name not in found:
            missing.add(index_name)
    return frozenset(missing)


def probe_schema_shape(bind: Engine | Connection) -> LandscapeSchemaShape:
    """Classify a Landscape schema without creating or altering objects."""
    from sqlalchemy import inspect

    inspector = inspect(bind)
    existing = set(inspector.get_table_names())
    expected = set(metadata.tables)

    if _sqlite_epoch_is_incompatible(bind):
        return LandscapeSchemaShape.DIVERGENT
    if not existing:
        return LandscapeSchemaShape.EMPTY
    if _landscape_identity_issue(bind, inspector, existing) is not None:
        return LandscapeSchemaShape.DIVERGENT
    if existing - expected:
        return LandscapeSchemaShape.FOREIGN

    present = existing & expected
    if not present:
        return LandscapeSchemaShape.FOREIGN

    issues = collect_metadata_shape_issues(
        inspector,
        metadata,
        dialect=bind.dialect,
        present_tables=present,
        allowed_missing_index_names=_ADDITIVE_INDEX_NAMES,
    )
    if issues:
        return LandscapeSchemaShape.DIVERGENT

    missing_tables = expected - existing
    if missing_tables - _ADDITIVE_TABLE_NAMES:
        return LandscapeSchemaShape.DIVERGENT

    if missing_tables or _missing_additive_indexes(inspector, present):
        return LandscapeSchemaShape.INCOMPLETE
    return LandscapeSchemaShape.MATCHES


def create_additive_indexes(bind: Engine | Connection) -> None:
    """Create explicitly additive Landscape indexes on existing tables."""
    for table in metadata.tables.values():
        for index in table.indexes:
            if index.name in _ADDITIVE_INDEX_NAMES:
                index.create(bind, checkfirst=True)


def _collect_missing_required_columns(inspector: Inspector) -> list[tuple[str, str]]:
    """Return required columns missing from existing tables."""
    existing_tables = set(inspector.get_table_names())
    missing: list[tuple[str, str]] = []
    for table_name, column_name in _REQUIRED_COLUMNS:
        if table_name not in existing_tables:
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            missing.append((table_name, column_name))
    return missing


def _collect_token_outcomes_shape_errors(
    inspector: Inspector,
    *,
    engine: Engine | None = None,
    inspect_sqlite_indexes: bool = False,
) -> list[str]:
    """Return ADR-019 shape errors for existing token_outcomes tables."""
    existing_tables = set(inspector.get_table_names())
    if "token_outcomes" not in existing_tables:
        return []

    columns = {column["name"]: column for column in inspector.get_columns("token_outcomes")}
    errors: list[str] = []

    if "is_terminal" in columns:
        errors.append("token_outcomes.is_terminal is stale; ADR-019 uses completed")
    if "outcome" in columns and columns["outcome"]["nullable"] is False:
        errors.append("token_outcomes.outcome nullable shape is stale; ADR-019 requires nullable outcome for BUFFERED rows")

    if inspect_sqlite_indexes and engine is not None:
        with engine.connect() as conn:
            index_sql = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'ix_token_outcomes_terminal_unique'"
            ).scalar_one_or_none()
        if index_sql is not None and "is_terminal" in str(index_sql).lower():
            errors.append("token_outcomes stale terminal index predicate references is_terminal; ADR-019 uses completed")

    return errors


class LandscapeDB:
    """Landscape database connection manager."""

    def __init__(
        self,
        connection_string: str,
        *,
        passphrase: str | None = None,
        dump_to_jsonl: bool = False,
        dump_to_jsonl_path: str | None = None,
        dump_to_jsonl_fail_on_error: bool = False,
        dump_to_jsonl_include_payloads: bool = False,
        dump_to_jsonl_payload_base_path: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                e.g., "sqlite:///./data/audit.db"
                      "postgresql://user@host/dbname"
            passphrase: SQLCipher encryption passphrase. When provided, the
                database is opened with AES-256 encryption via sqlcipher3.
                The passphrase is never stored in the URL or audit trail.
            dump_to_jsonl: Enable JSONL change journal for emergency backups
            dump_to_jsonl_path: Optional override path for JSONL journal
            dump_to_jsonl_fail_on_error: Fail if journal write fails
            dump_to_jsonl_include_payloads: Inline payloads in journal records
            dump_to_jsonl_payload_base_path: Payload store base path for inlining
        """
        self.connection_string = connection_string
        self._passphrase = passphrase
        self._engine: Engine | None = None
        self._journal: LandscapeJournal | None = None
        self._require_existing_schema = False
        self._read_only = False
        if dump_to_jsonl:
            journal_path = self._resolve_journal_path(
                connection_string,
                explicit_path=dump_to_jsonl_path,
            )
            self._journal = LandscapeJournal(
                journal_path,
                fail_on_error=dump_to_jsonl_fail_on_error,
                include_payloads=dump_to_jsonl_include_payloads,
                payload_base_path=dump_to_jsonl_payload_base_path,
            )
        self._setup_engine(**engine_kwargs)
        self._validate_schema()  # Check BEFORE create_tables
        # Pre-1.0 schema changes are deliberate recreate boundaries. Never
        # transform a populated older epoch in place: validation rejects it
        # with delete/recreate guidance; only a fresh/unstamped schema reaches
        # creation and current-epoch stamping below.
        self._sync_sqlite_schema_epoch()
        self._create_tables()
        self._create_additive_indexes()
        self._sync_schema_identity()
        self._sync_sqlite_schema_epoch()

    def _setup_engine(self, **engine_kwargs: Any) -> None:
        """Create and configure the database engine."""
        if self._passphrase is not None:
            if engine_kwargs:
                raise ValueError("SQLCipher construction does not accept SQLAlchemy engine kwargs")
            self._engine = self._create_sqlcipher_engine(self.connection_string, self._passphrase)
            LandscapeDB._configure_sqlite(self._engine)
        else:
            self._engine = create_engine(
                self.connection_string,
                echo=False,  # Set True for SQL debugging
                **engine_kwargs,
            )
            # SQLite-specific configuration
            if self.connection_string.startswith("sqlite"):
                LandscapeDB._configure_sqlite(self._engine)
        if self._journal is not None:
            self._journal.attach(self._engine)
        # Tier-1: probe the SQLite PRAGMAs we just configured — if any
        # didn't take effect, the audit DB does not meet the durability /
        # concurrency contract and we MUST refuse to open it.  Skipped for
        # non-SQLite backends (PostgreSQL has no equivalent surface here).
        if self._engine is not None and self.connection_string.startswith("sqlite"):
            LandscapeDB._verify_sqlite_pragmas(self._engine, self.connection_string)

    @staticmethod
    def _configure_sqlite(engine: Engine, *, read_only: bool = False) -> None:
        """Configure SQLite engine for reliability.

        Registers a connection event hook that sets:
        - PRAGMA journal_mode=WAL (better concurrency, writable engines only)
        - PRAGMA synchronous=NORMAL (canonical WAL crash-safety shape)
        - PRAGMA foreign_keys=ON (referential integrity)
        - PRAGMA busy_timeout=5000 (contention tolerance)
        - PRAGMA query_only=ON (read-only engines only)

        For writable engines it additionally takes manual control of
        transaction begin (pysqlite ``isolation_level = None`` plus an
        engine-level ``begin`` listener) so that transactions carrying the
        :data:`WRITE_INTENT_OPTION` execution option begin with
        ``BEGIN IMMEDIATE`` and all others with a plain DEFERRED ``BEGIN``.
        Read-only engines keep stock pysqlite autocommit-read behaviour and
        never emit any BEGIN.

        For SQLCipher engines, these PRAGMAs execute AFTER the creator callback
        returns (where PRAGMA key is issued), preserving the required ordering.

        The values set here MUST stay aligned with the invariants enforced by
        :meth:`_verify_sqlite_pragmas`; the probe is the audit-integrity gate
        that fails the database open if SQLite ever refuses to honour them.

        Args:
            engine: SQLAlchemy Engine to configure
        """

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]  # SQLAlchemy event passes DBAPI connection (has .cursor()) typed as object
            if read_only:
                cursor.execute("PRAGMA query_only=ON")
            else:
                # Enable WAL mode for better concurrency on writer-capable DBs.
                cursor.execute("PRAGMA journal_mode=WAL")
                # synchronous=NORMAL is the canonical pairing with WAL: durable
                # across app crashes, sufficiently durable across OS crashes,
                # without the per-write fsync overhead of FULL.
                cursor.execute("PRAGMA synchronous=NORMAL")
            # Enable foreign key enforcement
            cursor.execute("PRAGMA foreign_keys=ON")
            # Set busy timeout to avoid immediate SQLITE_BUSY errors under contention
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        if read_only:
            # Read-only engines keep stock pysqlite behaviour: SELECTs run in
            # autocommit and no BEGIN of any kind is ever emitted, so
            # dashboards / inspectors provably never contend for the WAL
            # write lock (option-c design F10 closure).
            return

        @event.listens_for(engine, "connect")
        def _disable_pysqlite_implicit_begin(dbapi_connection: object, connection_record: object) -> None:
            # pysqlite emits its own lazy BEGIN before the first DML and never
            # before SELECT.  Take manual control of transaction start so the
            # engine-level "begin" event below is the single place the begin
            # mode (DEFERRED vs IMMEDIATE) is decided — the documented
            # SQLAlchemy pysqlite recipe.  Without this, pysqlite would emit
            # its own BEGIN after our BEGIN IMMEDIATE ("cannot start a
            # transaction within a transaction").  The takeover is global per
            # writable engine rather than toggled per-transaction: a pooled
            # connection can never be checked in with surprising isolation
            # state, so ``engine.begin()`` rollback always undoes audit
            # writes.  sqlcipher3 mirrors the pysqlite ``isolation_level``
            # attribute, and PRAGMA key runs in the creator callback BEFORE
            # this event, so SQLCipher ordering is preserved.
            dbapi_connection.isolation_level = None  # type: ignore[attr-defined]  # SQLAlchemy event passes DBAPI connection typed as object

        @event.listens_for(engine, "begin")
        def _emit_begin(conn: Connection) -> None:
            # Belt: _configure_sqlite is only ever called for sqlite URLs, so
            # this listener is dialect-keyed by construction; guard anyway.
            if conn.dialect.name != "sqlite":
                return
            if conn.get_execution_options().get(WRITE_INTENT_OPTION, False):
                # Write intent declared (begin_write()/write_connection()):
                # take the WAL write lock at BEGIN so read-then-write
                # transactions cannot hit SQLITE_BUSY_SNAPSHOT mid-flight.
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                # Plain transactions keep DEFERRED semantics.  Owned
                # behaviour delta vs stock pysqlite: read transactions now
                # emit an explicit (lock-free) BEGIN where the driver
                # previously emitted nothing, making multi-SELECT blocks
                # snapshot-consistent.
                conn.exec_driver_sql("BEGIN")

    @staticmethod
    def _verify_sqlite_pragmas(engine: Engine, connection_string: str) -> None:
        """Probe SQLite to confirm the configured PRAGMAs actually took effect.

        Opens a connection (which triggers the ``connect`` event hook that
        applies the PRAGMAs), then reads each PRAGMA back and asserts the
        value matches the invariant.  Any mismatch raises
        :class:`AuditIntegrityError` — Tier-1 doctrine: the audit DB cannot
        proceed with weaker durability/concurrency guarantees than the audit
        subsystem was designed against.

        SQLite silently downgrades ``journal_mode`` to ``memory`` for
        ``:memory:`` databases (WAL requires an on-disk journal file); the
        probe selects the file-backed or memory-backed invariant set from
        the connection string.

        Args:
            engine: SQLAlchemy Engine configured by :meth:`_configure_sqlite`.
            connection_string: Original connection string, used to decide
                whether ``journal_mode=memory`` is acceptable.

        Raises:
            AuditIntegrityError: If any PRAGMA reports a value other than
                what :meth:`_configure_sqlite` requested.  The message names
                the PRAGMA, the expected value, and the observed value so
                operators can diagnose without needing to re-instrument.
        """
        # Resolve invariants once — :memory: is a single sanctioned deviation.
        url = make_url(connection_string)
        is_memory = url.database is None or url.database == ":memory:"
        invariants = _SQLITE_PRAGMA_INVARIANTS_MEMORY if is_memory else _SQLITE_PRAGMA_INVARIANTS_FILE

        with engine.connect() as conn:
            observed: dict[str, str] = {}
            for pragma, _expected in invariants:
                # PRAGMA names are static (literal table above), not user
                # input — f-string interpolation is safe here.
                result = conn.exec_driver_sql(f"PRAGMA {pragma}").scalar_one_or_none()
                observed[pragma] = "" if result is None else str(result).lower()

        mismatches: list[str] = []
        for pragma, expected in invariants:
            actual = observed[pragma]
            if actual != expected.lower():
                mismatches.append(f"PRAGMA {pragma}: expected {expected!r}, observed {actual!r}")

        if mismatches:
            # Audit DB cannot proceed with degraded guarantees.  Tier-1:
            # raise immediately, do not attempt remediation.  The message
            # avoids the connection string (path may be sensitive) but
            # names every offending PRAGMA so the operator can act.
            raise AuditIntegrityError(
                "Landscape SQLite PRAGMA invariants violated at engine startup; audit-integrity guarantees unmet. " + "; ".join(mismatches)
            )

    @staticmethod
    def _create_sqlcipher_engine(url: str, passphrase: str, *, read_only: bool = False) -> Engine:
        """Create a SQLAlchemy engine backed by SQLCipher (AES-256 encryption).

        Uses the creator callback pattern to keep the passphrase out of the
        connection URL entirely (prevents leaks in logs, tracebacks, repr()).

        PRAGMA key MUST be the first statement on a new SQLCipher connection.
        The creator issues it before returning, so SQLAlchemy's "connect" event
        (used by _configure_sqlite for WAL/FK/busy_timeout) fires afterwards.

        Args:
            url: SQLAlchemy SQLite URL (e.g., "sqlite:///./data/audit.db")
            passphrase: Encryption passphrase for PRAGMA key
            read_only: Open the encrypted SQLite file through a mode=ro URI.

        Returns:
            Configured SQLAlchemy Engine

        Raises:
            ImportError: If sqlcipher3 is not installed
            ValueError: If URL points to :memory: (SQLCipher requires a file)
        """
        try:
            import sqlcipher3
        except ImportError as exc:
            raise ImportError(
                "sqlcipher3 is required for encrypted audit databases. "
                "Install it with: uv pip install 'elspeth[security]'\n"
                "Note: requires libsqlcipher-dev system package."
            ) from exc

        parsed = make_url(url)

        # SQLCipher only works with SQLite — reject other backends early
        # to prevent silently opening a local file when the URL points elsewhere
        # (e.g., postgresql://host/db with ELSPETH_AUDIT_KEY set in env).
        driver = parsed.drivername.split("+")[0]  # "sqlite+aiosqlite" → "sqlite"
        if driver != "sqlite":
            raise ValueError(
                f"SQLCipher encryption requires a SQLite database URL, "
                f"got driver '{parsed.drivername}'. "
                f"Either remove the passphrase/encryption_key_env or change "
                f"the URL to sqlite:///path/to/audit.db"
            )

        db_path = parsed.database
        if db_path is None or db_path == ":memory:":
            raise ValueError("SQLCipher requires a file-backed database (cannot encrypt :memory:)")

        # Resolve relative paths the same way SQLite does
        resolved_path = str(Path(db_path).resolve())

        # Forward URL query params as connect kwargs (parity with non-encrypted
        # path, where create_engine extracts them automatically).  Coerce known
        # sqlite3.connect() params from their string URL representation.
        #
        # SQLite URI-style params (mode, cache, immutable, vfs) are NOT valid
        # connect() kwargs — they must be embedded in a file: URI when uri=True.
        _CONNECT_KWARGS = {"check_same_thread", "uri", "timeout", "detect_types", "cached_statements", "isolation_level", "factory"}
        # Match SQLAlchemy's pysqlite default: allow cross-thread usage so the
        # connection pool can hand connections to worker threads.  URL params
        # parsed below can still override this explicitly.
        connect_kwargs: dict[str, Any] = {"check_same_thread": False}
        uri_params: dict[str, str] = {}

        for key, raw_value in parsed.query.items():
            value = raw_value if isinstance(raw_value, str) else raw_value[0]
            if key in _CONNECT_KWARGS:
                if key in ("check_same_thread", "uri"):
                    connect_kwargs[key] = value.lower() in ("true", "1", "yes")
                elif key == "timeout":
                    connect_kwargs[key] = float(value)
                elif key in ("detect_types", "cached_statements"):
                    connect_kwargs[key] = int(value)
                else:
                    connect_kwargs[key] = value
            else:
                # URI-style param (mode, cache, immutable, vfs, etc.)
                uri_params[key] = value

        if read_only:
            uri_params["mode"] = "ro"
            uri_params.pop("immutable", None)

        # When URI params are present, build a file: URI and enable uri=True
        # so that SQLite interprets them via the URI interface.
        if uri_params:
            from urllib.parse import quote, urlencode

            file_uri = f"file:{quote(resolved_path, safe='/:')}?{urlencode(uri_params)}"
            connect_kwargs["uri"] = True
        else:
            file_uri = None

        def _creator() -> object:
            db = file_uri if file_uri is not None else resolved_path
            conn = sqlcipher3.connect(db, **connect_kwargs)
            # PRAGMA key MUST be the first statement — SQLCipher contract.
            # Escape double quotes in the passphrase (SQLite literal syntax:
            # a literal " inside a double-quoted string is written as "").
            # PRAGMA doesn't support parameter binding, so escaping is required.
            escaped = passphrase.replace('"', '""')
            conn.execute(f'PRAGMA key = "{escaped}"')
            return conn

        return create_engine("sqlite:///", creator=_creator, echo=False)

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        metadata.create_all(self.engine)

    def _create_additive_indexes(self) -> None:
        """Create non-gating performance indexes for existing schemas."""
        create_additive_indexes(self.engine)

    def _sync_schema_identity(self) -> None:
        """Stamp a freshly created store or verify its existing identity row."""
        with self.engine.connect() as connection:
            rows = read_schema_identities(connection, schema_identity_table)
        mismatch = schema_identity_mismatch(rows, store_kind="landscape", schema_epoch=SQLITE_SCHEMA_EPOCH)
        if rows and mismatch is None:
            return
        if rows:
            raise SchemaCompatibilityError(
                f"Landscape database schema identity mismatch ({mismatch}); "
                "uninstall this pre-1.0 deployment, delete/recreate the Landscape database, and reinstall."
            )

        # Re-read under write intent so concurrent fresh initializers cannot
        # both conclude that the singleton row is absent.
        with begin_write(self.engine) as conn:
            rows = read_schema_identities(conn, schema_identity_table)
            if not rows:
                insert_schema_identity(
                    conn,
                    schema_identity_table,
                    store_kind="landscape",
                    schema_epoch=SQLITE_SCHEMA_EPOCH,
                )
                return
            mismatch = schema_identity_mismatch(rows, store_kind="landscape", schema_epoch=SQLITE_SCHEMA_EPOCH)
        if mismatch is not None:
            raise SchemaCompatibilityError(
                f"Landscape database schema identity mismatch ({mismatch}); "
                "uninstall this pre-1.0 deployment, delete/recreate the Landscape database, and reinstall."
            )

    def _get_sqlite_schema_epoch(self) -> int:
        """Return SQLite schema epoch from PRAGMA user_version.

        Uses SQLite's built-in schema version slot as a lightweight marker for
        intentional pre-1.0 schema breaks. ELSPETH does not migrate between
        these epochs in place before 1.0.
        """
        if not self.connection_string.startswith("sqlite"):
            return 0

        with self.engine.connect() as conn:
            return int(conn.exec_driver_sql("PRAGMA user_version").scalar_one())

    def _set_sqlite_schema_epoch(self, epoch: int) -> None:
        """Persist the SQLite schema epoch in PRAGMA user_version.

        ``PRAGMA user_version`` is a transactional write in SQLite; carry
        write intent for uniformity with every other engine-side write.
        """
        if not self.connection_string.startswith("sqlite"):
            return

        with begin_write(self.engine) as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {int(epoch)}")

    def _sync_sqlite_schema_epoch(self) -> None:
        """Stamp compatible SQLite databases with the current schema epoch.

        New databases get the epoch immediately after create_all(). Existing
        compatible databases without an epoch are stamped in place. Any
        populated database carrying an older non-zero epoch is rejected at the
        one-way schema boundary. Call this only from schema-managing paths;
        read-only/inspection opens must not mutate the database.

        Raises:
            SchemaCompatibilityError: If the database has a non-zero epoch
                different from this code version's epoch.
        """
        if not self.connection_string.startswith("sqlite"):
            return

        current_epoch = self._get_sqlite_schema_epoch()
        if current_epoch > SQLITE_SCHEMA_EPOCH:
            raise SchemaCompatibilityError(
                "Cannot sync schema epoch: database has a newer epoch than this "
                "ELSPETH version supports.\n\n"
                f"Database epoch: {current_epoch}\n"
                f"Current epoch: {SQLITE_SCHEMA_EPOCH}\n\n"
                "This database was created or stamped by a newer ELSPETH version. "
                "Upgrade ELSPETH to open this database.\n\n"
                f"Database: {_safe_database_descriptor(self.connection_string)}"
            )
        if 0 < current_epoch < SQLITE_SCHEMA_EPOCH:
            from sqlalchemy import inspect

            present_landscape_tables = set(inspect(self.engine).get_table_names()) & set(metadata.tables)
            if present_landscape_tables:
                raise SchemaCompatibilityError(
                    "Cannot sync schema epoch: this existing Landscape database predates "
                    "the current one-way schema boundary. ELSPETH does not migrate it in place.\n\n"
                    f"Database epoch: {current_epoch}\n"
                    f"Current epoch: {SQLITE_SCHEMA_EPOCH}\n\n"
                    "Obtain archive/export approval where retention applies, then have the "
                    "database operator drop/recreate the Landscape database and initialize "
                    "a fresh schema. Rolling code back over the recreated database is unsafe.\n\n"
                    f"Database: {_safe_database_descriptor(self.connection_string)}"
                )

        if current_epoch == 0:
            from sqlalchemy import inspect

            # A genuinely fresh file is stamped only after create_all has
            # installed the complete current shape. Existing unstamped schemas
            # reach this point only after _validate_schema proved compatibility.
            if not (set(inspect(self.engine).get_table_names()) & set(metadata.tables)):
                return

        if current_epoch < SQLITE_SCHEMA_EPOCH:
            self._set_sqlite_schema_epoch(SQLITE_SCHEMA_EPOCH)

    def _validate_schema(self) -> None:
        """Validate that existing database has the required schema.

        For non-SQLite backends, validates table existence when
        _require_existing_schema is set (inspection callers like MCP/CLI).
        For SQLite, performs full validation of columns, foreign keys,
        check constraints, and indexes to catch stale local audit.db files.

        Raises:
            SchemaCompatibilityError: If database is missing required schema
                elements, or if an encrypted database is opened without the
                correct passphrase.

        Pre-1.0 callers validate only the current schema. Older schema epochs
        are recreate boundaries, not inputs to an in-place migration.
        """
        from sqlalchemy import inspect
        from sqlalchemy.exc import OperationalError

        try:
            inspector = inspect(self.engine)
            existing_tables = set(inspector.get_table_names())
        except OperationalError as e:
            error_msg = str(e)
            if self.connection_string.startswith("sqlite") and ("file is not a database" in error_msg or "file is encrypted" in error_msg):
                raise SchemaCompatibilityError(
                    "Cannot open Landscape database — file is encrypted or passphrase is incorrect.\n\n"
                    "If this is an encrypted (SQLCipher) database, ensure:\n"
                    "  1. The correct passphrase is set in the configured environment variable\n"
                    "     (landscape.encryption_key_env in settings.yaml, default: ELSPETH_AUDIT_KEY)\n"
                    "  2. backend: sqlcipher is set in settings.yaml\n\n"
                    f"Database: {_safe_database_descriptor(self.connection_string)}"
                ) from e
            raise
        validation_metadata = metadata
        expected_tables = set(validation_metadata.tables.keys())
        present_landscape_tables = existing_tables & expected_tables
        schema_epoch = self._get_sqlite_schema_epoch() if self.connection_string.startswith("sqlite") else 0
        identity_issue = (
            _landscape_identity_issue(
                self.engine,
                inspector,
                existing_tables,
                schema_epoch=SQLITE_SCHEMA_EPOCH,
                identity_required=SCHEMA_IDENTITY_TABLE_NAME in expected_tables,
            )
            if existing_tables
            else None
        )

        epoch_incompatible = schema_epoch not in (0, SQLITE_SCHEMA_EPOCH)
        if epoch_incompatible and not present_landscape_tables:
            raise SchemaCompatibilityError(
                "Landscape database schema is outdated.\n\n"
                f"schema epoch is incompatible:\nDatabase epoch: {schema_epoch}\nCurrent epoch: {SQLITE_SCHEMA_EPOCH}\n\n"
                f"Database: {_safe_database_descriptor(self.connection_string)}"
            )

        foreign_tables = sorted(existing_tables - expected_tables)
        if foreign_tables:
            raise SchemaCompatibilityError(
                "Landscape database contains foreign tables and cannot be opened as an ELSPETH audit database.\n\n"
                f"Unexpected tables: {', '.join(foreign_tables)}\n\n"
                f"Database: {_safe_database_descriptor(self.connection_string)}"
            )

        # If this looks like an existing Landscape database, all known tables must exist.
        # For brand-new DB files (no Landscape tables yet), creation happens in create_all().
        #
        # If _require_existing_schema is set (create_tables=False callers like MCP/CLI),
        # we require at least some Landscape tables to be present. An empty/non-Landscape
        # DB with create_tables=False would fail later with raw SQL errors — fail fast instead.
        if self._require_existing_schema and not present_landscape_tables:
            raise SchemaCompatibilityError(
                "Database does not contain any Landscape tables.\n\n"
                "This does not appear to be an ELSPETH audit database. "
                "Verify the database path is correct.\n\n"
                f"Database: {_safe_database_descriptor(self.connection_string)}"
            )
        allowed_missing_tables = frozenset() if self._require_existing_schema else _ADDITIVE_TABLE_NAMES
        missing_tables = sorted((expected_tables - existing_tables) - allowed_missing_tables) if present_landscape_tables else []

        # Some focused guard tests replace metadata with a name-only sentinel
        # so they can isolate the legacy high-signal diagnostics. Real
        # application metadata always contains SQLAlchemy Table objects.
        shape_issues = (
            collect_metadata_shape_issues(
                inspector,
                validation_metadata,
                dialect=self.engine.dialect,
                present_tables=present_landscape_tables,
                allowed_missing_index_names=_ADDITIVE_INDEX_NAMES,
            )
            if all(isinstance(table, Table) for table in validation_metadata.tables.values())
            else ()
        )

        # Full shape validation already covers every predecessor column. The
        # named current-column guard is retained for ordinary current-schema
        # validation, where it provides higher-signal diagnostics.
        missing_columns = _collect_missing_required_columns(inspector)
        token_outcomes_shape_errors = _collect_token_outcomes_shape_errors(
            inspector,
            engine=self.engine,
            inspect_sqlite_indexes=self.connection_string.startswith("sqlite"),
        )

        # Check for required foreign keys (Tier 1 audit integrity)
        missing_fks: list[tuple[str, str, str]] = []

        for table_name, column_name, referenced_table in _REQUIRED_FOREIGN_KEYS:
            # Check if table exists
            if table_name not in existing_tables:
                # Table will be created by create_all, skip
                continue

            # Check if FK exists AND targets the correct referenced table
            # SQLAlchemy inspector API guarantees constrained_columns and referred_table keys
            fks = inspector.get_foreign_keys(table_name)
            has_correct_fk = any(column_name in fk["constrained_columns"] and fk["referred_table"] == referenced_table for fk in fks)

            if not has_correct_fk:
                missing_fks.append((table_name, column_name, referenced_table))

        # Check for required composite foreign keys (Tier 1 audit integrity)
        missing_composite_fks: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []

        for table_name, constrained_columns, referenced_table, referenced_columns in _REQUIRED_COMPOSITE_FOREIGN_KEYS:
            if table_name not in existing_tables:
                continue

            fks = inspector.get_foreign_keys(table_name)
            has_correct_fk = any(
                tuple(fk["constrained_columns"]) == constrained_columns
                and fk["referred_table"] == referenced_table
                and tuple(fk["referred_columns"]) == referenced_columns
                for fk in fks
            )

            if not has_correct_fk:
                missing_composite_fks.append((table_name, constrained_columns, referenced_table, referenced_columns))

        forbidden_fks: list[tuple[str, tuple[str, ...], str, tuple[str, ...], str]] = []

        for table_name, constrained_columns, referenced_table, referenced_columns, reason in _FORBIDDEN_FOREIGN_KEYS:
            if table_name not in existing_tables:
                continue

            fks = inspector.get_foreign_keys(table_name)
            has_forbidden_fk = any(
                tuple(fk["constrained_columns"]) == constrained_columns
                and fk["referred_table"] == referenced_table
                and tuple(fk["referred_columns"]) == referenced_columns
                for fk in fks
            )

            if has_forbidden_fk:
                forbidden_fks.append((table_name, constrained_columns, referenced_table, referenced_columns, reason))

        # Check for required check constraints (Tier 1 audit integrity)
        missing_checks: list[tuple[str, str]] = []

        for table_name, constraint_name in _REQUIRED_CHECK_CONSTRAINTS:
            if table_name not in existing_tables:
                continue

            checks = inspector.get_check_constraints(table_name)
            has_constraint = any(c["name"] == constraint_name for c in checks)

            if not has_constraint:
                missing_checks.append((table_name, constraint_name))

        # Check for required indexes (Tier 1 audit integrity)
        missing_indexes: list[tuple[str, str]] = []

        for table_name, index_name in _REQUIRED_INDEXES:
            if table_name not in existing_tables:
                continue

            indexes = inspector.get_indexes(table_name)
            has_index = any(idx["name"] == index_name for idx in indexes)

            if not has_index:
                missing_indexes.append((table_name, index_name))

        # Triggers are physical integrity objects, not reflected as table
        # indexes/checks by SQLAlchemy. Require their exact stable names on a
        # current physical schema so an equivalent-looking table definition
        # cannot silently omit the seal/immutability enforcement.
        missing_triggers: list[str] = []
        should_validate_triggers = (
            bool(present_landscape_tables)
            and "audit_export_snapshot_chunks" in expected_tables
            and (self.engine.dialect.name != "sqlite" or schema_epoch == SQLITE_SCHEMA_EPOCH)
        )
        if should_validate_triggers:
            with self.engine.connect() as connection:
                if self.engine.dialect.name == "sqlite":
                    trigger_names = set(connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'trigger'").scalars())
                else:
                    trigger_names = set(
                        connection.exec_driver_sql(
                            "SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = current_schema()"
                        ).scalars()
                    )
            missing_triggers = sorted(set(_REQUIRED_TRIGGERS) - trigger_names)

        epoch_incompatible = bool(present_landscape_tables) and epoch_incompatible

        # Raise errors for missing columns, FKs, check constraints, indexes, or stale ADR-019 shapes.
        if (
            missing_tables
            or missing_columns
            or token_outcomes_shape_errors
            or missing_fks
            or missing_composite_fks
            or forbidden_fks
            or missing_checks
            or missing_indexes
            or missing_triggers
            or shape_issues
            or epoch_incompatible
            or identity_issue
        ):
            error_parts = []

            if epoch_incompatible:
                error_parts.append(f"schema epoch is incompatible:\nDatabase epoch: {schema_epoch}\nCurrent epoch: {SQLITE_SCHEMA_EPOCH}")

            if identity_issue:
                error_parts.append(
                    f"schema identity is incompatible ({identity_issue}); expected application elspeth, "
                    f"store landscape, epoch {SQLITE_SCHEMA_EPOCH}"
                )

            if missing_tables:
                missing_tables_str = ", ".join(missing_tables)
                error_parts.append(f"Missing tables: {missing_tables_str}")

            if missing_columns:
                missing_str = ", ".join(f"{t}.{c}" for t, c in missing_columns)
                error_parts.append(f"Missing columns: {missing_str}")

            if token_outcomes_shape_errors:
                error_parts.append("ADR-019 stale token_outcomes shape: " + "; ".join(token_outcomes_shape_errors))

            if missing_fks:
                missing_fk_str = ", ".join(f"{t}.{c} → {ref}" for t, c, ref in missing_fks)
                error_parts.append(f"Missing foreign keys: {missing_fk_str}")

            if missing_composite_fks:
                missing_composite_fk_str = ", ".join(
                    f"{table}({', '.join(columns)}) → {ref_table}({', '.join(ref_columns)})"
                    for table, columns, ref_table, ref_columns in missing_composite_fks
                )
                error_parts.append(f"Missing composite foreign keys: {missing_composite_fk_str}")

            if forbidden_fks:
                forbidden_fk_str = ", ".join(
                    f"{table}({', '.join(columns)}) → {ref_table}({', '.join(ref_columns)}) [{reason}]"
                    for table, columns, ref_table, ref_columns, reason in forbidden_fks
                )
                error_parts.append(f"Forbidden foreign keys: {forbidden_fk_str}")

            if missing_checks:
                missing_checks_str = ", ".join(f"{t}.{name}" for t, name in missing_checks)
                error_parts.append(f"Missing check constraints: {missing_checks_str}")

            if missing_indexes:
                missing_indexes_str = ", ".join(f"{t}.{name}" for t, name in missing_indexes)
                error_parts.append(f"Missing indexes: {missing_indexes_str}")

            if missing_triggers:
                error_parts.append(f"Missing triggers: {', '.join(missing_triggers)}")

            if shape_issues:
                shape_str = "; ".join(f"{issue.subject}: expected {issue.expected!r}, observed {issue.actual!r}" for issue in shape_issues)
                error_parts.append(f"Full metadata shape mismatches: {shape_str}")

            if (
                ("token_outcomes", "completed") in missing_columns
                or ("token_outcomes", "path") in missing_columns
                or token_outcomes_shape_errors
            ):
                error_parts.append(
                    "ADR-019 changed token_outcomes from the old single-axis outcome/is_terminal to "
                    "(TerminalOutcome, TerminalPath, completed). See "
                    f"{ADR019_CUTOVER_GUIDE} and replace the stale audit.db "
                    "before starting this ELSPETH version."
                )

            raise SchemaCompatibilityError(
                "Landscape database schema is outdated.\n\n" + "\n".join(error_parts) + "\n\n"
                f"Pre-1.0 schemas are not migrated in place. Uninstall the deployment,\n"
                f"delete/recreate the database, and reinstall this ELSPETH version.\n\n"
                f"Database: {_safe_database_descriptor(self.connection_string)}"
            )

    @property
    def engine(self) -> "Tier1Engine":
        """Return the SQLAlchemy engine, branded as Tier1Engine.

        ``Tier1Engine`` is a :func:`typing.NewType` over
        :class:`~sqlalchemy.engine.Engine` that carries the static guarantee
        that the engine passed backend-appropriate audit-integrity checks. For
        SQLite, that includes the PRAGMA integrity probe
        (:meth:`_verify_sqlite_pragmas`).  The only place in the codebase that
        may produce a ``Tier1Engine`` is this property — the ``cast()`` here
        is the single gated mint point.

        Raises:
            RuntimeError: If the database is not initialized (i.e. after
                :meth:`close` is called).
        """
        if self._engine is None:
            raise RuntimeError("Database not initialized")
        return cast("Tier1Engine", self._engine)

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    @classmethod
    def _from_parts(
        cls,
        connection_string: str,
        engine: Engine,
        *,
        passphrase: str | None = None,
        journal: LandscapeJournal | None = None,
        require_existing_schema: bool = False,
        read_only: bool = False,
    ) -> Self:
        """Construct instance from pre-created components.

        Single place that sets all instance attributes — eliminates drift risk
        from parallel ``cls.__new__()`` paths that manually assign fields.
        """
        instance = cls.__new__(cls)
        instance.connection_string = connection_string
        instance._passphrase = passphrase
        instance._engine = engine
        instance._journal = journal
        instance._require_existing_schema = require_existing_schema
        instance._read_only = read_only
        return instance

    @classmethod
    def in_memory(cls) -> Self:
        """Create an in-memory SQLite database for testing.

        Tables are created automatically.

        Returns:
            LandscapeDB instance with in-memory SQLite
        """
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        cls._configure_sqlite(engine)
        cls._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
        metadata.create_all(engine)
        instance = cls._from_parts("sqlite:///:memory:", engine)
        instance._sync_schema_identity()
        instance._sync_sqlite_schema_epoch()
        return instance

    @staticmethod
    def _sqlite_read_only_url(url: str) -> str:
        """Return a SQLite URI URL that opens the existing file read-only.

        SQLite ``immutable=1`` is only safe for static, read-only-directory
        snapshots with no WAL sidecar. Live WAL-mode audit databases may have
        committed schema or rows in the ``-wal`` file that immutable
        connections intentionally ignore.
        """
        parsed = make_url(url)
        if not parsed.drivername.startswith("sqlite"):
            return url

        database = parsed.database
        if database is None or database == ":memory:":
            raise ValueError("read_only=True requires a file-backed SQLite database")
        if database.startswith("file:"):
            raise ValueError("read_only=True expects a plain SQLite file path, not an existing file: URI")

        db_path = Path(database)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        immutable = not Path(f"{db_path}-wal").exists() and not os.access(db_path.parent, os.W_OK)
        uri_path = quote(str(db_path), safe="/:")
        if immutable:
            return f"{parsed.drivername}:///file:{uri_path}?mode=ro&immutable=1&uri=true"
        return f"{parsed.drivername}:///file:{uri_path}?mode=ro&uri=true"

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        passphrase: str | None = None,
        create_tables: bool = True,
        read_only: bool = False,
        dump_to_jsonl: bool = False,
        dump_to_jsonl_path: str | None = None,
        dump_to_jsonl_fail_on_error: bool = False,
        dump_to_jsonl_include_payloads: bool = False,
        dump_to_jsonl_payload_base_path: str | None = None,
        dump_to_jsonl_worker_suffix: str | None = None,
        **engine_kwargs: Any,
    ) -> Self:
        """Create database from connection URL.

        Args:
            url: SQLAlchemy connection URL
            passphrase: SQLCipher encryption passphrase. When provided, the
                database is opened with AES-256 encryption via sqlcipher3.
            create_tables: Whether to create tables if they don't exist.
                           Set to False when connecting to an existing database.
            read_only: Open an existing database for inspection only. SQLite
                uses a ``mode=ro`` URI so the database file itself is not
                writable. Static read-only-directory snapshots with no WAL
                sidecar use ``immutable=1``; live WAL databases do not, so
                committed WAL contents remain visible.
            dump_to_jsonl: Enable JSONL change journal for emergency backups
            dump_to_jsonl_path: Optional override path for JSONL journal.
                When set alongside ``dump_to_jsonl_worker_suffix``, it is the
                operator's responsibility to ensure distinct paths for each
                worker (explicit-path-at-N-over-1 is unsupported).
            dump_to_jsonl_fail_on_error: Fail if journal write fails
            dump_to_jsonl_include_payloads: Inline payloads in journal records
            dump_to_jsonl_payload_base_path: Payload store base path for inlining
            dump_to_jsonl_worker_suffix: Per-worker hex suffix (the uuid4 hex
                tail of the ``worker_id``).  When set, the derived journal
                path becomes ``db.journal.{suffix}.jsonl`` so followers on
                the same host write distinct files (ADR-030 §C.4 row 13).
                Ignored when ``dump_to_jsonl_path`` is supplied explicitly.

        Returns:
            LandscapeDB instance
        """
        if read_only and create_tables:
            raise ValueError("read_only=True requires create_tables=False")
        if read_only and dump_to_jsonl:
            raise ValueError("read_only=True cannot enable dump_to_jsonl")

        if passphrase is not None:
            if engine_kwargs:
                raise ValueError("SQLCipher construction does not accept SQLAlchemy engine kwargs")
            engine = cls._create_sqlcipher_engine(url, passphrase, read_only=read_only)
            cls._configure_sqlite(engine, read_only=read_only)
            if not read_only:
                # Tier-1 PRAGMA probe — see _verify_sqlite_pragmas docstring.
                cls._verify_sqlite_pragmas(engine, url)
        else:
            engine_url = cls._sqlite_read_only_url(url) if read_only and url.startswith("sqlite") else url
            engine = create_engine(engine_url, echo=False, **engine_kwargs)
            # SQLite-specific configuration
            if url.startswith("sqlite"):
                cls._configure_sqlite(engine, read_only=read_only)
                if not read_only:
                    cls._verify_sqlite_pragmas(engine, url)

        journal: LandscapeJournal | None = None
        if dump_to_jsonl:
            journal_path = cls._resolve_journal_path(
                url,
                explicit_path=dump_to_jsonl_path,
                worker_suffix=dump_to_jsonl_worker_suffix,
            )
            journal = LandscapeJournal(
                journal_path,
                fail_on_error=dump_to_jsonl_fail_on_error,
                include_payloads=dump_to_jsonl_include_payloads,
                payload_base_path=dump_to_jsonl_payload_base_path,
            )
            journal.attach(engine)

        instance = cls._from_parts(
            url,
            engine,
            passphrase=passphrase,
            journal=journal,
            require_existing_schema=not create_tables,
            read_only=read_only,
        )

        # Validate BEFORE create_all - catches old schema with missing columns
        # before we try to use it. For fresh DBs, validation passes (no tables yet).
        instance._validate_schema()

        if create_tables:
            instance._sync_sqlite_schema_epoch()
            metadata.create_all(engine)
            instance._create_additive_indexes()
            instance._sync_schema_identity()
            instance._sync_sqlite_schema_epoch()
        return instance

    @staticmethod
    def _resolve_journal_path(
        connection_string: str,
        *,
        explicit_path: str | None,
        worker_suffix: str | None = None,
    ) -> str:
        """Resolve the JSONL journal path at the database boundary."""
        url = make_url(connection_string)
        explicit_path = explicit_path or None
        if not url.drivername.startswith("sqlite"):
            if explicit_path is None:
                raise ValueError("dump_to_jsonl requires dump_to_jsonl_path for non-SQLite databases")
            return explicit_path

        db_path = LandscapeDB._journal_sqlite_db_path(url)
        if explicit_path is None:
            return LandscapeDB._derive_journal_path(connection_string, worker_suffix)

        allowed_root = db_path.parent.resolve()
        raw_path = Path(explicit_path)
        candidate = raw_path if raw_path.is_absolute() else allowed_root / raw_path
        resolved_path = candidate.resolve()
        try:
            resolved_path.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                f"dump_to_jsonl_path escapes allowed journal root {allowed_root}: {explicit_path!r} -> {resolved_path}"
            ) from exc
        return str(resolved_path)

    @property
    def is_read_only(self) -> bool:
        """Whether this handle was opened ``read_only=True`` (inspection only).

        Read-only handles route :meth:`connection` through
        :meth:`read_only_connection`, refuse :meth:`write_connection`, and
        must never be handed to write repositories (e.g.
        ``RecorderFactory`` skips ``TokenSchedulerRepository`` construction
        for them).
        """
        return self._read_only

    @staticmethod
    def _derive_journal_path(connection_string: str, worker_suffix: str | None = None) -> str:
        """Derive a default JSONL journal path from the connection string.

        Args:
            connection_string: SQLAlchemy connection string (must be SQLite).
            worker_suffix: Optional per-worker hex suffix.  When ``None``
                (the default N=1 leader case) the path is unchanged:
                ``db.journal.jsonl``.  When set the path becomes
                ``db.journal.{worker_suffix}.jsonl`` so that multiple workers
                on one host write to distinct files (ADR-030 §C.4 row 13;
                design line 284 / G line 455).

                Derive from ``worker_id`` (``worker:{run_id}:{HEX}``) by
                splitting on ``:`` and taking the last field.

        Note — FORENSIC-ONLY at N>1:
            Per-worker paths fix the file-corruption half of the N>1 journal
            problem, not the ordering half.  Records carry per-statement
            timestamps (``journal.py`` lines 39, 121) buffered to commit;
            statement-time is **not** cross-process WAL commit order.  At
            N>1 the per-worker journal is a forensic aid, **not** a
            replayable log.  The authoritative replay order is
            ``run_coordination_events.seq`` (AUTOINCREMENT — G line 409).
            Restore tooling must gate on single-worker provenance; a true
            in-transaction ``journal_seq`` total order is deferred to a
            future release.
        """
        url = make_url(connection_string)
        db_path = LandscapeDB._journal_sqlite_db_path(url)
        if worker_suffix is None:
            # N=1 leader path: byte-for-byte unchanged.
            return str(db_path.with_suffix(".journal.jsonl"))
        if _JOURNAL_WORKER_SUFFIX_RE.fullmatch(worker_suffix) is None:
            raise ValueError("dump_to_jsonl_worker_suffix must be a non-empty lowercase hex string")
        # N>1 follower path: embed the hex suffix before the extension.
        return str(db_path.parent / f"{db_path.stem}.journal.{worker_suffix}.jsonl")

    @staticmethod
    def _journal_sqlite_db_path(url: Any) -> Path:
        if not url.drivername.startswith("sqlite"):
            raise ValueError("dump_to_jsonl requires dump_to_jsonl_path for non-SQLite databases")
        database = url.database
        if database is None or database == ":memory:":
            raise ValueError("dump_to_jsonl requires a file-backed SQLite database")
        return Path(database)

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Get a database connection with automatic transaction handling.

        Uses engine.begin() for proper transaction semantics:
        - Auto-commits on successful block exit
        - Auto-rolls back on exception

        Usage:
            with db.connection() as conn:
                conn.execute(runs_table.insert().values(...))
            # Committed automatically if no exception raised
        """
        if self._read_only:
            with self.read_only_connection() as conn:
                yield conn
            return

        with _maybe_serialize_shared_connection(self.engine), self.engine.begin() as conn:
            yield conn

    @contextmanager
    def write_connection(self) -> Iterator[Connection]:
        """Transaction-scoped connection carrying WRITE INTENT.

        Sibling of :meth:`connection` for write transactions.  On SQLite the
        transaction begins with ``BEGIN IMMEDIATE``, taking the WAL write
        lock at BEGIN so cross-process read-then-write shapes cannot abort
        with the non-retryable ``SQLITE_BUSY_SNAPSHOT`` (ADR-030 §D5).
        Commit/rollback semantics are identical to :meth:`connection`:
        auto-commit on successful block exit, auto-rollback on exception.

        Raises:
            RuntimeError: If this handle was opened ``read_only=True`` —
                write intent on a read-only audit handle is a programming
                error, not a recoverable condition.
        """
        if self._read_only:
            raise RuntimeError("write_connection() is not available on a read-only LandscapeDB handle")

        with begin_write(self.engine) as conn:
            yield conn

    @contextmanager
    def read_only_connection(self) -> Iterator[Connection]:
        """Get a database connection that rejects all write operations.

        Defense-in-depth for untrusted SQL execution (e.g., MCP query tool).
        For SQLite, sets PRAGMA query_only = ON at the connection level and,
        on writable engines only, resets it to OFF in the finally block so the
        pooled DBAPI connection is returned in a writable state. Read-only
        engines (``from_url(read_only=True)``) keep query_only armed; SQLite
        file-backed read-only handles also open through a ``mode=ro`` URI, so
        query_only is defense in depth rather than the sole write barrier. For
        PostgreSQL, marks the current transaction READ ONLY. Unsupported
        backends fail closed instead of yielding a writable transaction.
        """
        dialect_name = self.engine.dialect.name
        with _maybe_serialize_shared_connection(self.engine), self.engine.begin() as conn:
            if dialect_name == "sqlite":
                conn.execute(text("PRAGMA query_only = ON"))
            elif dialect_name == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
            else:
                raise RuntimeError(f"read_only_connection is unsupported for backend '{dialect_name}'")
            try:
                yield conn
            finally:
                if dialect_name == "sqlite" and not self._read_only:
                    conn.execute(text("PRAGMA query_only = OFF"))
