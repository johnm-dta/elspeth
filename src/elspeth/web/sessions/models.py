"""SQLAlchemy Core table definitions for the session database.

Tables: sessions, chat_messages, composition_states, runs, blobs,
blob_run_links, blob_inline_resolutions, user_secrets.

Current schema bootstrap lives in ``sessions/schema.py``. Pre-release
session databases are created from this metadata and stale runtime DBs
are deleted/recreated rather than migrated.

All tables live in a dedicated session database, separate from the
Landscape audit database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import (
    DDL,
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
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.types import JSON

from elspeth.core.schema_identity import create_schema_identity_table

# ``SESSION_SCHEMA_EPOCH`` — schema version sentinel. Bump this constant
# whenever a table is added, removed, or otherwise altered in a way that
# requires the operator to delete the existing session DB. The startup
# validator (``schema._assert_schema_sentinels``) reads ``PRAGMA user_version``
# and crashes if it does not match, producing an actionable
# "Delete the session DB file and restart" message rather than a cryptic
# SQLAlchemy error the first time a new code path touches the stale DB.
#
# Pattern mirrors ``SQLITE_SCHEMA_EPOCH`` in ``core/landscape/schema.py``
# but is independent: the Landscape and session DBs are separate files
# with separate lifecycles and separate epochs.
#
# Epoch history (pre-1.0 policy — bumps require DB recreation):
#   1 → initial schema.
#   2 → interpretation_events_table added; operators upgrading across
#        this boundary MUST delete their session DB.
#   3 → composition_proposals gains user_message_id so explicit-approval
#        replay can preserve inline-blob chat-message provenance when
#        accepting a deferred proposal.
#   4 → composer_completion_events_table added (Phase 6A completion
#        gestures: mark_ready_for_review, export_yaml audit events).
#   5 → composer_completion_events gains per-event-type partial CHECK
#        constraints (Phase 6A post-merge hardening): payload_digest +
#        expires_at are NOT NULL iff event_type='mark_ready_for_review';
#        composition_state_id is NOT NULL for both event types. SQLite
#        does not support ALTER TABLE ADD CONSTRAINT, so the constraint
#        change is a fresh-schema bump.
#   6 → user_preferences gains tutorial_completed_at for the Phase 4
#        hello-world tutorial first-run/completed gate. Pre-release
#        policy remains delete-and-recreate for stale session DBs.
#   7 → resolved interpretation DELETE trigger permits whole-session
#        archival cascades while preserving direct-delete protection.
#   8 → ``composition_states.provenance`` closed enum gains the
#        ``tutorial_normalization`` value (Phase 4 hello-world tutorial
#        template-normalization writer). SQLite cannot ALTER a CHECK
#        constraint in place, so the operator deletes the staging
#        session DB and the schema is recreated. Pre-release policy
#        remains delete-and-recreate for stale session DBs.
#   9 → ``sessions.archived_at`` added so user-side archive can hide
#        sessions with durable run/completion history rather than deleting
#        audit-bearing rows. Unrun sessions may still be physically deleted.
#   10 → ``interpretation_events.kind`` added; surface-specific
#        ``auto_interpreted_opt_out`` rows now carry reviewed LLM artefacts
#        and V2 argument hashes.
#   11 → composition_proposals gains composer provenance fields so deferred
#        explicit-approval accepts can replay inline-blob writes with the same
#        model/provider/skill/arguments context as the original compose turn.
#   12 → blob_inline_resolutions added for runtime inline-content blob
#        substitution audit rows.
#   13 → composer provenance CHECK constraints reject blank/whitespace strings
#        as missing for blobs and composition proposals.
#   14 → blobs LLM-authored provenance CHECK constraint requires a non-blank
#        created_from_message_id anchor alongside the five creating_* fields.
#   15 → blob_inline_resolutions.blob_id preserved as historical audit data
#        without a live blobs.id foreign key, so completed-run audit rows do
#        not prevent blob deletion.
#   16 → blobs LLM-authored provenance CHECK constraint now rejects any
#        creating_* field on verbatim rows; whitespace strings can no longer
#        bypass the verbatim-side nullability invariant.
#   17 → interpretation_events.kind CHECK gains pipeline_decision for
#        LLM-authored cleanup/shape decisions that gate execution.
#   18 → sessions.forked_from_session_id self-referential foreign key constraint
#        removed to allow physical deletion of parent sessions (no-durable-history
#        archive path) when child forks exist.
#   19 → ``composition_states.sources`` added so named multi-source composer
#        states survive save/load instead of collapsing to the legacy singular
#        ``source`` compatibility column.
#   20 → ``sessions.forked_from_message_id`` gains an ON DELETE SET NULL
#        foreign key to ``chat_messages.id`` so source-message deletion cannot
#        leave dangling fork provenance.
#   21 → ``sessions.auth_provider_type`` and
#        ``user_secrets.auth_provider_type`` gain closed-list CHECK
#        constraints for the supported auth-provider namespaces.
#   22 → ``composition_states.provenance`` closed enum gains the
#        ``post_compose`` value for successful send-message/recompose state
#        advances. SQLite cannot ALTER a CHECK constraint in place, so
#        pre-release policy remains delete-and-recreate for stale session DBs.
#   23 → ``run_events.sequence`` added as a durable per-run replay cursor so
#        websocket reconnects can replay in insertion order and de-duplicate
#        live queue events already delivered by the persisted replay.
#   24 -> no SQL-shape change; bumped in lockstep with GUIDED_SESSION_SCHEMA_VERSION
#        5->6 (composer_meta JSON adds GuidedSession.profile +
#        advisor_checkpoint_passes_used + advisor_signoff_escape_offered) so a
#        stale sessions.db fail-closes at boot via _assert_schema_sentinels instead
#        of lazy-500-ing per guided row on GuidedSession.from_dict. Pre-release
#        delete-and-recreate policy; see docs/runbooks/staging-session-db-recreation.md.
#   25 -> no SQL-shape change; bumped in lockstep with GUIDED_SESSION_SCHEMA_VERSION
#        6->7 (composer_meta JSON drops the vestigial GuidedSession.profile.entry_seed
#        key). Invalidates the WHOLE sessions DB at boot, not just guided rows: this
#        is the eager fail-close that keeps a stale entry_seed-bearing profile blob
#        from lazy-500-ing deep in WorkflowProfile.from_dict's closed-key-set check.
#        Pre-release delete-and-recreate policy; see runbook above.
#   26 -> ``user_preferences`` gains the first-run tutorial resume columns
#        (``tutorial_stage`` + ``tutorial_session_id`` + ``tutorial_run_id`` +
#        ``tutorial_source_data_hash``) so a reload mid-tutorial resumes at the
#        persisted stage instead of restarting at Welcome (elspeth-918f4434b3).
#        Pre-release delete-and-recreate policy; see runbook above.
#   27 -> ``user_preferences`` gains ``freeform_intro_dismissed_at`` so the
#        freeform empty-state introduction can be dismissed account-wide.
#        Pre-release delete-and-recreate policy; see runbook above.
#   28 -> ``elspeth_schema_identity`` gives SQLite and PostgreSQL the same
#        application/store/epoch proof, including semantic-only schema bumps.
#   29 -> guided schema 8 invalidates schema-7 composer metadata and chain
#        proposal state and adds durable, fenced guided operation reservations
#        plus their append-only lease/takeover event history. Pre-release
#        policy remains delete-and-recreate for stale session databases.
SESSION_SCHEMA_EPOCH = 29

_SQLITE_ASCII_WHITESPACE = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)"
_POSTGRESQL_ASCII_WHITESPACE = "chr(9) || chr(10) || chr(11) || chr(12) || chr(13) || chr(32)"
_AUTH_PROVIDER_TYPE_CHECK = "auth_provider_type IN ('local', 'oidc', 'entra')"


def _sql_non_blank_text(column_name: str, *, dialect: Literal["sqlite", "postgresql"]) -> str:
    if dialect == "sqlite":
        return f"length(trim({column_name}, {_SQLITE_ASCII_WHITESPACE})) > 0"
    return f"length(btrim({column_name}, {_POSTGRESQL_ASCII_WHITESPACE})) > 0"


def _composition_proposals_composer_provenance_check(*, dialect: Literal["sqlite", "postgresql"]) -> str:
    return (
        "((composer_model_identifier IS NULL AND composer_model_version IS NULL AND "
        "composer_provider IS NULL AND composer_skill_hash IS NULL AND tool_arguments_hash IS NULL) OR "
        "(composer_model_identifier IS NOT NULL AND composer_model_version IS NOT NULL AND "
        "composer_provider IS NOT NULL AND composer_skill_hash IS NOT NULL AND tool_arguments_hash IS NOT NULL AND "
        f"{_sql_non_blank_text('composer_model_identifier', dialect=dialect)} AND "
        f"{_sql_non_blank_text('composer_model_version', dialect=dialect)} AND "
        f"{_sql_non_blank_text('composer_provider', dialect=dialect)} AND "
        f"{_sql_non_blank_text('composer_skill_hash', dialect=dialect)} AND "
        f"{_sql_non_blank_text('tool_arguments_hash', dialect=dialect)}))"
    )


def _blobs_creating_llm_provenance_check(*, dialect: Literal["sqlite", "postgresql"]) -> str:
    return (
        "((creation_modality IN ('llm_generated', 'disambiguated', 'llm_generated_then_amended')) AND "
        "created_from_message_id IS NOT NULL AND "
        f"{_sql_non_blank_text('created_from_message_id', dialect=dialect)} AND "
        "creating_model_identifier IS NOT NULL AND creating_model_version IS NOT NULL AND "
        "creating_provider IS NOT NULL AND creating_composer_skill_hash IS NOT NULL AND "
        "creating_arguments_hash IS NOT NULL AND "
        f"{_sql_non_blank_text('creating_model_identifier', dialect=dialect)} AND "
        f"{_sql_non_blank_text('creating_model_version', dialect=dialect)} AND "
        f"{_sql_non_blank_text('creating_provider', dialect=dialect)} AND "
        f"{_sql_non_blank_text('creating_composer_skill_hash', dialect=dialect)} AND "
        f"{_sql_non_blank_text('creating_arguments_hash', dialect=dialect)}) OR "
        "(creation_modality = 'verbatim' AND "
        "creating_model_identifier IS NULL AND creating_model_version IS NULL AND "
        "creating_provider IS NULL AND creating_composer_skill_hash IS NULL AND "
        "creating_arguments_hash IS NULL)"
    )


# ``SESSION_DB_APPLICATION_ID`` — project-unique SQLite ``application_id``.
# Stored in ``PRAGMA application_id`` so forensics tooling can confirm a
# given SQLite file is in fact an ELSPETH session DB rather than some
# other SQLite file that happens to live at the configured path.
#
# Hex 0x454C5350 spells "ELSP" (E=0x45, L=0x4C, S=0x53, P=0x50). Magic
# numbers in SQLite's ``application_id`` slot are conventionally chosen
# so a hexdump of the first 100 bytes of the file makes the project
# identifiable to a human auditor.
SESSION_DB_APPLICATION_ID = 0x454C5350

metadata = MetaData()
schema_identity_table = create_schema_identity_table(metadata)

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("auth_provider_type", String, nullable=False, default="local"),
    Column("title", String, nullable=False),
    # Default trust_mode is auto_commit, not explicit_approve.
    #
    # The explicit_approve flow (commit 1d2abca8e, 2026-05-14) intercepts
    # mutation tool calls *before* the validator runs and turns them into
    # pending proposals. The LLM gets back ``success=True`` /
    # ``status=APPROVAL_REQUIRED`` and stops iterating; the validator
    # only fires later at accept time, with no path back to the LLM for
    # self-correction. With a small composer model (gpt-5.4-mini), the
    # LLM's normal retry-on-validation-error loop is the load-bearing
    # mechanism for converging on a valid pipeline. Intercepting before
    # validation silently removes it, producing consistent stuck-proposal
    # dead-ends across multi-plugin pipelines (observed empirically
    # 2026-05-15 across multiple session retries with the exact prompt
    # that ran reliably 2026-05-14 under the pre-intercept default).
    #
    # explicit_approve remains a supported value via
    # ``PATCH /sessions/{id}/composer/preferences`` for operators who
    # want a hard human-approval gate on every mutation. Setting it as
    # the *default* is the regression; restoring auto_commit as the
    # default restores convergence. The proper fix to make
    # explicit_approve viable as a default is to dry-run the tool before
    # creating the proposal (so the LLM sees validation errors and can
    # retry within the same compose turn) — out of scope for the
    # default-revert; tracked as a follow-up.
    # ``default="auto_commit"`` is the Python-side INSERT default —
    # SQLAlchemy supplies it on inserts that omit ``trust_mode``. The
    # ``server_default`` is the DDL default baked into the table at
    # CREATE TABLE time and is preserved here for symmetry, but the
    # live staging DB was created when the DDL default was
    # "explicit_approve" — SQLite does not retroactively change column
    # defaults on existing tables. The Python-side ``default`` is what
    # actually takes effect for newly-inserted rows in already-deployed
    # databases. Per the project DB policy ("DB migration = delete the
    # old DB"), changing the DDL alone is insufficient.
    Column("trust_mode", String, nullable=False, default="auto_commit", server_default="auto_commit"),
    Column("density_default", String, nullable=False, server_default="high"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    # Historical provenance only. This deliberately is not a live
    # ForeignKey("sessions.id") so no-durable-history archive can physically
    # delete a parent session while preserving child fork provenance.
    Column(
        "forked_from_session_id",
        String,
        nullable=True,
    ),
    Column("forked_from_message_id", String, nullable=True),
    ForeignKeyConstraint(
        ["forked_from_message_id"],
        ["chat_messages.id"],
        name="fk_sessions_forked_from_message",
        ondelete="SET NULL",
    ),
    # ``interpretation_review_disabled`` — per-session "stop asking" toggle for
    # LLM-surfaced interpretation review. Fast-path read by the compose loop;
    # the authoritative audit record is the ``opted_out``
    # row in ``interpretation_events_table``.
    #
    # Two-source-of-truth tension (F-35): the opted-out state is represented
    # both by this boolean column (fast-path read by the compose loop) and by
    # the existence of a row in ``interpretation_events`` with
    # ``choice='opted_out'`` (authoritative audit record). They MUST remain
    # consistent. The service's write path sets both atomically within a
    # single transaction (``_session_write_lock`` held throughout).
    #
    # If a future audit finds the boolean ``true`` but no ``opted_out`` row
    # exists, that is a bug in the service's write path — crash on read
    # (offensive programming). The boolean is a read-cache; the
    # ``interpretation_events`` row is the source of truth.
    #
    # F-35 follow-up: consider enforcing this constraint with a trigger or
    # computed column in a future schema revision. Deferred because SQLite
    # does not support CHECK constraints that cross tables.
    Column(
        "interpretation_review_disabled",
        Boolean,
        nullable=False,
        server_default=text("false"),
    ),
    Column("archived_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "trust_mode IN ('explicit_approve', 'auto_commit')",
        name="ck_sessions_trust_mode",
    ),
    CheckConstraint(
        "density_default IN ('high', 'medium', 'low')",
        name="ck_sessions_density_default",
    ),
    CheckConstraint(
        _AUTH_PROVIDER_TYPE_CHECK,
        name="ck_sessions_auth_provider_type",
    ),
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
    # role='assistant'; _assert_parent_assistant_message guard
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
    Column("sources", JSON, nullable=True),
    Column("nodes", JSON, nullable=True),
    Column("edges", JSON, nullable=True),
    Column("outputs", JSON, nullable=True),
    Column("metadata_", JSON, nullable=True),
    Column("is_valid", Boolean, nullable=False, default=False),
    Column("validation_errors", JSON, nullable=True),
    # Operational/audit metadata produced by the composer pipeline that
    # describes *how this state was reached* (distinct from ``metadata_``,
    # which carries the user-facing PipelineMetadata name/description).
    # Currently only ``repair_turns_used`` is surfaced; absence (NULL) is
    # honest for revert/fork paths where no compose produced this version.
    Column("composer_meta", JSON, nullable=True),
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
    # addition. The Python Literal counterpart lives at
    # ``web/sessions/protocol.py::CompositionStateProvenance``; the two
    # are paired contracts (extending one without the other lets the
    # writer pass while the DB rejects the row, or vice versa). Schedule
    # 1A treats this as a DB-only audit column: it is NOT surfaced on
    # ``CompositionStateRecord`` / ``CompositionStateResponse``.
    # Read-side hydration is deferred to Schedule 1B+ per plan §1053-1061.
    Column("provenance", String, nullable=False),
    UniqueConstraint("session_id", "version", name="uq_composition_state_version"),
    # Composite uniqueness target for composite FKs on chat_messages /
    # runs. The primary key already makes `id` unique on its own; this
    # constraint exists solely so SQL engines (including Postgres) will
    # accept (id, session_id) as an FK reference.
    UniqueConstraint("id", "session_id", name="uq_composition_state_id_session"),
    # Closed enum: every value corresponds to a documented writer path
    # in spec §4.1.2 (``session_fork`` is the cross-session fork-copy
    # value).
    # Adding a value here without amending the spec creates an
    # untraceable writer category in the audit DB.
    #
    # The original persist-path values were actively written as of
    # elspeth-obs-f217c634aa (closed by the same commit that retired the
    # dormant-value friction block here). The previous block warned that
    # three values (``convergence_persist``, ``plugin_crash_persist``,
    # ``preflight_persist``) had no writer; verification revealed the
    # call sites already existed in ``web/sessions/routes.py`` but were
    # passing through ``save_composition_state``'s hardcoded
    # ``"session_seed"`` label. The fix threads ``provenance`` through
    # the public API as a required keyword argument so all four writer
    # categories (session_seed, convergence_persist,
    # plugin_crash_persist, preflight_persist) are distinguishable in
    # the audit DB. Active writer map:
    #
    #   - ``tool_call``               — service.py compose-loop atomic write
    #   - ``convergence_persist``     — routes.py _handle_convergence_error
    #   - ``plugin_crash_persist``    — routes.py _handle_plugin_crash
    #   - ``preflight_persist``       — routes.py _handle_runtime_preflight_failure
    #   - ``tutorial_normalization``  — DORMANT (no live writer). Formerly
    #                                    written by tutorial_service.py's
    #                                    pre-execution template normalizer
    #                                    (Phase 4 hello-world tutorial), which
    #                                    rewrote bare ``{{ field }}`` placeholders
    #                                    to the ``row.field`` namespace before the
    #                                    live pass. Removed for tutorial-vs-regular
    #                                    backend parity (the composer already emits
    #                                    ``row.field`` templates; a 10-run live
    #                                    battery showed the rewrite firing 0/10).
    #                                    The value is retained in this CHECK so
    #                                    historical audit rows remain representable;
    #                                    re-activation is a governance action per
    #                                    the NO SILENT EXTENSION block below.
    #   - ``post_compose``            — routes/messages.py send_message and
    #                                    routes/composer.py recompose successful
    #                                    LLM-driven state advances, including the
    #                                    transition_consumed metadata-only row.
    #   - ``session_seed``            — service.py create_session + set_active_state
    #   - ``session_fork``            — service.py fork_session_at_message and
    #                                    routes/sessions.py fork blob-reference
    #                                    rewrite, which is part of the same fork
    #                                    writer path.
    #   - ``interpretation_resolve``  — routes.py /resolve handler:
    #                                    a composition-state row written when the
    #                                    user resolves an LLM-surfaced interpretation
    #                                    (accept_as_drafted / amend) so the patched
    #                                    prompt template is committed alongside the
    #                                    ``interpretation_events`` row.
    #
    # NO SILENT EXTENSION. Adding another value MUST include all three
    # of: (a) a spec amendment documenting the writer path and the audit
    # semantics that distinguish it from neighbouring values; (b) an
    # integration test that drives the writer and asserts the row was
    # committed with the new ``provenance`` value; (c) a Filigree ticket
    # linking the change back to this enum so the audit history shows the
    # addition as a deliberate governance step rather than a drive-by
    # edit. Mirror also goes into ``CompositionStateProvenance`` at
    # ``protocol.py``. See the parallel ``audit_access_log_table`` "INERT
    # IN PHASE 1A" block below for the same closed-list-of-permitted-
    # writers posture.
    CheckConstraint(
        "provenance IN ('tool_call', 'convergence_persist', 'plugin_crash_persist', 'preflight_persist', 'tutorial_normalization', 'post_compose', 'session_seed', 'session_fork', 'interpretation_resolve')",
        name="ck_composition_states_provenance",
    ),
)

composition_proposals_table = Table(
    "composition_proposals",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("tool_call_id", String, nullable=False),
    # Originating user chat message for proposals created by the composer
    # in explicit_approve mode. Nullable for legacy/imported proposals and
    # future non-chat proposal sources. When present, the composite FK forces
    # same-session ownership so accept/replay cannot bind blobs to another
    # session's message.
    Column("user_message_id", String, nullable=True),
    # Composer provenance captured at proposal creation. Nullable for
    # legacy/imported/manual proposals, but all-or-none when present. Accept
    # replay uses these columns for inline/blob-backed source writes so a
    # deferred proposal cannot silently downgrade composer-authored bytes to
    # verbatim merely because the original compose-loop context is gone.
    Column("composer_model_identifier", String, nullable=True),
    Column("composer_model_version", String, nullable=True),
    Column("composer_provider", String, nullable=True),
    Column("composer_skill_hash", String, nullable=True),
    Column("tool_arguments_hash", String, nullable=True),
    Column("tool_name", String, nullable=False),
    Column("status", String, nullable=False),
    Column("summary", Text, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("affects", JSON, nullable=False),
    # Raw arguments are retained only for replay/execution. Normal
    # API/UI surfaces must expose ``arguments_redacted_json`` instead.
    Column("arguments_json", JSON, nullable=False),
    Column("arguments_redacted_json", JSON, nullable=False),
    Column("base_state_id", String, nullable=True),
    Column("committed_state_id", String, nullable=True),
    Column("audit_event_id", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["base_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_composition_proposals_base_state_session",
    ),
    ForeignKeyConstraint(
        ["committed_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_composition_proposals_committed_state_session",
    ),
    ForeignKeyConstraint(
        ["user_message_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_composition_proposals_user_message_session",
        ondelete="RESTRICT",
    ),
    UniqueConstraint(
        "session_id",
        "tool_call_id",
        name="uq_composition_proposals_session_tool_call",
    ),
    # Composite target for guided_operations.proposal_id. The proposal id is
    # globally unique already; the pair exists to make same-session custody a
    # database invariant on both SQLite and PostgreSQL.
    UniqueConstraint("id", "session_id", name="uq_composition_proposals_id_session"),
    CheckConstraint(
        "status IN ('pending', 'committed', 'rejected')",
        name="ck_composition_proposals_status",
    ),
    CheckConstraint(
        "(status = 'committed') = (committed_state_id IS NOT NULL)",
        name="ck_composition_proposals_committed_state",
    ),
    CheckConstraint(
        _composition_proposals_composer_provenance_check(dialect="sqlite"),
        name="ck_composition_proposals_composer_provenance_all_or_none",
    ).ddl_if(dialect="sqlite"),
    CheckConstraint(
        _composition_proposals_composer_provenance_check(dialect="postgresql"),
        name="ck_composition_proposals_composer_provenance_all_or_none",
    ).ddl_if(dialect="postgresql"),
)


def _lower_sha256_check(column_name: str, *, dialect: Literal["sqlite", "postgresql"]) -> str:
    base = f"length({column_name}) = 64"
    if dialect == "sqlite":
        return f"{base} AND {column_name} NOT GLOB '*[^a-f0-9]*'"
    return f"{base} AND {column_name} ~ '^[a-f0-9]+$'"


# One bounded reservation row per client-authored mutating action. Raw request
# bodies, user intent, and provider errors are deliberately absent: replay is
# bound by a canonical request hash and a small immutable result locator.
guided_operations_table = Table(
    "guided_operations",
    metadata,
    Column("session_id", String(128), nullable=False),
    Column("operation_id", String(128), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("status", String(16), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("lease_token", String(256), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("attempt", Integer, nullable=False),
    Column("originating_message_id", String(128), nullable=True),
    Column("proposal_id", String(128), nullable=True),
    Column("result_kind", String(32), nullable=True),
    Column("result_state_id", String(128), nullable=True),
    Column("result_session_id", String(128), nullable=True),
    Column("response_hash", String(64), nullable=True),
    Column("failure_code", String(128), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("settled_at", DateTime(timezone=True), nullable=True),
    PrimaryKeyConstraint("session_id", "operation_id", name="pk_guided_operations"),
    UniqueConstraint(
        "session_id",
        "operation_id",
        "request_hash",
        name="uq_guided_operations_request_binding",
    ),
    ForeignKeyConstraint(["session_id"], ["sessions.id"], name="fk_guided_operations_session", ondelete="CASCADE"),
    ForeignKeyConstraint(
        ["originating_message_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_guided_operations_originating_message_session",
        ondelete="RESTRICT",
    ),
    ForeignKeyConstraint(
        ["proposal_id", "session_id"],
        ["composition_proposals.id", "composition_proposals.session_id"],
        name="fk_guided_operations_proposal_session",
        ondelete="RESTRICT",
    ),
    ForeignKeyConstraint(
        ["result_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_guided_operations_result_state_session",
        ondelete="RESTRICT",
    ),
    ForeignKeyConstraint(["result_session_id"], ["sessions.id"], name="fk_guided_operations_result_session", ondelete="RESTRICT"),
    CheckConstraint("length(operation_id) >= 1 AND length(operation_id) <= 128", name="ck_guided_operations_operation_id_bounded"),
    CheckConstraint(
        "originating_message_id IS NULL OR (length(originating_message_id) >= 1 AND length(originating_message_id) <= 128)",
        name="ck_guided_operations_originating_message_id_bounded",
    ),
    CheckConstraint(
        "proposal_id IS NULL OR (length(proposal_id) >= 1 AND length(proposal_id) <= 128)",
        name="ck_guided_operations_proposal_id_bounded",
    ),
    CheckConstraint(
        "result_state_id IS NULL OR (length(result_state_id) >= 1 AND length(result_state_id) <= 128)",
        name="ck_guided_operations_result_state_id_bounded",
    ),
    CheckConstraint(
        "result_session_id IS NULL OR (length(result_session_id) >= 1 AND length(result_session_id) <= 128)",
        name="ck_guided_operations_result_session_id_bounded",
    ),
    CheckConstraint(
        "kind IN ('guided_start', 'guided_respond', 'guided_chat', 'guided_convert', "
        "'guided_reenter', 'state_revert', 'session_fork', 'guided_plan')",
        name="ck_guided_operations_kind",
    ),
    CheckConstraint("status IN ('in_progress', 'completed', 'failed')", name="ck_guided_operations_status"),
    CheckConstraint(_lower_sha256_check("request_hash", dialect="sqlite"), name="ck_guided_operations_request_hash").ddl_if(
        dialect="sqlite"
    ),
    CheckConstraint(_lower_sha256_check("request_hash", dialect="postgresql"), name="ck_guided_operations_request_hash").ddl_if(
        dialect="postgresql"
    ),
    CheckConstraint("attempt >= 1", name="ck_guided_operations_attempt"),
    CheckConstraint("updated_at >= created_at", name="ck_guided_operations_updated_after_created"),
    CheckConstraint("settled_at IS NULL OR settled_at >= created_at", name="ck_guided_operations_settled_after_created"),
    CheckConstraint(
        "lease_token IS NULL OR (length(lease_token) >= 1 AND length(lease_token) <= 256)",
        name="ck_guided_operations_lease_token_bounded",
    ),
    CheckConstraint(
        "result_kind IS NULL OR result_kind IN ('composition_state', 'session', 'proposal')",
        name="ck_guided_operations_result_kind",
    ),
    CheckConstraint(
        "failure_code IS NULL OR failure_code IN ('provider_unavailable', 'provider_timeout', "
        "'invalid_provider_response', 'integrity_error', 'custody_error', 'operation_failed')",
        name="ck_guided_operations_failure_code",
    ),
    CheckConstraint(
        "(status = 'in_progress' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL "
        "AND settled_at IS NULL AND result_kind IS NULL AND result_state_id IS NULL "
        "AND response_hash IS NULL AND failure_code IS NULL) OR "
        "(status = 'completed' AND lease_token IS NULL AND lease_expires_at IS NULL "
        "AND settled_at IS NOT NULL AND result_kind IS NOT NULL AND response_hash IS NOT NULL AND failure_code IS NULL) OR "
        "(status = 'failed' AND lease_token IS NULL AND lease_expires_at IS NULL "
        "AND settled_at IS NOT NULL AND result_kind IS NULL AND result_state_id IS NULL "
        "AND result_session_id IS NULL AND proposal_id IS NULL AND response_hash IS NULL AND failure_code IS NOT NULL)",
        name="ck_guided_operations_status_bundle",
    ),
    CheckConstraint(
        "(status = 'in_progress' AND "
        "(result_session_id IS NULL OR kind = 'session_fork') AND "
        "(proposal_id IS NULL OR kind IN ('guided_respond', 'guided_chat', 'guided_plan'))) OR "
        "(status = 'completed' AND ("
        "(kind = 'session_fork' AND result_kind = 'session' AND result_session_id IS NOT NULL "
        "AND result_state_id IS NULL AND proposal_id IS NULL) OR "
        "(kind = 'guided_plan' AND result_kind = 'proposal' AND proposal_id IS NOT NULL "
        "AND result_state_id IS NULL AND result_session_id IS NULL) OR "
        "(kind NOT IN ('session_fork', 'guided_plan') AND result_kind = 'composition_state' "
        "AND result_state_id IS NOT NULL AND result_session_id IS NULL "
        "AND (proposal_id IS NULL OR kind IN ('guided_respond', 'guided_chat'))))) OR "
        "(status = 'failed' AND result_kind IS NULL AND result_state_id IS NULL "
        "AND result_session_id IS NULL AND proposal_id IS NULL)",
        name="ck_guided_operations_result_locator",
    ),
    CheckConstraint(
        "response_hash IS NULL OR (length(response_hash) = 64 AND response_hash NOT GLOB '*[^a-f0-9]*')",
        name="ck_guided_operations_response_hash",
    ).ddl_if(dialect="sqlite"),
    CheckConstraint(
        "response_hash IS NULL OR (length(response_hash) = 64 AND response_hash ~ '^[a-f0-9]+$')",
        name="ck_guided_operations_response_hash",
    ).ddl_if(dialect="postgresql"),
)
Index("ix_guided_operations_status_lease", guided_operations_table.c.status, guided_operations_table.c.lease_expires_at)


# Append-only evidence for initial claims, renewals, takeovers, and terminal
# settlement. Lease tokens and request/provider text never enter this table.
guided_operation_events_table = Table(
    "guided_operation_events",
    metadata,
    Column("session_id", String(128), nullable=False),
    Column("operation_id", String(128), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("event_kind", String(16), nullable=False),
    Column("actor", String(128), nullable=False),
    Column("attempt", Integer, nullable=False),
    Column("prior_attempt", Integer, nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("request_hash", String(64), nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("session_id", "operation_id", "sequence", name="pk_guided_operation_events"),
    ForeignKeyConstraint(
        ["session_id", "operation_id", "request_hash"],
        ["guided_operations.session_id", "guided_operations.operation_id", "guided_operations.request_hash"],
        name="fk_guided_operation_events_operation",
        ondelete="CASCADE",
    ),
    CheckConstraint("sequence >= 1", name="ck_guided_operation_events_sequence"),
    CheckConstraint("event_kind IN ('claimed', 'renewed', 'taken_over', 'completed', 'failed')", name="ck_guided_operation_events_kind"),
    CheckConstraint("length(actor) >= 1 AND length(actor) <= 128", name="ck_guided_operation_events_actor_bounded"),
    CheckConstraint("attempt >= 1", name="ck_guided_operation_events_attempt"),
    CheckConstraint(_lower_sha256_check("request_hash", dialect="sqlite"), name="ck_guided_operation_events_request_hash").ddl_if(
        dialect="sqlite"
    ),
    CheckConstraint(_lower_sha256_check("request_hash", dialect="postgresql"), name="ck_guided_operation_events_request_hash").ddl_if(
        dialect="postgresql"
    ),
    CheckConstraint(
        "(event_kind IN ('claimed', 'renewed') AND prior_attempt IS NULL AND lease_expires_at IS NOT NULL) OR "
        "(event_kind = 'taken_over' AND prior_attempt IS NOT NULL AND prior_attempt = attempt - 1 AND lease_expires_at IS NOT NULL) OR "
        "(event_kind IN ('completed', 'failed') AND prior_attempt IS NULL AND lease_expires_at IS NULL)",
        name="ck_guided_operation_events_bundle",
    ),
)

# Per-event ``payload`` JSON contract (Tier-1 schema; CLAUDE.md
# §"Three-Tier Trust Model" — values written here are our own data and
# must satisfy the contract on read or crash):
#
# Payload contract for event_type="trust_mode.changed":
#   trust_mode: str — the new value the PATCH set
#       (vocabulary: 'explicit_approve' | 'auto_commit' per the
#       CHECK constraint on sessions_table.trust_mode; see
#       models.py:150).
#   prior_trust_mode: str — the value the PATCH overwrote
#       (added Phase 8 B1; same vocabulary as trust_mode above;
#       required for the per-session session-switched telemetry
#       counter to satisfy the audit-primacy superset rule).
#   density_default: str — the new density_default value
#       (vocabulary per the column's CHECK constraint).
# Adding a new key here is a Tier-1 schema-cohort change (per
# CLAUDE.md "DB migration = delete the old DB"); document the
# key, its vocabulary, and the owning phase at the same time.
#
# Payload contract for event_type="proposal.created":
#   Legacy rows retain the exact historical
#   {tool_call_id, tool_name, status="pending"} shape.
#   Canonical set_pipeline rows use closed
#   schema="pipeline_proposal_created.v1" metadata. It binds the exact
#   private arguments, manifest-redacted audit projection, provenance,
#   tagged base, surface, draft/reviewed-anchor/skill hashes, repair count,
#   deferred-intent ids, optional supersession pair, and custody result.
#   Readers must never reinterpret malformed canonical metadata as legacy.
#
# Payload contract for event_type="proposal.accepted":
#   Legacy rows retain their historical shape. Canonical pipeline rows use
#   schema="pipeline_proposal_accepted.v1" and bind the draft, committed
#   state id/content, final composer metadata, and the exact durable redacted
#   dispatch envelope (including its canonical executor-content hash).
#
# Payload contract for event_type="proposal.rejected":
#   Legacy rows retain their historical shape. Canonical pipeline rows use
#   schema="pipeline_proposal_rejected.v1" with a closed reason_code and an
#   optional durable dispatch binding; free-form request/provider/exception
#   text is never persisted in the canonical terminal event.
proposal_events_table = Table(
    "proposal_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "proposal_id",
        String,
        ForeignKey(
            "composition_proposals.id",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=True,
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "event_type IN ('proposal.created', 'proposal.accepted', 'proposal.rejected', 'trust_mode.changed')",
        name="ck_proposal_events_type",
    ),
)
Index(
    "ix_proposal_events_session_created",
    proposal_events_table.c.session_id,
    proposal_events_table.c.created_at,
)

interpretation_events_table = Table(
    "interpretation_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    # Composite FK forces same-session ownership: an interpretation event in
    # session B cannot reference a composition state owned by session A.
    # Mirrors the pattern at composition_proposals.
    # F-1/F-12: nullable=True — NULL for session-level
    # auto_interpreted_opt_out marker rows; populated for user_approved rows
    # and surface-specific auto_interpreted_opt_out rows.
    Column("composition_state_id", String, nullable=True),
    # The LLM transform's node_id within composition_states.nodes that this
    # interpretation binds into. Validated at the writer boundary to exist;
    # NOT a foreign key because nodes live inside a JSON column, not a
    # separate table.
    # F-1/F-12: nullable=True — NULL for session-level
    # auto_interpreted_opt_out marker rows; populated for user_approved rows
    # and surface-specific auto_interpreted_opt_out rows.
    Column("affected_node_id", String, nullable=True),
    # The provider tool_call_id from the LLM call that surfaced this
    # interpretation. NOT a foreign key to chat_messages because the tool
    # call may still be in flight when this row is inserted.
    # F-1/F-12: nullable=True — NULL for session-level
    # auto_interpreted_opt_out marker rows; populated for user_approved rows
    # and surface-specific auto_interpreted_opt_out rows.
    Column("tool_call_id", String, nullable=True),
    # Audit-mandatory fields (source-conditional: see CHECKs below):
    # NULL for session-level auto_interpreted_opt_out marker rows (no LLM
    # surfacing occurred); NOT NULL for user_approved rows and
    # surface-specific auto_interpreted_opt_out rows; NULL for
    # auto_interpreted_no_surfaces rows except kind, which records the
    # rejected review class.
    # F-1: all nullable=True — the CHECKs below enforce source-conditional
    # presence.
    Column("user_term", Text, nullable=True),
    Column("kind", String, nullable=True),
    Column("llm_draft", Text, nullable=True),
    Column("accepted_value", Text, nullable=True),  # None until resolved
    Column("choice", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True), nullable=True),
    Column("actor", String, nullable=False),
    # Audit provenance: snapshot of the composer LLM context at surfacing time.
    # F-1: nullable=True for model_identifier, model_version, provider,
    # composer_skill_hash — NULL for session-level auto_interpreted_opt_out
    # marker rows (no LLM was consulted), populated for surface-specific
    # auto_interpreted_opt_out rows.
    Column("model_identifier", String, nullable=True),
    Column("model_version", String, nullable=True),
    Column("provider", String, nullable=True),
    Column("composer_skill_hash", String, nullable=True),
    # NULL until resolved. Session-level auto_interpreted_opt_out marker rows
    # have no LLM-supplied content to hash; surface-specific opt-out rows are
    # born resolved with accepted_value, arguments_hash, and
    # hash_domain_version='v2'.
    # F-12 (hash domain versioning): hash_domain_version records the field set
    # used to compute arguments_hash. New writes use
    # contracts/composer_interpretation.py:INTERPRETATION_HASH_DOMAIN_V2.
    # NULL for rows without a hash (opt-out, pending). NOT NULL once resolved.
    Column("arguments_hash", String, nullable=True),
    Column("hash_domain_version", String, nullable=True),
    # Structural source of this row. Closed enum — see governance block
    # on ``composition_states.provenance`` above for the same posture.
    Column("interpretation_source", String, nullable=False),
    # F-19 (runtime model snapshot at resolve time): nullable columns
    # populated when the user resolves the event, capturing what model the
    # affected LLM transform will use at runtime (for audit drift detection).
    Column("runtime_model_identifier_at_resolve", String, nullable=True),
    Column("runtime_model_version_at_resolve", String, nullable=True),
    # Cross-DB hash anchor (Option A — see §"Hash-anchored cross-DB linkage"
    # in 18-phase-5b-surface-llm-interpretation.md). Populated by
    # resolve_interpretation_event at the same time as accepted_value is
    # committed. NULL until resolved; NULL for session-level
    # auto_interpreted_opt_out marker rows and for non-prompt-template
    # surface opt-out rows. For user_approved rows and surface-specific
    # auto_interpreted_opt_out rows that resolve an llm_prompt_template,
    # this is SHA-256 over the rfc8785 canonical JSON of the
    # resolved prompt-template string, using
    # ``CANONICAL_VERSION = "sha256-rfc8785-v1"`` (contracts/hashing.py).
    # NOT part of INTERPRETATION_HASH_DOMAIN_V2 — it covers a different
    # input (the resolved prompt string) and serves as a cross-DB anchor
    # only. Pair to ``calls.resolved_prompt_template_hash`` in the L1
    # Landscape audit DB; equality across both DBs is the audit-tooling
    # cross-anchor invariant.
    Column("resolved_prompt_template_hash", String(64), nullable=True),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_interpretation_events_state_session",
    ),
    UniqueConstraint(
        "id",
        "session_id",
        name="uq_interpretation_events_id_session",
    ),
    # Closed enum on choice. Adding a value requires (a) amending this plan,
    # (b) extending InterpretationChoice in contracts/composer_interpretation.py,
    # (c) updating the closed-enum tests, and (d) a writer-path audit.
    # NO SILENT EXTENSION. See composition_states governance block above.
    CheckConstraint(
        "choice IN ('pending', 'accepted_as_drafted', 'amended', 'opted_out', 'abandoned')",
        name="ck_interpretation_events_choice",
    ),
    # Closed enum on interpretation_source. Adding a value requires the same
    # four-step ceremony as choice. NO SILENT EXTENSION.
    CheckConstraint(
        "interpretation_source IN ('user_approved', 'auto_interpreted_opt_out', 'auto_interpreted_no_surfaces')",
        name="ck_interpretation_events_source",
    ),
    # Closed enum on kind. Adding a value requires the same ceremony as the
    # InterpretationKind contract enum: contract amendment, schema update,
    # closed-enum tests, and writer-path audit.
    CheckConstraint(
        "kind IS NULL OR kind IN ('vague_term', 'invented_source', 'llm_prompt_template', 'pipeline_decision', 'llm_model_choice')",
        name="ck_interpretation_events_kind",
    ),
    # Auto-interpreted rows are born resolved by definition. They never
    # represent a user acceptance/amendment or an abandoned pending surface.
    CheckConstraint(
        "(interpretation_source NOT IN ('auto_interpreted_opt_out', 'auto_interpreted_no_surfaces')) OR choice = 'opted_out'",
        name="ck_interpretation_events_auto_source_choice",
    ),
    # F-1/F-12 (source-keyed opt-out shapes): session-level opt-out marker
    # rows have no LLM context and no hash; surface-specific opt-out rows
    # are born resolved with kind, surface/provenance fields, accepted_value,
    # and a V2 arguments_hash.
    CheckConstraint(
        "(interpretation_source != 'auto_interpreted_opt_out') OR "
        "((kind IS NULL AND composition_state_id IS NULL AND affected_node_id IS NULL AND "
        "  tool_call_id IS NULL AND user_term IS NULL AND llm_draft IS NULL AND "
        "  accepted_value IS NULL AND model_identifier IS NULL AND model_version IS NULL AND "
        "  provider IS NULL AND composer_skill_hash IS NULL AND arguments_hash IS NULL AND "
        "  hash_domain_version IS NULL) OR "
        " (kind IS NOT NULL AND composition_state_id IS NOT NULL AND affected_node_id IS NOT NULL AND "
        "  tool_call_id IS NOT NULL AND user_term IS NOT NULL AND llm_draft IS NOT NULL AND "
        "  accepted_value IS NOT NULL AND model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        "  provider IS NOT NULL AND composer_skill_hash IS NOT NULL AND arguments_hash IS NOT NULL AND "
        "  hash_domain_version = 'v2'))",
        name="ck_interpretation_events_opt_out_shape",
    ),
    # user_approved rows must have kind and all surface/provenance fields
    # populated.
    CheckConstraint(
        "(interpretation_source != 'user_approved') OR "
        "(composition_state_id IS NOT NULL AND affected_node_id IS NOT NULL AND "
        " tool_call_id IS NOT NULL AND user_term IS NOT NULL AND kind IS NOT NULL AND llm_draft IS NOT NULL AND "
        " model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        " provider IS NOT NULL AND composer_skill_hash IS NOT NULL)",
        name="ck_interpretation_events_user_approved_required",
    ),
    # auto_interpreted_no_surfaces rows: the five surface fields
    # (composition_state_id, affected_node_id, tool_call_id, user_term,
    # llm_draft) must be NULL (no surfacing occurred); kind and the four
    # LLM provenance fields must be NOT NULL (the LLM was consulted for
    # the rate-cap fallback auto-interpretation). This is the middle shape
    # between session opt-out marker rows and user_approved rows.
    CheckConstraint(
        "(interpretation_source != 'auto_interpreted_no_surfaces') OR "
        "(composition_state_id IS NULL AND affected_node_id IS NULL AND "
        " tool_call_id IS NULL AND user_term IS NULL AND kind IS NOT NULL AND llm_draft IS NULL AND "
        " model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        " provider IS NOT NULL AND composer_skill_hash IS NOT NULL)",
        name="ck_interpretation_events_no_surfaces_shape",
    ),
    # If choice is anything other than 'pending', resolved_at MUST be
    # populated. For opted_out rows resolved_at records the opt-out
    # timestamp.
    CheckConstraint(
        "(choice = 'pending') = (resolved_at IS NULL)",
        name="ck_interpretation_events_resolved_at_status",
    ),
    # accepted_value is populated only for accepted_as_drafted/amended
    # user-approved rows and surface-specific auto_interpreted_opt_out rows.
    CheckConstraint(
        "((choice IN ('accepted_as_drafted', 'amended')) OR "
        " (interpretation_source = 'auto_interpreted_opt_out' AND kind IS NOT NULL)) "
        "= (accepted_value IS NOT NULL)",
        name="ck_interpretation_events_accepted_value_status",
    ),
)

# Partial unique index: only one pending interpretation per
# (session_id, tool_call_id). After resolution (choice != 'pending'), the
# same tool_call_id is allowed to recur (which it won't in practice, but
# the index does not need to over-constrain).
# F-26: See web/sessions/schema.py:_validate_partial_index_dialect_symmetry
# for the schema-validator gate that enforces both sqlite_where and
# postgresql_where are set consistently.
Index(
    "uq_interpretation_events_pending_tool_call",
    interpretation_events_table.c.session_id,
    interpretation_events_table.c.tool_call_id,
    unique=True,
    sqlite_where=interpretation_events_table.c.choice == "pending",
    postgresql_where=interpretation_events_table.c.choice == "pending",
)

Index(
    "ix_interpretation_events_session_created",
    interpretation_events_table.c.session_id,
    interpretation_events_table.c.created_at,
)

# F-11: index on composition_state_id for the common lookup pattern
# "all interpretation events for this composition state". Verify with
# EXPLAIN QUERY PLAN; the test in
# tests/unit/web/sessions/test_interpretation_events_table.py asserts
# SEARCH USING INDEX.
Index(
    "ix_interpretation_events_composition_state",
    interpretation_events_table.c.composition_state_id,
)

# ``composer_completion_events_table`` (Phase 6A — completion gestures).
#
# One row per completion-gesture audit event. Two event types in v1:
#
#   * ``mark_ready_for_review`` — user signed a snapshot of a composition
#     state for review-only sharing. ``payload_digest`` carries the
#     content-address of the snapshot blob in the payload store;
#     ``expires_at`` carries the lifetime stamped onto the signed token.
#   * ``export_yaml`` — user exported the pipeline YAML for the named
#     composition state. ``payload_digest`` and ``expires_at`` stay NULL.
#
# Mirrors the Phase 18 ``interpretation_events_table`` precedent of "one new
# table per event family, closed-enum CHECK on event_type, nullable optional
# columns for event-type-specific data" rather than splitting into two tables
# or extending ``proposal_events_table`` / ``audit_access_log_table`` whose
# closed CHECKs are governance boundaries (see plan 19a §"Audit-event
# recording" for the adjudication).
#
# CLOSED LIST on ``event_type``. Adding a third value requires (a) amending
# 19a (or its successor), (b) the same four-step ceremony as the other
# closed enums in this file, and (c) a schema-change cohort with epoch bump.
# NO SILENT EXTENSION.
composer_completion_events_table = Table(
    "composer_completion_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "composition_state_id",
        String,
        ForeignKey("composition_states.id"),
        nullable=True,
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Content-address of the snapshot blob in the payload store.
    # Populated for ``mark_ready_for_review`` events; NULL for ``export_yaml``.
    Column("payload_digest", String, nullable=True),
    # Token expiry stamped on the signed share token at issuance time.
    # Populated for ``mark_ready_for_review`` events; NULL for ``export_yaml``.
    Column("expires_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        "event_type IN ('mark_ready_for_review', 'export_yaml')",
        name="ck_composer_completion_events_type",
    ),
    # Per-event-type partial NULL/NOT-NULL constraints.
    #
    # ``payload_digest`` and ``expires_at`` are nullable at the column
    # level because ``export_yaml`` rows don't carry them — but they MUST
    # be present for ``mark_ready_for_review`` rows (the audit row would
    # otherwise be useless: no blob pointer, no token expiry). The
    # ``(A) = (B IS NOT NULL)`` idiom asserts the iff relation:
    #   1 = 1  (event=mark... and digest present): OK
    #   0 = 0  (event=export and digest absent):   OK
    #   1 = 0  (event=mark... and digest absent):  CHECK fails — bug
    #   0 = 1  (event=export and digest present):  CHECK fails — bug
    #
    # ``composition_state_id`` is the audit-anchor for both event types
    # (the state being marked / the state being exported). Both writers
    # populate it; the constraint pins that contract at the DB level so
    # a future writer cannot silently drop it.
    CheckConstraint(
        "(event_type = 'mark_ready_for_review') = (payload_digest IS NOT NULL)",
        name="ck_composer_completion_events_digest_iff_mark_ready",
    ),
    CheckConstraint(
        "(event_type = 'mark_ready_for_review') = (expires_at IS NOT NULL)",
        name="ck_composer_completion_events_expires_iff_mark_ready",
    ),
    CheckConstraint(
        "composition_state_id IS NOT NULL",
        name="ck_composer_completion_events_composition_state_id_required",
    ),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_composer_completion_events_state_session",
    ),
)

Index(
    "ix_composer_completion_events_session_created",
    composer_completion_events_table.c.session_id,
    composer_completion_events_table.c.created_at,
)

# ``skill_markdown_history`` (F-5c) — content-addressed archive of every
# distinct ``pipeline_composer.md`` version seen at runtime.
#
# One row per (SHA-256 hash, filename) pair. The compose loop upserts
# (INSERT OR IGNORE) on first use of a hash, capturing the exact text that
# was in memory when the LLM was prompted. This makes every
# ``composer_skill_hash`` on ``interpretation_events`` rows forensically
# traceable: an auditor can retrieve the exact skill prompt from this
# table.
#
# Storage cost is negligible — one row per distinct deploy of the skill
# markdown. Content is TEXT (not BLOB) because the skill file is UTF-8
# Markdown.
skill_markdown_history_table = Table(
    "skill_markdown_history",
    metadata,
    Column("hash", String, primary_key=True),  # SHA-256 hex, 64 chars
    Column("filename", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
)


@dataclass(frozen=True, slots=True)
class PostgresqlAuditDDL:
    """One immutable PostgreSQL audit-trigger installation unit."""

    table: Table
    trigger_name: str
    function_name: str
    function_sql: str
    trigger_sql: str


# This tuple is the sole authority for the PostgreSQL audit DDL installed by
# both fresh metadata bootstrap and the release-0.7.1 one-shot migration.  Use
# CREATE FUNCTION rather than CREATE OR REPLACE: both supported callers prove
# the functions absent before installation, so replacement would only hide an
# unrecognized mixed state.
POSTGRESQL_AUDIT_DDL_COHORT: tuple[PostgresqlAuditDDL, ...] = (
    PostgresqlAuditDDL(
        table=interpretation_events_table,
        trigger_name="trg_interpretation_events_immutable_resolved",
        function_name="elspeth_interpretation_events_immutable_resolved",
        function_sql="""
        CREATE FUNCTION elspeth_interpretation_events_immutable_resolved()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF OLD.resolved_at IS NOT NULL AND (
            NEW.accepted_value IS DISTINCT FROM OLD.accepted_value OR
            NEW.resolved_at IS DISTINCT FROM OLD.resolved_at OR
            NEW.actor IS DISTINCT FROM OLD.actor OR
            NEW.choice IS DISTINCT FROM OLD.choice
          ) THEN
            RAISE EXCEPTION 'interpretation_events: resolved rows are immutable'
              USING ERRCODE = '23000';
          END IF;
          RETURN NEW;
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_interpretation_events_immutable_resolved "
            "BEFORE UPDATE ON interpretation_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_interpretation_events_immutable_resolved()"
        ),
    ),
    PostgresqlAuditDDL(
        table=interpretation_events_table,
        trigger_name="trg_interpretation_events_no_delete_resolved",
        function_name="elspeth_interpretation_events_no_delete_resolved",
        function_sql="""
        CREATE FUNCTION elspeth_interpretation_events_no_delete_resolved()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF OLD.resolved_at IS NOT NULL
             AND EXISTS (SELECT 1 FROM sessions WHERE sessions.id = OLD.session_id) THEN
            RAISE EXCEPTION 'interpretation_events: resolved rows are append-only'
              USING ERRCODE = '23000';
          END IF;
          RETURN OLD;
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_interpretation_events_no_delete_resolved "
            "BEFORE DELETE ON interpretation_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_interpretation_events_no_delete_resolved()"
        ),
    ),
    PostgresqlAuditDDL(
        table=composer_completion_events_table,
        trigger_name="trg_composer_completion_events_no_update",
        function_name="elspeth_composer_completion_events_no_update",
        function_sql="""
        CREATE FUNCTION elspeth_composer_completion_events_no_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          RAISE EXCEPTION 'composer_completion_events is append-only; UPDATE is forbidden'
            USING ERRCODE = '23000';
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_composer_completion_events_no_update "
            "BEFORE UPDATE ON composer_completion_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_composer_completion_events_no_update()"
        ),
    ),
    PostgresqlAuditDDL(
        table=composer_completion_events_table,
        trigger_name="trg_composer_completion_events_no_delete",
        function_name="elspeth_composer_completion_events_no_delete",
        function_sql="""
        CREATE FUNCTION elspeth_composer_completion_events_no_delete()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          RAISE EXCEPTION 'composer_completion_events is append-only; DELETE is forbidden'
            USING ERRCODE = '23000';
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_composer_completion_events_no_delete "
            "BEFORE DELETE ON composer_completion_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_composer_completion_events_no_delete()"
        ),
    ),
    PostgresqlAuditDDL(
        table=chat_messages_table,
        trigger_name="trg_chat_messages_immutable_content",
        function_name="elspeth_chat_messages_immutable_content",
        function_sql="""
        CREATE FUNCTION elspeth_chat_messages_immutable_content()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          RAISE EXCEPTION 'chat_messages.content is append-only'
            USING ERRCODE = '23000';
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_chat_messages_immutable_content "
            "BEFORE UPDATE OF content ON chat_messages "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_chat_messages_immutable_content()"
        ),
    ),
    PostgresqlAuditDDL(
        table=chat_messages_table,
        trigger_name="trg_chat_messages_no_delete",
        function_name="elspeth_chat_messages_no_delete",
        function_sql="""
        CREATE FUNCTION elspeth_chat_messages_no_delete()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF EXISTS (SELECT 1 FROM sessions WHERE id = OLD.session_id) THEN
            RAISE EXCEPTION 'chat_messages rows are append-only'
              USING ERRCODE = '23000';
          END IF;
          RETURN OLD;
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_chat_messages_no_delete "
            "BEFORE DELETE ON chat_messages "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_chat_messages_no_delete()"
        ),
    ),
    PostgresqlAuditDDL(
        table=guided_operations_table,
        trigger_name="trg_guided_operations_terminal_immutable",
        function_name="elspeth_guided_operations_terminal_immutable",
        function_sql="""
        CREATE FUNCTION elspeth_guided_operations_terminal_immutable()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF OLD.status IN ('completed', 'failed') THEN
            RAISE EXCEPTION 'guided_operations terminal rows are immutable'
              USING ERRCODE = '23000';
          END IF;
          RETURN NEW;
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_guided_operations_terminal_immutable "
            "BEFORE UPDATE ON guided_operations "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_guided_operations_terminal_immutable()"
        ),
    ),
    PostgresqlAuditDDL(
        table=guided_operation_events_table,
        trigger_name="trg_guided_operation_events_no_update",
        function_name="elspeth_guided_operation_events_no_update",
        function_sql="""
        CREATE FUNCTION elspeth_guided_operation_events_no_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          RAISE EXCEPTION 'guided_operation_events is append-only; UPDATE is forbidden'
            USING ERRCODE = '23000';
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_guided_operation_events_no_update "
            "BEFORE UPDATE ON guided_operation_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_guided_operation_events_no_update()"
        ),
    ),
    PostgresqlAuditDDL(
        table=guided_operation_events_table,
        trigger_name="trg_guided_operation_events_no_delete",
        function_name="elspeth_guided_operation_events_no_delete",
        function_sql="""
        CREATE FUNCTION elspeth_guided_operation_events_no_delete()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF EXISTS (SELECT 1 FROM sessions WHERE id = OLD.session_id) THEN
            RAISE EXCEPTION 'guided_operation_events is append-only; DELETE is forbidden'
              USING ERRCODE = '23000';
          END IF;
          RETURN OLD;
        END;
        $$
        """,
        trigger_sql=(
            "CREATE TRIGGER trg_guided_operation_events_no_delete "
            "BEFORE DELETE ON guided_operation_events "
            "FOR EACH ROW EXECUTE FUNCTION elspeth_guided_operation_events_no_delete()"
        ),
    ),
)

# F-4 / F-23 — append-only triggers.
#
# ``trg_interpretation_events_immutable_resolved`` protects the audit
# integrity of resolved rows: once ``resolved_at`` is non-NULL, the four
# settled fields (``accepted_value``, ``resolved_at``, ``actor``,
# ``choice``) cannot change. Updates to non-settled columns on resolved
# rows are still permitted (e.g. backfilling provenance), but flipping the
# decision is not.
#
# ``trg_interpretation_events_no_delete_resolved`` extends that resolved-row
# protection to direct DELETE while deliberately leaving unresolved PENDING
# rows deletable for orphan recovery. Whole-session archival remains allowed:
# SQLite foreign-key cascades execute the child BEFORE DELETE trigger after
# the parent ``sessions`` row is gone from the trigger's view, so the trigger
# uses parent-row existence to distinguish direct child deletion from the
# schema-owned archive cascade.
#
# ``trg_chat_messages_immutable_content`` enforces append-only semantics
# for ``chat_messages.content`` — once a message is written, its body
# cannot be edited. ``chat_messages`` is an audit anchor via
# ``blobs.created_from_message_id``; without this trigger, post-hoc
# editing would invalidate the lineage audit.
#
# ``trg_chat_messages_no_delete`` protects the same anchor against row
# deletion. A missing chat row makes blob lineage unverifiable even if the
# blob FK never existed, so the trigger is unconditional.
#
# ``IF NOT EXISTS`` makes the SQLite DDL idempotent across repeated
# ``metadata.create_all`` calls. ``event.listen`` is **table-scoped**, not
# metadata-scoped, so the trigger DDL fires only when this specific table
# is created (not on every metadata.create_all() call for tables that
# already exist). Each invariant has dialect-specific DDL below. SQLite uses
# ``RAISE(ABORT)``; PostgreSQL uses PL/pgSQL functions that raise SQLSTATE
# 23000 so SQLAlchemy classifies the refusal as an integrity error. Trigger
# names are identical across dialects because sessions.schema validates the
# live catalogue by those stable names.
event.listen(
    interpretation_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_interpretation_events_immutable_resolved "
        "BEFORE UPDATE ON interpretation_events "
        "FOR EACH ROW BEGIN "
        "  SELECT CASE "
        "    WHEN OLD.resolved_at IS NOT NULL AND ("
        "      NEW.accepted_value IS NOT OLD.accepted_value OR "
        "      NEW.resolved_at IS NOT OLD.resolved_at OR "
        "      NEW.actor IS NOT OLD.actor OR "
        "      NEW.choice IS NOT OLD.choice"
        "    ) THEN RAISE(ABORT, 'interpretation_events: resolved rows are immutable') "
        "  END; "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    interpretation_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_interpretation_events_no_delete_resolved "
        "BEFORE DELETE ON interpretation_events "
        "FOR EACH ROW "
        "WHEN OLD.resolved_at IS NOT NULL "
        "AND EXISTS (SELECT 1 FROM sessions WHERE sessions.id = OLD.session_id) "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'interpretation_events: resolved rows are append-only'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

# Phase 6A — completion-event triggers.
#
# Unlike ``interpretation_events`` (which permits DELETE on PENDING rows for
# orphan recovery) completion events have no recovery path. Both triggers
# are **unconditional ABORT**: every recorded mark-ready-for-review or YAML
# export is a permanent audit fact.
#
# Both triggers ship from day 1 — completion events have no recovery
# path, unlike the ``interpretation_events`` PENDING-row carve-out where a
# follow-up landed the DELETE trigger after the original cohort.
event.listen(
    composer_completion_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_composer_completion_events_no_update "
        "BEFORE UPDATE ON composer_completion_events "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'composer_completion_events is append-only; UPDATE is forbidden'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    composer_completion_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_composer_completion_events_no_delete "
        "BEFORE DELETE ON composer_completion_events "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'composer_completion_events is append-only; DELETE is forbidden'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    chat_messages_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_chat_messages_immutable_content "
        "BEFORE UPDATE OF content ON chat_messages "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'chat_messages.content is append-only'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    chat_messages_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_chat_messages_no_delete "
        "BEFORE DELETE ON chat_messages "
        "FOR EACH ROW "
        "WHEN EXISTS (SELECT 1 FROM sessions WHERE id = OLD.session_id) "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'chat_messages rows are append-only'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    guided_operations_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_guided_operations_terminal_immutable "
        "BEFORE UPDATE ON guided_operations "
        "FOR EACH ROW "
        "WHEN OLD.status IN ('completed', 'failed') "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'guided_operations terminal rows are immutable'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    guided_operation_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_guided_operation_events_no_update "
        "BEFORE UPDATE ON guided_operation_events "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'guided_operation_events is append-only; UPDATE is forbidden'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

event.listen(
    guided_operation_events_table,
    "after_create",
    DDL(  # type: ignore[no-untyped-call]
        "CREATE TRIGGER IF NOT EXISTS trg_guided_operation_events_no_delete "
        "BEFORE DELETE ON guided_operation_events "
        "FOR EACH ROW "
        "WHEN EXISTS (SELECT 1 FROM sessions WHERE id = OLD.session_id) "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'guided_operation_events is append-only; DELETE is forbidden'); "
        "END;"
    ).execute_if(dialect="sqlite"),
)

for postgresql_audit_ddl in POSTGRESQL_AUDIT_DDL_COHORT:
    event.listen(
        postgresql_audit_ddl.table,
        "after_create",
        DDL(postgresql_audit_ddl.function_sql).execute_if(dialect="postgresql"),  # type: ignore[no-untyped-call]
    )
    event.listen(
        postgresql_audit_ddl.table,
        "after_create",
        DDL(postgresql_audit_ddl.trigger_sql).execute_if(dialect="postgresql"),  # type: ignore[no-untyped-call]
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
#
# BOTH ``sqlite_where=`` AND ``postgresql_where=`` must be set to the same
# predicate. Without ``postgresql_where=`` SQLAlchemy emits a non-partial
# unique index on ``session_id`` alone on Postgres, which silently
# over-restricts the invariant from "at most one ACTIVE run per session"
# to "at most one run per session ever" — a real audit-integrity defect
# (the second run in a session would fail to insert with a unique-violation
# unrelated to its actual status). Mirrors the project pattern at
# ``uq_chat_messages_tool_call_id`` above where both keys are set.
#
# The shared schema-shape collector validates the live index's name, ordered
# columns, uniqueness, and dialect-specific WHERE predicate.  The session
# validator additionally enforces model-side sqlite/postgresql predicate
# symmetry before inspecting either runtime dialect.
Index(
    "uq_runs_one_active_per_session",
    runs_table.c.session_id,
    unique=True,
    sqlite_where=runs_table.c.status.in_(["pending", "running"]),
    postgresql_where=runs_table.c.status.in_(["pending", "running"]),
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
    # --- Inline-blob provenance ---
    # creation_modality: closed enum — how this blob's content was produced.
    # Non-nullable with default "verbatim" so blobs created before these
    # columns existed retain a valid value when the DB is rebuilt (we have no
    # users yet; the operator deletes-and-recreates per
    # project_db_migration_policy).
    #
    # Tier 1 crash-on-anomaly applies to reads: the read path
    # (_row_to_blob_record / _blob_row_to_tool_dict) asserts the value is a
    # valid CreationModality member; the SQL CHECK below mirrors the
    # closed enum at the DB layer.
    #
    # CLOSED LIST — do not extend without design review.  Three locations
    # must change together: the CreationModality StrEnum at
    # contracts/enums.py, this column's CHECK, and the catalog-side
    # tier-3 plumbing on the read/write paths (tools.py).  No fourth
    # location.
    Column("creation_modality", Text, nullable=False, server_default="verbatim"),
    # created_from_message_id: FK to chat_messages.id of the user message
    # that triggered the set_pipeline / create_blob call.  The composite
    # FK (created_from_message_id, session_id) → (chat_messages.id,
    # chat_messages.session_id) closes the cross-session lineage hole the
    # same way fk_chat_messages_parent_assistant_session does for tool
    # rows.  Nullable to admit (a) the migration-window default insert
    # path before the route layer is wired, and (b) future programmatic
    # blob creation paths that have no originating user message
    # (e.g. pipeline-emitted blobs whose audit anchor is the run record).
    # LLM-authored modalities narrow this column-level nullability through
    # ck_blobs_creating_llm_provenance_nullability: they MUST carry a
    # non-blank originating message anchor.
    Column("created_from_message_id", Text, nullable=True),
    # LLM-provenance columns: populated for the three LLM-authored
    # modalities (llm_generated, disambiguated,
    # llm_generated_then_amended); NULL for verbatim.  Required together —
    # a blob cannot claim LLM authorship without naming the model,
    # version, provider, the composer-skill prompt hash, and the
    # tool-call arguments hash.  The all-or-nothing invariant is enforced
    # by the ck_blobs_creating_llm_provenance_nullability CHECK below;
    # the L0 _LLM_AUTHORED_CREATION_MODALITIES frozenset is the Python
    # mirror used by the write path's pre-insert guard.
    Column("creating_model_identifier", String, nullable=True),
    Column("creating_model_version", String, nullable=True),
    Column("creating_provider", String, nullable=True),
    Column("creating_composer_skill_hash", String, nullable=True),
    Column("creating_arguments_hash", String, nullable=True),
    # Composite FK: (created_from_message_id, session_id) must reference an
    # existing (chat_messages.id, chat_messages.session_id) pair.  Mirrors
    # fk_chat_messages_parent_assistant_session above.  ON DELETE RESTRICT
    # (NOT CASCADE) because the blob is the audit anchor — deleting its
    # originating message would silently truncate the lineage walk and
    # leave the audit trail confidently asserting "this blob came from
    # message X" while X no longer exists.  An operator who genuinely
    # wants to delete a message must first delete or re-anchor every blob
    # bound to it; the RESTRICT failure surfaces that explicitly.
    ForeignKeyConstraint(
        ["created_from_message_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_blobs_created_from_message_session",
        ondelete="RESTRICT",
    ),
    CheckConstraint(
        "creation_modality IN ('verbatim', 'llm_generated', 'disambiguated', 'llm_generated_then_amended')",
        name="ck_blobs_creation_modality",
    ),
    # LLM-provenance nullability invariant: the three LLM-authored
    # modalities MUST carry a non-blank created_from_message_id plus all
    # five creating_* fields; verbatim MUST carry no creating_* fields and
    # may carry or omit the triggering message anchor. Keep the verbatim
    # branch explicit: a biconditional over the LLM-populated predicate
    # lets partial/whitespace creating_* values on verbatim rows pass as
    # "not fully populated", which weakens the audit nullability contract.
    CheckConstraint(
        _blobs_creating_llm_provenance_check(dialect="sqlite"),
        name="ck_blobs_creating_llm_provenance_nullability",
    ).ddl_if(dialect="sqlite"),
    CheckConstraint(
        _blobs_creating_llm_provenance_check(dialect="postgresql"),
        name="ck_blobs_creating_llm_provenance_nullability",
    ).ddl_if(dialect="postgresql"),
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
    # side (``re.compile(r"^[a-f0-9]{64}$")``).
    #
    # Two dialect-conditional expressions enforce the same invariant:
    # SQLite uses ``NOT GLOB '*[^a-f0-9]*'`` (Postgres has no GLOB
    # operator); PostgreSQL uses POSIX regex ``~ '^[a-f0-9]+$'`` (SQLite
    # has no built-in POSIX regex). Both reject the same set of
    # malformed content_hash values on rows with status='ready':
    # NULL, length≠64, or any non-lowercase-hex character. The
    # ``length(...) = 64`` clause anchors the length check on both sides,
    # and the character-class checks anchor the alphabet — together they
    # are equivalent to ``re.compile(r"^[a-f0-9]{64}$")`` on the write
    # side.
    #
    # The shared name ``ck_blobs_ready_hash`` lets the schema validator
    # (sessions/schema.py:_validate_named_checks) pair the live constraint
    # with the dialect-active metadata constraint. The SQL text is still
    # compared after the ``ddl_if(dialect=...)`` filter, so a same-named
    # stale CHECK with weaker semantics is rejected at startup.
    #
    # If a third dialect is introduced, add its V0 check expression here
    # with a matching ``ddl_if(dialect=...)`` instead of adding a
    # migration path.
    CheckConstraint(
        "status != 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 64 AND content_hash NOT GLOB '*[^a-f0-9]*')",
        name="ck_blobs_ready_hash",
    ).ddl_if(dialect="sqlite"),
    CheckConstraint(
        "status != 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 64 AND content_hash ~ '^[a-f0-9]+$')",
        name="ck_blobs_ready_hash",
    ).ddl_if(dialect="postgresql"),
)

# Index for the reverse-lookup path: "given a chat message, which inline
# blobs were created from it?"  Backs the inline-blob lineage walk —
# chat_messages.id → blobs.created_from_message_id → blobs.id — without
# triggering a full table scan when an auditor opens the lineage drawer
# on a session with thousands of blobs.
Index(
    "ix_blobs_created_from_message_id",
    blobs_table.c.created_from_message_id,
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

blob_inline_resolutions_table = Table(
    "blob_inline_resolutions",
    metadata,
    Column(
        "run_id",
        String,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("attempt", Integer, nullable=False, server_default="1"),
    Column("field_path", String, nullable=False),
    Column("blob_id", String, nullable=False),
    Column("content_hash", String, nullable=False),
    Column("byte_length", Integer, nullable=False),
    Column("mime_type", String, nullable=False),
    Column("encoding", String, nullable=False),
    Column("resolved_at", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint(
        "run_id",
        "field_path",
        "blob_id",
        "attempt",
        name="pk_blob_inline_resolutions",
    ),
    CheckConstraint(
        "length(content_hash) = 64",
        name="ck_blob_inline_resolutions_hash_format",
    ),
    CheckConstraint(
        "encoding IN ('utf-8', 'utf-8-sig', 'utf-16', 'latin-1')",
        name="ck_blob_inline_resolutions_encoding",
    ),
    CheckConstraint(
        "field_path LIKE 'source.options.%' OR field_path LIKE 'node:%.options.%' OR field_path LIKE 'output:%.options.%'",
        name="ck_blob_inline_resolutions_field_path",
    ),
    CheckConstraint(
        "byte_length >= 0",
        name="ck_blob_inline_resolutions_byte_length",
    ),
)
Index("ix_blob_inline_resolutions_blob_id", blob_inline_resolutions_table.c.blob_id)
Index("ix_blob_inline_resolutions_run_id", blob_inline_resolutions_table.c.run_id)

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
    Column("sequence", Integer, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("event_type", String, nullable=False),
    Column("data", JSON, nullable=False),
    UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
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
    CheckConstraint(
        _AUTH_PROVIDER_TYPE_CHECK,
        name="ck_user_secrets_auth_provider_type",
    ),
)
Index("ix_user_secrets_user_provider", user_secrets_table.c.user_id, user_secrets_table.c.auth_provider_type)

# ``user_preferences`` — per-user composer settings.
#
# One row per user holding (a) the user's chosen default composer mode for
# NEW sessions, (b) the dismissal timestamp for the one-time "we changed
# the default" banner, and (c) the completed-at timestamp for the first-run
# hello-world tutorial. Per-session mode toggles live in chat panel state
# and do NOT touch this row; this is exclusively account-level preference
# state.
#
# ``user_id`` is opaque and matches ``sessions_table.user_id``. No FK is
# declared because auth providers vary across deployments and there is no
# canonical users table in the session DB to reference.
#
# CLOSED-LIST default_composer_mode. Permitted values are exactly
# {"guided", "freeform"} — enforced at the Tier-3 boundary by Pydantic
# Literal and at Tier-1 read time by the PreferencesService read guard
# (any stored value outside this set crashes the read with the offending
# value named). Extending the set requires (a) a Pydantic ``ComposerMode``
# Literal amendment, (b) a service read-guard amendment, and (c) a UI
# affordance for the new mode — do not extend silently here.
user_preferences_table = Table(
    "user_preferences",
    metadata,
    Column("user_id", String, primary_key=True),
    Column(
        "default_composer_mode",
        String,
        nullable=False,
        server_default="guided",
    ),
    # NULL = banner not yet dismissed; non-NULL = dismissed-at timestamp.
    Column("banner_dismissed_at", DateTime(timezone=True), nullable=True),
    # NULL = freeform introduction visible; non-NULL = dismissed account-wide.
    Column("freeform_intro_dismissed_at", DateTime(timezone=True), nullable=True),
    # NULL = tutorial not completed/reset; non-NULL = completed-at timestamp.
    Column("tutorial_completed_at", DateTime(timezone=True), nullable=True),
    # First-run tutorial resume state (elspeth-918f4434b3). NULL = no
    # in-progress tutorial (the Welcome bookend is never persisted — nothing
    # has started). Non-NULL = the macro stage to resume at after a reload.
    # The stage vocabulary mirrors the frontend TutorialStep union
    # (tutorialMachine.ts) minus "welcome"; the CHECK constraint below plus
    # the Tier-1 read guard in ``PreferencesService._row_to_prefs`` are the
    # closed-list pair, same lockstep rule as ``default_composer_mode``.
    Column("tutorial_stage", String, nullable=True),
    # The tutorial session the persisted stage belongs to, so resume
    # re-attaches to the SAME session instead of abandoning it. run id +
    # source-data hash are recorded once the tutorial run completes so the
    # audit step can resume with zero re-execution (no silent LLM re-spend).
    Column("tutorial_session_id", String, nullable=True),
    Column("tutorial_run_id", String, nullable=True),
    Column("tutorial_source_data_hash", String, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "default_composer_mode IN ('guided', 'freeform')",
        name="ck_user_preferences_default_composer_mode",
    ),
    CheckConstraint(
        "tutorial_stage IS NULL OR tutorial_stage IN ('guided', 'run', 'audit', 'graduation')",
        name="ck_user_preferences_tutorial_stage",
    ),
    # Empty-string user_id would silently key a "shared" preferences row
    # that any unauthenticated principal could write to if upstream auth
    # ever regressed. SQLite's PRIMARY KEY accepts '' as a distinct
    # value, so the only defence at the schema layer is an explicit
    # length check. Mirrors the empty-string guards on other
    # principal-keyed tables.
    CheckConstraint(
        "length(user_id) > 0",
        name="ck_user_preferences_user_id_non_empty",
    ),
)

# ``audit_access_log`` — INERT IN PHASE 1A.
#
# This table records who viewed audit-grade message data (the eventual
# ``include_tool_rows=true`` route surface). 1A lands the table SCHEMA
# ONLY: no route writes it, no service method writes it, no fixture
# writes it. The destructive session-DB schema reset
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
