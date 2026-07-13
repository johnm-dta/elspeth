"""Regression tests for dialect-aware SQLAlchemy metadata shape checks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import (
    CHAR,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import CITEXT, DOUBLE_PRECISION, ENUM, JSONB
from sqlalchemy.engine import Connection, Dialect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateTable

from elspeth.core.landscape.schema import metadata as landscape_metadata
from elspeth.core.landscape.schema import runs_table
from elspeth.core.schema_shape import (
    SchemaShapeIssue,
    collect_metadata_shape_issues,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import metadata as session_metadata


def test_seeded_from_cache_uses_dialect_native_boolean_server_default() -> None:
    postgres_ddl = str(CreateTable(runs_table).compile(dialect=postgresql.dialect()))
    sqlite_ddl = str(CreateTable(runs_table).compile(dialect=sqlite.dialect()))

    assert "seeded_from_cache BOOLEAN DEFAULT false NOT NULL" in postgres_ddl
    assert "seeded_from_cache BOOLEAN DEFAULT 0 NOT NULL" not in postgres_ddl
    assert "seeded_from_cache BOOLEAN DEFAULT 0 NOT NULL" in sqlite_ddl


def _demo_metadata(
    mutation: str | None = None,
    *,
    extra_ordinary_index: bool = False,
    omit_ordinary_index: bool = False,
    postgres_only_objects: bool = False,
) -> MetaData:
    metadata = MetaData()
    Table("parents", metadata, Column("id", String(64), primary_key=True))

    id_column = Column("id", String(64), primary_key=mutation != "primary_key", nullable=False)
    label_column = Column(
        "label",
        Text if mutation == "type" else String(64),
        nullable=mutation == "nullable",
    )
    status_column = Column(
        "status",
        String(16),
        nullable=False,
        server_default=None if mutation == "default_presence" else text("'LIVE'" if mutation == "default" else "'live'"),
    )
    parent_id_column = Column("parent_id", String(64), nullable=False)
    extra_column = Column("unexpected", String(8), nullable=True)

    columns = [id_column, label_column, status_column, parent_id_column]
    if mutation == "column_order":
        columns = [label_column, id_column, status_column, parent_id_column]
    elif mutation == "additional_column":
        columns.append(extra_column)

    table = Table(
        "demo",
        metadata,
        *columns,
        ForeignKeyConstraint(
            ["parent_id"],
            ["parents.id"],
            name="fk_demo_parent",
            onupdate="RESTRICT" if mutation == "fk_onupdate" else "CASCADE",
            ondelete="SET NULL" if mutation == "fk_ondelete" else "CASCADE",
            deferrable=mutation != "fk_deferrability",
            initially="IMMEDIATE" if mutation == "fk_deferrability" else "DEFERRED",
        ),
        CheckConstraint(
            "status IN ('LIVE', 'bundled')" if mutation == "check_literal" else "status IN ('live', 'bundled')",
            name="ck_demo_status",
        ),
        UniqueConstraint(
            "id" if mutation == "unique_constraint" else "label",
            "status",
        ),
    )

    unique_columns = (
        (table.c.status, table.c.label)
        if mutation == "unique_index_order"
        else (table.c.status, table.c.parent_id)
        if mutation == "unique_index_columns"
        else (table.c.label, table.c.status)
    )
    live_predicate = table.c.status == ("bundled" if mutation == "unique_index_predicate" else "live")
    Index(
        "uq_demo_live",
        *unique_columns,
        unique=True,
        sqlite_where=live_predicate,
        postgresql_where=live_predicate,
    )

    if not omit_ordinary_index:
        ordinary_columns = (
            (table.c.status, table.c.label)
            if mutation == "index_order"
            else (table.c.id, table.c.status)
            if mutation == "index_columns"
            else (table.c.label, table.c.status)
        )
        Index("ix_demo_label_status", *ordinary_columns)
    if extra_ordinary_index:
        Index("ix_demo_external_performance", table.c.parent_id)

    if mutation == "additional_unique_index":
        Index("uq_demo_unexpected", table.c.parent_id, unique=True)
    elif mutation == "additional_unique_constraint":
        table.append_constraint(UniqueConstraint(table.c.parent_id))
    elif mutation == "additional_check":
        table.append_constraint(CheckConstraint("length(label) > 0", name="ck_demo_unexpected"))
    elif mutation == "additional_fk":
        table.append_constraint(ForeignKeyConstraint([table.c.label], [metadata.tables["parents"].c.id], name="fk_demo_unexpected"))

    if postgres_only_objects:
        table.append_constraint(CheckConstraint("length(label) < 65", name="ck_demo_postgres_only_attached").ddl_if(dialect="postgresql"))
        Index("ix_demo_postgres_only", table.c.parent_id).ddl_if(dialect="postgresql")

    return metadata


def _sqlite_issues(
    expected: MetaData,
    actual: MetaData,
    *,
    allowed_missing_index_names: frozenset[str] = frozenset(),
) -> tuple[SchemaShapeIssue, ...]:
    engine = create_engine("sqlite:///:memory:")
    actual.create_all(engine)
    return collect_metadata_shape_issues(
        inspect(engine),
        expected,
        dialect=engine.dialect,
        present_tables=frozenset({"demo"}),
        allowed_missing_index_names=allowed_missing_index_names,
    )


@pytest.mark.parametrize(
    ("mutation", "subject_fragment"),
    [
        ("column_order", "demo column mismatch"),
        ("additional_column", "demo column mismatch"),
        ("primary_key", "demo.id primary-key mismatch"),
        ("nullable", "demo.label nullable mismatch"),
        ("type", "demo.label type mismatch"),
        ("default", "demo.status server-default mismatch"),
        ("default_presence", "demo.status server-default mismatch"),
        ("fk_ondelete", "demo foreign-key mismatch"),
        ("fk_onupdate", "demo foreign-key mismatch"),
        ("fk_deferrability", "demo foreign-key mismatch"),
        ("additional_fk", "demo foreign-key mismatch"),
        ("unique_constraint", "demo UNIQUE constraint mismatch"),
        ("additional_unique_constraint", "demo UNIQUE constraint mismatch"),
        ("check_literal", "demo.ck_demo_status CHECK constraint SQL mismatch"),
        ("additional_check", "demo CHECK constraint mismatch"),
        ("unique_index_columns", "demo.uq_demo_live unique index column mismatch"),
        ("unique_index_order", "demo.uq_demo_live unique index column mismatch"),
        ("unique_index_predicate", "demo.uq_demo_live unique index predicate mismatch"),
        ("index_columns", "demo.ix_demo_label_status index column mismatch"),
        ("index_order", "demo.ix_demo_label_status index column mismatch"),
        ("additional_unique_index", "demo unique index mismatch"),
    ],
)
def test_collector_reports_each_isolated_sqlite_shape_drift(mutation: str, subject_fragment: str) -> None:
    issues = _sqlite_issues(_demo_metadata(), _demo_metadata(mutation))

    assert issues
    assert any(subject_fragment in issue.subject for issue in issues)


def test_collector_aggregates_multiple_issues_in_deterministic_order() -> None:
    actual = _demo_metadata("additional_column")
    demo = actual.tables["demo"]
    demo.append_constraint(CheckConstraint("length(label) > 0", name="ck_demo_unexpected"))

    first = _sqlite_issues(_demo_metadata(), actual)
    second = _sqlite_issues(_demo_metadata(), actual)

    assert len(first) >= 2
    assert first == second
    assert first[0].subject == "demo column mismatch"


def test_collector_rejects_proposal_events_fk_without_deferred_deferrability() -> None:
    def proposal_metadata(*, deferred: bool) -> MetaData:
        metadata = MetaData()
        Table("sessions", metadata, Column("id", String, primary_key=True))
        Table("composition_proposals", metadata, Column("id", String, primary_key=True))
        Table(
            "proposal_events",
            metadata,
            Column("id", String, primary_key=True),
            Column("session_id", String, nullable=False),
            Column("proposal_id", String, nullable=False),
            ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
            ForeignKeyConstraint(
                ["proposal_id"],
                ["composition_proposals.id"],
                ondelete="CASCADE",
                deferrable=True if deferred else None,
                initially="DEFERRED" if deferred else None,
            ),
        )
        return metadata

    engine = create_engine("sqlite:///:memory:")
    proposal_metadata(deferred=False).create_all(engine)
    issues = collect_metadata_shape_issues(
        inspect(engine),
        proposal_metadata(deferred=True),
        dialect=engine.dialect,
        present_tables=frozenset({"proposal_events"}),
    )

    assert any(issue.subject == "proposal_events foreign-key mismatch" for issue in issues)


def test_collector_rejects_changed_unnamed_landscape_composite_unique() -> None:
    def run_sources_metadata(*, drifted: bool) -> MetaData:
        metadata = MetaData()
        Table(
            "run_sources",
            metadata,
            Column("run_id", String(64), nullable=False),
            Column("source_name", String(64), nullable=False),
            Column("plugin_name", String(128), nullable=False),
            UniqueConstraint("run_id", "plugin_name" if drifted else "source_name"),
        )
        return metadata

    engine = create_engine("sqlite:///:memory:")
    run_sources_metadata(drifted=True).create_all(engine)
    issues = collect_metadata_shape_issues(
        inspect(engine),
        run_sources_metadata(drifted=False),
        dialect=engine.dialect,
        present_tables=frozenset({"run_sources"}),
    )

    assert any(issue.subject == "run_sources UNIQUE constraint mismatch" for issue in issues)


def test_collector_accepts_fresh_minimal_sqlite_schema() -> None:
    assert _sqlite_issues(_demo_metadata(), _demo_metadata()) == ()


def test_collector_tolerates_additional_ordinary_nonunique_index() -> None:
    assert _sqlite_issues(_demo_metadata(), _demo_metadata(extra_ordinary_index=True)) == ()


def test_collector_allows_named_index_to_be_missing_only_when_explicitly_allowed() -> None:
    actual = _demo_metadata(omit_ordinary_index=True)

    rejected = _sqlite_issues(_demo_metadata(), actual)
    accepted = _sqlite_issues(
        _demo_metadata(),
        actual,
        allowed_missing_index_names=frozenset({"ix_demo_label_status"}),
    )

    assert any("ix_demo_label_status" in issue.subject for issue in rejected)
    assert accepted == ()


def test_allowed_index_name_still_requires_full_shape_when_present() -> None:
    issues = _sqlite_issues(
        _demo_metadata(),
        _demo_metadata("index_order"),
        allowed_missing_index_names=frozenset({"ix_demo_label_status"}),
    )

    assert any("ix_demo_label_status" in issue.subject for issue in issues)


def test_collector_honors_ddl_if_dialect_filtering() -> None:
    expected = _demo_metadata(postgres_only_objects=True)
    actual = _demo_metadata()

    assert _sqlite_issues(expected, actual) == ()


def test_fresh_full_session_schema_has_no_shape_issues() -> None:
    engine = create_session_engine("sqlite:///:memory:")
    session_metadata.create_all(engine)

    assert (
        collect_metadata_shape_issues(
            inspect(engine),
            session_metadata,
            dialect=engine.dialect,
            present_tables=frozenset(session_metadata.tables),
        )
        == ()
    )


def test_fresh_full_landscape_schema_has_no_shape_issues() -> None:
    engine = create_engine("sqlite:///:memory:")
    landscape_metadata.create_all(engine)

    assert (
        collect_metadata_shape_issues(
            inspect(engine),
            landscape_metadata,
            dialect=engine.dialect,
            present_tables=frozenset(landscape_metadata.tables),
        )
        == ()
    )


class _StaticInspector:
    """Small Inspector-shaped test double for pinned PostgreSQL reflection forms."""

    def __init__(
        self,
        *,
        columns: list[dict[str, Any]],
        primary_key: list[str],
        foreign_keys: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        unique_constraints: list[dict[str, Any]],
        indexes: list[dict[str, Any]],
        bind: Connection | None = None,
    ) -> None:
        self._columns = columns
        self._primary_key = primary_key
        self._foreign_keys = foreign_keys
        self._checks = checks
        self._unique_constraints = unique_constraints
        self._indexes = indexes
        self.bind = bind

    def get_columns(self, table_name: str) -> list[dict[str, Any]]:
        assert table_name == "pg_demo"
        return self._columns

    def get_pk_constraint(self, table_name: str) -> dict[str, Any]:
        assert table_name == "pg_demo"
        return {"constrained_columns": self._primary_key}

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        assert table_name == "pg_demo"
        return self._foreign_keys

    def get_check_constraints(self, table_name: str) -> list[dict[str, Any]]:
        assert table_name == "pg_demo"
        return self._checks

    def get_unique_constraints(self, table_name: str) -> list[dict[str, Any]]:
        assert table_name == "pg_demo"
        return self._unique_constraints

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        assert table_name == "pg_demo"
        return self._indexes


def _postgres_equivalence_metadata() -> MetaData:
    metadata = MetaData()
    Table("parents", metadata, Column("id", Integer, primary_key=True), schema="public")
    table = Table(
        "pg_demo",
        metadata,
        Column("seq", Integer, primary_key=True),
        Column("score", DOUBLE_PRECISION, nullable=False),
        Column("observed_at", DateTime(timezone=True), nullable=False),
        Column("payload", Text, nullable=False, server_default=text("'{}'")),
        Column("status", String(16), nullable=False, server_default=text("'live'")),
        Column("parent_id", Integer, nullable=False),
        ForeignKeyConstraint(
            ["parent_id"],
            ["public.parents.id"],
            name="fk_pg_demo_parent",
        ),
        CheckConstraint("status IN ('live', 'bundled')", name="ck_pg_demo_status"),
        UniqueConstraint("status", "parent_id", name="uq_pg_demo_status_parent"),
    )
    Index(
        "ix_pg_demo_live",
        table.c.parent_id,
        postgresql_where=table.c.status == "live",
        sqlite_where=table.c.status == "live",
    )
    return metadata


def _postgres_equivalence_inspector(*, literal_case_drift: bool = False) -> _StaticInspector:
    return _StaticInspector(
        columns=[
            {
                "name": "seq",
                "type": Integer(),
                "nullable": False,
                "default": "nextval('pg_demo_seq_seq'::regclass)",
            },
            {"name": "score", "type": postgresql.FLOAT(), "nullable": False, "default": None},
            {
                "name": "observed_at",
                "type": postgresql.TIMESTAMP(timezone=True),
                "nullable": False,
                "default": None,
            },
            {"name": "payload", "type": Text(), "nullable": False, "default": "('{}'::text)"},
            {
                "name": "status",
                "type": String(16),
                "nullable": False,
                "default": "'LIVE'::character varying" if literal_case_drift else "'live'::character varying",
            },
            {"name": "parent_id", "type": Integer(), "nullable": False, "default": None},
        ],
        primary_key=["seq"],
        foreign_keys=[
            {
                "name": "fk_pg_demo_parent",
                "constrained_columns": ["parent_id"],
                "referred_schema": None,
                "referred_table": "parents",
                "referred_columns": ["id"],
                "options": {
                    "onupdate": "NO ACTION",
                    "ondelete": "NO ACTION",
                    "deferrable": False,
                    "initially": "IMMEDIATE",
                    "match": "SIMPLE",
                },
            }
        ],
        checks=[
            {
                "name": "ck_pg_demo_status",
                "sqltext": "((public.pg_demo.status)::text = ANY (ARRAY['live'::character varying, 'bundled'::character varying]))",
            }
        ],
        unique_constraints=[
            {"name": "uq_pg_demo_status_parent", "column_names": ["status", "parent_id"]},
        ],
        indexes=[
            {
                "name": "uq_pg_demo_status_parent",
                "column_names": ["status", "parent_id"],
                "unique": True,
                "duplicates_constraint": "uq_pg_demo_status_parent",
                "dialect_options": {},
            },
            {
                "name": "ix_pg_demo_live",
                "column_names": ["parent_id"],
                "unique": False,
                "dialect_options": {"postgresql_where": "((public.pg_demo.status)::text = 'live'::text)"},
            },
        ],
    )


def test_collector_pins_known_postgresql_reflection_equivalences() -> None:
    dialect: Dialect = postgresql.dialect()
    dialect.default_schema_name = "public"

    issues = collect_metadata_shape_issues(
        _postgres_equivalence_inspector(),  # type: ignore[arg-type]
        _postgres_equivalence_metadata(),
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )

    assert issues == ()


def test_postgresql_literal_case_drift_is_not_normalized_away() -> None:
    dialect: Dialect = postgresql.dialect()
    dialect.default_schema_name = "public"

    issues = collect_metadata_shape_issues(
        _postgres_equivalence_inspector(literal_case_drift=True),  # type: ignore[arg-type]
        _postgres_equivalence_metadata(),
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.status server-default mismatch" for issue in issues)


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        (
            "status IS NULL OR status IN ('live', 'bundled')",
            "status IS NULL OR (status::text = ANY (ARRAY['live'::text, 'bundled'::text]::text[]))",
        ),
        (
            "status NOT IN ('live', 'bundled')",
            "status::text <> ALL (ARRAY['live'::text, 'bundled'::text]::text[])",
        ),
        ("path LIKE 'source.options.%'", "path::text ~~ 'source.options.%'::text"),
        (
            "length(btrim(path, chr(9) || chr(10) || chr(32))) > 0",
            "length(btrim(path, (chr(9) || chr(10)) || chr(32))) > 0",
        ),
        (
            "(path IS NULL AND status IS NULL) OR (path IS NOT NULL AND status IS NOT NULL)",
            "path IS NULL AND status IS NULL OR path IS NOT NULL AND status IS NOT NULL",
        ),
        (
            "(status = 'live' AND length(path) > 0) OR status = 'bundled'",
            "status::text = 'live'::text AND length(path) > 0 OR status::text = 'bundled'::text",
        ),
        (
            "(status = 'live' AND length(path) > 0 AND length(status) > 0) OR status = 'bundled'",
            "status::text = 'live'::text AND length(path) > 0 AND length(status) > 0 OR status::text = 'bundled'::text",
        ),
        (
            "status <> 'live' OR (path IS NULL OR status IS NULL)",
            "status::text <> 'live'::text OR path IS NULL OR status IS NULL",
        ),
        (
            "(status = 'live') = (path IS NOT NULL)",
            "status::text = 'live'::text) = (path IS NOT NULL",
        ),
        (
            "(status = 'live' OR path IS NULL) = (path IS NOT NULL)",
            "status::text = 'live'::text OR path IS NULL) = (path IS NOT NULL",
        ),
    ],
)
def test_collector_pins_postgresql_check_reflection_forms(declared_sql: str, reflected_sql: str) -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("status", String(16), nullable=True),
        Column("path", Text, nullable=True),
        CheckConstraint(declared_sql, name="ck_pg_demo_shape"),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "status", "type": String(16), "nullable": True, "default": None},
            {"name": "path", "type": Text(), "nullable": True, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[{"name": "ck_pg_demo_shape", "sqltext": reflected_sql}],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert issues == ()


def test_postgresql_integer_default_reflection_matches_quoted_declared_scalar() -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("attempt", Integer, nullable=False, server_default="1"),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "attempt", "type": Integer(), "nullable": False, "default": "1"},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert issues == ()


def test_postgresql_partial_index_array_cast_and_qualification_are_equivalent() -> None:
    dialect = postgresql.dialect()
    dialect.default_schema_name = "public"
    metadata = MetaData()
    table = Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("status", String(16), nullable=False),
    )
    predicate = table.c.status.in_(["pending", "running"])
    Index(
        "uq_pg_demo_active",
        table.c.id,
        unique=True,
        sqlite_where=predicate,
        postgresql_where=predicate,
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "status", "type": String(16), "nullable": False, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[
            {
                "name": "uq_pg_demo_active",
                "column_names": ["id"],
                "unique": True,
                "dialect_options": {
                    "postgresql_where": "public.pg_demo.status::text = ANY ((ARRAY['pending'::text, 'running'::text])::text[])"
                },
            }
        ],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )

    assert issues == ()


@pytest.mark.parametrize(
    ("expected_default", "actual_default"),
    [
        ("'live'", "'LIVE'"),
        ('"MiXeD"', '"mixed"'),
        (r"E'live\\n'", r"E'LIVE\\n'"),
        ("$$live$$", "$$LIVE$$"),
    ],
)
def test_default_normalization_preserves_quoted_literal_bytes(expected_default: str, actual_default: str) -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Text, nullable=False, server_default=text(expected_default)),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "value", "type": Text(), "nullable": False, "default": actual_default},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.value server-default mismatch" for issue in issues)


_ALLOWED_BUILTIN_PROOF_ROWS = [
    ("allowed", "length", 1, "1001"),
    ("allowed", "btrim", 1, "1002"),
    ("allowed", "btrim", 2, "1003"),
    ("allowed", "chr", 1, "1004"),
    ("source", "text", 0, "25"),
    ("source", "varchar", 0, "1043"),
    ("source", "int4", 0, "23"),
    ("operator_allowed", "||", 2, "654"),
    ("text_result", "chr", 1, "int4"),
    ("operator_text_result", "||", 2, "text,text"),
]


def _builtin_visibility_connection(
    *blocked_calls: tuple[str, int, str],
    literal_only_calls: tuple[tuple[str, int, str], ...] = (),
    int4_literal_calls: tuple[tuple[str, int, str], ...] = (),
    variadic_calls: tuple[tuple[str, int, str], ...] = (),
    blocked_concat_shapes: tuple[tuple[str, str], ...] = (),
    text_result_calls: tuple[tuple[str, int, str], ...] = (),
    operator_text_result_shapes: tuple[tuple[str, str], ...] = (),
    proof_rows: list[tuple[str, str, int, str | None]] | None = None,
    query_error: bool = False,
) -> Connection:
    connection = MagicMock(spec=Connection)
    connection.in_transaction.side_effect = [False, True]
    if query_error:
        connection.execute.side_effect = SQLAlchemyError("visibility proof unavailable")
    else:
        connection.execute.return_value.all.return_value = (
            proof_rows
            if proof_rows is not None
            else [
                *(
                    row
                    for row in _ALLOWED_BUILTIN_PROOF_ROWS
                    if not (row[0] == "text_result" and (row[1], row[2], row[3]) in blocked_calls)
                    and not (row[0] == "operator_text_result" and tuple(str(row[3]).split(",", maxsplit=1)) in blocked_concat_shapes)
                ),
                *(("blocked", name, arity, family) for name, arity, family in blocked_calls),
                *(("literal_only", name, arity, family) for name, arity, family in literal_only_calls),
                *(("int4_literal", name, arity, family) for name, arity, family in int4_literal_calls),
                *(("variadic", name, arity, family) for name, arity, family in variadic_calls),
                *(("operator_blocked", "||", 2, f"{left},{right}") for left, right in blocked_concat_shapes),
                *(("text_result", name, arity, family) for name, arity, family in text_result_calls),
                *(("operator_text_result", "||", 2, f"{left},{right}") for left, right in operator_text_result_shapes),
            ]
        )
    return connection


def _static_check_issues(
    declared_sql: str,
    reflected_sql: str,
    *,
    blocked_builtin_calls: tuple[tuple[str, int, str], ...] | None = None,
    literal_only_builtin_calls: tuple[tuple[str, int, str], ...] = (),
    int4_literal_builtin_calls: tuple[tuple[str, int, str], ...] = (),
    variadic_builtin_calls: tuple[tuple[str, int, str], ...] = (),
    blocked_concat_shapes: tuple[tuple[str, str], ...] = (),
    text_result_builtin_calls: tuple[tuple[str, int, str], ...] = (),
    operator_text_result_shapes: tuple[tuple[str, str], ...] = (),
    builtin_proof_rows: list[tuple[str, str, int, str | None]] | None = None,
    builtin_visibility_query_error: bool = False,
    builtin_connection: Connection | None = None,
) -> tuple[SchemaShapeIssue, ...]:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("a", JSONB),
        Column("b", JSONB),
        Column("c", JSONB),
        Column("absid", Integer),
        Column("amount", Integer),
        Column("value", Integer),
        Column("value_text", String(16)),
        Column("code", CHAR(2)),
        Column("email", CITEXT),
        Column("priority", ENUM("low", "high", name="priority_level")),
        Column("__elspeth_literal_0__", String(16)),
        CheckConstraint(declared_sql, name="ck_pg_demo_false_equivalence"),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "a", "type": JSONB(), "nullable": True, "default": None},
            {"name": "b", "type": JSONB(), "nullable": True, "default": None},
            {"name": "c", "type": JSONB(), "nullable": True, "default": None},
            {"name": "absid", "type": Integer(), "nullable": True, "default": None},
            {"name": "amount", "type": Integer(), "nullable": True, "default": None},
            {"name": "value", "type": Integer(), "nullable": True, "default": None},
            {"name": "value_text", "type": String(16), "nullable": True, "default": None},
            {"name": "code", "type": CHAR(2), "nullable": True, "default": None},
            {"name": "email", "type": CITEXT(), "nullable": True, "default": None},
            {
                "name": "priority",
                "type": ENUM("low", "high", name="priority_level"),
                "nullable": True,
                "default": None,
            },
            {"name": "__elspeth_literal_0__", "type": String(16), "nullable": True, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[{"name": "ck_pg_demo_false_equivalence", "sqltext": reflected_sql}],
        unique_constraints=[],
        indexes=[],
        bind=(
            builtin_connection
            if builtin_connection is not None
            else None
            if blocked_builtin_calls is None
            else _builtin_visibility_connection(
                *blocked_builtin_calls,
                literal_only_calls=literal_only_builtin_calls,
                int4_literal_calls=int4_literal_builtin_calls,
                variadic_calls=variadic_builtin_calls,
                blocked_concat_shapes=blocked_concat_shapes,
                text_result_calls=text_result_builtin_calls,
                operator_text_result_shapes=operator_text_result_shapes,
                proof_rows=builtin_proof_rows,
                query_error=builtin_visibility_query_error,
            )
        ),
    )
    return collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("NOT (a = 'x' AND b = 'y')", "NOT a = 'x' AND b = 'y'"),
        ("abs(id) > 0", "absid > 0"),
        ("value::integer > 0", "value > 0"),
        ("value::integer > 0", "value::bigint > 0"),
        ("amount > 0", "amount::integer > 0"),
        ("trusted.next_id() > 0", "attacker.next_id() > 0"),
        ("trusted.pg_demo.amount > 0", "attacker.pg_demo.amount > 0"),
        ("__elspeth_literal_0__ = 'safe'", "'safe' = 'safe'"),
    ],
)
def test_collector_rejects_unproven_expression_equivalences(declared_sql: str, reflected_sql: str) -> None:
    issues = _static_check_issues(declared_sql, reflected_sql)

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_jsonb_concatenation_grouping() -> None:
    issues = _static_check_issues("((a || b) || c) = a", "(a || (b || c)) = a")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_explicit_varchar_cast_inside_function_argument() -> None:
    issues = _static_check_issues("trusted.f(value_text::text) = true", "trusted.f(value_text) = true")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_citext_equality_semantics() -> None:
    issues = _static_check_issues("email = 'Admin'", "email::text = 'Admin'::text")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_native_enum_ordering_semantics() -> None:
    issues = _static_check_issues("priority < 'high'", "priority::text < 'high'::text")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("value_text = 'live'", "value_text::text = 'live'::text"),
        (
            "value_text IN ('live', 'bundled')",
            "value_text::text = ANY (ARRAY['live'::text, 'bundled'::text]::text[])",
        ),
    ],
)
def test_collector_allows_actual_side_varchar_cast_noise_at_comparison_nodes(
    declared_sql: str,
    reflected_sql: str,
) -> None:
    assert _static_check_issues(declared_sql, reflected_sql) == ()


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("length(value_text) > 0", "length(value_text::text) > 0"),
        ("btrim(value_text, ' ') <> ''", "btrim(value_text::text, ' '::text) <> ''::text"),
    ],
)
def test_collector_allows_actual_side_varchar_cast_noise_in_text_builtins(
    declared_sql: str,
    reflected_sql: str,
) -> None:
    assert _static_check_issues(declared_sql, reflected_sql, blocked_builtin_calls=()) == ()


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("length(value_text, 'suffix') > 0", "length(value_text::text, 'suffix') > 0"),
        (
            "btrim(value_text, ' ', 'extra') <> ''",
            "btrim(value_text::text, ' ', 'extra') <> ''",
        ),
    ],
)
def test_collector_rejects_text_cast_relaxation_for_non_builtin_arities(
    declared_sql: str,
    reflected_sql: str,
) -> None:
    issues = _static_check_issues(declared_sql, reflected_sql, blocked_builtin_calls=())

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_fails_closed_without_builtin_visibility_proof() -> None:
    issues = _static_check_issues("length(value_text) > 0", "length(value_text::text) > 0")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_fails_closed_when_builtin_visibility_query_fails() -> None:
    issues = _static_check_issues(
        "length(value_text) > 0",
        "length(value_text::text) > 0",
        blocked_builtin_calls=(),
        builtin_visibility_query_error=True,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize(
    "invalid_detail",
    ["", "not-an-oid", "0", "-1", "16384"],
)
@pytest.mark.parametrize("allowed_name", ["length", "chr"])
def test_collector_fails_closed_for_invalid_allowed_builtin_oid(invalid_detail: str, allowed_name: str) -> None:
    proof_rows = [*_ALLOWED_BUILTIN_PROOF_ROWS]
    allowed_index = next(index for index, row in enumerate(proof_rows) if row[:2] == ("allowed", allowed_name))
    proof_rows[allowed_index] = ("allowed", allowed_name, 1, invalid_detail)
    issues = _static_check_issues(
        "length(value_text) > 0",
        "length(value_text::text) > 0",
        blocked_builtin_calls=(),
        builtin_proof_rows=proof_rows,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize("invalid_detail", ["", "not-an-oid", "0", "-1", "16384"])
@pytest.mark.parametrize("source_family", ["text", "int4"])
def test_collector_fails_closed_for_invalid_source_type_oid(invalid_detail: str, source_family: str) -> None:
    proof_rows = [*_ALLOWED_BUILTIN_PROOF_ROWS]
    source_index = next(index for index, row in enumerate(proof_rows) if row[:2] == ("source", source_family))
    proof_rows[source_index] = ("source", source_family, 0, invalid_detail)
    issues = _static_check_issues(
        "length(value_text) > 0",
        "length(value_text::text) > 0",
        blocked_builtin_calls=(),
        builtin_proof_rows=proof_rows,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize(
    "missing_index",
    [index for index, row in enumerate(_ALLOWED_BUILTIN_PROOF_ROWS) if row[0] in {"allowed", "source", "operator_allowed"}],
)
def test_collector_fails_closed_for_missing_catalog_proof_row(missing_index: int) -> None:
    proof_rows = [row for index, row in enumerate(_ALLOWED_BUILTIN_PROOF_ROWS) if index != missing_index]
    issues = _static_check_issues(
        "length(value_text) > 0",
        "length(value_text::text) > 0",
        blocked_builtin_calls=(),
        builtin_proof_rows=proof_rows,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_fails_closed_for_invalid_text_concat_operator_oid() -> None:
    proof_rows = [*_ALLOWED_BUILTIN_PROOF_ROWS]
    operator_index = next(index for index, row in enumerate(proof_rows) if row[0] == "operator_allowed")
    proof_rows[operator_index] = ("operator_allowed", "||", 2, "16384")
    issues = _static_check_issues(
        "btrim(value_text, chr(49) || chr(50)) IS NOT NULL",
        "btrim(value_text::text, chr(49) || chr(50)) IS NOT NULL",
        blocked_builtin_calls=(),
        builtin_proof_rows=proof_rows,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize(
    ("blocked_call", "declared_sql", "fresh_reflected_sql", "mutated_reflected_sql"),
    [
        (
            ("length", 1, "varchar"),
            "length(value_text) > 0",
            "length(value_text) > 0",
            "length(value_text::text) > 0",
        ),
        (
            ("btrim", 1, "varchar"),
            "btrim(value_text) = 'CUSTOM'",
            "btrim(value_text) = 'CUSTOM'::text",
            "btrim(value_text::text) = 'CUSTOM'::text",
        ),
    ],
)
def test_collector_rejects_cast_relaxation_when_visible_builtin_shadow_exists(
    blocked_call: tuple[str, int, str],
    declared_sql: str,
    fresh_reflected_sql: str,
    mutated_reflected_sql: str,
) -> None:
    assert _static_check_issues(declared_sql, fresh_reflected_sql, blocked_builtin_calls=(blocked_call,)) == ()

    issues = _static_check_issues(
        declared_sql,
        mutated_reflected_sql,
        blocked_builtin_calls=(blocked_call,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("length(value_text) > 0", "length(value_text::text) > 0"),
        (
            "btrim(value_text) <> ''",
            "btrim(value_text::text) <> ''::text",
        ),
        (
            "btrim(value_text, ' ') <> ''",
            "btrim(value_text::text, ' '::text) <> ''::text",
        ),
    ],
)
def test_collector_keeps_builtin_relaxation_for_unrelated_visible_overloads(
    declared_sql: str,
    reflected_sql: str,
) -> None:
    assert _static_check_issues(declared_sql, reflected_sql, blocked_builtin_calls=()) == ()


def test_collector_keeps_btrim_relaxation_for_known_text_second_expression() -> None:
    assert (
        _static_check_issues(
            "btrim(value_text, chr(32)) IS NOT NULL",
            "btrim(value_text::text, chr(32)) IS NOT NULL",
            blocked_builtin_calls=(),
        )
        == ()
    )


def test_collector_rejects_btrim_relaxation_for_unknown_second_argument_family() -> None:
    issues = _static_check_issues(
        "btrim(value_text, value) IS NOT NULL",
        "btrim(value_text::text, value) IS NOT NULL",
        blocked_builtin_calls=(),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_rejects_unknown_literal_shadow_column_cast_mutation() -> None:
    call_key = ("btrim_unknown", 2, "varchar")
    assert (
        _static_check_issues(
            "btrim(value_text, '1') IS NOT NULL",
            "btrim(value_text, '1') IS NOT NULL",
            blocked_builtin_calls=(call_key,),
        )
        == ()
    )

    issues = _static_check_issues(
        "btrim(value_text, '1') IS NOT NULL",
        "btrim(value_text::text, '1') IS NOT NULL",
        blocked_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_int4_coercion_for_unique_unknown_literal_btrim_winner() -> None:
    call_key = ("btrim_unknown", 2, "varchar")
    declared_sql = "btrim(value_text, '1') = 'CUSTOM'"
    reflected_sql = "btrim(value_text, 1) = 'CUSTOM'::text"

    assert (
        _static_check_issues(
            declared_sql,
            reflected_sql,
            blocked_builtin_calls=(call_key,),
            int4_literal_builtin_calls=(call_key,),
            text_result_builtin_calls=(call_key,),
        )
        == ()
    )

    mutated_issues = _static_check_issues(
        declared_sql,
        "btrim(value_text::text, 1) = 'CUSTOM'::text",
        blocked_builtin_calls=(call_key,),
        int4_literal_builtin_calls=(call_key,),
        text_result_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in mutated_issues)


def test_collector_rejects_shadowed_chr_column_cast_mutation() -> None:
    blocked_chr = ("chr", 1, "int4")
    declared_sql = "btrim(value_text, chr(49)) IS NOT NULL"
    assert _static_check_issues(declared_sql, declared_sql, blocked_builtin_calls=(blocked_chr,)) == ()

    issues = _static_check_issues(
        declared_sql,
        "btrim(value_text::varchar, chr(49)) IS NOT NULL",
        blocked_builtin_calls=(blocked_chr,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_uses_unique_exact_chr_winner_text_result_for_outer_cast_relaxation() -> None:
    blocked_chr = ("chr", 1, "int4")
    proof_kwargs = {
        "blocked_builtin_calls": (blocked_chr,),
        "text_result_builtin_calls": (blocked_chr,),
    }

    assert (
        _static_check_issues(
            "btrim(value_text, chr(49)) IS NOT NULL",
            "btrim(value_text::text, chr(49)) IS NOT NULL",
            **proof_kwargs,
        )
        == ()
    )

    changed_inner_issues = _static_check_issues(
        "btrim(value_text, chr(49)) IS NOT NULL",
        "btrim(value_text::text, chr(50)) IS NOT NULL",
        **proof_kwargs,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in changed_inner_issues)


@pytest.mark.parametrize(
    "chr_argument",
    ["'49'::varchar", "2147483648", "-2147483649"],
)
def test_collector_rejects_chr_without_proven_int4_argument(chr_argument: str) -> None:
    issues = _static_check_issues(
        f"btrim(value_text, chr({chr_argument})) IS NOT NULL",
        f"btrim(value_text::text, chr({chr_argument})) IS NOT NULL",
        blocked_builtin_calls=(),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


@pytest.mark.parametrize("chr_argument", ["0", "2147483647", "-2147483648", "'49'::int4"])
def test_collector_accepts_chr_with_proven_int4_argument(chr_argument: str) -> None:
    assert (
        _static_check_issues(
            f"btrim(value_text, chr({chr_argument})) IS NOT NULL",
            f"btrim(value_text::text, chr({chr_argument})) IS NOT NULL",
            blocked_builtin_calls=(),
        )
        == ()
    )


def test_collector_trusts_qualified_chr_bootstrap_identity_despite_unqualified_shadow() -> None:
    assert (
        _static_check_issues(
            "btrim(value_text, pg_catalog.chr(49)) IS NOT NULL",
            "btrim(value_text::text, pg_catalog.chr(49)) IS NOT NULL",
            blocked_builtin_calls=(("chr", 1, "int4"),),
        )
        == ()
    )


def test_collector_fails_closed_for_qualified_chr_without_validated_identity() -> None:
    proof_rows = [*_ALLOWED_BUILTIN_PROOF_ROWS]
    chr_index = next(index for index, row in enumerate(proof_rows) if row[:2] == ("allowed", "chr"))
    proof_rows[chr_index] = ("allowed", "chr", 1, "16384")
    issues = _static_check_issues(
        "btrim(value_text, pg_catalog.chr(49)) IS NOT NULL",
        "btrim(value_text::text, pg_catalog.chr(49)) IS NOT NULL",
        blocked_builtin_calls=(),
        builtin_proof_rows=proof_rows,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_requires_proven_text_operand_for_text_concatenation() -> None:
    assert (
        _static_check_issues(
            "btrim(value_text, chr(49) || ' ') IS NOT NULL",
            "btrim(value_text::text, chr(49) || ' ') IS NOT NULL",
            blocked_builtin_calls=(),
        )
        == ()
    )

    issues = _static_check_issues(
        "btrim(value_text, 'a' || 'b') IS NOT NULL",
        "btrim(value_text::text, 'a' || 'b') IS NOT NULL",
        blocked_builtin_calls=(),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_rejects_text_concat_relaxation_when_operator_identity_is_blocked() -> None:
    declared_sql = "btrim(value_text, chr(49) || chr(50)) IS NOT NULL"
    assert (
        _static_check_issues(
            declared_sql,
            declared_sql,
            blocked_builtin_calls=(),
            blocked_concat_shapes=(("text", "text"),),
        )
        == ()
    )

    issues = _static_check_issues(
        declared_sql,
        "btrim(value_text::text, chr(49) || chr(50)) IS NOT NULL",
        blocked_builtin_calls=(),
        blocked_concat_shapes=(("text", "text"),),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_uses_unique_exact_concat_winner_text_result_for_outer_cast_relaxation() -> None:
    declared_sql = "btrim(value_text, chr(49) || chr(50)) IS NOT NULL"
    proof_kwargs = {
        "blocked_builtin_calls": (),
        "blocked_concat_shapes": (("text", "text"),),
        "operator_text_result_shapes": (("text", "text"),),
    }

    assert (
        _static_check_issues(
            declared_sql,
            "btrim(value_text::text, chr(49) || chr(50)) IS NOT NULL",
            **proof_kwargs,
        )
        == ()
    )

    changed_inner_issues = _static_check_issues(
        declared_sql,
        "btrim(value_text::text, chr(49) || chr(51)) IS NOT NULL",
        **proof_kwargs,
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in changed_inner_issues)


def test_collector_ignores_blocked_concat_shape_for_different_operand_families() -> None:
    assert (
        _static_check_issues(
            "btrim(value_text, chr(49) || chr(50)) IS NOT NULL",
            "btrim(value_text::text, chr(49) || chr(50)) IS NOT NULL",
            blocked_builtin_calls=(),
            blocked_concat_shapes=(("varchar", "varchar"),),
        )
        == ()
    )


def test_collector_allows_only_literal_casts_for_exact_btrim_varchar_text_shadow() -> None:
    call_key = ("btrim_unknown", 2, "varchar")
    declared_sql = "btrim(value_text, ' ') = 'CUSTOM'"
    fresh_reflected_sql = "btrim(value_text, ' '::text) = 'CUSTOM'::text"
    assert (
        _static_check_issues(
            declared_sql,
            fresh_reflected_sql,
            blocked_builtin_calls=(call_key,),
            literal_only_builtin_calls=(call_key,),
        )
        == ()
    )

    mutated_issues = _static_check_issues(
        declared_sql,
        "btrim(value_text::text, ' '::text) = 'CUSTOM'::text",
        blocked_builtin_calls=(call_key,),
        literal_only_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in mutated_issues)


def test_collector_rejects_literal_casts_for_ambiguous_btrim_two_argument_shadow() -> None:
    call_key = ("btrim_unknown", 2, "varchar")
    issues = _static_check_issues(
        "btrim(value_text, ' ') = 'CUSTOM'",
        "btrim(value_text, ' '::text) = 'CUSTOM'::text",
        blocked_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_variadic_shadow_argument_identity() -> None:
    call_key = ("length", 1, "varchar")
    assert (
        _static_check_issues(
            "length(value_text) > 0",
            "length(VARIADIC ARRAY[value_text]) > 0",
            blocked_builtin_calls=(call_key,),
            variadic_builtin_calls=(call_key,),
        )
        == ()
    )

    issues = _static_check_issues(
        "length(value_text) > 0",
        "length(VARIADIC ARRAY[value_text::text]) > 0",
        blocked_builtin_calls=(call_key,),
        variadic_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_variadic_shadow_later_argument_identity() -> None:
    call_key = ("btrim", 2, "varchar")
    assert (
        _static_check_issues(
            "btrim(value_text, chr(32)) IS NOT NULL",
            "btrim(value_text, VARIADIC ARRAY[chr(32)]) IS NOT NULL",
            blocked_builtin_calls=(call_key,),
            variadic_builtin_calls=(call_key,),
        )
        == ()
    )

    issues = _static_check_issues(
        "btrim(value_text, chr(32)) IS NOT NULL",
        "btrim(value_text::text, VARIADIC ARRAY[chr(32)]) IS NOT NULL",
        blocked_builtin_calls=(call_key,),
        variadic_builtin_calls=(call_key,),
    )

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_preserves_explicit_char_to_text_comparison_cast() -> None:
    issues = _static_check_issues("code::text = 'A '", "code = 'A '::bpchar")

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_allows_reflected_text_literal_for_explicit_char_to_text_comparison() -> None:
    assert _static_check_issues("code::text = 'A '", "code::text = 'A '::text") == ()


def test_collector_allows_reflected_bpchar_literal_for_uncast_char_comparison() -> None:
    assert _static_check_issues("code = 'A '", "code = 'A '::bpchar") == ()


@pytest.mark.parametrize(
    ("declared_sql", "reflected_sql"),
    [
        ("trusted.f('live'::text) = true", "trusted.f('live') = true"),
        ("trusted.f(ARRAY['live']::text[]) = true", "trusted.f(ARRAY['live']) = true"),
    ],
)
def test_collector_preserves_casts_that_control_function_overload(
    declared_sql: str,
    reflected_sql: str,
) -> None:
    issues = _static_check_issues(declared_sql, reflected_sql)

    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_collector_rejects_cast_added_to_explicit_nextval_default() -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Integer, nullable=False, server_default=text("nextval('trusted_seq'::regclass)")),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {
                "name": "value",
                "type": Integer(),
                "nullable": False,
                "default": "nextval('trusted_seq'::regclass)::bigint",
            },
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.value server-default mismatch" for issue in issues)


def test_collector_rejects_unrelated_implicit_autoincrement_sequence() -> None:
    metadata = MetaData()
    Table("pg_demo", metadata, Column("seq", Integer, primary_key=True))
    inspector = _StaticInspector(
        columns=[
            {
                "name": "seq",
                "type": Integer(),
                "nullable": False,
                "default": "nextval('unrelated_sequence'::regclass)",
            }
        ],
        primary_key=["seq"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.seq server-default mismatch" for issue in issues)


def test_collector_rejects_right_named_implicit_sequence_from_wrong_schema() -> None:
    metadata = MetaData()
    Table("pg_demo", metadata, Column("seq", Integer, primary_key=True))
    inspector = _StaticInspector(
        columns=[
            {
                "name": "seq",
                "type": Integer(),
                "nullable": False,
                "default": "nextval('attacker.pg_demo_seq_seq'::regclass)",
            }
        ],
        primary_key=["seq"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )
    dialect = postgresql.dialect()
    dialect.default_schema_name = "public"

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.seq server-default mismatch" for issue in issues)


def test_collector_rejects_quoted_implicit_sequence_identifier_containing_dot() -> None:
    metadata = MetaData()
    Table("pg_demo", metadata, Column("id", Integer, primary_key=True))
    inspector = _StaticInspector(
        columns=[
            {
                "name": "id",
                "type": Integer(),
                "nullable": False,
                "default": "nextval('\"public.pg_demo_id_seq\"'::regclass)",
            }
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )
    dialect = postgresql.dialect()
    dialect.default_schema_name = "public"

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.id server-default mismatch" for issue in issues)


@pytest.mark.parametrize(
    ("declared_nulls_not_distinct", "reflected_nulls_not_distinct"),
    [(False, True), (True, False)],
)
def test_collector_rejects_unique_constraint_null_semantics_drift(
    declared_nulls_not_distinct: bool,
    reflected_nulls_not_distinct: bool,
) -> None:
    metadata = MetaData()
    table = Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Integer),
    )
    constraint = (
        UniqueConstraint("value", name="uq_pg_demo_value", postgresql_nulls_not_distinct=True)
        if declared_nulls_not_distinct
        else UniqueConstraint("value", name="uq_pg_demo_value")
    )
    table.append_constraint(constraint)
    reflected_options = {"postgresql_nulls_not_distinct": reflected_nulls_not_distinct}
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "value", "type": Integer(), "nullable": True, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[
            {
                "name": "uq_pg_demo_value",
                "column_names": ["value"],
                "dialect_options": reflected_options,
            }
        ],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo UNIQUE constraint mismatch" for issue in issues)


@pytest.mark.parametrize(
    ("declared_nulls_not_distinct", "reflected_nulls_not_distinct"),
    [(False, True), (True, False)],
)
def test_collector_rejects_unique_index_null_semantics_drift(
    declared_nulls_not_distinct: bool,
    reflected_nulls_not_distinct: bool,
) -> None:
    metadata = MetaData()
    table = Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Integer),
    )
    if declared_nulls_not_distinct:
        Index("uq_pg_demo_value", table.c.value, unique=True, postgresql_nulls_not_distinct=True)
    else:
        Index("uq_pg_demo_value", table.c.value, unique=True)
    reflected_options = {"postgresql_nulls_not_distinct": True} if reflected_nulls_not_distinct else {}
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "value", "type": Integer(), "nullable": True, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[
            {
                "name": "uq_pg_demo_value",
                "column_names": ["value"],
                "unique": True,
                "dialect_options": reflected_options,
            }
        ],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo.uq_pg_demo_value unique index mismatch" for issue in issues)


def _postgres_fk_issues(
    *,
    default_schema: str = "public",
    expected_schema: str | None = "public",
    constrained_columns: list[str] | None = None,
    referred_schema: str | None = None,
    referred_table: str = "parents",
    referred_columns: list[str] | None = None,
    deferrable: bool = True,
    initially: str = "DEFERRED",
    match: str = "FULL",
) -> tuple[SchemaShapeIssue, ...]:
    metadata = MetaData()
    parent_key = "parents.id" if expected_schema is None else f"{expected_schema}.parents.id"
    Table("parents", metadata, Column("id", Integer, primary_key=True), schema=expected_schema)
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("parent_id", Integer, nullable=False),
        Column("other_parent_id", Integer, nullable=False),
        ForeignKeyConstraint(
            ["parent_id"],
            [parent_key],
            deferrable=True,
            initially="DEFERRED",
            match="FULL",
        ),
    )
    dialect = postgresql.dialect()
    dialect.default_schema_name = default_schema
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "parent_id", "type": Integer(), "nullable": False, "default": None},
            {"name": "other_parent_id", "type": Integer(), "nullable": False, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[
            {
                "name": None,
                "constrained_columns": constrained_columns or ["parent_id"],
                "referred_schema": referred_schema,
                "referred_table": referred_table,
                "referred_columns": referred_columns or ["id"],
                "options": {
                    "deferrable": deferrable,
                    "initially": initially,
                    "match": match,
                },
            }
        ],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )
    return collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=dialect,
        present_tables=frozenset({"pg_demo"}),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"constrained_columns": ["other_parent_id"]},
        {"referred_schema": "other"},
        {"referred_table": "other_parents"},
        {"referred_columns": ["other_id"]},
        {"deferrable": False},
        {"initially": "IMMEDIATE"},
        {"match": "SIMPLE"},
    ],
)
def test_collector_rejects_each_isolated_foreign_key_shape_drift(changes: dict[str, Any]) -> None:
    assert any(issue.subject == "pg_demo foreign-key mismatch" for issue in _postgres_fk_issues(**changes))


def test_postgresql_public_schema_is_not_default_under_custom_search_path() -> None:
    issues = _postgres_fk_issues(default_schema="tenant", expected_schema="public", referred_schema=None)

    assert any(issue.subject == "pg_demo foreign-key mismatch" for issue in issues)


def test_postgresql_reflected_default_schema_name_matches_unqualified_metadata() -> None:
    assert _postgres_fk_issues(default_schema="tenant", expected_schema=None, referred_schema="tenant") == ()


@pytest.mark.parametrize(
    "extra_index",
    [
        {
            "name": "ix_pg_demo_expression",
            "column_names": [None],
            "expressions": ["(1 / amount)"],
            "unique": False,
            "dialect_options": {},
        },
        {
            "name": "ix_pg_demo_partial",
            "column_names": ["amount"],
            "unique": False,
            "dialect_options": {"postgresql_where": "amount > 0"},
        },
        {
            "name": "ix_pg_demo_custom_ops",
            "column_names": ["amount"],
            "unique": False,
            "dialect_options": {"postgresql_ops": {"amount": "numeric_ops"}},
        },
    ],
)
def test_collector_rejects_nonordinary_extra_nonunique_index(extra_index: dict[str, Any]) -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("amount", Numeric, nullable=False),
    )
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "amount", "type": Numeric(), "nullable": False, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[extra_index],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any(issue.subject == "pg_demo unexpected index mismatch" for issue in issues)


@pytest.mark.parametrize("name", [None, "ck_pg_demo_duplicate"])
def test_collector_preserves_check_constraint_multiplicity(name: str | None) -> None:
    metadata = MetaData()
    Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        CheckConstraint("id > 0", name=name),
    )
    inspector = _StaticInspector(
        columns=[{"name": "id", "type": Integer(), "nullable": False, "default": None}],
        primary_key=["id"],
        foreign_keys=[],
        checks=[
            {"name": name, "sqltext": "id > 0"},
            {"name": name, "sqltext": "id > 0"},
        ],
        unique_constraints=[],
        indexes=[],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any("CHECK constraint" in issue.subject for issue in issues)


def test_collector_honors_ddl_if_callable_for_constraints_and_indexes() -> None:
    def never_emit(*_args: object, **_kwargs: object) -> bool:
        return False

    metadata = MetaData()
    table = Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        CheckConstraint("id > 0", name="ck_never_emit").ddl_if(dialect="postgresql", callable_=never_emit),
    )
    Index("ix_never_emit", table.c.id).ddl_if(dialect="postgresql", callable_=never_emit)
    inspector = _StaticInspector(
        columns=[{"name": "id", "type": Integer(), "nullable": False, "default": None}],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[],
    )

    assert (
        collect_metadata_shape_issues(
            inspector,  # type: ignore[arg-type]
            metadata,
            dialect=postgresql.dialect(),
            present_tables=frozenset({"pg_demo"}),
        )
        == ()
    )


def test_collector_rejects_same_named_index_uniqueness_drift() -> None:
    metadata = MetaData()
    table = Table("pg_demo", metadata, Column("id", Integer, primary_key=True))
    Index("ix_pg_demo_id", table.c.id)
    inspector = _StaticInspector(
        columns=[{"name": "id", "type": Integer(), "nullable": False, "default": None}],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[
            {
                "name": "ix_pg_demo_id",
                "column_names": ["id"],
                "unique": True,
                "dialect_options": {},
            }
        ],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    assert any("ix_pg_demo_id" in issue.subject for issue in issues)


@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize(
    ("declared_ops", "reflected_ops", "expect_issue"),
    [
        (None, {"value": "text_pattern_ops"}, True),
        ({"value": "text_pattern_ops"}, None, True),
        ({"value": "text_pattern_ops"}, {"value": "text_pattern_ops"}, False),
    ],
)
def test_collector_compares_postgresql_index_operator_classes(
    unique: bool,
    declared_ops: dict[str, str] | None,
    reflected_ops: dict[str, str] | None,
    expect_issue: bool,
) -> None:
    metadata = MetaData()
    table = Table(
        "pg_demo",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", String(16), nullable=False),
    )
    if declared_ops is None:
        Index("ix_pg_demo_value", table.c.value, unique=unique)
    else:
        Index("ix_pg_demo_value", table.c.value, unique=unique, postgresql_ops=declared_ops)
    dialect_options = {} if reflected_ops is None else {"postgresql_ops": reflected_ops}
    inspector = _StaticInspector(
        columns=[
            {"name": "id", "type": Integer(), "nullable": False, "default": None},
            {"name": "value", "type": String(16), "nullable": False, "default": None},
        ],
        primary_key=["id"],
        foreign_keys=[],
        checks=[],
        unique_constraints=[],
        indexes=[
            {
                "name": "ix_pg_demo_value",
                "column_names": ["value"],
                "unique": unique,
                "dialect_options": dialect_options,
            }
        ],
    )

    issues = collect_metadata_shape_issues(
        inspector,  # type: ignore[arg-type]
        metadata,
        dialect=postgresql.dialect(),
        present_tables=frozenset({"pg_demo"}),
    )

    subject = f"pg_demo.ix_pg_demo_value {'unique index' if unique else 'index'} mismatch"
    if expect_issue:
        assert any(issue.subject == subject for issue in issues)
    else:
        assert issues == ()
