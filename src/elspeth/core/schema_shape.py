"""Shared, dialect-aware SQLAlchemy metadata shape comparison.

The comparator is deliberately fail-closed.  It recognizes only the narrow
reflection equivalences documented below; unfamiliar differences are returned
as :class:`SchemaShapeIssue` values for the caller to surface.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, overload

from sqlalchemy import (
    CHAR,
    TEXT,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.engine import Connection, Dialect, Engine
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.ddl import CreateConstraint
from sqlalchemy.sql.schema import Constraint, Table

if TYPE_CHECKING:
    from sqlalchemy.sql.ddl import ExecutableDDLElement


type _Ast = tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SchemaShapeIssue:
    """One reflected-schema difference from the declared metadata."""

    subject: str
    expected: object
    actual: object


@dataclass(frozen=True, slots=True)
class _IndexShape:
    unique: bool
    columns: tuple[str, ...]
    expressions: tuple[_Ast, ...]
    predicate: _Ast | None
    nulls_not_distinct: bool
    operator_classes: tuple[tuple[str, str], ...]


type _UniqueConstraintShape = tuple[frozenset[str], bool]


type _ForeignKeyShape = tuple[
    tuple[str, ...],
    str | None,
    str,
    tuple[str, ...],
    str | None,
    str | None,
    bool,
    str,
    str,
]

type _TextBuiltinCallKey = tuple[str, int, str]
type _TextConcatShape = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _TextBuiltinProof:
    pg_catalog_calls: frozenset[tuple[str, int]]
    safe_calls: frozenset[_TextBuiltinCallKey]
    literal_only_calls: frozenset[_TextBuiltinCallKey]
    int4_literal_calls: frozenset[_TextBuiltinCallKey]
    variadic_calls: frozenset[_TextBuiltinCallKey]
    text_result_calls: frozenset[_TextBuiltinCallKey]
    safe_concat_shapes: frozenset[_TextConcatShape]
    text_result_concat_shapes: frozenset[_TextConcatShape]


_EMPTY_TEXT_BUILTIN_PROOF = _TextBuiltinProof(
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
)

_ALL_TEXT_BUILTIN_CALL_KEYS: frozenset[_TextBuiltinCallKey] = frozenset(
    {
        ("length", 1, "text"),
        ("length", 1, "varchar"),
        ("btrim", 1, "text"),
        ("btrim", 1, "varchar"),
        ("btrim", 2, "text"),
        ("btrim", 2, "varchar"),
        ("btrim_unknown", 2, "text"),
        ("btrim_unknown", 2, "varchar"),
        ("chr", 1, "int4"),
    }
)
_ALLOWED_TEXT_BUILTIN_SIGNATURES = frozenset({("length", 1), ("btrim", 1), ("btrim", 2), ("chr", 1)})
_TEXT_RESULT_CALL_KEYS = frozenset(
    {
        ("chr", 1, "int4"),
        ("btrim_unknown", 2, "text"),
        ("btrim_unknown", 2, "varchar"),
    }
)
_INT4_LITERAL_CALL_KEYS = frozenset(
    {
        ("btrim_unknown", 2, "text"),
        ("btrim_unknown", 2, "varchar"),
    }
)
_ALL_TEXT_CONCAT_SHAPES: frozenset[_TextConcatShape] = frozenset(
    {
        ("text", "text"),
        ("text", "varchar"),
        ("varchar", "text"),
        ("varchar", "varchar"),
        ("text", "unknown"),
        ("unknown", "text"),
        ("varchar", "unknown"),
        ("unknown", "varchar"),
    }
)
# PostgreSQL reserves bootstrap OIDs below FirstNormalObjectId.  This is
# stronger identity evidence than namespace membership: user-created objects
# in pg_catalog still receive normal-range OIDs and remain shadow candidates.
_FIRST_NORMAL_POSTGRESQL_OID = 16_384
_TEXT_BUILTIN_IDENTITY_PROOF = text(
    """
    WITH allowed(proname, arity, allowed_oid) AS (
        VALUES
          (:length_name, 1, pg_catalog.to_regprocedure(:length_signature)::oid),
          (:btrim_name, 1, pg_catalog.to_regprocedure(:btrim_one_signature)::oid),
          (:btrim_name, 2, pg_catalog.to_regprocedure(:btrim_two_signature)::oid),
          (:chr_name, 1, pg_catalog.to_regprocedure(:chr_signature)::oid)
    ),
    sources(source_family, source_oid) AS (
        VALUES
          (:text_family, pg_catalog.to_regtype(:text_type)::oid),
          (:varchar_family, pg_catalog.to_regtype(:varchar_type)::oid),
          (:int4_family, pg_catalog.to_regtype(:int4_type)::oid)
    ),
    allowed_operator(operator_oid) AS (
        SELECT pg_catalog.to_regoperator(:text_concat_signature)::oid
    ),
    call_shapes(call_kind, proname, arity, source_family) AS (
        VALUES
          (:length_name, :length_name, 1, :text_family),
          (:length_name, :length_name, 1, :varchar_family),
          (:btrim_name, :btrim_name, 1, :text_family),
          (:btrim_name, :btrim_name, 1, :varchar_family),
          (:btrim_name, :btrim_name, 2, :text_family),
          (:btrim_name, :btrim_name, 2, :varchar_family),
          (:btrim_unknown_kind, :btrim_name, 2, :text_family),
          (:btrim_unknown_kind, :btrim_name, 2, :varchar_family),
          (:chr_name, :chr_name, 1, :int4_family)
    ),
    expected_arguments(call_kind, proname, arity, source_family, arg_index, expected_oid, is_unknown) AS (
        SELECT call_shapes.call_kind, call_shapes.proname, call_shapes.arity,
               call_shapes.source_family, 0, sources.source_oid, false
        FROM call_shapes
        JOIN sources ON sources.source_family = call_shapes.source_family
        UNION ALL
        SELECT :btrim_name, :btrim_name, 2, sources.source_family, 1,
               text_source.source_oid, false
        FROM sources
        CROSS JOIN sources AS text_source
        WHERE sources.source_family IN (:text_family, :varchar_family)
          AND text_source.source_family = :text_family
        UNION ALL
        SELECT :btrim_unknown_kind, :btrim_name, 2, sources.source_family, 1,
               NULL::oid, true
        FROM sources
        WHERE sources.source_family IN (:text_family, :varchar_family)
    ),
    candidates AS (
        SELECT
          call_shapes.call_kind AS proname,
          call_shapes.arity,
          sources.source_family,
          sources.source_oid,
          proc.oid AS candidate_oid,
          proc.pronargs::integer AS candidate_pronargs,
          proc.pronargdefaults::integer AS candidate_pronargdefaults,
          proc.provariadic AS candidate_variadic_oid,
          CASE
            WHEN proc.provariadic <> 0 AND proc.pronargs = 1 THEN proc.provariadic
            ELSE proc.proargtypes[0]
          END AS candidate_first_arg_oid,
          proc.proargtypes[1] AS candidate_second_arg_oid
        FROM pg_catalog.pg_proc AS proc
        JOIN allowed
          ON allowed.proname = proc.proname
        JOIN pg_catalog.pg_proc AS allowed_proc
          ON allowed_proc.oid = allowed.allowed_oid
        JOIN call_shapes
          ON call_shapes.proname = allowed.proname
         AND call_shapes.arity = allowed.arity
         AND call_shapes.arity >= (
             proc.pronargs
             - proc.pronargdefaults
             - CASE WHEN proc.provariadic <> 0 THEN 1 ELSE 0 END
         )
         AND (proc.provariadic <> 0 OR call_shapes.arity <= proc.pronargs)
        JOIN sources ON sources.source_family = call_shapes.source_family
        WHERE allowed.allowed_oid IS NOT NULL
          AND proc.oid <> allowed.allowed_oid
          AND proc.oid >= :first_normal_oid
          AND pg_catalog.pg_function_is_visible(proc.oid)
          AND NOT EXISTS (
              SELECT 1
              FROM expected_arguments
              WHERE expected_arguments.call_kind = call_shapes.call_kind
                AND expected_arguments.proname = call_shapes.proname
                AND expected_arguments.arity = call_shapes.arity
                AND expected_arguments.source_family = sources.source_family
                AND NOT expected_arguments.is_unknown
                AND NOT (
                    expected_arguments.expected_oid = CASE
                      WHEN proc.provariadic <> 0
                       AND expected_arguments.arg_index >= proc.pronargs - 1
                        THEN proc.provariadic
                      ELSE proc.proargtypes[expected_arguments.arg_index]
                    END
                    OR EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_cast AS casts
                        WHERE casts.castsource = expected_arguments.expected_oid
                          AND casts.casttarget = CASE
                            WHEN proc.provariadic <> 0
                             AND expected_arguments.arg_index >= proc.pronargs - 1
                              THEN proc.provariadic
                            ELSE proc.proargtypes[expected_arguments.arg_index]
                          END
                          AND casts.castcontext = 'i'
                    )
                )
          )
          AND (
              SELECT count(*)
              FROM expected_arguments
              WHERE expected_arguments.call_kind = call_shapes.call_kind
                AND expected_arguments.proname = call_shapes.proname
                AND expected_arguments.arity = call_shapes.arity
                AND expected_arguments.source_family = sources.source_family
                AND NOT expected_arguments.is_unknown
                AND expected_arguments.expected_oid = CASE
                  WHEN proc.provariadic <> 0
                   AND expected_arguments.arg_index >= proc.pronargs - 1
                    THEN proc.provariadic
                  ELSE proc.proargtypes[expected_arguments.arg_index]
                END
          ) >= (
              SELECT count(*)
              FROM expected_arguments
              WHERE expected_arguments.call_kind = call_shapes.call_kind
                AND expected_arguments.proname = call_shapes.proname
                AND expected_arguments.arity = call_shapes.arity
                AND expected_arguments.source_family = sources.source_family
                AND NOT expected_arguments.is_unknown
                AND expected_arguments.expected_oid = allowed_proc.proargtypes[expected_arguments.arg_index]
          )
    ),
    exact_call_text_results AS (
        SELECT
          call_shapes.call_kind AS proname,
          call_shapes.arity,
          sources.source_family
        FROM pg_catalog.pg_proc AS proc
        JOIN call_shapes
          ON call_shapes.proname = proc.proname
         AND call_shapes.call_kind = :chr_name
         AND call_shapes.arity >= (
             proc.pronargs
             - proc.pronargdefaults
             - CASE WHEN proc.provariadic <> 0 THEN 1 ELSE 0 END
         )
         AND (proc.provariadic <> 0 OR call_shapes.arity <= proc.pronargs)
        JOIN sources ON sources.source_family = call_shapes.source_family
        CROSS JOIN LATERAL (
            SELECT source_oid
            FROM sources
            WHERE source_family = :text_family
        ) AS text_source
        WHERE proc.prokind = 'f'
          AND pg_catalog.pg_function_is_visible(proc.oid)
          AND NOT EXISTS (
              SELECT 1
              FROM expected_arguments
              WHERE expected_arguments.call_kind = call_shapes.call_kind
                AND expected_arguments.proname = call_shapes.proname
                AND expected_arguments.arity = call_shapes.arity
                AND expected_arguments.source_family = sources.source_family
                AND (
                    expected_arguments.is_unknown
                    OR expected_arguments.expected_oid <> CASE
                      WHEN proc.provariadic <> 0
                       AND expected_arguments.arg_index >= proc.pronargs - 1
                        THEN proc.provariadic
                      ELSE proc.proargtypes[expected_arguments.arg_index]
                    END
                )
          )
        GROUP BY call_shapes.call_kind, call_shapes.arity, sources.source_family
        HAVING count(*) = 1
          AND bool_and(proc.prorettype = text_source.source_oid)
    ),
    unknown_literal_exact_first_candidates AS (
        SELECT
          call_shapes.call_kind AS proname,
          call_shapes.arity,
          sources.source_family,
          proc.pronargs::integer AS candidate_pronargs,
          proc.pronargdefaults::integer AS candidate_pronargdefaults,
          proc.provariadic AS candidate_variadic_oid,
          proc.proargtypes[1] AS candidate_second_arg_oid,
          proc.prorettype AS candidate_result_oid
        FROM pg_catalog.pg_proc AS proc
        JOIN call_shapes
          ON call_shapes.proname = proc.proname
         AND call_shapes.call_kind = :btrim_unknown_kind
         AND call_shapes.arity >= (
             proc.pronargs
             - proc.pronargdefaults
             - CASE WHEN proc.provariadic <> 0 THEN 1 ELSE 0 END
         )
         AND (proc.provariadic <> 0 OR call_shapes.arity <= proc.pronargs)
        JOIN sources ON sources.source_family = call_shapes.source_family
        WHERE proc.prokind = 'f'
          AND pg_catalog.pg_function_is_visible(proc.oid)
          AND proc.proargtypes[0] = sources.source_oid
    ),
    unknown_literal_winners AS (
        SELECT
          candidates.proname,
          candidates.arity,
          candidates.source_family,
          bool_and(
              candidates.candidate_pronargs = 2
              AND candidates.candidate_pronargdefaults = 0
              AND candidates.candidate_variadic_oid = 0
              AND candidates.candidate_second_arg_oid = (
                  SELECT source_oid FROM sources WHERE source_family = :int4_family
              )
          ) AS accepts_int4_literal,
          bool_and(
              candidates.candidate_pronargs = 2
              AND candidates.candidate_pronargdefaults = 0
              AND candidates.candidate_variadic_oid = 0
              AND candidates.candidate_second_arg_oid = (
                  SELECT source_oid FROM sources WHERE source_family = :int4_family
              )
              AND candidates.candidate_result_oid = (
                  SELECT source_oid FROM sources WHERE source_family = :text_family
              )
          ) AS returns_text
        FROM unknown_literal_exact_first_candidates AS candidates
        GROUP BY candidates.proname, candidates.arity, candidates.source_family
        HAVING count(*) = 1
    ),
    operator_shapes(left_family, right_family) AS (
        VALUES
          (:text_family, :text_family),
          (:text_family, :varchar_family),
          (:varchar_family, :text_family),
          (:varchar_family, :varchar_family),
          (:text_family, :unknown_family),
          (:unknown_family, :text_family),
          (:varchar_family, :unknown_family),
          (:unknown_family, :varchar_family)
    ),
    operator_candidates AS (
        SELECT operator_shapes.left_family, operator_shapes.right_family
        FROM pg_catalog.pg_operator AS candidate
        CROSS JOIN allowed_operator
        JOIN pg_catalog.pg_operator AS builtin
          ON builtin.oid = allowed_operator.operator_oid
        CROSS JOIN operator_shapes
        LEFT JOIN sources AS left_source
          ON left_source.source_family = operator_shapes.left_family
        LEFT JOIN sources AS right_source
          ON right_source.source_family = operator_shapes.right_family
        WHERE allowed_operator.operator_oid IS NOT NULL
          AND candidate.oprname = :text_concat_name
          AND candidate.oid <> allowed_operator.operator_oid
          AND candidate.oid >= :first_normal_oid
          AND pg_catalog.pg_operator_is_visible(candidate.oid)
          AND (
              operator_shapes.left_family = :unknown_family
              OR candidate.oprleft = left_source.source_oid
              OR EXISTS (
                  SELECT 1 FROM pg_catalog.pg_cast AS left_cast
                  WHERE left_cast.castsource = left_source.source_oid
                    AND left_cast.casttarget = candidate.oprleft
                    AND left_cast.castcontext = 'i'
              )
          )
          AND (
              operator_shapes.right_family = :unknown_family
              OR candidate.oprright = right_source.source_oid
              OR EXISTS (
                  SELECT 1 FROM pg_catalog.pg_cast AS right_cast
                  WHERE right_cast.castsource = right_source.source_oid
                    AND right_cast.casttarget = candidate.oprright
                    AND right_cast.castcontext = 'i'
              )
          )
          AND (
              CASE WHEN candidate.oprleft = left_source.source_oid THEN 1 ELSE 0 END
              + CASE WHEN candidate.oprright = right_source.source_oid THEN 1 ELSE 0 END
          ) >= (
              CASE WHEN builtin.oprleft = left_source.source_oid THEN 1 ELSE 0 END
              + CASE WHEN builtin.oprright = right_source.source_oid THEN 1 ELSE 0 END
          )
    ),
    exact_operator_text_results AS (
        SELECT operator_shapes.left_family, operator_shapes.right_family
        FROM pg_catalog.pg_operator AS candidate
        CROSS JOIN operator_shapes
        JOIN sources AS left_source
          ON left_source.source_family = operator_shapes.left_family
        JOIN sources AS right_source
          ON right_source.source_family = operator_shapes.right_family
        CROSS JOIN LATERAL (
            SELECT source_oid
            FROM sources
            WHERE source_family = :text_family
        ) AS text_source
        WHERE candidate.oprname = :text_concat_name
          AND pg_catalog.pg_operator_is_visible(candidate.oid)
          AND candidate.oprleft = left_source.source_oid
          AND candidate.oprright = right_source.source_oid
        GROUP BY operator_shapes.left_family, operator_shapes.right_family
        HAVING count(*) = 1
          AND bool_and(candidate.oprresult = text_source.source_oid)
    ),
    blocked AS (
        SELECT DISTINCT proname, arity, source_family
        FROM candidates
    ),
    literal_only AS (
        SELECT candidates.proname, candidates.arity, candidates.source_family
        FROM candidates
        WHERE candidates.proname IN (:btrim_name, :btrim_unknown_kind)
          AND candidates.arity = 2
        GROUP BY candidates.proname, candidates.arity, candidates.source_family
        HAVING count(*) = 1
          AND bool_and(candidates.candidate_first_arg_oid = candidates.source_oid)
          AND bool_and(candidates.candidate_second_arg_oid = (
              SELECT source_oid FROM sources WHERE source_family = :text_family
          ))
          AND bool_and(candidates.candidate_variadic_oid = 0)
          AND bool_and(candidates.candidate_pronargs = 2)
          AND bool_and(candidates.candidate_pronargdefaults = 0)
    ),
    variadic_only AS (
        SELECT candidates.proname, candidates.arity, candidates.source_family
        FROM candidates
        GROUP BY candidates.proname, candidates.arity, candidates.source_family
        HAVING count(*) = 1
          AND bool_and(candidates.candidate_variadic_oid <> 0)
          AND bool_and(candidates.candidate_first_arg_oid = candidates.source_oid)
    )
    SELECT 'allowed' AS row_kind, proname, arity, allowed_oid::text AS detail
    FROM allowed
    UNION ALL
    SELECT 'source' AS row_kind, source_family AS proname, 0 AS arity, source_oid::text AS detail
    FROM sources
    UNION ALL
    SELECT 'operator_allowed' AS row_kind, :text_concat_name AS proname, 2 AS arity,
           operator_oid::text AS detail
    FROM allowed_operator
    UNION ALL
    SELECT DISTINCT 'operator_blocked' AS row_kind, :text_concat_name AS proname, 2 AS arity,
           left_family || ',' || right_family AS detail
    FROM operator_candidates
    UNION ALL
    SELECT 'operator_text_result' AS row_kind, :text_concat_name AS proname, 2 AS arity,
           left_family || ',' || right_family AS detail
    FROM exact_operator_text_results
    UNION ALL
    SELECT 'blocked' AS row_kind, proname, arity, source_family AS detail
    FROM blocked
    UNION ALL
    SELECT 'literal_only' AS row_kind, proname, arity, source_family AS detail
    FROM literal_only
    UNION ALL
    SELECT 'variadic' AS row_kind, proname, arity, source_family AS detail
    FROM variadic_only
    UNION ALL
    SELECT 'text_result' AS row_kind, proname, arity, source_family AS detail
    FROM exact_call_text_results
    UNION ALL
    SELECT 'int4_literal' AS row_kind, proname, arity, source_family AS detail
    FROM unknown_literal_winners
    WHERE accepts_int4_literal
    UNION ALL
    SELECT 'text_result' AS row_kind, proname, arity, source_family AS detail
    FROM unknown_literal_winners
    WHERE returns_text
    """
)


def collect_metadata_shape_issues(
    inspector: Inspector,
    metadata: MetaData,
    *,
    dialect: Dialect,
    present_tables: AbstractSet[str],
    allowed_missing_index_names: AbstractSet[str] = frozenset(),
) -> tuple[SchemaShapeIssue, ...]:
    """Collect every reflected-schema difference for ``present_tables``.

    The caller owns the table-set policy.  This function compares the complete
    declared shape of each named table while tolerating only additional
    ordinary non-unique indexes and explicitly allowed missing indexes.
    """

    issues: list[SchemaShapeIssue] = []
    text_builtin_proof = _proven_pg_catalog_text_builtin_calls(inspector, dialect)
    for table_name in sorted(present_tables):
        if table_name not in metadata.tables:
            issues.append(SchemaShapeIssue(f"{table_name} metadata table mismatch", "declared table", "missing"))
            continue
        table = metadata.tables[table_name]
        _collect_column_issues(issues, inspector, table, dialect)
        _collect_foreign_key_issues(issues, inspector, table, dialect)
        _collect_check_issues(issues, inspector, table, dialect, text_builtin_proof)
        _collect_unique_constraint_issues(issues, inspector, table, dialect)
        _collect_index_issues(
            issues,
            inspector,
            table,
            dialect,
            text_builtin_proof=text_builtin_proof,
            allowed_missing_index_names=allowed_missing_index_names,
        )
    return tuple(issues)


def _proven_pg_catalog_text_builtin_calls(
    inspector: Inspector,
    dialect: Dialect,
) -> _TextBuiltinProof:
    if dialect.name != "postgresql":
        return _EMPTY_TEXT_BUILTIN_PROOF
    bind = getattr(inspector, "bind", None)
    try:
        if isinstance(bind, Connection):
            proof_rows = _text_builtin_identity_rows_on_connection(bind)
        elif isinstance(bind, Engine):
            with bind.connect() as connection:
                proof_rows = _text_builtin_identity_rows_on_connection(connection)
        else:
            return _EMPTY_TEXT_BUILTIN_PROOF
    except SQLAlchemyError:
        return _EMPTY_TEXT_BUILTIN_PROOF
    if proof_rows is None:
        return _EMPTY_TEXT_BUILTIN_PROOF

    allowed: dict[tuple[str, int], int] = {}
    sources: dict[str, int] = {}
    blocked: set[_TextBuiltinCallKey] = set()
    literal_only: set[_TextBuiltinCallKey] = set()
    int4_literal: set[_TextBuiltinCallKey] = set()
    variadic: set[_TextBuiltinCallKey] = set()
    text_results: set[_TextBuiltinCallKey] = set()
    operator_allowed_oid: int | None = None
    blocked_concat_shapes: set[_TextConcatShape] = set()
    text_result_concat_shapes: set[_TextConcatShape] = set()
    try:
        for row_kind, raw_name, raw_arity, raw_detail in proof_rows:
            name = str(raw_name)
            arity = int(raw_arity)
            if row_kind == "allowed":
                key = (name, arity)
                oid = int(raw_detail)
                if key in allowed or not 0 < oid < _FIRST_NORMAL_POSTGRESQL_OID:
                    return _EMPTY_TEXT_BUILTIN_PROOF
                allowed[key] = oid
            elif row_kind == "source":
                oid = int(raw_detail)
                if name in sources or arity != 0 or not 0 < oid < _FIRST_NORMAL_POSTGRESQL_OID:
                    return _EMPTY_TEXT_BUILTIN_PROOF
                sources[name] = oid
            elif row_kind == "blocked":
                blocked.add((name, arity, str(raw_detail)))
            elif row_kind == "literal_only":
                literal_only.add((name, arity, str(raw_detail)))
            elif row_kind == "int4_literal":
                int4_literal.add((name, arity, str(raw_detail)))
            elif row_kind == "variadic":
                variadic.add((name, arity, str(raw_detail)))
            elif row_kind == "text_result":
                text_results.add((name, arity, str(raw_detail)))
            elif row_kind == "operator_allowed":
                oid = int(raw_detail)
                if name != "||" or arity != 2 or operator_allowed_oid is not None or not 0 < oid < _FIRST_NORMAL_POSTGRESQL_OID:
                    return _EMPTY_TEXT_BUILTIN_PROOF
                operator_allowed_oid = oid
            elif row_kind == "operator_blocked":
                left_family, separator, right_family = str(raw_detail).partition(",")
                shape = (left_family, right_family)
                if name != "||" or arity != 2 or not separator or shape not in _ALL_TEXT_CONCAT_SHAPES:
                    return _EMPTY_TEXT_BUILTIN_PROOF
                blocked_concat_shapes.add(shape)
            elif row_kind == "operator_text_result":
                left_family, separator, right_family = str(raw_detail).partition(",")
                shape = (left_family, right_family)
                if name != "||" or arity != 2 or not separator or shape not in _ALL_TEXT_CONCAT_SHAPES:
                    return _EMPTY_TEXT_BUILTIN_PROOF
                text_result_concat_shapes.add(shape)
            else:
                return _EMPTY_TEXT_BUILTIN_PROOF
    except (TypeError, ValueError):
        return _EMPTY_TEXT_BUILTIN_PROOF
    literal_only_keys = frozenset(
        {
            ("btrim", 2, "text"),
            ("btrim", 2, "varchar"),
            ("btrim_unknown", 2, "text"),
            ("btrim_unknown", 2, "varchar"),
        }
    )
    if (
        set(allowed) != _ALLOWED_TEXT_BUILTIN_SIGNATURES
        or set(sources) != {"text", "varchar", "int4"}
        or not blocked <= _ALL_TEXT_BUILTIN_CALL_KEYS
        or not literal_only <= literal_only_keys
        or not literal_only <= blocked
        or not int4_literal <= _INT4_LITERAL_CALL_KEYS
        or not int4_literal <= blocked
        or not variadic <= _ALL_TEXT_BUILTIN_CALL_KEYS
        or not variadic <= blocked
        or not text_results <= _TEXT_RESULT_CALL_KEYS
        or not (text_results - {("chr", 1, "int4")}) <= int4_literal
        or operator_allowed_oid is None
    ):
        return _EMPTY_TEXT_BUILTIN_PROOF
    return _TextBuiltinProof(
        pg_catalog_calls=frozenset(allowed),
        safe_calls=_ALL_TEXT_BUILTIN_CALL_KEYS - blocked,
        literal_only_calls=frozenset(literal_only),
        int4_literal_calls=frozenset(int4_literal),
        variadic_calls=frozenset(variadic),
        text_result_calls=frozenset(text_results),
        safe_concat_shapes=_ALL_TEXT_CONCAT_SHAPES - blocked_concat_shapes,
        text_result_concat_shapes=frozenset(text_result_concat_shapes),
    )


def _text_builtin_identity_rows_on_connection(connection: Connection) -> list[Any] | None:
    was_idle = not connection.in_transaction()
    nested_transaction = None
    try:
        if not was_idle:
            nested_transaction = connection.begin_nested()
        rows = connection.execute(
            _TEXT_BUILTIN_IDENTITY_PROOF,
            {
                "length_name": "length",
                "length_signature": "pg_catalog.length(text)",
                "btrim_name": "btrim",
                "btrim_one_signature": "pg_catalog.btrim(text)",
                "btrim_two_signature": "pg_catalog.btrim(text,text)",
                "btrim_unknown_kind": "btrim_unknown",
                "chr_name": "chr",
                "chr_signature": "pg_catalog.chr(integer)",
                "text_family": "text",
                "text_type": "pg_catalog.text",
                "varchar_family": "varchar",
                "varchar_type": "pg_catalog.varchar",
                "int4_family": "int4",
                "int4_type": "pg_catalog.int4",
                "unknown_family": "unknown",
                "text_concat_name": "||",
                "text_concat_signature": "pg_catalog.||(text,text)",
                "first_normal_oid": _FIRST_NORMAL_POSTGRESQL_OID,
            },
        ).all()
        if nested_transaction is not None:
            nested_transaction.commit()
        return list(rows)
    except SQLAlchemyError:
        if nested_transaction is not None and nested_transaction.is_active:
            nested_transaction.rollback()
        return None
    finally:
        if was_idle and connection.in_transaction():
            connection.rollback()


def _collect_column_issues(
    issues: list[SchemaShapeIssue],
    inspector: Inspector,
    table: Table,
    dialect: Dialect,
) -> None:
    table_name = table.name
    inspected_columns = inspector.get_columns(table_name)
    expected_names = tuple(column.name for column in table.columns)
    actual_names = tuple(str(column["name"]) for column in inspected_columns)
    if expected_names != actual_names:
        issues.append(SchemaShapeIssue(f"{table_name} column mismatch", expected_names, actual_names))

    pk = inspector.get_pk_constraint(table_name)
    actual_pk = frozenset(str(name) for name in cast("Sequence[object]", pk.get("constrained_columns") or ()))
    columns_by_name = {str(column["name"]): column for column in inspected_columns}
    for column in table.columns:
        if column.name not in columns_by_name:
            continue
        actual = columns_by_name[column.name]
        expected_primary_key = bool(column.primary_key)
        actual_primary_key = column.name in actual_pk
        if expected_primary_key != actual_primary_key:
            issues.append(
                SchemaShapeIssue(
                    f"{table_name}.{column.name} primary-key mismatch",
                    expected_primary_key,
                    actual_primary_key,
                )
            )

        expected_nullable = bool(column.nullable)
        actual_nullable = bool(actual["nullable"])
        if expected_nullable != actual_nullable:
            issues.append(
                SchemaShapeIssue(
                    f"{table_name}.{column.name} nullable mismatch",
                    expected_nullable,
                    actual_nullable,
                )
            )

        expected_type = _normalize_type_sql(str(column.type.compile(dialect=dialect)), dialect)
        actual_type_object = actual["type"]
        actual_type = _normalize_type_sql(str(actual_type_object.compile(dialect=dialect)), dialect)
        if expected_type != actual_type:
            issues.append(
                SchemaShapeIssue(
                    f"{table_name}.{column.name} type mismatch",
                    expected_type,
                    actual_type,
                )
            )

        expected_default = dialect.ddl_compiler(
            dialect,
            cast("ExecutableDDLElement", None),
        ).get_column_default_string(column)
        raw_actual_default = actual.get("default")
        actual_default = None if raw_actual_default is None else str(raw_actual_default)
        if _implicit_postgres_sequence_default(column, actual_default, dialect):
            actual_default = None
        expected_default_ast = _default_ast(expected_default, dialect, column, expected=True)
        actual_default_ast = _default_ast(actual_default, dialect, column, expected=False)
        if not _default_asts_equivalent(expected_default_ast, actual_default_ast, column, dialect):
            issues.append(
                SchemaShapeIssue(
                    f"{table_name}.{column.name} server-default mismatch",
                    None if expected_default_ast is None else repr(expected_default_ast),
                    None if actual_default_ast is None else repr(actual_default_ast),
                )
            )


def _implicit_postgres_sequence_default(column: Column[Any], actual_default: str | None, dialect: Dialect) -> bool:
    if dialect.name != "postgresql" or actual_default is None:
        return False
    if not column.primary_key or not column.autoincrement or column.server_default is not None:
        return False
    type_sql = _normalize_type_sql(str(column.type.compile(dialect=dialect)), dialect)
    if type_sql not in {"INTEGER", "BIGINT", "SMALLINT"}:
        return False
    match = re.fullmatch(
        r"nextval\(\s*'((?:''|[^'])+)'\s*::\s*regclass\s*\)",
        actual_default.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return False
    sequence_reference = match.group(1).replace("''", "'")
    reference_parts = _parse_postgres_identifier(sequence_reference)
    if reference_parts is None or len(reference_parts) not in {1, 2}:
        return False
    sequence_name = reference_parts[-1]
    expected_schema = column.table.schema or dialect.default_schema_name
    if len(reference_parts) == 2:
        sequence_schema = reference_parts[0]
        if expected_schema is None or sequence_schema != expected_schema:
            return False
    elif column.table.schema is not None and column.table.schema != dialect.default_schema_name:
        return False
    expected_name = f"{column.table.name}_{column.name}_seq"
    max_identifier_length = dialect.max_identifier_length
    if len(expected_name) > max_identifier_length:
        expected_name = expected_name[:max_identifier_length]
    return sequence_name == expected_name


def _parse_postgres_identifier(value: str) -> tuple[str, ...] | None:
    """Parse a regclass identifier without treating dots inside quotes as separators."""
    parts: list[str] = []
    position = 0
    while position < len(value):
        if value[position] == '"':
            position += 1
            component: list[str] = []
            while position < len(value):
                if value[position] != '"':
                    component.append(value[position])
                    position += 1
                    continue
                if position + 1 < len(value) and value[position + 1] == '"':
                    component.append('"')
                    position += 2
                    continue
                position += 1
                break
            else:
                return None
            if not component:
                return None
            parts.append("".join(component))
        else:
            match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", value[position:])
            if match is None:
                return None
            parts.append(match.group(0).lower())
            position += len(match.group(0))
        if position == len(value):
            return tuple(parts)
        if value[position] != ".":
            return None
        position += 1
        if position == len(value):
            return None
    return None


def _collect_foreign_key_issues(
    issues: list[SchemaShapeIssue],
    inspector: Inspector,
    table: Table,
    dialect: Dialect,
) -> None:
    expected = Counter(
        _expected_foreign_key_shape(constraint, dialect)
        for constraint in table.foreign_key_constraints
        if _ddl_object_applies_to_dialect(constraint, dialect)
    )
    actual = Counter(_actual_foreign_key_shape(fk, dialect) for fk in inspector.get_foreign_keys(table.name))
    if expected != actual:
        issues.append(
            SchemaShapeIssue(
                f"{table.name} foreign-key mismatch",
                _sorted_counter_items(expected),
                _sorted_counter_items(actual),
            )
        )


def _expected_foreign_key_shape(constraint: ForeignKeyConstraint, dialect: Dialect) -> _ForeignKeyShape:
    elements = tuple(constraint.elements)
    if not elements:
        return ((), None, "<missing>", (), None, None, False, "IMMEDIATE", "SIMPLE")
    referred_table = elements[0].column.table
    return (
        tuple(element.parent.name for element in elements),
        _normalize_referred_schema(referred_table.schema, dialect),
        referred_table.name,
        tuple(element.column.name for element in elements),
        _normalize_fk_action(constraint.onupdate),
        _normalize_fk_action(constraint.ondelete),
        bool(constraint.deferrable),
        _normalize_initially(constraint.initially),
        _normalize_match(constraint.match),
    )


def _actual_foreign_key_shape(fk: Mapping[str, Any], dialect: Dialect) -> _ForeignKeyShape:
    options = cast("Mapping[str, Any]", fk.get("options") or {})
    return (
        tuple(str(column) for column in cast("Sequence[object]", fk.get("constrained_columns") or ())),
        _normalize_referred_schema(_optional_string(fk.get("referred_schema")), dialect),
        str(fk["referred_table"]),
        tuple(str(column) for column in cast("Sequence[object]", fk.get("referred_columns") or ())),
        _normalize_fk_action(_optional_string(options.get("onupdate"))),
        _normalize_fk_action(_optional_string(options.get("ondelete"))),
        bool(options.get("deferrable", False)),
        _normalize_initially(_optional_string(options.get("initially"))),
        _normalize_match(_optional_string(options.get("match"))),
    )


def _collect_check_issues(
    issues: list[SchemaShapeIssue],
    inspector: Inspector,
    table: Table,
    dialect: Dialect,
    text_builtin_proof: _TextBuiltinProof,
) -> None:
    expected: Counter[tuple[str | None, _Ast]] = Counter()
    for constraint in table.constraints:
        if type(constraint) is not CheckConstraint or not _ddl_object_applies_to_dialect(constraint, dialect):
            continue
        sql = str(constraint.sqltext.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        expected[(_optional_string(constraint.name), _expected_expression_ast(sql, dialect, table))] += 1

    actual: Counter[tuple[str | None, _Ast]] = Counter()
    for reflected in inspector.get_check_constraints(table.name):
        name = _optional_string(reflected.get("name"))
        actual[(name, _expression_ast(str(reflected["sqltext"]), dialect, table))] += 1

    expected_names = Counter(name for (name, _sql), count in expected.items() for _ in range(count))
    actual_names = Counter(name for (name, _sql), count in actual.items() for _ in range(count))
    if expected_names != actual_names:
        issues.append(
            SchemaShapeIssue(
                f"{table.name} CHECK constraint mismatch",
                _sorted_check_counter(expected),
                _sorted_check_counter(actual),
            )
        )
    for name in sorted(set(expected_names) & set(actual_names), key=lambda item: item or ""):
        expected_sql = Counter({sql: count for (entry_name, sql), count in expected.items() if entry_name == name})
        actual_sql = Counter({sql: count for (entry_name, sql), count in actual.items() if entry_name == name})
        context = _expression_context(table, dialect, text_builtin_proof)
        if not _expression_counters_equivalent(expected_sql, actual_sql, context):
            display_name = name if name is not None else "<unnamed>"
            issues.append(
                SchemaShapeIssue(
                    f"{table.name}.{display_name} CHECK constraint SQL mismatch",
                    sorted((repr(sql), count) for sql, count in expected_sql.items()),
                    sorted((repr(sql), count) for sql, count in actual_sql.items()),
                )
            )


def _collect_unique_constraint_issues(
    issues: list[SchemaShapeIssue],
    inspector: Inspector,
    table: Table,
    dialect: Dialect,
) -> None:
    # Constraint names are not semantic on PostgreSQL: unnamed declarations
    # receive generated names at reflection time.  Column membership and NULL
    # equality semantics form the integrity-bearing signature; column order is
    # irrelevant for uniqueness.
    expected: Counter[_UniqueConstraintShape] = Counter(
        (
            frozenset(column.name for column in constraint.columns),
            _expected_postgresql_nulls_not_distinct(constraint, dialect),
        )
        for constraint in table.constraints
        if type(constraint) is UniqueConstraint and _ddl_object_applies_to_dialect(constraint, dialect)
    )
    actual: Counter[_UniqueConstraintShape] = Counter(
        (
            frozenset(str(column) for column in cast("Sequence[object]", reflected.get("column_names") or ())),
            _actual_postgresql_nulls_not_distinct(reflected, dialect),
        )
        for reflected in inspector.get_unique_constraints(table.name)
    )
    if expected != actual:
        issues.append(
            SchemaShapeIssue(
                f"{table.name} UNIQUE constraint mismatch",
                _sorted_unique_constraint_signatures(expected),
                _sorted_unique_constraint_signatures(actual),
            )
        )


def _collect_index_issues(
    issues: list[SchemaShapeIssue],
    inspector: Inspector,
    table: Table,
    dialect: Dialect,
    *,
    text_builtin_proof: _TextBuiltinProof,
    allowed_missing_index_names: AbstractSet[str],
) -> None:
    expected_unique: dict[str, _IndexShape] = {}
    expected_ordinary: dict[str, _IndexShape] = {}
    for index in sorted(table.indexes, key=lambda item: item.name or ""):
        if index.name is None or not _ddl_object_applies_to_dialect(index, dialect):
            continue
        shape = _expected_index_shape(index, dialect)
        target = expected_unique if shape.unique else expected_ordinary
        target[str(index.name)] = shape

    actual_unique: dict[str, _IndexShape] = {}
    actual_ordinary: dict[str, _IndexShape] = {}
    for reflected in inspector.get_indexes(table.name):
        if reflected.get("duplicates_constraint"):
            continue
        name = reflected.get("name")
        if name is None:
            # An unnamed reflected index cannot satisfy any declared index;
            # unique ones remain integrity-bearing extras below.
            name = "<unnamed>"
        shape = _actual_index_shape(reflected, dialect, table)
        target = actual_unique if shape.unique else actual_ordinary
        target[str(name)] = shape

    unexpected_unique = sorted(set(actual_unique) - set(expected_unique))
    if unexpected_unique:
        issues.append(
            SchemaShapeIssue(
                f"{table.name} unique index mismatch",
                sorted(expected_unique),
                sorted(actual_unique),
            )
        )

    unexpected_nonordinary = sorted(
        name for name in set(actual_ordinary) - set(expected_ordinary) if not _is_ordinary_extra_index(actual_ordinary[name])
    )
    if unexpected_nonordinary:
        issues.append(
            SchemaShapeIssue(
                f"{table.name} unexpected index mismatch",
                "additional simple-column non-unique indexes only",
                unexpected_nonordinary,
            )
        )

    _compare_expected_indexes(
        issues,
        table,
        "unique index",
        expected_unique,
        actual_unique,
        allowed_missing_index_names,
        dialect,
        text_builtin_proof,
    )
    _compare_expected_indexes(
        issues,
        table,
        "index",
        expected_ordinary,
        actual_ordinary,
        allowed_missing_index_names,
        dialect,
        text_builtin_proof,
    )
    # Additional ordinary non-unique indexes are deliberately compatible.


def _compare_expected_indexes(
    issues: list[SchemaShapeIssue],
    table: Table,
    label: str,
    expected: Mapping[str, _IndexShape],
    actual: Mapping[str, _IndexShape],
    allowed_missing_index_names: AbstractSet[str],
    dialect: Dialect,
    text_builtin_proof: _TextBuiltinProof,
) -> None:
    table_name = table.name
    for name in sorted(expected):
        if name not in actual:
            if name not in allowed_missing_index_names:
                issues.append(SchemaShapeIssue(f"{table_name}.{name} {label} mismatch", expected[name], None))
            continue
        expected_shape = expected[name]
        actual_shape = actual[name]
        if not _index_shapes_equivalent(expected_shape, actual_shape, table, dialect, text_builtin_proof):
            if expected_shape.columns != actual_shape.columns:
                subject = f"{table_name}.{name} {label} column mismatch"
            elif not _optional_expression_asts_equivalent(
                expected_shape.predicate,
                actual_shape.predicate,
                table,
                dialect,
                text_builtin_proof,
            ):
                subject = f"{table_name}.{name} {label} predicate mismatch"
            else:
                subject = f"{table_name}.{name} {label} mismatch"
            issues.append(SchemaShapeIssue(subject, expected_shape, actual_shape))


def _index_shapes_equivalent(
    expected: _IndexShape,
    actual: _IndexShape,
    table: Table,
    dialect: Dialect,
    text_builtin_proof: _TextBuiltinProof,
) -> bool:
    if (
        expected.unique != actual.unique
        or expected.columns != actual.columns
        or expected.nulls_not_distinct != actual.nulls_not_distinct
        or expected.operator_classes != actual.operator_classes
        or len(expected.expressions) != len(actual.expressions)
    ):
        return False
    context = _expression_context(table, dialect, text_builtin_proof)
    if not all(
        _expression_asts_equivalent(expected_expression, actual_expression, context)
        for expected_expression, actual_expression in zip(expected.expressions, actual.expressions, strict=True)
    ):
        return False
    return _optional_expression_asts_equivalent(
        expected.predicate,
        actual.predicate,
        table,
        dialect,
        text_builtin_proof,
    )


def _optional_expression_asts_equivalent(
    expected: _Ast | None,
    actual: _Ast | None,
    table: Table,
    dialect: Dialect,
    text_builtin_proof: _TextBuiltinProof,
) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return _expression_asts_equivalent(expected, actual, _expression_context(table, dialect, text_builtin_proof))


def _expected_index_shape(index: Index, dialect: Dialect) -> _IndexShape:
    table = index.table
    if table is None:
        raise RuntimeError("Schema-shape comparison requires attached indexes")
    columns: list[str] = []
    expressions: list[_Ast] = []
    for expression in index.expressions:
        name = getattr(expression, "name", None)
        if name is not None and str(name) in table.c:
            columns.append(str(name))
            expressions.append(("column", str(name)))
        else:
            columns.append("<expression>")
            compiled = (
                expression
                if isinstance(expression, str)
                else str(expression.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
            )
            expressions.append(_expected_expression_ast(compiled, dialect, table))
    predicate = _index_expected_predicate(index, dialect)
    nulls_not_distinct = _expected_postgresql_nulls_not_distinct(index, dialect)
    operator_classes = _expected_postgresql_operator_classes(index, dialect)
    return _IndexShape(bool(index.unique), tuple(columns), tuple(expressions), predicate, nulls_not_distinct, operator_classes)


def _actual_index_shape(index: Mapping[str, Any], dialect: Dialect, table: Table) -> _IndexShape:
    columns = tuple(str(column) if column is not None else "<expression>" for column in index.get("column_names") or ())
    raw_expressions = cast("Sequence[object]", index.get("expressions") or ())
    expressions = tuple(
        ("column", str(column))
        if column is not None
        else _expression_ast(str(raw_expressions[position]), dialect, table)
        if position < len(raw_expressions)
        else ("unknown-expression",)
        for position, column in enumerate(index.get("column_names") or ())
    )
    raw_options = cast("Mapping[str, Any]", index.get("dialect_options") or {})
    key = f"{dialect.name}_where"
    raw_predicate = raw_options.get(key)
    predicate = None if raw_predicate is None else _expression_ast(str(raw_predicate), dialect, table)
    nulls_not_distinct = _actual_postgresql_nulls_not_distinct(index, dialect)
    operator_classes = _actual_postgresql_operator_classes(index, dialect)
    return _IndexShape(bool(index.get("unique", False)), columns, expressions, predicate, nulls_not_distinct, operator_classes)


def _is_ordinary_extra_index(shape: _IndexShape) -> bool:
    return (
        not shape.unique
        and shape.predicate is None
        and not shape.nulls_not_distinct
        and not shape.operator_classes
        and bool(shape.columns)
        and all(column != "<expression>" for column in shape.columns)
        and shape.expressions == tuple(("column", column) for column in shape.columns)
    )


def _index_expected_predicate(index: Index, dialect: Dialect) -> _Ast | None:
    table = index.table
    if table is None:
        raise RuntimeError("Schema-shape comparison requires attached indexes")
    if dialect.name not in index.dialect_options:
        return None
    options = index.dialect_options[dialect.name]
    if "where" not in options or options["where"] is None:
        return None
    where = options["where"]
    compiled = where if isinstance(where, str) else str(where.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    return _expected_expression_ast(compiled, dialect, table)


def _expected_postgresql_nulls_not_distinct(obj: Constraint | Index, dialect: Dialect) -> bool:
    if dialect.name != "postgresql":
        return False
    return obj.dialect_options["postgresql"].get("nulls_not_distinct") is True


def _actual_postgresql_nulls_not_distinct(reflected: Mapping[str, Any], dialect: Dialect) -> bool:
    if dialect.name != "postgresql":
        return False
    options = cast("Mapping[str, Any]", reflected.get("dialect_options") or {})
    return options.get("postgresql_nulls_not_distinct") is True


def _expected_postgresql_operator_classes(index: Index, dialect: Dialect) -> tuple[tuple[str, str], ...]:
    if dialect.name != "postgresql":
        return ()
    return _normalize_postgresql_operator_classes(index.dialect_options["postgresql"].get("ops"))


def _actual_postgresql_operator_classes(
    reflected: Mapping[str, Any],
    dialect: Dialect,
) -> tuple[tuple[str, str], ...]:
    if dialect.name != "postgresql":
        return ()
    options = cast("Mapping[str, Any]", reflected.get("dialect_options") or {})
    return _normalize_postgresql_operator_classes(options.get("postgresql_ops"))


def _normalize_postgresql_operator_classes(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        return (("<invalid>", repr(value)),)
    return tuple(sorted((str(key), str(operator_class).strip()) for key, operator_class in value.items()))


def _ddl_object_applies_to_dialect(obj: Constraint | Index, dialect: Dialect) -> bool:
    ddl_if = getattr(obj, "_ddl_if", None)
    if ddl_if is None:
        return True
    target = ddl_if.dialect
    if isinstance(target, str) and target != dialect.name:
        return False
    if target is not None and not isinstance(target, str) and dialect.name not in target:
        return False
    if ddl_if.callable_ is None:
        return True
    compiler = dialect.ddl_compiler(dialect, cast("ExecutableDDLElement", None))
    ddl = CreateIndex(obj) if isinstance(obj, Index) else CreateConstraint(obj)
    return bool(
        ddl_if.callable_(
            ddl,
            obj,
            None,
            state=ddl_if.state,
            dialect=dialect,
            compiler=compiler,
        )
    )


def _normalize_type_sql(sql: str, dialect: Dialect) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip()).upper()
    if dialect.name == "postgresql" and normalized == "FLOAT":
        return "DOUBLE PRECISION"
    return normalized


@dataclass(frozen=True, slots=True)
class _SqlToken:
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class _ExpressionContext:
    table: Table
    dialect: Dialect
    text_columns: frozenset[str]
    varchar_columns: frozenset[str]
    char_columns: frozenset[str]
    text_builtin_proof: _TextBuiltinProof


_SQL_OPERATORS = (
    "!~~",
    "!~*",
    "::",
    "<=",
    ">=",
    "<>",
    "!=",
    "||",
    "~~",
    "~*",
    "!~",
    "~",
    "=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "%",
)


def _default_ast(
    sql: str | None,
    dialect: Dialect,
    column: Column[Any],
    *,
    expected: bool,
) -> _Ast | None:
    if sql is None:
        return None
    context = _expression_context(column.table, dialect)
    ast = _parse_sql_expression(_tokenize_sql(sql), context)
    canonical = _canonicalize_ast(ast, context)
    if expected and dialect.name == "postgresql" and dialect.paramstyle in {"format", "pyformat"}:
        canonical = _undouble_percent_literals(canonical)
    type_sql = _normalize_type_sql(str(column.type.compile(dialect=dialect)), dialect)
    if type_sql in {"INTEGER", "BIGINT", "SMALLINT"} and canonical[0] == "literal":
        integer = re.fullmatch(r"'([+-]?\d+)'", cast("str", canonical[1]))
        if integer is not None:
            canonical = ("number", integer.group(1))
    return canonical


def _expression_ast(sql: str, dialect: Dialect, table: Table) -> _Ast:
    context = _expression_context(table, dialect)
    return _canonicalize_ast(_parse_sql_expression(_tokenize_sql(sql), context), context)


def _expected_expression_ast(sql: str, dialect: Dialect, table: Table) -> _Ast:
    context = _expression_context(table, dialect)
    ast = _canonicalize_ast(_parse_sql_expression(_tokenize_sql(sql), context), context)
    if dialect.name == "postgresql" and dialect.paramstyle in {"format", "pyformat"}:
        ast = _undouble_percent_literals(ast)
    return ast


def _expression_context(
    table: Table,
    dialect: Dialect,
    text_builtin_proof: _TextBuiltinProof = _EMPTY_TEXT_BUILTIN_PROOF,
) -> _ExpressionContext:
    text_families = {
        column.name: family for column in table.columns if (family := _plain_postgresql_text_family(column, dialect)) is not None
    }
    return _ExpressionContext(
        table=table,
        dialect=dialect,
        text_columns=frozenset(text_families),
        varchar_columns=frozenset(name for name, family in text_families.items() if family == "varchar"),
        char_columns=frozenset(column.name for column in table.columns if type(column.type) is CHAR),
        text_builtin_proof=text_builtin_proof,
    )


def _plain_postgresql_text_family(column: Column[Any], dialect: Dialect) -> str | None:
    if dialect.name != "postgresql" or type(column.type) not in {String, Text, VARCHAR, TEXT}:
        return None
    compiled = _normalize_type_sql(str(column.type.compile(dialect=dialect)), dialect)
    if compiled == "TEXT":
        return "text"
    if re.fullmatch(r"(?:VARCHAR|CHARACTER VARYING)(?:\(\d+\))?", compiled) is not None:
        return "varchar"
    return None


def _is_plain_postgresql_text_column(column: Column[Any], dialect: Dialect) -> bool:
    return _plain_postgresql_text_family(column, dialect) is not None


def _tokenize_sql(sql: str) -> tuple[_SqlToken, ...]:
    tokens: list[_SqlToken] = []
    index = 0
    while index < len(sql):
        character = sql[index]
        if character.isspace():
            index += 1
            continue
        if character == "$":
            delimiter_match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[index:])
            if delimiter_match is not None:
                delimiter = delimiter_match.group(0)
                end = sql.find(delimiter, index + len(delimiter))
                if end != -1:
                    end += len(delimiter)
                    tokens.append(_SqlToken("literal", sql[index:end]))
                    index = end
                    continue
        escaped_string = character in {"e", "E"} and index + 1 < len(sql) and sql[index + 1] == "'"
        if character == "'" or escaped_string:
            start = index
            quote_index = index + 1 if escaped_string else index
            index = quote_index + 1
            while index < len(sql):
                if escaped_string and sql[index] == "\\" and index + 1 < len(sql):
                    index += 2
                    continue
                if sql[index] == "'":
                    if index + 1 < len(sql) and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            tokens.append(_SqlToken("literal", sql[start:index]))
            continue
        if character == '"':
            start = index
            index += 1
            while index < len(sql):
                if sql[index] == '"':
                    if index + 1 < len(sql) and sql[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            tokens.append(_SqlToken("quoted_identifier", sql[start:index]))
            continue
        if character.isalpha() or character == "_":
            end = index + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] in {"_", "$"}):
                end += 1
            tokens.append(_SqlToken("identifier", sql[index:end].lower()))
            index = end
            continue
        if character.isdigit():
            end = index + 1
            while end < len(sql) and (sql[end].isdigit() or sql[end] == "."):
                end += 1
            tokens.append(_SqlToken("number", sql[index:end]))
            index = end
            continue
        operator = next((candidate for candidate in _SQL_OPERATORS if sql.startswith(candidate, index)), None)
        if operator is not None:
            tokens.append(_SqlToken("operator", operator))
            index += len(operator)
            continue
        punctuation_kind = {"(": "lparen", ")": "rparen", "[": "lbracket", "]": "rbracket", ",": "comma", ".": "dot"}.get(character)
        if punctuation_kind is not None:
            tokens.append(_SqlToken(punctuation_kind, character))
            index += 1
            continue
        tokens.append(_SqlToken("unknown", character))
        index += 1
    return tuple(tokens)


class _SqlExpressionParser:
    def __init__(self, tokens: tuple[_SqlToken, ...], context: _ExpressionContext) -> None:
        self.tokens = tokens
        self.context = context
        self.position = 0

    def parse(self) -> _Ast:
        expression = self._parse_expression(0)
        if self.position != len(self.tokens):
            return ("raw", tuple((token.kind, token.value) for token in self.tokens))
        return expression

    def _parse_expression(self, minimum_precedence: int) -> _Ast:
        left = self._parse_prefix()
        while True:
            operator = self._peek_binary_operator()
            if operator is None:
                break
            name, precedence, token_count = operator
            if precedence < minimum_precedence:
                break
            self.position += token_count
            if name in {"in", "not in"}:
                values = self._parse_parenthesized_list()
                left = ("in", name == "not in", left, values)
                continue
            if name in {"is", "is not"}:
                right = self._parse_prefix()
                left = ("is", name == "is not", left, right)
                continue
            right = self._parse_expression(precedence + 1)
            left = ("binary", name, left, right)
        return left

    def _parse_prefix(self) -> _Ast:
        token = self._peek()
        if token is None:
            return ("missing",)
        if token.kind == "identifier" and token.value == "not":
            self.position += 1
            return ("unary", "not", self._parse_expression(25))
        if token.kind == "operator" and token.value in {"+", "-"}:
            self.position += 1
            return ("unary", token.value, self._parse_expression(70))
        expression = self._parse_primary()
        while self._match("operator", "::"):
            expression = ("cast", expression, self._parse_cast_type())
        return expression

    def _parse_primary(self) -> _Ast:
        token = self._peek()
        if token is None:
            return ("missing",)
        if token.kind == "lparen":
            self.position += 1
            expression = self._parse_expression(0)
            if not self._match("rparen"):
                return ("raw", tuple((item.kind, item.value) for item in self.tokens))
            return expression
        if token.kind == "literal":
            self.position += 1
            return ("literal", token.value)
        if token.kind == "number":
            self.position += 1
            return ("number", token.value)
        if token.kind in {"identifier", "quoted_identifier"}:
            if token.kind == "identifier" and token.value == "array" and self._peek(1, "lbracket"):
                self.position += 2
                values = self._parse_delimited_values("rbracket")
                return ("array", values)
            parts = [self._consume_name_component()]
            while self._match("dot"):
                parts.append(self._consume_name_component())
            name = tuple(parts)
            if self._match("lparen"):
                arguments = self._parse_delimited_values("rparen")
                return ("call", name, arguments)
            return self._canonical_name(name)
        self.position += 1
        return ("unknown", token.kind, token.value)

    def _parse_parenthesized_list(self) -> tuple[_Ast, ...]:
        if not self._match("lparen"):
            return (("missing-list",),)
        return self._parse_delimited_values("rparen")

    def _parse_delimited_values(self, closing_kind: str) -> tuple[_Ast, ...]:
        values: list[_Ast] = []
        if self._match(closing_kind):
            return ()
        while self.position < len(self.tokens):
            if self._match("identifier", "variadic"):
                values.append(("variadic", self._parse_expression(0)))
            else:
                values.append(self._parse_expression(0))
            if self._match(closing_kind):
                return tuple(values)
            if not self._match("comma"):
                return (*values, ("missing-delimiter",))
        return (*values, ("missing-close", closing_kind))

    def _parse_cast_type(self) -> tuple[str, ...]:
        parts: list[str] = []
        depth = 0
        while (token := self._peek()) is not None:
            if token.kind == "lbracket" and self._peek(1, "rbracket"):
                parts.extend(("[", "]"))
                self.position += 2
                continue
            if depth == 0 and (
                token.kind in {"comma", "rparen", "rbracket"}
                or (token.kind == "operator" and token.value != "*")
                or (token.kind == "identifier" and token.value in {"and", "or", "is", "in", "like", "glob"})
            ):
                break
            if token.kind == "lparen":
                depth += 1
            elif token.kind == "rparen":
                depth -= 1
            parts.append(token.value.lower() if token.kind == "identifier" else token.value)
            self.position += 1
        return tuple(parts)

    def _peek_binary_operator(self) -> tuple[str, int, int] | None:
        token = self._peek()
        if token is None:
            return None
        if token.kind == "identifier":
            if token.value == "or":
                return "or", 10, 1
            if token.value == "and":
                return "and", 20, 1
            if token.value == "not" and self._peek(1, "identifier", "in"):
                return "not in", 30, 2
            if token.value == "not" and self._peek(1, "identifier", "like"):
                return "not like", 30, 2
            if token.value in {"in", "like", "glob"}:
                return token.value, 30, 1
            if token.value == "is":
                if self._peek(1, "identifier", "not"):
                    return "is not", 30, 2
                return "is", 30, 1
        if token.kind != "operator":
            return None
        precedence = {
            "=": 30,
            "<>": 30,
            "!=": 30,
            "<": 30,
            ">": 30,
            "<=": 30,
            ">=": 30,
            "~~": 30,
            "!~~": 30,
            "~": 30,
            "!~": 30,
            "~*": 30,
            "!~*": 30,
            "||": 40,
            "+": 50,
            "-": 50,
            "*": 60,
            "/": 60,
            "%": 60,
        }.get(token.value)
        return None if precedence is None else (token.value, precedence, 1)

    def _canonical_name(self, parts: tuple[tuple[str, str], ...]) -> _Ast:
        final_kind, final_value = parts[-1]
        if final_kind == "identifier" and final_value in self.context.table.c:
            if len(parts) == 1:
                return ("column", final_value)
            preceding_kind, preceding_value = parts[-2]
            if len(parts) == 2 and preceding_kind == "identifier" and preceding_value == self.context.table.name.lower():
                return ("column", final_value)
            if len(parts) == 3 and preceding_kind == "identifier" and preceding_value == self.context.table.name.lower():
                schema_kind, schema_value = parts[0]
                expected_schema = self.context.table.schema or self.context.dialect.default_schema_name
                if schema_kind == "identifier" and expected_schema is not None and schema_value == expected_schema.lower():
                    return ("column", final_value)
        return ("name", parts)

    def _consume_name_component(self) -> tuple[str, str]:
        token = self._peek()
        if token is None or token.kind not in {"identifier", "quoted_identifier"}:
            return "missing", ""
        self.position += 1
        return token.kind, token.value

    @overload
    def _peek(self, offset: int = 0) -> _SqlToken | None: ...

    @overload
    def _peek(self, offset: int, kind: str, value: str | None = None) -> bool: ...

    def _peek(self, offset: int = 0, kind: str | None = None, value: str | None = None) -> _SqlToken | bool | None:
        position = self.position + offset
        if position >= len(self.tokens):
            return False if kind is not None else None
        token = self.tokens[position]
        if kind is None:
            return token
        return token.kind == kind and (value is None or token.value == value)

    def _match(self, kind: str, value: str | None = None) -> bool:
        if not self._peek(0, kind, value):
            return False
        self.position += 1
        return True


def _parse_sql_expression(tokens: tuple[_SqlToken, ...], context: _ExpressionContext) -> _Ast:
    paired = _split_reflected_pair(tokens)
    if paired is not None:
        left_tokens, right_tokens = paired
        return (
            "paired",
            _SqlExpressionParser(left_tokens, context).parse(),
            _SqlExpressionParser(right_tokens, context).parse(),
        )
    return _SqlExpressionParser(tokens, context).parse()


def _split_reflected_pair(tokens: tuple[_SqlToken, ...]) -> tuple[tuple[_SqlToken, ...], tuple[_SqlToken, ...]] | None:
    separator = None
    for index in range(len(tokens) - 2):
        if tokens[index].kind == "rparen" and tokens[index + 1] == _SqlToken("operator", "=") and tokens[index + 2].kind == "lparen":
            separator = index
    if separator is None:
        return None
    left = tokens[:separator]
    right = tokens[separator + 3 :]
    left_balance = sum(token.kind == "lparen" for token in left) - sum(token.kind == "rparen" for token in left)
    if left_balance == 1 and left and left[0].kind == "lparen":
        left = left[1:]
    if right and right[-1].kind == "rparen":
        right = right[:-1]
    return left, right


def _canonicalize_ast(ast: _Ast, context: _ExpressionContext) -> _Ast:
    kind = ast[0]
    if kind in {"literal", "number", "column", "name", "unknown", "missing", "raw"}:
        return ast
    if kind == "unary":
        return ("unary", ast[1], _canonicalize_ast(cast("_Ast", ast[2]), context))
    if kind == "cast":
        expression = _canonicalize_ast(cast("_Ast", ast[1]), context)
        cast_type = cast("tuple[str, ...]", ast[2])
        return ("cast", expression, cast_type)
    if kind == "array":
        return ("array", tuple(_canonicalize_ast(item, context) for item in cast("tuple[_Ast, ...]", ast[1])))
    if kind == "variadic":
        return ("variadic", _canonicalize_ast(cast("_Ast", ast[1]), context))
    if kind == "call":
        return (
            "call",
            ast[1],
            tuple(_canonicalize_ast(item, context) for item in cast("tuple[_Ast, ...]", ast[2])),
        )
    if kind == "in":
        return (
            "in",
            ast[1],
            _canonicalize_ast(cast("_Ast", ast[2]), context),
            tuple(_canonicalize_ast(item, context) for item in cast("tuple[_Ast, ...]", ast[3])),
        )
    if kind == "is":
        return (
            "is",
            ast[1],
            _canonicalize_ast(cast("_Ast", ast[2]), context),
            _canonicalize_ast(cast("_Ast", ast[3]), context),
        )
    if kind == "paired":
        return (
            "paired",
            _canonicalize_ast(cast("_Ast", ast[1]), context),
            _canonicalize_ast(cast("_Ast", ast[2]), context),
        )
    if kind == "binary":
        raw_operator = cast("str", ast[1])
        operator = {"!=": "<>", "~~": "like", "!~~": "not like"}.get(raw_operator, raw_operator)
        left = _canonicalize_ast(cast("_Ast", ast[2]), context)
        right = _canonicalize_ast(cast("_Ast", ast[3]), context)
        if operator in {"and", "or"}:
            return ("binary", operator, *_flatten_associative(operator, left, right))
        return ("binary", operator, left, right)
    return ast


def _flatten_associative(operator: str, left: _Ast, right: _Ast) -> tuple[_Ast, ...]:
    flattened: list[_Ast] = []
    for expression in (left, right):
        if expression[:2] == ("binary", operator):
            flattened.extend(cast("tuple[_Ast, ...]", expression[2:]))
        else:
            flattened.append(expression)
    return tuple(flattened)


_COMPARISON_OPERATORS = frozenset({"=", "<>", "<", ">", "<=", ">=", "like", "not like", "~", "!~", "~*", "!~*", "glob"})
_TEXT_CAST_TYPES = frozenset({"text", "charactervarying", "varchar"})
_TEXT_ARRAY_CAST_TYPES = frozenset({"text[]", "charactervarying[]", "varchar[]"})
_LENGTH_BUILTIN_NAMES = frozenset(
    {
        (("identifier", "length"),),
        (("identifier", "pg_catalog"), ("identifier", "length")),
    }
)
_BTRIM_BUILTIN_NAMES = frozenset(
    {
        (("identifier", "btrim"),),
        (("identifier", "pg_catalog"), ("identifier", "btrim")),
    }
)
_CHR_BUILTIN_NAME = (("identifier", "chr"),)
_QUALIFIED_CHR_BUILTIN_NAME = (("identifier", "pg_catalog"), ("identifier", "chr"))


def _default_asts_equivalent(
    expected: _Ast | None,
    actual: _Ast | None,
    column: Column[Any],
    dialect: Dialect,
) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if expected == actual:
        return True
    if dialect.name != "postgresql" or actual[0] != "cast" or expected != actual[1]:
        return False
    cast_type = "".join(cast("tuple[str, ...]", actual[2]))
    if type(column.type) is CHAR:
        return cast_type == "bpchar" and expected[0] == "literal"
    return _is_plain_postgresql_text_column(column, dialect) and cast_type in _TEXT_CAST_TYPES and expected[0] == "literal"


def _expression_counters_equivalent(
    expected: Counter[_Ast],
    actual: Counter[_Ast],
    context: _ExpressionContext,
) -> bool:
    expected_items = [expression for expression, count in expected.items() for _ in range(count)]
    actual_items = [expression for expression, count in actual.items() for _ in range(count)]
    if len(expected_items) != len(actual_items):
        return False

    matched_expected_by_actual: list[int | None] = [None] * len(actual_items)

    def assign(expected_index: int, visited_actual: set[int]) -> bool:
        for actual_index, actual_expression in enumerate(actual_items):
            if actual_index in visited_actual or not _expression_asts_equivalent(
                expected_items[expected_index],
                actual_expression,
                context,
            ):
                continue
            visited_actual.add(actual_index)
            previous_expected = matched_expected_by_actual[actual_index]
            if previous_expected is None or assign(previous_expected, visited_actual):
                matched_expected_by_actual[actual_index] = expected_index
                return True
        return False

    return all(assign(expected_index, set()) for expected_index in range(len(expected_items)))


def _expression_asts_equivalent(expected: _Ast, actual: _Ast, context: _ExpressionContext) -> bool:
    if expected == actual:
        return True
    if expected[0] == "in":
        reflected_in = _actual_any_as_in(expected, actual, context)
        if reflected_in is not None:
            actual = reflected_in
    if expected[0] != actual[0]:
        return False

    kind = expected[0]
    if kind in {"literal", "number", "column", "name", "unknown", "missing", "raw"}:
        return False
    if kind == "unary":
        return expected[1] == actual[1] and _expression_asts_equivalent(cast("_Ast", expected[2]), cast("_Ast", actual[2]), context)
    if kind == "cast":
        return expected[2] == actual[2] and _expression_asts_equivalent(cast("_Ast", expected[1]), cast("_Ast", actual[1]), context)
    if kind == "array":
        return _expression_sequences_equivalent(cast("tuple[_Ast, ...]", expected[1]), cast("tuple[_Ast, ...]", actual[1]), context)
    if kind == "call":
        if expected[1] != actual[1]:
            return False
        expected_arguments = cast("tuple[_Ast, ...]", expected[2])
        actual_arguments = cast("tuple[_Ast, ...]", actual[2])
        call_key = _text_builtin_call_key(expected[1], expected_arguments, context)
        if call_key is not None and call_key in context.text_builtin_proof.variadic_calls:
            expanded = _expanded_variadic_array_arguments(actual_arguments)
            if expanded is not None:
                actual_arguments = expanded
        if call_key is not None and call_key in context.text_builtin_proof.safe_calls:
            return _text_builtin_arguments_equivalent(expected_arguments, actual_arguments, context)
        if call_key is not None and call_key in context.text_builtin_proof.literal_only_calls:
            return _text_builtin_literal_arguments_equivalent(expected_arguments, actual_arguments, context)
        if call_key is not None and call_key in context.text_builtin_proof.int4_literal_calls:
            return _text_builtin_int4_literal_arguments_equivalent(expected_arguments, actual_arguments, context)
        return _expression_sequences_equivalent(expected_arguments, actual_arguments, context)
    if kind == "in":
        expected_values = cast("tuple[_Ast, ...]", expected[3])
        actual_values = cast("tuple[_Ast, ...]", actual[3])
        if expected[1] != actual[1] or len(expected_values) != len(actual_values):
            return False
        expected_left = cast("_Ast", expected[2])
        actual_left = cast("_Ast", actual[2])
        if not _comparison_operand_equivalent(expected_left, actual_left, None, None, context):
            return False
        return all(
            _comparison_operand_equivalent(expected_value, actual_value, expected_left, actual_left, context)
            for expected_value, actual_value in zip(expected_values, actual_values, strict=True)
        )
    if kind == "is":
        return expected[1] == actual[1] and _expression_sequences_equivalent(
            cast("tuple[_Ast, ...]", expected[2:]),
            cast("tuple[_Ast, ...]", actual[2:]),
            context,
        )
    if kind == "paired":
        return _expression_sequences_equivalent(
            cast("tuple[_Ast, ...]", expected[1:]),
            cast("tuple[_Ast, ...]", actual[1:]),
            context,
        )
    if kind == "binary":
        if expected[1] != actual[1] or len(expected) != len(actual):
            return False
        operator = cast("str", expected[1])
        expected_operands = cast("tuple[_Ast, ...]", expected[2:])
        actual_operands = cast("tuple[_Ast, ...]", actual[2:])
        if operator in _COMPARISON_OPERATORS and len(expected_operands) == 2:
            return _comparison_operand_equivalent(
                expected_operands[0],
                actual_operands[0],
                expected_operands[1],
                actual_operands[1],
                context,
            ) and _comparison_operand_equivalent(
                expected_operands[1],
                actual_operands[1],
                expected_operands[0],
                actual_operands[0],
                context,
            )
        return _expression_sequences_equivalent(expected_operands, actual_operands, context)
    return False


def _expression_sequences_equivalent(
    expected: tuple[_Ast, ...],
    actual: tuple[_Ast, ...],
    context: _ExpressionContext,
) -> bool:
    return len(expected) == len(actual) and all(
        _expression_asts_equivalent(expected_item, actual_item, context)
        for expected_item, actual_item in zip(expected, actual, strict=True)
    )


def _expanded_variadic_array_arguments(arguments: tuple[_Ast, ...]) -> tuple[_Ast, ...] | None:
    if not arguments or arguments[-1][0] != "variadic":
        return None
    array = cast("_Ast", arguments[-1][1])
    if array[0] != "array":
        return None
    values = cast("tuple[_Ast, ...]", array[1])
    return (*arguments[:-1], *values)


def _text_builtin_arguments_equivalent(
    expected: tuple[_Ast, ...],
    actual: tuple[_Ast, ...],
    context: _ExpressionContext,
) -> bool:
    return len(expected) == len(actual) and all(
        _expression_asts_equivalent(expected_argument, actual_argument, context)
        or _actual_plain_text_column_cast(expected_argument, actual_argument, context)
        or _actual_untyped_literal_text_cast(expected_argument, actual_argument, context)
        for expected_argument, actual_argument in zip(expected, actual, strict=True)
    )


def _text_builtin_literal_arguments_equivalent(
    expected: tuple[_Ast, ...],
    actual: tuple[_Ast, ...],
    context: _ExpressionContext,
) -> bool:
    if len(expected) != len(actual) or not expected:
        return False
    if not _expression_asts_equivalent(expected[0], actual[0], context):
        return False
    return all(
        _expression_asts_equivalent(expected_argument, actual_argument, context)
        or _actual_untyped_literal_text_cast(expected_argument, actual_argument, context)
        for expected_argument, actual_argument in zip(expected[1:], actual[1:], strict=True)
    )


def _text_builtin_int4_literal_arguments_equivalent(
    expected: tuple[_Ast, ...],
    actual: tuple[_Ast, ...],
    context: _ExpressionContext,
) -> bool:
    if len(expected) != 2 or len(actual) != 2:
        return False
    if not _expression_asts_equivalent(expected[0], actual[0], context):
        return False
    return _expression_asts_equivalent(expected[1], actual[1], context) or _unknown_literal_int4_equivalent(
        expected[1],
        actual[1],
    )


def _unknown_literal_int4_equivalent(expected: _Ast, actual: _Ast) -> bool:
    if expected[0] != "literal":
        return False
    match = re.fullmatch(r"'([+-]?\d+)'", cast("str", expected[1]))
    if match is None:
        return False
    expected_value = int(match.group(1))
    if not -(2**31) <= expected_value <= 2**31 - 1:
        return False
    sign = 1
    number = actual
    if actual[0] == "unary" and actual[1] in {"+", "-"}:
        sign = -1 if actual[1] == "-" else 1
        number = cast("_Ast", actual[2])
    if number[0] != "number" or re.fullmatch(r"\d+", cast("str", number[1])) is None:
        return False
    return expected_value == sign * int(cast("str", number[1]))


def _text_cast_builtin_kind(name: object, arity: int) -> str | None:
    if name in _LENGTH_BUILTIN_NAMES and arity == 1:
        return "length"
    if name in _BTRIM_BUILTIN_NAMES and arity in {1, 2}:
        return "btrim"
    return None


def _text_builtin_call_key(
    name: object,
    arguments: tuple[_Ast, ...],
    context: _ExpressionContext,
) -> _TextBuiltinCallKey | None:
    kind = _text_cast_builtin_kind(name, len(arguments))
    if kind is None or not arguments or arguments[0][0] != "column":
        return None
    if kind == "btrim" and len(arguments) == 2:
        if arguments[1][0] == "literal":
            kind = "btrim_unknown"
        elif not _known_postgresql_text_expression(arguments[1], context):
            return None
    column_name = cast("str", arguments[0][1])
    if column_name not in context.text_columns:
        return None
    family = "varchar" if column_name in context.varchar_columns else "text"
    return kind, len(arguments), family


def _known_postgresql_text_expression(expression: _Ast, context: _ExpressionContext) -> bool:
    return _postgresql_text_expression_family(expression, context) is not None


def _postgresql_text_expression_family(expression: _Ast, context: _ExpressionContext) -> str | None:
    kind = expression[0]
    if kind == "column":
        column_name = cast("str", expression[1])
        if column_name not in context.text_columns:
            return None
        return "varchar" if column_name in context.varchar_columns else "text"
    if kind == "cast":
        cast_type = "".join(cast("tuple[str, ...]", expression[2]))
        if cast_type in {"text", "pg_catalog.text"}:
            return "text"
        if cast_type in _TEXT_CAST_TYPES or cast_type in {"pg_catalog.varchar", "pg_catalog.charactervarying"}:
            return "varchar"
        return None
    if kind == "call":
        arguments = cast("tuple[_Ast, ...]", expression[2])
        if len(arguments) == 1 and _proven_postgresql_int4_expression(arguments[0]):
            if expression[1] == _QUALIFIED_CHR_BUILTIN_NAME and ("chr", 1) in context.text_builtin_proof.pg_catalog_calls:
                return "text"
            if expression[1] == _CHR_BUILTIN_NAME and ("chr", 1, "int4") in context.text_builtin_proof.text_result_calls:
                return "text"
        return None
    if kind == "binary" and expression[1] == "||":
        operands = cast("tuple[_Ast, ...]", expression[2:])
        if len(operands) != 2:
            return None
        families = tuple(_postgresql_text_expression_family(operand, context) for operand in operands)
        contextual_families = tuple(
            family if family is not None else "unknown" if operand[0] == "literal" else None
            for operand, family in zip(operands, families, strict=True)
        )
        if None in contextual_families or all(family == "unknown" for family in contextual_families):
            return None
        shape = cast("_TextConcatShape", contextual_families)
        if shape == ("text", "text"):
            return "text" if shape in context.text_builtin_proof.text_result_concat_shapes else None
        if shape in context.text_builtin_proof.safe_concat_shapes or shape in context.text_builtin_proof.text_result_concat_shapes:
            return "text"
        return None
    return None


def _proven_postgresql_int4_expression(expression: _Ast) -> bool:
    if expression[0] == "cast":
        cast_type = "".join(cast("tuple[str, ...]", expression[2]))
        return cast_type in {"integer", "int4", "pg_catalog.integer", "pg_catalog.int4"}
    sign = 1
    number = expression
    if expression[0] == "unary" and expression[1] in {"+", "-"}:
        sign = -1 if expression[1] == "-" else 1
        number = cast("_Ast", expression[2])
    if number[0] != "number" or re.fullmatch(r"\d+", cast("str", number[1])) is None:
        return False
    value = sign * int(cast("str", number[1]))
    return -(2**31) <= value <= 2**31 - 1


def _actual_plain_text_column_cast(expected: _Ast, actual: _Ast, context: _ExpressionContext) -> bool:
    if context.dialect.name != "postgresql" or expected[0] != "column" or actual[0] != "cast" or expected != actual[1]:
        return False
    cast_type = "".join(cast("tuple[str, ...]", actual[2]))
    return expected[1] in context.text_columns and cast_type in _TEXT_CAST_TYPES


def _actual_untyped_literal_text_cast(expected: _Ast, actual: _Ast, context: _ExpressionContext) -> bool:
    if context.dialect.name != "postgresql" or expected[0] != "literal" or actual[0] != "cast" or expected != actual[1]:
        return False
    cast_type = "".join(cast("tuple[str, ...]", actual[2]))
    return cast_type in _TEXT_CAST_TYPES


def _comparison_operand_equivalent(
    expected: _Ast,
    actual: _Ast,
    peer_expected: _Ast | None,
    peer_actual: _Ast | None,
    context: _ExpressionContext,
) -> bool:
    if _expression_asts_equivalent(expected, actual, context):
        return True
    if _actual_plain_text_column_cast(expected, actual, context):
        return True
    if context.dialect.name != "postgresql" or actual[0] != "cast" or expected != actual[1]:
        return False
    cast_type = "".join(cast("tuple[str, ...]", actual[2]))
    if expected[0] != "literal" or peer_expected is None:
        return False
    allowed_cast_types = _literal_reflection_cast_types(peer_expected, peer_actual, context)
    return cast_type in allowed_cast_types


def _literal_reflection_cast_types(
    peer_expected: _Ast,
    peer_actual: _Ast | None,
    context: _ExpressionContext,
) -> frozenset[str]:
    if peer_expected[0] == "column":
        peer_name = cast("str", peer_expected[1])
        if peer_name in context.text_columns:
            return _TEXT_CAST_TYPES
        if peer_name in context.char_columns:
            return frozenset({"bpchar"})
        return frozenset()
    if peer_expected[0] == "call":
        arguments = cast("tuple[_Ast, ...]", peer_expected[2])
        exact_reflected_call = peer_actual == peer_expected
        call_key = _text_builtin_call_key(peer_expected[1], arguments, context)
        if (
            call_key is not None
            and call_key[0] in {"btrim", "btrim_unknown"}
            and (
                call_key in context.text_builtin_proof.safe_calls
                or call_key in context.text_builtin_proof.literal_only_calls
                or call_key in context.text_builtin_proof.text_result_calls
                or exact_reflected_call
            )
        ):
            return _TEXT_CAST_TYPES
        return frozenset()
    if peer_expected[0] != "cast" or "".join(cast("tuple[str, ...]", peer_expected[2])) not in _TEXT_CAST_TYPES:
        return frozenset()
    inner = cast("_Ast", peer_expected[1])
    if inner[0] == "column" and inner[1] in context.char_columns:
        return _TEXT_CAST_TYPES
    return frozenset()


def _actual_any_as_in(expected: _Ast, actual: _Ast, context: _ExpressionContext) -> _Ast | None:
    negated = bool(expected[1])
    operator = "<>" if negated else "="
    function_name = "all" if negated else "any"
    if actual[:2] != ("binary", operator):
        return None
    right = cast("_Ast", actual[3])
    if right[0] != "call" or right[1] != (("identifier", function_name),):
        return None
    arguments = cast("tuple[_Ast, ...]", right[2])
    if len(arguments) != 1:
        return None
    array = arguments[0]
    if array[0] == "cast":
        expected_left = cast("_Ast", expected[2])
        cast_type = "".join(cast("tuple[str, ...]", array[2]))
        if expected_left[0] != "column" or expected_left[1] not in context.text_columns or cast_type not in _TEXT_ARRAY_CAST_TYPES:
            return None
        array = cast("_Ast", array[1])
    if array[0] != "array":
        return None
    return ("in", negated, actual[2], array[1])


def _undouble_percent_literals(ast: _Ast) -> _Ast:
    def visit(value: object) -> object:
        if not isinstance(value, tuple):
            return value
        if value and value[0] == "literal":
            return ("literal", cast("str", value[1]).replace("%%", "%"))
        return tuple(visit(item) for item in value)

    return cast("_Ast", visit(ast))


def _normalize_fk_action(value: str | None) -> str | None:
    if value is None or value.upper() == "NO ACTION":
        return None
    return re.sub(r"\s+", " ", value.strip()).upper()


def _normalize_initially(value: str | None) -> str:
    return "IMMEDIATE" if value is None else value.strip().upper()


def _normalize_match(value: str | None) -> str:
    return "SIMPLE" if value is None else value.strip().upper()


def _normalize_referred_schema(value: str | None, dialect: Dialect) -> str | None:
    if dialect.name == "postgresql" and value is not None and value == dialect.default_schema_name:
        return None
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _sorted_counter_items(counter: Counter[_ForeignKeyShape]) -> list[tuple[object, int]]:
    return sorted(counter.items(), key=lambda item: repr(item[0]))


def _sorted_unique_constraint_signatures(
    counter: Counter[_UniqueConstraintShape],
) -> list[tuple[tuple[str, ...], bool, int]]:
    return sorted((tuple(sorted(columns)), nulls_not_distinct, count) for (columns, nulls_not_distinct), count in counter.items())


def _sorted_check_counter(counter: Counter[tuple[str | None, _Ast]]) -> list[tuple[str, str, int]]:
    return sorted(("<unnamed>" if name is None else name, repr(sql), count) for (name, sql), count in counter.items())
