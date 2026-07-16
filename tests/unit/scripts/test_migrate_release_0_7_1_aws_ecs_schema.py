"""Contract tests for the release-0.7.1 AWS ECS schema migrator."""

from __future__ import annotations

import io
import json
import re
from types import MappingProxyType

import pytest
from scripts import migrate_release_0_7_1_aws_ecs_schema as migration

from elspeth.core.schema_identity import SCHEMA_IDENTITY_TABLE_NAME
from elspeth.web.sessions.models import POSTGRESQL_AUDIT_DDL_COHORT


@pytest.mark.parametrize(
    ("session_state", "landscape_state", "expected"),
    [
        (
            migration.SessionState.RELEASE_0_7_0,
            migration.LandscapeState.RELEASE_0_7_0_WIDTH_16,
            migration.MigrationPlan.APPLY_BOTH,
        ),
        (
            migration.SessionState.RELEASE_0_7_0,
            migration.LandscapeState.RELEASE_0_7_0_WIDTH_32,
            migration.MigrationPlan.APPLY_BOTH,
        ),
        (
            migration.SessionState.CURRENT,
            migration.LandscapeState.RELEASE_0_7_0_WIDTH_16,
            migration.MigrationPlan.RESUME_LANDSCAPE,
        ),
        (
            migration.SessionState.CURRENT,
            migration.LandscapeState.RELEASE_0_7_0_WIDTH_32,
            migration.MigrationPlan.RESUME_LANDSCAPE,
        ),
        (
            migration.SessionState.CURRENT,
            migration.LandscapeState.CURRENT,
            migration.MigrationPlan.ALREADY_APPLIED,
        ),
    ],
)
def test_closed_plan_accepts_only_supported_states(
    session_state: migration.SessionState,
    landscape_state: migration.LandscapeState,
    expected: migration.MigrationPlan,
) -> None:
    assert migration.select_migration_plan(session_state, landscape_state) is expected


@pytest.mark.parametrize(
    ("session_state", "landscape_state"),
    [
        (migration.SessionState.RELEASE_0_7_0, migration.LandscapeState.CURRENT),
        (migration.SessionState.INVALID, migration.LandscapeState.RELEASE_0_7_0_WIDTH_16),
        (migration.SessionState.RELEASE_0_7_0, migration.LandscapeState.INVALID),
        (migration.SessionState.INVALID, migration.LandscapeState.INVALID),
    ],
)
def test_closed_plan_rejects_reverse_partial_and_unrecognized_states(
    session_state: migration.SessionState,
    landscape_state: migration.LandscapeState,
) -> None:
    with pytest.raises(migration.MigrationFailure) as exc_info:
        migration.select_migration_plan(session_state, landscape_state)

    assert exc_info.value.code is migration.ResultCode.PRECONDITION_FAILED


def test_summary_has_one_closed_redacted_shape() -> None:
    summary = migration.MigrationSummary(
        status=migration.ResultStatus.FAILED,
        code=migration.ResultCode.DATABASE_ERROR,
        already_applied=False,
        session_state=migration.SessionState.NOT_CHECKED,
        landscape_state=migration.LandscapeState.NOT_CHECKED,
    )

    assert summary.to_dict() == {
        "already_applied": False,
        "code": "DATABASE_ERROR",
        "landscape_state": "not_checked",
        "session_state": "not_checked",
        "status": "failed",
    }
    assert type(summary.to_dict()) is dict


def test_cli_requires_exact_apply_flag_before_reading_environment() -> None:
    output = io.StringIO()

    exit_code = migration.main(
        [],
        environ=MappingProxyType({}),
        stdout=output,
    )

    assert exit_code == 2
    assert json.loads(output.getvalue()) == {
        "already_applied": False,
        "code": "APPLY_REQUIRED",
        "landscape_state": "not_checked",
        "session_state": "not_checked",
        "status": "failed",
    }


def test_cli_accepts_owner_urls_only_from_named_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str]] = []

    def fake_run(session_url: str, landscape_url: str) -> migration.MigrationSummary:
        captured.append((session_url, landscape_url))
        return migration.MigrationSummary(
            status=migration.ResultStatus.SUCCEEDED,
            code=migration.ResultCode.ALREADY_APPLIED,
            already_applied=True,
            session_state=migration.SessionState.CURRENT,
            landscape_state=migration.LandscapeState.CURRENT,
        )

    monkeypatch.setattr(migration, "run_migration", fake_run)
    output = io.StringIO()
    session_secret = "postgresql+psycopg://owner:session-secret@db/session"
    landscape_secret = "postgresql+psycopg://owner:landscape-secret@db/landscape"

    exit_code = migration.main(
        ["--apply"],
        environ=MappingProxyType(
            {
                migration.SESSION_OWNER_URL_ENV: session_secret,
                migration.LANDSCAPE_OWNER_URL_ENV: landscape_secret,
            }
        ),
        stdout=output,
    )

    assert exit_code == 0
    assert captured == [(session_secret, landscape_secret)]
    assert session_secret not in output.getvalue()
    assert landscape_secret not in output.getvalue()
    assert "session-secret" not in output.getvalue()
    assert "landscape-secret" not in output.getvalue()


def test_cli_suppresses_exception_text_that_may_contain_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_url = "postgresql+psycopg://owner:do-not-print@db/session"

    def fail(_session_url: str, _landscape_url: str) -> migration.MigrationSummary:
        raise RuntimeError(f"connection failed for {secret_url}")

    monkeypatch.setattr(migration, "run_migration", fail)
    output = io.StringIO()

    exit_code = migration.main(
        ["--apply"],
        environ=MappingProxyType(
            {
                migration.SESSION_OWNER_URL_ENV: secret_url,
                migration.LANDSCAPE_OWNER_URL_ENV: secret_url.replace("session", "landscape"),
            }
        ),
        stdout=output,
    )

    assert exit_code == 1
    assert json.loads(output.getvalue())["code"] == "INTERNAL_ERROR"
    assert secret_url not in output.getvalue()
    assert "do-not-print" not in output.getvalue()


def test_target_table_sets_pin_only_the_versioned_identity_delta() -> None:
    session_pre = migration.release_0_7_0_session_table_names()
    landscape_pre = migration.release_0_7_0_landscape_table_names()

    assert SCHEMA_IDENTITY_TABLE_NAME not in session_pre
    assert SCHEMA_IDENTITY_TABLE_NAME not in landscape_pre
    assert "sessions" in session_pre
    assert "run_sources" in landscape_pre


def test_shared_postgresql_audit_ddl_is_immutable_complete_and_non_destructive() -> None:
    assert type(POSTGRESQL_AUDIT_DDL_COHORT) is tuple
    assert len(POSTGRESQL_AUDIT_DDL_COHORT) == 6
    assert len({entry.trigger_name for entry in POSTGRESQL_AUDIT_DDL_COHORT}) == 6
    assert len({entry.function_name for entry in POSTGRESQL_AUDIT_DDL_COHORT}) == 6

    migration_statements = migration.release_0_7_1_ddl_statements(hash_width=16)
    all_statements = (
        tuple(statement for entry in POSTGRESQL_AUDIT_DDL_COHORT for statement in (entry.function_sql, entry.trigger_sql))
        + migration_statements
    )
    normalized = "\n".join(all_statements).upper()
    for forbidden in (
        r"\bDROP\s+",
        r"\bTRUNCATE\s+",
        r"\bDELETE\s+FROM\b",
        r"\bCREATE\s+OR\s+REPLACE\b",
        r"\bVARCHAR\(16\)",
    ):
        assert re.search(forbidden, normalized) is None
    assert "VARCHAR(32)" in normalized


def test_dockerfile_copies_only_versioned_migrator_to_fixed_runtime_path() -> None:
    dockerfile = migration.REPOSITORY_ROOT.joinpath("Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY --chown=elspeth:elspeth scripts/migrate_release_0_7_1_aws_ecs_schema.py /app/ops/migrate_release_0_7_1_aws_ecs_schema.py"
    ) in dockerfile
