"""Schema-level tests: the user_preferences table exists with the expected columns."""

from elspeth.web.sessions.models import metadata


def test_user_preferences_table_registered() -> None:
    """The user_preferences table is registered on the shared metadata."""
    assert "user_preferences" in metadata.tables


def test_user_preferences_table_columns() -> None:
    """The user_preferences table has the expected columns."""
    table = metadata.tables["user_preferences"]
    column_names = {c.name for c in table.columns}
    assert column_names == {
        "user_id",
        "default_composer_mode",
        "banner_dismissed_at",
        "freeform_intro_dismissed_at",
        "tutorial_completed_at",
        "tutorial_stage",
        "tutorial_session_id",
        "tutorial_run_id",
        "tutorial_source_data_hash",
        "updated_at",
    }


def test_freeform_intro_dismissed_at_is_nullable_timestamp() -> None:
    table = metadata.tables["user_preferences"]
    assert table.c.freeform_intro_dismissed_at.nullable


def test_user_preferences_user_id_is_primary_key() -> None:
    """user_id is the primary key (one row per user)."""
    table = metadata.tables["user_preferences"]
    pk_columns = {c.name for c in table.primary_key.columns}
    assert pk_columns == {"user_id"}


def test_default_composer_mode_has_server_default_guided() -> None:
    """The stored default for new rows is 'guided' even at the DB level."""
    table = metadata.tables["user_preferences"]
    column = table.c.default_composer_mode
    # Server default's `.arg` is "guided" when set via `server_default="guided"`.
    assert column.server_default is not None
    assert "guided" in str(column.server_default.arg)


def test_default_composer_mode_check_constraint_closes_the_enum() -> None:
    """A DB-level CHECK rejects writes outside {'guided', 'freeform'}.

    Defense in depth: the Pydantic boundary rejects bad input at the API
    edge, and the service's Tier-1 read guard crashes on stored garbage —
    this check also blocks direct-SQL writes from persisting an invalid
    mode in the first place. Mirrors the codebase pattern across every
    other closed-enum column in sessions/models.py (trust_mode,
    density_default, blob status, run status, audit_access_log
    writer_principal).
    """
    from sqlalchemy import CheckConstraint

    table = metadata.tables["user_preferences"]
    check_names = {c.name for c in table.constraints if isinstance(c, CheckConstraint)}
    assert "ck_user_preferences_default_composer_mode" in check_names


def test_tutorial_completed_at_is_nullable_timestamp() -> None:
    """NULL means first-run tutorial is not complete; timestamp means done."""
    table = metadata.tables["user_preferences"]
    column = table.c.tutorial_completed_at
    assert column.nullable


def test_tutorial_stage_check_constraint_closes_the_enum() -> None:
    """The tutorial resume stage is a closed enum at the DB level too
    (elspeth-918f4434b3). NULL = no in-progress tutorial; 'welcome' is
    deliberately NOT stored (nothing has started)."""
    from sqlalchemy import CheckConstraint

    table = metadata.tables["user_preferences"]
    check_names = {c.name for c in table.constraints if isinstance(c, CheckConstraint)}
    assert "ck_user_preferences_tutorial_stage" in check_names


def test_tutorial_resume_columns_are_nullable() -> None:
    """All four resume columns are NULL when no tutorial is in progress."""
    table = metadata.tables["user_preferences"]
    for name in ("tutorial_stage", "tutorial_session_id", "tutorial_run_id", "tutorial_source_data_hash"):
        assert table.c[name].nullable, name
