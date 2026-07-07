"""Public API tests for the landscape package facade."""

from __future__ import annotations


def test_landscape_root_does_not_export_raw_schema_tables() -> None:
    """Raw SQLAlchemy schema objects belong to the schema submodule."""
    import elspeth.core.landscape as landscape
    from elspeth.core.landscape import schema

    raw_schema_names = ("metadata", "runs_table", "nodes_table", "tokens_table", "token_outcomes_table")

    assert schema.metadata is not None
    assert schema.runs_table is not None
    assert schema.nodes_table is not None
    assert schema.tokens_table is not None
    assert schema.token_outcomes_table is not None

    missing = object()
    leaked_names = [name for name in raw_schema_names if getattr(landscape, name, missing) is not missing]
    exported_names = [name for name in raw_schema_names if name in landscape.__all__]
    assert leaked_names == []
    assert exported_names == []


def test_landscape_root_keeps_stable_audit_entry_points() -> None:
    """The facade still exposes documented audit entry points."""
    import elspeth.core.landscape as landscape

    for name in ("LandscapeDB", "RecorderFactory", "LandscapeExporter", "QueryRepository", "explain"):
        assert hasattr(landscape, name)
        assert name in landscape.__all__
