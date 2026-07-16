"""Epoch-26 sink-effect metadata and portable mechanical invariants."""

from __future__ import annotations

from sqlalchemy import inspect

from elspeth.core.landscape.database import _REQUIRED_CHECK_CONSTRAINTS, _REQUIRED_INDEXES, LandscapeDB
from elspeth.core.landscape.schema import metadata


def test_effect_tables_and_linkage_columns_exist() -> None:
    expected = {
        "sink_effect_streams",
        "sink_effects",
        "sink_effect_members",
        "sink_effect_attempts",
        "audit_export_snapshots",
        "audit_export_snapshot_chunks",
        "sink_effect_export_snapshots",
    }
    assert expected <= set(metadata.tables)
    assert "sink_effect_id" in metadata.tables["operations"].c
    assert metadata.tables["operations"].c.sink_effect_id.nullable
    assert "sink_effect_id" in metadata.tables["artifacts"].c
    assert metadata.tables["artifacts"].c.produced_by_state_id.nullable
    assert metadata.tables["artifacts"].c.publication_performed.nullable is False
    assert metadata.tables["artifacts"].c.publication_evidence_kind.nullable is False


def test_effect_input_xor_and_lifecycle_checks_are_required() -> None:
    required = set(_REQUIRED_CHECK_CONSTRAINTS)
    for name in (
        "ck_sink_effects_input_kind_xor",
        "ck_sink_effects_lifecycle",
        "ck_sink_effects_stream_shape",
        "ck_sink_effects_descriptor_mode",
        "ck_sink_effect_members_input_kind",
        "ck_sink_effect_export_snapshots_input_kind",
    ):
        assert (
            "sink_effects" if name.startswith("ck_sink_effects_") else name.removeprefix("ck_").rsplit("_input_kind", 1)[0],
            name,
        ) in required


def test_fresh_sqlite_reflects_effect_fks_and_unique_operation_link() -> None:
    db = LandscapeDB.in_memory()
    try:
        inspector = inspect(db.engine)
        operation_fks = inspector.get_foreign_keys("operations")
        assert any(fk["constrained_columns"] == ["sink_effect_id"] and fk["referred_table"] == "sink_effects" for fk in operation_fks)
        operation_indexes = inspector.get_indexes("operations")
        effect_index = next(item for item in operation_indexes if item["name"] == "uq_operations_sink_effect_id")
        assert effect_index["unique"] == 1
        assert effect_index["column_names"] == ["sink_effect_id"]
    finally:
        db.close()


def test_effect_indexes_are_startup_required() -> None:
    required = set(_REQUIRED_INDEXES)
    assert ("operations", "uq_operations_sink_effect_id") in required
    assert ("sink_effect_streams", "uq_sink_effect_stream_identity") in required
    assert ("sink_effect_members", "uq_sink_effect_member_binding") in required
